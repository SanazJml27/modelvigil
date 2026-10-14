"""The four agents.

  Monitor        deterministic, every week: label-free + lagged-label signals
  Diagnostician  agentic (LLM or scripted policy): hypothesis-driven root-cause
                 investigation via tools, verified by a deterministic checker
  Remediation    deterministic: candidate fixes + shadow evaluation
  Governance     PCCP envelope, human approval gate, vigilance flag, audit

Design rule: the LLM chooses *which question to ask next* and explains the
answer; it never computes a statistic or changes the model directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import detectors as D
from . import diagnostics as G
from . import remediation as R
from .audit import AuditLog
from .config import (LABEL_LAG_WEEKS, LABEL_WINDOW_WEEKS, PCCP_PLAN, SUBGROUP_ATTRS, THRESHOLDS)
from .llm import Backend, MockBackend
from .model import ModelVersion

LABELLED_PREFIXES = ("OE", "AUROC", "SUBGROUP")

ROOT_CAUSES = ["unit_change", "feed_outage", "new_population_unverified", "benign_covariate_shift",
               "harmful_covariate_shift", "subgroup_degradation", "concept_shift",
               "prevalence_shift", "insufficient_evidence"]


# ============================================================== context
@dataclass
class Context:
    stream: object
    dev: pd.DataFrame
    ref: pd.DataFrame
    model: ModelVersion
    pipeline: R.DataPipeline
    audit: AuditLog
    baseline: dict
    raw: dict = field(default_factory=dict)
    week: int = 0
    label_floor: int = 1
    seed: int = 0
    watchlist: set = field(default_factory=set)
    _cache: dict = field(default_factory=dict, repr=False)

    def batch(self, weeks) -> pd.DataFrame:
        weeks = [weeks] if isinstance(weeks, int) else list(weeks)
        df = pd.concat([self.raw[w] for w in weeks], ignore_index=True)
        return self.pipeline.apply(df).reset_index(drop=True)

    def scored(self, weeks, model: ModelVersion | None = None):
        m = model or self.model
        key = (tuple([weeks] if isinstance(weeks, int) else weeks), id(m), id(self.pipeline))
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            df = self.batch(weeks)
            self._cache[key] = (df, m.predict_proba(df), m, self.pipeline)  # keep refs so ids stay unique
        df, p = self._cache[key][:2]
        return df, p

    def labelled_weeks(self, since: int | None = None) -> list[int]:
        last = self.week - LABEL_LAG_WEEKS
        first = max(self.label_floor, last - LABEL_WINDOW_WEEKS + 1, since or 1, 1)
        return [w for w in range(first, last + 1)]

    def labelled_since(self, since: int) -> list[int]:
        last = self.week - LABEL_LAG_WEEKS
        return [w for w in range(max(since, self.label_floor, 1), last + 1)]


# ============================================================== monitor
class MonitorAgent:
    """Label-free signals fire immediately (they are high-specificity data
    checks). Label-based signals must persist for two consecutive weekly
    assessments (statistical process control run rule): weekly O/E is noisy
    and 16 weekly looks would otherwise inflate false alarms."""
    name = "monitor"

    def __init__(self):
        self._prev_raw: set[str] = set()

    def assess(self, ctx: Context) -> dict:
        rec = self._assess(ctx)
        raw = set(rec["signals"])
        rec["raw_signals"] = sorted(raw)
        rec["signals"] = {k: v for k, v in rec["signals"].items()
                          if not k.startswith(LABELLED_PREFIXES) or k in self._prev_raw}
        self._prev_raw = raw
        return rec

    def _assess(self, ctx: Context) -> dict:
        T = THRESHOLDS
        cur, p = ctx.scored(ctx.week)
        sig: dict[str, dict] = {}

        dq = D.data_quality(ctx.ref, cur)
        for f, r in dq["out_of_range"].items():
            if r > T.out_of_range_rate:
                sig[f"DQ_RANGE:{f}"] = {"rate": r}
        for k, v in dq["mode_spike"].items():
            if v["excess_fraction"] > T.mode_spike:
                sig[f"DQ_SPIKE:{k}"] = v

        dc = D.domain_classifier(ctx.ref, cur, seed=ctx.week)
        if dc["auc"] > T.domain_auc:
            sig["DRIFT_MULTIVARIATE"] = dc

        ss = D.score_summary(p)
        rel = (ss["alert_rate"] - ctx.baseline["alert_rate"]) / ctx.baseline["alert_rate"]
        if abs(rel) > T.alert_rate_rel_change:
            sig["ALERT_RATE"] = {"alert_rate": ss["alert_rate"], "baseline": ctx.baseline["alert_rate"],
                                 "relative_change": round(rel, 3)}

        pooled = [w for w in (ctx.week - 1, ctx.week) if w >= 1]
        _, pp = ctx.scored(pooled)
        cbpe = D.cbpe_auroc(pp)
        if ctx.baseline["cbpe_auroc"] - cbpe > T.cbpe_auc_drop:
            sig["CBPE_DROP"] = {"cbpe_auroc": round(cbpe, 4), "baseline": ctx.baseline["cbpe_auroc"]}

        for u in G.new_population(ctx.ref, cur)["findings"]:
            sig[f"NEW_SUBGROUP:{u['subgroup']}"] = u

        lab_metrics = None
        lw = ctx.labelled_weeks()
        if lw:
            ldf, lp = ctx.scored(lw)
            lab_metrics = D.realised_metrics(ldf["y"].to_numpy(), lp)
            if len(lw) >= 2 and not lab_metrics.get("insufficient") and lab_metrics["events"] >= 20:
                lo, hi = lab_metrics["oe_ci"]
                if (lab_metrics["oe"] < T.oe_low or lab_metrics["oe"] > T.oe_high) and (lo > 1 or hi < 1):
                    sig["OE"] = {k: lab_metrics[k] for k in ("oe", "oe_ci", "events", "n")}
                drop = ctx.baseline["auroc"] - lab_metrics["auroc"]
                if drop > T.auc_drop and drop > T.z_overall * lab_metrics["auroc_se"]:
                    sig["AUROC"] = {"auroc": lab_metrics["auroc"], "baseline": ctx.baseline["auroc"],
                                    "se": lab_metrics["auroc_se"]}
                for fl in G.subgroup_localisation(ldf, lp, ctx.ref)["flagged"]:
                    sig[f"SUBGROUP:{fl['subgroup']}"] = fl
            lab_metrics = lab_metrics | {"weeks": lw}

        return {"week": ctx.week, "signals": sig, "score": ss, "cbpe_auroc": round(cbpe, 4),
                "domain_auc": dc["auc"], "labelled": lab_metrics}


# ============================================================== tools
TOOL_SCHEMAS = [
    {"name": "get_incident_signals",
     "description": "Return the monitoring signals that triggered this incident and the recent weekly trend "
                    "(alert rate, expected AUROC, multivariate drift AUROC, lagged realised metrics).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "check_unit_mismatch",
     "description": "Test whether a lab/device unit changed: per-site median ratio vs reference compared "
                    "with known conversion factors (e.g. lactate mg/dL vs mmol/L).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "check_imputation_artifact",
     "description": "Test for a dropped data feed silently back-filled with a default value: excess "
                    "probability mass at a single value, localised by site.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "check_new_population",
     "description": "Check whether a subgroup (site, sex, age band) absent from development data has appeared.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "feature_drift_ranking",
     "description": "Rank features by population stability index (current week vs reference).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "assess_harm",
     "description": "Is observed input drift harmful? Confidence-based expected AUROC (label-free) plus "
                    "realised metrics on any labelled data since incident onset.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "localise_subgroups",
     "description": "On lagged labelled data: which subgroups are miscalibrated, and does excluding the worst "
                    "one bring the rest of the population back into control (local vs global problem)?",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "test_residual_effects",
     "description": "Model-revision likelihood-ratio test on labelled data: does any feature add information "
                    "beyond the model's own logit? Significant residual effects indicate concept shift.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "check_calibration_pattern",
     "description": "On labelled data: is O/E out of control while AUROC and calibration slope are preserved "
                    "(the signature of an outcome-prevalence shift)?",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "submit_diagnosis",
     "description": "Submit the final root-cause diagnosis. Must be supported by tool evidence; it will be "
                    "independently verified.",
     "input_schema": {"type": "object", "properties": {
         "root_cause": {"type": "string", "enum": ROOT_CAUSES},
         "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
         "scope": {"type": "object", "description": "affected feature/site/subgroup, if any"},
         "evidence": {"type": "array", "items": {"type": "string"}},
         "rationale": {"type": "string"}},
         "required": ["root_cause", "confidence", "evidence", "rationale"]}},
]


class Toolbox:
    def __init__(self, ctx: Context, incident: "Incident", history: list[dict]):
        self.ctx, self.incident, self.history = ctx, incident, history
        self.schemas = TOOL_SCHEMAS

    # -- helpers
    def post_change_weeks(self) -> list[int]:
        """Labelled weeks after the most likely change-point in weekly O/E
        (single change-point Poisson likelihood ratio). Diagnosing on a window
        that mixes pre- and post-change weeks dilutes every test."""
        if self.incident.label_weeks is not None:
            return self.incident.label_weeks
        ctx = self.ctx
        weeks = ctx.labelled_since(max(1, ctx.week - LABEL_LAG_WEEKS - 7))
        default = ctx.labelled_weeks()
        chosen, info = default, {"method": "default window"}
        if len(weeks) >= 3:
            O, E = [], []
            for w in weeks:
                d, p = ctx.scored(w)
                O.append(d["y"].sum()); E.append(p.sum())
            O, E = np.array(O, float), np.array(E, float)

            def ll(o, e):
                r = o.sum() / e.sum()
                return float(np.sum(o * np.log(r * e) - r * e))
            l0 = ll(O, E)
            best = max(range(1, len(weeks)), key=lambda k: ll(O[:k], E[:k]) + ll(O[k:], E[k:]))
            lr = 2 * (ll(O[:best], E[:best]) + ll(O[best:], E[best:]) - l0)
            info = {"method": "change-point", "candidate_onset_week": weeks[best], "lr": round(lr, 2)}
            if lr > 3.84:
                chosen = weeks[best:]
        self.incident.label_weeks = chosen
        self.incident.label_onset_info = info | {"weeks_used": chosen}
        return chosen

    def _labelled(self, since_onset: bool = False):
        ctx = self.ctx
        weeks = ctx.labelled_since(self.incident.onset_week) if since_onset else self.post_change_weeks()
        if not weeks:
            return None, None
        return ctx.scored(weeks)

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}

    # -- tools
    def t_get_incident_signals(self):
        trend = [{"week": h["week"], "alert_rate": h["score"]["alert_rate"], "cbpe_auroc": h["cbpe_auroc"],
                  "domain_auc": h["domain_auc"],
                  "labelled_oe": (h["labelled"] or {}).get("oe"),
                  "labelled_auroc": (h["labelled"] or {}).get("auroc")} for h in self.history[-6:]]
        return {"week": self.ctx.week, "onset_week": self.incident.onset_week,
                "signals": self.incident.signals, "trend": trend, "baseline": self.ctx.baseline,
                "label_lag_weeks": LABEL_LAG_WEEKS,
                "labelled_weeks_for_diagnosis": self.post_change_weeks() if any(
                    k.startswith(LABELLED_PREFIXES) for k in self.incident.signals) else None,
                "change_point": self.incident.label_onset_info}

    def t_check_unit_mismatch(self):
        return G.unit_mismatch(self.ctx.ref, self.ctx.batch(self.ctx.week))

    def t_check_imputation_artifact(self):
        return G.imputation_artifact(self.ctx.ref, self.ctx.batch(self.ctx.week))

    def t_check_new_population(self):
        return G.new_population(self.ctx.ref, self.ctx.batch(self.ctx.week))

    def t_feature_drift_ranking(self):
        fd = D.feature_drift(self.ctx.ref, self.ctx.batch(self.ctx.week))
        ranked = sorted(fd.items(), key=lambda kv: -kv[1]["psi"])
        return {"ranking": [{"feature": k} | v for k, v in ranked[:5]]}

    def t_assess_harm(self):
        ctx = self.ctx
        _, p = ctx.scored([w for w in (ctx.week - 1, ctx.week) if w >= self.incident.onset_week])
        ldf, lp = self._labelled(since_onset=True)
        lab = D.realised_metrics(ldf["y"].to_numpy(), lp) if ldf is not None else None
        return G.harm_assessment(None, p, lab, ctx.baseline)

    def t_localise_subgroups(self):
        ldf, lp = self._labelled()
        if ldf is None:
            return {"hypothesis": "subgroup_degradation", "supported": False, "reason": "no labelled data yet"}
        res = G.subgroup_localisation(ldf, lp, self.ctx.ref)
        # local vs global: does removing the subgroup restore the rest?
        res["localised"] = None
        for fl in res["flagged"]:
            attr, val = fl["subgroup"].split("=", 1)
            rest = (ldf[attr].astype(str) != val).to_numpy()
            r = D.realised_metrics(ldf["y"].to_numpy()[rest], lp[rest])
            lo, hi = r["oe_ci"]
            # same rule the monitor uses to raise the overall O/E signal
            in_ctrl = (THRESHOLDS.oe_low <= r["oe"] <= THRESHOLDS.oe_high) or (lo <= 1 <= hi)
            # heterogeneity: is the subgroup's O/E different from the rest's? (var log O/E ~ 1/O)
            het_z = abs(np.log(fl["oe"] / r["oe"])) / np.sqrt(1 / fl["events"] + 1 / max(r["events"], 1))
            fl["rest_oe"], fl["rest_in_control"], fl["heterogeneity_z"] = r["oe"], bool(in_ctrl), round(float(het_z), 2)
            if in_ctrl and het_z > THRESHOLDS.z_subgroup and res["localised"] is None:
                res["localised"] = fl["subgroup"]
        res["supported"] = res["localised"] is not None
        return res

    def t_test_residual_effects(self):
        ldf, lp = self._labelled()
        if ldf is None:
            return {"hypothesis": "concept_shift", "supported": False, "reason": "no labelled data yet"}
        return G.residual_effects(ldf, lp, self.ctx.ref, self.ctx.model.predict_proba(self.ctx.ref))

    def t_check_calibration_pattern(self):
        ldf, lp = self._labelled()
        if ldf is None:
            return {"hypothesis": "prevalence_shift", "supported": False, "reason": "no labelled data yet"}
        return G.calibration_pattern(D.realised_metrics(ldf["y"].to_numpy(), lp), self.ctx.baseline)

    def t_submit_diagnosis(self, root_cause, confidence, evidence, rationale, scope=None):
        if root_cause not in ROOT_CAUSES:
            return {"error": f"root_cause must be one of {ROOT_CAUSES}"}
        return {"accepted_for_verification": True}


# ============================================================== diagnostician
SYSTEM_PROMPT = """You are the Diagnostician agent in ModelVigil, a post-market surveillance system for a
deployed hospital model that predicts 48-hour clinical deterioration. A monitoring incident has
been opened. Your job is to find the ROOT CAUSE using the tools - never guess.

Method (hypothesis-driven, cheapest and most actionable first):
1. Review the signals.
2. Rule out data-pipeline faults (unit change, feed outage / imputation artefact) - these are common,
   abrupt, and fixed without touching the model.
3. If the signals are label-free only: check for a new population, then assess whether the drift
   is harmful (expected performance) or benign.
4. If lagged-label signals exist: first decide local vs global (localise_subgroups); then test for
   concept shift (residual effects); then for prevalence shift (calibration pattern).
5. Submit with submit_diagnosis. Cite the numbers you saw. If evidence is inconclusive, say
   insufficient_evidence - an honest 'not yet' is safer than a wrong cause.
Your diagnosis is independently re-verified; unsupported claims are rejected and logged."""


def _fmt(x):
    return json.dumps(x, default=str)


def scripted_policy(incident: "Incident"):
    """The deterministic reference policy (also the offline 'mock LLM')."""
    labelled = any(k.startswith(LABELLED_PREFIXES) for k in incident.signals)

    def submit(rc, conf, evidence, rationale, scope=None):
        return (f"Concluding: {rc}.", "submit_diagnosis",
                {"root_cause": rc, "confidence": conf, "scope": scope or {}, "evidence": evidence,
                 "rationale": rationale})

    def policy(hist):
        done = {name: res for name, _, res in hist}
        if "get_incident_signals" not in done:
            return ("Review the triggering signals and recent trend.", "get_incident_signals", {})
        if "check_unit_mismatch" not in done:
            return ("Rule out pipeline faults first: a unit change is abrupt, common and fixable "
                    "without touching the model.", "check_unit_mismatch", {})
        um = done["check_unit_mismatch"]
        if um["supported"]:
            f = um["findings"][0]
            return submit("unit_change", "high",
                          [f"{f['feature']} at site {f['site']}: median ratio {f['observed_ratio']} vs reference "
                           f"matches {f['suspected_unit']} conversion factor {f['conversion_factor']:.3g}"],
                          "Values are a constant multiple of the expected distribution at one site only, "
                          "consistent with an LIS unit change, not a clinical change.",
                          {"feature": f["feature"], "site": f["site"]})
        if "check_imputation_artifact" not in done:
            return ("Next pipeline hypothesis: a dropped feed back-filled with a default value.",
                    "check_imputation_artifact", {})
        ia = done["check_imputation_artifact"]
        if ia["supported"]:
            f = ia["findings"][0]
            return submit("feed_outage", "high",
                          [f"{f['feature']} at site {f['site']}: {f['excess_fraction']:.0%} excess records at "
                           f"exactly {f['value']:g} (~{f['est_affected_records']} records this week)"],
                          "A single-value spike localised to one site is the signature of an interface "
                          "engine back-filling a dropped device feed. True low values are being masked.",
                          {"feature": f["feature"], "site": f["site"], "value": f["value"]})
        if labelled:
            if "localise_subgroups" not in done:
                return ("Labelled signals present. Is the degradation local or global?", "localise_subgroups", {})
            ls = done["localise_subgroups"]
            if ls.get("supported"):
                fl = next(f for f in ls["flagged"] if f["subgroup"] == ls["localised"])
                return submit("subgroup_degradation", "high",
                              [f"{fl['subgroup']}: O/E {fl['oe']} (z={fl['z']}), sensitivity {fl['sensitivity']}, "
                               f"{fl['missed_events']} missed events",
                               f"Remaining population O/E {fl['rest_oe']} - in control"],
                              "Miscalibration is confined to one subgroup; the rest of the population is "
                              "in control, so a global fix would be wrong.",
                              {"subgroup": ls["localised"]})
            if "test_residual_effects" not in done:
                return ("Global problem. Test whether predictor-outcome relationships changed.",
                        "test_residual_effects", {})
            rr = done["test_residual_effects"]
            if rr.get("supported"):
                s = rr["significant"][0]
                return submit("concept_shift", "medium" if rr["events"] < 150 else "high",
                              [f"{s['feature']} carries residual information beyond the model logit "
                               f"changed vs reference (LR={s['lr_stat']}, p={s['p']:.1e}, "
                               f"change in residual coef {s['change_in_residual_coef_per_sd']}/SD)"],
                              "The relationship between a predictor and the outcome has changed; monotone "
                              "recalibration cannot restore discrimination.",
                              {"features": [x["feature"] for x in rr["significant"]]})
            if "check_calibration_pattern" not in done:
                return ("No residual effects. Check for a pure prevalence shift.", "check_calibration_pattern", {})
            cp = done["check_calibration_pattern"]
            if cp.get("supported"):
                return submit("prevalence_shift", "high",
                              [f"O/E {cp['oe']} (95% CI {cp['oe_ci']}) with AUROC {cp['auroc']} "
                               f"(baseline {cp['auroc_baseline']}) and slope {cp['slope']}",
                               "No feature carries residual information"],
                              "Event rate changed while ranking of patients is preserved: label/prevalence "
                              "shift. Intercept recalibration is the indicated fix.")
            return submit("insufficient_evidence", "low", ["labelled signals present but no hypothesis supported"],
                          "Keep the incident open and re-investigate as more labels arrive.")
        # label-free only
        if "check_new_population" not in done:
            return ("Label-free drift only. Has a population the model never saw appeared?",
                    "check_new_population", {})
        npop = done["check_new_population"]
        if npop["supported"]:
            f = npop["findings"][0]
            return submit("new_population_unverified", "medium",
                          [f"{f['subgroup']} appeared ({f['share']:.0%} of patients); most shifted features: "
                           f"{', '.join(k for k, _ in f['most_shifted_features'])}"],
                          "Model performance in this population is unknown until outcomes arrive. Enhanced "
                          "surveillance with prioritised outcome review.", {"subgroup": f["subgroup"]})
        if "assess_harm" not in done:
            return ("Is the drift harmful? Check expected performance.", "assess_harm", {})
        ha = done["assess_harm"]
        if ha["supported"]:
            return submit("benign_covariate_shift", "medium",
                          [f"Expected AUROC {ha['cbpe_auroc']} (drop {ha['cbpe_drop']}); {ha['note']}"],
                          "Case-mix changed but the model is expected to perform as before. No model change; "
                          "confirm when labels arrive.")
        return submit("harmful_covariate_shift", "medium",
                      [f"Expected AUROC drop {ha['cbpe_drop']}"],
                      "Input drift is predicted to degrade performance. Escalate for review.")

    return policy


def verify_diagnosis(dx: dict, tb: Toolbox) -> tuple[bool, str]:
    """Independent deterministic check that the claimed cause is supported."""
    rc = dx.get("root_cause")
    checks = {
        "unit_change": lambda: tb.t_check_unit_mismatch()["supported"],
        "feed_outage": lambda: tb.t_check_imputation_artifact()["supported"],
        "new_population_unverified": lambda: tb.t_check_new_population()["supported"],
        "benign_covariate_shift": lambda: tb.t_assess_harm()["supported"],
        "harmful_covariate_shift": lambda: not tb.t_assess_harm()["supported"],
        "subgroup_degradation": lambda: tb.t_localise_subgroups()["supported"],
        "concept_shift": lambda: tb.t_test_residual_effects()["supported"],
        "prevalence_shift": lambda: tb.t_check_calibration_pattern()["supported"]
                                    and not tb.t_test_residual_effects()["supported"],
        "insufficient_evidence": lambda: True,
    }
    if rc not in checks:
        return False, f"unknown root cause {rc!r}"
    ok = bool(checks[rc]())
    return ok, "verified" if ok else f"claimed {rc} not supported by deterministic re-test"


@dataclass
class Incident:
    id: str
    opened_week: int
    onset_week: int
    signals: dict
    diagnosis: dict | None = None
    plan: dict | None = None
    decision: dict | None = None
    transcript: list = field(default_factory=list)
    label_weeks: list | None = None
    label_onset_info: dict | None = None


class DiagnosticianAgent:
    name = "diagnostician"

    def __init__(self, backend: Backend, max_steps: int = 14):
        self.backend, self.max_steps = backend, max_steps

    def investigate(self, ctx: Context, incident: Incident, history: list[dict]) -> dict:
        tb = Toolbox(ctx, incident, history)
        backend = self.backend
        if isinstance(backend, MockBackend) or backend is None:
            backend = MockBackend(scripted_policy(incident))
        dx = self._loop(backend, ctx, incident, tb)
        ok, why = (False, "agent did not submit a diagnosis") if dx is None else verify_diagnosis(dx, tb)
        ctx.audit.append(ctx.week, "verifier", "diagnosis_verification",
                         {"incident": incident.id, "backend": backend.name,
                          "claimed": dx and dx.get("root_cause"), "ok": ok, "detail": why})
        if not ok:
            # fall back to the scripted reference policy and record the disagreement
            rejected = dx and dx.get("root_cause")
            dx = self._loop(MockBackend(scripted_policy(incident)), ctx, incident, tb, tag="fallback")
            ok, why2 = verify_diagnosis(dx, tb)
            dx["note"] = f"{backend.name} diagnosis {rejected!r} rejected ({why}); scripted policy used ({why2})"
            ctx.audit.append(ctx.week, "verifier", "diagnosis_verification",
                             {"incident": incident.id, "backend": "mock:fallback",
                              "claimed": dx.get("root_cause"), "ok": ok, "detail": why2})
        dx["verified"] = ok
        dx["backend"] = backend.name
        return dx

    def _loop(self, backend: Backend, ctx: Context, incident: Incident, tb: Toolbox, tag: str = "") -> dict | None:
        user = (f"Incident {incident.id} opened at week {ctx.week} (onset week {incident.onset_week}). "
                f"Triggering signals: {list(incident.signals)}. Investigate and submit a diagnosis.")
        messages = [{"role": "user", "content": user}]
        for _ in range(self.max_steps):
            out = backend.step(SYSTEM_PROMPT, messages, tb.schemas)
            if not out["tool_calls"]:
                messages.append({"role": "assistant", "content": out["text"] or "(thinking)"})
                messages.append({"role": "user", "content": "Use a tool, or call submit_diagnosis."})
                continue
            content = ([{"type": "text", "text": out["text"]}] if out["text"] else []) + [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]} for c in out["tool_calls"]]
            messages.append({"role": "assistant", "content": content})
            results, final = [], None
            for c in out["tool_calls"]:
                res = tb.call(c["name"], c["input"])
                incident.transcript.append({"thought": out["text"], "tool": c["name"], "input": c["input"],
                                            "result": res, "backend": backend.name + (f":{tag}" if tag else "")})
                ctx.audit.append(ctx.week, self.name, "tool_call",
                                 {"incident": incident.id, "backend": backend.name, "thought": out["text"],
                                  "tool": c["name"], "input": c["input"], "result": res})
                if c["name"] == "submit_diagnosis" and "error" not in res:
                    final = dict(c["input"])
                results.append({"type": "tool_result", "tool_use_id": c["id"], "content": _fmt(res)})
            messages.append({"role": "user", "content": results})
            if final:
                return final
        return None


# ============================================================== remediation
ACTION_FOR = {
    "unit_change": "data_pipeline_fix", "feed_outage": "data_pipeline_fix",
    "prevalence_shift": "recalibrate", "concept_shift": "retrain",
    "subgroup_degradation": "subgroup_restriction", "new_population_unverified": "enhanced_surveillance",
    "benign_covariate_shift": "no_change", "harmful_covariate_shift": "enhanced_surveillance",
    "insufficient_evidence": "no_change",
}


def _bump(v: str, major=False) -> str:
    a, b, _ = (int(x) for x in v.split("."))
    return f"{a + 1}.0.0" if major else f"{a}.{b + 1}.0"


class RemediationAgent:
    name = "remediation"

    def plan(self, ctx: Context, incident: Incident) -> dict:
        dx = incident.diagnosis
        rc = dx["root_cause"]
        scope = dx.get("scope") or {}
        category = ACTION_FOR[rc]
        plan = {"category": category, "root_cause": rc, "recommendations": [], "candidate": None,
                "interim": None, "shadow": None, "verification": None}

        if rc == "unit_change":
            f, site = scope["feature"], scope["site"]
            onset = self._onset(ctx, lambda df: any(x["feature"] == f and x["site"] == site
                                                    for x in G.unit_mismatch(ctx.ref, df)["findings"]))
            factor = next(x["conversion_factor"] for x in G.unit_mismatch(ctx.ref, ctx.batch(ctx.week))["findings"]
                          if x["feature"] == f and x["site"] == site)
            corr = R.Correction("unit_conversion", f, site, onset, factor=factor)
            plan |= {"action": "data_pipeline_fix", "correction": corr,
                     "verification": R.verify_pipeline_fix(ctx.model, ctx.raw[ctx.week], ctx.pipeline,
                                                           ctx.pipeline.with_correction(corr), ctx.baseline)}
            fixed = ctx.pipeline.with_correction(corr).apply(ctx.raw[ctx.week])
            plan["verification"]["fault_cleared"] = not any(
                x["feature"] == f and x["site"] == site for x in G.unit_mismatch(ctx.ref, fixed)["findings"])
            plan["recommendations"] += [
                f"Apply verified conversion: {corr.describe()}",
                f"Open LIS change ticket for site {site}; confirm with laboratory the reporting-unit change date",
                f"Re-score all site-{site} patients since week {onset}; notify ward leads of spurious alerts"]
        elif rc == "feed_outage":
            f, site, val = scope["feature"], scope["site"], scope["value"]
            onset = self._onset(ctx, lambda df: any(x["feature"] == f and x["site"] == site
                                                    for x in G.imputation_artifact(ctx.ref, df)["findings"]))
            corr = R.Correction("sentinel_to_missing", f, site, onset, value=val)
            plan |= {"action": "data_pipeline_fix", "correction": corr,
                     "verification": R.verify_pipeline_fix(ctx.model, ctx.raw[ctx.week], ctx.pipeline,
                                                           ctx.pipeline.with_correction(corr), ctx.baseline)}
            fixed = ctx.pipeline.with_correction(corr).apply(ctx.raw[ctx.week])
            plan["verification"]["fault_cleared"] = not any(
                x["feature"] == f and x["site"] == site for x in G.imputation_artifact(ctx.ref, fixed)["findings"])
            plan["recommendations"] += [
                f"Interim: {corr.describe()}",
                f"P1 ticket to device-integration team: restore {f} feed at site {site}",
                f"Clinical safety notice to site {site}: {f} values of {val:g} may be back-filled; verify manually"]
        elif rc in ("prevalence_shift", "concept_shift", "subgroup_degradation"):
            lab = ctx.batch(incident.label_weeks or ctx.labelled_weeks())
            # subgroup / post-shift data is sparse and may only exist in the latest weeks -> random split
            fit, test = R.temporal_split(lab) if rc == "prevalence_shift" else R.random_split(lab, seed=ctx.seed)
            cands = {}
            if rc == "prevalence_shift":
                cands["recalibrate_intercept"] = R.candidate_recalibrate_intercept(ctx.model, fit, _bump(ctx.model.version))
                cands["recalibrate_logistic"] = R.candidate_recalibrate_logistic(ctx.model, fit, _bump(ctx.model.version))
            elif rc == "concept_shift":
                cands["recalibrate_logistic"] = R.candidate_recalibrate_logistic(ctx.model, fit, _bump(ctx.model.version))
                feats = (scope or {}).get("features") or []
                if feats:
                    cands["model_revision"] = R.candidate_revision(ctx.model, fit, feats, _bump(ctx.model.version, major=True))
                cands["full_refit"] = R.candidate_retrain(ctx.model, ctx.dev, fit, _bump(ctx.model.version, major=True),
                                                          seed=ctx.seed)
            else:
                attr, val = scope["subgroup"].split("=", 1)
                if attr == "site":
                    cands["site_recalibration"] = R.candidate_site_recalibration(
                        ctx.model, fit, val, _bump(ctx.model.version))
            shadow = R.shadow_compare(ctx.model, cands, test) if cands else None
            plan["shadow"] = shadow
            if rc == "prevalence_shift":
                best = min(cands, key=lambda k: shadow[k]["brier"])
                plan |= {"action": best, "candidate": cands[best]}
                plan["recommendations"].append(
                    f"Deploy {cands[best].version} ({cands[best].note}); re-baseline alert burden with ward leads")
            elif rc == "concept_shift":
                upd = [k for k in ("model_revision", "full_refit") if k in cands]
                best = min(upd, key=lambda k: shadow[k]["brier"])
                shadow["retrain"] = shadow[best]          # the proposed retrain = best update strategy
                plan |= {"action": "retrain", "candidate": cands[best], "update_strategy": best,
                         "interim": {"action": "recalibrate_logistic", "candidate": cands["recalibrate_logistic"]}}
                plan["recommendations"] += [
                    "Interim (inside PCCP): logistic recalibration to restore calibration while the update is reviewed",
                    f"Proposed: {best.replace('_', ' ')} -> {cands[best].version} ({cands[best].note}) - substantial "
                    "modification, requires clinical-safety review, updated technical documentation and validation report",
                    "Investigate the clinical-practice change behind the shift (e.g. new care bundle) with clinical leads"]
            else:
                plan |= {"action": "subgroup_restriction", "candidate": cands.get("site_recalibration")}
                plan["recommendations"] += [
                    f"Suspend autonomous alerting for {scope['subgroup']}; revert to standard early-warning score there",
                    f"Interim after sign-off: {cands['site_recalibration'].note}" if cands else
                    "No subgroup-specific recalibration available for this attribute",
                    f"Commission local validation for {scope['subgroup']} before re-enabling"]
        elif rc in ("new_population_unverified", "harmful_covariate_shift"):
            plan |= {"action": "enhanced_surveillance"}
            plan["recommendations"] += [f"Prioritise outcome chart review for {scope.get('subgroup', 'affected patients')}",
                                        "Lower subgroup alarm thresholds until performance is confirmed"]
        else:
            plan |= {"action": "no_change"}
            plan["recommendations"].append("No model change. Acknowledge drift; confirm against lagged outcomes.")
        ctx.audit.append(ctx.week, self.name, "plan", {"incident": incident.id} | _plan_public(plan))
        return plan

    @staticmethod
    def _onset(ctx: Context, present) -> int:
        """Earliest week (walking back) in which the fault is present."""
        onset = ctx.week
        for w in range(ctx.week - 1, 0, -1):
            if present(ctx.pipeline.apply(ctx.raw[w])):
                onset = w
            else:
                break
        return onset


def _plan_public(plan: dict) -> dict:
    out = {k: v for k, v in plan.items() if k not in ("candidate", "correction", "interim")}
    if plan.get("candidate") is not None:
        out["candidate"] = {"version": plan["candidate"].version, "note": plan["candidate"].note,
                            "fingerprint": plan["candidate"].fingerprint()}
    if plan.get("correction") is not None:
        out["correction"] = plan["correction"].describe()
    if plan.get("interim"):
        out["interim"] = {"action": plan["interim"]["action"], "version": plan["interim"]["candidate"].version}
    return out


# ============================================================== governance
VIGILANCE_CAUSES = {"feed_outage", "prevalence_shift", "concept_shift", "subgroup_degradation", "unit_change"}


class GovernanceAgent:
    name = "governance"

    def __init__(self, approval: str = "policy"):
        self.approval = approval

    def review(self, ctx: Context, incident: Incident) -> dict:
        plan = incident.plan
        action = plan["action"]
        shadow, cand = plan.get("shadow"), plan.get("candidate")
        cand_name = action if shadow and action in shadow else None
        pccp_action = {"data_pipeline_fix": "data_pipeline_fix"}.get(action, action)
        check = R.pccp_check(pccp_action, cand, shadow, cand_name=cand_name)

        if plan["category"] == "data_pipeline_fix":
            if not plan["verification"]["fault_cleared"]:
                check = {"inside_envelope": False, "reasons": ["re-test on corrected batch still detects the fault"]}

        decision = {"action": action, "pccp": check, "approver": None, "approved": False,
                    "vigilance_flag": incident.diagnosis["root_cause"] in VIGILANCE_CAUSES}
        if self.approval == "auto":
            decision |= {"approved": True, "approver": "auto (demo mode)"}
        elif self.approval == "interactive":
            decision |= self._ask(incident, check)
        else:  # policy
            if check["inside_envelope"]:
                decision |= {"approved": True, "approver": "PCCP policy (pre-authorised change)"}
            else:
                ok = self._simulated_reviewer(plan)
                decision |= {"approved": ok, "approver": "clinical safety officer (simulated)",
                             "review_note": "shadow evaluation non-inferior; accepted" if ok else
                             "shadow evaluation does not support the change; rejected"}
                if not ok and plan.get("interim"):
                    decision["fallback_to_interim"] = True

        ctx.audit.append(ctx.week, self.name, "decision", {"incident": incident.id} | decision)
        self._apply(ctx, incident, decision)
        return decision

    @staticmethod
    def _simulated_reviewer(plan: dict) -> bool:
        if plan["action"] == "subgroup_restriction":
            return True   # restricting autonomy in an unsafe subgroup is always accepted
        sh = plan.get("shadow")
        if not sh or plan["action"] not in sh:
            return True
        cur, cand = sh["current"], sh[plan["action"]]
        return cand["auroc"] - cur["auroc"] > PCCP_PLAN.min_shadow_auc_noninferiority and cand["brier"] <= cur["brier"] + 1e-3

    @staticmethod
    def _ask(incident: Incident, check: dict) -> dict:
        print(f"\n=== HUMAN APPROVAL REQUIRED: incident {incident.id} ===")
        print(f"Diagnosis : {incident.diagnosis['root_cause']} ({incident.diagnosis['confidence']})")
        print(f"Rationale : {incident.diagnosis['rationale']}")
        print(f"Action    : {incident.plan['action']}")
        for r in incident.plan["recommendations"]:
            print(f"  - {r}")
        print(f"PCCP      : {'inside' if check['inside_envelope'] else 'OUTSIDE'} envelope - {'; '.join(check['reasons'])}")
        ans = input("Approve? [y/N] ").strip().lower()
        return {"approved": ans == "y", "approver": "human (interactive)"}

    def _apply(self, ctx: Context, incident: Incident, decision: dict):
        plan = incident.plan
        if not decision["approved"]:
            interim = plan.get("interim")
            if interim and decision.get("fallback_to_interim"):
                self._deploy(ctx, interim["candidate"], incident, "interim")
            return
        if plan["category"] == "data_pipeline_fix":
            ctx.pipeline = ctx.pipeline.with_correction(plan["correction"])
            ctx.audit.append(ctx.week, self.name, "pipeline_change",
                             {"incident": incident.id, "correction": plan["correction"].describe()})
        elif plan.get("candidate") is not None:
            self._deploy(ctx, plan["candidate"], incident, plan["action"])
        if plan["action"] == "enhanced_surveillance":
            ctx.watchlist.add((incident.diagnosis.get("scope") or {}).get("subgroup", "all"))

    @staticmethod
    def _deploy(ctx: Context, m: ModelVersion, incident: Incident, why: str):
        old = ctx.model
        ctx.model = m
        ctx.label_floor = ctx.week + 1   # prospective verification: only post-change outcomes count
        lw = [w for w in (ctx.week - 1, ctx.week) if w >= 1]
        if lw:  # re-baseline score-based label-free statistics on the population the new config now faces
            _, p = ctx.scored(lw)
            ctx.baseline = ctx.baseline | {"alert_rate": D.score_summary(p)["alert_rate"],
                                           "mean_risk": D.score_summary(p)["mean_risk"],
                                           "cbpe_auroc": round(D.cbpe_auroc(p), 4)}
        ctx.audit.append(ctx.week, "governance", "model_change",
                         {"incident": incident.id, "reason": why, "from": old.version, "to": m.version,
                          "from_fp": old.fingerprint(), "to_fp": m.fingerprint(), "note": m.note})
