"""Weekly surveillance loop: score -> monitor -> (incident) diagnose -> plan ->
govern -> apply. Also records oracle outcomes (simulation only) so the
benchmark can quantify patient-level impact versus no monitoring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import detectors as D
from .agents import (LABELLED_PREFIXES, Context, DiagnosticianAgent, GovernanceAgent, Incident,
                     MonitorAgent, RemediationAgent, _plan_public)
from .audit import AuditLog
from .config import ALERT_THRESHOLD, RunConfig
from .llm import make_backend
from .model import fit_model
from .remediation import DataPipeline
from .simulator import SCENARIOS, HospitalStream


def _log(cfg, msg):
    if not cfg.quiet:
        print(msg)


def run(cfg: RunConfig, out_dir: str | Path | None = None) -> dict:
    stream = HospitalStream(cfg.scenario, seed=cfg.seed, n_per_week=cfg.n_per_week,
                            n_weeks=cfg.n_weeks, shift_week=cfg.shift_week)
    ref_all = stream.reference(cfg.n_ref)
    cut = int(len(ref_all) * 2 / 3)
    dev, ref = ref_all.iloc[:cut].reset_index(drop=True), ref_all.iloc[cut:].reset_index(drop=True)
    model0, cal = fit_model(dev, seed=cfg.seed)
    p_ref = model0.predict_proba(ref)
    rm = D.realised_metrics(ref["y"], p_ref)
    baseline = {"auroc": rm["auroc"], "oe": rm["oe"], "cbpe_auroc": round(D.cbpe_auroc(p_ref), 4)} | \
        D.score_summary(p_ref)

    out = Path(out_dir) if out_dir else None
    audit = AuditLog(out / "audit.jsonl" if out else None)
    audit.append(0, "system", "deployment",
                 {"scenario": cfg.scenario, "seed": cfg.seed, "model": model0.version,
                  "fingerprint": model0.fingerprint(), "baseline": baseline, "alert_threshold": ALERT_THRESHOLD})
    ctx = Context(stream, dev, ref, model0, DataPipeline(), audit, baseline, seed=cfg.seed)

    backend = None if cfg.llm == "mock" else make_backend(cfg.llm)
    monitor, diag = MonitorAgent(), DiagnosticianAgent(backend)
    remed, gov = RemediationAgent(), GovernanceAgent(cfg.approval)

    acknowledged: set[str] = set()
    weekly, incidents = [], []
    _log(cfg, f"[ModelVigil] scenario={cfg.scenario} seed={cfg.seed} | baseline AUROC {baseline['auroc']:.3f}, "
              f"alert rate {baseline['alert_rate']:.1%}")

    for w in range(1, cfg.n_weeks + 1):
        ctx.week = w
        ctx.raw[w] = stream.week(w)

        # live predictions this week (configuration in force BEFORE today's actions) + oracle outcomes
        live_df, live_p = ctx.scored(w)
        cf_p = model0.predict_proba(ctx.raw[w])
        y = live_df["y"].to_numpy()
        oracle = {}
        for arm, p in (("live", live_p), ("no_monitoring", cf_p)):
            a = p >= ALERT_THRESHOLD
            oracle[arm] = D.realised_metrics(y, p) | {"alerts": int(a.sum()),
                                                      "false_alerts": int((a & (y == 0)).sum())}

        rec = monitor.assess(ctx)
        rec["oracle"] = oracle
        rec["model_version"] = ctx.model.version
        weekly.append(rec)
        audit.append(w, "monitor", "weekly_assessment",
                     {"signals": list(rec["signals"]), "alert_rate": rec["score"]["alert_rate"],
                      "cbpe_auroc": rec["cbpe_auroc"], "domain_auc": rec["domain_auc"],
                      "labelled": rec["labelled"]})

        new = set(rec["signals"]) - acknowledged
        if not new:
            continue
        inc = Incident(id=f"INC-{cfg.scenario[:4].upper()}-{w:02d}", opened_week=w, onset_week=w,
                       signals={k: rec["signals"][k] for k in sorted(new)})
        audit.append(w, "monitor", "incident_opened", {"incident": inc.id, "signals": list(inc.signals)})
        _log(cfg, f"  week {w:2d}  INCIDENT {inc.id}: {', '.join(inc.signals)}")

        inc.diagnosis = diag.investigate(ctx, inc, weekly)
        _log(cfg, f"           diagnosis: {inc.diagnosis['root_cause']} ({inc.diagnosis['confidence']}, "
                  f"verified={inc.diagnosis['verified']})")
        inc.plan = remed.plan(ctx, inc)
        inc.decision = gov.review(ctx, inc)
        _log(cfg, f"           action: {inc.plan['action']} -> "
                  f"{'APPROVED' if inc.decision['approved'] else 'NOT APPROVED'} by {inc.decision['approver']}")

        rc = inc.diagnosis["root_cause"]
        changed = inc.decision["approved"] and inc.plan["category"] in (
            "data_pipeline_fix", "recalibrate", "retrain", "subgroup_restriction")
        if rc == "insufficient_evidence":
            pass                              # re-investigate next week
        elif changed:
            # if the signals persist after a fix, the fix failed -> re-escalate
            acknowledged -= {k for k in inc.signals if not k.startswith("NEW_SUBGROUP")}
            acknowledged |= {k for k in inc.signals if k.startswith("NEW_SUBGROUP")}
        else:
            acknowledged |= set(inc.signals)
        incidents.append(inc)

    scen = SCENARIOS[cfg.scenario]
    result = {
        "config": {"scenario": cfg.scenario, "seed": cfg.seed, "llm": cfg.llm, "approval": cfg.approval,
                   "shift_week": cfg.shift_week, "n_weeks": cfg.n_weeks, "n_per_week": cfg.n_per_week},
        "scenario": {"description": scen.description, "root_cause": scen.root_cause,
                     "correct_action": scen.correct_action, "harmful": scen.harmful},
        "baseline": baseline, "calibration_check": cal,
        "weekly": weekly, "incidents": [_incident_public(i) for i in incidents],
        "final_model": {"version": ctx.model.version, "note": ctx.model.note, "fingerprint": ctx.model.fingerprint()},
        "pipeline_corrections": [c.describe() for c in ctx.pipeline.corrections],
    }
    result["impact"] = impact(result)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _incident_public(inc: Incident) -> dict:
    return {"id": inc.id, "opened_week": inc.opened_week, "signals": inc.signals,
            "labelled_weeks_used": inc.label_weeks, "change_point": inc.label_onset_info,
            "diagnosis": inc.diagnosis, "plan": _plan_public(inc.plan), "decision": inc.decision,
            "transcript": inc.transcript}


def impact(result: dict) -> dict:
    """Patient-level effect over post-shift weeks (oracle; simulation only)."""
    sw = result["config"]["shift_week"]
    post = [r for r in result["weekly"] if r["week"] >= sw]

    def tot(key, arm):
        return int(sum(r["oracle"][arm].get(key, 0) or 0 for r in post))

    res = {"post_shift_events": tot("events", "live"), "post_shift_patients": tot("n", "live"),
           **{f"{k}_{arm}": tot(k, arm) for arm in ("live", "no_monitoring")
              for k in ("missed_events", "alerts", "false_alerts")}}
    # decision-curve net benefit at the deployed threshold (Vickers & Elkin):
    # NB = TP/n - FP/n * pt/(1-pt); reported per 1,000 patients
    w = ALERT_THRESHOLD / (1 - ALERT_THRESHOLD)
    n = max(res["post_shift_patients"], 1)
    for arm in ("live", "no_monitoring"):
        tp = res["post_shift_events"] - res[f"missed_events_{arm}"]
        res[f"net_benefit_per_1000_{arm}"] = round(1000 * (tp - w * res[f"false_alerts_{arm}"]) / n, 1)
    return res
