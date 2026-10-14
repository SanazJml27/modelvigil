"""Candidate fixes and their shadow evaluation.

Nothing is deployed on faith: every candidate change is scored on held-out
recent labelled data (or, for pipeline fixes, on the live batch) against the
currently deployed configuration before it can be proposed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ALERT_THRESHOLD, PCCP_PLAN, SUBGROUP_ATTRS
from .detectors import _logit, calibration_fit, cbpe_auroc, realised_metrics, score_summary
from .model import ModelVersion, fit_model


# --------------------------------------------------------------- pipeline
@dataclass
class Correction:
    kind: str            # "unit_conversion" | "sentinel_to_missing"
    feature: str
    site: str
    from_week: int
    factor: float | None = None
    value: float | None = None

    def describe(self) -> str:
        if self.kind == "unit_conversion":
            return (f"divide {self.feature} at site {self.site} by {self.factor:g} "
                    f"(records from week {self.from_week})")
        return (f"treat {self.feature}={self.value:g} at site {self.site} as missing "
                f"(records from week {self.from_week}) until the feed is restored")


@dataclass
class DataPipeline:
    corrections: list[Correction] = field(default_factory=list)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.corrections:
            return df
        df = df.copy()
        for c in self.corrections:
            m = (df["site"] == c.site) & (df["week"] >= c.from_week)
            if c.kind == "unit_conversion":
                df.loc[m, c.feature] = df.loc[m, c.feature] / c.factor
            elif c.kind == "sentinel_to_missing":
                df.loc[m & (df[c.feature] == c.value), c.feature] = np.nan
        return df

    def with_correction(self, c: Correction) -> "DataPipeline":
        return DataPipeline(self.corrections + [c])


# ------------------------------------------------------------ candidates
def _subgroup_max_dev(df: pd.DataFrame, p: np.ndarray) -> tuple[float, str]:
    worst, name = 0.0, ""
    for a in SUBGROUP_ATTRS:
        for val, idx in df.groupby(a).groups.items():
            m = df.index.get_indexer(idx)
            if df["y"].to_numpy()[m].sum() < 10:
                continue
            oe = df["y"].to_numpy()[m].sum() / np.asarray(p)[m].sum()
            if abs(oe - 1) > worst:
                worst, name = abs(oe - 1), f"{a}={val}"
    return round(float(worst), 3), name


def evaluate(df: pd.DataFrame, p: np.ndarray) -> dict:
    r = realised_metrics(df["y"].to_numpy(), p)
    dev, name = _subgroup_max_dev(df, p)
    return {k: r.get(k) for k in ("auroc", "oe", "brier", "sensitivity", "ppv", "missed_events", "slope")} | {
        "alert_rate": score_summary(p)["alert_rate"], "worst_subgroup_oe_dev": dev, "worst_subgroup": name}


def random_split(lab: pd.DataFrame, frac: float = 0.6, seed: int = 0):
    idx = np.random.default_rng(seed).permutation(len(lab))
    cut = int(len(lab) * frac)
    return lab.iloc[idx[:cut]].reset_index(drop=True), lab.iloc[idx[cut:]].reset_index(drop=True)


def temporal_split(lab: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on earlier labelled weeks, evaluate on the most recent one."""
    last = lab["week"].max()
    fit, test = lab[lab["week"] < last], lab[lab["week"] == last]
    if len(fit) < 300:
        return random_split(lab)
    return fit.reset_index(drop=True), test.reset_index(drop=True)


def candidate_recalibrate_intercept(model: ModelVersion, fit: pd.DataFrame, version: str) -> ModelVersion:
    a = calibration_fit(fit["y"], model.predict_proba(fit))["citl_intercept"]
    ra, rb = model.recal
    return model.derive(version=version, note=f"intercept recalibration ({a:+.3f})", recal=(ra + a, rb))


def candidate_recalibrate_logistic(model: ModelVersion, fit: pd.DataFrame, version: str) -> ModelVersion:
    c = calibration_fit(fit["y"], model.predict_proba(fit))
    a, b = c["logistic_a"], c["slope"]
    ra, rb = model.recal
    return model.derive(version=version, note=f"logistic recalibration (a={a:+.3f}, b={b:.3f})",
                        recal=(a + b * ra, b * rb))


def candidate_retrain(model: ModelVersion, dev: pd.DataFrame, fit: pd.DataFrame, version: str,
                      seed: int = 0, recent_weight: float = 4.0) -> ModelVersion:
    combo = pd.concat([dev, fit], ignore_index=True)
    w = np.r_[np.ones(len(dev)), np.full(len(fit), recent_weight)]
    new, _ = fit_model(combo, seed=seed, sample_weight=w, version=version,
                       note=f"retrained on development + {len(fit)} recent labelled cases (weight {recent_weight:g})")
    new.parent = model.version
    return new


def candidate_revision(model: ModelVersion, fit: pd.DataFrame, features: list[str], version: str) -> ModelVersion:
    """Model revision (Steyerberg's updating hierarchy, step 3): keep the
    deployed model's logit, re-estimate intercept/slope and add new terms for
    the features whose outcome relationship changed. Data-efficient when only
    a few hundred recent outcomes are available."""
    from .detectors import LOG_FEATURES
    from sklearn.linear_model import LogisticRegression
    z = _logit(model.predict_proba(fit))
    cols, terms = [z], []
    for f in features:
        x = fit[f].astype(float).to_numpy()
        lg = f in LOG_FEATURES
        x = np.log(np.clip(x, 1e-3, None)) if lg else x
        mu, sd = float(np.nanmean(x)), float(np.nanstd(x) + 1e-9)
        cols.append(np.nan_to_num((x - mu) / sd))
        terms.append([f, lg, mu, sd])
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(np.column_stack(cols), fit["y"])
    a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    terms = [t + [float(c)] for t, c in zip(terms, lr.coef_[0, 1:])]
    # compose with any existing revision (z above already includes it)
    rev = {"a": a, "b": b, "terms": terms}
    if model.revision is not None:
        old = model.revision
        rev = {"a": a + b * old["a"], "b": b * old["b"],
               "terms": [t[:4] + [b * t[4]] for t in old["terms"]] + terms}
    desc = ", ".join(f"{t[0]} {t[4]:+.2f}/SD" for t in terms)
    return model.derive(version=version, revision=rev,
                        note=f"model revision: logit x {b:.2f} {a:+.2f}; new terms {desc}")


def candidate_site_recalibration(model: ModelVersion, fit: pd.DataFrame, site: str, version: str) -> ModelVersion:
    g = fit[fit["site"] == site]
    c = calibration_fit(g["y"], model.predict_proba(g))
    sr = dict(model.site_recal)
    oa, ob = sr.get(site, (0.0, 1.0))
    sr[site] = (c["logistic_a"] + c["slope"] * oa, c["slope"] * ob)   # compose with any existing site map
    return model.derive(version=version, site_recal=sr,
                        note=f"site-{site} logistic recalibration (a={c['logistic_a']:+.3f}, b={c['slope']:.3f})")


def shadow_compare(current: ModelVersion, candidates: dict[str, ModelVersion],
                   test: pd.DataFrame) -> dict:
    out = {"current": evaluate(test, current.predict_proba(test))}
    for name, m in candidates.items():
        out[name] = evaluate(test, m.predict_proba(test))
    return out


def pccp_check(action: str, model: ModelVersion | None, shadow: dict | None,
               current_name: str = "current", cand_name: str | None = None) -> dict:
    """Is the proposed change inside the pre-authorised envelope?"""
    reasons = []
    if action in PCCP_PLAN.requires_review:
        return {"inside_envelope": False, "reasons": [f"'{action}' is outside the PCCP: "
                "potential substantial modification, requires clinical-safety review"]}
    if action not in PCCP_PLAN.allowed_auto:
        return {"inside_envelope": False, "reasons": [f"unknown action '{action}'"]}
    if model is not None and action.startswith("recalibrate"):
        a, b = model.recal
        if abs(a) > PCCP_PLAN.max_abs_intercept_shift:
            reasons.append(f"cumulative intercept shift {a:+.2f} exceeds ±{PCCP_PLAN.max_abs_intercept_shift}")
        lo, hi = PCCP_PLAN.slope_bounds
        if not lo <= b <= hi:
            reasons.append(f"cumulative slope {b:.2f} outside [{lo}, {hi}]")
    if shadow and cand_name:
        cur, cand = shadow[current_name], shadow[cand_name]
        if cand["auroc"] is not None and cur["auroc"] is not None and \
                cand["auroc"] - cur["auroc"] < PCCP_PLAN.min_shadow_auc_noninferiority:
            reasons.append("shadow AUROC inferior to current model")
        if cand["worst_subgroup_oe_dev"] > PCCP_PLAN.max_subgroup_oe_deviation:
            reasons.append(f"subgroup {cand['worst_subgroup']} |O/E-1|={cand['worst_subgroup_oe_dev']} "
                           f"exceeds {PCCP_PLAN.max_subgroup_oe_deviation}")
    return {"inside_envelope": not reasons, "reasons": reasons or ["all PCCP conditions met"]}


def verify_pipeline_fix(model: ModelVersion, raw_batch: pd.DataFrame, pipe_before: DataPipeline,
                        pipe_after: DataPipeline, ref_summary: dict) -> dict:
    """Label-free verification of a data fix on the live batch."""
    pb = model.predict_proba(pipe_before.apply(raw_batch))
    pa = model.predict_proba(pipe_after.apply(raw_batch))
    return {"alert_rate_before": score_summary(pb)["alert_rate"],
            "alert_rate_after": score_summary(pa)["alert_rate"],
            "alert_rate_reference": ref_summary["alert_rate"],
            "mean_risk_before": round(float(pb.mean()), 4), "mean_risk_after": round(float(pa.mean()), 4),
            "mean_risk_reference": ref_summary["mean_risk"],
            "cbpe_before": round(cbpe_auroc(pb), 4), "cbpe_after": round(cbpe_auroc(pa), 4)}
