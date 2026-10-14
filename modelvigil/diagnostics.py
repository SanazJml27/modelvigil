"""Root-cause hypothesis tests.

Each function tests ONE falsifiable hypothesis about why a deployed model's
behaviour changed and returns structured evidence. The Diagnostician agent
decides which hypotheses to test and in what order; the statistics are fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from .config import FEATURES, FEATURE_NAMES, SUBGROUP_ATTRS, THRESHOLDS, UNIT_CONVERSIONS
from .detectors import LOG_FEATURES, _logit, cbpe_auroc, psi, realised_metrics


def unit_mismatch(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """H: a laboratory/device unit changed. Test: per-site median ratio vs
    reference matches a known conversion factor (within 20%)."""
    findings = []
    for f, convs in UNIT_CONVERSIONS.items():
        ref_med = float(ref[f].median())
        for site, g in cur.groupby("site"):
            if len(g) < 30:
                continue
            ratio = float(g[f].median()) / ref_med
            for unit, factor in convs:
                if factor is None:
                    continue
                if abs(np.log(ratio / factor)) < np.log(1.2):
                    findings.append({"feature": f, "site": site, "observed_ratio": round(ratio, 3),
                                     "suspected_unit": unit, "expected_unit": FEATURES[f]["unit"],
                                     "conversion_factor": factor})
    return {"hypothesis": "unit_change", "supported": bool(findings), "findings": findings}


def imputation_artifact(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """H: a data feed is dropping values and an interface engine back-fills a
    default. Test: excess probability mass at a single value, localised to a site."""
    findings = []
    for f in FEATURE_NAMES:
        if f in ("comorbidity", "emergency_adm", "age"):
            continue
        for site, g in cur.groupby("site"):
            gx = g[f].dropna()
            if len(gx) < 30:
                continue
            val = gx.mode().iloc[0]
            excess = float((gx == val).mean() - (ref[f] == val).mean())
            if excess > THRESHOLDS.mode_spike:
                findings.append({"feature": f, "site": site, "value": float(val),
                                 "excess_fraction": round(excess, 3),
                                 "est_affected_records": int(round(excess * len(gx)))})
    return {"hypothesis": "feed_outage", "supported": bool(findings), "findings": findings}


def harm_assessment(ref_p: np.ndarray, cur_p: np.ndarray, labelled: dict | None,
                    baseline: dict) -> dict:
    """H: the observed input drift is benign. Test: label-free expected
    performance (CBPE) is preserved AND (if available) realised performance
    and calibration are in control."""
    est = cbpe_auroc(cur_p)
    est_drop = baseline["cbpe_auroc"] - est
    label_ok = None
    if labelled and not labelled.get("insufficient"):
        label_ok = (THRESHOLDS.oe_low <= labelled["oe"] <= THRESHOLDS.oe_high
                    and baseline["auroc"] - labelled["auroc"] < THRESHOLDS.auc_drop)
    benign = est_drop < THRESHOLDS.cbpe_auc_drop and label_ok is not False
    return {"hypothesis": "benign_covariate_shift", "supported": bool(benign),
            "cbpe_auroc": round(est, 4), "cbpe_drop": round(est_drop, 4),
            "labelled_in_control": label_ok,
            "note": "CBPE is valid under covariate shift only; labelled confirmation pending"
                    if label_ok is None else "confirmed against lagged outcomes"}


def _ll(model, X, y):
    p = np.clip(model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def residual_effects(df: pd.DataFrame, p: np.ndarray, ref: pd.DataFrame | None = None,
                     p_ref: np.ndarray | None = None) -> dict:
    """H: a predictor-outcome relationship changed (concept shift).

    Model-revision likelihood-ratio test (Steyerberg) made *differential*:
    every model has some baseline misspecification, so we test whether a
    feature's residual association with the outcome (beyond the model's own
    logit) has CHANGED between the reference cohort and recent labelled data.
    Period main effect and logit-by-period interaction are in both models, so
    pure prevalence or calibration-slope changes are absorbed, not mistaken
    for concept shift."""
    if ref is None:
        ref, p_ref = df.iloc[:0], np.array([])
    both = pd.concat([ref.assign(_period=0.0), df.assign(_period=1.0)], ignore_index=True)
    y = both["y"].to_numpy()
    z = _logit(np.r_[p_ref, p])
    per = both["_period"].to_numpy()
    base_cols = [z, per, z * per] if len(ref) else [z]
    Xb = np.column_stack(base_cols)
    base = LogisticRegression(C=1e6, max_iter=2000).fit(Xb, y)
    res = []
    for f in FEATURE_NAMES:
        x = both[f].astype(float).to_numpy()
        if f in LOG_FEATURES:
            x = np.log(np.clip(x, 1e-3, None))
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
        x = np.nan_to_num(x)
        if len(ref):
            X0 = np.column_stack(base_cols + [x])
            X1 = np.column_stack(base_cols + [x, x * per])
            m0 = LogisticRegression(C=1e6, max_iter=2000).fit(X0, y)
            ll0 = _ll(m0, X0, y)
        else:
            X1 = np.column_stack(base_cols + [x])
            ll0 = _ll(base, Xb, y)
        m1 = LogisticRegression(C=1e6, max_iter=2000).fit(X1, y)
        lr_stat = max(0.0, 2 * (_ll(m1, X1, y) - ll0))
        res.append({"feature": f, "lr_stat": round(lr_stat, 2), "p": float(stats.chi2.sf(lr_stat, 1)),
                    "change_in_residual_coef_per_sd": round(float(m1.coef_[0, -1]), 3)})
    res.sort(key=lambda r: r["p"])
    sig = [r for r in res if r["p"] < 0.001 / len(FEATURE_NAMES)]
    return {"hypothesis": "concept_shift", "supported": bool(sig), "significant": sig, "top": res[:3],
            "n_recent": int(len(df)), "events": int(df["y"].sum()), "n_reference": int(len(ref))}


def calibration_pattern(labelled: dict, baseline: dict) -> dict:
    """H: outcome prevalence shifted while discrimination held (label shift).
    Test: O/E out of control, AUROC preserved, calibration slope ~1."""
    if not labelled or labelled.get("insufficient"):
        return {"hypothesis": "prevalence_shift", "supported": False, "reason": "insufficient labels"}
    lo, hi = labelled["oe_ci"]
    oe_off = (labelled["oe"] < THRESHOLDS.oe_low or labelled["oe"] > THRESHOLDS.oe_high) and (lo > 1 or hi < 1)
    auc_ok = baseline["auroc"] - labelled["auroc"] < THRESHOLDS.auc_drop
    slope_ok = 0.75 <= labelled["slope"] <= 1.3
    return {"hypothesis": "prevalence_shift", "supported": bool(oe_off and auc_ok and slope_ok),
            "oe": labelled["oe"], "oe_ci": labelled["oe_ci"], "auroc": labelled["auroc"],
            "auroc_baseline": baseline["auroc"], "slope": labelled["slope"],
            "citl_intercept": labelled["citl_intercept"]}


def subgroup_localisation(df: pd.DataFrame, p: np.ndarray, ref: pd.DataFrame) -> dict:
    """H: degradation is concentrated in one subgroup (site, sex, age band).
    Test: subgroup O/E CI excludes 1 at Bonferroni-adjusted level, or the
    subgroup is new (not present in the development data)."""
    flagged, unseen = [], []
    for a in SUBGROUP_ATTRS:
        ref_levels = set(ref[a].unique())
        for val, idx in df.groupby(a).groups.items():
            m = df.index.get_indexer(idx)
            if val not in ref_levels:
                unseen.append({"subgroup": f"{a}={val}", "n": int(len(m))})
            r = realised_metrics(df["y"].to_numpy()[m], np.asarray(p)[m])
            if r.get("insufficient") or r["events"] < THRESHOLDS.min_subgroup_events:
                continue
            exp = float(np.asarray(p)[m].sum())
            z = (r["events"] - exp) / np.sqrt(max(exp, 1e-9))
            if abs(z) > THRESHOLDS.z_subgroup and abs(r["oe"] - 1) > 0.25:
                flagged.append({"subgroup": f"{a}={val}", "oe": r["oe"], "z": round(float(z), 2),
                                "auroc": r["auroc"], "events": r["events"], "n": r["n"],
                                "sensitivity": r["sensitivity"], "missed_events": r["missed_events"]})
    flagged.sort(key=lambda d: -abs(d["z"]))
    return {"hypothesis": "subgroup_degradation", "supported": bool(flagged),
            "flagged": flagged, "unseen_subgroups": unseen}


def new_population(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """Unlabelled check: has a subgroup appeared that the model never saw?"""
    unseen = []
    for a in SUBGROUP_ATTRS:
        new = set(cur[a].unique()) - set(ref[a].unique())
        for v in new:
            g = cur[cur[a] == v]
            drift = {f: round(psi(ref[f], g[f]), 3) for f in FEATURE_NAMES}
            top = sorted(drift.items(), key=lambda kv: -kv[1])[:3]
            unseen.append({"subgroup": f"{a}={v}", "n": int(len(g)),
                           "share": round(len(g) / len(cur), 3), "most_shifted_features": top})
    return {"hypothesis": "new_population", "supported": bool(unseen), "findings": unseen}
