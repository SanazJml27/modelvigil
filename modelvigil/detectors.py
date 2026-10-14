"""Deterministic monitoring primitives.

Label-free detectors run the moment a batch is scored. Label-based metrics
run on the lagged, chart-reviewed outcomes. None of these functions call an
LLM - agents *use* them as tools, they never re-implement the statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from .config import ALERT_THRESHOLD, FEATURES, FEATURE_NAMES

LOG_FEATURES = {"lactate", "creatinine", "wbc"}


# ---------------------------------------------------------------- utilities
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)binary feature
        vals = np.unique(np.concatenate([ref, cur]))
        r = np.array([(ref == v).mean() for v in vals])
        c = np.array([(cur == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _design(df: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    cols = []
    for f in FEATURE_NAMES:
        x, r = df[f].astype(float).to_numpy(), ref[f].astype(float).to_numpy()
        if f in LOG_FEATURES:
            x, r = np.log(np.clip(x, 1e-3, None)), np.log(np.clip(r, 1e-3, None))
        mu, sd = np.nanmean(r), np.nanstd(r) + 1e-9
        cols.append(np.nan_to_num((x - mu) / sd))
    return np.column_stack(cols)


# ---------------------------------------------------------- label-free tools
def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    out = {}
    for f in FEATURE_NAMES:
        r, c = ref[f].astype(float), cur[f].astype(float)
        out[f] = {
            "psi": round(psi(r, c), 4),
            "ks_p": float(stats.ks_2samp(r.dropna(), c.dropna()).pvalue),
            "mean_ref": round(float(r.mean()), 3),
            "mean_cur": round(float(c.mean()), 3),
        }
    return out


def domain_classifier(ref: pd.DataFrame, cur: pd.DataFrame, seed: int = 0) -> dict:
    """Classifier two-sample test: can a model tell reference from current?
    AUROC ~0.5 means no detectable multivariate shift."""
    rs = ref.sample(n=min(len(ref), len(cur) * 2), random_state=seed)
    X = np.vstack([_design(rs, ref), _design(cur, ref)])
    y = np.r_[np.zeros(len(rs)), np.ones(len(cur))]
    clf = LogisticRegression(max_iter=500, C=1.0)
    p = cross_val_predict(clf, X, y, cv=4, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, p))
    coefs = clf.fit(X, y).coef_[0]
    order = np.argsort(-np.abs(coefs))
    return {"auc": round(auc, 4),
            "top_features": [FEATURE_NAMES[i] for i in order[:3]]}


def data_quality(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """Out-of-physiological-range values, missingness and single-value spikes,
    overall and by site."""
    res = {"out_of_range": {}, "missing": {}, "mode_spike": {}}
    for f, meta in FEATURES.items():
        lo, hi = meta["range"]
        x = cur[f].astype(float)
        res["out_of_range"][f] = round(float(((x < lo) | (x > hi)).mean()), 4)
        res["missing"][f] = round(float(x.isna().mean() - ref[f].isna().mean()), 4)
        if f in ("comorbidity", "emergency_adm"):
            continue
        for site, g in cur.groupby("site"):
            gx = g[f].dropna()
            if len(gx) < 30:
                continue
            mode_val = gx.mode().iloc[0]
            frac_cur = float((gx == mode_val).mean())
            frac_ref = float((ref[f] == mode_val).mean())
            spike = frac_cur - frac_ref
            if spike > 0.05:
                res["mode_spike"][f"{f}@{site}"] = {"value": float(mode_val),
                                                    "excess_fraction": round(spike, 3)}
    return res


def score_summary(p: np.ndarray, threshold: float = ALERT_THRESHOLD) -> dict:
    p = np.asarray(p)
    return {"n": int(len(p)), "mean_risk": round(float(p.mean()), 4),
            "alert_rate": round(float((p >= threshold).mean()), 4)}


def cbpe_auroc(p: np.ndarray) -> float:
    """Confidence-based performance estimation: expected AUROC if the model
    were calibrated on this batch. Each case counts as a positive with weight
    p and a negative with weight 1-p. Valid under covariate shift only -
    blind to prevalence and concept shift, which is why labels still matter."""
    p = np.asarray(p, float)
    y = np.r_[np.ones(len(p)), np.zeros(len(p))]
    s = np.r_[p, p]
    w = np.r_[p, 1 - p]
    return float(roc_auc_score(y, s, sample_weight=w))


# ------------------------------------------------------------- label tools
def auroc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error."""
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 1e-12)))


def calibration_fit(y: np.ndarray, p: np.ndarray) -> dict:
    """Calibration-in-the-large (intercept with slope fixed at 1) and logistic
    calibration slope."""
    y, z = np.asarray(y, float), _logit(p)
    a = 0.0
    for _ in range(50):  # Newton on intercept offset
        q = 1 / (1 + np.exp(-(a + z)))
        g, h = np.sum(y - q), np.sum(q * (1 - q))
        a += g / max(h, 1e-9)
        if abs(g) < 1e-8:
            break
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return {"citl_intercept": round(float(a), 4),
            "logistic_a": round(float(lr.intercept_[0]), 4),
            "slope": round(float(lr.coef_[0, 0]), 4)}


def realised_metrics(y: np.ndarray, p: np.ndarray, threshold: float = ALERT_THRESHOLD) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    out = {"n": int(len(y)), "events": n_pos}
    if n_pos < 5 or n_neg < 5:
        return out | {"insufficient": True}
    auc = float(roc_auc_score(y, p))
    exp = float(p.sum())
    lo, hi = stats.chi2.ppf(0.025, 2 * n_pos) / 2, stats.chi2.ppf(0.975, 2 * (n_pos + 1)) / 2
    alerts = p >= threshold
    out |= {
        "auroc": round(auc, 4), "auroc_se": round(auroc_se(auc, n_pos, n_neg), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "oe": round(n_pos / exp, 4), "oe_ci": (round(lo / exp, 4), round(hi / exp, 4)),
        "sensitivity": round(float(alerts[y == 1].mean()), 4),
        "ppv": round(float(y[alerts].mean()) if alerts.any() else float("nan"), 4),
        "missed_events": int(((~alerts) & (y == 1)).sum()),
    }
    out |= calibration_fit(y, p)
    return out


def subgroup_metrics(df: pd.DataFrame, p: np.ndarray, attrs) -> dict:
    res = {}
    for a in attrs:
        for val, idx in df.groupby(a).groups.items():
            m = df.index.get_indexer(idx)
            res[f"{a}={val}"] = realised_metrics(df["y"].to_numpy()[m], np.asarray(p)[m])
    return res
