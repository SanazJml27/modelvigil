"""The deployed clinical prediction model under surveillance.

A gradient-boosted 48h-deterioration model with Platt calibration. Model
versions are immutable; every modification produces a new version with a
parent pointer, so the audit trail can always reconstruct what was live.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import FEATURE_NAMES


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class ModelVersion:
    estimator: HistGradientBoostingClassifier
    platt: tuple[float, float]                 # a + b * raw_logit
    recal: tuple[float, float] = (0.0, 1.0)    # post-hoc recalibration a + b * logit
    site_recal: dict = field(default_factory=dict)   # optional per-site recalibration
    revision: dict | None = None               # optional model revision: extra feature terms on the logit
    version: str = "1.0.0"
    parent: str | None = None
    note: str = "initial release"

    def raw_logit(self, X: pd.DataFrame) -> np.ndarray:
        p = self.estimator.predict_proba(X[FEATURE_NAMES])[:, 1]
        return _logit(p)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        z = self.platt[0] + self.platt[1] * self.raw_logit(df)
        z = self.recal[0] + self.recal[1] * z
        if self.revision:
            r = self.revision
            z = r["a"] + r["b"] * z
            for f, use_log, mu, sd, coef in r["terms"]:
                x = df[f].astype(float).to_numpy()
                x = np.log(np.clip(x, 1e-3, None)) if use_log else x
                z = z + coef * np.nan_to_num((x - mu) / sd)
        if self.site_recal and "site" in df:
            for site, (a, b) in self.site_recal.items():
                m = (df["site"] == site).to_numpy()
                z[m] = a + b * z[m]
        return _sigmoid(z)

    def fingerprint(self) -> str:
        payload = json.dumps({"v": self.version, "platt": self.platt, "recal": self.recal,
                              "site": self.site_recal, "rev": self.revision}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def derive(self, *, version: str, note: str, recal=None, site_recal=None,
               estimator=None, platt=None, revision=None) -> "ModelVersion":
        return ModelVersion(
            estimator=estimator or self.estimator,
            platt=platt or self.platt,
            recal=recal if recal is not None else self.recal,
            site_recal=site_recal if site_recal is not None else dict(self.site_recal),
            revision=revision if revision is not None else self.revision,
            version=version, parent=self.version, note=note)


def fit_model(df: pd.DataFrame, seed: int = 0, sample_weight=None,
              version: str = "1.0.0", note: str = "initial release") -> tuple[ModelVersion, dict]:
    """Fit on a development cohort: 70% train, 30% calibration."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(0.7 * len(df))
    tr, ca = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    w_tr = None if sample_weight is None else np.asarray(sample_weight)[idx[:cut]]
    est = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=seed)
    est.fit(tr[FEATURE_NAMES], tr["y"], sample_weight=w_tr)
    z = _logit(est.predict_proba(ca[FEATURE_NAMES])[:, 1]).reshape(-1, 1)
    lr = LogisticRegression(C=1e6).fit(z, ca["y"])
    mv = ModelVersion(est, (float(lr.intercept_[0]), float(lr.coef_[0, 0])),
                      version=version, note=note)
    p = mv.predict_proba(ca)
    return mv, {"calibration_auroc": float(roc_auc_score(ca["y"], p)),
                "calibration_oe": float(ca["y"].sum() / p.sum())}
