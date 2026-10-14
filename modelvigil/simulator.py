"""Synthetic hospital: adult inpatients scored weekly for 48-hour clinical
deterioration (unplanned ICU transfer or death).

The data-generating process is known, so every injected failure has a
ground-truth root cause and a ground-truth correct action. That is what makes
the agents *benchmarkable* - the scarcity of public post-deployment monitoring
benchmarks is one of the open gaps this project addresses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FEATURE_NAMES

SITES_REF = ["A", "B", "C"]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    root_cause: str            # ground-truth label for the benchmark
    correct_action: str        # ground-truth action category
    harmful: bool


SCENARIOS: dict[str, Scenario] = {
    "stable": Scenario(
        "stable", "No change. Any alarm is a false alarm.",
        "none", "no_change", False),
    "benign_covariate": Scenario(
        "benign_covariate",
        "Older, more comorbid case-mix after a geriatric ward opens. The model "
        "is well specified for these patients: inputs drift but performance holds.",
        "benign_covariate_shift", "no_change", False),
    "lab_unit_change": Scenario(
        "lab_unit_change",
        "Site B's laboratory system starts reporting lactate in mg/dL instead "
        "of mmol/L. Risk scores inflate and alerts flood the wards.",
        "unit_change", "data_pipeline_fix", True),
    "feed_outage": Scenario(
        "feed_outage",
        "The SpO2 device feed at site C drops half its readings; the interface "
        "engine silently back-fills the default value 97%. Hypoxic patients are missed.",
        "feed_outage", "data_pipeline_fix", True),
    "prevalence_shift": Scenario(
        "prevalence_shift",
        "Respiratory-virus season raises deterioration risk through factors "
        "the model does not see. Inputs look unchanged; the model under-predicts.",
        "prevalence_shift", "recalibrate", True),
    "concept_shift": Scenario(
        "concept_shift",
        "A new early-sepsis bundle (lactate-triggered fluids/antibiotics) breaks "
        "the lactate-outcome relationship. Discrimination decays; recalibration "
        "cannot fix it.",
        "concept_shift", "retrain", True),
    "subgroup_site": Scenario(
        "subgroup_site",
        "A renal/transplant unit (site D) joins the network. Creatinine carries "
        "far more risk there; overall metrics barely move while site D is "
        "systematically under-triaged.",
        "subgroup_degradation", "subgroup_restriction", True),
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _generate(n: int, rng: np.random.Generator, *, age_shift: float = 0.0,
              comorb_shift: float = 0.0, sites=None, site_p=None,
              renal_site: str | None = None) -> pd.DataFrame:
    sites = sites or SITES_REF
    site = rng.choice(sites, size=n, p=site_p)
    sex = rng.choice(["F", "M"], size=n)
    age = np.clip(rng.normal(64 + age_shift, 16, n), 18, 100)
    comorb = np.clip(rng.poisson(1.6 + 0.025 * (age - 64) + comorb_shift), 0, 12)
    emergency = rng.binomial(1, 0.55, n)

    # latent acute severity drives correlated vitals/labs
    sev = rng.normal(0, 1, n) + 0.25 * emergency + 0.01 * (age - 64)
    hr = rng.normal(84 + 9 * sev, 11)
    rr = rng.normal(18 + 2.6 * sev, 2.6)
    sbp = rng.normal(128 - 9 * sev, 17)
    spo2 = np.clip(rng.normal(95.5 - 1.6 * sev, 2.0), 70, 100)
    temp = rng.normal(37.1 + 0.35 * sev, 0.6)
    lactate = np.exp(rng.normal(np.log(1.4) + 0.28 * sev, 0.35))
    creat = np.exp(rng.normal(np.log(85) + 0.12 * sev + 0.02 * comorb, 0.35))
    wbc = np.exp(rng.normal(np.log(9) + 0.15 * sev, 0.35))

    renal = np.zeros(n, dtype=bool)
    if renal_site is not None:
        renal = site == renal_site
        creat = np.where(renal, creat * np.exp(rng.normal(0.9, 0.35, n)), creat)

    df = pd.DataFrame({
        "age": age.round(0), "heart_rate": hr.round(0), "resp_rate": rr.round(0),
        "sbp": sbp.round(0), "spo2": spo2.round(0), "temp": temp.round(1),
        "lactate": lactate.round(2), "creatinine": creat.round(0), "wbc": wbc.round(1),
        "comorbidity": comorb.astype(float), "emergency_adm": emergency.astype(float),
        "site": site, "sex": sex,
    })
    df["age_band"] = pd.cut(df["age"], [0, 50, 70, 200], labels=["<50", "50-69", "70+"]).astype(str)
    df["_renal"] = renal
    return df


def true_logit(df: pd.DataFrame, *, intercept_shift: float = 0.0,
               lactate_mult: float = 1.0, renal_creat_mult: float = 1.0) -> np.ndarray:
    """Ground-truth risk model (unknown to the deployed model)."""
    lac_term = 1.35 * np.log(df["lactate"] / 1.4)
    creat_term = 0.9 * np.log(df["creatinine"] / 85)
    creat_term = np.where(df["_renal"], creat_term * renal_creat_mult + 0.55, creat_term)
    return (
        -4.45 + intercept_shift
        + 0.022 * (df["age"] - 64)
        + 0.030 * (df["heart_rate"] - 84)
        + 0.16 * (df["resp_rate"] - 18)
        - 0.022 * (df["sbp"] - 128)
        - 0.30 * (df["spo2"] - 95.5)
        + 0.45 * np.abs(df["temp"] - 37.1)
        + lactate_mult * lac_term
        + creat_term
        + 0.18 * df["comorbidity"]
        + 0.40 * df["emergency_adm"]
    )


class HospitalStream:
    """Generates a reference cohort and a weekly deployment stream with one
    injected scenario starting at ``shift_week``."""

    def __init__(self, scenario: str, seed: int = 0, n_per_week: int = 600,
                 n_weeks: int = 16, shift_week: int = 5):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
        self.scenario = SCENARIOS[scenario]
        self.seed = seed
        self.n_per_week = n_per_week
        self.n_weeks = n_weeks
        self.shift_week = shift_week
        self._weeks: dict[int, pd.DataFrame] = {}

    # -- reference -------------------------------------------------------
    def reference(self, n: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed * 1000 + 1)
        df = _generate(n, rng)
        df["y"] = rng.binomial(1, _sigmoid(true_logit(df)))
        df["week"] = 0
        return df.drop(columns="_renal")

    # -- deployment ------------------------------------------------------
    def week(self, w: int) -> pd.DataFrame:
        if w in self._weeks:
            return self._weeks[w]
        rng = np.random.default_rng(self.seed * 1000 + 100 + w)
        s = self.scenario.name
        active = w >= self.shift_week
        gen_kw, logit_kw = {}, {}
        if active and s == "benign_covariate":
            gen_kw = dict(age_shift=9.0, comorb_shift=0.6)
        if active and s == "subgroup_site":
            gen_kw = dict(sites=SITES_REF + ["D"], site_p=[0.24, 0.24, 0.22, 0.30], renal_site="D")
            logit_kw = dict(renal_creat_mult=2.2)
        if active and s == "prevalence_shift":
            logit_kw = dict(intercept_shift=0.85)
        if active and s == "concept_shift":
            logit_kw = dict(lactate_mult=-0.5, intercept_shift=0.0)

        df = _generate(self.n_per_week, rng, **gen_kw)
        df["y"] = rng.binomial(1, _sigmoid(true_logit(df, **logit_kw)))

        # data-pipeline faults are applied AFTER outcomes are generated: the
        # patient is the same, only the recorded value is corrupted
        if active and s == "lab_unit_change":
            m = df["site"] == "B"
            df.loc[m, "lactate"] = (df.loc[m, "lactate"] * 9.01).round(1)
        if active and s == "feed_outage":
            m = (df["site"] == "C") & (rng.random(len(df)) < 0.55)
            df.loc[m, "spo2"] = 97.0

        df["week"] = w
        df = df.drop(columns="_renal")
        self._weeks[w] = df
        return df
