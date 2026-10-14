"""Static configuration: feature dictionary, monitoring thresholds and the
Predetermined Change Control Plan (PCCP) envelope.

Everything that a clinical safety officer or notified body would want to
inspect lives here, in one place, as plain data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Feature dictionary (FHIR Observation-aligned; LOINC codes where applicable)
# ---------------------------------------------------------------------------
FEATURES: dict[str, dict] = {
    "age":            {"loinc": "30525-0", "unit": "a",       "range": (16, 105)},
    "heart_rate":     {"loinc": "8867-4",  "unit": "/min",    "range": (25, 220)},
    "resp_rate":      {"loinc": "9279-1",  "unit": "/min",    "range": (4, 60)},
    "sbp":            {"loinc": "8480-6",  "unit": "mm[Hg]",  "range": (50, 250)},
    "spo2":           {"loinc": "59408-5", "unit": "%",       "range": (60, 100)},
    "temp":           {"loinc": "8310-5",  "unit": "Cel",     "range": (33, 42.5)},
    "lactate":        {"loinc": "2524-7",  "unit": "mmol/L",  "range": (0.3, 20)},
    "creatinine":     {"loinc": "14682-9", "unit": "umol/L",  "range": (20, 1500)},
    "wbc":            {"loinc": "6690-2",  "unit": "10*9/L",  "range": (0.5, 60)},
    "comorbidity":    {"loinc": None,      "unit": "score",   "range": (0, 12)},
    "emergency_adm":  {"loinc": None,      "unit": "bool",    "range": (0, 1)},
}
FEATURE_NAMES: list[str] = list(FEATURES)

# Subgroup attributes monitored for equity / site-level degradation.
# They are NOT model inputs.
SUBGROUP_ATTRS: list[str] = ["site", "sex", "age_band"]

# Known laboratory unit conversions (factor = value_in_wrong_unit / value_in_expected_unit)
UNIT_CONVERSIONS: dict[str, list[tuple[str, float]]] = {
    "lactate":    [("mg/dL", 9.01)],
    "creatinine": [("mg/dL", 1 / 88.4)],
    "temp":       [("degF", None)],  # affine; handled separately
    "wbc":        [("/uL", 1000.0)],
}

# ---------------------------------------------------------------------------
# Deployment parameters
# ---------------------------------------------------------------------------
ALERT_THRESHOLD = 0.20        # predicted 48h deterioration risk that triggers a rapid-response review
LABEL_LAG_WEEKS = 2           # outcomes (ICU transfer / death within 48h, chart-reviewed) arrive late
LABEL_WINDOW_WEEKS = 3        # rolling window of labelled weeks used for realised metrics


@dataclass(frozen=True)
class MonitoringThresholds:
    # label-free
    psi_feature: float = 0.25            # population stability index, per feature
    domain_auc: float = 0.60             # multivariate drift (classifier two-sample test)
    out_of_range_rate: float = 0.02      # physiologically implausible values
    mode_spike: float = 0.15             # excess mass at a single value (imputation artefact)
    alert_rate_rel_change: float = 0.35  # relative change in alert burden
    cbpe_auc_drop: float = 0.03          # confidence-based expected AUROC drop
    # label-based (lagged)
    oe_low: float = 0.80                 # observed/expected events
    oe_high: float = 1.25
    auc_drop: float = 0.05
    z_overall: float = 1.96
    z_subgroup: float = 2.81             # ~Bonferroni for ~10 subgroups
    min_subgroup_events: int = 12


THRESHOLDS = MonitoringThresholds()


@dataclass(frozen=True)
class PCCP:
    """Predetermined Change Control Plan.

    Modifications inside this envelope were pre-specified in the technical
    documentation and may be enacted after a single sign-off. Anything outside
    is a potential *substantial modification* (EU AI Act Art. 3(23), 43(4))
    and requires full clinical-safety review.
    """
    allowed_auto: tuple[str, ...] = ("recalibrate_intercept", "recalibrate_logistic",
                                     "data_pipeline_fix", "enhanced_surveillance", "no_change")
    max_abs_intercept_shift: float = 1.5
    slope_bounds: tuple[float, float] = (0.6, 1.5)
    min_shadow_auc_noninferiority: float = -0.01   # candidate AUROC - current AUROC must exceed this
    max_subgroup_oe_deviation: float = 0.35        # |O/E-1| in any subgroup after change
    requires_review: tuple[str, ...] = ("retrain", "subgroup_restriction", "rollback")


PCCP_PLAN = PCCP()


@dataclass
class RunConfig:
    scenario: str = "lab_unit_change"
    seed: int = 7
    n_ref: int = 9000
    n_per_week: int = 600
    n_weeks: int = 16
    shift_week: int = 5
    approval: str = "policy"       # policy | interactive | auto
    llm: str = "mock"              # mock | anthropic
    out_dir: str = "runs"
    quiet: bool = False
    extra: dict = field(default_factory=dict)
