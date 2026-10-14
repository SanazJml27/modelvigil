import json

import numpy as np
import pytest

from modelvigil import detectors as D
from modelvigil import diagnostics as G
from modelvigil.agents import Incident, scripted_policy
from modelvigil.audit import AuditLog, verify_chain
from modelvigil.config import RunConfig
from modelvigil.engine import run
from modelvigil.llm import MockBackend
from modelvigil.model import fit_model
from modelvigil.remediation import pccp_check
from modelvigil.simulator import HospitalStream


@pytest.fixture(scope="module")
def ref_model():
    s = HospitalStream("stable", seed=3)
    ref = s.reference(6000)
    m, info = fit_model(ref.iloc[:4000], seed=3)
    return s, ref.iloc[4000:].reset_index(drop=True), m, info


# ------------------------------------------------------------ primitives
def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert D.psi(x, x) < 1e-6
    assert D.psi(x, x + 1.5) > 0.5


def test_cbpe_matches_truth_when_calibrated():
    rng = np.random.default_rng(1)
    p = rng.beta(1, 6, 20000)
    y = rng.binomial(1, p)
    from sklearn.metrics import roc_auc_score
    assert abs(D.cbpe_auroc(p) - roc_auc_score(y, p)) < 0.02


def test_model_is_calibrated_and_discriminates(ref_model):
    _, hold, m, info = ref_model
    r = D.realised_metrics(hold["y"], m.predict_proba(hold))
    assert r["auroc"] > 0.82
    assert 0.85 < r["oe"] < 1.15


# ------------------------------------------------------------ hypotheses
def test_unit_mismatch_detected_only_at_affected_site(ref_model):
    _, hold, _, _ = ref_model
    s = HospitalStream("lab_unit_change", seed=3)
    res = G.unit_mismatch(hold, s.week(6))
    assert res["supported"]
    assert {(f["feature"], f["site"]) for f in res["findings"]} == {("lactate", "B")}
    assert not G.unit_mismatch(hold, s.week(2))["supported"]


def test_imputation_artifact_detected(ref_model):
    _, hold, _, _ = ref_model
    s = HospitalStream("feed_outage", seed=3)
    res = G.imputation_artifact(hold, s.week(7))
    assert res["supported"] and res["findings"][0]["feature"] == "spo2"
    assert res["findings"][0]["site"] == "C"


def test_residual_effects_flags_concept_but_not_prevalence(ref_model):
    _, hold, m, _ = ref_model
    import pandas as pd
    p_ref = m.predict_proba(hold)
    for scen, expect in [("concept_shift", True), ("prevalence_shift", False)]:
        s = HospitalStream(scen, seed=3)
        d = pd.concat([s.week(w) for w in (6, 7, 8)], ignore_index=True)
        res = G.residual_effects(d, m.predict_proba(d), hold, p_ref)
        assert res["supported"] is expect, (scen, res["top"])


# ------------------------------------------------------------ governance
def test_pccp_blocks_retrain_and_allows_pipeline_fix():
    assert not pccp_check("retrain", None, None)["inside_envelope"]
    assert pccp_check("data_pipeline_fix", None, None)["inside_envelope"]


def test_audit_chain_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(5):
        log.append(i, "t", "e", {"v": i, "x": np.float64(0.5)})
    assert verify_chain(tmp_path / "a.jsonl")[0]
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    e = json.loads(lines[2]); e["payload"]["v"] = 99
    lines[2] = json.dumps(e, sort_keys=True)
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    ok, msg = verify_chain(tmp_path / "a.jsonl")
    assert not ok and "entry 2" in msg


def test_mock_backend_speaks_tool_protocol():
    inc = Incident("X", 1, 1, {"DRIFT_MULTIVARIATE": {}})
    b = MockBackend(scripted_policy(inc))
    out = b.step("sys", [{"role": "user", "content": "go"}], [])
    assert out["tool_calls"][0]["name"] == "get_incident_signals"


# ------------------------------------------------------------ end to end
@pytest.mark.parametrize("scenario,cause,action", [
    ("lab_unit_change", "unit_change", "data_pipeline_fix"),
    ("feed_outage", "feed_outage", "data_pipeline_fix"),
    ("prevalence_shift", "prevalence_shift", "recalibrate"),
    ("subgroup_site", "subgroup_degradation", "subgroup_restriction"),
])
def test_end_to_end_first_actionable_diagnosis(scenario, cause, action, tmp_path):
    res = run(RunConfig(scenario=scenario, seed=7, quiet=True), out_dir=tmp_path)
    actionable = [i for i in res["incidents"] if i["diagnosis"]["root_cause"]
                  not in ("benign_covariate_shift", "new_population_unverified", "insufficient_evidence")]
    assert actionable, "no actionable incident"
    first = actionable[0]
    assert first["diagnosis"]["root_cause"] == cause
    assert first["plan"]["category"] == action
    assert first["diagnosis"]["verified"]
    assert verify_chain(tmp_path / "audit.jsonl")[0]


def test_stable_scenario_raises_no_incident():
    res = run(RunConfig(scenario="stable", seed=7, quiet=True))
    assert res["incidents"] == []


# ------------------------------------------------------------ LLM layer safety
class _HallucinatingBackend:
    """Pretends to be an LLM that skips the investigation and asserts a cause."""
    name = "fake-llm"

    def step(self, system, messages, tools):
        return {"text": "Obviously a unit change.", "tool_calls": [{
            "id": "t1", "name": "submit_diagnosis",
            "input": {"root_cause": "unit_change", "confidence": "high",
                      "evidence": ["vibes"], "rationale": "looks like it"}}]}


def test_verifier_rejects_unsupported_llm_diagnosis():
    import modelvigil.engine as E
    orig = E.make_backend
    E.make_backend = lambda kind: _HallucinatingBackend()
    try:
        res = run(RunConfig(scenario="prevalence_shift", seed=7, quiet=True, llm="fake"))
    finally:
        E.make_backend = orig
    first = res["incidents"][0]["diagnosis"]
    assert first["root_cause"] == "prevalence_shift"      # scripted fallback took over
    assert "rejected" in first["note"]


def test_anthropic_backend_parses_tool_use(monkeypatch):
    from types import SimpleNamespace as NS
    from modelvigil import llm
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    pytest.importorskip("anthropic")
    b = llm.AnthropicBackend(model="test-model")
    fake = NS(content=[NS(type="text", text="checking units"),
                       NS(type="tool_use", id="tu_1", name="check_unit_mismatch", input={})])
    b.client = NS(messages=NS(create=lambda **kw: fake))
    out = b.step("sys", [{"role": "user", "content": "go"}], [])
    assert out["text"] == "checking units"
    assert out["tool_calls"] == [{"id": "tu_1", "name": "check_unit_mismatch", "input": {}}]
