"""Known-answer tests for the MPC evidence layer (impact_pipeline.evidence)."""
import json
import math

import numpy as np
import pytest

from impact_pipeline import evidence as E

CE = E.ComponentEvidence
P = E.ComponentStatus.PRESENT
A = E.ComponentStatus.ABSENT
U = E.ComponentStatus.UNDEFINED


def _ev(principle, estimate, null_mean=0.0, null_sd=1.0, **kw):
    kw.setdefault("bearer_id", "sub-01/awake/run-1")
    kw.setdefault("protocol_id", "proto-A")
    return CE(principle, estimate, null_mean, null_sd, **kw)


def _all(estimate=5.0, **kw):
    return {p: [_ev(p, estimate, **kw)] for p in E.PRINCIPLES}


# --------------------------------------------------------------------------
# component status
# --------------------------------------------------------------------------
def test_component_status_known_answers():
    st, m, r = E.component_status(_ev("RAM", 5.0))
    assert (st, m, r) == (P, 5.0, None)
    # strict threshold: z_present * null_sd = 1.645 exactly is not PRESENT
    assert E.component_status(_ev("RAM", 1.645))[0] == U
    assert E.component_status(_ev("RAM", 1.646))[0] == P
    # equivalence region |excess| <= delta_equiv * null_sd (inclusive)
    assert E.component_status(_ev("RAM", 1.0))[:2] == (A, 1.0)
    assert E.component_status(_ev("RAM", -0.9))[0] == A
    st, m, r = E.component_status(_ev("RAM", 1.3))
    assert st == U and r == E.REASON_INCONCLUSIVE and m == pytest.approx(1.3)
    # far below the null: not equivalent, not present -> inconclusive
    assert E.component_status(_ev("RAM", -4.0))[2] == E.REASON_INCONCLUSIVE
    # null scale matters: excess 3 on a null of sd 2 is only 1.5 null SDs
    assert E.component_status(_ev("RAM", 3.0, null_sd=2.0))[0] == U


def test_component_status_uses_the_sampling_se_in_both_tests():
    z = 1.6448536269514722  # z_{0.95}
    # PRESENT needs the lower bound excess - z*se above 1.645 null SDs
    st, m, _ = E.component_status(_ev("PDI", 2.0, se=0.5))
    assert st == U and m == pytest.approx(2.0 / math.sqrt(1.25))
    assert E.component_status(_ev("PDI", 1.645 + z * 0.5 + 1e-6, se=0.5))[0] == P
    # ABSENT needs |excess| + z*se <= delta_equiv * null_sd (TOST)
    assert E.component_status(_ev("PDI", 0.5, se=0.2))[0] == A
    assert E.component_status(_ev("PDI", 0.5, se=0.4))[0] == U
    # stricter settings are honoured
    assert E.component_status(_ev("PDI", 5.0), z_present=6.0, delta_equiv=1.0)[0] == U
    assert E.component_status(_ev("PDI", 1.5), delta_equiv=1.645)[0] == A


@pytest.mark.parametrize(
    "ev,reason",
    [
        (_ev("NAS", np.nan, defined=False, reason="missing_events"), "missing_events"),
        (_ev("NAS", 2.0, defined=False), "NOT_DEFINED"),
        (_ev("NAS", np.nan), "NON_FINITE_ESTIMATE"),
        (_ev("NAS", 2.0, null_mean=np.nan), E.REASON_NO_NULL),
        (_ev("NAS", 2.0, null_sd=0.0), E.REASON_DEGENERATE_NULL),
        (_ev("NAS", 2.0, null_sd=np.nan), E.REASON_DEGENERATE_NULL),
        (_ev("NAS", 2.0, se=-1.0), "INVALID_SE"),
    ],
)
def test_component_status_undefined_reasons(ev, reason):
    st, m, r = E.component_status(ev)
    assert st == U and math.isnan(m) and r == reason


def test_component_status_parameter_validation():
    with pytest.raises(ValueError, match="delta_equiv"):
        E.component_status(_ev("RAM", 1.0), delta_equiv=2.0)
    with pytest.raises(ValueError, match="alpha"):
        E.component_status(_ev("RAM", 1.0), alpha=0.7)
    with pytest.raises(ValueError):
        E.component_status_array(1.0, 0.0, 1.0, delta_equiv=-1.0)


# --------------------------------------------------------------------------
# Kleene logic
# --------------------------------------------------------------------------
def test_strong_kleene_truth_tables():
    order = [A, U, P]  # F < U < T
    for a in order:
        for b in order:
            assert E.kleene_and([a, b]) == order[min(order.index(a), order.index(b))]
            assert E.kleene_or([a, b]) == order[max(order.index(a), order.index(b))]
    assert E.kleene_and([]) == P and E.kleene_or([]) == A
    assert E.kleene_and(["PRESENT", True, None]) == U
    assert E.kleene_or([E.Verdict.NOT_ATTRIBUTED, False]) == A
    with pytest.raises(ValueError):
        E.kleene_and(["maybe"])


# --------------------------------------------------------------------------
# MPC verdict
# --------------------------------------------------------------------------
def test_verdict_attributed_absent_missing_and_necessity_set():
    v = E.mpc_verdict(_all())
    assert v.verdict == E.Verdict.ATTRIBUTED and v.reasons == []
    assert all(s == P for s in v.component_status.values())
    ev = _all()
    ev["NAS"] = [_ev("NAS", 0.3)]
    v = E.mpc_verdict(ev)
    assert v.verdict == E.Verdict.NOT_ATTRIBUTED and v.reasons == ["ABSENT:NAS"]
    del ev["NAS"]
    v = E.mpc_verdict(ev)
    assert v.verdict == E.Verdict.UNDETERMINED and v.reasons == ["MISSING:NAS"]
    # contested necessity: without NAS in N the same evidence is ATTRIBUTED
    v = E.mpc_verdict(ev, necessity_set="RAM, pdi,IIM,SRPI")
    assert v.verdict == E.Verdict.ATTRIBUTED
    assert v.necessity_set == ("RAM", "PDI", "IIM", "SRPI")


def test_veto_is_not_compensated_by_huge_other_components():
    ev = _all(estimate=1e6)
    ev["SRPI"] = [_ev("SRPI", 0.0)]
    assert E.mpc_verdict(ev).verdict == E.Verdict.NOT_ATTRIBUTED


def test_reason_codes_are_stable_strings():
    ev = {
        "RAM": [_ev("RAM", 1.3)],
        "PDI": [_ev("PDI", 2.0, null_mean=np.nan)],
        "NAS": [_ev("NAS", np.nan, defined=False, reason="insufficient;shape")],
        "IIM": [_ev("IIM", np.nan, defined=False, reason="NOT_IMPLEMENTED",
                    channel="directional")],
        "SRPI": [_ev("SRPI", 9.0, null_sd=0.0)],
    }
    v = E.mpc_verdict(ev)
    assert v.verdict == E.Verdict.UNDETERMINED
    assert v.reasons == [
        "INCONCLUSIVE:RAM",
        "NO_NULL_CALIBRATION:PDI",
        "UNDEFINED:NAS:insufficient,shape",
        "NOT_IMPLEMENTED:IIM:directional",
        "UNDEFINED:SRPI:DEGENERATE_NULL",
    ]
    assert v.reason_string == ";".join(v.reasons)
    assert E.parse_reason("UNDEFINED:NAS:a:b") == ("UNDEFINED", "NAS", "a:b")
    assert E.parse_reason("BEARER_MISMATCH") == ("BEARER_MISMATCH", None, None)
    assert E.verdict_from_reasons(v.reasons) == v.verdict
    json.dumps(v.to_dict())  # JSON-safe


def test_channels_are_combined_by_disjunction():
    # covert responder: behavioural feedback channel credibly null, covert
    # neural channel present -> RAM PRESENT
    ev = _all()
    ev["RAM"] = [
        _ev("RAM", 0.1, channel="behavioural_feedback"),
        _ev("RAM", 4.0, channel="covert_neural"),
    ]
    v = E.mpc_verdict(ev)
    assert v.verdict == E.Verdict.ATTRIBUTED
    assert v.channels["RAM"] == {"behavioural_feedback": A, "covert_neural": P}
    assert v.margins["RAM"] == pytest.approx(4.0)
    # disconnected: exogenous channel null, neural channel not implemented ->
    # UNDETERMINED, never NOT_ATTRIBUTED
    ev["RAM"][1] = _ev("RAM", np.nan, defined=False, reason="NOT_IMPLEMENTED",
                       channel="covert_neural")
    v = E.mpc_verdict(ev)
    assert v.verdict == E.Verdict.UNDETERMINED
    assert v.reasons == ["NOT_IMPLEMENTED:RAM:covert_neural"]
    assert v.channel_reasons["RAM"] == {"covert_neural": "NOT_IMPLEMENTED"}
    # ABSENT only when every channel is ABSENT
    ev["RAM"][1] = _ev("RAM", 0.2, channel="covert_neural")
    assert E.mpc_verdict(ev).reasons == ["ABSENT:RAM"]


def test_bearer_and_protocol_mismatch_force_undetermined():
    ev = _all()
    ev["IIM"] = [_ev("IIM", 5.0, bearer_id="sub-02/awake/run-1")]
    v = E.mpc_verdict(ev)
    assert v.verdict == E.Verdict.UNDETERMINED and v.reasons == ["BEARER_MISMATCH"]
    # never NOT_ATTRIBUTED either, even with a credible absence
    ev["NAS"] = [_ev("NAS", 0.0)]
    v = E.mpc_verdict(ev)
    assert v.verdict == E.Verdict.UNDETERMINED
    assert v.reasons == ["BEARER_MISMATCH", "ABSENT:NAS"]
    assert E.mpc_verdict(ev, require_same_bearer=False).verdict == (
        E.Verdict.NOT_ATTRIBUTED
    )
    # a mismatching principle outside N is not part of the attribution
    ev["NAS"] = [_ev("NAS", 5.0)]
    nset = ("RAM", "PDI", "NAS", "SRPI")
    assert E.mpc_verdict(ev, necessity_set=nset).verdict == E.Verdict.ATTRIBUTED
    # undeclared bearers do not clash
    ev["IIM"] = [_ev("IIM", 5.0, bearer_id=None)]
    assert E.mpc_verdict(ev).verdict == E.Verdict.ATTRIBUTED
    ev["IIM"] = [_ev("IIM", 5.0, protocol_id="proto-B")]
    assert E.mpc_verdict(ev).reasons == ["PROTOCOL_MISMATCH"]
    assert E.mpc_verdict(ev, require_same_protocol=False).verdict == (
        E.Verdict.ATTRIBUTED
    )
    # an undefined component keeps its declared bearer (missingness safety)
    ev = _all()
    ev["IIM"] = [_ev("IIM", np.nan, defined=False, reason="x", bearer_id="other")]
    assert E.mpc_verdict(ev).reasons[0] == "BEARER_MISMATCH"


def test_failed_bearer_coherence_forces_undetermined():
    ev = _all()
    assert E.mpc_verdict(ev, bearer_coherence=True).verdict == E.Verdict.ATTRIBUTED
    v = E.mpc_verdict(ev, bearer_coherence={"coherent": False})
    assert v.verdict == E.Verdict.UNDETERMINED
    assert v.reasons == ["BEARER_MISMATCH:COHERENCE"]


def test_mpc_verdict_input_validation():
    with pytest.raises(ValueError, match="declares principle"):
        E.mpc_verdict({"RAM": [_ev("PDI", 1.0)]})
    with pytest.raises(ValueError, match="empty"):
        E.mpc_verdict(_all(), necessity_set=())
    with pytest.raises(ValueError, match="unknown principle"):
        E.mpc_verdict(_all(), necessity_set=("RAM", "PHI"))
    with pytest.raises(TypeError):
        E.mpc_verdict({"RAM": [1.0]})
    # a single ComponentEvidence is accepted in place of a list
    assert E.mpc_verdict({p: _ev(p, 5.0) for p in E.PRINCIPLES}).verdict == (
        E.Verdict.ATTRIBUTED
    )


# --------------------------------------------------------------------------
# applicability registry
# --------------------------------------------------------------------------
def _registry_payload():
    return {
        "version": 1,
        "entries": [
            {"principle": "IIM", "estimator": "compute_IIM:*", "substrate": "fmri",
             "grain": ["schaefer*", "aal116"],
             "regime": {"n_time": {"min": 300}, "modality": "fmri"},
             "note": "MPC-Bench family B"},
            {"principle": "IIM", "estimator": "compute_IIM:v3", "validated": False},
            {"principle": "NAS", "estimator": "compute_NAS"},
        ],
    }


def test_applicability_registry_matching(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(_registry_payload()))
    reg = E.ApplicabilityRegistry.from_json(path)
    ok = reg.is_validated("IIM", "compute_IIM:v5", "fmri", "schaefer400",
                          n_time=400, modality="fmri")
    assert ok == (True, None)
    assert reg.is_validated("IIM", "compute_IIM:v5", "fmri", "aal116",
                            n_time=300, modality="FMRI") == (True, None)
    assert reg.is_validated("IIM", "compute_IIM:v5", "fmri", "schaefer400",
                            n_time=200, modality="fmri") == (
        False, "REGIME_MISMATCH:n_time")
    assert reg.is_validated("IIM", "compute_IIM:v5", "fmri", "schaefer400")[1] == (
        "REGIME_MISMATCH:n_time")
    assert reg.is_validated("IIM", "compute_IIM:v5", "eeg", "schaefer400",
                            n_time=400, modality="fmri") == (False, "NO_ENTRY")
    assert reg.is_validated("IIM", "compute_IIM:v3", "fmri", "schaefer400",
                            n_time=400, modality="fmri") == (
        False, "MARKED_NOT_VALIDATED")
    assert reg.is_validated("NAS", "compute_NAS", "eeg", "anything") == (True, None)
    assert reg.is_validated("NAS", None) == (False, "ESTIMATOR_UNDECLARED")
    assert reg.to_dict()["entries"][0]["note"] == "MPC-Bench family B"
    with pytest.raises(ValueError, match="unknown principle"):
        E.ApplicabilityRegistry.from_dict([{"principle": "PHI", "estimator": "x"}])
    with pytest.raises(ValueError, match="missing"):
        E.ApplicabilityRegistry.from_dict([{"principle": "IIM"}])


def test_unvalidated_estimator_makes_the_component_undefined():
    reg = E.ApplicabilityRegistry.from_dict(_registry_payload())
    ev = {p: [_ev(p, 5.0, estimator="compute_NAS", substrate="fmri")]
          for p in ("NAS",)}
    ev["IIM"] = [_ev("IIM", 5.0, estimator="compute_IIM:v3", substrate="fmri",
                     grain="schaefer400")]
    v = E.mpc_verdict(ev, necessity_set=("NAS", "IIM"), registry=reg,
                      regime={"n_time": 400, "modality": "fmri"})
    assert v.verdict == E.Verdict.UNDETERMINED
    assert v.reasons == ["ESTIMATOR_NOT_VALIDATED:IIM:compute_IIM:v3"]
    assert v.channel_reasons["IIM"]["default"] == (
        "ESTIMATOR_NOT_VALIDATED:MARKED_NOT_VALIDATED")
    ev["IIM"] = [_ev("IIM", 5.0, estimator="compute_IIM:v5", substrate="fmri",
                     grain="schaefer400")]
    v = E.mpc_verdict(ev, necessity_set=("NAS", "IIM"), registry=reg,
                      regime={"n_time": 400, "modality": "fmri"})
    assert v.verdict == E.Verdict.ATTRIBUTED


# --------------------------------------------------------------------------
# two-anchor normalisation and degree
# --------------------------------------------------------------------------
def test_two_anchor_normalize():
    assert E.two_anchor_normalize(3.0, 2.0, 4.0) == pytest.approx(0.5)
    assert E.two_anchor_normalize(2.0, 2.0, 4.0) == 0.0
    assert E.two_anchor_normalize(6.0, 2.0, 4.0) == pytest.approx(2.0)
    assert math.isnan(E.two_anchor_normalize(3.0, 2.0, 2.0))  # ref <= null
    assert math.isnan(E.two_anchor_normalize(3.0, 5.0, 4.0))
    assert math.isnan(E.two_anchor_normalize(np.nan, 0.0, 1.0))
    out = E.two_anchor_normalize(
        np.array([0.0, 1.0, 2.0]), 0.0, np.array([2.0, 2.0, 0.0])
    )
    np.testing.assert_allclose(out, [0.0, 0.5, np.nan])


def test_degree_known_answers_and_cap():
    c = {"RAM": 0.25, "PDI": 1.0}
    assert E.degree(c, cap=None) == pytest.approx(0.5)  # geometric
    assert E.degree(c, p=1.0) == pytest.approx(0.625)
    assert E.degree(c, p=-math.inf) == E.weakest_link(c) == 0.25
    assert E.degree(c, p=math.inf) == 1.0
    assert E.degree(c, {"RAM": 3.0, "PDI": 1.0}, cap=None) == pytest.approx(
        0.25 ** 0.75)
    # the cap blocks compensation: one weak component, four huge ones
    c5 = {"RAM": 0.1, "PDI": 10.0, "NAS": 10.0, "IIM": 10.0, "SRPI": 10.0}
    assert E.degree(c5) == pytest.approx(0.1 ** 0.2)
    assert E.degree(c5, cap=None) == pytest.approx((0.1 * 10.0 ** 4) ** 0.2)
    # zero component: zero for p <= 0; zero-weight components are ignored
    assert E.degree({"a": 0.0, "b": 1.0}) == 0.0
    assert E.degree({"a": 0.0, "b": 1.0}, p=-2.0) == 0.0
    assert E.degree({"a": 0.0, "b": 1.0}, p=2.0) == pytest.approx(math.sqrt(0.5))
    assert E.degree({"a": np.nan, "b": 0.5}, {"a": 0.0, "b": 1.0}) == 0.5
    assert math.isnan(E.degree({"a": np.nan, "b": 0.5}))
    assert E.degree({"a": -0.5, "b": 1.0}, p=1.0) == 0.5  # below null counts as 0
    with pytest.raises(ValueError):
        E.degree(c, {"RAM": -1.0, "PDI": 1.0})
    with pytest.raises(ValueError):
        E.degree(c, cap=0.0)


def test_degree_power_mean_inequality_and_monotonicity():
    rng = np.random.default_rng(0)
    ps = (-math.inf, -2.0, -0.5, 0.0, 0.5, 1.0, 3.0, math.inf)
    for _ in range(2000):
        c = rng.uniform(0.01, 1.5, 5)
        w = rng.uniform(0.1, 1.0, 5)
        d = [E.degree(c, w, p=p) for p in ps]
        assert all(a <= b + 1e-12 for a, b in zip(d, d[1:]))
        j = int(rng.integers(5))
        c2 = c.copy()
        c2[j] += rng.uniform(0.0, 0.5)
        assert E.degree(c2, w) >= E.degree(c, w) - 1e-12


def test_degree_interval_delta_and_bootstrap():
    J = 4
    c = {k: 0.6 for k in "abcd"}
    se = {k: 0.05 for k in "abcd"}
    geo = E.degree_interval(c, se)
    assert geo["degree"] == pytest.approx(0.6)
    assert geo["se"] == pytest.approx(0.05 / math.sqrt(J))  # sum w_j^2 se^2
    assert geo["lo"] == pytest.approx(0.6 - 1.959964 * 0.025, rel=1e-5)
    ari = E.degree_interval({"a": 0.2, "b": 0.8}, [0.1, 0.3], p=1.0)
    assert ari["se"] == pytest.approx(math.sqrt(0.25 * 0.01 + 0.25 * 0.09))
    wl = E.degree_interval({"a": 0.2, "b": 0.8}, [0.1, 0.3], p=-math.inf)
    assert wl["degree"] == 0.2 and wl["se"] == pytest.approx(0.1)
    capped = E.degree_interval({"a": 0.5, "b": 3.0}, [0.1, 5.0])
    # the capped component has zero derivative
    assert capped["se"] == pytest.approx(0.5 * math.sqrt(0.5) * 0.1 / 0.5)
    zero = E.degree_interval({"a": 0.0, "b": 1.0}, [0.1, 0.1])
    assert zero["degree"] == 0.0 and math.isnan(zero["se"])
    boot = E.degree_interval(c, se, method="bootstrap", n_boot=4000, seed=1)
    assert boot["se"] == pytest.approx(geo["se"], rel=0.1)
    assert boot["lo"] < 0.6 < boot["hi"] and boot["n_boot"] == 4000
    again = E.degree_interval(c, se, method="bootstrap", n_boot=4000, seed=1)
    assert again == boot
    samples = np.column_stack([np.linspace(0.4, 0.8, 101)] * 4)
    sb = E.degree_interval(c, method="bootstrap", samples=samples, level=0.9)
    assert sb["lo"] == pytest.approx(0.42) and sb["hi"] == pytest.approx(0.78)
    with pytest.raises(ValueError):
        E.degree_interval(c, se, method="jackknife")


def test_verdict_stability():
    V = E.Verdict
    out = E.verdict_stability([V.ATTRIBUTED] * 7 + [V.UNDETERMINED] * 3,
                              reference=V.ATTRIBUTED)
    assert out["modal_verdict"] == "ATTRIBUTED"
    assert out["flip_rate"] == pytest.approx(0.3)
    assert out["reference_flip_rate"] == pytest.approx(0.3)
    assert out["counts"] == {"ATTRIBUTED": 7, "NOT_ATTRIBUTED": 0, "UNDETERMINED": 3}
    tie = E.verdict_stability(["ATTRIBUTED", "UNDETERMINED"])
    assert tie["modal_verdict"] == "UNDETERMINED"  # conservative tie-break
    mv = E.mpc_verdict(_all())
    assert E.verdict_stability([mv, mv])["flip_rate"] == 0.0
    assert E.verdict_stability([])["modal_verdict"] is None


# --------------------------------------------------------------------------
# bearer coherence
# --------------------------------------------------------------------------
def _two_systems(seed=0, n_time=600, coupling=0.8):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((6, n_time))
    for t in range(1, n_time):
        x[:3, t] += 0.5 * x[:3, t - 1]
        x[3:, t] += 0.5 * x[3:, t - 1] + coupling * x[0, t - 1]
    return x


def test_bearer_coherence_known_answers():
    coupled = _two_systems(coupling=0.8)
    res = E.bearer_coherence(coupled, {"A": [0, 1, 2], "B": [3, 4, 5]},
                             n_surrogates=99, seed=0)
    assert res["coherent"] and res["reason"] is None
    assert res["pairs"][0]["p"] == pytest.approx(0.01) and res["min_z"] > 5
    indep = _two_systems(coupling=0.0)
    res0 = E.bearer_coherence(indep, {"A": [0, 1, 2], "B": [3, 4, 5]},
                              n_surrogates=99, seed=0)
    assert not res0["coherent"] and res0["reason"] == "INCOHERENT:A|B"
    # overlapping node sets share a bearer by identity
    res_ov = E.bearer_coherence(indep, [[0, 1, 2], [2, 3]], n_surrogates=99, seed=0)
    assert res_ov["coherent"] and res_ov["set_names"] == ["set0", "set1"]
    # deterministic given the seed; lag 0 supported
    again = E.bearer_coherence(coupled, {"A": [0, 1, 2], "B": [3, 4, 5]},
                               n_surrogates=99, seed=0)
    assert again["pairs"] == res["pairs"]
    assert E.bearer_coherence(coupled, {"A": [0], "B": [3]}, lag=0,
                              n_surrogates=49)["coherent"]
    # verdict wiring
    v = E.mpc_verdict(_all(), bearer_coherence=res0)
    assert v.verdict == E.Verdict.UNDETERMINED


def test_bearer_coherence_edge_cases():
    x = _two_systems()
    assert E.bearer_coherence(x, {"A": [0, 1]})["coherent"]
    assert E.bearer_coherence(x, {"A": [0], "B": [3]}, n_surrogates=0)["reason"] == (
        "NO_SURROGATES")
    x[5] = 1.0
    res = E.bearer_coherence(x, {"A": [0], "B": [5]}, n_surrogates=10)
    assert not res["coherent"] and res["reason"] == "DEGENERATE_SET:B"
    with pytest.raises(ValueError, match="out of range"):
        E.bearer_coherence(x, {"A": [0], "B": [9]})
    with pytest.raises(ValueError, match="finite"):
        bad = x.copy()
        bad[0, 0] = np.nan
        E.bearer_coherence(bad, {"A": [0], "B": [1]})
