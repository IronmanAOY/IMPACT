"""Known-answer tests for the MPC evidence layer v2 (impact_pipeline.evidence)."""
import hashlib
import json
import math

import numpy as np
import pytest

from impact_pipeline import evidence as E
from impact_pipeline import mpc_metrics as mm

CE = E.ComponentEvidence
P = E.ComponentStatus.PRESENT
A = E.ComponentStatus.ABSENT
U = E.ComponentStatus.UNDEFINED
V = E.Verdict
Z95 = 1.6448536269514722  # z_{0.95}


def _ev(principle, estimate, null_mean=0.0, null_sd=0.0, **kw):
    """Evidence on a reference 1 above an analytic null 0 (c = estimate)."""
    kw.setdefault("reference", 1.0)
    kw.setdefault("se", 0.05)
    kw.setdefault("bearer_id", "sub-01/awake/run-1")
    kw.setdefault("protocol_id", "proto-A")
    return CE(principle, estimate, null_mean, null_sd, **kw)


def _all(estimate=1.0, **kw):
    return {p: [_ev(p, estimate, **kw)] for p in E.PRINCIPLES}


# --------------------------------------------------------------------------
# construct-scale component status (V2-2)
# --------------------------------------------------------------------------
def test_component_assessment_known_answers():
    a = E.component_assessment(_ev("RAM", 1.0, se=0.1))
    assert a.status == P and a.reason is None
    assert a.c == 1.0 and a.se == pytest.approx(0.1)
    assert a.lower == pytest.approx(1.0 - Z95 * 0.1)
    assert a.upper == pytest.approx(1.0 + Z95 * 0.1)
    assert a.margin_present == pytest.approx(1.0 - Z95 * 0.1 - 0.25)
    assert a.margin_absent == pytest.approx(0.10 - (1.0 + Z95 * 0.1))
    # the two-anchor scale: c = (m - nu) / (rho - nu)
    a = E.component_assessment(_ev("NAS", 3.0, null_mean=1.0, reference=5.0))
    assert a.c == pytest.approx(0.5)
    # exact computations need no sampling SE; the cutoffs are strict
    ex = dict(se=0.0, exact=True)
    assert E.component_assessment(_ev("IIM", 0.25, **ex)).status == U
    assert E.component_assessment(_ev("IIM", 0.2501, **ex)).status == P
    assert E.component_assessment(_ev("IIM", 0.10, **ex)).status == U
    assert E.component_assessment(_ev("IIM", 0.0999, **ex)).status == A
    st, margin, reason = E.component_status(_ev("IIM", 0.2, **ex))
    assert (st, reason) == (U, E.REASON_INCONCLUSIVE)
    assert margin == pytest.approx(0.2 - 0.25)


def test_absent_includes_estimates_credibly_below_the_null():
    # v1 called this INCONCLUSIVE; v2 has no asymmetry favouring
    # non-falsification: an upper bound below delta is ABSENT.
    a = E.component_assessment(_ev("RAM", -1.0, se=0.1))
    assert a.status == A and a.c == -1.0
    assert a.upper == pytest.approx(-1.0 + Z95 * 0.1)
    # near zero but too uncertain: inconclusive
    assert E.component_assessment(_ev("RAM", 0.0, se=0.1)).status == U
    assert E.component_assessment(_ev("RAM", 0.0, se=0.05)).status == A


def test_se_combines_sampling_null_monte_carlo_and_reference_error():
    # estimate scale: m = 0.5, nu = 0 (K = 25, sd 0.5 -> se_nu = 0.1), rho = 1
    ev = _ev("PDI", 0.5, null_mean=0.0, null_sd=0.5, n_null=25, se=0.2,
             reference=1.0, reference_se=0.1)
    a = E.component_assessment(ev)
    assert a.c == pytest.approx(0.5)
    assert a.se_sampling == pytest.approx(0.2)  # se_m / (rho - nu)
    assert a.se_null == pytest.approx(0.05)  # |(c - 1) / (rho - nu)| se_nu
    assert a.se_reference == pytest.approx(0.05)  # |c / (rho - nu)| se_rho
    assert a.se == pytest.approx(math.sqrt(0.2 ** 2 + 0.05 ** 2 + 0.05 ** 2))
    # excess scale (reference = rho - nu): dc/dnu = -1 / (rho - nu)
    ev = _ev("PDI", 0.5, null_mean=0.0, null_sd=0.5, n_null=25, se=0.2,
             reference=1.0, reference_se=0.1, reference_scale="excess")
    a = E.component_assessment(ev)
    assert a.se_null == pytest.approx(0.1)
    assert a.se == pytest.approx(math.sqrt(0.04 + 0.01 + 0.0025))
    # at c = 1 on the estimate scale the null error cancels to first order
    a = E.component_assessment(_ev("PDI", 1.0, null_sd=0.5, n_null=25, se=0.2))
    assert a.se_null == 0.0
    # an unavailable reference SE (None/NaN) counts as 0
    ev = _ev("PDI", 0.5, se=0.2, reference_se=float("nan"))
    assert E.component_assessment(ev).se_reference == 0.0


@pytest.mark.parametrize("scale", ["estimate", "excess"])
def test_delta_method_se_matches_monte_carlo_and_finite_differences(scale):
    m0, nu0, sd_null, k, se_m, rho0, se_rho = 2.0, 0.5, 0.8, 40, 0.05, 3.0, 0.04
    ref0 = rho0 if scale == "estimate" else rho0 - nu0
    ev = CE("NAS", m0, nu0, sd_null, se=se_m, n_null=k, reference=ref0,
            reference_se=se_rho, reference_scale=scale)
    a = E.component_assessment(ev)

    def c_of(m, nu, ref):
        return (m - nu) / (ref if scale == "excess" else ref - nu)

    h = 1e-6
    grads = [
        (c_of(m0 + h, nu0, ref0) - c_of(m0 - h, nu0, ref0)) / (2 * h),
        (c_of(m0, nu0 + h, ref0) - c_of(m0, nu0 - h, ref0)) / (2 * h),
        (c_of(m0, nu0, ref0 + h) - c_of(m0, nu0, ref0 - h)) / (2 * h),
    ]
    ses = [se_m, sd_null / math.sqrt(k), se_rho]
    fd = math.sqrt(sum((g * s) ** 2 for g, s in zip(grads, ses)))
    assert a.se == pytest.approx(fd, rel=1e-6)
    rng = np.random.default_rng(7)
    n = 200_000
    draws = c_of(
        m0 + se_m * rng.standard_normal(n),
        nu0 + ses[1] * rng.standard_normal(n),
        ref0 + se_rho * rng.standard_normal(n),
    )
    assert float(np.std(draws)) == pytest.approx(a.se, rel=0.02)


@pytest.mark.parametrize(
    "ev,reason",
    [
        (_ev("NAS", np.nan, defined=False, reason="missing_events"), "missing_events"),
        (_ev("NAS", 2.0, defined=False), "NOT_DEFINED"),
        (_ev("NAS", np.nan), "NON_FINITE_ESTIMATE"),
        (_ev("NAS", 2.0, null_mean=np.nan), E.REASON_NO_NULL),
        (_ev("NAS", 2.0, null_sd=np.nan, n_null=1), E.REASON_DEGENERATE_NULL),
        (_ev("NAS", 2.0, null_sd=-1.0, n_null=5), E.REASON_DEGENERATE_NULL),
        (_ev("NAS", 2.0, n_null=-1), E.REASON_DEGENERATE_NULL),
        (_ev("NAS", 2.0, reference=None), E.REASON_INVALID_ANCHORS),
        (_ev("NAS", 2.0, null_mean=1.0, reference=1.0), E.REASON_INVALID_ANCHORS),
        (_ev("NAS", 2.0, null_mean=2.0, reference=1.0), E.REASON_INVALID_ANCHORS),
        (_ev("NAS", 2.0, reference=0.0, reference_scale="excess"),
         E.REASON_INVALID_ANCHORS),
        (_ev("NAS", 2.0, reference=np.inf), E.REASON_INVALID_ANCHORS),
        (_ev("NAS", 2.0, se=-1.0), E.REASON_INVALID_SE),
        (_ev("NAS", 2.0, reference_se=-0.1), E.REASON_INVALID_SE),
        (_ev("NAS", 2.0, se=0.0), E.REASON_NO_SAMPLING_SE),
        (_ev("NAS", 2.0, se=np.nan), E.REASON_NO_SAMPLING_SE),
        (_ev("NAS", 2.0, se=None), E.REASON_NO_SAMPLING_SE),
    ],
)
def test_component_status_undefined_reasons(ev, reason):
    st, m, r = E.component_status(ev)
    assert st == U and math.isnan(m) and r == reason


def test_missing_sampling_se_keeps_c_but_not_a_status():
    a = E.component_assessment(_ev("RAM", 0.8, se=0.0))
    assert a.status == U and a.reason == E.REASON_NO_SAMPLING_SE
    assert a.c == pytest.approx(0.8) and math.isnan(a.lower)
    # the same estimate from an exact (known-TPM) computation is determinate
    assert E.component_assessment(_ev("RAM", 0.8, se=0.0, exact=True)).status == P
    # a zero-variance null is fine (se_nu = 0) when there are surrogates
    assert E.component_assessment(_ev("RAM", 0.8, null_sd=0.0, n_null=10)).status == P


def test_se_from_few_replicates_uses_the_student_t_quantile():
    from scipy.stats import t as t_dist

    # only a sampling part (analytic null, no reference SE): nu_eff = se_df
    a = E.component_assessment(_ev("PDI", 0.55, se=0.15, se_df=4))
    t4 = t_dist.ppf(0.95, 4)
    assert a.df == pytest.approx(4.0) and a.quantile == pytest.approx(t4)
    assert a.lower == pytest.approx(0.55 - t4 * 0.15)
    assert a.upper == pytest.approx(0.55 + t4 * 0.15)
    # the normal quantile would call this PRESENT; t_4 does not
    assert E.component_assessment(_ev("PDI", 0.55, se=0.15)).status == P
    assert a.status == U and a.reason == E.REASON_INCONCLUSIVE
    # without se_df the SE counts as known (normal quantile, as before)
    b = E.component_assessment(_ev("PDI", 0.55, se=0.15))
    assert math.isinf(b.df) and b.quantile == pytest.approx(Z95)
    # Welch-Satterthwaite: a known null Monte-Carlo part raises the df
    c = E.component_assessment(
        _ev("PDI", 0.6, null_sd=0.3, n_null=9, se=0.1, se_df=4))
    s_samp, s_null = 0.1, abs(0.6 - 1.0) * 0.3 / 3.0
    var = s_samp ** 2 + s_null ** 2
    assert c.df == pytest.approx(4.0 * (var / s_samp ** 2) ** 2)
    assert c.quantile == pytest.approx(t_dist.ppf(0.95, c.df))
    # exact computations have no sampling part: normal quantile
    d = E.component_assessment(_ev("IIM", 0.6, se=0.0, exact=True, se_df=4))
    assert math.isinf(d.df)
    # invalid degrees of freedom
    for bad in (0, -1.0):
        e = E.component_assessment(_ev("PDI", 0.6, se=0.1, se_df=bad))
        assert e.status == U and e.reason == E.REASON_INVALID_SE
    assert E.component_assessment(_ev("PDI", 0.6, se=0.1, se_df=None)).status == P
    assert E.component_assessment(
        _ev("PDI", 0.6, se=0.1, se_df=float("nan"))).status == P
    # the assessment dict carries the degrees of freedom and the quantile
    rec = a.to_dict()
    assert rec["df"] == pytest.approx(4.0) and rec["quantile"] == pytest.approx(t4)
    assert b.to_dict()["df"] is None


def test_status_array_matches_scalar_with_degrees_of_freedom():
    rng = np.random.default_rng(12)
    n = 4000
    est = rng.normal(0.4, 0.5, n)
    nm = rng.normal(0, 0.1, n)
    nsd = rng.gamma(2.0, 0.1, n)
    k = rng.integers(0, 40, n)
    ref = nm + rng.gamma(2.0, 0.5, n)
    se = np.abs(rng.normal(0, 0.2, n))
    sdf = rng.choice(np.array([np.nan, 1.0, 2.0, 4.0, 19.0, -1.0, 0.0]), size=n)
    codes, margins = E.component_status_array(
        est, nm, nsd, se, n_null=k, reference=ref, se_df=sdf)
    n_t = 0
    for i in range(n):
        ev = CE("NAS", est[i], nm[i], nsd[i], se=se[i], n_null=int(k[i]),
                reference=ref[i], se_df=sdf[i])
        st, m, _ = E.component_status(ev)
        assert E._STATUS_TO_CODE[st] == codes[i]
        if np.isfinite(m) or np.isfinite(margins[i]):
            assert m == margins[i]
        n_t += bool(np.isfinite(sdf[i]) and sdf[i] > 0)
    assert n_t > 1000
    # the t quantile is never more liberal than the normal one
    codes_z, _ = E.component_status_array(est, nm, nsd, se, n_null=k, reference=ref)
    ok = np.isfinite(sdf) & (sdf > 0)
    assert np.all(np.abs(codes[ok]) <= np.abs(codes_z[ok]))


def test_cutoff_and_alpha_validation():
    with pytest.raises(ValueError, match="delta"):
        E.normalize_cutoff((0.1, 0.25))
    with pytest.raises(ValueError, match="finite"):
        E.normalize_cutoff((np.inf, 0.1))
    with pytest.raises(ValueError, match="pair"):
        E.normalize_cutoff(0.25)
    assert E.normalize_cutoff((0.3, 0.3)) == (0.3, 0.3)  # delta = z allowed
    with pytest.raises(ValueError, match="alpha"):
        E.component_status(_ev("RAM", 1.0), alpha=0.7)
    with pytest.raises(ValueError):
        E.component_status_array(1.0, 0.0, 1.0, cutoff=(0.1, 0.2))
    with pytest.raises(ValueError, match="reference_scale"):
        CE("RAM", 1.0, reference_scale="raw")
    # stricter settings are honoured
    assert E.component_status(_ev("PDI", 0.5), cutoff=(0.6, 0.1))[0] == U
    assert E.component_status(_ev("PDI", 0.5, se=0.2), alpha=0.25)[0] == P
    assert E.component_status(_ev("PDI", 0.5, se=0.2), alpha=0.05)[0] == U


def test_node_sets_are_validated():
    assert CE("RAM", 1.0, nodes=[3, 1, 2]).nodes == (1, 2, 3)
    for bad in ([], [1, 1], [-1], [0.5], [True, False]):
        with pytest.raises(ValueError):
            CE("RAM", 1.0, nodes=bad)


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
    assert E.kleene_or([V.EXCLUDED, False]) == A
    assert E.kleene_and([V.MPC_CONSISTENT, V.UNDETERMINED]) == U
    with pytest.raises(ValueError):
        E.kleene_and(["maybe"])
    with pytest.raises(ValueError):
        E.kleene_and(["ATTRIBUTED"])  # v1 names are gone


def test_verdict_names_are_the_exclusion_rule():
    assert [v.value for v in V] == ["EXCLUDED", "MPC_CONSISTENT", "UNDETERMINED"]
    assert not hasattr(V, "ATTRIBUTED") and not hasattr(V, "NOT_ATTRIBUTED")


# --------------------------------------------------------------------------
# MPC verdict
# --------------------------------------------------------------------------
def test_verdict_consistent_excluded_missing_and_necessity_set():
    v = E.mpc_verdict(_all())
    assert v.verdict == V.MPC_CONSISTENT and v.reasons == []
    assert all(s == P for s in v.component_status.values())
    assert v.construct_values() == {p: pytest.approx(1.0) for p in E.PRINCIPLES}
    ev = _all()
    ev["NAS"] = [_ev("NAS", 0.0, se=0.02)]
    v = E.mpc_verdict(ev)
    assert v.verdict == V.EXCLUDED and v.reasons == ["ABSENT:NAS"]
    del ev["NAS"]
    v = E.mpc_verdict(ev)
    assert v.verdict == V.UNDETERMINED and v.reasons == ["MISSING:NAS"]
    # contested necessity: without NAS in N the same evidence is not excluded
    v = E.mpc_verdict(ev, necessity_set="RAM, pdi,IIM,SRPI")
    assert v.verdict == V.MPC_CONSISTENT
    assert v.necessity_set == ("RAM", "PDI", "IIM", "SRPI")


def test_veto_is_not_compensated_by_huge_other_components():
    ev = _all(estimate=1e6)
    ev["SRPI"] = [_ev("SRPI", 0.0, se=0.01)]
    assert E.mpc_verdict(ev).verdict == V.EXCLUDED


def test_missing_sampling_se_makes_the_verdict_undetermined():
    ev = _all()
    ev["IIM"] = [_ev("IIM", 1.0, se=0.0)]
    v = E.mpc_verdict(ev)
    assert v.verdict == V.UNDETERMINED and v.reasons == ["NO_SAMPLING_SE:IIM"]
    # ... but never hides a credible absence elsewhere (veto)
    ev["NAS"] = [_ev("NAS", 0.0, se=0.02)]
    assert E.mpc_verdict(ev).verdict == V.EXCLUDED


def test_reason_codes_are_stable_strings():
    ev = {
        "RAM": [_ev("RAM", 0.2)],
        "PDI": [_ev("PDI", 2.0, null_mean=np.nan)],
        "NAS": [_ev("NAS", np.nan, defined=False, reason="insufficient;shape")],
        "IIM": [_ev("IIM", np.nan, defined=False, reason="NOT_IMPLEMENTED",
                    channel="directional")],
        "SRPI": [_ev("SRPI", 0.9, se=0.0)],
    }
    v = E.mpc_verdict(ev)
    assert v.verdict == V.UNDETERMINED
    assert v.reasons == [
        "INCONCLUSIVE:RAM",
        "NO_NULL_CALIBRATION:PDI",
        "UNDEFINED:NAS:insufficient,shape",
        "NOT_IMPLEMENTED:IIM:directional",
        "NO_SAMPLING_SE:SRPI",
    ]
    ev["SRPI"] = [_ev("SRPI", 0.9, null_mean=2.0, reference=1.0)]
    assert E.mpc_verdict(ev).reasons[-1] == "INVALID_ANCHORS:SRPI"
    assert v.reason_string == ";".join(v.reasons)
    assert E.parse_reason("UNDEFINED:NAS:a:b") == ("UNDEFINED", "NAS", "a:b")
    assert E.parse_reason("BEARER_MISMATCH") == ("BEARER_MISMATCH", None, None)
    assert E.parse_reason("SOURCE_INCOHERENT:UNTESTED") == (
        "SOURCE_INCOHERENT", None, "UNTESTED")
    assert E.parse_reason("MISSING_CHANNEL:RAM:covert_neural") == (
        "MISSING_CHANNEL", "RAM", "covert_neural")
    assert E.verdict_from_reasons(v.reasons) == v.verdict
    assert E.verdict_from_reasons([]) == V.MPC_CONSISTENT
    assert E.verdict_from_reasons(["ABSENT:RAM", "NO_SAMPLING_SE:IIM"]) == V.EXCLUDED
    assert E.verdict_from_reasons(["SOURCE_INCOHERENT", "ABSENT:RAM"]) == (
        V.UNDETERMINED)
    json.dumps(v.to_dict())  # JSON-safe


def test_legacy_null_sd_thresholds_are_rejected():
    with pytest.raises(TypeError, match="construct-scale cutoffs"):
        E.mpc_verdict(_all(), z_present=1.645)
    with pytest.raises(TypeError, match="z_presnt"):
        E.mpc_verdict(_all(), z_presnt=2.0)
    with pytest.raises(ValueError, match="delta"):
        E.mpc_verdict({}, cutoffs=(0.1, 0.5))
    assert E.mpc_verdict(_all(), cutoffs=(0.5, 0.2), alpha=0.1).verdict == (
        V.MPC_CONSISTENT)
    v = E.mpc_verdict(_all(), cutoffs={"RAM": (0.99, 0.1)})
    assert v.reasons == ["INCONCLUSIVE:RAM"]


def test_channels_are_combined_by_disjunction():
    # covert responder: behavioural feedback channel credibly absent, covert
    # neural channel present -> RAM PRESENT
    ev = _all()
    ev["RAM"] = [
        _ev("RAM", 0.0, se=0.02, channel="behavioural_feedback"),
        _ev("RAM", 0.8, channel="covert_neural"),
    ]
    v = E.mpc_verdict(ev)
    assert v.verdict == V.MPC_CONSISTENT
    assert v.channels["RAM"] == {"behavioural_feedback": A, "covert_neural": P}
    assert v.margins["RAM"] == pytest.approx(0.8 - Z95 * 0.05 - 0.25)
    assert v.principle_assessment["RAM"].c == pytest.approx(0.8)
    # disconnected: exogenous channel absent, neural channel not implemented ->
    # UNDETERMINED, never EXCLUDED
    ev["RAM"][1] = _ev("RAM", np.nan, defined=False, reason="NOT_IMPLEMENTED",
                       channel="covert_neural")
    v = E.mpc_verdict(ev)
    assert v.verdict == V.UNDETERMINED
    assert v.reasons == ["NOT_IMPLEMENTED:RAM:covert_neural"]
    assert v.channel_reasons["RAM"] == {"covert_neural": "NOT_IMPLEMENTED"}
    # ABSENT only when every channel is ABSENT
    ev["RAM"][1] = _ev("RAM", 0.02, se=0.02, channel="covert_neural")
    assert E.mpc_verdict(ev).reasons == ["ABSENT:RAM"]


def test_declared_channel_without_evidence_is_missing_channel():
    proto = E.Protocol(channels={"RAM": ("behavioural_feedback", "covert_neural")})
    ev = {p: [_ev(p, 1.0, protocol_id=proto.protocol_id)] for p in E.PRINCIPLES}
    ev["RAM"] = [_ev("RAM", 0.0, se=0.02, channel="behavioural_feedback",
                     protocol_id=proto.protocol_id)]
    v = E.mpc_verdict(ev, proto)
    # the absent behavioural channel cannot exclude while covert_neural is
    # declared but unmeasured
    assert v.verdict == V.UNDETERMINED
    assert v.component_status["RAM"] == U
    assert v.reasons == ["MISSING_CHANNEL:RAM:covert_neural"]
    assert v.channels["RAM"] == {"behavioural_feedback": A, "covert_neural": U}
    assert v.channel_reasons["RAM"] == {"covert_neural": "MISSING_CHANNEL"}
    # evidence of an undeclared channel is ignored (reported)
    ev["RAM"].append(_ev("RAM", 1.0, channel="default",
                         protocol_id=proto.protocol_id))
    v = E.mpc_verdict(ev, proto)
    assert v.ignored_channels == {"RAM": ["default"]}
    assert v.verdict == V.UNDETERMINED
    # a principle with declared channels and no evidence at all
    del ev["RAM"]
    v = E.mpc_verdict(ev, proto)
    assert v.reasons == ["MISSING_CHANNEL:RAM:behavioural_feedback",
                         "MISSING_CHANNEL:RAM:covert_neural"]


def test_bearer_and_protocol_mismatch_force_undetermined():
    ev = _all()
    ev["IIM"] = [_ev("IIM", 1.0, bearer_id="sub-02/awake/run-1")]
    v = E.mpc_verdict(ev)
    assert v.verdict == V.UNDETERMINED and v.reasons == ["BEARER_MISMATCH"]
    # never EXCLUDED either, even with a credible absence
    ev["NAS"] = [_ev("NAS", 0.0, se=0.01)]
    v = E.mpc_verdict(ev)
    assert v.verdict == V.UNDETERMINED
    assert v.reasons == ["BEARER_MISMATCH", "ABSENT:NAS"]
    assert E.mpc_verdict(ev, require_same_bearer=False).verdict == V.EXCLUDED
    # a mismatching principle outside N is not part of the verdict
    ev["NAS"] = [_ev("NAS", 1.0)]
    nset = ("RAM", "PDI", "NAS", "SRPI")
    assert E.mpc_verdict(ev, necessity_set=nset).verdict == V.MPC_CONSISTENT
    # undeclared bearers do not clash
    ev["IIM"] = [_ev("IIM", 1.0, bearer_id=None)]
    assert E.mpc_verdict(ev).verdict == V.MPC_CONSISTENT
    ev["IIM"] = [_ev("IIM", 1.0, protocol_id="proto-B")]
    assert E.mpc_verdict(ev).reasons == ["PROTOCOL_MISMATCH"]
    assert E.mpc_verdict(ev, require_same_protocol=False).verdict == (
        V.MPC_CONSISTENT)
    # an undefined component keeps its declared bearer (missingness safety)
    ev = _all()
    ev["IIM"] = [_ev("IIM", np.nan, defined=False, reason="x", bearer_id="other")]
    assert E.mpc_verdict(ev).reasons[0] == "BEARER_MISMATCH"


def test_explicit_protocol_checks_the_evidence_protocol_id():
    proto = E.Protocol()
    ev = {p: [_ev(p, 1.0, protocol_id=proto.protocol_id)] for p in E.PRINCIPLES}
    v = E.mpc_verdict(ev, proto)
    assert v.verdict == V.MPC_CONSISTENT and v.protocol_hash == proto.hash
    ev["PDI"] = [_ev("PDI", 1.0, protocol_id=E.Protocol(alpha=0.1).protocol_id)]
    assert E.mpc_verdict(ev, proto).reasons == ["PROTOCOL_MISMATCH"]
    with pytest.raises(ValueError, match="Protocol"):
        E.mpc_verdict(ev, proto, necessity_set=("RAM",))
    with pytest.raises(ValueError, match="Protocol"):
        E.mpc_verdict(ev, proto, cutoffs=(0.3, 0.1))


def test_single_source_constraint_on_node_sets():
    def ev_nodes(ns_by_p, **kw):
        return {p: [_ev(p, 1.0, nodes=ns_by_p.get(p, (0, 1, 2, 3)), **kw)]
                for p in E.PRINCIPLES}

    same = ev_nodes({})
    assert E.mpc_verdict(same).verdict == V.MPC_CONSISTENT
    split = ev_nodes({"IIM": (4, 5)})
    v = E.mpc_verdict(split)
    assert v.verdict == V.UNDETERMINED and v.reasons == ["SOURCE_INCOHERENT:UNTESTED"]
    ok = {"dependent": True, "sets": [[0, 1, 2, 3], [4, 5]]}
    assert E.mpc_verdict(split, joint_dependence=ok).verdict == V.MPC_CONSISTENT
    bad = {"dependent": False, "sets": ok["sets"], "reason": "SOURCE_INCOHERENT"}
    assert E.mpc_verdict(split, joint_dependence=bad).reasons == ["SOURCE_INCOHERENT"]
    flagged = dict(bad, reason="OVERLAPPING_NODE_SETS")
    assert E.mpc_verdict(split, joint_dependence=flagged).reasons == [
        "SOURCE_INCOHERENT:OVERLAPPING_NODE_SETS"]
    # a test over other node sets does not cover this evidence
    other = {"dependent": True, "sets": [[0, 1, 2, 3], [4, 6]]}
    assert E.mpc_verdict(split, joint_dependence=other).reasons == [
        "SOURCE_INCOHERENT:UNTESTED"]
    # an undeclared node set (None) next to declared ones can never be covered
    mixed = ev_nodes({})
    mixed["IIM"] = [_ev("IIM", 1.0, nodes=None)]
    assert E.mpc_verdict(mixed, joint_dependence=ok).reasons == [
        "SOURCE_INCOHERENT:UNTESTED"]
    # node sets of principles outside N do not matter; other source rules
    assert E.mpc_verdict(split, necessity_set=("RAM", "NAS")).verdict == (
        V.MPC_CONSISTENT)
    for rule in ("same_bearer", "none"):
        proto = E.Protocol(source_rule=rule)
        ev = {p: [dataclass_replace(e, protocol_id=proto.protocol_id) for e in items]
              for p, items in split.items()}
        assert E.mpc_verdict(ev, proto).verdict == V.MPC_CONSISTENT
    assert E.mpc_verdict(split, joint_dependence=True).verdict == V.MPC_CONSISTENT


def dataclass_replace(ev, **kw):
    import dataclasses

    return dataclasses.replace(ev, **kw)


def test_declared_null_family_must_match():
    proto = E.Protocol(null_families={"NAS": "circular_shift"})
    ev = {p: [_ev(p, 1.0, protocol_id=proto.protocol_id)] for p in E.PRINCIPLES}
    ev["NAS"] = [_ev("NAS", 1.0, null_family="phase_randomize",
                     protocol_id=proto.protocol_id)]
    v = E.mpc_verdict(ev, proto)
    assert v.reasons == [
        "UNDEFINED:NAS:NULL_FAMILY_MISMATCH:circular_shift/phase_randomize"]
    ev["NAS"] = [_ev("NAS", 1.0, null_family="circular_shift",
                     protocol_id=proto.protocol_id)]
    assert E.mpc_verdict(ev, proto).verdict == V.MPC_CONSISTENT


def test_failed_bearer_coherence_forces_undetermined():
    ev = _all()
    assert E.mpc_verdict(ev, bearer_coherence=True).verdict == V.MPC_CONSISTENT
    v = E.mpc_verdict(ev, bearer_coherence={"coherent": False})
    assert v.verdict == V.UNDETERMINED
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
    assert E.mpc_verdict({p: _ev(p, 1.0) for p in E.PRINCIPLES}).verdict == (
        V.MPC_CONSISTENT)


# --------------------------------------------------------------------------
# protocol (V2-3)
# --------------------------------------------------------------------------
def _protocol():
    return E.Protocol(
        necessity_set=("RAM", "NAS", "IIM"),
        channels={"RAM": ["behavioural_feedback", "covert_neural"]},
        cutoffs={"IIM": (0.3, 0.05)},
        alpha=0.025,
        null_families={"NAS": "block_circular_shift", "IIM": "circular_shift"},
        reference={"kind": "external", "values": {"NAS": 0.2, "RAM:covert_neural": 1.5},
                   "se": {"NAS": 0.01}},
        source_rule="single_source",
        estimators={"NAS": {"mode": "capacity", "workspace_nodes": [0, 1, 2]},
                    "RAM": {"update": "prediction_error"}},
        bearer_nodes={"NAS": [0, 1, 2, 3, 4], "IIM": [5, 6, 7]},
        name="paper-1 preregistration draft",
    )


def test_protocol_json_round_trip_and_hash(tmp_path):
    proto = _protocol()
    d = proto.to_dict()
    assert d["schema"] == E.PROTOCOL_SCHEMA
    assert d["cutoffs"]["IIM"] == [0.3, 0.05] and d["cutoffs"]["RAM"] == [0.25, 0.1]
    assert d["bearer_nodes"] == {"NAS": [0, 1, 2, 3, 4], "IIM": [5, 6, 7]}
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    assert proto.hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert proto.protocol_id == "sha256:" + proto.hash
    path = tmp_path / "protocol.json"
    proto.to_json(path)
    again = E.Protocol.from_json(path)
    assert again == proto and again.hash == proto.hash
    assert E.resolve_protocol(str(path)) == proto
    assert E.resolve_protocol(d) == proto
    assert E.Protocol.from_dict(json.loads(json.dumps(d))).hash == proto.hash
    # defaults are materialised: equal content -> equal hash
    assert E.Protocol().hash == E.Protocol(
        cutoffs={p: (0.25, 0.1) for p in E.PRINCIPLES},
        necessity_set="SRPI,IIM,NAS,PDI,RAM").hash
    # any declared change changes the hash
    assert proto.replace(alpha=0.05).hash != proto.hash
    assert proto.replace(cutoffs={"IIM": (0.3, 0.06)}).hash != proto.hash
    assert proto.cutoff_for("IIM") == (0.3, 0.05)
    assert proto.channels_for("RAM") == ("behavioural_feedback", "covert_neural")
    assert proto.channels_for("PDI") is None
    opts = proto.estimator_options("NAS")
    opts["mode"] = "legacy"  # a copy: the protocol stays immutable
    assert proto.estimator_options("NAS")["mode"] == "capacity"
    with pytest.raises(Exception):
        proto.alpha = 0.5
    # read-only all the way down (review of stream E2): nested reference and
    # estimator values cannot be changed, so the hash cannot drift
    h = proto.hash
    with pytest.raises(TypeError):
        proto.reference["values"]["NAS"] = 9.0
    with pytest.raises((TypeError, AttributeError)):
        proto.estimators["NAS"]["workspace_nodes"].append(9)
    assert proto.hash == h
    assert proto.estimator_options("NAS")["workspace_nodes"] == [0, 1, 2]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"cutoffs": {"RAM": (0.1, 0.2)}}, "delta"),
        ({"cutoffs": {"PHI": (0.3, 0.1)}}, "unknown principle"),
        ({"channels": {"RAM": []}}, "channels"),
        ({"channels": {"RAM": ["a", "a"]}}, "channels"),
        ({"alpha": 0.0}, "alpha"),
        ({"source_rule": "principle_0"}, "source_rule"),
        ({"reference": {"kind": "median"}}, "reference kind"),
        ({"reference": {"kind": "external"}}, "values"),
        ({"reference": {"kind": "external", "values": {"NAS": np.nan}}}, "finite"),
        ({"bearer_nodes": {"NAS": [1, 1]}}, "duplicate"),
        ({"estimators": {"NAS": "capacity"}}, "object"),
        ({"null_families": {"NAS": ""}}, "null family"),
    ],
)
def test_protocol_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        E.Protocol(**kwargs)


def test_protocol_from_dict_is_strict():
    with pytest.raises(ValueError, match="unknown fields"):
        E.Protocol.from_dict({"necessity_set": ["RAM"], "z_present": 1.645})
    with pytest.raises(ValueError, match="schema"):
        E.Protocol.from_dict({"schema": "impact-mpc-protocol/1"})


# --------------------------------------------------------------------------
# applicability registry (V2-5)
# --------------------------------------------------------------------------
def _entry(**kw):
    entry = {
        "estimator": "compute_NAS:capacity",
        "version": "nas-v2-2026.09",
        "substrate": "eeg_like_forward",
        "grain": "*",
        "regime": {"T_min": 1000, "nodes_min": 8, "nodes_max": 64,
                   "fs_or_tr": {"min": 0.002, "max": 0.01}, "snr_min": 0.5},
        "evidence": {"run_id": "bench-2026-10-01-a", "null_false_present_rate": 0.04,
                     "recovery_slope": 0.8, "cross_talk": 0.03},
        "status": "validated",
    }
    entry.update(kw)
    return entry


def _regime(**kw):
    regime = {"n_time": 2000, "n_nodes": 16, "tr": 0.004, "snr": 1.0}
    regime.update(kw)
    return regime


def test_registry_v2_entry_criteria_and_matching(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema": E.REGISTRY_SCHEMA, "version": "2026-10",
                                "entries": [_entry()]}))
    reg = E.ApplicabilityRegistry.from_json(path)
    est = E.estimator_id("NAS", "capacity", mm.ESTIMATOR_VERSIONS["NAS"])
    assert est == "compute_NAS:capacity@nas-v2-2026.09"
    assert reg.entries[0].principle == "NAS"  # derived from the estimator name
    # an empirical substrate is covered by its forward-modelled validation
    assert reg.is_validated("NAS", est, "eeg", "aal116", **_regime()) == (True, None)
    assert reg.is_validated("NAS", est, "eeg_like_forward", None, **_regime())[0]
    assert reg.is_validated("NAS", est, "fmri", "x", **_regime()) == (False, "NO_ENTRY")
    # the version is pinned
    assert reg.is_validated("NAS", "compute_NAS:capacity@nas-v3", "eeg", "x",
                            **_regime()) == (False, "NO_ENTRY")
    assert reg.is_validated("NAS", "compute_NAS:capacity", "eeg", "x",
                            **_regime()) == (False, "NO_ENTRY")
    assert reg.is_validated("NAS", "compute_NAS:legacy@nas-v2-2026.09", "eeg", "x",
                            **_regime()) == (False, "NO_ENTRY")
    # the named regime bounds
    for bad, key in (
        (_regime(n_time=999), "T_min"),
        (_regime(n_nodes=4), "nodes_min"),
        (_regime(n_nodes=65), "nodes_max"),
        (_regime(tr=0.5), "fs_or_tr"),
        (_regime(snr=0.1), "snr_min"),
        ({k: v for k, v in _regime().items() if k != "snr"}, "snr_min"),
    ):
        assert reg.is_validated("NAS", est, "eeg", "x", **bad) == (
            False, f"REGIME_MISMATCH:{key}")
    assert reg.is_validated("NAS", None) == (False, "ESTIMATOR_UNDECLARED")
    d = reg.to_dict()
    assert d["entries"][0]["evidence"]["run_id"] == "bench-2026-10-01-a"
    assert d["entries"][0]["verified"] is True


@pytest.mark.parametrize(
    "change,match",
    [
        ({"evidence": {"null_false_present_rate": 0.04, "recovery_slope": 0.8}},
         "run_id"),
        ({"evidence": {"run_id": "r", "null_false_present_rate": 0.08,
                       "recovery_slope": 0.8}}, "false-PRESENT"),
        ({"evidence": {"run_id": "r", "null_false_present_rate": 0.04,
                       "recovery_slope": 0.0}}, "recovery slope"),
        ({"evidence": {"run_id": "r", "null_false_present_rate": 0.04,
                       "recovery_slope": 0.5, "recovery_monotone": False}},
         "monotone"),
        ({"version": "nas-v2-*"}, "pinned"),
        ({"substrate": "*"}, "substrate"),
        ({"substrate": "eeg"}, "forward-modelled"),
        ({"status": "certified"}, "status"),
        ({"status": "validated", "validated": False}, "contradicts"),
        ({"validated": "false"}, "true or false"),
        ({"regime": {"nodes_min": 10, "nodes_max": 5}}, "nodes_min"),
        ({"estimator": "my_estimator"}, "principle"),
    ],
)
def test_registry_rejects_entries_that_miss_the_criteria(change, match):
    with pytest.raises(ValueError, match=match):
        E.ApplicabilityRegistry.from_dict({"entries": [_entry(**change)]})


def test_registry_criteria_options():
    low = _entry(evidence={"run_id": "r", "null_false_present_rate": 0.0,
                           "recovery_slope": 0.4})
    # conservative estimators pass the default one-sided criterion ...
    assert E.ApplicabilityRegistry.from_dict({"entries": [low]}).entries
    # ... and fail the literal alpha +- 0.02 band on request
    with pytest.raises(ValueError, match="below"):
        E.ApplicabilityRegistry.from_dict(
            {"criteria": {"two_sided": True}, "entries": [low]})
    with pytest.raises(ValueError, match="cross-talk"):
        E.ApplicabilityRegistry.from_dict(
            {"criteria": {"cross_talk_max": 0.01}, "entries": [_entry()]})
    with pytest.raises(ValueError, match="unknown keys"):
        E.ApplicabilityRegistry.from_dict({"criteria": {"tolerance": 1}, "entries": []})
    with pytest.raises(ValueError, match="schema"):
        E.ApplicabilityRegistry.from_dict({"schema": "x", "entries": []})


def test_registry_statuses_and_legacy_entries():
    payload = {"entries": [
        _entry(),
        _entry(estimator="compute_NAS:legacy", status="provisional", evidence={}),
        _entry(version="nas-v1", status="not_validated", evidence={}),
    ]}
    reg = E.ApplicabilityRegistry.from_dict(payload)
    assert reg.is_validated("NAS", "compute_NAS:legacy@nas-v2-2026.09", "eeg", "x",
                            **_regime()) == (False, "MARKED_NOT_VALIDATED:provisional")
    assert reg.is_validated("NAS", "compute_NAS:capacity@nas-v1", "eeg", "x",
                            **_regime()) == (False, "MARKED_NOT_VALIDATED")
    legacy = [{"principle": "NAS", "estimator": "compute_NAS:*"},
              {"principle": "IIM", "estimator": "compute_IIM:*", "validated": False}]
    with pytest.raises(ValueError, match="entry criteria"):
        E.ApplicabilityRegistry.from_dict(legacy)  # strict by default
    reg = E.ApplicabilityRegistry.from_dict(legacy, strict=False)
    assert reg.entries[0].verified is False
    assert reg.is_validated("NAS", "compute_NAS:legacy@nas-v2", "fmri") == (True, None)
    assert reg.is_validated("IIM", "compute_IIM:x@v", "fmri") == (
        False, "MARKED_NOT_VALIDATED")


def test_unregistered_estimator_is_estimator_not_validated():
    reg = E.ApplicabilityRegistry.from_dict({"entries": [_entry()]})
    est = "compute_NAS:capacity@nas-v2-2026.09"
    ev = {"NAS": [_ev("NAS", 1.0, estimator=est, substrate="eeg")],
          "IIM": [_ev("IIM", 1.0, estimator="compute_IIM:bidirectional@iim-v4",
                      substrate="eeg")]}
    v = E.mpc_verdict(ev, necessity_set=("NAS", "IIM"), registry=reg,
                      regime=_regime())
    assert v.verdict == V.UNDETERMINED
    assert v.reasons == [
        "ESTIMATOR_NOT_VALIDATED:IIM:compute_IIM:bidirectional@iim-v4"]
    assert v.channel_reasons["IIM"]["default"] == "ESTIMATOR_NOT_VALIDATED:NO_ENTRY"
    # per-item regime (e.g. the component's bearer size) overrides the run's
    ev["NAS"] = [_ev("NAS", 1.0, estimator=est, substrate="eeg", regime={"n_nodes": 4})]
    v = E.mpc_verdict(ev, necessity_set=("NAS",), registry=reg, regime=_regime())
    assert v.reasons == [f"ESTIMATOR_NOT_VALIDATED:NAS:{est}"]
    assert v.channel_reasons["NAS"]["default"].endswith("REGIME_MISMATCH:nodes_min")


def test_estimator_versions_are_declared_for_every_principle():
    assert set(mm.ESTIMATOR_VERSIONS) == set(E.PRINCIPLES)
    assert mm.ESTIMATOR_VERSIONS["IIM"] == mm.IIM_ALGORITHM_VERSION
    assert E.split_estimator("compute_IIM:directional@iim-v4-2026.09") == (
        "compute_IIM:directional", "iim-v4-2026.09")
    assert E.split_estimator("compute_IIM:v5") == ("compute_IIM:v5", None)
    assert E.split_estimator(None) == (None, None)


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
    out = E.verdict_stability([V.MPC_CONSISTENT] * 7 + [V.UNDETERMINED] * 3,
                              reference=V.MPC_CONSISTENT)
    assert out["modal_verdict"] == "MPC_CONSISTENT"
    assert out["flip_rate"] == pytest.approx(0.3)
    assert out["reference_flip_rate"] == pytest.approx(0.3)
    assert out["counts"] == {"EXCLUDED": 0, "MPC_CONSISTENT": 7, "UNDETERMINED": 3}
    # a tie for the top count has no determinate mode
    tie = E.verdict_stability(["MPC_CONSISTENT", "EXCLUDED"])
    assert tie["modal_verdict"] == "UNDETERMINED" and tie["flip_rate"] == 1.0
    mv = E.mpc_verdict(_all())
    assert E.verdict_stability([mv, mv])["flip_rate"] == 0.0
    assert E.verdict_stability([])["modal_verdict"] is None
    with pytest.raises(ValueError):
        E.verdict_stability(["ATTRIBUTED"])


# --------------------------------------------------------------------------
# single-source constraint: joint dependence (V2-4)
# --------------------------------------------------------------------------
def _modules(seed=0, n_time=800, couplings=(0.8, 0.8)):
    """Three 3-node AR(1) modules; module k+1 is driven by module k (lag 1)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((9, n_time))
    for t in range(1, n_time):
        x[:, t] += 0.5 * x[:, t - 1]
        x[3:6, t] += couplings[0] * x[0, t - 1]
        x[6:9, t] += couplings[1] * x[3, t - 1]
    return x


SETS = {"RAM": [0, 1, 2], "NAS": [3, 4, 5], "IIM": [6, 7, 8]}


def test_joint_dependence_known_answers():
    coupled = E.joint_dependence(_modules(), SETS, n_surrogates=99, seed=0)
    assert coupled["dependent"] and coupled["reason"] is None
    assert coupled["p"] == pytest.approx(0.01) and coupled["z"] > 5
    assert coupled["sets"] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert [b["significant"] for b in coupled["per_block"]] == [True] * 3
    indep = E.joint_dependence(_modules(couplings=(0, 0)), SETS, n_surrogates=99)
    assert not indep["dependent"] and indep["reason"] == "SOURCE_INCOHERENT"
    assert indep["p"] > 0.05 and indep["tc_bits"] < coupled["tc_bits"]
    # deterministic given the seed
    again = E.joint_dependence(_modules(), SETS, n_surrogates=99, seed=0)
    assert again["per_block"] == coupled["per_block"] and again["p"] == coupled["p"]
    # lagged dependence alone (lag 0 would miss a pure 1-step drive)
    lag0 = E.joint_dependence(_modules(), SETS, lag=0, n_surrogates=49)
    assert lag0["tc_bits"] < coupled["tc_bits"]


def test_joint_dependence_each_criterion_rejects_a_partial_patchwork():
    # modules RAM and NAS coupled, IIM an independent patch
    x = _modules(couplings=(0.8, 0.0))
    total = E.joint_dependence(x, SETS, n_surrogates=99, criterion="total")
    each = E.joint_dependence(x, SETS, n_surrogates=99, criterion="each")
    assert total["dependent"]
    assert not each["dependent"] and each["reason"] == "SOURCE_INCOHERENT"
    assert [b["significant"] for b in each["per_block"]] == [True, True, False]


def test_joint_dependence_with_too_few_surrogates_is_not_a_finding():
    """Regression (review of stream E2): with K surrogates the smallest p is
    1 / (K + 1); for K < 19 (alpha 0.05) a strongly coupled system (z ~ 40)
    was reported as SOURCE_INCOHERENT as if tested. It is now
    INSUFFICIENT_SURROGATES (still not dependent: the verdict stays
    UNDETERMINED, with an honest reason)."""
    x = _modules()
    few = E.joint_dependence(x, SETS, n_surrogates=10)
    assert few["p"] == pytest.approx(1 / 11) and few["z"] > 5
    assert not few["dependent"] and few["reason"] == "INSUFFICIENT_SURROGATES"
    assert "INSUFFICIENT_SURROGATES" in few["flags"] and few["min_surrogates"] == 19
    ev = {p: [_ev(p, 1.0, nodes=SETS[p])] for p in SETS}
    v = E.mpc_verdict(ev, necessity_set=list(SETS), joint_dependence=few)
    assert v.reasons == ["SOURCE_INCOHERENT:INSUFFICIENT_SURROGATES"]
    # K = 19 reaches alpha = 0.05 exactly
    ok = E.joint_dependence(x, SETS, n_surrogates=19)
    assert ok["dependent"] and ok["p"] == pytest.approx(0.05) and ok["reason"] is None
    # Holm over 3 blocks ("each") needs 1 / (K + 1) <= alpha / 3: K >= 59
    each = E.joint_dependence(x, SETS, n_surrogates=40, criterion="each")
    assert each["reason"] == "INSUFFICIENT_SURROGATES" and each["min_surrogates"] == 59
    assert E.joint_dependence(x, SETS, n_surrogates=59, criterion="each")["dependent"]
    # an independent system with enough surrogates is still a finding
    indep = E.joint_dependence(_modules(couplings=(0, 0)), SETS, n_surrogates=19)
    assert indep["reason"] == "SOURCE_INCOHERENT"


def test_joint_dependence_overlapping_and_degenerate_sets():
    x = _modules(couplings=(0, 0))
    # identical sets are one source; a single set is trivially coherent
    one = E.joint_dependence(x, {"RAM": [0, 1], "NAS": [1, 0]}, n_surrogates=9)
    assert one["dependent"] and one["flags"] == ["SINGLE_NODE_SET"]
    assert one["set_names"] == ["RAM|NAS"]
    # nested / overlapping sets are tested on their disjoint parts (atoms)
    nested = E.joint_dependence(x, {"IIM": [0, 1, 2], "NAS": [0, 1, 2, 3, 4, 5]},
                                n_surrogates=49)
    assert "OVERLAPPING_NODE_SETS" in nested["flags"]
    assert [b["nodes"] for b in nested["blocks"]] == [[0, 1, 2], [3, 4, 5]]
    assert not nested["dependent"]  # the independent rest of NAS's bearer
    flagged = E.joint_dependence(x, {"a": [0, 1], "b": [1, 2]}, overlap="flag")
    assert not flagged["dependent"] and flagged["reason"] == "OVERLAPPING_NODE_SETS"
    x2 = x.copy()
    x2[3:6] = 1.0
    deg = E.joint_dependence(x2, SETS, n_surrogates=9)
    assert not deg["dependent"] and deg["reason"] == "DEGENERATE_NODE_SET:NAS"
    assert E.joint_dependence(x, SETS, n_surrogates=0)["reason"] == "NO_SURROGATES"
    with pytest.raises(ValueError, match="out of range"):
        E.joint_dependence(x, {"a": [0], "b": [99]})
    with pytest.raises(ValueError, match="finite"):
        bad = x.copy()
        bad[0, 0] = np.nan
        E.joint_dependence(bad, SETS)
    with pytest.raises(ValueError, match="criterion"):
        E.joint_dependence(x, SETS, criterion="max")


def test_joint_dependence_result_gates_the_verdict():
    x = _modules()
    res = E.joint_dependence(x, SETS, n_surrogates=49)
    ev = {p: [_ev(p, 1.0, nodes=SETS[p])] for p in SETS}
    v = E.mpc_verdict(ev, necessity_set=list(SETS), joint_dependence=res)
    assert v.verdict == V.MPC_CONSISTENT
    res0 = E.joint_dependence(_modules(couplings=(0, 0)), SETS, n_surrogates=49)
    v = E.mpc_verdict(ev, necessity_set=list(SETS), joint_dependence=res0)
    assert v.verdict == V.UNDETERMINED and v.reasons == ["SOURCE_INCOHERENT"]


# --------------------------------------------------------------------------
# legacy pairwise bearer coherence
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
    res_ov = E.bearer_coherence(indep, [[0, 1, 2], [2, 3]], n_surrogates=99, seed=0)
    assert res_ov["coherent"] and res_ov["set_names"] == ["set0", "set1"]
    again = E.bearer_coherence(coupled, {"A": [0, 1, 2], "B": [3, 4, 5]},
                               n_surrogates=99, seed=0)
    assert again["pairs"] == res["pairs"]
    assert E.bearer_coherence(coupled, {"A": [0], "B": [3]}, lag=0,
                              n_surrogates=49)["coherent"]
    v = E.mpc_verdict(_all(), bearer_coherence=res0)
    assert v.verdict == V.UNDETERMINED


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


# --------------------------------------------------------------------------
# numerical stability of the degree (review regressions of stream I1)
# --------------------------------------------------------------------------
def _log_sum_exp_power_mean(x, w, p):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float) / np.sum(w)
    a = np.log(w) + p * np.log(x)
    m = a.max()
    return math.exp((m + math.log(np.exp(a - m).sum())) / p)


def test_degree_is_stable_for_extreme_exponents():
    c = [0.1, 0.2, 0.3]
    assert E.degree(c, p=1000.0) == pytest.approx(0.3, abs=1e-3)
    assert E.degree(c, p=-500.0) == pytest.approx(0.1, abs=1e-3)
    rng = np.random.default_rng(3)
    for _ in range(300):
        x = rng.uniform(1e-3, 1.0, 5)
        w = rng.uniform(0.1, 1.0, 5)
        for p in (-3000.0, -300.0, -2.0, -0.3, 0.4, 3.0, 300.0, 3000.0):
            assert E.degree(x, w, p=p) == pytest.approx(
                _log_sum_exp_power_mean(x, w, p), rel=1e-12)
    assert E.degree([0.0, 0.5], p=-1000.0) == 0.0
    assert E.degree([0.0, 0.5], p=1000.0) == pytest.approx(0.5 * 0.5 ** 1e-3)
    assert E.degree([0.0, 0.0], p=3.0) == 0.0


def test_degree_interval_delta_uses_the_chain_rule_through_floor_and_cap():
    c, se = {"a": -0.3, "b": 0.8}, [0.1, 0.2]
    assert E.degree_interval(c, se, p=1.0)["se"] == pytest.approx(0.5 * 0.2)
    assert E.degree_interval(c, se, p=-math.inf)["se"] == 0.0
    assert E.degree_interval(c, se, p=0.5)["se"] == pytest.approx(0.05)
    rng = np.random.default_rng(11)
    for p in (-60.0, -3.0, 0.0, 0.5, 2.0, 60.0):
        x = rng.uniform(0.05, 0.95, 4)
        h, s = 1e-7, 1e-3
        g = [(E.degree(x + h * e, p=p) - E.degree(x - h * e, p=p)) / (2 * h)
             for e in np.eye(4)]
        out = E.degree_interval(list(x), [s] * 4, p=p)
        assert out["se"] == pytest.approx(s * math.sqrt(sum(v * v for v in g)),
                                          rel=1e-5)
