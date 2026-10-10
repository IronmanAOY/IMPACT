# -*- coding: utf-8 -*-
"""
The status rule ``tost-v2`` against hand-computed cases: TOST decisions and
the precedence of the UNDEFINED reasons, the NAS intersection-union and
union, channel combination, the concordance route, the SE-method contract
(``se_df`` read from the record), the IIM rank gate, registry licensing per
status direction, the SE reversion, exact values, the amplitude scale, the
agreement of the three entry points, UNDEFINED never ABSENT, and the
testability gating of the precision table.

Hand computations use the one-sided Student-t quantiles
t(9; 0.95) = 1.833, t(9; 0.975) = 2.262, t(9; 0.99) = 2.821,
t(9; 0.995) = 3.250, t(12; 0.99) = 2.681 and z(0.99) = 2.326.
All components below sit on the excess scale with reference 1 and an
analytic null mean 0, so c = estimate, se_c = se and df = se_df.
"""
import dataclasses
import math

import numpy as np
import pytest

from impact_pipeline import evidence as v1
from impact_pipeline import evidence_v2 as E
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import testability as T

P5 = ("RAM", "PDI", "NAS", "IIM", "SRPI")
REF = {"kind": "external", "scale": "excess",
       "values": {"RAM": 1.0, "PDI": 1.0, "IIM": 1.0, "SRPI": 1.0,
                  "NAS:receive": 1.0, "NAS:return": 1.0}}
CELL = {"principle": "PDI", "substrate": "family_a", "observation_stage": "source",
        "view": "source", "bearer": "non_workspace", "absent": True, "present": False}
PROTO = E.ProtocolV3(reference=REF, directions={"NAS": ["receive", "return"]},
                     rank_gates={"IIM": 0.05}, concordance_route=[CELL])
COHORT = E.ProtocolV3(reference={"kind": "cohort_high_state", "session": "awake"})
U, PRESENT, ABSENT = "UNDEFINED", "PRESENT", "ABSENT"


def item(principle="RAM", c=0.0, se=0.03, se_df=9.0, **kw):
    base = dict(principle=principle, estimate=c, null_mean=0.0, null_sd=0.0,
                n_null=0, se=se, se_df=se_df, reference=1.0,
                reference_scale="excess")
    base.update(kw)
    return E.ComponentEvidenceV2(**base)


def nas(c_r, se_r, c_b, se_b, se_df=9.0):
    return [item("NAS", c_r, se_r, se_df, direction="receive"),
            item("NAS", c_b, se_b, se_df, direction="return")]


def status_of(a):
    return a.status.value, a.reason, R.PRESENT_NOT_REACHABLE in a.flags


# --------------------------------------------------------------------------
# the rule block and the quantiles
# --------------------------------------------------------------------------
def test_status_rule_block_is_the_designed_one():
    rule = E.StatusRule()
    assert rule.to_dict() == {
        "version": "tost-v2", "alpha": 0.05, "alpha_absent": 0.01,
        "absent_test": "tost", "null_violation": True,
        "present_reachability_flag": True,
    }
    assert rule.alpha_absent == pytest.approx(rule.alpha / E.N_DECL, rel=1e-15)
    assert E.StatusRule.from_dict(rule.to_dict()) == rule


@pytest.mark.parametrize("bad", [
    {"version": "tost-v1"}, {"alpha_absent": 0.06}, {"alpha_absent": 0.0},
    {"absent_test": "one_sided"}, {"null_violation": 1},
    {"present_reachability_flag": "yes"}, {"alpha": 0.6},
])
def test_status_rule_block_is_validated(bad):
    with pytest.raises(ValueError):
        E.StatusRule(**bad)


def test_status_rule_block_from_dict_needs_every_key():
    d = E.StatusRule().to_dict()
    with pytest.raises(ValueError):
        E.StatusRule.from_dict({k: v for k, v in d.items() if k != "null_violation"})
    with pytest.raises(ValueError):
        E.StatusRule.from_dict({**d, "extra": 1})


def test_quantiles_match_the_t_table():
    assert E.quantile(0.05, 9) == pytest.approx(1.833, abs=5e-4)
    assert E.quantile(0.025, 9) == pytest.approx(2.262, abs=5e-4)
    assert E.quantile(0.01, 9) == pytest.approx(2.821, abs=5e-4)
    assert E.quantile(0.005, 9) == pytest.approx(3.250, abs=5e-4)
    assert E.quantile(0.01, 12) == pytest.approx(2.681, abs=5e-4)
    assert E.quantile(0.01, math.inf) == pytest.approx(2.326, abs=5e-4)
    assert E.quantile(0.01, float("nan")) == E.quantile(0.01)  # missing df: normal


@pytest.mark.parametrize("df, members, s_a", [
    (9, 1, 0.035), (12, 1, 0.037), (19, 1, 0.039), (199, 1, 0.043),
    (9, 2, 0.031), (19, 2, 0.035),
])
def test_implied_precision_of_the_design(df, members, s_a):
    """s_A = delta / q_A (preregistration v2, section 3.4), NAS per direction
    at alpha_A / 2."""
    assert round(T.absent_precision(df, members=members), 3) == s_a


def test_member_levels():
    rule = E.StatusRule()
    assert E.member_levels(rule) == (0.05, 0.01)
    assert E.member_levels(rule, n_directions=2) == (0.05, 0.005)
    assert E.member_levels(rule, n_channels=3) == (0.05 / 3, 0.01)
    with pytest.raises(ValueError):
        E.member_levels(rule, 0)


# --------------------------------------------------------------------------
# TOST decisions and reasons, hand-computed (df 9)
# --------------------------------------------------------------------------
CASES = [
    # c, se, status, reason, PRESENT_NOT_REACHABLE   (hand computation)
    (0.00, 0.030, ABSENT, None, False),   # c -+ 2.821*0.03 = -+0.085, inside -+0.1
    (0.00, 0.040, U, "ABSENT_NOT_REACHABLE", False),  # 2.821*0.04 = 0.113 >= 0.1
    (0.05, 0.020, U, "INCONCLUSIVE", False),  # upper 0.106 >= 0.1, 0.056 < 0.1
    (-0.05, 0.020, U, "INCONCLUSIVE", False),  # lower -0.106 <= -0.1
    (-0.30, 0.050, U, "NULL_MODEL_VIOLATED", False),  # -0.3+1.833*0.05 = -0.208
    (-0.15, 0.050, U, "ABSENT_NOT_REACHABLE", False),  # -0.058 > -0.1; 0.141 >= 0.1
    (0.60, 0.100, PRESENT, None, False),  # 0.6-0.183 = 0.417 > 0.25
    (0.40, 0.050, PRESENT, None, False),  # 0.308 > 0.25
    (0.30, 0.050, U, "ABSENT_NOT_REACHABLE", False),  # 0.208 <= 0.25
    (0.12, 0.010, U, "INCONCLUSIVE", False),  # upper 0.148; 0.028 < 0.1
    (0.40, 0.500, U, "ABSENT_NOT_REACHABLE", True),  # 1.833*0.5 = 0.917 >= 0.75
    (2.00, 0.500, PRESENT, None, False),  # 2-0.917 = 1.083 > 0.25
    (-0.50, 0.500, U, "ABSENT_NOT_REACHABLE", True),  # upper_P 0.417 > -0.1
    (-2.00, 0.500, U, "NULL_MODEL_VIOLATED", True),  # upper_P -1.083 < -0.1
]


@pytest.mark.parametrize("c, se, status, reason, pnr", CASES)
def test_tost_decisions_and_reasons_hand_computed(c, se, status, reason, pnr):
    a = E.assess_item(item("RAM", c, se), PROTO)
    assert status_of(a) == (status, reason, pnr)
    assert a.route == "tost"
    arr = E.status_c(c, se, 9)
    assert (arr.status.item(), arr.reason.item(), bool(arr.present_not_reachable)) == (
        status, reason, pnr)


def test_absence_is_two_sided_where_v1_was_one_sided():
    """c = -0.30 +- 0.05 was ABSENT under v1 (upper bound below delta); v2
    calls it a null-model violation, UNDEFINED."""
    ev = item("RAM", -0.30, 0.05)
    old = v1.component_assessment(ev)
    assert old.status is v1.ComponentStatus.ABSENT
    assert E.assess_item(ev, PROTO).reason == R.NULL_MODEL_VIOLATED


@pytest.mark.parametrize("se_df, status, reason", [
    (9.0, U, "ABSENT_NOT_REACHABLE"),  # 2.821*0.036 = 0.1016 >= 0.1
    (12.0, ABSENT, None),  # 2.681*0.036 = 0.0965 < 0.1
    (None, ABSENT, None),  # normal quantile: 2.326*0.036 = 0.0838
])
def test_degrees_of_freedom_enter_the_tost(se_df, status, reason):
    a = E.assess_item(item("RAM", 0.0, 0.036, se_df), PROTO)
    assert (a.status.value, a.reason) == (status, reason)


def test_null_violation_switch():
    rule = E.StatusRule(null_violation=False)
    a = E.assess_item(item("RAM", -0.30, 0.05), PROTO, status_rule=rule)
    assert a.reason == R.ABSENT_NOT_REACHABLE  # next reason in order
    rule = E.StatusRule(present_reachability_flag=False)
    a = E.assess_item(item("RAM", 0.40, 0.50), PROTO, status_rule=rule)
    assert a.flags == ()


@pytest.mark.parametrize("c, status, reason", [
    (0.0999, ABSENT, None), (0.10, U, "INCONCLUSIVE"), (-0.10, U, "INCONCLUSIVE"),
    (-0.1001, U, "NULL_MODEL_VIOLATED"), (0.25, U, "INCONCLUSIVE"),
    (0.2501, PRESENT, None), (-0.0999, ABSENT, None),
])
def test_exact_values_use_strict_inequalities(c, status, reason):
    """Exact computations: PRESENT iff c > z, ABSENT iff |c| < delta."""
    a = E.assess_item(item("IIM", c, 0.0, None, exact=True, se_method="exact"), PROTO)
    assert (a.status.value, a.reason, a.route) == (status, reason, "exact")
    assert a.flags == ()


@pytest.mark.parametrize("c, status, reason", [
    (0.26, PRESENT, None), (0.09, ABSENT, None), (-0.09, ABSENT, None),
    (-0.11, U, "NULL_MODEL_VIOLATED"), (0.20, U, "INCONCLUSIVE"), (1.0, PRESENT, None),
])
def test_exact_values_ignore_the_null_and_reference_uncertainty(c, status, reason):
    """The exact route decides on c alone (se_c = 0) also when the null mean
    is Monte-Carlo and the protocol's anchor has an SE; the two parts stay
    reported. The v1 arithmetic would give se_c > 0 here (0.26 would not be
    PRESENT, 0.09 not ABSENT). The array entry point agrees."""
    proto = E.ProtocolV3(reference={**REF, "se": {"IIM": 0.2}},
                         directions={"NAS": ["receive", "return"]},
                         rank_gates={"IIM": 0.05})
    ev = item("IIM", c, 0.0, None, exact=True, se_method="exact", n_null=19,
              null_sd=0.3)
    assert v1.component_assessment(ev, cutoff=(0.25, 0.1)).se > 0
    a = E.assess_item(ev, proto)
    assert (a.status.value, a.reason, a.route) == (status, reason, "exact")
    assert a.se == 0.0 and a.se_null > 0 and a.se_reference > 0
    assert a.lower == a.upper == a.lower_absent == a.upper_absent == a.c == c
    assert a.flags == ()
    arr = E.assess_array(c, 0.0, 0.3, 0.0, n_null=19, reference=1.0, reference_se=0.2,
                         reference_scale="excess", exact=True)
    assert (arr.status.item(), arr.reason.item(), float(arr.se)) == (
        status, reason, 0.0)
    sc = E.status_c(a.c, a.se, a.df, exact=True)
    assert (sc.status.item(), sc.reason.item()) == (status, reason)


# --------------------------------------------------------------------------
# reason precedence
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw, reason", [
    (dict(defined=False), "NOT_DEFINED"),
    (dict(defined=False, reason="SAMPLING_UNRESOLVED"), "SAMPLING_UNRESOLVED"),
    (dict(defined=False, reason="ESTIMATOR_ERROR:ValueError"),
     "ESTIMATOR_ERROR:ValueError"),
    (dict(defined=False, reason="insufficient_updates"),
     "NOT_DEFINED:insufficient_updates"),
    (dict(defined=False, reason="a;b"), "NOT_DEFINED:a,b"),
    (dict(defined=False, reason="NOT_IMPLEMENTED"), "NOT_IMPLEMENTED:default"),
    (dict(defined=False, reason="NOT_IMPLEMENTED:endogenous"),
     "NOT_IMPLEMENTED:endogenous"),
    (dict(defined=False, reason="ABSENT"), "NOT_DEFINED:ABSENT"),
    (dict(c=float("nan")), "NON_FINITE_ESTIMATE"),
    (dict(null_mean=float("nan")), "NO_NULL_CALIBRATION"),
    (dict(n_null=1.5), "DEGENERATE_NULL"),
    (dict(n_null=5, null_sd=float("nan")), "DEGENERATE_NULL"),
    (dict(reference=None), "INVALID_ANCHORS"),
    (dict(reference=-1.0), "INVALID_ANCHORS"),
    (dict(se=-0.1), "INVALID_SE"),
    (dict(se_df=0.0), "INVALID_SE"),
    (dict(reference_se=-1.0), "INVALID_SE"),
    (dict(se_method="bogus"), "INVALID_SE"),
    (dict(se=0.0), "NO_SAMPLING_SE"),
    (dict(se=float("nan")), "NO_SAMPLING_SE"),
    # earlier checks win
    (dict(c=float("nan"), reference=None, se=-1.0), "NON_FINITE_ESTIMATE"),
    (dict(reference=None, se=-1.0), "INVALID_ANCHORS"),
    (dict(se=-1.0, se_method="bogus"), "INVALID_SE"),
    (dict(se=0.0, se_method="bogus"), "INVALID_SE"),
])
def test_validity_reasons_in_order(kw, reason):
    a = E.assess_item(item(**kw), COHORT)  # cohort reference: the item's own anchor
    assert (a.status.value, a.reason) == (U, reason)
    assert R.status_for_reason(a.reason) == "UNDEFINED"


def test_check_order_by_peeling_one_failure_at_a_time():
    """An item that fails every check carries the first reason of
    CHECK_ORDER; repairing the failures one by one walks through the order."""
    proto = E.ProtocolV3(
        reference={"kind": "cohort_high_state", "session": "awake"},
        null_families={"RAM": "trial_circular_shift"},
        estimator_versions={"RAM": "ram-v3-2026.10"})
    kw = dict(estimator="compute_RAM:x@ram-v2-2026.09", null_family="onset_jitter",
              defined=False, c=float("nan"), null_mean=float("nan"), n_null=1.5,
              reference=None, se=-1.0)
    registry = StubRegistry("no", "no")
    repairs = [
        ("estimator", "compute_RAM:pe_readout@ram-v3-2026.10"), ("registry", None),
        ("null_family", "trial_circular_shift"), ("defined", True), ("c", 0.0),
        ("null_mean", 0.0), ("n_null", 0), ("reference", 1.0), ("se", 0.0),
        ("se", 0.03),
    ]
    assert len(repairs) == len(E.CHECK_ORDER)
    for (name, code), (key, value) in zip(E.CHECK_ORDER, repairs):
        a = E.assess_item(item(**kw), proto, registry=registry)
        assert R.parse_reason(a.reason)[0] == code, name
        if key == "registry":
            registry = StubRegistry("yes", "yes")
        else:
            kw[key] = value
    assert E.assess_item(item(**kw), proto, registry=registry).status.value == ABSENT


def test_protocol_checks_precede_the_item_checks():
    proto = E.ProtocolV3(
        reference={"kind": "cohort_high_state", "session": "awake"},
        null_families={"RAM": "trial_circular_shift"},
        estimator_versions={"RAM": "ram-v3-2026.10"})
    good = dict(estimator="compute_RAM:pe_readout@ram-v3-2026.10",
                null_family="trial_circular_shift")
    assert E.assess_item(item(**good), proto).status.value == ABSENT
    a = E.assess_item(item(**{**good, "null_family": "onset_jitter"}), proto)
    assert a.reason == "NULL_FAMILY_MISMATCH:trial_circular_shift/onset_jitter"
    a = E.assess_item(item(**{**good, "estimator": "compute_RAM:x@ram-v2-2026.09",
                              "null_family": "onset_jitter", "c": float("nan")}), proto)
    assert a.reason == "ESTIMATOR_NOT_VALIDATED:not_admitted"
    a = E.assess_item(item(**{**good, "estimator": None}), proto)
    assert a.reason == "ESTIMATOR_NOT_VALIDATED:not_admitted"


# --------------------------------------------------------------------------
# NAS: intersection-union PRESENT, union ABSENT at alpha_A / 2
# --------------------------------------------------------------------------
NAS_CASES = [
    # c_R, se_R, c_B, se_B, status, reason, flag     (q_A/2 = 3.250, q_P = 1.833)
    (0.01, 0.02, 0.50, 0.05, ABSENT, None, False),  # R: 0.01 -+ 0.065 inside
    (0.01, 0.03, 0.50, 0.05, U, "INCONCLUSIVE", False),  # R upper 0.1075; 0.0975 < 0.1
    (0.60, 0.05, 0.50, 0.05, PRESENT, None, False),  # 0.508 and 0.408 > 0.25
    (0.60, 0.05, 0.30, 0.05, U, "ABSENT_NOT_REACHABLE", False),  # B 0.208; both 0.1625
    (0.60, 0.05, -0.30, 0.05, U, "NULL_MODEL_VIOLATED", False),  # B upper_P -0.208
    (0.00, 0.02, 0.00, 0.02, ABSENT, None, False),
    (0.60, 0.50, 0.50, 0.05, U, "ABSENT_NOT_REACHABLE", True),  # R: 1.833*0.5 >= 0.75
]


@pytest.mark.parametrize("c_r, se_r, c_b, se_b, status, reason, flag", NAS_CASES)
def test_nas_intersection_union_and_union(c_r, se_r, c_b, se_b, status, reason, flag):
    pa = E.assess_principle("NAS", nas(c_r, se_r, c_b, se_b), PROTO)
    assert (pa.status.value, pa.reason, R.PRESENT_NOT_REACHABLE in pa.flags) == (
        status, reason, flag)
    rec = pa.record_fields()
    assert rec["c"] == min(c_r, c_b)
    assert (rec["c_R"], rec["c_B"]) == (c_r, c_b)
    members = pa.deciding.members
    assert [m.alpha_absent for m in members] == [0.005, 0.005]
    assert [m.alpha for m in members] == [0.05, 0.05]
    arr = E.status_c_directional([[c_r, c_b]], [[se_r, se_b]], 9)
    assert (arr.status[0], arr.reason[0], bool(arr.present_not_reachable[0])) == (
        status, reason, flag)
    assert arr.c[0] == min(c_r, c_b)


def test_nas_union_is_split_over_the_directions():
    """c = 0.01 +- 0.03: a single test at alpha_A passes (0.0946 < 0.1), the
    per-direction test at alpha_A / 2 does not (0.1075)."""
    assert E.status_c(0.01, 0.03, 9).status.item() == ABSENT
    pa = E.assess_principle("NAS", nas(0.01, 0.03, 0.01, 0.03), PROTO)
    assert pa.status.value == U


def test_nas_missing_direction_blocks_present_but_not_a_union_absent():
    ev_r = item("NAS", 0.6, 0.05, direction="receive")
    pa = E.assess_principle("NAS", [ev_r], PROTO)
    assert (pa.status.value, pa.reason) == (U, "MISSING_CHANNEL:default:return")
    assert pa.reasons == ("MISSING_CHANNEL:NAS:default:return",)
    pa = E.assess_principle("NAS", [item("NAS", 0.0, 0.02, direction="receive")], PROTO)
    assert pa.status.value == ABSENT


@pytest.mark.parametrize("other", ["missing", "non_finite", "no_anchor", "undefined"])
def test_nas_absent_through_one_direction_reports_its_value(other):
    """ABSENT through the receive direction while the return direction has
    no value: a decided status carries a finite c (the receive value), the
    record and the verdict are built, and the array entry point agrees."""
    ref = REF if other != "no_anchor" else {
        **REF, "values": {k: v for k, v in REF["values"].items() if k != "NAS:return"}}
    proto = E.ProtocolV3(reference=ref, directions={"NAS": ["receive", "return"]})
    evs = [item("NAS", 0.01, 0.02, direction="receive", protocol_id=proto.protocol_id)]
    if other == "non_finite":
        evs.append(item("NAS", float("nan"), 0.02, direction="return",
                        protocol_id=proto.protocol_id))
    elif other == "no_anchor":
        evs.append(item("NAS", 0.5, 0.02, direction="return",
                        protocol_id=proto.protocol_id))
    elif other == "undefined":
        evs.append(item("NAS", 0.5, 0.02, direction="return", defined=False,
                        reason="SAMPLING_UNRESOLVED", protocol_id=proto.protocol_id))
    pa = E.assess_principle("NAS", evs, proto)
    assert pa.status.value == ABSENT and pa.c == 0.01
    rec = pa.record_fields()
    assert (rec["status"], rec["c"], rec["c_R"], rec["c_B"]) == (
        ABSENT, 0.01, 0.01, None)
    assert rec["se_c"] == 0.02 and rec["df_c"] == 9.0
    v = E.mpc_verdict({"NAS": evs}, proto)
    assert v.verdict is v1.Verdict.EXCLUDED and v.construct["NAS"]["c"] == 0.01
    arr = E.status_c_directional([[0.01, np.nan]], [[0.02, 0.02]], 9)
    assert (arr.status[0], arr.c[0], arr.se[0]) == (ABSENT, 0.01, 0.02)


def test_nas_undefined_with_a_direction_without_value_reports_no_c():
    """min(c_R, c_B) is undefined when a direction has no value and the
    status is UNDEFINED; the defined direction stays in its own field."""
    pa = E.assess_principle("NAS", [item("NAS", 0.6, 0.05, direction="receive")], PROTO)
    assert pa.status.value == U
    rec = pa.record_fields()
    assert (rec["c"], rec["se_c"], rec["c_R"], rec["c_B"]) == (None, None, 0.6, None)
    arr = E.status_c_directional([[0.6, np.nan]], [[0.05, 0.05]], 9)
    assert arr.status[0] == U and np.isnan(arr.c[0])


def test_nas_direction_declarations_are_enforced():
    with pytest.raises(ValueError, match="every item needs one"):
        E.assess_principle("NAS", [item("NAS", 0.1, 0.02)], PROTO)
    with pytest.raises(ValueError, match="declares no directions"):
        E.assess_principle("RAM", [item("RAM", 0.1, 0.02, direction="receive")], PROTO)
    with pytest.raises(ValueError, match="more than one item"):
        E.assess_principle("NAS", nas(0, .02, 0, .02) + [
            item("NAS", 0.0, 0.02, direction="return")], PROTO)
    pa = E.assess_principle("NAS", nas(0, .02, 0, .02) + [
        item("NAS", 0.0, 0.02, direction="sideways")], PROTO)
    assert pa.ignored == ("default:sideways",)


def test_nas_per_direction_anchors_come_from_the_protocol():
    ref = {"kind": "external", "scale": "excess",
           "values": {"NAS:receive": 2.0, "NAS:return": 0.5}}
    proto = E.ProtocolV3(reference=ref, directions={"NAS": ["receive", "return"]})
    evs = nas(1.0, 0.02, 0.25, 0.02)  # the items' own reference 1.0 is replaced
    pa = E.assess_principle("NAS", evs, proto)
    assert pa.deciding.directions == {"receive": 0.5, "return": 0.5}
    proto2 = E.ProtocolV3(reference={**ref, "values": {"NAS:receive": 2.0}},
                          directions={"NAS": ["receive", "return"]})
    pa = E.assess_principle("NAS", evs, proto2)
    assert pa.reason == R.INVALID_ANCHORS


# --------------------------------------------------------------------------
# channels: OR, PRESENT through k channels at alpha / k
# --------------------------------------------------------------------------
def test_channels_present_union_is_split_and_absent_needs_every_channel():
    two = E.ProtocolV3(reference={"kind": "cohort_high_state", "session": "awake"},
                       channels={"RAM": ["a", "b"]})
    one = E.ProtocolV3(reference={"kind": "cohort_high_state", "session": "awake"},
                       channels={"RAM": ["a"]})
    # 0.45 - 1.833 * 0.1 = 0.267 at alpha; 0.45 - 2.262 * 0.1 = 0.224 at alpha / 2
    a = item("RAM", 0.45, 0.1, channel="a")
    b = item("RAM", 0.00, 0.03, channel="b")
    assert E.assess_principle("RAM", [a], one).status.value == PRESENT
    pa = E.assess_principle("RAM", [a, b], two)
    assert (pa.status.value, pa.reason) == (U, "ABSENT_NOT_REACHABLE")
    assert pa.channels == {"a": v1.ComponentStatus.UNDEFINED,
                           "b": v1.ComponentStatus.ABSENT}
    assert pa.assessments["a"].alpha == pytest.approx(0.025)
    a0 = item("RAM", 0.0, 0.03, channel="a")
    assert E.assess_principle("RAM", [a0, b], two).status.value == ABSENT
    pa = E.assess_principle("RAM", [b], two)
    assert (pa.status.value, pa.reasons) == (U, ("MISSING_CHANNEL:RAM:a",))
    pa = E.assess_principle("RAM", [b, item("RAM", 0, .03, channel="z")], two)
    assert pa.ignored == ("z",)


# --------------------------------------------------------------------------
# the concordance route
# --------------------------------------------------------------------------
def pdi(c, *, se=0.0, se_method="concordant", se_df=None, view="source", **kw):
    return item("PDI", c, se, se_df, se_method=se_method, substrate="family_a",
                observation_stage="source", view=view, content_bearer="non_workspace",
                **kw)


@pytest.mark.parametrize("ev, status, reason, flags", [
    (pdi(0.05), ABSENT, None, (R.ABSENT_BY_CONCORDANCE,)),
    (pdi(0.05, n_null=19, null_sd=0.4), ABSENT, None, (R.ABSENT_BY_CONCORDANCE,)),
    (pdi(-0.0999), ABSENT, None, (R.ABSENT_BY_CONCORDANCE,)),
    (pdi(0.50), U, "NO_SAMPLING_SE", ()),  # PRESENT route not admitted
    (pdi(0.15), U, "INCONCLUSIVE", ()),
    (pdi(-0.20), U, "NULL_MODEL_VIOLATED", ()),
    (pdi(0.05, view="eeg64"), U, "NO_SAMPLING_SE", ()),  # another cell
    (pdi(0.05, se_method="jackknife_contiguous_10", se_df=9.0), U, "NO_SAMPLING_SE",
     ()),
    (pdi(0.05, se_method=None), U, "NO_SAMPLING_SE", ()),  # v1 components
    (pdi(0.05, se=0.01), U, "INVALID_SE", ()),
    (pdi(0.05, se_df=9.0), U, "INVALID_SE", ()),
])
def test_concordance_route_only_when_admitted(ev, status, reason, flags):
    a = E.assess_item(ev, PROTO)
    assert (a.status.value, a.reason, a.flags) == (status, reason, flags)
    if status != U:
        assert a.route == "concordance"


def test_concordance_present_route_and_unadmitted_protocols():
    cell = {**CELL, "present": True}
    proto = E.ProtocolV3(reference=REF, concordance_route=[cell])
    a = E.assess_item(pdi(0.5), proto)
    assert (a.status.value, a.flags) == (PRESENT, (R.PRESENT_BY_CONCORDANCE,))
    a = E.assess_item(pdi(0.05), E.ProtocolV3(reference=REF))
    assert (a.status.value, a.reason) == (U, "NO_SAMPLING_SE")
    assert E.ProtocolV3(reference=REF, concordance_route=[cell]).concordance_admission(
        "PDI", "family_a", "source", "source", "non_workspace") == (True, True)


# --------------------------------------------------------------------------
# the SE-method contract: se_df read from the record per se_method
# --------------------------------------------------------------------------
@pytest.mark.parametrize("principle, method, kw, status, reason", [
    # c = 0, se = 0.036: df 12 passes the TOST (0.0965), df 9 does not (0.1016)
    ("IIM", "circular_block_bootstrap_10pct_B50", dict(se_df=12.0), ABSENT, None),
    ("IIM", "circular_block_bootstrap_10pct_B50", dict(se_df=9.0), U,
     "ABSENT_NOT_REACHABLE"),
    ("IIM", "circular_block_bootstrap_10pct_B50", dict(se_df=49.0), U, "INVALID_SE"),
    ("IIM", "circular_block_bootstrap_10pct_B50", dict(se_df=None), U, "INVALID_SE"),
    ("NAS", "jackknife_contiguous_10", dict(se_df=9.0, direction="receive"), U,
     "ABSENT_NOT_REACHABLE"),
    ("PDI", "jackknife_contiguous_10", dict(se_df=10.0), U, "INVALID_SE"),
    ("RAM", "shift_null_sd", dict(se_df=69.0, n_null=70, null_sd=0.0), ABSENT, None),
    ("RAM", "shift_null_sd", dict(se_df=70.0, n_null=70, null_sd=0.0), U, "INVALID_SE"),
    ("RAM", "jackknife_trials_10", dict(se_df=9.0), U, "ABSENT_NOT_REACHABLE"),
    ("SRPI", "jackknife_pairs_10", dict(se_df=9.0), U, "ABSENT_NOT_REACHABLE"),
    ("SRPI", "hoeffding", dict(se_df=37.3), ABSENT, None),
    ("SRPI", "hoeffding", dict(se_df=None), U, "INVALID_SE"),
    ("NAS", "jackknife_pairs_10", dict(se_df=9.0, direction="receive"), U,
     "INVALID_SE"),
    ("RAM", "exact", dict(se_df=9.0), U, "INVALID_SE"),
])
def test_se_df_is_read_from_the_record_per_se_method(
        principle, method, kw, status, reason):
    ev = item(principle, 0.0, 0.036, se_method=method, **kw)
    a = E.assess_item(ev, PROTO)
    assert (a.status.value, a.reason) == (status, reason)
    if a.reason != "INVALID_SE":
        assert a.df == kw["se_df"]


def test_protocol_se_methods_restrict_the_items():
    proto = E.ProtocolV3(reference=REF, se_methods={"IIM": ["jackknife_contiguous_10"]})
    ok = item("IIM", 0.0, 0.03, 9.0, se_method="jackknife_contiguous_10")
    bad = item("IIM", 0.0, 0.03, 12.0, se_method="circular_block_bootstrap_10pct_B50")
    assert E.assess_item(ok, proto).status.value == ABSENT
    assert E.assess_item(bad, proto).reason == R.INVALID_SE


def test_a_declared_se_contract_binds_items_without_a_method():
    """Where the protocol declares the admitted SE methods of a principle, an
    item must name its method: a block bootstrap reporting B - 1 = 49
    without its method would otherwise pass the TOST with df 49."""
    proto = E.ProtocolV3(reference=REF,
                         se_methods={"IIM": ["circular_block_bootstrap_10pct_B50"]})
    bare = item("IIM", 0.0, 0.036, 49.0)
    assert E.assess_item(bare, PROTO).status.value == ABSENT  # no declaration
    a = E.assess_item(bare, proto)
    assert (a.status.value, a.reason) == (U, "INVALID_SE")
    named = item("IIM", 0.0, 0.036, 12.0,
                 se_method="circular_block_bootstrap_10pct_B50")
    assert E.assess_item(named, proto).status.value == ABSENT
    # principles without a declaration keep the v1 meaning of se_df
    assert E.assess_item(item("RAM", 0.0, 0.03), proto).status.value == ABSENT
    # an item the estimator declared undefined needs no method
    a = E.assess_item(item("IIM", 0.0, 0.036, 49.0, defined=False,
                           reason="INSUFFICIENT_OCCUPANCY"), proto)
    assert a.reason == "INSUFFICIENT_OCCUPANCY"


def test_se_method_table_matches_the_contract():
    m = E.SE_METHODS
    assert m["circular_block_bootstrap_10pct_B50"].df_values == (12.0, 9.0)
    assert m["jackknife_contiguous_20"].df_values == (19.0,)
    assert m["shift_null_sd"].df_rule == "n_null_minus_1"
    assert m["hoeffding"].df_rule == "welch_satterthwaite"
    assert not m["concordant"].sampling_se and m["concordant"].principles == ("PDI",)
    for name in ("jackknife_contiguous_10", "jackknife_interleaved_10",
                 "jackknife_interleaved_20"):
        assert "NAS" in m[name].principles


# --------------------------------------------------------------------------
# IIM rank gate, registry licensing, SE reversion
# --------------------------------------------------------------------------
@pytest.mark.parametrize("c, se, p_ind, status, reason", [
    (0.6, 0.1, 0.03, PRESENT, None),
    (0.6, 0.1, 0.05, PRESENT, None),
    (0.6, 0.1, 0.06, U, "INCONCLUSIVE:NULL_NOT_EXCEEDED"),
    (0.6, 0.1, None, U, "NO_NULL_CALIBRATION"),
    (0.6, 0.1, float("nan"), U, "NO_NULL_CALIBRATION"),
    (0.0, 0.02, 0.9, ABSENT, None),  # the gate only removes PRESENT
    (0.3, 0.05, 0.9, U, "ABSENT_NOT_REACHABLE"),
])
def test_iim_rank_gate(c, se, p_ind, status, reason):
    ev = item("IIM", c, se, 12.0, se_method="circular_block_bootstrap_10pct_B50",
              p_ind=p_ind)
    a = E.assess_item(ev, PROTO)
    assert (a.status.value, a.reason) == (status, reason)
    no_gate = E.ProtocolV3(reference=REF)
    assert E.assess_item(ev, no_gate).status.value == (PRESENT if c == 0.6 else status)


class StubRegistry:
    def __init__(self, present, absent):
        self.present, self.absent, self.calls = present, absent, []

    def admission(self, principle, estimator, substrate, grain, **regime):
        self.calls.append((principle, estimator, substrate, grain, regime))
        return {"present": self.present, "absent": self.absent}


class StubRegistryV1:
    def __init__(self, ok):
        self.ok = ok

    def is_validated(self, principle, estimator, substrate=None, grain=None, **regime):
        return (True, None) if self.ok else (False, "NO_ENTRY")


@pytest.mark.parametrize("present, absent, c, se, status, reason", [
    ("no", "yes", 0.6, 0.1, U, "ESTIMATOR_NOT_VALIDATED:not_admitted"),
    ("no", "yes", 0.0, 0.02, ABSENT, None),
    ("yes", "no", 0.0, 0.02, U, "ESTIMATOR_NOT_VALIDATED:not_admitted"),
    ("yes", "no", 0.6, 0.1, PRESENT, None),
    ("yes", "vacuous", 0.0, 0.02, ABSENT, None),
    ("not_observable", "yes", 0.6, 0.1, U, "ESTIMATOR_NOT_VALIDATED:not_observable"),
    ("not_observable", "not_observable", 0.0, 0.02, U,
     "ESTIMATOR_NOT_VALIDATED:not_observable"),
    ("no", "no", 0.0, 0.02, U, "ESTIMATOR_NOT_VALIDATED:not_admitted"),
    ("no", "yes", 0.3, 0.05, U, "ABSENT_NOT_REACHABLE"),  # undecided: own reason
    (False, True, 0.6, 0.1, U, "ESTIMATOR_NOT_VALIDATED:not_admitted"),
])
def test_registry_licenses_each_status_direction(
        present, absent, c, se, status, reason):
    reg = StubRegistry(present, absent)
    a = E.assess_item(item("RAM", c, se, regime={"n_time": 100}), PROTO,
                      registry=reg, regime={"fs": 20})
    assert (a.status.value, a.reason) == (status, reason)
    assert reg.calls[0][4] == {"fs": 20, "n_time": 100}


def test_v1_registry_licenses_both_directions_together():
    assert E.assess_item(item(c=0.0, se=0.02), PROTO,
                         registry=StubRegistryV1(True)).status.value == ABSENT
    a = E.assess_item(item(c=0.0, se=0.02), PROTO, registry=StubRegistryV1(False))
    assert a.reason == "ESTIMATOR_NOT_VALIDATED:not_admitted"
    with pytest.raises(ValueError):
        E.assess_item(item(), PROTO, registry=StubRegistry("maybe", "yes"))


def test_se_reversion_reclassifies_absent_only():
    uncal = [("RAM", "jackknife_trials_10")]
    ab = item("RAM", 0.0, 0.03, se_method="jackknife_trials_10")
    pr = item("RAM", 0.6, 0.1, se_method="jackknife_trials_10")
    a = E.assess_item(ab, PROTO, uncalibrated_se_methods=uncal)
    assert (a.status.value, a.reason) == (U, "SE_NOT_CALIBRATED")
    a = E.assess_item(pr, PROTO, uncalibrated_se_methods=uncal)
    assert a.status.value == PRESENT
    other = [("RAM", "shift_null_sd")]
    a = E.assess_item(ab, PROTO, uncalibrated_se_methods=other)
    assert a.status.value == ABSENT


# --------------------------------------------------------------------------
# the three entry points agree; v1 arithmetic unchanged
# --------------------------------------------------------------------------
def _random_raw(rng, n):
    est = rng.normal(0.2, 0.6, n)
    nm = rng.normal(0.0, 0.2, n)
    nsd = np.abs(rng.normal(0.1, 0.1, n))
    nn = rng.choice([0, 0, 19, 39], n).astype(float)
    se = np.abs(rng.normal(0.08, 0.1, n))
    ref = rng.normal(1.0, 0.5, n)
    rse = np.abs(rng.normal(0.02, 0.03, n))
    sdf = rng.choice([np.nan, 9.0, 12.0, 19.0], n)
    scale = rng.choice(["estimate", "excess"], n)
    # sprinkle invalid inputs
    for arr, bad in ((est, np.nan), (nm, np.nan), (nn, 1.5), (nsd, np.nan),
                     (ref, -5.0), (se, -0.1), (se, 0.0), (rse, -0.1), (sdf, 0.0)):
        hit = rng.random(n) < 0.03
        arr[hit] = bad
    return est, nm, nsd, nn, se, ref, rse, sdf, scale


def test_three_entry_points_agree_on_random_inputs():
    rng = np.random.default_rng(123)
    n = 3000
    est, nm, nsd, nn, se, ref, rse, sdf, scale = _random_raw(rng, n)
    exact = rng.random(n) < 0.08
    se[exact & (rng.random(n) < 0.7)] = 0.0  # mostly exact values without an SE
    routes = set()
    for sc in ("estimate", "excess"):
        idx = np.flatnonzero(scale == sc)
        arr = E.assess_array(est[idx], nm[idx], nsd[idx], se[idx], n_null=nn[idx],
                             reference=ref[idx], reference_se=rse[idx],
                             reference_scale=sc, se_df=sdf[idx], exact=exact[idx])
        assert set(arr.status) == {PRESENT, ABSENT, U}
        assert set(arr.reason) - {None} == {
            "NULL_MODEL_VIOLATED", "ABSENT_NOT_REACHABLE", "INCONCLUSIVE",
            "NON_FINITE_ESTIMATE", "NO_NULL_CALIBRATION", "DEGENERATE_NULL",
            "INVALID_ANCHORS", "INVALID_SE", "NO_SAMPLING_SE"}
        assert arr.present_not_reachable.any()
        for j, i in enumerate(idx):
            ev = E.ComponentEvidenceV2(
                principle="RAM", estimate=est[i], null_mean=nm[i], null_sd=nsd[i],
                n_null=nn[i], se=se[i], se_df=None if np.isnan(sdf[i]) else sdf[i],
                reference=ref[i], reference_se=rse[i], reference_scale=sc,
                exact=bool(exact[i]))
            a = E.assess_item(ev, COHORT)
            routes.add(a.route)
            assert a.status.value == arr.status[j]
            assert a.reason == arr.reason[j]
            assert (R.PRESENT_NOT_REACHABLE in a.flags) == bool(
                arr.present_not_reachable[j])
            for name in ("c", "se", "df", "lower", "upper", "lower_absent",
                         "upper_absent", "margin_present", "margin_absent"):
                x, y = getattr(a, name), float(getattr(arr, name)[j])
                assert (math.isnan(x) and math.isnan(y)) or x == y, (name, x, y)
            old = v1.component_assessment(ev)
            if a.route in ("tost", "exact"):
                # c and df exactly as v1 computes them; se_c too, except on
                # the exact route, which decides on c alone
                assert (a.c, a.df) == (old.c, old.df)
                assert a.se == (old.se if a.route == "tost" else 0.0)
                sc_arr = E.status_c(a.c, a.se, a.df, exact=a.route == "exact")
                assert sc_arr.status.item() == a.status.value
                assert sc_arr.reason.item() == a.reason
                assert bool(sc_arr.present_not_reachable) == (
                    R.PRESENT_NOT_REACHABLE in a.flags)
    assert routes == {None, "tost", "exact"}


def test_directional_entry_points_agree_on_random_inputs():
    rng = np.random.default_rng(321)
    n = 1500
    c = rng.normal(0.1, 0.4, (n, 2))
    se = np.abs(rng.normal(0.05, 0.2, (n, 2)))
    se[rng.random((n, 2)) < 0.03] = 0.0
    df = rng.choice([9.0, 19.0], (n, 2))
    arr = E.status_c_directional(c, se, df)
    assert set(arr.status) == {PRESENT, ABSENT, U}
    assert {"NULL_MODEL_VIOLATED", "ABSENT_NOT_REACHABLE", "INCONCLUSIVE",
            "NO_SAMPLING_SE"} <= set(arr.reason)
    assert arr.present_not_reachable.any()
    for i in range(n):
        evs = [item("NAS", c[i, 0], se[i, 0], df[i, 0], direction="receive"),
               item("NAS", c[i, 1], se[i, 1], df[i, 1], direction="return")]
        pa = E.assess_principle("NAS", evs, PROTO)
        assert (pa.status.value, pa.reason) == (arr.status[i], arr.reason[i])
        flag = R.PRESENT_NOT_REACHABLE in pa.flags
        assert flag == bool(arr.present_not_reachable[i])
        d = pa.deciding
        for name in ("c", "se", "df", "margin_present", "margin_absent"):
            x, y = getattr(d, name), float(getattr(arr, name)[i])
            assert (math.isnan(x) and math.isnan(y)) or x == y, (name, x, y)


# --------------------------------------------------------------------------
# UNDEFINED is never ABSENT
# --------------------------------------------------------------------------
def test_undefined_never_absent_on_random_items():
    rng = np.random.default_rng(7)
    cells = [CELL, {**CELL, "view": "eeg64", "present": True}]
    proto = E.ProtocolV3(reference=REF, directions={"NAS": ["receive", "return"]},
                         rank_gates={"IIM": 0.05}, concordance_route=cells)
    regs = [None, StubRegistry("no", "yes"), StubRegistry("yes", "no"),
            StubRegistry("not_observable", "yes")]
    methods = [None, "jackknife_contiguous_10", "concordant", "bogus", "exact"]
    seen = set()
    for _ in range(4000):
        p = rng.choice(["RAM", "PDI", "IIM", "SRPI"])
        method = methods[rng.integers(len(methods))]
        kw = dict(se_method=method, substrate="family_a", observation_stage="source",
                  view=rng.choice(["source", "eeg64"]), content_bearer="non_workspace",
                  p_ind=rng.choice([0.01, 0.2, np.nan]),
                  exact=bool(rng.random() < 0.1), defined=bool(rng.random() > 0.05))
        se = float(rng.choice([0.0, 0.01, 0.03, 0.1, 0.5]))
        df = rng.choice([None, 9.0, 12.0])
        ev = item(p, float(rng.normal(0.1, 0.5)), se, df, **kw)
        reg = regs[rng.integers(len(regs))]
        a = E.assess_item(ev, proto, registry=reg,
                          uncalibrated_se_methods=[("RAM", "jackknife_contiguous_10")])
        R.check_status(a.status.value, a.reason, a.flags)
        seen.add(a.status.value)
        if a.status.value == ABSENT:
            if a.route == "concordance":
                assert abs(a.c) < a.cutoff_absent
            else:
                assert a.lower_absent > -a.cutoff_absent
                assert a.upper_absent < a.cutoff_absent
        else:
            assert R.ABSENT_BY_CONCORDANCE not in a.flags
        if a.status.value == U:
            assert R.status_for_reason(a.reason) == "UNDEFINED"
    assert seen == {PRESENT, ABSENT, U}


def test_random_verdicts_give_valid_component_records():
    """Random evidence of all five principles (NAS directions missing,
    non-finite, undefined or without an anchor; registry and SE reversion
    removing decisions): every verdict is built, every principle's record
    fields make a valid ``mpc-bench-result/3`` component record, a decided
    status carries a finite c, and NAS reports min(c_R, c_B) where both are
    finite."""
    from impact_pipeline.v2 import records as REC

    rng = np.random.default_rng(17)
    refs = [REF, {**REF, "values": {k: v for k, v in REF["values"].items()
                                    if k != "NAS:return"}}]
    protos = [E.ProtocolV3(reference=r, directions={"NAS": ["receive", "return"]},
                           rank_gates={"IIM": 0.05}, concordance_route=[CELL])
              for r in refs]
    regs = [None, StubRegistry("no", "yes"), StubRegistry("yes", "no")]
    seen = set()
    for _ in range(1500):
        proto = protos[rng.integers(len(protos))]
        pid = proto.protocol_id

        def draw(p, **kw):
            c = float(rng.normal(0.1, 0.4)) if rng.random() > 0.05 else float("nan")
            se = float(rng.choice([0.0, 0.01, 0.02, 0.05, 0.4]))
            return item(p, c, se, defined=bool(rng.random() > 0.05),
                        reason="SAMPLING_UNRESOLVED", protocol_id=pid, **kw)

        ev = {p: draw(p) for p in ("RAM", "IIM", "SRPI")}
        ev["IIM"] = dataclasses.replace(ev["IIM"], p_ind=float(rng.choice([0.01, 0.2])))
        ev["PDI"] = draw("PDI", se_method=str(rng.choice(
            ["concordant", "jackknife_contiguous_10"])), substrate="family_a",
            observation_stage="source", view="source", content_bearer="non_workspace")
        ev["PDI"] = dataclasses.replace(
            ev["PDI"], se=0.0 if ev["PDI"].se_method == "concordant" else 0.02,
            se_df=None if ev["PDI"].se_method == "concordant" else 9.0)
        ev["NAS"] = [draw("NAS", direction=d) for d in ("receive", "return")
                     if rng.random() > 0.15]
        v = E.mpc_verdict(ev, proto, registry=regs[rng.integers(len(regs))],
                          uncalibrated_se_methods=[("SRPI", "jackknife_pairs_10")])
        assert E.verdict_from_reasons(v.reasons) == v.verdict
        for p in E.PRINCIPLES:
            fields = v.construct[p]
            seen.add((p, fields["status"]))
            REC.ComponentRecord(
                principle=p, estimator_version="test", declaration_id="R",
                observation_stage="source", protocol_id=pid,
                protocol_hash=proto.hash, **fields)
            if fields["status"] != U:
                assert fields["c"] is not None and math.isfinite(fields["c"])
            if p == "NAS" and fields["c_R"] is not None and fields["c_B"] is not None:
                assert fields["c"] == min(fields["c_R"], fields["c_B"])
    assert {s for _, s in seen} == {PRESENT, ABSENT, U}
    assert ("NAS", ABSENT) in seen and ("PDI", ABSENT) in seen


def test_every_reason_of_the_layer_is_in_the_vocabulary():
    for code in E._REASON_TABLE[1:]:
        assert R.status_for_reason(code) == "UNDEFINED"
    assert set(E._DECISION_REASONS) <= set(R.REASONS)


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------
def full(c=(0.6, 0.6, 0.6, 0.6, 0.6), se=0.05, proto=PROTO):
    ram, pdi_, nas_, iim, srpi = c
    pid = proto.protocol_id
    return {
        "RAM": item("RAM", ram, se, protocol_id=pid),
        "PDI": item("PDI", pdi_, se, protocol_id=pid),
        "NAS": [item("NAS", nas_, se, direction="receive", protocol_id=pid),
                item("NAS", nas_, se, direction="return", protocol_id=pid)],
        "IIM": item("IIM", iim, se, p_ind=0.01, protocol_id=pid),
        "SRPI": item("SRPI", srpi, se, protocol_id=pid),
    }


def test_verdicts_follow_the_kleene_and():
    v = E.mpc_verdict(full(), PROTO)
    assert isinstance(v, E.MPCVerdictV2)
    assert (v.verdict, v.reasons) == (v1.Verdict.MPC_CONSISTENT, [])
    v = E.mpc_verdict(full((0.6, 0.6, 0.0, 0.6, 0.6), se=0.02), PROTO)
    assert v.verdict is v1.Verdict.EXCLUDED and "ABSENT:NAS" in v.reasons
    v = E.mpc_verdict(full((0.6, 0.6, 0.6, 0.3, 0.6)), PROTO)
    assert v.verdict is v1.Verdict.UNDETERMINED
    assert v.reasons == ["ABSENT_NOT_REACHABLE:IIM"]
    for verdict in (v, E.mpc_verdict(full((0.0, 0.6, 0.6, 0.6, 0.6), se=0.02), PROTO)):
        assert E.verdict_from_reasons(verdict.reasons) == verdict.verdict
    d = v.to_dict()
    assert d["schema"] == "impact-mpc-protocol/3"
    assert d["construct"]["NAS"]["c_R"] == 0.6
    assert d["status_rule"]["alpha_absent"] == 0.01 and d["sensitivity"] is False


def test_undefined_components_never_make_a_verdict_excluded():
    ev = full((0.6, 0.6, 0.6, 0.6, -0.3))  # SRPI NULL_MODEL_VIOLATED
    v = E.mpc_verdict(ev, PROTO)
    assert v.verdict is v1.Verdict.UNDETERMINED
    assert v.reasons == ["NULL_MODEL_VIOLATED:SRPI"]
    ev["SRPI"] = item("SRPI", 0.0, 0.0, protocol_id=PROTO.protocol_id)
    v = E.mpc_verdict(ev, PROTO)
    assert v.reasons == ["NO_SAMPLING_SE:SRPI"] and v.verdict is v1.Verdict.UNDETERMINED


def test_verdict_over_the_anchored_set_and_the_declared_set():
    proto = E.ProtocolV3(reference=REF, directions={"NAS": ["receive", "return"]},
                         rank_gates={"IIM": 0.05},
                         necessity_set=("RAM", "PDI", "NAS", "IIM"))
    assert proto.status_rule.alpha_absent == 0.01  # alpha_A does not follow N_anch
    ev = full((0.6, 0.6, 0.6, 0.6, 0.0), se=0.02, proto=proto)
    v = E.mpc_verdict(ev, proto)
    assert v.verdict is v1.Verdict.MPC_CONSISTENT
    assert v.verdict_declared is v1.Verdict.EXCLUDED
    assert v.reasons_declared == ["ABSENT:SRPI"]
    assert v.component_status["SRPI"] is v1.ComponentStatus.ABSENT


def test_missing_principles_and_global_codes():
    ev = full()
    del ev["PDI"]
    v = E.mpc_verdict(ev, PROTO)
    assert v.reasons == ["MISSING:PDI"] and v.principle_reasons["PDI"] == "MISSING"
    ev = full()
    ev["RAM"] = item("RAM", 0.6, 0.05, protocol_id="sha256:" + "0" * 64)
    v = E.mpc_verdict(ev, PROTO)
    assert v.verdict is v1.Verdict.UNDETERMINED and "PROTOCOL_MISMATCH" in v.reasons
    assert E.mpc_verdict(ev, PROTO, require_same_protocol=False).verdict is (
        v1.Verdict.MPC_CONSISTENT)
    ev = full()
    ev["RAM"] = item("RAM", 0.6, 0.05, bearer_id="other", protocol_id=PROTO.protocol_id)
    ev["PDI"] = item("PDI", 0.6, 0.05, bearer_id="system",
                     protocol_id=PROTO.protocol_id)
    assert "BEARER_MISMATCH" in E.mpc_verdict(ev, PROTO).reasons


def test_sensitivity_rule_is_recorded():
    rule = E.StatusRule(alpha_absent=0.0125)  # alpha / |N_anch| with four anchored
    ev = full((0.6, 0.6, 0.6, 0.6, 0.0), se=0.0364)
    # SRPI: 2.821 * 0.0364 = 0.1027 (fails at 0.01); t(9; 0.9875) * 0.0364 < 0.1
    assert E.mpc_verdict(ev, PROTO).component_status["SRPI"].value == U
    v = E.mpc_verdict(ev, PROTO, status_rule=rule)
    assert v.component_status["SRPI"].value == ABSENT
    assert v.sensitivity is True and v.status_rule["alpha_absent"] == 0.0125


# --------------------------------------------------------------------------
# amplitude scale of quadratic statistics
# --------------------------------------------------------------------------
def test_amplitude_cutoffs_and_values():
    assert E.amplitude_cutoff((0.25, 0.10)) == (0.0625, pytest.approx(0.01))
    assert E.amplitude_value(-0.04) == pytest.approx(-0.2)
    assert E.is_quadratic("nas-v3-2026.10") and E.is_quadratic("iim-v5-2026.10")
    assert not E.is_quadratic("ram-v3-2026.10") and not E.is_quadratic("pdi-v3-2026.10")
    assert not E.is_quadratic("srpi-v2-2026.09") and not E.is_quadratic(None)


def test_amplitude_rejudging_of_quadratic_absents():
    proto = E.ProtocolV3(reference=REF, estimator_versions={
        "IIM": "iim-v5-2026.10", "RAM": "ram-v3-2026.10"})
    est = {"IIM": "compute_IIM:directional@iim-v5-2026.10",
           "RAM": "compute_RAM:pe_readout@ram-v3-2026.10"}
    # c = 0.08 +- 0.005: ABSENT on c (0.066..0.094), PRESENT on the amplitude
    # scale (0.08 - 1.833*0.005 = 0.0708 > 0.25**2): sqrt(0.08) = 0.28 > z
    iim = item("IIM", 0.08, 0.005, estimator=est["IIM"])
    assert E.assess_item(iim, proto).status.value == ABSENT
    assert E.assess_item(iim, proto, scale="amplitude").status.value == PRESENT
    # the amplitude margin on c is delta**2 = 0.01: 2.821 * 0.003 = 0.0085 passes
    iim0 = item("IIM", 0.0, 0.003, estimator=est["IIM"])
    assert E.assess_item(iim0, proto, scale="amplitude").status.value == ABSENT
    iim1 = item("IIM", 0.0, 0.005, estimator=est["IIM"])  # 0.0141 >= 0.01
    a = E.assess_item(iim1, proto, scale="amplitude")
    assert (a.status.value, a.reason) == (U, "ABSENT_NOT_REACHABLE")
    ram = item("RAM", 0.08, 0.005, estimator=est["RAM"])  # first order: unchanged
    assert E.assess_item(ram, proto, scale="amplitude").status.value == ABSENT
    arr = E.status_c(0.08, 0.005, 9, cutoff=E.amplitude_cutoff((0.25, 0.1)))
    assert arr.status.item() == PRESENT
    with pytest.raises(ValueError):
        E.assess_item(iim, proto, scale="log")


# --------------------------------------------------------------------------
# testability gating (precision table)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("se, df, members, pi0", [
    # preregistration v2, section 3.7: pi0 at c = 0 from the development
    # precision
    (0.020, 9, 2, 0.92), (0.025, 9, 2, 0.55), (0.030, 9, 2, 0.07),
    (0.023, 9, 2, 0.73), (0.050, 12, 1, 0.00), (0.32, 9, 1, 0.00),
    (0.021, 54, 1, 0.98), (0.10, 9, 1, 0.00),
])
def test_analytic_pi0_reproduces_the_reachability_table(se, df, members, pi0):
    assert round(T.pi0_analytic([se], df, members=members), 2) == pi0


def test_analytic_pi1():
    # Phi((1 - 0.25) / 0.2 - 1.833) = Phi(1.917) = 0.972
    assert T.pi1_analytic([0.2], 9) == pytest.approx(0.972, abs=1e-3)
    assert T.pi1_analytic([0.0], 9) == 1.0
    assert math.isnan(T.pi1_analytic([np.nan], 9))


def runs(statuses, se=0.02, df=9.0, seed0=320):
    out = []
    for i, (st, reason) in enumerate(statuses):
        out.append({"status": st, "reason": reason, "flags": [], "se_c": se,
                    "df_c": df, "seed": seed0 + i})
    return out


def test_gate_and_testability_rows():
    rows = runs([(ABSENT, None)] * 37 + [(U, "ABSENT_NOT_REACHABLE")] * 3)
    row = T.testability_row(family="A-R", principle="NAS", witness="W_NAS_no_workspace",
                            kind="absent", runs=rows, members=2)
    assert row["pi"] == 37 / 40 and row["decisive"] and row["outcome"] == "decisive"
    assert row["replacement"] is None
    assert row["predicted_absent_not_reachable_rate"] == 3 / 40
    assert row["pi_analytic"] == pytest.approx(0.92, abs=5e-3)
    assert round(row["s_A"], 3) == 0.031 and row["attainable_se_c"] == 0.02
    rows = runs([(ABSENT, None)] * 30 + [(U, "ABSENT_NOT_REACHABLE")] * 10)
    row = T.testability_row(family="A-R", principle="IIM", witness="W_IIM_feedforward",
                            kind="absent", runs=rows)
    assert not row["decisive"] and row["outcome"] == "NOT_TESTABLE_BY_DESIGN"
    assert row["replacement"] == "target not PRESENT in >= 80 % of seeds"
    rows = runs([(PRESENT, None)] * 35 + [(U, "INCONCLUSIVE")] * 5)
    row = T.testability_row(family="A-R", principle="RAM", witness="PC_nominal",
                            kind="present", runs=rows)
    assert row["pi"] == 0.875 and not row["decisive"]
    assert row["replacement"] == "paired contrast with the positive control"
    assert row["predicted_absent_not_reachable_rate"] is None
    assert T.gate("absent", 0.9)["decisive"] and not T.gate("absent", float("nan"))[
        "decisive"]


def test_gate_reads_development_runs_only():
    rows = runs([(ABSENT, None)] * 40)
    with pytest.raises(T.TestabilityError, match="at least 40"):
        T.testability_row(family="A-R", principle="NAS", witness="w", kind="absent",
                          runs=rows[:39])
    bad = runs([(ABSENT, None)] * 40, seed0=20000)
    with pytest.raises(T.TestabilityError, match="development"):
        T.testability_row(family="A-R", principle="NAS", witness="w", kind="absent",
                          runs=bad)
    bad = runs([(ABSENT, None)] * 40, seed0=10000)
    with pytest.raises(T.TestabilityError):
        T.testability_row(family="A-R", principle="NAS", witness="w", kind="absent",
                          runs=bad)
    bad = runs([(ABSENT, "INCONCLUSIVE")] * 40)
    with pytest.raises(T.TestabilityError):
        T.testability_row(family="A-R", principle="NAS", witness="w", kind="absent",
                          runs=bad)
    bad = runs([(U, "NOT_A_REASON")] * 40)
    with pytest.raises(T.TestabilityError):
        T.testability_row(family="A-R", principle="NAS", witness="w", kind="absent",
                          runs=bad)


@pytest.mark.parametrize("dose, nominal, median_c, on", [
    (1.0, 1.0, 0.5, True), (0.5, 1.0, 0.2, True),  # both thresholds inclusive
    (0.49, 1.0, 0.5, False), (1.0, 1.0, 0.19, False), (2.0, 1.0, None, False),
    (0.15, 0.3, 0.25, True),  # eta at half the nominal 0.3
])
def test_mechanism_on_labels(dose, nominal, median_c, on):
    assert T.mechanism_on(dose, nominal, median_c) is on


def test_development_median_c_reads_development_runs_only():
    runs = [{"c": v, "seed": 320 + i} for i, v in enumerate([0.1, 0.4, None, 0.3])]
    assert T.development_median_c(runs) == pytest.approx(0.3)
    assert T.development_median_c([{"c": None}]) is None
    with pytest.raises(T.TestabilityError):
        T.development_median_c([{"c": 0.3, "seed": 20001}])
    with pytest.raises(ValueError):
        T.mechanism_on(1.0, 0.0, 0.5)


def test_status_rates():
    rows = runs([(PRESENT, None), (ABSENT, None), (U, "INCONCLUSIVE:NULL_NOT_EXCEEDED"),
                 (U, "NULL_MODEL_VIOLATED")])
    rows[2]["flags"] = [R.PRESENT_NOT_REACHABLE]
    rates = T.status_rates(rows)
    assert rates["PRESENT"] == rates["ABSENT"] == 0.25 and rates["UNDEFINED"] == 0.5
    assert rates["INCONCLUSIVE"] == 0.25 and rates["NULL_MODEL_VIOLATED"] == 0.25
    assert rates["PRESENT_NOT_REACHABLE"] == 0.25 and rates["INVALID_ANCHORS"] == 0.0
