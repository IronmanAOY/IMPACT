# -*- coding: utf-8 -*-
"""The central UNDEFINED vocabulary: every reason maps to UNDEFINED, never to
ABSENT, in the vocabulary itself and through the v1 evidence layer."""
import pytest

from impact_pipeline import evidence as E
from impact_pipeline.v2 import reasons as R

# Every reason and flag that preregistration v2, section 3.2, names, and the
# HUB_NOT_PRIVILEGED flag of docs/metrics_v2.md.
DESIGN_REASONS = (
    "INVALID_ANCHORS", "INCONCLUSIVE", "NO_SAMPLING_SE", "NULL_FAMILY_MISMATCH",
    "MISSING_CHANNEL", "ABSENT_NOT_REACHABLE", "NULL_MODEL_VIOLATED",
    "SAMPLING_UNRESOLVED", "OBSERVATION_MIXED_NOT_ADMITTED", "INSUFFICIENT_OCCUPANCY",
    "MACRO_RANK_DEFICIENT", "NOT_APPLICABLE_OBSERVATION_MODEL",
    "INSUFFICIENT_TIMEPOINTS", "NO_NULL_CALIBRATION", "INSUFFICIENT_UPDATES",
    "ESTIMATOR_ERROR", "SE_NOT_CALIBRATED", "ESTIMATOR_NOT_VALIDATED",
)
DESIGN_FLAGS = (
    "PRESENT_NOT_REACHABLE", "ANCHOR_NOT_REPLICATED", "HUB_NOT_PRIVILEGED",
    "ABSENT_BY_CONCORDANCE", "PRESENT_BY_CONCORDANCE",
)


def instances(reason: R.Reason):
    """Every admissible spelling of a reason (bare and with details)."""
    out = []
    if reason.detail in (R.DETAIL_NONE, R.DETAIL_OPTIONAL, R.DETAIL_ENUM_OPTIONAL):
        out.append(reason.code)
    if reason.detail in (R.DETAIL_OPTIONAL, R.DETAIL_REQUIRED):
        out.append(f"{reason.code}:SomeDetail_1")
    if reason.detail in (R.DETAIL_ENUM, R.DETAIL_ENUM_OPTIONAL):
        out.extend(f"{reason.code}:{d}" for d in reason.details)
    return out


ALL_INSTANCES = [text for r in R.REASONS.values() for text in instances(r)]


def test_the_design_vocabulary_is_complete():
    assert set(DESIGN_REASONS) <= set(R.REASONS)
    assert set(DESIGN_FLAGS) <= set(R.FLAGS)
    assert R.REASONS["ESTIMATOR_NOT_VALIDATED"].details == (
        "not_admitted", "not_observable")
    assert R.REASONS["INCONCLUSIVE"].details == ("NULL_NOT_EXCEEDED",)


@pytest.mark.parametrize("text", ALL_INSTANCES)
def test_every_reason_maps_to_undefined_never_absent(text):
    status = R.status_for_reason(text)
    assert status == R.UNDEFINED
    assert status != R.ABSENT
    assert R.is_reason(text)
    assert R.lookup(text).status == R.UNDEFINED


def test_every_reason_has_at_least_one_spelling_and_a_meaning():
    for code, reason in R.REASONS.items():
        assert instances(reason), code
        assert reason.meaning and reason.source, code
        assert reason.detail in R.DETAIL_POLICIES
    assert all(row["status"] == R.UNDEFINED for row in R.reason_table())
    assert [row["code"] for row in R.reason_table()] == list(R.REASONS)


def test_reasons_flags_and_statuses_are_disjoint():
    assert not set(R.REASONS) & set(R.FLAGS)
    assert not set(R.REASONS) & set(R.STATUSES)
    assert not set(R.FLAGS) & set(R.STATUSES)
    # the v1 decomposition rule reads any 'ABSENT' kind as an exclusion; no
    # reason may be spelled so that it parses as that kind
    for text in ALL_INSTANCES:
        assert E.parse_reason(text)[0] != E.REASON_ABSENT


def test_v1_reasons_keep_their_v1_spelling():
    assert R.INVALID_ANCHORS == E.REASON_INVALID_ANCHORS
    assert R.INCONCLUSIVE == E.REASON_INCONCLUSIVE
    assert R.NO_SAMPLING_SE == E.REASON_NO_SAMPLING_SE
    assert R.NULL_FAMILY_MISMATCH == E.REASON_NULL_FAMILY_MISMATCH
    assert R.MISSING_CHANNEL == E.REASON_MISSING_CHANNEL
    assert R.NO_NULL_CALIBRATION == E.REASON_NO_NULL
    assert R.DEGENERATE_NULL == E.REASON_DEGENERATE_NULL
    assert R.INVALID_SE == E.REASON_INVALID_SE
    assert R.NOT_IMPLEMENTED == E.REASON_NOT_IMPLEMENTED
    assert R.MISSING == E.REASON_MISSING
    assert R.ESTIMATOR_NOT_VALIDATED == E.REASON_NOT_VALIDATED
    assert set(R.V1_REASONS) | set(R.V2_REASONS) == set(R.REASONS)


def _present(principle):
    return E.ComponentEvidence(principle, 1.0, 0.0, 0.1, se=0.01, n_null=30,
                               reference=1.0, reference_scale="excess")


@pytest.mark.parametrize("text", ALL_INSTANCES)
def test_the_v1_evidence_layer_maps_every_reason_to_undefined(text):
    """A component that carries any reason is UNDEFINED in the evidence
    layer, and a verdict over it is never EXCLUDED."""
    item = E.ComponentEvidence("NAS", float("nan"), defined=False, reason=text)
    a = E.component_assessment(item)
    assert a.status == E.ComponentStatus.UNDEFINED
    assert a.reason == text
    evidence = {p: [_present(p)] for p in E.PRINCIPLES}
    evidence["NAS"] = [item]
    v = E.mpc_verdict(evidence)
    assert v.component_status["NAS"] == E.ComponentStatus.UNDEFINED
    assert v.verdict == E.Verdict.UNDETERMINED
    assert not any(code.startswith("ABSENT") for code in v.reasons)
    assert E.verdict_from_reasons(v.reasons) == E.Verdict.UNDETERMINED
    # the same component alone: UNDETERMINED, never EXCLUDED
    assert E.mpc_verdict({"NAS": [item]}, necessity_set=("NAS",)).verdict == (
        E.Verdict.UNDETERMINED)


def test_the_present_fixture_is_present():
    v = E.mpc_verdict({p: [_present(p)] for p in E.PRINCIPLES})
    assert v.verdict == E.Verdict.MPC_CONSISTENT


@pytest.mark.parametrize("text", [
    None, "", "ABSENT", "PRESENT", "UNDEFINED", "NOT_A_REASON", "not_a_reason",
    "PRESENT_NOT_REACHABLE", "ABSENT_BY_CONCORDANCE",  # flags
    "ESTIMATOR_ERROR",  # detail required
    "ESTIMATOR_NOT_VALIDATED", "ESTIMATOR_NOT_VALIDATED:other",  # enumerated
    "INCONCLUSIVE:SOMETHING_ELSE",
    "INVALID_ANCHORS:detail", "SAMPLING_UNRESOLVED:x",  # no detail allowed
    "NOT_DEFINED:a;b", "NOT_DEFINED:", "MISSING_CHANNEL:a\nb",
])
def test_unknown_or_malformed_reasons_are_refused(text):
    assert not R.is_reason(text)
    with pytest.raises(R.UnknownReasonError):
        R.status_for_reason(text)


def test_format_and_parse():
    assert R.format_reason("INCONCLUSIVE") == "INCONCLUSIVE"
    assert R.format_reason("INCONCLUSIVE", "NULL_NOT_EXCEEDED") == (
        "INCONCLUSIVE:NULL_NOT_EXCEEDED")
    assert R.parse_reason("NULL_FAMILY_MISMATCH:onset_jitter/circular_shift") == (
        "NULL_FAMILY_MISMATCH", "onset_jitter/circular_shift")
    assert R.parse_reason("NO_SAMPLING_SE") == ("NO_SAMPLING_SE", None)
    assert R.not_validated() == "ESTIMATOR_NOT_VALIDATED:not_admitted"
    assert R.not_validated("not_observable") == "ESTIMATOR_NOT_VALIDATED:not_observable"
    with pytest.raises(R.UnknownReasonError):
        R.format_reason("SAMPLING_UNRESOLVED", "x")
    with pytest.raises(R.UnknownReasonError):
        R.not_validated("maybe")


def test_v1_estimator_reasons_can_be_carried_as_details():
    """A v1 estimator's own reason (free text, possibly with ';' or ':')
    travels as the detail of NOT_DEFINED and stays UNDEFINED."""
    raw = "agency_contract_violation:missing_agency_events; see log\nline 2"
    detail = R.clean_detail(raw)
    assert ";" not in detail and "\n" not in detail
    text = R.format_reason("NOT_DEFINED", detail)
    assert R.parse_reason(text) == ("NOT_DEFINED", detail)
    assert R.status_for_reason(text) == R.UNDEFINED
    with pytest.raises(R.UnknownReasonError):
        R.format_reason("NOT_DEFINED", raw)


def test_estimator_error_names_the_exception_type():
    assert R.estimator_error(ValueError("bad band")) == "ESTIMATOR_ERROR:ValueError"
    assert R.estimator_error(ZeroDivisionError) == "ESTIMATOR_ERROR:ZeroDivisionError"
    assert R.estimator_error("LinAlgError") == "ESTIMATOR_ERROR:LinAlgError"
    import numpy as np

    assert R.estimator_error(np.linalg.LinAlgError("x")) == (
        "ESTIMATOR_ERROR:LinAlgError")
    assert R.is_estimator_error("ESTIMATOR_ERROR:KeyError")
    assert not R.is_estimator_error("INCONCLUSIVE")
    assert not R.is_estimator_error(None)
    assert R.status_for_reason(R.estimator_error(KeyError("k"))) == R.UNDEFINED
    with pytest.raises(R.UnknownReasonError):
        R.estimator_error("not a type; name")


def test_check_status_pairs_statuses_reasons_and_flags():
    R.check_status("PRESENT")
    R.check_status("ABSENT")
    R.check_status("UNDEFINED", "ABSENT_NOT_REACHABLE", ["PRESENT_NOT_REACHABLE"])
    R.check_status("ABSENT", None, ["ABSENT_BY_CONCORDANCE"])
    R.check_status("PRESENT", None, ["PRESENT_BY_CONCORDANCE", "HUB_NOT_PRIVILEGED"])
    with pytest.raises(ValueError):
        R.check_status("ABSENT", "INCONCLUSIVE")  # ABSENT with a reason
    with pytest.raises(ValueError):
        R.check_status("PRESENT", "NULL_MODEL_VIOLATED")
    with pytest.raises(R.UnknownReasonError):
        R.check_status("UNDEFINED")  # UNDEFINED without a reason
    with pytest.raises(R.UnknownReasonError):
        R.check_status("UNDEFINED", "SOMETHING")
    with pytest.raises(ValueError):
        R.check_status("EXCLUDED")
    with pytest.raises(ValueError):
        R.check_status("PRESENT", None, ["PRESENT_NOT_REACHABLE"])
    with pytest.raises(ValueError):
        R.check_status("UNDEFINED", "INCONCLUSIVE", ["ABSENT_BY_CONCORDANCE"])
    with pytest.raises(ValueError):
        R.check_status("ABSENT", None, ["PRESENT_BY_CONCORDANCE"])
    with pytest.raises(ValueError):
        R.check_status("UNDEFINED", "INCONCLUSIVE", ["NOT_A_FLAG"])


def test_flags_are_never_statuses_or_reasons():
    for code in R.FLAGS:
        assert R.check_flag(code).code == code
        assert not R.is_reason(code)
    assert set(R.ROUTE_MARKERS) == {"ABSENT_BY_CONCORDANCE", "PRESENT_BY_CONCORDANCE"}
    with pytest.raises(ValueError):
        R.check_flag("INCONCLUSIVE")
