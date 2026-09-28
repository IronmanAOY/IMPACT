import math

import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm


def test_ci_undefined_when_iim_undefined():
    # D1: an unmeasurable weighted component makes CI undefined (NaN), never 0.
    out = mm.compute_CI(
        ram=1.0,
        pdi=1.0,
        nas=1.0,
        iim=float("nan"),
        srpi=1.0,
        defined={"IIM": False},
        return_details=True,
    )
    assert math.isnan(out["value"])
    assert out["defined"] is False
    assert out["missing"] == ["IIM"]
    assert "IIM" in out["undefined_weighted_components"]
    assert math.isnan(mm.compute_CI(1.0, 1.0, 1.0, float("nan"), 1.0))


def test_ci_measured_zero_component_is_zero_not_undefined():
    out = mm.compute_CI(0.0, 1.0, 1.0, 1.0, 1.0, return_details=True)
    assert out["value"] == 0.0
    assert out["defined"] is True
    assert out["missing"] == []


def test_ci_zero_weight_component_may_be_undefined():
    # Previously 0*log(NaN) made CI NaN; a zero-weight component is now ignored.
    w = {"RAM": 1, "PDI": 1, "NAS": 1, "IIM": 1}
    val = mm.compute_CI(0.5, 0.03, 0.2, 0.1, float("nan"), weights=w)
    expected = float(np.exp(np.mean(np.log([0.5, 0.03, 0.2, 0.1]))))
    assert val == pytest.approx(expected, rel=1e-9)


def test_ci_is_weighted_geometric_mean_of_reference_normalised_components():
    refs = {"RAM": 2.0, "PDI": 0.5, "NAS": 4.0, "IIM": 1.0, "SRPI": 0.25}
    vals = (1.0, 1.0, 2.0, 0.5, 0.5)
    out = mm.compute_CI(*vals, references=refs, return_details=True)
    norms = [v / refs[k] for v, k in zip(vals, ("RAM", "PDI", "NAS", "IIM", "SRPI"))]
    assert out["value"] == pytest.approx(float(np.prod(norms) ** 0.2), rel=1e-9)
    # Normalising every component by its reference makes a reference-valued run CI == 1.
    assert mm.compute_CI(*refs.values(), references=refs) == pytest.approx(
        1.0, rel=1e-9
    )


@pytest.mark.parametrize("bad_ref", [0.0, -1.0, float("nan"), float("inf")])
def test_ci_invalid_reference_is_undefined_without_floor(bad_ref):
    # D3: no 1e-12 floor and no exception; an unusable reference makes CI undefined.
    refs = {"RAM": 1.0, "PDI": 1.0, "NAS": bad_ref, "IIM": 1.0, "SRPI": 1.0}
    out = mm.compute_CI(1.0, 1.0, 1.0, 1.0, 1.0, references=refs, return_details=True)
    assert math.isnan(out["value"])
    assert out["defined"] is False
    assert out["missing"] == ["NAS_reference"]


def test_ci_reference_dict_missing_component_is_not_silently_one():
    # Old code silently used 1.0 for a component absent from the reference dict.
    refs = {"RAM": 1.0, "PDI": 1.0, "NAS": 1.0, "IIM": 1.0}
    out = mm.compute_CI(1.0, 1.0, 1.0, 1.0, 1.0, references=refs, return_details=True)
    assert math.isnan(out["value"]) and out["missing"] == ["SRPI_reference"]
    # Explicitly passing no references keeps the documented unit normalisation.
    assert mm.compute_CI(1.0, 1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)
