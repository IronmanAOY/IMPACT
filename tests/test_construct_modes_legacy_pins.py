"""
Legacy defaults of RAM/PDI/NAS/SRPI pinned to the code before the construct
revisions.

The construct revisions add opt-in modes and keywords; every
default must keep the old behaviour. Comparing ``mode='legacy'`` with the
default inside the same code base cannot detect a changed default, so these
values were produced by the code at commit c565a23 (before those revisions) on
the inputs below and are pinned here.
"""

import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm

REL = 1e-9


def _inputs():
    rng = np.random.default_rng(20260928)
    tr = 0.1
    goal = 0.5 + 3.0 * np.arange(20)
    stim, fb = goal + 0.8, goal + 2.0
    ram_ts = rng.standard_normal((12, int((fb[-1] + 3.0) / tr)))
    for k, s in enumerate(stim):
        i = int(round(s / tr))
        ram_ts[:6, i + 1 : i + 6] += 1.0 + 0.1 * k
        g = int(round(goal[k] / tr))
        ram_ts[6:, g + 1 : g + 6] += 0.5 * rng.standard_normal()
    fbv = rng.random(20)
    bundle = dict(
        onsets=stim.tolist(),
        goal_onsets=goal.tolist(),
        feedback_onsets=fb.tolist(),
        feedback_values=fbv.tolist(),
    )
    pdi_ts = rng.standard_normal((10, 400)).cumsum(axis=1)
    pdi_base = [rng.standard_normal((10, 300)), rng.standard_normal((10, 350))]
    nas_ts = rng.standard_normal((16, 300)).cumsum(axis=1)
    srpi_ts = rng.standard_normal((9, 600))
    on = np.arange(20, 580, 14).astype(float)
    for o in on[::2]:
        srpi_ts[:4, int(o) + 2 : int(o) + 6] += 1.0
    return tr, ram_ts, bundle, pdi_ts, pdi_base, nas_ts, srpi_ts, on


def test_legacy_defaults_match_the_pre_revision_code():
    tr, ram_ts, bundle, pdi_ts, pdi_base, nas_ts, srpi_ts, on = _inputs()
    ram = mm.compute_RAM(
        ram_ts,
        tr=tr,
        stimulus_onsets=bundle,
        response_model="boxcar",
        response_window_sec=0.5,
        quality_null_samples=30,
        return_details=True,
    )
    assert ram["magnitude_term"] == pytest.approx(1.8024090195747375, rel=REL)
    assert ram["components"]["goal_alignment"] == pytest.approx(
        0.49150931741995973, rel=REL
    )
    assert ram["components"]["feedback_integration"] == 0.0
    assert ram["components"]["adaptive_update"] == 0.0
    assert ram["value"] == 0.0
    assert ram["impact_channel"] == "untyped" and ram["update"] == "feedback_magnitude"

    pdi = mm.compute_PDI(pdi_ts, baseline_ts=pdi_base)
    assert pdi == pytest.approx(0.2802221564055817, rel=REL)

    nas = mm.compute_NAS(
        nas_ts,
        tr=1.0,
        tau=0.2,
        bands=[(0.05, 0.2)],
        band_weights=[1.0],
        window_len=60,
        step_len=30,
    )
    assert nas == pytest.approx(0.1960860578148668, rel=REL)

    srpi = mm.compute_SRPI(
        srpi_ts,
        tr=1.0,
        self_onsets=on[::2],
        nonself_onsets=on[1::2],
        pre_window_sec=2.0,
        response_lag_sec=2.0,
        response_window_sec=4.0,
        return_details=True,
    )
    raw = srpi["components_raw"]
    assert srpi["separability_cv_auc"] == pytest.approx(0.9, rel=REL)
    assert raw["reactivity_bias"] == pytest.approx(0.08020178742266772, rel=REL)
    assert raw["representational_separability"] == pytest.approx(0.8, rel=REL)
    assert raw["self_pattern_stability"] == pytest.approx(0.1212651089525556, rel=REL)
    assert raw["internal_state_coupling"] == 0.0
    assert srpi["value"] == 0.0 and srpi["mode"] == "legacy"
