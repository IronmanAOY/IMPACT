"""Declared bearer (``bearer_nodes``) for RAM/PDI/NAS/SRPI and header hygiene."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm


def test_resolve_bearer_nodes_validation():
    assert mm._resolve_bearer_nodes(None, 5) is None
    np.testing.assert_array_equal(mm._resolve_bearer_nodes([3, 0, 2], 5), [0, 2, 3])
    np.testing.assert_array_equal(
        mm._resolve_bearer_nodes(np.array([1.0, 4.0]), 5), [1, 4]
    )
    mask = np.array([True, False, True, False, False])
    np.testing.assert_array_equal(mm._resolve_bearer_nodes(mask, 5), [0, 2])
    for bad, msg in (
        ([], "at least one"),
        ([0, 0], "duplicate"),
        ([5], "out of range"),
        ([-1], "out of range"),
        ([0.5], "integer"),
        (["a"], "integer"),
        (np.array([True, False]), "length 5"),
    ):
        with pytest.raises(ValueError, match=msg):
            mm._resolve_bearer_nodes(bad, 5)


def _coupled_block_plus_noise(seed, n_time=1200, n_block=12, gain=0.4):
    """Nodes 0..11: hub (0..3) that receives from and returns to 4..11.
    Nodes 12..23: independent AR(1) noise (a second, disconnected module)."""
    rng = np.random.default_rng(seed)
    n = 2 * n_block
    hub, per = np.arange(4), np.arange(4, n_block)
    w_in = rng.standard_normal((hub.size, per.size)) / np.sqrt(per.size)
    w_out = rng.standard_normal((per.size, hub.size)) / 2.0
    x = np.zeros((n, n_time))
    for t in range(1, n_time):
        prev = x[:, t - 1]
        new = 0.6 * prev + rng.standard_normal(n)
        new[per] += gain * w_out @ prev[hub]
        new[hub] += gain * w_in @ prev[per]
        x[:, t] = new
    return x


def test_nas_capacity_is_restricted_to_the_declared_bearer():
    x = _coupled_block_plus_noise(0)
    loop = mm.compute_NAS(
        x,
        tr=1.0,
        mode="capacity",
        workspace_nodes=[0, 1, 2, 3],
        bearer_nodes=np.arange(12),
        return_details=True,
    )
    other = mm.compute_NAS(
        x,
        tr=1.0,
        mode="capacity",
        workspace_nodes=[12, 13, 14, 15],
        bearer_nodes=np.arange(12, 24),
        return_details=True,
    )
    assert loop["NAS_z"] > 3.0
    assert abs(other["NAS_z"]) < 3.0
    assert loop["bearer_nodes"] == list(range(12))
    assert other["workspace_nodes"] == [12, 13, 14, 15]
    # Same result as calling on the bearer rows with hub positions remapped.
    sub = mm.compute_NAS(
        x[12:],
        tr=1.0,
        mode="capacity",
        workspace_nodes=[0, 1, 2, 3],
        return_details=True,
    )
    assert sub["value"] == other["value"]
    with pytest.raises(ValueError, match="not part of bearer_nodes"):
        mm.compute_NAS(
            x,
            tr=1.0,
            mode="capacity",
            workspace_nodes=[0, 1],
            bearer_nodes=np.arange(12, 24),
        )


def test_nas_legacy_bearer_equals_subset():
    x = _coupled_block_plus_noise(1, n_time=300)
    kw = dict(
        tr=1.0,
        tau=0.2,
        bands=[(0.05, 0.2)],
        band_weights=[1.0],
        window_len=60,
        step_len=30,
    )
    bearer = [2, 5, 7, 9, 11, 13, 17, 20]
    a = mm.compute_NAS(x, bearer_nodes=bearer, return_details=True, **kw)
    b = mm.compute_NAS(x[bearer], return_details=True, **kw)
    assert a["value"] == b["value"]
    assert a["workspace_nodes"] == [bearer[i] for i in b["workspace_nodes"]]
    ws = mm.compute_NAS(x, bearer_nodes=bearer, workspace_nodes=[5, 9], **kw)
    assert ws == mm.compute_NAS(x[bearer], workspace_nodes=[1, 3], **kw)


def test_pdi_bearer_equals_subset_for_both_modes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 400)).cumsum(axis=1)
    base = [rng.standard_normal((10, 300)), rng.standard_normal((10, 350))]
    bearer = [1, 3, 4, 8]
    legacy = mm.compute_PDI(x, baseline_ts=base, bearer_nodes=bearer)
    assert legacy == mm.compute_PDI(x[bearer], baseline_ts=[b[bearer] for b in base])
    stacked = np.stack([base[0], base[0]])
    assert mm.compute_PDI(
        x, baseline_ts=stacked, bearer_nodes=bearer
    ) == mm.compute_PDI(x[bearer], baseline_ts=stacked[:, bearer])
    ex = mm.compute_PDI(
        x, mode="surrogate_excess", bearer_nodes=bearer, return_details=True
    )
    assert ex["value"] == mm.compute_PDI(x[bearer], mode="surrogate_excess")
    assert ex["bearer_nodes"] == bearer
    with pytest.raises(ValueError, match="same\\s+number of nodes"):
        mm.compute_PDI(x, baseline_ts=[b[bearer] for b in base], bearer_nodes=bearer)


def test_ram_bearer_equals_subset():
    rng = np.random.default_rng(2)
    n_regions, tr = 12, 0.1
    goal = 0.5 + 3.0 * np.arange(20)
    stim, fb = goal + 0.8, goal + 2.0
    ts = rng.standard_normal((n_regions, int((fb[-1] + 3.0) / tr)))
    for k, s in enumerate(stim):
        i = int(round(s / tr))
        ts[:6, i + 1 : i + 6] += 1.0 + 0.1 * k
    bundle = dict(
        onsets=stim.tolist(),
        goal_onsets=goal.tolist(),
        feedback_onsets=fb.tolist(),
        feedback_values=list(rng.random(20)),
    )
    kw = dict(
        tr=tr,
        stimulus_onsets=bundle,
        response_model="boxcar",
        response_window_sec=0.5,
        quality_null_samples=20,
        return_details=True,
    )
    a = mm.compute_RAM(ts, bearer_nodes=[0, 2, 4, 6, 8], **kw)
    b = mm.compute_RAM(ts[[0, 2, 4, 6, 8]], **kw)
    assert a["bearer_nodes"] == [0, 2, 4, 6, 8]
    assert a["magnitude_term"] == b["magnitude_term"]
    assert a["components"] == b["components"]
    assert mm.compute_RAM(ts, **kw)["bearer_nodes"] is None


def test_srpi_bearer_equals_subset_for_both_modes():
    rng = np.random.default_rng(3)
    ts = rng.standard_normal((9, 600))
    on = np.arange(20, 580, 14).astype(float)
    kw = dict(
        tr=1.0,
        pre_window_sec=2.0,
        response_lag_sec=2.0,
        response_window_sec=4.0,
        return_details=True,
    )
    bearer = [0, 1, 2, 5, 6]
    a = mm.compute_SRPI(
        ts, self_onsets=on[::2], nonself_onsets=on[1::2], bearer_nodes=bearer, **kw
    )
    b = mm.compute_SRPI(ts[bearer], self_onsets=on[::2], nonself_onsets=on[1::2], **kw)
    assert a["value"] == b["value"] and a["bearer_nodes"] == bearer
    rows = []
    for s, o in zip(on[::2], on[1::2]):
        rows.append(dict(onset=s, trial_type="self_caused", yoked_to=None, phase_bin=0))
        rows.append(dict(onset=o, trial_type="other_caused", yoked_to=s, phase_bin=0))
    ag = mm.compute_SRPI(
        ts,
        mode="agency",
        agency_events=rows,
        bearer_nodes=bearer,
        agency_null_permutations=20,
        **kw,
    )
    ref = mm.compute_SRPI(
        ts[bearer], mode="agency", agency_events=rows, agency_null_permutations=20, **kw
    )
    assert ag["value"] == ref["value"] and ag["bearer_nodes"] == bearer
    und = mm.compute_SRPI(
        ts, self_onsets=[], nonself_onsets=[], bearer_nodes=bearer, **kw
    )
    assert und["bearer_nodes"] == bearer and und["mode"] == "legacy"


def test_unused_header_imports_are_gone():
    for name in (
        "make_first_level_design_matrix",
        "run_glm",
        "mutual_info_regression",
        "mutual_info_score",
        "pd",
        "_safe_mutual_info_score",
    ):
        assert not hasattr(mm, name), name
    code = (
        "import sys, impact_pipeline.mpc_metrics; "
        "print('nilearn.glm.first_level' in sys.modules, "
        "'sklearn.feature_selection' in sys.modules)"
    )
    src = str(Path(mm.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.split() == ["False", "False"]
