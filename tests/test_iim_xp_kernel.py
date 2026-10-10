"""
Array-module (xp) IIM Psi kernel.

``impact_pipeline.iim_xp`` re-implements the phase-1 Psi contribution and the
cut-Psi evaluation with NumPy/CuPy array operations (vectorised over
mechanism states and purviews) so the Psi work can run on the Hunter MI300A.
On NumPy it must agree with the numba/host reference kernel
(``mpc_metrics._iim_phase1_chunk_contribution``) to 1e-10. The selection by
hardware backend is exercised with a NumPy-backed fake ROCm CuPy module.
"""

import glob
import json
import os
import sys
import tempfile

import numpy as np
import pytest

from impact_pipeline import hardware_backend as hb
from impact_pipeline import iim_xp
from impact_pipeline import mpc_metrics as mm
from _helpers import _fake_rocm_cupy

TOL = 1e-10


def _random_problem(seed, n, base, n_obs=50, state_by_node=False):
    rng = np.random.default_rng(seed)
    n_states = base ** n
    if state_by_node:
        probs = rng.random((n_states, n, base)) ** 2
        probs /= probs.sum(axis=2, keepdims=True)
        tpm = mm.iim_tpm_from_unit_probabilities(probs, base=base)
    else:
        tpm = rng.random((n_states, n_states)) ** 3
        tpm /= tpm.sum(axis=1, keepdims=True)
    states = iim_xp.state_table(n, base)
    curr = states[rng.integers(0, n_states, size=n_obs)]
    return tpm, states, curr


CASES = [
    # (seed, n, base, max_mechanism_size, max_purview_size, state_by_node)
    (0, 2, 2, 2, 2, False),
    (1, 3, 2, 3, 3, False),
    (2, 3, 3, 3, 2, True),
    (3, 4, 2, 4, 4, False),
    (4, 4, 2, 2, 3, True),
    (5, 4, 3, 2, 2, False),
    (6, 5, 2, 3, 2, True),
]


@pytest.mark.parametrize("seed,n,base,ms,ps,sbn", CASES)
def test_psi_contribution_matches_numba_kernel(seed, n, base, ms, ps, sbn):
    tpm, states, curr = _random_problem(seed, n, base, state_by_node=sbn)
    nodes = tuple(range(n))
    mechs = tuple(mm._iim_enumerate_subsets(nodes, ms))
    purv = tuple(mm._iim_enumerate_subsets(nodes, ps))
    ref = mm._iim_phase1_chunk_contribution(mechs, purv, base, tpm, curr, states)
    assert ref > 0.0
    for budget in (None, 7, 1 << 10):  # default, tiny and mid batch sizes
        out = iim_xp.psi_contribution(
            mechs, purv, base, tpm, curr, states, max_elements=budget
        )
        assert out == pytest.approx(ref, rel=TOL, abs=TOL)
    # chunked over mechanisms (as the Hunter phase-1 shards do)
    parts = [
        iim_xp.psi_contribution(mechs[i:i + 3], purv, base, tpm, curr, states)
        for i in range(0, len(mechs), 3)
    ]
    assert sum(parts) == pytest.approx(ref, rel=TOL, abs=TOL)


@pytest.mark.parametrize("seed,n,base", [(10, 3, 2), (11, 4, 2), (12, 3, 3)])
@pytest.mark.parametrize("cut_mode", ["bidirectional", "directional"])
def test_cut_psi_matches_numba_kernel_for_every_cut(seed, n, base, cut_mode):
    tpm, states, curr = _random_problem(seed, n, base, state_by_node=True)
    mechs = tuple(mm._iim_enumerate_subsets(tuple(range(n)), n))
    workspace = iim_xp.IIMXpWorkspace(np, states, base)
    for part_a, part_b in mm._iim_all_system_cuts(n, "all", cut_mode):
        host_cut = mm._iim_build_cut_tpm_for_mode(
            tpm, states, base, part_a, part_b, cut_mode=cut_mode
        )
        xp_cut = iim_xp.cut_tpm(tpm, n, base, part_a, part_b, cut_mode)
        np.testing.assert_allclose(xp_cut, host_cut, atol=1e-14)
        ref = mm._iim_phase1_chunk_contribution(
            mechs, mechs, base, host_cut, curr, states
        )
        out = iim_xp.psi_contribution(
            mechs, mechs, base, xp_cut, curr, states, workspace=workspace
        )
        assert out == pytest.approx(ref, rel=TOL, abs=TOL)


def test_observation_weights_match_numba_kernel_and_repeated_rows():
    tpm, states, _curr = _random_problem(20, 3, 2)
    rng = np.random.default_rng(20)
    counts = rng.integers(0, 4, size=states.shape[0])
    mechs = tuple(mm._iim_enumerate_subsets((0, 1, 2), 3))
    # integer weights on the state table == repeating each state that often
    repeated = np.repeat(states, counts, axis=0)
    ref = mm._iim_phase1_chunk_contribution(mechs, mechs, 2, tpm, repeated, states)
    weighted_numba = mm._iim_phase1_chunk_contribution(
        mechs, mechs, 2, tpm, states, states, obs_weights=counts.astype(float)
    )
    weighted_xp = iim_xp.psi_contribution(
        mechs, mechs, 2, tpm, states, states, obs_weights=counts.astype(float)
    )
    assert weighted_numba == pytest.approx(ref, rel=1e-12, abs=1e-14)
    assert weighted_xp == pytest.approx(ref, rel=TOL, abs=TOL)


def test_workspace_cache_eviction_keeps_results():
    tpm, states, curr = _random_problem(30, 4, 2)
    mechs = tuple(mm._iim_enumerate_subsets(tuple(range(4)), 4))
    ref = iim_xp.psi_contribution(mechs, mechs, 2, tpm, curr, states)
    tiny = iim_xp.IIMXpWorkspace(np, states, 2, cache_elements=40)
    assert iim_xp.psi_contribution(
        mechs, mechs, 2, tpm, curr, states, workspace=tiny
    ) == pytest.approx(ref, rel=1e-13)


def test_size_one_mechanisms_and_purviews_contribute_zero():
    tpm, states, curr = _random_problem(31, 3, 2)
    singles = ((0,), (1,), (2,))
    assert iim_xp.psi_contribution(singles, singles, 2, tpm, curr, states) == 0.0
    pairs = ((0, 1), (1, 2))
    assert iim_xp.psi_contribution(pairs, singles, 2, tpm, curr, states) == 0.0


# ---------------------------------------------------------------------------
# compute_IIM with the xp kernel
# ---------------------------------------------------------------------------


def _coupled(seed, n=4, t=160):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, t)
    for i in range(1, n):
        x[i, 1:] += 0.7 * x[i - 1, :-1]
    return x


@pytest.mark.parametrize("cut_mode", ["bidirectional", "directional"])
def test_compute_iim_xp_kernel_matches_numba_kernel(cut_mode):
    ts = _coupled(1)
    kw = dict(bins=2, return_details=True, cut_mode=cut_mode, null_surrogates=2)
    host = mm.compute_IIM(ts, psi_kernel="numba", **kw)
    dev = mm.compute_IIM(ts, psi_kernel="xp", **kw)
    assert host["psi_kernel"] == "numba" and dev["psi_kernel"] == "xp"
    assert dev["phase2_mode"] == "xp"
    for key in (
        "Psi_full",
        "Psi_mip_preserved",
        "raw",
        "canonical",
        "value",
        "IIM_null_mean",
        "IIM_null_sd",
        "IIM_excess",
    ):
        assert dev[key] == pytest.approx(host[key], rel=TOL, abs=TOL), key
    assert dev["Delta_Psi_null"] == pytest.approx(
        host["Delta_Psi_null"], rel=TOL, abs=TOL
    )


def test_xp_kernel_uses_no_sqlite_kernel_cache(tmp_path):
    pid_caches = os.path.join(
        tempfile.gettempdir(), f"iim_kernel_cache_{os.getpid()}_*"
    )
    before = set(glob.glob(pid_caches))
    info = mm.compute_IIM(
        _coupled(2, n=3), bins=2, psi_kernel="xp", return_details=True
    )
    assert info["induced_partition_cache_disposal"] == "unused"
    assert info["induced_partition_cache_path"] is None
    assert set(glob.glob(pid_caches)) == before


def test_xp_kernel_resumes_a_partial_numba_phase1_checkpoint(tmp_path):
    ts = _coupled(3)
    ckpt = tmp_path / "iim.json"
    ref = mm.compute_IIM(ts, bins=2, psi_kernel="numba", return_details=True)
    mm.compute_IIM(ts, bins=2, psi_kernel="numba", checkpoint_path=str(ckpt))
    # Rewrite the checkpoint as an interrupted phase 1 (first half of the
    # mechanisms done by the numba kernel, no cuts scored yet).
    prep = mm.prepare_iim_problem(ts, bins=2)
    mechs = prep["mechanisms_all"]
    done = len(mechs) // 2
    partial = mm._iim_phase1_chunk_contribution(
        mechs[:done],
        prep["purviews_all"],
        2,
        prep["tpm_full"],
        prep["curr_obs"],
        prep["states_full"],
    )
    payload = json.loads(ckpt.read_text())
    payload.update(
        status="running",
        psi_full=None,
        completed_cuts={},
        best={"psi_preserved_max": None, "mip_cut": None},
        phase1_mechanisms_done=done,
        phase1_total_mechanisms=len(mechs),
        phase1_psi_partial=partial,
    )
    ckpt.write_text(json.dumps(payload))
    # Kernels agree to rounding, so the xp kernel continues the numba prefix.
    resumed = mm.compute_IIM(
        ts, bins=2, psi_kernel="xp", checkpoint_path=str(ckpt), return_details=True
    )
    assert resumed["checkpoint_resumed"] is True
    assert resumed["phase1_resumed_partial"] is True
    assert resumed["Psi_full"] == pytest.approx(ref["Psi_full"], rel=TOL)
    assert resumed["Psi_mip_preserved"] == pytest.approx(
        ref["Psi_mip_preserved"], rel=TOL
    )


# ---------------------------------------------------------------------------
# Kernel selection by hardware backend
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_apu(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_rocm_cupy())
    monkeypatch.setattr(hb, "_EIGH_DEVICE_STATUS", {})
    monkeypatch.setattr(hb, "_THREAD_LIMITS_APPLIED", True)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(var, "1")
    monkeypatch.delenv(mm.IIM_PSI_KERNEL_ENV, raising=False)
    return hb.resolve_hardware_backend("hunter-apu")


def test_kernel_selection_by_backend_and_environment(monkeypatch, fake_apu):
    monkeypatch.delenv(mm.IIM_PSI_KERNEL_ENV, raising=False)
    assert mm.resolve_iim_psi_kernel("auto", "cpu") == "numba"
    assert mm.resolve_iim_psi_kernel("auto", fake_apu) == "xp"
    assert mm.resolve_iim_psi_kernel("host", fake_apu) == "numba"
    assert mm.resolve_iim_psi_kernel("xp", "cpu") == "xp"
    monkeypatch.setenv(mm.IIM_PSI_KERNEL_ENV, "xp")
    assert mm.resolve_iim_psi_kernel("auto", "cpu") == "xp"
    assert mm.resolve_iim_psi_kernel("numba", "cpu") == "numba"  # explicit wins
    monkeypatch.setenv(mm.IIM_PSI_KERNEL_ENV, "gpu-please")
    with pytest.raises(ValueError, match=mm.IIM_PSI_KERNEL_ENV):
        mm.resolve_iim_psi_kernel("auto", "cpu")
    with pytest.raises(ValueError, match="psi_kernel"):
        mm.resolve_iim_psi_kernel("cuda", "cpu")


def test_compute_iim_on_an_accelerator_backend_runs_the_xp_kernel(
    fake_apu, monkeypatch
):
    calls = {"xp": 0}
    real = iim_xp.psi_contribution

    def counting(*args, **kwargs):
        calls["xp"] += 1
        assert kwargs["xp"] is sys.modules["cupy"]
        return real(*args, **kwargs)

    monkeypatch.setattr(iim_xp, "psi_contribution", counting)
    ts = _coupled(4, n=3)
    dev = mm.compute_IIM(ts, bins=2, hardware_backend=fake_apu, return_details=True)
    host = mm.compute_IIM(ts, bins=2, return_details=True)
    assert dev["psi_kernel"] == "xp" and calls["xp"] > 0
    for key in ("Psi_full", "Psi_mip_preserved", "raw"):
        assert dev[key] == pytest.approx(host[key], rel=TOL, abs=TOL)


def test_exact_tpm_on_an_accelerator_backend(fake_apu):
    rng = np.random.default_rng(8)
    probs = rng.random((8, 3))
    tpm = mm.iim_tpm_from_unit_probabilities(probs)
    dev = mm.compute_IIM_from_tpm(tpm, hardware_backend=fake_apu, return_details=True)
    host = mm.compute_IIM_from_tpm(tpm, return_details=True)
    assert dev["psi_kernel"] == "xp" and host["psi_kernel"] == "numba"
    assert dev["Delta_Psi"] == pytest.approx(host["Delta_Psi"], rel=TOL, abs=TOL)
