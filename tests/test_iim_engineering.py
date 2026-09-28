"""Regression tests for IIM orchestration: parallel fallbacks, kernel caches,
checkpoint signatures, subsystem selection / state budget and numba caching."""

import concurrent.futures
import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm


def _ts(seed=3, n=4, t=200):
    return np.random.RandomState(seed).randn(n, t)


class _InProcessExecutor:
    """ProcessPoolExecutor stand-in that runs tasks synchronously in-process.

    ``fail_construct`` makes the n-th construction raise; ``fail_task`` is a
    callable (submit_index, fn, args) -> exception or None.
    """

    state = {}

    def __init__(self, max_workers=None, initializer=None, initargs=()):
        st = self.state
        st["constructed"] = st.get("constructed", 0) + 1
        if st["constructed"] in st.get("fail_construct", ()):
            raise RuntimeError("simulated pool start-up failure")
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, *args):
        st = self.state
        st["submitted"] = st.get("submitted", 0) + 1
        fut = concurrent.futures.Future()
        exc = st.get("fail_task", lambda *_: None)(st["submitted"], fn, args)
        if exc is not None:
            st.setdefault("failed_at", []).append(st["submitted"])
            fut.set_exception(exc)
        else:
            fut.set_result(fn(*args))
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        mm._iim_phase1_worker_cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


@pytest.fixture
def in_process_pool(monkeypatch):
    _InProcessExecutor.state = {}
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _InProcessExecutor)
    yield _InProcessExecutor.state
    mm._iim_phase1_worker_cleanup()


def test_parallel_fallback_does_not_double_count_psi(in_process_pool):
    ts = _ts()
    serial = mm.compute_IIM(ts, bins=2, return_details=True)
    # Reusable pool fails to start -> per-call pool; its 5th task crashes after
    # earlier (non-zero) chunks were already aggregated -> sequential fallback.
    in_process_pool["fail_construct"] = (1,)
    in_process_pool["fail_task"] = lambda i, fn, args: (
        RuntimeError("worker crash") if i == 5 else None
    )
    fallback = mm.compute_IIM(
        ts, bins=2, return_details=True, phase1_parallel_workers=2, phase1_chunk_size=2
    )
    assert in_process_pool.get("failed_at") == [5]
    assert fallback["phase_parallel_enabled"] is False
    assert any("per_call_pool" in msg for msg in fallback["phase_parallel_fallbacks"])
    assert fallback["Psi_full"] == pytest.approx(
        serial["Psi_full"], rel=1e-12, abs=1e-15
    )
    assert fallback["raw"] == pytest.approx(serial["raw"], rel=1e-12, abs=1e-15)


def test_kernel_cache_miss_does_not_disable_parallelism(in_process_pool):
    ts = _ts()
    serial = mm.compute_IIM(ts, bins=2, return_details=True)

    def _miss_once(i, fn, args):
        lookup_only = bool(args[5]) if len(args) > 5 else False
        if lookup_only and not in_process_pool.get("missed"):
            in_process_pool["missed"] = i
            return mm._IIMKernelCacheMissError("simulated missing kernel key")
        return None

    in_process_pool["fail_task"] = _miss_once
    info = mm.compute_IIM(
        ts, bins=2, return_details=True, phase1_parallel_workers=2, phase1_chunk_size=2
    )
    assert in_process_pool.get("missed")
    # Parallel execution continued after the miss (more tasks were submitted).
    assert in_process_pool["submitted"] > in_process_pool["missed"]
    assert info["phase2_cache_misses"] == 1
    assert info["phase_parallel_enabled"] is True
    assert info["phase_parallel_fallbacks"] == []
    for key in ("Psi_full", "Psi_mip_preserved", "raw"):
        assert info[key] == pytest.approx(serial[key], rel=1e-12, abs=1e-15)


def test_worker_flushes_kernel_cache_writes_per_chunk(tmp_path):
    ts = _ts()
    prep = mm.prepare_iim_problem(ts, bins=2)
    cache_path = tmp_path / "worker_cache.sqlite3"
    spec_curr = {"mode": "memmap", "path": str(tmp_path / "curr.npy")}
    spec_states = {"mode": "memmap", "path": str(tmp_path / "states.npy")}
    spec_tpm = {"mode": "memmap", "path": str(tmp_path / "tpm.npy")}
    np.save(spec_curr["path"], prep["curr_obs"])
    np.save(spec_states["path"], prep["states_full"])
    np.save(spec_tpm["path"], prep["tpm_full"])
    try:
        mm._iim_phase_worker_init_static(
            spec_curr, spec_states, prep["bins_used"], prep["purviews_all"]
        )
        cache_spec = {
            "enabled": True,
            "path": str(cache_path),
            "memory_entries": 10_000,
            "flush_batch": 100_000,
        }
        mm._iim_phase_worker_run_chunk_for_tpm(
            spec_tpm, prep["mechanisms_all"], cache_spec, None, True, False
        )
        # Without closing the worker's cache (pool workers skip atexit), a fresh
        # reader must already see the kernel values.
        reader = mm._IIMDiskKernelCache(str(cache_path), signature=None)
        n_rows = reader.conn.execute("SELECT COUNT(*) FROM kernel_cache").fetchone()[0]
        reader.close()
        assert n_rows > 0
    finally:
        mm._iim_phase1_worker_cleanup()


def test_checkpoint_is_not_reused_when_estimator_parameters_change(tmp_path):
    ts = _ts(seed=0, t=150)
    ckpt = str(tmp_path / "run.iim_checkpoint.json")
    kw = dict(bins=2, return_details=True, checkpoint_path=ckpt)
    first = mm.compute_IIM(ts, tpm_estimator="joint_laplace", tpm_alpha=1e-3, **kw)
    resumed = mm.compute_IIM(ts, tpm_estimator="joint_laplace", tpm_alpha=1e-3, **kw)
    assert resumed["checkpoint_resumed"] is True
    assert resumed["Psi_full"] == pytest.approx(first["Psi_full"])

    changed_alpha = mm.compute_IIM(
        ts, tpm_estimator="joint_laplace", tpm_alpha=0.5, **kw
    )
    fresh_alpha = mm.compute_IIM(
        ts, tpm_estimator="joint_laplace", tpm_alpha=0.5, bins=2, return_details=True
    )
    assert changed_alpha["checkpoint_resumed"] is False
    assert changed_alpha["Psi_full"] == pytest.approx(fresh_alpha["Psi_full"])
    assert changed_alpha["Psi_full"] != pytest.approx(first["Psi_full"])

    changed_estimator = mm.compute_IIM(
        ts, tpm_estimator="node_shrinkage", tpm_alpha=0.5, **kw
    )
    assert changed_estimator["checkpoint_resumed"] is False

    with open(ckpt, encoding="utf-8") as f:
        signature = json.load(f)["signature"]
    for key in (
        "iim_algorithm_version",
        "tpm_alpha",
        "tpm_estimator",
        "max_state_space",
        "bins_used",
        "lag_trs",
    ):
        assert key in signature


def test_temporary_kernel_cache_is_removed():
    tmp = tempfile.gettempdir()
    before = set(glob.glob(os.path.join(tmp, "iim_kernel_cache_*")))
    info = mm.compute_IIM(_ts(seed=1, t=120), bins=2, return_details=True)
    after = set(glob.glob(os.path.join(tmp, "iim_kernel_cache_*")))
    assert info["induced_partition_cache_disposal"] == "always"
    assert not os.path.exists(info["induced_partition_cache_path"])
    assert not (after - before)


def test_checkpoint_kernel_cache_removed_on_completion_and_explicit_cache_kept(
    tmp_path,
):
    ckpt = tmp_path / "run.iim_checkpoint.json"
    info = mm.compute_IIM(
        _ts(seed=1, t=120), bins=2, return_details=True, checkpoint_path=str(ckpt)
    )
    assert info["induced_partition_cache_disposal"] == "on_terminal"
    assert ckpt.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [ckpt.name]

    explicit = tmp_path / "explicit_cache.sqlite3"
    info2 = mm.compute_IIM(
        _ts(seed=1, t=120), bins=2, return_details=True, kernel_cache_path=str(explicit)
    )
    assert info2["induced_partition_cache_disposal"] == "keep"
    assert explicit.exists()


def test_explicit_kernel_cache_is_not_reused_across_different_data(tmp_path):
    # Kernel-cache keys do not identify the data, so a caller-provided cache
    # written for one run must be invalidated (signature check) before another
    # run uses it; previously stale cut-kernel values changed Psi^kappa.
    shared = str(tmp_path / "shared_cache.sqlite3")
    a, b = _ts(seed=0, t=150), _ts(seed=1, t=150)
    mm.compute_IIM(a, bins=2, return_details=True, kernel_cache_path=shared)
    via_shared = mm.compute_IIM(
        b, bins=2, return_details=True, kernel_cache_path=shared
    )
    fresh = mm.compute_IIM(b, bins=2, return_details=True)
    for key in ("Psi_full", "Psi_mip_preserved", "raw"):
        assert via_shared[key] == pytest.approx(fresh[key], rel=1e-12, abs=1e-15)


def test_null_calibration_resumes_from_checkpoints(tmp_path):
    ckpt = tmp_path / "run.iim_checkpoint.json"
    kw = dict(
        bins=2,
        return_details=True,
        checkpoint_path=str(ckpt),
        null_surrogates=3,
        null_seed=7,
    )
    first = mm.compute_IIM(_ts(seed=2, n=3, t=150), **kw)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 4  # observed + 3 surrogate checkpoints, no kernel caches left
    assert not [n for n in names if "sqlite" in n]
    second = mm.compute_IIM(_ts(seed=2, n=3, t=150), **kw)
    assert second["checkpoint_resumed"] is True
    assert second["Delta_Psi_null"] == pytest.approx(first["Delta_Psi_null"])
    assert second["canonical_calibrated"] == pytest.approx(
        first["canonical_calibrated"]
    )


def test_node_selection_is_explicit_and_degenerate_ranking_is_flagged(caplog):
    rng = np.random.RandomState(0)
    raw = rng.randn(12, 120)
    z = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)
    with caplog.at_level(logging.WARNING, logger="impact_pipeline.mpc_metrics"):
        info = mm.compute_IIM(z, bins=2, max_nodes=3, return_details=True)
    assert info["node_selection_rule"] == "variance"
    assert info["node_selection_degenerate"] is True
    assert info["selected_nodes"] == [0, 1, 2]
    assert any("degenerate" in rec.getMessage() for rec in caplog.records)

    scaled = raw * np.arange(1, 13)[:, None]
    by_var = mm.compute_IIM(scaled, bins=2, max_nodes=3, return_details=True)
    assert by_var["selected_nodes"] == [9, 10, 11]
    assert by_var["node_selection_degenerate"] is False

    by_index = mm.compute_IIM(
        scaled, bins=2, max_nodes=3, node_selection="index", return_details=True
    )
    assert by_index["selected_nodes"] == [0, 1, 2]
    explicit = mm.compute_IIM(
        scaled, bins=2, node_indices=[4, 7, 2], return_details=True
    )
    assert explicit["node_selection_rule"] == "explicit"
    assert explicit["selected_nodes"] == [2, 4, 7]


def test_variance_ties_are_relative_to_each_variance_not_to_the_largest():
    # One high-variance node must not make all other variances "tied": the
    # tolerance is relative to the variances being compared.
    rng = np.random.RandomState(0)
    raw = rng.randn(6, 400)
    z = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)
    x = z * np.sqrt([1e7, 1.0, 2.0, 3.0, 4.0, 5.0])[:, None]
    idx, info = mm._iim_select_nodes_with_info(x, 3)
    assert idx.tolist() == [0, 4, 5]
    assert info["degenerate_ranking"] is False

    # Zero-variance nodes rank below positive variances, above all-NaN nodes.
    y = np.vstack([z[:2], np.zeros((1, 400)), np.full((1, 400), np.nan)])
    idx, info = mm._iim_select_nodes_with_info(y, 3)
    assert idx.tolist() == [0, 1, 2]
    assert info["degenerate_ranking"] is False


def test_undefined_result_keeps_the_null_calibration_schema():
    calibrated = mm.compute_IIM(
        _ts(seed=0, n=3, t=100),
        bins=2,
        max_mechanism_size=1,
        return_details=True,
        null_surrogates=3,
    )
    assert calibrated["defined"] is False
    assert calibrated["IIM_null_undefined_reason"] == "observed_iim_undefined"
    assert calibrated["IIM_null_calibrated"] is False
    assert np.isnan(calibrated["value"])
    assert np.isnan(calibrated["canonical_calibrated"])
    plain = mm.compute_IIM(
        _ts(seed=0, n=3, t=100), bins=2, max_mechanism_size=1, return_details=True
    )
    assert plain["IIM_null_undefined_reason"] is None
    assert set(plain) == set(calibrated)


def test_state_budget_adjustments_are_recorded(caplog):
    ts = np.random.RandomState(0).randn(6, 150)
    with caplog.at_level(logging.WARNING, logger="impact_pipeline.mpc_metrics"):
        prep = mm.prepare_iim_problem(ts, bins=3, max_state_space=100)
    assert (prep["bins_requested"], prep["bins_used"], prep["n_nodes_used"]) == (
        3,
        2,
        6,
    )
    assert prep["budget_adjustments"] == ["bins 3->2 (3^6 > max_state_space=100)"]
    assert any("state budget" in rec.getMessage() for rec in caplog.records)

    nodes_first = mm.prepare_iim_problem(
        ts, bins=3, max_state_space=100, state_budget_policy="reduce_nodes_first"
    )
    assert (nodes_first["bins_used"], nodes_first["n_nodes_used"]) == (3, 4)
    assert len(nodes_first["budget_adjustments"]) == 2

    strict = mm.prepare_iim_problem(
        ts, bins=3, max_state_space=100, state_budget_policy="error"
    )
    assert strict["defined"] is False
    assert strict["undefined_reason"] == "state_space_too_large"

    within = mm.prepare_iim_problem(ts, bins=2, max_state_space=100)
    assert within["budget_adjustments"] == []


def test_numba_disk_cache_only_with_numba_cache_dir(tmp_path):
    src_root = Path(mm.__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from impact_pipeline import mpc_metrics as mm\n"
        "kind = type(mm._jsd_numba._cache).__name__ if mm.NUMBA_AVAILABLE else 'none'\n"
        "print(mm._NUMBA_DISK_CACHE, kind)\n"
    ) % str(src_root)
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_CACHE_DIR"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    flag, cache_type = out.stdout.split()
    assert flag == "False"
    assert cache_type in ("NullCache", "none")

    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba_cache")
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[0] == "True"


def test_requested_calibration_that_cannot_run_is_nan_not_raw():
    # min_shift larger than half the run: no circular-shift surrogate exists.
    info = mm.compute_IIM(
        _ts(seed=0, n=3, t=40),
        bins=2,
        return_details=True,
        null_surrogates=3,
        null_min_shift=30,
    )
    assert info["defined"] is True
    assert np.isfinite(info["canonical"])
    assert info["IIM_null_calibrated"] is False
    assert info["IIM_null_undefined_reason"].startswith("surrogates_unavailable")
    assert np.isnan(info["canonical_calibrated"])
    assert np.isnan(info["value"])
