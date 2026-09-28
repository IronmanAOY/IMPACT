"""
Hunter IIM surrogate calibration, provenance and device Psi kernel (stream I2).

The Hunter campaign turns compute_IIM's surrogate-null calibration into extra
campaign runs (K circular-shift surrogates per real run, seeded and recorded).
The reducer must reproduce the local calibration exactly: same surrogates, same
cut sample, same null statistic (Delta_Psi) and the same fields
(IIM_null_mean/sd, IIM_z, IIM_null_p, IIM_excess, canonical_calibrated).
Shards on an accelerator target must run their Psi work through the
array-module kernel (a NumPy-backed fake ROCm CuPy stands in for the MI300A).
"""

import csv
import dataclasses
import json
import os
import sys
import types

import numpy as np
import pytest

from impact_pipeline import hardware_backend as hb
from impact_pipeline import hunter_iim
from impact_pipeline import mpc_metrics as mm
from impact_pipeline.execution_profiles import get_execution_profile
from impact_pipeline.hunter_iim import (
    collect_iim_results_by_path,
    prepare_hunter_campaign,
)

K_NULL = 3
IIM_KW = dict(bins=2, lag_trs=1, n_parts=3, max_nodes=4, max_mechanism_size=2,
              max_purview_size=2)
NULL_FIELDS = (
    "value",
    "canonical_calibrated",
    "IIM_calibrated",
    "IIM_raw",
    "Delta_Psi",
    "Delta_Psi_null_mean",
    "Delta_Psi_null_sd",
    "IIM_null_mean",
    "IIM_null_sd",
    "IIM_z",
    "IIM_null_p",
    "IIM_excess",
    "raw",
    "canonical",
    "Psi_full",
    "Psi_mip_preserved",
)


@pytest.fixture(autouse=True)
def clean_hunter_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(("IMPACT_HUNTER_", "IMPACT_IIM_", "PMI_LOCAL_RANK", "PBS_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("IMPACT_REPO_ROOT", raising=False)
    monkeypatch.setenv("IMPACT_IIM_CACHE_DIR", str(tmp_path / "node_local"))


def _coupled_run(t, seed):
    """Four nodes, a lagged 0->1->2 chain plus an independent node."""
    rng = np.random.RandomState(seed)
    x = rng.randn(4, t)
    x[1, 1:] += 0.9 * x[0, :-1]
    x[2, 1:] += 0.9 * x[1, :-1]
    return x


def _campaign(tmp_path, *, n_null=K_NULL, target="cpu", null_min_shift=None,
              n_runs=2, t=64, **extra):
    data_dir = tmp_path / "prep"
    paths = []
    for i, ses in enumerate(("awake", "deep")[:n_runs]):
        run_dir = data_dir / "s1" / ses / "audio"
        run_dir.mkdir(parents=True)
        p = run_dir / "s1_run-1_schaefer400_ts.npy"
        np.save(p, _coupled_run(t, 10 + i).T, allow_pickle=False)
        paths.append(p)
    profile = dataclasses.replace(
        get_execution_profile("hunter"),
        hunter_phase1_shards_per_run=2,
        hunter_cut_shards_per_run=2,
        hunter_phase1_workers_per_task=1,
        hunter_shared_memory=False,
    )
    campaign_dir = tmp_path / "campaign"
    manifest = prepare_hunter_campaign(
        data_dir=data_dir,
        atlas="schaefer400",
        sessions=("awake", "deep")[:n_runs],
        condition="audio",
        stimulus_onsets=None,
        subjects=None,
        campaign_dir=campaign_dir,
        execution_profile=profile,
        iim_bins=IIM_KW["bins"],
        iim_lag_trs=IIM_KW["lag_trs"],
        iim_n_parts=IIM_KW["n_parts"],
        iim_max_timepoints=None,
        iim_max_nodes=IIM_KW["max_nodes"],
        iim_max_mechanism_size=IIM_KW["max_mechanism_size"],
        iim_max_purview_size=IIM_KW["max_purview_size"],
        step2_context={"hardware_target": target},
        hardware_target=target,
        iim_null_surrogates=n_null,
        iim_null_min_shift=null_min_shift,
        **extra,
    )
    return campaign_dir, manifest, paths


def _run_all(campaign_dir, manifest):
    for i in range(len(manifest["phase1_tasks"])):
        hunter_iim.run_phase1_shard(campaign_dir, i)
    for i in range(len(manifest["cut_tasks"])):
        hunter_iim.run_cut_shard(campaign_dir, i)
    hunter_iim.run_reduce_all(campaign_dir)
    return collect_iim_results_by_path(campaign_dir)


def _local(path, **kwargs):
    return mm.compute_IIM(np.load(path).T, return_details=True, **IIM_KW, **kwargs)


def _assert_same_calibration(hunter, local):
    assert hunter["defined"] is True and local["defined"] is True
    for key in NULL_FIELDS:
        h, v = hunter[key], local[key]
        if isinstance(v, float) and np.isnan(v):
            assert h is None or np.isnan(h), key
        else:
            assert h == pytest.approx(v, rel=1e-9, abs=1e-12), key
    assert hunter["Delta_Psi_null"] == pytest.approx(
        local["Delta_Psi_null"], rel=1e-9, abs=1e-12
    )
    for key in (
        "IIM_null_n",
        "IIM_null_method",
        "IIM_null_seed",
        "IIM_null_min_shift",
        "IIM_null_failed",
        "IIM_null_undefined_reason",
        "IIM_null_calibrated",
        "mip_cut",
    ):
        got = hunter[key]
        want = local[key]
        if key == "mip_cut":
            want = None if want is None else [list(want[0]), list(want[1])]
        assert got == want, key


def test_hunter_surrogate_calibration_matches_local_compute_iim(tmp_path):
    campaign_dir, manifest, paths = _campaign(tmp_path)
    real = [r for r in manifest["runs"] if not r["is_null_surrogate"]]
    nulls = [r for r in manifest["runs"] if r["is_null_surrogate"]]
    assert len(real) == 2 and len(nulls) == 2 * K_NULL
    results = _run_all(campaign_dir, manifest)
    assert set(results) >= {str(p.resolve()) for p in paths}
    for p in paths:
        hunter = results[str(p.resolve())]
        local = _local(p, null_surrogates=K_NULL)
        _assert_same_calibration(hunter, local)
        assert hunter["IIM_null_n"] == K_NULL
        # calibrated canonical value is the returned value (as in compute_IIM)
        assert hunter["value"] == pytest.approx(hunter["canonical_calibrated"])
    # the table written next to the step-2 outputs carries the same numbers
    with open(campaign_dir / "iim_results.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    for row in rows:
        hunter = results[row["ts_path"]]
        assert float(row["IIM_z"]) == pytest.approx(hunter["IIM_z"], rel=1e-12)
        assert row["tpm_estimator"] == "node_shrinkage"
        assert row["iim_algorithm_version"] == mm.IIM_ALGORITHM_VERSION


def test_surrogate_runs_are_seeded_recorded_and_identical_to_compute_iim(tmp_path):
    campaign_dir, manifest, paths = _campaign(tmp_path, n_runs=1)
    real = manifest["runs"][0]
    info = real["iim_null"]
    assert info["n_surrogates"] == K_NULL
    assert info["method"] == "circular_shift"
    assert info["seed"] == 0  # compute_IIM's default (null_seed=None, rng=0)
    prep = mm.prepare_iim_problem(
        np.load(paths[0]).T, bins=2, lag_trs=1, n_parts=3, max_nodes=4,
        max_mechanism_size=2, max_purview_size=2,
    )
    t_sel = prep["ts_selected"].shape[1]
    assert info["min_shift"] == mm.iim_null_min_shift(None, 1, t_sel)
    expected = mm.iim_null_surrogate_series(
        prep["ts_selected"], K_NULL, "circular_shift", seed=0,
        min_shift=info["min_shift"],
    )
    assert len(info["run_keys"]) == K_NULL
    by_key = {r["run_key"]: r for r in manifest["runs"]}
    for k, key in enumerate(info["run_keys"]):
        null_meta = by_key[key]
        assert null_meta["is_null_surrogate"] is True
        assert null_meta["null_of"] == real["run_key"]
        assert null_meta["null_index"] == k
        assert null_meta["null_seed"] == 0
        assert null_meta["node_selection_rule"] == "index"
        assert null_meta["state_budget_policy"] == "error"
        stored = np.load(null_meta["surrogate_ts_path"])
        np.testing.assert_array_equal(stored, expected[k])
        on_disk = json.loads(
            (campaign_dir / "runs" / key / "meta.json").read_text(encoding="utf-8")
        )
        assert on_disk["null_of"] == real["run_key"]
    # surrogate shards are ordinary campaign tasks
    task_runs = {t["run_key"] for t in manifest["cut_tasks"]}
    assert set(info["run_keys"]) <= task_runs
    real_meta = json.loads(
        (campaign_dir / "runs" / real["run_key"] / "meta.json").read_text()
    )
    assert real_meta["iim_null"]["run_keys"] == info["run_keys"]


def test_meta_records_estimator_version_node_selection_and_budget(tmp_path):
    data_dir = tmp_path / "prep"
    run_dir = data_dir / "s1" / "awake" / "audio"
    run_dir.mkdir(parents=True)
    rng = np.random.RandomState(0)
    # 8 nodes x 3 bins = 6561 states > max_state_space: bins are reduced (logged)
    np.save(run_dir / "s1_run-1_schaefer400_ts.npy", rng.randn(60, 8))
    profile = dataclasses.replace(
        get_execution_profile("hunter"),
        hunter_phase1_shards_per_run=1,
        hunter_cut_shards_per_run=1,
        hunter_phase1_workers_per_task=1,
    )
    manifest = prepare_hunter_campaign(
        data_dir=data_dir,
        atlas="schaefer400",
        sessions=("awake",),
        condition="audio",
        stimulus_onsets=None,
        subjects=None,
        campaign_dir=tmp_path / "campaign",
        execution_profile=profile,
        iim_bins=3,
        iim_lag_trs=1,
        iim_n_parts=2,
        iim_max_timepoints=None,
        iim_max_nodes=None,
        iim_max_mechanism_size=1,
        iim_max_purview_size=1,
        step2_context={},
        iim_tpm_estimator="per_unit",
        iim_node_selection="index",
    )
    meta = json.loads(
        (
            tmp_path / "campaign" / "runs" / manifest["runs"][0]["run_key"]
            / "meta.json"
        ).read_text()
    )
    assert meta["tpm_estimator"] == "per_unit"
    assert meta["iim_algorithm_version"] == mm.IIM_ALGORITHM_VERSION
    assert meta["node_selection_rule"] == "index"
    assert meta["selected_nodes"] == list(range(8))
    assert meta["bins_requested"] == 3 and meta["bins_used"] == 2
    assert meta["budget_adjustments"] and "bins 3->2" in meta["budget_adjustments"][0]
    assert meta["state_budget_policy"] == "reduce_bins_first"
    assert meta["iim_settings"]["tpm_estimator"] == "per_unit"
    assert manifest["iim_algorithm_version"] == mm.IIM_ALGORITHM_VERSION
    assert manifest["iim_settings"]["null_surrogates"] == 0


def test_unavailable_surrogates_are_nan_with_the_local_reason(tmp_path):
    # min_shift above T/2 leaves no admissible circular shift.
    campaign_dir, manifest, paths = _campaign(
        tmp_path, n_runs=1, null_min_shift=40, t=64
    )
    assert manifest["runs"][0]["iim_null"]["run_keys"] == []
    hunter = _run_all(campaign_dir, manifest)[str(paths[0].resolve())]
    local = _local(paths[0], null_surrogates=K_NULL, null_min_shift=40)
    assert str(local["IIM_null_undefined_reason"]).startswith("surrogates_unavailable")
    assert hunter["IIM_null_undefined_reason"] == local["IIM_null_undefined_reason"]
    assert hunter["IIM_null_failed"] == local["IIM_null_failed"] == K_NULL
    assert np.isnan(local["value"]) and np.isnan(hunter["value"])


@pytest.mark.parametrize(
    "bearer,t,reason",
    [
        ([3], 64, "insufficient_bearer_nodes"),  # one bearer node
        ([2, 0, 1], 1, "insufficient_shape"),  # stops before the bearer check
    ],
)
def test_undefined_runs_record_the_requested_estimator_settings(
    tmp_path, bearer, t, reason
):
    # The problem is undefined before the estimator runs.
    campaign_dir, manifest, paths = _campaign(
        tmp_path, n_runs=1, t=t, iim_bearer_nodes=bearer,
        iim_tpm_estimator="per_unit", iim_cut_mode="directional",
    )
    meta = manifest["runs"][0]
    assert meta["defined"] is False
    assert meta["undefined_reason"] == reason
    assert manifest["phase1_tasks"] == [] and manifest["cut_tasks"] == []
    hunter = _run_all(campaign_dir, manifest)[str(paths[0].resolve())]
    local = _local(
        paths[0], null_surrogates=K_NULL, bearer_nodes=bearer,
        tpm_estimator="per_unit", cut_mode="directional",
    )
    assert local["undefined_reason"] == reason
    for key in (
        "defined",
        "undefined_reason",
        "tpm_estimator",
        "cut_mode",
        "bearer_nodes",
        "iim_algorithm_version",
        "IIM_null_undefined_reason",
        "IIM_null_seed",
        "IIM_null_n",
    ):
        assert hunter[key] == local[key], key
    assert "selected_nodes" not in meta  # never reached, not filled with defaults
    with open(campaign_dir / "iim_results.csv", encoding="utf-8") as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["tpm_estimator"] == "per_unit" and row["cut_mode"] == "directional"


def test_uncalibrated_campaign_keeps_the_null_schema(tmp_path):
    campaign_dir, manifest, paths = _campaign(tmp_path, n_null=0, n_runs=1)
    hunter = _run_all(campaign_dir, manifest)[str(paths[0].resolve())]
    local = _local(paths[0])
    assert hunter["IIM_null_calibrated"] is False and hunter["IIM_null_n"] == 0
    assert hunter["value"] == pytest.approx(local["value"], rel=1e-9)
    assert hunter["IIM_null_min_shift"] == local["IIM_null_min_shift"]
    assert hunter["IIM_raw"] == pytest.approx(local["IIM_raw"], rel=1e-9)


# ---------------------------------------------------------------------------
# Accelerator targets: the shards' Psi runs through the array-module kernel
# ---------------------------------------------------------------------------


def _fake_rocm_cupy():
    cp = types.ModuleType("cupy")
    cp.__version__ = "13.6.0+fake-rocm"
    cp.__getattr__ = lambda name: getattr(np, name)  # NumPy stands in for HIP
    cp.asnumpy = np.asarray
    cp.linalg = np.linalg
    cp.add = np.add
    cp.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            is_hip=lambda: True,
            getDeviceCount=lambda: 4,
            getDeviceProperties=lambda i: {"name": b"AMD Instinct MI300A"},
        )
    )
    return cp


@pytest.fixture
def fake_apu(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_rocm_cupy())
    monkeypatch.setattr(hb, "_EIGH_DEVICE_STATUS", {})
    monkeypatch.setattr(hb, "_THREAD_LIMITS_APPLIED", True)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(var, "1")
    monkeypatch.delenv(mm.IIM_PSI_KERNEL_ENV, raising=False)
    assert hb.resolve_hardware_backend("hunter-apu").accelerator


def test_accelerator_shards_run_psi_on_the_device_kernel(
    tmp_path, fake_apu, monkeypatch
):
    def no_host_kernel(**_kwargs):
        raise AssertionError("Psi must not run on the host kernel on hunter-apu")

    monkeypatch.setattr(hunter_iim, "_compute_psi_for_problem", no_host_kernel)
    campaign_dir, manifest, paths = _campaign(tmp_path, target="hunter-apu", n_runs=1)
    results = _run_all(campaign_dir, manifest)
    run_key = manifest["runs"][0]["run_key"]
    run_dir = campaign_dir / "runs" / run_key
    p1 = json.loads((run_dir / "phase1_shards" / "shard_0000.json").read_text())
    cut = json.loads((run_dir / "cut_shards" / "shard_0000.json").read_text())
    assert p1["psi_kernel"] == "xp" and cut["psi_kernel"] == "xp"
    assert "hunter-apu" in p1["timing"]["hardware_backend"]
    hunter = results[str(paths[0].resolve())]
    assert hunter["psi_kernel"] == "xp"
    # same numbers as the local CPU reference (numba kernel)
    local = _local(paths[0], null_surrogates=K_NULL, psi_kernel="numba")
    _assert_same_calibration(hunter, local)
    # the generated phase-1 and cut jobs both request the accelerator
    for name in ("01_phase1_shards.pbs", "02_cut_shards.pbs"):
        text = (campaign_dir / "pbs" / name).read_text()
        assert "--hardware-target hunter-apu" in text, name


def test_psi_kernel_can_be_forced_back_to_the_host_kernel(
    tmp_path, fake_apu, monkeypatch
):
    monkeypatch.setenv(mm.IIM_PSI_KERNEL_ENV, "numba")
    campaign_dir, manifest, _paths = _campaign(
        tmp_path, target="hunter-apu", n_runs=1, n_null=0
    )
    out = hunter_iim.run_phase1_shard(campaign_dir, 0)
    assert out["psi_kernel"] == "numba"


def test_slurm_phase1_jobs_of_an_accelerator_campaign_run_on_cpu_nodes(
    tmp_path, monkeypatch
):
    """
    The optional Slurm backend runs phase-1 shards on its CPU partition with
    --hardware-target cpu: the shard must honour the job's target instead of
    requiring the campaign's accelerator there.
    """
    import importlib

    from impact_pipeline.hardware_backend import HardwareBackendError

    real_import = importlib.import_module

    def no_cupy(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("no CuPy on the CPU partition")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", no_cupy)
    monkeypatch.setenv("IMPACT_HUNTER_SCHEDULER", "slurm")
    campaign_dir, manifest, _paths = _campaign(
        tmp_path, target="gpu", n_runs=1, n_null=0, build_hardware_backend="cpu"
    )
    p1 = (campaign_dir / "slurm" / "01_phase1_shards.sbatch").read_text()
    assert "--hardware-target cpu --hunter-stage phase1-shard" in p1
    out = hunter_iim.run_phase1_shard(campaign_dir, 0, hardware_target="cpu")
    assert out["psi_kernel"] == "numba"
    assert "cpu" in out["timing"]["hardware_backend"]
    packed = hunter_iim.run_packed_shard(
        campaign_dir, "phase1-shard", 1, shards_per_node=1, hardware_target="cpu"
    )
    assert packed["psi_kernel"] == "numba"
    # without an explicit job target the campaign's target is still strict
    with pytest.raises(HardwareBackendError):
        hunter_iim.run_phase1_shard(campaign_dir, 0)


def test_run_pipeline_passes_the_job_target_to_the_shards(tmp_path, monkeypatch):
    import run_pipeline

    seen = []
    monkeypatch.setattr(
        run_pipeline,
        "run_phase1_shard",
        lambda campaign_dir, index, **kw: seen.append(("p1", index, kw)),
    )
    monkeypatch.setattr(
        run_pipeline,
        "run_cut_shard",
        lambda campaign_dir, index, **kw: seen.append(("cut", index, kw)),
    )
    monkeypatch.setattr(
        run_pipeline,
        "run_packed_shard",
        lambda campaign_dir, stage, index, spn, **kw: seen.append((stage, index, kw)),
    )
    bids = tmp_path / "bids"
    bids.mkdir()
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "tiny", "BIDSVersion": "1.8.0"})
    )
    common = dict(
        dataset_id="ds003171",
        bids_root_override=str(bids),
        execution_mode="hunter",
        hunter_campaign_dir=str(tmp_path / "campaign"),
        hardware_target="cpu",
        mpc_metrics=["IIM"],
        compute_ci=False,
    )
    out = str(tmp_path / "out")
    run_pipeline.main(out, hunter_stage="phase1-shard", hunter_task_index=2, **common)
    run_pipeline.main(out, hunter_stage="cut-shard", hunter_task_index=3, **common)
    run_pipeline.main(
        out, hunter_stage="phase1-shard", hunter_array_index=1, **common
    )
    assert [kw.get("hardware_target") for _s, _i, kw in seen] == ["cpu"] * 3
    assert [(s, i) for s, i, _kw in seen] == [
        ("p1", 2),
        ("cut", 3),
        ("phase1-shard", 1),
    ]


# ---------------------------------------------------------------------------
# End to end through run_pipeline.main (--hunter-iim-null-surrogates)
# ---------------------------------------------------------------------------


def test_main_build_with_null_surrogates_and_finalize_table(tmp_path, monkeypatch):
    import run_pipeline

    bids = tmp_path / "bids"
    out = tmp_path / "out"
    for subj in ("01",):
        func = bids / f"sub-{subj}" / "func"
        func.mkdir(parents=True)
        (func / f"sub-{subj}_task-audioawake_run-01_bold.json").write_text(
            json.dumps({"RepetitionTime": 2.0})
        )
        for i, ses in enumerate(("awake", "deep")):
            d = out / "preprocessed" / subj / ses / "audio"
            d.mkdir(parents=True)
            np.save(d / f"{subj}_run-1_schaefer400_ts.npy", _coupled_run(60, i).T)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "tiny", "BIDSVersion": "1.8.0"})
    )
    monkeypatch.setattr(
        run_pipeline,
        "compute_baseline_metrics",
        lambda df_mean, **k: df_mean.assign(
            mean_conn=0.1, modularity=0.2, pci_fmri=0.3
        ),
    )
    monkeypatch.setattr(run_pipeline, "bootstrap_ci", lambda *a, **k: (0.0, 1.0))
    monkeypatch.setattr(
        run_pipeline, "permutation_test_auc", lambda *a, **k: (0.5, 1.0)
    )
    monkeypatch.setattr(run_pipeline, "compare_models", lambda *a, **k: {})
    monkeypatch.setattr(run_pipeline, "create_doc", lambda **k: None)
    common = dict(
        dataset_id="ds003171",
        bids_root_override=str(bids),
        execution_mode="hunter",
        mpc_metrics=["IIM"],
        compute_ci=False,
        iim_max_nodes_override=4,
        iim_max_mechanism_size_override=2,
        iim_max_purview_size_override=2,
        iim_n_parts_override=3,
    )
    run_pipeline.main(
        str(out),
        hunter_stage="build-campaign",
        hunter_phase1_shards_per_run=1,
        hunter_cut_shards_per_run=1,
        hunter_workers_per_task=1,
        hunter_iim_null_surrogates=2,
        **common,
    )
    campaign = out / "cache" / "hunter_iim_campaign"
    manifest = json.loads((campaign / "campaign_manifest.json").read_text())
    assert manifest["iim_settings"]["null_surrogates"] == 2
    assert sum(r["is_null_surrogate"] for r in manifest["runs"]) == 4
    for i in range(len(manifest["phase1_tasks"])):
        hunter_iim.run_phase1_shard(campaign, i)
    for i in range(len(manifest["cut_tasks"])):
        hunter_iim.run_cut_shard(campaign, i)
    run_pipeline.main(
        str(out), hunter_stage="reduce-all", hunter_campaign_dir=str(campaign), **common
    )
    run_pipeline.main(
        str(out),
        hunter_stage="finalize-pipeline",
        hunter_campaign_dir=str(campaign),
        **common,
    )
    with open(out / "cache" / "hunter_iim_results.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    for row in rows:
        local = mm.compute_IIM(
            np.load(row["ts_path"]).T,
            bins=3,
            lag_trs=1,
            n_parts=3,
            max_nodes=4,
            max_mechanism_size=2,
            max_purview_size=2,
            null_surrogates=2,
            return_details=True,
        )
        assert float(row["canonical_calibrated"]) == pytest.approx(
            local["canonical_calibrated"], rel=1e-9, abs=1e-12
        )
        assert float(row["IIM_null_mean"]) == pytest.approx(
            local["IIM_null_mean"], rel=1e-9, abs=1e-12
        )
        assert row["IIM_null_seed"] == "0"
        assert row["IIM_null_method"] == "circular_shift"
    prov = json.loads((out / "cache" / "provenance_manifest.json").read_text())
    assert prov["status"] == "completed"
