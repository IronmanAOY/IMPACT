import dataclasses
import json
import os

import numpy as np
import pytest

from impact_pipeline import hunter_iim
from impact_pipeline.execution_profiles import get_execution_profile
from impact_pipeline.hunter_iim import (
    collect_iim_results_by_path,
    prepare_hunter_campaign,
    run_cut_reduce,
    run_cut_shard,
    run_phase1_reduce,
    run_phase1_shard,
)
from impact_pipeline import mpc_metrics as mm


@pytest.fixture(autouse=True)
def clean_hunter_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(("IMPACT_HUNTER_", "IMPACT_IIM_", "PMI_LOCAL_RANK", "PBS_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IMPACT_IIM_CACHE_DIR", str(tmp_path / "node_local"))


def test_hunter_iim_matches_direct_compute(tmp_path):
    data_dir = tmp_path / "prep"
    run_dir = data_dir / "s1" / "awake" / "audio"
    run_dir.mkdir(parents=True)

    rng = np.random.RandomState(5)
    ts_time_region = rng.rand(64, 4)
    ts_path = run_dir / "s1_run-1_schaefer400_ts.npy"
    np.save(ts_path, ts_time_region, allow_pickle=False)

    hunter_profile = dataclasses.replace(
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
        sessions=("awake",),
        condition="audio",
        stimulus_onsets=None,
        subjects=None,
        campaign_dir=campaign_dir,
        execution_profile=hunter_profile,
        iim_bins=2,
        iim_lag_trs=1,
        iim_n_parts=3,
        iim_max_timepoints=None,
        iim_max_nodes=4,
        iim_max_mechanism_size=2,
        iim_max_purview_size=2,
        step2_context={},
    )

    for i, _task in enumerate(manifest["phase1_tasks"]):
        run_phase1_shard(campaign_dir, i)
    run_phase1_reduce(campaign_dir, 0)
    for i, _task in enumerate(manifest["cut_tasks"]):
        run_cut_shard(campaign_dir, i)
    run_cut_reduce(campaign_dir, 0)

    hunter_map = collect_iim_results_by_path(campaign_dir)
    hunter_info = hunter_map[str(ts_path.resolve())]
    direct_info = mm.compute_IIM(
        ts_time_region.T,
        bins=2,
        lag_trs=1,
        n_parts=3,
        max_nodes=4,
        max_mechanism_size=2,
        max_purview_size=2,
        return_details=True,
        phase1_parallel_workers=None,
    )

    assert hunter_info["defined"] is True
    assert direct_info["defined"] is True
    assert hunter_info["raw"] == pytest.approx(direct_info["raw"], rel=1e-9, abs=1e-9)
    assert hunter_info["canonical"] == pytest.approx(direct_info["canonical"], rel=1e-9, abs=1e-9)
    assert hunter_info["Psi_full"] == pytest.approx(direct_info["Psi_full"], rel=1e-9, abs=1e-9)
    assert hunter_info["Psi_mip_preserved"] == pytest.approx(
        direct_info["Psi_mip_preserved"],
        rel=1e-9,
        abs=1e-9,
    )


def test_hunter_slurm_scripts_embed_handoff_runtime(monkeypatch, tmp_path):
    # Hunter itself runs PBS Pro (default backend); Slurm stays available as an
    # optional generic backend selected explicitly.
    monkeypatch.setenv("IMPACT_HUNTER_SCHEDULER", "slurm")
    monkeypatch.setenv("IMPACT_HUNTER_CONDA_ENV", "impact-synergy-clean")
    monkeypatch.setenv("IMPACT_HUNTER_SLURM_ACCOUNT", "project123")
    monkeypatch.setenv("IMPACT_HUNTER_CPU_PARTITION", "cpu-test")
    monkeypatch.setenv("IMPACT_HUNTER_APU_PARTITION", "apu-test")
    monkeypatch.setenv(
        "IMPACT_HUNTER_SLURM_SETUP",
        "module load miniforge\nexport MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl",
    )

    data_dir = tmp_path / "prep"
    run_dir = data_dir / "s1" / "awake" / "audio"
    run_dir.mkdir(parents=True)
    ts_path = run_dir / "s1_run-1_schaefer400_ts.npy"
    np.save(ts_path, np.random.RandomState(11).rand(32, 3), allow_pickle=False)

    hunter_profile = dataclasses.replace(
        get_execution_profile("hunter"),
        hunter_phase1_shards_per_run=1,
        hunter_cut_shards_per_run=1,
        hunter_phase1_workers_per_task=1,
        hunter_shared_memory=False,
    )
    campaign_dir = tmp_path / "campaign"
    prepare_hunter_campaign(
        data_dir=data_dir,
        atlas="schaefer400",
        sessions=("awake",),
        condition="audio",
        stimulus_onsets=None,
        subjects=None,
        campaign_dir=campaign_dir,
        execution_profile=hunter_profile,
        iim_bins=2,
        iim_lag_trs=1,
        iim_n_parts=2,
        iim_max_timepoints=None,
        iim_max_nodes=3,
        iim_max_mechanism_size=2,
        iim_max_purview_size=2,
        step2_context={"hardware_target": "cpu"},
    )

    phase1_script = (campaign_dir / "slurm" / "01_phase1_shards.sbatch").read_text(
        encoding="utf-8"
    )
    cut_script = (campaign_dir / "slurm" / "03_cut_shards.sbatch").read_text(
        encoding="utf-8"
    )
    submit_script = (campaign_dir / "slurm" / "00_submit_all.sh").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --account=project123" in phase1_script
    assert "#SBATCH --partition=cpu-test" in phase1_script
    assert "#SBATCH --partition=cpu-test" in cut_script
    assert "module load miniforge" in phase1_script
    # --no-capture-output streams job logs instead of buffering them until exit.
    assert (
        "conda run --no-capture-output -n impact-synergy-clean python" in phase1_script
    )
    assert 'campaign_dir="$(cd "${script_dir}/.." && pwd)"' in submit_script
    # Site setup runs before nounset is enabled.
    assert phase1_script.index("module load miniforge") < phase1_script.index(
        "\nset -u\n"
    )


# ---------------------------------------------------------------------------
# Packed PBS execution, idempotency, checkpoints and the end-to-end flow
# ---------------------------------------------------------------------------


def _multi_run_campaign(tmp_path, *, phase1=3, cut=2, n_parts=4, max_nodes=4, bins=2):
    data_dir = tmp_path / "prep"
    rng = np.random.RandomState(11)
    paths = []
    for subj in ("s1", "s2"):
        for ses in ("awake", "deep"):
            run_dir = data_dir / subj / ses / "audio"
            run_dir.mkdir(parents=True)
            p = run_dir / f"{subj}_run-1_schaefer400_ts.npy"
            np.save(p, rng.rand(64, 5), allow_pickle=False)
            paths.append(p)
    profile = dataclasses.replace(
        get_execution_profile("hunter"),
        hunter_phase1_shards_per_run=phase1,
        hunter_cut_shards_per_run=cut,
        hunter_phase1_workers_per_task=1,
        hunter_shared_memory=False,
    )
    campaign_dir = tmp_path / "campaign"
    manifest = prepare_hunter_campaign(
        data_dir=data_dir,
        atlas="schaefer400",
        sessions=("awake", "deep"),
        condition="audio",
        stimulus_onsets=None,
        subjects=None,
        campaign_dir=campaign_dir,
        execution_profile=profile,
        iim_bins=bins,
        iim_lag_trs=1,
        iim_n_parts=n_parts,
        iim_max_timepoints=None,
        iim_max_nodes=max_nodes,
        iim_max_mechanism_size=2,
        iim_max_purview_size=2,
        step2_context={"hardware_target": "cpu"},
    )
    return campaign_dir, manifest, paths


def _run_all_packed(campaign_dir, manifest, stages=("phase1-shard", "cut-shard")):
    spn = int(manifest["scheduler"]["shards_per_node"])
    keys = {"phase1-shard": "phase1_tasks", "cut-shard": "cut_tasks"}
    for stage in stages:
        for array_index in range(
            hunter_iim.packed_array_size(len(manifest[keys[stage]]), spn)
        ):
            for rank in range(spn):
                hunter_iim.run_packed_shard(
                    campaign_dir,
                    stage,
                    array_index,
                    spn,
                    env={"PMI_LOCAL_RANK": str(rank)},
                )


def _direct(path, *, n_parts=4, max_nodes=4, bins=2):
    return mm.compute_IIM(
        np.load(path).T,
        bins=bins,
        lag_trs=1,
        n_parts=n_parts,
        max_nodes=max_nodes,
        max_mechanism_size=2,
        max_purview_size=2,
        return_details=True,
        phase1_parallel_workers=None,
    )


def test_packed_pbs_campaign_matches_direct_compute(tmp_path):
    campaign_dir, manifest, paths = _multi_run_campaign(tmp_path)
    assert manifest["scheduler"]["scheduler"] == "pbs"
    assert manifest["cut_requires_phase1"] is False
    # Phase-1 and cut shards are independent in the PBS DAG: run the cuts first.
    _run_all_packed(campaign_dir, manifest, stages=("cut-shard", "phase1-shard"))
    hunter_iim.run_reduce_all(campaign_dir)

    results = collect_iim_results_by_path(campaign_dir)
    for p in paths:
        hunter_info = results[str(p.resolve())]
        direct = _direct(p)
        assert hunter_info["defined"] is True and direct["defined"] is True
        for key in ("raw", "canonical", "Psi_full", "Psi_mip_preserved"):
            assert hunter_info[key] == pytest.approx(
                direct[key], rel=1e-9, abs=1e-12
            ), key
        assert hunter_info["bins_used"] == direct["bins_used"]
        assert hunter_info["n_cuts_evaluated"] == direct["n_cuts_evaluated"]

    summary = hunter_iim.summarize_campaign_timing(campaign_dir)
    assert summary["phase1-shard"]["n_tasks"] == len(manifest["phase1_tasks"])
    assert summary["cut-shard"]["n_tasks"] == len(manifest["cut_tasks"])
    assert summary["reduce-all"]["n_tasks"] == 1
    shard = json.loads(
        (
            campaign_dir
            / "runs"
            / manifest["cut_tasks"][0]["run_key"]
            / "cut_shards"
            / "shard_0000.json"
        ).read_text()
    )
    assert shard["timing"]["wall_seconds"] >= 0 and shard["timing"]["host"]
    # node-local kernel caches are removed after each task
    cache_root = tmp_path / "node_local" / "impact_iim_kernels"
    assert not cache_root.exists() or not any(cache_root.iterdir())
    status = hunter_iim.campaign_status(campaign_dir)
    assert status["stages"]["cut-shard"]["missing"] == 0
    assert status["final_results"] == len(paths)


def test_shards_skip_when_complete_and_force_recomputes(tmp_path, monkeypatch):
    campaign_dir, manifest, _paths = _multi_run_campaign(tmp_path)
    _run_all_packed(campaign_dir, manifest)
    before_p1 = hunter_iim.run_phase1_shard(campaign_dir, 0)
    before_cut = hunter_iim.run_cut_shard(campaign_dir, 0)

    def boom(**_kwargs):
        raise AssertionError("completed shard must not be recomputed")

    monkeypatch.setattr(hunter_iim, "_compute_psi_for_problem", boom)
    assert (
        hunter_iim.run_phase1_shard(campaign_dir, 0)["psi_partial"]
        == before_p1["psi_partial"]
    )
    assert (
        hunter_iim.run_cut_shard(campaign_dir, 0)["best_psi"] == before_cut["best_psi"]
    )

    monkeypatch.setenv("IMPACT_HUNTER_FORCE", "1")
    with pytest.raises(AssertionError, match="must not be recomputed"):
        hunter_iim.run_phase1_shard(campaign_dir, 0)


def test_code_version_change_invalidates_completed_shards(tmp_path, monkeypatch):
    campaign_dir, _manifest, _paths = _multi_run_campaign(tmp_path)
    hunter_iim.run_phase1_shard(campaign_dir, 0)
    calls = {"n": 0}
    real = hunter_iim._compute_psi_for_problem

    def counting(**kwargs):
        calls["n"] += 1
        return real(**kwargs)

    monkeypatch.setattr(hunter_iim, "_compute_psi_for_problem", counting)
    hunter_iim.run_phase1_shard(campaign_dir, 0)
    assert calls["n"] == 0
    monkeypatch.setattr(hunter_iim, "_runtime_code_version", lambda: "different-commit")
    hunter_iim.run_phase1_shard(campaign_dir, 0)
    assert calls["n"] == 1


def test_cut_shard_resumes_from_per_cut_checkpoint(tmp_path, monkeypatch):
    campaign_dir, manifest, _paths = _multi_run_campaign(tmp_path, cut=1)
    reference = hunter_iim.run_cut_shard(campaign_dir, 0)
    n_cuts = len(reference["cut_scores"])
    assert n_cuts >= 2
    out_path = (
        campaign_dir
        / "runs"
        / manifest["cut_tasks"][0]["run_key"]
        / "cut_shards"
        / "shard_0000.json"
    )
    out_path.unlink()

    real = hunter_iim._compute_psi_for_problem
    calls = {"n": 0}

    def dies_on_second_cut(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("walltime")
        return real(**kwargs)

    monkeypatch.setattr(hunter_iim, "_compute_psi_for_problem", dies_on_second_cut)
    with pytest.raises(KeyboardInterrupt):
        hunter_iim.run_cut_shard(campaign_dir, 0)
    partial_path = out_path.with_name("shard_0000.partial.json")
    assert len(json.loads(partial_path.read_text())["cut_scores"]) == 1

    calls["n"] = 10  # no further interruptions
    resumed = hunter_iim.run_cut_shard(campaign_dir, 0)
    assert calls["n"] == 10 + (n_cuts - 1)
    assert resumed["timing"]["resumed_cuts"] == 1
    assert resumed["cut_scores"] == pytest.approx(reference["cut_scores"])
    assert resumed["best_cut"] == reference["best_cut"]
    assert not partial_path.exists()


def test_reducers_reject_stale_or_missing_shards(tmp_path):
    campaign_dir, manifest, _paths = _multi_run_campaign(tmp_path)
    _run_all_packed(campaign_dir, manifest)
    run_key = manifest["runs"][0]["run_key"]
    shard_path = campaign_dir / "runs" / run_key / "phase1_shards" / "shard_0000.json"
    rec = json.loads(shard_path.read_text())
    rec["identity"]["problem_digest"] = "stale"
    shard_path.write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="different campaign build"):
        hunter_iim.run_phase1_reduce(campaign_dir, 0)

    (
        campaign_dir
        / "runs"
        / manifest["runs"][1]["run_key"]
        / "cut_shards"
        / "shard_0000.json"
    ).unlink()
    with pytest.raises(RuntimeError) as excinfo:
        hunter_iim.run_reduce_all(campaign_dir)
    assert run_key in str(excinfo.value)
    assert manifest["runs"][1]["run_key"] in str(
        excinfo.value
    ) or "phase1-reduce" in str(excinfo.value)


def _rebuild_same_dir(tmp_path, *, max_nodes):
    profile = dataclasses.replace(
        get_execution_profile("hunter"),
        hunter_phase1_shards_per_run=3,
        hunter_cut_shards_per_run=2,
        hunter_phase1_workers_per_task=1,
        hunter_shared_memory=False,
    )
    return prepare_hunter_campaign(
        data_dir=tmp_path / "prep",
        atlas="schaefer400",
        sessions=("awake", "deep"),
        condition="audio",
        stimulus_onsets=None,
        subjects=None,
        campaign_dir=tmp_path / "campaign",
        execution_profile=profile,
        iim_bins=2,
        iim_lag_trs=1,
        iim_n_parts=4,
        iim_max_timepoints=None,
        iim_max_nodes=max_nodes,
        iim_max_mechanism_size=2,
        iim_max_purview_size=2,
        step2_context={"hardware_target": "cpu"},
    )


def test_rebuild_never_reuses_results_of_the_previous_build(tmp_path):
    """
    Rebuilding a campaign directory with other IIM settings must not let status
    or finalize treat the previous build's reduced results as current.
    """
    campaign_dir, manifest, _paths = _multi_run_campaign(tmp_path, max_nodes=4)
    _run_all_packed(campaign_dir, manifest)
    hunter_iim.run_reduce_all(campaign_dir)
    assert hunter_iim.campaign_status(campaign_dir)["final_results"] == 4
    old_final = campaign_dir / "runs" / manifest["runs"][0]["run_key"]
    old_final = json.loads((old_final / "final_result.json").read_text())
    assert old_final["n_nodes_used"] == 4

    rebuilt = _rebuild_same_dir(tmp_path, max_nodes=3)
    status = hunter_iim.campaign_status(campaign_dir)
    # nothing of the new build has run yet
    assert status["final_results"] == 0
    assert status["stages"]["phase1-shard"]["complete"] == 0
    assert status["stages"]["cut-shard"]["missing"] == len(rebuilt["cut_tasks"])
    with pytest.raises(FileNotFoundError, match="final IIM result"):
        collect_iim_results_by_path(campaign_dir)

    # a final result left over from another build is refused by finalize
    run0 = campaign_dir / "runs" / rebuilt["runs"][0]["run_key"]
    (run0 / "final_result.json").write_text(json.dumps(old_final))
    for other in rebuilt["runs"][1:]:
        (campaign_dir / "runs" / other["run_key"] / "final_result.json").write_text(
            json.dumps({**old_final, "problem_digest": other["problem_digest"]})
        )
    with pytest.raises(RuntimeError, match="current campaign build"):
        collect_iim_results_by_path(campaign_dir)

    _run_all_packed(campaign_dir, rebuilt)
    hunter_iim.run_reduce_all(campaign_dir)
    results = collect_iim_results_by_path(campaign_dir)
    assert {r["n_nodes_used"] for r in results.values()} == {3}
    assert hunter_iim.campaign_status(campaign_dir)["final_results"] == 4


def test_reducers_reject_shards_from_mixed_code_versions(tmp_path, monkeypatch):
    campaign_dir, manifest, _paths = _multi_run_campaign(tmp_path)
    run_key = manifest["runs"][0]["run_key"]
    first_p1 = [t for t in manifest["phase1_tasks"] if t["run_key"] == run_key]
    idx = [manifest["phase1_tasks"].index(t) for t in first_p1]
    hunter_iim.run_phase1_shard(campaign_dir, idx[0])
    # e.g. `git pull` while the phase-1 array is still running
    monkeypatch.setattr(hunter_iim, "_runtime_code_version", lambda: "other-commit")
    for i in idx[1:]:
        hunter_iim.run_phase1_shard(campaign_dir, i)
    with pytest.raises(RuntimeError, match="different code versions"):
        hunter_iim.run_phase1_reduce(campaign_dir, 0)

    # after a resubmission under one code version the run reduces again
    hunter_iim.run_phase1_shard(campaign_dir, idx[0])
    p1 = hunter_iim.run_phase1_reduce(campaign_dir, 0)
    assert p1["code_version"] == "other-commit"
    cut_idx = [
        i for i, t in enumerate(manifest["cut_tasks"]) if t["run_key"] == run_key
    ]
    for i in cut_idx:
        hunter_iim.run_cut_shard(campaign_dir, i)
    final = hunter_iim.run_cut_reduce(campaign_dir, 0)
    assert final["defined"] and final["code_version"] == "other-commit"

    # cut shards computed by another version than Psi_full are refused as well
    monkeypatch.setattr(hunter_iim, "_runtime_code_version", lambda: "third-commit")
    hunter_iim.run_cut_shard(campaign_dir, cut_idx[0])
    with pytest.raises(RuntimeError, match="different code versions"):
        hunter_iim.run_cut_reduce(campaign_dir, 0)


def _psi_problem():
    prep = mm.prepare_iim_problem(
        np.random.RandomState(0).rand(5, 200),
        bins=2,
        lag_trs=1,
        n_parts=2,
        rng=0,
        partition_mode="all",
        max_nodes=5,
        max_mechanism_size=2,
        max_purview_size=2,
    )
    return dict(
        tpm=prep["tpm_full"],
        curr_obs=prep["curr_obs"],
        states_full=prep["states_full"],
        base=int(prep["bins_used"]),
        mechanisms=prep["mechanisms_all"],
        purviews=prep["purviews_all"],
        phase1_chunk_size=2,
        phase1_shared_memory=False,
    )


def test_parallel_psi_with_the_shared_kernel_cache_does_not_fail(tmp_path):
    """
    Hunter shards run the Psi chunks in a worker pool that shares one SQLite
    kernel cache. Opening a new cache file from several workers at once raced
    on the WAL switch ("database is locked") in most runs, failing the shard.
    """
    problem = _psi_problem()
    reference = hunter_iim._compute_psi_for_problem(
        **problem, phase1_parallel_workers=1
    )
    for attempt in range(4):
        psi = hunter_iim._compute_psi_for_problem(
            **problem,
            phase1_parallel_workers=4,
            kernel_cache_path=str(tmp_path / f"kernel_{attempt}.sqlite3"),
        )
        assert psi == pytest.approx(reference, rel=1e-12, abs=1e-15)


def test_parallel_psi_recomputes_without_cache_when_sqlite_fails(
    tmp_path, monkeypatch, caplog
):
    class CorruptAfterCreation(mm._IIMDiskKernelCache):
        def close(self):
            super().close()
            (tmp_path / "kernel.sqlite3").write_bytes(b"not a database" * 64)

    problem = _psi_problem()
    reference = hunter_iim._compute_psi_for_problem(
        **problem, phase1_parallel_workers=1
    )
    monkeypatch.setattr(hunter_iim, "_IIMDiskKernelCache", CorruptAfterCreation)
    psi = hunter_iim._compute_psi_for_problem(
        **problem,
        phase1_parallel_workers=3,
        kernel_cache_path=str(tmp_path / "kernel.sqlite3"),
    )
    assert psi == pytest.approx(reference, rel=1e-12, abs=1e-15)
    assert "without the cache" in caplog.text


def test_shard_timing_counts_worker_process_cpu(tmp_path):
    """The Psi work runs in worker processes; their CPU time must be reported."""
    data_dir = tmp_path / "prep"
    run_dir = data_dir / "s1" / "awake" / "audio"
    run_dir.mkdir(parents=True)
    np.save(
        run_dir / "s1_run-1_schaefer400_ts.npy",
        np.random.RandomState(0).rand(200, 6),
        allow_pickle=False,
    )
    profile = dataclasses.replace(
        get_execution_profile("hunter"),
        hunter_phase1_shards_per_run=1,
        hunter_cut_shards_per_run=1,
        hunter_phase1_workers_per_task=2,
        hunter_phase1_chunk_size=2,
        hunter_shared_memory=False,
    )
    campaign_dir = tmp_path / "campaign"
    prepare_hunter_campaign(
        data_dir=data_dir,
        atlas="schaefer400",
        sessions=("awake",),
        condition="audio",
        stimulus_onsets=None,
        subjects=None,
        campaign_dir=campaign_dir,
        execution_profile=profile,
        iim_bins=2,
        iim_lag_trs=1,
        iim_n_parts=2,
        iim_max_timepoints=None,
        iim_max_nodes=5,
        iim_max_mechanism_size=2,
        iim_max_purview_size=2,
        step2_context={"hardware_target": "cpu"},
    )
    timing = hunter_iim.run_phase1_shard(campaign_dir, 0)["timing"]
    assert timing["children_cpu_seconds"] > 0
    assert timing["cpu_seconds"] == pytest.approx(
        timing["process_cpu_seconds"] + timing["children_cpu_seconds"]
    )
    summary = hunter_iim.summarize_campaign_timing(campaign_dir)["phase1-shard"]
    assert summary["cpu_seconds_total"] >= summary["process_cpu_seconds_total"]


def test_packed_shard_uses_the_built_packing(tmp_path):
    campaign_dir, manifest, _paths = _multi_run_campaign(tmp_path)
    assert manifest["scheduler"]["shards_per_node"] == 4
    # no explicit packing: the campaign's 4 shards per node, i.e. task 4*1 + 2
    out = hunter_iim.run_packed_shard(
        campaign_dir, "phase1-shard", 1, env={"PMI_LOCAL_RANK": "2"}
    )
    assert out["task_index"] == manifest["phase1_tasks"][6]["task_index"]
    assert out["run_key"] == manifest["phase1_tasks"][6]["run_key"]
    with pytest.raises(ValueError, match="does not match the campaign packing"):
        hunter_iim.run_packed_shard(
            campaign_dir, "phase1-shard", 1, 2, env={"PMI_LOCAL_RANK": "0"}
        )


def test_main_hunter_end_to_end_matches_local_iim(tmp_path, monkeypatch):
    import pandas as pd

    import run_pipeline

    bids = tmp_path / "bids"
    out = tmp_path / "out"
    rng = np.random.RandomState(3)
    for subj in ("01", "02"):
        func = bids / f"sub-{subj}" / "func"
        func.mkdir(parents=True)
        (func / f"sub-{subj}_task-audioawake_run-01_bold.json").write_text(
            json.dumps({"RepetitionTime": 2.0})
        )
        for ses in ("awake", "deep"):
            d = out / "preprocessed" / subj / ses / "audio"
            d.mkdir(parents=True)
            np.save(d / f"{subj}_run-1_schaefer400_ts.npy", rng.rand(60, 4))
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "tiny", "BIDSVersion": "1.8.0"})
    )

    # Steps 3-9 are covered elsewhere; keep this test on the Hunter hand-off.
    captured = {}
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
    monkeypatch.setattr(run_pipeline, "create_doc", lambda **k: captured.update(k))

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
        hunter_phase1_shards_per_run=2,
        hunter_cut_shards_per_run=2,
        hunter_workers_per_task=1,
        **common,
    )
    campaign = out / "cache" / "hunter_iim_campaign"
    manifest = json.loads((campaign / "campaign_manifest.json").read_text())
    spn = manifest["scheduler"]["shards_per_node"]
    for stage, key in (("phase1-shard", "phase1_tasks"), ("cut-shard", "cut_tasks")):
        for a in range(hunter_iim.packed_array_size(len(manifest[key]), spn)):
            for rank in range(spn):
                monkeypatch.setenv("PMI_LOCAL_RANK", str(rank))
                run_pipeline.main(
                    str(out),
                    hunter_stage=stage,
                    hunter_campaign_dir=str(campaign),
                    hunter_array_index=a,
                    hunter_shards_per_node=spn,
                    **common,
                )
    monkeypatch.delenv("PMI_LOCAL_RANK")
    run_pipeline.main(
        str(out), hunter_stage="reduce-all", hunter_campaign_dir=str(campaign), **common
    )
    run_pipeline.main(
        str(out),
        hunter_stage="finalize-pipeline",
        hunter_campaign_dir=str(campaign),
        **common,
    )

    df = pd.read_csv(out / "cache" / "step2_df.csv", dtype={"subject": str})
    assert {"hardware_target", "hardware_backend"}.issubset(df.columns)
    for run in manifest["runs"]:
        direct = mm.compute_IIM(
            np.load(run["ts_path"]).T,
            bins=3,
            lag_trs=1,
            n_parts=3,
            max_nodes=4,
            max_mechanism_size=2,
            max_purview_size=2,
            return_details=True,
            phase1_parallel_workers=None,
        )
        rows = df[(df["subject"] == run["subject"]) & (df["session"] == run["session"])]
        assert not rows.empty
        assert rows["IIM"].iloc[0] == pytest.approx(
            direct["canonical"], rel=1e-9, abs=1e-12
        )
    # the robustness atlases are absent here: recorded as skipped, not a crash
    assert (
        captured["atlas_results"]["aal116"]["skipped"]["reason"]
        == "missing_atlas_timeseries"
    )
    prov = json.loads((out / "cache" / "provenance_manifest.json").read_text())
    assert prov["status"] == "completed"
    assert prov["hunter_campaign"]["scheduler"]["scheduler"] == "pbs"
    assert (campaign / "timing_summary.json").exists()
