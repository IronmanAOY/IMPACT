"""
PBS Pro backend for HLRS Hunter.

Covers generated *.pbs content (directives, arrays, dependencies), packing
math (4 shards per mi300a node via PALS PMI_LOCAL_RANK), single-task jobs
(PBS arrays need >= 2 subjobs), walltime/array guards, env-var aliases, the
CPU-queue option for reduce/finalize, and the build-campaign regression
through run_pipeline.main().
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import run_pipeline
from impact_pipeline import hunter_iim
from impact_pipeline.execution_profiles import get_execution_profile
from impact_pipeline.hunter_iim import (
    default_cpu_bind,
    default_gpu_bind,
    packed_array_size,
    parse_walltime,
    prepare_hunter_campaign,
    resolve_hunter_settings,
    resolve_iim_cache_dir,
    resolve_packed_task_index,
)

HUNTER_ENV_PREFIXES = (
    "IMPACT_HUNTER_",
    "IMPACT_IIM_",
    "IMPACT_CONDA_ENV",
    "PMI_LOCAL_RANK",
    "PBS_",
)


@pytest.fixture(autouse=True)
def _clean_hunter_env(monkeypatch):
    import os

    for name in list(os.environ):
        if name.startswith(HUNTER_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)


def _make_prep(
    tmp_path, n_subjects=1, sessions=("awake",), n_regions=4, n_time=48, seed=0
):
    data_dir = tmp_path / "prep"
    rng = np.random.RandomState(seed)
    for s in range(n_subjects):
        subj = f"s{s + 1}"
        for ses in sessions:
            run_dir = data_dir / subj / ses / "audio"
            run_dir.mkdir(parents=True)
            np.save(
                run_dir / f"{subj}_run-1_schaefer400_ts.npy",
                rng.rand(n_time, n_regions),
                allow_pickle=False,
            )
    return data_dir


def _build(
    tmp_path,
    *,
    n_subjects=1,
    sessions=("awake",),
    phase1=2,
    cut=2,
    max_nodes=4,
    n_parts=3,
    scheduler=None,
    overrides=None,
    hardware_target="cpu",
):
    profile = dataclasses.replace(
        get_execution_profile("hunter"),
        hunter_phase1_shards_per_run=phase1,
        hunter_cut_shards_per_run=cut,
        hunter_phase1_workers_per_task=1,
        hunter_shared_memory=False,
    )
    data_dir = _make_prep(tmp_path, n_subjects=n_subjects, sessions=sessions)
    campaign_dir = tmp_path / "campaign"
    manifest = prepare_hunter_campaign(
        data_dir=data_dir,
        atlas="schaefer400",
        sessions=sessions,
        condition="audio",
        stimulus_onsets=None,
        subjects=None,
        campaign_dir=campaign_dir,
        execution_profile=profile,
        iim_bins=2,
        iim_lag_trs=1,
        iim_n_parts=n_parts,
        iim_max_timepoints=None,
        iim_max_nodes=max_nodes,
        iim_max_mechanism_size=2,
        iim_max_purview_size=2,
        step2_context={
            "hardware_target": hardware_target,
            "dataset_id": "ds003171",
            "data_origin": "dummy",
            "out_dir": str(tmp_path / "out"),
        },
        hardware_target=hardware_target,
        scheduler=scheduler,
        settings_overrides=overrides,
    )
    return campaign_dir, manifest


def _read(campaign_dir, name):
    return (campaign_dir / "pbs" / name).read_text(encoding="utf-8")


def test_pbs_is_default_and_scripts_carry_hunter_directives(tmp_path, monkeypatch):
    setup = tmp_path / "hunter_pbs_setup.sh"
    setup.write_text("module load cray-python\n", encoding="utf-8")
    monkeypatch.setenv("IMPACT_HUNTER_SETUP_FILE", str(setup))
    monkeypatch.setenv("IMPACT_HUNTER_PBS_GROUP_LIST", "hpcproj")
    monkeypatch.setenv("IMPACT_HUNTER_PYTHON", "/ws/venvs/impact-hunter/bin/python3")

    campaign_dir, manifest = _build(
        tmp_path, n_subjects=2, sessions=("awake", "deep"), phase1=4, cut=3
    )
    assert manifest["scheduler"]["scheduler"] == "pbs"
    assert not (campaign_dir / "slurm").exists()
    p1 = _read(campaign_dir, "01_phase1_shards.pbs")
    cut = _read(campaign_dir, "02_cut_shards.pbs")
    red = _read(campaign_dir, "03_reduce_finalize.pbs")

    for text in (p1, cut, red):
        assert text.splitlines()[0] == "#!/bin/bash"
        assert "#SBATCH" not in text
        assert "#PBS -l ws13=True" in text
        assert "#PBS -W group_list=hpcproj" in text
        assert "#PBS -j oe" in text
        # setup is sourced before nounset
        assert text.index(f"source {setup.resolve()}") < text.index("\nset -u\n")
        assert "/ws/venvs/impact-hunter/bin/python3" in text
        assert "IMPACT_SKIP_ENV_CHECK=1" in text

    n_p1 = len(manifest["phase1_tasks"])  # 4 runs x 3 mechanism shards (<= 4 requested)
    assert f"#PBS -J 0-{packed_array_size(n_p1, 4) - 1}" in p1
    assert "#PBS -r y" in p1
    assert "#PBS -l select=1:node_type=mi300a:mpiprocs=4" in p1
    assert "#PBS -l walltime=24:00:00" in p1
    assert 'array_index="${PBS_ARRAY_INDEX:-0}"' in p1
    assert (
        "mpiexec -n 4 --ppn 4 --cpu-bind list:0-23:24-47:48-71:72-95 "
        "--gpu-bind list:0:1:2:3" in p1
    )
    assert "--hunter-stage phase1-shard --hunter-array-index" in p1
    assert "--hunter-shards-per-node 4" in p1
    # phase-1 and cut shards both request the campaign target (cpu here)
    assert "--hardware-target cpu" in p1
    assert "--hardware-target cpu" in cut
    # one merged reduce + finalize job, not one node per run
    assert "#PBS -J" not in red
    assert (
        "--hunter-stage reduce-all" in red and "--hunter-stage finalize-pipeline" in red
    )
    assert red.index("reduce-all") < red.index("finalize-pipeline")
    assert "#PBS -l select=1:node_type=mi300a\n" in red


def test_single_task_stage_is_plain_job_not_array(tmp_path):
    # 1 run, 3 mechanism shards and 1 cut shard -> <= 4 tasks -> one node -> no -J.
    campaign_dir, manifest = _build(tmp_path, phase1=3, cut=1)
    assert packed_array_size(len(manifest["phase1_tasks"]), 4) == 1
    for name in ("01_phase1_shards.pbs", "02_cut_shards.pbs"):
        text = _read(campaign_dir, name)
        assert "#PBS -J" not in text
        assert "#PBS -r y" not in text
        assert "array_index=0\n" in text


def test_unpacked_single_task_uses_task_index(tmp_path):
    campaign_dir, manifest = _build(
        tmp_path, phase1=1, cut=1, overrides={"shards_per_node": 1}
    )
    text = _read(campaign_dir, "01_phase1_shards.pbs")
    assert "mpiexec" not in text
    assert "--hunter-task-index" in text
    assert "#PBS -J" not in text  # a single subjob must not be an array


def test_submit_script_uses_qsub_whole_array_afterok(tmp_path):
    campaign_dir, _manifest = _build(
        tmp_path, n_subjects=2, sessions=("awake", "deep"), phase1=4, cut=3
    )
    submit = _read(campaign_dir, "00_submit_all.sh")
    assert "sbatch" not in submit
    assert 'jid_p1=$(qsub "${script_dir}/01_phase1_shards.pbs")' in submit
    assert 'jid_cut=$(qsub "${script_dir}/02_cut_shards.pbs")' in submit
    assert (
        "qsub -W depend=afterok:${jid_p1}:${jid_cut} "
        '"${script_dir}/03_reduce_finalize.pbs"' in submit
    )
    assert 'cd "${script_dir}/logs"' in submit
    plan = json.loads(_read(campaign_dir, "campaign_plan.json"))
    assert plan["stages"]["phase1-shard"]["subjobs"] == packed_array_size(
        plan["stages"]["phase1-shard"]["tasks"], 4
    )


def test_smoke_script_runs_all_stages_in_one_test_queue_job(tmp_path):
    campaign_dir, manifest = _build(
        tmp_path, n_subjects=2, sessions=("awake",), phase1=4, cut=3
    )
    smoke = _read(campaign_dir, "90_smoke_all_in_one.pbs")
    assert "#PBS -q test" in smoke
    assert "#PBS -l walltime=00:25:00" in smoke
    assert "#PBS -J" not in smoke
    assert "impact_pipeline.hardware_selftest" in smoke
    n_p1 = packed_array_size(len(manifest["phase1_tasks"]), 4)
    assert f"array_index<{n_p1}" in smoke
    assert (
        smoke.index("phase1-shard")
        < smoke.index("cut-shard")
        < smoke.index("reduce-all")
        < smoke.index("finalize-pipeline")
    )


def test_packing_math_and_local_rank():
    assert packed_array_size(0, 4) == 0
    assert packed_array_size(1, 4) == 1
    assert packed_array_size(4, 4) == 1
    assert packed_array_size(5, 4) == 2
    assert packed_array_size(8704, 4) == 2176
    assert resolve_packed_task_index(0, 4, env={"PMI_LOCAL_RANK": "0"}) == 0
    assert resolve_packed_task_index(3, 4, env={"PMI_LOCAL_RANK": "2"}) == 14
    assert resolve_packed_task_index(7, 1, env={}) == 7
    with pytest.raises(RuntimeError, match="PMI_LOCAL_RANK"):
        resolve_packed_task_index(0, 4, env={})
    with pytest.raises(RuntimeError):
        resolve_packed_task_index(0, 4, env={"PMI_LOCAL_RANK": "4"})
    assert default_cpu_bind(96, 4) == "list:0-23:24-47:48-71:72-95"
    assert default_cpu_bind(96, 2) == "list:0-47:48-95"
    assert default_cpu_bind(96, 1) is None
    assert default_gpu_bind(4, 4) == "list:0:1:2:3"
    assert (
        default_gpu_bind(4, 2) is None
    )  # multi-GPU-per-rank syntax unverified -> configure explicitly


def test_packed_workers_default_to_cores_per_rank_minus_two():
    profile = get_execution_profile("hunter")
    effective, settings = resolve_hunter_settings(profile, env={})
    assert settings["shards_per_node"] == 4
    assert effective.hunter_phase1_workers_per_task == 22
    effective, settings = resolve_hunter_settings(
        profile,
        overrides={
            "shards_per_node": 2,
            "workers_per_task": 8,
            "cut_shards_per_run": 64,
        },
        env={},
    )
    assert settings["cpu_bind"] == "list:0-47:48-95"
    assert effective.hunter_phase1_workers_per_task == 8
    assert effective.hunter_cut_shards_per_run == 64
    effective, _ = resolve_hunter_settings(
        profile, env={"IMPACT_HUNTER_PHASE1_SHARDS_PER_RUN": "5"}
    )
    assert effective.hunter_phase1_shards_per_run == 5


def test_walltime_limits():
    profile = get_execution_profile("hunter")
    assert parse_walltime("24:00:00") == 86400
    assert parse_walltime("1-00:00:00") == 86400
    with pytest.raises(ValueError, match="24:00:00"):
        resolve_hunter_settings(profile, env={"IMPACT_HUNTER_CUT_TIME": "30:00:00"})
    with pytest.raises(ValueError, match="test"):
        resolve_hunter_settings(
            profile,
            env={
                "IMPACT_HUNTER_PBS_QUEUE": "test",
                "IMPACT_HUNTER_CUT_TIME": "01:00:00",
            },
        )
    _, settings = resolve_hunter_settings(
        profile, env={"IMPACT_HUNTER_PBS_QUEUE": "test"}
    )
    assert (
        settings["phase1_time"]
        == settings["cut_time"]
        == settings["reduce_time"]
        == "00:25:00"
    )


def test_array_size_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPACT_HUNTER_MAX_ARRAY_SIZE", "1")
    with pytest.raises(ValueError, match="IMPACT_HUNTER_MAX_ARRAY_SIZE"):
        _build(tmp_path, n_subjects=2, sessions=("awake", "deep"), phase1=4, cut=3)


def test_deprecated_slurm_env_names_are_still_accepted(tmp_path, monkeypatch):
    setup = tmp_path / "legacy_setup.sh"
    setup.write_text("module load cray-python\n", encoding="utf-8")
    monkeypatch.setenv("IMPACT_HUNTER_SLURM_SETUP_FILE", str(setup))
    monkeypatch.setenv("IMPACT_HUNTER_SLURM_SETUP", "export FOO=1")
    monkeypatch.setenv("IMPACT_HUNTER_SLURM_ACCOUNT", "legacyproj")
    campaign_dir, _ = _build(tmp_path)
    text = _read(campaign_dir, "01_phase1_shards.pbs")
    assert f"source {setup.resolve()}" in text
    assert "export FOO=1" in text
    assert "#PBS -W group_list=legacyproj" in text


def test_missing_setup_file_fails_at_build(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPACT_HUNTER_SETUP_FILE", str(tmp_path / "missing.sh"))
    with pytest.raises(FileNotFoundError):
        _build(tmp_path)


def test_reduce_job_uses_cpu_queue_only_when_configured(tmp_path, monkeypatch):
    campaign_dir, _ = _build(tmp_path)
    # default: academic users cannot use genoa, so reduce/finalize stay on mi300a
    assert "node_type=mi300a" in _read(campaign_dir, "03_reduce_finalize.pbs")

    monkeypatch.setenv("IMPACT_HUNTER_PBS_CPU_QUEUE", "pre")
    monkeypatch.setenv("IMPACT_HUNTER_PBS_CPU_NODE_TYPE", "genoa3tb64c")
    other = tmp_path / "second"
    other.mkdir()
    campaign_dir2, manifest2 = _build(other)
    red = _read(campaign_dir2, "03_reduce_finalize.pbs")
    assert "#PBS -q pre" in red
    assert "#PBS -l select=1:node_type=genoa3tb64c\n" in red
    assert "--hardware-target cpu" in red
    assert manifest2["scheduler"]["cpu_queue"] == "pre"


def test_workspace_resource_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPACT_HUNTER_PBS_WORKSPACE_RESOURCE", "none")
    campaign_dir, _ = _build(tmp_path)
    assert "ws13" not in _read(campaign_dir, "01_phase1_shards.pbs")


def test_conda_is_optional_launcher(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPACT_HUNTER_CONDA_ENV", "impact-rocm")
    campaign_dir, _ = _build(tmp_path)
    text = _read(campaign_dir, "01_phase1_shards.pbs")
    assert "conda run --no-capture-output -n impact-rocm python" in text
    assert "IMPACT_SKIP_ENV_CHECK" not in text


def test_slurm_times_keep_slurm_semantics():
    """A bare Slurm --time is minutes and D-HH is days-hours (not PBS seconds)."""
    profile = get_execution_profile("hunter")
    _, settings = resolve_hunter_settings(
        profile,
        scheduler="slurm",
        env={
            "IMPACT_HUNTER_CUT_TIME": "60",
            "IMPACT_HUNTER_PHASE1_TIME": "1-12",
            "IMPACT_HUNTER_REDUCE_TIME": "02:00:00",
        },
    )
    assert settings["cut_time"] == "60"
    assert settings["phase1_time"] == "1-12"
    assert settings["reduce_time"] == "02:00:00"


def test_slurm_backend_is_optional_and_keeps_gpu_off_reducers(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPACT_HUNTER_GPUS_PER_TASK", "1")
    monkeypatch.setenv("IMPACT_HUNTER_SLURM_ARRAY_THROTTLE", "50")
    # Build on the CPU; the scripts are generated for an accelerator target.
    campaign_dir, manifest = _build(tmp_path, scheduler="slurm", hardware_target="cpu")
    manifest["hardware_target"] = "hunter-apu"
    hunter_iim.write_hunter_scheduler_scripts(campaign_dir, manifest)
    slurm = campaign_dir / "slurm"
    assert manifest["scheduler"]["scheduler"] == "slurm"
    cut = (slurm / "03_cut_shards.sbatch").read_text()
    assert "#SBATCH --gpus-per-task=1" in cut
    assert "#SBATCH --partition=apu" in cut
    assert "%50" in cut
    for name in (
        "01_phase1_shards.sbatch",
        "02_phase1_reduce.sbatch",
        "04_cut_reduce.sbatch",
        "05_finalize_pipeline.sbatch",
    ):
        text = (slurm / name).read_text()
        assert "--gpus-per-task" not in text, name
        assert "#SBATCH --partition=cpu" in text, name
        assert "--hardware-target cpu" in text, name
        assert "#SBATCH --output=" in text, name


def test_iim_cache_dir_prefers_node_local_storage(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    assert resolve_iim_cache_dir({"IMPACT_IIM_CACHE_DIR": str(explicit)}) == explicit
    assert (
        resolve_iim_cache_dir({"TMPDIR": str(tmp_path / "ramdisk")})
        == tmp_path / "ramdisk"
    )
    fake_local = tmp_path / "localscratch"
    (fake_local / "123.hunter-pbs01").mkdir(parents=True)
    monkeypatch.setattr(hunter_iim, "LOCALSCRATCH_ROOT", fake_local)
    got = resolve_iim_cache_dir(
        {"PBS_JOBID": "123.hunter-pbs01", "TMPDIR": str(tmp_path / "ramdisk")}
    )
    assert got == fake_local / "123.hunter-pbs01"


def _tiny_bids_and_prep(tmp_path):
    bids = tmp_path / "bids"
    out = tmp_path / "out"
    rng = np.random.RandomState(1)
    for subj in ("01", "02"):
        func = bids / f"sub-{subj}" / "func"
        func.mkdir(parents=True)
        (func / f"sub-{subj}_task-audioawake_run-01_bold.json").write_text(
            json.dumps({"RepetitionTime": 2.0})
        )
        for ses in ("awake", "deep"):
            d = out / "preprocessed" / subj / ses / "audio"
            d.mkdir(parents=True)
            np.save(d / f"{subj}_run-1_schaefer400_ts.npy", rng.rand(40, 3))
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "tiny", "BIDSVersion": "1.8.0"})
    )
    return bids, out


def test_build_campaign_through_main_regression(tmp_path):
    """Before the fix, hunter build-campaign via main() raised TypeError.

    (_run_hunter_stage was called without data_origin/dataset_role/provenance_label.)
    """
    bids, out = _tiny_bids_and_prep(tmp_path)
    run_pipeline.main(
        str(out),
        dataset_id="ds003171",
        bids_root_override=str(bids),
        execution_mode="hunter",
        hunter_stage="build-campaign",
        mpc_metrics=["PDI", "NAS", "IIM"],
        compute_ci=False,
        iim_max_nodes_override=3,
        iim_n_parts_override=2,
    )
    campaign = out / "cache" / "hunter_iim_campaign"
    manifest = json.loads((campaign / "campaign_manifest.json").read_text())
    ctx = manifest["step2_context"]
    assert ctx["data_origin"] == "real"
    assert ctx["dataset_role"] == "study_data"
    assert ctx["provenance_label"] == "real_study_data"
    assert ctx["iim_settings"]["iim_max_nodes"] == 3
    assert manifest["scheduler"]["scheduler"] == "pbs"
    assert (campaign / "pbs" / "00_submit_all.sh").exists()
    prov = json.loads((out / "cache" / "provenance_manifest.json").read_text())
    assert prov["status"] == "hunter_campaign_built"
    assert prov["parameters"]["iim"]["iim_max_nodes"] == 3


def _protocol_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("MPC protocol:")]


def test_stage_jobs_log_the_campaign_protocol(tmp_path, caplog, monkeypatch):
    """The PBS stage jobs run without --protocol, so the command line resolves
    the default protocol; their log must name the protocol the campaign was
    built with (and computes with), not that default."""
    import logging

    from impact_pipeline import run_synergy_ci
    from impact_pipeline.evidence import Protocol

    bids, out = _tiny_bids_and_prep(tmp_path)
    derived = (Path(run_pipeline.root) / "protocols" / "examples"
               / "mpc_default_v1_schaefer400_7networks_hub.json").resolve()
    default = run_pipeline.resolve_cli_protocol(None, "real")
    assert Path(default) == Path(run_pipeline.DEFAULT_EMPIRICAL_PROTOCOL)
    derived_hash = Protocol.from_json(derived).hash
    default_hash = Protocol.from_json(default).hash
    assert derived_hash != default_hash
    common = dict(
        dataset_id="ds003171",
        bids_root_override=str(bids),
        execution_mode="hunter",
        mpc_metrics=["IIM"],
        compute_ci=False,
        iim_max_nodes_override=3,
        iim_n_parts_override=2,
    )
    caplog.set_level(logging.INFO)
    run_pipeline.main(str(out), hunter_stage="build-campaign",
                      protocol=str(derived), **common)
    (line,) = _protocol_lines(caplog)
    assert str(derived) in line and derived_hash[:12] in line
    campaign = out / "cache" / "hunter_iim_campaign"

    class _Stop(Exception):
        pass

    captured = {}

    def fake_run_s_ci(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(run_synergy_ci, "run_s_ci", fake_run_s_ci)
    monkeypatch.setattr(run_pipeline, "collect_iim_results_by_path", lambda d: {})
    # (stage, expected error, campaign directory given as in the PBS jobs)
    stages = (("status", None, True), ("phase1-reduce", FileNotFoundError, True),
              ("finalize-pipeline", _Stop, True), ("status", None, False))
    for stage, error, campaign_given in stages:
        caplog.clear()
        # what a stage job's command line resolves: the default protocol
        kwargs = dict(hunter_stage=stage, hunter_run_index=0, protocol=default,
                      **common)
        if campaign_given:
            kwargs["hunter_campaign_dir"] = str(campaign)
        if error is None:
            run_pipeline.main(str(out), **kwargs)
        else:
            with pytest.raises(error):
                run_pipeline.main(str(out), **kwargs)
        lines = _protocol_lines(caplog)
        assert len(lines) == 1, (stage, lines)
        assert str(derived) in lines[0], (stage, lines)
        assert derived_hash[:12] in lines[0], (stage, lines)
        assert default_hash[:12] not in lines[0], (stage, lines)
        assert "the campaign's protocol" in lines[0]
        notes = [r.getMessage() for r in caplog.records
                 if "not the command-line protocol" in r.getMessage()]
        assert len(notes) == 1 and default_hash[:12] in notes[0], (stage, notes)
        # With the campaign directory given, the protocol line comes first,
        # before the hardware (and dataset and provenance) setup; the default
        # directory is known only once the output directory is resolved.
        messages = [r.getMessage() for r in caplog.records]
        protocol_at = messages.index(lines[0])
        hardware_at = next(i for i, m in enumerate(messages)
                           if m.startswith("Hardware target:"))
        assert (protocol_at < hardware_at) == campaign_given, (stage, messages)
    # and finalize computes the verdicts under the campaign's protocol
    assert Protocol.from_dict(captured["protocol"]).hash == derived_hash


def test_campaign_protocol_falls_back_without_a_manifest(tmp_path):
    options = run_pipeline._mpc_evidence_options(
        protocol=run_pipeline.DEFAULT_EMPIRICAL_PROTOCOL)
    prov, from_campaign = run_pipeline._hunter_campaign_protocol(
        tmp_path / "missing", options, run_pipeline.DEFAULT_EMPIRICAL_PROTOCOL)
    assert from_campaign is False
    assert prov == run_pipeline._mpc_protocol_provenance(
        options, run_pipeline.DEFAULT_EMPIRICAL_PROTOCOL)
    # a campaign built with a flag-built protocol says so
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    flags = run_pipeline._mpc_evidence_options()
    (campaign / "campaign_manifest.json").write_text(json.dumps(
        {"step2_context": {"mpc_evidence": flags, "run_parameters": {
            "mpc_protocol": run_pipeline._mpc_protocol_provenance(flags)}}}))
    prov, from_campaign = run_pipeline._hunter_campaign_protocol(
        campaign, options, run_pipeline.DEFAULT_EMPIRICAL_PROTOCOL)
    assert from_campaign is True
    assert prov["hash"] is None and prov["note"] == "default protocol from flags"


def test_only_non_build_hunter_jobs_are_stage_jobs():
    stage_job = run_pipeline._is_hunter_stage_job
    assert stage_job("hunter", "finalize-pipeline")
    assert stage_job("HUNTER", "cut-shard")
    assert not stage_job("hunter", None)
    assert not stage_job("hunter", "build-campaign")
    assert not stage_job("local", "finalize-pipeline")
    assert not stage_job("no-such-mode", "status")


def test_build_campaign_requires_iim(tmp_path):
    bids, out = _tiny_bids_and_prep(tmp_path)
    with pytest.raises(ValueError, match="IIM"):
        run_pipeline.main(
            str(out),
            dataset_id="ds003171",
            bids_root_override=str(bids),
            execution_mode="hunter",
            hunter_stage="build-campaign",
            mpc_metrics=["PDI", "NAS"],
            compute_ci=False,
        )


def test_build_campaign_on_login_node_without_accelerator(tmp_path, monkeypatch):
    """hunter-apu campaigns build without CuPy/APU; jobs still request hunter-apu."""
    import importlib

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("no cupy on the login node")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    bids, out = _tiny_bids_and_prep(tmp_path)
    run_pipeline.main(
        str(out),
        dataset_id="ds003171",
        bids_root_override=str(bids),
        execution_mode="hunter",
        hunter_stage="build-campaign",
        hardware_target="hunter-apu",
        mpc_metrics=["IIM"],
        compute_ci=False,
        iim_max_nodes_override=3,
        iim_n_parts_override=2,
    )
    campaign = out / "cache" / "hunter_iim_campaign"
    manifest = json.loads((campaign / "campaign_manifest.json").read_text())
    assert manifest["hardware_target"] == "hunter-apu"
    assert manifest["build_hardware_backend"] == "cpu->cpu"
    cut = (campaign / "pbs" / "02_cut_shards.pbs").read_text()
    assert "--hardware-target hunter-apu" in cut
    # GPU use is mandatory on Hunter: phase-1 Psi runs on the APU as well
    p1 = (campaign / "pbs" / "01_phase1_shards.pbs").read_text()
    assert "--hardware-target hunter-apu" in p1
    # a local (non-build) stage still resolves the target strictly
    with pytest.raises(RuntimeError, match="Hardware target unavailable"):
        run_pipeline.main(
            str(out),
            dataset_id="ds003171",
            bids_root_override=str(bids),
            hardware_target="hunter-apu",
            mpc_metrics=["IIM"],
            compute_ci=False,
        )


def test_hunter_stage_names_are_validated():
    with pytest.raises(ValueError, match="Unknown hunter stage"):
        run_pipeline.main(
            str(Path("unused")), execution_mode="hunter", hunter_stage="phase1-shrad"
        )


def test_ci_reference_is_forwarded_to_the_campaign_and_finalize(tmp_path, monkeypatch):
    """--ci-reference given at build time reaches the finalize stage's CI."""
    from impact_pipeline import run_synergy_ci

    bids, out = _tiny_bids_and_prep(tmp_path)
    ref = tmp_path / "refs" / "ci_reference.json"
    ref.parent.mkdir()
    comps = ("RAM", "PDI", "NAS", "IIM", "SRPI")
    ref.write_text(json.dumps({"references": {c: 1.0 for c in comps}}))
    monkeypatch.chdir(tmp_path)  # a relative path must survive the batch jobs
    common = dict(
        dataset_id="ds003171",
        bids_root_override=str(bids),
        execution_mode="hunter",
        mpc_metrics=["IIM"],
        compute_ci=False,
        iim_max_nodes_override=3,
        iim_n_parts_override=2,
    )
    run_pipeline.main(
        str(out),
        hunter_stage="build-campaign",
        ci_reference="refs/ci_reference.json",
        **common,
    )
    campaign = out / "cache" / "hunter_iim_campaign"
    ctx = json.loads((campaign / "campaign_manifest.json").read_text())[
        "step2_context"
    ]
    assert ctx["ci_reference"] == str(ref.resolve())
    prov = json.loads((out / "cache" / "provenance_manifest.json").read_text())
    assert prov["parameters"]["ci_reference"] == str(ref.resolve())

    captured = {}

    class _Stop(Exception):
        pass

    def fake_run_s_ci(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(run_synergy_ci, "run_s_ci", fake_run_s_ci)
    monkeypatch.setattr(run_pipeline, "collect_iim_results_by_path", lambda d: {})
    with pytest.raises(_Stop):
        # the finalize job's own command line does not repeat --ci-reference
        run_pipeline.main(
            str(out),
            hunter_stage="finalize-pipeline",
            hunter_campaign_dir=str(campaign),
            **common,
        )
    assert captured["ci_reference"] == str(ref.resolve())


def test_jobs_run_an_explicit_checkout_and_export_it(tmp_path, monkeypatch):
    monkeypatch.delenv("IMPACT_REPO_ROOT", raising=False)
    checkout = tmp_path / "ws" / "impact-synergy-pipeline"
    checkout.mkdir(parents=True)
    (checkout / "run_pipeline.py").write_text("# entry point\n")
    campaign_dir, manifest = _build(tmp_path, overrides={"repo_root": str(checkout)})
    assert manifest["scheduler"]["repo_root"] == str(checkout.resolve())
    for name in ("01_phase1_shards.pbs", "02_cut_shards.pbs", "03_reduce_finalize.pbs"):
        text = _read(campaign_dir, name)
        assert f"cd {checkout.resolve()}" in text, name
        assert f"export IMPACT_REPO_ROOT={checkout.resolve()}" in text, name
        assert str(checkout.resolve() / "run_pipeline.py") in text, name
    with pytest.raises(FileNotFoundError, match="run_pipeline.py"):
        _build(tmp_path / "bad", overrides={"repo_root": str(tmp_path)})


def test_non_editable_install_needs_repo_root(tmp_path, monkeypatch):
    from impact_pipeline import provenance

    site = tmp_path / "site-packages" / "impact_pipeline"
    site.mkdir(parents=True)
    monkeypatch.setattr(provenance, "__file__", str(site / "provenance.py"))
    monkeypatch.delenv("IMPACT_REPO_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="--repo-root"):
        _build(tmp_path / "a")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "run_pipeline.py").write_text("# entry point\n")
    monkeypatch.setenv("IMPACT_REPO_ROOT", str(checkout))
    _campaign_dir, manifest = _build(tmp_path / "b")
    assert manifest["scheduler"]["repo_root"] == str(checkout.resolve())


def test_shard_code_version_is_package_version_plus_commit(tmp_path):
    import impact_pipeline

    campaign_dir, manifest = _build(tmp_path)
    base = impact_pipeline.__version__.split("+", 1)[0]
    assert manifest["kernel_code_version"].startswith(base + "+")
    assert manifest["code_version"]["package_version"] == impact_pipeline.__version__
    shard = hunter_iim.run_phase1_shard(campaign_dir, 0)
    assert shard["identity"]["code_version"] == manifest["kernel_code_version"]
