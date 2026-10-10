"""
Orchestration in run_pipeline.py: fMRIPrep hand-off, step-6 atlas
robustness, replication guard, synthetic/real separation, provenance and CLI.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_pipeline

REPO = Path(run_pipeline.__file__).resolve().parent


def _bids(root: Path, subjects=("01", "02"), synthetic=False):
    root.mkdir(parents=True, exist_ok=True)
    desc = {"Name": "mini", "BIDSVersion": "1.8.0"}
    if synthetic:
        desc.update({"SyntheticData": True, "DatasetType": "synthetic"})
    (root / "dataset_description.json").write_text(json.dumps(desc))
    for subj in subjects:
        func = root / f"sub-{subj}" / "func"
        func.mkdir(parents=True, exist_ok=True)
        (func / f"sub-{subj}_task-audioawake_run-01_bold.nii.gz").write_bytes(b"")
        (func / f"sub-{subj}_task-audioawake_run-01_bold.json").write_text(
            json.dumps({"RepetitionTime": 2.0})
        )
    return root


# --------------------------------------------------------------------------- fMRIPrep


def test_ensure_fmriprep_mounts_license_by_path_and_skips_done_subjects(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _bids(tmp_path / "bids")
    lic = tmp_path / "license.txt"  # README example name, not fs_license.txt
    lic.write_text("dummy")
    out = tmp_path / "bids" / "derivatives" / "fmriprep"
    done = out / "sub-02" / "func"
    done.mkdir(parents=True)
    stem = "sub-02_task-audioawake_run-01_space-MNI152NLin2009cAsym"
    (done / f"{stem}_desc-preproc_bold.nii.gz").write_bytes(b"")
    (out / "sub-02.html").write_text("report")
    cmds = []
    monkeypatch.setattr(
        run_pipeline.subprocess, "run", lambda cmd, check: cmds.append(cmd)
    )
    monkeypatch.setenv("IMPACT_FMRIPREP_NTHREADS", "4")

    ran = run_pipeline.ensure_fmriprep(
        bids_dir="bids",  # relative: docker -v needs absolute host paths
        fmriprep_out=str(out),
        work_dir=str(tmp_path / "work"),
        fs_license=str(lic),
        freesurf_out=str(tmp_path / "bids" / "derivatives" / "freesurfer"),
    )
    assert ran == ["01"]
    cmd = cmds[0]
    assert f"{lic.resolve()}:/opt/freesurfer/license.txt:ro" in cmd
    assert cmd[cmd.index("--fs-license-file") + 1] == "/opt/freesurfer/license.txt"
    assert f"{(tmp_path / 'bids').resolve()}:/data" in cmd
    assert cmd[cmd.index("--nthreads") + 1] == "4"
    assert "nipreps/fmriprep:25.1.3" in cmd


def test_fmriprep_dir_resolution(tmp_path):
    bids = _bids(tmp_path / "bids")
    out = tmp_path / "out"
    canonical = (bids / "derivatives" / "fmriprep").resolve()
    assert run_pipeline._resolve_fmriprep_dir(None, bids, out) == canonical
    assert (
        run_pipeline._resolve_fmriprep_dir(tmp_path / "custom", bids, out)
        == (tmp_path / "custom").resolve()
    )
    (out / "fmriprep" / "sub-01").mkdir(parents=True)  # legacy location only
    assert (
        run_pipeline._resolve_fmriprep_dir(None, bids, out)
        == (out / "fmriprep").resolve()
    )
    (canonical / "sub-01").mkdir(parents=True)
    assert run_pipeline._resolve_fmriprep_dir(None, bids, out) == canonical


def test_preprocessing_without_derivatives_names_canonical_location(tmp_path):
    bids = _bids(tmp_path / "bids")
    with pytest.raises(FileNotFoundError, match="derivatives/fmriprep"):
        run_pipeline.main(
            str(tmp_path / "out"),
            dataset_id="ds003171",
            bids_root_override=str(bids),
            run_preprocessing_flag=True,
            mpc_metrics=["PDI"],
            compute_ci=False,
        )


# --------------------------------------------------------------------------- step 6


def _prep_tree(
    root: Path,
    atlases=("schaefer400",),
    subjects=("01", "02"),
    sessions=("awake", "deep"),
):
    for subj in subjects:
        for ses in sessions:
            d = root / subj / ses / "audio"
            d.mkdir(parents=True, exist_ok=True)
            for atlas in atlases:
                np.save(d / f"{subj}_run-1_{atlas}_ts.npy", np.zeros((10, 3)))
    return root


def _robustness_kwargs(prep, df, context):
    return dict(
        prep_out=prep,
        bids_root=None,
        sessions=("awake", "deep"),
        compute_ci=False,
        df=df,
        subjects=None,
        dataset_id="ds003171",
        pdi_params={},
        pdi_require_explicit_params=True,
        pdi_require_strict_baseline=True,
        pdi_primary_endpoint="anchor",
        nas_params={},
        srpi_params={},
        srpi_require_explicit_params=True,
        context=context,
    )


def test_atlas_robustness_honours_iim_settings_and_skips_missing_atlases(
    tmp_path, monkeypatch
):
    # legacy 'aal90' files exist (the 116-label AAL); Shen-268 was never extracted
    prep = _prep_tree(tmp_path / "prep", atlases=("schaefer400", "aal90"))
    captured = {}

    def fake_atlas_check(data_dir, atlases, **kwargs):
        captured.update(kwargs, atlases=atlases)
        return {a: {"roughness": {}, "metrics": {}, "notes": []} for a in atlases}

    monkeypatch.setattr(run_pipeline, "atlas_check", fake_atlas_check)
    df = pd.DataFrame(columns=["subject", "session", "theta", "S", "PDI", "NAS", "IIM"])
    iim = {
        "iim_max_nodes": 6,
        "iim_n_parts": 12,
        "iim_max_mechanism_size": None,
        "iim_enable_parallel": False,
    }
    res = run_pipeline._run_atlas_robustness(
        **_robustness_kwargs(
            prep,
            df,
            {
                "condition": "audio",
                "tr": 2.0,
                "iim_settings": iim,
                "hardware_target": "cpu",
                "execution_mode": "local",
            },
        )
    )
    assert captured["atlases"] == ("aal90",)
    assert captured["iim_max_nodes"] == 6 and captured["iim_n_parts"] == 12
    assert captured["iim_enable_parallel"] is False
    assert "iim_max_mechanism_size" not in captured  # None = default, not forwarded
    assert captured["condition"] == "audio" and captured["tr"] == 2.0
    assert captured["mpc_metrics"] == ["PDI", "NAS", "IIM"]
    assert "aal116" in res and "aal90" not in res
    assert res["shen268"]["skipped"]["reason"] == "missing_atlas_timeseries"


def test_atlas_robustness_in_hunter_finalize_excludes_undistributed_iim(
    tmp_path, monkeypatch
):
    prep = _prep_tree(tmp_path / "prep", atlases=("schaefer400", "aal116", "shen268"))
    captured = {}
    monkeypatch.setattr(
        run_pipeline,
        "atlas_check",
        lambda data_dir, atlases, **k: captured.update(k)
        or {a: {"notes": []} for a in atlases},
    )
    df = pd.DataFrame(columns=["subject", "session", "theta", "S", "NAS", "IIM"])
    res = run_pipeline._run_atlas_robustness(
        **_robustness_kwargs(
            prep, df, {"condition": "audio", "tr": 2.0, "execution_mode": "hunter"}
        )
    )
    assert captured["mpc_metrics"] == ["NAS"]
    assert any("IIM omitted" in n for n in res["aal116"]["notes"])
    disabled = run_pipeline._run_atlas_robustness(
        **_robustness_kwargs(prep, df, {"enabled": False})
    )
    assert "skipped" in disabled


def test_atlas_check_records_missing_robustness_timeseries_instead_of_crashing(
    tmp_path,
):
    """
    Absent atlas files must not raise FileNotFoundError out of step 6.
    Besides the run_pipeline pre-check above, atlas_check itself (step 6)
    records the atlas as skipped (real compute_synergy_ci, nothing mocked).
    """
    from impact_pipeline.atlas_robustness import atlas_check

    prep = _prep_tree(tmp_path / "prep", atlases=("schaefer400",))
    res = atlas_check(
        str(prep),
        atlases=("aal90",),
        sessions=("awake", "deep"),
        tr=2.0,
        mpc_metrics=[],
    )
    assert "missing time series for atlas 'aal90'" in res["aal90"]["skipped"]


# ------------------------------------------------------------------ main() guards


def _reuse_step2_setup(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    cache = out / "cache"
    (out / "preprocessed" / "01" / "awake" / "audio").mkdir(parents=True)
    cache.mkdir(parents=True)
    bids = _bids(tmp_path / "bids", subjects=("01",))
    pd.DataFrame(
        [
            {
                "subject": "01",
                "session": "awake",
                "theta": 0.6,
                "S": 0.2,
                "IIM": 0.9,
                "hardware_target": "cpu",
            },
            {
                "subject": "01",
                "session": "deep",
                "theta": 0.6,
                "S": 0.1,
                "IIM": 0.8,
                "hardware_target": "cpu",
            },
        ]
    ).to_csv(cache / "step2_df.csv", index=False)
    pd.DataFrame(
        [
            {"subject": "01", "session": "awake", "S": 0.2},
            {"subject": "01", "session": "deep", "S": 0.1},
        ]
    ).to_csv(cache / "step2_df_mean.csv", index=False)
    pd.DataFrame([{"theta": 0.6, "t_S": 1.0, "p_S": 0.3, "d_S": 0.5}]).to_csv(
        cache / "step2_theta_stats.csv", index=False
    )
    captured = {}
    monkeypatch.setattr(
        run_pipeline,
        "compute_baseline_metrics",
        lambda df_mean, **k: df_mean.assign(
            mean_conn=0.1, modularity=0.2, pci_fmri=0.3
        ),
    )
    monkeypatch.setattr(run_pipeline, "bootstrap_ci", lambda *a, **k: (0.1, 0.9))
    monkeypatch.setattr(
        run_pipeline, "permutation_test_auc", lambda *a, **k: (0.6, 0.2)
    )
    monkeypatch.setattr(run_pipeline, "compare_models", lambda *a, **k: {})
    monkeypatch.setattr(run_pipeline, "atlas_check", lambda *a, **k: {})
    monkeypatch.setattr(run_pipeline, "create_doc", lambda **k: captured.update(k))
    return out, bids, captured


def test_replication_is_checked_before_the_expensive_steps(tmp_path, monkeypatch):
    out, bids, _captured = _reuse_step2_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_pipeline, "_resolve_replication_root", lambda: tmp_path / "no_melbourne"
    )
    monkeypatch.setattr(
        run_pipeline,
        "_persist_step2_outputs",
        lambda *a, **k: pytest.fail("step 2 reached"),
    )
    with pytest.raises(FileNotFoundError, match="--run-replication"):
        run_pipeline.main(
            str(out),
            dataset_id="ds003171",
            bids_root_override=str(bids),
            reuse_step2=True,
            mpc_metrics=["IIM"],
            compute_ci=False,
            run_replication_flag=True,
        )


def test_replication_failure_is_recorded_and_report_still_written(
    tmp_path, monkeypatch
):
    out, bids, captured = _reuse_step2_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(run_pipeline, "_check_replication_inputs", lambda root: None)

    def broken_replication(**_kwargs):
        raise TypeError("run_preprocessing() missing 1 required positional argument")

    monkeypatch.setattr(run_pipeline, "run_replication", broken_replication)
    run_pipeline.main(
        str(out),
        dataset_id="ds003171",
        bids_root_override=str(bids),
        reuse_step2=True,
        mpc_metrics=["IIM"],
        compute_ci=False,
        run_replication_flag=True,
    )
    assert "TypeError" in captured["repl"]["error"]
    prov = json.loads((out / "cache" / "provenance_manifest.json").read_text())
    assert prov["status"] == "completed"
    assert prov["parameters"]["run_replication"] is True
    assert prov["parameters"]["iim"]["iim_checkpoint_every_cuts"] == 1
    assert prov["code_version"]["source"] in {
        "git",
        "unavailable",
        "IMPACT_CODE_VERSION",
    }
    assert prov["hardware_backend"]["target"] == "cpu"
    assert "numpy" in prov["runtime"]["packages"]


def test_synthetic_dataset_requires_dummy_origin(tmp_path, monkeypatch):
    out, _bids_dir, _captured = _reuse_step2_setup(tmp_path, monkeypatch)
    synthetic = _bids(tmp_path / "synthetic_bids", subjects=("01",), synthetic=True)
    with pytest.raises(ValueError, match="--data-origin dummy"):
        run_pipeline.main(
            str(out),
            dataset_id="ds003171",
            bids_root_override=str(synthetic),
            reuse_step2=True,
            mpc_metrics=["IIM"],
            compute_ci=False,
        )


def test_subject_ids_are_normalised_once(tmp_path, monkeypatch):
    from impact_pipeline import run_synergy_ci

    out, bids, _captured = _reuse_step2_setup(tmp_path, monkeypatch)
    seen = {}

    def fake_run_s_ci(**kwargs):
        seen["subjects"] = kwargs["subjects"]
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(run_synergy_ci, "run_s_ci", fake_run_s_ci)
    with pytest.raises(RuntimeError, match="stop after capture"):
        run_pipeline.main(
            str(out),
            dataset_id="ds003171",
            bids_root_override=str(bids),
            subjects=["sub-01", "01"],
            mpc_metrics=["IIM"],
            compute_ci=False,
        )
    assert seen["subjects"] == ["01"]


def test_metric_subset_keeps_hardware_provenance_columns():
    df = pd.DataFrame(
        [
            {
                "subject": "1",
                "session": "awake",
                "theta": 0.5,
                "S": 0.1,
                "IIM": 0.2,
                "hardware_target": "hunter-apu",
                "hardware_backend": "cupy",
                "hardware_runtime": "rocm",
            }
        ]
    )
    mean = pd.DataFrame(
        [
            {
                "subject": "1",
                "session": "awake",
                "S": 0.1,
                "hardware_target": "hunter-apu",
            }
        ]
    )
    out_df, out_mean = run_pipeline._apply_metric_subset(
        df, mean, mpc_metrics=["IIM"], compute_ci=False
    )
    assert {"hardware_target", "hardware_backend", "hardware_runtime"}.issubset(
        out_df.columns
    )
    assert "hardware_target" in out_mean.columns


def test_eeg_config_defines_disjoint_rest_rules():
    cfg = run_pipeline.DATASET_CONFIGS["ds005620"]
    assert cfg["eeg_session_rules"]["deep"][0] == ("sed2", "rest", "first")
    assert cfg["eeg_rest_rules"]["deep"][0] == ("sed2", "rest", "after_first")
    assert cfg["eeg_rest_rules"]["awake"] == [("awake", "EO")]


def test_cli_rejects_unknown_hunter_stage_and_exposes_new_flags():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "run_pipeline.py"),
            "--hunter-stage",
            "phase1-shrad",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2
    assert "invalid choice" in proc.stderr
    helptext = subprocess.run(
        [sys.executable, str(REPO / "run_pipeline.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    for flag in (
        "--hunter-scheduler",
        "--hunter-array-index",
        "--hunter-shards-per-node",
        "--hunter-cut-shards-per-run",
        "--hunter-workers-per-task",
        "--assume-tr",
        "--fmriprep-dir",
        "--no-atlas-robustness",
    ):
        assert flag in helptext
