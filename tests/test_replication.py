import json

import numpy as np
import pandas as pd
import pytest

from impact_pipeline import replication


def _bids_with_derivatives(root, with_deriv=True):
    (root / "sub-01" / "func").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "melb", "BIDSVersion": "1.6.0"})
    )
    (root / "sub-01" / "func" / "sub-01_task-audioawake_run-01_bold.json").write_text(
        json.dumps({"RepetitionTime": 2.0})
    )
    if with_deriv:
        d = root / "derivatives" / "fmriprep" / "sub-01" / "func"
        d.mkdir(parents=True)
        name = (
            "sub-01_task-audioawake_run-01_space-MNI152NLin2009cAsym"
            "_desc-preproc_bold.nii.gz"
        )
        (d / name).write_bytes(b"")


def test_finds_fmriprep_derivatives(tmp_path):
    _bids_with_derivatives(tmp_path)
    root, n, _ = replication._find_fmriprep_derivatives(str(tmp_path))
    assert root.endswith("derivatives/fmriprep") and n == 1


def test_missing_derivatives_and_missing_dataset_raise_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Replication dataset not found"):
        replication.run_replication(
            str(tmp_path / "nope"),
            str(tmp_path / "out"),
            "schaefer400",
            ("awake", "deep"),
        )
    _bids_with_derivatives(tmp_path, with_deriv=False)
    with pytest.raises(RuntimeError, match="No fMRIPrep desc-preproc BOLD"):
        replication.run_replication(
            str(tmp_path), str(tmp_path / "out"), "schaefer400", ("awake", "deep")
        )


def test_run_replication_wiring_and_paired_statistics(tmp_path, monkeypatch):
    _bids_with_derivatives(tmp_path)
    calls = {}

    def fake_prep(**kw):
        calls["prep"] = kw
        return {
            "written_runs": [
                {"state": s, "condition": "audio"} for s in ("awake", "deep")
            ],
            "skipped_runs": [],
        }

    def fake_ci(data_dir, atlas, **kw):
        calls["ci"] = kw
        rng = np.random.RandomState(0)
        rows = []
        for i in range(12):
            base = rng.randn()
            rows.append(
                {"subject": f"s{i}", "session": "awake", "theta": 0.5, "S": base + 1.0}
            )
            rows.append(
                {
                    "subject": f"s{i}",
                    "session": "deep",
                    "theta": 0.5,
                    "S": base + rng.randn() * 0.1,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(replication, "run_preprocessing", fake_prep)
    monkeypatch.setattr(replication, "compute_synergy_ci", fake_ci)
    res = replication.run_replication(
        str(tmp_path),
        str(tmp_path / "out"),
        "schaefer400",
        ("awake", "deep"),
        n_boot=500,
    )
    # run_preprocessing gets all required arguments, out_root included.
    assert calls["prep"]["out_root"].endswith("preprocessed")
    assert calls["prep"]["fmriprep_deriv"].endswith("derivatives/fmriprep")
    # Only S is needed: no MPC metrics, so missing SRPI params cannot raise.
    assert calls["ci"]["compute_mpc"] is False and calls["ci"]["tr"] == 2.0
    # 'ci' is the CI of the paired mean difference, and cohend is dz.
    assert res["ci"][0] < res["delta_S"] < res["ci"][1]
    assert res["delta_S"] == pytest.approx(1.0, abs=0.15)
    assert res["cohend"] > 3 and res["n_pairs"] == 12


# -- Source gone (GitHub 404): fail fast unless a local copy is supplied -------


def test_replication_root_resolution_explicit_env_then_legacy(tmp_path, monkeypatch):
    monkeypatch.delenv(replication.REPLICATION_ROOT_ENV, raising=False)
    repo = tmp_path / "repo"
    # Nothing present: the first legacy candidate (missing) is returned.
    assert replication.default_replication_root(repo) == (
        repo / "data" / "scratch" / "melbourne"
    )
    (repo / "data" / "melbourne_propofol").mkdir(parents=True)
    assert replication.default_replication_root(repo) == (
        repo / "data" / "melbourne_propofol"
    )
    monkeypatch.setenv(replication.REPLICATION_ROOT_ENV, str(tmp_path / "envroot"))
    assert replication.resolve_replication_root(None, repo) == tmp_path / "envroot"
    explicit = tmp_path / "explicit"
    assert replication.resolve_replication_root(str(explicit), repo) == explicit


def test_missing_replication_data_message_is_actionable(tmp_path):
    with pytest.raises(replication.ReplicationDataUnavailable) as err:
        replication.check_replication_inputs(tmp_path / "melbourne")
    msg = str(err.value)
    for needle in (
        "Replication dataset not found",
        "HTTP 404",
        replication.MELBOURNE_SOURCE_URL,
        "--replication-root",
        replication.REPLICATION_ROOT_ENV,
        "without --run-replication",
    ):
        assert needle in msg
    assert isinstance(err.value, FileNotFoundError)
    with pytest.raises(replication.ReplicationDataUnavailable):
        replication.check_replication_inputs(None)
    # A folder that is not a BIDS dataset.
    (tmp_path / "notbids").mkdir()
    with pytest.raises(replication.ReplicationInputsMissing, match="not a BIDS"):
        replication.check_replication_inputs(tmp_path / "notbids")


def test_run_replication_fails_before_any_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        replication,
        "run_preprocessing",
        lambda **_: pytest.fail("preprocessing started without data"),
    )
    monkeypatch.setattr(
        replication,
        "compute_synergy_ci",
        lambda *a, **k: pytest.fail("metrics started without data"),
    )
    with pytest.raises(FileNotFoundError, match="--replication-root"):
        replication.run_replication(
            str(tmp_path / "gone"),
            str(tmp_path / "out"),
            "schaefer400",
            ("awake", "deep"),
        )


def test_run_preprocessing_is_called_with_its_real_signature(tmp_path, monkeypatch):
    import inspect

    from impact_pipeline import preprocessing

    _bids_with_derivatives(tmp_path)
    seen = {}

    def fake_prep(**kw):
        # Guards against run_preprocessing(data_root, prep) -> TypeError at runtime.
        inspect.signature(preprocessing.run_preprocessing).bind(**kw)
        seen.update(kw)
        return {
            "written_runs": [{"state": "awake", "condition": "audio"}],
            "skipped_runs": [{"task": "rest"}],
        }

    monkeypatch.setattr(replication, "run_preprocessing", fake_prep)
    with pytest.raises(replication.ReplicationInputsMissing, match="deep"):
        replication.run_replication(
            str(tmp_path), str(tmp_path / "out"), "schaefer400", ("awake", "deep")
        )
    assert seen["dataset_id"] == replication.REPLICATION_TASK_GRAMMAR
    assert tuple(seen["states"]) == ("awake", "deep")


def _local_copy(root, subjects):
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "local copy", "BIDSVersion": "1.6.0"})
    )
    for sub in subjects:
        func = root / f"sub-{sub}" / "func"
        deriv = root / "derivatives" / "fmriprep" / f"sub-{sub}" / "func"
        func.mkdir(parents=True)
        deriv.mkdir(parents=True)
        for task in ("audioawake", "audiodeep"):
            stem = f"sub-{sub}_task-{task}_run-01"
            (func / f"{stem}_bold.nii.gz").write_bytes(b"")
            (func / f"{stem}_bold.json").write_text(
                json.dumps({"RepetitionTime": 2.0})
            )
            space = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            (deriv / f"{stem}_{space}").write_bytes(b"")


def test_replication_end_to_end_on_a_local_copy(tmp_path, monkeypatch):
    """Real run_preprocessing (fMRIPrep cleaning stubbed) and real S."""
    from pathlib import Path

    from impact_pipeline import preprocessing

    root = tmp_path / "melbourne_copy"
    subjects = [f"{i:02d}" for i in range(1, 7)]
    _local_copy(root, subjects)

    def fake_preprocess_subject(
        bids_root, fmriprep_deriv, subj, bf, out_dir, assume_tr=None
    ):
        deep = "deep" in bf.entities["task"]
        rng = np.random.default_rng(int(subj) + (7 if deep else 0))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        run = int(bf.entities["run"])
        np.save(
            Path(out_dir) / f"{subj}_run-{run}_schaefer400_ts.npy",
            rng.standard_normal((80, 8)),
        )
        return {
            "subject": subj,
            "task": bf.entities["task"],
            "run": run,
            "tr_source": "sidecar",
        }

    monkeypatch.setattr(preprocessing, "preprocess_subject", fake_preprocess_subject)
    res = replication.run_replication(
        str(root), str(tmp_path / "out"), "schaefer400", ("awake", "deep"), n_boot=200
    )
    assert res["n_pairs"] == len(subjects)
    assert np.isfinite(res["delta_S"])
    assert res["ci"][0] <= res["delta_S"] <= res["ci"][1]
    assert (tmp_path / "out" / "preprocessed" / "01" / "deep" / "audio").is_dir()


def _pipeline_inputs(tmp_path):
    bids = tmp_path / "bids"
    (bids / "sub-01" / "func").mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "mini", "BIDSVersion": "1.8.0"})
    )
    out = tmp_path / "outputs"
    (out / "cache").mkdir(parents=True)
    return bids, out


def test_run_pipeline_replication_root_fails_fast_or_is_used(tmp_path, monkeypatch):
    import run_pipeline

    bids, out = _pipeline_inputs(tmp_path)
    monkeypatch.setattr(
        run_pipeline,
        "_persist_step2_outputs",
        lambda *a, **k: pytest.fail("metric computation reached"),
    )
    monkeypatch.setattr(
        run_pipeline,
        "_resolve_replication_root",
        lambda: pytest.fail("explicit --replication-root must win"),
    )
    kwargs = dict(
        dataset_id="ds003171",
        bids_root_override=str(bids),
        reuse_step2=True,
        mpc_metrics=["IIM"],
        compute_ci=False,
        run_replication_flag=True,
    )
    with pytest.raises(FileNotFoundError) as err:
        run_pipeline.main(str(out), replication_root=str(tmp_path / "nope"), **kwargs)
    assert "HTTP 404" in str(err.value) and "--replication-root" in str(err.value)
    # A supplied local copy passes the pre-check (the run then proceeds).
    _local_copy(tmp_path / "copy", ["01"])
    run_pipeline._check_replication_inputs(tmp_path / "copy")


def test_run_pipeline_cli_exposes_replication_root():
    import subprocess
    import sys

    import run_pipeline

    helptext = subprocess.run(
        [sys.executable, run_pipeline.__file__, "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    assert "--replication-root" in helptext


def test_cli_without_data_exits_2_with_the_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(replication.REPLICATION_ROOT_ENV, str(tmp_path / "none"))
    assert replication.main([]) == 2
    err = capsys.readouterr().err
    assert "HTTP 404" in err and "--replication-root" in err
    assert replication.main(["--replication-root", str(tmp_path / "x")]) == 2
