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
    # run_preprocessing gets all required arguments (old code omitted out_root).
    assert calls["prep"]["out_root"].endswith("preprocessed")
    assert calls["prep"]["fmriprep_deriv"].endswith("derivatives/fmriprep")
    # Only S is needed: no MPC metrics (old code raised on missing SRPI params).
    assert calls["ci"]["compute_mpc"] is False and calls["ci"]["tr"] == 2.0
    # 'ci' is now the CI of the paired mean difference, and cohend is dz.
    assert res["ci"][0] < res["delta_S"] < res["ci"][1]
    assert res["delta_S"] == pytest.approx(1.0, abs=0.15)
    assert res["cohend"] > 3 and res["n_pairs"] == 12
