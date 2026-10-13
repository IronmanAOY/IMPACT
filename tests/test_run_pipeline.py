import json

import numpy as np
import pandas as pd

import run_pipeline


def test_smoke_reuse_step2(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs"
    cache_dir = out_dir / "cache"
    prep_dir = out_dir / "preprocessed" / "sub-01" / "awake" / "audio"
    bids_dir = tmp_path / "bids"
    func_dir = bids_dir / "sub-01" / "func"

    cache_dir.mkdir(parents=True)
    prep_dir.mkdir(parents=True)
    func_dir.mkdir(parents=True)

    # Minimal fMRI sidecar required by step-6 TR lookup.
    (func_dir / "sub-01_task-audioawake_run-01_bold.json").write_text(
        json.dumps({"RepetitionTime": 2.0}),
        encoding="utf-8",
    )

    # Cached step-2 tables.
    pd.DataFrame(
        [
            {"subject": "01", "session": "awake", "theta": 0.6, "S": 0.2, "PDI": 0.1, "NAS": 0.3, "IIM": 0.9},
            {"subject": "01", "session": "deep", "theta": 0.6, "S": 0.1, "PDI": 0.05, "NAS": 0.31, "IIM": 0.92},
        ]
    ).to_csv(cache_dir / "step2_df.csv", index=False)
    pd.DataFrame(
        [
            {"subject": "01", "session": "awake", "S": 0.2},
            {"subject": "01", "session": "deep", "S": 0.1},
        ]
    ).to_csv(cache_dir / "step2_df_mean.csv", index=False)
    pd.DataFrame([{"theta": 0.6, "t_S": 1.0, "p_S": 0.3, "d_S": 0.5}]).to_csv(
        cache_dir / "step2_theta_stats.csv", index=False
    )

    monkeypatch.setattr(
        run_pipeline,
        "compute_baseline_metrics",
        lambda *a, **k: pd.DataFrame(
            [{"subject": "01", "session": "awake", "S": 0.2, "mean_conn": 0.1, "modularity": 0.2, "pci_fmri": 0.3},
             {"subject": "01", "session": "deep", "S": 0.1, "mean_conn": 0.1, "modularity": 0.2, "pci_fmri": 0.3}]
        ),
    )
    monkeypatch.setattr(run_pipeline, "bootstrap_ci", lambda *a, **k: (0.1, 0.9))
    monkeypatch.setattr(run_pipeline, "permutation_test_auc", lambda *a, **k: (0.6, 0.2))
    monkeypatch.setattr(run_pipeline, "motion_covariate_analysis", lambda *a, **k: {"coef_awake": [0.0], "p_awake": [1.0]})
    monkeypatch.setattr(run_pipeline, "atlas_check", lambda *a, **k: {"aal90": {"awake": 0.1, "deep": 0.1}})
    monkeypatch.setattr(run_pipeline, "compare_models", lambda *a, **k: {"mean_conn": {"delta_auc": 0.0, "p_val": 1.0}})
    monkeypatch.setattr(run_pipeline, "create_doc", lambda *a, **k: None)

    run_pipeline.main(
        str(out_dir),
        dataset_id="ds003171",
        bids_root_override=str(bids_dir),
        reuse_step2=True,
        mpc_metrics=["PDI", "NAS", "IIM"],
        compute_ci=False,
    )


def test_smoke_reuse_step2_ds005620(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs"
    cache_dir = (out_dir / "ds005620" / "cache")
    prep_dir = out_dir / "ds005620" / "preprocessed" / "sub-1010" / "awake" / "eeg"
    bids_dir = tmp_path / "bids_ds005620"

    cache_dir.mkdir(parents=True)
    prep_dir.mkdir(parents=True)
    (bids_dir / "sub-1010" / "eeg").mkdir(parents=True)

    pd.DataFrame(
        [
            {"subject": "1010", "session": "awake", "theta": 0.6, "S": 0.2, "PDI": 0.1, "NAS": 0.3, "IIM": 0.9},
            {"subject": "1010", "session": "deep", "theta": 0.6, "S": 0.1, "PDI": 0.05, "NAS": 0.31, "IIM": 0.92},
        ]
    ).to_csv(cache_dir / "step2_df.csv", index=False)
    pd.DataFrame(
        [
            {"subject": "1010", "session": "awake", "S": 0.2},
            {"subject": "1010", "session": "deep", "S": 0.1},
        ]
    ).to_csv(cache_dir / "step2_df_mean.csv", index=False)
    pd.DataFrame([{"theta": 0.6, "t_S": 1.0, "p_S": 0.3, "d_S": 0.5}]).to_csv(
        cache_dir / "step2_theta_stats.csv", index=False
    )

    monkeypatch.setattr(
        run_pipeline,
        "compute_baseline_metrics",
        lambda *a, **k: pd.DataFrame(
            [{"subject": "1010", "session": "awake", "S": 0.2, "mean_conn": 0.1, "modularity": 0.2, "pci_fmri": 0.3},
             {"subject": "1010", "session": "deep", "S": 0.1, "mean_conn": 0.1, "modularity": 0.2, "pci_fmri": 0.3}]
        ),
    )
    monkeypatch.setattr(run_pipeline, "bootstrap_ci", lambda *a, **k: (0.1, 0.9))
    monkeypatch.setattr(run_pipeline, "permutation_test_auc", lambda *a, **k: (0.6, 0.2))
    monkeypatch.setattr(run_pipeline, "compare_models", lambda *a, **k: {"mean_conn": {"delta_auc": 0.0, "p_val": 1.0}})
    monkeypatch.setattr(run_pipeline, "create_doc", lambda *a, **k: None)

    run_pipeline.main(
        str(out_dir),
        dataset_id="ds005620",
        bids_root_override=str(bids_dir),
        reuse_step2=True,
        mpc_metrics=["PDI", "NAS", "IIM"],
        compute_ci=False,
    )


def test_statistics_step_excludes_undefined_ci_and_writes_outputs(
    tmp_path, monkeypatch
):
    """Steps 4-9 on cached step-2 tables with real statistics (no stats mocks)."""
    out_dir = tmp_path / "outputs"
    cache_dir = out_dir / "cache"
    bids_dir = tmp_path / "bids"
    cache_dir.mkdir(parents=True)
    func = bids_dir / "sub-01" / "func"
    func.mkdir(parents=True)
    (func / "sub-01_task-audioawake_run-01_bold.json").write_text(
        json.dumps({"RepetitionTime": 2.0}), encoding="utf-8"
    )
    rng = np.random.RandomState(0)
    rows = []
    for i in range(8):
        subj = f"{i:02d}CB"
        base = rng.randn()
        for ses, shift in (("awake", 1.0), ("deep", 0.0)):
            prep = out_dir / "preprocessed" / subj / ses / "audio"
            prep.mkdir(parents=True)
            (prep / "mean_fd.txt").write_text(f"{rng.uniform(0.05, 0.4):.4f}")
            np.save(prep / f"{subj}_run-1_schaefer400_ts.npy", rng.randn(40, 5))
            undefined = (i == 0 and ses == "deep")
            ci = 1.0 + 0.4 * shift + base * 0.1 + rng.randn() * 0.05
            ci = np.nan if undefined else ci
            for theta in (0.3, 0.6):
                rows.append({
                    "subject": subj, "session": ses, "theta": theta,
                    "S": base + shift * theta + rng.randn() * 0.1,
                    "CI": ci, "CI_defined": not undefined,
                    "CI_missing": "SRPI" if undefined else "",
                    "CI_reference": "cohort_high_state",
                    "RAM": 0.1 + 0.05 * shift, "PDI": 0.02 + 0.01 * shift,
                    "NAS": 0.3, "IIM": 0.4,
                    "SRPI": np.nan if undefined else 0.5,
                })
    df = pd.DataFrame(rows)
    df.to_csv(cache_dir / "step2_df.csv", index=False)
    df.groupby(["subject", "session"], as_index=False).agg(
        S=("S", "mean"), CI=("CI", "mean"), RAM=("RAM", "mean"), PDI=("PDI", "mean"),
        NAS=("NAS", "mean"), IIM=("IIM", "mean"), SRPI=("SRPI", "mean"),
    ).to_csv(cache_dir / "step2_df_mean.csv", index=False)
    theta_rows = [
        {"theta": t, "t_S": 3.0, "p_S": p, "d_S": 1.0}
        for t, p in ((0.3, 0.01), (0.6, 0.04))
    ]
    pd.DataFrame(theta_rows).to_csv(cache_dir / "step2_theta_stats.csv", index=False)

    monkeypatch.setattr(
        run_pipeline, "atlas_check", lambda *a, **k: {"aal90": {"skipped": "test"}}
    )
    captured = {}
    monkeypatch.setattr(run_pipeline, "create_doc", lambda *a, **k: captured.update(k))

    run_pipeline.main(
        str(out_dir),
        dataset_id="ds003171",
        bids_root_override=str(bids_dir),
        reuse_step2=True,
        compute_ci=True,
    )
    summary = out_dir / "stats" / "statistics_summary.json"
    stats = json.loads(summary.read_text(encoding="utf-8"))
    cd = stats["ci_definedness"]
    # One undefined run (repeated at 2 thetas) is counted once, per run.
    assert cd["count_unit"] == "run" and cd["n_table_rows"] == 32
    assert cd["n_rows_undefined"] == 1 and cd["n_rows_defined"] == 15
    assert cd["n_subjects_complete_pairs"] == 7
    assert stats["ci_test"]["n_pairs"] == 7
    assert stats["ci_test"]["n_subjects_excluded"] == 1
    assert stats["ci_reference"]["ci_reference"] == "cohort_high_state"
    assert stats["auc_CI"]["p"] is not None and stats["auc_S"]["lo"] is not None
    theta_perm = stats["theta_permutation_S"]
    assert all("p_S_holm" in v for v in theta_perm.values())
    expected = {"mean_conn", "modularity", "lzc", "mean_conn [CI]"}
    assert set(stats["model_comparison"]) >= expected
    assert (out_dir / "stats" / "component_tests.csv").exists()
    theta_tab = pd.read_csv(out_dir / "stats" / "theta_tests_S.csv")
    np.testing.assert_allclose(theta_tab["p_S_holm"], [0.02, 0.04])
    motion = pd.read_csv(out_dir / "stats" / "motion_covariates.csv").iloc[0]
    assert motion["n_rows_undefined_score"] == 1 and motion["fd_condition"] == "audio"
    assert captured["stats"]["ci_definedness"]["n_rows_undefined"] == 1
    mc = stats["model_comparison"]["mean_conn"]
    assert {"delta_discrimination", "p_discrimination_holm"}.issubset(mc)
    assert captured["LABEL_CI"] == "CI (reference-normalised)"
    # CI status columns survive the metric-subset filter.
    assert {"CI_defined", "CI_missing", "CI_reference"}.issubset(captured["df"].columns)
