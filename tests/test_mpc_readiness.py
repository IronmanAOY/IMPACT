import numpy as np
import pandas as pd

from impact_pipeline.mpc_readiness import check_mpc_readiness


def _write_events(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)


def _write_ts(path, n_time=80, n_regions=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    arr = rng.randn(n_time, n_regions)
    np.save(path, arr)


def test_readiness_all_mpcs_ready(tmp_path):
    prep = tmp_path / "preprocessed"
    bids = tmp_path / "bids"
    subj = "01"
    ses = "awake"

    ts_path = prep / subj / ses / "audio" / f"{subj}_run-1_schaefer400_ts.npy"
    _write_ts(ts_path)
    _write_ts(prep / subj / ses / "rest" / f"{subj}_run-1_schaefer400_ts.npy")
    _write_ts(prep / subj / "deep" / "rest" / f"{subj}_run-1_schaefer400_ts.npy")

    ev_path = bids / f"sub-{subj}" / "func" / f"sub-{subj}_task-audioawake_run-01_events.tsv"
    # RAM needs goal, stimulus and valued feedback events (>= 6 stimulus
    # events for the cross-validated goal alignment, >= 3 feedback values).
    rows = []
    for i in range(6):
        t0 = 0.2 + 1.8 * i
        for dt, label, reward in (
            (0.0, "goal_cue", np.nan),
            (0.6, "audio_stim", np.nan),
            (1.2, "feedback", float(i % 3)),
        ):
            row = {"onset": t0 + dt, "duration": 0.1, "trial_type": label}
            rows.append(dict(row, reward=reward))
    for i in range(3):
        t0 = 11.0 + 1.2 * i
        rows.append({"onset": t0, "duration": 0.1, "trial_type": "self_name"})
        rows.append({"onset": t0 + 0.6, "duration": 0.1, "trial_type": "other_name"})
    _write_events(ev_path, rows)

    df, summary = check_mpc_readiness(
        prep_root=str(prep),
        bids_root=str(bids),
        atlas="schaefer400",
        condition="audio",
        sessions=[ses],
    )
    assert df.shape[0] == 1
    row = df.iloc[0]
    assert bool(row["RAM_ready"])
    assert bool(row["PDI_ready"])
    assert bool(row["PDI_anchor_ready"])
    assert bool(row["PDI_task_ready"])
    assert bool(row["NAS_ready"])
    assert bool(row["IIM_ready"])
    assert bool(row["SRPI_ready"])
    assert bool(row["CI_ready"])
    assert summary["metrics"]["CI"]["ready"] == 1


def test_readiness_ram_not_ready_without_feedback(tmp_path):
    prep = tmp_path / "preprocessed"
    bids = tmp_path / "bids"
    subj = "02"
    ses = "awake"

    ts_path = prep / subj / ses / "audio" / f"{subj}_run-1_schaefer400_ts.npy"
    _write_ts(ts_path)

    ev_path = bids / f"sub-{subj}" / "func" / f"sub-{subj}_task-audioawake_run-01_events.tsv"
    _write_events(
        ev_path,
        [
            {"onset": 0.30, "duration": 0.1, "trial_type": "audio_stim"},
            {"onset": 1.30, "duration": 0.1, "trial_type": "audio_stim"},
        ],
    )

    df, _ = check_mpc_readiness(
        prep_root=str(prep),
        bids_root=str(bids),
        atlas="schaefer400",
        condition="audio",
        sessions=[ses],
    )
    row = df.iloc[0]
    assert not bool(row["RAM_ready"])
    assert not bool(row["PDI_ready"])
    assert not bool(row["PDI_anchor_ready"])
    assert not bool(row["PDI_task_ready"])
    assert row["RAM_reason"] in {"missing_feedback_events", "missing_or_nonvarying_feedback_values"}
    assert not bool(row["CI_ready"])
