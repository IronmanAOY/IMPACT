import numpy as np
import pandas as pd
import pytest

from impact_pipeline.motion_model import _weighted_session_fd, motion_covariate_analysis


def test_motion_model(tmp_path):
    data_dir = tmp_path / "data"
    row = {'subject': 's1', 'session': 'awake', 'S': 0.5, 'CI': 0.5}
    df = pd.DataFrame([row])
    d = data_dir / "s1" / "awake"
    d.mkdir(parents=True)
    (d / "mean_fd.txt").write_text("0.2")
    res = motion_covariate_analysis(df, str(data_dir))
    assert 'coef_awake' in res and 'p_awake' in res


def _cond(root, subj, ses, cond, fd, n_time, n_regions=7):
    d = root / subj / ses / cond
    d.mkdir(parents=True, exist_ok=True)
    (d / "mean_fd.txt").write_text(str(fd))
    np.save(d / f"{subj}_run-1_schaefer400_ts.npy", np.zeros((n_time, n_regions)))


def test_fd_is_weighted_by_timepoints_not_regions(tmp_path):
    _cond(tmp_path, "s1", "awake", "audio", 1.0, n_time=300)
    _cond(tmp_path, "s1", "awake", "rest", 0.0, n_time=100)
    # (1.0*300 + 0.0*100) / 400 = 0.75; weighting by n_regions would give 0.5
    root = str(tmp_path)
    fd_all = _weighted_session_fd(root, "s1", "awake", "schaefer400")
    assert fd_all == pytest.approx(0.75)
    fd_audio = _weighted_session_fd(
        root, "s1", "awake", "schaefer400", condition="audio"
    )
    assert fd_audio == pytest.approx(1.0)


def test_missing_fd_and_undefined_ci_are_excluded_and_counted(tmp_path):
    rng = np.random.RandomState(0)
    rows = []
    for i in range(8):
        for ses in ("awake", "deep"):
            fd = rng.uniform(0.05, 0.5)
            if not (i == 0 and ses == "deep"):
                _cond(tmp_path, f"s{i}", ses, "audio", fd, n_time=50)
            rows.append({"subject": f"s{i}", "session": ses, "CI": rng.rand()})
    df = pd.DataFrame(rows)
    df.loc[3, "CI"] = np.nan
    res = motion_covariate_analysis(
        df, str(tmp_path), condition="audio", sessions=("awake", "deep")
    )
    assert isinstance(res, pd.DataFrame)
    r = res.iloc[0]
    assert r["n_subject_sessions_missing_fd"] == 1  # counted, no FileNotFoundError
    assert r["n_rows_undefined_score"] == 1
    assert r["n_awake"] + r["n_deep"] == 14
    assert r["n_pairs"] == 6


def test_fd_adjusted_state_contrast_recovers_true_difference(tmp_path):
    rng = np.random.RandomState(1)
    rows = []
    for i in range(30):
        base = rng.randn()
        for ses, state in (("awake", 1.0), ("deep", 0.0)):
            fd = rng.uniform(0.05, 0.6)
            _cond(tmp_path, f"s{i}", ses, "audio", fd, n_time=40)
            rows.append({"subject": f"s{i}", "session": ses,
                         "CI": base + 0.5 * state + 2.0 * fd + rng.randn() * 0.05})
    res = motion_covariate_analysis(
        pd.DataFrame(rows), str(tmp_path), condition="audio", sessions=("awake", "deep")
    ).iloc[0]
    assert res["delta_intercept"] == pytest.approx(0.5, abs=0.05)
    assert res["delta_fd_coef"] == pytest.approx(2.0, abs=0.15)
    assert res["p_delta_intercept"] < 1e-6


def _run(root, subj, ses, cond, run, fd, n_time, n_regions=7):
    d = root / subj / ses / cond
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{subj}_run-{run}_mean_fd.txt").write_text(str(fd))
    np.save(d / f"{subj}_run-{run}_schaefer400_ts.npy", np.zeros((n_time, n_regions)))
    return d


def test_per_run_fd_is_weighted_by_that_runs_timepoints(tmp_path):
    d = _run(tmp_path, "s1", "awake", "audio", 1, 1.0, n_time=300)
    _run(tmp_path, "s1", "awake", "audio", 2, 0.0, n_time=100)
    # The folder-level file (from a stale or overwritten write) is not used
    # when per-run files exist.
    (d / "mean_fd.txt").write_text("0.5")
    fd = _weighted_session_fd(str(tmp_path), "s1", "awake", "schaefer400", "audio")
    assert fd == pytest.approx((1.0 * 300 + 0.0 * 100) / 400)
    # Pooling conditions mixes per-run and legacy (folder-level) folders.
    legacy = tmp_path / "s1" / "awake" / "rest"
    legacy.mkdir()
    (legacy / "mean_fd.txt").write_text("0.2")
    np.save(legacy / "s1_run-1_schaefer400_ts.npy", np.zeros((200, 7)))
    fd_all = _weighted_session_fd(str(tmp_path), "s1", "awake", "schaefer400")
    assert fd_all == pytest.approx((300.0 + 0.0 + 0.2 * 200) / 600)


def test_undefined_run_fd_is_excluded_not_imputed(tmp_path):
    _run(tmp_path, "s1", "awake", "audio", 1, "nan", n_time=300)
    _run(tmp_path, "s1", "awake", "audio", 2, 0.4, n_time=100)
    fd = _weighted_session_fd(str(tmp_path), "s1", "awake", "schaefer400", "audio")
    assert fd == pytest.approx(0.4)
    _run(tmp_path, "s2", "awake", "audio", 1, "nan", n_time=50)
    assert np.isnan(
        _weighted_session_fd(str(tmp_path), "s2", "awake", "schaefer400", "audio")
    )
    df = pd.DataFrame(
        [
            {"subject": "s1", "session": "awake", "CI": 0.5},
            {"subject": "s2", "session": "awake", "CI": 0.4},
        ]
    )
    res = motion_covariate_analysis(df, str(tmp_path), condition="audio").iloc[0]
    assert res["n_subject_sessions_missing_fd"] == 1
