"""PDI baseline routing.

EEG/fMRI rest baselines live under <subj>/<session>/rest; otherwise NaN + reason.
"""

import shutil

import numpy as np
import pytest

from impact_pipeline.synergy_ci import compute_synergy_ci
from test_synergy_ci import _PDI_PARAMS


def _write(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return path


def _run(base, sessions=("awake", "deep"), condition="eeg", endpoint="anchor", **kw):
    return compute_synergy_ci(
        str(base),
        "eeg64",
        [0.5],
        sessions=sessions,
        condition=condition,
        mpc_metrics=("PDI",),
        compute_ci=False,
        pdi_params=_PDI_PARAMS,
        pdi_require_explicit_params=True,
        pdi_primary_endpoint=endpoint,
        pdi_require_strict_baseline=kw.pop("strict", True),
        **kw,
    )


def _eeg_layout(base, with_rest=True, seed=0):
    rng = np.random.RandomState(seed)
    for ses in ("awake", "deep"):
        _write(
            base / "1010" / ses / "eeg" / "1010_run-1_eeg64_ts.npy", rng.randn(300, 6)
        )
        if with_rest:
            _write(
                base / "1010" / ses / "rest" / "1010_run-1_eeg64_ts.npy",
                rng.randn(300, 6),
            )


def test_eeg_rest_folders_define_pdi_for_both_sessions(tmp_path):
    _eeg_layout(tmp_path)
    df = _run(tmp_path).set_index("session")
    for ses in ("awake", "deep"):
        assert np.isfinite(df.loc[ses, "PDI"])
        assert df.loc[ses, "PDI_anchor_reason"] == "ok"
        assert df.loc[ses, "PDI_primary_source"] == "anchor"
        assert df.loc[ses, "PDI_anchor_baseline_paths"].endswith(
            "deep/rest/1010_run-1_eeg64_ts.npy"
        )
        assert df.loc[ses, "PDI_task_baseline_paths"].endswith(
            f"{ses}/rest/1010_run-1_eeg64_ts.npy"
        )


def test_missing_rest_gives_nan_with_reason(tmp_path):
    _eeg_layout(tmp_path, with_rest=False)
    df = _run(tmp_path)
    assert df["PDI"].isna().all()
    assert (df["PDI_anchor_reason"] == "missing_deep_rest_baseline").all()
    assert (df["PDI_task_reason"] == "missing_state_rest_baseline").all()
    assert (df["PDI_primary_source"] == "undefined").all()


def test_task_run_is_never_its_own_baseline(tmp_path):
    _eeg_layout(tmp_path)
    # EEG preprocessing may write the same recording to <ses>/eeg and <ses>/rest.
    for ses in ("awake", "deep"):
        shutil.copy(
            tmp_path / "1010" / ses / "eeg" / "1010_run-1_eeg64_ts.npy",
            tmp_path / "1010" / ses / "rest" / "1010_run-1_eeg64_ts.npy",
        )
    df = _run(tmp_path, endpoint="task").set_index("session")
    for ses in ("awake", "deep"):
        # Old code: PDI_task == 0.0 exactly (a spurious measured zero).
        assert np.isnan(df.loc[ses, "PDI_task"])
        assert df.loc[ses, "PDI_task_reason"] == "state_baseline_identical_to_task_run"
    assert (
        df.loc["deep", "PDI_anchor_reason"] == "anchor_baseline_identical_to_task_run"
    )
    assert df.loc["awake", "PDI_anchor_reason"] == "ok"


def test_condition_rest_uses_other_rest_runs_as_baseline(tmp_path):
    rng = np.random.RandomState(1)
    for run in (1, 2):
        _write(
            tmp_path / "1010" / "deep" / "rest" / f"1010_run-{run}_eeg64_ts.npy",
            rng.randn(300, 6),
        )
    df = _run(tmp_path, sessions=("deep",), condition="rest", endpoint="task")
    assert len(df) == 2
    assert np.isfinite(df["PDI_task"]).all()
    assert (df["PDI_task_baseline_n_runs"] == 1).all()  # leave-one-run-out


def test_anchor_session_is_configurable(tmp_path):
    rng = np.random.RandomState(2)
    for ses in ("awake", "sed2"):
        _write(
            tmp_path / "1010" / ses / "eeg" / "1010_run-1_eeg64_ts.npy",
            rng.randn(300, 6),
        )
    _write(
        tmp_path / "1010" / "sed2" / "rest" / "1010_run-1_eeg64_ts.npy",
        rng.randn(300, 6),
    )
    df = _run(tmp_path, sessions=("awake", "sed2"))
    assert (df["PDI_anchor_reason"] == "missing_deep_rest_baseline").all()
    df2 = _run(tmp_path, sessions=("awake", "sed2"), pdi_anchor_session="sed2")
    assert (df2["PDI_anchor_reason"] == "ok").all()
    assert np.isfinite(df2["PDI"]).all()


def test_region_mismatch_reason(tmp_path):
    rng = np.random.RandomState(3)
    _write(
        tmp_path / "1010" / "awake" / "eeg" / "1010_run-1_eeg64_ts.npy",
        rng.randn(300, 6),
    )
    _write(
        tmp_path / "1010" / "deep" / "rest" / "1010_run-1_eeg64_ts.npy",
        rng.randn(300, 5),
    )
    df = _run(tmp_path, sessions=("awake",))
    assert df.loc[0, "PDI_anchor_reason"] == "anchor_baseline_region_mismatch"
    assert np.isnan(df.loc[0, "PDI"])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_legacy_fallback_keeps_task_baseline_provenance(tmp_path):
    rng = np.random.RandomState(4)
    _write(
        tmp_path / "1010" / "awake" / "eeg" / "1010_run-1_eeg64_ts.npy",
        rng.randn(300, 6),
    )
    _write(
        tmp_path / "1010" / "deep" / "rest" / "1010_run-1_eeg64_ts.npy",
        rng.randn(300, 5),
    )
    _write(
        tmp_path / "1010" / "recovery" / "rest" / "1010_run-1_eeg64_ts.npy",
        rng.randn(300, 6),
    )
    df = _run(tmp_path, sessions=("awake",), strict=False)
    row = df.iloc[0]
    assert row["PDI_primary_source"] == "legacy_rest_pool"
    assert np.isfinite(row["PDI"])
    assert np.isnan(row["PDI_task"])
    # PDI_task is undefined, so its baseline provenance must stay empty.
    assert row["PDI_task_baseline_paths"] == ""
    assert row["PDI_task_baseline_n_runs"] == 0
