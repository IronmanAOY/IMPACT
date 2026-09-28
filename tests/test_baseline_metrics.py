import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from impact_pipeline.baseline_metrics import (
    compute_baseline_metrics,
    lempel_ziv_complexity,
    lzc,
    mean_conn,
    modularity,
    pci_fmri,
)


def test_mean_conn():
    ts = np.eye(3)
    assert np.isclose(mean_conn(ts), -0.5)


def test_modularity_no_louvain(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'community', None)
    ts = np.random.rand(5, 5)
    assert modularity(ts) >= 0


def test_pci_fmri():
    ts = np.zeros((4, 4))
    val = pci_fmri(ts)
    assert 0 <= val <= 1


def test_pci_fmri_is_degenerate_and_excluded():
    # Documented: ~1e-118 for 400 regions and proportional to scan length.
    t = np.random.RandomState(0).randn(200, 400)
    assert pci_fmri(t) < 1e-100
    assert pci_fmri(t[:100]) == pytest.approx(pci_fmri(t) / 2)


def test_lz76_known_answers():
    # Kaspar & Schuster (1987) example: 0·001·10·100·1000·101 -> 6 phrases.
    assert lempel_ziv_complexity([int(c) for c in "0001101001000101"]) == 6
    assert lempel_ziv_complexity([0] * 10) == 2
    assert lempel_ziv_complexity([0, 1] * 10) == 3


def test_lzc_separates_irregular_from_regular_activity():
    rng = np.random.RandomState(1)
    noise = lzc(rng.randn(300, 40))
    smooth = lzc(lfilter([1], [1, -0.97], rng.randn(300, 40), axis=0))
    assert noise == pytest.approx(1.0, abs=0.05)
    assert smooth < 0.9 * noise
    # Not driven by length or pattern counts: a fresh noise sample gives ~1 again.
    assert lzc(rng.randn(300, 40)) == pytest.approx(noise, abs=0.05)
    assert np.isnan(lzc(np.ones((50, 4))))


def test_modularity_is_seeded():
    ts = np.random.RandomState(2).randn(120, 30)
    assert modularity(ts) == modularity(ts)


def test_baseline_uses_analysed_condition_runs(tmp_path):
    rng = np.random.RandomState(3)
    for cond, n_runs in (("audio", 2), ("rest", 1)):
        d = tmp_path / "s1" / "awake" / cond
        d.mkdir(parents=True)
        for run in range(1, n_runs + 1):
            np.save(d / f"s1_run-{run}_schaefer400_ts.npy", rng.randn(80, 12))
    df = pd.DataFrame([{"subject": "s1", "session": "awake", "S": 0.1}])
    out = compute_baseline_metrics(df, str(tmp_path), "schaefer400", condition="audio")
    row = out.iloc[0]
    assert row["n_baseline_runs"] == 2
    assert "/audio/" in row["ts_path"] and "/rest/" not in row["ts_path"]
    assert {"mean_conn", "modularity", "lzc"}.issubset(out.columns)
    assert "pci_fmri" not in out.columns
    legacy = compute_baseline_metrics(df, str(tmp_path), "schaefer400")
    assert "/rest/" in legacy.iloc[0]["ts_path"]
