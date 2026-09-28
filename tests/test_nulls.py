"""Known-answer tests for the generic surrogate generators (impact_pipeline.nulls)."""
import numpy as np
import pandas as pd
import pytest

from impact_pipeline import nulls


def _coupled_pair(seed=0, n_time=2048, lag=3):
    """Node 1 is a lagged, noisy copy of node 0 (both AR(1))."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_time)
    for t in range(1, n_time):
        x[t] = 0.7 * x[t - 1] + rng.standard_normal()
    y = np.roll(x, lag) + 0.3 * rng.standard_normal(n_time)
    return np.vstack([x, y])


def _amp(ts):
    return np.abs(np.fft.rfft(ts, axis=1))


def _cross(ts):
    f = np.fft.rfft(ts - ts.mean(axis=1, keepdims=True), axis=1)
    return f[0] * np.conj(f[1])


def _xcorr_peak(ts, max_lag=10):
    a = (ts[0] - ts[0].mean()) / ts[0].std()
    b = (ts[1] - ts[1].mean()) / ts[1].std()
    return max(
        abs(float(np.mean(a * np.roll(b, -k)))) for k in range(-max_lag, max_lag + 1)
    )


def test_circular_shift_preserves_spectra_and_marginals_and_destroys_coupling():
    ts = _coupled_pair()
    assert _xcorr_peak(ts) > 0.9
    peaks = []
    for seed in range(20):
        s = nulls.circular_shift(ts, np.random.default_rng(seed))
        np.testing.assert_allclose(_amp(s), _amp(ts), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(np.sort(s, axis=1), np.sort(ts, axis=1))
        peaks.append(_xcorr_peak(s))
    # Every surrogate loses the lag-3 coupling (only the AR tail can survive).
    assert max(peaks) < 0.25


def test_circular_shift_respects_min_shift_and_rejects_short_runs():
    ts = np.arange(40, dtype=float)[None, :].repeat(3, axis=0)
    for seed in range(50):
        s = nulls.circular_shift(ts, np.random.default_rng(seed), min_shift=10)
        shifts = [int(np.argmin(np.abs(row - 0.0))) for row in s]
        assert shifts[0] == 0  # node 0 is the reference
        assert all(10 <= sh <= 30 for sh in shifts[1:])
    with pytest.raises(ValueError, match="too short"):
        nulls.circular_shift(ts, 0, min_shift=25)


def test_multivariate_phase_randomisation_preserves_spectra_and_cross_correlation():
    ts = _coupled_pair()
    for seed in range(5):
        s = nulls.phase_randomize(ts, np.random.default_rng(seed), multivariate=True)
        np.testing.assert_allclose(_amp(s), _amp(ts), rtol=1e-7, atol=1e-7)
        # Shared phases keep every cross-spectrum, hence the circular
        # cross-covariance at every lag.
        np.testing.assert_allclose(_cross(s), _cross(ts), rtol=1e-6, atol=1e-6)
        assert _xcorr_peak(s) == pytest.approx(_xcorr_peak(ts), abs=1e-6)
        assert not np.allclose(s, ts)


def test_univariate_phase_randomisation_destroys_cross_correlation():
    ts = _coupled_pair()
    peaks = []
    for seed in range(20):
        s = nulls.phase_randomize(ts, np.random.default_rng(seed), multivariate=False)
        np.testing.assert_allclose(_amp(s), _amp(ts), rtol=1e-7, atol=1e-7)
        peaks.append(_xcorr_peak(s))
    assert np.median(peaks) < 0.2


def test_iaaft_preserves_marginals_exactly_and_spectra_approximately():
    rng = np.random.default_rng(1)
    noise = 0.1 * rng.standard_normal((2, 2048))
    ts = np.exp(0.5 * _coupled_pair(seed=2)) + noise  # skewed marginals

    def rel_err(s):
        return np.linalg.norm(_amp(s) - _amp(ts)) / np.linalg.norm(_amp(ts))

    s = nulls.iaaft(ts, np.random.default_rng(0))
    np.testing.assert_allclose(np.sort(s, axis=1), np.sort(ts, axis=1))
    assert rel_err(s) < 0.05
    # the few default iterations already beat the plain rank-remapped surrogate
    errs = [rel_err(nulls.iaaft(ts, np.random.default_rng(0), n_iter=k))
            for k in (0, 3, 10)]
    assert errs[0] > errs[1] > errs[2]
    # shared initial phases keep most of the cross-correlation ...
    assert _xcorr_peak(s) > 0.7 * _xcorr_peak(ts)
    # ... independent ones destroy it
    si = nulls.iaaft(ts, np.random.default_rng(0), multivariate=False)
    assert _xcorr_peak(si) < 0.3
    with pytest.raises(ValueError, match="finite"):
        nulls.iaaft(np.array([[1.0, np.nan, 2.0, 3.0]]), 0)


def test_label_permutation_preserves_counts_within_phase_bins():
    rng = np.random.default_rng(0)
    n = 60
    ev = pd.DataFrame(
        {
            "onset": np.arange(n, dtype=float),
            "trial_type": np.where(
                np.arange(n) % 3 == 0, "self_caused", "other_caused"
            ),
            "phase_bin": np.repeat([0, 1, 2], n // 3),
        }
    )
    ev.loc[ev.index % 7 == 0, "trial_type"] = "goal_cue"
    changed = False
    for seed in range(10):
        out = nulls.permute_labels(
            ev, np.random.default_rng(seed), among=("self_caused", "other_caused")
        )
        np.testing.assert_array_equal(out["onset"], ev["onset"])
        # goal cues are not exchanged
        assert (out["trial_type"] == "goal_cue").equals(ev["trial_type"] == "goal_cue")
        before = ev.groupby(["phase_bin", "trial_type"]).size()
        after = out.groupby(["phase_bin", "trial_type"]).size()
        pd.testing.assert_series_equal(before, after)
        changed |= not out["trial_type"].equals(ev["trial_type"])
    assert changed
    assert ev["trial_type"].iloc[0] == "goal_cue"  # input not modified
    del rng


def test_label_permutation_on_event_bundles():
    bundle = {
        "onsets": [1.0, 2.0],
        "self_onsets": [1.0, 3.0, 5.0],
        "nonself_onsets": [2.0, 4.0, 6.0, 8.0],
        "feedback_onsets": [1.5, 2.5, 3.5],
        "feedback_values": [1.0, -1.0, 0.5],
    }
    pooled = sorted(bundle["self_onsets"] + bundle["nonself_onsets"])
    seen = set()
    for seed in range(20):
        out = nulls.permute_labels(bundle, np.random.default_rng(seed))
        assert len(out["self_onsets"]) == 3 and len(out["nonself_onsets"]) == 4
        assert sorted(out["self_onsets"] + out["nonself_onsets"]) == pooled
        assert sorted(out["feedback_values"]) == sorted(bundle["feedback_values"])
        assert out["feedback_onsets"] == bundle["feedback_onsets"]
        assert out["onsets"] == bundle["onsets"]
        seen.add(tuple(out["self_onsets"]))
    assert len(seen) > 5
    assert bundle["self_onsets"] == [1.0, 3.0, 5.0]
    with pytest.raises(ValueError, match="labels"):
        nulls.permute_labels([1.0, 2.0], 0)


def test_onset_jitter_modes():
    ev = pd.DataFrame({"onset": [1.0, 5.0, 9.0], "trial_type": ["a", "b", "c"]})
    out = nulls.jitter_onsets(ev, np.random.default_rng(0), max_jitter=0.5)
    assert np.all(np.abs(out["onset"] - ev["onset"]) <= 0.5)
    assert list(out["trial_type"]) == ["a", "b", "c"]
    # rigid (common) shift keeps inter-event intervals modulo the run length
    bundle = {"onsets": [1.0, 5.0], "feedback_onsets": [2.0, 6.0],
              "feedback_values": [1.0, -1.0]}
    for seed in range(20):
        out = nulls.jitter_onsets(bundle, np.random.default_rng(seed), t_max=20.0,
                                  common=True, min_shift=2.0)
        shift = (out["onsets"][0] - 1.0) % 20.0
        assert 2.0 <= shift <= 18.0
        for key in ("onsets", "feedback_onsets"):
            expect = (np.asarray(bundle[key]) + shift) % 20.0
            np.testing.assert_allclose(out[key], expect, atol=1e-9)
        assert out["feedback_values"] == bundle["feedback_values"]
    # uniform re-placement within the run
    arr = nulls.jitter_onsets([1.0, 2.0, 3.0], np.random.default_rng(0), t_max=10.0)
    assert len(arr) == 3 and all(0.0 <= v < 10.0 for v in arr)
    with pytest.raises(ValueError):
        nulls.jitter_onsets([1.0], 0)


def test_component_null_known_answer_and_determinism():
    ts = _coupled_pair(n_time=1024)

    def corr(x, _events):
        return float(np.corrcoef(x[0], np.roll(x[1], -3))[0, 1])

    res = nulls.component_null(corr, ts, None, "circular_shift", n=60, seed=7)
    null_mean, null_sd, samples, n_failed = res  # unpacks as a 4-tuple
    assert n_failed == 0 and samples.shape == (60,)
    assert abs(null_mean) < 0.1 and 0.0 < null_sd < 0.2
    assert (corr(ts, None) - null_mean) / null_sd > 10
    again = nulls.component_null(corr, ts, None, kind="circular", n=60, seed=7)
    np.testing.assert_array_equal(again.samples, samples)
    other = nulls.component_null(corr, ts, None, "circular_shift", n=60, seed=8)
    assert not np.array_equal(other.samples, samples)
    # the multivariate phase null keeps the coupling: no excess
    ph = nulls.component_null(corr, ts, None, "phase", n=20, seed=0)
    assert ph.null_mean == pytest.approx(corr(ts, None), abs=0.05)


def test_component_null_counts_failures_and_handles_event_nulls():
    ts = np.random.default_rng(0).standard_normal((2, 100))
    bundle = {"self_onsets": [1.0, 2.0, 3.0], "nonself_onsets": [4.0, 5.0, 6.0]}
    calls = {"n": 0}

    def flaky(_ts, ev):
        calls["n"] += 1
        if calls["n"] % 2:
            return float("nan")
        if calls["n"] % 4 == 0:
            raise ValueError("degenerate surrogate")
        return {"value": float(np.mean(ev["self_onsets"]))}

    res = nulls.component_null(flaky, ts, bundle, surrogate="label_permutation",
                               n=12, seed=0)
    assert res.n_failed == 9 and res.samples.size == 3
    assert np.all((res.samples >= 2.0) & (res.samples <= 5.0))
    # a surrogate that cannot be generated fails the whole null
    short = nulls.component_null(lambda x, e: 1.0, ts[:, :5], None,
                                 "circular_shift", n=4, seed=0, min_shift=3)
    assert short.n_failed == 4 and np.isnan(short.null_mean)
    # programming errors propagate
    with pytest.raises(TypeError):
        nulls.component_null(lambda x, e: 1.0 + None, ts, None, n=2)
    with pytest.raises(ValueError, match="unknown surrogate"):
        nulls.component_null(lambda x, e: 1.0, ts, None, kind="bogus", n=1)


def test_derive_seed_is_stable_and_label_specific():
    a = nulls.derive_seed(0, "RAM", "s1/awake/audio/f.npy")
    assert a == nulls.derive_seed(0, "RAM", "s1/awake/audio/f.npy")
    assert a != nulls.derive_seed(0, "SRPI", "s1/awake/audio/f.npy")
    assert a != nulls.derive_seed(1, "RAM", "s1/awake/audio/f.npy")
    assert 0 <= a < 2 ** 32
