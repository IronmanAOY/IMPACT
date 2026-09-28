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


def test_derive_seed_known_values():
    # sha256-based: identical on every platform and Python process
    assert nulls.derive_seed(0, "RAM", "s1/awake/audio/f.npy") == 1689109511
    assert nulls.derive_seed(7, "IIM", "a/b/c/d.npy") == 217286542


def test_min_shift_requires_a_common_shift():
    # an independent jitter has no minimum displacement; ignoring min_shift
    # silently would misdeclare the null family
    with pytest.raises(ValueError, match="common"):
        nulls.jitter_onsets([1.0, 2.0], 0, max_jitter=1.0, min_shift=0.5)
    with pytest.raises(ValueError, match="common"):
        nulls.jitter_onsets([1.0, 2.0], 0, t_max=10.0, min_shift=0.5)
    out = nulls.jitter_onsets([1.0, 2.0], 0, max_jitter=1.0, min_shift=0.5,
                              common=True)
    assert 0.5 <= abs(out[0] - 1.0) <= 1.0
    assert out[1] - out[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# moving-block bootstrap (sampling SE of the evidence layer)
# --------------------------------------------------------------------------
def _ar1(seed, n_time, phi, n_nodes=2):
    rng = np.random.default_rng(seed)
    x = np.zeros((n_nodes, n_time))
    e = rng.standard_normal((n_nodes, n_time))
    for t in range(1, n_time):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def test_block_bootstrap_resamples_contiguous_blocks_deterministically():
    x = np.arange(3 * 50, dtype=float).reshape(3, 50)
    reps = list(nulls.block_bootstrap(x, None, 7, 5, seed=3))
    assert len(reps) == 5
    for ts_b, ev_b in reps:
        assert ts_b.shape == x.shape and ev_b is None
        # each block of 7 samples is a contiguous run of the original
        idx = ts_b[0].astype(int)
        for start in range(0, 50, 7):
            assert np.all(np.diff(idx[start:start + 7]) == 1)
        np.testing.assert_array_equal(ts_b[1] - ts_b[0], 50.0)  # nodes stay aligned
    again = list(nulls.block_bootstrap(x, None, 7, 5, seed=3))
    for (a, _), (b, _) in zip(reps, again):
        np.testing.assert_array_equal(a, b)
    assert nulls.default_block_len(400) == 20 and nulls.default_block_len(401) == 21
    with pytest.raises(ValueError, match="block_len"):
        nulls.block_bootstrap(x, None, 51, 1)
    with pytest.raises(ValueError, match="block_len"):
        nulls.block_bootstrap(x, None, 0, 1)
    with pytest.raises(ValueError, match="tr"):
        nulls.block_bootstrap(x, [1.0], 5, 1, tr=0.0)


def test_block_bootstrap_moves_events_with_their_samples():
    """Known answer: an event keeps its sample (ts_b at the new onset equals
    ts at the old onset); events of undrawn blocks are dropped and events of
    blocks drawn twice are duplicated."""
    tr = 0.5
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 400))
    samples = np.sort(rng.choice(400, size=60, replace=False))
    events = pd.DataFrame({
        "onset": samples * tr,
        "trial_type": ["stim"] * 60,
        "sample": samples,
    })
    n_total = 0
    for ts_b, ev_b in nulls.block_bootstrap(x, events, 25, 20, seed=1, tr=tr):
        new_idx = np.rint(ev_b["onset"].to_numpy() / tr).astype(int)
        np.testing.assert_array_equal(ts_b[:, new_idx], x[:, ev_b["sample"]])
        assert np.all(np.diff(ev_b["onset"].to_numpy()) >= 0)
        n_total += len(ev_b)
    assert 0.8 * 60 * 20 < n_total < 1.2 * 60 * 20  # on average all events kept
    # a plain onset list maps onto the resampled series as well
    ts_b, on_b = next(nulls.block_bootstrap(x, list(samples * tr), 25, 1, seed=2,
                                            tr=tr))
    moved = ts_b[0, np.rint(np.asarray(on_b) / tr).astype(int)]
    assert set(moved) <= set(x[0, samples])


def test_block_bootstrap_keeps_bundle_lists_aligned():
    tr = 1.0
    x = np.vstack([np.arange(200, dtype=float), np.zeros(200)])
    fb = [float(s) for s in range(5, 200, 10)]
    bundle = {
        "onsets": [float(s) for s in range(3, 200, 7)],
        "feedback_onsets": fb,
        "feedback_values": [s * 10.0 for s in fb],  # value encodes its onset
        "choice_onsets": fb,
        "choices": [int(s) for s in fb],
        "rewards": [-s for s in fb],
        "self_onsets": fb[::2],
        "self_onsets_phase_bin": [int(s) % 3 for s in fb[::2]],
        "impact_channels": ["behavioural_feedback"],
        "channel_bundles": {
            "behavioural_feedback": {
                "onsets": fb, "feedback_onsets": fb, "feedback_values": fb,
            },
        },
        "agency_events": {"onset": [10.0, 12.0, 150.0], "event_id": [1, 2, 3],
                          "yoked_to": [None, 1, None],
                          "trial_type": ["self_caused", "other_caused",
                                         "self_caused"]},
        "note": "untouched",
    }
    n_yoked = 0
    for ts_b, b in nulls.block_bootstrap(x, bundle, 20, 10, seed=4, tr=tr):
        # every onset list maps back to the original samples it came from
        for key in ("onsets", "feedback_onsets", "choice_onsets", "self_onsets"):
            src = ts_b[0, np.rint(np.asarray(b[key]) / tr).astype(int)]
            assert set(src) <= set(bundle[key])
        src_fb = ts_b[0, np.rint(np.asarray(b["feedback_onsets"])).astype(int)]
        np.testing.assert_array_equal(b["feedback_values"], src_fb * 10.0)
        np.testing.assert_array_equal(b["choices"], src_fb.astype(int))
        np.testing.assert_array_equal(b["rewards"], -src_fb)
        src_self = ts_b[0, np.rint(np.asarray(b["self_onsets"])).astype(int)]
        np.testing.assert_array_equal(b["self_onsets_phase_bin"],
                                      src_self.astype(int) % 3)
        sub = b["channel_bundles"]["behavioural_feedback"]
        src_sub = ts_b[0, np.rint(np.asarray(sub["feedback_onsets"])).astype(int)]
        np.testing.assert_array_equal(sub["feedback_values"], src_sub)
        assert b["impact_channels"] == ["behavioural_feedback"]
        assert b["note"] == "untouched"
        ag = pd.DataFrame(b["agency_events"])
        ids = dict(zip(ag["event_id"], ag["onset"]))
        for y, on in zip(ag["yoked_to"], ag["onset"]):
            if isinstance(y, str):
                # the replay points at its self-caused event in the same copy
                assert y.startswith("1@b") and on - ids[y] == pytest.approx(2.0)
                n_yoked += 1
    assert n_yoked > 0
    with pytest.raises(ValueError, match="align"):
        next(nulls.block_bootstrap(x, {"feedback_onsets": [1.0, 2.0],
                                       "feedback_values": [1.0]}, 20, 1, tr=tr))


def test_onset_valued_yoking_moves_with_its_block():
    x = np.zeros((1, 100))
    ev = pd.DataFrame({"onset": [10.0, 11.0, 50.0],
                       "trial_type": ["self_caused", "other_caused", "self_caused"],
                       "yoked_to": [np.nan, 10.0, np.nan]})
    n_kept = n_self = 0
    for ts_b, ev_b in nulls.block_bootstrap(x, ev, 20, 30, seed=0, tr=1.0):
        # a replay is kept only together with its self-caused event
        replays = ev_b[ev_b["trial_type"] == "other_caused"]
        assert replays["yoked_to"].notna().all()
        for on, y in zip(replays["onset"], replays["yoked_to"]):
            assert on - y == pytest.approx(1.0)
            assert y in set(ev_b.loc[ev_b["trial_type"] == "self_caused", "onset"])
            n_kept += 1
        n_self += int((ev_b["trial_type"] == "self_caused").sum())
    assert n_kept > 0 and n_self > n_kept  # unpaired self events stay


def test_component_bootstrap_se_known_answers():
    # iid noise: the SE of the mean is sigma / sqrt(T)
    n_time = 4000
    x = np.random.default_rng(5).standard_normal((1, n_time))

    def mean0(ts, _ev):
        return float(ts[0].mean())

    res = nulls.component_bootstrap_se(mean0, x, n=400, seed=0)
    assert res.block_len == 64 and res.n_failed == 0 and res.samples.size == 400
    assert res.se == pytest.approx(1.0 / np.sqrt(n_time), rel=0.15)
    # AR(1), phi = 0.8: SE of the mean = sd_x / sqrt(T) * sqrt((1+phi)/(1-phi));
    # the blocks keep the dependence (an iid resample misses the factor 3)
    ar = _ar1(6, n_time, 0.8, n_nodes=1)
    res = nulls.component_bootstrap_se(mean0, ar, n=400, seed=0, block_len=200)
    expected = np.sqrt(1 / (1 - 0.64)) / np.sqrt(n_time) * 3.0
    assert res.se == pytest.approx(expected, rel=0.25)
    iid = nulls.component_bootstrap_se(mean0, ar, n=400, seed=0, block_len=1)
    assert iid.se < 0.5 * res.se
    again = nulls.component_bootstrap_se(mean0, ar, n=400, seed=0, block_len=200)
    np.testing.assert_array_equal(again.samples, res.samples)

    def details(ts, _ev):
        m = float(ts[0].mean())
        if m > 0.05:
            raise ValueError("degenerate replicate")
        return {"value": np.nan, "raw": m}

    out = nulls.component_bootstrap_se(details, ar, n=200, seed=1, block_len=200,
                                       statistic=lambda d: d["raw"])
    assert out.n_failed > 0 and out.samples.size + out.n_failed == 200
    assert np.all(out.samples <= 0.05)
    one = nulls.component_bootstrap_se(mean0, ar, n=1, seed=0)
    assert np.isnan(one.se) and one.samples.size == 1

    def bug(ts, _ev):
        raise TypeError("a bug is not a failed replicate")

    with pytest.raises(TypeError):
        nulls.component_bootstrap_se(bug, ar, n=2)


def test_component_bootstrap_se_with_events():
    """An event-locked statistic keeps its meaning under the resampling."""
    tr = 1.0
    rng = np.random.default_rng(8)
    x = rng.standard_normal((1, 3000))
    onsets = np.sort(rng.choice(np.arange(5, 2990), size=200, replace=False))
    x[0, onsets + 1] += 2.0  # evoked response one sample after each event

    def evoked(ts, ev):
        idx = np.rint(np.asarray(ev) / tr).astype(int)
        idx = idx[idx + 1 < ts.shape[1]]
        return float(np.mean(ts[0, idx + 1]))

    observed = evoked(x, list(onsets * tr))
    res = nulls.component_bootstrap_se(evoked, x, list(onsets * tr), n=300, seed=0,
                                       tr=tr, block_len=50)
    # centred on the observed statistic (a few events are cut at block
    # junctions), with the SE of a mean over 200 unit-variance responses
    assert np.mean(res.samples) == pytest.approx(observed, abs=0.05)
    assert res.se == pytest.approx(1.0 / np.sqrt(200), rel=0.3)
