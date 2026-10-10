"""Known-answer and null tests for the SRPI estimator (compute_SRPI)."""
import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm

FMRI_KW = dict(
    modality="fmri",
    pre_window_sec=2.0,
    response_lag_sec=4.0,
    response_window_sec=6.0,
    covariance_ridge=1e-3,
    component_weights=(0.35, 0.25, 0.20, 0.20),
    min_events_per_class=3,
    sample_reliability_tau=4.0,
    return_details=True,
)


def test_pre_event_window_lies_strictly_before_onset():
    ramp = np.tile(np.arange(200, dtype=float), (3, 1))
    pre, delta, used = mm._event_locked_state_deltas(
        ramp, np.array([50, 120]), lag_samples=4, pre_samples=2, post_samples=3
    )
    # Pre window [e-2, e): mean index e-1.5 (not e+lag-1.5, after onset).
    assert pre[:, 0].tolist() == pytest.approx([48.5, 118.5])
    # Response window [e+4, e+7): mean index e+5.
    assert (pre[:, 0] + delta[:, 0]).tolist() == pytest.approx([55.0, 125.0])
    assert used.tolist() == [50, 120]


def test_pre_event_state_unaffected_by_post_onset_activity():
    rng = np.random.RandomState(0)
    ts = rng.randn(5, 400)
    events = np.arange(20, 380, 20)
    pre_a, _, _ = mm._event_locked_state_deltas(ts, events, 4, 2, 6)
    ts_b = ts.copy()
    for e in events:
        ts_b[:, e:e + 4] += 100.0  # onset and the lag period only
    pre_b, _, _ = mm._event_locked_state_deltas(ts_b, events, 4, 2, 6)
    np.testing.assert_allclose(pre_a, pre_b)


def _srpi_task(seed, separable, n_regions=200, n_events=40, tr=1.0, amp=1.0):
    rng = np.random.RandomState(seed)
    n_tp = n_events * 16 + 40
    ts = rng.randn(n_regions, n_tp)
    on = 20 + 16 * np.arange(n_events) + rng.randint(0, 3, n_events)
    half = n_events // 2
    lab = rng.permutation(np.r_[np.ones(half), np.zeros(half)]).astype(bool)
    p_self, p_non = rng.randn(n_regions), rng.randn(n_regions)
    for o, is_self in zip(on, lab):
        pat = (p_self if is_self else p_non) if separable else 0.5 * (p_self + p_non)
        ts[:, o + 4:o + 10] += amp * pat[:, None]
    return ts, tr, (on[lab] * tr).tolist(), (on[~lab] * tr).tolist()


def test_separability_detects_distinct_self_patterns():
    for seed in range(3):
        ts, tr, s, n = _srpi_task(seed, separable=True)
        d = mm.compute_SRPI(ts, tr=tr, self_onsets=s, nonself_onsets=n, **FMRI_KW)
        assert d["separability_cv_auc"] > 0.95
        assert d["components_raw"]["representational_separability"] > 0.9


def test_separability_does_not_saturate_with_random_labels():
    seps, aucs, vals = [], [], []
    for seed in range(12):
        # Same evoked response for every event; self/non-self labels random.
        ts, tr, s, n = _srpi_task(seed, separable=False)
        d = mm.compute_SRPI(ts, tr=tr, self_onsets=s, nonself_onsets=n, **FMRI_KW)
        seps.append(d["components_raw"]["representational_separability"])
        aucs.append(d["separability_cv_auc"])
        vals.append(d["value"])
    # Guards against 1 - exp(-d^2/2) = 1.0 in every null run (p >> n).
    assert np.median(seps) < 0.1
    assert np.mean(seps) < 0.25
    assert abs(np.mean(aucs) - 0.5) < 0.1
    assert np.mean(vals) < 0.05


def test_srpi_near_zero_on_noise_with_random_labels():
    vals = []
    for seed in range(10):
        rng = np.random.RandomState(50 + seed)
        ts = rng.randn(64, 600)
        on = rng.permutation(np.arange(10, 580, 12).astype(float))
        d = mm.compute_SRPI(ts, tr=1.0, self_onsets=on[:20].tolist(),
                            nonself_onsets=on[20:40].tolist(), **FMRI_KW)
        assert d["undefined_reason"] is None
        vals.append(d["value"])
    assert np.mean(vals) < 0.05
    assert np.median(vals) == 0.0


def test_internal_state_coupling_uses_absolute_correlation():
    # Self responses scale with the pre-event state along a latent axis; the
    # sign of that coupling is not identifiable (SVD axis sign), so flipping
    # it must leave the internal-state term unchanged.
    rng = np.random.RandomState(7)
    n_regions, tr = 12, 1.0
    base = 0.3 * rng.randn(n_regions, 900)
    pattern = rng.randn(n_regions)
    axis = rng.randn(n_regions)
    axis -= axis.dot(pattern) / pattern.dot(pattern) * pattern  # orthogonal
    self_on = np.arange(20, 440, 20)
    non_on = np.arange(450, 880, 20)
    latent = np.clip(rng.randn(self_on.size), -2.0, 2.0)

    def build(sign):
        ts = base.copy()
        for e, z in zip(self_on, latent):
            ts[:, e - 2:e] += 0.5 * z * axis[:, None]
            ts[:, e + 4:e + 10] += (4.0 + sign * 1.5 * z) * pattern[:, None]
        for e in non_on:
            ts[:, e + 4:e + 10] += 4.0 * pattern[:, None]
        return mm.compute_SRPI(ts, tr=tr, self_onsets=self_on.tolist(),
                               nonself_onsets=non_on.tolist(), **FMRI_KW)

    pos, neg = build(+1.0), build(-1.0)
    i_pos = pos["components_raw"]["internal_state_coupling"]
    i_neg = neg["components_raw"]["internal_state_coupling"]
    assert i_pos > 0.3
    assert i_neg == pytest.approx(i_pos, abs=0.1)


def test_min_events_per_class_below_three_is_rejected():
    # Cross-validated separability needs 2 training events per class, so with
    # min_events_per_class=2 SRPI could never be defined; the parameter is now
    # validated (>= 3) instead of silently returning separability_undefined.
    ts = np.random.RandomState(1).randn(10, 300)
    kw = dict(FMRI_KW, min_events_per_class=2)
    with pytest.raises(ValueError, match="min_events_per_class must be >= 3"):
        mm.compute_SRPI(
            ts, tr=1.0, self_onsets=[20.0, 60.0], nonself_onsets=[100.0, 140.0], **kw
        )
    with pytest.raises(ValueError, match="min_events_per_class must be >= 3"):
        mm.compute_SRPI(ts, tr=1.0, mode="agency", **kw)


def test_three_events_per_class_are_enough_for_cv_separability():
    ts = np.random.RandomState(1).randn(10, 300)
    d = mm.compute_SRPI(
        ts,
        tr=1.0,
        self_onsets=[20.0, 60.0, 180.0],
        nonself_onsets=[100.0, 140.0, 220.0],
        **FMRI_KW,
    )
    assert d["undefined_reason"] is None
    assert np.isfinite(d["separability_cv_auc"])
