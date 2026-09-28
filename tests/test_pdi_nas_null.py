"""Null / ground-truth tests for the optional PDI and NAS surrogate calibration
(remediation spec D7) and for undefined-input handling (NaN, never 0.0)."""

import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm

N_SURR = 19
SEEDS = range(6)
FMRI_NAS = dict(
    tr=2.0,
    tau=0.2,
    bands=((0.01, 0.1),),
    band_weights=(1.0,),
    window_len=30,
    step_len=15,
)
EEG_NAS = dict(
    tr=0.02,
    tau=0.2,
    bands=((1.0, 20.0),),
    band_weights=(1.0,),
    window_len=120,
    step_len=60,
)


def _ar1(n, t, phi, seed, burn=100):
    rng = np.random.RandomState(seed)
    x = np.zeros((n, t + burn))
    for k in range(1, t + burn):
        x[:, k] = phi * x[:, k - 1] + rng.randn(n)
    return x[:, burn:]


def _switching(n, t, seed, n_states=4, dwell=12, noise=0.3):
    """Metastable switching among distinct spatial patterns (non-Gaussian)."""
    rng = np.random.RandomState(seed)
    patterns = 1.5 * rng.randn(n_states, n)
    x = np.zeros((n, t))
    state = 0
    for k in range(t):
        if rng.rand() < 1.0 / dwell:
            state = rng.randint(n_states)
        x[:, k] = patterns[state] + noise * rng.randn(n)
    return x


def _broadcast(n, t, seed):
    """A 4-node workspace shares a stochastic ~5 Hz driver (fs=50 Hz) that is
    broadcast with a 2-sample lag to all other nodes."""
    rng = np.random.RandomState(seed)
    drv = np.zeros(t + 50)
    r, w = 0.95, 2.0 * np.pi * 5.0 / 50.0
    for k in range(2, t + 50):
        drv[k] = 2.0 * r * np.cos(w) * drv[k - 1] - r * r * drv[k - 2] + rng.randn()
    drv = drv[50:] / drv[50:].std()
    x = 0.5 * rng.randn(n, t)
    x[:4] += drv
    x[4:] += 0.6 * np.roll(drv, 2)
    return x


def _pdi(task, base, seed):
    return mm.compute_PDI(
        task,
        baseline_ts=[base],
        return_details=True,
        null_surrogates=N_SURR,
        null_seed=seed,
    )


# ---------------------------------------------------------------- PDI


def test_pdi_calibration_removes_positive_null_bias():
    # Task and baseline drawn from the same AR(0.5) process.
    runs = [
        _pdi(_ar1(8, 200, 0.5, 2 * s), _ar1(8, 200, 0.5, 2 * s + 1), s) for s in SEEDS
    ]
    raw = np.asarray([d["raw"] for d in runs])
    excess = np.asarray([d["PDI_excess"] for d in runs])
    # Uncalibrated PDI is >= 0 by construction and > 0 almost always under the null.
    assert np.all(raw >= 0.0) and np.mean(raw > 0.0) >= 0.8
    assert abs(float(np.mean(excess))) < 1e-3
    assert float(np.mean([d["PDI_calibrated"] for d in runs])) < 2e-3


def test_pdi_calibration_removes_autocorrelation_confound():
    # A smoother (AR 0.9) task against a white baseline scores high on raw PDI
    # purely through temporal predictability; spectrum-preserving surrogates
    # remove it.
    runs = [
        _pdi(_ar1(8, 200, 0.9, s), np.random.RandomState(100 + s).randn(8, 200), s)
        for s in SEEDS
    ]
    assert float(np.mean([d["raw"] for d in runs])) > 0.05
    assert abs(float(np.mean([d["PDI_excess"] for d in runs]))) < 5e-3
    assert abs(float(np.mean([d["PDI_z"] for d in runs]))) < 1.5


def test_pdi_calibration_is_positive_for_differentiated_dynamics():
    null_runs = [
        _pdi(_ar1(8, 200, 0.5, 2 * s), _ar1(8, 200, 0.5, 2 * s + 1), s) for s in SEEDS
    ]
    runs = [_pdi(_switching(8, 200, s), _ar1(8, 200, 0.5, 50 + s), s) for s in SEEDS]
    excess = np.asarray([d["PDI_excess"] for d in runs])
    null_excess = float(np.mean([d["PDI_excess"] for d in null_runs]))
    assert np.all(excess > 0.0)
    assert float(np.mean(excess)) > 5.0 * max(abs(null_excess), 1e-3)
    assert float(np.mean([d["PDI_z"] for d in runs])) > 2.0
    for d in runs:
        assert d["PDI_null_calibrated"] is True
        assert d["PDI_null_n"] == N_SURR
        assert d["value"] == pytest.approx(d["PDI_calibrated"])
        assert d["PDI_calibrated"] == pytest.approx(max(d["PDI_excess"], 0.0))


def test_pdi_scalar_return_is_calibrated_value_when_requested():
    task, base = _switching(8, 200, 0), _ar1(8, 200, 0.5, 50)
    details = _pdi(task, base, 0)
    scalar = mm.compute_PDI(
        task, baseline_ts=[base], null_surrogates=N_SURR, null_seed=0
    )
    assert scalar == pytest.approx(details["PDI_calibrated"])
    legacy = mm.compute_PDI(task, baseline_ts=[base])
    assert legacy == pytest.approx(details["raw"])


@pytest.mark.parametrize(
    "task, baseline, reason",
    [
        (
            np.full((4, 50), np.nan),
            [np.random.RandomState(0).randn(4, 50)],
            "no_finite_task_values",
        ),
        (
            np.random.RandomState(0).randn(4, 50),
            [np.full((4, 50), np.nan)],
            "no_finite_baseline_values",
        ),
        (
            np.random.RandomState(0).randn(1, 50),
            [np.random.RandomState(1).randn(1, 50)],
            "insufficient_regions",
        ),
        (
            np.random.RandomState(0).randn(4, 3),
            [np.random.RandomState(1).randn(4, 3)],
            "insufficient_timepoints",
        ),
        (np.random.RandomState(0).randn(4, 50), [], "empty_baseline"),
    ],
)
def test_pdi_undefined_inputs_return_nan_with_reason(task, baseline, reason):
    assert np.isnan(mm.compute_PDI(task, baseline_ts=baseline))
    details = mm.compute_PDI(task, baseline_ts=baseline, return_details=True)
    assert details["defined"] is False
    assert details["undefined_reason"] == reason
    assert np.isnan(details["value"])


# ---------------------------------------------------------------- NAS


def test_nas_calibration_removes_selection_bias_on_noise():
    runs = [
        mm.compute_NAS(
            np.random.RandomState(s).randn(20, 200),
            return_details=True,
            null_surrogates=N_SURR,
            null_seed=s,
            **FMRI_NAS
        )
        for s in SEEDS
    ]
    # Selecting the broadcast set / workspace from the scored data makes
    # uncalibrated NAS clearly positive on independent white noise.
    assert float(np.mean([d["raw"] for d in runs])) > 0.1
    assert abs(float(np.mean([d["NAS_excess"] for d in runs]))) < 0.01
    assert abs(float(np.nanmean([d["NAS_z"] for d in runs]))) < 1.0


def test_nas_calibration_is_positive_for_broadcast_dynamics():
    noise = [
        mm.compute_NAS(
            np.random.RandomState(300 + s).randn(20, 360),
            return_details=True,
            null_surrogates=N_SURR,
            null_seed=s,
            **EEG_NAS
        )
        for s in SEEDS
    ]
    runs = [
        mm.compute_NAS(
            _broadcast(20, 360, s),
            return_details=True,
            null_surrogates=N_SURR,
            null_seed=s,
            **EEG_NAS
        )
        for s in SEEDS
    ]
    excess = np.asarray([d["NAS_excess"] for d in runs])
    noise_excess = float(np.mean([d["NAS_excess"] for d in noise]))
    assert np.all(excess > 0.05)
    assert float(np.mean(excess)) > 3.0 * max(abs(noise_excess), 0.01)
    assert float(np.mean([d["NAS_calibrated"] for d in runs])) > float(
        np.mean([d["NAS_calibrated"] for d in noise])
    )
    for d in runs:
        assert d["value"] == pytest.approx(d["NAS_calibrated"])
        assert 0.0 <= d["value"] <= 1.0


def test_nas_undefined_shape_returns_nan():
    tiny = np.random.RandomState(0).randn(5, 3)
    assert np.isnan(mm.compute_NAS(tiny, **FMRI_NAS))
    details = mm.compute_NAS(tiny, return_details=True, **FMRI_NAS)
    assert details["defined"] is False
    assert details["undefined_reason"] == "insufficient_shape"
    assert np.isnan(mm.compute_NAS(np.random.RandomState(0).randn(1, 100), **FMRI_NAS))


def test_nas_uncalibrated_value_unchanged_by_new_options():
    x = _broadcast(20, 360, 0)
    plain = mm.compute_NAS(x, **EEG_NAS)
    details = mm.compute_NAS(x, return_details=True, **EEG_NAS)
    assert details["value"] == pytest.approx(plain)
    assert details["NAS_null_calibrated"] is False
    assert np.isnan(details["NAS_z"])


# ---------------------------------------------------------------- surrogates


def test_circular_shift_surrogate_preserves_each_node_exactly():
    x = _ar1(5, 120, 0.8, 0)
    s = mm._surrogate_timeseries(
        x, "circular_shift", np.random.RandomState(0), min_shift=12
    )
    np.testing.assert_array_equal(s[0], x[0])
    for i in range(1, 5):
        shifts = [k for k in range(120) if np.array_equal(np.roll(x[i], k), s[i])]
        assert shifts and 12 <= shifts[0] <= 108


def test_phase_randomized_surrogate_preserves_marginals_spectra_and_nan_positions():
    x = _ar1(4, 256, 0.9, 1)
    x[:, 0] += 0.5 * x[:, 1]
    x[2, 10] = np.nan
    s = mm._surrogate_timeseries(x, "phase_randomize", np.random.RandomState(3))
    assert np.isnan(s[2, 10]) and np.isnan(s).sum() == 1
    for i in range(4):
        np.testing.assert_allclose(
            np.sort(s[i][np.isfinite(s[i])]), np.sort(x[i][np.isfinite(x[i])])
        )
    # Lag-1 autocorrelation is kept (IAAFT); the order is destroyed.
    for i in (0, 1, 3):
        ac_x = np.corrcoef(x[i, :-1], x[i, 1:])[0, 1]
        ac_s = np.corrcoef(s[i, :-1], s[i, 1:])[0, 1]
        assert abs(ac_s - ac_x) < 0.05
        assert not np.allclose(s[i], x[i])


def test_requested_calibration_that_cannot_run_is_nan_not_raw():
    task, base = _ar1(4, 60, 0.5, 0), _ar1(4, 60, 0.5, 1)
    pdi = mm.compute_PDI(
        task,
        baseline_ts=[base],
        return_details=True,
        null_surrogates=3,
        null_method="circular_shift",
        null_min_shift=50,
    )
    assert np.isfinite(pdi["raw"])
    assert pdi["PDI_null_undefined_reason"].startswith("surrogates_unavailable")
    assert np.isnan(pdi["value"])
    nas = mm.compute_NAS(
        np.random.RandomState(0).randn(20, 200),
        return_details=True,
        null_surrogates=3,
        null_min_shift=150,
        **FMRI_NAS,
    )
    assert np.isfinite(nas["raw"])
    assert nas["NAS_null_undefined_reason"].startswith("surrogates_unavailable")
    assert np.isnan(nas["value"])
