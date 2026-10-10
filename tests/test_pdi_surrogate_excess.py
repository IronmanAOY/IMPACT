"""Known-answer, null and property tests for compute_PDI(mode='surrogate_excess')."""

import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm

EXCESS = dict(mode="surrogate_excess", return_details=True)


def _linear_gaussian(seed, n_nodes=12, n_time=1500):
    """Stable VAR(1) with cross-coupling: a linear Gaussian process."""
    rng = np.random.default_rng(seed)
    a = 0.85 * np.eye(n_nodes) + 0.05 * rng.standard_normal((n_nodes, n_nodes))
    a *= 0.95 / max(abs(np.linalg.eigvals(a)))
    x = np.zeros((n_nodes, n_time))
    for t in range(1, n_time):
        x[:, t] = a @ x[:, t - 1] + rng.standard_normal(n_nodes)
    return x


def _pattern_switching(
    seed, n_patterns, n_nodes=16, n_time=2000, dwell=40.0, noise=0.5
):
    """Multistable dynamics: dwell in one of K random +-1 patterns, then jump."""
    rng = np.random.default_rng(seed)
    pats = rng.choice([-1.0, 1.0], size=(n_patterns, n_nodes))
    drive = np.zeros((n_nodes, n_time))
    t = 0
    while t < n_time:
        d = int(rng.exponential(dwell)) + 5
        drive[:, t : t + d] = pats[rng.integers(n_patterns)][:, None]
        t += d
    x = np.zeros_like(drive)
    for t in range(1, n_time):
        x[:, t] = 0.7 * x[:, t - 1] + 0.3 * drive[:, t]
    return x + noise * rng.standard_normal((n_nodes, n_time))


def test_lz76_known_answers():
    assert mm.lz76_complexity([int(c) for c in "0001101001000101"]) == 6
    assert mm.lz76_complexity([0, 0, 0, 0]) == 2
    assert mm.lz76_complexity([]) == 0
    assert mm.lz76_complexity(["a"]) == 1
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 20000)
    norm = mm.lz76_complexity(bits) * np.log2(bits.size) / bits.size
    assert 0.8 < norm < 1.1
    assert mm.lz76_complexity(np.tile([0, 1], 5000)) < 10


def test_linear_gaussian_data_is_about_zero():
    zs = []
    for seed in range(6):
        d = mm.compute_PDI(_linear_gaussian(seed), null_seed=seed, **EXCESS)
        assert d["defined"] and d["mode"] == "surrogate_excess"
        zs.append(d["PDI_z"])
        assert all(abs(z) < 3.5 for z in d["components_z"].values())
    print("PDI surrogate excess, linear Gaussian: z", np.round(zs, 2))
    assert np.all(np.abs(zs) < 3.0)
    assert abs(np.mean(zs)) < 1.5


def test_independent_noise_is_about_zero_with_iaaft_surrogates():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((10, 1200)) ** 3  # non-Gaussian marginals
    d = mm.compute_PDI(x, excess_surrogate="iaaft", null_seed=1, **EXCESS)
    assert d["surrogate_method"] == "iaaft" and d["PDI_null_method"] == "iaaft"
    assert abs(d["PDI_z"]) < 3.0


def test_repertoire_entropy_of_pattern_switching_falls_below_linear_gaussian():
    # Maximum-entropy property of the Gaussian null: for the same auto- and
    # cross-spectra a linear Gaussian process occupies more global states, so
    # the repertoire entropy of multistable pattern switching is *below* its
    # spectrum-preserving surrogates (signed excess < 0).
    for seed in range(3):
        for k in (6, 12):
            d = mm.compute_PDI(_pattern_switching(seed, k), null_seed=seed, **EXCESS)
            assert d["components_z"]["repertoire_entropy"] < -3.0
            assert d["components"]["repertoire_entropy"] < 0.0


def test_negative_result_surrogate_excess_is_not_a_differentiation_score():
    """
    Documented negative result.

    The v1 design expected multistable pattern switching to exceed
    spectrum-matched linear Gaussian surrogates. It cannot: for fixed auto-
    and cross-spectra a linear Gaussian process maximises entropy and entropy
    rate, so entropy-type repertoire statistics of structured dynamics fall
    *below* such a null, and the aggregate excess has either sign across
    seeds. mode='surrogate_excess' is therefore not a positive "more
    differentiated than chance" score. mode='repertoire' (count of recurring,
    distinguishable states; tests/test_pdi_repertoire.py) replaces it and
    detects the same multistability.
    """
    zs = [
        mm.compute_PDI(_pattern_switching(seed, k), null_seed=seed, **EXCESS)["PDI_z"]
        for seed in range(3)
        for k in (6, 12)
    ]
    print("PDI surrogate excess, pattern switching: z", np.round(zs, 2))
    # Not reliably above the Gaussian null, and clearly below it for some seeds.
    assert not all(z > 1.645 for z in zs)
    assert min(zs) < -1.645
    # The construct-grounded replacement on the same data: >= 4 recurring,
    # separated states and more than 1.5 bits above circular-shift surrogates.
    for seed in range(3):
        d = mm.compute_PDI(
            _pattern_switching(seed, 6),
            mode="repertoire",
            null_seed=seed,
            null_surrogates=9,
            return_details=True,
        )
        assert d["n_states"] >= 4 and d["value"] > 1.5


def test_continuous_effective_dimensionality_is_preserved_by_fourier_surrogates():
    # Why d_eff is computed on binarised nodes: the covariance (hence the
    # participation ratio) of the continuous data is exactly preserved by
    # multivariate Fourier surrogates, so its excess would be identically 0.
    x = _pattern_switching(0, 6)

    def pr(a):
        z = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
        ev = np.linalg.svd(z, compute_uv=False) ** 2
        return ev.sum() ** 2 / (ev**2).sum()

    surr = mm._phase_randomized_surrogate(x, np.random.RandomState(0))
    assert pr(surr) == pytest.approx(pr(x), rel=1e-9)
    feats_obs, _ = mm._pdi_global_state_features(x, 5)
    feats_sur, _ = mm._pdi_global_state_features(surr, 5)
    assert feats_obs[2] != pytest.approx(feats_sur[2], rel=1e-6)


def test_signed_value_is_the_weighted_mean_of_component_excesses():
    x = _pattern_switching(1, 12)
    d = mm.compute_PDI(x, null_seed=2, excess_weights=(2.0, 1.0, 1.0), **EXCESS)
    w = np.asarray([d["weights"][k] for k in mm.PDI_EXCESS_COMPONENTS])
    ex = np.asarray([d["components"][k] for k in mm.PDI_EXCESS_COMPONENTS])
    assert w.sum() == pytest.approx(1.0)
    assert d["value"] == pytest.approx(float(np.dot(w, ex)))
    assert d["value"] == pytest.approx(d["raw"] - d["PDI_null_mean"])
    # No clipping anywhere: negative excess stays negative.
    assert d["value"] < 0.0
    assert d["PDI_calibrated"] == pytest.approx(d["PDI_excess"])
    assert (
        mm.compute_PDI(
            x, null_seed=2, excess_weights=(2.0, 1.0, 1.0), mode="surrogate_excess"
        )
        == d["value"]
    )


def test_surrogate_excess_ignores_baseline_and_legacy_default_is_unchanged():
    x = _linear_gaussian(0, n_time=300)
    base = _linear_gaussian(1, n_time=300)
    a = mm.compute_PDI(x, baseline_ts=base, null_seed=0, **EXCESS)
    b = mm.compute_PDI(x, null_seed=0, **EXCESS)
    assert a["baseline_ignored"] is True and b["baseline_ignored"] is False
    assert a["value"] == b["value"]
    assert a["n_surrogates_requested"] == mm.PDI_EXCESS_DEFAULT_SURROGATES
    legacy = mm.compute_PDI(x, baseline_ts=base)
    assert legacy == mm.compute_PDI(x, baseline_ts=base, mode="legacy")
    assert mm.compute_PDI(x, baseline_ts=base, return_details=True)["mode"] == "legacy"


def test_surrogate_excess_undefined_and_validation():
    rng = np.random.default_rng(0)
    one = mm.compute_PDI(rng.standard_normal((1, 500)), **EXCESS)
    assert one["undefined_reason"] == "insufficient_regions" and np.isnan(one["value"])
    short = mm.compute_PDI(rng.standard_normal((4, 10)), **EXCESS)
    assert short["undefined_reason"] == "insufficient_timepoints"
    bad = rng.standard_normal((4, 300))
    bad[0, 5] = np.nan
    assert mm.compute_PDI(bad, **EXCESS)["undefined_reason"] == "non_finite_timeseries"
    flat = np.zeros((4, 300))
    assert mm.compute_PDI(flat, **EXCESS)["undefined_reason"] == "no_variance"
    x = rng.standard_normal((4, 300))
    with pytest.raises(ValueError, match="mode must be one of"):
        mm.compute_PDI(x, mode="excess")
    with pytest.raises(ValueError, match="excess_weights"):
        mm.compute_PDI(x, mode="surrogate_excess", excess_weights=(0, 0, 0))
    with pytest.raises(ValueError, match="excess_surrogate"):
        mm.compute_PDI(x, mode="surrogate_excess", excess_surrogate="shuffle")
    with pytest.raises(ValueError, match="null_surrogates >= 2"):
        mm.compute_PDI(x, mode="surrogate_excess", null_surrogates=1)
