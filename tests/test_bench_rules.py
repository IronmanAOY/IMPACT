"""Truth tables of the rival attribution rules (MPC-Bench rule audit) on
hand-made component vectors, including undefined (NaN) components, and
known-answer tests for the LZ76 complexity marker."""

import numpy as np
import pytest

from impact_pipeline.bench import lz
from impact_pipeline.bench import rules as R

A, N, U = R.MPC_CONSISTENT, R.EXCLUDED, R.UNDETERMINED
nan = np.nan

# Rows: one deficit / one missing / all missing / all high / measured zero /
# all low / legacy compensation example.
C = np.array(
    [
        [0.9, 0.9, 0.9, 0.9, 0.1],
        [0.9, 0.9, 0.9, 0.9, nan],
        [nan, nan, nan, nan, nan],
        [0.9, 0.9, 0.9, 0.9, 0.9],
        [0.0, 2.0, 2.0, 2.0, 2.0],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.01, 10.0, 10.0, 10.0, 10.0],
    ]
)

TRUTH = {
    "union": [A, A, N, A, A, N, A],
    "count_1": [A, A, N, A, A, N, A],
    "count_4": [A, A, N, A, A, N, A],
    "count_5": [N, N, N, A, N, N, N],
    "arithmetic_mean": [A, A, U, A, A, N, A],
    "geometric_mean_uncapped": [A, N, N, A, N, N, A],
    "geometric_mean_capped": [A, U, U, A, N, N, N],
    "weakest_link": [N, U, U, A, N, N, N],
    "dcm_naive_bayes": [A, A, N, A, A, N, A],
    "chalmers_product": [N, N, N, A, N, N, N],
    "IIM_only": [A, A, U, A, A, N, A],
    "NAS_only": [A, A, U, A, A, N, A],
}


def test_rule_truth_tables():
    df = R.apply_rules(C)
    for rule, expected in TRUTH.items():
        assert list(df[rule]) == expected, rule


def test_rule_scores_known_values():
    gm = R.rule_geometric_mean_uncapped(C).score
    assert gm[0] == pytest.approx((0.9**4 * 0.1) ** 0.2)
    # Legacy compensation: CI(0.01, 10, 10, 10, 10) = 100 ** (1/5) ~ 2.51.
    assert gm[6] == pytest.approx(100.0**0.2)
    assert gm[1] == 0.0  # missing -> 0 in the legacy rule
    capped = R.rule_geometric_mean_capped(C).score
    assert capped[6] == pytest.approx(0.01**0.2)
    assert np.isnan(capped[1])
    assert R.rule_arithmetic_mean(C).score[1] == pytest.approx(0.9)
    assert R.rule_weakest_link(C).score[0] == pytest.approx(0.1)
    # DCM-like naive Bayes: prior odds 1/5, LR+ = 0.8/0.2 = 4, LR- = 0.25.
    post = R.rule_dcm_naive_bayes(C).score
    odds0 = 0.2 * 4**4 * 0.25
    assert post[0] == pytest.approx(odds0 / (1 + odds0))
    odds1 = 0.2 * 4**4  # missing indicator skipped
    assert post[1] == pytest.approx(odds1 / (1 + odds1))
    assert post[2] == pytest.approx(1.0 / 6.0)  # prior only
    chal = R.rule_chalmers_product(C).score
    assert chal[0] == pytest.approx(0.9**4 * 0.1)
    assert chal[1] == pytest.approx(0.9**4 * 0.5)  # missing -> prior credence
    assert chal[2] == pytest.approx(0.5**5)


def test_count_k_is_monotone_in_k():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(500, 5))
    X[rng.random(X.shape) < 0.2] = nan
    prev = None
    for k in range(1, 6):
        attr = R.rule_count_k(X, k).decision == A
        if prev is not None:
            assert np.all(attr <= prev)
        prev = attr
    assert np.array_equal(R.rule_count_k(X, 1).decision, R.rule_union(X).decision)
    with pytest.raises(ValueError):
        R.rule_count_k(X, 0)


def test_power_mean_family_limits_and_order():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.05, 1.0, size=(200, 5))
    lo = R.power_mean(X, -np.inf)
    g = R.power_mean(X, 0.0)
    h = R.power_mean(X, -1.0)
    ar = R.power_mean(X, 1.0)
    hi = R.power_mean(X, np.inf)
    assert np.allclose(lo, X.min(axis=1))
    assert np.allclose(hi, X.max(axis=1))
    assert np.allclose(ar, X.mean(axis=1))
    assert np.allclose(g, np.exp(np.log(X).mean(axis=1)))
    assert np.all(lo <= h + 1e-12) and np.all(h <= g + 1e-12)
    assert np.all(g <= ar + 1e-12) and np.all(ar <= hi + 1e-12)
    # Cap and zero handling.
    assert R.power_mean([[2.0] * 5], 0.0, cap=1.0)[0] == pytest.approx(1.0)
    assert R.power_mean([[0.0, 1, 1, 1, 1]], -1.0)[0] == 0.0
    assert np.isnan(R.power_mean([[nan, 1, 1, 1, 1]], 1.0)[0])
    # A zero weight ignores that component (and its NaN).
    w = [1, 1, 1, 1, 0]
    assert R.power_mean([[0.5, 0.5, 0.5, 0.5, nan]], 0.0, weights=w)[
        0
    ] == pytest.approx(0.5)


def test_impact_rule_kleene_truth_table():
    Z = np.array(
        [
            [3, 3, 3, 3, 3],  # all PRESENT
            [3, 3, 3, 3, 0],  # one ABSENT (TOST) -> veto
            [3, 3, 3, 3, nan],  # one undefined
            [3, 3, 3, 3, 1.3],  # inconclusive (neither present nor absent)
            [0, nan, 3, 3, 3],  # ABSENT beats undefined
        ],
        dtype=float,
    )
    out = R.rule_impact(Z)
    assert list(out.decision) == [A, N, U, U, N]
    assert list(out.score) == [5, 4, 4, 4, 3]
    # Restricting the necessity set drops the deficit.
    out4 = R.rule_impact(Z, necessity_set=("RAM", "PDI", "NAS", "IIM"))
    assert list(out4.decision) == [A, A, A, A, N]
    # An SE widens both tests: PRESENT needs z - 1.645 se > 1.645.
    st = R.component_status_z(
        np.array([2.0, 2.0, 0.5, 0.5]), se=np.array([0.0, 0.5, 0.0, 0.5])
    )
    assert list(st) == [R.PRESENT, R.UNDEFINED, R.ABSENT, R.UNDEFINED]


def test_component_status_parameters_match_evidence_layer_checks():
    # Overlapping PRESENT / ABSENT regions, non-finite thresholds and a bad
    # alpha are refused (as in evidence.component_status).
    with pytest.raises(ValueError):
        R.component_status_z(1.0, delta_equiv=2.0, z_present=1.645)
    with pytest.raises(ValueError):
        R.component_status_z(1.0, z_present=np.inf)
    with pytest.raises(ValueError):
        R.component_status_z(1.0, alpha=0.0)
    with pytest.raises(ValueError):
        R.component_status_z(1.0, delta_equiv=-0.1)
    # A negative SE is invalid evidence: UNDEFINED, never ABSENT or PRESENT.
    st = R.component_status_z(np.array([0.0, 5.0]), se=np.array([-0.5, -0.5]))
    assert list(st) == [R.UNDEFINED, R.UNDEFINED]


def test_impact_rule_never_attributes_from_missing_components():
    rng = np.random.default_rng(2)
    Z = rng.normal(2.0, 2.0, size=(20000, 5))
    base = R.rule_impact(Z).decision
    Zm = Z.copy()
    Zm[rng.random(Z.shape) < 0.3] = nan
    miss = R.rule_impact(Zm).decision
    # Missingness never creates MPC_CONSISTENT, never flips a determinate verdict.
    assert not np.any((miss == A) & (base != A))
    det = miss != U
    assert np.all(miss[det] == base[det])


def test_status_matches_evidence_layer_when_available():
    ev = pytest.importorskip("impact_pipeline.evidence")
    rng = np.random.default_rng(3)
    for z, se in zip(rng.normal(1.0, 2.0, 200), rng.uniform(0, 1, 200)):
        e = ev.ComponentEvidence(
            principle="RAM", estimate=float(z), null_mean=0.0, null_sd=1.0, se=float(se)
        )
        status = ev.component_status(e)[0]
        assert getattr(status, "value", status) == R.component_status_z(z, se)


def test_logistic_classifier_cross_validated():
    rng = np.random.default_rng(4)
    n = 200
    y = rng.integers(0, 2, n)
    X = rng.normal(0.3, 0.1, size=(n, 5)) + 0.5 * y[:, None]
    X[rng.random(X.shape) < 0.1] = nan
    out = R.rule_logistic(X, y, cv=5, seed=0)
    acc = np.mean((out.decision == A) == (y == 1))
    assert acc > 0.9
    assert np.all((out.score >= 0) & (out.score <= 1))
    held = R.rule_logistic(X, y, C_test=[[0.9] * 5, [0.2] * 5])
    assert list(held.decision) == [A, N]
    df = R.apply_rules(X, labels=y, Z=rng.normal(size=(n, 5)))
    assert {"logistic", "impact"} <= set(df.columns)


def test_calibrated_likelihood_ratios_and_markers():
    P = np.array(
        [[1, 1, 1, 1, 1], [1, 0, 1, nan, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, nan]]
    )
    sens, spec = R.calibrate_likelihood_ratios(P, [1, 1, 0, 0], alpha=0.0)
    assert sens[0] == pytest.approx(1.0) and spec[0] == pytest.approx(1.0)
    assert sens[1] == pytest.approx(0.5) and spec[1] == pytest.approx(0.5)
    assert sens[3] == pytest.approx(1.0)  # NaN skipped
    out = R.rule_single_marker([0.2, nan, 0.9], threshold=0.5, name="LZc")
    assert list(out.decision) == [N, U, A]
    df = R.apply_rules(
        C, markers={"LZc": np.linspace(0, 1, len(C))}, marker_thresholds={"LZc": 0.5}
    )
    assert "LZc" in df.columns
    with pytest.raises(ValueError):
        R.apply_rules(C, markers={"PhiR": np.ones(len(C))})


def test_rule_properties_table_is_complete():
    keys = {"compensatory", "missingness", "veto", "abstains", "prior_dependent"}
    for name, props in R.RULE_PROPERTIES.items():
        assert set(props) == keys, name
    assert R.RULE_PROPERTIES["impact"]["missingness"] == "kleene"
    assert R.RULE_PROPERTIES["geometric_mean_uncapped"]["missingness"] == "zero"
    with pytest.raises(ValueError):
        R.as_component_matrix(np.ones((3, 4)))


# --------------------------------------------------------------------------
# LZ76 complexity
# --------------------------------------------------------------------------


def test_lz76_known_parsing():
    # Lempel & Ziv (1976) / Kaspar & Schuster (1987): 0.001.10.100.1000.101
    seq = [int(ch) for ch in "0001101001000101"]
    assert lz.lz76_complexity(seq) == 6
    assert lz.lz76_complexity([0] * 64) == 2
    assert lz.lz76_complexity([0, 1] * 32) == 3
    assert lz.lz76_complexity([1]) == 1
    assert lz.lz76_complexity([]) == 0
    assert lz.lz76_complexity(list("abab")) == lz.lz76_complexity([0, 1, 0, 1])


def test_lzc_normalisation_on_random_and_regular_data():
    rng = np.random.default_rng(5)
    noise = rng.standard_normal((8, 2000))
    val = lz.lzc(noise)
    assert 0.9 < val < 1.1
    assert 0.9 < lz.lzc(noise, normalize="shuffle", seed=1) < 1.1
    t = np.arange(2000)
    regular = np.stack([np.sin(2 * np.pi * t / 50.0 + k) for k in range(8)])
    assert lz.lzc(regular) < 0.2
    assert lz.lzc(regular, normalize=None) == float(
        lz.lz76_complexity(lz.binarize_median(regular).T.reshape(-1))
    )
    seg = lz.lzc(noise, segment_samples=500)
    assert 0.85 < seg < 1.15
    assert np.isnan(lz.lzc(np.full((2, 10), np.nan)))
    with pytest.raises(ValueError):
        lz.lzc(noise, normalize="bogus")


def test_median_binarisation():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    b = lz.binarize_median(x)
    assert b.tolist() == [[0, 0, 1, 1], [1, 1, 0, 0]]
