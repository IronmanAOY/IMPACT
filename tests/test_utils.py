import numpy as np
import pytest
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

from impact_pipeline.utils import compute_midrank, delong_roc_test, fast_delong


def _bruteforce_delong(y, scores):
    """Reference DeLong (1988) via explicit placement values (O(m*n) kernel)."""
    y = np.asarray(y).astype(int)
    comps10, comps01, aucs = [], [], []
    for s in scores:
        s = np.asarray(s, dtype=float)
        pos, neg = s[y == 1], s[y == 0]
        psi = (pos[:, None] > neg[None, :]).astype(float)
        psi += 0.5 * (pos[:, None] == neg[None, :])
        aucs.append(psi.mean())
        comps10.append(psi.mean(axis=1))
        comps01.append(psi.mean(axis=0))
    m, n = int((y == 1).sum()), int((y == 0).sum())
    cov = np.cov(np.vstack(comps10)) / m + np.cov(np.vstack(comps01)) / n
    return np.asarray(aucs), np.atleast_2d(cov)


def test_midrank():
    x = [1, 2, 2, 3]
    mid = compute_midrank(x)
    assert len(mid) == 4
    np.testing.assert_allclose(mid, [1.0, 2.5, 2.5, 4.0])
    z = np.random.RandomState(0).randint(0, 5, size=50).astype(float)
    np.testing.assert_allclose(compute_midrank(z), rankdata(z))


def test_fast_delong():
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2])
    auc, var = fast_delong(y_true, y_score)
    assert 0 <= auc <= 1
    # Perfect separation: AUC must be exactly 1 (the old code returned 0.5).
    assert auc == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_delong_matches_sklearn_and_bruteforce_with_ties(seed):
    rng = np.random.RandomState(seed)
    y = np.r_[np.ones(17), np.zeros(23)].astype(int)
    s = np.round(rng.randn(40) + 0.7 * y, 1)  # rounding creates ties
    auc, var = fast_delong(y, s)
    ref_auc, ref_cov = _bruteforce_delong(y, [s])
    assert auc == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert auc == pytest.approx(ref_auc[0], abs=1e-12)
    assert var == pytest.approx(ref_cov[0, 0], rel=1e-10)
    # Order invariance (the old variance changed when rows were shuffled).
    perm = rng.permutation(40)
    auc_p, var_p = fast_delong(y[perm], s[perm])
    assert auc_p == pytest.approx(auc, abs=1e-12)
    assert var_p == pytest.approx(var, rel=1e-10)


def test_hanley_mcneil_published_auc():
    # Hanley & McNeil (1982) CT rating data: published area 0.893.
    normal_counts = [33, 6, 6, 11, 2]
    abnormal_counts = [3, 2, 2, 11, 33]
    ratings = np.arange(1, 6)
    scores = np.r_[
        np.repeat(ratings, normal_counts), np.repeat(ratings, abnormal_counts)
    ]
    y = np.r_[np.zeros(sum(normal_counts)), np.ones(sum(abnormal_counts))].astype(int)
    auc, var = fast_delong(y, scores)
    assert round(auc, 3) == 0.893
    assert var == pytest.approx(_bruteforce_delong(y, [scores])[1][0, 0], rel=1e-10)


def test_delong_paired_test_uses_covariance():
    rng = np.random.RandomState(0)
    y = np.r_[np.ones(20), np.zeros(20)].astype(int)
    s1 = rng.randn(40) + 0.6 * y
    s2 = s1 + rng.randn(40) * 0.5
    det = delong_roc_test(y, s1, s2, return_details=True)
    aucs, cov = _bruteforce_delong(y, [s1, s2])
    z = (aucs[0] - aucs[1]) / np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    assert det["p"] == pytest.approx(2 * norm.sf(abs(z)), rel=1e-9)
    assert det["auc1"] == pytest.approx(roc_auc_score(y, s1))
    assert delong_roc_test(y, s1, s2) == pytest.approx(det["p"])
    # Correlated scores: the covariance term shrinks var(delta) well below var1+var2.
    assert det["var_delta"] < 0.5 * (cov[0, 0] + cov[1, 1])


def test_delong_null_calibration_and_power():
    rng = np.random.RandomState(123)
    n_sim, rej_null, rej_alt = 300, 0, 0
    y = np.r_[np.ones(30), np.zeros(30)].astype(int)
    for _ in range(n_sim):
        latent = rng.randn(60) + 1.5 * y
        a = latent + rng.randn(60) * 0.7
        b = latent + rng.randn(60) * 0.7          # same AUC in expectation (H0)
        c = 0.3 * latent + rng.randn(60) * 2.0    # clearly lower AUC (H1)
        rej_null += delong_roc_test(y, a, b) < 0.05
        rej_alt += delong_roc_test(y, a, c) < 0.05
    assert 0.02 <= rej_null / n_sim <= 0.09
    assert rej_alt / n_sim > 0.6


def test_delong_requires_finite_scores_and_two_cases_per_class():
    # The old compute_midrank looped forever on NaN (NaN != NaN never closes a tie).
    with pytest.raises(ValueError):
        compute_midrank([0.1, np.nan, 0.3])
    with pytest.raises(ValueError):
        fast_delong(np.array([1, 0, 1, 0]), np.array([0.1, np.nan, 0.3, 0.2]))
    with pytest.raises(ValueError):
        fast_delong(np.array([1, 0]), np.array([0.1, 0.2]))
