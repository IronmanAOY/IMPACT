"""The SVD fallback of the v2 estimators (impact_pipeline.v2.numerics)."""

import numpy as np
import pytest
import scipy.linalg

from impact_pipeline.v2 import declared_inputs as DI
from impact_pipeline.v2 import numerics as NUM


def _collinear(seed=0, n=400, k=12):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 3))
    return np.column_stack([base @ rng.normal(size=3) for _ in range(k)])


@pytest.fixture
def gesdd_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")
    monkeypatch.setattr(NUM.np.linalg, "svd", fail)
    monkeypatch.setattr(NUM.np.linalg, "lstsq", fail)


def test_numpy_results_pass_through_unchanged():
    x = _collinear()
    u, s, vt = NUM.svd(x)
    u0, s0, vt0 = np.linalg.svd(x, full_matrices=False)
    assert np.array_equal(u, u0) and np.array_equal(s, s0) and np.array_equal(vt, vt0)
    assert np.array_equal(NUM.svd(x, compute_uv=False), np.linalg.svd(x, compute_uv=False))
    assert NUM.matrix_rank(x) == np.linalg.matrix_rank(x) == 3
    b = x @ np.ones(x.shape[1])
    assert np.array_equal(NUM.lstsq(x, b)[0], np.linalg.lstsq(x, b, rcond=None)[0])


def test_the_fallback_gives_the_same_decomposition(gesdd_fails):
    x = _collinear()
    u, s, vt = NUM.svd(x)
    s_ref = scipy.linalg.svd(x, compute_uv=False)
    assert np.allclose(s, s_ref, rtol=1e-12, atol=1e-12 * s_ref[0])
    assert np.allclose((u * s) @ vt, x, atol=1e-10)
    assert NUM.svd(x, compute_uv=False).shape == (x.shape[1],)
    assert NUM.matrix_rank(x) == 3


def test_the_least_squares_fallback_solves(gesdd_fails):
    rng = np.random.default_rng(1)
    a = np.column_stack([rng.normal(size=200), np.ones(200)])
    beta = np.array([2.0, -1.0])
    x, _, rank, _ = NUM.lstsq(a, a @ beta, rcond=None)
    assert rank == 2 and np.allclose(x, beta)


def test_the_input_basis_span_survives_a_failing_driver(gesdd_fails):
    x = _collinear(n=300, k=9)
    basis = DI.InputBasis.__new__(DI.InputBasis)
    object.__setattr__(basis, "values", x.T)
    object.__setattr__(basis, "lags", ())
    span = basis.orthonormal_span()
    assert span.shape == (300, 3)
    assert np.allclose(span.T @ span, np.eye(3), atol=1e-10)
