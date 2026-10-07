"""
SVD-based linear algebra with a fallback driver for the v2 estimators.

LAPACK's divide-and-conquer SVD (``gesdd``, which ``numpy.linalg`` calls)
occasionally fails to converge on exactly rank-deficient matrices, such as
an input basis whose lagged and filtered cue columns are collinear. It then
raises ``LinAlgError`` although the matrix is finite and well scaled. Each
function here calls numpy first, so every result numpy computes stays
bit-identical. Only when numpy raises does it repeat the computation with
SciPy's QR-iteration driver (``gesvd``; ``gelsy`` for least squares), which
converges on these matrices.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def svd(a, full_matrices: bool = False, compute_uv: bool = True):
    """``numpy.linalg.svd`` with the ``gesvd`` driver as fallback; the
    default ``full_matrices=False`` is the one every v2 caller uses."""
    try:
        return np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    except np.linalg.LinAlgError:
        import scipy.linalg

        return scipy.linalg.svd(np.asarray(a, dtype=float),
                                full_matrices=full_matrices, compute_uv=compute_uv,
                                lapack_driver="gesvd")


def matrix_rank(a, tol: Optional[float] = None) -> int:
    """``numpy.linalg.matrix_rank`` (same default tolerance) on :func:`svd`."""
    a = np.asarray(a, dtype=float)
    if a.ndim < 2:
        return int(not np.all(a == 0))
    s = svd(a, compute_uv=False)
    if s.size == 0:
        return 0
    if tol is None:
        tol = s.max() * max(a.shape[-2:]) * np.finfo(s.dtype).eps
    return int(np.count_nonzero(s > tol))


def lstsq(a, b, rcond=None):
    """``numpy.linalg.lstsq`` with SciPy's ``gelsy`` (pivoted QR) as
    fallback. The fallback returns the solution and the rank, but no
    residuals and no singular values (empty arrays), which v2 callers do not read."""
    try:
        return np.linalg.lstsq(a, b, rcond=rcond)
    except np.linalg.LinAlgError:
        import scipy.linalg

        a = np.asarray(a, dtype=float)
        cond = None if rcond is None else float(rcond)
        x, _, rank, _ = scipy.linalg.lstsq(a, np.asarray(b, dtype=float), cond=cond,
                                           lapack_driver="gelsy")
        return x, np.empty(0), int(rank), np.empty(0)
