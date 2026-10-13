"""
Lempel-Ziv complexity (LZ76).

``lz76_complexity`` counts the phrases of the Lempel & Ziv (1976) exhaustive
parsing with the algorithm of Kaspar & Schuster (1987, Phys. Rev. A 36, 842),
e.g. 6 for ``0001101001000101``. The PDI LZ statistic of
:mod:`impact_pipeline.mpc_metrics` and the LZc comparator of
:mod:`impact_pipeline.baseline_metrics` both count phrases with it.
"""

from __future__ import annotations

import numpy as np

try:  # numba is a core dependency of the package; keep a pure-Python fallback.
    from numba import njit
except Exception:  # pragma: no cover

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        return wrap(args[0]) if args and callable(args[0]) else wrap


@njit(cache=False)
def _lz76_kernel(s):
    n = s.shape[0]
    if n == 0:
        return 0
    if n == 1:
        return 1
    i = 0
    k = 1
    ell = 1
    c = 1
    k_max = 1
    while True:
        if s[i + k - 1] == s[ell + k - 1]:
            k += 1
            if ell + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == ell:
                c += 1
                ell += k_max
                if ell + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


def lz76_complexity(sequence) -> int:
    """Number of phrases of the LZ76 parsing of a symbol sequence."""
    s = np.asarray(sequence).reshape(-1)
    if s.dtype.kind not in "biu":
        _, s = np.unique(s, return_inverse=True)
    return int(_lz76_kernel(np.ascontiguousarray(s.astype(np.int64))))
