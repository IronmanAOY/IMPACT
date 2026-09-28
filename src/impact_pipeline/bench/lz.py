"""
Lempel-Ziv complexity (LZ76) of median-binarised multichannel data.

``lz76_complexity`` counts the phrases of the Lempel & Ziv (1976) exhaustive
parsing with the algorithm of Kaspar & Schuster (1987, Phys. Rev. A 36, 842).
``lzc`` follows Schartner et al. (2015, 2017): every channel is binarised at
its median, the binary matrix is concatenated time point by time point
(all channels at t, then t + 1, ...) and the phrase count is normalised,
either by the asymptotic bound for a random binary sequence ``n / log2(n)``
(``normalize='n_log_n'``) or by the complexity of the same sequence after a
random shuffle (``normalize='shuffle'``).
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


NORMALIZATIONS = ("n_log_n", "shuffle", None)


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


def binarize_median(ts) -> np.ndarray:
    """Per-channel median split of a channels x time array (1 above the median)."""
    x = np.asarray(ts, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"ts must be 1D or 2D (channels x time), got {x.shape}")
    med = np.nanmedian(x, axis=1, keepdims=True)
    return (x > med).astype(np.int8)


def _lzc_single(x: np.ndarray, normalize, n_shuffles: int, rng) -> float:
    seq = binarize_median(x).T.reshape(-1)
    c = lz76_complexity(seq)
    n = seq.size
    if normalize is None:
        return float(c)
    if normalize == "n_log_n":
        if n < 2:
            return float("nan")
        return float(c * np.log2(n) / n)
    ref = np.mean(
        [lz76_complexity(rng.permutation(seq)) for _ in range(max(1, int(n_shuffles)))]
    )
    return float(c / ref) if ref > 0 else float("nan")


def lzc(
    ts, normalize="n_log_n", n_shuffles: int = 1, seed: int = 0, segment_samples=None
) -> float:
    """
    Normalised LZ76 complexity of a channels x time array (see module
    docstring). With ``segment_samples`` the run is cut into consecutive
    non-overlapping segments (a trailing remainder shorter than a segment is
    dropped), each segment is binarised at its own medians and scored, and the
    mean is returned (the parsing cost grows faster than linearly with the
    sequence length). NaN samples are not allowed (NaN is returned).
    """
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}")
    x = np.asarray(ts, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.size == 0 or not np.all(np.isfinite(x)):
        return float("nan")
    rng = np.random.default_rng(int(seed))
    if segment_samples is None:
        return _lzc_single(x, normalize, n_shuffles, rng)
    seg = int(segment_samples)
    if seg < 2:
        raise ValueError("segment_samples must be >= 2")
    n_seg = x.shape[1] // seg
    if n_seg == 0:
        return float("nan")
    vals = [
        _lzc_single(x[:, k * seg : (k + 1) * seg], normalize, n_shuffles, rng)
        for k in range(n_seg)
    ]
    return float(np.mean(vals))
