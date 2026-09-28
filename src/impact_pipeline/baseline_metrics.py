import os
import re
import glob
import logging
import numpy as np
import pandas as pd
import networkx as nx

try:
    from numba import njit
except Exception:  # pragma: no cover - numba is optional
    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

log = logging.getLogger(__name__)

# Comparators written by compute_baseline_metrics and used by compare_models.
BASELINE_METRICS = ('mean_conn', 'modularity', 'lzc')
BASELINE_SEED = 0


def mean_conn(ts: np.ndarray) -> float:
    corr = np.corrcoef(ts.T)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return corr[mask].mean()


def modularity(ts, threshold=0.2, random_state=BASELINE_SEED):
    corr = np.corrcoef(ts.T)
    adj = (np.abs(corr) > threshold).astype(int)
    G = nx.from_numpy_array(adj)
    try:
        import community as louvain
        part = louvain.best_partition(G, random_state=random_state)
        return louvain.modularity(part, G)
    except ImportError:
        log.warning("python-louvain not installed; modularity=0.0")
        return 0.0


def pci_fmri(ts):
    """
    Deprecated legacy comparator (not a PCI and not used by the pipeline).

    It returns n_unique_binary_patterns / 2**n_regions, which is ~1e-118 for 400
    regions and grows with the number of timepoints, so it cannot discriminate
    states. Kept only for backward compatibility; use ``lzc`` instead.
    """
    bin_ts = (ts >= np.median(ts, axis=0)).astype(int)
    patterns = set(map(tuple, bin_ts))
    return len(patterns) / (2**ts.shape[1])


@njit(cache=False)
def _lz76_count(s):
    # Kaspar & Schuster (1987) LZ76 phrase count of a 0/1 sequence.
    n = s.shape[0]
    if n == 0:
        return 0
    if n == 1:
        return 1
    i = 0
    k = 1
    ln = 1
    c = 1
    k_max = 1
    while True:
        if s[i + k - 1] == s[ln + k - 1]:
            k += 1
            if ln + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == ln:
                c += 1
                ln += k_max
                if ln + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


def lempel_ziv_complexity(bits) -> int:
    """LZ76 phrase count (Kaspar & Schuster 1987) of a binary sequence."""
    s = np.ascontiguousarray(np.asarray(bits).ravel().astype(np.int8))
    return int(_lz76_count(s))


LZC_MAX_BITS_PER_EPOCH = 16384


def lzc(ts, random_state=BASELINE_SEED, max_bits_per_epoch=LZC_MAX_BITS_PER_EPOCH):
    """
    Normalised multichannel Lempel-Ziv complexity (Schartner et al. 2015 style).

    ts is (time, regions). Each region is binarised at its median, the series is
    cut into consecutive epochs of at most ``max_bits_per_epoch`` bits (whole time
    steps; LZ76 counting is quadratic in length), each epoch's binary matrix is
    concatenated time step by time step, and its LZ76 phrase count is divided by
    the count of the same bits randomly shuffled (fixed seed). The epoch ratios are
    averaged: ~1 for maximally irregular activity, lower for regular activity.
    Returns NaN when the input is too short, non-finite or constant.
    """
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.all(np.isfinite(x)):
        return np.nan
    bits = (x > np.median(x, axis=0, keepdims=True)).astype(np.int8)
    n_time, n_reg = bits.shape
    win = max(2, int(max_bits_per_epoch) // max(1, n_reg))
    rng = np.random.RandomState(random_state)
    ratios = []
    for start in range(0, n_time, win):
        seq = bits[start:start + win].ravel(order="C")
        if seq.size < 2 or seq.min() == seq.max():
            continue
        c_ref = lempel_ziv_complexity(rng.permutation(seq))
        if c_ref > 0:
            ratios.append(lempel_ziv_complexity(seq) / c_ref)
    return float(np.mean(ratios)) if ratios else np.nan


def _run_number(path):
    m = re.search(r"run-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 9999


def _find_ts_files(base_dir, subject, session, atlas, condition=None):
    """
    Locate time-series files for one subject/session.

    With ``condition`` (recommended; the pipeline passes the analysed condition):
    only ``<base>/<sub>/<ses>/<condition>/<sub>_*_<atlas>_ts.npy`` are returned,
    i.e. the same runs used for S/CI. Without it (legacy), files under the
    session's 'rest' folder and the session folder are searched.
    """
    if condition is not None:
        fname = f"{subject}_*_{atlas}_ts.npy"
        pat = os.path.join(base_dir, subject, session, condition, fname)
        hits = sorted(set(glob.glob(pat)))
        return hits, [pat]

    roots = [
        os.path.join(base_dir, subject, session, 'rest'),
        os.path.join(base_dir, subject, session),
    ]
    patterns = [
        f"{subject}_task-rest_run-*_{atlas}_ts.npy",
        f"{subject}_run-*_{atlas}_ts.npy",
        f"{subject}_*_rest_*_{atlas}_ts.npy",   # extra safety
        f"{subject}_*_{atlas}_ts.npy",          # fallback
    ]

    tried = []
    hits = []
    for root in roots:
        for pat in patterns:
            full = os.path.join(root, pat)
            tried.append(full)
            hits.extend(glob.glob(full))
        # final recursive fallback (use sparingly)
        rec = os.path.join(root, "**", f"{subject}_*_{atlas}_ts.npy")
        tried.append(rec)
        hits.extend(glob.glob(rec, recursive=True))

    hits = sorted(set(hits))
    return hits, tried


def compute_baseline_metrics(df, data_dir, atlas, condition=None):
    """
    Baseline comparators (mean_conn, modularity, lzc) per subject/session.

    With ``condition`` the comparators are computed on the same condition runs as
    the MPC metrics/S and averaged over runs (as S is). Without it the legacy
    rule applies (prefer one rest run), which compares different data and is
    therefore logged as a warning.
    """
    if condition is None:
        log.warning(
            "compute_baseline_metrics: no condition given; using legacy "
            "rest-preferring file selection (comparators may not match the "
            "condition used for S/CI)."
        )
    rows = []
    for _, r in df.iterrows():
        candidates, tried = _find_ts_files(
            data_dir, r.subject, r.session, atlas, condition=condition
        )

        if not candidates:
            msg = (
                f"No TS file for {r.subject}/{r.session}.\n"
                "Looked in these patterns:\n  - " + "\n  - ".join(tried)
            )
            raise FileNotFoundError(msg)

        if condition is None:
            # Legacy: prefer files found under 'rest' and the lowest run number
            def pref_key(p):
                in_rest = "/rest/" in p.replace("\\", "/")
                return (0 if in_rest else 1, _run_number(p), p)

            chosen = [sorted(candidates, key=pref_key)[0]]
        else:
            chosen = sorted(candidates, key=lambda p: (_run_number(p), p))

        vals = {m: [] for m in BASELINE_METRICS}
        for path in chosen:
            ts = np.load(path)
            vals['mean_conn'].append(mean_conn(ts))
            vals['modularity'].append(modularity(ts))
            vals['lzc'].append(lzc(ts))

        rows.append({
            **r.to_dict(),
            'ts_path': ";".join(chosen),
            'baseline_condition': (
                condition if condition is not None else "legacy_rest_preferred"
            ),
            'n_baseline_runs': len(chosen),
            **{
                m: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                for m, v in vals.items()
            },
        })
        log.info("Baseline TS for %s/%s → %s", r.subject, r.session, ";".join(chosen))

    return pd.DataFrame(rows)
