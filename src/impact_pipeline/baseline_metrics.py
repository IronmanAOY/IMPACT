import os
import re
import glob
import logging
import numpy as np
import pandas as pd
import networkx as nx

from impact_pipeline.lz import lz76_complexity

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


def lempel_ziv_complexity(bits) -> int:
    """LZ76 phrase count (Kaspar & Schuster 1987) of a binary sequence."""
    return lz76_complexity(np.asarray(bits).ravel().astype(np.int8))


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


def _find_ts_files(base_dir, subject, session, atlas, condition):
    """
    Locate the time-series files of one subject/session/condition:
    ``<base>/<sub>/<ses>/<condition>/<sub>_*_<atlas>_ts.npy``, i.e. the same
    runs used for S/CI.
    """
    fname = f"{subject}_*_{atlas}_ts.npy"
    pat = os.path.join(base_dir, subject, session, condition, fname)
    hits = sorted(set(glob.glob(pat)))
    return hits, [pat]


def compute_baseline_metrics(df, data_dir, atlas, condition=None):
    """
    Baseline comparators (mean_conn, modularity, lzc) per subject/session.

    The comparators are computed on the runs of ``condition``, the condition
    the MPC metrics/S use, and averaged over runs (as S is). ``condition`` is
    required (the pipeline passes the analysed condition; use ``'rest'`` for
    rest runs).
    """
    if condition is None:
        raise ValueError(
            "compute_baseline_metrics needs the analysed condition (e.g. "
            "condition='rest'), so that the comparators use the runs of S/CI."
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
            'baseline_condition': condition,
            'n_baseline_runs': len(chosen),
            **{
                m: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                for m, v in vals.items()
            },
        })
        log.info("Baseline TS for %s/%s → %s", r.subject, r.session, ";".join(chosen))

    return pd.DataFrame(rows)
