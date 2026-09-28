import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
import pandas as pd
import warnings
import itertools
import json
import os
import hashlib
import time
import math
import atexit
import tempfile
import shutil
import weakref
import concurrent.futures
import sqlite3
from collections import OrderedDict
from multiprocessing import shared_memory
from scipy.stats import entropy
from scipy.stats import median_abs_deviation
from scipy.signal import butter, sosfiltfilt, hilbert
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import logging

from impact_pipeline.hardware_backend import (
    accelerated_corrcoef,
    accelerated_dot,
    accelerated_pinv_dot,
    accelerated_row_norm,
    accelerated_solve,
    accelerated_svd_values,
    accelerated_zscore,
    get_array_module,
    resolve_hardware_backend,
    to_numpy,
)

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

log = logging.getLogger(__name__)

# numba's on-disk JIT cache defaults to a __pycache__ directory next to this
# file, i.e. inside the source tree (read-only in containers / HPC installs, and
# written regardless of PYTHONDONTWRITEBYTECODE). Only enable it when the user
# points NUMBA_CACHE_DIR at a writable cache location (e.g. node-local scratch).
_NUMBA_DISK_CACHE = bool(os.environ.get("NUMBA_CACHE_DIR", "").strip())


class _IIMDiskKernelCache:
    """
    Disk-backed cache for cut-kernel values keyed by:
      (direction, m_mask, m_key, z_mask, induced_partition_key)

    Uses a small in-memory LRU front-cache plus SQLite persistence with
    batched writes to keep RAM bounded while allowing cache reuse across cuts
    and resumes.
    """

    def __init__(
        self,
        path: str,
        signature: dict | None = None,
        memory_entries: int = 300_000,
        flush_batch: int = 5_000,
    ):
        self.path = str(path)
        self.memory_entries = max(10_000, int(memory_entries))
        self.flush_batch = max(100, int(flush_batch))
        self._mem = OrderedDict()
        self._pending = {}
        self.hits_mem = 0
        self.hits_disk = 0
        self.misses = 0
        self.writes = 0

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-100000")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kernel_cache (
                d INTEGER NOT NULL,
                m_mask INTEGER NOT NULL,
                m_key INTEGER NOT NULL,
                z_mask INTEGER NOT NULL,
                pi INTEGER NOT NULL,
                v REAL NOT NULL,
                PRIMARY KEY (d, m_mask, m_key, z_mask, pi)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

        if signature is not None:
            sig_txt = json.dumps(signature, sort_keys=True, ensure_ascii=True)
            row = self.conn.execute("SELECT v FROM meta WHERE k='signature'").fetchone()
            old_sig = None if row is None else str(row[0])
            if old_sig is not None and old_sig != sig_txt:
                self.conn.execute("DELETE FROM kernel_cache")
                self.conn.execute("DELETE FROM meta WHERE k='signature'")
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(k, v) VALUES ('signature', ?)",
                    (sig_txt,),
                )
                self.conn.commit()
            elif old_sig is None:
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(k, v) VALUES ('signature', ?)",
                    (sig_txt,),
                )
                self.conn.commit()

    def _touch_mem(self, key, val):
        self._mem[key] = float(val)
        self._mem.move_to_end(key, last=True)
        while len(self._mem) > self.memory_entries:
            self._mem.popitem(last=False)

    def get(self, key):
        if key in self._mem:
            self.hits_mem += 1
            v = self._mem[key]
            self._mem.move_to_end(key, last=True)
            return float(v)

        if key in self._pending:
            self.hits_mem += 1
            v = float(self._pending[key])
            self._touch_mem(key, v)
            return v

        row = None
        for attempt in range(6):
            try:
                row = self.conn.execute(
                    """
                    SELECT v FROM kernel_cache
                    WHERE d=? AND m_mask=? AND m_key=? AND z_mask=? AND pi=?
                    """,
                    (int(key[0]), int(key[1]), int(key[2]), int(key[3]), int(key[4])),
                ).fetchone()
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt >= 5:
                    raise
                time.sleep(0.01 * (2 ** attempt))
        if row is None:
            self.misses += 1
            return None
        v = float(row[0])
        self.hits_disk += 1
        self._touch_mem(key, v)
        return v

    def set(self, key, value):
        v = float(value)
        self._touch_mem(key, v)
        self._pending[key] = v
        if len(self._pending) >= self.flush_batch:
            self.flush()

    def flush(self):
        if not self._pending:
            return
        rows = [
            (int(k[0]), int(k[1]), int(k[2]), int(k[3]), int(k[4]), float(v))
            for k, v in self._pending.items()
        ]
        for attempt in range(6):
            try:
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO kernel_cache(d, m_mask, m_key, z_mask, pi, v)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                self.conn.commit()
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt >= 5:
                    raise
                time.sleep(0.01 * (2 ** attempt))
        self.writes += int(len(rows))
        self._pending.clear()

    def stats(self):
        return {
            "hits_mem": int(self.hits_mem),
            "hits_disk": int(self.hits_disk),
            "misses": int(self.misses),
            "writes": int(self.writes),
            "path": self.path,
        }

    def close(self):
        try:
            self.flush()
        finally:
            try:
                self.conn.close()
            except Exception:
                pass


@njit(cache=_NUMBA_DISK_CACHE)
def _build_partition_maps_numba(base, z_len, posA, posB):
    nZ = 1
    for _ in range(z_len):
        nZ *= base
    mapA = np.empty(nZ, dtype=np.int64)
    mapB = np.empty(nZ, dtype=np.int64)
    zv = np.empty(z_len, dtype=np.int64)

    nA = int(posA.shape[0])
    nB = int(posB.shape[0])
    for kz in range(nZ):
        v = kz
        for i in range(z_len - 1, -1, -1):
            zv[i] = v % base
            v //= base

        keyA = 0
        for i in range(nA):
            keyA = keyA * base + int(zv[int(posA[i])])
        mapA[kz] = keyA

        keyB = 0
        for i in range(nB):
            keyB = keyB * base + int(zv[int(posB[i])])
        mapB[kz] = keyB
    return mapA, mapB


@njit(cache=_NUMBA_DISK_CACHE)
def _compose_product_distribution_numba(pA, pB, mapA, mapB):
    n = int(mapA.shape[0])
    q = np.empty(n, dtype=np.float64)
    s = 0.0
    for i in range(n):
        v = float(pA[int(mapA[i])]) * float(pB[int(mapB[i])])
        q[i] = v
        s += v
    if s <= 0.0:
        inv = 1.0 / float(max(1, n))
        for i in range(n):
            q[i] = inv
    else:
        inv = 1.0 / s
        for i in range(n):
            q[i] *= inv
    return q


@njit(cache=_NUMBA_DISK_CACHE)
def _jsd_numba(p, q):
    n = int(p.shape[0])
    sp = 0.0
    sq = 0.0
    for i in range(n):
        sp += float(p[i])
        sq += float(q[i])
    if sp <= 0.0 or sq <= 0.0:
        return 0.0

    kl_pm = 0.0
    kl_qm = 0.0
    for i in range(n):
        pi = float(p[i]) / sp
        qi = float(q[i]) / sq
        mi = 0.5 * (pi + qi)
        if pi > 0.0 and mi > 0.0:
            kl_pm += pi * (np.log(pi / mi) / np.log(2.0))
        if qi > 0.0 and mi > 0.0:
            kl_qm += qi * (np.log(qi / mi) / np.log(2.0))
    return 0.5 * (kl_pm + kl_qm)


_IIM_PHASE1_CTX = None

# Bump whenever a change alters Psi, cut scores or cached kernel values, so that
# checkpoints and kernel caches written by older code are not reused.
# v4: both mechanism/purview bipartition pairings in the MIP search, state-by-node
#     TPM estimators, explicit node selection / state-budget policy.
IIM_ALGORITHM_VERSION = "iim-v4-2026.09"


def _iim_remove_sqlite_files(path):
    """Remove an SQLite database and its WAL/SHM/journal side files (best effort)."""
    if not path:
        return
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.remove(f"{path}{suffix}")
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning(
                "Could not remove IIM kernel cache file %s%s: %s", path, suffix, exc
            )


class _IIMKernelCacheMissError(RuntimeError):
    """Raised when lookup-only IIM aggregation hits a missing kernel cache key."""


def _iim_encode_vals(vals, base):
    key = 0
    for v in vals:
        key = key * base + int(v)
    return int(key)


def _iim_decode_key(key, k, base):
    out = [0] * k
    v = int(key)
    for i in range(k - 1, -1, -1):
        out[i] = v % base
        v //= base
    return tuple(out)


def _iim_subset_key_matrix(states, subset, base):
    if len(subset) == 0:
        return np.zeros(states.shape[0], dtype=np.int64)
    cols = states[:, subset].astype(np.int64, copy=False)
    mult = (base ** np.arange(len(subset) - 1, -1, -1, dtype=np.int64))
    return (cols * mult).sum(axis=1).astype(np.int64)


def _iim_enumerate_bipartitions(nodes):
    n = len(nodes)
    if n < 2:
        return []
    out = []
    node_set = tuple(nodes)
    for k in range(1, (n // 2) + 1):
        for A in itertools.combinations(node_set, k):
            A = tuple(A)
            B = tuple(x for x in node_set if x not in A)
            if k == n - k and A[0] > B[0]:
                continue
            out.append((A, B))
    return out


def _iim_marginalize(dist_full, keys, n_keys):
    p = np.bincount(keys, weights=dist_full, minlength=n_keys).astype(float)
    s = p.sum()
    if s <= 0:
        return np.ones(n_keys, dtype=float) / float(n_keys)
    return p / s


def _iim_enumerate_subsets(nodes, max_size):
    out = []
    for r in range(1, min(int(max_size), len(nodes)) + 1):
        out.extend(tuple(c) for c in itertools.combinations(nodes, r))
    return out


def _iim_discretize_per_node(arr, base_bins):
    n, t = arr.shape
    out = np.zeros((n, t), dtype=np.int16)
    for i in range(n):
        x = arr[i]
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            continue
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if np.isclose(vmin, vmax):
            continue
        edges = np.quantile(finite, np.linspace(0.0, 1.0, int(base_bins) + 1))
        if np.unique(edges).size != int(base_bins) + 1:
            edges = np.linspace(vmin, vmax, int(base_bins) + 1)
        bins_inner = edges[1:-1]
        z = np.digitize(x, bins_inner, right=False)
        z = np.clip(z, 0, int(base_bins) - 1)
        z[~np.isfinite(x)] = 0
        out[i] = z.astype(np.int16)
    return out


IIM_NODE_SELECTION_RULES = ("variance", "index")
IIM_STATE_BUDGET_POLICIES = ("reduce_bins_first", "reduce_nodes_first", "error")
# Relative variance spread below which a variance ranking is treated as tied
# (e.g. z-scored ROI series, where all variances equal 1 up to float noise).
_IIM_VARIANCE_TIE_RTOL = 1e-6


def _iim_select_nodes_with_info(arr, max_n, rule="variance", node_indices=None):
    """
    Deterministic, declared node selection for the IIM subsystem.

    rule='variance': highest temporal variance; ties (relative spread below
        ``_IIM_VARIANCE_TIE_RTOL``) are broken by ascending node index, and a
        fully tied ranking is flagged as degenerate (it then equals rule='index').
    rule='index': the first ``max_n`` nodes in input order (a predeclared order).
    node_indices: explicit node list (overrides ``rule``); truncated to ``max_n``.
    """
    n = int(arr.shape[0])
    max_n = int(max_n)
    info = {"rule": str(rule), "degenerate_ranking": False}
    if node_indices is not None:
        idx = np.asarray(node_indices, dtype=int).reshape(-1)
        if (
            idx.size == 0
            or np.any(idx < 0)
            or np.any(idx >= n)
            or np.unique(idx).size != idx.size
        ):
            raise ValueError(
                f"node_indices must be unique indices in [0, {n}), got {idx.tolist()}"
            )
        info["rule"] = "explicit"
        return np.sort(idx[:max_n].astype(int)), info
    if rule not in IIM_NODE_SELECTION_RULES:
        raise ValueError(
            f"node_selection must be one of {IIM_NODE_SELECTION_RULES}, got {rule!r}"
        )
    if n <= max_n:
        return np.arange(n, dtype=int), info
    if rule == "index":
        return np.arange(max_n, dtype=int), info
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var = np.nanvar(arr, axis=1)
    var = np.where(np.isfinite(var), var, -np.inf)
    # Ties are relative to the variances themselves (log scale), not to the
    # largest variance: one high-variance node must not collapse the ranking of
    # all others. Anchoring at the maximum keeps variances that are equal up to
    # float noise (z-scored input) on the same level. Zero variance ranks below
    # any positive variance and above undefined (all-NaN) nodes.
    positive = np.isfinite(var) & (var > 0)
    ranked = np.where(np.isfinite(var), -np.finfo(float).max, -np.inf)
    if np.any(positive):
        log_ref = float(np.log(np.max(var[positive])))
        ranked[positive] = np.round(
            (np.log(var[positive]) - log_ref) / _IIM_VARIANCE_TIE_RTOL
        )
    order = np.lexsort((np.arange(n), -ranked))
    idx = order[:max_n]
    cutoff = ranked[idx[-1]]
    # Ranking is arbitrary when the max_n-th node ties with an unselected one.
    info["degenerate_ranking"] = bool(np.any(ranked[order[max_n:]] == cutoff))
    return np.sort(idx.astype(int)), info


def _iim_select_nodes(arr, max_n, rule="variance", node_indices=None):
    idx, _info = _iim_select_nodes_with_info(
        arr, max_n, rule=rule, node_indices=node_indices
    )
    return idx


def _iim_node_conditional_tpm(curr_keys, nxt, states_full, base, alpha, estimator):
    """
    State-by-node TPM: T(s'|s) = prod_i p_i(s'_i | s) (conditional independence
    of next-step node states given the current system state, as in IIT).

    'node_laplace'   : p_i(.|s) = (C_i(s, .) + alpha) / (n_s + alpha*K).
    'node_shrinkage' : James-Stein shrinkage (Hausser & Strimmer 2009) of the
                       row-wise ML estimate towards the node's own
                       transition p_i(x'_i | x_i) (same lag, pooled over all
                       transitions, i.e. no cross-node influence),
                       with data-driven intensity lambda_s in [0, 1]. Rows with
                       n_s <= 1 or unobserved rows fall back to that target, so
                       unsupported cross-node dependence is not invented from a
                       single transition and unobserved rows are not uniform.
    """
    n_states, n = int(states_full.shape[0]), int(states_full.shape[1])
    base = int(base)
    curr_keys = np.asarray(curr_keys, dtype=np.int64)
    n_obs = np.bincount(curr_keys, minlength=n_states).astype(float)
    curr_states = (
        states_full[curr_keys] if curr_keys.size else np.zeros((0, n), dtype=np.int16)
    )
    tpm = np.ones((n_states, n_states), dtype=float)
    for i in range(n):
        x_next = np.asarray(nxt[:, i], dtype=np.int64)
        counts = np.zeros((n_states, base), dtype=float)
        np.add.at(counts, (curr_keys, x_next), 1.0)
        if estimator == "node_laplace":
            p_i = (counts + float(alpha)) / (n_obs[:, None] + float(alpha) * base)
        else:
            self_counts = np.zeros((base, base), dtype=float)
            np.add.at(
                self_counts,
                (np.asarray(curr_states[:, i], dtype=np.int64), x_next),
                1.0,
            )
            self_p = (self_counts + float(alpha)) / (
                self_counts.sum(axis=1, keepdims=True) + float(alpha) * base
            )
            target = self_p[np.asarray(states_full[:, i], dtype=np.int64)]
            with np.errstate(divide="ignore", invalid="ignore"):
                theta = np.where(
                    n_obs[:, None] > 0, counts / np.maximum(n_obs[:, None], 1.0), target
                )
                num = 1.0 - np.sum(theta * theta, axis=1)
                den = (n_obs - 1.0) * np.sum((target - theta) ** 2, axis=1)
                lam = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
            lam = np.where(n_obs <= 1.0, 1.0, np.clip(lam, 0.0, 1.0))
            p_i = lam[:, None] * target + (1.0 - lam[:, None]) * theta
        p_i = p_i / np.sum(p_i, axis=1, keepdims=True)
        tpm *= p_i[:, np.asarray(states_full[:, i], dtype=np.int64)]
    tpm /= np.sum(tpm, axis=1, keepdims=True)
    return tpm


IIM_TPM_ESTIMATORS = ("node_shrinkage", "node_laplace", "joint_laplace")


def _iim_build_states_and_tpm(
    disc,
    base,
    lag,
    alpha,
    hardware_backend=None,
    estimator="joint_laplace",
):
    if estimator not in IIM_TPM_ESTIMATORS:
        raise ValueError(
            f"tpm_estimator must be one of {IIM_TPM_ESTIMATORS}, got {estimator!r}"
        )
    n, t = disc.shape
    t_eff = int(t - lag)
    curr = disc[:, :t_eff].T.astype(np.int16, copy=False)
    nxt = disc[:, lag:].T.astype(np.int16, copy=False)

    full_subset = tuple(range(int(n)))
    curr_keys = _iim_subset_key_matrix(curr, full_subset, int(base))
    nxt_keys = _iim_subset_key_matrix(nxt, full_subset, int(base))

    n_states = int(int(base) ** int(n))
    sid = np.arange(n_states, dtype=np.int64)[:, None]
    powv = (int(base) ** np.arange(n - 1, -1, -1, dtype=np.int64))[None, :]
    states_full = ((sid // powv) % int(base)).astype(np.int16)

    if estimator != "joint_laplace":
        # Small (n_states x K) count tables: computed on the host for every backend
        # so that CPU and accelerator runs are bit-identical.
        tpm = _iim_node_conditional_tpm(
            curr_keys, nxt, states_full, base, alpha, estimator
        )
        return curr, tpm, states_full

    backend = resolve_hardware_backend(hardware_backend)
    if backend.accelerator:
        xp = get_array_module(backend)
        counts_d = xp.zeros((n_states, n_states), dtype=xp.float64)
        xp.add.at(counts_d, (xp.asarray(curr_keys), xp.asarray(nxt_keys)), 1.0)
        row_sum_d = counts_d.sum(axis=1, keepdims=True)
        tpm = to_numpy((counts_d + float(alpha)) / (row_sum_d + float(alpha) * n_states))
    else:
        counts = np.zeros((n_states, n_states), dtype=float)
        np.add.at(counts, (curr_keys, nxt_keys), 1.0)
        row_sum = counts.sum(axis=1, keepdims=True)
        tpm = (counts + float(alpha)) / (row_sum + float(alpha) * n_states)
    return curr, tpm, states_full


def _iim_build_phase1_chunks_adaptive(
    remaining_mechanisms,
    max_chunk_size,
    workers,
):
    """
    Contiguous mechanism chunks (resume-safe with prefix-based checkpoints).
    With several workers: target 4 mechanisms per task for the first 50% of
    mechanisms, 2 for the next 30% and 1 for the last 20%, while keeping enough
    tasks to occupy all workers.
    """
    mechs = tuple(remaining_mechanisms)
    if not mechs:
        return []
    max_chunk = max(1, int(max_chunk_size))
    workers_eff = max(1, int(workers))
    if workers_eff <= 1:
        return [
            tuple(mechs[i:i + max_chunk])
            for i in range(0, len(mechs), max_chunk)
        ]

    chunks = []
    idx = 0
    n_total = int(len(mechs))
    while idx < n_total:
        rem = int(n_total - idx)
        active_workers = max(1, min(workers_eff, rem))

        progress = float(idx) / float(max(1, n_total))
        if progress >= 0.80:
            stage_target = 1
        elif progress >= 0.50:
            stage_target = 2
        else:
            stage_target = 4

        stage_target = min(int(stage_target), int(max_chunk))
        c = min(stage_target, max(1, rem // active_workers))
        c = max(1, min(int(c), rem))
        chunks.append(tuple(mechs[idx:idx + c]))
        idx += c
    return chunks


def _iim_build_cut_tpm(tpm, states_full, base, A, B, hardware_backend=None):
    A = tuple(sorted(A))
    B = tuple(sorted(B))
    n_states = int(states_full.shape[0])
    keyA = _iim_subset_key_matrix(states_full, A, int(base))
    keyB = _iim_subset_key_matrix(states_full, B, int(base))
    nA = int(int(base) ** len(A))
    nB = int(int(base) ** len(B))

    rowsA = [np.where(keyA == k)[0] for k in range(nA)]
    rowsB = [np.where(keyB == k)[0] for k in range(nB)]

    backend = resolve_hardware_backend(hardware_backend)
    if backend.accelerator:
        xp = get_array_module(backend)
        tpm_d = xp.asarray(tpm, dtype=xp.float64)
        keyA_d = xp.asarray(keyA)
        keyB_d = xp.asarray(keyB)

        def _marginalize_d(dist_full_d, keys_d, n_keys):
            p_d = xp.bincount(keys_d, weights=dist_full_d, minlength=int(n_keys)).astype(xp.float64)
            s_d = p_d.sum()
            if float(to_numpy(s_d)) <= 0:
                return xp.ones(int(n_keys), dtype=xp.float64) / float(n_keys)
            return p_d / s_d

        pA_next_d = xp.zeros((nA, nA), dtype=xp.float64)
        pB_next_d = xp.zeros((nB, nB), dtype=xp.float64)
        for ka in range(nA):
            rr = rowsA[ka]
            if rr.size == 0:
                pA_next_d[ka, :] = 1.0 / float(nA)
            else:
                dnext_d = tpm_d[xp.asarray(rr), :].mean(axis=0)
                pA_next_d[ka, :] = _marginalize_d(dnext_d, keyA_d, nA)

        for kb in range(nB):
            rr = rowsB[kb]
            if rr.size == 0:
                pB_next_d[kb, :] = 1.0 / float(nB)
            else:
                dnext_d = tpm_d[xp.asarray(rr), :].mean(axis=0)
                pB_next_d[kb, :] = _marginalize_d(dnext_d, keyB_d, nB)

        tpm_cut_d = pA_next_d[keyA_d][:, keyA_d] * pB_next_d[keyB_d][:, keyB_d]
        row_sums_d = tpm_cut_d.sum(axis=1, keepdims=True)
        bad_d = row_sums_d[:, 0] <= 0
        if bool(to_numpy((~bad_d).any())):
            tpm_cut_d[~bad_d, :] = tpm_cut_d[~bad_d, :] / row_sums_d[~bad_d]
        if bool(to_numpy(bad_d.any())):
            tpm_cut_d[bad_d, :] = 1.0 / float(n_states)
        return to_numpy(tpm_cut_d)

    pA_next = np.zeros((nA, nA), dtype=float)
    pB_next = np.zeros((nB, nB), dtype=float)
    for ka in range(nA):
        rr = rowsA[ka]
        if rr.size == 0:
            pA_next[ka, :] = 1.0 / float(nA)
        else:
            dnext = tpm[rr, :].mean(axis=0)
            pA_next[ka, :] = _iim_marginalize(dnext, keyA, nA)

    for kb in range(nB):
        rr = rowsB[kb]
        if rr.size == 0:
            pB_next[kb, :] = 1.0 / float(nB)
        else:
            dnext = tpm[rr, :].mean(axis=0)
            pB_next[kb, :] = _iim_marginalize(dnext, keyB, nB)

    tpm_cut = pA_next[keyA][:, keyA] * pB_next[keyB][:, keyB]
    row_sums = tpm_cut.sum(axis=1, keepdims=True)
    bad = row_sums[:, 0] <= 0
    if np.any(~bad):
        tpm_cut[~bad, :] = tpm_cut[~bad, :] / row_sums[~bad]
    if np.any(bad):
        tpm_cut[bad, :] = 1.0 / float(n_states)
    return tpm_cut


def _iim_all_system_cuts(n_nodes_sys, part_mode):
    nodes = tuple(range(int(n_nodes_sys)))
    all_cuts = _iim_enumerate_bipartitions(nodes)
    if str(part_mode) == "balanced":
        out = []
        for A, B in all_cuts:
            if abs(len(A) - len(B)) <= 1:
                out.append((A, B))
        return out
    return all_cuts


def _iim_cut_to_key(A, B):
    return f"{','.join(map(str, A))}|{','.join(map(str, B))}"


def _iim_cut_from_payload(payload):
    if (
        isinstance(payload, (list, tuple))
        and len(payload) == 2
        and isinstance(payload[0], (list, tuple))
        and isinstance(payload[1], (list, tuple))
    ):
        return (tuple(int(x) for x in payload[0]), tuple(int(x) for x in payload[1]))
    return None


def prepare_iim_problem(
    ts: np.ndarray,
    *,
    bins: int = 3,
    lag_trs: int = 1,
    n_parts: int | None = None,
    rng: int | np.random.RandomState = 0,
    partition_mode: str = "all",
    max_nodes: int | None = None,
    max_mechanism_size: int | None = None,
    max_purview_size: int | None = None,
    tpm_alpha: float = 1e-3,
    max_state_space: int = 1500,
    hardware_backend=None,
    tpm_estimator: str = "node_shrinkage",
    node_selection: str = "variance",
    node_indices=None,
    state_budget_policy: str = "reduce_bins_first",
    log_label: str | None = None,
):
    """
    Build the discrete IIM problem (subsystem, discretisation, TPM, mechanisms,
    purviews and system cuts). Shared by ``compute_IIM`` and the Hunter path.

    Subsystem choice is explicit and recorded: ``node_selection`` ('variance' =
    highest temporal variance with index tie-break, 'index' = first nodes in
    input order) or ``node_indices`` (explicit list). The state budget
    ``bins**nodes <= max_state_space`` is enforced by ``state_budget_policy``
    ('reduce_bins_first' [legacy], 'reduce_nodes_first', or 'error' which returns
    an undefined problem); every adjustment is logged and returned in
    ``budget_adjustments``.
    """
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    if partition_mode not in {"all", "balanced"}:
        raise ValueError("partition_mode must be 'all' or 'balanced'")
    if tpm_estimator not in IIM_TPM_ESTIMATORS:
        raise ValueError(f"tpm_estimator must be one of {IIM_TPM_ESTIMATORS}")
    if node_selection not in IIM_NODE_SELECTION_RULES:
        raise ValueError(f"node_selection must be one of {IIM_NODE_SELECTION_RULES}")
    if state_budget_policy not in IIM_STATE_BUDGET_POLICIES:
        raise ValueError(
            f"state_budget_policy must be one of {IIM_STATE_BUDGET_POLICIES}"
        )
    if bins < 2:
        raise ValueError("bins must be >= 2")
    if lag_trs < 1:
        raise ValueError("lag_trs must be >= 1")
    if n_parts is not None and int(n_parts) < 1:
        raise ValueError("n_parts must be >= 1 or None for exhaustive search")
    if max_nodes is not None and int(max_nodes) < 2:
        raise ValueError("max_nodes must be >= 2 or None")
    if max_mechanism_size is not None and int(max_mechanism_size) < 1:
        raise ValueError("max_mechanism_size must be >= 1 or None")
    if max_purview_size is not None and int(max_purview_size) < 1:
        raise ValueError("max_purview_size must be >= 1 or None")
    if tpm_alpha <= 0:
        raise ValueError("tpm_alpha must be > 0")
    if max_state_space < 16:
        raise ValueError("max_state_space must be >= 16")

    n_regions, n_time = ts.shape
    if n_regions < 2 or n_time - int(lag_trs) < 1:
        return {
            "defined": False,
            "undefined_reason": "insufficient_shape",
            "n_regions_input": int(n_regions),
            "n_time_input": int(n_time),
        }

    if isinstance(rng, np.random.RandomState):
        rand_state = rng
    else:
        rand_state = np.random.RandomState(rng)

    label = str(log_label) if log_label else "IIM"
    n_candidates = (
        int(n_regions) if node_indices is None else int(np.asarray(node_indices).size)
    )
    use_nodes = n_candidates if max_nodes is None else min(int(max_nodes), n_candidates)
    selected, selection_info = _iim_select_nodes_with_info(
        ts, use_nodes, rule=node_selection, node_indices=node_indices
    )
    eff_bins = int(bins)
    n_sel = int(selected.size)
    budget_adjustments = []

    def _over_budget():
        return (int(eff_bins) ** int(n_sel)) > int(max_state_space)

    def _budget_note(what, old, new):
        return (
            f"{what} {old}->{new} "
            f"({eff_bins}^{n_sel} > max_state_space={int(max_state_space)})"
        )

    if n_sel >= 2 and _over_budget() and state_budget_policy == "error":
        log.warning(
            "[%s] IIM state budget exceeded: bins=%d nodes=%d "
            "(%d states > max_state_space=%d); "
            "state_budget_policy='error' -> undefined.",
            label,
            eff_bins,
            n_sel,
            int(eff_bins) ** int(n_sel),
            int(max_state_space),
        )
        return {
            "defined": False,
            "undefined_reason": "state_space_too_large",
            "n_regions_input": int(n_regions),
            "n_time_input": int(n_time),
            "n_nodes_used": int(n_sel),
            "bins_used": int(eff_bins),
            "bins_requested": int(bins),
            "state_budget_policy": str(state_budget_policy),
        }
    while n_sel >= 2 and _over_budget():
        reduce_bins = eff_bins > 2 and (
            state_budget_policy == "reduce_bins_first" or n_sel <= 2
        )
        if reduce_bins:
            budget_adjustments.append(_budget_note("bins", eff_bins, eff_bins - 1))
            eff_bins -= 1
            continue
        budget_adjustments.append(_budget_note("nodes", n_sel, n_sel - 1))
        n_sel -= 1
        selected, selection_info = _iim_select_nodes_with_info(
            ts, n_sel, rule=node_selection, node_indices=node_indices
        )
    if budget_adjustments:
        log.warning(
            "[%s] IIM state budget (%s): requested bins=%d nodes=%d -> "
            "using bins=%d nodes=%d; %s",
            label,
            state_budget_policy,
            int(bins),
            int(use_nodes),
            int(eff_bins),
            int(n_sel),
            "; ".join(budget_adjustments),
        )
    if n_sel < 2:
        return {
            "defined": False,
            "undefined_reason": "state_space_too_large",
            "n_regions_input": int(n_regions),
            "n_time_input": int(n_time),
            "n_nodes_used": int(n_sel),
            "bins_used": int(eff_bins),
            "bins_requested": int(bins),
            "budget_adjustments": list(budget_adjustments),
        }
    ts_sel = np.asarray(ts[selected, :], dtype=float)
    if selection_info.get("degenerate_ranking", False):
        log.warning(
            "[%s] IIM node selection rule '%s' is degenerate (tied variances, e.g. "
            "standardised input); selected lowest-index nodes %s. Declare "
            "node_selection='index' or pass node_indices to make the subsystem "
            "choice explicit.",
            label,
            selection_info.get("rule"),
            selected.tolist(),
        )
    else:
        log.info(
            "[%s] IIM subsystem: rule=%s nodes=%s bins=%d",
            label,
            selection_info.get("rule"),
            selected.tolist(),
            int(eff_bins),
        )

    mech_size_eff = (
        int(n_sel)
        if max_mechanism_size is None
        else int(min(int(max_mechanism_size), int(n_sel)))
    )
    purv_size_eff = (
        int(n_sel)
        if max_purview_size is None
        else int(min(int(max_purview_size), int(n_sel)))
    )
    all_nodes = tuple(range(int(n_sel)))
    mechanisms_all = tuple(_iim_enumerate_subsets(all_nodes, mech_size_eff))
    purviews_all = tuple(_iim_enumerate_subsets(all_nodes, purv_size_eff))
    if not mechanisms_all or not purviews_all:
        return {
            "defined": False,
            "undefined_reason": "no_valid_mechanisms_or_purviews",
            "n_regions_input": int(n_regions),
            "n_time_input": int(n_time),
            "n_nodes_used": int(n_sel),
            "bins_used": int(eff_bins),
            "max_mechanism_size_used": int(mech_size_eff),
            "max_purview_size_used": int(purv_size_eff),
        }

    cuts = _iim_all_system_cuts(n_sel, partition_mode)
    if not cuts:
        return {
            "defined": False,
            "undefined_reason": "no_valid_cuts",
            "n_regions_input": int(n_regions),
            "n_time_input": int(n_time),
            "n_nodes_used": int(n_sel),
            "bins_used": int(eff_bins),
            "max_mechanism_size_used": int(mech_size_eff),
            "max_purview_size_used": int(purv_size_eff),
        }
    if n_parts is None or int(n_parts) >= len(cuts):
        cuts_eval = tuple(cuts)
    else:
        idx = rand_state.choice(len(cuts), size=int(n_parts), replace=False)
        cuts_eval = tuple(cuts[int(i)] for i in idx)

    disc = _iim_discretize_per_node(ts_sel, eff_bins)
    curr_obs, tpm_full, states_full = _iim_build_states_and_tpm(
        disc,
        eff_bins,
        int(lag_trs),
        float(tpm_alpha),
        hardware_backend=hardware_backend,
        estimator=str(tpm_estimator),
    )
    n_states = int(tpm_full.shape[0])
    n_rows_observed = int(
        np.unique(_iim_subset_key_matrix(curr_obs, tuple(range(n_sel)), eff_bins)).size
    )

    return {
        "defined": True,
        "undefined_reason": None,
        "n_regions_input": int(n_regions),
        "n_time_input": int(n_time),
        "selected_nodes": tuple(int(x) for x in selected.tolist()),
        "node_selection_rule": str(selection_info.get("rule")),
        "node_selection_degenerate": bool(
            selection_info.get("degenerate_ranking", False)
        ),
        "bins_requested": int(bins),
        "state_budget_policy": str(state_budget_policy),
        "budget_adjustments": list(budget_adjustments),
        "tpm_estimator": str(tpm_estimator),
        "n_states": int(n_states),
        "n_transitions": int(curr_obs.shape[0]),
        "n_states_observed": int(n_rows_observed),
        "ts_selected": ts_sel,
        "disc": disc,
        "curr_obs": curr_obs,
        "tpm_full": tpm_full,
        "states_full": states_full,
        "bins_used": int(eff_bins),
        "lag_trs": int(lag_trs),
        "n_nodes_used": int(n_sel),
        "max_nodes_requested": (None if max_nodes is None else int(max_nodes)),
        "max_mechanism_size_used": int(mech_size_eff),
        "max_purview_size_used": int(purv_size_eff),
        "n_parts_requested": (None if n_parts is None else int(n_parts)),
        "mechanisms_all": mechanisms_all,
        "purviews_all": purviews_all,
        "cuts_eval": cuts_eval,
        "cuts_payload": tuple((list(A), list(B)) for A, B in cuts_eval),
        "partition_mode": str(partition_mode),
        "tpm_alpha": float(tpm_alpha),
        "max_state_space": int(max_state_space),
    }


def _iim_phase1_chunk_contribution(
    mechanisms_chunk,
    purviews,
    base,
    tpm,
    curr_obs,
    states_full,
    static_cache=None,
    obs_state_cache=None,
    kernel_cache=None,
    cut_mask_a=None,
    use_induced_partition_cache=False,
    kernel_cache_lookup_only=False,
):
    n_states = int(states_full.shape[0])
    if static_cache is None:
        static_cache = {}
    static_sig = (int(base), int(states_full.shape[0]), int(states_full.shape[1]))
    if static_cache.get("_sig") != static_sig:
        static_cache.clear()
        static_cache["_sig"] = static_sig
    subset_cache = static_cache.setdefault("subset_cache", {})
    map_cache = static_cache.setdefault("map_cache", {})
    bip_cache = static_cache.setdefault("bip_cache", {})
    subset_mask_cache = static_cache.setdefault("subset_mask_cache", {})

    if obs_state_cache is None:
        obs_state_cache = {}
    obs_sig = (
        int(base),
        int(curr_obs.shape[0]),
        int(curr_obs.shape[1]),
    )
    if obs_state_cache.get("_sig") != obs_sig:
        obs_state_cache.clear()
        obs_state_cache["_sig"] = obs_sig

    # tpm-specific caches: valid only for this call
    effect_cache = {}
    cause_cache = {}

    def _subset_mask(subset):
        subset = tuple(sorted(subset))
        if subset in subset_mask_cache:
            return int(subset_mask_cache[subset])
        mask = 0
        for nn in subset:
            mask |= (1 << int(nn))
        subset_mask_cache[subset] = int(mask)
        return int(mask)

    def _induced_partition_key(m_mask, z_mask):
        if cut_mask_a is None:
            return 0
        u_mask = int(m_mask) | int(z_mask)
        part_a = int(u_mask & int(cut_mask_a))
        part_b = int(u_mask ^ part_a)
        if part_a == 0 or part_b == 0:
            return 0
        return int(part_a if part_a < part_b else part_b)

    def _get_subset_cache(subset):
        subset = tuple(sorted(subset))
        if subset in subset_cache:
            return subset_cache[subset]
        keys = _iim_subset_key_matrix(states_full, subset, base)
        n_keys = int(base ** len(subset))
        rows_by_key = [np.where(keys == k)[0] for k in range(n_keys)]
        rec = {
            "subset": subset,
            "keys": keys,
            "n_keys": n_keys,
            "rows_by_key": rows_by_key,
        }
        subset_cache[subset] = rec
        return rec

    def _partition_maps(Z, ZA, ZB):
        key = (tuple(Z), tuple(ZA), tuple(ZB))
        if key in map_cache:
            return map_cache[key]
        Z = tuple(Z)
        ZA = tuple(ZA)
        ZB = tuple(ZB)
        posA = [Z.index(x) for x in ZA]
        posB = [Z.index(x) for x in ZB]
        if NUMBA_AVAILABLE:
            posA_arr = np.asarray(posA, dtype=np.int64)
            posB_arr = np.asarray(posB, dtype=np.int64)
            mapA, mapB = _build_partition_maps_numba(
                int(base),
                int(len(Z)),
                posA_arr,
                posB_arr,
            )
        else:
            nZ = int(base ** len(Z))
            mapA = np.zeros(nZ, dtype=np.int64)
            mapB = np.zeros(nZ, dtype=np.int64)
            for kz in range(nZ):
                zv = _iim_decode_key(kz, len(Z), base)
                mapA[kz] = _iim_encode_vals([zv[p] for p in posA], base)
                mapB[kz] = _iim_encode_vals([zv[p] for p in posB], base)
        map_cache[key] = (mapA, mapB)
        return mapA, mapB

    def _effect(M, m_key, Z):
        M = tuple(sorted(M))
        Z = tuple(sorted(Z))
        key = (M, int(m_key), Z)
        if key in effect_cache:
            return effect_cache[key]
        cM = _get_subset_cache(M)
        cZ = _get_subset_cache(Z)
        rows = cM["rows_by_key"][int(m_key)]
        if rows.size == 0:
            p = np.ones(cZ["n_keys"], dtype=float) / float(cZ["n_keys"])
        else:
            d_next = tpm[rows, :].mean(axis=0)
            p = _iim_marginalize(d_next, cZ["keys"], cZ["n_keys"])
        effect_cache[key] = p
        return p

    def _cause(M, m_key, Z):
        M = tuple(sorted(M))
        Z = tuple(sorted(Z))
        key = (M, int(m_key), Z)
        if key in cause_cache:
            return cause_cache[key]
        cM = _get_subset_cache(M)
        cZ = _get_subset_cache(Z)
        cols = cM["rows_by_key"][int(m_key)]
        if cols.size == 0:
            p = np.ones(cZ["n_keys"], dtype=float) / float(cZ["n_keys"])
        else:
            likelihood = tpm[:, cols].sum(axis=1)
            s = float(np.sum(likelihood))
            if s <= 0:
                post_prev = np.ones(n_states, dtype=float) / float(n_states)
            else:
                post_prev = likelihood / s
            p = _iim_marginalize(post_prev, cZ["keys"], cZ["n_keys"])
        cause_cache[key] = p
        return p

    def _min_partition_divergence(M, m_key, Z, direction):
        M = tuple(sorted(M))
        Z = tuple(sorted(Z))
        if len(M) < 2 or len(Z) < 2:
            return 0.0
        if bool(use_induced_partition_cache) and (kernel_cache is not None):
            m_mask = _subset_mask(M)
            z_mask = _subset_mask(Z)
            pi_key = _induced_partition_key(m_mask, z_mask)
            d_key = 0 if direction == "effect" else 1
            cache_key = (int(d_key), int(m_mask), int(m_key), int(z_mask), int(pi_key))
            v_cached = kernel_cache.get(cache_key)
            if v_cached is not None:
                return float(v_cached)
            if bool(kernel_cache_lookup_only):
                raise _IIMKernelCacheMissError(
                    f"Missing kernel cache key: d={d_key}, m_mask={m_mask}, m_key={int(m_key)}, z_mask={z_mask}, pi={pi_key}"
                )
        if M not in bip_cache:
            bip_cache[M] = _iim_enumerate_bipartitions(M)
        if Z not in bip_cache:
            bip_cache[Z] = _iim_enumerate_bipartitions(Z)
        m_parts = bip_cache[M]
        z_parts = bip_cache[Z]
        if not m_parts or not z_parts:
            return 0.0
        m_vals = _iim_decode_key(int(m_key), len(M), base)
        posM = {node: i for i, node in enumerate(M)}
        p_full = _effect(M, m_key, Z) if direction == "effect" else _cause(M, m_key, Z)
        best = np.inf
        nZ = int(base ** len(Z))
        for MA, MB in m_parts:
            keyA = _iim_encode_vals([m_vals[posM[nn]] for nn in MA], base)
            keyB = _iim_encode_vals([m_vals[posM[nn]] for nn in MB], base)
            for ZA, ZB in z_parts:
                # Bipartitions are enumerated as unordered pairs, so both pairings
                # (MA->ZA, MB->ZB) and (MA->ZB, MB->ZA) must be searched; otherwise
                # the MIP depends on node labelling and phi is overestimated.
                for Z1, Z2 in ((ZA, ZB), (ZB, ZA)):
                    pA = (
                        _effect(MA, keyA, Z1)
                        if direction == "effect"
                        else _cause(MA, keyA, Z1)
                    )
                    pB = (
                        _effect(MB, keyB, Z2)
                        if direction == "effect"
                        else _cause(MB, keyB, Z2)
                    )
                    mapA, mapB = _partition_maps(Z, Z1, Z2)
                    if NUMBA_AVAILABLE:
                        q = _compose_product_distribution_numba(pA, pB, mapA, mapB)
                        d = float(_jsd_numba(p_full, q))
                    else:
                        q = pA[mapA] * pB[mapB]
                        q_sum = float(np.sum(q))
                        if q_sum <= 0:
                            q = np.ones(nZ, dtype=float) / float(nZ)
                        else:
                            q = q / q_sum
                        p_full_n = p_full / max(float(np.sum(p_full)), 1e-12)
                        d = float(
                            0.5 * entropy(p_full_n, 0.5 * (p_full_n + q), base=2)
                            + 0.5 * entropy(q, 0.5 * (p_full_n + q), base=2)
                        )
                    if d < best:
                        best = d
        out = float(best) if np.isfinite(best) else 0.0
        if bool(use_induced_partition_cache) and (kernel_cache is not None):
            kernel_cache.set(cache_key, out)
        return out

    def _get_mechanism_state_freq(M):
        M = tuple(sorted(M))
        cached = obs_state_cache.get(M)
        if cached is not None:
            return cached
        obs_keys = _iim_subset_key_matrix(curr_obs, M, base)
        if obs_keys.size == 0:
            uk = np.asarray([], dtype=np.int64)
            wt = np.asarray([], dtype=float)
        else:
            uk, cnt = np.unique(obs_keys, return_counts=True)
            wt = cnt.astype(float) / float(cnt.sum())
            uk = uk.astype(np.int64, copy=False)
            wt = wt.astype(float, copy=False)
        obs_state_cache[M] = (uk, wt)
        return uk, wt

    psi_terms = []
    for M in mechanisms_chunk:
        M = tuple(sorted(M))
        uk, wt = _get_mechanism_state_freq(M)
        if uk.size == 0:
            continue
        phi_M_terms = []
        for m_key, w_m in zip(uk, wt):
            phi_e = 0.0
            phi_c = 0.0
            for Z in purviews:
                d_e = _min_partition_divergence(M, int(m_key), Z, "effect")
                d_c = _min_partition_divergence(M, int(m_key), Z, "cause")
                if d_e > phi_e:
                    phi_e = d_e
                if d_c > phi_c:
                    phi_c = d_c
            phi_state = min(phi_e, phi_c)
            phi_M_terms.append(float(w_m) * float(phi_state))
        phi_M = float(math.fsum(phi_M_terms)) if phi_M_terms else 0.0
        w_size = 1.0 / float(len(M))
        psi_terms.append(float(w_size) * phi_M)
    return float(math.fsum(psi_terms)) if psi_terms else 0.0


def _iim_phase1_worker_cleanup():
    global _IIM_PHASE1_CTX
    if not isinstance(_IIM_PHASE1_CTX, dict):
        _IIM_PHASE1_CTX = None
        return
    kernel_cache = _IIM_PHASE1_CTX.get("kernel_cache")
    if kernel_cache is not None:
        try:
            kernel_cache.close()
        except Exception:
            pass
    shms = _IIM_PHASE1_CTX.get("shms", [])
    for shm in shms:
        try:
            shm.close()
        except Exception:
            pass
    tpm_shm = _IIM_PHASE1_CTX.get("tpm_shm")
    if tpm_shm is not None:
        try:
            tpm_shm.close()
        except Exception:
            pass
    _IIM_PHASE1_CTX = None


def _iim_open_readonly_array_spec(spec):
    mode = str(spec.get("mode", "shared_memory"))
    if mode == "shared_memory":
        shm = shared_memory.SharedMemory(name=str(spec["name"]))
        arr = np.ndarray(
            tuple(spec["shape"]),
            dtype=np.dtype(spec["dtype"]),
            buffer=shm.buf,
        )
        arr.flags.writeable = False
        return arr, shm
    if mode == "memmap":
        arr = np.load(str(spec["path"]), mmap_mode="r")
        arr.flags.writeable = False
        return arr, None
    raise RuntimeError(f"Unsupported IIM array mode: {mode}")


def _iim_phase_worker_init_static(spec_curr, spec_states, base, purviews):
    global _IIM_PHASE1_CTX
    _iim_phase1_worker_cleanup()
    curr_obs, shm_curr = _iim_open_readonly_array_spec(spec_curr)
    states_full, shm_states = _iim_open_readonly_array_spec(spec_states)
    shms = []
    if shm_curr is not None:
        shms.append(shm_curr)
    if shm_states is not None:
        shms.append(shm_states)
    _IIM_PHASE1_CTX = {
        "base": int(base),
        "purviews": tuple(tuple(z) for z in purviews),
        "curr_obs": curr_obs,
        "states_full": states_full,
        "shms": shms,
        "tpm": None,
        "tpm_shm": None,
        "tpm_token": None,
        # Reused across chunks/cuts for this worker.
        "static_cache": {},
        "obs_state_cache": {},
        "kernel_cache": None,
        "kernel_cache_token": None,
    }
    atexit.register(_iim_phase1_worker_cleanup)


def _iim_phase_worker_bind_tpm(spec_tpm):
    if not isinstance(_IIM_PHASE1_CTX, dict):
        raise RuntimeError("IIM worker context not initialized.")
    mode = str(spec_tpm.get("mode", "shared_memory"))
    token = (
        mode,
        str(spec_tpm.get("name", "")),
        str(spec_tpm.get("path", "")),
    )
    if _IIM_PHASE1_CTX.get("tpm_token") == token and _IIM_PHASE1_CTX.get("tpm") is not None:
        return

    old_shm = _IIM_PHASE1_CTX.get("tpm_shm")
    if old_shm is not None:
        try:
            old_shm.close()
        except Exception:
            pass
    _IIM_PHASE1_CTX["tpm_shm"] = None

    tpm, shm = _iim_open_readonly_array_spec(spec_tpm)
    _IIM_PHASE1_CTX["tpm"] = tpm
    _IIM_PHASE1_CTX["tpm_shm"] = shm
    _IIM_PHASE1_CTX["tpm_token"] = token


def _iim_phase_worker_bind_kernel_cache(cache_spec):
    if not isinstance(_IIM_PHASE1_CTX, dict):
        raise RuntimeError("IIM worker context not initialized.")

    if not isinstance(cache_spec, dict) or not bool(cache_spec.get("enabled", False)):
        old_cache = _IIM_PHASE1_CTX.get("kernel_cache")
        if old_cache is not None:
            try:
                old_cache.close()
            except Exception:
                pass
        _IIM_PHASE1_CTX["kernel_cache"] = None
        _IIM_PHASE1_CTX["kernel_cache_token"] = None
        return

    path = str(cache_spec.get("path", "")).strip()
    if not path:
        raise RuntimeError("IIM worker kernel cache path is empty.")
    mem_entries = int(cache_spec.get("memory_entries", 50_000))
    flush_batch = int(cache_spec.get("flush_batch", 5_000))
    token = (path, int(mem_entries), int(flush_batch))

    if (
        _IIM_PHASE1_CTX.get("kernel_cache_token") == token
        and _IIM_PHASE1_CTX.get("kernel_cache") is not None
    ):
        return

    old_cache = _IIM_PHASE1_CTX.get("kernel_cache")
    if old_cache is not None:
        try:
            old_cache.close()
        except Exception:
            pass

    cache = _IIMDiskKernelCache(
        path,
        signature=None,
        memory_entries=int(mem_entries),
        flush_batch=int(flush_batch),
    )
    _IIM_PHASE1_CTX["kernel_cache"] = cache
    _IIM_PHASE1_CTX["kernel_cache_token"] = token


def _iim_phase_worker_run_chunk_for_tpm(
    spec_tpm,
    mechanisms_chunk,
    cache_spec=None,
    cut_mask_a=None,
    use_induced_partition_cache=False,
    kernel_cache_lookup_only=False,
):
    if not isinstance(_IIM_PHASE1_CTX, dict):
        raise RuntimeError("IIM worker context not initialized.")
    _iim_phase_worker_bind_tpm(spec_tpm)
    _iim_phase_worker_bind_kernel_cache(cache_spec)
    mechanisms = tuple(tuple(m) for m in mechanisms_chunk)
    kernel_cache = _IIM_PHASE1_CTX.get("kernel_cache")
    psi_chunk = _iim_phase1_chunk_contribution(
        mechanisms,
        _IIM_PHASE1_CTX["purviews"],
        int(_IIM_PHASE1_CTX["base"]),
        _IIM_PHASE1_CTX["tpm"],
        _IIM_PHASE1_CTX["curr_obs"],
        _IIM_PHASE1_CTX["states_full"],
        static_cache=_IIM_PHASE1_CTX.get("static_cache"),
        obs_state_cache=_IIM_PHASE1_CTX.get("obs_state_cache"),
        kernel_cache=kernel_cache,
        cut_mask_a=cut_mask_a,
        use_induced_partition_cache=bool(use_induced_partition_cache) and (kernel_cache is not None),
        kernel_cache_lookup_only=bool(kernel_cache_lookup_only),
    )
    if kernel_cache is not None and not bool(kernel_cache_lookup_only):
        # Pool workers exit without running atexit handlers, so pending writes
        # must be flushed per chunk or they are silently lost.
        kernel_cache.flush()
    return float(psi_chunk), int(len(mechanisms))


def _safe_mutual_info_score(labels_a, labels_b):
    """
    Compute MI while suppressing sklearn's high-cardinality class warning.
    For discretized continuous time-series this warning is expected and not
    informative for our use case.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50% of the number of samples.*",
            category=UserWarning,
        )
        return mutual_info_score(labels_a, labels_b)


def _coerce_ram_event_bundle(stimulus_onsets):
    """
    Normalize RAM event input into a structured bundle.

    Supported inputs:
      - list/array-like of onset seconds (legacy)
      - dict with keys:
          onsets (or stimulus_onsets/stim_onsets/events)
          goal_onsets
          feedback_onsets
          feedback_values
    """
    bundle = {
        "onsets": [],
        "goal_onsets": [],
        "feedback_onsets": [],
        "feedback_values": None,
    }
    if stimulus_onsets is None:
        return bundle
    if isinstance(stimulus_onsets, dict):
        onsets = stimulus_onsets.get("onsets")
        if onsets is None:
            for k in ("stimulus_onsets", "stim_onsets", "events"):
                if k in stimulus_onsets:
                    onsets = stimulus_onsets.get(k)
                    break
        bundle["onsets"] = [] if onsets is None else onsets
        bundle["goal_onsets"] = stimulus_onsets.get("goal_onsets", [])
        bundle["feedback_onsets"] = stimulus_onsets.get("feedback_onsets", [])
        bundle["feedback_values"] = stimulus_onsets.get("feedback_values")
        return bundle

    bundle["onsets"] = stimulus_onsets
    return bundle


def _coerce_numeric_feedback(values):
    """
    Convert feedback labels/values to numeric.

    If numeric conversion fails for all entries, factor-encode categorical
    labels deterministically.
    """
    if values is None:
        return np.empty(0, dtype=float)

    raw = np.asarray(list(values), dtype=object).reshape(-1)
    if raw.size == 0:
        return np.empty(0, dtype=float)

    out = np.full(raw.shape, np.nan, dtype=float)
    for i, v in enumerate(raw):
        try:
            out[i] = float(v)
        except Exception:
            out[i] = np.nan

    if np.isfinite(out).any():
        return out

    labels = np.asarray([str(v) for v in raw], dtype=str)
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(float)


def _sanitize_onset_seconds(onsets, tr, n_tp):
    """
    Keep finite in-range onsets, map to nearest sample index, and deduplicate
    by index while preserving temporal order.
    """
    arr = np.asarray(onsets if onsets is not None else [], dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr, np.empty(0, dtype=np.int64)

    tr = float(tr)
    max_t = max(0.0, (int(n_tp) - 1) * tr)
    arr = arr[(arr >= 0.0) & (arr <= (max_t + 0.5 * tr))]
    if arr.size == 0:
        return arr, np.empty(0, dtype=np.int64)

    idx = np.rint(arr / tr).astype(np.int64)
    valid = (idx >= 0) & (idx < int(n_tp))
    arr = arr[valid]
    idx = idx[valid]
    if idx.size == 0:
        return arr, np.empty(0, dtype=np.int64)

    order = np.argsort(idx, kind="mergesort")
    idx = idx[order]
    arr = arr[order]
    keep = np.ones(idx.shape[0], dtype=bool)
    if idx.size > 1:
        keep[1:] = idx[1:] != idx[:-1]
    return arr[keep], idx[keep]


def _in_range_onset_seconds(onsets, tr, n_tp):
    """
    Finite onsets inside the run, sorted, without per-sample deduplication.

    Used for the magnitude regressor, where two events falling in the same
    sample are still two responses.
    """
    arr = np.asarray(onsets if onsets is not None else [], dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    max_t = max(0.0, (int(n_tp) - 1) * float(tr))
    arr = arr[(arr >= 0.0) & (arr <= (max_t + 0.5 * float(tr)))]
    return np.sort(arr)


def _sanitize_onsets_with_values(onsets, values, tr, n_tp):
    """
    Sanitize event onsets while keeping per-event values aligned.

    Same rules as :func:`_sanitize_onset_seconds` (finite, in range, sorted,
    one event per sample), but each dropped onset drops its own value, events
    with a non-finite value are dropped, and events mapping to the same sample
    are merged (onsets and values averaged). Returns ``(seconds, idx, values)``.
    """
    on = np.asarray(onsets if onsets is not None else [], dtype=float).reshape(-1)
    val = np.asarray(values, dtype=float).reshape(-1)
    if on.size != val.size:
        raise ValueError("onsets and values must have the same length")
    empty = (
        np.empty(0, dtype=float),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=float),
    )
    keep = np.isfinite(on) & np.isfinite(val)
    on = on[keep]
    val = val[keep]
    if on.size == 0:
        return empty
    tr = float(tr)
    max_t = max(0.0, (int(n_tp) - 1) * tr)
    keep = (on >= 0.0) & (on <= (max_t + 0.5 * tr))
    on = on[keep]
    val = val[keep]
    idx = np.rint(on / tr).astype(np.int64)
    keep = (idx >= 0) & (idx < int(n_tp))
    idx = idx[keep]
    on = on[keep]
    val = val[keep]
    if idx.size == 0:
        return empty
    uniq, inv = np.unique(idx, return_inverse=True)
    cnts = np.bincount(inv, minlength=uniq.size).astype(float)
    sec = np.bincount(inv, weights=on, minlength=uniq.size) / cnts
    vals = np.bincount(inv, weights=val, minlength=uniq.size) / cnts
    return sec, uniq.astype(np.int64), vals


def _window_mean_vectors(ts, event_idx, pre_samples, post_samples, post_lag_samples=0):
    """
    Compute event-locked mean vectors over pre/post windows.

    The pre window is ``[e - pre_samples, e)`` (strictly before onset) and the
    post window is ``[e + post_lag_samples, e + post_lag_samples +
    post_samples)``; the lag accounts for response delays (e.g. hemodynamic).

    Returns
    -------
    pre_vecs, post_vecs, used_idx
      pre_vecs/post_vecs: arrays of shape (n_events, n_regions)
      used_idx: event indices (sample domain) after edge filtering
    """
    n_regions, n_tp = ts.shape
    pre_samples = max(0, int(pre_samples))
    post_samples = max(1, int(post_samples))
    lag = max(0, int(post_lag_samples))
    event_idx = np.asarray(event_idx, dtype=np.int64).reshape(-1)

    if event_idx.size == 0:
        empty = np.empty((0, n_regions), dtype=float)
        return empty, empty, np.empty(0, dtype=np.int64)

    valid = (event_idx - pre_samples >= 0) & (event_idx + lag + post_samples <= n_tp)
    idx = event_idx[valid]
    if idx.size == 0:
        empty = np.empty((0, n_regions), dtype=float)
        return empty, empty, np.empty(0, dtype=np.int64)

    post_offsets = np.arange(lag, lag + post_samples, dtype=np.int64)
    post_ix = idx[:, None] + post_offsets[None, :]
    post_vecs = ts[:, post_ix].mean(axis=2).T

    if pre_samples > 0:
        pre_offsets = np.arange(-pre_samples, 0, dtype=np.int64)
        pre_ix = idx[:, None] + pre_offsets[None, :]
        pre_vecs = ts[:, pre_ix].mean(axis=2).T
    else:
        pre_vecs = np.zeros((idx.size, n_regions), dtype=float)

    return pre_vecs, post_vecs, idx


def _safe_abs_corr(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = int(min(a.size, b.size))
    if n < 3:
        return np.nan
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return np.nan
    r = np.corrcoef(aa, bb)[0, 1]
    if not np.isfinite(r):
        return np.nan
    return float(np.clip(abs(r), 0.0, 1.0))


def _sample_reliability(n_samples, tau=4.0):
    n = max(0, int(n_samples))
    tau = max(float(tau), 1e-6)
    return float(1.0 - np.exp(-float(n) / tau))


def _prediction_error_signal(values):
    """
    Simple online prediction-error proxy:
      delta_t = x_t - mean(x_<t)
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        return np.empty(0, dtype=float)

    pred = np.empty_like(x)
    pred[0] = x[0]
    if x.size > 1:
        csum = np.cumsum(x[:-1], dtype=float)
        pred[1:] = csum / np.arange(1, x.size, dtype=float)
    return x - pred


# RAM event-count contract (mirrored by event_parsing.RAM_MIN_* for readiness).
_RAM_MIN_GOAL_PAIRS = 6
_RAM_MIN_FEEDBACK_EVENTS = 3
_RAM_MIN_ADAPTIVE_UPDATES = 3

# Canonical (Glover 1999) HRF with the parameters of nilearn's ``glover_hrf``:
# difference of gamma densities (delay 6 s, undershoot 12 s, dispersion 0.9 s,
# undershoot ratio 0.48), truncated at 32 s. Evaluated analytically on the
# true time axis (seconds) instead of on an oversampled grid.
_HRF_TIME_LENGTH_SEC = 32.0
_HRF_DELAY = 6.0
_HRF_UNDERSHOOT = 12.0
_HRF_DISPERSION = 0.9
_HRF_U_DISPERSION = 0.9
_HRF_RATIO = 0.48


def _gamma_pdf(t, shape, scale):
    log_norm = math.lgamma(shape) + shape * math.log(scale)
    return np.exp((shape - 1.0) * np.log(t) - t / scale - log_norm)


def _glover_hrf_unnormalized(t):
    t = np.asarray(t, dtype=float)
    out = np.zeros(t.shape, dtype=float)
    m = (t > 0.0) & (t <= _HRF_TIME_LENGTH_SEC)
    if not np.any(m):
        return out
    tm = t[m]
    peak = _gamma_pdf(tm, _HRF_DELAY / _HRF_DISPERSION, _HRF_DISPERSION)
    under = _gamma_pdf(tm, _HRF_UNDERSHOOT / _HRF_U_DISPERSION, _HRF_U_DISPERSION)
    out[m] = peak - _HRF_RATIO * under
    return out


_HRF_PEAK_CACHE = {}


def canonical_hrf_peak():
    """Return ``(peak_time_seconds, peak_value)`` of the canonical HRF (1 ms grid)."""
    if "peak" not in _HRF_PEAK_CACHE:
        grid = np.arange(0.0, _HRF_TIME_LENGTH_SEC, 1e-3)
        h = _glover_hrf_unnormalized(grid)
        k = int(np.argmax(h))
        _HRF_PEAK_CACHE["peak"] = (float(grid[k]), float(h[k]))
    return _HRF_PEAK_CACHE["peak"]


def canonical_hrf(t):
    """Canonical HRF evaluated at times ``t`` (seconds), peak-normalised to 1."""
    _, peak_val = canonical_hrf_peak()
    return _glover_hrf_unnormalized(t) / peak_val


def _hrf_regressor(onsets_sec, tr, n_tp):
    """
    Stimulus regressor: sum of peak-normalised canonical HRFs at the exact
    onset times, sampled at the frame times ``k * tr``. A unit-amplitude
    response therefore has regression weight 1.
    """
    frame_times = np.arange(int(n_tp), dtype=float) * float(tr)
    reg = np.zeros(int(n_tp), dtype=float)
    for onset in np.asarray(onsets_sec, dtype=float).reshape(-1):
        reg += canonical_hrf(frame_times - float(onset))
    return reg


def _peak_is_interior(k, n):
    """
    True when the discrete maximum ``k`` of an ``n``-sample search curve is
    away from both ends (guard band of 5% of the range, at least one
    sample); otherwise the curve may be a truncated rising/decaying edge
    (e.g. the response to the next event) rather than a resolved peak.
    """
    edge = max(1, int(round(0.05 * (int(n) - 1))))
    return edge <= int(k) <= int(n) - 1 - edge


def _parabolic_peak_offset(y, k):
    """Sub-sample offset in [-0.5, 0.5] of the discrete maximum ``y[k]``."""
    if k <= 0 or k >= y.size - 1:
        return 0.0
    a, b, c = float(y[k - 1]), float(y[k]), float(y[k + 1])
    den = a - 2.0 * b + c
    if (not np.isfinite(den)) or den >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def _fir_latency_seconds(ts, stim_idx, tr, fir_window):
    """
    Time-to-peak (seconds) of the baseline-corrected event-related response.

    Epochs span ``[-fir_window, +fir_window]`` around each onset. The
    pre-onset half is the baseline; the latency is the post-onset lag that
    maximises the global field power (RMS across regions) of the event
    average, refined by parabolic interpolation. A maximum within 5% of
    either end of the post-onset search range means no peak was resolved
    (``NaN``).
    """
    half = int(round(float(fir_window) / float(tr)))
    if half < 2:
        return float("nan"), "fir_window_too_short"
    n_tp = int(ts.shape[1])
    idx = np.asarray(stim_idx, dtype=np.int64)
    idx = idx[(idx - half >= 0) & (idx + half < n_tp)]
    if idx.size == 0:
        return float("nan"), "no_events_with_full_fir_window"
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    avg = ts[:, idx[:, None] + offsets[None, :]].mean(axis=1)
    baseline = avg[:, :half].mean(axis=1, keepdims=True)
    post = avg[:, half:] - baseline
    gfp = np.sqrt(np.mean(post * post, axis=0))
    k = int(np.argmax(gfp))
    if not _peak_is_interior(k, gfp.size):
        return float("nan"), "latency_peak_at_search_boundary"
    return float((k + _parabolic_peak_offset(gfp, k)) * float(tr)), None


def _xcorr_latency_seconds(ts, stick, tr, maxlag):
    """
    Lag (seconds, >= 0) maximising the mean squared correlation between the
    stimulus stick function and the regional signals, refined by parabolic
    interpolation. Only non-negative lags are searched (a response follows
    its stimulus); a maximum at the ends of the search range is unresolved.
    """
    maxlag = int(maxlag)
    n_regions, n_tp = ts.shape
    stick = np.asarray(stick, dtype=float)
    if maxlag < 2:
        return float("nan"), "xcorr_maxlag_too_short"
    if np.std(stick) <= 1e-12:
        return float("nan"), "stimulus_regressor_constant"
    energy = []
    for lag in range(0, maxlag + 1):
        m = n_tp - lag
        if m < 3:
            break
        a = ts[:, lag:]
        b = stick[:m]
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean()
        sa = np.sqrt(np.sum(a * a, axis=1))
        sb = float(np.sqrt(np.sum(b * b)))
        if sb <= 1e-12:
            energy.append(0.0)
            continue
        r = (a @ b) / (np.maximum(sa, 1e-12) * sb)
        r[sa <= 1e-12] = 0.0
        energy.append(float(np.mean(r * r)))
    energy = np.asarray(energy, dtype=float)
    if energy.size < 3:
        return float("nan"), "xcorr_too_few_lags"
    k = int(np.argmax(energy))
    if not _peak_is_interior(k, energy.size):
        return float("nan"), "latency_peak_at_search_boundary"
    return float((k + _parabolic_peak_offset(energy, k)) * float(tr)), None


def _blocked_fold_ids(groups, n_folds):
    """
    Time-blocked cross-validation folds with about equal numbers of rows.

    ``groups`` are sorted, contiguous labels (rows that share a label, e.g.
    responses paired with the same goal event, stay in one fold). Returns
    ``(fold_ids, n_folds_used)``.
    """
    groups = np.asarray(groups).reshape(-1)
    n = groups.size
    uniq, start, counts = np.unique(groups, return_index=True, return_counts=True)
    order = np.argsort(start, kind="mergesort")
    uniq, start, counts = uniq[order], start[order], counts[order]
    k = int(max(2, min(int(n_folds), n // 3)))
    before = np.concatenate([[0], np.cumsum(counts)[:-1]])
    fold_of_group = np.minimum(k - 1, (before * k) // max(1, n))
    fold_ids = np.empty(n, dtype=np.int64)
    for g_start, g_count, f in zip(start, counts, fold_of_group):
        fold_ids[g_start:g_start + g_count] = f
    used = np.unique(fold_ids)
    remap = {int(f): i for i, f in enumerate(used)}
    return np.asarray([remap[int(f)] for f in fold_ids], dtype=np.int64), int(used.size)


def _ridge_whitened_basis(xc, ridge):
    """
    Ridge-whitening in the row space of centred training data ``xc``.

    Returns ``(v, d, z)``: an orthonormal row-space basis ``v`` (p x r), the
    whitening factors ``d = (ev + lam)^(-1/2)`` for the covariance eigenvalues
    ``ev``, and the whitened training scores ``z = xc @ v * d``. The ridge
    ``lam = ridge * mean(ev)`` is relative to the mean non-zero eigenvalue, so
    its effect does not depend on whether ``p`` exceeds the number of events.
    """
    n = int(xc.shape[0])
    if n < 2:
        return None
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    if s.size == 0 or not np.isfinite(s).all():
        return None
    tol = max(xc.shape) * np.finfo(float).eps * float(s[0])
    keep = s > max(tol, 1e-12)
    if not np.any(keep):
        return None
    s = s[keep]
    v = vt[keep].T
    ev = (s * s) / float(n - 1)
    lam = float(ridge) * float(np.mean(ev))
    d = 1.0 / np.sqrt(ev + lam)
    return v, d, (xc @ v) * d


def _cv_canonical_corr(x, y, fold_ids, n_folds, ridge, x_side=None):
    """
    Held-out correlation of the first ridge-CCA pair.

    For each fold, canonical weights are fitted on the other folds and applied
    to the held-out events; held-out projections are centred within fold (so
    slow drifts between folds cannot create correlation) and pooled.
    """
    if x_side is None:
        x_side = []
        for f in range(int(n_folds)):
            te = fold_ids == f
            trn = ~te
            mx = x[trn].mean(axis=0)
            basis = _ridge_whitened_basis(x[trn] - mx, ridge)
            if basis is None:
                return float("nan"), None
            vx, dx, zx = basis
            x_side.append((te, trn, zx, ((x[te] - mx) @ vx) * dx))
    u_parts = []
    v_parts = []
    for te, trn, zx, xt_w in x_side:
        my = y[trn].mean(axis=0)
        basis = _ridge_whitened_basis(y[trn] - my, ridge)
        if basis is None:
            return float("nan"), x_side
        vy, dy, zy = basis
        a, _, bt = np.linalg.svd(zx.T @ zy, full_matrices=False)
        u = xt_w @ a[:, 0]
        v = (((y[te] - my) @ vy) * dy) @ bt[0]
        u_parts.append(u - u.mean())
        v_parts.append(v - v.mean())
    u = np.concatenate(u_parts)
    v = np.concatenate(v_parts)
    su = float(np.sqrt(np.sum(u * u)))
    sv = float(np.sqrt(np.sum(v * v)))
    if su <= 1e-12 or sv <= 1e-12:
        return float("nan"), x_side
    return float(np.clip(np.dot(u, v) / (su * sv), -1.0, 1.0)), x_side


def _phase_randomized_surrogate(ts, rng):
    """
    Multivariate phase-randomised surrogate (Prichard & Theiler, 1994).

    The same random phases are applied to every region, which preserves each
    region's power spectrum and the cross-spectra (autocorrelation and
    inter-regional correlation) while destroying time-locking to events.
    """
    n_tp = int(ts.shape[1])
    spec = np.fft.rfft(ts, axis=1)
    phases = rng.uniform(0.0, 2.0 * np.pi, spec.shape[1])
    phases[0] = 0.0
    if n_tp % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(spec * np.exp(1j * phases)[None, :], n=n_tp, axis=1)


def _abs_corr_chance_corrected(a, b, n_null, rng):
    """
    |corr(a, b)| corrected for its positive chance level.

    The null permutes ``b`` (an exogenous signal such as feedback values).
    Returns ``(score, abs_r, null_mean)`` with
    ``score = [(|r| - m0) / (1 - m0)]_+``.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    abs_r = _safe_abs_corr(a, b)
    if (not np.isfinite(abs_r)) or a.size != b.size or not (
        np.isfinite(a).all() and np.isfinite(b).all()
    ):
        return float("nan"), float(abs_r), float("nan")
    if int(n_null) <= 0:
        return float(abs_r), float(abs_r), 0.0
    n = a.size
    az = (a - a.mean()) / a.std()
    bz = (b - b.mean()) / b.std()
    perms = np.argsort(rng.random_sample((int(n_null), n)), axis=1)
    r_null = np.abs(bz[perms] @ az) / float(n)
    m0 = float(np.mean(r_null))
    if m0 >= 1.0 - 1e-12:
        return float("nan"), float(abs_r), m0
    score = float(np.clip((abs_r - m0) / (1.0 - m0), 0.0, 1.0))
    return score, float(abs_r), m0


def _lsa_hrf_design(onsets_by_type, tr, n_tp):
    """
    Least-squares-all (beta-series) design for trial-wise HRF amplitudes.

    One peak-normalised canonical-HRF regressor per event (all event types
    jointly) plus an intercept, so overlapping hemodynamic responses of
    nearby events are separated by the model instead of leaking into each
    other's windows. Events whose modelled response peak lies outside the run
    (regressor maximum < 0.5) are dropped. Returns ``(pinv, keep_masks,
    slices, full_rank)`` with one boolean mask and one column slice per event
    type; ``full_rank`` is ``False`` when the trial-wise amplitudes are not
    identifiable (more events than samples, or coinciding events).
    """
    frame_times = np.arange(int(n_tp), dtype=float) * float(tr)
    cols = []
    keep_masks = []
    slices = []
    start = 0
    for onsets in onsets_by_type:
        onsets = np.asarray(onsets, dtype=float).reshape(-1)
        keep = np.zeros(onsets.size, dtype=bool)
        for i, onset in enumerate(onsets):
            reg = canonical_hrf(frame_times - onset)
            if float(np.max(reg)) >= 0.5:
                cols.append(reg)
                keep[i] = True
        n_keep = int(keep.sum())
        keep_masks.append(keep)
        slices.append(slice(start, start + n_keep))
        start += n_keep
    cols.append(np.ones(int(n_tp), dtype=float))
    design = np.column_stack(cols)
    full_rank = int(np.linalg.matrix_rank(design)) == int(design.shape[1])
    return np.linalg.pinv(design), keep_masks, slices, full_rank


def _quality_patterns(z, spec):
    """
    Trial-wise response patterns used by G, F and U for data ``z``.

    ``spec['mode'] == 'glm'``: rows of the least-squares-all betas.
    ``spec['mode'] == 'window'``: event-locked window means (post windows
    delayed by ``spec['lag']``). Event sets are fixed by ``spec`` so the same
    extraction can be re-run on surrogate data.
    """
    out = {}
    if spec["mode"] == "glm":
        beta = spec["pinv"] @ z.T
        s_stim, s_goal, s_fb = spec["slices"]
        out["stim"] = beta[s_stim]
        out["goal"] = beta[s_goal]
        # With the opt-in feedback proxy the feedback events are the stimulus
        # events; they are not modelled twice.
        fb_beta = beta[s_stim] if spec.get("fb_from_stim") else beta[s_fb]
        out["fb_strength"] = np.sqrt(np.sum(fb_beta ** 2, axis=1))
    else:
        lag = spec["lag"]
        _, out["stim"], _ = _window_mean_vectors(
            z, spec["stim_used"], 0, spec["resp"], lag
        )
        _, out["goal"], _ = _window_mean_vectors(
            z, spec["goal_used"], 0, spec["gobj"], lag
        )
        fb_pre, fb_post, _ = _window_mean_vectors(
            z, spec["fb_used"], spec["fb"], spec["fb"], lag
        )
        out["fb_strength"] = np.sqrt(np.sum((fb_post - fb_pre) ** 2, axis=1))
    if spec.get("proxy_idx") is not None:
        out["proxy_pre"], _, _ = _window_mean_vectors(
            z, spec["proxy_idx"], spec["gpre"], 1
        )
    return out


def _goal_alignment_component(
    ts_z,
    spec,
    patterns,
    ridge,
    cv_folds,
    n_null,
    rng,
):
    """
    Goal-alignment component G in [0,1].

    Each stimulus response is paired with the objective representation of its
    nearest preceding goal event (or, as an opt-in proxy, with the
    pre-stimulus state). Their canonical association is measured out of
    sample -- ridge-CCA fitted on time-blocked training folds, correlation of
    the held-out projections -- and chance-corrected against surrogates:
    phase-randomised copies of the data re-analysed through the identical
    event design, which keeps the dependence that the design and the noise
    autocorrelation alone induce between paired windows/betas. The result is
    attenuated by event-count reliability. (The in-sample canonical
    correlation is not used: with more regions than events it is ~1 for pure
    noise.)
    """
    info = {
        "goal_source": spec["goal_source"],
        "n_goal_response_pairs": 0,
        "goal_alignment_cv_corr": float("nan"),
        "goal_alignment_null_mean": float("nan"),
        "goal_alignment_p_null": float("nan"),
    }
    if spec["goal_source"] is None:
        return float("nan"), "missing_goal_events", info
    xi = spec["goal_pairs_x"]
    yi = spec["goal_pairs_y"]
    n_pairs = int(yi.size)
    info["n_goal_response_pairs"] = n_pairs
    if n_pairs < _RAM_MIN_GOAL_PAIRS:
        return float("nan"), "insufficient_goal_response_pairs", info
    fold_ids, k = _blocked_fold_ids(spec["goal_groups"], cv_folds)
    if k < 2:
        return float("nan"), "goal_alignment_undefined", info
    x_key = "proxy_pre" if spec["goal_source"] == "pre_stimulus_state_proxy" else "goal"

    def _score(pat):
        r, _ = _cv_canonical_corr(
            np.asarray(pat[x_key], dtype=float)[xi],
            np.asarray(pat["stim"], dtype=float)[yi],
            fold_ids,
            k,
            ridge,
        )
        return r

    r_obs = _score(patterns)
    info["goal_alignment_cv_corr"] = float(r_obs)
    if not np.isfinite(r_obs):
        return float("nan"), "goal_alignment_undefined", info
    null = []
    for _ in range(int(n_null)):
        r_s = _score(_quality_patterns(_phase_randomized_surrogate(ts_z, rng), spec))
        if np.isfinite(r_s):
            null.append(r_s)
    null = np.asarray(null, dtype=float)
    m0 = float(null.mean()) if null.size else 0.0
    info["goal_alignment_null_mean"] = m0
    if null.size:
        n_ge = float(np.sum(null >= r_obs))
        info["goal_alignment_p_null"] = (1.0 + n_ge) / (1.0 + null.size)
    if m0 >= 1.0 - 1e-12:
        return float("nan"), "goal_alignment_not_identifiable", info
    score = float(np.clip((r_obs - m0) / (1.0 - m0), 0.0, 1.0))
    rel = _sample_reliability(n_pairs, tau=4.0)
    return float(np.clip(score * rel, 0.0, 1.0)), None, info


def _feedback_integration_component(fb_strength, fb_signal, n_null, rng):
    """
    Feedback-integration component F in [0,1].

    Chance-corrected |corr| between the feedback-locked neural response
    strength and the feedback signal of the same events (aligned 1:1),
    attenuated by event-count reliability.
    """
    n = int(np.asarray(fb_strength).size)
    if n < _RAM_MIN_FEEDBACK_EVENTS:
        return float("nan"), n, "insufficient_feedback_events"
    score, _, _ = _abs_corr_chance_corrected(fb_strength, fb_signal, n_null, rng)
    if not np.isfinite(score):
        return float("nan"), n, "feedback_integration_undefined"
    rel = _sample_reliability(n, tau=4.0)
    return float(np.clip(score * rel, 0.0, 1.0)), n, None


def _adaptive_update_component(
    stim_response_vecs,
    stim_used_idx,
    feedback_idx,
    feedback_signal,
    n_null,
    rng,
):
    """
    Adaptive-update component U in [0,1].

    For consecutive stimulus responses k -> k+1, the update magnitude
    ``||r_{k+1} - r_k||`` is paired with the mean |feedback| of the feedback
    events in ``[s_k, s_{k+1})`` (the feedback that can inform the next
    response); updates without intervening feedback are skipped. U is the
    chance-corrected |corr| of the pairs times event-count reliability.
    """
    stim_response_vecs = np.asarray(stim_response_vecs, dtype=float)
    if stim_response_vecs.ndim != 2 or stim_response_vecs.shape[0] < 2:
        return float("nan"), 0, "insufficient_adaptive_updates"
    update_mag = np.sqrt(np.sum(np.diff(stim_response_vecs, axis=0) ** 2, axis=1))
    stim_used_idx = np.asarray(stim_used_idx, dtype=np.int64).reshape(-1)
    fb_idx = np.asarray(feedback_idx, dtype=np.int64).reshape(-1)
    drive_abs = np.abs(np.asarray(feedback_signal, dtype=float).reshape(-1))
    left = np.searchsorted(fb_idx, stim_used_idx[:-1], side="left")
    right = np.searchsorted(fb_idx, stim_used_idx[1:], side="left")
    counts = right - left
    csum = np.concatenate([[0.0], np.cumsum(drive_abs)])
    has = counts > 0
    n = int(has.sum())
    if n < _RAM_MIN_ADAPTIVE_UPDATES:
        return float("nan"), n, "insufficient_adaptive_updates"
    drive = (csum[right[has]] - csum[left[has]]) / counts[has]
    score, _, _ = _abs_corr_chance_corrected(update_mag[has], drive, n_null, rng)
    if not np.isfinite(score):
        return float("nan"), n, "adaptive_update_undefined"
    rel = _sample_reliability(n, tau=4.0)
    return float(np.clip(score * rel, 0.0, 1.0)), n, None


def compute_RAM(
    ts: np.ndarray,
    tr: float = 1.0,
    stimulus_onsets=None,
    epsilon: float = None,
    magnitude_scale: float = 0.5,
    response_model: str = "hrf",
    response_boxcar_width_sec: float = None,
    latency_method: str = "hrf_peak",
    fir_window: float = 20.0,
    xcorr_maxlag: int = 10,
    quality_weights=(1.0, 1.0, 1.0),
    goal_pre_window_sec: float = 2.0,
    response_window_sec: float = 3.0,
    goal_objective_window_sec: float = 2.0,
    feedback_window_sec: float = 2.0,
    quality_ridge: float = 1.0,
    require_explicit_feedback: bool = True,
    return_details: bool = False,
    hardware_backend=None,
    require_explicit_goals: bool = True,
    quality_response_estimate: str = "auto",
    quality_lag_sec: float = None,
    quality_cv_folds: int = 5,
    quality_null_samples: int = 200,
    quality_random_state: int = 0,
):
    """
    Responsiveness–Adaptation Metric (RAM).

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_time)
        Timeseries matrix for each ROI.
    tr : float
        Repetition time (sampling interval) in seconds.
    stimulus_onsets : list/array-like or dict, optional
        Event specification. Legacy mode accepts a list/array of stimulus onset
        times (in seconds). Structured mode accepts a dict with:
          - ``onsets`` (or ``stimulus_onsets``): stimulus onsets in seconds
          - ``goal_onsets``: objective/cue onsets in seconds
          - ``feedback_onsets``: feedback/outcome onsets in seconds
          - ``feedback_values``: scalar feedback labels/values aligned 1:1 to
            ``feedback_onsets`` (numeric or categorical)
    epsilon : float, optional
        Small constant added to denominator. When ``None`` (default), uses
        ``tr`` so the stabilizer equals the temporal resolution (Δt_res).
    magnitude_scale : float, optional
        Multiplicative scale applied to the response magnitude term M
        (default 0.5).
    response_model : {'hrf', 'boxcar', 'stick'}, optional
        Event-response regressor used for the magnitude term. ``'hrf'`` sums
        peak-normalised canonical (Glover) HRFs placed at the exact onset
        times and sampled at the frame times ``k * tr`` (so a response of
        unit peak amplitude has weight 1). ``'boxcar'`` and ``'stick'`` are
        intended for high-sampling-rate modalities such as EEG. Goal and
        feedback events are modelled with the same response model as
        nuisance regressors, so M is the stimulus amplitude only; RAM is
        undefined when this design is rank-deficient.
    response_boxcar_width_sec : float, optional
        Width of the post-stimulus boxcar regressor when
        ``response_model='boxcar'``. Defaults to ``response_window_sec``.
    latency_method : {'hrf_peak', 'xcorr', 'fir'}, optional
        Method used to estimate the latency term T in the denominator.
        * 'hrf_peak' (default): the model latency -- the canonical HRF peak
          (about 5 s) for ``'hrf'``, half the boxcar width for ``'boxcar'``.
          This is a model constant, not a measurement; it is not available
          for ``'stick'``.
        * 'xcorr': the non-negative lag (samples up to ``xcorr_maxlag``) that
          maximises the mean squared correlation between the stimulus stick
          function and the regional signals.
        * 'fir': time-to-peak of the baseline-corrected event average
          (global field power across regions) within ``fir_window`` after
          onset; the ``fir_window`` before onset is the baseline.
        Measured latencies are refined by parabolic interpolation; a maximum
        at the edge of the search range leaves T (and RAM) undefined.
    fir_window : float, optional
        Baseline and search window (seconds) on either side of each onset for
        the FIR latency estimate.
    xcorr_maxlag : int, optional
        Maximum lag (in samples) explored for the cross-correlation latency.
    quality_weights : tuple(float, float, float), optional
        Non-negative weights ``(w_G, w_F, w_U)`` for quality components
        Goal-alignment (G), Feedback-integration (F), and Adaptive-update (U).
    goal_pre_window_sec : float, optional
        Pre-stimulus window (seconds) used for the implicit objective-state
        proxy (only with ``require_explicit_goals=False``).
    response_window_sec : float, optional
        Stimulus response window (seconds, ``'window'`` estimate), starting
        ``quality_lag_sec`` after onset.
    goal_objective_window_sec : float, optional
        Post-goal-event window (seconds, ``'window'`` estimate), starting
        ``quality_lag_sec`` after the goal onset.
    feedback_window_sec : float, optional
        Feedback-locked window length (seconds, ``'window'`` estimate): the
        pre-feedback baseline ``[f - W, f)`` and the response
        ``[f + lag, f + lag + W)``.
    quality_ridge : float, optional
        Ridge for the cross-validated canonical correlation in G, relative to
        the mean non-zero eigenvalue of the training covariance (default 1.0).
    require_explicit_feedback : bool, optional
        When ``True`` (default), RAM is marked undefined unless explicit
        feedback events and feedback values are provided. With ``False``,
        stimulus events and a prediction-error proxy stand in for missing
        feedback (opt-in proxy).
    return_details : bool, optional
        If ``True``, returns a dict with speed/quality sub-terms and component
        diagnostics instead of only the scalar RAM value.
    require_explicit_goals : bool, optional
        When ``True`` (default), RAM is undefined without goal events. With
        ``False``, pre-stimulus activity stands in for the objective state
        (opt-in proxy).
    quality_response_estimate : {'auto', 'glm', 'window'}, optional
        How trial-wise response patterns for G, F and U are estimated.
        ``'glm'``: least-squares-all canonical-HRF betas, one regressor per
        goal, stimulus and feedback event, so overlapping hemodynamic
        responses are separated. ``'window'``: event-locked window means.
        ``'auto'`` (default) uses ``'glm'`` for ``response_model='hrf'`` and
        ``'window'`` otherwise.
    quality_lag_sec : float, optional
        Delay (seconds) applied to all post-event windows in the ``'window'``
        estimate. ``None`` derives it from the response model: for ``'hrf'``
        the canonical HRF peak minus half the response window (window centred
        on the hemodynamic peak), otherwise 0.
    quality_cv_folds : int, optional
        Number of time-blocked cross-validation folds for G.
    quality_null_samples : int, optional
        Null draws for chance correction: phase-randomised surrogates for G,
        permutations of the feedback signal for F and U (0 disables the
        correction; the chance level is then taken as 0).
    quality_random_state : int, optional
        Seed for the null draws.

    Returns
    -------
    float or dict
        Scalar RAM value (default) or a diagnostics dict when
        ``return_details=True``. Undefined RAM is ``NaN``; the details carry
        ``undefined_reason``.

    Notes
    -----
    RAM is implemented as:
      RAM = (M / (T + epsilon)) * Q
    with quality term:
      Q = (G^w_G * F^w_F * U^w_U)^(1 / (w_G + w_F + w_U))
    where G, F, U are data-derived, chance-corrected values in [0,1] from
    event-locked neural activity. A weighted component that cannot be
    measured makes RAM undefined (NaN); a measured component of 0 gives
    RAM = 0.
    """
    # ensure ts has shape (n_regions, n_tp)
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    if tr is None or tr <= 0:
        raise ValueError("tr must be a positive number for RAM")
    if magnitude_scale <= 0:
        raise ValueError("magnitude_scale must be > 0")
    response_model_norm = str(response_model).strip().lower()
    if response_model_norm not in {"hrf", "boxcar", "stick"}:
        raise ValueError("response_model must be one of {'hrf', 'boxcar', 'stick'}")
    if latency_method not in {"hrf_peak", "xcorr", "fir"}:
        raise ValueError(
            "latency_method must be one of 'hrf_peak', 'xcorr', or 'fir'"
        )
    if latency_method == "hrf_peak" and response_model_norm == "stick":
        raise ValueError(
            "latency_method='hrf_peak' needs a response model with an intrinsic "
            "latency ('hrf' or 'boxcar'); use 'fir' or 'xcorr' with 'stick'."
        )
    estimate = str(quality_response_estimate).strip().lower()
    if estimate not in {"auto", "glm", "window"}:
        raise ValueError(
            "quality_response_estimate must be one of {'auto', 'glm', 'window'}"
        )
    if estimate == "auto":
        estimate = "glm" if response_model_norm == "hrf" else "window"
    if response_boxcar_width_sec is not None and float(response_boxcar_width_sec) <= 0:
        raise ValueError("response_boxcar_width_sec must be > 0 when provided")
    if epsilon is None:
        epsilon = float(tr)
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    w = np.asarray(quality_weights, dtype=float).reshape(-1)
    if w.size != 3:
        raise ValueError("quality_weights must be a 3-tuple: (w_G, w_F, w_U)")
    if np.any(w < 0):
        raise ValueError("quality_weights must be non-negative")
    if not np.any(w > 0):
        raise ValueError("At least one quality weight must be > 0")
    if goal_pre_window_sec < 0 or response_window_sec <= 0:
        raise ValueError("goal_pre_window_sec must be >= 0 and response_window_sec > 0")
    if goal_objective_window_sec <= 0 or feedback_window_sec <= 0:
        raise ValueError("goal_objective_window_sec and feedback_window_sec must be > 0")
    if quality_ridge <= 0:
        raise ValueError("quality_ridge must be > 0")
    if quality_lag_sec is not None and float(quality_lag_sec) < 0:
        raise ValueError("quality_lag_sec must be >= 0 when provided")
    if int(quality_cv_folds) < 2:
        raise ValueError("quality_cv_folds must be >= 2")
    if int(quality_null_samples) < 0:
        raise ValueError("quality_null_samples must be >= 0")

    backend = resolve_hardware_backend(hardware_backend)
    n_regions, n_tp = ts.shape
    event_bundle = _coerce_ram_event_bundle(stimulus_onsets)
    stim_onsets_s, stim_idx = _sanitize_onset_seconds(event_bundle["onsets"], tr=tr, n_tp=n_tp)
    goal_onsets_s, goal_idx = _sanitize_onset_seconds(
        event_bundle["goal_onsets"], tr=tr, n_tp=n_tp
    )

    hrf_peak_sec, _ = canonical_hrf_peak()
    if quality_lag_sec is None:
        if response_model_norm == "hrf":
            quality_lag_sec = max(0.0, hrf_peak_sec - 0.5 * float(response_window_sec))
        else:
            quality_lag_sec = 0.0
    quality_lag_sec = float(quality_lag_sec)

    nan = float("nan")
    details = {
        "value": nan,
        "undefined_reason": None,
        "speed_term": nan,
        "quality_term": nan,
        "magnitude_term": nan,
        "latency_term": nan,
        "latency_seconds": nan,
        "latency_method": str(latency_method),
        "latency_undefined_reason": None,
        "epsilon": float(epsilon),
        "response_model": response_model_norm,
        "quality_response_estimate": estimate,
        "quality_lag_sec": quality_lag_sec if estimate == "window" else nan,
        "components": {
            "goal_alignment": nan,
            "feedback_integration": nan,
            "adaptive_update": nan,
        },
        "component_undefined_reasons": {
            "goal_alignment": None,
            "feedback_integration": None,
            "adaptive_update": None,
        },
        "weights": {
            "goal_alignment": float(w[0]),
            "feedback_integration": float(w[1]),
            "adaptive_update": float(w[2]),
        },
        "goal_source": None,
        "feedback_signal_source": None,
        "goal_alignment_cv_corr": nan,
        "goal_alignment_null_mean": nan,
        "goal_alignment_p_null": nan,
        "n_stimulus_events": int(stim_idx.size),
        "n_goal_events": int(goal_idx.size),
        "n_feedback_events": 0,
        "n_goal_response_pairs": 0,
        "n_feedback_events_used": 0,
        "n_adaptive_updates": 0,
    }

    def _undefined(reason):
        details["value"] = nan
        details["undefined_reason"] = str(reason)
        return details if return_details else nan

    if stim_idx.size == 0:
        return _undefined("missing_stimulus_events")
    if not np.all(np.isfinite(ts)):
        # M, T and the quality terms would be NaN or silently zero-filled.
        return _undefined("non_finite_timeseries")
    log.debug(
        (
            "[compute_RAM] ts shape: %d regions x %d timepoints, tr=%.6f, "
            "n_onsets=%d, latency_method=%s"
        ),
        n_regions,
        n_tp,
        float(tr),
        int(stim_idx.size),
        latency_method,
    )

    # 1) build stick regressor
    stick = np.zeros(n_tp)
    stick[stim_idx] = 1

    # 2) create stimulus-response regressor on the true time axis (every
    # in-range event, including events that share a sample).
    width_samples = 1
    if response_model_norm == "boxcar":
        width_sec = (
            float(response_window_sec)
            if response_boxcar_width_sec is None
            else float(response_boxcar_width_sec)
        )
        width_samples = max(1, int(round(width_sec / float(tr))))

    def _model_regressor(onsets_sec):
        if response_model_norm == "hrf":
            return _hrf_regressor(onsets_sec, tr=tr, n_tp=n_tp)
        out = np.zeros(n_tp, dtype=float)
        starts = np.rint(np.asarray(onsets_sec) / float(tr)).astype(np.int64)
        for idx in starts.tolist():
            end = min(n_tp, int(idx) + width_samples)
            if end > int(idx):
                out[int(idx):end] += 1.0
        return out / float(width_samples)

    stim_all_s = _in_range_onset_seconds(event_bundle["onsets"], tr=tr, n_tp=n_tp)
    if response_model_norm == "hrf":
        reg = _model_regressor(stim_all_s)
        default_latency = hrf_peak_sec
    elif response_model_norm == "boxcar":
        reg = _model_regressor(stim_all_s)
        default_latency = 0.5 * float(width_samples) * float(tr)
    else:
        reg = stick.astype(float, copy=True)
        default_latency = float("nan")
    # Annotated goal and feedback events are modelled as nuisance regressors
    # (same response model) so that their responses -- which overlap the
    # stimulus response for fMRI -- do not leak into the stimulus amplitude.
    nuisance = []
    for key in ("goal_onsets", "feedback_onsets"):
        on = _in_range_onset_seconds(event_bundle[key], tr=tr, n_tp=n_tp)
        col = _model_regressor(on) if on.size else None
        if col is not None and np.any(col != 0.0):
            nuisance.append(col)
    # design matrix: stimulus regressor + nuisance event regressors + intercept
    X = np.column_stack([reg] + nuisance + [np.ones(n_tp)])
    magnitude_reason = None
    if int(np.linalg.matrix_rank(X)) < int(X.shape[1]):
        # e.g. stimulus events coinciding with goal/feedback events: the
        # stimulus amplitude is not identifiable (RAM undefined, see below).
        magnitude_reason = "magnitude_design_rank_deficient"
        abs_mean_beta = nan
    else:
        # 3) solve for betas via pseudo-inverse: betas shape
        # (n_cols × n_regions); transpose ts so rows are time samples
        betas = accelerated_pinv_dot(X, ts.T, backend=backend)
        stim_betas = betas[0, :]  # first row corresponds to stimulus regressor

        # use mean absolute β as amplitude (M), with optional scale factor
        abs_mean_beta = float(np.mean(np.abs(stim_betas))) * float(magnitude_scale)
    details["magnitude_term"] = abs_mean_beta

    # 4) estimate latency T (seconds, >= 0 by construction; never clipped)
    if latency_method == "hrf_peak":
        T, latency_reason = float(default_latency), None
    elif latency_method == "xcorr":
        T, latency_reason = _xcorr_latency_seconds(
            ts, stick, tr=tr, maxlag=xcorr_maxlag
        )
    else:
        T, latency_reason = _fir_latency_seconds(
            ts, stim_idx, tr=tr, fir_window=fir_window
        )
    details["latency_seconds"] = float(T)
    details["latency_undefined_reason"] = latency_reason
    latency_term = float(T) + float(epsilon)
    speed_term = abs_mean_beta / latency_term if np.isfinite(T) else nan
    details["latency_term"] = float(latency_term)
    details["speed_term"] = float(speed_term)

    # 5) goal/feedback structure (strict by default: measured inputs only)
    fb_onsets_in = event_bundle["feedback_onsets"]
    fb_onsets_raw = np.asarray(
        [] if fb_onsets_in is None else fb_onsets_in, dtype=float
    ).reshape(-1)
    fb_values_raw = _coerce_numeric_feedback(event_bundle.get("feedback_values"))
    if fb_values_raw.size > 0 and fb_values_raw.size != fb_onsets_raw.size:
        if require_explicit_feedback or fb_onsets_raw.size > 0:
            return _undefined("feedback_values_misaligned")
        fb_values_raw = np.empty(0, dtype=float)
    if fb_values_raw.size > 0:
        fb_onsets_s, feedback_idx, feedback_values = _sanitize_onsets_with_values(
            fb_onsets_raw, fb_values_raw, tr=tr, n_tp=n_tp
        )
    else:
        fb_onsets_s, feedback_idx = _sanitize_onset_seconds(
            fb_onsets_raw, tr=tr, n_tp=n_tp
        )
        feedback_values = None
    details["n_feedback_events"] = int(feedback_idx.size)

    if require_explicit_feedback:
        if (
            feedback_idx.size == 0
            or feedback_values is None
            or np.unique(feedback_values).size < 2
        ):
            return _undefined("missing_explicit_feedback")
    if require_explicit_goals and goal_idx.size == 0:
        return _undefined("missing_goal_events")

    ts_z = accelerated_zscore(ts, axis=1, backend=backend, eps=1e-12)
    rng = np.random.RandomState(quality_random_state)

    # Opt-in feedback proxy: stimulus events stand in for feedback events.
    fb_from_stim = feedback_values is None and feedback_idx.size == 0
    if fb_from_stim:
        fb_onsets_s, feedback_idx = stim_onsets_s.copy(), stim_idx.copy()

    # 6) trial-wise response patterns (fixed event sets, re-usable on surrogates)
    if estimate == "glm":
        pinv, keep_masks, slices, full_rank = _lsa_hrf_design(
            [stim_onsets_s, goal_onsets_s, [] if fb_from_stim else fb_onsets_s],
            tr=tr,
            n_tp=n_tp,
        )
        if not full_rank:
            # Minimum-norm betas of a rank-deficient beta-series design are not
            # trial-wise measurements; G/F/U would be computed on artefacts.
            return _undefined("trialwise_design_rank_deficient")
        stim_used = stim_idx[keep_masks[0]]
        goal_used = goal_idx[keep_masks[1]]
        fb_keep = keep_masks[0] if fb_from_stim else keep_masks[2]
        spec = {
            "mode": "glm",
            "pinv": pinv,
            "slices": slices,
            "fb_from_stim": fb_from_stim,
        }
    else:
        lag_samples = max(0, int(round(quality_lag_sec / float(tr))))
        resp_samples = max(1, int(round(response_window_sec / float(tr))))
        gobj_samples = max(1, int(round(goal_objective_window_sec / float(tr))))
        fb_samples = max(1, int(round(feedback_window_sec / float(tr))))
        stim_used = stim_idx[stim_idx + lag_samples + resp_samples <= n_tp]
        goal_used = goal_idx[goal_idx + lag_samples + gobj_samples <= n_tp]
        fb_keep = (feedback_idx - fb_samples >= 0) & (
            feedback_idx + lag_samples + fb_samples <= n_tp
        )
        spec = {
            "mode": "window",
            "lag": lag_samples,
            "resp": resp_samples,
            "gobj": gobj_samples,
            "fb": fb_samples,
            "stim_used": stim_used,
            "goal_used": goal_used,
            "fb_used": feedback_idx[fb_keep],
        }

    # Goal pairing: each stimulus with its nearest preceding goal event.
    spec["goal_source"] = None
    if goal_used.size > 0:
        pos = np.searchsorted(goal_used, stim_used, side="right") - 1
        valid = pos >= 0
        spec.update(
            goal_source="explicit_goal_events",
            goal_pairs_x=pos[valid],
            goal_pairs_y=np.flatnonzero(valid),
            goal_groups=pos[valid],
        )
    elif goal_idx.size == 0 and not require_explicit_goals:
        gpre = max(1, int(round(goal_pre_window_sec / float(tr))))
        has_pre = stim_used - gpre >= 0
        spec.update(
            goal_source="pre_stimulus_state_proxy",
            proxy_idx=stim_used[has_pre],
            gpre=gpre,
            goal_pairs_x=np.arange(int(has_pre.sum())),
            goal_pairs_y=np.flatnonzero(has_pre),
            goal_groups=np.arange(int(has_pre.sum())),
        )
    else:
        spec.update(
            goal_source="explicit_goal_events",
            goal_pairs_x=np.empty(0, dtype=np.int64),
            goal_pairs_y=np.empty(0, dtype=np.int64),
            goal_groups=np.empty(0, dtype=np.int64),
        )
    patterns = _quality_patterns(ts_z, spec)
    stim_response_vecs = patterns["stim"]

    # Feedback signal aligned 1:1 with feedback events.
    if feedback_values is not None:
        feedback_signal = np.asarray(feedback_values, dtype=float)
        details["feedback_signal_source"] = "explicit_values"
    else:
        # Opt-in proxy: prediction error of the most recent stimulus response.
        resp_energy = np.sqrt(np.sum(stim_response_vecs ** 2, axis=1))
        pe = _prediction_error_signal(resp_energy)
        j = np.searchsorted(stim_used, feedback_idx, side="right") - 1
        feedback_signal = np.full(feedback_idx.size, np.nan)
        feedback_signal[j >= 0] = pe[j[j >= 0]] if pe.size else np.nan
        details["feedback_signal_source"] = "prediction_error_proxy"
    fb_ok = np.isfinite(feedback_signal)

    g_score, g_reason, g_info = _goal_alignment_component(
        ts_z=ts_z,
        spec=spec,
        patterns=patterns,
        ridge=float(quality_ridge),
        cv_folds=int(quality_cv_folds),
        n_null=int(quality_null_samples),
        rng=rng,
    )
    details.update(g_info)
    f_sel = fb_ok[fb_keep]
    f_score, n_fb_used, f_reason = _feedback_integration_component(
        fb_strength=np.asarray(patterns["fb_strength"])[f_sel],
        fb_signal=feedback_signal[fb_keep][f_sel],
        n_null=int(quality_null_samples),
        rng=rng,
    )
    u_score, n_updates, u_reason = _adaptive_update_component(
        stim_response_vecs=stim_response_vecs,
        stim_used_idx=stim_used,
        feedback_idx=feedback_idx[fb_ok],
        feedback_signal=feedback_signal[fb_ok],
        n_null=int(quality_null_samples),
        rng=rng,
    )
    details["n_feedback_events_used"] = int(n_fb_used)
    details["n_adaptive_updates"] = int(n_updates)

    names = ("goal_alignment", "feedback_integration", "adaptive_update")
    components = np.asarray([g_score, f_score, u_score], dtype=float)
    reasons = (g_reason, f_reason, u_reason)
    for name, val, reason in zip(names, components, reasons):
        details["components"][name] = float(val)
        details["component_undefined_reasons"][name] = reason

    if magnitude_reason is not None:
        return _undefined(magnitude_reason)
    undefined_weighted = [
        reasons[i] or f"{names[i]}_undefined"
        for i in range(3)
        if w[i] > 0.0 and not np.isfinite(components[i])
    ]
    if undefined_weighted:
        return _undefined(undefined_weighted[0])

    # Exact weighted geometric mean over weighted components: a measured zero
    # yields Q=0 (defined), an unmeasurable component makes RAM undefined.
    wpos = w > 0.0
    comp_w = np.clip(components[wpos], 0.0, 1.0)
    if np.any(comp_w <= 0.0):
        quality_term = 0.0
    else:
        quality_term = float(np.exp(np.sum(w[wpos] * np.log(comp_w)) / np.sum(w[wpos])))
        quality_term = float(np.clip(quality_term, 0.0, 1.0))
    details["quality_term"] = quality_term

    if not np.isfinite(T):
        return _undefined(latency_reason or "latency_undefined")

    ram_value = float(speed_term * quality_term)
    if not np.isfinite(ram_value):
        return _undefined("non_finite_result")
    details["value"] = ram_value
    if not return_details:
        return ram_value
    return details

def compute_PDI(
    ts: np.ndarray,
    bins: int = 10,
    baseline_ts: np.ndarray = None,
    weighted: bool = True,
    normalize: bool = False,
    clip_negative: bool = True,
    stability_segments: int = 4,
    noise_penalty_kappa: float = 1.0,
    component_weights: tuple = (0.35, 0.25, 0.20, 0.20),
    ordinal_order: int = 3,
    multiscale_max_scale: int = 5,
    eps: float = 1e-12,
    hardware_backend=None,
    return_details: bool = False,
    null_surrogates: int = 0,
    null_method: str = "phase_randomize",
    null_seed: int | None = None,
    null_min_shift: int | None = None,
) -> float:
    """
    Composite measurable Phenomenal Differentiation Index (PDI).

    PDI is operationalized from EEG/fMRI timeseries as a baseline-referenced
    composition of four observable dimensions:
      1) normalized excess repertoire differentiation (Xi_norm),
      2) spatial repertoire divergence across regions (Delta_spatial),
      3) effective representational dimensionality (D_eff_norm), and
      4) multiscale ordinal complexity (C_multi).

    Each dimension is measured for observed and baseline runs, baseline
    variability is explicitly attenuated per dimension, and the aggregate is
    corrected by temporal stability and differential noise penalties.

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_time)
        Task timeseries.
    bins : int, optional
        Number of histogram bins for discretization. The maximum entropy is
        ``log2(bins)``.
    baseline_ts : ndarray, optional
        Baseline (rest) timeseries. Can be provided as a 2D array of shape
        (n_regions, n_time) corresponding to a single run, or a 3D array of
        shape (n_runs, n_regions, n_time) corresponding to multiple baseline
        runs. If ``None``, the baseline entropy is approximated by shuffling
        each region of ``ts``.
    weighted : bool, optional
        Only used when ``baseline_ts`` is 3D. When ``True`` (default), the
        baseline component statistics from multiple runs are weighted by their
        number of timepoints. When ``False``, a simple mean across runs is used.
    normalize : bool, optional
        When ``True`` returns a bounded [0,1] representation. With
        ``clip_negative=False``, signed values are mapped by ``0.5*(x+1)``.
    clip_negative : bool, optional
        When ``True`` (default), negative gains are clipped to zero.
    stability_segments : int, optional
        Number of contiguous segments for split-stability estimation.
    noise_penalty_kappa : float, optional
        Strength of differential-noise attenuation.
    component_weights : tuple, optional
        Non-negative weights for (Xi_norm, Delta_spatial, D_eff_norm, C_multi).
    ordinal_order : int, optional
        Permutation order used for multiscale ordinal complexity.
    multiscale_max_scale : int, optional
        Maximum coarse-graining scale for multiscale ordinal complexity.
    eps : float, optional
        Numerical stability constant.
    return_details : bool, optional
        When True, return a dictionary with components, definedness and the
        null-calibration fields instead of a float.
    null_surrogates : int, optional
        Number of surrogate task runs for null calibration (0 = off). Each
        surrogate is scored against the same baseline(s); the null-corrected
        value is ``PDI_excess = PDI_obs - mean(PDI_null)`` and
        ``PDI_z = PDI_excess / sd(PDI_null)``. When enabled, the returned value is
        ``PDI_calibrated = max(PDI_excess, 0)`` (``PDI_excess`` if
        ``clip_negative=False``).
    null_method : str, optional
        Surrogate type (see ``_surrogate_timeseries``). The default
        ``'phase_randomize'`` (multivariate, amplitude-adjusted) keeps each
        region's marginal and all linear auto-/cross-correlation, so the
        calibrated value is the differentiation not explained by a linear
        Gaussian process with the task's spectra. This removes the positive
        clipping bias and the sensitivity of Xi to autocorrelation/smoothness
        (filtering, TR).
    null_seed : int or None, optional
        Seed of the surrogate generator (default 0).
    null_min_shift : int or None, optional
        Minimum shift for ``null_method='circular_shift'``.

    Returns
    -------
    float or dict
        The PDI value (NaN when undefined), or details when requested.

    Notes
    -----
    With ``clip_negative=True`` (default), the output is non-negative and
    increases when observed differentiation exceeds baseline with good temporal
    stability and low differential noise. Uncalibrated PDI is positively
    biased under the null (per-component gains are clipped at 0 before
    weighting), so cross-condition comparisons should use the calibrated value.
    PDI is undefined (NaN) for fewer than 2 regions, fewer than
    ``max(4, ordinal_order + 1)`` timepoints, an empty baseline list, or runs
    without finite values.
    """
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    if stability_segments < 2:
        raise ValueError("stability_segments must be >= 2")
    if noise_penalty_kappa < 0:
        raise ValueError("noise_penalty_kappa must be >= 0")
    if ordinal_order < 3:
        raise ValueError("ordinal_order must be >= 3")
    if multiscale_max_scale < 1:
        raise ValueError("multiscale_max_scale must be >= 1")

    comp_w = np.asarray(component_weights, dtype=float)
    if comp_w.shape != (4,):
        raise ValueError("component_weights must contain 4 values")
    if np.any(comp_w < 0) or float(np.sum(comp_w)) <= 0:
        raise ValueError("component_weights must be non-negative and not all zero")
    comp_w = comp_w / float(np.sum(comp_w))
    if int(null_surrogates) < 0:
        raise ValueError("null_surrogates must be >= 0")
    if null_method not in SURROGATE_METHODS:
        raise ValueError(f"null_method must be one of {SURROGATE_METHODS}")
    backend = resolve_hardware_backend(hardware_backend)
    null_seed_eff = _resolve_null_seed(null_seed, 0)

    def _undefined(reason):
        if return_details:
            out = {
                "value": np.nan,
                "raw": np.nan,
                "defined": False,
                "undefined_reason": str(reason),
            }
            out.update(
                _metric_null_fields(
                    "PDI", None, null_method, null_seed_eff, clip_negative
                )
            )
            return out
        return np.nan

    # Measured inputs only: an undefined PDI is NaN with a reason, never 0.0.
    n_regions_in, n_time_in = ts.shape
    if n_regions_in < 2:
        return _undefined("insufficient_regions")
    if n_time_in < max(4, int(ordinal_order) + 1):
        return _undefined("insufficient_timepoints")
    if not np.any(np.isfinite(ts)):
        return _undefined("no_finite_task_values")

    # compute baseline reference set
    if baseline_ts is None:
        # fallback: shuffled surrogate destroys temporal structure while
        # preserving marginal amplitudes.
        baseline_source = "shuffled_task_surrogate"
        n_regions = ts.shape[0]
        rng = np.random.RandomState(0)
        shuffled = np.stack([rng.permutation(ts[r]) for r in range(n_regions)])
        baseline_ts_use = [shuffled]
    else:
        baseline_source = "provided"
        # baseline_ts may be provided as a list/tuple of runs (each run is 2D)
        if isinstance(baseline_ts, (list, tuple)):
            baseline_ts_use = list(baseline_ts)
        elif isinstance(baseline_ts, np.ndarray):
            if baseline_ts.ndim == 2:
                baseline_ts_use = [baseline_ts]
            elif baseline_ts.ndim == 3:
                # split 3D array into list of 2D runs
                baseline_ts_use = [baseline_ts[i] for i in range(baseline_ts.shape[0])]
            else:
                raise ValueError(
                    f"baseline_ts must be 2D, 3D or a list/tuple of 2D arrays; got shape {baseline_ts.shape}"
                )
        else:
            raise ValueError(
                "baseline_ts must be a numpy array or a list/tuple of numpy arrays"
            )

    if not baseline_ts_use:
        return _undefined("empty_baseline")
    if any(
        not np.any(np.isfinite(np.asarray(run, dtype=float))) for run in baseline_ts_use
    ):
        return _undefined("no_finite_baseline_values")

    # Use shared bin edges across observed + baseline to keep response alphabet fixed.
    arrays_for_edges = [ts.ravel()] + [run.ravel() for run in baseline_ts_use]
    all_vals = np.concatenate(arrays_for_edges)
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size == 0:
        return _undefined("no_finite_values")
    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    edges = np.linspace(vmin, vmax, bins + 1)
    bins_inner = edges[1:-1]
    log2_bins = float(np.log2(bins))

    def _disc(run_ts: np.ndarray) -> np.ndarray:
        x = np.asarray(run_ts, dtype=float)
        x = np.nan_to_num(x, nan=vmin, posinf=vmax, neginf=vmin)
        z = np.digitize(x, bins_inner, right=False)
        return np.clip(z, 0, bins - 1).astype(np.int16)

    def _p_from_disc(disc: np.ndarray) -> np.ndarray:
        cnt = np.bincount(disc.ravel(), minlength=bins).astype(float)
        s = float(cnt.sum())
        if s <= 0:
            return np.ones(bins, dtype=float) / float(bins)
        return cnt / s

    def _jsd(p: np.ndarray, q: np.ndarray) -> float:
        p = p / max(float(np.sum(p)), eps)
        q = q / max(float(np.sum(q)), eps)
        m = 0.5 * (p + q)
        return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))

    def _conditional_entropy_1step(disc: np.ndarray) -> float:
        n_regions, n_time = disc.shape
        if n_time < 2:
            return 0.0
        cond_vals = []
        for r in range(n_regions):
            prev = disc[r, :-1].astype(np.int64, copy=False)
            nxt = disc[r, 1:].astype(np.int64, copy=False)
            joint = np.zeros((bins, bins), dtype=float)
            np.add.at(joint, (prev, nxt), 1.0)
            total = float(joint.sum())
            if total <= 0:
                cond_vals.append(0.0)
                continue
            p_joint = joint / total
            p_prev = p_joint.sum(axis=1)
            h_joint = float(entropy(p_joint.ravel(), base=2))
            h_prev = float(entropy(p_prev, base=2))
            cond_vals.append(max(h_joint - h_prev, 0.0))
        return float(np.mean(cond_vals)) if cond_vals else 0.0

    def _stability(disc: np.ndarray) -> float:
        n_time = disc.shape[1]
        if n_time < 4:
            return 1.0
        n_seg = int(min(stability_segments, max(2, n_time // 2)))
        idx_segments = np.array_split(np.arange(n_time), n_seg)
        ps = []
        for seg in idx_segments:
            if seg.size == 0:
                continue
            p = _p_from_disc(disc[:, seg])
            ps.append(p)
        if len(ps) < 2:
            return 1.0
        d = []
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                d.append(_jsd(ps[i], ps[j]))
        if not d:
            return 1.0
        return float(np.clip(1.0 - np.mean(d), 0.0, 1.0))

    def _region_probabilities(disc: np.ndarray) -> np.ndarray:
        n_regions = disc.shape[0]
        p = np.empty((n_regions, bins), dtype=float)
        for r in range(n_regions):
            cnt = np.bincount(disc[r], minlength=bins).astype(float)
            s = float(cnt.sum())
            if s <= 0:
                p[r] = 1.0 / float(bins)
            else:
                p[r] = cnt / s
        return p

    def _spatial_repertoire_divergence(disc: np.ndarray) -> float:
        n_regions = disc.shape[0]
        if n_regions < 2:
            return 0.0
        p_reg = _region_probabilities(disc)
        h_reg = entropy(p_reg, axis=1, base=2)
        acc = 0.0
        pairs = 0
        for i in range(n_regions - 1):
            m = 0.5 * (p_reg[i + 1:] + p_reg[i])
            h_m = entropy(m, axis=1, base=2)
            jsd = h_m - 0.5 * (h_reg[i + 1:] + h_reg[i])
            jsd = np.clip(jsd, 0.0, 1.0)
            acc += float(np.sum(jsd))
            pairs += int(jsd.size)
        if pairs <= 0:
            return 0.0
        return float(np.clip(acc / float(pairs), 0.0, 1.0))

    def _effective_dimensionality(run_ts: np.ndarray) -> float:
        x = accelerated_zscore(run_ts, axis=1, backend=backend, eps=eps)
        q = int(min(x.shape))
        if q <= 1:
            return 0.0
        try:
            svals = accelerated_svd_values(x, backend=backend)
        except np.linalg.LinAlgError:
            return 0.0
        var = np.square(svals)
        total = float(np.sum(var))
        if total <= 0:
            return 0.0
        denom = float(np.sum(np.square(var)))
        if denom <= 0:
            return 0.0
        d_eff = (total * total) / (denom + eps)
        d_eff = float(np.clip(d_eff, 1.0, float(q)))
        return float(np.clip((d_eff - 1.0) / (float(q) - 1.0), 0.0, 1.0))

    n_perm = int(math.factorial(ordinal_order))
    perm_uniform = np.ones(n_perm, dtype=float) / float(n_perm)
    perm_log_norm = float(np.log2(max(2, n_perm)))
    perm_base = np.power(
        ordinal_order,
        np.arange(ordinal_order - 1, -1, -1, dtype=np.int64),
        dtype=np.int64,
    )
    perm_lookup = np.full(int(ordinal_order ** ordinal_order), -1, dtype=np.int64)
    perms = np.asarray(list(itertools.permutations(range(ordinal_order))), dtype=np.int64)
    perm_codes = np.sum(perms * perm_base, axis=1, dtype=np.int64)
    perm_lookup[perm_codes] = np.arange(n_perm, dtype=np.int64)
    delta = np.zeros(n_perm, dtype=float)
    delta[0] = 1.0
    jsd_max_perm = max(_jsd(delta, perm_uniform), eps)

    def _ordinal_prob_1d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size < ordinal_order:
            return None
        win = np.lib.stride_tricks.sliding_window_view(x, ordinal_order)
        if win.shape[0] == 0:
            return None
        ranks = np.argsort(win, axis=1, kind="mergesort")
        codes = np.sum(ranks * perm_base, axis=1, dtype=np.int64)
        ids = perm_lookup[codes]
        ids = ids[ids >= 0]
        if ids.size == 0:
            return None
        cnt = np.bincount(ids, minlength=n_perm).astype(float)
        s = float(np.sum(cnt))
        if s <= 0:
            return None
        return cnt / s

    def _multiscale_ordinal_complexity(run_ts: np.ndarray) -> float:
        x = np.asarray(run_ts, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        n_regions, n_time = x.shape
        if n_time < ordinal_order + 1:
            return 0.0
        max_scale_eff = int(min(multiscale_max_scale, n_time // (ordinal_order + 1)))
        if max_scale_eff < 1:
            return 0.0
        vals = []
        for scale in range(1, max_scale_eff + 1):
            usable = (n_time // scale) * scale
            if usable < (ordinal_order + 1):
                continue
            coarse = x[:, :usable].reshape(n_regions, usable // scale, scale).mean(axis=2)
            for r in range(n_regions):
                p_ord = _ordinal_prob_1d(coarse[r])
                if p_ord is None:
                    continue
                h_norm = float(entropy(p_ord, base=2) / (perm_log_norm + eps))
                h_norm = float(np.clip(h_norm, 0.0, 1.0))
                q_jsd = _jsd(p_ord, perm_uniform) / jsd_max_perm
                q_jsd = float(np.clip(q_jsd, 0.0, 1.0))
                vals.append(h_norm * q_jsd)
        if not vals:
            return 0.0
        return float(np.clip(np.mean(vals), 0.0, 1.0))

    def _noise_index(run_ts: np.ndarray) -> float:
        x = np.asarray(run_ts, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mad_x = median_abs_deviation(x, axis=1, scale=1.0)
        if x.shape[1] < 2:
            mad_d = np.zeros(x.shape[0], dtype=float)
        else:
            dx = np.diff(x, axis=1)
            mad_d = median_abs_deviation(dx, axis=1, scale=1.0)
        ratio = mad_d / (mad_x + eps)
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size == 0:
            return 0.0
        return float(np.median(ratio))

    def _run_stats(run_ts: np.ndarray) -> dict:
        if run_ts.ndim != 2:
            raise ValueError(
                f"each run should be 2D (n_regions×n_time), got shape {run_ts.shape}"
            )
        disc = _disc(run_ts)
        p = _p_from_disc(disc)
        H = float(entropy(p, base=2))
        h = _conditional_entropy_1step(disc)
        xi_norm = float(np.clip((H - h) / (log2_bins + eps), 0.0, 1.0))
        delta_spatial = _spatial_repertoire_divergence(disc)
        d_eff_norm = _effective_dimensionality(run_ts)
        c_multi = _multiscale_ordinal_complexity(run_ts)
        S = _stability(disc)
        nu = _noise_index(run_ts)
        return {
            "H": H,
            "h": h,
            "xi_norm": xi_norm,
            "delta_spatial": delta_spatial,
            "d_eff_norm": d_eff_norm,
            "c_multi": c_multi,
            "S": S,
            "nu": nu,
        }

    obs = _run_stats(ts)
    base_stats = [_run_stats(run) for run in baseline_ts_use]

    if weighted:
        w = np.asarray([run.size for run in baseline_ts_use], dtype=float)
    else:
        w = np.ones(len(base_stats), dtype=float)
    if np.sum(w) <= 0:
        w = np.ones(len(base_stats), dtype=float)
    w = w / float(np.sum(w))

    x_xi_arr = np.asarray([d["xi_norm"] for d in base_stats], dtype=float)
    x_delta_arr = np.asarray([d["delta_spatial"] for d in base_stats], dtype=float)
    x_deff_arr = np.asarray([d["d_eff_norm"] for d in base_stats], dtype=float)
    x_cmulti_arr = np.asarray([d["c_multi"] for d in base_stats], dtype=float)
    s_arr = np.asarray([d["S"] for d in base_stats], dtype=float)
    nu_arr = np.asarray([d["nu"] for d in base_stats], dtype=float)

    x_xi_base = float(np.sum(w * x_xi_arr))
    x_delta_base = float(np.sum(w * x_delta_arr))
    x_deff_base = float(np.sum(w * x_deff_arr))
    x_cmulti_base = float(np.sum(w * x_cmulti_arr))
    S_base = float(np.sum(w * s_arr))
    nu_base = float(np.sum(w * nu_arr))
    x_xi_sd = float(np.sqrt(np.sum(w * (x_xi_arr - x_xi_base) ** 2)))
    x_delta_sd = float(np.sqrt(np.sum(w * (x_delta_arr - x_delta_base) ** 2)))
    x_deff_sd = float(np.sqrt(np.sum(w * (x_deff_arr - x_deff_base) ** 2)))
    x_cmulti_sd = float(np.sqrt(np.sum(w * (x_cmulti_arr - x_cmulti_base) ** 2)))

    contrasts = np.asarray(
        [
            (obs["xi_norm"] - x_xi_base) / (1.0 + x_xi_sd),
            (obs["delta_spatial"] - x_delta_base) / (1.0 + x_delta_sd),
            (obs["d_eff_norm"] - x_deff_base) / (1.0 + x_deff_sd),
            (obs["c_multi"] - x_cmulti_base) / (1.0 + x_cmulti_sd),
        ],
        dtype=float,
    )
    contrasts = np.clip(contrasts, -1.0, 1.0)
    gain_pos = np.clip(contrasts, 0.0, 1.0)
    core_positive = float(np.sum(comp_w * gain_pos))
    core_signed = float(np.sum(comp_w * contrasts))

    # Stability correction: if task stability is below baseline stability,
    # down-weight the gain.
    c_stab = float(np.clip(obs["S"] / (S_base + eps), 0.0, 1.0))

    # Differential-noise correction from robust first-difference ratio.
    c_noise = float(np.exp(-noise_penalty_kappa * max(0.0, obs["nu"] - nu_base)))

    core = core_positive if clip_negative else core_signed
    pdi_raw = core * c_stab * c_noise
    if clip_negative:
        pdi_raw = max(float(pdi_raw), 0.0)

    if not normalize:
        pdi_value = float(pdi_raw)
    elif clip_negative:
        # bounded normalization in [0,1]
        pdi_value = float(np.clip(pdi_raw, 0.0, 1.0))
    else:
        pdi_value = float(np.clip(0.5 * (pdi_raw + 1.0), 0.0, 1.0))

    null_stats = None
    null_reason = None
    if int(null_surrogates) > 0:
        # Surrogate task runs scored against the same baseline(s).
        null_rng = np.random.RandomState(null_seed_eff)
        null_vals = []
        for _ in range(int(null_surrogates)):
            try:
                surr = _surrogate_timeseries(
                    ts, null_method, null_rng, min_shift=null_min_shift
                )
            except ValueError as exc:
                null_reason = f"surrogates_unavailable: {exc}"
                break
            null_vals.append(
                compute_PDI(
                    surr,
                    bins=bins,
                    baseline_ts=baseline_ts,
                    weighted=weighted,
                    normalize=normalize,
                    clip_negative=clip_negative,
                    stability_segments=stability_segments,
                    noise_penalty_kappa=noise_penalty_kappa,
                    component_weights=component_weights,
                    ordinal_order=ordinal_order,
                    multiscale_max_scale=multiscale_max_scale,
                    eps=eps,
                    hardware_backend=backend,
                )
            )
        null_stats = _null_calibration_stats(pdi_value, null_vals)
        if null_reason is None and null_stats["null_n"] == 0:
            null_reason = "all_surrogates_undefined"
    null_fields = _metric_null_fields(
        "PDI", null_stats, null_method, null_seed_eff, clip_negative, null_reason
    )
    # Calibration requested but no valid surrogate -> NaN, never the raw value.
    value = null_fields["PDI_calibrated"] if int(null_surrogates) > 0 else pdi_value

    if not return_details:
        return float(value)
    component_names = ("xi_norm", "delta_spatial", "d_eff_norm", "c_multi")
    out = {
        "value": float(value),
        "raw": float(pdi_value),
        "pdi_core": float(core),
        "c_stab": float(c_stab),
        "c_noise": float(c_noise),
        "contrasts": {k: float(v) for k, v in zip(component_names, contrasts)},
        "observed_components": {
            k: float(obs[k]) for k in component_names + ("S", "nu")
        },
        "baseline_components": {
            "xi_norm": x_xi_base,
            "delta_spatial": x_delta_base,
            "d_eff_norm": x_deff_base,
            "c_multi": x_cmulti_base,
            "S": S_base,
            "nu": nu_base,
        },
        "baseline_source": baseline_source,
        "n_baseline_runs": int(len(baseline_ts_use)),
        "defined": True,
        "undefined_reason": None,
    }
    out.update(null_fields)
    return out


def compute_NAS(
    ts: np.ndarray,
    tr: float = None,
    zthr: float = 1.0,
    eps: float = 0.2,
    tau: float = None,
    lambda_phase: float = 0.5,
    alpha: float = 0.20,
    beta: float = 0.16,
    gamma: float = 0.14,
    delta: float = 0.12,
    eta: float = 0.16,
    zeta: float = 0.12,
    rho: float = 0.10,
    bands: list = None,
    band_weights: list = None,
    window_len: int = None,
    step_len: int = None,
    max_triads: int = 5000,
    random_state: int = 0,
    workspace_nodes: np.ndarray = None,
    workspace_quantile: float = 0.2,
    workspace_min_size: int = 4,
    directed_lag: int = 1,
    reverberation_lags: tuple = (2, 3, 4),
    baseline_ts: np.ndarray = None,
    boost_against_baseline: bool = False,
    normalize: bool = True,
    hardware_backend=None,
    return_details: bool = False,
    null_surrogates: int = 0,
    null_method: str = "circular_shift",
    null_seed: int | None = None,
    null_min_shift: int | None = None,
) -> float:
    """
    Theory-aligned Network Activation Synchrony (NAS).

    NAS operationalizes GNW-like broadcast dynamics using seven measurable
    components from EEG/fMRI node×time series:
      1) L: intra-broadcast synchrony,
      2) B: broadcast reach to non-broadcast nodes,
      3) H: higher-order triadic closure in broadcast nodes,
      4) D: temporal stability of synchrony graphs,
      5) W: dedicated-workspace recruitment with ignition boost,
      6) E: directed broadcast efficacy (lagged asymmetry),
      7) R: workspace reverberatory persistence (reportability proxy).

    Components are computed per band and aggregated by weighted geometric mean.

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_time)
        Timeseries matrix.
    tr : float
        Sample interval in seconds. Must be explicit and > 0.
    zthr : float, optional
        Legacy parameter kept for backward compatibility.
    eps : float, optional
        Legacy compatibility parameter (not used in current strict estimator).
    tau : float
        Broadcast-node quantile tolerance in [0,1). Broadcast set is
        strength >= quantile(1-tau).
    lambda_phase : float, optional
        Mixing factor between PLV and envelope coupling in [0,1].
    alpha, beta, gamma, delta, eta, zeta, rho : float, optional
        Exponents for (L,B,H,D,W,E,R). Automatically normalized to sum to 1.
    bands : list[(lo,hi)]
        Explicit frequency bands in Hz.
    band_weights : list[float]
        Non-negative weights for each band. Normalized internally.
    window_len : int
        Window length in samples.
    step_len : int
        Window step in samples.
    max_triads : int, optional
        Maximum number of triads sampled per window for H computation.
    random_state : int, optional
        Seed for triad sampling.
    workspace_nodes : array-like[int], optional
        Optional predefined dedicated-workspace node indices. If None,
        workspace is inferred from global synchrony-centrality quantile.
    workspace_quantile : float, optional
        Fraction of top-central nodes used when inferring workspace.
    workspace_min_size : int, optional
        Minimum number of nodes in inferred workspace.
    directed_lag : int, optional
        Lag (samples) for directed broadcast asymmetry.
    reverberation_lags : tuple[int], optional
        Positive lags used for workspace reverberation persistence.
    baseline_ts : ndarray, optional
        Baseline timeseries for optional boost computation.
    boost_against_baseline : bool, optional
        If True and ``baseline_ts`` is provided, returns max(NAS - NAS_base, 0).
    normalize : bool, optional
        If True, clip final value to [0,1].
    return_details : bool, optional
        When True, return a dictionary with per-band values, workspace nodes,
        definedness and null-calibration fields instead of a float.
    null_surrogates : int, optional
        Number of surrogates for null calibration (0 = off). The broadcast set
        V (top-tau strength) and, unless ``workspace_nodes`` is given, the
        workspace G are selected from the same data they are scored on, so NAS
        is positive for independent noise. Surrogates repeat the whole
        selection-and-scoring procedure on data with the cross-node coupling
        removed, so ``NAS_excess = NAS_obs - mean(NAS_null)`` and
        ``NAS_z = NAS_excess / sd(NAS_null)`` are free of that selection bias.
        When enabled, the returned value is ``NAS_calibrated = max(NAS_excess, 0)``.
    null_method : str, optional
        Surrogate type (see ``_surrogate_timeseries``); default
        ``'circular_shift'`` (independent shift per node: keeps each node's
        spectrum/marginal, destroys cross-node phase and envelope coupling).
    null_seed : int or None, optional
        Seed of the surrogate generator (default ``random_state``).
    null_min_shift : int or None, optional
        Minimum circular shift in samples (default 10% of the run).

    Returns
    -------
    float or dict
        NAS value in [0,1] when ``normalize=True``; NaN when the input has fewer
        than 2 regions or fewer than 4 timepoints.
    """
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    _ = (zthr, eps)  # retained only for backward compatibility
    n_regions, n_time = ts.shape
    if int(null_surrogates) < 0:
        raise ValueError("null_surrogates must be >= 0")
    if null_method not in SURROGATE_METHODS:
        raise ValueError(f"null_method must be one of {SURROGATE_METHODS}")
    null_seed_eff = _resolve_null_seed(null_seed, random_state)
    if n_regions < 2 or n_time < 4:
        # Undefined, not zero (measured inputs only).
        if return_details:
            out = {
                "value": np.nan,
                "raw": np.nan,
                "defined": False,
                "undefined_reason": "insufficient_shape",
            }
            out.update(_metric_null_fields("NAS", None, null_method, null_seed_eff))
            return out
        return np.nan
    backend = resolve_hardware_backend(hardware_backend)

    if tr is None or (not np.isfinite(tr)) or float(tr) <= 0:
        raise ValueError("compute_NAS requires explicit positive tr")
    if tau is None:
        raise ValueError("compute_NAS requires explicit tau")
    if not (0.0 <= tau < 1.0):
        raise ValueError("tau must be in [0, 1)")
    if not (0.0 <= lambda_phase <= 1.0):
        raise ValueError("lambda_phase must be in [0,1]")
    if max_triads < 1:
        raise ValueError("max_triads must be >= 1")
    if directed_lag < 1:
        raise ValueError("directed_lag must be >= 1")
    if not (0.0 < workspace_quantile <= 1.0):
        raise ValueError("workspace_quantile must be in (0,1]")
    if workspace_min_size < 2:
        raise ValueError("workspace_min_size must be >= 2")

    comp_w = np.array([alpha, beta, gamma, delta, eta, zeta, rho], dtype=float)
    if np.any(comp_w < 0):
        raise ValueError("alpha/beta/gamma/delta/eta/zeta/rho must be non-negative")
    if float(comp_w.sum()) <= 0:
        raise ValueError("sum of NAS component exponents must be > 0")
    comp_w = comp_w / float(comp_w.sum())
    alpha, beta, gamma, delta, eta, zeta, rho = [float(v) for v in comp_w]

    # Node-wise z-scoring for scale robustness.
    x = np.asarray(ts, dtype=float)
    means = np.nanmean(x, axis=1, keepdims=True)
    stds = np.nanstd(x, axis=1, ddof=0, keepdims=True) + 1e-12
    x = (x - means) / stds
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    fs = 1.0 / float(tr)

    if bands is None:
        raise ValueError("compute_NAS requires explicit frequency bands")
    band_specs = []
    for b in bands:
        if b is None:
            raise ValueError("compute_NAS does not allow None band entries")
        if (not isinstance(b, (tuple, list))) or len(b) != 2:
            raise ValueError("bands must be a list of (low, high) tuples")
        lo, hi = b
        lo = None if lo is None else float(lo)
        hi = None if hi is None else float(hi)
        if lo is None or hi is None:
            raise ValueError("compute_NAS requires finite low/high for all bands")
        band_specs.append((lo, hi))
    if not band_specs:
        raise ValueError("compute_NAS requires a non-empty bands list")

    if band_weights is None:
        raise ValueError("compute_NAS requires explicit band_weights")
    bw = np.asarray(band_weights, dtype=float)
    if bw.shape[0] != len(band_specs):
        raise ValueError("band_weights length must match number of bands")
    if np.any(bw < 0) or float(bw.sum()) <= 0:
        raise ValueError("band_weights must be non-negative and sum > 0")
    bw = bw / float(bw.sum())

    if window_len is None:
        raise ValueError("compute_NAS requires explicit window_len")
    wlen = int(window_len)
    if wlen < 4:
        raise ValueError("window_len must be >= 4")
    if wlen > n_time:
        raise ValueError(f"window_len={wlen} exceeds run length n_time={n_time}")
    if step_len is None:
        raise ValueError("compute_NAS requires explicit step_len")
    step = int(step_len)
    if step < 1:
        raise ValueError("step_len must be >= 1")
    if step > wlen:
        raise ValueError("step_len must be <= window_len")

    starts = list(range(0, max(1, n_time - wlen + 1), step))
    if not starts:
        starts = [0]
    if starts[-1] != max(0, n_time - wlen):
        starts.append(max(0, n_time - wlen))
    starts = sorted(set(starts))

    rng_master = np.random.RandomState(int(random_state))

    def _global_workspace_indices(x_full: np.ndarray) -> np.ndarray:
        if workspace_nodes is not None:
            idx = np.asarray(workspace_nodes, dtype=int).reshape(-1)
            idx = idx[(idx >= 0) & (idx < n_regions)]
            idx = np.unique(idx)
            if idx.size >= 2:
                return idx
        A0 = np.abs(accelerated_corrcoef(x_full, backend=backend))
        if A0.ndim != 2:
            A0 = np.zeros((n_regions, n_regions), dtype=float)
        A0 = np.nan_to_num(A0, nan=0.0, posinf=1.0, neginf=0.0)
        A0 = np.clip(A0, 0.0, 1.0)
        np.fill_diagonal(A0, 0.0)
        strength0 = A0.sum(axis=1) / float(max(1, n_regions - 1))
        k = int(np.ceil(float(workspace_quantile) * float(n_regions)))
        k = max(int(workspace_min_size), k)
        k = min(max(2, k), n_regions)
        idx = np.argpartition(strength0, -k)[-k:]
        return np.sort(np.asarray(idx, dtype=int))

    G = _global_workspace_indices(x)
    G_set = set(G.tolist())

    def _band_filter(data, lo, hi, fs_local):
        if lo is None or hi is None or fs_local is None:
            raise ValueError("compute_NAS requires valid band edges and tr/fs")
        nyq = 0.5 * fs_local
        if lo <= 0 or hi <= lo or hi >= nyq:
            raise ValueError(
                f"compute_NAS invalid band ({lo}, {hi}) for Nyquist {nyq}"
            )
        wn = [lo / nyq, hi / nyq]
        try:
            sos = butter(2, wn, btype="band", output="sos")
            fil = sosfiltfilt(sos, data, axis=1)
            return fil, True
        except ValueError:
            raise ValueError(f"compute_NAS bandpass failed for band ({lo}, {hi})")

    def _window_adjacency(data_w, use_phase, lambda_mix):
        if use_phase:
            analytic = hilbert(data_w, axis=1)
            phase = np.angle(analytic)
            env = np.abs(analytic)

            zc = np.exp(1j * phase)
            plv = np.abs(
                accelerated_dot(zc, zc.conj().T, backend=backend) / float(data_w.shape[1])
            ).astype(float)
            ecoh = np.abs(accelerated_corrcoef(env, backend=backend))
            A = lambda_mix * plv + (1.0 - lambda_mix) * ecoh
        else:
            A = np.abs(accelerated_corrcoef(data_w, backend=backend))

        if A.ndim != 2:
            A = np.zeros((n_regions, n_regions), dtype=float)
        A = np.nan_to_num(A, nan=0.0, posinf=1.0, neginf=0.0)
        A = np.clip(A, 0.0, 1.0)
        np.fill_diagonal(A, 0.0)
        return A

    def _triadic_intensity(Avv, max_n, rng_local):
        nV = Avv.shape[0]
        if nV < 3:
            return 0.0
        total = nV * (nV - 1) * (nV - 2) // 6
        vals = []
        if total <= max_n:
            for i, j, k in itertools.combinations(range(nV), 3):
                vals.append((Avv[i, j] * Avv[i, k] * Avv[j, k]) ** (1.0 / 3.0))
        else:
            seen = set()
            max_attempts = int(max_n * 25)
            attempts = 0
            while len(vals) < max_n and attempts < max_attempts:
                tri = tuple(sorted(rng_local.choice(nV, size=3, replace=False).tolist()))
                attempts += 1
                if tri in seen:
                    continue
                seen.add(tri)
                i, j, k = tri
                vals.append((Avv[i, j] * Avv[i, k] * Avv[j, k]) ** (1.0 / 3.0))
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _directed_broadcast_advantage(data_w: np.ndarray, V: np.ndarray) -> float:
        if V.size == 0:
            return 0.0
        lag = int(directed_lag)
        if data_w.shape[1] <= (lag + 2):
            return 0.0
        x0 = data_w[:, :-lag]
        x1 = data_w[:, lag:]
        x0 = x0 - np.mean(x0, axis=1, keepdims=True)
        x1 = x1 - np.mean(x1, axis=1, keepdims=True)
        x0 = x0 / (np.std(x0, axis=1, keepdims=True) + 1e-12)
        x1 = x1 / (np.std(x1, axis=1, keepdims=True) + 1e-12)
        C_lag = accelerated_dot(x0, x1.T, backend=backend) / float(max(1, x0.shape[1]))
        C_lag = np.nan_to_num(C_lag, nan=0.0, posinf=0.0, neginf=0.0)
        A_dir = np.maximum(C_lag - C_lag.T, 0.0)
        np.fill_diagonal(A_dir, 0.0)

        Vbar = np.setdiff1d(np.arange(n_regions), V, assume_unique=True)
        if Vbar.size == 0:
            return 0.0
        out = float(np.mean(A_dir[np.ix_(V, Vbar)]))
        back = float(np.mean(A_dir[np.ix_(Vbar, V)]))
        return float(np.clip((out - back) / (out + back + 1e-12), 0.0, 1.0))

    rev_lags = np.asarray([int(lag) for lag in reverberation_lags], dtype=int)
    rev_lags = rev_lags[rev_lags > 0]

    nas_bands = []
    band_components = []
    for b_idx, (flo, fhi) in enumerate(band_specs):
        xb, can_phase = _band_filter(x, flo, fhi, fs)
        use_phase = bool(can_phase and xb.shape[1] >= 8)
        if not use_phase:
            raise ValueError(
                "compute_NAS requires phase/envelope synchrony; "
                "bandpass unavailable or window too short (<8 samples)"
            )

        L_vals, B_vals, H_vals = [], [], []
        W_vals, E_vals, R_vals = [], [], []
        A_seq = []
        for w_idx, st in enumerate(starts):
            en = min(st + wlen, n_time)
            data_w = xb[:, st:en]
            if data_w.shape[1] < 4:
                continue

            A = _window_adjacency(data_w, use_phase=use_phase, lambda_mix=lambda_phase)
            A_seq.append(A)

            strength = A.sum(axis=1) / float(max(1, n_regions - 1))
            q = float(np.quantile(strength, 1.0 - tau))
            V = np.where(strength >= q)[0]
            if V.size < 2:
                L_vals.append(0.0)
                B_vals.append(0.0)
                H_vals.append(0.0)
                W_vals.append(0.0)
                E_vals.append(0.0)
                R_vals.append(0.0)
                continue

            Avv = A[np.ix_(V, V)]
            iu = np.triu_indices(V.size, k=1)
            L = float(Avv[iu].mean()) if iu[0].size > 0 else 0.0

            Vbar = np.setdiff1d(np.arange(n_regions), V, assume_unique=True)
            if Vbar.size > 0:
                B = float(A[np.ix_(V, Vbar)].mean())
            else:
                B = 0.0

            rng_local = np.random.RandomState(
                int(rng_master.randint(0, 2**31 - 1) + 31 * b_idx + 7 * w_idx)
            )
            H = _triadic_intensity(Avv, max_n=max_triads, rng_local=rng_local)

            Vbar = np.setdiff1d(np.arange(n_regions), V, assume_unique=True)
            n_hit = int(sum(1 for i in V if i in G_set))
            recruit = float(n_hit / float(max(1, G.size)))
            if Vbar.size >= 2:
                A_nn = A[np.ix_(Vbar, Vbar)]
                iu_nn = np.triu_indices(Vbar.size, k=1)
                non_sync = float(A_nn[iu_nn].mean()) if iu_nn[0].size > 0 else 0.0
            else:
                non_sync = 0.0
            ignition = float(np.clip((L - non_sync) / (1.0 - non_sync + 1e-12), 0.0, 1.0))
            W = float(np.clip(recruit * ignition, 0.0, 1.0))

            E = _directed_broadcast_advantage(data_w, V)

            g_sig = np.mean(data_w[G, :], axis=0) if G.size > 0 else np.mean(data_w, axis=0)
            g_sig = g_sig - float(np.mean(g_sig))
            g_den = float(np.linalg.norm(g_sig))
            rev_vals = []
            if g_den > 0 and rev_lags.size > 0:
                for lag in rev_lags.tolist():
                    if lag >= g_sig.shape[0] - 1:
                        continue
                    a = g_sig[:-lag]
                    b = g_sig[lag:]
                    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                    if den <= 0:
                        continue
                    rev_vals.append(max(float(np.dot(a, b) / den), 0.0))
            rev = float(np.mean(rev_vals)) if rev_vals else 0.0
            R = float(np.clip(recruit * rev, 0.0, 1.0))

            L_vals.append(L)
            B_vals.append(B)
            H_vals.append(H)
            W_vals.append(W)
            E_vals.append(E)
            R_vals.append(R)

        if not L_vals:
            nas_bands.append(0.0)
            band_components.append({})
            continue

        L_bar = float(np.mean(L_vals))
        B_bar = float(np.mean(B_vals))
        H_bar = float(np.mean(H_vals))
        W_bar = float(np.mean(W_vals))
        E_bar = float(np.mean(E_vals))
        R_bar = float(np.mean(R_vals))

        if len(A_seq) <= 1:
            D = 1.0
        else:
            dvals = []
            for A0, A1 in zip(A_seq[:-1], A_seq[1:]):
                num = float(np.linalg.norm(A1 - A0, ord="fro"))
                den = float(np.linalg.norm(A1, ord="fro") + np.linalg.norm(A0, ord="fro") + 1e-12)
                dvals.append(num / den)
            D = float(np.clip(1.0 - np.mean(dvals), 0.0, 1.0))

        comps = np.asarray(
            [
                max(L_bar, 0.0),
                max(B_bar, 0.0),
                max(H_bar, 0.0),
                max(D, 0.0),
                max(W_bar, 0.0),
                max(E_bar, 0.0),
                max(R_bar, 0.0),
            ],
            dtype=float,
        )
        if np.any((comps <= 0.0) & (comp_w > 0.0)):
            nas_f = 0.0
        else:
            nas_f = float(np.exp(np.sum(comp_w * np.log(comps + 1e-12))))
        nas_bands.append(float(nas_f))
        band_components.append(
            {
                k: float(v)
                for k, v in zip(("L", "B", "H", "D", "W", "E", "R"), comps.tolist())
            }
        )

    nas_unboosted = float(np.dot(bw, np.asarray(nas_bands, dtype=float)))
    nas_val = nas_unboosted
    nas_base = None
    if boost_against_baseline and baseline_ts is not None:
        nas_base = compute_NAS(
            baseline_ts,
            tr=tr,
            zthr=zthr,
            eps=eps,
            tau=tau,
            lambda_phase=lambda_phase,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            eta=eta,
            zeta=zeta,
            rho=rho,
            bands=band_specs,
            band_weights=bw.tolist(),
            window_len=wlen,
            step_len=step,
            max_triads=max_triads,
            random_state=random_state,
            workspace_nodes=G,
            workspace_quantile=workspace_quantile,
            workspace_min_size=workspace_min_size,
            directed_lag=directed_lag,
            reverberation_lags=tuple(int(v) for v in rev_lags.tolist()),
            baseline_ts=None,
            boost_against_baseline=False,
            normalize=False,
            hardware_backend=backend,
        )
        nas_val = max(nas_val - nas_base, 0.0)

    def _finalize(v):
        if nas_base is not None:
            v = max(float(v) - float(nas_base), 0.0)
        return float(np.clip(v, 0.0, 1.0)) if normalize else float(v)

    nas_value = float(np.clip(nas_val, 0.0, 1.0)) if normalize else float(nas_val)

    null_stats = None
    null_reason = None
    if int(null_surrogates) > 0:
        null_rng = np.random.RandomState(null_seed_eff)
        null_vals = []
        for _ in range(int(null_surrogates)):
            try:
                surr = _surrogate_timeseries(
                    x, null_method, null_rng, min_shift=null_min_shift
                )
            except ValueError as exc:
                null_reason = f"surrogates_unavailable: {exc}"
                break
            surr_unboosted = compute_NAS(
                surr,
                tr=tr,
                tau=tau,
                lambda_phase=lambda_phase,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                eta=eta,
                zeta=zeta,
                rho=rho,
                bands=band_specs,
                band_weights=bw.tolist(),
                window_len=wlen,
                step_len=step,
                max_triads=max_triads,
                random_state=random_state,
                workspace_nodes=workspace_nodes,
                workspace_quantile=workspace_quantile,
                workspace_min_size=workspace_min_size,
                directed_lag=directed_lag,
                reverberation_lags=tuple(int(v) for v in rev_lags.tolist()),
                baseline_ts=None,
                boost_against_baseline=False,
                normalize=False,
                hardware_backend=backend,
            )
            null_vals.append(_finalize(surr_unboosted))
        null_stats = _null_calibration_stats(nas_value, null_vals)
        if null_reason is None and null_stats["null_n"] == 0:
            null_reason = "all_surrogates_undefined"
    null_fields = _metric_null_fields(
        "NAS", null_stats, null_method, null_seed_eff, undefined_reason=null_reason
    )
    # Calibration requested but no valid surrogate -> NaN, never the raw value.
    value = null_fields["NAS_calibrated"] if int(null_surrogates) > 0 else nas_value

    if not return_details:
        return float(value)
    out = {
        "value": float(value),
        "raw": float(nas_value),
        "nas_unboosted": float(nas_unboosted),
        "nas_baseline": (None if nas_base is None else float(nas_base)),
        "band_values": [float(v) for v in nas_bands],
        "band_components": band_components,
        "workspace_nodes": [int(v) for v in G.tolist()],
        "defined": True,
        "undefined_reason": None,
    }
    out.update(null_fields)
    return out


SURROGATE_METHODS = (
    "circular_shift",
    "phase_randomize",
    "phase_randomize_independent",
    "shuffle",
)
_SURROGATE_IAAFT_ITERATIONS = 10


def _surrogate_timeseries(ts, method, rng, min_shift=None):
    """
    One surrogate of a node x time matrix for null calibration.

    - 'circular_shift': independent circular shift of every node except node 0
      (shifts >= ``min_shift`` samples, default 10% of the run). Preserves each
      node's marginal and autocorrelation exactly; destroys cross-node alignment.
    - 'phase_randomize': multivariate Fourier surrogate (the same random phase
      offsets for all nodes), amplitude-adjusted by IAAFT iterations (rank
      remap to each node's original values / re-impose the original amplitude
      spectrum). Preserves marginals, auto- and (approximately) cross-spectra,
      i.e. linear-Gaussian structure; destroys nonlinear/non-Gaussian structure.
    - 'phase_randomize_independent': as above with independent phases per node
      (destroys cross-node coupling, keeps each node's spectrum and marginal).
    - 'shuffle': independent random permutation per node (destroys all temporal
      and cross-node structure; keeps marginals).
    NaN positions are kept in place.
    """
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2:
        raise ValueError(
            f"surrogate input must be 2D (n_nodes x n_time), got {x.shape}"
        )
    n, t = x.shape
    if method == "circular_shift":
        lo = int(math.ceil(0.1 * t)) if min_shift is None else int(min_shift)
        lo = max(1, lo)
        hi = t - lo
        if hi < lo:
            raise ValueError(
                f"run too short (n_time={t}) for circular-shift surrogates "
                f"with min_shift={lo}"
            )
        candidates = np.arange(lo, hi + 1)
        shifts = rng.choice(
            candidates, size=max(0, n - 1), replace=bool(candidates.size < n - 1)
        )
        out = x.copy()
        for i in range(1, n):
            out[i] = np.roll(x[i], int(shifts[i - 1]))
        return out
    if method == "shuffle":
        return np.stack([x[i, rng.permutation(t)] for i in range(n)]) if n else x.copy()
    if method in ("phase_randomize", "phase_randomize_independent"):
        nan_mask = ~np.isfinite(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_mean = np.nanmean(np.where(nan_mask, np.nan, x), axis=1, keepdims=True)
        row_mean = np.nan_to_num(row_mean, nan=0.0)
        filled = np.where(nan_mask, row_mean, x)
        centred = filled - row_mean
        spec = np.fft.rfft(centred, axis=1)
        n_freq = spec.shape[1]
        rows = 1 if method == "phase_randomize" else n
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(rows, n_freq))
        phases[:, 0] = 0.0
        if t % 2 == 0 and n_freq > 1:
            phases[:, -1] = 0.0
        surr = np.fft.irfft(spec * np.exp(1j * phases), n=t, axis=1)
        finite_idx = [np.flatnonzero(~nan_mask[i]) for i in range(n)]
        sorted_vals = [np.sort(x[i, finite_idx[i]]) for i in range(n)]
        target_amp = np.abs(spec)
        out = filled.copy()
        # IAAFT refinement (Schreiber & Schmitz 1996): alternate between the
        # original amplitude spectrum and the original marginal (rank remap of
        # the finite samples only; NaN positions keep the row-mean filler).
        for it in range(_SURROGATE_IAAFT_ITERATIONS + 1):
            for i in range(n):
                idx = finite_idx[i]
                order = np.argsort(surr[i, idx], kind="mergesort")
                out[i, idx[order]] = sorted_vals[i]
            if it == _SURROGATE_IAAFT_ITERATIONS:
                break
            cur = np.fft.rfft(out - row_mean, axis=1)
            surr = (
                np.fft.irfft(target_amp * np.exp(1j * np.angle(cur)), n=t, axis=1)
                + row_mean
            )
        out[nan_mask] = np.nan
        return out
    raise ValueError(f"null_method must be one of {SURROGATE_METHODS}, got {method!r}")


def _null_calibration_stats(observed, null_values):
    """Excess over a surrogate null, z-score and one-sided empirical p-value."""
    vals = np.asarray(null_values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    out = {
        "null_n": int(vals.size),
        "null_mean": np.nan,
        "null_sd": np.nan,
        "z": np.nan,
        "p": np.nan,
        "excess": np.nan,
    }
    obs = float(observed)
    if vals.size == 0 or not np.isfinite(obs):
        return out
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
    out["null_mean"] = mean
    out["null_sd"] = sd
    out["excess"] = float(obs - mean)
    if np.isfinite(sd) and sd > 0:
        out["z"] = float((obs - mean) / sd)
    out["p"] = float((1.0 + np.sum(vals >= obs)) / (vals.size + 1.0))
    return out


def _iim_null_fields(stats, psi_full, delta_psi, null_values=None, meta=None):
    """
    IIM null-calibration fields. The null statistic is the integration mass
    Delta_Psi = Psi_full - max_kappa Psi^kappa (bits) of surrogates with
    independent nodes. It is also expressed on the IIM_raw scale by dividing by
    the observed Psi_full (IIM_null_mean/IIM_null_sd), so that
    IIM_z = (IIM_raw - IIM_null_mean) / IIM_null_sd. The ratio itself is not
    calibrated directly: it is scale-free, so finite-sample noise gives it an
    O(1) value that no surrogate comparison can separate from real coupling.
    """
    psi_scale = float(psi_full) + 1e-12 if np.isfinite(psi_full) else np.nan
    if stats is None:
        stats = _null_calibration_stats(np.nan, [])
    excess = float(stats["excess"])
    out = {
        "IIM_null_calibrated": bool(stats["null_n"] > 0),
        "IIM_raw": (
            np.nan if not np.isfinite(delta_psi) else float(delta_psi / psi_scale)
        ),
        "Delta_Psi": float(delta_psi),
        "Delta_Psi_null": [float(v) for v in (null_values or [])],
        "Delta_Psi_null_mean": float(stats["null_mean"]),
        "Delta_Psi_null_sd": float(stats["null_sd"]),
        "IIM_null_n": int(stats["null_n"]),
        "IIM_null_mean": (
            float(stats["null_mean"] / psi_scale) if stats["null_n"] else np.nan
        ),
        "IIM_null_sd": (
            float(stats["null_sd"] / psi_scale) if stats["null_n"] else np.nan
        ),
        "IIM_z": float(stats["z"]),
        "IIM_null_p": float(stats["p"]),
        "IIM_excess": excess,
        "canonical_calibrated": (
            float(max(excess, 0.0)) if np.isfinite(excess) else np.nan
        ),
    }
    out["IIM_calibrated"] = out["canonical_calibrated"]
    out.update(meta or {})
    return out


def _metric_null_fields(
    prefix, stats, method, seed, clip_negative=True, undefined_reason=None
):
    """Null-calibration fields ``<prefix>_*`` for PDI/NAS details dictionaries."""
    if stats is None:
        stats = _null_calibration_stats(np.nan, [])
    excess = float(stats["excess"])
    if np.isfinite(excess):
        calibrated = float(max(excess, 0.0)) if clip_negative else excess
    else:
        calibrated = np.nan
    return {
        f"{prefix}_null_undefined_reason": undefined_reason,
        f"{prefix}_null_calibrated": bool(stats["null_n"] > 0),
        f"{prefix}_null_method": str(method),
        f"{prefix}_null_seed": seed,
        f"{prefix}_null_n": int(stats["null_n"]),
        f"{prefix}_null_mean": float(stats["null_mean"]),
        f"{prefix}_null_sd": float(stats["null_sd"]),
        f"{prefix}_z": float(stats["z"]),
        f"{prefix}_null_p": float(stats["p"]),
        f"{prefix}_excess": excess,
        f"{prefix}_calibrated": calibrated,
    }


def _resolve_null_seed(null_seed, fallback):
    if null_seed is not None:
        return int(null_seed)
    if isinstance(fallback, (int, np.integer)):
        return int(fallback)
    return 0


def compute_IIM(
    ts: np.ndarray,
    bins: int = 3,
    lag_trs: int = 1,
    n_parts: int | None = None,
    rng: int = 0,
    method: str = "causal",
    partition_mode: str = "all",
    clamp: bool = True,
    scale: float = 1.0,
    max_nodes: int | None = None,
    max_mechanism_size: int | None = None,
    max_purview_size: int | None = None,
    tpm_alpha: float = 1e-3,
    max_state_space: int = 1500,
    return_details: bool = False,
    checkpoint_path: str | None = None,
    resume_from_checkpoint: bool = True,
    checkpoint_every_cuts: int = 1,
    progress_log_every_cuts: int = 1,
    progress_label: str | None = None,
    phase1_parallel_workers: int | None = None,
    phase1_chunk_size: int = 8,
    phase1_shared_memory: bool = True,
    use_induced_partition_cache: bool = True,
    kernel_cache_path: str | None = None,
    kernel_cache_memory_entries: int = 300_000,
    kernel_cache_flush_batch: int = 5_000,
    hardware_backend=None,
    tpm_estimator: str = "node_shrinkage",
    node_selection: str = "variance",
    node_indices=None,
    state_budget_policy: str = "reduce_bins_first",
    null_surrogates: int = 0,
    null_method: str = "circular_shift",
    null_seed: int | None = None,
    null_min_shift: int | None = None,
    keep_kernel_cache: bool = False,
) -> float:
    """
    IIT-leaning Integrated Information Metric (IIM) from an empirical causal TPM.

    The method estimates:
      1) a discrete transition probability matrix (TPM) over a reduced node set,
      2) mechanism-level irreducibility via cause/effect repertoires, and
      3) system-level irreducibility by Minimum Information Partition (MIP)
         over bipartition cuts.

    Let Ψ be the weighted sum of mechanism irreducibility terms in the intact
    TPM and Ψ^κ the same quantity under cut κ. We evaluate:

        IIM_raw = (Ψ - max_k Ψ^κ) / (Ψ + eps)

    where max_k Ψ^κ corresponds to the cut that preserves the most causal
    structure (MIP under this preserved-structure form). Canonical mapping:

        IIM_can = clip(IIM_raw, 0, 1) # CI-facing bounded integration score

    keeps the CI-facing term in [0, 1] while retaining signed raw diagnostics.

    Definitions used by this implementation:

    - Mechanism/purview partitions are bipartitions into two non-empty parts;
      both pairings (M1->Z1, M2->Z2) and (M1->Z2, M2->Z1) are searched, so phi
      is the minimum over all such partitions and invariant to node labelling.
      Consequently mechanisms or purviews of size 1 have no admissible
      partition and contribute exactly 0 to Psi (Psi effectively sums over
      |M| >= 2 with weight 1/|M|).
    - TPM (``tpm_estimator``): 'node_shrinkage' (default) estimates the TPM in
      state-by-node form (next-step node states conditionally independent
      given the current state, as in IIT) with James-Stein shrinkage of each
      row towards the node's own transition p(x_i' | x_i) (the "no cross-node
      influence" model, estimated from all transitions); 'node_laplace' uses Laplace
      smoothing per node; 'joint_laplace' is the legacy joint K^n x K^n Laplace
      estimate (alpha=``tpm_alpha``), which turns finite-sample noise into
      spurious non-factorisable repertoires. Unobserved rows are not uniform
      under the node estimators.
    - Subsystem: ``node_selection`` / ``node_indices`` and the state budget
      (``state_budget_policy``) are explicit; every adjustment is logged and
      returned (``budget_adjustments``, ``node_selection_degenerate``).

    Null calibration (``null_surrogates > 0``). IIM_raw is a scale-free ratio:
    for independent processes the estimated Psi is pure finite-sample bias,
    and cutting (which averages TPM rows) removes more of that bias than the
    intact TPM carries, so IIM_raw is O(1) even for independent noise and does
    not increase with coupling. The null statistic is therefore the integration
    mass Delta_Psi = Psi - max_kappa Psi^kappa (bits) of surrogates with
    independent nodes (default: independent circular shifts, which keep each
    node's marginal and autocorrelation; same subsystem, bins, lag and cut
    family). Reported: ``IIM_raw``, ``Delta_Psi``, ``Delta_Psi_null_mean/sd``,
    ``IIM_null_mean/sd`` (= Delta_Psi null moments / Psi, i.e. on the IIM_raw
    scale), ``IIM_z``, ``IIM_null_p`` (one-sided, (1+#null>=obs)/(n+1)),
    ``IIM_excess`` = Delta_Psi - mean null (bits) and the calibrated canonical
    value ``canonical_calibrated`` = ``IIM_calibrated`` = max(IIM_excess, 0),
    which is ~0 for independent nodes and increases with coupling strength.
    When calibration is on, the returned ``value`` is the calibrated canonical
    value (``IIM_excess`` if ``clamp=False``); ``raw``/``canonical`` keep their
    uncalibrated definitions for backward compatibility.

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_time)
        Timeseries matrix.
    bins : int, optional
        Target number of discrete levels per node. May be reduced to satisfy
        ``max_state_space`` according to ``state_budget_policy`` (logged).
    lag_trs : int, optional
        Lag (in samples) for transition construction.
    n_parts : int or None, optional
        Number of system cuts evaluated for MIP search. If ``None`` (default),
        all possible unique bipartitions are evaluated (exhaustive MIP search).
    rng : int or np.random.RandomState, optional
        Seed or RandomState used for sampled cuts.
    method : {'causal', 'random'}, optional
        Causal TPM method selector. ``'random'`` is accepted as legacy alias
        for ``'causal'``.
    partition_mode : {'all', 'balanced'}, optional
        Cut candidate regime. ``'balanced'`` restricts to near-equal cuts.
    clamp : bool, optional
        If True (default), return canonical IIM in [0,1]. If False, return raw.
    scale : float, optional
        Multiplicative factor applied to the returned metric value.
    max_nodes : int or None, optional
        Max number of nodes retained for causal TPM estimation. If ``None``,
        no explicit node-count cap is applied before the state-space budget
        enforcement.
    max_mechanism_size : int or None, optional
        Largest mechanism size included in Ψ. If ``None`` (default), all
        mechanism sizes up to the selected subsystem size are included.
    max_purview_size : int or None, optional
        Largest purview size included in Ψ. If ``None`` (default), all purview
        sizes up to the selected subsystem size are included.
    tpm_alpha : float, optional
        Laplace pseudo-count ('joint_laplace', 'node_laplace'; for
        'node_shrinkage' it only smooths the per-node shrinkage target).
    max_state_space : int, optional
        Upper bound on number of discrete system states used in TPM.
    return_details : bool, optional
        When True, return diagnostics dictionary.
    checkpoint_path : str or None, optional
        Path to JSON checkpoint file. When provided, per-cut progress is
        persisted and can be resumed on a later run.
    resume_from_checkpoint : bool, optional
        If True (default), reuse compatible checkpoint progress when
        ``checkpoint_path`` exists.
    checkpoint_every_cuts : int, optional
        Persist checkpoint after every N newly completed cuts (default 1).
    progress_log_every_cuts : int, optional
        Emit cut-level progress logs every N completed cuts (default 1).
    progress_label : str or None, optional
        Optional short label included in progress logs (e.g. run/file id).
    phase1_parallel_workers : int or None, optional
        Number of worker processes used for mechanism-chunk parallelization
        inside each Ψ computation (phase 1 and per-cut Ψ in phase 2). ``None``
        or values <= 1 disable intra-task parallelization.
    phase1_chunk_size : int, optional
        Number of mechanisms per submitted worker chunk when intra-task
        parallelization is enabled.
    phase1_shared_memory : bool, optional
        If True (default), stage read-only TPM/state arrays in shared memory
        for intra-task workers to minimize RAM duplication.
    use_induced_partition_cache : bool, optional
        Deprecated opt-out flag. Induced-partition caching is always enabled
        by default in this implementation and this flag is ignored.
    kernel_cache_path : str or None, optional
        SQLite path for disk-backed induced-partition cache. If omitted, a
        checkpoint-adjacent path (when checkpointing) or temp file is used.
    kernel_cache_memory_entries : int, optional
        In-memory LRU front-cache size for the kernel cache.
    kernel_cache_flush_batch : int, optional
        Number of pending inserts before batched SQLite flush.
    hardware_backend : optional
        Backend for TPM/cut-TPM construction (see ``hardware_backend``).
    tpm_estimator : {'node_shrinkage', 'node_laplace', 'joint_laplace'}, optional
        TPM estimator (see above). 'joint_laplace' reproduces pre-v4 values.
    node_selection : {'variance', 'index'}, optional
        Declared subsystem rule when more nodes than ``max_nodes`` are given:
        highest temporal variance (index tie-break; flagged as degenerate for
        standardised inputs) or the first nodes in input order.
    node_indices : sequence of int or None, optional
        Explicit subsystem (overrides ``node_selection``).
    state_budget_policy : {'reduce_bins_first', 'reduce_nodes_first', 'error'}
        How ``bins**nodes <= max_state_space`` is enforced; 'error' makes the
        result undefined instead of adjusting.
    null_surrogates : int, optional
        Number of surrogates for null calibration (0 = off, default).
    null_method : str, optional
        Surrogate type (see ``_surrogate_timeseries``), default 'circular_shift'.
    null_seed : int or None, optional
        Surrogate seed (default: ``rng`` when it is an int, else 0).
    null_min_shift : int or None, optional
        Minimum circular shift in samples (default max(lag+1, 10% of T)).
    keep_kernel_cache : bool, optional
        Keep an auto-created kernel cache after completion. By default temp
        caches are always removed and checkpoint-adjacent caches are removed
        once the checkpoint holds a terminal status; an explicit
        ``kernel_cache_path`` is never removed.

    Returns
    -------
    float or dict
        Scaled scalar value (default), or details when requested.
    """
    backend = resolve_hardware_backend(hardware_backend)

    # Parallel-execution state shared by all Psi evaluations of this run.
    # A broken reusable pool only disables the reusable pool (per-call pools are
    # still tried); a failed per-call pool disables intra-task parallelism.
    # Kernel-cache misses are not execution failures and never disable either.
    phase_parallel_runtime_enabled = True
    phase_reusable_runtime_enabled = True
    phase_parallel_fallbacks = []

    def _compute_psi(
        tpm,
        curr_obs,
        states_full,
        base,
        mech_size,
        purv_size,
        resume_done_mechanisms=0,
        resume_psi_partial=np.nan,
        progress_cb=None,
        phase1_parallel_workers=None,
        phase1_chunk_size=8,
        phase1_shared_memory=True,
        mechanisms=None,
        purviews=None,
        static_cache=None,
        obs_state_cache=None,
        parallel_runtime=None,
        kernel_cache=None,
        cut_mask_a=None,
        use_induced_partition_cache=False,
        kernel_cache_lookup_only=False,
    ):
        nonlocal phase_parallel_runtime_enabled
        nonlocal phase_reusable_runtime_enabled
        _, n_nodes = states_full.shape
        if mechanisms is None:
            all_nodes = tuple(range(n_nodes))
            mechanisms = tuple(_iim_enumerate_subsets(all_nodes, mech_size))
        else:
            mechanisms = tuple(tuple(sorted(m)) for m in mechanisms)
        if purviews is None:
            all_nodes = tuple(range(n_nodes))
            purviews = tuple(_iim_enumerate_subsets(all_nodes, purv_size))
        else:
            purviews = tuple(tuple(sorted(z)) for z in purviews)
        if not mechanisms or not purviews:
            return 0.0

        # empirical mechanism-state frequencies from observed current states
        try:
            done_start = int(resume_done_mechanisms)
        except (TypeError, ValueError):
            done_start = 0
        done_start = max(0, min(done_start, int(len(mechanisms))))
        try:
            psi = float(resume_psi_partial)
        except (TypeError, ValueError):
            psi = np.nan
        if not np.isfinite(psi):
            psi = 0.0
        n_mechanisms = int(len(mechanisms))
        if done_start >= n_mechanisms:
            if progress_cb is not None:
                progress_cb(int(n_mechanisms), int(n_mechanisms), float(psi))
            return float(psi)

        parallel_workers_eff = 1
        if phase1_parallel_workers is not None:
            try:
                parallel_workers_eff = max(1, int(phase1_parallel_workers))
            except (TypeError, ValueError):
                parallel_workers_eff = 1
        chunk_size_eff = max(1, int(phase1_chunk_size))
        remaining_mechanisms = mechanisms[done_start:]
        chunks = _iim_build_phase1_chunks_adaptive(
            remaining_mechanisms,
            max_chunk_size=chunk_size_eff,
            workers=parallel_workers_eff,
        )
        worker_cache_spec = None
        if bool(use_induced_partition_cache) and (kernel_cache is not None):
            cache_path = str(getattr(kernel_cache, "path", "")).strip()
            if cache_path:
                cache_mem_total = max(10_000, int(getattr(kernel_cache, "memory_entries", 100_000)))
                cache_flush_batch = max(100, int(getattr(kernel_cache, "flush_batch", 5_000)))
                per_worker_mem_entries = max(
                    10_000,
                    int(cache_mem_total // max(1, parallel_workers_eff)),
                )
                worker_cache_spec = {
                    "enabled": True,
                    "path": cache_path,
                    "memory_entries": int(per_worker_mem_entries),
                    "flush_batch": int(cache_flush_batch),
                }
        use_parallel = (
            phase_parallel_runtime_enabled
            and
            parallel_workers_eff > 1
            and len(remaining_mechanisms) > 1
        )
        # Psi accumulated before this call (resume prefix); any fallback restarts
        # from here so partially aggregated parallel chunks are never re-added.
        psi_start = float(psi)
        if (
            use_parallel
            and (parallel_runtime is not None)
            and phase_reusable_runtime_enabled
        ):
            try:
                return float(
                    parallel_runtime.run_chunks(
                        tpm=tpm,
                        chunks=chunks,
                        psi_start=float(psi_start),
                        done_start=int(done_start),
                        total_mechanisms=int(n_mechanisms),
                        progress_cb=progress_cb,
                        cache_spec=worker_cache_spec,
                        cut_mask_a=(None if cut_mask_a is None else int(cut_mask_a)),
                        use_induced_partition_cache=bool(use_induced_partition_cache),
                        kernel_cache_lookup_only=bool(kernel_cache_lookup_only),
                    )
                )
            except _IIMKernelCacheMissError:
                # A lookup-only cache miss is a data condition, not a pool failure:
                # the caller recomputes this cut; parallelism stays enabled.
                raise
            except Exception as exc:
                phase_reusable_runtime_enabled = False
                phase_parallel_fallbacks.append(
                    f"reusable_pool: {type(exc).__name__}: {exc}"
                )
                log.warning(
                    "[%s] IIM reusable phase workers failed (%s); falling back to per-call execution.",
                    run_label,
                    exc,
                )

        if use_parallel:
            parallel_workers_eff = min(parallel_workers_eff, len(remaining_mechanisms))
            owner_shms = []
            owner_tmp_dir = None
            owner_tmp_files = []
            try:
                def _mk_memmap_spec(label, arr):
                    nonlocal owner_tmp_dir
                    if owner_tmp_dir is None:
                        owner_tmp_dir = tempfile.mkdtemp(prefix="iim_phase1_")
                    path = os.path.join(owner_tmp_dir, f"{label}.npy")
                    np.save(path, np.ascontiguousarray(arr), allow_pickle=False)
                    owner_tmp_files.append(path)
                    return {
                        "mode": "memmap",
                        "path": str(path),
                    }

                def _mk_shm_spec(arr):
                    arr_c = np.ascontiguousarray(arr)
                    shm = shared_memory.SharedMemory(create=True, size=arr_c.nbytes)
                    shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
                    shm_arr[...] = arr_c
                    owner_shms.append(shm)
                    return {
                        "mode": "shared_memory",
                        "name": str(shm.name),
                        "shape": list(arr_c.shape),
                        "dtype": str(arr_c.dtype),
                    }

                try:
                    if not bool(phase1_shared_memory):
                        raise RuntimeError("phase1_shared_memory_disabled")
                    spec_tpm = _mk_shm_spec(tpm)
                    spec_curr = _mk_shm_spec(curr_obs)
                    spec_states = _mk_shm_spec(states_full)
                except Exception as shm_exc:
                    for shm in owner_shms:
                        try:
                            shm.close()
                        except Exception:
                            pass
                        try:
                            shm.unlink()
                        except Exception:
                            pass
                    owner_shms = []
                    log.warning(
                        "[%s] IIM phase intra-task shared-memory unavailable (%s); using read-only memmap fallback.",
                        run_label,
                        shm_exc,
                    )
                    spec_tpm = _mk_memmap_spec("tpm", tpm)
                    spec_curr = _mk_memmap_spec("curr_obs", curr_obs)
                    spec_states = _mk_memmap_spec("states_full", states_full)

                future_to_idx = {}
                completed = {}
                next_idx = 0
                done_abs = int(done_start)
                total_abs = int(n_mechanisms)
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=int(parallel_workers_eff),
                    initializer=_iim_phase_worker_init_static,
                    initargs=(spec_curr, spec_states, int(base), tuple(purviews)),
                ) as ex:
                    max_in_flight = max(1, int(parallel_workers_eff) * 2)
                    next_submit_idx = 0
                    n_chunks = int(len(chunks))

                    while next_submit_idx < min(n_chunks, max_in_flight):
                        idx = int(next_submit_idx)
                        chunk = chunks[idx]
                        fut = ex.submit(
                            _iim_phase_worker_run_chunk_for_tpm,
                            spec_tpm,
                            chunk,
                            worker_cache_spec,
                            (None if cut_mask_a is None else int(cut_mask_a)),
                            bool(use_induced_partition_cache),
                            bool(kernel_cache_lookup_only),
                        )
                        future_to_idx[fut] = idx
                        next_submit_idx += 1

                    while future_to_idx:
                        done_set, _ = concurrent.futures.wait(
                            tuple(future_to_idx.keys()),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for fut in done_set:
                            idx = int(future_to_idx.pop(fut))
                            chunk_psi, chunk_len = fut.result()
                            completed[idx] = (float(chunk_psi), int(chunk_len))
                        while (
                            next_submit_idx < n_chunks
                            and len(future_to_idx) < max_in_flight
                        ):
                            idx = int(next_submit_idx)
                            chunk = chunks[idx]
                            fut = ex.submit(
                                _iim_phase_worker_run_chunk_for_tpm,
                                spec_tpm,
                                chunk,
                                worker_cache_spec,
                                (None if cut_mask_a is None else int(cut_mask_a)),
                                bool(use_induced_partition_cache),
                                bool(kernel_cache_lookup_only),
                            )
                            future_to_idx[fut] = idx
                            next_submit_idx += 1
                        while next_idx in completed:
                            cpsi, clen = completed.pop(next_idx)
                            psi = float(math.fsum((float(psi), float(cpsi))))
                            done_abs += int(clen)
                            if progress_cb is not None:
                                progress_cb(int(done_abs), int(total_abs), float(psi))
                            next_idx += 1
                return float(psi)
            except _IIMKernelCacheMissError:
                raise
            except Exception as exc:
                phase_parallel_runtime_enabled = False
                phase_parallel_fallbacks.append(
                    f"per_call_pool: {type(exc).__name__}: {exc}"
                )
                log.warning(
                    "[%s] IIM intra-task phase parallelization failed (%s); falling back to sequential mechanisms.",
                    run_label,
                    exc,
                )
            finally:
                for shm in owner_shms:
                    try:
                        shm.close()
                    except Exception:
                        pass
                    try:
                        shm.unlink()
                    except Exception:
                        pass
                for p in owner_tmp_files:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                if owner_tmp_dir and os.path.isdir(owner_tmp_dir):
                    try:
                        shutil.rmtree(owner_tmp_dir, ignore_errors=True)
                    except Exception:
                        pass

        # Sequential path (also the fallback after a failed parallel attempt):
        # restart from the pre-call prefix so completed parallel chunks are not
        # counted twice.
        psi = float(psi_start)
        done_abs = int(done_start)
        total_abs = int(n_mechanisms)
        for chunk in chunks:
            chunk_psi = _iim_phase1_chunk_contribution(
                chunk,
                purviews,
                int(base),
                tpm,
                curr_obs,
                states_full,
                static_cache=static_cache,
                obs_state_cache=obs_state_cache,
                kernel_cache=kernel_cache,
                cut_mask_a=cut_mask_a,
                use_induced_partition_cache=bool(use_induced_partition_cache),
                kernel_cache_lookup_only=bool(kernel_cache_lookup_only),
            )
            chunk_len = int(len(chunk))
            psi = float(math.fsum((float(psi), float(chunk_psi))))
            done_abs += chunk_len
            if progress_cb is not None:
                progress_cb(int(done_abs), int(total_abs), float(psi))
        return float(psi)

    def _build_cut_tpm(tpm, states_full, base, A, B):
        return _iim_build_cut_tpm(
            tpm,
            states_full,
            base,
            A,
            B,
            hardware_backend=backend,
        )

    def _cut_to_key(A, B):
        return f"{','.join(map(str, A))}|{','.join(map(str, B))}"

    def _cut_from_payload(payload):
        if (
            isinstance(payload, (list, tuple))
            and len(payload) == 2
            and isinstance(payload[0], (list, tuple))
            and isinstance(payload[1], (list, tuple))
        ):
            return (tuple(int(x) for x in payload[0]), tuple(int(x) for x in payload[1]))
        return None

    # ---------- input checks ----------
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    n_regions, n_time = ts.shape
    if method not in {"causal", "random"}:
        raise ValueError("method must be 'causal' (or legacy alias 'random')")
    if partition_mode not in {"all", "balanced"}:
        raise ValueError("partition_mode must be 'all' or 'balanced'")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    if n_parts is not None and int(n_parts) < 1:
        raise ValueError("n_parts must be >= 1 or None for exhaustive search")
    if max_nodes is not None and int(max_nodes) < 2:
        raise ValueError("max_nodes must be >= 2 or None")
    if max_mechanism_size is not None and int(max_mechanism_size) < 1:
        raise ValueError("max_mechanism_size must be >= 1 or None")
    if max_purview_size is not None and int(max_purview_size) < 1:
        raise ValueError("max_purview_size must be >= 1 or None")
    if tpm_alpha <= 0:
        raise ValueError("tpm_alpha must be > 0")
    if max_state_space < 16:
        raise ValueError("max_state_space must be >= 16")
    if checkpoint_every_cuts < 1:
        raise ValueError("checkpoint_every_cuts must be >= 1")
    if progress_log_every_cuts < 1:
        raise ValueError("progress_log_every_cuts must be >= 1")
    if phase1_parallel_workers is not None and int(phase1_parallel_workers) < 1:
        raise ValueError("phase1_parallel_workers must be >= 1 or None")
    if int(phase1_chunk_size) < 1:
        raise ValueError("phase1_chunk_size must be >= 1")

    def _undefined_payload(reason, psi_full=np.nan, psi_mip=np.nan, extra=None):
        payload_extra = extra or {}
        if return_details:
            out = {
                "value": np.nan,
                "raw": np.nan,
                "canonical": np.nan,
                "clipped": np.nan,  # legacy alias
                "iim_plus": np.nan,  # legacy alias
                "scale": scale,
                "I_full": float(psi_full),  # legacy field name
                "min_partition_sum": float(psi_mip),  # legacy field name
                "Psi_full": float(psi_full),
                "Psi_mip_preserved": float(psi_mip),
                "defined": False,
                "undefined_reason": reason,
                "iim_algorithm_version": IIM_ALGORITHM_VERSION,
                "tpm_estimator": str(tpm_estimator),
            }
            # Same null-calibration schema as defined results (no surrogates are
            # run when the observed IIM itself is undefined).
            out.update(
                _iim_null_fields(
                    None,
                    psi_full=np.nan,
                    delta_psi=np.nan,
                    meta={
                        "IIM_null_method": str(null_method),
                        "IIM_null_seed": _resolve_null_seed(null_seed, rng),
                        "IIM_null_min_shift": (
                            None if null_min_shift is None else int(null_min_shift)
                        ),
                        "IIM_null_failed": 0,
                        "IIM_null_undefined_reason": (
                            "observed_iim_undefined"
                            if int(null_surrogates) > 0
                            else None
                        ),
                    },
                )
            )
            out.update(payload_extra)
            return out
        return np.nan

    run_label = str(progress_label).strip() if progress_label is not None else ""
    if not run_label:
        run_label = "IIM"
    if not bool(use_induced_partition_cache):
        log.info(
            "[%s] IIM induced-partition cache opt-out is deprecated; keeping cache enabled.",
            run_label,
        )
    use_induced_partition_cache = True

    # not enough regions or timepoints
    if n_regions < 2 or n_time - lag_trs < 1:
        return _undefined_payload(
            "insufficient_shape",
            extra={"checkpoint_path": checkpoint_path, "checkpoint_resumed": False},
        )

    # set up random generator
    if isinstance(rng, np.random.RandomState):
        rand_state = rng
    else:
        rand_state = np.random.RandomState(rng)

    # legacy alias
    _ = method
    if int(null_surrogates) < 0:
        raise ValueError("null_surrogates must be >= 0")
    if null_method not in SURROGATE_METHODS:
        raise ValueError(f"null_method must be one of {SURROGATE_METHODS}")
    # Snapshot so surrogate runs evaluate exactly the same (possibly sampled) cuts.
    cut_rng_state = rand_state.get_state()

    prep = prepare_iim_problem(
        ts,
        bins=int(bins),
        lag_trs=int(lag_trs),
        n_parts=n_parts,
        rng=rand_state,
        partition_mode=str(partition_mode),
        max_nodes=max_nodes,
        max_mechanism_size=max_mechanism_size,
        max_purview_size=max_purview_size,
        tpm_alpha=float(tpm_alpha),
        max_state_space=int(max_state_space),
        hardware_backend=backend,
        tpm_estimator=str(tpm_estimator),
        node_selection=str(node_selection),
        node_indices=node_indices,
        state_budget_policy=str(state_budget_policy),
        log_label=run_label,
    )
    if not bool(prep.get("defined", False)):
        return _undefined_payload(
            str(prep.get("undefined_reason", "iim_prepare_failed")),
            extra={
                "n_nodes_used": prep.get("n_nodes_used"),
                "bins_used": prep.get("bins_used"),
                "bins_requested": prep.get("bins_requested"),
                "budget_adjustments": prep.get("budget_adjustments"),
            },
        )

    selected = np.asarray(prep["selected_nodes"], dtype=int)
    selected_nodes_list = [int(x) for x in selected.tolist()]
    ts_sel = np.asarray(prep["ts_selected"], dtype=float)
    disc = np.asarray(prep["disc"], dtype=np.int16)
    curr_obs = np.asarray(prep["curr_obs"], dtype=np.int16)
    tpm_full = np.asarray(prep["tpm_full"], dtype=float)
    states_full = np.asarray(prep["states_full"], dtype=np.int16)
    eff_bins = int(prep["bins_used"])
    n_sel = int(prep["n_nodes_used"])
    mech_size_eff = int(prep["max_mechanism_size_used"])
    purv_size_eff = int(prep["max_purview_size_used"])
    mechanisms_all = tuple(tuple(m) for m in prep["mechanisms_all"])
    purviews_all = tuple(tuple(z) for z in prep["purviews_all"])
    # Static structures reused for all Ψ evaluations in this run (phase-1 + all cuts).
    psi_static_cache = {}
    psi_obs_state_cache = {}

    cuts_eval = [tuple(x) for x in prep["cuts_eval"]]
    cut_key_to_cut = {_cut_to_key(A, B): (A, B) for A, B in cuts_eval}
    cuts_payload = [list(x) for x in prep["cuts_payload"]]

    class _ReusablePhaseParallelRuntime:
        def __init__(
            self,
            workers,
            base,
            purviews,
            curr_obs,
            states_full,
            use_shared_memory,
            run_label,
        ):
            self.workers = int(max(1, workers))
            self.base = int(base)
            self.purviews = tuple(tuple(z) for z in purviews)
            self.use_shared_memory = bool(use_shared_memory)
            self.run_label = str(run_label)
            self.static_shms = []
            self.static_files = []
            self.tmp_dir = tempfile.mkdtemp(prefix="iim_phase_runtime_")
            self.executor = None

            spec_curr = self._mk_static_spec("curr_obs", curr_obs)
            spec_states = self._mk_static_spec("states_full", states_full)
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_iim_phase_worker_init_static,
                initargs=(spec_curr, spec_states, self.base, self.purviews),
            )

        def _mk_memmap_spec(self, label, arr, remember_static):
            path = os.path.join(self.tmp_dir, f"{label}_{os.getpid()}_{time.time_ns()}.npy")
            np.save(path, np.ascontiguousarray(arr), allow_pickle=False)
            if remember_static:
                self.static_files.append(path)
            return {"mode": "memmap", "path": str(path)}, None, path

        def _mk_shm_spec(self, arr, remember_static):
            arr_c = np.ascontiguousarray(arr)
            shm = shared_memory.SharedMemory(create=True, size=int(arr_c.nbytes))
            shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
            shm_arr[...] = arr_c
            if remember_static:
                self.static_shms.append(shm)
            return {
                "mode": "shared_memory",
                "name": str(shm.name),
                "shape": list(arr_c.shape),
                "dtype": str(arr_c.dtype),
            }, shm, None

        def _mk_static_spec(self, label, arr):
            if self.use_shared_memory:
                try:
                    spec, _, _ = self._mk_shm_spec(arr, remember_static=True)
                    return spec
                except Exception as exc:
                    log.warning(
                        "[%s] IIM reusable workers: shared-memory unavailable for %s (%s); using memmap.",
                        self.run_label,
                        label,
                        exc,
                    )
            spec, _, _ = self._mk_memmap_spec(label, arr, remember_static=True)
            return spec

        def _mk_tpm_spec(self, tpm):
            if self.use_shared_memory:
                try:
                    return self._mk_shm_spec(tpm, remember_static=False)
                except Exception as exc:
                    log.warning(
                        "[%s] IIM reusable workers: shared-memory unavailable for cut TPM (%s); using memmap.",
                        self.run_label,
                        exc,
                    )
            return self._mk_memmap_spec("tpm_cut", tpm, remember_static=False)

        @staticmethod
        def _cleanup_tpm_spec(spec_shm, spec_path):
            if spec_shm is not None:
                try:
                    spec_shm.close()
                except Exception:
                    pass
                try:
                    spec_shm.unlink()
                except Exception:
                    pass
            if spec_path and os.path.exists(spec_path):
                try:
                    os.remove(spec_path)
                except Exception:
                    pass

        def run_chunks(
            self,
            tpm,
            chunks,
            psi_start,
            done_start,
            total_mechanisms,
            progress_cb,
            cache_spec=None,
            cut_mask_a=None,
            use_induced_partition_cache=False,
            kernel_cache_lookup_only=False,
        ):
            if self.executor is None:
                raise RuntimeError("Reusable phase runtime is not initialized.")
            if not chunks:
                if progress_cb is not None:
                    progress_cb(int(total_mechanisms), int(total_mechanisms), float(psi_start))
                return float(psi_start)

            spec_tpm, tpm_shm, tpm_path = self._mk_tpm_spec(tpm)
            psi = float(psi_start)
            done_abs = int(done_start)
            total_abs = int(total_mechanisms)
            future_to_idx = {}
            completed = {}
            next_idx = 0
            try:
                max_in_flight = max(1, int(self.workers) * 2)
                next_submit_idx = 0
                n_chunks = int(len(chunks))

                while next_submit_idx < min(n_chunks, max_in_flight):
                    idx = int(next_submit_idx)
                    chunk = chunks[idx]
                    fut = self.executor.submit(
                        _iim_phase_worker_run_chunk_for_tpm,
                        spec_tpm,
                        chunk,
                        cache_spec,
                        cut_mask_a,
                        bool(use_induced_partition_cache),
                        bool(kernel_cache_lookup_only),
                    )
                    future_to_idx[fut] = idx
                    next_submit_idx += 1

                while future_to_idx:
                    done_set, _ = concurrent.futures.wait(
                        tuple(future_to_idx.keys()),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for fut in done_set:
                        idx = int(future_to_idx.pop(fut))
                        chunk_psi, chunk_len = fut.result()
                        completed[idx] = (float(chunk_psi), int(chunk_len))
                    while (
                        next_submit_idx < n_chunks
                        and len(future_to_idx) < max_in_flight
                    ):
                        idx = int(next_submit_idx)
                        chunk = chunks[idx]
                        fut = self.executor.submit(
                            _iim_phase_worker_run_chunk_for_tpm,
                            spec_tpm,
                            chunk,
                            cache_spec,
                            cut_mask_a,
                            bool(use_induced_partition_cache),
                            bool(kernel_cache_lookup_only),
                        )
                        future_to_idx[fut] = idx
                        next_submit_idx += 1
                    while next_idx in completed:
                        cpsi, clen = completed.pop(next_idx)
                        psi = float(math.fsum((float(psi), float(cpsi))))
                        done_abs += int(clen)
                        if progress_cb is not None:
                            progress_cb(int(done_abs), int(total_abs), float(psi))
                        next_idx += 1
                return float(psi)
            finally:
                # The pool outlives an aborted call (e.g. a lookup-only cache
                # miss), so drop its queued chunks instead of computing them.
                for fut in future_to_idx:
                    fut.cancel()
                self._cleanup_tpm_spec(tpm_shm, tpm_path)

        def close(self):
            if self.executor is not None:
                try:
                    self.executor.shutdown(wait=True, cancel_futures=False)
                except Exception:
                    pass
                self.executor = None
            for shm in self.static_shms:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass
            self.static_shms = []
            for path in self.static_files:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            self.static_files = []
            if self.tmp_dir and os.path.isdir(self.tmp_dir):
                try:
                    shutil.rmtree(self.tmp_dir, ignore_errors=True)
                except Exception:
                    pass
                self.tmp_dir = None

    phase_parallel_runtime = None
    try:
        phase1_parallel_workers_eff = 1
        if phase1_parallel_workers is not None:
            try:
                phase1_parallel_workers_eff = max(1, int(phase1_parallel_workers))
            except (TypeError, ValueError):
                phase1_parallel_workers_eff = 1
        if phase1_parallel_workers_eff > 1 and len(mechanisms_all) > 1:
            try:
                phase_parallel_runtime = _ReusablePhaseParallelRuntime(
                    workers=phase1_parallel_workers_eff,
                    base=eff_bins,
                    purviews=purviews_all,
                    curr_obs=curr_obs,
                    states_full=states_full,
                    use_shared_memory=bool(phase1_shared_memory),
                    run_label=run_label,
                )
            except Exception as exc:
                phase_parallel_runtime = None
                log.warning(
                    "[%s] IIM reusable phase-worker pool init failed (%s); using per-call parallelization.",
                    run_label,
                    exc,
                )
    except Exception:
        phase_parallel_runtime = None

    def _close_phase_parallel_runtime():
        nonlocal phase_parallel_runtime
        if phase_parallel_runtime is not None:
            phase_parallel_runtime.close()
            phase_parallel_runtime = None

    checkpoint_resumed = False
    checkpoint_reused_cuts = 0
    checkpoint_used_psi_full = False
    phase1_mechanisms_done = 0
    phase1_total_mechanisms = None
    phase1_eta_seconds = None
    phase1_psi_partial = np.nan
    phase1_resumed_partial = False
    iim_phase = "init"
    phase2_mode = "unknown"
    materialization_done_mechanisms = 0
    materialization_total_mechanisms = None
    materialization_est_scale = None
    materialization_key_checks = 0
    materialization_missing_groups = 0
    materialization_computed_groups = 0
    completed_cut_scores = {}
    psi_preserved_max = -np.inf
    mip_cut = None
    psi_full = np.nan

    # Everything that changes Psi / cut scores / kernel values must be in the
    # signature, otherwise a resume (or a reused kernel cache) silently mixes
    # results from different estimators or code versions.
    try:
        disc_view = memoryview(np.ascontiguousarray(disc))
        disc_hash = hashlib.sha1(disc_view).hexdigest()
    except Exception:
        disc_hash = None
    cuts_hash = hashlib.sha1(
        json.dumps(cuts_payload, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    signature = {
        "iim_algorithm_version": IIM_ALGORITHM_VERSION,
        "n_regions_input": int(n_regions),
        "n_time_input": int(n_time),
        "n_nodes_used": int(n_sel),
        "selected_nodes": [int(x) for x in selected.tolist()],
        "node_selection_rule": str(prep.get("node_selection_rule")),
        "bins_requested": int(bins),
        "bins_used": int(eff_bins),
        "lag_trs": int(lag_trs),
        "tpm_estimator": str(tpm_estimator),
        "tpm_alpha": float(tpm_alpha),
        "max_state_space": int(max_state_space),
        "state_budget_policy": str(state_budget_policy),
        "partition_mode": str(partition_mode),
        "n_parts_requested": (None if n_parts is None else int(n_parts)),
        "max_nodes_requested": (None if max_nodes is None else int(max_nodes)),
        "max_mechanism_size_used": int(mech_size_eff),
        "max_purview_size_used": int(purv_size_eff),
        "n_cuts_total": int(len(cuts_eval)),
        "cuts_hash": cuts_hash,
        "disc_hash": disc_hash,
    }
    if checkpoint_path:
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                if ckpt.get("signature") == signature:
                    checkpoint_resumed = True
                    psi_ck = ckpt.get("psi_full")
                    if psi_ck is not None:
                        try:
                            psi_ck = float(psi_ck)
                        except (TypeError, ValueError):
                            psi_ck = np.nan
                        if np.isfinite(psi_ck):
                            psi_full = psi_ck
                            checkpoint_used_psi_full = True

                    ck_completed = ckpt.get("completed_cuts", {})
                    if isinstance(ck_completed, dict):
                        for k, v in ck_completed.items():
                            if k not in cut_key_to_cut:
                                continue
                            try:
                                v = float(v)
                            except (TypeError, ValueError):
                                continue
                            if not np.isfinite(v):
                                continue
                            completed_cut_scores[k] = v
                        checkpoint_reused_cuts = int(len(completed_cut_scores))

                    ph1_done = ckpt.get("phase1_mechanisms_done")
                    ph1_total = ckpt.get("phase1_total_mechanisms")
                    ph1_psi = ckpt.get("phase1_psi_partial")
                    try:
                        ph1_done = int(ph1_done) if ph1_done is not None else 0
                    except (TypeError, ValueError):
                        ph1_done = 0
                    try:
                        ph1_total = int(ph1_total) if ph1_total is not None else None
                    except (TypeError, ValueError):
                        ph1_total = None
                    try:
                        ph1_psi = float(ph1_psi) if ph1_psi is not None else np.nan
                    except (TypeError, ValueError):
                        ph1_psi = np.nan
                    if ph1_total is not None and ph1_total > 0:
                        phase1_mechanisms_done = max(0, min(ph1_done, ph1_total))
                        phase1_total_mechanisms = ph1_total
                    if np.isfinite(ph1_psi):
                        phase1_psi_partial = float(ph1_psi)
                    if checkpoint_used_psi_full and np.isfinite(psi_full):
                        phase1_psi_partial = float(psi_full)

                    best_block = ckpt.get("best", {})
                    if isinstance(best_block, dict):
                        best_v = best_block.get("psi_preserved_max")
                        try:
                            best_v = float(best_v)
                        except (TypeError, ValueError):
                            best_v = -np.inf
                        if np.isfinite(best_v):
                            psi_preserved_max = float(best_v)
                            cut_payload = _cut_from_payload(best_block.get("mip_cut"))
                            if cut_payload is not None:
                                mip_cut = cut_payload

                    if not np.isfinite(psi_preserved_max):
                        psi_preserved_max = -np.inf
                    if (mip_cut is None) or (not np.isfinite(psi_preserved_max)):
                        for ck, cv in completed_cut_scores.items():
                            if cv > psi_preserved_max:
                                psi_preserved_max = cv
                                mip_cut = cut_key_to_cut.get(ck)

                    log.info(
                        "[%s] IIM checkpoint resume: reused_cuts=%d/%d, reused_psi_full=%s, file=%s",
                        run_label,
                        checkpoint_reused_cuts,
                        len(cuts_eval),
                        checkpoint_used_psi_full,
                        checkpoint_path,
                    )
                    if (
                        (not checkpoint_used_psi_full)
                        and (phase1_total_mechanisms is not None)
                        and (phase1_mechanisms_done > 0)
                    ):
                        if np.isfinite(phase1_psi_partial):
                            phase1_resumed_partial = True
                            log.info(
                                "[%s] IIM checkpoint partial phase-1 resume prepared: %d/%d mechanisms, psi_partial=%.6f.",
                                run_label,
                                int(phase1_mechanisms_done),
                                int(phase1_total_mechanisms),
                                float(phase1_psi_partial),
                            )
                        else:
                            log.info(
                                "[%s] IIM checkpoint has partial phase-1 progress %d/%d but no psi_partial; restarting phase-1 from scratch.",
                                run_label,
                                int(phase1_mechanisms_done),
                                int(phase1_total_mechanisms),
                            )
                            phase1_mechanisms_done = 0
                            phase1_total_mechanisms = None
                else:
                    log.warning(
                        "[%s] IIM checkpoint signature mismatch, starting fresh: %s",
                        run_label,
                        checkpoint_path,
                    )
            except Exception as exc:
                log.warning(
                    "[%s] IIM checkpoint load failed (%s), starting fresh: %s",
                    run_label,
                    exc,
                    checkpoint_path,
                )

    induced_kernel_cache = None
    induced_kernel_cache_path_eff = None
    # Cache file ownership: an explicit kernel_cache_path belongs to the caller
    # and is kept; a checkpoint-adjacent cache is kept only while the run is
    # resumable (deleted once the checkpoint holds a terminal status); an
    # auto-created temp cache is always deleted (also on abnormal exit).
    if kernel_cache_path is not None:
        induced_kernel_cache_path_eff = str(kernel_cache_path)
        kernel_cache_disposal = "keep"
    elif checkpoint_path:
        induced_kernel_cache_path_eff = f"{checkpoint_path}.kernel.sqlite3"
        kernel_cache_disposal = "keep" if bool(keep_kernel_cache) else "on_terminal"
    else:
        induced_kernel_cache_path_eff = os.path.join(
            tempfile.gettempdir(),
            f"iim_kernel_cache_{os.getpid()}_{time.time_ns()}.sqlite3",
        )
        kernel_cache_disposal = "keep" if bool(keep_kernel_cache) else "always"
    kernel_cache_finalizer = None
    try:
        induced_kernel_cache = _IIMDiskKernelCache(
            induced_kernel_cache_path_eff,
            signature=signature,
            memory_entries=int(kernel_cache_memory_entries),
            flush_batch=int(kernel_cache_flush_batch),
        )
        if kernel_cache_disposal == "always":
            kernel_cache_finalizer = weakref.finalize(
                induced_kernel_cache,
                _iim_remove_sqlite_files,
                induced_kernel_cache_path_eff,
            )
        log.info(
            "[%s] IIM induced-partition cache: enabled path=%s mem_entries=%d "
            "flush_batch=%d disposal=%s",
            run_label,
            induced_kernel_cache_path_eff,
            int(kernel_cache_memory_entries),
            int(kernel_cache_flush_batch),
            kernel_cache_disposal,
        )
    except Exception as exc:
        induced_kernel_cache = None
        log.warning(
            "[%s] IIM induced-partition cache init failed (%s); continuing without disk-backed cache.",
            run_label,
            exc,
        )

    def _close_induced_cache(terminal=True):
        nonlocal induced_kernel_cache
        if induced_kernel_cache is not None:
            try:
                induced_kernel_cache.close()
            except Exception:
                pass
            induced_kernel_cache = None
        if kernel_cache_disposal == "always" or (
            kernel_cache_disposal == "on_terminal" and terminal
        ):
            _iim_remove_sqlite_files(induced_kernel_cache_path_eff)
            if kernel_cache_finalizer is not None:
                kernel_cache_finalizer.detach()

    def _write_checkpoint(status, undefined_reason=None):
        if not checkpoint_path:
            return
        payload = {
            "version": 4,
            "status": status,
            "updated_unix": float(time.time()),
            "signature": signature,
            "iim_phase": str(iim_phase),
            "phase2_mode": str(phase2_mode),
            "psi_full": (None if not np.isfinite(psi_full) else float(psi_full)),
            "completed_cuts": {k: float(v) for k, v in completed_cut_scores.items()},
            "best": {
                "psi_preserved_max": (
                    None
                    if not np.isfinite(psi_preserved_max)
                    else float(psi_preserved_max)
                ),
                "mip_cut": (
                    None if mip_cut is None else [list(mip_cut[0]), list(mip_cut[1])]
                ),
            },
            "n_cuts_total": int(len(cuts_eval)),
            "undefined_reason": undefined_reason,
            "phase1_mechanisms_done": int(phase1_mechanisms_done),
            "phase1_total_mechanisms": (
                None
                if phase1_total_mechanisms is None
                else int(phase1_total_mechanisms)
            ),
            "phase1_eta_seconds": (
                None if phase1_eta_seconds is None else float(phase1_eta_seconds)
            ),
            "phase1_psi_partial": (
                None
                if not np.isfinite(phase1_psi_partial)
                else float(phase1_psi_partial)
            ),
            "phase1_resumed_partial": bool(phase1_resumed_partial),
            "materialization_done_mechanisms": int(materialization_done_mechanisms),
            "materialization_total_mechanisms": (
                None
                if materialization_total_mechanisms is None
                else int(materialization_total_mechanisms)
            ),
            "materialization_est_scale": (
                None
                if materialization_est_scale is None
                else int(materialization_est_scale)
            ),
            "materialization_key_checks": int(materialization_key_checks),
            "materialization_missing_groups": int(materialization_missing_groups),
            "materialization_computed_groups": int(materialization_computed_groups),
        }
        tmp_path = f"{checkpoint_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, sort_keys=True)
            os.replace(tmp_path, checkpoint_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    if not np.isfinite(psi_full):
        iim_phase = "phase1_psi"
        resume_done_mechanisms = 0
        resume_psi_for_compute = np.nan
        if (
            phase1_resumed_partial
            and (phase1_total_mechanisms is not None)
            and np.isfinite(phase1_psi_partial)
            and (phase1_mechanisms_done > 0)
        ):
            resume_done_mechanisms = int(phase1_mechanisms_done)
            resume_psi_for_compute = float(phase1_psi_partial)
        else:
            phase1_mechanisms_done = 0
            phase1_total_mechanisms = None
            phase1_eta_seconds = None
            phase1_psi_partial = np.nan

        log.info(
            "[%s] IIM phase 1/2: computing Psi_full (nodes=%d, bins=%d, cuts=%d, resume_done=%d)",
            run_label,
            int(n_sel),
            int(eff_bins),
            int(len(cuts_eval)),
            int(resume_done_mechanisms),
        )
        phase1_log_step_pct = 5.0
        if phase1_total_mechanisms is not None and int(phase1_total_mechanisms) > 0:
            phase1_next_log_pct = (
                100.0 * float(phase1_mechanisms_done) / float(int(phase1_total_mechanisms))
            )
        else:
            phase1_next_log_pct = 0.0
        phase1_last_logged_done = -1
        phase1_started_at = time.time()
        phase1_start_done = int(resume_done_mechanisms)

        def _phase1_progress_cb(done, total, psi_partial):
            nonlocal phase1_mechanisms_done
            nonlocal phase1_total_mechanisms
            nonlocal phase1_eta_seconds
            nonlocal phase1_psi_partial
            nonlocal phase1_next_log_pct
            nonlocal phase1_last_logged_done
            phase1_mechanisms_done = int(done)
            phase1_total_mechanisms = int(total) if total is not None else None
            phase1_psi_partial = float(psi_partial) if np.isfinite(psi_partial) else np.nan
            if total is None or int(total) <= 0:
                return
            total_i = int(total)
            done_i = int(done)
            pct = 100.0 * float(done_i) / float(total_i)
            should_log = (done_i == total_i) or (pct >= phase1_next_log_pct)
            if should_log and done_i != phase1_last_logged_done:
                elapsed = max(float(time.time() - phase1_started_at), 1e-9)
                done_delta = max(0, int(done_i - phase1_start_done))
                rate = float(done_delta) / elapsed
                if rate > 0:
                    eta = max(float(total_i - done_i) / rate, 0.0)
                    phase1_eta_seconds = float(eta)
                else:
                    eta = np.nan
                    phase1_eta_seconds = None
                eta_txt = "na" if not np.isfinite(eta) else f"{eta:.1f}s"
                log.info(
                    "[%s] IIM phase 1/2 progress: mechanisms=%d/%d (%.1f%%), eta=%s, psi_partial=%.6f",
                    run_label,
                    done_i,
                    total_i,
                    pct,
                    eta_txt,
                    float(psi_partial),
                )
                if checkpoint_path:
                    _write_checkpoint("running")
                phase1_last_logged_done = done_i
                while phase1_next_log_pct <= pct:
                    phase1_next_log_pct += phase1_log_step_pct

        psi_full = _compute_psi(
            tpm_full,
            curr_obs,
            states_full,
            eff_bins,
            mech_size_eff,
            purv_size_eff,
            resume_done_mechanisms=resume_done_mechanisms,
            resume_psi_partial=resume_psi_for_compute,
            progress_cb=_phase1_progress_cb,
            phase1_parallel_workers=phase1_parallel_workers,
            phase1_chunk_size=phase1_chunk_size,
            phase1_shared_memory=phase1_shared_memory,
            mechanisms=mechanisms_all,
            purviews=purviews_all,
            static_cache=psi_static_cache,
            obs_state_cache=psi_obs_state_cache,
            parallel_runtime=phase_parallel_runtime,
            kernel_cache=induced_kernel_cache,
            cut_mask_a=None,
            use_induced_partition_cache=bool(induced_kernel_cache is not None),
        )
        phase1_psi_partial = float(psi_full) if np.isfinite(psi_full) else np.nan
        _write_checkpoint("running")
    else:
        iim_phase = "phase1_done"
        log.info(
            "[%s] IIM phase 1/2: Psi_full loaded from checkpoint (nodes=%d, bins=%d, cuts=%d)",
            run_label,
            int(n_sel),
            int(eff_bins),
            int(len(cuts_eval)),
        )

    if not np.isfinite(psi_full) or psi_full <= 0:
        iim_phase = "undefined"
        _write_checkpoint("undefined", undefined_reason="nonpositive_psi_full")
        _close_phase_parallel_runtime()
        _close_induced_cache()
        return _undefined_payload(
            "nonpositive_psi_full",
            psi_full=float(psi_full) if np.isfinite(psi_full) else np.nan,
            extra={
                "n_nodes_used": int(n_sel),
                "bins_used": int(eff_bins),
                "checkpoint_path": checkpoint_path,
                "checkpoint_resumed": bool(checkpoint_resumed),
                "checkpoint_reused_cuts": int(checkpoint_reused_cuts),
                "checkpoint_used_psi_full": bool(checkpoint_used_psi_full),
                "phase1_resumed_partial": bool(phase1_resumed_partial),
                "phase1_psi_partial": (
                    float(phase1_psi_partial) if np.isfinite(phase1_psi_partial) else np.nan
                ),
            },
        )

    def _materialize_induced_kernel_cases():
        """
        Precompute unique kernel values keyed by induced unlabeled partition (pi),
        then allow cut-stage aggregation to run in lookup-only mode.

        Returns
        -------
        bool
            True when materialization ran successfully and lookup-only aggregation
            should be used; False to keep legacy per-cut recomputation behavior.
        """
        if induced_kernel_cache is None:
            return False
        if len(cuts_eval) <= 1:
            return False
        if all(_cut_to_key(A, B) in completed_cut_scores for A, B in cuts_eval):
            # Resumed run with every cut already scored: nothing to materialize.
            return False

        n_nodes_sys = int(states_full.shape[1])
        all_nodes_sys = tuple(range(n_nodes_sys))

        def _subset_mask_fast(subset):
            m = 0
            for nn in subset:
                m |= (1 << int(nn))
            return int(m)

        def _induced_pi_from_masks(m_mask, z_mask, cut_mask):
            u_mask = int(m_mask) | int(z_mask)
            part_a = int(u_mask & int(cut_mask))
            part_b = int(u_mask ^ part_a)
            if part_a == 0 or part_b == 0:
                return 0
            return int(part_a if part_a < part_b else part_b)

        def _mask_to_cut(mask):
            A = tuple(nn for nn in all_nodes_sys if ((int(mask) >> int(nn)) & 1))
            if len(A) == 0 or len(A) == n_nodes_sys:
                return None
            B = tuple(nn for nn in all_nodes_sys if not ((int(mask) >> int(nn)) & 1))
            return (A, B)

        cut_masks_eval = []
        for A, _B in cuts_eval:
            cm = 0
            for nn in A:
                cm |= (1 << int(nn))
            cut_masks_eval.append(int(cm))

        mechanism_entries = []
        for M in mechanisms_all:
            M = tuple(sorted(M))
            if len(M) < 2:
                continue
            obs_keys = _iim_subset_key_matrix(curr_obs, M, eff_bins)
            if obs_keys.size == 0:
                continue
            uk = np.unique(obs_keys).astype(np.int64, copy=False)
            if uk.size == 0:
                continue
            mechanism_entries.append((M, _subset_mask_fast(M), uk))

        purview_entries = []
        for Z in purviews_all:
            Z = tuple(sorted(Z))
            if len(Z) < 2:
                continue
            purview_entries.append((Z, _subset_mask_fast(Z)))

        if not mechanism_entries or not purview_entries:
            return False

        nonlocal iim_phase
        nonlocal phase2_mode
        nonlocal materialization_done_mechanisms
        nonlocal materialization_total_mechanisms
        nonlocal materialization_est_scale
        nonlocal materialization_key_checks
        nonlocal materialization_missing_groups
        nonlocal materialization_computed_groups

        est_scale = int(len(mechanism_entries)) * int(len(purview_entries)) * int(len(cuts_eval))
        iim_phase = "phase2_materialize"
        phase2_mode = "lookup_only"
        materialization_done_mechanisms = 0
        materialization_total_mechanisms = int(len(mechanism_entries))
        materialization_est_scale = int(est_scale)
        materialization_key_checks = 0
        materialization_missing_groups = 0
        materialization_computed_groups = 0

        log.info(
            "[%s] IIM phase 2 prep: materializing induced kernel cases (mechanisms=%d, purviews=%d, cuts=%d, est_scale=%d).",
            run_label,
            int(len(mechanism_entries)),
            int(len(purview_entries)),
            int(len(cuts_eval)),
            int(est_scale),
        )

        t0 = time.time()
        tpm_by_pi = OrderedDict()
        tpm_by_pi[0] = tpm_full
        max_cached_pi_tpms = 4

        total_directional_keys = 0
        missing_groups = 0
        computed_groups = 0

        for m_idx, (M, m_mask, uk) in enumerate(mechanism_entries, start=1):
            materialization_done_mechanisms = int(m_idx)
            for Z, z_mask in purview_entries:
                pis = set()
                for cut_mask in cut_masks_eval:
                    pis.add(_induced_pi_from_masks(m_mask, z_mask, cut_mask))
                if not pis:
                    continue

                for pi in pis:
                    pi = int(pi)
                    if pi == 0:
                        tpm_ref = tpm_full
                    else:
                        if pi in tpm_by_pi:
                            tpm_ref = tpm_by_pi[pi]
                            tpm_by_pi.move_to_end(pi, last=True)
                        else:
                            cut_pair = _mask_to_cut(pi)
                            if cut_pair is None:
                                tpm_ref = tpm_full
                            else:
                                tpm_ref = _build_cut_tpm(
                                    tpm_full,
                                    states_full,
                                    eff_bins,
                                    cut_pair[0],
                                    cut_pair[1],
                                )
                            tpm_by_pi[pi] = tpm_ref
                            while (len(tpm_by_pi) - (1 if 0 in tpm_by_pi else 0)) > int(max_cached_pi_tpms):
                                old_nonzero = None
                                for k_tmp in tpm_by_pi.keys():
                                    if int(k_tmp) != 0:
                                        old_nonzero = int(k_tmp)
                                        break
                                if old_nonzero is None:
                                    break
                                tpm_by_pi.pop(old_nonzero, None)

                    cut_mask_for_key = int(pi)

                    missing_rows = []
                    for m_key in uk:
                        mk = int(m_key)
                        total_directional_keys += 2
                        materialization_key_checks = int(total_directional_keys)
                        key_e = (0, int(m_mask), int(mk), int(z_mask), int(pi))
                        key_c = (1, int(m_mask), int(mk), int(z_mask), int(pi))
                        v_e = induced_kernel_cache.get(key_e)
                        v_c = induced_kernel_cache.get(key_c)
                        if (v_e is not None) and (v_c is not None):
                            continue

                        missing_groups += 1
                        materialization_missing_groups = int(missing_groups)
                        obs_row = np.zeros(n_nodes_sys, dtype=np.int16)
                        m_vals = _iim_decode_key(int(mk), len(M), eff_bins)
                        for p, nn in enumerate(M):
                            obs_row[int(nn)] = int(m_vals[p])
                        missing_rows.append(obs_row)

                    if missing_rows:
                        # One call per (M, Z, pi) group: all missing mechanism states
                        # share the call's repertoire caches. Each state's kernel value
                        # is written to the cache independently of the others.
                        _iim_phase1_chunk_contribution(
                            (M,),
                            (Z,),
                            int(eff_bins),
                            tpm_ref,
                            np.asarray(missing_rows, dtype=np.int16),
                            states_full,
                            static_cache=psi_static_cache,
                            obs_state_cache=None,
                            kernel_cache=induced_kernel_cache,
                            cut_mask_a=int(cut_mask_for_key),
                            use_induced_partition_cache=True,
                            kernel_cache_lookup_only=False,
                        )
                        computed_groups += len(missing_rows)
                        materialization_computed_groups = int(computed_groups)

            if (
                m_idx == int(len(mechanism_entries))
                or (m_idx % 10 == 0)
            ):
                elapsed = max(float(time.time() - t0), 1e-9)
                rate = float(computed_groups) / elapsed
                log.info(
                    "[%s] IIM materialization progress: mechanisms=%d/%d, computed_groups=%d, missing_groups=%d, key_checks=%d, rate=%.2f/s",
                    run_label,
                    int(m_idx),
                    int(len(mechanism_entries)),
                    int(computed_groups),
                    int(missing_groups),
                    int(total_directional_keys),
                    float(rate),
                )
                if checkpoint_path:
                    _write_checkpoint("running")

        try:
            induced_kernel_cache.flush()
        except Exception:
            pass

        elapsed = max(float(time.time() - t0), 1e-9)
        materialization_done_mechanisms = int(len(mechanism_entries))
        materialization_key_checks = int(total_directional_keys)
        materialization_missing_groups = int(missing_groups)
        materialization_computed_groups = int(computed_groups)
        log.info(
            "[%s] IIM materialization complete: computed_groups=%d, missing_groups=%d, key_checks=%d, elapsed=%.1fs",
            run_label,
            int(computed_groups),
            int(missing_groups),
            int(total_directional_keys),
            float(elapsed),
        )
        return True

    lookup_only_cut_aggregation = False
    phase2_cache_misses = 0
    try:
        lookup_only_cut_aggregation = bool(_materialize_induced_kernel_cases())
    except Exception as exc:
        lookup_only_cut_aggregation = False
        phase2_mode = "recompute"
        iim_phase = "phase2_cuts_recompute"
        log.warning(
            "[%s] IIM induced-kernel materialization failed (%s); using legacy per-cut recomputation.",
            run_label,
            exc,
        )
    if lookup_only_cut_aggregation:
        phase2_mode = "lookup_only"
        iim_phase = "phase2_cuts_lookup"
    else:
        phase2_mode = "recompute"
        if iim_phase != "phase2_cuts_recompute":
            iim_phase = "phase2_cuts_recompute"

    log.info(
        "[%s] IIM phase 2/2: cut search start (%d total, %d already done, lookup_only=%s)",
        run_label,
        int(len(cuts_eval)),
        int(len(completed_cut_scores)),
        bool(lookup_only_cut_aggregation),
    )
    cuts_done = int(len(completed_cut_scores))
    last_logged_done = -1
    for A, B in cuts_eval:
        cut_key = _cut_to_key(A, B)
        if cut_key in completed_cut_scores:
            psi_cut = float(completed_cut_scores[cut_key])
        else:
            cut_mask_a = 0
            for nn in A:
                cut_mask_a |= (1 << int(nn))
            if bool(lookup_only_cut_aggregation) and (induced_kernel_cache is not None):
                try:
                    psi_cut = _compute_psi(
                        tpm_full,
                        curr_obs,
                        states_full,
                        eff_bins,
                        mech_size_eff,
                        purv_size_eff,
                        phase1_parallel_workers=phase1_parallel_workers,
                        phase1_chunk_size=phase1_chunk_size,
                        phase1_shared_memory=phase1_shared_memory,
                        mechanisms=mechanisms_all,
                        purviews=purviews_all,
                        static_cache=psi_static_cache,
                        obs_state_cache=psi_obs_state_cache,
                        parallel_runtime=phase_parallel_runtime,
                        kernel_cache=induced_kernel_cache,
                        cut_mask_a=int(cut_mask_a),
                        use_induced_partition_cache=True,
                        kernel_cache_lookup_only=True,
                    )
                except _IIMKernelCacheMissError:
                    # Recompute this cut only; keep lookup-only aggregation (and
                    # parallelism) for later cuts unless misses are systematic.
                    phase2_cache_misses += 1
                    if phase2_cache_misses >= 3:
                        lookup_only_cut_aggregation = False
                        phase2_mode = "recompute"
                        iim_phase = "phase2_cuts_recompute"
                    log.warning(
                        "[%s] IIM lookup-only aggregation cache miss (%d so far); "
                        "recomputing cut %s%s.",
                        run_label,
                        int(phase2_cache_misses),
                        cut_key,
                        (
                            ""
                            if lookup_only_cut_aggregation
                            else " and all remaining cuts"
                        ),
                    )
                    tpm_cut = _build_cut_tpm(tpm_full, states_full, eff_bins, A, B)
                    psi_cut = _compute_psi(
                        tpm_cut,
                        curr_obs,
                        states_full,
                        eff_bins,
                        mech_size_eff,
                        purv_size_eff,
                        phase1_parallel_workers=phase1_parallel_workers,
                        phase1_chunk_size=phase1_chunk_size,
                        phase1_shared_memory=phase1_shared_memory,
                        mechanisms=mechanisms_all,
                        purviews=purviews_all,
                        static_cache=psi_static_cache,
                        obs_state_cache=psi_obs_state_cache,
                        parallel_runtime=phase_parallel_runtime,
                        kernel_cache=induced_kernel_cache,
                        cut_mask_a=int(cut_mask_a),
                        use_induced_partition_cache=bool(induced_kernel_cache is not None),
                        kernel_cache_lookup_only=False,
                    )
            else:
                tpm_cut = _build_cut_tpm(tpm_full, states_full, eff_bins, A, B)
                psi_cut = _compute_psi(
                    tpm_cut,
                    curr_obs,
                    states_full,
                    eff_bins,
                    mech_size_eff,
                    purv_size_eff,
                    phase1_parallel_workers=phase1_parallel_workers,
                    phase1_chunk_size=phase1_chunk_size,
                    phase1_shared_memory=phase1_shared_memory,
                    mechanisms=mechanisms_all,
                    purviews=purviews_all,
                    static_cache=psi_static_cache,
                    obs_state_cache=psi_obs_state_cache,
                    parallel_runtime=phase_parallel_runtime,
                    kernel_cache=induced_kernel_cache,
                    cut_mask_a=int(cut_mask_a),
                    use_induced_partition_cache=bool(induced_kernel_cache is not None),
                    kernel_cache_lookup_only=False,
                )
            completed_cut_scores[cut_key] = float(psi_cut)
            cuts_done += 1
            if (cuts_done % int(checkpoint_every_cuts) == 0) or (cuts_done == len(cuts_eval)):
                _write_checkpoint("running")

        if psi_cut > psi_preserved_max:
            psi_preserved_max = psi_cut
            mip_cut = (A, B)

        if (
            cuts_done != last_logged_done
            and ((cuts_done % int(progress_log_every_cuts) == 0) or (cuts_done == len(cuts_eval)))
        ):
            pct = 100.0 * float(cuts_done) / float(max(1, len(cuts_eval)))
            raw_so_far = float((psi_full - psi_preserved_max) / (psi_full + 1e-12))
            log.info(
                "[%s] IIM cut progress: %d/%d (%.1f%%), best_raw_so_far=%.6f",
                run_label,
                int(cuts_done),
                int(len(cuts_eval)),
                pct,
                raw_so_far,
            )
            last_logged_done = cuts_done

    if not np.isfinite(psi_preserved_max):
        iim_phase = "undefined"
        _write_checkpoint("undefined", undefined_reason="mip_not_found")
        _close_phase_parallel_runtime()
        _close_induced_cache()
        return _undefined_payload(
            "mip_not_found",
            psi_full=float(psi_full),
            psi_mip=np.nan,
            extra={
                "n_nodes_used": int(n_sel),
                "bins_used": int(eff_bins),
                "checkpoint_path": checkpoint_path,
                "checkpoint_resumed": bool(checkpoint_resumed),
                "checkpoint_reused_cuts": int(checkpoint_reused_cuts),
                "checkpoint_used_psi_full": bool(checkpoint_used_psi_full),
                "phase1_resumed_partial": bool(phase1_resumed_partial),
                "phase1_psi_partial": (
                    float(phase1_psi_partial) if np.isfinite(phase1_psi_partial) else np.nan
                ),
            },
        )

    raw = float((psi_full - psi_preserved_max) / (psi_full + 1e-12))
    canonical = float(np.clip(raw, 0.0, 1.0))
    # Legacy positive-only decomposition term retained for diagnostics.
    iim_plus = float(np.clip(raw, 0.0, 1.0))
    delta_psi = float(psi_full - psi_preserved_max)

    iim_phase = "complete"
    _write_checkpoint("complete")
    induced_cache_enabled = bool(induced_kernel_cache is not None)
    induced_cache_stats = (
        None if induced_kernel_cache is None else induced_kernel_cache.stats()
    )
    _close_phase_parallel_runtime()
    _close_induced_cache()

    # ---------- surrogate-null calibration ----------
    null_seed_eff = _resolve_null_seed(null_seed, rng)
    null_min_shift_eff = (
        int(null_min_shift)
        if null_min_shift is not None
        else max(int(lag_trs) + 1, int(math.ceil(0.1 * float(ts_sel.shape[1]))))
    )
    null_meta = {
        "IIM_null_method": str(null_method),
        "IIM_null_seed": int(null_seed_eff),
        "IIM_null_min_shift": int(null_min_shift_eff),
        "IIM_null_failed": 0,
        "IIM_null_undefined_reason": None,
    }
    null_deltas = []
    if int(null_surrogates) > 0:
        null_rng = np.random.RandomState(null_seed_eff)
        n_failed = 0
        for k in range(int(null_surrogates)):
            try:
                surr = _surrogate_timeseries(
                    ts_sel, null_method, null_rng, min_shift=null_min_shift_eff
                )
            except ValueError as exc:
                reason = f"surrogates_unavailable: {exc}"
                null_meta["IIM_null_undefined_reason"] = reason
                n_failed = int(null_surrogates)
                break
            cut_rng = np.random.RandomState()
            cut_rng.set_state(cut_rng_state)
            surr_ckpt = (
                None
                if not checkpoint_path
                else f"{checkpoint_path}.null-{null_method}"
                f"-seed{null_seed_eff}-{k:04d}.json"
            )
            surr_info = compute_IIM(
                surr,
                bins=int(eff_bins),
                lag_trs=int(lag_trs),
                n_parts=n_parts,
                rng=cut_rng,
                partition_mode=str(partition_mode),
                max_mechanism_size=int(mech_size_eff),
                max_purview_size=int(purv_size_eff),
                tpm_alpha=float(tpm_alpha),
                max_state_space=int(max_state_space),
                return_details=True,
                checkpoint_path=surr_ckpt,
                resume_from_checkpoint=bool(resume_from_checkpoint),
                checkpoint_every_cuts=int(checkpoint_every_cuts),
                progress_log_every_cuts=int(progress_log_every_cuts),
                progress_label=f"{run_label}/null{k:03d}",
                phase1_parallel_workers=phase1_parallel_workers,
                phase1_chunk_size=int(phase1_chunk_size),
                phase1_shared_memory=bool(phase1_shared_memory),
                kernel_cache_memory_entries=int(kernel_cache_memory_entries),
                kernel_cache_flush_batch=int(kernel_cache_flush_batch),
                hardware_backend=backend,
                tpm_estimator=str(tpm_estimator),
                node_selection="index",
                state_budget_policy="error",
                null_surrogates=0,
            )
            if bool(surr_info.get("defined", False)):
                null_deltas.append(
                    float(surr_info["Psi_full"]) - float(surr_info["Psi_mip_preserved"])
                )
            elif surr_info.get("undefined_reason") == "nonpositive_psi_full":
                # No estimable causal structure in the surrogate: no integration.
                null_deltas.append(0.0)
            else:
                n_failed += 1
        null_meta["IIM_null_failed"] = int(n_failed)
        if not null_deltas and null_meta["IIM_null_undefined_reason"] is None:
            null_meta["IIM_null_undefined_reason"] = "all_surrogates_undefined"
        null_stats = _null_calibration_stats(delta_psi, null_deltas)
        log.info(
            "[%s] IIM null calibration (%s, n=%d): Delta_Psi=%.6g "
            "null_mean=%.6g null_sd=%.6g z=%.3f p=%.3f",
            run_label,
            null_method,
            int(null_stats["null_n"]),
            delta_psi,
            float(null_stats["null_mean"]),
            float(null_stats["null_sd"]),
            float(null_stats["z"]),
            float(null_stats["p"]),
        )
    else:
        null_stats = None
    null_fields = _iim_null_fields(
        null_stats,
        psi_full=psi_full,
        delta_psi=delta_psi,
        null_values=null_deltas,
        meta=null_meta,
    )

    if int(null_surrogates) > 0:
        # Calibration was requested: never fall back silently to the
        # uncalibrated ratio (NaN + IIM_null_undefined_reason instead).
        selected = (
            null_fields["canonical_calibrated"] if clamp else null_fields["IIM_excess"]
        )
    else:
        selected = canonical if clamp else raw
    value = float(selected * scale)

    if return_details:
        return {
            "value": value,
            "raw": raw,
            "canonical": canonical,
            "clipped": canonical,  # legacy alias
            "iim_plus": iim_plus,  # legacy alias
            "scale": scale,
            "I_full": float(psi_full),  # legacy field name
            "min_partition_sum": float(psi_preserved_max),  # legacy field name
            "Psi_full": float(psi_full),
            "Psi_mip_preserved": float(psi_preserved_max),
            "n_nodes_used": int(n_sel),
            "bins_used": int(eff_bins),
            "max_nodes_requested": (None if max_nodes is None else int(max_nodes)),
            "max_mechanism_size_used": int(mech_size_eff),
            "max_purview_size_used": int(purv_size_eff),
            "n_parts_requested": (None if n_parts is None else int(n_parts)),
            "n_cuts_evaluated": int(len(cuts_eval)),
            "mip_cut": mip_cut,
            "checkpoint_path": checkpoint_path,
            "checkpoint_resumed": bool(checkpoint_resumed),
            "checkpoint_reused_cuts": int(checkpoint_reused_cuts),
            "checkpoint_used_psi_full": bool(checkpoint_used_psi_full),
            "phase1_resumed_partial": bool(phase1_resumed_partial),
            "phase1_psi_partial": (
                float(phase1_psi_partial) if np.isfinite(phase1_psi_partial) else np.nan
            ),
            "phase1_mechanisms_done": int(phase1_mechanisms_done),
            "phase1_total_mechanisms": (
                None
                if phase1_total_mechanisms is None
                else int(phase1_total_mechanisms)
            ),
            "phase1_eta_seconds": (
                None if phase1_eta_seconds is None else float(phase1_eta_seconds)
            ),
            "phase1_parallel_workers": (
                None
                if phase1_parallel_workers is None
                else int(phase1_parallel_workers)
            ),
            "phase1_chunk_size": int(phase1_chunk_size),
            "phase1_shared_memory": bool(phase1_shared_memory),
            "induced_partition_cache_enabled": bool(induced_cache_enabled),
            "induced_partition_cache_path": induced_kernel_cache_path_eff,
            "induced_partition_cache_stats": induced_cache_stats,
            "induced_partition_cache_disposal": kernel_cache_disposal,
            "phase2_cache_misses": int(phase2_cache_misses),
            "phase_parallel_enabled": bool(phase_parallel_runtime_enabled),
            "phase_parallel_fallbacks": list(phase_parallel_fallbacks),
            "iim_algorithm_version": IIM_ALGORITHM_VERSION,
            "tpm_estimator": str(tpm_estimator),
            "tpm_alpha": float(tpm_alpha),
            "lag_trs": int(lag_trs),
            "selected_nodes": [int(x) for x in selected_nodes_list],
            "node_selection_rule": prep.get("node_selection_rule"),
            "node_selection_degenerate": bool(
                prep.get("node_selection_degenerate", False)
            ),
            "bins_requested": int(bins),
            "state_budget_policy": str(state_budget_policy),
            "budget_adjustments": list(prep.get("budget_adjustments", [])),
            "n_states": int(prep.get("n_states", 0)),
            "n_states_observed": int(prep.get("n_states_observed", 0)),
            "n_transitions": int(prep.get("n_transitions", 0)),
            **null_fields,
            "defined": True,
            "undefined_reason": None,
        }
    return value


_COMPUTE_CI_DEPRECATION_WARNED = False


def compute_CI(
    ram: float,
    pdi: float,
    nas: float,
    iim: float,
    srpi: float,
    references: dict = None,
    weights: dict = None,
    defined: dict = None,
    eps: float = 1e-12,
    return_details: bool = False,
):
    """
    Deprecated alias of the legacy geometric-mean CI (not a gate).

    Emits a ``DeprecationWarning`` once per process and otherwise behaves
    exactly as before (see ``_compute_ci_legacy``). Attribution uses
    ``impact_pipeline.evidence.mpc_verdict`` (three-valued MPC verdict) and
    the secondary summary ``impact_pipeline.evidence.degree`` (MPC degree,
    only for ATTRIBUTED verdicts).
    """
    global _COMPUTE_CI_DEPRECATION_WARNED
    if not _COMPUTE_CI_DEPRECATION_WARNED:
        _COMPUTE_CI_DEPRECATION_WARNED = True
        warnings.warn(
            "compute_CI is deprecated: the legacy geometric-mean CI is not an "
            "attribution gate. Use impact_pipeline.evidence.mpc_verdict for the "
            "MPC verdict and impact_pipeline.evidence.degree for the MPC degree.",
            DeprecationWarning,
            stacklevel=2,
        )
    return _compute_ci_legacy(
        ram,
        pdi,
        nas,
        iim,
        srpi,
        references=references,
        weights=weights,
        defined=defined,
        eps=eps,
        return_details=return_details,
    )


def _compute_ci_legacy(
    ram: float,
    pdi: float,
    nas: float,
    iim: float,
    srpi: float,
    references: dict = None,
    weights: dict = None,
    defined: dict = None,
    eps: float = 1e-12,
    return_details: bool = False,
):
    """
    Reference-normalised weighted geometric Consciousness Index (CI); legacy
    geometric-mean CI (not a gate), kept for backward compatibility.

    CI = (RAM*^alpha) (PDI+*^beta) (NAS*^gamma) (IIM_can*^delta) (SRPI*^rho)

    where each * component is divided by an explicit reference mean
    (``references``; the pipeline default is the cohort high-state mean, see
    ``synergy_ci.resolve_ci_references``). NAS enters directly; the legacy
    HypergraphSynergy statistic S is not part of CI.

    Definedness is three-valued:
      - a component that could not be measured is undefined (NaN), never 0;
      - if any component with non-zero weight is undefined, or its reference is
        non-finite or <= 0, CI is undefined and the returned value is NaN;
      - a measured component <= 0 with non-zero weight is a legitimate measured
        zero and gives CI = 0.
    Components with zero weight are ignored (they need not be defined).
    """
    comp_keys = ("RAM", "PDI", "NAS", "IIM", "SRPI")
    comp_vals = {
        "RAM": float(ram),
        "PDI": float(pdi),
        "NAS": float(nas),
        "IIM": float(iim),
        "SRPI": float(srpi),
    }

    if references is None:
        # Explicit opt-out of normalisation (unit references).
        references = {k: 1.0 for k in comp_keys}
    else:
        # A component missing from a supplied reference dict (or given as None /
        # non-numeric) has no reference (no silent 1.0); it is treated like a
        # non-finite reference below.
        def _ref_value(val):
            try:
                return np.nan if val is None else float(val)
            except (TypeError, ValueError):
                return np.nan

        references = {k: _ref_value(references.get(k)) for k in comp_keys}

    if weights is None:
        weights = {k: 1.0 / len(comp_keys) for k in comp_keys}
    else:
        weights = {k: float(weights.get(k, 0.0)) for k in comp_keys}
        if any((not np.isfinite(v)) or v < 0 for v in weights.values()):
            raise ValueError("weights must be finite and non-negative")
        wsum = sum(weights.values())
        if wsum <= 0:
            raise ValueError("weights must sum to a positive value")
        weights = {k: v / wsum for k, v in weights.items()}

    if defined is None:
        defined = {k: True for k in comp_keys}
    else:
        defined = {k: bool(defined.get(k, True)) for k in comp_keys}
    # Non-finite component values are treated as undefined.
    for k in comp_keys:
        if not np.isfinite(comp_vals[k]):
            defined[k] = False

    norm = {}
    undefined_weighted = []
    invalid_references = []
    for k in comp_keys:
        if not defined[k]:
            norm[k] = np.nan
            if weights[k] > 0:
                undefined_weighted.append(k)
            continue
        ref = references[k]
        if (not np.isfinite(ref)) or ref <= 0:
            # No floor: an unusable reference makes the normalised component undefined.
            norm[k] = np.nan
            if weights[k] > 0:
                invalid_references.append(k)
            continue
        # Metrics are defined as non-negative components in CI.
        norm[k] = max(comp_vals[k] / ref, 0.0)

    missing = list(undefined_weighted) + [f"{k}_reference" for k in invalid_references]
    ci_defined = not missing
    if not ci_defined:
        # Undefined, not zero: missing evidence is never converted into a value.
        ci_val = float("nan")
    # Weighted geometric mean; a measured zero with non-zero weight zeros CI.
    elif any((weights[k] > 0.0 and norm[k] <= 0.0) for k in comp_keys):
        ci_val = 0.0
    else:
        log_ci = 0.0
        for k in comp_keys:
            if weights[k] > 0.0:
                log_ci += weights[k] * np.log(norm[k] + eps)
        ci_val = float(np.exp(log_ci))

    if return_details:
        return {
            "value": ci_val,
            "defined": bool(ci_defined),
            "missing": missing,
            "normalized_components": norm,
            "weights": weights,
            "references": references,
            "defined_components": defined,
            "undefined_weighted_components": undefined_weighted,
            "invalid_reference_components": invalid_references,
        }
    return ci_val


def _srpi_undefined_result(
    reason,
    return_details,
    counts,
    modality,
    windows_sec,
    eps,
):
    if not return_details:
        return float("nan")
    return {
        "value": float("nan"),
        "undefined_reason": str(reason),
        "components": {
            "reactivity_bias": float("nan"),
            "representational_separability": float("nan"),
            "self_pattern_stability": float("nan"),
            "internal_state_coupling": float("nan"),
        },
        "components_raw": {
            "reactivity_bias": float("nan"),
            "representational_separability": float("nan"),
            "self_pattern_stability": float("nan"),
            "internal_state_coupling": float("nan"),
        },
        "signed_reactivity_bias": float("nan"),
        "separability_cv_auc": float("nan"),
        "reliability": float("nan"),
        "weights": {
            "reactivity_bias": float("nan"),
            "representational_separability": float("nan"),
            "self_pattern_stability": float("nan"),
            "internal_state_coupling": float("nan"),
        },
        "counts": dict(counts),
        "modality": str(modality),
        "windows_sec": dict(windows_sec),
        "eps": float(eps),
    }


def _event_locked_state_deltas(ts_z, event_idx, lag_samples, pre_samples, post_samples):
    """
    Event-locked pre-event state and response change.

    For an event at sample ``e`` the pre-event state is the mean over
    ``[e - pre_samples, e)`` (strictly before onset) and the response is the
    mean over ``[e + lag, e + lag + post_samples)``; ``d_e = r_e - p_e``.
    """
    pre_vecs, post_vecs, used = _window_mean_vectors(
        ts_z,
        np.asarray(event_idx, dtype=np.int64).reshape(-1),
        pre_samples=pre_samples,
        post_samples=post_samples,
        post_lag_samples=int(lag_samples),
    )
    if used.size == 0:
        n_regions = int(ts_z.shape[0])
        empty = np.empty((0, n_regions), dtype=float)
        return empty, empty, np.empty(0, dtype=np.int64)
    delta_vecs = post_vecs - pre_vecs
    return pre_vecs, delta_vecs, used


def _ledoit_wolf_shrinkage(xc):
    """
    Ledoit-Wolf (2004) optimal shrinkage intensity for centred data ``xc``
    (n x p) toward the scaled identity, computed from the n x n Gram matrix.
    """
    xc = np.asarray(xc, dtype=float)
    n, p = xc.shape
    if n < 2 or p < 1:
        return 1.0
    g = xc @ xc.T
    s_fro2 = float(np.sum(g * g)) / float(n * n)
    mu = float(np.trace(g)) / float(n * p)
    delta2 = (s_fro2 - mu * mu * p) / float(p)
    if (not np.isfinite(delta2)) or delta2 <= 0.0:
        return 1.0
    diag = np.diag(g)
    g2_diag = np.einsum("ij,ji->i", g, g)
    beta_bar2 = float(np.sum(diag * diag - 2.0 * g2_diag / n + s_fro2))
    beta_bar2 /= float(n * n * p)
    beta2 = min(max(beta_bar2, 0.0), delta2)
    return float(np.clip(beta2 / delta2, 0.0, 1.0))


def _shrinkage_lda_direction(a, b, shrinkage_floor, hardware_backend=None):
    """
    Unit discriminant direction ``w ∝ (Σ_s)^-1 (μ_a - μ_b)`` and midpoint.

    ``Σ_s = (1 - s) S + s (tr S / p) I`` with Ledoit-Wolf shrinkage ``s``
    (at least ``shrinkage_floor``); solved in the n x n dual (Woodbury), so
    the cost does not grow with p^2.
    """
    ma = a.mean(axis=0)
    mb = b.mean(axis=0)
    resid = np.vstack([a - ma, b - mb])
    n_r, p = resid.shape
    dof = float(max(1, n_r - 2))
    shrink = max(_ledoit_wolf_shrinkage(resid), float(shrinkage_floor))
    shrink = float(min(shrink, 1.0))
    mu = float(np.sum(resid * resid)) / (dof * float(p))
    alpha = shrink * mu
    if (not np.isfinite(alpha)) or alpha <= 1e-300:
        return None, None
    beta = (1.0 - shrink) / dof
    delta = ma - mb
    if beta > 0.0:
        k = alpha * np.eye(n_r) + beta * (resid @ resid.T)
        try:
            sol = accelerated_solve(k, resid @ delta, backend=hardware_backend)
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(k).dot(resid @ delta)
        w = (delta - beta * (resid.T @ sol)) / alpha
    else:
        w = delta / alpha
    nrm = float(np.linalg.norm(w))
    if (not np.isfinite(nrm)) or nrm <= 1e-300:
        return None, None
    return w / nrm, 0.5 * (ma + mb)


def _auc_from_scores(pos, neg):
    pos = np.asarray(pos, dtype=float).reshape(-1)
    neg = np.asarray(neg, dtype=float).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = np.sum(pos[:, None] > neg[None, :])
    eq = np.sum(pos[:, None] == neg[None, :])
    return float((gt + 0.5 * eq) / float(pos.size * neg.size))


def _cv_separability(x_self, x_non, shrinkage_floor, n_folds=5, hardware_backend=None):
    """
    Cross-validated, chance-corrected self/non-self separability.

    Shrinkage-LDA fitted on time-blocked, class-stratified training folds;
    held-out events are scored along the unit discriminant relative to the
    training midpoint and the pooled scores give an AUC. Returns
    ``(S, auc)`` with ``S = clip(2 * (AUC - 0.5), 0, 1)`` (0 at chance).
    """
    x_self = np.asarray(x_self, dtype=float)
    x_non = np.asarray(x_non, dtype=float)
    n_s, n_n = int(x_self.shape[0]), int(x_non.shape[0])
    k = int(min(int(n_folds), n_s, n_n))
    if k < 2:
        return float("nan"), float("nan")
    fold_s = np.empty(n_s, dtype=np.int64)
    fold_n = np.empty(n_n, dtype=np.int64)
    for f, chunk in enumerate(np.array_split(np.arange(n_s), k)):
        fold_s[chunk] = f
    for f, chunk in enumerate(np.array_split(np.arange(n_n), k)):
        fold_n[chunk] = f
    score_s = np.full(n_s, np.nan)
    score_n = np.full(n_n, np.nan)
    for f in range(k):
        tr_s = x_self[fold_s != f]
        tr_n = x_non[fold_n != f]
        if tr_s.shape[0] < 2 or tr_n.shape[0] < 2:
            return float("nan"), float("nan")
        w, mid = _shrinkage_lda_direction(
            tr_s,
            tr_n,
            shrinkage_floor=shrinkage_floor,
            hardware_backend=hardware_backend,
        )
        if w is None:
            return float("nan"), float("nan")
        score_s[fold_s == f] = (x_self[fold_s == f] - mid) @ w
        score_n[fold_n == f] = (x_non[fold_n == f] - mid) @ w
    auc = _auc_from_scores(score_s, score_n)
    if not np.isfinite(auc):
        return float("nan"), float("nan")
    return float(np.clip(2.0 * (auc - 0.5), 0.0, 1.0)), auc


def _mean_pairwise_pattern_similarity(mat, hardware_backend=None):
    x = np.asarray(mat, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.nan
    x = x - x.mean(axis=1, keepdims=True)
    norms = accelerated_row_norm(x, axis=1, backend=hardware_backend)
    keep = norms > 1e-12
    x = x[keep]
    norms = norms[keep]
    if x.shape[0] < 2:
        return np.nan
    x = x / norms[:, None]
    sim = accelerated_dot(x, x.T, backend=hardware_backend)
    iu = np.triu_indices(sim.shape[0], k=1)
    vals = sim[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.clip(vals.mean(), -1.0, 1.0))


def _leading_latent_axis(pre_a, pre_b, hardware_backend=None):
    x = np.vstack([np.asarray(pre_a, dtype=float), np.asarray(pre_b, dtype=float)])
    if x.ndim != 2 or x.shape[0] < 2:
        return None
    x = x - x.mean(axis=0, keepdims=True)
    try:
        backend = resolve_hardware_backend(hardware_backend)
        if backend.accelerator:
            xp = get_array_module(backend)
            _, _, vt_d = xp.linalg.svd(xp.asarray(x), full_matrices=False)
            vt = to_numpy(vt_d)
        else:
            _, _, vt = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vt.ndim != 2 or vt.shape[0] == 0:
        return None
    axis = np.asarray(vt[0], dtype=float).reshape(-1)
    nrm = float(np.linalg.norm(axis))
    if nrm <= 1e-12:
        return None
    return axis / nrm


def compute_SRPI(
    ts: np.ndarray,
    tr: float = None,
    self_onsets: list = None,
    nonself_onsets: list = None,
    directional: bool = False,
    eps: float = 1e-8,
    modality: str = "fmri",
    pre_window_sec: float = 2.0,
    response_lag_sec: float = 4.0,
    response_window_sec: float = 6.0,
    covariance_ridge: float = 1e-3,
    component_weights=(0.35, 0.25, 0.20, 0.20),
    min_events_per_class: int = 3,
    sample_reliability_tau: float = 4.0,
    return_details: bool = False,
    hardware_backend=None,
    separability_cv_folds: int = 5,
) -> float:
    """
    Self-Referential Processing Index (SRPI).

    SRPI quantifies a measurable minimal self-model from self vs non-self
    event-locked activity using four components:
      1) self-reactivity bias,
      2) self/non-self representational separability,
      3) excess within-self pattern stability, and
      4) excess coupling of self responses to pre-event internal state.

    The final SRPI is the weighted geometric mean of these components after an
    event-count reliability attenuation.

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_time)
        Timeseries matrix.
    tr : float
        Sampling interval (seconds).
    self_onsets : list of float
        Self-related event onsets (seconds).
    nonself_onsets : list of float
        Non-self event onsets (seconds).
    directional : bool, optional
        If ``True``, maps the signed self-vs-nonself reactivity contrast from
        [-1,1] into [0,1] for the reactivity component; if ``False`` (default),
        only positive self-preference contributes.
    eps : float, optional
        Numerical stabilizer.
    modality : {'fmri', 'eeg'}, optional
        Modality label used for traceability in diagnostics.
    pre_window_sec : float, optional
        Pre-event window (seconds) for internal-state estimation; it covers
        ``[onset - pre_window_sec, onset)``, strictly before the event.
    response_lag_sec : float, optional
        Post-onset lag (seconds) before response window begins.
    response_window_sec : float, optional
        Response window length (seconds): ``[onset + lag, onset + lag + W)``.
    covariance_ridge : float, optional
        Ridge regularization for separability, used as the minimum covariance
        shrinkage ``ridge / (1 + ridge)`` of the shrinkage-LDA (the
        Ledoit-Wolf intensity is used when larger).
    component_weights : tuple(float, float, float, float), optional
        Weights for (reactivity, separability, stability, internal coupling).
    min_events_per_class : int, optional
        Minimum required self and non-self events after windowing.
    sample_reliability_tau : float, optional
        Saturation constant for event-count reliability attenuation.
    return_details : bool, optional
        If ``True``, returns diagnostics dict.
    separability_cv_folds : int, optional
        Number of time-blocked, class-stratified folds for the
        cross-validated separability.

    Notes
    -----
    Separability is out-of-sample: a shrinkage-LDA is fitted on training
    folds of the response changes ``d_e`` and scored on held-out events;
    ``S = [2 (AUC_cv - 0.5)]_+`` is 0 at chance. (The in-sample Mahalanobis
    distance saturates at 1 whenever regions outnumber events.)

    The internal-state coupling uses absolute correlations,
    ``I = [|corr(u_s, m_s)| - |corr(u_n, m_n)|]_+``, because the sign of the
    leading latent axis of the pre-event states is arbitrary (SVD sign), so
    only the strength of state-response coupling is identifiable.
    """
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    if tr is None or (not np.isfinite(tr)) or float(tr) <= 0:
        raise ValueError("tr must be provided and > 0 for SRPI")
    if int(separability_cv_folds) < 2:
        raise ValueError("separability_cv_folds must be >= 2")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if pre_window_sec <= 0 or response_window_sec <= 0:
        raise ValueError("pre_window_sec and response_window_sec must be > 0")
    if response_lag_sec < 0:
        raise ValueError("response_lag_sec must be >= 0")
    if covariance_ridge <= 0:
        raise ValueError("covariance_ridge must be > 0")
    if min_events_per_class < 2:
        raise ValueError("min_events_per_class must be >= 2")
    if sample_reliability_tau <= 0:
        raise ValueError("sample_reliability_tau must be > 0")

    weights = np.asarray(component_weights, dtype=float).reshape(-1)
    if weights.size != 4:
        raise ValueError(
            "component_weights must be a 4-tuple: "
            "(reactivity, separability, stability, internal_coupling)"
        )
    if np.any(weights < 0):
        raise ValueError("component_weights must be non-negative")
    if not np.any(weights > 0):
        raise ValueError("At least one SRPI component weight must be > 0")

    backend = resolve_hardware_backend(hardware_backend)
    n_regions, n_tp = ts.shape
    _, self_idx = _sanitize_onset_seconds(self_onsets, tr=tr, n_tp=n_tp)
    _, non_idx = _sanitize_onset_seconds(nonself_onsets, tr=tr, n_tp=n_tp)

    counts = {
        "n_self_events_raw": int(self_idx.size),
        "n_nonself_events_raw": int(non_idx.size),
        "n_self_events_used": 0,
        "n_nonself_events_used": 0,
    }
    windows_sec = {
        "pre_window_sec": float(pre_window_sec),
        "response_lag_sec": float(response_lag_sec),
        "response_window_sec": float(response_window_sec),
    }
    if self_idx.size == 0 and non_idx.size == 0:
        return _srpi_undefined_result(
            reason="missing_self_and_nonself_events",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    if self_idx.size == 0:
        return _srpi_undefined_result(
            reason="missing_self_events",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    if non_idx.size == 0:
        return _srpi_undefined_result(
            reason="missing_nonself_events",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )

    ts_z = accelerated_zscore(ts, axis=1, backend=backend, eps=1e-12)
    pre_samples = max(1, int(round(float(pre_window_sec) / float(tr))))
    lag_samples = int(round(float(response_lag_sec) / float(tr)))
    post_samples = max(1, int(round(float(response_window_sec) / float(tr))))

    pre_self, delta_self, used_self = _event_locked_state_deltas(
        ts_z=ts_z,
        event_idx=self_idx,
        lag_samples=lag_samples,
        pre_samples=pre_samples,
        post_samples=post_samples,
    )
    pre_non, delta_non, used_non = _event_locked_state_deltas(
        ts_z=ts_z,
        event_idx=non_idx,
        lag_samples=lag_samples,
        pre_samples=pre_samples,
        post_samples=post_samples,
    )
    counts["n_self_events_used"] = int(used_self.size)
    counts["n_nonself_events_used"] = int(used_non.size)

    if used_self.size < int(min_events_per_class):
        return _srpi_undefined_result(
            reason="insufficient_self_events_after_windowing",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    if used_non.size < int(min_events_per_class):
        return _srpi_undefined_result(
            reason="insufficient_nonself_events_after_windowing",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )

    mag_self = accelerated_row_norm(delta_self, axis=1, backend=backend)
    mag_non = accelerated_row_norm(delta_non, axis=1, backend=backend)
    gamma_self = float(np.mean(mag_self))
    gamma_non = float(np.mean(mag_non))
    signed_bias = float((gamma_self - gamma_non) / (gamma_self + gamma_non + float(eps)))
    signed_bias = float(np.clip(signed_bias, -1.0, 1.0))
    if directional:
        c_reactivity = 0.5 * (signed_bias + 1.0)
    else:
        c_reactivity = max(signed_bias, 0.0)
    c_reactivity = float(np.clip(c_reactivity, 0.0, 1.0))

    ridge = float(covariance_ridge)
    c_sep, sep_auc = _cv_separability(
        delta_self,
        delta_non,
        shrinkage_floor=ridge / (1.0 + ridge),
        n_folds=int(separability_cv_folds),
        hardware_backend=backend,
    )
    if not np.isfinite(c_sep):
        return _srpi_undefined_result(
            reason="separability_undefined",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    rho_self = _mean_pairwise_pattern_similarity(delta_self, hardware_backend=backend)
    rho_non = _mean_pairwise_pattern_similarity(delta_non, hardware_backend=backend)
    if (not np.isfinite(rho_self)) or (not np.isfinite(rho_non)):
        return _srpi_undefined_result(
            reason="stability_undefined",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    c_stability = float(np.clip((float(rho_self) - float(rho_non)) / 2.0, 0.0, 1.0))

    axis = _leading_latent_axis(pre_self, pre_non, hardware_backend=backend)
    if axis is None:
        return _srpi_undefined_result(
            reason="internal_state_axis_undefined",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    state_self = pre_self.dot(axis)
    state_non = pre_non.dot(axis)
    corr_self = _safe_abs_corr(state_self, mag_self)
    corr_non = _safe_abs_corr(state_non, mag_non)
    if (not np.isfinite(corr_self)) or (not np.isfinite(corr_non)):
        return _srpi_undefined_result(
            reason="internal_state_coupling_undefined",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    c_internal = float(np.clip(float(corr_self) - float(corr_non), 0.0, 1.0))

    reliability = _sample_reliability(
        min(int(used_self.size), int(used_non.size)),
        tau=float(sample_reliability_tau),
    )
    comp_raw = np.asarray(
        [c_reactivity, c_sep, c_stability, c_internal],
        dtype=float,
    )
    comp = np.clip(comp_raw * float(reliability), 0.0, 1.0)

    if np.any((comp <= 0.0) & (weights > 0.0)):
        srpi_value = 0.0
    else:
        srpi_value = float(
            np.exp(np.sum(weights * np.log(comp)) / np.sum(weights))
        )
        srpi_value = float(np.clip(srpi_value, 0.0, 1.0))

    if not return_details:
        return srpi_value

    return {
        "value": float(srpi_value),
        "undefined_reason": None,
        "components": {
            "reactivity_bias": float(comp[0]),
            "representational_separability": float(comp[1]),
            "self_pattern_stability": float(comp[2]),
            "internal_state_coupling": float(comp[3]),
        },
        "components_raw": {
            "reactivity_bias": float(comp_raw[0]),
            "representational_separability": float(comp_raw[1]),
            "self_pattern_stability": float(comp_raw[2]),
            "internal_state_coupling": float(comp_raw[3]),
        },
        "signed_reactivity_bias": float(signed_bias),
        "separability_cv_auc": float(sep_auc),
        "reliability": float(reliability),
        "weights": {
            "reactivity_bias": float(weights[0]),
            "representational_separability": float(weights[1]),
            "self_pattern_stability": float(weights[2]),
            "internal_state_coupling": float(weights[3]),
        },
        "counts": dict(counts),
        "modality": str(modality),
        "windows_sec": dict(windows_sec),
        "eps": float(eps),
    }
