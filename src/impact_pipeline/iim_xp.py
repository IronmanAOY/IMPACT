"""
Array-module (NumPy / CuPy) kernels for the IIM Psi computation.

These functions compute the same quantities as the numba/host kernel
``mpc_metrics._iim_phase1_chunk_contribution`` (phase-1 Psi contribution of a
set of mechanisms, and hence the Psi of a cut TPM), written once against an
array module ``xp`` so that they run unchanged on NumPy (host) or CuPy (GPU/APU,
e.g. ROCm on the HLRS Hunter MI300A). Only functions shared by NumPy >= 1.24 and
CuPy 13/14 are used (``reshape``, ``sum``, ``matmul``, ``where``, ``log``,
fancy indexing); no NumPy-2-only API.

Computation (per mechanism M, all observed states of M at once):

* The TPM is viewed as a tensor with one axis per current node and one per
  next node (state index = big-endian code of the node states, node 0 most
  significant, as in ``prepare_iim_problem``).
* Effect rows ``G_Y`` (``K^|Y| x S``): mean over the current states that share a
  Y-state of the next-state distribution. Cause rows ``C_Y``: the likelihood of
  a next Y-state over all current states, normalised to a posterior (uniform
  prior over current states).
* Purview marginals are taken for all purviews of the same size at once with a
  one-hot marginalisation matrix (one GEMM per mechanism subset batch), so the
  work is vectorised over mechanism states and purviews.
* The partitioned repertoire ``p(Z1|M1) x p(Z2|M2)`` is formed by summing the
  purview marginal of each part over the positions of the other part with
  ``keepdims`` and broadcasting; both pairings of every mechanism/purview
  bipartition are searched (as in the host kernel), the minimum Jensen-Shannon
  divergence (bits) over partitions is phi for that purview, phi_e/phi_c are
  the maxima over purviews and phi = min(phi_e, phi_c).

Batch sizes are bounded by ``IMPACT_IIM_XP_MAX_ELEMENTS`` (elements of the
largest temporary array; default 2**23).

Numerical parity with the host kernel is tested to 1e-10 on NumPy
(``tests/test_iim_xp_kernel.py``); on an accelerator it is checked by
``python -m impact_pipeline.hardware_selftest``.
"""

from __future__ import annotations

import itertools
import math
import os
from collections import OrderedDict

import numpy as np

XP_MAX_ELEMENTS_ENV = "IMPACT_IIM_XP_MAX_ELEMENTS"
XP_CACHE_ELEMENTS_ENV = "IMPACT_IIM_XP_CACHE_ELEMENTS"
_DEFAULT_MAX_ELEMENTS = 1 << 23
_DEFAULT_CACHE_ELEMENTS = 1 << 27
_LOG2 = math.log(2.0)
CUT_MODES = ("bidirectional", "directional")


def _env_int(name, default):
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _to_host(value):
    if isinstance(value, np.ndarray):
        return value
    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy  # noqa: F401  (only reached when CuPy arrays are passed)

        return value.get()
    return np.asarray(value)


def _is_host_module(xp) -> bool:
    return xp is np


# ---------------------------------------------------------------------------
# Combinatorics (host); identical enumeration order to mpc_metrics
# ---------------------------------------------------------------------------


def enumerate_subsets(nodes, max_size):
    out = []
    for r in range(1, min(int(max_size), len(nodes)) + 1):
        out.extend(tuple(c) for c in itertools.combinations(nodes, r))
    return out


def enumerate_bipartitions(nodes):
    """Unordered bipartitions into two non-empty parts (same order as mpc_metrics)."""
    n = len(nodes)
    if n < 2:
        return []
    out = []
    node_set = tuple(nodes)
    for k in range(1, (n // 2) + 1):
        for part_a in itertools.combinations(node_set, k):
            part_a = tuple(part_a)
            part_b = tuple(x for x in node_set if x not in part_a)
            if k == n - k and part_a[0] > part_b[0]:
                continue
            out.append((part_a, part_b))
    return out


def subset_keys(states, subset, base):
    """Big-endian code of ``states[:, subset]`` (host)."""
    states = np.asarray(states)
    if len(subset) == 0:
        return np.zeros(states.shape[0], dtype=np.int64)
    cols = states[:, list(subset)].astype(np.int64, copy=False)
    mult = int(base) ** np.arange(len(subset) - 1, -1, -1, dtype=np.int64)
    return (cols * mult).sum(axis=1).astype(np.int64)


def mechanism_state_weights(curr_obs, mechanism, base, obs_weights=None):
    """
    Observed states of ``mechanism`` and their relative frequencies.

    Without ``obs_weights`` every row of ``curr_obs`` counts once (empirical
    frequencies, as in compute_IIM); with ``obs_weights`` (one non-negative
    weight per row, e.g. a stationary distribution over all system states)
    the weights are summed per mechanism state and states with zero weight
    are dropped.
    """
    keys = subset_keys(curr_obs, mechanism, base)
    if keys.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=float)
    if obs_weights is None:
        uk, cnt = np.unique(keys, return_counts=True)
        wt = cnt.astype(float) / float(cnt.sum())
        return uk.astype(np.int64, copy=False), wt.astype(float, copy=False)
    w = np.asarray(obs_weights, dtype=float).reshape(-1)
    if w.shape[0] != keys.shape[0]:
        raise ValueError(
            f"obs_weights has {w.shape[0]} entries for {keys.shape[0]} observed states"
        )
    uk, inv = np.unique(keys, return_inverse=True)
    tot = np.bincount(np.asarray(inv).reshape(-1), weights=w, minlength=uk.size)
    keep = tot > 0
    uk = uk[keep]
    tot = tot[keep]
    if tot.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=float)
    return uk.astype(np.int64, copy=False), (tot / float(tot.sum())).astype(float)


def _decode_digits(keys, length, base):
    keys = np.asarray(keys, dtype=np.int64)
    out = np.zeros((keys.size, int(length)), dtype=np.int64)
    v = keys.copy()
    for i in range(int(length) - 1, -1, -1):
        out[:, i] = v % int(base)
        v //= int(base)
    return out


def _encode_digits(digits, base):
    digits = np.asarray(digits, dtype=np.int64)
    if digits.shape[1] == 0:
        return np.zeros(digits.shape[0], dtype=np.int64)
    mult = int(base) ** np.arange(digits.shape[1] - 1, -1, -1, dtype=np.int64)
    return (digits * mult).sum(axis=1)


def state_table(n_nodes, base):
    """All system states (rows) in the big-endian order used by the IIM code."""
    n_states = int(base) ** int(n_nodes)
    sid = np.arange(n_states, dtype=np.int64)[:, None]
    powv = (int(base) ** np.arange(int(n_nodes) - 1, -1, -1, dtype=np.int64))[None, :]
    return ((sid // powv) % int(base)).astype(np.int16)


# ---------------------------------------------------------------------------
# Cut TPMs (tensor form, any array module)
# ---------------------------------------------------------------------------


def _normalize_rows(xp, mat, width):
    s = mat.sum(axis=-1, keepdims=True)
    return xp.where(s > 0, mat / xp.where(s > 0, s, 1.0), 1.0 / float(width))


def _reduce(arr, axes, keepdims=False):
    axes = tuple(int(a) for a in axes)
    if not axes:
        return arr
    return arr.sum(axis=axes, keepdims=keepdims)


def cut_tpm(tpm, n_nodes, base, part_a, part_b, cut_mode="bidirectional", xp=None):
    """
    TPM of a system cut, as an ``xp`` array.

    ``cut_mode='bidirectional'`` (the compute_IIM cut): both directions between
    A and B are replaced by noise; T_cut(s'|s) = p_A(s'_A|s_A) p_B(s'_B|s_B)
    where p_A is the A-marginal of the next state averaged uniformly over the
    current state of B (identical to ``mpc_metrics._iim_build_cut_tpm``).

    ``cut_mode='directional'``: only the connections A -> B are severed (IIT
    unidirectional cut). Every unit i receives its cut inputs as independent
    noise: T_cut(s'|s) = prod_{i in A} p_i(s'_i|s) prod_{j in B} pbar_j(s'_j|s_B)
    with pbar_j the per-unit transition averaged uniformly over the current
    state of A. This is defined on the state-by-node (conditionally
    independent) form of the TPM; for a TPM that is not conditionally
    independent the within-step correlations are discarded as well.
    """
    xp = np if xp is None else xp
    n = int(n_nodes)
    k = int(base)
    n_states = k ** n
    part_a = tuple(sorted(int(x) for x in part_a))
    part_b = tuple(sorted(int(x) for x in part_b))
    if sorted(part_a + part_b) != list(range(n)) or not part_a or not part_b:
        raise ValueError(f"({part_a}, {part_b}) is not a bipartition of {n} nodes")
    if cut_mode not in CUT_MODES:
        raise ValueError(f"cut_mode must be one of {CUT_MODES}, got {cut_mode!r}")
    t = xp.asarray(tpm, dtype=xp.float64)
    t4 = t.reshape((k,) * (2 * n))
    if cut_mode == "bidirectional":
        factors = []
        for part in (part_a, part_b):
            other = [i for i in range(n) if i not in part]
            m = _reduce(t4, [n + i for i in other])
            if other:
                m = m.sum(axis=tuple(other)) / float(k ** len(other))
            width = k ** len(part)
            m = _normalize_rows(xp, m.reshape(width, width), width)
            shape = [1] * (2 * n)
            for i in part:
                shape[i] = k
                shape[n + i] = k
            factors.append(m.reshape(tuple(shape)))
        out = (factors[0] * factors[1]).reshape(n_states, n_states)
    else:
        out = None
        for i in range(n):
            p_i = _reduce(t4, [n + j for j in range(n) if j != i])  # (cur..., K)
            if i in part_b:
                p_i = p_i.sum(axis=part_a, keepdims=True) / float(k ** len(part_a))
            p_i = _normalize_rows(xp, p_i, k)
            shape = list(p_i.shape[:n]) + [1] * n
            shape[n + i] = k
            p_i = p_i.reshape(tuple(shape))
            out = p_i if out is None else out * p_i
        out = xp.broadcast_to(out, (k,) * (2 * n)).reshape(n_states, n_states)
    return _normalize_rows(xp, out, n_states)


def state_by_node_deviation(tpm, n_nodes, base, xp=None):
    """Max |T - prod_i p_i(s'_i|s)|: 0 for a conditionally independent TPM."""
    xp = np if xp is None else xp
    n = int(n_nodes)
    k = int(base)
    t = xp.asarray(tpm, dtype=xp.float64)
    t4 = t.reshape((k,) * (2 * n))
    prod = None
    for i in range(n):
        p_i = _reduce(t4, [n + j for j in range(n) if j != i])
        shape = [k] * n + [1] * n
        shape[n + i] = k
        p_i = p_i.reshape(tuple(shape))
        prod = p_i if prod is None else prod * p_i
    dev = xp.abs(xp.broadcast_to(prod, (k,) * (2 * n)) - t4).max()
    return float(_to_host(dev))


# ---------------------------------------------------------------------------
# Psi
# ---------------------------------------------------------------------------


class IIMXpWorkspace:
    """
    Reusable device state for Psi evaluations on one state space.

    Holds the one-hot purview marginalisers (independent of the TPM, reused
    across cuts) and the per-TPM conditional rows. ``bind(tpm)`` switches the
    TPM (a new object resets the row cache).
    """

    def __init__(self, xp, states_full, base, cache_elements=None):
        self.xp = np if xp is None else xp
        self.states = np.asarray(states_full)
        self.base = int(base)
        self.n = int(self.states.shape[1])
        self.n_states = int(self.states.shape[0])
        if self.n_states != self.base ** self.n:
            raise ValueError(
                f"states_full has {self.n_states} rows, expected {self.base}**{self.n}"
            )
        self.cache_elements = (
            _env_int(XP_CACHE_ELEMENTS_ENV, _DEFAULT_CACHE_ELEMENTS)
            if cache_elements is None
            else int(cache_elements)
        )
        self._onehot = OrderedDict()
        self._onehot_elements = 0
        self._rows = {}
        self._rows_elements = 0
        self._tpm_ref = None
        self._t4_eff = None
        self._t4_cause = None

    def bind(self, tpm):
        if tpm is self._tpm_ref:
            return
        xp = self.xp
        k, n = self.base, self.n
        t = xp.asarray(tpm, dtype=xp.float64)
        if t.shape != (self.n_states, self.n_states):
            raise ValueError(
                f"TPM shape {tuple(t.shape)} does not match {self.n_states} states"
            )
        self._tpm_ref = tpm
        self._t4_eff = t.reshape((k,) * n + (self.n_states,))
        self._t4_cause = t.reshape((self.n_states,) + (k,) * n)
        self._rows = {}
        self._rows_elements = 0

    def cond_rows(self, direction, subset):
        """(K^|Y| x S) effect rows (mean next-state distributions) or cause
        posteriors."""
        key = (direction, subset)
        rows = self._rows.get(key)
        if rows is not None:
            return rows
        xp = self.xp
        k, n, s = self.base, self.n, self.n_states
        other = [i for i in range(n) if i not in subset]
        width = k ** len(subset)
        if direction == "effect":
            rows = _reduce(self._t4_eff, other)
            if other:
                rows = rows / float(k ** len(other))
            rows = rows.reshape(width, s)
        else:
            lik = _reduce(self._t4_cause, [1 + i for i in other]).reshape(s, width)
            rows = _normalize_rows(xp, lik.T, s)
        if self._rows_elements + width * s > self.cache_elements:
            self._rows = {}
            self._rows_elements = 0
        self._rows[key] = rows
        self._rows_elements += width * s
        return rows

    def onehot(self, purviews):
        """(S x len(purviews)*K^r) one-hot marginaliser for same-size purviews."""
        key = tuple(purviews)
        mat = self._onehot.get(key)
        if mat is not None:
            self._onehot.move_to_end(key)
            return mat
        r = len(purviews[0])
        width = self.base ** r
        host = np.zeros((self.n_states, len(purviews) * width), dtype=np.float64)
        rows = np.arange(self.n_states)
        for j, z in enumerate(purviews):
            host[rows, j * width + subset_keys(self.states, z, self.base)] = 1.0
        mat = host if _is_host_module(self.xp) else self.xp.asarray(host)
        size = int(host.size)
        while self._onehot and self._onehot_elements + size > self.cache_elements:
            _old_key, old = self._onehot.popitem(last=False)
            self._onehot_elements -= int(old.size)
        self._onehot[key] = mat
        self._onehot_elements += size
        return mat


def _jsd_last(xp, p, q, width):
    """Jensen-Shannon divergence (bits) along the last axis (host-kernel semantics)."""
    sp = p.sum(axis=-1, keepdims=True)
    sq = q.sum(axis=-1, keepdims=True)
    ok = (sp[..., 0] > 0) & (sq[..., 0] > 0)
    pn = p / xp.where(sp > 0, sp, 1.0)
    qn = q / xp.where(sq > 0, sq, 1.0)
    m = 0.5 * (pn + qn)
    m_safe = xp.where(m > 0, m, 1.0)
    log_p = xp.log(xp.where(pn > 0, pn, 1.0) / m_safe)
    log_q = xp.log(xp.where(qn > 0, qn, 1.0) / m_safe)
    tp = xp.where((pn > 0) & (m > 0), pn * log_p, 0.0)
    tq = xp.where((qn > 0) & (m > 0), qn * log_q, 0.0)
    js = 0.5 * (tp.sum(axis=-1) + tq.sum(axis=-1)) / _LOG2
    return xp.where(ok, js, 0.0)


def _group_purviews(purviews):
    by_size = OrderedDict()
    for z in purviews:
        z = tuple(sorted(int(x) for x in z))
        if len(z) < 2:
            continue  # no admissible bipartition: contributes 0
        by_size.setdefault(len(z), []).append(z)
    return OrderedDict(sorted(by_size.items()))


def _purview_part_splits(r):
    """
    (Z1 positions, Z2 positions) for every mechanism-part assignment of the
    purview bipartitions: Z1 runs over all non-empty proper position subsets,
    which enumerates both pairings (M1->Z1, M2->Z2) and (M1->Z2, M2->Z1) of
    every unordered purview bipartition exactly once.
    """
    out = []
    for pos_a, pos_b in enumerate_bipartitions(tuple(range(int(r)))):
        out.append((pos_a, pos_b))
        out.append((pos_b, pos_a))
    return out


def _split_product(a_src, b_src, z1, z2, shape):
    """p(Z1|M1) x p(Z2|M2) on the full purview from the parts' purview marginals."""
    qa = a_src.sum(axis=tuple(2 + i for i in z2), keepdims=True)
    qb = b_src.sum(axis=tuple(2 + i for i in z1), keepdims=True)
    return (qa * qb).reshape(shape)


def _phi_mechanism_states(ws, mechanism, uk, purviews_by_size, max_elements):
    """phi(m) = min(phi_e, phi_c) for all observed states ``uk`` of ``mechanism``."""
    xp = ws.xp
    k = ws.base
    s = ws.n_states
    m_len = len(mechanism)
    n_m = int(uk.size)
    digits = _decode_digits(uk, m_len, k)
    y_subsets = enumerate_subsets(mechanism, m_len)
    y_index = {y: i for i, y in enumerate(y_subsets)}
    pos = {node: i for i, node in enumerate(mechanism)}
    y_keys = {y: _encode_digits(digits[:, [pos[v] for v in y]], k) for y in y_subsets}
    n_y = len(y_subsets)
    m_bips = enumerate_bipartitions(mechanism)
    budget = max(1, int(max_elements))
    phi = {}
    for direction in ("effect", "cause"):
        best_z = np.zeros(n_m, dtype=float)
        for r, zs in purviews_by_size.items():
            width = k ** r
            splits = _purview_part_splits(r)
            m_chunk = max(1, min(n_m, budget // max(1, n_y * max(s, width))))
            z_chunk = max(1, min(len(zs), budget // max(1, n_y * m_chunk * width)))
            for m0 in range(0, n_m, m_chunk):
                m1 = min(n_m, m0 + m_chunk)
                mc = m1 - m0
                rows = xp.stack(
                    [
                        ws.cond_rows(direction, y)[
                            y_keys[y][m0:m1]
                            if _is_host_module(xp)
                            else xp.asarray(y_keys[y][m0:m1])
                        ]
                        for y in y_subsets
                    ]
                ).reshape(n_y * mc, s)
                for z0 in range(0, len(zs), z_chunk):
                    z_batch = zs[z0:z0 + z_chunk]
                    zc = len(z_batch)
                    marg = (rows @ ws.onehot(z_batch)).reshape(n_y, mc, zc, width)
                    marg = _normalize_rows(xp, marg, width)
                    p_full = marg[y_index[mechanism]][None]
                    shape_t = (mc, zc) + (k,) * r
                    q_shape = (mc, zc, width)
                    p_chunk = max(
                        1, min(len(splits), budget // max(1, mc * zc * width))
                    )
                    best = None
                    for part_ma, part_mb in m_bips:
                        a_src = marg[y_index[part_ma]].reshape(shape_t)
                        b_src = marg[y_index[part_mb]].reshape(shape_t)
                        for p0 in range(0, len(splits), p_chunk):
                            # Partitioned repertoires p(Z1|M1) x p(Z2|M2) for a batch
                            # of purview splits: each part's marginal is summed over
                            # the positions of the other part (keepdims) and the two
                            # broadcast to the full purview.
                            q = xp.stack(
                                [
                                    _split_product(a_src, b_src, z1, z2, q_shape)
                                    for z1, z2 in splits[p0:p0 + p_chunk]
                                ]
                            )
                            q = _normalize_rows(xp, q, width)
                            d = _jsd_last(xp, p_full, q, width).min(axis=0)
                            best = d if best is None else xp.minimum(best, d)
                    best = xp.where(xp.isfinite(best), best, 0.0)
                    z_max = np.asarray(_to_host(best.max(axis=1)), dtype=float)
                    best_z[m0:m1] = np.maximum(best_z[m0:m1], z_max)
        phi[direction] = best_z
    return np.minimum(phi["effect"], phi["cause"])


def psi_contribution(
    mechanisms,
    purviews,
    base,
    tpm,
    curr_obs,
    states_full,
    *,
    obs_weights=None,
    xp=None,
    workspace=None,
    max_elements=None,
):
    """
    Psi contribution of ``mechanisms`` for ``tpm`` (the xp counterpart of
    ``mpc_metrics._iim_phase1_chunk_contribution``): sum over mechanisms M of
    (1/|M|) * sum over observed M-states m of w(m) * min(phi_e, phi_c).
    Mechanisms or purviews of size 1 have no admissible bipartition and
    contribute 0. Returns a Python float.
    """
    if workspace is None:
        workspace = IIMXpWorkspace(xp, states_full, base)
    elif xp is not None and workspace.xp is not xp:
        raise ValueError("workspace was created for a different array module")
    workspace.bind(tpm)
    budget = (
        _env_int(XP_MAX_ELEMENTS_ENV, _DEFAULT_MAX_ELEMENTS)
        if max_elements is None
        else int(max_elements)
    )
    purviews_by_size = _group_purviews(purviews)
    terms = []
    for mech in mechanisms:
        mech = tuple(sorted(int(x) for x in mech))
        if len(mech) < 2 or not purviews_by_size:
            continue
        uk, wt = mechanism_state_weights(curr_obs, mech, base, obs_weights)
        if uk.size == 0:
            continue
        phi = _phi_mechanism_states(workspace, mech, uk, purviews_by_size, budget)
        phi_m = math.fsum((wt * phi).tolist())
        terms.append(phi_m / float(len(mech)))
    return float(math.fsum(terms)) if terms else 0.0
