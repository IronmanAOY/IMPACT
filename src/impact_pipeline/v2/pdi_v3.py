"""
PDI v3 (``pdi-v3-2026.10``): content differentiation as the number of
recurring, distinguishable multi-node states of a declared content bearer.

**Construct.** PDI is the number of recurring, distinguishable multi-node
states of a declared *content bearer*: all nodes except the declared access
(workspace) nodes. Ignition and other global access-state alternations are
not content states by construct. Content is not identifiable from passive,
unlabelled observation alone, so it is identified by the access-node
declaration. The full-bearer count is an exclusion-valid upper bound whose
PRESENT is not content-specific: with access nodes declared it is reported
as a point value only (never a status); without a declaration (sensor data)
it is the only reading, and the component is marked ``upper_bound``. ``B`` is
in bits of a count, not a quadratic statistic.

**Algorithm** (declared in :class:`PDIParams`; unchanged from v1 unless
stated):

1. Window patterns as v1 (nodes z-scored over the run, non-overlapping
   windows of ``w = 5`` samples, window mean), column-centred.
2. (new) ``r = min(10, rank)`` leading principal components, each scaled to
   unit variance (sphering).
3. (new) Split with offset: the windows are cut into 6 contiguous blocks
   after a cyclic rotation by ``o * n_w / 6`` windows, ``o`` in
   ``{0, 1/4, 1/2, 3/4}``; blocks alternate between halves A and B and the
   first window of every block (and of block 0 when the split is rotated) is
   purged, so no window of one half is next to a window of the other.
   ``o = 0`` is the v1 split.
4. For ``k = 2 .. k_max`` (``k_max = min(12, floor(min(n_A, n_B) / 3))``),
   every distinct k-means solution (Lloyd, k-means++, 8 starts, seed
   ``(seed * 1009 + k) mod 2^32``) fitted on one half is checked on the
   other: every state recurs (>= 3 held-out windows) and persists (mean
   held-out dwell >= 2.5 windows), and (new) every pair of states is
   separated by a density valley on its **Fisher axis**
   ``w = S_w^-1 (c_j - c_i)`` with ``S_w`` the Ledoit-Wolf-shrunk pooled
   within-state covariance of the fitting half under the candidate's own
   labels (held-out windows projected so that the centroids sit at 0 and 1;
   Gaussian kernel of bandwidth 0.1; valley ratio <= 0.4). A direction passes
   if one candidate passes; ``k`` passes if both directions pass; ``K_o`` is
   the largest passing ``k`` (1 if none).
5. (new) ``B`` = mean over the split offsets of ``log2 K_o`` (bits).
6. Null (family ``circular_shift``): 19 surrogates of the bearer, every node
   but node 0 shifted independently by >= 10 % of the run, each scored by the
   identical ``B``.
7. SE: delete-a-group jackknife over 10 contiguous time blocks (``B``
   recomputed on each replicate with the same offsets); ``se_df = 9``; SEs
   below ``1e-9`` bits are set to 0.
8. (new) Concordance: a component is *concordant* when all
   ``(10 + 1) x 4 = 44`` counts (the run and its 10 replicates at the 4
   offsets) are identical; ``se`` is then 0, ``se_method`` is
   :data:`SE_METHOD_CONCORDANT` (``concordant``) and ``se_df`` is None,
   because a concordant component carries no sampling SE (the SE-method
   contract of the status rule). A concordant component is **not** exact
   (``exact`` is always False): the status rule decides it through the
   concordance route only where the protocol admits that route for its
   (substrate, observation regime, bearer) cell, and otherwise it is
   ``UNDEFINED(NO_SAMPLING_SE)``. A non-concordant component with a zero
   jackknife SE keeps :data:`SE_METHOD`, so it is ``NO_SAMPLING_SE`` too.
   The bearer actually counted is ``details['counted_bearer']``
   (``non_workspace`` only when access nodes were declared and removed,
   ``full`` otherwise), the token of the route's bearer cell.

The v1 family-C carrier recording is declared not applicable before any run
(``NOT_APPLICABLE_OBSERVATION_MODEL``): its contexts are phase-coded on the
carrier, and neither window means nor window power of Re z carry them. The
count is then reported descriptively (point value only, no null or SE).

**Output.** :func:`compute_pdi_v3` returns a dict with ``estimate`` (``B``),
``value`` (excess ``B - nu``), the null moments and family, ``se``,
``se_df``, ``se_method``, ``concordant``, ``exact`` (always False),
``defined``, ``reason`` and ``details`` (declared and counted bearer and
its reading, per-offset counts, the jackknife SE and df before the
concordance rule, the full-bearer upper bound, per-k valley ratios, state
occupancy and dwell, optionally the partition for the bench evaluator's
oracle AMI);
:func:`component_fields` maps it onto the estimator fields of a
``mpc-bench-result/3`` component. The status is decided by the status rule,
never here.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline import mpc_metrics as mm
from impact_pipeline.v2 import numerics as NUM
from impact_pipeline.v2 import reasons as R

PRINCIPLE = "PDI"
ESTIMATOR_VERSION = "pdi-v3-2026.10"
MODE = "repertoire"
VARIANT = "unlabelled"
GEOMETRY = "discriminant"
ESTIMATOR_ID = f"compute_PDI:{MODE}_{GEOMETRY}@{ESTIMATOR_VERSION}"
NULL_FAMILY = "circular_shift"
JACKKNIFE_GROUPS = 10
SE_METHOD = "jackknife_contiguous_10"
# The SE method of a concordant component: no sampling SE (se = 0, no
# se_df); the status rule dispatches the concordance route on this name.
SE_METHOD_CONCORDANT = "concordant"
SE_METHODS = (SE_METHOD, SE_METHOD_CONCORDANT)
SE_ZERO_BITS = 1e-9
SPLIT_OFFSETS = (0.0, 0.25, 0.5, 0.75)

BEARER_NON_WORKSPACE = "non_workspace"
BEARER_FULL = "full"
BEARERS = (BEARER_NON_WORKSPACE, BEARER_FULL)
READING_CONTENT = "content"
READING_UPPER_BOUND = "upper_bound"

# Observation models on which PDI is declared not applicable before any run
# (UNDEFINED(NOT_APPLICABLE_OBSERVATION_MODEL), never ABSENT).
NOT_APPLICABLE_OBSERVATION_MODELS = {
    "C1": (
        "v1 family-C carrier recording (Re z at 20 Hz of 1.15-1.25 Hz "
        "carriers): the contexts are phase-coded on the carrier, so neither "
        "window means nor window power of Re z carry them"
    ),
}
OBSERVATION_MODEL_ALIASES = {"C": "C1"}

_SD_FLOOR = 1e-12


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PDIParams:
    """Declared estimator settings (bench defaults; hash-covered in the
    protocol through :meth:`to_dict`)."""

    window: int = 5
    features: str = "mean"
    n_components: int = 10
    n_blocks: int = 6
    gap: int = 1
    offsets: Tuple[float, ...] = SPLIT_OFFSETS
    max_states: int = 12
    n_init: int = 8
    min_state_windows: int = 3
    min_dwell: float = 2.5
    valley: float = 0.4
    bandwidth: float = 0.1
    n_null: int = 19
    null_min_shift: Optional[int] = None
    bearer: str = BEARER_NON_WORKSPACE
    full_bearer_upper_bound: bool = True

    def __post_init__(self):
        set_ = object.__setattr__
        set_(self, "offsets", tuple(float(o) for o in self.offsets))
        for name in ("window", "n_components", "n_blocks", "gap", "max_states",
                     "n_init", "min_state_windows", "n_null"):
            v = getattr(self, name)
            if isinstance(v, bool) or int(v) != v:
                raise ValueError(f"{name} must be an integer")
            set_(self, name, int(v))
        if self.window < 1 or self.n_components < 1 or self.n_init < 1:
            raise ValueError("window, n_components and n_init must be >= 1")
        if self.n_blocks < 2 or self.gap < 0 or self.max_states < 2:
            raise ValueError("n_blocks >= 2, gap >= 0 and max_states >= 2 required")
        if self.min_state_windows < 1 or self.n_null < 0:
            raise ValueError("min_state_windows >= 1 and n_null >= 0 required")
        if self.features not in mm.PDI_REPERTOIRE_FEATURES:
            raise ValueError(f"features must be one of {mm.PDI_REPERTOIRE_FEATURES}")
        if not self.offsets or any(not 0.0 <= o < 1.0 for o in self.offsets):
            raise ValueError("offsets must lie in [0, 1)")
        if len(set(self.offsets)) != len(self.offsets):
            raise ValueError("offsets must be distinct")
        for name in ("min_dwell", "valley", "bandwidth"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be > 0")
        if self.null_min_shift is not None and int(self.null_min_shift) < 1:
            raise ValueError("null_min_shift must be >= 1")
        if self.bearer not in BEARERS:
            raise ValueError(f"bearer must be one of {BEARERS}")

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["offsets"] = list(self.offsets)
        return out

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping]) -> "PDIParams":
        """Parameters from a protocol block; unknown keys are refused."""
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - names)
        if unknown:
            raise ValueError(f"unknown PDI parameters {unknown}")
        return cls(**dict(payload))

    @property
    def n_counts(self) -> int:
        """Counts behind the concordance rule: (G + 1) data sets x offsets."""
        return (JACKKNIFE_GROUPS + 1) * len(self.offsets)


def normalize_observation_model(observation_model) -> Optional[str]:
    """The canonical name of an observation model (``'C'`` is the v1
    family-C carrier recording ``'C1'``); None stays None."""
    if observation_model is None:
        return None
    name = str(observation_model).strip()
    return OBSERVATION_MODEL_ALIASES.get(name, name)


def not_applicable(observation_model) -> Optional[str]:
    """The declared reason why PDI is not applicable to an observation
    model, or None when it is applicable."""
    return NOT_APPLICABLE_OBSERVATION_MODELS.get(
        normalize_observation_model(observation_model))


def content_bearer(n_nodes: int, workspace_nodes=None,
                   bearer: str = BEARER_NON_WORKSPACE):
    """
    The rows PDI is counted on and the reading of the count:
    ``(rows, reading, excluded)``. With ``bearer='non_workspace'`` and access
    nodes declared, the rows are every node except the declared workspace
    nodes (reading ``content``); without a declaration, or with
    ``bearer='full'``, every node (reading ``upper_bound``: a content-agnostic
    upper bound whose PRESENT is not content-specific).
    """
    if bearer not in BEARERS:
        raise ValueError(f"bearer must be one of {BEARERS}")
    n = int(n_nodes)
    ws = sorted({int(i) for i in (workspace_nodes or [])})
    bad = [i for i in ws if not 0 <= i < n]
    if bad:
        raise ValueError(f"workspace nodes {bad} outside 0..{n - 1}")
    if bearer == BEARER_NON_WORKSPACE and ws:
        rows = np.asarray([i for i in range(n) if i not in set(ws)], dtype=np.int64)
        return rows, READING_CONTENT, ws
    return np.arange(n, dtype=np.int64), READING_UPPER_BOUND, []


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def ledoit_wolf(x) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf shrinkage covariance of the rows of ``x`` (centred; maximum
    likelihood scale ``1/n``) towards ``mu I`` with ``mu = tr(S) / p``:
    ``(1 - a) S + a mu I`` with ``a = min(b^2, d^2) / d^2``,
    ``d^2 = ||S - mu I||_F^2`` and ``b^2 = n^-2 sum_k ||x_k x_k' - S||_F^2``
    (Ledoit and Wolf 2004). Returns ``(covariance, a)``.
    """
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    s = xc.T @ xc / float(n)
    if p == 1:
        return s, 0.0
    mu = float(np.trace(s)) / p
    d2 = float(np.sum((s - mu * np.eye(p)) ** 2))
    x2 = xc * xc
    b2 = (float(np.sum(x2.T @ x2)) / n - float(np.sum(s * s))) / n
    b2 = min(b2, d2)
    a = 0.0 if b2 <= 0.0 or d2 <= 0.0 else b2 / d2
    return (1.0 - a) * s + a * mu * np.eye(p), float(a)


def fisher_axis(c_i, c_j, s_inv) -> Tuple[np.ndarray, float]:
    """``w = S_w^-1 (c_j - c_i)`` and the normaliser ``(c_j - c_i)' w``, so
    that ``t = (x - c_i)' w / normaliser`` puts ``c_i`` at 0 and ``c_j`` at 1."""
    u = np.asarray(c_j, dtype=float) - np.asarray(c_i, dtype=float)
    w = np.asarray(s_inv, dtype=float) @ u
    return w, float(u @ w)


def split_halves(n_win: int, n_blocks: int, gap: int, offset: float = 0.0):
    """
    Two interleaved halves of contiguous blocks after a cyclic rotation by
    ``round(offset * n_win / n_blocks)`` windows: ``(half, block)`` per
    window, ``-1`` for purged windows (the first ``gap`` windows of every
    block, and of block 0 when the split is rotated). ``offset = 0`` is the
    v1 split (``mpc_metrics._pdi_half_blocks``).
    """
    n_win, n_blocks, gap = int(n_win), int(n_blocks), int(gap)
    if not 0.0 <= float(offset) < 1.0:
        raise ValueError("offset must lie in [0, 1)")
    sh = int(round(float(offset) * n_win / float(n_blocks)))
    idx = (np.arange(n_win) + sh) % max(n_win, 1)
    half = np.full(n_win, -1, dtype=np.int64)
    block = np.full(n_win, -1, dtype=np.int64)
    for b, chunk in enumerate(np.array_split(np.arange(n_win), n_blocks)):
        pos = idx[chunk]
        if b > 0 or sh > 0:
            pos = pos[gap:]
        half[pos] = b % 2
        block[pos] = b
    return half, block


def _valley_ratio(t_i, t_j, bandwidth):
    def kde(tt, at):
        return np.mean(np.exp(-0.5 * ((tt[:, None] - at[None, :]) / bandwidth) ** 2),
                       axis=0)

    g_i = kde(t_i, np.asarray([np.mean(t_i), 0.5]))
    g_j = kde(t_j, np.asarray([0.5, np.mean(t_j)]))
    peak = min(float(g_i[0]), float(g_j[1]))
    return float((g_i[1] + g_j[0]) / max(peak, 1e-300))


def partition_check(c, x_fit, lab_fit, x_test, block_test, params: "PDIParams"):
    """
    Held-out check of one candidate partition (centroids ``c`` and labels
    ``lab_fit`` fitted on ``x_fit``): recurrence and dwell as v1, then the
    Fisher-axis valley of every pair of states. Returns ``(passed,
    worst_valley_ratio, min_dwell)``.
    """
    k = c.shape[0]
    lab = mm._pdi_nearest_centroid(x_test, c)
    counts = np.bincount(lab, minlength=int(k))
    if int(counts.min()) < params.min_state_windows:
        return False, float("nan"), float("nan")
    same = block_test[1:] == block_test[:-1]
    a, b = lab[:-1][same], lab[1:][same]
    dwell = float("inf")
    for i in range(int(k)):
        m = a == i
        if not np.any(m):
            return False, float("nan"), float("nan")
        dwell = min(dwell, 1.0 / max(1.0 - float(np.mean(b[m] == i)), 1e-12))
    if dwell < float(params.min_dwell):
        return False, float("nan"), float(dwell)
    s_w, _ = ledoit_wolf(x_fit - c[lab_fit])
    s_inv = np.linalg.pinv(s_w, hermitian=True)
    worst = 0.0
    for i in range(int(k)):
        for j in range(i + 1, int(k)):
            w, den = fisher_axis(c[i], c[j], s_inv)
            if not den > 1e-24:
                return False, float("inf"), float(dwell)
            t_i = ((x_test[lab == i] - c[i]) @ w) / den
            t_j = ((x_test[lab == j] - c[i]) @ w) / den
            worst = max(worst, _valley_ratio(t_i, t_j, float(params.bandwidth)))
            if worst > float(params.valley):
                return False, float(worst), float(dwell)
    return True, float(worst), float(dwell)


def separation_check(x_fit, x_test, block_test, k, rng, params: "PDIParams"):
    """One direction for ``k`` states: every distinct k-means solution on
    ``x_fit`` (lowest inertia first) is checked; the direction passes if one
    does. Returns the first passing ``(True, valley, dwell)``, otherwise the
    failing solution with the smallest finite valley ratio."""
    best = None
    for _, c, lab_fit in mm._pdi_kmeans_solutions(x_fit, k, rng, params.n_init):
        res = partition_check(c, x_fit, lab_fit, x_test, block_test, params)
        if res[0]:
            return res
        if best is None or (np.isfinite(res[1]) and not res[1] >= best[1]):
            best = res
    return best


def sphered_scores(x, params: "PDIParams"):
    """Window patterns of ``x``, column-centred, reduced to ``min(r_max,
    rank)`` principal components scaled to unit variance. Returns
    ``(scores, info)``; ``scores`` is None when the run has no variance."""
    pat = mm._pdi_window_patterns(np.asarray(x, dtype=float), params.window,
                                  params.features)
    pat = pat - pat.mean(axis=0, keepdims=True)
    info = {"n_windows": int(pat.shape[0]), "n_components": 0}
    if pat.shape[0] < 2:
        info["undefined_reason"] = "insufficient_timepoints"
        return None, info
    _, s, vt = NUM.svd(pat, full_matrices=False)
    if s.size == 0 or not np.isfinite(s).all() or s[0] <= 1e-12:
        info["undefined_reason"] = "no_variance"
        return None, info
    rank = int(np.sum(s > max(pat.shape) * np.finfo(float).eps * s[0]))
    r = int(max(1, min(params.n_components, rank)))
    scores = pat @ vt[:r].T
    scores = scores / np.maximum(scores.std(axis=0, keepdims=True), _SD_FLOOR)
    info.update({"n_components": r, "undefined_reason": None})
    return scores, info


def count_states(x, *, offset: float = 0.0, params=None, seed: int = 0,
                 describe: bool = False) -> dict:
    """
    ``K_o`` of one run at one split offset (algorithm steps 1-4). Returns
    ``{n_states, bits, k_max, n_windows, n_components, offset,
    undefined_reason}`` and, with ``describe``, per-k diagnostics.
    """
    p = PDIParams.from_mapping(params)
    scores, info = sphered_scores(x, p)
    out = {"n_states": 1, "bits": 0.0, "k_max": 1, "offset": float(offset)}
    out.update(info)
    if scores is None:
        return out
    half, block = split_halves(scores.shape[0], p.n_blocks, p.gap, offset)
    x_a, x_b = scores[half == 0], scores[half == 1]
    b_a, b_b = block[half == 0], block[half == 1]
    n_half = min(x_a.shape[0], x_b.shape[0])
    k_max = int(min(p.max_states, n_half // p.min_state_windows))
    out["k_max"] = k_max
    if k_max < 2:
        out["undefined_reason"] = "insufficient_windows"
        return out
    k_hat, per_k = 1, []
    for k in range(2, k_max + 1):
        rng = np.random.RandomState((int(seed) * 1009 + k) % (2 ** 32))
        ok_ab, v_ab, d_ab = separation_check(x_a, x_b, b_b, k, rng, p)
        ok_ba, v_ba, d_ba = False, float("nan"), float("nan")
        if ok_ab:
            ok_ba, v_ba, d_ba = separation_check(x_b, x_a, b_a, k, rng, p)
        passed = bool(ok_ab and ok_ba)
        if describe:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_k.append({
                    "k": k,
                    "passed": passed,
                    "valley_ratio": float(np.nanmax([v_ab, v_ba])),
                    "min_dwell_windows": float(np.nanmin([d_ab, d_ba])),
                })
        if passed:
            k_hat = k
    out["n_states"] = int(k_hat)
    out["bits"] = float(np.log2(k_hat))
    if describe:
        out["per_k"] = per_k
    return out


def split_averaged_bits(x, params=None, seed: int = 0, describe: bool = False) -> dict:
    """``B`` = mean over the split offsets of ``log2 K_o``. Returns ``{bits,
    counts, undefined_reason}`` (and per-offset diagnostics with
    ``describe``); ``bits`` is NaN when a count is undefined."""
    p = PDIParams.from_mapping(params)
    counts, per, reason = [], [], None
    for o in p.offsets:
        res = count_states(x, offset=o, params=p, seed=seed, describe=describe)
        if res.get("undefined_reason") is not None:
            reason = res["undefined_reason"]
            counts.append(None)
        else:
            counts.append(int(res["n_states"]))
        per.append(res)
    out = {"counts": counts, "undefined_reason": reason, "bits": float("nan")}
    if reason is None:
        out["bits"] = float(np.mean(np.log2(np.asarray(counts, dtype=float))))
    if describe:
        out["per_offset"] = per
    return out


def counts_concordant(counts, expected: Optional[int] = None) -> bool:
    """True iff every count is defined and all are identical (and, with
    ``expected``, there are exactly that many)."""
    flat = list(np.asarray(counts, dtype=object).reshape(-1))
    if expected is not None and len(flat) != int(expected):
        return False
    if not flat or any(c is None for c in flat):
        return False
    vals = np.asarray(flat, dtype=float)
    if not np.all(np.isfinite(vals)):
        return False
    return bool(np.all(vals == vals[0]))


def jackknife_keep(n_time: int, groups: int, group: int) -> np.ndarray:
    """Samples kept with contiguous block ``group`` of ``groups`` deleted
    (the v1 time-series rule, ``bench.export.block_keep``)."""
    from impact_pipeline.bench.export import block_keep

    return block_keep(int(n_time), int(groups), int(group))


def state_descriptors(x, k: int, params=None, seed: int = 0) -> dict:
    """Occupancy and mean dwell (windows) of the ``k``-state partition of all
    windows (k-means on the sphered scores, ``RandomState(seed)``, as v1
    describes its count), and the window labels."""
    p = PDIParams.from_mapping(params)
    scores, _ = sphered_scores(x, p)
    if scores is None or int(k) < 2:
        n = 0 if scores is None else scores.shape[0]
        return {"labels": np.zeros(n, dtype=np.int64), "occupancy": [1.0],
                "mean_dwell_windows": [float(n)]}
    rng = np.random.RandomState(int(seed) % (2 ** 32))
    _, lab = mm._pdi_kmeans(scores, int(k), rng, p.n_init)
    occ = np.bincount(lab, minlength=int(k)) / float(lab.size)
    change = np.flatnonzero(np.diff(lab) != 0)
    starts = np.concatenate([[0], change + 1])
    ends = np.concatenate([change + 1, [lab.size]])
    runs = ends - starts
    dwell = [float(np.mean(runs[lab[starts] == s])) if np.any(lab[starts] == s)
             else float("nan") for s in range(int(k))]
    return {"labels": lab.astype(np.int64), "occupancy": [float(v) for v in occ],
            "mean_dwell_windows": dwell}


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------
def _result(details, *, reason=None, estimate=float("nan"), null_mean=float("nan"),
            null_sd=float("nan"), n_null=0, se=float("nan"), se_df=None,
            se_method=None, concordant=False) -> dict:
    if reason is not None:
        R.lookup(reason)
    defined = reason is None
    return {
        "principle": PRINCIPLE,
        "estimator_version": ESTIMATOR_VERSION,
        "estimator_id": ESTIMATOR_ID,
        "defined": defined,
        "reason": reason,
        "estimate": float(estimate),
        "value": float(estimate - null_mean) if defined else float("nan"),
        "null_mean": float(null_mean),
        "null_sd": float(null_sd),
        "n_null": int(n_null),
        "null_family": NULL_FAMILY if n_null else None,
        "se": float(se),
        "se_df": None if se_df is None else float(se_df),
        "se_method": se_method,
        "concordant": bool(concordant),
        "exact": False,
        "details": details,
    }


def _not_defined(detail) -> str:
    return R.format_reason(R.NOT_DEFINED, R.clean_detail(detail))


def _definedness(x, params) -> Optional[str]:
    n_nodes, n_time = x.shape
    if n_nodes < 2:
        return "insufficient_regions"
    if n_time < 2 * params.window:
        return "insufficient_timepoints"
    if not np.all(np.isfinite(x)):
        return "non_finite_timeseries"
    if float(np.max(x.std(axis=1))) <= _SD_FLOOR:
        return "no_variance"
    return None


def _full_bearer(x_all, p, seed, return_partition) -> dict:
    """The full-bearer upper bound: a point value only (no null, no SE,
    never a status). It is reported, so a failure here is recorded and never
    touches the counted component."""
    out = {"bits": float("nan"), "counts": None, "undefined_reason": None,
           "reading": READING_UPPER_BOUND, "status": "never"}
    bad = _definedness(x_all, p)
    if bad is not None:
        out["undefined_reason"] = bad
        return out
    try:
        full = split_averaged_bits(x_all, p, seed)
        out.update(bits=full["bits"], counts=full["counts"],
                   undefined_reason=full["undefined_reason"])
        if return_partition and full["undefined_reason"] is None:
            kf = int(full["counts"][0])
            out["partition"] = {
                "bearer": READING_UPPER_BOUND, "offset": 0.0, "n_states": kf,
                "window_samples": int(p.window),
                "labels": state_descriptors(x_all, kf, p, seed)["labels"].tolist()}
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        out["undefined_reason"] = f"error:{type(exc).__name__}"
    return out


def compute_pdi_v3(
    ts,
    workspace_nodes: Optional[Sequence[int]] = None,
    *,
    params=None,
    seed: int = 0,
    null_seed: int = 0,
    observation_model=None,
    dt: Optional[float] = None,
    return_partition: bool = False,
    describe_not_applicable: bool = True,
) -> dict:
    """
    PDI v3 of one run (module docstring).

    ``ts`` is the node x time recording of the system bearer;
    ``workspace_nodes`` the declared access nodes (row indices of ``ts``);
    ``params`` the declared settings (:class:`PDIParams` or a mapping, with
    ``bearer`` ``'non_workspace'`` or ``'full'``); ``seed`` the k-means seed
    of every count (run, surrogates and replicates); ``null_seed`` the seed
    of the surrogate draws; ``observation_model`` names the recording
    (``'C1'``/``'C'`` is declared not applicable); ``dt`` (s) only adds the
    smallest countable dwell to the details; ``return_partition`` adds the
    offset-0 window labels of the counted and of the full bearer (for the
    bench evaluator's oracle AMI; the estimator never reads oracle
    channels). Undefined results carry ``NOT_APPLICABLE_OBSERVATION_MODEL``
    or ``NOT_DEFINED:<estimator reason>``.
    """
    p = PDIParams.from_mapping(params)
    x_all = np.asarray(ts, dtype=float)
    if x_all.ndim != 2:
        raise ValueError(f"ts must be 2D (nodes x time), got shape {x_all.shape}")
    rows, reading, excluded = content_bearer(x_all.shape[0], workspace_nodes, p.bearer)
    x = x_all[rows]
    model = normalize_observation_model(observation_model)
    details = {
        "estimator_id": ESTIMATOR_ID,
        "mode": MODE,
        "variant": VARIANT,
        "geometry": GEOMETRY,
        "params": p.to_dict(),
        "seed": int(seed),
        "null_seed": int(null_seed),
        "observation_model": model,
        "bearer": p.bearer,
        "counted_bearer": (BEARER_NON_WORKSPACE if reading == READING_CONTENT
                           else BEARER_FULL),
        "bearer_reading": reading,
        "content_specific": reading == READING_CONTENT,
        "bearer_nodes": [int(i) for i in rows],
        "excluded_workspace_nodes": excluded,
    }
    if dt is not None:
        details["min_countable_dwell_sec"] = float(p.min_dwell * p.window * float(dt))

    na = not_applicable(model)
    if na is not None:
        details["not_applicable"] = na
        if describe_not_applicable:
            try:
                if _definedness(x, p) is None:
                    desc = split_averaged_bits(x, p, seed)
                    details["descriptive"] = {
                        "bits": desc["bits"], "counts": desc["counts"],
                        "undefined_reason": desc["undefined_reason"]}
                else:
                    details["descriptive"] = {"undefined_reason": _definedness(x, p)}
            except Exception as exc:  # noqa: BLE001 - descriptive only
                details["descriptive"] = {"error": type(exc).__name__}
        return _result(details, reason=R.NOT_APPLICABLE_OBSERVATION_MODEL)

    bad = _definedness(x, p)
    if bad is not None:
        return _result(details, reason=_not_defined(bad))

    obs = split_averaged_bits(x, p, seed, describe=True)
    details["counts"] = obs["counts"]
    details["n_windows"] = int(obs["per_offset"][0].get("n_windows", 0))
    details["n_components"] = int(obs["per_offset"][0].get("n_components", 0))
    details["k_max"] = [int(r.get("k_max", 1)) for r in obs["per_offset"]]
    if obs["undefined_reason"] is not None:
        return _result(details, reason=_not_defined(obs["undefined_reason"]))
    details["per_k"] = {str(o): r.get("per_k", [])
                        for o, r in zip(p.offsets, obs["per_offset"])}
    details["n_states_at_cap"] = [bool(r["n_states"] == r["k_max"])
                                  for r in obs["per_offset"]]
    bits = float(obs["bits"])

    # null: circular-shift surrogates of the bearer, scored by the same B
    rng = np.random.RandomState(int(null_seed))
    null = []
    for _ in range(int(p.n_null)):
        try:
            surr = mm._surrogate_timeseries(x, "circular_shift", rng,
                                            min_shift=p.null_min_shift)
        except ValueError:
            return _result(details, reason=_not_defined("surrogates_unavailable"))
        null.append(split_averaged_bits(surr, p, seed)["bits"])
    null = np.asarray(null, dtype=float)
    details["null_values"] = [float(v) for v in null]
    fin = null[np.isfinite(null)]
    if fin.size < 2:
        return _result(details, reason=_not_defined("insufficient_surrogates"))
    nu, s_nu = float(np.mean(fin)), float(np.std(fin, ddof=1))

    # jackknife over contiguous time blocks; concordance over all counts
    reps, rep_counts = [], []
    for g in range(JACKKNIFE_GROUPS):
        keep = jackknife_keep(x.shape[1], JACKKNIFE_GROUPS, g)
        res = split_averaged_bits(x[:, keep], p, seed)
        reps.append(res["bits"])
        rep_counts.append(res["counts"])
    reps = np.asarray(reps, dtype=float)
    details["jackknife_bits"] = [float(v) for v in reps]
    details["jackknife_counts"] = rep_counts
    ok = reps[np.isfinite(reps)]
    if ok.size >= 2:
        g = ok.size
        se = float(np.sqrt((g - 1) / g * np.sum((ok - ok.mean()) ** 2)))
        se = 0.0 if se < SE_ZERO_BITS else se
        se_df = float(g - 1)
    else:
        se, se_df = float("nan"), None
    details["se_jackknife"] = se
    details["se_df_jackknife"] = se_df
    all_counts = [obs["counts"]] + rep_counts
    concordant = counts_concordant(all_counts, expected=p.n_counts)
    details["concordant"] = concordant
    details["n_concordance_counts"] = int(sum(len(c) for c in all_counts))
    if concordant:
        # no sampling SE: the route, not a t interval, decides it
        se, se_df, se_method = 0.0, None, SE_METHOD_CONCORDANT
    else:
        se_method = SE_METHOD

    # full-bearer upper bound (point value only) and partitions
    k0 = int(obs["counts"][0])
    desc = state_descriptors(x, k0, p, seed)
    details["state_occupancy"] = desc["occupancy"]
    details["state_mean_dwell_windows"] = desc["mean_dwell_windows"]
    if return_partition:
        details["partition"] = {"bearer": reading, "offset": 0.0, "n_states": k0,
                                "window_samples": int(p.window),
                                "labels": desc["labels"].tolist()}
    if reading == READING_CONTENT and p.full_bearer_upper_bound:
        details["full_bearer"] = _full_bearer(x_all, p, seed, return_partition)
    return _result(details, estimate=bits, null_mean=nu, null_sd=s_nu,
                   n_null=int(fin.size), se=se, se_df=se_df, se_method=se_method,
                   concordant=concordant)


def component_fields(result: Mapping) -> dict:
    """The estimator fields of a ``mpc-bench-result/3`` component
    (``estimate``, null moments and family, ``se``, ``se_df``,
    ``se_method``, ``details`` with the concordance flag and the bearer
    reading); the runner adds status, reason, ``c`` and the protocol fields.
    An undefined result carries no SE."""
    defined = bool(result.get("defined"))
    details = dict(result.get("details") or {})
    details["defined"] = defined
    details["estimator_reason"] = result.get("reason")
    details["concordant"] = bool(result.get("concordant"))
    details["exact"] = False
    return {
        "estimate": result.get("estimate") if defined else None,
        "null_mean": result.get("null_mean") if defined else None,
        "null_sd": result.get("null_sd") if defined else None,
        "n_null": int(result.get("n_null") or 0) if defined else 0,
        "null_family": result.get("null_family") if defined else None,
        "se": result.get("se") if defined else None,
        "se_df": result.get("se_df") if defined else None,
        "se_method": result.get("se_method") if defined else None,
        "details": details,
    }


__all__ = [
    "BEARERS",
    "BEARER_FULL",
    "BEARER_NON_WORKSPACE",
    "ESTIMATOR_ID",
    "ESTIMATOR_VERSION",
    "JACKKNIFE_GROUPS",
    "NOT_APPLICABLE_OBSERVATION_MODELS",
    "NULL_FAMILY",
    "PDIParams",
    "PRINCIPLE",
    "READING_CONTENT",
    "READING_UPPER_BOUND",
    "SE_METHOD",
    "SE_METHODS",
    "SE_METHOD_CONCORDANT",
    "SE_ZERO_BITS",
    "SPLIT_OFFSETS",
    "component_fields",
    "compute_pdi_v3",
    "content_bearer",
    "count_states",
    "counts_concordant",
    "fisher_axis",
    "jackknife_keep",
    "ledoit_wolf",
    "normalize_observation_model",
    "not_applicable",
    "partition_check",
    "separation_check",
    "split_averaged_bits",
    "split_halves",
    "sphered_scores",
    "state_descriptors",
]
