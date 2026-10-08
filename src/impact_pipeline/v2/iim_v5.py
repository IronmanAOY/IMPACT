"""
IIM v5 (``iim-v5-2026.10``): observational transition-level integration of
a declared macro process.

**Construct.** IIM v5 measures the part of the one-step transition structure
of the declared macro variables that is lost under the least destructive
unidirectional cut, from a state-by-node node-shrinkage TPM of
median-binarised macro signals, conditional on the declared recorded
drivers, in bits, in excess of an independence null that keeps each node's
own dynamics and its relation to the drivers. It is identified as loop
integration only under causal sufficiency relative to the recorded
variables, conditional independence of next states given state and drivers,
no instantaneous mixing and a full-rank grain, adequate occupancy, sampling
that resolves the interaction time scale, approximate stationarity and an
approximately Markov macro state. Under an unrecorded common driver it
measures observational dependence; IIM PRESENT on human data is therefore
not evidence of recurrent integration. At sensor level it is lagged
integration among orthogonalised rank-safe clusters. It is not IIT's Phi.
Degenerate TPMs are UNDEFINED, never ABSENT. ``Delta_Psi`` is quadratic in a
weak coupling amplitude, so ``delta = 0.10`` corresponds to about 0.32 of the
reference amplitude.

**Algorithm** (declared in :class:`IIMParams`; the Psi functional, the
node-shrinkage TPM and the cut TPMs are the frozen v1 building blocks):

1. Macro signals: the mean over the nodes of each declared macro node
   (:func:`macro_signals`), or the given rows.
2. Rank condition: the zero-lag correlation matrix of the macro signals must
   have ``lambda_min / lambda_max >= 1e-6``, else
   ``UNDEFINED(MACRO_RANK_DEFICIENT)`` (clusters that partition a montage
   under the average reference fail by construction). A macro node without
   variance can never visit half of the macro states, so it is
   ``INSUFFICIENT_OCCUPANCY``.
3. Pre-processing ``zca`` (sensor substrates): symmetric (Loewdin, ZCA)
   orthogonalisation ``C^(-1/2) (y - mean)``; zero-lag coupling is then not
   measured. Median binarisation per node makes the magnitude-preserving
   variant of Colclough et al. (2015) and plain ZCA the same discrete
   process.
4. Conditioning on declared recorded drivers, never on the hub or any other
   variable caused by the macro nodes. Binary drivers with at most four joint
   states are *stratified* (``strata``: one label per sample): per-stratum
   node-shrinkage TPMs, ``m = sum_s pi_s Delta_Psi(T_s, w_s)`` with ``pi_s``
   the share of transition pairs in stratum ``s``. Every other driver is
   *residualised* on the common input basis of
   :mod:`impact_pipeline.v2.declared_inputs` (``basis``: an
   :class:`~impact_pipeline.v2.declared_inputs.InputBasis` with the IIM lag
   set ``{0, 1, ..., l_max}``): each macro signal is projected, with an
   intercept, on :meth:`InputBasis.orthonormal_span` from the first complete
   sample on. The basis is rank-deficient by construction (a filtered copy at
   lag ``l`` is a combination of the copy at ``l + 1`` and the raw input at
   ``l``), so the projection is rank-revealing and the conditioning is
   counted by the basis's effective rank (``details['basis']['rank']``), not
   by its raw columns.
5. Median binarisation per node (v1 rule), transition pairs ``(x(t), x(t +
   lag))``; a pair belongs to the stratum of its current sample.
6. Occupancy gate: in every stratum all ``2^n`` macro states are visited as
   current states and the rarest row has at least ``N_min = 25`` transition
   pairs (the binomial SE of a transition probability is then <= 0.10 =
   delta), else ``UNDEFINED(INSUFFICIENT_OCCUPANCY)``. ``N_min`` may only be
   raised at calibration (50 or 100; :data:`N_MIN_ALLOWED`).
7. Statistic: ``Delta_Psi_K = Psi(T) - max_kappa Psi(T^kappa)`` (bits) with
   ``K`` the 14 ordered bipartitions (``directional``: A -> B severed, the
   receivers get independent noise; the primary form) or the 7 unordered
   ones (``bidirectional``: reported, with its own anchor). A TPM without
   mechanism-level integration (``Psi(T) <= 0``) has ``Delta_Psi = 0``, as v1
   scores such surrogates (v1 leaves such an observed run undefined). Psi is
   evaluated by the array-module (``xp``) kernel of
   :mod:`impact_pipeline.iim_xp` on NumPy; this is the only IIM path that
   uses it, the v1 ``compute_IIM`` keeps its own kernel resolution. Without
   strata, basis (or with an empty one) or pre-processing the estimate and
   every null draw equal v1 ``compute_IIM`` (bins 2, node shrinkage) for
   either cut mode wherever v1 is defined.
8. Independence null (``K = 39`` on the bench, 19 in family-B cells; fewer
   than 19 finite draws: ``NO_NULL_CALIBRATION``). Without strata: the v1
   circular shift of every macro signal but the first (shifts of at least
   ``max(lag + 1, ceil(0.1 T))`` samples; the v1 surrogate draws). With a
   basis the order is **shift, then project**: the pre-processed series is
   shifted relative to the others and to ``U``, then every conditioning
   regression is fitted on the surrogate (``project_then_shift``, the
   non-adopted order, is available for the calibration report). With
   strata: the **stratified pair rotation** rotates, within every stratum,
   the pair sequence ``(x_i(t), x_i(t + lag))`` of every node but the first
   by an independent shift in ``[ceil(0.1 N_s), N_s - ceil(0.1 N_s)]``;
   each node's own transitions within every stratum are kept exactly and
   cross-node alignment is destroyed.
9. Rank p-value ``p_ind = (1 + #{m*_b >= m}) / (K + 1)`` over the finite
   draws (exact under exchangeability). The status rule's IIM rank gate
   turns a PRESENT with ``p_ind > 0.05`` into
   ``UNDEFINED(INCONCLUSIVE:NULL_NOT_EXCEEDED)``; it never gives ABSENT.
10. Sampling SE: circular block bootstrap of the analysed series (and of the
    strata labels), blocks of ``ceil(0.1 T)`` samples (10 blocks), ``B =
    50``, the statistic recomputed on every resample. ``se_df = 12`` was
    declared before any data (the block-count approximation 12.5 of the
    SE's variability, rounded down; never ``B - 1``); calibration (CD-3)
    lowered it to 9, because a one-sided 1 % tail exceeded 0.02 in some
    (protocol, class) twin cells, each by one session. That is the rule's
    conservative outcome (a larger q, fewer ABSENT calls), not evidence of
    anti-conservative tails. The other calibration outcome, the
    delete-a-group jackknife over 10 contiguous blocks (``se_df = 9``), was
    not taken.
11. Observation gate: a ``sensor`` or ``source_estimate`` observation is
    ``UNDEFINED(OBSERVATION_MIXED_NOT_ADMITTED)`` unless an admitting
    registry entry exists (``observation_admitted=True``; the forward-arm
    admission runs score with it, because their scoring is the admission
    test). The data-definedness gates (rank, occupancy) come first, so the v1
    quadrant montage under the average reference keeps its own reason.

**Calibration parameters** (decided on development seeds by the
calibration package, declared once here): ``N_min``
(:data:`N_MIN_DEFAULT`, :data:`N_MIN_ALLOWED`), the bootstrap ``se_df``
(:data:`BOOTSTRAP_SE_DF`, :data:`BOOTSTRAP_SE_DF_ALLOWED`) and the SE
method (:data:`SE_METHOD_DEFAULT`, fallback :data:`SE_METHOD_JACKKNIFE`).

**Exact path.** :func:`exact_delta_psi` evaluates a known TPM with its
stationary state weights through the same Psi machinery (it equals v1
``compute_IIM_from_tpm``); :func:`exact_iim` adds the background-conditioned
target of the family-B hidden driver (per-driver TPMs, 0 at every driver
weight). Exact components carry the SE method ``exact`` (``exact = True``,
no sampling SE): the status rule decides them on ``c`` alone, and a protocol
that scores them must admit ``exact`` among its IIM SE methods.

**Output.** :func:`compute_iim_v5` returns a dict with the primary cut
mode's ``estimate`` (``m``), ``value`` (excess ``m - nu``), null moments and
family, ``p_ind``, ``se``, ``se_df``, ``se_method``, ``defined``, ``reason``,
``cuts`` (the same fields per computed cut mode) and ``details``
(occupancy, strata, rank condition, pre-processing, basis rank, null order,
surrogate and bootstrap values). :func:`component_fields` maps it onto the
estimator fields of a ``mpc-bench-result/3`` component and :func:`evidence`
onto a :class:`~impact_pipeline.evidence_v2.ComponentEvidenceV2`. The status
is decided by the status rule, never here. :func:`select_cut_modes` reads
the result of a subset of the computed cut modes from a run (every cut mode
is computed on the same pairs, surrogates and resamples), so a reported
form need not recompute a cut its primary run already reports.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline import iim_xp
from impact_pipeline import mpc_metrics as mm
from impact_pipeline.v2 import reasons as R

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
PRINCIPLE = "IIM"
ESTIMATOR_VERSION = "iim-v5-2026.10"
CUT_MODES = tuple(iim_xp.CUT_MODES)  # ("bidirectional", "directional")
PRIMARY_CUT_MODE = "directional"
SECONDARY_CUT_MODES = ("bidirectional",)
BINS = 2
TPM_ESTIMATOR = "node_shrinkage"
TPM_ALPHA = 1e-3  # v1 compute_IIM default
PSI_KERNEL = "xp"
MIN_MACRO_NODES = 2
MAX_MACRO_NODES = 6

NULL_FAMILY_SHIFT = "circular_shift"
NULL_FAMILY_STRATIFIED = "stratified_pair_rotation"
NULL_FAMILIES = (NULL_FAMILY_SHIFT, NULL_FAMILY_STRATIFIED)
NULL_ORDER_SHIFT_THEN_PROJECT = "shift_then_project"
NULL_ORDER_PROJECT_THEN_SHIFT = "project_then_shift"
NULL_ORDERS = (NULL_ORDER_SHIFT_THEN_PROJECT, NULL_ORDER_PROJECT_THEN_SHIFT)
N_NULL_BENCH = 39
N_NULL_FAMILY_B = 19
MIN_NULL_DRAWS = 19
NULL_MIN_SHIFT_FRAC = 0.1
RANK_GATE_ALPHA = 0.05
RANK_TOL = 1e-6
MAX_STRATA = 4
PREPROCESS_NONE = "none"
PREPROCESS_ZCA = "zca"
PREPROCESSINGS = (PREPROCESS_NONE, PREPROCESS_ZCA)

# Calibration parameters (development calibration, CD-3), declared once.
N_MIN_DEFAULT = 25
N_MIN_ALLOWED = (25, 50, 100)
SE_METHOD_BOOTSTRAP = "circular_block_bootstrap_10pct_B50"
SE_METHOD_JACKKNIFE = "jackknife_contiguous_10"
SE_METHOD_EXACT = "exact"
SE_METHODS = (SE_METHOD_BOOTSTRAP, SE_METHOD_JACKKNIFE)
SE_METHOD_DEFAULT = SE_METHOD_BOOTSTRAP
BOOTSTRAP_BLOCK_FRAC = 0.10
BOOTSTRAP_REPLICATES = 50
# Declared 12 before any data; CD-3 lowered it to 9 (the one-sided 1 % tail
# of the bootstrap exceeded 0.02 in some (protocol, class) twin cells).
BOOTSTRAP_SE_DF = 9.0
BOOTSTRAP_SE_DF_ALLOWED = (12.0, 9.0)
JACKKNIFE_GROUPS = 10
JACKKNIFE_SE_DF = 9.0

# Observation stages (records vocabulary) whose observation mixes sources.
MIXED_OBSERVATION_STAGES = ("sensor", "source_estimate")
OBSERVATION_STAGES = ("source", "sensor", "source_estimate", "bold")

# Keys of the estimator's own random streams, SeedSequence([seed, key]).
STRATIFIED_NULL_STREAM = 1
BOOTSTRAP_STREAM = 2

_SD_FLOOR = 1e-12


def estimator_id(cut_mode: str = PRIMARY_CUT_MODE) -> str:
    """``compute_IIM:<cut mode>@iim-v5-2026.10``."""
    if cut_mode not in CUT_MODES:
        raise ValueError(f"cut_mode must be one of {CUT_MODES}, got {cut_mode!r}")
    return f"compute_{PRINCIPLE}:{cut_mode}@{ESTIMATOR_VERSION}"


ESTIMATOR_ID = estimator_id(PRIMARY_CUT_MODE)


# --------------------------------------------------------------------------
# declared settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class IIMParams:
    """
    The declared IIM v5 settings of a protocol. ``cut_mode`` is the primary
    form, ``report_cut_modes`` the reported ones (computed on the same
    surrogates and resamples). ``n_null`` is ``K``; ``se_method`` is
    :data:`SE_METHOD_BOOTSTRAP`, :data:`SE_METHOD_JACKKNIFE` or None (no
    sampling SE: the status rule then gives ``NO_SAMPLING_SE``);
    ``bootstrap_se_df`` is the declared ``se_df`` of the bootstrap;
    ``null_order`` is the order of shift and projection of a residualised
    null (``shift_then_project``; the other order is reported only).
    """

    cut_mode: str = PRIMARY_CUT_MODE
    report_cut_modes: Tuple[str, ...] = SECONDARY_CUT_MODES
    n_null: int = N_NULL_BENCH
    n_min: int = N_MIN_DEFAULT
    rank_tol: float = RANK_TOL
    preprocess: str = PREPROCESS_NONE
    se_method: Optional[str] = SE_METHOD_DEFAULT
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    bootstrap_block_frac: float = BOOTSTRAP_BLOCK_FRAC
    bootstrap_se_df: float = BOOTSTRAP_SE_DF
    jackknife_groups: int = JACKKNIFE_GROUPS
    null_order: str = NULL_ORDER_SHIFT_THEN_PROJECT
    null_min_shift_frac: float = NULL_MIN_SHIFT_FRAC
    max_strata: int = MAX_STRATA

    def __post_init__(self):
        put = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731
        if self.cut_mode not in CUT_MODES:
            raise ValueError(f"cut_mode must be one of {CUT_MODES}")
        reported = tuple(str(c) for c in (self.report_cut_modes or ()))
        if any(c not in CUT_MODES for c in reported):
            raise ValueError(f"report_cut_modes must be cut modes of {CUT_MODES}")
        put("report_cut_modes", tuple(c for i, c in enumerate(reported)
                                      if c != self.cut_mode and c not in reported[:i]))
        if int(self.n_null) < 0:
            raise ValueError("n_null must be >= 0")
        put("n_null", int(self.n_null))
        if int(self.n_min) not in N_MIN_ALLOWED:
            raise ValueError(f"n_min must be one of {N_MIN_ALLOWED} (it may only be "
                             "raised at calibration)")
        put("n_min", int(self.n_min))
        if not (0.0 < float(self.rank_tol) < 1.0):
            raise ValueError("rank_tol must lie in (0, 1)")
        if self.preprocess not in PREPROCESSINGS:
            raise ValueError(f"preprocess must be one of {PREPROCESSINGS}")
        if self.se_method is not None and self.se_method not in SE_METHODS:
            raise ValueError(f"se_method must be one of {SE_METHODS} or None")
        if int(self.bootstrap_replicates) < 2:
            raise ValueError("bootstrap_replicates must be >= 2")
        put("bootstrap_replicates", int(self.bootstrap_replicates))
        if not (0.0 < float(self.bootstrap_block_frac) <= 0.5):
            raise ValueError("bootstrap_block_frac must lie in (0, 0.5]")
        if float(self.bootstrap_se_df) not in BOOTSTRAP_SE_DF_ALLOWED:
            raise ValueError(f"bootstrap_se_df must be one of {BOOTSTRAP_SE_DF_ALLOWED}")
        put("bootstrap_se_df", float(self.bootstrap_se_df))
        if int(self.jackknife_groups) < 2:
            raise ValueError("jackknife_groups must be >= 2")
        put("jackknife_groups", int(self.jackknife_groups))
        if self.null_order not in NULL_ORDERS:
            raise ValueError(f"null_order must be one of {NULL_ORDERS}")
        if not (0.0 < float(self.null_min_shift_frac) < 0.5):
            raise ValueError("null_min_shift_frac must lie in (0, 0.5)")
        if not (1 <= int(self.max_strata) <= MAX_STRATA):
            raise ValueError(f"max_strata must lie in 1..{MAX_STRATA}")
        put("max_strata", int(self.max_strata))

    @property
    def cut_modes(self) -> Tuple[str, ...]:
        """The primary cut mode, then the reported ones."""
        return (self.cut_mode,) + tuple(self.report_cut_modes)

    @property
    def se_method_name(self) -> Optional[str]:
        """The SE method of the result: the contract name for the declared
        bootstrap (10 % blocks, B = 50) or jackknife (10 groups); another
        configuration is named after its settings and is outside the
        SE-method contract (the status rule then gives ``INVALID_SE``)."""
        if self.se_method == SE_METHOD_BOOTSTRAP:
            pct = round(100.0 * self.bootstrap_block_frac, 6)
            name = (f"circular_block_bootstrap_{pct:g}pct_"
                    f"B{self.bootstrap_replicates}")
            return name
        if self.se_method == SE_METHOD_JACKKNIFE:
            return f"jackknife_contiguous_{self.jackknife_groups}"
        return None

    @property
    def se_df(self) -> Optional[float]:
        if self.se_method == SE_METHOD_BOOTSTRAP:
            return float(self.bootstrap_se_df)
        if self.se_method == SE_METHOD_JACKKNIFE:
            return float(self.jackknife_groups - 1)
        return None

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["report_cut_modes"] = list(self.report_cut_modes)
        return out

    @classmethod
    def from_mapping(cls, payload=None) -> "IIMParams":
        if payload is None:
            return cls()
        if isinstance(payload, IIMParams):
            return payload
        if not isinstance(payload, Mapping):
            raise TypeError("params must be an IIMParams or a mapping")
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - names)
        if unknown:
            raise ValueError(f"unknown IIM v5 parameters {unknown}")
        data = dict(payload)
        if "report_cut_modes" in data:
            data["report_cut_modes"] = tuple(data["report_cut_modes"] or ())
        return cls(**data)


# --------------------------------------------------------------------------
# pre-processing
# --------------------------------------------------------------------------
def macro_signals(ts, macro_nodes: Optional[Mapping] = None) -> np.ndarray:
    """The macro signals (macro nodes x time): the mean over the nodes of each
    declared macro node, in declaration order, or the rows of ``ts`` when no
    grain is declared."""
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"ts must be 2D (nodes x time), got shape {x.shape}")
    if macro_nodes is None:
        return x.copy()
    rows = []
    for name, idx in dict(macro_nodes).items():
        idx = np.asarray(list(idx), dtype=int)
        if idx.size == 0 or np.any(idx < 0) or np.any(idx >= x.shape[0]):
            raise ValueError(f"macro node {name!r} has invalid node indices")
        rows.append(x[idx].mean(axis=0))
    return np.vstack(rows)


def zero_lag_rank(y, tol: float = RANK_TOL) -> dict:
    """The rank condition: eigenvalues of the zero-lag correlation matrix of
    the macro signals, ``ratio = lambda_min / lambda_max`` and ``ok = ratio >=
    tol``. Clusters whose means are linearly dependent (for example four
    quadrants that partition a montage under the average reference, whose
    means sum to zero) give ``ratio`` near 1e-16."""
    y = np.asarray(y, dtype=float)
    yc = y - y.mean(axis=1, keepdims=True)
    sd = yc.std(axis=1)
    if np.any(~np.isfinite(sd)) or np.any(sd <= _SD_FLOOR):
        return {"ok": False, "ratio": float("nan"), "eigenvalues": None,
                "constant_nodes": [int(i) for i in np.flatnonzero(~(sd > _SD_FLOOR))]}
    z = yc / sd[:, None]
    corr = (z @ z.T) / z.shape[1]
    w = np.linalg.eigvalsh(0.5 * (corr + corr.T))
    lam_max = float(w[-1])
    ratio = float(max(w[0], 0.0) / lam_max) if lam_max > 0 else float("nan")
    return {"ok": bool(math.isfinite(ratio) and ratio >= float(tol)), "ratio": ratio,
            "eigenvalues": [float(v) for v in w], "constant_nodes": []}


def zca(y) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric (Loewdin, ZCA) orthogonalisation of the macro signals:
    ``W (y - mean)`` with ``W = C^(-1/2)`` the symmetric inverse square root
    of the zero-lag covariance ``C``. Returns ``(y_tilde, W)``; ``y_tilde``
    has the identity as its zero-lag covariance. Needs a full-rank ``C``."""
    y = np.asarray(y, dtype=float)
    yc = y - y.mean(axis=1, keepdims=True)
    cov = (yc @ yc.T) / yc.shape[1]
    w, v = np.linalg.eigh(0.5 * (cov + cov.T))
    if not (w[0] > _SD_FLOOR * max(float(w[-1]), _SD_FLOOR)):
        raise np.linalg.LinAlgError("zero-lag covariance is singular")
    W = (v * (1.0 / np.sqrt(w))) @ v.T
    return W @ yc, W


def residualise(y, span) -> np.ndarray:
    """Residuals of every row of ``y`` (macro x time) after a least-squares
    fit on an intercept and the columns of ``span`` (time x r, centred and
    orthonormal, e.g. :meth:`InputBasis.orthonormal_span`)."""
    y = np.asarray(y, dtype=float)
    yc = y - y.mean(axis=1, keepdims=True)
    q = np.asarray(span, dtype=float)
    if q.ndim != 2 or q.shape[0] != yc.shape[1]:
        raise ValueError("span must be time x rank with the series' length")
    if q.shape[1] == 0:
        return yc
    return yc - (yc @ q) @ q.T


def binarise(y) -> np.ndarray:
    """Median binarisation per node (the v1 rule for 2 bins)."""
    return mm._iim_discretize_per_node(np.asarray(y, dtype=float), BINS)


def transition_pairs(x, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Current and next states ``(x(t), x(t + lag))`` as (pairs x nodes)
    int16 arrays."""
    x = np.asarray(x)
    lag = int(lag)
    if lag < 1 or lag >= x.shape[1]:
        raise ValueError("lag must lie in 1 .. n_time - 1")
    return (x[:, :-lag].T.astype(np.int16), x[:, lag:].T.astype(np.int16))


def joint_strata(labels, max_strata: int = MAX_STRATA) -> Tuple[np.ndarray, tuple]:
    """The joint state of the stratifying drivers per sample: ``labels`` is
    one integer label per sample (one driver) or drivers x samples. Returns
    the stratum index per sample (0 .. S - 1, in sorted order of the joint
    labels) and the joint labels. Raises ``ValueError`` for more than
    ``max_strata`` joint states (such drivers are residualised instead)."""
    a = np.asarray(labels)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] == 0:
        raise ValueError("strata are one label per sample (or drivers x samples)")
    if a.dtype.kind == "b":
        a = a.astype(np.int64)
    if a.dtype.kind == "f":
        if not np.all(np.isfinite(a)) or not np.all(a == np.round(a)):
            raise ValueError("strata labels must be integers")
        a = a.astype(np.int64)
    if a.dtype.kind not in "iu":
        raise ValueError("strata labels must be integers")
    keys, codes = np.unique(a.T, axis=0, return_inverse=True)
    if keys.shape[0] > int(max_strata):
        raise ValueError(f"{keys.shape[0]} joint driver states exceed max_strata="
                         f"{max_strata}; residualise these drivers instead")
    alphabet = tuple(tuple(int(v) for v in k) for k in keys)
    return np.asarray(codes, dtype=np.int64).reshape(-1), alphabet


def shrinkage_intensity(curr, nxt) -> np.ndarray:
    """The James-Stein intensities ``lambda`` (states x nodes) of the v1
    node-shrinkage TPM of a set of transition pairs: the weight of each
    row's no-cross-influence target (the node's own transition) against the
    row's maximum-likelihood estimate; 1 for rows with at most one pair.
    ``p_i(.|x) = lambda target + (1 - lambda) ML`` reproduces the v1 TPM."""
    curr = np.asarray(curr, dtype=np.int64)
    nxt = np.asarray(nxt, dtype=np.int64)
    n = curr.shape[1]
    states = np.asarray(_workspace(n).states, dtype=np.int64)
    n_states = states.shape[0]
    keys = iim_xp.subset_keys(curr, tuple(range(n)), BINS)
    n_obs = np.bincount(keys, minlength=n_states).astype(float)
    lam = np.ones((n_states, n))
    for i in range(n):
        counts = np.zeros((n_states, BINS))
        np.add.at(counts, (keys, nxt[:, i]), 1.0)
        self_counts = np.zeros((BINS, BINS))
        np.add.at(self_counts, (curr[:, i], nxt[:, i]), 1.0)
        self_p = (self_counts + TPM_ALPHA) / (self_counts.sum(axis=1, keepdims=True)
                                             + TPM_ALPHA * BINS)
        target = self_p[states[:, i]]
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = np.where(n_obs[:, None] > 0, counts / np.maximum(n_obs[:, None], 1.0),
                             target)
            num = 1.0 - np.sum(theta * theta, axis=1)
            den = (n_obs - 1.0) * np.sum((target - theta) ** 2, axis=1)
            li = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
        lam[:, i] = np.where(n_obs <= 1.0, 1.0, np.clip(li, 0.0, 1.0))
    return lam


def occupancy(curr, strata=None, n_min: int = N_MIN_DEFAULT, nxt=None) -> dict:
    """Occupancy diagnostics and the gate (C4) per stratum: counts of every
    macro state as a current state, visited states, rarest visited row, the
    share of the two most frequent states, the effective number of states
    ``exp(H)``, the total-variation distance between the state distributions
    of the first and the second half of the stratum's pairs and, with
    ``nxt``, the uniform-weight mean shrinkage intensity (the share of the
    repertoires supplied by the no-cross-influence target). ``passes`` iff in
    every stratum all ``2^n`` states are visited and the rarest row has
    ``>= n_min`` pairs."""
    curr = np.asarray(curr)
    n = curr.shape[1]
    n_states = BINS ** n
    keys = iim_xp.subset_keys(curr, tuple(range(n)), BINS)
    s = np.zeros(keys.size, dtype=np.int64) if strata is None else np.asarray(strata)
    out = {"n_min": int(n_min), "n_states": int(n_states), "strata": []}
    passes = True
    for code in np.unique(s):
        sel = s == code
        cnt = np.bincount(keys[sel], minlength=n_states)
        vis = cnt[cnt > 0]
        tot = float(cnt.sum())
        p = vis / tot if tot > 0 else vis
        k_s = keys[sel]
        half = k_s.size // 2
        p1 = np.bincount(k_s[:half], minlength=n_states) / max(half, 1)
        p2 = np.bincount(k_s[half:], minlength=n_states) / max(k_s.size - half, 1)
        row = {
            "stratum": int(code),
            "n_pairs": int(cnt.sum()),
            "counts": [int(v) for v in cnt],
            "n_visited": int(vis.size),
            "n_unvisited": int(n_states - vis.size),
            "rarest_visited_row": int(vis.min()) if vis.size else 0,
            "top2_share": float(np.sort(cnt)[::-1][:2].sum() / tot) if tot else float("nan"),
            "effective_states": float(np.exp(-np.sum(p * np.log(p)))) if vis.size else 0.0,
            "split_half_tv": float(0.5 * np.abs(p1 - p2).sum()) if half else float("nan"),
        }
        if nxt is not None:
            row["mean_shrinkage"] = float(np.mean(shrinkage_intensity(
                curr[sel], np.asarray(nxt)[sel])))
        row["passes"] = bool(row["n_unvisited"] == 0 and row["rarest_visited_row"] >= n_min)
        passes = passes and row["passes"]
        out["strata"].append(row)
    out["passes"] = bool(passes and out["strata"])
    out["rarest_row"] = min((r["rarest_visited_row"] for r in out["strata"]), default=0)
    out["n_unvisited"] = max((r["n_unvisited"] for r in out["strata"]), default=n_states)
    return out


# --------------------------------------------------------------------------
# Delta_Psi (xp kernel, v1 building blocks)
# --------------------------------------------------------------------------
_WORKSPACES: Dict[int, "iim_xp.IIMXpWorkspace"] = {}


def _workspace(n: int):
    ws = _WORKSPACES.get(int(n))
    if ws is None:
        ws = iim_xp.IIMXpWorkspace(np, iim_xp.state_table(int(n), BINS), BINS)
        _WORKSPACES[int(n)] = ws
    return ws


def _subsets(n: int):
    return tuple(iim_xp.enumerate_subsets(tuple(range(int(n))), int(n)))


def system_cuts(n: int, cut_mode: str):
    """The system cuts of a cut mode, in the order of v1 ``compute_IIM``:
    every unordered bipartition (``bidirectional``), or both orders of every
    bipartition (``directional``: 14 ordered cuts for 4 nodes)."""
    if cut_mode not in CUT_MODES:
        raise ValueError(f"cut_mode must be one of {CUT_MODES}")
    bips = iim_xp.enumerate_bipartitions(tuple(range(int(n))))
    if cut_mode == "directional":
        return [cut for a, b in bips for cut in ((a, b), (b, a))]
    return list(bips)


def _psi(tpm, curr_obs, weights, n):
    ws = _workspace(n)
    subs = _subsets(n)
    return float(iim_xp.psi_contribution(subs, subs, BINS, tpm, curr_obs, ws.states,
                                         obs_weights=weights, xp=np, workspace=ws))


def _delta_psi_tpm(tpm, curr_obs, weights, cut_modes, n, zero_rule=True) -> dict:
    """``{cut_mode: {delta_psi, psi_full, psi_mip_preserved, mip_cut}}`` of one
    TPM; with ``zero_rule`` a TPM with ``Psi <= 0`` has ``Delta_Psi = 0``."""
    tpm = np.asarray(tpm, dtype=float)
    psi_full = _psi(tpm, curr_obs, weights, n)
    out = {}
    for mode in cut_modes:
        best, best_cut = -math.inf, None
        for a, b in system_cuts(n, mode):
            tc = iim_xp.cut_tpm(tpm, n, BINS, a, b, cut_mode=mode, xp=np)
            v = _psi(tc, curr_obs, weights, n)
            if v > best:
                best, best_cut = v, (tuple(int(i) for i in a), tuple(int(i) for i in b))
        zero = bool(zero_rule and not (math.isfinite(psi_full) and psi_full > 0))
        out[mode] = {
            "delta_psi": 0.0 if zero else float(psi_full - best),
            "psi_full": float(psi_full),
            "psi_mip_preserved": float(best),
            "mip_cut": [list(best_cut[0]), list(best_cut[1])] if best_cut else None,
            "psi_full_nonpositive": zero,
        }
    return out


def node_shrinkage_tpm(curr, nxt) -> np.ndarray:
    """The v1 state-by-node node-shrinkage TPM of a set of transition pairs."""
    curr = np.asarray(curr)
    n = curr.shape[1]
    keys = iim_xp.subset_keys(curr, tuple(range(n)), BINS)
    return mm._iim_node_conditional_tpm(keys, np.asarray(nxt), _workspace(n).states,
                                        BINS, TPM_ALPHA, TPM_ESTIMATOR)


def delta_psi(curr, nxt, cut_modes=CUT_MODES) -> dict:
    """``Delta_Psi`` (bits) per cut mode of one set of transition pairs (one
    stratum): node-shrinkage TPM, empirical current-state weights."""
    curr = np.asarray(curr, dtype=np.int16)
    n = curr.shape[1]
    tpm = node_shrinkage_tpm(curr, nxt)
    return _delta_psi_tpm(tpm, curr, None, tuple(cut_modes), n)


def conditioned_delta_psi(curr, nxt, strata=None, cut_modes=CUT_MODES) -> dict:
    """The background-conditioned statistic ``m = sum_s pi_s Delta_Psi(T_s,
    w_s)`` per cut mode (``pi_s`` the share of pairs in stratum ``s``):
    ``{"m": {cut: m}, "strata": [{stratum, n_pairs, share, delta_psi}]}``.
    Without strata it is ``Delta_Psi`` of all pairs."""
    curr = np.asarray(curr, dtype=np.int16)
    nxt = np.asarray(nxt, dtype=np.int16)
    cut_modes = tuple(cut_modes)
    if strata is None:
        d = delta_psi(curr, nxt, cut_modes)
        return {"m": {c: d[c]["delta_psi"] for c in cut_modes},
                "strata": [{"stratum": 0, "n_pairs": int(curr.shape[0]), "share": 1.0,
                            "delta_psi": {c: d[c]["delta_psi"] for c in cut_modes},
                            "psi_full": d[cut_modes[0]]["psi_full"],
                            "mip_cut": {c: d[c]["mip_cut"] for c in cut_modes}}]}
    s = np.asarray(strata)
    total = float(s.size)
    acc = {c: 0.0 for c in cut_modes}
    rows = []
    for code in np.unique(s):
        sel = s == code
        d = delta_psi(curr[sel], nxt[sel], cut_modes)
        share = float(sel.sum()) / total
        for c in cut_modes:
            acc[c] += share * d[c]["delta_psi"]
        rows.append({"stratum": int(code), "n_pairs": int(sel.sum()), "share": share,
                     "delta_psi": {c: d[c]["delta_psi"] for c in cut_modes},
                     "psi_full": d[cut_modes[0]]["psi_full"],
                     "mip_cut": {c: d[c]["mip_cut"] for c in cut_modes}})
    return {"m": acc, "strata": rows}


# --------------------------------------------------------------------------
# exact path (known TPMs)
# --------------------------------------------------------------------------
def exact_delta_psi(tpm, cut_mode: str = PRIMARY_CUT_MODE, weights=None) -> dict:
    """Exact ``Delta_Psi`` of a known state-by-state TPM with the stationary
    state weights (or ``weights`` over all states), through the Psi machinery
    of the sampled estimator; equals v1 ``compute_IIM_from_tpm``. A
    directional cut needs a state-by-node (conditionally independent) TPM."""
    t = np.asarray(tpm, dtype=float)
    n_states = t.shape[0]
    n = int(round(math.log(n_states, BINS)))
    if t.ndim != 2 or t.shape != (n_states, n_states) or BINS ** n != n_states:
        raise ValueError("tpm must be square with 2**n states")
    rows = t.sum(axis=1)
    if np.any(np.abs(rows - 1.0) > 1e-8) or np.any(t < 0):
        raise ValueError("tpm rows must be probability distributions")
    t = t / rows[:, None]
    if cut_mode == "directional":
        dev = iim_xp.state_by_node_deviation(t, n, BINS)
        if dev > 1e-9:
            raise ValueError("a directional cut needs a state-by-node TPM "
                             f"(max deviation {dev:.3g})")
    if weights is None:
        w = mm.iim_stationary_distribution(t)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != n_states or np.any(w < 0) or not w.sum() > 0:
            raise ValueError("weights must be non-negative over all states")
        w = w / w.sum()
    states = _workspace(n).states
    out = _delta_psi_tpm(t, states, w, (cut_mode,), n, zero_rule=False)[cut_mode]
    out["weights"] = [float(v) for v in w]
    return out


def hidden_driver_parts(net) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Per-driver TPMs of a family-B ``hidden_driver`` network (from its joint
    chain over (state, driver); index ``2 * state + (driver == +1)``), the
    stationary driver probabilities and the stationary state weights within
    each driver state (states x 2)."""
    joint = getattr(net, "joint_tpm", None)
    if joint is None:
        raise ValueError("not a hidden-driver network")
    from impact_pipeline.bench.generators import stationary_distribution

    k = joint.shape[0] // 2
    tpms = tuple(joint[np.arange(k) * 2 + d][:, 0::2] + joint[np.arange(k) * 2 + d][:, 1::2]
                 for d in (0, 1))
    pij = stationary_distribution(joint).reshape(k, 2)
    p_d = pij.sum(axis=0)
    w = pij / np.where(p_d > 0, p_d, 1.0)[None, :]
    return tpms, p_d, w


def exact_iim(net, cut_mode: str = PRIMARY_CUT_MODE, *, conditioned: bool = False,
              observational: bool = False) -> dict:
    """Exact target of a family-B network: ``Delta_Psi`` of its (interventional)
    TPM with stationary weights; for the ``hidden_driver`` network with
    ``conditioned`` the background-conditioned value ``sum_d P(d)
    Delta_Psi(T_d, w_d)`` (the target of the stratified estimator with the
    driver recorded; 0 at every driver weight), with ``observational`` the
    exact one-step TPM of the observed process (not state-by-node, so only
    bidirectional cuts are defined; the undeclared driver is not
    identified)."""
    if conditioned and getattr(net, "joint_tpm", None) is not None:
        tpms, p_d, w = hidden_driver_parts(net)
        total = 0.0
        parts = []
        for d in (0, 1):
            r = exact_delta_psi(tpms[d], cut_mode, weights=w[:, d])
            total += float(p_d[d]) * r["delta_psi"]
            parts.append({"driver": (-1, 1)[d], "p": float(p_d[d]),
                          "delta_psi": r["delta_psi"]})
        return {"delta_psi": float(total), "cut_mode": cut_mode, "target": "conditioned",
                "parts": parts}
    if observational and getattr(net, "observational_tpm", None) is not None:
        r = exact_delta_psi(net.observational_tpm, cut_mode, weights=net.stationary)
        r.update(cut_mode=cut_mode, target="observational")
        return r
    r = exact_delta_psi(net.tpm, cut_mode)
    r.update(cut_mode=cut_mode, target="tpm")
    return r


def exact_result(delta_psi_value: float, cut_mode: str = PRIMARY_CUT_MODE,
                 details: Optional[Mapping] = None) -> dict:
    """An exact (known-TPM) result in the format of :func:`compute_iim_v5`:
    the exact independence null is 0 (analytic, no draws), the SE method is
    ``exact`` without a sampling SE (``se = 0``, no ``se_df``), so the status
    rule decides it on ``c`` alone (PRESENT iff ``c > z``, ABSENT iff ``|c| <
    delta``)."""
    v = float(delta_psi_value)
    cut = {"cut_mode": cut_mode, "estimator_id": estimator_id(cut_mode),
           "estimate": v, "value": v, "null_mean": 0.0, "null_sd": 0.0, "n_null": 0,
           "null_family": None, "null_values": [], "p_ind": None, "se": 0.0,
           "se_df": None, "se_method": SE_METHOD_EXACT}
    return _result(dict(details or {}), cut_mode, {cut_mode: cut}, exact=True)


# --------------------------------------------------------------------------
# nulls, rank p-value, resampling
# --------------------------------------------------------------------------
def rank_p_value(m: float, null) -> float:
    """``(1 + #{m*_b >= m}) / (K + 1)`` over the finite draws (NaN without a
    finite estimate or draw)."""
    v = np.asarray(null, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if not math.isfinite(float(m)) or v.size == 0:
        return float("nan")
    return float((1 + int(np.sum(v >= float(m)))) / (v.size + 1))


def null_min_shift(n_time: int, lag: int, frac: float = NULL_MIN_SHIFT_FRAC) -> int:
    """Minimum circular shift ``max(lag + 1, ceil(frac T))`` (v1 rule)."""
    return max(int(lag) + 1, int(math.ceil(float(frac) * float(n_time))))


def circular_shift_surrogates(y, n: int, seed: int, min_shift: int) -> list:
    """``n`` circular-shift surrogates of ``y`` (every row but the first
    shifted independently by at least ``min_shift``): the v1 draws of
    ``compute_IIM`` for the same seed."""
    return mm.iim_null_surrogate_series(np.asarray(y, dtype=float), int(n),
                                        NULL_FAMILY_SHIFT, int(seed), int(min_shift))


def stratified_pair_rotation(curr, nxt, strata, rng, min_frac: float = NULL_MIN_SHIFT_FRAC):
    """One stratified pair-rotation surrogate: within every stratum (pairs
    ``I_s``, ``N_s`` of them, in time order) the pair sequence ``(x_i(t),
    x_i(t + lag))`` of every node but the first is rotated cyclically by
    an independent ``r_i`` in ``[ceil(f N_s), N_s - ceil(f N_s)]``. Strata
    and node 0 are unchanged and every node keeps its own pairs within every
    stratum. Returns ``(curr, nxt, shifts)``; None when a stratum is too
    short to rotate."""
    curr = np.asarray(curr)
    nxt = np.asarray(nxt)
    s = np.asarray(strata)
    c2, n2 = curr.copy(), nxt.copy()
    shifts = {}
    for code in np.unique(s):
        idx = np.flatnonzero(s == code)
        m = idx.size
        lo = max(1, int(math.ceil(float(min_frac) * m)))
        hi = m - lo
        if hi < lo:
            return None
        row = []
        for i in range(1, curr.shape[1]):
            r = int(rng.integers(lo, hi + 1))
            c2[idx, i] = np.roll(curr[idx, i], r)
            n2[idx, i] = np.roll(nxt[idx, i], r)
            row.append(r)
        shifts[int(code)] = row
    return c2, n2, shifts


def block_length(n_time: int, frac: float = BOOTSTRAP_BLOCK_FRAC) -> int:
    """Bootstrap block length ``ceil(frac T)`` (at least 1)."""
    return max(1, int(math.ceil(float(frac) * int(n_time))))


def n_blocks(n_time: int, frac: float = BOOTSTRAP_BLOCK_FRAC) -> int:
    """Blocks per bootstrap resample, ``ceil(T / block_length)``: 10 for 10 %
    blocks of any run longer than 90 samples."""
    return int(math.ceil(int(n_time) / block_length(n_time, frac)))


def bootstrap_indices(n_time: int, rng, frac: float = BOOTSTRAP_BLOCK_FRAC) -> np.ndarray:
    """One circular block-bootstrap resample: ``n_blocks`` blocks of
    ``block_length`` consecutive samples (wrapping at the end), block starts
    uniform over the run, truncated to ``T`` samples."""
    n_time = int(n_time)
    length = block_length(n_time, frac)
    starts = rng.integers(0, n_time, size=n_blocks(n_time, frac))
    idx = np.concatenate([(int(s) + np.arange(length)) % n_time for s in starts])
    return idx[:n_time]


def jackknife_keep(n_time: int, groups: int, group: int) -> np.ndarray:
    """Sample indices with contiguous block ``group`` of ``groups`` deleted
    (the v1 bench rule)."""
    from impact_pipeline.bench.export import block_keep

    return block_keep(int(n_time), int(groups), int(group))


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------
def _not_defined(detail) -> str:
    return R.format_reason(R.NOT_DEFINED, R.clean_detail(detail))


def _result(details, cut_mode, cuts, *, reason=None, exact=False) -> dict:
    if reason is not None:
        R.lookup(reason)
    defined = reason is None
    prim = dict(cuts.get(cut_mode) or {})
    nan = float("nan")
    return {
        "principle": PRINCIPLE,
        "estimator_version": ESTIMATOR_VERSION,
        "estimator_id": estimator_id(cut_mode),
        "cut_mode": cut_mode,
        "defined": defined,
        "reason": reason,
        "estimate": float(prim.get("estimate", nan)) if defined else nan,
        "value": float(prim.get("value", nan)) if defined else nan,
        "null_mean": float(prim.get("null_mean", nan)) if defined else nan,
        "null_sd": float(prim.get("null_sd", nan)) if defined else nan,
        "n_null": int(prim.get("n_null", 0)) if defined else 0,
        "null_family": prim.get("null_family") if defined else None,
        "p_ind": prim.get("p_ind") if defined else None,
        "se": float(prim.get("se", nan)) if defined else nan,
        "se_df": prim.get("se_df") if defined else None,
        "se_method": prim.get("se_method") if defined else None,
        "exact": bool(exact),
        "cuts": {k: dict(v) for k, v in cuts.items()} if defined else {},
        "details": details,
    }


def _statistic(y, strata, lag, cut_modes):
    """m per cut mode of a pre-processed series (macro x time) and its
    per-sample strata (or None)."""
    curr, nxt = transition_pairs(binarise(y), lag)
    s = None if strata is None else np.asarray(strata)[: curr.shape[0]]
    return conditioned_delta_psi(curr, nxt, s, cut_modes)["m"]


def compute_iim_v5(
    ts,
    *,
    lag: int,
    params=None,
    macro_nodes: Optional[Mapping] = None,
    strata=None,
    basis=None,
    null_seed: int = 0,
    se_seed: Optional[int] = None,
    observation_stage: str = "source",
    observation_admitted: bool = False,
) -> dict:
    """
    IIM v5 of one run (module docstring).

    ``ts`` holds the macro signals (macro x time) or, with ``macro_nodes``
    (``{name: node indices}``), the node recording; ``lag`` is the transition
    lag in samples (the system's ``iim_lag_samples``); ``params`` the
    declared settings (:class:`IIMParams` or a mapping); ``strata`` the
    declared binary drivers (one label per sample, or drivers x samples;
    stratified); ``basis`` the declared input basis
    (:class:`~impact_pipeline.v2.declared_inputs.InputBasis` on the same
    samples; residualised; a basis without columns, as declaration ``none``
    gives, conditions on nothing and leaves the run uncropped);
    ``null_seed`` seeds the independence null (the v1 surrogate stream
    without strata) and ``se_seed`` (default ``null_seed``) the bootstrap.
    ``observation_stage`` is the record
    vocabulary's stage; a mixed stage is gated unless
    ``observation_admitted``. Undefined results carry a reason of the
    central vocabulary (``MACRO_RANK_DEFICIENT``, ``INSUFFICIENT_OCCUPANCY``,
    ``NO_NULL_CALIBRATION``, ``OBSERVATION_MIXED_NOT_ADMITTED`` or
    ``NOT_DEFINED:<detail>``).
    """
    p = IIMParams.from_mapping(params)
    empty_basis = basis is not None and getattr(basis, "n_columns", None) == 0
    if strata is not None and basis is not None and not empty_basis:
        raise ValueError("declare drivers either as strata (binary, <= 4 joint "
                         "states) or as an input basis, not both")
    if observation_stage not in OBSERVATION_STAGES:
        raise ValueError(f"observation_stage must be one of {OBSERVATION_STAGES}")
    cut_modes = p.cut_modes
    se_seed = int(null_seed) if se_seed is None else int(se_seed)
    y = macro_signals(ts, macro_nodes)
    n, n_time = y.shape
    details = {
        "estimator_id": estimator_id(p.cut_mode),
        "params": p.to_dict(),
        "lag": int(lag),
        "n_macro": int(n),
        "n_time": int(n_time),
        "null_seed": int(null_seed),
        "se_seed": int(se_seed),
        "observation_stage": observation_stage,
        "observation_admitted": bool(observation_admitted),
        "psi_kernel": PSI_KERNEL,
        "tpm_estimator": TPM_ESTIMATOR,
        "conditioning": ("stratify" if strata is not None
                         else "residualise" if basis is not None and not empty_basis
                         else "none"),
    }

    def undefined(reason):
        return _result(details, p.cut_mode, {}, reason=reason)

    # shape and values
    if not MIN_MACRO_NODES <= n <= MAX_MACRO_NODES:
        return undefined(_not_defined(f"macro_nodes_{n}_outside_"
                                      f"{MIN_MACRO_NODES}_{MAX_MACRO_NODES}"))
    if not np.all(np.isfinite(y)):
        return undefined(_not_defined("non_finite_timeseries"))
    if int(lag) < 1 or n_time - int(lag) < 2:
        return undefined(_not_defined("insufficient_timepoints"))

    # rank condition (constant nodes cannot visit every macro state)
    rank = zero_lag_rank(y, p.rank_tol)
    details["rank_condition"] = {k: rank[k] for k in ("ok", "ratio", "eigenvalues")}
    if rank["constant_nodes"]:
        details["occupancy"] = {"passes": False, "constant_nodes": rank["constant_nodes"],
                                "n_min": p.n_min}
        return undefined(R.INSUFFICIENT_OCCUPANCY)
    if not rank["ok"]:
        return undefined(R.MACRO_RANK_DEFICIENT)

    # pre-processing
    details["preprocess"] = p.preprocess
    if p.preprocess == PREPROCESS_ZCA:
        try:
            y, w_zca = zca(y)
        except np.linalg.LinAlgError:
            return undefined(R.MACRO_RANK_DEFICIENT)
        details["zca_matrix"] = [[float(v) for v in row] for row in w_zca]

    # conditioning inputs, cropped to the analysed samples
    s_all = None
    span = None
    start = 0
    if strata is not None:
        codes, alphabet = joint_strata(strata, p.max_strata)
        if codes.size != n_time:
            raise ValueError("strata need one label per sample")
        s_all = codes
        details["strata_alphabet"] = [list(a) for a in alphabet]
    if basis is not None:
        if int(basis.n_time) != n_time:
            raise ValueError("the input basis has another number of samples")
        if not empty_basis:
            start = int(basis.first_complete_sample)
            span = basis.orthonormal_span(start)
        details["basis"] = {
            "declaration": basis.declaration_id,
            "shared_inputs": basis.shared_inputs,
            "n_columns": int(basis.n_columns),
            "rank": 0 if span is None else int(span.shape[1]),
            "channels": list(basis.channels),
            "lags": list(basis.lags),
            "taus": list(basis.taus),
            "first_complete_sample": start,
            "sha256": basis.sha256(),
        }
        if span is not None:
            details["null_order"] = p.null_order
    y = y[:, start:]
    if s_all is not None:
        s_all = s_all[start:]
    n_eff = y.shape[1]
    details["n_analysed"] = int(n_eff)
    y_pre = y
    y_stat = residualise(y_pre, span) if span is not None else y_pre

    # transition pairs and the occupancy gate
    x = binarise(y_stat)
    curr, nxt = transition_pairs(x, lag)
    s_pairs = None if s_all is None else s_all[: curr.shape[0]]
    occ = occupancy(curr, s_pairs, p.n_min, nxt=nxt)
    details["occupancy"] = occ
    details["n_pairs"] = int(curr.shape[0])
    if not occ["passes"]:
        return undefined(R.INSUFFICIENT_OCCUPANCY)

    # observation gate (after the data-definedness gates)
    if observation_stage in MIXED_OBSERVATION_STAGES and not observation_admitted:
        return undefined(R.OBSERVATION_MIXED_NOT_ADMITTED)

    # estimate
    est = conditioned_delta_psi(curr, nxt, s_pairs, cut_modes)
    m = est["m"]
    details["strata"] = est["strata"]

    # independence null
    null = {c: [] for c in cut_modes}
    min_shift = null_min_shift(n_eff, lag, p.null_min_shift_frac)
    details["null_min_shift"] = int(min_shift)
    if s_pairs is not None:
        null_family = NULL_FAMILY_STRATIFIED
        rng = np.random.default_rng(np.random.SeedSequence(
            [int(null_seed), STRATIFIED_NULL_STREAM]))
        rot_shifts = []
        for _ in range(p.n_null):
            sur = stratified_pair_rotation(curr, nxt, s_pairs, rng, p.null_min_shift_frac)
            if sur is None:
                for c in cut_modes:
                    null[c].append(float("nan"))
                continue
            vals = conditioned_delta_psi(sur[0], sur[1], s_pairs, cut_modes)["m"]
            rot_shifts.append(sur[2])
            for c in cut_modes:
                null[c].append(float(vals[c]))
        details["null_rotations"] = [{str(k): v for k, v in r.items()} for r in rot_shifts]
    else:
        null_family = NULL_FAMILY_SHIFT
        src = y_pre if (span is None or p.null_order == NULL_ORDER_SHIFT_THEN_PROJECT) \
            else y_stat
        try:
            surrogates = circular_shift_surrogates(src, p.n_null, null_seed, min_shift) \
                if p.n_null else []
        except ValueError:
            surrogates = []
        for sur in surrogates:
            if span is not None and p.null_order == NULL_ORDER_SHIFT_THEN_PROJECT:
                sur = residualise(sur, span)
            vals = _statistic(sur, None, lag, cut_modes)
            for c in cut_modes:
                null[c].append(float(vals[c]))
    details["null_family"] = null_family
    finite = {c: np.asarray(null[c], dtype=float) for c in cut_modes}
    finite = {c: v[np.isfinite(v)] for c, v in finite.items()}
    n_finite = min(int(v.size) for v in finite.values()) if finite else 0
    details["null_values"] = {c: [float(v) for v in null[c]] for c in cut_modes}
    if n_finite < MIN_NULL_DRAWS:
        details["n_null_finite"] = n_finite
        return undefined(R.NO_NULL_CALIBRATION)

    # sampling SE
    se = {c: float("nan") for c in cut_modes}
    reps = {c: [] for c in cut_modes}
    if p.se_method == SE_METHOD_BOOTSTRAP:
        rng_b = np.random.default_rng(np.random.SeedSequence([se_seed, BOOTSTRAP_STREAM]))
        for _ in range(p.bootstrap_replicates):
            idx = bootstrap_indices(n_eff, rng_b, p.bootstrap_block_frac)
            vals = _statistic(y_stat[:, idx], None if s_all is None else s_all[idx],
                              lag, cut_modes)
            for c in cut_modes:
                reps[c].append(float(vals[c]))
        for c in cut_modes:
            r = np.asarray(reps[c], dtype=float)
            r = r[np.isfinite(r)]
            se[c] = float(np.std(r, ddof=1)) if r.size >= 2 else float("nan")
        details["bootstrap"] = {"block_length": block_length(n_eff, p.bootstrap_block_frac),
                                "n_blocks": n_blocks(n_eff, p.bootstrap_block_frac),
                                "replicates": {c: reps[c] for c in cut_modes}}
    elif p.se_method == SE_METHOD_JACKKNIFE:
        g = p.jackknife_groups
        for gi in range(g):
            keep = jackknife_keep(n_eff, g, gi)
            vals = _statistic(y_stat[:, keep], None if s_all is None else s_all[keep],
                              lag, cut_modes)
            for c in cut_modes:
                reps[c].append(float(vals[c]))
        for c in cut_modes:
            r = np.asarray(reps[c], dtype=float)
            ok = r[np.isfinite(r)]
            se[c] = (float(np.sqrt((ok.size - 1) / ok.size * np.sum((ok - ok.mean()) ** 2)))
                     if ok.size >= 2 else float("nan"))
        details["jackknife"] = {"groups": g, "replicates": {c: reps[c] for c in cut_modes}}

    cuts = {}
    for c in cut_modes:
        v = finite[c]
        nu = float(np.mean(v))
        cuts[c] = {
            "cut_mode": c,
            "estimator_id": estimator_id(c),
            "estimate": float(m[c]),
            "value": float(m[c] - nu),
            "null_mean": nu,
            "null_sd": float(np.std(v, ddof=1)),
            "n_null": int(v.size),
            "null_family": null_family,
            "null_values": [float(t) for t in null[c]],
            "p_ind": rank_p_value(m[c], v),
            "se": se[c],
            "se_df": p.se_df if p.se_method is not None else None,
            "se_method": p.se_method_name,
        }
    return _result(details, p.cut_mode, cuts)


# Reasons decided before any cut mode is evaluated: a result undefined for one
# of them is the result of every set of cut modes on the same inputs.
_CUT_FREE_REASONS = (R.MACRO_RANK_DEFICIENT, R.INSUFFICIENT_OCCUPANCY,
                     R.OBSERVATION_MIXED_NOT_ADMITTED, R.NOT_DEFINED)


def cut_modes_computed(result: Mapping) -> Tuple[str, ...]:
    """The cut modes a result of :func:`compute_iim_v5` carries (its primary
    and reported ones, from the declared settings in its details)."""
    params = (result.get("details") or {}).get("params")
    if not isinstance(params, Mapping):
        return ()
    return IIMParams.from_mapping(params).cut_modes


def select_cut_modes(result: Mapping, params=None) -> dict:
    """
    The result :func:`compute_iim_v5` returns with the settings ``params``
    on the same inputs, read from ``result``, a run whose settings differ
    from ``params`` only in the cut modes and computed every cut mode
    ``params`` asks for (primary and reported).

    Every cut mode is computed on the same transition pairs, surrogates and
    resamples (their draws do not depend on the cut modes), so the values of
    a cut mode do not depend on the other cut modes of the run: this is the
    bidirectional form of a scoring whose primary run already reports the
    bidirectional cut. Raises ``ValueError`` when the settings differ in
    anything else, when a requested cut mode was not computed, or when the
    run is undefined for a reason that depends on the cut modes (fewer than
    19 finite null draws counts the draws of every computed cut mode).
    """
    want = IIMParams.from_mapping(params)
    have_params = (result.get("details") or {}).get("params")
    if not isinstance(have_params, Mapping):
        raise ValueError("the result carries no declared settings")
    have = IIMParams.from_mapping(have_params)
    others = {k: v for k, v in want.to_dict().items()
              if k not in ("cut_mode", "report_cut_modes")}
    if others != {k: v for k, v in have.to_dict().items()
                  if k not in ("cut_mode", "report_cut_modes")}:
        raise ValueError("the settings differ in more than the cut modes")
    missing = [c for c in want.cut_modes if c not in have.cut_modes]
    if missing:
        raise ValueError(f"cut modes {missing} were not computed")
    reason = result.get("reason")
    if reason is not None and R.parse_reason(reason)[0] not in _CUT_FREE_REASONS:
        raise ValueError(f"an undefined result ({reason}) depends on its cut modes")
    keep = want.cut_modes
    details = dict(result.get("details") or {})
    details["estimator_id"] = estimator_id(want.cut_mode)
    details["params"] = want.to_dict()
    for key in ("null_values",):
        if isinstance(details.get(key), Mapping):
            details[key] = {c: details[key][c] for c in keep}
    for key in ("bootstrap", "jackknife"):
        block = details.get(key)
        if isinstance(block, Mapping) and isinstance(block.get("replicates"), Mapping):
            block = dict(block)
            block["replicates"] = {c: block["replicates"][c] for c in keep}
            details[key] = block
    if isinstance(details.get("strata"), list):
        rows = []
        for row in details["strata"]:
            row = dict(row)
            for k in ("delta_psi", "mip_cut"):
                if isinstance(row.get(k), Mapping):
                    row[k] = {c: row[k][c] for c in keep}
            rows.append(row)
        details["strata"] = rows
    if reason is not None:
        return _result(details, want.cut_mode, {}, reason=reason)
    cuts = {c: dict(result["cuts"][c]) for c in keep}
    return _result(details, want.cut_mode, cuts, exact=bool(result.get("exact")))


def compute_iim_v5_system(system, *, basis=None, strata=None, params=None,
                          null_seed: int = 0, se_seed: Optional[int] = None,
                          observation_stage: Optional[str] = None,
                          observation_admitted: bool = False,
                          macro_nodes: Optional[Mapping] = None,
                          lag: Optional[int] = None) -> dict:
    """:func:`compute_iim_v5` on a bench system: the declared macro grain
    ``meta['iim_macro_nodes']`` and transition lag ``meta['iim_lag_samples']``
    (default 2) unless given; a forward-modelled system is a ``sensor``
    (EEG-like) or ``bold`` observation, every other one ``source``."""
    meta = dict(getattr(system, "meta", None) or {})
    grain = macro_nodes if macro_nodes is not None else meta.get("iim_macro_nodes")
    if lag is None:
        lag = int(meta.get("iim_lag_samples", 2))
    if observation_stage is None:
        sub = str(meta.get("substrate") or "")
        observation_stage = ("sensor" if sub.startswith("eeg")
                             else "bold" if sub.startswith("bold") else "source")
    return compute_iim_v5(system.ts, lag=lag, params=params, macro_nodes=grain,
                          strata=strata, basis=basis, null_seed=null_seed,
                          se_seed=se_seed, observation_stage=observation_stage,
                          observation_admitted=observation_admitted)


# --------------------------------------------------------------------------
# sensor clusters
# --------------------------------------------------------------------------
QUADRANTS = (("L_ant", -1, 1), ("R_ant", 1, 1), ("L_post", -1, -1), ("R_post", 1, -1))


def rank_safe_clusters(positions, n_per_cluster: Optional[int] = None) -> Dict[str, list]:
    """Default sensor clusters of the v2 sensor pipeline: the ``k = min(8,
    floor(n_sensors / 8))`` electrodes nearest the centroid of each quadrant
    (by the signs of the left-right and anterior-posterior coordinates).
    The clusters are disjoint and leave the other electrodes out, so they do
    not partition the montage (no linear constraint under the average
    reference)."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError("positions must be n_sensors x (2 or 3)")
    n_sensors = pos.shape[0]
    k = min(8, n_sensors // 8) if n_per_cluster is None else int(n_per_cluster)
    if k < 1:
        raise ValueError("too few sensors for four clusters")
    out = {}
    for name, sx, sy in QUADRANTS:
        idx = [i for i in range(n_sensors)
               if np.sign(pos[i, 0]) == sx and np.sign(pos[i, 1]) == sy]
        if len(idx) < k:
            raise ValueError(f"quadrant {name} has fewer than {k} electrodes")
        cen = pos[idx].mean(axis=0)
        d = np.linalg.norm(pos[idx] - cen, axis=1)
        out[name] = sorted(int(idx[j]) for j in np.argsort(d, kind="stable")[:k])
    return out


# --------------------------------------------------------------------------
# record fields and evidence items
# --------------------------------------------------------------------------
def _cut_view(result: Mapping, cut_mode: Optional[str]) -> dict:
    cut = result.get("cut_mode") if cut_mode is None else cut_mode
    if not result.get("defined"):
        return {"cut_mode": cut}
    view = (result.get("cuts") or {}).get(cut)
    if view is None:
        raise KeyError(f"cut mode {cut!r} was not computed")
    return dict(view)


def component_fields(result: Mapping, cut_mode: Optional[str] = None) -> dict:
    """The estimator fields of a ``mpc-bench-result/3`` component for one cut
    mode (default: the primary): ``estimate``, null moments and family,
    ``se``, ``se_df``, ``se_method`` and ``details`` (``p_ind``, the cut mode,
    the estimator's definedness and reason, occupancy summary, ``exact``).
    The runner adds status, reason, ``c`` and the protocol fields."""
    defined = bool(result.get("defined"))
    v = _cut_view(result, cut_mode)
    occ = (result.get("details") or {}).get("occupancy") or {}
    details = {
        "defined": defined,
        "estimator_reason": result.get("reason"),
        "estimator_id": estimator_id(v["cut_mode"]),
        "cut_mode": v["cut_mode"],
        "p_ind": v.get("p_ind") if defined else None,
        "exact": bool(result.get("exact")),
        "occupancy_passes": occ.get("passes"),
        "occupancy_rarest_row": occ.get("rarest_row"),
        "occupancy_n_unvisited": occ.get("n_unvisited"),
    }
    se = v.get("se") if defined else None
    if se is not None and not math.isfinite(float(se)):
        se = None
    return {
        "estimate": v.get("estimate") if defined else None,
        "null_mean": v.get("null_mean") if defined else None,
        "null_sd": v.get("null_sd") if defined else None,
        "n_null": int(v.get("n_null") or 0) if defined else 0,
        "null_family": v.get("null_family") if defined else None,
        "se": se,
        "se_df": v.get("se_df") if defined else None,
        "se_method": v.get("se_method") if defined else None,
        "details": details,
    }


def evidence(result: Mapping, cut_mode: Optional[str] = None, *, reference=None,
             reference_se=None, reference_scale: str = "excess", substrate=None,
             grain=None, channel: str = "default", observation_stage=None,
             view=None, bearer_id=None, protocol_id=None, regime=None):
    """The evidence item (:class:`~impact_pipeline.evidence_v2.
    ComponentEvidenceV2`) of one cut mode of a result, for the status rule:
    estimate, null moments, family and size, SE with its method and
    ``se_df``, ``p_ind`` for the rank gate, ``exact`` for known-TPM values
    and the estimator id ``compute_IIM:<cut mode>@iim-v5-2026.10``. The
    anchor usually comes from the protocol (an external reference replaces
    the item's)."""
    from impact_pipeline.evidence_v2 import ComponentEvidenceV2

    defined = bool(result.get("defined"))
    v = _cut_view(result, cut_mode)
    nan = float("nan")
    se = v.get("se", nan) if defined else nan
    se_df = v.get("se_df") if defined else None
    p_ind = v.get("p_ind") if defined else None
    if p_ind is not None and not math.isfinite(float(p_ind)):
        p_ind = None
    return ComponentEvidenceV2(
        principle=PRINCIPLE,
        estimate=float(v.get("estimate", nan)) if defined else nan,
        null_mean=float(v.get("null_mean", nan)) if defined else nan,
        null_sd=float(v.get("null_sd", nan)) if defined else nan,
        se=float(se) if se is not None else nan,
        channel=channel,
        defined=defined,
        reason=result.get("reason"),
        reference=reference,
        reference_se=reference_se,
        reference_scale=reference_scale,
        bearer_id=bearer_id,
        protocol_id=protocol_id,
        substrate=substrate,
        grain=grain,
        estimator=estimator_id(v["cut_mode"]),
        null_family=v.get("null_family") if defined else None,
        n_null=int(v.get("n_null") or 0) if defined else 0,
        exact=bool(result.get("exact")),
        regime=regime,
        se_df=None if se_df is None else float(se_df),
        se_method=v.get("se_method") if defined else None,
        p_ind=p_ind,
        observation_stage=observation_stage,
        view=view,
    )


__all__ = [
    "BINS",
    "BOOTSTRAP_BLOCK_FRAC",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SE_DF",
    "BOOTSTRAP_SE_DF_ALLOWED",
    "CUT_MODES",
    "ESTIMATOR_ID",
    "ESTIMATOR_VERSION",
    "IIMParams",
    "JACKKNIFE_GROUPS",
    "JACKKNIFE_SE_DF",
    "MAX_STRATA",
    "MIN_NULL_DRAWS",
    "MIXED_OBSERVATION_STAGES",
    "NULL_FAMILIES",
    "NULL_FAMILY_SHIFT",
    "NULL_FAMILY_STRATIFIED",
    "NULL_ORDERS",
    "NULL_ORDER_PROJECT_THEN_SHIFT",
    "NULL_ORDER_SHIFT_THEN_PROJECT",
    "N_MIN_ALLOWED",
    "N_MIN_DEFAULT",
    "N_NULL_BENCH",
    "N_NULL_FAMILY_B",
    "PRIMARY_CUT_MODE",
    "PRINCIPLE",
    "PSI_KERNEL",
    "RANK_GATE_ALPHA",
    "RANK_TOL",
    "SECONDARY_CUT_MODES",
    "SE_METHODS",
    "SE_METHOD_BOOTSTRAP",
    "SE_METHOD_DEFAULT",
    "SE_METHOD_EXACT",
    "SE_METHOD_JACKKNIFE",
    "binarise",
    "block_length",
    "bootstrap_indices",
    "circular_shift_surrogates",
    "component_fields",
    "compute_iim_v5",
    "compute_iim_v5_system",
    "conditioned_delta_psi",
    "cut_modes_computed",
    "delta_psi",
    "estimator_id",
    "evidence",
    "exact_delta_psi",
    "exact_iim",
    "exact_result",
    "hidden_driver_parts",
    "jackknife_keep",
    "joint_strata",
    "macro_signals",
    "n_blocks",
    "node_shrinkage_tpm",
    "null_min_shift",
    "occupancy",
    "rank_p_value",
    "rank_safe_clusters",
    "residualise",
    "select_cut_modes",
    "shrinkage_intensity",
    "stratified_pair_rotation",
    "system_cuts",
    "transition_pairs",
    "zca",
    "zero_lag_rank",
]
