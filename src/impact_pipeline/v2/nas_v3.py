"""
NAS v3 (``nas-v3-2026.10``): conditional receive-and-return between a
declared hub block and declared periphery blocks, beyond a declared
exogenous-input basis.

**Construct.** NAS v3 measures bidirectional directed (lagged,
linear-Gaussian) dependence between a declared hub block and each declared
periphery block, with blocks represented by their mean signals, beyond a
declared exogenous-input basis, at lags that resolve a declared coupling time
scale, in an unmixed or admitted observation domain. PRESENT requires both
directions credibly above ``z`` on their own anchors; ABSENT requires at least
one direction credibly negligible (both decided by the status rule
``tost-v2``, never here). It is read as *workspace receive-and-return* only
if the shared inputs are declared complete (ID-1), ``dt <= tau_c / 2`` (ID-2)
and the observation is unmixed or admitted (ID-3); otherwise it is
*hub-periphery dependence*. Hidden shared inputs inflate NAS and do not in
general deflate it, so under a partial or empty declaration PRESENT
certifies dependence only while ABSENT keeps its meaning. NAS does not tell a
loop from a feed-forward relay, does not see zero-mean pattern broadcast
(block means cancel it), does not certify which block is the workspace, and
does not measure nonlinear or interventional effects. The statistic is
quadratic in a weak coupling amplitude, so ``delta = 0.10`` on ``c``
corresponds to about 0.32 of the reference coupling amplitude.

**Algorithm** (bench defaults in :class:`NASParams`; the keys of the
protocol's ``estimators.NAS`` block are the field names).

1. *Gates, in this order* (each gives UNDEFINED, never ABSENT):
   shape and declarations (``NOT_DEFINED:<reason>`` as v1:
   ``non_finite_timeseries``, ``hub_or_periphery_empty``,
   ``invalid_workspace``, ``run_too_short_for_circular_shift``);
   resolvability (``SAMPLING_UNRESOLVED`` iff ``dt > tau_c / 2``);
   observation (``OBSERVATION_MIXED_NOT_ADMITTED`` for ``sensor_mixing`` or
   ``source_estimate`` without an admitting registry entry); block
   representation (``NOT_DEFINED:no_variance`` for a block without
   variance), parameter budget and rank (``INSUFFICIENT_TIMEPOINTS``); null
   (``NOT_DEFINED:transfer_null_degenerate``; a non-finite observed
   statistic gives ``NOT_DEFINED:non_finite_statistic``).
2. *Coupling time scale and lags.* ``tau_c`` (``coupling_timescale_sec``) is
   the declared time constant of the coupled units, read from the generator
   constants on the bench (0.1 s on families A and C and on the Hopf model;
   :func:`impact_pipeline.v2.declared_inputs.coupling_timescale`) and
   declared by the protocol on human data. The resolvability gate compares
   the sampling interval with this ``tau_c`` only. The lag set ``L =
   sorted unique round(geomspace(1, l_max, min(4, l_max)))`` with ``l_max =
   ceil(tau_l / dt)`` and the filtered input copies (``tau_l``, ``3 tau_l``,
   ``10 tau_l``) are built from the *lag time scale* ``tau_l``
   (``lag_timescale_sec``), which equals ``tau_c`` unless the bench's
   sensitivity analysis sets it to 0.05 s or 0.2 s
   (:data:`LAG_TIMESCALE_SENSITIVITY_SEC`). The sensitivity analysis changes
   the lags and the basis, never the gate: families A and C (``dt = 0.05 s``,
   ``tau_c = 0.1 s``) are resolved at every sensitivity time scale (lags
   ``{1}`` at 0.05 s, ``{1, 2}`` at 0.1 s, ``{1, 2, 3, 4}`` at 0.2 s), while a
   BOLD-like view at ``TR = 2 s`` is unresolved at all of them. Lag sets:
   ``{1, 2}`` at 20 Hz, ``{1, 3, 9, 25}`` at 250 Hz, ``{1, 4, 14, 50}`` at
   500 Hz.
3. *Blocks.* Declared hub ``H`` and periphery blocks ``P_1 .. P_m``; every
   node is z-scored over the run. Primary representation
   (``block_representation = 'block_mean'``): the mean of the block's
   z-scored nodes (one dimension per block, a population signal). Secondary
   (``'all_units'``): every non-constant node of a block if it has at most
   ``d`` nodes, else its ``d`` leading principal components (v1
   ``_block_components``), with ``d`` lowered through 8, 4, 2, 1 until both
   budget conditions hold.
4. *Declared inputs* (:mod:`impact_pipeline.v2.declared_inputs`): the common
   basis ``U(t) = [u(t - l), u~_tau(t - l)]`` for ``l`` in ``{0} U L``.
   **The basis is rank-deficient by construction**: a filtered copy at lag
   ``l`` is a combination of the copy at lag ``l + 1`` and the raw input at
   lag ``l``, and lagged or filtered copies of the slow phase span two
   dimensions plus decaying transients (family A under the complete
   declaration R: 156 columns, rank 71). NAS v3 therefore enters the basis
   through its orthonormal span on the regression rows
   (:meth:`InputBasis.orthonormal_span`, centred, rank cut 1e-10 of the
   largest singular value) and solves every regression with a
   rank-revealing pseudo-inverse of the Jacobi-scaled Gram block
   (eigenvalues below ``SOLVE_RTOL`` of the largest are dropped), so that
   collinear regressors (raw basis columns, rank-deficient source
   estimates, a regressor that vanishes on a jackknife replicate) never
   enter as noise. **The conditioning budget counts the effective rank of
   the basis, not its raw columns**: ``n_par = 1 + rank(U) + |L| sum_B
   d_B``.
5. *Budget and rank* (``INSUFFICIENT_TIMEPOINTS`` otherwise): ``T - l_max >=
   10 n_par`` and ``sum_B d_B <= 0.8 x`` the numerical rank of the z-scored
   observation (relative singular-value cut 1e-8; source estimates from
   ``n`` electrodes have rank at most ``n - 1``). On a direct, full-rank
   observation the rank condition binds as soon as every node is used
   (families A and C: 30 units > 0.8 x 30), so the secondary statistic
   there runs on 4 leading components per block; ``rank_fraction`` is the
   one declared parameter of this rule.
6. *Statistic* (rows ``t = l_max .. T - 1``; one Gram matrix of ``[1, U,
   pasts, currents]`` per surrogate, every regression a Schur complement):
   with ``Z_j = {Y_k^L, k != j} U {U}``,
   ``receive_j = 1/2 ln det Sigma(Y_H | Y_H^L, Z_j) - 1/2 ln det Sigma(Y_H |
   Y_H^L, Z_j, Y_j^L)`` and ``return_j = 1/2 ln det Sigma(Y_j | Y_j^L, Z_j)
   - 1/2 ln det Sigma(Y_j | Y_j^L, Z_j, Y_H^L)`` (conditional Geweke
   causalities in nats); ``R = mean_j receive_j`` (``te_in``), ``B = mean_j
   return_j`` (``te_out``). On the v1 geometry (5 leading components of the
   hub and of the pooled periphery, no inputs) the same code reproduces the
   frozen v1 ``compute_NAS(mode='capacity')`` statistic and null to 1e-12
   (:func:`pooled_statistic`).
7. *Null* (family ``block_circular_shift``, kept from v1): the hub block is
   circularly shifted relative to the periphery and ``U`` (which stay
   aligned with each other) by ``s`` drawn uniformly from ``{ceil(0.1 T),
   ..., T - ceil(0.1 T)}``, ``K = 19`` draws from ``RandomState(null_seed)``
   with ``null_seed = seed * 1000 + 17``; then every regression, including
   the projection on ``U``, is fitted on the shifted data (**shift, then
   project**; :data:`NULL_ORDER`). Per direction: null mean, SD, excess
   ``m - nu``, ``z`` and the rank p-value ``(1 + #{null >= m}) / (K + 1)``;
   per block the same for ``receive_j`` and ``return_j``; coverage = share of
   blocks with both per-block ``z > 1.645``. A null is degenerate when its
   SD is not finite or is zero up to rounding (not above
   :data:`NULL_SD_RTOL` x the largest absolute draw; identical draws give an
   SD of about 1e-20 rather than 0): the component is then UNDEFINED and a
   degenerate per-block null has no ``z``.
8. *Directions.* Each direction is a member of the directional principle
   (``receive`` = ``c_R``, ``return`` = ``c_B``) with its own estimate, null
   and SE (:func:`evidence_items`); the protocol anchors each direction
   separately (reference keys ``NAS:receive`` and ``NAS:return``,
   :func:`anchor_reference`). "Significant" means both per-direction
   ``z > 1.645``.
9. *SE* (``se_method``, one of :data:`SE_METHODS`; chosen at development
   calibration from the twins, default :data:`SE_METHOD_DEFAULT`, the
   contiguous G = 10 fallback): delete-a-group jackknife with ``G`` groups,
   contiguous (``G`` equal stretches of the run) or interleaved (blocks of
   ``b = round(max(5 s, 50 tau_c) / dt)`` samples, group = block index mod
   ``G``); replicate ``g`` drops every regression row whose window ``[t -
   l_max, t]`` touches a sample of group ``g`` (no concatenation across
   gaps) and recomputes both directions without a null; ``se = sqrt((G - 1)
   / G sum (theta_g - mean)^2)``, ``se_df = G - 1``.
10. *Descriptors* (reported, never gating; each in its own ``try`` block and
    computed after the gated value, so a descriptor can never abort it):
    the frozen v1 statistic (``pooled_v1``), the rank-1 pooled statistic
    (``pooled_rank1``: one leading component per pooled block, the v1
    diagnosis' rank variant), ``conditioning_delta`` (per-direction excess
    with ``U`` minus without, same null shifts), ``zero_lag_coupling``
    (first canonical correlation of the full-model residuals of the hub and
    the stacked periphery), ``bic_order`` (BIC-optimal number of leading
    lags of the full model), ``metastability`` (SD and mean of the v1
    Kuramoto order parameter over the declared bands, observed values only;
    a band at or above Nyquist gives ``reason = band_above_nyquist`` and
    nothing else) and, on request, the legacy L/B/H ``profile``. The
    statistic descriptors are computed only for a defined gated value; the
    signal descriptors also for an undefined one (a BOLD-like view reports
    its band reason next to ``SAMPLING_UNRESOLVED``).
11. *Identifiability record* ``{shared_inputs, observation,
    hub_privileged}`` in every result (``hub_privileged = 'not_tested'``:
    the hub-identity table is a separate Tier-B item; it scores each
    periphery block as the hub with :func:`compute_nas_v3` on the same
    inputs and seed, hence the same basis, lags, null shifts and jackknife
    groups, whose per-direction replicates every result returns for paired
    SEs; :class:`GramDesign` and :func:`nas_statistics` are hub-agnostic).

The v1 estimator (``compute_NAS``, ``nas-v2-2026.09``) is untouched: under
the v1 protocol a v1 BOLD task still ends in the v1 band error.

**Output.** :func:`compute_nas_v3` returns a dict with ``defined``,
``reason``, ``directions`` (per direction: estimate, null moments and draws,
excess, z, p, SE and its replicates), the limiting direction's estimate,
null and SE at the top level (the direction with the smaller ``z``, as
reported by v1; the status never reads it), ``significant``,
``identifiability`` and ``details`` (lags, basis, blocks, budget, per-block
values, coverage, null shifts, descriptors). :func:`component_fields` maps
it onto the estimator fields of a ``mpc-bench-result/3`` component and
:func:`evidence_items` onto the two direction items of the status rule.
:func:`nas_v3_system` scores a bench system under one declaration.
"""

from __future__ import annotations

import dataclasses
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline import mpc_metrics as mm
from impact_pipeline.v2 import declared_inputs as DI
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import records as _records

PRINCIPLE = "NAS"
ESTIMATOR_VERSION = "nas-v3-2026.10"
MODE = "conditional_capacity"
ESTIMATOR_ID = f"compute_NAS:{MODE}@{ESTIMATOR_VERSION}"
NULL_FAMILY = "block_circular_shift"
NULL_ORDER = DI.NULL_ORDER  # 'shift_then_project'

# Directions (members of the directional principle; evidence_v2.NAS_DIRECTIONS)
# and the v1 names of their statistics.
RECEIVE, RETURN = "receive", "return"
DIRECTIONS = (RECEIVE, RETURN)
DIRECTION_STATISTICS = {RECEIVE: "te_in", RETURN: "te_out"}

# Null (kept from v1).
N_NULL = 19
NULL_MIN_SHIFT_FRACTION = 0.1
NULL_SEED_FACTOR = 1000
NULL_SEED_OFFSET = 17

# Block representations (design N8).
REPRESENTATION_BLOCK_MEAN = "block_mean"  # primary
REPRESENTATION_ALL_UNITS = "all_units"  # secondary (preregistered)
REPRESENTATIONS = (REPRESENTATION_BLOCK_MEAN, REPRESENTATION_ALL_UNITS)
SECONDARY_DIMENSIONS = (8, 4, 2, 1)
RANK_FRACTION = 0.8
BUDGET_FACTOR = 10.0
OBSERVATION_RANK_RTOL = 1e-8
MIN_BLOCK_SIZE = 2  # smaller declared blocks are merged (whole brain)

# Coupling time scale: the sensitivity time scales of the lag set and basis.
LAG_TIMESCALE_SENSITIVITY_SEC = DI.TAU_C_SENSITIVITY_SEC  # (0.05, 0.2)

# Rank-revealing solves: eigenvalues of the Jacobi-scaled Gram block below
# SOLVE_RTOL x the largest are dropped (exact collinearity sits at ~1e-14).
SOLVE_RTOL = 1e-12
_SD_FLOOR = 1e-12
# A null SD at the rounding level of the null draws is zero: identical draws
# give an SD of about 1e-20 through the rounding of their mean (and a z of
# about 1e17), so the degenerate-null gate compares the SD with the draws'
# magnitude rather than with exact zero.
NULL_SD_RTOL = 1e-10

# SE methods (section 2.0.5; the evidence layer's SE-method contract).
SE_CONTIGUOUS, SE_INTERLEAVED = "contiguous", "interleaved"
SE_SCHEMES = (SE_CONTIGUOUS, SE_INTERLEAVED)
SE_GROUPS = (10, 20)
SE_METHODS = tuple(f"jackknife_{s}_{g}" for s in SE_SCHEMES for g in SE_GROUPS)
# Development calibration chooses the method on the twins (calibrated method
# with the largest df; ties and failure -> contiguous G = 10). This default is
# the single place the protocol overrides through ``se_method``.
SE_METHOD_DEFAULT = "jackknife_contiguous_10"
INTERLEAVE_BLOCK_MIN_SEC = 5.0
INTERLEAVE_BLOCK_TAU_MULTIPLE = 50.0

# "Significant": both per-direction z above the one-sided 5 % normal quantile.
Z_SIGNIFICANT = 1.645

# Observations (identifiability vocabulary of the records).
OBSERVATION_DIRECT = "direct"
OBSERVATIONS = _records.OBSERVATIONS
OBSERVATIONS_MIXED = ("sensor_mixing", "source_estimate")
OBSERVATION_STAGE_OF = {
    "direct": "source",
    "sensor_mixing": "sensor",
    "source_estimate": "source_estimate",
    "hemodynamic": "bold",
}
_OBSERVATION_OF_SUBSTRATE = {
    "eeg_like_forward": "sensor_mixing",
    "bold_like_forward": "hemodynamic",
}

# Descriptors (design 3.11).
DESCRIPTOR_POOLED_V1 = "pooled_v1"
DESCRIPTOR_POOLED_RANK1 = "pooled_rank1"
DESCRIPTOR_CONDITIONING_DELTA = "conditioning_delta"
DESCRIPTOR_ZERO_LAG = "zero_lag_coupling"
DESCRIPTOR_BIC = "bic_order"
DESCRIPTOR_METASTABILITY = "metastability"
DESCRIPTOR_PROFILE = "profile"
STATISTIC_DESCRIPTORS = (DESCRIPTOR_POOLED_V1, DESCRIPTOR_POOLED_RANK1,
                         DESCRIPTOR_CONDITIONING_DELTA, DESCRIPTOR_ZERO_LAG,
                         DESCRIPTOR_BIC)
SIGNAL_DESCRIPTORS = (DESCRIPTOR_METASTABILITY, DESCRIPTOR_PROFILE)
ALL_DESCRIPTORS = STATISTIC_DESCRIPTORS + SIGNAL_DESCRIPTORS
DEFAULT_DESCRIPTORS = STATISTIC_DESCRIPTORS + (DESCRIPTOR_METASTABILITY,)
V1_COMPONENTS = 5
METASTABILITY_BANDS = ((0.5, 4.0),)  # the v1 bench band
# The v1 bench's legacy profile parameters (bench/export.py,
# BENCH_ESTIMATOR_PARAMS['NAS']); the profile descriptor is opt-in.
PROFILE_PARAMETERS = {"tau": 0.2, "bands": [(0.5, 4.0)], "band_weights": [1.0],
                      "window_len": 100, "step_len": 50, "random_state": 0}
BAND_ABOVE_NYQUIST = "band_above_nyquist"

# Estimator-level reasons (NOT_DEFINED:<detail>, the v1 names).
NON_FINITE_TIMESERIES = "non_finite_timeseries"
HUB_OR_PERIPHERY_EMPTY = "hub_or_periphery_empty"
INVALID_WORKSPACE = "invalid_workspace"
NO_VARIANCE = "no_variance"
RUN_TOO_SHORT = "run_too_short_for_circular_shift"
TRANSFER_NULL_DEGENERATE = "transfer_null_degenerate"
NON_FINITE_STATISTIC = "non_finite_statistic"


class NASError(ValueError):
    """A malformed NAS declaration or call."""


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------
def _opt_positive(value, what) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise NASError(f"{what} must be a number")
    v = float(value)
    if not (math.isfinite(v) and v > 0):
        raise NASError(f"{what} must be finite and > 0, got {value!r}")
    return v


@dataclass(frozen=True)
class NASParams:
    """
    Declared estimator settings (bench defaults; hash-covered in the
    protocol's ``estimators.NAS`` block through :meth:`to_dict`).
    ``coupling_timescale_sec`` is ``tau_c`` of the resolvability gate (None:
    read from the generator constants by :func:`nas_v3_system`; required by
    :func:`compute_nas_v3`); ``lag_timescale_sec`` builds the lag set and the
    filtered input copies (None: ``tau_c``; the sensitivity analysis uses
    0.05 s and 0.2 s); ``se_method`` is the calibrated SE method;
    ``interleave_block_sec`` overrides ``max(5 s, 50 tau_c)``; ``descriptors``
    names the descriptors to compute.
    """

    mode: str = MODE
    block_representation: str = REPRESENTATION_BLOCK_MEAN
    coupling_timescale_sec: Optional[float] = None
    lag_timescale_sec: Optional[float] = None
    n_null: int = N_NULL
    null_min_shift_fraction: float = NULL_MIN_SHIFT_FRACTION
    se_method: str = SE_METHOD_DEFAULT
    interleave_block_sec: Optional[float] = None
    secondary_dimensions: Tuple[int, ...] = SECONDARY_DIMENSIONS
    rank_fraction: float = RANK_FRACTION
    budget_factor: float = BUDGET_FACTOR
    descriptors: Tuple[str, ...] = DEFAULT_DESCRIPTORS
    v1_components: int = V1_COMPONENTS
    metastability_bands: Tuple[Tuple[float, float], ...] = METASTABILITY_BANDS

    def __post_init__(self):
        set_ = object.__setattr__
        if self.mode != MODE:
            raise NASError(f"mode must be {MODE!r}, got {self.mode!r}")
        if self.block_representation not in REPRESENTATIONS:
            raise NASError(f"block_representation must be one of {REPRESENTATIONS}")
        set_(self, "coupling_timescale_sec",
             _opt_positive(self.coupling_timescale_sec, "coupling_timescale_sec"))
        set_(self, "lag_timescale_sec",
             _opt_positive(self.lag_timescale_sec, "lag_timescale_sec"))
        set_(self, "interleave_block_sec",
             _opt_positive(self.interleave_block_sec, "interleave_block_sec"))
        for name in ("n_null", "v1_components"):
            v = getattr(self, name)
            if isinstance(v, bool) or int(v) != v:
                raise NASError(f"{name} must be an integer")
            set_(self, name, int(v))
        if self.n_null < 2:
            raise NASError("n_null must be >= 2 (the null needs an SD)")
        if self.v1_components < 1:
            raise NASError("v1_components must be >= 1")
        f = float(self.null_min_shift_fraction)
        if not 0.0 < f < 0.5:
            raise NASError("null_min_shift_fraction must lie in (0, 0.5)")
        set_(self, "null_min_shift_fraction", f)
        if self.se_method not in SE_METHODS:
            raise NASError(f"se_method must be one of {SE_METHODS}")
        dims = tuple(int(d) for d in self.secondary_dimensions)
        if (not dims or any(d < 1 for d in dims)
                or list(dims) != sorted(set(dims), reverse=True)):
            raise NASError("secondary_dimensions must be distinct, decreasing and >= 1")
        set_(self, "secondary_dimensions", dims)
        for name in ("rank_fraction", "budget_factor"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0):
                raise NASError(f"{name} must be > 0")
            set_(self, name, v)
        if self.rank_fraction > 1.0:
            raise NASError("rank_fraction must be <= 1")
        desc = tuple(str(d) for d in self.descriptors)
        unknown = sorted(set(desc) - set(ALL_DESCRIPTORS))
        if unknown or len(set(desc)) != len(desc):
            raise NASError(f"descriptors must be distinct names of {ALL_DESCRIPTORS}")
        set_(self, "descriptors", desc)
        bands = []
        for b in self.metastability_bands:
            if not isinstance(b, (tuple, list)) or len(b) != 2:
                raise NASError("metastability_bands must be (low, high) pairs")
            bands.append((float(b[0]), float(b[1])))
        set_(self, "metastability_bands", tuple(bands))

    @property
    def se_scheme(self) -> str:
        return self.se_method.split("_")[1]

    @property
    def se_groups(self) -> int:
        return int(self.se_method.rsplit("_", 1)[1])

    def coupling_timescale(self) -> float:
        if self.coupling_timescale_sec is None:
            raise NASError("coupling_timescale_sec (tau_c) is not declared")
        return float(self.coupling_timescale_sec)

    def lag_timescale(self) -> float:
        return float(self.lag_timescale_sec if self.lag_timescale_sec is not None
                     else self.coupling_timescale())

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["secondary_dimensions"] = list(self.secondary_dimensions)
        out["descriptors"] = list(self.descriptors)
        out["metastability_bands"] = [list(b) for b in self.metastability_bands]
        return out

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping]) -> "NASParams":
        """Parameters from a protocol's ``estimators.NAS`` block (the template
        declares ``mode``, ``block_representation`` and
        ``coupling_timescale_sec``); unknown keys are refused."""
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - names)
        if unknown:
            raise NASError(f"unknown NAS parameters {unknown}")
        return cls(**dict(payload))

    def replace(self, **kw) -> "NASParams":
        return dataclasses.replace(self, **kw)


def null_seed_for(seed) -> int:
    """The null seed of a system seed: ``seed * 1000 + 17``."""
    return int(seed) * NULL_SEED_FACTOR + NULL_SEED_OFFSET


def null_min_shift(n_time: int, fraction: float = NULL_MIN_SHIFT_FRACTION) -> int:
    """Minimum circular shift ``ceil(fraction T)`` (at least 1), computed as
    v1 does for ``fraction = 0.1``."""
    return max(1, int(math.ceil(float(fraction) * int(n_time))))


def null_shifts(n_time: int, n_null: int = N_NULL, null_seed: int = 0,
                min_shift: Optional[int] = None) -> List[int]:
    """The hub-block shifts of the null: ``n_null`` draws of
    ``RandomState(null_seed).randint(lo, T - lo + 1)`` (the v1 sequence)."""
    lo = null_min_shift(n_time) if min_shift is None else max(1, int(min_shift))
    rng = np.random.RandomState(int(null_seed))
    return [int(rng.randint(lo, int(n_time) - lo + 1)) for _ in range(int(n_null))]


def lag_plan(coupling_timescale_sec: float, dt: float,
             lag_timescale_sec: Optional[float] = None) -> dict:
    """
    The resolvability gate and the lags of one (``tau_c``, ``tau_l``, ``dt``):
    ``resolved`` iff ``dt <= tau_c / 2`` (``tau_c`` the coupling time scale);
    ``lags`` and ``basis_taus`` from the lag time scale ``tau_l`` (default
    ``tau_c``).
    """
    tau_c = float(coupling_timescale_sec)
    tau_l = tau_c if lag_timescale_sec is None else float(lag_timescale_sec)
    dt = float(dt)
    return {
        "coupling_timescale_sec": tau_c,
        "lag_timescale_sec": tau_l,
        "dt": dt,
        "resolved": bool(DI.sampling_resolved(tau_c, dt)),
        "l_max": int(DI.l_max(tau_l, dt)),
        "lags": [int(v) for v in DI.nas_lag_set(tau_l, dt)],
        "basis_lags": [0] + [int(v) for v in DI.nas_lag_set(tau_l, dt)],
        "basis_taus": [float(v) for v in DI.basis_taus(tau_l)],
    }


def interleave_block_samples(dt: float, coupling_timescale_sec: float,
                             block_sec: Optional[float] = None) -> int:
    """Interleaved jackknife block ``b = round(max(5 s, 50 tau_c) / dt)``
    samples (``block_sec`` overrides the length in seconds)."""
    sec = (float(block_sec) if block_sec is not None else
           max(INTERLEAVE_BLOCK_MIN_SEC,
               INTERLEAVE_BLOCK_TAU_MULTIPLE * float(coupling_timescale_sec)))
    return max(1, int(round(sec / float(dt))))


def observation_of(meta: Optional[Mapping]) -> str:
    """The declared observation of a system: ``meta['observation']`` when
    declared, ``sensor_mixing`` for EEG-like and ``hemodynamic`` for
    BOLD-like forward views, ``direct`` otherwise."""
    meta = meta or {}
    obs = meta.get("observation")
    if obs is not None:
        if obs not in OBSERVATIONS:
            raise NASError(f"observation must be one of {OBSERVATIONS}")
        return str(obs)
    return _OBSERVATION_OF_SUBSTRATE.get(str(meta.get("substrate")), OBSERVATION_DIRECT)


def observation_stage(observation: str) -> str:
    """The record's ``observation_stage`` of an observation."""
    try:
        return OBSERVATION_STAGE_OF[observation]
    except KeyError:
        raise NASError(f"observation must be one of {OBSERVATIONS}") from None


def identifiability(shared_inputs: str, observation: str = OBSERVATION_DIRECT,
                    hub_privileged="not_tested") -> dict:
    """The identifiability record (registry v3 vocabulary)."""
    return _records.validate_identifiability({
        "shared_inputs": shared_inputs, "observation": observation,
        "hub_privileged": hub_privileged})


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------
def declared_blocks(meta: Mapping, min_block_size: int = MIN_BLOCK_SIZE
                    ) -> Tuple[str, List[int], "OrderedDict[str, List[int]]"]:
    """
    ``(hub_name, hub_nodes, periphery_blocks)`` of a bench system from its
    declared meta: the hub is ``workspace_nodes``; the periphery blocks are
    the declared ``modules`` (in ``module_order``) without the hub nodes,
    empty blocks dropped, and blocks with fewer than ``min_block_size``
    nodes merged into the largest block of the same hemisphere (name prefix
    before ``_``; the largest block overall when the hemisphere has no
    other; ties by declared order). ``hub_name`` is the module whose nodes
    are the hub (``W`` in families A and C), else ``hub``.
    """
    hub = [int(i) for i in (meta.get("workspace_nodes") or [])]
    modules = meta.get("modules") or {}
    if not hub or not modules:
        raise NASError("the meta declares no workspace_nodes or no modules")
    order = list(meta.get("module_order") or sorted(modules))
    order += [k for k in modules if k not in order]
    hub_set = set(hub)
    hub_name = "hub"
    for k in order:
        if sorted(int(i) for i in modules[k]) == sorted(hub_set):
            hub_name = str(k)
            break
    blocks = OrderedDict()
    for k in order:
        rest = [int(i) for i in modules[k] if int(i) not in hub_set]
        if rest:
            blocks[str(k)] = rest
    small = [k for k, v in blocks.items() if len(v) < int(min_block_size)]
    big = [k for k in blocks if k not in small]
    for k in small:
        if not big:
            break
        hemi = k.split("_")[0]
        cands = [c for c in big if c.split("_")[0] == hemi and "_" in c] or big
        target = max(cands, key=lambda c: (len(blocks[c]), -order.index(c)))
        blocks[target] = sorted(blocks[target] + blocks.pop(k))
    return hub_name, hub, blocks


def zscore_rows(x) -> Tuple[np.ndarray, np.ndarray]:
    """Rows z-scored over time (constant rows become 0) and the mask of
    non-constant rows."""
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    ok = sd > _SD_FLOOR
    z = (x - mu) / np.where(ok, sd, 1.0)
    return np.where(ok, z, 0.0), ok[:, 0]


def observation_rank(xz, rtol: float = OBSERVATION_RANK_RTOL) -> int:
    """Numerical rank of the z-scored observation (singular values above
    ``rtol`` x the largest)."""
    xz = np.asarray(xz, dtype=float)
    if xz.size == 0:
        return 0
    s = np.linalg.svd(xz, compute_uv=False)
    if not s.size or s[0] <= 0:
        return 0
    return int(np.sum(s > float(rtol) * s[0]))


def block_representation(x_block, representation: str = REPRESENTATION_BLOCK_MEAN,
                         d: int = 1) -> Optional[np.ndarray]:
    """
    The representation (dims x T) of one block's raw node series: the mean
    of the z-scored nodes (``block_mean``), or (``all_units``) every
    non-constant z-scored node when there are at most ``d`` of them, else
    the ``d`` leading principal-component series (v1 ``_block_components``).
    None when the block has no variance.
    """
    z, ok = zscore_rows(x_block)
    if not ok.any():
        return None
    if representation == REPRESENTATION_BLOCK_MEAN:
        y = z.mean(axis=0, keepdims=True)
        return y if float(y.std()) > _SD_FLOOR else None
    if representation != REPRESENTATION_ALL_UNITS:
        raise NASError(f"representation must be one of {REPRESENTATIONS}")
    units = z[ok]
    if units.shape[0] <= int(d):
        return units
    return mm._block_components(np.asarray(x_block, dtype=float), int(d))


# --------------------------------------------------------------------------
# Gram design and the statistic
# --------------------------------------------------------------------------
def _past(y: np.ndarray, lags: Sequence[int], l_max: int) -> np.ndarray:
    """Past columns (n x d|L|), lag-major as v1 ``_lagged_rows``."""
    T = y.shape[1]
    return np.vstack([y[:, l_max - lag:T - lag] for lag in lags]).T


class GramDesign:
    """
    The regression design of one scoring: columns ``[1 | U | per block:
    past at the lags L, current]`` on the rows ``t = l_max .. T - 1`` and its
    Gram matrix. ``span`` is the orthonormal span of the input basis on these
    rows (n x rank) or None. Hub-agnostic: any block can be shifted
    (:meth:`shifted`) and any block can play the hub in
    :func:`nas_statistics` (the hub-identity table reuses one design).
    """

    def __init__(self, blocks: Mapping[str, np.ndarray], lags: Sequence[int],
                 span: Optional[np.ndarray] = None):
        if not blocks:
            raise NASError("a design needs at least one block")
        self.lags = tuple(sorted({int(v) for v in lags}))
        if not self.lags or self.lags[0] < 1:
            raise NASError("lags must be positive integers")
        self.l_max = int(self.lags[-1])
        self.blocks = OrderedDict((str(k), np.asarray(v, dtype=float))
                                  for k, v in blocks.items())
        shapes = {v.shape[1] for v in self.blocks.values()}
        if len(shapes) != 1 or any(v.ndim != 2 for v in self.blocks.values()):
            raise NASError("blocks must be (dims x T) arrays of one length")
        self.n_time = int(shapes.pop())
        self.n = self.n_time - self.l_max
        if self.n < 2:
            raise NASError("the run is shorter than the lags")
        cols = [np.ones((self.n, 1))]
        idx: Dict = {"one": np.arange(0, 1)}
        pos = 1
        if span is not None and np.asarray(span).shape[1] > 0:
            span = np.asarray(span, dtype=float)
            if span.shape[0] != self.n:
                raise NASError("the input span must cover the regression rows")
            cols.append(span)
            idx["U"] = np.arange(pos, pos + span.shape[1])
            pos += span.shape[1]
        else:
            idx["U"] = np.arange(pos, pos)
        for k, y in self.blocks.items():
            p = _past(y, self.lags, self.l_max)
            c = y[:, self.l_max:].T
            idx[("past", k)] = np.arange(pos, pos + p.shape[1])
            pos += p.shape[1]
            idx[("cur", k)] = np.arange(pos, pos + c.shape[1])
            pos += c.shape[1]
            cols.extend([p, c])
        self.idx = idx
        self.A = np.ascontiguousarray(np.hstack(cols))
        self.G = self.A.T @ self.A

    @property
    def n_columns(self) -> int:
        return int(self.A.shape[1])

    @property
    def rank_u(self) -> int:
        return int(self.idx["U"].size)

    def dims(self, name) -> int:
        return int(self.blocks[name].shape[0])

    def past_columns(self, name, n_lags: Optional[int] = None) -> np.ndarray:
        """Past columns of a block, optionally only its first ``n_lags`` lags."""
        cols = self.idx[("past", name)]
        if n_lags is None:
            return cols
        return cols[: int(n_lags) * self.dims(name)]

    def block_columns(self, name, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(columns, values)`` of a block's past and current for series ``y``."""
        cols = np.r_[self.idx[("past", name)], self.idx[("cur", name)]]
        vals = np.hstack([_past(y, self.lags, self.l_max), y[:, self.l_max:].T])
        return cols, vals

    def shifted(self, name, shift: int) -> np.ndarray:
        """The Gram matrix with block ``name`` circularly shifted by ``shift``
        samples relative to every other block and to ``U`` (``np.roll``, as
        v1); the regressions are fitted afterwards (shift, then project)."""
        if not int(shift):
            return self.G
        y = np.roll(self.blocks[name], int(shift), axis=1)
        cols, vals = self.block_columns(name, y)
        cross = vals.T @ self.A
        cross[:, cols] = vals.T @ vals
        G = self.G.copy()
        G[cols, :] = cross
        G[:, cols] = cross.T
        return G

    def kept(self, keep: np.ndarray) -> Tuple[np.ndarray, int]:
        """Gram matrix and row count of the rows ``keep`` (jackknife)."""
        keep = np.asarray(keep, dtype=bool)
        Ak = self.A[keep]
        return Ak.T @ Ak, int(keep.sum())


def psd_solve(S: np.ndarray, B: np.ndarray, rtol: float = SOLVE_RTOL
              ) -> Tuple[np.ndarray, int]:
    """
    Minimum-norm solution of ``S x = B`` for a positive semi-definite Gram
    block ``S``: Jacobi scaling, eigen-decomposition and a rank cut at
    ``rtol`` x the largest eigenvalue (rank-revealing). Returns the solution
    and the effective rank.
    """
    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float)
    out = np.zeros((S.shape[0],) + B.shape[1:])
    if S.size == 0:
        return out, 0
    d = np.sqrt(np.clip(np.diag(S), 0.0, None))
    keep = d > 0
    if not keep.any():
        return out, 0
    dk = d[keep]
    C = S[np.ix_(keep, keep)] / np.outer(dk, dk)
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    good = w > float(rtol) * w[-1]
    Vg = V[:, good]
    Bk = B[keep] / (dk[:, None] if B.ndim == 2 else dk)
    if B.ndim == 2:
        xk = Vg @ ((Vg.T @ Bk) / w[good][:, None])
        out[keep] = xk / dk[:, None]
    else:
        xk = Vg @ ((Vg.T @ Bk) / w[good])
        out[keep] = xk / dk
    return out, int(good.sum())


def residual_covariance(G: np.ndarray, n: int, x_cols, y_cols
                        ) -> Tuple[np.ndarray, int]:
    """Residual covariance ``(S_yy - S_yx S_xx^+ S_xy) / n`` of the targets
    ``y_cols`` regressed on ``x_cols`` (both index arrays into ``G``) and the
    effective rank of the regressors."""
    x = np.asarray(x_cols, dtype=int)
    y = np.asarray(y_cols, dtype=int)
    Syy = G[np.ix_(y, y)]
    Sxy = G[np.ix_(x, y)]
    coef, rank = psd_solve(G[np.ix_(x, x)], Sxy)
    S = (Syy - Sxy.T @ coef) / float(n)
    return 0.5 * (S + S.T), rank


def _logdet(S: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(S)
    return float(ld) if sign > 0 else float("nan")


def _sub(S: np.ndarray, positions) -> np.ndarray:
    p = np.asarray(positions, dtype=int)
    return S[np.ix_(p, p)]


def nas_statistics(design: GramDesign, hub: str, periphery: Sequence[str], *,
                   G: Optional[np.ndarray] = None, n: Optional[int] = None,
                   full_residuals: bool = False) -> dict:
    """
    The conditional receive and return statistics (nats) of ``hub`` with each
    block of ``periphery`` on Gram matrix ``G`` (default: the design's) with
    ``n`` rows: ``receive`` and ``return`` per block, ``R`` and ``B`` (their
    means) and, with ``full_residuals``, the residual covariance of every
    current given ``[1, U, all pasts]`` (for the zero-lag descriptor). Three
    kinds of regressions: the full model (shared by every block and both
    directions), the model without the hub's past (every return) and one
    model without ``P_j``'s past per block (receive_j).
    """
    G = design.G if G is None else G
    n = design.n if n is None else int(n)
    periphery = [str(k) for k in periphery]
    if not periphery or hub in periphery:
        raise NASError("the periphery must be non-empty and exclude the hub")
    idx = design.idx
    base = [idx["one"], idx["U"]]
    past = {k: idx[("past", k)] for k in [hub] + periphery}
    cur = {k: idx[("cur", k)] for k in [hub] + periphery}
    targets = np.concatenate([cur[hub]] + [cur[k] for k in periphery])
    pos, start = {}, 0
    for k in [hub] + periphery:
        pos[k] = np.arange(start, start + cur[k].size)
        start += cur[k].size
    x_full = np.concatenate(base + [past[hub]] + [past[k] for k in periphery])
    S_full, rank_full = residual_covariance(G, n, x_full, targets)
    x_no_hub = np.concatenate(base + [past[k] for k in periphery])
    per_targets = np.concatenate([cur[k] for k in periphery])
    S_no_hub, _ = residual_covariance(G, n, x_no_hub, per_targets)
    ld_hub_full = _logdet(_sub(S_full, pos[hub]))
    receive, ret = [], []
    off = cur[hub].size
    for j in periphery:
        others = [past[k] for k in periphery if k != j]
        x_no_j = np.concatenate(base + [past[hub]] + others)
        S_no_j, _ = residual_covariance(G, n, x_no_j, cur[hub])
        receive.append(0.5 * (_logdet(S_no_j) - ld_hub_full))
        pj = pos[j] - off
        ret.append(0.5 * (_logdet(_sub(S_no_hub, pj)) - _logdet(_sub(S_full, pos[j]))))
    out = {
        "receive": [float(v) for v in receive],
        "return": [float(v) for v in ret],
        "R": float(np.mean(receive)),
        "B": float(np.mean(ret)),
        "rank_full_model": int(rank_full),
        "n_rows": int(n),
    }
    if full_residuals:
        out["S_full"] = S_full
        out["positions"] = pos
    return out


def calibrate(observed: float, null_values: Sequence[float]) -> dict:
    """Null mean, SD (ddof 1), excess, z and rank p of one statistic (the v1
    ``_null_calibration_stats``)."""
    st = mm._null_calibration_stats(observed, null_values)
    return {"observed": float(observed), "null_mean": float(st["null_mean"]),
            "null_sd": float(st["null_sd"]), "n_null": int(st["null_n"]),
            "excess": float(st["excess"]), "z": float(st["z"]), "p": float(st["p"])}


def null_degenerate(null_values: Sequence[float], null_sd: float) -> bool:
    """Whether a null is degenerate: fewer than two finite draws, or an SD
    that is not finite or not above :data:`NULL_SD_RTOL` x the largest
    absolute draw (zero up to rounding)."""
    v = np.asarray(null_values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    sd = float(null_sd)
    if v.size < 2 or not math.isfinite(sd):
        return True
    return sd <= NULL_SD_RTOL * float(np.max(np.abs(v)))


def _calibrated_run(design: GramDesign, hub: str, periphery: Sequence[str],
                    shifts: Sequence[int], *, full_residuals: bool = False) -> dict:
    """Observed statistics and their hub-shift null on one design."""
    obs = nas_statistics(design, hub, periphery, full_residuals=full_residuals)
    null_R, null_B = [], []
    null_rec = [[] for _ in periphery]
    null_ret = [[] for _ in periphery]
    for s in shifts:
        st = nas_statistics(design, hub, periphery, G=design.shifted(hub, s))
        null_R.append(st["R"])
        null_B.append(st["B"])
        for j in range(len(periphery)):
            null_rec[j].append(st["receive"][j])
            null_ret[j].append(st["return"][j])
    return {"observed": obs, "null_R": null_R, "null_B": null_B,
            "null_receive": null_rec, "null_return": null_ret,
            "cal": {RECEIVE: calibrate(obs["R"], null_R),
                    RETURN: calibrate(obs["B"], null_B)}}


def pooled_statistic(x, hub: Sequence[int], periphery: Sequence[int],
                     lags: Sequence[int] = (1, 2), *, n_components: int = V1_COMPONENTS,
                     n_null: int = N_NULL, null_seed: int = 0,
                     min_shift: Optional[int] = None) -> dict:
    """
    The pooled statistic on the v1 geometry, computed by the v3 Gram code:
    hub and pooled periphery each reduced to ``n_components`` leading
    components (v1 ``_block_components``), no inputs, lags ``lags``, the v1
    hub-shift null. With ``n_components = 5`` and the v1 lags this is the
    frozen v1 ``compute_NAS(mode='capacity')`` statistic (``te_in``,
    ``te_out``, their nulls, the limiting direction of smaller ``z`` and its
    excess ``value``); with 1 it is the rank-1 descriptor. ``significant``:
    both ``z > 1.645``.
    """
    x = np.asarray(x, dtype=float)
    hub = np.asarray(hub, dtype=int)
    periphery = np.asarray(periphery, dtype=int)
    lags = tuple(sorted({int(v) for v in lags}))
    out = {"n_components": int(n_components), "lags": list(lags), "defined": False,
           "reason": None}
    comp_h = mm._block_components(x[hub], int(n_components))
    comp_p = mm._block_components(x[periphery], int(n_components))
    if comp_h is None or comp_p is None:
        out["reason"] = NO_VARIANCE
        return out
    n_time = x.shape[1]
    n_par = 1 + (comp_h.shape[0] + comp_p.shape[0]) * len(lags)
    out.update(components_hub=int(comp_h.shape[0]),
               components_periphery=int(comp_p.shape[0]))
    if n_time - max(lags) <= 3 * n_par:
        out["reason"] = "insufficient_timepoints_for_transfer_model"
        return out
    design = GramDesign({"H": comp_h, "P": comp_p}, lags)
    shifts = null_shifts(n_time, n_null, null_seed, min_shift)
    run = _calibrated_run(design, "H", ["P"], shifts)
    cal_in, cal_out = run["cal"][RECEIVE], run["cal"][RETURN]
    for name, cal, null in (("in", cal_in, run["null_R"]),
                            ("out", cal_out, run["null_B"])):
        out.update({f"te_{name}": cal["observed"],
                    f"te_{name}_null_mean": cal["null_mean"],
                    f"te_{name}_null_sd": cal["null_sd"],
                    f"te_{name}_excess": cal["excess"],
                    f"te_{name}_z": cal["z"], f"te_{name}_p": cal["p"],
                    f"te_{name}_null": [float(v) for v in null]})
    z_pair = np.asarray([cal_in["z"], cal_out["z"]], dtype=float)
    out["null_shifts"] = shifts
    if not np.all(np.isfinite(z_pair)):
        out["reason"] = TRANSFER_NULL_DEGENERATE
        return out
    lim = int(np.argmin(z_pair))
    cal = (cal_in, cal_out)[lim]
    out.update(defined=True, limiting_direction=("in", "out")[lim],
               value=cal["excess"], raw=cal["observed"],
               null_mean=cal["null_mean"], null_sd=cal["null_sd"],
               significant=bool(np.all(z_pair > Z_SIGNIFICANT)))
    return out


# --------------------------------------------------------------------------
# jackknife
# --------------------------------------------------------------------------
def jackknife_groups(n_time: int, groups: int, scheme: str = SE_CONTIGUOUS,
                     block_samples: Optional[int] = None) -> np.ndarray:
    """
    The jackknife group of every sample: ``contiguous`` cuts the run into
    ``groups`` stretches at ``round(linspace(0, T, G + 1))``; ``interleaved``
    cuts it into blocks of ``block_samples`` and assigns block ``i`` to group
    ``i mod G``.
    """
    n_time, groups = int(n_time), int(groups)
    if groups < 2:
        raise NASError("a jackknife needs at least two groups")
    t = np.arange(n_time)
    if scheme == SE_CONTIGUOUS:
        edges = np.linspace(0, n_time, groups + 1).round().astype(int)
        return np.clip(np.searchsorted(edges, t, side="right") - 1, 0, groups - 1)
    if scheme == SE_INTERLEAVED:
        if block_samples is None or int(block_samples) < 1:
            raise NASError("the interleaved scheme needs block_samples >= 1")
        return (t // int(block_samples)) % groups
    raise NASError(f"scheme must be one of {SE_SCHEMES}")


def jackknife_keep(group_of_sample: np.ndarray, l_max: int, groups: int
                   ) -> np.ndarray:
    """
    Keep masks (G x n) over the regression rows ``t = l_max .. T - 1``:
    replicate ``g`` drops every row whose window ``[t - l_max, t]`` touches a
    sample of group ``g``.
    """
    gid = np.asarray(group_of_sample, dtype=int)
    T = gid.size
    l_max = int(l_max)
    rows = np.arange(l_max, T)
    keeps = np.ones((int(groups), rows.size), dtype=bool)
    for g in range(int(groups)):
        c = np.concatenate([[0], np.cumsum(gid == g)])
        touched = c[rows + 1] - c[rows - l_max] > 0
        keeps[g] = ~touched
    return keeps


def jackknife_se(replicates: Sequence[float]) -> float:
    """``sqrt((G - 1) / G sum (theta_g - mean)^2)`` (NaN unless every
    replicate is finite)."""
    r = np.asarray(replicates, dtype=float)
    if r.size < 2 or not np.all(np.isfinite(r)):
        return float("nan")
    g = r.size
    return float(np.sqrt((g - 1) / g * np.sum((r - r.mean()) ** 2)))


# --------------------------------------------------------------------------
# descriptors
# --------------------------------------------------------------------------
def _metastability(x, dt, bands) -> dict:
    fs = 1.0 / float(dt)
    nyq = 0.5 * fs
    out = {"value": None, "mean_order_parameter": None,
           "bands": [list(b) for b in bands], "nyquist_hz": nyq, "reason": None}
    if not bands:
        out["reason"] = "no_bands_declared"
        return out
    if any(hi >= nyq for _lo, hi in bands):
        out["reason"] = BAND_ABOVE_NYQUIST
        return out
    if any(lo <= 0 or hi <= lo for lo, hi in bands):
        out["reason"] = "invalid_band"
        return out
    vals, means = [], []
    for lo, hi in bands:
        m, r = mm._kuramoto_metastability(x, float(lo), float(hi), fs)
        vals.append(m)
        means.append(r)
    out.update(value=float(np.mean(vals)), mean_order_parameter=float(np.mean(means)))
    return out


def _profile(x, dt, hub) -> dict:
    out = {"L": None, "B": None, "H": None, "reason": None}
    if len(hub) < 2:
        out["reason"] = "workspace_too_small_for_profile_descriptors"
        return out
    nyq = 0.5 / float(dt)
    if any(hi >= nyq for _lo, hi in PROFILE_PARAMETERS["bands"]):
        out["reason"] = BAND_ABOVE_NYQUIST
        return out
    legacy = mm.compute_NAS(np.asarray(x, dtype=float), tr=float(dt),
                            workspace_nodes=np.asarray(hub, dtype=int),
                            return_details=True, normalize=False,
                            **PROFILE_PARAMETERS)
    bw = np.asarray(PROFILE_PARAMETERS["band_weights"], dtype=float)
    bw = bw / float(bw.sum())
    for key in ("L", "B", "H"):
        vals = [c.get(key, float("nan")) for c in legacy["band_components"]]
        out[key] = float(np.dot(bw, np.asarray(vals, dtype=float)))
    return out


def _zero_lag(S_full, positions, hub, periphery) -> dict:
    """First canonical correlation between the full-model residuals of the
    hub and of the stacked periphery."""
    ph = positions[hub]
    pp = np.concatenate([positions[k] for k in periphery])
    Shh, Spp = _sub(S_full, ph), _sub(S_full, pp)
    Shp = S_full[np.ix_(ph, pp)]

    def inv_sqrt(S):
        w, V = np.linalg.eigh(0.5 * (S + S.T))
        good = w > SOLVE_RTOL * max(float(w[-1]), 0.0)
        return (V[:, good] / np.sqrt(w[good])) @ V[:, good].T

    M = inv_sqrt(Shh) @ Shp @ inv_sqrt(Spp)
    s = np.linalg.svd(M, compute_uv=False)
    return {"value": float(min(1.0, s[0])) if s.size else None}


def _bic_order(design: GramDesign, hub, periphery) -> dict:
    names = [hub] + list(periphery)
    targets = np.concatenate([design.idx[("cur", k)] for k in names])
    D = int(targets.size)
    n = design.n
    rows = []
    for q in range(1, len(design.lags) + 1):
        x = np.concatenate([design.idx["one"], design.idx["U"]]
                           + [design.past_columns(k, q) for k in names])
        S, _ = residual_covariance(design.G, n, x, targets)
        k_q = D * (1 + design.rank_u + D * q)
        bic = n * _logdet(S) + k_q * math.log(n)
        rows.append({"n_lags": q, "lags": list(design.lags[:q]), "bic": float(bic)})
    finite = [r for r in rows if math.isfinite(r["bic"])]
    best = min(finite, key=lambda r: r["bic"]) if finite else None
    return {"order": None if best is None else best["n_lags"],
            "lags": None if best is None else best["lags"], "table": rows}


def _run_descriptor(name, fn, store):
    try:
        store[name] = fn()
    except Exception as exc:  # noqa: BLE001 - a descriptor never aborts NAS
        store[name] = {"error": type(exc).__name__, "message": R.clean_detail(exc)}


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------
def _not_defined(detail) -> str:
    return R.format_reason(R.NOT_DEFINED, R.clean_detail(detail))


def _nan_direction(name, se_method) -> dict:
    nan = float("nan")
    return {"direction": name, "statistic": DIRECTION_STATISTICS[name],
            "estimate": nan, "null_mean": nan, "null_sd": nan, "n_null": 0,
            "null_family": None, "excess": nan, "z": nan, "p": nan,
            "null_values": [], "se": nan, "se_df": None, "se_method": se_method,
            "jackknife": []}


def _result(details, ident, *, reason=None, directions=None, se_method=None) -> dict:
    if reason is not None:
        R.lookup(reason)
    defined = reason is None
    dirs = directions or {d: _nan_direction(d, se_method) for d in DIRECTIONS}
    nan = float("nan")
    out = {
        "principle": PRINCIPLE,
        "estimator_version": ESTIMATOR_VERSION,
        "estimator_id": ESTIMATOR_ID,
        "defined": defined,
        "reason": reason,
        "directions": dirs,
        "identifiability": ident,
        "significant": False,
        "limiting_direction": None,
        "estimate": nan, "value": nan, "null_mean": nan, "null_sd": nan,
        "n_null": 0, "null_family": None, "se": nan, "se_df": None,
        "se_method": se_method, "exact": False,
        "details": details,
    }
    if defined:
        zs = [dirs[d]["z"] for d in DIRECTIONS]
        lim = DIRECTIONS[int(np.argmin(zs))]
        dd = dirs[lim]
        out.update(
            significant=bool(all(z > Z_SIGNIFICANT for z in zs)),
            limiting_direction=lim, estimate=dd["estimate"], value=dd["excess"],
            null_mean=dd["null_mean"], null_sd=dd["null_sd"], n_null=dd["n_null"],
            null_family=NULL_FAMILY, se=dd["se"], se_df=dd["se_df"])
    return out


def _check_nodes(nodes, what) -> List[int]:
    items = list(nodes)
    if any(isinstance(v, (bool, np.bool_)) for v in items):
        raise NASError(f"{what} must be integer node indices, not a boolean mask")
    arr = np.asarray(items, dtype=float).reshape(-1)
    if arr.size and (np.any(arr != np.round(arr)) or np.any(~np.isfinite(arr))):
        raise NASError(f"{what} must be integer node indices")
    return [int(v) for v in arr]


def _inputs_basis(inputs, lags, taus, n_time, dt):
    """``(InputBasis or None, shared_inputs)`` of the declared inputs."""
    if inputs is None:
        return None, "none"
    if isinstance(inputs, DI.InputBasis):
        basis = inputs
        if tuple(basis.lags) != (0,) + tuple(lags):
            raise NASError(f"the input basis has lags {basis.lags}; NAS needs "
                           f"{(0,) + tuple(lags)}")
        if len(basis.taus) != len(taus) or not all(
                math.isclose(a, b, rel_tol=1e-9) for a, b in zip(basis.taus, taus)):
            raise NASError(f"the input basis has filter time constants "
                           f"{basis.taus}; NAS needs {tuple(taus)}")
    elif isinstance(inputs, DI.DeclaredInputs):
        basis = DI.input_basis(inputs, lags=(0,) + tuple(lags), taus=taus)
    else:
        raise NASError("inputs must be DeclaredInputs, an InputBasis or None")
    if basis.n_time != int(n_time) or not math.isclose(basis.dt, float(dt),
                                                       rel_tol=1e-9):
        raise NASError("the input basis does not match the recording's grid")
    return basis, basis.shared_inputs


def _representations(x, hub, blocks, representation, dims_ladder, budget, xz_rank,
                     n_time, l_max, n_lags, rank_u, params):
    """The representations of hub and blocks and the budget record; reason
    when no allowed dimension passes."""
    names = ["__hub__"] + list(blocks)
    nodes = [hub] + [blocks[k] for k in blocks]
    tried = []
    for d in (dims_ladder if representation == REPRESENTATION_ALL_UNITS else (1,)):
        reps = []
        for nd in nodes:
            y = block_representation(x[nd], representation, d)
            if y is None:
                return None, {"tried": tried}, _not_defined(NO_VARIANCE)
            reps.append(y)
        total = int(sum(y.shape[0] for y in reps))
        n_par = 1 + int(rank_u) + int(n_lags) * total
        rows = int(n_time - l_max)
        ok_t = rows >= params.budget_factor * n_par
        ok_r = total <= params.rank_fraction * xz_rank
        tried.append({"d_max": int(d), "total_dims": total, "n_par": n_par,
                      "rows": rows, "timepoints_ok": bool(ok_t), "rank_ok": bool(ok_r)})
        if ok_t and ok_r:
            budget.update(d_max=int(d), total_dims=total, n_par=n_par,
                          block_dims={nm: int(y.shape[0])
                                      for nm, y in zip(names, reps)})
            return reps, {"tried": tried}, None
    return None, {"tried": tried}, R.INSUFFICIENT_TIMEPOINTS


def compute_nas_v3(
    ts,
    *,
    dt: float,
    hub: Sequence[int],
    blocks: Mapping[str, Sequence[int]],
    inputs=None,
    params=None,
    seed: int = 0,
    null_seed: Optional[int] = None,
    hub_name: str = "hub",
    observation: str = OBSERVATION_DIRECT,
    observation_admitted: bool = False,
    override_observation_gate: bool = False,
) -> dict:
    """
    NAS v3 of one recording (module docstring).

    ``ts`` is nodes x time, ``dt`` the sampling interval (s); ``hub`` the
    declared hub rows and ``blocks`` the declared periphery blocks
    (``{name: rows}``, in order); ``inputs`` the declared exogenous inputs
    (:class:`~impact_pipeline.v2.declared_inputs.DeclaredInputs`, whose basis
    is built here with the NAS lags, an :class:`InputBasis` with lags ``{0}
    U L`` and the filter time constants of the lag plan, or None for no
    inputs); ``params`` a :class:`NASParams` or a protocol block with
    ``coupling_timescale_sec`` declared; ``seed`` the
    system seed (null seed ``seed * 1000 + 17`` unless ``null_seed`` is
    given); ``observation`` one of the identifiability observations;
    ``observation_admitted`` whether a registry entry admits a mixed
    observation; ``override_observation_gate`` computes NAS on a mixed
    observation without admission for the forward-model admission arm
    (labelled ``observation_gate = 'overridden'``).
    """
    p = NASParams.from_mapping(params)
    tau_c = p.coupling_timescale()
    tau_l = p.lag_timescale()
    dt = float(dt)
    if not (math.isfinite(dt) and dt > 0):
        raise NASError("dt must be finite and > 0")
    if observation not in OBSERVATIONS:
        raise NASError(f"observation must be one of {OBSERVATIONS}")
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2:
        raise NASError(f"ts must be 2D (nodes x time), got shape {x.shape}")
    n_nodes, n_time = x.shape
    plan = lag_plan(tau_c, dt, tau_l)
    lags = tuple(plan["lags"])
    l_max = int(plan["l_max"])
    hub_rows = _check_nodes(hub, "hub")
    block_rows = OrderedDict((str(k), _check_nodes(v, f"block {k}"))
                             for k, v in blocks.items())
    if len(block_rows) != len(blocks):
        raise NASError("periphery block names must be distinct as strings")
    if str(hub_name) in block_rows:
        raise NASError(f"hub_name {hub_name!r} is also a periphery block name")
    if inputs is None:
        shared = "none"
    elif isinstance(inputs, (DI.DeclaredInputs, DI.InputBasis)):
        shared = inputs.shared_inputs
    else:
        raise NASError("inputs must be DeclaredInputs, an InputBasis or None")
    ident = identifiability(shared, observation)
    seed_null = null_seed_for(seed) if null_seed is None else int(null_seed)
    details = {
        "estimator_id": ESTIMATOR_ID,
        "mode": MODE,
        "params": p.to_dict(),
        "representation": p.block_representation,
        "lag_plan": plan,
        "lags": list(lags),
        "hub_name": str(hub_name),
        "hub_nodes": hub_rows,
        "blocks": {k: v for k, v in block_rows.items()},
        "n_time": int(n_time),
        "dt": dt,
        "seed": int(seed),
        "null_seed": seed_null,
        "null_order": NULL_ORDER,
        "observation": observation,
        "observation_gate": "not_applicable",
        "declaration": ("none" if inputs is None else
                        getattr(inputs, "label", None) or inputs.declaration_id),
        "descriptors": {},
    }
    se_method = p.se_method

    def finish(reason=None, directions=None):
        res = _result(details, ident, reason=reason, directions=directions,
                      se_method=se_method)
        _signal_descriptors(details["descriptors"], x, dt, hub_rows, p)
        return res

    # gate 1: shape and declarations
    if not np.all(np.isfinite(x)):
        return finish(_not_defined(NON_FINITE_TIMESERIES))
    periphery_rows = [i for v in block_rows.values() for i in v]
    if not hub_rows or not block_rows or any(not v for v in block_rows.values()):
        return finish(_not_defined(HUB_OR_PERIPHERY_EMPTY))
    all_rows = hub_rows + periphery_rows
    if (min(all_rows) < 0 or max(all_rows) >= n_nodes
            or len(set(all_rows)) != len(all_rows)):
        return finish(_not_defined(INVALID_WORKSPACE))
    lo_shift = null_min_shift(n_time, p.null_min_shift_fraction)
    if n_time - lo_shift < lo_shift:
        return finish(_not_defined(RUN_TOO_SHORT))
    # gate 2: resolvability (the coupling time scale, never the lag scale)
    if not plan["resolved"]:
        return finish(R.SAMPLING_UNRESOLVED)
    # gate 3: observation
    if observation in OBSERVATIONS_MIXED:
        if observation_admitted:
            details["observation_gate"] = "admitted"
        elif override_observation_gate:
            details["observation_gate"] = "overridden"
        else:
            details["observation_gate"] = "not_admitted"
            return finish(R.OBSERVATION_MIXED_NOT_ADMITTED)
    # inputs, budget and representation
    basis, _shared = _inputs_basis(inputs, lags, plan["basis_taus"], n_time, dt)
    span = None
    if basis is not None and basis.n_columns:
        span = basis.orthonormal_span(start=l_max)
    rank_u = 0 if span is None else int(span.shape[1])
    details["basis"] = None if basis is None else {
        "declaration": basis.declaration_id, "shared_inputs": basis.shared_inputs,
        "channels": list(basis.channels), "n_columns": int(basis.n_columns),
        "rank": rank_u, "lags": list(basis.lags), "taus": list(basis.taus),
        "dropped": [list(d) for d in basis.dropped], "sha256": basis.sha256()}
    xz, _ok = zscore_rows(x[sorted(all_rows)])
    obs_rank = observation_rank(xz)
    budget = {"rank_u": rank_u, "n_lags": len(lags), "observation_rank": obs_rank,
              "budget_factor": p.budget_factor, "rank_fraction": p.rank_fraction}
    details["budget"] = budget
    reps, ladder, reason = _representations(
        x, hub_rows, block_rows, p.block_representation, p.secondary_dimensions,
        budget, obs_rank, n_time, l_max, len(lags), rank_u, p)
    budget.update(ladder)
    if reason is not None:
        return finish(reason)
    hub_key = str(hub_name)
    design_blocks = OrderedDict([(hub_key, reps[0])] + list(zip(block_rows, reps[1:])))
    periphery = list(block_rows)
    design = GramDesign(design_blocks, lags, span)
    shifts = null_shifts(n_time, p.n_null, seed_null, lo_shift)
    details["null_shifts"] = shifts
    details["n_columns"] = design.n_columns
    run = _calibrated_run(design, hub_key, periphery, shifts, full_residuals=True)
    obs = run["observed"]
    details["rank_full_model"] = obs["rank_full_model"]
    # per-block values and coverage
    per_block = {}
    n_bidir = 0
    for j, k in enumerate(periphery):
        cr = calibrate(obs["receive"][j], run["null_receive"][j])
        cb = calibrate(obs["return"][j], run["null_return"][j])
        cr["null_values"] = [float(v) for v in run["null_receive"][j]]
        cb["null_values"] = [float(v) for v in run["null_return"][j]]
        for c in (cr, cb):
            c["null_degenerate"] = null_degenerate(c["null_values"], c["null_sd"])
            if c["null_degenerate"]:
                c["z"] = float("nan")
        both = bool(cr["z"] > Z_SIGNIFICANT and cb["z"] > Z_SIGNIFICANT)
        n_bidir += int(both)
        per_block[k] = {RECEIVE: cr, RETURN: cb, "bidirectional": both}
    details["per_block"] = per_block
    details["coverage"] = float(n_bidir / len(periphery))
    details["n_bidirectional"] = int(n_bidir)
    # gate 5: null
    for d in DIRECTIONS:
        cal = run["cal"][d]
        if not math.isfinite(cal["observed"]):
            return finish(_not_defined(NON_FINITE_STATISTIC))
        null = run["null_R"] if d == RECEIVE else run["null_B"]
        if null_degenerate(null, cal["null_sd"]) or not math.isfinite(cal["z"]):
            return finish(_not_defined(TRANSFER_NULL_DEGENERATE))
    # SE: delete-a-group jackknife (both directions)
    scheme, groups = p.se_scheme, p.se_groups
    block_samples = (interleave_block_samples(dt, tau_c, p.interleave_block_sec)
                     if scheme == SE_INTERLEAVED else None)
    gid = jackknife_groups(n_time, groups, scheme, block_samples)
    reps_R, reps_B, se_reason = [], [], None
    if np.unique(gid).size < groups:
        se_reason = "jackknife_groups_empty"
    else:
        for keep in jackknife_keep(gid, l_max, groups):
            Gk, nk = design.kept(keep)
            st = nas_statistics(design, hub_key, periphery, G=Gk, n=nk)
            reps_R.append(st["R"])
            reps_B.append(st["B"])
    details["jackknife"] = {"scheme": scheme, "groups": groups,
                            "block_samples": block_samples, "reason": se_reason}
    directions = {}
    for d, reps, null in ((RECEIVE, reps_R, run["null_R"]),
                          (RETURN, reps_B, run["null_B"])):
        cal = run["cal"][d]
        se = jackknife_se(reps) if se_reason is None else float("nan")
        if se_reason is None and not math.isfinite(se):
            details["jackknife"]["reason"] = "non_finite_replicate"
        directions[d] = {
            "direction": d, "statistic": DIRECTION_STATISTICS[d],
            "estimate": cal["observed"], "null_mean": cal["null_mean"],
            "null_sd": cal["null_sd"], "n_null": cal["n_null"],
            "null_family": NULL_FAMILY, "excess": cal["excess"], "z": cal["z"],
            "p": cal["p"], "significant": bool(cal["z"] > Z_SIGNIFICANT),
            "null_values": [float(v) for v in null], "se": se,
            "se_df": float(groups - 1), "se_method": se_method,
            "jackknife": [float(v) for v in reps]}
    res = finish(None, directions)
    _statistic_descriptors(details["descriptors"], x, hub_rows, periphery_rows,
                           design, hub_key, periphery, run, shifts, seed_null,
                           lo_shift, p)
    return res


def _signal_descriptors(store, x, dt, hub_rows, p):
    if DESCRIPTOR_METASTABILITY in p.descriptors:
        _run_descriptor(DESCRIPTOR_METASTABILITY,
                        lambda: _metastability(x, dt, p.metastability_bands), store)
    if DESCRIPTOR_PROFILE in p.descriptors:
        _run_descriptor(DESCRIPTOR_PROFILE, lambda: _profile(x, dt, hub_rows), store)


def _statistic_descriptors(store, x, hub_rows, periphery_rows, design, hub_key,
                           periphery, run, shifts, seed_null, lo_shift, p):
    per_sorted = sorted(periphery_rows)
    if DESCRIPTOR_POOLED_V1 in p.descriptors:
        _run_descriptor(DESCRIPTOR_POOLED_V1, lambda: pooled_statistic(
            x, hub_rows, per_sorted, design.lags, n_components=p.v1_components,
            n_null=p.n_null, null_seed=seed_null, min_shift=lo_shift), store)
    if DESCRIPTOR_POOLED_RANK1 in p.descriptors:
        _run_descriptor(DESCRIPTOR_POOLED_RANK1, lambda: pooled_statistic(
            x, hub_rows, per_sorted, design.lags, n_components=1,
            n_null=p.n_null, null_seed=seed_null, min_shift=lo_shift), store)
    if DESCRIPTOR_CONDITIONING_DELTA in p.descriptors:
        def delta():
            if design.rank_u == 0:
                return {RECEIVE: 0.0, RETURN: 0.0, "inputs": False}
            bare = GramDesign(design.blocks, design.lags, None)
            run0 = _calibrated_run(bare, hub_key, periphery, shifts)
            return {d: float(run["cal"][d]["excess"] - run0["cal"][d]["excess"])
                    for d in DIRECTIONS} | {
                "excess_without_inputs": {d: run0["cal"][d]["excess"]
                                          for d in DIRECTIONS},
                "inputs": True}
        _run_descriptor(DESCRIPTOR_CONDITIONING_DELTA, delta, store)
    if DESCRIPTOR_ZERO_LAG in p.descriptors:
        obs = run["observed"]
        _run_descriptor(DESCRIPTOR_ZERO_LAG, lambda: _zero_lag(
            obs["S_full"], obs["positions"], hub_key, periphery), store)
    if DESCRIPTOR_BIC in p.descriptors:
        _run_descriptor(DESCRIPTOR_BIC, lambda: _bic_order(design, hub_key, periphery),
                        store)
    run["observed"].pop("S_full", None)
    run["observed"].pop("positions", None)


# --------------------------------------------------------------------------
# bench systems
# --------------------------------------------------------------------------
def system_coupling_timescale(system, params=None) -> float:
    """
    ``tau_c`` of a system: the protocol's ``coupling_timescale_sec`` when
    declared, checked against the generator constants when the system has
    them; else the generator constants. A declaration that contradicts the
    generator raises (the sensitivity analysis varies
    ``lag_timescale_sec``, never ``tau_c``).
    """
    p = NASParams.from_mapping(params)
    try:
        derived = DI.coupling_timescale(system).tau_c_sec
    except DI.DeclaredInputsError:
        derived = None
    declared = p.coupling_timescale_sec
    if declared is None:
        if derived is None:
            raise NASError("no generator constant gives tau_c; declare "
                           "coupling_timescale_sec")
        return float(derived)
    if derived is not None and not math.isclose(declared, derived, rel_tol=1e-9):
        raise NASError(
            f"coupling_timescale_sec {declared} contradicts the generator's "
            f"tau_c {derived}; vary lag_timescale_sec for the sensitivity analysis")
    return float(declared)


def nas_v3_system(system, declaration_id: str = "R", *, params=None,
                  recorded=None, declared=None, include_endogenous: bool = False,
                  seed: Optional[int] = None, null_seed: Optional[int] = None,
                  observation: Optional[str] = None,
                  observation_admitted: bool = False,
                  override_observation_gate: bool = False) -> dict:
    """
    NAS v3 of a bench system under one declaration: hub and periphery blocks
    from the declared meta (:func:`declared_blocks`), ``tau_c`` from
    :func:`system_coupling_timescale`, the inputs recorded by the recording
    device and declared by ``declaration_id`` (or ``declared``), the
    observation from the meta (:func:`observation_of`) and the system seed.
    """
    meta = system.meta or {}
    p = NASParams.from_mapping(params)
    p = p.replace(coupling_timescale_sec=system_coupling_timescale(system, p))
    hub_name, hub, blocks = declared_blocks(meta)
    if declared is None:
        rec = recorded if recorded is not None else DI.record_inputs(system)
        declared = DI.declare(rec, declaration_id,
                              include_endogenous=include_endogenous)
    obs = observation_of(meta) if observation is None else observation
    seed = int(meta.get("seed", 0) or 0) if seed is None else int(seed)
    return compute_nas_v3(
        system.ts, dt=float(system.dt), hub=hub, blocks=blocks, inputs=declared,
        params=p, seed=seed, null_seed=null_seed, hub_name=hub_name,
        observation=obs, observation_admitted=observation_admitted,
        override_observation_gate=override_observation_gate)


# --------------------------------------------------------------------------
# records, evidence and anchors
# --------------------------------------------------------------------------
def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def component_fields(result: Mapping) -> dict:
    """
    The estimator fields of a ``mpc-bench-result/3`` component: ``estimate``,
    null moments and family, ``se``, ``se_df`` and ``se_method`` of the
    limiting direction (smaller ``z``; reported as in v1, never read by the
    status), the ``identifiability`` record and ``details`` with both
    directions; the runner adds status, reason, ``c``, ``c_R``, ``c_B`` and
    the protocol fields from the status rule. An undefined result carries no
    estimate or SE.
    """
    defined = bool(result.get("defined"))
    details = dict(result.get("details") or {})
    details.update(defined=defined, estimator_reason=result.get("reason"),
                   directions=result.get("directions"),
                   limiting_direction=result.get("limiting_direction"),
                   significant=bool(result.get("significant")), exact=False)

    def f(key):
        v = result.get(key)
        if not defined or v is None:
            return None
        v = float(v)
        return v if math.isfinite(v) else None

    return {
        "estimate": f("estimate"),
        "null_mean": f("null_mean"),
        "null_sd": f("null_sd"),
        "n_null": int(result.get("n_null") or 0) if defined else 0,
        "null_family": result.get("null_family") if defined else None,
        "se": f("se"),
        "se_df": f("se_df"),
        "se_method": result.get("se_method") if defined else None,
        "identifiability": result.get("identifiability"),
        "details": _jsonable(details),
    }


def evidence_items(result: Mapping, *, references: Optional[Mapping] = None,
                   **fields) -> tuple:
    """
    The two direction items (``receive``, ``return``) of the status rule
    (:class:`impact_pipeline.evidence_v2.ComponentEvidenceV2`), each with its
    own estimate, null, SE and SE method; an undefined result gives two
    undefined items with its reason. ``references`` optionally maps a
    direction to ``(reference_excess, reference_se)`` (otherwise the
    protocol's ``NAS:<direction>`` anchors apply); ``fields`` are passed on
    (``protocol_id``, ``substrate``, ``grain``, ``bearer_id``, ``nodes``,
    ``regime``, ``observation_stage``, ``view``).
    """
    from impact_pipeline import evidence_v2 as E

    defined = bool(result.get("defined"))
    out = []
    for d in DIRECTIONS:
        dd = (result.get("directions") or {}).get(d) or _nan_direction(
            d, result.get("se_method"))
        kw = dict(principle=PRINCIPLE, direction=d, estimator=ESTIMATOR_ID,
                  defined=defined, reason=result.get("reason"),
                  estimate=float(dd["estimate"]) if defined else float("nan"),
                  null_mean=float(dd["null_mean"]) if defined else float("nan"),
                  null_sd=float(dd["null_sd"]) if defined else float("nan"),
                  n_null=int(dd["n_null"]) if defined else 0,
                  null_family=NULL_FAMILY,
                  se=float(dd["se"]) if defined else float("nan"),
                  se_df=dd["se_df"] if defined else None,
                  se_method=dd["se_method"] if defined else result.get("se_method"))
        if references is not None and d in references:
            ref, ref_se = references[d]
            kw.update(reference=float(ref),
                      reference_se=None if ref_se is None else float(ref_se),
                      reference_scale="excess")
        kw.update(fields)
        out.append(E.ComponentEvidenceV2(**kw))
    return tuple(out)


def direction_excesses(results: Sequence[Mapping]) -> Dict[str, np.ndarray]:
    """Per-direction excesses of several results (NaN for undefined ones),
    the inputs of the per-direction anchors."""
    out = {d: [] for d in DIRECTIONS}
    for r in results:
        for d in DIRECTIONS:
            v = float("nan")
            if r.get("defined"):
                v = float(r["directions"][d]["excess"])
            out[d].append(v)
    return {d: np.asarray(v, dtype=float) for d, v in out.items()}


def anchor_reference(results: Sequence[Mapping]) -> dict:
    """
    The per-direction anchors of a reference block of ``PC_nominal`` results
    (one family, declaration, representation and view): the mean finite
    excess of each direction and its SE (``sd / sqrt(n)``), keyed
    ``NAS:receive`` and ``NAS:return`` as the protocol's external reference
    reads them. Validity and specificity are judged by
    :mod:`impact_pipeline.v2.testability` on the same excesses.
    """
    ex = direction_excesses(results)
    values, ses, counts = {}, {}, {}
    for d in DIRECTIONS:
        x = ex[d][np.isfinite(ex[d])]
        key = f"{PRINCIPLE}:{d}"
        counts[key] = int(x.size)
        if x.size:
            values[key] = float(x.mean())
        if x.size >= 2:
            ses[key] = float(x.std(ddof=1) / math.sqrt(x.size))
    return {"values": values, "se": ses, "n_finite": counts}


def anchor_attributability(anchor_complete: float, anchor_task_only: float) -> float:
    """Shared-input share of the task-only (H) anchor relative to the
    complete-declaration (R) anchor of one direction: ``1 - rho_R /
    rho_H`` (excess scale)."""
    a, b = float(anchor_complete), float(anchor_task_only)
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
        return float("nan")
    return 1.0 - a / b


__all__ = [
    "ALL_DESCRIPTORS",
    "DEFAULT_DESCRIPTORS",
    "DIRECTIONS",
    "DIRECTION_STATISTICS",
    "ESTIMATOR_ID",
    "ESTIMATOR_VERSION",
    "GramDesign",
    "LAG_TIMESCALE_SENSITIVITY_SEC",
    "MODE",
    "NASError",
    "NASParams",
    "NULL_FAMILY",
    "NULL_ORDER",
    "N_NULL",
    "OBSERVATIONS_MIXED",
    "PRINCIPLE",
    "RECEIVE",
    "REPRESENTATIONS",
    "REPRESENTATION_ALL_UNITS",
    "REPRESENTATION_BLOCK_MEAN",
    "RETURN",
    "SECONDARY_DIMENSIONS",
    "SE_METHODS",
    "SE_METHOD_DEFAULT",
    "Z_SIGNIFICANT",
    "anchor_attributability",
    "anchor_reference",
    "block_representation",
    "calibrate",
    "component_fields",
    "compute_nas_v3",
    "declared_blocks",
    "direction_excesses",
    "evidence_items",
    "identifiability",
    "interleave_block_samples",
    "jackknife_groups",
    "jackknife_keep",
    "jackknife_se",
    "lag_plan",
    "nas_statistics",
    "nas_v3_system",
    "null_degenerate",
    "null_min_shift",
    "null_seed_for",
    "null_shifts",
    "observation_of",
    "observation_rank",
    "observation_stage",
    "pooled_statistic",
    "psd_solve",
    "residual_covariance",
    "system_coupling_timescale",
    "zscore_rows",
]
