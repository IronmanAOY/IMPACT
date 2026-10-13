"""
Evidence layer: null-anchored component evidence on a construct scale, the
three-valued MPC verdict (an exclusion rule) and the MPC degree.

Terminology:

- **MPC profile**: the five null-anchored components (RAM, PDI, NAS, IIM, SRPI)
  on the two-anchor construct scale ``c = (m - nu) / (rho - nu)`` (``nu``: mean
  of the declared null family, ``rho``: reference anchor; 0 = null, 1 =
  reference) with sampling intervals.
- **MPC verdict**. Necessary conditions license only exclusion:

  - ``EXCLUDED``: some principle of the necessity set ``N`` is credibly ABSENT
    (under two auxiliary premises: the necessity of ``N``, and the measurement
    validity of the estimator in its registered domain);
  - ``MPC_CONSISTENT``: every principle of ``N`` is PRESENT, i.e.
    consciousness is *not excluded* by the MPC stance. This is **not** an
    attribution of consciousness (no sufficiency claim);
  - ``UNDETERMINED``: otherwise, with reason codes.

  Each component is PRESENT (T), ABSENT (F) or UNDEFINED (U); the channels of
  one principle are combined by strong-Kleene OR, the principles of ``N`` by
  strong-Kleene AND. Missing or undefined evidence can only make the verdict
  UNDETERMINED; it never becomes a zero, and it never flips a determinate
  verdict.
- **MPC degree**: a capped power mean of the construct-scale components,
  computed only for MPC_CONSISTENT verdicts. It is a reference-relative
  evidence summary, not a level of consciousness and not an anaesthesia-depth
  index.
- **Single-source constraint** (formerly "Principle 0"): the evidence must come
  from one declared bearer under one protocol, and components measured on
  different node sets need joint dependence above null
  (:func:`joint_dependence`).

Two status rules decide a component; the :class:`Protocol` selects one. A
protocol without a ``status_rule`` block uses the interval rule (the
empirical pipeline); a protocol with the block uses the TOST rule (version
``tost``, the benchmark). :func:`mpc_verdict` judges by the protocol's rule.

**Interval rule** (:func:`component_assessment`). ``se_c`` combines the
sampling SE of the estimate (block bootstrap), the Monte-Carlo error of the
null mean (``null_sd / sqrt(n_null)``) and, when available, the reference
uncertainty, propagated through the ratio (delta method). With
``z_a = z_{1-alpha}``: PRESENT if ``c - z_a se_c > z_j``; ABSENT if
``c + z_a se_c < delta_j`` (this includes estimates significantly *below* the
null); otherwise UNDEFINED/INCONCLUSIVE. The cutoffs ``delta_j <= z_j`` are
construct-scale constants declared in the :class:`Protocol` (defaults 0.25 and
0.10). An empirical estimate without a sampling SE is UNDEFINED
(``NO_SAMPLING_SE``) unless it is an exact computation (``exact=True``). An SE
from few replicates declares its degrees of freedom (``se_df``: jackknife
groups - 1, bootstrap replicates - 1); ``z_a`` is then the Student ``t``
quantile with the Welch-Satterthwaite degrees of freedom of ``se_c``.

Reason codes (stable strings, ``;``-joined in tables):

- ``MISSING:<P>``: no evidence for principle ``P`` of the necessity set.
- ``MISSING_CHANNEL:<P>:<channel>``: a channel declared by the protocol has no
  evidence item (counts as UNDEFINED, so ``P`` is never ABSENT through it).
- ``NO_NULL_CALIBRATION:<P>``: a defined estimate without a null family.
- ``NO_SAMPLING_SE:<P>``: an empirical estimate without a sampling SE.
- ``INVALID_ANCHORS:<P>``: the reference does not lie above the null mean (or
  one of them is not finite).
- ``INCONCLUSIVE:<P>``: neither credibly above ``z_j`` nor credibly below
  ``delta_j``.
- ``UNDEFINED:<P>:<reason>``: the estimator or the null is undefined (e.g.
  ``DEGENERATE_NULL``, ``INVALID_SE``, ``NULL_FAMILY_MISMATCH``;
  ``UNDEFINED:NAS:NO_DECLARED_WORKSPACE`` when NAS capacity has no declared
  hub).
- ``NOT_IMPLEMENTED:<P>:<channel>``: the evidence channel is not implemented.
- ``ESTIMATOR_NOT_VALIDATED:<P>:<estimator>``: the applicability registry does
  not validate the estimator (version) for this substrate/grain/regime.
- ``ABSENT:<P>``: every channel of ``P`` is credibly absent (exclusion).
- ``BEARER_MISMATCH`` / ``PROTOCOL_MISMATCH`` / ``SOURCE_INCOHERENT``: the
  evidence does not come from one declared bearer / protocol / source
  (``SOURCE_INCOHERENT:UNTESTED`` when components come from different node
  sets and no joint-dependence test covers exactly those sets;
  ``SOURCE_INCOHERENT:INSUFFICIENT_SURROGATES`` when the test had too few
  surrogates to reach its level). These force UNDETERMINED.

The verdict is recoverable from the reasons alone (see
:func:`verdict_from_reasons`): no reasons <=> MPC_CONSISTENT; a global code =>
UNDETERMINED; otherwise any ``ABSENT`` code => EXCLUDED; else UNDETERMINED.
The reasons of a verdict over ``N`` are the union of the single-principle
reasons over ``P in N`` plus the global codes.

**TOST rule** (:func:`assess_item`, :func:`tost_verdict`). For a component
with construct value ``c``, SE ``se_c`` and degrees of freedom ``df``
(computed as by the interval rule: delta method over the sampling, null and
reference parts; Welch-Satterthwaite ``df`` from the item's ``se_df``),
``q_P = t(df, 1 - alpha)`` and ``q_A = t(df, 1 - alpha_A)`` with
``alpha_A = alpha / |N_decl|`` (0.01 for the five declared principles):

* PRESENT iff ``c - q_P se_c > z``;
* ABSENT iff ``c - q_A se_c > -delta`` and ``c + q_A se_c < delta``
  (two one-sided tests);
* otherwise UNDEFINED with the first applicable reason:
  ``NULL_MODEL_VIOLATED`` iff ``c + q_P se_c < -delta``;
  ``ABSENT_NOT_REACHABLE`` iff ``q_A se_c >= delta``; else ``INCONCLUSIVE``.
  Flag ``PRESENT_NOT_REACHABLE`` iff UNDEFINED and ``q_P se_c >= 1 - z``.

Exact computations (known-TPM values, ``exact=True`` without a sampling
SE) use the same inequalities with ``se_c = 0``: PRESENT iff ``c > z``,
ABSENT iff ``|c| < delta``. A data-derived ``se = 0`` is never exact: it
is ``UNDEFINED(NO_SAMPLING_SE)`` unless the component's SE method is
``concordant`` and the protocol admits the concordance route for its
(principle, substrate, observation stage, view, content bearer) cell; there
it is ABSENT iff ``|c| < delta`` and PRESENT iff ``c > z``, each direction
only if admitted, marked ``ABSENT_BY_CONCORDANCE`` / ``PRESENT_BY_CONCORDANCE``.

Each principle feeds one ABSENT decision into the verdict at ``alpha_A``. A
principle with directions (NAS: ``receive`` and ``return``) is PRESENT iff
every direction is PRESENT at ``alpha`` (intersection-union) and ABSENT iff
some direction passes the TOST at ``alpha_A / k`` (union, Bonferroni over
the ``k`` directions). Channels of a principle are combined by
strong-Kleene OR: ABSENT needs every channel ABSENT (an intersection, each at
``alpha_A``), PRESENT through any of ``k`` channels is a union and is tested
at ``alpha / k`` per channel. ``se_df`` is read from the item and must agree
with its SE method (:data:`SE_METHODS`). IIM PRESENT also needs the rank
gate ``p_ind <= 0.05`` when the protocol declares it. Every UNDEFINED reason
comes from :mod:`impact_pipeline.reasons` and is never counted as ABSENT;
the verdict is the strong-Kleene AND over the protocol's necessity set (the
anchored principles), and the AND over the declared set is reported beside
it.

Three entry points share one implementation of the TOST decision
(:func:`_decide`): :func:`assess_item` (one evidence item under a protocol),
:func:`assess_array` (vectorised from raw inputs) and :func:`status_c`
(vectorised on the construct scale), plus :func:`status_c_directional` for
directional principles. The rule's fields and reason codes are documented in
``docs/metrics.md`` (sections 9 and 10).
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist
from types import MappingProxyType
from typing import Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline import reasons as R

PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")

# Estimator versions per principle, recorded in ComponentEvidence.estimator
# ("compute_<P>:<mode>@<version>") and pinned by registry entries: first the
# estimator of mpc_metrics (mpc_metrics.ESTIMATOR_VERSIONS), then its variants
# in impact_pipeline.estimators. Bump a version whenever a change alters that
# estimator's output for fixed inputs and parameters (any mode).
ESTIMATOR_VERSIONS = {
    "RAM": ("ram-2026.09", "ram-pe-2026.10"),
    "PDI": ("pdi-2026.09", "pdi-declared-2026.10"),
    "NAS": ("nas-2026.09", "nas-declared-2026.10"),
    "IIM": ("iim-2026.09", "iim-macro-2026.10"),
    "SRPI": ("srpi-2026.09",),
}


class Verdict(str, Enum):
    EXCLUDED = "EXCLUDED"
    MPC_CONSISTENT = "MPC_CONSISTENT"
    UNDETERMINED = "UNDETERMINED"


class ComponentStatus(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNDEFINED = "UNDEFINED"


PRESENT, ABSENT, UNDEFINED = (
    ComponentStatus.PRESENT, ComponentStatus.ABSENT, ComponentStatus.UNDEFINED)


# Strong-Kleene integer coding: T=1, U=0, F=-1 (AND = min, OR = max).
CODE_T, CODE_U, CODE_F = 1, 0, -1
_STATUS_TO_CODE = {
    ComponentStatus.PRESENT: CODE_T,
    ComponentStatus.UNDEFINED: CODE_U,
    ComponentStatus.ABSENT: CODE_F,
}
_CODE_TO_STATUS = {v: k for k, v in _STATUS_TO_CODE.items()}
_CODE_TO_VERDICT = {
    CODE_T: Verdict.MPC_CONSISTENT,
    CODE_U: Verdict.UNDETERMINED,
    CODE_F: Verdict.EXCLUDED,
}
_VERDICT_TO_CODE = {v: k for k, v in _CODE_TO_VERDICT.items()}

REASON_MISSING = "MISSING"
REASON_MISSING_CHANNEL = "MISSING_CHANNEL"
REASON_UNDEFINED = "UNDEFINED"
REASON_INCONCLUSIVE = "INCONCLUSIVE"
REASON_ABSENT = "ABSENT"
REASON_NO_NULL = "NO_NULL_CALIBRATION"
REASON_NO_SAMPLING_SE = "NO_SAMPLING_SE"
REASON_INVALID_ANCHORS = "INVALID_ANCHORS"
REASON_DEGENERATE_NULL = "DEGENERATE_NULL"
REASON_INVALID_SE = "INVALID_SE"
REASON_NULL_FAMILY_MISMATCH = "NULL_FAMILY_MISMATCH"
# Detail of ``UNDEFINED:NAS:<detail>``: NAS ``mode='capacity'`` needs a
# declared hub (``workspace_nodes``); without one the pipeline records the
# component as UNDEFINED instead of calling the estimator.
REASON_NO_DECLARED_WORKSPACE = "NO_DECLARED_WORKSPACE"
REASON_NOT_VALIDATED = "ESTIMATOR_NOT_VALIDATED"
REASON_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
REASON_BEARER_MISMATCH = "BEARER_MISMATCH"
REASON_PROTOCOL_MISMATCH = "PROTOCOL_MISMATCH"
REASON_SOURCE_INCOHERENT = "SOURCE_INCOHERENT"
SOURCE_UNTESTED = "UNTESTED"
# joint_dependence with too few surrogates to reach its level (reported as
# SOURCE_INCOHERENT:INSUFFICIENT_SURROGATES in a verdict).
REASON_INSUFFICIENT_SURROGATES = "INSUFFICIENT_SURROGATES"
GLOBAL_REASON_KINDS = (
    REASON_BEARER_MISMATCH,
    REASON_PROTOCOL_MISMATCH,
    REASON_SOURCE_INCOHERENT,
)
PRINCIPLE_REASON_KINDS = (
    REASON_MISSING,
    REASON_MISSING_CHANNEL,
    REASON_NO_NULL,
    REASON_NO_SAMPLING_SE,
    REASON_INVALID_ANCHORS,
    REASON_INCONCLUSIVE,
    REASON_UNDEFINED,
    REASON_NOT_IMPLEMENTED,
    REASON_NOT_VALIDATED,
    REASON_ABSENT,
)
# Channel reasons that become ``<KIND>:<P>`` codes (no detail).
_BARE_PRINCIPLE_REASONS = (
    REASON_INCONCLUSIVE,
    REASON_NO_NULL,
    REASON_NO_SAMPLING_SE,
    REASON_INVALID_ANCHORS,
)
REASON_SEPARATOR = ";"

# Default construct-scale cutoffs (z_j, delta_j) and one-sided error rate.
DEFAULT_CUTOFF = (0.25, 0.10)
DEFAULT_ALPHA = 0.05
# ``reference`` is the anchor rho on the estimate scale ("estimate", the
# default) or the reference excess rho - nu directly ("excess").
REFERENCE_SCALES = ("estimate", "excess")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _as_float(val):
    try:
        return float("nan") if val is None else float(val)
    except (TypeError, ValueError):
        return float("nan")


@functools.lru_cache(maxsize=64)
def _z_quantile(alpha):
    return float(NormalDist().inv_cdf(1.0 - float(alpha)))


def _one_sided_quantile(alpha, df=math.inf):
    """
    One-sided ``1 - alpha`` quantile: Student ``t`` with ``df`` degrees of
    freedom, or the normal quantile for an infinite (or non-finite) ``df``.
    """
    if not math.isfinite(df):
        return _z_quantile(alpha)
    from scipy.special import stdtrit

    return float(stdtrit(float(df), 1.0 - float(alpha)))


def _effective_df(se_df, s_samp, var_c):
    """
    Welch-Satterthwaite degrees of freedom of ``se_c`` when only its sampling
    part ``s_samp`` (with ``se_df`` degrees of freedom) is estimated from few
    replicates; the null and reference parts count as known. Infinite without
    a finite ``se_df`` or without a sampling part.
    """
    if not (math.isfinite(se_df) and s_samp > 0):
        return math.inf
    ratio = var_c / (s_samp * s_samp)
    return se_df * ratio * ratio


def _check_alpha(alpha):
    a = _as_float(alpha)
    if not (0.0 < a <= 0.5):
        raise ValueError("alpha must be in (0, 0.5]")
    return a


def normalize_cutoff(cutoff) -> tuple:
    """
    Construct-scale cutoffs ``(z, delta)`` of one principle: finite, with
    ``delta <= z`` (exclusivity: PRESENT and ABSENT cannot both hold).
    """
    try:
        z, delta = cutoff
    except (TypeError, ValueError):
        raise ValueError(
            f"a cutoff must be a pair (z, delta), got {cutoff!r}"
        ) from None
    z, delta = _as_float(z), _as_float(delta)
    if not (math.isfinite(z) and math.isfinite(delta)):
        raise ValueError("cutoffs z and delta must be finite")
    if delta > z:
        raise ValueError(f"cutoff delta ({delta}) must be <= z ({z})")
    return float(z), float(delta)


def _node_tuple(nodes, what="node set"):
    """Sorted tuple of distinct non-negative node indices (no masks)."""
    out = []
    for v in np.asarray(nodes, dtype=object).reshape(-1):
        if isinstance(v, (bool, np.bool_)):
            raise ValueError(f"{what} must list node indices, not a boolean mask")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{what} has a non-numeric node {v!r}") from None
        if not (math.isfinite(f) and f.is_integer() and f >= 0):
            raise ValueError(f"{what} has an invalid node index {v!r}")
        out.append(int(f))
    if not out:
        raise ValueError(f"{what} must not be empty")
    if len(set(out)) != len(out):
        raise ValueError(f"{what} has duplicate node indices")
    return tuple(sorted(out))


def _json_value(obj):
    """Deep copy of ``obj`` as plain JSON types (tuples -> lists)."""
    if isinstance(obj, Mapping):
        return {str(k): _json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_value(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_value(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"not JSON-serialisable: {type(obj)!r}")


def _freeze_json(obj):
    """Read-only copy of a JSON value (dicts -> mappingproxy, lists -> tuples)."""
    if isinstance(obj, Mapping):
        return MappingProxyType({str(k): _freeze_json(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze_json(v) for v in obj)
    return obj


def _canonical_json(payload) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def split_estimator(estimator):
    """``"compute_NAS:capacity@nas-2026.09"`` ->
    ``("compute_NAS:capacity", "nas-2026.09")``."""
    if estimator is None:
        return None, None
    name, sep, version = str(estimator).rpartition("@")
    if not sep:
        return str(estimator), None
    return name, (version or None)


def estimator_id(principle, mode, version):
    """Evidence estimator id ``compute_<P>:<mode>@<version>``."""
    return f"compute_{principle}:{mode}@{version}"


# --------------------------------------------------------------------------
# component evidence and status
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentEvidence:
    """
    One piece of null-anchored evidence for one principle through one channel.

    ``estimate`` is the estimator statistic ``m``; ``null_mean``/``null_sd`` are
    the moments of the declared null family (same estimator and configuration
    on ``n_null`` surrogates; ``n_null=0`` declares an analytic null mean
    without Monte-Carlo error). ``se`` is the sampling SE of ``estimate`` (e.g.
    a block bootstrap within the recording); 0 or NaN means unknown.
    ``se_df`` is the number of degrees of freedom of ``se`` when it comes from
    few replicates (delete-a-group jackknife with ``G`` groups: ``G - 1``;
    bootstrap with ``B`` valid replicates: ``B - 1``); the one-sided bounds
    then use the Student ``t`` quantile (Welch-Satterthwaite degrees of
    freedom of ``se_c``) instead of the normal quantile. None/NaN treats
    ``se`` as known (normal quantile).
    ``reference`` is the reference anchor ``rho`` on the estimate scale
    (``reference_scale="estimate"``) or the reference excess ``rho - nu``
    (``reference_scale="excess"``), with SE ``reference_se`` (None/NaN:
    unavailable). ``exact=True`` marks exact known-TPM computations, which need
    no sampling SE. ``bearer_id``/``protocol_id`` declare the system and the
    protocol the evidence was measured on, ``nodes`` the node indices of the
    bearer that fed the estimator (single-source constraint), ``estimator`` the
    estimator as ``compute_<P>:<mode>@<version>`` and ``regime`` per-item
    regime values for the applicability registry.

    Fields read by the TOST rule only: ``se_method`` (the SE method of the
    contract, :data:`SE_METHODS`; None keeps the plain meaning of
    ``se_df``), ``direction`` (a member of a directional principle, for
    example NAS ``receive``/``return``), ``p_ind`` (the IIM rank p-value
    against the independence null), and the cell of the concordance route:
    ``substrate``, ``observation_stage``, ``view`` and ``content_bearer``
    (the declared PDI bearer, for example ``non_workspace``).
    """

    principle: str
    estimate: float
    null_mean: float = float("nan")
    null_sd: float = float("nan")
    se: float = 0.0
    channel: str = "default"
    defined: bool = True
    reason: str | None = None
    reference: float | None = None
    bearer_id: str | None = None
    protocol_id: str | None = None
    substrate: str | None = None
    grain: str | None = None
    estimator: str | None = None
    null_family: str | None = None
    n_null: int = 0
    reference_se: float | None = None
    reference_scale: str = "estimate"
    exact: bool = False
    nodes: tuple | None = None
    regime: Mapping | None = field(default=None, hash=False)
    se_df: float | None = None
    se_method: str | None = None
    direction: str | None = None
    p_ind: float | None = None
    observation_stage: str | None = None
    view: str | None = None
    content_bearer: str | None = None

    def __post_init__(self):
        if self.reference_scale not in REFERENCE_SCALES:
            raise ValueError(
                f"reference_scale must be one of {REFERENCE_SCALES}, "
                f"got {self.reference_scale!r}"
            )
        if self.nodes is not None:
            object.__setattr__(self, "nodes", _node_tuple(self.nodes, "nodes"))
        for name in ("se_method", "direction", "observation_stage", "view",
                     "content_bearer"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, str) or not v.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        if self.direction is not None and not _NAME_RE.match(self.direction):
            raise ValueError(f"malformed direction {self.direction!r}")
        if self.p_ind is not None:
            p = _as_float(self.p_ind)
            if math.isfinite(p) and not (0.0 <= p <= 1.0):
                raise ValueError("p_ind must lie in [0, 1]")

    @property
    def estimator_version(self) -> str | None:
        return split_estimator(self.estimator)[1]


@dataclass(frozen=True)
class ComponentAssessment:
    """
    Construct-scale assessment of one piece of evidence: the status and its
    reason, ``c``, its SE ``se`` and the parts (``se_sampling``, ``se_null``
    = Monte-Carlo error of the null mean, ``se_reference``), the one-sided
    ``1 - alpha`` bounds ``lower``/``upper`` and the cutoffs. ``c`` is reported
    whenever the anchors are valid, also when the status is UNDEFINED for lack
    of a sampling SE. ``df`` is the (effective) degrees of freedom of ``se``
    (infinite: normal quantile) and ``quantile`` the one-sided ``1 - alpha``
    quantile of the bounds.
    """

    status: ComponentStatus
    reason: str | None
    c: float
    se: float
    se_sampling: float
    se_null: float
    se_reference: float
    lower: float
    upper: float
    cutoff_present: float
    cutoff_absent: float
    alpha: float
    df: float = math.inf
    quantile: float = float("nan")

    @property
    def margin_present(self) -> float:
        """``lower - z``: > 0 iff the component is PRESENT."""
        return self.lower - self.cutoff_present

    @property
    def margin_absent(self) -> float:
        """``delta - upper``: > 0 iff the component is ABSENT."""
        return self.cutoff_absent - self.upper

    def to_dict(self) -> dict:
        def _f(v):
            return None if not math.isfinite(v) else float(v)

        return {
            "status": self.status.value,
            "reason": self.reason,
            "c": _f(self.c),
            "se": _f(self.se),
            "se_sampling": _f(self.se_sampling),
            "se_null": _f(self.se_null),
            "se_reference": _f(self.se_reference),
            "lower": _f(self.lower),
            "upper": _f(self.upper),
            "cutoff_present": self.cutoff_present,
            "cutoff_absent": self.cutoff_absent,
            "margin_present": _f(self.margin_present),
            "margin_absent": _f(self.margin_absent),
            "alpha": self.alpha,
            "df": _f(self.df),
            "quantile": _f(self.quantile),
        }


def component_assessment(ev, *, cutoff=DEFAULT_CUTOFF, alpha=DEFAULT_ALPHA):
    """
    Construct-scale status of one piece of evidence.

    ``c = (estimate - null_mean) / (reference - null_mean)`` (or divided by the
    reference excess for ``reference_scale="excess"``). Its SE combines, by
    the delta method with independent parts: the sampling SE ``se`` of the
    estimate, the Monte-Carlo error ``null_sd / sqrt(n_null)`` of the null
    mean (derivative ``(c - 1) / (rho - nu)``, or ``-1 / (rho - nu)`` on the
    excess scale) and ``reference_se`` (derivative ``-c / (rho - nu)``).
    With ``z_a = z_{1-alpha}`` and ``cutoff = (z, delta)``:

    - PRESENT if ``c - z_a se_c > z``;
    - ABSENT if ``c + z_a se_c < delta`` (also for estimates credibly below
      the null);
    - UNDEFINED otherwise (``INCONCLUSIVE``).

    When ``ev.se_df`` is given (an SE from few replicates), ``z_a`` is the
    Student ``t`` quantile ``t_{1-alpha, nu}`` with the Welch-Satterthwaite
    degrees of freedom ``nu = se_df (se_c^2 / s_samp^2)^2`` (``s_samp`` = the
    sampling part ``se / (rho - nu)``; the null and reference parts count as
    known), so ``nu = se_df`` when the sampling SE dominates and ``nu`` grows
    without bound when it is negligible.

    UNDEFINED reasons, in order: not defined (``ev.reason`` or
    ``NOT_DEFINED``), ``NON_FINITE_ESTIMATE``, ``NO_NULL_CALIBRATION``,
    ``DEGENERATE_NULL`` (``n_null > 0`` without a finite ``null_sd >= 0``),
    ``INVALID_ANCHORS`` (reference not finite or not above the null),
    ``INVALID_SE`` (negative ``se`` or ``reference_se``, or ``se_df <= 0``)
    and ``NO_SAMPLING_SE`` (``se`` 0/NaN for an estimate that is not
    ``exact``). Returns a :class:`ComponentAssessment`.
    """
    z_cut, d_cut = normalize_cutoff(cutoff)
    alpha = _check_alpha(alpha)
    nan = float("nan")

    def _undefined(reason, c=nan):
        return ComponentAssessment(
            ComponentStatus.UNDEFINED, str(reason), c, nan, nan, nan, nan, nan, nan,
            z_cut, d_cut, alpha,
        )

    if not bool(ev.defined):
        return _undefined(ev.reason or "NOT_DEFINED")
    est = _as_float(ev.estimate)
    if not math.isfinite(est):
        return _undefined(ev.reason or "NON_FINITE_ESTIMATE")
    null_mean = _as_float(ev.null_mean)
    if not math.isfinite(null_mean):
        return _undefined(REASON_NO_NULL)
    n_null = _as_float(ev.n_null)
    if not (math.isfinite(n_null) and n_null >= 0 and n_null.is_integer()):
        return _undefined(REASON_DEGENERATE_NULL)
    se_nu = 0.0  # n_null = 0: analytic null mean
    if n_null > 0:
        null_sd = _as_float(ev.null_sd)
        if not (math.isfinite(null_sd) and null_sd >= 0):
            return _undefined(REASON_DEGENERATE_NULL)
        se_nu = null_sd / math.sqrt(n_null)
    ref = _as_float(ev.reference)
    excess_scale = ev.reference_scale == "excess"
    denom = ref if excess_scale else ref - null_mean
    if not (math.isfinite(ref) and math.isfinite(denom) and denom > 0):
        return _undefined(REASON_INVALID_ANCHORS)
    c = (est - null_mean) / denom
    ref_se = _as_float(ev.reference_se)
    if math.isfinite(ref_se) and ref_se < 0:
        return _undefined(REASON_INVALID_SE, c)
    if not math.isfinite(ref_se):
        ref_se = 0.0  # reference uncertainty unavailable
    se = _as_float(ev.se)
    if math.isfinite(se) and se < 0:
        return _undefined(REASON_INVALID_SE, c)
    se_df = _as_float(ev.se_df)
    if math.isfinite(se_df) and se_df <= 0:
        return _undefined(REASON_INVALID_SE, c)
    has_se = math.isfinite(se) and se > 0
    if not has_se:
        if not bool(ev.exact):
            return _undefined(REASON_NO_SAMPLING_SE, c)
        se = 0.0
    d_nu = (-1.0 / denom) if excess_scale else ((c - 1.0) / denom)
    d_ref = -c / denom
    s_samp = se / denom
    s_null = abs(d_nu) * se_nu
    s_ref = abs(d_ref) * ref_se
    var_c = s_samp * s_samp + s_null * s_null + s_ref * s_ref
    se_c = math.sqrt(var_c)
    df = _effective_df(se_df, s_samp, var_c)
    z_a = _one_sided_quantile(alpha, df)
    lower = c - z_a * se_c
    upper = c + z_a * se_c
    if lower > z_cut:
        status, reason = ComponentStatus.PRESENT, None
    elif upper < d_cut:
        status, reason = ComponentStatus.ABSENT, None
    else:
        status, reason = ComponentStatus.UNDEFINED, REASON_INCONCLUSIVE
    return ComponentAssessment(
        status, reason, c, se_c, s_samp, s_null, s_ref, lower, upper, z_cut, d_cut,
        alpha, df, z_a,
    )


def component_status(ev, *, cutoff=DEFAULT_CUTOFF, alpha=DEFAULT_ALPHA):
    """
    ``(ComponentStatus, margin, reason)`` of :func:`component_assessment`;
    ``margin = lower - z`` is the presence margin on the construct scale
    (> 0 iff PRESENT; NaN when the bound is undefined).
    """
    a = component_assessment(ev, cutoff=cutoff, alpha=alpha)
    return a.status, a.margin_present, a.reason


def component_status_array(
    estimate,
    null_mean,
    null_sd,
    se=0.0,
    *,
    n_null=0,
    reference=np.nan,
    reference_se=0.0,
    exact=False,
    reference_scale="estimate",
    cutoff=DEFAULT_CUTOFF,
    alpha=DEFAULT_ALPHA,
    se_df=np.nan,
):
    """
    Vectorised numeric core of :func:`component_assessment` (same
    inequalities, same operation order, same Student-``t`` quantile for a
    finite ``se_df``). Returns ``(codes, margins)`` with codes in
    {1 (PRESENT), 0 (UNDEFINED), -1 (ABSENT)} and the presence margins
    ``lower - z``; invalid inputs give code 0 and a NaN margin.
    """
    z_cut, d_cut = normalize_cutoff(cutoff)
    alpha = _check_alpha(alpha)
    if reference_scale not in REFERENCE_SCALES:
        raise ValueError(f"reference_scale must be one of {REFERENCE_SCALES}")
    est, nm, nsd, s, nn, ref, rse, sdf = np.broadcast_arrays(
        *(
            np.asarray(np.nan if v is None else v, dtype=float)
            for v in (estimate, null_mean, null_sd, se, n_null, reference,
                      reference_se, se_df)
        )
    )
    ex = np.broadcast_to(np.asarray(exact, dtype=bool), est.shape)
    z_a = _z_quantile(alpha)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        nn_ok = np.isfinite(nn) & (nn >= 0) & (nn == np.floor(nn))
        null_ok = np.isfinite(nm) & nn_ok & (
            (nn == 0) | (np.isfinite(nsd) & (nsd >= 0))
        )
        se_nu = np.where(nn > 0, nsd / np.sqrt(np.where(nn > 0, nn, 1.0)), 0.0)
        denom = ref if reference_scale == "excess" else ref - nm
        anchors_ok = np.isfinite(ref) & np.isfinite(denom) & (denom > 0)
        rse_ok = ~(np.isfinite(rse) & (rse < 0))
        rse_eff = np.where(np.isfinite(rse), rse, 0.0)
        se_sign_ok = ~(np.isfinite(s) & (s < 0)) & ~(np.isfinite(sdf) & (sdf <= 0))
        has_se = np.isfinite(s) & (s > 0)
        s_eff = np.where(has_se, s, 0.0)
        valid = (
            np.isfinite(est) & null_ok & anchors_ok & rse_ok & se_sign_ok
            & (has_se | ex)
        )
        c = (est - nm) / denom
        if reference_scale == "excess":
            d_nu = -1.0 / denom
        else:
            d_nu = (c - 1.0) / denom
        d_ref = -c / denom
        s_samp = s_eff / denom
        s_null = np.abs(d_nu) * se_nu
        s_ref = np.abs(d_ref) * rse_eff
        var_c = s_samp * s_samp + s_null * s_null + s_ref * s_ref
        se_c = np.sqrt(var_c)
        # Student-t quantile where the SE has finite degrees of freedom
        t_rows = valid & np.isfinite(sdf) & (s_samp > 0)
        q = np.full(est.shape, z_a)
        if np.any(t_rows):
            from scipy.special import stdtrit

            ratio = var_c[t_rows] / (s_samp[t_rows] * s_samp[t_rows])
            df_eff = sdf[t_rows] * ratio * ratio
            fin = np.isfinite(df_eff)
            q_t = np.full(df_eff.shape, z_a)
            q_t[fin] = stdtrit(df_eff[fin], 1.0 - float(alpha))
            q[t_rows] = q_t
        lower = c - q * se_c
        upper = c + q * se_c
        present = valid & (lower > z_cut)
        absent = valid & ~present & (upper < d_cut)
        margin = np.where(valid, lower - z_cut, np.nan)
    codes = np.where(present, CODE_T, np.where(absent, CODE_F, CODE_U)).astype(np.int8)
    return codes, margin


# --------------------------------------------------------------------------
# strong Kleene logic
# --------------------------------------------------------------------------
def _to_code(value):
    if isinstance(value, (bool, np.bool_)):
        return CODE_T if value else CODE_F
    if value is None:
        return CODE_U
    if isinstance(value, (int, np.integer)) and int(value) in (CODE_T, CODE_U, CODE_F):
        return int(value)
    txt = value.value if isinstance(value, Enum) else str(value)
    for enum in (ComponentStatus, Verdict):
        try:
            member = enum(txt)
        except ValueError:
            continue
        return _STATUS_TO_CODE[member] if enum is ComponentStatus else (
            _VERDICT_TO_CODE[member]
        )
    raise ValueError(f"not a three-valued status: {value!r}")


def kleene_and(statuses):
    """Strong-Kleene conjunction (F dominates, then U; empty -> PRESENT)."""
    return _CODE_TO_STATUS[min((_to_code(s) for s in statuses), default=CODE_T)]


def kleene_or(statuses):
    """Strong-Kleene disjunction (T dominates, then U; empty -> ABSENT)."""
    return _CODE_TO_STATUS[max((_to_code(s) for s in statuses), default=CODE_F)]


def kleene_verdict_codes(principle_codes, necessity_mask=None):
    """
    Vectorised Kleene AND over the last axis (principles), restricted to the
    necessity mask (principles outside it count as T, the AND identity).
    Returns verdict codes (1 MPC_CONSISTENT, 0 UNDETERMINED, -1 EXCLUDED).
    """
    codes = np.asarray(principle_codes, dtype=np.int8)
    if necessity_mask is not None:
        codes = np.where(np.asarray(necessity_mask, dtype=bool), codes, CODE_T)
    return codes.min(axis=-1)


def kleene_or_codes(channel_codes, has_channel=None):
    """
    Vectorised Kleene OR over the last axis (channels). ``has_channel`` marks
    real channels (padding counts as F, the OR identity); a principle without
    any channel is MISSING and therefore U.
    """
    codes = np.asarray(channel_codes, dtype=np.int8)
    if has_channel is None:
        return codes.max(axis=-1)
    has = np.asarray(has_channel, dtype=bool)
    out = np.where(has, codes, CODE_F).max(axis=-1)
    return np.where(has.any(axis=-1), out, CODE_U).astype(np.int8)


# --------------------------------------------------------------------------
# TOST rule: constants
# --------------------------------------------------------------------------
STATUS_RULE_VERSION = "tost"
STATUS_RULE_VERSIONS = (STATUS_RULE_VERSION,)
ABSENT_TESTS = ("tost",)
N_DECL = len(PRINCIPLES)
# alpha_A = alpha / |N_decl| with the five declared principles.
DEFAULT_ALPHA_ABSENT = 0.01
STATUS_RULE_KEYS = (
    "version", "alpha", "alpha_absent", "absent_test", "null_violation",
    "present_reachability_flag",
)
# Directions of directional principles: NAS receive (c_R) and return (c_B).
NAS_DIRECTIONS = ("receive", "return")
DIRECTION_FIELDS = {"NAS": {"receive": "c_R", "return": "c_B"}}
# A TOST protocol may also declare the reference ``pending`` (no anchor yet:
# every component is UNDEFINED(INVALID_ANCHORS); templates only).
REFERENCE_PENDING = "pending"
# Status routes of an assessment.
ROUTE_TOST, ROUTE_EXACT, ROUTE_CONCORDANCE, ROUTE_DIRECTIONAL = (
    "tost", "exact", "concordance", "directional")
# Values of a registry admission per status direction.
ADMISSION_VALUES = ("yes", "no", "vacuous", "not_observable")
SCALES = ("construct", "amplitude")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+=@-]*$")
_REL_TOL = 1e-12


# --------------------------------------------------------------------------
# SE-method contract
# --------------------------------------------------------------------------
DF_FIXED = "fixed"
DF_N_NULL_MINUS_1 = "n_null_minus_1"
DF_NONE = "none"


@dataclass(frozen=True)
class SeMethod:
    """One SE method of the contract: the principles that may use it, the
    rule that fixes ``se_df`` and whether it gives a sampling SE."""

    name: str
    principles: Tuple[str, ...]
    df_rule: str
    df_values: Tuple[float, ...] = ()
    sampling_se: bool = True
    meaning: str = ""


SE_METHOD_CONCORDANT = "concordant"
SE_METHOD_EXACT = "exact"
_SE_METHOD_LIST = (
    SeMethod("jackknife_contiguous_10", ("NAS", "IIM", "PDI"), DF_FIXED, (9.0,),
             meaning="delete-a-group jackknife over 10 contiguous blocks "
                     "(se_df = G - 1 = 9)"),
    SeMethod("jackknife_contiguous_20", ("NAS",), DF_FIXED, (19.0,),
             meaning="delete-a-group jackknife over 20 contiguous blocks "
                     "(se_df = 19)"),
    SeMethod("jackknife_interleaved_10", ("NAS",), DF_FIXED, (9.0,),
             meaning="delete-a-group jackknife over 10 interleaved block groups "
                     "(se_df = 9)"),
    SeMethod("jackknife_interleaved_20", ("NAS",), DF_FIXED, (19.0,),
             meaning="delete-a-group jackknife over 20 interleaved block groups "
                     "(se_df = 19)"),
    SeMethod("circular_block_bootstrap_10pct_B50", ("IIM",), DF_FIXED, (12.0, 9.0),
             meaning="circular block bootstrap, blocks of 10 % of the run, B = 50; "
                     "se_df = 12 (the block-count approximation 12.5 rounded "
                     "down), 9 if lowered at calibration; never B - 1"),
    SeMethod("shift_null_sd", ("RAM",), DF_N_NULL_MINUS_1,
             meaning="SD of the exact trial-shift null (se_df = n_null - 1)"),
    SeMethod("jackknife_trials_10", ("RAM",), DF_FIXED, (9.0,),
             meaning="delete-a-group jackknife over 10 trial groups (se_df = 9)"),
    SeMethod("jackknife_pairs_10", ("SRPI",), DF_FIXED, (9.0,),
             meaning="delete-a-group jackknife over 10 pair groups (se_df = 9)"),
    SeMethod(SE_METHOD_CONCORDANT, ("PDI",), DF_NONE, sampling_se=False,
             meaning="all resampled counts agree (se = 0); decided only through "
                     "an admitted concordance route, never exact"),
    SeMethod(SE_METHOD_EXACT, ("IIM",), DF_NONE, sampling_se=False,
             meaning="deterministic known-TPM value (exact = True, se = 0)"),
)
SE_METHODS: Mapping[str, SeMethod] = MappingProxyType(
    {m.name: m for m in _SE_METHOD_LIST})

# Scale type of each estimator version: a quadratic (information-like)
# statistic makes delta correspond to about sqrt(delta) of the reference
# amplitude; such statistics are re-judged on the amplitude scale.
SCALE_QUADRATIC, SCALE_FIRST_ORDER, SCALE_COUNT = "quadratic", "first_order", "count"
SCALE_TYPES = MappingProxyType({
    "nas-declared-2026.10": SCALE_QUADRATIC,
    "nas-2026.09": SCALE_QUADRATIC,
    "iim-macro-2026.10": SCALE_QUADRATIC,
    "iim-2026.09": SCALE_QUADRATIC,
    "srpi-2026.09": SCALE_FIRST_ORDER,
    "ram-pe-2026.10": SCALE_FIRST_ORDER,
    "pdi-declared-2026.10": SCALE_COUNT,
    "pdi-2026.09": SCALE_COUNT,
})


def is_quadratic(estimator_version) -> bool:
    """True iff the estimator version's statistic is quadratic in a weak
    effect amplitude (unknown versions are not)."""
    return SCALE_TYPES.get(str(estimator_version)) == SCALE_QUADRATIC


def amplitude_cutoff(cutoff) -> tuple:
    """The cutoffs on ``c`` that judge a quadratic statistic on the amplitude
    scale ``a = sign(c) sqrt(|c|)``: ``a`` is monotone in ``c``, so the
    interval statements ``g(c - q se) > z`` and ``-delta < g(c -+ q se) <
    delta`` hold iff the same statements hold for ``c`` with ``z**2`` and
    ``delta**2``."""
    z, delta = normalize_cutoff(cutoff)
    if z < 0 or delta < 0:
        raise ValueError("amplitude cutoffs need z >= 0 and delta >= 0")
    return float(z * z), float(delta * delta)


def amplitude_value(c):
    """``sign(c) sqrt(|c|)`` element-wise (NaN stays NaN)."""
    arr = np.asarray(c, dtype=float)
    out = np.sign(arr) * np.sqrt(np.abs(arr))
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------
# status rule block
# --------------------------------------------------------------------------
def _strict_bool(value, what):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} must be true or false, got {value!r}")
    return bool(value)


@dataclass(frozen=True)
class StatusRule:
    """The hash-covered ``status_rule`` block of a TOST protocol."""

    version: str = STATUS_RULE_VERSION
    alpha: float = DEFAULT_ALPHA
    alpha_absent: float = DEFAULT_ALPHA_ABSENT
    absent_test: str = "tost"
    null_violation: bool = True
    present_reachability_flag: bool = True

    def __post_init__(self):
        put = functools.partial(object.__setattr__, self)
        if self.version not in STATUS_RULE_VERSIONS:
            raise ValueError(
                f"status_rule.version must be one of {STATUS_RULE_VERSIONS}, "
                f"got {self.version!r}")
        put("alpha", _check_alpha(self.alpha))
        a_abs = _as_float(self.alpha_absent)
        if not (0.0 < a_abs <= self.alpha):
            raise ValueError("status_rule.alpha_absent must lie in (0, alpha]")
        put("alpha_absent", float(a_abs))
        if self.absent_test not in ABSENT_TESTS:
            raise ValueError(f"status_rule.absent_test must be one of {ABSENT_TESTS}")
        put("null_violation", _strict_bool(self.null_violation,
                                           "status_rule.null_violation"))
        put("present_reachability_flag", _strict_bool(
            self.present_reachability_flag, "status_rule.present_reachability_flag"))

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in STATUS_RULE_KEYS}

    @classmethod
    def from_dict(cls, payload) -> "StatusRule":
        if isinstance(payload, StatusRule):
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError("status_rule must be an object")
        unknown = sorted(set(payload) - set(STATUS_RULE_KEYS))
        missing = sorted(set(STATUS_RULE_KEYS) - set(payload))
        if unknown:
            raise ValueError(f"status_rule has unknown keys {unknown}")
        if missing:
            raise ValueError(f"status_rule misses keys {missing}")
        return cls(**dict(payload))


DEFAULT_STATUS_RULE = StatusRule()


def member_levels(rule: StatusRule, n_channels=1, n_directions=1) -> tuple:
    """``(alpha_present, alpha_absent)`` of one member test: PRESENT through
    one of ``n_channels`` OR-combined channels is tested at
    ``alpha / n_channels``; ABSENT through one of ``n_directions``
    union-combined directions at ``alpha_A / n_directions``."""
    k_c, k_d = int(n_channels), int(n_directions)
    if k_c < 1 or k_d < 1:
        raise ValueError("member counts must be >= 1")
    return rule.alpha / k_c, rule.alpha_absent / k_d


def _coerce_rule(rule) -> StatusRule:
    if rule is None:
        return DEFAULT_STATUS_RULE
    if isinstance(rule, StatusRule):
        return rule
    return StatusRule.from_dict(rule)


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------
PROTOCOL_SCHEMA = "impact-mpc-protocol/1"
# single_source: one bearer_id, and joint dependence above null when the
# components come from different node sets; same_bearer: one bearer_id only;
# none: no source checks (exploratory use).
SOURCE_RULES = ("single_source", "same_bearer", "none")
REFERENCE_KINDS = ("cohort_high_state", "external")
DEFAULT_REFERENCE = {"kind": "cohort_high_state", "session": "awake"}
_PROTOCOL_FIELDS = (
    "necessity_set",
    "channels",
    "cutoffs",
    "alpha",
    "null_families",
    "reference",
    "source_rule",
    "estimators",
    "bearer_nodes",
    "name",
)
# The fields of a TOST protocol beyond those of every protocol.
_TOST_FIELDS = (
    "status_rule", "declared_necessity_set", "directions", "rank_gates",
    "estimator_versions", "se_methods", "shared_inputs_declaration",
    "concordance_route", "anchors", "precision",
)
# Declarations of the bench and their identifiability level.
DECLARATION_LEVELS = MappingProxyType({
    "R": "complete", "H": "partial", "P": "partial", "Q10": "partial",
    "Q25": "partial", "J": "partial", "none": "none",
})
_CELL_KEYS = ("principle", "substrate", "observation_stage", "view", "bearer")
_ROUTE_KEYS = _CELL_KEYS + ("absent", "present", "provisional", "battery")


def normalize_necessity_set(necessity_set) -> tuple:
    """
    Canonical necessity set: a non-empty subset of :data:`PRINCIPLES` in
    canonical order. Accepts a sequence or a comma-separated string; ``None``
    means all five principles.
    """
    if necessity_set is None:
        return PRINCIPLES
    if isinstance(necessity_set, str):
        items = [s.strip() for s in necessity_set.split(",")]
    else:
        items = [str(s).strip() for s in necessity_set]
    items = [s.upper() for s in items if s]
    unknown = sorted(set(items) - set(PRINCIPLES))
    if unknown:
        raise ValueError(f"unknown principle(s) in necessity set: {unknown}")
    if not items:
        raise ValueError("the necessity set must not be empty")
    return tuple(p for p in PRINCIPLES if p in set(items))


def _principle_key(key, what):
    p = str(key).strip().upper()
    if p not in PRINCIPLES:
        raise ValueError(f"{what}: unknown principle {key!r}")
    return p


def _normalize_reference(reference):
    ref = dict(DEFAULT_REFERENCE if reference is None else reference)
    kind = str(ref.get("kind", "")).strip()
    if kind not in REFERENCE_KINDS:
        raise ValueError(f"reference kind must be one of {REFERENCE_KINDS}")
    if kind == "cohort_high_state":
        unknown = sorted(set(ref) - {"kind", "session"})
        if unknown:
            raise ValueError(f"cohort reference has unknown keys {unknown}")
        session = str(ref.get("session", DEFAULT_REFERENCE["session"])).strip()
        if not session:
            raise ValueError("cohort reference needs a session name")
        return {"kind": kind, "session": session}
    unknown = sorted(set(ref) - {"kind", "values", "se", "scale", "source"})
    if unknown:
        raise ValueError(f"external reference has unknown keys {unknown}")
    scale = str(ref.get("scale", "excess"))
    if scale not in REFERENCE_SCALES:
        raise ValueError(f"reference scale must be one of {REFERENCE_SCALES}")
    out = {"kind": kind, "scale": scale, "values": {}, "se": {}}
    for part in ("values", "se"):
        raw = ref.get(part) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"external reference '{part}' must be an object")
        for k, v in raw.items():
            p, _, ch = str(k).partition(":")
            key = _principle_key(p, f"reference {part}") + (f":{ch}" if ch else "")
            val = _as_float(v)
            if not math.isfinite(val):
                raise ValueError(f"reference {part} for {k} must be finite")
            if part == "se" and val < 0:
                raise ValueError(f"reference se for {k} must be >= 0")
            out[part][key] = val
    if not out["values"]:
        raise ValueError("an external reference needs 'values'")
    if ref.get("source") is not None:
        out["source"] = str(ref["source"])
    return out


def _normalize_shared_inputs(value):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("shared_inputs_declaration must be an object or null")
    from impact_pipeline.bench import records as _records

    unknown = sorted(set(value) - {"id", "shared_inputs", "inputs"})
    if unknown:
        raise ValueError(f"shared_inputs_declaration has unknown keys {unknown}")
    did = value.get("id")
    if did not in _records.KNOWN_DECLARATIONS:
        raise ValueError(f"shared_inputs_declaration.id must be one of "
                         f"{_records.KNOWN_DECLARATIONS}, got {did!r}")
    level = value.get("shared_inputs")
    if level not in _records.SHARED_INPUTS:
        raise ValueError(f"shared_inputs_declaration.shared_inputs must be one of "
                         f"{_records.SHARED_INPUTS}")
    want = DECLARATION_LEVELS.get(did)
    if want is not None and level != want:
        raise ValueError(f"declaration {did!r} is {want!r}, not {level!r}")
    out = {"id": did, "shared_inputs": level}
    if "inputs" in value:
        inputs = value["inputs"]
        if isinstance(inputs, str) or not isinstance(inputs, Sequence):
            raise ValueError("shared_inputs_declaration.inputs must be a list")
        names = [str(s).strip() for s in inputs]
        if any(not s for s in names) or len(set(names)) != len(names):
            raise ValueError("declared inputs must be distinct non-empty names")
        out["inputs"] = sorted(names)
    return out


def _normalize_route(entries, se_methods):
    if entries is None:
        return ()
    if isinstance(entries, (str, Mapping)) or not isinstance(entries, Sequence):
        raise ValueError("concordance_route must be a list of cells")
    from impact_pipeline.bench import records as _records

    cells, seen = [], set()
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ValueError("a concordance-route cell must be an object")
        unknown = sorted(set(raw) - set(_ROUTE_KEYS))
        missing = sorted(set(_CELL_KEYS + ("absent", "present")) - set(raw))
        if unknown or missing:
            raise ValueError(f"concordance-route cell: unknown {unknown}, "
                             f"missing {missing}")
        p = _principle_key(raw["principle"], "concordance_route")
        if p not in SE_METHODS[SE_METHOD_CONCORDANT].principles:
            raise ValueError(f"the concordance route is not defined for {p}")
        allowed = se_methods.get(p)
        if allowed is not None and SE_METHOD_CONCORDANT not in allowed:
            raise ValueError(f"{p}: the protocol's se_methods do not include "
                             f"{SE_METHOD_CONCORDANT!r}")
        cell = {"principle": p}
        for k in ("substrate", "view", "bearer"):
            v = raw[k]
            if not isinstance(v, str) or not _TOKEN_RE.match(v):
                raise ValueError(f"concordance-route {k} must be a token, got {v!r}")
            cell[k] = v
        if raw["observation_stage"] not in _records.OBSERVATION_STAGES:
            raise ValueError(
                f"observation_stage must be one of {_records.OBSERVATION_STAGES}")
        cell["observation_stage"] = raw["observation_stage"]
        cell["absent"] = _strict_bool(raw["absent"], "concordance_route.absent")
        cell["present"] = _strict_bool(raw["present"], "concordance_route.present")
        if not (cell["absent"] or cell["present"]):
            raise ValueError("a concordance-route cell admits at least one direction")
        cell["provisional"] = _strict_bool(raw.get("provisional", False),
                                           "concordance_route.provisional")
        battery = raw.get("battery", {})
        if not isinstance(battery, Mapping):
            raise ValueError("concordance_route.battery must be an object")
        cell["battery"] = _json_value(battery)
        key = tuple(cell[k] for k in _CELL_KEYS)
        if key in seen:
            raise ValueError(f"duplicate concordance-route cell {key}")
        seen.add(key)
        cells.append(cell)
    cells.sort(key=lambda d: tuple(d[k] for k in _CELL_KEYS))
    return tuple(_freeze_json(c) for c in cells)


@dataclass(frozen=True, eq=False)
class Protocol:
    """
    Declared measurement protocol of an MPC verdict.

    - ``necessity_set``: the principles ``N`` whose conjunction the verdict
      tests (default all five);
    - ``channels``: per principle the declared evidence channels; a declared
      channel without an evidence item is UNDEFINED (``MISSING_CHANNEL``);
      principles without a declaration use the channels of their evidence;
    - ``cutoffs``: per principle the construct-scale cutoffs ``(z, delta)``
      (default :data:`DEFAULT_CUTOFF`); ``alpha``: one-sided error rate;
    - ``null_families``: per principle the declared null family; evidence
      whose ``null_family`` differs is UNDEFINED (``NULL_FAMILY_MISMATCH``);
    - ``reference``: the reference anchor, ``{"kind": "cohort_high_state",
      "session": ...}`` or ``{"kind": "external", "values": {P: ..},
      "se": {P: ..}, "scale": "excess"|"estimate"}`` (keys ``P`` or
      ``P:channel``). The high-state runs are part of a cohort reference
      (their ``c`` averages 1 by construction), so their verdicts are not
      independent tests of necessity; such tests need an external reference;
    - ``source_rule``: one of :data:`SOURCE_RULES` (single-source constraint);
    - ``estimators``: per principle estimator keyword options (modes, e.g.
      ``{"NAS": {"mode": "capacity"}, "RAM": {"update": "prediction_error"}}``);
    - ``bearer_nodes``: per principle the bearer node indices that feed the
      estimator (default: all nodes);
    - ``name``: a label.

    ``status_rule`` selects the status rule: None (the default) is the
    interval rule; a :class:`StatusRule` block (or its dict, or ``"tost"``
    for the default block at the protocol's ``alpha``) is the TOST rule. Its
    ``alpha`` equals the protocol's and its ``alpha_absent`` equals
    ``alpha / |declared_necessity_set|`` (0.01). Only a TOST protocol
    declares the further fields:

    - ``declared_necessity_set``: N_decl, always the five principles (fixed
      core), so ``alpha_A = alpha / 5`` in every protocol; ``necessity_set``
      is the anchored set N_anch over which the verdict is taken (a subset
      of N_decl);
    - ``directions``: per directional principle its members
      (``{"NAS": ["receive", "return"]}``); reference keys ``P:<direction>``
      or ``P:<channel>:<direction>`` hold the per-direction anchors;
    - ``rank_gates``: per principle the level of a rank gate on PRESENT
      (``{"IIM": 0.05}``);
    - ``estimator_versions``: per principle the estimator version the
      protocol dispatches (evidence of another version is
      ``ESTIMATOR_NOT_VALIDATED:not_admitted``);
    - ``se_methods``: per principle the admitted SE methods;
    - ``shared_inputs_declaration``: ``{"id", "shared_inputs"[, "inputs"]}``;
    - ``concordance_route``: the admitted cells of the PDI concordance route;
    - ``anchors``: the anchor block (validity, specificity, N_anch, fallback);
    - ``precision``: the testability table (hash-covered precision block).

    The reference of a TOST protocol may also be ``{"kind": "pending"}`` (its
    default: no anchor, every component UNDEFINED(INVALID_ANCHORS)); with an
    ``external`` reference the anchors of the protocol replace those of the
    evidence items, so every status is on the protocol's fixed scale.

    The canonical JSON (:meth:`to_dict`, all cutoffs materialised, the TOST
    fields only for a TOST protocol) is hashed with SHA-256 (:attr:`hash`);
    :attr:`protocol_id` is what evidence records as ``protocol_id``.
    """

    necessity_set: tuple = PRINCIPLES
    channels: Mapping = field(default_factory=dict)
    cutoffs: Mapping = field(default_factory=dict)
    alpha: float = DEFAULT_ALPHA
    null_families: Mapping = field(default_factory=dict)
    reference: Mapping | None = None
    source_rule: str = "single_source"
    estimators: Mapping = field(default_factory=dict)
    bearer_nodes: Mapping = field(default_factory=dict)
    name: str | None = None
    status_rule: object = None
    declared_necessity_set: tuple = PRINCIPLES
    directions: Mapping = field(default_factory=dict)
    rank_gates: Mapping = field(default_factory=dict)
    estimator_versions: Mapping = field(default_factory=dict)
    se_methods: Mapping = field(default_factory=dict)
    shared_inputs_declaration: Mapping | None = None
    concordance_route: tuple = ()
    anchors: Mapping | None = None
    precision: Mapping | None = None

    def __post_init__(self):
        put = functools.partial(object.__setattr__, self)
        tost = self.status_rule is not None
        if not tost:
            self._init_interval(put)
        ref, pending = self.reference, False
        if tost:
            ref = dict(ref) if ref is not None else {"kind": REFERENCE_PENDING}
            pending = str(ref.get("kind", "")).strip() == REFERENCE_PENDING
            if pending:
                unknown = sorted(set(ref) - {"kind", "note"})
                if unknown:
                    raise ValueError(f"pending reference has unknown keys {unknown}")
        put("necessity_set", normalize_necessity_set(self.necessity_set))
        channels = {}
        for k, v in dict(self.channels or {}).items():
            p = _principle_key(k, "channels")
            items = [v] if isinstance(v, str) else list(v or [])
            names = tuple(str(c).strip() for c in items)
            if not names or any(not c for c in names) or len(set(names)) != len(names):
                raise ValueError(f"channels for {p} must be distinct non-empty names")
            channels[p] = names
        put("channels", MappingProxyType(channels))
        cutoffs = {p: DEFAULT_CUTOFF for p in PRINCIPLES}
        for k, v in dict(self.cutoffs or {}).items():
            cutoffs[_principle_key(k, "cutoffs")] = normalize_cutoff(v)
        put("cutoffs", MappingProxyType(cutoffs))
        put("alpha", _check_alpha(self.alpha))
        families = {}
        for k, v in dict(self.null_families or {}).items():
            if v is None or not str(v).strip():
                raise ValueError(f"null family for {k} must be a non-empty name")
            families[_principle_key(k, "null_families")] = str(v).strip()
        put("null_families", MappingProxyType(families))
        # read-only all the way down: the hash must not change after creation
        put("reference", _freeze_json(_normalize_reference(None if pending else ref)))
        if self.source_rule not in SOURCE_RULES:
            raise ValueError(f"source_rule must be one of {SOURCE_RULES}")
        estimators = {}
        for k, v in dict(self.estimators or {}).items():
            if not isinstance(v, Mapping):
                raise ValueError(f"estimator options for {k} must be an object")
            estimators[_principle_key(k, "estimators")] = _freeze_json(_json_value(v))
        put("estimators", MappingProxyType(estimators))
        bearers = {}
        for k, v in dict(self.bearer_nodes or {}).items():
            p = _principle_key(k, "bearer_nodes")
            bearers[p] = None if v is None else _node_tuple(v, f"bearer_nodes[{p}]")
        put("bearer_nodes", MappingProxyType(bearers))
        if self.name is not None:
            put("name", str(self.name))
        if pending:
            pend = {"kind": REFERENCE_PENDING}
            if ref.get("note") is not None:
                pend["note"] = str(ref["note"])
            put("reference", _freeze_json(pend))
        if tost:
            self._init_tost(put)

    def _init_interval(self, put):
        given = [
            name for name, unset in (
                ("declared_necessity_set",
                 normalize_necessity_set(self.declared_necessity_set) == PRINCIPLES),
                ("directions", not self.directions),
                ("rank_gates", not self.rank_gates),
                ("estimator_versions", not self.estimator_versions),
                ("se_methods", not self.se_methods),
                ("shared_inputs_declaration", self.shared_inputs_declaration is None),
                ("concordance_route", not self.concordance_route),
                ("anchors", self.anchors is None),
                ("precision", self.precision is None),
            ) if not unset
        ]
        if given:
            raise ValueError(f"protocol fields {given} need a status_rule block")
        put("declared_necessity_set", PRINCIPLES)
        for name in ("directions", "rank_gates", "estimator_versions", "se_methods"):
            put(name, MappingProxyType({}))
        put("concordance_route", ())

    def _init_tost(self, put):
        declared = normalize_necessity_set(self.declared_necessity_set)
        if declared != PRINCIPLES:
            # N_decl is the fixed core, so alpha_A = alpha / 5 = 0.01 in every
            # protocol and statuses transfer between protocols without
            # re-judging
            raise ValueError(
                f"declared_necessity_set must be the five principles {PRINCIPLES}, "
                f"got {declared}; restrict necessity_set (N_anch) instead")
        put("declared_necessity_set", declared)
        if not set(self.necessity_set) <= set(declared):
            raise ValueError("necessity_set must be a subset of declared_necessity_set")
        rule = self.status_rule
        if isinstance(rule, str):
            if rule != STATUS_RULE_VERSION:
                raise ValueError(f"status_rule must be a block or {STATUS_RULE_VERSION!r}"
                                 f", got {rule!r}")
            rule = StatusRule(alpha=self.alpha, alpha_absent=self.alpha / len(declared))
        rule = StatusRule.from_dict(rule)
        if rule.alpha != self.alpha:
            raise ValueError("status_rule.alpha must equal the protocol's alpha")
        want = self.alpha / len(declared)
        if not math.isclose(rule.alpha_absent, want, rel_tol=_REL_TOL, abs_tol=0.0):
            raise ValueError(
                f"status_rule.alpha_absent must be alpha / |declared_necessity_set| "
                f"= {want!r}, got {rule.alpha_absent!r}")
        # the verdict needs alpha_absent x |necessity_set| <= alpha; the two
        # checks above imply it, and it is kept as a guard
        if rule.alpha_absent * len(self.necessity_set) > rule.alpha * (1 + _REL_TOL):
            raise ValueError("alpha_absent x |necessity_set| exceeds alpha")
        put("status_rule", rule)
        directions = {}
        for k, v in dict(self.directions or {}).items():
            p = _principle_key(k, "directions")
            names = [v] if isinstance(v, str) else list(v or [])
            names = tuple(str(n).strip() for n in names)
            if len(names) < 2 or len(set(names)) != len(names) or any(
                    not _NAME_RE.match(n) for n in names):
                raise ValueError(
                    f"directions of {p} must be two or more distinct lower-case names")
            chans = set(self.channels.get(p) or ("default",))
            if chans & set(names):
                raise ValueError(
                    f"directions of {p} must differ from its channel names")
            directions[p] = names
        put("directions", MappingProxyType(directions))
        gates = {}
        for k, v in dict(self.rank_gates or {}).items():
            p = _principle_key(k, "rank_gates")
            lvl = _as_float(v)
            if not (0.0 < lvl <= 0.5):
                raise ValueError(f"rank gate of {p} must lie in (0, 0.5]")
            gates[p] = float(lvl)
        put("rank_gates", MappingProxyType(gates))
        versions = {}
        for k, v in dict(self.estimator_versions or {}).items():
            p = _principle_key(k, "estimator_versions")
            allowed = set(ESTIMATOR_VERSIONS[p])
            if v not in allowed:
                raise ValueError(f"estimator version of {p} must be one of "
                                 f"{sorted(allowed)}, got {v!r}")
            versions[p] = str(v)
        put("estimator_versions", MappingProxyType(versions))
        methods = {}
        for k, v in dict(self.se_methods or {}).items():
            p = _principle_key(k, "se_methods")
            names = [v] if isinstance(v, str) else list(v or [])
            names = tuple(sorted(set(str(n) for n in names)))
            if not names:
                raise ValueError(f"se_methods of {p} must not be empty")
            for n in names:
                m = SE_METHODS.get(n)
                if m is None or p not in m.principles:
                    raise ValueError(f"SE method {n!r} is not defined for {p}")
            methods[p] = names
        put("se_methods", MappingProxyType(methods))
        shared = self.shared_inputs_declaration
        put("shared_inputs_declaration", None if shared is None
            else _freeze_json(_normalize_shared_inputs(shared)))
        put("concordance_route", _normalize_route(self.concordance_route, methods))
        self._check_reference_keys()
        from impact_pipeline import testability as T

        put("anchors", None if self.anchors is None else _freeze_json(
            T.validate_anchors_block(self.anchors, declared, self.necessity_set)))
        put("precision", None if self.precision is None else _freeze_json(
            T.validate_precision_block(self.precision)))

    def _check_reference_keys(self):
        if self.reference.get("kind") != "external":
            return
        keys = set(self.reference.get("values", {})) | set(self.reference.get("se", {}))
        for key in keys:
            parts = key.split(":")
            p = parts[0]
            dirs = self.directions.get(p, ())
            chans = self.channels.get(p)
            if dirs:
                if chans is None:
                    chans = ("default",)  # the implicit channel of directional items
                # a direction is anchored by its own key only: a pooled or a
                # channel-only key would never be read (every item of the
                # principle would be INVALID_ANCHORS without notice)
                if len(parts) == 2:
                    ok = parts[1] in dirs
                elif len(parts) == 3:
                    ok = parts[2] in dirs and parts[1] in chans
                else:
                    ok = False
                if not ok:
                    raise ValueError(
                        f"reference key {key!r}: {p} is directional {dirs}; its "
                        f"anchors are keyed {p}:<direction> or "
                        f"{p}:<channel>:<direction>")
                continue
            if len(parts) == 2:
                ok = chans is None or parts[1] in chans
            else:
                ok = len(parts) == 1
            if not ok:
                raise ValueError(f"reference key {key!r} names no declared channel "
                                 "or direction")

    # -- accessors -----------------------------------------------------------
    def cutoff_for(self, principle) -> tuple:
        return self.cutoffs.get(str(principle), DEFAULT_CUTOFF)

    def channels_for(self, principle):
        """Declared channels of ``principle`` (None when not declared)."""
        return self.channels.get(str(principle))

    def directions_for(self, principle):
        """Declared directions of a principle (None when not directional)."""
        return self.directions.get(str(principle))

    def rank_gate_for(self, principle):
        return self.rank_gates.get(str(principle))

    def estimator_version_for(self, principle):
        return self.estimator_versions.get(str(principle))

    def se_methods_for(self, principle):
        return self.se_methods.get(str(principle))

    def estimator_options(self, principle) -> dict:
        """A fresh (mutable, JSON-typed) copy of the principle's options."""
        return _json_value(self.estimators.get(str(principle), {}))

    def concordance_admission(self, principle, substrate, observation_stage, view,
                              bearer):
        """``(absent_admitted, present_admitted)`` of a concordance cell, or
        None when the cell is not admitted."""
        key = (str(principle), substrate, observation_stage, view, bearer)
        for cell in self.concordance_route:
            if tuple(cell[k] for k in _CELL_KEYS) == key:
                return bool(cell["absent"]), bool(cell["present"])
        return None

    def reference_for(self, principle, channel="default", direction=None) -> dict:
        """The anchor of a (principle, channel[, direction]) from an external
        reference: keys ``P:channel:direction`` then ``P:direction`` for a
        direction, ``P:channel`` then ``P`` otherwise. Empty when there is
        none (or for a cohort or pending reference)."""
        ref = self.reference
        if ref.get("kind") != "external":
            return {}
        values, ses = ref.get("values") or {}, ref.get("se") or {}
        p = str(principle)
        if direction is not None:
            keys = (f"{p}:{channel}:{direction}", f"{p}:{direction}")
        else:
            keys = (f"{p}:{channel}", p)
        for key in keys:
            if key in values:
                return {
                    "reference": float(values[key]),
                    "reference_se": float(ses[key]) if key in ses else None,
                    "reference_scale": str(ref.get("scale", "excess")),
                }
        return {}

    def anchor_status(self, principle):
        if self.anchors is None:
            return None
        entry = (self.anchors.get("principles") or {}).get(str(principle))
        return None if entry is None else entry.get("status")

    @property
    def fallback(self):
        return None if self.anchors is None else self.anchors.get("fallback")

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        out = {
            "schema": PROTOCOL_SCHEMA,
            "necessity_set": list(self.necessity_set),
            "channels": {p: list(v) for p, v in self.channels.items()},
            "cutoffs": {p: list(self.cutoffs[p]) for p in PRINCIPLES},
            "alpha": float(self.alpha),
            "null_families": dict(self.null_families),
            "reference": _json_value(self.reference),
            "source_rule": self.source_rule,
            "estimators": {p: _json_value(v) for p, v in self.estimators.items()},
            "bearer_nodes": {
                p: (None if v is None else list(v))
                for p, v in self.bearer_nodes.items()
            },
            "name": self.name,
        }
        if self.status_rule is None:
            return out
        out.update({
            "status_rule": self.status_rule.to_dict(),
            "declared_necessity_set": list(self.declared_necessity_set),
            "directions": {p: list(v) for p, v in self.directions.items()},
            "rank_gates": dict(self.rank_gates),
            "estimator_versions": dict(self.estimator_versions),
            "se_methods": {p: list(v) for p, v in self.se_methods.items()},
            "shared_inputs_declaration": (
                None if self.shared_inputs_declaration is None
                else _json_value(self.shared_inputs_declaration)),
            "concordance_route": [_json_value(c) for c in self.concordance_route],
            "anchors": None if self.anchors is None else _json_value(self.anchors),
            "precision": (None if self.precision is None
                          else _json_value(self.precision)),
        })
        return out

    @property
    def hash(self) -> str:
        """SHA-256 of the canonical JSON of :meth:`to_dict`."""
        payload = _canonical_json(self.to_dict()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def protocol_id(self) -> str:
        return f"sha256:{self.hash}"

    @property
    def schema(self) -> str:
        return PROTOCOL_SCHEMA

    def __eq__(self, other):
        if not isinstance(other, Protocol):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash(self.hash)

    def __reduce__(self):
        # read-only mapping views do not pickle; the canonical dict does
        return (Protocol.from_dict, (self.to_dict(),))

    def to_json(self, path=None, indent=2) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        return text

    @classmethod
    def from_dict(cls, payload) -> "Protocol":
        if not isinstance(payload, Mapping):
            raise ValueError("a protocol must be a JSON object")
        data = dict(payload)
        schema = data.pop("schema", PROTOCOL_SCHEMA)
        if schema != PROTOCOL_SCHEMA:
            raise ValueError(f"unsupported protocol schema {schema!r}")
        unknown = sorted(set(data) - set(_PROTOCOL_FIELDS) - set(_TOST_FIELDS))
        if unknown:
            raise ValueError(f"protocol has unknown fields {unknown}")
        if data.get("status_rule") is not None:
            data["status_rule"] = StatusRule.from_dict(data["status_rule"])
        if "cutoffs" in data:
            data["cutoffs"] = {k: tuple(v) for k, v in dict(data["cutoffs"]).items()}
        for key in ("necessity_set", "declared_necessity_set"):
            if key in data and data[key] is not None:
                data[key] = normalize_necessity_set(data[key])
        return cls(**data)

    @classmethod
    def from_json(cls, path) -> "Protocol":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def replace(self, **changes) -> "Protocol":
        return dataclasses.replace(self, **changes)


def resolve_protocol(protocol):
    """``None`` | :class:`Protocol` | JSON path | dict -> Protocol (or None)."""
    if protocol is None or isinstance(protocol, Protocol):
        return protocol
    if isinstance(protocol, (str, os.PathLike)):
        return Protocol.from_json(os.fspath(protocol))
    if isinstance(protocol, Mapping):
        return Protocol.from_dict(protocol)
    raise TypeError("protocol must be None, a Protocol, a JSON path or a dict")


def _tost_protocol(protocol) -> Protocol:
    proto = resolve_protocol(protocol)
    if proto is None or proto.status_rule is None:
        raise TypeError("a protocol with a status_rule block (the TOST rule) is "
                        "required")
    return proto


# --------------------------------------------------------------------------
# MPC verdict
# --------------------------------------------------------------------------
@dataclass
class MPCVerdict:
    """
    Result of :func:`mpc_verdict`. ``component_status`` holds the
    per-principle status (Kleene OR over channels), ``margins`` the largest
    finite presence margin (``lower - z``) over the channels, ``channels`` the
    per-channel statuses, ``channel_reasons`` the per-channel reason (None when
    determinate), ``assessments`` the per-channel :class:`ComponentAssessment`
    and ``principle_assessment`` the assessment of the channel that decides
    the principle (highest status, then largest presence margin).
    ``ignored_channels`` lists evidence channels outside the protocol's
    declaration (not used).
    """

    verdict: Verdict
    reasons: list
    component_status: dict
    margins: dict
    channels: dict
    necessity_set: tuple = PRINCIPLES
    channel_reasons: dict = field(default_factory=dict)
    assessments: dict = field(default_factory=dict)
    principle_assessment: dict = field(default_factory=dict)
    ignored_channels: dict = field(default_factory=dict)
    protocol_hash: str | None = None

    @property
    def reason_string(self) -> str:
        return REASON_SEPARATOR.join(self.reasons)

    def construct_values(self) -> dict:
        """``c`` of the deciding channel per principle (NaN when undefined)."""
        return {
            p: (a.c if a is not None else float("nan"))
            for p, a in self.principle_assessment.items()
        }

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "component_status": {k: v.value for k, v in self.component_status.items()},
            "margins": {
                k: (None if not math.isfinite(v) else float(v))
                for k, v in self.margins.items()
            },
            "channels": {
                p: {c: s.value for c, s in ch.items()}
                for p, ch in self.channels.items()
            },
            "channel_reasons": {p: dict(ch) for p, ch in self.channel_reasons.items()},
            "assessments": {
                p: {c: a.to_dict() for c, a in ch.items()}
                for p, ch in self.assessments.items()
            },
            "ignored_channels": {p: list(v) for p, v in self.ignored_channels.items()},
            "necessity_set": list(self.necessity_set),
            "protocol_hash": self.protocol_hash,
        }


def _clean_detail(text):
    return str(text).replace(REASON_SEPARATOR, ",").replace("\n", " ").strip()


def _channel_reason_code(principle, ev_or_channel, reason, estimator=None):
    channel = getattr(ev_or_channel, "channel", ev_or_channel)
    if reason in _BARE_PRINCIPLE_REASONS:
        return f"{reason}:{principle}"
    if reason in (REASON_NOT_IMPLEMENTED, REASON_MISSING_CHANNEL):
        return f"{reason}:{principle}:{_clean_detail(channel)}"
    if reason is not None and reason.startswith(REASON_NOT_VALIDATED):
        est = _clean_detail(estimator) if estimator else "undeclared"
        return f"{REASON_NOT_VALIDATED}:{principle}:{est}"
    return f"{REASON_UNDEFINED}:{principle}:{_clean_detail(reason)}"


def parse_reason(code):
    """Split a reason code into ``(kind, principle | None, detail | None)``."""
    parts = str(code).split(":", 2)
    kind = parts[0]
    if kind in GLOBAL_REASON_KINDS:
        return kind, None, (":".join(parts[1:]) or None)
    principle = parts[1] if len(parts) > 1 else None
    detail = parts[2] if len(parts) > 2 else None
    return kind, principle, detail


def verdict_from_reasons(reasons) -> Verdict:
    """The decomposition rule: the verdict implied by a list of reason codes."""
    kinds = [parse_reason(r)[0] for r in reasons]
    if not kinds:
        return Verdict.MPC_CONSISTENT
    if any(k in GLOBAL_REASON_KINDS for k in kinds):
        return Verdict.UNDETERMINED
    if REASON_ABSENT in kinds:
        return Verdict.EXCLUDED
    return Verdict.UNDETERMINED


def _dedupe(seq):
    return list(dict.fromkeys(seq))


def _coherence_ok(bearer_coherence):
    if isinstance(bearer_coherence, (bool, np.bool_)):
        return bool(bearer_coherence)
    if isinstance(bearer_coherence, Mapping):
        return bool(bearer_coherence.get("coherent", False))
    raise TypeError("bearer_coherence must be a bool or a bearer_coherence() result")


def _source_reason(gated, joint):
    """
    Single-source check of the node sets of the gated evidence: None when one
    node set is used or a covering joint-dependence test found dependence;
    otherwise a ``SOURCE_INCOHERENT[:detail]`` code. A joint-dependence result
    must test exactly the evidence node sets (an undeclared node set, None,
    can never be covered).
    """
    sets = {ev.nodes for ev in gated}
    if len(sets) <= 1:
        return None
    untested = f"{REASON_SOURCE_INCOHERENT}:{SOURCE_UNTESTED}"
    if joint is None:
        return untested
    if isinstance(joint, (bool, np.bool_)):
        return None if bool(joint) else REASON_SOURCE_INCOHERENT
    if not isinstance(joint, Mapping):
        raise TypeError(
            "joint_dependence must be a bool or a joint_dependence() result"
        )
    tested = {_node_tuple(s) for s in joint.get("sets", [])}
    if None in sets or tested != sets:
        return untested
    if bool(joint.get("dependent", False)):
        return None
    why = joint.get("reason")
    if why and why != REASON_SOURCE_INCOHERENT:
        return f"{REASON_SOURCE_INCOHERENT}:{_clean_detail(why)}"
    return REASON_SOURCE_INCOHERENT


@functools.lru_cache(maxsize=256)
def _default_protocol(nset, cutoff_items, alpha, rule):
    """Protocol of a keyword-only mpc_verdict call (immutable, so cached)."""
    return Protocol(
        necessity_set=nset, cutoffs=dict(cutoff_items), alpha=alpha, source_rule=rule
    )


_LEGACY_STATUS_KWARGS = ("z_present", "delta_equiv")


def _status_settings(status_kwargs):
    unknown = sorted(set(status_kwargs) - {"cutoffs", "alpha"})
    legacy = [k for k in unknown if k in _LEGACY_STATUS_KWARGS]
    if legacy:
        raise TypeError(
            f"{legacy}: the null-SD thresholds were replaced by construct-scale "
            "cutoffs (use cutoffs={P: (z, delta)} or a Protocol)"
        )
    if unknown:
        raise TypeError(f"unexpected status keyword(s): {unknown}")
    cutoffs = status_kwargs.get("cutoffs")
    if cutoffs is not None and not isinstance(cutoffs, Mapping):
        cutoffs = {p: cutoffs for p in PRINCIPLES}  # one pair for every principle
    return cutoffs, status_kwargs.get("alpha")


def mpc_verdict(evidence: Mapping, protocol=None, **kwargs) -> MPCVerdict:
    """
    Three-valued MPC verdict (exclusion rule) from per-principle evidence,
    judged by the protocol's status rule: :func:`tost_verdict` for a
    protocol with a ``status_rule`` block, the interval rule
    (:func:`interval_verdict`) otherwise or without a protocol. ``kwargs``
    are the keywords of that function.
    """
    proto = resolve_protocol(protocol)
    if proto is not None and proto.status_rule is not None:
        return tost_verdict(evidence, proto, **kwargs)
    return interval_verdict(evidence, proto, **kwargs)


def interval_verdict(
    evidence: Mapping,
    protocol=None,
    *,
    necessity_set=None,
    registry=None,
    regime=None,
    joint_dependence=None,
    bearer_coherence=None,
    require_same_bearer=None,
    require_same_protocol=True,
    **status_kwargs,
) -> MPCVerdict:
    """
    The MPC verdict under the interval rule (:func:`component_assessment`).

    ``evidence`` maps a principle to a ComponentEvidence or a sequence of them
    (one per channel). ``protocol`` (:class:`Protocol`, dict or JSON path)
    declares the necessity set, channels, cutoffs, alpha, null families and
    source rule; without it a default protocol is built from
    ``necessity_set``, ``cutoffs`` (a pair for every principle or a mapping)
    and ``alpha`` (these keywords must not be combined with a protocol).

    Per principle the channel statuses are combined by Kleene OR (a principle
    is ABSENT only if every channel is ABSENT; a declared channel without an
    item is UNDEFINED, ``MISSING_CHANNEL``; items of undeclared channels are
    ignored and listed in ``ignored_channels``); the verdict is the Kleene AND
    over the necessity set: T -> MPC_CONSISTENT, F -> EXCLUDED,
    U -> UNDETERMINED. With ``registry``
    (:class:`~impact_pipeline.registry.ApplicabilityRegistry`), evidence from
    an estimator that is not validated for its principle,
    substrate, grain and regime (``regime`` updated by the item's own
    ``regime``) is UNDEFINED.

    Global codes (UNDETERMINED whatever the components say), over the gated
    evidence of the necessity set (undefined items keep their declarations,
    so missingness can never resolve a mismatch):

    - ``BEARER_MISMATCH`` for more than one declared ``bearer_id`` (source
      rules ``single_source``/``same_bearer``; ``BEARER_MISMATCH:COHERENCE``
      for a failed legacy ``bearer_coherence`` diagnostic);
    - ``PROTOCOL_MISMATCH`` for more than one declared ``protocol_id``, or an
      id other than the given protocol's (``require_same_protocol``);
    - ``SOURCE_INCOHERENT`` (source rule ``single_source``) when the evidence
      comes from more than one node set and ``joint_dependence`` (a
      :func:`joint_dependence` result over exactly those node sets, or a bool)
      does not show dependence above null (``:UNTESTED`` without a covering
      test).
    """
    cutoffs, alpha = _status_settings(status_kwargs)
    explicit = protocol is not None
    proto = resolve_protocol(protocol)
    if explicit and proto.status_rule is not None:
        raise ValueError("a protocol with a status_rule block is judged by "
                         "tost_verdict")
    if explicit:
        clash = [
            name for name, val in (
                ("necessity_set", necessity_set), ("cutoffs", cutoffs),
                ("alpha", alpha), ("require_same_bearer", require_same_bearer),
            ) if val is not None
        ]
        if clash:
            raise ValueError(f"pass {clash} through the Protocol, not alongside it")
    else:
        rule = "single_source" if require_same_bearer in (None, True) else "none"
        proto = _default_protocol(
            normalize_necessity_set(necessity_set),
            tuple(sorted(
                (_principle_key(k, "cutoffs"), normalize_cutoff(v))
                for k, v in dict(cutoffs or {}).items()
            )),
            DEFAULT_ALPHA if alpha is None else alpha,
            rule,
        )
    nset = proto.necessity_set
    items_by_p = {}
    for key, items in dict(evidence or {}).items():
        if isinstance(items, ComponentEvidence):
            items = [items]
        items = list(items or [])
        for ev in items:
            if not isinstance(ev, ComponentEvidence):
                raise TypeError(f"evidence for {key} must be ComponentEvidence")
            if ev.principle != key:
                raise ValueError(
                    f"evidence keyed {key!r} declares principle {ev.principle!r}"
                )
        items_by_p[str(key)] = items
    ordered = [p for p in PRINCIPLES if p in items_by_p or p in nset]
    ordered += sorted(p for p in items_by_p if p not in PRINCIPLES)
    regime = dict(regime or {})

    statuses, margins, channels, channel_reasons = {}, {}, {}, {}
    assessments, deciding, ignored, kept_by_p = {}, {}, {}, {}
    p_codes, p_reasons = {}, {}
    for p in ordered:
        items = items_by_p.get(p, [])
        declared = proto.channels_for(p)
        if declared is not None:
            kept = [ev for ev in items if ev.channel in declared]
            extra = sorted({ev.channel for ev in items if ev.channel not in declared})
            if extra:
                ignored[p] = extra
        else:
            kept = items
        kept_by_p[p] = kept
        channels[p], channel_reasons[p], assessments[p] = {}, {}, {}
        if not kept and declared is None:
            p_codes[p] = CODE_U
            p_reasons[p] = [f"{REASON_MISSING}:{p}"]
            margins[p] = float("nan")
            statuses[p] = ComponentStatus.UNDEFINED
            deciding[p] = None
            continue
        cut = proto.cutoff_for(p)
        family = proto.null_families.get(p)
        codes, u_reasons, best = [], [], float("nan")
        best_key = {}
        for ev in kept:
            reason = None
            if registry is not None:
                item_regime = {**regime, **dict(ev.regime or {})}
                ok, why = registry.is_validated(
                    p, ev.estimator, ev.substrate, ev.grain, **item_regime
                )
                if not ok:
                    reason = f"{REASON_NOT_VALIDATED}:{why}"
            if reason is None and family is not None and ev.null_family is not None:
                if str(ev.null_family) != family:
                    reason = (
                        f"{REASON_NULL_FAMILY_MISMATCH}:{family}/{ev.null_family}"
                    )
            if reason is None:
                a = component_assessment(ev, cutoff=cut, alpha=proto.alpha)
            else:
                a = ComponentAssessment(
                    ComponentStatus.UNDEFINED, reason, *([float("nan")] * 7),
                    cut[0], cut[1], proto.alpha,
                )
            code = _STATUS_TO_CODE[a.status]
            margin = a.margin_present
            if math.isfinite(margin) and not (margin <= best):
                best = margin
            key = (code, margin if math.isfinite(margin) else -math.inf)
            if ev.channel not in best_key or key > best_key[ev.channel]:
                best_key[ev.channel] = key
                assessments[p][ev.channel] = a
            prev = _STATUS_TO_CODE.get(channels[p].get(ev.channel), CODE_F)
            channels[p][ev.channel] = _CODE_TO_STATUS[max(prev, code)]
            if code == CODE_U:
                channel_reasons[p].setdefault(ev.channel, a.reason)
                u_reasons.append(_channel_reason_code(p, ev, a.reason, ev.estimator))
            codes.append(code)
        for ch in declared or ():
            if ch not in channels[p]:
                channels[p][ch] = ComponentStatus.UNDEFINED
                channel_reasons[p][ch] = REASON_MISSING_CHANNEL
                u_reasons.append(_channel_reason_code(p, ch, REASON_MISSING_CHANNEL))
                codes.append(CODE_U)
        channel_reasons[p] = {
            ch: r for ch, r in channel_reasons[p].items()
            if channels[p][ch] == ComponentStatus.UNDEFINED
        }
        code_p = max(codes)
        p_codes[p] = code_p
        statuses[p] = _CODE_TO_STATUS[code_p]
        margins[p] = best
        deciding[p] = (
            max(
                assessments[p].values(),
                key=lambda a: (
                    _STATUS_TO_CODE[a.status],
                    a.margin_present if math.isfinite(a.margin_present) else -math.inf,
                ),
            )
            if assessments[p] else None
        )
        if code_p == CODE_T:
            p_reasons[p] = []
        elif code_p == CODE_F:
            p_reasons[p] = [f"{REASON_ABSENT}:{p}"]
        else:
            p_reasons[p] = _dedupe(u_reasons)

    global_reasons = []
    gated = [ev for p in nset for ev in kept_by_p.get(p, [])]
    same_bearer = proto.source_rule in ("single_source", "same_bearer")
    if same_bearer and len({e.bearer_id for e in gated} - {None}) > 1:
        global_reasons.append(REASON_BEARER_MISMATCH)
    if bearer_coherence is not None and not _coherence_ok(bearer_coherence):
        global_reasons.append(f"{REASON_BEARER_MISMATCH}:COHERENCE")
    if require_same_protocol:
        ids = {e.protocol_id for e in gated} - {None}
        if len(ids) > 1 or (explicit and ids and ids != {proto.protocol_id}):
            global_reasons.append(REASON_PROTOCOL_MISMATCH)
    if proto.source_rule == "single_source":
        why = _source_reason(gated, joint_dependence)
        if why is not None:
            global_reasons.append(why)

    code = min(p_codes[p] for p in nset)
    if global_reasons:
        code = CODE_U
    reasons = global_reasons + [r for p in nset for r in p_reasons[p]]
    return MPCVerdict(
        verdict=_CODE_TO_VERDICT[code],
        reasons=reasons,
        component_status=statuses,
        margins=margins,
        channels=channels,
        necessity_set=nset,
        channel_reasons=channel_reasons,
        assessments=assessments,
        principle_assessment=deciding,
        ignored_channels=ignored,
        protocol_hash=proto.hash if explicit else None,
    )


# --------------------------------------------------------------------------
# TOST rule: quantiles and the decision (one implementation)
# --------------------------------------------------------------------------
def quantile(alpha, df=math.inf) -> float:
    """One-sided ``1 - alpha`` quantile: Student ``t`` with ``df`` degrees of
    freedom, the normal quantile for an infinite or missing ``df``."""
    a = float(alpha)
    if not (0.0 < a <= 0.5):
        raise ValueError("alpha must be in (0, 0.5]")
    return _one_sided_quantile(a, _as_float(df))


def _quantiles(alpha, df) -> np.ndarray:
    a = float(alpha)
    if not (0.0 < a <= 0.5):
        raise ValueError("alpha must be in (0, 0.5]")
    d = np.asarray(df, dtype=float)
    q = np.full(d.shape, _z_quantile(a))
    fin = np.isfinite(d) & (d > 0)
    if np.any(fin):
        from scipy.special import stdtrit

        # each distinct df once (the same values as element-wise evaluation)
        uniq, inv = np.unique(d[fin], return_inverse=True)
        q[fin] = stdtrit(uniq, 1.0 - a)[inv]
    return q


# Reason indices of the vectorised core (0 = no reason).
_REASON_TABLE = (
    None,
    R.NULL_MODEL_VIOLATED,
    R.ABSENT_NOT_REACHABLE,
    R.INCONCLUSIVE,
    R.NON_FINITE_ESTIMATE,
    R.NO_NULL_CALIBRATION,
    R.DEGENERATE_NULL,
    R.INVALID_ANCHORS,
    R.INVALID_SE,
    R.NO_SAMPLING_SE,
)
_RI = {r: i for i, r in enumerate(_REASON_TABLE)}
_DECISION_REASONS = frozenset(
    (R.NULL_MODEL_VIOLATED, R.ABSENT_NOT_REACHABLE, R.INCONCLUSIVE))
_FIRST_VALIDITY_INDEX = _RI[R.NON_FINITE_ESTIMATE]


def _decide(c, se, q_p, q_a, z, delta, null_violation=True):
    """
    The TOST rule on the construct scale for valid inputs (finite ``c``,
    finite ``se >= 0``): status codes, reason indices (0 for PRESENT and
    ABSENT) and the bounds ``c -+ q_P se`` and ``c -+ q_A se``. Every entry
    point of this module decides through this function.
    """
    c, se, q_p, q_a, z, delta = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (c, se, q_p, q_a, z, delta)))
    with np.errstate(invalid="ignore", over="ignore"):
        w_p = q_p * se
        w_a = q_a * se
        lower_p, upper_p = c - w_p, c + w_p
        lower_a, upper_a = c - w_a, c + w_a
        present = lower_p > z
        absent = ~present & (lower_a > -delta) & (upper_a < delta)
        undefined = ~(present | absent)
        if null_violation:
            nmv = undefined & (upper_p < -delta)
        else:
            nmv = np.zeros(c.shape, dtype=bool)
        anr = undefined & ~nmv & (w_a >= delta)
    codes = np.where(present, CODE_T, np.where(absent, CODE_F, CODE_U)).astype(np.int8)
    ridx = np.where(
        nmv, _RI[R.NULL_MODEL_VIOLATED],
        np.where(anr, _RI[R.ABSENT_NOT_REACHABLE],
                 np.where(undefined, _RI[R.INCONCLUSIVE], 0)),
    ).astype(np.int8)
    return codes, ridx, lower_p, upper_p, lower_a, upper_a


def _present_not_reachable(codes, se, q_p, z):
    with np.errstate(invalid="ignore", over="ignore"):
        w = np.asarray(q_p, dtype=float) * np.asarray(se, dtype=float)
        reach = np.isfinite(w) & (w >= 1.0 - np.asarray(z))
        return (np.asarray(codes) == CODE_U) & reach


class StatusArrays(NamedTuple):
    """Vectorised statuses: ``status`` (object array of status strings),
    ``codes`` (1 PRESENT, 0 UNDEFINED, -1 ABSENT), ``reason`` (object array;
    None for PRESENT and ABSENT), ``present_not_reachable`` (the flag),
    ``c``, ``se``, ``df``, the quantiles, the PRESENT-level bounds
    ``lower``/``upper``, the TOST bounds ``lower_absent``/``upper_absent``
    and the margins (``margin_present > 0`` iff PRESENT,
    ``margin_absent > 0`` iff the TOST holds)."""

    status: np.ndarray
    codes: np.ndarray
    reason: np.ndarray
    present_not_reachable: np.ndarray
    c: np.ndarray
    se: np.ndarray
    df: np.ndarray
    q_present: np.ndarray
    q_absent: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    lower_absent: np.ndarray
    upper_absent: np.ndarray
    margin_present: np.ndarray
    margin_absent: np.ndarray


def _status_strings(codes):
    out = np.empty(np.shape(codes), dtype=object)
    for code, st in _CODE_TO_STATUS.items():
        out[np.asarray(codes) == code] = st.value
    return out


def _reason_strings(ridx):
    ridx = np.asarray(ridx)
    out = np.empty(ridx.shape, dtype=object)
    for i, r in enumerate(_REASON_TABLE):
        out[ridx == i] = r
    return out


def _arrays(codes, ridx, pnr, c, se, df, q_p, q_a, lp, up, la, ua, z, delta, valid):
    nan = np.nan
    with np.errstate(invalid="ignore"):
        m_p = np.where(valid, lp - z, nan)
        m_a = np.where(valid, np.minimum(delta - ua, la + delta), nan)
    return StatusArrays(
        _status_strings(codes), codes, _reason_strings(ridx), pnr,
        c, np.where(valid, se, nan), np.where(valid, df, nan),
        np.where(valid, q_p, nan), np.where(valid, q_a, nan),
        np.where(valid, lp, nan), np.where(valid, up, nan),
        np.where(valid, la, nan), np.where(valid, ua, nan), m_p, m_a,
    )


def _levels(rule, alpha_present, alpha_absent):
    rule = _coerce_rule(rule)
    a_p = rule.alpha if alpha_present is None else float(alpha_present)
    a_a = rule.alpha_absent if alpha_absent is None else float(alpha_absent)
    return rule, a_p, a_a


def status_c(c, se, df=None, *, cutoff=DEFAULT_CUTOFF, rule=None, exact=False,
             alpha_present=None, alpha_absent=None) -> StatusArrays:
    """
    The TOST rule on the construct scale:
    statuses of ``c`` with SE ``se`` and degrees of freedom ``df`` (None,
    NaN or infinite: normal quantile). ``cutoff`` is ``(z, delta)`` or a
    per-column array pair. Invalid rows are UNDEFINED: a non-finite ``c``
    (``NON_FINITE_ESTIMATE``), a negative SE or ``df <= 0`` (``INVALID_SE``),
    a zero or missing SE without ``exact`` (``NO_SAMPLING_SE``); an ``exact``
    row may have SE 0. ``alpha_present`` / ``alpha_absent`` override the
    rule's levels for one member test (:func:`member_levels`).
    """
    rule, a_p, a_a = _levels(rule, alpha_present, alpha_absent)
    z_arr, d_arr = (np.asarray(v, dtype=float) for v in cutoff)
    if (np.any(~np.isfinite(z_arr)) or np.any(~np.isfinite(d_arr))
            or np.any(d_arr > z_arr)):
        raise ValueError("cutoffs must be finite with delta <= z")
    c = np.asarray(c, dtype=float)
    se = np.asarray(se, dtype=float)
    dfa = np.asarray(np.nan if df is None else df, dtype=float)
    ex = np.asarray(exact, dtype=bool)
    c, se, dfa, ex, z_b, d_b = np.broadcast_arrays(c, se, dfa, ex, z_arr, d_arr)
    with np.errstate(invalid="ignore"):
        se_eff = np.where(ex & ~(np.isfinite(se) & (se > 0)), 0.0, se)
        bad_se = (np.isfinite(se) & (se < 0)) | (np.isfinite(dfa) & (dfa <= 0))
        no_se = ~(np.isfinite(se) & (se > 0)) & ~ex
    vidx = np.select(
        [~np.isfinite(c), bad_se, no_se],
        [_RI[R.NON_FINITE_ESTIMATE], _RI[R.INVALID_SE], _RI[R.NO_SAMPLING_SE]],
        default=0,
    ).astype(np.int8)
    valid = vidx == 0
    df_eff = np.where(np.isfinite(dfa) & (dfa > 0), dfa, np.inf)
    q_p = _quantiles(a_p, df_eff)
    q_a = _quantiles(a_a, df_eff)
    codes, ridx, lp, up, la, ua = _decide(
        np.where(valid, c, 0.0), np.where(valid, se_eff, 0.0), q_p, q_a, z_b, d_b,
        rule.null_violation)
    codes = np.where(valid, codes, CODE_U).astype(np.int8)
    ridx = np.where(valid, ridx, vidx).astype(np.int8)
    if rule.present_reachability_flag:
        pnr = valid & _present_not_reachable(codes, se_eff, q_p, z_b)
    else:
        pnr = np.zeros(codes.shape, dtype=bool)
    return _arrays(codes, ridx, pnr, c, se_eff, df_eff, q_p, q_a, lp, up, la, ua,
                   z_b, d_b, valid)


def _raw_construct(estimate, null_mean, null_sd, se, n_null, reference, reference_se,
                   exact, reference_scale, se_df):
    """c, se_c, df, the validity reason index and the exact-route mask (an
    ``exact`` row without a sampling SE) of raw inputs, with the arithmetic
    and the order of checks of :func:`component_assessment`."""
    if reference_scale not in REFERENCE_SCALES:
        raise ValueError(f"reference_scale must be one of {REFERENCE_SCALES}")
    est, nm, nsd, s, nn, ref, rse, sdf = np.broadcast_arrays(
        *(np.asarray(np.nan if v is None else v, dtype=float)
          for v in (estimate, null_mean, null_sd, se, n_null, reference,
                    reference_se, se_df)))
    ex = np.broadcast_to(np.asarray(exact, dtype=bool), est.shape)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        nn_ok = np.isfinite(nn) & (nn >= 0) & (nn == np.floor(nn))
        nsd_bad = (nn > 0) & ~(np.isfinite(nsd) & (nsd >= 0))
        denom = ref if reference_scale == "excess" else ref - nm
        anchors_ok = np.isfinite(ref) & np.isfinite(denom) & (denom > 0)
        rse_neg = np.isfinite(rse) & (rse < 0)
        se_neg = np.isfinite(s) & (s < 0)
        sdf_bad = np.isfinite(sdf) & (sdf <= 0)
        has_se = np.isfinite(s) & (s > 0)
        vidx = np.select(
            [~np.isfinite(est), ~np.isfinite(nm), ~nn_ok, nsd_bad, ~anchors_ok,
             rse_neg, se_neg, sdf_bad, ~has_se & ~ex],
            [_RI[R.NON_FINITE_ESTIMATE], _RI[R.NO_NULL_CALIBRATION],
             _RI[R.DEGENERATE_NULL], _RI[R.DEGENERATE_NULL], _RI[R.INVALID_ANCHORS],
             _RI[R.INVALID_SE], _RI[R.INVALID_SE], _RI[R.INVALID_SE],
             _RI[R.NO_SAMPLING_SE]],
            default=0,
        ).astype(np.int8)
        c = (est - nm) / denom
        # c is reported from the anchor check on (as component_assessment does)
        c = np.where((vidx == 0) | (vidx >= _RI[R.INVALID_SE]), c, np.nan)
        se_nu = np.where(nn > 0, nsd / np.sqrt(np.where(nn > 0, nn, 1.0)), 0.0)
        s_eff = np.where(has_se, s, 0.0)
        rse_eff = np.where(np.isfinite(rse), rse, 0.0)
        if reference_scale == "excess":
            d_nu = -1.0 / denom
        else:
            d_nu = (c - 1.0) / denom
        d_ref = -c / denom
        s_samp = s_eff / denom
        s_null = np.abs(d_nu) * se_nu
        s_ref = np.abs(d_ref) * rse_eff
        var_c = s_samp * s_samp + s_null * s_null + s_ref * s_ref
        se_c = np.sqrt(var_c)
        ratio = var_c / (s_samp * s_samp)
        df = np.where(np.isfinite(sdf) & (s_samp > 0), sdf * ratio * ratio, np.inf)
        df = np.where(np.isfinite(df), df, np.inf)
    return c, se_c, df, vidx, ex & ~has_se


def assess_array(estimate, null_mean, null_sd, se=0.0, *, n_null=0,
                 reference=np.nan, reference_se=0.0, exact=False,
                 reference_scale="estimate", se_df=np.nan, cutoff=DEFAULT_CUTOFF,
                 rule=None, alpha_present=None, alpha_absent=None) -> StatusArrays:
    """
    Vectorised TOST rule from raw inputs (the arguments of
    :func:`component_status_array`): ``c``, ``se_c`` and ``df`` as the
    interval rule computes them, then the TOST decision. Invalid rows carry
    the validity reasons in the interval rule's order
    (``NON_FINITE_ESTIMATE``, ``NO_NULL_CALIBRATION``,
    ``DEGENERATE_NULL``, ``INVALID_ANCHORS``, ``INVALID_SE``,
    ``NO_SAMPLING_SE``). An ``exact`` row without a sampling SE is decided
    on ``c`` alone (``se_c = 0``), as in :func:`assess_item`.
    """
    rule, a_p, a_a = _levels(rule, alpha_present, alpha_absent)
    z, delta = normalize_cutoff(cutoff)
    c, se_c, df, vidx, exact_route = _raw_construct(
        estimate, null_mean, null_sd, se, n_null, reference, reference_se, exact,
        reference_scale, se_df)
    se_c = np.where(exact_route, 0.0, se_c)
    valid = vidx == 0
    q_p = _quantiles(a_p, np.where(valid, df, np.inf))
    q_a = _quantiles(a_a, np.where(valid, df, np.inf))
    codes, ridx, lp, up, la, ua = _decide(
        np.where(valid, c, 0.0), np.where(valid, se_c, 0.0), q_p, q_a, z, delta,
        rule.null_violation)
    codes = np.where(valid, codes, CODE_U).astype(np.int8)
    ridx = np.where(valid, ridx, vidx).astype(np.int8)
    if rule.present_reachability_flag:
        pnr = valid & _present_not_reachable(codes, se_c, q_p, z)
    else:
        pnr = np.zeros(codes.shape, dtype=bool)
    return _arrays(codes, ridx, pnr, c, se_c, df, q_p, q_a, lp, up, la, ua, z, delta,
                   valid)


def directional_codes(member_codes) -> np.ndarray:
    """Combined status codes of directional members (last axis): PRESENT iff
    every member is PRESENT (intersection-union), ABSENT iff some member is
    ABSENT (union), else UNDEFINED."""
    m = np.asarray(member_codes, dtype=np.int8)
    out = np.where(np.all(m == CODE_T, axis=-1), CODE_T,
                   np.where(np.any(m == CODE_F, axis=-1), CODE_F, CODE_U))
    return out.astype(np.int8)


def status_c_directional(c, se, df=None, *, cutoff=DEFAULT_CUTOFF, rule=None,
                         n_channels=1) -> StatusArrays:
    """
    The status of a directional principle on the construct scale: ``c``,
    ``se`` and ``df`` carry the directions on the last axis (``k`` members).
    Each member is tested for PRESENT at ``alpha / n_channels`` and for ABSENT
    at ``alpha_A / k``; the combination is :func:`directional_codes`, the
    reason the first member validity reason, else ``NULL_MODEL_VIOLATED`` if
    a member has it, else ``ABSENT_NOT_REACHABLE`` if no member can pass the
    TOST, else ``INCONCLUSIVE``. The reported ``c`` is the smallest member
    value (with that member's SE, df and bounds): NaN for an UNDEFINED row
    with a non-finite member, the smallest finite member value for a decided
    row (an ABSENT through one direction while the other has no value); the
    margins are the smallest member PRESENT margin and the largest member
    TOST margin.
    """
    rule = _coerce_rule(rule)
    c = np.asarray(c, dtype=float)
    if c.ndim < 1 or c.shape[-1] < 2:
        raise ValueError("directional values need two or more members on the last axis")
    k = c.shape[-1]
    a_p, a_a = member_levels(rule, n_channels, k)
    m = status_c(c, se, df, cutoff=cutoff, rule=rule, alpha_present=a_p,
                 alpha_absent=a_a)
    z, delta = (np.asarray(v, dtype=float) for v in cutoff)
    codes = directional_codes(m.codes)
    ridx_m = np.zeros(m.codes.shape, dtype=np.int8)
    for i, r in enumerate(_REASON_TABLE):
        if r is not None:
            ridx_m[m.reason == r] = i
    validity = ridx_m >= _FIRST_VALIDITY_INDEX
    has_validity = np.any(validity, axis=-1)
    first_validity = np.take_along_axis(
        ridx_m, np.argmax(validity, axis=-1)[..., None], axis=-1)[..., 0]
    any_nmv = np.any(ridx_m == _RI[R.NULL_MODEL_VIOLATED], axis=-1)
    with np.errstate(invalid="ignore"):
        unreachable = np.all(m.q_absent * m.se >= delta, axis=-1)
    und = codes == CODE_U
    ridx = np.where(~und, 0, np.where(
        has_validity, first_validity, np.where(
            any_nmv, _RI[R.NULL_MODEL_VIOLATED], np.where(
                unreachable, _RI[R.ABSENT_NOT_REACHABLE], _RI[R.INCONCLUSIVE]))))
    with np.errstate(invalid="ignore"):
        pnr_member = np.isfinite(m.se) & (m.q_present * m.se >= 1.0 - z)
    pnr = und & np.any(pnr_member, axis=-1) if rule.present_reachability_flag else (
        np.zeros(codes.shape, dtype=bool))
    pnr = pnr & ~has_validity
    # a decided row always has a finite member (PRESENT: all, ABSENT: the
    # member that passed the TOST)
    report = np.all(np.isfinite(m.c), axis=-1) | ~und
    arg = np.argmin(np.where(np.isfinite(m.c), m.c, np.inf), axis=-1)[..., None]

    def pick(a):
        return np.where(report, np.take_along_axis(a, arg, axis=-1)[..., 0], np.nan)

    with np.errstate(invalid="ignore"):
        m_p = np.where(np.all(np.isfinite(m.margin_present), axis=-1),
                       np.min(m.margin_present, axis=-1), np.nan)
        m_a_raw = np.where(np.isfinite(m.margin_absent), m.margin_absent, -np.inf)
        m_a = np.max(m_a_raw, axis=-1)
        m_a = np.where(np.isfinite(m_a), m_a, np.nan)
    return StatusArrays(
        _status_strings(codes), codes, _reason_strings(ridx), pnr,
        pick(m.c), pick(m.se), pick(m.df), pick(m.q_present), pick(m.q_absent),
        pick(m.lower), pick(m.upper), pick(m.lower_absent), pick(m.upper_absent),
        m_p, m_a,
    )


# --------------------------------------------------------------------------
# TOST rule: assessments
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TostAssessment:
    """
    The TOST assessment of one channel member (or of a directional channel,
    with its per-direction ``members``): the status, its reason (UNDEFINED
    only, from the central vocabulary) and flags; ``c``, ``se`` (``se_c``)
    and its parts, ``df``; the quantiles and the bounds at the PRESENT level
    (``lower``/``upper`` = ``c -+ q_P se``) and of the TOST
    (``lower_absent``/``upper_absent`` = ``c -+ q_A se``); the margins
    (``margin_present > 0`` iff PRESENT by the rule, ``margin_absent > 0``
    iff the TOST holds); the cutoffs and the levels ``alpha`` (PRESENT test)
    and ``alpha_absent`` (each one-sided test of the TOST); ``route`` (tost,
    exact, concordance, directional, or None when the inputs were invalid).
    """

    status: ComponentStatus
    reason: Optional[str]
    flags: Tuple[str, ...] = ()
    c: float = float("nan")
    se: float = float("nan")
    se_sampling: float = float("nan")
    se_null: float = float("nan")
    se_reference: float = float("nan")
    df: float = float("nan")
    q_present: float = float("nan")
    q_absent: float = float("nan")
    lower: float = float("nan")
    upper: float = float("nan")
    lower_absent: float = float("nan")
    upper_absent: float = float("nan")
    margin_present: float = float("nan")
    margin_absent: float = float("nan")
    cutoff_present: float = DEFAULT_CUTOFF[0]
    cutoff_absent: float = DEFAULT_CUTOFF[1]
    alpha: float = DEFAULT_ALPHA
    alpha_absent: float = DEFAULT_ALPHA_ABSENT
    route: Optional[str] = None
    se_method: Optional[str] = None
    channel: str = "default"
    direction: Optional[str] = None
    members: Tuple["TostAssessment", ...] = ()
    scale: str = "construct"

    def __post_init__(self):
        R.check_status(self.status.value, self.reason, self.flags)

    @property
    def directions(self) -> dict:
        """``{direction: c}`` of a directional assessment."""
        return {m.direction: m.c for m in self.members}

    def to_dict(self) -> dict:
        def _f(v):
            return None if not math.isfinite(v) else float(v)

        out = {
            "status": self.status.value,
            "reason": self.reason,
            "flags": list(self.flags),
            "c": _f(self.c),
            "se": _f(self.se),
            "se_sampling": _f(self.se_sampling),
            "se_null": _f(self.se_null),
            "se_reference": _f(self.se_reference),
            "df": _f(self.df),
            "q_present": _f(self.q_present),
            "q_absent": _f(self.q_absent),
            "lower": _f(self.lower),
            "upper": _f(self.upper),
            "lower_absent": _f(self.lower_absent),
            "upper_absent": _f(self.upper_absent),
            "margin_present": _f(self.margin_present),
            "margin_absent": _f(self.margin_absent),
            "cutoff_present": self.cutoff_present,
            "cutoff_absent": self.cutoff_absent,
            "alpha": self.alpha,
            "alpha_absent": self.alpha_absent,
            "route": self.route,
            "se_method": self.se_method,
            "channel": self.channel,
            "direction": self.direction,
            "scale": self.scale,
        }
        if self.members:
            out["members"] = {m.direction: m.to_dict() for m in self.members}
        return out


def _vocab_reason(text, default) -> str:
    """A reason of the vocabulary: the text itself when it is one, else
    ``NOT_DEFINED:<cleaned text>`` (the estimator's own reason), else the
    default."""
    if text is None or not str(text).strip():
        return default
    t = str(text)
    if R.is_reason(t):
        return t
    detail = R.clean_detail(t)
    return R.format_reason(R.NOT_DEFINED, detail) if detail else R.NOT_DEFINED


class _Levels(NamedTuple):
    cutoff: tuple
    rule: StatusRule
    alpha_present: float
    alpha_absent: float
    scale: str


def _undefined_assessment(reason, lv: _Levels, ev=None, *, c=float("nan"),
                          channel="default", direction=None) -> TostAssessment:
    R.lookup(reason)
    return TostAssessment(
        UNDEFINED, reason, (), c=float(c),
        cutoff_present=lv.cutoff[0], cutoff_absent=lv.cutoff[1],
        alpha=lv.alpha_present, alpha_absent=lv.alpha_absent,
        se_method=(ev.se_method if ev is not None else None),
        channel=(ev.channel if ev is not None else channel),
        direction=(ev.direction if ev is not None else direction),
        scale=lv.scale,
    )


def _se_contract_violation(ev, principle, allowed) -> bool:
    """True iff the item breaks the SE-method contract: no method where the
    protocol declares the admitted methods of the principle, an unknown
    method, a method not allowed for the principle or not declared by the
    protocol, ``se_df`` inconsistent with the method's rule, a sampling SE
    on a method without one, or ``exact`` inconsistent with the method.
    Without a method and without a protocol declaration ``se_df`` keeps its
    plain meaning."""
    name = ev.se_method
    if name is None:
        # a declared contract binds every item: an item that does not name
        # its method could carry, e.g., B - 1 for a block bootstrap
        return allowed is not None
    method = SE_METHODS.get(name)
    if method is None or principle not in method.principles:
        return True
    if allowed is not None and name not in allowed:
        return True
    se = _as_float(ev.se)
    se_df = _as_float(ev.se_df)
    exact = bool(ev.exact)
    if (name == SE_METHOD_EXACT) != exact:
        return True
    if not method.sampling_se:
        return (math.isfinite(se) and se > 0) or math.isfinite(se_df)
    if method.df_rule == DF_FIXED:
        return not (math.isfinite(se_df) and se_df in method.df_values)
    if method.df_rule == DF_N_NULL_MINUS_1:
        n_null = _as_float(ev.n_null)
        return not (math.isfinite(se_df) and n_null >= 2 and se_df == n_null - 1)
    return False  # pragma: no cover - every rule is handled above


class Admission(NamedTuple):
    """Whether a status direction is licensed for an item (registry
    ``admitted_for_present`` / ``admitted_for_absent``) and, when not, the
    detail of ``ESTIMATOR_NOT_VALIDATED``."""

    present: bool
    absent: bool
    present_kind: str = R.NOT_ADMITTED
    absent_kind: str = R.NOT_ADMITTED


def _admission_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value), R.NOT_ADMITTED
    v = str(value).strip().lower()
    if v not in ADMISSION_VALUES:
        raise ValueError(f"admission value must be one of {ADMISSION_VALUES} or a "
                         f"bool, got {value!r}")
    if v == "not_observable":
        return False, R.NOT_OBSERVABLE
    return v in ("yes", "vacuous"), R.NOT_ADMITTED


def registry_admission(registry, principle, ev, regime=None) -> Optional[Admission]:
    """
    The admission of an item by ``registry``: an object with
    ``admission(principle, estimator, substrate, grain, **regime)`` returning
    ``{"present": .., "absent": ..}`` (values of :data:`ADMISSION_VALUES` or
    bools; ``vacuous`` admits; :class:`~impact_pipeline.registry.
    ApplicabilityRegistry`), or else one with ``is_validated`` (one decision
    for both directions). None without a registry.
    """
    if registry is None:
        return None
    item_regime = {**dict(regime or {}), **dict(ev.regime or {})}
    if hasattr(registry, "admission"):
        out = registry.admission(principle, ev.estimator, ev.substrate, ev.grain,
                                 **item_regime)
        if not isinstance(out, Mapping) or not {"present", "absent"} <= set(out):
            raise ValueError("registry.admission must return {'present', 'absent'}")
        p_ok, p_kind = _admission_value(out["present"])
        a_ok, a_kind = _admission_value(out["absent"])
        return Admission(p_ok, a_ok, p_kind, a_kind)
    if hasattr(registry, "is_validated"):
        ok, _why = registry.is_validated(principle, ev.estimator, ev.substrate,
                                         ev.grain, **item_regime)
        return Admission(bool(ok), bool(ok))
    raise TypeError("registry must provide admission() or is_validated()")


# The checks of one evidence item before the decision, in order, with the
# reason code each one gives (the first that applies decides).
CHECK_ORDER = (
    ("estimator_version", R.ESTIMATOR_NOT_VALIDATED),
    ("registry", R.ESTIMATOR_NOT_VALIDATED),
    ("null_family", R.NULL_FAMILY_MISMATCH),
    ("not_defined", R.NOT_DEFINED),
    ("non_finite_estimate", R.NON_FINITE_ESTIMATE),
    ("null_mean", R.NO_NULL_CALIBRATION),
    ("null_size", R.DEGENERATE_NULL),
    ("anchor", R.INVALID_ANCHORS),
    ("se", R.INVALID_SE),
    ("sampling_se", R.NO_SAMPLING_SE),
)


def _assess_member(ev, principle, lv: _Levels, *, pre_reason=None, se_allowed=None,
                   route=None, rank_gate=None, admission=None,
                   uncalibrated=frozenset()) -> TostAssessment:
    """The TOST assessment of one evidence item (one channel member)."""
    z, delta = lv.cutoff
    rule = lv.rule
    if pre_reason is not None:
        return _undefined_assessment(pre_reason, lv, ev)
    if not bool(ev.defined):
        reason = _vocab_reason(ev.reason, R.NOT_DEFINED)
        if reason == R.NOT_IMPLEMENTED:  # name the channel, as the interval rule does
            reason = R.format_reason(R.NOT_IMPLEMENTED, R.clean_detail(ev.channel))
        return _undefined_assessment(reason, lv, ev)
    if not math.isfinite(_as_float(ev.estimate)):
        return _undefined_assessment(
            _vocab_reason(ev.reason, R.NON_FINITE_ESTIMATE), lv, ev)
    a1 = component_assessment(ev, cutoff=lv.cutoff, alpha=lv.alpha_present)
    if a1.status is UNDEFINED and a1.reason not in (R.INCONCLUSIVE, R.NO_SAMPLING_SE):
        return _undefined_assessment(a1.reason, lv, ev, c=a1.c)
    if _se_contract_violation(ev, principle, se_allowed):
        return _undefined_assessment(R.INVALID_SE, lv, ev, c=a1.c)
    se_method = ev.se_method
    flags = []
    if a1.reason == R.NO_SAMPLING_SE:
        if se_method != SE_METHOD_CONCORDANT or route is None:
            return _undefined_assessment(R.NO_SAMPLING_SE, lv, ev, c=a1.c)
        absent_ok, present_ok = route
        c = float(a1.c)
        codes, ridx, lp, up, la, ua = _decide(c, 0.0, 0.0, 0.0, z, delta,
                                              rule.null_violation)
        code, reason = int(codes), _REASON_TABLE[int(ridx)]
        if (code == CODE_T and not present_ok) or (code == CODE_F and not absent_ok):
            return _undefined_assessment(R.NO_SAMPLING_SE, lv, ev, c=c)
        if code == CODE_T:
            flags.append(R.PRESENT_BY_CONCORDANCE)
        elif code == CODE_F:
            flags.append(R.ABSENT_BY_CONCORDANCE)
        se_c, df, q_p, q_a = 0.0, float("nan"), float("nan"), float("nan")
        parts = (0.0, float("nan"), float("nan"))
        route_name = ROUTE_CONCORDANCE
    else:
        exact = bool(ev.exact) and a1.se_sampling == 0
        # an exact value is decided on c alone (se_c = 0); the Monte-Carlo
        # error of the null mean and the reference SE stay reported in the
        # parts but do not enter the decision
        c, df = float(a1.c), float(a1.df)
        se_c = 0.0 if exact else float(a1.se)
        q_p, q_a = quantile(lv.alpha_present, df), quantile(lv.alpha_absent, df)
        codes, ridx, lp, up, la, ua = _decide(c, se_c, q_p, q_a, z, delta,
                                              rule.null_violation)
        code, reason = int(codes), _REASON_TABLE[int(ridx)]
        parts = (float(a1.se_sampling), float(a1.se_null), float(a1.se_reference))
        route_name = ROUTE_EXACT if exact else ROUTE_TOST
    # post-decision conditions, each of which can only remove a decision; the
    # rank gate concerns estimated values (an exact value has no null draws)
    if code == CODE_T and rank_gate is not None and route_name != ROUTE_EXACT:
        p = _as_float(ev.p_ind)
        if not math.isfinite(p):
            code, reason = CODE_U, R.NO_NULL_CALIBRATION
        elif p > rank_gate:
            code, reason = CODE_U, R.format_reason(R.INCONCLUSIVE, R.NULL_NOT_EXCEEDED)
    if admission is not None:
        if code == CODE_T and not admission.present:
            code, reason = CODE_U, R.not_validated(admission.present_kind)
        elif code == CODE_F and not admission.absent:
            code, reason = CODE_U, R.not_validated(admission.absent_kind)
    if code == CODE_F and (principle, se_method) in uncalibrated:
        code, reason = CODE_U, R.SE_NOT_CALIBRATED
    if code != CODE_T:
        flags = [f for f in flags if f != R.PRESENT_BY_CONCORDANCE]
    if code != CODE_F:
        flags = [f for f in flags if f != R.ABSENT_BY_CONCORDANCE]
    if (code == CODE_U and rule.present_reachability_flag and math.isfinite(q_p)
            and q_p * se_c >= 1.0 - z):
        flags.append(R.PRESENT_NOT_REACHABLE)
    lower_p, upper_p = float(lp), float(up)
    lower_a, upper_a = float(la), float(ua)
    return TostAssessment(
        _CODE_TO_STATUS[code], reason, tuple(flags), c=c, se=se_c,
        se_sampling=parts[0], se_null=parts[1], se_reference=parts[2], df=df,
        q_present=q_p, q_absent=q_a, lower=lower_p, upper=upper_p,
        lower_absent=lower_a, upper_absent=upper_a,
        margin_present=lower_p - z,
        margin_absent=min(delta - upper_a, lower_a + delta),
        cutoff_present=z, cutoff_absent=delta, alpha=lv.alpha_present,
        alpha_absent=lv.alpha_absent, route=route_name, se_method=se_method,
        channel=ev.channel, direction=ev.direction, scale=lv.scale,
    )


def _combine_members(members, lv: _Levels, channel) -> TostAssessment:
    """The status of a directional channel from its member assessments, in
    declared direction order (see :func:`status_c_directional`)."""
    z, delta = lv.cutoff
    codes = [_STATUS_TO_CODE[m.status] for m in members]
    code = int(directional_codes(np.asarray(codes, dtype=np.int8)))
    reason = None
    flags = []
    if code == CODE_U:
        validity = [m.reason for m in members
                    if m.status is UNDEFINED and m.reason not in _DECISION_REASONS]
        if validity:
            reason = validity[0]
        elif lv.rule.null_violation and any(
                m.reason == R.NULL_MODEL_VIOLATED for m in members):
            reason = R.NULL_MODEL_VIOLATED
        elif all(math.isfinite(m.q_absent * m.se) and m.q_absent * m.se >= delta
                 for m in members):
            reason = R.ABSENT_NOT_REACHABLE
        else:
            reason = R.INCONCLUSIVE
        if (not validity and lv.rule.present_reachability_flag and any(
                math.isfinite(m.q_present * m.se) and m.q_present * m.se >= 1.0 - z
                for m in members)):
            flags.append(R.PRESENT_NOT_REACHABLE)
    # the reported value is min over the directions: undefined when a
    # direction has no value, except for a decided status, which always has
    # a finite member (ABSENT through one direction) and reports the
    # smallest finite one
    finite = [m for m in members if math.isfinite(m.c)]
    if finite and (len(finite) == len(members) or code != CODE_U):
        arg = min(finite, key=lambda m: m.c)
    else:
        arg = None
    m_p = [m.margin_present for m in members]
    m_a = [m.margin_absent for m in members if math.isfinite(m.margin_absent)]
    nan = float("nan")
    return TostAssessment(
        _CODE_TO_STATUS[code], reason, tuple(flags),
        c=(arg.c if arg is not None else nan),
        se=(arg.se if arg is not None else nan),
        se_sampling=(arg.se_sampling if arg is not None else nan),
        se_null=(arg.se_null if arg is not None else nan),
        se_reference=(arg.se_reference if arg is not None else nan),
        df=(arg.df if arg is not None else nan),
        q_present=(arg.q_present if arg is not None else nan),
        q_absent=(arg.q_absent if arg is not None else nan),
        lower=(arg.lower if arg is not None else nan),
        upper=(arg.upper if arg is not None else nan),
        lower_absent=(arg.lower_absent if arg is not None else nan),
        upper_absent=(arg.upper_absent if arg is not None else nan),
        margin_present=(min(m_p) if all(math.isfinite(v) for v in m_p) else nan),
        margin_absent=(max(m_a) if m_a else nan),
        cutoff_present=z, cutoff_absent=delta, alpha=lv.alpha_present,
        alpha_absent=lv.rule.alpha_absent, route=ROUTE_DIRECTIONAL,
        se_method=(arg.se_method if arg is not None else members[0].se_method),
        channel=channel, direction=None, members=tuple(members), scale=lv.scale,
    )


# --------------------------------------------------------------------------
# TOST rule: items, principles and the verdict
# --------------------------------------------------------------------------
def with_protocol_reference(ev, protocol):
    """The item with the anchor of the protocol: an external reference
    replaces the item's own (no matching key: no anchor), a pending reference
    removes it, a cohort reference keeps the item's (computed per cohort)."""
    proto = _tost_protocol(protocol)
    kind = proto.reference.get("kind")
    if kind == "cohort_high_state":
        return ev
    ref = {} if kind == REFERENCE_PENDING else proto.reference_for(
        ev.principle, ev.channel, ev.direction)
    if not ref:
        return dataclasses.replace(ev, reference=None, reference_se=None)
    return dataclasses.replace(ev, reference=ref["reference"],
                               reference_se=ref["reference_se"],
                               reference_scale=ref["reference_scale"])


def _uncalibrated_set(pairs):
    out = set()
    for pair in pairs or ():
        p, method = pair
        out.add((_principle_key(p, "uncalibrated_se_methods"), str(method)))
    return frozenset(out)


def _item_pre_reason(ev, principle, proto, admission):
    want = proto.estimator_version_for(principle)
    if want is not None and split_estimator(ev.estimator)[1] != want:
        return R.not_validated(R.NOT_ADMITTED)
    if admission is not None and not (admission.present or admission.absent):
        kind = (R.NOT_OBSERVABLE if R.NOT_OBSERVABLE in (
            admission.present_kind, admission.absent_kind) else R.NOT_ADMITTED)
        return R.not_validated(kind)
    family = proto.null_families.get(principle)
    used = ev.null_family
    if family is not None and used is not None and str(used) != family:
        return R.format_reason(R.NULL_FAMILY_MISMATCH,
                               R.clean_detail(f"{family}/{ev.null_family}"))
    return None


def _scale_cutoff(proto, principle, scale):
    cut = proto.cutoff_for(principle)
    if scale == "construct":
        return cut
    if scale != "amplitude":
        raise ValueError(f"scale must be one of {SCALES}")
    return amplitude_cutoff(cut) if is_quadratic(
        proto.estimator_version_for(principle)) else cut


def _item_levels(proto, principle, rule, n_channels, scale):
    dirs = proto.directions_for(principle)
    a_p, a_a = member_levels(rule, n_channels, len(dirs) if dirs else 1)
    return _Levels(_scale_cutoff(proto, principle, scale), rule, a_p, a_a, scale)


def _assess_one(ev, principle, proto, lv, *, registry, regime, uncalibrated):
    ev = with_protocol_reference(ev, proto)
    admission = registry_admission(registry, principle, ev, regime)
    pre = _item_pre_reason(ev, principle, proto, admission)
    route = None
    if ev.se_method == SE_METHOD_CONCORDANT:
        route = proto.concordance_admission(
            principle, ev.substrate, ev.observation_stage, ev.view,
            ev.content_bearer)
    return _assess_member(
        ev, principle, lv, pre_reason=pre, se_allowed=proto.se_methods_for(principle),
        route=route, rank_gate=proto.rank_gate_for(principle), admission=admission,
        uncalibrated=uncalibrated)


def assess_item(ev, protocol, *, registry=None, regime=None, n_channels=None,
                uncalibrated_se_methods=(), status_rule=None,
                scale="construct") -> TostAssessment:
    """
    The TOST assessment of one evidence item under a TOST protocol, at the levels
    the protocol implies for it (``alpha / k_channels`` for PRESENT,
    ``alpha_A / k_directions`` for the TOST of a direction member). Order of
    the checks: estimator version, registry (no direction admitted), null
    family, the validity checks (estimator not defined, non-finite
    estimate, null, anchors, SE), the SE-method contract, no sampling SE
    (or the concordance route), the decision, then the conditions that can
    only remove a decision: the rank gate, the registry direction, the SE
    reversion (``uncalibrated_se_methods``: ``(principle, se_method)``
    pairs found anti-conservative on the twins).
    """
    proto = _tost_protocol(protocol)
    p = _principle_key(ev.principle, "evidence")
    rule = proto.status_rule if status_rule is None else _coerce_rule(status_rule)
    if n_channels is None:
        n_channels = len(proto.channels_for(p) or (ev.channel,))
    lv = _item_levels(proto, p, rule, n_channels, scale)
    return _assess_one(ev, p, proto, lv, registry=registry, regime=regime,
                       uncalibrated=_uncalibrated_set(uncalibrated_se_methods))


@dataclass(frozen=True)
class PrincipleAssessment:
    """
    The status of one principle under a TOST protocol: the Kleene OR over its
    channels (a directional channel combined over its directions first),
    the vocabulary reason (UNDEFINED only) and flags of the deciding channel,
    the verdict reason codes (``KIND:P[:detail]``), the per-channel statuses,
    reasons and assessments, the largest PRESENT margin and the ignored
    channels.
    """

    principle: str
    status: ComponentStatus
    reason: Optional[str]
    flags: Tuple[str, ...]
    reasons: Tuple[str, ...]
    channels: Mapping
    channel_reasons: Mapping
    assessments: Mapping
    deciding: Optional[TostAssessment]
    margin: float
    ignored: Tuple[str, ...] = ()
    kept: Tuple = field(default=(), repr=False)

    def __post_init__(self):
        R.check_status(self.status.value, self.reason, self.flags)

    @property
    def c(self) -> float:
        return self.deciding.c if self.deciding is not None else float("nan")

    def record_fields(self) -> dict:
        """The status fields of a bench result component record:
        ``status``, ``reason``, ``flags``, ``c`` (NAS: the smaller direction),
        ``c_R``/``c_B`` (NAS directions), ``se_c`` and ``df_c``."""
        d = self.deciding

        def _f(v):
            return None if v is None or not math.isfinite(v) else float(v)

        out = {"status": self.status.value, "reason": self.reason,
               "flags": list(self.flags), "c": None, "se_c": None, "df_c": None}
        # the direction fields are always present for a directional principle
        # (None without a value), so every NAS record has the same keys
        for direction, name in DIRECTION_FIELDS.get(self.principle, {}).items():
            out[name] = None if d is None else _f(d.directions.get(direction))
        if d is not None:
            out.update(c=_f(d.c), se_c=_f(d.se), df_c=_f(d.df))
        if self.status is not UNDEFINED and out["c"] is None:  # pragma: no cover
            raise AssertionError("a decided status carries a finite c")
        return out


def _verdict_code(principle, reason) -> str:
    code, detail = R.parse_reason(reason)
    return f"{code}:{principle}" + (f":{detail}" if detail else "")


def _group_items(principle, items, proto):
    """Items per declared channel (and direction); duplicates and items that
    do not fit the protocol's directions are refused."""
    declared = proto.channels_for(principle)
    dirs = proto.directions_for(principle)
    kept, ignored, by_key = [], [], {}
    for ev in items:
        direction = ev.direction
        if dirs is None and direction is not None:
            raise ValueError(f"{principle}: the protocol declares no directions, but "
                             f"an item has direction {direction!r}")
        if dirs is not None and direction is None:
            raise ValueError(f"{principle}: the protocol declares directions {dirs}; "
                             "every item needs one")
        if declared is not None and ev.channel not in declared:
            ignored.append(ev.channel)
            continue
        if dirs is not None and direction not in dirs:
            ignored.append(f"{ev.channel}:{direction}")
            continue
        key = (ev.channel, direction)
        if key in by_key:
            raise ValueError(f"{principle}: more than one item for channel "
                             f"{ev.channel!r}" + (f", direction {direction!r}"
                                                  if direction else ""))
        by_key[key] = ev
        kept.append(ev)
    channels = list(declared) if declared is not None else list(
        dict.fromkeys(ev.channel for ev in kept))
    return channels, by_key, kept, tuple(sorted(set(ignored)))


def assess_principle(principle, items, protocol, *, registry=None, regime=None,
                     uncalibrated_se_methods=(), status_rule=None,
                     scale="construct") -> PrincipleAssessment:
    """The status of one principle from its evidence items under a TOST
    protocol (see :class:`PrincipleAssessment` and :func:`assess_item`)."""
    proto = _tost_protocol(protocol)
    p = _principle_key(principle, "principle")
    rule = proto.status_rule if status_rule is None else _coerce_rule(status_rule)
    uncal = _uncalibrated_set(uncalibrated_se_methods)
    items = list(items or ())
    for ev in items:
        if not isinstance(ev, ComponentEvidence):
            raise TypeError(f"evidence for {p} must be ComponentEvidence")
        if ev.principle != p:
            raise ValueError(
                f"evidence keyed {p!r} declares principle {ev.principle!r}")
    channels, by_key, kept, ignored = _group_items(p, items, proto)
    flags_extra = ()
    if proto.anchor_status(p) == "valid_nonspecific":
        flags_extra = (R.NONSPECIFIC_ANCHOR,)
    if not channels:
        return PrincipleAssessment(
            p, UNDEFINED, R.MISSING, flags_extra, (f"{R.MISSING}:{p}",), {}, {}, {},
            None, float("nan"), ignored, tuple(kept))
    lv = _item_levels(proto, p, rule, len(channels), scale)
    dirs = proto.directions_for(p)
    statuses, ch_reasons, assessments, u_codes = {}, {}, {}, []
    for ch in channels:
        if dirs is None:
            ev = by_key.get((ch, None))
            if ev is None:
                a = _undefined_assessment(
                    R.format_reason(R.MISSING_CHANNEL, R.clean_detail(ch)), lv,
                    channel=ch)
            else:
                a = _assess_one(ev, p, proto, lv, registry=registry, regime=regime,
                                uncalibrated=uncal)
        else:
            members = []
            for d in dirs:
                ev = by_key.get((ch, d))
                if ev is None:
                    members.append(_undefined_assessment(
                        R.format_reason(R.MISSING_CHANNEL, R.clean_detail(f"{ch}:{d}")),
                        lv, channel=ch, direction=d))
                else:
                    members.append(_assess_one(ev, p, proto, lv, registry=registry,
                                               regime=regime, uncalibrated=uncal))
            a = _combine_members(members, lv, ch)
        assessments[ch] = a
        statuses[ch] = a.status
        if a.status is UNDEFINED:
            ch_reasons[ch] = a.reason
            u_codes.append(_verdict_code(p, a.reason))
    code = max(_STATUS_TO_CODE[s] for s in statuses.values())
    status = _CODE_TO_STATUS[code]

    def _key(a):
        m = a.margin_present
        return (_STATUS_TO_CODE[a.status], m if math.isfinite(m) else -math.inf)

    deciding = max(assessments.values(), key=_key)
    finite = [a.margin_present for a in assessments.values()
              if math.isfinite(a.margin_present)]
    margin = max(finite) if finite else float("nan")
    if code == CODE_T:
        reasons = ()
    elif code == CODE_F:
        reasons = (f"{REASON_ABSENT}:{p}",)
    else:
        reasons = tuple(dict.fromkeys(u_codes))
    reason = deciding.reason if status is UNDEFINED else None
    flags = tuple(deciding.flags) + flags_extra
    return PrincipleAssessment(
        p, status, reason, flags, reasons, statuses, ch_reasons, assessments,
        deciding, margin, ignored, tuple(kept))


@dataclass
class TostVerdict(MPCVerdict):
    """
    Result of :func:`tost_verdict`: the fields of :class:`MPCVerdict` (the
    verdict over the
    protocol's necessity set N_anch, the per-principle statuses, margins,
    channels, reasons and assessments) plus the per-principle vocabulary
    reasons, flags and construct record (``c``, ``c_R``, ``c_B``, ``se_c``,
    ``df_c``), the declared necessity set and the verdict over it
    (descriptive), the anchor fallback, the status rule used and whether it
    differs from the protocol's (a sensitivity analysis) and the scale.
    """

    principle_reasons: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    construct: dict = field(default_factory=dict)
    declared_necessity_set: tuple = PRINCIPLES
    verdict_declared: Optional[Verdict] = None
    reasons_declared: list = field(default_factory=list)
    fallback: Optional[str] = None
    status_rule: dict = field(default_factory=dict)
    sensitivity: bool = False
    scale: str = "construct"
    schema: str = PROTOCOL_SCHEMA

    def to_dict(self) -> dict:
        out = super().to_dict()
        out.update({
            "principle_reasons": dict(self.principle_reasons),
            "flags": {p: list(v) for p, v in self.flags.items()},
            "construct": {p: dict(v) for p, v in self.construct.items()},
            "declared_necessity_set": list(self.declared_necessity_set),
            "verdict_declared": (None if self.verdict_declared is None
                                 else self.verdict_declared.value),
            "reasons_declared": list(self.reasons_declared),
            "fallback": self.fallback,
            "status_rule": dict(self.status_rule),
            "sensitivity": self.sensitivity,
            "scale": self.scale,
            "schema": self.schema,
        })
        return out


def _global_reasons(gated, proto, joint_dependence, bearer_coherence,
                    require_same_protocol):
    out = []
    if proto.source_rule in ("single_source", "same_bearer") and len(
            {e.bearer_id for e in gated} - {None}) > 1:
        out.append(REASON_BEARER_MISMATCH)
    if bearer_coherence is not None and not _coherence_ok(bearer_coherence):
        out.append(f"{REASON_BEARER_MISMATCH}:COHERENCE")
    if require_same_protocol:
        ids = {e.protocol_id for e in gated} - {None}
        if len(ids) > 1 or (ids and ids != {proto.protocol_id}):
            out.append(REASON_PROTOCOL_MISMATCH)
    if proto.source_rule == "single_source":
        why = _source_reason(gated, joint_dependence)
        if why is not None:
            out.append(why)
    return out


def tost_verdict(evidence: Mapping, protocol, *, registry=None, regime=None,
                 joint_dependence=None, bearer_coherence=None,
                 require_same_protocol=True, uncalibrated_se_methods=(),
                 status_rule=None, scale="construct",
                 joint_dependence_declared=None) -> TostVerdict:
    """
    The three-valued verdict under a TOST protocol: every principle with
    evidence or in the declared necessity set is assessed
    (:func:`assess_principle`); the verdict is the strong-Kleene AND over the
    protocol's necessity set (N_anch), forced to UNDETERMINED by a global
    code (``BEARER_MISMATCH``, ``PROTOCOL_MISMATCH``, ``SOURCE_INCOHERENT``,
    as in :func:`interval_verdict`); the AND over the declared set is
    reported as
    ``verdict_declared`` (descriptive; ``joint_dependence_declared`` is the
    joint-dependence result over its node sets). ``status_rule`` replaces
    the protocol's rule for a sensitivity analysis (for example
    ``alpha / |N_anch|``); ``scale="amplitude"`` re-judges the quadratic
    statistics on the amplitude scale.
    """
    proto = _tost_protocol(protocol)
    rule = proto.status_rule if status_rule is None else _coerce_rule(status_rule)
    items_by_p = {}
    for key, items in dict(evidence or {}).items():
        p = _principle_key(key, "evidence")
        if p != key:
            raise ValueError(f"evidence key {key!r} must be a canonical principle name")
        items_by_p[p] = [items] if isinstance(items, ComponentEvidence) else list(
            items or [])
    nset, declared = proto.necessity_set, proto.declared_necessity_set
    wanted = set(items_by_p) | set(nset) | set(declared)
    ordered = [p for p in PRINCIPLES if p in wanted]
    pas = {
        p: assess_principle(p, items_by_p.get(p, []), proto, registry=registry,
                            regime=regime,
                            uncalibrated_se_methods=uncalibrated_se_methods,
                            status_rule=rule, scale=scale)
        for p in ordered
    }

    def _and(principles, joint):
        gated = [ev for p in principles for ev in pas[p].kept]
        glob = _global_reasons(gated, proto, joint, bearer_coherence,
                               require_same_protocol)
        code = min(_STATUS_TO_CODE[pas[p].status] for p in principles)
        if glob:
            code = CODE_U
        reasons = glob + [r for p in principles for r in pas[p].reasons]
        return _CODE_TO_VERDICT[code], reasons

    verdict, reasons = _and(nset, joint_dependence)
    if tuple(declared) == tuple(nset):
        v_decl, r_decl = verdict, list(reasons)
    else:
        v_decl, r_decl = _and(declared, joint_dependence_declared)
    return TostVerdict(
        verdict=verdict,
        reasons=reasons,
        component_status={p: pa.status for p, pa in pas.items()},
        margins={p: pa.margin for p, pa in pas.items()},
        channels={p: dict(pa.channels) for p, pa in pas.items()},
        necessity_set=nset,
        channel_reasons={p: dict(pa.channel_reasons) for p, pa in pas.items()},
        assessments={p: dict(pa.assessments) for p, pa in pas.items()},
        principle_assessment={p: pa.deciding for p, pa in pas.items()},
        ignored_channels={p: list(pa.ignored) for p, pa in pas.items() if pa.ignored},
        protocol_hash=proto.hash,
        principle_reasons={p: pa.reason for p, pa in pas.items()},
        flags={p: tuple(pa.flags) for p, pa in pas.items()},
        construct={p: pa.record_fields() for p, pa in pas.items()},
        declared_necessity_set=declared,
        verdict_declared=v_decl,
        reasons_declared=r_decl,
        fallback=proto.fallback,
        status_rule=rule.to_dict(),
        sensitivity=rule != proto.status_rule,
        scale=scale,
    )


# --------------------------------------------------------------------------
# MPC degree
# --------------------------------------------------------------------------
def two_anchor_normalize(value, null_anchor, reference):
    """
    Two-anchor normalisation ``(value - null) / (reference - null)``: 0 at the
    null anchor, 1 at the reference. NaN when any input is non-finite or the
    reference does not lie above the null. Works element-wise on arrays.
    """
    v, n, r = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (value, null_anchor, reference))
    )
    ok = np.isfinite(v) & np.isfinite(n) & np.isfinite(r) & (r > n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(ok, (v - n) / np.where(ok, r - n, 1.0), np.nan)
    return float(out) if out.ndim == 0 else out


def _components_and_weights(c, weights):
    if isinstance(c, Mapping):
        names = list(c.keys())
        vals = np.asarray([_as_float(c[k]) for k in names], dtype=float)
    else:
        vals = np.asarray([_as_float(v) for v in c], dtype=float)
        names = list(range(vals.size))
    if weights is None:
        w = np.ones(vals.size, dtype=float)
    elif isinstance(weights, Mapping):
        w = np.asarray([_as_float(weights.get(k, 0.0)) for k in names], dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != vals.size:
            raise ValueError("weights must align with the components")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    keep = w > 0
    if not np.any(keep):
        raise ValueError("at least one weight must be positive")
    return [n for n, k in zip(names, keep) if k], vals[keep], w[keep] / w[keep].sum()


def _check_cap_p(p, cap):
    if cap is not None and not (float(cap) > 0):
        raise ValueError("cap must be > 0 (or None for no cap)")
    if math.isnan(float(p)):
        raise ValueError("p must not be NaN")


def _power_mean_rows(x, w, p, cap):
    """Weighted power mean of each row of ``x`` (finite, floored at 0, capped)."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    if cap is not None and math.isfinite(float(cap)):
        x = np.minimum(x, float(cap))
    p = float(p)
    if p == -math.inf:
        return x.min(axis=-1)
    if p == math.inf:
        return x.max(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if p == 0.0:
            return np.exp(np.log(x) @ w)
        # Scale each row by the extreme that dominates the mean (max for p > 0,
        # min for p < 0): the ratios then lie in [0, 1] (p > 0) or [1, inf)
        # (p < 0), so ratio**p stays in [0, 1] and the weighted sum is at least
        # the positive weight of the extreme; no overflow/underflow for large
        # |p|. A zero row extreme gives 0 (all zero for p > 0; any zero for p < 0).
        m = x.max(axis=-1) if p > 0 else x.min(axis=-1)
        pos = m > 0
        m_safe = np.where(pos, m, 1.0)
        val = m_safe * (((x / m_safe[..., None]) ** p) @ w) ** (1.0 / p)
        return np.where(pos, val, 0.0)


def degree(c, weights=None, *, p=0.0, cap=1.0) -> float:
    """
    MPC degree: weighted power mean of ``min(c_j, cap)`` over the components
    with positive weight (equal weights by default). ``p=0`` is the geometric
    mean, ``p=-inf`` the weakest link, ``p=1`` the arithmetic mean;
    ``cap=None`` disables the cap. Components are construct-scale values
    (0 = null, 1 = reference); values below 0 count as 0. NaN if any weighted
    component is non-finite. Only meaningful for MPC_CONSISTENT verdicts: a
    reference-relative evidence summary, not a level of consciousness.
    """
    _check_cap_p(p, cap)
    _, vals, w = _components_and_weights(c, weights)
    if not np.all(np.isfinite(vals)):
        return float("nan")
    return float(_power_mean_rows(vals[None, :], w, p, cap)[0])


def weakest_link(c, weights=None, *, cap=1.0) -> float:
    """Weakest-link bound: ``degree(c, weights, p=-inf, cap=cap)``."""
    return degree(c, weights, p=-math.inf, cap=cap)


def _degree_gradient(vals, w, p, cap, d):
    """
    Gradient of the capped, floored power mean with respect to the raw
    components (chain rule through ``min(max(c, 0), cap)``): components at or
    above the cap and strictly below the null (floored at 0) have zero
    derivative; a component exactly at 0 with ``p <= 0`` gives NaN (the power
    mean is not differentiable there).
    """
    capped = (cap is not None) and math.isfinite(float(cap))
    active = (vals < float(cap)) if capped else np.ones(vals.size, dtype=bool)
    active = active & ~(vals < 0.0)
    x = np.maximum(vals, 0.0)
    grad = np.zeros(vals.size, dtype=float)
    if p in (-math.inf, math.inf):
        xc = np.minimum(x, float(cap)) if capped else x
        j = int(np.argmin(xc) if p == -math.inf else np.argmax(xc))
        grad[j] = 1.0 if active[j] else 0.0
        return grad
    # dM/dx_j = w_j (x_j / M)**(p - 1) (p = 0: w_j M / x_j), written as a
    # ratio so that large |p| does not overflow.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        g = w * (x / d) ** (p - 1.0)
    return np.where(active, g, 0.0)


def degree_interval(
    c,
    se=None,
    weights=None,
    *,
    p=0.0,
    cap=1.0,
    level=0.95,
    method="delta",
    n_boot=2000,
    seed=0,
    samples=None,
) -> dict:
    """
    Interval for the MPC degree.

    - ``method="delta"``: first-order delta method with independent component
      SEs ``se`` (mapping or sequence aligned with ``c``); the cap and the
      floor at 0 are treated as flat (zero derivative for components at or
      above the cap or strictly below 0) and the weakest link differentiates
      through its (first) minimising component. The SE is NaN where the power
      mean is not differentiable (a component exactly at 0 with ``p <= 0``);
      a component strictly below 0 with ``p <= 0`` pins the degree at 0 and
      gives SE 0 (use the bootstrap for the chance of leaving that floor).
    - ``method="bootstrap"``: percentile interval of the degree over bootstrap
      replicates of the components: ``samples`` (``B x J`` array aligned with
      ``c``, or a mapping of arrays), or else ``n_boot`` parametric draws
      ``N(c_j, se_j)`` from ``numpy.random.default_rng(seed)``.

    Returns ``{"degree", "lo", "hi", "se", "method", "level", "n_boot"}``.
    """
    _check_cap_p(p, cap)
    if not (0.0 < float(level) < 1.0):
        raise ValueError("level must be in (0, 1)")
    names, vals, w = _components_and_weights(c, weights)
    d = degree(dict(zip(names, vals)), dict(zip(names, w)), p=p, cap=cap)
    all_names = list(c.keys()) if isinstance(c, Mapping) else list(range(len(c)))
    if se is None:
        se_vals = np.zeros(vals.size)
    elif isinstance(se, Mapping):
        se_vals = np.asarray([_as_float(se.get(k, np.nan)) for k in names], dtype=float)
    else:
        se_all = np.asarray(se, dtype=float).reshape(-1)
        if se_all.size != len(all_names):
            raise ValueError("se must align with the components")
        se_vals = se_all[[all_names.index(k) for k in names]]
    out = {"degree": d, "lo": float("nan"), "hi": float("nan"), "se": float("nan"),
           "method": str(method), "level": float(level), "n_boot": 0}
    if not math.isfinite(d):
        return out
    if method == "delta":
        grad = _degree_gradient(vals, w, float(p), cap, d)
        var = float(np.sum(grad ** 2 * se_vals ** 2))
        if math.isfinite(var):
            s = math.sqrt(var)
            z = float(NormalDist().inv_cdf(0.5 + 0.5 * float(level)))
            out.update(se=s, lo=d - z * s, hi=d + z * s)
        return out
    if method != "bootstrap":
        raise ValueError("method must be 'delta' or 'bootstrap'")
    if samples is not None:
        if isinstance(samples, Mapping):
            boot = np.column_stack([np.asarray(samples[k], dtype=float) for k in names])
        else:
            arr = np.asarray(samples, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != len(all_names):
                raise ValueError("samples must be a (B x J) array aligned with c")
            boot = arr[:, [all_names.index(k) for k in names]]
    else:
        if not np.all(np.isfinite(se_vals)):
            return out
        rng = np.random.default_rng(seed)
        boot = vals[None, :] + rng.standard_normal((int(n_boot), vals.size)) * se_vals
    boot = boot[np.all(np.isfinite(boot), axis=1)]
    if boot.shape[0] < 2:
        return out
    degs = _power_mean_rows(boot, w, p, cap)
    degs = degs[np.isfinite(degs)]
    if degs.size < 2:
        return out
    q = (0.5 - 0.5 * float(level), 0.5 + 0.5 * float(level))
    lo, hi = np.quantile(degs, q)
    out.update(lo=float(lo), hi=float(hi), se=float(np.std(degs, ddof=1)),
               n_boot=int(degs.size))
    return out


def verdict_stability(verdicts, reference=None) -> dict:
    """
    Stability of a verdict across resamples (bootstrap, seeds, reasonable
    parameter settings). ``modal_verdict`` is the unique most frequent
    verdict; a tie for the top count gives UNDETERMINED (no determinate
    verdict holds for a majority of resamples). ``flip_rate`` is the fraction
    of verdicts that differ from the modal one. With ``reference`` (e.g. the
    verdict on the original data) the fraction differing from it is returned
    as ``reference_flip_rate``.
    """
    vs = []
    for v in verdicts:
        v = v.verdict if isinstance(v, MPCVerdict) else v
        vs.append(Verdict(v.value if isinstance(v, Enum) else str(v)))
    counts = {v.value: 0 for v in Verdict}
    for v in vs:
        counts[v.value] += 1
    out = {"n": len(vs), "counts": counts, "modal_verdict": None,
           "flip_rate": float("nan")}
    if vs:
        top = max(counts.values())
        leaders = [v for v in Verdict if counts[v.value] == top]
        modal = leaders[0] if len(leaders) == 1 else Verdict.UNDETERMINED
        out["modal_verdict"] = modal.value
        out["flip_rate"] = 1.0 - counts[modal.value] / len(vs)
    if reference is not None:
        ref = reference.verdict if isinstance(reference, MPCVerdict) else reference
        ref = Verdict(ref.value if isinstance(ref, Enum) else str(ref))
        out["reference"] = ref.value
        out["reference_flip_rate"] = (
            sum(v != ref for v in vs) / len(vs) if vs else float("nan")
        )
    return out


# --------------------------------------------------------------------------
# single-source constraint: joint dependence of the component node sets
# --------------------------------------------------------------------------
def _set_signal(ts, idx, max_dim):
    x = np.asarray(ts[idx], dtype=float)
    sd = x.std(axis=1)
    x = x[sd > 0]
    if x.shape[0] == 0:
        return None
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    if max_dim is not None and x.shape[0] > int(max_dim):
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        x = (u[:, : int(max_dim)].T @ x)
    return x


def _logdet(c, ridge):
    dim = c.shape[0]
    c = c + float(ridge) * (np.trace(c) / dim) * np.eye(dim)
    return float(np.linalg.slogdet(c)[1])


def _embed(sig, lag):
    """Rows ``[s(t), s(t-1), .., s(t-lag)]`` for ``t = lag .. T-1``."""
    t = sig.shape[1]
    return np.vstack([sig[:, lag - k: t - k] for k in range(lag + 1)])


def _total_correlation(blocks, lag, ridge):
    """
    Gaussian total correlation (bits) between lag-embedded blocks, and each
    block's multi-information with all other blocks.
    """
    emb = [_embed(b, lag) for b in blocks]
    sizes = [e.shape[0] for e in emb]
    c = np.atleast_2d(np.cov(np.vstack(emb)))
    edges = np.cumsum([0] + sizes)
    ld_blocks = [
        _logdet(c[edges[i]:edges[i + 1], edges[i]:edges[i + 1]], ridge)
        for i in range(len(emb))
    ]
    ld_all = _logdet(c, ridge)
    tc = 0.5 * (sum(ld_blocks) - ld_all) / math.log(2.0)
    each = []
    for i in range(len(emb)):
        rest = np.r_[0:edges[i], edges[i + 1]:edges[-1]]
        ld_rest = _logdet(c[np.ix_(rest, rest)], ridge)
        each.append(max(0.5 * (ld_blocks[i] + ld_rest - ld_all) / math.log(2.0), 0.0))
    return max(tc, 0.0), each


def joint_dependence(
    ts,
    node_sets,
    lag=1,
    n_surrogates=200,
    seed=0,
    *,
    n_components=1,
    min_shift=None,
    alpha=0.05,
    criterion="total",
    overlap="disjoint",
    ridge=1e-6,
) -> dict:
    """
    Single-source constraint: joint dependence above null between the node
    sets that feed the components.

    Each node set (``node_sets``: mapping name -> node indices, e.g. the
    per-component bearer nodes, or a sequence of index lists; identical sets
    are merged) is summarised by the first ``n_components`` principal
    components of its z-scored nodes and lag-embedded
    (``[s(t), .., s(t-lag)]``, so contemporaneous and lagged cross-set
    dependence both count). The statistic is the Gaussian total correlation
    (multi-information, bits) ``0.5 [sum_i log det C_ii - log det C]`` of the
    embedded blocks. The null independently circular-shifts each block's
    summary (block 0 is the reference; shifts of at least ``min_shift``
    samples, default 10% of the run): between-set dependence is destroyed and
    within-set structure kept. ``p = (1 + #{null >= obs}) / (K + 1)`` with
    ``numpy.random.default_rng(seed)``.

    Overlapping node sets (``OVERLAPPING_NODE_SETS`` in ``flags``) are
    computed on disjoint parts (``overlap="disjoint"``: nodes are grouped by
    the sets that contain them, and those atoms are the blocks), or the test
    is refused (``overlap="flag"``: not dependent, reason
    ``OVERLAPPING_NODE_SETS``). ``criterion="total"`` requires the total
    correlation above null; ``criterion="each"`` requires each block's
    multi-information with the rest above null (Holm-corrected), which a
    partly coupled patchwork does not pass. Per-block results are reported
    either way.

    Returns ``{"dependent", "tc_bits", "null_mean", "null_sd", "z", "p",
    "sets", "set_names", "blocks", "per_block", "flags", "reason", ...}``;
    ``sets`` are the distinct node sets (sorted index lists) that the verdict
    checks against its evidence; ``reason`` is None when dependent, else
    ``SOURCE_INCOHERENT``, ``OVERLAPPING_NODE_SETS``, ``NO_SURROGATES``,
    ``INSUFFICIENT_SURROGATES`` (the smallest attainable p, ``1 / (K + 1)``,
    exceeds the level the criterion needs: ``alpha``, or ``alpha / blocks``
    for ``"each"``; ``min_surrogates`` is the K it needs) or
    ``DEGENERATE_NODE_SET:<name>``.
    """
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"ts must be 2D (n_nodes x n_time), got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("joint_dependence needs a finite time series")
    if criterion not in ("total", "each"):
        raise ValueError("criterion must be 'total' or 'each'")
    if overlap not in ("disjoint", "flag"):
        raise ValueError("overlap must be 'disjoint' or 'flag'")
    alpha = _check_alpha(alpha)
    n_nodes, t = x.shape
    lag = int(lag)
    if lag < 0 or lag >= t - 2:
        raise ValueError("lag must be >= 0 and shorter than the run")
    if int(n_components) < 1:
        raise ValueError("n_components must be >= 1")
    if isinstance(node_sets, Mapping):
        named = [(str(k), v) for k, v in node_sets.items()]
    else:
        named = [(f"set{i}", v) for i, v in enumerate(node_sets)]
    sets, names = [], []
    for name, idx in named:
        s = _node_tuple(idx, f"node set {name!r}")
        if s[-1] >= n_nodes:
            raise ValueError(f"node set {name!r} is out of range")
        if s in sets:
            names[sets.index(s)].append(name)
            continue
        sets.append(s)
        names.append([name])
    out = {
        "dependent": True,
        "tc_bits": float("nan"),
        "null_mean": float("nan"),
        "null_sd": float("nan"),
        "z": float("nan"),
        "p": float("nan"),
        "sets": [list(s) for s in sets],
        "set_names": ["|".join(n) for n in names],
        "blocks": [],
        "per_block": [],
        "flags": [],
        "reason": None,
        "lag": lag,
        "n_surrogates": int(n_surrogates),
        "seed": seed,
        "alpha": alpha,
        "criterion": criterion,
        "n_components": int(n_components),
    }
    if len(sets) < 2:
        out["flags"].append("SINGLE_NODE_SET")
        return out
    overlapping = any(
        set(a) & set(b) for i, a in enumerate(sets) for b in sets[i + 1:]
    )
    if overlapping:
        out["flags"].append("OVERLAPPING_NODE_SETS")
        if overlap == "flag":
            out.update(dependent=False, reason="OVERLAPPING_NODE_SETS")
            return out
        member = {}
        for node in sorted(set().union(*map(set, sets))):
            sig = tuple(i for i, s in enumerate(sets) if node in s)
            member.setdefault(sig, []).append(node)
        blocks = [
            ("&".join(out["set_names"][i] for i in sig), tuple(nodes))
            for sig, nodes in sorted(member.items())
        ]
    else:
        blocks = list(zip(out["set_names"], sets))
    out["blocks"] = [{"name": n, "nodes": list(b)} for n, b in blocks]
    sig = []
    for name, idx in blocks:
        s = _set_signal(x, list(idx), int(n_components))
        if s is None:
            out.update(dependent=False, reason=f"DEGENERATE_NODE_SET:{name}")
            return out
        sig.append(s)
    if int(n_surrogates) < 1:
        out.update(dependent=False, reason="NO_SURROGATES")
        return out
    lo = int(math.ceil(0.1 * t)) if min_shift is None else int(min_shift)
    lo = max(1, lo)
    if t - lo < lo:
        raise ValueError("run too short for the circular-shift null")
    obs, obs_each = _total_correlation(sig, lag, ridge)
    rng = np.random.default_rng(seed)
    k = int(n_surrogates)
    null = np.empty(k)
    null_each = np.empty((k, len(sig)))
    for i in range(k):
        shifted = [sig[0]] + [
            np.roll(s, int(rng.integers(lo, t - lo + 1)), axis=1) for s in sig[1:]
        ]
        null[i], null_each[i] = _total_correlation(shifted, lag, ridge)
    mean = float(null.mean())
    sd = float(null.std(ddof=1)) if k > 1 else float("nan")
    z = (obs - mean) / sd if (math.isfinite(sd) and sd > 0) else float("nan")
    pval = float((1.0 + np.sum(null >= obs)) / (k + 1.0))
    per_block = []
    for j, (name, _idx) in enumerate(blocks):
        col = null_each[:, j]
        bsd = float(col.std(ddof=1)) if k > 1 else float("nan")
        per_block.append({
            "name": name,
            "mi_bits": float(obs_each[j]),
            "null_mean": float(col.mean()),
            "null_sd": bsd,
            "z": (
                float((obs_each[j] - col.mean()) / bsd)
                if (math.isfinite(bsd) and bsd > 0) else float("nan")
            ),
            "p": float((1.0 + np.sum(col >= obs_each[j])) / (k + 1.0)),
        })
    reject = _holm_reject([b["p"] for b in per_block], alpha)
    for b, rej in zip(per_block, reject):
        b["significant"] = bool(rej)
    dependent = pval <= alpha if criterion == "total" else bool(np.all(reject))
    reason = None if dependent else REASON_SOURCE_INCOHERENT
    # The smallest attainable p is 1 / (K + 1); below the level the criterion
    # needs (alpha, or alpha / blocks for Holm) no data can show dependence,
    # so the result is "not testable at this K", not "tested and absent".
    needed = alpha if criterion == "total" else alpha / len(blocks)
    if 1.0 / (k + 1.0) > needed + 1e-12:
        out["flags"].append(REASON_INSUFFICIENT_SURROGATES)
        dependent, reason = False, REASON_INSUFFICIENT_SURROGATES
    out.update(
        tc_bits=float(obs), null_mean=mean, null_sd=sd, z=float(z), p=pval,
        per_block=per_block, dependent=bool(dependent), reason=reason,
        min_surrogates=int(math.ceil(1.0 / needed - 1.0 - 1e-9)),
    )
    return out


# --------------------------------------------------------------------------
# legacy pairwise bearer-coherence diagnostic
# --------------------------------------------------------------------------
def _gaussian_mi_bits(x, y, ridge):
    z = np.vstack([x, y])
    c = np.cov(z)
    c = np.atleast_2d(c)
    dim = c.shape[0]
    c = c + float(ridge) * (np.trace(c) / dim) * np.eye(dim)
    dx = x.shape[0]
    _, ld_x = np.linalg.slogdet(c[:dx, :dx])
    _, ld_y = np.linalg.slogdet(c[dx:, dx:])
    _, ld_xy = np.linalg.slogdet(c)
    return max(0.5 * (ld_x + ld_y - ld_xy) / math.log(2.0), 0.0)


def _lagged_mi(a, b, lag, ridge):
    if lag == 0:
        return _gaussian_mi_bits(a, b, ridge)
    ab = _gaussian_mi_bits(a[:, :-lag], b[:, lag:], ridge)
    ba = _gaussian_mi_bits(b[:, :-lag], a[:, lag:], ridge)
    return 0.5 * (ab + ba)


def _holm_reject(pvals, alpha):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = p.size
    reject = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def bearer_coherence(
    ts,
    node_sets,
    lag=1,
    n_surrogates=200,
    seed=0,
    *,
    max_dim=3,
    min_shift=None,
    alpha=0.05,
    ridge=1e-6,
) -> dict:
    """
    Legacy pairwise bearer-coherence diagnostic (superseded for verdicts by
    :func:`joint_dependence`, the single-source constraint): is there measured
    coupling above null between every pair of node sets?

    For every pair of node sets (``node_sets``: mapping name -> node indices,
    or a sequence of index lists) the symmetric lagged Gaussian mutual
    information ``0.5 [I(A_t; B_{t+lag}) + I(B_t; A_{t+lag})]`` (bits; ``I(A;B)``
    for ``lag=0``) is computed between the z-scored set signals (reduced to
    their first ``max_dim`` principal components). The null shifts the whole B
    block circularly by at least ``min_shift`` samples (default 10% of the run;
    within-set structure and autocorrelation are kept, cross-set alignment is
    destroyed), ``n_surrogates`` times with ``numpy.random.default_rng(seed)``.
    The one-sided p-value is ``(1 + #{null >= obs}) / (n + 1)``; the bearer is
    coherent when every pair is significant at ``alpha`` after Holm correction.
    Overlapping sets are allowed (shared nodes are coupled by identity).

    Returns ``{"coherent", "pairs", "min_z", "reason", ...}``; ``reason`` is
    None when coherent, else ``INCOHERENT:<a>|<b>,...``, ``DEGENERATE_SET:<name>``
    or ``NO_SURROGATES``.
    """
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"ts must be 2D (n_nodes x n_time), got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("bearer_coherence needs a finite time series")
    n_nodes, t = x.shape
    if isinstance(node_sets, Mapping):
        named = [(str(k), v) for k, v in node_sets.items()]
    else:
        named = [(f"set{i}", v) for i, v in enumerate(node_sets)]
    lag = int(lag)
    if lag < 0 or lag >= t - 2:
        raise ValueError("lag must be >= 0 and shorter than the run")
    sets = []
    for name, idx in named:
        idx = np.unique(np.asarray(list(idx), dtype=int).reshape(-1))
        if idx.size == 0 or idx.min() < 0 or idx.max() >= n_nodes:
            raise ValueError(f"node set {name!r} is empty or out of range")
        sets.append((name, idx))
    out = {
        "coherent": True,
        "pairs": [],
        "min_z": float("nan"),
        "reason": None,
        "lag": lag,
        "n_surrogates": int(n_surrogates),
        "seed": seed,
        "alpha": float(alpha),
        "max_dim": max_dim,
        "set_names": [s[0] for s in sets],
    }
    if len(sets) < 2:
        return out
    sig = {}
    for name, idx in sets:
        s = _set_signal(x, idx, max_dim)
        if s is None:
            out.update(coherent=False, reason=f"DEGENERATE_SET:{name}")
            return out
        sig[name] = s
    if int(n_surrogates) < 1:
        out.update(coherent=False, reason="NO_SURROGATES")
        return out
    lo = int(math.ceil(0.1 * t)) if min_shift is None else int(min_shift)
    lo = max(1, lo)
    if t - lo < lo:
        raise ValueError("run too short for the circular-shift null")
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a_name, b_name = sets[i][0], sets[j][0]
            a, b = sig[a_name], sig[b_name]
            obs = _lagged_mi(a, b, lag, ridge)
            shifts = rng.integers(lo, t - lo + 1, size=int(n_surrogates))
            null = np.asarray(
                [_lagged_mi(a, np.roll(b, int(s), axis=1), lag, ridge) for s in shifts]
            )
            mean = float(null.mean())
            sd = float(null.std(ddof=1)) if null.size > 1 else float("nan")
            z = (obs - mean) / sd if (math.isfinite(sd) and sd > 0) else float("nan")
            pval = float((1.0 + np.sum(null >= obs)) / (null.size + 1.0))
            pairs.append({"a": a_name, "b": b_name, "mi_bits": float(obs),
                          "null_mean": mean, "null_sd": sd, "z": float(z), "p": pval})
    reject = _holm_reject([pr["p"] for pr in pairs], float(alpha))
    for pr, rej in zip(pairs, reject):
        pr["significant"] = bool(rej)
    zs = [pr["z"] for pr in pairs if math.isfinite(pr["z"])]
    out["pairs"] = pairs
    out["min_z"] = float(min(zs)) if zs else float("nan")
    if not all(reject):
        bad = [f"{pr['a']}|{pr['b']}" for pr in pairs if not pr["significant"]]
        out.update(coherent=False, reason="INCOHERENT:" + ",".join(bad))
    return out
