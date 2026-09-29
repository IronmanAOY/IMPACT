"""
Evidence layer: null-anchored component evidence on a construct scale, the
three-valued MPC verdict (an exclusion rule) and the MPC degree.

Terminology (code, docs and paper):

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

Component status (:func:`component_assessment`). ``se_c`` combines the
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
"""
from __future__ import annotations

import dataclasses
import fnmatch
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
from typing import Mapping

import numpy as np

PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")


class Verdict(str, Enum):
    EXCLUDED = "EXCLUDED"
    MPC_CONSISTENT = "MPC_CONSISTENT"
    UNDETERMINED = "UNDETERMINED"


class ComponentStatus(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNDEFINED = "UNDEFINED"


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
# component as UNDEFINED instead of calling the estimator (1.1.0 post-freeze).
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

# Construct-scale cutoffs (z_j, delta_j) and the one-sided error rate. The
# initial defaults are to be justified by the MPC-Bench dose-response.
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

    def __post_init__(self):
        if self.reference_scale not in REFERENCE_SCALES:
            raise ValueError(
                f"reference_scale must be one of {REFERENCE_SCALES}, "
                f"got {self.reference_scale!r}"
            )
        if self.nodes is not None:
            object.__setattr__(self, "nodes", _node_tuple(self.nodes, "nodes"))


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
# protocol
# --------------------------------------------------------------------------
PROTOCOL_SCHEMA = "impact-mpc-protocol/2"
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

    The canonical JSON (:meth:`to_dict`, all cutoffs materialised) is hashed
    with SHA-256 (:attr:`hash`); :attr:`protocol_id` is what evidence records
    as ``protocol_id``.
    """

    necessity_set: tuple = PRINCIPLES
    channels: Mapping = field(default_factory=dict)
    cutoffs: Mapping = field(default_factory=dict)
    alpha: float = DEFAULT_ALPHA
    null_families: Mapping = field(default_factory=dict)
    reference: Mapping = field(default_factory=lambda: dict(DEFAULT_REFERENCE))
    source_rule: str = "single_source"
    estimators: Mapping = field(default_factory=dict)
    bearer_nodes: Mapping = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self):
        put = functools.partial(object.__setattr__, self)
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
        put("reference", _freeze_json(_normalize_reference(self.reference)))
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

    # -- accessors -----------------------------------------------------------
    def cutoff_for(self, principle) -> tuple:
        return self.cutoffs.get(str(principle), DEFAULT_CUTOFF)

    def channels_for(self, principle):
        """Declared channels of ``principle`` (None when not declared)."""
        return self.channels.get(str(principle))

    def estimator_options(self, principle) -> dict:
        """A fresh (mutable, JSON-typed) copy of the principle's options."""
        return _json_value(self.estimators.get(str(principle), {}))

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        ref = _json_value(self.reference)
        return {
            "schema": PROTOCOL_SCHEMA,
            "necessity_set": list(self.necessity_set),
            "channels": {p: list(v) for p, v in self.channels.items()},
            "cutoffs": {p: list(self.cutoffs[p]) for p in PRINCIPLES},
            "alpha": float(self.alpha),
            "null_families": dict(self.null_families),
            "reference": ref,
            "source_rule": self.source_rule,
            "estimators": {p: _json_value(v) for p, v in self.estimators.items()},
            "bearer_nodes": {
                p: (None if v is None else list(v))
                for p, v in self.bearer_nodes.items()
            },
            "name": self.name,
        }

    @property
    def hash(self) -> str:
        """SHA-256 of the canonical JSON of :meth:`to_dict`."""
        payload = _canonical_json(self.to_dict()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def protocol_id(self) -> str:
        return f"sha256:{self.hash}"

    def __eq__(self, other):
        if not isinstance(other, Protocol):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash(self.hash)

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
        unknown = sorted(set(data) - set(_PROTOCOL_FIELDS))
        if unknown:
            raise ValueError(f"protocol has unknown fields {unknown}")
        if "cutoffs" in data:
            data["cutoffs"] = {k: tuple(v) for k, v in dict(data["cutoffs"]).items()}
        if "necessity_set" in data and data["necessity_set"] is not None:
            data["necessity_set"] = normalize_necessity_set(data["necessity_set"])
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
            f"{legacy}: the null-SD thresholds of the v1 status rule were replaced "
            "by construct-scale cutoffs (use cutoffs={P: (z, delta)} or a Protocol)"
        )
    if unknown:
        raise TypeError(f"unexpected status keyword(s): {unknown}")
    cutoffs = status_kwargs.get("cutoffs")
    if cutoffs is not None and not isinstance(cutoffs, Mapping):
        cutoffs = {p: cutoffs for p in PRINCIPLES}  # one pair for every principle
    return cutoffs, status_kwargs.get("alpha")


def mpc_verdict(
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
    Three-valued MPC verdict (exclusion rule) from per-principle evidence.

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
    U -> UNDETERMINED. With ``registry`` (:class:`ApplicabilityRegistry`),
    evidence from an estimator that is not validated for its principle,
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
# applicability registry
# --------------------------------------------------------------------------
REGISTRY_SCHEMA = "impact-mpc-registry/2"
REGISTRY_STATUSES = ("validated", "provisional", "not_validated", "rejected")
# Validation substrates of the registry (extensible). Empirical substrates are
# covered only by entries validated on their forward-modelled counterpart.
REGISTRY_SUBSTRATES = (
    "synthetic_rate",
    "ising_exact",
    "stuart_landau",
    "eeg_like_forward",
    "bold_like_forward",
)
EMPIRICAL_SUBSTRATES = {
    "eeg": ("eeg_like_forward",),
    "fmri": ("bold_like_forward",),
    "bold": ("bold_like_forward",),
}
# Named regime constraints: key -> (query key, relation).
REGIME_BOUNDS = {
    "T_min": ("n_time", "min"),
    "nodes_min": ("n_nodes", "min"),
    "nodes_max": ("n_nodes", "max"),
    "snr_min": ("snr", "min"),
}
DEFAULT_REGISTRY_CRITERIA = {
    "alpha": DEFAULT_ALPHA,
    "false_present_tolerance": 0.02,
    "two_sided": False,
    "min_recovery_slope": 0.0,
    "cross_talk_max": None,
}
_ESTIMATOR_PRINCIPLE_RE = re.compile(r"^compute_(RAM|PDI|NAS|IIM|SRPI)(?![A-Za-z0-9_])")


def split_estimator(estimator):
    """``"compute_NAS:capacity@nas-v2"`` -> ``("compute_NAS:capacity", "nas-v2")``."""
    if estimator is None:
        return None, None
    name, sep, version = str(estimator).rpartition("@")
    if not sep:
        return str(estimator), None
    return name, (version or None)


def estimator_id(principle, mode, version):
    """Evidence estimator id ``compute_<P>:<mode>@<version>``."""
    return f"compute_{principle}:{mode}@{version}"


@dataclass(frozen=True)
class RegistryEntry:
    """
    One applicability-registry entry: an estimator (``estimator`` pattern and
    ``version``) validated -- or explicitly not validated -- for a principle
    on a substrate, grain and regime, with the benchmark ``evidence`` that
    supports it.

    String fields match with shell wildcards (``*``, ``?``; case-insensitive)
    and may be lists of alternatives. ``regime`` maps a key to a scalar
    (equality), a list (membership) or ``{"min": .., "max": ..}``; the named
    keys ``T_min``, ``nodes_min``, ``nodes_max`` and ``snr_min`` bound the
    query's ``n_time``, ``n_nodes`` and ``snr``; ``fs_or_tr`` is matched
    against the query's ``fs_or_tr`` or else ``tr`` (the sample interval in
    seconds in the pipeline) and ``bins`` against ``bins``. A query that omits
    a constrained key does not match.
    """

    principle: str
    estimator: object
    substrate: object = "*"
    grain: object = "*"
    regime: Mapping = field(default_factory=dict)
    validated: bool = True
    note: str | None = None
    version: object = "*"
    status: str = "validated"
    evidence: Mapping = field(default_factory=dict)
    verified: bool = True


def _match_field(pattern, value):
    if isinstance(pattern, (list, tuple)):
        return any(_match_field(p, value) for p in pattern)
    pat = str(pattern)
    if pat == "*":
        return True
    if value is None:
        return False
    return fnmatch.fnmatchcase(str(value).lower(), pat.lower())


def _regime_value_matches(spec, value):
    if isinstance(spec, Mapping):
        x = _as_float(value)
        if not math.isfinite(x):
            return False
        lo, hi = spec.get("min"), spec.get("max")
        return (lo is None or x >= float(lo)) and (hi is None or x <= float(hi))
    if isinstance(spec, (list, tuple)):
        return any(_regime_value_matches(s, value) for s in spec)
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        x = _as_float(value)
        return math.isfinite(x) and math.isclose(x, float(spec), rel_tol=1e-9)
    return str(spec).lower() == str(value).lower()


def _regime_failure(constraints, regime):
    for key, spec in dict(constraints or {}).items():
        if key in REGIME_BOUNDS:
            qkey, rel = REGIME_BOUNDS[key]
            x = _as_float(regime.get(qkey))
            ok = math.isfinite(x) and (x >= float(spec) if rel == "min"
                                       else x <= float(spec))
        elif key == "fs_or_tr":
            qkey = "fs_or_tr" if "fs_or_tr" in regime else "tr"
            ok = qkey in regime and _regime_value_matches(spec, regime[qkey])
        else:
            ok = key in regime and _regime_value_matches(spec, regime[key])
        if not ok:
            return str(key)
    return None


def _check_regime_spec(regime):
    if not isinstance(regime, Mapping):
        raise ValueError("registry entry 'regime' must be an object")
    for key in REGIME_BOUNDS:
        if key in regime:
            v = _as_float(regime[key])
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"regime {key} must be a finite number >= 0")
    if "nodes_min" in regime and "nodes_max" in regime:
        if float(regime["nodes_min"]) > float(regime["nodes_max"]):
            raise ValueError("regime nodes_min must be <= nodes_max")
    return dict(regime)


def _criteria_failure(evidence, criteria):
    """First unmet entry criterion of a validated entry (None when all hold)."""
    run_id = evidence.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return "evidence.run_id (benchmark run id) is required"
    rate = _as_float(evidence.get("null_false_present_rate"))
    if not (math.isfinite(rate) and 0.0 <= rate <= 1.0):
        return "evidence.null_false_present_rate must be a rate in [0, 1]"
    alpha, tol = float(criteria["alpha"]), float(criteria["false_present_tolerance"])
    if rate > alpha + tol + 1e-12:
        return f"null false-PRESENT rate {rate} exceeds alpha + {tol} ({alpha + tol})"
    if criteria["two_sided"] and rate < alpha - tol - 1e-12:
        return f"null false-PRESENT rate {rate} is below alpha - {tol}"
    slope = _as_float(evidence.get("recovery_slope"))
    if not math.isfinite(slope):
        return "evidence.recovery_slope must be a finite number"
    if not slope > float(criteria["min_recovery_slope"]):
        return f"recovery slope {slope} is not above {criteria['min_recovery_slope']}"
    if evidence.get("recovery_monotone") is False:
        return "recovery is not monotone"
    cmax = criteria.get("cross_talk_max")
    if cmax is not None:
        ct = _as_float(evidence.get("cross_talk"))
        if not (math.isfinite(ct) and 0.0 <= ct <= float(cmax)):
            return (
                f"cross-talk {evidence.get('cross_talk')!r} is not within [0, {cmax}]"
            )
    return None


class ApplicabilityRegistry:
    """
    Registry of estimator versions validated (on MPC-Bench) for a principle,
    substrate, grain and regime. JSON layout (schema ``impact-mpc-registry/2``)::

        {"schema": "impact-mpc-registry/2", "version": "2026-09",
         "criteria": {"alpha": 0.05, "false_present_tolerance": 0.02},
         "entries": [
           {"estimator": "compute_NAS:capacity", "version": "nas-v2-2026.09",
            "substrate": "eeg_like_forward", "grain": "*",
            "regime": {"T_min": 2000, "nodes_min": 16, "nodes_max": 128,
                       "fs_or_tr": {"min": 0.002, "max": 0.01}, "snr_min": 0.5},
            "evidence": {"run_id": "bench-2026-10-01-a",
                         "null_false_present_rate": 0.043,
                         "recovery_slope": 0.81, "cross_talk": 0.03},
            "status": "validated"}]}

    The principle is read from ``principle`` or derived from the estimator
    name (``compute_<P>...``). Entry criteria for ``status: validated``
    (strict mode, the default): a pinned ``version`` (no wildcard), an
    explicit validation ``substrate`` that is not an empirical one
    (``eeg``/``fmri``/``bold``: human EEG/fMRI entries require validation on
    forward-modelled data, ``eeg_like_forward``/``bold_like_forward``), and
    ``evidence`` with a benchmark ``run_id``, a null false-PRESENT rate of at
    most ``alpha + false_present_tolerance`` (both bounds of ``alpha +- tol``
    with ``"two_sided": true``), a ``recovery_slope`` above
    ``min_recovery_slope`` (monotone recovery; ``recovery_monotone: false``
    fails) and, when ``cross_talk_max`` is declared, ``cross_talk`` within it.
    Entries that fail are rejected with a ValueError. ``strict=False`` also
    accepts the legacy v1 layout (``validated: true`` without evidence); such
    entries are marked ``verified=False``. A query on an empirical substrate
    (``eeg``, ``fmri``) is matched against the entries of its forward-model
    substrates.
    """

    def __init__(self, entries=(), *, source=None, version=None, criteria=None,
                 strict=True):
        crit = dict(DEFAULT_REGISTRY_CRITERIA)
        unknown = sorted(set(criteria or {}) - set(crit))
        if unknown:
            raise ValueError(f"registry criteria has unknown keys {unknown}")
        crit.update(dict(criteria or {}))
        _check_alpha(crit["alpha"])
        if not float(crit["false_present_tolerance"]) >= 0:
            raise ValueError("false_present_tolerance must be >= 0")
        self.criteria = crit
        self.strict = bool(strict)
        self.entries = tuple(
            e if isinstance(e, RegistryEntry) else self._entry(e, crit, self.strict)
            for e in entries
        )
        self.source = source
        self.version = version

    @staticmethod
    def _entry(raw, criteria=None, strict=True):
        criteria = dict(DEFAULT_REGISTRY_CRITERIA if criteria is None else criteria)
        if not isinstance(raw, Mapping):
            raise ValueError("registry entries must be JSON objects")
        if "estimator" not in raw:
            raise ValueError(f"registry entry is missing ['estimator']: {raw}")
        principle = raw.get("principle")
        if principle is None:
            m = _ESTIMATOR_PRINCIPLE_RE.match(str(raw["estimator"]))
            if m is None:
                raise ValueError(
                    f"registry entry is missing ['principle'] (not derivable from "
                    f"{raw['estimator']!r}): {raw}"
                )
            principle = m.group(1)
        principle = str(principle).upper()
        if principle not in PRINCIPLES:
            raise ValueError(f"registry entry has unknown principle {principle!r}")
        regime = _check_regime_spec(raw.get("regime") or {})
        validated = raw.get("validated")
        if validated is not None and not isinstance(validated, (bool, np.bool_)):
            # bool("false") is True: a quoted flag must not validate silently.
            raise ValueError(
                f"registry entry 'validated' must be true or false, got {validated!r}"
            )
        status = raw.get("status")
        if status is None:
            status = "not_validated" if validated is False else "validated"
        status = str(status)
        if status not in REGISTRY_STATUSES:
            raise ValueError(
                f"registry entry status must be one of {REGISTRY_STATUSES}"
            )
        if validated is not None and bool(validated) != (status == "validated"):
            raise ValueError("registry entry 'validated' contradicts its 'status'")
        evidence = raw.get("evidence") or {}
        if not isinstance(evidence, Mapping):
            raise ValueError("registry entry 'evidence' must be an object")
        version = raw.get("version", "*")
        substrate = raw.get("substrate", "*")
        verified = True
        if status == "validated":
            legacy = "evidence" not in raw and "status" not in raw
            if legacy and not strict:
                verified = False
            else:
                problems = []
                if version is None or any(ch in str(version) for ch in "*?[") or (
                    isinstance(version, (list, tuple))
                ):
                    problems.append("a pinned 'version' is required")
                subs = (
                    substrate if isinstance(substrate, (list, tuple)) else [substrate]
                )
                if any(str(s) == "*" for s in subs):
                    problems.append("an explicit validation 'substrate' is required")
                if any(str(s).lower() in EMPIRICAL_SUBSTRATES for s in subs):
                    problems.append(
                        "empirical substrates need validation on forward-modelled "
                        "data (eeg_like_forward / bold_like_forward)"
                    )
                why = _criteria_failure(evidence, criteria)
                if why:
                    problems.append(why)
                if problems:
                    raise ValueError(
                        f"registry entry for {raw['estimator']!r} does not meet the "
                        f"entry criteria: {'; '.join(problems)}"
                    )
        return RegistryEntry(
            principle=principle,
            estimator=raw["estimator"],
            substrate=substrate,
            grain=raw.get("grain", "*"),
            regime=regime,
            validated=(status == "validated"),
            note=raw.get("note"),
            version=version,
            status=status,
            evidence=dict(evidence),
            verified=verified,
        )

    @classmethod
    def from_dict(cls, payload, source=None, strict=True):
        criteria = None
        if isinstance(payload, Mapping):
            schema = payload.get("schema", REGISTRY_SCHEMA)
            if schema != REGISTRY_SCHEMA:
                raise ValueError(f"unsupported registry schema {schema!r}")
            entries = payload.get("entries", [])
            version = payload.get("version")
            criteria = payload.get("criteria")
        elif isinstance(payload, (list, tuple)):
            entries, version = payload, None
        else:
            raise ValueError("registry must be a JSON object or a list of entries")
        return cls(entries, source=source, version=version, criteria=criteria,
                   strict=strict)

    @classmethod
    def from_json(cls, path, strict=True):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload, source=str(path), strict=strict)

    def to_dict(self) -> dict:
        return {
            "schema": REGISTRY_SCHEMA,
            "version": self.version,
            "source": self.source,
            "strict": self.strict,
            "criteria": dict(self.criteria),
            "entries": [
                {
                    "principle": e.principle,
                    "estimator": e.estimator,
                    "version": e.version,
                    "substrate": e.substrate,
                    "grain": e.grain,
                    "regime": dict(e.regime),
                    "evidence": dict(e.evidence),
                    "status": e.status,
                    "validated": e.validated,
                    "verified": e.verified,
                    "note": e.note,
                }
                for e in self.entries
            ],
        }

    def is_validated(self, principle, estimator, substrate=None, grain=None, **regime):
        """
        ``(True, None)`` if a validated entry matches the principle, estimator
        name and version (``estimator`` is ``name@version``), substrate, grain
        and regime; otherwise ``(False, reason)`` with reason
        ``ESTIMATOR_UNDECLARED``, ``NO_ENTRY``, ``MARKED_NOT_VALIDATED``
        (``:<status>`` for provisional/rejected entries) or
        ``REGIME_MISMATCH:<key>``. An explicitly not-validated entry wins over
        a validated one.
        """
        if estimator is None:
            return False, "ESTIMATOR_UNDECLARED"
        name, version = split_estimator(estimator)
        p = str(principle).upper()
        subs = [substrate]
        if substrate is not None:
            subs += list(EMPIRICAL_SUBSTRATES.get(str(substrate).lower(), ()))
        cands = [
            e for e in self.entries
            if e.principle == p
            and _match_field(e.estimator, name)
            and (str(e.version) == "*" or _match_field(e.version, version))
            and any(_match_field(e.substrate, s) for s in subs)
            and _match_field(e.grain, grain)
        ]
        for e in cands:
            if not e.validated and _regime_failure(e.regime, regime) is None:
                return False, (
                    "MARKED_NOT_VALIDATED" if e.status == "not_validated"
                    else f"MARKED_NOT_VALIDATED:{e.status}"
                )
        failures = []
        for e in cands:
            if not e.validated:
                continue
            key = _regime_failure(e.regime, regime)
            if key is None:
                return True, None
            failures.append(key)
        if failures:
            return False, f"REGIME_MISMATCH:{failures[0]}"
        return False, "NO_ENTRY"


def resolve_registry(registry, strict=True):
    """``None`` | registry | JSON path | dict/list -> ApplicabilityRegistry."""
    if registry is None or isinstance(registry, ApplicabilityRegistry):
        return registry
    if isinstance(registry, (str, os.PathLike)):
        return ApplicabilityRegistry.from_json(os.fspath(registry), strict=strict)
    if isinstance(registry, (dict, list, tuple)):
        return ApplicabilityRegistry.from_dict(registry, strict=strict)
    raise TypeError(
        "applicability_registry must be None, an ApplicabilityRegistry, a JSON "
        "path or a registry dict."
    )


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
