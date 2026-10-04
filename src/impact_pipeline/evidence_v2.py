"""
Evidence layer v2 (MPC-Bench v2): the status rule ``tost-v2``, the protocol
schema ``impact-mpc-protocol/3`` and the dispatcher between the v1 and the v2
layer.

The v1 layer (:mod:`impact_pipeline.evidence`) is frozen. It stays the only
judge of ``impact-mpc-protocol/2`` protocols: :func:`mpc_verdict`,
:func:`assess_component` and :func:`load_protocol` hand a protocol without a
``status_rule`` block (or no protocol at all) to the v1 functions unchanged,
so every v1 result is reproduced bit for bit. A protocol with the block has
schema ``impact-mpc-protocol/3`` (:class:`ProtocolV3`) and is judged here;
the v1 functions refuse it.

**Status rule** ``tost-v2``. For a component with construct value
``c = (m - nu) / (rho - nu)``, SE ``se_c`` and degrees of freedom ``df``
(computed exactly as in v1: delta method over the sampling, null and
reference parts; Welch-Satterthwaite ``df`` from the record's ``se_df``),
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

**Consumption contract.** Each principle feeds one ABSENT decision into the
verdict at ``alpha_A``. A principle with directions (NAS: ``receive`` and
``return``) is PRESENT iff every direction is PRESENT at ``alpha``
(intersection-union) and ABSENT iff some direction passes the TOST at
``alpha_A / k`` (union, Bonferroni over the ``k`` directions). Channels of a
principle are combined by strong-Kleene OR: ABSENT needs every channel
ABSENT (an intersection, each at ``alpha_A``), PRESENT through any of ``k``
channels is a union and is tested at ``alpha / k`` per channel. ``se_df`` is
read from the record and must agree with its SE method (:data:`SE_METHODS`).
IIM PRESENT also needs the rank gate ``p_ind <= 0.05`` when the protocol
declares it. Every UNDEFINED reason comes from
:mod:`impact_pipeline.v2.reasons` and is never counted as ABSENT; the verdict
is the strong-Kleene AND over the protocol's necessity set (the anchored
principles), and the AND over the declared set is reported beside it.

Three entry points share one implementation of the decision
(:func:`_decide`): :func:`assess_item` (one evidence item under a protocol),
:func:`assess_array` (vectorised from raw inputs) and :func:`status_c`
(vectorised on the construct scale, for the rule audit), plus
:func:`status_c_directional` for directional principles. The test suite
checks that they agree.

The documentation of every rule, field and reason code is
``docs/metrics_v2.md``.
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
from types import MappingProxyType
from typing import Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline import evidence as v1
from impact_pipeline.v2 import ESTIMATOR_VERSIONS_V1, ESTIMATOR_VERSIONS_V2
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import records as _records

# The v1 functions as imported, so that the dispatcher always reaches the
# frozen implementation.
_v1_mpc_verdict = v1.mpc_verdict
_v1_component_assessment = v1.component_assessment

PRINCIPLES = v1.PRINCIPLES
ComponentStatus = v1.ComponentStatus
Verdict = v1.Verdict
PRESENT, ABSENT, UNDEFINED = (
    ComponentStatus.PRESENT, ComponentStatus.ABSENT, ComponentStatus.UNDEFINED)
CODE_T, CODE_U, CODE_F = v1.CODE_T, v1.CODE_U, v1.CODE_F
_CODE_TO_STATUS = {CODE_T: PRESENT, CODE_U: UNDEFINED, CODE_F: ABSENT}
_STATUS_TO_CODE = {v: k for k, v in _CODE_TO_STATUS.items()}
_CODE_TO_VERDICT = {
    CODE_T: Verdict.MPC_CONSISTENT, CODE_U: Verdict.UNDETERMINED,
    CODE_F: Verdict.EXCLUDED,
}

# v1 helpers re-exported for v2 callers (verdict decomposition is unchanged).
parse_verdict_reason = v1.parse_reason
verdict_from_reasons = v1.verdict_from_reasons
kleene_verdict_codes = v1.kleene_verdict_codes

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
STATUS_RULE_VERSION = "tost-v2"
STATUS_RULE_VERSIONS = (STATUS_RULE_VERSION,)
ABSENT_TESTS = ("tost",)
PROTOCOL_SCHEMA_V2 = v1.PROTOCOL_SCHEMA
PROTOCOL_SCHEMA_V3 = "impact-mpc-protocol/3"
PROTOCOL_SCHEMAS = (PROTOCOL_SCHEMA_V2, PROTOCOL_SCHEMA_V3)
DEFAULT_ALPHA = v1.DEFAULT_ALPHA
DEFAULT_CUTOFF = v1.DEFAULT_CUTOFF
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
# Reference kinds of a /3 protocol: the v1 kinds plus ``pending`` (no anchor
# yet: every component is UNDEFINED(INVALID_ANCHORS); templates only).
REFERENCE_PENDING = "pending"
REFERENCE_KINDS_V3 = v1.REFERENCE_KINDS + (REFERENCE_PENDING,)
# Status routes of an assessment.
ROUTE_TOST, ROUTE_EXACT, ROUTE_CONCORDANCE, ROUTE_DIRECTIONAL = (
    "tost", "exact", "concordance", "directional")
# Values of a registry admission (registry v3) per status direction.
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
DF_WELCH_SATTERTHWAITE = "welch_satterthwaite"
DF_NONE = "none"
DF_RULES = (DF_FIXED, DF_N_NULL_MINUS_1, DF_WELCH_SATTERTHWAITE, DF_NONE)


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
             meaning="delete-a-group jackknife over 10 pair groups (v1 SRPI; "
                     "se_df = 9)"),
    SeMethod("hoeffding", ("SRPI",), DF_WELCH_SATTERTHWAITE,
             meaning="Hoeffding-decomposition SE with Welch-Satterthwaite "
                     "se_df (SRPI v3)"),
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
    "nas-v3-2026.10": SCALE_QUADRATIC,
    "nas-v2-2026.09": SCALE_QUADRATIC,
    "iim-v5-2026.10": SCALE_QUADRATIC,
    "iim-v4-2026.09": SCALE_QUADRATIC,
    "srpi-v3-2026.10": SCALE_QUADRATIC,
    "srpi-v2-2026.09": SCALE_FIRST_ORDER,
    "ram-v3-2026.10": SCALE_FIRST_ORDER,
    "pdi-v3-2026.10": SCALE_COUNT,
    "pdi-v2-2026.09": SCALE_COUNT,
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
    z, delta = v1.normalize_cutoff(cutoff)
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
    """The hash-covered ``status_rule`` block of a /3 protocol."""

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
        put("alpha", v1._check_alpha(self.alpha))
        a_abs = v1._as_float(self.alpha_absent)
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
# quantiles and the decision (one implementation)
# --------------------------------------------------------------------------
def quantile(alpha, df=math.inf) -> float:
    """One-sided ``1 - alpha`` quantile: Student ``t`` with ``df`` degrees of
    freedom, the normal quantile for an infinite or missing ``df`` (as v1)."""
    a = float(alpha)
    if not (0.0 < a <= 0.5):
        raise ValueError("alpha must be in (0, 0.5]")
    return v1._one_sided_quantile(a, v1._as_float(df))


def _quantiles(alpha, df) -> np.ndarray:
    a = float(alpha)
    if not (0.0 < a <= 0.5):
        raise ValueError("alpha must be in (0, 0.5]")
    d = np.asarray(df, dtype=float)
    q = np.full(d.shape, v1._z_quantile(a))
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
    The status rule on the construct scale for valid inputs (finite ``c``,
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
    The status rule on the construct scale (the rule-audit entry point):
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
    and the order of checks of v1 ``component_assessment``."""
    if reference_scale not in v1.REFERENCE_SCALES:
        raise ValueError(f"reference_scale must be one of {v1.REFERENCE_SCALES}")
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
        # c is reported from the anchor check on (as v1 does)
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
    Vectorised status rule from raw inputs (the arguments of v1
    ``component_status_array``): ``c``, ``se_c`` and ``df`` as v1 computes
    them, then the v2 decision. Invalid rows carry the v1 validity reasons
    in v1's order (``NON_FINITE_ESTIMATE``, ``NO_NULL_CALIBRATION``,
    ``DEGENERATE_NULL``, ``INVALID_ANCHORS``, ``INVALID_SE``,
    ``NO_SAMPLING_SE``). An ``exact`` row without a sampling SE is decided
    on ``c`` alone (``se_c = 0``), as in :func:`assess_item`.
    """
    rule, a_p, a_a = _levels(rule, alpha_present, alpha_absent)
    z, delta = v1.normalize_cutoff(cutoff)
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
# evidence items and assessments
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentEvidenceV2(v1.ComponentEvidence):
    """
    One evidence item of the v2 layer: the v1 fields plus ``se_method`` (the
    SE method of the contract, :data:`SE_METHODS`; None keeps the v1 meaning
    of ``se_df``), ``direction`` (a member of a directional principle, for
    example NAS ``receive``/``return``), ``p_ind`` (the IIM rank p-value
    against the independence null), and the cell of the concordance route:
    ``substrate`` (v1 field), ``observation_stage``, ``view`` and
    ``content_bearer`` (the declared PDI bearer, for example
    ``non_workspace``).
    """

    se_method: Optional[str] = None
    direction: Optional[str] = None
    p_ind: Optional[float] = None
    observation_stage: Optional[str] = None
    view: Optional[str] = None
    content_bearer: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("se_method", "direction", "observation_stage", "view",
                     "content_bearer"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, str) or not v.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        if self.direction is not None and not _NAME_RE.match(self.direction):
            raise ValueError(f"malformed direction {self.direction!r}")
        if self.p_ind is not None:
            p = v1._as_float(self.p_ind)
            if math.isfinite(p) and not (0.0 <= p <= 1.0):
                raise ValueError("p_ind must lie in [0, 1]")

    @property
    def estimator_version(self) -> Optional[str]:
        return v1.split_estimator(self.estimator)[1]


def _attr(ev, name, default=None):
    return getattr(ev, name, default)


@dataclass(frozen=True)
class AssessmentV2:
    """
    The v2 assessment of one channel member (or of a directional channel,
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
    members: Tuple["AssessmentV2", ...] = ()
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
                          channel="default", direction=None) -> AssessmentV2:
    R.lookup(reason)
    return AssessmentV2(
        UNDEFINED, reason, (), c=float(c),
        cutoff_present=lv.cutoff[0], cutoff_absent=lv.cutoff[1],
        alpha=lv.alpha_present, alpha_absent=lv.alpha_absent,
        se_method=_attr(ev, "se_method"),
        channel=(ev.channel if ev is not None else channel),
        direction=(_attr(ev, "direction") if ev is not None else direction),
        scale=lv.scale,
    )


def _se_contract_violation(ev, principle, allowed) -> bool:
    """True iff the item breaks the SE-method contract: no method where the
    protocol declares the admitted methods of the principle, an unknown
    method, a method not allowed for the principle or not declared by the
    protocol, ``se_df`` inconsistent with the method's rule, a sampling SE
    on a method without one, or ``exact`` inconsistent with the method.
    Without a method and without a protocol declaration ``se_df`` keeps its
    v1 meaning."""
    name = _attr(ev, "se_method")
    if name is None:
        # a declared contract binds every item: an item that does not name
        # its method could carry, e.g., B - 1 for a block bootstrap
        return allowed is not None
    method = SE_METHODS.get(name)
    if method is None or principle not in method.principles:
        return True
    if allowed is not None and name not in allowed:
        return True
    se = v1._as_float(ev.se)
    se_df = v1._as_float(ev.se_df)
    exact = bool(ev.exact)
    if (name == SE_METHOD_EXACT) != exact:
        return True
    if not method.sampling_se:
        return (math.isfinite(se) and se > 0) or math.isfinite(se_df)
    if method.df_rule == DF_FIXED:
        return not (math.isfinite(se_df) and se_df in method.df_values)
    if method.df_rule == DF_N_NULL_MINUS_1:
        n_null = v1._as_float(ev.n_null)
        return not (math.isfinite(se_df) and n_null >= 2 and se_df == n_null - 1)
    if method.df_rule == DF_WELCH_SATTERTHWAITE:
        return not (math.isfinite(se_df) and se_df > 0)
    return False  # pragma: no cover - every rule is handled above


class Admission(NamedTuple):
    """Whether a status direction is licensed for an item (registry v3:
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
    The admission of an item by ``registry``: a registry v3 object with
    ``admission(principle, estimator, substrate, grain, **regime)`` returning
    ``{"present": .., "absent": ..}`` (values of :data:`ADMISSION_VALUES` or
    bools; ``vacuous`` admits), or a v1 :class:`~impact_pipeline.evidence.
    ApplicabilityRegistry` (one decision for both directions). None without a
    registry.
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
    raise TypeError("registry must provide admission() (v3) or is_validated() (v1)")


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
                   uncalibrated=frozenset()) -> AssessmentV2:
    """The v2 assessment of one evidence item (one channel member)."""
    z, delta = lv.cutoff
    rule = lv.rule
    if pre_reason is not None:
        return _undefined_assessment(pre_reason, lv, ev)
    if not bool(ev.defined):
        reason = _vocab_reason(ev.reason, R.NOT_DEFINED)
        if reason == R.NOT_IMPLEMENTED:  # name the channel, as v1 does
            reason = R.format_reason(R.NOT_IMPLEMENTED, R.clean_detail(ev.channel))
        return _undefined_assessment(reason, lv, ev)
    if not math.isfinite(v1._as_float(ev.estimate)):
        return _undefined_assessment(
            _vocab_reason(ev.reason, R.NON_FINITE_ESTIMATE), lv, ev)
    a1 = _v1_component_assessment(ev, cutoff=lv.cutoff, alpha=lv.alpha_present)
    if a1.status is UNDEFINED and a1.reason not in (R.INCONCLUSIVE, R.NO_SAMPLING_SE):
        return _undefined_assessment(a1.reason, lv, ev, c=a1.c)
    if _se_contract_violation(ev, principle, se_allowed):
        return _undefined_assessment(R.INVALID_SE, lv, ev, c=a1.c)
    se_method = _attr(ev, "se_method")
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
        p = v1._as_float(_attr(ev, "p_ind"))
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
    return AssessmentV2(
        _CODE_TO_STATUS[code], reason, tuple(flags), c=c, se=se_c,
        se_sampling=parts[0], se_null=parts[1], se_reference=parts[2], df=df,
        q_present=q_p, q_absent=q_a, lower=lower_p, upper=upper_p,
        lower_absent=lower_a, upper_absent=upper_a,
        margin_present=lower_p - z,
        margin_absent=min(delta - upper_a, lower_a + delta),
        cutoff_present=z, cutoff_absent=delta, alpha=lv.alpha_present,
        alpha_absent=lv.alpha_absent, route=route_name, se_method=se_method,
        channel=ev.channel, direction=_attr(ev, "direction"), scale=lv.scale,
    )


def _combine_members(members, lv: _Levels, channel) -> AssessmentV2:
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
    return AssessmentV2(
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
# protocol schema /3
# --------------------------------------------------------------------------
_V3_EXTRA_FIELDS = (
    "status_rule", "declared_necessity_set", "directions", "rank_gates",
    "estimator_versions", "se_methods", "shared_inputs_declaration",
    "concordance_route", "anchors", "precision",
)
_V3_FIELDS = tuple(v1._PROTOCOL_FIELDS) + _V3_EXTRA_FIELDS
# The vocabularies of the result schema mpc-bench-result/3.
SHARED_INPUT_LEVELS = _records.SHARED_INPUTS
KNOWN_DECLARATIONS = _records.KNOWN_DECLARATIONS
OBSERVATION_STAGES = _records.OBSERVATION_STAGES
# Declarations of the bench and their identifiability level (design 2.0.1).
DECLARATION_LEVELS = MappingProxyType({
    "R": "complete", "H": "partial", "P": "partial", "Q10": "partial",
    "Q25": "partial", "J": "partial", "none": "none",
})
_CELL_KEYS = ("principle", "substrate", "observation_stage", "view", "bearer")
_ROUTE_KEYS = _CELL_KEYS + ("absent", "present", "provisional", "battery")


def _normalize_shared_inputs(value):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("shared_inputs_declaration must be an object or null")
    unknown = sorted(set(value) - {"id", "shared_inputs", "inputs"})
    if unknown:
        raise ValueError(f"shared_inputs_declaration has unknown keys {unknown}")
    did = value.get("id")
    if did not in KNOWN_DECLARATIONS:
        raise ValueError(f"shared_inputs_declaration.id must be one of "
                         f"{KNOWN_DECLARATIONS}, got {did!r}")
    level = value.get("shared_inputs")
    if level not in SHARED_INPUT_LEVELS:
        raise ValueError(f"shared_inputs_declaration.shared_inputs must be one of "
                         f"{SHARED_INPUT_LEVELS}")
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
    cells, seen = [], set()
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ValueError("a concordance-route cell must be an object")
        unknown = sorted(set(raw) - set(_ROUTE_KEYS))
        missing = sorted(set(_CELL_KEYS + ("absent", "present")) - set(raw))
        if unknown or missing:
            raise ValueError(f"concordance-route cell: unknown {unknown}, "
                             f"missing {missing}")
        p = v1._principle_key(raw["principle"], "concordance_route")
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
        if raw["observation_stage"] not in OBSERVATION_STAGES:
            raise ValueError(f"observation_stage must be one of {OBSERVATION_STAGES}")
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
        cell["battery"] = v1._json_value(battery)
        key = tuple(cell[k] for k in _CELL_KEYS)
        if key in seen:
            raise ValueError(f"duplicate concordance-route cell {key}")
        seen.add(key)
        cells.append(cell)
    cells.sort(key=lambda d: tuple(d[k] for k in _CELL_KEYS))
    return tuple(v1._freeze_json(c) for c in cells)


@dataclass(frozen=True, eq=False)
class ProtocolV3:
    """
    Measurement protocol of schema ``impact-mpc-protocol/3``: every field of
    the v1 :class:`~impact_pipeline.evidence.Protocol` (normalised by the v1
    code) plus

    - ``status_rule``: the :class:`StatusRule` block (``tost-v2``);
      ``alpha`` equals the protocol's ``alpha`` and ``alpha_absent`` equals
      ``alpha / |declared_necessity_set|`` (0.01);
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

    ``reference`` may also be ``{"kind": "pending"}`` (no anchor: every
    component UNDEFINED(INVALID_ANCHORS)); with an ``external`` reference the
    anchors of the protocol replace those of the evidence items, so every
    status is on the protocol's frozen scale. The hash is the SHA-256 of the
    canonical JSON of :meth:`to_dict`.
    """

    necessity_set: tuple = PRINCIPLES
    channels: Mapping = field(default_factory=dict)
    cutoffs: Mapping = field(default_factory=dict)
    alpha: float = DEFAULT_ALPHA
    null_families: Mapping = field(default_factory=dict)
    reference: Mapping = field(default_factory=lambda: {"kind": REFERENCE_PENDING})
    source_rule: str = "single_source"
    estimators: Mapping = field(default_factory=dict)
    bearer_nodes: Mapping = field(default_factory=dict)
    name: Optional[str] = None
    status_rule: object = None
    declared_necessity_set: tuple = PRINCIPLES
    directions: Mapping = field(default_factory=dict)
    rank_gates: Mapping = field(default_factory=dict)
    estimator_versions: Mapping = field(default_factory=dict)
    se_methods: Mapping = field(default_factory=dict)
    shared_inputs_declaration: Optional[Mapping] = None
    concordance_route: tuple = ()
    anchors: Optional[Mapping] = None
    precision: Optional[Mapping] = None

    def __post_init__(self):
        put = functools.partial(object.__setattr__, self)
        ref = dict(self.reference) if self.reference is not None else {
            "kind": REFERENCE_PENDING}
        pending = str(ref.get("kind", "")).strip() == REFERENCE_PENDING
        if pending:
            unknown = sorted(set(ref) - {"kind", "note"})
            if unknown:
                raise ValueError(f"pending reference has unknown keys {unknown}")
        base = v1.Protocol(
            necessity_set=self.necessity_set, channels=self.channels,
            cutoffs=self.cutoffs, alpha=self.alpha, null_families=self.null_families,
            reference=(None if pending else ref), source_rule=self.source_rule,
            estimators=self.estimators, bearer_nodes=self.bearer_nodes, name=self.name,
        )
        for name in v1._PROTOCOL_FIELDS:
            put(name, getattr(base, name))
        if pending:
            pend = {"kind": REFERENCE_PENDING}
            if ref.get("note") is not None:
                pend["note"] = str(ref["note"])
            put("reference", v1._freeze_json(pend))
        put("_v1_dict", base.to_dict())
        declared = v1.normalize_necessity_set(self.declared_necessity_set)
        if declared != PRINCIPLES:
            # N_decl is the fixed core, so alpha_A = alpha / 5 = 0.01 in every
            # protocol (bench, component-level, forward, paper 2) and statuses
            # transfer between protocols without re-judging
            raise ValueError(
                f"declared_necessity_set must be the five principles {PRINCIPLES}, "
                f"got {declared}; restrict necessity_set (N_anch) instead")
        put("declared_necessity_set", declared)
        if not set(self.necessity_set) <= set(declared):
            raise ValueError("necessity_set must be a subset of declared_necessity_set")
        rule = self.status_rule
        if rule is None:
            rule = StatusRule(alpha=self.alpha, alpha_absent=self.alpha / len(declared))
        rule = StatusRule.from_dict(rule)
        if rule.alpha != self.alpha:
            raise ValueError("status_rule.alpha must equal the protocol's alpha")
        want = self.alpha / len(declared)
        if not math.isclose(rule.alpha_absent, want, rel_tol=_REL_TOL, abs_tol=0.0):
            raise ValueError(
                f"status_rule.alpha_absent must be alpha / |declared_necessity_set| "
                f"= {want!r}, got {rule.alpha_absent!r}")
        # E4 needs alpha_absent x |necessity_set| <= alpha; the two checks
        # above imply it, and it is kept as a guard
        if rule.alpha_absent * len(self.necessity_set) > rule.alpha * (1 + _REL_TOL):
            raise ValueError("alpha_absent x |necessity_set| exceeds alpha")
        put("status_rule", rule)
        directions = {}
        for k, v in dict(self.directions or {}).items():
            p = v1._principle_key(k, "directions")
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
            p = v1._principle_key(k, "rank_gates")
            lvl = v1._as_float(v)
            if not (0.0 < lvl <= 0.5):
                raise ValueError(f"rank gate of {p} must lie in (0, 0.5]")
            gates[p] = float(lvl)
        put("rank_gates", MappingProxyType(gates))
        versions = {}
        for k, v in dict(self.estimator_versions or {}).items():
            p = v1._principle_key(k, "estimator_versions")
            allowed = set(ESTIMATOR_VERSIONS_V2[p]) | {ESTIMATOR_VERSIONS_V1[p]}
            if v not in allowed:
                raise ValueError(f"estimator version of {p} must be one of "
                                 f"{sorted(allowed)}, got {v!r}")
            versions[p] = str(v)
        put("estimator_versions", MappingProxyType(versions))
        methods = {}
        for k, v in dict(self.se_methods or {}).items():
            p = v1._principle_key(k, "se_methods")
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
            else v1._freeze_json(_normalize_shared_inputs(shared)))
        put("concordance_route", _normalize_route(self.concordance_route, methods))
        self._check_reference_keys()
        from impact_pipeline.v2 import testability as T

        put("anchors", None if self.anchors is None else v1._freeze_json(
            T.validate_anchors_block(self.anchors, declared, self.necessity_set)))
        put("precision", None if self.precision is None else v1._freeze_json(
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
        return v1._json_value(self.estimators.get(str(principle), {}))

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
        out = json.loads(json.dumps(self._v1_dict))
        out["schema"] = PROTOCOL_SCHEMA_V3
        out["reference"] = v1._json_value(self.reference)
        out.update({
            "status_rule": self.status_rule.to_dict(),
            "declared_necessity_set": list(self.declared_necessity_set),
            "directions": {p: list(v) for p, v in self.directions.items()},
            "rank_gates": dict(self.rank_gates),
            "estimator_versions": dict(self.estimator_versions),
            "se_methods": {p: list(v) for p, v in self.se_methods.items()},
            "shared_inputs_declaration": (
                None if self.shared_inputs_declaration is None
                else v1._json_value(self.shared_inputs_declaration)),
            "concordance_route": [v1._json_value(c) for c in self.concordance_route],
            "anchors": None if self.anchors is None else v1._json_value(self.anchors),
            "precision": (None if self.precision is None
                          else v1._json_value(self.precision)),
        })
        return out

    @property
    def hash(self) -> str:
        payload = v1._canonical_json(self.to_dict()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def protocol_id(self) -> str:
        return f"sha256:{self.hash}"

    @property
    def schema(self) -> str:
        return PROTOCOL_SCHEMA_V3

    def __eq__(self, other):
        if not isinstance(other, ProtocolV3):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash(self.hash)

    def __reduce__(self):
        # read-only mapping views do not pickle; the canonical dict does
        return (ProtocolV3.from_dict, (self.to_dict(),))

    def to_json(self, path=None, indent=2) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        return text

    @classmethod
    def from_dict(cls, payload) -> "ProtocolV3":
        if not isinstance(payload, Mapping):
            raise ValueError("a protocol must be a JSON object")
        data = dict(payload)
        schema = data.pop("schema", None)
        if schema is None:
            schema = PROTOCOL_SCHEMA_V3 if "status_rule" in data else PROTOCOL_SCHEMA_V2
        if schema != PROTOCOL_SCHEMA_V3:
            raise ValueError(f"not an {PROTOCOL_SCHEMA_V3} protocol (schema "
                             f"{schema!r}); v1 protocols are read by "
                             "evidence.Protocol")
        unknown = sorted(set(data) - set(_V3_FIELDS))
        if unknown:
            raise ValueError(f"protocol has unknown fields {unknown}")
        if "status_rule" not in data:
            raise ValueError(
                f"an {PROTOCOL_SCHEMA_V3} protocol needs a status_rule block")
        data["status_rule"] = StatusRule.from_dict(data["status_rule"])
        if "cutoffs" in data:
            data["cutoffs"] = {k: tuple(v) for k, v in dict(data["cutoffs"]).items()}
        for key in ("necessity_set", "declared_necessity_set"):
            if key in data and data[key] is not None:
                data[key] = v1.normalize_necessity_set(data[key])
        return cls(**data)

    @classmethod
    def from_json(cls, path) -> "ProtocolV3":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def replace(self, **changes) -> "ProtocolV3":
        return dataclasses.replace(self, **changes)


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------
def protocol_schema(payload) -> str:
    """The schema of a protocol object, dict or JSON path: /3 iff the
    protocol carries a ``status_rule`` block (or declares /3)."""
    if isinstance(payload, ProtocolV3):
        return PROTOCOL_SCHEMA_V3
    if isinstance(payload, v1.Protocol):
        return PROTOCOL_SCHEMA_V2
    if isinstance(payload, (str, os.PathLike)):
        with open(os.fspath(payload), "r", encoding="utf-8") as f:
            payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise TypeError("protocol must be a Protocol, a ProtocolV3, a JSON path or "
                        "a dict")
    schema = payload.get("schema")
    if schema is None:
        return PROTOCOL_SCHEMA_V3 if "status_rule" in payload else PROTOCOL_SCHEMA_V2
    if schema not in PROTOCOL_SCHEMAS:
        raise ValueError(f"unsupported protocol schema {schema!r}")
    if schema == PROTOCOL_SCHEMA_V2 and "status_rule" in payload:
        raise ValueError("a protocol with a status_rule block has schema "
                         f"{PROTOCOL_SCHEMA_V3}")
    return schema


def load_protocol(protocol):
    """``None`` | Protocol | ProtocolV3 | JSON path | dict -> None, a v1
    :class:`~impact_pipeline.evidence.Protocol` (schema /2, read by the v1
    code) or a :class:`ProtocolV3`."""
    if protocol is None or isinstance(protocol, (v1.Protocol, ProtocolV3)):
        return protocol
    if isinstance(protocol, (str, os.PathLike)):
        with open(os.fspath(protocol), "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif isinstance(protocol, Mapping):
        payload = protocol
    else:
        raise TypeError("protocol must be None, a Protocol, a ProtocolV3, a JSON path "
                        "or a dict")
    if protocol_schema(payload) == PROTOCOL_SCHEMA_V3:
        return ProtocolV3.from_dict(payload)
    return v1.Protocol.from_dict(payload)


def is_v3(protocol) -> bool:
    return isinstance(load_protocol(protocol), ProtocolV3)


def _require_v3(protocol) -> ProtocolV3:
    proto = load_protocol(protocol)
    if not isinstance(proto, ProtocolV3):
        raise TypeError(f"an {PROTOCOL_SCHEMA_V3} protocol is required")
    return proto


def with_protocol_reference(ev, protocol):
    """The item with the anchor of the protocol: an external reference
    replaces the item's own (no matching key: no anchor), a pending reference
    removes it, a cohort reference keeps the item's (computed per cohort)."""
    proto = _require_v3(protocol)
    kind = proto.reference.get("kind")
    if kind == "cohort_high_state":
        return ev
    ref = {} if kind == REFERENCE_PENDING else proto.reference_for(
        ev.principle, ev.channel, _attr(ev, "direction"))
    if not ref:
        return dataclasses.replace(ev, reference=None, reference_se=None)
    return dataclasses.replace(ev, reference=ref["reference"],
                               reference_se=ref["reference_se"],
                               reference_scale=ref["reference_scale"])


def _uncalibrated_set(pairs):
    out = set()
    for pair in pairs or ():
        p, method = pair
        out.add((v1._principle_key(p, "uncalibrated_se_methods"), str(method)))
    return frozenset(out)


def _item_pre_reason(ev, principle, proto, admission):
    want = proto.estimator_version_for(principle)
    if want is not None and v1.split_estimator(ev.estimator)[1] != want:
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
    if _attr(ev, "se_method") == SE_METHOD_CONCORDANT:
        route = proto.concordance_admission(
            principle, ev.substrate, _attr(ev, "observation_stage"), _attr(ev, "view"),
            _attr(ev, "content_bearer"))
    return _assess_member(
        ev, principle, lv, pre_reason=pre, se_allowed=proto.se_methods_for(principle),
        route=route, rank_gate=proto.rank_gate_for(principle), admission=admission,
        uncalibrated=uncalibrated)


def assess_item(ev, protocol, *, registry=None, regime=None, n_channels=None,
                uncalibrated_se_methods=(), status_rule=None,
                scale="construct") -> AssessmentV2:
    """
    The v2 assessment of one evidence item under a /3 protocol, at the levels
    the protocol implies for it (``alpha / k_channels`` for PRESENT,
    ``alpha_A / k_directions`` for the TOST of a direction member). Order of
    the checks: estimator version, registry (no direction admitted), null
    family, the v1 validity checks (estimator not defined, non-finite
    estimate, null, anchors, SE), the SE-method contract, no sampling SE
    (or the concordance route), the decision, then the conditions that can
    only remove a decision: the rank gate, the registry direction, the SE
    reversion (``uncalibrated_se_methods``: ``(principle, se_method)``
    pairs found anti-conservative on the twins).
    """
    proto = _require_v3(protocol)
    p = v1._principle_key(ev.principle, "evidence")
    rule = proto.status_rule if status_rule is None else _coerce_rule(status_rule)
    if n_channels is None:
        n_channels = len(proto.channels_for(p) or (ev.channel,))
    lv = _item_levels(proto, p, rule, n_channels, scale)
    return _assess_one(ev, p, proto, lv, registry=registry, regime=regime,
                       uncalibrated=_uncalibrated_set(uncalibrated_se_methods))


@dataclass(frozen=True)
class PrincipleAssessment:
    """
    The status of one principle under a /3 protocol: the Kleene OR over its
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
    deciding: Optional[AssessmentV2]
    margin: float
    ignored: Tuple[str, ...] = ()
    kept: Tuple = field(default=(), repr=False)

    def __post_init__(self):
        R.check_status(self.status.value, self.reason, self.flags)

    @property
    def c(self) -> float:
        return self.deciding.c if self.deciding is not None else float("nan")

    def record_fields(self) -> dict:
        """The status fields of a ``mpc-bench-result/3`` component record:
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
        direction = _attr(ev, "direction")
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
    """The status of one principle from its evidence items under a /3
    protocol (see :class:`PrincipleAssessment` and :func:`assess_item`)."""
    proto = _require_v3(protocol)
    p = v1._principle_key(principle, "principle")
    rule = proto.status_rule if status_rule is None else _coerce_rule(status_rule)
    uncal = _uncalibrated_set(uncalibrated_se_methods)
    items = list(items or ())
    for ev in items:
        if not isinstance(ev, v1.ComponentEvidence):
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
        reasons = (f"{v1.REASON_ABSENT}:{p}",)
    else:
        reasons = tuple(dict.fromkeys(u_codes))
    reason = deciding.reason if status is UNDEFINED else None
    flags = tuple(deciding.flags) + flags_extra
    return PrincipleAssessment(
        p, status, reason, flags, reasons, statuses, ch_reasons, assessments,
        deciding, margin, ignored, tuple(kept))


@dataclass
class MPCVerdictV2(v1.MPCVerdict):
    """
    Result of :func:`verdict_v3`: the fields of the v1
    :class:`~impact_pipeline.evidence.MPCVerdict` (the verdict over the
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
    schema: str = PROTOCOL_SCHEMA_V3

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
        out.append(v1.REASON_BEARER_MISMATCH)
    if bearer_coherence is not None and not v1._coherence_ok(bearer_coherence):
        out.append(f"{v1.REASON_BEARER_MISMATCH}:COHERENCE")
    if require_same_protocol:
        ids = {e.protocol_id for e in gated} - {None}
        if len(ids) > 1 or (ids and ids != {proto.protocol_id}):
            out.append(v1.REASON_PROTOCOL_MISMATCH)
    if proto.source_rule == "single_source":
        why = v1._source_reason(gated, joint_dependence)
        if why is not None:
            out.append(why)
    return out


def verdict_v3(evidence: Mapping, protocol, *, registry=None, regime=None,
               joint_dependence=None, bearer_coherence=None, require_same_protocol=True,
               uncalibrated_se_methods=(), status_rule=None,
               scale="construct", joint_dependence_declared=None) -> MPCVerdictV2:
    """
    The three-valued verdict under a /3 protocol: every principle with
    evidence or in the declared necessity set is assessed
    (:func:`assess_principle`); the verdict is the strong-Kleene AND over the
    protocol's necessity set (N_anch), forced to UNDETERMINED by a global
    code (``BEARER_MISMATCH``, ``PROTOCOL_MISMATCH``, ``SOURCE_INCOHERENT``,
    as v1); the AND over the declared set is reported as
    ``verdict_declared`` (descriptive; ``joint_dependence_declared`` is the
    joint-dependence result over its node sets). ``status_rule`` replaces
    the protocol's rule for a sensitivity analysis (for example
    ``alpha / |N_anch|``); ``scale="amplitude"`` re-judges the quadratic
    statistics on the amplitude scale.
    """
    proto = _require_v3(protocol)
    rule = proto.status_rule if status_rule is None else _coerce_rule(status_rule)
    items_by_p = {}
    for key, items in dict(evidence or {}).items():
        p = v1._principle_key(key, "evidence")
        if p != key:
            raise ValueError(f"evidence key {key!r} must be a canonical principle name")
        items_by_p[p] = [items] if isinstance(items, v1.ComponentEvidence) else list(
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
    return MPCVerdictV2(
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


def mpc_verdict(evidence: Mapping, protocol=None, **kwargs):
    """
    The dispatcher: a /3 protocol is judged by :func:`verdict_v3`; no
    protocol or a /2 protocol goes, with every argument unchanged, to the
    frozen v1 :func:`impact_pipeline.evidence.mpc_verdict` (v1 results bit
    for bit).
    """
    proto = None if protocol is None else load_protocol(protocol)
    if isinstance(proto, ProtocolV3):
        return verdict_v3(evidence, proto, **kwargs)
    return _v1_mpc_verdict(evidence, protocol, **kwargs)


def assess_component(ev, protocol=None, **kwargs):
    """
    The dispatcher for one evidence item: a /3 protocol gives
    :func:`assess_item`; a /2 protocol the v1 ``component_assessment`` at the
    protocol's cutoff and alpha (as the v1 verdict computes it); no protocol
    the v1 ``component_assessment`` with ``kwargs`` (``cutoff``, ``alpha``).
    """
    proto = load_protocol(protocol)
    if isinstance(proto, ProtocolV3):
        return assess_item(ev, proto, **kwargs)
    if proto is None:
        return _v1_component_assessment(ev, **kwargs)
    if kwargs:
        raise TypeError("a /2 protocol fixes cutoff and alpha; pass no keywords")
    return _v1_component_assessment(ev, cutoff=proto.cutoff_for(ev.principle),
                                    alpha=proto.alpha)


__all__ = [
    "ABSENT_TESTS",
    "ADMISSION_VALUES",
    "Admission",
    "AssessmentV2",
    "CHECK_ORDER",
    "ComponentEvidenceV2",
    "DECLARATION_LEVELS",
    "DEFAULT_ALPHA",
    "DEFAULT_ALPHA_ABSENT",
    "DEFAULT_CUTOFF",
    "DEFAULT_STATUS_RULE",
    "DIRECTION_FIELDS",
    "MPCVerdictV2",
    "NAS_DIRECTIONS",
    "N_DECL",
    "PROTOCOL_SCHEMAS",
    "PROTOCOL_SCHEMA_V2",
    "PROTOCOL_SCHEMA_V3",
    "PrincipleAssessment",
    "ProtocolV3",
    "REFERENCE_KINDS_V3",
    "REFERENCE_PENDING",
    "SCALE_TYPES",
    "SE_METHODS",
    "SE_METHOD_CONCORDANT",
    "SE_METHOD_EXACT",
    "STATUS_RULE_KEYS",
    "STATUS_RULE_VERSION",
    "SeMethod",
    "StatusArrays",
    "StatusRule",
    "amplitude_cutoff",
    "amplitude_value",
    "assess_array",
    "assess_component",
    "assess_item",
    "assess_principle",
    "directional_codes",
    "is_quadratic",
    "is_v3",
    "kleene_verdict_codes",
    "load_protocol",
    "member_levels",
    "mpc_verdict",
    "protocol_schema",
    "quantile",
    "registry_admission",
    "status_c",
    "status_c_directional",
    "verdict_from_reasons",
    "verdict_v3",
    "with_protocol_reference",
]
