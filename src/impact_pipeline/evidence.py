"""
Evidence layer: null-anchored component evidence, the three-valued MPC verdict
and the MPC degree.

Terminology (code, docs and paper):

- **MPC profile**: the five null-anchored components (RAM, PDI, NAS, IIM, SRPI)
  with their null calibration and margins.
- **MPC verdict**: ``ATTRIBUTED`` / ``NOT_ATTRIBUTED`` / ``UNDETERMINED`` plus
  stable reason codes. Each component is PRESENT (T), ABSENT (F) or UNDEFINED
  (U) relative to its declared null family; channels of one principle are
  combined by strong-Kleene OR, principles of the necessity set ``N`` by
  strong-Kleene AND. Missing or undefined evidence can only make the verdict
  UNDETERMINED; it never becomes a zero, and it never flips a determinate
  verdict.
- **MPC degree**: a capped power mean of reference-normalised components
  (two-anchor scale: 0 = null, 1 = reference), computed only when the verdict
  is ATTRIBUTED. It is a reference-relative evidence summary, not a "level of
  consciousness" and not an anaesthesia-depth index.

Reason codes (stable strings, ``;``-joined in tables):

- ``MISSING:<P>``: no evidence for principle ``P`` in the necessity set.
- ``NO_NULL_CALIBRATION:<P>``: a defined estimate without a null family.
- ``INCONCLUSIVE:<P>``: neither significantly above the null nor equivalent to it.
- ``UNDEFINED:<P>:<reason>``: the estimator (or the null) is undefined.
- ``NOT_IMPLEMENTED:<P>:<channel>``: the evidence channel is not implemented.
- ``ESTIMATOR_NOT_VALIDATED:<P>:<estimator>``: the applicability registry does
  not validate the estimator for this substrate/grain/regime.
- ``ABSENT:<P>``: every channel of ``P`` is credibly null (veto).
- ``BEARER_MISMATCH`` / ``PROTOCOL_MISMATCH``: the evidence does not come from
  one declared bearer / protocol (``BEARER_MISMATCH:COHERENCE`` when a
  bearer-coherence diagnostic failed). These force UNDETERMINED.

The verdict is recoverable from the reasons alone (see
:func:`verdict_from_reasons`): no reasons <=> ATTRIBUTED; a bearer/protocol
code => UNDETERMINED; otherwise any ``ABSENT`` code => NOT_ATTRIBUTED; else
UNDETERMINED. The reasons of a verdict over ``N`` are the union of the
single-principle reasons over ``P in N`` plus the global codes.
"""
from __future__ import annotations

import fnmatch
import functools
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist
from typing import Mapping

import numpy as np

PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")


class Verdict(str, Enum):
    ATTRIBUTED = "ATTRIBUTED"
    NOT_ATTRIBUTED = "NOT_ATTRIBUTED"
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
    CODE_T: Verdict.ATTRIBUTED,
    CODE_U: Verdict.UNDETERMINED,
    CODE_F: Verdict.NOT_ATTRIBUTED,
}
_VERDICT_TO_CODE = {v: k for k, v in _CODE_TO_VERDICT.items()}

REASON_MISSING = "MISSING"
REASON_UNDEFINED = "UNDEFINED"
REASON_INCONCLUSIVE = "INCONCLUSIVE"
REASON_ABSENT = "ABSENT"
REASON_NO_NULL = "NO_NULL_CALIBRATION"
REASON_DEGENERATE_NULL = "DEGENERATE_NULL"
REASON_NOT_VALIDATED = "ESTIMATOR_NOT_VALIDATED"
REASON_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
REASON_BEARER_MISMATCH = "BEARER_MISMATCH"
REASON_PROTOCOL_MISMATCH = "PROTOCOL_MISMATCH"
GLOBAL_REASON_KINDS = (REASON_BEARER_MISMATCH, REASON_PROTOCOL_MISMATCH)
PRINCIPLE_REASON_KINDS = (
    REASON_MISSING,
    REASON_NO_NULL,
    REASON_INCONCLUSIVE,
    REASON_UNDEFINED,
    REASON_NOT_IMPLEMENTED,
    REASON_NOT_VALIDATED,
    REASON_ABSENT,
)
REASON_SEPARATOR = ";"


# --------------------------------------------------------------------------
# component evidence and status
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentEvidence:
    """
    One piece of null-anchored evidence for one principle through one channel.

    ``estimate`` is the raw estimator value; ``null_mean``/``null_sd`` are the
    moments of the declared null family (same estimator and configuration on
    surrogate data, ``n_null`` draws); ``se`` is the sampling SE of the
    estimate (0 when unknown). ``reference`` is the unit anchor used by
    :func:`two_anchor_normalize`. ``bearer_id``/``protocol_id`` declare the
    system and protocol the evidence was measured on (Principle 0).
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


@functools.lru_cache(maxsize=64)
def _z_quantile(alpha):
    return float(NormalDist().inv_cdf(1.0 - float(alpha)))


def _check_status_params(z_present, delta_equiv, alpha):
    if not (math.isfinite(z_present) and math.isfinite(delta_equiv)):
        raise ValueError("z_present and delta_equiv must be finite")
    if delta_equiv < 0:
        raise ValueError("delta_equiv must be >= 0")
    if delta_equiv > z_present:
        # Otherwise PRESENT and ABSENT regions could overlap.
        raise ValueError("delta_equiv must be <= z_present")
    if not (0.0 < alpha <= 0.5):
        raise ValueError("alpha must be in (0, 0.5]")


def _as_float(val):
    try:
        return float("nan") if val is None else float(val)
    except (TypeError, ValueError):
        return float("nan")


def component_status(ev, *, z_present=1.645, delta_equiv=1.0, alpha=0.05):
    """
    Three-valued status of one piece of evidence.

    With ``excess = estimate - null_mean`` and ``z_a = z_{1-alpha}``:

    - margin ``z = excess / sqrt(null_sd**2 + se**2)`` (null-SD units);
    - PRESENT if the lower one-sided ``1-alpha`` bound of the excess exceeds
      ``z_present * null_sd``: ``excess - z_a*se > z_present*null_sd``;
    - ABSENT if TOST equivalence to the null holds:
      ``|excess| + z_a*se <= delta_equiv*null_sd``;
    - UNDEFINED otherwise with reason ``INCONCLUSIVE`` (this includes
      estimates far *below* the null, which indicate a misspecified null).

    Evidence that is not defined, has a non-finite estimate, no null
    (``NO_NULL_CALIBRATION``), a degenerate null (``null_sd`` not finite > 0:
    ``DEGENERATE_NULL``) or an invalid ``se`` is UNDEFINED with that reason and
    a NaN margin. Returns ``(ComponentStatus, margin, reason)``.
    """
    _check_status_params(z_present, delta_equiv, alpha)
    nan = float("nan")
    if not bool(ev.defined):
        return ComponentStatus.UNDEFINED, nan, str(ev.reason or "NOT_DEFINED")
    est = _as_float(ev.estimate)
    if not math.isfinite(est):
        return ComponentStatus.UNDEFINED, nan, str(ev.reason or "NON_FINITE_ESTIMATE")
    null_mean = _as_float(ev.null_mean)
    if not math.isfinite(null_mean):
        return ComponentStatus.UNDEFINED, nan, REASON_NO_NULL
    null_sd = _as_float(ev.null_sd)
    if not (math.isfinite(null_sd) and null_sd > 0):
        return ComponentStatus.UNDEFINED, nan, REASON_DEGENERATE_NULL
    se = _as_float(ev.se)
    if not (math.isfinite(se) and se >= 0):
        return ComponentStatus.UNDEFINED, nan, "INVALID_SE"
    z_a = _z_quantile(alpha)
    excess = est - null_mean
    margin = excess / math.sqrt(null_sd * null_sd + se * se)
    if excess - z_a * se > z_present * null_sd:
        return ComponentStatus.PRESENT, margin, None
    if abs(excess) + z_a * se <= delta_equiv * null_sd:
        return ComponentStatus.ABSENT, margin, None
    return ComponentStatus.UNDEFINED, margin, REASON_INCONCLUSIVE


def component_status_array(
    estimate, null_mean, null_sd, se=0.0, *, z_present=1.645, delta_equiv=1.0,
    alpha=0.05,
):
    """
    Vectorised numeric core of :func:`component_status` (same inequalities).
    Returns ``(codes, margins)`` with codes in {1 (PRESENT), 0 (UNDEFINED),
    -1 (ABSENT)}; invalid inputs give code 0 and a NaN margin.
    """
    _check_status_params(z_present, delta_equiv, alpha)
    est, nm, nsd, s = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (estimate, null_mean, null_sd, se))
    )
    valid = (
        np.isfinite(est) & np.isfinite(nm) & np.isfinite(nsd) & (nsd > 0)
        & np.isfinite(s) & (s >= 0)
    )
    z_a = _z_quantile(alpha)
    with np.errstate(invalid="ignore", divide="ignore"):
        excess = est - nm
        margin = np.where(valid, excess / np.sqrt(nsd * nsd + s * s), np.nan)
        present = valid & (excess - z_a * s > z_present * nsd)
        absent = valid & ~present & (np.abs(excess) + z_a * s <= delta_equiv * nsd)
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
    Returns verdict codes (1 ATTRIBUTED, 0 UNDETERMINED, -1 NOT_ATTRIBUTED).
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
# MPC verdict
# --------------------------------------------------------------------------
@dataclass
class MPCVerdict:
    """
    Result of :func:`mpc_verdict`. ``component_status`` holds the per-principle
    status (Kleene OR over channels), ``margins`` the largest finite channel
    margin, ``channels`` the per-channel statuses and ``channel_reasons`` the
    per-channel reason (None when determinate).
    """

    verdict: Verdict
    reasons: list
    component_status: dict
    margins: dict
    channels: dict
    necessity_set: tuple = PRINCIPLES
    channel_reasons: dict = field(default_factory=dict)

    @property
    def reason_string(self) -> str:
        return REASON_SEPARATOR.join(self.reasons)

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
            "necessity_set": list(self.necessity_set),
        }


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


def _clean_detail(text):
    return str(text).replace(REASON_SEPARATOR, ",").replace("\n", " ").strip()


def _channel_reason_code(principle, ev, reason):
    if reason == REASON_INCONCLUSIVE:
        return f"{REASON_INCONCLUSIVE}:{principle}"
    if reason == REASON_NO_NULL:
        return f"{REASON_NO_NULL}:{principle}"
    if reason == REASON_NOT_IMPLEMENTED:
        return f"{REASON_NOT_IMPLEMENTED}:{principle}:{_clean_detail(ev.channel)}"
    if reason is not None and reason.startswith(REASON_NOT_VALIDATED):
        est = _clean_detail(ev.estimator) if ev.estimator else "undeclared"
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
        return Verdict.ATTRIBUTED
    if any(k in GLOBAL_REASON_KINDS for k in kinds):
        return Verdict.UNDETERMINED
    if REASON_ABSENT in kinds:
        return Verdict.NOT_ATTRIBUTED
    return Verdict.UNDETERMINED


def _dedupe(seq):
    return list(dict.fromkeys(seq))


def _coherence_ok(bearer_coherence):
    if isinstance(bearer_coherence, (bool, np.bool_)):
        return bool(bearer_coherence)
    if isinstance(bearer_coherence, Mapping):
        return bool(bearer_coherence.get("coherent", False))
    raise TypeError("bearer_coherence must be a bool or a bearer_coherence() result")


def mpc_verdict(
    evidence: Mapping,
    *,
    necessity_set=PRINCIPLES,
    require_same_bearer=True,
    registry=None,
    require_same_protocol=True,
    regime=None,
    bearer_coherence=None,
    **status_kwargs,
) -> MPCVerdict:
    """
    Three-valued MPC verdict from per-principle evidence.

    ``evidence`` maps a principle to a ComponentEvidence or a sequence of them
    (one per channel). Per principle the channel statuses are combined by
    Kleene OR (a principle is ABSENT only if every channel is ABSENT); the
    verdict is the Kleene AND over ``necessity_set``: T -> ATTRIBUTED,
    F -> NOT_ATTRIBUTED, U -> UNDETERMINED. With ``registry``
    (:class:`ApplicabilityRegistry`), evidence from an estimator that is not
    validated for its principle/substrate/grain/``regime`` is UNDEFINED.
    Evidence of the necessity set that declares more than one ``bearer_id``
    (``require_same_bearer``) or ``protocol_id`` (``require_same_protocol``),
    or a failed ``bearer_coherence`` diagnostic, makes the verdict
    UNDETERMINED whatever the components say. Undefined evidence keeps its
    declared bearer/protocol, so making a component undefined can never
    resolve a mismatch (missingness safety). ``status_kwargs`` go to
    :func:`component_status`.
    """
    nset = normalize_necessity_set(necessity_set)
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
    p_codes, p_reasons = {}, {}
    for p in ordered:
        items = items_by_p.get(p, [])
        channels[p], channel_reasons[p] = {}, {}
        if not items:
            p_codes[p] = CODE_U
            p_reasons[p] = [f"{REASON_MISSING}:{p}"]
            margins[p] = float("nan")
            statuses[p] = ComponentStatus.UNDEFINED
            continue
        codes, u_reasons, best = [], [], float("nan")
        for ev in items:
            reason = None
            if registry is not None:
                ok, why = registry.is_validated(
                    p, ev.estimator, ev.substrate, ev.grain, **regime
                )
                if not ok:
                    reason = f"{REASON_NOT_VALIDATED}:{why}"
            if reason is None:
                st, margin, reason = component_status(ev, **status_kwargs)
            else:
                st, margin = ComponentStatus.UNDEFINED, float("nan")
            code = _STATUS_TO_CODE[st]
            if math.isfinite(margin) and not (margin <= best):
                best = margin
            prev = _STATUS_TO_CODE.get(channels[p].get(ev.channel), CODE_F)
            channels[p][ev.channel] = _CODE_TO_STATUS[max(prev, code)]
            if code == CODE_U:
                channel_reasons[p].setdefault(ev.channel, reason)
                u_reasons.append(_channel_reason_code(p, ev, reason))
            codes.append(code)
        channel_reasons[p] = {
            ch: r for ch, r in channel_reasons[p].items()
            if channels[p][ch] == ComponentStatus.UNDEFINED
        }
        code_p = max(codes)
        p_codes[p] = code_p
        statuses[p] = _CODE_TO_STATUS[code_p]
        margins[p] = best
        if code_p == CODE_T:
            p_reasons[p] = []
        elif code_p == CODE_F:
            p_reasons[p] = [f"{REASON_ABSENT}:{p}"]
        else:
            p_reasons[p] = _dedupe(u_reasons)

    global_reasons = []
    gated = [ev for p in nset for ev in items_by_p.get(p, [])]
    if require_same_bearer and len({e.bearer_id for e in gated} - {None}) > 1:
        global_reasons.append(REASON_BEARER_MISMATCH)
    if bearer_coherence is not None and not _coherence_ok(bearer_coherence):
        global_reasons.append(f"{REASON_BEARER_MISMATCH}:COHERENCE")
    if require_same_protocol and len({e.protocol_id for e in gated} - {None}) > 1:
        global_reasons.append(REASON_PROTOCOL_MISMATCH)

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
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if p == 0.0:
            return np.exp(np.log(x) @ w)
        zero = (x <= 0).any(axis=-1)
        pos = np.where(x > 0, x, 1.0)
        val = ((pos ** p) @ w) ** (1.0 / p) if p < 0 else ((x ** p) @ w) ** (1.0 / p)
        return np.where(zero & (p < 0), 0.0, val)


def degree(c, weights=None, *, p=0.0, cap=1.0) -> float:
    """
    MPC degree: weighted power mean of ``min(c_j, cap)`` over the components
    with positive weight (equal weights by default). ``p=0`` is the geometric
    mean, ``p=-inf`` the weakest link, ``p=1`` the arithmetic mean;
    ``cap=None`` disables the cap. Components are reference-normalised values
    (0 = null, 1 = reference); values below 0 count as 0. NaN if any weighted
    component is non-finite. Only meaningful for ATTRIBUTED verdicts.
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
    capped = (cap is not None) and math.isfinite(float(cap))
    active = (vals < float(cap)) if capped else np.ones(vals.size, dtype=bool)
    x = np.maximum(vals, 0.0)
    grad = np.zeros(vals.size, dtype=float)
    if p in (-math.inf, math.inf):
        xc = np.minimum(x, float(cap)) if capped else x
        j = int(np.argmin(xc) if p == -math.inf else np.argmax(xc))
        grad[j] = 1.0 if active[j] else 0.0
        return grad
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if p == 0.0:
            g = w * d / x
        else:
            g = w * x ** (p - 1.0) * d ** (1.0 - p)
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
      SEs ``se`` (mapping or sequence aligned with ``c``); the cap is treated
      as flat (zero derivative for components at or above it) and the weakest
      link differentiates through its (first) minimising component. The SE is
      NaN where the power mean is not differentiable (a zero component with
      ``p <= 0``).
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
    parameter settings). ``modal_verdict`` is the most frequent verdict (ties
    resolved conservatively: UNDETERMINED, then NOT_ATTRIBUTED, then
    ATTRIBUTED); ``flip_rate`` is the fraction of verdicts that differ from it.
    With ``reference`` (e.g. the verdict on the original data) the fraction
    differing from it is returned as ``reference_flip_rate``.
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
        order = (Verdict.UNDETERMINED, Verdict.NOT_ATTRIBUTED, Verdict.ATTRIBUTED)
        modal = max(order, key=lambda v: (counts[v.value], -order.index(v)))
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
@dataclass(frozen=True)
class RegistryEntry:
    """
    One validated (or explicitly not validated) estimator configuration.

    String fields match with shell wildcards (``*``, ``?``; case-insensitive)
    and may be lists of alternatives. ``regime`` maps a regime key to a
    scalar (equality), a list (membership) or ``{"min": .., "max": ..}``
    (inclusive range); a query that omits a constrained key does not match.
    """

    principle: str
    estimator: object
    substrate: object = "*"
    grain: object = "*"
    regime: Mapping = field(default_factory=dict)
    validated: bool = True
    note: str | None = None


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
        if key not in regime or not _regime_value_matches(spec, regime[key]):
            return str(key)
    return None


class ApplicabilityRegistry:
    """
    Registry of estimator configurations validated (e.g. on MPC-Bench) for a
    principle, substrate (e.g. ``fmri``/``eeg``/``bench_family_a``), grain
    (e.g. an atlas or macro-node grain) and regime. JSON layout::

        {"version": 1,
         "entries": [{"principle": "IIM", "estimator": "compute_IIM:*",
                      "substrate": "fmri", "grain": "schaefer*",
                      "regime": {"n_time": {"min": 300}},
                      "validated": true, "note": "MPC-Bench family B"}]}
    """

    def __init__(self, entries=(), *, source=None, version=None):
        self.entries = tuple(
            e if isinstance(e, RegistryEntry) else self._entry(e) for e in entries
        )
        self.source = source
        self.version = version

    @staticmethod
    def _entry(raw):
        if not isinstance(raw, Mapping):
            raise ValueError("registry entries must be JSON objects")
        missing = [k for k in ("principle", "estimator") if k not in raw]
        if missing:
            raise ValueError(f"registry entry is missing {missing}: {raw}")
        principle = str(raw["principle"]).upper()
        if principle not in PRINCIPLES:
            raise ValueError(f"registry entry has unknown principle {principle!r}")
        regime = raw.get("regime") or {}
        if not isinstance(regime, Mapping):
            raise ValueError("registry entry 'regime' must be an object")
        return RegistryEntry(
            principle=principle,
            estimator=raw["estimator"],
            substrate=raw.get("substrate", "*"),
            grain=raw.get("grain", "*"),
            regime=dict(regime),
            validated=bool(raw.get("validated", True)),
            note=raw.get("note"),
        )

    @classmethod
    def from_dict(cls, payload, source=None):
        if isinstance(payload, Mapping):
            entries = payload.get("entries", [])
            version = payload.get("version")
        elif isinstance(payload, (list, tuple)):
            entries, version = payload, None
        else:
            raise ValueError("registry must be a JSON object or a list of entries")
        return cls(entries, source=source, version=version)

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload, source=str(path))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "entries": [
                {
                    "principle": e.principle,
                    "estimator": e.estimator,
                    "substrate": e.substrate,
                    "grain": e.grain,
                    "regime": dict(e.regime),
                    "validated": e.validated,
                    "note": e.note,
                }
                for e in self.entries
            ],
        }

    def is_validated(self, principle, estimator, substrate=None, grain=None, **regime):
        """
        ``(True, None)`` if a validated entry matches the principle, estimator,
        substrate, grain and regime; otherwise ``(False, reason)`` with reason
        ``ESTIMATOR_UNDECLARED``, ``NO_ENTRY``, ``MARKED_NOT_VALIDATED`` or
        ``REGIME_MISMATCH:<key>``. An explicitly not-validated entry wins over
        a validated one.
        """
        if estimator is None:
            return False, "ESTIMATOR_UNDECLARED"
        p = str(principle).upper()
        cands = [
            e for e in self.entries
            if e.principle == p
            and _match_field(e.estimator, estimator)
            and _match_field(e.substrate, substrate)
            and _match_field(e.grain, grain)
        ]
        if any(not e.validated and _regime_failure(e.regime, regime) is None
               for e in cands):
            return False, "MARKED_NOT_VALIDATED"
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


# --------------------------------------------------------------------------
# bearer coherence (Principle 0)
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
    Bearer-coherence diagnostic (Principle 0): is there measured coupling above
    null between the node sets that supply the components?

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
