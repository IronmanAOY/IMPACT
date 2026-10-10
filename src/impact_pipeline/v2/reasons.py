"""
Central vocabulary of UNDEFINED reasons and flags for MPC-Bench v2.

Every reason a v2 component can carry is defined here, once. A reason
explains why a component is UNDEFINED; it never makes a component ABSENT
(``UNDEFINED`` is not ``ABSENT``, part of the fixed core). The evidence layer
maps every reason to UNDEFINED through :func:`status_for_reason`, and the unit
tests and the integrity audit enumerate :data:`REASONS` to check it.

Format. A reason is ``CODE`` or ``CODE:detail``. The code is one of
:data:`REASONS`; whether a detail is allowed, required or restricted to a
fixed set is declared per code (for example ``ESTIMATOR_ERROR:ValueError``,
``ESTIMATOR_NOT_VALIDATED:not_admitted``, ``INCONCLUSIVE:NULL_NOT_EXCEEDED``).
A detail never contains the verdict separator ``;``.

Flags (:data:`FLAGS`) are markers written next to a status: they never are a
status and never replace a reason (for example ``PRESENT_NOT_REACHABLE`` on an
UNDEFINED component, or the concordance route markers of PDI).

Verdict-level codes of the v1 evidence layer (``ABSENT:<P>``,
``MISSING:<P>``, ``BEARER_MISMATCH`` ...) are built from these component
reasons by the evidence layer; ``ABSENT`` is a status there, not a reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

PRESENT, ABSENT, UNDEFINED = "PRESENT", "ABSENT", "UNDEFINED"
STATUSES = (PRESENT, ABSENT, UNDEFINED)
SEPARATOR = ":"
VERDICT_SEPARATOR = ";"

# Detail policies of a reason code.
DETAIL_NONE = "none"  # the bare code only
DETAIL_OPTIONAL = "optional"  # bare code or CODE:<free detail>
DETAIL_REQUIRED = "required"  # CODE:<free detail> only
DETAIL_ENUM = "enum"  # CODE:<one of details> only
DETAIL_ENUM_OPTIONAL = "enum_optional"  # bare code or CODE:<one of details>
DETAIL_POLICIES = (
    DETAIL_NONE,
    DETAIL_OPTIONAL,
    DETAIL_REQUIRED,
    DETAIL_ENUM,
    DETAIL_ENUM_OPTIONAL,
)
_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_TYPE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


class UnknownReasonError(ValueError):
    """A reason string whose code (or detail) is not in the vocabulary."""


@dataclass(frozen=True)
class Reason:
    """One UNDEFINED reason: its code, the component that emits it, its
    meaning, the detail policy and, for enumerated details, the allowed
    values. ``v1`` marks reasons inherited unchanged from the v1 evidence
    layer."""

    code: str
    source: str
    meaning: str
    detail: str = DETAIL_NONE
    details: Tuple[str, ...] = ()
    v1: bool = False

    @property
    def status(self) -> str:
        """The status every reason maps to."""
        return UNDEFINED


@dataclass(frozen=True)
class Flag:
    """A marker written next to a status (never a status, never a reason)."""

    code: str
    source: str
    meaning: str


# --------------------------------------------------------------------------
# reason codes
# --------------------------------------------------------------------------
# v1, unchanged
INVALID_ANCHORS = "INVALID_ANCHORS"
INCONCLUSIVE = "INCONCLUSIVE"
NO_SAMPLING_SE = "NO_SAMPLING_SE"
NULL_FAMILY_MISMATCH = "NULL_FAMILY_MISMATCH"
MISSING_CHANNEL = "MISSING_CHANNEL"
# further reasons of the v1 component assessment and verdict, kept unchanged
# because the v2 evidence layer keeps the v1 validity checks
NO_NULL_CALIBRATION = "NO_NULL_CALIBRATION"
DEGENERATE_NULL = "DEGENERATE_NULL"
INVALID_SE = "INVALID_SE"
NON_FINITE_ESTIMATE = "NON_FINITE_ESTIMATE"
NOT_DEFINED = "NOT_DEFINED"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
MISSING = "MISSING"
# v2
ABSENT_NOT_REACHABLE = "ABSENT_NOT_REACHABLE"
NULL_MODEL_VIOLATED = "NULL_MODEL_VIOLATED"
SAMPLING_UNRESOLVED = "SAMPLING_UNRESOLVED"
OBSERVATION_MIXED_NOT_ADMITTED = "OBSERVATION_MIXED_NOT_ADMITTED"
INSUFFICIENT_OCCUPANCY = "INSUFFICIENT_OCCUPANCY"
MACRO_RANK_DEFICIENT = "MACRO_RANK_DEFICIENT"
NOT_APPLICABLE_OBSERVATION_MODEL = "NOT_APPLICABLE_OBSERVATION_MODEL"
INSUFFICIENT_TIMEPOINTS = "INSUFFICIENT_TIMEPOINTS"
INSUFFICIENT_UPDATES = "INSUFFICIENT_UPDATES"
ESTIMATOR_ERROR = "ESTIMATOR_ERROR"
SE_NOT_CALIBRATED = "SE_NOT_CALIBRATED"
ESTIMATOR_NOT_VALIDATED = "ESTIMATOR_NOT_VALIDATED"

# details
NULL_NOT_EXCEEDED = "NULL_NOT_EXCEEDED"  # IIM rank gate failed
NOT_ADMITTED = "not_admitted"
NOT_OBSERVABLE = "not_observable"

_REASON_LIST = (
    Reason(
        INVALID_ANCHORS,
        "v1 evidence layer",
        "the reference anchor is not finite or does not lie above the null mean",
        v1=True,
    ),
    Reason(
        INCONCLUSIVE,
        "v1 evidence layer; status rule; IIM rank gate",
        "neither PRESENT nor ABSENT at this precision; detail NULL_NOT_EXCEEDED "
        "when the IIM rank gate failed (p_ind > 0.05)",
        detail=DETAIL_ENUM_OPTIONAL,
        details=(NULL_NOT_EXCEEDED,),
        v1=True,
    ),
    Reason(
        NO_SAMPLING_SE,
        "v1 evidence layer",
        "an empirical estimate without a sampling SE (a data-derived se = 0 is "
        "never exact)",
        v1=True,
    ),
    Reason(
        NULL_FAMILY_MISMATCH,
        "v1 evidence layer",
        "the component's null family differs from the protocol's declared "
        "family; detail '<declared>/<actual>'",
        detail=DETAIL_OPTIONAL,
        v1=True,
    ),
    Reason(
        MISSING_CHANNEL,
        "v1 evidence layer",
        "a channel declared by the protocol has no evidence item; detail: the "
        "channel",
        detail=DETAIL_OPTIONAL,
        v1=True,
    ),
    Reason(
        NO_NULL_CALIBRATION,
        "v1 evidence layer; IIM v5 independence null",
        "no null mean, or fewer than 19 finite null draws",
        v1=True,
    ),
    Reason(
        DEGENERATE_NULL,
        "v1 evidence layer",
        "invalid null size, or a Monte-Carlo null without a finite SD",
        v1=True,
    ),
    Reason(
        INVALID_SE,
        "v1 evidence layer",
        "a negative SE or reference SE, or se_df <= 0",
        v1=True,
    ),
    Reason(
        NON_FINITE_ESTIMATE,
        "v1 evidence layer",
        "the estimator returned a non-finite value",
        v1=True,
    ),
    Reason(
        NOT_DEFINED,
        "v1 evidence layer; v1 estimators on the v2 path",
        "the estimator declared its value undefined; detail: the estimator's "
        "own reason",
        detail=DETAIL_OPTIONAL,
        v1=True,
    ),
    Reason(
        NOT_IMPLEMENTED,
        "v1 evidence layer",
        "a declared evidence channel is not implemented (for example the "
        "perturbational and endogenous RAM channels); detail: the channel",
        detail=DETAIL_OPTIONAL,
        v1=True,
    ),
    Reason(
        MISSING,
        "v1 evidence layer",
        "no evidence for a principle of the necessity set",
        v1=True,
    ),
    Reason(
        ABSENT_NOT_REACHABLE,
        "status rule",
        "q_A se_c >= delta: the equivalence test cannot pass at this precision",
    ),
    Reason(
        NULL_MODEL_VIOLATED,
        "status rule",
        "c + q_P se_c < -delta: credibly below the null by more than the margin",
    ),
    Reason(
        SAMPLING_UNRESOLVED,
        "NAS v3 resolvability gate",
        "dt > tau_c / 2: the sampling does not resolve the coupling time scale",
    ),
    Reason(
        OBSERVATION_MIXED_NOT_ADMITTED,
        "observation gates of NAS v3 and IIM v5",
        "sensor or source-estimate observation without an admitting registry "
        "entry",
    ),
    Reason(
        INSUFFICIENT_OCCUPANCY,
        "IIM v5 occupancy gate",
        "a macro state unvisited, or the rarest visited row has fewer than "
        "N_min transition pairs",
    ),
    Reason(
        MACRO_RANK_DEFICIENT,
        "IIM v5 rank condition of the macro signals",
        "zero-lag correlation of the cluster means with "
        "lambda_min / lambda_max < 1e-6",
    ),
    Reason(
        NOT_APPLICABLE_OBSERVATION_MODEL,
        "SRPI, RAM-PE v3 and PDI v3 on family C1",
        "declared inapplicable to the observation model (C1 carrier recording)",
    ),
    Reason(
        INSUFFICIENT_TIMEPOINTS,
        "NAS v3 secondary representation",
        "T - l_max < 10 n_par at every allowed block dimension",
    ),
    Reason(
        INSUFFICIENT_UPDATES,
        "RAM-PE v3",
        "fewer than 30 updates or fewer than two options",
    ),
    Reason(
        ESTIMATOR_ERROR,
        "v2 runner",
        "the estimator raised; detail: the exception type; other components "
        "are unaffected",
        detail=DETAIL_REQUIRED,
    ),
    Reason(
        SE_NOT_CALIBRATED,
        "HCv2-4 reversion rule",
        "the SE method was found anti-conservative on the twins",
    ),
    Reason(
        ESTIMATOR_NOT_VALIDATED,
        "registry v3",
        "no admitting registry entry for the substrate and regime, for this "
        "status direction (not_admitted) or for this observation "
        "(not_observable)",
        detail=DETAIL_ENUM,
        details=(NOT_ADMITTED, NOT_OBSERVABLE),
    ),
)

# --------------------------------------------------------------------------
# flags
# --------------------------------------------------------------------------
PRESENT_NOT_REACHABLE = "PRESENT_NOT_REACHABLE"
ANCHOR_NOT_REPLICATED = "ANCHOR_NOT_REPLICATED"
HUB_NOT_PRIVILEGED = "HUB_NOT_PRIVILEGED"
ABSENT_BY_CONCORDANCE = "ABSENT_BY_CONCORDANCE"
PRESENT_BY_CONCORDANCE = "PRESENT_BY_CONCORDANCE"
NONSPECIFIC_ANCHOR = "NONSPECIFIC_ANCHOR"

_FLAG_LIST = (
    Flag(
        PRESENT_NOT_REACHABLE,
        "status rule",
        "UNDEFINED and q_P se_c >= 1 - z: PRESENT cannot be reached at this "
        "precision",
    ),
    Flag(
        ANCHOR_NOT_REPLICATED,
        "HCv2-6",
        "the anchor status did not replicate on the confirmatory replication "
        "block; the frozen protocol is kept",
    ),
    Flag(
        HUB_NOT_PRIVILEGED,
        "NAS hub identity (Tier B2)",
        "the declared hub is not privileged over every pseudo-hub; restricts "
        "the construct wording, never the status",
    ),
    Flag(
        ABSENT_BY_CONCORDANCE,
        "PDI concordance route",
        "route marker: ABSENT through the admitted concordance route",
    ),
    Flag(
        PRESENT_BY_CONCORDANCE,
        "PDI concordance route",
        "route marker: PRESENT through the admitted concordance route",
    ),
    Flag(
        NONSPECIFIC_ANCHOR,
        "anchor specificity gate",
        "the anchor failed the specificity gate; the principle leaves N_anch "
        "but keeps its component-level evaluation",
    ),
)


def _registry(items, what):
    out = {}
    for item in items:
        if not _CODE_RE.match(item.code):
            raise ValueError(f"malformed {what} code {item.code!r}")
        if item.code in out:
            raise ValueError(f"duplicate {what} code {item.code!r}")
        out[item.code] = item
    return MappingProxyType(out)


REASONS: Mapping[str, Reason] = _registry(_REASON_LIST, "reason")
FLAGS: Mapping[str, Flag] = _registry(_FLAG_LIST, "flag")
if set(REASONS) & set(FLAGS):  # pragma: no cover - guarded by the tests
    raise ValueError("a code is both a reason and a flag")
if set(REASONS) & set(STATUSES):  # pragma: no cover - guarded by the tests
    raise ValueError("a reason code equals a status")
V1_REASONS = tuple(code for code, r in REASONS.items() if r.v1)
V2_REASONS = tuple(code for code, r in REASONS.items() if not r.v1)
ROUTE_MARKERS = {ABSENT_BY_CONCORDANCE: ABSENT, PRESENT_BY_CONCORDANCE: PRESENT}


# --------------------------------------------------------------------------
# parsing and formatting
# --------------------------------------------------------------------------
def parse_reason(text) -> Tuple[str, Optional[str]]:
    """Split ``CODE[:detail]`` into ``(code, detail or None)`` (no checks)."""
    code, sep, detail = str(text).partition(SEPARATOR)
    return code, (detail if sep else None)


def _check_detail(reason: Reason, detail: Optional[str], text: str) -> None:
    policy = reason.detail
    if detail is None:
        if policy in (DETAIL_REQUIRED, DETAIL_ENUM):
            raise UnknownReasonError(f"{text!r}: {reason.code} needs a detail")
        return
    if detail == "" or VERDICT_SEPARATOR in detail or "\n" in detail:
        raise UnknownReasonError(f"{text!r}: malformed detail")
    if policy == DETAIL_NONE:
        raise UnknownReasonError(f"{text!r}: {reason.code} takes no detail")
    if policy in (DETAIL_ENUM, DETAIL_ENUM_OPTIONAL) and detail not in reason.details:
        raise UnknownReasonError(
            f"{text!r}: detail must be one of {reason.details}"
        )


def lookup(text) -> Reason:
    """The :class:`Reason` of a reason string; raises
    :class:`UnknownReasonError` for an unknown code, a flag, a status or a
    detail that the code does not allow."""
    if text is None:
        raise UnknownReasonError("no reason given")
    text = str(text)
    code, detail = parse_reason(text)
    if code in FLAGS:
        raise UnknownReasonError(f"{text!r} is a flag, not a reason")
    if code in STATUSES:
        raise UnknownReasonError(f"{text!r} is a status, not a reason")
    reason = REASONS.get(code)
    if reason is None:
        raise UnknownReasonError(f"unknown reason code {code!r} in {text!r}")
    _check_detail(reason, detail, text)
    return reason


def is_reason(text) -> bool:
    """True iff ``text`` is a valid reason string of the vocabulary."""
    try:
        lookup(text)
    except UnknownReasonError:
        return False
    return True


def clean_detail(text) -> str:
    """A free-text detail made safe for a reason string (``;`` becomes
    ``,``, line breaks become spaces), as the v1 evidence layer does."""
    return str(text).replace(VERDICT_SEPARATOR, ",").replace("\n", " ").strip()


def format_reason(code: str, detail=None) -> str:
    """``CODE`` or ``CODE:detail``, validated against the vocabulary."""
    text = str(code) if detail is None else f"{code}{SEPARATOR}{detail}"
    lookup(text)
    return text


def status_for_reason(text) -> str:
    """The status a reason implies: UNDEFINED for every reason of the
    vocabulary (never ABSENT); unknown strings raise
    :class:`UnknownReasonError`, so a new reason must be registered here
    before any component can carry it."""
    return lookup(text).status


def estimator_error(exc) -> str:
    """``ESTIMATOR_ERROR:<type>`` for an exception, exception class or type
    name (the runner records the message separately)."""
    if isinstance(exc, BaseException):
        name = type(exc).__name__
    elif isinstance(exc, type) and issubclass(exc, BaseException):
        name = exc.__name__
    else:
        name = str(exc)
    if not _TYPE_NAME_RE.match(name):
        raise UnknownReasonError(f"not an exception type name: {name!r}")
    return format_reason(ESTIMATOR_ERROR, name)


def not_validated(kind: str = NOT_ADMITTED) -> str:
    """``ESTIMATOR_NOT_VALIDATED:not_admitted`` or ``...:not_observable``."""
    return format_reason(ESTIMATOR_NOT_VALIDATED, kind)


def is_estimator_error(text) -> bool:
    return text is not None and parse_reason(text)[0] == ESTIMATOR_ERROR


def check_flag(code) -> Flag:
    """The :class:`Flag` of a flag code; raises ``ValueError`` otherwise."""
    flag = FLAGS.get(str(code))
    if flag is None:
        raise ValueError(f"unknown flag {code!r}")
    return flag


def check_status(status, reason=None, flags=()) -> None:
    """
    Validate a component's ``(status, reason, flags)``: the status is one of
    :data:`STATUSES`; UNDEFINED carries a reason of the vocabulary; PRESENT and
    ABSENT carry no reason; every flag is known; ``PRESENT_NOT_REACHABLE``
    only on UNDEFINED; a concordance route marker only on its own status.
    Raises ``ValueError`` (``UnknownReasonError`` for a bad reason).
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    if status == UNDEFINED:
        lookup(reason)
    elif reason is not None:
        raise ValueError(f"a {status} component carries no reason (got {reason!r})")
    for f in flags:
        check_flag(f)
        if f == PRESENT_NOT_REACHABLE and status != UNDEFINED:
            raise ValueError("PRESENT_NOT_REACHABLE marks UNDEFINED components only")
        if f in ROUTE_MARKERS and ROUTE_MARKERS[f] != status:
            raise ValueError(f"{f} marks {ROUTE_MARKERS[f]} components only")


def reason_table() -> list:
    """Rows ``{code, source, meaning, detail, details, v1, status}`` of every
    reason, in declaration order (for documentation and the audit)."""
    return [
        {
            "code": r.code,
            "source": r.source,
            "meaning": r.meaning,
            "detail": r.detail,
            "details": list(r.details),
            "v1": r.v1,
            "status": r.status,
        }
        for r in REASONS.values()
    ]


__all__ = [
    "ABSENT",
    "DETAIL_POLICIES",
    "FLAGS",
    "Flag",
    "PRESENT",
    "REASONS",
    "ROUTE_MARKERS",
    "Reason",
    "STATUSES",
    "UNDEFINED",
    "UnknownReasonError",
    "V1_REASONS",
    "V2_REASONS",
    "check_flag",
    "check_status",
    "clean_detail",
    "estimator_error",
    "format_reason",
    "is_estimator_error",
    "is_reason",
    "lookup",
    "not_validated",
    "parse_reason",
    "reason_table",
    "status_for_reason",
]
