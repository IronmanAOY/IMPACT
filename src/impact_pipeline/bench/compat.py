"""
Compatibility shim between MPC-Bench and the evidence layer
(``impact_pipeline.evidence``) across its verdict-name revision.

The necessity-only stance renames the verdicts:

    ATTRIBUTED      -> MPC_CONSISTENT  (every principle in N is PRESENT: the
                                        MPC stance does not exclude
                                        consciousness; no sufficiency claim)
    NOT_ATTRIBUTED  -> EXCLUDED        (some principle in N is credibly ABSENT)
    UNDETERMINED    -> UNDETERMINED

The bench is written against the v2 names. This module accepts either naming
from an installed evidence layer (or from older result files and witness
catalogues) and always returns v2 names. It also builds ``ComponentEvidence``
items with only the fields the installed dataclass declares, and calls
``mpc_verdict`` with the v1 keyword interface (``necessity_set=``) or the v2
protocol interface (``mpc_verdict(evidence, protocol)``), whichever the
installed module offers.
"""

from __future__ import annotations

import dataclasses
import inspect
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence

MPC_CONSISTENT = "MPC_CONSISTENT"
EXCLUDED = "EXCLUDED"
UNDETERMINED = "UNDETERMINED"
VERDICTS = (MPC_CONSISTENT, EXCLUDED, UNDETERMINED)
V1_TO_V2 = {
    "ATTRIBUTED": MPC_CONSISTENT,
    "NOT_ATTRIBUTED": EXCLUDED,
    "UNDETERMINED": UNDETERMINED,
}
PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")


def verdict_name(value) -> Optional[str]:
    """
    v2 verdict name of ``value`` (an Enum member, a string in either naming,
    or None). Unknown strings raise ``ValueError``; None stays None.
    """
    if value is None:
        return None
    txt = value.value if isinstance(value, Enum) else str(value)
    txt = txt.strip().upper()
    if txt in VERDICTS:
        return txt
    if txt in V1_TO_V2:
        return V1_TO_V2[txt]
    raise ValueError(f"not an MPC verdict: {value!r}")


def verdict_names(values: Sequence) -> list:
    return [verdict_name(v) for v in values]


def _status_name(value) -> Optional[str]:
    if value is None:
        return None
    return value.value if isinstance(value, Enum) else str(value)


def evidence_field_names(ev_module) -> tuple:
    """Field names of the installed ``ComponentEvidence`` dataclass."""
    cls = ev_module.ComponentEvidence
    if dataclasses.is_dataclass(cls):
        return tuple(f.name for f in dataclasses.fields(cls))
    return tuple(inspect.signature(cls).parameters)


def make_component_evidence(ev_module, **fields):
    """``ComponentEvidence`` with the fields the installed class accepts
    (others, e.g. ``exact`` or ``reference`` on a v1 layer, are dropped)."""
    names = set(evidence_field_names(ev_module))
    kw = {k: v for k, v in fields.items() if k in names}
    return ev_module.ComponentEvidence(**kw)


def _accepted(fn, kwargs: Mapping) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def build_protocol(ev_module, necessity_set, channels, **kwargs):
    """
    v2 ``Protocol`` of the installed layer with the keywords it accepts, or
    None on a v1 layer (no ``Protocol`` class).
    """
    proto_cls = getattr(ev_module, "Protocol", None)
    if proto_cls is None:
        return None
    kw = {"necessity_set": tuple(necessity_set), "channels": dict(channels)}
    kw.update(kwargs)
    return proto_cls(**_accepted(proto_cls, kw))


def call_mpc_verdict(
    ev_module, items: Mapping, necessity_set=PRINCIPLES, protocol_kwargs=None
):
    """
    Run the installed ``mpc_verdict`` on ``items`` (principle -> list of
    ComponentEvidence): with a v2 protocol object when the layer defines
    ``Protocol``, else with ``necessity_set=``. Returns the layer's result.
    """
    nset = tuple(necessity_set)
    channels = {
        p: tuple(dict.fromkeys(getattr(e, "channel", "default") for e in items[p]))
        for p in nset
        if p in items
    }
    proto = build_protocol(ev_module, nset, channels, **(protocol_kwargs or {}))
    if proto is not None:
        return ev_module.mpc_verdict(items, proto)
    return ev_module.mpc_verdict(items, necessity_set=nset)


def verdict_record(result) -> dict:
    """Plain dict (v2 names) from a layer's verdict object."""
    verdict = getattr(result, "verdict", None)
    return {
        "verdict": verdict_name(verdict),
        "reasons": list(getattr(result, "reasons", []) or []),
        "component_status": {
            k: _status_name(s)
            for k, s in (getattr(result, "component_status", {}) or {}).items()
        },
        "margins": dict(getattr(result, "margins", {}) or {}),
    }


def normalise_counts(counts: Mapping) -> Dict[str, int]:
    """Verdict counts keyed in either naming -> v2 names (summed)."""
    out = {v: 0 for v in VERDICTS}
    for k, n in counts.items():
        out[verdict_name(k)] += int(n)
    return out
