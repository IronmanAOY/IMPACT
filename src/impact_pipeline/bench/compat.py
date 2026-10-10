"""
Verdict names of MPC-Bench and calls into the evidence layer
(``impact_pipeline.evidence``).

The necessity-only stance renames the verdicts:

    ATTRIBUTED      -> MPC_CONSISTENT  (every principle in N is PRESENT: the
                                        MPC stance does not exclude
                                        consciousness; no sufficiency claim)
    NOT_ATTRIBUTED  -> EXCLUDED        (some principle in N is credibly ABSENT)
    UNDETERMINED    -> UNDETERMINED

The bench is written against the v2 names. This module maps the v1 names of
older result files and witness catalogues to them and always returns v2
names. It also builds ``ComponentEvidence`` items with only the fields the
dataclass declares, and calls ``mpc_verdict`` with a ``Protocol``.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Dict, Mapping, Optional

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


def make_component_evidence(ev_module, **fields):
    """``ComponentEvidence`` with the fields the dataclass declares (others
    are dropped)."""
    names = {f.name for f in dataclasses.fields(ev_module.ComponentEvidence)}
    kw = {k: v for k, v in fields.items() if k in names}
    return ev_module.ComponentEvidence(**kw)


def call_mpc_verdict(ev_module, items: Mapping, necessity_set=PRINCIPLES):
    """
    Run ``mpc_verdict`` on ``items`` (principle -> list of ComponentEvidence)
    with a ``Protocol`` of ``necessity_set`` and the channels of the items.
    Returns the layer's result.
    """
    nset = tuple(necessity_set)
    channels = {
        p: tuple(dict.fromkeys(getattr(e, "channel", "default") for e in items[p]))
        for p in nset
        if p in items
    }
    proto = ev_module.Protocol(necessity_set=nset, channels=channels)
    return ev_module.mpc_verdict(items, proto)


def normalise_counts(counts: Mapping) -> Dict[str, int]:
    """Verdict counts keyed in either naming -> v2 names (summed)."""
    out = {v: 0 for v in VERDICTS}
    for k, n in counts.items():
        out[verdict_name(k)] += int(n)
    return out
