"""
RAM-only arm of MPC-Bench v2 (design 2.3 item 6, 3.1, 3.2).

The RAM-PE hypotheses run on family-A agents with 160 bandit trials
(configuration preset ``ram_only_160``; the joint bench stays at the v1
length of 80 trials). Only RAM-PE is scored, under the family protocol
``A-RAM160``, with its component status and no verdict; RAM-PE does not
condition on inputs, so the declaration R is recorded and nothing else.

Classes (11): PC_nominal, W_RAM_no_plasticity, the plasticity doses
``eta`` in {0, 0.1, 0.2, 0.3, 0.6}, the reflex arc, scrambled feedback
(RAM-PE mechanism on, a construct-boundary test), K = 1 and g_b = 0 (the
SNR changes of the non-selectivity part). Seeds 20000-20039 (440 tasks);
development dry run 352-371.

The doses ``eta = 0`` and ``eta = 0.3`` (nominal) realise the same system
as W_RAM_no_plasticity and PC_nominal of the same seed; they are kept as
their own classes, as planned, and tagged ``same_system_as`` so that the
duplicate detector's same-family pairs are expected.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from impact_pipeline.bench.designs_v2 import (
    CONFIRMATORY,
    DEVELOPMENT,
    Design,
    TaskSpec,
    make_scorings,
)
from impact_pipeline.bench.designs_v2 import family_a as FA

MODULE = "ram_only"
FAMILY = "A"
CATALOGUE_FAMILY = "A"
PRESET = "ram_only_160"
PROTOCOLS = {"R": "A-RAM160"}
PRINCIPLES_SCORED = ("RAM",)
ETA_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.6)
NOMINAL_ETA = 0.3

# (class id, builder, catalogue id or knobs)
CLASSES = (
    ("PC_nominal", "catalogue", "PC_nominal"),
    ("W_RAM_no_plasticity", "catalogue", "W_RAM_no_plasticity"),
    *((f"eta_{level:g}", "agent", {"eta": level}) for level in ETA_LEVELS),
    ("reflex_arc", "catalogue", "adversarial_reflex_arc"),
    ("scrambled_feedback", "catalogue", "adversarial_scrambled_feedback"),
    ("K_1", "agent", {"K": 1}),
    ("g_b_0", "agent", {"g_b": 0.0}),
)
CLASS_IDS = tuple(c[0] for c in CLASSES)
SAME_SYSTEM_AS = {"eta_0": "W_RAM_no_plasticity", "eta_0.3": "PC_nominal"}
# the knob a dose class varies from the nominal agent, recorded as the
# sweep_knob / sweep_level tags the hypotheses select doses by
SWEEP_KNOBS = ("eta", "K", "g_b")

SEEDS = {"arm": {CONFIRMATORY: range(20000, 20040), DEVELOPMENT: range(352, 372)}}


def scorings():
    """One scoring (R, RAM-PE only) with the component status and no
    verdict."""
    return make_scorings(PROTOCOLS, PRINCIPLES_SCORED, verdict=False)


def class_task(design: str, cls: str, seed: int, replicate: int = 0,
               module: str = MODULE) -> TaskSpec:
    """The task of one RAM-only class (also used by the twins and anchors)."""
    for cid, builder, what in CLASSES:
        if cid != cls:
            continue
        tags = {"ram_class": cid}
        if cid in SAME_SYSTEM_AS:
            tags["same_system_as"] = SAME_SYSTEM_AS[cid]
        if builder == "catalogue":
            return FA.catalogue_task(design, FAMILY, CATALOGUE_FAMILY, what, seed,
                                     scorings(), replicate=replicate, preset=PRESET,
                                     tags=tags, module=module, name=cid)
        from impact_pipeline.bench.generators import NOMINAL_KNOBS

        kn = NOMINAL_KNOBS.replace(**what)
        tags["bits"] = list(kn.bits())
        if "eta" in what:
            tags["eta"] = float(what["eta"])
        (knob, level), = what.items()
        if knob in SWEEP_KNOBS:
            tags["sweep_knob"], tags["sweep_level"] = knob, float(level)
        return FA.agent_task(design, FAMILY, CATALOGUE_FAMILY, cid, seed,
                             kn.to_dict(), scorings(), replicate=replicate,
                             preset=PRESET, tags=tags, module=module)
    raise ValueError(f"unknown RAM-only class {cls!r}; one of {CLASS_IDS}")


def arm(split: str, *, seeds: Optional[Sequence[int]] = None,
        systems: Optional[Sequence[str]] = None) -> List[TaskSpec]:
    """``RAM160``: every class (``systems``: a subset) on every seed of the
    arm."""
    out = []
    for cls in FA._select(CLASS_IDS, systems, "RAM-only classes"):
        for seed in FA._seeds(SEEDS["arm"], split, seeds):
            out.append(class_task("RAM160", cls, seed))
    return out


ADEMP = {
    "RAM160": {
        "aims": "specificity and sensitivity of RAM-PE (no RAM-PE without "
                "plasticity; dose-response; predicted non-selectivity against "
                "SNR changes; the contingency construct boundary)",
        "data": "family-A agents at 160 bandit trials: PC_nominal, "
                "W_RAM_no_plasticity, eta in {0, 0.1, 0.2, 0.3, 0.6}, the reflex "
                "arc, scrambled feedback, K = 1 and g_b = 0; 40 seeds",
        "estimands": "RAM-PE c and status per class; paired contrasts with "
                     "PC_nominal on shared seeds",
        "methods": "RAM-PE by the version the A-RAM160 protocol names (ram-v3-"
                   "2026.10), judged by tost-v2 on the A-RAM160 anchor; the "
                   "component status only (no verdict)",
        "performance": "H0 cell rule at eta = 0, Spearman rho(c, eta), paired "
                       "contrast proportions, CP bound of the PC PRESENT rate, "
                       "Hodges-Lehmann and Newcombe intervals: HCv2-16 and -17",
    },
}

DESIGNS = (
    Design("RAM160", FAMILY, "RAM-only arm at 160 trials (RAM-PE only)", arm,
           {CONFIRMATORY: 11 * 40, DEVELOPMENT: 11 * 20}, ademp=ADEMP["RAM160"]),
)

__all__ = [
    "ADEMP",
    "CLASSES",
    "CLASS_IDS",
    "DESIGNS",
    "ETA_LEVELS",
    "PRESET",
    "PROTOCOLS",
    "SAME_SYSTEM_AS",
    "SEEDS",
    "SWEEP_KNOBS",
    "arm",
    "class_task",
    "scorings",
]
