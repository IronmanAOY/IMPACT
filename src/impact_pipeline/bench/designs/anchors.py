"""
Reference blocks and anchor replication blocks of MPC-Bench.

Anchors are computed on development seeds only: the reference block 900-939
holds PC_nominal and the own-lesion witnesses of every family protocol, on
the same seeds, so that the anchor (the positive control's excess over its
null), its validity and its specificity (paired contrast with the own
lesion) can be fixed before a confirmatory run. The same systems are re-run on the
confirmatory replication block 20900-20919 (family A: 20900-20939, below) to
test that every anchor status replicates.

==============  ==============================================  =========  ===========
design          systems                                         dev        confirmatory
==============  ==============================================  =========  ===========
A_anchors       PC_nominal and the six single deficits          900-939    20900-20939
                (A-R and A-H, every reported form)
C_anchors       PC_nominal, W_NAS_no_workspace,                 900-939    20900-20919
                W_NAS_broadcast_only, W_IIM_feedforward (C-R
                and C-H; IIM included: the reference-block
                anchors are the only IIM output on family C on
                development seeds)
RAM160_anchors  PC_nominal and W_RAM_no_plasticity at 160       900-939    20900-20919
                trials (A-RAM160)
==============  ==============================================  =========  ===========

Task counts: 280, 160 and 80 (development); 280, 80 and 40 (confirmatory).
Where the replication power on development data is below 0.9 at 20 seeds, a
design's replication block extends to 20939
(:data:`REPLICATION_SEEDS_EXTENDED`). That value is declared once, in
:data:`CALIBRATION_PENDING`; the builders and the task counts read it, and
the run manifest records it. It extends family A and no other design.

The anchors of the forward views (``forward_anchor_replication``,
:mod:`impact_pipeline.bench.designs.forward`) follow the same rule per
forward arm: the arm's anchor condition serves every view of the arm, so a
view below 0.9 extends the runs of its arm and every view of the arm reads
the extension (as A-R reads the family-A one). It extends the Hopf arm and
neither family-A arm.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from impact_pipeline.bench.designs import (
    CONFIRMATORY,
    DEVELOPMENT,
    PRINCIPLES,
    Design,
    TaskSpec,
    make_scorings,
)
from impact_pipeline.bench import seeds as S
from impact_pipeline.bench.designs import family_a as FA
from impact_pipeline.bench.designs import family_c as FC
from impact_pipeline.bench.designs import ram_only as RO

MODULE = "anchors"
REFERENCE_SEEDS = tuple(S.REFERENCE_BLOCK)
REPLICATION_SEEDS = tuple(range(20900, 20920))
REPLICATION_SEEDS_EXTENDED = tuple(range(20900, 20940))

# Own-lesion witness of each principle.
OWN_LESION = {
    "NAS": "W_NAS_no_workspace",
    "IIM": "W_IIM_feedforward",
    "SRPI": "W_SRPI_no_efference",
    "RAM": "W_RAM_no_plasticity",
    "PDI": "W_PDI_single_attractor",
}
A_SYSTEMS = FA.EXTENDED_SYSTEMS
C_SYSTEMS = FC.EXTENDED_SYSTEMS
RAM_CLASSES = ("PC_nominal", "W_RAM_no_plasticity")
ANCHOR_DESIGNS = ("A_anchors", "C_anchors", "RAM160_anchors")
# The forward arms whose anchor condition runs on the replication block
# (forward.ARMS, declared here so that this module does not import the
# forward layer; a test keeps the two equal).
FORWARD_ARMS = ("hopf", "forward_a_eeg", "forward_a_bold")

# The calibration values of the anchor designs, declared here once: the
# anchor designs and the forward arms whose confirmatory replication block
# extends to 20900-20939. Every run manifest records them.
CALIBRATION_PENDING = {
    "replication_extended": {
        "value": {"A_anchors": True, "C_anchors": False, "RAM160_anchors": False},
        "meaning": "anchor designs whose replication block extends to "
                   "20900-20939 because the replication power at 20 seeds is "
                   "below 0.9: family A, no other design",
    },
    "forward_replication_extended": {
        "value": {arm: arm == "hopf" for arm in FORWARD_ARMS},
        "meaning": "forward arms whose anchor runs on the replication block "
                   "extend to 20900-20939 because the validity-only "
                   "replication power of one of their views at 20 seeds is "
                   "below 0.9: the Hopf arm, neither family-A arm",
    },
}


def is_replication_extended(design: str) -> bool:
    """The declared extension of a design's replication block."""
    return bool(CALIBRATION_PENDING["replication_extended"]["value"][design])


def n_replication_seeds(design: str) -> int:
    return len(REPLICATION_SEEDS_EXTENDED if is_replication_extended(design)
               else REPLICATION_SEEDS)


def is_forward_replication_extended(arm: str) -> bool:
    """The declared extension of a forward arm's replication block."""
    return bool(CALIBRATION_PENDING["forward_replication_extended"]["value"][arm])


def forward_replication_seeds(arm: str) -> tuple:
    """The replication seeds of a forward arm's anchor condition."""
    return (REPLICATION_SEEDS_EXTENDED if is_forward_replication_extended(arm)
            else REPLICATION_SEEDS)


def anchor_seeds(split: str, seeds=None, replication_extended: bool = False) -> tuple:
    if seeds is not None:
        out = tuple(int(s) for s in seeds)
        S.check_seeds(out, split)
        return out
    if split == DEVELOPMENT:
        return REFERENCE_SEEDS
    return REPLICATION_SEEDS_EXTENDED if replication_extended else REPLICATION_SEEDS


def _extended(design: str, value: Optional[bool]) -> bool:
    return is_replication_extended(design) if value is None else bool(value)


def a_anchors(split: str, *, seeds: Optional[Sequence[int]] = None,
              systems: Optional[Sequence[str]] = None,
              replication_extended: Optional[bool] = None) -> List[TaskSpec]:
    """``A_anchors``: PC_nominal and the six single deficits under A-R and A-H
    with every form that carries its own anchor."""
    sc = make_scorings(FA.PROTOCOLS, PRINCIPLES, FA.FORMS["witnesses"],
                       form_declarations=FA.FORM_DECLARATIONS)
    block = anchor_seeds(split, seeds, _extended("A_anchors", replication_extended))
    return [
        FA.catalogue_task("A_anchors", FA.FAMILY, FA.CATALOGUE_FAMILY, sid, seed, sc,
                          module=MODULE, tags={"anchor_role": _role(sid)})
        for sid in FA._select(A_SYSTEMS, systems, "anchor systems")
        for seed in block
    ]


def c_anchors(split: str, *, seeds: Optional[Sequence[int]] = None,
              systems: Optional[Sequence[str]] = None,
              replication_extended: Optional[bool] = None) -> List[TaskSpec]:
    """``C_anchors`` (every principle, IIM included)."""
    sc = make_scorings(FC.PROTOCOLS, PRINCIPLES, FC.FORMS["witnesses"],
                       form_declarations=FA.FORM_DECLARATIONS)
    block = anchor_seeds(split, seeds, _extended("C_anchors", replication_extended))
    return [
        FA.catalogue_task("C_anchors", FC.FAMILY, FC.CATALOGUE_FAMILY, sid, seed, sc,
                          module=MODULE, tags={"anchor_role": _role(sid)})
        for sid in FA._select(C_SYSTEMS, systems, "anchor systems")
        for seed in block
    ]


def ram_anchors(split: str, *, seeds: Optional[Sequence[int]] = None,
                systems: Optional[Sequence[str]] = None,
                replication_extended: Optional[bool] = None) -> List[TaskSpec]:
    """``RAM160_anchors``: PC_nominal and W_RAM_no_plasticity at 160 trials."""
    block = anchor_seeds(split, seeds,
                         _extended("RAM160_anchors", replication_extended))
    return [
        RO.class_task("RAM160_anchors", cls, seed, module=MODULE)
        for cls in FA._select(RAM_CLASSES, systems, "anchor classes")
        for seed in block
    ]


def _role(system_id: str) -> str:
    if system_id == "PC_nominal":
        return "positive_control"
    own = [p for p, w in OWN_LESION.items() if w == system_id]
    return f"own_lesion:{own[0]}" if own else "single_deficit"


_ANCHOR_AIMS = (
    "the anchors of the family protocols: on the development reference block "
    "the anchor (the positive control's excess over its null), its validity and "
    "its specificity against the own lesion fix N_anch and testability before "
    "a confirmatory run; on the confirmatory replication block each anchor "
    "status must replicate")
_ANCHOR_PERFORMANCE = (
    "validity: one-sided 95 % t lower bound of the mean excess > 0; specificity: "
    "mean paired contrast >= 0.5 x anchor with lower bound > 0; replication of "
    "every development status, with replication power >= 0.9 checked before a "
    "confirmatory run")

ADEMP = {
    "A_anchors": {
        "aims": _ANCHOR_AIMS + " (A-R and A-H, every form with its own anchor)",
        "data": "PC_nominal and the six single deficits of family A on the "
                "same seeds: 900-939 (development), 20900-20939 (confirmatory; "
                "extended for replication power)",
        "estimands": "per protocol, form and principle: the mean excess of "
                     "PC_nominal over its null and the paired contrast with the "
                     "own lesion",
        "methods": (FA.JOINT_BENCH_METHODS + "; reported forms: NAS secondary, "
                    "IIM bidirectional, NAS at tau_c 0.05 s and 0.2 s"),
        "performance": _ANCHOR_PERFORMANCE,
    },
    "C_anchors": {
        "aims": (_ANCHOR_AIMS + " (C-R and C-H; the only IIM output on family "
                 "C on development seeds)"),
        "data": "PC_nominal, W_NAS_no_workspace, W_NAS_broadcast_only and "
                "W_IIM_feedforward of family C on the same seeds: 900-939, "
                "20900-20919",
        "estimands": "per protocol, form and principle: the mean excess of "
                     "PC_nominal over its null and the paired contrast with the "
                     "own lesion",
        "methods": (FC.C_METHODS + " (except these anchors); reported forms: NAS "
                    "secondary, IIM bidirectional, NAS at tau_c 0.05 s and 0.2 s"),
        "performance": _ANCHOR_PERFORMANCE,
    },
    "RAM160_anchors": {
        "aims": _ANCHOR_AIMS + " (A-RAM160)",
        "data": "PC_nominal and W_RAM_no_plasticity at 160 trials on the same "
                "seeds: 900-939, 20900-20919",
        "estimands": "the mean RAM-PE excess of PC_nominal over its null and "
                     "the paired contrast with W_RAM_no_plasticity",
        "methods": "RAM-PE (ram-pe-2026.10) judged by tost; component "
                   "statuses only",
        "performance": _ANCHOR_PERFORMANCE,
    },
}

DESIGNS = (
    Design("A_anchors", FA.FAMILY,
           "family-A reference block (development) and anchor replication block "
           "(confirmatory)", a_anchors,
           {DEVELOPMENT: 7 * 40, CONFIRMATORY: 7 * n_replication_seeds("A_anchors")},
           ademp=ADEMP["A_anchors"]),
    Design("C_anchors", FC.FAMILY,
           "family-C reference block and anchor replication block", c_anchors,
           {DEVELOPMENT: 4 * 40, CONFIRMATORY: 4 * n_replication_seeds("C_anchors")},
           ademp=ADEMP["C_anchors"]),
    Design("RAM160_anchors", RO.FAMILY,
           "RAM-only reference block and anchor replication block", ram_anchors,
           {DEVELOPMENT: 2 * 40,
            CONFIRMATORY: 2 * n_replication_seeds("RAM160_anchors")},
           ademp=ADEMP["RAM160_anchors"]),
)

__all__ = [
    "ADEMP",
    "ANCHOR_DESIGNS",
    "A_SYSTEMS",
    "CALIBRATION_PENDING",
    "C_SYSTEMS",
    "DESIGNS",
    "FORWARD_ARMS",
    "OWN_LESION",
    "RAM_CLASSES",
    "REFERENCE_SEEDS",
    "REPLICATION_SEEDS",
    "REPLICATION_SEEDS_EXTENDED",
    "a_anchors",
    "anchor_seeds",
    "c_anchors",
    "forward_replication_seeds",
    "is_forward_replication_extended",
    "is_replication_extended",
    "n_replication_seeds",
    "ram_anchors",
]
