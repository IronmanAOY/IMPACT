"""
Family-C1 designs of MPC-Bench v2 (design 3.1, 3.2 and 3.7).

C1 is the v1 family C (Stuart-Landau units recorded as Re z at 20 Hz),
unchanged, scored under R and H from the same simulation (protocols
``C1-R`` and ``C1-H``). NAS and IIM are scored; SRPI, RAM-PE and PDI are
declared ``NOT_APPLICABLE_OBSERVATION_MODEL`` on the carrier recording
(UNDEFINED, never ABSENT) and are kept in the primary scoring so that every
record says so.

==============  ===============================================  ===========  =======
design          systems                                          confirm.     dev
==============  ===============================================  ===========  =======
C1_witnesses    the 13 family-C witnesses of the catalogue (the  20000-20019  372-383
                duplicated family-A nulls replaced by            (4 to 20044)
                N_uncoupled, plus N_modules_disconnected);
                PC_nominal, W_NAS_no_workspace,
                W_NAS_broadcast_only and W_IIM_feedforward on 25
                more seeds
C1_sweeps       g_b and c_int (10 levels each)                   20000-20009  372-375
C1_factorial    the 2^5 cells                                    20000-20009  376-379
==============  ===============================================  ===========  =======

Task counts (confirmatory): 13 x 20 + 4 x 25 = 360 (the design table says
"about 14 x 20"; the catalogue defines 13 family-C witnesses), 200 and 320.

IIM v5 on C1 is held out until the freeze: on the development split the
C1 scorings leave IIM out (the reference-block anchors of the ``anchors``
module are the only IIM output on C1 before the freeze).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from impact_pipeline.bench.designs_v2 import (
    CONFIRMATORY,
    DEVELOPMENT,
    PRINCIPLES,
    Design,
    TaskSpec,
    make_scorings,
)
from impact_pipeline.bench.designs_v2 import family_a as FA

MODULE = "family_c1"
FAMILY = "C1"
CATALOGUE_FAMILY = "C"
PROTOCOLS = {"R": "C1-R", "H": "C1-H"}
EXTENDED_SYSTEMS = ("PC_nominal", "W_NAS_no_workspace", "W_NAS_broadcast_only",
                    "W_IIM_feedforward")
# Principles declared not applicable on the C1 carrier recording.
NOT_APPLICABLE = ("RAM", "PDI", "SRPI")
# IIM on C1 is held out: no output before the freeze outside the anchors.
HELD_OUT_PRINCIPLES = ("IIM",)
SWEEP_KNOBS = ("g_b", "c_int")

SEEDS = {
    "witnesses": {CONFIRMATORY: range(20000, 20020), DEVELOPMENT: range(372, 384)},
    "witnesses_extended": {CONFIRMATORY: range(20020, 20045)},
    "sweeps": {CONFIRMATORY: range(20000, 20010), DEVELOPMENT: range(372, 376)},
    "factorial": {CONFIRMATORY: range(20000, 20010), DEVELOPMENT: range(376, 380)},
}
FORMS = FA.FORMS


def principles(split: str) -> tuple:
    """The principles scored on C1 in a split (IIM only on the confirmatory
    split)."""
    if split == CONFIRMATORY:
        return PRINCIPLES
    return tuple(p for p in PRINCIPLES if p not in HELD_OUT_PRINCIPLES)


def _scorings(split, forms, modes=("system",)):
    return make_scorings(PROTOCOLS, principles(split), forms, bearer_modes=modes,
                         form_declarations=FA.FORM_DECLARATIONS)


def witnesses(split: str, *, seeds: Optional[Sequence[int]] = None,
              systems: Optional[Sequence[str]] = None,
              forms: Optional[Sequence[str]] = None) -> List[TaskSpec]:
    """``C1_witnesses``."""
    ids = FA.first(FA._select(FA.catalogue_systems("witness", CATALOGUE_FAMILY),
                              systems, "witnesses"), EXTENDED_SYSTEMS)
    forms = FORMS["witnesses"] if forms is None else tuple(forms)
    base = FA._seeds(SEEDS["witnesses"], split, seeds)
    ext = () if seeds is not None else FA.seeds_of(SEEDS["witnesses_extended"], split)
    out = []
    for sid in ids:
        modes = FA.bearer_modes(sid)
        for seed in base + (ext if sid in EXTENDED_SYSTEMS else ()):
            out.append(FA.catalogue_task("C1_witnesses", FAMILY, CATALOGUE_FAMILY, sid,
                                         seed, _scorings(split, forms, modes),
                                         module=MODULE))
    return out


def sweeps(split: str, *, seeds: Optional[Sequence[int]] = None,
           knobs: Optional[Sequence[str]] = None,
           forms: Optional[Sequence[str]] = None) -> List[TaskSpec]:
    """``C1_sweeps``: g_b and c_int at the v1 levels."""
    from impact_pipeline.bench.generators import NOMINAL_KNOBS

    forms = FORMS["default"] if forms is None else tuple(forms)
    out = []
    for knob in FA._select(SWEEP_KNOBS, knobs, "sweep knobs"):
        for li, level in enumerate(FA.sweep_levels(knob)):
            kn = NOMINAL_KNOBS.replace(**{knob: level})
            tags = {"sweep_knob": knob, "sweep_level": float(level),
                    "bits": list(kn.bits())}
            for seed in FA._seeds(SEEDS["sweeps"], split, seeds):
                out.append(FA.agent_task("C1_sweeps", FAMILY, CATALOGUE_FAMILY,
                                         f"sweep_{knob}_l{li:02d}", seed,
                                         kn.to_dict(), _scorings(split, forms),
                                         tags=tags, module=MODULE))
    return out


def factorial(split: str, *, seeds: Optional[Sequence[int]] = None,
              cells: Optional[Sequence[str]] = None,
              forms: Optional[Sequence[str]] = None) -> List[TaskSpec]:
    """``C1_factorial``: the 2^5 cells."""
    from impact_pipeline.bench.factorial import factorial_cells

    forms = FORMS["default"] if forms is None else tuple(forms)
    all_cells = factorial_cells()
    wanted = FA._select([c["cell_id"] for c in all_cells], cells, "factorial cells")
    out = []
    for cell in all_cells:
        if cell["cell_id"] not in wanted:
            continue
        for seed in FA._seeds(SEEDS["factorial"], split, seeds):
            out.append(FA.agent_task("C1_factorial", FAMILY, CATALOGUE_FAMILY,
                                     cell["cell_id"], seed, cell["knobs"],
                                     _scorings(split, forms),
                                     tags={"bits": list(cell["bits"])},
                                     module=MODULE))
    return out


# The methods of every C1 scoring (also used by the C1 twins and anchors).
C1_METHODS = (FA.JOINT_BENCH_METHODS + "; SRPI, RAM-PE and PDI declared "
              "NOT_APPLICABLE_OBSERVATION_MODEL on the carrier recording "
              "(UNDEFINED, never ABSENT); IIM held out on C1 before the freeze")

ADEMP = {
    "C1_witnesses": {
        "aims": "NAS and IIM identification on a second substrate (oscillator "
                "carriers), and definedness of the principles declared not "
                "applicable there",
        "data": "family C1 (the v1 family C, Stuart-Landau units recorded as "
                "Re z at 20 Hz): the 13 catalogue witnesses on 20 seeds, "
                "PC_nominal, W_NAS_no_workspace, W_NAS_broadcast_only and "
                "W_IIM_feedforward on 45; R and H from one simulation",
        "estimands": "per system, protocol and principle the construct value c, "
                     "the status and its reason; per system and protocol the "
                     "verdict; paired contrasts with PC_nominal",
        "methods": (C1_METHODS + "; reported forms: NAS secondary, IIM "
                    "bidirectional, NAS at tau_c 0.05 s and 0.2 s"),
        "performance": "status rates with seed-clustered bounds, paired-contrast "
                       "proportions, |c_H - c_R| agreement and verdict "
                       "specificity: HCv2-1, -5, -7, -8(d), -14 (C1 parts), -22 "
                       "and -23; no ABSENT or PRESENT on a NOT_APPLICABLE "
                       "component (integrity audit)",
    },
    "C1_sweeps": {
        "aims": "dose-response of NAS and IIM on family C1",
        "data": "family-C1 agents with g_b or c_int at 10 levels and the other "
                "knobs nominal; 10 seeds; R and H",
        "estimands": "c and status of NAS and IIM per dose level",
        "methods": (C1_METHODS + "; reported forms: NAS secondary, IIM "
                    "bidirectional"),
        "performance": "Spearman and OLS trends of c on the dose and ABSENT among "
                       "mechanism-on levels: HCv2-5, -7(iv) and -14(f)",
    },
    "C1_factorial": {
        "aims": "no PRESENT without the mechanism on family C1",
        "data": "the 2^5 family-C1 cells; 10 seeds; R and H",
        "estimands": "status of NAS and IIM per cell, own mechanism on or off",
        "methods": (C1_METHODS + "; reported forms: NAS secondary, IIM "
                    "bidirectional"),
        "performance": "P(PRESENT | own mechanism off) per off-cell (H0 cell "
                       "rule) and ABSENT among on-cells: HCv2-5 and -24",
    },
}

DESIGNS = (
    Design("C1_witnesses", FAMILY,
           "family-C1 witnesses under R and H (45 seed clusters for PC_nominal, "
           "W_NAS_no_workspace, W_NAS_broadcast_only and W_IIM_feedforward)",
           witnesses, {CONFIRMATORY: 13 * 20 + 4 * 25, DEVELOPMENT: 13 * 12},
           ademp=ADEMP["C1_witnesses"]),
    Design("C1_sweeps", FAMILY, "family-C1 dose sweeps of g_b and c_int", sweeps,
           {CONFIRMATORY: 20 * 10, DEVELOPMENT: 20 * 4}, ademp=ADEMP["C1_sweeps"]),
    Design("C1_factorial", FAMILY, "family-C1 2^5 factorial", factorial,
           {CONFIRMATORY: 32 * 10, DEVELOPMENT: 32 * 4},
           ademp=ADEMP["C1_factorial"]),
)

__all__ = [
    "ADEMP",
    "C1_METHODS",
    "CATALOGUE_FAMILY",
    "DESIGNS",
    "EXTENDED_SYSTEMS",
    "FAMILY",
    "HELD_OUT_PRINCIPLES",
    "NOT_APPLICABLE",
    "PROTOCOLS",
    "SEEDS",
    "factorial",
    "principles",
    "sweeps",
    "witnesses",
]
