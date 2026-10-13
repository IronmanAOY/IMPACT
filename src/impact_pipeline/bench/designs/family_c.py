"""
Family-C designs of MPC-Bench.

Family C (Stuart-Landau units recorded as Re z at 20 Hz) is scored under R
and H from the same simulation (protocols ``C-R`` and ``C-H``). NAS and IIM
are scored; SRPI, RAM-PE and PDI are declared
``NOT_APPLICABLE_OBSERVATION_MODEL`` on the carrier recording (UNDEFINED,
never ABSENT) and are kept in the primary scoring so that every record says
so.

==============  ===============================================  ===========  =======
design          systems                                          confirm.     dev
==============  ===============================================  ===========  =======
C_witnesses     the 13 family-C witnesses of the catalogue (the  20000-20019  372-383
                duplicated family-A nulls replaced by            (4 to 20044,
                N_uncoupled, plus N_modules_disconnected);       3 of them
                PC_nominal, W_NAS_no_workspace,                  to 20128)
                W_NAS_broadcast_only and W_IIM_feedforward on 25
                more seeds, the three single deficits on 84
                more
C_sweeps        g_b and c_int (10 levels each)                   20000-20009  372-375
                                                                 (c_int to
                                                                 20064)
C_factorial     the 2^5 cells                                    20000-20009  376-379
==============  ===============================================  ===========  =======

Task counts (confirmatory): 13 x 20 + 4 x 25 + 3 x 84 = 612 (the catalogue
defines 13 family-C witnesses), 20 x 10 + 10 x 55 = 750 and 320.

Seed counts. The single deficits of families A and C on shared seeds form
one seed cluster per seed, so the three family-C single deficits
W_NAS_no_workspace, W_NAS_broadcast_only and W_IIM_feedforward run on 129
confirmatory seeds (20000-20128), enough seed clusters for a pooled
Clopper-Pearson bound on the verdict specificity; family A keeps 45. As in
family A, the c_int sweep runs its 10 levels on 65 confirmatory seeds each
(20000-20064; g_b stays on 10).

Macro IIM on family C is held out: on the development split the family-C
scorings leave IIM out (the reference-block anchors of the ``anchors``
module are the only IIM output on family C on development seeds).
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
from impact_pipeline.bench.designs import family_a as FA

MODULE = "family_c"
FAMILY = "C"
CATALOGUE_FAMILY = "C"
PROTOCOLS = {"R": "C-R", "H": "C-H"}
EXTENDED_SYSTEMS = ("PC_nominal", "W_NAS_no_workspace", "W_NAS_broadcast_only",
                    "W_IIM_feedforward")
# The family-C single deficits on 129 seed clusters each (module
# docstring).
SPECIFICITY_SYSTEMS = ("W_NAS_no_workspace", "W_NAS_broadcast_only",
                       "W_IIM_feedforward")
# Principles declared not applicable on the family-C carrier recording.
NOT_APPLICABLE = ("RAM", "PDI", "SRPI")
# IIM on family C is held out: no development output outside the anchors.
HELD_OUT_PRINCIPLES = ("IIM",)
SWEEP_KNOBS = ("g_b", "c_int")

SEEDS = {
    "witnesses": {CONFIRMATORY: range(20000, 20020), DEVELOPMENT: range(372, 384)},
    "witnesses_extended": {CONFIRMATORY: range(20020, 20045)},
    "specificity_extended": {CONFIRMATORY: range(20045, 20129)},
    "sweeps": {CONFIRMATORY: range(20000, 20010), DEVELOPMENT: range(372, 376)},
    "sweeps_extended": {CONFIRMATORY: range(20010, 20065)},
    "factorial": {CONFIRMATORY: range(20000, 20010), DEVELOPMENT: range(376, 380)},
}
FORMS = FA.FORMS


def principles(split: str) -> tuple:
    """The principles scored on family C in a split (IIM only on the
    confirmatory split)."""
    if split == CONFIRMATORY:
        return PRINCIPLES
    return tuple(p for p in PRINCIPLES if p not in HELD_OUT_PRINCIPLES)


def _scorings(split, forms, modes=("system",)):
    return make_scorings(PROTOCOLS, principles(split), forms, bearer_modes=modes,
                         form_declarations=FA.FORM_DECLARATIONS)


def witnesses(split: str, *, seeds: Optional[Sequence[int]] = None,
              systems: Optional[Sequence[str]] = None,
              forms: Optional[Sequence[str]] = None) -> List[TaskSpec]:
    """``C_witnesses``: every family-C witness on the witness seeds,
    PC_nominal and the three single deficits on 25 more confirmatory seeds,
    and the three single deficits on 84 more."""
    ids = FA.first(FA._select(FA.catalogue_systems("witness", CATALOGUE_FAMILY),
                              systems, "witnesses"), EXTENDED_SYSTEMS)
    forms = FORMS["witnesses"] if forms is None else tuple(forms)
    base = FA._seeds(SEEDS["witnesses"], split, seeds)
    ext = () if seeds is not None else FA.seeds_of(SEEDS["witnesses_extended"], split)
    spec_ext = (() if seeds is not None
                else FA.seeds_of(SEEDS["specificity_extended"], split))
    out = []
    for sid in ids:
        modes = FA.bearer_modes(sid)
        for seed in (base + (ext if sid in EXTENDED_SYSTEMS else ())
                     + (spec_ext if sid in SPECIFICITY_SYSTEMS else ())):
            out.append(FA.catalogue_task("C_witnesses", FAMILY, CATALOGUE_FAMILY, sid,
                                         seed, _scorings(split, forms, modes),
                                         module=MODULE))
    return out


def sweeps(split: str, *, seeds: Optional[Sequence[int]] = None,
           knobs: Optional[Sequence[str]] = None,
           forms: Optional[Sequence[str]] = None) -> List[TaskSpec]:
    """``C_sweeps``: g_b and c_int at the family-A sweep levels, the c_int
    levels also on the extended sweep seeds (as in family A)."""
    forms = FORMS["default"] if forms is None else tuple(forms)
    return FA._agent_sweeps("C_sweeps", FAMILY, CATALOGUE_FAMILY, SWEEP_KNOBS,
                            SEEDS, split, lambda: _scorings(split, forms),
                            seeds=seeds, knobs=knobs, module=MODULE)


def factorial(split: str, *, seeds: Optional[Sequence[int]] = None,
              cells: Optional[Sequence[str]] = None,
              forms: Optional[Sequence[str]] = None) -> List[TaskSpec]:
    """``C_factorial``: the 2^5 cells."""
    forms = FORMS["default"] if forms is None else tuple(forms)
    return FA._agent_factorial("C_factorial", FAMILY, CATALOGUE_FAMILY, SEEDS,
                               split, lambda: _scorings(split, forms),
                               seeds=seeds, cells=cells, module=MODULE)


# The methods of every family-C scoring (also used by its twins and anchors).
C_METHODS = (FA.JOINT_BENCH_METHODS + "; SRPI, RAM-PE and PDI declared "
             "NOT_APPLICABLE_OBSERVATION_MODEL on the carrier recording "
             "(UNDEFINED, never ABSENT); IIM held out on family C")

ADEMP = {
    "C_witnesses": {
        "aims": "NAS and IIM identification on a second substrate (oscillator "
                "carriers), and definedness of the principles declared not "
                "applicable there",
        "data": "family C (Stuart-Landau units recorded as Re z at 20 Hz): "
                "the 13 catalogue witnesses on 20 seeds, PC_nominal on 45 and "
                "W_NAS_no_workspace, W_NAS_broadcast_only and W_IIM_feedforward "
                "on 129; R and H from one simulation",
        "estimands": "per system, protocol and principle the construct value c, "
                     "the status and its reason; per system and protocol the "
                     "verdict; paired contrasts with PC_nominal",
        "methods": (C_METHODS + "; reported forms: NAS secondary, IIM "
                    "bidirectional, NAS at tau_c 0.05 s and 0.2 s"),
        "performance": "status rates with seed-clustered bounds, paired-contrast "
                       "proportions, |c_H - c_R| agreement and verdict "
                       "specificity; no ABSENT or PRESENT on a NOT_APPLICABLE "
                       "component",
    },
    "C_sweeps": {
        "aims": "dose-response of NAS and IIM on family C",
        "data": "family-C agents with g_b or c_int at 10 levels and the other "
                "knobs nominal; 10 seeds (c_int 65); R and H",
        "estimands": "c and status of NAS and IIM per dose level",
        "methods": (C_METHODS + "; reported forms: NAS secondary, IIM "
                    "bidirectional"),
        "performance": "Spearman and OLS trends of c on the dose and ABSENT among "
                       "mechanism-on levels",
    },
    "C_factorial": {
        "aims": "no PRESENT without the mechanism on family C",
        "data": "the 2^5 family-C cells; 10 seeds; R and H",
        "estimands": "status of NAS and IIM per cell, own mechanism on or off",
        "methods": (C_METHODS + "; reported forms: NAS secondary, IIM "
                    "bidirectional"),
        "performance": "P(PRESENT | own mechanism off) per off-cell (H0 cell "
                       "rule) and ABSENT among on-cells",
    },
}

DESIGNS = (
    Design("C_witnesses", FAMILY,
           "family-C witnesses under R and H (45 seed clusters for PC_nominal, "
           "129 for W_NAS_no_workspace, W_NAS_broadcast_only and "
           "W_IIM_feedforward)",
           witnesses, {CONFIRMATORY: 13 * 20 + 4 * 25 + 3 * 84,
                       DEVELOPMENT: 13 * 12},
           ademp=ADEMP["C_witnesses"]),
    Design("C_sweeps", FAMILY,
           "family-C dose sweeps of g_b and c_int (c_int on 65 seeds)", sweeps,
           {CONFIRMATORY: 20 * 10 + 10 * 55, DEVELOPMENT: 20 * 4},
           ademp=ADEMP["C_sweeps"]),
    Design("C_factorial", FAMILY, "family-C 2^5 factorial", factorial,
           {CONFIRMATORY: 32 * 10, DEVELOPMENT: 32 * 4},
           ademp=ADEMP["C_factorial"]),
)

__all__ = [
    "ADEMP",
    "CATALOGUE_FAMILY",
    "C_METHODS",
    "DESIGNS",
    "EXTENDED_SYSTEMS",
    "FAMILY",
    "HELD_OUT_PRINCIPLES",
    "NOT_APPLICABLE",
    "PROTOCOLS",
    "SEEDS",
    "SPECIFICITY_SYSTEMS",
    "factorial",
    "principles",
    "sweeps",
    "witnesses",
]
