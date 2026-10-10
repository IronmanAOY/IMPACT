"""
Family-A designs of MPC-Bench v2 (preregistration v2 sections 4.1 and 4.2).

Family A is the v1 routing with the recording device: the v1 dynamics,
scored under the complete declaration R (task events, context cues, slow
phase, recorded drivers) and the task-only declaration H from the **same**
simulation (one task, several scorings). Protocols: ``A-R`` and ``A-H``;
the held-out declarations P, Q10, Q25 and J are judged by ``A-P``,
``A-Q10``, ``A-Q25`` and ``A-J`` (the declaration-specific copies of
``A-R``).

==================  ==========================================  =============  ======
design              systems                                     confirmatory   dev
==================  ==========================================  =============  ======
A_witnesses         the 16 family-A witnesses of the catalogue  20000-20019;   320-331
                    (PW_patchwork in the principle-bearer        PC_nominal and
                    mode); PC_nominal and the six v1             the six
                    single deficits on 25 more seeds; the two    deficits to
                    null witnesses on 26 more                    20044; the
                                                                 nulls to
                                                                 20045
A_sweeps            g_b (10 levels 0-2), c_int (10 levels       20000-20009;   332-335
                    0-1.2), K {2, 3, 4, 6, 8, 12}               c_int to
                                                                 20064
A_factorial         the 2^5 mechanism cells                     20000-20009    336-339
A_adversaries       adversarial_common_driver, reflex_arc,      20000-20019    340-351
                    random_label_self_other, scrambled_feedback,
                    ADV_NAS_staggered_driver (three variants)
A_heldout           ADV_NAS_staggered_tau10 and _sat (NAS       20000-20019    smoke
                    under R)                                                   980-984
==================  ==========================================  =============  ======

Task counts (confirmatory): 16 x 20 + 7 x 25 + 2 x 26 = 547, 26 x 10 +
10 x 55 = 810, 32 x 10 = 320, 7 x 20 = 140 and 2 x 20 = 40.

Null witnesses. N_independent_noise and N_ar1 run on 46 confirmatory seeds
(20000-20045): with the RAM-PE rows of the null-calibration generator left
out of HCv2-1 (UNDEFINED by construction), the two nulls hold the RAM-PE
rows of HCv2-1, and 2 x 46 = 92 seed clusters is the smallest size whose
pooled Clopper-Pearson bound (at 0.05 / 5) stays below 0.07 with one
PRESENT event (CD-11: seeds are resized, never thresholds).

The c_int sweep. Its 10 levels run on 65 confirmatory seeds each
(20000-20064; g_b and K stay on 10) for HCv2-14(f), the Spearman trend
of IIM-dir on c_int under R. The levels share their seeds (and the null
seed derived from each), and the development values carry a seed effect,
so the operating characteristic resamples whole seed profiles: on the
development rates P(SUPPORTED) is 0.16 at 10 seeds per level and 0.82 at
65 (CD-11).

Held-out conditions. On the confirmatory witness tasks of seeds
20000-20019, PC_nominal, W_NAS_no_workspace, N_modules_disconnected,
W_IIM_feedforward and O_inert are also scored (NAS and IIM) under P, Q10,
Q25 and J: 5 x 20 x 4 = 400 re-scorings of the witness simulations. On
W_PDI_single_attractor (20000-20044) the PDI content bearer is also scored
with module S declared as the access node. On the development split the
held-out conditions exist only as the smoke tasks of ``A_heldout`` (seeds
980-984, outputs discarded unread by the runner).

Patchwork. PW_patchwork is recorded in the principle-bearer mode (every
principle on its own module, IIM on the IIM module's three sub-groups), the
data of the Tier-B single-source test. Its system-bearer grain, one macro
node per module, gives IIM five macro nodes, about thirty times the cost of
a four-node run for every statistic of the null and the bootstrap, and no
Tier-A hypothesis reads it.

Forms. The primary scoring of every task of the joint bench (witnesses,
sweeps, factorial, adversaries) carries all five principles and the
verdict; these tasks also report the NAS secondary representation and the
bidirectional IIM cut, and the witness tasks the NAS sensitivity at
``tau_c`` 0.05 s and 0.2 s under R (:data:`FORMS`,
:data:`FORM_DECLARATIONS`). The held-out staggered drivers are scored for
NAS alone and the held-out re-scorings for NAS and IIM, with component
statuses only (no verdict). PC_nominal and the six single deficits are
built first (the most decision-relevant runs first), and the PDI
partitions of W_PDI_single_attractor are kept for the comparison with the
oracle ignition gate.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from impact_pipeline.bench import adversarial_v2 as A2
from impact_pipeline.bench.designs_v2 import (
    CONFIRMATORY,
    DEVELOPMENT,
    PRINCIPLES,
    Design,
    DesignError,
    TaskSpec,
    make_scorings,
    seeds_of,
    task_id,
)
from impact_pipeline.v2 import seeds as S

MODULE = "family_a"
FAMILY = "A"
CATALOGUE_FAMILY = "A"
PROTOCOLS = {"R": "A-R", "H": "A-H"}
HELD_OUT_PROTOCOLS = {"P": "A-P", "Q10": "A-Q10", "Q25": "A-Q25", "J": "A-J"}

# PC_nominal and the six v1 single deficits: 45 seed clusters.
EXTENDED_SYSTEMS = (
    "PC_nominal",
    "W_RAM_no_plasticity",
    "W_PDI_single_attractor",
    "W_NAS_no_workspace",
    "W_NAS_broadcast_only",
    "W_IIM_feedforward",
    "W_SRPI_no_efference",
)
# The null witnesses of HCv2-1: 46 seed clusters each (module docstring).
NULL_WITNESSES = ("N_independent_noise", "N_ar1")
# Witness simulations re-scored under the held-out declarations.
HELD_OUT_RESCORED = ("PC_nominal", "W_NAS_no_workspace", "N_modules_disconnected",
                     "W_IIM_feedforward", "O_inert")
HELD_OUT_RESCORED_PRINCIPLES = ("NAS", "IIM")
MISDECLARED_ACCESS_SYSTEM = "W_PDI_single_attractor"
HELD_OUT_SYSTEM_PRINCIPLES = ("NAS",)
# systems recorded in the principle-bearer mode only (module docstring)
PRINCIPLE_BEARER_SYSTEMS = ("PW_patchwork",)


def bearer_modes(system_id: str) -> tuple:
    """The bearer modes a witness is scored in: the principle-bearer mode
    for the patchwork, the system bearer for every other system."""
    return ("principle",) if system_id in PRINCIPLE_BEARER_SYSTEMS else ("system",)


SEEDS = {
    "witnesses": {CONFIRMATORY: range(20000, 20020), DEVELOPMENT: range(320, 332)},
    "witnesses_extended": {CONFIRMATORY: range(20020, 20045)},
    "null_witnesses_extended": {CONFIRMATORY: range(20020, 20046)},
    "held_out_rescorings": {CONFIRMATORY: range(20000, 20020)},
    "misdeclared_access": {CONFIRMATORY: range(20000, 20045)},
    "sweeps": {CONFIRMATORY: range(20000, 20010), DEVELOPMENT: range(332, 336)},
    "sweeps_extended": {CONFIRMATORY: range(20010, 20065)},
    "factorial": {CONFIRMATORY: range(20000, 20010), DEVELOPMENT: range(336, 340)},
    "adversaries": {CONFIRMATORY: range(20000, 20020), DEVELOPMENT: range(340, 352)},
    "held_out": {CONFIRMATORY: range(20000, 20020), DEVELOPMENT: S.SMOKE_SEEDS},
}

# Dose levels of the sweeps: g_b and c_int at the v1 levels (10 each, the
# first switching the mechanism off), K at {2, 3, 4, 6, 8, 12}.
SWEEP_KNOBS = ("g_b", "c_int", "K")
K_LEVELS = (2, 3, 4, 6, 8, 12)
# Knobs whose levels also run on the extended sweep seeds (module docstring).
EXTENDED_SWEEP_KNOBS = ("c_int",)

FORMS = {
    "witnesses": ("nas_secondary", "iim_bidirectional", "nas_tau_0.05", "nas_tau_0.2"),
    "default": ("nas_secondary", "iim_bidirectional"),
}
# The tau_c sensitivity of NAS is reported under the complete declaration.
FORM_DECLARATIONS = {"nas_tau_0.05": ("R",), "nas_tau_0.2": ("R",)}
# Witnesses whose PDI partitions (and the oracle labels per window) are kept
# in the record for the comparison with the oracle ignition gate.
PDI_PARTITION_SYSTEMS = ("W_PDI_single_attractor",)


def sweep_levels(knob: str) -> tuple:
    """The dose levels of a family-A sweep."""
    from impact_pipeline.bench.sweeps import sweep_levels as v1_levels

    if knob == "K":
        return K_LEVELS
    if knob in ("g_b", "c_int"):
        return tuple(v1_levels(knob, 10))
    raise DesignError(f"family-A sweeps vary {SWEEP_KNOBS}, not {knob!r}")


# --------------------------------------------------------------------------
# shared helpers (also used by the other family-A based designs)
# --------------------------------------------------------------------------
def first(values, leading):
    """``values`` with the members of ``leading`` first (in that order): the
    most decision-relevant systems run first."""
    return (tuple(v for v in leading if v in values)
            + tuple(v for v in values if v not in leading))


def _select(values, wanted, what):
    if wanted is None:
        return tuple(values)
    wanted = tuple(wanted)
    unknown = sorted(set(wanted) - set(values))
    if unknown:
        raise DesignError(f"unknown {what} {unknown}")
    return tuple(v for v in values if v in wanted)


def _seeds(table, split, seeds):
    base = seeds_of(table, split)
    if seeds is None:
        return base
    out = tuple(int(s) for s in seeds)
    S.check_seeds(out, split)
    return out


def sweep_seeds(seed_table, knob, split, seeds=None) -> tuple:
    """The seeds of one sweep knob: the sweep seeds, and the extended sweep
    seeds for a knob in :data:`EXTENDED_SWEEP_KNOBS` (``seed_table`` is a
    family's ``SEEDS``). A seed restriction replaces both blocks."""
    base = _seeds(seed_table["sweeps"], split, seeds)
    if seeds is not None or knob not in EXTENDED_SWEEP_KNOBS:
        return base
    return base + seeds_of(seed_table["sweeps_extended"], split)


def catalogue_systems(kind: str, family: str = CATALOGUE_FAMILY,
                      held_out: Optional[bool] = None) -> tuple:
    """Catalogue ids of a kind (``witness``/``adversary``) defined for a
    family, in catalogue order (optionally only held-out or kept ones)."""
    out = []
    for sid in A2.system_ids(kind=kind, family=family):
        if held_out is None or bool(A2.get_entry(sid).get("held_out")) == held_out:
            out.append(sid)
    return tuple(out)


def catalogue_task(design: str, family: str, catalogue_family: str, system_id: str,
                   seed: int, scorings, *, variant=None, replicate: int = 0,
                   preset=None, held_out: bool = False, tags=None,
                   module: str = MODULE, suffix=None, name=None) -> TaskSpec:
    """A task that simulates a catalogue system. ``name`` replaces the
    catalogue id in the task id and the record's system name (for example a
    RAM-only class); ``suffix`` (default: the variant) is appended to the
    task id. A variant stays out of the system name (it is in
    ``params['variant']``), so the variants of one catalogue system share
    its name, as the hypotheses select them."""
    entry = A2.get_entry(system_id)
    params = {"id": system_id, "family": catalogue_family}
    if variant is not None:
        params["variant"] = variant
    if preset is not None:
        params["preset"] = preset
    base = system_id if name is None else name
    system = base
    tg = {"kind": entry["kind"], "class": entry.get("class")}
    tg.update(tags or {})
    return TaskSpec(
        task_id=task_id(design, base, seed, replicate,
                        suffix=variant if suffix is None else suffix),
        design=design, family=family, system=system, seed=int(seed),
        builder="catalogue", params=params, replicate=int(replicate),
        scorings=scorings, held_out=held_out, tags=tg, design_module=module)


def agent_task(design: str, family: str, catalogue_family: str, system: str,
               seed: int, knobs: dict, scorings, *, replicate: int = 0,
               preset=None, tags=None, module: str = MODULE) -> TaskSpec:
    """A task that simulates the family's agent at given knobs (sweep levels
    and factorial cells: no catalogue entry)."""
    params = {"family": catalogue_family, "knobs": dict(knobs)}
    if preset is not None:
        params["preset"] = preset
    return TaskSpec(
        task_id=task_id(design, system, seed, replicate), design=design,
        family=family, system=system, seed=int(seed), builder="agent",
        params=params, replicate=int(replicate), scorings=scorings,
        tags=dict(tags or {}), design_module=module)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def witnesses(split: str, *, seeds: Optional[Sequence[int]] = None,
              systems: Optional[Sequence[str]] = None,
              forms: Optional[Sequence[str]] = None) -> List[TaskSpec]:
    """``A_witnesses``: every family-A witness on the witness seeds, PC_nominal
    and the six single deficits on 25 more confirmatory seeds and the two
    null witnesses on 26 more; the held-out re-scorings on the confirmatory
    split only."""
    ids = first(_select(catalogue_systems("witness"), systems, "witnesses"),
                EXTENDED_SYSTEMS)
    forms = FORMS["witnesses"] if forms is None else tuple(forms)
    base_seeds = _seeds(SEEDS["witnesses"], split, seeds)
    ext_seeds = (() if seeds is not None
                 else seeds_of(SEEDS["witnesses_extended"], split))
    null_ext_seeds = (() if seeds is not None
                      else seeds_of(SEEDS["null_witnesses_extended"], split))
    rescored = set(seeds_of(SEEDS["held_out_rescorings"], split))
    misdeclared = set(seeds_of(SEEDS["misdeclared_access"], split))
    out = []
    for sid in ids:
        sys_seeds = (base_seeds + (ext_seeds if sid in EXTENDED_SYSTEMS else ())
                     + (null_ext_seeds if sid in NULL_WITNESSES else ()))
        modes = bearer_modes(sid)
        tags = {"pdi_partition": True} if sid in PDI_PARTITION_SYSTEMS else None
        for seed in sys_seeds:
            sc = make_scorings(PROTOCOLS, PRINCIPLES, forms, bearer_modes=modes,
                               form_declarations=FORM_DECLARATIONS)
            if sid in HELD_OUT_RESCORED and seed in rescored:
                sc += make_scorings(HELD_OUT_PROTOCOLS, HELD_OUT_RESCORED_PRINCIPLES,
                                    held_out=True, verdict=False)
            if sid == MISDECLARED_ACCESS_SYSTEM and seed in misdeclared:
                sc += make_scorings({"R": PROTOCOLS["R"]}, ("PDI",),
                                    ("pdi_misdeclared_access",), primary=False)
            out.append(catalogue_task("A_witnesses", FAMILY, CATALOGUE_FAMILY, sid,
                                      seed, sc, tags=tags))
    return out


def _agent_sweeps(design: str, family: str, catalogue_family: str,
                  sweep_knobs: Sequence[str], seed_table, split: str,
                  scorings: Callable[[], tuple], *, seeds=None, knobs=None,
                  module: str = MODULE) -> List[TaskSpec]:
    """The sweep tasks of a family's agent (families A and C1): every dose
    level of each knob in ``sweep_knobs``, the other knobs nominal, on the
    knob's seeds (:func:`sweep_seeds` of ``seed_table``); ``scorings()``
    gives the scorings of one task."""
    from impact_pipeline.bench.generators import NOMINAL_KNOBS

    out = []
    for knob in _select(sweep_knobs, knobs, "sweep knobs"):
        for li, level in enumerate(sweep_levels(knob)):
            kn = NOMINAL_KNOBS.replace(**{knob: level})
            system = f"sweep_{knob}_l{li:02d}"
            tags = {"sweep_knob": knob, "sweep_level": float(level),
                    "bits": list(kn.bits())}
            for seed in sweep_seeds(seed_table, knob, split, seeds):
                out.append(agent_task(design, family, catalogue_family, system,
                                      seed, kn.to_dict(), scorings(), tags=tags,
                                      module=module))
    return out


def _agent_factorial(design: str, family: str, catalogue_family: str,
                     seed_table, split: str, scorings: Callable[[], tuple], *,
                     seeds=None, cells=None, module: str = MODULE) -> List[TaskSpec]:
    """The 2^5 factorial cells of a family's agent (families A and C1; a 1
    keeps the nominal dose) on the seeds of ``seed_table['factorial']``;
    ``scorings()`` gives the scorings of one task."""
    from impact_pipeline.bench.factorial import factorial_cells

    all_cells = factorial_cells()
    wanted = _select([c["cell_id"] for c in all_cells], cells, "factorial cells")
    out = []
    for cell in all_cells:
        if cell["cell_id"] not in wanted:
            continue
        for seed in _seeds(seed_table["factorial"], split, seeds):
            out.append(agent_task(design, family, catalogue_family,
                                  cell["cell_id"], seed, cell["knobs"], scorings(),
                                  tags={"bits": list(cell["bits"])}, module=module))
    return out


def sweeps(split: str, *, seeds: Optional[Sequence[int]] = None,
           knobs: Optional[Sequence[str]] = None,
           forms: Optional[Sequence[str]] = None) -> List[TaskSpec]:
    """``A_sweeps``: g_b, c_int and K dose levels, the other knobs nominal;
    the c_int levels also on the extended sweep seeds."""
    forms = FORMS["default"] if forms is None else tuple(forms)
    return _agent_sweeps("A_sweeps", FAMILY, CATALOGUE_FAMILY, SWEEP_KNOBS, SEEDS,
                         split, lambda: make_scorings(PROTOCOLS, PRINCIPLES, forms),
                         seeds=seeds, knobs=knobs)


def factorial(split: str, *, seeds: Optional[Sequence[int]] = None,
              cells: Optional[Sequence[str]] = None,
              forms: Optional[Sequence[str]] = None) -> List[TaskSpec]:
    """``A_factorial``: the 2^5 cells (a 1 keeps the nominal dose)."""
    forms = FORMS["default"] if forms is None else tuple(forms)
    return _agent_factorial("A_factorial", FAMILY, CATALOGUE_FAMILY, SEEDS, split,
                            lambda: make_scorings(PROTOCOLS, PRINCIPLES, forms),
                            seeds=seeds, cells=cells)


def adversaries(split: str, *, seeds: Optional[Sequence[int]] = None,
                systems: Optional[Sequence[str]] = None,
                forms: Optional[Sequence[str]] = None) -> List[TaskSpec]:
    """``A_adversaries``: the kept v1 adversaries and the three staggered-
    driver variants."""
    forms = FORMS["default"] if forms is None else tuple(forms)
    ids = _select(catalogue_systems("adversary", held_out=False), systems,
                  "adversaries")
    out = []
    for sid in ids:
        variants = list(A2.get_entry(sid).get("variants") or []) or [None]
        for variant in variants:
            for seed in _seeds(SEEDS["adversaries"], split, seeds):
                out.append(catalogue_task(
                    "A_adversaries", FAMILY, CATALOGUE_FAMILY, sid, seed,
                    make_scorings(PROTOCOLS, PRINCIPLES, forms), variant=variant))
    return out


def held_out(split: str, *, seeds: Optional[Sequence[int]] = None) -> List[TaskSpec]:
    """``A_heldout``: confirmatory, the two held-out staggered-driver systems
    (NAS under R). Development: smoke tasks on seeds 980-984 of every
    held-out condition of family A (the two systems, the re-scorings under
    P, Q10, Q25 and J, the misdeclared access node), whose outputs the
    runner discards."""
    ids = catalogue_systems("adversary", held_out=True)
    sd = _seeds(SEEDS["held_out"], split, seeds)
    if split == DEVELOPMENT and not all(S.is_smoke_seed(s) for s in sd):
        raise DesignError("held-out conditions run on development seeds only as "
                          "smoke tests (seeds 980-984)")
    out = []
    for sid in ids:
        for variant in list(A2.get_entry(sid).get("variants") or []) or [None]:
            for seed in sd:
                out.append(catalogue_task(
                    "A_heldout", FAMILY, CATALOGUE_FAMILY, sid, seed,
                    make_scorings({"R": PROTOCOLS["R"]}, HELD_OUT_SYSTEM_PRINCIPLES,
                                  verdict=False),
                    variant=variant, held_out=True))
    if split == DEVELOPMENT:
        for sid in HELD_OUT_RESCORED:
            for seed in sd:
                sc = make_scorings(HELD_OUT_PROTOCOLS, HELD_OUT_RESCORED_PRINCIPLES,
                                   held_out=True, verdict=False)
                out.append(catalogue_task("A_heldout", FAMILY, CATALOGUE_FAMILY, sid,
                                          seed, sc, held_out=True,
                                          suffix="rescored"))
        for seed in sd:
            sc = make_scorings({"R": PROTOCOLS["R"]}, ("PDI",),
                               ("pdi_misdeclared_access",), primary=False)
            out.append(catalogue_task("A_heldout", FAMILY, CATALOGUE_FAMILY,
                                      MISDECLARED_ACCESS_SYSTEM, seed, sc,
                                      held_out=True, suffix="misdeclared"))
    return out


# The methods of every joint-bench scoring (also used by the other designs).
JOINT_BENCH_METHODS = (
    "each principle scored by the estimator version its family protocol names "
    "(nas-v3-2026.10, iim-v5-2026.10, ram-v3-2026.10, pdi-v3-2026.10 and the v1 "
    "SRPI srpi-v2-2026.09), every estimator in its own try block, and judged by "
    "the tost-v2 status rule on the protocol's anchors (construct scale c, "
    "z = 0.25, delta = 0.10, alpha = 0.05, alpha_A = 0.01); the verdict is the "
    "strong-Kleene AND over the protocol's anchored necessity set")

ADEMP = {
    "A_witnesses": {
        "aims": "construct validity of each instrument and of the verdict on "
                "systems defined by mechanism (positive control, single deficits, "
                "nulls, out-of-scope and patchwork systems), identifiability "
                "under the complete and the task-only declaration, and (held "
                "out) the cost of a misspecified declaration",
        "data": "family-A agents (v1 routing, recording device; 80 trials, 30 "
                "reafference pairs), the 16 catalogue witnesses on 20 seeds and "
                "PC_nominal with the six single deficits on 45 and the two null "
                "witnesses on 46; R and H from one "
                "simulation; P, Q10, Q25 and J re-scorings of five witnesses on "
                "20 seeds and the misdeclared PDI access node (held out)",
        "estimands": "per system, protocol and principle the construct value c, "
                     "the status and its reason; per system and protocol the "
                     "verdict; paired contrasts with PC_nominal on shared seeds",
        "methods": (JOINT_BENCH_METHODS + "; reported forms: NAS secondary, IIM "
                    "bidirectional, NAS at tau_c 0.05 s and 0.2 s"),
        "performance": "status rates per class with seed-clustered bounds (H0 "
                       "cell rule, Clopper-Pearson, cluster bootstrap), paired "
                       "contrast proportions (three-zone rules) and verdict "
                       "specificity: HCv2-1, -3, -5, -7, -8, -9, -14, -18 to -20, "
                       "-22 and -23",
    },
    "A_sweeps": {
        "aims": "dose-response and exclusion safety of the instruments along "
                "the mechanism doses",
        "data": "family-A agents with g_b (10 levels 0-2, K = 6), c_int (10 "
                "levels 0-1.2) or K in {2, 3, 4, 6, 8, 12} varied and the other "
                "knobs nominal; 10 seeds (c_int 65, for HCv2-14(f)); R and H",
        "estimands": "c and status per principle and dose level",
        "methods": (JOINT_BENCH_METHODS + "; reported forms: NAS secondary, IIM "
                    "bidirectional"),
        "performance": "Spearman and OLS trends of c on the dose, ABSENT rates "
                       "among mechanism-on levels, PRESENT across the v1 masking "
                       "window: HCv2-5, -7(iv), -14(f) and -19",
    },
    "A_factorial": {
        "aims": "no PRESENT without the mechanism, and no false exclusion with "
                "it, over all combinations of the five switches",
        "data": "the 2^5 family-A cells over eta, K, g_b, c_int and e (a 1 keeps "
                "the nominal dose); 10 seeds; R and H",
        "estimands": "status per principle and cell, with the cell's own "
                     "mechanism on or off",
        "methods": (JOINT_BENCH_METHODS + "; reported forms: NAS secondary, IIM "
                    "bidirectional"),
        "performance": "P(PRESENT | own mechanism off) per off-cell (H0 cell "
                       "rule) and ABSENT among on-cells: HCv2-5, -16, -18, -19(a) "
                       "and -24",
    },
    "A_adversaries": {
        "aims": "resistance to constructions that fake a mechanism (hidden or "
                "declared drivers, reflexes, label noise, scrambled feedback, "
                "edge-free staggered drivers)",
        "data": "the kept v1 adversaries and ADV_NAS_staggered_driver "
                "(hierarchical, uniform, reversed); 20 seeds; R and H",
        "estimands": "c, status and verdict per adversary and declaration; NAS "
                     "per-direction significance",
        "methods": (JOINT_BENCH_METHODS + "; reported forms: NAS secondary, IIM "
                    "bidirectional"),
        "performance": "PRESENT rates under R (H0 cell rule) and significance "
                       "rates under H (three-zone rules), potency reported: "
                       "HCv2-1(iii), -8(b) and the linear comparator of -9(e)",
    },
    "A_heldout": {
        "aims": "the stated limits of the declared-input correction (held out: "
                "predictions written from mechanism before any estimator output)",
        "data": "the hierarchical staggered driver with time constants x 10 and "
                "with a saturating transform; 20 seeds; NAS under R. "
                "Development: smoke tests on seeds 980-984, outputs discarded",
        "estimands": "NAS c and per-direction significance under R",
        "methods": "NAS by the version the family protocol names, judged by "
                   "tost-v2 on the A-R anchors; component statuses only",
        "performance": "significance rate (three-zone at 0.5) and the paired sign "
                       "test against the linear hierarchical driver: HCv2-9(d, e)",
    },
}

DESIGNS = (
    Design("A_witnesses", FAMILY,
           "family-A witnesses under R and H (45 seed clusters for PC_nominal and "
           "the six single deficits, 46 for the two null witnesses) with the "
           "held-out re-scorings",
           witnesses, {CONFIRMATORY: 16 * 20 + 7 * 25 + 2 * 26,
                       DEVELOPMENT: 16 * 12},
           ademp=ADEMP["A_witnesses"]),
    Design("A_sweeps", FAMILY,
           "family-A dose sweeps of g_b, c_int and K (c_int on 65 seeds)",
           sweeps, {CONFIRMATORY: 26 * 10 + 10 * 55, DEVELOPMENT: 26 * 4},
           ademp=ADEMP["A_sweeps"]),
    Design("A_factorial", FAMILY, "family-A 2^5 factorial", factorial,
           {CONFIRMATORY: 32 * 10, DEVELOPMENT: 32 * 4}, ademp=ADEMP["A_factorial"]),
    Design("A_adversaries", FAMILY,
           "kept v1 adversaries and the staggered hidden driver (three variants)",
           adversaries, {CONFIRMATORY: 7 * 20, DEVELOPMENT: 7 * 12},
           ademp=ADEMP["A_adversaries"]),
    Design("A_heldout", FAMILY,
           "held-out staggered-driver systems (confirmatory); smoke tests of every "
           "family-A held-out condition (development)",
           held_out, {CONFIRMATORY: 2 * 20, DEVELOPMENT: (2 + 5 + 1) * 5},
           ademp=ADEMP["A_heldout"]),
)

# Re-scorings of the witness simulations under the held-out declarations
# (counted as scorings, not tasks): 5 systems x 20 seeds x 4 declarations.
EXPECTED_HELD_OUT_RESCORINGS = {CONFIRMATORY: 5 * 20 * 4}

__all__ = [
    "ADEMP",
    "CATALOGUE_FAMILY",
    "DESIGNS",
    "EXPECTED_HELD_OUT_RESCORINGS",
    "EXTENDED_SWEEP_KNOBS",
    "EXTENDED_SYSTEMS",
    "FAMILY",
    "FORMS",
    "FORM_DECLARATIONS",
    "HELD_OUT_PROTOCOLS",
    "HELD_OUT_RESCORED",
    "JOINT_BENCH_METHODS",
    "K_LEVELS",
    "NULL_WITNESSES",
    "PDI_PARTITION_SYSTEMS",
    "PRINCIPLE_BEARER_SYSTEMS",
    "PROTOCOLS",
    "SEEDS",
    "SWEEP_KNOBS",
    "adversaries",
    "agent_task",
    "bearer_modes",
    "catalogue_systems",
    "catalogue_task",
    "factorial",
    "first",
    "held_out",
    "sweep_levels",
    "sweep_seeds",
    "sweeps",
    "witnesses",
]
