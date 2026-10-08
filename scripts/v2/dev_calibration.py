#!/usr/bin/env python
"""
Development calibration of MPC-Bench v2: one command per calibration item,
development seeds 0-999 only.

The development calibration fixes, before the freeze, every threshold and
setting the design leaves open (the calibration decisions CD-1 to CD-14).
This script produces the development outputs those decisions are taken
from; it takes no decision itself. The protocol builder
(``scripts/v2/build_protocols_v2.py``) reads the outputs and a
calibration-decisions file and writes the frozen protocols.

Items
-----
``plan`` prints every item with its runs, task counts and the projected
CPU-hours; :data:`ITEMS` is the one table of them. Each item writes to
``outputs/mpcbench_v2/dev_calibration/<item>/<run>/`` (``--root``), one
directory per run of the item (a runner plan, the family-B validation, the
oracle checks or the constants), with ``provenance.json`` beside the
outputs: the item and run, the seeds and systems, the runner settings (the
principles scored, the v1 null size and jackknife groups, the PDI k-means
seed and the IIM macro-node cap), every pending calibration parameter, the
estimator options a run overrides, the release it runs under, the code
identity and the command line. Runner runs resume (completed task ids are
skipped, failed tasks and an interrupted last line run again); the
family-B validation resumes in the same way; the constants and oracle
checks are recomputed.

Seed and held-out policy
------------------------
Every task seed must be a development seed (0-999); a seed of 1000 or more
is refused before anything is simulated. A held-out condition (no v2
estimator output before the freeze) runs only as a smoke test on the seeds
980-984, whose outputs the runner discards unread (``smoke.jsonl`` keeps
status lines only); every other held-out task is refused. Family-B cells of
the held-out non-monotone regime (ring coupling 0.9 and 1.5,
:data:`FAMILY_B_HELD_OUT`) are left out of the development run, and so is
IIM on the Hopf arm's v2 sensor views above G = 0 (the G sweep of the
sensor pipeline is held out at every regime; :data:`HO6_VIEWS`).

The one exception is written into the design: the reference-block anchors of
the forward views at the held-out regime (seeds 900-939) may be computed
once the held-out predictions are committed. They run only through an
explicit, logged release step, ``release-held-out-anchors --predictions
FILE ...``: it checks that every predictions file is tracked by git and
unchanged since its last commit, that no output of the released item exists
yet, and appends the release (files, their SHA-256 and commits, ``HEAD``,
time, note) to ``held_out_release_log.jsonl`` under the root. The item
``anchors_forward_held_out`` refuses to run without a release whose
predictions files are still unchanged; its tasks carry the release id
(``tags.held_out_release``), which the protocol builder checks against the
log.

Commands
--------
::

    python scripts/v2/dev_calibration.py plan [--items a,b] [--workers 12] [--json F]
    python scripts/v2/dev_calibration.py run <item>[,<item>...] [--workers 12]
        [--root DIR] [--seeds 900-901] [--systems PC_nominal] [--limit N]
        [--protocol-dir DIR] [--tag TAG] [--no-resume]
    python scripts/v2/dev_calibration.py release-held-out-anchors
        --predictions FILE [FILE ...] [--note TEXT]
    python scripts/v2/dev_calibration.py status [--root DIR]

``--seeds``, ``--systems`` and ``--limit`` restrict a run (smoke tests of the
calibration machinery on a tiny grid); a restricted run must use its own
``--root`` or ``--tag``, because a resumed run continues the same plan only.
Under a ``--seeds`` or ``--systems`` restriction the runs of an item that
have no task there are skipped (an item with none is refused).
``--protocol-dir`` runs the item with the protocols of a directory (for
example a re-run of the reference block with the SE methods of the
decisions); ``--tag`` then writes to ``<item>@<tag>/``.

Sequence
--------
1. ``run constants,oracle_checks`` (CD-1, CD-13);
2. ``run anchors_A,anchors_C1,anchors_RAM160 --workers 12`` (CD-7, CD-11);
3. ``run twins,twins_nas_se --workers 12`` (CD-2 to CD-4);
4. ``run pdi_concordance --workers 12`` (CD-5);
5. ``run dry_run --workers 12`` (CD-6 to CD-10) and ``run smoke``;
6. once the held-out predictions are committed:
   ``release-held-out-anchors --predictions FILE ...``, then
   ``run anchors_forward_held_out --workers 12``;
7. ``scripts/v2/build_protocols_v2.py template`` writes the decisions file
   with the rules' suggestions; the decisions are taken there, then
   ``build_protocols_v2.py build --decisions FILE``. A decided SE method
   other than the one the runs used: re-run the affected items with
   ``--protocol-dir <the build> --tag decided`` and build again. The
   optional items run only where a decision needs them (the SE fallbacks,
   the occupancy cells at a raised ``N_min``, the forward anchors at the
   development regime, the development evaluation of the dry run).

The development evaluation (``dry_run_evaluation``, CD-10) re-judges the
development records under the generated protocols before the evaluator
reads them: the runs were judged under the draft protocols, whose anchors
are pending, so their stored statuses are ``UNDEFINED(INVALID_ANCHORS)``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from impact_pipeline.bench import designs_v2 as D  # noqa: E402
from impact_pipeline.bench import run_bench_v2 as RB  # noqa: E402
from impact_pipeline.bench.designs_v2 import family_a as FA  # noqa: E402
from impact_pipeline.v2 import provenance as PV  # noqa: E402
from impact_pipeline.v2 import records as REC  # noqa: E402
from impact_pipeline.v2 import seeds as S  # noqa: E402

CALIBRATION_VERSION = "mpc-bench-dev-calibration/1.0.0"
PROVENANCE_SCHEMA = "mpc-bench-dev-calibration-run/1"
RELEASE_SCHEMA = "mpc-bench-held-out-release/1"
DEFAULT_ROOT = REPO_ROOT / "outputs" / "mpcbench_v2" / "dev_calibration"
PROVENANCE_JSON = "provenance.json"
RELEASE_LOG = "held_out_release_log.jsonl"
CONSTANTS_JSON = "constants.json"
FAMILY_B_JSONL = "iim_validation_v2.jsonl"
DEVELOPMENT = S.DEVELOPMENT

# kinds of runs
RUNNER, BATTERY, FAMILY_B, MANIPULATION, CONSTANTS, EVALUATE = (
    "runner", "battery", "family_b", "manipulation", "constants", "evaluate")
KINDS = (RUNNER, BATTERY, FAMILY_B, MANIPULATION, CONSTANTS, EVALUATE)

# --------------------------------------------------------------------------
# declared constants of the calibration (one place each)
# --------------------------------------------------------------------------
# The seeds of the development blocks (design 3.7; protocols/v2/seed_map_v2.json).
ORACLE_SEEDS = tuple(range(320, 360))
ORACLE_FAMILIES = ("A", "C")
# The PDI concordance battery runs on 850-899 and 940-979 (90 seeds); see
# the comment at CONCORDANCE_MIN_RUNS.
BATTERY_SEEDS = tuple(range(850, 900)) + tuple(range(940, 980))
BATTERY_DESIGN = "pdi_concordance_battery"
BATTERY_PROTOCOL = "A-R"
CONTENT_ON, CONTENT_OFF = "on", "off"
# The PDI concordance battery (CD-5): content-on classes (K in {2, 3, 6};
# g_b at the v1 sweep levels 0.67 and 1.56 with K = 6; K = 6 at the nominal
# g_b = 1.0 is PC_nominal) and the no-content classes. (class, content,
# builder, catalogue id or knobs).
BATTERY_CLASSES = (
    ("K2", CONTENT_ON, "agent", {"K": 2}),
    ("K3", CONTENT_ON, "agent", {"K": 3}),
    ("K6", CONTENT_ON, "catalogue", "PC_nominal"),
    ("g_b_0.67", CONTENT_ON, "agent", {"g_b": 0.666667}),
    ("g_b_1.56", CONTENT_ON, "agent", {"g_b": 1.555556}),
    ("W_PDI_single_attractor", CONTENT_OFF, "catalogue", "W_PDI_single_attractor"),
    ("W_PDI_no_multistability", CONTENT_OFF, "catalogue", "W_PDI_no_multistability"),
    ("O_hypersynchronous", CONTENT_OFF, "catalogue", "O_hypersynchronous"),
    ("N_ar1", CONTENT_OFF, "catalogue", "N_ar1"),
)
BATTERY_CLASS_IDS = tuple(c[0] for c in BATTERY_CLASSES)
# The admission rule of CD-5 needs >= 300 content-on (ABSENT route) and >= 300
# no-content runs (PRESENT route) per cell: 0 events in 300 runs is the
# smallest count whose one-sided 95 % Clopper-Pearson bound is below
# alpha_A = 0.01. The 30 seeds first planned (870-899) give only 150 and 120
# runs, so the battery was enlarged to 90 seeds before any battery output
# existed: 450 content-on and 360 no-content runs, which leaves room for
# runs without an estimator output.
CONCORDANCE_MIN_RUNS = 300
N_CONTENT_ON_CLASSES = sum(c[1] == CONTENT_ON for c in BATTERY_CLASSES)
N_NO_CONTENT_CLASSES = len(BATTERY_CLASSES) - N_CONTENT_ON_CLASSES
# Family-B cells of the held-out non-monotone regime (design 1.4, HO-5): no
# v2 estimator output before the freeze. (system, parameter, values)
FAMILY_B_HELD_OUT = (("ring", "coupling", (0.9, 1.5)),)
# IIM's sensitivity to the Hopf coupling at sensor level (design 1.4, HO-6):
# the G sweep of the v2 sensor pipeline, read by HCv2-15 (e) (FMb1 and FMd on
# the sensor views), is held out at every regime. The forward design runs its
# development purposes at the development regime with every G level; the
# calibration keeps IIM on these views only at G = 0 (HCv2-15 (a), a
# replication of development results) and leaves it out of the scorings at
# every other level, as family C1 leaves IIM out on the development split
# (HO-4). The released reference anchors at the held-out regime are the
# design's exception and keep it. (arm, principle, views, purposes)
HO6_ARM = "hopf"
HO6_PRINCIPLE = "IIM"
HO6_VIEWS = ("eeg64", "eeg64_noref", "eeglow", "mne_template")
HO6_PURPOSES = ("dry_run", "dev_regime", "reference_development")
# The alternative SE methods whose twin calibration the decisions compare
# with the default method of the runner's protocols (CD-2 for NAS; the
# fallbacks of CD-3 and CD-4 are optional items).
NAS_SE_ALTERNATIVES = ("jackknife_contiguous_20", "jackknife_interleaved_10",
                       "jackknife_interleaved_20")
IIM_SE_FALLBACK = "jackknife_contiguous_10"
RAM_SE_FALLBACK = "jackknife_trials_10"
# The raised IIM occupancy thresholds N_min of CD-3 (iim_v5.N_MIN_ALLOWED
# above the default): the development occupancy cells of HCv2-12 (c) are
# re-run under each, designated at that N_min, if a cell fails at the
# default.
OCCUPANCY_N_MIN_ALTERNATIVES = (50, 100)
# The adversary potency of CD-9 is a paired ratio with the positive control
# on the same seed (common random numbers): PC_nominal runs on the
# development adversary seeds as well.
ADVERSARY_SEEDS = D.seeds_of(FA.SEEDS["adversaries"], DEVELOPMENT)
EVALUATION_RECORDS = "development_records.jsonl"
# Wall time of a run: CPU-h x contention / workers (design 7.2: x 1.39 measured
# with 12 workers).
CONTENTION = 1.39
DEFAULT_WORKERS = 12

# Cost model of the plan: CPU seconds per task (process time of the worker,
# one BLAS thread), measured on one development seed of every runner design
# by the integration run at commit d3bcb09 (two lanes of 5 workers on 12
# performance cores; development seeds only) and, for the family-B
# validation, one development seed per seed block. A projection, not a
# measurement of these runs: one seed per system, seed-to-seed spread not
# measured.
COST_SOURCE = ("integration run at commit d3bcb09 on development seeds: CPU s per "
               "task by design (process time, one BLAS thread)")
CPU_S_PER_TASK = {
    "A_adversaries": 110.60, "A_anchors": 97.55, "A_factorial": 116.28,
    "A_sweeps": 113.12, "A_twins": 119.22, "A_witnesses": 105.83,
    "C1_anchors": 74.56, "C1_factorial": 3.46, "C1_sweeps": 3.76,
    "C1_twins": 3.05, "C1_witnesses": 4.21, "RAM160": 0.48,
    "RAM160_anchors": 0.32, "RAM160_twins": 0.31,
    "forward_anchor_replication": 118.83, "forward_family_a": 127.00,
    "forward_family_a_bold": 40.77, "null_calibration": 34.83,
    "whole_brain": 107.97, "family_b": 13.19,
    # held-out smoke tasks (status lines of the integration run)
    "A_heldout": 93.77,
}
# Runs restricted to some principles: the components' own seconds of a
# family-A task (development seed 900: IIM about 33 s per declaration, PDI
# about 31-34 s once, SRPI 3.6 s, every NAS scoring together 1.4-1.9 s,
# RAM-PE 0.07 s) plus the simulation (0.15-0.2 s); C1 twins carry no IIM on
# development seeds (IIM is held out on C1 outside the anchors).
RESTRICTED_CPU_S_PER_TASK = {
    ("A_twins", ("NAS",)): 2.5, ("C1_twins", ("NAS",)): 1.5,
    ("A_twins", ("IIM",)): 70.0, ("RAM160_twins", ("RAM",)): 0.31,
    (BATTERY_DESIGN, ("PDI",)): 35.0,
}
# The oracle checks (prerequisite M and the realisation checks) per seed of
# families A and C: one development seed took 9.7 s including the start-up
# of the process (calibration smoke test, CD-13).
ORACLE_CPU_S_PER_SEED = 8.0


class CalibrationError(ValueError):
    """A calibration run that the seed, held-out or release policy refuses,
    or an unknown item."""


class NoTasks(CalibrationError):
    """A run without tasks (under a ``--seeds`` or ``--systems`` restriction
    the run is skipped; an item none of whose runs has a task is refused)."""


# --------------------------------------------------------------------------
# the items
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Run:
    """One run of an item: a runner plan of ``designs`` (``purpose`` for the
    forward arms), the PDI battery, the family-B validation, the oracle
    checks, the constants or an evaluation. ``seeds`` and ``systems``
    restrict the designs' plan, ``principles`` the principles scored
    (:class:`RunSettings`), ``estimator_options`` overrides estimator
    options of every protocol of the plan (``{principle: options}``; for the
    family-B validation the IIM settings, ``n_min`` selecting the occupancy
    cells of HCv2-12 (c) designated at that ``N_min``), ``released`` marks
    the held-out-regime reference anchors."""

    name: str
    kind: str
    designs: Tuple[str, ...] = ()
    purpose: Optional[str] = None
    seeds: Optional[Tuple[int, ...]] = None
    systems: Optional[Tuple[str, ...]] = None
    principles: Optional[Tuple[str, ...]] = None
    estimator_options: Mapping = field(default_factory=dict)
    released: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise CalibrationError(f"run {self.name}: kind must be one of {KINDS}")


@dataclass(frozen=True)
class Item:
    """One calibration item: the decisions it informs, its runs, whether it
    belongs to the planned development calibration or is optional (taken
    only if a decision needs it) and whether it needs the held-out release."""

    name: str
    decisions: Tuple[str, ...]
    description: str
    runs: Tuple[Run, ...]
    optional: bool = False
    requires_release: bool = False
    after: Tuple[str, ...] = ()


FORWARD_ARMS = ("whole_brain", "forward_family_a", "forward_family_a_bold")

ITEMS: Tuple[Item, ...] = (
    Item("constants", ("CD-1",),
         "deterministic constants: tau_c and lag sets per substrate from the "
         "generator constants, the family-B exact targets (every cell, ring 1.5 "
         "included), the anchors and the strict monotonicity of the HCv2-12 (a) "
         "sweeps; no estimator",
         (Run("constants", CONSTANTS),)),
    Item("oracle_checks", ("CD-13",),
         "prerequisite M and the realisation checks of the new systems on "
         "development seeds 320-359 (oracle channels only)",
         (Run("manipulation", MANIPULATION, seeds=ORACLE_SEEDS),)),
    Item("anchors_A", ("CD-7", "CD-11"),
         "family-A reference block 900-939: PC_nominal and the six single deficits "
         "under A-R and A-H with every form that carries its own anchor",
         (Run("A_anchors", RUNNER, ("A_anchors",)),)),
    Item("anchors_C1", ("CD-7", "CD-11"),
         "family-C1 reference block 900-939: PC_nominal, W_NAS_no_workspace, "
         "W_NAS_broadcast_only and W_IIM_feedforward (IIM included: the only IIM "
         "output on C1 before the freeze)",
         (Run("C1_anchors", RUNNER, ("C1_anchors",)),)),
    Item("anchors_RAM160", ("CD-7", "CD-11"),
         "RAM-only reference block 900-939: PC_nominal and W_RAM_no_plasticity at "
         "160 trials",
         (Run("RAM160_anchors", RUNNER, ("RAM160_anchors",)),)),
    Item("anchors_forward_held_out", ("CD-7",),
         "the anchor conditions of the forward views (Hopf G_nom, PC_nominal of "
         "both family-A arms) on 900-939 at the held-out regime; only after the "
         "held-out predictions are committed (release step)",
         (Run("forward_reference", RUNNER, ("forward_anchor_replication",),
              purpose="reference", released=True),),
         requires_release=True),
    Item("twins", ("CD-2", "CD-3", "CD-4"),
         "development twin networks 820-824: family-A classes (r = 0..6), C1 "
         "classes (r = 0..6) and RAM-only doses (r = 0..7) under the default SE "
         "methods",
         (Run("A_twins", RUNNER, ("A_twins",)),
          Run("C1_twins", RUNNER, ("C1_twins",)),
          Run("RAM160_twins", RUNNER, ("RAM160_twins",)))),
    Item("twins_nas_se", ("CD-2",),
         "the same family-A and C1 twins, NAS only, under each alternative NAS SE "
         "method (contiguous G = 20, interleaved G = 10 and 20)",
         tuple(Run(f"nas_{m}", RUNNER, ("A_twins", "C1_twins"), principles=("NAS",),
                   estimator_options={"NAS": {"se_method": m}})
               for m in NAS_SE_ALTERNATIVES)),
    Item("pdi_concordance", ("CD-5",),
         "PDI concordance battery on 850-899 and 940-979: content-on classes "
         "K 2, 3, 6 and g_b 0.67, 1.56 at K = 6; no-content classes "
         "W_PDI_single_attractor, W_PDI_no_multistability, O_hypersynchronous, "
         "N_ar1 (PDI only)",
         (Run("battery", BATTERY, (BATTERY_DESIGN,), seeds=BATTERY_SEEDS,
              principles=("PDI",)),)),
    Item("dry_run", ("CD-6", "CD-7", "CD-8", "CD-9", "CD-10"),
         "every Tier-A design at about 15 % scale on its development seeds "
         "(320-439, 804-819): family A, RAM-only arm, C1, the forward arms at the "
         "development regime, the null-calibration generator and family B; "
         "PC_nominal also on the adversary seeds (the paired potency ratio)",
         (Run("A_witnesses", RUNNER, ("A_witnesses",)),
          Run("A_sweeps_factorial", RUNNER, ("A_sweeps", "A_factorial")),
          Run("A_adversaries", RUNNER, ("A_adversaries",)),
          Run("A_adversary_controls", RUNNER, ("A_witnesses",), seeds=ADVERSARY_SEEDS,
              systems=("PC_nominal",)),
          Run("RAM160", RUNNER, ("RAM160",)),
          Run("C1", RUNNER, ("C1_witnesses", "C1_sweeps", "C1_factorial")),
          Run("forward_dry_run", RUNNER, FORWARD_ARMS, purpose="dry_run"),
          Run("forward_dev_regime", RUNNER, FORWARD_ARMS, purpose="dev_regime"),
          Run("null_calibration", RUNNER, ("null_calibration",)),
          Run("family_b", FAMILY_B, ("family_b",)))),
    Item("smoke", (),
         "plumbing of every held-out condition on the smoke seeds 980-984 (family "
         "A and the forward arms' anchor conditions at the held-out regime); the "
         "runner discards the outputs unread",
         (Run("A_heldout", RUNNER, ("A_heldout",)),
          Run("forward_smoke", RUNNER, FORWARD_ARMS, purpose="smoke"))),
    # optional items: run only when a decision needs them
    Item("anchors_forward_dev_regime", (),
         "the anchor conditions of the forward views on 900-939 at the development "
         "regime (development judgments of the forward dry run)",
         (Run("forward_reference_development", RUNNER,
              ("forward_anchor_replication",), purpose="reference_development"),),
         optional=True),
    Item("twins_iim_jackknife", ("CD-3",),
         "the family-A twins, IIM only, under the fallback SE method (contiguous "
         "jackknife, G = 10): needed only if the block bootstrap fails its twin "
         "calibration",
         (Run("iim_jackknife_contiguous_10", RUNNER, ("A_twins",), principles=("IIM",),
              estimator_options={"IIM": {"se_method": IIM_SE_FALLBACK}}),),
         optional=True),
    Item("occupancy_n_min", ("CD-3",),
         "the development occupancy cells of HCv2-12 (c) designated and scored at "
         "the raised N_min 50 and 100: needed only if a cell fails its 90 % "
         "criterion at the default N_min",
         tuple(Run(f"n_min_{n}", FAMILY_B, ("family_b",),
                   estimator_options={"IIM": {"n_min": int(n)}})
               for n in OCCUPANCY_N_MIN_ALTERNATIVES),
         optional=True),
    Item("twins_ram_jackknife", ("CD-4",),
         "the RAM-only twins under the fallback SE method (trial jackknife, G = "
         "10): needed only if the shift-null SD fails its twin calibration",
         (Run("ram_jackknife_trials_10", RUNNER, ("RAM160_twins",), principles=("RAM",),
              estimator_options={"RAM": {"se_method": RAM_SE_FALLBACK}}),),
         optional=True),
    Item("dry_run_evaluation", ("CD-10",),
         "the evaluator on the development records with the generated protocols "
         "(outputs flagged DEVELOPMENT - NOT A RESULT)",
         (Run("evaluation", EVALUATE),), optional=True,
         after=("dry_run", "anchors_A", "anchors_C1", "anchors_RAM160", "twins",
                "oracle_checks")),
)
ITEM_NAMES = tuple(i.name for i in ITEMS)


def get_item(name: str) -> Item:
    for it in ITEMS:
        if it.name == name:
            return it
    raise CalibrationError(f"unknown calibration item {name!r}; one of {ITEM_NAMES}")


def run_dir(root, item: str, run: str, tag: Optional[str] = None) -> Path:
    """``<root>/<item>[@<tag>]/<run>``."""
    base = item if not tag else f"{item}@{tag}"
    return Path(root) / base / run


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------
def battery_tasks(seeds: Optional[Sequence[int]] = None,
                  systems: Optional[Sequence[str]] = None) -> List[D.TaskSpec]:
    """The PDI concordance battery: every class of :data:`BATTERY_CLASSES` on
    every battery seed, PDI under the A-R protocol, component statuses only;
    tags ``battery_class`` and ``content``."""
    from impact_pipeline.bench.designs_v2 import family_a as FA
    from impact_pipeline.bench.generators import NOMINAL_KNOBS

    seeds = BATTERY_SEEDS if seeds is None else tuple(int(s) for s in seeds)
    classes = BATTERY_CLASS_IDS if systems is None else tuple(systems)
    unknown = sorted(set(classes) - set(BATTERY_CLASS_IDS))
    if unknown:
        raise CalibrationError(f"unknown battery classes {unknown}; one of "
                               f"{BATTERY_CLASS_IDS}")
    sc = D.make_scorings({"R": BATTERY_PROTOCOL}, ("PDI",), verdict=False)
    out = []
    for cls, content, builder, what in BATTERY_CLASSES:
        if cls not in classes:
            continue
        tags = {"battery_class": cls, "content": content}
        for seed in seeds:
            if builder == "catalogue":
                t = FA.catalogue_task(BATTERY_DESIGN, "A", "A", what, seed, sc,
                                      tags=tags, module=None, name=f"battery_{cls}")
            else:
                kn = NOMINAL_KNOBS.replace(**what)
                t = FA.agent_task(BATTERY_DESIGN, "A", "A", f"battery_{cls}", seed,
                                  kn.to_dict(), sc,
                                  tags=dict(tags, bits=list(kn.bits())), module=None)
            out.append(t)
    return out


def family_b_held_out(cell) -> bool:
    """Whether a family-B cell belongs to the held-out non-monotone regime
    (:data:`FAMILY_B_HELD_OUT`)."""
    params = dict(cell.params)
    for system, key, values in FAMILY_B_HELD_OUT:
        if cell.system == system and key in params and any(
                math.isclose(float(params[key]), float(v)) for v in values):
            return True
    return False


def family_b_tasks(seeds: Optional[Sequence[int]] = None,
                   n_min: Optional[int] = None) -> Tuple[list, list]:
    """The development family-B tasks without the held-out cells, and the ids
    of the cells left out. With ``n_min``: only the occupancy cells of
    HCv2-12 (c), designated at that ``N_min``."""
    from impact_pipeline.bench.designs_v2 import family_b as FB

    if n_min is None:
        tasks = FB.tasks(DEVELOPMENT, seeds=seeds)
    else:
        tasks = [t for t in FB.tasks(DEVELOPMENT, hypotheses=("HCv2-12",), seeds=seeds,
                                     n_min=int(n_min))
                 if "c" in t.cell.parts]
    keep = [t for t in tasks if not family_b_held_out(t.cell)]
    dropped = sorted({t.cell.cell_id for t in tasks if family_b_held_out(t.cell)})
    return keep, dropped


def ho6_withheld(tags: Mapping, view: str) -> bool:
    """Whether IIM on a scoring of ``view`` of a task with ``tags`` belongs
    to the held-out G sweep of the v2 sensor pipeline (:data:`HO6_VIEWS`;
    a Hopf task of a development purpose above G = 0; an unreadable dose
    counts as above)."""
    if (tags.get("arm") != HO6_ARM or tags.get("purpose") not in HO6_PURPOSES
            or view not in HO6_VIEWS):
        return False
    try:
        return float(tags.get("dose")) != 0.0
    except (TypeError, ValueError):
        return True


def withhold_ho6(tasks: Sequence[D.TaskSpec]) -> Tuple[List[D.TaskSpec], List[str]]:
    """The tasks with IIM left out of every scoring :func:`ho6_withheld`
    names (a scoring left without a principle is dropped), and the ids
    ``task/scoring`` of the scorings changed."""
    out, changed = [], []
    for t in tasks:
        tags = t.tags or {}
        scorings, touched = [], False
        for s in t.scorings:
            if HO6_PRINCIPLE in s.principles and ho6_withheld(tags, s.view):
                touched = True
                changed.append(f"{t.task_id}/{s.scoring_id}")
                keep = tuple(p for p in s.principles if p != HO6_PRINCIPLE)
                if not keep:
                    continue
                s = dataclasses.replace(s, principles=keep)
            scorings.append(s)
        out.append(dataclasses.replace(t, scorings=tuple(scorings)) if touched else t)
    return out, changed


def runner_tasks(run: Run, *, seeds=None, systems=None,
                 release: Optional[Mapping] = None,
                 withheld: Optional[list] = None) -> List[D.TaskSpec]:
    """The tasks of a runner run (or of the battery), checked by
    :func:`check_tasks`; the released held-out reference anchors are carried
    with the release id and without the held-out mark (the release makes
    exactly these development outputs admissible). A ``seeds`` or
    ``systems`` restriction reaches the designs whose builders take it; the
    tasks of a design whose builder takes no such option (the sweeps and the
    factorial have no ``systems``) are filtered by the task's seed or system
    name, and a design whose builder refuses the restriction (it has no task
    there) is left out of the run. A restriction that leaves no task raises
    :class:`NoTasks`. IIM on
    the held-out G sweep of the v2 sensor pipeline is left out
    (:func:`withhold_ho6`; the scorings changed are appended to
    ``withheld``)."""
    restricted = seeds is not None or systems is not None
    seeds = run.seeds if seeds is None else seeds
    systems = run.systems if systems is None else systems
    refused: List[str] = []
    if run.kind == BATTERY:
        tasks = battery_tasks(seeds, systems)
    else:
        # design by design: under a restriction a design that has nothing
        # there (its builder refuses the seeds or systems) leaves the other
        # designs of the run their tasks
        tasks = []
        for name in run.designs:
            try:
                tasks.extend(RB.build_tasks(
                    [name], DEVELOPMENT, seeds=None if seeds is None else list(seeds),
                    systems=None if systems is None else list(systems),
                    purpose=run.purpose))
            except (D.DesignError, ValueError) as exc:
                if not restricted:
                    raise
                refused.append(str(exc))
        ids = [t.task_id for t in tasks]
        if len(set(ids)) != len(ids):
            raise CalibrationError(f"run {run.name}: duplicate task ids")
        if seeds is not None:
            keep = {int(s) for s in seeds}
            tasks = [t for t in tasks if _takes(t.design, "seeds")
                     or int(t.seed) in keep]
        if systems is not None:
            keep_sys = {str(s) for s in systems}
            tasks = [t for t in tasks if _takes(t.design, "systems")
                     or t.system in keep_sys]
    if not tasks:
        why = f" ({'; '.join(refused)})" if refused else ""
        raise NoTasks(f"run {run.name}: no task at the requested seeds and "
                      f"systems{why}")
    tasks, changed = withhold_ho6(tasks)
    if withheld is not None:
        withheld.extend(changed)
    if run.released:
        if release is None:
            raise CalibrationError(
                f"run {run.name}: the held-out-regime reference anchors need the "
                "release step (release-held-out-anchors) first")
        tasks = [declassify(t, release["release_id"]) for t in tasks]
    check_tasks(tasks, run)
    return tasks


def _takes(design: str, option: str) -> bool:
    """Whether the builder of a registered design takes ``option``."""
    import inspect

    try:
        params = inspect.signature(D.get_design(design).build).parameters
    except (D.DesignError, KeyError, ValueError):
        return False
    return option in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def declassify(task: D.TaskSpec, release_id: str) -> D.TaskSpec:
    """A released reference task: the held-out mark removed from the task
    and its scorings, ``tags.held_out_release`` set. Only the forward
    reference anchors at the held-out regime are ever released."""
    tags = dict(task.tags)
    if tags.get("purpose") != "reference" or tags.get("regime") != "held_out" or (
            not tags.get("anchor")) or int(task.seed) not in S.REFERENCE_BLOCK:
        raise CalibrationError(f"{task.task_id}: only the forward anchor conditions at "
                               "the held-out regime on the reference block "
                               f"{S.REFERENCE_BLOCK.start}-{S.REFERENCE_BLOCK.stop - 1} "
                               "are released")
    tags["held_out_release"] = str(release_id)
    scorings = tuple(dataclasses.replace(s, held_out=False) for s in task.scorings)
    return dataclasses.replace(task, held_out=False, scorings=scorings, tags=tags)


def check_tasks(tasks: Sequence[D.TaskSpec], run: Optional[Run] = None) -> None:
    """The calibration's policy, before anything is simulated: development
    seeds only (a seed of 1000 or more is refused); a held-out condition only
    on the smoke seeds 980-984 (the runner discards its outputs)."""
    if not tasks:
        raise NoTasks("no tasks")
    high = sorted({t.seed for t in tasks if int(t.seed) > S.DEV_SEED_MAX})
    if high:
        raise CalibrationError(
            f"the development calibration uses seeds 0-{S.DEV_SEED_MAX} only; refused "
            f"seeds {high[:5]}")
    try:
        S.check_seeds([t.seed for t in tasks], DEVELOPMENT)
    except S.SeedPolicyError as exc:
        raise CalibrationError(str(exc)) from None
    bad = [t.task_id for t in tasks if t.has_held_out and not t.smoke]
    if bad:
        raise CalibrationError(
            f"held-out conditions run on development seeds only as smoke tests "
            f"(seeds 980-984, outputs discarded): refused {bad[:5]}")
    if run is not None and run.released:
        for t in tasks:
            if not t.tags.get("held_out_release"):
                raise CalibrationError(f"{t.task_id}: a released run carries its "
                                       "release id")


def protocol_overrides(tasks: Sequence[D.TaskSpec], options: Mapping,
                       protocol_dir=None) -> Dict[str, object]:
    """``{key: ProtocolV3}``: every protocol of the plan whose estimators the
    run changes, as the runner resolves it (the generated file, else the
    draft) with ``options`` merged into the estimator blocks and the SE
    methods admitted."""
    from impact_pipeline import evidence_v2 as E

    if not options:
        return {}
    keys = RB.protocol_keys(tasks)
    resolved = RB.resolve_protocols(keys, directory=protocol_dir)
    out = {}
    for key, rp in resolved.items():
        payload = rp.protocol.to_dict()
        est = payload.get("estimators") or {}
        touched = False
        for p, opts in options.items():
            if p not in est and p not in (payload.get("se_methods") or {}):
                continue
            est.setdefault(p, {}).update(dict(opts))
            m = dict(opts).get("se_method")
            if m is not None:
                methods = set((payload.get("se_methods") or {}).get(p) or ())
                payload.setdefault("se_methods", {})[p] = sorted(methods | {m})
            touched = True
        if touched:
            payload["estimators"] = est
            out[key] = E.ProtocolV3.from_dict(payload)
    return out


# --------------------------------------------------------------------------
# the held-out release
# --------------------------------------------------------------------------
def _git(*args, cwd=REPO_ROOT) -> Optional[str]:
    try:
        proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                              text=True, timeout=60, check=False)
    except Exception:  # noqa: BLE001
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _rel(path: Path, repo_root: Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(Path(repo_root).resolve()))
    except ValueError:
        raise CalibrationError(f"{path}: the predictions must be files of this "
                               "repository") from None


def predictions_state(paths: Sequence, repo_root=REPO_ROOT) -> List[dict]:
    """Each predictions file: path, SHA-256, tracked, clean (no uncommitted
    change), the last commit that touched it and its commit time."""
    out = []
    for p in paths:
        p = Path(p)
        if not p.is_absolute():
            p = Path(repo_root) / p
        if not p.is_file():
            raise CalibrationError(f"no predictions file {p}")
        rel = _rel(p, repo_root)
        tracked = _git("ls-files", "--error-unmatch", "--", rel,
                       cwd=repo_root) is not None
        status = _git("status", "--porcelain", "--", rel, cwd=repo_root)
        commit = _git("log", "-1", "--format=%H", "--", rel, cwd=repo_root) or None
        when = (_git("log", "-1", "--format=%cI", "--", rel, cwd=repo_root) or None
                if commit else None)
        out.append({"path": rel, "sha256": PV.file_sha256(p), "tracked": tracked,
                    "clean": status == "", "commit": commit, "committed_at": when})
    return out


def read_release_log(root=DEFAULT_ROOT) -> List[dict]:
    path = Path(root) / RELEASE_LOG
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def released_items() -> Tuple[str, ...]:
    return tuple(i.name for i in ITEMS if i.requires_release)


def release_held_out_anchors(predictions: Sequence, *, root=DEFAULT_ROOT,
                             note: Optional[str] = None,
                             repo_root=REPO_ROOT) -> dict:
    """
    The release step: every predictions file is tracked and unchanged since
    its last commit, and no output of a released item exists yet; the release
    is appended to the log under ``root`` and returned.
    """
    if not predictions:
        raise CalibrationError("name the committed held-out predictions "
                               "(--predictions FILE ...)")
    state = predictions_state(predictions, repo_root)
    bad = [s["path"] for s in state if not (s["tracked"] and s["clean"]
                                            and s["commit"])]
    if bad:
        raise CalibrationError(
            f"the held-out predictions must be committed and unchanged before the "
            f"release: {bad}")
    for name in released_items():
        # the item's own directory and its tagged re-runs (<item>@<tag>)
        dirs = [d for d in sorted(Path(root).glob(f"{name}*"))
                if d.is_dir() and (d.name == name or d.name.startswith(name + "@"))]
        existing = [str(p) for d in dirs for p in sorted(d.glob("**/" + RB.RESULTS_JSONL))]
        if any(Path(p).stat().st_size for p in existing):
            raise CalibrationError(f"outputs of {name} exist before the release: "
                                   f"{existing[:3]}")
    head = _git("rev-parse", "HEAD", cwd=repo_root)
    body = {"schema": RELEASE_SCHEMA, "items": list(released_items()),
            "predictions": state, "head": head, "note": note,
            "released_unix": round(time.time(), 3)}
    rid = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    entry = {"release_id": rid[:16], **body}
    log = Path(root) / RELEASE_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def current_release(root=DEFAULT_ROOT, repo_root=REPO_ROOT) -> dict:
    """The latest release of the log whose predictions files are still
    unchanged (SHA-256 and clean); refused otherwise."""
    log = read_release_log(root)
    if not log:
        raise CalibrationError(
            "no held-out release: run release-held-out-anchors --predictions FILE "
            "... once the held-out predictions are committed")
    entry = log[-1]
    now = {s["path"]: s for s in predictions_state(
        [s["path"] for s in entry["predictions"]], repo_root)}
    changed = [s["path"] for s in entry["predictions"]
               if now[s["path"]]["sha256"] != s["sha256"]
               or not now[s["path"]]["clean"]]
    if changed:
        raise CalibrationError(f"held-out predictions changed after release "
                               f"{entry['release_id']}: {changed}")
    return entry


# --------------------------------------------------------------------------
# constants (CD-1)
# --------------------------------------------------------------------------
def constants() -> dict:
    """The deterministic constants of CD-1: ``tau_c`` and the lag sets of
    every bench substrate from the generator constants, the human-EEG
    default, and the family-B exact targets of every cell, the anchors and
    the strict monotonicity of the HCv2-12 (a) sweeps (no estimator)."""
    from impact_pipeline.bench import whole_brain as wb
    from impact_pipeline.bench.designs_v2 import family_b as FB
    from impact_pipeline.bench.generators import AgentConfig
    from impact_pipeline.v2 import declared_inputs as DI
    from scripts.v2 import iim_validation_v2 as V

    agent = dataclasses.asdict(AgentConfig())
    hopf = dataclasses.asdict(wb.WholeBrainConfig())
    substrates = {
        "family_A": ({"dynamics": "rate", "config": agent}, agent["dt"]),
        "family_C1": ({"dynamics": "stuart_landau", "config": agent}, agent["dt"]),
        "hopf_source": ({"dynamics": "hopf_whole_brain", "whole_brain_config": hopf},
                        1.0 / hopf["fs_out"]),
        "hopf_bold": ({"dynamics": "hopf_whole_brain", "whole_brain_config": hopf},
                      2.0),
        "null_calibration": ({"family": "null_calibration"}, agent["dt"]),
    }
    lag_plans = {}
    for name, (meta, dt) in substrates.items():
        ts = DI.coupling_timescale(meta)
        lag_plans[name] = {"tau_c": dataclasses.asdict(ts),
                           "lag_plan": DI.lag_plan(meta, dt=dt).to_dict()}
    for fs in (250.0, 500.0):
        lag_plans[f"human_eeg_{int(fs)}Hz"] = {
            "tau_c": {"tau_c_sec": DI.TAU_C_DEFAULT_SEC, "source": "human-EEG default"},
            "lag_plan": DI.lag_plan(tau_c=DI.TAU_C_DEFAULT_SEC, dt=1.0 / fs).to_dict()}
    exact = V.exact_table().to_dict(orient="records")
    ring = {}
    for c in FB.NON_MONOTONE_RING:
        ring[str(c)] = {cut: FB._exact("ring", (("coupling", float(c)),), cut, "tpm")
                        for cut in FB.CUT_MODES}
    return {
        "schema": "mpc-bench-dev-constants/1",
        "lag_plans": lag_plans,
        "family_b": {
            "anchors": {a: {cut: FB.anchor_value(cut, a) for cut in FB.CUT_MODES}
                        for a in FB.ANCHORS},
            "non_monotone_ring": ring,
            "sweep_monotonicity": FB.sweep_monotonicity(),
            "exact_table": exact,
        },
        "tier_b": "the oracle reciprocal gain of the hub-identity test (Tier B) is "
                  "frozen with its own package",
    }


def strict_json(obj):
    """``obj`` with every non-finite float (NaN, +-inf) replaced by None, so
    that :data:`CONSTANTS_JSON` is strict JSON (null, not the ``NaN``
    literal that strict parsers refuse); every other value is unchanged."""
    import numpy as np

    if isinstance(obj, Mapping):
        return {k: strict_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return obj if math.isfinite(float(obj)) else None
    return obj


def write_constants(path: Path, payload: Mapping) -> None:
    """Write the constants as strict JSON (:func:`strict_json`)."""
    Path(path).write_text(
        json.dumps(strict_json(payload), indent=2, sort_keys=True, default=float,
                   allow_nan=False) + "\n",
        encoding="utf-8")


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------
def _settings(run: Run) -> "RB.RunSettings":
    return RB.RunSettings(principles=run.principles)


def _provenance(item: Item, run: Run, *, n_tasks, seeds, settings, extra=None,
                argv=None, release=None) -> dict:
    return {
        "schema": PROVENANCE_SCHEMA,
        "calibration_version": CALIBRATION_VERSION,
        "item": item.name,
        "decisions": list(item.decisions),
        "run": run.name,
        "kind": run.kind,
        "designs": list(run.designs),
        "purpose": run.purpose,
        "n_tasks": int(n_tasks),
        "seeds": sorted({int(s) for s in seeds}) if seeds else [],
        "settings": settings,
        "estimator_options": json.loads(json.dumps(run.estimator_options)),
        "calibration_pending": RB.calibration_pending(),
        "iim_max_macro_nodes": RB.DEFAULT_SETTINGS.iim_max_macro_nodes,
        "release": None if release is None else {
            "release_id": release["release_id"],
            "predictions": release["predictions"]},
        "code": PV.code_identity(REPO_ROOT),
        "environment": PV.environment_versions(),
        "argv": list(argv or sys.argv),
        **(extra or {}),
    }


def _write_provenance(out: Path, prov: dict, started: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / PROVENANCE_JSON
    history = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8")).get("history", [])
        except ValueError:
            history = []
    entry = {"started_unix": round(started, 3), "finished_unix": round(time.time(), 3),
             "outcome": prov.get("outcome")}
    prov = dict(prov, history=history + [entry])
    path.write_text(json.dumps(prov, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def run_item(name: str, *, root=DEFAULT_ROOT, workers: int = 1, seeds=None,
             systems=None, limit: Optional[int] = None, protocol_dir=None,
             tag: Optional[str] = None, resume: bool = True, argv=None) -> dict:
    """Run (or resume) every run of an item; returns ``{run: summary}``.
    Under a ``seeds`` or ``systems`` restriction a run without tasks is
    skipped (``{"skipped": why}``, nothing written); an item none of whose
    runs has a task is refused."""
    item = get_item(name)
    release = current_release(root) if item.requires_release else None
    restricted = seeds is not None or systems is not None
    out = {}
    for run in item.runs:
        try:
            out[run.name] = run_one(item, run, root=root, workers=workers, seeds=seeds,
                                    systems=systems, limit=limit,
                                    protocol_dir=protocol_dir, tag=tag, resume=resume,
                                    argv=argv, release=release)
        except NoTasks as exc:
            if not restricted:
                raise
            out[run.name] = {"skipped": str(exc)}
    if out and all("skipped" in v for v in out.values()):
        raise NoTasks(f"item {name}: no run has a task at the requested seeds and "
                      "systems")
    return out


def run_one(item: Item, run: Run, *, root=DEFAULT_ROOT, workers: int = 1, seeds=None,
            systems=None, limit=None, protocol_dir=None, tag=None, resume=True,
            argv=None, release=None) -> dict:
    out = run_dir(root, item.name, run.name, tag)
    started = time.time()
    if seeds is not None:
        check_seed_list(list(seeds))
    if run.kind == CONSTANTS:
        out.mkdir(parents=True, exist_ok=True)
        write_constants(out / CONSTANTS_JSON, constants())
        prov = _provenance(item, run, n_tasks=0, seeds=(), settings=None, argv=argv)
        prov["outcome"] = {"ok": True, "file": CONSTANTS_JSON}
        _write_provenance(out, prov, started)
        return prov["outcome"]
    if run.kind == MANIPULATION:
        sd = list(seeds if seeds is not None else run.seeds)
        if limit is not None:
            sd = sd[: int(limit)]
        check_seed_list(sd)
        man = RB.run_manipulation(sd, out, families=ORACLE_FAMILIES)
        prov = _provenance(item, run, n_tasks=len(sd), seeds=sd, settings=None,
                           argv=argv)
        prov["outcome"] = {"complete": man["complete"], "all_usable": man["all_usable"]}
        _write_provenance(out, prov, started)
        return prov["outcome"]
    if run.kind == FAMILY_B:
        from scripts.v2 import iim_validation_v2 as V

        override = dict(run.estimator_options.get("IIM") or {})
        tasks, dropped = family_b_tasks(seeds, override.get("n_min"))
        if limit is not None:
            tasks = tasks[: int(limit)]
        if not tasks:
            raise NoTasks(f"run {run.name}: no family-B task at these seeds")
        check_seed_list([t.seed for t in tasks])
        res = V.run(out, split=DEVELOPMENT, workers=workers, task_list=tasks,
                    params_override=override or None)
        meta = res["summary"]
        prov = _provenance(item, run, n_tasks=len(tasks), seeds=[t.seed for t in tasks],
                           settings=None, argv=argv,
                           extra={"held_out_cells_left_out": dropped})
        prov["outcome"] = {"n_run": meta["n_run"], "n_records": meta["n_records"],
                           "n_task_errors": meta["n_task_errors"]}
        _write_provenance(out, prov, started)
        return prov["outcome"]
    if run.kind == EVALUATE:
        return run_evaluation(item, run, root=root, out=out, protocol_dir=protocol_dir,
                              argv=argv, started=started)
    withheld: List[str] = []
    tasks = runner_tasks(run, seeds=seeds, systems=systems, release=release,
                         withheld=withheld)
    if limit is not None:
        tasks = tasks[: int(limit)]
        kept = {t.task_id for t in tasks}
        withheld = [w for w in withheld if w.split("/", 1)[0] in kept]
    settings = _settings(run)
    overrides = protocol_overrides(tasks, run.estimator_options, protocol_dir)
    prov = _provenance(item, run, n_tasks=len(tasks), seeds=[t.seed for t in tasks],
                       settings=settings.to_dict(), argv=argv, release=release,
                       extra={"protocol_dir": None if protocol_dir is None
                              else str(protocol_dir),
                              "protocol_overrides": {k: v.hash
                                                     for k, v in overrides.items()},
                              "held_out_iim_withheld": {
                                  "rule": "HO-6: IIM on the v2 sensor views "
                                          f"{list(HO6_VIEWS)} of the Hopf arm only "
                                          "at G = 0 in the development purposes",
                                  "n_scorings": len(withheld)}})
    man = RB.run_tasks(tasks, out, n_workers=int(workers), settings=settings,
                       resume=resume, protocol_dir=protocol_dir,
                       protocol_overrides=overrides or None,
                       label=f"dev-calibration:{item.name}/{run.name}")
    leaked = smoke_records(out, tasks)
    if leaked:  # pragma: no cover - the runner discards smoke outputs
        raise CalibrationError(f"smoke outputs were recorded: {leaked[:3]}")
    prov["outcome"] = {"n_run": man["n_run"], "n_errors": man["n_errors"],
                       "n_component_errors": man["n_component_errors"],
                       "n_smoke": man["n_smoke"],
                       "plan_ok": man["plan_differences"]["ok"],
                       "curtailed": len(man["curtailed_task_ids"])}
    _write_provenance(out, prov, started)
    return prov["outcome"]


def check_seed_list(seeds: Sequence[int]) -> None:
    high = sorted({int(s) for s in seeds if int(s) > S.DEV_SEED_MAX})
    if high:
        raise CalibrationError(
            f"the development calibration uses seeds 0-{S.DEV_SEED_MAX} only; refused "
            f"seeds {high[:5]}")
    try:
        S.check_seeds(seeds, DEVELOPMENT)
    except S.SeedPolicyError as exc:
        raise CalibrationError(str(exc)) from None


def smoke_records(out: Path, tasks: Sequence[D.TaskSpec]) -> List[str]:
    """Smoke task ids that have a record in the results (none may)."""
    smoke = {t.task_id for t in tasks if t.smoke}
    if not smoke:
        return []
    latest, _ = RB.read_results(Path(out) / RB.RESULTS_JSONL)
    return sorted(smoke & set(latest))


def result_files(root=DEFAULT_ROOT, items: Optional[Sequence[str]] = None,
                 tagged: bool = True) -> List[Path]:
    """The record files of the calibration outputs (runner results and the
    family-B validation), optionally of some items only and without the
    tagged re-runs (``<item>@<tag>``)."""
    root = Path(root)
    out = []
    for name in (FAMILY_B_JSONL, RB.RESULTS_JSONL):
        for p in sorted(root.glob(f"**/{name}")):
            head = p.relative_to(root).parts[0]
            item = head.split("@", 1)[0]
            if not tagged and "@" in head:
                continue
            if items is None or item in items:
                out.append(p)
    return sorted(out)


def rejudged_records(files: Sequence, protocols: Mapping) -> Tuple[list, dict]:
    """The development records of ``files`` re-judged under ``protocols``
    (the generated ones): the development runs were judged under the draft
    protocols, whose anchors are pending, so their stored statuses are
    ``UNDEFINED(INVALID_ANCHORS)``; every component and verdict of a scoring
    whose protocol is generated is judged again from its stored estimator
    output, as the runner would have judged it (the family-B records keep
    their statuses: their protocols are fixed by the exact anchors).
    Returns the records and a summary."""
    from scripts.v2 import build_protocols_v2 as BP

    out, n_rejudged, failed, missing = [], 0, [], set()
    for path in files:
        latest, _ = RB.read_results(path)
        for tid in sorted(latest):
            rec = latest[tid]
            if rec.design == "family_b":
                out.append(rec)
                continue
            missing.update(s.protocol_id for s in rec.scorings
                           if s.protocol_id not in protocols)
            try:
                new = BP.rejudge_record(rec, protocols)
            except Exception as exc:  # noqa: BLE001 - reported, the record kept
                failed.append(f"{tid}: {type(exc).__name__}: {exc}")
                out.append(rec)
                continue
            n_rejudged += 1
            out.append(new)
    return out, {"n_records": len(out), "n_rejudged": n_rejudged,
                 "not_rejudged": failed[:20], "n_not_rejudged": len(failed),
                 "protocols_missing": sorted(missing)}


def run_evaluation(item: Item, run: Run, *, root, out: Path, protocol_dir, argv,
                   started) -> dict:
    """The evaluator in development mode on the calibration records of the
    items it depends on (untagged runs; the alternative-SE twin runs and the
    battery are not hypothesis data), re-judged under the generated
    protocols (:func:`rejudged_records`), with the generated mechanism-on
    table."""
    from impact_pipeline.v2 import hypothesis_engine as HE
    from scripts.v2 import build_protocols_v2 as BP

    files = [str(p) for p in result_files(root, items=item.after, tagged=False)]
    manip = [str(p) for p in sorted((Path(root) / "oracle_checks").glob("**/*.csv"))]
    proto = Path(protocol_dir or RB.GENERATED_DIR)
    protocols = dict(HE.load_protocol_files([proto])) if proto.is_dir() else {}
    if "A-R" not in protocols:
        raise CalibrationError(f"{proto}: no generated protocols; the development "
                               "evaluation runs after build_protocols_v2.py build")
    if not files:
        raise CalibrationError(f"no development records of {list(item.after)} under "
                               f"{root}")
    try:  # the builder's development policy (seeds, smoke, held-out outputs)
        BP.DevData(root, files=files)
    except BP.BuildError as exc:
        raise CalibrationError(f"development evaluation refused: {exc}") from None
    out.mkdir(parents=True, exist_ok=True)
    recs, summary = rejudged_records(files, protocols)
    records = out / EVALUATION_RECORDS
    REC.write_jsonl(recs, records)
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "bench_hypotheses_v2.py"),
           "--development", "--out", str(out), "--protocols", str(proto),
           "--records", str(records)]
    mech = proto / BP.MECHANISM_ON
    if mech.is_file():
        cmd += ["--mechanism-on", str(mech)]
    if manip:
        cmd += ["--manipulation", *manip]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    prov = _provenance(item, run, n_tasks=0, seeds=(), settings=None, argv=argv,
                       extra={"command": cmd, "record_files": files,
                              "rejudging": summary})
    prov["outcome"] = {"returncode": proc.returncode,
                       "n_rejudged": summary["n_rejudged"],
                       "n_not_rejudged": summary["n_not_rejudged"],
                       "stderr_tail": proc.stderr[-2000:]}
    _write_provenance(out, prov, started)
    return prov["outcome"]


# --------------------------------------------------------------------------
# plan and status
# --------------------------------------------------------------------------
def _task_cost(task_design: str, principles) -> Optional[float]:
    if principles:
        key = (task_design, tuple(principles))
        if key in RESTRICTED_CPU_S_PER_TASK:
            return RESTRICTED_CPU_S_PER_TASK[key]
    return CPU_S_PER_TASK.get(task_design)


def plan_run(run: Run) -> dict:
    """Tasks and projected CPU seconds of one run (its full development plan)."""
    if run.kind == CONSTANTS:
        return {"n_tasks": 0, "cpu_s": 0.0, "by_design": {}}
    if run.kind == EVALUATE:
        return {"n_tasks": 0, "cpu_s": 60.0, "by_design": {}}
    if run.kind == MANIPULATION:
        n = len(run.seeds or ())
        return {"n_tasks": n, "cpu_s": n * ORACLE_CPU_S_PER_SEED,
                "by_design": {"manipulation": n}}
    if run.kind == FAMILY_B:
        tasks, dropped = family_b_tasks(
            n_min=(run.estimator_options.get("IIM") or {}).get("n_min"))
        return {"n_tasks": len(tasks), "cpu_s": len(tasks) * CPU_S_PER_TASK["family_b"],
                "by_design": {"family_b": len(tasks)},
                "held_out_cells_left_out": dropped}
    withheld: List[str] = []
    if run.released:
        tasks = RB.build_tasks(list(run.designs), DEVELOPMENT, purpose=run.purpose)
    else:
        tasks = runner_tasks(run, withheld=withheld)
    by_design: Dict[str, int] = {}
    cpu, unknown = 0.0, set()
    for t in tasks:
        by_design[t.design] = by_design.get(t.design, 0) + 1
        if t.smoke:
            c = CPU_S_PER_TASK.get("A_heldout")
        else:
            c = _task_cost(t.design, run.principles)
        if c is None:
            unknown.add(t.design)
            continue
        cpu += c
    out = {"n_tasks": len(tasks), "cpu_s": cpu, "by_design": by_design,
           "n_smoke": sum(t.smoke for t in tasks)}
    if withheld:
        # the cost model prices a Hopf task with every scoring: an upper bound
        out["held_out_iim_withheld"] = len(withheld)
    if run.kind == BATTERY:
        # the runs of each admission route (one cell per substrate, stage,
        # view and bearer; the battery is one cell)
        out["content_on_runs"] = sum(t.tags.get("content") == CONTENT_ON for t in tasks)
        out["no_content_runs"] = sum(t.tags.get("content") == CONTENT_OFF
                                     for t in tasks)
    if unknown:
        out["unpriced_designs"] = sorted(unknown)
    return out


def plan(items: Optional[Sequence[str]] = None, workers: int = DEFAULT_WORKERS) -> dict:
    """Every item (or the named ones) with its runs, tasks and projected CPU
    hours, and the totals of the planned and the optional items."""
    names = list(items) if items else list(ITEM_NAMES)
    rows, totals = [], {"planned": [0, 0.0], "optional": [0, 0.0]}
    for name in names:
        it = get_item(name)
        runs = []
        for run in it.runs:
            p = plan_run(run)
            runs.append({"run": run.name, "kind": run.kind,
                         "designs": list(run.designs), "purpose": run.purpose,
                         "principles": None if run.principles is None
                         else list(run.principles), **p,
                         "cpu_h": round(p["cpu_s"] / 3600.0, 3)})
        n = sum(r["n_tasks"] for r in runs)
        cpu_h = sum(r["cpu_s"] for r in runs) / 3600.0
        key = "optional" if it.optional else "planned"
        totals[key][0] += n
        totals[key][1] += cpu_h
        rows.append({
            "item": it.name, "decisions": list(it.decisions),
            "description": it.description, "optional": it.optional,
            "requires_release": it.requires_release, "after": list(it.after),
            "n_tasks": n, "cpu_h": round(cpu_h, 2),
            "wall_h": round(cpu_h * CONTENTION / max(1, int(workers)), 2),
            "command": (f"python scripts/v2/dev_calibration.py run {it.name} "
                        f"--workers {int(workers)}"),
            "runs": runs,
        })
    return {
        "schema": "mpc-bench-dev-calibration-plan/1",
        "calibration_version": CALIBRATION_VERSION,
        "workers": int(workers),
        "contention": CONTENTION,
        "cost_source": COST_SOURCE,
        "items": rows,
        "totals": {k: {"n_tasks": v[0], "cpu_h": round(v[1], 2),
                       "wall_h": round(v[1] * CONTENTION / max(1, int(workers)), 2)}
                   for k, v in totals.items()},
    }


def status(root=DEFAULT_ROOT) -> List[dict]:
    """What exists under the root: per run directory the records, task
    errors and smoke status lines, and the provenance outcome."""
    root = Path(root)
    rows = []
    for prov_path in sorted(root.glob(f"**/{PROVENANCE_JSON}")):
        d = prov_path.parent
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        latest, _ = RB.read_results(d / RB.RESULTS_JSONL)
        fb = d / FAMILY_B_JSONL
        n_fb = (sum(1 for line in fb.read_text(encoding="utf-8").splitlines()
                    if line.strip()) if fb.exists() else 0)
        smoke = d / RB.SMOKE_JSONL
        n_smoke = (sum(1 for line in smoke.read_text(encoding="utf-8").splitlines()
                       if line.strip()) if smoke.exists() else 0)
        rows.append({
            "dir": str(d.relative_to(root)), "item": prov.get("item"),
            "run": prov.get("run"), "planned": prov.get("n_tasks"),
            "records": len(latest) + n_fb,
            "task_errors": sum(r.status == REC.TASK_ERROR for r in latest.values()),
            "smoke_lines": n_smoke, "outcome": prov.get("outcome"),
        })
    return rows


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def _ints(text) -> Optional[List[int]]:
    return None if text is None else RB._parse_ints(text)


def _strs(text) -> Optional[List[str]]:
    return None if text is None else RB._parse_list(text)


def _print_plan(p: dict) -> None:
    w = p["workers"]
    print(f"development calibration (seeds 0-999); {w} workers, wall = CPU-h x "
          f"{p['contention']} / {w}")
    print(f"cost model: {p['cost_source']}")
    for row in p["items"]:
        flag = " (optional)" if row["optional"] else ""
        rel = " [needs the held-out release]" if row["requires_release"] else ""
        print(f"\n{row['item']}{flag}{rel}  {', '.join(row['decisions']) or '-'}: "
              f"{row['n_tasks']} tasks, {row['cpu_h']:.2f} CPU-h, "
              f"{row['wall_h']:.2f} h wall")
        print(f"  {row['description']}")
        for r in row["runs"]:
            extra = (f" principles {','.join(r['principles'])}" if r["principles"]
                     else "")
            print(f"  - {r['run']:28s} {r['kind']:12s} tasks {r['n_tasks']:5d} "
                  f"{r['cpu_h']:7.2f} CPU-h{extra}")
            if r.get("held_out_iim_withheld"):
                print(f"    IIM left out of {r['held_out_iim_withheld']} Hopf "
                      "sensor-view scorings above G = 0 (held out, HO-6)")
            if "content_on_runs" in r:
                print(f"    content-on runs {r['content_on_runs']}, no-content runs "
                      f"{r['no_content_runs']} (the admission rule needs "
                      f"{CONCORDANCE_MIN_RUNS} per route and cell)")
        print(f"  command: {row['command']}")
    for k, v in p["totals"].items():
        print(f"\ntotal {k}: {v['n_tasks']} tasks, {v['cpu_h']:.2f} CPU-h, "
              f"{v['wall_h']:.2f} h wall")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dev_calibration.py",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    pl = sub.add_parser("plan", help="the items with tasks and projected CPU-h")
    pl.add_argument("--items", default=None)
    pl.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    pl.add_argument("--json", default=None)
    rn = sub.add_parser("run", help="run (or resume) calibration items")
    rn.add_argument("items")
    rn.add_argument("--root", default=str(DEFAULT_ROOT))
    rn.add_argument("--workers", type=int, default=1)
    rn.add_argument("--seeds", default=None)
    rn.add_argument("--systems", default=None)
    rn.add_argument("--limit", type=int, default=None)
    rn.add_argument("--protocol-dir", default=None)
    rn.add_argument("--tag", default=None)
    rn.add_argument("--no-resume", action="store_true")
    rl = sub.add_parser("release-held-out-anchors",
                        help="log the release of the held-out-regime reference "
                             "anchors once the held-out predictions are committed")
    rl.add_argument("--predictions", nargs="+", required=True)
    rl.add_argument("--note", default=None)
    rl.add_argument("--root", default=str(DEFAULT_ROOT))
    st = sub.add_parser("status", help="what the calibration root holds")
    st.add_argument("--root", default=str(DEFAULT_ROOT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            p = plan(_strs(args.items), args.workers)
            if args.json:
                Path(args.json).write_text(json.dumps(p, indent=1) + "\n",
                                           encoding="utf-8")
            _print_plan(p)
            return 0
        if args.command == "release-held-out-anchors":
            entry = release_held_out_anchors(args.predictions, root=args.root,
                                             note=args.note)
            print(f"held-out release {entry['release_id']} logged in "
                  f"{Path(args.root) / RELEASE_LOG}")
            return 0
        if args.command == "status":
            for r in status(args.root):
                print(json.dumps(r, sort_keys=True))
            return 0
        rc = 0
        for name in _strs(args.items) or []:
            res = run_item(name, root=args.root, workers=args.workers,
                           seeds=_ints(args.seeds), systems=_strs(args.systems),
                           limit=args.limit, protocol_dir=args.protocol_dir,
                           tag=args.tag, resume=not args.no_resume,
                           argv=["dev_calibration.py", *argv])
            for run, out in res.items():
                print(f"{name}/{run}: {json.dumps(out, sort_keys=True, default=str)}")
                if out.get("n_errors") or out.get("n_task_errors") or (
                        out.get("plan_ok") is False) or out.get("returncode"):
                    rc = 1
        return rc
    except (CalibrationError, RB.RunPolicyError, S.SeedPolicyError,
            D.DesignError) as exc:
        print(f"dev_calibration: refused: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:  # a design module's own policy error
        print(f"dev_calibration: refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
