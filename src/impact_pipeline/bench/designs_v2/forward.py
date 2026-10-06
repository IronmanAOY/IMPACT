"""
Forward-model arms of MPC-Bench v2: the task builders of the Hopf arm and of
the forward-modelled family-A arms, their views and scorings, and the
admission designs the registry v3 builder applies to their records.

Arms (design 3.2, 3.6)
----------------------
``hopf``
    the v1 whole-brain Hopf model (76 regions, 250 Hz). G sweep: the eight v1
    levels 0-4 x 20 seeds; hub lesion and its size-matched random-edge
    lesion at ``G_nom = 1.142857`` (a sweep level) x 20; ``G = 0`` extended
    to 61 seeds and ``G_nom`` to 150 (371 simulations). Views: ``source``,
    ``eeg64``, ``eeg64_noref``, ``eeglow``, ``mne_template`` (first 60 s of
    the run) and ``bold`` (the whole run of 600 s, the v1 BOLD duration of
    300 volumes at TR 2 s). One simulation serves every view: its first
    60 s are, sample for sample, the 60-s simulation of the same seed
    (:func:`truncate_whole_brain`). Principles NAS and IIM; no inputs.
``forward_a_eeg``
    family-A agents (v1 routing) through the EEG-like model at 20 Hz:
    PC_nominal (K = 6) x 150, N_ar1 x 61, W_PDI_no_multistability x 61 and
    K in {1, 2, 3} x 20 (332 simulations). The v1 1-40 Hz band is above
    Nyquist at 20 Hz, so these views are recorded without a sensor band
    (as in development). Views ``source``, ``eeg64``, ``eeglow``. PDI
    (full-bearer upper bound on sensors), IIM descriptive; declaration R.
``forward_a_bold``
    slow-context agents (catalogue preset ``slow_context_bold``: context
    dwell 30-60 s, trial 12 s, ITI jitter 2 s) through the BOLD-like model
    (TR 2 s, drive = the rate signal; PDI window 5 TR): PC_nominal up to
    150 (curtailed), W_PDI_no_multistability x 61, K in {1, 2, 3} x 20
    (<= 271 simulations). Views ``source``, ``bold``. PDI on the content
    bearer, IIM descriptive; declaration R.

Purposes and regimes
--------------------
``confirmatory``: seeds from 20000, the held-out regime (lead-field seed
20261001, width 0.6). ``replication``: the anchor conditions on the
confirmatory replication block 20900-20919 at the held-out regime (the
forward views' anchor replication, HCv2-6). ``dry_run`` (seeds 384-399) and
``dev_regime``
(804-819): the same conditions at about 15 % scale, at the development
regime. ``reference`` (900-939): the anchor conditions (``G_nom``,
PC_nominal) at the held-out regime, computed only after the held-out
predictions are committed; ``reference_development``: the same seeds at the
development regime. ``smoke`` (980-984): the anchor conditions at the
held-out regime, outputs discarded unread. No other development task may
use the held-out regime (:func:`check_regime_policy`).

Record contract (what the registry builder and the hypotheses read)
-------------------------------------------------------------------
The v2 runner runs these tasks (:func:`runner_task`; designs
``whole_brain``, ``forward_family_a``, ``forward_family_a_bold`` and
``forward_anchor_replication``). A task record (``mpc-bench-result/3``)
keeps the forward task's id, carries the design of its arm
(:data:`RECORD_DESIGN_OF_ARM`, the one table of these names: the
hypotheses select by them and the registry builder reads the arms' designs
only; the anchor runs are filed under :data:`ANCHOR_REPLICATION_DESIGN`)
and stores :meth:`ForwardTask.to_config` in ``config['forward']`` (arm,
condition and its spec, dose, purpose, regime and its parameters); each
scoring has ``view`` = the view name, ``estimator_form`` from
:func:`scoring_plan`, the view's family protocol
(:func:`protocol_key_of`: ``hopf-eeg64``, ``fwdA-eeglow``,
``fwdA_bold-bold``, ``hopf-eeg64+iim_v1_quadrants``) and ``details`` with
:func:`impact_pipeline.bench.forward_v2.scoring_details` of the view (its
registry regime keys). Components are scored under the view's family
protocol without a registry (the admission run decides the registry);
sensor and source-estimate scorings carry the observation-gate label
``admission_run`` (NAS N10, IIM C7), which the runner maps onto NAS v3's
``override_observation_gate`` and IIM v5's ``observation_admitted``, and
IIM's ``macro_nodes`` option picks the rank-safe clusters or the v1
quadrants of a sensor view.

A run carries what some criterion, hypothesis or anchor reads on it
(:func:`scoring_plan`): the v1 quadrant comparator only at ``G = 0``
(HCv2-12(d), HCv2-15(b)); on the lesion conditions only NAS (FMb2 is the
only lesion criterion); the source view of a family-A arm, which is no
admission view, only on the runs whose source contrast FMd reads; and IIM
on the family-A arms, which is descriptive (design 3.1), as its value and
null without the bootstrap SE that only a status needs. Every forward
protocol scores IIM's primary (directional) cut only (``report_cut_modes``
empty): no forward scoring reports another cut.

Curtailment (BOLD arm): the on-runs PC_nominal and K in {2, 3} are judged
for FMabs in seed order (:func:`curtailment_order`); once the demonstration
has become impossible the runner skips only the :func:`curtailable` tasks,
the PC_nominal runs that no other criterion reads, so FMa, FMb1 and FMd keep
their planned runs.

Options in :func:`scoring_plan` use the protocol vocabulary (for example
``pdi_bearer``); the runner maps them onto the estimators' parameter names.
``n_low`` (the low-density montage) is a parameter of every builder with the
default :data:`impact_pipeline.bench.forward_v2.N_LOW_DEFAULT` (32); the
development calibration replaces it by the paper-2 value (CD-12).

This module must not import ``impact_pipeline.mpc_metrics``.
"""

from __future__ import annotations

import copy
import functools
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline.bench import forward_v2 as F2
from impact_pipeline.bench.generators import BenchSystem
from impact_pipeline.v2 import registry_v3 as RV
from impact_pipeline.v2 import seeds as S

DESIGN = "forward"
ARM_HOPF = "hopf"
ARM_A_EEG = "forward_a_eeg"
ARM_A_BOLD = "forward_a_bold"
ARMS = (ARM_HOPF, ARM_A_EEG, ARM_A_BOLD)

CONFIRMATORY = "confirmatory"
REPLICATION = "replication"
DRY_RUN = "dry_run"
DEV_REGIME = "dev_regime"
REFERENCE = "reference"
REFERENCE_DEVELOPMENT = "reference_development"
SMOKE = "smoke"
PURPOSES = (CONFIRMATORY, REPLICATION, DRY_RUN, DEV_REGIME, REFERENCE,
            REFERENCE_DEVELOPMENT, SMOKE)
HELD_OUT_PURPOSES = (CONFIRMATORY, REPLICATION, REFERENCE, SMOKE)
# purposes on confirmatory seeds: the admission runs and the anchor
# replication blocks
CONFIRMATORY_PURPOSES = (CONFIRMATORY, REPLICATION)
# purposes whose records an admission reads (and whose on-runs may be
# curtailed); reference and smoke runs only anchor or test the plumbing
ADMISSION_PURPOSES = (CONFIRMATORY, DRY_RUN, DEV_REGIME)
DEVELOPMENT_SEED_BLOCKS = MappingProxyType({
    DRY_RUN: range(384, 400),
    DEV_REGIME: range(804, 820),
    REFERENCE: S.REFERENCE_BLOCK,
    REFERENCE_DEVELOPMENT: S.REFERENCE_BLOCK,
    SMOKE: S.SMOKE_SEEDS,
})
DRY_RUN_FRACTION = 0.15
CONFIRMATORY_SEED_BASE = S.CONFIRMATORY_SEED_MIN
# the anchor replication block of the forward views (HCv2-6; seed map 3.7)
REPLICATION_SEEDS = range(20900, 20920)

HOPF_SOURCE_DURATION_S = 60.0
# v1 run_bench.BOLD_MIN_DURATION_SEC: 300 volumes at TR 2 s
HOPF_BOLD_DURATION_S = 600.0
ADMISSION_RUN = "admission_run"

VIEWS_OF_ARM = MappingProxyType({
    ARM_HOPF: ("source", "eeg64", "eeg64_noref", "eeglow", "mne_template", "bold"),
    ARM_A_EEG: ("source", "eeg64", "eeglow"),
    ARM_A_BOLD: ("source", "bold"),
})
DECLARATION_OF_ARM = MappingProxyType({
    ARM_HOPF: "none", ARM_A_EEG: "R", ARM_A_BOLD: "R"})
INPUTS_DECLARED_OF_ARM = MappingProxyType({
    ARM_HOPF: "none", ARM_A_EEG: "complete", ARM_A_BOLD: "complete"})
BOLD_DRIVE_OF_ARM = MappingProxyType({ARM_HOPF: "envelope", ARM_A_BOLD: "signal"})
# sensor band of the arm's EEG views (None: no band-pass at 20 Hz)
SENSOR_BAND_OF_ARM = MappingProxyType({ARM_HOPF: (1.0, 40.0), ARM_A_EEG: None})
FAMILY_OF_ARM = MappingProxyType({ARM_HOPF: "whole_brain", ARM_A_EEG: "A",
                                  ARM_A_BOLD: "A"})
SLOW_CONTEXT_PRESET = "slow_context_bold"
# The curtailment group of the BOLD on-runs (PDI exclusion safety).
CURTAIL_GROUP_A_BOLD = "forward_a_bold_on"


class ForwardDesignError(ValueError):
    """An invalid forward task, purpose or regime."""


# --------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    """One condition of an arm: its label, confirmatory seed count, dose
    (G or K), what realises it and its curtailment group; ``anchor`` marks
    the reference condition (G_nom, PC_nominal)."""

    arm: str
    label: str
    n_confirmatory: int
    dose: Optional[float]
    spec: Mapping
    anchor: bool = False
    curtail_group: Optional[str] = None


def _hopf_conditions() -> Tuple[Condition, ...]:
    from impact_pipeline.bench import adversarial_v2 as A2
    from impact_pipeline.bench import whole_brain as wb

    g_nom = A2.g_nom()
    out = []
    for g in wb.g_sweep_levels():
        n = 61 if g == 0 else 150 if math.isclose(g, g_nom) else 20
        out.append(Condition(ARM_HOPF, f"hopf_G{g:g}", n, float(g),
                             MappingProxyType({"G": float(g), "lesion": "none"}),
                             anchor=math.isclose(g, g_nom)))
    for man in A2.lesion_conditions_g_nom():
        cfg = man["config"]
        out.append(Condition(ARM_HOPF, f"hopf_{man['name']}", 20, float(cfg.G),
                             MappingProxyType({"G": float(cfg.G), "lesion": cfg.lesion,
                                               "n_lesion_edges": cfg.n_lesion_edges})))
    return tuple(out)


def _family_a_conditions(arm: str) -> Tuple[Condition, ...]:
    bold = arm == ARM_A_BOLD
    preset = SLOW_CONTEXT_PRESET if bold else None
    on = CURTAIL_GROUP_A_BOLD if bold else None
    out = [Condition(arm, "PC_nominal", 150, 6.0,
                     MappingProxyType({"system": "PC_nominal", "preset": preset}),
                     anchor=True, curtail_group=on)]
    for k in (1, 2, 3):
        out.append(Condition(
            arm, f"PC_nominal_K{k}", 20, float(k),
            MappingProxyType({"system": "PC_nominal", "knobs": {"K": k},
                              "preset": preset}),
            curtail_group=(on if k > 1 else None)))
    out.append(Condition(arm, "W_PDI_no_multistability", 61, None,
                         MappingProxyType({"system": "W_PDI_no_multistability",
                                           "preset": preset})))
    if not bold:
        out.append(Condition(arm, "N_ar1", 61, None,
                             MappingProxyType({"system": "N_ar1", "preset": None})))
    return tuple(out)


_CONDITIONS_CACHE: Dict[str, Tuple[Condition, ...]] = {}


def conditions(arm: str) -> Tuple[Condition, ...]:
    """The conditions of an arm, in plan order."""
    if arm not in ARMS:
        raise ForwardDesignError(f"arm must be one of {ARMS}")
    if arm not in _CONDITIONS_CACHE:
        _CONDITIONS_CACHE[arm] = (_hopf_conditions() if arm == ARM_HOPF
                                  else _family_a_conditions(arm))
    return _CONDITIONS_CACHE[arm]


def condition(arm: str, label: str) -> Condition:
    for c in conditions(arm):
        if c.label == label:
            return c
    raise ForwardDesignError(f"arm {arm} has no condition {label!r}")


def g_nom_label() -> str:
    (c,) = [c for c in conditions(ARM_HOPF) if c.anchor]
    return c.label


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------
def regime_of_purpose(purpose: str) -> F2.ForwardRegime:
    if purpose not in PURPOSES:
        raise ForwardDesignError(f"purpose must be one of {PURPOSES}")
    return F2.HELD_OUT_REGIME if purpose in HELD_OUT_PURPOSES else F2.DEVELOPMENT_REGIME


def arm_regime(arm: str, purpose: str,
               n_low: int = F2.N_LOW_DEFAULT) -> F2.ForwardRegime:
    """The forward regime of an arm's tasks: the purpose's regime with the
    arm's sensor band and the low-density montage size."""
    reg = regime_of_purpose(purpose)
    band = SENSOR_BAND_OF_ARM.get(arm, reg.band)
    return reg.replace(band=band, n_low=int(n_low))


@dataclass(frozen=True)
class ForwardTask:
    """One simulation of a forward arm and the views scored on it."""

    task_id: str
    arm: str
    condition: str
    seed: int
    purpose: str
    regime: str
    views: Tuple[str, ...]
    dose: Optional[float]
    curtail_group: Optional[str]
    n_low: int
    design: str = DESIGN
    replicate: int = 0

    @property
    def family(self) -> str:
        return FAMILY_OF_ARM[self.arm]

    @property
    def system(self) -> str:
        return self.condition

    @property
    def split(self) -> str:
        return S.split_of(self.seed)

    @property
    def declaration_id(self) -> str:
        return DECLARATION_OF_ARM[self.arm]

    def to_dict(self) -> dict:
        out = asdict(self)
        out["views"] = list(self.views)
        return out

    def to_config(self) -> dict:
        """The ``config`` of the task record: the task and the regime
        parameters under ``forward``."""
        cond = condition(self.arm, self.condition)
        return {"forward": {
            **self.to_dict(),
            "condition_spec": _plain(cond.spec),
            "regime_parameters": arm_regime(self.arm, self.purpose,
                                            self.n_low).to_dict(),
        }}


def _plain(obj):
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _seeds(cond: Condition, purpose: str) -> List[int]:
    if purpose == CONFIRMATORY:
        return list(range(CONFIRMATORY_SEED_BASE, CONFIRMATORY_SEED_BASE
                          + cond.n_confirmatory))
    if purpose == REPLICATION:
        return list(REPLICATION_SEEDS) if cond.anchor else []
    block = list(DEVELOPMENT_SEED_BLOCKS[purpose])
    if purpose in (DRY_RUN, DEV_REGIME):
        n = min(len(block), int(math.ceil(DRY_RUN_FRACTION * cond.n_confirmatory)))
        return block[:n]
    return block if cond.anchor else []


def build_tasks(purpose: str = CONFIRMATORY, arms: Sequence[str] = ARMS,
                n_low: int = F2.N_LOW_DEFAULT) -> List[ForwardTask]:
    """The tasks of the forward arms for one purpose (module docstring), in
    plan order (arm, condition, seed)."""
    if purpose not in PURPOSES:
        raise ForwardDesignError(f"purpose must be one of {PURPOSES}")
    tasks = []
    for arm in arms:
        if arm not in ARMS:
            raise ForwardDesignError(f"arm must be one of {ARMS}")
        for cond in conditions(arm):
            for seed in _seeds(cond, purpose):
                tasks.append(_forward_task(arm, cond, seed, purpose, n_low))
    check_regime_policy(tasks)
    return tasks


def _forward_task(arm: str, cond: Condition, seed: int, purpose: str,
                  n_low: int) -> ForwardTask:
    regime = regime_of_purpose(purpose).name
    return ForwardTask(
        task_id=f"{DESIGN}-{arm}-{cond.label}-{regime}-s{int(seed):05d}",
        arm=arm, condition=cond.label, seed=int(seed), purpose=purpose,
        regime=regime, views=VIEWS_OF_ARM[arm], dose=cond.dose,
        curtail_group=(cond.curtail_group if purpose in ADMISSION_PURPOSES
                       else None),
        n_low=int(n_low))


def forward_task(arm: str, condition_label: str, seed: int, purpose: str,
                 n_low: int = F2.N_LOW_DEFAULT) -> ForwardTask:
    """The forward task of (arm, condition, seed, purpose), as
    :func:`build_tasks` builds it, checked against the purpose's seeds and
    the regime policy (the runner rebuilds its tasks from these keys)."""
    if purpose not in PURPOSES:
        raise ForwardDesignError(f"purpose must be one of {PURPOSES}")
    cond = condition(arm, condition_label)
    if int(seed) not in _seeds(cond, purpose):
        raise ForwardDesignError(f"seed {seed} is not a {purpose} seed of {arm} "
                                 f"{condition_label}")
    t = _forward_task(arm, cond, int(seed), purpose, n_low)
    check_regime_policy([t])
    return t


def check_regime_policy(tasks: Sequence[ForwardTask]) -> None:
    """
    The seed and regime policy of the forward arms: every task's seed obeys
    the v2 seed policy for its purpose (confirmatory seeds only for
    ``confirmatory``); the held-out regime appears on development seeds only
    on the reference block (900-939, anchors after the held-out predictions
    are committed) and the smoke seeds (980-984, outputs discarded); every
    confirmatory task is at the held-out regime, and the replication runs
    use the replication block 20900-20919.
    """
    for t in tasks:
        split = S.split_of(t.seed)
        if t.purpose not in PURPOSES:
            raise ForwardDesignError(f"{t.task_id}: unknown purpose {t.purpose!r}")
        if (t.purpose in CONFIRMATORY_PURPOSES) != (split == S.CONFIRMATORY):
            raise ForwardDesignError(
                f"{t.task_id}: purpose {t.purpose} on a {split} seed")
        if t.purpose == REPLICATION and t.seed not in REPLICATION_SEEDS:
            raise ForwardDesignError(f"{t.task_id}: seed {t.seed} is outside the "
                                     "replication block")
        if (t.purpose not in CONFIRMATORY_PURPOSES
                and t.seed not in DEVELOPMENT_SEED_BLOCKS[t.purpose]):
            raise ForwardDesignError(f"{t.task_id}: seed {t.seed} is outside the "
                                     f"{t.purpose} block")
        want = regime_of_purpose(t.purpose).name
        if t.regime != want:
            raise ForwardDesignError(
                f"{t.task_id}: purpose {t.purpose} runs at the {want} regime")
        if t.regime == F2.HELD_OUT_REGIME.name and split == S.DEVELOPMENT:
            if t.purpose not in (REFERENCE, SMOKE):
                raise ForwardDesignError(
                    f"{t.task_id}: the held-out regime on a development seed is "
                    "allowed for reference anchors and smoke tests only")
            if not condition(t.arm, t.condition).anchor:
                raise ForwardDesignError(
                    f"{t.task_id}: only the anchor conditions run at the held-out "
                    "regime before the freeze")


def curtailment_order(tasks: Sequence[ForwardTask]) -> Dict[str, List[ForwardTask]]:
    """
    The tasks of each curtailment group in evaluation order (seed, then the
    order of the arm's on-conditions). A runner feeds the group's false
    ABSENTs in this order to :class:`impact_pipeline.v2.registry_v3.
    Curtailment`; once it has stopped, the runner skips the group's
    remaining :func:`curtailable` tasks and still runs every other task,
    because FMb1 and FMd read them.
    """
    groups: Dict[str, List[ForwardTask]] = {}
    for t in tasks:
        if t.curtail_group:
            groups.setdefault(t.curtail_group, []).append(t)
    for g, ts in groups.items():
        rank = {c.label: i for i, c in enumerate(conditions(ts[0].arm))}
        ts.sort(key=lambda t: (t.seed, rank[t.condition]))
    return groups


@functools.lru_cache(maxsize=None)
def _seeds_other_criteria_read(arm: str, purpose: str) -> Mapping[str, frozenset]:
    """``{condition: seeds}`` of the runs that a criterion other than FMabs
    reads: FMa (every null run), FMb2 (every lesion run), and FMb1 and FMd
    (the dose and concordance conditions on the seeds common to every dose
    level)."""
    out: Dict[str, set] = {}
    for d in admission_designs(purpose):
        if d.arm != arm:
            continue
        common = frozenset.intersection(*(
            frozenset(_seeds(condition(arm, c), purpose)) for c in d.dose_conditions))
        for c in set(d.dose_conditions) | set(d.concordance):
            out.setdefault(c, set()).update(common)
        for c in tuple(d.null_conditions) + tuple(d.lesion or ()):
            out.setdefault(c, set()).update(_seeds(condition(arm, c), purpose))
    return MappingProxyType({c: frozenset(s) for c, s in out.items()})


def curtailable(task: ForwardTask) -> bool:
    """
    True iff a runner may skip ``task`` once its curtailment group has
    stopped: an on-run that only FMabs reads. On the BOLD arm these are the
    PC_nominal runs beyond the seeds of the K sweep (seeds 20020-20149 of
    the confirmatory plan); the K = 2 and K = 3 runs and the first
    PC_nominal seeds are always run, so curtailment changes the FMabs
    decision and nothing else.
    """
    if not task.curtail_group or task.purpose not in ADMISSION_PURPOSES:
        return False
    read = _seeds_other_criteria_read(task.arm, task.purpose)
    return task.seed not in read.get(task.condition, frozenset())


# --------------------------------------------------------------------------
# realisation
# --------------------------------------------------------------------------
def truncate_whole_brain(system: BenchSystem, duration_s: float) -> BenchSystem:
    """
    The first ``duration_s`` seconds of a whole-brain run as the system a
    run of that duration produces with the same seed: the series, envelope
    and order parameter cut, the order statistics recomputed and the
    configuration's duration and ``n_time`` updated (the Euler-Maruyama
    draws of the first samples do not depend on the run length).
    """
    if system.meta.get("dynamics") != "hopf_whole_brain":
        raise ForwardDesignError("only whole-brain runs are truncated")
    fs = float(system.meta["sampling_frequency"])
    n = int(round(float(duration_s) * fs))
    if not 1 <= n <= system.n_time:
        raise ForwardDesignError("duration outside the run")
    meta = copy.deepcopy(system.meta)
    meta["n_time"] = n
    meta["whole_brain_config"]["duration_sec"] = float(duration_s)
    oracle = dict(system.oracle)
    R = np.asarray(system.oracle["order_parameter"])[:n]
    oracle["order_parameter"] = R
    oracle["mean_order"] = float(R.mean())
    oracle["metastability"] = float(R.std())
    oracle["envelope"] = np.asarray(system.oracle["envelope"])[:, :n]
    return BenchSystem(ts=np.ascontiguousarray(system.ts[:, :n]),
                       events=system.events.copy(), meta=meta, oracle=oracle)


def _hopf_source(task: ForwardTask) -> Tuple[BenchSystem, BenchSystem]:
    from impact_pipeline.bench import whole_brain as wb

    spec = condition(task.arm, task.condition).spec
    duration = HOPF_BOLD_DURATION_S if "bold" in task.views else HOPF_SOURCE_DURATION_S
    cfg = wb.WholeBrainConfig().replace(
        G=float(spec["G"]), lesion=spec["lesion"],
        n_lesion_edges=spec.get("n_lesion_edges"), duration_sec=float(duration))
    run = wb.simulate_whole_brain(cfg, task.seed)
    src = (truncate_whole_brain(run, HOPF_SOURCE_DURATION_S)
           if duration > HOPF_SOURCE_DURATION_S else run)
    return src, run


def _family_a_source(task: ForwardTask) -> BenchSystem:
    from impact_pipeline.bench import adversarial_v2 as A2
    from impact_pipeline.bench import generators as g

    spec = condition(task.arm, task.condition).spec
    preset = spec.get("preset")
    knobs = spec.get("knobs")
    if knobs:
        cfg = g.config_from_dict(A2.config_preset(preset)) if preset else None
        system = g.make_system("family_a", g.knobs_from_dict(dict(knobs)), cfg,
                               task.seed)
    else:
        system = A2.build_system(spec["system"], task.seed, "A", preset=preset)
    # the source is the bench's own system, meta included (the condition is
    # in the record), so the duplicate detector sees one configuration where
    # a forward run and a joint-bench run share a seed (PC_nominal on the
    # witness and anchor seeds)
    return system


@dataclass(frozen=True)
class Realisation:
    """A task's source system (the analysed source run) and its views."""

    task: ForwardTask
    source: BenchSystem
    views: Mapping[str, BenchSystem]


def realise(task: ForwardTask) -> Realisation:
    """Simulate the task's source and build its views at the task's regime
    (:func:`impact_pipeline.bench.forward_v2.observe`)."""
    check_regime_policy([task])
    regime = arm_regime(task.arm, task.purpose, task.n_low)
    if task.arm == ARM_HOPF:
        src, run = _hopf_source(task)
        views = F2.observe(src, task.views, regime, task.seed, bold_source=run,
                           bold_drive=BOLD_DRIVE_OF_ARM[task.arm],
                           inputs_declared=INPUTS_DECLARED_OF_ARM[task.arm])
    else:
        src = _family_a_source(task)
        views = F2.observe(src, task.views, regime, task.seed,
                           bold_drive=BOLD_DRIVE_OF_ARM.get(task.arm, "envelope"),
                           inputs_declared=INPUTS_DECLARED_OF_ARM[task.arm])
    return Realisation(task, src, MappingProxyType(dict(views)))


# --------------------------------------------------------------------------
# scorings
# --------------------------------------------------------------------------
PRIMARY = "primary"
IIM_V1_QUADRANTS = "iim_v1_quadrants"
ROLE_ADMISSION = "admission"
ROLE_DESCRIPTIVE = "descriptive"
ROLE_COMPARATOR = "comparator"
# the source run of a family-A arm: the contrast FMd compares the views with
ROLE_SOURCE_CONTRAST = "source_contrast"

_GATE = {"observation_gate": ADMISSION_RUN}
_IIM_SENSOR = {"preprocess": "zca", "macro_nodes": "rank_safe_clusters", **_GATE}
# The source-estimate view is an EEG substrate: the v2 pipeline (rank
# condition and ZCA) on the source model's declared macro nodes (HCv2-15).
_IIM_SOURCE_ESTIMATE = {"preprocess": "zca", **_GATE}
_IIM_V1 = {"preprocess": "none", "macro_nodes": "v1_quadrants", **_GATE}
# IIM on the family-A arms is descriptive (design 3.1): its value and null,
# without the bootstrap SE that only a status needs (a status is never read)
_IIM_DESCRIPTIVE = {"se_method": None}
# The forward views score the primary (directional) cut only: no admission
# criterion, hypothesis or form scoring reads a reported cut there, and the
# reported bidirectional cut costs a third of every IIM statistic
_IIM_PRIMARY_CUT = {"report_cut_modes": []}
# Conditions on which the v1 quadrant comparator is read: G = 0 (HCv2-12(d)
# under the average reference, HCv2-15(b) without a reference).
COMPARATOR_CONDITIONS = MappingProxyType({ARM_HOPF: ("hopf_G0",)})
# Principles scored on the lesion conditions of the Hopf arm: only NAS has a
# lesion criterion (FMb2); IIM's admission reads no lesion run.
LESION_PRINCIPLES_OF_ARM = MappingProxyType({ARM_HOPF: ("NAS",)})


def _scorings(arm: str, view: str) -> List[dict]:
    stage = F2.view_spec(view).stage
    mixed = stage in ("sensor", "source_estimate")
    out = []
    if arm == ARM_HOPF:
        principles = ("IIM",) if view == "eeg64_noref" else ("NAS", "IIM")
        opts = {"IIM": dict(_IIM_PRIMARY_CUT)}
        if mixed:
            iim = _IIM_SENSOR if stage == "sensor" else _IIM_SOURCE_ESTIMATE
            opts = {"NAS": dict(_GATE), "IIM": dict(iim, **_IIM_PRIMARY_CUT)}
            opts = {p: o for p, o in opts.items() if p in principles}
        out.append({"estimator_form": PRIMARY, "principles": principles,
                    "options": opts, "roles": {p: ROLE_ADMISSION for p in principles}})
        if stage == "sensor" and view in ("eeg64", "eeg64_noref"):
            out.append({"estimator_form": IIM_V1_QUADRANTS, "principles": ("IIM",),
                        "options": {"IIM": dict(_IIM_V1, **_IIM_PRIMARY_CUT)},
                        "roles": {"IIM": ROLE_COMPARATOR}})
        return out
    opts = {"IIM": dict(_IIM_DESCRIPTIVE, **_IIM_PRIMARY_CUT)}
    if stage == "sensor":
        opts = {"PDI": {"pdi_bearer": "full"},
                "IIM": dict(_IIM_SENSOR, **_IIM_DESCRIPTIVE, **_IIM_PRIMARY_CUT)}
    pdi_role = ROLE_SOURCE_CONTRAST if stage == "source" else ROLE_ADMISSION
    out.append({"estimator_form": PRIMARY, "principles": ("PDI", "IIM"),
                "options": opts,
                "roles": {"PDI": pdi_role, "IIM": ROLE_DESCRIPTIVE}})
    return out


@functools.lru_cache(maxsize=None)
def _source_contrast_runs(arm: str, purpose: str) -> Mapping[str, frozenset]:
    """``{condition: seeds}`` of a family-A arm's runs whose source view FMd
    reads: the two concordance conditions on the seeds they share."""
    out: Dict[str, frozenset] = {}
    for d in admission_designs(purpose):
        if d.arm != arm:
            continue
        hi, lo = d.concordance
        common = (frozenset(_seeds(condition(arm, hi), purpose))
                  & frozenset(_seeds(condition(arm, lo), purpose)))
        for c in (hi, lo):
            out[c] = out.get(c, frozenset()) | common
    return MappingProxyType(out)


def reads_view(task: ForwardTask, view: str) -> bool:
    """Whether some criterion or anchor reads ``view`` on this run: every
    view of the Hopf arm and every forward view of a family-A arm; the
    source view of a family-A arm, which is no admission view, only on the
    runs whose source contrast FMd reads (and on every anchor, reference
    and smoke run)."""
    if task.arm == ARM_HOPF or F2.view_spec(view).stage != "source":
        return True
    if task.purpose not in ADMISSION_PURPOSES:
        return True
    return task.seed in _source_contrast_runs(task.arm, task.purpose).get(
        task.condition, frozenset())


def scoring_plan(task: ForwardTask) -> List[dict]:
    """The scorings of a task: one per (view, estimator form) with its id,
    view, stage, declaration, principles, options (protocol vocabulary) and
    the role of each principle (admission, descriptive or comparator).
    A run carries what some criterion, hypothesis or anchor reads on it: the
    v1 quadrant comparator only on :data:`COMPARATOR_CONDITIONS`, only the
    principles with a lesion criterion on a lesion condition
    (:data:`LESION_PRINCIPLES_OF_ARM`) and the source view of a family-A arm
    only where :func:`reads_view` says so."""
    cond = condition(task.arm, task.condition)
    lesion = cond.spec.get("lesion", "none") not in (None, "none")
    out = []
    for view in task.views:
        if not reads_view(task, view):
            continue
        stage = F2.view_spec(view).stage
        for s in _scorings(task.arm, view):
            if (s["estimator_form"] == IIM_V1_QUADRANTS and task.condition
                    not in COMPARATOR_CONDITIONS.get(task.arm, ())):
                continue
            principles = list(s["principles"])
            if lesion:
                keep = LESION_PRINCIPLES_OF_ARM.get(task.arm, tuple(principles))
                principles = [p for p in principles if p in keep]
                if not principles:
                    continue
                s = dict(s, options={p: o for p, o in s["options"].items()
                                     if p in principles},
                         roles={p: r for p, r in s["roles"].items() if p in principles})
            out.append({
                "scoring_id": f"{view}:{s['estimator_form']}",
                "view": view,
                "observation_stage": stage,
                "declaration_id": task.declaration_id,
                "estimator_form": s["estimator_form"],
                "principles": principles,
                "options": copy.deepcopy(s["options"]),
                "roles": dict(s["roles"]),
            })
    return out


def entry_grain(principle: str, view: str) -> str:
    """The registry grain of an admission entry: IIM on sensor views is
    admitted for the rank-safe electrode clusters only."""
    if principle == "IIM" and F2.view_spec(view).stage == "sensor":
        return "electrode_clusters_rank_safe"
    return "*"


# --------------------------------------------------------------------------
# admission designs
# --------------------------------------------------------------------------
# FMa Bonferroni divisor declared per principle and arm where it differs from
# the number of null classes: IIM on the Hopf arm is judged at 0.05 / 4 over
# its four mixed views (HCv2-15(a)); PDI at 0.05 / 2 in every view
# (HCv2-21), the BOLD view included, although the BOLD plan has one null
# class (61 seeds demonstrate 0.07 with 0 events at either level).
FMA_M = MappingProxyType({("IIM", ARM_HOPF): 4, ("PDI", ARM_A_BOLD): 2})


def _planned_on(arm: str, on: Sequence[str], purpose: str) -> int:
    return sum(len(_seeds(condition(arm, c), purpose)) for c in on)


def admission_designs(purpose: str = CONFIRMATORY) -> Tuple[RV.AdmissionDesign, ...]:
    """
    The admission designs of the forward arms (registry v3 FM criteria):
    NAS and IIM on every Hopf view (null ``G = 0``, on ``G_nom``, dose the G
    sweep, concordance ``G_nom`` vs ``G = 0``; NAS also the hub vs matched
    random lesion), PDI on the family-A EEG views (nulls
    W_PDI_no_multistability and N_ar1) and on the BOLD view (null
    W_PDI_no_multistability), with on-runs PC_nominal and K in {2, 3}, dose
    K in {1, 2, 3, 6} and concordance K = 6 vs K = 1. The planned on-run
    count of the curtailment is the plan's for ``purpose``.
    """
    g0, gn = "hopf_G0", g_nom_label()
    hopf_dose = {c.label: c.dose for c in conditions(ARM_HOPF)
                 if c.spec.get("lesion") == "none"}
    out = []
    for p in ("NAS", "IIM"):
        views = tuple(v for v in VIEWS_OF_ARM[ARM_HOPF]
                      if p in [q for s in _scorings(ARM_HOPF, v) if s["estimator_form"]
                               == PRIMARY for q in s["principles"]])
        out.append(RV.AdmissionDesign(
            principle=p, arm=ARM_HOPF, views=views, null_conditions=(g0,),
            on_conditions=(gn,), dose_conditions=hopf_dose, concordance=(gn, g0),
            lesion=(("hopf_lesion_hub", "hopf_lesion_random_matched_hub")
                    if p == "NAS" else None),
            fma_m=FMA_M.get((p, ARM_HOPF)),
            n_planned_on=_planned_on(ARM_HOPF, (gn,), purpose)))
    k_dose = {"PC_nominal_K1": 1.0, "PC_nominal_K2": 2.0, "PC_nominal_K3": 3.0,
              "PC_nominal": 6.0}
    on = ("PC_nominal", "PC_nominal_K2", "PC_nominal_K3")
    for arm, views, nulls in (
            (ARM_A_EEG, ("eeg64", "eeglow"), ("W_PDI_no_multistability", "N_ar1")),
            (ARM_A_BOLD, ("bold",), ("W_PDI_no_multistability",))):
        out.append(RV.AdmissionDesign(
            principle="PDI", arm=arm, views=views, null_conditions=nulls,
            on_conditions=on, dose_conditions=k_dose,
            concordance=("PC_nominal", "PC_nominal_K1"),
            fma_m=FMA_M.get(("PDI", arm)),
            n_planned_on=_planned_on(arm, on, purpose)))
    return tuple(out)


# --------------------------------------------------------------------------
# the v2 runner adapter
# --------------------------------------------------------------------------
# The record design of each arm: the names the v2 hypotheses select by
# (vocabulary design.whole_brain, design.forward and
# design.forward_anchor_replication of protocols/v2/hypotheses_v2.json) and
# the names the registry builder reads. They are declared here and nowhere
# else; the hypotheses file and the registry builder are tested against them.
RECORD_DESIGN_OF_ARM = MappingProxyType({
    ARM_HOPF: "whole_brain",
    ARM_A_EEG: "forward_family_a",
    ARM_A_BOLD: "forward_family_a_bold",
})
ANCHOR_REPLICATION_DESIGN = "forward_anchor_replication"
# the records an admission reads (the anchor design's never enter a registry)
ADMISSION_RECORD_DESIGNS = frozenset(RECORD_DESIGN_OF_ARM.values())
RECORD_DESIGNS = ADMISSION_RECORD_DESIGNS | {ANCHOR_REPLICATION_DESIGN}
# purposes of the arm designs and of the anchor design
ARM_PURPOSES = (CONFIRMATORY, DRY_RUN, DEV_REGIME, SMOKE)
ANCHOR_PURPOSES = (REPLICATION, REFERENCE, REFERENCE_DEVELOPMENT)
MODULE = "forward"
BUILDER = "forward"
# protocol keys: one protocol per (arm, view), and per reported form
PROTOCOL_PREFIX_OF_ARM = MappingProxyType({
    ARM_HOPF: "hopf", ARM_A_EEG: "fwdA", ARM_A_BOLD: "fwdA_bold"})


def record_design(arm: str, purpose: str) -> str:
    """The ``design`` of a task record: the anchor design for the anchor
    purposes, else the arm's design."""
    if arm not in ARMS:
        raise ForwardDesignError(f"arm must be one of {ARMS}")
    if purpose not in PURPOSES:
        raise ForwardDesignError(f"purpose must be one of {PURPOSES}")
    return ANCHOR_REPLICATION_DESIGN if purpose in ANCHOR_PURPOSES else (
        RECORD_DESIGN_OF_ARM[arm])


def protocol_key_of(arm: str, view: str, form: str = PRIMARY) -> str:
    """The family protocol of an arm's view (``hopf-eeg64``) or of one of its
    forms (``hopf-eeg64+iim_v1_quadrants``)."""
    base = f"{PROTOCOL_PREFIX_OF_ARM[arm]}-{view}"
    return base if form == PRIMARY else f"{base}+{form}"


def protocol_options() -> Dict[str, dict]:
    """``{protocol key: {"declaration": id, "estimators": {principle:
    options}}}`` of every forward protocol: the declaration of the arm and
    the scoring options of :func:`scoring_plan` (protocol vocabulary)."""
    out = {}
    for arm in ARMS:
        for view in VIEWS_OF_ARM[arm]:
            for s in _scorings(arm, view):
                key = protocol_key_of(arm, view, s["estimator_form"])
                out[key] = {"declaration": DECLARATION_OF_ARM[arm],
                            "estimators": copy.deepcopy(s["options"])}
    return out


def _draft(key: str):
    def payload():
        from impact_pipeline.bench import run_bench_v2 as RB

        spec = protocol_options()[key]
        return RB.draft_payload(key, spec["declaration"], spec["estimators"])
    return payload


PROTOCOL_DRAFTS = {key: _draft(key) for key in protocol_options()}


def runner_task(ft: ForwardTask):
    """The v2 runner's task of a forward task: the same task id, the record
    design of its arm and purpose, one scoring per (view, estimator form)
    of :func:`scoring_plan` under the view's protocol, component statuses
    only; held out at the held-out regime."""
    from impact_pipeline.bench import designs_v2 as DV

    held = ft.purpose in HELD_OUT_PURPOSES
    scorings = []
    for s in scoring_plan(ft):
        scorings.append(DV.ScoringSpec(
            scoring_id=s["scoring_id"], declaration_id=s["declaration_id"],
            protocol_key=protocol_key_of(ft.arm, s["view"], s["estimator_form"]),
            view=s["view"], estimator_form=s["estimator_form"],
            principles=tuple(s["principles"]), held_out=held, verdict=False))
    cond = condition(ft.arm, ft.condition)
    return DV.TaskSpec(
        task_id=ft.task_id, design=record_design(ft.arm, ft.purpose),
        family=ft.family, system=ft.condition, seed=ft.seed, builder=BUILDER,
        params={"arm": ft.arm, "condition": ft.condition, "purpose": ft.purpose,
                "n_low": ft.n_low},
        scorings=tuple(scorings), held_out=held,
        tags={"arm": ft.arm, "purpose": ft.purpose, "regime": ft.regime,
              "dose": ft.dose, "anchor": bool(cond.anchor),
              "curtail_group": ft.curtail_group, "curtailable": curtailable(ft)},
        design_module=MODULE)


def forward_task_of(task) -> ForwardTask:
    """The forward task of a runner task (:func:`runner_task`)."""
    p = task.params
    ft = forward_task(p["arm"], p["condition"], task.seed, p["purpose"],
                      p.get("n_low", F2.N_LOW_DEFAULT))
    if ft.task_id != task.task_id:
        raise ForwardDesignError(f"{task.task_id}: not the forward task {ft.task_id}")
    return ft


def build_forward_system(task):
    """The runner's system builder: the task's realisation (one simulation,
    every view of it at the task's regime) as a
    :class:`~impact_pipeline.bench.designs_v2.MultiViewSystem`."""
    from impact_pipeline.bench import designs_v2 as DV

    real = realise(forward_task_of(task))
    return DV.MultiViewSystem(source=real.source, views=dict(real.views))


def _not_a_transform(source, task):
    raise ForwardDesignError(f"{task.task_id}: the forward views come from the "
                             "task's realisation, not from a transform")


def _runner_views() -> dict:
    from impact_pipeline.bench import designs_v2 as DV

    out = {}
    for name in sorted({v for vs in VIEWS_OF_ARM.values() for v in vs}):
        if name == DV.SOURCE_VIEW:
            continue
        spec = F2.view_spec(name)
        out[name] = DV.ViewSpec(name, spec.stage, F2.OBSERVATION_OF_STAGE[spec.stage],
                                transform=_not_a_transform,
                                description=f"forward view {name} (forward_v2)")
    return out


SYSTEM_BUILDERS = {BUILDER: build_forward_system}
VIEWS = _runner_views()
RECORD_CONFIG = {"forward": lambda task: forward_task_of(task).to_config()["forward"]}


def _select(tasks: List[ForwardTask], seeds, systems, design: str) -> List[ForwardTask]:
    if seeds is not None:
        want = {int(s) for s in seeds}
        S.check_seeds(want)
        tasks = [t for t in tasks if t.seed in want]
    if systems is not None:
        known = {c.label for t in tasks for c in conditions(t.arm)}
        unknown = sorted(set(systems) - known)
        if unknown:
            raise ForwardDesignError(f"{design}: unknown conditions {unknown}")
        tasks = [t for t in tasks if t.condition in set(systems)]
    if not tasks:
        raise ForwardDesignError(f"{design}: no task at the requested seeds and "
                                 "conditions")
    return tasks


def _arm_design(arm: str):
    def build(split: str, *, purpose: Optional[str] = None, seeds=None, systems=None,
              n_low: int = F2.N_LOW_DEFAULT):
        purpose = purpose or (CONFIRMATORY if split == S.CONFIRMATORY else DRY_RUN)
        if purpose not in ARM_PURPOSES:
            raise ForwardDesignError(f"{RECORD_DESIGN_OF_ARM[arm]}: purpose must be "
                                     f"one of {ARM_PURPOSES}")
        fts = build_tasks(purpose, arms=[arm], n_low=n_low)
        if seeds is not None or systems is not None:
            fts = _select(fts, seeds, systems, RECORD_DESIGN_OF_ARM[arm])
        return [runner_task(t) for t in fts]
    return build


def anchor_replication(split: str, *, purpose: Optional[str] = None, seeds=None,
                       systems=None, arms: Sequence[str] = ARMS,
                       n_low: int = F2.N_LOW_DEFAULT):
    """``forward_anchor_replication``: the anchor condition of every arm
    (``G_nom``; PC_nominal) on the replication block 20900-20919 at the
    held-out regime (confirmatory), or on the reference block 900-939 at the
    development regime (development; ``purpose='reference'`` gives the
    held-out regime, a held-out condition before the freeze)."""
    purpose = purpose or (REPLICATION if split == S.CONFIRMATORY
                          else REFERENCE_DEVELOPMENT)
    if purpose not in ANCHOR_PURPOSES:
        raise ForwardDesignError(f"{ANCHOR_REPLICATION_DESIGN}: purpose must be one "
                                 f"of {ANCHOR_PURPOSES}")
    fts = build_tasks(purpose, arms=arms, n_low=n_low)
    if seeds is not None or systems is not None:
        fts = _select(fts, seeds, systems, ANCHOR_REPLICATION_DESIGN)
    return [runner_task(t) for t in fts]


_FORWARD_METHODS = (
    "the view's family protocol (one per arm and view, draft until the "
    "development anchors are computed) judges each component by tost-v2 "
    "without a registry: the admission run is the registry's test, so sensor "
    "and source-estimate scorings carry the observation gate admission_run; "
    "component statuses only")
ADEMP = {
    "whole_brain": {
        "aims": "admission of NAS and IIM on human-like observations of a "
                "whole-brain model with known coupling: specificity at G = 0, "
                "dose, lesion and source concordance, exclusion safety at G_nom "
                "(registry v3), per view",
        "data": "the v1 Hopf model (76 regions, 250 Hz): G sweep 0-4 in 8 levels "
                "(20 seeds; G = 0 61, G_nom 150), hub lesion and size-matched "
                "random lesion at G_nom (20); one 600-s run per seed observed as "
                "source, EEG-64 (average and no reference), EEG-low, MNE template "
                "(first 60 s) and BOLD (whole run)",
        "estimands": "per view and principle: status and c per condition; the "
                     "admission flags of each view",
        "methods": ("NAS v3 and IIM v5 (v2 sensor pipeline: rank-safe clusters "
                    "and ZCA; the v1 quadrant pipeline as comparator); "
                    + _FORWARD_METHODS),
        "performance": "FM0, FMa, FMb1, FMb2 (NAS), FMd and FMabs with "
                       "curtailment (registry v3): HCv2-10, HCv2-12(d), HCv2-15, "
                       "IA-3",
    },
    "forward_family_a": {
        "aims": "admission of PDI (full-bearer upper bound) on EEG-like "
                "observations of family-A agents with known repertoires",
        "data": "family-A agents through the EEG-like model at 20 Hz without a "
                "sensor band: PC_nominal 150, N_ar1 61, W_PDI_no_multistability "
                "61, K in {1, 2, 3} x 20; views source, EEG-64 and EEG-low; "
                "declaration R",
        "estimands": "PDI status and c per view and condition (IIM descriptive); "
                     "the admission flags of each EEG view",
        "methods": "PDI v3 (content bearer on the source, full bearer on the "
                   "sensors) and IIM v5; " + _FORWARD_METHODS,
        "performance": "FM0, FMa at 0.05 / 2, FMb1 over K, FMd (K = 6 vs 1) and "
                       "FMabs: HCv2-21",
    },
    "forward_family_a_bold": {
        "aims": "admission of PDI on BOLD-like observations of family-A agents "
                "with slow contexts (predicted not admitted for ABSENT)",
        "data": "slow-context agents (context dwell 30-60 s) through the "
                "BOLD-like model (TR 2 s): PC_nominal up to 150 (curtailed), "
                "W_PDI_no_multistability 61, K in {1, 2, 3} x 20; views source "
                "and BOLD; declaration R",
        "estimands": "PDI status and c per view and condition (IIM descriptive); "
                     "the admission flags of the BOLD view",
        "methods": "PDI v3 on the content bearer (window 5 TR) and IIM v5; "
                   + _FORWARD_METHODS + "; the PC_nominal runs only FMabs reads "
                   "stop in seed order at the first event that makes the FMabs "
                   "demonstration impossible",
        "performance": "FM0, FMa at 0.05 / 2, FMb1 over K, FMd and curtailed "
                       "FMabs: HCv2-21",
    },
    ANCHOR_REPLICATION_DESIGN: {
        "aims": "the anchors of the forward views: validity of each view's "
                "reference condition, fixed on the development reference block "
                "and replicated on the confirmatory replication block",
        "data": "G_nom of the Hopf arm and PC_nominal of both family-A arms: "
                "900-939 (development regime; the held-out regime only after "
                "the held-out predictions are committed) and 20900-20919 "
                "(held-out regime)",
        "estimands": "per view and principle the mean excess of the reference "
                     "condition over its null",
        "methods": "as the arm designs; " + _FORWARD_METHODS,
        "performance": "anchor validity (one-sided 95 % t lower bound of the "
                       "mean excess > 0) and its replication: HCv2-6(b)",
    },
}


def _design(name: str, build, confirmatory_tasks: int, description: str):
    from impact_pipeline.bench import designs_v2 as DV

    return DV.Design(name, "forward", description, build,
                     {DV.CONFIRMATORY: confirmatory_tasks}, ademp=ADEMP[name])


# Confirmatory task counts of the design document (3.2; the BOLD arm's 271 is
# the planned maximum before curtailment) and of the replication block.
DOCUMENT_TASKS = MappingProxyType({
    "whole_brain": 371, "forward_family_a": 332, "forward_family_a_bold": 271,
    ANCHOR_REPLICATION_DESIGN: len(ARMS) * len(REPLICATION_SEEDS)})


def _designs():
    return (
        _design("whole_brain", _arm_design(ARM_HOPF), DOCUMENT_TASKS["whole_brain"],
                "Hopf arm: G sweep, G_nom lesions, every view (NAS, IIM)"),
        _design("forward_family_a", _arm_design(ARM_A_EEG),
                DOCUMENT_TASKS["forward_family_a"],
                "forward-modelled family A through the EEG-like model (PDI, IIM)"),
        _design("forward_family_a_bold", _arm_design(ARM_A_BOLD),
                DOCUMENT_TASKS["forward_family_a_bold"],
                "forward-modelled family A with slow contexts through the "
                "BOLD-like model (PDI, IIM; curtailed)"),
        _design(ANCHOR_REPLICATION_DESIGN, anchor_replication,
                DOCUMENT_TASKS[ANCHOR_REPLICATION_DESIGN],
                "reference conditions of the forward views (reference block; "
                "confirmatory replication block)"),
    )


DESIGNS = _designs()


class RunnerCurtailment:
    """
    The curtailed sampling of the forward arms in the v2 runner (design 3.6):
    the on-runs of an arm's curtailment group are evaluated in seed order
    (:func:`curtailment_order`), and once the false ABSENTs among them make
    the FMabs demonstration impossible at the planned on-run count, in every
    admission view of the group, the group's remaining :func:`curtailable`
    runs are skipped. A demonstration that is impossible before any event
    (a development dry run plans too few on-runs) stops at its first event,
    as the confirmatory plan does. The decision for a run depends only on
    the records of the runs before it in that order, so the kept records do
    not depend on the number of workers. Controller contract:
    :func:`impact_pipeline.bench.run_bench_v2.curtailment_controllers`.
    """

    def __init__(self, tasks: Sequence):
        crit = RV.DEFAULT_ADMISSION_CRITERIA
        fts = []
        for t in tasks:
            if getattr(t, "builder", None) != BUILDER:
                continue
            ft = forward_task_of(t)
            if ft.curtail_group:
                fts.append(ft)
        self._groups: Dict[tuple, dict] = {}
        self._group_of: Dict[str, tuple] = {}
        self._events: Dict[str, dict] = {}
        self._decided: Dict[str, bool] = {}
        by_purpose: Dict[str, List[ForwardTask]] = {}
        for ft in fts:
            by_purpose.setdefault(ft.purpose, []).append(ft)
        for purpose, group_tasks in sorted(by_purpose.items()):
            designs = admission_designs(purpose)
            for g, order in curtailment_order(group_tasks).items():
                arm = order[0].arm
                thresholds = {}
                for d in designs:
                    if d.arm != arm:
                        continue
                    k_max = RV.max_demonstrable_events(
                        d.n_planned_on, crit["fmabs_bound"], crit["fmabs_level"])
                    for v in d.views:
                        thresholds[(d.principle, v)] = max(int(k_max), 0)
                key = (purpose, g)
                self._groups[key] = {
                    "order": [t.task_id for t in order],
                    "gated": [t.task_id for t in order if curtailable(t)],
                    "thresholds": thresholds,
                    "events": {k: 0 for k in thresholds},
                    "stop_after": {k: None for k in thresholds},
                    "pos": 0,
                }
                for t in order:
                    self._group_of[t.task_id] = key

    def __bool__(self) -> bool:
        return any(st["gated"] for st in self._groups.values())

    def gated_ids(self) -> List[str]:
        return [tid for st in self._groups.values() for tid in st["gated"]]

    @staticmethod
    def _stopped(st) -> bool:
        return bool(st["thresholds"]) and all(
            st["events"][k] > st["thresholds"][k] for k in st["thresholds"])

    def _record_events(self, record, keys) -> dict:
        out = {}
        if getattr(record, "status", None) == "error":
            return out
        for s in getattr(record, "scorings", ()) or ():
            if s.estimator_form != PRIMARY:
                continue
            for p, v in keys:
                comp = (s.components or {}).get(p)
                if s.view == v and comp is not None:
                    out[(p, v)] = comp.status == "ABSENT"
        return out

    def feed(self, task_id: str, record) -> None:
        key = self._group_of.get(task_id)
        if key is None or task_id in self._events:
            return
        st = self._groups[key]
        self._events[task_id] = self._record_events(record, st["thresholds"])
        self._advance(st)

    def _advance(self, st) -> None:
        gated = set(st["gated"])
        while st["pos"] < len(st["order"]):
            tid = st["order"][st["pos"]]
            if tid in gated and tid not in self._decided:
                self._decided[tid] = self._stopped(st)
            if self._decided.get(tid):
                st["pos"] += 1
                continue
            if tid not in self._events:
                break
            for k, ev in self._events[tid].items():
                if ev and st["stop_after"][k] is None:
                    st["events"][k] += 1
                    if st["events"][k] > st["thresholds"][k]:
                        st["stop_after"][k] = tid
            st["pos"] += 1

    def decision(self, task_id: str) -> Optional[bool]:
        """None while undecided, True to skip, False to run; False for a task
        that is not gated."""
        key = self._group_of.get(task_id)
        if key is None or task_id not in self._groups[key]["gated"]:
            return False
        self._advance(self._groups[key])
        return self._decided.get(task_id)

    def summary(self) -> dict:
        groups = []
        for (purpose, g), st in self._groups.items():
            groups.append({
                "purpose": purpose, "group": g, "n_runs": len(st["order"]),
                "n_gated": len(st["gated"]), "evaluated": st["pos"],
                "views": [{"principle": p, "view": v,
                           "events": st["events"][(p, v)],
                           "max_events": st["thresholds"][(p, v)],
                           "stop_after": st["stop_after"][(p, v)]}
                          for p, v in st["thresholds"]],
                "stopped": self._stopped(st),
                "skipped": [t for t in st["gated"] if self._decided.get(t)],
            })
        return {"module": MODULE, "rule": "fmabs_curtailment", "groups": groups}


def RUNNER_CURTAILMENT(tasks):  # noqa: N802 - the runner's hook name
    """The curtailment controller of a plan's forward tasks (None when no
    task may be skipped)."""
    ctl = RunnerCurtailment(tasks)
    return ctl if ctl else None


__all__ = [
    "ADEMP",
    "ADMISSION_PURPOSES",
    "ADMISSION_RECORD_DESIGNS",
    "ADMISSION_RUN",
    "ANCHOR_PURPOSES",
    "ANCHOR_REPLICATION_DESIGN",
    "ARMS",
    "ARM_PURPOSES",
    "ARM_A_BOLD",
    "ARM_A_EEG",
    "ARM_HOPF",
    "CURTAIL_GROUP_A_BOLD",
    "Condition",
    "DESIGN",
    "DRY_RUN_FRACTION",
    "FMA_M",
    "ForwardDesignError",
    "ForwardTask",
    "HOPF_BOLD_DURATION_S",
    "HOPF_SOURCE_DURATION_S",
    "PURPOSES",
    "Realisation",
    "VIEWS_OF_ARM",
    "admission_designs",
    "arm_regime",
    "build_tasks",
    "check_regime_policy",
    "condition",
    "conditions",
    "curtailable",
    "curtailment_order",
    "entry_grain",
    "g_nom_label",
    "realise",
    "regime_of_purpose",
    "scoring_plan",
    "truncate_whole_brain",
    "BUILDER",
    "COMPARATOR_CONDITIONS",
    "CONFIRMATORY_PURPOSES",
    "LESION_PRINCIPLES_OF_ARM",
    "DESIGNS",
    "DOCUMENT_TASKS",
    "PROTOCOL_DRAFTS",
    "PROTOCOL_PREFIX_OF_ARM",
    "RECORD_CONFIG",
    "RECORD_DESIGNS",
    "RECORD_DESIGN_OF_ARM",
    "REPLICATION",
    "REPLICATION_SEEDS",
    "RUNNER_CURTAILMENT",
    "RunnerCurtailment",
    "SYSTEM_BUILDERS",
    "VIEWS",
    "anchor_replication",
    "build_forward_system",
    "forward_task",
    "forward_task_of",
    "protocol_key_of",
    "protocol_options",
    "reads_view",
    "record_design",
    "runner_task",
]
