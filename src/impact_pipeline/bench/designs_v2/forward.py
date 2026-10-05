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
20261001, width 0.6). ``dry_run`` (seeds 384-399) and ``dev_regime``
(804-819): the same conditions at about 15 % scale, at the development
regime. ``reference`` (900-939): the anchor conditions (``G_nom``,
PC_nominal) at the held-out regime, computed only after the held-out
predictions are committed; ``reference_development``: the same seeds at the
development regime. ``smoke`` (980-984): the anchor conditions at the
held-out regime, outputs discarded unread. No other development task may
use the held-out regime (:func:`check_regime_policy`).

Record contract (what the registry builder reads)
-------------------------------------------------
A task record (``mpc-bench-result/3``) of this design stores
:meth:`ForwardTask.to_config` as ``config`` (``config['forward']`` names the
arm, condition, dose, purpose and regime); each scoring has ``view`` = the
view name, ``estimator_form`` from :func:`scoring_plan`, and
``details`` = :func:`impact_pipeline.bench.forward_v2.scoring_details` of the
view (its registry regime keys). Components are scored under the view's
family protocol without a registry (the admission run decides the
registry); sensor and source-estimate scorings carry the observation-gate
label ``admission_run`` (NAS N10, IIM C7).

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
DRY_RUN = "dry_run"
DEV_REGIME = "dev_regime"
REFERENCE = "reference"
REFERENCE_DEVELOPMENT = "reference_development"
SMOKE = "smoke"
PURPOSES = (CONFIRMATORY, DRY_RUN, DEV_REGIME, REFERENCE, REFERENCE_DEVELOPMENT, SMOKE)
HELD_OUT_PURPOSES = (CONFIRMATORY, REFERENCE, SMOKE)
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
    regime = regime_of_purpose(purpose).name
    tasks = []
    for arm in arms:
        if arm not in ARMS:
            raise ForwardDesignError(f"arm must be one of {ARMS}")
        for cond in conditions(arm):
            for seed in _seeds(cond, purpose):
                tasks.append(ForwardTask(
                    task_id=f"{DESIGN}-{arm}-{cond.label}-{regime}-s{int(seed):05d}",
                    arm=arm, condition=cond.label, seed=int(seed), purpose=purpose,
                    regime=regime, views=VIEWS_OF_ARM[arm], dose=cond.dose,
                    curtail_group=(cond.curtail_group
                                   if purpose in ADMISSION_PURPOSES else None),
                    n_low=int(n_low)))
    check_regime_policy(tasks)
    return tasks


def check_regime_policy(tasks: Sequence[ForwardTask]) -> None:
    """
    The seed and regime policy of the forward arms: every task's seed obeys
    the v2 seed policy for its purpose (confirmatory seeds only for
    ``confirmatory``); the held-out regime appears on development seeds only
    on the reference block (900-939, anchors after the held-out predictions
    are committed) and the smoke seeds (980-984, outputs discarded); every
    confirmatory task is at the held-out regime.
    """
    for t in tasks:
        split = S.split_of(t.seed)
        if t.purpose not in PURPOSES:
            raise ForwardDesignError(f"{t.task_id}: unknown purpose {t.purpose!r}")
        if (t.purpose == CONFIRMATORY) != (split == S.CONFIRMATORY):
            raise ForwardDesignError(
                f"{t.task_id}: purpose {t.purpose} on a {split} seed")
        if (t.purpose != CONFIRMATORY
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
    system.meta["forward_condition"] = task.condition
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


def _scorings(arm: str, view: str) -> List[dict]:
    stage = F2.view_spec(view).stage
    mixed = stage in ("sensor", "source_estimate")
    out = []
    if arm == ARM_HOPF:
        principles = ("IIM",) if view == "eeg64_noref" else ("NAS", "IIM")
        opts = {}
        if mixed:
            iim = _IIM_SENSOR if stage == "sensor" else _IIM_SOURCE_ESTIMATE
            opts = {"NAS": dict(_GATE), "IIM": dict(iim)}
            opts = {p: o for p, o in opts.items() if p in principles}
        out.append({"estimator_form": PRIMARY, "principles": principles,
                    "options": opts, "roles": {p: ROLE_ADMISSION for p in principles}})
        if stage == "sensor" and view in ("eeg64", "eeg64_noref"):
            out.append({"estimator_form": IIM_V1_QUADRANTS, "principles": ("IIM",),
                        "options": {"IIM": dict(_IIM_V1)},
                        "roles": {"IIM": ROLE_COMPARATOR}})
        return out
    opts = {}
    if stage == "sensor":
        opts = {"PDI": {"pdi_bearer": "full"}, "IIM": dict(_IIM_SENSOR)}
    pdi_role = ROLE_SOURCE_CONTRAST if stage == "source" else ROLE_ADMISSION
    out.append({"estimator_form": PRIMARY, "principles": ("PDI", "IIM"),
                "options": opts,
                "roles": {"PDI": pdi_role, "IIM": ROLE_DESCRIPTIVE}})
    return out


def scoring_plan(task: ForwardTask) -> List[dict]:
    """The scorings of a task: one per (view, estimator form) with its id,
    view, stage, declaration, principles, options (protocol vocabulary) and
    the role of each principle (admission, descriptive or comparator)."""
    out = []
    for view in task.views:
        stage = F2.view_spec(view).stage
        for s in _scorings(task.arm, view):
            out.append({
                "scoring_id": f"{view}:{s['estimator_form']}",
                "view": view,
                "observation_stage": stage,
                "declaration_id": task.declaration_id,
                "estimator_form": s["estimator_form"],
                "principles": list(s["principles"]),
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


__all__ = [
    "ADMISSION_PURPOSES",
    "ADMISSION_RUN",
    "ARMS",
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
]
