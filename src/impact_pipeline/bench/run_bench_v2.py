"""
MPC-Bench v2 runner: one simulation per task, several scorings, every
estimator isolated, records of schema ``mpc-bench-result/3``.

The v1 runner (:mod:`impact_pipeline.bench.run_bench`) is frozen and keeps
its single try block, so v1 re-runs still reproduce the v1 errors. This
runner is used for every v2 design (:mod:`impact_pipeline.bench.designs_v2`).

Task execution (:func:`run_task`)
---------------------------------
1. The system is built once by the task's system builder (``catalogue``:
   a witness or adversary of the v2 catalogue; ``agent``: the family-A or
   family-C agent at given knobs; design modules may register more, such as
   ``forward``, which returns one simulation with every view of it at the
   task's regime, a :class:`~impact_pipeline.bench.designs_v2.MultiViewSystem`).
   A design module's record blocks (``RECORD_CONFIG``, for example
   ``config['forward']``) are added to the record's ``config``.
2. The simulation block records the duplicate-detector hashes
   ``sha256(ts)`` and ``sha256(raw_ts)``, the twin hashes (structural and
   schedule), the shape, ``dt`` and the simulation time.
3. Each scoring (declaration, view, estimator form, protocol) dispatches
   every principle to the estimator version its family protocol names
   (:data:`SCORERS`). Each estimator runs in its own try block: a raising
   estimator gives ``UNDEFINED(ESTIMATOR_ERROR:<type>)`` for that component
   and touches nothing else; the task status is then
   ``ok_with_component_errors``. An output that the evidence layer or the
   ``/3`` record refuses (for example ``se_df <= 0``) is an estimator error
   of that component in the same way, and the verdict is taken again
   without it. An estimator output that does not depend on the declaration
   (RAM-PE, PDI, SRPI) is computed once per view and form and reused by the
   other declarations of the same simulation (an estimator that reads the
   protocol's anchor, such as RAM-PE for its ``absent_reachable``
   descriptor, only under the same anchor).
4. The status rule ``tost-v2`` of the scoring's protocol judges the
   components (:func:`impact_pipeline.evidence_v2.verdict_v3`); primary
   scorings of the joint bench carry the verdict, form scorings and arms
   that score one principle by design the component statuses only. Each
   scoring records the input bases its estimators used, with their
   effective rank (the conditioning budget; the lagged filtered copies are
   collinear by construction).

Estimator dispatch (:data:`SCORERS`, by estimator version)
-----------------------------------------------------------
* ``ram-v3-2026.10`` and ``pdi-v3-2026.10``: :mod:`impact_pipeline.v2.ram_v3`
  and :mod:`impact_pipeline.v2.pdi_v3` through their public functions.
* ``srpi-v2-2026.09`` (the v1 SRPI) and the v1 fallbacks ``nas-v2-2026.09``,
  ``iim-v4-2026.09``, ``pdi-v2-2026.09``, ``ram-v2-2026.09``: the unchanged
  v1 in-memory path (:func:`impact_pipeline.bench.export.run_in_memory`) with
  the v1 null size (19 surrogates) and jackknife (10 groups), one principle
  at a time; each principle's block of that function is independent of the
  others, so the component equals the one a v1 run computes.
* ``nas-v3-2026.10``: :func:`impact_pipeline.v2.nas_v3.compute_nas_v3` on the
  bearer view, with the hub and periphery blocks the system declares
  (:func:`~impact_pipeline.v2.nas_v3.declared_blocks`), ``tau_c`` of the
  protocol checked against the generator constants (the ``tau_c``
  sensitivity forms vary ``lag_timescale_sec``, the lags and filtered input
  copies, never the resolvability gate's ``tau_c``), the scoring's declared
  inputs and the view's observation.
* ``iim-v5-2026.10``: :func:`impact_pipeline.v2.iim_v5.compute_iim_v5` on the
  bearer view's macro nodes at the system's transition lag
  (``iim_lag_samples``, the v1 rule), residualised on the scoring's input
  basis (families A and C1 declare no binary drivers to stratify on). One
  run per inputs computes every cut mode the simulation's scorings of those
  inputs record, and each scoring reads its own protocol's cut modes from
  it (:func:`impact_pipeline.v2.iim_v5.select_cut_modes`): the
  bidirectional form beside its primary scoring reads the cut the primary
  already reports instead of computing it again, and every record holds
  what its own run would hold. A grain with more macro nodes than the
  run's declared cap (``RunSettings.iim_max_macro_nodes``, 4) is refused
  as an estimator error of that component before any statistic runs.
* Runner options (:data:`RUNNER_OPTIONS`, never passed to an estimator):
  ``observation_gate = admission_run`` scores a mixed observation of a
  forward-arm admission run (NAS v3 ``override_observation_gate``, IIM v5
  ``observation_admitted``), and IIM's ``macro_nodes`` picks the view's
  declared grain, the rank-safe electrode clusters or the v1 quadrants.
* The Tier-B ``srpi-v3-2026.10``: the estimator module, through its hook
  ``score_bench_v2(context)`` when it defines one, otherwise through its
  ``compute_<p>_v<n>`` function called with the keyword arguments of
  :meth:`ScoringContext.candidates` that its signature names, the result
  mapped like the merged v3 modules' (``component_fields``; per-direction
  results under ``directions``). A module that is not in the tree yet is
  reported as unavailable before any task runs.

Protocols (:func:`resolve_protocols`)
-------------------------------------
Family protocols are keyed ``A-R``, ``A-H``, ``A-P``, ``A-Q10``, ``A-Q25``,
``A-J``, ``A-RAM160``, ``C1-R``, ``C1-H`` and, for an estimator form,
``<family protocol>+<form>``. The frozen files are read from
``protocols/v2/generated/mpc_bench_v2_<key>.json`` (written by the protocol
builder). Before they exist, development runs derive a *draft* from the
template ``protocols/v2/mpc_bench_v2_template.json`` (the declaration set,
the form's estimator options, the reference ``pending``, so every component
is UNDEFINED(INVALID_ANCHORS) and the raw outputs are what the anchors are
computed from). Drafts admit the SE method ``exact`` for IIM, so that an
exact known-TPM value is decided on ``c`` rather than refused. A design
module declares the drafts of its own protocols (``PROTOCOL_DRAFTS``: the
forward arms' ``hopf-eeg64``, ``fwdA-eeglow``, ...; the null-calibration
generator's ``A-none``); :func:`protocol_keys` registers a plan's design
modules before the keys are resolved, so a fresh process resolves them, and
a draft key declared twice is refused. Every record names its protocol by
the key (``protocol_id``), the convention the evaluator and the integrity
audit load the frozen files by
(:func:`impact_pipeline.v2.hypothesis_engine.load_protocol_files`).
Confirmatory runs use generated protocols only.

Run hygiene (design 3.8)
------------------------
Development runs use seeds 0-999 only, confirmatory runs seeds >= 20000 only
and only through :func:`impact_pipeline.v2.provenance.confirmatory_guard`
(clean tree, the freeze tag's ``src/`` and ``scripts/`` trees), with the
generated protocols and the default runner settings. Held-out conditions run
on the development split only on the smoke seeds 980-984, whose outputs are
discarded unread (``smoke.jsonl`` holds status lines only). The run plan
(``run_plan.json``) is the list of task specs the design builders returned,
with the runner settings and the protocol hashes; the records are appended
to ``results.jsonl`` one line per task, and a run resumes from partial
results (a truncated last line and failed tasks are dropped and re-run) only
under the same plan, settings and protocols. Per-task timing carries the
worker's CPU time (``cpu_s``) and the load average; the manifest records the
environment, the code identity and the byte hashes of the protocol files
read. Workers are spawned processes with one BLAS thread each.

Curtailed sampling (:func:`curtailment_controllers`): a design module may
stop some of its runs early by a preregistered rule (the forward BOLD arm's
FMabs). Its gated runs are released in the rule's order and kept only when
the records before them say so, so the kept records, and the curtailed task
ids the manifest lists, do not depend on the number of workers; a resumed
run takes the same decisions from the records it finds. Such a plan is
never split into shards (:func:`shard`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import math
import multiprocessing
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline.bench import designs_v2 as D
from impact_pipeline.v2 import GENERATOR_VERSION_V2
from impact_pipeline.v2 import hypothesis_engine as _names
from impact_pipeline.v2 import provenance as PV
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import records as REC
from impact_pipeline.v2 import seeds as S

log = logging.getLogger("impact_pipeline.bench.v2")

RUNNER_VERSION = "mpc-bench-runner/2.0.0"
RESULTS_JSONL = "results.jsonl"
SMOKE_JSONL = "smoke.jsonl"
COMPONENTS_CSV = "components.csv"
MANIFEST = "run_manifest.json"
PLAN_JSON = "run_plan.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "protocols" / "v2" / "mpc_bench_v2_template.json"
GENERATED_DIR = REPO_ROOT / "protocols" / "v2" / "generated"
# the file-name convention of protocol keys (hypothesis_engine.protocol_file_name)
PROTOCOL_FILE = _names.PROTOCOL_FILE_PREFIX + "{key}" + _names.PROTOCOL_FILE_SUFFIX

# The v1 in-memory path: null size and jackknife groups of the v1 bench.
V1_NULL_SURROGATES = 19
V1_SE_GROUPS = 10
V1_SE_METHODS = {
    "SRPI": "jackknife_pairs_10",
    "NAS": "jackknife_contiguous_10",
    "IIM": "jackknife_contiguous_10",
    "PDI": "jackknife_contiguous_10",
}

# Parameters that development calibration fixes before the freeze. Each is
# declared once: the runner's here (recorded in every task through the run
# settings), the designs' in their module (the anchor replication blocks in
# designs_v2.anchors); every run manifest records all of them
# (:func:`calibration_pending`).
CALIBRATION_PENDING = {
    "pdi_kmeans_seed": {
        "value": 0,
        "meaning": "k-means seed of every PDI v3 count (run, surrogates, "
                   "replicates); an integer, or None for the task's null seed. "
                   "Value 0: the seed of the v2 development feasibility work "
                   "and of every development calibration run, kept as "
                   "decided (not tuned on data); the v1 bench used the task's "
                   "null seed",
    },
}

# The largest IIM v5 macro grain a run computes (RunSettings.iim_max_macro_nodes):
# every Tier-A substrate declares at most four macro nodes (the four
# periphery modules of families A and C1, the four family-B units, the v1
# grain of four groups of the null-calibration generator, the Hopf model's
# four cortical macro nodes and the sensor views' four electrode groups), and
# the patchwork is recorded in its principle-bearer grain (the three
# sub-groups of its IIM module; designs_v2.family_a). A five-node grain (32
# macro states, 30 directional bipartitions) costs about 28 times a four-node
# one for every statistic of the null and the bootstrap.
IIM_MAX_MACRO_NODES = 4

# Principles declared not applicable on an observation model by the runner,
# for estimators that do not declare it themselves (the v1 SRPI and the v1
# fallbacks). RAM-PE v3 and PDI v3 declare C1 themselves.
NOT_APPLICABLE = {
    "C1": {
        "SRPI": "v1 family-C carrier recording (Re z at 20 Hz): SRPI declared not "
                "applicable (design S6)",
        "RAM": "v1 family-C carrier recording: RAM declared not applicable",
        "PDI": "v1 family-C carrier recording: PDI declared not applicable",
    },
}

# The family protocols whose drafts the runner derives from the template:
# key -> (family, declaration).
FAMILY_PROTOCOLS = {
    "A-R": ("A", "R"), "A-H": ("A", "H"), "A-P": ("A", "P"), "A-Q10": ("A", "Q10"),
    "A-Q25": ("A", "Q25"), "A-J": ("A", "J"), "A-RAM160": ("A", "R"),
    "C1-R": ("C1", "R"), "C1-H": ("C1", "H"),
}

# Protocol option names that differ from the estimators' parameter names
# (RAM-PE facets F, G, speed; the PDI option pdi_bearer). The renaming tables
# live in one place, impact_pipeline.v2.hypothesis_engine, which the
# evaluator and the integrity audit read as well; the runner translates
# through them. PROTOCOL_OPTION_NAMES documents every option an estimator
# of this runner reads.
RAM_FACET_NAMES = _names.RAM_FACET_NAMES
PDI_OPTION_NAMES = _names.PDI_OPTION_NAMES
PROTOCOL_OPTION_NAMES = {
    "RAM": {"facets_not_applicable": "facets (G, F, speed -> goal_alignment, "
                                     "feedback_magnitude, speed)",
            "update": "mode (ram_v3.MODE)"},
    "PDI": {"pdi_bearer": "bearer (PDIParams.bearer)", "mode": "mode (pdi_v3.MODE)",
            "access_module": "runner: the module declared as the access nodes"},
    "NAS": {"coupling_timescale_sec": "tau_c of the resolvability gate (NASParams; "
                                      "checked against the generator constants)",
            "lag_timescale_sec": "lag set and filtered input copies (NASParams; "
                                 "the tau_c sensitivity forms)",
            "observation_gate": "runner: 'admission_run' scores a mixed observation "
                                "without admission (override_observation_gate), "
                                "because the forward-arm scoring is the admission "
                                "test"},
    "IIM": {"coupling_timescale_sec": "runner: tau_c of the input basis and lag set",
            "lag_timescale_sec": "runner: lag set and filtered copies of the basis",
            "tpm_estimator": "the estimator's TPM estimator (iim_v5.TPM_ESTIMATOR); "
                             "not a parameter",
            "preprocess": "pre-processing (IIMParams.preprocess: none or zca)",
            "observation_gate": "runner: 'admission_run' scores a mixed observation "
                                "as admitted (observation_admitted), because the "
                                "forward-arm scoring is the admission test",
            "macro_nodes": "runner: the macro grain of the view, 'declared' (the "
                           "view's iim_macro_nodes), 'rank_safe_clusters' (the "
                           "declared grain, which must be the rank-safe electrode "
                           "clusters) or 'v1_quadrants' (the v1 electrode "
                           "quadrants the forward layer keeps beside them)"},
}
# Options the runner reads itself (never passed to the estimator), with their
# admitted values. OBSERVATION_GATE_ADMISSION_RUN is the forward arms' label
# (designs_v2.forward.ADMISSION_RUN).
OBSERVATION_GATE_ADMISSION_RUN = "admission_run"
MACRO_NODES_DECLARED = "declared"
MACRO_NODES_RANK_SAFE = "rank_safe_clusters"
MACRO_NODES_V1_QUADRANTS = "v1_quadrants"
RANK_SAFE_GRAIN = "electrode_clusters_rank_safe"
RUNNER_OPTIONS = {
    "NAS": {"observation_gate": (OBSERVATION_GATE_ADMISSION_RUN,)},
    "IIM": {"observation_gate": (OBSERVATION_GATE_ADMISSION_RUN,),
            "macro_nodes": (MACRO_NODES_DECLARED, MACRO_NODES_RANK_SAFE,
                            MACRO_NODES_V1_QUADRANTS)},
}
DIRECTION_ALIASES = {
    "receive": "receive", "r": "receive", "in": "receive", "te_in": "receive",
    "return": "return", "b": "return", "out": "return", "te_out": "return",
}


class RunPolicyError(RuntimeError):
    """A run the seed, held-out or protocol policy refuses."""


class ProtocolOptionError(ValueError):
    """A protocol option that contradicts the estimator it configures."""


class EstimatorUnavailableError(ImportError):
    """An estimator module the protocol dispatches is not in the tree yet."""


# --------------------------------------------------------------------------
# settings and seeds
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RunSettings:
    """Runner settings recorded with every task: the principles scored
    (None: every principle of a scoring), the v1 in-memory null size and
    jackknife groups, the PDI k-means seed (:data:`CALIBRATION_PENDING`),
    whether an unavailable estimator module may be recorded as an estimator
    error instead of refusing the run, and the cap on the number of IIM v5
    macro nodes (:data:`IIM_MAX_MACRO_NODES`; None: no cap)."""

    principles: Optional[Tuple[str, ...]] = None
    v1_null_surrogates: int = V1_NULL_SURROGATES
    v1_se_groups: int = V1_SE_GROUPS
    pdi_kmeans_seed: Optional[int] = CALIBRATION_PENDING["pdi_kmeans_seed"]["value"]
    allow_unavailable_estimators: bool = False
    iim_max_macro_nodes: Optional[int] = IIM_MAX_MACRO_NODES

    def __post_init__(self):
        if self.principles is not None:
            object.__setattr__(self, "principles", D._principles(
                self.principles, "settings"))
        if int(self.v1_se_groups) == 1 or int(self.v1_se_groups) < 0:
            raise ValueError("v1_se_groups must be 0 or >= 2")
        if self.iim_max_macro_nodes is not None and int(self.iim_max_macro_nodes) < 1:
            raise ValueError("iim_max_macro_nodes must be None or >= 1")

    def wants(self, principle: str) -> bool:
        return self.principles is None or principle in self.principles

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["principles"] = None if self.principles is None else list(self.principles)
        return out

    @classmethod
    def from_dict(cls, payload: Optional[Mapping]) -> "RunSettings":
        data = dict(payload or {})
        if data.get("principles") is not None:
            data["principles"] = tuple(data["principles"])
        return cls(**data)


DEFAULT_SETTINGS = RunSettings()


def calibration_pending() -> dict:
    """Every pending calibration parameter, by name: the runner's
    (:data:`CALIBRATION_PENDING`) and the designs' (``anchors.<name>``)."""
    from impact_pipeline.bench.designs_v2 import anchors as AN

    out = dict(CALIBRATION_PENDING)
    out.update({f"anchors.{k}": v for k, v in AN.CALIBRATION_PENDING.items()})
    return out


def null_seed(seed: int, replicate: int = 0) -> int:
    """Seed of the surrogate draws of a task: ``seed * 1000 + 17`` for the run
    itself (the v1 rule), and for twin ``r >= 1`` a draw of the new key
    ``SeedSequence([seed, r, 17])``, so twin sessions do not share null
    draws. The draw is reduced below ``2**31`` so that an estimator may
    offset it (``RandomState(seed + k)``) without leaving the legacy seed
    range."""
    seed, r = int(seed), int(replicate)
    if r == 0:
        return seed * 1000 + 17
    return int(np.random.SeedSequence([seed, r, 17]).generate_state(1)[0]) % 2 ** 31


# --------------------------------------------------------------------------
# protocol options -> estimator parameters (the one mapping)
# --------------------------------------------------------------------------
def translate_facets(facets: Mapping) -> dict:
    """Protocol facet keys (``G``, ``F``, ``speed``) -> RAM-PE v3 facet names
    (``goal_alignment``, ``feedback_magnitude``, ``speed``), through the one
    map (:func:`impact_pipeline.v2.hypothesis_engine.ram_facets_for_estimator`)."""
    try:
        out = _names.ram_facets_for_estimator(facets)
    except ValueError as exc:
        raise ProtocolOptionError(str(exc)) from None
    return {k: str(v) for k, v in out.items()}


def ram_v3_params(options: Mapping) -> dict:
    """RAM-PE v3 parameters from protocol options: ``update`` must be the
    estimator's mode, the declared not-applicable facets (protocol names) must
    equal ``ram_v3.FACETS_NOT_APPLICABLE``; every other key is a
    :class:`~impact_pipeline.v2.ram_v3.RAMParams` field."""
    from impact_pipeline.v2 import ram_v3

    opts = dict(options or {})
    mode = opts.pop("update", None)
    if mode is not None and mode != ram_v3.MODE:
        raise ProtocolOptionError(f"RAM update {mode!r}; RAM-PE v3 computes "
                                  f"{ram_v3.MODE!r}")
    facets = opts.pop("facets_not_applicable", None)
    if facets is not None and translate_facets(facets) != ram_v3.FACETS_NOT_APPLICABLE:
        raise ProtocolOptionError(
            f"RAM facets {translate_facets(facets)} differ from the estimator's "
            f"{ram_v3.FACETS_NOT_APPLICABLE}")
    ram_v3.RAMParams.from_mapping(opts)
    return opts


def pdi_v3_params(options: Mapping) -> Tuple[dict, dict]:
    """PDI v3 parameters and runner options from protocol options: ``mode``
    must be the estimator's, ``pdi_bearer`` is ``PDIParams.bearer``,
    ``access_module`` (runner) names the module declared as the access nodes;
    every other key is a :class:`~impact_pipeline.v2.pdi_v3.PDIParams`
    field."""
    from impact_pipeline.v2 import pdi_v3

    opts = dict(options or {})
    mode = opts.pop("mode", None)
    if mode is not None and mode != pdi_v3.MODE:
        raise ProtocolOptionError(f"PDI mode {mode!r}; PDI v3 computes {pdi_v3.MODE!r}")
    for proto_name, name in PDI_OPTION_NAMES.items():
        if proto_name in opts:
            value = opts.pop(proto_name)
            if name in opts and opts[name] != value:
                raise ProtocolOptionError(f"PDI {name} declared twice with different "
                                          "values")
            opts[name] = value
    runner = {"access_module": opts.pop("access_module", None)}
    pdi_v3.PDIParams.from_mapping(opts)
    return opts, runner


def split_runner_options(principle: str, options: Mapping) -> Tuple[dict, dict]:
    """``(estimator options, runner options)`` of a protocol's estimator
    block: the keys of :data:`RUNNER_OPTIONS` (checked against their admitted
    values) are the runner's, every other key the estimator's."""
    opts = dict(options or {})
    runner = {}
    for key, allowed in RUNNER_OPTIONS.get(principle, {}).items():
        if key in opts:
            val = opts.pop(key)
            if val not in allowed:
                raise ProtocolOptionError(f"{principle} {key}={val!r}; one of "
                                          f"{allowed}")
            runner[key] = val
    return opts, runner


def nas_v3_params(options: Mapping) -> dict:
    """NAS v3 parameters from protocol options: the protocol's NAS block uses
    the estimator's own names (:class:`~impact_pipeline.v2.nas_v3.NASParams`),
    so nothing is renamed; the runner's options (:data:`RUNNER_OPTIONS`) are
    left out; unknown keys are refused."""
    from impact_pipeline.v2 import nas_v3

    opts, _runner = split_runner_options("NAS", options)
    nas_v3.NASParams.from_mapping(opts)
    return opts


def iim_v5_params(options: Mapping) -> dict:
    """IIM v5 parameters from protocol options: ``tpm_estimator`` must name
    the estimator's TPM estimator and is not a parameter;
    ``coupling_timescale_sec`` and ``lag_timescale_sec`` set the runner's
    input basis; the runner's options (:data:`RUNNER_OPTIONS`) are left out;
    every other key is an :class:`~impact_pipeline.v2.iim_v5.IIMParams`
    field."""
    from impact_pipeline.v2 import iim_v5

    opts, _runner = split_runner_options("IIM", options)
    tpm = opts.pop("tpm_estimator", None)
    if tpm is not None and tpm != iim_v5.TPM_ESTIMATOR:
        raise ProtocolOptionError(f"IIM tpm_estimator {tpm!r}; IIM v5 computes "
                                  f"{iim_v5.TPM_ESTIMATOR!r}")
    opts.pop("coupling_timescale_sec", None)
    opts.pop("lag_timescale_sec", None)
    iim_v5.IIMParams.from_mapping(opts)
    return opts


def v1_params(principle: str, options: Mapping) -> dict:
    """v1 estimator parameters (one block of ``export.BENCH_ESTIMATOR_PARAMS``
    overrides) from protocol options: mode keys must equal the bench's fixed
    modes, pipeline-only options are skipped (as ``export.protocol_params``
    does for v1 protocols)."""
    from impact_pipeline.bench import export as X

    opts = dict(options or {})
    out = {}
    for key, val in opts.items():
        if key in X._BENCH_MODE_KEYS.get(principle, ()):
            want = X.OPTIONAL_MODES.get(principle, {}).get(key)
            if val != want:
                raise ProtocolOptionError(f"{principle} {key}={val!r}; the v1 path "
                                          f"computes {want!r}")
            continue
        pipeline_only = X._PIPELINE_ONLY_OPTIONS.get(principle, ())
        if key in pipeline_only or key == "report_cut_modes":
            continue
        out[key] = val
    return out


# --------------------------------------------------------------------------
# views and system builders
# --------------------------------------------------------------------------
VIEWS: Dict[str, D.ViewSpec] = {D.SOURCE_VIEW: D.SOURCE}
SYSTEM_BUILDERS: Dict[str, Callable] = {}
PROTOCOL_DRAFTS: Dict[str, Callable] = {}
# config blocks a design module adds to its records: {module: {key: fn}}
RECORD_CONFIG: Dict[str, Dict[str, Callable]] = {}
# keys of a record's config that the runner writes itself
RUNNER_CONFIG_KEYS = ("builder", "params", "tags", "held_out", "design_module",
                      "system_generator_version", "system_config_sha256",
                      "null_seed", "runner_version", "settings")
_LOADED_MODULES: set = set()


def register_view(view: D.ViewSpec, replace: bool = False) -> None:
    if not isinstance(view, D.ViewSpec):
        raise TypeError("a view is a designs_v2.ViewSpec")
    if view.name in VIEWS and not replace and VIEWS[view.name] != view:
        raise ValueError(f"view {view.name!r} is already registered")
    VIEWS[view.name] = view


def register_system_builder(name: str, fn: Callable, replace: bool = False) -> None:
    if name in SYSTEM_BUILDERS and not replace and SYSTEM_BUILDERS[name] is not fn:
        raise ValueError(f"system builder {name!r} is already registered")
    SYSTEM_BUILDERS[name] = fn


def load_design_module(name: Optional[str]) -> None:
    """Load a design module and register its system builders, views and
    protocol drafts (idempotent; every worker does it for its tasks)."""
    if name is None or name in _LOADED_MODULES:
        return
    mod = D.load_module(name)
    for bname, fn in dict(getattr(mod, "SYSTEM_BUILDERS", {}) or {}).items():
        register_system_builder(bname, fn)
    for view in dict(getattr(mod, "VIEWS", {}) or {}).values():
        register_view(view)
    drafts = dict(getattr(mod, "PROTOCOL_DRAFTS", {}) or {})
    taken = sorted(k for k, fn in drafts.items()
                   if str(k).split("+", 1)[0] in FAMILY_PROTOCOLS
                   or PROTOCOL_DRAFTS.get(k, fn) is not fn)
    if taken:
        raise RunPolicyError(f"design module {name}: protocol drafts {taken} are "
                             "already declared by the runner or another module")
    PROTOCOL_DRAFTS.update(drafts)
    extra = dict(getattr(mod, "RECORD_CONFIG", {}) or {})
    clash = sorted(set(extra) & set(RUNNER_CONFIG_KEYS))
    if clash:
        raise RunPolicyError(f"design module {name}: record config keys {clash} are "
                             "the runner's")
    RECORD_CONFIG[name] = extra
    _LOADED_MODULES.add(name)


def _agent_config(params: Mapping):
    from impact_pipeline.bench import adversarial_v2 as A2
    from impact_pipeline.bench.generators import config_from_dict

    over = {}
    if params.get("preset"):
        over.update(A2.config_preset(params["preset"]))
    over.update(dict(params.get("config") or {}))
    return config_from_dict(over) if over else None


def build_catalogue_system(task: D.TaskSpec):
    """A witness or adversary of the v2 catalogue (``params``: ``id``,
    ``family`` A or C, optional ``variant``, ``preset``, ``config``)."""
    from impact_pipeline.bench import adversarial_v2 as A2

    p = task.params
    return A2.build_system(p["id"], task.seed, p.get("family", "A"),
                           variant=p.get("variant"), replicate=task.replicate,
                           config=p.get("config"), preset=p.get("preset"))


def build_agent_system(task: D.TaskSpec):
    """The family-A (``family`` A) or family-C (C) agent at ``knobs``, with an
    optional ``preset`` and ``config`` (sweep levels, factorial cells, RAM-only
    doses)."""
    from impact_pipeline.bench.factorial import FAMILY_GENERATOR
    from impact_pipeline.bench.generators import knobs_from_dict, make_system

    p = task.params
    fam = str(p.get("family", "A")).upper()
    return make_system(FAMILY_GENERATOR[fam], knobs_from_dict(p.get("knobs")),
                       _agent_config(p), task.seed, replicate=task.replicate)


register_system_builder("catalogue", build_catalogue_system)
register_system_builder("agent", build_agent_system)

# Oracle keys the recording device reads beside a catalogue channel's own key
# (the catalogue's RECORDABLE_CHANNELS): the device recognises a continuous
# driver by the adversary label next to it.
_RECORDED_ORACLE_EXTRA_KEYS = {"driver": ("adversarial",)}


def recorded_inputs(system, task: D.TaskSpec):
    """The recording device's table of a task's system. For a catalogue
    system only the channels the catalogue lists as logged
    (``recorded_inputs``, read through the catalogue's channel-to-oracle
    map) are visible to the device; other systems are recorded as the
    device finds them."""
    from impact_pipeline.v2 import declared_inputs as DI

    if task.builder == "catalogue":
        from impact_pipeline.bench import adversarial_v2 as A2
        from impact_pipeline.bench.generators import BenchSystem

        entry = A2.get_entry(task.params["id"])
        keep = set()
        for ch in entry.get("recorded_inputs") or ():
            keep.add(A2.RECORDABLE_CHANNELS[ch][0])
            keep.update(_RECORDED_ORACLE_EXTRA_KEYS.get(ch, ()))
        oracle = {k: v for k, v in (system.oracle or {}).items() if k in keep}
        system = BenchSystem(ts=system.ts, events=system.events, meta=system.meta,
                             oracle=oracle, rest_ts=None)
    return DI.record_inputs(system)


# --------------------------------------------------------------------------
# protocols
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedProtocol:
    """A protocol of a run: its key, the :class:`ProtocolV3`, where it came from
    (``generated``, ``draft`` or ``override``) and its file."""

    key: str
    protocol: object
    source: str
    path: Optional[str] = None

    @property
    def hash(self) -> str:
        return self.protocol.hash

    def to_payload(self) -> dict:
        return {"key": self.key, "protocol": self.protocol.to_dict(),
                "source": self.source, "path": self.path, "hash": self.hash}

    @classmethod
    def from_payload(cls, payload: Mapping) -> "ResolvedProtocol":
        from impact_pipeline import evidence_v2 as E

        proto = E.ProtocolV3.from_dict(payload["protocol"])
        if payload.get("hash") and proto.hash != payload["hash"]:
            raise RunPolicyError(f"protocol {payload['key']}: hash changed in transit")
        return cls(payload["key"], proto, payload["source"], payload.get("path"))


def protocol_path(key: str, directory=None) -> Path:
    """The generated file of a protocol key (one naming rule for the runner,
    the evaluator and the integrity audit:
    :func:`impact_pipeline.v2.hypothesis_engine.protocol_file_name`)."""
    d = Path(directory) if directory is not None else GENERATED_DIR
    return d / _names.protocol_file_name(key)


def draft_payload(key: str, declaration_id: str,
                  estimator_options: Optional[Mapping] = None) -> dict:
    """The payload of a development draft derived from the template (module
    docstring): the protocol ``key`` in its name, the declaration, ``exact``
    among the IIM SE methods, the reference ``pending`` and the template's
    estimator blocks updated with ``estimator_options`` (``{principle:
    options}``). Every draft of the runner and of the design modules is
    built here."""
    from impact_pipeline import evidence_v2 as E

    payload = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    payload["name"] = f"mpc-bench-v2-{key}-draft"
    payload["shared_inputs_declaration"] = {
        "id": declaration_id, "shared_inputs": E.DECLARATION_LEVELS[declaration_id]}
    methods = dict(payload.get("se_methods") or {})
    methods["IIM"] = sorted(set(methods.get("IIM") or ()) | {E.SE_METHOD_EXACT})
    payload["se_methods"] = methods
    payload["reference"] = {
        "kind": "pending",
        "note": "draft of the v2 runner: anchors come from the development "
                "reference block; every component is UNDEFINED(INVALID_ANCHORS)",
    }
    est = {p: dict(v) for p, v in (payload.get("estimators") or {}).items()}
    for p, opts in dict(estimator_options or {}).items():
        est.setdefault(p, {}).update(dict(opts))
    payload["estimators"] = est
    return payload


def draft_protocol(key: str):
    """The draft of a family protocol (or of one of its forms) derived from the
    template (module docstring); a design module's draft rule
    (``PROTOCOL_DRAFTS``) for the protocols it declares."""
    from impact_pipeline import evidence_v2 as E

    base, form = D.split_protocol_key(key)
    if key in PROTOCOL_DRAFTS:
        payload = PROTOCOL_DRAFTS[key]()
        return E.ProtocolV3.from_dict(payload)
    if base not in FAMILY_PROTOCOLS:
        raise RunPolicyError(f"no protocol file and no draft rule for {key!r}")
    _family, decl = FAMILY_PROTOCOLS[base]
    return E.ProtocolV3.from_dict(
        draft_payload(key, decl, D.ESTIMATOR_FORMS[form].options))


def resolve_protocols(keys: Iterable[str], *, directory=None, allow_drafts: bool = True,
                      overrides: Optional[Mapping] = None
                      ) -> Dict[str, ResolvedProtocol]:
    """The protocols of a run: an override (a ProtocolV3, dict or path), else
    the generated file, else (when allowed) the draft."""
    from impact_pipeline import evidence_v2 as E

    overrides = dict(overrides or {})
    out = {}
    for key in sorted(set(keys)):
        if key in overrides:
            proto = overrides[key]
            if not isinstance(proto, E.ProtocolV3):
                proto = E.load_protocol(proto)
            if not isinstance(proto, E.ProtocolV3):
                raise RunPolicyError(f"protocol {key}: a v2 run needs a /3 protocol")
            out[key] = ResolvedProtocol(key, proto, "override")
            continue
        path = protocol_path(key, directory)
        if path.is_file():
            out[key] = ResolvedProtocol(key, E.ProtocolV3.from_json(path), "generated",
                                        str(path))
            continue
        if not allow_drafts:
            raise RunPolicyError(f"protocol {key}: no generated file at {path} "
                                 "(drafts are not allowed in this run)")
        out[key] = ResolvedProtocol(key, draft_protocol(key), "draft")
    return out


# --------------------------------------------------------------------------
# scorers
# --------------------------------------------------------------------------
@dataclass
class Member:
    """The estimator fields of one evidence item (a direction of a
    directional principle, or the principle itself)."""

    direction: Optional[str] = None
    estimate: Optional[float] = None
    null_mean: Optional[float] = None
    null_sd: Optional[float] = None
    n_null: int = 0
    null_family: Optional[str] = None
    se: Optional[float] = None
    se_df: Optional[float] = None
    se_method: Optional[str] = None
    defined: bool = True
    reason: Optional[str] = None
    exact: bool = False
    p_ind: Optional[float] = None
    content_bearer: Optional[str] = None

    def to_dict(self) -> dict:
        return _sanitize(dataclasses.asdict(self))


@dataclass
class ScorerOutput:
    """What a scorer returns: the estimator id (``compute_<P>:<mode>@<version>``),
    one member per evidence item and the estimator's details."""

    estimator_id: str
    members: List[Member]
    details: dict = field(default_factory=dict)


class Scorer:
    """Base of the estimator adapters. ``directions``: the evidence directions
    the scorer produces (None: one non-directional item);
    ``uses_declaration``: the output depends on the input declaration;
    ``uses_reference``: the output depends on the protocol's anchor of the
    principle (so it is not reused under another protocol's anchor);
    ``declares_observation_model``: the estimator declares not-applicable
    observation models itself."""

    principle: str = ""
    version: str = ""
    kind: str = ""
    directions: Optional[Tuple[str, ...]] = None
    uses_declaration: bool = False
    uses_reference: bool = False
    declares_observation_model: bool = False

    def available(self) -> bool:
        return True

    def check_options(self, options: Mapping) -> None:
        """Raise :class:`ProtocolOptionError` when the protocol's options
        contradict the estimator (checked before a run starts)."""

    def score(self, ctx: "ScoringContext") -> ScorerOutput:  # pragma: no cover
        raise NotImplementedError


class RamV3Scorer(Scorer):
    principle, version, kind = "RAM", "ram-v3-2026.10", "ram_v3"
    declares_observation_model = True
    # the anchor enters the estimator's absent_reachable descriptor
    uses_reference = True

    def check_options(self, options):
        ram_v3_params(options)

    def score(self, ctx):
        from impact_pipeline.v2 import ram_v3

        params = ram_v3_params(ctx.options)
        view = ctx.bearer()
        res = ram_v3.compute_ram_v3(
            view["ts"], ctx.system.dt, ctx.system.events,
            observation_model=ctx.observation_model, params=params,
            reference=ctx.reference_excess())
        f = ram_v3.component_fields(res)
        m = Member(estimate=f["estimate"], null_mean=f["null_mean"],
                   null_sd=f["null_sd"], n_null=f["n_null"],
                   null_family=f["null_family"], se=f["se"], se_df=f["se_df"],
                   se_method=f["se_method"], defined=bool(res["defined"]),
                   reason=res["reason"], exact=False)
        return ScorerOutput(res.get("estimator_id") or ram_v3.ESTIMATOR_ID, [m],
                            f["details"])


class PdiV3Scorer(Scorer):
    principle, version, kind = "PDI", "pdi-v3-2026.10", "pdi_v3"
    declares_observation_model = True

    def check_options(self, options):
        pdi_v3_params(options)

    def score(self, ctx):
        from impact_pipeline.v2 import pdi_v3

        params, runner = pdi_v3_params(ctx.options)
        view = ctx.bearer()
        ws = view["workspace_nodes"]
        if runner["access_module"]:
            ws = ctx.module_nodes(runner["access_module"], view)
        seed = ctx.settings.pdi_kmeans_seed
        seed = ctx.null_seed if seed is None else int(seed)
        partition = bool(ctx.task.tags.get("pdi_partition"))
        res = pdi_v3.compute_pdi_v3(
            view["ts"], ws, params=params, seed=seed, null_seed=ctx.null_seed,
            observation_model=ctx.observation_model, dt=ctx.system.dt,
            return_partition=partition)
        f = pdi_v3.component_fields(res)
        details = dict(f["details"])
        details["kmeans_seed"] = seed
        if runner["access_module"]:
            details["declared_access_module"] = runner["access_module"]
        if partition:
            details["oracle_windows"] = pdi_oracle_windows(ctx.system, params)
            details["ami_ignition_full_bearer"] = _ami_ignition(
                (details.get("full_bearer") or {}).get("partition"),
                details["oracle_windows"])
        m = Member(estimate=f["estimate"], null_mean=f["null_mean"],
                   null_sd=f["null_sd"], n_null=f["n_null"],
                   null_family=f["null_family"], se=f["se"], se_df=f["se_df"],
                   se_method=f["se_method"], defined=bool(res["defined"]),
                   reason=res["reason"], exact=False,
                   content_bearer=details.get("counted_bearer"))
        return ScorerOutput(res.get("estimator_id") or pdi_v3.ESTIMATOR_ID, [m],
                            details)


class V1InMemoryScorer(Scorer):
    """A v1 estimator through the unchanged v1 in-memory path."""

    kind = "v1_in_memory"

    def __init__(self, principle: str, version: str):
        self.principle, self.version = principle, version

    def check_options(self, options):
        v1_params(self.principle, options)

    def score(self, ctx):
        from impact_pipeline import evidence as v1
        from impact_pipeline.bench import export as X

        params = v1_params(self.principle, ctx.options)
        res = X.run_in_memory(
            ctx.system, metrics=[self.principle],
            params={self.principle: params} if params else None,
            null_surrogates=ctx.settings.v1_null_surrogates,
            null_seed=ctx.null_seed, bearer_mode=ctx.scoring.bearer_mode,
            se_groups=ctx.settings.v1_se_groups)
        c = res["components"][self.principle]
        modes = res.get("estimator_modes", {}).get(self.principle)
        estimator = v1.estimator_id(self.principle,
                                    X._component_mode(self.principle, modes),
                                    self.version)
        se_df = X._se_df_of(c)
        groups = ctx.settings.v1_se_groups
        se_method = (V1_SE_METHODS.get(self.principle)
                     if groups == V1_SE_GROUPS and c.get("se") is not None else None)
        m = Member(estimate=_finite(c.get("estimate")),
                   null_mean=_finite(c.get("null_mean")),
                   null_sd=_finite(c.get("null_sd")), n_null=int(c.get("n_null") or 0),
                   null_family=c.get("null_family"), se=_finite(c.get("se")),
                   se_df=se_df, se_method=se_method, defined=bool(c.get("defined")),
                   reason=c.get("reason"), exact=bool(c.get("exact", False)))
        details = {k: v for k, v in c.items() if k != "se_replicates"}
        details["se_replicates"] = c.get("se_replicates")
        details["estimator_modes"] = modes
        details["path"] = "impact_pipeline.bench.export.run_in_memory"
        return ScorerOutput(estimator, [m], _sanitize(details))


class NasV3Scorer(Scorer):
    """NAS v3 (module docstring): one item per direction."""

    principle, version, kind = "NAS", "nas-v3-2026.10", "nas_v3"
    directions = ("receive", "return")
    uses_declaration = True

    def check_options(self, options):
        nas_v3_params(options)

    def score(self, ctx):
        from impact_pipeline.v2 import nas_v3

        p = nas_v3.NASParams.from_mapping(nas_v3_params(ctx.options))
        _opts, runner = split_runner_options("NAS", ctx.options)
        # tau_c of the gate: the protocol's, checked against the generator
        p = p.replace(coupling_timescale_sec=nas_v3.system_coupling_timescale(
            ctx.system, p))
        view = ctx.bearer()
        hub_name, hub, blocks = ctx.nas_blocks(view)
        # a forward-arm admission run scores the mixed observation: its
        # scoring is the admission test (labelled 'overridden' by NAS v3)
        res = nas_v3.compute_nas_v3(
            view["ts"], dt=float(ctx.system.dt), hub=hub, blocks=blocks,
            inputs=ctx.declared(), params=p, seed=int(ctx.task.seed),
            null_seed=ctx.null_seed, hub_name=hub_name,
            observation=ctx.view.observation,
            override_observation_gate=(runner.get("observation_gate")
                                       == OBSERVATION_GATE_ADMISSION_RUN))
        out = output_from_result(res, nas_v3, self)
        # both directions as the estimator reports them (z, excess, null
        # draws, jackknife replicates), for the evaluator and the audit
        out.details["directions"] = _sanitize(res.get("directions") or {})
        return out


class MacroGrainTooLargeError(ValueError):
    """An IIM v5 macro grain with more macro nodes than the run's declared
    cap (:attr:`RunSettings.iim_max_macro_nodes`)."""


class IimV5Scorer(Scorer):
    """
    IIM v5 (module docstring): the cut modes of the protocol (the primary
    and the reported ones), on the view's declared macro grain or the grain
    the runner option ``macro_nodes`` names (:data:`RUNNER_OPTIONS`); a
    forward-arm admission run (``observation_gate = admission_run``) scores
    a mixed observation as admitted.

    Every cut mode of a run is computed on the same transition pairs,
    surrogates and resamples, so the values of a cut mode do not depend on
    the other cut modes of the run. The runner therefore computes, in one
    run per inputs, every cut mode the simulation's scorings of those inputs
    record (:func:`iim_cut_modes_read`), and each scoring reads its own
    protocol's cut modes from that run
    (:func:`impact_pipeline.v2.iim_v5.select_cut_modes`): the bidirectional
    form (``iim_bidirectional``) reads the cut its primary scoring already
    reports, its record naming the scoring the run was computed in, and
    every record holds exactly what its own run would hold.

    A grain with more macro nodes than the declared cap
    (:attr:`RunSettings.iim_max_macro_nodes`, 4: the grain of every Tier-A
    substrate) is refused before any statistic is computed
    (:class:`MacroGrainTooLargeError`, an estimator error of that component):
    the cost of a transition-level statistic grows with the number of macro
    states and ordered bipartitions (a five-node grain costs about 28 times
    a four-node one), so a grain no design declares never runs for hours.
    """

    principle, version, kind = "IIM", "iim-v5-2026.10", "iim_v5"
    uses_declaration = True

    def check_options(self, options):
        iim_v5_params(options)

    @staticmethod
    def macro_nodes(ctx, view: Mapping, choice: str) -> Mapping:
        """The macro grain (rows of the bearer view) of a ``macro_nodes``
        choice."""
        meta = ctx.system.meta or {}
        if choice == MACRO_NODES_DECLARED:
            return view["macro_nodes"]
        if choice == MACRO_NODES_RANK_SAFE:
            if meta.get("iim_grain") != RANK_SAFE_GRAIN:
                raise ProtocolOptionError(
                    f"macro_nodes {choice!r}: the view declares the grain "
                    f"{meta.get('iim_grain')!r}, not the rank-safe clusters")
            return view["macro_nodes"]
        quad = (meta.get("forward_v2") or {}).get("iim_quadrants_v1")
        if not quad:
            raise ProtocolOptionError(f"macro_nodes {choice!r}: the view keeps no v1 "
                                      "electrode quadrants")
        pos = {g: k for k, g in enumerate(view["nodes"])}
        return {str(k): [pos[int(i)] for i in v if int(i) in pos]
                for k, v in quad.items()}

    def score(self, ctx):
        from impact_pipeline.v2 import iim_v5

        params = iim_v5_params(ctx.options)
        _opts, runner = split_runner_options("IIM", ctx.options)
        view = ctx.bearer()
        macro = self.macro_nodes(ctx, view, runner.get("macro_nodes",
                                                       MACRO_NODES_DECLARED))
        cap = ctx.settings.iim_max_macro_nodes
        if cap is not None and len(macro) > int(cap):
            raise MacroGrainTooLargeError(
                f"the IIM grain of view {ctx.view.name!r} ({ctx.scoring.bearer_mode} "
                f"bearer) has {len(macro)} macro nodes; the run's cap is {int(cap)}")
        admitted = runner.get("observation_gate") == OBSERVATION_GATE_ADMISSION_RUN
        lag = ctx.iim_lag()
        basis = ctx.basis("IIM")
        p = iim_v5.IIMParams.from_mapping(params)
        read = (ctx._cache.get(IIM_CUT_MODES_READ) or {}).get(
            iim_input_group(ctx.view.name, ctx.scoring.bearer_mode,
                            ctx.scoring.declaration_id, ctx.options))
        run_params = params
        if read:
            # the run computes the cut modes of every scoring of these inputs
            # (this scoring's primary first; ``read`` holds its own as well)
            extra = [c for c in iim_v5.CUT_MODES
                     if c != p.cut_mode and (c in read or c in p.report_cut_modes)]
            run_params = dict(params, report_cut_modes=extra)
        same_inputs = {k: v for k, v in p.to_dict().items()
                       if k not in ("cut_mode", "report_cut_modes")}
        key = ("iim_v5_run", ctx.view.name, ctx.scoring.bearer_mode,
               ctx.scoring.declaration_id, round(ctx.lag_timescale(), 12),
               json.dumps(same_inputs, sort_keys=True),
               json.dumps({k: [int(i) for i in v] for k, v in macro.items()}),
               int(lag), bool(admitted))
        res, derived_from, covered = None, None, False
        for run, where in ctx._cache.get(key, ()):
            if set(p.cut_modes) <= set(iim_v5.cut_modes_computed(run)):
                covered = True
                try:
                    res = iim_v5.select_cut_modes(run, p)
                except ValueError:
                    continue
                derived_from = where
                break

        def own_run():
            return iim_v5.compute_iim_v5(
                view["ts"], lag=lag, params=p.to_dict(), macro_nodes=macro,
                basis=basis, null_seed=ctx.null_seed,
                observation_stage=ctx.view.observation_stage,
                observation_admitted=admitted)

        if res is None and covered:
            # a run of these inputs computed this scoring's cut modes but is
            # undefined for a reason that counts every computed cut mode (too
            # few finite null draws): its own run decides
            res = own_run()
        elif res is None:
            run = iim_v5.compute_iim_v5(
                view["ts"], lag=lag, params=run_params, macro_nodes=macro, basis=basis,
                null_seed=ctx.null_seed, observation_stage=ctx.view.observation_stage,
                observation_admitted=admitted)
            ctx._cache.setdefault(key, []).append((run, ctx.scoring.scoring_id))
            # the record keeps this scoring's own cut modes, whatever else the
            # run computed for the other scorings of the simulation
            if tuple(iim_v5.cut_modes_computed(run)) == tuple(p.cut_modes):
                res = run
            else:
                try:
                    res = iim_v5.select_cut_modes(run, p)
                except ValueError:
                    # undefined for a reason that counts every computed cut
                    # mode: as above
                    res = own_run()
        out = output_from_result(res, iim_v5, self)
        if derived_from is not None:
            out.details["cut_modes_computed_in_scoring"] = derived_from
        if runner.get("macro_nodes"):
            out.details["macro_nodes"] = runner["macro_nodes"]
        return out


IIM_CUT_MODES_READ = "iim_cut_modes_read"


def iim_input_group(view: str, bearer_mode: str, declaration_id: str,
                    options: Mapping) -> tuple:
    """The inputs of an IIM v5 run up to its cut modes: the view, bearer,
    declaration and the protocol's IIM options without ``cut_mode`` and
    ``report_cut_modes``."""
    rest = {k: v for k, v in dict(options or {}).items()
            if k not in ("cut_mode", "report_cut_modes")}
    return (view, bearer_mode, declaration_id, json.dumps(rest, sort_keys=True))


def iim_cut_modes_read(task: D.TaskSpec, protocols: Mapping[str, "ResolvedProtocol"],
                       settings: "RunSettings" = None) -> Dict[tuple, set]:
    """``{input group: cut modes}``: the cut modes (primary and reported) of
    the task's IIM v5 scorings per :func:`iim_input_group`, every cut mode a
    record of the simulation keeps; one run per group computes them all."""
    from impact_pipeline.v2 import iim_v5

    out: Dict[tuple, set] = {}
    for spec in task.scorings:
        if "IIM" not in spec.principles or (settings is not None
                                            and not settings.wants("IIM")):
            continue
        rp = protocols.get(spec.protocol_key)
        if rp is None or rp.protocol.estimator_version_for("IIM") != (
                IimV5Scorer.version):
            continue
        opts = rp.protocol.estimator_options("IIM")
        try:
            cuts = iim_v5.IIMParams.from_mapping(iim_v5_params(opts)).cut_modes
        except (ValueError, TypeError):
            continue  # the scorer reports the option error as the component's
        key = iim_input_group(spec.view, spec.bearer_mode, spec.declaration_id, opts)
        out.setdefault(key, set()).update(cuts)
    return out


class ModuleScorer(Scorer):
    """An estimator module of the v2 round (contract in the module
    docstring)."""

    kind = "module"
    HOOK = "score_bench_v2"
    # a hook reads the whole scoring context, the protocol's anchors included
    uses_reference = True

    def __init__(self, principle, version, module, function, *, uses_declaration,
                 directions=None, basis_estimator=None):
        self.principle, self.version = principle, version
        self.module, self.function = module, function
        self.uses_declaration = bool(uses_declaration)
        self.directions = None if directions is None else tuple(directions)
        self.basis_estimator = basis_estimator

    def available(self) -> bool:
        if self.module in sys.modules:
            return True
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            return False

    def _module(self):
        try:
            return importlib.import_module(self.module)
        except ModuleNotFoundError as exc:
            if exc.name == self.module:
                raise EstimatorUnavailableError(
                    f"{self.version}: module {self.module} is not merged yet") from None
            raise

    def score(self, ctx):
        mod = self._module()
        hook = getattr(mod, self.HOOK, None)
        if callable(hook):
            result = hook(ctx)
        else:
            fn = getattr(mod, self.function, None)
            if not callable(fn):
                raise EstimatorUnavailableError(
                    f"{self.module} defines neither {self.function} nor {self.HOOK}")
            result = fn(**call_kwargs(fn, ctx.candidates(self)))
        return output_from_result(result, mod, self)


SCORERS: Dict[str, Scorer] = {}


def register_scorer(scorer: Scorer, replace: bool = False) -> None:
    if scorer.version in SCORERS and not replace:
        raise ValueError(f"a scorer for {scorer.version} is registered")
    SCORERS[scorer.version] = scorer


for _s in (
    RamV3Scorer(),
    PdiV3Scorer(),
    V1InMemoryScorer("SRPI", "srpi-v2-2026.09"),
    V1InMemoryScorer("NAS", "nas-v2-2026.09"),
    V1InMemoryScorer("IIM", "iim-v4-2026.09"),
    V1InMemoryScorer("PDI", "pdi-v2-2026.09"),
    V1InMemoryScorer("RAM", "ram-v2-2026.09"),
    NasV3Scorer(),
    IimV5Scorer(),
    ModuleScorer("SRPI", "srpi-v3-2026.10", "impact_pipeline.v2.srpi_v3",
                 "compute_srpi_v3", uses_declaration=False),
):
    register_scorer(_s)


@contextlib.contextmanager
def scorer_override(scorer: Scorer):
    """Replace the scorer of one estimator version for the duration of the
    block (tests and stubs; in-process runs only)."""
    old = SCORERS.get(scorer.version)
    SCORERS[scorer.version] = scorer
    try:
        yield scorer
    finally:
        if old is None:
            SCORERS.pop(scorer.version, None)
        else:
            SCORERS[scorer.version] = old


def scorer_for(version: Optional[str], principle: str) -> Scorer:
    if version is None:
        raise RunPolicyError(f"the protocol names no estimator version for {principle}")
    sc = SCORERS.get(version)
    if sc is None:
        raise RunPolicyError(f"no scorer for estimator version {version!r}")
    if sc.principle != principle:
        raise RunPolicyError(f"estimator version {version!r} is a {sc.principle} "
                             f"estimator, not {principle}")
    return sc


def call_kwargs(fn: Callable, candidates: Mapping[str, Callable]) -> dict:
    """The keyword arguments of ``fn`` that the runner supplies (lazy
    candidates, evaluated only when named by the signature). A required
    parameter the runner does not know raises ``TypeError``."""
    sig = inspect.signature(fn)
    out = {}
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name in candidates:
            out[name] = candidates[name]()
        elif p.default is p.empty:
            raise TypeError(f"{getattr(fn, '__module__', '?')}."
                            f"{getattr(fn, '__name__', fn)} needs {name!r}, which the "
                            f"runner does not supply (define score_bench_v2(context))")
    return out


def _member_from(d: Mapping, *, direction=None,
                 top: Optional[Mapping] = None) -> Member:
    top = top or {}
    det = d.get("details") if isinstance(d.get("details"), Mapping) else {}
    defined = d.get("defined", top.get("defined"))
    est = _finite(d.get("estimate"))
    if defined is None:
        defined = est is not None
    reason = d.get("reason", top.get("reason") if not defined else None)
    p_ind = d.get("p_ind", det.get("p_ind", top.get("p_ind")))
    return Member(
        direction=direction, estimate=est, null_mean=_finite(d.get("null_mean")),
        null_sd=_finite(d.get("null_sd")), n_null=int(d.get("n_null") or 0),
        null_family=d.get("null_family"), se=_finite(d.get("se")),
        se_df=_finite(d.get("se_df")), se_method=d.get("se_method"),
        defined=bool(defined), reason=None if defined else reason,
        exact=bool(d.get("exact", False)), p_ind=_finite(p_ind),
        content_bearer=d.get("content_bearer", det.get("counted_bearer")))


def output_from_result(result, module, scorer: Scorer) -> ScorerOutput:
    """Map an estimator result onto a :class:`ScorerOutput`: a ScorerOutput is
    taken as it is; a mapping gives one member (through the module's
    ``component_fields`` when it has one) or, with ``directions`` (or
    ``members``), one member per direction."""
    if isinstance(result, ScorerOutput):
        return result
    if not isinstance(result, Mapping):
        raise TypeError(f"{scorer.version}: the estimator returned "
                        f"{type(result).__name__}, not a mapping")
    mode = result.get("mode") or "default"
    est_id = result.get("estimator_id") or (
        f"compute_{scorer.principle}:{mode}@{scorer.version}")
    fields_fn = getattr(module, "component_fields", None)
    per = result.get("directions", result.get("members"))
    if isinstance(per, Mapping) and per:
        members, sub_details = [], {}
        for name, sub in per.items():
            if not isinstance(sub, Mapping):
                raise TypeError(f"{scorer.version}: direction {name!r} is not "
                                "a mapping")
            d = DIRECTION_ALIASES.get(str(name).lower(), str(name))
            members.append(_member_from(sub, direction=d, top=result))
            sub_details[d] = sub.get("details")
        details = dict(result.get("details") or {})
        details["directions"] = sub_details
    else:
        f = fields_fn(result) if callable(fields_fn) else result
        members = [_member_from(f, top=result)]
        details = dict(f.get("details") or result.get("details") or {})
    return ScorerOutput(str(est_id), members, _sanitize(details))


# --------------------------------------------------------------------------
# scoring context
# --------------------------------------------------------------------------
class ScoringContext:
    """
    Everything a scorer may read for one (task, scoring, principle): the task
    and scoring specs, the simulated ``source`` and the observed ``system``
    (the view applied), the view, the protocol and its key, the principle's
    protocol ``options``, the run settings, the registry, the null seed, the
    observation model (the task's family) and lazily built, cached inputs:
    the recording device's table (:meth:`recorded`), the declared inputs
    (:meth:`declared`), the input basis (:meth:`basis`), the bearer view
    (:meth:`bearer`) and the periphery blocks (:meth:`blocks`).
    """

    def __init__(self, task, scoring, source, system, view, resolved, principle,
                 settings, registry, cache):
        self.task, self.scoring = task, scoring
        self.source, self.system, self.view = source, system, view
        self.resolved = resolved
        self.protocol = resolved.protocol
        self.protocol_key = resolved.key
        self.principle = principle
        self.options = self.protocol.estimator_options(principle)
        self.settings, self.registry = settings, registry
        self._cache = cache
        self.null_seed = null_seed(task.seed, task.replicate)
        self.observation_model = task.family
        # cache keys of the input bases this context built or read
        self.used_bases: List[tuple] = []

    # -- inputs ---------------------------------------------------------------
    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def recorded(self):
        return self._memo(("recorded", self.view.name),
                          lambda: recorded_inputs(self.system, self.task))

    def declared(self):
        from impact_pipeline.v2 import declared_inputs as DI

        return self._memo(("declared", self.view.name, self.scoring.declaration_id),
                          lambda: DI.declare(self.recorded(),
                                             self.scoring.declaration_id))

    def tau_c(self) -> float:
        """The coupling time scale of this scoring: the protocol option
        ``coupling_timescale_sec`` (the family value), else the generator
        constant."""
        val = self.options.get("coupling_timescale_sec")
        if val is not None:
            return float(val)
        return self.system_tau_c()

    def lag_timescale(self) -> float:
        """The time scale of the lags and filtered input copies: the protocol
        option ``lag_timescale_sec`` (the ``tau_c`` sensitivity forms), else
        :meth:`tau_c`. The resolvability gate always uses ``tau_c``."""
        val = self.options.get("lag_timescale_sec")
        if val is not None:
            return float(val)
        return self.tau_c()

    def coupling_timescale(self):
        """The generator constant's coupling time scale of the observed
        system (:func:`~impact_pipeline.v2.declared_inputs.coupling_timescale`),
        whatever ``tau_c`` the protocol declares."""
        from impact_pipeline.v2 import declared_inputs as DI

        return DI.coupling_timescale(self.system)

    def system_tau_c(self) -> float:
        return self.coupling_timescale().tau_c_sec

    def lag_plan(self):
        """The lags and filtered copies at :meth:`lag_timescale` (its
        ``resolved`` refers to that time scale; the estimators' gate uses
        :meth:`tau_c`)."""
        from impact_pipeline.v2 import declared_inputs as DI

        return DI.lag_plan(tau_c=self.lag_timescale(), dt=self.system.dt)

    def iim_lag(self) -> int:
        """The IIM transition lag in samples: the system's declared
        ``iim_lag_samples`` (default 2), as the v1 path reads it."""
        return int((self.system.meta or {}).get("iim_lag_samples", 2))

    def basis(self, estimator: str = "NAS"):
        from impact_pipeline.v2 import declared_inputs as DI

        tau = self.lag_timescale()
        key = ("basis", self.view.name, self.scoring.declaration_id, estimator,
               round(tau, 12))
        if key not in self.used_bases:
            self.used_bases.append(key)
        return self._memo(key, lambda: DI.input_basis(self.declared(), tau_c=tau,
                                                      estimator=estimator))

    def bearer(self, principle: Optional[str] = None) -> dict:
        from impact_pipeline.bench import export as X

        p = principle or self.principle
        return self._memo(("bearer", self.view.name, p, self.scoring.bearer_mode),
                          lambda: X.bearer_view(self.system, p,
                                                self.scoring.bearer_mode))

    def blocks(self, principle: Optional[str] = None) -> Dict[str, List[int]]:
        """The declared periphery blocks (every module except the one of the
        workspace nodes) as row indices of the bearer view."""
        view = self.bearer(principle)
        pos = {g: k for k, g in enumerate(view["nodes"])}
        ws = set(self.system.meta.get("workspace_nodes") or [])
        out = {}
        for name, nodes in (self.system.meta.get("modules") or {}).items():
            nodes = [int(i) for i in nodes]
            if ws and set(nodes) & ws:
                continue
            local = [pos[i] for i in nodes if i in pos]
            if local:
                out[str(name)] = local
        return out

    def nas_blocks(self, view: Optional[dict] = None):
        """``(hub_name, hub rows, periphery blocks)`` of NAS v3: the system's
        declared blocks (:func:`impact_pipeline.v2.nas_v3.declared_blocks`,
        with the bearer's workspace as the hub) as rows of the bearer view."""
        from impact_pipeline.v2 import nas_v3

        view = view or self.bearer("NAS")
        nodes = list(view["nodes"])
        pos = {g: k for k, g in enumerate(nodes)}
        meta = dict(self.system.meta or {})
        meta["workspace_nodes"] = [nodes[i] for i in (view["workspace_nodes"] or [])]
        hub_name, hub, blocks = nas_v3.declared_blocks(meta)
        local = {k: [pos[i] for i in v if i in pos] for k, v in blocks.items()}
        return (hub_name, [pos[i] for i in hub if i in pos],
                {k: v for k, v in local.items() if v})

    def module_nodes(self, module: str, view: Optional[dict] = None) -> List[int]:
        view = view or self.bearer()
        pos = {g: k for k, g in enumerate(view["nodes"])}
        nodes = (self.system.meta.get("modules") or {}).get(module)
        if not nodes:
            raise ValueError(f"the system has no module {module!r}")
        return [pos[int(i)] for i in nodes if int(i) in pos]

    def macro_ts(self, principle: Optional[str] = None) -> np.ndarray:
        view = self.bearer(principle)
        return np.stack([view["ts"][np.asarray(v, dtype=int)].mean(axis=0)
                         for v in view["macro_nodes"].values()])

    def identifiability(self) -> Optional[dict]:
        return identifiability(self.scoring.declaration_id, self.view)

    def regime(self) -> dict:
        """The registry regime keys of this scoring (:func:`view_regime`)."""
        return view_regime(self.view, self.system)

    def reference_excess(self) -> Optional[float]:
        ref = self.protocol.reference_for(self.principle)
        if ref and ref.get("reference_scale") == "excess":
            return float(ref["reference"])
        return None

    def candidates(self, scorer: Scorer) -> Dict[str, Callable]:
        """Lazy keyword arguments a module estimator may name (see
        :func:`call_kwargs`)."""
        p = self.principle
        est = getattr(scorer, "basis_estimator", None) or "NAS"
        ts = lambda: self.bearer(p)["ts"]  # noqa: E731
        ws = lambda: self.bearer(p)["workspace_nodes"]  # noqa: E731
        blocks = lambda: self.blocks(p)  # noqa: E731
        declared = self.declared
        basis = lambda: self.basis(est)  # noqa: E731
        opts = lambda: dict(self.options)  # noqa: E731
        seed = lambda: self.null_seed  # noqa: E731
        return {
            "ts": ts, "x": ts, "data": ts, "timeseries": ts,
            "dt": lambda: self.system.dt, "tr": lambda: self.system.dt,
            "fs": lambda: 1.0 / self.system.dt,
            "workspace_nodes": ws, "hub_nodes": ws, "hub": ws,
            "blocks": blocks, "periphery_blocks": blocks, "periphery": blocks,
            "modules": blocks,
            "macro_nodes": lambda: self.bearer(p)["macro_nodes"],
            "macro_ts": lambda: self.macro_ts(p),
            "recorded": self.recorded, "declared": declared,
            "declared_inputs": declared, "inputs": declared,
            "declaration": declared, "basis": basis, "input_basis": basis,
            "tau_c": self.tau_c, "lag_plan": self.lag_plan,
            "lag_timescale": self.lag_timescale, "lag": self.iim_lag,
            # the generator constant, apart from a form's sensitivity tau_c
            # (for an estimator that judges sampling resolution on the
            # system's own time scale)
            "system_tau_c": self.system_tau_c,
            "coupling_timescale": self.coupling_timescale,
            "params": opts, "options": opts,
            "seed": seed, "null_seed": seed, "random_state": seed,
            "observation": lambda: self.view.observation,
            "observation_stage": lambda: self.view.observation_stage,
            "observation_model": lambda: self.observation_model,
            "view": lambda: self.view.name, "registry": lambda: self.registry,
            "regime": self.regime,
            "identifiability": self.identifiability,
            "events": lambda: self.system.events, "system": lambda: self.system,
            "meta": lambda: self.system.meta,
        }


def view_regime(view: D.ViewSpec, system) -> dict:
    """The registry regime keys of an observed system: those the forward
    layer attached to the view's system (``meta['forward_v2']['regime']``,
    which depend on the task's regime), else the view's static keys."""
    block = (getattr(system, "meta", None) or {}).get("forward_v2")
    if isinstance(block, Mapping) and isinstance(block.get("regime"), Mapping):
        return dict(block["regime"])
    return dict(view.regime)


def identifiability(declaration_id: str, view: D.ViewSpec) -> Optional[dict]:
    """The identifiability record of a component (``hub_privileged`` not
    tested in Tier A); None for declarations outside the bench vocabulary."""
    from impact_pipeline import evidence_v2 as E

    level = E.DECLARATION_LEVELS.get(declaration_id)
    if level is None:
        return None
    return {"shared_inputs": level, "observation": view.observation,
            "hub_privileged": "not_tested"}


def adjusted_mutual_info(labels_a, labels_b) -> Optional[float]:
    """Adjusted mutual information of two labellings of the same items
    (arithmetic-mean normalisation, expected mutual information under the
    hypergeometric model of fixed marginals; Vinh, Epps and Bailey 2010).
    1 for identical partitions up to relabelling, about 0 for independent
    ones; 1 when both have a single label, 0 when only one of them has.
    None for empty or unequal-length input."""
    from scipy.special import gammaln

    a, b = np.asarray(labels_a), np.asarray(labels_b)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        return None
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    ka, kb = int(ia.max()) + 1, int(ib.max()) + 1
    if ka == 1 and kb == 1:
        return 1.0
    if ka == 1 or kb == 1:
        return 0.0
    n = float(a.size)
    table = np.zeros((ka, kb))
    np.add.at(table, (ia, ib), 1.0)
    ra, cb = table.sum(axis=1), table.sum(axis=0)
    nz = table > 0
    expected = np.outer(ra, cb)[nz]
    mi = float(np.sum(table[nz] / n * np.log(n * table[nz] / expected)))
    h_a = float(-np.sum(ra / n * np.log(ra / n)))
    h_b = float(-np.sum(cb / n * np.log(cb / n)))
    emi = 0.0
    for x in ra:
        for y in cb:
            lo, hi = int(max(1, x + y - n)), int(min(x, y))
            if lo > hi:
                continue
            nij = np.arange(lo, hi + 1, dtype=float)
            logp = (gammaln(x + 1) + gammaln(y + 1) + gammaln(n - x + 1)
                    + gammaln(n - y + 1) - gammaln(n + 1) - gammaln(nij + 1)
                    - gammaln(x - nij + 1) - gammaln(y - nij + 1)
                    - gammaln(n - x - y + nij + 1))
            emi += float(np.sum(nij / n * np.log(n * nij / (x * y)) * np.exp(logp)))
    denom = 0.5 * (h_a + h_b) - emi
    eps = np.finfo(float).eps
    denom = min(denom, -eps) if denom < 0 else max(denom, eps)
    return float((mi - emi) / denom)


def _ami_ignition(partition: Optional[Mapping], windows: Mapping) -> Optional[float]:
    """AMI of a PDI partition's window labels with the oracle ignition state
    of the same windows (None when either is missing)."""
    labels = (partition or {}).get("labels")
    ign = (windows or {}).get("ignition")
    if not labels or not ign:
        return None
    n = min(len(labels), len(ign))
    return adjusted_mutual_info(labels[:n], ign[:n])


def pdi_oracle_windows(system, params: Mapping) -> dict:
    """Oracle labels per PDI window (offset 0, the estimator's windows of
    ``PDIParams.window`` samples): the ignition state (gate above half its
    full value ``g_b``, and no ignition without the amplification, as the
    manipulation check defines it) and the context, majority per window, for
    the bench evaluator's comparison with the counted partitions. Read from
    the oracle by the runner, never by the estimator."""
    from impact_pipeline.bench import manipulation_v2 as MV
    from impact_pipeline.v2 import pdi_v3

    w = int(pdi_v3.PDIParams.from_mapping(dict(params or {})).window)
    oracle = system.oracle or {}
    n = system.n_time // w
    out = {"window_samples": w, "n_windows": int(n)}
    gate = oracle.get("ignition_gate")
    g_b = float(((system.meta or {}).get("knobs") or {}).get("g_b", 0.0) or 0.0)
    if gate is not None and n:
        amplified = MV.ignition_occupancy(system)["amplification_gain"] > 0
        on = (np.asarray(gate, float)[: n * w] > 0.5 * g_b if g_b > 0 and amplified
              else np.zeros(n * w, bool))
        out["ignition"] = [int(v) for v in on.reshape(n, w).mean(axis=1) > 0.5]
    ctx = oracle.get("context_state")
    if ctx is not None and n:
        c = np.asarray(ctx)[: n * w].reshape(n, w)
        out["context"] = [int(np.bincount(row - row.min()).argmax() + row.min())
                          for row in c.astype(np.int64)]
    return out


# --------------------------------------------------------------------------
# scoring one simulation
# --------------------------------------------------------------------------
def _finite(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sanitize(obj):
    """A JSON-compatible copy (numpy to Python, non-finite to None, unknown
    objects to their string)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return str(obj)


@dataclass
class _Outcome:
    output: Optional[ScorerOutput]
    error: Optional[BaseException] = None
    not_applicable: Optional[str] = None
    seconds: Optional[float] = None
    load_average: Optional[tuple] = None
    computed_in: Optional[str] = None
    bases: tuple = ()


def _evidence(principle, outcome: _Outcome, ctx: ScoringContext, scorer: Scorer):
    """The evidence items of one component (ComponentEvidenceV2 per member)."""
    from impact_pipeline import evidence_v2 as E

    proto = ctx.protocol
    dirs = proto.directions_for(principle)
    common = dict(
        principle=principle, channel="default",
        bearer_id=ctx.bearer().get("bearer_id", "system"),
        protocol_id=proto.protocol_id, substrate=ctx.system.meta.get("substrate"),
        grain=ctx.system.meta.get("iim_grain") if principle == "IIM" else None,
        observation_stage=ctx.view.observation_stage, view=ctx.view.name,
        regime=ctx.regime() or None,
    )
    if outcome.output is None:
        reason = (R.estimator_error(outcome.error) if outcome.error is not None
                  else R.NOT_APPLICABLE_OBSERVATION_MODEL)
        est = f"compute_{principle}:undefined@{scorer.version}"
        return [E.ComponentEvidenceV2(estimate=float("nan"), defined=False,
                                      reason=reason, estimator=est, direction=d,
                                      **common)
                for d in (dirs or (None,))]
    items, seen = [], set()
    for m in outcome.output.members:
        if (m.direction is None) != (dirs is None) or (
                dirs is not None and m.direction not in dirs):
            raise ValueError(f"{principle}: the estimator's direction "
                             f"{m.direction!r} is not one of the protocol's {dirs}")
        if m.direction in seen:
            raise ValueError(f"{principle}: more than one estimator item for "
                             f"direction {m.direction!r}")
        seen.add(m.direction)
        items.append(E.ComponentEvidenceV2(
            estimate=float("nan") if m.estimate is None else float(m.estimate),
            null_mean=float("nan") if m.null_mean is None else float(m.null_mean),
            null_sd=float("nan") if m.null_sd is None else float(m.null_sd),
            se=0.0 if m.se is None else float(m.se), se_df=m.se_df,
            defined=bool(m.defined), reason=m.reason,
            estimator=outcome.output.estimator_id, null_family=m.null_family,
            n_null=int(m.n_null or 0), exact=bool(m.exact), se_method=m.se_method,
            direction=m.direction, p_ind=m.p_ind, content_bearer=m.content_bearer,
            **common))
    return items


def _component_record(principle, outcome, ctx, scorer, verdict, error_reason=None):
    """The ``mpc-bench-result/3`` record of one component."""
    scoring, view = ctx.scoring, ctx.view
    base = dict(
        principle=principle, estimator_version=scorer.version,
        declaration_id=scoring.declaration_id,
        observation_stage=view.observation_stage, protocol_id=ctx.protocol_key,
        protocol_hash=ctx.resolved.hash, replicate=ctx.task.replicate,
        identifiability=ctx.identifiability(), seconds=outcome.seconds,
        load_average=outcome.load_average,
    )
    if error_reason is not None:
        exc = outcome.error
        details = {"error": f"{type(exc).__name__}: {exc}" if exc is not None
                   else error_reason, "scorer": scorer.kind}
        if outcome.computed_in is not None:
            details["computed_in_scoring"] = outcome.computed_in
        return REC.ComponentRecord(status=R.UNDEFINED, reason=error_reason,
                                   details=details, **base)
    fields = verdict.construct[principle]
    out = outcome.output
    members = [] if out is None else list(out.members)
    pick = members[0] if members else Member(defined=False)
    deciding = verdict.principle_assessment.get(principle)
    if deciding is not None and deciding.members and len(members) > 1:
        cs = {m.direction: m.c for m in deciding.members if math.isfinite(m.c)}
        if cs:
            want = min(cs, key=cs.get)
            pick = next((m for m in members if m.direction == want), pick)
    details = {"scorer": scorer.kind}
    if out is not None:
        details["estimator_id"] = out.estimator_id
        details["estimator"] = out.details
        details["defined"] = bool(pick.defined)
        details["estimator_reason"] = pick.reason
        if len(members) > 1 or (members and members[0].direction is not None):
            details["members"] = {str(m.direction): m.to_dict() for m in members}
        if pick.p_ind is not None:
            details["p_ind"] = pick.p_ind
    else:
        details["not_applicable"] = outcome.not_applicable
    if outcome.computed_in is not None:
        details["computed_in_scoring"] = outcome.computed_in
    if deciding is not None:
        details["assessment"] = _sanitize(deciding.to_dict())
    rec = dict(
        status=fields["status"], reason=fields["reason"], flags=tuple(fields["flags"]),
        estimate=pick.estimate, null_mean=pick.null_mean, null_sd=pick.null_sd,
        n_null=int(pick.n_null or 0), null_family=pick.null_family, se=pick.se,
        se_df=pick.se_df, se_method=pick.se_method, c=fields["c"],
        se_c=fields["se_c"], df_c=fields["df_c"], details=_sanitize(details),
    )
    if principle == "NAS":
        rec["c_R"], rec["c_B"] = fields.get("c_R"), fields.get("c_B")
    return REC.ComponentRecord(**rec, **base)


def _verdict_block(v) -> dict:
    return _sanitize({
        "verdict": v.verdict.value,
        "reasons": list(v.reasons),
        "necessity_set": list(v.necessity_set),
        "declared_necessity_set": list(v.declared_necessity_set),
        "verdict_declared": (None if v.verdict_declared is None
                             else v.verdict_declared.value),
        "reasons_declared": list(v.reasons_declared),
        "component_status": {p: s.value for p, s in v.component_status.items()},
        "fallback": v.fallback,
        "protocol_hash": v.protocol_hash,
        "status_rule": v.status_rule,
    })


def _observe(view: D.ViewSpec, source, task):
    return source if view.transform is None else view.transform(source, task)


def score_simulation(task: D.TaskSpec, source,
                     protocols: Mapping[str, ResolvedProtocol],
                     settings: RunSettings = DEFAULT_SETTINGS, registry=None,
                     views: Optional[Mapping] = None
                     ) -> Tuple[REC.ScoringRecord, ...]:
    """Every scoring of one simulated system (module docstring, steps 3-4).
    ``views`` holds the observed systems a builder returned with the
    simulation (:class:`~impact_pipeline.bench.designs_v2.MultiViewSystem`);
    a view outside it is observed through its transform."""
    from impact_pipeline import evidence_v2 as E

    cache: dict = {IIM_CUT_MODES_READ: iim_cut_modes_read(task, protocols, settings)}
    outputs: dict = {}
    systems: dict = dict(views or {})
    out = []
    for spec in task.scorings:
        resolved = protocols[spec.protocol_key]
        view = VIEWS[spec.view]
        if spec.view not in systems:
            systems[spec.view] = _observe(view, source, task)
        system = systems[spec.view]
        principles = [p for p in spec.principles if settings.wants(p)]
        ctxs, outcomes, scorers, evidence = {}, {}, {}, {}
        for p in principles:
            scorer = scorer_for(resolved.protocol.estimator_version_for(p), p)
            ctx = ScoringContext(task, spec, source, system, view, resolved, p,
                                 settings, registry, cache)
            key = (p, scorer.version, spec.view, spec.bearer_mode,
                   json.dumps(ctx.options, sort_keys=True),
                   spec.declaration_id if scorer.uses_declaration else None,
                   json.dumps(resolved.protocol.reference_for(p), sort_keys=True)
                   if scorer.uses_reference else None)
            if key in outputs:
                first, where = outputs[key]
                outcome = dataclasses.replace(first, computed_in=where)
            else:
                outcome = _run_scorer(scorer, ctx, task.family)
                outcome = dataclasses.replace(outcome, bases=tuple(ctx.used_bases))
                outputs[key] = (outcome, spec.scoring_id)
            try:
                items = _evidence(p, outcome, ctx, scorer)
            except Exception as exc:  # noqa: BLE001 - an invalid output is an error
                outcome = _invalid_output(outcome, exc)
                items = _evidence(p, outcome, ctx, scorer)
            ctxs[p], outcomes[p], scorers[p], evidence[p] = ctx, outcome, scorer, items
        # The status rule and the component records validate the estimator's
        # fields (for example se_df > 0, a token null family). An output that
        # fails there is an estimator error of that component alone: it is
        # re-judged as one and the verdict is taken again, so it never costs
        # the other components (an error component always validates).
        regime = view_regime(view, system) or None
        for _attempt in range(3):
            try:
                verdict = E.verdict_v3(evidence, resolved.protocol, registry=registry,
                                       regime=regime)
            except Exception:  # noqa: BLE001 - find the components that raise
                invalid = _unassessable(evidence, resolved.protocol, registry, regime)
                if not invalid:
                    raise
            else:
                comps, invalid = {}, {}
                for p in principles:
                    o = outcomes[p]
                    err = R.estimator_error(o.error) if o.error is not None else None
                    try:
                        comps[p] = _component_record(p, o, ctxs[p], scorers[p],
                                                     verdict, err)
                    except Exception as exc:  # noqa: BLE001 - see above
                        if err is not None:
                            raise
                        invalid[p] = exc
                if not invalid:
                    break
            for p, exc in invalid.items():
                outcomes[p] = _invalid_output(outcomes[p], exc)
                evidence[p] = _evidence(p, outcomes[p], ctxs[p], scorers[p])
        else:  # pragma: no cover - error components always validate
            raise RuntimeError(f"{task.task_id}/{spec.scoring_id}: the components "
                               "could not be judged")
        details = {
            "protocol_source": resolved.source,
            "bearer_mode": spec.bearer_mode,
            "held_out": spec.held_out,
            "principles": list(spec.principles),
        }
        if system is not source:
            details["view_ts_sha256"] = PV.array_sha256(system.ts)
        if isinstance((system.meta or {}).get("forward_v2"), Mapping):
            # the view, its stage and observation, the registry substrate and
            # the regime keys the registry builder reads
            from impact_pipeline.bench import forward_v2 as F2

            details.update(F2.scoring_details(system))
        decl_key = ("declared", spec.view, spec.declaration_id)
        if decl_key in cache:
            dec = cache[decl_key]
            details["declaration"] = _sanitize({
                "sha256": dec.sha256(), "shared_inputs": dec.shared_inputs,
                "event_types": list(dec.event_types),
                "state_types": list(dec.state_types),
                "channels": sorted(dec.channels),
                "excluded_event_types": list(dec.excluded_event_types),
                "label_error": dec.details.get("label_error"),
                "onset_jitter": dec.details.get("onset_jitter"),
            })
        # the bases this scoring's estimators used (also when their output was
        # computed in an earlier scoring of the same simulation)
        bases = {f"{k[3]}:tau_c={k[4]:g}": k for p in principles
                 for k in outcomes[p].bases if k in cache}
        if bases:
            # the effective rank, not the column count, is the conditioning
            # budget of a basis (lagged filtered copies are collinear)
            details["basis"] = {name: _basis_summary(cache, key)
                                for name, key in sorted(bases.items())}
        out.append(REC.ScoringRecord(
            scoring_id=spec.scoring_id, declaration_id=spec.declaration_id,
            observation_stage=view.observation_stage, view=view.name,
            estimator_form=spec.estimator_form, protocol_id=spec.protocol_key,
            protocol_hash=resolved.hash, components=comps,
            verdict=_verdict_block(verdict) if spec.verdict else None,
            details=details))
    return tuple(out)


def _basis_summary(cache: dict, key: tuple) -> dict:
    """Columns, effective rank (computed once per basis), hash, lags and time
    constants of a cached input basis."""
    skey = ("basis_summary",) + key[1:]
    if skey not in cache:
        b = cache[key]
        cache[skey] = {"n_columns": b.n_columns, "rank": b.rank(),
                       "sha256": b.sha256(), "lags": list(b.lags),
                       "taus": list(b.taus)}
    return cache[skey]


def _unassessable(evidence: Mapping, protocol, registry, regime) -> dict:
    """The principles whose evidence the status rule refuses (it raises),
    with the exception."""
    from impact_pipeline import evidence_v2 as E

    out = {}
    for p, items in evidence.items():
        try:
            E.assess_principle(p, items, protocol, registry=registry, regime=regime)
        except Exception as exc:  # noqa: BLE001 - reported as the component's
            out[p] = exc
    return out


def _invalid_output(outcome: _Outcome, exc: BaseException) -> _Outcome:
    """An estimator output the evidence or record layer refuses: an error of
    that component (``ESTIMATOR_ERROR:<type>``), with its timing kept."""
    return _Outcome(None, error=exc, seconds=outcome.seconds,
                    load_average=outcome.load_average,
                    computed_in=outcome.computed_in, bases=outcome.bases)


def _run_scorer(scorer: Scorer, ctx: ScoringContext, family: str) -> _Outcome:
    na = None
    if not scorer.declares_observation_model:
        na = NOT_APPLICABLE.get(family, {}).get(scorer.principle)
    if na is not None:
        return _Outcome(None, not_applicable=na, seconds=0.0)
    with PV.timed() as t:
        try:
            output = scorer.score(ctx)
            error = None
        except Exception as exc:  # noqa: BLE001 - isolation of every estimator
            output, error = None, exc
    return _Outcome(output, error, seconds=t["seconds"], load_average=t["load_average"])


# --------------------------------------------------------------------------
# one task
# --------------------------------------------------------------------------
def simulation_block(system, seconds=None) -> dict:
    """The simulation block of a record: duplicate-detector and twin hashes,
    shape, ``dt`` and simulation time."""
    from impact_pipeline.bench import manipulation_v2 as MV

    raw = getattr(system, "raw_ts", None)
    if raw is None and isinstance(system.oracle, Mapping):
        raw = system.oracle.get("raw_ts")
    out = {
        "ts_sha256": PV.array_sha256(system.ts),
        "raw_ts_sha256": None if raw is None else PV.array_sha256(raw),
        "structural_hash": None,
        "schedule_hash": None,
        "n_nodes": int(system.n_nodes),
        "n_time": int(system.n_time),
        "dt": float(system.dt),
        "seconds": seconds,
    }
    for key, fn in (("structural_hash", MV.structural_hash),
                    ("schedule_hash", MV.schedule_hash)):
        try:
            out[key] = fn(system)
        except Exception:  # noqa: BLE001 - not every system has the channels
            out[key] = None
    return out


# meta keys that name or version a realised system without configuring it
SYSTEM_LABEL_META_KEYS = ("witness_id", "system_id", "generator_version", "replicate")


def system_config_digest(system) -> Optional[str]:
    """SHA-256 of the realised configuration of a simulated system: its meta
    without the naming keys (:data:`SYSTEM_LABEL_META_KEYS`). Two designs
    that spell one system differently (the catalogue's PC_nominal, the
    nominal level of a sweep, the all-on factorial cell) share it, so the
    duplicate detector can tell one configuration under several designs
    from two configurations that produce the same recording."""
    meta = {k: v for k, v in dict(getattr(system, "meta", None) or {}).items()
            if k not in SYSTEM_LABEL_META_KEYS}
    text = json.dumps(_sanitize(meta), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_task(task: D.TaskSpec, protocols: Mapping[str, ResolvedProtocol],
             settings: RunSettings = DEFAULT_SETTINGS, *, registry=None,
             provenance: Optional[Mapping] = None,
             confirmatory: bool = False) -> REC.TaskRecord:
    """
    Simulate and score one task. Simulation and scoring failures are recorded
    (task ``error``; a raising estimator is a component error), never raised.
    A confirmatory seed is refused unless the run passed the confirmatory
    guard (``confirmatory=True``).
    """
    if task.split == S.CONFIRMATORY and not confirmatory:
        raise RunPolicyError(f"{task.task_id}: seed {task.seed} is confirmatory; "
                             "confirmatory tasks run only through the confirmatory "
                             "guard")
    if task.split == S.DEVELOPMENT and confirmatory:
        raise RunPolicyError(f"{task.task_id}: a confirmatory run uses seeds >= "
                             f"{S.CONFIRMATORY_SEED_MIN}")
    if task.split == S.DEVELOPMENT and task.has_held_out and not task.smoke:
        raise RunPolicyError(f"{task.task_id}: held-out conditions run on development "
                             "seeds only as smoke tests (seeds 980-984)")
    t_all = time.perf_counter()
    cpu_all = time.process_time()
    timing: dict = {}
    simulation, scorings, error, gen_version = None, (), None, None
    config_digest = None
    extra_config: dict = {}
    try:
        load_design_module(task.design_module)
        builder = SYSTEM_BUILDERS.get(task.builder)
        if builder is None:
            raise RunPolicyError(f"no system builder {task.builder!r}")
        for key, fn in RECORD_CONFIG.get(task.design_module, {}).items():
            extra_config[key] = _sanitize(fn(task))
        with PV.timed() as t:
            built = builder(task)
        views = None
        if isinstance(built, D.MultiViewSystem):
            source, views = built.source, dict(built.views)
        else:
            source = built
        timing["simulate_s"] = t["seconds"]
        gen_version = (source.meta or {}).get("generator_version")
        simulation = simulation_block(source, t["seconds"])
        config_digest = system_config_digest(source)
        with PV.timed() as t:
            scorings = score_simulation(task, source, protocols, settings, registry,
                                        views=views)
        timing["score_s"] = t["seconds"]
    except Exception as exc:  # noqa: BLE001 - recorded, the run continues
        error = f"{type(exc).__name__}: {exc}"
        scorings = ()
    timing["total_s"] = round(time.perf_counter() - t_all, 4)
    # CPU time of the worker process (one BLAS thread): the cost of the task
    timing["cpu_s"] = round(time.process_time() - cpu_all, 4)
    timing["load_average"] = PV.load_average()
    timing["components_s"] = {
        f"{s.scoring_id}:{p}": c.seconds for s in scorings
        for p, c in s.components.items() if c.seconds is not None
        and not (c.details or {}).get("computed_in_scoring")}
    config = {
        "builder": task.builder,
        "params": task.params,
        "tags": task.tags,
        "held_out": task.held_out,
        "design_module": task.design_module,
        "system_generator_version": gen_version,
        "system_config_sha256": config_digest,
        "null_seed": null_seed(task.seed, task.replicate),
        "runner_version": RUNNER_VERSION,
        "settings": settings.to_dict(),
    }
    config.update(extra_config)
    return REC.TaskRecord(
        task_id=task.task_id, design=task.design, family=task.family,
        system=task.system, seed=task.seed, generator_version=GENERATOR_VERSION_V2,
        status=REC.derive_task_status(scorings, error), replicate=task.replicate,
        config=config, simulation=simulation, scorings=scorings, error=error,
        timing=timing, provenance=dict(provenance or {}))


def smoke_summary(record: REC.TaskRecord) -> dict:
    """What a smoke task leaves behind: status lines only, no estimator
    output (held-out conditions are discarded unread)."""
    comps = [c for s in record.scorings for c in s.components.values()]
    return {
        "task_id": record.task_id,
        "design": record.design,
        "seed": record.seed,
        "status": record.status,
        "error": record.error,
        "n_scorings": len(record.scorings),
        "n_components": len(comps),
        "n_component_errors": sum(c.is_estimator_error for c in comps),
        "total_s": (record.timing or {}).get("total_s"),
        "cpu_s": (record.timing or {}).get("cpu_s"),
        "discarded": True,
    }


# --------------------------------------------------------------------------
# plan checks
# --------------------------------------------------------------------------
def protocol_keys(tasks: Iterable[D.TaskSpec]) -> List[str]:
    """The protocol keys of a plan, sorted. The plan's design modules are
    registered first (:func:`load_design_module`), so the keys whose drafts
    a design module declares (``PROTOCOL_DRAFTS``) resolve in a fresh
    process: ``resolve_protocols(protocol_keys(tasks))`` needs nothing
    else."""
    tasks = list(tasks)
    for t in tasks:
        load_design_module(t.design_module)
    return sorted({s.protocol_key for t in tasks for s in t.scorings})


def check_plan(tasks: Sequence[D.TaskSpec], protocols: Mapping[str, ResolvedProtocol],
               *, confirmatory: bool = False,
               settings: RunSettings = DEFAULT_SETTINGS) -> dict:
    """
    Refuse a run that breaks the policy before anything is simulated: one
    split, matching ``confirmatory``; held-out conditions on development
    seeds only as smoke tests; unique task ids; registered builders and
    views; for every scoring a protocol whose declaration matches, an
    estimator version per principle with a scorer of the right principle
    and directions, and (unless allowed) an available estimator module;
    generated protocols and the default runner settings only in a
    confirmatory run. Returns a summary.
    """
    if not tasks:
        raise RunPolicyError("no tasks")
    if confirmatory and settings != DEFAULT_SETTINGS:
        # every principle of the plan, merged estimators only, the frozen v1
        # null size and jackknife groups and the frozen calibration values
        raise RunPolicyError("a confirmatory run uses the default runner settings, "
                             f"not {settings.to_dict()}")
    for t in tasks:
        load_design_module(t.design_module)
    split = S.check_seeds([t.seed for t in tasks],
                          S.CONFIRMATORY if confirmatory else S.DEVELOPMENT)
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise RunPolicyError("duplicate task ids in the plan")
    unavailable = set()
    for t in tasks:
        if t.builder not in SYSTEM_BUILDERS:
            raise RunPolicyError(f"{t.task_id}: no system builder {t.builder!r}")
        if split == S.DEVELOPMENT and t.has_held_out and not t.smoke:
            raise RunPolicyError(f"{t.task_id}: held-out conditions run on development "
                                 "seeds only as smoke tests (seeds 980-984)")
        for s in t.scorings:
            rp = protocols.get(s.protocol_key)
            if rp is None:
                raise RunPolicyError(f"{t.task_id}/{s.scoring_id}: no protocol "
                                     f"{s.protocol_key!r}")
            if confirmatory and rp.source != "generated":
                raise RunPolicyError(f"protocol {s.protocol_key}: a confirmatory run "
                                     f"uses generated protocols only ({rp.source})")
            if s.view not in VIEWS:
                raise RunPolicyError(f"{t.task_id}/{s.scoring_id}: unknown view "
                                     f"{s.view!r}")
            decl = (rp.protocol.shared_inputs_declaration or {}).get("id")
            if decl is not None and decl != s.declaration_id:
                raise RunPolicyError(f"{t.task_id}/{s.scoring_id}: declaration "
                                     f"{s.declaration_id} judged by protocol "
                                     f"{s.protocol_key} of declaration {decl}")
            for p in s.principles:
                if not settings.wants(p):
                    continue
                sc = scorer_for(rp.protocol.estimator_version_for(p), p)
                dirs = rp.protocol.directions_for(p)
                if (sc.directions is None) != (dirs is None) or (
                        dirs is not None and set(sc.directions) != set(dirs)):
                    raise RunPolicyError(
                        f"protocol {s.protocol_key}: {p} directions {dirs} but "
                        f"{sc.version} produces {sc.directions}")
                try:
                    sc.check_options(rp.protocol.estimator_options(p))
                except (ProtocolOptionError, ValueError) as exc:
                    raise RunPolicyError(f"protocol {s.protocol_key}: {p} options: "
                                         f"{exc}") from exc
                if not sc.available():
                    unavailable.add(sc.version)
    if unavailable and not settings.allow_unavailable_estimators:
        raise RunPolicyError(
            "estimator modules not merged yet: " + ", ".join(sorted(unavailable))
            + " (restrict the principles, or allow them to be recorded as "
              "estimator errors)")
    return {"split": split, "n_tasks": len(tasks),
            "n_scorings": sum(len(t.scorings) for t in tasks),
            "unavailable_estimators": sorted(unavailable),
            "protocols": {k: {"hash": v.hash, "source": v.source, "path": v.path}
                          for k, v in sorted(protocols.items())}}


def shard(tasks: Sequence[D.TaskSpec], shard_index: int = 0,
          n_shards: int = 1) -> List[D.TaskSpec]:
    """A deterministic shard: every ``n_shards``-th task id in sorted order,
    the tasks kept in plan order (one shard is the plan itself). A plan with
    curtailed sampling (:func:`curtailment_controllers`) is not split: a
    gated run's decision reads the records of the runs before it in the
    rule's order, which another shard would hold."""
    n_shards, shard_index = int(n_shards), int(shard_index)
    if n_shards < 1 or not 0 <= shard_index < n_shards:
        raise ValueError("need 0 <= shard_index < n_shards")
    if n_shards > 1 and curtailment_controllers(tasks):
        raise RunPolicyError("a plan with curtailed sampling runs as one shard: "
                             "the stopping rule reads the records of the whole "
                             "plan in its order")
    keep = set(sorted(t.task_id for t in tasks)[shard_index::n_shards])
    return [t for t in tasks if t.task_id in keep]


# --------------------------------------------------------------------------
# results files and resume
# --------------------------------------------------------------------------
def read_results(path) -> Tuple[Dict[str, REC.TaskRecord], dict]:
    """The latest valid record per task id of a results file, and what was
    dropped: lines that do not parse (a run interrupted mid-write) and
    superseded records."""
    p = Path(path)
    latest: Dict[str, REC.TaskRecord] = {}
    info = {"n_lines": 0, "n_bad_lines": 0, "n_superseded": 0}
    if not p.exists():
        return latest, info
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            info["n_lines"] += 1
            try:
                rec = REC.loads(line)
            except (ValueError, TypeError):
                info["n_bad_lines"] += 1
                continue
            if rec.task_id in latest:
                info["n_superseded"] += 1
            latest[rec.task_id] = rec
    return latest, info


def prepare_resume(path, plan_ids: Sequence[str]) -> Tuple[set, dict]:
    """
    Make a results file resumable: keep the latest record of every planned
    task that did not fail, drop unparsable lines, superseded records and
    failed tasks (they are run again), rewrite the file when something was
    dropped. Returns the done task ids and a report.
    """
    p = Path(path)
    latest, info = read_results(p)
    planned = set(plan_ids)
    unexpected = sorted(set(latest) - planned)
    if unexpected:
        raise RunPolicyError(f"{p} holds tasks outside this plan: {unexpected[:5]}")
    keep = {tid: rec for tid, rec in latest.items() if rec.status != REC.TASK_ERROR}
    rerun = sorted(tid for tid, rec in latest.items() if rec.status == REC.TASK_ERROR)
    info["rerun_failed"] = rerun
    rewrite = info["n_bad_lines"] or info["n_superseded"] or rerun
    if p.exists() and rewrite:
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for tid in sorted(keep):
                fh.write(REC.dumps(keep[tid]) + "\n")
        os.replace(tmp, p)
    info["rewritten"] = bool(p.exists() and rewrite)
    return set(keep), info


def _end_with_newline(path) -> None:
    """Terminate an interrupted last line, so that appended records start on
    a line of their own (the broken line is then skipped by the readers)."""
    p = Path(path)
    if p.exists() and p.stat().st_size:
        with open(p, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            last = fh.read(1)
        if last != b"\n":
            with open(p, "a", encoding="utf-8") as fh:
                fh.write("\n")


def smoke_done(path) -> set:
    """Smoke tasks whose status line says they ran (not failed)."""
    p = Path(path)
    out = set()
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("status") != REC.TASK_ERROR:
                out.add(row.get("task_id"))
    return out


def write_components_csv(out_dir) -> Optional[Path]:
    """``components.csv``: one row per (task, scoring, component) of the
    latest records."""
    import pandas as pd

    out = Path(out_dir)
    latest, _ = read_results(out / RESULTS_JSONL)
    rows = [row for tid in sorted(latest) for row in latest[tid].component_rows()]
    path = out / COMPONENTS_CSV
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# running a plan
# --------------------------------------------------------------------------
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")


def _worker_init():
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")


@contextlib.contextmanager
def _quiet():
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(previous)


@contextlib.contextmanager
def _single_threaded_children():
    """Spawned workers start with one BLAS / OpenMP thread each (whatever the
    parent's settings), so that 12 workers use 12 cores and a worker's
    floating-point reductions do not depend on a thread count."""
    saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    try:
        yield
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


_WORKER_PROTOCOLS: Dict[str, ResolvedProtocol] = {}


def _execute(task_payload, protocol_payloads, settings_payload, provenance,
             confirmatory, registry=None):
    """Run one task in a worker; a smoke task returns its status line only."""
    task = D.TaskSpec.from_dict(task_payload)
    protos = {}
    for key, payload in protocol_payloads.items():
        cached = _WORKER_PROTOCOLS.get(key)
        if cached is None or cached.hash != payload.get("hash"):
            cached = ResolvedProtocol.from_payload(payload)
            _WORKER_PROTOCOLS[key] = cached
        protos[key] = cached
    rec = run_task(task, protos, RunSettings.from_dict(settings_payload),
                   registry=registry, provenance=provenance, confirmatory=confirmatory)
    if task.smoke:
        return "smoke", smoke_summary(rec)
    return "record", rec.to_dict()


def curtailment_controllers(tasks: Sequence[D.TaskSpec]) -> list:
    """
    The curtailment controllers of a plan: one per design module that
    declares ``RUNNER_CURTAILMENT(tasks)`` (built on that module's tasks).
    A controller's contract:

    * ``gated_ids()``: the ids of the tasks that may be skipped, in the order
      the rule evaluates them;
    * ``feed(task_id, record)``: the record of a task of the plan (tasks the
      controller does not know are ignored);
    * ``decision(task_id)`` for a gated task: None while a task before it in
      the rule's order has no record, else True (skip) or False (run). It
      depends only on the records of the tasks before it, so the records a
      run keeps do not depend on the number of workers or the completion
      order;
    * ``summary()``: a JSON block for the run manifest.
    """
    by_mod: Dict[str, List[D.TaskSpec]] = {}
    for t in tasks:
        if t.design_module is not None:
            by_mod.setdefault(t.design_module, []).append(t)
    out = []
    for name, ts in by_mod.items():
        fn = getattr(D.load_module(name), "RUNNER_CURTAILMENT", None)
        if callable(fn):
            c = fn(ts)
            if c is not None:
                out.append(c)
    return out


def _make_pool(n_workers: int) -> concurrent.futures.Executor:
    """The worker pool of a parallel run: ``n_workers`` spawned processes,
    one BLAS thread each."""
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_worker_init,
        mp_context=multiprocessing.get_context("spawn"))


def _run_parallel(regular, gated, gate_of, by_id, n_workers, emit, curtailed,
                  args) -> None:
    """Run a plan on ``n_workers`` spawned processes: every regular task at
    once; the gated tasks in their rule's order, at most ``n_workers`` of
    them ahead of their decision. A gated task's record is kept only once
    its decision says so (records computed ahead of a stop are discarded),
    so the kept records equal those of a serial run."""
    payloads, settings_payload, small_prov, confirmatory, registry = args
    queue = list(gated)
    buffered: Dict[str, tuple] = {}
    pending: Dict[concurrent.futures.Future, str] = {}
    with _single_threaded_children(), _make_pool(n_workers) as ex:

        def submit(t):
            fut = ex.submit(_execute, t.to_dict(), payloads, settings_payload,
                            small_prov, confirmatory, registry)
            pending[fut] = t.task_id

        def top_up():
            # gated runs computed ahead of their decision, running or waiting
            # in the buffer: at most n_workers, so a stop wastes little
            ahead = sum(1 for tid in pending.values() if tid in gate_of) + len(buffered)
            while queue and ahead < n_workers:
                tid = queue.pop(0)
                if gate_of[tid].decision(tid):
                    curtailed.append(tid)
                    continue
                submit(by_id[tid])
                ahead += 1

        def flush():
            moved = True
            while moved:
                moved = False
                for tid in [g for g in gated if g in buffered]:
                    decision = gate_of[tid].decision(tid)
                    if decision is None:
                        continue
                    kind, payload = buffered.pop(tid)
                    if decision:
                        curtailed.append(tid)
                    else:
                        emit(kind, payload)
                    moved = True

        for t in regular:
            submit(t)
        top_up()
        while pending:
            finished, _ = concurrent.futures.wait(
                list(pending), return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in finished:
                tid = pending.pop(fut)
                if tid in gate_of:
                    buffered[tid] = fut.result()
                else:
                    emit(*fut.result())
            flush()
            top_up()
        flush()
        if buffered or queue:  # pragma: no cover - every predecessor has run
            raise RuntimeError("curtailment: gated tasks left without a decision: "
                               f"{sorted(buffered) + queue}")


def run_tasks(tasks: Sequence[D.TaskSpec], out_dir, *, n_workers: int = 1,
              settings: RunSettings = DEFAULT_SETTINGS, resume: bool = True,
              confirmatory: bool = False, freeze_tag: Optional[str] = None,
              protocol_dir=None, allow_drafts: Optional[bool] = None,
              protocol_overrides: Optional[Mapping] = None, registry=None,
              repo_root=None, label: Optional[str] = None) -> dict:
    """
    Run a plan into ``out_dir``: the plan check, the confirmatory guard
    (``confirmatory=True``: clean tree, freeze tag, seeds >= 20000, generated
    protocols), ``run_plan.json``, resume from ``results.jsonl``, the tasks
    (``n_workers`` spawned processes, one BLAS thread each), ``smoke.jsonl``
    for smoke tasks, ``components.csv`` and ``run_manifest.json``. Returns the
    manifest.
    """
    tasks = list(tasks)
    if allow_drafts is None:
        allow_drafts = not confirmatory
    if confirmatory and (allow_drafts or protocol_overrides):
        raise RunPolicyError("a confirmatory run uses generated protocols only")
    seeds = [t.seed for t in tasks]
    if confirmatory:
        try:
            prov = PV.run_provenance(repo_root or PV.REPO_ROOT, seeds=seeds,
                                     confirmatory=True,
                                     freeze_tag=freeze_tag or PV.FREEZE_TAG_V2)
        except PV.ConfirmatoryGuardError as exc:
            raise RunPolicyError(f"confirmatory guard: {exc}") from exc
    else:
        prov = PV.run_provenance(repo_root or PV.REPO_ROOT, seeds=seeds,
                                 confirmatory=False)
    # the design modules register their protocol drafts, views and builders
    for t in tasks:
        load_design_module(t.design_module)
    protocols = resolve_protocols(protocol_keys(tasks), directory=protocol_dir,
                                  allow_drafts=allow_drafts,
                                  overrides=protocol_overrides)
    summary = check_plan(tasks, protocols, confirmatory=confirmatory,
                         settings=settings)
    # byte hashes of the protocol files read (the records carry the
    # protocols' content hashes)
    prov["protocol_files"] = [PV.protocol_file_record(rp.path)
                              for _, rp in sorted(protocols.items()) if rp.path]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_ids = [t.task_id for t in tasks]
    plan_path = out / PLAN_JSON
    plan_payload = {"schema": D.DESIGN_SCHEMA, "runner_version": RUNNER_VERSION,
                    "split": summary["split"], "task_ids": plan_ids,
                    "tasks": [t.to_dict() for t in tasks],
                    "settings": settings.to_dict(),
                    "protocols": {k: v["hash"]
                                  for k, v in summary["protocols"].items()}}
    if plan_path.exists() and resume:
        # a resumed run continues the same plan: the same tasks, settings and
        # protocols, so that its records stay comparable with the earlier ones
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        differ = [k for k in ("task_ids", "tasks", "settings", "protocols")
                  if old.get(k) != plan_payload[k]]
        if differ:
            raise RunPolicyError(f"{plan_path} holds another plan ({', '.join(differ)} "
                                 "differ); use a new output directory or "
                                 "resume=False")
    jsonl = out / RESULTS_JSONL
    if not resume and jsonl.exists():
        raise RunPolicyError(f"{jsonl} exists; resume or use a new output directory")
    plan_path.write_text(json.dumps(plan_payload, indent=1, sort_keys=True),
                         encoding="utf-8")
    smoke_ids = {t.task_id for t in tasks if t.smoke}
    if resume:
        done, resume_info = prepare_resume(
            jsonl, [i for i in plan_ids if i not in smoke_ids])
    else:
        done, resume_info = set(), {}
    if resume:
        done |= smoke_done(out / SMOKE_JSONL) & smoke_ids
    todo = [t for t in tasks if t.task_id not in done]
    # early stopping by a design's preregistered rule (curtailed sampling):
    # the gated tasks run in the controller's order, each only while its
    # decision (read from the records of the tasks before it) keeps it
    controllers = curtailment_controllers(tasks)
    if controllers and done:
        kept, _ = read_results(jsonl)
        for tid in [i for i in plan_ids if i in kept]:
            for c in controllers:
                c.feed(tid, kept[tid])
    gate_of = {tid: c for c in controllers for tid in c.gated_ids()}
    by_id = {t.task_id: t for t in todo}
    regular = [t for t in todo if t.task_id not in gate_of]
    gated = [tid for c in controllers for tid in c.gated_ids() if tid in by_id]
    curtailed: List[str] = [tid for tid in gate_of
                            if tid not in by_id and gate_of[tid].decision(tid)]
    small_prov = {
        "runner_version": RUNNER_VERSION,
        "git_sha": prov["code"].get("git_sha"),
        "git_dirty": prov["code"].get("git_dirty"),
        "trees": prov["code"].get("trees"),
        "confirmatory": bool(confirmatory),
        "freeze_tag": prov["code"].get("freeze_tag"),
        "freeze_tag_commit": prov["code"].get("freeze_tag_commit"),
        "label": label,
    }
    payloads = {k: v.to_payload() for k, v in protocols.items()}
    counts = {"n_run": 0, "n_errors": 0, "n_component_errors": 0, "n_smoke": 0,
              "component_errors": {}, "components_by_design": {}}
    t_start = time.time()
    for path in (jsonl, out / SMOKE_JSONL):
        _end_with_newline(path)
    with open(jsonl, "a", encoding="utf-8") as fh, \
            open(out / SMOKE_JSONL, "a", encoding="utf-8") as fs:

        def _emit(kind, payload):
            if kind == "smoke":
                fs.write(json.dumps(payload, sort_keys=True) + "\n")
                fs.flush()
                counts["n_smoke"] += 1
                return
            rec = REC.TaskRecord.from_dict(payload)
            for c in controllers:
                c.feed(rec.task_id, rec)
            fh.write(REC.dumps(rec) + "\n")
            fh.flush()
            counts["n_run"] += 1
            counts["n_errors"] += rec.status == REC.TASK_ERROR
            per = counts["components_by_design"].setdefault(
                rec.design, {"n_components": 0, "n_component_errors": 0})
            for s in rec.scorings:
                for p, c in s.components.items():
                    per["n_components"] += 1
                    if c.is_estimator_error:
                        per["n_component_errors"] += 1
                        counts["n_component_errors"] += 1
                        counts["component_errors"][p] = (
                            counts["component_errors"].get(p, 0) + 1)
            log.info("MPC-Bench v2 %s: %s (%.2f s)", rec.task_id, rec.status,
                     (rec.timing or {}).get("total_s", float("nan")))

        settings_payload = settings.to_dict()

        def _run(t):
            return _execute(t.to_dict(), payloads, settings_payload, small_prov,
                            confirmatory, registry)

        if int(n_workers) <= 1:
            with _quiet():
                for t in regular:
                    _emit(*_run(t))
                for tid in gated:
                    decision = gate_of[tid].decision(tid)
                    if decision is None:  # pragma: no cover - its predecessors ran
                        raise RuntimeError(f"{tid}: the curtailment rule has no "
                                           "decision after its earlier runs")
                    if decision:
                        curtailed.append(tid)
                        continue
                    _emit(*_run(by_id[tid]))
        else:
            _run_parallel(regular, gated, gate_of, by_id, int(n_workers), _emit,
                          curtailed, (payloads, settings_payload, small_prov,
                                      confirmatory, registry))
    latest, _ = read_results(jsonl)
    curtailed_ids = sorted(set(curtailed))
    diff = REC.plan_differences(latest.values(),
                                [i for i in plan_ids if i not in smoke_ids
                                 and i not in set(curtailed_ids)])
    manifest = {
        "schema": "mpc-bench-run-manifest/3",
        "runner_version": RUNNER_VERSION,
        "result_schema": REC.RESULT_SCHEMA,
        "label": label,
        "provenance": _sanitize(prov),
        "settings": settings.to_dict(),
        "calibration_pending": calibration_pending(),
        "plan": {"path": PLAN_JSON, "n_tasks": len(tasks), "split": summary["split"],
                 "n_scorings": summary["n_scorings"], "n_smoke": len(smoke_ids),
                 "designs": sorted({t.design for t in tasks})},
        "protocols": summary["protocols"],
        "unavailable_estimators": summary["unavailable_estimators"],
        "resume": _sanitize(resume_info),
        "n_skipped_done": len(tasks) - len(todo),
        **counts,
        "plan_differences": diff,
        "curtailed_task_ids": curtailed_ids,
        "curtailment": [_sanitize(c.summary()) for c in controllers],
        "duplicate_simulations": REC.duplicate_simulations(latest.values()),
        "wall_s": round(time.time() - t_start, 3),
        "confirmatory": bool(confirmatory),
    }
    (out / MANIFEST).write_text(json.dumps(_sanitize(manifest), indent=2,
                                           sort_keys=True), encoding="utf-8")
    if latest:
        write_components_csv(out)
    return manifest


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def _parse_ints(text) -> Optional[List[int]]:
    if text is None:
        return None
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _parse_list(text) -> Optional[List[str]]:
    if text is None:
        return None
    return [t.strip() for t in str(text).split(",") if t.strip()]


def build_tasks(names: Sequence[str], split: str, **options) -> List[D.TaskSpec]:
    """The plan of several designs (:func:`designs_v2.build_plan`: each
    builder gets the options its signature accepts, ``seeds``, ``systems``,
    ...; every task carries its design module). A ``purpose`` is refused
    for a design whose builder declares none."""
    if options.get("purpose") is not None:
        for name in names:
            build = D.get_design(name).build
            if "purpose" not in inspect.signature(build).parameters:
                raise RunPolicyError(f"design {name} declares no purposes")
    return D.build_plan(names, split, **options)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_bench_v2.py",
                                 description="MPC-Bench v2 runner and run plans")
    sub = ap.add_subparsers(dest="command", required=True)
    pl = sub.add_parser("plan", help="print the run-plan table of a split")
    pl.add_argument("--split", default=S.CONFIRMATORY, choices=S.SPLITS)
    pl.add_argument("--designs", default=None, help="comma-separated designs")
    pl.add_argument("--json", default=None, help="write the table as JSON here")
    dz = sub.add_parser("designs", help="list the design modules and designs")
    dz.add_argument("--module", default=None,
                    help="print the comma-separated designs of one module (empty "
                         "when it is not merged yet; exit status 3 when it is "
                         "merged but defines no designs for this runner)")
    mp = sub.add_parser("manipulation",
                        help="prerequisite M: switch and realisation checks (oracle "
                             "only, no estimator)")
    mp.add_argument("--seeds", required=True)
    mp.add_argument("--families", default="A,C")
    mp.add_argument("--out", required=True)
    mp.add_argument("--confirmatory", action="store_true")
    mp.add_argument("--freeze-tag", default=None)
    for name in ("run", "list"):
        p = sub.add_parser(name, help="run a plan" if name == "run"
                           else "print the task ids of a plan")
        p.add_argument("designs", help="comma-separated design names")
        p.add_argument("--split", default=S.DEVELOPMENT, choices=S.SPLITS)
        p.add_argument("--seeds", default=None, help="e.g. 320-323 (within the split)")
        p.add_argument("--systems", default=None, help="comma-separated systems")
        p.add_argument("--purpose", default=None,
                       help="the purpose of designs that declare purposes (the "
                            "forward arms: dry_run, dev_regime, smoke, reference, ...)")
        p.add_argument("--n-shards", type=int, default=1)
        p.add_argument("--shard-index", type=int, default=0)
        if name == "run":
            p.add_argument("--out", required=True)
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--principles", default=None)
            p.add_argument("--protocol-dir", default=None)
            p.add_argument("--no-drafts", action="store_true")
            p.add_argument("--allow-unavailable-estimators", action="store_true")
            p.add_argument("--no-resume", action="store_true")
            p.add_argument("--confirmatory", action="store_true")
            p.add_argument("--freeze-tag", default=None)
            p.add_argument("--label", default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        return _main(args)
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"run_bench_v2: error: {exc}", file=sys.stderr)
        return 2


# generator family of the manipulation checks -> family label of the records
MANIPULATION_FAMILY = {"C": "C1"}


def run_manipulation(seeds: Sequence[int], out_dir, *, families=("A", "C"),
                     confirmatory: bool = False, freeze_tag: Optional[str] = None,
                     repo_root=None) -> dict:
    """
    The prerequisite M (:func:`impact_pipeline.bench.manipulation_v2.
    prerequisite_m`): the frozen switch checks and the realisation checks of
    the new systems, written to ``manipulation_switches.csv``,
    ``manipulation_realisation.csv`` and ``manipulation_summary.csv`` with a
    manifest. Oracle channels only, so it may run on development seeds
    before the freeze; confirmatory seeds go through the confirmatory guard.
    ``complete`` (every check computed) is reported apart from the pass/fail
    of the checks.
    """
    from impact_pipeline.bench import manipulation_v2 as MV

    seeds = [int(s) for s in seeds]
    try:
        prov = PV.run_provenance(repo_root or PV.REPO_ROOT, seeds=seeds,
                                 confirmatory=confirmatory,
                                 freeze_tag=freeze_tag or PV.FREEZE_TAG_V2)
    except PV.ConfirmatoryGuardError as exc:
        raise RunPolicyError(f"confirmatory guard: {exc}") from exc
    if not confirmatory:
        S.assert_development(seeds)
    res = MV.prerequisite_m(seeds, families=tuple(families))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("switches", "realisation", "summary"):
        table = res[name]
        if "family" in getattr(table, "columns", ()):
            # the records name the v1 family C "C1"; usability is looked up
            # by the record's family, so the checks carry the same label
            table = table.assign(family=table["family"].replace(MANIPULATION_FAMILY))
        table.to_csv(out / f"manipulation_{name}.csv", index=False)
    summary = res["summary"]
    passed = (bool(summary["usable"].all()) if "usable" in summary.columns else None)
    manifest = {
        "schema": "mpc-bench-manipulation-manifest/2",
        "runner_version": RUNNER_VERSION,
        "check_versions": res["check_versions"],
        "seeds": seeds, "split": res["split"], "families": list(families),
        "complete": bool(res["complete"]), "all_usable": passed,
        "provenance": _sanitize(prov), "confirmatory": bool(confirmatory),
    }
    (out / "manipulation_manifest.json").write_text(
        json.dumps(_sanitize(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _main(args) -> int:
    if args.command == "manipulation":
        man = run_manipulation(_parse_ints(args.seeds), args.out,
                               families=tuple(_parse_list(args.families)),
                               confirmatory=args.confirmatory,
                               freeze_tag=args.freeze_tag)
        print(f"prerequisite M: complete={man['complete']} "
              f"all_usable={man['all_usable']}; results in {args.out}")
        # 0 iff every check was computed; pass/fail is reported, not the code
        return 0 if man["complete"] else 1
    if args.command == "designs" and args.module:
        try:
            print(",".join(d.name for d in D.load_module(args.module).DESIGNS))
        except D.DesignNotAvailableError:
            print("")
        except D.DesignNotRunnableError as exc:
            # merged, but not run here: a plan step must not skip it silently
            print(f"run_bench_v2: {exc}", file=sys.stderr)
            return 3
        return 0
    if args.command == "designs":
        for m, desc in D.DESIGN_MODULES.items():
            try:
                mod = D.load_module(m)
                names = ", ".join(d.name for d in mod.DESIGNS)
                print(f"{m}: {names}")
            except D.DesignNotAvailableError:
                print(f"{m}: not merged yet ({desc})")
            except D.DesignNotRunnableError as exc:
                print(f"{m}: {exc}")
        return 0
    if args.command == "plan":
        rows = D.plan_table(args.split, _parse_list(args.designs))
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        for r in rows:
            exp = ("" if r["expected"] is None
                   else f" (design document: {r['expected']})")
            print(f"{r['design']:16s} {r['split']:12s} tasks {r['n_tasks']:5d}{exp} "
                  f"scorings {r['n_scorings']:6d} held-out scorings "
                  f"{r['n_held_out_scorings']:4d} "
                  f"seeds {r['seed_min']}-{r['seed_max']}")
        return 0 if all(r["matches"] is not False for r in rows) else 1
    tasks = build_tasks(_parse_list(args.designs), args.split,
                        seeds=_parse_ints(args.seeds),
                        systems=_parse_list(args.systems), purpose=args.purpose)
    tasks = shard(tasks, args.shard_index, args.n_shards)
    if args.command == "list":
        for t in tasks:
            print(t.task_id)
        return 0
    settings = RunSettings(
        principles=(None if args.principles is None
                    else tuple(_parse_list(args.principles))),
        allow_unavailable_estimators=args.allow_unavailable_estimators)
    man = run_tasks(tasks, args.out, n_workers=args.workers, settings=settings,
                    resume=not args.no_resume, confirmatory=args.confirmatory,
                    freeze_tag=args.freeze_tag, protocol_dir=args.protocol_dir,
                    allow_drafts=False if args.no_drafts or args.confirmatory else None,
                    label=args.label)
    print(f"MPC-Bench v2: {man['n_run']} task(s) run, {man['n_errors']} task error(s), "
          f"{man['n_component_errors']} component error(s), {man['n_smoke']} smoke "
          f"task(s) discarded; results in {args.out}")
    return 1 if man["n_errors"] or not man["plan_differences"]["ok"] else 0


__all__ = [
    "CALIBRATION_PENDING",
    "DEFAULT_SETTINGS",
    "OBSERVATION_GATE_ADMISSION_RUN",
    "RECORD_CONFIG",
    "RUNNER_OPTIONS",
    "EstimatorUnavailableError",
    "FAMILY_PROTOCOLS",
    "IIM_MAX_MACRO_NODES",
    "Member",
    "IimV5Scorer",
    "MacroGrainTooLargeError",
    "MANIPULATION_FAMILY",
    "ModuleScorer",
    "NasV3Scorer",
    "PDI_OPTION_NAMES",
    "NOT_APPLICABLE",
    "PROTOCOL_OPTION_NAMES",
    "RAM_FACET_NAMES",
    "RUNNER_VERSION",
    "ResolvedProtocol",
    "RunPolicyError",
    "RunSettings",
    "SCORERS",
    "Scorer",
    "ScorerOutput",
    "ScoringContext",
    "SYSTEM_BUILDERS",
    "SYSTEM_LABEL_META_KEYS",
    "VIEWS",
    "adjusted_mutual_info",
    "build_agent_system",
    "build_catalogue_system",
    "build_tasks",
    "calibration_pending",
    "call_kwargs",
    "check_plan",
    "curtailment_controllers",
    "draft_payload",
    "draft_protocol",
    "iim_cut_modes_read",
    "iim_input_group",
    "iim_v5_params",
    "load_design_module",
    "main",
    "nas_v3_params",
    "null_seed",
    "output_from_result",
    "pdi_v3_params",
    "prepare_resume",
    "ram_v3_params",
    "read_results",
    "recorded_inputs",
    "run_manipulation",
    "register_scorer",
    "register_system_builder",
    "register_view",
    "resolve_protocols",
    "run_task",
    "run_tasks",
    "scorer_for",
    "scorer_override",
    "score_simulation",
    "shard",
    "simulation_block",
    "smoke_done",
    "smoke_summary",
    "split_runner_options",
    "system_config_digest",
    "translate_facets",
    "v1_params",
    "view_regime",
    "write_components_csv",
]
