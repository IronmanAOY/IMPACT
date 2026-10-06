"""
Null-calibration generator v2 of MPC-Bench v2 (design 3.1, 3.2, 3.7;
HCv2-1, HCv2-3, HCv2-18).

The null data are those of the v1 calibration: the generator functions of
``scripts/null_calibration.py`` (``null_system`` with ``null_timeseries``
and ``null_events``, the replicate seed derivation ``_seed`` and the
applicability table ``NULL_KINDS``) are imported read-only from the v1
script, never copied (:func:`v1_generator`). A v2 null system holds, sample
for sample, the recording and the events table that v1 draws for the same
replicate seed; only the declared partition of the nodes is new.

Cells and seeds
---------------
Null kinds ``ar1``, ``pink``, ``surrogate_iid`` and ``surrogate_linear`` x
run length T in {1200, 2400} samples (20 Hz) x {8, 16} nodes: 16 cells.
Confirmatory: seed base 20000 and 50 replicates per cell (800 tasks).
Development: seed base 400 and ``ceil(0.15 x 50) = 8`` replicates per cell
(128 tasks; the dry run at about 15 % scale). The task seed is the seed base
(the seed map classifies task seeds and seed bases; seeds derived inside a
task are not classified); the system of replicate ``r`` of a cell is drawn
from the v1 derivation ``_seed(base, kind, T, nodes, r)``, computed inside
the task by the system builder, so building a plan derives no seed.

Surrogate draws: the runner derives a task's null seed from the task seed
(``seed * 1000 + 17``), so every task of one seed base, in every cell,
draws its surrogates from the same stream; v1 drew them per replicate
(``_seed(seed, "null")``). The draws do not depend on the data, so each
replicate's test keeps its level, but the replicates of a cell are
independent only given those shared draws, not unconditionally: a cell's
PRESENT count is binomial given the draws. Per-replicate draws would need a
per-task key in the runner's null-seed rule.

Partition (the hub partition)
-----------------------------
The hub is the first quarter of the nodes and the remaining nodes form
three equal periphery blocks (``P1``, ``P2``, ``P3``), declared as the
system's ``modules`` with the hub as ``workspace_nodes``. NAS reads the hub
and the three blocks; PDI's content bearer excludes the hub; IIM keeps the
v1 grain rule (means of ``iim_macro_nodes = 4`` equal groups of the nodes
outside the workspace), now outside the v2 hub. With 8 nodes the hub and
the IIM grain are those of v1; with 16 nodes v1's workspace was the first
three nodes.

Scoring
-------
Every principle that the v1 applicability table names for the null kind
(``surrogate_linear`` keeps linear cross-dependence and is a null only for
RAM-PE, PDI and SRPI) is scored once, with the declaration ``none`` (no
inputs), by the estimator versions of the family-A protocol, and judged on
the A-R anchors and cutoffs: the classification protocol ``A-none`` is the
A-R protocol with the declaration ``none`` and nothing else changed
(:func:`classification_protocol`). Its development draft is derived from
the generated A-R protocol in the runner's default protocol directory when
that exists, else from the A-R draft (:func:`draft_protocol`; a run with
another protocol directory needs its own A-none file there for the two to
share anchors); the protocol builder writes the generated file
from the generated A-R in the same way (``scripts/v2/null_calibration_v2.py
protocol``). The draft is declared in :data:`PROTOCOL_DRAFTS` only; the
runner registers it with the module when it collects a plan's protocol keys
(``run_bench_v2.protocol_keys``), so importing this module changes nothing
in the runner. The scorings report component statuses only (no verdict);
v1's descriptive verdict rates are recomputed from them by the summary of
``scripts/v2/null_calibration_v2.py``.

SRPI (the v1 instrument, ``srpi-v2-2026.09``) runs through the v1 in-memory
path of the runner on the v1 events table, which carries the agency pairs
(self-caused events with stimulus-identical, phase-matched yoked replays),
so the parsed agency events equal v1's and the SRPI component equals that
of v1's bench path for the same system and null seed (the null seed itself
is the runner's, above); v1's fallback for a runner without agency events
(``srpi_agency_component``) is never needed, and the summary counts any
component that would have needed it.

Record contract (what the evaluator and the integrity audit read)
-----------------------------------------------------------------
Design ``null_calibration`` (vocabulary ``design.null_calibration``),
family ``null``, system = the null kind, task tags ``null_kind``,
``n_time``, ``n_nodes``, ``null_replicate``, ``seed_base`` and ``cell``
(``config.tags.null_kind`` is the hypotheses' ``null_kind`` field;
``n_time`` and ``n_nodes`` are read from the record's simulation block).
"""

from __future__ import annotations

import importlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline.bench.designs_v2 import (
    CONFIRMATORY,
    DEVELOPMENT,
    Design,
    TaskSpec,
    make_scorings,
    task_id,
)
from impact_pipeline.v2 import GENERATOR_VERSION_V2
from impact_pipeline.v2 import seeds as S

MODULE = "null_calibration"
DESIGN = "null_calibration"
FAMILY = "null"
BUILDER = "null_calibration"
NULL_CALIBRATION_VERSION = "null-calibration/3.0.0"
# the meta family of the v1 null systems (the recording device takes the
# family-A coupling time scale for it)
META_FAMILY = "null_calibration"

# the cell grid (design 3.2)
KINDS = ("ar1", "pink", "surrogate_iid", "surrogate_linear")
N_TIMES = (1200, 2400)
N_NODES = (8, 16)
CONFIRMATORY_REPLICATES = 50
DRY_RUN_FRACTION = 0.15
REPLICATES = {
    CONFIRMATORY: CONFIRMATORY_REPLICATES,
    DEVELOPMENT: int(math.ceil(DRY_RUN_FRACTION * CONFIRMATORY_REPLICATES)),
}
SEED_BASE = {CONFIRMATORY: S.CONFIRMATORY_SEED_MIN, DEVELOPMENT: 400}

# v1 generator settings of the stored v1 calibration (its null_calibration.json)
DT = 0.05
IIM_MACRO_NODES = 4
AR_COEF = 0.5
PINK_BETA = 1.0

# the hub partition
PARTITION = "hub_first_quarter_three_equal_blocks"
HUB_MODULE = "hub"
PERIPHERY_BLOCKS = ("P1", "P2", "P3")

# declaration and classification protocol
DECLARATION = "none"
ANCHOR_PROTOCOL = "A-R"
PROTOCOL_KEY = "A-none"
NULL_KIND_TAG = "null_kind"

REPO_ROOT = Path(__file__).resolve().parents[4]
V1_MODULE = "scripts.null_calibration"
V1_SCRIPT = REPO_ROOT / "scripts" / "null_calibration.py"


class NullCalibrationError(ValueError):
    """An invalid null-calibration cell, partition or protocol, or a v1
    generator that is not this checkout's."""


# --------------------------------------------------------------------------
# the v1 generator (imported, never copied)
# --------------------------------------------------------------------------
def v1_generator():
    """
    The v1 null-calibration script ``scripts/null_calibration.py`` as the
    module ``scripts.null_calibration`` (the name the regression gate
    imports it under, so both share one module object). The checkout root
    is appended to ``sys.path`` when the runner was started without it; a
    module of that name from anywhere but this checkout is refused.
    """
    try:
        mod = importlib.import_module(V1_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name not in ("scripts", V1_MODULE):
            raise
        if not V1_SCRIPT.is_file():
            raise NullCalibrationError(
                f"the v1 null-calibration generator ({V1_SCRIPT}) is not in this "
                "installation; run from a checkout of the repository") from None
        root = str(REPO_ROOT)
        if root not in sys.path:
            sys.path.append(root)
        importlib.invalidate_caches()
        mod = importlib.import_module(V1_MODULE)
    origin = Path(getattr(mod, "__file__", "") or "").resolve()
    if origin != V1_SCRIPT.resolve():
        raise NullCalibrationError(
            f"{V1_MODULE} resolves to {origin}, not to this checkout's {V1_SCRIPT}")
    return mod


def applicable_principles(kind: str) -> Tuple[str, ...]:
    """The principles for which ``kind`` is a null (v1's ``NULL_KINDS``)."""
    table = v1_generator().NULL_KINDS
    if kind not in table:
        raise NullCalibrationError(f"unknown null kind {kind!r}; one of "
                                   f"{sorted(table)}")
    return tuple(table[kind])


def system_seed(seed_base: int, kind: str, n_time: int, n_nodes: int,
                replicate: int) -> int:
    """The seed of one null system: v1's derivation from the seed base
    (``_seed(base, kind, T, nodes, replicate)``)."""
    return int(v1_generator()._seed(int(seed_base), str(kind), int(n_time),
                                    int(n_nodes), int(replicate)))


# --------------------------------------------------------------------------
# the hub partition
# --------------------------------------------------------------------------
def hub_partition(n_nodes: int) -> dict:
    """
    The declared partition of ``n_nodes`` nodes: the hub is the first
    quarter, the other nodes form three equal periphery blocks of at least
    two nodes (NAS merges smaller blocks), so ``n_nodes`` is a multiple of
    4 and at least 8.
    """
    n = int(n_nodes)
    if n < 8 or n % 4:
        raise NullCalibrationError(
            f"the hub partition needs a node count that is a multiple of 4 and at "
            f"least 8 (a hub of the first quarter and three equal periphery "
            f"blocks of at least two nodes), got {n_nodes!r}")
    q = n // 4
    modules = {HUB_MODULE: list(range(q))}
    for j, name in enumerate(PERIPHERY_BLOCKS, start=1):
        modules[name] = list(range(j * q, (j + 1) * q))
    return {
        "workspace_nodes": list(range(q)),
        "modules": modules,
        "module_order": [HUB_MODULE, *PERIPHERY_BLOCKS],
        "partition": PARTITION,
    }


def iim_macro_nodes(n_nodes: int, workspace: Sequence[int],
                    k: int = IIM_MACRO_NODES) -> Tuple[Dict[str, List[int]], str]:
    """The IIM macro nodes by v1's grain rule (``k`` equal groups of the
    nodes outside the workspace, ``np.array_split`` order) and the grain
    label."""
    ws = {int(i) for i in workspace}
    rest = [i for i in range(int(n_nodes)) if i not in ws]
    k = max(2, min(int(k), len(rest)))
    groups = np.array_split(np.asarray(rest), k)
    return ({f"m{i}": [int(v) for v in g] for i, g in enumerate(groups)},
            f"null_macro_{k}")


# --------------------------------------------------------------------------
# the system builder
# --------------------------------------------------------------------------
def _params(task: TaskSpec) -> dict:
    p = dict(task.params)
    need = ("kind", "n_time", "n_nodes", "null_replicate", "seed_base")
    missing = [k for k in need if k not in p]
    if missing:
        raise NullCalibrationError(f"{task.task_id}: params lack {missing}")
    if int(p["seed_base"]) != int(task.seed):
        raise NullCalibrationError(f"{task.task_id}: the task seed is the seed base")
    if task.replicate:
        raise NullCalibrationError(f"{task.task_id}: null systems have no twins")
    return p


def build_null_system(task: TaskSpec):
    """
    The null system of one task: v1's ``null_system`` at the replicate seed
    (:func:`system_seed`), with v1's recording and events table unchanged
    and the meta declaring the hub partition (:func:`hub_partition`) and
    the IIM grain outside the hub.
    """
    from impact_pipeline.bench.generators import BenchSystem

    p = _params(task)
    kind, n_time, n_nodes = str(p["kind"]), int(p["n_time"]), int(p["n_nodes"])
    rep, base = int(p["null_replicate"]), int(p["seed_base"])
    applicable_principles(kind)
    part = hub_partition(n_nodes)
    nc = v1_generator()
    seed = system_seed(base, kind, n_time, n_nodes, rep)
    k_macro = int(p.get("iim_macro_nodes", IIM_MACRO_NODES))
    v1 = nc.null_system(kind, n_nodes, n_time, seed, dt=float(p.get("dt", DT)),
                        iim_macro_nodes=k_macro,
                        ar_coef=float(p.get("ar_coef", AR_COEF)),
                        pink_beta=float(p.get("pink_beta", PINK_BETA)))
    macro, grain = iim_macro_nodes(n_nodes, part["workspace_nodes"], k_macro)
    meta = dict(v1.meta)
    meta.update(part)
    meta.update({
        "iim_macro_nodes": macro,
        "iim_grain": grain,
        "generator_version": GENERATOR_VERSION_V2,
        "null_calibration_version": NULL_CALIBRATION_VERSION,
        "seed_base": base,
        "null_replicate": rep,
    })
    return BenchSystem(ts=v1.ts, events=v1.events, meta=meta, oracle=dict(v1.oracle))


SYSTEM_BUILDERS = {BUILDER: build_null_system}


# --------------------------------------------------------------------------
# the classification protocol
# --------------------------------------------------------------------------
_ANCHOR_NAME = re.compile(r"(?<![A-Za-z0-9])" + re.escape(ANCHOR_PROTOCOL)
                          + r"(?![A-Za-z0-9])")


def classification_protocol(base: Mapping) -> dict:
    """
    The protocol that judges the null-calibration components: the A-R
    protocol ``base`` (a payload with declaration R; anchors, cutoffs,
    estimators, SE methods and precision block kept) with the declaration
    ``none`` and its name renamed from A-R to A-none.
    """
    from impact_pipeline import evidence_v2 as E

    payload = json.loads(json.dumps(dict(base)))
    decl = (payload.get("shared_inputs_declaration") or {}).get("id")
    if decl != "R":
        raise NullCalibrationError(
            f"the classification protocol is derived from the {ANCHOR_PROTOCOL} "
            f"protocol (declaration R), got declaration {decl!r}")
    payload["shared_inputs_declaration"] = {
        "id": DECLARATION, "shared_inputs": E.DECLARATION_LEVELS[DECLARATION]}
    name = payload.get("name")
    if name is None:
        payload["name"] = PROTOCOL_KEY
    else:
        renamed = _ANCHOR_NAME.sub(PROTOCOL_KEY, str(name))
        payload["name"] = renamed if renamed != name else f"{name}:{PROTOCOL_KEY}"
    return payload


def draft_protocol() -> dict:
    """The development draft of ``A-none``: :func:`classification_protocol`
    of the generated A-R protocol when it exists, else of the runner's A-R
    draft (anchors pending)."""
    from impact_pipeline import evidence_v2 as E
    from impact_pipeline.bench import run_bench_v2 as RB

    path = RB.protocol_path(ANCHOR_PROTOCOL)
    base = (E.ProtocolV3.from_json(path) if path.is_file()
            else RB.draft_protocol(ANCHOR_PROTOCOL))
    return classification_protocol(base.to_dict())


PROTOCOL_DRAFTS = {PROTOCOL_KEY: draft_protocol}


# --------------------------------------------------------------------------
# tasks and the design
# --------------------------------------------------------------------------
def scorings(kind: str):
    """One scoring: declaration ``none`` under ``A-none``, the principles
    for which the kind is a null, component statuses only."""
    return make_scorings({DECLARATION: PROTOCOL_KEY}, applicable_principles(kind),
                         verdict=False)


def cell_id(kind: str, n_time: int, n_nodes: int) -> str:
    return f"{kind}:T{int(n_time)}:N{int(n_nodes)}"


def cell_task(kind: str, n_time: int, n_nodes: int, replicate: int,
              seed_base: int) -> TaskSpec:
    """The task of replicate ``replicate`` of one cell from a seed base."""
    if kind not in KINDS:
        raise NullCalibrationError(f"unknown null kind {kind!r}; one of {KINDS}")
    hub_partition(n_nodes)
    n_time, n_nodes, rep = int(n_time), int(n_nodes), int(replicate)
    if n_time <= 0 or rep < 0:
        raise NullCalibrationError("n_time must be > 0 and replicate >= 0")
    base = int(seed_base)
    return TaskSpec(
        task_id=task_id(DESIGN, kind, base,
                        suffix=f"T{n_time}-N{n_nodes}-rep{rep:03d}"),
        design=DESIGN, family=FAMILY, system=kind, seed=base, builder=BUILDER,
        params={"kind": kind, "n_time": n_time, "n_nodes": n_nodes,
                "null_replicate": rep, "seed_base": base, "dt": DT,
                "iim_macro_nodes": IIM_MACRO_NODES, "ar_coef": AR_COEF,
                "pink_beta": PINK_BETA},
        scorings=scorings(kind),
        tags={NULL_KIND_TAG: kind, "n_time": n_time, "n_nodes": n_nodes,
              "null_replicate": rep, "seed_base": base,
              "cell": cell_id(kind, n_time, n_nodes)},
        design_module=MODULE)


def _select(values, allowed, what) -> Tuple:
    if values is None:
        return tuple(allowed)
    vals = tuple(values)
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise NullCalibrationError(f"unknown {what} {bad}; one of {tuple(allowed)}")
    return vals


def cells(split: str, *, seeds: Optional[Sequence[int]] = None,
          systems: Optional[Sequence[str]] = None,
          kinds: Optional[Sequence[str]] = None,
          n_times: Optional[Sequence[int]] = None,
          n_nodes: Optional[Sequence[int]] = None,
          replicates: Optional[int] = None) -> List[TaskSpec]:
    """
    ``null_calibration``: every cell (kind x T x nodes) with its replicates
    from each seed base (``seeds``; default the split's base). ``systems``
    or ``kinds`` restrict the null kinds; ``n_times``, ``n_nodes`` and
    ``replicates`` restrict the grid of a development run.
    """
    if split not in (CONFIRMATORY, DEVELOPMENT):
        raise NullCalibrationError(f"split must be one of "
                                   f"{(CONFIRMATORY, DEVELOPMENT)}")
    if systems is not None and kinds is not None and tuple(systems) != tuple(kinds):
        raise NullCalibrationError("systems and kinds name the null kinds twice")
    ks = _select(kinds if kinds is not None else systems, KINDS, "null kinds")
    ts = tuple(int(t) for t in (n_times if n_times is not None else N_TIMES))
    ns = tuple(int(n) for n in (n_nodes if n_nodes is not None else N_NODES))
    n_rep = REPLICATES[split] if replicates is None else int(replicates)
    if n_rep < 1:
        raise NullCalibrationError("replicates must be >= 1")
    bases = tuple(int(s) for s in (seeds if seeds is not None else (SEED_BASE[split],)))
    for b in bases:
        if S.split_of(b) != split:
            raise NullCalibrationError(f"seed base {b} is not a {split} seed")
    out = []
    for base in bases:
        for kind in ks:
            for t in ts:
                for n in ns:
                    for rep in range(n_rep):
                        out.append(cell_task(kind, t, n, rep, base))
    return out


ADEMP = {
    DESIGN: {
        "aims": "calibration of PRESENT on data without any mechanism (HCv2-1), "
                "interval calibration at the null (HCv2-3) and no PDI states "
                "without contents (HCv2-18)",
        "data": "the v1 null-calibration generator imported read-only: independent "
                "AR(1) (0.5), independent 1/f noise (beta 1), per-node phase-"
                "randomised surrogates of a coupled VAR(1) and multivariate "
                "phase-randomised surrogates of a coupled tanh network (a null for "
                "RAM-PE, PDI and SRPI only), with data-independent bandit and agency "
                "events; T in {1200, 2400} samples at 20 Hz, 8 and 16 nodes, hub = "
                "first quarter of the nodes and three equal periphery blocks; 50 "
                "replicates per cell from seed base 20000 (v1 seed derivation)",
        "estimands": "the false-PRESENT rate per (cell, principle); kappa0 = SD(c) / "
                     "RMS(se_c) and both one-sided tail rates per principle (NAS per "
                     "direction); the PDI PRESENT rate per cell",
        "methods": "the principles for which the kind is a null, by the estimator "
                   "versions of the family-A protocol with the declaration none, "
                   "judged by tost-v2 on the A-R anchors and cutoffs (protocol "
                   "A-none); component statuses only (no verdict)",
        "performance": "H0 cell rule with bound alpha + 0.02 = 0.07 (HCv2-1, "
                       "HCv2-18); kappa0 chi-square interval and cluster-robust tail "
                       "bounds (HCv2-3); v1's rate tables with Wilson intervals "
                       "reported",
    },
}

N_CELLS = len(KINDS) * len(N_TIMES) * len(N_NODES)
DESIGNS = (
    Design(DESIGN, FAMILY, "null-calibration generator v2 (v1 null data, hub "
           "partition, declaration none under the A-R classification)", cells,
           {CONFIRMATORY: N_CELLS * REPLICATES[CONFIRMATORY],
            DEVELOPMENT: N_CELLS * REPLICATES[DEVELOPMENT]},
           ademp=ADEMP[DESIGN]),
)


__all__ = [
    "ADEMP",
    "ANCHOR_PROTOCOL",
    "BUILDER",
    "CONFIRMATORY_REPLICATES",
    "DECLARATION",
    "DESIGN",
    "DESIGNS",
    "FAMILY",
    "HUB_MODULE",
    "KINDS",
    "MODULE",
    "NULL_CALIBRATION_VERSION",
    "NULL_KIND_TAG",
    "N_NODES",
    "N_TIMES",
    "NullCalibrationError",
    "PARTITION",
    "PERIPHERY_BLOCKS",
    "PROTOCOL_DRAFTS",
    "PROTOCOL_KEY",
    "REPLICATES",
    "SEED_BASE",
    "SYSTEM_BUILDERS",
    "applicable_principles",
    "build_null_system",
    "cell_id",
    "cell_task",
    "cells",
    "classification_protocol",
    "draft_protocol",
    "hub_partition",
    "iim_macro_nodes",
    "scorings",
    "system_seed",
    "v1_generator",
]
