"""
Null-calibration design of MPC-Bench: null systems (data without the
mechanism each estimator measures) in a grid of cells, scored with the
declaration ``none`` under the A-R classification.

Null kinds (data generated here, seeded per replicate with
:func:`system_seed`):

- ``ar1``: independent AR(1) per node (``ar_coef``);
- ``pink``: independent 1/f noise per node (spectral exponent ``pink_beta``);
- ``surrogate_iid``: per-node (independent-phase) Fourier surrogates of a
  coupled VAR(1): every node keeps its spectrum, cross-node dependence is
  destroyed;
- ``surrogate_linear``: multivariate (shared-phase) Fourier surrogates of a
  coupled non-linear (tanh) network: a linear Gaussian process with the same
  auto- and cross-spectra. It is a null only for PDI (non-linear repertoire)
  and for the event-locked RAM and SRPI, not for NAS or IIM (``NULL_KINDS``).

Events: a synthetic task schedule (goal cue, stimulus, response with a random
choice, feedback with a random +-1 reward; then self-caused events with
stimulus-identical, phase-matched yoked replays) that is independent of the
data, so RAM and SRPI are also null.

Cells and seeds
---------------
Null kinds ``ar1``, ``pink``, ``surrogate_iid`` and ``surrogate_linear`` x
run length T in {1200, 2400} samples (20 Hz) x {8, 16} nodes: 16 cells.
Confirmatory: seed base 20000 and 50 replicates per cell (800 tasks).
Development: seed base 400 and ``ceil(0.15 x 50) = 8`` replicates per cell
(128 tasks; the dry run at about 15 % scale). The task seed is the seed base
(the seed policy classifies task seeds and seed bases; seeds derived inside a
task are not classified); the system of replicate ``r`` of a cell is drawn
from ``_seed(base, kind, T, nodes, r)`` (:func:`system_seed`), computed
inside the task by the system builder, so building a plan derives no seed.

Surrogate draws: the runner derives a task's null seed from the task seed
(``seed * 1000 + 17``), so every task of one seed base, in every cell,
draws its surrogates from the same stream. The draws do not depend on the
data, so each replicate's test keeps its level, but the replicates of a cell
are independent only given those shared draws, not unconditionally: a
cell's PRESENT count is binomial given the draws. Per-replicate draws would
need a per-task key in the runner's null-seed rule.

Partition (the hub partition)
-----------------------------
The hub is the first quarter of the nodes and the remaining nodes form
three equal periphery blocks (``P1``, ``P2``, ``P3``), declared as the
system's ``modules`` with the hub as ``workspace_nodes``. NAS reads the hub
and the three blocks; PDI's content bearer excludes the hub; IIM takes the
means of ``iim_macro_nodes = 4`` equal groups of the nodes outside the hub.

Scoring
-------
Every principle that :data:`NULL_KINDS` names for the null kind is scored
once, with the declaration ``none`` (no inputs), by the estimator versions
of the family-A protocol, and judged on the A-R anchors and cutoffs: the
classification protocol ``A-none`` is the A-R protocol with the declaration
``none`` and nothing else changed (:func:`classification_protocol`). Its
development draft is derived from the A-R protocol file in the runner's
default protocol directory when that exists, else from the A-R draft
(:func:`draft_protocol`; a run with another protocol directory needs its own
A-none file there for the two to share anchors); ``scripts/null_calibration.py
protocol`` writes the protocol file from an A-R protocol file in the same
way (the shipped ``protocols/bench/A-none.json`` also drops IIM's reported
cut, which no scoring of this design reads). The draft is declared in
:data:`PROTOCOL_DRAFTS` only; the runner registers it with the module
when it collects a plan's protocol keys
(``runner.protocol_keys``), so importing this module changes nothing
in the runner. The scorings report component statuses only (no verdict);
the descriptive verdict rates are recomputed from them by the summary of
``scripts/null_calibration.py``.

SRPI (``srpi-2026.09``) runs through the in-memory path of the runner on the
events table above, which carries the agency pairs (self-caused events with
stimulus-identical, phase-matched yoked replays).

Record contract (what the evaluator reads)
------------------------------------------
Design ``null_calibration`` (vocabulary ``design.null_calibration``),
family ``null``, system = the null kind, task tags ``null_kind``,
``n_time``, ``n_nodes``, ``null_replicate``, ``seed_base`` and ``cell``
(``config.tags.null_kind`` is the hypotheses' ``null_kind`` field;
``n_time`` and ``n_nodes`` are read from the record's simulation block).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from impact_pipeline.bench.designs import (
    CONFIRMATORY,
    DEVELOPMENT,
    Design,
    TaskSpec,
    make_scorings,
    task_id,
)
from impact_pipeline.bench import seeds as S
from impact_pipeline.bench.generators import GENERATOR_VERSION

MODULE = "null_calibration"
DESIGN = "null_calibration"
FAMILY = "null"
BUILDER = "null_calibration"
NULL_CALIBRATION_VERSION = "null-calibration/1.0.0"

# the cell grid
KINDS = ("ar1", "pink", "surrogate_iid", "surrogate_linear")
PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")
# the principles for which each kind is a null (module docstring)
NULL_KINDS = {
    "ar1": PRINCIPLES,
    "pink": PRINCIPLES,
    "surrogate_iid": PRINCIPLES,
    "surrogate_linear": ("RAM", "PDI", "SRPI"),
}
N_TIMES = (1200, 2400)
N_NODES = (8, 16)
CONFIRMATORY_REPLICATES = 50
DRY_RUN_FRACTION = 0.15
REPLICATES = {
    CONFIRMATORY: CONFIRMATORY_REPLICATES,
    DEVELOPMENT: int(math.ceil(DRY_RUN_FRACTION * CONFIRMATORY_REPLICATES)),
}
SEED_BASE = {CONFIRMATORY: S.CONFIRMATORY_SEED_MIN, DEVELOPMENT: 400}

# generator settings of the null systems
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


class NullCalibrationError(ValueError):
    """An invalid null-calibration cell, partition or protocol."""


# --------------------------------------------------------------------------
# null data
# --------------------------------------------------------------------------
def _seed(base, *keys) -> int:
    text = "|".join([str(int(base))] + [str(k) for k in keys])
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _standardize(x):
    x = x - x.mean(axis=1, keepdims=True)
    return x / (x.std(axis=1, keepdims=True) + 1e-12)


def _coupled(n_nodes, n_time, rng, nonlinear=False, radius=0.9, burn=200):
    a = rng.standard_normal((n_nodes, n_nodes))
    a *= float(radius) / max(np.max(np.abs(np.linalg.eigvals(a))), 1e-12)
    x = np.zeros((n_nodes, n_time + burn))
    e = rng.standard_normal(x.shape)
    for k in range(1, x.shape[1]):
        drive = a @ x[:, k - 1]
        x[:, k] = (np.tanh(1.5 * drive) if nonlinear else drive) + e[:, k]
    return x[:, burn:]


def null_timeseries(kind, n_nodes, n_time, rng, *, ar_coef=0.5, pink_beta=1.0):
    """``(n_nodes, n_time)`` z-scored null data of family ``kind``."""
    from impact_pipeline import nulls

    n_nodes, n_time = int(n_nodes), int(n_time)
    if kind == "ar1":
        if not -1.0 < float(ar_coef) < 1.0:
            raise ValueError("ar_coef must be in (-1, 1)")
        e = rng.standard_normal((n_nodes, n_time + 200))
        x = np.zeros_like(e)
        for k in range(1, x.shape[1]):
            x[:, k] = float(ar_coef) * x[:, k - 1] + e[:, k]
        return _standardize(x[:, 200:])
    if kind == "pink":
        f = np.fft.rfftfreq(n_time)
        amp = np.zeros_like(f)
        amp[1:] = f[1:] ** (-float(pink_beta) / 2.0)
        spec = (rng.standard_normal((n_nodes, f.size))
                + 1j * rng.standard_normal((n_nodes, f.size))) * amp[None, :]
        return _standardize(np.fft.irfft(spec, n=n_time, axis=1))
    if kind == "surrogate_iid":
        base = _coupled(n_nodes, n_time, rng)
        return _standardize(nulls.phase_randomize(base, rng, multivariate=False))
    if kind == "surrogate_linear":
        base = _coupled(n_nodes, n_time, rng, nonlinear=True)
        return _standardize(nulls.phase_randomize(base, rng, multivariate=True))
    raise ValueError(
        f"unknown null kind {kind!r}; expected one of {sorted(NULL_KINDS)}"
    )


def null_events(n_time, dt, rng, *, task_fraction=0.5, rhythm_hz=0.25,
                n_phase_bins=4):
    """
    Synthetic, data-independent events table in the MPC-Bench layout: bandit
    trials (goal cue, stimulus +0.6 s, response +1.2 s, feedback +1.8 s,
    random choice and +-1 reward) over the first ``task_fraction`` of the
    run, then self-caused events each followed one rhythm period later by a
    stimulus-identical, phase-matched ``other_caused`` replay.
    """
    from impact_pipeline.bench.generators import EVENT_COLUMNS

    dur = float(n_time) * float(dt)
    period = 1.0 / float(rhythm_hz)
    phi0 = float(rng.uniform(0, 2 * np.pi))

    def _pbin(t):
        ph = (2 * np.pi * rhythm_hz * t + phi0) % (2 * np.pi)
        return int(ph // (2 * np.pi / n_phase_bins))

    rows = []

    def _row(t, d, tt, value, channel, trial=None, choice=None, reward=None,
             yoked=None, eid=None):
        rows.append({"onset": round(t, 6), "duration": d, "trial_type": tt,
                     "value": value, "choice": choice, "reward": reward,
                     "yoked_to": yoked, "phase_bin": _pbin(t),
                     "impact_channel": channel, "trial": trial, "event_id": eid})

    t, i = 1.0, 0
    while t + 2.5 < task_fraction * dur:
        c = int(rng.integers(0, 2))
        r = float(rng.choice([-1.0, 1.0]))
        ch = "behavioural_feedback"
        _row(t, 0.3, "goal_cue", int(rng.integers(0, 2)), ch, i, eid=f"goal{i:04d}")
        _row(t + 0.6, 0.3, "stimulus", int(rng.integers(0, 2)), ch, i,
             eid=f"stim{i:04d}")
        _row(t + 1.2, 0.15, "response", c, ch, i, choice=c, eid=f"resp{i:04d}")
        _row(t + 1.8, 0.3, "feedback", r, ch, i, choice=c, reward=r,
             eid=f"fb{i:04d}")
        t += 3.0 + float(rng.uniform(0, 1))
        i += 1
    j = 0
    t = max(t, task_fraction * dur)
    while t + period + 1.0 < dur:
        sid = int(rng.integers(0, 4))
        _row(t, 0.2, "self_caused", sid, "agency", eid=f"self{j:04d}")
        _row(t + period, 0.2, "other_caused", sid, "agency", yoked=f"self{j:04d}",
             eid=f"other{j:04d}")
        # next self-caused event after this replay (no overlap with it)
        t += period + 1.5 + float(rng.uniform(0, 1))
        j += 1
    df = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    df = df.sort_values(["onset", "event_id"], kind="mergesort").reset_index(drop=True)
    for col in ("trial", "choice"):
        df[col] = df[col].astype("Int64")
    return df


def null_system(kind, n_nodes, n_time, seed, *, dt=0.05, iim_macro_nodes=4,
                ar_coef=0.5, pink_beta=1.0):
    """A ``BenchSystem`` holding null data, a null event design and the meta."""
    from impact_pipeline.bench.generators import BenchSystem

    rng = np.random.default_rng(int(seed))
    ts = null_timeseries(kind, n_nodes, n_time, rng, ar_coef=ar_coef,
                         pink_beta=pink_beta)
    events = null_events(n_time, dt, rng)
    n_ws = max(2, int(n_nodes) // 5)
    rest = list(range(n_ws, int(n_nodes)))
    k = max(2, min(int(iim_macro_nodes), len(rest)))
    groups = np.array_split(np.asarray(rest), k)
    meta = {
        "family": "null_calibration", "null_kind": kind, "seed": int(seed),
        "dt": float(dt), "bearer_nodes": list(range(int(n_nodes))),
        "workspace_nodes": list(range(n_ws)),
        "iim_macro_nodes": {f"m{i}": [int(v) for v in g] for i, g in enumerate(groups)},
        "iim_grain": f"null_macro_{k}", "iim_bins": 2, "iim_lag_samples": 2,
    }
    return BenchSystem(ts=ts, events=events, meta=meta, oracle={"null_kind": kind})


def applicable_principles(kind: str) -> Tuple[str, ...]:
    """The principles for which ``kind`` is a null (:data:`NULL_KINDS`)."""
    if kind not in NULL_KINDS:
        raise NullCalibrationError(f"unknown null kind {kind!r}; one of "
                                   f"{sorted(NULL_KINDS)}")
    return tuple(NULL_KINDS[kind])


def system_seed(seed_base: int, kind: str, n_time: int, n_nodes: int,
                replicate: int) -> int:
    """The seed of one null system, derived from the seed base
    (``_seed(base, kind, T, nodes, replicate)``)."""
    return int(_seed(int(seed_base), str(kind), int(n_time), int(n_nodes),
                     int(replicate)))


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
    """The IIM macro nodes by the grain rule of :func:`null_system` (``k``
    equal groups of the nodes outside the workspace, ``np.array_split``
    order) and the grain label."""
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
    The null system of one task: :func:`null_system` at the replicate seed
    (:func:`system_seed`), with the meta declaring the hub partition
    (:func:`hub_partition`) and the IIM grain outside the hub.
    """
    from impact_pipeline.bench.generators import BenchSystem

    p = _params(task)
    kind, n_time, n_nodes = str(p["kind"]), int(p["n_time"]), int(p["n_nodes"])
    rep, base = int(p["null_replicate"]), int(p["seed_base"])
    applicable_principles(kind)
    part = hub_partition(n_nodes)
    seed = system_seed(base, kind, n_time, n_nodes, rep)
    k_macro = int(p.get("iim_macro_nodes", IIM_MACRO_NODES))
    null = null_system(kind, n_nodes, n_time, seed, dt=float(p.get("dt", DT)),
                       iim_macro_nodes=k_macro,
                       ar_coef=float(p.get("ar_coef", AR_COEF)),
                       pink_beta=float(p.get("pink_beta", PINK_BETA)))
    macro, grain = iim_macro_nodes(n_nodes, part["workspace_nodes"], k_macro)
    meta = dict(null.meta)
    meta.update(part)
    meta.update({
        "iim_macro_nodes": macro,
        "iim_grain": grain,
        "generator_version": GENERATOR_VERSION,
        "null_calibration_version": NULL_CALIBRATION_VERSION,
        "seed_base": base,
        "null_replicate": rep,
    })
    return BenchSystem(ts=null.ts, events=null.events, meta=meta,
                       oracle=dict(null.oracle))


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
    from impact_pipeline import evidence as E

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
    of the A-R protocol file when it exists, else of the runner's A-R
    draft (anchors pending)."""
    from impact_pipeline import evidence as E
    from impact_pipeline.bench import runner as RB

    path = RB.protocol_path(ANCHOR_PROTOCOL)
    base = (E.Protocol.from_json(path) if path.is_file()
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
        "aims": "calibration of PRESENT on data without any mechanism, "
                "interval calibration at the null and no PDI states without "
                "contents",
        "data": "the null systems of this design: independent "
                "AR(1) (0.5), independent 1/f noise (beta 1), per-node phase-"
                "randomised surrogates of a coupled VAR(1) and multivariate "
                "phase-randomised surrogates of a coupled tanh network (a null for "
                "RAM-PE, PDI and SRPI only), with data-independent bandit and agency "
                "events; T in {1200, 2400} samples at 20 Hz, 8 and 16 nodes, hub = "
                "first quarter of the nodes and three equal periphery blocks; 50 "
                "replicates per cell from seed base 20000",
        "estimands": "the false-PRESENT rate per (cell, principle); kappa0 = SD(c) / "
                     "RMS(se_c) and both one-sided tail rates per principle (NAS per "
                     "direction); the PDI PRESENT rate per cell",
        "methods": "the principles for which the kind is a null, by the estimator "
                   "versions of the family-A protocol with the declaration none, "
                   "judged by tost on the A-R anchors and cutoffs (protocol "
                   "A-none); component statuses only (no verdict)",
        "performance": "H0 cell rule with bound alpha + 0.02 = 0.07; kappa0 "
                       "chi-square interval and cluster-robust tail bounds; rate "
                       "tables with Wilson intervals reported",
    },
}

N_CELLS = len(KINDS) * len(N_TIMES) * len(N_NODES)
DESIGNS = (
    Design(DESIGN, FAMILY, "null-calibration design (null data, hub partition, "
           "declaration none under the A-R classification)", cells,
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
    "NULL_KINDS",
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
    "null_events",
    "null_system",
    "null_timeseries",
    "scorings",
    "system_seed",
]
