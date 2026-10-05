"""
Family-B design of MPC-Bench v2: the IIM v5 validation cells on binary
networks with exact TPMs (hypotheses HCv2-2, HCv2-11, HCv2-12 and HCv2-13).

Every cell is one simulated system (a family-B network of
:func:`~impact_pipeline.bench.generators.family_b_network`, four units,
``beta = 1``, self-coupling 0.6, unless stated) at one run length ``T``; a
task is one cell at one seed; a task's simulation is scored under one or
more declarations (``none``: the system has no exogenous driver;
``recorded``: the driver is declared and conditioned on; ``hidden``: the
driver exists and is not declared; ``label_error``: the declared driver
labels are flipped with probability ``q``) and always in both cut modes.
The two cut-mode scorings of one declaration share the trajectory, the
surrogates and the resamples, so they form **one cluster** in pooled bounds:
the cluster id of every scoring is its task id. Trajectory seeds are hashed
from (generator, system, parameters, ``T``, seed), never from the cut mode
or the declaration.

Cells (confirmatory seed blocks of ``protocols/v2/seed_map_v2.json``;
development mirrors are disjoint sub-blocks of 400-439):

* **HCv2-2** (20000-20199, n = 200; development 400-414). (i) Independent
  units, ``T`` in {1000, 3000, 10000, 30000}. (ii) Independent units and an
  irrelevant recorded binary Markov driver (persistence 0.9, two strata),
  ``T`` in {3000, 10000, 30000}, stratified. (iii) Four independent AR(1)
  macro signals (coefficients U(0.5, 0.9), 20 Hz, 12,000 samples) plus a
  recorded driver added to every node: (a) a continuous AR(1) driver with
  time constant 0.3 s, (b) a six-level switching driver (dwell U(1, 3) s,
  one N(0, 1) pattern per level) low-pass filtered per node with the time
  constants 0.1, 0.3, 1.0 and 0.3 s (inside the basis span); residualised
  on the common input basis (``tau_c = 0.1 s``, lag 2), shift-then-project
  null, the project-then-shift null reported on the same data. ``K = 19``.
  18 decisive cells (cut modes counted).
* **HCv2-11** (20280-20319, n = 40; development 415-419). Feed-forward star,
  coupling {0, 0.2, ..., 1.0} x ``T`` {10000, 30000}; statuses with the v5
  SE and the rank gate on the family-B scale.
* **HCv2-12** (a: 20240-20279, development 420-424) ring coupling {0, 0.1,
  ..., 0.5} and XOR noise {0.5, 0.4, 0.3, 0.2, 0.15, 0.1} at ``T`` = 30000
  (decisive) and 10000 (reported); (b: 20420-20459, development 425-429)
  ring coupling 0.45, 0.9 and 1.5 at ``T`` = 30000; (c, d: 20460-20499,
  development 430-434) (c) the cells of :data:`OCCUPANCY_CANDIDATES` whose
  expected rarest-row count ``T pi_min`` (exact stationary distribution) is
  ``<= N_min / 2`` (the gate should fire) or ``>= 4 N_min`` (the estimator
  should be defined), with ``N_min`` of IIM v5; (d) all-to-all 0.6 at ``T``
  = 1000, the hypersynchronous system on the family-A layout and the v1
  electrode-quadrant montage under the average reference on Hopf ``G = 0``
  sources (expected reasons ``INSUFFICIENT_OCCUPANCY``,
  ``INSUFFICIENT_OCCUPANCY``, ``MACRO_RANK_DEFICIENT``).
* **HCv2-13** (20320-20419, n = 100; development 435-439). Hidden driver
  (weight 1.0, persistence 0.9), ``T`` {3000, 10000, 30000}, scored with the
  driver recorded (stratified) and hidden; at ``T`` = 10000 also with label
  errors ``q`` in {0.1, 0.25} (each recorded label flipped when its uniform
  draw from ``SeedSequence([trajectory seed, 41])`` is below ``q``, so the
  flips are nested in ``q``). The occupancy gate applies per stratum: the
  rarest state given the driver has an expected count of about 11 pairs at
  ``T`` = 3000 and 37 at 10000 (110 at 30000), so with ``N_min = 25`` the
  recorded scoring is mostly ``INSUFFICIENT_OCCUPANCY`` at 3000 and in part
  at 10000.

Scorings carry the v5 SE only where a hypothesis reads a status (HCv2-11;
HCv2-13 recorded at every ``T``, hidden at ``T`` = 30000); the other
scorings are value-only (excess, ``p_ind``) and are judged under the
protocol without an SE contract (``UNDEFINED(NO_SAMPLING_SE)``).

The construct scale of family B is anchored on the exact value of the ring
at coupling 0.45 (``beta = 1``, self-coupling 0.6, stationary weights), per
cut mode, with reference SE 0; the all-to-all network at coupling 0.4 is
the second anchor, reported. :func:`family_b_protocol` builds the
``impact-mpc-protocol/3`` protocol of a (cut mode, anchor, null family):
necessity set {IIM}, the IIM rank gate, the v5 SE methods and ``exact``
(exact known-TPM values are decided on ``c`` alone). The exact targets
(:func:`exact_targets`) are deterministic constants of the generator.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline.v2 import GENERATOR_VERSION_V2
from impact_pipeline.v2 import iim_v5 as IIM
from impact_pipeline.v2 import seeds as S

DESIGN = "family_b"
DESIGN_VERSION = "mpc-bench-family-b-design/2.0.0"
FAMILY = "B"
SUBSTRATE = "family_b"
VIEW = "macro"
GENERATOR_VERSION = GENERATOR_VERSION_V2
N_UNITS = 4
BETA = 1.0
SELF_COUPLING = 0.6
LAG_BINARY = 1
CUT_MODES = ("directional", "bidirectional")
K_NULL = IIM.N_NULL_FAMILY_B
HASH_LABEL = "iim_validation_v2"

# generators of the cells
GEN_BINARY = "family_b"
GEN_BINARY_MARKOV_DRIVER = "family_b_markov_driver"
GEN_AR1_DRIVER = "ar1_driver"
GEN_HYPERSYNCHRONOUS = "hypersynchronous"
GEN_HOPF_EEG = "whole_brain_eeg"
GENERATORS = (GEN_BINARY, GEN_BINARY_MARKOV_DRIVER, GEN_AR1_DRIVER,
              GEN_HYPERSYNCHRONOUS, GEN_HOPF_EEG)

# declarations of family B (records vocabulary)
DECL_NONE = "none"
DECL_RECORDED = "recorded"
DECL_HIDDEN = "hidden"
DECL_LABEL_ERROR = "label_error"
DECLARATIONS = (DECL_NONE, DECL_RECORDED, DECL_HIDDEN, DECL_LABEL_ERROR)
CONDITIONINGS = ("none", "stratify", "residualise")
LABEL_ERROR_STREAM = "label_error"  # SeedSequence([trajectory seed, 41])

# the irrelevant driver of HCv2-2 (ii), the hidden driver of HCv2-13
MARKOV_DRIVER_PERSISTENCE = 0.9
HIDDEN_DRIVER_WEIGHT = 1.0
HIDDEN_DRIVER_PERSISTENCE = 0.9

# the residualised calibration cells of HCv2-2 (iii)
AR1_DT = 0.05
AR1_N_TIME = 12000
AR1_COEF_RANGE = (0.5, 0.9)
AR1_TAU_C = 0.1
AR1_LAG = 2
AR1_DRIVER_WEIGHT = 1.0
AR1_CONTINUOUS_DRIVER_TAU = 0.3
SWITCHING_LEVELS = 6
SWITCHING_DWELL_SEC = (1.0, 3.0)
SWITCHING_NODE_TAUS = (0.1, 0.3, 1.0, 0.3)
DRIVER_KINDS = ("continuous_ar1", "switching_filtered")

# anchors of the family-B construct scale (exact values, CD-1)
ANCHORS = {
    "ring_0.45": ("ring", (("coupling", 0.45),)),
    "all_to_all_0.4": ("all_to_all", (("coupling", 0.4),)),
}
PRIMARY_ANCHOR = "ring_0.45"
SECOND_ANCHOR = "all_to_all_0.4"
# the design's values of the primary anchor (bits), to four significant digits
ANCHOR_DESIGN_VALUES = {("ring_0.45", "bidirectional"): 0.04513,
                        ("ring_0.45", "directional"): 0.02909}

# HCv2-12 (a), (b) and (c)
RING_SWEEP = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
XOR_SWEEP = (0.5, 0.4, 0.3, 0.2, 0.15, 0.1)
NON_MONOTONE_RING = (0.45, 0.9, 1.5)
OCCUPANCY_T = (1000, 3000, 10000, 30000)
OCCUPANCY_CANDIDATES = (
    ("ring", (("coupling", 0.45),)),
    ("ring", (("coupling", 0.9),)),
    ("all_to_all", (("coupling", 0.4),)),
    ("all_to_all", (("coupling", 0.6),)),
    ("xor_loop", (("noise", 0.1),)),
)
STAR_COUPLINGS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# seed blocks: confirmatory blocks are the seed map's family-B parts;
# development mirrors are disjoint sub-blocks of 400-439
SEED_BLOCKS = {
    "HCv2-2": {S.CONFIRMATORY: (20000, 20199), S.DEVELOPMENT: (400, 414)},
    "HCv2-11": {S.CONFIRMATORY: (20280, 20319), S.DEVELOPMENT: (415, 419)},
    "HCv2-12(a)": {S.CONFIRMATORY: (20240, 20279), S.DEVELOPMENT: (420, 424)},
    "HCv2-12(b)": {S.CONFIRMATORY: (20420, 20459), S.DEVELOPMENT: (425, 429)},
    "HCv2-12(c, d)": {S.CONFIRMATORY: (20460, 20499), S.DEVELOPMENT: (430, 434)},
    "HCv2-13": {S.CONFIRMATORY: (20320, 20419), S.DEVELOPMENT: (435, 439)},
}
HYPOTHESES = ("HCv2-2", "HCv2-11", "HCv2-12", "HCv2-13")


# --------------------------------------------------------------------------
# cells, scorings, tasks
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Scoring:
    """One declaration under which a task's simulation is scored (in both
    cut modes): ``conditioning`` is ``none``, ``stratify`` (binary driver)
    or ``residualise`` (input basis); ``se`` asks for the v5 sampling SE
    (statuses need it); ``role`` is ``decisive`` or ``reported``."""

    declaration: str
    conditioning: str = "none"
    label_error_q: float = 0.0
    null_order: str = IIM.NULL_ORDER_SHIFT_THEN_PROJECT
    se: bool = False
    role: str = "decisive"

    def __post_init__(self):
        if self.declaration not in DECLARATIONS:
            raise ValueError(f"declaration must be one of {DECLARATIONS}")
        if self.conditioning not in CONDITIONINGS:
            raise ValueError(f"conditioning must be one of {CONDITIONINGS}")
        if self.null_order not in IIM.NULL_ORDERS:
            raise ValueError(f"null_order must be one of {IIM.NULL_ORDERS}")
        if not 0.0 <= float(self.label_error_q) < 0.5:
            raise ValueError("label_error_q must lie in [0, 0.5)")
        if (self.declaration == DECL_LABEL_ERROR) != (float(self.label_error_q) > 0):
            raise ValueError("label_error_q > 0 exactly for the label_error declaration")
        if self.role not in ("decisive", "reported"):
            raise ValueError("role must be decisive or reported")

    @property
    def key(self) -> str:
        """Scoring key (a record token)."""
        k = self.declaration
        if self.declaration == DECL_LABEL_ERROR:
            k += f"={self.label_error_q:g}"
        if self.conditioning == "residualise":
            k += f":{self.null_order}"
        return k

    @property
    def shared_inputs(self) -> str:
        """Identifiability level: ``complete`` when every exogenous driver is
        declared (or none exists), ``partial`` with label errors, ``none``
        when the driver is hidden."""
        return {DECL_NONE: "complete", DECL_RECORDED: "complete",
                DECL_LABEL_ERROR: "partial", DECL_HIDDEN: "none"}[self.declaration]

    def to_dict(self) -> dict:
        return {"declaration": self.declaration, "conditioning": self.conditioning,
                "label_error_q": float(self.label_error_q),
                "null_order": self.null_order, "se": bool(self.se), "role": self.role,
                "key": self.key}


def _params_tuple(params) -> Tuple[Tuple[str, object], ...]:
    if isinstance(params, Mapping):
        params = params.items()
    return tuple(sorted((str(k), v) for k, v in params))


def _value_token(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


@dataclass(frozen=True)
class Cell:
    """One family-B cell: a system at one run length, its hypothesis, parts
    and seed block, the transition lag, ``K``, the cut modes and the
    scorings. ``expected`` holds a part's declared expectation where the
    cell exists for it (the occupancy designation of HCv2-12 (c), the
    reason of (d))."""

    hypothesis: str
    parts: Tuple[str, ...]
    block: str
    generator: str
    system: str
    params: Tuple[Tuple[str, object], ...]
    n_time: int
    lag: int
    scorings: Tuple[Scoring, ...]
    n_null: int = K_NULL
    cut_modes: Tuple[str, ...] = CUT_MODES
    role: str = "decisive"
    expected: Tuple[Tuple[str, object], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "params", _params_tuple(self.params))
        object.__setattr__(self, "expected", _params_tuple(self.expected))
        if self.generator not in GENERATORS:
            raise ValueError(f"generator must be one of {GENERATORS}")
        if self.block not in SEED_BLOCKS:
            raise ValueError(f"unknown seed block {self.block!r}")
        if not self.scorings:
            raise ValueError("a cell needs at least one scoring")
        keys = [s.key for s in self.scorings]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate scorings")
        if any(c not in IIM.CUT_MODES for c in self.cut_modes):
            raise ValueError("unknown cut mode")

    @property
    def param_dict(self) -> dict:
        return dict(self.params)

    @property
    def system_key(self) -> str:
        toks = [self.system] + [f"{k}={_value_token(v)}" for k, v in self.params]
        return "/".join(toks)

    @property
    def condition(self) -> Optional[str]:
        """The HCv2-2 null condition of the cell (``independent``,
        ``stratified``, ``residualised_continuous`` or
        ``residualised_switching``; None for the other hypotheses), the label
        the hypotheses select the rank-calibration cells by."""
        if self.hypothesis != "HCv2-2":
            return None
        if self.generator == GEN_AR1_DRIVER:
            kind = self.param_dict.get("driver")
            return {"continuous_ar1": "residualised_continuous",
                    "switching_filtered": "residualised_switching"}.get(kind)
        if self.generator == GEN_BINARY_MARKOV_DRIVER:
            return "stratified"
        return "independent"

    @property
    def cell_id(self) -> str:
        """``<hypothesis>[.<parts>]/<system>[/<param>=<value>...]/T<n>`` (a
        record token; ``T0`` is the generator's own run length)."""
        head = self.hypothesis + (("." + "+".join(self.parts)) if self.parts else "")
        return f"{head}/{self.system_key}/T{self.n_time}"

    def seeds(self, split: str) -> range:
        lo, hi = SEED_BLOCKS[self.block][split]
        return range(lo, hi + 1)

    def to_dict(self) -> dict:
        return {"cell_id": self.cell_id, "hypothesis": self.hypothesis,
                "parts": list(self.parts), "block": self.block,
                "condition": self.condition,
                "generator": self.generator, "system": self.system,
                "params": dict(self.params), "n_time": int(self.n_time),
                "lag": int(self.lag), "n_null": int(self.n_null),
                "cut_modes": list(self.cut_modes), "role": self.role,
                "expected": dict(self.expected),
                "scorings": [s.to_dict() for s in self.scorings]}


def hash_seed(*keys) -> int:
    """A 32-bit seed hashed from ``keys`` (SHA-256 of their ``|``-joined text;
    the v1 rule of the IIM validation)."""
    text = "|".join(str(k) for k in keys)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def trajectory_seed(cell: Cell, seed: int) -> int:
    """The simulation seed of a family-B or synthetic cell, hashed from
    (generator, system, parameters, ``T``, seed): the same for both cut modes
    and every declaration of a task, different for every other cell."""
    return hash_seed(HASH_LABEL, cell.generator, cell.system,
                     _canonical(dict(cell.params)), int(cell.n_time), int(seed))


@dataclass(frozen=True)
class Task:
    """One cell at one seed. ``cluster_id`` (= ``task_id``) groups every
    scoring of the task's simulation, both cut modes included."""

    cell: Cell
    seed: int
    split: str

    @property
    def task_id(self) -> str:
        return f"{FAMILY}:{self.cell.cell_id}:s{int(self.seed):05d}"

    @property
    def cluster_id(self) -> str:
        return self.task_id

    @property
    def trajectory_seed(self) -> int:
        return trajectory_seed(self.cell, self.seed)

    @property
    def simulation_seed(self) -> int:
        """The seed handed to the generator: the hashed trajectory seed for
        family-B and synthetic cells, the task seed for bench generators
        (hypersynchronous system, Hopf model), whose own streams derive from
        it."""
        if self.cell.generator in (GEN_HYPERSYNCHRONOUS, GEN_HOPF_EEG):
            return int(self.seed)
        return self.trajectory_seed

    def scoring_seeds(self, scoring: Scoring) -> Tuple[int, int]:
        """``(null_seed, se_seed)`` of a scoring (shared by its cut modes)."""
        base = self.trajectory_seed
        return (hash_seed(HASH_LABEL, "null", base, scoring.key),
                hash_seed(HASH_LABEL, "se", base, scoring.key))


# --------------------------------------------------------------------------
# cell catalogue
# --------------------------------------------------------------------------
def _binary(hyp, parts, block, system, params, T, scorings, role="decisive",
            expected=()):
    return Cell(hyp, tuple(parts), block, GEN_BINARY, system, params, int(T),
                LAG_BINARY, tuple(scorings), role=role, expected=expected)


def _hcv2_2_cells() -> List[Cell]:
    out = []
    none = (Scoring(DECL_NONE),)
    for T in (1000, 3000, 10000, 30000):
        out.append(_binary("HCv2-2", ("i",), "HCv2-2", "independent", (), T, none))
    rec = (Scoring(DECL_RECORDED, "stratify"),)
    for T in (3000, 10000, 30000):
        out.append(Cell("HCv2-2", ("ii",), "HCv2-2", GEN_BINARY_MARKOV_DRIVER,
                        "independent", (("driver_persistence",
                                         MARKOV_DRIVER_PERSISTENCE),),
                        T, LAG_BINARY, rec))
    resid = (Scoring(DECL_RECORDED, "residualise",
                     null_order=IIM.NULL_ORDER_SHIFT_THEN_PROJECT),
             Scoring(DECL_RECORDED, "residualise",
                     null_order=IIM.NULL_ORDER_PROJECT_THEN_SHIFT, role="reported"))
    for kind in DRIVER_KINDS:
        out.append(Cell("HCv2-2", ("iii",), "HCv2-2", GEN_AR1_DRIVER, "ar1",
                        (("driver", kind),), AR1_N_TIME, AR1_LAG, resid))
    return out


def _hcv2_11_cells() -> List[Cell]:
    sc = (Scoring(DECL_NONE, se=True),)
    return [_binary("HCv2-11", (), "HCv2-11", "feedforward_star",
                    (("coupling", float(c)),), T, sc)
            for T in (10000, 30000) for c in STAR_COUPLINGS]


def occupancy_designation(kind: str, params, n_time: int,
                          n_min: int = IIM.N_MIN_DEFAULT) -> Optional[str]:
    """HCv2-12 (c): ``undefined`` when the expected rarest-row count ``T
    pi_min`` (exact stationary distribution) is ``<= N_min / 2``, ``defined``
    when it is ``>= 4 N_min``, else None (not designated)."""
    t_pi = expected_rarest_count(kind, params, n_time)
    if t_pi <= n_min / 2.0:
        return "undefined"
    if t_pi >= 4.0 * n_min:
        return "defined"
    return None


def expected_rarest_count(kind: str, params, n_time: int) -> float:
    """``T pi_min`` of a family-B network."""
    return float(n_time) * float(_pi_min(kind, _params_tuple(params)))


@functools.lru_cache(maxsize=None)
def _pi_min(kind, params) -> float:
    return float(network(kind, params).stationary.min())


def _hcv2_12_cells(n_min: int = IIM.N_MIN_DEFAULT) -> List[Cell]:
    out = []
    none = (Scoring(DECL_NONE),)
    for T, role in ((30000, "decisive"), (10000, "reported")):
        for c in RING_SWEEP:
            out.append(_binary("HCv2-12", ("a",), "HCv2-12(a)", "ring",
                               (("coupling", float(c)),), T, none, role=role))
        for q in XOR_SWEEP:
            out.append(_binary("HCv2-12", ("a",), "HCv2-12(a)", "xor_loop",
                               (("noise", float(q)),), T, none, role=role))
    for c in NON_MONOTONE_RING:
        out.append(_binary("HCv2-12", ("b",), "HCv2-12(b)", "ring",
                           (("coupling", float(c)),), 30000, none))
    d_key = ("all_to_all", (("coupling", 0.6),), 1000)
    seen_d = False
    for kind, params in OCCUPANCY_CANDIDATES:
        for T in OCCUPANCY_T:
            des = occupancy_designation(kind, params, T, n_min)
            if des is None:
                continue
            parts = ("c",)
            expected = [("occupancy", des),
                        ("t_pi_min", round(expected_rarest_count(kind, params, T), 6)),
                        ("n_min", int(n_min))]
            if (kind, _params_tuple(params), T) == (d_key[0], _params_tuple(d_key[1]),
                                                    d_key[2]):
                parts = ("c", "d")
                expected.append(("reason", "INSUFFICIENT_OCCUPANCY"))
                seen_d = True
            out.append(_binary("HCv2-12", parts, "HCv2-12(c, d)", kind, params, T,
                               none, expected=expected))
    if not seen_d:
        out.append(_binary("HCv2-12", ("d",), "HCv2-12(c, d)", d_key[0], d_key[1],
                           d_key[2], none,
                           expected=(("reason", "INSUFFICIENT_OCCUPANCY"),)))
    out.append(Cell("HCv2-12", ("d",), "HCv2-12(c, d)", GEN_HYPERSYNCHRONOUS,
                    "O_hypersynchronous", (("template_family", "A"),), 0, 0, none,
                    expected=(("reason", "INSUFFICIENT_OCCUPANCY"),)))
    out.append(Cell("HCv2-12", ("d",), "HCv2-12(c, d)", GEN_HOPF_EEG,
                    "hopf_eeg_quadrants", (("G", 0.0), ("reference", "average")), 0, 0,
                    none, expected=(("reason", "MACRO_RANK_DEFICIENT"),)))
    return out


def _hcv2_13_cells() -> List[Cell]:
    out = []
    params = (("driver_persistence", HIDDEN_DRIVER_PERSISTENCE),
              ("driver_weight", HIDDEN_DRIVER_WEIGHT))
    for T in (3000, 10000, 30000):
        # (a) needs statuses of the recorded scoring at every T; (b) needs the
        # hidden scoring's statuses at T = 30000 only (its median c elsewhere)
        sc = [Scoring(DECL_RECORDED, "stratify", se=True),
              Scoring(DECL_HIDDEN, "none", se=(T == 30000))]
        parts = ("a", "b")
        if T == 10000:
            sc += [Scoring(DECL_LABEL_ERROR, "stratify", label_error_q=q)
                   for q in (0.1, 0.25)]
            parts = ("a", "b", "c")
        out.append(_binary("HCv2-13", parts, "HCv2-13", "hidden_driver", params, T, sc))
    return out


def cells(hypotheses: Optional[Iterable[str]] = None,
          n_min: int = IIM.N_MIN_DEFAULT) -> List[Cell]:
    """The family-B cells of the given hypotheses (default all), in a fixed
    order. ``n_min`` sets the occupancy designation of HCv2-12 (c) (the IIM
    v5 ``N_min``; it changes only if calibration raises it)."""
    want = HYPOTHESES if hypotheses is None else tuple(hypotheses)
    unknown = sorted(set(want) - set(HYPOTHESES))
    if unknown:
        raise ValueError(f"unknown hypotheses {unknown}; one of {HYPOTHESES}")
    builders = {"HCv2-2": _hcv2_2_cells, "HCv2-11": _hcv2_11_cells,
                "HCv2-12": lambda: _hcv2_12_cells(n_min), "HCv2-13": _hcv2_13_cells}
    out = []
    for h in HYPOTHESES:
        if h in want:
            out.extend(builders[h]())
    ids = [c.cell_id for c in out]
    if len(set(ids)) != len(ids):  # pragma: no cover - guarded by the tests
        raise ValueError("duplicate family-B cell ids")
    return out


def tasks(split: str = S.DEVELOPMENT, *, hypotheses: Optional[Iterable[str]] = None,
          seeds: Optional[Iterable[int]] = None,
          n_min: int = IIM.N_MIN_DEFAULT) -> List[Task]:
    """The tasks of a split: every cell at every seed of its block (or at the
    given ``seeds`` that fall inside its block). The seeds must belong to the
    split (:mod:`impact_pipeline.v2.seeds`)."""
    if split not in S.SPLITS:
        raise ValueError(f"split must be one of {S.SPLITS}")
    chosen = None
    if seeds is not None:
        chosen = sorted({int(s) for s in seeds})
        S.check_seeds(chosen, split)
    out = []
    for cell in cells(hypotheses, n_min):
        block = cell.seeds(split)
        use = block if chosen is None else [s for s in chosen if s in block]
        out.extend(Task(cell, int(s), split) for s in use)
    if out:
        S.check_seeds([t.seed for t in out], split)
    return out


def plan(split: str = S.DEVELOPMENT, n_min: int = IIM.N_MIN_DEFAULT) -> dict:
    """Counts per hypothesis: cells, decisive cut-mode cells, tasks and
    scorings (one scoring per declaration and cut mode)."""
    out = {}
    for h in HYPOTHESES:
        cs = cells([h], n_min)
        n_tasks = sum(len(c.seeds(split)) for c in cs)
        out[h] = {
            "cells": len(cs),
            "cut_mode_cells": sum(len(c.cut_modes) for c in cs),
            "tasks": int(n_tasks),
            "scorings": int(sum(len(c.seeds(split)) * len(c.scorings) * len(c.cut_modes)
                                for c in cs)),
            "blocks": sorted({c.block for c in cs}),
        }
    return out


# --------------------------------------------------------------------------
# networks, exact targets, anchors
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def network(kind: str, params: Tuple[Tuple[str, object], ...] = ()):
    """The family-B network of a cell (four units, ``beta = 1``, self-coupling
    0.6 unless the parameters say otherwise)."""
    from impact_pipeline.bench.generators import family_b_network

    kw = {"n": N_UNITS, "beta": BETA, "self_coupling": SELF_COUPLING}
    kw.update({k: v for k, v in params if k not in ("template_family",)})
    return family_b_network(kind, **kw)


@functools.lru_cache(maxsize=None)
def _exact(kind, params, cut_mode, target) -> Optional[float]:
    net = network(kind, params)
    try:
        if target == "conditioned":
            return float(IIM.exact_iim(net, cut_mode, conditioned=True)["delta_psi"])
        if target == "observational":
            return float(IIM.exact_iim(net, cut_mode, observational=True)["delta_psi"])
        return float(IIM.exact_iim(net, cut_mode)["delta_psi"])
    except ValueError:
        return None


def anchor_value(cut_mode: str, anchor: str = PRIMARY_ANCHOR) -> float:
    """The exact anchor (bits) of a cut mode: ``Delta_Psi`` of the anchor
    network with its stationary weights."""
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {tuple(ANCHORS)}")
    kind, params = ANCHORS[anchor]
    v = _exact(kind, params, cut_mode, "tpm")
    if v is None or not v > 0:  # pragma: no cover - deterministic constant
        raise ValueError(f"anchor {anchor} has no positive exact value")
    return float(v)


def exact_targets(cell: Cell, scoring: Optional[Scoring] = None) -> Dict[str, Optional[dict]]:
    """Per cut mode the exact target of a cell under a scoring: ``{value,
    c, target}`` (``c`` on the primary anchor of that cut mode), or None
    where no exact target exists (synthetic and bench-generator cells; the
    hidden driver undeclared is not identified, its exact observational
    value is given for the bidirectional cut only). The hidden driver with
    the driver declared has the background-conditioned target 0 (label
    errors keep the conditioned target of the recorded driver)."""
    out: Dict[str, Optional[dict]] = {}
    for cut in cell.cut_modes:
        if cell.generator in (GEN_AR1_DRIVER, GEN_HYPERSYNCHRONOUS, GEN_HOPF_EEG):
            out[cut] = ({"value": 0.0, "c": 0.0, "target": "independent_given_driver"}
                        if cell.generator == GEN_AR1_DRIVER else None)
            continue
        if cell.generator == GEN_BINARY_MARKOV_DRIVER:
            target = "tpm"
        elif cell.system == "hidden_driver":
            target = ("observational" if scoring is not None
                      and scoring.declaration == DECL_HIDDEN else "conditioned")
        else:
            target = "tpm"
        v = _exact(cell.system, cell.params, cut, target)
        out[cut] = (None if v is None else
                    {"value": v, "c": v / anchor_value(cut), "target": target})
    return out


def sweep_monotonicity() -> dict:
    """CD-1 check of the HCv2-12 (a) sweeps: the exact values per cut mode
    and whether they are strictly monotone in the dose (a level that breaks
    strict monotonicity in either cut family is dropped before the freeze)."""
    out = {}
    for kind, key, levels in (("ring", "coupling", RING_SWEEP),
                              ("xor_loop", "noise", XOR_SWEEP)):
        row = {"levels": list(levels)}
        for cut in CUT_MODES:
            vals = [_exact(kind, ((key, float(v)),), cut, "tpm") for v in levels]
            d = np.diff(np.asarray(vals, dtype=float))
            row[cut] = {"values": vals,
                        "strictly_monotone": bool(np.all(d > 0) or np.all(d < 0))}
        out[kind] = row
    return out


# --------------------------------------------------------------------------
# protocols
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def family_b_protocol(cut_mode: str = IIM.PRIMARY_CUT_MODE, *,
                      anchor: str = PRIMARY_ANCHOR,
                      null_family: str = IIM.NULL_FAMILY_SHIFT,
                      sampling_se: bool = True):
    """The ``impact-mpc-protocol/3`` protocol of family B for one cut mode,
    anchor and null family: necessity set {IIM} (the other principles are not
    scored on family B), the exact anchor as an external reference on the
    excess scale with SE 0, the IIM rank gate at 0.05, the v5 estimator
    version, and the IIM SE methods of the v5 contract plus ``exact`` (exact
    known-TPM values carry no sampling SE and are decided on ``c`` alone).
    The status rule is ``tost-v2`` with ``alpha_A = 0.01``.

    ``sampling_se=False`` is the protocol of value-only scorings (the cells
    whose hypotheses read the excess and ``p_ind`` but no status): it
    declares no SE contract, so such a component is
    ``UNDEFINED(NO_SAMPLING_SE)`` by construction (its name ends in
    ``-values``)."""
    from impact_pipeline import evidence_v2 as EV

    if cut_mode not in IIM.CUT_MODES:
        raise ValueError(f"cut_mode must be one of {IIM.CUT_MODES}")
    if null_family not in IIM.NULL_FAMILIES:
        raise ValueError(f"null_family must be one of {IIM.NULL_FAMILIES}")
    a = anchor_value(cut_mode, anchor)
    kind, params = ANCHORS[anchor]
    suffix = "" if sampling_se else "-values"
    payload = {
        "schema": EV.PROTOCOL_SCHEMA_V3,
        "name": f"mpc-bench-v2-{FAMILY}-{anchor}-{cut_mode}-{null_family}{suffix}",
        "necessity_set": ["IIM"],
        "declared_necessity_set": list(EV.PRINCIPLES),
        "channels": {"IIM": ["default"]},
        "cutoffs": {"IIM": [0.25, 0.10]},
        "alpha": 0.05,
        "null_families": {"IIM": null_family},
        "reference": {
            "kind": "external", "scale": "excess",
            "values": {"IIM": a}, "se": {"IIM": 0.0},
            "source": (f"exact Delta_Psi ({cut_mode}) of the family-B {kind} network "
                       f"{dict(params)}, beta {BETA:g}, self-coupling "
                       f"{SELF_COUPLING:g}, stationary weights"),
        },
        "source_rule": "single_source",
        "estimators": {"IIM": {
            "cut_mode": cut_mode,
            "report_cut_modes": [c for c in CUT_MODES if c != cut_mode],
            "bins": IIM.BINS,
            "tpm_estimator": IIM.TPM_ESTIMATOR,
            "lag_samples": LAG_BINARY,
            "n_null": K_NULL,
            "n_min": IIM.N_MIN_DEFAULT,
            "null_family": null_family,
            "null_order": IIM.NULL_ORDER_SHIFT_THEN_PROJECT,
            "psi_kernel": IIM.PSI_KERNEL,
        }},
        "bearer_nodes": {},
        "status_rule": EV.DEFAULT_STATUS_RULE.to_dict(),
        "directions": {},
        "rank_gates": {"IIM": IIM.RANK_GATE_ALPHA},
        "estimator_versions": {"IIM": IIM.ESTIMATOR_VERSION},
        "se_methods": ({"IIM": sorted(IIM.SE_METHODS + (IIM.SE_METHOD_EXACT,))}
                       if sampling_se else {}),
        "shared_inputs_declaration": None,
        "concordance_route": [],
        "anchors": None,
        "precision": None,
    }
    return EV.ProtocolV3.from_dict(payload)


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Simulation:
    """A simulated cell: the recording (``ts``: macro signals, or nodes with
    ``macro_nodes``), the transition lag, the sampling interval, the recorded
    driver (``driver_states``: one integer label per sample; or
    ``driver_channel``: continuous values), the observation stage and the
    simulation seed. ``basis_inputs`` is what the recording device receives
    for a residualised cell."""

    ts: np.ndarray
    lag: int
    dt: float
    seed: int
    macro_nodes: Optional[Mapping] = None
    driver_states: Optional[np.ndarray] = None
    driver_channel: Optional[np.ndarray] = None
    observation_stage: str = "source"
    meta: Mapping = field(default_factory=dict)


def markov_binary(n_time: int, persistence: float, rng) -> np.ndarray:
    """A binary Markov chain (0/1) that keeps its state with probability
    ``persistence`` (stationary start)."""
    u = rng.random(int(n_time))
    d = np.empty(int(n_time), dtype=np.int64)
    d[0] = int(u[0] < 0.5)
    for t in range(1, int(n_time)):
        d[t] = d[t - 1] if u[t] < persistence else 1 - d[t - 1]
    return d


def switching_levels(n_time: int, dt: float, rng, n_levels: int = SWITCHING_LEVELS,
                     dwell_sec: Tuple[float, float] = SWITCHING_DWELL_SEC) -> np.ndarray:
    """Per-sample level of a switching driver: dwell U(dwell_sec) (at least
    one sample), then a level drawn uniformly."""
    out = np.empty(int(n_time), dtype=np.int64)
    t = 0
    while t < n_time:
        dur = max(1, int(round(float(rng.uniform(*dwell_sec)) / float(dt))))
        out[t:t + dur] = int(rng.integers(0, int(n_levels)))
        t += dur
    return out


def ar1_driver_system(kind: str, seed: int, n_time: int = AR1_N_TIME,
                      dt: float = AR1_DT) -> Tuple[np.ndarray, dict]:
    """HCv2-2 (iii): four independent AR(1) macro signals (coefficients
    U(0.5, 0.9), unit innovations) plus a recorded driver added to every node
    with weight 1: ``continuous_ar1`` (AR(1) with time constant 0.3 s, unit
    variance) or ``switching_filtered`` (six levels, dwell U(1, 3) s, one
    N(0, 1) pattern per level and node, low-pass filtered per node with the
    basis filter at the time constants :data:`SWITCHING_NODE_TAUS`). Returns
    the series and a record with the recorded driver (``channel`` or
    ``states``), the AR coefficients and the drive added to the nodes
    (``drive``; an oracle quantity, never handed to the estimator)."""
    from scipy.signal import lfilter

    from impact_pipeline.v2.declared_inputs import lowpass

    if kind not in DRIVER_KINDS:
        raise ValueError(f"driver kind must be one of {DRIVER_KINDS}")
    rng = np.random.default_rng(int(seed))
    coef = rng.uniform(*AR1_COEF_RANGE, size=N_UNITS)
    e = rng.standard_normal((N_UNITS, int(n_time)))
    x = np.vstack([lfilter([1.0], [1.0, -a], e[i]) for i, a in enumerate(coef)])
    rec = {"coefficients": [float(a) for a in coef]}
    if kind == "continuous_ar1":
        phi = math.exp(-float(dt) / AR1_CONTINUOUS_DRIVER_TAU)
        xi = rng.standard_normal(int(n_time))
        d = np.empty(int(n_time))
        d[0] = xi[0]
        scale = math.sqrt(1.0 - phi * phi)
        for t in range(1, int(n_time)):
            d[t] = phi * d[t - 1] + scale * xi[t]
        drive = np.tile(AR1_DRIVER_WEIGHT * d, (N_UNITS, 1))
        rec["channel"] = d
    else:
        lev = switching_levels(n_time, dt, rng)
        pat = rng.standard_normal((SWITCHING_LEVELS, N_UNITS))
        drive = np.vstack([lowpass(AR1_DRIVER_WEIGHT * pat[lev, i], dt, tau)
                           for i, tau in enumerate(SWITCHING_NODE_TAUS)])
        rec["states"] = lev
    rec["drive"] = drive
    return x + drive, rec


def simulate(task: Task) -> Simulation:
    """The simulation of a task (deterministic in its simulation seed)."""
    from impact_pipeline.bench.generators import make_system, sample_binary_trajectory

    cell = task.cell
    seed = task.simulation_seed
    p = cell.param_dict
    if cell.generator in (GEN_BINARY, GEN_BINARY_MARKOV_DRIVER):
        kind = cell.system
        net_params = tuple((k, v) for k, v in cell.params if k != "driver_persistence"
                           or kind == "hidden_driver")
        net = network(kind, net_params)
        if kind == "hidden_driver":
            traj, drv = sample_binary_trajectory(net, cell.n_time, seed=seed,
                                                 return_driver=True)
            states = (np.asarray(drv) > 0).astype(np.int64)
        else:
            traj = sample_binary_trajectory(net, cell.n_time, seed=seed)
            states = None
        if cell.generator == GEN_BINARY_MARKOV_DRIVER:
            rng = np.random.default_rng(hash_seed(HASH_LABEL, "irrelevant_driver", seed))
            states = markov_binary(cell.n_time, float(p["driver_persistence"]), rng)
        return Simulation(ts=np.asarray(traj, dtype=float).T, lag=cell.lag, dt=1.0,
                          seed=seed, driver_states=states)
    if cell.generator == GEN_AR1_DRIVER:
        y, rec = ar1_driver_system(p["driver"], seed)
        return Simulation(ts=y, lag=cell.lag, dt=AR1_DT, seed=seed,
                          driver_states=rec.get("states"),
                          driver_channel=rec.get("channel"),
                          meta={"coefficients": rec["coefficients"]})
    if cell.generator == GEN_HYPERSYNCHRONOUS:
        sys_ = make_system("hypersynchronous", seed=seed,
                           template_family=p.get("template_family", "A"))
    else:
        sys_ = make_system("whole_brain_eeg", seed=seed,
                           whole_brain_config={"G": float(p.get("G", 0.0))},
                           reference=str(p.get("reference", "average")))
    meta = dict(sys_.meta)
    return Simulation(ts=np.asarray(sys_.ts, dtype=float),
                      lag=int(meta.get("iim_lag_samples", 2)), dt=float(sys_.dt),
                      seed=seed, macro_nodes=dict(meta["iim_macro_nodes"]),
                      observation_stage=("sensor" if cell.generator == GEN_HOPF_EEG
                                         else "source"),
                      meta={"iim_grain": meta.get("iim_grain"),
                            "family": meta.get("family")})


def label_error_states(states, q: float, seed: int) -> Tuple[np.ndarray, int]:
    """The recorded binary driver with label errors: every label whose
    uniform draw (stream ``SeedSequence([seed, 41])``, one draw per sample
    whatever ``q``) is below ``q`` is flipped, so the flips at ``q = 0.1`` are
    a subset of those at ``q = 0.25``. Returns the labels and the number of
    flips."""
    s = np.asarray(states, dtype=np.int64)
    rng = np.random.default_rng(S.stream_seed_sequence(int(seed), LABEL_ERROR_STREAM))
    u = rng.random(s.size)
    hit = u < float(q)
    return np.where(hit, 1 - s, s), int(hit.sum())


def input_basis(sim: Simulation):
    """The common input basis of a residualised cell: the recording device
    records the driver (continuous channel or state track), declaration R
    selects it, and the IIM basis is built with ``tau_c = 0.1 s``."""
    from impact_pipeline.v2 import declared_inputs as DI

    system = SimpleNamespace(ts=sim.ts, events=None,
                             meta={"dt": float(sim.dt), "seed": None}, oracle={})
    extra_states = None if sim.driver_states is None else {"driver_level": sim.driver_states}
    extra_channels = (None if sim.driver_channel is None
                      else {"driver": sim.driver_channel})
    rec = DI.record_inputs(system, extra_states=extra_states,
                           extra_channels=extra_channels)
    dec = DI.declare(rec, "R")
    return DI.input_basis(dec, tau_c=AR1_TAU_C, estimator="IIM")


def scoring_inputs(task: Task, sim: Simulation, scoring: Scoring) -> dict:
    """The conditioning inputs of one scoring: ``strata`` (the recorded
    binary driver, with label errors where declared) or ``basis``."""
    out = {"strata": None, "basis": None, "label_errors": None}
    if scoring.conditioning == "stratify":
        if sim.driver_states is None:
            raise ValueError("a stratified scoring needs a recorded binary driver")
        states = sim.driver_states
        if scoring.declaration == DECL_LABEL_ERROR:
            states, n_flip = label_error_states(states, scoring.label_error_q,
                                                task.trajectory_seed)
            out["label_errors"] = {"q": float(scoring.label_error_q),
                                   "n_flipped": n_flip,
                                   "stream": [int(task.trajectory_seed),
                                              S.STREAM_KEYS[LABEL_ERROR_STREAM]]}
        out["strata"] = states
    elif scoring.conditioning == "residualise":
        out["basis"] = input_basis(sim)
    return out


__all__ = [
    "ANCHORS",
    "ANCHOR_DESIGN_VALUES",
    "Cell",
    "CUT_MODES",
    "DECLARATIONS",
    "DESIGN",
    "DESIGN_VERSION",
    "FAMILY",
    "GENERATORS",
    "HYPOTHESES",
    "K_NULL",
    "OCCUPANCY_CANDIDATES",
    "PRIMARY_ANCHOR",
    "SECOND_ANCHOR",
    "SEED_BLOCKS",
    "Scoring",
    "Simulation",
    "Task",
    "anchor_value",
    "ar1_driver_system",
    "cells",
    "exact_targets",
    "expected_rarest_count",
    "family_b_protocol",
    "hash_seed",
    "input_basis",
    "label_error_states",
    "markov_binary",
    "network",
    "occupancy_designation",
    "plan",
    "scoring_inputs",
    "simulate",
    "sweep_monotonicity",
    "switching_levels",
    "tasks",
    "trajectory_seed",
]
