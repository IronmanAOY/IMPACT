"""
Designs of MPC-Bench v2: the task vocabulary, the registry of design modules
and the run-plan tables.

A *design* turns a split (``development`` or ``confirmatory``) into a list of
:class:`TaskSpec`. A task is **one simulation** (a system, a seed and a twin
replicate, built by a named system builder) with **several scorings**
(:class:`ScoringSpec`): an input declaration, an observation view, an
estimator form, the principles scored and the family protocol that judges
them. The v2 runner (:mod:`impact_pipeline.bench.run_bench_v2`) runs exactly
the task specs a builder returns, and the run-plan tables of the
preregistration are generated from the same builders (:func:`plan_table`),
so plan and builders cannot drift apart (design 3.8, generated run plans).

Registry
--------
:data:`DESIGN_MODULES` pre-declares every design module of the v2 round,
including the modules that other parts of the work add (family B, the
forward-model arms, the null-calibration generator and the Tier-B arms).
:func:`load_module` imports one; a module that is not in the tree yet raises
:class:`DesignNotAvailableError` ("not merged yet") instead of an
``ImportError`` from deep inside the runner.

Module contract. A design module defines

* ``DESIGNS``: a tuple of :class:`Design` (name, family, builder, the task
  counts the design document states per split, the ADEMP statement);
* optionally ``SYSTEM_BUILDERS``: ``{name: builder(task) -> system}`` for
  systems that the runner's built-in builders (``catalogue`` and ``agent``)
  do not cover, ``VIEWS``: ``{name: ViewSpec}`` for observation views
  other than the source view, and ``PROTOCOL_DRAFTS``: ``{protocol key:
  () -> protocol dict}`` for development drafts of protocols that are not
  family-A or family-C1 protocols (generated protocol files take
  precedence).

The runner registers the builders, views and drafts of a module when it
loads the module (in every worker process, through
:attr:`TaskSpec.design_module`).

Estimator forms (:data:`ESTIMATOR_FORMS`) are the reported variants of a
scoring (the NAS secondary representation, the bidirectional IIM cut, the
NAS ``tau_c`` sensitivity, the misdeclared PDI access node): each has its
own protocol ``<family protocol>+<form>``, which differs from the family
protocol only in the form's estimator options.

Seeds
-----
Every task seed is classified by the v2 seed policy
(:mod:`impact_pipeline.v2.seeds`): development 0-999, confirmatory >= 20000,
the v1 block 10000-19999 never used. A builder returns the tasks of one
split only. Held-out conditions (no v2 estimator output before the freeze)
are built for the confirmatory split and, on the development split, only on
the smoke seeds 980-984, whose outputs the runner discards unread.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from impact_pipeline.v2 import PRINCIPLES
from impact_pipeline.v2 import records as _records
from impact_pipeline.v2 import seeds as S

DESIGN_SCHEMA = "mpc-bench-design/1"
SPLITS = S.SPLITS
DEVELOPMENT, CONFIRMATORY = S.DEVELOPMENT, S.CONFIRMATORY

# Every design module of the v2 round (Tier A, then Tier B), pre-declared so
# that a missing module is reported as "not merged yet".
DESIGN_MODULES: Mapping[str, str] = {
    "family_a": "family A (v1 routing with the recording device): witnesses, "
                "sweeps, factorial, adversaries and held-out conditions",
    "family_c1": "family C1 (the v1 family C): witnesses, sweeps and factorial",
    "ram_only": "RAM-only arm at 160 trials",
    "twins": "twin sessions of families A and C1 and of the RAM-only arm",
    "anchors": "reference blocks (development) and anchor replication blocks "
               "(confirmatory)",
    "family_b": "family B cells (binary units with exact TPMs) of the IIM "
                "hypotheses",
    "forward": "Hopf arm and forward-modelled family A (views, regimes, "
               "admission)",
    "null_calibration": "null-calibration generator v2",
    "srpi_only": "SRPI-only arm (SRPI v3; Tier B)",
    "patchwork_v2": "single source via joint dependence, part (a) (Tier B)",
    "tier_b_arms": "Tier-B arms: zero-mean broadcast, slow drift, forward "
                   "RAM-PE and SRPI arms, IIM comparator and long arm",
}
TIER_B_MODULES = ("srpi_only", "patchwork_v2", "tier_b_arms")
PACKAGE = __name__

# The default view: the simulated (source) recording itself.
SOURCE_VIEW = "source"
PRIMARY_FORM = "primary"
BEARER_MODES = ("system", "principle")

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/=@-]*$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


class DesignError(ValueError):
    """A malformed task, scoring or design."""


class DesignNotAvailableError(ImportError):
    """A pre-declared design module that is not in the tree yet."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _token(value, what, pattern=_TOKEN) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise DesignError(f"{what} must be a token, got {value!r}")
    return value


def _json_value(value, what):
    """A JSON-compatible deep copy (sorted mapping keys, lists for tuples)."""
    try:
        text = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DesignError(f"{what} is not JSON-compatible: {exc}") from None
    return json.loads(text)


def _principles(values, what) -> Tuple[str, ...]:
    vals = tuple(values)
    unknown = [p for p in vals if p not in PRINCIPLES]
    if unknown or len(set(vals)) != len(vals):
        raise DesignError(f"{what}: principles must be distinct members of "
                          f"{PRINCIPLES}, got {vals}")
    return tuple(p for p in PRINCIPLES if p in vals)


# --------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ViewSpec:
    """
    An observation view of a simulation: its ``observation_stage`` (record
    schema vocabulary: source, sensor, source_estimate, bold), its
    identifiability ``observation`` (direct, sensor_mixing, source_estimate,
    hemodynamic), the ``transform`` that maps the simulated system to the
    observed one (``transform(system, task) -> system``; None is the
    identity) and the registry ``regime`` keys of the view.
    """

    name: str
    observation_stage: str
    observation: str
    transform: Optional[Callable] = field(default=None, compare=False)
    regime: Mapping = field(default_factory=dict, compare=False)
    description: str = ""

    def __post_init__(self):
        _token(self.name, "view name", _NAME)
        if self.observation_stage not in _records.OBSERVATION_STAGES:
            raise DesignError(f"view {self.name}: observation_stage must be one of "
                              f"{_records.OBSERVATION_STAGES}")
        if self.observation not in _records.OBSERVATIONS:
            raise DesignError(f"view {self.name}: observation must be one of "
                              f"{_records.OBSERVATIONS}")
        object.__setattr__(self, "regime", _json_value(dict(self.regime or {}),
                                                       f"view {self.name} regime"))


SOURCE = ViewSpec(SOURCE_VIEW, "source", "direct",
                  description="the simulated recording itself")


# --------------------------------------------------------------------------
# scorings and tasks
# --------------------------------------------------------------------------
def scoring_id(declaration_id: str, form: str = PRIMARY_FORM, view: str = SOURCE_VIEW,
               bearer_mode: str = "system") -> str:
    """The canonical scoring id: the declaration, ``+form`` for a form other
    than the primary one, ``/view`` for a view other than the source and
    ``@principle`` for the principle-bearer mode (e.g. ``R``,
    ``H+nas_secondary``, ``R/eeg64``, ``R@principle``)."""
    out = str(declaration_id)
    if form != PRIMARY_FORM:
        out += f"+{form}"
    if view != SOURCE_VIEW:
        out += f"/{view}"
    if bearer_mode != "system":
        out += f"@{bearer_mode}"
    return out


@dataclass(frozen=True)
class ScoringSpec:
    """
    One scoring of a task's simulation: the input declaration, the protocol
    that judges it (``protocol_key``, e.g. ``A-R`` or ``A-R+nas_secondary``),
    the observation view, the estimator form, the principles scored, the
    bearer mode (``system``, or ``principle`` for per-principle bearers of a
    patchwork), whether it is a held-out condition and whether it carries the
    verdict (the primary scorings of the joint bench; form scorings, the
    held-out re-scorings and the arms that score one principle by design,
    such as the RAM-only arm, report component statuses only: a verdict over
    one principle is not a verdict about the system).
    """

    scoring_id: str
    declaration_id: str
    protocol_key: str
    view: str = SOURCE_VIEW
    estimator_form: str = PRIMARY_FORM
    principles: Tuple[str, ...] = PRINCIPLES
    bearer_mode: str = "system"
    held_out: bool = False
    verdict: bool = True

    def __post_init__(self):
        _token(self.scoring_id, "scoring_id")
        if self.declaration_id not in _records.KNOWN_DECLARATIONS:
            raise DesignError(f"scoring {self.scoring_id}: declaration must be one of "
                              f"{_records.KNOWN_DECLARATIONS}")
        _token(self.protocol_key, "protocol_key")
        _token(self.view, "view", _NAME)
        _token(self.estimator_form, "estimator_form", _NAME)
        if self.bearer_mode not in BEARER_MODES:
            raise DesignError(f"bearer_mode must be one of {BEARER_MODES}")
        object.__setattr__(self, "principles",
                           _principles(self.principles, f"scoring {self.scoring_id}"))
        object.__setattr__(self, "held_out", bool(self.held_out))
        object.__setattr__(self, "verdict", bool(self.verdict))

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["principles"] = list(self.principles)
        return out

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ScoringSpec":
        data = dict(payload)
        data["principles"] = tuple(data.get("principles", PRINCIPLES))
        return cls(**data)


@dataclass(frozen=True)
class TaskSpec:
    """
    One task of a design: one simulation (``builder`` with ``params``, for
    ``seed`` and twin ``replicate``) and its scorings. ``family`` is the
    family of the record (``A``, ``C1``, ...); ``system`` names the simulated
    system (a catalogue id, a factorial cell or a sweep level); ``tags`` are
    labels for the evaluator (class, sweep knob and level, twin network,
    the witness run a check refers to). ``held_out`` marks a held-out system
    (every scoring is then held out). ``design_module`` names the design
    module, which the runner loads to register its builders and views.
    """

    task_id: str
    design: str
    family: str
    system: str
    seed: int
    builder: str = "catalogue"
    params: Mapping = field(default_factory=dict, hash=False)
    replicate: int = 0
    scorings: Tuple[ScoringSpec, ...] = ()
    held_out: bool = False
    tags: Mapping = field(default_factory=dict, hash=False)
    design_module: Optional[str] = None

    def __post_init__(self):
        _token(self.task_id, "task_id")
        _token(self.design, "design", _NAME)
        _token(self.family, "family", _NAME)
        _token(self.system, f"task {self.task_id}: system")
        _token(self.builder, f"task {self.task_id}: builder", _NAME)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DesignError(f"task {self.task_id}: seed must be an integer")
        try:
            S.split_of(self.seed)
        except S.SeedPolicyError as exc:
            raise DesignError(f"task {self.task_id}: {exc}") from None
        r = self.replicate
        if isinstance(r, bool) or not isinstance(r, int) or not (
                0 <= r <= S.TWIN_REPLICATE_MAX):
            raise DesignError(f"task {self.task_id}: replicate must be 0 or a twin "
                              f"index 1-{S.TWIN_REPLICATE_MAX}")
        scorings = []
        for s in tuple(self.scorings or ()):
            if isinstance(s, Mapping):
                s = ScoringSpec.from_dict(s)
            if not isinstance(s, ScoringSpec):
                raise DesignError(f"task {self.task_id}: scorings must be ScoringSpec")
            scorings.append(s)
        ids = [s.scoring_id for s in scorings]
        if len(set(ids)) != len(ids):
            raise DesignError(f"task {self.task_id}: duplicate scoring ids {ids}")
        if self.held_out:
            scorings = [dataclasses.replace(s, held_out=True) for s in scorings]
        object.__setattr__(self, "scorings", tuple(scorings))
        object.__setattr__(self, "held_out", bool(self.held_out))
        object.__setattr__(self, "params", _json_value(dict(self.params or {}),
                                                       f"task {self.task_id} params"))
        object.__setattr__(self, "tags", _json_value(dict(self.tags or {}),
                                                     f"task {self.task_id} tags"))
        if self.design_module is not None and self.design_module not in DESIGN_MODULES:
            raise DesignError(f"task {self.task_id}: unknown design module "
                              f"{self.design_module!r}")

    @property
    def split(self) -> str:
        return S.split_of(self.seed)

    @property
    def smoke(self) -> bool:
        return S.is_smoke_seed(self.seed)

    @property
    def has_held_out(self) -> bool:
        return self.held_out or any(s.held_out for s in self.scorings)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "design": self.design,
            "family": self.family,
            "system": self.system,
            "seed": self.seed,
            "builder": self.builder,
            "params": json.loads(json.dumps(self.params)),
            "replicate": self.replicate,
            "scorings": [s.to_dict() for s in self.scorings],
            "held_out": self.held_out,
            "tags": json.loads(json.dumps(self.tags)),
            "design_module": self.design_module,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "TaskSpec":
        data = dict(payload)
        data["scorings"] = tuple(ScoringSpec.from_dict(s)
                                 for s in data.get("scorings") or ())
        return cls(**data)


def task_id(design: str, system: str, seed: int, replicate: int = 0,
            suffix: Optional[str] = None) -> str:
    """``<design>-<system>[-<suffix>][-r<r>]-s<seed>`` (seed zero-padded)."""
    parts = [design, system]
    if suffix:
        parts.append(suffix)
    if replicate:
        parts.append(f"r{int(replicate)}")
    parts.append(f"s{int(seed):05d}")
    return "-".join(parts)


# --------------------------------------------------------------------------
# designs and the registry
# --------------------------------------------------------------------------
# The parts of an ADEMP statement (Morris, White and Crowther 2019): aims,
# data-generating mechanisms, estimands, methods and performance measures.
ADEMP_KEYS = ("aims", "data", "estimands", "methods", "performance")


@dataclass(frozen=True)
class Design:
    """
    One design: its ``name`` (unique over every module), the record
    ``family``, a description, the ``build(split, **options)`` function, the
    task counts the design document states for each split
    (``expected_tasks``; the plan test compares them with the builder) and
    its ADEMP statement (``ademp``: one text per :data:`ADEMP_KEYS`), which
    the run-plan tables carry beside the counts.
    """

    name: str
    family: str
    description: str
    build: Callable = field(compare=False)
    expected_tasks: Mapping[str, int] = field(default_factory=dict)
    tier: str = "A"
    module: Optional[str] = None
    ademp: Mapping[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        _token(self.name, "design name", _NAME)
        bad = sorted(set(self.expected_tasks) - set(SPLITS))
        if bad:
            raise DesignError(f"design {self.name}: unknown splits {bad}")
        ademp = dict(self.ademp or {})
        if ademp and (set(ademp) != set(ADEMP_KEYS) or not all(
                isinstance(v, str) and v.strip() for v in ademp.values())):
            raise DesignError(f"design {self.name}: an ADEMP statement has one "
                              f"non-empty text for each of {ADEMP_KEYS}")
        object.__setattr__(self, "ademp", {k: ademp[k] for k in ADEMP_KEYS
                                           if k in ademp})

    def tasks(self, split: str, **options) -> List[TaskSpec]:
        if split not in SPLITS:
            raise DesignError(f"split must be one of {SPLITS}")
        out = list(self.build(split, **options))
        for t in out:
            if t.design != self.name:
                raise DesignError(f"design {self.name} built a task of design "
                                  f"{t.design}")
            if t.split != split:
                raise DesignError(f"design {self.name} ({split}) built task "
                                  f"{t.task_id} on a {t.split} seed")
        return out


_MODULE_CACHE: Dict[str, object] = {}


def load_module(name: str):
    """Import a pre-declared design module; :class:`DesignNotAvailableError`
    when it is not in the tree yet, :class:`DesignError` for an undeclared
    name."""
    if name not in DESIGN_MODULES:
        raise DesignError(f"unknown design module {name!r}; one of "
                          f"{sorted(DESIGN_MODULES)}")
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    full = f"{PACKAGE}.{name}"
    try:
        mod = importlib.import_module(full)
    except ModuleNotFoundError as exc:
        if exc.name == full:
            raise DesignNotAvailableError(
                f"design module {name!r} ({DESIGN_MODULES[name]}) is not merged "
                "yet") from None
        raise
    designs = getattr(mod, "DESIGNS", None)
    if not isinstance(designs, tuple) or not all(isinstance(d, Design)
                                                 for d in designs):
        raise DesignError(f"design module {name!r} must define DESIGNS as a tuple "
                          "of Design")
    _MODULE_CACHE[name] = mod
    return mod


def available_modules() -> Tuple[str, ...]:
    """The pre-declared design modules present in the tree."""
    out = []
    for name in DESIGN_MODULES:
        try:
            load_module(name)
        except DesignNotAvailableError:
            continue
        out.append(name)
    return tuple(out)


def missing_modules() -> Tuple[str, ...]:
    return tuple(n for n in DESIGN_MODULES if n not in available_modules())


def designs(modules: Optional[Iterable[str]] = None) -> Dict[str, Design]:
    """Every design of the given (default: every available) module, by name,
    with ``module`` set; design names are unique over all modules."""
    names = tuple(modules) if modules is not None else available_modules()
    out: Dict[str, Design] = {}
    for m in names:
        for d in load_module(m).DESIGNS:
            if d.name in out:
                raise DesignError(f"design {d.name!r} is defined twice")
            out[d.name] = dataclasses.replace(d, module=m)
    return out


def get_design(name: str) -> Design:
    """A design by name, searched in every available module; the error names
    the modules that are not merged yet."""
    found = designs().get(name)
    if found is None:
        missing = missing_modules()
        hint = (f" (not merged yet: {', '.join(missing)})" if missing else "")
        raise DesignError(f"unknown design {name!r}{hint}")
    return found


def _accepted_options(fn: Callable, options: Mapping) -> dict:
    """The options (not None) that a builder's signature accepts."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return {k: v for k, v in options.items() if v is not None}
    return {k: v for k, v in options.items() if v is not None and k in params}


def build_plan(names: Sequence[str], split: str, **options) -> List[TaskSpec]:
    """The tasks of several designs for one split, in design order, with
    unique task ids and ``design_module`` set. Each builder receives the
    options (``seeds``, ``systems``, ...; None means the design's own) that
    its signature accepts."""
    out, seen = [], set()
    for name in names:
        d = get_design(name)
        for t in d.tasks(split, **_accepted_options(d.build, options)):
            if t.task_id in seen:
                raise DesignError(f"duplicate task id {t.task_id}")
            seen.add(t.task_id)
            if t.design_module != d.module:
                t = dataclasses.replace(t, design_module=d.module)
            out.append(t)
    return out


def plan_table(split: str, names: Optional[Sequence[str]] = None) -> List[dict]:
    """
    One row per design of the split (generated from the builders): tasks,
    scorings, held-out tasks and scorings, smoke tasks, distinct systems and
    seeds, seed range, and the task count the design document states
    (``expected``; None where it states none) with ``matches``, and the
    design's ADEMP statement (``ademp``).
    """
    ds = designs()
    rows = []
    for name in (names if names is not None else list(ds)):
        d = ds[name] if name in ds else get_design(name)
        tasks = d.tasks(split)
        seeds = sorted({t.seed for t in tasks})
        n_sc = sum(len(t.scorings) for t in tasks)
        expected = d.expected_tasks.get(split)
        rows.append({
            "design": d.name,
            "module": d.module,
            "family": d.family,
            "tier": d.tier,
            "split": split,
            "n_tasks": len(tasks),
            "n_scorings": n_sc,
            "n_held_out_tasks": sum(t.has_held_out for t in tasks),
            "n_held_out_scorings": sum(s.held_out for t in tasks for s in t.scorings),
            "n_smoke_tasks": sum(t.smoke for t in tasks),
            "n_systems": len({(t.system, t.params.get("variant")) for t in tasks}),
            "n_seeds": len(seeds),
            "seed_min": seeds[0] if seeds else None,
            "seed_max": seeds[-1] if seeds else None,
            "expected": expected,
            "matches": None if expected is None else len(tasks) == expected,
            "ademp": dict(d.ademp),
        })
    return rows


def seeds_of(spec: Mapping, split: str) -> Tuple[int, ...]:
    """The seeds of a ``{split: range or sequence}`` table (empty when the
    split has none)."""
    vals = spec.get(split, ())
    return tuple(int(s) for s in vals)


# --------------------------------------------------------------------------
# estimator forms and scoring helpers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EstimatorForm:
    """
    An estimator form: the principles it scores and the estimator-option
    overrides that turn the family protocol into the form's protocol
    (``<family protocol>+<form>``, hash-covered like every protocol field).
    A held-out form (no v2 output before the freeze) is built only where
    held-out conditions are.
    """

    name: str
    principles: Tuple[str, ...]
    options: Mapping = field(default_factory=dict)
    held_out: bool = False
    description: str = ""

    def __post_init__(self):
        _token(self.name, "form name", _NAME)
        object.__setattr__(self, "principles",
                           _principles(self.principles, f"form {self.name}"))
        opts = _json_value(dict(self.options or {}), f"form {self.name} options")
        bad = sorted(set(opts) - set(self.principles))
        if bad:
            raise DesignError(f"form {self.name}: options for unscored "
                              f"principles {bad}")
        object.__setattr__(self, "options", opts)


# The estimator forms of the bench. The primary form is the protocol itself;
# every other form is reported beside it (design 2.1 items 5 and 8, 2.0.3,
# 2.2 item 4, 2.4 item 2).
ESTIMATOR_FORMS: Mapping[str, EstimatorForm] = {
    f.name: f for f in (
        EstimatorForm(PRIMARY_FORM, PRINCIPLES, {},
                      description="the family protocol's estimators"),
        EstimatorForm("nas_secondary", ("NAS",),
                      {"NAS": {"block_representation": "secondary"}},
                      description="NAS on all nodes (|B| <= 8) or 8 leading PCs "
                                  "per block; preregistered secondary"),
        EstimatorForm("iim_bidirectional", ("IIM",),
                      {"IIM": {"cut_mode": "bidirectional", "report_cut_modes": []}},
                      description="IIM with bidirectional cuts (reported "
                                  "secondary with its own anchor)"),
        EstimatorForm("nas_tau_0.05", ("NAS",),
                      {"NAS": {"coupling_timescale_sec": 0.05}},
                      description="NAS at tau_c = 0.05 s (reported sensitivity)"),
        EstimatorForm("nas_tau_0.2", ("NAS",),
                      {"NAS": {"coupling_timescale_sec": 0.2}},
                      description="NAS at tau_c = 0.2 s (reported sensitivity)"),
        EstimatorForm("pdi_misdeclared_access", ("PDI",),
                      {"PDI": {"access_module": "S"}}, held_out=True,
                      description="PDI content bearer with module S declared as "
                                  "the access (workspace) nodes: the cost of a "
                                  "wrong access declaration (held out)"),
    )
}


def protocol_key(base: str, form: str = PRIMARY_FORM) -> str:
    """The protocol of a form: the family protocol for the primary form,
    ``<family protocol>+<form>`` otherwise."""
    _token(base, "protocol key")
    if form != PRIMARY_FORM and form not in ESTIMATOR_FORMS:
        raise DesignError(f"unknown estimator form {form!r}")
    return base if form == PRIMARY_FORM else f"{base}+{form}"


def split_protocol_key(key: str) -> Tuple[str, str]:
    """``(family protocol, form)`` of a protocol key."""
    base, sep, form = str(key).partition("+")
    if not sep:
        return base, PRIMARY_FORM
    if form not in ESTIMATOR_FORMS:
        raise DesignError(f"protocol key {key!r} names an unknown form {form!r}")
    return base, form


def make_scorings(protocols: Mapping[str, str], principles: Sequence[str] = PRINCIPLES,
                  forms: Sequence[str] = (), *,
                  bearer_modes: Sequence[str] = ("system",),
                  view: str = SOURCE_VIEW, held_out: bool = False,
                  verdict: bool = True, primary: bool = True,
                  form_declarations: Optional[Mapping[str, Sequence[str]]] = None
                  ) -> Tuple[ScoringSpec, ...]:
    """
    The scorings of one simulation: for each bearer mode, the primary
    scoring of every declaration (``protocols``: ``{declaration: family
    protocol key}``) over ``principles`` (omitted with ``primary=False``),
    then each form's scoring of every declaration (or of the declarations
    ``form_declarations`` names for it) over the form's principles that are
    scored here. Forms report component statuses only (no verdict).
    """
    form_declarations = dict(form_declarations or {})
    principles = _principles(principles, "make_scorings")
    out = []
    for mode in bearer_modes:
        if primary:
            for decl, base in protocols.items():
                out.append(ScoringSpec(scoring_id(decl, PRIMARY_FORM, view, mode),
                                       decl, base, view, PRIMARY_FORM, principles,
                                       mode, held_out, verdict))
        for form in forms:
            spec = ESTIMATOR_FORMS.get(form)
            if spec is None or form == PRIMARY_FORM:
                raise DesignError(f"unknown estimator form {form!r}")
            fp = tuple(p for p in spec.principles if p in principles)
            if not fp:
                continue
            decls = form_declarations.get(form)
            for decl, base in protocols.items():
                if decls is not None and decl not in decls:
                    continue
                out.append(ScoringSpec(
                    scoring_id(decl, form, view, mode), decl, protocol_key(base, form),
                    view, form, fp, mode, held_out or spec.held_out, False))
    return tuple(out)


__all__ = [
    "ADEMP_KEYS",
    "BEARER_MODES",
    "CONFIRMATORY",
    "DESIGN_MODULES",
    "DESIGN_SCHEMA",
    "DEVELOPMENT",
    "Design",
    "DesignError",
    "DesignNotAvailableError",
    "ESTIMATOR_FORMS",
    "EstimatorForm",
    "PRIMARY_FORM",
    "SOURCE",
    "SOURCE_VIEW",
    "SPLITS",
    "ScoringSpec",
    "TIER_B_MODULES",
    "TaskSpec",
    "ViewSpec",
    "available_modules",
    "build_plan",
    "designs",
    "get_design",
    "load_module",
    "make_scorings",
    "missing_modules",
    "plan_table",
    "protocol_key",
    "scoring_id",
    "seeds_of",
    "split_protocol_key",
    "task_id",
]
