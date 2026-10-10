"""
Result schema ``mpc-bench-result/3`` of MPC-Bench v2.

A task record holds **one simulation and several scorings**: the same
simulated system is scored under several input declarations (R, H, ...),
observation views and estimator forms. Every component of a scoring carries
the estimate, the null moments and family, ``se``, ``se_df``, ``se_method``,
the construct value ``c`` (NAS also ``c_R`` and ``c_B``), status, reason,
flags, declaration id, identifiability record, observation stage, estimator
version, family-protocol id and hash, replicate index, seconds and load
average, so that a flat component table needs no join. A PRESENT or ABSENT
component carries a finite ``c``; an UNDEFINED one carries a reason of the
central vocabulary (:mod:`~impact_pipeline.v2.reasons`). The replicate index
is 0 or a twin index of the seed policy.

Task status: ``ok``; ``ok_with_component_errors`` when some component raised
(that component is ``UNDEFINED(ESTIMATOR_ERROR:<type>)`` and no other
component is touched); ``error`` when the task itself failed (``error`` holds
the message).

The dataclasses validate on construction; ``to_dict`` / ``from_dict`` and
``dumps`` / ``loads`` round-trip exactly (non-finite floats are stored as
``null``; unknown keys are refused, extras go into the ``details`` mappings).
The v1 schema ``mpc-bench-result/2`` and its runner are unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import math
import numbers
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Tuple

from impact_pipeline.v2 import PRINCIPLES, RESULT_SCHEMA
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import seeds as S

TASK_OK = "ok"
TASK_OK_WITH_COMPONENT_ERRORS = "ok_with_component_errors"
TASK_ERROR = "error"
TASK_STATUSES = (TASK_OK, TASK_OK_WITH_COMPONENT_ERRORS, TASK_ERROR)

VERDICTS = ("MPC_CONSISTENT", "EXCLUDED", "UNDETERMINED")
# Declarations of the bench (preregistration v2, section 3.2); family B
# declares its drivers as recorded, hidden or label-error drivers.
KNOWN_DECLARATIONS = (
    "R", "H", "P", "Q10", "Q25", "J", "none", "recorded", "hidden", "label_error",
)
# Observation stages (registry v3 regime key ``observation_stage``).
OBSERVATION_STAGES = ("source", "sensor", "source_estimate", "bold")
# Identifiability record (registry v3 vocabulary).
SHARED_INPUTS = ("complete", "partial", "none")
OBSERVATIONS = ("direct", "source_estimate", "sensor_mixing", "hemodynamic")
HUB_PRIVILEGED = (True, False, "not_tested")
IDENTIFIABILITY_KEYS = ("shared_inputs", "observation", "hub_privileged")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/=@-]*$")


class RecordSchemaError(ValueError):
    """A record that does not follow ``mpc-bench-result/3``."""


# --------------------------------------------------------------------------
# field helpers
# --------------------------------------------------------------------------
def _fail(where, msg):
    raise RecordSchemaError(f"{where}: {msg}")


def _float(value, where) -> Optional[float]:
    """None for None/NaN/inf, else a Python float."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        _fail(where, f"expected a number or null, got {value!r}")
    v = float(value)
    return v if math.isfinite(v) else None


def _int(value, where, minimum=None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        _fail(where, f"expected an integer, got {value!r}")
    v = int(value)
    if minimum is not None and v < minimum:
        _fail(where, f"must be >= {minimum}, got {v}")
    return v


def _opt_int(value, where, minimum=None) -> Optional[int]:
    return None if value is None else _int(value, where, minimum)


def _str(value, where, pattern=_TOKEN) -> str:
    if not isinstance(value, str) or not value:
        _fail(where, f"expected a non-empty string, got {value!r}")
    if pattern is not None and not pattern.match(value):
        _fail(where, f"malformed value {value!r}")
    return value


def _opt_str(value, where, pattern=None) -> Optional[str]:
    return None if value is None else _str(value, where, pattern)


def _hash(value, where) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        _fail(where, f"expected a 64-digit lowercase hex SHA-256, got {value!r}")
    return value


def _opt_hash(value, where) -> Optional[str]:
    return None if value is None else _hash(value, where)


def _jsonable(value, where):
    """A JSON-compatible deep copy (dicts with string keys, lists, str, int,
    finite float -> float, non-finite float -> None, bool, None)."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                _fail(where, f"mapping keys must be strings, got {k!r}")
            out[k] = _jsonable(v, f"{where}.{k}")
        return out
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, where) for v in value]
    if hasattr(value, "tolist"):  # numpy arrays and scalars
        return _jsonable(value.tolist(), where)
    _fail(where, f"not JSON-compatible: {type(value).__name__}")


def _replicate(value, where) -> int:
    """A replicate index: 0 (the run itself) or a twin index
    ``1 <= r <= seeds.TWIN_REPLICATE_MAX``."""
    r = _int(value, where, 0)
    if r > S.TWIN_REPLICATE_MAX:
        _fail(where, f"must be <= {S.TWIN_REPLICATE_MAX} (the twin replicate "
              f"range), got {r}")
    return r


def _load_average(value, where) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__iter__"):
        _fail(where, "load average is a triple (1, 5, 15 min)")
    vals = tuple(value)
    if len(vals) != 3:
        _fail(where, "load average is a triple (1, 5, 15 min)")
    out = tuple(_float(v, where) for v in vals)
    if any(v is None or v < 0 for v in out):
        _fail(where, "load averages are finite and >= 0")
    return out


def _check_keys(payload, allowed, required, where):
    if not isinstance(payload, Mapping):
        _fail(where, f"expected a mapping, got {type(payload).__name__}")
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        _fail(where, f"unknown keys {unknown}")
    missing = sorted(k for k in required if k not in payload)
    if missing:
        _fail(where, f"missing keys {missing}")


def validate_identifiability(value, where="identifiability") -> Optional[dict]:
    """``{shared_inputs, observation, hub_privileged}`` with the registry v3
    vocabulary, or None."""
    if value is None:
        return None
    _check_keys(value, IDENTIFIABILITY_KEYS, IDENTIFIABILITY_KEYS, where)
    if value["shared_inputs"] not in SHARED_INPUTS:
        _fail(where, f"shared_inputs must be one of {SHARED_INPUTS}")
    if value["observation"] not in OBSERVATIONS:
        _fail(where, f"observation must be one of {OBSERVATIONS}")
    hp = value["hub_privileged"]
    if not any(hp is v for v in (True, False)) and hp != "not_tested":
        _fail(where, f"hub_privileged must be one of {HUB_PRIVILEGED}")
    return {k: value[k] for k in IDENTIFIABILITY_KEYS}


# --------------------------------------------------------------------------
# component
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentRecord:
    """One principle scored once (one declaration, view and estimator form)."""

    principle: str
    status: str
    estimator_version: str
    declaration_id: str
    observation_stage: str
    protocol_id: str
    protocol_hash: str
    replicate: int = 0
    reason: Optional[str] = None
    flags: Tuple[str, ...] = ()
    estimate: Optional[float] = None
    null_mean: Optional[float] = None
    null_sd: Optional[float] = None
    n_null: int = 0
    null_family: Optional[str] = None
    se: Optional[float] = None
    se_df: Optional[float] = None
    se_method: Optional[str] = None
    c: Optional[float] = None
    c_R: Optional[float] = None
    c_B: Optional[float] = None
    se_c: Optional[float] = None
    df_c: Optional[float] = None
    identifiability: Optional[dict] = field(default=None, hash=False)
    seconds: Optional[float] = None
    load_average: Optional[Tuple[float, float, float]] = None
    details: dict = field(default_factory=dict, hash=False)

    def __post_init__(self):
        w = f"component {self.principle!r}"
        if self.principle not in PRINCIPLES:
            _fail(w, f"principle must be one of {PRINCIPLES}")
        if isinstance(self.flags, (str, bytes, Mapping)) or not hasattr(
                self.flags, "__iter__"):
            _fail(w, f"flags must be a sequence of flag codes, got {self.flags!r}")
        flags = tuple(self.flags)
        if len(set(flags)) != len(flags):
            _fail(w, "duplicate flags")
        try:
            R.check_status(self.status, self.reason, flags)
        except ValueError as exc:
            _fail(w, str(exc))
        _str(self.estimator_version, f"{w}.estimator_version")
        _str(self.declaration_id, f"{w}.declaration_id")
        if self.observation_stage not in OBSERVATION_STAGES:
            _fail(w, f"observation_stage must be one of {OBSERVATION_STAGES}")
        _str(self.protocol_id, f"{w}.protocol_id")
        _hash(self.protocol_hash, f"{w}.protocol_hash")
        set_ = object.__setattr__
        set_(self, "flags", flags)
        set_(self, "replicate", _replicate(self.replicate, f"{w}.replicate"))
        set_(self, "n_null", _int(self.n_null, f"{w}.n_null", 0))
        for name in ("estimate", "null_mean", "null_sd", "se", "se_df", "c",
                     "c_R", "c_B", "se_c", "df_c", "seconds"):
            set_(self, name, _float(getattr(self, name), f"{w}.{name}"))
        if self.status != R.UNDEFINED and self.c is None:
            # a decided status is a statement about c; without a finite c it
            # cannot have been reached by the status rule
            _fail(w, f"a {self.status} component carries a finite construct "
                  "value c")
        for name in ("se", "null_sd", "se_c", "seconds"):
            v = getattr(self, name)
            if v is not None and v < 0:
                _fail(w, f"{name} must be >= 0")
        for name in ("se_df", "df_c"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                _fail(w, f"{name} must be > 0")
        if self.principle != "NAS" and (self.c_R is not None or self.c_B is not None):
            _fail(w, "c_R and c_B are NAS fields")
        _opt_str(self.null_family, f"{w}.null_family", _TOKEN)
        _opt_str(self.se_method, f"{w}.se_method", _TOKEN)
        set_(self, "identifiability",
             validate_identifiability(self.identifiability, f"{w}.identifiability"))
        set_(self, "load_average",
             _load_average(self.load_average, f"{w}.load_average"))
        set_(self, "details", _jsonable(dict(self.details or {}), f"{w}.details"))

    @property
    def is_estimator_error(self) -> bool:
        return R.is_estimator_error(self.reason)

    def to_dict(self) -> dict:
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name in ("flags", "load_average") and v is not None:
                v = list(v)
            out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ComponentRecord":
        names = [f.name for f in dataclasses.fields(cls)]
        required = [f.name for f in dataclasses.fields(cls)
                    if f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING]
        _check_keys(payload, names, required, "component")
        return cls(**dict(payload))


def component_error(
    principle: str,
    exc,
    *,
    estimator_version: str,
    declaration_id: str,
    observation_stage: str,
    protocol_id: str,
    protocol_hash: str,
    replicate: int = 0,
    identifiability=None,
    seconds=None,
    load_average=None,
) -> ComponentRecord:
    """The record of a component whose estimator raised:
    ``UNDEFINED(ESTIMATOR_ERROR:<type>)`` with the message in ``details``."""
    return ComponentRecord(
        principle=principle,
        status=R.UNDEFINED,
        reason=R.estimator_error(exc),
        estimator_version=estimator_version,
        declaration_id=declaration_id,
        observation_stage=observation_stage,
        protocol_id=protocol_id,
        protocol_hash=protocol_hash,
        replicate=replicate,
        identifiability=identifiability,
        seconds=seconds,
        load_average=load_average,
        details={"error": f"{type(exc).__name__}: {exc}"
                 if isinstance(exc, BaseException) else str(exc)},
    )


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoringRecord:
    """One scoring of the task's simulation: a declaration, an observation
    view and an estimator form, under one family protocol."""

    scoring_id: str
    declaration_id: str
    observation_stage: str
    view: str
    estimator_form: str
    protocol_id: str
    protocol_hash: str
    components: Mapping[str, ComponentRecord] = field(default_factory=dict, hash=False)
    verdict: Optional[dict] = field(default=None, hash=False)
    details: dict = field(default_factory=dict, hash=False)

    def __post_init__(self):
        w = f"scoring {self.scoring_id!r}"
        _str(self.scoring_id, "scoring.scoring_id")
        _str(self.declaration_id, f"{w}.declaration_id")
        if self.observation_stage not in OBSERVATION_STAGES:
            _fail(w, f"observation_stage must be one of {OBSERVATION_STAGES}")
        _str(self.view, f"{w}.view")
        _str(self.estimator_form, f"{w}.estimator_form")
        _str(self.protocol_id, f"{w}.protocol_id")
        _hash(self.protocol_hash, f"{w}.protocol_hash")
        comps = {}
        for key, comp in dict(self.components or {}).items():
            if isinstance(comp, Mapping):
                comp = ComponentRecord.from_dict(comp)
            if not isinstance(comp, ComponentRecord):
                _fail(w, f"component {key!r} is not a ComponentRecord")
            if comp.principle != key:
                _fail(w, f"component key {key!r} differs from its principle "
                      f"{comp.principle!r}")
            for name in ("declaration_id", "observation_stage", "protocol_id",
                         "protocol_hash"):
                if getattr(comp, name) != getattr(self, name):
                    _fail(w, f"component {key}: {name} differs from the scoring's")
            comps[key] = comp
        object.__setattr__(self, "components", {p: comps[p] for p in PRINCIPLES
                                                if p in comps})
        if self.verdict is not None:
            v = _jsonable(dict(self.verdict), f"{w}.verdict")
            if v.get("verdict") not in VERDICTS:
                _fail(w, f"verdict.verdict must be one of {VERDICTS}")
            object.__setattr__(self, "verdict", v)
        object.__setattr__(self, "details",
                           _jsonable(dict(self.details or {}), f"{w}.details"))

    def to_dict(self) -> dict:
        return {
            "scoring_id": self.scoring_id,
            "declaration_id": self.declaration_id,
            "observation_stage": self.observation_stage,
            "view": self.view,
            "estimator_form": self.estimator_form,
            "protocol_id": self.protocol_id,
            "protocol_hash": self.protocol_hash,
            "components": {p: c.to_dict() for p, c in self.components.items()},
            "verdict": self.verdict,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ScoringRecord":
        names = [f.name for f in dataclasses.fields(cls)]
        required = ["scoring_id", "declaration_id", "observation_stage", "view",
                    "estimator_form", "protocol_id", "protocol_hash"]
        _check_keys(payload, names, required, "scoring")
        return cls(**dict(payload))


# --------------------------------------------------------------------------
# simulation and task
# --------------------------------------------------------------------------
SIMULATION_KEYS = (
    "ts_sha256", "raw_ts_sha256", "structural_hash", "schedule_hash",
    "n_nodes", "n_time", "dt", "seconds",
)


def validate_simulation(value, where="simulation") -> Optional[dict]:
    """The simulation block: the duplicate-detector hashes ``ts_sha256`` and
    ``raw_ts_sha256`` (None when there is no separate raw series), the twin
    hashes ``structural_hash`` and ``schedule_hash``, the shape, ``dt`` and
    the simulation time."""
    if value is None:
        return None
    _check_keys(value, SIMULATION_KEYS, ("ts_sha256", "n_nodes", "n_time", "dt"), where)
    out = {k: value.get(k) for k in SIMULATION_KEYS}
    out["ts_sha256"] = _hash(out["ts_sha256"], f"{where}.ts_sha256")
    for k in ("raw_ts_sha256", "structural_hash", "schedule_hash"):
        out[k] = _opt_hash(out[k], f"{where}.{k}")
    out["n_nodes"] = _int(out["n_nodes"], f"{where}.n_nodes", 1)
    out["n_time"] = _int(out["n_time"], f"{where}.n_time", 1)
    out["dt"] = _float(out["dt"], f"{where}.dt")
    if out["dt"] is None or out["dt"] <= 0:
        _fail(where, "dt must be > 0")
    out["seconds"] = _float(out["seconds"], f"{where}.seconds")
    return out


def derive_task_status(scorings: Iterable[ScoringRecord], error=None) -> str:
    """``error`` when the task failed, ``ok_with_component_errors`` when a
    component raised, else ``ok``."""
    if error:
        return TASK_ERROR
    for s in scorings:
        if any(c.is_estimator_error for c in s.components.values()):
            return TASK_OK_WITH_COMPONENT_ERRORS
    return TASK_OK


@dataclass(frozen=True)
class TaskRecord:
    """One task: one simulation and its scorings (``mpc-bench-result/3``)."""

    task_id: str
    design: str
    family: str
    system: str
    seed: int
    generator_version: str
    status: str
    replicate: int = 0
    split: Optional[str] = None
    config: dict = field(default_factory=dict, hash=False)
    simulation: Optional[dict] = field(default=None, hash=False)
    scorings: Tuple[ScoringRecord, ...] = ()
    error: Optional[str] = None
    timing: dict = field(default_factory=dict, hash=False)
    provenance: dict = field(default_factory=dict, hash=False)
    schema: str = RESULT_SCHEMA

    def __post_init__(self):
        w = f"task {self.task_id!r}"
        if self.schema != RESULT_SCHEMA:
            _fail(w, f"schema must be {RESULT_SCHEMA!r}, got {self.schema!r}")
        _str(self.task_id, "task.task_id")
        for name in ("design", "family", "system", "generator_version"):
            _str(getattr(self, name), f"{w}.{name}")
        set_ = object.__setattr__
        set_(self, "seed", _int(self.seed, f"{w}.seed"))
        set_(self, "replicate", _replicate(self.replicate, f"{w}.replicate"))
        try:
            split = S.split_of(self.seed)
        except S.SeedPolicyError as exc:
            _fail(w, str(exc))
        if self.split is not None and self.split != split:
            _fail(w, f"split {self.split!r} differs from the seed policy's {split!r}")
        set_(self, "split", split)
        scorings = []
        for s in tuple(self.scorings or ()):
            if isinstance(s, Mapping):
                s = ScoringRecord.from_dict(s)
            if not isinstance(s, ScoringRecord):
                _fail(w, "scorings must be ScoringRecord objects")
            for c in s.components.values():
                if c.replicate != self.replicate:
                    _fail(w, f"scoring {s.scoring_id}: component {c.principle} "
                          "replicate differs from the task's")
            scorings.append(s)
        ids = [s.scoring_id for s in scorings]
        if len(set(ids)) != len(ids):
            _fail(w, "duplicate scoring ids")
        set_(self, "scorings", tuple(scorings))
        if self.error is not None:
            _str(self.error, f"{w}.error", pattern=None)
        if self.status not in TASK_STATUSES:
            _fail(w, f"status must be one of {TASK_STATUSES}")
        want = derive_task_status(scorings, self.error)
        if self.status != want:
            _fail(w, f"status {self.status!r} is inconsistent with the record "
                  f"(expected {want!r})")
        set_(self, "config", _jsonable(dict(self.config or {}), f"{w}.config"))
        set_(self, "simulation",
             validate_simulation(self.simulation, f"{w}.simulation"))
        if self.status != TASK_ERROR and self.simulation is None:
            _fail(w, "a task that did not fail records its simulation")
        timing = _jsonable(dict(self.timing or {}), f"{w}.timing")
        if "load_average" in timing and timing["load_average"] is not None:
            _load_average(timing["load_average"], f"{w}.timing.load_average")
        set_(self, "timing", timing)
        set_(self, "provenance",
             _jsonable(dict(self.provenance or {}), f"{w}.provenance"))

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "design": self.design,
            "family": self.family,
            "system": self.system,
            "seed": self.seed,
            "replicate": self.replicate,
            "split": self.split,
            "generator_version": self.generator_version,
            "config": self.config,
            "simulation": self.simulation,
            "scorings": [s.to_dict() for s in self.scorings],
            "status": self.status,
            "error": self.error,
            "timing": self.timing,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "TaskRecord":
        names = [f.name for f in dataclasses.fields(cls)]
        required = ["schema", "task_id", "design", "family", "system", "seed",
                    "generator_version", "status"]
        _check_keys(payload, names, required, "task")
        return cls(**dict(payload))

    def component_rows(self) -> Iterator[dict]:
        """One flat row per (scoring, component): task keys, scoring keys and
        every component field (identifiability flattened to
        ``identifiability_<key>``, flags ``;``-joined, details omitted)."""
        for s in self.scorings:
            for c in s.components.values():
                row = {
                    "task_id": self.task_id,
                    "design": self.design,
                    "family": self.family,
                    "system": self.system,
                    "seed": self.seed,
                    "split": self.split,
                    "task_status": self.status,
                    "scoring_id": s.scoring_id,
                    "view": s.view,
                    "estimator_form": s.estimator_form,
                }
                d = c.to_dict()
                d.pop("details")
                ident = d.pop("identifiability") or {}
                d["flags"] = R.VERDICT_SEPARATOR.join(d["flags"])
                la = d.pop("load_average")
                for i, name in enumerate(("load_1m", "load_5m", "load_15m")):
                    d[name] = None if la is None else la[i]
                for k in IDENTIFIABILITY_KEYS:
                    d[f"identifiability_{k}"] = ident.get(k)
                row.update(d)
                yield row


# --------------------------------------------------------------------------
# record-level integrity checks (used by the integrity audit)
# --------------------------------------------------------------------------
def duplicate_simulations(records: Iterable[TaskRecord]) -> List[dict]:
    """
    Groups of distinct tasks whose simulations are identical: the same
    ``ts_sha256``, or the same ``raw_ts_sha256`` (the duplicate detector).
    Each group lists its hash kind and value, the task ids and families, and
    ``cross_family`` (a system that appears in more than one family). Tasks
    without a simulation block are skipped.
    """
    groups: dict = {}
    for rec in records:
        sim = rec.simulation
        if sim is None:
            continue
        for kind in ("ts_sha256", "raw_ts_sha256"):
            h = sim.get(kind)
            if h is not None:
                groups.setdefault((kind, h), {})[rec.task_id] = rec.family
    out = []
    for (kind, h), tasks in sorted(groups.items()):
        if len(tasks) > 1:
            fams = sorted(set(tasks.values()))
            out.append({"hash_kind": kind, "hash": h, "task_ids": sorted(tasks),
                        "families": fams, "cross_family": len(fams) > 1})
    return out


def protocol_hash_mismatches(records: Iterable[TaskRecord],
                             frozen: Mapping[str, str]) -> List[dict]:
    """
    Components whose family protocol differs from the frozen one: a
    ``protocol_id`` not in ``frozen`` (``{protocol_id: hash}``) or a hash that
    differs from the frozen hash. A scoring without components is checked
    through its own protocol id and hash (``principle`` None). The evaluator
    refuses such records.
    """
    out = []
    for rec in records:
        for s in rec.scorings:
            if not s.components:
                want = frozen.get(s.protocol_id)
                if want != s.protocol_hash:
                    out.append({"task_id": rec.task_id, "scoring_id": s.scoring_id,
                                "principle": None, "protocol_id": s.protocol_id,
                                "protocol_hash": s.protocol_hash,
                                "frozen_hash": want})
                continue
            for c in s.components.values():
                want = frozen.get(c.protocol_id)
                if want != c.protocol_hash:
                    out.append({"task_id": rec.task_id, "scoring_id": s.scoring_id,
                                "principle": c.principle,
                                "protocol_id": c.protocol_id,
                                "protocol_hash": c.protocol_hash,
                                "frozen_hash": want})
    return out


def plan_differences(records: Iterable[TaskRecord],
                     planned_task_ids: Iterable[str]) -> dict:
    """Task ids of a run against its generated plan: ``missing`` (planned,
    no record), ``unexpected`` (a record outside the plan), ``duplicated``
    (more than one record) and ``duplicated_in_plan`` (a task id planned
    twice); ``ok`` iff all four are empty."""
    planned = Counter(planned_task_ids)
    seen = Counter(rec.task_id for rec in records)
    out = {
        "missing": sorted(set(planned) - set(seen)),
        "unexpected": sorted(set(seen) - set(planned)),
        "duplicated": sorted(t for t, n in seen.items() if n > 1),
        "duplicated_in_plan": sorted(t for t, n in planned.items() if n > 1),
    }
    out["ok"] = not any(out.values())
    return out


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------
def dumps(record: TaskRecord) -> str:
    """One JSON line (sorted keys, no NaN)."""
    return json.dumps(record.to_dict(), sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def loads(text: str) -> TaskRecord:
    return TaskRecord.from_dict(json.loads(text))


def write_jsonl(records: Iterable[TaskRecord], path) -> None:
    """Append records to a JSON-lines file, one :func:`dumps` line each
    (a runner resumes by appending)."""
    p = Path(path)
    with open(p, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(dumps(rec) + "\n")


def read_jsonl(path) -> List[TaskRecord]:
    """Every record of a JSON-lines file (blank lines skipped); a record that
    breaks the schema raises :class:`RecordSchemaError` with its line."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                out.append(loads(line))
            except (ValueError, TypeError) as exc:  # schema, JSON or type errors
                raise RecordSchemaError(f"{path}:{n}: {exc}") from exc
    return out


__all__ = [
    "ComponentRecord",
    "IDENTIFIABILITY_KEYS",
    "KNOWN_DECLARATIONS",
    "OBSERVATIONS",
    "OBSERVATION_STAGES",
    "RESULT_SCHEMA",
    "RecordSchemaError",
    "SHARED_INPUTS",
    "SIMULATION_KEYS",
    "ScoringRecord",
    "TASK_ERROR",
    "TASK_OK",
    "TASK_OK_WITH_COMPONENT_ERRORS",
    "TASK_STATUSES",
    "TaskRecord",
    "VERDICTS",
    "component_error",
    "derive_task_status",
    "dumps",
    "duplicate_simulations",
    "loads",
    "plan_differences",
    "protocol_hash_mismatches",
    "read_jsonl",
    "validate_identifiability",
    "validate_simulation",
    "write_jsonl",
]
