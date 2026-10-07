"""
Recording device, input declarations and the common input basis of
MPC-Bench v2 (design sections 2.0.1-2.0.3).

NAS v3 and IIM v5 condition on the same declared exogenous inputs, built here
in three steps.

Recording device (:func:`record_inputs`)
----------------------------------------
At simulation time the device builds a :class:`RecordedInputs` table from
what an experimenter could log. Estimators receive this table (through a
declaration), never the oracle. It holds

(i) the task events of ``system.events``, copied unchanged; v1 estimators
    keep reading ``system.events`` itself and see exactly the v1 table;
(ii) ``context_cue`` events: one cue per context switch, with the label of
    the new context as ``value`` and the time until the next cue as
    ``duration``, plus one cue at the start of the recording for the context
    in force there. A switch that redraws the context already in force
    changes nothing in the system and is not logged, so consecutive cues
    always carry different labels;
(iii) the slow-rhythm phase as the continuous channels ``slow_phase_sin`` and
    ``slow_phase_cos`` on the sample grid;
(iv) further exogenous drivers a generator exposes: the hidden driver of the
    v1 ``adversarial_common_driver`` system (channel ``common_driver``), and
    any state track or continuous channel a generator puts in its oracle
    under ``recordable_states`` (``{name: integer label per sample}``) or
    ``recordable_channels`` (``{name: values per sample}``, one row or a
    ``(k, n_time)`` block), or that the caller passes as ``extra_states`` /
    ``extra_channels``. These count as recorded drivers of the declarations.

The device only reads the system: ``ts``, ``events``, ``meta`` and ``oracle``
stay bit-identical, so every v1 estimator returns the same output with and
without the export. Sources sampled on a finer grid than the observation
(forward-modelled BOLD) are placed on the observation grid the way the
forward model samples them: cue onsets stay in seconds, continuous channels
are point-sampled at ``t_k = k * dt``.

Declarations (:func:`declare`, :data:`DECLARATIONS`)
----------------------------------------------------
A declaration selects and possibly corrupts what the analyst conditions on;
its identifier is the hash-covered protocol field
``shared_inputs_declaration``.

========  ======================================================  ==========
id        content                                                 shared
                                                                  inputs
========  ======================================================  ==========
R         exogenous task events, context cues, slow phase and     complete
          every recorded driver
H         exogenous task events (the v1 information set)          partial
P         R without the slow phase (held out)                     partial
Q10, Q25  R with each context-cue label replaced, with            partial
          probability q, by a uniformly drawn other label
          (stream ``SeedSequence([seed, 41])``; held out)
J         R with each context-cue onset shifted by                partial
          U(-0.25, 0.25) s (stream ``SeedSequence([seed, 42])``;
          held out)
none      no inputs (resting data; the paper-2 default)           none
========  ======================================================  ==========

Exogeneity rule: only inputs set independently of the system's state enter
the basis. On the bench these are ``goal_cue``, ``stimulus``,
``other_caused``, ``context_cue`` and the slow phase (and recorded drivers);
``response``, ``feedback``, ``action`` and ``self_caused`` depend on the
system's state and are excluded, as is any event type the rule does not
name. ``include_endogenous=True`` declares every event type (the all-events
sensitivity analysis).

Label errors draw, for every context cue in onset order, one uniform number
and one replacement index from the label-error stream whatever ``q`` is, so
the cues relabelled at ``q = 0.10`` are a subset of those relabelled at
``q = 0.25`` and receive the same wrong labels (a nested dose). The labels
are those of the paradigm (``K`` contexts, ``0 .. K - 1``); with a single
context there is no other label and nothing is relabelled. Jittered onsets
are clipped to the recording, rounded to 1 us like the events table, and
each cue then stays in force until the next (jittered) cue.

Input basis (:func:`input_basis`)
---------------------------------
For every declared channel ``u``: event types give one boxcar per level over
``[onset, onset + duration)``; a cue track (``context_cue``, recorded
drivers) is a state that stays in force until the next cue and is coded by
one indicator per label except the smallest (reference) label, because the
indicators of all labels sum to the intercept; continuous channels are
z-scored over the run. The basis is

    ``U(t) = [u(t - l), u~_tau(t - l)]`` for ``l`` in ``{0} U L`` and
    ``tau`` in ``{tau_c, 3 tau_c, 10 tau_c}``,

where ``u~_tau(t) = a u~_tau(t - 1) + (1 - a) u(t)``, ``a = exp(-dt / tau)``,
is the causal first-order low-pass started at 0, and ``L`` is the
estimator's lag set: NAS ``L = sorted unique round(geomspace(1, l_max,
min(4, l_max)))``, IIM ``L = {1, ..., l_max}``, with ``l_max = ceil(tau_c /
dt)``. Lag 0 is included because the inputs are exogenous. Samples before a
lag reaches back into the recording are 0 (``first_complete_sample``).
Every conditioned null shifts the raw series first and fits the projections
on ``U`` afterwards (``NULL_ORDER = 'shift_then_project'``).

Coupling time scale (:func:`coupling_timescale`)
------------------------------------------------
``tau_c`` is the intrinsic time constant of the coupled units, read from the
generator constants before any estimator runs: family A ``AgentConfig.tau``,
family C ``1 / AgentConfig.sl_rate``, the Hopf whole-brain model ``1 /
WholeBrainConfig.rate`` (0.1 s on every bench substrate). Families A and C
at 20 Hz get the v1 lag set ``{1, 2}``, the Hopf model at 250 Hz ``{1, 3, 9,
25}``. The resolvability gate ``dt <= tau_c / 2``
(:func:`sampling_resolved`) is applied by the estimators.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from impact_pipeline.v2 import numerics as _numerics
from impact_pipeline.v2 import records as _records
from impact_pipeline.v2 import seeds as _seeds
from impact_pipeline.v2.provenance import array_sha256

DECLARED_INPUTS_VERSION = "mpc-bench-declared-inputs/1.0.0"

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
CONTEXT_CUE = "context_cue"
SLOW_PHASE = "slow_phase"
SLOW_PHASE_CHANNELS = ("slow_phase_sin", "slow_phase_cos")
COMMON_DRIVER = "common_driver"
# Oracle keys through which a generator exposes further exogenous inputs.
RECORDABLE_STATES_KEY = "recordable_states"
RECORDABLE_CHANNELS_KEY = "recordable_channels"

# Exogeneity rule (design 2.0.1; NAS N3).
EXOGENOUS_TASK_EVENT_TYPES = ("goal_cue", "stimulus", "other_caused")
ENDOGENOUS_EVENT_TYPES = ("response", "feedback", "action", "self_caused")
EXOGENOUS_INPUTS = EXOGENOUS_TASK_EVENT_TYPES + (CONTEXT_CUE, SLOW_PHASE)

CODING_EVENT = "event"  # boxcar per level
CODING_STATE = "state"  # in force until the next cue; reference-coded
CUE_COLUMNS = ("onset", "duration", "trial_type", "value", "event_id")
DECLARED_EVENT_COLUMNS = ("onset", "duration", "trial_type", "value", "coding")
ONSET_DECIMALS = 6  # the events table rounds onsets to 1 us

CUE_JITTER_HALF_WIDTH_SEC = 0.25
LABEL_ERROR_STREAM = "label_error"  # SeedSequence([seed, 41])
CUE_JITTER_STREAM = "cue_jitter"  # SeedSequence([seed, 42])

# Coupling time scale and lags (design 2.0.3).
TAU_C_DEFAULT_SEC = 0.1  # human-EEG default; bench substrates derive theirs
TAU_C_SENSITIVITY_SEC = (0.05, 0.2)  # reported on the bench, never chosen after
BASIS_TAU_MULTIPLIERS = (1.0, 3.0, 10.0)
MAX_NAS_LAGS = 4
ESTIMATORS = ("NAS", "IIM")
NULL_ORDER = "shift_then_project"
SPAN_RTOL = 1e-10

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class DeclaredInputsError(ValueError):
    """Recorded inputs, a declaration or a basis request that cannot be
    built."""


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DeclarationSpec:
    """What one declaration selects from the recorded inputs."""

    declaration_id: str
    description: str
    shared_inputs: str
    task_events: bool
    context_cue: bool
    slow_phase: bool
    drivers: bool
    label_error_q: float = 0.0
    onset_jitter_sec: float = 0.0
    held_out: bool = False

    def __post_init__(self):
        if self.shared_inputs not in _records.SHARED_INPUTS:
            raise DeclaredInputsError(
                f"shared_inputs must be one of {_records.SHARED_INPUTS}")
        if not 0.0 <= float(self.label_error_q) <= 1.0:
            raise DeclaredInputsError("label_error_q must lie in [0, 1]")
        if not float(self.onset_jitter_sec) >= 0.0:
            raise DeclaredInputsError("onset_jitter_sec must be >= 0")
        if (self.label_error_q or self.onset_jitter_sec) and not self.context_cue:
            raise DeclaredInputsError(
                "label errors and onset jitter act on declared context cues")

    def to_dict(self) -> dict:
        return {
            "declaration_id": self.declaration_id,
            "description": self.description,
            "shared_inputs": self.shared_inputs,
            "task_events": bool(self.task_events),
            "context_cue": bool(self.context_cue),
            "slow_phase": bool(self.slow_phase),
            "drivers": bool(self.drivers),
            "label_error_q": float(self.label_error_q),
            "onset_jitter_sec": float(self.onset_jitter_sec),
            "held_out": bool(self.held_out),
        }


def _spec(did, description, shared, task, ctx, slow, drivers, q=0.0, jitter=0.0,
          held_out=False):
    return DeclarationSpec(did, description, shared, task, ctx, slow, drivers,
                           q, jitter, held_out)


DECLARATIONS: Mapping[str, DeclarationSpec] = {
    "R": _spec("R", "complete: exogenous task events, context cues, slow phase "
               "and every recorded driver", "complete", True, True, True, True),
    "H": _spec("H", "task only (the v1 information set): exogenous task events",
               "partial", True, False, False, False),
    "P": _spec("P", "R without the slow phase", "partial", True, True, False,
               True, held_out=True),
    "Q10": _spec("Q10", "R with each context-cue label replaced with "
                 "probability 0.10 by a uniformly drawn other label",
                 "partial", True, True, True, True, q=0.10, held_out=True),
    "Q25": _spec("Q25", "R with each context-cue label replaced with "
                 "probability 0.25 by a uniformly drawn other label",
                 "partial", True, True, True, True, q=0.25, held_out=True),
    "J": _spec("J", "R with each context-cue onset shifted by U(-0.25, 0.25) s",
               "partial", True, True, True, True,
               jitter=CUE_JITTER_HALF_WIDTH_SEC, held_out=True),
    "none": _spec("none", "no inputs (resting data; the paper-2 default)",
                  "none", False, False, False, False),
}
DECLARATION_IDS = tuple(DECLARATIONS)
HELD_OUT_DECLARATIONS = tuple(d for d, s in DECLARATIONS.items() if s.held_out)


def declaration_spec(declaration_id: str) -> DeclarationSpec:
    try:
        return DECLARATIONS[declaration_id]
    except KeyError:
        raise DeclaredInputsError(
            f"unknown declaration {declaration_id!r}; one of {DECLARATION_IDS}"
        ) from None


def exogenous_event_types(types: Iterable[str], include_endogenous: bool = False
                          ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    ``(declared, excluded)`` event types under the exogeneity rule: the
    exogenous task types in rule order, then (with ``include_endogenous``,
    the all-events sensitivity) every other type in sorted order. Without it
    every type the rule does not name exogenous is excluded.
    """
    present = []
    for t in types:
        if t not in present:
            present.append(str(t))
    declared = [t for t in EXOGENOUS_TASK_EVENT_TYPES if t in present]
    others = sorted(t for t in present if t not in EXOGENOUS_TASK_EVENT_TYPES)
    if include_endogenous:
        return tuple(declared + others), ()
    return tuple(declared), tuple(others)


# --------------------------------------------------------------------------
# coupling time scale and lag sets
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CouplingTimescale:
    """``tau_c`` and the generator constant it was read from."""

    tau_c_sec: float
    constant: str
    constant_value: float
    rule: str

    def to_dict(self) -> dict:
        return {"tau_c_sec": self.tau_c_sec, "constant": self.constant,
                "constant_value": self.constant_value, "rule": self.rule}


def _meta_of(system_or_meta) -> Mapping:
    meta = getattr(system_or_meta, "meta", system_or_meta)
    if not isinstance(meta, Mapping):
        raise DeclaredInputsError("expected a system or its meta mapping")
    return meta


def _positive(value, what) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise DeclaredInputsError(f"{what} is not a number: {value!r}") from None
    if not (math.isfinite(v) and v > 0):
        raise DeclaredInputsError(f"{what} must be finite and > 0, got {value!r}")
    return v


def coupling_timescale(system_or_meta) -> CouplingTimescale:
    """
    ``tau_c`` of a bench system, from its generator constants (never from
    data): rate units ``AgentConfig.tau``; Stuart-Landau units
    ``1 / AgentConfig.sl_rate``; the Hopf whole-brain model ``1 /
    WholeBrainConfig.rate``; the null-calibration generator is classified
    under the family-A protocol and takes the family-A default
    ``AgentConfig.tau``. Forward-modelled systems keep their source's meta
    and hence its ``tau_c``. Raises :class:`DeclaredInputsError` when the
    meta names no such constant (declare ``tau_c`` then, e.g.
    :data:`TAU_C_DEFAULT_SEC` for human EEG).
    """
    meta = _meta_of(system_or_meta)
    dyn = meta.get("dynamics")
    if dyn in ("rate", "stuart_landau"):
        cfg = meta.get("config")
        if not isinstance(cfg, Mapping):
            raise DeclaredInputsError(f"{dyn} system without its AgentConfig in meta")
        if dyn == "rate":
            tau = _positive(cfg.get("tau"), "AgentConfig.tau")
            return CouplingTimescale(tau, "AgentConfig.tau", tau,
                                     "rate-unit time constant")
        rate = _positive(cfg.get("sl_rate"), "AgentConfig.sl_rate")
        return CouplingTimescale(1.0 / rate, "AgentConfig.sl_rate", rate,
                                 "1 / Stuart-Landau amplitude rate")
    if dyn == "hopf_whole_brain":
        cfg = meta.get("whole_brain_config")
        if not isinstance(cfg, Mapping):
            raise DeclaredInputsError("whole-brain system without its config in meta")
        rate = _positive(cfg.get("rate"), "WholeBrainConfig.rate")
        return CouplingTimescale(1.0 / rate, "WholeBrainConfig.rate", rate,
                                 "1 / Hopf amplitude rate")
    if meta.get("family") == "null_calibration":
        from impact_pipeline.bench.generators import AgentConfig

        tau = float(AgentConfig().tau)
        return CouplingTimescale(tau, "AgentConfig.tau", tau,
                                 "family-A default (null-calibration generator)")
    raise DeclaredInputsError(
        f"no generator constant gives tau_c for dynamics {dyn!r} "
        f"(family {meta.get('family')!r}); declare tau_c")


def l_max(tau_c: float, dt: float) -> int:
    """``ceil(tau_c / dt)`` (at least 1), robust to the binary representation
    of the two numbers (0.1 / 0.004 is 25, not 26)."""
    tau_c = _positive(tau_c, "tau_c")
    dt = _positive(dt, "dt")
    return max(1, int(math.ceil(round(tau_c / dt, 9))))


def sampling_resolved(tau_c: float, dt: float) -> bool:
    """The resolvability condition ``dt <= tau_c / 2`` (``SAMPLING_UNRESOLVED``
    otherwise; the gate itself is the estimators')."""
    tau_c = _positive(tau_c, "tau_c")
    dt = _positive(dt, "dt")
    return round(dt / tau_c, 9) <= 0.5


def nas_lag_set(tau_c: float, dt: float) -> Tuple[int, ...]:
    """NAS lag set ``sorted unique round(geomspace(1, l_max, min(4, l_max)))``:
    ``{1, 2}`` at 20 Hz, ``{1, 3, 9, 25}`` at 250 Hz, ``{1, 4, 14, 50}`` at
    500 Hz (``tau_c = 0.1 s``)."""
    lm = l_max(tau_c, dt)
    q = min(MAX_NAS_LAGS, lm)
    lags = np.unique(np.round(np.geomspace(1.0, float(lm), q)).astype(int))
    return tuple(int(v) for v in lags)


def iim_lag_set(tau_c: float, dt: float) -> Tuple[int, ...]:
    """IIM lag set ``{1, ..., l_max}``."""
    return tuple(range(1, l_max(tau_c, dt) + 1))


def lag_set(tau_c: float, dt: float, estimator: str = "NAS") -> Tuple[int, ...]:
    if estimator == "NAS":
        return nas_lag_set(tau_c, dt)
    if estimator == "IIM":
        return iim_lag_set(tau_c, dt)
    raise DeclaredInputsError(f"estimator must be one of {ESTIMATORS}")


def basis_lags(tau_c: float, dt: float, estimator: str = "NAS") -> Tuple[int, ...]:
    """The lags of the input basis: ``{0} U L``."""
    return (0,) + lag_set(tau_c, dt, estimator)


def basis_taus(tau_c: float) -> Tuple[float, ...]:
    """Time constants of the filtered copies: ``tau_c, 3 tau_c, 10 tau_c``."""
    tau_c = _positive(tau_c, "tau_c")
    return tuple(round(float(m * tau_c), 12) for m in BASIS_TAU_MULTIPLIERS)


@dataclass(frozen=True)
class LagPlan:
    """Lags and resolvability of one (``tau_c``, ``dt``) pair."""

    tau_c_sec: float
    dt: float
    l_max: int
    resolved: bool
    nas_lags: Tuple[int, ...]
    iim_lags: Tuple[int, ...]
    basis_taus: Tuple[float, ...]

    def basis_lags(self, estimator: str = "NAS") -> Tuple[int, ...]:
        return (0,) + (self.nas_lags if estimator == "NAS" else self.iim_lags)

    def to_dict(self) -> dict:
        return {"tau_c_sec": self.tau_c_sec, "dt": self.dt, "l_max": self.l_max,
                "resolved": self.resolved, "nas_lags": list(self.nas_lags),
                "iim_lags": list(self.iim_lags), "basis_taus": list(self.basis_taus)}


def lag_plan(system_or_meta=None, *, tau_c: Optional[float] = None,
             dt: Optional[float] = None) -> LagPlan:
    """The lag plan of a system (``tau_c`` from its generator constants, ``dt``
    from its meta) or of explicit ``tau_c`` and ``dt``."""
    if tau_c is None:
        if system_or_meta is None:
            raise DeclaredInputsError("lag_plan needs a system or tau_c")
        tau_c = coupling_timescale(system_or_meta).tau_c_sec
    if dt is None:
        if system_or_meta is None:
            raise DeclaredInputsError("lag_plan needs a system or dt")
        dt = _meta_of(system_or_meta).get("dt")
    tau_c, dt = _positive(tau_c, "tau_c"), _positive(dt, "dt")
    return LagPlan(tau_c, dt, l_max(tau_c, dt), sampling_resolved(tau_c, dt),
                   nas_lag_set(tau_c, dt), iim_lag_set(tau_c, dt),
                   basis_taus(tau_c))


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _value(v):
    """A JSON-friendly Python scalar for an event value (None for missing)."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        if not math.isfinite(f):
            return None
        return int(f) if f.is_integer() else f
    return str(v)


def _level_key(v):
    v = _value(v)
    if v is None:
        return (2, 0.0, "")
    if isinstance(v, (int, float)):
        return (0, float(v), "")
    return (1, 0.0, str(v))


def level_label(v) -> str:
    """The label of an event level in channel names (``n/a`` when missing)."""
    v = _value(v)
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def _check_name(name, what) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise DeclaredInputsError(f"{what} name must be an identifier, got {name!r}")
    return name


def _round_onset(x):
    return np.round(np.asarray(x, dtype=float), ONSET_DECIMALS)


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _frame_payload(df: pd.DataFrame, columns: Sequence[str]) -> list:
    rows = []
    for rec in df.loc[:, list(columns)].itertuples(index=False, name=None):
        out = []
        for c, v in zip(columns, rec):
            if c in ("onset", "duration"):
                finite = v is not None and math.isfinite(float(v))
                v = round(float(v), 9) if finite else None
            else:
                v = _value(v)
            out.append(v)
        rows.append(out)
    return rows


# --------------------------------------------------------------------------
# the recording device
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RecordedInputs:
    """
    What an experimenter could log during one simulated run: the task events
    (a copy of ``system.events``), cue tracks (``context_cue`` and recorded
    driver states, columns :data:`CUE_COLUMNS`, onsets and durations in
    seconds) with their label alphabets, and continuous channels on the
    observation grid (``channels``, grouped by ``channel_groups``). Arrays
    are read-only copies.
    """

    n_time: int
    dt: float
    seed: Optional[int]
    task_events: pd.DataFrame
    cue_events: pd.DataFrame
    cue_alphabets: Mapping[str, Tuple]
    channels: Mapping[str, np.ndarray]
    channel_groups: Mapping[str, Tuple[str, ...]]
    source: Mapping = field(default_factory=dict)

    @property
    def cue_types(self) -> Tuple[str, ...]:
        return tuple(self.cue_alphabets)

    @property
    def has_context_cues(self) -> bool:
        return CONTEXT_CUE in self.cue_alphabets

    @property
    def has_slow_phase(self) -> bool:
        return SLOW_PHASE in self.channel_groups

    @property
    def driver_cue_types(self) -> Tuple[str, ...]:
        return tuple(t for t in self.cue_alphabets if t != CONTEXT_CUE)

    @property
    def driver_groups(self) -> Tuple[str, ...]:
        return tuple(g for g in self.channel_groups if g != SLOW_PHASE)

    def cues(self, trial_type: str = CONTEXT_CUE) -> pd.DataFrame:
        """The cue rows of one track, in onset order (a copy)."""
        sub = self.cue_events[self.cue_events["trial_type"] == trial_type]
        return sub.reset_index(drop=True).copy()

    def events_table(self) -> pd.DataFrame:
        """Task events and cues in one table, sorted by onset (a new frame;
        ``system.events`` is not touched)."""
        parts = [self.task_events]
        if len(self.cue_events):
            parts.append(self.cue_events)
        df = pd.concat(parts, ignore_index=True, sort=False)
        if "event_id" in df.columns:
            df = df.sort_values(["onset", "event_id"], kind="mergesort",
                                na_position="last")
        else:
            df = df.sort_values(["onset"], kind="mergesort")
        return df.reset_index(drop=True)

    def sha256(self) -> str:
        payload = {
            "version": DECLARED_INPUTS_VERSION,
            "n_time": self.n_time,
            "dt": round(self.dt, 12),
            "seed": self.seed,
            "task_events": _frame_payload(
                self.task_events,
                [c for c in ("onset", "duration", "trial_type", "value")
                 if c in self.task_events.columns]),
            "cue_events": _frame_payload(self.cue_events, CUE_COLUMNS),
            "cue_alphabets": {k: [_value(v) for v in a]
                              for k, a in self.cue_alphabets.items()},
            "channels": {k: array_sha256(v) for k, v in self.channels.items()},
            "channel_groups": {k: list(v) for k, v in self.channel_groups.items()},
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _source_dt(meta: Mapping) -> Optional[float]:
    cfg = meta.get("config")
    if isinstance(cfg, Mapping) and cfg.get("dt") is not None:
        return float(cfg["dt"])
    wb = meta.get("whole_brain_config")
    if isinstance(wb, Mapping) and wb.get("fs_out"):
        return 1.0 / float(wb["fs_out"])
    return None


def _grid(n_src: int, n_obs: int, dt_obs: float, meta: Mapping, what: str):
    """``(step, dt_src)`` mapping a source series of ``n_src`` samples onto
    the observation grid (``step`` source samples per observation sample)."""
    if n_src == n_obs:
        return 1, dt_obs
    dt_src = _source_dt(meta)
    if dt_src is None or dt_src <= 0:
        raise DeclaredInputsError(
            f"{what}: {n_src} samples on an unknown grid, observation {n_obs}")
    step = int(round(dt_obs / dt_src))
    if (step < 1 or not math.isclose(step * dt_src, dt_obs, rel_tol=1e-6)
            or int(math.ceil(n_src / step)) != n_obs):
        raise DeclaredInputsError(
            f"{what}: {n_src} samples at {dt_src} s do not map onto "
            f"{n_obs} samples at {dt_obs} s")
    return step, dt_src


def _state_array(values, what) -> np.ndarray:
    a = np.asarray(values)
    if a.ndim != 1 or a.size == 0:
        raise DeclaredInputsError(f"{what}: a state track is one label per sample")
    if a.dtype.kind == "b":
        a = a.astype(np.int64)
    if a.dtype.kind == "f":
        if not np.all(np.isfinite(a)) or not np.all(a == np.round(a)):
            raise DeclaredInputsError(f"{what}: state labels must be integers")
        a = a.astype(np.int64)
    if a.dtype.kind not in "iu":
        raise DeclaredInputsError(f"{what}: state labels must be integers")
    return a.astype(np.int64)


def _cues_from_state(name: str, state: np.ndarray, dt_src: float) -> pd.DataFrame:
    """One cue per change of the state (and one at sample 0)."""
    n = state.size
    idx = np.r_[0, np.flatnonzero(state[1:] != state[:-1]) + 1].astype(np.int64)
    ends = np.r_[idx[1:], n]
    tag = "ctx" if name == CONTEXT_CUE else f"{name}_"
    return pd.DataFrame({
        "onset": _round_onset(idx * dt_src),
        "duration": _round_onset((ends - idx) * dt_src),
        "trial_type": name,
        "value": state[idx].astype(np.int64),
        "event_id": [f"{tag}{j:04d}" for j in range(idx.size)],
    }, columns=list(CUE_COLUMNS))


def _channel_block(name: str, values, what) -> Dict[str, np.ndarray]:
    a = np.asarray(values, dtype=float)
    if a.ndim == 1:
        return {name: a}
    if a.ndim == 2 and a.shape[0] >= 1:
        if a.shape[0] == 1:
            return {name: a[0]}
        return {f"{name}_{i}": a[i] for i in range(a.shape[0])}
    raise DeclaredInputsError(f"{what}: a channel is one value per sample")


def record_inputs(system, *, extra_states: Optional[Mapping] = None,
                  extra_channels: Optional[Mapping] = None) -> RecordedInputs:
    """
    The recording device (module docstring): build the :class:`RecordedInputs`
    of a simulated system without touching it. ``extra_states`` (``{name:
    integer label per sample}``) and ``extra_channels`` (``{name: values per
    sample}``) add recorded drivers to what the generator exposes.
    """
    meta = system.meta or {}
    oracle = system.oracle or {}
    ts = np.asarray(system.ts)
    if ts.ndim != 2:
        raise DeclaredInputsError("ts must be nodes x time")
    n_time = int(ts.shape[1])
    dt = _positive(meta.get("dt"), "meta['dt']")
    seed = meta.get("seed")
    seed = None if seed is None else int(seed)

    events = system.events
    if isinstance(events, pd.DataFrame):
        task_events = events.copy(deep=True)
    else:
        task_events = pd.DataFrame(columns=["onset", "duration", "trial_type", "value"])
    task_types = (set(task_events["trial_type"].astype(str)) if len(task_events)
                  else set())

    states: Dict[str, np.ndarray] = {}
    alphabets: Dict[str, Tuple] = {}
    channels: Dict[str, np.ndarray] = {}
    groups: Dict[str, Tuple[str, ...]] = {}

    def add_state(name, values, what, alphabet=None):
        _check_name(name, "state track")
        if (name in states or name in task_types or name in channels
                or name in groups):
            raise DeclaredInputsError(f"{what}: {name!r} is already recorded")
        states[name] = _state_array(values, what)
        if alphabet is not None:
            alphabets[name] = tuple(alphabet)

    def add_group(group, block, what):
        _check_name(group, "channel")
        if group in groups or group in states or group in task_types:
            raise DeclaredInputsError(f"{what}: {group!r} is already recorded")
        for ch in block:
            if ch in channels or ch in states:
                raise DeclaredInputsError(f"{what}: channel {ch!r} is already recorded")
        channels.update(block)
        groups[group] = tuple(block)

    ctx = oracle.get("context_state")
    if ctx is not None:
        knobs = meta.get("knobs") if isinstance(meta.get("knobs"), Mapping) else {}
        k = knobs.get("K") if knobs else None
        add_state(CONTEXT_CUE, ctx, "oracle['context_state']",
                  alphabet=None if k is None else range(int(k)))
    phase = oracle.get("slow_phase")
    if phase is not None:
        ph = np.asarray(phase, dtype=float)
        add_group(SLOW_PHASE, {SLOW_PHASE_CHANNELS[0]: np.sin(ph),
                               SLOW_PHASE_CHANNELS[1]: np.cos(ph)},
                  "oracle['slow_phase']")
    if oracle.get("adversarial") == COMMON_DRIVER and oracle.get("driver") is not None:
        add_group(COMMON_DRIVER, {COMMON_DRIVER: np.asarray(oracle["driver"], float)},
                  "oracle['driver']")
    for where, source in ((f"oracle[{RECORDABLE_STATES_KEY!r}]",
                           oracle.get(RECORDABLE_STATES_KEY)),
                          ("extra_states", extra_states)):
        for name, values in (source or {}).items():
            add_state(name, values, f"{where}[{name!r}]")
    for where, source in ((f"oracle[{RECORDABLE_CHANNELS_KEY!r}]",
                           oracle.get(RECORDABLE_CHANNELS_KEY)),
                          ("extra_channels", extra_channels)):
        for name, values in (source or {}).items():
            _check_name(name, "channel")
            add_group(name, _channel_block(name, values, f"{where}[{name!r}]"),
                      f"{where}[{name!r}]")

    # cue tracks: onsets in seconds on their own (source) grid
    cue_frames = []
    steps = {}
    for name, state in states.items():
        step, dt_src = _grid(state.size, n_time, dt, meta, f"state track {name!r}")
        steps[name] = step
        cue_frames.append(_cues_from_state(name, state, dt_src))
        seen = tuple(int(v) for v in np.unique(state))
        alpha = alphabets.get(name)
        alphabets[name] = tuple(sorted(set(seen) | set(int(v) for v in (alpha or ()))))
    cue_events = (pd.concat(cue_frames, ignore_index=True) if cue_frames
                  else pd.DataFrame(columns=list(CUE_COLUMNS)))
    if len(cue_events):
        cue_events = cue_events.sort_values(["onset", "event_id"], kind="mergesort")
        cue_events = cue_events.reset_index(drop=True)
    # continuous channels on the observation grid (point samples)
    out_channels = {}
    for name, values in channels.items():
        if values.ndim != 1:
            raise DeclaredInputsError(f"channel {name!r} is not one value per sample")
        if not np.all(np.isfinite(values)):
            raise DeclaredInputsError(f"channel {name!r} has non-finite values")
        step, _ = _grid(values.size, n_time, dt, meta, f"channel {name!r}")
        steps[name] = step
        out_channels[name] = _readonly(np.array(values[::step][:n_time], dtype=float))
    source = {
        "family": meta.get("family"),
        "generator_version": meta.get("generator_version"),
        "grid_steps": steps,
    }
    return RecordedInputs(
        n_time=n_time, dt=dt, seed=seed, task_events=task_events,
        cue_events=cue_events, cue_alphabets={k: alphabets[k] for k in states},
        channels=out_channels, channel_groups=groups, source=source)


# --------------------------------------------------------------------------
# declared inputs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DeclaredInputs:
    """
    The inputs one declaration hands to an estimator: declared event and cue
    rows (columns :data:`DECLARED_EVENT_COLUMNS`; ``coding`` is ``event`` or
    ``state``), the label alphabets of the cue tracks, and the declared
    continuous channels (raw values; z-scored by the basis). ``details``
    records what the declaration did (excluded event types, relabelled cues,
    onset shifts, stream keys).
    """

    declaration_id: str
    shared_inputs: str
    n_time: int
    dt: float
    seed: Optional[int]
    events: pd.DataFrame
    cue_alphabets: Mapping[str, Tuple]
    channels: Mapping[str, np.ndarray]
    include_endogenous: bool = False
    declared_event_types: Tuple[str, ...] = ()
    excluded_event_types: Tuple[str, ...] = ()
    details: Mapping = field(default_factory=dict)

    @property
    def spec(self) -> DeclarationSpec:
        return declaration_spec(self.declaration_id)

    @property
    def label(self) -> str:
        """The scoring label: the declaration id, with ``+all_events`` for the
        all-events sensitivity."""
        return self.declaration_id + ("+all_events" if self.include_endogenous else "")

    @property
    def event_types(self) -> Tuple[str, ...]:
        """The declared event types in rule order (exogenous types first),
        restricted to types with at least one declared row."""
        ev = self.events[self.events["coding"] == CODING_EVENT]
        return _type_order(ev, self.declared_event_types)

    @property
    def state_types(self) -> Tuple[str, ...]:
        return tuple(self.cue_alphabets)

    def cues(self, trial_type: str = CONTEXT_CUE) -> pd.DataFrame:
        sub = self.events[self.events["trial_type"] == trial_type]
        return sub.reset_index(drop=True).copy()

    def identifiability(self, observation: str = "direct",
                        hub_privileged="not_tested") -> dict:
        """The identifiability record of a component scored under this
        declaration (record schema vocabulary)."""
        return _records.validate_identifiability({
            "shared_inputs": self.shared_inputs, "observation": observation,
            "hub_privileged": hub_privileged})

    def events_table(self) -> pd.DataFrame:
        return self.events.copy()

    def sha256(self) -> str:
        payload = {
            "version": DECLARED_INPUTS_VERSION,
            "declaration": self.declaration_id,
            "include_endogenous": bool(self.include_endogenous),
            "n_time": self.n_time,
            "dt": round(self.dt, 12),
            "events": _frame_payload(self.events, DECLARED_EVENT_COLUMNS),
            "cue_alphabets": {k: [_value(v) for v in a]
                              for k, a in self.cue_alphabets.items()},
            "channels": {k: array_sha256(v) for k, v in self.channels.items()},
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _type_order(events: pd.DataFrame, order: Sequence[str]) -> Tuple[str, ...]:
    """Trial types present in ``events``: those of ``order`` first, in that
    order, then any other type in sorted order."""
    present = set(events["trial_type"].astype(str)) if len(events) else set()
    first = [t for t in order if t in present]
    return tuple(first + sorted(present - set(first)))


def _stream_rng(seed, stream: str) -> np.random.Generator:
    if seed is None:
        raise DeclaredInputsError(
            f"the {stream} stream needs the system seed (meta['seed'])")
    return np.random.default_rng(_seeds.stream_seed_sequence(seed, stream))


def _relabel(labels: np.ndarray, alphabet: Tuple, q: float, seed):
    """Label errors (module docstring): returns new labels and the indices of
    the relabelled cues. Draws ``u`` and a replacement index for every cue."""
    rng = _stream_rng(seed, LABEL_ERROR_STREAM)
    n = labels.size
    u = rng.random(n)
    alphabet = tuple(int(a) for a in alphabet)
    out = labels.copy()
    if len(alphabet) < 2:
        return out, np.empty(0, dtype=np.int64)
    pick = rng.integers(0, len(alphabet) - 1, size=n)
    hit = np.flatnonzero(u < q)
    for j in hit:
        others = [a for a in alphabet if a != int(labels[j])]
        out[j] = others[int(pick[j])]
    return out, hit.astype(np.int64)


def _jitter(onsets: np.ndarray, half_width: float, n_time: int, dt: float, seed):
    """Onset jitter (module docstring): returns jittered, clipped, rounded
    onsets in the original cue order and the drawn shifts."""
    rng = _stream_rng(seed, CUE_JITTER_STREAM)
    shift = rng.uniform(-half_width, half_width, size=onsets.size)
    t_end = (n_time - 1) * dt
    new = np.clip(_round_onset(onsets + shift), 0.0, round(t_end, ONSET_DECIMALS))
    return new, shift


def _tile_durations(onsets: np.ndarray, n_time: int, dt: float) -> np.ndarray:
    ends = np.r_[onsets[1:], n_time * dt]
    return _round_onset(np.maximum(ends - onsets, 0.0))


def declare(recorded: RecordedInputs, declaration_id: str, *,
            include_endogenous: bool = False, seed: Optional[int] = None
            ) -> DeclaredInputs:
    """
    Apply one declaration (:data:`DECLARATIONS`) to recorded inputs. Label
    errors and onset jitter use the system seed (``recorded.seed``) unless
    ``seed`` is given; nothing else is random. ``include_endogenous``
    declares every task event type (all-events sensitivity).
    """
    spec = declaration_spec(declaration_id)
    seed = recorded.seed if seed is None else int(seed)
    rows = []
    excluded: Tuple[str, ...] = ()
    declared_types: Tuple[str, ...] = ()
    details: dict = {"version": DECLARED_INPUTS_VERSION,
                     "spec": spec.to_dict(), "seed": seed}

    if spec.task_events and len(recorded.task_events):
        te = recorded.task_events
        declared_types, excluded = exogenous_event_types(
            te["trial_type"].astype(str), include_endogenous)
        for ttype in declared_types:
            sub = te[te["trial_type"].astype(str) == ttype]
            for on, du, v in zip(sub["onset"], sub["duration"], sub["value"]):
                rows.append((float(on), float(du), ttype, _value(v), CODING_EVENT))
        details["declared_event_types"] = list(declared_types)
    details["excluded_event_types"] = list(excluded)

    alphabets: Dict[str, Tuple] = {}
    cue_types = []
    if spec.context_cue and recorded.has_context_cues:
        cue_types.append(CONTEXT_CUE)
    if spec.drivers:
        cue_types.extend(recorded.driver_cue_types)
    for ctype in cue_types:
        cues = recorded.cues(ctype)
        on = cues["onset"].to_numpy(dtype=float)
        du = cues["duration"].to_numpy(dtype=float)
        lab = cues["value"].to_numpy(dtype=np.int64)
        if ctype == CONTEXT_CUE and spec.label_error_q > 0:
            lab, hit = _relabel(lab, recorded.cue_alphabets[ctype],
                                float(spec.label_error_q), seed)
            details["label_error"] = {
                "q": float(spec.label_error_q),
                "stream": [seed, _seeds.STREAM_KEYS[LABEL_ERROR_STREAM]],
                "n_cues": int(lab.size), "n_relabelled": int(hit.size),
                "relabelled": [int(j) for j in hit],
            }
        if ctype == CONTEXT_CUE and spec.onset_jitter_sec > 0:
            new_on, shift = _jitter(on, float(spec.onset_jitter_sec),
                                    recorded.n_time, recorded.dt, seed)
            order = np.argsort(new_on, kind="stable")
            on, lab = new_on[order], lab[order]
            du = _tile_durations(on, recorded.n_time, recorded.dt)
            details["onset_jitter"] = {
                "half_width_sec": float(spec.onset_jitter_sec),
                "stream": [seed, _seeds.STREAM_KEYS[CUE_JITTER_STREAM]],
                "n_cues": int(on.size),
                "max_abs_shift_sec": (float(np.max(np.abs(shift))) if shift.size
                                      else 0.0),
                "reordered": bool(np.any(order != np.arange(order.size))),
            }
        for o, d, v in zip(on, du, lab):
            rows.append((float(o), float(d), ctype, int(v), CODING_STATE))
        alphabets[ctype] = tuple(recorded.cue_alphabets[ctype])

    events = pd.DataFrame(rows, columns=list(DECLARED_EVENT_COLUMNS))
    events["value"] = events["value"].astype(object)
    if len(events):
        events = events.sort_values("onset", kind="mergesort").reset_index(drop=True)

    chans: Dict[str, np.ndarray] = {}
    groups = []
    if spec.slow_phase and recorded.has_slow_phase:
        groups.append(SLOW_PHASE)
    if spec.drivers:
        groups.extend(recorded.driver_groups)
    for grp in groups:
        for ch in recorded.channel_groups[grp]:
            chans[ch] = recorded.channels[ch]
    details["declared_channel_groups"] = groups
    details["declared_cue_types"] = list(alphabets)
    return DeclaredInputs(
        declaration_id=spec.declaration_id, shared_inputs=spec.shared_inputs,
        n_time=recorded.n_time, dt=recorded.dt, seed=seed, events=events,
        cue_alphabets=alphabets, channels=chans,
        include_endogenous=bool(include_endogenous),
        declared_event_types=tuple(declared_types),
        excluded_event_types=tuple(excluded), details=details)


def declare_all(recorded: RecordedInputs,
                declaration_ids: Sequence[str] = DECLARATION_IDS,
                **kwargs) -> Dict[str, DeclaredInputs]:
    """Several declarations of the same recording (one simulation, several
    scorings)."""
    return {d: declare(recorded, d, **kwargs) for d in declaration_ids}


def declaration_protocol_entry(declaration_id: str, *,
                               include_endogenous: bool = False) -> dict:
    """
    The hash-covered description of a declaration for a protocol file: the
    declaration, the exogeneity rule, the corruption streams and the basis
    rule. Deterministic and JSON-serialisable.
    """
    spec = declaration_spec(declaration_id)
    return {
        "version": DECLARED_INPUTS_VERSION,
        "shared_inputs_declaration": spec.declaration_id,
        "declaration": spec.to_dict(),
        "exogeneity": {
            "exogenous_inputs": list(EXOGENOUS_INPUTS),
            "endogenous_excluded": list(ENDOGENOUS_EVENT_TYPES),
            "unknown_event_types": "excluded",
            "include_endogenous": bool(include_endogenous),
        },
        "streams": {"label_error": _seeds.STREAM_KEYS[LABEL_ERROR_STREAM],
                    "cue_jitter": _seeds.STREAM_KEYS[CUE_JITTER_STREAM]},
        "basis": {
            "channels": "event boxcar per level; cue states reference-coded "
                        "(smallest label); continuous z-scored",
            "lags": "{0} U L; NAS L = round(geomspace(1, l_max, min(4, l_max))), "
                    "IIM L = 1..l_max, l_max = ceil(tau_c / dt)",
            "filtered_copies_tau_multipliers": list(BASIS_TAU_MULTIPLIERS),
            "filter": "causal first-order low-pass, a = exp(-dt / tau), zero start",
            "null_order": NULL_ORDER,
        },
    }


# --------------------------------------------------------------------------
# input channels and the basis
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class InputChannels:
    """The declared channels ``u(t)`` on the sample grid (before lags and
    filtering): ``values`` (n_channels x n_time), names, kinds (``event``,
    ``state``, ``continuous``), the reference label of every cue track and the
    dropped channels with their reason."""

    values: np.ndarray
    names: Tuple[str, ...]
    kinds: Tuple[str, ...]
    reference_levels: Mapping[str, object]
    dropped: Tuple[Tuple[str, str], ...]
    n_time: int
    dt: float

    @property
    def n_channels(self) -> int:
        return len(self.names)


def _sample(x: float, dt: float) -> int:
    return int(np.rint(float(x) / dt))


def _event_boxcars(events: pd.DataFrame, order: Sequence[str], n_time: int,
                   dt: float, names, kinds, cols, dropped) -> None:
    for ttype in _type_order(events, order):
        sub = events[events["trial_type"].astype(str) == ttype]
        levels = sorted({_value(v) for v in sub["value"]}, key=_level_key)
        for lev in levels:
            u = np.zeros(n_time)
            for on, du, v in zip(sub["onset"], sub["duration"], sub["value"]):
                if _value(v) != lev or not math.isfinite(float(on)):
                    continue
                a = _sample(on, dt)
                n = _sample(du, dt) if math.isfinite(float(du)) else 1
                a0, b = max(a, 0), min(n_time, a + max(1, n))
                if a0 < b:
                    u[a0:b] = 1.0
            name = f"{ttype}={level_label(lev)}"
            if not u.any():
                dropped.append((name, "empty"))
            elif u.all():
                dropped.append((name, "constant"))
            else:
                names.append(name)
                kinds.append("event")
                cols.append(u)


def _state_sequence(cues: pd.DataFrame, n_time: int, dt: float):
    """Per-sample label of one cue track, the mask of samples on which a cue
    is in force (none before the first cue) and the labels in force on at
    least one sample."""
    seq = np.zeros(n_time, dtype=np.int64)
    active = np.zeros(n_time, dtype=bool)
    on = cues["onset"].to_numpy(dtype=float)
    du = cues["duration"].to_numpy(dtype=float)
    lab = cues["value"].to_numpy(dtype=np.int64)
    order = np.argsort(on, kind="stable")
    on, du, lab = on[order], du[order], lab[order]
    starts = np.array([_sample(x, dt) for x in on], dtype=np.int64)
    for j in range(on.size):
        a = starts[j]
        n = _sample(du[j], dt) if math.isfinite(du[j]) else n_time
        end = a + max(1, n)
        if j + 1 < on.size:
            end = min(end, starts[j + 1])
        a0, b = max(a, 0), min(n_time, end)
        if a0 < b:
            seq[a0:b] = lab[j]
            active[a0:b] = True
    present = sorted({int(v) for v in seq[active]})
    return seq, active, present


def input_channels(declared: DeclaredInputs) -> InputChannels:
    """The declared channels ``u(t)`` (module docstring): event boxcars per
    level, reference-coded cue states and z-scored continuous channels.
    Channels that carry nothing beyond the intercept are dropped (an empty or
    constant boxcar, a constant continuous channel, a cue track with a single
    label)."""
    n, dt = declared.n_time, declared.dt
    names: List[str] = []
    kinds: List[str] = []
    cols: List[np.ndarray] = []
    dropped: List[Tuple[str, str]] = []
    refs: Dict[str, object] = {}
    ev = declared.events
    _event_boxcars(ev[ev["coding"] == CODING_EVENT], declared.declared_event_types,
                   n, dt, names, kinds, cols, dropped)
    for ctype in declared.cue_alphabets:
        cues = ev[(ev["coding"] == CODING_STATE) & (ev["trial_type"] == ctype)]
        seq, active, present = _state_sequence(cues, n, dt)
        if not present:
            dropped.append((ctype, "empty"))
            continue
        refs[ctype] = present[0]
        dropped.append((f"{ctype}={present[0]}", "reference"))
        for lev in present[1:]:
            u = (active & (seq == lev)).astype(float)
            names.append(f"{ctype}={lev}")
            kinds.append("state")
            cols.append(u)
    for name, x in declared.channels.items():
        x = np.asarray(x, dtype=float)
        sd = float(x.std())
        if not (math.isfinite(sd) and sd > 0):
            dropped.append((name, "constant"))
            continue
        names.append(name)
        kinds.append("continuous")
        cols.append((x - x.mean()) / sd)
    values = np.vstack(cols) if cols else np.zeros((0, n))
    return InputChannels(values=_readonly(values), names=tuple(names),
                         kinds=tuple(kinds), reference_levels=refs,
                         dropped=tuple(dropped), n_time=n, dt=dt)


def lowpass(u: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Causal first-order low-pass along the last axis, ``y(t) = a y(t - 1) +
    (1 - a) u(t)`` with ``a = exp(-dt / tau)`` and ``y(-1) = 0``."""
    from scipy.signal import lfilter

    a = math.exp(-_positive(dt, "dt") / _positive(tau, "tau"))
    return lfilter([1.0 - a], [1.0, -a], np.asarray(u, dtype=float), axis=-1)


def _lagged(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(x)
    if lag == 0:
        out[...] = x
    elif lag < x.shape[-1]:
        out[..., lag:] = x[..., :-lag]
    return out


@dataclass(frozen=True)
class InputBasis:
    """
    The input basis ``U`` (n_columns x n_time). Columns are ordered by
    channel, then copy (raw, then the filtered copies in ``taus`` order), then
    lag; ``column_channel``, ``column_tau`` (None for the raw copy) and
    ``column_lag`` describe each column. Samples before
    ``first_complete_sample`` hold zero-filled lags.
    """

    values: np.ndarray
    columns: Tuple[str, ...]
    column_channel: Tuple[str, ...]
    column_tau: Tuple[Optional[float], ...]
    column_lag: Tuple[int, ...]
    channels: Tuple[str, ...]
    channel_kinds: Tuple[str, ...]
    lags: Tuple[int, ...]
    taus: Tuple[float, ...]
    dt: float
    n_time: int
    declaration_id: str
    shared_inputs: str
    reference_levels: Mapping[str, object] = field(default_factory=dict)
    dropped: Tuple[Tuple[str, str], ...] = ()

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def first_complete_sample(self) -> int:
        return max(self.lags) if self.lags else 0

    def column_indices(self, channel: Optional[str] = None, tau="any",
                       lag: Optional[int] = None) -> np.ndarray:
        """Indices of the columns of one channel, copy (``tau=None`` for the
        raw copy) and/or lag."""
        keep = np.ones(self.n_columns, dtype=bool)
        if channel is not None:
            keep &= np.array([c == channel for c in self.column_channel], dtype=bool)
        if tau != "any":
            keep &= np.array([
                (t is None and tau is None) or (t is not None and tau is not None
                                                and math.isclose(t, tau))
                for t in self.column_tau], dtype=bool)
        if lag is not None:
            keep &= np.array([lg == lag for lg in self.column_lag], dtype=bool)
        return np.flatnonzero(keep)

    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(_canonical_json({
            "version": DECLARED_INPUTS_VERSION, "columns": list(self.columns),
            "declaration": self.declaration_id, "dt": round(self.dt, 12),
        }).encode("utf-8"))
        h.update(array_sha256(self.values).encode("ascii"))
        return h.hexdigest()

    def orthonormal_span(self, start: Optional[int] = None, rtol: float = SPAN_RTOL,
                         center: bool = True) -> np.ndarray:
        """
        An orthonormal basis (rows ``start:`` x rank) of the column space of
        ``U`` on samples ``start:`` (default ``first_complete_sample``),
        centred so that, together with an intercept, it spans the same model
        as the raw columns up to the rank cut ``rtol`` (relative to the
        largest singular value). Projections on it are numerically stable
        where the raw columns are nearly collinear (the lagged and filtered
        copies of a slow sinusoid span two dimensions plus decaying start-up
        transients).
        """
        s0 = self.first_complete_sample if start is None else int(start)
        X = np.asarray(self.values[:, s0:], dtype=float).T
        if X.shape[1] == 0 or X.shape[0] == 0:
            return np.zeros((X.shape[0], 0))
        if center:
            X = X - X.mean(axis=0, keepdims=True)
        U, s, _ = _numerics.svd(X, full_matrices=False)
        if not s.size or s[0] <= 0:
            return np.zeros((X.shape[0], 0))
        r = int(np.sum(s > float(rtol) * s[0]))
        return U[:, :r]

    def rank(self, start: Optional[int] = None, rtol: float = SPAN_RTOL) -> int:
        return int(self.orthonormal_span(start, rtol).shape[1])


def input_basis(declared: DeclaredInputs, *, tau_c: Optional[float] = None,
                estimator: str = "NAS", lags: Optional[Sequence[int]] = None,
                taus: Optional[Sequence[float]] = None) -> InputBasis:
    """
    The common input basis of a declaration (module docstring). By default
    the lags are ``{0} U L`` of ``estimator`` (``NAS`` or ``IIM``) and the
    filtered copies have time constants ``tau_c, 3 tau_c, 10 tau_c``; both
    can be given explicitly (then ``tau_c`` may be omitted). The basis has
    no intercept column and cue states are reference-coded, so an estimator
    fits it together with an intercept (or on centred series). Because
    ``u~(t) = a u~(t - 1) + (1 - a) u(t)``, a filtered copy at lag ``l`` is a
    combination of the copy at lag ``l + 1`` and the raw input at lag ``l``:
    with contiguous lags (every IIM lag set, NAS at 20 Hz) every filtered
    copy but the last lag is exactly collinear, and with sparse lags the
    copies of a slow filter are nearly so. Estimators therefore use a
    rank-revealing solve or :meth:`InputBasis.orthonormal_span`.
    """
    dt = declared.dt
    if lags is None:
        if tau_c is None:
            raise DeclaredInputsError("input_basis needs tau_c or explicit lags")
        lags = basis_lags(tau_c, dt, estimator)
    if taus is None:
        if tau_c is None:
            raise DeclaredInputsError("input_basis needs tau_c or explicit taus")
        taus = basis_taus(tau_c)
    lags = tuple(int(v) for v in lags)
    if not lags or any(v < 0 for v in lags) or len(set(lags)) != len(lags):
        raise DeclaredInputsError(f"lags must be distinct and >= 0, got {lags}")
    taus = tuple(_positive(t, "tau") for t in taus)
    ch = input_channels(declared)
    n = declared.n_time
    copies = [(None, ch.values)]
    if ch.n_channels:
        copies += [(t, lowpass(ch.values, dt, t)) for t in taus]
    blocks = []
    columns, col_ch, col_tau, col_lag = [], [], [], []
    for i, name in enumerate(ch.names):
        for tau, series in copies:
            u = series[i]
            for lg in lags:
                blocks.append(_lagged(u, lg))
                tag = "" if tau is None else f"~{tau:g}s"
                columns.append(f"{name}{tag}@lag{lg}")
                col_ch.append(name)
                col_tau.append(tau)
                col_lag.append(lg)
    values = np.vstack(blocks) if blocks else np.zeros((0, n))
    return InputBasis(
        values=_readonly(np.ascontiguousarray(values, dtype=float)),
        columns=tuple(columns), column_channel=tuple(col_ch),
        column_tau=tuple(col_tau), column_lag=tuple(col_lag),
        channels=ch.names, channel_kinds=ch.kinds, lags=lags, taus=taus, dt=dt,
        n_time=n, declaration_id=declared.label,
        shared_inputs=declared.shared_inputs,
        reference_levels=dict(ch.reference_levels), dropped=ch.dropped)


def system_basis(system, declaration_id: str = "R", *, estimator: str = "NAS",
                 tau_c: Optional[float] = None, include_endogenous: bool = False,
                 recorded: Optional[RecordedInputs] = None) -> InputBasis:
    """Record, declare and build the basis of one system in one call
    (``tau_c`` from the generator constants unless given)."""
    rec = recorded if recorded is not None else record_inputs(system)
    dec = declare(rec, declaration_id, include_endogenous=include_endogenous)
    if tau_c is None:
        tau_c = coupling_timescale(system).tau_c_sec
    return input_basis(dec, tau_c=tau_c, estimator=estimator)


__all__ = [
    "BASIS_TAU_MULTIPLIERS",
    "CODING_EVENT",
    "CODING_STATE",
    "COMMON_DRIVER",
    "CONTEXT_CUE",
    "CUE_COLUMNS",
    "CUE_JITTER_HALF_WIDTH_SEC",
    "CUE_JITTER_STREAM",
    "CouplingTimescale",
    "DECLARATIONS",
    "DECLARATION_IDS",
    "DECLARED_EVENT_COLUMNS",
    "DECLARED_INPUTS_VERSION",
    "DeclarationSpec",
    "DeclaredInputs",
    "DeclaredInputsError",
    "ENDOGENOUS_EVENT_TYPES",
    "ESTIMATORS",
    "EXOGENOUS_INPUTS",
    "EXOGENOUS_TASK_EVENT_TYPES",
    "HELD_OUT_DECLARATIONS",
    "InputBasis",
    "InputChannels",
    "LABEL_ERROR_STREAM",
    "LagPlan",
    "MAX_NAS_LAGS",
    "NULL_ORDER",
    "RECORDABLE_CHANNELS_KEY",
    "RECORDABLE_STATES_KEY",
    "RecordedInputs",
    "SLOW_PHASE",
    "SLOW_PHASE_CHANNELS",
    "TAU_C_DEFAULT_SEC",
    "TAU_C_SENSITIVITY_SEC",
    "basis_lags",
    "basis_taus",
    "coupling_timescale",
    "declaration_protocol_entry",
    "declaration_spec",
    "declare",
    "declare_all",
    "exogenous_event_types",
    "iim_lag_set",
    "input_basis",
    "input_channels",
    "l_max",
    "lag_plan",
    "lag_set",
    "level_label",
    "lowpass",
    "nas_lag_set",
    "record_inputs",
    "sampling_resolved",
    "system_basis",
]
