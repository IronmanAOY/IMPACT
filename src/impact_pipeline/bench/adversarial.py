"""
MPC-Bench adversarial systems: constructions designed to fool one estimator
at a time.

Each construction lacks at least one principle's mechanism by construction.
The oracle labels it with its intended mechanism pattern (``PRINCIPLES``
order) in ``intended_bits``, the estimator it is designed to fool and the
expected verdict ``EXCLUDED``; the catalogue (``systems.yaml``,
:mod:`impact_pipeline.bench.catalogue`) describes each one and may label the
catalogue entry differently (``adversary_intended_pattern``).

=========================  ===================================================
kind                       construction (designed to fool)
=========================  ===================================================
common_driver              a hidden bursty driver reaches every node, the
                           declared hub first (shorter delay): global
                           co-activation and hub-leads-periphery lagged
                           dependence without any receive-and-return edge
                           (NAS: pseudo-broadcast)
reflex_arc                 feedforward stimulus -> response mapping with a
                           fixed reflex gain, no plasticity, loops, workspace,
                           repertoire or efference copy (RAM responsiveness
                           term, evoked-response markers: responsive but not
                           adaptive)
random_label_self_other    agent without efference copy whose self/other
                           labels are dealt at random within each yoked,
                           stimulus-identical, phase-matched pair (SRPI null:
                           separability from order or action confounds)
scrambled_feedback         plastic agent whose feedback is +-1 with
                           probability 1/2 whatever the choice (RAM null:
                           feedback-locked updates without goal-directed
                           adaptation)
=========================  ===================================================

Staggered hidden driver (``ADV_NAS_staggered_*``)
-------------------------------------------------
No edge between any units. One hidden switching driver ``d(t)`` takes one of
``n_levels = 6`` levels, each held for a dwell ~ U(1, 3) s (a new level is
drawn uniformly, repeats allowed). Every level carries one random pattern per
module (entries ~ N(0, 1), one value per unit), and the module's units
receive the pattern of the active level, ``u_i(t) = P[d(t), i]``, low-pass
filtered with the module's time constant ``tau_m``:

    y_i(t) = a_m y_i(t - 1) + (1 - a_m) u_i(t),   a_m = exp(-dt / tau_m),

started at ``y_i(0) = u_i(0)``. The recorded unit is ``x_i = f(y_i)`` plus
stationary AR(1) noise (coefficient 0.7, SD 0.5); ``f`` is the identity, or
``tanh(2 y)`` for the saturating variant. Time constants by position
relative to the declared hub W (the module order is the family-A
topological order, so upstream modules come before W):

============  ========  =====  ==========  =========  ========================
variant       upstream  hub    downstream  transform  system
============  ========  =====  ==========  =========  ========================
hierarchical  0.05 s    0.3 s  0.8 s       linear     ADV_NAS_staggered_driver
uniform       0.3 s     0.3 s  0.3 s       linear     ADV_NAS_staggered_driver
reversed      0.8 s     0.3 s  0.05 s      linear     ADV_NAS_staggered_driver
tau10         0.5 s     3 s    8 s         linear     ADV_NAS_staggered_tau10
saturating    0.05 s    0.3 s  0.8 s       tanh(2 y)  ADV_NAS_staggered_sat
============  ========  =====  ==========  =========  ========================

``tau10`` and ``saturating`` are held-out conditions. In the hierarchical
variant the upstream modules lead the hub and the hub leads the downstream
modules, so both transfer directions are inflated without any
receive-and-return edge (a shared input plus a hierarchy of intrinsic time
constants; Murray et al. 2014, doi:10.1038/nn.3862).

Every draw comes from the reserved stream ``SeedSequence([seed, 43])``
(:func:`impact_pipeline.bench.seeds.stream_seed_sequence`), split into the
driver schedule, the patterns and the noise, so the variants of one seed
share driver, patterns and noise (common random numbers) and differ only in
their time constants or transform. The time base, the module layout, the
declared meta and the events table are those of the nominal family-A run of
the same seed and configuration (as for the common driver), so event-locked
estimators are defined; the events do not act on the system.

The oracle holds the driver level per sample as ``context_state`` (what an
experimenter logs: the recording device exports it as ``context_cue`` under
the complete declaration R, exactly like the family-A context), the switch
times and levels, the patterns, the filtered drive ``drive_filtered`` (``y``)
and the noise-free signal ``hidden_signal`` (``f(y)``), and the module time
constants. There is no slow rhythm.

This module must not import ``impact_pipeline.mpc_metrics``.
"""

from __future__ import annotations

import math
from dataclasses import replace as dc_replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter

from impact_pipeline.bench.generators import (
    NOMINAL_KNOBS,
    OFF_KNOBS,
    PRINCIPLES,
    AgentConfig,
    BenchSystem,
    _like_meta,
    _samples,
    _streams,
    config_from_dict,
    simulate_family_a,
)
from impact_pipeline.bench.seeds import stream_seed_sequence

ADVERSARIAL_VERSION = "mpc-bench-adversarial/1.0.0"
ADVERSARIAL_KINDS = (
    "common_driver",
    "reflex_arc",
    "random_label_self_other",
    "scrambled_feedback",
)
EXPECTED_VERDICT = "EXCLUDED"

# The oracle labels of each construction (module docstring).
ADVERSARIAL_CATALOGUE = {
    "common_driver": {"designed_to_fool": "NAS", "intended_pattern": [0, 0, 0, 0, 0]},
    "reflex_arc": {"designed_to_fool": "RAM", "intended_pattern": [0, 0, 0, 0, 0]},
    "random_label_self_other": {"designed_to_fool": "SRPI",
                                "intended_pattern": [1, 1, 1, 1, 0]},
    "scrambled_feedback": {"designed_to_fool": "RAM",
                           "intended_pattern": [0, 1, 1, 1, 1]},
}


# ---------------------------------------------------------------------------
# Common-driver pseudo-broadcast
# ---------------------------------------------------------------------------


def common_driver_system(
    like: Optional[BenchSystem] = None,
    seed: int = 0,
    driver_tau_sec: float = 1.0,
    burst_threshold: float = 1.0,
    hub_lag_sec: float = 0.0,
    periphery_lag_sec: tuple = (0.1, 0.25),
    noise_ar: float = 0.7,
    noise_sd: float = 0.5,
    config: Optional[AgentConfig] = None,
) -> BenchSystem:
    """
    Pseudo-broadcast: a hidden driver ``d`` (Ornstein-Uhlenbeck with time
    constant ``driver_tau_sec``, passed through a soft threshold so that it
    produces all-or-none global bursts) reaches every node with a random
    positive gain, the declared workspace hub with delay ``hub_lag_sec`` and
    every periphery node with a delay ~ U(``periphery_lag_sec``); each node
    adds independent AR(1) noise. There is no edge between nodes, so nothing
    is received by the hub or returned by it.
    """
    like = like if like is not None else simulate_family_a(None, config, seed)
    rng = _streams(seed, 8)[6]
    n, T = like.ts.shape
    dt = like.dt
    a = float(np.exp(-dt / float(driver_tau_sec)))
    xi = rng.standard_normal(T)
    ou = np.zeros(T)
    for t in range(1, T):
        ou[t] = a * ou[t - 1] + np.sqrt(1.0 - a * a) * xi[t]
    driver = 1.0 / (1.0 + np.exp(-6.0 * (ou - float(burst_threshold))))
    hub = np.asarray(like.meta.get("workspace_nodes") or [], dtype=int)
    lags = np.round(
        rng.uniform(periphery_lag_sec[0], periphery_lag_sec[1], size=n) / dt
    ).astype(int)
    lags[hub] = int(round(float(hub_lag_sec) / dt))
    gains = rng.uniform(0.8, 1.2, size=n)
    eps = rng.standard_normal((n, T))
    noise = np.zeros((n, T))
    for t in range(1, T):
        noise[:, t] = noise_ar * noise[:, t - 1] + np.sqrt(1 - noise_ar**2) * eps[:, t]
    ts = np.zeros((n, T))
    for i in range(n):
        shifted = np.r_[np.zeros(lags[i]), driver[: T - lags[i]]]
        ts[i] = gains[i] * shifted + float(noise_sd) * noise[i]
    extra = {
        "adversarial": "common_driver",
        "adversarial_version": ADVERSARIAL_VERSION,
        "driver_tau_sec": float(driver_tau_sec),
        "burst_threshold": float(burst_threshold),
    }
    meta = _like_meta(like, "adversarial_common_driver", extra)
    oracle = _adversarial_oracle(
        "common_driver",
        seed,
        {
            "driver": driver,
            "driver_lags_samples": lags,
            "driver_gains": gains,
            "unit_adjacency": np.zeros((n, n)),
            "burst_fraction": float(np.mean(driver > 0.5)),
        },
    )
    return BenchSystem(ts=ts, events=like.events.copy(), meta=meta, oracle=oracle)


# ---------------------------------------------------------------------------
# Reflex arc, random self/other labels, scrambled feedback
# ---------------------------------------------------------------------------


def reflex_arc_system(
    config: Optional[AgentConfig] = None, seed: int = 0, reflex_gain: float = 3.0
) -> BenchSystem:
    """Family-A agent with every switch off and a fixed reflex mapping."""
    cfg = config if config is not None else AgentConfig()
    cfg = dc_replace(cfg, reflex_gain=float(reflex_gain))
    s = simulate_family_a(OFF_KNOBS, cfg, seed)
    return _relabel(s, "reflex_arc", seed, {"reflex_gain": float(reflex_gain)})


def random_label_self_other_system(
    config: Optional[AgentConfig] = None, seed: int = 0
) -> BenchSystem:
    """
    Agent without efference copy whose self/other labels are dealt by a fair
    coin within each yoked pair (the ``yoked_to`` / ``event_id`` columns are
    rewritten consistently, so the SRPI-agency contract still holds).
    """
    s = simulate_family_a(NOMINAL_KNOBS.replace(e=0.0), config, seed)
    rng = _streams(seed, 8)[6]
    ev = s.events.copy()
    swapped = []
    selfs = ev.index[ev.trial_type == "self_caused"]
    by_id = {ev.at[i, "event_id"]: i for i in selfs}
    for j in ev.index[ev.trial_type == "other_caused"]:
        i = by_id[ev.at[j, "yoked_to"]]
        swap = bool(rng.random() < 0.5)
        swapped.append(swap)
        if not swap:
            continue
        sid, oid = ev.at[i, "event_id"], ev.at[j, "event_id"]
        ev.at[i, "trial_type"], ev.at[j, "trial_type"] = "other_caused", "self_caused"
        ev.at[i, "event_id"], ev.at[j, "event_id"] = oid, sid
        ev.at[j, "yoked_to"] = None
        ev.at[i, "yoked_to"] = sid
    s.events = ev.reset_index(drop=True)
    return _relabel(
        s,
        "random_label_self_other",
        seed,
        {"labels_swapped": np.asarray(swapped, dtype=bool)},
    )


def scrambled_feedback_system(
    config: Optional[AgentConfig] = None, seed: int = 0
) -> BenchSystem:
    """Plastic agent whose feedback carries no action-outcome contingency."""
    cfg = config if config is not None else AgentConfig()
    cfg = dc_replace(cfg, feedback_mode="scrambled")
    s = simulate_family_a(NOMINAL_KNOBS, cfg, seed)
    return _relabel(s, "scrambled_feedback", seed, {})


# ---------------------------------------------------------------------------
# Shared helpers and dispatcher
# ---------------------------------------------------------------------------


def _adversarial_oracle(kind: str, seed: int, extra: dict) -> dict:
    entry = ADVERSARIAL_CATALOGUE[kind]
    out = {
        "family": f"adversarial_{kind}",
        "adversarial": kind,
        "seed": int(seed),
        "intended_bits": list(entry["intended_pattern"]),
        "mechanisms": {
            p: bool(b) for p, b in zip(PRINCIPLES, entry["intended_pattern"])
        },
        "designed_to_fool": entry["designed_to_fool"],
        "expected_verdict": EXPECTED_VERDICT,
    }
    out.update(extra)
    return out


def _relabel(s: BenchSystem, kind: str, seed: int, extra: dict) -> BenchSystem:
    """Adversarial labels on an agent-based system (its oracle is kept)."""
    s.meta = dict(s.meta)
    s.meta.update(
        {
            "family": f"adversarial_{kind}",
            "adversarial": kind,
            "adversarial_version": ADVERSARIAL_VERSION,
        }
    )
    agent_bits = s.oracle.get("intended_bits")
    s.oracle.update(_adversarial_oracle(kind, seed, extra))
    s.oracle["agent_knob_bits"] = agent_bits
    return s


def make_adversarial(kind: str, config=None, seed: int = 0, **kwargs) -> BenchSystem:
    """Build one adversarial system (``config``: AgentConfig or dict)."""
    if kind not in ADVERSARIAL_KINDS:
        raise ValueError(f"kind must be one of {ADVERSARIAL_KINDS}, got {kind!r}")
    cfg = (
        config
        if isinstance(config, AgentConfig) or config is None
        else (config_from_dict(config))
    )
    if kind == "common_driver":
        return common_driver_system(seed=seed, config=cfg, **kwargs)
    if kind == "reflex_arc":
        return reflex_arc_system(cfg, seed, **kwargs)
    if kind == "random_label_self_other":
        return random_label_self_other_system(cfg, seed, **kwargs)
    return scrambled_feedback_system(cfg, seed, **kwargs)


# ---------------------------------------------------------------------------
# Staggered hidden driver
# ---------------------------------------------------------------------------

STAGGERED_GENERATOR = "staggered_driver"
STAGGERED_N_LEVELS = 6
STAGGERED_DWELL_SEC = (1.0, 3.0)
STAGGERED_NOISE_AR = 0.7
STAGGERED_NOISE_SD = 0.5
STAGGERED_SATURATION_GAIN = 2.0
# variant -> time constants (upstream, hub, downstream) in seconds, the
# transform after the low-pass, and whether the condition is held out.
STAGGERED_VARIANTS = {
    "hierarchical": {"tau_sec": (0.05, 0.3, 0.8), "transform": "linear",
                     "held_out": False},
    "uniform": {"tau_sec": (0.3, 0.3, 0.3), "transform": "linear",
                "held_out": False},
    "reversed": {"tau_sec": (0.8, 0.3, 0.05), "transform": "linear",
                 "held_out": False},
    "tau10": {"tau_sec": (0.5, 3.0, 8.0), "transform": "linear",
              "held_out": True},
    "saturating": {"tau_sec": (0.05, 0.3, 0.8), "transform": "tanh",
                   "held_out": True},
}
STAGGERED_SYSTEM_VARIANTS = {
    "ADV_NAS_staggered_driver": ("hierarchical", "uniform", "reversed"),
    "ADV_NAS_staggered_tau10": ("tau10",),
    "ADV_NAS_staggered_sat": ("saturating",),
}
STAGGERED_INTENDED_PATTERN = (0, None, 0, 0, 0)
STAGGERED_EXPECTED_VERDICT = "EXCLUDED"


def staggered_module_taus(variant: str, module_order: Sequence[str],
                          hub: str = "W") -> Dict[str, float]:
    """Module time constants (s) of a staggered-driver variant: by position
    relative to the hub in the topological module order."""
    if variant not in STAGGERED_VARIANTS:
        raise ValueError(
            f"variant must be one of {tuple(STAGGERED_VARIANTS)}, got {variant!r}")
    order = list(module_order)
    if hub not in order:
        raise ValueError(f"hub {hub!r} is not in the module order {order}")
    up, mid, down = STAGGERED_VARIANTS[variant]["tau_sec"]
    k = order.index(hub)
    return {
        name: float(up if i < k else mid if i == k else down)
        for i, name in enumerate(order)
    }


def _switching_levels(rng: np.random.Generator, n_time: int, dt: float,
                      n_levels: int, dwell_sec: Tuple[float, float]):
    """Switch indices and levels: each dwell ~ U(dwell_sec) (at least one
    sample), then a level drawn uniformly from ``n_levels``."""
    lo, hi = (float(v) for v in dwell_sec)
    switch_idx, levels = [], []
    t = 0
    while t < n_time:
        dur = max(1, _samples(rng.uniform(lo, hi), dt))
        switch_idx.append(t)
        levels.append(int(rng.integers(0, int(n_levels))))
        t += dur
    return np.asarray(switch_idx, dtype=int), np.asarray(levels, dtype=int)


def _levels_per_sample(switch_idx: np.ndarray, levels: np.ndarray,
                       n_time: int) -> np.ndarray:
    ends = np.r_[switch_idx[1:], n_time]
    return np.repeat(levels, ends - switch_idx).astype(np.int64)[:n_time]


def _unit_patterns(patterns: np.ndarray, modules: Mapping[str, Sequence[int]],
                   module_order: Sequence[str], n_nodes: int) -> np.ndarray:
    """``(n_levels, n_nodes)`` drive pattern of every unit."""
    out = np.zeros((patterns.shape[0], int(n_nodes)))
    for k, name in enumerate(module_order):
        idx = np.asarray(modules[name], dtype=int)
        out[:, idx] = patterns[:, k, : idx.size]
    return out


def staggered_drive_input(system: BenchSystem) -> np.ndarray:
    """The unfiltered drive ``u`` (units x time) of a staggered-driver system,
    rebuilt from its oracle (patterns, driver level per sample and gain)."""
    o = system.oracle
    if o.get("adversarial") != STAGGERED_GENERATOR:
        raise ValueError("not a staggered-driver system")
    unit_pat = np.asarray(o["driver_unit_patterns"], dtype=float)
    level = np.asarray(o["context_state"], dtype=int)
    return float(o["drive_gain"]) * unit_pat[level].T


def staggered_driver_system(
    variant: str = "hierarchical",
    seed: int = 0,
    config: Optional[AgentConfig] = None,
    like: Optional[BenchSystem] = None,
    drive_gain: float = 1.0,
    n_levels: int = STAGGERED_N_LEVELS,
    dwell_sec: Tuple[float, float] = STAGGERED_DWELL_SEC,
    noise_ar: float = STAGGERED_NOISE_AR,
    noise_sd: float = STAGGERED_NOISE_SD,
) -> BenchSystem:
    """
    Staggered hidden-driver adversary (module docstring). ``like`` is the
    template (default: the nominal family-A run of ``seed`` and ``config``):
    time base, module layout, declared meta and events.
    """
    if variant not in STAGGERED_VARIANTS:
        raise ValueError(
            f"variant must be one of {tuple(STAGGERED_VARIANTS)}, got {variant!r}")
    if int(n_levels) < 2:
        raise ValueError("n_levels must be >= 2")
    if not (math.isfinite(drive_gain) and drive_gain > 0):
        raise ValueError("drive_gain must be a finite value > 0")
    if not -1.0 < float(noise_ar) < 1.0 or not float(noise_sd) >= 0:
        raise ValueError("noise_ar must be in (-1, 1) and noise_sd >= 0")
    lo, hi = (float(v) for v in dwell_sec)
    if not 0 < lo <= hi:
        raise ValueError("dwell_sec must satisfy 0 < low <= high")
    spec = STAGGERED_VARIANTS[variant]
    like = like if like is not None else simulate_family_a(None, config, seed)
    n, T = like.ts.shape
    dt = float(like.dt)
    order = list(like.meta["module_order"])
    modules = {k: [int(i) for i in v] for k, v in like.meta["modules"].items()}
    m_max = max(len(v) for v in modules.values())
    taus = staggered_module_taus(variant, order)

    sched_ss, pattern_ss, noise_ss = stream_seed_sequence(
        seed, "staggered_driver").spawn(3)
    sched_rng = np.random.default_rng(sched_ss)
    pattern_rng = np.random.default_rng(pattern_ss)
    noise_rng = np.random.default_rng(noise_ss)

    switch_idx, levels = _switching_levels(sched_rng, T, dt, n_levels, (lo, hi))
    level = _levels_per_sample(switch_idx, levels, T)
    patterns = pattern_rng.standard_normal((int(n_levels), len(order), m_max))
    unit_pat = _unit_patterns(patterns, modules, order, n)
    u = float(drive_gain) * unit_pat[level].T

    y = np.zeros((n, T))
    for name in order:
        idx = np.asarray(modules[name], dtype=int)
        a = float(np.exp(-dt / taus[name]))
        y[idx], _ = lfilter([1.0 - a], [1.0, -a], u[idx], axis=1,
                            zi=a * u[idx, :1])
    if spec["transform"] == "tanh":
        x = np.tanh(STAGGERED_SATURATION_GAIN * y)
    else:
        x = y
    eps = noise_rng.standard_normal((n, T + 1))
    scale = math.sqrt(1.0 - float(noise_ar) ** 2)
    noise, _ = lfilter([scale], [1.0, -float(noise_ar)], eps[:, 1:], axis=1,
                       zi=float(noise_ar) * eps[:, :1])
    ts = x + float(noise_sd) * noise

    system_id = next(
        sid for sid, vs in STAGGERED_SYSTEM_VARIANTS.items() if variant in vs)
    extra = {
        "adversarial": STAGGERED_GENERATOR,
        "adversarial_version": ADVERSARIAL_VERSION,
        "staggered_variant": variant,
        "system_id": system_id,
    }
    meta = _like_meta(like, f"adversarial_{STAGGERED_GENERATOR}", extra)
    pattern = list(STAGGERED_INTENDED_PATTERN)
    oracle = {
        "family": f"adversarial_{STAGGERED_GENERATOR}",
        "adversarial": STAGGERED_GENERATOR,
        "seed": int(seed),
        "intended_bits": pattern,
        "mechanisms": {
            p: (None if b is None else bool(b)) for p, b in zip(PRINCIPLES, pattern)
        },
        "designed_to_fool": "NAS",
        "expected_verdict": STAGGERED_EXPECTED_VERDICT,
        "staggered_variant": variant,
        "held_out": bool(spec["held_out"]),
        "module_tau_sec": dict(taus),
        "transform": spec["transform"],
        "saturation_gain": (
            STAGGERED_SATURATION_GAIN if spec["transform"] == "tanh" else None),
        "drive_gain": float(drive_gain),
        "noise_ar": float(noise_ar),
        "noise_sd": float(noise_sd),
        "n_levels": int(n_levels),
        "dwell_sec": [lo, hi],
        "context_state": level,
        "driver_switch_idx": switch_idx,
        "driver_levels": levels,
        "driver_patterns": patterns,
        "driver_unit_patterns": unit_pat,
        "drive_filtered": y,
        "hidden_signal": x,
        "unit_adjacency": np.zeros((n, n)),
        "module_adjacency": np.zeros((len(order), len(order))),
        "module_graph_acyclic": True,
        "template_seed": like.meta.get("seed"),
    }
    return BenchSystem(ts=ts, events=like.events.copy(), meta=meta, oracle=oracle)


__all__ = [
    "ADVERSARIAL_CATALOGUE",
    "ADVERSARIAL_KINDS",
    "ADVERSARIAL_VERSION",
    "EXPECTED_VERDICT",
    "STAGGERED_GENERATOR",
    "STAGGERED_SYSTEM_VARIANTS",
    "STAGGERED_VARIANTS",
    "common_driver_system",
    "make_adversarial",
    "random_label_self_other_system",
    "reflex_arc_system",
    "scrambled_feedback_system",
    "staggered_drive_input",
    "staggered_driver_system",
    "staggered_module_taus",
]
