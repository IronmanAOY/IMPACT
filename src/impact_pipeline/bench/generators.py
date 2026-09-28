"""
MPC-Bench generators: white-box systems with switchable mechanisms.

Families
--------
A  Modular stochastic rate-network agent (``tanh`` units, Euler-Maruyama with
   step ``dt``). Five mechanism switches, each off (0) or on with a
   continuous dose:

   ======  ==========================================================  ======
   knob    mechanism                                                   target
   ======  ==========================================================  ======
   eta     feedback-driven plasticity of the stimulus-response gain    RAM
   K       repertoire: number of context attractor patterns (K=1 off)  PDI
   g_b     workspace receive-and-return broadcast gain with ignition;  NAS
           ``ff_only`` removes the receive path (broadcast only)
   c_int   recurrent (backward) inter-module loops; 0 = feedforward     IIM
   e       efference-copy tag and attenuation of reafferent input      SRPI
   ======  ==========================================================  ======

   Task: a reversal two-armed bandit with goal cues and +-1 feedback, then a
   reafference phase in which the agent's own actions cause sensory events
   (``self_caused``) and every such event is replayed later without an action
   (``other_caused``), stimulus-identical and phase-matched to an endogenous
   slow rhythm.
B  Binary / kinetic-Ising networks with exact transition probability matrices
   (independent, ring, all-to-all, feedforward star, XOR loop, hidden common
   driver) and a trajectory sampler (IIM ground truth).
C  Stuart-Landau (Hopf) oscillator network with the family-A switch semantics
   and task (held-out family; run only after the code freeze). Design 2
   (generator 1.1.0): resonant input drives, coupling near the network's Hopf
   bifurcation and bifurcation-parameter ignition, validated by the
   preregistered oracle manipulation checks of
   :mod:`impact_pipeline.bench.manipulation`.

Outputs of families A/C (:class:`BenchSystem`): ``ts`` (nodes x time),
``events`` (BIDS events table), ``meta`` (declared information an analyst may
use: knobs, seed, dt, workspace nodes, bearer nodes, IIM macro-node grain) and
``oracle`` (hidden ground truth; written to a separate file and never read by
estimators).

Mechanism semantics are defined by the generator, never by metric values, and
this module must not import ``impact_pipeline.mpc_metrics``.

Random streams: one ``SeedSequence(seed)`` is split into independent streams
for network structure, task schedule, process noise and choices, and every
stream draws the same numbers whatever the knob values. Two systems with the
same seed and different knobs therefore share their structural and noise
realisation (common random numbers), so a knob contrast is paired.

States of family-B networks use the big-endian convention of
``mpc_metrics._iim_*`` kernels: node 0 is the most significant digit of the
state index.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

GENERATOR_VERSION = "mpc-bench-generators/1.1.0"
PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")
SWITCH_FOR_PRINCIPLE = {
    "RAM": "eta",
    "PDI": "K",
    "NAS": "g_b",
    "IIM": "c_int",
    "SRPI": "e",
}
EVENT_COLUMNS = (
    "onset",
    "duration",
    "trial_type",
    "value",
    "choice",
    "reward",
    "yoked_to",
    "phase_bin",
    "impact_channel",
    "trial",
    "event_id",
)
TRIAL_TYPES = (
    "goal_cue",
    "stimulus",
    "response",
    "feedback",
    "action",
    "self_caused",
    "other_caused",
)
IMPACT_CHANNEL_TASK = "behavioural_feedback"
IMPACT_CHANNEL_AGENCY = "agency"
DYNAMICS = ("rate", "stuart_landau")
# Substrate labels of the applicability registry (spec v2, V2-5).
SUBSTRATE_OF_DYNAMICS = {"rate": "synthetic_rate", "stuart_landau": "stuart_landau"}
FAMILY_C_DESIGN = 2

# Module layouts of family A/C in topological (feedforward) order. W is the
# workspace hub; the other modules are the periphery.
MODULE_LAYOUTS = {
    4: ("S", "W", "M", "V"),
    5: ("S", "C", "W", "M", "V"),
    6: ("S", "C", "A", "W", "M", "V"),
}
MODULE_ROLES = {
    "S": "sensory",
    "C": "context",
    "A": "association",
    "W": "workspace_hub",
    "M": "motor",
    "V": "evaluative",
}
_N_PATTERN_BANK = 32
FEEDBACK_MODES = ("contingent", "scrambled")


# ---------------------------------------------------------------------------
# Knobs and configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Knobs:
    """
    Mechanism switches of families A and C (and of the patchwork).

    A mechanism is off when its dose is 0 (``K == 1`` for the repertoire).
    ``k_gain`` scales the context-pattern drive (dose of the repertoire
    mechanism besides the pattern count ``K``). ``ff_only`` keeps the
    workspace broadcast but removes its receive path.
    """

    eta: float = 0.3
    K: int = 6
    k_gain: float = 1.0
    g_b: float = 1.0
    ff_only: bool = False
    c_int: float = 0.6
    e: float = 1.0

    def __post_init__(self):
        for name in ("eta", "k_gain", "g_b", "c_int", "e"):
            val = float(getattr(self, name))
            if not (math.isfinite(val) and val >= 0):
                raise ValueError(f"{name} must be a finite value >= 0")
        if int(self.K) != self.K or int(self.K) < 1 or int(self.K) > _N_PATTERN_BANK:
            raise ValueError(f"K must be an integer in [1, {_N_PATTERN_BANK}]")

    def bits(self) -> Tuple[int, int, int, int, int]:
        """Mechanism-present pattern in ``PRINCIPLES`` order (1 = present)."""
        return (
            int(self.eta > 0),
            int(self.K > 1 and self.k_gain > 0),
            int(self.g_b > 0 and not self.ff_only),
            int(self.c_int > 0),
            int(self.e > 0),
        )

    def to_dict(self) -> dict:
        out = asdict(self)
        out["K"] = int(out["K"])
        out["ff_only"] = bool(out["ff_only"])
        return out

    def replace(self, **kwargs) -> "Knobs":
        return replace(self, **kwargs)

    @classmethod
    def from_bits(cls, bits: Sequence[int], nominal: "Knobs" = None) -> "Knobs":
        """
        Knobs realising a 5-bit mechanism pattern (``PRINCIPLES`` order): a 1
        keeps the nominal dose, a 0 switches the mechanism off (NAS off means
        ``g_b = 0``; ``ff_only`` is a separate witness variant).
        """
        bits = tuple(int(b) for b in bits)
        if len(bits) != 5 or any(b not in (0, 1) for b in bits):
            raise ValueError(f"bits must be five 0/1 values, got {bits!r}")
        base = nominal if nominal is not None else NOMINAL_KNOBS
        kw = {}
        if not bits[0]:
            kw["eta"] = 0.0
        if not bits[1]:
            kw["K"] = 1
        if not bits[2]:
            kw["g_b"] = 0.0
            kw["ff_only"] = False
        if not bits[3]:
            kw["c_int"] = 0.0
        if not bits[4]:
            kw["e"] = 0.0
        return base.replace(**kw)


NOMINAL_KNOBS = Knobs()
OFF_KNOBS = Knobs(eta=0.0, K=1, k_gain=1.0, g_b=0.0, ff_only=False, c_int=0.0, e=0.0)


def knobs_from_dict(d: Optional[dict], base: Knobs = None) -> Knobs:
    """Nominal knobs updated by a (possibly partial) dict."""
    base = base if base is not None else NOMINAL_KNOBS
    if not d:
        return base
    unknown = sorted(set(d) - set(base.to_dict()))
    if unknown:
        raise ValueError(f"Unknown knob(s): {unknown}")
    kw = dict(d)
    if "K" in kw:
        kw["K"] = int(kw["K"])
    if "ff_only" in kw:
        kw["ff_only"] = bool(kw["ff_only"])
    return base.replace(**kw)


@dataclass(frozen=True)
class AgentConfig:
    """
    Structure, dynamics and task constants of families A and C.

    All times are in seconds and are rounded to the sample grid ``dt``. The
    network constants are declared in ``meta`` and never change with knobs.
    """

    n_modules: int = 5
    units_per_module: int = 6
    dt: float = 0.05
    tau: float = 0.1
    sigma: float = 0.8
    # Couplings (per-module block totals; blocks are divided by the size).
    spectral_radius: float = 0.85
    c_loc: float = 0.5
    loc_jitter: float = 0.3
    w_ff: float = 1.0
    w_hub: float = 1.0
    w_ign: float = 1.5
    ignition_slope: float = 10.0
    hub_bias: float = 0.0
    w_adapt: float = 1.5
    tau_adapt: float = 1.0
    ignition_threshold: float = 0.3
    # Task drives.
    a_goal: float = 0.8
    a_stim: float = 1.2
    g_sm: float = 0.8
    a_motor: float = 1.0
    a_fb: float = 1.2
    a_tag: float = 1.0
    efference_attenuation: float = 0.5
    ctx_gain: float = 0.5
    ctx_dwell: Tuple[float, float] = (1.0, 3.0)
    slow_freq_hz: float = 0.25
    slow_gain: float = 0.15
    slow_stim_mod: float = 0.3
    n_phase_bins: int = 4
    # Bandit. ``feedback_mode='scrambled'`` delivers +-1 feedback with
    # probability 1/2 whatever the choice (no action-outcome contingency;
    # adversarial RAM null). ``reflex_gain`` adds a fixed stimulus-specific
    # drive (stimulus id a -> action a) to the stimulus-response pathway
    # (reflex arc; 0 in families A/C).
    n_trials: int = 80
    p_reward: float = 0.8
    feedback_mode: str = "contingent"
    reflex_gain: float = 0.0
    reversal_block: Tuple[int, int] = (15, 25)
    choice_beta: float = 5.0
    trial_sec: float = 3.0
    iti_jitter_sec: float = 1.0
    goal_to_stim_sec: float = 0.6
    stim_to_response_sec: float = 0.6
    response_to_feedback_sec: float = 0.6
    readout_delay_sec: float = 0.25
    # Reafference.
    n_reafference_pairs: int = 30
    reafference_delay_sec: float = 0.2
    reafference_gap_sec: float = 2.0
    # Run layout.
    lead_sec: float = 2.0
    phase_gap_sec: float = 3.0
    tail_sec: float = 3.0
    rest_sec: float = 0.0
    # Stuart-Landau (family C, design 2) constants: rate lam (1/s) of the
    # amplitude dynamics, per-node natural frequency ~ U(sl_freq_hz), coupling
    # kappa on the family-A matrix W, bifurcation parameters (sl_a < 0:
    # damped nodes; the nominal network sits below the Hopf bifurcation of
    # its dominant mode, a + kappa * rho(W) < 0), resonant input gain, hub
    # ignition (bifurcation-parameter increment and adaptation) and noise.
    sl_substeps: int = 5
    sl_freq_hz: Tuple[float, float] = (1.15, 1.25)
    sl_rate: float = 10.0
    sl_coupling: float = 1.1
    sl_input_gain: float = 1.0
    sl_a: float = -1.0
    sl_hub_a: float = -2.0
    sl_ignition_gain: float = 2.5
    sl_adapt_gain: float = 2.0
    sl_ignition_threshold: float = 0.38
    sl_ignition_slope: float = 20.0
    sl_gate_tau: float = 0.2
    sl_sigma: float = 0.15

    def __post_init__(self):
        if int(self.n_modules) not in MODULE_LAYOUTS:
            raise ValueError(f"n_modules must be one of {sorted(MODULE_LAYOUTS)}")
        if not 4 <= int(self.units_per_module) <= 8:
            raise ValueError("units_per_module must be in [4, 8]")
        if not (self.dt > 0 and self.tau > 0):
            raise ValueError("dt and tau must be > 0")
        if self.dt > self.tau:
            # Euler step dt/tau > 1 overshoots the fixed point (oscillating or
            # divergent rate dynamics).
            raise ValueError("dt must be <= tau (Euler step dt/tau <= 1)")
        if int(self.sl_substeps) < 1:
            raise ValueError("sl_substeps must be >= 1")
        if not (self.sl_rate > 0 and self.sl_gate_tau > 0):
            raise ValueError("sl_rate and sl_gate_tau must be > 0")
        if self.sl_rate * self.dt / int(self.sl_substeps) > 0.5:
            # Euler-Maruyama substep of the amplitude dynamics must be small.
            raise ValueError("sl_rate * dt / sl_substeps must be <= 0.5")
        if self.n_trials < 1 or self.n_reafference_pairs < 1:
            raise ValueError("n_trials and n_reafference_pairs must be >= 1")
        if self.n_phase_bins < 1:
            raise ValueError("n_phase_bins must be >= 1")
        if not 0.5 <= self.p_reward <= 1.0:
            raise ValueError("p_reward must be in [0.5, 1]")
        if self.feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"feedback_mode must be one of {FEEDBACK_MODES}")
        if not (math.isfinite(self.reflex_gain) and self.reflex_gain >= 0):
            raise ValueError("reflex_gain must be a finite value >= 0")

    def to_dict(self) -> dict:
        out = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}

    def replace(self, **kwargs) -> "AgentConfig":
        return replace(self, **kwargs)

    @property
    def slow_period_samples(self) -> int:
        """Slow-rhythm period on the sample grid (so replays are phase-exact)."""
        return max(2, int(round(1.0 / (self.slow_freq_hz * self.dt))))


def config_from_dict(d: Optional[dict], base: AgentConfig = None) -> AgentConfig:
    base = base if base is not None else AgentConfig()
    if not d:
        return base
    unknown = sorted(set(d) - set(base.to_dict()))
    if unknown:
        raise ValueError(f"Unknown config field(s): {unknown}")
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
    return base.replace(**kw)


@dataclass
class BenchSystem:
    """One simulated run: node x time series, events, declared meta, hidden oracle."""

    ts: np.ndarray
    events: pd.DataFrame
    meta: dict
    oracle: dict
    rest_ts: Optional[np.ndarray] = None

    @property
    def n_nodes(self) -> int:
        return int(self.ts.shape[0])

    @property
    def n_time(self) -> int:
        return int(self.ts.shape[1])

    @property
    def dt(self) -> float:
        return float(self.meta["dt"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _streams(seed: int, n: int = 6) -> List[np.random.Generator]:
    """Independent, knob-invariant random streams derived from one seed."""
    if int(seed) < 0:
        raise ValueError("seed must be >= 0")
    children = np.random.SeedSequence(int(seed)).spawn(n)
    return [np.random.default_rng(c) for c in children]


def _samples(sec: float, dt: float) -> int:
    return int(round(float(sec) / float(dt)))


def _pos_block(rng: np.random.Generator, m_out: int, m_in: int) -> np.ndarray:
    """Positive random block (mean ~1 per entry) normalised by the input size."""
    return np.abs(rng.normal(1.0, 0.25, size=(m_out, m_in))) / float(m_in)


def is_acyclic(adjacency: np.ndarray) -> bool:
    """True when the directed graph ``adjacency[source, target] != 0`` has no
    cycle other than self-loops (Kahn's algorithm)."""
    a = (np.asarray(adjacency) != 0).astype(int)
    np.fill_diagonal(a, 0)
    indeg = a.sum(axis=0)
    queue = [i for i in range(a.shape[0]) if indeg[i] == 0]
    seen = 0
    while queue:
        i = queue.pop()
        seen += 1
        for j in np.flatnonzero(a[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(int(j))
    return seen == a.shape[0]


def _phase(t_idx, cfg: AgentConfig, phi0: float) -> np.ndarray:
    period = cfg.slow_period_samples
    return np.mod(
        2.0 * np.pi * np.asarray(t_idx, dtype=float) / period + phi0, 2.0 * np.pi
    )


def _phase_bin(t_idx, cfg: AgentConfig, phi0: float) -> np.ndarray:
    ph = _phase(t_idx, cfg, phi0)
    return np.minimum(
        (ph / (2.0 * np.pi / cfg.n_phase_bins)).astype(int), cfg.n_phase_bins - 1
    )


# ---------------------------------------------------------------------------
# Network construction (families A/C)
# ---------------------------------------------------------------------------


@dataclass
class _Net:
    """Unit-level coupling ``W[target, source]`` and task routing of an agent."""

    W: np.ndarray
    bias: np.ndarray
    modules: Dict[str, np.ndarray]
    module_order: Tuple[str, ...]
    module_adjacency: np.ndarray
    routes: Dict[str, object]
    context_modules: List[np.ndarray]
    patterns: Dict[str, np.ndarray]
    extra_meta: dict = field(default_factory=dict)
    extra_oracle: dict = field(default_factory=dict)


def _module_indices(order: Sequence[str], m: int) -> Dict[str, np.ndarray]:
    return {name: np.arange(k * m, (k + 1) * m) for k, name in enumerate(order)}


def draw_patterns(
    rng: np.random.Generator, sizes: Dict[str, int]
) -> Dict[str, np.ndarray]:
    """+-1 input patterns sized to their target unit groups (fixed draw order)."""
    return {
        "stimulus": rng.choice([-1.0, 1.0], size=(2, int(sizes["stim"]))),
        "reafference": rng.choice([-1.0, 1.0], size=(2, int(sizes["reaff"]))),
        "tag": rng.choice([-1.0, 1.0], size=int(sizes["tag"])),
        "goal": rng.choice([-1.0, 1.0], size=(2, int(sizes["goal"]))),
        "feedback": rng.choice([-1.0, 1.0], size=int(sizes["feedback"])),
    }


def normalise_coupling(
    builder_raw, knobs: Knobs, cfg: AgentConfig, rng: np.random.Generator
) -> _Net:
    """
    Build a network with ``builder_raw`` and scale its coupling so that the
    *nominal* network of the same seed has spectral radius
    ``cfg.spectral_radius`` (a knob-invariant factor: switching a mechanism
    off removes coupling instead of renormalising the rest). The raw builder
    must draw the same random numbers whatever the knob values.
    """
    nominal_rng = np.random.Generator(type(rng.bit_generator)())
    nominal_rng.bit_generator.state = rng.bit_generator.state
    nominal = builder_raw(NOMINAL_KNOBS, cfg, nominal_rng)
    radius = float(np.max(np.abs(np.linalg.eigvals(nominal.W))))
    scale = float(cfg.spectral_radius) / radius if radius > 0 else 1.0
    net = builder_raw(knobs, cfg, rng)
    net.W = net.W * scale
    net.module_adjacency = net.module_adjacency * scale
    net.extra_meta = dict(net.extra_meta)
    net.extra_meta["coupling_scale"] = scale
    net.extra_meta["nominal_spectral_radius"] = float(cfg.spectral_radius)
    net.extra_meta["spectral_radius"] = float(np.max(np.abs(np.linalg.eigvals(net.W))))
    return net


def _build_family_a_raw(
    knobs: Knobs, cfg: AgentConfig, rng: np.random.Generator
) -> _Net:
    order = MODULE_LAYOUTS[int(cfg.n_modules)]
    m = int(cfg.units_per_module)
    mods = _module_indices(order, m)
    n = len(order) * m
    W = np.zeros((n, n))
    bias = np.zeros(n)

    # Within-module local recurrence.
    for name in order:
        idx = mods[name]
        g = rng.normal(size=(m, m))
        g = 0.5 * (g + g.T)
        np.fill_diagonal(g, 0.0)
        blk = (
            cfg.c_loc / m * (np.ones((m, m)) - np.eye(m))
            + cfg.loc_jitter / np.sqrt(m) * g
        )
        W[np.ix_(idx, idx)] = blk

    periph = [x for x in order if x != "W"]
    ff_edges = [(periph[i], periph[i + 1]) for i in range(len(periph) - 1)]
    if ("S", "M") not in ff_edges:
        ff_edges.append(("S", "M"))
    madj = np.zeros((len(order), len(order)))
    pos = {name: k for k, name in enumerate(order)}
    for a, b in ff_edges:
        fwd = _pos_block(rng, m, m)
        bwd = _pos_block(rng, m, m)
        W[np.ix_(mods[b], mods[a])] += cfg.w_ff * fwd
        madj[pos[a], pos[b]] += cfg.w_ff
        if knobs.c_int > 0:
            W[np.ix_(mods[a], mods[b])] += knobs.c_int * bwd
            madj[pos[b], pos[a]] += knobs.c_int

    rho = float(knobs.c_int) / float(NOMINAL_KNOBS.c_int)
    hub = mods["W"]
    for p in periph:
        upstream = pos[p] < pos["W"]
        recv_blk = _pos_block(rng, m, m)
        ret_blk = _pos_block(rng, m, m)
        recv = (
            0.0 if knobs.ff_only else knobs.g_b * cfg.w_hub * (1.0 if upstream else rho)
        )
        ret = knobs.g_b * cfg.w_hub * (rho if upstream else 1.0)
        if recv > 0:
            W[np.ix_(hub, mods[p])] += recv * recv_blk
            madj[pos[p], pos["W"]] += recv
        if ret > 0:
            W[np.ix_(mods[p], hub)] += ret * ret_blk
            madj[pos["W"], pos[p]] += ret
    bias[hub] = cfg.hub_bias

    patterns = draw_patterns(
        rng, {k: m for k in ("stim", "reaff", "tag", "goal", "feedback")}
    )
    ctx_bank = rng.choice([-1.0, 1.0], size=(len(order), _N_PATTERN_BANK, m))
    halves = np.array_split(mods["M"], 2)
    goal_mod = "C" if "C" in mods else "W"
    routes = {
        "stim": mods["S"],
        "goal": mods[goal_mod],
        "action": halves,
        "feedback": mods["V"],
        "reaff": mods["S"],
        "tag": mods["S"],
        "motor_self": halves,
        "hub": hub,
        "slow": np.arange(n),
    }
    return _Net(
        W=W,
        bias=bias,
        modules=mods,
        module_order=tuple(order),
        module_adjacency=madj,
        routes=routes,
        context_modules=[mods[name] for name in order],
        patterns={**patterns, "context_bank": ctx_bank},
    )


def build_family_a_network(
    knobs: Knobs, cfg: AgentConfig, rng: np.random.Generator
) -> _Net:
    """
    Unit-level coupling of the modular agent. All random blocks are drawn in a
    fixed order whatever the knob values (a knob only scales its blocks), and
    the coupling is scaled by the knob-invariant factor of
    :func:`normalise_coupling`.

    Inter-module structure (module order = topological order):
      * forward periphery edges (``w_ff``): consecutive periphery modules and
        S -> M (task pathway);
      * backward periphery edges: the reverse of every forward edge, gain
        ``c_int`` (none when ``c_int == 0``: feedforward only);
      * workspace hub W: receives from every periphery module and returns to
        every periphery module with gain ``g_b * w_hub`` (receive-and-return).
        Ignition is an all-or-none amplification of the hub (gain
        ``g_b * w_ign``) once its mean activity exceeds
        ``ignition_threshold``, terminated by adaptation (``w_adapt``,
        ``tau_adapt``). Hub edges that point backwards in the module order
        (hub -> upstream, downstream -> hub) are scaled by
        ``c_int / c_int_nominal``, so ``c_int = 0`` leaves a feedforward relay
        (upstream -> W -> downstream) and an acyclic module graph.
        ``ff_only`` removes every periphery -> hub edge.
    """
    return normalise_coupling(_build_family_a_raw, knobs, cfg, rng)


# ---------------------------------------------------------------------------
# Task schedule (open-loop parts; knob-invariant)
# ---------------------------------------------------------------------------


def _task_schedule(cfg: AgentConfig, rng: np.random.Generator) -> dict:
    dt = cfg.dt
    lead = _samples(cfg.lead_sec, dt)
    g2s = _samples(cfg.goal_to_stim_sec, dt)
    s2r = _samples(cfg.stim_to_response_sec, dt)
    r2f = _samples(cfg.response_to_feedback_sec, dt)
    trial_len = _samples(cfg.trial_sec, dt)
    jitter = _samples(cfg.iti_jitter_sec, dt)
    n_tr = int(cfg.n_trials)

    goal_ids = rng.integers(0, 2, size=n_tr)
    stim_ids = rng.integers(0, 2, size=n_tr)
    jit = rng.integers(0, jitter + 1, size=n_tr)
    choice_u = rng.random(n_tr)
    reward_u = rng.random(n_tr)
    lo, hi = cfg.reversal_block
    good = np.zeros(n_tr, dtype=int)
    arm = int(rng.integers(0, 2))
    k = 0
    while k < n_tr:
        blen = int(rng.integers(int(lo), int(hi) + 1))
        good[k : k + blen] = arm
        arm = 1 - arm
        k += blen

    goal_idx = np.zeros(n_tr, dtype=int)
    t = lead
    for i in range(n_tr):
        goal_idx[i] = t
        t += trial_len + int(jit[i])
    stim_idx = goal_idx + g2s
    resp_idx = stim_idx + s2r
    fb_idx = resp_idx + r2f

    # Reafference phase.
    period = cfg.slow_period_samples
    delay = _samples(cfg.reafference_delay_sec, dt)
    gap = _samples(cfg.reafference_gap_sec, dt)
    n_pairs = int(cfg.n_reafference_pairs)
    offs = rng.integers(0, period, size=n_pairs)
    mult = rng.integers(1, 3, size=n_pairs)
    reaff_ids = rng.integers(0, 2, size=n_pairs)
    cursor = int(t) + _samples(cfg.phase_gap_sec, dt)
    act_idx = np.zeros(n_pairs, dtype=int)
    self_idx = np.zeros(n_pairs, dtype=int)
    other_idx = np.zeros(n_pairs, dtype=int)
    for i in range(n_pairs):
        act_idx[i] = cursor + int(offs[i])
        self_idx[i] = act_idx[i] + delay
        other_idx[i] = self_idx[i] + int(mult[i]) * period
        cursor = other_idx[i] + gap
    n_time = int(cursor) + _samples(cfg.tail_sec, dt)

    switch_idx, switch_u = _context_switches(cfg, rng, n_time)

    return {
        "n_time": int(n_time),
        "goal_idx": goal_idx,
        "stim_idx": stim_idx,
        "resp_idx": resp_idx,
        "fb_idx": fb_idx,
        "goal_ids": goal_ids,
        "stim_ids": stim_ids,
        "good_arm": good,
        "choice_u": choice_u,
        "reward_u": reward_u,
        "act_idx": act_idx,
        "self_idx": self_idx,
        "other_idx": other_idx,
        "reaff_ids": reaff_ids,
        "reaff_mult": mult,
        "ctx_switch_idx": switch_idx,
        "ctx_switch_u": switch_u,
        "phi0": float(rng.uniform(0.0, 2.0 * np.pi)),
        "bandit_end_idx": int(fb_idx[-1] + _samples(1.0, dt)) if n_tr else lead,
        "reafference_start_idx": int(act_idx[0]) if n_pairs else int(t),
    }


def _context_switches(cfg: AgentConfig, rng: np.random.Generator, n_time: int):
    """Context switch times (dwell ~ U(ctx_dwell)) and uniform draws mapped to
    a pattern index for any K (knob-invariant)."""
    d_lo, d_hi = cfg.ctx_dwell
    switch_idx = [0]
    while switch_idx[-1] < n_time:
        switch_idx.append(
            switch_idx[-1] + max(1, _samples(rng.uniform(d_lo, d_hi), cfg.dt))
        )
    switch_idx = np.asarray(switch_idx[:-1], dtype=int)
    return switch_idx, rng.random(switch_idx.size)


def _context_states(sched: dict, K: int, n_time: int) -> np.ndarray:
    k_at_switch = np.minimum((sched["ctx_switch_u"] * int(K)).astype(int), int(K) - 1)
    state = np.zeros(n_time, dtype=int)
    idx = sched["ctx_switch_idx"]
    for j in range(idx.size):
        end = idx[j + 1] if j + 1 < idx.size else n_time
        state[idx[j] : end] = k_at_switch[j]
    return state


# ---------------------------------------------------------------------------
# Simulation engine (shared by families A, C and the patchwork)
# ---------------------------------------------------------------------------


def _add_pulse(inp, units, start, dur, vec):
    end = min(inp.shape[1], start + dur)
    if start >= end:
        return
    inp[np.ix_(units, np.arange(start, end))] += np.asarray(vec, dtype=float).reshape(
        -1, 1
    )


def _simulate_agent(
    net: _Net,
    knobs: Knobs,
    cfg: AgentConfig,
    sched: dict,
    noise_rng: np.random.Generator,
    dynamics: str,
    struct_rng: np.random.Generator,
    task_inputs: bool = True,
    n_time: Optional[int] = None,
) -> dict:
    """Closed-loop simulation. Returns activity, inputs and oracle channels."""
    if dynamics not in DYNAMICS:
        raise ValueError(f"dynamics must be one of {DYNAMICS}")
    dt = cfg.dt
    n = net.W.shape[0]
    T = int(sched["n_time"] if n_time is None else n_time)
    inp = np.zeros((n, T))
    routes = net.routes
    pat = net.patterns
    phi0 = sched["phi0"]
    t_all = np.arange(T)
    slow = np.sin(_phase(t_all, cfg, phi0))

    # Endogenous inputs: slow rhythm and context patterns (repertoire K).
    inp[routes["slow"], :] += cfg.slow_gain * slow[None, :]
    ctx_state = _context_states(sched, knobs.K, T)
    ctx_amp = cfg.ctx_gain * knobs.k_gain
    if ctx_amp > 0:
        bank = pat["context_bank"]
        for mi, idx in enumerate(net.context_modules):
            inp[idx, :] += ctx_amp * bank[mi][ctx_state].T

    goal_dur = _samples(0.3, dt)
    stim_dur = _samples(0.3, dt)
    map_dur = _samples(cfg.stim_to_response_sec, dt)
    motor_dur = _samples(0.15, dt)
    fb_dur = _samples(0.3, dt)
    reaff_dur = _samples(0.2, dt)
    act_dur = _samples(0.1, dt)
    readout_delay = _samples(cfg.readout_delay_sec, dt)

    n_tr = int(cfg.n_trials) if task_inputs else 0
    q = np.zeros(2)
    q_trace = np.zeros((n_tr, 2))
    choices = np.full(n_tr, -1, dtype=int)
    rewards = np.zeros(n_tr)
    pes = np.zeros(n_tr)
    p_choice1 = np.zeros(n_tr)
    handlers = {}
    reaff_atten = np.ones(int(cfg.n_reafference_pairs))
    tag_trace = np.zeros(T)
    if task_inputs:
        for i in range(n_tr):
            g0 = int(sched["goal_idx"][i])
            s0 = int(sched["stim_idx"][i])
            _add_pulse(
                inp,
                routes["goal"],
                g0,
                goal_dur,
                cfg.a_goal * pat["goal"][sched["goal_ids"][i]],
            )
            stim_vec = (
                cfg.a_stim
                * pat["stimulus"][sched["stim_ids"][i]]
                * (1.0 + cfg.slow_stim_mod * slow[s0])
            )
            _add_pulse(inp, routes["stim"], s0, stim_dur, stim_vec)
            handlers.setdefault(s0, []).append(("map", i))
            handlers.setdefault(int(sched["resp_idx"][i]), []).append(("respond", i))
            handlers.setdefault(int(sched["fb_idx"][i]), []).append(("learn", i))
        for j in range(int(cfg.n_reafference_pairs)):
            a0 = int(sched["act_idx"][j])
            s1 = int(sched["self_idx"][j])
            o1 = int(sched["other_idx"][j])
            rid = int(sched["reaff_ids"][j])
            _add_pulse(
                inp,
                routes["motor_self"][rid],
                a0,
                act_dur,
                cfg.a_motor * np.ones(routes["motor_self"][rid].size),
            )
            base_vec = (
                cfg.a_stim
                * pat["reafference"][rid]
                * (1.0 + cfg.slow_stim_mod * slow[s1])
            )
            att = 1.0 / (1.0 + cfg.efference_attenuation * knobs.e)
            reaff_atten[j] = att
            _add_pulse(inp, routes["reaff"], s1, reaff_dur, att * base_vec)
            if knobs.e > 0:
                _add_pulse(
                    inp, routes["tag"], s1, reaff_dur, knobs.e * cfg.a_tag * pat["tag"]
                )
                tag_trace[s1 : s1 + reaff_dur] = knobs.e * cfg.a_tag
            # Yoked replay: identical stimulus vector (same phase => same
            # slow modulation), no action, no efference copy.
            _add_pulse(inp, routes["reaff"], o1, reaff_dur, base_vec)

    hub = routes["hub"]
    act_units = routes["action"]
    act = np.zeros((n, T))
    h = np.zeros(hub.size)
    ign_trace = np.zeros(T)
    gate_trace = np.zeros(T)
    env = None
    omega = None

    if dynamics == "rate":
        alpha = dt / cfg.tau
        noise_sd = cfg.sigma * np.sqrt(dt)
        x = np.zeros(n)
        for t in range(T):
            if t in handlers:
                _run_handlers(
                    handlers[t],
                    t,
                    act,
                    readout_delay,
                    act_units,
                    inp,
                    q,
                    q_trace,
                    choices,
                    rewards,
                    pes,
                    p_choice1,
                    sched,
                    cfg,
                    knobs,
                    map_dur,
                    motor_dur,
                    fb_dur,
                    routes,
                    pat,
                )
            u = net.W @ x + inp[:, t] + net.bias
            s_ign = 1.0 / (
                1.0
                + np.exp(
                    -cfg.ignition_slope
                    * (float(np.mean(x[hub])) - cfg.ignition_threshold)
                )
            )
            u[hub] += knobs.g_b * (cfg.w_ign * s_ign - cfg.w_adapt * h)
            x = x + alpha * (-x + np.tanh(u)) + noise_sd * noise_rng.standard_normal(n)
            h = h + (dt / cfg.tau_adapt) * (-h + np.maximum(x[hub], 0.0))
            act[:, t] = x
            ign_trace[t] = float(np.mean(x[hub]))
            gate_trace[t] = knobs.g_b * s_ign
        ts = act
    else:
        # Family C (design 2): Stuart-Landau oscillators in rate units,
        #   dz_j = { i w_j z_j + lam [ (a_j(t) - |z_j|^2) z_j + kappa (W z)_j
        #            + I_j(t) e^{i w_j t} ] } dt + lam sigma dB_j,
        # with task and endogenous inputs I_j(t) delivered as resonant drives
        # (at each node's own frequency) and hub ignition acting on the hub's
        # bifurcation parameter. In the frame rotating with each node the
        # envelope obeys the family-A rate equations with a cubic saturation,
        # so every switch acts through the same couplings, while the recorded
        # signal Re z is an oscillation whose amplitude and phase carry them.
        sub = max(1, int(cfg.sl_substeps))
        dti = dt / sub
        lam = float(cfg.sl_rate)
        f_lo, f_hi = cfg.sl_freq_hz
        omega = 2.0 * np.pi * struct_rng.uniform(f_lo, f_hi, size=n)
        a_vec = np.full(n, float(cfg.sl_a))
        a_vec[hub] = float(cfg.sl_hub_a)
        kappa_w = float(cfg.sl_coupling) * net.W
        noise_sd = lam * cfg.sl_sigma * np.sqrt(dti / 2.0)
        z = np.zeros(n, dtype=complex)
        ts = np.zeros((n, T))
        env = np.zeros((n, T), dtype=np.complex64)
        rot_step = np.exp(1j * omega * dti)
        rot = np.ones(n, dtype=complex)
        hub_lp = 0.0
        for t in range(T):
            if t in handlers:
                _run_handlers(
                    handlers[t],
                    t,
                    act,
                    readout_delay,
                    act_units,
                    inp,
                    q,
                    q_trace,
                    choices,
                    rewards,
                    pes,
                    p_choice1,
                    sched,
                    cfg,
                    knobs,
                    map_dur,
                    motor_dur,
                    fb_dur,
                    routes,
                    pat,
                )
            drive = cfg.sl_input_gain * inp[:, t] + net.bias
            for _ in range(sub):
                amp = np.abs(z)
                hub_amp = float(np.mean(amp[hub]))
                # The gate reads the hub amplitude low-passed over sl_gate_tau
                # (all-or-none episodes rather than cycle-by-cycle flicker).
                hub_lp += (dti / cfg.sl_gate_tau) * (hub_amp - hub_lp)
                s_ign = 1.0 / (
                    1.0
                    + np.exp(
                        -cfg.sl_ignition_slope * (hub_lp - cfg.sl_ignition_threshold)
                    )
                )
                a_t = a_vec.copy()
                a_t[hub] += knobs.g_b * (
                    cfg.sl_ignition_gain * s_ign - cfg.sl_adapt_gain * h
                )
                dz = 1j * omega * z + lam * (
                    (a_t - amp * amp) * z + kappa_w @ z + drive * rot
                )
                z = (
                    z
                    + dti * dz
                    + noise_sd
                    * (noise_rng.standard_normal(n) + 1j * noise_rng.standard_normal(n))
                )
                rot = rot * rot_step
                # Adaptation builds up while the hub is ignited (terminates
                # the episode and makes the hub refractory afterwards).
                h = h + (dti / cfg.tau_adapt) * (-h + s_ign)
            rot = rot / np.abs(rot)
            ts[:, t] = z.real
            amp = np.abs(z)
            act[:, t] = amp
            env[:, t] = z * np.conj(rot)
            ign_trace[t] = float(np.mean(amp[hub]))
            gate_trace[t] = knobs.g_b * s_ign

    return {
        "ts": ts,
        "readout": act,
        "inputs": inp,
        "ctx_state": ctx_state,
        "slow": slow,
        "q_trace": q_trace,
        "choices": choices,
        "rewards": rewards,
        "pes": pes,
        "p_choice1": p_choice1,
        "hub_trace": ign_trace,
        "ignition_gate": gate_trace,
        "tag_trace": tag_trace,
        "reaff_attenuation": reaff_atten,
        "hidden_envelope": env,
        "oscillator_frequency_hz": None if omega is None else omega / (2.0 * np.pi),
    }


def _run_handlers(
    items,
    t,
    act,
    readout_delay,
    act_units,
    inp,
    q,
    q_trace,
    choices,
    rewards,
    pes,
    p_choice1,
    sched,
    cfg,
    knobs,
    map_dur,
    motor_dur,
    fb_dur,
    routes,
    pat,
):
    for kind, i in items:
        if kind == "map":
            # Plastic stimulus-response pathway: action-specific gains 1 + q_a
            # (plus the fixed reflex mapping stimulus a -> action a, if any).
            q_trace[i] = q
            stim_id = int(sched["stim_ids"][i])
            for a in (0, 1):
                gain = cfg.g_sm * (1.0 + q[a]) + cfg.reflex_gain * float(stim_id == a)
                _add_pulse(
                    inp,
                    act_units[a],
                    t,
                    map_dur,
                    gain * np.ones(act_units[a].size),
                )
        elif kind == "respond":
            s0 = int(sched["stim_idx"][i]) + readout_delay
            lo, hi = min(s0, t - 1), t
            m0 = float(np.mean(act[act_units[0], lo:hi]))
            m1 = float(np.mean(act[act_units[1], lo:hi]))
            p1 = 1.0 / (1.0 + np.exp(-cfg.choice_beta * (m1 - m0)))
            c = int(sched["choice_u"][i] < p1)
            p_choice1[i] = p1
            choices[i] = c
            if cfg.feedback_mode == "scrambled":
                p_win = 0.5
            else:
                p_win = (
                    cfg.p_reward
                    if c == int(sched["good_arm"][i])
                    else 1.0 - cfg.p_reward
                )
            rewards[i] = 1.0 if sched["reward_u"][i] < p_win else -1.0
            _add_pulse(
                inp,
                act_units[c],
                t,
                motor_dur,
                cfg.a_motor * np.ones(act_units[c].size),
            )
            _add_pulse(
                inp,
                routes["feedback"],
                int(sched["fb_idx"][i]),
                fb_dur,
                cfg.a_fb * rewards[i] * pat["feedback"],
            )
        elif kind == "learn":
            # Rescorla-Wagner update of the chosen gain with a counterfactual
            # update of the unchosen one (complementary two-armed reversal
            # bandit); eta = 0 freezes the stimulus-response mapping.
            c = int(choices[i])
            pes[i] = rewards[i] - q[c]
            pe_u = -rewards[i] - q[1 - c]
            q[c] = q[c] + knobs.eta * pes[i]
            q[1 - c] = q[1 - c] + knobs.eta * pe_u


def _events_table(sched: dict, sim: dict, cfg: AgentConfig) -> pd.DataFrame:
    dt = cfg.dt
    phi0 = sched["phi0"]
    rows = []

    def _row(
        idx,
        dur,
        ttype,
        value,
        channel,
        trial=np.nan,
        choice=np.nan,
        reward=np.nan,
        yoked=None,
        eid=None,
    ):
        rows.append(
            {
                "onset": round(float(idx) * dt, 6),
                "duration": round(float(dur), 6),
                "trial_type": ttype,
                "value": value,
                "choice": choice,
                "reward": reward,
                "yoked_to": yoked,
                "phase_bin": int(_phase_bin(idx, cfg, phi0)),
                "impact_channel": channel,
                "trial": trial,
                "event_id": eid,
            }
        )

    for i in range(int(cfg.n_trials)):
        c = int(sim["choices"][i])
        r = float(sim["rewards"][i])
        _row(
            sched["goal_idx"][i],
            0.3,
            "goal_cue",
            int(sched["goal_ids"][i]),
            IMPACT_CHANNEL_TASK,
            trial=i,
            eid=f"goal{i:04d}",
        )
        _row(
            sched["stim_idx"][i],
            0.3,
            "stimulus",
            int(sched["stim_ids"][i]),
            IMPACT_CHANNEL_TASK,
            trial=i,
            eid=f"stim{i:04d}",
        )
        _row(
            sched["resp_idx"][i],
            0.15,
            "response",
            c,
            IMPACT_CHANNEL_TASK,
            trial=i,
            choice=c,
            eid=f"resp{i:04d}",
        )
        _row(
            sched["fb_idx"][i],
            0.3,
            "feedback",
            r,
            IMPACT_CHANNEL_TASK,
            trial=i,
            choice=c,
            reward=r,
            eid=f"fb{i:04d}",
        )
    for j in range(int(cfg.n_reafference_pairs)):
        rid = int(sched["reaff_ids"][j])
        _row(
            sched["act_idx"][j],
            0.1,
            "action",
            rid,
            IMPACT_CHANNEL_AGENCY,
            eid=f"act{j:04d}",
        )
        _row(
            sched["self_idx"][j],
            0.2,
            "self_caused",
            rid,
            IMPACT_CHANNEL_AGENCY,
            eid=f"self{j:04d}",
        )
        _row(
            sched["other_idx"][j],
            0.2,
            "other_caused",
            rid,
            IMPACT_CHANNEL_AGENCY,
            yoked=f"self{j:04d}",
            eid=f"other{j:04d}",
        )
    df = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    df = df.sort_values(["onset", "event_id"], kind="mergesort").reset_index(drop=True)
    for col in ("trial", "choice"):
        df[col] = df[col].astype("Int64")
    return df


def _ignition_events(gate: np.ndarray, g_b: float, dt: float) -> List[float]:
    """Onsets of ignition: the all-or-none hub amplification switching on
    (gate ``g_b * s_ign`` crossing ``0.5 * g_b``; none when ``g_b == 0``)."""
    if float(g_b) <= 0 or gate.size < 2:
        return []
    above = gate > 0.5 * float(g_b)
    onsets = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    return [round(float(i) * dt, 6) for i in onsets]


def _assemble_system(
    family: str,
    dynamics: str,
    net: _Net,
    knobs: Knobs,
    cfg: AgentConfig,
    seed: int,
    sched: dict,
    sim: dict,
    rest: Optional[dict],
    extra_meta: Optional[dict] = None,
) -> BenchSystem:
    dt = cfg.dt
    events = _events_table(sched, sim, cfg)
    n = net.W.shape[0]
    periph = [name for name in net.module_order if name != "W" and name in net.modules]
    module_of_unit = [""] * n
    for name, idx in net.modules.items():
        for i in idx:
            module_of_unit[int(i)] = name
    meta = {
        "generator_version": GENERATOR_VERSION,
        "family": family,
        "dynamics": dynamics,
        "substrate": SUBSTRATE_OF_DYNAMICS[dynamics],
        "seed": int(seed),
        "dt": float(dt),
        "sampling_frequency": float(1.0 / dt),
        "n_nodes": int(n),
        "n_time": int(sim["ts"].shape[1]),
        "knobs": knobs.to_dict(),
        "config": cfg.to_dict(),
        "principles": list(PRINCIPLES),
        "switch_for_principle": dict(SWITCH_FOR_PRINCIPLE),
        "module_order": list(net.module_order),
        "module_roles": {k: MODULE_ROLES.get(k, k) for k in net.module_order},
        "modules": {k: [int(i) for i in v] for k, v in net.modules.items()},
        "module_of_unit": module_of_unit,
        "workspace_nodes": [int(i) for i in net.routes["hub"]],
        "bearer_nodes": list(range(int(n))),
        "iim_macro_nodes": {k: [int(i) for i in net.modules[k]] for k in periph},
        "iim_grain": "module_mean_periphery",
        "iim_lag_samples": 2,
        "iim_bins": 2,
        "slow_rhythm_hz": float(1.0 / (cfg.slow_period_samples * dt)),
        "n_phase_bins": int(cfg.n_phase_bins),
        "bandit_end_sec": round(float(sched["bandit_end_idx"]) * dt, 6),
        "reafference_start_sec": round(float(sched["reafference_start_idx"]) * dt, 6),
        "has_rest_run": rest is not None,
    }
    if dynamics == "stuart_landau":
        meta["family_c_design"] = FAMILY_C_DESIGN
    meta.update(extra_meta or {})
    meta.update(net.extra_meta)
    if "iim_macro_nodes" in net.extra_meta:
        meta["iim_macro_nodes"] = net.extra_meta["iim_macro_nodes"]

    q_per_sample = np.zeros((2, sim["ts"].shape[1]))
    if int(cfg.n_trials):
        # Step function of the plastic gains in force at each sample.
        edges = list(sched["stim_idx"]) + [sim["ts"].shape[1]]
        for i in range(len(edges) - 1):
            q_per_sample[:, edges[i] : edges[i + 1]] = sim["q_trace"][i][:, None]
    correct = (sim["choices"] == sched["good_arm"][: len(sim["choices"])]).astype(float)
    oracle = {
        "family": family,
        "seed": int(seed),
        "intended_bits": list(knobs.bits()),
        "mechanisms": {p: bool(b) for p, b in zip(PRINCIPLES, knobs.bits())},
        "unit_adjacency": net.W.T.copy(),
        "module_adjacency": net.module_adjacency.copy(),
        "module_graph_acyclic": bool(is_acyclic(net.module_adjacency)),
        "context_state": sim["ctx_state"],
        "context_states_visited": int(np.unique(sim["ctx_state"]).size),
        "q_by_trial": sim["q_trace"],
        "q_per_sample": q_per_sample,
        "prediction_errors": sim["pes"],
        "choices": sim["choices"],
        "rewards": sim["rewards"],
        "p_choice1": sim["p_choice1"],
        "good_arm": sched["good_arm"],
        "accuracy": float(np.mean(correct)) if correct.size else float("nan"),
        "hub_activity": sim["hub_trace"],
        "ignition_gate": sim["ignition_gate"],
        "ignition_onsets_sec": _ignition_events(sim["ignition_gate"], knobs.g_b, dt),
        "efference_tag": sim["tag_trace"],
        "reafference_attenuation": sim["reaff_attenuation"],
        "slow_phase": _phase(np.arange(sim["ts"].shape[1]), cfg, sched["phi0"]),
        "action_onsets_sec": [round(float(i) * dt, 6) for i in sched["act_idx"]],
        "inputs": sim["inputs"],
    }
    if sim.get("hidden_envelope") is not None:
        # Oscillator families: demodulated complex state z_j e^{-i w_j t}
        # (hidden; the recorded signal is Re z) and natural frequencies.
        oracle["hidden_envelope"] = sim["hidden_envelope"]
        oracle["oscillator_frequency_hz"] = sim["oscillator_frequency_hz"]
    oracle.update(net.extra_oracle)
    return BenchSystem(
        ts=sim["ts"],
        events=events,
        meta=meta,
        oracle=oracle,
        rest_ts=None if rest is None else rest["ts"],
    )


def _simulate_modular(
    family: str,
    dynamics: str,
    builder,
    knobs: Optional[Knobs],
    config: Optional[AgentConfig],
    seed: int,
    extra_meta: Optional[dict] = None,
) -> BenchSystem:
    knobs = knobs if knobs is not None else NOMINAL_KNOBS
    cfg = config if config is not None else AgentConfig()
    (
        struct_rng,
        task_rng,
        noise_rng,
        sl_rng,
        rest_task_rng,
        rest_noise_rng,
        _null_rng,
        rest_sl_rng,
    ) = _streams(seed, 8)
    net = builder(knobs, cfg, struct_rng)
    sched = _task_schedule(cfg, task_rng)
    # The structural stream is shared: family C draws its frequencies from a
    # copy so that the rest run sees the same oscillators.
    sl_state = sl_rng.bit_generator.state
    sim = _simulate_agent(net, knobs, cfg, sched, noise_rng, dynamics, sl_rng)
    rest = None
    if cfg.rest_sec > 0:
        n_rest = _samples(cfg.rest_sec, cfg.dt)
        sw_idx, sw_u = _context_switches(cfg, rest_task_rng, n_rest)
        rest_sched = dict(sched)
        rest_sched.update(
            {"n_time": n_rest, "ctx_switch_idx": sw_idx, "ctx_switch_u": sw_u}
        )
        rest_sl_rng.bit_generator.state = sl_state
        rest = _simulate_agent(
            net,
            knobs,
            cfg,
            rest_sched,
            rest_noise_rng,
            dynamics,
            rest_sl_rng,
            task_inputs=False,
        )
    return _assemble_system(
        family, dynamics, net, knobs, cfg, seed, sched, sim, rest, extra_meta=extra_meta
    )


def simulate_family_a(
    knobs: Optional[Knobs] = None,
    config: Optional[AgentConfig] = None,
    seed: int = 0,
) -> BenchSystem:
    """Family A: modular stochastic rate-network agent (see module docstring)."""
    return _simulate_modular("A", "rate", build_family_a_network, knobs, config, seed)


def simulate_family_c(
    knobs: Optional[Knobs] = None,
    config: Optional[AgentConfig] = None,
    seed: int = 0,
) -> BenchSystem:
    """
    Family C (held out; design 2): Stuart-Landau oscillators on the family-A
    module graph with the same five switches and task. Node ``j`` follows

        dz_j = { i w_j z_j + lam [ (a_j(t) - |z_j|^2) z_j + kappa (W z)_j
                 + I_j(t) e^{i w_j t} ] } dt + lam sigma dB_j

    (``lam = sl_rate``, ``w_j = 2 pi f_j`` with ``f_j ~ U(sl_freq_hz)``,
    ``kappa = sl_coupling``, ``sigma = sl_sigma``, complex white noise). Task
    and endogenous inputs ``I_j`` (the family-A input channels, gain
    ``sl_input_gain``) are resonant drives at each node's own frequency. The
    hub's bifurcation parameter is ``sl_hub_a + g_b (sl_ignition_gain s -
    sl_adapt_gain h)``: the ignition gate ``s`` is a steep sigmoid
    (``sl_ignition_slope``) of the hub amplitude low-passed over
    ``sl_gate_tau`` crossing ``sl_ignition_threshold``, and ``h`` is an
    adaptation that integrates the gate (time constant ``tau_adapt``), so
    ignitions are all-or-none episodes that end by adaptation. With the
    defaults the dominant linear mode of the nominal network is just below its
    Hopf bifurcation (``sl_a + kappa * spectral_radius = -1 + 1.1 * 0.85``),
    so every coupling switch changes the dynamics (preregistered manipulation
    checks in :mod:`impact_pipeline.bench.manipulation`). The recorded signal
    is ``Re z`` at the family-A sampling interval (``sl_substeps``
    Euler-Maruyama substeps per sample); choices are read out from ``|z|``.
    The oracle adds the demodulated envelope ``z_j e^{-i w_j t}``
    (``hidden_envelope``) and the natural frequencies.

    Design 1 (generator 1.0.0: detuned 0.5-1 Hz oscillators, weak coupling
    0.3 W z against a 4x additive input drive) failed the manipulation checks:
    the NAS and IIM switches changed the recorded signal by 3-4 % and ignition
    occurred in < 1 % of samples.
    """
    return _simulate_modular(
        "C", "stuart_landau", build_family_a_network, knobs, config, seed
    )


# ---------------------------------------------------------------------------
# Null and over-excluded systems
# ---------------------------------------------------------------------------


NULL_KINDS = ("independent_noise", "ar1")


def _like_meta(like: BenchSystem, family: str, extra: dict) -> dict:
    meta = dict(like.meta)
    meta.update(
        {"family": family, "knobs": None, "template_family": like.meta.get("family")}
    )
    meta.update(extra)
    return meta


def _null_oracle(like: BenchSystem, family: str, seed: int, extra: dict) -> dict:
    out = {
        "family": family,
        "seed": int(seed),
        "intended_bits": [0, 0, 0, 0, 0],
        "mechanisms": {p: False for p in PRINCIPLES},
        "template_seed": like.meta.get("seed"),
    }
    out.update(extra)
    return out


def null_system(
    kind: str, like: BenchSystem, seed: int = 0, ar_coef: float = 0.9
) -> BenchSystem:
    """
    Null system with the declared meta and the events table of ``like`` (so
    event-locked estimators are defined) and a time series without any
    mechanism: ``'independent_noise'`` (white Gaussian) or ``'ar1'``
    (independent AR(1) per node, coefficient ``ar_coef``). Each node is scaled
    to the mean and SD of the corresponding node of ``like``.
    """
    if kind not in NULL_KINDS:
        raise ValueError(f"kind must be one of {NULL_KINDS}")
    if not -1.0 < ar_coef < 1.0:
        raise ValueError("ar_coef must be in (-1, 1)")
    rng = _streams(seed, 8)[6]
    n, t = like.ts.shape
    xi = rng.standard_normal((n, t))
    if kind == "ar1":
        x = np.zeros((n, t))
        x[:, 0] = xi[:, 0]
        scale = np.sqrt(1.0 - ar_coef**2)
        for k in range(1, t):
            x[:, k] = ar_coef * x[:, k - 1] + scale * xi[:, k]
    else:
        x = xi
    mu = like.ts.mean(axis=1, keepdims=True)
    sd = like.ts.std(axis=1, keepdims=True)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)
    ts = mu + sd * x
    extra = {
        "null_kind": kind,
        "ar_coef": float(ar_coef) if kind == "ar1" else None,
        "seed": int(seed),
    }
    return BenchSystem(
        ts=ts,
        events=like.events.copy(),
        meta=_like_meta(like, "null", extra),
        oracle=_null_oracle(like, "null", seed, {"null_kind": kind}),
    )


def hypersynchronous_system(
    like: BenchSystem,
    seed: int = 0,
    freq_hz: float = 3.0,
    noise_sd: float = 0.05,
) -> BenchSystem:
    """
    Hypersynchronous-undifferentiated system (seizure-like): every node is the
    same oscillation (random positive gain per node) plus small independent
    noise, with the events table of ``like``. No task processing, a single
    global state trajectory.
    """
    rng = _streams(seed, 8)[6]
    n, t = like.ts.shape
    dt = like.dt
    tt = np.arange(t) * dt
    common = np.sin(2.0 * np.pi * float(freq_hz) * tt + rng.uniform(0, 2 * np.pi))
    gains = rng.uniform(0.8, 1.2, size=(n, 1))
    ts = gains * common[None, :] + float(noise_sd) * rng.standard_normal((n, t))
    extra = {
        "hypersync_freq_hz": float(freq_hz),
        "hypersync_noise_sd": float(noise_sd),
        "seed": int(seed),
    }
    return BenchSystem(
        ts=ts,
        events=like.events.copy(),
        meta=_like_meta(like, "hypersynchronous", extra),
        oracle=_null_oracle(like, "hypersynchronous", seed, {"common_signal": common}),
    )


# ---------------------------------------------------------------------------
# Family B: binary / kinetic-Ising networks with exact TPMs
# ---------------------------------------------------------------------------


BINARY_NETWORK_KINDS = (
    "independent",
    "ring",
    "all_to_all",
    "feedforward_star",
    "xor_loop",
    "hidden_driver",
)


@dataclass(frozen=True)
class BinaryNetwork:
    """
    Exact binary network.

    ``tpm[s, s']`` is the state-by-state transition probability P(s'|s)
    (for ``hidden_driver``: the interventional TPM with the hidden driver
    marginalised at its stationary distribution, independent of do(s));
    ``tpm_sbn[s, i]`` = P(s'_i = 1 | s) for conditionally independent kinds
    (None otherwise); ``connectivity[source, target]`` = 1 for direct causal
    influence among observed units (self-loops included). States use the
    big-endian order of ``mpc_metrics`` (node 0 most significant).
    """

    kind: str
    n: int
    states: np.ndarray
    tpm: np.ndarray
    tpm_sbn: Optional[np.ndarray]
    connectivity: np.ndarray
    params: dict
    stationary: np.ndarray
    observational_tpm: Optional[np.ndarray] = None
    joint_tpm: Optional[np.ndarray] = None


def binary_states(n: int, base: int = 2) -> np.ndarray:
    """All ``base**n`` states, big-endian (as mpc_metrics' ``states_full``)."""
    n = int(n)
    sid = np.arange(int(base) ** n, dtype=np.int64)[:, None]
    powv = (int(base) ** np.arange(n - 1, -1, -1, dtype=np.int64))[None, :]
    return ((sid // powv) % int(base)).astype(np.int16)


def state_index(state: Sequence[int], base: int = 2) -> int:
    key = 0
    for v in state:
        key = key * int(base) + int(v)
    return int(key)


def sbn_to_sbs(tpm_sbn: np.ndarray, states: np.ndarray) -> np.ndarray:
    """State-by-node P(s'_i=1|s) -> state-by-state P(s'|s) (cond. independence)."""
    p1 = np.asarray(tpm_sbn, dtype=float)
    out = np.ones((p1.shape[0], states.shape[0]))
    for i in range(states.shape[1]):
        on = states[:, i] == 1
        out *= np.where(on[None, :], p1[:, i : i + 1], 1.0 - p1[:, i : i + 1])
    return out


def _is_stationary(pi: np.ndarray, tpm: np.ndarray, atol: float = 1e-9) -> bool:
    return bool(
        np.all(np.isfinite(pi))
        and pi.min() > -atol
        and abs(pi.sum() - 1.0) < 1e-6
        and np.allclose(pi @ tpm, pi, atol=atol)
    )


def stationary_distribution(tpm: np.ndarray, max_squarings: int = 80) -> np.ndarray:
    """
    Stationary distribution pi = pi T.

    For an irreducible chain it is unique and solved directly. For a
    reducible chain (several closed classes, e.g. a noise-free XOR loop), or
    when the direct solution is not a valid stationary distribution
    (numerically reducible chains, e.g. a very large ``beta``), the Cesaro
    limit reached from the uniform distribution is returned: the limit of the
    lazy chain (I + T) / 2, which has the same stationary distributions and is
    aperiodic, computed by repeated squaring. It is the unique stationary
    distribution whenever that exists.
    """
    from scipy.sparse.csgraph import connected_components

    tpm = np.asarray(tpm, dtype=float)
    k = tpm.shape[0]
    n_classes = connected_components(tpm > 0, directed=True, connection="strong")[0]
    a = tpm.T - np.eye(k)
    a[-1, :] = 1.0
    b = np.zeros(k)
    b[-1] = 1.0
    pi = None
    if n_classes == 1:
        try:
            pi = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            pi = None
    if pi is not None and _is_stationary(pi, tpm):
        pi = np.clip(pi, 0.0, None)
        return pi / pi.sum()
    lazy = 0.5 * (np.eye(k) + tpm)
    p = np.full(k, 1.0 / k)
    for _ in range(int(max_squarings)):
        q = p @ lazy
        if np.max(np.abs(q - p)) < 1e-13:
            break
        lazy = lazy @ lazy
        lazy /= lazy.sum(axis=1, keepdims=True)
        p = np.full(k, 1.0 / k) @ lazy
    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    if not _is_stationary(p, tpm, atol=1e-7):
        raise ValueError("stationary distribution did not converge")
    return p


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def family_b_network(
    kind: str,
    n: int = 4,
    beta: float = 1.0,
    coupling: float = 0.6,
    self_coupling: float = 0.6,
    field_h: float = 0.0,
    noise: float = 0.05,
    driver_weight: float = 1.0,
    driver_persistence: float = 0.9,
) -> BinaryNetwork:
    """
    Exact binary network of the given ``kind``.

    Kinetic Ising kinds use P(s'_i = 1 | s) = sigmoid(2 beta (sum_j J_ij
    sigma_j + h_i)) with spins sigma = 2 s - 1 and ``J[i, j]`` the influence
    of j on i: ``independent`` (self-coupling only), ``ring`` (bidirectional
    nearest neighbours), ``all_to_all``, ``feedforward_star`` (unit 0 drives
    all others; no feedback). ``xor_loop``: s'_i = XOR(s_{i-1}, s_{i+1}) with
    flip probability ``noise`` (n >= 3). ``hidden_driver``: non-interacting
    self-coupled units sharing a hidden binary driver d with persistence
    ``driver_persistence``; ``tpm`` is interventional (driver marginalised at
    its stationary 1/2), ``observational_tpm`` the exact stationary one-step
    P(s_{t+1} | s_t) of the observed process and ``joint_tpm`` the Markov chain
    over (s, d) (index = 2 * state + (d == +1)).
    """
    if kind not in BINARY_NETWORK_KINDS:
        raise ValueError(f"kind must be one of {BINARY_NETWORK_KINDS}")
    n = int(n)
    if n < 2 or n > 10:
        raise ValueError("n must be in [2, 10]")
    if kind in ("ring", "xor_loop") and n < 3:
        raise ValueError(f"{kind} needs n >= 3")
    states = binary_states(n)
    sig = 2.0 * states.astype(float) - 1.0
    params = {
        "kind": kind,
        "n": n,
        "beta": float(beta),
        "coupling": float(coupling),
        "self_coupling": float(self_coupling),
        "field_h": float(field_h),
    }
    observational = None
    joint = None
    if kind == "xor_loop":
        if not 0.0 <= noise <= 0.5:
            raise ValueError("noise must be in [0, 0.5]")
        params["noise"] = float(noise)
        sbn = np.zeros((states.shape[0], n))
        cm = np.zeros((n, n), dtype=int)
        for i in range(n):
            left, right = (i - 1) % n, (i + 1) % n
            xor = np.bitwise_xor(states[:, left], states[:, right])
            sbn[:, i] = np.where(xor == 1, 1.0 - noise, noise)
            cm[left, i] = 1
            cm[right, i] = 1
        tpm = sbn_to_sbs(sbn, states)
    elif kind == "hidden_driver":
        if not 0.0 < driver_persistence < 1.0:
            raise ValueError("driver_persistence must be in (0, 1)")
        params.update(
            {
                "driver_weight": float(driver_weight),
                "driver_persistence": float(driver_persistence),
            }
        )
        J = np.eye(n) * float(self_coupling)
        cm = (J != 0).astype(int).T
        per_driver = []
        for d in (-1.0, 1.0):
            p1 = _sigmoid(
                2.0
                * beta
                * (sig * float(self_coupling) + float(driver_weight) * d + field_h)
            )
            per_driver.append(sbn_to_sbs(p1, states))
        tpm = 0.5 * (per_driver[0] + per_driver[1])
        sbn = None
        k = states.shape[0]
        joint = np.zeros((2 * k, 2 * k))
        pd_ = float(driver_persistence)
        for s in range(k):
            for di in (0, 1):
                row = 2 * s + di
                for dj in (0, 1):
                    p_d = pd_ if dj == di else 1.0 - pd_
                    joint[row, dj::2] = per_driver[di][s] * p_d
        pi_joint = stationary_distribution(joint).reshape(k, 2)
        p_s = pi_joint.sum(axis=1, keepdims=True)
        # P(d | s) under stationarity; for states of zero stationary mass the
        # conditional is undefined and the driver prior (1/2, 1/2) is used,
        # i.e. those rows equal the interventional TPM.
        post = np.where(p_s > 0, pi_joint / np.where(p_s > 0, p_s, 1.0), 0.5)
        observational = post[:, [0]] * per_driver[0] + post[:, [1]] * per_driver[1]
    else:
        J = np.eye(n) * float(self_coupling)
        if kind == "ring":
            J += float(coupling) * (
                np.roll(np.eye(n), 1, axis=1) + np.roll(np.eye(n), -1, axis=1)
            )
        elif kind == "all_to_all":
            J += float(coupling) * (np.ones((n, n)) - np.eye(n))
        elif kind == "feedforward_star":
            J[1:, 0] = float(coupling)
        sbn = _sigmoid(2.0 * beta * (sig @ J.T + float(field_h)))
        tpm = sbn_to_sbs(sbn, states)
        cm = (J != 0).astype(int).T
        params["J"] = J.tolist()
    tpm = tpm / tpm.sum(axis=1, keepdims=True)
    stat = stationary_distribution(observational if observational is not None else tpm)
    return BinaryNetwork(
        kind=kind,
        n=n,
        states=states,
        tpm=tpm,
        tpm_sbn=sbn,
        connectivity=cm,
        params=params,
        stationary=stat,
        observational_tpm=observational,
        joint_tpm=joint,
    )


def sample_binary_trajectory(
    net: BinaryNetwork,
    n_steps: int,
    seed: int = 0,
    initial_state: Optional[Sequence[int]] = None,
    return_driver: bool = False,
):
    """
    Sample ``n_steps`` states (``n_steps x n`` int8) from the network's own
    dynamics (the joint chain with the hidden driver for ``hidden_driver``;
    the driver sequence is returned with ``return_driver=True``). The
    initial state is drawn from the stationary distribution unless given.
    """
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    rng = np.random.default_rng(int(seed))
    k = net.states.shape[0]
    out = np.zeros((n_steps, net.n), dtype=np.int8)
    driver = None
    if net.joint_tpm is not None:
        cdf = np.cumsum(net.joint_tpm, axis=1)
        pi = stationary_distribution(net.joint_tpm)
        if initial_state is None:
            j = int(rng.choice(2 * k, p=pi))
        else:
            j = 2 * state_index(initial_state) + int(rng.integers(0, 2))
        driver = np.zeros(n_steps, dtype=np.int8)
        u = rng.random(n_steps)
        for t in range(n_steps):
            out[t] = net.states[j // 2]
            driver[t] = 1 if (j % 2) else -1
            j = min(int(np.searchsorted(cdf[j], u[t], side="right")), 2 * k - 1)
    else:
        cdf = np.cumsum(net.tpm, axis=1)
        if initial_state is None:
            s = int(rng.choice(k, p=net.stationary))
        else:
            s = state_index(initial_state)
        u = rng.random(n_steps)
        for t in range(n_steps):
            out[t] = net.states[s]
            s = min(int(np.searchsorted(cdf[s], u[t], side="right")), k - 1)
    if return_driver:
        return out, driver
    return out


def empirical_tpm(
    trajectory: np.ndarray, lag: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Row-normalised transition counts (NaN rows where a state was not visited)
    and the visit counts, from a ``time x n`` binary trajectory."""
    traj = np.asarray(trajectory, dtype=np.int64)
    n = traj.shape[1]
    powv = 2 ** np.arange(n - 1, -1, -1, dtype=np.int64)
    keys = traj @ powv
    k = 2**n
    counts = np.zeros((k, k))
    np.add.at(counts, (keys[:-lag], keys[lag:]), 1.0)
    visits = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tpm = counts / visits[:, None]
    tpm[visits == 0] = np.nan
    return tpm, visits


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


GENERATOR_NAMES = (
    "family_a",
    "family_c",
    "patchwork",
    "null_independent_noise",
    "null_ar1",
    "hypersynchronous",
)
# External and adversarial generators (not built around the five switches;
# held out: confirmatory runs only). Built lazily by make_system.
WHOLE_BRAIN_GENERATORS = ("whole_brain", "whole_brain_eeg", "whole_brain_bold")
ADVERSARIAL_GENERATORS = (
    "adversarial_parity_grid",
    "adversarial_hypersynchrony",
    "adversarial_common_driver",
    "adversarial_reflex_arc",
    "adversarial_random_label_self_other",
    "adversarial_scrambled_feedback",
)
EXTERNAL_GENERATOR_NAMES = WHOLE_BRAIN_GENERATORS + ADVERSARIAL_GENERATORS


def make_system(
    generator: str,
    knobs: Optional[Knobs] = None,
    config: Optional[AgentConfig] = None,
    seed: int = 0,
    template_family: str = "A",
    **kwargs,
) -> BenchSystem:
    """
    Build one system by generator name. Null and hypersynchronous systems take
    their events and declared meta from a nominal run of ``template_family``
    with the same seed and config. External generators
    (``EXTERNAL_GENERATOR_NAMES``): ``adversarial_<kind>`` (see
    :mod:`impact_pipeline.bench.adversarial`; ``config`` is the agent
    config) and ``whole_brain`` / ``whole_brain_eeg`` / ``whole_brain_bold``
    (source, EEG-like and BOLD-like observations; ``whole_brain_config=`` a
    :class:`~impact_pipeline.bench.whole_brain.WholeBrainConfig` or dict,
    other keywords go to the forward model).
    """
    if generator == "family_a":
        return simulate_family_a(knobs, config, seed)
    if generator == "family_c":
        return simulate_family_c(knobs, config, seed)
    if generator == "patchwork":
        from impact_pipeline.bench.patchwork import simulate_patchwork

        return simulate_patchwork(knobs, config, seed, **kwargs)
    if generator.startswith("adversarial_"):
        from impact_pipeline.bench.adversarial import make_adversarial

        return make_adversarial(
            generator[len("adversarial_") :], config, seed, **kwargs
        )
    if generator in WHOLE_BRAIN_GENERATORS:
        from impact_pipeline.bench import forward, whole_brain

        wb_cfg = kwargs.pop("whole_brain_config", None)
        if not isinstance(wb_cfg, whole_brain.WholeBrainConfig):
            wb_cfg = whole_brain.whole_brain_config_from_dict(wb_cfg)
        source = whole_brain.simulate_whole_brain(wb_cfg, seed)
        if generator == "whole_brain_eeg":
            return forward.eeg_forward(source, seed=seed, **kwargs)
        if generator == "whole_brain_bold":
            return forward.bold_forward(source, seed=seed, **kwargs)
        return source
    if generator in ("null_independent_noise", "null_ar1", "hypersynchronous"):
        tmpl = (
            simulate_family_c
            if str(template_family).upper() == "C"
            else simulate_family_a
        )
        like = tmpl(None, config, seed)
        if generator == "hypersynchronous":
            return hypersynchronous_system(like, seed=seed, **kwargs)
        kind = "independent_noise" if generator == "null_independent_noise" else "ar1"
        return null_system(kind, like, seed=seed, **kwargs)
    raise ValueError(
        f"generator must be one of {GENERATOR_NAMES + EXTERNAL_GENERATOR_NAMES}, "
        f"got {generator!r}"
    )


def all_bit_patterns() -> List[Tuple[int, ...]]:
    """The 32 cells of {0,1}^5 in lexicographic order (PRINCIPLES order)."""
    return [tuple(int(b) for b in bits) for bits in itertools.product((0, 1), repeat=5)]


def summarise_oracle(system: BenchSystem) -> dict:
    """Scalar oracle summaries (hidden channels) used by tests and reports."""
    o = system.oracle
    out = {"intended_bits": list(o.get("intended_bits") or [])}
    for key in (
        "adversarial",
        "designed_to_fool",
        "mean_order",
        "metastability",
        "n_edges_removed",
        "inter_module_coupling",
        "G",
    ):
        if key in o and not isinstance(o[key], np.ndarray):
            out[key] = o[key]
    if "ignition_gate" in o:
        gate = np.asarray(o["ignition_gate"], dtype=float)
        knobs = system.meta.get("knobs") or {}
        g_b = float(knobs.get("g_b", 0.0) or 0.0)
        out["ignition_occupancy"] = (
            float(np.mean(gate > 0.5 * g_b)) if g_b > 0 and gate.size else 0.0
        )
    if "q_by_trial" in o:
        q = np.asarray(o["q_by_trial"], dtype=float)
        out["q_range"] = float(np.ptp(q)) if q.size else 0.0
        out["accuracy"] = float(o.get("accuracy", math.nan))
        out["context_states_visited"] = int(o.get("context_states_visited", 0))
        out["n_ignitions"] = int(len(o.get("ignition_onsets_sec", [])))
        out["module_graph_acyclic"] = bool(o.get("module_graph_acyclic"))
        out["efference_tag_max"] = (
            float(np.max(o["efference_tag"])) if len(o["efference_tag"]) else 0.0
        )
        att = np.asarray(o.get("reafference_attenuation", []), dtype=float)
        out["reafference_attenuation_mean"] = float(att.mean()) if att.size else 1.0
    return out
