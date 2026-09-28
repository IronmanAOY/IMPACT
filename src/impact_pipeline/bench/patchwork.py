"""
MPC-Bench patchwork: five mutually disconnected modules, each implementing
exactly one principle's mechanism (same-bearer / bearer-failure class).

Every principle's mechanism is present *somewhere*, but no single bearer has
all five: there is no coupling (and no shared endogenous input) between the
modules. Module-internal layout (sub-groups from ``np.array_split``):

* ``RAM``  sensory | action-0 | action-1 | value: plastic stimulus-response
  gains (``eta``), bandit stimuli, choices and +-1 feedback;
* ``PDI``  context patterns switching among ``K`` attractors (only module
  receiving the repertoire drive);
* ``NAS``  hub | periphery-1 | periphery-2: receive-and-return broadcast with
  ignition (``g_b``, ``ff_only``);
* ``IIM``  three sub-groups in a chain with backward loops (``c_int``);
* ``SRPI`` motor-0 | motor-1 | sensory: self-caused reafference with the
  efference-copy tag and attenuation (``e``), yoked replays and the slow
  rhythm used for phase matching.

``meta['principle_bearers']`` declares the node set supplying each principle;
``meta['iim_macro_nodes']`` is the system grain (one macro node per module),
``meta['principle_macro_nodes']['IIM']`` the IIM module's own sub-groups.

Graded patchworks (spec v2, V2-4): ``inter_module_coupling = lambda`` in
[0, 1] adds bidirectional random positive couplings between neighbouring
modules of the ring RAM - PDI - NAS - IIM - SRPI - RAM, scaled by ``lambda``
(0 = the disconnected patchwork, 1 = the nominal inter-module gain: the
forward inter-module gain of the integrated family-A agent of the same
config and seed, i.e. ``w_ff`` times its knob-invariant coupling scale).
The inter-module blocks are drawn after every other structural draw, so
``lambda = 0`` reproduces the disconnected
patchwork exactly and a ``lambda`` sweep is paired (common random numbers).
The mechanisms stay in their modules; only the joint dependence between the
modules (the single-source constraint) is graded.
"""

from __future__ import annotations

import functools
from typing import Optional

import numpy as np

from impact_pipeline.bench.generators import (
    AgentConfig,
    BenchSystem,
    Knobs,
    NOMINAL_KNOBS,
    _N_PATTERN_BANK,
    _Net,
    _pos_block,
    _simulate_modular,
    build_family_a_network,
    draw_patterns,
)

PATCHWORK_MODULES = ("RAM", "PDI", "NAS", "IIM", "SRPI")
# Neighbouring modules coupled by the graded patchwork (a ring).
INTER_MODULE_EDGES = tuple(
    (PATCHWORK_MODULES[k], PATCHWORK_MODULES[(k + 1) % len(PATCHWORK_MODULES)])
    for k in range(len(PATCHWORK_MODULES))
)


def _local_block(rng: np.random.Generator, cfg: AgentConfig, m: int) -> np.ndarray:
    g = rng.normal(size=(m, m))
    g = 0.5 * (g + g.T)
    np.fill_diagonal(g, 0.0)
    return (
        cfg.c_loc / m * (np.ones((m, m)) - np.eye(m)) + cfg.loc_jitter / np.sqrt(m) * g
    )


def _connect(W, rng, targets, sources, gain):
    blk = _pos_block(rng, targets.size, sources.size)
    if gain > 0:
        W[np.ix_(targets, sources)] += gain * blk


def _build_patchwork_raw(
    knobs: Knobs, cfg: AgentConfig, rng: np.random.Generator
) -> _Net:
    m = int(cfg.units_per_module)
    mods = {
        name: np.arange(k * m, (k + 1) * m) for k, name in enumerate(PATCHWORK_MODULES)
    }
    n = len(PATCHWORK_MODULES) * m
    W = np.zeros((n, n))
    bias = np.zeros(n)
    for name in PATCHWORK_MODULES:
        idx = mods[name]
        W[np.ix_(idx, idx)] = _local_block(rng, cfg, m)

    ram_s, ram_a0, ram_a1, ram_v = np.array_split(mods["RAM"], 4)
    _connect(W, rng, ram_a0, ram_s, cfg.w_ff)
    _connect(W, rng, ram_a1, ram_s, cfg.w_ff)
    _connect(W, rng, ram_v, np.concatenate([ram_a0, ram_a1]), cfg.w_ff)

    hub, per1, per2 = np.array_split(mods["NAS"], 3)
    for p in (per1, per2):
        recv = 0.0 if knobs.ff_only else knobs.g_b * cfg.w_hub
        _connect(W, rng, hub, p, recv)
        _connect(W, rng, p, hub, knobs.g_b * cfg.w_hub)
    bias[hub] = cfg.hub_bias

    iim_groups = np.array_split(mods["IIM"], 3)
    sub_adj = np.zeros((3, 3))
    for a in range(2):
        fwd_src, fwd_tgt = iim_groups[a], iim_groups[a + 1]
        _connect(W, rng, fwd_tgt, fwd_src, cfg.w_ff)
        _connect(W, rng, fwd_src, fwd_tgt, knobs.c_int)
        sub_adj[a, a + 1] = cfg.w_ff
        sub_adj[a + 1, a] = knobs.c_int

    srpi_m0, srpi_m1, srpi_s = np.array_split(mods["SRPI"], 3)
    _connect(W, rng, srpi_s, np.concatenate([srpi_m0, srpi_m1]), cfg.w_ff)

    sizes = {
        "stim": ram_s.size,
        "reaff": srpi_s.size,
        "tag": srpi_s.size,
        "goal": ram_s.size,
        "feedback": ram_v.size,
    }
    patterns = draw_patterns(rng, sizes)
    ctx_bank = rng.choice([-1.0, 1.0], size=(1, _N_PATTERN_BANK, m))
    routes = {
        "stim": ram_s,
        "goal": ram_s,
        "action": [ram_a0, ram_a1],
        "feedback": ram_v,
        "reaff": srpi_s,
        "tag": srpi_s,
        "motor_self": [srpi_m0, srpi_m1],
        "hub": hub,
        "slow": mods["SRPI"],
    }
    extra_meta = {
        "patchwork": True,
        "principle_bearers": {p: [int(i) for i in mods[p]] for p in PATCHWORK_MODULES},
        "iim_macro_nodes": {p: [int(i) for i in mods[p]] for p in PATCHWORK_MODULES},
        "iim_grain": "module_mean_all_modules",
        "principle_macro_nodes": {
            "IIM": {f"IIM{k}": [int(i) for i in g] for k, g in enumerate(iim_groups)}
        },
        "principle_workspace_nodes": {"NAS": [int(i) for i in hub]},
    }
    # Inter-module blocks of the graded patchwork: always drawn (last), used
    # only when inter_module_coupling > 0 (see build_patchwork_network).
    inter_blocks = {}
    for a, b in INTER_MODULE_EDGES:
        inter_blocks[(b, a)] = _pos_block(rng, m, m)  # a -> b
        inter_blocks[(a, b)] = _pos_block(rng, m, m)  # b -> a
    extra_oracle = {
        "modules_connected": False,
        "iim_submodule_adjacency": sub_adj,
        "principle_bearers": {p: [int(i) for i in mods[p]] for p in PATCHWORK_MODULES},
        "_inter_blocks": inter_blocks,
    }
    return _Net(
        W=W,
        bias=bias,
        modules=mods,
        module_order=PATCHWORK_MODULES,
        module_adjacency=np.zeros((len(PATCHWORK_MODULES), len(PATCHWORK_MODULES))),
        routes=routes,
        context_modules=[mods["PDI"]],
        patterns={**patterns, "context_bank": ctx_bank},
        extra_meta=extra_meta,
        extra_oracle=extra_oracle,
    )


def build_patchwork_network(
    knobs: Knobs,
    cfg: AgentConfig,
    rng: np.random.Generator,
    inter_module_coupling: float = 0.0,
) -> _Net:
    """
    Block-diagonal patchwork coupling (see module docstring). Each module is
    scaled separately by a knob-invariant factor so that its *nominal* block
    has spectral radius ``cfg.spectral_radius`` (every module is as close to
    criticality as the integrated agent). With ``inter_module_coupling =
    lambda > 0`` the ring of neighbouring modules is coupled in both
    directions with gain ``lambda`` times the nominal inter-module gain of
    the family-A agent (module docstring).
    """
    lam = float(inter_module_coupling)
    if not (np.isfinite(lam) and 0.0 <= lam <= 1.0):
        raise ValueError("inter_module_coupling must be in [0, 1]")
    nominal_rng = np.random.Generator(type(rng.bit_generator)())
    nominal_rng_state = rng.bit_generator.state
    nominal_rng.bit_generator.state = nominal_rng_state
    nominal = _build_patchwork_raw(NOMINAL_KNOBS, cfg, nominal_rng)
    net = _build_patchwork_raw(knobs, cfg, rng)
    scales = {}
    for name, idx in net.modules.items():
        blk = nominal.W[np.ix_(idx, idx)]
        radius = float(np.max(np.abs(np.linalg.eigvals(blk))))
        scales[name] = float(cfg.spectral_radius) / radius if radius > 0 else 1.0
        net.W[np.ix_(idx, idx)] *= scales[name]
    extra_oracle = dict(net.extra_oracle)
    blocks = extra_oracle.pop("_inter_blocks")
    pos = {p: k for k, p in enumerate(PATCHWORK_MODULES)}
    madj = np.zeros((len(PATCHWORK_MODULES), len(PATCHWORK_MODULES)))
    # Nominal inter-module gain: the forward inter-module gain of the
    # integrated family-A agent with the same config (w_ff times its
    # knob-invariant coupling scale), built from a copy of the stream.
    ref_rng = np.random.Generator(type(rng.bit_generator)())
    ref_rng.bit_generator.state = nominal_rng_state
    ref = build_family_a_network(NOMINAL_KNOBS, cfg, ref_rng)
    nominal_gain = float(cfg.w_ff) * float(ref.extra_meta["coupling_scale"])
    if lam > 0:
        for (tgt, src), blk in blocks.items():
            gain = lam * nominal_gain
            net.W[np.ix_(net.modules[tgt], net.modules[src])] += gain * blk
            madj[pos[src], pos[tgt]] = gain
    net.module_adjacency = madj
    extra_oracle["modules_connected"] = bool(lam > 0)
    extra_oracle["inter_module_coupling"] = lam
    net.extra_oracle = extra_oracle
    net.extra_meta = dict(net.extra_meta)
    net.extra_meta["coupling_scale"] = scales
    net.extra_meta["nominal_spectral_radius"] = float(cfg.spectral_radius)
    net.extra_meta["spectral_radius"] = float(np.max(np.abs(np.linalg.eigvals(net.W))))
    net.extra_meta["patchwork_inter_module_coupling"] = lam
    return net


def simulate_patchwork(
    knobs: Optional[Knobs] = None,
    config: Optional[AgentConfig] = None,
    seed: int = 0,
    dynamics: str = "rate",
    inter_module_coupling: float = 0.0,
) -> BenchSystem:
    """
    Simulate the patchwork (rate units by default; ``dynamics='stuart_landau'``
    for the held-out family-C variant) with the family-A task and events;
    ``inter_module_coupling`` in [0, 1] grades it (module docstring).
    """
    knobs = knobs if knobs is not None else NOMINAL_KNOBS
    family = "patchwork" if dynamics == "rate" else "patchwork_C"
    builder = functools.partial(
        build_patchwork_network, inter_module_coupling=float(inter_module_coupling)
    )
    return _simulate_modular(family, dynamics, builder, knobs, config, seed)


def patchwork_sweep_levels(n_levels: int = 6) -> list:
    """Inter-module coupling levels 0 .. 1 (first level: disconnected)."""
    if int(n_levels) < 2:
        raise ValueError("n_levels must be >= 2")
    return [round(float(v), 6) for v in np.linspace(0.0, 1.0, int(n_levels))]
