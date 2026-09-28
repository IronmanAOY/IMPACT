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
"""

from __future__ import annotations

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
    draw_patterns,
)

PATCHWORK_MODULES = ("RAM", "PDI", "NAS", "IIM", "SRPI")


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
    extra_oracle = {
        "modules_connected": False,
        "iim_submodule_adjacency": sub_adj,
        "principle_bearers": {p: [int(i) for i in mods[p]] for p in PATCHWORK_MODULES},
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
    knobs: Knobs, cfg: AgentConfig, rng: np.random.Generator
) -> _Net:
    """
    Block-diagonal patchwork coupling (see module docstring). Each module is
    scaled separately by a knob-invariant factor so that its *nominal* block
    has spectral radius ``cfg.spectral_radius`` (every module is as close to
    criticality as the integrated agent).
    """
    nominal_rng = np.random.Generator(type(rng.bit_generator)())
    nominal_rng.bit_generator.state = rng.bit_generator.state
    nominal = _build_patchwork_raw(NOMINAL_KNOBS, cfg, nominal_rng)
    net = _build_patchwork_raw(knobs, cfg, rng)
    scales = {}
    for name, idx in net.modules.items():
        blk = nominal.W[np.ix_(idx, idx)]
        radius = float(np.max(np.abs(np.linalg.eigvals(blk))))
        scales[name] = float(cfg.spectral_radius) / radius if radius > 0 else 1.0
        net.W[np.ix_(idx, idx)] *= scales[name]
    net.extra_meta = dict(net.extra_meta)
    net.extra_meta["coupling_scale"] = scales
    net.extra_meta["nominal_spectral_radius"] = float(cfg.spectral_radius)
    net.extra_meta["spectral_radius"] = float(np.max(np.abs(np.linalg.eigvals(net.W))))
    return net


def simulate_patchwork(
    knobs: Optional[Knobs] = None,
    config: Optional[AgentConfig] = None,
    seed: int = 0,
    dynamics: str = "rate",
) -> BenchSystem:
    """
    Simulate the patchwork (rate units by default; ``dynamics='stuart_landau'``
    for the held-out family-C variant) with the family-A task and events.
    """
    knobs = knobs if knobs is not None else NOMINAL_KNOBS
    family = "patchwork" if dynamics == "rate" else "patchwork_C"
    return _simulate_modular(
        family, dynamics, build_patchwork_network, knobs, config, seed
    )
