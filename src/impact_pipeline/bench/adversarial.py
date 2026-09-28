"""
MPC-Bench adversarial systems: constructions designed to fool one estimator
(or one single-marker rule) at a time (Paper-1 spec v2, V2-6).

Each system is defined by mechanism, documented in ``ADVERSARIAL_CATALOGUE``
with the estimator it is designed to fool, its intended mechanism pattern
(``PRINCIPLES`` order; the stipulated ground truth of the construction) and
the verdict a valid necessity rule should return. None of them is claimed to
be conscious; each one lacks at least one principle by construction, so the
expected verdict is ``EXCLUDED`` (or at least not ``MPC_CONSISTENT``).

=========================  ===================================================
kind                       construction (designed to fool)
=========================  ===================================================
parity_grid                Aaronson-style inert parity network: every unit
                           computes the XOR of its neighbours on a 3-regular
                           circulant (Moebius-ladder) expander; exact binary
                           TPM; the system rests in its all-off fixed point
                           and has no inputs, so the recorded signal is
                           measurement noise (IIM on the known TPM / an
                           IIM-only rule: high integration, no responsiveness,
                           no differentiation)
hypersynchrony             one strong common oscillation in every node
                           (synchrony-based broadcast markers, Phi_R and IIM
                           on observed data: high synchrony, low
                           differentiation)
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

This module must not import ``impact_pipeline.mpc_metrics``.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Optional

import numpy as np
import pandas as pd

from impact_pipeline.bench.generators import (
    NOMINAL_KNOBS,
    OFF_KNOBS,
    PRINCIPLES,
    AgentConfig,
    BenchSystem,
    BinaryNetwork,
    _like_meta,
    _streams,
    binary_states,
    config_from_dict,
    hypersynchronous_system,
    sbn_to_sbs,
    simulate_family_a,
    stationary_distribution,
)

ADVERSARIAL_VERSION = "mpc-bench-adversarial/1.0.0"
ADVERSARIAL_KINDS = (
    "parity_grid",
    "hypersynchrony",
    "common_driver",
    "reflex_arc",
    "random_label_self_other",
    "scrambled_feedback",
)
EXPECTED_VERDICT = "EXCLUDED"

ADVERSARIAL_CATALOGUE = {
    "parity_grid": {
        "designed_to_fool": "IIM",
        "fooled_rules": ["IIM_only", "IIM_exact_tpm"],
        "intended_pattern": [0, 0, 0, 1, 0],
        "mechanism": "XOR (parity) units on a 3-regular circulant expander, "
        "resting in the all-off fixed point; no inputs or outputs",
        "claim": "Integrated-information measures computed on the causal "
        "structure (known TPM) are high for an inert parity grid that neither "
        "responds nor differentiates (Aaronson 2014 critique of IIT; blog "
        "post, not in the verified reference index).",
        "references": ["tononi2016integrated", "doerig2019unfolding"],
    },
    "hypersynchrony": {
        "designed_to_fool": "NAS",
        "fooled_rules": ["NAS_only", "PhiR", "IIM_only"],
        "intended_pattern": [0, 0, 0, 0, 0],
        "mechanism": "one common oscillation with random positive gains plus "
        "small independent noise in every node",
        "claim": "Synchrony is not global availability: a seizure-like common "
        "oscillation is maximally synchronous and undifferentiated.",
        "references": ["tononi1994measure", "casali2013theoretically"],
    },
    "common_driver": {
        "designed_to_fool": "NAS",
        "fooled_rules": ["NAS_only", "PhiR"],
        "intended_pattern": [0, 0, 0, 0, 0],
        "mechanism": "hidden bursty driver reaching the declared hub with no "
        "delay and the periphery with longer delays; no edges between nodes",
        "claim": "A common driver produces ignition-like global co-activation "
        "and hub-leads-periphery lagged dependence without any receive-"
        "transform-return loop (pseudo-broadcast).",
        "references": ["dehaene2011experimental", "deco2017hierarchy"],
    },
    "reflex_arc": {
        "designed_to_fool": "RAM",
        "fooled_rules": ["union", "count_1"],
        "intended_pattern": [0, 0, 0, 0, 0],
        "mechanism": "family-A agent with every switch off and a fixed reflex "
        "gain from stimulus a to action a (feedforward, non-plastic)",
        "claim": "Responsiveness is not adaptation: a reflex arc answers every "
        "stimulus reliably but never changes its mapping with feedback.",
        "references": ["oizumi2014phenomenology", "casali2013theoretically"],
    },
    "random_label_self_other": {
        "designed_to_fool": "SRPI",
        "fooled_rules": ["union", "count_4", "arithmetic_mean"],
        "intended_pattern": [1, 1, 1, 1, 0],
        "mechanism": "nominal agent without efference copy (e = 0); within "
        "each yoked pair the self_caused / other_caused labels are dealt by a "
        "fair coin (stimulus identity and phase matching kept)",
        "claim": "Self/other separability must come from the efference "
        "mechanism, not from the order of events or the preceding action.",
        "references": ["blakemore1998central", "blakemore2000cant"],
    },
    "scrambled_feedback": {
        "designed_to_fool": "RAM",
        "fooled_rules": ["union", "count_4", "arithmetic_mean"],
        "intended_pattern": [0, 1, 1, 1, 1],
        "mechanism": "nominal agent whose +-1 feedback has probability 1/2 "
        "whatever the choice; the gains keep updating on uninformative "
        "feedback",
        "claim": "Feedback-locked plasticity without action-outcome "
        "contingency is not goal-directed adaptation.",
        "references": ["pennartz2019indicators", "butlin2023consciousness"],
    },
}


# ---------------------------------------------------------------------------
# (i) Aaronson-style parity expander
# ---------------------------------------------------------------------------


def circulant_neighbours(n: int, topology: str = "mobius") -> list:
    """
    Neighbour lists of the parity graph: ``'mobius'`` (default) the 3-regular
    circulant C_n(1, n/2) (Moebius ladder; n even, n >= 4), a small expander;
    ``'ring'`` the 2-regular cycle (n >= 3).
    """
    n = int(n)
    if topology == "mobius":
        if n < 4 or n % 2:
            raise ValueError("the Moebius ladder needs an even n >= 4")
        return [sorted({(i - 1) % n, (i + 1) % n, (i + n // 2) % n}) for i in range(n)]
    if topology == "ring":
        if n < 3:
            raise ValueError("the ring needs n >= 3")
        return [sorted({(i - 1) % n, (i + 1) % n}) for i in range(n)]
    raise ValueError("topology must be 'mobius' or 'ring'")


def parity_network(
    n: int = 6, topology: str = "mobius", noise: float = 0.0
) -> BinaryNetwork:
    """
    Exact parity network: ``s'_i = XOR_{j in N(i)} s_j``, flipped with
    probability ``noise``. The all-off state is a fixed point for
    ``noise = 0``. States use the family-B big-endian order.
    """
    if not 0.0 <= float(noise) <= 0.5:
        raise ValueError("noise must be in [0, 0.5]")
    nbrs = circulant_neighbours(n, topology)
    n = len(nbrs)
    states = binary_states(n)
    sbn = np.zeros((states.shape[0], n))
    cm = np.zeros((n, n), dtype=int)
    for i, nb in enumerate(nbrs):
        par = np.zeros(states.shape[0], dtype=np.int16)
        for j in nb:
            par ^= states[:, j]
            cm[j, i] = 1
        sbn[:, i] = np.where(par == 1, 1.0 - noise, noise)
    tpm = sbn_to_sbs(sbn, states)
    return BinaryNetwork(
        kind=f"parity_{topology}",
        n=n,
        states=states,
        tpm=tpm,
        tpm_sbn=sbn,
        connectivity=cm,
        params={"kind": f"parity_{topology}", "n": n, "noise": float(noise)},
        stationary=stationary_distribution(tpm),
    )


def parity_grid_system(
    like: Optional[BenchSystem] = None,
    seed: int = 0,
    n: int = 6,
    topology: str = "mobius",
    measurement_sd: float = 0.05,
    config: Optional[AgentConfig] = None,
) -> BenchSystem:
    """
    Inert parity grid observed through small independent measurement noise:
    the network rests in its all-off fixed point (noise-free parity
    dynamics, no inputs), so the recorded ``ts`` (``n`` units x the template
    duration) is ``0 + measurement_sd * N(0, 1)``. ``meta['exact_tpm']``
    declares the known TPM (state-by-state, big-endian) and
    ``meta['exact_tpm_state']`` the current state (0 = all off) for
    IIT-style analyses; the events table of the template is kept so
    event-locked estimators are defined (the system ignores the events).
    """
    like = like if like is not None else simulate_family_a(None, config, seed)
    net = parity_network(n, topology, noise=0.0)
    rng = _streams(seed, 8)[6]
    T = like.n_time
    state = np.zeros((net.n, T))
    ts = state + float(measurement_sd) * rng.standard_normal((net.n, T))
    half = net.n // 2
    extra = {
        "adversarial": "parity_grid",
        "substrate": "ising_exact",
        "n_nodes": int(net.n),
        "n_time": int(T),
        "modules": {"P0": list(range(half)), "P1": list(range(half, net.n))},
        "module_order": ["P0", "P1"],
        "workspace_nodes": [0],
        "bearer_nodes": list(range(net.n)),
        "iim_macro_nodes": {f"u{i}": [i] for i in range(min(net.n, 4))},
        "iim_grain": "units",
        "exact_tpm": net.tpm.tolist(),
        "exact_tpm_state": 0,
        "exact_tpm_kind": net.kind,
        "measurement_sd": float(measurement_sd),
        "adversarial_version": ADVERSARIAL_VERSION,
    }
    meta = _like_meta(like, "adversarial_parity_grid", extra)
    oracle = _adversarial_oracle(
        "parity_grid",
        seed,
        {
            "hidden_state": state,
            "connectivity": net.connectivity,
            "fixed_point": [0] * net.n,
        },
    )
    return BenchSystem(ts=ts, events=like.events.copy(), meta=meta, oracle=oracle)


# ---------------------------------------------------------------------------
# (ii) Hypersynchrony and (iii) common-driver pseudo-broadcast
# ---------------------------------------------------------------------------


def hypersynchrony_system(
    like: Optional[BenchSystem] = None,
    seed: int = 0,
    freq_hz: float = 3.0,
    noise_sd: float = 0.05,
    config: Optional[AgentConfig] = None,
) -> BenchSystem:
    """Strong common oscillation in every node (template layout and events)."""
    like = like if like is not None else simulate_family_a(None, config, seed)
    s = hypersynchronous_system(like, seed=seed, freq_hz=freq_hz, noise_sd=noise_sd)
    s.meta.update(
        {
            "family": "adversarial_hypersynchrony",
            "adversarial": "hypersynchrony",
            "adversarial_version": ADVERSARIAL_VERSION,
        }
    )
    common = s.oracle.get("common_signal")
    s.oracle = _adversarial_oracle("hypersynchrony", seed, {"common_signal": common})
    return s


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
# (iv) Reflex arc, (v) random self/other labels, (vi) scrambled feedback
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
    if kind == "parity_grid":
        return parity_grid_system(seed=seed, config=cfg, **kwargs)
    if kind == "hypersynchrony":
        return hypersynchrony_system(seed=seed, config=cfg, **kwargs)
    if kind == "common_driver":
        return common_driver_system(seed=seed, config=cfg, **kwargs)
    if kind == "reflex_arc":
        return reflex_arc_system(cfg, seed, **kwargs)
    if kind == "random_label_self_other":
        return random_label_self_other_system(cfg, seed, **kwargs)
    return scrambled_feedback_system(cfg, seed, **kwargs)


def catalogue_table() -> pd.DataFrame:
    """The catalogue as a table (one row per kind)."""
    rows = []
    for k in ADVERSARIAL_KINDS:
        e = ADVERSARIAL_CATALOGUE[k]
        rows.append(
            {
                "kind": k,
                "designed_to_fool": e["designed_to_fool"],
                "fooled_rules": ",".join(e["fooled_rules"]),
                "intended_pattern": "".join(str(b) for b in e["intended_pattern"]),
                "expected_verdict": EXPECTED_VERDICT,
                "mechanism": e["mechanism"],
            }
        )
    return pd.DataFrame(rows)
