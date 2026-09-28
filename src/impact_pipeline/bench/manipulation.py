"""
Preregistered manipulation checks of the MPC-Bench mechanism switches.

A switch is *effective* when switching its mechanism off changes that
mechanism's **oracle signature** (a statistic of hidden generator channels
that the estimators never see) by at least a declared relative amount at the
nominal dose, and the signature is not negligible at the nominal dose
(declared floor). The checks read only ``system.oracle`` (and, for family A,
whose hidden state is the recorded state without measurement noise, the
recorded array), never an MPC estimator, so they validate the generator
design without looking at metric values (and can be run on the held-out
family C before the code freeze without unblinding it).

Oracle signatures (``H`` = hidden state: family A the unit activity ``x``;
family C the demodulated complex envelope ``z_j e^{-i w_j t}``; module
signals are module means of ``H``; Granger gains use ``GRANGER_LAGS`` lags of
real and imaginary parts):

=========  ===================  ===============================================
switch     signature            definition
=========  ===================  ===============================================
eta (RAM)  reversal_tracking    corr(P(choice = 1), rewarded arm) over bandit
                                trials (hidden choice probabilities follow the
                                reversals only through feedback plasticity)
K (PDI)    context_information  fraction of the variance of the smoothed unit
                                state explained by the active context pattern
                                (between-context R^2; 0 with one context)
g_b (NAS)  receive_return       min(mean Granger gain hub <- periphery, mean
                                gain periphery <- hub): both directions of the
                                workspace loop (``ff_only`` removes receive)
c_int      backward_dependence  mean Granger gain of each upstream periphery
(IIM)                           module from each downstream module
e (SRPI)   self_other_contrast  ||R_self - R_other|| / (||R_self|| +
                                ||R_other||) of the mean evoked sensory-module
                                response to self-caused events and their
                                yoked, stimulus-identical replays
=========  ===================  ===============================================

Relative change = (s_nominal - s_off) / |s_nominal|. A check passes when
``relative_change >= RELATIVE_CHANGE_THRESHOLD[switch]`` and ``s_nominal >=
NOMINAL_FLOOR[switch]``. Thresholds are fixed here before any confirmatory
run (``MANIPULATION_CHECK_VERSION``).

This module must not import ``impact_pipeline.mpc_metrics``.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from impact_pipeline.bench.generators import (
    NOMINAL_KNOBS,
    AgentConfig,
    BenchSystem,
    Knobs,
)

MANIPULATION_CHECK_VERSION = "mpc-bench-manipulation/1.0.0"
GRANGER_LAGS = 5
SMOOTH_SEC = 0.5
RESPONSE_WINDOW_SEC = 0.4
PRE_WINDOW_SEC = 0.2

# switch name -> (principle, signature, knob changes that switch it off)
SWITCHES = {
    "eta": ("RAM", "reversal_tracking", {"eta": 0.0}),
    "K": ("PDI", "context_information", {"K": 1}),
    "g_b": ("NAS", "receive_return", {"g_b": 0.0}),
    "ff_only": ("NAS", "receive_return", {"ff_only": True}),
    "c_int": ("IIM", "backward_dependence", {"c_int": 0.0}),
    "e": ("SRPI", "self_other_contrast", {"e": 0.0}),
}
RELATIVE_CHANGE_THRESHOLD = {
    "eta": 0.5,
    "K": 0.5,
    "g_b": 0.5,
    "ff_only": 0.5,
    "c_int": 0.5,
    "e": 0.5,
}
NOMINAL_FLOOR = {
    "eta": 0.2,
    "K": 0.05,
    "g_b": 0.002,
    "ff_only": 0.002,
    "c_int": 0.002,
    "e": 0.1,
}


# ---------------------------------------------------------------------------
# Hidden channels
# ---------------------------------------------------------------------------


def hidden_state(system: BenchSystem, frame: str = "lab") -> np.ndarray:
    """
    Hidden unit state (units x time). Rate families: the unit activity.
    Oscillator families (complex): ``frame='lab'`` the state ``z`` itself,
    ``frame='envelope'`` the demodulated envelope ``z_j e^{-i w_j t}`` (each
    node in the frame rotating at its own natural frequency).
    """
    if frame not in ("lab", "envelope"):
        raise ValueError("frame must be 'lab' or 'envelope'")
    env = system.oracle.get("hidden_envelope")
    if env is None:
        return np.asarray(system.ts, dtype=float)
    env = np.asarray(env, dtype=complex)
    if frame == "envelope":
        return env
    freq = np.asarray(system.oracle["oscillator_frequency_hz"], dtype=float)
    t = (np.arange(env.shape[1]) + 1) * float(system.dt)
    return env * np.exp(2j * np.pi * freq[:, None] * t[None, :])


def _real_features(x: np.ndarray) -> np.ndarray:
    """Rows of a (possibly complex) signal as real features (Re, Im)."""
    x = np.atleast_2d(x)
    if np.iscomplexobj(x):
        return np.concatenate([x.real, x.imag], axis=0)
    return x


def _module_signals(system: BenchSystem) -> Dict[str, np.ndarray]:
    h = hidden_state(system, "lab")
    return {
        name: h[np.asarray(idx, dtype=int)].mean(axis=0)
        for name, idx in system.meta["modules"].items()
    }


def _standardise(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return x / np.where(sd > 0, sd, 1.0)


def _lagged(x: np.ndarray, lags: int) -> np.ndarray:
    """Design matrix of lags 1..L of the rows of ``x`` (time x features)."""
    t = x.shape[1]
    return np.concatenate([x[:, lags - k : t - k].T for k in range(1, lags + 1)], 1)


def granger_gain(target, source, lags: int = GRANGER_LAGS) -> float:
    """
    Fraction of the residual variance of ``target`` (given its own ``lags``
    past samples) explained by ``lags`` past samples of ``source``; both may
    be complex (real and imaginary parts are separate regressors/targets).
    OLS with an intercept; residual variances are degrees-of-freedom adjusted
    (``RSS / (N - k)``), so the gain of an unrelated source is centred at 0
    whatever the run length (before the floor: negative values are set to 0,
    so its mean is slightly positive, of order ``sqrt(k) / N``).
    """
    y = _standardise(_real_features(target))
    s = _standardise(_real_features(source))
    t = y.shape[1]
    Y = y[:, lags:].T
    own = np.c_[np.ones(t - lags), _lagged(y, lags)]
    both = np.c_[own, _lagged(s, lags)]
    n_obs = Y.shape[0]
    if n_obs <= both.shape[1] + 1:
        return float("nan")
    r_own = Y - own @ np.linalg.lstsq(own, Y, rcond=None)[0]
    r_both = Y - both @ np.linalg.lstsq(both, Y, rcond=None)[0]
    var_own = float(np.sum(r_own**2)) / (n_obs - own.shape[1])
    var_both = float(np.sum(r_both**2)) / (n_obs - both.shape[1])
    if var_own <= 0:
        return 0.0
    return float(max(0.0, 1.0 - var_both / var_own))


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def reversal_tracking(system: BenchSystem) -> float:
    o = system.oracle
    p = np.asarray(o.get("p_choice1", []), dtype=float)
    good = np.asarray(o.get("good_arm", []), dtype=float)[: p.size]
    if p.size < 3 or np.std(p) == 0 or np.std(good) == 0:
        return 0.0
    return float(np.corrcoef(p, good)[0, 1])


def context_information(system: BenchSystem) -> float:
    """Between-context share of the variance of the smoothed unit state."""
    labels = np.asarray(system.oracle.get("context_state"), dtype=int)
    x = _real_features(hidden_state(system, "envelope"))
    w = max(1, int(round(SMOOTH_SEC / system.dt)))
    if w > 1:
        kern = np.ones(w) / w
        x = np.stack([np.convolve(r, kern, mode="same") for r in x])
    x = x - x.mean(axis=1, keepdims=True)
    total = float(np.sum(x * x))
    if total <= 0 or np.unique(labels).size < 2:
        return 0.0
    between = 0.0
    for k in np.unique(labels):
        sel = labels == k
        mk = x[:, sel].mean(axis=1)
        between += float(sel.sum()) * float(np.sum(mk * mk))
    return between / total


def _periphery(system: BenchSystem) -> List[str]:
    return [m for m in system.meta["module_order"] if m != "W"]


def receive_return(system: BenchSystem) -> float:
    sig = _module_signals(system)
    hub = sig["W"]
    periph = _periphery(system)
    receive = np.mean([granger_gain(hub, sig[p]) for p in periph])
    ret = np.mean([granger_gain(sig[p], hub) for p in periph])
    return float(min(receive, ret))


def backward_dependence(system: BenchSystem) -> float:
    sig = _module_signals(system)
    periph = _periphery(system)
    gains = [
        granger_gain(sig[periph[i]], sig[periph[j]])
        for i in range(len(periph))
        for j in range(i + 1, len(periph))
    ]
    return float(np.mean(gains)) if gains else 0.0


def self_other_contrast(system: BenchSystem) -> float:
    """
    Noise-corrected contrast between the evoked sensory-module responses to
    self-caused events and their yoked replays: with paired differences
    ``d_i = r(self_i) - r(other_i)`` and sums ``s_i``, ``o_i`` averaged over
    two disjoint halves of the pairs (even / odd), squared norms are
    estimated without noise bias by inner products across halves:
    ``sqrt(max(d1.d2, 0)) / (sqrt(max(s1.s2, 0)) + sqrt(max(o1.o2, 0)))``.
    About 0 when self and other responses do not differ.
    """
    ev = system.events
    dt = system.dt
    h = hidden_state(system, "envelope")[
        np.asarray(system.meta["modules"]["S"], dtype=int)
    ]
    pre = max(1, int(round(PRE_WINDOW_SEC / dt)))
    post = max(1, int(round(RESPONSE_WINDOW_SEC / dt)))

    def _response(onset):
        i = int(round(float(onset) / dt))
        if i - pre < 0 or i + post > h.shape[1]:
            return None
        return h[:, i : i + post].mean(axis=1) - h[:, i - pre : i].mean(axis=1)

    selfs = ev[ev.trial_type == "self_caused"].set_index("event_id")["onset"]
    pairs = []
    for _, row in ev[ev.trial_type == "other_caused"].iterrows():
        key = row.get("yoked_to")
        if key is None or key not in selfs.index:
            continue
        rs, ro = _response(selfs.loc[key]), _response(row["onset"])
        if rs is not None and ro is not None:
            pairs.append((rs, ro))
    if len(pairs) < 4:
        return 0.0
    halves = (pairs[0::2], pairs[1::2])
    d = [np.mean([a - b for a, b in hv], axis=0) for hv in halves]
    s = [np.mean([a for a, _ in hv], axis=0) for hv in halves]
    o = [np.mean([b for _, b in hv], axis=0) for hv in halves]

    def _dot(u, v):
        return float(np.real(np.vdot(u, v)))

    den = np.sqrt(max(_dot(*s), 0.0)) + np.sqrt(max(_dot(*o), 0.0))
    if den <= 0:
        return 0.0
    return float(np.sqrt(max(_dot(*d), 0.0)) / den)


SIGNATURES: Dict[str, Callable[[BenchSystem], float]] = {
    "reversal_tracking": reversal_tracking,
    "context_information": context_information,
    "receive_return": receive_return,
    "backward_dependence": backward_dependence,
    "self_other_contrast": self_other_contrast,
}


def oracle_signatures(system: BenchSystem, names: Optional[Iterable[str]] = None):
    """All (or the named) oracle signatures of one system."""
    names = list(SIGNATURES) if names is None else list(names)
    return {k: SIGNATURES[k](system) for k in names}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def relative_change(nominal: float, off: float) -> float:
    if not np.isfinite(nominal) or nominal == 0:
        return float("nan")
    return float((nominal - off) / abs(nominal))


def manipulation_check(
    simulate: Callable[..., BenchSystem],
    seed: int,
    config: Optional[AgentConfig] = None,
    switches: Iterable[str] = tuple(SWITCHES),
    nominal: Knobs = None,
) -> pd.DataFrame:
    """
    Manipulation check of ``switches`` for one generator (``simulate(knobs,
    config, seed)``, e.g. ``simulate_family_c``) and seed: one row per switch
    with the nominal and switched-off signature, the relative change, the
    declared threshold and floor, and ``passed``.
    """
    base = nominal if nominal is not None else NOMINAL_KNOBS
    nom = simulate(base, config, int(seed))
    rows = []
    cache = {}
    for sw in switches:
        principle, sig_name, off_kw = SWITCHES[sw]
        if sig_name not in cache:
            cache[sig_name] = SIGNATURES[sig_name](nom)
        s_nom = cache[sig_name]
        off = simulate(base.replace(**off_kw), config, int(seed))
        s_off = SIGNATURES[sig_name](off)
        rc = relative_change(s_nom, s_off)
        thr = RELATIVE_CHANGE_THRESHOLD[sw]
        floor = NOMINAL_FLOOR[sw]
        rows.append(
            {
                "switch": sw,
                "principle": principle,
                "signature": sig_name,
                "seed": int(seed),
                "nominal": s_nom,
                "off": s_off,
                "relative_change": rc,
                "threshold": thr,
                "floor": floor,
                "passed": bool(np.isfinite(rc) and rc >= thr and s_nom >= floor),
                "family": nom.meta.get("family"),
                "check_version": MANIPULATION_CHECK_VERSION,
            }
        )
    return pd.DataFrame(rows)


def manipulation_report(
    simulate: Callable[..., BenchSystem],
    seeds: Iterable[int],
    config: Optional[AgentConfig] = None,
    switches: Iterable[str] = tuple(SWITCHES),
) -> pd.DataFrame:
    """:func:`manipulation_check` over several seeds (rows concatenated)."""
    return pd.concat(
        [manipulation_check(simulate, s, config, switches) for s in seeds],
        ignore_index=True,
    )
