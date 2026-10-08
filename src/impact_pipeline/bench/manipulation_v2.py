"""
Realisation checks of the MPC-Bench v2 systems (``mpc-bench-manipulation/
1.1.0``: additions to the frozen 1.0.0 switch checks of
:mod:`impact_pipeline.bench.manipulation`, whose signatures, thresholds and
floors are unchanged and reused here).

A new system is *realised* when its defining mechanism is present or absent
in the generator's hidden channels as stated. The checks read only the
system's oracle, its declared layout and (family A, whose hidden state is the
recorded state) the recorded array, never an MPC estimator, so they can run
on held-out conditions before the freeze.

Gated checks (a system is usable in a family iff its checks pass in at least
90 % of the seeds, 36 of 40; :func:`summarise_realisation`):

===========================  =======================  =======================
check                        systems                  passes iff
===========================  =======================  =======================
no_hub_periphery_path        N_modules_disconnected   no directed path between
                                                      a hub unit and a
                                                      periphery unit, either
                                                      way, in the coupling
                                                      matrix (deterministic)
no_coupling                  N_uncoupled              the coupling matrix is
                                                      zero (deterministic)
no_ignition                  W_PDI_no_multistability  ignition occupancy
                                                      < 0.05
driver_reaches_every_module  ADV_NAS_staggered_*      per-module context
                                                      information of the
                                                      driver >= the v1 floor
                                                      of the repertoire check
                                                      (0.05) in every module
stated_time_constants        ADV_NAS_staggered_*      the filtered drive
                                                      realises each module's
                                                      stated time constant
                                                      and the stated
                                                      transform
slow_context_dwell           PC_nominal under the     every complete context
                             slow_context_bold        run lasts at least 30 s
                             preset (the forward      and the preset's trial
                             BOLD arm; family A)      timing is in force
===========================  =======================  =======================

The slow-context check is the realisation check of the slow-context BOLD
agents (design 3.4). Design 3.5 does not list it; it is a gate because a
failed realisation means that the condition of the forward BOLD arm was
not produced. Its check id is
``PC_nominal:slow_context_bold/slow_context_dwell``, and the hypotheses
file requires it only for the rows of that arm (``oracle_checks``).

:func:`prerequisite_m` collects the data of the prerequisite M: the frozen
1.0.0 switch checks on families A and C1 together with these checks, and
the usability of every switch and new system per family, and the reported
checks below.

Reported checks (never gates; rows with ``gate`` False): PC_half lies
between the switched-off and the nominal medians of each 1.0.0 signature
(:func:`pc_half_check` per seed, :func:`pc_half_summary` per family and
switch); twins share the structural hash and differ in the schedule hash,
and replicate 0 is the witness run itself (:func:`twin_check`; the
integrity audit IA-10 checks the same on the records).

Ignition occupancy is the share of samples in which the hub's all-or-none
amplification is on: the ignition gate ``g_b s_ign`` above half its full
value ``g_b``, counted only when the amplification gain (``w_ign``; family C
``sl_ignition_gain``) times ``g_b`` is positive. With the amplification
removed there is no ignition, wherever the hub's activity sits; the share of
samples with the gate above half (the v1 ``summarise_oracle`` occupancy,
which then only says whether the hub's activity exceeds the ignition
threshold) is reported as ``hub_above_threshold_share``.

The structural and schedule hashes (:func:`structural_hash`,
:func:`schedule_hash`) are also the twin hashes of the v2 result schema.

This module must not import ``impact_pipeline.mpc_metrics``.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from impact_pipeline.bench import manipulation as M
from impact_pipeline.bench.adversarial_v2 import (
    STAGGERED_SYSTEM_VARIANTS,
    build_system,
    config_preset,
    get_entry,
    staggered_drive_input,
    system_ids,
)
from impact_pipeline.bench.generators import (
    NOMINAL_KNOBS,
    AgentConfig,
    BenchSystem,
    config_from_dict,
    knobs_from_dict,
    simulate_family_a,
    simulate_family_c,
)
from impact_pipeline.v2.seeds import check_seeds

MANIPULATION_CHECK_VERSION_V2 = "mpc-bench-manipulation/1.1.0"
BASE_CHECK_VERSION = M.MANIPULATION_CHECK_VERSION
USABLE_PASS_SHARE = 0.9
NEAR_THRESHOLD_BAND = 0.05
IGNITION_OCCUPANCY_MAX = 0.05
CONTEXT_FLOOR = M.NOMINAL_FLOOR["K"]
TIME_CONSTANT_RTOL = 1e-6
SLOW_CONTEXT_MIN_DWELL_SEC = 30.0
SLOW_CONTEXT_PRESET = "slow_context_bold"
SLOW_CONTEXT_SYSTEM = "PC_nominal"
SLOW_CONTEXT_CHECK = "slow_context_dwell"
SLOW_CONTEXT_FAMILIES = ("A",)
SLOW_CONTEXT_CHECK_ID = (f"{SLOW_CONTEXT_SYSTEM}:{SLOW_CONTEXT_PRESET}/"
                         f"{SLOW_CONTEXT_CHECK}")
# the confirmatory twin replicates of families A and C1 (design 3.2)
TWIN_REPLICATES = tuple(range(1, 7))
TWIN_CHECK = "twin_hashes"
PC_HALF_SYSTEM = "PC_half"
PC_HALF_CHECK = "between_off_and_nominal"
REALISATION_COLUMNS = ("system_id", "family", "seed", "variant", "check", "value",
                       "threshold", "passed", "gate", "details", "check_version")

STAGGERED_IDS = tuple(STAGGERED_SYSTEM_VARIANTS)
CHECKS_BY_SYSTEM = {
    "N_modules_disconnected": ("no_hub_periphery_path",),
    "N_uncoupled": ("no_coupling",),
    "W_PDI_no_multistability": ("no_ignition",),
    **{sid: ("driver_reaches_every_module", "stated_time_constants")
       for sid in STAGGERED_IDS},
}
EXOGENOUS_EVENT_TYPES = ("goal_cue", "stimulus", "action", "self_caused",
                         "other_caused")


# ---------------------------------------------------------------------------
# Hashes (twins; result schema)
# ---------------------------------------------------------------------------


def _canonical(obj):
    if isinstance(obj, Mapping):
        items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return {str(k): _canonical(v) for k, v in items}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _digest(parts: Sequence[tuple]) -> str:
    h = hashlib.sha256()
    for name, value in parts:
        h.update(f"<{name}>".encode("utf-8"))
        if isinstance(value, np.ndarray) and value.dtype.hasobject:
            # The bytes of an object array are pointers, which differ between
            # processes: hash its values instead.
            value = value.tolist()
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            h.update(f"{arr.dtype.str}|{arr.shape}|".encode("ascii"))
            h.update(arr.tobytes())
        else:
            h.update(json.dumps(_canonical(value), sort_keys=True,
                                allow_nan=True).encode("utf-8"))
    return h.hexdigest()


STRUCTURAL_META_KEYS = ("n_nodes", "dt", "dynamics", "module_order", "modules",
                        "workspace_nodes")
STRUCTURAL_ORACLE_KEYS = ("unit_adjacency", "module_adjacency",
                          "oscillator_frequency_hz", "driver_patterns")
SCHEDULE_ORACLE_KEYS = ("context_state", "slow_phase", "good_arm")


def structural_hash(system: BenchSystem) -> str:
    """SHA-256 of the realised network: declared layout (nodes, ``dt``,
    modules, workspace) and the hidden structure present in the oracle (unit
    and module coupling, oscillator frequencies, driver patterns). Twins of a
    run share it. It is not a system identity: systems of one seed that
    differ only outside these channels share it too (knobs that leave the
    coupling unchanged, such as ``eta``, ``K`` or ``e``; configuration
    fields such as ``w_ign``; the time constants and transform of the
    staggered-driver variants), so twin sets are formed by system and seed,
    and the hash checks them."""
    meta, o = system.meta, system.oracle
    parts = [(f"meta.{k}", meta.get(k)) for k in STRUCTURAL_META_KEYS]
    for k in STRUCTURAL_ORACLE_KEYS:
        if o.get(k) is not None:
            parts.append((f"oracle.{k}", np.asarray(o[k])))
    return _digest(parts)


def schedule_hash(system: BenchSystem) -> str:
    """SHA-256 of the realised task schedule: every event's onset, duration
    and type, the values of the exogenous events (cues, stimuli, actions and
    their sensory consequences; not choices or rewards, which depend on the
    dynamics), and the hidden schedule channels in the oracle (context or
    driver level per sample, slow-rhythm phase, rewarded arm). Of the knobs
    it depends only on the repertoire size ``K``, which maps the context
    draws to patterns; twins of a run differ in it."""
    ev = system.events
    exo = ev["trial_type"].isin(EXOGENOUS_EVENT_TYPES).to_numpy()
    parts = [
        ("events.onset", ev["onset"].to_numpy(dtype=float)),
        ("events.duration", ev["duration"].to_numpy(dtype=float)),
        ("events.trial_type", ev["trial_type"].astype(str).tolist()),
        ("events.exogenous_value",
         [float(v) if e else None for v, e in zip(ev["value"].tolist(), exo)]),
    ]
    for k in SCHEDULE_ORACLE_KEYS:
        if system.oracle.get(k) is not None:
            parts.append((f"oracle.{k}", np.asarray(system.oracle[k])))
    return _digest(parts)


# ---------------------------------------------------------------------------
# Signatures of the new systems
# ---------------------------------------------------------------------------


def _reach(adjacency: np.ndarray, sources: Sequence[int]) -> np.ndarray:
    """Nodes reachable from ``sources`` along ``adjacency[source, target] !=
    0`` (sources included)."""
    a = np.asarray(adjacency) != 0
    seen = np.zeros(a.shape[0], dtype=bool)
    stack = [int(s) for s in sources]
    seen[stack] = True
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(a[i] & ~seen):
            seen[j] = True
            stack.append(int(j))
    return seen


def hub_periphery_paths(system: BenchSystem) -> dict:
    """Directed paths between the declared hub units and the periphery units
    in the unit coupling matrix (``oracle['unit_adjacency']``, source x
    target): ``hub_to_periphery``, ``periphery_to_hub`` and the number of
    hub-periphery unit pairs connected either way."""
    adj = np.asarray(system.oracle["unit_adjacency"])
    hub = [int(i) for i in system.meta["workspace_nodes"]]
    periph = [int(i) for name, idx in system.meta["modules"].items()
              for i in idx if int(i) not in hub]
    from_hub = _reach(adj, hub)
    to_hub = _reach(adj.T, hub)
    pairs = 0
    for p in periph:
        pairs += int(from_hub[p]) + int(to_hub[p])
    direct = 0.0
    if periph and hub:
        direct = float(max(np.max(np.abs(adj[np.ix_(hub, periph)])),
                           np.max(np.abs(adj[np.ix_(periph, hub)]))))
    return {
        "hub_to_periphery": bool(from_hub[periph].any()),
        "periphery_to_hub": bool(to_hub[periph].any()),
        "connected_pairs": int(pairs),
        "max_abs_direct_coupling": direct,
    }


def _amplification_gain(system: BenchSystem) -> float:
    cfg = system.meta.get("config") or {}
    oscillators = system.meta.get("dynamics") == "stuart_landau"
    key = "sl_ignition_gain" if oscillators else "w_ign"
    return float(cfg.get(key, 0.0) or 0.0)


def ignition_occupancy(system: BenchSystem) -> dict:
    """Ignition occupancy (module docstring) and the v1 share of samples
    with the ignition gate above half (``hub_above_threshold_share``)."""
    gate = np.asarray(system.oracle.get("ignition_gate", []), dtype=float)
    g_b = float((system.meta.get("knobs") or {}).get("g_b", 0.0) or 0.0)
    gain = _amplification_gain(system)
    above = float(np.mean(gate > 0.5 * g_b)) if g_b > 0 and gate.size else 0.0
    occ = above if g_b * gain > 0 else 0.0
    return {"ignition_occupancy": occ, "hub_above_threshold_share": above,
            "amplification_gain": gain * g_b}


def module_context_information(
    system: BenchSystem, labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Per module, the between-label share of the variance of the smoothed
    hidden unit state (the 1.0.0 ``context_information`` restricted to the
    module's units, real and imaginary parts of an oscillator envelope as
    separate features; labels default to ``oracle['context_state']``)."""
    if labels is None:
        labels = system.oracle["context_state"]
    labels = np.asarray(labels, dtype=int)
    h = M.hidden_state(system, "envelope")
    w = max(1, int(round(M.SMOOTH_SEC / system.dt)))
    kern = np.ones(w) / w
    uniq = np.unique(labels)
    out = {}
    for name in system.meta["module_order"]:
        idx = np.asarray(system.meta["modules"][name], dtype=int)
        xm = M._real_features(h[idx])
        if w > 1:
            xm = np.stack([np.convolve(r, kern, mode="same") for r in xm])
        xm = xm - xm.mean(axis=1, keepdims=True)
        total = float(np.sum(xm * xm))
        if total <= 0 or uniq.size < 2:
            out[name] = 0.0
            continue
        between = 0.0
        for k in uniq:
            sel = labels == k
            mk = xm[:, sel].mean(axis=1)
            between += float(sel.sum()) * float(np.sum(mk * mk))
        out[name] = between / total
    return out


def realised_time_constants(system: BenchSystem) -> Dict[str, float]:
    """Per module, the time constant realised by the filtered drive of a
    staggered-driver system: the least-squares ``a`` of ``y(t) - u(t) = a
    (y(t-1) - u(t))`` over the module's units, ``tau = -dt / ln a``."""
    y = np.asarray(system.oracle["drive_filtered"], dtype=float)
    u = staggered_drive_input(system)
    out = {}
    for name in system.meta["module_order"]:
        idx = np.asarray(system.meta["modules"][name], dtype=int)
        cur = y[idx, 1:] - u[idx, 1:]
        prev = y[idx, :-1] - u[idx, 1:]
        a = float(np.sum(cur * prev) / np.sum(prev * prev))
        out[name] = float(-system.dt / math.log(a)) if 0 < a < 1 else float("nan")
    return out


def _transform_error(system: BenchSystem) -> float:
    o = system.oracle
    y = np.asarray(o["drive_filtered"], dtype=float)
    x = np.asarray(o["hidden_signal"], dtype=float)
    want = np.tanh(float(o["saturation_gain"]) * y) if o["transform"] == "tanh" else y
    return float(np.max(np.abs(x - want)))


# ---------------------------------------------------------------------------
# Gated checks
# ---------------------------------------------------------------------------


def _row(system_id, family, seed, variant, check, value, threshold, passed,
         details=None) -> dict:
    return {
        "system_id": system_id,
        "family": family,
        "seed": int(seed),
        "variant": variant,
        "check": check,
        "value": value,
        "threshold": threshold,
        "passed": bool(passed),
        "gate": True,
        "details": details or {},
        "check_version": MANIPULATION_CHECK_VERSION_V2,
    }


def check_system(system: BenchSystem, system_id: str, seed: int, family: str = "A",
                 variant: Optional[str] = None) -> List[dict]:
    """The gated realisation checks of one built system (one row each)."""
    rows = []
    for check in CHECKS_BY_SYSTEM.get(system_id, ()):
        if check == "no_hub_periphery_path":
            p = hub_periphery_paths(system)
            rows.append(_row(system_id, family, seed, variant, check,
                             p["connected_pairs"], 0,
                             not (p["hub_to_periphery"] or p["periphery_to_hub"]), p))
        elif check == "no_coupling":
            v = float(np.max(np.abs(np.asarray(system.oracle["unit_adjacency"]))))
            rows.append(_row(system_id, family, seed, variant, check, v, 0.0, v == 0.0))
        elif check == "no_ignition":
            occ = ignition_occupancy(system)
            rows.append(_row(system_id, family, seed, variant, check,
                             occ["ignition_occupancy"], IGNITION_OCCUPANCY_MAX,
                             occ["ignition_occupancy"] < IGNITION_OCCUPANCY_MAX, occ))
        elif check == "driver_reaches_every_module":
            ci = module_context_information(system)
            low = min(ci.values())
            rows.append(_row(system_id, family, seed, variant, check, low,
                             CONTEXT_FLOOR, low >= CONTEXT_FLOOR, ci))
        elif check == "stated_time_constants":
            stated = dict(system.oracle["module_tau_sec"])
            real = realised_time_constants(system)
            rel = {k: abs(real[k] - stated[k]) / stated[k] for k in stated}
            terr = _transform_error(system)
            worst = max(rel.values())
            ok = bool(np.isfinite(worst) and worst <= TIME_CONSTANT_RTOL
                      and terr == 0.0)
            rows.append(_row(system_id, family, seed, variant, check, worst,
                             TIME_CONSTANT_RTOL, ok,
                             {"stated": stated, "realised": real,
                              "transform": system.oracle["transform"],
                              "transform_max_error": terr}))
        else:  # pragma: no cover - CHECKS_BY_SYSTEM names only the above
            raise ValueError(f"unknown check {check!r}")
    return rows


def _variants(system_id: str) -> List[Optional[str]]:
    return list(STAGGERED_SYSTEM_VARIANTS.get(system_id, (None,)))


def realisation_check(system_id: str, seed: int, family: str = "A",
                      variant: Optional[str] = None, config=None,
                      catalogue: Optional[Mapping] = None) -> pd.DataFrame:
    """Build one system and run its gated checks (one row per check)."""
    if system_id not in CHECKS_BY_SYSTEM:
        raise ValueError(f"no realisation check for {system_id!r}; one of "
                         f"{sorted(CHECKS_BY_SYSTEM)}")
    system = build_system(system_id, seed, family, variant=variant, config=config,
                          catalogue=catalogue)
    variant = system.meta.get("adversary_variant", variant)
    return pd.DataFrame(check_system(system, system_id, seed, family, variant))


def realisation_report(seeds: Iterable[int], system_ids: Optional[Sequence[str]] = None,
                       families: Sequence[str] = ("A", "C"), config=None,
                       catalogue: Optional[Mapping] = None) -> pd.DataFrame:
    """Gated checks of every new system (or ``system_ids``), in every family
    it is defined in among ``families``, for every variant and seed."""
    ids = list(system_ids) if system_ids is not None else list(CHECKS_BY_SYSTEM)
    frames = []
    for sid in ids:
        entry = get_entry(sid, catalogue)
        for fam in [f for f in families if f in entry["families"]]:
            for var in _variants(sid):
                for s in seeds:
                    frames.append(realisation_check(sid, int(s), fam, var, config,
                                                    catalogue))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _usability(df: pd.DataFrame, keys: Sequence[str], value: str,
               pass_share: float) -> pd.DataFrame:
    g = df.groupby(list(keys), sort=False)
    out = g.agg(n=("passed", "size"), passes=("passed", "sum"),
                min_value=(value, "min"), max_value=(value, "max")).reset_index()
    out["passes"] = out["passes"].astype(int)
    out["required"] = [int(math.ceil(pass_share * n - 1e-12)) for n in out["n"]]
    out["usable"] = out["passes"] >= out["required"]
    rate = out["passes"] / out["n"]
    out["near_threshold"] = (rate - pass_share).abs() <= NEAR_THRESHOLD_BAND
    return out


def summarise_realisation(df: pd.DataFrame, pass_share: float = USABLE_PASS_SHARE
                          ) -> pd.DataFrame:
    """Per (system, variant, family, check): seeds, passes and ``usable``
    (passes >= ceil(pass_share x seeds); 36 of 40 at the default), with
    ``near_threshold`` flagging pass rates within 0.05 of the share."""
    d = df.copy()
    d["variant"] = d["variant"].fillna("")
    out = _usability(d, ["system_id", "variant", "family", "check"], "value",
                     pass_share)
    out["check_version"] = MANIPULATION_CHECK_VERSION_V2
    return out


def slow_context_report(seeds: Iterable[int], families: Sequence[str] = ("A", "C"),
                        config=None, catalogue: Optional[Mapping] = None
                        ) -> pd.DataFrame:
    """The gated slow-context check (:func:`slow_context_check`) in the
    realisation-table columns, for every seed of every family among
    ``families`` that has the slow-context arm (family A)."""
    frames = [slow_context_check(int(s), family=f, config=config,
                                 catalogue=catalogue)[list(REALISATION_COLUMNS)]
              for f in families if f in SLOW_CONTEXT_FAMILIES for s in seeds]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def twin_systems(family: str, catalogue: Optional[Mapping] = None) -> List[str]:
    """The catalogue witnesses with twin sessions in ``family`` (``A``, or
    ``C`` for C1; the catalogue's ``twin_families``), in catalogue order."""
    return [sid for sid in system_ids(catalogue, kind="witness", family=family)
            if family in (get_entry(sid, catalogue).get("twin_families") or ())]


_SIMULATORS = {"A": simulate_family_a, "C": simulate_family_c}


def _unit(task: tuple) -> pd.DataFrame:
    """One unit of :func:`prerequisite_m` (a module-level function, so that
    a worker process can run it)."""
    kind, sid, fam, var, seed, config, catalogue = task
    if kind == "switch":
        cfg = None if config is None else _agent_config(config)
        return M.manipulation_check(_SIMULATORS[fam], seed, cfg)
    if kind == "realisation":
        return realisation_check(sid, seed, fam, var, config, catalogue)
    if kind == "slow_context":
        return slow_context_report([seed], (fam,), config, catalogue)
    if kind == "pc_half":
        return pc_half_check(seed, fam, config, catalogue)
    if kind == "twins":
        return twin_check(sid, seed, TWIN_REPLICATES, fam, config=config,
                          catalogue=catalogue)
    raise ValueError(f"unknown unit {kind!r}")  # pragma: no cover


def _units(seeds, families, config, catalogue, reported) -> Dict[str, List[tuple]]:
    """The units of every table, in the order of the sequential reports
    (switches: family, seed; realisation: system, family, variant, seed)."""
    out: Dict[str, List[tuple]] = {
        "switches": [("switch", None, f, None, s, config, catalogue)
                     for f in families for s in seeds],
        "realisation": [],
        "slow_context": [("slow_context", None, f, None, s, config, catalogue)
                         for f in families if f in SLOW_CONTEXT_FAMILIES
                         for s in seeds],
    }
    for sid in CHECKS_BY_SYSTEM:
        entry = get_entry(sid, catalogue)
        for fam in [f for f in families if f in entry["families"]]:
            for var in _variants(sid):
                out["realisation"] += [("realisation", sid, fam, var, s, config,
                                        catalogue) for s in seeds]
    if reported:
        out["pc_half"] = [("pc_half", None, f, None, s, config, catalogue)
                          for f in families for s in seeds]
        out["twins"] = [("twins", sid, f, None, s, config, catalogue)
                        for f in families for sid in twin_systems(f, catalogue)
                        for s in seeds]
    return out


def _run_units(units: Sequence[tuple], workers: int) -> List[pd.DataFrame]:
    if workers <= 1 or len(units) <= 1:
        return [_unit(u) for u in units]
    import concurrent.futures
    import multiprocessing

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_unit, units, chunksize=1))


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prerequisite_m(seeds: Iterable[int], families: Sequence[str] = ("A", "C"),
                   config=None, catalogue: Optional[Mapping] = None,
                   pass_share: float = USABLE_PASS_SHARE, *, reported: bool = True,
                   workers: int = 1) -> dict:
    """
    The data of the prerequisite M: the frozen 1.0.0 checks of every switch
    (eta, K, g_b, ff_only, c_int, e) on families A and C1 and the gated
    realisation checks of the new systems and of the slow-context arm, per
    seed (``realisation``: the new systems first, then the slow-context
    rows), and their usability per family (``summary``: one row per switch
    or system check). ``complete`` is True only if every gated check was
    computed (a finite value in every row).

    With ``reported`` (the default) the reported checks are added, never
    gates: ``pc_half`` (the PC_half, nominal and switched-off signature per
    seed), ``pc_half_summary`` (:func:`pc_half_summary`) and ``twins``
    (:func:`twin_check` of the twin witnesses of each family against
    replicates 1-6). ``workers`` > 1 runs the units on that many spawned
    processes; the tables are the same as with one.

    The seeds must satisfy the v2 seed policy (one split; ``split`` is
    derived from them).
    """
    seeds = [int(s) for s in seeds]
    split = check_seeds(seeds)
    families = tuple(families)
    units = _units(seeds, families, config, catalogue, reported)
    order = list(units)
    flat = [u for name in order for u in units[name]]
    results = _run_units(flat, int(workers))
    tables, i = {}, 0
    for name in order:
        n = len(units[name])
        tables[name] = results[i:i + n]
        i += n
    switches = _concat(tables["switches"])
    real = _concat(tables["realisation"] + tables["slow_context"])
    sw = _usability(switches, ["family", "switch"], "relative_change", pass_share)
    sw.insert(0, "kind", "switch")
    sw["check_version"] = BASE_CHECK_VERSION
    rs = summarise_realisation(real, pass_share)
    rs.insert(0, "kind", "new_system")
    values = np.r_[switches["relative_change"].to_numpy(dtype=float),
                   real["value"].to_numpy(dtype=float)]
    out = {
        "switches": switches,
        "realisation": real,
        "summary": pd.concat([sw, rs], ignore_index=True),
        "complete": bool(values.size and np.isfinite(values).all()),
        "check_versions": [BASE_CHECK_VERSION, MANIPULATION_CHECK_VERSION_V2],
        "seeds": seeds,
        "split": split,
    }
    if reported:
        pc = _concat(tables["pc_half"])
        out["pc_half"] = pc
        out["pc_half_summary"] = pc_half_summary(pc) if not pc.empty else pd.DataFrame()
        out["twins"] = _concat(tables["twins"])
    return out


# ---------------------------------------------------------------------------
# Reported checks
# ---------------------------------------------------------------------------


def pc_half_check(seed: int, family: str = "A", config=None,
                  catalogue: Optional[Mapping] = None) -> pd.DataFrame:
    """The 1.0.0 signature of every switch on PC_half, on the nominal run and
    on the run with that switch off (same seed: paired)."""
    simulate = simulate_family_a if family == "A" else simulate_family_c
    half = build_system("PC_half", seed, family, config=config, catalogue=catalogue)
    cfg_obj = None if config is None else _agent_config(config)
    nominal = simulate(NOMINAL_KNOBS, cfg_obj, int(seed))
    half_knobs = knobs_from_dict(get_entry("PC_half", catalogue)["knobs"]).to_dict()
    cache = {}
    rows = []
    for sw, (principle, sig, off_kw) in M.SWITCHES.items():
        if sig not in cache:
            cache[sig] = (M.SIGNATURES[sig](nominal), M.SIGNATURES[sig](half))
        s_nom, s_half = cache[sig]
        off = simulate(NOMINAL_KNOBS.replace(**off_kw), cfg_obj, int(seed))
        rows.append({
            "family": family, "seed": int(seed), "switch": sw, "principle": principle,
            "signature": sig, "nominal": s_nom, "half": s_half,
            "off": M.SIGNATURES[sig](off),
            "half_knobs": half_knobs,
            "check_version": MANIPULATION_CHECK_VERSION_V2,
        })
    return pd.DataFrame(rows)


def _agent_config(config):
    return config if isinstance(config, AgentConfig) else config_from_dict(dict(config))


def pc_half_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (family, switch): the medians over seeds of the switched-off,
    PC_half and nominal signatures and ``between`` (the PC_half median lies
    between the other two). Reported, never a gate: each row is also a
    reported check row (``system_id`` PC_half, ``variant`` the switch,
    ``check`` between_off_and_nominal, ``passed`` = ``between``, ``gate``
    False)."""
    g = df.groupby(["family", "switch", "signature"], sort=False)
    out = g.agg(n=("seed", "nunique"), median_off=("off", "median"),
                median_half=("half", "median"),
                median_nominal=("nominal", "median")).reset_index()
    lo = np.minimum(out["median_off"], out["median_nominal"])
    hi = np.maximum(out["median_off"], out["median_nominal"])
    out["between"] = (out["median_half"] >= lo) & (out["median_half"] <= hi)
    out["gate"] = False
    out["system_id"] = PC_HALF_SYSTEM
    out["variant"] = out["switch"]
    out["check"] = PC_HALF_CHECK
    out["passed"] = out["between"]
    out["check_version"] = MANIPULATION_CHECK_VERSION_V2
    return out


def twin_check(system_id: str, seed: int, replicates: Sequence[int] = (1, 2),
               family: str = "A", preset: Optional[str] = None, config=None,
               catalogue: Optional[Mapping] = None,
               reference: Optional[BenchSystem] = None) -> pd.DataFrame:
    """Twins of one witness run against its replicate 0: shared structural
    hash, different schedule hash and different series (one row per
    replicate; ``check`` twin_hashes, reported). ``reference`` (e.g. the
    stored witness run) is compared with replicate 0 bit for bit."""
    base = build_system(system_id, seed, family, replicate=0, preset=preset,
                        config=config, catalogue=catalogue)
    s0, d0 = structural_hash(base), schedule_hash(base)
    same_as_reference = None
    if reference is not None:
        same_as_reference = bool(np.array_equal(reference.ts, base.ts)
                                 and reference.events.equals(base.events))
    rows = []
    for r in replicates:
        twin = build_system(system_id, seed, family, replicate=int(r), preset=preset,
                            config=config, catalogue=catalogue)
        s1, d1 = structural_hash(twin), schedule_hash(twin)
        same_struct = s1 == s0
        new_sched = d1 != d0
        new_ts = not (twin.ts.shape == base.ts.shape
                      and np.array_equal(twin.ts, base.ts))
        rows.append({
            "system_id": system_id, "family": family, "seed": int(seed),
            "replicate": int(r), "structural_hash": s1, "schedule_hash": d1,
            "structural_hash_r0": s0, "schedule_hash_r0": d0,
            "same_structure": same_struct, "new_schedule": new_sched,
            "new_series": new_ts, "r0_equals_reference": same_as_reference,
            "passed": bool(same_struct and new_sched and new_ts
                           and same_as_reference is not False),
            "check": TWIN_CHECK,
            "gate": False, "check_version": MANIPULATION_CHECK_VERSION_V2,
        })
    return pd.DataFrame(rows)


def context_runs_sec(system: BenchSystem) -> np.ndarray:
    """Lengths (s) of the runs of constant context in ``oracle['context_state']``,
    without the last (truncated) run. A run can span several dwells when the
    same pattern is drawn twice."""
    st = np.asarray(system.oracle["context_state"])
    change = np.flatnonzero(np.diff(st) != 0) + 1
    edges = np.r_[0, change]
    return np.diff(edges) * float(system.dt)


def slow_context_check(seed: int, system_id: str = SLOW_CONTEXT_SYSTEM,
                       family: str = "A", catalogue: Optional[Mapping] = None,
                       config=None) -> pd.DataFrame:
    """The slow-context preset realised (gated): every complete context run
    lasts at least 30 s, and the preset's trial timing is in force. One row
    in the realisation-table columns (``variant`` is the preset; ``value``
    the shortest complete run in s, 0 when no context change completes a
    run), with ``n_runs``, ``shortest_run_sec``, ``median_run_sec`` and
    ``config_in_force`` also as columns of their own."""
    s = build_system(system_id, seed, family, preset=SLOW_CONTEXT_PRESET,
                     config=config, catalogue=catalogue)
    runs = context_runs_sec(s)
    want = config_preset(SLOW_CONTEXT_PRESET, catalogue)
    cfg = s.meta["config"]
    timing_ok = all(_same(cfg[k], v) for k, v in want.items())
    shortest = float(runs.min()) if runs.size else float("nan")
    extra = {
        "n_runs": int(runs.size),
        "shortest_run_sec": shortest,
        "median_run_sec": float(np.median(runs)) if runs.size else float("nan"),
        "config_in_force": timing_ok,
    }
    passed = bool(runs.size and shortest >= SLOW_CONTEXT_MIN_DWELL_SEC - s.dt
                  and timing_ok)
    row = _row(system_id, family, seed, SLOW_CONTEXT_PRESET, SLOW_CONTEXT_CHECK,
               shortest if runs.size else 0.0, SLOW_CONTEXT_MIN_DWELL_SEC, passed,
               dict(extra))
    return pd.DataFrame([{**row, **extra}])


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(a) == list(b)
    return a == b


__all__ = [
    "CHECKS_BY_SYSTEM",
    "IGNITION_OCCUPANCY_MAX",
    "MANIPULATION_CHECK_VERSION_V2",
    "SLOW_CONTEXT_CHECK_ID",
    "check_system",
    "context_runs_sec",
    "hub_periphery_paths",
    "ignition_occupancy",
    "module_context_information",
    "pc_half_check",
    "pc_half_summary",
    "prerequisite_m",
    "realisation_check",
    "realisation_report",
    "realised_time_constants",
    "schedule_hash",
    "slow_context_check",
    "slow_context_report",
    "structural_hash",
    "summarise_realisation",
    "twin_check",
    "twin_systems",
]
