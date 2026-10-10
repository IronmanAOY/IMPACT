"""
MPC-Bench v2 systems: the staggered hidden-driver adversaries, and the
catalogue and builder of every v2 witness and adversary.

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

``tau10`` and ``saturating`` are held-out conditions (no v2 estimator output
before the v2 freeze; oracle checks only). In the hierarchical variant the
upstream modules lead the hub and the hub leads the downstream modules, so
both transfer directions are inflated without any receive-and-return edge
(a shared input plus a hierarchy of intrinsic time constants; Murray et al.
2014, doi:10.1038/nn.3862).

Every draw comes from the reserved stream ``SeedSequence([seed, 43])``
(:func:`impact_pipeline.v2.seeds.stream_seed_sequence`), split into the
driver schedule, the patterns and the noise, so the variants of one seed
share driver, patterns and noise (common random numbers) and differ only in
their time constants or transform. The time base, the module layout, the
declared meta and the events table are those of the nominal family-A run of
the same seed and configuration (as for the v1 common driver), so
event-locked estimators are defined; the events do not act on the system.

The oracle holds the driver level per sample as ``context_state`` (what an
experimenter logs: the recording device exports it as ``context_cue`` under
the complete declaration R, exactly like the family-A context), the switch
times and levels, the patterns, the filtered drive ``drive_filtered`` (``y``)
and the noise-free signal ``hidden_signal`` (``f(y)``), and the module time
constants. There is no slow rhythm.

Catalogue and builder
---------------------
``witnesses_v2.yaml`` (schema 3) lists every witness and adversary of the v2
benchmark. Entries of origin v1 inherit their v1 definition
(:mod:`impact_pipeline.bench.witnesses`, :mod:`impact_pipeline.bench.adversarial`)
and override only what v2 changes; :func:`build_system` realises any entry
for a family, seed, configuration preset and twin replicate. A v1 witness
built at ``replicate = 0`` is the system of the v1 witness builder bit for
bit; a v1 adversary is the v1 system with its v2 labels added.

This module must not import ``impact_pipeline.mpc_metrics``.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import lfilter

from impact_pipeline.bench.generators import (
    ADVERSARIAL_GENERATORS,
    PRINCIPLES,
    AgentConfig,
    BenchSystem,
    _like_meta,
    _samples,
    config_from_dict,
    knobs_from_dict,
    make_system,
    simulate_family_a,
)

ADVERSARIAL_V2_VERSION = "mpc-bench-adversarial/2.0.0"
CATALOGUE_V2_PATH = Path(__file__).with_name("witnesses_v2.yaml")
CATALOGUE_SCHEMA_VERSION = 3
VERDICTS = ("MPC_CONSISTENT", "EXCLUDED", "UNDETERMINED")
FAMILIES = ("A", "C")
FAMILY_GENERATORS = {"A": "family_a", "C": "family_c"}

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
    from impact_pipeline.v2 import GENERATOR_VERSION_V2
    from impact_pipeline.v2.seeds import stream_seed_sequence

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
        "adversarial_version": ADVERSARIAL_V2_VERSION,
        "generator_version": GENERATOR_VERSION_V2,
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


# ---------------------------------------------------------------------------
# Catalogue (witnesses_v2.yaml)
# ---------------------------------------------------------------------------

WITNESS_CLASSES_V2 = (
    "positive_control",
    "single_deficit",
    "targeted_null",
    "null",
    "over_excluded",
    "patchwork",
)
WITNESS_GENERATORS_V2 = (
    "family_a",
    "patchwork",
    "null_independent_noise",
    "null_ar1",
    "hypersynchronous",
)
# Channels the recording device may export under the complete declaration,
# with the oracle key holding them and their kind.
RECORDABLE_CHANNELS = {
    "context_cue": ("context_state", "labels"),
    "slow_phase": ("slow_phase", "phase"),
    "driver": ("driver", "continuous"),
}
WITNESS_REQUIRED = (
    "id", "origin", "class", "generator", "families", "knobs", "config",
    "target", "mechanism_removed", "intended_pattern", "counterexample",
    "expected_verdict",
)
ADVERSARY_REQUIRED = (
    "id", "origin", "generator", "families", "designed_to_fool",
    "intended_pattern", "expected_verdict", "mechanism", "claim", "references",
)
_ENTRY_DEFAULTS = {
    "twin_families": [],
    "recorded_inputs": [],
    "held_out": False,
}


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is in the environment
        raise ImportError("PyYAML is needed to read the v2 system catalogue") from exc
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _v1_witnesses() -> Dict[str, dict]:
    from impact_pipeline.bench.witnesses import load_witnesses

    return {w["id"]: w for w in load_witnesses()["witnesses"]}


def _v1_adversary(generator: str) -> dict:
    from impact_pipeline.bench import adversarial as adv

    kind = generator[len("adversarial_"):]
    if kind not in adv.ADVERSARIAL_CATALOGUE:
        raise ValueError(f"no v1 adversary {generator!r}")
    entry = copy.deepcopy(adv.ADVERSARIAL_CATALOGUE[kind])
    entry["expected_verdict"] = adv.EXPECTED_VERDICT
    return entry


def resolve_catalogue(raw: Mapping) -> dict:
    """The catalogue with every v1-origin entry merged with its v1
    definition and the defaults filled in. A relabelled v1 adversary keeps
    its v1 labels as ``v1_intended_pattern`` and ``v1_expected_verdict``."""
    cat = copy.deepcopy(dict(raw))
    v1w = None
    out_w = []
    for entry in cat.get("witnesses") or []:
        entry = dict(entry)
        if entry.get("origin") == "v1":
            v1w = v1w if v1w is not None else _v1_witnesses()
            if entry["id"] not in v1w:
                raise ValueError(f"{entry['id']}: no v1 witness of that id")
            base = copy.deepcopy(v1w[entry["id"]])
            base.setdefault("config", {})
            base.update(entry)
            entry = base
        for k, v in _ENTRY_DEFAULTS.items():
            entry.setdefault(k, copy.deepcopy(v))
        out_w.append(entry)
    cat["witnesses"] = out_w
    out_a = []
    for entry in cat.get("adversaries") or []:
        entry = dict(entry)
        if entry.get("origin") == "v1":
            base = _v1_adversary(str(entry.get("generator", "")))
            for key in ("intended_pattern", "expected_verdict"):
                if key in entry and entry[key] != base[key]:
                    entry[f"v1_{key}"] = base[key]
            base.update(entry)
            entry = base
        entry.setdefault("variants", [])
        for k, v in _ENTRY_DEFAULTS.items():
            entry.setdefault(k, copy.deepcopy(v))
        out_a.append(entry)
    cat["adversaries"] = out_a
    return cat


def _pattern_errors(wid: str, pat) -> List[str]:
    if not isinstance(pat, list) or len(pat) != 5 or any(
            v not in (0, 1, None) for v in pat):
        return [f"{wid}: intended_pattern must be five 0/1/null values"]
    return []


def _reference_errors(wid: str, refs, known: Mapping) -> List[str]:
    if not refs:
        return [f"{wid}: no references"]
    missing = [r for r in refs if r not in known]
    return [f"{wid}: reference keys without a DOI {missing}"] if missing else []


def validate_catalogue_v2(cat: Mapping) -> List[str]:
    """Schema and consistency errors of a resolved catalogue (empty when
    valid)."""
    errors = []
    if cat.get("schema_version") != CATALOGUE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CATALOGUE_SCHEMA_VERSION}")
    if tuple(cat.get("principles") or ()) != PRINCIPLES:
        errors.append(f"principles must be {list(PRINCIPLES)}")
    if tuple(cat.get("verdicts") or ()) != VERDICTS:
        errors.append(f"verdicts must be {list(VERDICTS)}")
    refs = cat.get("references") or {}
    for key, doi in refs.items():
        if not (isinstance(doi, str) and doi.startswith("10.")):
            errors.append(f"reference {key}: not a DOI ({doi!r})")
    presets = cat.get("config_presets") or {}
    for name, p in presets.items():
        try:
            config_from_dict(p.get("config"))
        except (TypeError, ValueError) as exc:
            errors.append(f"config preset {name}: {exc}")
    for name, p in (cat.get("whole_brain_presets") or {}).items():
        errors.extend(f"whole-brain preset {name}: {e}"
                      for e in _whole_brain_preset_errors(p))

    seen = set()
    for w in cat.get("witnesses") or []:
        wid = w.get("id", "<missing id>")
        missing = [f for f in WITNESS_REQUIRED if f not in w]
        if missing:
            errors.append(f"{wid}: missing {missing}")
            continue
        if wid in seen:
            errors.append(f"{wid}: duplicate id")
        seen.add(wid)
        if w["origin"] not in ("v1", "v2"):
            errors.append(f"{wid}: origin must be v1 or v2")
        if w["class"] not in WITNESS_CLASSES_V2:
            errors.append(f"{wid}: unknown class {w['class']!r}")
        if w["generator"] not in WITNESS_GENERATORS_V2:
            errors.append(f"{wid}: unknown generator {w['generator']!r}")
        pat = w["intended_pattern"]
        pat_errors = _pattern_errors(wid, pat)
        errors.extend(pat_errors)
        if w["expected_verdict"] not in VERDICTS:
            errors.append(f"{wid}: expected_verdict {w['expected_verdict']!r}")
        fams = list(w["families"] or [])
        if not fams or any(f not in FAMILIES for f in fams):
            errors.append(f"{wid}: families must be a non-empty subset of {FAMILIES}")
        twins = list(w.get("twin_families") or [])
        if any(f not in fams for f in twins):
            errors.append(f"{wid}: twin_families must be a subset of families")
        if twins and w["generator"] != "family_a":
            errors.append(f"{wid}: twins exist for the agent generator only")
        bad_ch = [c for c in w.get("recorded_inputs") or []
                  if c not in RECORDABLE_CHANNELS]
        if bad_ch:
            errors.append(f"{wid}: unknown recorded inputs {bad_ch}")
        if not pat_errors:
            cls = w["class"]
            if cls == "single_deficit" and (
                    w["target"] not in PRINCIPLES
                    or pat != [0 if p == w["target"] else 1 for p in PRINCIPLES]):
                errors.append(f"{wid}: a single deficit is the co-atom of its target")
            if cls == "targeted_null" and (
                    w["target"] not in PRINCIPLES
                    or pat[PRINCIPLES.index(w["target"])] != 0):
                errors.append(f"{wid}: a targeted null has its target off")
            if cls == "positive_control" and pat != [1] * 5:
                errors.append(f"{wid}: a positive control has every mechanism on")
            if cls in ("null", "over_excluded") and any(v == 1 for v in pat):
                errors.append(f"{wid}: a {cls} system has no mechanism on")
        try:
            kn = knobs_from_dict(w["knobs"])
            config_from_dict(w["config"])
        except (TypeError, ValueError) as exc:
            errors.append(f"{wid}: bad knobs or config ({exc})")
            continue
        if w["generator"] == "family_a" and not pat_errors:
            bits = list(kn.bits())
            if any(e is not None and e != b for e, b in zip(pat, bits)):
                errors.append(f"{wid}: knobs realise {bits}, intended {pat}")
        ce = w["counterexample"]
        if not isinstance(ce, dict) or not ce.get("claim") or not ce.get(
                "formalised_as"):
            errors.append(f"{wid}: counterexample needs claim and formalised_as")
        elif w["origin"] == "v2":
            errors.extend(_reference_errors(wid, ce.get("references"), refs))

    for a in cat.get("adversaries") or []:
        aid = a.get("id", "<missing id>")
        missing = [f for f in ADVERSARY_REQUIRED if f not in a]
        if missing:
            errors.append(f"{aid}: missing {missing}")
            continue
        if aid in seen:
            errors.append(f"{aid}: duplicate id")
        seen.add(aid)
        gen = a["generator"]
        if a["origin"] == "v2":
            if gen != STAGGERED_GENERATOR:
                errors.append(f"{aid}: unknown v2 generator {gen!r}")
            errors.extend(_reference_errors(aid, a.get("references"), refs))
        elif a["origin"] == "v1":
            if gen not in ADVERSARIAL_GENERATORS:
                errors.append(f"{aid}: unknown v1 generator {gen!r}")
        else:
            errors.append(f"{aid}: origin must be v1 or v2")
        errors.extend(_pattern_errors(aid, a["intended_pattern"]))
        if a["expected_verdict"] not in VERDICTS:
            errors.append(f"{aid}: expected_verdict {a['expected_verdict']!r}")
        if list(a["families"]) != ["A"]:
            errors.append(f"{aid}: adversaries are defined in family A")
        if a["designed_to_fool"] not in PRINCIPLES:
            errors.append(f"{aid}: designed_to_fool must be a principle")
        bad_ch = [c for c in a.get("recorded_inputs") or []
                  if c not in RECORDABLE_CHANNELS]
        if bad_ch:
            errors.append(f"{aid}: unknown recorded inputs {bad_ch}")
        if gen == STAGGERED_GENERATOR:
            variants = list(a.get("variants") or [])
            expected = list(STAGGERED_SYSTEM_VARIANTS.get(aid, ()))
            if variants != expected:
                errors.append(f"{aid}: variants {variants}, expected {expected}")
            held = {bool(STAGGERED_VARIANTS[v]["held_out"]) for v in variants
                    if v in STAGGERED_VARIANTS}
            if held != {bool(a.get("held_out"))}:
                errors.append(f"{aid}: held_out does not match its variants")
        elif a.get("variants"):
            errors.append(f"{aid}: only the staggered driver has variants")
    for r in cat.get("removed") or []:
        if r.get("id") in seen:
            errors.append(f"{r.get('id')}: listed as removed and as active")
        if not r.get("reason"):
            errors.append(f"{r.get('id')}: removal without a reason")
    return errors


def load_catalogue_v2(path=None, validate: bool = True) -> dict:
    """The resolved v2 catalogue (default ``witnesses_v2.yaml``); raises
    ``ValueError`` listing every error when ``validate`` and it is invalid."""
    p = Path(path) if path is not None else CATALOGUE_V2_PATH
    cat = resolve_catalogue(_read_yaml(p))
    if validate:
        errors = validate_catalogue_v2(cat)
        if errors:
            raise ValueError("Invalid v2 system catalogue: " + "; ".join(errors))
    cat["source"] = str(p)
    return cat


_CATALOGUE_CACHE: Dict[str, dict] = {}


def _catalogue(catalogue: Optional[Mapping]) -> Mapping:
    if catalogue is not None:
        return catalogue
    key = str(CATALOGUE_V2_PATH)
    if key not in _CATALOGUE_CACHE:
        _CATALOGUE_CACHE[key] = load_catalogue_v2()
    return _CATALOGUE_CACHE[key]


def get_entry(system_id: str, catalogue: Optional[Mapping] = None) -> dict:
    """The resolved catalogue entry of a witness or adversary (a copy), with
    ``kind`` set to ``'witness'`` or ``'adversary'``."""
    cat = _catalogue(catalogue)
    for key, kind in (("witnesses", "witness"), ("adversaries", "adversary")):
        for e in cat.get(key) or []:
            if e["id"] == system_id:
                out = copy.deepcopy(e)
                out["kind"] = kind
                return out
    removed = [r["id"] for r in cat.get("removed") or []]
    if system_id in removed:
        raise KeyError(f"{system_id!r} is not part of the v2 benchmark (removed)")
    raise KeyError(f"Unknown v2 system {system_id!r}")


def system_ids(catalogue: Optional[Mapping] = None, kind: Optional[str] = None,
               family: Optional[str] = None) -> List[str]:
    """Ids of the catalogue's witnesses and adversaries (``kind``
    ``'witness'`` or ``'adversary'``), optionally those defined in
    ``family``."""
    cat = _catalogue(catalogue)
    out = []
    for key, k in (("witnesses", "witness"), ("adversaries", "adversary")):
        if kind is not None and kind != k:
            continue
        out.extend(e["id"] for e in cat.get(key) or []
                   if family is None or family in e["families"])
    return out


def config_preset(name: str, catalogue: Optional[Mapping] = None) -> dict:
    """The agent-configuration overrides of a named preset (a copy)."""
    presets = _catalogue(catalogue).get("config_presets") or {}
    if name not in presets:
        raise KeyError(f"Unknown config preset {name!r}; one of {sorted(presets)}")
    return copy.deepcopy(dict(presets[name]["config"]))


def _merged_config(config, preset: Optional[str], entry_config: Mapping,
                   catalogue) -> Optional[AgentConfig]:
    """The caller's configuration, then the preset, then the system's own
    overrides (which may not contradict the preset). None when all three
    are empty, so a v1 system is built with exactly the v1 call."""
    base = config
    if isinstance(base, Mapping):
        base = config_from_dict(dict(base))
    over = config_preset(preset, catalogue) if preset else {}
    entry_config = dict(entry_config or {})
    clash = {
        k: (over[k], v) for k, v in entry_config.items()
        if k in over and _norm(over[k]) != _norm(v)
    }
    if clash:
        raise ValueError(f"preset {preset!r} contradicts the system's configuration "
                         f"{clash}")
    over.update(entry_config)
    if not over:
        return base
    return config_from_dict(over, base=base if base is not None else AgentConfig())


def _norm(v):
    return tuple(v) if isinstance(v, (list, tuple)) else v


def build_system(
    system_id: str,
    seed: int,
    family: str = "A",
    *,
    variant: Optional[str] = None,
    replicate: int = 0,
    config: Union[None, AgentConfig, Mapping] = None,
    preset: Optional[str] = None,
    catalogue: Optional[Mapping] = None,
) -> BenchSystem:
    """
    Realise a catalogue entry for ``family`` (``'A'``, or ``'C'`` for C1),
    ``seed`` and twin ``replicate`` (``>= 1`` only for agent witnesses).
    ``config`` is the base agent configuration, ``preset`` a named
    configuration preset (e.g. ``'ram_only_160'``); the system's own
    configuration overrides apply last. ``variant`` selects a staggered-
    driver variant (default: the entry's first). Witnesses carry
    ``meta['witness_id']`` and the oracle labels ``witness_*`` exactly as the
    v1 witness builder writes them; adversaries carry ``adversary_*``.
    """
    entry = get_entry(system_id, catalogue)
    family = str(family).upper()
    if family not in entry["families"]:
        raise ValueError(f"{system_id} is not defined for family {family}")
    cfg = _merged_config(config, preset, entry.get("config") or {}, catalogue)
    gen = entry["generator"]
    if entry["kind"] == "witness":
        kw = {}
        if gen == "family_a":
            gen = FAMILY_GENERATORS[family]
            kw["replicate"] = replicate
        elif gen == "patchwork":
            kw["dynamics"] = "rate" if family == "A" else "stuart_landau"
        if gen not in ("family_a", "family_c") and replicate:
            raise ValueError(f"{system_id}: twins exist for agent witnesses only")
        system = make_system(gen, knobs_from_dict(entry["knobs"]), cfg, seed,
                             template_family=family, **kw)
        system.meta["witness_id"] = entry["id"]
        system.oracle["witness_id"] = entry["id"]
        system.oracle["witness_intended_pattern"] = list(entry["intended_pattern"])
        system.oracle["witness_expected_verdict"] = entry["expected_verdict"]
        return system
    if replicate:
        raise ValueError(f"{system_id}: twins exist for agent witnesses only")
    if gen == STAGGERED_GENERATOR:
        variants = list(entry["variants"])
        variant = variant if variant is not None else variants[0]
        if variant not in variants:
            raise ValueError(f"{system_id}: variant must be one of {variants}")
        system = staggered_driver_system(variant, seed, config=cfg)
        system.meta["adversary_variant"] = variant
    else:
        if variant is not None:
            raise ValueError(f"{system_id} has no variants")
        system = make_system(gen, None, cfg, seed)
    system.meta["adversary_id"] = entry["id"]
    system.oracle["adversary_id"] = entry["id"]
    system.oracle["adversary_intended_pattern"] = list(entry["intended_pattern"])
    system.oracle["adversary_expected_verdict"] = entry["expected_verdict"]
    return system


def recorded_channels(system: BenchSystem, system_id: Optional[str] = None,
                      catalogue: Optional[Mapping] = None) -> Dict[str, dict]:
    """The channels an experimenter logs for the complete declaration of a
    catalogue system: ``{channel: {'oracle_key', 'kind', 'value'}}``."""
    sid = system_id or system.meta.get("witness_id") or system.meta.get(
        "adversary_id")
    if sid is None:
        raise ValueError("the system carries no catalogue id; pass system_id")
    entry = get_entry(sid, catalogue)
    out = {}
    for ch in entry.get("recorded_inputs") or []:
        key, kind = RECORDABLE_CHANNELS[ch]
        if key not in system.oracle:
            raise KeyError(f"{sid}: recorded input {ch!r} needs oracle[{key!r}]")
        out[ch] = {"oracle_key": key, "kind": kind,
                   "value": np.asarray(system.oracle[key])}
    return out


# ---------------------------------------------------------------------------
# Whole-brain conditions at G_nom
# ---------------------------------------------------------------------------


def _whole_brain_preset_errors(p: Mapping) -> List[str]:
    from impact_pipeline.bench.whole_brain import LESIONS, g_sweep_levels

    errors = []
    levels = g_sweep_levels()
    idx = p.get("g_sweep_index")
    if not isinstance(idx, int) or not 0 <= idx < len(levels):
        return [f"g_sweep_index must index the v1 G sweep {levels}"]
    if p.get("G") != levels[idx]:
        errors.append(f"G {p.get('G')!r} is not the sweep level {levels[idx]}")
    bad = [k for k in p.get("lesions") or [] if k not in LESIONS or k in (
        "none", "random")]
    if bad or not p.get("lesions"):
        errors.append(f"lesions must name lesion kinds, got {p.get('lesions')}")
    return errors


def g_nom(catalogue: Optional[Mapping] = None) -> float:
    """``G_nom``: the G-sweep level of the lesion conditions (1.142857)."""
    from impact_pipeline.bench.whole_brain import g_sweep_levels

    p = _catalogue(catalogue)["whole_brain_presets"]["lesions_g_nom"]
    return float(g_sweep_levels()[int(p["g_sweep_index"])])


def lesion_conditions_g_nom(
    base=None, catalogue: Optional[Mapping] = None
) -> List[dict]:
    """The lesion conditions at ``G_nom``: each lesion of the preset followed
    by its size-matched random-edge lesion (the v1 construction of
    :func:`impact_pipeline.bench.whole_brain.manipulations`, at a sweep
    level so that the unlesioned baseline is a sweep condition)."""
    from impact_pipeline.bench import whole_brain as wb

    p = _catalogue(catalogue)["whole_brain_presets"]["lesions_g_nom"]
    base = base if base is not None else wb.WholeBrainConfig()
    if isinstance(base, Mapping):
        base = wb.whole_brain_config_from_dict(dict(base))
    return wb.manipulations(g_levels=(), lesions=tuple(p["lesions"]),
                            base=base.replace(G=g_nom(catalogue)))


__all__ = [
    "ADVERSARIAL_V2_VERSION",
    "CATALOGUE_V2_PATH",
    "RECORDABLE_CHANNELS",
    "STAGGERED_SYSTEM_VARIANTS",
    "STAGGERED_VARIANTS",
    "build_system",
    "config_preset",
    "g_nom",
    "get_entry",
    "lesion_conditions_g_nom",
    "load_catalogue_v2",
    "recorded_channels",
    "resolve_catalogue",
    "staggered_drive_input",
    "staggered_driver_system",
    "staggered_module_taus",
    "system_ids",
    "validate_catalogue_v2",
]
