"""
System catalogue of MPC-Bench (``systems.yaml``): every witness and adversary
of the benchmark, defined by mechanism (generator, knobs and configuration),
never by metric values, with its loader, validator and builder.

:func:`load_catalogue` reads and validates the file; :func:`build_system`
realises any entry for a family, seed, configuration preset and twin
replicate. Witnesses are built by their generator
(:func:`impact_pipeline.bench.generators.make_system`; ``family_a`` is the
family-A agent in family A and the family-C oscillator network in family C),
adversaries by :mod:`impact_pipeline.bench.adversarial`.

This module must not import ``impact_pipeline.mpc_metrics``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import numpy as np

from impact_pipeline.bench.adversarial import (
    STAGGERED_GENERATOR,
    STAGGERED_SYSTEM_VARIANTS,
    STAGGERED_VARIANTS,
    staggered_driver_system,
)
from impact_pipeline.bench.generators import (
    ADVERSARIAL_GENERATORS,
    PRINCIPLES,
    AgentConfig,
    BenchSystem,
    config_from_dict,
    knobs_from_dict,
    make_system,
)

CATALOGUE_PATH = Path(__file__).with_name("systems.yaml")
CATALOGUE_SCHEMA_VERSION = 1
VERDICTS = ("MPC_CONSISTENT", "EXCLUDED", "UNDETERMINED")
FAMILIES = ("A", "C")
FAMILY_GENERATORS = {"A": "family_a", "C": "family_c"}

WITNESS_CLASSES = (
    "positive_control",
    "single_deficit",
    "targeted_null",
    "null",
    "over_excluded",
    "patchwork",
)
WITNESS_GENERATORS = (
    "family_a",
    "patchwork",
    "null_independent_noise",
    "null_ar1",
    "hypersynchronous",
)
ADVERSARY_GENERATORS = (STAGGERED_GENERATOR,) + ADVERSARIAL_GENERATORS
# Channels the recording device may export under the complete declaration,
# with the oracle key holding them and their kind.
RECORDABLE_CHANNELS = {
    "context_cue": ("context_state", "labels"),
    "slow_phase": ("slow_phase", "phase"),
    "driver": ("driver", "continuous"),
}
WITNESS_REQUIRED = (
    "id", "class", "generator", "families", "knobs", "config", "target",
    "mechanism_removed", "intended_pattern", "counterexample", "expected_verdict",
)
ADVERSARY_REQUIRED = (
    "id", "generator", "families", "designed_to_fool", "intended_pattern",
    "expected_verdict", "mechanism", "claim", "references",
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
        raise ImportError("PyYAML is needed to read the system catalogue") from exc
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_catalogue(raw: Mapping) -> dict:
    """The catalogue with the entry defaults filled in (a copy)."""
    cat = copy.deepcopy(dict(raw))
    out_w = []
    for entry in cat.get("witnesses") or []:
        entry = dict(entry)
        for k, v in _ENTRY_DEFAULTS.items():
            entry.setdefault(k, copy.deepcopy(v))
        out_w.append(entry)
    cat["witnesses"] = out_w
    out_a = []
    for entry in cat.get("adversaries") or []:
        entry = dict(entry)
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


def _reference_errors(wid: str, refs) -> List[str]:
    if not refs or not all(isinstance(r, str) and r for r in refs):
        return [f"{wid}: no references"]
    return []


def validate_catalogue(cat: Mapping) -> List[str]:
    """Schema and consistency errors of a resolved catalogue (empty when
    valid)."""
    errors = []
    if cat.get("schema_version") != CATALOGUE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CATALOGUE_SCHEMA_VERSION}")
    if tuple(cat.get("principles") or ()) != PRINCIPLES:
        errors.append(f"principles must be {list(PRINCIPLES)}")
    if tuple(cat.get("verdicts") or ()) != VERDICTS:
        errors.append(f"verdicts must be {list(VERDICTS)}")
    for key, doi in (cat.get("references") or {}).items():
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
        if w["class"] not in WITNESS_CLASSES:
            errors.append(f"{wid}: unknown class {w['class']!r}")
        if w["generator"] not in WITNESS_GENERATORS:
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
        else:
            errors.extend(_reference_errors(wid, ce.get("references")))

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
        if gen not in ADVERSARY_GENERATORS:
            errors.append(f"{aid}: unknown generator {gen!r}")
        errors.extend(_reference_errors(aid, a.get("references")))
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
    return errors


def load_catalogue(path=None, validate: bool = True) -> dict:
    """The resolved catalogue (default ``systems.yaml``); raises
    ``ValueError`` listing every error when ``validate`` and it is invalid."""
    p = Path(path) if path is not None else CATALOGUE_PATH
    cat = resolve_catalogue(_read_yaml(p))
    if validate:
        errors = validate_catalogue(cat)
        if errors:
            raise ValueError("Invalid system catalogue: " + "; ".join(errors))
    cat["source"] = str(p)
    return cat


_CATALOGUE_CACHE: Dict[str, dict] = {}


def _catalogue(catalogue: Optional[Mapping]) -> Mapping:
    if catalogue is not None:
        return catalogue
    key = str(CATALOGUE_PATH)
    if key not in _CATALOGUE_CACHE:
        _CATALOGUE_CACHE[key] = load_catalogue()
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
    raise KeyError(f"Unknown system {system_id!r}")


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
    are empty, so the system is built with the generator's defaults."""
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
    Realise a catalogue entry for ``family`` (``'A'`` or ``'C'``), ``seed``
    and twin ``replicate`` (``>= 1`` only for agent witnesses). ``config``
    is the base agent configuration, ``preset`` a named configuration
    preset (e.g. ``'ram_only_160'``); the system's own configuration
    overrides apply last. ``variant`` selects a staggered-driver variant
    (default: the entry's first). Witnesses carry ``meta['witness_id']``
    and the oracle labels ``witness_*``; adversaries carry ``adversary_*``.
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
        return [f"g_sweep_index must index the G sweep {levels}"]
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
    by its size-matched random-edge lesion (the construction of
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
    "ADVERSARY_GENERATORS",
    "CATALOGUE_PATH",
    "RECORDABLE_CHANNELS",
    "build_system",
    "config_preset",
    "g_nom",
    "get_entry",
    "lesion_conditions_g_nom",
    "load_catalogue",
    "recorded_channels",
    "resolve_catalogue",
    "system_ids",
    "validate_catalogue",
]
