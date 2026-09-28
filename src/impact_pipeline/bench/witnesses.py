"""
Loader for the MPC-Bench witness catalogue (``witnesses.yaml``).

PyYAML is optional: when it cannot be imported the JSON twin
``witnesses.json`` (same content) is read instead. Witnesses are defined by
mechanism (generator + knobs); :func:`build_witness_system` realises one for a
given family and seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from impact_pipeline.bench.generators import (
    PRINCIPLES,
    AgentConfig,
    BenchSystem,
    Knobs,
    knobs_from_dict,
    make_system,
)

WITNESS_YAML = Path(__file__).with_name("witnesses.yaml")
WITNESS_JSON = Path(__file__).with_name("witnesses.json")
VERDICTS = ("ATTRIBUTED", "NOT_ATTRIBUTED", "UNDETERMINED")
WITNESS_CLASSES = (
    "positive_control",
    "single_deficit",
    "null",
    "over_excluded",
    "patchwork",
)
REQUIRED_FIELDS = (
    "id",
    "class",
    "generator",
    "families",
    "knobs",
    "target",
    "mechanism_removed",
    "intended_pattern",
    "counterexample",
    "expected_verdict",
)
FAMILY_GENERATORS = {
    ("family_a", "A"): "family_a",
    ("family_a", "C"): "family_c",
}


def _read(path: Path) -> dict:
    if path.suffix in (".yaml", ".yml"):
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_witnesses(path=None, validate: bool = True) -> dict:
    """
    Load the catalogue. With ``path=None`` the packaged YAML is read, or its
    JSON twin when PyYAML is unavailable. The result carries ``source``.
    """
    if path is not None:
        cat = _read(Path(path))
        source = str(path)
    else:
        try:
            import yaml  # noqa: F401

            cat = _read(WITNESS_YAML)
            source = str(WITNESS_YAML)
        except ImportError:
            cat = _read(WITNESS_JSON)
            source = str(WITNESS_JSON)
    if validate:
        errors = validate_catalogue(cat)
        if errors:
            raise ValueError("Invalid witness catalogue: " + "; ".join(errors))
    cat = dict(cat)
    cat["source"] = source
    return cat


def validate_catalogue(cat: dict) -> List[str]:
    """Schema and consistency errors (empty list when valid)."""
    errors = []
    if tuple(cat.get("principles") or ()) != PRINCIPLES:
        errors.append(f"principles must be {list(PRINCIPLES)}")
    seen = set()
    for w in cat.get("witnesses") or []:
        wid = w.get("id", "<missing id>")
        missing = [f for f in REQUIRED_FIELDS if f not in w]
        if missing:
            errors.append(f"{wid}: missing {missing}")
            continue
        if wid in seen:
            errors.append(f"{wid}: duplicate id")
        seen.add(wid)
        if w["class"] not in WITNESS_CLASSES:
            errors.append(f"{wid}: unknown class {w['class']!r}")
        pat = w["intended_pattern"]
        if len(pat) != 5 or any(v not in (0, 1, None) for v in pat):
            errors.append(f"{wid}: intended_pattern must be five 0/1/null values")
        if w["expected_verdict"] not in VERDICTS:
            errors.append(f"{wid}: expected_verdict {w['expected_verdict']!r}")
        if w["class"] == "single_deficit":
            if w["target"] not in PRINCIPLES:
                errors.append(f"{wid}: single_deficit needs a target principle")
            elif pat != [0 if p == w["target"] else 1 for p in PRINCIPLES]:
                errors.append(
                    f"{wid}: single-deficit pattern must be the co-atom of its target"
                )
        for fam in w["families"]:
            if fam not in ("A", "C"):
                errors.append(f"{wid}: unknown family {fam!r}")
        try:
            kn = knobs_from_dict(w["knobs"])
        except (TypeError, ValueError) as exc:
            errors.append(f"{wid}: bad knobs ({exc})")
            continue
        if w["generator"] in ("family_a", "patchwork"):
            expected = [v for v in pat]
            bits = list(kn.bits())
            if w["generator"] == "family_a" and any(
                e is not None and e != b for e, b in zip(expected, bits)
            ):
                errors.append(f"{wid}: knobs realise {bits}, intended {expected}")
        ce = w["counterexample"]
        if (
            not isinstance(ce, dict)
            or not ce.get("claim")
            or not ce.get("formalised_as")
        ):
            errors.append(f"{wid}: counterexample needs claim and formalised_as")
    return errors


def get_witness(witness_id: str, catalogue: Optional[dict] = None) -> dict:
    cat = catalogue if catalogue is not None else load_witnesses()
    for w in cat["witnesses"]:
        if w["id"] == witness_id:
            return w
    raise KeyError(f"Unknown witness {witness_id!r}")


def witness_knobs(witness: dict) -> Knobs:
    return knobs_from_dict(witness.get("knobs") or {})


def build_witness_system(
    witness: dict,
    seed: int,
    family: str = "A",
    config: Optional[AgentConfig] = None,
) -> BenchSystem:
    """Realise a witness for ``family`` ('A' or held-out 'C') and ``seed``."""
    family = str(family).upper()
    if family not in witness["families"]:
        raise ValueError(f"Witness {witness['id']} is not defined for family {family}")
    gen = witness["generator"]
    kw = {}
    if gen == "family_a":
        gen = FAMILY_GENERATORS[(gen, family)]
    elif gen == "patchwork":
        kw["dynamics"] = "rate" if family == "A" else "stuart_landau"
    system = make_system(
        gen, witness_knobs(witness), config, seed, template_family=family, **kw
    )
    system.meta["witness_id"] = witness["id"]
    system.oracle["witness_id"] = witness["id"]
    system.oracle["witness_intended_pattern"] = list(witness["intended_pattern"])
    system.oracle["witness_expected_verdict"] = witness["expected_verdict"]
    return system
