#!/usr/bin/env python
"""
Evaluate the paper-2 hypothesis registry (``predictions/registry.yaml``) on
episode-level results.

Refusals (exit code 2, ``predictions_refusal.json``):

- the registry does not validate against ``predictions/registry.schema.json``
  or breaks a semantic rule (H0-H10 present and unique; ds003171, ds005620 and
  ds006623 are exploratory / calibration only; a confirmatory dataset must be
  ``never_accessed``; H2 is auxiliary; a frozen registry has a freeze tag,
  protocol hashes and registered estimator versions);
- confirmatory evaluation of a draft registry (use ``--exploratory``), or of a
  frozen registry whose freeze tag is missing from the repository or whose
  file differs from its version at the tag (``--skip-freeze-tag-check``
  disables this check and is recorded in the report);
- results from an unregistered protocol hash, an unregistered (principle,
  estimator, version) or an undeclared dataset, or results without the
  columns that identify them (``--allow-unregistered`` downgrades this to an
  explicitly labelled exploratory run).

Results table (CSV; one row per episode): ``dataset``, ``episode_id``,
``report_positive`` (bool or 0/1) or ``report`` (positive/negative),
``MPC_verdict`` (v1 or v2 names), ``MPC_reason`` (``;``-joined codes; the v1
code ``BEARER_MISMATCH:COHERENCE`` counts as ``SOURCE_INCOHERENT``),
``<P>_status``, ``protocol_hash``, ``<P>_estimator``,
``<P>_estimator_version``; optional ``<P>_c`` and the H2 outcome column, and
``comparator_<name>`` decisions for H10 (True/False/1/0, empty = abstain).
Episodes whose report label is missing or unrecognised are excluded from
every hypothesis (never counted as report-negative; ``n_report_unknown``).
H0 reads ``null_calibration_rates.csv`` (``--null-calibration``); cells whose
statuses did not come from the installed evidence layer (``status_impl``)
make H0 NOT_EVALUABLE.

Outcomes per hypothesis and stratum (confirmatory datasets / exploratory
datasets): SUPPORTED, FALSIFIED, INDETERMINATE (also below the registered
minimum n), AUXILIARY_REPORTED (H2) or NOT_EVALUABLE (inputs missing).

Example::

    python scripts/run_predictions.py --registry predictions/registry.yaml \
        --results episodes.csv --null-calibration null_calibration_rates.csv \
        --out outputs/predictions
    python scripts/run_predictions.py --validate-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (SRC_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from impact_pipeline import necessity as nc  # noqa: E402

RUNNER_VERSION = "run-predictions/1.0.0"
DEFAULT_REGISTRY = REPO_ROOT / "predictions" / "registry.yaml"
DEFAULT_SCHEMA = REPO_ROOT / "predictions" / "registry.schema.json"
PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")
REQUIRED_HYPOTHESES = tuple(f"H{i}" for i in range(11))
EXPLORATORY_ONLY = ("ds003171", "ds005620", "ds006623")
NOT_EVALUABLE = "NOT_EVALUABLE"
AUXILIARY_REPORTED = "AUXILIARY_REPORTED"
REFUSAL_EXIT = 2


class RegistryRefusal(Exception):
    """The registry or the results may not be evaluated; ``reasons`` lists why."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


# --------------------------------------------------------------------------
# hashing and loading
# --------------------------------------------------------------------------
def canonical_json_sha256(obj) -> str:
    """SHA-256 of the canonical JSON text (sorted keys, no whitespace)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_registry(path=DEFAULT_REGISTRY) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is in environment.yml
        raise RegistryRefusal([f"PyYAML is required to read {path}: {exc}"]) from exc
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise RegistryRefusal([f"{path} is not a mapping"])
    return data


def load_schema(path=DEFAULT_SCHEMA) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# JSON-schema subset validator (keywords used by registry.schema.json)
# --------------------------------------------------------------------------
_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}
SUPPORTED_KEYWORDS = {
    "$schema", "$id", "title", "description", "type", "required", "properties",
    "additionalProperties", "enum", "const", "pattern", "minLength", "minimum",
    "maximum", "exclusiveMinimum", "exclusiveMaximum", "items", "minItems",
    "uniqueItems",
}


def validate_schema(instance, schema, path="$") -> list:
    """Errors (strings) of ``instance`` against the schema subset."""
    errors = []
    unknown = set(schema) - SUPPORTED_KEYWORDS
    if unknown:
        errors.append(f"{path}: schema uses unsupported keywords {sorted(unknown)}")
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[t](instance) for t in types):
            return errors + [f"{path}: expected {'/'.join(types)}, got "
                             f"{type(instance).__name__}"]
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: {instance!r} does not match {schema['pattern']}")
    if _TYPES["number"](instance):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {instance} <= {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: {instance} >= {schema['exclusiveMaximum']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required {key!r}")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, val in instance.items():
            if key in props:
                errors += validate_schema(val, props[key], f"{path}.{key}")
            elif extra is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(extra, dict):
                errors += validate_schema(val, extra, f"{path}.{key}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if schema.get("uniqueItems"):
            seen = [json.dumps(v, sort_keys=True) for v in instance]
            if len(set(seen)) != len(seen):
                errors.append(f"{path}: items are not unique")
        if "items" in schema:
            for i, val in enumerate(instance):
                errors += validate_schema(val, schema["items"], f"{path}[{i}]")
    return errors


def jsonschema_errors(instance, schema):
    """Cross-check with the jsonschema package when installed (else None)."""
    try:
        import jsonschema
    except ImportError:
        return None
    validator = jsonschema.Draft202012Validator(schema)
    return sorted(e.message for e in validator.iter_errors(instance))


# --------------------------------------------------------------------------
# semantic rules
# --------------------------------------------------------------------------
_STAT_RULES = {
    "null_false_present_rate": ("null_band", {"measurement"}),
    "excluded_rate_report_positive": ("symmetric_upper_lower", {"verdict"}),
    "aggregation_exponent": ("auxiliary_report", {"auxiliary"}),
    "absent_rate_report_positive": ("symmetric_necessity", {"component"}),
    "reason_rate_report_positive": ("symmetric_upper_lower", {"verdict"}),
    "coverage": ("symmetric_upper_lower", {"measurement"}),
    "selective_accuracy_gap": ("non_inferiority", {"comparative"}),
}


def semantic_errors(reg: dict) -> list:
    errs = []
    hyps = reg.get("hypotheses") or []
    ids = [h.get("id") for h in hyps]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        errs.append(f"duplicate hypothesis ids {dup}")
    missing = [h for h in REQUIRED_HYPOTHESES if h not in ids]
    if missing:
        errs.append(f"missing hypotheses {missing}")
    for h in hyps:
        hid, stat = h.get("id"), h.get("statistic")
        rule = _STAT_RULES.get(stat)
        if rule is None:
            continue
        if h.get("decision") != rule[0]:
            errs.append(f"{hid}: statistic {stat} needs decision {rule[0]}")
        if h.get("level") not in rule[1]:
            errs.append(f"{hid}: statistic {stat} needs level {sorted(rule[1])}")
        if stat == "absent_rate_report_positive" and not h.get("principle"):
            errs.append(f"{hid}: a component hypothesis needs a principle")
        if stat == "reason_rate_report_positive" and not h.get("reason_code"):
            errs.append(f"{hid}: reason_rate_report_positive needs reason_code")
        stance_level = h.get("level") in ("verdict", "component")
        if h.get("counts_for_stance") and not stance_level:
            errs.append(f"{hid}: only verdict/component hypotheses count for "
                        "the stance")
    h2 = next((h for h in hyps if h.get("id") == "H2"), None)
    if h2 is not None and (h2.get("counts_for_stance")
                           or h2.get("level") != "auxiliary"):
        errs.append("H2 (aggregation exponent) must be auxiliary and not count for "
                    "the stance")
    dsets = {d.get("id"): d for d in reg.get("datasets") or []}
    for ds in EXPLORATORY_ONLY:
        d = dsets.get(ds)
        if d is None:
            errs.append(f"dataset {ds} must be declared (prior access)")
        elif d.get("prior_access") != "exploratory_calibration":
            errs.append(f"dataset {ds} must have prior_access exploratory_calibration")
    for ds, d in dsets.items():
        if (d.get("role") == "confirmatory"
                and d.get("prior_access") != "never_accessed"):
            errs.append(f"dataset {ds}: confirmatory role requires never_accessed data")
    if reg.get("status") == "frozen":
        if not reg.get("freeze_tag"):
            errs.append("a frozen registry needs a freeze_tag")
        for p in reg.get("protocols") or []:
            if not p.get("hash"):
                errs.append(f"frozen registry: protocol {p.get('id')} has no hash")
            elif p.get("hash_algorithm") == "canonical_json_sha256" and p.get("spec"):
                if canonical_json_sha256(p["spec"]) != p["hash"]:
                    errs.append(f"protocol {p.get('id')}: hash does not match its spec")
            for pr in p.get("necessity_set") or []:
                if not any(e.get("principle") == pr and e.get("version")
                           and e.get("registration") == "registered"
                           for e in reg.get("estimators") or []):
                    errs.append(f"frozen registry: no registered estimator for {pr}")
    return errs


def verify_freeze_tag(reg: dict, registry_path, repo_root=REPO_ROOT) -> list:
    """
    Problems with the freeze of a frozen registry: the tag must exist in the
    repository and the registry file at the tag must be byte-identical to the
    evaluated one (no edits after the freeze). Read-only git queries.
    """
    import subprocess

    tag = reg.get("freeze_tag")
    if reg.get("status") != "frozen" or not tag:
        return ["registry is not frozen"]
    probs = []
    ok = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--verify", "-q",
                         f"refs/tags/{tag}"], capture_output=True)
    if ok.returncode != 0:
        return [f"freeze tag {tag!r} not found in {repo_root}"]
    try:
        rel = Path(registry_path).resolve().relative_to(Path(repo_root).resolve())
    except ValueError:
        return [f"registry {registry_path} is outside the repository"]
    shown = subprocess.run(["git", "-C", str(repo_root), "show",
                            f"{tag}:{rel.as_posix()}"], capture_output=True)
    if shown.returncode != 0:
        probs.append(f"registry {rel} is not in the freeze tag {tag!r}")
    elif hashlib.sha256(shown.stdout).hexdigest() != file_sha256(registry_path):
        probs.append(f"registry {rel} differs from its version at tag {tag!r}")
    return probs


def validate_registry(reg: dict, schema: dict | None = None) -> list:
    schema = load_schema() if schema is None else schema
    return validate_schema(reg, schema) + semantic_errors(reg)


# --------------------------------------------------------------------------
# results registration checks
# --------------------------------------------------------------------------
def registered_protocols(reg):
    return {p["hash"]: p for p in reg.get("protocols") or [] if p.get("hash")}


def registered_estimators(reg):
    return {(e["principle"], str(e["estimator"]), str(e["version"]))
            for e in reg.get("estimators") or []
            if e.get("version") and e.get("registration") == "registered"}


def registration_problems(df: pd.DataFrame, reg: dict) -> list:
    """Why the results are not from registered protocols/estimators/datasets."""
    probs = []
    if "dataset" not in df.columns:
        probs.append("results lack the 'dataset' column")
    else:
        declared = {d["id"] for d in reg.get("datasets") or []}
        unknown = sorted(set(df["dataset"].astype(str)) - declared)
        if unknown:
            probs.append(f"undeclared datasets {unknown}")
    protos = registered_protocols(reg)
    if "protocol_hash" not in df.columns:
        probs.append("results lack the 'protocol_hash' column")
        nsets = set()
    else:
        hashes = sorted(set(df["protocol_hash"].astype(str)))
        bad = [h for h in hashes if h not in protos]
        if bad:
            probs.append(f"unregistered protocol hashes {bad}")
        nsets = {p for h in hashes if h in protos for p in protos[h]["necessity_set"]}
    ests = registered_estimators(reg)
    for p in sorted(nsets or set(PRINCIPLES), key=PRINCIPLES.index):
        ec, vc = f"{p}_estimator", f"{p}_estimator_version"
        if ec not in df.columns or vc not in df.columns:
            probs.append(f"results lack {ec}/{vc}")
            continue
        pairs = set(zip(df[ec].astype(str), df[vc].astype(str)))
        bad = sorted(f"{e}@{v}" for e, v in pairs if (p, e, v) not in ests)
        if bad:
            probs.append(f"unregistered {p} estimator versions {bad}")
    return probs


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
_REPORT_TRUE = ("true", "1", "1.0", "yes", "positive", "report_positive")
_REPORT_FALSE = ("false", "0", "0.0", "no", "negative", "report_negative")


def _parse_label(v):
    """True / False for a recognised report label, None otherwise (missing or
    unrecognised labels are never guessed)."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(v)):
            return None
        return {1.0: True, 0.0: False}.get(float(v))
    txt = str(v).strip().lower()
    if txt in _REPORT_TRUE:
        return True
    if txt in _REPORT_FALSE:
        return False
    return None


def report_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Report label per episode: 1.0 (report-positive), 0.0 (report-negative)
    or NaN (missing / unrecognised; such episodes are excluded from every
    hypothesis rather than counted as report-negative).
    """
    if "report_positive" in df.columns:
        col = df["report_positive"]
    elif "report" in df.columns:
        col = df["report"]
    else:
        raise KeyError("results need 'report_positive' or 'report'")
    lab = [_parse_label(v) for v in col.tolist()]
    return np.asarray([np.nan if v is None else float(v) for v in lab], dtype=float)


def report_positive(df: pd.DataFrame) -> np.ndarray:
    """Boolean report-positive mask; raises if any label is missing or
    unrecognised (use :func:`report_labels` to filter them first)."""
    lab = report_labels(df)
    if np.isnan(lab).any():
        raise ValueError(f"{int(np.isnan(lab).sum())} episodes have a missing or "
                         "unrecognised report label")
    return lab.astype(bool)


def _params(h, defaults):
    out = dict(defaults)
    out.update({k: v for k, v in h.items() if v is not None})
    return out


def _gate_minimum_n(res, n, minimum_n):
    res["n"] = int(n)
    res["minimum_n"] = minimum_n
    res["outcome_ignoring_minimum_n"] = res["outcome"]
    if minimum_n is not None and int(n) < int(minimum_n):
        res["outcome"] = nc.INDETERMINATE
        res["reason"] = "below_minimum_n"
        if "sensitivity_outcome" in res:
            res["sensitivity_outcome"] = nc.INDETERMINATE
    return res


# v1 spellings of v2 reason codes (spec V2-4 renamed the bearer-coherence
# failure of the v1 evidence layer to the single-source code).
REASON_ALIASES = {"SOURCE_INCOHERENT": ("BEARER_MISMATCH:COHERENCE",)}


def _has_reason(text, code):
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return False
    parts = [s.strip() for s in str(text).split(";")]
    codes = (code,) + REASON_ALIASES.get(code, ())
    return any(s == c or s.startswith(c + ":") for s in parts for c in codes)


def eval_hypothesis(h, df, defaults, null_rates=None) -> dict:
    """One hypothesis on one stratum of the results."""
    p = _params(h, defaults)
    stat = h["statistic"]
    alpha = float(p["alpha"])
    base = {"hypothesis": h["id"], "statistic": stat, "counts_for_stance":
            bool(h["counts_for_stance"]), "level": h["level"]}
    try:
        if stat == "null_false_present_rate":
            if null_rates is None or null_rates.empty:
                return {**base, "outcome": NOT_EVALUABLE,
                        "reason": "no_null_calibration"}
            band = float(p.get("null_band", 0.02))
            r = null_rates
            if "status_impl" in r.columns:
                impl = r["status_impl"].astype(str)
                foreign = sorted(set(impl[impl != "evidence"]))
                if foreign:
                    # statuses from the local fallback rule, not the installed
                    # (registered) evidence layer, cannot calibrate it
                    return {**base, "outcome": NOT_EVALUABLE,
                            "reason": "statuses_not_from_evidence_layer:"
                                      + ",".join(foreign)}
            n_min = int(r["n"].min())
            within = (r["false_present_rate"] - alpha).abs() <= band + 1e-12
            outside = (r["false_present_lo"] > alpha + band) | (
                r["false_present_hi"] < alpha - band)
            if outside.any():
                outcome, reason = nc.FALSIFIED, "rate_ci_outside_band"
            elif within.all():
                outcome, reason = nc.SUPPORTED, "all_rates_within_band"
            else:
                outcome, reason = nc.INDETERMINATE, "some_rates_outside_band"
            res = {**base, "outcome": outcome, "reason": reason,
                   "rate": float(r["false_present_rate"].max()),
                   "n_cells": int(len(r)), "n_cells_outside": int(outside.sum())}
            return _gate_minimum_n(res, n_min, h.get("minimum_n"))
        lab = report_labels(df)
        known = ~np.isnan(lab)
        n_unknown = int((~known).sum())
        if n_unknown:
            df = df[known]
        pos = lab[known].astype(bool)
        base["n_report_unknown"] = n_unknown
        sens = p.get("sensitivity_missing")
        if stat == "excluded_rate_report_positive":
            res = nc.verdict_level_summary(df["MPC_verdict"], pos, p["epsilon"], alpha,
                                           missing=p["missing"])
            out = {**base, "outcome": res["outcome"], "reason": res["reason"],
                   "rate": res["rate"], "lower": res["lower"], "upper": res["upper"],
                   "coverage": res["coverage"]}
            if sens:
                alt = nc.verdict_level_summary(df["MPC_verdict"], pos, p["epsilon"],
                                               alpha, missing=sens)
                out.update(sensitivity_missing=sens,
                           sensitivity_outcome=alt["outcome"])
            return _gate_minimum_n(out, res["n_positive"], h.get("minimum_n"))
        if stat == "absent_rate_report_positive":
            col = f"{h['principle']}_status"
            if col not in df.columns:
                return {**base, "outcome": NOT_EVALUABLE, "reason": f"missing_{col}"}
            floor = float(p.get("variation_floor", 0.0))
            res = nc.necessity_from_statuses(
                df[col], pos, p["tau"], alpha, missing=p["missing"],
                variation_floor=floor)
            out = {**base, "outcome": res["outcome"], "reason": res["reason"],
                   "rate": res["rate"], "lower": res["lower"], "upper": res["upper"],
                   "varies": res["varies"], "principle": h["principle"]}
            if sens:
                alt = nc.necessity_from_statuses(df[col], pos, p["tau"], alpha,
                                                 missing=sens, variation_floor=floor)
                out.update(sensitivity_missing=sens,
                           sensitivity_outcome=alt["outcome"])
            return _gate_minimum_n(out, res["n_positive"], h.get("minimum_n"))
        if stat == "reason_rate_report_positive":
            if "MPC_reason" not in df.columns:
                return {**base, "outcome": NOT_EVALUABLE,
                        "reason": "missing_MPC_reason"}
            hit = df["MPC_reason"].map(lambda t: _has_reason(t, h["reason_code"]))
            k = int(np.sum(pos & hit.to_numpy()))
            n = int(pos.sum())
            res = nc.symmetric_necessity(k, n, p["epsilon"], alpha, varies=True)
            out = {**base, "outcome": res["outcome"], "reason": res["reason"],
                   "rate": res["rate"], "lower": res["lower"], "upper": res["upper"]}
            return _gate_minimum_n(out, n, h.get("minimum_n"))
        if stat == "coverage":
            v = df["MPC_verdict"].map(nc.normalize_verdict)
            mask = np.ones(len(df), dtype=bool)
            if p.get("population") == "report_positive":
                mask = pos
            elif p.get("population") == "report_negative":
                mask = ~pos
            undet = (v[mask] != nc.EXCLUDED) & (v[mask] != nc.MPC_CONSISTENT)
            n, k = int(mask.sum()), int(undet.sum())
            res = nc.symmetric_necessity(k, n, 1.0 - float(p["kappa"]), alpha,
                                         varies=True)
            out = {**base, "outcome": res["outcome"], "reason": res["reason"],
                   "rate": 1.0 - res["rate"] if n else float("nan"),
                   "lower": 1.0 - res["upper"], "upper": 1.0 - res["lower"]}
            return _gate_minimum_n(out, n, h.get("minimum_n"))
        if stat == "selective_accuracy_gap":
            return _gate_minimum_n(
                {**base, **_accuracy_gap(df, pos, h, p, alpha)},
                int(len(df)), h.get("minimum_n"))
        if stat == "aggregation_exponent":
            res = {**base, **_aggregation_exponent(df, h, p)}
            if res["outcome"] != AUXILIARY_REPORTED:
                return res
            # the estimate is still reported, but not as a registered result
            return _gate_minimum_n(res, res["n"], h.get("minimum_n"))
    except KeyError as exc:
        return {**base, "outcome": NOT_EVALUABLE, "reason": f"missing_column:{exc}"}
    return {**base, "outcome": NOT_EVALUABLE, "reason": f"unknown_statistic:{stat}"}


_DECISION_TRUE = ("true", "1", "1.0", "yes", "positive", "consistent",
                  "mpc_consistent", "not_excluded")
_DECISION_FALSE = ("false", "0", "0.0", "no", "negative", "excluded")
_DECISION_ABSTAIN = ("", "nan", "none", "abstain", "undetermined")


def _decisions_from_column(col):
    """
    Comparator decisions: True (not excluded), False (excluded) or None
    (abstain). Booleans, 0/1 numbers (a CSV column with abstentions is read
    as float) and the labels above are accepted; anything else raises.
    """
    out = []
    for v in col:
        if v is None:
            out.append(None)
        elif isinstance(v, (bool, np.bool_)):
            out.append(bool(v))
        elif isinstance(v, (int, float, np.integer, np.floating)):
            f = float(v)
            if math.isnan(f):
                out.append(None)
            elif f in (0.0, 1.0):
                out.append(f == 1.0)
            else:
                raise ValueError(f"comparator decision {v!r} is not 0/1")
        else:
            txt = str(v).strip().lower()
            if txt in _DECISION_ABSTAIN:
                out.append(None)
            elif txt in _DECISION_TRUE:
                out.append(True)
            elif txt in _DECISION_FALSE:
                out.append(False)
            else:
                raise ValueError(f"unrecognised comparator decision {v!r}")
    return out


def _accuracy(dec, truth, idx):
    made = [(dec[i], truth[i]) for i in idx if dec[i] is not None]
    if not made:
        return float("nan")
    return float(np.mean([d == t for d, t in made]))


def _best(values):
    """Largest finite value (NaN when none is finite; order-independent)."""
    fin = [v for v in values if math.isfinite(v)]
    return max(fin) if fin else float("nan")


def _accuracy_gap(df, pos, h, p, alpha):
    v = df["MPC_verdict"].map(nc.normalize_verdict).tolist()
    impact = [None if x not in (nc.EXCLUDED, nc.MPC_CONSISTENT) else
              (x == nc.MPC_CONSISTENT) for x in v]
    comps = {}
    for name in h.get("comparators") or []:
        col = f"comparator_{name}"
        if col in df.columns:
            try:
                comps[name] = _decisions_from_column(df[col].tolist())
            except ValueError as exc:
                return {"outcome": NOT_EVALUABLE,
                        "reason": f"invalid_comparator_column:{col}:{exc}"}
    if not comps:
        return {"outcome": NOT_EVALUABLE, "reason": "no_comparator_columns"}
    truth = list(pos)
    idx = np.arange(len(df))

    def _gap(sel):
        best = _best([_accuracy(d, truth, sel) for d in comps.values()])
        return _accuracy(impact, truth, sel) - best

    point = _gap(idx)
    rng = np.random.default_rng(int(p.get("seed", 0)))
    boots = np.asarray([_gap(rng.integers(0, len(df), len(df)))
                        for _ in range(int(p.get("n_boot", 2000)))])
    boots = boots[np.isfinite(boots)]
    if boots.size < 10 or not math.isfinite(point):
        return {"outcome": nc.INDETERMINATE, "reason": "too_few_decisions",
                "rate": point}
    lo, hi = float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))
    margin = float(p["margin"])
    if lo > -margin:
        outcome, reason = nc.SUPPORTED, "lower_bound_above_minus_margin"
    elif hi < -margin:
        outcome, reason = nc.FALSIFIED, "upper_bound_below_minus_margin"
    else:
        outcome, reason = nc.INDETERMINATE, "interval_contains_minus_margin"
    cov = float(np.mean([d is not None for d in impact]))
    return {"outcome": outcome, "reason": reason, "rate": point, "lower": lo,
            "upper": hi, "coverage": cov, "comparators_used": ",".join(sorted(comps))}


def _aggregation_exponent(df, h, p):
    from simulate_rule_recovery import fit_exponent

    nset = [x for x in PRINCIPLES]
    cols = [f"{x}_c" for x in nset]
    ycol = h.get("outcome_column")
    missing = [c for c in cols + [ycol] if c is None or c not in df.columns]
    if missing:
        return {"outcome": NOT_EVALUABLE, "reason": f"missing_columns:{missing}"}
    v = df["MPC_verdict"].map(nc.normalize_verdict)
    sub = df[(v == nc.MPC_CONSISTENT).to_numpy()]
    C = sub[cols].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    ok = np.all(np.isfinite(C), axis=1) & np.isfinite(y)
    if ok.sum() < 10:
        return {"outcome": NOT_EVALUABLE, "reason": "too_few_consistent_rows",
                "n": int(ok.sum())}
    fit = fit_exponent(np.maximum(C[ok], 0.01), y[ok])
    return {"outcome": AUXILIARY_REPORTED, "reason": "auxiliary_not_counted",
            "rate": fit["p_hat"], "lower": fit["ci_lo"], "upper": fit["ci_hi"],
            "n": int(ok.sum())}


def dataset_roles(reg):
    return {d["id"]: d["role"] for d in reg.get("datasets") or []}


def stance_summary(rows):
    counted = [r for r in rows if r["counts_for_stance"]]
    if not counted:
        return {"stance": "NOT_EVALUATED"}
    fals = [r["hypothesis"] for r in counted if r["outcome"] == nc.FALSIFIED]
    supp = [r["hypothesis"] for r in counted if r["outcome"] == nc.SUPPORTED]
    if fals:
        stance = "FALSIFIED"
    elif len(supp) == len(counted):
        stance = "SUPPORTED"
    else:
        stance = "INDETERMINATE"
    return {"stance": stance, "falsified_by": fals, "supported": supp,
            "n_counted": len(counted)}


def evaluate(reg, results: pd.DataFrame, *, null_rates=None, exploratory=False,
             allow_unregistered=False) -> dict:
    """
    Evaluate every hypothesis. Raises :class:`RegistryRefusal` on the refusal
    conditions of the module docstring except the freeze-tag check, which
    needs the registry file and is done by :func:`verify_freeze_tag` (the CLI
    runs it before this function). Episodes with a missing or unrecognised
    report label are excluded (counted in ``n_report_unknown``). Returns
    ``rows`` (one per hypothesis and stratum), ``stance`` per stratum and
    ``registration_problems``.
    """
    errs = validate_registry(reg)
    if errs:
        raise RegistryRefusal([f"registry invalid: {e}" for e in errs])
    if not exploratory and reg.get("status") != "frozen":
        raise RegistryRefusal(["the registry is a draft: confirmatory evaluation "
                               "needs status 'frozen' (use --exploratory)"])
    probs = registration_problems(results, reg)
    if probs and not allow_unregistered:
        raise RegistryRefusal([f"results not registered: {p}" for p in probs])
    roles = dataset_roles(reg)
    ds = results["dataset"].astype(str).map(roles) if "dataset" in results else None
    strata = {}
    if exploratory or probs:
        strata["exploratory"] = results
    else:
        conf = results[(ds == "confirmatory").to_numpy()]
        expl = results[(ds != "confirmatory").to_numpy()]
        if len(conf):
            strata["confirmatory"] = conf
        if len(expl):
            strata["exploratory"] = expl
    defaults = dict(reg["defaults"])
    rows, stance = [], {}
    for name, df in strata.items():
        srows = []
        for h in reg["hypotheses"]:
            r = eval_hypothesis(h, df, defaults, null_rates)
            r["stratum"] = name
            r["unregistered"] = bool(probs)
            srows.append(r)
        rows.extend(srows)
        stance[name] = stance_summary(srows)
    return {"rows": rows, "stance": stance, "registration_problems": probs}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _provenance():
    try:
        from impact_pipeline.provenance import collect_code_version

        info = collect_code_version(REPO_ROOT)
    except Exception as exc:  # noqa: BLE001 - provenance must not fail a run
        info = {"code_version": "unknown", "error": str(exc)}
    info["script_sha256"] = file_sha256(__file__)
    return info


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    ap.add_argument("--results", default=None)
    ap.add_argument("--null-calibration", default=None,
                    help="null_calibration_rates.csv for H0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exploratory", action="store_true",
                    help="evaluate a draft registry; every outcome is exploratory")
    ap.add_argument("--allow-unregistered", action="store_true",
                    help="evaluate unregistered results, labelled exploratory")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--skip-freeze-tag-check", action="store_true",
                    help="do not verify the freeze tag (recorded in the report)")
    ap.add_argument("--hash-protocol", default=None,
                    help="print the canonical JSON SHA-256 of a protocol JSON file")
    args = ap.parse_args(argv)

    if args.hash_protocol:
        obj = json.loads(Path(args.hash_protocol).read_text(encoding="utf-8"))
        print(canonical_json_sha256(obj))
        return 0
    try:
        reg = load_registry(args.registry)
        errs = validate_registry(reg, load_schema(args.schema))
    except RegistryRefusal as exc:
        errs = exc.reasons
        reg = None
    if args.validate_only or args.results is None:
        print(json.dumps({"registry": args.registry, "valid": not errs,
                          "errors": errs}, indent=2))
        return 0 if not errs else REFUSAL_EXIT
    out = Path(args.out or "predictions_out")
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": RUNNER_VERSION,
        "registry": str(Path(args.registry).resolve()),
        "registry_sha256": file_sha256(args.registry),
        "results": str(Path(args.results).resolve()),
        "results_sha256": file_sha256(args.results),
        "mode": "exploratory" if args.exploratory else "confirmatory",
        "allow_unregistered": bool(args.allow_unregistered),
        "provenance": _provenance(),
    }
    if errs:
        _write_json(out / "predictions_refusal.json", {**meta, "reasons": errs})
        print("REFUSED: " + "; ".join(errs), file=sys.stderr)
        return REFUSAL_EXIT
    if not args.exploratory:
        meta["freeze_tag_checked"] = not args.skip_freeze_tag_check
        # a draft registry is refused by evaluate() with its own reason
        check = not args.skip_freeze_tag_check and reg.get("status") == "frozen"
        tag_probs = verify_freeze_tag(reg, args.registry) if check else []
        if tag_probs:
            _write_json(out / "predictions_refusal.json",
                        {**meta, "reasons": tag_probs})
            print("REFUSED: " + "; ".join(tag_probs), file=sys.stderr)
            return REFUSAL_EXIT
    results = pd.read_csv(args.results)
    null_rates = pd.read_csv(args.null_calibration) if args.null_calibration else None
    if args.null_calibration:
        meta["null_calibration_sha256"] = file_sha256(args.null_calibration)
    try:
        res = evaluate(reg, results, null_rates=null_rates,
                       exploratory=args.exploratory,
                       allow_unregistered=args.allow_unregistered)
    except RegistryRefusal as exc:
        _write_json(out / "predictions_refusal.json", {**meta, "reasons": exc.reasons})
        print("REFUSED: " + "; ".join(exc.reasons), file=sys.stderr)
        return REFUSAL_EXIT
    pd.DataFrame(res["rows"]).to_csv(out / "predictions_results.csv", index=False)
    _write_json(out / "predictions_report.json",
                {**meta, "stance": res["stance"],
                 "registration_problems": res["registration_problems"],
                 "registry_status": reg.get("status"),
                 "freeze_tag": reg.get("freeze_tag")})
    print(json.dumps(res["stance"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
