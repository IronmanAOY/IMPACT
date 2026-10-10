"""
Testability gating, anchors and necessity sets of MPC-Bench v2.

These are the pre-freeze rules that read development data only (reference
blocks, development dry runs; seeds 0-999) and write hash-covered blocks of
the ``impact-mpc-protocol/3`` family protocols. They never read a
confirmatory seed and never read the construct value ``c`` of a target: the
gate reads statuses and precision only.

**Testability gating** (preregistration v2, section 3.7). For every
ABSENT-type hypothesis part (target ABSENT on an own-lesion or construct-null
witness) the development rate ``pi0`` is the empirical ABSENT rate of the
target on the witness under the frozen v2 rule (n >= 40 runs); the analytic
``mean max(0, 2 Phi(delta / se_c - q_A) - 1)`` is reported beside it. For
every "PRESENT in >= 80 %" part, ``pi1`` is the empirical PRESENT rate on the
positive control (analytic ``mean Phi((1 - z) / se_c - q_P)`` beside it). A
part is decisive only if its rate is >= 0.9; otherwise it is
``NOT_TESTABLE_BY_DESIGN`` before the freeze and replaced by the weaker claim
("not PRESENT in >= 80 %" for ABSENT parts, a paired contrast with the
positive control for PRESENT parts), and its predicted
``ABSENT_NOT_REACHABLE`` rate is written down. The rows form the protocol's
``precision`` block (:func:`precision_block`).

**Anchors and necessity sets** (preregistration v2, section 3.6). On the
development reference block (seeds 900-939) a principle's anchor is valid
iff at least 36 of the 40 excesses of PC_nominal are finite and the one-sided
95 % Student-t lower bound of their mean is > 0 (:func:`anchor_validity`);
it is specific iff the mean paired contrast PC minus own lesion on the same
seeds is >= 0.5 x the anchor and its one-sided 95 % lower bound is > 0
(:func:`anchor_specificity`; otherwise ``NONSPECIFIC_ANCHOR``). ``N_anch`` =
the declared principles that are valid and specific; the fallback is F0 (all
declared principles anchored), F1 (two or more but not all), F2 (one) or F3
(none). The block is the protocol's ``anchors`` field
(:func:`anchors_block`); ``alpha_A`` does not depend on ``N_anch``.

**Mechanism on** (preregistration v2, section 6.4). A sweep dose counts as
"mechanism on" for a principle and protocol only if it is at least 0.5 x the
nominal dose and the development median ``c`` at that dose is at least
``2 delta`` (:func:`mechanism_on`); only such rows enter the false-exclusion
checks.

**Implied precision.** ``s_A = delta / q_A`` is the largest ``se_c`` at which
the TOST can pass at ``c = 0`` (:func:`absent_precision`), ``s_P = (1 - z) /
q_P`` the largest at which a reference-level component can be PRESENT
(:func:`present_precision`).
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from impact_pipeline.v2 import PRINCIPLES
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import seeds as S

PRECISION_SCHEMA = "impact-mpc-precision/1"
ANCHORS_SCHEMA = "impact-mpc-anchors/1"
DECISIVE_THRESHOLD = 0.9
MIN_RUNS = 40
KIND_ABSENT, KIND_PRESENT = "absent", "present"
KINDS = (KIND_ABSENT, KIND_PRESENT)
DECISIVE = "decisive"
NOT_TESTABLE_BY_DESIGN = "NOT_TESTABLE_BY_DESIGN"
OUTCOMES = (DECISIVE, NOT_TESTABLE_BY_DESIGN)
REPLACEMENT_CLAIMS = {
    KIND_ABSENT: "target not PRESENT in >= 80 % of seeds",
    KIND_PRESENT: "paired contrast with the positive control",
}
RATE_KEYS = (
    "PRESENT", "ABSENT", "UNDEFINED", "INCONCLUSIVE", "ABSENT_NOT_REACHABLE",
    "NULL_MODEL_VIOLATED", "INVALID_ANCHORS", "PRESENT_NOT_REACHABLE",
)
# anchors
ANCHOR_BLOCK_SIZE = 40
ANCHOR_MIN_FINITE = 36
ANCHOR_ALPHA = 0.05
SPECIFICITY_RATIO = 0.5
REFERENCE_SEEDS = (S.REFERENCE_BLOCK.start, S.REFERENCE_BLOCK.stop - 1)
ANCHOR_VALID_SPECIFIC = "valid_specific"
ANCHOR_VALID_NONSPECIFIC = "valid_nonspecific"
ANCHOR_INVALID = "invalid"
ANCHOR_STATUSES = (ANCHOR_VALID_SPECIFIC, ANCHOR_VALID_NONSPECIFIC, ANCHOR_INVALID)
ANCHOR_RULE = {
    "block_size": ANCHOR_BLOCK_SIZE,
    "min_finite": ANCHOR_MIN_FINITE,
    "alpha": ANCHOR_ALPHA,
    "specificity_ratio": SPECIFICITY_RATIO,
}

_ROW_KEYS = (
    "family", "principle", "witness", "kind", "n", "members", "pi", "pi_analytic",
    "s_A", "s_P", "attainable_se_c", "df", "rates", "decisive", "outcome",
    "replacement", "predicted_absent_not_reachable_rate",
)
_ROW_FLOATS = ("pi", "pi_analytic", "s_A", "s_P", "attainable_se_c", "df",
               "predicted_absent_not_reachable_rate")


class TestabilityError(ValueError):
    """Development data that the gating or anchor rules cannot use (too few
    runs, a non-development seed, a malformed status)."""


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------
def _ev():
    from impact_pipeline import evidence_v2 as E

    return E


def _finite_or_none(v) -> Optional[float]:
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def absent_precision(df=math.inf, *, delta=0.10, alpha_absent=0.01, members=1) -> float:
    """``s_A = delta / q_A`` with ``q_A = t(df, 1 - alpha_absent / members)``:
    the largest ``se_c`` at which the TOST can pass at ``c = 0``."""
    return float(delta) / _ev().quantile(float(alpha_absent) / int(members), df)


def present_precision(df=math.inf, *, z=0.25, alpha=0.05) -> float:
    """``s_P = (1 - z) / q_P``: the largest ``se_c`` at which a component at
    the reference (``c = 1``) can be PRESENT."""
    return (1.0 - float(z)) / _ev().quantile(float(alpha), df)


def _se_df_arrays(se_c, df):
    se = np.asarray(se_c, dtype=float).ravel()
    if df is None:
        d = np.full(se.shape, np.inf)
    elif np.ndim(df) == 0:
        d = np.full(se.shape, float(df))
    else:
        d = np.asarray(df, dtype=float).ravel()
    if d.shape != se.shape:
        raise ValueError("se_c and df must have the same length")
    d = np.where(np.isfinite(d) & (d > 0), d, np.inf)
    ok = np.isfinite(se) & (se >= 0)
    return se[ok], d[ok]


def pi0_analytic(se_c, df=None, *, delta=0.10, alpha_absent=0.01, members=1) -> float:
    """Mean over runs of ``max(0, 2 Phi(delta / se_c - q_A) - 1)``, the TOST
    power at ``c = 0`` with each run's own ``se_c`` and ``df`` (NaN without a
    usable run)."""
    from scipy.special import ndtr

    se, d = _se_df_arrays(se_c, df)
    if se.size == 0:
        return float("nan")
    q = _ev()._quantiles(float(alpha_absent) / int(members), d)
    with np.errstate(divide="ignore"):
        x = np.where(se > 0, float(delta) / np.where(se > 0, se, 1.0), np.inf)
    return float(np.mean(np.maximum(0.0, 2.0 * ndtr(x - q) - 1.0)))


def pi1_analytic(se_c, df=None, *, z=0.25, alpha=0.05) -> float:
    """Mean over runs of ``Phi((1 - z) / se_c - q_P)``, the power to call a
    reference-level component PRESENT (NaN without a usable run)."""
    from scipy.special import ndtr

    se, d = _se_df_arrays(se_c, df)
    if se.size == 0:
        return float("nan")
    q = _ev()._quantiles(float(alpha), d)
    with np.errstate(divide="ignore"):
        x = np.where(se > 0, (1.0 - float(z)) / np.where(se > 0, se, 1.0), np.inf)
    return float(np.mean(ndtr(x - q)))


# --------------------------------------------------------------------------
# development runs and the gate
# --------------------------------------------------------------------------
def _check_runs(runs, min_runs) -> list:
    out = []
    for i, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise TestabilityError(f"run {i}: expected a mapping")
        status = run.get("status")
        flags = tuple(run.get("flags") or ())
        try:
            R.check_status(status, run.get("reason"), flags)
        except ValueError as exc:
            raise TestabilityError(f"run {i}: {exc}") from None
        if run.get("seed") is not None:
            try:
                split = S.split_of(run["seed"])
            except S.SeedPolicyError as exc:
                raise TestabilityError(f"run {i}: {exc}") from None
            if split != S.DEVELOPMENT:
                raise TestabilityError(
                    f"run {i}: seed {run['seed']} is not a development seed; the gate "
                    "reads development data only")
        out.append(run)
    if len(out) < int(min_runs):
        raise TestabilityError(f"{len(out)} runs; the gate needs at least {min_runs}")
    return out


def status_rates(runs) -> dict:
    """Rates of the statuses, of the UNDEFINED reasons named in
    :data:`RATE_KEYS` and of the ``PRESENT_NOT_REACHABLE`` flag."""
    runs = list(runs)
    n = len(runs)
    out = {k: 0 for k in RATE_KEYS}
    for run in runs:
        out[run["status"]] += 1
        if run["status"] == R.UNDEFINED:
            code = R.parse_reason(run.get("reason"))[0]
            if code in out:
                out[code] += 1
        if R.PRESENT_NOT_REACHABLE in tuple(run.get("flags") or ()):
            out["PRESENT_NOT_REACHABLE"] += 1
    return {k: (v / n if n else None) for k, v in out.items()}


def gate(kind, pi, threshold=DECISIVE_THRESHOLD) -> dict:
    """``{"decisive", "outcome", "replacement"}``: decisive iff
    ``pi >= threshold``; otherwise ``NOT_TESTABLE_BY_DESIGN`` with the
    replacement claim of the kind."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    p = _finite_or_none(pi)
    decisive = p is not None and p >= float(threshold)
    return {
        "decisive": decisive,
        "outcome": DECISIVE if decisive else NOT_TESTABLE_BY_DESIGN,
        "replacement": None if decisive else REPLACEMENT_CLAIMS[kind],
    }


def testability_row(*, family, principle, witness, kind, runs, rule=None,
                    cutoff=None, members=1, threshold=DECISIVE_THRESHOLD,
                    min_runs=MIN_RUNS) -> dict:
    """
    One row of the testability table from development runs of a (family
    protocol, principle, witness): mappings with ``status``, ``reason``,
    ``flags``, ``se_c``, ``df_c`` and optionally ``seed`` (component records
    computed under the frozen v2 rule). ``kind`` is ``absent`` (``pi0``: the
    ABSENT rate) or ``present`` (``pi1``: the PRESENT rate); ``members`` is
    the number of union members of the TOST (2 for the NAS directions).
    """
    E = _ev()
    rule = E._coerce_rule(rule)
    z, delta = E.v1.normalize_cutoff(E.DEFAULT_CUTOFF if cutoff is None else cutoff)
    if principle not in PRINCIPLES:
        raise ValueError(f"unknown principle {principle!r}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    runs = _check_runs(runs, min_runs)
    rates = status_rates(runs)
    se = np.asarray([E.v1._as_float(r.get("se_c")) for r in runs], dtype=float)
    df = np.asarray([E.v1._as_float(r.get("df_c")) for r in runs], dtype=float)
    fin_se = se[np.isfinite(se)]
    fin_df = df[np.isfinite(df) & (df > 0)]
    df_med = float(np.median(fin_df)) if fin_df.size else math.inf
    if kind == KIND_ABSENT:
        pi = rates["ABSENT"]
        pi_an = pi0_analytic(se, df, delta=delta, alpha_absent=rule.alpha_absent,
                             members=members)
        predicted_anr = rates["ABSENT_NOT_REACHABLE"]
    else:
        pi = rates["PRESENT"]
        pi_an = pi1_analytic(se, df, z=z, alpha=rule.alpha)
        predicted_anr = None
    g = gate(kind, pi, threshold)
    row = {
        "family": str(family),
        "principle": principle,
        "witness": str(witness),
        "kind": kind,
        "n": len(runs),
        "members": int(members),
        "pi": pi,
        "pi_analytic": _finite_or_none(pi_an),
        "s_A": absent_precision(df_med, delta=delta, alpha_absent=rule.alpha_absent,
                                members=members),
        "s_P": present_precision(df_med, z=z, alpha=rule.alpha),
        "attainable_se_c": float(np.median(fin_se)) if fin_se.size else None,
        "df": _finite_or_none(df_med),
        "rates": rates,
        **g,
        "predicted_absent_not_reachable_rate": predicted_anr,
    }
    return validate_precision_row(row, min_runs=min_runs)


def validate_precision_row(row, *, min_runs=MIN_RUNS) -> dict:
    """A testability row checked and normalised (plain JSON types)."""
    if not isinstance(row, Mapping):
        raise ValueError("a precision row must be an object")
    unknown = sorted(set(row) - set(_ROW_KEYS))
    missing = sorted(set(_ROW_KEYS) - set(row))
    if unknown or missing:
        raise ValueError(f"precision row: unknown keys {unknown}, missing {missing}")
    out = {k: row[k] for k in _ROW_KEYS}
    for k in ("family", "witness"):
        if not isinstance(out[k], str) or not out[k]:
            raise ValueError(f"precision row {k} must be a non-empty string")
    if out["principle"] not in PRINCIPLES:
        raise ValueError(f"precision row principle must be one of {PRINCIPLES}")
    if out["kind"] not in KINDS:
        raise ValueError(f"precision row kind must be one of {KINDS}")
    for k in ("n", "members"):
        if isinstance(out[k], bool) or not isinstance(out[k], (int, np.integer)):
            raise ValueError(f"precision row {k} must be an integer")
        out[k] = int(out[k])
    if out["n"] < int(min_runs):
        raise ValueError(f"precision row n must be >= {min_runs}")
    if out["members"] < 1:
        raise ValueError("precision row members must be >= 1")
    for k in _ROW_FLOATS:
        if out[k] is not None:
            if isinstance(out[k], bool):
                raise ValueError(f"precision row {k} must be a number or null")
            out[k] = _finite_or_none(out[k])
    for k in ("pi", "pi_analytic", "predicted_absent_not_reachable_rate"):
        if out[k] is not None and not (0.0 <= out[k] <= 1.0):
            raise ValueError(f"precision row {k} must lie in [0, 1]")
    rates = out["rates"]
    if not isinstance(rates, Mapping) or set(rates) != set(RATE_KEYS):
        raise ValueError(f"precision row rates must have the keys {RATE_KEYS}")
    out["rates"] = {k: _finite_or_none(rates[k]) for k in RATE_KEYS}
    if not isinstance(out["decisive"], bool):
        raise ValueError("precision row decisive must be true or false")
    if out["outcome"] != (DECISIVE if out["decisive"] else NOT_TESTABLE_BY_DESIGN):
        raise ValueError("precision row outcome contradicts decisive")
    want = None if out["decisive"] else REPLACEMENT_CLAIMS[out["kind"]]
    if out["replacement"] != want:
        raise ValueError(
            "precision row replacement must be the kind's replacement claim")
    return out


def _row_key(row):
    return (row["family"], row["principle"], row["witness"], row["kind"])


def precision_block(rows, *, threshold=DECISIVE_THRESHOLD, min_runs=MIN_RUNS) -> dict:
    """The hash-covered ``precision`` block of a family protocol: the gate's
    threshold and minimum run count and the testability rows in canonical
    order. A row whose gate outcome disagrees with the threshold is
    refused."""
    return validate_precision_block({
        "schema": PRECISION_SCHEMA, "threshold": float(threshold),
        "min_runs": int(min_runs), "rows": list(rows)})


def validate_precision_block(block) -> dict:
    """The ``precision`` block checked and normalised (rows sorted by
    family, principle, witness and kind; one row per key)."""
    if not isinstance(block, Mapping):
        raise ValueError("the precision block must be an object")
    unknown = sorted(set(block) - {"schema", "threshold", "min_runs", "rows"})
    if unknown:
        raise ValueError(f"precision block has unknown keys {unknown}")
    if block.get("schema") != PRECISION_SCHEMA:
        raise ValueError(f"precision block schema must be {PRECISION_SCHEMA!r}")
    thr = _finite_or_none(block.get("threshold"))
    if thr is None or not (0.0 < thr <= 1.0):
        raise ValueError("precision threshold must lie in (0, 1]")
    mr = block.get("min_runs")
    if isinstance(mr, bool) or not isinstance(mr, (int, np.integer)) or mr < 1:
        raise ValueError("precision min_runs must be a positive integer")
    rows = block.get("rows")
    if isinstance(rows, (str, Mapping)) or not isinstance(rows, Sequence):
        raise ValueError("precision rows must be a list")
    out_rows, seen = [], set()
    for r in rows:
        row = validate_precision_row(r, min_runs=int(mr))
        if row["decisive"] != (row["pi"] is not None and row["pi"] >= thr):
            raise ValueError(f"precision row {_row_key(row)}: decisive disagrees with "
                             "the threshold")
        key = _row_key(row)
        if key in seen:
            raise ValueError(f"duplicate precision row {key}")
        seen.add(key)
        out_rows.append(row)
    out_rows.sort(key=_row_key)
    return {"schema": PRECISION_SCHEMA, "threshold": thr, "min_runs": int(mr),
            "rows": out_rows}


# --------------------------------------------------------------------------
# "mechanism on" labels of sweep doses
# --------------------------------------------------------------------------
MECHANISM_ON_DOSE_RATIO = 0.5
MECHANISM_ON_C_RATIO = 2.0


def development_median_c(runs) -> Optional[float]:
    """The median finite ``c`` of development runs (mappings with ``c`` and
    optionally ``seed``; a non-development seed is refused). None without a
    finite value."""
    values = []
    for i, run in enumerate(runs):
        if run.get("seed") is not None:
            try:
                split = S.split_of(run["seed"])
            except S.SeedPolicyError as exc:
                raise TestabilityError(f"run {i}: {exc}") from None
            if split != S.DEVELOPMENT:
                raise TestabilityError(
                    f"run {i}: seed {run['seed']} is not a development seed")
        c = _finite_or_none(run.get("c"))
        if c is not None:
            values.append(c)
    return float(np.median(values)) if values else None


def mechanism_on(dose, nominal_dose, development_median_c, *, delta=0.10,
                 dose_ratio=MECHANISM_ON_DOSE_RATIO,
                 c_ratio=MECHANISM_ON_C_RATIO) -> bool:
    """
    Whether a sweep dose counts as "mechanism on" for a principle and
    protocol: the dose is at least ``dose_ratio`` (0.5) x the nominal dose
    and the development median ``c`` at that dose is at least ``c_ratio``
    (2) x ``delta``. A missing median is not "on".
    """
    d, nom = float(dose), float(nominal_dose)
    if not (math.isfinite(d) and math.isfinite(nom) and nom > 0):
        raise ValueError("dose and a positive nominal dose are required")
    med = _finite_or_none(development_median_c)
    return bool(d >= dose_ratio * nom and med is not None
                and med >= c_ratio * float(delta))


# --------------------------------------------------------------------------
# anchors and necessity sets
# --------------------------------------------------------------------------
def _t_lower(values, alpha):
    """``(n, mean, se, df, lower)`` of the one-sided ``1 - alpha`` Student-t
    lower bound of the mean (NaN with fewer than two values)."""
    x = np.asarray(values, dtype=float)
    n = int(x.size)
    if n < 2:
        mean = float(x.mean()) if n else float("nan")
        return n, mean, float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n))
    q = _ev().quantile(alpha, n - 1)
    return n, mean, se, float(n - 1), mean - q * se


def anchor_validity(excess, *, block_size=ANCHOR_BLOCK_SIZE,
                    min_finite=ANCHOR_MIN_FINITE, alpha=ANCHOR_ALPHA) -> dict:
    """
    Anchor validity on a reference block: the PC_nominal excesses of the
    block's ``block_size`` seeds (NaN for a run without a finite excess).
    Valid iff at least ``min_finite`` are finite and the one-sided
    ``1 - alpha`` Student-t lower bound of their mean is > 0.
    """
    x = np.asarray(excess, dtype=float).ravel()
    if x.size != int(block_size):
        raise TestabilityError(f"the reference block has {block_size} seeds, got "
                               f"{x.size} excesses")
    fin = x[np.isfinite(x)]
    n, mean, se, df, lower = _t_lower(fin, alpha)
    valid = n >= int(min_finite) and math.isfinite(lower) and lower > 0
    return {"n": int(x.size), "n_finite": n, "mean": _finite_or_none(mean),
            "se": _finite_or_none(se), "df": _finite_or_none(df),
            "lower": _finite_or_none(lower), "valid": bool(valid)}


def anchor_specificity(pc_excess, lesion_excess, *, anchor=None,
                       ratio=SPECIFICITY_RATIO, alpha=ANCHOR_ALPHA) -> dict:
    """
    Anchor specificity: paired contrasts PC_nominal minus the principle's own
    lesion on the same seeds (pairs with both excesses finite). Specific iff
    the mean contrast is >= ``ratio`` x the anchor (default: the mean finite
    PC excess) and its one-sided ``1 - alpha`` lower bound is > 0.
    """
    pc = np.asarray(pc_excess, dtype=float).ravel()
    les = np.asarray(lesion_excess, dtype=float).ravel()
    if pc.shape != les.shape:
        raise TestabilityError("PC and lesion excesses must be paired by seed")
    ok = np.isfinite(pc) & np.isfinite(les)
    d = pc[ok] - les[ok]
    n, mean, se, df, lower = _t_lower(d, alpha)
    if anchor is None:
        fin = pc[np.isfinite(pc)]
        anchor = float(fin.mean()) if fin.size else float("nan")
    anchor = float(anchor)
    specific = (math.isfinite(mean) and math.isfinite(anchor) and math.isfinite(lower)
                and mean >= float(ratio) * anchor and lower > 0)
    return {"n_pairs": n, "mean_contrast": _finite_or_none(mean),
            "se": _finite_or_none(se), "df": _finite_or_none(df),
            "lower": _finite_or_none(lower), "anchor": _finite_or_none(anchor),
            "ratio": float(ratio), "specific": bool(specific)}


def anchor_entry(validity, specificity=None) -> dict:
    """The anchor entry of a principle: ``invalid`` unless valid;
    ``valid_specific`` iff the specificity gate passed; else
    ``valid_nonspecific`` (no own-lesion contrast counts as not specific)."""
    valid = bool(validity["valid"])
    specific = valid and specificity is not None and bool(specificity["specific"])
    status = (ANCHOR_INVALID if not valid else
              ANCHOR_VALID_SPECIFIC if specific else ANCHOR_VALID_NONSPECIFIC)
    return {"status": status, "valid": valid, "specific": specific,
            "validity": dict(validity),
            "specificity": None if specificity is None else dict(specificity)}


def combine_anchor_entries(members: Mapping) -> dict:
    """The anchor entry of a directional principle (NAS) from its
    per-direction entries: valid iff every direction is valid, specific iff
    every direction is specific."""
    if not members:
        raise ValueError("no direction entries")
    valid = all(bool(e["valid"]) for e in members.values())
    specific = valid and all(bool(e["specific"]) for e in members.values())
    status = (ANCHOR_INVALID if not valid else
              ANCHOR_VALID_SPECIFIC if specific else ANCHOR_VALID_NONSPECIFIC)
    return {"status": status, "valid": valid, "specific": specific,
            "members": {k: dict(v) for k, v in members.items()}}


def necessity_from_anchors(entries: Mapping, declared=PRINCIPLES) -> tuple:
    """``N_anch``: the declared principles whose anchor is valid and specific,
    in canonical order."""
    return tuple(p for p in PRINCIPLES if p in set(declared)
                 and (entries.get(p) or {}).get("status") == ANCHOR_VALID_SPECIFIC)


def fallback_level(n_anch: int, n_decl: int) -> str:
    """F0 (every declared principle anchored), F1 (two or more), F2 (one),
    F3 (none)."""
    n_anch, n_decl = int(n_anch), int(n_decl)
    if not (0 <= n_anch <= n_decl) or n_decl < 1:
        raise ValueError("need 0 <= n_anch <= n_decl and n_decl >= 1")
    if n_anch == n_decl:
        return "F0"
    if n_anch >= 2:
        return "F1"
    return "F2" if n_anch == 1 else "F3"


def anchors_block(entries: Mapping, *, declared=PRINCIPLES,
                  reference_seeds=REFERENCE_SEEDS) -> dict:
    """The hash-covered ``anchors`` block of a family protocol: the rule, the
    reference seeds, one entry per declared principle, ``N_anch`` and the
    fallback. The protocol's ``necessity_set`` is ``N_anch`` (the declared
    set under F3, where verdict parts are not evaluable)."""
    declared = tuple(p for p in PRINCIPLES if p in set(declared))
    n_anch = necessity_from_anchors(entries, declared)
    block = {
        "schema": ANCHORS_SCHEMA,
        "rule": dict(ANCHOR_RULE),
        "reference_seeds": [int(reference_seeds[0]), int(reference_seeds[1])],
        "principles": {p: entries[p] for p in declared if p in entries},
        "necessity_set": list(n_anch),
        "fallback": fallback_level(len(n_anch), len(declared)),
    }
    return validate_anchors_block(block, declared, n_anch or declared)


def validate_anchors_block(block, declared=PRINCIPLES, necessity_set=None) -> dict:
    """The ``anchors`` block checked against the protocol's declared and
    anchored necessity sets (``necessity_set`` must equal ``N_anch``, or the
    declared set under F3) and normalised to plain JSON."""
    E = _ev()
    if not isinstance(block, Mapping):
        raise ValueError("the anchors block must be an object")
    keys = {"schema", "rule", "reference_seeds", "principles", "necessity_set",
            "fallback"}
    unknown, missing = sorted(set(block) - keys), sorted(keys - set(block))
    if unknown or missing:
        raise ValueError(f"anchors block: unknown keys {unknown}, missing {missing}")
    if block["schema"] != ANCHORS_SCHEMA:
        raise ValueError(f"anchors block schema must be {ANCHORS_SCHEMA!r}")
    if dict(block["rule"]) != ANCHOR_RULE:
        raise ValueError(f"anchors block rule must be {ANCHOR_RULE}")
    seeds = list(block["reference_seeds"])
    if len(seeds) != 2 or seeds[0] > seeds[1]:
        raise ValueError("reference_seeds must be [first, last]")
    try:
        S.assert_development(seeds)
    except S.SeedPolicyError as exc:
        raise ValueError(f"anchors block: {exc}") from None
    declared = tuple(p for p in PRINCIPLES if p in set(declared))
    principles = dict(block["principles"])
    if set(principles) != set(declared):
        raise ValueError("the anchors block needs one entry per declared principle")
    for p, entry in principles.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"anchor entry of {p} must be an object")
        st = entry.get("status")
        if st not in ANCHOR_STATUSES:
            raise ValueError(f"anchor status of {p} must be one of {ANCHOR_STATUSES}")
        if entry.get("valid") is not (st != ANCHOR_INVALID) or entry.get(
                "specific") is not (st == ANCHOR_VALID_SPECIFIC):
            raise ValueError(
                f"anchor entry of {p}: valid/specific contradict the status")
    n_anch = necessity_from_anchors(principles, declared)
    if list(block["necessity_set"]) != list(n_anch):
        raise ValueError(f"anchors block necessity_set must be N_anch = {list(n_anch)}")
    fb = fallback_level(len(n_anch), len(declared))
    if block["fallback"] != fb:
        raise ValueError(f"anchors block fallback must be {fb}")
    if necessity_set is not None:
        want = n_anch if n_anch else declared
        if tuple(necessity_set) != tuple(want):
            raise ValueError(f"the protocol's necessity_set must be {list(want)} "
                             f"(fallback {fb})")
    out = E.v1._json_value(block)
    out["principles"] = {p: out["principles"][p] for p in declared}
    return out


__all__ = [
    "ANCHORS_SCHEMA",
    "ANCHOR_ALPHA",
    "ANCHOR_BLOCK_SIZE",
    "ANCHOR_INVALID",
    "ANCHOR_MIN_FINITE",
    "ANCHOR_RULE",
    "ANCHOR_STATUSES",
    "ANCHOR_VALID_NONSPECIFIC",
    "ANCHOR_VALID_SPECIFIC",
    "DECISIVE",
    "DECISIVE_THRESHOLD",
    "KINDS",
    "MECHANISM_ON_C_RATIO",
    "MECHANISM_ON_DOSE_RATIO",
    "MIN_RUNS",
    "NOT_TESTABLE_BY_DESIGN",
    "PRECISION_SCHEMA",
    "RATE_KEYS",
    "REFERENCE_SEEDS",
    "REPLACEMENT_CLAIMS",
    "SPECIFICITY_RATIO",
    "TestabilityError",
    "absent_precision",
    "anchor_entry",
    "anchor_specificity",
    "anchor_validity",
    "anchors_block",
    "combine_anchor_entries",
    "development_median_c",
    "fallback_level",
    "gate",
    "mechanism_on",
    "necessity_from_anchors",
    "pi0_analytic",
    "pi1_analytic",
    "precision_block",
    "present_precision",
    "status_rates",
    "testability_row",
    "validate_anchors_block",
    "validate_precision_block",
    "validate_precision_row",
]
