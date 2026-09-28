"""
MPC-Bench rule audit on **estimated** component statuses (spec v2, V2-6).

Input: bench records (``run_bench`` JSONL: per system the estimated
components with their null families and jackknife SEs, markers, the exact
IIM of declared TPMs, and the design labels). Nothing here reads oracle
channels except the stipulated design truth (``intended_bits``) used to score
the rules.

Pipeline (:func:`audit`)
------------------------
1. Construct scale (V2-2): ``c = (m - nu) / (rho - nu)`` with ``nu`` the
   system's own null mean and ``rho`` the reference anchor (mean estimate of
   the reference systems: design positive controls of the *training* fold),
   ``se_c`` from the jackknife SE of ``m``, the Monte-Carlo error of ``nu``
   and the SE of ``rho`` (:func:`impact_pipeline.bench.rules.construct_scale`).
2. Missingness scenarios: ``none``; ``RAM+SRPI missing``; ``SRPI missing``;
   ``random20`` (every component undefined independently with probability
   0.2, fixed seed). Missing components are NaN for every rule.
3. Label noise: training labels flipped with probability ``p`` (default 0,
   0.1, 0.2). Everything learned from labels uses the noisy training labels:
   reference anchors, DCM likelihood ratios, the logistic classifier and the
   single-marker thresholds (Youden's J). Evaluation always uses the clean
   stipulated truth. Folds: 2-fold cross-fitting by seed parity (a system is
   never scored by a model fitted on its own seed).
4. Rules: the IMPaCT v2 rule (``impact_c``: construct-scale statuses with
   genuine SEs, strong-Kleene AND), the installed evidence layer's verdict
   (``evidence_layer``, when importable; v1 or v2 semantics, names mapped to
   v2), every rival rule of :mod:`impact_pipeline.bench.rules` on the ``c``
   matrix (presence ``c >= 0.5``), and single-marker comparators (IIM-only,
   NAS-only, LZc, Gaussian Phi_R, exact-TPM IIM when declared).
5. Outputs: per rule, scenario, noise level and class the verdict
   distribution P(verdict | class); coverage (fraction determinate);
   selective risk (error among determinate decisions on systems with a
   stipulated truth) and selective accuracy (1 - selective risk);
   risk-coverage curves (determinate decisions ranked by the rule's
   confidence) and their area (AURC, the mean selective risk over the ranked
   prefixes up to the rule's own coverage). A rule without a confidence (the
   evidence layer's verdicts) gets its full-coverage point only and no AURC.

Truth: a system whose stipulated mechanism pattern has every principle of
the necessity set present is a positive (``MPC_CONSISTENT`` correct); one
with any principle absent is a negative (``EXCLUDED`` correct). Systems
without a stipulated pattern (whole-brain) and patchworks (whose truth
depends on the bearer and the single-source constraint) are reported in
P(verdict | class) only.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from impact_pipeline.bench import rules as R
from impact_pipeline.bench.compat import verdict_name

AUDIT_VERSION = "mpc-bench-audit/1.0.0"
PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")
SCENARIOS = ("none", "RAM+SRPI_missing", "SRPI_missing", "random20")
LABEL_NOISE_LEVELS = (0.0, 0.1, 0.2)
PRESENCE_THRESHOLD = 0.5
RANDOM_MISSING_P = 0.2
MARKERS = ("LZc", "PhiR_bits", "exact_IIM")
COVERAGE_GRID = tuple(np.round(np.linspace(0.1, 1.0, 10), 2))


# ---------------------------------------------------------------------------
# Records -> arrays
# ---------------------------------------------------------------------------


def class_of(rec: dict) -> str:
    """Coarse design class of a bench record."""
    design = rec.get("design")
    gen = str(rec.get("generator") or "")
    if design == "adversarial" or gen.startswith("adversarial_"):
        return f"adversarial:{rec.get('cell_id')}"
    if design == "whole_brain" or gen.startswith("whole_brain"):
        return f"whole_brain:{rec.get('cell_id')}:{gen}"
    if gen == "patchwork":
        lam = rec.get("sweep_level")
        lam = 0.0 if lam is None else float(lam)
        return f"patchwork:lambda={lam:g}:{rec.get('bearer_mode', 'system')}"
    if design == "witnesses" and rec.get("witness_id"):
        return f"witness:{rec['witness_id']}"
    bits = rec.get("intended_bits") or []
    if not bits:
        return f"other:{gen}"
    n_off = int(sum(1 for b in bits if int(b) == 0))
    if n_off == 0:
        return "all_present"
    if n_off == 1:
        return f"single_deficit:{PRINCIPLES[[int(b) for b in bits].index(0)]}"
    return f"multi_deficit:{n_off}"


def truth_of(rec: dict, necessity_set: Sequence[str] = PRINCIPLES) -> Optional[int]:
    """1 (MPC_CONSISTENT correct), 0 (EXCLUDED correct) or None (no truth)."""
    gen = str(rec.get("generator") or "")
    if gen == "patchwork" or gen.startswith("whole_brain"):
        return None
    bits = rec.get("intended_bits") or []
    if len(bits) != len(PRINCIPLES) or any(b is None for b in bits):
        return None
    idx = [PRINCIPLES.index(p) for p in necessity_set]
    return int(all(int(bits[i]) == 1 for i in idx))


def records_to_arrays(
    records: Iterable[dict], necessity_set: Sequence[str] = PRINCIPLES
) -> dict:
    """Stack ok-status records into component arrays (n x 5) and labels
    (truth relative to ``necessity_set``, see :func:`truth_of`)."""
    rows = [r for r in records if r.get("status", "ok") == "ok"]
    n = len(rows)
    shape = (n, len(PRINCIPLES))
    est = np.full(shape, np.nan)
    nu = np.full(shape, np.nan)
    sd0 = np.full(shape, np.nan)
    k = np.zeros(shape)
    se = np.full(shape, np.nan)
    markers = {m: np.full(n, np.nan) for m in MARKERS}
    for i, r in enumerate(rows):
        comps = r.get("components") or {}
        for j, p in enumerate(PRINCIPLES):
            c = comps.get(p)
            if not c or not c.get("defined"):
                continue
            est[i, j] = _f(c.get("estimate"))
            nu[i, j] = _f(c.get("null_mean"))
            sd0[i, j] = _f(c.get("null_sd"))
            k[i, j] = float(c.get("n_null") or 0)
            se[i, j] = _f(c.get("se"))
        mk = r.get("markers") or {}
        markers["LZc"][i] = _f(mk.get("LZc"))
        markers["PhiR_bits"][i] = _f(mk.get("PhiR_bits"))
        markers["exact_IIM"][i] = _f((r.get("exact_iim") or {}).get("value"))
    truth = [truth_of(r, necessity_set) for r in rows]
    return {
        "records": rows,
        "task_id": [r.get("task_id") for r in rows],
        "seed": np.asarray([int(r.get("seed", 0)) for r in rows]),
        "class": [class_of(r) for r in rows],
        "truth": np.asarray([np.nan if t is None else t for t in truth], dtype=float),
        "estimate": est,
        "null_mean": nu,
        "null_sd": sd0,
        "n_null": k,
        "se": se,
        "markers": markers,
    }


def _f(v) -> float:
    try:
        return float("nan") if v is None else float(v)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Scenarios, anchors, confidence
# ---------------------------------------------------------------------------


def missing_mask(scenario: str, shape, seed: int = 0) -> np.ndarray:
    """Boolean mask of components made undefined by a missingness scenario."""
    m = np.zeros(shape, dtype=bool)
    if scenario == "none":
        return m
    if scenario == "RAM+SRPI_missing":
        m[:, [0, 4]] = True
    elif scenario == "SRPI_missing":
        m[:, 4] = True
    elif scenario == "random20":
        m = np.random.default_rng(int(seed)).random(shape) < RANDOM_MISSING_P
    else:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    return m


def flip_labels(y: np.ndarray, p: float, seed: int = 0) -> np.ndarray:
    """Labels flipped independently with probability ``p`` (NaN kept)."""
    y = np.asarray(y, dtype=float).copy()
    if p <= 0:
        return y
    rng = np.random.default_rng(int(seed))
    flip = (rng.random(y.shape) < float(p)) & np.isfinite(y)
    y[flip] = 1.0 - y[flip]
    return y


def reference_anchors(estimate: np.ndarray, labels: np.ndarray) -> tuple:
    """Per-principle reference anchor (mean estimate of label-1 rows) and its
    SE (sd / sqrt(n)); NaN where no reference row is finite."""
    rows = np.asarray(labels) == 1
    rho = np.full(estimate.shape[1], np.nan)
    se_rho = np.full(estimate.shape[1], np.nan)
    for j in range(estimate.shape[1]):
        v = estimate[rows, j]
        v = v[np.isfinite(v)]
        if v.size:
            rho[j] = float(v.mean())
            se_rho[j] = float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0
    return rho, se_rho


def youden_threshold(values: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 on finite rows."""
    v = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float)
    ok = np.isfinite(v) & np.isfinite(y)
    v, y = v[ok], y[ok]
    if v.size < 2 or np.unique(y).size < 2:
        return float("nan")
    cand = np.unique(v)
    best, thr = -np.inf, float(cand[0])
    for t in cand:
        pred = v >= t
        sens = np.mean(pred[y == 1])
        spec = np.mean(~pred[y == 0])
        if sens + spec - 1 > best:
            best, thr = sens + spec - 1, float(t)
    return thr


def impact_c_confidence(
    c,
    se,
    necessity_idx,
    alpha=0.05,
    z_present=R.C_PRESENT_DEFAULT,
    delta_absent=R.C_ABSENT_DEFAULT,
) -> tuple:
    """
    Decision margins of the v2 rule: ``(consistent, excluded)`` with
    ``consistent = min_j (lower_j - z_present)`` (finite when every principle
    has a finite bound) and ``excluded = max_j (delta_absent - upper_j)`` over
    components credibly below ``delta_absent`` (NaN when none).
    """
    zc = norm.ppf(1.0 - alpha)
    c = np.asarray(c, dtype=float)[:, necessity_idx]
    s = np.asarray(se, dtype=float)[:, necessity_idx]
    with np.errstate(invalid="ignore"):
        lower = c - zc * s - z_present
        upper = delta_absent - (c + zc * s)
        cons = np.where(np.all(np.isfinite(lower), axis=1), lower.min(axis=1), np.nan)
        upper = np.where(upper > 0, upper, -np.inf)
        excl = upper.max(axis=1) if upper.shape[1] else np.full(c.shape[0], -np.inf)
    excl = np.where(np.isfinite(excl), excl, np.nan)
    return cons, excl


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------


def _decisions_from_output(out: R.RuleOutput, threshold: float):
    dec = np.asarray(out.decision, dtype=object)
    conf = np.abs(np.asarray(out.score, dtype=float) - float(threshold))
    return dec, conf


def _evidence_layer_verdicts(arrs, c_missing, rho, se_rho, necessity_set):
    """Verdicts of the installed evidence layer on the same evidence (missing
    components dropped: MISSING); None if the layer cannot be imported."""
    try:
        from impact_pipeline import evidence as ev
    except Exception:
        return None
    from impact_pipeline.bench import compat

    n = arrs["estimate"].shape[0]
    out = np.empty(n, dtype=object)
    for i in range(n):
        items = {}
        for j, p in enumerate(PRINCIPLES):
            m = arrs["estimate"][i, j]
            if c_missing[i, j] or not np.isfinite(m):
                continue
            se_m = arrs["se"][i, j]
            items[p] = [
                compat.make_component_evidence(
                    ev,
                    principle=p,
                    estimate=float(m),
                    null_mean=float(arrs["null_mean"][i, j]),
                    null_sd=float(arrs["null_sd"][i, j]),
                    se=float(se_m) if np.isfinite(se_m) else 0.0,
                    reference=None if not np.isfinite(rho[j]) else float(rho[j]),
                    reference_se=(float(se_rho[j]) if np.isfinite(se_rho[j]) else None),
                    n_null=int(arrs["n_null"][i, j]),
                    bearer_id="system",
                )
            ]
        try:
            v = compat.call_mpc_verdict(ev, items, necessity_set=necessity_set)
            out[i] = compat.verdict_name(getattr(v, "verdict", None))
        except Exception:  # pragma: no cover - depends on the other stream's API
            out[i] = R.UNDETERMINED
    return out


def apply_audit_rules(
    arrs: dict,
    fit_rows: np.ndarray,
    eval_rows: np.ndarray,
    train_labels: np.ndarray,
    c_missing: np.ndarray,
    necessity_set: Sequence[str] = PRINCIPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, tuple]:
    """
    Decisions and confidences of every rule for ``eval_rows``, with every
    label-dependent quantity fitted on ``fit_rows`` (labels ``train_labels``,
    aligned with all rows). Returns ``{rule: (decision, confidence)}``.
    """
    est = arrs["estimate"]
    y_fit = train_labels[fit_rows]
    rho, se_rho = reference_anchors(est[fit_rows], y_fit)
    c, se_c = R.construct_scale(
        est,
        arrs["null_mean"],
        rho[None, :],
        se_estimate=arrs["se"],
        null_sd=arrs["null_sd"],
        n_null=np.where(arrs["n_null"] > 0, arrs["n_null"], np.inf),
        se_reference=se_rho[None, :],
    )
    c = np.where(c_missing, np.nan, c)
    se_c = np.where(c_missing, np.nan, se_c)
    idx = [PRINCIPLES.index(p) for p in necessity_set]
    out = {}

    ce = c[eval_rows]
    imp = R.rule_impact_c(ce, se_c[eval_rows], necessity_set=necessity_set, alpha=alpha)
    cons, excl = impact_c_confidence(ce, se_c[eval_rows], idx, alpha)
    conf = np.where(
        imp.decision == R.MPC_CONSISTENT,
        cons,
        np.where(imp.decision == R.EXCLUDED, excl, np.nan),
    )
    out["impact_c"] = (np.asarray(imp.decision, dtype=object), conf)

    layer = _evidence_layer_verdicts(
        {
            k: (v[eval_rows] if isinstance(v, np.ndarray) and v.ndim == 2 else v)
            for k, v in arrs.items()
            if k in ("estimate", "null_mean", "null_sd", "n_null", "se")
        },
        c_missing[eval_rows],
        rho,
        se_rho,
        necessity_set,
    )
    if layer is not None:
        out["evidence_layer"] = (layer, np.full(layer.size, np.nan))

    thr = PRESENCE_THRESHOLD
    rival = [
        R.rule_union(ce, thr),
        *[R.rule_count_k(ce, k, thr) for k in range(1, 6)],
        R.rule_arithmetic_mean(ce, thr),
        R.rule_geometric_mean_uncapped(ce, thr),
        R.rule_geometric_mean_capped(ce, thr),
        R.rule_power_mean(ce, -1.0, thr),
        R.rule_power_mean(ce, 1.0, thr),
        R.rule_weakest_link(ce, thr),
        R.rule_chalmers_product(ce),
        R.rule_single_marker(ce[:, 3], thr, "IIM_only"),
        R.rule_single_marker(ce[:, 2], thr, "NAS_only"),
    ]
    rule_thr = {"union": 1.0, "chalmers_product": 0.5}
    for o in rival:
        t = rule_thr.get(
            o.rule, float(o.rule.split("_")[1]) if o.rule.startswith("count_") else thr
        )
        out[o.rule] = _decisions_from_output(o, t)

    # DCM-like naive Bayes with likelihood ratios calibrated on the fit rows.
    fit_known = np.isfinite(y_fit)
    if fit_known.sum() >= 4 and np.unique(y_fit[fit_known]).size == 2:
        sens, spec = R.calibrate_likelihood_ratios(
            R.presence(c[fit_rows][fit_known], thr), y_fit[fit_known]
        )
        o = R.rule_dcm_naive_bayes(ce, thr, sensitivity=sens, specificity=spec)
        out["dcm_naive_bayes"] = _decisions_from_output(o, 0.5)
        try:
            o = R.rule_logistic(
                c[fit_rows][fit_known],
                y_fit[fit_known].astype(int),
                C_test=ce,
                seed=seed,
            )
            out["logistic"] = _decisions_from_output(o, 0.5)
        except ValueError:
            pass
        for name, values in arrs["markers"].items():
            v_fit = values[fit_rows][fit_known]
            if np.isfinite(v_fit).sum() < 4:
                continue
            t = youden_threshold(v_fit, y_fit[fit_known])
            if np.isfinite(t):
                o = R.rule_single_marker(values[eval_rows], t, name)
                out[name] = _decisions_from_output(o, t)
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def risk_coverage(decision, confidence, truth) -> pd.DataFrame:
    """
    Selective risk against coverage: determinate decisions on rows with a
    truth, ranked by confidence (descending; NaN confidences tie last), the
    risk of the top-k for every k, coverage k / (rows with a truth).
    """
    dec = np.asarray(decision, dtype=object)
    y = np.asarray(truth, dtype=float)
    known = np.isfinite(y)
    det = known & np.isin(dec, (R.MPC_CONSISTENT, R.EXCLUDED))
    n_known = int(known.sum())
    if n_known == 0 or det.sum() == 0:
        return pd.DataFrame(columns=["coverage", "risk", "k"])
    pred = (dec[det] == R.MPC_CONSISTENT).astype(float)
    err = (pred != y[det]).astype(float)
    conf = np.asarray(confidence, dtype=float)[det]
    order = np.argsort(-np.where(np.isfinite(conf), conf, -np.inf), kind="mergesort")
    cum = np.cumsum(err[order])
    k = np.arange(1, det.sum() + 1)
    return pd.DataFrame({"coverage": k / n_known, "risk": cum / k, "k": k})


def summarise(decisions: pd.DataFrame) -> dict:
    """P(verdict | class), coverage, selective risk and AURC tables."""
    keys = ["rule", "scenario", "label_noise"]
    pvc = (
        decisions.groupby(keys + ["class", "decision"]).size().rename("n").reset_index()
    )
    tot = pvc.groupby(keys + ["class"])["n"].transform("sum")
    pvc["p"] = pvc["n"] / tot
    cov_rows, rc_rows = [], []
    for key, sub in decisions.groupby(keys):
        det = sub["decision"].isin((R.MPC_CONSISTENT, R.EXCLUDED))
        known = sub["truth"].notna()
        pred = (sub["decision"] == R.MPC_CONSISTENT).astype(float)
        err = det & known & (pred != sub["truth"])
        n_det_known = int((det & known).sum())
        pos = known & (sub["truth"] == 1)
        neg = known & (sub["truth"] == 0)
        cov_rows.append(
            dict(
                zip(keys, key),
                n=int(len(sub)),
                coverage=float(det.mean()),
                coverage_with_truth=float(det[known].mean()) if known.any() else np.nan,
                selective_risk=(
                    (float(err.sum()) / n_det_known) if n_det_known else np.nan
                ),
                selective_accuracy=(
                    1.0 - float(err.sum()) / n_det_known if n_det_known else np.nan
                ),
                p_excluded_given_positive=(
                    float((sub.loc[pos, "decision"] == R.EXCLUDED).mean())
                    if pos.any()
                    else np.nan
                ),
                p_consistent_given_negative=(
                    float((sub.loc[neg, "decision"] == R.MPC_CONSISTENT).mean())
                    if neg.any()
                    else np.nan
                ),
            )
        )
        rc = risk_coverage(sub["decision"], sub["confidence"], sub["truth"])
        # A rule without a confidence (the evidence layer's verdicts) cannot
        # rank its decisions: only its full determinate set is meaningful, so
        # it gets no partial-coverage risks and no AURC.
        ranked = bool(np.isfinite(sub.loc[det & known, "confidence"]).any())
        if not ranked:
            rc = rc.iloc[len(rc) - 1 :] if len(rc) else rc
        aurc = float(rc["risk"].mean()) if (len(rc) and ranked) else np.nan
        for cg in COVERAGE_GRID:
            hit = rc[rc["coverage"] <= cg + 1e-12]
            rc_rows.append(
                dict(
                    zip(keys, key),
                    coverage=float(cg),
                    risk=float(hit["risk"].iloc[-1]) if len(hit) else np.nan,
                    reached=bool(len(hit) and hit["coverage"].iloc[-1] >= cg - 1e-9),
                    aurc=aurc,
                    ranked=ranked,
                )
            )
    return {
        "p_verdict_given_class": pvc,
        "coverage": pd.DataFrame(cov_rows),
        "risk_coverage": pd.DataFrame(rc_rows),
    }


def audit(
    records: Iterable[dict],
    scenarios: Sequence[str] = SCENARIOS,
    label_noise: Sequence[float] = LABEL_NOISE_LEVELS,
    necessity_set: Sequence[str] = PRINCIPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Run the rule audit (module docstring) and return ``{'decisions',
    'p_verdict_given_class', 'coverage', 'risk_coverage', 'settings'}``.
    """
    arrs = records_to_arrays(records, necessity_set)
    n = len(arrs["task_id"])
    if n == 0:
        raise ValueError("no usable records")
    folds = (arrs["seed"] % 2).astype(int)
    rows_all = []
    in_sample = False
    for scen in scenarios:
        if scen not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        # Seeded by the scenario itself, so a scenario's mask does not depend
        # on which other scenarios are run or in which order.
        miss = missing_mask(
            scen, arrs["estimate"].shape, seed=seed + 101 * SCENARIOS.index(scen)
        )
        for p_noise in label_noise:
            noisy = flip_labels(
                arrs["truth"], p_noise, seed=seed + 7 + int(1000 * p_noise)
            )
            for fold in (0, 1):
                eval_rows = np.flatnonzero(folds == fold)
                fit_rows = np.flatnonzero(folds != fold)
                if eval_rows.size == 0:
                    continue
                if fit_rows.size == 0:  # a single seed: fit and score in-sample
                    fit_rows = eval_rows
                    in_sample = True
                res = apply_audit_rules(
                    arrs, fit_rows, eval_rows, noisy, miss, necessity_set, alpha, seed
                )
                for rule, (dec, conf) in res.items():
                    for k, i in enumerate(eval_rows):
                        rows_all.append(
                            {
                                "task_id": arrs["task_id"][i],
                                "seed": int(arrs["seed"][i]),
                                "class": arrs["class"][i],
                                "truth": arrs["truth"][i],
                                "rule": rule,
                                "scenario": scen,
                                "label_noise": float(p_noise),
                                "decision": verdict_name(dec[k]),
                                "confidence": float(conf[k]),
                                "fold": int(fold),
                            }
                        )
    decisions = pd.DataFrame(rows_all)
    with np.errstate(invalid="ignore"):
        se_ok = np.isfinite(arrs["se"]) & (arrs["se"] > 0)
    out = summarise(decisions)
    out["decisions"] = decisions
    out["settings"] = {
        "audit_version": AUDIT_VERSION,
        "scenarios": list(scenarios),
        "label_noise": [float(x) for x in label_noise],
        "necessity_set": list(necessity_set),
        "alpha": float(alpha),
        "presence_threshold": PRESENCE_THRESHOLD,
        "c_present": R.C_PRESENT_DEFAULT,
        "c_absent": R.C_ABSENT_DEFAULT,
        "random_missing_p": RANDOM_MISSING_P,
        "folds": "2-fold cross-fitting by seed parity",
        # True when every system has the same seed parity: anchors, likelihood
        # ratios, the classifier and marker thresholds were then fitted on the
        # scored systems themselves (not cross-fitted; not for reporting).
        "in_sample_fit": bool(in_sample),
        "n_systems": int(n),
        "n_with_truth": int(np.isfinite(arrs["truth"]).sum()),
        # Defined components with a usable sampling SE (records run without
        # --se-groups have none, and impact_c is then UNDETERMINED throughout).
        "fraction_defined_with_se": (
            float(np.mean(se_ok[np.isfinite(arrs["estimate"])]))
            if np.isfinite(arrs["estimate"]).any()
            else float("nan")
        ),
        "seed": int(seed),
    }
    return out
