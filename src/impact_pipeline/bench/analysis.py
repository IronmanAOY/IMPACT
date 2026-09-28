"""
MPC-Bench analysis: construct-scale assessment of bench records under a
protocol, candidate-cutoff grids, dose-response on the construct scale and
the reference anchors per family.

Used by ``scripts/calibrate_bench.py`` (development calibration: seeds
0-999, families A/B) and ``scripts/bench_hypotheses.py`` (the preregistered
confirmatory hypotheses HC1-HC10). Statuses and verdicts at a protocol's own
cutoffs always come from the evidence layer
(:func:`impact_pipeline.bench.export.evidence_verdict`, i.e.
``evidence.mpc_verdict``); :func:`status_from_bounds` re-applies the same
status rule to the recorded ``c``, ``se_c`` and degrees of freedom for other
candidate cutoffs ``(z, delta, alpha)`` (development calibration only).

Reference anchors: the protocol's external reference is the family-A
positive control on development reference seeds. A family whose systems
live on another substrate (family C) is anchored on its own nominal positive
control (``PC_nominal``) on a disjoint block of reference seeds, with the same
rule (:func:`family_reference`); the protocol is then the frozen protocol
with that reference (:func:`protocol_for_family`), everything else equal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from impact_pipeline.bench.generators import PRINCIPLES  # noqa: F401 (re-exported)

ANALYSIS_VERSION = "mpc-bench-analysis/1.0.0"
RESULTS_JSONL = "results.jsonl"
STATUS_PRESENT = "PRESENT"
STATUS_ABSENT = "ABSENT"
STATUS_UNDEFINED = "UNDEFINED"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def read_records(paths) -> List[dict]:
    """Latest record per task id from every ``results.jsonl`` below the given
    directories (or JSONL files); only records with ``status == 'ok'``."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    latest = {}
    for base in paths:
        base = Path(base)
        files = [base] if base.is_file() else sorted(base.rglob(RESULTS_JSONL))
        for path in files:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        latest[rec["task_id"]] = rec
    return [latest[k] for k in sorted(latest) if latest[k].get("status") == "ok"]


def check_split(records: Iterable[dict], allowed: Sequence[str]) -> None:
    """Refuse records outside the allowed splits (``development`` /
    ``confirmatory``): development calibration must never read held-out
    data."""
    bad = sorted(
        {r.get("task_id") for r in records if r.get("split") not in set(allowed)}
    )
    if bad:
        raise ValueError(
            f"records outside the allowed splits {tuple(allowed)}: {bad[:5]}"
            f"{' ...' if len(bad) > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# Reference anchors
# ---------------------------------------------------------------------------


def family_reference(
    records: Iterable[dict], alpha: float = 0.05, allow_confirmatory: bool = False
) -> tuple:
    """
    External reference of the positive-control records of one family
    (:func:`impact_pipeline.bench.reference.reference_from_records`, with its
    anchor-validity rule at ``alpha``). Returns ``(reference, summary)``.
    """
    from impact_pipeline.bench.reference import reference_from_records

    return reference_from_records(
        list(records), alpha=alpha, allow_confirmatory=allow_confirmatory
    )


def protocol_for_family(protocol, reference: Optional[dict] = None):
    """The frozen protocol, or the frozen protocol with ``reference`` (a
    family's own positive-control anchor) and everything else unchanged."""
    from impact_pipeline import evidence as ev

    proto = ev.resolve_protocol(protocol)
    if reference is None:
        return proto
    if proto.replace(reference=reference) == proto:
        return proto  # the protocol's own reference: nothing changes
    name = f"{proto.name or 'protocol'} [family reference]"
    return proto.replace(reference=reference, name=name)


def same_except_reference(a, b) -> bool:
    """True when two protocols differ at most in ``reference`` and ``name``."""
    from impact_pipeline import evidence as ev

    da = dict(ev.resolve_protocol(a).to_dict())
    db = dict(ev.resolve_protocol(b).to_dict())
    for d in (da, db):
        d.pop("reference", None)
        d.pop("name", None)
    return da == db


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

_DESIGN_KEYS = (
    "task_id",
    "design",
    "family",
    "generator",
    "cell_id",
    "witness_id",
    "sweep_knob",
    "sweep_level",
    "bearer_mode",
    "seed",
    "split",
)


def _f(v) -> float:
    try:
        return float("nan") if v is None else float(v)
    except (TypeError, ValueError):
        return float("nan")


def principle_reasons(reasons: Sequence[str]) -> Dict[str, str]:
    """Per-principle reason code (first one) from verdict reason codes."""
    from impact_pipeline import evidence as ev

    out = {}
    for code in reasons or []:
        kind, principle, _ = ev.parse_reason(code)
        if principle is not None and principle not in out:
            out[principle] = code
    return out


def assess(records: Iterable[dict], protocols) -> tuple:
    """
    Evidence-layer assessment of bench records.

    ``protocols``: one protocol (``evidence.Protocol``, dict or path) for
    every record, or a mapping ``family -> protocol`` (key ``'*'`` = default).
    Returns ``(components, verdicts)``: one row per (record, principle) with
    the design labels, the estimate, null moments, jackknife SE, the
    construct-scale ``c``, ``se_c``, ``df``, one-sided bounds, the status and
    its reason code; and one row per record with the verdict, its reason
    codes, the necessity set and the protocol hash.
    """
    from impact_pipeline import evidence as ev
    from impact_pipeline.bench.export import evidence_verdict

    if isinstance(protocols, Mapping) and not _looks_like_protocol(protocols):
        table = {k: ev.resolve_protocol(v) for k, v in protocols.items()}
    else:
        table = {"*": ev.resolve_protocol(protocols)}
    comp_rows, verdict_rows = [], []
    for rec in records:
        proto = table.get(str(rec.get("family")), table.get("*"))
        if proto is None:
            raise ValueError(f"no protocol for family {rec.get('family')!r}")
        meta = {"substrate": (rec.get("system") or {}).get("substrate")}
        res = {
            "components": rec.get("components") or {},
            "estimator_modes": rec.get("estimator_modes") or {},
        }
        v = evidence_verdict(res, meta, protocol=proto)
        base = {k: rec.get(k) for k in _DESIGN_KEYS}
        bits = rec.get("intended_bits") or []
        base["intended_bits"] = "".join(str(int(b)) for b in bits) if bits else None
        for k, val in (rec.get("knobs") or {}).items():
            base[f"knob_{k}"] = val
        why = principle_reasons(v.get("reasons") or [])
        for p, c in res["components"].items():
            status = (v.get("component_status") or {}).get(p)
            comp_rows.append(
                {
                    **base,
                    "principle": p,
                    "estimate": _f(c.get("estimate")),
                    "null_mean": _f(c.get("null_mean")),
                    "null_sd": _f(c.get("null_sd")),
                    "n_null": int(c.get("n_null") or 0),
                    "null_family": c.get("null_family"),
                    "se": _f(c.get("se")),
                    "se_n": int(c.get("se_n") or 0),
                    "defined": bool(c.get("defined")),
                    "estimator_reason": c.get("reason"),
                    "c": _f((v.get("c") or {}).get(p)),
                    "se_c": _f((v.get("c_se") or {}).get(p)),
                    "df": _f((v.get("c_df") or {}).get(p)),
                    "c_lower": _f((v.get("c_lower") or {}).get(p)),
                    "c_upper": _f((v.get("c_upper") or {}).get(p)),
                    "status": status,
                    "status_reason": why.get(p),
                    "in_necessity_set": p in (v.get("necessity_set") or []),
                }
            )
        verdict_rows.append(
            {
                **base,
                "verdict": v.get("verdict"),
                "reasons": ";".join(v.get("reasons") or []),
                "necessity_set": ",".join(v.get("necessity_set") or []),
                "protocol_hash": v.get("protocol_hash"),
            }
        )
    return pd.DataFrame(comp_rows), pd.DataFrame(verdict_rows)


def _looks_like_protocol(obj) -> bool:
    return isinstance(obj, Mapping) and (
        "necessity_set" in obj or "schema" in obj or "cutoffs" in obj
    )


def components_raw(records: Iterable[dict]) -> pd.DataFrame:
    """One row per (record, principle) with the design labels and the raw
    component fields (estimate, null moments, jackknife SE and replicates)."""
    rows = []
    for rec in records:
        base = {k: rec.get(k) for k in _DESIGN_KEYS}
        bits = rec.get("intended_bits") or []
        base["intended_bits"] = "".join(str(int(b)) for b in bits) if bits else None
        modes = rec.get("estimator_modes") or {}
        for p, c in (rec.get("components") or {}).items():
            rows.append(
                {
                    **base,
                    "principle": p,
                    "estimate": _f(c.get("estimate")),
                    "null_mean": _f(c.get("null_mean")),
                    "null_sd": _f(c.get("null_sd")),
                    "n_null": int(c.get("n_null") or 0),
                    "null_family": c.get("null_family"),
                    "se": _f(c.get("se")),
                    "se_n": int(c.get("se_n") or 0),
                    "se_replicates": c.get("se_replicates"),
                    "defined": bool(c.get("defined")),
                    "estimator_reason": c.get("reason"),
                    "bearer_id": c.get("bearer_id"),
                    "cut_mode": (
                        (modes.get("IIM") or {}).get("cut_mode") if p == "IIM" else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def assess_components(frame: pd.DataFrame, reference: dict, protocol) -> pd.DataFrame:
    """
    Construct-scale assessment of component rows given as raw columns
    (``principle``, ``estimate``, ``null_mean``, ``null_sd``, ``n_null``,
    ``se``, ``se_n``, ``defined``) against an external ``reference``, with the
    protocol's cutoffs and alpha (``evidence.component_assessment``; jackknife
    SEs with ``se_n - 1`` degrees of freedom, as the bench verdicts). Adds
    ``c``, ``se_c``, ``se_sampling``, ``se_null``, ``se_reference``, ``df``,
    ``c_lower``, ``c_upper``, ``status`` and ``status_reason``.
    """
    from impact_pipeline import evidence as ev

    proto = ev.resolve_protocol(protocol)
    rows = []
    vals, ses = (reference or {}).get("values", {}), (reference or {}).get("se", {})
    for r in frame.itertuples(index=False):
        p = r.principle
        anchor = {}
        if p in vals:
            anchor = {
                "reference": vals[p],
                "reference_se": ses.get(p),
                "reference_scale": (reference or {}).get("scale", "excess"),
            }
        se_n = int(getattr(r, "se_n", 0) or 0)
        se = _f(r.se)
        item = ev.ComponentEvidence(
            principle=p,
            estimate=_f(r.estimate),
            null_mean=_f(r.null_mean),
            null_sd=_f(r.null_sd),
            n_null=int(r.n_null or 0),
            se=0.0 if not math.isfinite(se) else se,
            se_df=float(se_n - 1) if se_n >= 2 else None,
            defined=bool(r.defined),
            **anchor,
        )
        a = ev.component_assessment(item, cutoff=proto.cutoff_for(p), alpha=proto.alpha)
        rows.append(
            {
                "c": a.c,
                "se_c": a.se,
                "se_sampling": a.se_sampling,
                "se_null": a.se_null,
                "se_reference": a.se_reference,
                "df": a.df,
                "c_lower": a.lower,
                "c_upper": a.upper,
                "status": a.status.value,
                "status_reason": a.reason,
            }
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ---------------------------------------------------------------------------
# Candidate cutoffs
# ---------------------------------------------------------------------------


def one_sided_quantile(alpha: float, df) -> np.ndarray:
    """Student-t ``1 - alpha`` quantile per degrees of freedom (normal for
    infinite / non-finite df), as ``evidence.component_assessment``."""
    from scipy.stats import norm
    from scipy.stats import t as t_dist

    df = np.asarray(df, dtype=float)
    q = np.full(df.shape, float(norm.ppf(1.0 - float(alpha))))
    fin = np.isfinite(df) & (df > 0)
    q[fin] = t_dist.ppf(1.0 - float(alpha), df[fin])
    return q


def status_from_bounds(c, se_c, df, z: float, delta: float, alpha: float) -> np.ndarray:
    """
    The evidence layer's status rule on recorded ``c``, ``se_c`` and ``df``
    for candidate cutoffs: PRESENT if ``c - q se_c > z``, ABSENT if
    ``c + q se_c < delta``, UNDEFINED otherwise (and whenever ``c`` or
    ``se_c`` is not finite). ``q`` is the one-sided Student-t quantile.
    """
    if float(delta) > float(z):
        raise ValueError("delta must be <= z")
    c = np.asarray(c, dtype=float)
    se = np.asarray(se_c, dtype=float)
    q = one_sided_quantile(alpha, df)
    ok = np.isfinite(c) & np.isfinite(se) & (se >= 0)
    lower = c - q * se
    upper = c + q * se
    out = np.full(c.shape, STATUS_UNDEFINED, dtype=object)
    out[ok & (lower > float(z))] = STATUS_PRESENT
    out[ok & (upper < float(delta))] = STATUS_ABSENT
    return out


def cutoff_grid(
    comp: pd.DataFrame,
    group_cols: Sequence[str],
    z_values: Sequence[float],
    delta_values: Sequence[float],
    alphas: Sequence[float],
) -> pd.DataFrame:
    """
    PRESENT / ABSENT / UNDEFINED rates per group for every candidate
    ``(z, delta, alpha)`` with ``delta <= z``; rows whose status is UNDEFINED
    for another reason (no anchor, no SE) count as UNDEFINED.
    """
    rows = []
    for alpha in alphas:
        for z in z_values:
            for d in delta_values:
                if d > z:
                    continue
                st = status_from_bounds(
                    comp["c"], comp["se_c"], comp["df"], z, d, alpha
                )
                tmp = comp.assign(_st=st)
                for key, sub in tmp.groupby(list(group_cols), dropna=False):
                    key = key if isinstance(key, tuple) else (key,)
                    n = int(len(sub))
                    rows.append(
                        {
                            **dict(zip(group_cols, key)),
                            "alpha": float(alpha),
                            "z": float(z),
                            "delta": float(d),
                            "n": n,
                            "present_rate": float(
                                (sub["_st"] == STATUS_PRESENT).mean()
                            ),
                            "absent_rate": float((sub["_st"] == STATUS_ABSENT).mean()),
                            "undefined_rate": float(
                                (sub["_st"] == STATUS_UNDEFINED).mean()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dose-response on the construct scale
# ---------------------------------------------------------------------------


def dose_response(comp: pd.DataFrame) -> tuple:
    """
    Sweep records: per (knob, level, principle) the mean, SD and n of ``c``
    and the status rates at the protocol cutoffs; per (knob, principle) the
    OLS slope of ``c`` on the dose rescaled to [0, 1], Spearman's rho of
    ``c`` with the dose over all runs (and its p value), and ``c`` at the
    lowest (off) and at the nominal level. Returns ``(levels, slopes)``.
    """
    from scipy.stats import spearmanr

    from impact_pipeline.bench.generators import NOMINAL_KNOBS

    sw = comp[comp["design"] == "sweep"].copy()
    if sw.empty:
        return pd.DataFrame(), pd.DataFrame()
    sw["sweep_level"] = sw["sweep_level"].astype(float)
    lev = (
        sw.groupby(["sweep_knob", "sweep_level", "principle"])
        .agg(
            n=("c", "size"),
            n_c=("c", lambda s: int(np.isfinite(s).sum())),
            c_mean=("c", "mean"),
            c_sd=("c", "std"),
            se_c_median=("se_c", "median"),
            present_rate=("status", lambda s: float((s == STATUS_PRESENT).mean())),
            absent_rate=("status", lambda s: float((s == STATUS_ABSENT).mean())),
        )
        .reset_index()
    )
    nominal = NOMINAL_KNOBS.to_dict()
    rows = []
    for (knob, p), sub in sw.groupby(["sweep_knob", "principle"]):
        x = sub["sweep_level"].to_numpy(dtype=float)
        y = sub["c"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        span = float(np.ptp(x)) if x.size else 0.0
        slope = rho = pval = float("nan")
        if ok.sum() >= 3 and span > 0 and np.ptp(x[ok]) > 0:
            xs = (x[ok] - x.min()) / span
            slope = float(np.polyfit(xs, y[ok], 1)[0])
            r = spearmanr(x[ok], y[ok])
            rho, pval = float(r.correlation), float(r.pvalue)
        levels = sorted(set(x.tolist()))
        off = sub[sub["sweep_level"] == levels[0]]["c"]
        # the sweep grid need not contain the nominal dose: the grid level
        # closest to it stands in for it
        nom_dose = float(nominal.get(knob, np.nan))
        nom_level = (
            min(levels, key=lambda v: abs(v - nom_dose))
            if math.isfinite(nom_dose) and levels
            else float("nan")
        )
        nom = sub[sub["sweep_level"] == nom_level]["c"]
        rows.append(
            {
                "knob": knob,
                "principle": p,
                "n": int(ok.sum()),
                "slope_c_per_unit_dose": slope,
                "spearman_rho": rho,
                "spearman_p": pval,
                "c_off_mean": float(off.mean()) if len(off) else float("nan"),
                "c_nominal_mean": float(nom.mean()) if len(nom) else float("nan"),
                "nominal_dose": nom_dose,
                "nominal_level": nom_level,
            }
        )
    return lev, pd.DataFrame(rows)


# Switch -> target principle (the knob each principle's mechanism uses).
KNOB_TARGET = {"eta": "RAM", "K": "PDI", "g_b": "NAS", "c_int": "IIM", "e": "SRPI"}
# Switch -> the single-deficit witness that realises its off dose (all other
# mechanisms nominal).
SWITCH_WITNESS = {
    "eta": "W_RAM_no_plasticity",
    "K": "W_PDI_single_attractor",
    "g_b": "W_NAS_no_workspace",
    "c_int": "W_IIM_feedforward",
    "e": "W_SRPI_no_efference",
}
# Declared structural dependencies of the generator design (witnesses.yaml
# notes): removing every backward loop (c_int = 0) also removes part of the
# workspace receive-and-return, and removing the workspace (g_b = 0) removes
# the recurrent loops through the hub. Their off-target effects are
# reported, not counted as cross-talk.
DECLARED_DEPENDENCIES = frozenset({("c_int", "NAS"), ("g_b", "IIM")})


def witness_contrasts(comp: pd.DataFrame) -> pd.DataFrame:
    """
    Paired witness contrasts: per (family, switch, principle) the median,
    over seeds, of ``c(PC_nominal) - c(witness of the switch)`` (same seed =
    same network and noise; only the switched mechanism differs), its number
    of seeds, and whether the pair is the principle's own switch or a
    declared structural dependency.
    """
    w = comp[(comp["design"] == "witnesses") & (comp["bearer_mode"] == "system")]
    rows = []
    for fam, wf in w.groupby("family"):
        pc = wf[wf["witness_id"] == "PC_nominal"].set_index(["seed", "principle"])["c"]
        for switch, wid in SWITCH_WITNESS.items():
            ww = wf[wf["witness_id"] == wid].set_index(["seed", "principle"])["c"]
            d = (pc - ww).dropna()
            for p, sub in d.groupby(level="principle"):
                rows.append(
                    {
                        "family": fam,
                        "switch": switch,
                        "principle": p,
                        "own": KNOB_TARGET[switch] == p,
                        "declared_dependency": (switch, p) in DECLARED_DEPENDENCIES,
                        "n": int(len(sub)),
                        "median_delta_c": float(sub.median()),
                        "fraction_ge_0.5": float((sub >= 0.5).mean()),
                    }
                )
    return pd.DataFrame(rows)


def se_calibration(comp: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Jackknife SE of the estimate against the between-seed SD of the
    estimate for groups of identically configured systems (``by``)."""
    rows = []
    for key, sub in comp.groupby(list(by) + ["principle"], dropna=False):
        est = sub["estimate"].to_numpy(dtype=float)
        se = sub["se"].to_numpy(dtype=float)
        fe, fs = np.isfinite(est), np.isfinite(se)
        key = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(list(by) + ["principle"], key)),
                "n": int(len(sub)),
                "between_seed_sd": (
                    float(np.std(est[fe], ddof=1)) if fe.sum() > 1 else np.nan
                ),
                "jackknife_se_mean": float(np.mean(se[fs])) if fs.any() else np.nan,
                "jackknife_se_median": float(np.median(se[fs])) if fs.any() else np.nan,
                "n_se_zero": int(np.sum(se[fs] == 0)),
                "null_mc_se_mean": float(
                    np.nanmean(
                        sub["null_sd"] / np.sqrt(sub["n_null"].where(sub["n_null"] > 0))
                    )
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["se_ratio"] = out["jackknife_se_mean"] / out["between_seed_sd"]
    return out


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple:
    """Two-sided Wilson interval of a proportion."""
    from scipy.stats import norm

    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))
