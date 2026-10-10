#!/usr/bin/env python
"""
Operating characteristics of the MPC-Bench v2 decision rules.

Two uses:

``--reproduce-synth-oc``
    recomputes every number of the synthesis check of the v2 decision
    rules: (A) seeds needed for demonstrated bounds with 0 events, (B) TOST
    quantiles and reachability at ``alpha_A = 0.01``, (C) the restated HR1
    rule, (D) twin SE calibration, (E) the three-zone rule, (F) the RAM-PE
    PRESENT rule, (G) the any-event bound, (H) the H0 cell rule for rates
    near alpha. The Monte-Carlo sections draw from ``default_rng(20260930)``
    in a fixed order (C, then D, then H), so the printed lines equal the
    check's original log (``synth_oc.log``, see ``--compare-log``) line for
    line; the Clopper-Pearson decisions are looked up from precomputed
    thresholds instead of being recomputed per draw.

``--rates RATES.json``
    the operating characteristics of hypothesis parts from development
    rates (calibration decision CD-11): per part ``P(SUPPORTED | development
    rate)`` and ``P(FALSIFIED | correct estimator)``, both exact for the
    single-cell rules (three-zone, three-zone at most, reversed three-zone,
    the count rule, the lower-bound rule, demonstration; a part of several
    cells, given by ``n_cells``, is combined as the evaluator combines its
    cells) and by simulation for the H0 cell rule; and, for the single-cell
    rules, the smallest seed count per cell at which a part meets
    ``P(SUPPORTED) >= 0.8`` and ``P(FALSIFIED | correct) <= 0.05``. A part
    that fails is resized (more seeds), never re-thresholded; if no size up
    to ``--max-n`` works (or the rule is the simulated H0 cell rule, sized
    by hand from :func:`oc_h0_cell`), it is reported as
    INDETERMINATE-capable.

The rules are those of :mod:`impact_pipeline.v2.hypothesis_engine` (the
same Clopper-Pearson bounds). Arithmetic only: no bench data, no estimator,
no seed of the bench is touched (the simulation streams are the analysis's
own ``default_rng`` seeds).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from impact_pipeline.v2 import hypothesis_engine as HE  # noqa: E402

OC_VERSION = "mpc-bench-v2-operating-characteristics/1.0.0"
SYNTH_OC_SEED = 20260930
TARGET_SUPPORTED = 0.8
TARGET_FALSIFIED = 0.05


# --------------------------------------------------------------------------
# Clopper-Pearson thresholds (decisions without per-draw quantiles)
# --------------------------------------------------------------------------
def upper_ge_table(n: int, level: float, bound: float) -> np.ndarray:
    """``out[k] = cp_upper(k, n, level) >= bound`` for k = 0..n."""
    return np.array([HE.cp_upper(k, n, level) >= bound for k in range(n + 1)])


def lower_gt_table(n: int, level: float, bound: float) -> np.ndarray:
    """``out[k] = cp_lower(k, n, level) > bound`` for k = 0..n."""
    return np.array([HE.cp_lower(k, n, level) > bound for k in range(n + 1)])


# --------------------------------------------------------------------------
# exact operating characteristics of single-cell rules
# --------------------------------------------------------------------------
def _pmf(n, p):
    k = np.arange(n + 1)
    return k, stats.binom.pmf(k, n, p)


def oc_three_zone(n, p, x=0.8, alpha=0.05, m=1) -> dict:
    """``SUPPORTED`` iff k/n >= x; ``FALSIFIED`` iff CP upper (alpha/m) < x."""
    k, pk = _pmf(n, p)
    sup = pk[k >= math.ceil(x * n - 1e-9)].sum()
    fal = pk[~upper_ge_table(n, alpha / m, x)].sum()
    return {"supported": float(sup), "falsified": float(fal)}


def oc_three_zone_at_most(n, p, x, alpha=0.05, m=1) -> dict:
    k, pk = _pmf(n, p)
    sup = pk[k <= math.floor(x * n + 1e-9)].sum()
    fal = pk[lower_gt_table(n, alpha / m, x)].sum()
    return {"supported": float(sup), "falsified": float(fal)}


def oc_reversed_three_zone(n, p, x=0.8, alpha=0.05) -> dict:
    """``SUPPORTED`` iff CP upper (alpha) < x; ``FALSIFIED`` iff k/n >= x."""
    k, pk = _pmf(n, p)
    sup = pk[~upper_ge_table(n, alpha, x)].sum()
    fal = pk[k >= math.ceil(x * n - 1e-9)].sum()
    return {"supported": float(sup), "falsified": float(fal)}


def oc_count_at_most(n, p, max_rate=0.1, falsify_above=0.1, alpha=0.05) -> dict:
    k, pk = _pmf(n, p)
    sup = pk[k <= math.floor(max_rate * n + 1e-9)].sum()
    fal = pk[lower_gt_table(n, alpha, falsify_above)].sum()
    return {"supported": float(sup), "falsified": float(fal)}


def oc_rate_lower_bound(n, p, x=0.5, alpha=0.05) -> dict:
    """``SUPPORTED`` iff CP lower (alpha) > x; ``FALSIFIED`` iff CP upper
    (alpha) < x."""
    k, pk = _pmf(n, p)
    sup = pk[lower_gt_table(n, alpha, x)].sum()
    fal = pk[~upper_ge_table(n, alpha, x)].sum()
    return {"supported": float(sup), "falsified": float(fal)}


def oc_demonstration(n, p, bound, level=0.05) -> dict:
    """``SUPPORTED`` iff CP upper (level) < bound (else not demonstrated)."""
    k, pk = _pmf(n, p)
    sup = pk[~upper_ge_table(n, level, bound)].sum()
    return {"supported": float(sup), "falsified": float(1.0 - sup)}


def oc_h0_cell(
    n_cells: Sequence[int],
    rates: Sequence[float],
    *,
    bound=0.07,
    alpha=0.05,
    n_pool: Optional[int] = None,
    pool_rate: Optional[float] = None,
    sims=4000,
    seed=0,
) -> dict:
    """The H0 cell rule by simulation: cells ``n_cells`` with true rates
    ``rates`` (FALSIFIED if some cell's CP lower bound at alpha/m exceeds the
    bound); one pooled bound at ``alpha`` over ``n_pool`` clusters (default:
    the sum of the cells) at ``pool_rate`` (default: the cells' mean rate)."""
    rng = np.random.default_rng(seed)
    n_cells = [int(n) for n in n_cells]
    m = len(n_cells)
    tables = {n: lower_gt_table(n, alpha / m, bound) for n in set(n_cells)}
    n_pool = int(sum(n_cells) if n_pool is None else n_pool)
    pr = float(np.mean(rates) if pool_rate is None else pool_rate)
    pool_tab = upper_ge_table(n_pool, alpha, bound)
    sup = fal = 0
    for _ in range(int(sims)):
        ks = [rng.binomial(n, r) for n, r in zip(n_cells, rates)]
        f = any(tables[n][k] for n, k in zip(n_cells, ks))
        kp = rng.binomial(n_pool, pr)
        s = (not f) and not pool_tab[kp]
        fal += f
        sup += s
    return {"supported": sup / sims, "falsified": fal / sims, "sims": int(sims)}


def oc_kappa(
    df, kappa=1.0, *, point=(0.8, 1.25), inside=(0.67, 1.5), level=0.9
) -> dict:
    """Exact OC of the SE-calibration rule with ``kappa_hat = kappa sqrt(X /
    df)``, ``X ~ chi2(df)``: SUPPORTED iff the point lies in ``point`` and the
    ``level`` interval inside ``inside``; FALSIFIED iff the interval lies
    entirely outside ``point``."""
    a = (1 - level) / 2
    lo_f = math.sqrt(df / stats.chi2.ppf(1 - a, df))
    hi_f = math.sqrt(df / stats.chi2.ppf(a, df))

    def x_of(kh):  # X such that kappa_hat = kh
        return df * (kh / kappa) ** 2

    lo_k = max(point[0], inside[0] / lo_f)
    hi_k = min(point[1], inside[1] / hi_f)
    sup = (
        max(0.0, stats.chi2.cdf(x_of(hi_k), df) - stats.chi2.cdf(x_of(lo_k), df))
        if (hi_k > lo_k)
        else 0.0
    )
    fal = stats.chi2.cdf(x_of(point[0] / hi_f), df) + stats.chi2.sf(
        x_of(point[1] / lo_f), df
    )
    return {"supported": float(sup), "falsified": float(fal), "factor": [lo_f, hi_f]}


RULE_OC = {
    "three_zone": lambda n, p, prm: oc_three_zone(
        n, p, prm.get("x", 0.8), prm.get("alpha", 0.05), prm.get("m", 1)
    ),
    "three_zone_at_most": lambda n, p, prm: oc_three_zone_at_most(
        n, p, prm["x"], prm.get("alpha", 0.05), prm.get("m", 1)
    ),
    "reversed_three_zone": lambda n, p, prm: oc_reversed_three_zone(
        n, p, prm.get("x", 0.8), prm.get("alpha", 0.05)
    ),
    "count_at_most": lambda n, p, prm: oc_count_at_most(
        n,
        p,
        prm.get("max_rate", 0.1),
        prm.get("falsify_above", 0.1),
        prm.get("alpha", 0.05),
    ),
    "rate_lower_bound": lambda n, p, prm: oc_rate_lower_bound(
        n, p, prm.get("x", 0.5), prm.get("alpha", 0.05)
    ),
    "demonstration": lambda n, p, prm: oc_demonstration(
        n, p, prm["bound"], prm.get("level", 0.05)
    ),
}
# the rate of a correct estimator at which P(FALSIFIED) is checked
CORRECT_RATE = {
    "three_zone": lambda prm: prm.get("x", 0.8),
    "three_zone_at_most": lambda prm: prm["x"],
    "reversed_three_zone": lambda prm: 0.0,
    "count_at_most": lambda prm: prm.get("max_rate", 0.1),
    "rate_lower_bound": lambda prm: prm.get("x", 0.5),
    "demonstration": lambda prm: 0.0,
    "h0_cell": lambda prm: 0.05,
}


def part_oc(
    rule: str,
    params: Mapping,
    n: int,
    development_rate: float,
    correct_rate: Optional[float] = None,
    *,
    n_cells=None,
    sims=4000,
    seed=0,
) -> dict:
    """P(SUPPORTED | development rate) and P(FALSIFIED | correct estimator)
    of one part at ``n`` seeds per cell. With ``n_cells`` (the seeds of each
    cell) a part of several cells is combined as the evaluator combines it:
    SUPPORTED iff every cell is (product over cells), FALSIFIED iff some
    cell is (cells taken as independent), and a three-zone rule without a
    stated ``m`` uses ``m`` = the number of cells, as the evaluator does."""
    prm = dict(params or {})
    cr = CORRECT_RATE[rule](prm) if correct_rate is None else float(correct_rate)
    if rule == "h0_cell":
        cells = list(n_cells or [n])
        sup = oc_h0_cell(
            cells,
            [development_rate] * len(cells),
            bound=prm["bound"],
            alpha=prm.get("alpha", 0.05),
            sims=sims,
            seed=seed,
        )
        cor = oc_h0_cell(
            cells,
            [cr] * len(cells),
            bound=prm["bound"],
            alpha=prm.get("alpha", 0.05),
            sims=sims,
            seed=seed + 1,
        )
        return {
            "supported": sup["supported"],
            "falsified_if_correct": cor["falsified"],
            "correct_rate": cr,
        }
    if rule not in RULE_OC:
        raise ValueError(f"no operating characteristic for rule {rule!r}")
    cells = [int(n)] if not n_cells else [int(c) for c in n_cells]
    if rule in ("three_zone", "three_zone_at_most") and "m" not in prm:
        prm["m"] = len(cells)
    supported, not_falsified = 1.0, 1.0
    for nc in cells:
        supported *= RULE_OC[rule](nc, float(development_rate), prm)["supported"]
        not_falsified *= 1.0 - RULE_OC[rule](nc, cr, prm)["falsified"]
    return {
        "supported": supported,
        "falsified_if_correct": 1.0 - not_falsified,
        "correct_rate": cr,
    }


def required_n(
    rule: str,
    params: Mapping,
    development_rate: float,
    correct_rate: Optional[float] = None,
    *,
    n_min=10,
    n_max=400,
    target_supported=TARGET_SUPPORTED,
    target_falsified=TARGET_FALSIFIED,
    k_cells: Optional[int] = None,
    **kw,
) -> Optional[int]:
    """The smallest seed count per cell in ``[n_min, n_max]`` at which a part
    (of ``k_cells`` cells of that size, when given) meets both targets (None
    when none does: INDETERMINATE-capable)."""
    for n in range(int(n_min), int(n_max) + 1):
        if k_cells:
            kw["n_cells"] = [n] * int(k_cells)
        oc = part_oc(rule, params, n, development_rate, correct_rate, **kw)
        if (
            oc["supported"] >= target_supported
            and oc["falsified_if_correct"] <= target_falsified
        ):
            return n
    return None


def evaluate_rates(entries: Sequence[Mapping], *, n_max=400) -> List[dict]:
    """CD-11 table from development rates: entries ``{part, rule, params, n,
    development_rate[, correct_rate, n_cells]}``. Each row reports the OC
    at the planned n, whether both targets hold, and the resized n."""
    out = []
    for e in entries:
        rule, prm = e["rule"], dict(e.get("params") or {})
        oc = part_oc(
            rule,
            prm,
            int(e["n"]),
            float(e["development_rate"]),
            e.get("correct_rate"),
            n_cells=e.get("n_cells"),
        )
        ok = (
            oc["supported"] >= TARGET_SUPPORTED
            and oc["falsified_if_correct"] <= TARGET_FALSIFIED
        )
        resized = (
            None
            if ok
            else (
                required_n(
                    rule,
                    prm,
                    float(e["development_rate"]),
                    e.get("correct_rate"),
                    n_min=int(e["n"]),
                    n_max=n_max,
                    k_cells=len(e["n_cells"]) if e.get("n_cells") else None,
                )
                if rule != "h0_cell"
                else None
            )
        )
        out.append(
            {
                "part": e.get("part"),
                "rule": rule,
                "n": int(e["n"]),
                "development_rate": float(e["development_rate"]),
                **oc,
                "targets_met": ok,
                "resized_n": resized,
                "indeterminate_capable": (not ok and resized is None),
            }
        )
    return out


# --------------------------------------------------------------------------
# reproduction of the synthesis check
# --------------------------------------------------------------------------
def section_a() -> List[dict]:
    rows = []
    for bound, level, lab in [
        (0.07, 0.05, "m=1"),
        (0.07, 0.025, "m=2"),
        (0.07, 0.0125, "m=4"),
        (0.07, 0.05 / 6, "m=6"),
        (0.02, 0.05, "ABSENT admission 0.02"),
        (0.05, 0.05, "ABSENT admission 0.05"),
        (0.01, 0.05, "0.01"),
    ]:
        rows.append(
            {
                "bound": bound,
                "level": level,
                "label": lab,
                "n": HE.n_needed_zero_events(bound, level),
            }
        )
    return rows


def _pi0(se, q, delta=0.1):
    return max(0.0, 2 * stats.norm.cdf(delta / se - q) - 1)


def section_b() -> dict:
    qs = []
    for df in (9, 12, 19, 49, 69, 199):
        q_a = stats.t.ppf(1 - 0.01, df)
        q_a2 = stats.t.ppf(1 - 0.005, df)
        q_p = stats.t.ppf(0.95, df)
        qs.append(
            {
                "df": df,
                "q_P": q_p,
                "q_A": q_a,
                "q_A2": q_a2,
                "s_A": 0.1 / q_a,
                "s_A_nas": 0.1 / q_a2,
            }
        )
    pis = []
    for lab, se, df, split in [
        ("NAS A under R, per direction", 0.020, 9, True),
        ("NAS A under R, per direction", 0.025, 9, True),
        ("NAS C contiguous, per direction", 0.030, 9, True),
        ("NAS C interleaved, per direction", 0.023, 9, True),
        ("IIM-dir bootstrap", 0.050, 12, False),
        ("SRPI v1 (30 pairs)", 0.32, 9, False),
        ("SRPI v3 lesion 120 pairs", 0.021, 54, False),
        ("PDI jackknife (non-concordant)", 0.10, 9, False),
    ]:
        q = stats.t.ppf(1 - (0.005 if split else 0.01), df)
        pis.append({"label": lab, "se": se, "df": df, "pi0": _pi0(se, q)})
    return {"quantiles": qs, "pi0": pis}


SECTION_C_CONFIGS = ((900, 50, 16 * 5 * 2 + 20), (600, 50, 200), (1500, 100, 200))


def section_c(rng, sims=4000, P=5) -> List[dict]:
    rows = []
    for n_pool, n_cell, m_cells in SECTION_C_CONFIGS:
        pool_bad = upper_ge_table(n_pool, 0.05 / (2 * P), 0.08)
        cell_bad = lower_gt_table(n_cell, 0.05 / m_cells, 0.10)
        for true in (0.03, 0.05, 0.06):
            sup = fal = 0
            for _ in range(sims):
                ks = rng.binomial(n_pool, true, size=2 * P)
                ok = not pool_bad[ks].any()
                cells = rng.binomial(n_cell, true, size=m_cells)
                f = bool(cell_bad[cells].any())
                fal += f
                sup += ok and not f
            rows.append(
                {
                    "n_pool": n_pool,
                    "n_cell": n_cell,
                    "m": m_cells,
                    "true": true,
                    "supported": sup / sims,
                    "falsified": fal / sims,
                }
            )
    return rows


SECTION_D_CONFIGS = ((10, 7), (10, 8), (8, 6), (5, 8))


def section_d(rng) -> List[dict]:
    rows = []
    for n_net, n_tw in SECTION_D_CONFIGS:
        df = n_net * (n_tw - 1)
        lo_f = np.sqrt(df / stats.chi2.ppf(0.95, df))
        hi_f = np.sqrt(df / stats.chi2.ppf(0.05, df))
        x = rng.chisquare(df, 200000)
        kap = np.sqrt(x / df)
        sup = np.mean(
            (kap >= 0.8) & (kap <= 1.25) & (kap * lo_f >= 0.67) & (kap * hi_f <= 1.5)
        )
        fal = np.mean((kap * hi_f < 0.8) | (kap * lo_f > 1.25))
        k15, k06 = 1.5 * kap, 0.6 * kap
        fal15 = np.mean((k15 * hi_f < 0.8) | (k15 * lo_f > 1.25))
        fal06 = np.mean((k06 * hi_f < 0.8) | (k06 * lo_f > 1.25))
        rows.append(
            {
                "networks": n_net,
                "twins": n_tw,
                "df": df,
                "factor": [lo_f, hi_f],
                "supported": float(sup),
                "falsified": float(fal),
                "falsified_1.5": float(fal15),
                "falsified_0.6": float(fal06),
            }
        )
    return rows


def section_e() -> List[dict]:
    rows = []
    for n in (20, 40, 45):
        for true in (0.8, 0.85, 0.9, 0.95):
            a = oc_three_zone(n, true, 0.8, 0.05, 1)
            b = oc_three_zone(n, true, 0.8, 0.05, 4)
            rows.append(
                {
                    "n": n,
                    "true": true,
                    "supported": a["supported"],
                    "falsified_m1": a["falsified"],
                    "falsified_m4": b["falsified"],
                }
            )
    return rows


def section_f() -> List[dict]:
    rows = []
    for n in (20, 40):
        kmin = min(k for k in range(n + 1) if HE.cp_lower(k, n, 0.05) > 0.5)
        for true in (0.6, 0.7, 0.8):
            rows.append(
                {
                    "n": n,
                    "k_min": kmin,
                    "true": true,
                    "supported": float(1 - stats.binom.cdf(kmin - 1, n, true)),
                }
            )
    return rows


def section_g() -> List[dict]:
    return [{"clusters": n, "upper": HE.cp_upper(0, n, 0.05)} for n in (20, 40, 42, 45)]


SECTION_H_CONFIGS = ((200, 14, 1400), (200, 14, 2800), (100, 12, 1200))


def section_h(rng, sims=4000) -> List[dict]:
    rows = []
    for n_cell, m, n_pool_eff in SECTION_H_CONFIGS:
        cell_bad = lower_gt_table(n_cell, 0.05 / m, 0.07)
        pool_bad = upper_ge_table(n_pool_eff, 0.05, 0.07)
        sup = fal = 0
        for _ in range(sims):
            cells = rng.binomial(n_cell, 0.05, size=m)
            f = bool(cell_bad[cells].any())
            kp = rng.binomial(n_pool_eff, 0.05)
            s = (not f) and not pool_bad[kp]
            fal += f
            sup += s
        rows.append(
            {
                "n_cell": n_cell,
                "m": m,
                "n_pool": n_pool_eff,
                "supported": sup / sims,
                "falsified": fal / sims,
            }
        )
    return rows


def reproduce_synth_oc(seed: int = SYNTH_OC_SEED) -> dict:
    """Every section of the synthesis check, with its printed lines."""
    rng = np.random.default_rng(seed)
    out = {"A": section_a(), "B": section_b()}
    out["C"] = section_c(rng)
    out["D"] = section_d(rng)
    out["E"] = section_e()
    out["F"] = section_f()
    out["G"] = section_g()
    out["H"] = section_h(rng)
    out["lines"] = format_synth_oc(out)
    return out


def format_synth_oc(r: Mapping) -> List[str]:
    """The printed lines of the synthesis check from computed sections."""
    lines = ["== A. seeds needed for 'demonstrated' bounds with 0 events"]
    for a in r["A"]:
        lines.append(
            f"  bound {a['bound']} level {a['level']:.4f} ({a['label']}): "
            f"n >= {a['n']}"
        )
    lines += [
        "",
        "== B. TOST quantiles and reachability at alpha_A = 0.01 (n_A fixed at "
        "|N_decl| = 5)",
    ]
    for q in r["B"]["quantiles"]:
        lines.append(
            f"  df {q['df']:3d}: q_P {q['q_P']:.3f}  q_A {q['q_A']:.3f}  q_A/2 (NAS "
            f"union) {q['q_A2']:.3f}  s_A = delta/q_A {q['s_A']:.4f}  s_A(NAS) "
            f"{q['s_A_nas']:.4f}"
        )
    lines.append("  pi0 (TOST power at c = 0, normal approx):")
    for p in r["B"]["pi0"]:
        lines.append(
            f"    {p['label']:36s} se_c {p['se']:.3f} df {p['df']:3d}: pi0 "
            f"{p['pi0']:.3f}"
        )
    lines += ["", "== C. HR1 (interval calibration at the null), restated rule"]
    for c in r["C"]:
        lines.append(
            f"  n_pool {c['n_pool']} n_cell {c['n_cell']} m {c['m']} true tail "
            f"{c['true']}: P(SUPPORTED) {c['supported']:.3f}  P(FALSIFIED) "
            f"{c['falsified']:.3f}"
        )
    lines += [
        "",
        "== D. Twin SE calibration: pooled within-network df = N_net*(n_twin-1)",
    ]
    for d in r["D"]:
        lines.append(
            f"  {d['networks']} networks x {d['twins']} twins (df {d['df']}): 90% "
            f"factor [{d['factor'][0]:.3f}, {d['factor'][1]:.3f}]; "
            f"P(SUPPORTED|kappa=1) {d['supported']:.3f}  P(FALSIFIED|1) "
            f"{d['falsified']:.4f}  P(FALSIFIED|1.5) {d['falsified_1.5']:.3f}  "
            f"P(FALSIFIED|0.6) {d['falsified_0.6']:.3f}"
        )
    lines += [
        "",
        "== E. Three-zone '>= 80 %' rule: SUPPORTED iff point >= 0.8; FALSIFIED "
        "iff CP upper(0.05/m) < 0.8",
    ]
    for e in r["E"]:
        lines.append(
            f"  n {e['n']} true {e['true']}: P(SUPPORTED) {e['supported']:.3f}  "
            f"P(FALSIFIED, m=1) {e['falsified_m1']:.3f}  (m=4) "
            f"{e['falsified_m4']:.3f}"
        )
    lines += ["", "== F. RAM-PE PRESENT on PC: SUPPORTED iff CP lower (0.05) > 0.5"]
    for f in r["F"]:
        lines.append(
            f"  n {f['n']}: needs >= {f['k_min']}/{f['n']}; true {f['true']}: "
            f"P(SUPPORTED) {f['supported']:.3f}"
        )
    lines += [
        "",
        "== G. Verdict specificity HC5v2: CP upper (0.05) at 0 events by seed "
        "clusters",
    ]
    for g in r["G"]:
        lines.append(f"  {g['clusters']} clusters: {g['upper']:.4f}")
    lines += [
        "",
        "== H. H0 cell rule for rates near alpha (rank exceedance, H-IIM-1 style)",
    ]
    for h in r["H"]:
        lines.append(
            f"  n_cell {h['n_cell']} m {h['m']} pooled clusters {h['n_pool']}: "
            f"P(SUPPORTED|0.05) {h['supported']:.3f} P(FALSIFIED|0.05) "
            f"{h['falsified']:.3f}"
        )
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--reproduce-synth-oc",
        action="store_true",
        help="print the lines of the synthesis check (synth_oc.log)",
    )
    ap.add_argument(
        "--compare-log",
        default=None,
        help="with --reproduce-synth-oc: compare with this log; exit 1 on a "
        "difference",
    )
    ap.add_argument(
        "--rates",
        default=None,
        help="JSON list of {part, rule, params, n, development_rate"
        "[, correct_rate, n_cells]} (CD-11)",
    )
    ap.add_argument("--out", default=None, help="write the JSON result here")
    ap.add_argument("--max-n", type=int, default=400)
    args = ap.parse_args(argv)
    result: Dict[str, object] = {"version": OC_VERSION}
    rc = 0
    if args.reproduce_synth_oc:
        rep = reproduce_synth_oc()
        print("\n".join(rep["lines"]))
        result["synth_oc"] = {k: v for k, v in rep.items() if k != "lines"}
        if args.compare_log:
            want = (
                Path(args.compare_log)
                .read_text(encoding="utf-8")
                .rstrip("\n")
                .split("\n")
            )
            diff = [
                (i, a, b) for i, (a, b) in enumerate(zip(rep["lines"], want)) if a != b
            ]
            if diff or len(want) != len(rep["lines"]):
                print(f"differs from {args.compare_log}: {diff[:5]}", file=sys.stderr)
                rc = 1
            else:
                print(f"identical to {args.compare_log}", file=sys.stderr)
    if args.rates:
        entries = json.loads(Path(args.rates).read_text(encoding="utf-8"))
        rows = evaluate_rates(entries, n_max=args.max_n)
        result["parts"] = rows
        for r in rows:
            print(json.dumps(r))
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8"
        )
    if not (args.reproduce_synth_oc or args.rates):
        ap.print_help()
    return rc


if __name__ == "__main__":
    sys.exit(main())
