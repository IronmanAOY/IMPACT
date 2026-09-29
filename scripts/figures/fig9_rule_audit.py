#!/usr/bin/env python
"""
Figure 9: audit of decision rules on estimated component statuses (HC9).

Inputs (from the rule audit, ``scripts/benchmark_attribution_rules.py``):

- ``--summary``: one row per rule and scenario with ``rule``, ``scenario``,
  ``coverage`` (fraction determinate) and ``selective_accuracy``; optional
  ``audit``, ``label_noise``, ``selective_risk``,
  ``p_excluded_given_positive``, ``p_consistent_given_negative``;
- ``--risk-coverage`` (``rule``, ``coverage``, ``risk``, optional
  ``scenario``, ``audit``, ``label_noise``, ``reached``) or ``--cases`` (per
  case ``rule``, ``confidence``, ``correct``, optional ``scenario``), from
  which risk-coverage curves are computed with
  ``impact_pipeline.necessity.risk_coverage``;
- ``--decisions`` (optional; ``audit_decisions.csv``: ``rule``, ``scenario``,
  ``label_noise``, ``class``, ``decision``): the HC9 statistics, computed with
  the classes of the preregistered evaluation (positive controls
  ``all_present`` and ``witness:PC_nominal``; single-deficit classes whose
  target is in the necessity set).

``--audit`` selects one audit (default ``primary_Nanch`` when present),
``--noise`` the label-noise level (default 0), ``--scenario`` the scenario of
panel A (default ``none`` when present, else the first).

Panels: A risk-coverage curves of the highlighted rules (the IMPaCT rule
``impact_c`` first) and every rule's operating point (coverage, selective
risk); B coverage, selective risk, P(EXCLUDED | positive) and
P(MPC_CONSISTENT | negative) per rule, filled for ``--scenario`` and open for
the missingness scenario (``RAM+SRPI_missing`` when present); C the HC9
quantities: the largest MPC_CONSISTENT rate over the single-deficit classes
(scenario none) and the MPC_CONSISTENT rate on positive controls, with the
preregistered bounds alpha = 0.05 (b) and alpha + 0.02 (c).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import figure_common as fc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

STEM = "fig9_rule_audit"
IMPACT_RULES = ("impact_c", "impact")
HIGHLIGHT = ("impact_c", "impact", "logistic", "NAS_only", "dcm_naive_bayes")
HIGHLIGHT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
HIGHLIGHT_MARKERS = ("o", "s", "^", "D")
DUPLICATES = {"evidence_layer": "impact_c"}
MISSING_SCENARIO = "RAM+SRPI_missing"
WITNESS_TARGET = {
    "W_RAM_no_plasticity": "RAM",
    "W_PDI_single_attractor": "PDI",
    "W_NAS_no_workspace": "NAS",
    "W_NAS_broadcast_only": "NAS",
    "W_IIM_feedforward": "IIM",
    "W_SRPI_no_efference": "SRPI",
}
RULE_LABEL = {
    "impact_c": "IMPaCT rule (impact_c)", "impact": "IMPaCT rule",
    "logistic": "logistic (supervised)", "dcm_naive_bayes": "DCM-like naive Bayes",
    "chalmers_product": "product of credences", "union": "union (any PRESENT)",
    "geometric_mean_uncapped": "geometric mean", "geometric_mean_capped":
    "capped geometric mean", "arithmetic_mean": "arithmetic mean",
    "weakest_link": "weakest link", "power_mean_p-1": "power mean p = -1",
    "power_mean_p1": "power mean p = 1", "NAS_only": "NAS only", "IIM_only":
    "IIM only", "PhiR_bits": "ΦR (Gaussian)", "LZc": "LZc", "exact_IIM":
    "exact-TPM IIM",
}


def curves_from_cases(cases: pd.DataFrame) -> pd.DataFrame:
    from impact_pipeline.necessity import risk_coverage

    rows = []
    keys = ["rule"] + (["scenario"] if "scenario" in cases.columns else [])
    for key, sub in cases.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rc = risk_coverage(sub["confidence"].to_numpy(float),
                           sub["correct"].astype(bool).to_numpy())
        for cov, risk in zip(rc["coverage"], rc["risk"]):
            rows.append({**dict(zip(keys, key)), "coverage": cov, "risk": risk,
                         "aurc": rc["aurc"]})
    return pd.DataFrame(rows)


def _lead_rule(rules):
    """The IMPaCT rule among ``rules`` (v2 ``impact_c`` first), or None."""
    return next((r for r in IMPACT_RULES if r in rules), None)


def _rule_order(rules):
    rules = list(dict.fromkeys(rules))
    lead = _lead_rule(rules)
    if lead is not None:
        rules.remove(lead)
        rules.insert(0, lead)
    return rules


def _filter(df, audit, noise):
    out = df
    if audit is not None and "audit" in out.columns:
        out = out[out["audit"].astype(str) == audit]
    if "label_noise" in out.columns:
        out = out[np.isclose(pd.to_numeric(out["label_noise"], errors="coerce"),
                             noise)]
    return out


def hc9_statistics(decisions, necessity_set, noise=0.0):
    """Per rule, scenario none: the largest MPC_CONSISTENT rate over the
    single-deficit classes and the MPC_CONSISTENT rate on positive controls;
    and, scenario RAM+SRPI missing, the positive-control rate (as
    ``bench_hypotheses.hc9``)."""
    d0 = decisions[np.isclose(pd.to_numeric(decisions["label_noise"],
                                            errors="coerce"), noise)]
    cls = d0["class"].astype(str)
    is_pos = cls.isin(["all_present", "witness:PC_nominal"])
    targets = set(necessity_set)
    single = cls.map(
        lambda c: (c.startswith("single_deficit:") and c.split(":", 1)[1] in targets)
        or (c.startswith("witness:W_")
            and WITNESS_TARGET.get(c.split(":", 1)[1]) in targets))
    cons = d0["decision"].astype(str) == "MPC_CONSISTENT"
    rows = []
    for rule, sub_idx in d0.groupby("rule").groups.items():
        sub = d0.loc[sub_idx]
        none = sub["scenario"] == "none"
        s_single = sub[none & single.loc[sub_idx]]
        per_class = (s_single.assign(c=cons.loc[s_single.index])
                     .groupby("class")["c"].mean())
        pos_none = none & is_pos.loc[sub_idx]
        pos_miss = (sub["scenario"] == MISSING_SCENARIO) & is_pos.loc[sub_idx]
        rows.append({
            "rule": rule,
            "max_single_deficit_consistent": (float(per_class.max())
                                              if len(per_class) else np.nan),
            "worst_class": (str(per_class.idxmax()) if len(per_class) else None),
            "positive_consistent_none": (float(cons.loc[pos_none[pos_none].index]
                                               .mean()) if pos_none.any()
                                         else np.nan),
            "n_positive": int(pos_none.sum()),
            "positive_consistent_missing": (float(cons.loc[pos_miss[pos_miss].index]
                                                  .mean()) if pos_miss.any()
                                            else np.nan),
        })
    return pd.DataFrame(rows)


@fc.styled
def make_figure(summary_path, out_dir, cases_path=None, risk_coverage_path=None,
                scenario=None, decisions_path=None, audit=None, noise=0.0,
                necessity_set=("NAS", "IIM", "SRPI"), alpha=0.05, band=0.02):
    summary = fc.read_table(summary_path, ("rule", "scenario", "coverage",
                                           "selective_accuracy"), "rule summary")
    if audit is None and "audit" in summary.columns:
        audits = list(dict.fromkeys(summary["audit"].astype(str)))
        audit = "primary_Nanch" if "primary_Nanch" in audits else audits[0]
    summary = _filter(summary, audit, noise).copy()
    summary = summary[~summary["rule"].isin(DUPLICATES)]
    if "selective_risk" not in summary.columns:
        summary["selective_risk"] = 1.0 - summary["selective_accuracy"]
    inputs = [summary_path]
    if cases_path is not None:
        curves = curves_from_cases(fc.read_table(
            cases_path, ("rule", "confidence", "correct"), "rule cases"))
        inputs.append(cases_path)
    elif risk_coverage_path is not None:
        rc_table = fc.read_table(risk_coverage_path, ("rule", "coverage", "risk"),
                                 "risk-coverage table")
        curves = _filter(rc_table, audit, noise)
        if "reached" in curves.columns:
            curves = curves[curves["reached"].astype(str).str.lower() == "true"]
        inputs.append(risk_coverage_path)
    else:
        curves = pd.DataFrame(columns=["rule", "coverage", "risk"])
    scenarios = list(dict.fromkeys(summary["scenario"].astype(str)))
    scen = scenario or ("none" if "none" in scenarios else
                        (scenarios[0] if scenarios else None))
    miss = MISSING_SCENARIO if MISSING_SCENARIO in scenarios else next(
        (s for s in scenarios if s != scen), None)
    hc9 = None
    if decisions_path is not None:
        dec = fc.read_table(decisions_path, ("rule", "scenario", "label_noise",
                                             "class", "decision"),
                            "audit decisions")
        dec = dec[~dec["rule"].isin(DUPLICATES)]
        hc9 = hc9_statistics(dec, necessity_set, noise)
        inputs.append(decisions_path)

    main = summary[summary["scenario"].astype(str) == scen].copy()
    main = main.sort_values("selective_risk")
    rules = _rule_order(main["rule"].astype(str).tolist())
    rules = [r for r in rules if r != _lead_rule(rules)]
    lead = _lead_rule(main["rule"].astype(str).tolist())
    rules = ([lead] if lead else []) + rules
    highlight = [r for r in HIGHLIGHT if r in rules][:4]
    hstyle = {r: (HIGHLIGHT_COLORS[i], HIGHLIGHT_MARKERS[i])
              for i, r in enumerate(highlight)}
    rows = []

    fc.setup_style()
    n = len(rules)
    fig = plt.figure(figsize=(fc.FULL_WIDTH, max(3.3, 0.15 * n + 1.3)))
    ncols = 5 if hc9 is not None else 4
    gs = fig.add_gridspec(1, 2 + ncols, width_ratios=[2.3, 1.45] + [1] * ncols,
                          left=0.07, right=0.975, top=0.9, bottom=0.2, wspace=0.16)

    # A: risk-coverage
    ax = fig.add_subplot(gs[0, 0])
    cv = curves
    if "scenario" in cv.columns and scen is not None:
        cv = cv[cv["scenario"].astype(str) == scen]
    for r in rules:
        m = main[main["rule"] == r]
        if m.empty or not np.isfinite(m["selective_risk"].iloc[0]):
            continue
        color, marker = hstyle.get(r, (fc.INK["neutral"], "o"))
        ax.plot(m["coverage"], m["selective_risk"], marker=marker, lw=0,
                ms=4.2 if r in hstyle else 3, color=color,
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=4 if r in hstyle else 3)
    for r in highlight:
        sub = cv[cv["rule"].astype(str) == r].sort_values("coverage")
        color, marker = hstyle[r]
        if len(sub):
            ax.plot(sub["coverage"], sub["risk"], color=color,
                    lw=1.6 if r == lead else 1.0, zorder=3)
            rows += [{"panel": "A", "rule": r, "coverage": c, "risk": k}
                     for c, k in zip(sub["coverage"], sub["risk"])]
    for r in highlight:
        m = main[main["rule"] == r]
        if len(m):
            cx = float(m["coverage"].iloc[0])
            right = cx < 0.6
            ax.annotate(RULE_LABEL.get(r, r), (cx, float(m["selective_risk"].iloc[0])),
                        xytext=(5 if right else -5, -9 if r == "logistic" else 4),
                        textcoords="offset points", fontsize=5.5,
                        ha="left" if right else "right", color=fc.INK["primary"])
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, max(0.75, float(np.nanmax(main["selective_risk"])) * 1.08
                       if len(main) else 0.75))
    ax.set_xlabel("coverage (fraction determinate)")
    ax.set_ylabel("selective risk (error rate among determinate)")
    ax.text(0.02, 0.98, "lines: risk-coverage curves\ndots: operating point of "
            "every rule (gray: others)", transform=ax.transAxes, va="top",
            fontsize=5.3, color=fc.INK["secondary"])
    fc.panel_title(ax, "A", f"Risk and coverage (scenario {scen})")

    # B (and C): per-rule dot columns
    y = np.arange(n)
    cols = [("coverage", "coverage"), ("selective_risk", "selective risk"),
            ("p_excluded_given_positive", "P(EXCLUDED\n| positive)"),
            ("p_consistent_given_negative", "P(MPC_CONS.\n| negative)")]
    first = None
    for k, (col, label) in enumerate(cols):
        ax = fig.add_subplot(gs[0, 2 + k], sharey=first)
        first = first or ax
        for s_i, (sc, filled) in enumerate(((scen, True), (miss, False))):
            if sc is None or col not in summary.columns:
                continue
            sub = summary[summary["scenario"].astype(str) == sc].set_index("rule")
            for i, r in enumerate(rules):
                if r not in sub.index:
                    continue
                v = float(sub.loc[r, col])
                if not np.isfinite(v):
                    continue
                color = hstyle.get(r, (fc.INK["secondary"], "o"))[0]
                ax.plot([v], [i + (0.18 if not filled else -0.0)], "o", ms=3.2,
                        color=color, mfc=color if filled else "white", mew=0.7)
                rows.append({"panel": "B", "rule": r, "scenario": sc,
                             "quantity": col, "value": v})
        if col not in summary.columns:
            ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                    color=fc.INK["muted"])
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
        ax.set_ylim(n - 0.5, -0.7)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(label, fontsize=6)
        if k == 0:
            ax.set_yticks(y, [RULE_LABEL.get(r, r) for r in rules], fontsize=6)
            for t, r in zip(ax.get_yticklabels(), rules):
                if r == lead:
                    t.set_fontweight("bold")
            fc.panel_title(ax, "B", f"Per rule (filled: {scen}; open: {miss})")
        else:
            ax.tick_params(labelleft=False)
    if hc9 is not None:
        ax = fig.add_subplot(gs[0, 2 + len(cols)], sharey=first)
        h = hc9.set_index("rule")
        for i, r in enumerate(rules):
            if r not in h.index:
                continue
            fool = h.loc[r, "max_single_deficit_consistent"]
            sens = h.loc[r, "positive_consistent_none"]
            color = hstyle.get(r, (fc.INK["secondary"], "o"))[0]
            if np.isfinite(fool) and np.isfinite(sens):
                ax.plot([fool, sens], [i, i], color=fc.INK["grid"], lw=1.2, zorder=1)
            if np.isfinite(fool):
                ax.plot([fool], [i], "o", ms=3.4, color=fc.NEGATIVE, zorder=3,
                        mew=0)
            if np.isfinite(sens):
                ax.plot([sens], [i], "o", ms=3.4, color=fc.POSITIVE, zorder=3,
                        mew=0)
            rows.append({"panel": "C", "rule": r,
                         "max_single_deficit_consistent": fool,
                         "worst_class": h.loc[r, "worst_class"],
                         "positive_consistent_none": sens,
                         "positive_consistent_missing":
                             h.loc[r, "positive_consistent_missing"],
                         "n_positive": h.loc[r, "n_positive"]})
        ax.axvline(alpha, color=fc.INK["secondary"], lw=0.6)
        ax.axvline(alpha + band, color=fc.INK["secondary"], lw=0.6, alpha=0.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
        ax.set_ylim(n - 0.5, -0.7)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelleft=False)
        ax.set_xlabel("P(MPC_CONS.)", fontsize=6)
        fc.panel_title(ax, "C", "HC9", fontsize=7.5)
        ax.text(-0.35, -0.15, "red: worst single-deficit\nclass; blue: positive\n"
                f"controls; lines: {alpha:g}, {alpha + band:g}",
                transform=ax.transAxes, fontsize=5.2, va="top",
                color=fc.INK["secondary"])
    prov = fc.provenance(__file__, inputs)
    data = pd.DataFrame(rows).assign(audit=audit, label_noise=noise)
    return fc.save_figure(fig, out_dir, STEM, prov, data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--summary", required=True)
    ap.add_argument("--cases", default=None)
    ap.add_argument("--risk-coverage", default=None)
    ap.add_argument("--decisions", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--audit", default=None)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--necessity-set", default="NAS,IIM,SRPI")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(make_figure(args.summary, args.out, args.cases, args.risk_coverage,
                      args.scenario, args.decisions, args.audit, args.noise,
                      tuple(s for s in args.necessity_set.split(",") if s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
