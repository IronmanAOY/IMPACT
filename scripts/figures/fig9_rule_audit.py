#!/usr/bin/env python
"""
Figure 9: audit of decision rules on estimated component statuses.

Inputs (from the rule-audit runs of MPC-Bench):

- ``--summary``: one row per rule and scenario (missingness / label-noise
  condition) with ``rule``, ``scenario``, ``coverage`` (fraction determinate)
  and ``selective_accuracy`` (accuracy on the determinate cases);
- ``--cases`` (per case: ``rule``, ``confidence``, ``correct``, optional
  ``scenario``) from which risk-coverage curves are computed with
  ``impact_pipeline.necessity.risk_coverage``, or ``--risk-coverage``
  (``rule``, ``coverage``, ``risk``, optional ``scenario``) with the curves.

Panels: A risk-coverage curves of each rule in ``--scenario`` (default: the
first scenario); the IMPaCT rule (``impact_c``, the v2 construct-scale rule of
``bench.audit``; the legacy ``impact`` when ``impact_c`` is absent) takes the
first colour and a thick line, rules beyond eight are drawn in gray. B coverage
and C selective accuracy per rule and scenario (first three scenarios
coloured, the rest gray).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import figure_common as fc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

STEM = "fig9_rule_audit"


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


IMPACT_RULES = ("impact_c", "impact")


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


@fc.styled
def make_figure(summary_path, out_dir, cases_path=None, risk_coverage_path=None,
                scenario=None):
    summary = fc.read_table(summary_path, ("rule", "scenario", "coverage",
                                           "selective_accuracy"), "rule summary")
    inputs = [summary_path]
    if cases_path is not None:
        curves = curves_from_cases(fc.read_table(
            cases_path, ("rule", "confidence", "correct"), "rule cases"))
        inputs.append(cases_path)
    elif risk_coverage_path is not None:
        curves = fc.read_table(risk_coverage_path, ("rule", "coverage", "risk"),
                               "risk-coverage table")
        inputs.append(risk_coverage_path)
    else:
        curves = pd.DataFrame(columns=["rule", "coverage", "risk"])
    scenarios = list(dict.fromkeys(summary["scenario"].astype(str)))
    scen = scenario or (scenarios[0] if scenarios else None)
    rules = _rule_order(list(summary["rule"].astype(str)) +
                        list(curves["rule"].astype(str)))
    colors = {r: (fc.SLOTS[i] if i < 8 else fc.INK["neutral"])
              for i, r in enumerate(rules)}
    fc.setup_style()
    fig = plt.figure(figsize=(7.4, max(3.0, 0.22 * len(rules) + 1.4)))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.15,
                          left=0.07, right=0.98, top=0.88, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    cv = curves
    if "scenario" in cv.columns and scen is not None:
        cv = cv[cv["scenario"].astype(str) == scen]
    for r in rules:
        sub = cv[cv["rule"].astype(str) == r].sort_values("coverage")
        if sub.empty:
            continue
        ax.step(sub["coverage"], sub["risk"], where="post", color=colors[r],
                lw=2.0 if r == rules[0] and _lead_rule(rules) else 1.2, label=r)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk (error rate)")
    fc.panel_title(ax, "A", f"Risk-coverage ({scen})" if scen else "Risk-coverage")
    if len(cv):
        ax.legend(fontsize=5.5, ncol=2)
    ax.set_box_aspect(1)
    scen_col = {s: (fc.SLOTS[i] if i < 3 else fc.INK["neutral"])
                for i, s in enumerate(scenarios)}
    y = {r: i for i, r in enumerate(rules)}
    for k, (col, label) in enumerate((("coverage", "coverage"),
                                      ("selective_accuracy", "selective accuracy"))):
        ax = fig.add_subplot(gs[0, 1 + k])
        for si, s in enumerate(scenarios):
            sub = summary[summary["scenario"].astype(str) == s]
            off = (si - (len(scenarios) - 1) / 2.0) * min(0.2, 0.6 / len(scenarios))
            ax.plot(sub[col], [y[str(r)] + off for r in sub["rule"]], "o", ms=4,
                    color=scen_col[s], label=s if k == 0 else None,
                    markeredgecolor=fc.INK["surface"], markeredgewidth=0.8)
        ax.set_yticks(range(len(rules)), rules if k == 0 else [""] * len(rules),
                      fontsize=6)
        ax.set_ylim(len(rules) - 0.5, -0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        fc.panel_title(ax, "BC"[k], label.capitalize())
        if k == 0:
            ax.legend(fontsize=5.5, loc="lower left")
    data = pd.concat([summary.assign(table="summary"), curves.assign(table="curves")],
                     ignore_index=True, sort=False)
    prov = fc.provenance(__file__, inputs)
    return fc.save_figure(fig, out_dir, STEM, prov, data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--summary", required=True)
    ap.add_argument("--cases", default=None)
    ap.add_argument("--risk-coverage", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(make_figure(args.summary, args.out, args.cases, args.risk_coverage,
                      args.scenario))
    return 0


if __name__ == "__main__":
    sys.exit(main())
