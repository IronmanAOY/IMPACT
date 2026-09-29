#!/usr/bin/env python
"""
Figure 5: null calibration of the component estimators (HC1).

Input (``--rates``): ``null_calibration_rates.csv`` from
``scripts/null_calibration.py`` (null_kind, n_time, n_nodes, principle, n,
false_present_rate, false_present_lo, false_present_hi; with the counts
``n_present``, ``n_absent``, ``n_undefined`` the intervals are exact
Clopper-Pearson intervals and panel B is drawn). Optional ``--replicates``
(``null_calibration_replicates.csv``: ``principle``, ``status``,
``status_reason``) splits UNDEFINED into inconclusive and without a construct
scale; optional ``--verdicts`` (``null_calibration_verdicts.csv``) adds the
verdict composition per null family (panel C).

Panels: A false-PRESENT rate (dot, 95% interval) of every principle with a
determinate status somewhere, per null cell (family, series length T, number
of nodes), against the preregistered bound alpha + 0.02; B status composition
per principle over all null systems (on null data ABSENT is the correct
status); C verdict composition per null family.
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

STEM = "fig5_null_calibration"
REQUIRED = ("null_kind", "n_time", "n_nodes", "principle", "n", "false_present_rate",
            "false_present_lo", "false_present_hi")
KIND_ORDER = ("ar1", "pink", "surrogate_iid", "surrogate_linear")
KIND_LABEL = {"ar1": "AR(1)", "pink": "pink (1/f)", "surrogate_iid": "i.i.d. surrogate",
              "surrogate_linear": "linear surrogate"}
NO_SCALE_REASONS = ("INVALID_ANCHORS", "NOT_DEFINED", "NO_NULL_CALIBRATION",
                    "DEGENERATE_NULL", "NON_FINITE_ESTIMATE")


def _cells(rates):
    kinds = [k for k in KIND_ORDER if k in set(rates["null_kind"])]
    kinds += sorted(set(rates["null_kind"]) - set(kinds))
    cells = []
    for k in kinds:
        sub = rates[rates["null_kind"] == k]
        for t, n in sorted({(int(a), int(b)) for a, b in zip(sub["n_time"],
                                                             sub["n_nodes"])}):
            cells.append((k, t, n))
    return cells


def _interval(row):
    if "n_present" in row and np.isfinite(row.get("n_present", np.nan)):
        return fc.clopper_pearson_interval(int(row["n_present"]), int(row["n"]))
    return float(row["false_present_lo"]), float(row["false_present_hi"])


def _plotted_principles(rates):
    if {"n_present", "n_absent"} <= set(rates.columns):
        det = rates.groupby("principle")[["n_present", "n_absent"]].sum().sum(axis=1)
        keep = [p for p in fc.PRINCIPLES if det.get(p, 0) > 0]
    else:
        keep = [p for p in fc.PRINCIPLES if p in set(rates["principle"])]
    return keep


def _composition(rates, replicates):
    """Per principle: counts of PRESENT, ABSENT, UNDEFINED (inconclusive) and
    UNDEFINED without a construct scale (or not defined)."""
    rows = []
    if replicates is not None:
        for p, sub in replicates.groupby("principle"):
            st = sub["status"].astype(str)
            reason = sub["status_reason"].astype(str)
            undef = st == "UNDEFINED"
            inconclusive = undef & reason.str.startswith("INCONCLUSIVE")
            rows.append({"principle": p, "n": int(len(sub)),
                         "PRESENT": int((st == "PRESENT").sum()),
                         "ABSENT": int((st == "ABSENT").sum()),
                         "UNDEFINED": int(inconclusive.sum()),
                         "NO_SCALE": int((undef & ~inconclusive).sum())})
    elif {"n_present", "n_absent", "n_undefined"} <= set(rates.columns):
        cols = ["n", "n_present", "n_absent", "n_undefined"]
        g = rates.groupby("principle")[cols].sum()
        for p, r in g.iterrows():
            rows.append({"principle": p, "n": int(r["n"]),
                         "PRESENT": int(r["n_present"]), "ABSENT": int(r["n_absent"]),
                         "UNDEFINED": int(r["n_undefined"]), "NO_SCALE": 0})
    out = pd.DataFrame(rows)
    if len(out):
        out["order"] = out["principle"].map({p: i for i, p in enumerate(fc.PRINCIPLES)})
        out = out.sort_values("order").drop(columns="order")
    return out


@fc.styled
def make_figure(rates_path, out_dir, verdicts_path=None, alpha=0.05, band=0.02,
                replicates_path=None):
    rates = fc.read_table(rates_path, REQUIRED, "null calibration rates")
    cells = _cells(rates)
    plotted = _plotted_principles(rates)
    bound = alpha + band
    replicates = None
    inputs = [rates_path]
    if replicates_path is not None:
        replicates = fc.read_table(replicates_path, ("principle", "status",
                                                     "status_reason"),
                                   "null calibration replicates")
        inputs.append(replicates_path)
    comp = _composition(rates, replicates)
    verdicts = None
    if verdicts_path is not None:
        verdicts = fc.read_table(verdicts_path, ("null_kind", "consistent_rate"),
                                 "null calibration verdicts")
        inputs.append(verdicts_path)

    fc.setup_style()
    n_a = max(1, len(plotted))
    fig = plt.figure(figsize=(fc.FULL_WIDTH, max(3.6, 0.15 * len(cells) + 1.4)))
    gs = fig.add_gridspec(3, n_a + 2, width_ratios=[1] * n_a + [0.62, 1.25],
                          height_ratios=[1, 1, 0.32], wspace=0.1, hspace=0.95,
                          left=0.155, right=0.955, top=0.885, bottom=0.11)
    rows = []
    ypos = np.arange(len(cells), dtype=float)
    xmax = max(0.12, float(np.nanmax(rates["false_present_hi"])) * 1.05)
    first_ax = None
    kinds = [c[0] for c in cells]
    for j, p in enumerate(plotted):
        ax = fig.add_subplot(gs[:, j], sharey=first_ax)
        first_ax = first_ax or ax
        ax.axvspan(bound, xmax, color=fc.INK["band"], lw=0, zorder=0)
        ax.axvline(bound, color=fc.INK["secondary"], lw=0.6, zorder=1)
        for i, (k, t, n) in enumerate(cells):
            sub = rates[(rates["null_kind"] == k) & (rates["n_time"] == t)
                        & (rates["n_nodes"] == n) & (rates["principle"] == p)]
            if sub.empty:
                ax.text(0.003, i, "not applicable", va="center", fontsize=5,
                        color=fc.INK["muted"])
                continue
            row = sub.iloc[0]
            v = float(row["false_present_rate"])
            lo, hi = _interval(row)
            ax.plot([lo, hi], [i, i], color=fc.SLOTS[0], lw=1.0, zorder=2,
                    solid_capstyle="butt")
            ax.plot([v], [i], "o", ms=3.2, color=fc.SLOTS[0], zorder=3,
                    markeredgecolor="white", markeredgewidth=0.5)
            rows.append({"panel": "A", "principle": p, "null_kind": k, "n_time": t,
                         "n_nodes": n, "n": int(row["n"]),
                         "n_present": row.get("n_present", np.nan),
                         "false_present_rate": v, "ci_lo": lo, "ci_hi": hi})
        ax.set_xlim(-0.004, xmax)
        ax.set_ylim(len(cells) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("false-PRESENT rate")
        ax.set_title(p, fontsize=7, color=fc.INK["primary"], fontweight="bold",
                     pad=3)
        if j == 0:
            ax.set_yticks(ypos, [f"{t} / {n}" for _, t, n in cells], fontsize=6)
            ax.text(-0.02, 1.012, "T / nodes", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=5.5, color=fc.INK["muted"])
            # null-family group labels left of the tick labels
            for k in dict.fromkeys(kinds):
                idx = [i for i, kk in enumerate(kinds) if kk == k]
                ax.text(-0.62, float(np.mean(idx)), fc_label(k), rotation=90,
                        ha="center", va="center", fontsize=6,
                        color=fc.INK["secondary"],
                        transform=ax.get_yaxis_transform())
        else:
            ax.tick_params(labelleft=False)
        for i in range(1, len(cells)):
            if kinds[i] != kinds[i - 1]:
                ax.axhline(i - 0.5, color=fc.INK["grid"], lw=0.6, zorder=0)
    if first_ax is not None:
        first_ax.annotate(f"bound {bound:g}", xy=(bound, len(cells) - 0.5),
                          xytext=(2, 2), textcoords="offset points", fontsize=5.5,
                          color=fc.INK["secondary"], annotation_clip=False)
        fig.text(0.01, 0.955, "$\\bf{A}$   False PRESENT on null systems "
                 "(dot: rate; line: exact 95% interval; shaded: above the "
                 "preregistered bound)", fontsize=7.5, color=fc.INK["primary"])

    # B: status composition per principle
    ax = fig.add_subplot(gs[0, n_a + 1])
    fc.panel_title(ax, "B", "Status on null systems")
    order = [("PRESENT", "PRESENT"), ("ABSENT", "ABSENT (correct)"),
             ("UNDEFINED", "UNDEFINED, inconclusive"),
             ("NO_SCALE", "UNDEFINED, no construct scale")]
    if len(comp):
        y = np.arange(len(comp))
        left = np.zeros(len(comp))
        for key, label in order:
            frac = (comp[key] / comp["n"]).to_numpy(float)
            ax.barh(y, frac, left=left, height=0.62, color=fc.STATUS_COLORS[key],
                    edgecolor="white", lw=0.6, label=label)
            left += frac
        for i, r in enumerate(comp.itertuples(index=False)):
            ax.text(1.02, i, f"{r.n}", va="center", fontsize=5.5,
                    color=fc.INK["muted"], transform=ax.get_yaxis_transform())
            rows.append({"panel": "B", "principle": r.principle, "n": r.n,
                         "PRESENT": r.PRESENT, "ABSENT": r.ABSENT,
                         "UNDEFINED_inconclusive": r.UNDEFINED,
                         "UNDEFINED_no_scale": r.NO_SCALE})
        ax.set_yticks(y, comp["principle"].tolist())
        ax.set_ylim(len(comp) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of null systems")
        ax.grid(axis="y", visible=False)
        ax.text(1.02, -0.75, "n", fontsize=5.5, color=fc.INK["muted"],
                transform=ax.get_yaxis_transform(), va="center")
    else:
        ax.text(0.5, 0.5, "no status counts in input", ha="center", va="center",
                color=fc.INK["muted"], transform=ax.transAxes)
        ax.set_axis_off()

    # C: verdict composition per null family
    ax = fig.add_subplot(gs[1, n_a + 1])
    fc.panel_title(ax, "C", "Verdicts on null systems")
    if verdicts is not None and {"excluded_rate", "undetermined_rate", "n"} <= set(
            verdicts.columns):
        v = verdicts.copy()
        kinds = [k for k in KIND_ORDER if k in set(v["null_kind"])]
        kinds += sorted(set(v["null_kind"]) - set(kinds))
        y = np.arange(len(kinds))
        left = np.zeros(len(kinds))
        agg = {}
        for k in kinds:
            s = v[v["null_kind"] == k]
            w = s["n"].astype(float)
            agg[k] = {
                "MPC_CONSISTENT": float((s["consistent_rate"] * w).sum() / w.sum()),
                "EXCLUDED": float((s["excluded_rate"] * w).sum() / w.sum()),
                "UNDETERMINED": float((s["undetermined_rate"] * w).sum() / w.sum()),
                "n": int(w.sum()),
                "necessity_set": ",".join(
                    sorted(set(s.get("necessity_set", pd.Series(["?"]))))),
            }
        for key in ("MPC_CONSISTENT", "EXCLUDED", "UNDETERMINED"):
            frac = np.array([agg[k][key] for k in kinds])
            ax.barh(y, frac, left=left, height=0.62, color=fc.VERDICT_COLORS[key],
                    edgecolor="white", lw=0.6, label=key.replace("_", " "))
            left += frac
        labels = []
        for k in kinds:
            ns = agg[k]["necessity_set"]
            short = "" if len(ns.split(",")) == 5 else "*"
            labels.append(f"{fc_label(k)}{short}")
            rows.append({"panel": "C", "null_kind": k, **agg[k]})
        ax.set_yticks(y, labels, fontsize=6)
        ax.set_ylim(len(kinds) - 0.5, -0.5)
        footnote = (
            "* necessity set RAM, PDI, SRPI (NAS and IIM not applicable); "
            "other families: all five principles"
            if any(lab.endswith("*") for lab in labels) else None)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of null systems")
        ax.grid(axis="y", visible=False)
        from matplotlib.patches import Patch

        handles = [
            Patch(color=fc.POSITIVE, label="PRESENT / MPC_CONSISTENT"),
            Patch(color=fc.NEGATIVE, label="ABSENT / EXCLUDED"),
            Patch(color=fc.NEUTRAL, label="UNDEFINED (inconclusive) / "
                  "UNDETERMINED"),
            Patch(facecolor=fc.NO_SCALE, edgecolor=fc.INK["axis"], lw=0.4,
                  label="UNDEFINED: no construct scale or not defined"),
        ]
        lax = fig.add_subplot(gs[2, n_a:])
        lax.set_axis_off()
        lax.legend(handles=handles, loc="lower left", ncol=1, fontsize=5.5,
                   handlelength=1.0, handleheight=0.8, labelspacing=0.3,
                   bbox_to_anchor=(0.0, -0.1))
        if footnote:
            lax.text(0.0, -0.3, footnote, fontsize=5, color=fc.INK["muted"],
                     transform=lax.transAxes, va="top", wrap=True)
    else:
        ax.text(0.5, 0.5, "no verdict input", ha="center", va="center",
                color=fc.INK["muted"], transform=ax.transAxes)
        ax.set_axis_off()

    data = pd.DataFrame(rows)
    prov = fc.provenance(__file__, inputs)
    return fc.save_figure(fig, out_dir, STEM, prov, data)


def fc_label(kind):
    return KIND_LABEL.get(kind, str(kind).replace("_", " "))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rates", required=True)
    ap.add_argument("--verdicts", default=None)
    ap.add_argument("--replicates", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)
    print(make_figure(args.rates, args.out, args.verdicts, args.alpha,
                      replicates_path=args.replicates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
