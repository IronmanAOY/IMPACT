#!/usr/bin/env python
"""
Figure 3: aggregation contours and the legacy-CI audit.

Input (``--audit-dir``, written by ``scripts/audit_aggregation.py``):
``aggregation_grid.csv`` (rule, x, y, value), ``legacy_compensation.csv``
(case, legacy_CI, capped_geometric, weakest_link) and ``implied_floors.csv``
(p, others, weight, threshold, floor).

Panels: A-D aggregate over two components (the other three at the reference)
for the legacy CI (uncapped geometric mean, evaluated with the legacy
function), the capped geometric mean (MPC degree), the weakest link and the
arithmetic mean, with the ``--threshold`` contour; E the compensation and
definedness demonstrations; F the component floor implied by a CI threshold
(weight 0.2, geometric mean) for the other components at 1, 2 and 10 times
the reference.
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

STEM = "fig3_aggregation"
CONTOUR_RULES = (
    ("legacy_CI", "Legacy CI (uncapped)"),
    ("capped_geometric", "Capped geometric"),
    ("weakest_link", "Weakest link"),
    ("arithmetic", "Arithmetic mean"),
)
DEMO_CASES = (
    ("compensation", "RAM 0.01, rest 10"),
    ("compensation_to_reference", "RAM 1e-4, rest 10"),
    ("tiny_positive", "RAM 1e-6, rest 1"),
    ("measured_zero", "RAM 0, rest 1"),
    ("undefined_component", "RAM NaN, rest 1"),
)
DEMO_RULES = (("legacy_CI", "legacy CI"), ("capped_geometric", "capped geometric"),
              ("weakest_link", "weakest link"))


@fc.styled
def make_figure(audit_dir, out_dir, threshold=0.5, floor_weight=0.2) -> dict:
    audit = Path(audit_dir)
    paths = {k: audit / f"{k}.csv" for k in
             ("aggregation_grid", "legacy_compensation", "implied_floors")}
    grid = fc.read_table(paths["aggregation_grid"], ("rule", "x", "y", "value"))
    comp = fc.read_table(paths["legacy_compensation"],
                         ("case",) + tuple(r for r, _ in DEMO_RULES))
    floors = fc.read_table(paths["implied_floors"],
                           ("p", "others", "weight", "threshold", "floor"))
    fc.setup_style()
    fig = plt.figure(figsize=(7.4, 6.6))
    top = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.3,
                           left=0.1, right=0.93, top=0.9, bottom=0.6)
    bottom = fig.add_gridspec(1, 2, wspace=0.4, left=0.2, right=0.97, top=0.47,
                              bottom=0.08)
    levels = np.linspace(0.0, 1.5, 16)
    cmap = fc.sequential_cmap()
    xname = str(grid["x_name"].iloc[0]) if "x_name" in grid else "component 1"
    yname = str(grid["y_name"].iloc[0]) if "y_name" in grid else "component 2"
    fig.text(0.1, 0.975, "Two components varied, the other three at the reference "
             f"(dot); black line: aggregate = {threshold:g}",
             fontsize=7.5, color=fc.INK["secondary"], va="top")
    cs = None
    for i, (rule, title) in enumerate(CONTOUR_RULES):
        ax = fig.add_subplot(top[0, i])
        fc.panel_title(ax, "ABCD"[i], title, fontsize=7.5)
        sub = grid[grid["rule"] == rule]
        if sub.empty:
            ax.text(0.5, 0.5, "not in input", ha="center", va="center",
                    color=fc.INK["muted"], transform=ax.transAxes)
            continue
        piv = sub.pivot_table(index="y", columns="x", values="value")
        X, Y = np.meshgrid(piv.columns.to_numpy(float), piv.index.to_numpy(float))
        Z = np.clip(piv.to_numpy(float), 0, levels[-1])
        cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend="max")
        ax.contour(X, Y, piv.to_numpy(float), levels=[threshold],
                   colors=fc.INK["primary"], linewidths=1.0)
        ax.plot([1], [1], marker="o", ms=4, color=fc.INK["primary"],
                markeredgecolor=fc.INK["surface"], markeredgewidth=1.0)
        ax.set_xlabel(f"{xname} / ref.")
        if i == 0:
            ax.set_ylabel(f"{yname} / ref.")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(False)
        ax.set_aspect("equal")
    if cs is not None:
        cax = fig.add_subplot(top[0, 4])
        cb = fig.colorbar(cs, cax=cax)
        cb.set_label("aggregate", color=fc.INK["secondary"])
        cb.outline.set_visible(False)

    ax = fig.add_subplot(bottom[0, 0])
    cases = [(c, lab) for c, lab in DEMO_CASES if c in set(comp["case"])]
    ypos = np.arange(len(cases), dtype=float)
    h = 0.26
    rows = []
    for j, (rule, lab) in enumerate(DEMO_RULES):
        vals = [float(comp.loc[comp["case"] == c, rule].iloc[0]) for c, _ in cases]
        yy = ypos + (j - 1) * h
        fin = np.isfinite(vals)
        ax.barh(yy[fin], np.asarray(vals)[fin], height=h * 0.85, color=fc.SLOTS[j],
                label=lab)
        for k, v in enumerate(vals):
            rows.append({"panel": "E", "case": cases[k][0], "rule": rule, "value": v})
            if not np.isfinite(v):
                ax.text(0.03, yy[k], "undefined", va="center", fontsize=5.5,
                        color=fc.INK["muted"])
            elif rule == "legacy_CI":
                ax.text(v + 0.05, yy[k], f"{v:.3g}", va="center", fontsize=6,
                        color=fc.INK["secondary"])
    ax.axvline(1.0, color=fc.INK["axis"], lw=0.8)
    ax.set_yticks(ypos, [lab for _, lab in cases])
    ax.set_ylim(len(cases) - 0.5, -0.5)
    ax.set_xlim(0, 3.0)
    ax.set_xlabel("aggregate (reference = 1)")
    fc.panel_title(ax, "E", "Compensation and definedness")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(bottom[0, 1])
    fsub = floors[(floors["p"] == 0.0)
                  & np.isclose(floors["weight"], float(floor_weight))]
    for j, m in enumerate(sorted(fsub["others"].unique())[:3]):
        s = fsub[fsub["others"] == m].sort_values("threshold")
        ax.plot(s["threshold"], s["floor"], marker="o", color=fc.SLOTS[j],
                label=f"rest at {m:g}x reference",
                markeredgecolor=fc.INK["surface"], markeredgewidth=1.0)
        for _, r in s.iterrows():
            rows.append({"panel": "F", "case": f"others={m:g}", "rule": "p=0",
                         "threshold": r["threshold"], "value": r["floor"]})
    ax.set_yscale("log")
    ax.set_xlabel("CI threshold t")
    ax.set_ylabel(f"implied floor of one component (w = {floor_weight:g})")
    fc.panel_title(ax, "F", "Implied floor (t / M^(1-w))^(1/w)")
    ax.legend(loc="lower right")

    g = grid.copy()
    g["panel"] = "A-D"
    data = pd.concat([g, pd.DataFrame(rows)], ignore_index=True, sort=False)
    prov = fc.provenance(__file__, list(paths.values()))
    return fc.save_figure(fig, out_dir, STEM, prov, data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)
    print(make_figure(args.audit_dir, args.out, args.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
