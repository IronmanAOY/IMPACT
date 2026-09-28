#!/usr/bin/env python
"""
Figure 6: IIM validation on systems with known transition probability matrices.

Input (``--iim``, CSV, one row per system, run length and seed): ``system``
(e.g. independent, ring, all_to_all, star, chain), ``n_time``,
``delta_psi_est`` (sampled compute_IIM Delta_Psi, bits) and
``delta_psi_exact`` (compute_IIM_from_tpm on the generating TPM). Optional:
``seed``, and for the coupling sweep ``coupling`` and ``iim_z`` (calibrated
margin against circular-shift surrogates).

Panels: A median absolute error of the sampled Delta_Psi against the exact
value versus run length (log-log, one line per system); B sampled versus exact
Delta_Psi at the longest run length (identity line; systems labelled
directly); C median IIM margin (interquartile band) versus coupling with the
z = 1.645 PRESENT threshold (v1 rule), when the sweep columns are present.
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

STEM = "fig6_iim_validation"
REQUIRED = ("system", "n_time", "delta_psi_est", "delta_psi_exact")


@fc.styled
def make_figure(iim_path, out_dir, z_present=1.645):
    df = fc.read_table(iim_path, REQUIRED, "IIM validation table")
    df = df.copy()
    df["abs_error"] = (df["delta_psi_est"] - df["delta_psi_exact"]).abs()
    systems = list(dict.fromkeys(df["system"].astype(str)))[:8]
    fc.setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    rows = []

    ax = axes[0]
    for j, s in enumerate(systems):
        sub = df[df["system"] == s].groupby("n_time")["abs_error"].median()
        sub = sub[sub > 0]
        if sub.empty:
            continue
        ax.plot(sub.index, sub.values, marker="o", color=fc.SLOTS[j], label=s,
                markeredgecolor=fc.INK["surface"], markeredgewidth=1.0)
        rows += [{"panel": "A", "system": s, "n_time": t, "median_abs_error": v}
                 for t, v in sub.items()]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("run length T (samples)")
    ax.set_ylabel("median |sampled - exact| (bits)")
    fc.panel_title(ax, "A", "Convergence to the exact value")
    ax.legend(fontsize=6)

    ax = axes[1]
    t_max = df["n_time"].max()
    last = df[df["n_time"] == t_max]
    med = last.groupby("system")[["delta_psi_exact", "delta_psi_est"]].median()
    lim = float(np.nanmax(med.to_numpy())) * 1.15 if len(med) else 1.0
    lim = lim if lim > 0 else 1.0
    ax.plot([0, lim], [0, lim], color=fc.INK["axis"], lw=0.8)
    ax.plot(last["delta_psi_exact"], last["delta_psi_est"], "o", ms=3,
            color=fc.SLOTS[0], alpha=0.35, markeredgewidth=0)
    ax.plot(med["delta_psi_exact"], med["delta_psi_est"], "o", ms=5, color=fc.SLOTS[0],
            markeredgecolor=fc.INK["surface"], markeredgewidth=1.0)
    for s, r in med.iterrows():
        ax.annotate(str(s), (r["delta_psi_exact"], r["delta_psi_est"]),
                    xytext=(4, -3), textcoords="offset points", fontsize=6,
                    color=fc.INK["secondary"])
        rows.append({"panel": "B", "system": s, "n_time": t_max,
                     "delta_psi_exact": r["delta_psi_exact"],
                     "delta_psi_est_median": r["delta_psi_est"]})
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("exact Delta_Psi (bits)")
    ax.set_ylabel("sampled Delta_Psi (bits)")
    fc.panel_title(ax, "B", f"Sampled vs exact (T = {int(t_max)})")

    ax = axes[2]
    fc.panel_title(ax, "C", "Monotone in coupling")
    if {"coupling", "iim_z"} <= set(df.columns) and df["coupling"].notna().any():
        g = df.dropna(subset=["coupling", "iim_z"]).groupby("coupling")["iim_z"]
        q = g.quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color=fc.SLOTS[0], alpha=0.12,
                        lw=0)
        ax.plot(q.index, q[0.5], marker="o", color=fc.SLOTS[0],
                markeredgecolor=fc.INK["surface"], markeredgewidth=1.0)
        ax.axhline(z_present, color=fc.INK["axis"], lw=0.8)
        ax.text(q.index.min(), z_present, f" z = {z_present:g}", va="bottom",
                fontsize=6, color=fc.INK["muted"])
        ax.set_xlabel("coupling")
        ax.set_ylabel("IIM margin z (median, IQR)")
        rows += [{"panel": "C", "coupling": c, "iim_z_q25": r[0.25],
                  "iim_z_median": r[0.5], "iim_z_q75": r[0.75]}
                 for c, r in q.iterrows()]
    else:
        ax.text(0.5, 0.5, "no coupling sweep in input", ha="center", va="center",
                color=fc.INK["muted"], transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    prov = fc.provenance(__file__, [iim_path])
    return fc.save_figure(fig, out_dir, STEM, prov, pd.DataFrame(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iim", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(make_figure(args.iim, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
