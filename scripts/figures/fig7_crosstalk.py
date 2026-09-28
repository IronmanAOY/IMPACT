#!/usr/bin/env python
"""
Figure 7: cross-talk heatmap and dose-response of the MPC-Bench sweeps.

Input (``--sweep``): the ``results.csv`` of ``scripts/run_bench.py sweep``
(columns ``sweep_knob``, ``sweep_level`` and the null-standardised margins
``<P>_z``; one row per task). The knob of each principle is the generator's
mechanism switch (``bench.generators.SWITCH_FOR_PRINCIPLE``: eta -> RAM,
K -> PDI, g_b -> NAS, c_int -> IIM, e -> SRPI).

Panels: A cross-talk matrix: OLS slope of each component margin on the dose
(levels rescaled to [0, 1], ``bench.sweeps.dose_response_slopes``), rows =
swept knob, columns = component; target cells outlined; diverging scale
centred on 0. B-F dose-response per knob: mean margin (+- SE over seeds) of
the target component in its colour and of the other components in gray,
with the z = 1.645 PRESENT threshold (v1 rule).
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
from matplotlib.patches import Rectangle  # noqa: E402

STEM = "fig7_crosstalk"
DEFAULT_SWITCH = {"RAM": "eta", "PDI": "K", "NAS": "g_b", "IIM": "c_int", "SRPI": "e"}


def switch_map():
    try:
        from impact_pipeline.bench.generators import SWITCH_FOR_PRINCIPLE

        return dict(SWITCH_FOR_PRINCIPLE)
    except Exception:  # noqa: BLE001 - the figure works on CSVs alone
        return dict(DEFAULT_SWITCH)


def slopes_table(df, zcols):
    from impact_pipeline.bench.sweeps import dose_response_slopes

    return dose_response_slopes(df, zcols)


@fc.styled
def make_figure(sweep_path, out_dir, z_present=1.645):
    zcols = [f"{p}_z" for p in fc.PRINCIPLES]
    df = fc.read_table(sweep_path, ("sweep_knob", "sweep_level"), "sweep results")
    zcols = [c for c in zcols if c in df.columns]
    if not zcols:
        raise ValueError("sweep results have no <P>_z columns")
    df = df.copy()
    for c in zcols + ["sweep_level"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["sweep_knob", "sweep_level"])
    sw = switch_map()
    knob_target = {k: p for p, k in sw.items()}
    knobs = [sw[p] for p in fc.PRINCIPLES if sw[p] in set(df["sweep_knob"])]
    knobs += sorted(set(df["sweep_knob"]) - set(knobs))
    slopes = slopes_table(df[df["sweep_knob"].isin(knobs)], zcols)
    M = np.full((len(knobs), len(zcols)), np.nan)
    for _, r in slopes.iterrows():
        M[knobs.index(r["knob"]), zcols.index(r["value"])] = r["slope"]
    fc.setup_style()
    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1.0], hspace=0.6, wspace=0.45,
                          left=0.09, right=0.97, top=0.93, bottom=0.15)
    ax = fig.add_subplot(gs[0, 1:4])
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, cmap=fc.diverging_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(zcols)), [c[:-2] for c in zcols])
    ax.set_yticks(range(len(knobs)),
                  [f"{k} ({knob_target.get(k, '?')})" for k in knobs])
    ax.set_xlabel("component margin z")
    ax.set_ylabel("swept knob (target)")
    ax.grid(False)
    for i, k in enumerate(knobs):
        tgt = knob_target.get(k)
        if tgt is not None and f"{tgt}_z" in zcols:
            j = zcols.index(f"{tgt}_z")
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   ec=fc.INK["primary"], lw=1.2))
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("slope of z on dose (0 -> 1)", color=fc.INK["secondary"])
    cb.outline.set_visible(False)
    fc.panel_title(ax, "A", "Cross-talk: dose effect on every component")

    rows = [{"panel": "A", "knob": r["knob"], "component": r["value"][:-2],
             "slope": r["slope"], "n": r["n"]} for _, r in slopes.iterrows()]
    for i, k in enumerate(knobs[:5]):
        ax = fig.add_subplot(gs[1, i])
        sub = df[df["sweep_knob"] == k]
        tgt = knob_target.get(k)
        for c in zcols:
            g = sub.groupby("sweep_level")[c]
            mean, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))
            is_t = c == f"{tgt}_z"
            color = fc.PRINCIPLE_COLORS[c[:-2]] if is_t else fc.INK["neutral"]
            ax.plot(mean.index, mean.values, color=color, lw=2.0 if is_t else 1.0,
                    zorder=3 if is_t else 2)
            if is_t:
                ax.fill_between(mean.index, mean - se, mean + se, color=color,
                                alpha=0.12, lw=0)
            rows += [{"panel": "B-F", "knob": k, "component": c[:-2], "level": lv,
                      "mean_z": m, "se_z": s} for lv, m, s in
                     zip(mean.index, mean.values, se.values)]
        ax.axhline(z_present, color=fc.INK["axis"], lw=0.8)
        ax.set_xlabel(f"dose of {k}")
        if i == 0:
            ax.set_ylabel("margin z")
        fc.panel_title(ax, "BCDEF"[i], f"{k} -> {tgt}", fontsize=7.5)
    fig.text(0.09, 0.012, "B-F: target component in colour (+- SE), other "
             f"components gray; line: z = {z_present:g}", fontsize=6.5,
             color=fc.INK["secondary"])
    prov = fc.provenance(__file__, [sweep_path])
    return fc.save_figure(fig, out_dir, STEM, prov, pd.DataFrame(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(make_figure(args.sweep, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
