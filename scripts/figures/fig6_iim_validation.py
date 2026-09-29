#!/usr/bin/env python
"""
Figure 6: IIM validation on systems with known transition probability matrices
(family B, HC2).

Input (``--iim``, CSV, one row per task; ``scripts/iim_validation.py``):
``system``, ``n_time``, ``delta_psi_est`` (sampled compute_IIM Delta_Psi,
bits), ``delta_psi_exact`` (compute_IIM_from_tpm on the generating TPM);
optional ``cut_mode`` (``bidirectional`` / ``directional``), ``seed``,
``iim_z`` (calibrated margin against circular-shift surrogates), ``null_mean``
and ``coupling`` (ring coupling sweep; rows with a coupling are the sweep).

Panels (one line or marker style per system, the same in every panel):
A, B median |sampled - exact| versus run length T, bidirectional and
directional cuts (log-log; HC2 b); C sampled versus exact Delta_Psi at the
longest run (median and interquartile range; identity line); D calibrated
margin of independent units per T with the margin 1.645 and the share of runs
above it (HC2 c); E the same for the feedforward star (HC2 d); F ring coupling
sweep: calibrated excess (Delta_Psi minus its null mean; median and IQR) and
the exact Delta_Psi versus coupling (HC2 e). Without ``cut_mode`` every row is
treated as one cut mode; panels whose columns are missing say so.
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
SYSTEM_ORDER = ("independent", "feedforward_star", "ring", "all_to_all", "xor_loop",
                "hidden_driver")
SYSTEM_LABEL = {"independent": "independent", "feedforward_star": "feedforward star",
                "ring": "ring", "all_to_all": "all-to-all", "xor_loop": "XOR loop",
                "hidden_driver": "hidden driver"}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
CUT_ORDER = ("bidirectional", "directional")
Z_MARGIN = 1.645


def _systems(df):
    present = list(dict.fromkeys(df["system"].astype(str)))
    ordered = [s for s in SYSTEM_ORDER if s in present]
    return ordered + [s for s in present if s not in ordered]


def _style(systems):
    """Colour and marker per system; the hidden driver (reported only) is
    drawn in neutral gray."""
    out, k = {}, 0
    for s in systems:
        if s == "hidden_driver":
            out[s] = (fc.INK["muted"], "x")
        else:
            out[s] = (fc.SLOTS[k % len(fc.SLOTS)], MARKERS[k % len(MARKERS)])
            k += 1
    return out


def _label(s):
    return SYSTEM_LABEL.get(s, str(s).replace("_", " "))


def _no_input(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", color=fc.INK["muted"],
            transform=ax.transAxes, fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])


@fc.styled
def make_figure(iim_path, out_dir, z_present=Z_MARGIN):
    df = fc.read_table(iim_path, REQUIRED, "IIM validation table").copy()
    if "cut_mode" not in df.columns:
        df["cut_mode"] = "bidirectional"
    if "coupling" not in df.columns:
        df["coupling"] = np.nan
    for c in ("delta_psi_est", "delta_psi_exact", "n_time", "coupling"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    grid = df[df["coupling"].isna()].copy()
    sweep = df[df["coupling"].notna()].copy()
    grid["abs_error"] = (grid["delta_psi_est"] - grid["delta_psi_exact"]).abs()
    systems = _systems(grid) if len(grid) else _systems(df)
    style = _style(systems)
    cuts = [c for c in CUT_ORDER if c in set(df["cut_mode"])]
    cuts += sorted(set(df["cut_mode"]) - set(cuts))
    rows = []
    fc.setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(fc.FULL_WIDTH, 4.7))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.15,
                        wspace=0.38, hspace=0.62)

    # A, B: convergence of the median absolute error with T
    for k, cut in enumerate(cuts[:2]):
        ax = axes[0, k]
        sub = grid[grid["cut_mode"] == cut]
        for s in systems:
            med = sub[sub["system"] == s].groupby("n_time")["abs_error"].median()
            med = med[med > 0]
            if med.empty:
                continue
            color, marker = style[s]
            ax.plot(med.index, med.values, color=color, marker=marker, ms=3.2,
                    lw=1.1 if s != "hidden_driver" else 0.8, label=_label(s),
                    markeredgecolor="white", markeredgewidth=0.4)
            steps = np.diff(med.values)
            rows += [{"panel": "AB"[k], "cut_mode": cut, "system": s, "n_time": t,
                      "median_abs_error": v,
                      "strictly_decreasing": bool(np.all(steps < 0))}
                     for t, v in med.items()]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("run length T (samples)")
        if k == 0:
            ax.set_ylabel("median |sampled − exact| (bits)")
        fc.panel_title(ax, "AB"[k], f"Convergence, {cut} cuts")
    if len(cuts) < 2:
        _no_input(axes[0, 1], "one cut mode in input")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0), fontsize=6.5, handlelength=1.8,
               columnspacing=1.4)

    # C: sampled vs exact at the longest run
    ax = axes[0, 2]
    t_max = grid["n_time"].max() if len(grid) else np.nan
    last = grid[grid["n_time"] == t_max]
    lim_hi = 0.0
    for cut in cuts[:2]:
        filled = cut == cuts[0]
        for s in systems:
            sub = last[(last["system"] == s) & (last["cut_mode"] == cut)]
            sub = sub.dropna(subset=["delta_psi_exact", "delta_psi_est"])
            if sub.empty:
                continue
            color, marker = style[s]
            ex = float(sub["delta_psi_exact"].median())
            q = sub["delta_psi_est"].quantile([0.25, 0.5, 0.75])
            ax.plot([ex, ex], [q[0.25], q[0.75]], color=color, lw=0.9)
            ax.plot([ex], [q[0.5]], marker=marker, ms=4, color=color,
                    mfc=color if filled else "white", mew=0.8, lw=0)
            lim_hi = max(lim_hi, ex, float(q[0.75]))
            rows.append({"panel": "C", "cut_mode": cut, "system": s,
                         "n_time": t_max, "delta_psi_exact": ex,
                         "est_q25": q[0.25], "est_median": q[0.5],
                         "est_q75": q[0.75]})
    lim = max(lim_hi * 1.3, 0.02)
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.6)
    ax.plot([0, lim], [0, lim], color=fc.INK["axis"], lw=0.7, zorder=0)
    ax.set_xlim(-0.001, lim)
    ax.set_ylim(-0.001, lim)
    ax.set_xlabel("exact ΔΨ (bits)")
    ax.set_ylabel("sampled ΔΨ (bits)")
    title_t = f"{int(t_max):,}" if np.isfinite(t_max) else "?"
    fc.panel_title(ax, "C", f"Sampled vs exact, T = {title_t}")
    if len(cuts) >= 2:
        ax.text(0.03, 0.97, f"filled: {cuts[0]}\nopen: {cuts[1]}",
                transform=ax.transAxes, va="top", fontsize=5.5,
                color=fc.INK["secondary"])

    # D, E: calibrated margins of independent units and the feedforward star
    # (colour = system as in A-C; filled = first cut mode, open = second)
    for k, (system, part) in enumerate((("independent", "HC2 c"),
                                        ("feedforward_star", "HC2 d"))):
        ax = axes[1, k]
        sub = grid[grid["system"] == system]
        title = f"{_label(system)}: margin vs null ({part})"
        if "iim_z" not in grid.columns or sub.empty:
            _no_input(ax, f"no {_label(system)} margins in input")
            fc.panel_title(ax, "DE"[k], title)
            continue
        color = style.get(system, (fc.SLOTS[0], "o"))[0]
        ts = sorted(sub["n_time"].unique())
        xs = {t: i for i, t in enumerate(ts)}
        rng = np.random.default_rng(0)
        stats = []
        for j, cut in enumerate(cuts[:2]):
            off = (j - 0.5) * 0.4 if len(cuts) > 1 else 0.0
            filled = j == 0
            for t in ts:
                z = pd.to_numeric(sub[(sub["n_time"] == t)
                                      & (sub["cut_mode"] == cut)]["iim_z"],
                                  errors="coerce").dropna()
                if z.empty:
                    continue
                x = xs[t] + off + rng.uniform(-0.08, 0.08, len(z))
                ax.plot(x, z, "o", ms=2.4, color=color,
                        mfc=color if filled else "white", mew=0.6, alpha=0.85,
                        label=cut if t == ts[0] else None)
                ax.plot([xs[t] + off - 0.14, xs[t] + off + 0.14], [z.median()] * 2,
                        color=fc.INK["primary"], lw=1.0)
                frac = float((z > z_present).mean())
                stats.append((xs[t] + off, frac))
                rows.append({"panel": "DE"[k], "system": system, "cut_mode": cut,
                             "n_time": t, "n": int(len(z)),
                             "median_z": float(z.median()),
                             "rate_z_above": frac})
        ax.axhline(z_present, color=fc.INK["secondary"], lw=0.6)
        if system == "feedforward_star":
            ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
            z_max = float(pd.to_numeric(sub["iim_z"], errors="coerce").max())
            ticks = [v for v in (-10, 0, 10, 100, 1000, 10000)
                     if v <= max(10.0, z_max * 1.2)]
            ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
            ax.minorticks_off()
        for x, frac in stats:
            ax.text(x, 1.01, f"{frac:.0%}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=5, color=fc.INK["secondary"])
        ax.text(-0.6, 1.07, "filled: bidirectional, open: directional; "
                f"share of runs with z > {z_present:g}:",
                transform=ax.get_xaxis_transform(), ha="left", va="bottom",
                fontsize=5, color=fc.INK["muted"])
        ax.set_xticks(range(len(ts)), [f"{int(t):,}" for t in ts])
        ax.set_xlim(-0.6, len(ts) - 0.4)
        ax.set_xlabel("run length T (samples)")
        ax.set_ylabel("calibrated margin z")
        ax.grid(axis="x", visible=False)
        ax.text(len(ts) - 0.45, z_present, f"z = {z_present:g}", ha="right",
                va="bottom", fontsize=5.5, color=fc.INK["secondary"])
        ax.set_title(f"$\\bf{{{'DE'[k]}}}$   {title}", loc="left", fontsize=7.5,
                     color=fc.INK["primary"], pad=16)

    # F: ring coupling sweep (colour = ring; filled / open = cut mode; the
    # exact values in ink)
    ax = axes[1, 2]
    ax.set_title("$\\bf{F}$   Ring coupling sweep (HC2 e)", loc="left",
                 fontsize=7.5, color=fc.INK["primary"], pad=16)
    ring_color = style.get("ring", (fc.SLOTS[2], "^"))[0]
    if len(sweep) and "null_mean" in sweep.columns:
        sweep = sweep.copy()
        sweep["excess"] = sweep["delta_psi_est"] - pd.to_numeric(
            sweep["null_mean"], errors="coerce")
        for j, cut in enumerate(cuts[:2]):
            filled = j == 0
            s = sweep[sweep["cut_mode"] == cut]
            q = s.groupby("coupling")["excess"].quantile([0.25, 0.5, 0.75]).unstack()
            ex = s.groupby("coupling")["delta_psi_exact"].median()
            ax.fill_between(q.index, q[0.25], q[0.75], color=ring_color, alpha=0.15,
                            lw=0)
            ax.plot(q.index, q[0.5], color=ring_color, marker="o", ms=3.2, lw=1.2,
                    mfc=ring_color if filled else "white", mew=0.8,
                    label=f"calibrated excess, {cut}")
            ax.plot(ex.index, ex.values, color=fc.INK["primary"], lw=0.7,
                    marker="D", ms=2.4, mfc=fc.INK["primary"] if filled else "white",
                    mew=0.6, label=f"exact, {cut}")
            rows += [{"panel": "F", "cut_mode": cut, "coupling": c,
                      "excess_q25": r[0.25], "excess_median": r[0.5],
                      "excess_q75": r[0.75], "delta_psi_exact": ex.get(c)}
                     for c, r in q.iterrows()]
        ax.axhline(0, color=fc.INK["axis"], lw=0.6)
        ax.set_xlabel("ring coupling")
        ax.set_ylabel("\u0394\u03a8 (bits)")
        ax.legend(loc="upper left", fontsize=5.2, handlelength=1.6)
    elif len(sweep) and "iim_z" in sweep.columns:
        g = sweep.groupby("coupling")["iim_z"].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(g.index, g[0.25], g[0.75], color=fc.SLOTS[0], alpha=0.15,
                        lw=0)
        ax.plot(g.index, g[0.5], color=fc.SLOTS[0], marker="o", ms=3)
        ax.set_xlabel("coupling")
        ax.set_ylabel("calibrated margin z (median, IQR)")
        rows += [{"panel": "F", "coupling": c, "iim_z_q25": r[0.25],
                  "iim_z_median": r[0.5], "iim_z_q75": r[0.75]}
                 for c, r in g.iterrows()]
    else:
        _no_input(ax, "no coupling sweep in input")

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
