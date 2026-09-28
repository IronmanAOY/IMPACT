#!/usr/bin/env python
"""
Figure 8: witness x component profile, verdict strip and graded patchworks.

Inputs:

- ``--witnesses``: the ``results.csv`` of ``scripts/run_bench.py witnesses``
  (``witness_id``, ``verdict`` (v1 or v2 names), and ``<P>_status`` or the
  margins ``<P>_z``; optional ``intended_bits`` such as ``b10110``);
- ``--patchwork`` (optional): graded-patchwork results with ``coupling`` and
  ``verdict`` (and ``MPC_reason`` for the single-source code
  ``SOURCE_INCOHERENT``).

Panels: A fraction of seeds with the component PRESENT per witness (status
column, else margin > 1.645); dots mark the intended (mechanism-on) bits;
B verdict fractions per witness (EXCLUDED, MPC_CONSISTENT, UNDETERMINED);
C graded patchwork: MPC_CONSISTENT and SOURCE_INCOHERENT rates versus the
inter-module coupling.
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

STEM = "fig8_witnesses"
VERDICT_ORDER = ("EXCLUDED", "MPC_CONSISTENT", "UNDETERMINED")


def presence_matrix(df, witnesses, z_present=1.645):
    M = np.full((len(witnesses), len(fc.PRINCIPLES)), np.nan)
    for j, p in enumerate(fc.PRINCIPLES):
        scol, zcol = f"{p}_status", f"{p}_z"
        for i, w in enumerate(witnesses):
            sub = df[df["witness_id"] == w]
            if scol in df.columns:
                st = sub[scol].astype(str).str.upper()
                if len(st):
                    M[i, j] = float((st == "PRESENT").mean())
            elif zcol in df.columns:
                z = pd.to_numeric(sub[zcol], errors="coerce")
                if len(z):
                    M[i, j] = float((z > z_present).mean())
    return M


@fc.styled
def make_figure(witness_path, out_dir, patchwork_path=None):
    df = fc.read_table(witness_path, ("witness_id", "verdict"), "witness results")
    df = df.copy()
    df["verdict_v2"] = fc.normalize_verdicts(df["verdict"]).fillna("UNDETERMINED")
    witnesses = list(dict.fromkeys(df["witness_id"].astype(str)))
    M = presence_matrix(df, witnesses)
    fc.setup_style()
    height = max(3.2, 0.26 * len(witnesses) + 1.6)
    fig = plt.figure(figsize=(7.4, height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.2], wspace=0.3,
                          left=0.2, right=0.97, top=0.88, bottom=0.24)
    rows = []

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(M, cmap=fc.sequential_cmap(), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(fc.PRINCIPLES)), fc.PRINCIPLES)
    ax.set_yticks(range(len(witnesses)), witnesses, fontsize=6.5)
    ax.grid(False)
    if "intended_bits" in df.columns:
        for i, w in enumerate(witnesses):
            bits = str(df.loc[df["witness_id"] == w, "intended_bits"].iloc[0] or "")
            bits = bits[1:] if bits.startswith("b") else bits
            for j, b in enumerate(bits[:5]):
                if b == "1":
                    ax.plot(j, i, marker="o", ms=2.5, color=fc.INK["primary"])
    for i, w in enumerate(witnesses):
        for j, p in enumerate(fc.PRINCIPLES):
            rows.append({"panel": "A", "witness_id": w, "principle": p,
                         "fraction_present": M[i, j]})
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.14)
    cb.set_label("fraction PRESENT (dot: intended mechanism on)",
                 color=fc.INK["secondary"], fontsize=6.5)
    cb.outline.set_visible(False)
    fc.panel_title(ax, "A", "Component profile")

    ax = fig.add_subplot(gs[0, 1])
    frac = (df.groupby("witness_id")["verdict_v2"].value_counts(normalize=True)
            .unstack(fill_value=0.0).reindex(index=witnesses, columns=VERDICT_ORDER,
                                             fill_value=0.0))
    left = np.zeros(len(witnesses))
    for v in VERDICT_ORDER:
        ax.barh(np.arange(len(witnesses)), frac[v].to_numpy(), left=left, height=0.7,
                color=fc.VERDICT_COLORS[v], edgecolor=fc.INK["surface"], lw=1.0,
                label=v)
        left += frac[v].to_numpy()
    for w, r in frac.iterrows():
        rows.append({"panel": "B", "witness_id": w, **r.to_dict()})
    ax.set_ylim(len(witnesses) - 0.5, -0.5)
    ax.set_yticks(range(len(witnesses)), [""] * len(witnesses))
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of seeds")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=6)
    fc.panel_title(ax, "B", "Verdicts")

    ax = fig.add_subplot(gs[0, 2])
    fc.panel_title(ax, "C", "Graded patchwork")
    inputs = [witness_path]
    if patchwork_path is not None:
        pw = fc.read_table(patchwork_path, ("coupling", "verdict"), "patchwork results")
        inputs.append(patchwork_path)
        pw = pw.copy()
        pw["verdict_v2"] = fc.normalize_verdicts(pw["verdict"])
        g = pw.groupby("coupling")
        cons = g["verdict_v2"].apply(lambda s: float((s == "MPC_CONSISTENT").mean()))
        ax.plot(cons.index, cons.values, marker="o", color=fc.SLOTS[1],
                label="MPC_CONSISTENT", markeredgecolor=fc.INK["surface"],
                markeredgewidth=1.0)
        series = {"MPC_CONSISTENT": cons}
        if "MPC_reason" in pw.columns:
            inc = g["MPC_reason"].apply(lambda s: float(s.astype(str).str.contains(
                "SOURCE_INCOHERENT", regex=False).mean()))
            ax.plot(inc.index, inc.values, marker="o", color=fc.SLOTS[0],
                    label="SOURCE_INCOHERENT", markeredgecolor=fc.INK["surface"],
                    markeredgewidth=1.0)
            series["SOURCE_INCOHERENT"] = inc
        for name, s in series.items():
            rows += [{"panel": "C", "series": name, "coupling": c, "rate": v}
                     for c, v in s.items()]
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("inter-module coupling")
        ax.set_ylabel("fraction of seeds")
        ax.legend(fontsize=6, loc="center right")
    else:
        ax.text(0.5, 0.5, "no patchwork input", ha="center", va="center",
                color=fc.INK["muted"], transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    prov = fc.provenance(__file__, inputs)
    return fc.save_figure(fig, out_dir, STEM, prov, pd.DataFrame(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--witnesses", required=True)
    ap.add_argument("--patchwork", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(make_figure(args.witnesses, args.out, args.patchwork))
    return 0


if __name__ == "__main__":
    sys.exit(main())
