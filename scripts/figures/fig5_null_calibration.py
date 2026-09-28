#!/usr/bin/env python
"""
Figure 5: null calibration of the component estimators.

Input (``--rates``): ``null_calibration_rates.csv`` from
``scripts/null_calibration.py`` (null_kind, n_time, n_nodes, principle, n,
false_present_rate, false_present_lo, false_present_hi,
expected_rate_exchangeable_v1). Optional ``--verdicts``
(``null_calibration_verdicts.csv``) adds the verdict-level MPC_CONSISTENT rate
per null family.

Small multiples: one panel per null family (columns) and regime
``(T, nodes)`` (rows); per principle the false-PRESENT rate with its Wilson
95% interval, the alpha +- 0.02 entry band (gray wash, V2-5 a) and the rate
expected for an exchangeable Gaussian null with K surrogates under the v1
rule (open marker).
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


@fc.styled
def make_figure(rates_path, out_dir, verdicts_path=None, alpha=0.05, band=0.02):
    rates = fc.read_table(rates_path, REQUIRED, "null calibration rates")
    kinds = [k for k in KIND_ORDER if k in set(rates["null_kind"])]
    kinds += sorted(set(rates["null_kind"]) - set(kinds))
    regimes = sorted({(int(t), int(n)) for t, n in zip(rates["n_time"],
                                                       rates["n_nodes"])})
    fc.setup_style()
    ncol, nrow = len(kinds), len(regimes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 0.6, 1.7 * nrow + 0.9),
                             squeeze=False, sharey=True)
    ymax = max(0.2, float(np.nanmax(rates["false_present_hi"])) * 1.1)
    x = np.arange(len(fc.PRINCIPLES))
    for r, (t, n) in enumerate(regimes):
        for c, kind in enumerate(kinds):
            ax = axes[r, c]
            sub = rates[(rates["null_kind"] == kind) & (rates["n_time"] == t)
                        & (rates["n_nodes"] == n)].set_index("principle")
            ax.axhspan(alpha - band, alpha + band, color=fc.INK["band"], lw=0)
            ax.axhline(alpha, color=fc.INK["axis"], lw=0.8)
            for i, p in enumerate(fc.PRINCIPLES):
                if p not in sub.index:
                    ax.text(i, 0.01, "n/a", ha="center", fontsize=5.5,
                            color=fc.INK["muted"])
                    continue
                row = sub.loc[p]
                v = float(row["false_present_rate"])
                lo, hi = float(row["false_present_lo"]), float(row["false_present_hi"])
                ax.plot([i, i], [lo, hi], color=fc.PRINCIPLE_COLORS[p], lw=1.5)
                ax.plot([i], [v], marker="o", ms=5, color=fc.PRINCIPLE_COLORS[p],
                        markeredgecolor=fc.INK["surface"], markeredgewidth=1.0)
                exp = row.get("expected_rate_exchangeable_v1", np.nan)
                if np.isfinite(exp):
                    ax.plot([i + 0.25], [exp], marker="o", ms=3.5, mfc="none",
                            color=fc.INK["muted"], lw=0)
            ax.set_xticks(x, fc.PRINCIPLES, fontsize=6.5)
            ax.set_ylim(-0.02, ymax)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(kind.replace("_", " "), fontsize=8)
            if c == 0:
                ax.set_ylabel(f"T={t}, nodes={n}\nfalse-PRESENT rate")
    fig.text(0.01, 0.995, f"Null false-PRESENT rate (dot, 95% Wilson interval); gray "
             f"band: alpha {alpha:g} +- {band:g}; open circle: exchangeable-null "
             "expectation (v1 rule)", fontsize=7, color=fc.INK["secondary"], va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    inputs = [rates_path]
    data = rates.copy()
    if verdicts_path is not None:
        v = fc.read_table(verdicts_path, ("null_kind", "consistent_rate"),
                          "null calibration verdicts")
        v = v.assign(table="verdicts")
        data = pd.concat([data.assign(table="rates"), v], ignore_index=True, sort=False)
        inputs.append(verdicts_path)
    prov = fc.provenance(__file__, inputs)
    return fc.save_figure(fig, out_dir, STEM, prov, data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rates", required=True)
    ap.add_argument("--verdicts", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)
    print(make_figure(args.rates, args.out, args.verdicts, args.alpha))
    return 0


if __name__ == "__main__":
    sys.exit(main())
