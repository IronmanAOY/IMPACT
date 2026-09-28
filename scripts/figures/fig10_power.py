#!/usr/bin/env python
"""
Figure 10: power of the symmetric necessity criteria and identifiability of
the aggregation exponent.

Inputs: ``--power-dir`` with ``necessity_power.csv`` and
``necessity_min_n.csv`` (``scripts/necessity_power.py``) and
``--recovery-dir`` with ``rule_recovery_cells.csv``
(``scripts/simulate_rule_recovery.py``).

Panels: A probability of the correct outcome versus the number of
determinate report-positive episodes (SUPPORTED when necessity holds, solid;
FALSIFIED when it is violated, lighter) for up to three label-noise levels,
other parameters fixed at the first value on the grid (``tau`` selectable);
B stable minimum n for the target power versus label noise (one line per
tau); C bias of the recovered exponent p versus component reliability (one
line per true p) and D coverage of its profile-likelihood interval, at the
largest N and the middle inter-component correlation.
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

STEM = "fig10_power"
FIX = ("pi_nec", "pi_viol", "pi_unconscious", "base_rate", "sensitivity",
       "false_absent")


def _fixed(df, tau):
    sub = df[np.isclose(df["tau"], tau)]
    for k in FIX:
        if k in sub.columns and len(sub):
            sub = sub[np.isclose(sub[k], sorted(sub[k].unique())[0])]
    if "coverage" in sub.columns and len(sub):
        sub = sub[np.isclose(sub["coverage"], sub["coverage"].max())]
    return sub


@fc.styled
def make_figure(power_dir, recovery_dir, out_dir, tau=None, target_power=None):
    pdir, rdir = Path(power_dir), Path(recovery_dir)
    power = fc.read_table(pdir / "necessity_power.csv",
                          ("tau", "truth", "n", "power", "label_noise"),
                          "necessity power")
    min_n = fc.read_table(pdir / "necessity_min_n.csv",
                          ("tau", "truth", "label_noise", "target_power", "n_stable"),
                          "necessity min n")
    rec = fc.read_table(rdir / "rule_recovery_cells.csv",
                        ("p_true", "reliability", "rho", "n", "bias", "coverage"),
                        "rule recovery cells")
    tau = float(tau if tau is not None else sorted(power["tau"].unique())[0])
    target = float(target_power if target_power is not None
                   else sorted(min_n["target_power"].unique())[0])
    fc.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    rows = []

    ax = axes[0, 0]
    sub = _fixed(power, tau)
    noises = sorted(sub["label_noise"].unique())[:3]
    for j, lam in enumerate(noises):
        for truth, alpha_ in (("necessity_holds", 1.0), ("necessity_violated", 0.45)):
            s = sub[np.isclose(sub["label_noise"], lam) & (sub["truth"] == truth)]
            s = s.sort_values("n")
            if s.empty:
                continue
            lab = f"label noise {lam:g}" if truth == "necessity_holds" else None
            ax.plot(s["n"], s["power"], color=fc.SLOTS[j], alpha=alpha_,
                    lw=2.0 if truth == "necessity_holds" else 1.5, label=lab)
            rows += [{"panel": "A", "label_noise": lam, "truth": truth, "n": n,
                      "power": p} for n, p in zip(s["n"], s["power"])]
    ax.axhline(target, color=fc.INK["axis"], lw=0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("determinate report-positive episodes n")
    ax.set_ylabel("P(correct outcome)")
    fc.panel_title(ax, "A", f"Power (tau = {tau:g}; lighter: violated)")
    ax.legend(fontsize=6, loc="lower right")

    ax = axes[0, 1]
    m = min_n[np.isclose(min_n["target_power"], target)]
    taus = sorted(m["tau"].unique())[:3]
    for j, t in enumerate(taus):
        s = _fixed(m, t)
        # both truths must reach the target; NaN = not reached on the grid
        s = s.groupby("label_noise")["n_stable"].apply(
            lambda v: v.max() if v.notna().all() else np.nan)
        ax.plot(s.index, s.values, marker="o", color=fc.SLOTS[j], label=f"tau {t:g}",
                markeredgecolor=fc.INK["surface"], markeredgewidth=1.0)
        for lam in s.index[s.isna()]:
            ax.annotate("not reached", (lam, 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=5.5, color=fc.INK["muted"])
        rows += [{"panel": "B", "tau": t, "label_noise": lam, "n_stable": v}
                 for lam, v in s.items()]
    ax.set_xlabel("label noise (report-positive but not conscious)")
    ax.set_ylabel(f"min n for power {target:g} (both truths)")
    fc.panel_title(ax, "B", "Minimum sample size")
    ax.legend(fontsize=6)

    n_max = rec["n"].max()
    rhos = sorted(rec["rho"].unique())
    rho = rhos[len(rhos) // 2]
    r = rec[(rec["n"] == n_max) & np.isclose(rec["rho"], rho)]
    ptrue = sorted(r["p_true"].unique())
    for k, (col, ylabel, ref) in enumerate((("bias", "bias of p-hat", 0.0),
                                            ("coverage", "interval coverage", 0.95))):
        ax = axes[1, k]
        for j, pt in enumerate(ptrue[:8]):
            s = r[np.isclose(r["p_true"], pt)].sort_values("reliability")
            ax.plot(s["reliability"], s[col], marker="o", color=fc.SLOTS[j],
                    label=f"p = {pt:g}", markeredgecolor=fc.INK["surface"],
                    markeredgewidth=1.0)
            rows += [{"panel": "CD"[k], "p_true": pt, "reliability": rl, col: v}
                     for rl, v in zip(s["reliability"], s[col])]
        ax.axhline(ref, color=fc.INK["axis"], lw=0.8)
        ax.set_xlabel("component reliability")
        ax.set_ylabel(ylabel)
        fc.panel_title(ax, "CD"[k], f"Exponent recovery (N = {int(n_max)}, "
                       f"rho = {rho:g})" if k == 0 else "Profile-interval coverage")
        ax.legend(fontsize=6)
    fig.tight_layout()
    prov = fc.provenance(__file__, [pdir / "necessity_power.csv",
                                    pdir / "necessity_min_n.csv",
                                    rdir / "rule_recovery_cells.csv"])
    return fc.save_figure(fig, out_dir, STEM, prov, pd.DataFrame(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--power-dir", required=True)
    ap.add_argument("--recovery-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--target-power", type=float, default=None)
    args = ap.parse_args(argv)
    print(make_figure(args.power_dir, args.recovery_dir, args.out, args.tau,
                      args.target_power))
    return 0


if __name__ == "__main__":
    sys.exit(main())
