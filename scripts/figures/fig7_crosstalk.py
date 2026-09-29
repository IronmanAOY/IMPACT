#!/usr/bin/env python
"""
Figure 7: cross-talk and dose-response of the component estimators on the
construct scale (MPC-Bench families A and C; HC3 and the registry criteria).

Input (``--components``): one row per (run, principle) with the construct
score ``c`` of the frozen evidence layer, e.g. ``components.csv`` of
``scripts/bench_hypotheses.py`` (columns ``family``, ``design``,
``witness_id``, ``bearer_mode``, ``seed``, ``sweep_knob``, ``sweep_level``,
``principle``, ``c``; each family judged against its own anchor). A
principle without an anchor has ``c`` undefined everywhere and is shown as
such. ``--sweep`` is accepted as an alias.

Panels: one cross-talk matrix per family: the median over seeds of the paired
witness contrast ``c(PC_nominal) - c(single-deficit witness of the switch)``
(``bench.analysis.witness_contrasts``; rows = switch, columns = principle;
target cells outlined, the two declared structural dependencies marked);
below, the dose-response of each anchored principle to its own switch (mean
``c`` +- SE over seeds per dose; ``bench.analysis.dose_response``) per
family, with the frozen cutoffs z = 0.25 (PRESENT needs the lower bound
above it) and delta = 0.10 (ABSENT needs the upper bound below it) and the
nominal dose marked.
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
REQUIRED = ("design", "principle", "c")
KNOB_TARGET = {"eta": "RAM", "K": "PDI", "g_b": "NAS", "c_int": "IIM", "e": "SRPI"}
KNOB_LABEL = {"eta": "η (plasticity)", "K": "K (repertoire)",
              "g_b": "g_b (workspace)", "c_int": "c_int (loops)",
              "e": "e (efference)"}
DECLARED = {("c_int", "NAS"), ("g_b", "IIM")}
NOMINAL = {"eta": 0.3, "K": 6.0, "g_b": 1.0, "c_int": 0.6, "e": 1.0}
CUTOFFS = (0.25, 0.10)


def _analysis():
    from impact_pipeline.bench import analysis as A

    return A


def _nominal():
    try:
        from impact_pipeline.bench.generators import NOMINAL_KNOBS

        return {k: float(v) for k, v in NOMINAL_KNOBS.to_dict().items()
                if k in KNOB_TARGET}
    except Exception:  # noqa: BLE001 - the figure works on CSVs alone
        return dict(NOMINAL)


def _load(path):
    df = fc.read_table(path, REQUIRED, "construct-scale components")
    df = df.copy()
    df["c"] = pd.to_numeric(df["c"], errors="coerce")
    if "family" not in df.columns:
        df["family"] = "A"
    if "bearer_mode" not in df.columns:
        df["bearer_mode"] = "system"
    df["family"] = df["family"].astype(str)
    # optional columns of the evaluator layout that dose_response reads
    for col, default in (("se_c", np.nan), ("status", "UNDEFINED")):
        if col not in df.columns:
            df[col] = default
    return df


def _anchored(df, family):
    sub = df[df["family"] == family]
    fin = sub.groupby("principle")["c"].apply(lambda s: bool(np.isfinite(s).any()))
    return [p for p in fc.PRINCIPLES if fin.get(p, False)]


@fc.styled
def make_figure(components_path, out_dir, cutoffs=CUTOFFS):
    A = _analysis()
    df = _load(components_path)
    families = sorted(df["family"].unique())
    contrasts = A.witness_contrasts(df) if "witness_id" in df.columns else \
        pd.DataFrame()
    # dose-response per family (each family against its own anchor)
    parts = []
    for fam in families:
        fam_df = df[df["family"] == fam]
        if (fam_df["design"] == "sweep").any():
            parts.append(A.dose_response(fam_df)[0].assign(family=fam))
    levels = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    nominal = _nominal()
    rows = []
    fc.setup_style()
    fig = plt.figure(figsize=(fc.FULL_WIDTH, 5.0))
    top = fig.add_gridspec(1, len(families) + 1,
                           width_ratios=[1] * len(families) + [0.05],
                           left=0.14, right=0.93, top=0.92, bottom=0.6, wspace=0.35)
    knobs = [k for k in KNOB_TARGET if len(contrasts) == 0
             or k in set(contrasts.get("switch", []))]
    vmax = 1.0
    if len(contrasts):
        vmax = max(0.5, float(np.nanmax(np.abs(contrasts["median_delta_c"]))))
    im = None
    for f_i, fam in enumerate(families):
        ax = fig.add_subplot(top[0, f_i])
        anchored = _anchored(df, fam)
        M = np.full((len(knobs), len(fc.PRINCIPLES)), np.nan)
        sub = contrasts[contrasts["family"] == fam] if len(contrasts) else contrasts
        for r in (sub.to_dict("records") if len(sub) else []):
            if r["switch"] in knobs and r["principle"] in fc.PRINCIPLES:
                M[knobs.index(r["switch"]), fc.PRINCIPLES.index(r["principle"])] = \
                    r["median_delta_c"]
                rows.append({"panel": "ABCD"[f_i], "family": fam,
                             "switch": r["switch"], "principle": r["principle"],
                             "n": r["n"], "median_delta_c": r["median_delta_c"],
                             "fraction_ge_0.5": r.get("fraction_ge_0.5"),
                             "own": r["own"],
                             "declared_dependency": r["declared_dependency"]})
        im = ax.imshow(M, cmap=fc.diverging_cmap(), vmin=-vmax, vmax=vmax,
                       aspect="auto")
        for j, p in enumerate(fc.PRINCIPLES):
            if p not in anchored:
                ax.add_patch(Rectangle((j - 0.5, -0.5), 1, len(knobs),
                                       facecolor=fc.NO_SCALE, edgecolor="white",
                                       lw=0.8, zorder=2))
                ax.text(j, (len(knobs) - 1) / 2, "no anchor", rotation=90,
                        ha="center", va="center", fontsize=5.5,
                        color=fc.INK["muted"], zorder=3)
        for i, k in enumerate(knobs):
            for j, p in enumerate(fc.PRINCIPLES):
                v = M[i, j]
                if p not in anchored or not np.isfinite(v):
                    continue
                dark = abs(v) > 0.55 * vmax
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if dark else fc.INK["primary"], zorder=3)
                if KNOB_TARGET.get(k) == p:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           ec=fc.INK["primary"], lw=1.1, zorder=4))
                if (k, p) in DECLARED:
                    ax.text(j + 0.44, i - 0.42, "†", ha="right", va="top",
                            fontsize=6, color=fc.INK["secondary"], zorder=4)
        ax.set_xticks(range(len(fc.PRINCIPLES)), fc.PRINCIPLES)
        ax.set_yticks(range(len(knobs)),
                      [KNOB_LABEL.get(k, k) if f_i == 0 else "" for k in knobs])
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fc.panel_title(ax, "ABCD"[f_i], f"Cross-talk, family {fam}")
        if f_i == 0:
            ax.set_ylabel("switch removed (single-deficit witness)")
        ax.set_xlabel("principle (construct score c)")
    if im is not None:
        cax = fig.add_subplot(top[0, len(families)])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("median paired Δc = c(control) − c(witness)",
                     color=fc.INK["secondary"], fontsize=6)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6, length=2)

    # dose-response of the anchored principles to their own switch
    own_knobs = [k for k in ("g_b", "c_int", "e")
                 if len(levels) and k in set(levels["sweep_knob"])]
    if not own_knobs and len(levels):
        own_knobs = list(dict.fromkeys(levels["sweep_knob"]))[:3]
    bottom = fig.add_gridspec(1, max(1, len(own_knobs)), left=0.08, right=0.985,
                              top=0.425, bottom=0.09, wspace=0.3)
    letters = "CDEFGH"[len(families) - 2:] if len(families) >= 2 else "BCDEFG"
    fam_style = {fam: (fc.SLOTS[i], "os^D"[i % 4]) for i, fam in enumerate(families)}
    for i, k in enumerate(own_knobs):
        ax = fig.add_subplot(bottom[0, i])
        p = KNOB_TARGET.get(k)
        drawn = []
        for fam in families:
            lv = levels[(levels["sweep_knob"] == k) & (levels["principle"] == p)
                        & (levels["family"] == fam)]
            lv = lv.sort_values("sweep_level")
            lv = lv[np.isfinite(lv["c_mean"])]
            if lv.empty:
                continue
            se = lv["c_sd"] / np.sqrt(lv["n_c"].clip(lower=1))
            color, marker = fam_style[fam]
            ax.fill_between(lv["sweep_level"], lv["c_mean"] - se, lv["c_mean"] + se,
                            color=color, alpha=0.15, lw=0)
            ax.plot(lv["sweep_level"], lv["c_mean"], color=color, marker=marker,
                    ms=3, lw=1.3, label=f"family {fam}",
                    markeredgecolor="white", markeredgewidth=0.4)
            drawn.append(fam)
            rows += [{"panel": letters[i], "family": fam, "knob": k, "principle": p,
                      "dose": r.sweep_level, "n": r.n_c, "c_mean": r.c_mean,
                      "c_se": s_, "present_rate": r.present_rate,
                      "absent_rate": r.absent_rate}
                     for r, s_ in zip(lv.itertuples(index=False), se)]
        missing = [f for f in families if f not in drawn]
        ax.axhline(cutoffs[0], color=fc.POSITIVE, lw=0.7, alpha=0.8, zorder=1)
        ax.axhline(cutoffs[1], color=fc.NEGATIVE, lw=0.7, alpha=0.8, zorder=1)
        ax.axhline(0.0, color=fc.INK["axis"], lw=0.6, zorder=1)
        if k in nominal:
            ax.axvline(nominal[k], color=fc.INK["axis"], lw=0.6, zorder=0)
            ax.text(nominal[k], 1.0, " nominal", transform=ax.get_xaxis_transform(),
                    ha="left", va="top", fontsize=5.5, color=fc.INK["muted"])
        xr = ax.get_xlim()[1]
        ax.text(xr, cutoffs[0], "z ", color=fc.POSITIVE, fontsize=5.5, ha="right",
                va="bottom")
        ax.text(xr, cutoffs[1], "δ ", color=fc.NEGATIVE, fontsize=5.5,
                ha="right", va="top")
        if missing:
            ax.text(0.98, 0.04,
                    "; ".join(f"family {f}: no {p} anchor" for f in missing),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
                    color=fc.INK["muted"])
        ax.set_xlabel(f"dose of {KNOB_LABEL.get(k, k)}")
        if i == 0:
            ax.set_ylabel("construct score c (mean ± SE)")
        fc.panel_title(ax, letters[i], f"{k} → {p}")
        if i == 0:
            ax.legend(loc="upper left", fontsize=6)
    fig.text(0.14, 0.515, "Outlined: the switch's own principle (target); "
             "† declared structural dependency (reported, not counted as "
             "cross-talk). Gray: no construct scale.", fontsize=5.8,
             color=fc.INK["secondary"])
    prov = fc.provenance(__file__, [components_path])
    return fc.save_figure(fig, out_dir, STEM, prov, pd.DataFrame(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--components", "--sweep", dest="components", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(make_figure(args.components, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
