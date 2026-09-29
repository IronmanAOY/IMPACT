#!/usr/bin/env python
"""
Figure 8: component statuses and verdicts on the MPC-Bench witnesses, and the
graded patchwork (HC4, HC5, HC7, HC8).

Inputs:

- ``--witnesses``: component statuses, either long (one row per run and
  principle: ``family``, ``design``, ``witness_id``, ``bearer_mode``,
  ``principle``, ``status``, optional ``status_reason`` and
  ``intended_bits``; e.g. ``components.csv`` of ``scripts/bench_hypotheses.py``)
  or wide (one row per run: ``witness_id``, ``verdict``, ``<P>_status``,
  optional ``intended_bits``);
- ``--verdicts`` (optional): one row per run with ``task_id``, ``family``,
  ``design``, ``witness_id``, ``bearer_mode`` and ``verdict_val`` (under the
  anchored necessity set) and/or ``verdict`` (``verdicts.csv`` of the
  evaluator); without it the wide table's ``verdict`` column is used;
- ``--patchwork`` (optional): graded-patchwork runs, either with ``lambda``,
  ``bearer_mode``, ``family``, ``verdict_anch_single_source`` and
  ``verdict_anch_no_source_rule`` (``patchwork_runs.csv``) or with
  ``coupling``, ``verdict`` and ``MPC_reason``.

Panels: per family, the status composition of each (witness, principle) cell
as a small stacked bar (PRESENT | UNDEFINED | ABSENT; light gray: no construct
scale), rows labelled with the intended mechanism bits (RAM, PDI, NAS, IIM,
SRPI); per family, the verdict composition of each witness; the
MPC_CONSISTENT rate of the patchwork with per-principle bearers versus the
inter-module coupling, with and without the single-source rule.
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
from matplotlib.patches import Patch, Rectangle  # noqa: E402

STEM = "fig8_witnesses"
VERDICT_ORDER = ("MPC_CONSISTENT", "UNDETERMINED", "EXCLUDED")
WITNESS_ORDER = (
    ("PC_nominal", "system", "positive control"),
    ("W_RAM_no_plasticity", "system", "no plasticity"),
    ("W_PDI_single_attractor", "system", "single attractor"),
    ("W_NAS_no_workspace", "system", "no workspace"),
    ("W_NAS_broadcast_only", "system", "broadcast only"),
    ("W_IIM_feedforward", "system", "feedforward"),
    ("W_SRPI_no_efference", "system", "no efference copy"),
    ("PW_patchwork", "principle", "patchwork, per-principle bearers"),
    ("PW_patchwork", "system", "patchwork, one bearer"),
    ("O_inert", "system", "inert (all off)"),
    ("O_hypersynchronous", "system", "hypersynchronous"),
    ("N_independent_noise", "system", "independent noise"),
    ("N_ar1", "system", "AR(1) noise"),
)


def _long_statuses(df):
    """Long status table (family, witness_id, bearer_mode, principle, status,
    no_scale) from either input layout."""
    if {"principle", "status"} <= set(df.columns):
        out = df.copy()
        if "design" in out.columns:
            out = out[out["design"] == "witnesses"]
    else:
        recs = []
        for _, r in df.iterrows():
            for p in fc.PRINCIPLES:
                col = f"{p}_status"
                if col in df.columns:
                    recs.append({"witness_id": r["witness_id"], "principle": p,
                                 "status": r[col],
                                 "intended_bits": r.get("intended_bits"),
                                 "family": r.get("family", "A"),
                                 "bearer_mode": r.get("bearer_mode", "system")})
        out = pd.DataFrame(recs)
    for col, default in (("family", "A"), ("bearer_mode", "system")):
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default).astype(str)
    out["status"] = out["status"].astype(str).str.upper()
    reason = out.get("status_reason", pd.Series("", index=out.index)).astype(str)
    out["no_scale"] = (out["status"] == "UNDEFINED") & ~reason.str.startswith(
        "INCONCLUSIVE") & reason.ne("nan") & reason.ne("")
    return out


def _bits(value):
    s = "" if value is None or (isinstance(value, float) and np.isnan(value)) \
        else str(value)
    s = s[1:] if s.startswith("b") else s
    if "." in s:
        s = s.split(".")[0]
    return s.zfill(5) if s.isdigit() and len(s) <= 5 else s


def _rows(st):
    present = {(w, b) for w, b in zip(st["witness_id"], st["bearer_mode"])}
    rows = [(w, b, lab) for w, b, lab in WITNESS_ORDER if (w, b) in present]
    known = {(w, b) for w, b, _ in rows}
    for w, b in sorted(present - known):
        rows.append((w, b, str(w)))
    return rows


def _verdicts(verdict_df, wide, rows):
    """Verdict fractions per (family, witness, bearer mode)."""
    if verdict_df is not None:
        v = verdict_df.copy()
        if "design" in v.columns:
            v = v[v["design"] == "witnesses"]
        col = "verdict_val" if "verdict_val" in v.columns else "verdict"
        label = "anchored necessity set" if col == "verdict_val" else "protocol"
    elif wide is not None and "verdict" in wide.columns:
        v, col, label = wide.copy(), "verdict", "recorded verdict"
    else:
        return None, None
    for c, d in (("family", "A"), ("bearer_mode", "system")):
        if c not in v.columns:
            v[c] = d
        v[c] = v[c].fillna(d).astype(str)
    v["v2"] = fc.normalize_verdicts(v[col]).fillna("UNDETERMINED")
    out = {}
    for (fam, w, b), s in v.groupby(["family", "witness_id", "bearer_mode"]):
        out[(fam, w, b)] = {k: float((s["v2"] == k).mean()) for k in VERDICT_ORDER}
        out[(fam, w, b)]["n"] = int(len(s))
    return out, label


@fc.styled
def make_figure(witness_path, out_dir, patchwork_path=None, verdicts_path=None):
    raw = fc.read_table(witness_path, ("witness_id",), "witness statuses")
    wide = None if {"principle", "status"} <= set(raw.columns) else raw
    st = _long_statuses(raw)
    inputs = [witness_path]
    vdf = None
    if verdicts_path is not None:
        vdf = fc.read_table(verdicts_path, ("witness_id",), "witness verdicts")
        inputs.append(verdicts_path)
    families = sorted(st["family"].unique())
    rows_order = _rows(st)
    verdicts, vlabel = _verdicts(vdf, wide, rows_order)
    bits = {}
    if "intended_bits" in st.columns:
        for (w, b), s in st.groupby(["witness_id", "bearer_mode"]):
            vals = [x for x in s["intended_bits"] if _bits(x)]
            bits[(w, b)] = _bits(vals[0]) if vals else ""
    data = []
    fc.setup_style()
    n = len(rows_order)
    fig = plt.figure(figsize=(fc.FULL_WIDTH, max(3.6, 0.19 * n + 2.1)))
    nf = len(families)
    gs = fig.add_gridspec(2, 2 * nf + 1,
                          width_ratios=[1.35] * nf + [0.5] * nf + [0.02],
                          height_ratios=[0.19 * n + 0.3, 1.25],
                          left=0.215, right=0.985, top=0.93, bottom=0.09,
                          wspace=0.12, hspace=0.3)
    letters = iter("ABCDEFGHIJ")
    y = np.arange(n)
    for f_i, fam in enumerate(families):
        ax = fig.add_subplot(gs[0, f_i])
        sub = st[st["family"] == fam]
        for i, (w, b, _) in enumerate(rows_order):
            for j, p in enumerate(fc.PRINCIPLES):
                cell = sub[(sub["witness_id"] == w) & (sub["bearer_mode"] == b)
                           & (sub["principle"] == p)]
                if cell.empty:
                    continue
                k = len(cell)
                fr = {
                    "PRESENT": float((cell["status"] == "PRESENT").mean()),
                    "UNDEFINED": float(((cell["status"] == "UNDEFINED")
                                        & ~cell["no_scale"]).mean()),
                    "ABSENT": float((cell["status"] == "ABSENT").mean()),
                    "NO_SCALE": float(cell["no_scale"].mean()),
                }
                x0 = j - 0.45
                for key in ("PRESENT", "UNDEFINED", "NO_SCALE", "ABSENT"):
                    wdt = 0.9 * fr[key]
                    if wdt > 0:
                        ax.add_patch(Rectangle((x0, i - 0.36), wdt, 0.72,
                                               facecolor=fc.STATUS_COLORS[key],
                                               edgecolor="none", lw=0))
                    x0 += wdt
                data.append({"panel": "status", "family": fam, "witness_id": w,
                             "bearer_mode": b, "principle": p, "n": k, **fr})
        ax.set_xlim(-0.55, len(fc.PRINCIPLES) - 0.45)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(range(len(fc.PRINCIPLES)), fc.PRINCIPLES, fontsize=6)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=1)
        if f_i == 0:
            labels = []
            for w, b, lab in rows_order:
                bb = bits.get((w, b), "")
                labels.append(f"{lab}  {bb}" if bb else lab)
            ax.set_yticks(y, labels, fontsize=6)
            ax.text(-0.02, 1.0, "intended bits\n(RAM PDI NAS IIM SRPI)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5,
                    color=fc.INK["muted"])
        else:
            ax.set_yticks(y, [""] * n)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(1, n):
            ax.axhline(i - 0.5, color=fc.INK["grid"], lw=0.4)
        ax.set_title(f"$\\bf{{{next(letters)}}}$   Statuses, family {fam}",
                     loc="left", fontsize=7.5, color=fc.INK["primary"], pad=12)

    for f_i, fam in enumerate(families):
        ax = fig.add_subplot(gs[0, nf + f_i])
        if verdicts is None:
            ax.text(0.5, 0.5, "no verdicts", ha="center", va="center",
                    color=fc.INK["muted"], transform=ax.transAxes)
            ax.set_axis_off()
            continue
        left = np.zeros(n)
        for key in VERDICT_ORDER:
            vals = np.array([verdicts.get((fam, w, b), {}).get(key, np.nan)
                             for w, b, _ in rows_order])
            vals = np.nan_to_num(vals)
            ax.barh(y, vals, left=left, height=0.72, color=fc.VERDICT_COLORS[key],
                    edgecolor="white", lw=0.4)
            left += vals
        for w, b, _ in rows_order:
            if (fam, w, b) in verdicts:
                data.append({"panel": "verdicts", "family": fam, "witness_id": w,
                             "bearer_mode": b, **verdicts[(fam, w, b)],
                             "verdict_basis": vlabel})
        ax.set_xlim(0, 1)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks(y, [""] * n)
        ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"], fontsize=5.5)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=1)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"$\\bf{{{next(letters)}}}$  Verdicts, {fam}", loc="left",
                     fontsize=7.5, color=fc.INK["primary"], pad=12)

    # legend (bottom left) and graded patchwork (bottom right)
    lax = fig.add_subplot(gs[1, 0])
    lax.set_axis_off()
    handles = [
        Patch(color=fc.POSITIVE, label="PRESENT / MPC_CONSISTENT"),
        Patch(color=fc.NEUTRAL, label="UNDEFINED (inconclusive) / UNDETERMINED"),
        Patch(facecolor=fc.NO_SCALE, edgecolor=fc.INK["axis"], lw=0.4,
              label="UNDEFINED: no construct scale"),
        Patch(color=fc.NEGATIVE, label="ABSENT / EXCLUDED"),
    ]
    lax.legend(handles=handles, loc="upper left", fontsize=6, handlelength=1.0,
               handleheight=0.8, bbox_to_anchor=(-0.45, 0.95))
    note = ("Each cell: fraction of seeds per status\n(left to right). "
            f"Verdicts: {vlabel or 'none'}.")
    lax.text(-0.45, 0.12, note, transform=lax.transAxes, fontsize=5.5,
             color=fc.INK["secondary"], va="bottom")

    ax = fig.add_subplot(gs[1, 1:2 * nf] if nf > 1 else gs[1, 1:])
    letter = next(letters)
    fc.panel_title(ax, letter, "Graded patchwork, per-principle bearers")
    if patchwork_path is not None:
        pw = fc.read_table(patchwork_path, (), "patchwork results")
        inputs.append(patchwork_path)
        _patchwork_panel(ax, pw, data)
    else:
        ax.text(0.5, 0.5, "no patchwork input", ha="center", va="center",
                color=fc.INK["muted"], transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    prov = fc.provenance(__file__, inputs)
    return fc.save_figure(fig, out_dir, STEM, prov, pd.DataFrame(data))


def _patchwork_panel(ax, pw, data):
    pw = pw.copy()
    if {"lambda", "verdict_anch_single_source",
            "verdict_anch_no_source_rule"} <= set(pw.columns):
        if "bearer_mode" in pw.columns:
            pw = pw[pw["bearer_mode"].astype(str) == "principle"]
        if "design" in pw.columns and (pw["design"] == "patchwork_sweep").any():
            pw = pw[pw["design"] == "patchwork_sweep"]
        fams = sorted(pw["family"].astype(str).unique()) if "family" in pw else ["A"]
        for f_i, fam in enumerate(fams):
            sub = pw[pw["family"].astype(str) == fam] if "family" in pw else pw
            color = fc.SLOTS[f_i % len(fc.SLOTS)]
            for rule, col, filled in (("single-source rule",
                                       "verdict_anch_single_source", True),
                                      ("no source rule",
                                       "verdict_anch_no_source_rule", False)):
                g = sub.groupby("lambda")[col].apply(
                    lambda s: float((fc.normalize_verdicts(s) == "MPC_CONSISTENT")
                                    .mean()))
                ax.plot(g.index, g.values, color=color, marker="os"[f_i % 2],
                        ms=3.6 if f_i == 0 else 4.6, lw=1.2 if filled else 0.9,
                        mfc=color if filled else "white", mew=0.8,
                        zorder=3 + len(fams) - f_i,
                        label=f"family {fam}, {rule}")
                data += [{"panel": "patchwork", "family": fam, "rule": rule,
                          "lambda": lam, "mpc_consistent_rate": v,
                          "n": int((sub["lambda"] == lam).sum())}
                         for lam, v in g.items()]
        ax.set_xlabel("inter-module coupling λ (0 = disconnected)")
    elif {"coupling", "verdict"} <= set(pw.columns):
        pw["v2"] = fc.normalize_verdicts(pw["verdict"])
        g = pw.groupby("coupling")
        cons = g["v2"].apply(lambda s: float((s == "MPC_CONSISTENT").mean()))
        ax.plot(cons.index, cons.values, marker="o", color=fc.SLOTS[0], ms=3,
                label="MPC_CONSISTENT")
        if "MPC_reason" in pw.columns:
            inc = g["MPC_reason"].apply(lambda s: float(s.astype(str).str.contains(
                "SOURCE_INCOHERENT", regex=False).mean()))
            ax.plot(inc.index, inc.values, marker="o", color=fc.SLOTS[1], ms=3,
                    label="SOURCE_INCOHERENT")
            data += [{"panel": "patchwork", "series": "SOURCE_INCOHERENT",
                      "coupling": c, "rate": v} for c, v in inc.items()]
        data += [{"panel": "patchwork", "series": "MPC_CONSISTENT", "coupling": c,
                  "rate": v} for c, v in cons.items()]
        ax.set_xlabel("inter-module coupling")
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.8, color=fc.INK["secondary"], lw=0.6)
    ax.text(ax.get_xlim()[1], 0.8, "0.8 (HC7 b) ", ha="right", va="bottom",
            fontsize=5.5, color=fc.INK["secondary"])
    ax.set_ylabel("MPC_CONSISTENT rate")
    ax.legend(fontsize=5.5, loc="center right", ncol=2, handlelength=1.6,
              bbox_to_anchor=(1.0, 0.52), columnspacing=1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--witnesses", required=True)
    ap.add_argument("--verdicts", default=None)
    ap.add_argument("--patchwork", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(make_figure(args.witnesses, args.out, args.patchwork, args.verdicts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
