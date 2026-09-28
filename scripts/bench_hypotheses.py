#!/usr/bin/env python
"""
Preregistered computational hypotheses HC1-HC10 of paper 1 (MPC-Bench),
evaluated on the confirmatory runs made with the frozen code.

The decision rules, thresholds and inputs are those of
``docs/preregistration/MPC_BENCH_PREREGISTRATION.md`` (code-freeze tag
``mpcbench-freeze-v1``). Every status and verdict comes from the evidence
layer under the frozen protocols:

- ``protocols/mpc_bench_v1.json`` (necessity set: all five principles; the
  family-A positive-control anchor from development reference seeds);
- ``protocols/mpc_bench_v1_anchored.json`` (identical except the necessity
  set ``N_anch``: the principles with a valid reference anchor on the bench,
  i.e. a construct scale), used by the verdict-level hypotheses (HC4, HC5,
  HC7, HC9, HC10).

Family C (held out) is anchored on its own nominal positive control
(``PC_nominal``, confirmatory reference seeds 19000-19019; same anchor rule),
i.e. the frozen protocols with that reference and nothing else changed.
Adversarial systems (family-A agents and exact binary networks) use the
family-A anchor.

Outcomes: ``SUPPORTED``, ``FALSIFIED``, ``INDETERMINATE`` (the decision rule
does not decide) and ``NOT_EVALUABLE`` (a prerequisite is missing: input,
anchor, failed manipulation check). Every part of every hypothesis is
reported, whatever its outcome.

Refusals: records outside the confirmatory split, records whose provenance
does not carry the freeze tag, protocol files whose hashes differ from the
frozen ones, and records whose IIM cut mode differs from the protocol's.
``--development`` evaluates development runs to test this script; its
outputs are flagged ``DEVELOPMENT - NOT A RESULT``.

Example::

    python scripts/bench_hypotheses.py --out <ws>/hypotheses \\
        --freeze-tag mpcbench-freeze-v1 \\
        --bench-a <ws>/bench/A --bench-c <ws>/bench/C \\
        --reference-c <ws>/bench/reference_C \\
        --adversarial <ws>/bench/adversarial \\
        --null-calibration <ws>/null_calibration \\
        --iim-validation <ws>/iim_validation --audit <ws>/audit \\
        --manipulation <ws>/manipulation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline import evidence as ev  # noqa: E402
from impact_pipeline import necessity as nc  # noqa: E402
from impact_pipeline.bench import analysis as A  # noqa: E402

HYPOTHESES_VERSION = "mpc-bench-hypotheses/1.0.0"
FREEZE_TAG = "mpcbench-freeze-v1"
PROTOCOL = REPO_ROOT / "protocols" / "mpc_bench_v1.json"
PROTOCOL_ANCHORED = REPO_ROOT / "protocols" / "mpc_bench_v1_anchored.json"
# Hashes of the frozen protocol files (evidence.Protocol.hash); filled at the
# freeze, checked on every confirmatory evaluation.
EXPECTED_HASHES = {
    "mpc_bench_v1": "855f6a444b77030d33faa68fcb45e8576b931d2d681cf215d4dacdb57a6b2520",
    "mpc_bench_v1_anchored": (
        "780581f57d24251fc8ed8c39565f0a89a96740908f2ada3f65f8961cf1543f4c"
    ),
}
SUPPORTED, FALSIFIED = "SUPPORTED", "FALSIFIED"
INDETERMINATE, NOT_EVALUABLE = "INDETERMINATE", "NOT_EVALUABLE"

# Preregistered constants (see the preregistration document).
BAND = 0.02  # false-PRESENT bound = alpha + BAND
SEED_FRACTION = 0.80  # "in at least 80% of held-out seeds"
TARGET_DELTA_C = 0.50  # HC3: target change nominal - off, construct scale
SLOPE_RATIO = 0.20  # HC3: |off-target slope| < 0.2 x own slope
STABILITY_MARGIN = 2.0  # HC10: clear runs, |c - cutoff| > 2 se_c
FLIP_MAX = 0.10  # HC10: verdict-flip rate among clear runs
MIN_CLEAR_RUNS = 30  # HC10: fewer clear runs -> INDETERMINATE
MANIPULATION_PASS = 0.90  # prerequisite: a switch passes in >= 90% of seeds
HC2_Z = 1.645  # HC2: calibrated IIM margin (null-SD units)
HC2_BIDIR_POWER = 0.80
MEDIAN_Z_MAX = 0.5
# Switch (witness) -> target principle; the witnesses realise the off doses.
WITNESS_TARGET = {
    "W_RAM_no_plasticity": "RAM",
    "W_PDI_single_attractor": "PDI",
    "W_NAS_no_workspace": "NAS",
    "W_NAS_broadcast_only": "NAS",
    "W_IIM_feedforward": "IIM",
    "W_SRPI_no_efference": "SRPI",
}
SWITCH_WITNESS = A.SWITCH_WITNESS
# Declared structural dependencies (generator design, witnesses.yaml notes):
# c_int -> NAS and g_b -> IIM; reported, not counted against selectivity.
DECLARED_DEPENDENCIES = A.DECLARED_DEPENDENCIES
FAMILIES = ("A", "C")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _out(outcome, reason, **stats):
    return {"outcome": outcome, "reason": reason, **stats}


def _cp_lower(k, n, alpha=0.05):
    return float(nc.clopper_pearson(k, n, alpha, "lower")[0]) if n else float("nan")


def _cp_upper(k, n, alpha=0.05):
    return float(nc.clopper_pearson(k, n, alpha, "upper")[1]) if n else float("nan")


def _combine(parts):
    """Overall outcome of parts: FALSIFIED if any part is, SUPPORTED if all
    evaluable parts are and at least one is evaluable, else INDETERMINATE
    (NOT_EVALUABLE when no part is evaluable)."""
    outs = [p["outcome"] for p in parts]
    ev_ = [o for o in outs if o != NOT_EVALUABLE]
    if not ev_:
        return NOT_EVALUABLE
    if FALSIFIED in ev_:
        return FALSIFIED
    if all(o == SUPPORTED for o in ev_):
        return SUPPORTED
    return INDETERMINATE


def anchored(protocol):
    ref = protocol.reference
    return set(ref.get("values", {})) if ref.get("kind") == "external" else set()


def check_records(records, allow_development, freeze_tag):
    A.check_split(records, ("development",) if allow_development else ("confirmatory",))
    if allow_development:
        return
    bad = sorted(
        {
            r["task_id"]
            for r in records
            if ((r.get("provenance") or {}).get("code_version") or {}).get("freeze_tag")
            != freeze_tag
        }
    )
    if bad:
        raise ValueError(f"records without freeze tag {freeze_tag!r}: {bad[:5]}")


def check_cut_mode(records, protocol):
    want = (protocol.estimator_options("IIM") or {}).get("cut_mode", "bidirectional")
    bad = sorted(
        {
            r["task_id"]
            for r in records
            if "IIM" in (r.get("components") or {})
            and ((r.get("estimator_modes") or {}).get("IIM") or {}).get(
                "cut_mode", "bidirectional"
            )
            != want
        }
    )
    if bad:
        raise ValueError(
            f"IIM cut mode differs from the protocol's {want!r}: {bad[:5]}"
        )


# ---------------------------------------------------------------------------
# assessment of the bench records per family
# ---------------------------------------------------------------------------
def family_protocols(proto, proto_val, reference_c_records, allow_development):
    table = {"A": (proto, proto_val), "adversarial": (proto, proto_val)}
    info = {}
    if reference_c_records:
        ref_c, summ = A.family_reference(
            reference_c_records,
            alpha=proto.alpha,
            allow_confirmatory=not allow_development,
        )
        table["C"] = (
            A.protocol_for_family(proto, ref_c),
            A.protocol_for_family(proto_val, ref_c),
        )
        info["C"] = {
            "reference": ref_c,
            "summary": summ,
            "protocol_hash": table["C"][0].hash,
            "anchored_protocol_hash": table["C"][1].hash,
        }
    return table, info


def assess_family(records, protocols_by_family):
    """Component rows (raw + construct scale) and verdicts under both
    protocols, per family."""
    comps, verdicts = [], []
    for fam, recs in _by_family(records).items():
        if fam not in protocols_by_family:
            continue
        proto, proto_val = protocols_by_family[fam]
        raw = A.components_raw(recs)
        comps.append(A.assess_components(raw, proto.reference, proto))
        _, v_full = A.assess(recs, proto)
        _, v_val = A.assess(recs, proto_val)
        v = v_full.merge(
            v_val[["task_id", "verdict", "reasons", "necessity_set"]],
            on="task_id",
            suffixes=("", "_val"),
        )
        verdicts.append(v)
    comp = pd.concat(comps, ignore_index=True) if comps else pd.DataFrame()
    verd = pd.concat(verdicts, ignore_index=True) if verdicts else pd.DataFrame()
    return comp, verd


def _by_family(records):
    out = {}
    for r in records:
        fam = str(r.get("family"))
        out.setdefault("adversarial" if fam == "adversarial" else fam, []).append(r)
    return out


# ---------------------------------------------------------------------------
# prerequisite: manipulation checks
# ---------------------------------------------------------------------------
def manipulation_status(frames):
    if frames is None or frames.empty:
        return _out(NOT_EVALUABLE, "no manipulation-check input", per_switch=[]), {}
    rows, ok = [], {}
    for (fam, sw), sub in frames.groupby(["family", "switch"]):
        rate = float(sub["passed"].mean())
        rows.append(
            {
                "family": fam,
                "switch": sw,
                "n": int(len(sub)),
                "pass_rate": rate,
                "ok": rate >= MANIPULATION_PASS,
            }
        )
        ok[(fam, sw)] = rate >= MANIPULATION_PASS
    allok = all(ok.values())
    return (
        _out(
            SUPPORTED if allok else FALSIFIED,
            (
                "every switch passes in >= 90% of seeds"
                if allok
                else "some switch fails its manipulation check"
            ),
            per_switch=rows,
        ),
        ok,
    )


def _switch_ok(man_ok, family, switch):
    if not man_ok:
        return True  # no manipulation input: reported as a missing prerequisite
    return bool(man_ok.get((family, switch), False))


# ---------------------------------------------------------------------------
# HC1 null calibration
# ---------------------------------------------------------------------------
def hc1(null_calib_dir, comp, proto, alpha):
    cells = []
    anchored_p = anchored(proto)
    if null_calib_dir:
        rep = pd.read_csv(Path(null_calib_dir) / "null_calibration_replicates.csv")
        raw = rep.rename(columns={"null_kind": "kind"})
        raw = raw[
            [
                "kind",
                "n_time",
                "n_nodes",
                "replicate",
                "principle",
                "estimate",
                "null_mean",
                "null_sd",
                "n_null",
                "se",
                "se_n",
                "defined",
            ]
        ].copy()
        a = A.assess_components(raw, proto.reference, proto)
        a["null_class"] = (
            "null_calibration:"
            + a["kind"].astype(str)
            + ":T"
            + a["n_time"].astype(str)
            + ":N"
            + a["n_nodes"].astype(str)
        )
        cells.append(a[["null_class", "principle", "status"]])
    if len(comp):
        w = comp[comp["witness_id"].isin(["N_independent_noise", "N_ar1"])]
        if len(w):
            cells.append(
                w.assign(
                    null_class="witness:"
                    + w["family"].astype(str)
                    + ":"
                    + w["witness_id"].astype(str)
                )[["null_class", "principle", "status"]]
            )
    if not cells:
        return _out(NOT_EVALUABLE, "no null-calibration input")
    df = pd.concat(cells, ignore_index=True)
    rates = (
        df.assign(fp=df["status"] == "PRESENT", und=df["status"] == "UNDEFINED")
        .groupby(["null_class", "principle"])
        .agg(n=("fp", "size"), n_present=("fp", "sum"), undefined_rate=("und", "mean"))
        .reset_index()
    )
    rates["false_present_rate"] = rates["n_present"] / rates["n"]
    ev_rates = rates[rates["principle"].isin(anchored_p)]
    if ev_rates.empty:
        return _out(
            NOT_EVALUABLE, "no anchored principle", cells=rates.to_dict("records")
        )
    bound = alpha + BAND
    m = len(ev_rates)
    lo = [
        _cp_lower(k, n, alpha / m) for k, n in zip(ev_rates["n_present"], ev_rates["n"])
    ]
    ev_rates = ev_rates.assign(lower_familywise=lo)
    pooled = {}
    names = sorted(ev_rates["principle"].unique())
    for p in names:
        sub = ev_rates[ev_rates["principle"] == p]
        k, n = int(sub["n_present"].sum()), int(sub["n"].sum())
        pooled[p] = {
            "k": k,
            "n": n,
            "rate": k / n if n else float("nan"),
            "upper": _cp_upper(k, n, alpha / len(names)),
        }
    n_out = int((ev_rates["lower_familywise"] > bound).sum())
    if n_out:
        outcome, reason = (
            FALSIFIED,
            "a cell's family-wise lower bound exceeds alpha + 0.02",
        )
    elif all(v["upper"] < bound for v in pooled.values()):
        outcome, reason = SUPPORTED, "every pooled upper bound is below alpha + 0.02"
    else:
        outcome, reason = (
            INDETERMINATE,
            "a pooled upper bound is not below alpha + 0.02",
        )
    return _out(
        outcome,
        reason,
        bound=bound,
        pooled=pooled,
        n_cells=m,
        n_cells_outside=n_out,
        not_evaluable_principles=sorted(set(rates["principle"]) - anchored_p),
        cells=rates.to_dict("records"),
    )


# ---------------------------------------------------------------------------
# HC2 IIM ground truth (family B)
# ---------------------------------------------------------------------------
def hc2(iim_dir):
    if not iim_dir:
        return _out(NOT_EVALUABLE, "no IIM validation input")
    from scipy.stats import spearmanr

    df = pd.read_csv(Path(iim_dir) / "iim_validation.csv")
    df = df[df["status"] == "ok"]
    exact = pd.read_csv(Path(iim_dir) / "iim_validation_exact.csv")
    parts = {}
    ex = {(r.system, r.cut_mode): r.delta_psi_exact for r in exact.itertuples()}
    tol = 1e-9
    # (a) exact values (deterministic)
    a_ok = (
        abs(ex.get(("independent", "bidirectional"), np.nan)) < tol
        and abs(ex.get(("independent", "directional"), np.nan)) < tol
        and abs(ex.get(("feedforward_star", "directional"), np.nan)) < tol
        and ex.get(("feedforward_star", "bidirectional"), np.nan) > tol
        and ex.get(("independent", "bidirectional"), np.nan)
        < ex.get(("ring", "bidirectional"), np.nan)
        < ex.get(("all_to_all", "bidirectional"), np.nan)
    )
    parts["a_exact"] = _out(
        SUPPORTED if a_ok else FALSIFIED,
        "exact values (deterministic)",
        exact={f"{k[0]}|{k[1]}": v for k, v in ex.items()},
    )
    main = df[df["coupling"].isna()] if "coupling" in df else df
    # (b) bias shrinks monotonically with T (all systems but hidden_driver)
    rows, b_ok = [], True
    for (sysname, cm), sub in main.groupby(["system", "cut_mode"]):
        if sysname == "hidden_driver" or not np.isfinite(sub["delta_psi_exact"]).all():
            continue
        med = (
            sub.assign(err=(sub["delta_psi_est"] - sub["delta_psi_exact"]).abs())
            .groupby("n_time")["err"]
            .median()
            .sort_index()
        )
        mono = bool(np.all(np.diff(med.to_numpy()) < 0))
        b_ok &= mono
        rows.append(
            {
                "system": sysname,
                "cut_mode": cm,
                "monotone": mono,
                "median_abs_error": med.to_dict(),
            }
        )
    parts["b_bias_shrinks"] = _out(
        SUPPORTED if (rows and b_ok) else (FALSIFIED if rows else NOT_EVALUABLE),
        "median |error| decreases with T",
        rows=rows,
    )
    # (c) independent units calibrated
    ind = main[main["system"] == "independent"]
    c_rows, c_ok = [], True
    for (cm, t), sub in ind.groupby(["cut_mode", "n_time"]):
        z = sub["iim_z"].to_numpy(dtype=float)
        z = z[np.isfinite(z)]
        fp = float(np.mean(z > HC2_Z)) if z.size else float("nan")
        mz = float(np.median(z)) if z.size else float("nan")
        ok = bool(z.size and fp <= 0.05 + BAND and abs(mz) < MEDIAN_Z_MAX)
        c_ok &= ok
        c_rows.append(
            {
                "cut_mode": cm,
                "n_time": int(t),
                "n": int(z.size),
                "rate_z_above": fp,
                "median_z": mz,
                "ok": ok,
            }
        )
    parts["c_independent_calibrated"] = _out(
        SUPPORTED if (c_rows and c_ok) else (FALSIFIED if c_rows else NOT_EVALUABLE),
        "independent units: P(z > 1.645) <= 0.07 and |median z| < 0.5",
        rows=c_rows,
    )
    # (d) feedforward star: directional ~ null, bidirectional above null (T >= 3000)
    ff = main[(main["system"] == "feedforward_star") & (main["n_time"] >= 3000)]
    d_rows, d_ok = [], True
    for (cm, t), sub in ff.groupby(["cut_mode", "n_time"]):
        z = sub["iim_z"].to_numpy(dtype=float)
        z = z[np.isfinite(z)]
        rate = float(np.mean(z > HC2_Z)) if z.size else float("nan")
        ok = (rate <= 0.05 + BAND) if cm == "directional" else (rate >= HC2_BIDIR_POWER)
        d_ok &= bool(z.size and ok)
        d_rows.append(
            {
                "cut_mode": cm,
                "n_time": int(t),
                "n": int(z.size),
                "rate_z_above": rate,
                "ok": bool(ok),
            }
        )
    parts["d_feedforward"] = _out(
        SUPPORTED if (d_rows and d_ok) else (FALSIFIED if d_rows else NOT_EVALUABLE),
        "feedforward star: directional P(z>1.645) <= 0.07; bidirectional >= 0.8",
        rows=d_rows,
    )
    # (e) coupling sweep monotone
    sw = df[df["coupling"].notna()] if "coupling" in df else df.iloc[0:0]
    e_rows, e_ok = [], True
    for cm, sub in sw.groupby("cut_mode"):
        exc = (sub["delta_psi_est"] - sub["null_mean"]).to_numpy(dtype=float)
        r = spearmanr(sub["coupling"].to_numpy(dtype=float), exc)
        ok = bool(r.correlation > 0 and r.pvalue < 0.05)
        e_ok &= ok
        e_rows.append(
            {
                "cut_mode": cm,
                "rho": float(r.correlation),
                "p": float(r.pvalue),
                "n": int(len(sub)),
                "ok": ok,
            }
        )
    parts["e_coupling_monotone"] = _out(
        SUPPORTED if (e_rows and e_ok) else (FALSIFIED if e_rows else NOT_EVALUABLE),
        "Spearman rho(coupling, calibrated excess) > 0, p < 0.05",
        rows=e_rows,
    )
    hd = main[main["system"] == "hidden_driver"]
    parts["f_hidden_driver"] = _out(
        NOT_EVALUABLE,
        "reported only (observational estimator vs interventional TPM)",
        rows=(
            hd.groupby(["cut_mode", "n_time"])["iim_z"]
            .median()
            .reset_index()
            .to_dict("records")
        ),
    )
    return _out(
        _combine([v for k, v in parts.items() if k != "a_exact"]),
        "parts b-e (a is deterministic and reported)",
        parts=parts,
    )


# ---------------------------------------------------------------------------
# witness pairing helpers
# ---------------------------------------------------------------------------
def _witness_table(comp):
    w = comp[(comp["design"] == "witnesses") & (comp["bearer_mode"] == "system")]
    return w.set_index(["family", "witness_id", "seed", "principle"])


def _paired_delta(wt, family, witness, principle):
    """c(PC_nominal) - c(witness) per seed (same seed = same network)."""
    try:
        pc = wt.xs((family, "PC_nominal", principle), level=(0, 1, 3))["c"]
        ww = wt.xs((family, witness, principle), level=(0, 1, 3))["c"]
    except KeyError:
        return pd.Series(dtype=float)
    return (pc - ww).dropna()


# ---------------------------------------------------------------------------
# HC3 selectivity
# ---------------------------------------------------------------------------
def hc3(comp, protocols, man_ok):
    rows, parts = [], []
    if comp.empty:
        return _out(NOT_EVALUABLE, "no bench input")
    wt = _witness_table(comp)
    for fam in FAMILIES:
        if fam not in protocols:
            continue
        proto = protocols[fam][0]
        have = anchored(proto)
        z = {p: proto.cutoff_for(p)[0] for p in have}
        _, slopes = A.dose_response(comp[comp["family"] == fam])
        for switch, witness in SWITCH_WITNESS.items():
            target = A.KNOB_TARGET[switch]
            if not _switch_ok(man_ok, fam, switch):
                parts.append(
                    _out(
                        NOT_EVALUABLE,
                        "manipulation check failed",
                        family=fam,
                        switch=switch,
                    )
                )
                continue
            if target not in have:
                parts.append(
                    _out(
                        NOT_EVALUABLE,
                        "target has no anchor",
                        family=fam,
                        switch=switch,
                        target=target,
                    )
                )
                continue
            d_t = _paired_delta(wt, fam, witness, target)
            if d_t.empty:
                parts.append(
                    _out(
                        NOT_EVALUABLE,
                        "no paired witness runs",
                        family=fam,
                        switch=switch,
                    )
                )
                continue
            frac_t = float(np.mean(d_t >= TARGET_DELTA_C))
            ok = frac_t >= SEED_FRACTION
            off = {}
            for p in sorted(have - {target}):
                d = _paired_delta(wt, fam, witness, p)
                frac = float(np.mean(d.abs() < z[p])) if len(d) else float("nan")
                dep = (switch, p) in DECLARED_DEPENDENCIES
                off[p] = {
                    "fraction_below_z": frac,
                    "declared_dependency": dep,
                    "median_delta_c": float(d.median()) if len(d) else None,
                }
                if not dep:
                    ok &= bool(len(d) and frac >= SEED_FRACTION)
            srows, own, slope_ok = {}, np.nan, None
            if len(slopes) and switch in set(slopes["knob"]):
                s_own = slopes[
                    (slopes["knob"] == switch) & (slopes["principle"] == target)
                ]
                own = (
                    float(s_own["slope_c_per_unit_dose"].iloc[0])
                    if len(s_own)
                    else np.nan
                )
                # the target's own slope must be positive
                slope_ok = bool(np.isfinite(own) and own > 0)
                for p in sorted(have - {target}):
                    s = slopes[(slopes["knob"] == switch) & (slopes["principle"] == p)]
                    val = (
                        float(s["slope_c_per_unit_dose"].iloc[0]) if len(s) else np.nan
                    )
                    dep = (switch, p) in DECLARED_DEPENDENCIES
                    srows[p] = {"slope": val, "declared_dependency": dep}
                    if not dep:
                        slope_ok &= bool(
                            np.isfinite(val)
                            and np.isfinite(own)
                            and own > 0
                            and abs(val) < SLOPE_RATIO * own
                        )
            if not ok or slope_ok is False:
                outcome = FALSIFIED
            elif slope_ok is None:
                outcome = INDETERMINATE  # no sweep: the slope criterion is open
            else:
                outcome = SUPPORTED
            parts.append(
                _out(
                    outcome,
                    "target delta c >= 0.5 and off-target |delta c| < z in "
                    ">= 80% of seeds; off-target slopes < 0.2 x own slope",
                    family=fam,
                    switch=switch,
                    target=target,
                    n_seeds=int(len(d_t)),
                    target_fraction=frac_t,
                    target_median_delta_c=float(d_t.median()),
                    off_target=off,
                    own_slope=own,
                    off_target_slopes=srows,
                )
            )
            rows.append(parts[-1])
    return _out(_combine(parts), "every evaluable switch in both families", parts=parts)


# ---------------------------------------------------------------------------
# HC4 irredundancy (witnesses)
# ---------------------------------------------------------------------------
def hc4(comp, protocols, man_ok):
    if comp.empty:
        return _out(NOT_EVALUABLE, "no bench input")
    parts = []
    st = comp[(comp["design"] == "witnesses") & (comp["bearer_mode"] == "system")]
    irredundant = []
    for witness, target in WITNESS_TARGET.items():
        fam_ok = []
        for fam in FAMILIES:
            if fam not in protocols:
                continue
            nval = set(protocols[fam][1].necessity_set)
            if target not in nval:
                parts.append(
                    _out(
                        NOT_EVALUABLE,
                        "target not in the anchored set",
                        family=fam,
                        witness=witness,
                    )
                )
                fam_ok.append(None)
                continue
            switch = (
                next(k for k, v in SWITCH_WITNESS.items() if v == witness)
                if witness in SWITCH_WITNESS.values()
                else "g_b"
            )
            if not _switch_ok(
                man_ok, fam, "ff_only" if witness == "W_NAS_broadcast_only" else switch
            ):
                parts.append(
                    _out(
                        NOT_EVALUABLE,
                        "manipulation check failed",
                        family=fam,
                        witness=witness,
                    )
                )
                fam_ok.append(None)
                continue
            sub = st[(st["family"] == fam) & (st["witness_id"] == witness)]
            if sub.empty:
                parts.append(
                    _out(NOT_EVALUABLE, "no runs", family=fam, witness=witness)
                )
                fam_ok.append(None)
                continue
            piv = sub.pivot_table(
                index="seed", columns="principle", values="status", aggfunc="first"
            )
            good = piv[target] == "ABSENT"
            for p in nval - {target}:
                good &= piv.get(p, pd.Series("", index=piv.index)) == "PRESENT"
            frac = float(good.mean())
            ok = frac >= SEED_FRACTION
            fam_ok.append(ok)
            parts.append(
                _out(
                    SUPPORTED if ok else FALSIFIED,
                    "target ABSENT and the rest of N_anch PRESENT in >= 80%",
                    family=fam,
                    witness=witness,
                    target=target,
                    n_seeds=int(len(piv)),
                    fraction=frac,
                    target_absent_rate=float((piv[target] == "ABSENT").mean()),
                )
            )
        if fam_ok and all(x is True for x in fam_ok) and len(fam_ok) == len(FAMILIES):
            irredundant.append(witness)
    return _out(
        _combine(parts),
        "count reported whatever it is",
        parts=parts,
        irredundant_witnesses=irredundant,
        n_principles_irredundant=len({WITNESS_TARGET[w] for w in irredundant}),
    )


# ---------------------------------------------------------------------------
# HC5 specificity bound
# ---------------------------------------------------------------------------
def hc5(verd, protocols, alpha):
    if verd.empty:
        return _out(NOT_EVALUABLE, "no bench input")
    bound = alpha + BAND
    w = verd[(verd["design"] == "witnesses") & (verd["bearer_mode"] == "system")]
    rows = []
    for witness, target in WITNESS_TARGET.items():
        for fam in FAMILIES:
            if fam not in protocols or target not in protocols[fam][1].necessity_set:
                continue
            sub = w[(w["family"] == fam) & (w["witness_id"] == witness)]
            if sub.empty:
                continue
            k = int((sub["verdict_val"] == "MPC_CONSISTENT").sum())
            rows.append({"family": fam, "witness": witness, "k": k, "n": int(len(sub))})
    if not rows:
        return _out(NOT_EVALUABLE, "no single-deficit class with its target in N_anch")
    m = len(rows)
    for r in rows:
        r["rate"] = r["k"] / r["n"]
        r["lower_familywise"] = _cp_lower(r["k"], r["n"], alpha / m)
    k = sum(r["k"] for r in rows)
    n = sum(r["n"] for r in rows)
    upper = _cp_upper(k, n, alpha)
    if any(r["lower_familywise"] > bound for r in rows):
        out, why = FALSIFIED, "a class's family-wise lower bound exceeds alpha + 0.02"
    elif upper < bound:
        out, why = SUPPORTED, "pooled upper bound below alpha + 0.02"
    else:
        out, why = INDETERMINATE, "pooled upper bound not below alpha + 0.02"
    return _out(
        out, why, bound=bound, pooled={"k": k, "n": n, "upper": upper}, classes=rows
    )


# ---------------------------------------------------------------------------
# HC6 factorial: PRESENT when the mechanism is off
# ---------------------------------------------------------------------------
def hc6(comp, protocols, alpha):
    if comp.empty:
        return _out(NOT_EVALUABLE, "no bench input")
    f = comp[comp["design"] == "factorial"].copy()
    if f.empty:
        return _out(NOT_EVALUABLE, "no factorial runs")
    order = list(A.PRINCIPLES)
    f["bit"] = [
        int(b[order.index(p)]) if isinstance(b, str) and len(b) == 5 else -1
        for b, p in zip(f["intended_bits"], f["principle"])
    ]
    rows = []
    for fam in FAMILIES:
        if fam not in protocols:
            continue
        have = anchored(protocols[fam][0])
        sub = f[(f["family"] == fam) & (f["bit"] == 0) & f["principle"].isin(have)]
        for (cell, p), g in sub.groupby(["cell_id", "principle"]):
            rows.append(
                {
                    "family": fam,
                    "cell": cell,
                    "principle": p,
                    "k": int((g["status"] == "PRESENT").sum()),
                    "n": int(len(g)),
                }
            )
    if not rows:
        return _out(NOT_EVALUABLE, "no anchored principle with off cells")
    df = pd.DataFrame(rows)
    bound = alpha + BAND
    m = len(df)
    df["rate"] = df["k"] / df["n"]
    df["lower_familywise"] = [
        _cp_lower(k, n, alpha / m) for k, n in zip(df["k"], df["n"])
    ]
    pooled = {}
    names = sorted(df["principle"].unique())
    for p in names:
        g = df[df["principle"] == p]
        pooled[p] = {
            "k": int(g["k"].sum()),
            "n": int(g["n"].sum()),
            "upper": _cp_upper(
                int(g["k"].sum()), int(g["n"].sum()), alpha / len(names)
            ),
        }
    if (df["lower_familywise"] > bound).any():
        out, why = FALSIFIED, "a (cell, principle) lower bound exceeds alpha + 0.02"
    elif all(v["upper"] < bound for v in pooled.values()):
        out, why = SUPPORTED, "every pooled upper bound below alpha + 0.02"
    else:
        out, why = INDETERMINATE, "a pooled upper bound not below alpha + 0.02"
    on = f[(f["bit"] == 1)]
    absent_on = (
        on.groupby(["family", "cell_id", "principle"])["status"]
        .apply(lambda s: float((s == "ABSENT").mean()))
        .reset_index(name="absent_rate_when_on")
    )
    return _out(
        out,
        why,
        bound=bound,
        pooled=pooled,
        cells=df.to_dict("records"),
        revisions=df[df["rate"] > bound].to_dict("records"),
        absent_when_on=absent_on[absent_on["absent_rate_when_on"] > 0.2].to_dict(
            "records"
        ),
    )


# ---------------------------------------------------------------------------
# HC7 single-source constraint (patchwork)
# ---------------------------------------------------------------------------
def hc7(records, protocols):
    pw = [r for r in records if r.get("witness_id") == "PW_patchwork"]
    if not pw:
        return _out(NOT_EVALUABLE, "no patchwork witness runs")
    from impact_pipeline.bench.export import evidence_verdict

    parts = []
    for fam in FAMILIES:
        if fam not in protocols:
            continue
        pval = protocols[fam][1]
        unconstrained = pval.replace(source_rule="none")
        recs = [
            r
            for r in pw
            if r.get("family") == fam and r.get("bearer_mode") == "principle"
        ]
        if not recs:
            continue
        cons_rule, cons_free, reasons = 0, 0, []
        for r in recs:
            res = {
                "components": r["components"],
                "estimator_modes": r.get("estimator_modes") or {},
            }
            meta = {"substrate": (r.get("system") or {}).get("substrate")}
            v1 = evidence_verdict(res, meta, protocol=pval)
            v2 = evidence_verdict(res, meta, protocol=unconstrained)
            cons_rule += v1["verdict"] == "MPC_CONSISTENT"
            cons_free += v2["verdict"] == "MPC_CONSISTENT"
            reasons.extend(v1["reasons"])
        n = len(recs)
        a = _out(
            SUPPORTED if cons_rule == 0 else FALSIFIED,
            "single-source rule: no MPC_CONSISTENT for the disconnected patchwork",
            family=fam,
            n=n,
            consistent=int(cons_rule),
            reason_counts=pd.Series(reasons).value_counts().to_dict(),
        )
        b = _out(
            SUPPORTED if cons_free / n >= SEED_FRACTION else FALSIFIED,
            "unconstrained conjunctive rule: MPC_CONSISTENT in >= 80%",
            family=fam,
            n=n,
            consistent=int(cons_free),
            rate=cons_free / n,
        )
        parts += [dict(a, part="a"), dict(b, part="b")]
    if not parts:
        return _out(NOT_EVALUABLE, "no principle-bearer patchwork runs")
    return _out(
        _combine(parts),
        "parts a and b per family; the joint-dependence "
        "test of the positive control is not part of the bench records",
        parts=parts,
    )


# ---------------------------------------------------------------------------
# HC8 SRPI agency
# ---------------------------------------------------------------------------
def hc8(comp, protocols, man_ok):
    if comp.empty:
        return _out(NOT_EVALUABLE, "no bench input")
    wt = _witness_table(comp)
    parts = []
    st = comp[(comp["design"] == "witnesses") & (comp["bearer_mode"] == "system")]
    for fam in FAMILIES:
        if fam not in protocols:
            continue
        proto = protocols[fam][0]
        have = anchored(proto)
        if "SRPI" not in have or not _switch_ok(man_ok, fam, "e"):
            parts.append(
                _out(
                    NOT_EVALUABLE,
                    "no SRPI anchor or failed manipulation " "check",
                    family=fam,
                )
            )
            continue
        w = st[
            (st["family"] == fam)
            & (st["witness_id"] == "W_SRPI_no_efference")
            & (st["principle"] == "SRPI")
        ]
        if w.empty:
            parts.append(_out(NOT_EVALUABLE, "no runs", family=fam))
            continue
        absent = float((w["status"] == "ABSENT").mean())
        not_present = float((w["status"] != "PRESENT").mean())
        parts.append(
            _out(
                SUPPORTED if absent >= SEED_FRACTION else FALSIFIED,
                "SRPI ABSENT in >= 80% of seeds",
                family=fam,
                part="a",
                n=int(len(w)),
                absent_rate=absent,
                not_present_rate=not_present,
            )
        )
        for p in ("RAM", "NAS"):
            if p not in have:
                parts.append(
                    _out(NOT_EVALUABLE, f"{p} has no anchor", family=fam, part=f"b_{p}")
                )
                continue
            d = _paired_delta(wt, fam, "W_SRPI_no_efference", p)
            frac = (
                float(np.mean(d.abs() < proto.cutoff_for(p)[0])) if len(d) else np.nan
            )
            parts.append(
                _out(
                    (
                        SUPPORTED
                        if (len(d) and frac >= SEED_FRACTION)
                        else FALSIFIED if len(d) else NOT_EVALUABLE
                    ),
                    f"|c_{p}(lesion) - c_{p}(nominal)| < z in >= 80%",
                    family=fam,
                    part=f"b_{p}",
                    n=int(len(d)),
                    fraction=frac,
                )
            )
    parts.append(
        _out(
            NOT_EVALUABLE,
            "legacy SRPI comparison not run by the bench",
            part="c_legacy",
        )
    )
    return _out(_combine(parts), "parts a and b per family", parts=parts)


# ---------------------------------------------------------------------------
# HC9 rule audit
# ---------------------------------------------------------------------------
def hc9(audit_dir, alpha, nval=()):
    if not audit_dir:
        return _out(NOT_EVALUABLE, "no audit input")
    dec = pd.read_csv(Path(audit_dir) / "audit_decisions.csv")
    d0 = dec[dec["label_noise"] == 0.0]
    parts = []

    def is_pos(c):
        return str(c) in ("all_present", "witness:PC_nominal")

    def is_single(c):
        c = str(c)
        if c.startswith("single_deficit:"):
            return c.split(":", 1)[1] in set(nval) if nval else True
        if c.startswith("witness:W_"):
            w = c.split(":", 1)[1]
            return WITNESS_TARGET.get(w) in set(nval) if nval else w in WITNESS_TARGET
        return False

    def _rate(rule, scen, pred, verdict):
        sub = d0[
            (d0["rule"] == rule) & (d0["scenario"] == scen) & d0["class"].map(pred)
        ]
        return (
            float((sub["decision"] == verdict).mean()) if len(sub) else np.nan,
            int(len(sub)),
        )

    r_dcm, n1 = _rate("dcm_naive_bayes", "RAM+SRPI_missing", is_pos, "MPC_CONSISTENT")
    r_imp, _ = _rate("impact_c", "RAM+SRPI_missing", is_pos, "MPC_CONSISTENT")
    r_und, _ = _rate("impact_c", "RAM+SRPI_missing", is_pos, "UNDETERMINED")
    ok = bool(n1 and r_dcm > 0.5 and r_imp == 0.0)
    parts.append(
        _out(
            SUPPORTED if ok else (FALSIFIED if n1 else NOT_EVALUABLE),
            "RAM+SRPI missing: DCM-like skip-missing rule MPC_CONSISTENT > 0.5 "
            "on positives; impact_c never MPC_CONSISTENT",
            part="a",
            dcm_rate=r_dcm,
            impact_consistent=r_imp,
            impact_undetermined=r_und,
            n=n1,
        )
    )
    rivals = [
        "union",
        "count_1",
        "count_2",
        "count_3",
        "count_4",
        "arithmetic_mean",
        "geometric_mean_uncapped",
    ]
    fooled = {}
    none = d0[(d0["scenario"] == "none") & d0["class"].map(is_single)]
    for rule in rivals:
        sub = none[none["rule"] == rule]
        if sub.empty:
            continue
        per_class = sub.groupby("class")["decision"].apply(
            lambda s: float((s == "MPC_CONSISTENT").mean())
        )
        fooled[rule] = float(per_class.max())
    ok_b = bool(fooled) and all(v > alpha for v in fooled.values())
    parts.append(
        _out(
            SUPPORTED if ok_b else (FALSIFIED if fooled else NOT_EVALUABLE),
            "each rival rule MPC_CONSISTENT on some single-deficit class at a "
            "rate > alpha",
            part="b",
            max_rate_by_rule=fooled,
        )
    )
    sub = none[none["rule"] == "impact_c"]
    per_class = (
        sub.groupby("class")["decision"].apply(
            lambda s: float((s == "MPC_CONSISTENT").mean())
        )
        if len(sub)
        else pd.Series(dtype=float)
    )
    worst = float(per_class.max()) if len(per_class) else np.nan
    parts.append(
        _out(
            (
                SUPPORTED
                if (len(per_class) and worst <= alpha + BAND)
                else FALSIFIED if len(per_class) else NOT_EVALUABLE
            ),
            "impact_c MPC_CONSISTENT on each single-deficit class <= alpha + 0.02",
            part="c",
            worst_rate=worst,
            per_class=per_class.to_dict(),
        )
    )
    pos = d0[
        (d0["scenario"] == "none")
        & (d0["rule"] == "impact_c")
        & d0["class"].map(is_pos)
    ]
    parts.append(
        _out(
            NOT_EVALUABLE,
            "sensitivity cost (reported)",
            part="d",
            impact_consistent_on_positives=(
                float((pos["decision"] == "MPC_CONSISTENT").mean())
                if len(pos)
                else None
            ),
            n=int(len(pos)),
        )
    )
    return _out(
        _combine(parts), "parts a-c; the sensitivity cost (d) is reported", parts=parts
    )


# ---------------------------------------------------------------------------
# HC10 verdict stability (rescaled jackknife replicates)
# ---------------------------------------------------------------------------
def hc10(records, protocols):
    from impact_pipeline.bench.export import evidence_verdict

    n_clear, flips, rows = 0, [], []
    for r in records:
        fam = str(r.get("family"))
        if fam not in protocols or r.get("design") not in ("witnesses", "factorial"):
            continue
        pval = protocols[fam][1]
        res = {
            "components": r["components"],
            "estimator_modes": r.get("estimator_modes") or {},
        }
        meta = {"substrate": (r.get("system") or {}).get("substrate")}
        v = evidence_verdict(res, meta, protocol=pval)
        clear = True
        for p in pval.necessity_set:
            c, se = v["c"].get(p), v["c_se"].get(p)
            if c is None or se is None or not (se > 0):
                clear = False
                break
            z, d = pval.cutoff_for(p)
            if (
                abs(c - z) <= STABILITY_MARGIN * se
                or abs(c - d) <= STABILITY_MARGIN * se
            ):
                clear = False
                break
        if not clear:
            continue
        reps = {
            p: np.asarray(
                [
                    np.nan if x is None else x
                    for x in (r["components"][p].get("se_replicates") or [])
                ],
                float,
            )
            for p in pval.necessity_set
        }
        g = min(len(x) for x in reps.values()) if reps else 0
        if g < 2:
            continue
        n_clear += 1
        f = []
        for k in range(g):
            comps = {}
            for p, c in r["components"].items():
                comps[p] = dict(c)
                if p in reps:
                    x = reps[p]
                    fin = np.isfinite(x)
                    if not fin[k] or fin.sum() < 2:
                        continue
                    star = c["estimate"] + math.sqrt(fin.sum() - 1) * (
                        x[k] - x[fin].mean()
                    )
                    comps[p]["estimate"] = float(star)
            vk = evidence_verdict(
                {"components": comps, "estimator_modes": res["estimator_modes"]},
                meta,
                protocol=pval,
            )
            f.append(vk["verdict"] != v["verdict"])
        flips.append(float(np.mean(f)))
        rows.append(
            {"task_id": r["task_id"], "verdict": v["verdict"], "flip_rate": flips[-1]}
        )
    if n_clear < MIN_CLEAR_RUNS:
        return _out(
            INDETERMINATE if n_clear else NOT_EVALUABLE,
            f"fewer than {MIN_CLEAR_RUNS} clear runs",
            n_clear=n_clear,
            mean_flip_rate=float(np.mean(flips)) if flips else None,
        )
    rate = float(np.mean(flips))
    return _out(
        SUPPORTED if rate < FLIP_MAX else FALSIFIED,
        "mean verdict-flip rate among clear runs < 0.10",
        n_clear=n_clear,
        mean_flip_rate=rate,
        runs=rows,
    )


# ---------------------------------------------------------------------------
def load_protocols(allow_development):
    proto = ev.Protocol.from_json(PROTOCOL)
    proto_val = ev.Protocol.from_json(PROTOCOL_ANCHORED)
    if not A.same_except_reference(
        proto, proto_val.replace(necessity_set=proto.necessity_set)
    ):
        raise ValueError(
            "the anchored protocol must equal the bench protocol "
            "except for its necessity set and name"
        )
    if not allow_development:
        for key, pr in (("mpc_bench_v1", proto), ("mpc_bench_v1_anchored", proto_val)):
            want = EXPECTED_HASHES.get(key)
            if want is None or pr.hash != want:
                raise ValueError(
                    f"{key}: protocol hash {pr.hash} is not the frozen hash {want}"
                )
    return proto, proto_val


def run(
    out_dir,
    *,
    bench_a=(),
    bench_c=(),
    reference_c=(),
    adversarial=(),
    null_calibration=None,
    iim_validation=None,
    audit=None,
    manipulation=(),
    freeze_tag=FREEZE_TAG,
    allow_development=False,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proto, proto_val = load_protocols(allow_development)
    records = (
        A.read_records(list(bench_a) + list(bench_c) + list(adversarial))
        if (list(bench_a) + list(bench_c) + list(adversarial))
        else []
    )
    ref_c = A.read_records(list(reference_c)) if reference_c else []
    for recs in (records, ref_c):
        if recs:
            check_records(recs, allow_development, freeze_tag)
            check_cut_mode(recs, proto)
    protocols, fam_info = family_protocols(proto, proto_val, ref_c, allow_development)
    comp, verd = assess_family(records, protocols)
    if len(comp):
        comp.drop(columns=["se_replicates"], errors="ignore").to_csv(
            out / "components.csv", index=False
        )
        verd.to_csv(out / "verdicts.csv", index=False)
    man = None
    if manipulation:
        frames = [
            pd.read_csv(Path(d) / "manipulation_checks.csv") for d in manipulation
        ]
        man = pd.concat(frames, ignore_index=True)
    man_res, man_ok = manipulation_status(man)
    alpha = float(proto.alpha)
    results = {
        "M_manipulation": man_res,
        "HC1": hc1(null_calibration, comp, proto, alpha),
        "HC2": hc2(iim_validation),
        "HC3": hc3(comp, protocols, man_ok),
        "HC4": hc4(comp, protocols, man_ok),
        "HC5": hc5(verd, protocols, alpha),
        "HC6": hc6(comp, protocols, alpha),
        "HC7": hc7(records, protocols),
        "HC8": hc8(comp, protocols, man_ok),
        "HC9": hc9(audit, alpha, proto_val.necessity_set),
        "HC10": hc10(records, protocols),
    }
    from impact_pipeline.provenance import collect_code_version

    summary = {
        "version": HYPOTHESES_VERSION,
        "status": (
            "DEVELOPMENT - NOT A RESULT" if allow_development else "confirmatory"
        ),
        "freeze_tag": freeze_tag,
        "protocol_hash": proto.hash,
        "anchored_protocol_hash": proto_val.hash,
        "anchored_necessity_set": list(proto_val.necessity_set),
        "family_protocols": fam_info,
        "n_records": len(records),
        "outcomes": {k: v["outcome"] for k, v in results.items()},
        "results": results,
        "code_version": collect_code_version(REPO_ROOT),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "hypotheses.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"hypothesis": k, "outcome": v["outcome"], "reason": v["reason"]}
            for k, v in results.items()
        ]
    ).to_csv(out / "hypotheses_table.csv", index=False)
    return summary


def _dirs(text):
    return [s.strip() for s in str(text or "").split(",") if s.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench-a", default="")
    ap.add_argument("--bench-c", default="")
    ap.add_argument("--reference-c", default="")
    ap.add_argument("--adversarial", default="")
    ap.add_argument("--null-calibration", default=None)
    ap.add_argument("--iim-validation", default=None)
    ap.add_argument("--audit", default=None)
    ap.add_argument("--manipulation", default="")
    ap.add_argument("--freeze-tag", default=FREEZE_TAG)
    ap.add_argument(
        "--development",
        action="store_true",
        help="evaluate development runs (testing only; NOT A RESULT)",
    )
    args = ap.parse_args(argv)
    s = run(
        args.out,
        bench_a=_dirs(args.bench_a),
        bench_c=_dirs(args.bench_c),
        reference_c=_dirs(args.reference_c),
        adversarial=_dirs(args.adversarial),
        null_calibration=args.null_calibration,
        iim_validation=args.iim_validation,
        audit=args.audit,
        manipulation=_dirs(args.manipulation),
        freeze_tag=args.freeze_tag,
        allow_development=args.development,
    )
    print(json.dumps(s["outcomes"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
