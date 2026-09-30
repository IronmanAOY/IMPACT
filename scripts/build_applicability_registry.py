#!/usr/bin/env python
"""
Build the applicability registry from the confirmatory MPC-Bench results.

The registry (``evidence.ApplicabilityRegistry``, schema
``impact-mpc-registry/2``) lists the estimator versions validated for a
principle on a substrate, grain and regime. This script applies the entry
criteria preregistered in ``docs/preregistration/MPC_BENCH_PREREGISTRATION.md``
(section 4, as operationalised by
``scripts/calibrate_bench.py::entry_criteria``) to the confirmatory runs made
on the frozen code (tag ``mpcbench-freeze-v1``), unchanged:

(0) a valid reference anchor (the frozen family-A anchor; family C's anchor
    from its reference block 19000-19019, same anchor rule);
(a) false-PRESENT rate <= alpha + 0.02 on every statistical null class (the
    bench null witnesses of the family and the ``null_calibration.py`` grid
    re-anchored on the family's reference);
(b1) Spearman rho of ``c`` with the own knob's dose > 0 at p < alpha (sweeps);
(b2) PRESENT rate <= alpha + 0.02 on the systems without the mechanism (own
     sweep at its lowest dose, own single-deficit witness, ``O_inert``);
(c) in the paired witness contrasts the own switch changes ``c`` more than
    any other switch (declared structural dependencies excluded).

Only (principle, estimator, substrate) combinations that meet every
criterion become ``entries`` (status ``validated``); every other combination
that the confirmatory runs could speak to is listed under ``excluded`` with
the criteria it fails and the evidence. Substrates without a mechanism-switch
design (family B exact TPMs, the whole-brain model and its forward models)
cannot meet (0)-(c) as preregistered and are listed with their descriptive
evidence. Nothing here changes an estimator, a threshold or a protocol, and
no hypothesis outcome depends on the registry.

Example::

    python scripts/build_applicability_registry.py \\
        --results outputs/paper1_mpcbench \\
        --out protocols/applicability_registry_v1.json \\
        --evidence-dir outputs/paper1_mpcbench/registry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for _p in (SRC_ROOT, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import calibrate_bench as CB  # noqa: E402

from impact_pipeline import evidence as ev  # noqa: E402
from impact_pipeline.bench import analysis as A  # noqa: E402
from impact_pipeline.mpc_metrics import ESTIMATOR_VERSIONS  # noqa: E402
from impact_pipeline.necessity import clopper_pearson  # noqa: E402

BUILDER_VERSION = "mpc-applicability-registry-builder/1.0.0"
REGISTRY_VERSION = "1.0.0"
FREEZE_TAG = "mpcbench-freeze-v1"
PROTOCOL = REPO_ROOT / "protocols" / "mpc_bench_v1.json"
REFERENCE_SUMMARY = REPO_ROOT / "protocols" / "mpc_bench_v1_reference_summary.json"
FROZEN_PROTOCOL_HASH = (
    "855f6a444b77030d33faa68fcb45e8576b931d2d681cf215d4dacdb57a6b2520"
)
FAMILY_SUBSTRATE = {"A": "synthetic_rate", "C": "stuart_landau"}
BAND = CB.BAND

# Result layout below --results (the confirmatory run plan's folders).
LAYOUT = {
    "bench": "c2/bench/{family}",
    "reference_C": "c2/reference_C",
    "null_calibration": "c1/null_calibration",
    "hypotheses_c1": "c1/hypotheses_c1/hypotheses.json",
    "hypotheses_c2": "c2/hypotheses/hypotheses.json",
    "hc2_bias_variance": "c1/supplementary/hc2_bias_variance.csv",
    "whole_brain_status": "c3/summary/whole_brain_status_summary.csv",
    "whole_brain_summary": "c3/summary/whole_brain_summary.json",
    "whole_brain_G_sweep": "c3/summary/whole_brain_G_sweep.csv",
    "adversarial_status": "c3/summary/adversarial_status_summary.csv",
}
BENCH_DESIGNS = ("witnesses", "sweep")
# Intended mechanism bits of the adversarial constructions (RAM, PDI, NAS,
# IIM, SRPI), as declared by bench.adversarial (reported descriptively).
ADVERSARIAL_BITS = {
    "parity_grid": "00010",
    "hypersynchrony": "00000",
    "common_driver": "00000",
    "reflex_arc": "00000",
    "random_label_self_other": "11110",
    "scrambled_feedback": "01111",
}
PRINCIPLE_INDEX = {p: i for i, p in enumerate(("RAM", "PDI", "NAS", "IIM", "SRPI"))}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path) -> str:
    """Repository-relative path (the committed registry carries no local
    absolute paths)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return p.name


def _git(*args):
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _cp_upper(k, n, alpha=0.05):
    return float(clopper_pearson(k, n, alpha, "upper")[1]) if n else None


def _mode(principle, options):
    """Estimator mode in ``evidence.estimator_id`` naming."""
    opts = dict(options or {})
    if principle == "RAM":
        return str(opts.get("update") or "feedback_magnitude")
    if principle == "IIM":
        return str(opts.get("cut_mode") or "bidirectional")
    return str(opts.get("mode") or "legacy")


def estimator_name(proto, principle):
    return f"compute_{principle}:{_mode(principle, proto.estimator_options(principle))}"


def check_records(records, label, tag_tree):
    """Refuse records that are not confirmatory runs of the frozen tree."""
    if not records:
        raise ValueError(f"{label}: no ok records")
    A.check_split(records, ("confirmatory",))
    shas = set()
    for r in records:
        code = (r.get("provenance") or {}).get("code_version") or {}
        if code.get("freeze_tag") != FREEZE_TAG or code.get("git_dirty") is not False:
            raise ValueError(f"{label}: {r.get('task_id')} is not a clean run of "
                             f"{FREEZE_TAG}")
        shas.add(code.get("git_sha"))
        proto_hash = (r.get("verdict") or {}).get("protocol_hash")
        if proto_hash is not None and proto_hash != FROZEN_PROTOCOL_HASH:
            raise ValueError(f"{label}: {r.get('task_id')} used protocol {proto_hash}")
    for sha in shas:
        tree = _git("rev-parse", f"{sha}^{{tree}}")
        if tree != tag_tree:
            raise ValueError(
                f"{label}: commit {sha} does not have the source tree of {FREEZE_TAG}"
            )
    return sorted(shas)


def _results_files(dirs):
    out = []
    for d in dirs:
        for p in sorted(Path(d).rglob("results.jsonl")):
            out.append({"path": _rel(p), "sha256": _sha256(p)})
    return out


# ---------------------------------------------------------------------------
# families A and C: the preregistered entry criteria
# ---------------------------------------------------------------------------
def family_evidence(results, family, proto, tag_tree, nc_replicates):
    """Construct-scale components, null rates, dose-response, contrasts and
    off-mechanism rates of one family, and the entry-criteria table."""
    root = Path(results)
    if family == "A":
        reference = proto.reference
        anchor_summary = json.loads(REFERENCE_SUMMARY.read_text(encoding="utf-8"))
        fam_proto = proto
        ref_info = {
            "source": "frozen protocol protocols/mpc_bench_v1.json "
                      "(family-A PC_nominal, development seeds 900-919)",
            "reference_summary": str(REFERENCE_SUMMARY.relative_to(REPO_ROOT)),
        }
        ref_shas = []
    else:
        ref_recs = A.read_records(root / LAYOUT["reference_C"])
        ref_shas = check_records(ref_recs, "reference_C", tag_tree)
        reference, anchor_summary = A.family_reference(
            ref_recs, alpha=proto.alpha, allow_confirmatory=True
        )
        fam_proto = A.protocol_for_family(proto, reference)
        ref_info = {
            "source": "family-C PC_nominal, confirmatory reference seeds "
                      "19000-19019 (n=20), frozen anchor rule",
            "n_records": len(ref_recs),
        }
    dirs = [root / LAYOUT["bench"].format(family=family) / d for d in BENCH_DESIGNS]
    records = A.read_records(dirs)
    shas = check_records(records, f"bench {family}", tag_tree)
    comp = A.assess_components(A.components_raw(records), reference, fam_proto)
    wit = comp[comp["design"] == "witnesses"]
    null_parts = [
        wit[wit["witness_id"].isin(CB.NULL_WITNESSES)].assign(
            null_class=lambda d: "witness:" + d["witness_id"].astype(str)
        )
    ]
    nc_raw = nc_replicates[
        ["kind", "n_time", "n_nodes", "replicate", "principle", "estimate",
         "null_mean", "null_sd", "n_null", "se", "se_n", "defined"]
    ].copy()
    nc_raw["se"] = nc_raw["se"].astype(float)
    nca = A.assess_components(nc_raw, reference, fam_proto)
    nca["null_class"] = (
        "null_calibration:" + nca["kind"].astype(str) + ":T"
        + nca["n_time"].astype(str) + ":N" + nca["n_nodes"].astype(str)
    )
    null_parts.append(nca)
    nulls = pd.concat(null_parts, ignore_index=True)
    nulls["fp"] = nulls["status"] == A.STATUS_PRESENT
    null_rates = (
        nulls.groupby(["null_class", "principle"])
        .agg(n=("fp", "size"), k=("fp", "sum"), false_present_rate=("fp", "mean"))
        .reset_index()
    )
    null_rates["k"] = null_rates["k"].astype(int)
    levels, slopes = A.dose_response(comp)
    contrasts = A.witness_contrasts(comp)
    off = CB.off_dose_rates(comp)
    entry = CB.entry_criteria(anchor_summary, null_rates, slopes, fam_proto,
                              contrasts, off)
    entry.insert(0, "family", family)
    entry.insert(1, "substrate", FAMILY_SUBSTRATE[family])
    # counts behind the off-mechanism rate (criterion b2)
    off_counts = {}
    for knob, p in A.KNOB_TARGET.items():
        sub = comp[comp["principle"] == p]
        sw = sub[(sub["design"] == "sweep") & (sub["sweep_knob"] == knob)]
        if len(sw):
            lev = sw["sweep_level"].astype(float)
            sw = sw[lev == lev.min()]
        wt = sub[(sub["design"] == "witnesses") & (sub["bearer_mode"] == "system")
                 & sub["witness_id"].isin([A.SWITCH_WITNESS[knob], "O_inert"])]
        both = pd.concat([sw, wt])
        k_on = int((both["status"] == A.STATUS_PRESENT).sum())
        off_counts[p] = (k_on, int(len(both)))
    return {
        "family": family,
        "protocol": fam_proto,
        "reference": reference,
        "anchor_summary": anchor_summary,
        "reference_info": ref_info,
        "records": records,
        "git_shas": sorted(set(shas) | set(ref_shas)),
        "components": comp,
        "null_rates": null_rates,
        "levels": levels,
        "slopes": slopes,
        "contrasts": contrasts,
        "off_rates": off,
        "off_counts": off_counts,
        "entry": entry,
        "inputs": _results_files(dirs)
        + (_results_files([root / LAYOUT["reference_C"]]) if family == "C" else []),
        "nc_reassessed": nca,
    }


def criteria_records(records):
    """The records the entry criteria read: sweeps and witnesses without the
    patchwork (which enters no criterion and uses another IIM grain)."""
    return [r for r in records if r.get("witness_id") != "PW_patchwork"]


def _regime(records, principle, proto):
    """Regime envelope of the validation systems (what the queries of the
    evidence layer carry: ``n_time``, ``n_nodes``, ``tr``, ``bins``)."""
    records = criteria_records(records)
    t = [int(r["system"]["n_time"]) for r in records]
    dts = sorted({float(r["system"]["dt"]) for r in records})
    if principle == "IIM":
        opts = [r.get("estimator_modes", {}).get("IIM", {}) for r in records]
        nodes = sorted({int(o.get("n_macro_nodes")) for o in opts
                        if o.get("n_macro_nodes") is not None})
        bins = sorted({int(o.get("bins")) for o in opts if o.get("bins") is not None})
        regime = {"T_min": min(t), "nodes_min": min(nodes), "nodes_max": max(nodes)}
        regime["bins"] = bins[0] if len(bins) == 1 else bins
    else:
        nodes = sorted({int(r["system"]["n_nodes"]) for r in records})
        regime = {"T_min": min(t), "nodes_min": min(nodes), "nodes_max": max(nodes)}
    regime["fs_or_tr"] = {"min": dts[0], "max": dts[-1]}
    return regime


def _slope_row(fe, principle):
    knob = next(k for k, t in A.KNOB_TARGET.items() if t == principle)
    sl = fe["slopes"]
    row = sl[(sl["knob"] == knob) & (sl["principle"] == principle)]
    return knob, (row.iloc[0] if len(row) else None)


def _level_means(fe, knob, principle):
    lev = fe["levels"]
    sub = lev[(lev["sweep_knob"] == knob) & (lev["principle"] == principle)]
    sub = sub.sort_values("sweep_level")
    return [
        {"dose": float(r.sweep_level), "n": int(r.n), "c_mean": _num(r.c_mean),
         "present_rate": _num(r.present_rate), "absent_rate": _num(r.absent_rate)}
        for r in sub.itertuples(index=False)
    ]


def _contrast_summary(fe, principle):
    ct = fe["contrasts"]
    sub = ct[ct["principle"] == principle]
    return [
        {"switch": r.switch, "own": bool(r.own),
         "declared_dependency": bool(r.declared_dependency), "n": int(r.n),
         "median_delta_c": _num(r.median_delta_c)}
        for r in sub.itertuples(index=False)
    ]


def criteria_record(fe, principle):
    """Per-criterion results with the evidence (JSON-ready)."""
    e = fe["entry"].set_index("principle").loc[principle]
    alpha = float(fe["protocol"].alpha)
    nr = fe["null_rates"][fe["null_rates"]["principle"] == principle]
    k_null, n_null = int(nr["k"].sum()), int(nr["n"].sum())
    worst = nr.sort_values("false_present_rate", ascending=False).head(1)
    knob, srow = _slope_row(fe, principle)
    k_off, n_off = fe["off_counts"][principle]
    own_d = _num(e["own_switch_median_abs_delta_c"])
    other_d = _num(e["max_other_switch_median_abs_delta_c"])
    ratio = (other_d / own_d) if (own_d and other_d is not None) else None
    per = (fe["anchor_summary"].get("per_principle") or {}).get(principle, {})
    means = _level_means(fe, knob, principle)
    cm = [m["c_mean"] for m in means]
    nondecr = (
        bool(all(b >= a for a, b in zip(cm, cm[1:])))
        if cm and all(v is not None for v in cm) else None
    )
    return {
        "anchor_valid": bool(e["anchor_valid"]),
        "anchor": {"mean_excess": _num(per.get("mean")), "se": _num(per.get("se")),
                   "one_sided_lower_bound": _num(per.get("lower_bound")),
                   "n": per.get("n")},
        "null_ok": bool(e["null_ok"]),
        "null_false_present_max_rate": _num(e["null_false_present_max"]),
        "null_false_present_worst_class": (
            None if worst.empty or float(worst["false_present_rate"].iloc[0]) == 0
            else str(worst["null_class"].iloc[0])
        ),
        "null_false_present_pooled": {"k": k_null, "n": n_null,
                                      "rate": (k_null / n_null) if n_null else None,
                                      "cp_upper_95": _cp_upper(k_null, n_null)},
        "null_classes": int(len(nr)),
        "own_knob": knob,
        "monotone_ok": bool(e["monotone_ok"]),
        "own_knob_spearman_rho": _num(e["own_knob_spearman_rho"]),
        "own_knob_spearman_p": _num(e["own_knob_spearman_p"]),
        "own_knob_ols_slope_c_per_unit_dose": (
            None if srow is None else _num(srow["slope_c_per_unit_dose"])
        ),
        "own_knob_level_means_nondecreasing": nondecr,
        "off_dose_specific_ok": bool(e["off_dose_specific_ok"]),
        "off_mechanism_present": {"k": k_off, "n": n_off,
                                  "rate": (k_off / n_off) if n_off else None,
                                  "cp_upper_95": _cp_upper(k_off, n_off)},
        "crosstalk_ok": bool(e["crosstalk_ok"]),
        "own_switch_median_abs_delta_c": own_d,
        "max_other_switch_median_abs_delta_c": other_d,
        "cross_talk_ratio": ratio,
        "witness_contrasts": _contrast_summary(fe, principle),
        "own_knob_levels": means,
        "validated": bool(e["validated"]),
        "alpha": alpha,
    }


def failed_criteria(rec):
    fails = []
    if not rec["anchor_valid"]:
        # without an anchor c is undefined: (b1) and (c) cannot be computed,
        # and (a) and (b2) hold only because the status is never PRESENT
        return [
            "(0) no valid reference anchor",
            "(b1), (c) not evaluable: c is undefined without an anchor "
            "((a) and (b2) hold trivially: the status is UNDEFINED everywhere)",
        ]
    if not rec["null_ok"]:
        fails.append("(a) null false-PRESENT rate above alpha + 0.02")
    if not rec["monotone_ok"]:
        fails.append("(b1) no dose-response of c to the own knob "
                     "(Spearman rho > 0 at p < alpha)")
    if not rec["off_dose_specific_ok"]:
        fails.append("(b2) PRESENT without the mechanism above alpha + 0.02")
    if not rec["crosstalk_ok"]:
        fails.append("(c) another switch changes c more than the own switch")
    return fails


# ---------------------------------------------------------------------------
# substrates without the mechanism-switch design (descriptive evidence)
# ---------------------------------------------------------------------------
def family_b_evidence(results):
    path = Path(results) / LAYOUT["hypotheses_c1"]
    hyp = json.loads(path.read_text(encoding="utf-8"))
    hc2 = hyp["results"]["HC2"]
    parts = hc2["parts"]
    return {
        "source": {"path": _rel(path), "sha256": _sha256(path)},
        "HC2_outcome": hc2["outcome"],
        "part_outcomes": {k: v["outcome"] for k, v in parts.items()},
        "independent_units_rate_z_above_1.645": [
            {"cut_mode": r["cut_mode"], "n_time": r["n_time"], "n": r["n"],
             "rate": r["rate_z_above"], "median_z": _num(r["median_z"])}
            for r in parts["c_independent_calibrated"]["rows"]
        ],
        "median_abs_error_not_decreasing": [
            {"system": r["system"], "cut_mode": r["cut_mode"],
             "median_abs_error": r["median_abs_error"]}
            for r in parts["b_bias_shrinks"]["rows"] if not r["monotone"]
        ],
        "ring_coupling_spearman": [
            {"cut_mode": r["cut_mode"], "rho": _num(r["rho"]), "p": _num(r["p"]),
             "n": r["n"]}
            for r in parts["e_coupling_monotone"]["rows"]
        ],
        "feedforward_star_rate_z_above_1.645": [
            {"cut_mode": r["cut_mode"], "n_time": r["n_time"], "n": r["n"],
             "rate": r["rate_z_above"]}
            for r in parts["d_feedforward"]["rows"]
        ],
    }


def whole_brain_evidence(results):
    root = Path(results)
    status = pd.read_csv(root / LAYOUT["whole_brain_status"])
    summ = json.loads((root / LAYOUT["whole_brain_summary"]).read_text())
    gsw = pd.read_csv(root / LAYOUT["whole_brain_G_sweep"])
    st = status[status["protocol"] == "P_star"]
    out = {"sources": [
        {"path": _rel(root / LAYOUT[k]), "sha256": _sha256(root / LAYOUT[k])}
        for k in ("whole_brain_status", "whole_brain_summary", "whole_brain_G_sweep")
    ]}
    for obs in ("source", "eeg"):
        rows = {}
        for p in ("NAS", "IIM"):
            g0 = st[(st["observation"] == obs) & (st["cell_id"] == "G0")
                    & (st["principle"] == p)]
            gs = gsw[(gsw["observation"] == obs) & (gsw["quantity"] == f"{p}_c")]
            rows[p] = {
                "present_at_G0": (
                    None if g0.empty
                    else f"{int(g0['n_PRESENT'].iloc[0])}/{int(g0['n'].iloc[0])}"
                ),
                "c_median_at_G0": None if g0.empty else _num(g0["c_median"].iloc[0]),
                "spearman_rho_c_vs_G": (None if gs.empty else
                                        _num(gs["spearman_rho"].iloc[0])),
                "spearman_p": None if gs.empty else _num(gs["p_value"].iloc[0]),
            }
        out[obs] = rows
    att = summ.get("attempts", {})
    out["bold"] = {
        "tasks_planned": 140,
        "tasks_final_error": int(sum(1 for t in att.get("final_error_task_ids", [])
                                     if "-bold-" in t)),
        "error": "compute_NAS invalid band (0.5, 4.0) for Nyquist 0.25 "
                 "(TR = 2 s); run_task scores all estimators in one try block",
    }
    return out


def adversarial_evidence(results, principle):
    """PRESENT rates of ``principle`` on the adversarial constructions
    (synthetic_rate unless stated) with the mechanism absent by design;
    descriptive, not an entry criterion."""
    path = Path(results) / LAYOUT["adversarial_status"]
    st = pd.read_csv(path)
    st = st[(st["protocol"] == "P_star") & (st["principle"] == principle)]
    rows = []
    for r in st.itertuples(index=False):
        bits = ADVERSARIAL_BITS.get(r.cell_id)
        on = None if bits is None else bits[PRINCIPLE_INDEX[principle]] == "1"
        rows.append({"construction": r.cell_id, "mechanism_on_by_design": on,
                     "present": f"{int(r.n_PRESENT)}/{int(r.n)}",
                     "absent": f"{int(r.n_ABSENT)}/{int(r.n)}"})
    return {"source": {"path": _rel(path), "sha256": _sha256(path)}, "rows": rows}


def hypothesis_context(results):
    """Related preregistered outcomes of the frozen evaluator (families A and
    C: HC3 per switch, HC4 per witness, HC6 per principle, HC8), reported
    with the entries; they do not enter the entry criteria."""
    path = Path(results) / LAYOUT["hypotheses_c2"]
    if not path.is_file():
        return {}
    hyp = json.loads(path.read_text(encoding="utf-8"))
    res = hyp["results"]
    ctx = {"source": {"path": _rel(path), "sha256": _sha256(path)}}
    ctx["HC3"] = {
        f"{p.get('family')}:{p.get('switch')}->{p.get('target')}": {
            "outcome": p.get("outcome"),
            "target_fraction": p.get("target_fraction"),
            "target_median_delta_c": p.get("target_median_delta_c"),
        }
        for p in res.get("HC3", {}).get("parts", []) if isinstance(p, dict)
    }
    ctx["HC4"] = {
        f"{p.get('family')}:{p.get('witness')}": {
            "outcome": p.get("outcome"), "pass_fraction": p.get("fraction"),
            "target_absent_rate": p.get("target_absent_rate"),
            "target": p.get("target"), "n_seeds": p.get("n_seeds"),
        }
        for p in res.get("HC4", {}).get("parts", []) if isinstance(p, dict)
        and p.get("witness")
    }
    hc6 = {}
    bound = float(res.get("HC6", {}).get("bound", 0.07))
    for c in res.get("HC6", {}).get("cells", []):
        key = f"{c['family']}:{c['principle']}"
        d = hc6.setdefault(key, {"off_cells": 0, "point_rate_above_bound": 0,
                                 "familywise_lower_above_bound": 0, "k": 0, "n": 0})
        d["off_cells"] += 1
        d["point_rate_above_bound"] += int(c["rate"] > bound)
        d["familywise_lower_above_bound"] += int(c["lower_familywise"] > bound)
        d["k"] += int(c["k"])
        d["n"] += int(c["n"])
    ctx["HC6"] = hc6
    ctx["HC8"] = [p for p in res.get("HC8", {}).get("parts", []) if isinstance(p, dict)]
    ctx["note"] = ("HC1, HC2 and HC9 of this evaluator file are out of its scope "
                   "(see c1/ and c3/); they are not copied here.")
    return ctx


def related_outcomes(ctx, family, principle):
    """The preregistered outcomes that bear on one (family, principle)."""
    if not ctx:
        return {}
    knob = next(k for k, t in A.KNOB_TARGET.items() if t == principle)
    out = {}
    hc3 = ctx.get("HC3", {}).get(f"{family}:{knob}->{principle}")
    if hc3:
        out["HC3_own_switch"] = hc3
    wit = A.SWITCH_WITNESS[knob]
    hc4 = {
        k: v for k, v in ctx.get("HC4", {}).items()
        if k.startswith(f"{family}:")
        and (v.get("target") == principle or k.endswith(wit))
    }
    if hc4:
        out["HC4_witnesses"] = hc4
    hc6 = ctx.get("HC6", {}).get(f"{family}:{principle}")
    if hc6:
        out["HC6_factorial_off_cells"] = hc6
    if principle == "SRPI":
        out["HC8"] = [p for p in ctx.get("HC8", []) if p.get("family") == family]
    return out


def _fmt_note(family, principle, rel):
    """One-sentence summary of the related outcomes (data-driven)."""
    bits = []
    hc3 = rel.get("HC3_own_switch")
    if hc3:
        frac = hc3.get("target_fraction")
        bits.append(
            f"HC3 own switch {hc3['outcome']}"
            + (f" (target drop >= 0.5 in {frac:.0%} of seeds, 80% required)"
               if isinstance(frac, (int, float)) else ""))
    for wid, v in sorted(rel.get("HC4_witnesses", {}).items()):
        if v.get("target_absent_rate") is not None:
            bits.append(
                f"HC4 {wid.split(':', 1)[1]} {v['outcome']} (target ABSENT in "
                f"{v['target_absent_rate']:.0%} of seeds)")
    hc6 = rel.get("HC6_factorial_off_cells")
    if hc6:
        bits.append(
            f"HC6: {hc6['point_rate_above_bound']} of {hc6['off_cells']} factorial "
            f"cells without the mechanism have a point PRESENT rate above 0.07, "
            f"{hc6['familywise_lower_above_bound']} a family-wise lower bound "
            f"above 0.07 (pooled {hc6['k']}/{hc6['n']})")
    for part in rel.get("HC8", []):
        if part.get("part") == "a":
            bits.append(
                f"HC8(a) {part['outcome']} (efference lesion ABSENT in "
                f"{part['absent_rate']:.0%} of seeds, not PRESENT in "
                f"{part['not_present_rate']:.0%})")
    if not bits:
        return None
    return ("Related preregistered outcomes (not entry criteria; report them "
            "with the entry): " + "; ".join(bits) + ".")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def build(results, out_path, evidence_dir=None) -> dict:
    results = Path(results)
    proto = ev.Protocol.from_json(PROTOCOL)
    if proto.hash != FROZEN_PROTOCOL_HASH:
        raise ValueError(f"protocols/mpc_bench_v1.json hash {proto.hash} is not "
                         "the frozen hash")
    tag_commit = _git("rev-parse", f"{FREEZE_TAG}^{{commit}}")
    tag_tree = _git("rev-parse", f"{FREEZE_TAG}^{{tree}}")
    if not tag_tree:
        raise RuntimeError(f"freeze tag {FREEZE_TAG} not found")
    nc_path = results / LAYOUT["null_calibration"] / "null_calibration_replicates.csv"
    nc_json = results / LAYOUT["null_calibration"] / "null_calibration.json"
    nc_meta = json.loads(nc_json.read_text(encoding="utf-8"))
    if nc_meta.get("split") != "confirmatory" or nc_meta.get(
            "protocol_hash") != FROZEN_PROTOCOL_HASH:
        raise ValueError("null calibration is not the confirmatory frozen run")
    nc = pd.read_csv(nc_path).rename(columns={"null_kind": "kind"})

    fams = {f: family_evidence(results, f, proto, tag_tree, nc) for f in ("A", "C")}
    # the family-A re-assessment must reproduce the null-calibration statuses
    chk = fams["A"]["nc_reassessed"].merge(
        nc[["kind", "n_time", "n_nodes", "replicate", "principle", "status"]],
        on=["kind", "n_time", "n_nodes", "replicate", "principle"],
        suffixes=("", "_run"),
    )
    agree_a = float((chk["status"] == chk["status_run"]).mean())
    if agree_a != 1.0:
        raise ValueError(f"re-assessed null-calibration statuses disagree with the "
                         f"run ({agree_a:.4f})")

    ctx = hypothesis_context(results)
    criteria = {
        "alpha": float(proto.alpha),
        "false_present_tolerance": BAND,
        "two_sided": False,
        "min_recovery_slope": 0.0,
        "cross_talk_max": 1.0,
    }
    entries, excluded = [], []
    for fam, fe in fams.items():
        substrate = FAMILY_SUBSTRATE[fam]
        for p in ev.PRINCIPLES:
            rec = criteria_record(fe, p)
            name = estimator_name(fe["protocol"], p)
            version = ESTIMATOR_VERSIONS[p]
            if rec["validated"]:
                if not (rec["cross_talk_ratio"] is not None
                        and rec["cross_talk_ratio"] < 1.0):
                    raise AssertionError("validated entry without cross-talk ratio < 1")
                opts = fe["protocol"].estimator_options(p)
                grain = "*"
                if p == "IIM":
                    g = {r.get("estimator_modes", {}).get("IIM", {}).get("grain")
                         for r in criteria_records(fe["records"])}
                    grain = sorted(x for x in g if x)[0] if len(g) == 1 else sorted(g)
                evidence = {
                    "run_id": f"{FREEZE_TAG}:confirmatory:family-{fam}",
                    "null_false_present_rate": rec["null_false_present_max_rate"],
                    "null_false_present_rate_is": "maximum over the null classes",
                    "null_false_present_pooled": rec["null_false_present_pooled"],
                    "null_classes": rec["null_classes"],
                    "recovery_slope": rec["own_knob_ols_slope_c_per_unit_dose"],
                    "recovery_spearman_rho": rec["own_knob_spearman_rho"],
                    "recovery_spearman_p": rec["own_knob_spearman_p"],
                    "recovery_monotone": rec["monotone_ok"],
                    "recovery_monotone_is": "criterion (b1): Spearman rho > 0 at "
                                            "p < alpha",
                    "own_knob_level_means_nondecreasing":
                        rec["own_knob_level_means_nondecreasing"],
                    "off_mechanism_present": rec["off_mechanism_present"],
                    "cross_talk": rec["cross_talk_ratio"],
                    "cross_talk_is": "max |median paired delta c| of another switch "
                                     "/ own switch's (declared dependencies excluded)",
                    "anchor": rec["anchor"],
                    "own_knob": rec["own_knob"],
                    "witness_contrasts": rec["witness_contrasts"],
                    "records_git_sha": fe["git_shas"],
                }
                if p == "SRPI" and fam == "A":
                    evidence["adversarial_descriptive"] = adversarial_evidence(
                        results, p)
                entries.append({
                    "principle": p,
                    "estimator": name,
                    "version": version,
                    "substrate": substrate,
                    "grain": grain,
                    "regime": _regime(fe["records"], p, fe["protocol"]),
                    "evidence": evidence,
                    "status": "validated",
                    "conditions": {
                        "estimator_options": opts,
                        "null_family": fe["protocol"].null_families.get(p),
                        "null_surrogates": 19,
                        "sampling_se": "delete-a-group jackknife, G = 10",
                        "cutoffs": list(fe["protocol"].cutoff_for(p)),
                        "reference": (
                            "frozen family-A anchor (development seeds 900-919)"
                            if fam == "A" else
                            "family-C anchor (confirmatory reference seeds "
                            "19000-19019)"),
                    },
                    "related_preregistered_outcomes": related_outcomes(ctx, fam, p),
                    "note": _fmt_note(fam, p, related_outcomes(ctx, fam, p)),
                })
            else:
                excluded.append({
                    "principle": p,
                    "estimator": name,
                    "version": version,
                    "substrate": substrate,
                    "failed_criteria": failed_criteria(rec),
                    "criteria": {k: v for k, v in rec.items()
                                 if k not in ("witness_contrasts", "own_knob_levels")},
                    "related_preregistered_outcomes": related_outcomes(ctx, fam, p),
                    "note": _excluded_note(fam, p, rec),
                })
    excluded.extend(_non_switch_exclusions(results, proto))
    registry = {
        "schema": ev.REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "criteria": criteria,
        "entries": entries,
        "description": (
            "Applicability registry v1 of the MPC estimators, derived after the "
            "confirmatory MPC-Bench runs of the frozen code (tag "
            f"{FREEZE_TAG}) by applying the entry criteria preregistered in "
            "docs/preregistration/MPC_BENCH_PREREGISTRATION.md, section 4, "
            "unchanged. 'entries' holds only the combinations that meet every "
            "criterion; 'excluded' documents the others. No human EEG or fMRI "
            "use is covered: no estimator met the criteria on a forward-modelled "
            "substrate, so evidence on 'eeg'/'fmri' is ESTIMATOR_NOT_VALIDATED "
            "under this registry."
        ),
        "entry_criteria": {
            "0": "valid reference anchor (one-sided 95% t lower bound of the "
                 "positive control's mean excess > 0)",
            "a": "false-PRESENT rate <= alpha + 0.02 on every statistical null "
                 "class (point rate; one-sided, as preregistered)",
            "b1": "Spearman rho(c, own dose) > 0 at p < alpha (sweeps, 10 levels, "
                  "20 seeds)",
            "b2": "PRESENT rate <= alpha + 0.02 on the systems without the "
                  "mechanism (own sweep at dose 0, own single-deficit witness, "
                  "O_inert)",
            "c": "the own switch changes c more than any other switch in the "
                 "paired witness contrasts (declared dependencies c_int->NAS and "
                 "g_b->IIM excluded); registry field cross_talk = ratio, "
                 "cross_talk_max = 1",
            "operationalisation": "scripts/calibrate_bench.py::entry_criteria and "
                                  "::off_dose_rates (frozen code)",
        },
        "derived_from": {
            "freeze_tag": FREEZE_TAG,
            "freeze_commit": tag_commit,
            "freeze_tree": tag_tree,
            "records_git_sha": sorted({s for fe in fams.values()
                                       for s in fe["git_shas"]}),
            "protocol": "protocols/mpc_bench_v1.json",
            "protocol_hash": FROZEN_PROTOCOL_HASH,
            "family_C_protocol_hash": fams["C"]["protocol"].hash,
            "reference_family_A": fams["A"]["reference_info"],
            "reference_family_C": {**fams["C"]["reference_info"],
                                   "values": fams["C"]["reference"].get("values"),
                                   "se": fams["C"]["reference"].get("se")},
            "null_calibration": {
                "path": _rel(nc_path),
                "sha256": _sha256(nc_path),
                "statuses_reproduced_family_A": agree_a,
            },
            "bench_inputs": {
                fam: [{"path": i["path"], "sha256": i["sha256"]}
                      for i in fe["inputs"]]
                for fam, fe in fams.items()
            },
            "results_root": _rel(results) + " (git-ignored)",
            "builder": "scripts/build_applicability_registry.py",
            "builder_version": BUILDER_VERSION,
            "hypothesis_context": ctx,
        },
        "excluded": excluded,
        "scope_notes": [
            "Entries cover the validation regime only: 30-node MPC-Bench agents, "
            "about 11,500-12,400 samples at dt = 0.05 s, K = 19 null surrogates, "
            "jackknife G = 10, the frozen cutoffs (z, delta) = (0.25, 0.10).",
            "The whole-brain source model carries the substrate label "
            "stuart_landau too; it lies outside the family-C regime (76 regions, "
            "dt = 0.004 s, another IIM grain) and is not covered.",
            "Admission follows the preregistered entry criteria, which are "
            "weaker than the selectivity hypothesis HC3; related hypothesis "
            "outcomes are listed in each entry's note and must be reported "
            "with it.",
            "The registry does not change any preregistered hypothesis outcome "
            "and was built after they were evaluated.",
        ],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(registry, indent=2, default=_json_default) + "\n",
                        encoding="utf-8")
    loaded = ev.ApplicabilityRegistry.from_json(out_path, strict=True)
    checks = self_check(loaded, registry)
    if evidence_dir is not None:
        write_evidence(evidence_dir, fams, registry, checks, out_path)
    return {"registry": registry, "checks": checks, "families": fams}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _num(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(type(o))


def _excluded_note(fam, p, rec):
    if not rec["anchor_valid"]:
        a = rec["anchor"]
        what = {
            "RAM": "prediction-error adaptation term",
            "PDI": "unlabelled repertoire excess",
            "SRPI": "agency excess",
        }.get(p, "excess")
        return (f"The positive control's {what} over its null is not credibly "
                f"positive (mean {a['mean_excess']:.4g}, SE {a['se']:.2g}, "
                f"one-sided 95% lower bound {a['one_sided_lower_bound']:.3g}, "
                f"n = {a['n']}): no construct scale in family {fam}.")
    return None


def _non_switch_exclusions(results, proto):
    """Substrates the confirmatory runs reached without a mechanism-switch
    design: the preregistered criteria (0)-(c) cannot be met there."""
    out = []
    try:
        fb = family_b_evidence(results)
    except (FileNotFoundError, KeyError) as exc:  # pragma: no cover - reported
        fb = {"error": str(exc)}
    for cut in ("bidirectional", "directional"):
        out.append({
            "principle": "IIM",
            "estimator": f"compute_IIM:{cut}",
            "version": ESTIMATOR_VERSIONS["IIM"],
            "substrate": "ising_exact",
            "failed_criteria": [
                "(0) no construct-scale anchor exists for the exact-TPM systems "
                "(family B has no positive control)",
                "(a) independent units exceed the null margin 1.645 in more "
                "than 7% of runs in 5 of 8 (cut mode, T) cells (HC2 part c)",
                "(b) the error against the exact value does not shrink "
                "strictly with T for all-to-all and the feedforward star "
                "(HC2 part b)",
            ] + (["(d) the feedforward star exceeds its null under directional "
                  "cuts in 35-70% of runs (HC2 part d)"] if cut == "directional"
                 else []),
            "evidence": fb,
            "note": ("Family B tests IIM against exact values (HC2, FALSIFIED); "
                     "it is ground truth for the estimator, not a registry "
                     "validation."),
        })
    try:
        wb = whole_brain_evidence(results)
    except (FileNotFoundError, KeyError) as exc:  # pragma: no cover - reported
        wb = {"error": str(exc)}
    for p in ev.PRINCIPLES:
        out.append({
            "principle": p,
            "estimator": estimator_name(proto, p),
            "version": ESTIMATOR_VERSIONS[p],
            "substrate": "eeg_like_forward",
            "failed_criteria": [
                "(0) no substrate anchor (the whole-brain runs are judged with "
                "the family-A anchor)",
                "(b2), (c) not testable: the whole-brain model has no mechanism "
                "switches (external manipulations only; preregistered as "
                "descriptive)",
            ] + ({"RAM": ["RAM undefined: the model has no events"],
                  "SRPI": ["SRPI undefined: the model has no events"],
                  "PDI": ["PDI has no anchor"],
                  "IIM": ["(specificity, descriptive) IIM PRESENT in 10 of 10 "
                          "uncoupled (G = 0) EEG-like runs: volume conduction "
                          "read as integration"],
                  "NAS": []}[p]),
            "evidence": wb if p in ("NAS", "IIM") else None,
        })
        out.append({
            "principle": p,
            "estimator": estimator_name(proto, p),
            "version": ESTIMATOR_VERSIONS[p],
            "substrate": "bold_like_forward",
            "failed_criteria": [
                "no evidence: all 140 planned BOLD-like whole-brain tasks failed "
                "in the frozen runner (NAS band 0.5-4 Hz above the Nyquist "
                "frequency 0.25 Hz at TR = 2 s; the error ends the whole task) "
                "and were excluded after the preregistered re-run",
            ],
            "evidence": {"bold": wb.get("bold") if isinstance(wb, dict) else None,
                         "exploratory_rerun_without_NAS":
                             "c3/deviation_bold_noNAS (not preregistered, not "
                             "audited; cannot enter the registry)"},
        })
    return out


def self_check(loaded, registry) -> dict:
    """Every validated entry matches a query at its own regime; every excluded
    combination is refused; empirical substrates are not covered."""
    res = {"n_entries": len(loaded.entries), "entries": [], "excluded": [],
           "empirical": []}
    for e in registry["entries"]:
        reg = e["regime"]
        query = {"n_time": reg["T_min"], "n_nodes": reg["nodes_min"],
                 "tr": reg["fs_or_tr"]["min"]}
        if "bins" in reg:
            query["bins"] = reg["bins"]
        grain = e["grain"] if isinstance(e["grain"], str) and e["grain"] != "*" \
            else None
        est = f"{e['estimator']}@{e['version']}"
        ok, why = loaded.is_validated(e["principle"], est, e["substrate"], grain,
                                      **query)
        if not ok:
            raise AssertionError(
                f"entry does not match its own regime: {e['principle']} "
                f"{e['substrate']}: {why}")
        res["entries"].append({"principle": e["principle"], "substrate": e["substrate"],
                               "query": query, "validated": ok})
    for x in registry["excluded"]:
        est = f"{x['estimator']}@{x['version']}"
        ok, why = loaded.is_validated(x["principle"], est, x["substrate"], None,
                                      n_time=12000, n_nodes=30, tr=0.05, bins=2)
        if ok:
            raise AssertionError(f"excluded combination validates: {x}")
        res["excluded"].append({"principle": x["principle"],
                                "substrate": x["substrate"],
                                "estimator": x["estimator"], "reason": why})
    for sub in ("eeg", "fmri"):
        for p in ev.PRINCIPLES:
            ok, why = loaded.is_validated(p, f"compute_{p}:x@{ESTIMATOR_VERSIONS[p]}",
                                          sub, None, n_time=12000, n_nodes=30, tr=0.05)
            ok2 = any(
                loaded.is_validated(p, f"{e['estimator']}@{e['version']}", sub, None,
                                    n_time=10 ** 6, n_nodes=30, tr=0.05, bins=2)[0]
                for e in registry["entries"] if e["principle"] == p)
            if ok or ok2:
                raise AssertionError(f"empirical substrate {sub} covered for {p}")
            res["empirical"].append({"substrate": sub, "principle": p,
                                     "covered": False})
    return res


def write_evidence(evidence_dir, fams, registry, checks, out_path):
    out = Path(evidence_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([fe["entry"] for fe in fams.values()]).to_csv(
        out / "entry_criteria.csv", index=False)
    pd.concat([fe["null_rates"].assign(family=f) for f, fe in fams.items()]).to_csv(
        out / "null_false_present_by_class.csv", index=False)
    pd.concat([fe["slopes"].assign(family=f) for f, fe in fams.items()]).to_csv(
        out / "dose_response_slopes.csv", index=False)
    pd.concat([fe["levels"].assign(family=f) for f, fe in fams.items()]).to_csv(
        out / "dose_response_levels.csv", index=False)
    pd.concat([fe["contrasts"] for fe in fams.values()]).to_csv(
        out / "witness_contrasts.csv", index=False)
    pd.concat([fe["off_rates"].assign(family=f) for f, fe in fams.items()]).to_csv(
        out / "off_mechanism_present_rates.csv", index=False)
    summary = {
        "builder": BUILDER_VERSION,
        "builder_sha256": _sha256(Path(__file__)),
        "calibrate_bench_sha256": _sha256(SCRIPTS / "calibrate_bench.py"),
        "registry_path": _rel(out_path),
        "registry_sha256": _sha256(out_path),
        "validated": [
            {"principle": e["principle"], "estimator": e["estimator"],
             "version": e["version"], "substrate": e["substrate"]}
            for e in registry["entries"]
        ],
        "excluded": [
            {"principle": x["principle"], "estimator": x["estimator"],
             "substrate": x["substrate"], "failed_criteria": x["failed_criteria"]}
            for x in registry["excluded"]
        ],
        "self_check": checks,
        "code_git_head": _git("rev-parse", "HEAD"),
    }
    (out / "registry_build.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(REPO_ROOT / "outputs" / "paper1_mpcbench"),
                    help="root of the confirmatory result folders (c1-c3)")
    ap.add_argument("--out", default=str(REPO_ROOT / "protocols"
                                         / "applicability_registry_v1.json"))
    ap.add_argument("--evidence-dir", default=None,
                    help="also write the per-criterion evidence tables here")
    args = ap.parse_args(argv)
    res = build(args.results, args.out, args.evidence_dir)
    reg = res["registry"]
    print(f"validated entries: {len(reg['entries'])}")
    for e in reg["entries"]:
        print(f"  {e['principle']:5s} {e['estimator']}@{e['version']} on "
              f"{e['substrate']} (grain {e['grain']})")
    print(f"excluded combinations: {len(reg['excluded'])}")
    for x in reg["excluded"]:
        print(f"  {x['principle']:5s} {x['estimator']} on {x['substrate']}: "
              f"{'; '.join(x['failed_criteria'])[:140]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
