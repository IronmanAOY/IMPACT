#!/usr/bin/env python
"""
Development calibration of the MPC-Bench protocol (before the code freeze).

Reads development bench runs only (seeds 0-999, family A; any record from a
confirmatory split is refused) and writes the evidence the preregistered
protocol choices are based on:

1. reference anchors of the nominal positive control (``--reference``,
   ``bench.reference.reference_from_records`` with its anchor-validity rule
   at the template protocol's ``alpha``) and the candidate protocol (the
   template with that reference);
2. the construct-scale assessment of every development record under the
   candidate protocol (evidence layer: ``c``, ``se_c`` and its parts
   ``se_sampling`` / ``se_null`` / ``se_reference``, degrees of freedom,
   bounds, status) and a consistency check of the re-applied status rule
   (``bench.analysis.status_from_bounds``) against the evidence layer;
3. dose-response of ``c`` for each principle's own knob and the off-target
   knobs (``sweep`` records), witness status rates and verdicts;
4. false-PRESENT / ABSENT rates on null systems (null witnesses and, with
   ``--null-calibration``, the ``scripts/null_calibration.py`` replicates
   re-anchored on the candidate reference) and PRESENT / ABSENT rates at the
   nominal and mechanism-off doses, for a grid of candidate cutoffs
   ``(z, delta, alpha)``;
5. the jackknife SE against the between-seed SD of identically configured
   systems, and the share of ``var(c)`` due to the Monte-Carlo error of the
   null mean (choice of the number of surrogates ``K``);
6. the entry criteria per principle (anchor validity, null false-PRESENT
   rate <= alpha + 0.02, monotone own-knob recovery) that define the
   validated necessity subset used by the verdict-level hypotheses.

Outputs go to ``--out`` (CSV tables and ``calibration_summary.json``). The
numbers are development results: they justify the frozen choices and are
reported as such, never as confirmatory evidence.

Example::

    DEV=outputs/paper1_mpcbench/dev
    python scripts/calibrate_bench.py --out $DEV/calibration \\
        --template protocols/mpc_bench_v1.json \\
        --reference $DEV/reference_A/run \\
        --records $DEV/sweep_A,$DEV/witnesses_A \\
        --null-calibration $DEV/null_calibration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline.bench import analysis as A  # noqa: E402
from impact_pipeline.bench.generators import PRINCIPLES  # noqa: E402

CALIBRATION_VERSION = "mpc-bench-calibration/1.0.0"
Z_GRID = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
DELTA_GRID = (0.05, 0.1, 0.15, 0.2, 0.25)
ALPHA_GRID = (0.05, 0.025, 0.01)
# Statistical null systems (no coupling, no task processing): entry
# criterion (a). The over-excluded constructions are construct negatives
# (every mechanism absent, but coupled or driven): reported, and O_inert (the
# family-A agent with every switch off) enters the off-dose specificity (b2).
NULL_WITNESSES = ("N_independent_noise", "N_ar1")
CONSTRUCT_NEGATIVES = ("O_inert", "O_hypersynchronous")
BAND = 0.02


def assess_raw(frame: pd.DataFrame, reference: dict, protocol) -> pd.DataFrame:
    """``bench.analysis.assess_components`` (construct-scale assessment of raw
    component rows against an external reference)."""
    return A.assess_components(frame, reference, protocol)


def _components_raw(records) -> pd.DataFrame:
    return A.components_raw(records)


def off_dose_rates(comp: pd.DataFrame) -> pd.DataFrame:
    """PRESENT rate of each principle on systems without its mechanism: the
    lowest dose of its own knob (sweeps), its own single-deficit witness and
    the all-off agent ``O_inert``."""
    rows = []
    for knob, p in A.KNOB_TARGET.items():
        sub = comp[comp["principle"] == p]
        sw = sub[(sub["design"] == "sweep") & (sub["sweep_knob"] == knob)]
        if len(sw):
            lev = sw["sweep_level"].astype(float)
            sw = sw[lev == lev.min()]
        wit = sub[
            (sub["design"] == "witnesses")
            & (sub["bearer_mode"] == "system")
            & sub["witness_id"].isin([A.SWITCH_WITNESS[knob], "O_inert"])
        ]
        both = pd.concat([sw, wit])
        for label, part in (("sweep_off", sw), ("witness_off", wit), ("all", both)):
            rows.append(
                {
                    "principle": p,
                    "systems": label,
                    "n": int(len(part)),
                    "present_rate": (
                        float((part["status"] == "PRESENT").mean())
                        if len(part)
                        else float("nan")
                    ),
                    "c_median": (
                        float(part["c"].median()) if len(part) else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def entry_criteria(
    anchor_summary, null_rates, slopes, protocol, contrasts=None, off_rates=None
) -> pd.DataFrame:
    """
    Development entry criteria per principle (spec v2, V2-5, operationalised
    here): (0) a valid reference anchor; (a) false-PRESENT rate <= alpha +
    0.02 on every development statistical null class; (b) recovery of the
    own knob: (b1) Spearman rho of ``c`` with the dose > 0 at p < alpha
    (sweeps) and (b2) off-dose specificity: PRESENT rate <= alpha + 0.02 on
    the systems without the mechanism (:func:`off_dose_rates`); (c)
    acceptable cross-talk: in the paired witness contrasts the own switch
    changes ``c`` more (median absolute change) than any other switch,
    declared structural dependencies excluded. ``validated`` = all of them.
    """
    rows = []
    alpha = float(protocol.alpha)
    ct = contrasts if contrasts is not None else pd.DataFrame()
    off = off_rates if off_rates is not None else pd.DataFrame()
    for p in PRINCIPLES:
        per = anchor_summary["per_principle"].get(p, {})
        anchor_ok = p not in anchor_summary.get("no_anchor", [])
        nr = (
            null_rates[(null_rates["principle"] == p)]
            if len(null_rates)
            else null_rates
        )
        max_fp = float(nr["false_present_rate"].max()) if len(nr) else float("nan")
        null_ok = bool(len(nr)) and bool(max_fp <= alpha + BAND)
        knob = next(k for k, t in A.KNOB_TARGET.items() if t == p)
        sl = (
            slopes[(slopes["knob"] == knob) & (slopes["principle"] == p)]
            if len(slopes)
            else slopes
        )
        rho = float(sl["spearman_rho"].iloc[0]) if len(sl) else float("nan")
        pval = float(sl["spearman_p"].iloc[0]) if len(sl) else float("nan")
        mono_ok = bool(np.isfinite(rho) and rho > 0 and pval < alpha)
        own_d, other_d, cross_ok = float("nan"), float("nan"), False
        if len(ct):
            sub = ct[(ct["principle"] == p) & ~ct["declared_dependency"]]
            own_rows = sub[sub["own"]]
            others = sub[~sub["own"]]
            if len(own_rows) and len(others):
                own_d = float(own_rows["median_delta_c"].abs().min())
                other_d = float(others["median_delta_c"].abs().max())
                cross_ok = bool(own_d > other_d)
        off_rate = float("nan")
        if len(off):
            o = off[(off["principle"] == p) & (off["systems"] == "all")]
            off_rate = float(o["present_rate"].iloc[0]) if len(o) else float("nan")
        spec_ok = bool(np.isfinite(off_rate) and off_rate <= alpha + BAND)
        rows.append(
            {
                "principle": p,
                "anchor_valid": anchor_ok,
                "anchor_mean_excess": per.get("mean"),
                "anchor_se": per.get("se"),
                "anchor_lower_bound": per.get("lower_bound"),
                "null_false_present_max": max_fp,
                "null_ok": null_ok,
                "own_knob": knob,
                "own_knob_spearman_rho": rho,
                "own_knob_spearman_p": pval,
                "monotone_ok": mono_ok,
                "off_dose_present_rate": off_rate,
                "off_dose_specific_ok": spec_ok,
                "own_switch_median_abs_delta_c": own_d,
                "max_other_switch_median_abs_delta_c": other_d,
                "crosstalk_ok": cross_ok,
                "validated": bool(
                    anchor_ok and null_ok and mono_ok and spec_ok and cross_ok
                ),
            }
        )
    return pd.DataFrame(rows)


def run(
    out_dir,
    *,
    template,
    reference_dirs,
    record_dirs,
    null_calibration=None,
    extra_reference=None,
) -> dict:
    from impact_pipeline import evidence as ev

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tpl = ev.resolve_protocol(template)
    ref_recs = A.read_records(reference_dirs)
    A.check_split(ref_recs, ("development",))
    reference, anchor_summary = A.family_reference(ref_recs, alpha=tpl.alpha)
    proto = A.protocol_for_family(tpl, reference)
    proto.to_json(out / "candidate_protocol.json")
    records = A.read_records(record_dirs)
    A.check_split(records, ("development",))
    raw = _components_raw(records)
    comp = assess_raw(raw, reference, proto)
    comp.to_csv(out / "components_assessed.csv", index=False)
    # verdicts (and statuses) from the evidence layer itself
    ev_comp, verdicts = A.assess(records, proto)
    verdicts.to_csv(out / "verdicts.csv", index=False)
    merged = comp.merge(
        ev_comp[["task_id", "principle", "status"]],
        on=["task_id", "principle"],
        suffixes=("", "_layer"),
    )
    regrid = (
        A.status_from_bounds(
            merged["c"],
            merged["se_c"],
            merged["df"],
            *proto.cutoff_for("NAS"),
            proto.alpha,
        )
        if len({tuple(proto.cutoff_for(p)) for p in PRINCIPLES}) == 1
        else None
    )
    consistency = {
        "assess_raw_vs_layer_agree": bool(
            (merged["status"] == merged["status_layer"]).all()
        ),
        "n_rows": int(len(merged)),
    }
    if regrid is not None:
        fin = np.isfinite(merged["c"]) & np.isfinite(merged["se_c"])
        consistency["status_from_bounds_agree_on_finite"] = bool(
            (regrid[fin.to_numpy()] == merged.loc[fin, "status"].to_numpy()).all()
        )
    levels, slopes = A.dose_response(comp)
    levels.to_csv(out / "dose_response_levels.csv", index=False)
    slopes.to_csv(out / "dose_response_slopes.csv", index=False)
    wit = comp[comp["design"] == "witnesses"]
    wit_rates = (
        (
            wit.groupby(["witness_id", "bearer_mode", "principle"])
            .agg(
                n=("status", "size"),
                present=("status", lambda s: float((s == "PRESENT").mean())),
                absent=("status", lambda s: float((s == "ABSENT").mean())),
                c_median=("c", "median"),
                se_c_median=("se_c", "median"),
            )
            .reset_index()
        )
        if len(wit)
        else pd.DataFrame()
    )
    wit_rates.to_csv(out / "witness_status_rates.csv", index=False)
    if len(verdicts):
        vd = (
            verdicts.groupby(
                ["design", "witness_id", "cell_id", "bearer_mode"], dropna=False
            )["verdict"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        vd.to_csv(out / "verdict_counts.csv", index=False)
    # null systems: bench null witnesses and null_calibration replicates
    nulls = [
        wit[wit["witness_id"].isin(NULL_WITNESSES)].assign(
            null_class=lambda d: "witness:" + d["witness_id"].astype(str)
        )
    ]
    if null_calibration:
        nc = pd.read_csv(Path(null_calibration) / "null_calibration_replicates.csv")
        nc = nc.rename(columns={"null_kind": "kind"})
        nc_raw = nc[
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
        nc_raw["se"] = nc_raw["se"].astype(float)
        nca = assess_raw(nc_raw, reference, proto)
        nca["null_class"] = (
            "null_calibration:"
            + nca["kind"].astype(str)
            + ":T"
            + nca["n_time"].astype(str)
            + ":N"
            + nca["n_nodes"].astype(str)
        )
        nca.to_csv(out / "null_calibration_reassessed.csv", index=False)
        nulls.append(nca)
    null_df = pd.concat(nulls, ignore_index=True) if nulls else pd.DataFrame()
    null_grid = (
        A.cutoff_grid(
            null_df, ["null_class", "principle"], Z_GRID, DELTA_GRID, ALPHA_GRID
        )
        if len(null_df)
        else pd.DataFrame()
    )
    null_grid.to_csv(out / "null_cutoff_grid.csv", index=False)
    at_proto = (
        null_df.assign(fp=(null_df["status"] == "PRESENT")) if len(null_df) else null_df
    )
    null_rates = (
        (
            at_proto.groupby(["null_class", "principle"])
            .agg(n=("fp", "size"), false_present_rate=("fp", "mean"))
            .reset_index()
        )
        if len(at_proto)
        else pd.DataFrame(
            columns=["null_class", "principle", "n", "false_present_rate"]
        )
    )
    null_rates.to_csv(out / "null_false_present_at_protocol.csv", index=False)
    # nominal and mechanism-off doses of each principle's own knob
    sw = comp[comp["design"] == "sweep"].copy()
    if len(sw):
        sw["sweep_level"] = sw["sweep_level"].astype(float)
        first = sw.groupby("sweep_knob")["sweep_level"].transform("min")
        from impact_pipeline.bench.generators import NOMINAL_KNOBS

        nominal = NOMINAL_KNOBS.to_dict()
        sw["dose_class"] = np.where(
            sw["sweep_level"] == first,
            "off",
            np.where(
                np.isclose(
                    sw["sweep_level"], sw["sweep_knob"].map(nominal).astype(float)
                ),
                "nominal",
                "other",
            ),
        )
        own = sw[sw["principle"] == sw["sweep_knob"].map(A.KNOB_TARGET)]
        dose_grid = A.cutoff_grid(
            own[own["dose_class"] != "other"],
            ["principle", "dose_class"],
            Z_GRID,
            DELTA_GRID,
            ALPHA_GRID,
        )
        dose_grid.to_csv(out / "own_knob_cutoff_grid.csv", index=False)
    se_rows = comp[
        (comp["design"] == "witnesses") & (comp["witness_id"] == "PC_nominal")
    ]
    ref_comp = assess_raw(_components_raw(ref_recs), reference, proto)
    se_cal = A.se_calibration(
        pd.concat(
            [ref_comp.assign(group="reference_PC"), se_rows.assign(group="witness_PC")]
        ),
        ["group"],
    )
    se_cal.to_csv(out / "se_calibration.csv", index=False)
    parts = pd.concat([ref_comp, se_rows])
    parts = parts[np.isfinite(parts["se_c"]) & (parts["se_c"] > 0)]
    var_share = (
        parts.assign(
            null_share=parts["se_null"] ** 2 / parts["se_c"] ** 2,
            ref_share=parts["se_reference"] ** 2 / parts["se_c"] ** 2,
        )
        .groupby("principle")[["null_share", "ref_share"]]
        .median()
        .reset_index()
    )
    var_share.to_csv(out / "se_c_variance_shares.csv", index=False)
    contrasts = A.witness_contrasts(comp)
    contrasts.to_csv(out / "witness_contrasts.csv", index=False)
    off_rates = off_dose_rates(comp)
    off_rates.to_csv(out / "off_dose_present_rates.csv", index=False)
    neg = comp[
        (comp["design"] == "witnesses") & comp["witness_id"].isin(CONSTRUCT_NEGATIVES)
    ]
    if len(neg):
        (
            neg.groupby(["witness_id", "principle"])
            .agg(
                n=("status", "size"),
                present_rate=("status", lambda x: float((x == "PRESENT").mean())),
                c_median=("c", "median"),
            )
            .reset_index()
            .to_csv(out / "construct_negatives.csv", index=False)
        )
    entry = entry_criteria(
        anchor_summary, null_rates, slopes, proto, contrasts, off_rates
    )
    entry.to_csv(out / "entry_criteria.csv", index=False)
    extra = {}
    if extra_reference:
        for label, spec in extra_reference.items():
            rr = A.read_records(spec["reference"])
            A.check_split(rr, ("development",))
            rref, rsum = A.family_reference(rr, alpha=tpl.alpha)
            recs = A.read_records(spec["records"])
            A.check_split(recs, ("development",))
            pr = A.protocol_for_family(tpl, rref)
            cr = assess_raw(_components_raw(recs), rref, pr)
            cr.to_csv(out / f"components_assessed_{label}.csv", index=False)
            lv, sl = A.dose_response(cr)
            lv.to_csv(out / f"dose_response_levels_{label}.csv", index=False)
            sl.to_csv(out / f"dose_response_slopes_{label}.csv", index=False)
            w = cr[cr["design"] == "witnesses"]
            if len(w):
                (
                    w.groupby(["witness_id", "principle"])
                    .agg(
                        n=("status", "size"),
                        present=("status", lambda s: float((s == "PRESENT").mean())),
                        absent=("status", lambda s: float((s == "ABSENT").mean())),
                        c_median=("c", "median"),
                        se_c_median=("se_c", "median"),
                    )
                    .reset_index()
                ).to_csv(out / f"witness_status_rates_{label}.csv", index=False)
            extra[label] = {"reference": rref, "summary": rsum}
    from impact_pipeline.provenance import collect_code_version

    summary = {
        "version": CALIBRATION_VERSION,
        "split": "development",
        "template_protocol_hash": tpl.hash,
        "candidate_protocol_hash": proto.hash,
        "reference": reference,
        "anchor_summary": anchor_summary,
        "consistency": consistency,
        "entry_criteria": entry.to_dict(orient="records"),
        "validated_subset": [
            r["principle"] for r in entry.to_dict(orient="records") if r["validated"]
        ],
        "extra_references": extra,
        "inputs": {
            "reference": [str(p) for p in reference_dirs],
            "records": [str(p) for p in record_dirs],
            "null_calibration": str(null_calibration) if null_calibration else None,
            "n_records": len(records),
        },
        "grids": {"z": Z_GRID, "delta": DELTA_GRID, "alpha": ALPHA_GRID},
        "code_version": collect_code_version(REPO_ROOT),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return {
        "summary": summary,
        "components": comp,
        "entry": entry,
        "slopes": slopes,
        "levels": levels,
        "null_rates": null_rates,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--template", default=str(REPO_ROOT / "protocols" / "mpc_bench_v1.json")
    )
    ap.add_argument(
        "--reference",
        required=True,
        help="comma-separated run_bench dirs with PC_nominal records",
    )
    ap.add_argument(
        "--records",
        required=True,
        help="comma-separated run_bench result dirs (development)",
    )
    ap.add_argument(
        "--null-calibration",
        default=None,
        help="scripts/null_calibration.py output directory",
    )
    ap.add_argument(
        "--extra",
        default=None,
        help="JSON {label: {reference: [dirs], records: [dirs]}} for "
        "alternative estimator settings (e.g. IIM cut modes)",
    )
    args = ap.parse_args(argv)
    split = [s.strip() for s in args.records.split(",") if s.strip()]
    extra = json.loads(args.extra) if args.extra else None
    res = run(
        args.out,
        template=args.template,
        reference_dirs=[s.strip() for s in args.reference.split(",") if s.strip()],
        record_dirs=split,
        null_calibration=args.null_calibration,
        extra_reference=extra,
    )
    print(res["entry"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
