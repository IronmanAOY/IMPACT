#!/usr/bin/env python
"""
Null-calibration generator v2 of MPC-Bench v2: run the design and report it
in the tables of the v1 calibration.

The design (cells, seeds, hub partition, the declaration ``none`` under the
A-R classification) lives in ``impact_pipeline.bench.designs_v2.
null_calibration``; its tasks run through the v2 runner like every other
design (``scripts/run_bench_v2.py run null_calibration``, which the
development and confirmatory run scripts call). This script adds:

``plan``
    the cell grid of a split: tasks per cell, the seed base and the
    principles each null kind is scored for.
``run``
    a run of the design through the v2 runner (same plan check, guard,
    protocols, resume and records). A development run may reduce the grid
    (``--kinds``, ``--T``, ``--nodes``, ``--replicates``, ``--seed-base``);
    a confirmatory run is the full design of the plan, from the freeze tag.
``summarise``
    v1's tables from the v2 records: one row per (replicate, principle)
    (``null_calibration_v2_replicates.csv``), false-PRESENT, ABSENT and
    UNDEFINED rates per cell and principle with Wilson intervals
    (``..._rates.csv``) and the Kleene verdict rates over the principles
    scored (``..._verdicts.csv``), computed by v1's ``summarise`` (imported,
    not copied), and ``null_calibration_v2.json`` (sources, protocols,
    counts). The tables are descriptive; HCv2-1, HCv2-3 and HCv2-18 are
    decided by the evaluator on the records. The summary also counts the
    SRPI components that carry v1's trigger for its agency fallback (none
    are expected: the v2 path passes the agency events, as v1's bench path
    did).
``protocol``
    the classification protocol ``A-none`` (the A-R protocol with the
    declaration ``none``) written from an A-R protocol file, for the
    protocol builder and the freeze.

Examples::

    python scripts/v2/null_calibration_v2.py plan --split development
    python scripts/v2/null_calibration_v2.py run --split development \\
        --kinds ar1 --T 1200 --nodes 8 --replicates 2 --out <dir>
    python scripts/v2/null_calibration_v2.py summarise <dir> --out <dir>/summary
    python scripts/v2/null_calibration_v2.py protocol \\
        --base protocols/v2/generated/mpc_bench_v2_A-R.json \\
        --out protocols/v2/generated/mpc_bench_v2_A-none.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

from impact_pipeline.bench import run_bench_v2 as RB  # noqa: E402
from impact_pipeline.bench.designs_v2 import null_calibration as NCV  # noqa: E402
from impact_pipeline.v2 import provenance as PV  # noqa: E402
from impact_pipeline.v2 import records as REC  # noqa: E402
from impact_pipeline.v2 import seeds as S  # noqa: E402

SUMMARY_VERSION = "null-calibration-v2-summary/1.0.0"
REPLICATES_CSV = "null_calibration_v2_replicates.csv"
RATES_CSV = "null_calibration_v2_rates.csv"
VERDICTS_CSV = "null_calibration_v2_verdicts.csv"
SUMMARY_JSON = "null_calibration_v2.json"
STATUS_IMPL = "tost-v2"
DEFAULT_ALPHA = 0.05


class RunRefused(ValueError):
    """A run this script refuses before anything is simulated."""


# --------------------------------------------------------------------------
# plan and run
# --------------------------------------------------------------------------
def plan_rows(split: str) -> List[dict]:
    """One row per cell of the split's plan."""
    rows = {}
    for t in NCV.cells(split):
        key = (t.tags["null_kind"], t.tags["n_time"], t.tags["n_nodes"])
        r = rows.setdefault(key, {
            "cell": t.tags["cell"], "null_kind": key[0], "n_time": key[1],
            "n_nodes": key[2], "n_tasks": 0, "seed_base": t.seed,
            "principles": ",".join(t.scorings[0].principles),
            "protocol": t.scorings[0].protocol_key,
            "declaration": t.scorings[0].declaration_id})
        r["n_tasks"] += 1
    return list(rows.values())


def build_run_tasks(split: str, *, seed_bases=None, kinds=None, n_times=None,
                    n_nodes=None, replicates=None, confirmatory: bool = False):
    """The tasks of a run: the full design of the split, or (development
    only) a reduced grid."""
    reduced = any(v is not None for v in (seed_bases, kinds, n_times, n_nodes,
                                          replicates))
    if confirmatory and (split != S.CONFIRMATORY or reduced):
        raise RunRefused("a confirmatory run is the full confirmatory design "
                         "(no reduced grid, no other seed base)")
    if split == S.CONFIRMATORY and not confirmatory:
        raise RunRefused("confirmatory tasks run only with --confirmatory, through "
                         "the confirmatory guard")
    return NCV.cells(split, seeds=seed_bases, kinds=kinds, n_times=n_times,
                     n_nodes=n_nodes, replicates=replicates)


def run(out_dir, *, split: str = S.DEVELOPMENT, seed_bases=None, kinds=None,
        n_times=None, n_nodes=None, replicates=None, workers: int = 1,
        principles: Optional[Sequence[str]] = None, protocol_dir=None,
        no_drafts: bool = False, allow_unavailable_estimators: bool = False,
        resume: bool = True, confirmatory: bool = False,
        freeze_tag: Optional[str] = None, label: Optional[str] = None) -> dict:
    """Run the design (or a reduced development grid) through the v2
    runner; returns the run manifest."""
    tasks = build_run_tasks(split, seed_bases=seed_bases, kinds=kinds,
                            n_times=n_times, n_nodes=n_nodes, replicates=replicates,
                            confirmatory=confirmatory)
    RB.load_design_module(NCV.MODULE)
    settings = RB.RunSettings(
        principles=None if principles is None else tuple(principles),
        allow_unavailable_estimators=bool(allow_unavailable_estimators))
    return RB.run_tasks(tasks, out_dir, n_workers=int(workers), settings=settings,
                        resume=resume, confirmatory=confirmatory,
                        freeze_tag=freeze_tag, protocol_dir=protocol_dir,
                        allow_drafts=False if no_drafts or confirmatory else None,
                        label=label)


# --------------------------------------------------------------------------
# summary in v1's tables
# --------------------------------------------------------------------------
def _results_files(paths: Iterable) -> List[Path]:
    out = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.rglob(RB.RESULTS_JSONL)))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"no results at {p}")
    return out


def load_records(paths: Iterable) -> tuple:
    """The latest record per task of the null-calibration design from
    results files or run directories, and the files read (with SHA-256)."""
    recs, files = {}, []
    for f in _results_files(paths):
        latest, info = RB.read_results(f)
        files.append({"path": str(f), "sha256": PV.file_sha256(f), **info})
        for tid, rec in latest.items():
            if rec.design == NCV.DESIGN:
                recs[tid] = rec
    return [recs[k] for k in sorted(recs)], files


def _finite(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def agency_fix_needed(component) -> bool:
    """Whether an SRPI component carries v1's trigger for its agency
    fallback (the runner did not pass the agency events)."""
    nc = NCV.v1_generator()
    det = component.details or {}
    reasons = (det.get("estimator_reason"), (det.get("estimator") or {}).get("reason"),
               component.reason)
    return any(nc.AGENCY_FIX_REASON in str(r or "") for r in reasons)


def replicate_rows(records: Sequence) -> pd.DataFrame:
    """One row per (task, principle) of the primary scoring, in the columns
    of v1's replicate table (plus the v2 fields)."""
    rows = []
    for rec in records:
        if rec.design != NCV.DESIGN or rec.status == REC.TASK_ERROR:
            continue
        tags = (rec.config or {}).get("tags") or {}
        sim = rec.simulation or {}
        kind = tags["null_kind"]
        n_time, n_nodes = int(sim["n_time"]), int(sim["n_nodes"])
        rep, base = int(tags["null_replicate"]), int(tags["seed_base"])
        seconds = _finite((rec.timing or {}).get("total_s"))
        seed = NCV.system_seed(base, kind, n_time, n_nodes, rep)
        for sc in rec.scorings:
            if sc.estimator_form != "primary":
                continue
            for p, c in sc.components.items():
                est, nm, nsd = (_finite(c.estimate), _finite(c.null_mean),
                                _finite(c.null_sd))
                margin = ((est - nm) / nsd if None not in (est, nm, nsd) and nsd > 0
                          else float("nan"))
                det = c.details or {}
                assessment = det.get("assessment") or {}
                rows.append({
                    "null_kind": kind, "n_time": n_time, "n_nodes": n_nodes,
                    "replicate": rep, "seed": seed, "seed_base": base,
                    "task_id": rec.task_id, "principle": p,
                    "estimate": est, "null_mean": nm, "null_sd": nsd,
                    "n_null": int(c.n_null or 0), "null_family": c.null_family,
                    "statistic": (det.get("estimator") or {}).get("statistic"),
                    "defined": bool(det.get("defined", est is not None)),
                    "estimator_reason": det.get("estimator_reason"),
                    "margin": margin, "se": _finite(c.se), "se_df": _finite(c.se_df),
                    "se_method": c.se_method, "c": _finite(c.c),
                    "c_R": _finite(c.c_R), "c_B": _finite(c.c_B),
                    "c_se": _finite(c.se_c), "c_df": _finite(c.df_c),
                    "c_lower": _finite(assessment.get("lower")),
                    "c_upper": _finite(assessment.get("upper")),
                    "status": c.status, "status_reason": c.reason,
                    "flags": ";".join(c.flags),
                    "status_impl": STATUS_IMPL,
                    "runner": f"run_bench_v2:{c.estimator_version}",
                    "estimator_version": c.estimator_version,
                    "declaration_id": sc.declaration_id,
                    "protocol_id": sc.protocol_id, "protocol_hash": sc.protocol_hash,
                    "agency_fix_needed": p == "SRPI" and agency_fix_needed(c),
                    "seconds_system": float("nan") if seconds is None else seconds,
                })
    return pd.DataFrame(rows)


def summarise(paths: Iterable, out_dir, *, alpha: float = DEFAULT_ALPHA) -> dict:
    """Write v1's three tables and the summary JSON for the records under
    ``paths``; returns the summary."""
    nc = NCV.v1_generator()
    records, files = load_records(paths)
    if not records:
        raise ValueError("no null-calibration records found")
    rep = replicate_rows(records)
    if rep.empty:
        raise ValueError("the null-calibration records hold no scored components")
    split = S.check_seeds([r.seed for r in records])
    rates, verdicts = nc.summarise(rep, alpha=float(alpha))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out / REPLICATES_CSV, index=False)
    rates.to_csv(out / RATES_CSV, index=False)
    verdicts.to_csv(out / VERDICTS_CSV, index=False)
    protocols = sorted({(r.protocol_id, r.protocol_hash) for r in rep.itertuples()})
    summary = {
        "version": SUMMARY_VERSION,
        "design": NCV.DESIGN,
        "null_calibration_version": NCV.NULL_CALIBRATION_VERSION,
        "split": split,
        "alpha": float(alpha),
        "band": nc.BAND,
        "sources": files,
        "n_tasks": len(records),
        "n_task_errors": sum(r.status == REC.TASK_ERROR for r in records),
        "n_components": int(len(rep)),
        "n_estimator_errors": int(rep["status_reason"].astype(str)
                                  .str.startswith("ESTIMATOR_ERROR").sum()),
        "n_srpi_agency_fix_needed": int(rep["agency_fix_needed"].sum()),
        "cells": sorted({f"{k}:T{t}:N{n}" for k, t, n in
                         zip(rep["null_kind"], rep["n_time"], rep["n_nodes"])}),
        "seed_bases": sorted({int(b) for b in rep["seed_base"]}),
        "protocols": [{"protocol_id": p, "protocol_hash": h} for p, h in protocols],
        "declarations": sorted(set(rep["declaration_id"])),
        "applicability": {k: list(v) for k, v in nc.NULL_KINDS.items()},
        "tables": {"replicates": REPLICATES_CSV, "rates": RATES_CSV,
                   "verdicts": VERDICTS_CSV},
        "note": "descriptive tables in v1's layout; HCv2-1, HCv2-3 and HCv2-18 are "
                "decided by the evaluator on the records",
        "code": PV.code_identity(REPO_ROOT),
    }
    (out / SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True,
                                               default=str), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------
# the classification protocol file
# --------------------------------------------------------------------------
def write_protocol(base_path, out_path) -> dict:
    """Write ``A-none`` derived from the A-R protocol file ``base_path``
    (the file must be the A-R protocol: declaration R); returns its name
    and hash."""
    from impact_pipeline import evidence_v2 as E

    base = E.ProtocolV3.from_json(base_path)
    proto = E.ProtocolV3.from_dict(NCV.classification_protocol(base.to_dict()))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    proto.to_json(out)
    return {"name": proto.name, "hash": proto.hash, "path": str(out),
            "base_hash": base.hash}


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def _ints(text) -> Optional[List[int]]:
    return None if text is None else RB._parse_ints(text)


def _strs(text) -> Optional[List[str]]:
    return None if text is None else RB._parse_list(text)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="null_calibration_v2.py",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    pl = sub.add_parser("plan", help="the cell grid of a split")
    pl.add_argument("--split", default=S.DEVELOPMENT, choices=S.SPLITS)
    rn = sub.add_parser("run", help="run the design through the v2 runner")
    rn.add_argument("--out", required=True)
    rn.add_argument("--split", default=S.DEVELOPMENT, choices=S.SPLITS)
    rn.add_argument("--seed-base", default=None,
                    help="development seed base(s), e.g. 400 (default: the split's)")
    rn.add_argument("--kinds", default=None, help="null kinds (development only)")
    rn.add_argument("--T", default=None, help="run lengths (development only)")
    rn.add_argument("--nodes", default=None, help="node counts (development only)")
    rn.add_argument("--replicates", type=int, default=None,
                    help="replicates per cell (development only)")
    rn.add_argument("--workers", type=int, default=1)
    rn.add_argument("--principles", default=None)
    rn.add_argument("--protocol-dir", default=None)
    rn.add_argument("--no-drafts", action="store_true")
    rn.add_argument("--allow-unavailable-estimators", action="store_true")
    rn.add_argument("--no-resume", action="store_true")
    rn.add_argument("--confirmatory", action="store_true")
    rn.add_argument("--freeze-tag", default=None)
    rn.add_argument("--label", default=None)
    sm = sub.add_parser("summarise", help="v1's tables from the v2 records")
    sm.add_argument("results", nargs="+", help="results.jsonl files or run directories")
    sm.add_argument("--out", required=True)
    sm.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    pr = sub.add_parser("protocol", help="write A-none from an A-R protocol file")
    pr.add_argument("--base", required=True, help="the A-R protocol file")
    pr.add_argument("--out", required=True)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.command == "plan":
            rows = plan_rows(args.split)
            for r in rows:
                print(f"{r['cell']:28s} tasks {r['n_tasks']:3d} seed base "
                      f"{r['seed_base']} {r['declaration']} under {r['protocol']}: "
                      f"{r['principles']}")
            print(f"{len(rows)} cells, {sum(r['n_tasks'] for r in rows)} tasks")
            return 0
        if args.command == "run":
            man = run(args.out, split=args.split, seed_bases=_ints(args.seed_base),
                      kinds=_strs(args.kinds), n_times=_ints(args.T),
                      n_nodes=_ints(args.nodes), replicates=args.replicates,
                      workers=args.workers, principles=_strs(args.principles),
                      protocol_dir=args.protocol_dir, no_drafts=args.no_drafts,
                      allow_unavailable_estimators=args.allow_unavailable_estimators,
                      resume=not args.no_resume, confirmatory=args.confirmatory,
                      freeze_tag=args.freeze_tag, label=args.label)
            print(f"null calibration v2: {man['n_run']} task(s) run, "
                  f"{man['n_errors']} task error(s), {man['n_component_errors']} "
                  f"component error(s); results in {args.out}")
            return 1 if man["n_errors"] or not man["plan_differences"]["ok"] else 0
        if args.command == "summarise":
            s = summarise(args.results, args.out, alpha=args.alpha)
            print(f"null calibration v2 summary: {s['n_tasks']} task(s), "
                  f"{s['n_components']} component(s), {len(s['cells'])} cell(s); "
                  f"tables in {args.out}")
            return 0
        info = write_protocol(args.base, args.out)
        print(f"{info['name']} ({info['hash'][:12]}) written to {info['path']}")
        return 0
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"null_calibration_v2: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
