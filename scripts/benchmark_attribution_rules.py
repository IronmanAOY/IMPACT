#!/usr/bin/env python
"""
Rule audit of MPC-Bench on estimated component statuses (spec v2, V2-6).

Runs bench systems through the public estimators and the evidence layer
(``impact_pipeline.bench.run_bench``: components, null families, jackknife
SEs, markers), or reads existing run_bench results, then applies the IMPaCT
v2 rule, the installed evidence layer and every rival rule of
``impact_pipeline.bench.rules`` to the estimated component matrices under the
missingness scenarios (none; RAM+SRPI missing; SRPI missing; 20% random) and
training-label noise, with single-marker comparators (LZc, Gaussian Phi_R,
exact-TPM IIM). See ``impact_pipeline.bench.audit``.

Outputs (``--out``): ``audit_decisions.csv`` (one row per system x rule x
scenario x noise level), ``audit_p_verdict_given_class.csv``,
``audit_coverage.csv`` (coverage, selective risk, P(EXCLUDED | positive),
P(MPC_CONSISTENT | negative)), ``audit_risk_coverage.csv`` (risk at coverage
0.1 .. 1.0 and AURC) and ``audit_summary.json`` (settings, provenance, class
counts, estimator runtimes). Development runs are flagged ``not a result``.

Examples:
    # development smoke run (families A/B designs, dev seeds only)
    python scripts/benchmark_attribution_rules.py --run --designs factorial \
        --cells b11111,b01111,b10111,b11011,b11101,b11110 --seeds 0-1 \
        --null-surrogates 19 --se-groups 5 --workers 4 --out outputs/bench/audit_dev
    # audit existing results
    python scripts/benchmark_attribution_rules.py --results outputs/bench/factorial_A \
        --out outputs/bench/audit_factorial_A
    # confirmatory (after the code freeze): adds family C, adversarial, whole-brain
    python scripts/benchmark_attribution_rules.py --run --confirmatory \
        --freeze-tag <tag> --designs factorial,witnesses,adversarial,whole_brain \
        --family C --seeds 10000-10019 --out <workspace>/bench/audit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline.bench import audit as A  # noqa: E402
from impact_pipeline.bench import run_bench as RB  # noqa: E402

DEV_DESIGNS = ("factorial", "witnesses", "patchwork_sweep")
CONFIRMATORY_DESIGNS = DEV_DESIGNS + ("adversarial", "whole_brain")
NOTICE_DEV = (
    "Development-seed smoke run: not a result. No number from this output may "
    "be reported; confirmatory numbers come only from frozen-code runs."
)


def read_records(results_dir) -> list:
    """Latest record per task id from every results.jsonl below the folder."""
    latest = {}
    for path in sorted(Path(results_dir).rglob(RB.RESULTS_JSONL)):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    latest[rec["task_id"]] = rec
    return [latest[k] for k in sorted(latest)]


def make_tasks(args) -> list:
    seeds = RB.parse_seeds(args.seeds)
    designs = [d.strip() for d in args.designs.split(",") if d.strip()]
    allowed = CONFIRMATORY_DESIGNS if args.confirmatory else DEV_DESIGNS
    bad = sorted(set(designs) - set(allowed))
    if bad:
        raise ValueError(
            f"design(s) {bad} are not allowed here (development: {DEV_DESIGNS}; "
            f"confirmatory: {CONFIRMATORY_DESIGNS})"
        )
    config = RB._json_arg(args.config)
    tasks = []
    for d in designs:
        if d == "factorial":
            cells = (
                None if not args.cells else [c.strip() for c in args.cells.split(",")]
            )
            tasks += RB.factorial_tasks(
                seeds, family=args.family, config=config, cells=cells
            )
        elif d == "witnesses":
            tasks += RB.witness_tasks(seeds, family=args.family, config=config)
        elif d == "patchwork_sweep":
            tasks += RB.patchwork_sweep_tasks(seeds, family=args.family, config=config)
        elif d == "adversarial":
            tasks += RB.adversarial_tasks(seeds, config=config)
        elif d == "whole_brain":
            tasks += RB.whole_brain_tasks(seeds)
    return tasks


def runtime_summary(records) -> dict:
    """Mean wall-clock seconds per system: simulation, each estimator, SEs."""
    import numpy as np

    sims, total, est, se = [], [], {}, {}
    for r in records:
        t = r.get("timing") or {}
        if t.get("simulate_s") is not None:
            sims.append(float(t["simulate_s"]))
        if t.get("total_s") is not None:
            total.append(float(t["total_s"]))
        for p, c in (r.get("components") or {}).items():
            if c.get("seconds") is not None:
                est.setdefault(p, []).append(float(c["seconds"]))
            if c.get("se_seconds") is not None:
                se.setdefault(p, []).append(float(c["se_seconds"]))

    def _m(v):
        return float(np.mean(v)) if v else None

    return {
        "n_records": len(records),
        "simulate_s": _m(sims),
        "total_s": _m(total),
        "estimator_s": {p: _m(v) for p, v in est.items()},
        "jackknife_se_s": {p: _m(v) for p, v in se.items()},
    }


def build_parser():
    ap = argparse.ArgumentParser(
        prog="benchmark_attribution_rules.py", description=__doc__.split("\n\n")[0]
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--results", default=None, help="existing run_bench output folder")
    src.add_argument("--run", action="store_true", help="run the bench tasks first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", default="factorial")
    ap.add_argument("--family", default="A", choices=("A", "C"))
    ap.add_argument("--cells", default=None)
    ap.add_argument("--seeds", default="0-1")
    ap.add_argument("--config", default=None)
    ap.add_argument("--metrics", default=",".join(A.PRINCIPLES))
    ap.add_argument("--null-surrogates", type=int, default=19)
    ap.add_argument("--se-groups", type=int, default=5)
    ap.add_argument(
        "--protocol",
        default=None,
        help="protocol of the per-system verdicts (as run_bench --protocol)",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--confirmatory", action="store_true")
    ap.add_argument("--freeze-tag", default=None)
    ap.add_argument("--scenarios", default=",".join(A.SCENARIOS))
    ap.add_argument(
        "--label-noise", default=",".join(str(x) for x in A.LABEL_NOISE_LEVELS)
    )
    ap.add_argument("--necessity-set", default=",".join(A.PRINCIPLES))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _main(args)
    except (ValueError, RuntimeError) as exc:
        print(f"benchmark_attribution_rules: error: {exc}", file=sys.stderr)
        return 2


def _main(args) -> int:
    out = Path(args.out)
    t0 = time.time()
    code = None
    if args.run:
        tasks = make_tasks(args)
        RB.check_seed_policy(tasks, args.confirmatory)
        if args.confirmatory:
            code = RB.confirmatory_guard(RB.REPO_ROOT, args.freeze_tag)
        prov = RB.collect_provenance(vars(args), code_version=code)
        est_dir = out / "estimates"
        RB.run_tasks(
            tasks,
            est_dir,
            n_workers=args.workers,
            metrics=[m.strip() for m in args.metrics.split(",") if m.strip()],
            null_surrogates=args.null_surrogates,
            provenance=prov,
            confirmatory=args.confirmatory,
            se_groups=args.se_groups,
            protocol=RB.resolve_protocol_arg(args.protocol),
        )
        records = read_records(est_dir)
    else:
        records = read_records(args.results)
    if not records:
        raise ValueError("no bench records to audit")
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    noise = [float(x) for x in args.label_noise.split(",") if x.strip()]
    nset = [p.strip() for p in args.necessity_set.split(",") if p.strip()]
    res = A.audit(
        records,
        scenarios=scenarios,
        label_noise=noise,
        necessity_set=nset,
        alpha=args.alpha,
        seed=args.seed,
    )
    out.mkdir(parents=True, exist_ok=True)
    res["decisions"].to_csv(out / "audit_decisions.csv", index=False)
    res["p_verdict_given_class"].to_csv(
        out / "audit_p_verdict_given_class.csv", index=False
    )
    res["coverage"].to_csv(out / "audit_coverage.csv", index=False)
    res["risk_coverage"].to_csv(out / "audit_risk_coverage.csv", index=False)
    splits = sorted(
        {
            str(
                r.get("split")
                or (
                    "confirmatory"
                    if RB.seed_set(int(r.get("seed", 0))) == "confirmatory"
                    else "development"
                )
            )
            for r in records
        }
    )
    dev_only = splits == ["development"]
    shas = sorted(
        {
            ((r.get("provenance") or {}).get("code_version") or {}).get("git_sha")
            or "unknown"
            for r in records
        }
    )
    tags = sorted(
        {
            ((r.get("provenance") or {}).get("code_version") or {}).get("freeze_tag")
            or "none"
            for r in records
        }
    )
    summary = {
        "settings": res["settings"],
        "splits": splits,
        "development_only": dev_only,
        "notice": NOTICE_DEV if dev_only else "confirmatory run on frozen code",
        "git_sha": shas,
        "freeze_tags": tags,
        "n_records": len(records),
        "class_counts": res["decisions"]
        .drop_duplicates("task_id")["class"]
        .value_counts()
        .to_dict(),
        "rules": sorted(res["decisions"]["rule"].unique()),
        "runtime": runtime_summary(records),
        "audit_wall_s": round(time.time() - t0, 3),
    }
    (out / "audit_summary.json").write_text(
        json.dumps(RB._sanitize(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {"out": str(out), "n_records": len(records), "development_only": dev_only},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
