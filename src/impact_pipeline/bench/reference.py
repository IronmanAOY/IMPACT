"""
Reference anchor of the MPC-Bench construct scale (spec v2, V2-2).

The construct scale ``c = (m - nu) / (rho - nu)`` needs a reference anchor
``rho`` besides the null anchor ``nu``. The pipeline takes it from the
cohort's high state; the bench has no cohort, so its anchor is the **nominal
positive control** (witness ``PC_nominal``: every mechanism at its nominal
dose) of one family, averaged over development seeds, and recorded in the
bench protocol as an ``external`` reference (``evidence.Protocol``):

- scale ``excess`` (default): ``values[P]`` is the mean over the reference
  seeds of the positive control's excess ``m - nu`` over its own null, so a
  system's ``c`` is its excess relative to the positive control's excess (as
  the pipeline's cohort reference; no ``INVALID_ANCHORS`` when a system's
  null mean exceeds the positive control's raw estimate);
- scale ``estimate``: ``values[P]`` is the mean raw estimate ``m`` of the
  positive control (the audit's anchor), ``c = (m - nu) / (rho - nu)`` with
  the evaluated system's own ``nu``.

``se[P]`` is the SD of the per-seed values over ``sqrt(n)`` (``n >= 2``). A
principle with fewer than ``min_n`` finite values gets no anchor: its
evidence is then UNDEFINED (``INVALID_ANCHORS``), never judged against a
made-up reference. A mean excess ``<= 0`` is recorded as computed; the
evidence layer then reports ``INVALID_ANCHORS`` for that principle.

Reference seeds are development seeds (0-999; by convention the block
``REFERENCE_SEED_BLOCK`` = 900-999, which the other development designs do not
use by default) of the development families (A); family C and confirmatory
seeds are refused, so the anchor never touches held-out data.

CLI (``python -m impact_pipeline.bench.reference`` or
``scripts/bench_reference.py``)::

    # run the positive control on reference seeds (resumable) and write the
    # protocol with its reference
    python scripts/bench_reference.py --run --family A --seeds 900-907 \\
        --null-surrogates 19 --se-groups 5 --workers 4 \\
        --work-dir outputs/bench/reference_A \\
        --template protocols/mpc_bench_v1.json --out protocols/mpc_bench_v1.json
    # or from existing run_bench results of PC_nominal
    python scripts/bench_reference.py --results outputs/bench/reference_A \\
        --template protocols/mpc_bench_v1.json --out protocols/mpc_bench_v1.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from impact_pipeline.bench.generators import PRINCIPLES

REFERENCE_WITNESS = "PC_nominal"
REFERENCE_SEED_BLOCK = (900, 999)
REFERENCE_FAMILIES = ("A",)
SCALES = ("excess", "estimate")
REFERENCE_SUMMARY = "reference_summary.json"


def _mode_key(principle: str) -> str:
    return "update" if principle == "RAM" else "mode"


def reference_tasks(
    seeds: Iterable[int], family: str = "A", config: Optional[dict] = None
) -> list:
    """Positive-control tasks (``PC_nominal``) on development seeds only."""
    from impact_pipeline.bench.run_bench import seed_set, witness_tasks

    family = str(family).upper()
    if family not in REFERENCE_FAMILIES:
        raise ValueError(
            f"the reference anchor uses development families {REFERENCE_FAMILIES} "
            f"only (family {family} is held out)"
        )
    seeds = [int(s) for s in seeds]
    bad = [s for s in seeds if seed_set(s) != "dev"]
    if bad:
        raise ValueError(f"reference seeds must be development seeds (0-999): {bad}")
    return witness_tasks(
        seeds, family=family, witness_ids=[REFERENCE_WITNESS], config=config
    )


def _reference_records(records: Iterable[dict]) -> List[dict]:
    rows = [
        r
        for r in records
        if r.get("status", "ok") == "ok" and r.get("witness_id") == REFERENCE_WITNESS
    ]
    if not rows:
        raise ValueError(f"no successful {REFERENCE_WITNESS} records")
    held = sorted(
        {
            r.get("task_id")
            for r in rows
            if r.get("split", "development") != "development"
        }
    )
    if held:
        raise ValueError(f"confirmatory/held-out records cannot anchor: {held}")
    return rows


def check_modes(records: Sequence[dict], estimators: Optional[dict]) -> None:
    """Refuse records whose estimator modes differ from the protocol's."""
    for p, opts in dict(estimators or {}).items():
        key = _mode_key(p)
        if key not in (opts or {}):
            continue
        want = opts[key]
        for r in records:
            got = ((r.get("estimator_modes") or {}).get(p) or {}).get(key)
            if got is not None and got != want:
                raise ValueError(
                    f"{r.get('task_id')}: {p} {key}={got!r}, protocol declares "
                    f"{want!r}"
                )


def reference_from_records(
    records: Iterable[dict],
    scale: str = "excess",
    principles: Sequence[str] = PRINCIPLES,
    min_n: int = 2,
) -> tuple:
    """
    External reference (``evidence.Protocol`` layout) from positive-control
    records, and a summary (per principle: n, mean, sd, the seeds used).
    Returns ``(reference, summary)``.
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    rows = _reference_records(records)
    values, ses, per = {}, {}, {}
    for p in principles:
        vals = []
        for r in rows:
            c = (r.get("components") or {}).get(p) or {}
            if not c.get("defined"):
                continue
            m = _f(c.get("estimate"))
            v = m - _f(c.get("null_mean")) if scale == "excess" else m
            if math.isfinite(v):
                vals.append(v)
        arr = np.asarray(vals, dtype=float)
        n = int(arr.size)
        per[p] = {"n": n}
        if n < int(min_n):
            per[p]["anchor"] = f"none (fewer than {int(min_n)} finite values)"
            continue
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1))
        values[p] = mean
        ses[p] = sd / math.sqrt(n)
        per[p].update(mean=mean, sd=sd, se=ses[p])
    if not values:
        raise ValueError("no principle has a finite reference value")
    seeds = sorted(int(r.get("seed")) for r in rows)
    null_k = sorted({int(r.get("null_surrogates", 0)) for r in rows})
    se_groups = sorted({int(r.get("se_groups", 0)) for r in rows})
    families = sorted({str(r.get("family")) for r in rows})
    code = sorted(
        {
            str(((r.get("provenance") or {}).get("code_version") or {}).get("git_sha"))
            for r in rows
        }
    )
    source = (
        f"MPC-Bench {REFERENCE_WITNESS}, family {','.join(families)}, "
        f"development seeds {seeds[0]}-{seeds[-1]} (n={len(seeds)}), "
        f"null_surrogates={','.join(map(str, null_k))}, "
        f"se_groups={','.join(map(str, se_groups))}, scale {scale}, "
        f"bench {rows[0].get('bench_version')}, "
        f"generators {rows[0].get('generator_version')}, code {','.join(code)}; "
        + ", ".join(f"{p} n={per[p]['n']}" for p in principles)
    )
    reference = {
        "kind": "external",
        "scale": scale,
        "values": values,
        "se": ses,
        "source": source,
    }
    summary = {
        "witness": REFERENCE_WITNESS,
        "families": families,
        "seeds": seeds,
        "scale": scale,
        "min_n": int(min_n),
        "null_surrogates": null_k,
        "se_groups": se_groups,
        "per_principle": per,
        "code_git_sha": code,
    }
    return reference, summary


def protocol_with_reference(template, reference: dict, name: Optional[str] = None):
    """The template protocol (``evidence.Protocol``, dict or JSON path) with
    ``reference`` (and ``name``) replaced."""
    from impact_pipeline import evidence as ev

    proto = ev.resolve_protocol(template)
    if proto is None:
        raise ValueError("a template protocol is required")
    changes = {"reference": reference}
    if name is not None:
        changes["name"] = name
    return proto.replace(**changes)


def _f(v) -> float:
    try:
        return float("nan") if v is None else float(v)
    except (TypeError, ValueError):
        return float("nan")


def _read_records(results_dir) -> List[dict]:
    from impact_pipeline.bench.run_bench import RESULTS_JSONL

    latest = {}
    for path in sorted(Path(results_dir).rglob(RESULTS_JSONL)):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    latest[rec["task_id"]] = rec
    return [latest[k] for k in sorted(latest)]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bench_reference.py", description=__doc__.split("\n\n")[0]
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", action="store_true", help="run the positive control")
    src.add_argument("--results", default=None, help="existing run_bench results")
    ap.add_argument("--family", default="A")
    ap.add_argument("--seeds", default="900-907")
    ap.add_argument("--null-surrogates", type=int, default=19)
    ap.add_argument("--se-groups", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--config", default=None, help="generator config (JSON/path)")
    ap.add_argument("--params", default=None, help="estimator overrides (JSON/path)")
    ap.add_argument("--work-dir", default=None, help="run_bench output (--run)")
    ap.add_argument("--scale", default="excess", choices=SCALES)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--template", required=True, help="protocol JSON template")
    ap.add_argument("--out", required=True, help="protocol JSON to write")
    ap.add_argument("--name", default=None, help="protocol name")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    from impact_pipeline import evidence as ev
    from impact_pipeline.bench import run_bench as RB

    args = build_parser().parse_args(argv)
    template = ev.resolve_protocol(args.template)
    if args.run:
        if not args.work_dir:
            raise SystemExit("--work-dir is required with --run")
        tasks = reference_tasks(
            RB.parse_seeds(args.seeds), args.family, RB._json_arg(args.config)
        )
        RB.check_seed_policy(tasks, confirmatory=False)
        prov = RB.collect_provenance(vars(args))
        RB.run_tasks(
            tasks,
            args.work_dir,
            n_workers=args.workers,
            null_surrogates=args.null_surrogates,
            params=RB._json_arg(args.params),
            provenance=prov,
            markers=False,
            se_groups=args.se_groups,
            protocol=None,
        )
        records = _read_records(args.work_dir)
    else:
        records = _read_records(args.results)
    rows = _reference_records(records)
    check_modes(rows, template.to_dict()["estimators"])
    reference, summary = reference_from_records(
        rows, scale=args.scale, min_n=args.min_n
    )
    proto = protocol_with_reference(template, reference, name=args.name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    proto.to_json(out)
    summary["protocol_hash"] = proto.hash
    summary["protocol_path"] = str(out)
    side = out.with_name(out.stem + "_" + REFERENCE_SUMMARY)
    side.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({"protocol": str(out), "hash": proto.hash,
                      "values": reference["values"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
