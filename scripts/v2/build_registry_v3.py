#!/usr/bin/env python
"""
Build the applicability registry v3 (``impact-mpc-registry/3``) from the
records of the forward-model arms of MPC-Bench v2.

The script applies the frozen admission procedure
(:mod:`impact_pipeline.v2.registry_v3`: FM0, FMa, FMb1, FMb2 for NAS, FMd and
FMabs with curtailment) once, to the records of one purpose of the forward
design (:mod:`impact_pipeline.bench.designs_v2.forward`), and writes one
entry per (principle, estimator version, arm, view) with the flags
``admitted_for_present`` and ``admitted_for_absent``, the record of every
criterion and the regime constraints read from the records.

Inputs

``--results``
    JSON-lines files (or directories of them) of ``mpc-bench-result/3``
    task records; only records of the ``forward`` design are read. Each
    record follows the record contract of the forward design: ``config``
    from ``ForwardTask.to_config``, scorings named by view with the view's
    ``forward_v2.scoring_details`` in ``details``.
``--anchors``
    JSON with the anchor validity of every (arm, view, principle) on its
    reference block: ``{"anchors": [{"arm": .., "view": .., "principle":
    .., "valid": true, ...}]}`` or ``{"<arm>/<view>/<principle>": {"valid":
    ..}}``. A view without an anchor fails FM0.
``--purpose``
    ``confirmatory`` (default; seeds >= 20000 only) or a development dry run
    (``dry_run``, ``dev_regime``; seeds 0-999 only, and the registry is
    labelled as development). Reference and smoke records never enter a
    registry.

Only the primary estimator form of each admission view counts (the v1
quadrant pipeline of IIM is a comparator). Every record must be a task of
the purpose's plan at the plan's regime; a record outside the plan is
refused. The contrasts read each component's ``c`` (for NAS the reported
``c_NAS = min(c_R, c_B)``). Runs whose estimator raised are not
observations. Missing planned tasks are listed in the report (a curtailed
BOLD arm stops early by design).

Example::

    python scripts/v2/build_registry_v3.py --results outputs/v2/forward \\
        --anchors protocols/v2/generated/forward_anchors.json \\
        --run-id mpcbench-freeze-v2:confirmatory:forward \\
        --out protocols/v2/generated/applicability_registry_v3.json \\
        --report outputs/v2/forward/registry_v3_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from impact_pipeline.bench.designs_v2 import forward as D  # noqa: E402
from impact_pipeline.v2 import records as REC  # noqa: E402
from impact_pipeline.v2 import registry_v3 as RV  # noqa: E402
from impact_pipeline.v2 import seeds as S  # noqa: E402

BUILDER_VERSION = "mpc-bench-registry-builder/3.0.0"
REGISTRY_PURPOSES = D.ADMISSION_PURPOSES


class BuildError(ValueError):
    """Records or anchors that cannot build a registry."""


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def result_files(paths: Iterable) -> List[Path]:
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            out.append(p)
        else:
            raise BuildError(f"no such results file or directory: {p}")
    if not out:
        raise BuildError("no result files")
    return out


def read_records(paths: Iterable) -> Tuple[List[REC.TaskRecord], dict]:
    """The forward-design task records of the files, and their SHA-256s."""
    recs, digests = [], {}
    for f in result_files(paths):
        digests[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
        recs.extend(r for r in REC.read_jsonl(f) if r.design == D.DESIGN)
    ids = [r.task_id for r in recs]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise BuildError(f"duplicate task records {dup[:5]}")
    return recs, digests


def load_anchors(payload) -> Dict[Tuple[str, str, str], dict]:
    """``{(arm, view, principle): anchor}`` from either anchors layout."""
    if isinstance(payload, (str, Path)):
        payload = json.loads(Path(payload).read_text(encoding="utf-8"))
    out = {}
    if isinstance(payload, Mapping) and "anchors" in payload:
        for a in payload["anchors"]:
            key = (str(a["arm"]), str(a["view"]), str(a["principle"]))
            out[key] = {k: v for k, v in a.items()
                        if k not in ("arm", "view", "principle")}
    elif isinstance(payload, Mapping):
        for k, v in payload.items():
            parts = str(k).split("/")
            if len(parts) != 3:
                raise BuildError(f"anchor key {k!r} is not <arm>/<view>/<principle>")
            out[tuple(parts)] = (dict(v) if isinstance(v, Mapping)
                                 else {"valid": bool(v)})
    else:
        raise BuildError("anchors must be a JSON object")
    for key, a in out.items():
        if "valid" not in a:
            raise BuildError(f"anchor {key} has no 'valid'")
    return out


# --------------------------------------------------------------------------
# records -> admission runs
# --------------------------------------------------------------------------
def _forward_config(rec: REC.TaskRecord) -> dict:
    cfg = (rec.config or {}).get("forward")
    if not isinstance(cfg, Mapping):
        raise BuildError(f"{rec.task_id}: config['forward'] is missing")
    for key in ("arm", "condition", "purpose", "regime"):
        if key not in cfg:
            raise BuildError(f"{rec.task_id}: config['forward'] lacks {key!r}")
    return cfg


def admission_runs(records: Sequence[REC.TaskRecord], purpose: str):
    """
    ``{(arm, principle, version): [run, ...]}`` of the primary scorings and
    ``{(arm, view): [regime keys of each scoring]}`` and
    ``{(arm, view): substrate}``. Checks the seed policy (one split, the
    purpose's), that every record belongs to ``purpose`` and that it is a
    task of the purpose's plan with the plan's arm, condition, seed and
    regime: runs outside the frozen plan (extra seeds, another regime) never
    enter an admission.
    """
    if purpose not in REGISTRY_PURPOSES:
        raise BuildError(f"a registry is built from {REGISTRY_PURPOSES} records only")
    split = S.CONFIRMATORY if purpose == D.CONFIRMATORY else S.DEVELOPMENT
    if not records:
        raise BuildError("no forward-design records")
    S.check_seeds([r.seed for r in records], split)
    plan = {t.task_id: t for t in D.build_tasks(purpose)}
    runs: Dict[tuple, List[RV.AdmissionRun]] = {}
    regimes: Dict[tuple, List[dict]] = {}
    substrates: Dict[tuple, set] = {}
    for rec in records:
        cfg = _forward_config(rec)
        if cfg["purpose"] != purpose:
            raise BuildError(f"{rec.task_id}: purpose {cfg['purpose']!r}, building "
                             f"{purpose!r}")
        task = plan.get(rec.task_id)
        if task is None:
            raise BuildError(f"{rec.task_id}: not a task of the {purpose} plan")
        got = (str(cfg["arm"]), str(cfg["condition"]), int(rec.seed),
               str(cfg["regime"]))
        if got != (task.arm, task.condition, task.seed, task.regime):
            raise BuildError(f"{rec.task_id}: arm, condition, seed or regime differ "
                             "from the plan")
        if rec.status == REC.TASK_ERROR:
            continue
        arm = str(cfg["arm"])
        for s in rec.scorings:
            if s.estimator_form != D.PRIMARY:
                continue
            det = s.details or {}
            if det.get("view", s.view) != s.view or "regime" not in det:
                raise BuildError(f"{rec.task_id}/{s.scoring_id}: scoring details lack "
                                 "the view's regime keys")
            if det.get("regime_name", task.regime) != task.regime:
                raise BuildError(f"{rec.task_id}/{s.scoring_id}: view observed at the "
                                 f"{det['regime_name']} regime, planned {task.regime}")
            regimes.setdefault((arm, s.view), []).append(dict(det["regime"]))
            if det.get("substrate"):
                substrates.setdefault((arm, s.view), set()).add(str(det["substrate"]))
            for p, comp in s.components.items():
                key = (arm, p, comp.estimator_version)
                runs.setdefault(key, []).append(RV.AdmissionRun(
                    principle=p, view=s.view, condition=str(cfg["condition"]),
                    seed=rec.seed, status=comp.status, reason=comp.reason, c=comp.c,
                    dose=cfg.get("dose")))
    subs = {}
    for k, v in substrates.items():
        if len(v) != 1:
            raise BuildError(f"view {k} has several substrates {sorted(v)}")
        (subs[k],) = v
    return runs, regimes, subs


def plan_differences(records: Sequence[REC.TaskRecord], purpose: str) -> dict:
    """Planned tasks without a record (per arm and listed; a curtailed arm
    stops early by design) and records the plan does not contain."""
    plan = D.build_tasks(purpose)
    planned = {t.task_id: t.arm for t in plan}
    have = {r.task_id for r in records}
    missing = sorted(set(planned) - have)
    by_arm = {arm: sum(1 for t in missing if planned[t] == arm) for arm in D.ARMS}
    return {"n_planned": len(planned), "n_records": len(have), "missing": missing,
            "missing_by_arm": by_arm, "unexpected": sorted(have - set(planned))}


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def build_registry(records: Sequence[REC.TaskRecord], anchors: Mapping, *, run_id: str,
                   purpose: str = D.CONFIRMATORY, version: Optional[str] = None,
                   provenance: Optional[Mapping] = None):
    """The registry v3 of the records and the per-entry report rows."""
    if not str(run_id).strip():
        raise BuildError("a run id is required")
    runs, regimes, subs = admission_runs(records, purpose)
    entries, report = [], []
    for design in D.admission_designs(purpose):
        keys = sorted(k for k in runs
                      if k[0] == design.arm and k[1] == design.principle)
        for _arm, p, ver in keys:
            arm_runs = runs[(_arm, p, ver)]
            src = [r for r in arm_runs if r.view == design.source_view]
            for view in design.views:
                view_runs = [r for r in arm_runs if r.view == view]
                if not view_runs:
                    report.append({"principle": p, "version": ver, "arm": design.arm,
                                   "view": view, "skipped": "no runs"})
                    continue
                stage = regimes[(design.arm, view)][0]["observation_stage"]
                result = RV.evaluate_view(
                    design, view, view_runs, stage=stage,
                    anchor=anchors.get((design.arm, view, p)), source_runs=src)
                entry = RV.build_entry(
                    result, estimator=f"compute_{p}:*", version=ver,
                    substrate=subs[(design.arm, view)],
                    regimes=regimes[(design.arm, view)], run_id=run_id,
                    arm=design.arm, grain=D.entry_grain(p, view),
                    evidence={"purpose": purpose, "builder": BUILDER_VERSION})
                entries.append(entry)
                report.append({"principle": p, "version": ver, "arm": design.arm,
                               "view": view, **result.to_dict()})
    prov = {"builder": BUILDER_VERSION, "run_id": str(run_id), "purpose": purpose,
            "split": S.CONFIRMATORY if purpose == D.CONFIRMATORY else S.DEVELOPMENT,
            "n_records": len(records)}
    prov.update(dict(provenance or {}))
    reg = RV.RegistryV3(entries, version=version, provenance=prov)
    return reg, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", nargs="+", required=True,
                    help="JSON-lines result files or directories")
    ap.add_argument("--anchors", required=True, help="anchor validity JSON")
    ap.add_argument("--run-id", required=True, help="the benchmark run id")
    ap.add_argument("--purpose", default=D.CONFIRMATORY, choices=REGISTRY_PURPOSES)
    ap.add_argument("--version", default=None, help="registry version label")
    ap.add_argument("--out", required=True, help="registry JSON to write")
    ap.add_argument("--report", default=None, help="per-entry criteria report JSON")
    args = ap.parse_args(argv)
    try:
        records, digests = read_records(args.results)
        anchors = load_anchors(args.anchors)
        version = args.version or (args.run_id if args.purpose == D.CONFIRMATORY
                                   else f"development:{args.run_id}")
        reg, report = build_registry(
            records, anchors, run_id=args.run_id, purpose=args.purpose,
            version=version, provenance={"results_sha256": digests})
    except (BuildError, RV.RegistryV3Error, REC.RecordSchemaError,
            S.SeedPolicyError) as exc:
        print(f"build_registry_v3: {exc}", file=sys.stderr)
        return 1
    reg.to_json(args.out)
    if args.report:
        payload = {"registry_sha256": reg.hash(), "plan": plan_differences(
            records, args.purpose), "entries": report}
        Path(args.report).write_text(json.dumps(RV._jsonable(payload), indent=2,
                                                sort_keys=True) + "\n",
                                     encoding="utf-8")
    for e in reg.entries:
        print(f"{e.principle:5s} {e.version:16s} {e.arm:15s} {e.view:13s} "
              f"present={e.admitted_for_present:15s} absent={e.admitted_for_absent:15s}"
              f" failing={','.join(e.failing) or '-'}")
    print(f"registry {args.out}: {len(reg.entries)} entries, sha256 {reg.hash()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
