#!/usr/bin/env python
"""
IIM v5 validation on MPC-Bench family B (v2): the cells of hypotheses
HCv2-2, HCv2-11, HCv2-12 and HCv2-13 (:mod:`impact_pipeline.bench.designs_v2.
family_b`), scored with IIM v5 (``iim-v5-2026.10``) under the status rule
``tost-v2`` on the family-B construct scale.

Every task simulates one cell at one seed and scores it under each of the
cell's declarations in both cut modes (one ``mpc-bench-result/3`` task record
with one scoring per declaration and cut mode; the scorings of a task form
one cluster). Each component carries the estimate, the independence-null
moments and family, ``p_ind`` (rank p-value, in ``details``), the SE and its
method where the hypothesis needs a status, ``c`` on the primary exact anchor
(ring 0.45) with status and reason from the family-B protocol, and in
``details`` the exact target with ``c_exact``, the status under the second
anchor (all-to-all 0.4), the occupancy summary and the cluster id. A
component whose estimator raised is ``UNDEFINED(ESTIMATOR_ERROR:<type>)``
and leaves the other scorings untouched. Every component names its
family-B protocol by its key (``protocol_id`` ``B-<anchor>-<cut mode>-<null
family>[-values]``), the one convention of the evaluator and the integrity
audit (``hypothesis_engine.protocol_key``).

Outputs (``--out``): ``iim_validation_v2.jsonl`` (task records, appended;
the run resumes and skips completed task ids), ``iim_validation_v2_components.csv``
(one row per scoring), ``iim_validation_v2_summary.csv`` (descriptive rates
and medians per cell, scoring and cut mode; the decisions are the
evaluator's), ``iim_validation_v2_exact.csv`` (the exact targets, anchors and
their statuses on the exact path, and the strict monotonicity of the
HCv2-12 (a) sweeps) and ``iim_validation_v2.json`` (design, plan, protocol
hashes and provenance).

Seed policy: a development run uses the development mirrors (seeds 400-439)
and refuses any other seed; a confirmatory run (``--split confirmatory``)
needs the v2 confirmatory guard (clean tree equal to the freeze tag
``mpcbench-freeze-v2``, seeds >= 20000), checked before anything runs.

Example::

    python scripts/v2/iim_validation_v2.py --out outputs/v2_dev/iim_family_b \\
        --hypotheses HCv2-12 --workers 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from impact_pipeline import evidence_v2 as EV  # noqa: E402
from impact_pipeline.bench.designs_v2 import family_b as FB  # noqa: E402
from impact_pipeline.v2 import FREEZE_TAG_V2  # noqa: E402
from impact_pipeline.v2 import hypothesis_engine as HE  # noqa: E402
from impact_pipeline.v2 import iim_v5 as IIM  # noqa: E402
from impact_pipeline.v2 import provenance as PV  # noqa: E402
from impact_pipeline.v2 import reasons as R  # noqa: E402
from impact_pipeline.v2 import records as REC  # noqa: E402
from impact_pipeline.v2 import seeds as S  # noqa: E402

VALIDATION_VERSION = "iim-validation/2.0.0"
RESULTS_JSONL = "iim_validation_v2.jsonl"
COMPONENTS_CSV = "iim_validation_v2_components.csv"
SUMMARY_CSV = "iim_validation_v2_summary.csv"
EXACT_CSV = "iim_validation_v2_exact.csv"
SUMMARY_JSON = "iim_validation_v2.json"
OBSERVATIONS = {"source": "direct", "sensor": "sensor_mixing",
                "source_estimate": "source_estimate", "bold": "hemodynamic"}


# --------------------------------------------------------------------------
# one task
# --------------------------------------------------------------------------
def _params(cell: FB.Cell, scoring: FB.Scoring, override=None) -> IIM.IIMParams:
    kw = {
        "cut_mode": cell.cut_modes[0],
        "report_cut_modes": tuple(cell.cut_modes[1:]),
        "n_null": int(cell.n_null),
        "se_method": IIM.SE_METHOD_DEFAULT if scoring.se else None,
        "null_order": scoring.null_order,
    }
    kw.update(dict(override or {}))
    return IIM.IIMParams.from_mapping(kw)


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _identifiability(scoring: FB.Scoring, stage: str) -> dict:
    return {"shared_inputs": scoring.shared_inputs,
            "observation": OBSERVATIONS[stage], "hub_privileged": "not_tested"}


def _null_family(scoring: FB.Scoring, res: Optional[dict]) -> str:
    fam = (res or {}).get("null_family")
    if fam:
        return fam
    return (IIM.NULL_FAMILY_STRATIFIED if scoring.conditioning == "stratify"
            else IIM.NULL_FAMILY_SHIFT)


def protocol_for(scoring: FB.Scoring, cut: str, res: Optional[dict] = None,
                 anchor: str = FB.PRIMARY_ANCHOR):
    """The family-B protocol of a scoring: its cut mode, null family and
    anchor; value-only scorings (no SE) use the protocol without an SE
    contract, so their status is ``UNDEFINED(NO_SAMPLING_SE)``."""
    return FB.family_b_protocol(cut, anchor=anchor,
                                null_family=_null_family(scoring, res),
                                sampling_se=bool(scoring.se))


def _component(task: FB.Task, scoring: FB.Scoring, res: dict, cut: str, stage: str,
               timing: dict, exact: Optional[dict]) -> REC.ComponentRecord:
    proto = protocol_for(scoring, cut, res)
    ev = IIM.evidence(res, cut, substrate=FB.SUBSTRATE, observation_stage=stage,
                      view=FB.VIEW, protocol_id=HE.protocol_key(proto),
                      bearer_id=task.cluster_id)
    a = EV.assess_item(ev, proto)
    proto2 = protocol_for(scoring, cut, res, anchor=FB.SECOND_ANCHOR)
    a2 = EV.assess_item(ev, proto2)
    fields = IIM.component_fields(res, cut)
    details = dict(fields.pop("details"))
    details.update({
        "hypothesis": task.cell.hypothesis,
        "parts": list(task.cell.parts),
        "cell_id": task.cell.cell_id,
        "cluster_id": task.cluster_id,
        "scoring": scoring.key,
        "role": (scoring.role if task.cell.role == "decisive" else "reported"),
        "anchor": FB.PRIMARY_ANCHOR,
        "anchor_value": FB.anchor_value(cut),
        "exact_target": None if exact is None else exact["value"],
        "c_exact": None if exact is None else exact["c"],
        "exact_target_kind": None if exact is None else exact["target"],
        "second_anchor": {"anchor": FB.SECOND_ANCHOR,
                          "anchor_value": FB.anchor_value(cut, FB.SECOND_ANCHOR),
                          "status": a2.status.value, "reason": a2.reason,
                          "c": _f(a2.c)},
        "expected": dict(task.cell.expected),
        "null_values": ((res.get("cuts") or {}).get(cut) or {}).get("null_values"),
        "route": a.route,
    })
    return REC.ComponentRecord(
        principle=IIM.PRINCIPLE, status=a.status.value, reason=a.reason,
        flags=tuple(a.flags), estimator_version=IIM.ESTIMATOR_VERSION,
        declaration_id=scoring.declaration, observation_stage=stage,
        protocol_id=HE.protocol_key(proto), protocol_hash=proto.hash,
        c=_f(a.c), se_c=_f(a.se), df_c=_f(a.df),
        identifiability=_identifiability(scoring, stage),
        seconds=timing.get("seconds"), load_average=timing.get("load_average"),
        details=details, **fields)


def _error_component(task, scoring, cut, stage, exc, timing) -> REC.ComponentRecord:
    proto = protocol_for(scoring, cut)
    return REC.component_error(
        IIM.PRINCIPLE, exc, estimator_version=IIM.ESTIMATOR_VERSION,
        declaration_id=scoring.declaration, observation_stage=stage,
        protocol_id=HE.protocol_key(proto), protocol_hash=proto.hash,
        identifiability=_identifiability(scoring, stage),
        seconds=timing.get("seconds"), load_average=timing.get("load_average"))


def score_task(task: FB.Task, params_override=None) -> REC.TaskRecord:
    """Simulate a task and score it under every declaration of its cell in
    both cut modes (one scoring each)."""
    cell = task.cell
    t_all, cpu_all = time.perf_counter(), time.process_time()
    with PV.timed() as t_sim:
        sim = FB.simulate(task)
    stage = sim.observation_stage
    scorings = []
    for sc in cell.scorings:
        null_seed, se_seed = task.scoring_seeds(sc)
        extra = {}
        with PV.timed() as t:
            try:
                inputs = FB.scoring_inputs(task, sim, sc)
                extra = {"label_errors": inputs["label_errors"]}
                res = IIM.compute_iim_v5(
                    sim.ts, lag=sim.lag, params=_params(cell, sc, params_override),
                    macro_nodes=sim.macro_nodes, strata=inputs["strata"],
                    basis=inputs["basis"], null_seed=null_seed, se_seed=se_seed,
                    observation_stage=stage)
                err = None
            except Exception as exc:  # noqa: BLE001 - one component, recorded
                res, err = None, exc
        exact = FB.exact_targets(cell, sc)
        for cut in cell.cut_modes:
            if err is not None:
                comp = _error_component(task, sc, cut, stage, err, t)
            else:
                comp = _component(task, sc, res, cut, stage, t, exact.get(cut))
            scorings.append(REC.ScoringRecord(
                scoring_id=f"{sc.key}/{cut}", declaration_id=sc.declaration,
                observation_stage=stage, view=FB.VIEW, estimator_form=cut,
                protocol_id=comp.protocol_id, protocol_hash=comp.protocol_hash,
                components={IIM.PRINCIPLE: comp},
                details={"scoring": sc.to_dict(), "cluster_id": task.cluster_id,
                         "null_seed": int(null_seed), "se_seed": int(se_seed),
                         **extra}))
    ts = np.ascontiguousarray(sim.ts)
    simulation = {"ts_sha256": PV.array_sha256(ts), "raw_ts_sha256": None,
                  "structural_hash": None, "schedule_hash": None,
                  "n_nodes": int(ts.shape[0]), "n_time": int(ts.shape[1]),
                  "dt": float(sim.dt), "seconds": t_sim["seconds"]}
    status = REC.derive_task_status(scorings)
    return REC.TaskRecord(
        task_id=task.task_id, design=FB.DESIGN, family=FB.FAMILY,
        system=cell.system, seed=int(task.seed),
        generator_version=FB.GENERATOR_VERSION, status=status, split=task.split,
        config={"cell": cell.to_dict(), "trajectory_seed": int(task.trajectory_seed),
                "simulation_seed": int(task.simulation_seed),
                "cluster_id": task.cluster_id, "lag": int(sim.lag),
                "simulation_meta": dict(sim.meta),
                "validation_version": VALIDATION_VERSION,
                "design_version": FB.DESIGN_VERSION},
        simulation=simulation, scorings=tuple(scorings),
        timing={"simulation_seconds": t_sim["seconds"],
                "load_average": t_sim["load_average"],
                # the task's wall and CPU time (one BLAS thread): its cost
                "total_s": round(time.perf_counter() - t_all, 4),
                "cpu_s": round(time.process_time() - cpu_all, 4)})


def run_task(task: FB.Task, params_override=None) -> REC.TaskRecord:
    """:func:`score_task` with estimator logging and warnings silenced; a
    task that fails outside the estimators becomes an ``error`` record."""
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return score_task(task, params_override)
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                return REC.TaskRecord(
                    task_id=task.task_id, design=FB.DESIGN, family=FB.FAMILY,
                    system=task.cell.system, seed=int(task.seed),
                    generator_version=FB.GENERATOR_VERSION, status=REC.TASK_ERROR,
                    split=task.split, config={"cell": task.cell.to_dict()},
                    error=f"{type(exc).__name__}: {exc}")
    finally:
        logging.disable(previous)


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def component_table(records: Iterable[REC.TaskRecord]) -> pd.DataFrame:
    """One row per scoring: the flat component row plus the family-B fields
    (hypothesis, parts, cell, cluster, scoring, ``p_ind``, estimator
    definedness and reason, exact target and ``c_exact``, second-anchor
    status)."""
    rows = []
    for rec in records:
        by_id = {s.scoring_id: s for s in rec.scorings}
        for row in rec.component_rows():
            comp = by_id[row["scoring_id"]].components[row["principle"]]
            d = comp.details
            row.update({
                "hypothesis": d.get("hypothesis"),
                "parts": "+".join(d.get("parts") or []),
                "cell_id": d.get("cell_id"),
                "cluster_id": d.get("cluster_id") or rec.task_id,
                "scoring": d.get("scoring"),
                "role": d.get("role"),
                "cut_mode": d.get("cut_mode") or row.get("estimator_form"),
                "p_ind": d.get("p_ind"),
                "estimator_defined": d.get("defined"),
                "estimator_reason": d.get("estimator_reason"),
                "excess": (None if row.get("estimate") is None
                           or row.get("null_mean") is None
                           else row["estimate"] - row["null_mean"]),
                "exact_target": d.get("exact_target"),
                "c_exact": d.get("c_exact"),
                "second_anchor_status": (d.get("second_anchor") or {}).get("status"),
                "second_anchor_c": (d.get("second_anchor") or {}).get("c"),
                "expected": json.dumps(d.get("expected") or {}, sort_keys=True),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def summary_table(components: pd.DataFrame) -> pd.DataFrame:
    """Descriptive rates and medians per (cell, scoring, cut mode): runs,
    estimator-defined runs and their reasons, rank exceedance ``P(p_ind <=
    0.05)`` among defined runs, PRESENT / ABSENT rates, medians of the excess,
    ``c`` and ``se_c``, and ``c_exact``."""
    if components.empty:
        return pd.DataFrame()

    def med(col):
        v = pd.to_numeric(col, errors="coerce").dropna()
        return float(v.median()) if len(v) else float("nan")

    rows = []
    keys = ["hypothesis", "parts", "cell_id", "scoring", "cut_mode", "role"]
    for key, g in components.groupby(keys, sort=True, dropna=False):
        defined = g[g["estimator_defined"] == True]  # noqa: E712
        p = pd.to_numeric(defined["p_ind"], errors="coerce").dropna()
        reasons = g.loc[g["status"] == R.UNDEFINED, "reason"].fillna("").value_counts()
        rows.append({
            **dict(zip(keys, key)),
            "n": int(len(g)),
            "n_estimator_defined": int(len(defined)),
            "rank_exceedance": float((p <= 0.05).mean()) if len(p) else float("nan"),
            "present_rate": float((g["status"] == R.PRESENT).mean()),
            "absent_rate": float((g["status"] == R.ABSENT).mean()),
            "median_excess": med(g["excess"]),
            "median_c": med(g["c"]),
            "median_se_c": med(g["se_c"]),
            "c_exact": med(g["c_exact"]),
            "undefined_reasons": json.dumps(reasons.to_dict(), sort_keys=True),
        })
    return pd.DataFrame(rows)


def exact_table(hypotheses=None, n_min: int = IIM.N_MIN_DEFAULT) -> pd.DataFrame:
    """The exact (CD-1) table: per family-B cell, scoring target and cut mode
    the exact ``Delta_Psi`` and ``c_exact`` with its status on the exact path
    of the family-B protocol (SE method ``exact``), the anchors, and the
    strict monotonicity of the HCv2-12 (a) sweeps."""
    rows = []
    for anchor in (FB.PRIMARY_ANCHOR, FB.SECOND_ANCHOR):
        for cut in FB.CUT_MODES:
            rows.append({"row": "anchor", "anchor": anchor, "cut_mode": cut,
                         "exact": FB.anchor_value(cut, anchor)})
    seen = set()
    for cell in FB.cells(hypotheses, n_min):
        for sc in cell.scorings:
            for cut, ex in FB.exact_targets(cell, sc).items():
                if ex is None:
                    continue
                key = (cell.system_key, ex["target"], cut)
                if key in seen:
                    continue
                seen.add(key)
                proto = FB.family_b_protocol(cut)
                a = EV.assess_item(IIM.evidence(IIM.exact_result(ex["value"], cut), cut),
                                   proto)
                rows.append({"row": "cell", "system": cell.system_key,
                             "target": ex["target"], "cut_mode": cut,
                             "exact": ex["value"], "c_exact": ex["c"],
                             "exact_status": a.status.value, "exact_reason": a.reason,
                             "route": a.route})
    mono = FB.sweep_monotonicity()
    for kind, row in mono.items():
        for cut in FB.CUT_MODES:
            rows.append({"row": "sweep", "system": kind, "cut_mode": cut,
                         "strictly_monotone": row[cut]["strictly_monotone"],
                         "levels": json.dumps(row["levels"]),
                         "values": json.dumps(row[cut]["values"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def _done(path: Path) -> set:
    out = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") != REC.TASK_ERROR:
                    out.add(rec["task_id"])
    return out


def _latest(path: Path) -> List[REC.TaskRecord]:
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = REC.loads(line)
                latest[rec.task_id] = rec
    return [latest[k] for k in sorted(latest)]


def run(out_dir, *, split: str = S.DEVELOPMENT, hypotheses=None, seeds=None,
        workers: int = 1, freeze_tag: str = FREEZE_TAG_V2, limit: Optional[int] = None,
        task_list: Optional[List[FB.Task]] = None, params_override=None,
        write_exact: bool = True) -> dict:
    """Run (or resume) the family-B validation: the seed policy and, for a
    confirmatory run, the v2 confirmatory guard are checked before anything
    is simulated. ``task_list`` replaces the design's tasks (tests);
    ``params_override`` changes IIM settings (outside the contract; tests)."""
    if split not in S.SPLITS:
        raise ValueError(f"split must be one of {S.SPLITS}")
    todo_all = (FB.tasks(split, hypotheses=hypotheses, seeds=seeds)
                if task_list is None else list(task_list))
    if not todo_all:
        raise ValueError("no tasks selected")
    seeds_all = sorted({t.seed for t in todo_all})
    if any(t.split != split for t in todo_all):
        raise S.SeedPolicyError("task split differs from the run's split")
    if split == S.CONFIRMATORY:
        provenance = PV.run_provenance(seeds=seeds_all, confirmatory=True,
                                       freeze_tag=freeze_tag)
    else:
        S.assert_development(seeds_all)
        provenance = PV.run_provenance(seeds=seeds_all)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / RESULTS_JSONL
    done = _done(jsonl)
    todo = [t for t in todo_all if t.task_id not in done]
    if limit is not None:
        todo = todo[: int(limit)]
    t0 = time.time()
    with open(jsonl, "a", encoding="utf-8") as fh:
        def emit(rec):
            fh.write(REC.dumps(rec) + "\n")
            fh.flush()

        if int(workers) > 1 and len(todo) > 1:
            with ProcessPoolExecutor(max_workers=int(workers)) as ex:
                futs = [ex.submit(run_task, t, params_override) for t in todo]
                for f in as_completed(futs):
                    emit(f.result())
        else:
            for t in todo:
                emit(run_task(t, params_override))
    records = _latest(jsonl)
    comps = component_table(records)
    comps.to_csv(out / COMPONENTS_CSV, index=False)
    summary = summary_table(comps)
    summary.to_csv(out / SUMMARY_CSV, index=False)
    if write_exact:
        exact_table(hypotheses).to_csv(out / EXACT_CSV, index=False)
    protocols = {}
    for cut in FB.CUT_MODES:
        for fam in IIM.NULL_FAMILIES:
            for anchor in (FB.PRIMARY_ANCHOR, FB.SECOND_ANCHOR):
                for with_se in (True, False):
                    p = FB.family_b_protocol(cut, anchor=anchor, null_family=fam,
                                             sampling_se=with_se)
                    protocols[HE.protocol_key(p)] = {"name": p.name, "hash": p.hash}
    meta = {
        "version": VALIDATION_VERSION,
        "design": FB.DESIGN,
        "design_version": FB.DESIGN_VERSION,
        "estimator_version": IIM.ESTIMATOR_VERSION,
        "split": split,
        "hypotheses": list(FB.HYPOTHESES if hypotheses is None else hypotheses),
        "plan": FB.plan(split),
        "n_tasks_selected": len(todo_all),
        "n_run": len(todo),
        "n_records": len(records),
        "n_task_errors": sum(r.status == REC.TASK_ERROR for r in records),
        "seconds": round(time.time() - t0, 2),
        "protocols": protocols,
        "params_override": dict(params_override or {}),
        "provenance": provenance,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / SUMMARY_JSON).write_text(json.dumps(meta, indent=2, sort_keys=True,
                                               default=str), encoding="utf-8")
    return {"summary": meta, "records": records, "components": comps,
            "table": summary}


def _ints(text) -> Optional[tuple]:
    if text is None:
        return None
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return tuple(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=False, default=None)
    ap.add_argument("--split", default=S.DEVELOPMENT, choices=S.SPLITS)
    ap.add_argument("--hypotheses", default=None,
                    help=f"comma-separated subset of {','.join(FB.HYPOTHESES)}")
    ap.add_argument("--seeds", default=None,
                    help="seed subset (inside the split's blocks), e.g. 400-404")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="run at most N tasks")
    ap.add_argument("--freeze-tag", default=FREEZE_TAG_V2)
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args(argv)
    hyps = (None if not args.hypotheses else
            tuple(h.strip() for h in args.hypotheses.split(",") if h.strip()))
    if args.list:
        print(json.dumps(FB.plan(args.split), indent=2))
        return 0
    if not args.out:
        ap.error("--out is required")
    try:
        res = run(args.out, split=args.split, hypotheses=hyps, seeds=_ints(args.seeds),
                  workers=args.workers, freeze_tag=args.freeze_tag, limit=args.limit)
    except (PV.ConfirmatoryGuardError, S.SeedPolicyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    table = res["table"]
    if not table.empty:
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            print(table[["cell_id", "scoring", "cut_mode", "n", "n_estimator_defined",
                         "rank_exceedance", "present_rate", "absent_rate",
                         "median_c", "c_exact"]].to_string(index=False))
    return 0 if res["summary"]["n_task_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
