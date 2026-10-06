#!/usr/bin/env python
"""
Integrity audit of MPC-Bench v2 records (IA-1 to IA-10; deterministic
checks, not hypotheses).

The audit runs on the confirmatory (or, for testing, development) records
of schema ``mpc-bench-result/3`` before the evaluator. Records are read
leniently (one JSON object per line), so a record that breaks the schema is
reported instead of stopping the audit. Every failure is listed, the
affected task ids are collected in ``excluded_task_ids`` (the evaluator
drops them from every hypothesis), and the audit outcome is reported beside
the tally.

=====  ===============================================================
IA-1   the v1 regression gate (``scripts/v2/regression_gate.py``): its
       report, or a run of it
IA-2   no scoring loses a component because another component raised
       (a scoring carries every principle it declares, as far as the run's
       settings score it; records without the declaration are compared
       with the other scorings of their kind); the component error rate
       is <= 1 % per design; every error is
       ``UNDEFINED(ESTIMATOR_ERROR:<type>)``
IA-3   NAS is ``UNDEFINED(SAMPLING_UNRESOLVED)`` on every BOLD scoring and
       never ABSENT there; NAS is defined (never undefined by the sampling
       gate) on every family-A, C1 and Hopf-source scoring
IA-4   every reason of the vocabulary maps to UNDEFINED; no component
       carries a reason with a decided status (an ABSENT with an
       UNDEFINED reason); no ABSENT on degenerate TPMs (hypersynchrony,
       trapped all-to-all at T = 1000, the v1 quadrant montage); SRPI,
       RAM and PDI on C1 are ``UNDEFINED(NOT_APPLICABLE_OBSERVATION_MODEL)``
       (never PRESENT or ABSENT); no ABSENT under a registry direction that
       is not admitted (the forward arms' admission runs, which are judged
       without a registry because they decide it, are counted apart)
IA-5   no simulation is duplicated across families, seeds or twin
       replicates, or under a different configuration (``sha256(ts)``,
       ``sha256(raw_ts)``); duplicates of one configuration under two
       designs are listed as allowed
IA-6   each record's family-protocol hash equals the frozen protocol; the
       RAM-PE facet and PDI bearer declarations of the records equal the
       protocol's (through the one name map of the evaluator)
IA-7   every record follows the schema; task ids equal the generated plan
       (no missing, unexpected or duplicated task; a task the run skipped by
       a design's curtailment rule, listed in its manifest, and a smoke task
       of a development plan, whose outputs the runner discards, are
       expected to be missing and are reported); the seed policy
       (development 0-999, confirmatory >= 20000, never 10000-19999, one
       split per run)
IA-8   identities: NAS v3 equals the frozen v1 NAS on the v1 geometry to
       <= 1e-12 (report or estimator hook); the stored statuses equal the
       status rule; the three status-rule entry points agree
IA-9   v1 SRPI under the v2 rule: no SRPI ABSENT with ``|c| >= delta``;
       every sub-null outcome (``c <= -delta``) UNDEFINED; the number of
       SRPI ABSENTs is reported
IA-10  twins: one structural hash per network, a different schedule hash
       per replicate, and replicate 0 equal to the witness run bit for bit
=====  ===============================================================

Names (designs, systems, estimator forms, views, configuration keys) are
read from ``protocols/v2/hypotheses_v2.json`` (its vocabulary and field
aliases), the single place where the evaluator binds them.

Exit status: 0 when every check that ran passed, 1 when a check failed, 2
when the records are missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import List, Mapping, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from impact_pipeline.v2 import hypothesis_engine as HE  # noqa: E402
from impact_pipeline.v2 import reasons as R  # noqa: E402
from impact_pipeline.v2 import records as REC  # noqa: E402
from impact_pipeline.v2 import seeds as S  # noqa: E402

AUDIT_VERSION = "mpc-bench-v2-integrity-audit/1.0.0"
PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT_RUN"
MAX_COMPONENT_ERROR_RATE = 0.01
NAS_IDENTITY_TOL = 1e-12
# configuration keys that label a task without changing its simulation (a
# duplicate that differs only in these keys is one configuration under two
# designs, which is allowed)
CONFIG_LABEL_KEYS = (
    "cell_id",
    "sweep_knob",
    "sweep_level",
    "witness_id",
    "label",
    "design",
    "block",
    "plan",
    "replicate",
)
# keys of a v2 runner record's config that label a task (its tags, the
# design module, run settings and versions) rather than configure its system
RUNNER_LABEL_KEYS = (
    "tags",
    "design_module",
    "held_out",
    "null_seed",
    "runner_version",
    "settings",
    "system_generator_version",
    "system_config_sha256",
)
LIST_LIMIT = 50


def _check(cid, status, summary, failures=(), excluded=(), **details) -> dict:
    failures = list(failures)
    return {
        "id": cid,
        "status": status,
        "summary": summary,
        "n_failures": len(failures),
        "failures": failures[:LIST_LIMIT],
        "excluded_task_ids": sorted(set(excluded)),
        "details": details,
    }


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------
# JSON-lines files of a run directory that hold no task records: the v2
# runner's status lines of smoke tasks (their outputs are discarded)
NON_RECORD_FILES = ("smoke.jsonl",)


def _paths(items) -> List[Path]:
    """The record files of the given files and directories (a directory
    contributes its ``*.jsonl`` files except :data:`NON_RECORD_FILES`)."""
    out = []
    for it in items or ():
        p = Path(it)
        if p.is_dir():
            out.extend(f for f in sorted(p.rglob("*.jsonl"))
                       if f.name not in NON_RECORD_FILES)
        elif p.exists():
            out.append(p)
    return out


def read_raw_records(paths) -> tuple:
    """Raw record dicts of JSON-lines files (or directories of them) and the
    lines that are not JSON objects."""
    recs, bad = [], []
    for p in _paths(paths):
        with open(p, "r", encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    bad.append({"file": str(p), "line": n, "error": str(exc)})
                    continue
                if not isinstance(obj, dict):
                    bad.append({"file": str(p), "line": n, "error": "not an object"})
                    continue
                recs.append(obj)
    return recs, bad


def _vocab(spec, key, default=None) -> tuple:
    """The names of a vocabulary entry: a design, family or form may name
    several record values (for example a family-A and a family-C1 design)."""
    v = (spec.get("vocabulary") or {}).get(key, default)
    return tuple(v) if isinstance(v, (list, tuple)) else (v,)


def _fields(spec) -> HE.Fields:
    return HE.Fields(spec.get("fields"), spec.get("derived"))


def _components(records):
    """``(record, scoring, principle, component)`` for every component of
    raw record dicts."""
    for rec in records:
        for s in rec.get("scorings") or ():
            for p, c in (s.get("components") or {}).items():
                if isinstance(c, Mapping):
                    yield rec, s, p, c


def _flat(rec, s, c) -> dict:
    """A component row as the evaluator sees it (for field aliases)."""
    return {
        **{
            k: rec.get(k)
            for k in ("task_id", "design", "family", "system", "seed", "replicate")
        },
        "config": rec.get("config") or {},
        **{
            k: s.get(k)
            for k in (
                "scoring_id",
                "declaration_id",
                "observation_stage",
                "view",
                "estimator_form",
                "protocol_id",
            )
        },
        **c,
    }


def _code(reason):
    return None if reason is None else R.parse_reason(reason)[0]


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------
# IA-1
# --------------------------------------------------------------------------
def ia1_regression_gate(gate_report=None, run: Optional[str] = None) -> dict:
    """The v1 regression gate: a stored report (path or mapping) or a run
    (``run`` = ``quick`` or ``full``)."""
    rep = gate_report
    if isinstance(rep, (str, Path)):
        rep = json.loads(Path(rep).read_text(encoding="utf-8"))
    if rep is None and run:
        from scripts.v2 import regression_gate as G

        rep = G.run_gate(quick=(run == "quick"))
    if rep is None:
        return _check("IA-1", NOT_RUN, "no regression-gate report given")
    failed = [c for c in rep.get("checks") or () if c.get("status") == "FAIL"]
    ok = bool(rep.get("ok")) and not failed
    quick = bool(rep.get("quick"))
    summary = ("v1 regression gate PASS" if ok else "v1 regression gate FAIL") + (
        " (quick: re-runs not included)" if quick else ""
    )
    return _check(
        "IA-1",
        PASS if ok else FAIL,
        summary,
        [{"check": c.get("name"), "summary": c.get("summary")} for c in failed],
        quick=quick,
        version=rep.get("version"),
    )


# --------------------------------------------------------------------------
# IA-2
# --------------------------------------------------------------------------
def _scoring_kind(rec, s) -> tuple:
    """Scorings of one kind carry the same principles (a held-out
    re-scoring under P, Q or J scores NAS and IIM only; a secondary
    estimator form one principle)."""
    return (
        rec.get("design"),
        rec.get("family"),
        s.get("declaration_id"),
        s.get("estimator_form"),
        s.get("observation_stage"),
        s.get("view"),
    )


def _declared_principles(rec, s) -> Optional[set]:
    """The principles a scoring declares (the runner writes them to the
    scoring's ``details['principles']``), restricted to those the run's
    settings score (``config['settings']['principles']``); None for a
    record that does not declare them."""
    declared = (s.get("details") or {}).get("principles")
    if not isinstance(declared, (list, tuple)):
        return None
    out = {str(p) for p in declared}
    wanted = ((rec.get("config") or {}).get("settings") or {}).get("principles")
    if isinstance(wanted, (list, tuple)):
        out &= {str(p) for p in wanted}
    return out


def ia2_component_isolation(records, max_rate=MAX_COMPONENT_ERROR_RATE) -> dict:
    # a scoring is compared with what it declares; a record without the
    # declaration with the other scorings of its kind (a held-out re-scoring
    # declares NAS and IIM, a null kind its applicable principles, a lesion
    # run of the Hopf arm NAS alone)
    expected = {}
    for rec, s, p, c in _components(records):
        expected.setdefault(_scoring_kind(rec, s), set()).add(p)
    fails, excl = [], set()
    per_design = Counter()
    errors = Counter()
    for rec in records:
        if rec.get("status") == REC.TASK_ERROR:
            fails.append(
                {
                    "task_id": rec.get("task_id"),
                    "kind": "task_error",
                    "error": rec.get("error"),
                }
            )
            excl.add(rec.get("task_id"))
        for s in rec.get("scorings") or ():
            comps = s.get("components") or {}
            key = _scoring_kind(rec, s)
            has_error = False
            for p, c in comps.items():
                per_design[rec.get("design")] += 1
                code = _code(c.get("reason"))
                if code == R.ESTIMATOR_ERROR:
                    has_error = True
                    errors[rec.get("design")] += 1
                    if c.get("status") != R.UNDEFINED or not R.is_reason(
                        c.get("reason")
                    ):
                        fails.append(
                            {
                                "task_id": rec.get("task_id"),
                                "principle": p,
                                "kind": "malformed_error",
                                "status": c.get("status"),
                                "reason": c.get("reason"),
                            }
                        )
                        excl.add(rec.get("task_id"))
            declared = _declared_principles(rec, s)
            want = declared if declared is not None else expected.get(key, set())
            missing = sorted(want - set(comps))
            if missing and has_error:
                fails.append(
                    {
                        "task_id": rec.get("task_id"),
                        "scoring_id": s.get("scoring_id"),
                        "kind": "lost_components",
                        "missing": missing,
                    }
                )
                excl.add(rec.get("task_id"))
            elif missing:
                fails.append(
                    {
                        "task_id": rec.get("task_id"),
                        "scoring_id": s.get("scoring_id"),
                        "kind": "missing_components",
                        "missing": missing,
                    }
                )
                excl.add(rec.get("task_id"))
    rates = {str(d): errors[d] / n for d, n in per_design.items() if n}
    high = {d: r for d, r in rates.items() if r > max_rate}
    for d, r in sorted(high.items()):
        fails.append({"design": d, "kind": "error_rate", "rate": r})
    status = FAIL if fails else PASS
    return _check(
        "IA-2",
        status,
        f"component errors per design (max {max_rate:.0%}); "
        f"{sum(errors.values())} errors",
        fails,
        excl,
        error_rates=rates,
    )


# --------------------------------------------------------------------------
# IA-3
# --------------------------------------------------------------------------
NAS_UNDEFINED_BY_CONSTRUCTION = (
    R.SAMPLING_UNRESOLVED,
    R.NOT_DEFINED,
    R.NON_FINITE_ESTIMATE,
    R.INSUFFICIENT_TIMEPOINTS,
)


def ia3_nas_sampling(records, spec) -> dict:
    fam_a, fam_c = _vocab(spec, "family.A", "A"), _vocab(spec, "family.C1", "C")
    hopf = _vocab(spec, "design.whole_brain", "whole_brain")
    primary = _vocab(spec, "form.primary", "primary")
    fails, excl, n_bold, n_defined = [], set(), 0, 0
    for rec, s, p, c in _components(records):
        if p != "NAS":
            continue
        stage = s.get("observation_stage")
        code = _code(c.get("reason"))
        if stage == "bold":
            n_bold += 1
            if c.get("status") != R.UNDEFINED or code != R.SAMPLING_UNRESOLVED:
                fails.append(
                    {
                        "task_id": rec.get("task_id"),
                        "scoring_id": s.get("scoring_id"),
                        "kind": "bold_not_sampling_unresolved",
                        "status": c.get("status"),
                        "reason": c.get("reason"),
                    }
                )
                excl.add(rec.get("task_id"))
            continue
        bench = rec.get("family") in fam_a + fam_c and stage == "source"
        hopf_source = rec.get("design") in hopf and stage == "source"
        if (bench or hopf_source) and s.get("estimator_form") in primary + (None,):
            n_defined += 1
            if code in NAS_UNDEFINED_BY_CONSTRUCTION or (
                _f(c.get("estimate")) is None and code != R.ESTIMATOR_ERROR
            ):
                fails.append(
                    {
                        "task_id": rec.get("task_id"),
                        "scoring_id": s.get("scoring_id"),
                        "kind": "nas_undefined_on_bench",
                        "reason": c.get("reason"),
                    }
                )
                excl.add(rec.get("task_id"))
    return _check(
        "IA-3",
        FAIL if fails else PASS,
        f"NAS on {n_bold} BOLD scorings and {n_defined} family-A/C1/Hopf-source "
        "scorings",
        fails,
        excl,
    )


# --------------------------------------------------------------------------
# IA-4
# --------------------------------------------------------------------------
def _admitted(registry_rows, principle, view, direction) -> Optional[bool]:
    for r in registry_rows or ():
        if r.get("principle") == principle and r.get("view") == view:
            val = r.get(f"admitted_for_{direction}")
            return val in ("yes", "vacuous")
    return None


def ia4_undefined_never_absent(records, spec, registry=None) -> dict:
    fails, excl = [], set()
    vocab_bad = [code for code in R.REASONS if R.REASONS[code].status != R.UNDEFINED]
    for code in vocab_bad:
        fails.append({"kind": "reason_not_undefined", "reason": code})
    fields = _fields(spec)
    fam_c = _vocab(spec, "family.C1", "C")
    quadrant = _vocab(spec, "form.iim_v1_quadrant", "iim_v1_quadrant")
    fam_b = _vocab(spec, "family.B", "B")
    # the forward arms' records are the admission runs: they are judged
    # without a registry (their scoring is the registry's test)
    admission = set(
        _vocab(spec, "design.whole_brain", "whole_brain")
        + _vocab(spec, "design.forward", "forward_family_a")
        + _vocab(spec, "design.forward_anchor_replication",
                 "forward_anchor_replication")
    )
    reg = HE.registry_rows(registry) if registry is not None else None
    n = 0
    n_admission = 0
    for rec, s, p, c in _components(records):
        n += 1
        tid = rec.get("task_id")
        st, reason = c.get("status"), c.get("reason")
        where = {"task_id": tid, "scoring_id": s.get("scoring_id"), "principle": p}
        if st not in R.STATUSES:
            fails.append({**where, "kind": "unknown_status", "status": st})
            excl.add(tid)
            continue
        if st != R.UNDEFINED and reason is not None:
            fails.append({**where, "kind": f"{st}_with_reason", "reason": reason})
            excl.add(tid)
        if st == R.UNDEFINED and not R.is_reason(reason):
            fails.append(
                {**where, "kind": "reason_not_in_vocabulary", "reason": reason}
            )
            excl.add(tid)
        row = _flat(rec, s, c)
        if st == R.ABSENT and p == "IIM":
            degenerate = (
                rec.get("system") == "O_hypersynchronous"
                or s.get("estimator_form") in quadrant
                or (
                    rec.get("family") in fam_b
                    and fields.get(row, "@b_network") == "all_to_all"
                    and HE._finite(fields.get(row, "@b_T")) == 1000
                )
            )
            if degenerate:
                fails.append({**where, "kind": "absent_on_degenerate_tpm"})
                excl.add(tid)
        if rec.get("family") in fam_c and p in ("SRPI", "RAM", "PDI"):
            if st != R.UNDEFINED or _code(reason) != R.NOT_APPLICABLE_OBSERVATION_MODEL:
                fails.append(
                    {
                        **where,
                        "kind": "c1_not_applicable_violated",
                        "status": st,
                        "reason": reason,
                    }
                )
                excl.add(tid)
        mixed = s.get("observation_stage") in ("sensor", "source_estimate", "bold")
        if mixed and rec.get("design") in admission:
            n_admission += 1
            continue
        if st in (R.ABSENT, R.PRESENT) and mixed and reg is not None:
            ok = _admitted(reg, p, s.get("view"), st.lower())
            if ok is False or ok is None:
                fails.append(
                    {
                        **where,
                        "kind": f"{st}_without_admitted_direction",
                        "view": s.get("view"),
                    }
                )
                excl.add(tid)
    return _check(
        "IA-4",
        FAIL if fails else PASS,
        f"{len(R.REASONS)} reasons map to UNDEFINED; {n} components checked",
        fails,
        excl,
        registry_checked=reg is not None,
        n_admission_run_components=n_admission,
    )


# --------------------------------------------------------------------------
# IA-5
# --------------------------------------------------------------------------
def _structural_config(cfg) -> str:
    """The configuration a task specifies, without its labels."""
    d = {
        k: v
        for k, v in dict(cfg or {}).items()
        if k not in CONFIG_LABEL_KEYS + RUNNER_LABEL_KEYS
    }
    return json.dumps(d, sort_keys=True, default=str)


def _configuration(rec) -> str:
    """The configuration identity of a record: the realised configuration the
    v2 runner records (``config.system_config_sha256``; the catalogue's
    PC_nominal and a sweep's nominal level are one configuration), else the
    specified one."""
    cfg = rec.get("config") or {}
    return cfg.get("system_config_sha256") or _structural_config(cfg)


def ia5_duplicates(records) -> dict:
    groups = OrderedDict()
    for rec in records:
        sim = rec.get("simulation") or {}
        for kind in ("ts_sha256", "raw_ts_sha256"):
            h = sim.get(kind)
            if h:
                groups.setdefault((kind, h), []).append(rec)
    fails, allowed, excl = [], [], set()
    for (kind, h), recs in groups.items():
        ids = sorted({r.get("task_id") for r in recs})
        if len(ids) < 2:
            continue
        fams = sorted({str(r.get("family")) for r in recs})
        seeds = sorted({r.get("seed") for r in recs}, key=str)
        reps = sorted({r.get("replicate", 0) for r in recs}, key=str)
        cfgs = {_configuration(r) for r in recs}
        entry = {"hash_kind": kind, "hash": h, "task_ids": ids, "families": fams}
        if len(fams) > 1:
            fails.append({**entry, "kind": "cross_family"})
        elif len(seeds) > 1:
            fails.append({**entry, "kind": "different_seeds", "seeds": seeds})
        elif len(reps) > 1:
            fails.append(
                {**entry, "kind": "identical_twin_replicates", "replicates": reps}
            )
        elif len(cfgs) > 1:
            fails.append({**entry, "kind": "different_configuration"})
        else:
            allowed.append(entry)
            continue
        excl.update(ids)
    return _check(
        "IA-5",
        FAIL if fails else PASS,
        f"{len(fails)} duplicate groups, {len(allowed)} allowed (one configuration "
        "under several designs)",
        fails,
        excl,
        allowed=allowed[:LIST_LIMIT],
    )


# --------------------------------------------------------------------------
# IA-6
# --------------------------------------------------------------------------
def frozen_hashes(protocols: Mapping) -> dict:
    """``{protocol_id: hash}`` of loaded protocols (``ProtocolV3`` or
    mappings with a ``hash``)."""
    out = {}
    for pid, pr in protocols.items():
        out[str(pid)] = (
            pr.get("hash") if isinstance(pr, Mapping) else getattr(pr, "hash")
        )
    return out


def _estimator_block(pr, principle):
    est = (
        pr.get("estimators")
        if isinstance(pr, Mapping)
        else getattr(pr, "estimators", None)
    )
    return dict((est or {}).get(principle) or {})


def ia6_protocol_hashes(records, protocols: Optional[Mapping]) -> dict:
    if not protocols:
        return _check("IA-6", NOT_RUN, "no frozen protocols given")
    frozen = frozen_hashes(protocols)
    fails, excl = [], set()
    for rec in records:
        for s in rec.get("scorings") or ():
            comps = s.get("components") or {}
            items = list(comps.items()) or [(None, s)]
            for p, c in items:
                pid = c.get("protocol_id", s.get("protocol_id"))
                ph = c.get("protocol_hash", s.get("protocol_hash"))
                want = frozen.get(str(pid))
                if want != ph:
                    fails.append(
                        {
                            "task_id": rec.get("task_id"),
                            "scoring_id": s.get("scoring_id"),
                            "principle": p,
                            "kind": "hash_mismatch",
                            "protocol_id": pid,
                            "protocol_hash": ph,
                            "frozen_hash": want,
                        }
                    )
                    excl.add(rec.get("task_id"))
                    continue
                pr = protocols.get(str(pid))
                det = (c.get("details") or {}) if p else {}
                if p == "RAM" and "facets_not_applicable" in det:
                    want_f = HE.ram_facets_for_estimator(
                        _estimator_block(pr, "RAM").get("facets_not_applicable") or {}
                    )
                    if dict(det["facets_not_applicable"]) != want_f:
                        fails.append(
                            {
                                "task_id": rec.get("task_id"),
                                "principle": p,
                                "kind": "ram_facets_differ_from_protocol",
                            }
                        )
                        excl.add(rec.get("task_id"))
                if p == "PDI" and "bearer" in det:
                    want_b = HE.pdi_params_for_estimator(
                        _estimator_block(pr, "PDI")
                    ).get("bearer")
                    if want_b is not None and det["bearer"] != want_b:
                        fails.append(
                            {
                                "task_id": rec.get("task_id"),
                                "principle": p,
                                "kind": "pdi_bearer_differs_from_protocol",
                                "record": det["bearer"],
                                "protocol": want_b,
                            }
                        )
                        excl.add(rec.get("task_id"))
    return _check(
        "IA-6",
        FAIL if fails else PASS,
        f"family-protocol hashes against {len(frozen)} frozen protocols",
        fails,
        excl,
    )


# --------------------------------------------------------------------------
# IA-7
# --------------------------------------------------------------------------
def ia7_plan_and_seeds(
    records,
    planned_task_ids=None,
    split: Optional[str] = None,
    bad_lines=(),
    curtailed_task_ids=(),
    smoke_task_ids=(),
) -> dict:
    fails, excl = [], set()
    for b in bad_lines:
        fails.append({"kind": "unreadable_line", **b})
    for rec in records:
        try:
            REC.TaskRecord.from_dict(rec)
        except (ValueError, TypeError) as exc:
            fails.append(
                {"task_id": rec.get("task_id"), "kind": "schema", "error": str(exc)}
            )
            excl.add(rec.get("task_id"))
    seeds = []
    for rec in records:
        try:
            sp = S.split_of(rec.get("seed"))
        except S.SeedPolicyError as exc:
            fails.append(
                {
                    "task_id": rec.get("task_id"),
                    "kind": "seed_policy",
                    "error": str(exc),
                }
            )
            excl.add(rec.get("task_id"))
            continue
        seeds.append(rec.get("seed"))
        if rec.get("split") is not None and rec.get("split") != sp:
            fails.append(
                {
                    "task_id": rec.get("task_id"),
                    "kind": "split_label",
                    "split": rec.get("split"),
                    "derived": sp,
                }
            )
            excl.add(rec.get("task_id"))
    run_split = None
    if seeds:
        try:
            run_split = S.check_seeds(seeds, split)
        except S.SeedPolicyError as exc:
            fails.append({"kind": "mixed_or_wrong_split", "error": str(exc)})
    plan = None
    curtailed = sorted(set(curtailed_task_ids or ()))
    smoke = sorted(set(smoke_task_ids or ()))
    if planned_task_ids is not None:
        planned = list(planned_task_ids)
        counts = Counter(r.get("task_id") for r in records)
        stray = sorted(set(curtailed) - set(planned))
        for t in stray:
            fails.append({"task_id": t, "kind": "curtailed_outside_plan"})
        ran = sorted(set(curtailed) & set(counts))
        for t in ran:
            fails.append({"task_id": t, "kind": "curtailed_but_recorded"})
        # a smoke task's outputs are discarded by the runner: a record of
        # one is a held-out output that must not exist
        for t in sorted(set(smoke) & set(counts)):
            fails.append({"task_id": t, "kind": "smoke_output_recorded"})
            excl.add(t)
        plan = {
            "missing": sorted(set(planned) - set(counts) - set(curtailed) - set(smoke)),
            "unexpected": sorted(set(counts) - set(planned)),
            "duplicated": sorted(t for t, n in counts.items() if n > 1),
            "duplicated_in_plan": sorted(
                t for t, n in Counter(planned).items() if n > 1
            ),
        }
        for kind, ids in plan.items():
            for t in ids:
                fails.append({"task_id": t, "kind": kind})
        excl.update(plan["unexpected"])
        excl.update(plan["duplicated"])
    status = FAIL if fails else PASS
    summary = f"{len(records)} records; split {run_split}; " + (
        "plan compared" if plan is not None else "no plan given"
    )
    if curtailed:
        summary += f"; {len(curtailed)} tasks curtailed by design"
    if smoke:
        summary += f"; {len(smoke)} smoke tasks discarded by design"
    return _check(
        "IA-7",
        status,
        summary,
        fails,
        excl,
        plan_checked=plan is not None,
        split=run_split,
        curtailed_task_ids=curtailed[:LIST_LIMIT],
        n_curtailed=len(curtailed),
        n_smoke=len(smoke),
    )


# --------------------------------------------------------------------------
# IA-8
# --------------------------------------------------------------------------
def status_rule_entry_points_agree(n=400, seed=0) -> dict:
    """The three entry points of the status rule (``assess_item``,
    ``assess_array``, ``status_c``) on a deterministic grid of inputs:
    ``{"n", "disagreements"}``."""
    from impact_pipeline import evidence_v2 as E

    rng = np.random.default_rng(seed)
    proto = E.ProtocolV3(reference={"kind": "cohort_high_state", "session": "awake"})
    c = np.r_[rng.normal(0.3, 0.6, n), [0.0, 0.25, -0.1, 0.1, 1.0]]
    se = np.r_[np.abs(rng.normal(0.08, 0.08, n)) + 1e-4, [0.01, 0.02, 0.03, 0.035, 0.5]]
    df = np.r_[rng.choice([9.0, 12.0, 19.0, np.inf], n), [9.0, 9.0, 9.0, 9.0, 9.0]]
    arr = E.assess_array(
        c,
        0.0,
        0.0,
        se,
        reference=1.0,
        reference_scale="excess",
        se_df=np.where(np.isfinite(df), df, np.nan),
    )
    sc = E.status_c(c, se, np.where(np.isfinite(df), df, np.nan))
    bad = 0
    for i in range(c.size):
        ev = E.ComponentEvidenceV2(
            principle="RAM",
            estimate=float(c[i]),
            null_mean=0.0,
            null_sd=0.0,
            n_null=0,
            se=float(se[i]),
            se_df=None if not np.isfinite(df[i]) else float(df[i]),
            reference=1.0,
            reference_scale="excess",
        )
        a = E.assess_item(ev, proto)
        if not (
            a.status.value == arr.status[i] == sc.status[i]
            and a.reason == arr.reason[i] == sc.reason[i]
        ):
            bad += 1
    return {"n": int(c.size), "disagreements": bad}


SAMPLING_SE = None


def _sampling_methods():
    from impact_pipeline import evidence_v2 as E

    return {k for k, m in E.SE_METHODS.items() if m.sampling_se}


DECISION_CODES = (R.NULL_MODEL_VIOLATED, R.ABSENT_NOT_REACHABLE, R.INCONCLUSIVE)


def stored_status_identity(records, cutoff=(0.25, 0.10)) -> dict:
    """Re-judge every plain-TOST component (single member, sampling SE, a
    decision reason or a decided status) from its stored ``c``, ``se_c`` and
    ``df_c``; NAS (directional) and the exact and concordance routes are
    skipped, and an IIM PRESENT removed by the rank gate is consistent."""
    from impact_pipeline import evidence_v2 as E

    methods = _sampling_methods()
    checked, bad = 0, []
    for rec, s, p, c in _components(records):
        if p == "NAS" or c.get("se_method") not in methods:
            continue
        st, code = c.get("status"), _code(c.get("reason"))
        if st == R.UNDEFINED and code not in DECISION_CODES:
            continue
        cc, se, df = _f(c.get("c")), _f(c.get("se_c")), _f(c.get("df_c"))
        if cc is None or se is None or se <= 0:
            continue
        sa = E.status_c(cc, se, df, cutoff=cutoff)
        got = (
            str(sa.status.item() if hasattr(sa.status, "item") else sa.status),
            sa.reason.item() if hasattr(sa.reason, "item") else sa.reason,
        )
        checked += 1
        want = (st, code if st == R.UNDEFINED else None)
        ok = got[0] == want[0] and (got[1] == want[1] or st != R.UNDEFINED)
        if (
            not ok
            and p == "IIM"
            and got[0] == R.PRESENT
            and c.get("reason") == (f"{R.INCONCLUSIVE}:{R.NULL_NOT_EXCEEDED}")
        ):
            ok = True
        if not ok:
            bad.append(
                {
                    "task_id": rec.get("task_id"),
                    "scoring_id": s.get("scoring_id"),
                    "principle": p,
                    "stored": [st, c.get("reason")],
                    "rule": list(got),
                }
            )
    return {"checked": checked, "mismatches": bad}


def nas_identity(report=None) -> Optional[dict]:
    """The NAS v3 identity on the v1 geometry: a report ``{max_abs_diff}``
    (path or mapping) or, when the estimator module provides it, its hook
    ``nas_v3.v1_geometry_identity()``; None when neither is available."""
    if isinstance(report, (str, Path)):
        report = json.loads(Path(report).read_text(encoding="utf-8"))
    if report is not None:
        return dict(report)
    try:
        from impact_pipeline.v2 import nas_v3  # noqa: F401
    except ImportError:
        return None
    hook = getattr(nas_v3, "v1_geometry_identity", None)
    return None if hook is None else dict(hook())


def ia8_identities(records, nas_identity_report=None, cutoff=(0.25, 0.10)) -> dict:
    fails = []
    ep = status_rule_entry_points_agree()
    if ep["disagreements"]:
        fails.append({"kind": "entry_points_disagree", **ep})
    st = stored_status_identity(records, cutoff)
    for m in st["mismatches"]:
        fails.append({"kind": "stored_status_differs", **m})
    nas = nas_identity(nas_identity_report)
    if nas is not None:
        diff = HE._finite(nas.get("max_abs_diff"))
        if diff is None or diff > NAS_IDENTITY_TOL:
            fails.append(
                {
                    "kind": "nas_v3_v1_identity",
                    "max_abs_diff": nas.get("max_abs_diff"),
                    "tolerance": NAS_IDENTITY_TOL,
                }
            )
    status = FAIL if fails else PASS
    summary = (
        f"entry points agree on {ep['n']} inputs; {st['checked']} stored statuses "
        "re-judged; NAS v3 identity "
        + ("checked" if nas is not None else "not available")
    )
    excl = {m["task_id"] for m in st["mismatches"]}
    return _check(
        "IA-8", status, summary, fails, excl, nas_identity_checked=nas is not None
    )


# --------------------------------------------------------------------------
# IA-9
# --------------------------------------------------------------------------
SRPI_V1 = "srpi-v2-2026.09"


def ia9_srpi(records, delta=0.10) -> dict:
    fails, excl, n_absent, n = [], set(), 0, 0
    for rec, s, p, c in _components(records):
        if p != "SRPI" or c.get("estimator_version") != SRPI_V1:
            continue
        n += 1
        cc, st = _f(c.get("c")), c.get("status")
        if st == R.ABSENT:
            n_absent += 1
            if cc is None or abs(cc) >= delta:
                fails.append(
                    {
                        "task_id": rec.get("task_id"),
                        "kind": "absent_outside_margin",
                        "c": c.get("c"),
                    }
                )
                excl.add(rec.get("task_id"))
        if cc is not None and cc <= -delta and st != R.UNDEFINED:
            fails.append(
                {
                    "task_id": rec.get("task_id"),
                    "kind": "sub_null_decided",
                    "status": st,
                    "c": cc,
                }
            )
            excl.add(rec.get("task_id"))
    return _check(
        "IA-9",
        FAIL if fails else PASS,
        f"{n} v1 SRPI components; {n_absent} ABSENT",
        fails,
        excl,
        n_srpi_absent=n_absent,
    )


# --------------------------------------------------------------------------
# IA-10
# --------------------------------------------------------------------------
def ia10_twins(records, spec) -> dict:
    twin_designs = set(_vocab(spec, "design.twins", "twins")) | set(
        _vocab(spec, "design.ram_only_twins", "ram_only_twins")
    )
    nets = OrderedDict()
    for rec in records:
        key = (
            rec.get("family"),
            rec.get("system"),
            rec.get("seed"),
            _structural_config(rec.get("config")),
        )
        nets.setdefault(key, []).append(rec)
    fails, excl, n_nets = [], set(), 0
    for key, recs in nets.items():
        twins = [r for r in recs if r.get("design") in twin_designs]
        if not twins:
            continue
        n_nets += 1
        base = [
            r
            for r in recs
            if r.get("design") not in twin_designs and int(r.get("replicate") or 0) == 0
        ]
        ids = sorted({r.get("task_id") for r in recs})
        sims = {r.get("task_id"): (r.get("simulation") or {}) for r in twins + base}
        struct = {s.get("structural_hash") for s in sims.values()}
        if None in struct or len(struct) != 1:
            fails.append(
                {
                    "network": list(key[:3]),
                    "kind": "structural_hash",
                    "hashes": sorted(map(str, struct)),
                }
            )
            excl.update(ids)
        by_rep = {}
        for r in twins + base:
            by_rep.setdefault(int(r.get("replicate") or 0), set()).add(
                (r.get("simulation") or {}).get("schedule_hash")
            )
        sched = [h for hs in by_rep.values() for h in hs]
        if (
            None in sched
            or len({h for hs in by_rep.values() for h in hs}) != len(by_rep)
            or any(len(hs) != 1 for hs in by_rep.values())
        ):
            fails.append(
                {
                    "network": list(key[:3]),
                    "kind": "schedule_hash",
                    "replicates": sorted(by_rep),
                }
            )
            excl.update(ids)
        r0_twin = [r for r in twins if int(r.get("replicate") or 0) == 0]
        for t in r0_twin:
            ref = [b for b in base if (b.get("simulation") or {}).get("ts_sha256")]
            if ref and (t.get("simulation") or {}).get("ts_sha256") != (
                ref[0].get("simulation") or {}
            ).get("ts_sha256"):
                fails.append(
                    {
                        "network": list(key[:3]),
                        "kind": "r0_differs_from_witness",
                        "task_id": t.get("task_id"),
                    }
                )
                excl.update(ids)
    return _check(
        "IA-10", FAIL if fails else PASS, f"{n_nets} twin networks", fails, excl
    )


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------
def records_digest(records) -> str:
    """SHA-256 over the sorted canonical JSON of the records: the evaluator
    checks that it judges the records the audit saw."""
    lines = sorted(
        json.dumps(r, sort_keys=True, separators=(",", ":"), default=str)
        for r in records
    )
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def run_audit(
    records,
    *,
    spec=None,
    protocols=None,
    planned_task_ids=None,
    split=None,
    registry=None,
    gate_report=None,
    run_gate=None,
    nas_identity_report=None,
    bad_lines=(),
    curtailed_task_ids=(),
    smoke_task_ids=(),
) -> dict:
    """Every check on raw record dicts; ``ok`` iff no check FAILED (checks
    that could not run are NOT_RUN and listed). ``curtailed_task_ids``: the
    planned tasks the runs skipped by a design's curtailment rule (their
    manifests list them); ``smoke_task_ids``: the planned smoke tasks of a
    development plan (seeds 980-984), whose outputs the runner discards."""
    spec = spec if spec is not None else HE.load_spec()
    recs = list(records)
    checks = [
        ia1_regression_gate(gate_report, run_gate),
        ia2_component_isolation(recs),
        ia3_nas_sampling(recs, spec),
        ia4_undefined_never_absent(recs, spec, registry),
        ia5_duplicates(recs),
        ia6_protocol_hashes(recs, protocols),
        ia7_plan_and_seeds(recs, planned_task_ids, split, bad_lines,
                           curtailed_task_ids, smoke_task_ids),
        ia8_identities(recs, nas_identity_report),
        ia9_srpi(recs),
        ia10_twins(recs, spec),
    ]
    excluded = sorted(
        {t for c in checks for t in c["excluded_task_ids"] if t is not None}
    )
    return {
        "version": AUDIT_VERSION,
        "ok": all(c["status"] != FAIL for c in checks),
        "not_run": [c["id"] for c in checks if c["status"] == NOT_RUN],
        "n_records": len(recs),
        "records_sha256": records_digest(recs),
        "checks": checks,
        "excluded_task_ids": excluded,
        "spec_sha256": HE.spec_sha256(spec),
    }


def _load_protocols(items) -> dict:
    """``{protocol key: ProtocolV3}`` (``HE.load_protocol_files``): the key
    every record carries as ``protocol_id``."""
    return dict(HE.load_protocol_files(items))


def _load_curtailed(paths) -> List[str]:
    """The curtailed task ids of the run manifests (``run_manifest.json``)
    next to the given plan files or in the given run directories."""
    out = []
    for it in paths or ():
        p = Path(it)
        man = p / "run_manifest.json" if p.is_dir() else p.parent / "run_manifest.json"
        if man.is_file():
            out.extend(json.loads(man.read_text(encoding="utf-8")).get(
                "curtailed_task_ids") or [])
    return sorted(set(out))


def _read_plan(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def _load_plan(path):
    obj = _read_plan(path)
    if isinstance(obj, Mapping):
        obj = obj.get("task_ids") or obj.get("tasks") or []
    return [t if isinstance(t, str) else t["task_id"] for t in obj]


def _load_smoke(paths) -> List[str]:
    """The planned smoke tasks (seeds 980-984) of the given plans: the
    tasks of a plan that carries their seeds (the runner's ``run_plan.json``
    lists every task with its seed)."""
    out = []
    for path in paths or ():
        obj = _read_plan(path)
        tasks = obj.get("tasks") if isinstance(obj, Mapping) else obj
        for t in tasks or ():
            seed = t.get("seed") if isinstance(t, Mapping) else None
            if isinstance(seed, int) and S.is_smoke_seed(seed):
                out.append(t["task_id"])
    return sorted(set(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--records",
        nargs="+",
        required=True,
        help="JSON-lines record files or directories",
    )
    ap.add_argument("--hypotheses", default=None, help="hypotheses file (names)")
    ap.add_argument(
        "--protocols",
        nargs="*",
        default=(),
        help="frozen family protocol files or directories",
    )
    ap.add_argument(
        "--plan",
        nargs="*",
        default=None,
        help="the generated plans (task ids; the run plans of several runs are "
        "joined, and the curtailed tasks of their run manifests are expected)",
    )
    ap.add_argument("--split", choices=S.SPLITS, default=None)
    ap.add_argument("--registry", default=None, help="registry v3 entries (JSON)")
    ap.add_argument("--gate-report", default=None, help="regression-gate report (JSON)")
    ap.add_argument("--run-gate", choices=("quick", "full"), default=None)
    ap.add_argument(
        "--nas-identity", default=None, help="NAS v3 identity report (JSON)"
    )
    ap.add_argument("--out", default=None, help="write the audit report here")
    args = ap.parse_args(argv)
    recs, bad = read_raw_records(args.records)
    if not recs and not bad:
        print("no records found", file=sys.stderr)
        return 2
    spec = HE.load_spec(args.hypotheses)
    registry = (
        json.loads(Path(args.registry).read_text(encoding="utf-8"))
        if args.registry
        else None
    )
    rep = run_audit(
        recs,
        spec=spec,
        protocols=_load_protocols(args.protocols) or None,
        planned_task_ids=(
            [t for p in args.plan for t in _load_plan(p)] if args.plan else None
        ),
        curtailed_task_ids=_load_curtailed(args.plan) if args.plan else (),
        smoke_task_ids=_load_smoke(args.plan) if args.plan else (),
        split=args.split,
        registry=registry,
        gate_report=args.gate_report,
        run_gate=args.run_gate,
        nas_identity_report=args.nas_identity,
        bad_lines=bad,
    )
    for c in rep["checks"]:
        print(
            f"[{c['status']:7s}] {c['id']}: {c['summary']}"
            + (f" ({c['n_failures']} failures)" if c["n_failures"] else "")
        )
    print(
        f"integrity audit: {'PASS' if rep['ok'] else 'FAIL'}; "
        f"{len(rep['excluded_task_ids'])} records excluded"
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8"
        )
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
