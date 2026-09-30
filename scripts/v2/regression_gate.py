#!/usr/bin/env python
"""
v1 regression gate of MPC-Bench v2 (integrity audit IA-1).

The v2 tree must keep the v1 code path selectable and byte-identical. This
gate runs on every merge of the v2 work and checks, on the current tree:

``environment``
    the numerical stack equals the environment lock (``env_lock.py``);
``read_only_files``
    every read-only v1 file (estimators, evidence layer, v1 runner, frozen
    scripts, v1 protocols and registry, v1 docs) has the bytes it had when
    the v2 work started;
``v1_protocols``
    the frozen v1 protocol hashes (``evidence.Protocol.hash``) are unchanged
    and the files equal those at tag ``mpcbench-freeze-v1``;
``v1_constants``
    the v1 schema, bench, generator and estimator version strings;
``rejudge_records``
    every stored bench record of C1-C3 (``outputs/paper1_mpcbench``) is
    re-judged from its stored components under its v1 protocol: 0 status,
    0 verdict and 0 other differences in the verdict block;
``rejudge_evaluator``
    the frozen evaluator's component and verdict tables (family protocols
    included) are recomputed from the stored records and compared with the
    stored tables of C2 and of the verification's single evaluator call;
``rejudge_null_calibration``
    every stored null-calibration row is re-classified under the v1 protocol
    and status rule of the stored run, and the stored rate and verdict
    tables are rebuilt byte for byte from the re-classified statuses;
``verification_reruns``
    the 14 from-scratch re-runs of the v1 verification (three bench tasks,
    three IIM validation tasks, eight null-calibration components) are
    re-run on this tree and compared bit for bit with the stored records;
``stratified_reruns``
    about 30 stored bench records, stratified over designs, families and
    generators, are re-run from scratch and compared bit for bit (every
    field except wall times);
``bold_error``
    a v1 whole-brain BOLD task still ends in the v1 NAS band error under the
    v1 protocol (the v1 runner keeps its single try block).

The re-runs replay stored v1 tasks with their own seeds through the v1
runner; they never create new seeds or new results and never write into the
stored outputs. The stored outputs are located through ``--v1-outputs``,
``$MPCBENCH_V1_OUTPUTS``, this checkout's ``outputs/paper1_mpcbench`` or the
main checkout of a linked worktree.

Exit status: 0 when every check passed, 1 when a check failed, 2 when the
stored outputs are needed but missing. ``--quick`` runs the checks without
simulations (seconds); the full gate takes about 20 minutes with
``--workers 6``, bounded by the two system-bearer patchwork re-runs, which
are submitted first. ``--report PATH`` writes the full JSON report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import functools
import hashlib
import io
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GATE_VERSION = "mpc-bench-v1-regression-gate/1.0.0"
FREEZE_TAG_V1 = "mpcbench-freeze-v1"
V1_OUTPUTS_RELPATH = Path("outputs/paper1_mpcbench")
V1_OUTPUTS_ENV = "MPCBENCH_V1_OUTPUTS"
PASS, FAIL, SKIPPED = "pass", "fail", "skipped"

# Frozen v1 protocol hashes (evidence.Protocol.hash) and file bytes.
V1_PROTOCOLS = {
    "protocols/mpc_bench_v1.json": (
        "855f6a444b77030d33faa68fcb45e8576b931d2d681cf215d4dacdb57a6b2520"
    ),
    "protocols/mpc_bench_v1_anchored.json": (
        "780581f57d24251fc8ed8c39565f0a89a96740908f2ada3f65f8961cf1543f4c"
    ),
    "protocols/mpc_default_v1.json": (
        "383eb310cf4479d3260bc5f43f8972bd0b12104a221579fea17c40e410357267"
    ),
}
# Family-C protocols of the v1 evaluation (the frozen protocols with the
# family-C reference from reference_C), as recorded in C2.
V1_FAMILY_C_PROTOCOLS = {
    "protocol_hash": "45c9eea3b315f165cd45e2282fc7ce9759803ae8f5fe607821422ae0c38aaf00",
    "anchored_protocol_hash": (
        "18b8b27a8ad8f29245984b86927d7a009a25e1d1044daaae0dd6976c160e86b2"
    ),
}
# SHA-256 of every read-only v1 file when the v2 work started (commit
# e36e2a5); none of them may change during the v2 round.
READ_ONLY_V1_SHA256 = {
    "src/impact_pipeline/mpc_metrics.py":
        "673636240b704be507cf356453bdf823dc59738ea0557a768566a13aef1b4794",
    "src/impact_pipeline/nulls.py":
        "2f0cdffdc9787939e50b158a47d80f369f110ff2a5d9e0fd0a4172657cb9004d",
    "src/impact_pipeline/evidence.py":
        "ca4852de2307915b49b611fe8c00f6b2656dd6fd083dbeda6ecaf1046028638a",
    "src/impact_pipeline/iim_xp.py":
        "d1cb239777599b1ac2b26d2be6d534e9de7d0838d2a50d2bdae0e9bc130337de",
    "src/impact_pipeline/bench/export.py":
        "03772a0b57d35df30110e409895a37ad870d82c27dd7c989947afd4a9a0a8b6b",
    "src/impact_pipeline/bench/run_bench.py":
        "9993c534e24649c436a625ee4fa81d8876ce1f76b52d8a9971ea2fac3b62a83a",
    "src/impact_pipeline/bench/rules.py":
        "d7bb245efe93bf6a7952a471ffeaad18cf745866262610075a5bee3f59622bc0",
    "src/impact_pipeline/bench/forward.py":
        "621912b0c0de8e5e8584fcf08d18281fdea0016f26e7fef50fd4c9d474955202",
    "src/impact_pipeline/bench/whole_brain.py":
        "036d7a4788bc4b724c151b577b8e307cd1264ff25e3bd605749bfd0c633efe4a",
    "src/impact_pipeline/bench/manipulation.py":
        "52de52e12bd85e850f5c69649cc3b003c00c9dc4fd5442d1623180092395d943",
    "src/impact_pipeline/bench/witnesses.yaml":
        "a995d54399eb3d960a825df9f23c58b5cb3fea7b7029c833d634a63bd7ea87d9",
    "src/impact_pipeline/bench/witnesses.json":
        "ad4a74cb92cfaa355a0e12afe348ece19813e89e2e8c9943a3d05b74bbea5450",
    "scripts/bench_hypotheses.py":
        "956e094cc3b05cfaaab6d4fe327b0e47b3e129f871b8aa3d64c003b02d938c59",
    "scripts/null_calibration.py":
        "b9b9613aee66c80db32f06c2b2b35b75f5e3c8578ebe57332a5ab403f66db433",
    "scripts/iim_validation.py":
        "e6b1f73c22390f3b13fc3be954a5d4a18443cfae8ca3bb5553259f3a1f8f3840",
    "scripts/build_applicability_registry.py":
        "9f0fd51287ef06cadaa6466e505e4df3de2a2b833e48bb14114dc465d05fa5d5",
    "protocols/mpc_bench_v1.json":
        "9265421d03b4e060e04829d0ade23671d5872cbe5d9b7836012b8695e65380eb",
    "protocols/mpc_bench_v1_anchored.json":
        "8eacd76034ac3a7af747ceb89577a1017d6eaaad3f9df430026881df5dad221d",
    "protocols/mpc_default_v1.json":
        "26049bb46cbb58143d058f98d863150fe546a7eb5f15c11202341c6a6bee872e",
    "protocols/mpc_bench_v1_reference_summary.json":
        "26c85c5a929740e734b45238a27b079273f2b5e22a19455eeb88552c8439aa06",
    "protocols/applicability_registry_v1.json":
        "a300e08f6bd04b385896272af44a63d4a9c8f63c3828547f2e1aa1167f663ca9",
    "docs/metrics.md":
        "ab3a9b09cfb9c72dfd0c89466dcc1089c9a0bdec665c28c799384215ecf7926a",
    "docs/preregistration/MPC_BENCH_PREREGISTRATION.md":
        "e95be691aa5ec2cfd459837f9bcf46753b084d27c55d3c54f9b12ee4a1ca8b87",
}
V1_CONSTANTS = {
    "impact_pipeline.bench.run_bench.RESULT_SCHEMA": "mpc-bench-result/2",
    "impact_pipeline.bench.BENCH_VERSION": "2.0.0",
    "impact_pipeline.bench.generators.GENERATOR_VERSION": "mpc-bench-generators/1.1.0",
    "impact_pipeline.evidence.PROTOCOL_SCHEMA": "impact-mpc-protocol/2",
    "impact_pipeline.evidence.REGISTRY_SCHEMA": "impact-mpc-registry/2",
    "impact_pipeline.mpc_metrics.ESTIMATOR_VERSIONS": {
        "RAM": "ram-v2-2026.09",
        "PDI": "pdi-v2-2026.09",
        "NAS": "nas-v2-2026.09",
        "IIM": "iim-v4-2026.09",
        "SRPI": "srpi-v2-2026.09",
    },
}

# Stored bench-format records of C1-C3 (relative to the outputs root).
BENCH_RECORD_FILES = (
    "c1/hc1_full/bench_null_witness_records.jsonl",
    "c2/bench/A/factorial/results.jsonl",
    "c2/bench/A/patchwork_sweep/results.jsonl",
    "c2/bench/A/sweep/results.jsonl",
    "c2/bench/A/witnesses/results.jsonl",
    "c2/bench/C/factorial/results.jsonl",
    "c2/bench/C/patchwork_sweep/results.jsonl",
    "c2/bench/C/sweep/results.jsonl",
    "c2/bench/C/witnesses/results.jsonl",
    "c2/reference_C/results.jsonl",
    "c3/bench/adversarial/results.jsonl",
    "c3/bench/whole_brain/results.jsonl",
    "c3/deviation_bold_noNAS/results.jsonl",
)
# The 14 from-scratch re-runs of the v1 verification.
VERIFICATION_BENCH = (
    ("b1", "c2/bench/C/witnesses/results.jsonl",
     "witness-C-W_NAS_no_workspace-s10003"),
    ("b2", "c2/bench/A/witnesses/results.jsonl",
     "witness-A-W_SRPI_no_efference-s10007"),
    ("b3", "c3/bench/adversarial/results.jsonl",
     "adversarial-parity_grid-s10005"),
)
VERIFICATION_IIM = (
    "feedforward_star-n4-T3000-directional-s10004",
    "independent-n4-T10000-bidirectional-s10011",
    "sweep-ring-n4-T30000-c0.45-directional-s10013",
)
VERIFICATION_NULLCAL = (
    ("surrogate_linear", 1200, 16, 61),
    ("ar1", 1200, 8, 33),
)
BOLD_ERROR_TASK = ("c3/bench/whole_brain/results.jsonl", "wholebrain-G0-bold-s10002")
BOLD_ERROR_TEXT = "ValueError: compute_NAS invalid band (0.5, 4.0) for Nyquist 0.25"
# Strata of the stratified re-runs: (file, number of records, required field
# values). Within a file records are taken in the order of sha256(task_id),
# preferring a new (generator, sweep knob, bearer mode) until the stratum is
# filled; only records with status ok, and none of the verification re-runs.
# A patchwork scored with the system bearer costs about 20 min; the witness
# strata already hold one per family, so the graded patchworks are sampled
# with per-principle bearers.
STRATA = (
    ("c2/bench/A/witnesses/results.jsonl", 4, {}),
    ("c2/bench/C/witnesses/results.jsonl", 4, {}),
    ("c2/bench/A/factorial/results.jsonl", 2, {}),
    ("c2/bench/C/factorial/results.jsonl", 2, {}),
    ("c2/bench/A/sweep/results.jsonl", 3, {}),
    ("c2/bench/C/sweep/results.jsonl", 3, {}),
    ("c2/bench/A/patchwork_sweep/results.jsonl", 1, {"bearer_mode": "principle"}),
    ("c2/bench/C/patchwork_sweep/results.jsonl", 1, {"bearer_mode": "principle"}),
    ("c2/reference_C/results.jsonl", 1, {}),
    ("c3/bench/adversarial/results.jsonl", 5, {}),
    ("c3/bench/whole_brain/results.jsonl", 2, {}),
    ("c3/deviation_bold_noNAS/results.jsonl", 1, {}),
)
TASK_FIELDS = (
    "task_id", "design", "family", "generator", "cell_id", "knobs", "seed",
    "config", "witness_id", "sweep_knob", "sweep_level", "bearer_mode",
    "generator_kwargs",
)
# Floats of the null-calibration CSV carry 16 significant digits.
CSV_RTOL = 1e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _check(name, status, summary, **details) -> dict:
    return {"name": name, "status": status, "summary": summary, "details": details}


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args) -> Optional[bytes]:
    try:
        proc = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              timeout=60, check=False,
                              env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except Exception:  # noqa: BLE001
        return None
    return proc.stdout if proc.returncode == 0 else None


def locate_v1_outputs(explicit=None, env=None) -> Optional[Path]:
    """The stored v1 outputs (``outputs/paper1_mpcbench``) or None."""
    env = os.environ if env is None else env
    for value in (explicit, env.get(V1_OUTPUTS_ENV)):
        if value:
            p = Path(value).expanduser()
            if not (p / "c2").is_dir():
                raise FileNotFoundError(f"{p} does not hold the v1 outputs (no c2/)")
            return p
    candidates = [REPO_ROOT / V1_OUTPUTS_RELPATH]
    from impact_pipeline.v2.ownership import main_checkout_root

    main = main_checkout_root(REPO_ROOT)
    if main is not None:
        candidates.append(main / V1_OUTPUTS_RELPATH)
    for c in candidates:
        if (c / "c2").is_dir():
            return c
    return None


def load_jsonl(path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def strip_volatile(obj):
    """Drop wall times (``timing``, ``seconds``, ``*_seconds``,
    ``auditor_wall_s``) at every level."""
    if isinstance(obj, dict):
        return {
            k: strip_volatile(v) for k, v in obj.items()
            if not (k in ("timing", "seconds", "auditor_wall_s")
                    or k.endswith("_seconds"))
        }
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _same_scalar(a, b, rtol=0.0, strict_types=True) -> bool:
    """Equality of two JSON scalars. ``strict_types``: an int and a float
    differ even when equal in value (a record written as ``1`` is not the
    record written as ``1.0``). ``rtol`` > 0 (tables read from CSV, whose
    floats carry 16 significant digits) accepts relative differences up to
    ``rtol`` (absolute below 1)."""
    if _is_number(a) and _is_number(b):
        if strict_types and type(a) is not type(b):
            return False
        if a == b:
            return True
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return rtol > 0 and math.isfinite(fa) and math.isfinite(fb) and (
            abs(fa - fb) <= rtol * max(1.0, abs(fa), abs(fb)))
    if type(a) is not type(b):
        return False
    return a == b


def diff(a, b, path="", rtol=0.0, limit=20, strict_types=True) -> List[str]:
    """Paths where two JSON-like values differ (at most ``limit``); ints and
    floats differ unless ``strict_types`` is False."""
    out: List[str] = []

    def walk(x, y, p):
        if len(out) >= limit:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    out.append(f"{p}/{k}: only in {'rerun' if k in x else 'stored'}")
                else:
                    walk(x[k], y[k], f"{p}/{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                out.append(f"{p}: length {len(x)} != {len(y)}")
            else:
                for i, (u, v) in enumerate(zip(x, y)):
                    walk(u, v, f"{p}[{i}]")
        elif not _same_scalar(x, y, rtol, strict_types):
            out.append(f"{p}: {x!r} != {y!r}")

    walk(a, b, path)
    return out


def _json_roundtrip(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


@functools.lru_cache(maxsize=1)
def _protocols_by_hash() -> dict:
    """``{protocol hash: (path, Protocol)}`` of the frozen v1 protocols."""
    from impact_pipeline import evidence as ev

    out = {}
    for rel in V1_PROTOCOLS:
        proto = ev.Protocol.from_json(REPO_ROOT / rel)
        out[proto.hash] = (REPO_ROOT / rel, proto)
    return out


def _record_protocol(rec: dict):
    """``(path, Protocol)`` of the frozen protocol a stored bench record was
    judged under (its verdict's hash; the bench protocol for records without
    a verdict)."""
    ph = (rec.get("verdict") or {}).get("protocol_hash") or V1_PROTOCOLS[
        "protocols/mpc_bench_v1.json"]
    table = _protocols_by_hash()
    if ph not in table:
        raise ValueError(
            f"{rec.get('task_id')}: protocol hash {ph} is not a frozen v1 hash")
    return table[ph]


def _record_protocol_path(rec: dict) -> Path:
    return _record_protocol(rec)[0]


# ---------------------------------------------------------------------------
# static checks
# ---------------------------------------------------------------------------
def check_environment(allow_mismatch=False) -> dict:
    from scripts.v2 import env_lock

    rep = env_lock.report(REPO_ROOT)
    bad = rep["version_mismatches"]
    status = PASS if not bad or allow_mismatch else FAIL
    if not bad:
        summary = "environment equals the lock"
    else:
        summary = f"{len(bad)} version mismatch(es)" + (
            " (allowed by --allow-env-mismatch)" if allow_mismatch else "")
    return _check("environment", status, summary, mismatches=bad,
                  strict=rep["strict"], environment=rep["environment"])


def check_read_only_files() -> dict:
    changed, missing = [], []
    for rel, want in READ_ONLY_V1_SHA256.items():
        p = REPO_ROOT / rel
        if not p.is_file():
            missing.append(rel)
        elif _sha256(p) != want:
            changed.append(rel)
    ok = not changed and not missing
    return _check("read_only_files", PASS if ok else FAIL,
                  f"{len(READ_ONLY_V1_SHA256)} read-only v1 files unchanged" if ok
                  else f"changed {changed}, missing {missing}",
                  changed=changed, missing=missing, n_files=len(READ_ONLY_V1_SHA256))


def check_v1_protocols() -> dict:
    from impact_pipeline import evidence as ev

    rows, bad = [], []
    tag_ok = _git("rev-parse", "--verify", "--quiet", f"{FREEZE_TAG_V1}^{{commit}}")
    for rel, want in V1_PROTOCOLS.items():
        proto = ev.Protocol.from_json(REPO_ROOT / rel)
        row = {"file": rel, "protocol_hash": proto.hash, "expected": want,
               "hash_ok": proto.hash == want, "equals_freeze_tag": None}
        if tag_ok is not None:
            frozen = _git("show", f"{FREEZE_TAG_V1}:{rel}")
            row["equals_freeze_tag"] = (
                frozen is not None and frozen == (REPO_ROOT / rel).read_bytes())
        if not row["hash_ok"] or row["equals_freeze_tag"] is False:
            bad.append(rel)
        rows.append(row)
    summary = (f"{len(rows)} frozen protocol hashes unchanged"
               + ("" if tag_ok is not None else f" (tag {FREEZE_TAG_V1} not present)"))
    return _check("v1_protocols", PASS if not bad else FAIL,
                  summary if not bad else f"changed: {bad}", protocols=rows,
                  freeze_tag_present=tag_ok is not None)


def check_v1_constants() -> dict:
    import importlib

    bad = {}
    for dotted, want in V1_CONSTANTS.items():
        mod, _, name = dotted.rpartition(".")
        have = getattr(importlib.import_module(mod), name, None)
        if have != want:
            bad[dotted] = {"have": have, "want": want}
    return _check("v1_constants", PASS if not bad else FAIL,
                  f"{len(V1_CONSTANTS)} v1 version constants unchanged" if not bad
                  else f"changed: {sorted(bad)}", changed=bad)


# ---------------------------------------------------------------------------
# re-judging stored records
# ---------------------------------------------------------------------------
def rejudge_bench_record(rec: dict) -> dict:
    """The verdict block of a stored bench record recomputed from its stored
    components under its frozen protocol (exactly as the v1 runner did)."""
    from impact_pipeline.bench.export import evidence_verdict
    from impact_pipeline.bench.run_bench import _sanitize

    modes = rec.get("estimator_modes") or {}
    meta = {
        "substrate": (rec.get("system") or {}).get("substrate"),
        "iim_grain": (modes.get("IIM") or {}).get("grain"),
    }
    res = {"components": rec["components"],
           "estimator_modes": rec.get("estimator_modes")}
    label = (rec.get("verdict") or {}).get("run_label")
    v = evidence_verdict(res, meta, protocol_id=label,
                         protocol=_record_protocol(rec)[1])
    return _json_roundtrip(_sanitize(v))


def check_rejudge_records(outputs: Path) -> dict:
    per_file, examples = {}, []
    totals = {"records": 0, "with_verdict": 0, "status_diffs": 0, "verdict_diffs": 0,
              "other_diffs": 0}
    for rel in BENCH_RECORD_FILES:
        path = outputs / rel
        if not path.is_file():
            per_file[rel] = {"missing": True}
            totals["other_diffs"] += 1
            continue
        stats = {"records": 0, "with_verdict": 0, "status_diffs": 0,
                 "verdict_diffs": 0, "other_diffs": 0}
        for rec in load_jsonl(path):
            stats["records"] += 1
            stored = rec.get("verdict")
            if rec.get("status") != "ok" or not stored:
                continue
            stats["with_verdict"] += 1
            new = rejudge_bench_record(rec)
            st = sum(new["component_status"].get(p) != s
                     for p, s in stored["component_status"].items())
            st += len(set(new["component_status"]) ^ set(stored["component_status"]))
            vd = int(new["verdict"] != stored["verdict"]
                     or new["reasons"] != stored["reasons"])
            rest = {k: v for k, v in new.items()
                    if k not in ("component_status", "verdict", "reasons")}
            rest_stored = {k: v for k, v in stored.items()
                           if k not in ("component_status", "verdict", "reasons")}
            other = diff(rest, rest_stored)
            stats["status_diffs"] += st
            stats["verdict_diffs"] += vd
            stats["other_diffs"] += bool(other)
            if (st or vd or other) and len(examples) < 10:
                examples.append({"file": rel, "task_id": rec["task_id"],
                                 "diff": diff(new, stored)})
        per_file[rel] = stats
        for k in totals:
            totals[k] += stats[k]
    ok = not (totals["status_diffs"] or totals["verdict_diffs"]
              or totals["other_diffs"])
    return _check("rejudge_records", PASS if ok else FAIL,
                  f"{totals['with_verdict']} stored verdicts re-judged: "
                  f"{totals['status_diffs']} status, "
                  f"{totals['verdict_diffs']} verdict, "
                  f"{totals['other_diffs']} other differences",
                  totals=totals, files=per_file, examples=examples)


def _evaluator_tables(records, ref_c):
    """The frozen evaluator's ``components.csv`` and ``verdicts.csv`` text
    for the given records (family protocols from ``ref_c``)."""
    import scripts.bench_hypotheses as BH

    proto, proto_val = BH.load_protocols(False)
    protocols, info = BH.family_protocols(proto, proto_val, ref_c, False)
    with warnings.catch_warnings():
        # pandas deprecation notices raised inside the frozen evaluator
        warnings.simplefilter("ignore", FutureWarning)
        comp, verd = BH.assess_family(records, protocols)
    buf_c, buf_v = io.StringIO(), io.StringIO()
    comp.drop(columns=["se_replicates"], errors="ignore").to_csv(buf_c, index=False)
    verd.to_csv(buf_v, index=False)
    return buf_c.getvalue(), buf_v.getvalue(), info


def _csv_rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def _table_differences(new_text, stored_text, key_cols, status_cols):
    """Row-level comparison of two CSV tables: rows missing on either side,
    rows whose status columns differ, rows whose other columns differ."""
    new = {tuple(r[k] for k in key_cols): r for r in _csv_rows(new_text)}
    old = {tuple(r[k] for k in key_cols): r for r in _csv_rows(stored_text)}
    out = {"rows": len(old), "missing": len(set(old) ^ set(new)), "status_diffs": 0,
           "other_diffs": 0, "examples": []}
    for key in sorted(set(old) & set(new)):
        a, b = new[key], old[key]
        st = [c for c in status_cols if a.get(c) != b.get(c)]
        other = [c for c in b if c not in status_cols and a.get(c) != b.get(c)]
        out["status_diffs"] += bool(st)
        out["other_diffs"] += bool(other)
        if (st or other) and len(out["examples"]) < 5:
            out["examples"].append({"key": list(key), "status": st, "other": other})
    return out


def check_rejudge_evaluator(outputs: Path) -> dict:
    from impact_pipeline.bench import analysis as A

    cases = [
        ("c2", ["c2/bench/A", "c2/bench/C"], "c2/hypotheses"),
        ("verification_single_call",
         ["c2/bench/A", "c2/bench/C", "c3/bench/adversarial"],
         "verification/reruns/evaluator_single_call"),
    ]
    results, ok = {}, True
    ref_c = A.read_records([outputs / "c2/reference_C"])
    for name, dirs, stored_dir in cases:
        stored = outputs / stored_dir
        if not (stored / "components.csv").is_file():
            results[name] = {"missing": str(stored)}
            ok = False
            continue
        records = A.read_records([outputs / d for d in dirs])
        comp_text, verd_text, info = _evaluator_tables(records, ref_c)
        comp_stored = (stored / "components.csv").read_text(encoding="utf-8")
        verd_stored = (stored / "verdicts.csv").read_text(encoding="utf-8")
        res = {
            "records": len(records),
            "components_byte_identical": comp_text == comp_stored,
            "verdicts_byte_identical": verd_text == verd_stored,
            "family_c_protocol_hashes": {
                k: (info.get("C") or {}).get(k) for k in V1_FAMILY_C_PROTOCOLS},
        }
        res["family_c_protocols_ok"] = (
            res["family_c_protocol_hashes"] == V1_FAMILY_C_PROTOCOLS)
        res["components"] = _table_differences(
            comp_text, comp_stored, ("task_id", "principle"),
            ("status", "status_reason"))
        res["verdicts"] = _table_differences(
            verd_text, verd_stored, ("task_id",),
            ("verdict", "reasons", "verdict_val", "reasons_val"))
        case_ok = (res["family_c_protocols_ok"]
                   and not res["components"]["missing"]
                   and not res["components"]["status_diffs"]
                   and not res["verdicts"]["missing"]
                   and not res["verdicts"]["status_diffs"]
                   and res["components_byte_identical"]
                   and res["verdicts_byte_identical"])
        res["ok"] = case_ok
        ok = ok and case_ok
        results[name] = res

    def _word(flag):
        return "byte-identical" if flag else "DIFFER"

    summ = "; ".join(
        f"{k}: {v.get('records', 0)} records, components "
        f"{_word(v.get('components_byte_identical'))}, verdicts "
        f"{_word(v.get('verdicts_byte_identical'))}"
        for k, v in results.items())
    return _check("rejudge_evaluator", PASS if ok else FAIL, summ, cases=results)


def _nan_none(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def check_rejudge_null_calibration(outputs: Path) -> dict:
    import pandas as pd

    import scripts.null_calibration as NC

    path = outputs / "c1/null_calibration/null_calibration_replicates.csv"
    summary = outputs / "c1/null_calibration/null_calibration.json"
    if not path.is_file() or not summary.is_file():
        return _check("rejudge_null_calibration", FAIL,
                      "stored null calibration missing")
    from impact_pipeline import evidence as ev

    stored_summary = json.loads(summary.read_text(encoding="utf-8"))
    proto = ev.Protocol.from_dict(stored_summary["protocol"])
    status_rule = stored_summary["status_rule"]
    df = pd.read_csv(path, float_precision="round_trip")
    n_status = n_reason = n_c = 0
    examples, new_status = [], []
    for row in df.itertuples(index=False):
        comp = {
            "estimate": row.estimate, "null_mean": row.null_mean,
            "null_sd": row.null_sd, "se": row.se, "se_n": int(row.se_n),
            "defined": bool(row.defined),
            "reason": _nan_none(row.estimator_reason),
            "null_family": _nan_none(row.null_family), "n_null": int(row.n_null),
        }
        st, reason, _impl, a = NC.classify(row.principle, comp, proto,
                                           status_rule=status_rule)
        new_status.append(st)
        c_new = float("nan") if a is None else a.c
        bad_status = st != row.status
        bad_reason = (reason or None) != _nan_none(row.status_reason)
        bad_c = not _same_scalar(float(c_new), float(row.c), CSV_RTOL)
        n_status += bad_status
        n_reason += bad_reason
        n_c += bad_c
        if (bad_status or bad_reason or bad_c) and len(examples) < 10:
            examples.append({
                "null_kind": row.null_kind, "n_time": int(row.n_time),
                "n_nodes": int(row.n_nodes), "replicate": int(row.replicate),
                "principle": row.principle, "status": [st, row.status],
                "reason": [reason, _nan_none(row.status_reason)],
                "c": [c_new, row.c]})
    # the stored rate and verdict tables, rebuilt with the frozen script's
    # own summary from the re-classified statuses (the verdicts are the
    # strong-Kleene AND of each replicate's component statuses)
    rejudged = df.copy()
    rejudged["status"] = new_status
    alpha = float(stored_summary["design"]["alpha"])
    tables = {}
    for name, table in zip(("rates", "verdicts"),
                           NC.summarise(rejudged, alpha=alpha)):
        buf = io.StringIO()
        table.to_csv(buf, index=False)
        stored_csv = path.parent / f"null_calibration_{name}.csv"
        tables[name] = (stored_csv.is_file() and buf.getvalue()
                        == stored_csv.read_text(encoding="utf-8"))
    ok = not (n_status or n_reason or n_c) and all(tables.values())
    return _check("rejudge_null_calibration", PASS if ok else FAIL,
                  f"{len(df)} null-calibration components re-classified: "
                  f"{n_status} status, {n_reason} reason, "
                  f"{n_c} construct-value differences; rate and verdict "
                  f"tables {'byte-identical' if all(tables.values()) else 'DIFFER'}",
                  rows=len(df), protocol_hash=proto.hash, status_rule=status_rule,
                  tables_byte_identical=tables, examples=examples)


# ---------------------------------------------------------------------------
# re-runs (spawned single-threaded workers, like the v1 runner)
# ---------------------------------------------------------------------------
def _job_bench(stored: dict, protocol_path: str) -> dict:
    """Re-run one stored bench task through the v1 runner."""
    from impact_pipeline import evidence as ev
    from impact_pipeline.bench import run_bench as RB
    from impact_pipeline.bench.export import protocol_params
    from impact_pipeline.bench.factorial import BenchTask

    task = BenchTask(**{k: stored[k] for k in TASK_FIELDS})
    proto = ev.Protocol.from_json(protocol_path)
    params = protocol_params(proto, None)
    t0 = time.time()
    rec = RB.run_task(task, stored["metrics"], stored["null_surrogates"], params,
                      stored.get("provenance"), "markers" in stored, True,
                      stored["se_groups"], proto.to_dict())
    return {"record": _json_roundtrip(rec), "wall_s": round(time.time() - t0, 2)}


def _job_iim(task_id: str) -> dict:
    import scripts.iim_validation as IV

    tasks = IV.tasks_for(["independent", "feedforward_star"], 4,
                         [3000, 10000, 30000], range(10000, 10020),
                         ["bidirectional", "directional"],
                         sweep_kind="ring", couplings=[0.45])
    (task,) = [t for t in tasks if t["task_id"] == task_id]
    t0 = time.time()
    rec = IV.run_task(task, 19)
    rec = {k: IV._jsonable(v) for k, v in rec.items()}
    return {"record": _json_roundtrip(rec), "wall_s": round(time.time() - t0, 2)}


def _job_nullcal(kind: str, n_time: int, n_nodes: int, replicate: int,
                 summary_path: str) -> dict:
    import scripts.null_calibration as NC
    from impact_pipeline import evidence as ev
    from impact_pipeline.bench.export import protocol_params

    s = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    d = s["design"]
    proto = ev.Protocol.from_dict(s["protocol"])
    task = {"kind": kind, "n_time": int(n_time), "n_nodes": int(n_nodes),
            "replicate": int(replicate), "seed": int(s["seed"]),
            "dt": float(d["dt"]), "iim_macro_nodes": int(d["iim_macro_nodes"]),
            "ar_coef": float(d["ar_coef"]), "pink_beta": float(d["pink_beta"]),
            "null_surrogates": int(d["null_surrogates"]),
            "metrics": tuple(d["metrics"]),
            "params": protocol_params(proto, None), "protocol": proto.to_dict(),
            "se_groups": int(d["se_groups"]), "status_rule": s["status_rule"],
            "srpi_agency_fix": bool(d["srpi_agency_fix"])}
    t0 = time.time()
    rows = NC.run_replicate(task)
    return {"rows": rows, "wall_s": round(time.time() - t0, 2)}


def _run_jobs(jobs, workers: int, costs=None) -> list:
    """Run ``(fn, args)`` jobs in spawned workers with one BLAS/OpenMP thread
    each (as the v1 runner did); results in job order, exceptions as
    ``{'exception': text}``. With ``costs`` (expected seconds per job) the
    longest jobs are submitted first, which shortens the wall time."""
    from impact_pipeline.bench import run_bench as RB

    out = [None] * len(jobs)
    order = list(range(len(jobs)))
    if costs is not None:
        order.sort(key=lambda i: -float(costs[i] or 0.0))
    with RB._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, int(workers)), initializer=RB._worker_init,
            mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = {ex.submit(jobs[i][0], *jobs[i][1]): i for i in order}
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception as exc:  # noqa: BLE001 - reported as a failure
                out[i] = {"exception": f"{type(exc).__name__}: {exc}"}
    return out


def _stored_attempts(outputs: Path, rel: str, task_id: str) -> List[dict]:
    """Every stored record of a task id, in file order (the v1 runner appends
    a second attempt when a failed task was re-run)."""
    matches = [r for r in load_jsonl(outputs / rel) if r["task_id"] == task_id]
    if not matches:
        raise KeyError(f"{rel}: no record with task id {task_id}")
    return matches


def _stored_record(outputs: Path, rel: str, task_id: str) -> dict:
    """The latest stored record of a task id (as the v1 readers use it)."""
    return _stored_attempts(outputs, rel, task_id)[-1]


def verification_jobs(outputs: Path) -> list:
    """The 14 re-runs of the v1 verification as jobs with their references."""
    jobs = []
    for label, rel, task_id in VERIFICATION_BENCH:
        stored = _stored_record(outputs, rel, task_id)
        jobs.append({"kind": "bench", "label": label, "task": task_id,
                     "stored": stored, "fn": _job_bench,
                     "args": (stored, str(_record_protocol_path(stored)))})
    iv_path = outputs / "c1/iim_validation/iim_validation.jsonl"
    iv = {r["task_id"]: r for r in load_jsonl(iv_path)}
    for task_id in VERIFICATION_IIM:
        jobs.append({"kind": "iim", "label": task_id, "task": task_id,
                     "stored": iv[task_id], "fn": _job_iim, "args": (task_id,)})
    summary = str(outputs / "c1/null_calibration/null_calibration.json")
    for kind, n_time, n_nodes, rep in VERIFICATION_NULLCAL:
        jobs.append({"kind": "nullcal",
                     "label": f"{kind} T{n_time} N{n_nodes} rep {rep}",
                     "task": (kind, n_time, n_nodes, rep), "stored": None,
                     "fn": _job_nullcal,
                     "args": (kind, n_time, n_nodes, rep, summary)})
    return jobs


def _compare_bench(result: dict, stored: dict) -> List[str]:
    return diff(strip_volatile(result["record"]), strip_volatile(stored))


def _compare_iim(result: dict, stored: dict) -> List[str]:
    drop = ("seconds", "code_git_sha", "freeze_tag")
    a = {k: v for k, v in result["record"].items() if k not in drop}
    b = {k: v for k, v in stored.items() if k not in drop}
    return diff(a, b)


def _nullcal_references(outputs: Path):
    import pandas as pd

    csv_rows = pd.read_csv(
        outputs / "c1/null_calibration/null_calibration_replicates.csv",
        float_precision="round_trip")
    stored_rerun = outputs / "verification/reruns/nullcal.json"
    reruns = []
    if stored_rerun.is_file():
        reruns = json.loads(stored_rerun.read_text(encoding="utf-8"))
    return csv_rows, reruns


def _compare_nullcal_rows(rows: list, outputs: Path) -> dict:
    """Each re-run row against the stored rows of the replicates CSV
    (16-digit floats: relative tolerance 1e-12) and against the stored
    verification re-run (bit for bit)."""
    csv_rows, reruns = _nullcal_references(outputs)
    keys = ("null_kind", "n_time", "n_nodes", "replicate", "principle")
    out = []

    def _finite(d):
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in d.items()}

    def _no_wall(d):
        return {k: v for k, v in d.items() if k != "seconds_system"}

    for row in rows:
        row = _json_roundtrip(_no_wall(row))
        m = csv_rows
        for k in keys:
            m = m[m[k] == row[k]]
        res = {"component": f"{row['null_kind']} T{row['n_time']} N{row['n_nodes']} "
                            f"rep {row['replicate']} {row['principle']} "
                            f"({row['status']})"}
        if len(m) != 1:
            res["csv_diffs"] = [f"{len(m)} stored rows"]
        else:
            o = {k: _nan_none(v.item() if hasattr(v, "item") else v)
                 for k, v in _no_wall(m.iloc[0].to_dict()).items()}
            res["csv_diffs"] = diff(_finite(row), _finite(o), rtol=CSV_RTOL,
                                    strict_types=False)
        ref = [r for r in reruns if all(r.get(k) == row[k] for k in keys)]
        if ref:
            ref_row = _json_roundtrip(_no_wall(ref[0]))
            res["verification_rerun_diffs"] = diff(row, ref_row)
        else:
            res["verification_rerun_diffs"] = None
        res["identical"] = (not res["csv_diffs"]
                            and not res["verification_rerun_diffs"])
        out.append(res)
    return out


def check_verification_reruns(outputs: Path, workers: int) -> dict:
    jobs = verification_jobs(outputs)
    results = _run_jobs([(j["fn"], j["args"]) for j in jobs], workers)
    rows, n_ok = [], 0
    stored_reruns = {}
    for label, _rel, _tid in VERIFICATION_BENCH:
        p = outputs / "verification/reruns" / f"{label}.json"
        if p.is_file():
            stored_reruns[label] = json.loads(p.read_text(encoding="utf-8"))
    iim_reruns = {}
    p = outputs / "verification/reruns/iim.json"
    if p.is_file():
        iim_reruns = {r["task_id"]: r
                      for r in json.loads(p.read_text(encoding="utf-8"))}
    for job, res in zip(jobs, results):
        if "exception" in res:
            rows.append({"task": job["label"], "identical": False,
                         "error": res["exception"]})
            continue
        if job["kind"] == "bench":
            d = _compare_bench(res, job["stored"])
            ref = [r for r in stored_reruns.get(job["label"], [])
                   if r["task_id"] == job["task"]]
            d_ver = None
            if ref:
                drop = ("provenance",)
                d_ver = diff(strip_volatile({k: v for k, v in res["record"].items()
                                             if k not in drop}),
                             strip_volatile({k: v for k, v in ref[0].items()
                                             if k not in drop}))
            ok = not d and not d_ver
            rows.append({"task": job["task"], "identical": ok, "diffs": d,
                         "verification_rerun_diffs": d_ver, "wall_s": res["wall_s"]})
            n_ok += ok
        elif job["kind"] == "iim":
            d = _compare_iim(res, job["stored"])
            # the stored verification re-run holds a subset of the fields
            d_ver = None
            ref = iim_reruns.get(job["task"])
            if ref is not None:
                d_ver = diff({k: res["record"].get(k) for k in ref
                              if k != "seconds"},
                             {k: v for k, v in ref.items() if k != "seconds"})
            ok = not d and not d_ver
            rows.append({"task": job["task"], "identical": ok, "diffs": d,
                         "verification_rerun_diffs": d_ver,
                         "wall_s": res["wall_s"]})
            n_ok += ok
        else:
            for r in _compare_nullcal_rows(res["rows"], outputs):
                r["task"] = f"null_calibration {r.pop('component')}"
                r["wall_s"] = res["wall_s"]
                rows.append(r)
                n_ok += r["identical"]
    ok = n_ok == len(rows) == 14
    return _check("verification_reruns", PASS if ok else FAIL,
                  f"{n_ok}/{len(rows)} verification re-runs identical", reruns=rows)


def stratified_selection(outputs: Path, strata=STRATA) -> List[tuple]:
    """``(file, task_id)`` of the stratified re-run sample (deterministic).
    Strata keys already covered by a verification re-run of the same file
    count as seen, so the sample adds generators rather than repeating
    them."""
    skip = {t for _l, _r, t in VERIFICATION_BENCH} | {BOLD_ERROR_TASK[1]}
    out = []
    for rel, k, required in strata:
        all_recs = load_jsonl(outputs / rel)
        recs = [r for r in all_recs
                if r.get("status") == "ok" and r["task_id"] not in skip
                and all(r.get(f) == v for f, v in required.items())]
        recs.sort(key=lambda r: hashlib.sha256(r["task_id"].encode()).hexdigest())
        chosen = []
        seen = {(r.get("generator"), r.get("sweep_knob"), r.get("bearer_mode"))
                for r in all_recs if r["task_id"] in skip}
        for r in recs:
            key = (r.get("generator"), r.get("sweep_knob"), r.get("bearer_mode"))
            if key not in seen and len(chosen) < k:
                chosen.append(r)
                seen.add(key)
        for r in recs:
            if len(chosen) >= k:
                break
            if r not in chosen:
                chosen.append(r)
        out.extend((rel, r["task_id"]) for r in chosen)
    return out


def check_stratified_reruns(outputs: Path, workers: int) -> dict:
    sample = stratified_selection(outputs)
    stored = {}
    by_file = {}
    for rel, tid in sample:
        by_file.setdefault(rel, set()).add(tid)
    for rel, tids in by_file.items():
        for r in load_jsonl(outputs / rel):
            if r["task_id"] in tids:
                stored[(rel, r["task_id"])] = r
    jobs = [(_job_bench, (stored[key], str(_record_protocol_path(stored[key]))))
            for key in sample]
    costs = [(stored[key].get("timing") or {}).get("total_s") for key in sample]
    results = _run_jobs(jobs, workers, costs)
    rows, n_ok = [], 0
    for key, res in zip(sample, results):
        if "exception" in res:
            rows.append({"file": key[0], "task": key[1], "identical": False,
                         "error": res["exception"]})
            continue
        d = _compare_bench(res, stored[key])
        rows.append({"file": key[0], "task": key[1], "identical": not d, "diffs": d,
                     "wall_s": res["wall_s"]})
        n_ok += not d
    ok = n_ok == len(rows) and len(rows) > 0
    return _check("stratified_reruns", PASS if ok else FAIL,
                  f"{n_ok}/{len(rows)} stratified stored records reproduced "
                  "bit for bit", reruns=rows)


def check_bold_error(outputs: Path, workers: int = 1) -> dict:
    rel, task_id = BOLD_ERROR_TASK
    attempts = _stored_attempts(outputs, rel, task_id)
    stored = attempts[-1]
    job = (_job_bench, (stored, str(_record_protocol_path(stored))))
    (res,) = _run_jobs([job], 1)
    if "exception" in res:
        return _check("bold_error", FAIL, f"re-run raised {res['exception']}")
    rec = res["record"]
    # every stored attempt (the v1 run and its re-run of failed tasks) is the
    # same record up to wall times, and the re-run reproduces it
    d = [x for a in attempts for x in _compare_bench(res, a)]
    ok = (rec.get("status") == "error" and rec.get("error") == BOLD_ERROR_TEXT
          and not d)
    return _check("bold_error", PASS if ok else FAIL,
                  f"{task_id}: {rec.get('status')} ({rec.get('error')}); "
                  f"{len(attempts)} stored attempt(s) reproduced",
                  diffs=d, stored_attempts=len(attempts), wall_s=res["wall_s"])


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
STORED_CHECKS = ("rejudge_records", "rejudge_evaluator",
                 "rejudge_null_calibration")
RERUN_CHECKS = ("verification_reruns", "stratified_reruns", "bold_error")
STATIC_CHECKS = ("environment", "read_only_files", "v1_protocols", "v1_constants")
CHECK_NAMES = STATIC_CHECKS + STORED_CHECKS + RERUN_CHECKS


def run_gate(outputs=None, *, quick=False, workers=4, skip_stored=False,
             allow_env_mismatch=False, only: Optional[Iterable[str]] = None) -> dict:
    """Run the gate; returns the report (``ok``, ``checks``)."""
    t0 = time.time()
    wanted = None if only is None else set(only)
    if wanted is not None and wanted - set(CHECK_NAMES):
        raise ValueError(f"unknown checks {sorted(wanted - set(CHECK_NAMES))}; "
                         f"known: {list(CHECK_NAMES)}")

    def want(name):
        return wanted is None or name in wanted

    checks = []
    for name, fn in (("environment", lambda: check_environment(allow_env_mismatch)),
                     ("read_only_files", check_read_only_files),
                     ("v1_protocols", check_v1_protocols),
                     ("v1_constants", check_v1_constants)):
        if want(name):
            checks.append(fn())
    candidates = STORED_CHECKS if quick else STORED_CHECKS + RERUN_CHECKS
    stored_names = [n for n in candidates if want(n)]
    out_dir = None
    if stored_names:
        out_dir = None if skip_stored else locate_v1_outputs(outputs)
        if out_dir is None:
            status = SKIPPED if skip_stored else FAIL
            for n in stored_names:
                checks.append(_check(n, status, "stored v1 outputs not available"
                                     + (" (--skip-stored)" if skip_stored else "")))
        else:
            fns = {
                "rejudge_records": check_rejudge_records,
                "rejudge_evaluator": check_rejudge_evaluator,
                "rejudge_null_calibration": check_rejudge_null_calibration,
                "verification_reruns": functools.partial(
                    check_verification_reruns, workers=workers),
                "stratified_reruns": functools.partial(
                    check_stratified_reruns, workers=workers),
                "bold_error": check_bold_error,
            }
            for n in stored_names:
                t1 = time.time()
                try:
                    c = fns[n](out_dir)
                except Exception as exc:  # noqa: BLE001 - a crash fails the check
                    c = _check(n, FAIL,
                               f"check raised {type(exc).__name__}: {exc}")
                c["seconds"] = round(time.time() - t1, 1)
                checks.append(c)
    ok = all(c["status"] != FAIL for c in checks)
    return {
        "version": GATE_VERSION,
        "ok": ok,
        "quick": bool(quick),
        "v1_outputs": None if out_dir is None else str(out_dir),
        "checks": checks,
        "seconds": round(time.time() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--v1-outputs", default=None,
                    help=f"stored v1 outputs (default: ${V1_OUTPUTS_ENV}, "
                         f"./{V1_OUTPUTS_RELPATH} or the main checkout's)")
    ap.add_argument("--quick", action="store_true",
                    help="static checks and re-judging only (no simulations)")
    ap.add_argument("--workers", type=int, default=4, help="re-run workers (default 4)")
    ap.add_argument("--skip-stored", action="store_true",
                    help="skip every check that needs the stored v1 outputs")
    ap.add_argument("--allow-env-mismatch", action="store_true",
                    help="report an environment mismatch without failing")
    ap.add_argument("--only", default=None,
                    help="comma-separated check names to run")
    ap.add_argument("--list-sample", action="store_true",
                    help="print the stratified re-run sample and exit")
    ap.add_argument("--report", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)
    if args.list_sample:
        out = locate_v1_outputs(args.v1_outputs)
        if out is None:
            print("stored v1 outputs not found", file=sys.stderr)
            return 2
        for rel, tid in stratified_selection(out):
            print(f"{rel}\t{tid}")
        return 0
    only = None
    if args.only:
        only = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = sorted(set(only) - set(CHECK_NAMES))
        if unknown:
            ap.error(f"unknown checks {unknown}; known: {', '.join(CHECK_NAMES)}")
    rep = run_gate(args.v1_outputs, quick=args.quick, workers=args.workers,
                   skip_stored=args.skip_stored,
                   allow_env_mismatch=args.allow_env_mismatch, only=only)
    for c in rep["checks"]:
        print(f"[{c['status'].upper():7s}] {c['name']}: {c['summary']}")
    print(f"regression gate: {'PASS' if rep['ok'] else 'FAIL'} "
          f"({rep['seconds']} s)")
    if args.report:
        Path(args.report).write_text(json.dumps(rep, indent=1, default=str),
                                     encoding="utf-8")
    missing = any(c["status"] == FAIL
                  and c["summary"].startswith("stored v1 outputs not available")
                  for c in rep["checks"])
    if missing:
        return 2
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
