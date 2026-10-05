#!/usr/bin/env python
"""
Evaluator of the MPC-Bench v2 hypotheses (Tier A: HCv2-0 to HCv2-24).

Every decision is declared in ``protocols/v2/hypotheses_v2.json`` and
evaluated by :mod:`impact_pipeline.v2.hypothesis_engine`; this script only
loads the inputs, applies the refusals and writes the outputs.

Inputs
------
``--records``         ``mpc-bench-result/3`` JSON-lines files or directories
``--manipulation``    the per-seed tables of the prerequisite M: the switch
                      checks and the realisation checks
                      (``bench.manipulation_v2.prerequisite_m``; CSV or JSON)
``--registry``        registry v3 admission entries (JSON)
``--protocols``       the frozen family protocols (``impact-mpc-protocol/3``
                      files or directories): anchors, ``N_anch``, precision
                      blocks and concordance routes
``--audit``           the integrity-audit report; its ``excluded_task_ids``
                      leave every hypothesis, and the audit outcome is
                      reported beside the tally
``--mechanism-on``    the CD-8 table of mechanism-on labels (JSON list), when
                      it is not yet in the hypotheses file (development
                      evaluations only: a confirmatory evaluation reads the
                      final table of the frozen hypotheses file)
``--hypotheses``      the hypotheses file (default: the frozen one)

Refusals (confirmatory evaluation)
----------------------------------
Records outside the confirmatory split, records whose provenance does not
carry the freeze tag, an evaluator checkout that is not the frozen tree
(``provenance.confirmatory_guard``), a hypotheses file that differs from its
version at the freeze tag (the guard covers ``src/`` and ``scripts/`` only),
records that break the schema, records whose family-protocol hash differs
from the frozen protocol (IA-6), an integrity audit that is missing,
failed, left a check not run or was run on other records, manipulation
checks from another split than the records, a mechanism-on table given
outside the frozen hypotheses file, and a confirmatory evaluation without
the frozen family protocols.
``--development`` evaluates development records to test the evaluator; its
outputs are flagged ``DEVELOPMENT - NOT A RESULT``.

Outputs (``--out``)
-------------------
``hypotheses_v2.json`` (every hypothesis, part and cell with its numbers;
the tally; the hashes of every input), ``hypotheses_v2_parts.csv`` (one row
per part) and ``hypotheses_v2_tally.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from impact_pipeline.v2 import FREEZE_TAG_V2  # noqa: E402
from impact_pipeline.v2 import hypothesis_engine as HE  # noqa: E402
from impact_pipeline.v2 import records as REC  # noqa: E402
from impact_pipeline.v2 import seeds as S  # noqa: E402

EVALUATOR_VERSION = "mpc-bench-hypotheses-v2/1.0.0"


class EvaluationRefused(RuntimeError):
    """Inputs that a v2 evaluation must not use."""


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(items, patterns=("*.jsonl",)) -> List[Path]:
    out = []
    for it in items or ():
        p = Path(it)
        if p.is_dir():
            for pat in patterns:
                out.extend(sorted(p.rglob(pat)))
        elif p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


def read_records(paths) -> tuple:
    """Raw record dicts and the unreadable lines of JSON-lines files."""
    from scripts.v2 import integrity_audit as IA

    files = _files(paths)
    recs, bad = IA.read_raw_records(files)
    return recs, bad, files


def _read_table(path) -> List[dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        obj = json.loads(text)
        return list(obj if isinstance(obj, list) else obj.get("rows") or [])
    return list(csv.DictReader(io.StringIO(text)))


def manipulation_source(paths) -> List[dict]:
    """The ``manipulation`` rows from switch and realisation tables (a table
    with a ``switch`` column holds switch checks; one with ``system_id`` and
    ``check`` holds realisation checks)."""
    switches, real = [], []
    for p in _files(paths, ("*.csv", "*.json")):
        rows = _read_table(p)
        for r in rows:
            (switches if str(r.get("switch") or "").strip() else real).append(r)
    return HE.manipulation_rows(switches, real)


def load_protocols(paths) -> dict:
    from impact_pipeline import evidence_v2 as E

    out = {}
    for p in _files(paths, ("*.json",)):
        pr = E.load_protocol(p)
        out[str(getattr(pr, "name", None) or p.stem)] = pr
    return out


def check_records(recs, *, development: bool, freeze_tag: str, excluded=()) -> str:
    """The split of the records (refusing schema failures, mixed splits, the
    wrong split and, for a confirmatory evaluation, records without the
    freeze tag in their provenance)."""
    excluded = set(excluded)
    bad = []
    for r in recs:
        if r.get("task_id") in excluded:
            continue
        try:
            REC.TaskRecord.from_dict(r)
        except (ValueError, TypeError) as exc:
            bad.append(f"{r.get('task_id')}: {exc}")
    if bad:
        raise EvaluationRefused(f"records break the schema: {bad[:3]}")
    seeds = [r.get("seed") for r in recs if r.get("task_id") not in excluded]
    want = S.DEVELOPMENT if development else S.CONFIRMATORY
    try:
        split = S.check_seeds(seeds, want) if seeds else want
    except S.SeedPolicyError as exc:
        raise EvaluationRefused(f"seed policy: {exc}") from exc
    if not development:
        untagged = sorted(
            r.get("task_id")
            for r in recs
            if r.get("task_id") not in excluded
            and ((r.get("provenance") or {}).get("code") or {}).get("freeze_tag")
            != freeze_tag
        )
        if untagged:
            raise EvaluationRefused(
                f"records without freeze tag {freeze_tag!r}: " f"{untagged[:5]}"
            )
    return split


def check_protocol_hashes(recs, protocols, excluded=()) -> List[dict]:
    from scripts.v2 import integrity_audit as IA

    keep = [r for r in recs if r.get("task_id") not in set(excluded)]
    return IA.ia6_protocol_hashes(keep, protocols)["failures"] if protocols else []


def check_audit(audit_rep, recs) -> None:
    """A confirmatory evaluation needs an integrity audit that passed, ran
    every check and saw exactly these records."""
    from scripts.v2 import integrity_audit as IA

    if audit_rep is None:
        raise EvaluationRefused("a confirmatory evaluation needs the integrity audit")
    if not audit_rep.get("ok"):
        failed = [
            c["id"] for c in audit_rep.get("checks") or () if c.get("status") == "FAIL"
        ]
        raise EvaluationRefused(f"the integrity audit failed: {failed}")
    if audit_rep.get("not_run"):
        raise EvaluationRefused(f"integrity checks not run: {audit_rep['not_run']}")
    if audit_rep.get("records_sha256") != IA.records_digest(recs):
        raise EvaluationRefused("the integrity audit was run on other records")


def check_frozen_hypotheses(path, freeze_tag) -> None:
    """The hypotheses file must equal its version at the freeze tag (the
    confirmatory guard covers ``src/`` and ``scripts/``, not ``protocols/``)."""
    p = Path(path or HE.SPEC_PATH).resolve()
    try:
        rel = p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        raise EvaluationRefused(f"{p} is not a file of this checkout") from None
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"refs/tags/{freeze_tag}:{rel}"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise EvaluationRefused(f"{rel} is not part of the tag {freeze_tag!r}")
    if proc.stdout != p.read_bytes():
        raise EvaluationRefused(
            f"{rel} differs from its frozen version at {freeze_tag!r}"
        )


def _csv(rows, path, columns=None):
    rows = list(rows)
    cols = columns or (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def run(
    out_dir,
    *,
    records=(),
    manipulation=(),
    registry=None,
    protocols=(),
    audit=None,
    mechanism_on=None,
    hypotheses=None,
    development=False,
    freeze_tag=FREEZE_TAG_V2,
    check_code=True,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = HE.load_spec(hypotheses)
    audit_rep = json.loads(Path(audit).read_text(encoding="utf-8")) if audit else None
    recs, bad_lines, rec_files = read_records(records)
    if bad_lines:
        raise EvaluationRefused(f"unreadable record lines: {bad_lines[:3]}")
    if not development:
        if mechanism_on:
            raise EvaluationRefused(
                "a confirmatory evaluation reads the mechanism-on labels of the "
                "frozen hypotheses file, not a separate table"
            )
        check_audit(audit_rep, recs)
        if check_code:
            check_frozen_hypotheses(hypotheses, freeze_tag)
    excluded = set((audit_rep or {}).get("excluded_task_ids") or ())
    split = check_records(
        recs, development=development, freeze_tag=freeze_tag, excluded=excluded
    )
    code = None
    if not development and check_code:
        from impact_pipeline.v2 import provenance as PV

        seeds = [r["seed"] for r in recs if r.get("task_id") not in excluded]
        code = PV.confirmatory_guard(seeds, freeze_tag=freeze_tag)
    protos = load_protocols(protocols) if protocols else {}
    mism = check_protocol_hashes(recs, protos, excluded)
    if mism:
        if not development:
            raise EvaluationRefused(
                f"family-protocol hash mismatches (IA-6): {mism[:3]}"
            )
    sources = {}
    if manipulation:
        sources["manipulation"] = manipulation_source(manipulation)
    reg_obj = None
    if registry:
        reg_obj = json.loads(Path(registry).read_text(encoding="utf-8"))
        sources["registry"] = HE.registry_rows(reg_obj)
    mech = (
        json.loads(Path(mechanism_on).read_text(encoding="utf-8"))
        if mechanism_on
        else None
    )
    keep = [r for r in recs if r.get("task_id") not in excluded]
    try:
        ctx = HE.build_context(
            keep,
            sources=sources,
            protocols=protos or None,
            split=split,
            excluded_task_ids=excluded,
            mechanism_on=mech,
        )
    except S.SeedPolicyError as exc:
        raise EvaluationRefused(f"auxiliary inputs: {exc}") from exc
    if not development and not protos:
        # anchors, N_anch, the testability gates and the concordance routes
        # are read from the frozen family protocols
        raise EvaluationRefused("a confirmatory evaluation needs the frozen protocols")
    report = HE.evaluate(spec, ctx)
    report["evaluator_version"] = EVALUATOR_VERSION
    report["freeze_tag"] = None if development else freeze_tag
    report["n_records"] = len(keep)
    report["integrity_audit"] = (
        None
        if audit_rep is None
        else {
            "ok": audit_rep.get("ok"),
            "version": audit_rep.get("version"),
            "checks": {c["id"]: c["status"] for c in audit_rep.get("checks") or ()},
            "excluded_task_ids": sorted(excluded),
        }
    )
    report["protocol_hash_mismatches"] = mism
    report["code"] = code
    report["inputs"] = {
        "records": {str(p): _sha256(p) for p in rec_files},
        "hypotheses": str(hypotheses or HE.SPEC_PATH),
        "protocols": {k: getattr(v, "hash", None) for k, v in protos.items()},
        "registry": None if not registry else _sha256(registry),
        "audit": None if not audit else _sha256(audit),
        "mechanism_on": None if not mechanism_on else _sha256(mechanism_on),
        "manipulation": {
            str(p): _sha256(p) for p in _files(manipulation, ("*.csv", "*.json"))
        },
    }
    report["script_sha256"] = _sha256(__file__)
    (out / "hypotheses_v2.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8"
    )
    _csv(
        HE.parts_table(report),
        out / "hypotheses_v2_parts.csv",
        [
            "hypothesis",
            "hypothesis_outcome",
            "part",
            "role",
            "label",
            "rule",
            "outcome",
            "prediction",
            "reason",
        ],
    )
    tally = report["tally"]
    rows = [
        {"kind": "decisive_parts", "outcome": k, "count": v}
        for k, v in tally["decisive_parts"].items()
    ]
    rows += [
        {"kind": "hypotheses", "outcome": k, "count": v}
        for k, v in tally["hypotheses"].items()
    ]
    _csv(rows, out / "hypotheses_v2_tally.csv", ["kind", "outcome", "count"])
    return report


def _list(text) -> List[str]:
    return [s.strip() for s in str(text or "").split(",") if s.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", nargs="*", default=())
    ap.add_argument("--manipulation", nargs="*", default=())
    ap.add_argument("--registry", default=None)
    ap.add_argument("--protocols", nargs="*", default=())
    ap.add_argument("--audit", default=None)
    ap.add_argument("--mechanism-on", default=None)
    ap.add_argument("--hypotheses", default=None)
    ap.add_argument("--freeze-tag", default=FREEZE_TAG_V2)
    ap.add_argument(
        "--development",
        action="store_true",
        help="evaluate development records (testing only; NOT A RESULT)",
    )
    args = ap.parse_args(argv)
    try:
        rep = run(
            args.out,
            records=args.records,
            manipulation=args.manipulation,
            registry=args.registry,
            protocols=args.protocols,
            audit=args.audit,
            mechanism_on=args.mechanism_on,
            hypotheses=args.hypotheses,
            development=args.development,
            freeze_tag=args.freeze_tag,
        )
    except EvaluationRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(rep["outcomes"], indent=1))
    print(json.dumps(rep["tally"]["decisive_parts"]))
    if rep["evaluator_errors"]:
        print(
            f"evaluator errors in {len(rep['evaluator_errors'])} parts: "
            f"{rep['evaluator_errors'][:3]}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
