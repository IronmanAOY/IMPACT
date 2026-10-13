#!/usr/bin/env python
"""
Evaluate a hypotheses file on MPC-Bench records.

Every decision is declared in the hypotheses file (``--spec``, schema
``mpc-bench-hypotheses/1``) and evaluated by
:mod:`impact_pipeline.bench.hypotheses`; this script only loads the inputs,
applies the refusals and writes the outputs.

Inputs
------
``--spec``            the hypotheses file
``--records``         ``mpc-bench-result/1`` JSON-lines files or directories
``--manipulation``    the tables of the manipulation checks: the switch
                      checks, the realisation checks and the reported checks
                      (``bench.manipulation.oracle_checks``; CSV or JSON;
                      tables without a ``passed`` column, such as the
                      usability summary, are left out)
``--registry``        registry admission entries (JSON)
``--protocols``       the family protocols (files ``protocols/bench/<key>.json``
                      or their directories), looked up by the key the records carry
                      as ``protocol_id``: anchors, ``N_anch``, precision
                      blocks and concordance routes
``--mechanism-on``    a table of mechanism-on labels (JSON list), when it is
                      not in the hypotheses file (development evaluations
                      only: a confirmatory evaluation reads the final table of
                      the hypotheses file)

Refusals (confirmatory evaluation)
----------------------------------
A confirmatory evaluation needs ``--freeze-tag``, the git tag of the code
and hypotheses file fixed before the confirmatory run. It refuses records
outside the confirmatory split, records whose provenance does not carry the
tag, an evaluator checkout whose ``src/`` and ``scripts/`` differ from the
tag (``run_hygiene.confirmatory_guard``), a hypotheses file that differs
from its version at the tag, records that break the schema, records whose
family-protocol hash differs from the given protocol, manipulation checks
from another split than the records, a separate mechanism-on table, and an
evaluation without the family protocols. ``--development`` evaluates
development records to test the evaluator; its outputs are flagged
``DEVELOPMENT - NOT A RESULT``.

Outputs (``--out``)
-------------------
``hypotheses_report.json`` (every hypothesis, part and cell with its
numbers; the tally; the hashes of every input), ``hypotheses_parts.csv``
(one row per part) and ``hypotheses_tally.csv``.

Example::

    python scripts/evaluate_hypotheses.py --spec <hypotheses.json> \
        --records outputs/mpcbench/development --protocols protocols/bench \
        --development --out outputs/mpcbench/hypotheses
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
from typing import List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from impact_pipeline.bench import hypotheses as HE  # noqa: E402
from impact_pipeline.bench import records as REC  # noqa: E402
from impact_pipeline.bench import runner as RB  # noqa: E402
from impact_pipeline.bench import seeds as S  # noqa: E402

EVALUATOR_VERSION = "mpc-bench-evaluate-hypotheses/1.0.0"
REPORT_JSON = "hypotheses_report.json"
PARTS_CSV = "hypotheses_parts.csv"
TALLY_CSV = "hypotheses_tally.csv"


class EvaluationRefused(RuntimeError):
    """Inputs that an evaluation must not use."""


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


# JSON-lines files of a run directory that hold no task records: the
# runner's status lines of smoke tasks (their outputs are discarded)
NON_RECORD_FILES = ("smoke.jsonl",)


def read_records(paths) -> tuple:
    """Raw record dicts and the unreadable lines of JSON-lines files (a
    directory contributes its ``*.jsonl`` files except
    :data:`NON_RECORD_FILES`)."""
    files = []
    for it in paths or ():
        p = Path(it)
        if p.is_dir():
            files.extend(f for f in sorted(p.rglob("*.jsonl"))
                         if f.name not in NON_RECORD_FILES)
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(it)
    recs, bad = [], []
    for p in files:
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
    return recs, bad, files


def _read_table(path) -> List[dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        obj = json.loads(text)
        return list(obj if isinstance(obj, list) else obj.get("rows") or [])
    return list(csv.DictReader(io.StringIO(text)))


def manipulation_source(paths) -> List[dict]:
    """The ``manipulation`` rows from switch and realisation tables (a row
    with a ``switch`` holds a switch check; one with ``system_id`` and
    ``check`` a realisation check; a row with ``gate`` False is a reported
    check; :func:`HE.manipulation_rows` leaves out rows without a
    ``passed`` field)."""
    switches, real = [], []
    for p in _files(paths, ("*.csv", "*.json")):
        rows = _read_table(p)
        for r in rows:
            (switches if str(r.get("switch") or "").strip() else real).append(r)
    return HE.manipulation_rows(switches, real)


def load_protocols(paths) -> dict:
    """``{protocol key: Protocol}``: the key the records name their
    protocols by (``runner.load_protocol_files``)."""
    try:
        return dict(RB.load_protocol_files(paths))
    except ValueError as exc:
        raise EvaluationRefused(f"protocols: {exc}") from exc


def record_freeze_tag(rec):
    """The freeze tag a record's provenance names: the runners write it at
    ``provenance['freeze_tag']`` (the runner and the family-B validation);
    a full provenance block carries it at ``provenance['code']['freeze_tag']``."""
    prov = rec.get("provenance") or {}
    tag = prov.get("freeze_tag")
    if tag is None:
        tag = (prov.get("code") or {}).get("freeze_tag")
    return tag


def check_records(recs, *, development: bool, freeze_tag: str) -> str:
    """The split of the records (refusing schema failures, mixed splits, the
    wrong split and, for a confirmatory evaluation, records without the
    freeze tag in their provenance)."""
    bad = []
    for r in recs:
        try:
            REC.TaskRecord.from_dict(r)
        except (ValueError, TypeError) as exc:
            bad.append(f"{r.get('task_id')}: {exc}")
    if bad:
        raise EvaluationRefused(f"records break the schema: {bad[:3]}")
    seeds = [r.get("seed") for r in recs]
    want = S.DEVELOPMENT if development else S.CONFIRMATORY
    try:
        split = S.check_seeds(seeds, want) if seeds else want
    except S.SeedPolicyError as exc:
        raise EvaluationRefused(f"seed policy: {exc}") from exc
    if not development:
        untagged = sorted(
            r.get("task_id") for r in recs if record_freeze_tag(r) != freeze_tag
        )
        if untagged:
            raise EvaluationRefused(
                f"records without freeze tag {freeze_tag!r}: " f"{untagged[:5]}"
            )
    return split


def _estimator_block(pr, principle) -> dict:
    est = (
        pr.get("estimators")
        if isinstance(pr, Mapping)
        else getattr(pr, "estimators", None)
    )
    return dict((est or {}).get(principle) or {})


def check_protocol_hashes(recs, protocols) -> List[dict]:
    """Components whose protocol differs from the given family protocols: a
    ``protocol_hash`` other than the protocol's hash (or a ``protocol_id``
    without a protocol), RAM facets or a PDI bearer other than the protocol
    declares."""
    if not protocols:
        return []
    given = {
        str(pid): pr.get("hash") if isinstance(pr, Mapping) else getattr(pr, "hash")
        for pid, pr in protocols.items()
    }
    fails = []
    for rec in recs:
        for s in rec.get("scorings") or ():
            comps = s.get("components") or {}
            for p, c in list(comps.items()) or [(None, s)]:
                pid = c.get("protocol_id", s.get("protocol_id"))
                ph = c.get("protocol_hash", s.get("protocol_hash"))
                want = given.get(str(pid))
                if want != ph:
                    fails.append(
                        {
                            "task_id": rec.get("task_id"),
                            "scoring_id": s.get("scoring_id"),
                            "principle": p,
                            "kind": "hash_mismatch",
                            "protocol_id": pid,
                            "protocol_hash": ph,
                            "given_hash": want,
                        }
                    )
                    continue
                pr = protocols.get(str(pid))
                det = (c.get("details") or {}) if p else {}
                if p == "RAM" and "facets_not_applicable" in det:
                    want_f = RB.ram_facets_for_estimator(
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
                if p == "PDI" and "bearer" in det:
                    want_b = RB.pdi_params_for_estimator(
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
    return fails


def check_spec_at_tag(path, freeze_tag) -> None:
    """The hypotheses file must be a file of this checkout equal to its
    version at the freeze tag (the confirmatory guard covers ``src/`` and
    ``scripts/`` only)."""
    p = Path(path).resolve()
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
            f"{rel} differs from its version at {freeze_tag!r}"
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
    spec,
    records=(),
    manipulation=(),
    registry=None,
    protocols=(),
    mechanism_on=None,
    development=False,
    freeze_tag=None,
    check_code=True,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = spec
    spec = HE.load_spec(spec_path)
    recs, bad_lines, rec_files = read_records(records)
    if bad_lines:
        raise EvaluationRefused(f"unreadable record lines: {bad_lines[:3]}")
    if not development:
        if not freeze_tag:
            raise EvaluationRefused(
                "a confirmatory evaluation needs --freeze-tag (the git tag of "
                "the code and hypotheses file fixed before the confirmatory run)"
            )
        if mechanism_on:
            raise EvaluationRefused(
                "a confirmatory evaluation reads the mechanism-on labels of the "
                "hypotheses file, not a separate table"
            )
        if check_code:
            check_spec_at_tag(spec_path, freeze_tag)
    split = check_records(recs, development=development, freeze_tag=freeze_tag)
    code = None
    if not development and check_code:
        from impact_pipeline.bench import run_hygiene as PV

        seeds = [r["seed"] for r in recs]
        code = PV.confirmatory_guard(seeds, freeze_tag=freeze_tag)
    protos = load_protocols(protocols) if protocols else {}
    mism = check_protocol_hashes(recs, protos)
    if mism:
        if not development:
            raise EvaluationRefused(f"family-protocol hash mismatches: {mism[:3]}")
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
    try:
        ctx = HE.build_context(
            recs,
            sources=sources,
            protocols=protos or None,
            split=split,
            mechanism_on=mech,
        )
    except S.SeedPolicyError as exc:
        raise EvaluationRefused(f"auxiliary inputs: {exc}") from exc
    if not development and not protos:
        # anchors, N_anch, the testability gates and the concordance routes
        # are read from the family protocols
        raise EvaluationRefused("a confirmatory evaluation needs the family "
                                "protocols")
    report = HE.evaluate(spec, ctx)
    report["evaluator_version"] = EVALUATOR_VERSION
    report["freeze_tag"] = None if development else freeze_tag
    report["n_records"] = len(recs)
    report["protocol_hash_mismatches"] = mism
    report["code"] = code
    report["inputs"] = {
        "records": {str(p): _sha256(p) for p in rec_files},
        "spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
        "protocols": {k: getattr(v, "hash", None) for k, v in protos.items()},
        "registry": None if not registry else _sha256(registry),
        "mechanism_on": None if not mechanism_on else _sha256(mechanism_on),
        "manipulation": {
            str(p): _sha256(p) for p in _files(manipulation, ("*.csv", "*.json"))
        },
    }
    report["script_sha256"] = _sha256(__file__)
    (out / REPORT_JSON).write_text(
        json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8"
    )
    _csv(
        HE.parts_table(report),
        out / PARTS_CSV,
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
    _csv(rows, out / TALLY_CSV, ["kind", "outcome", "count"])
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", required=True, help="the hypotheses file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", nargs="*", default=())
    ap.add_argument("--manipulation", nargs="*", default=())
    ap.add_argument("--registry", default=None)
    ap.add_argument("--protocols", nargs="*", default=())
    ap.add_argument("--mechanism-on", default=None)
    ap.add_argument("--freeze-tag", default=None,
                    help="the git tag of the code and hypotheses file of a "
                         "confirmatory evaluation")
    ap.add_argument(
        "--development",
        action="store_true",
        help="evaluate development records (testing only; NOT A RESULT)",
    )
    args = ap.parse_args(argv)
    try:
        rep = run(
            args.out,
            spec=args.spec,
            records=args.records,
            manipulation=args.manipulation,
            registry=args.registry,
            protocols=args.protocols,
            mechanism_on=args.mechanism_on,
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
