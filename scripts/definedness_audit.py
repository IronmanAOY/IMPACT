#!/usr/bin/env python
"""
Definedness audit: which MPC principle / evidence channel is definable on
which dataset, from BIDS metadata only.

Read-only and metadata-only by construction:

- files are discovered by *listing* directories; recordings are recognised by
  their file names (``*_bold.nii[.gz]``, ``*_eeg.{edf,bdf,set,vhdr,fif}``) and
  are never opened;
- only JSON sidecars (``*.json``, including ``*_events.json`` level
  descriptions and ``dataset_description.json``) are parsed, and of
  ``*_events.tsv`` files only the header line is parsed (column names);
- every opened file is recorded in an access log with its size, mtime and
  the SHA-256 of its bytes (the whole file is hashed so the log identifies
  the exact version touched; for events files only the header line is
  interpreted: ``bytes_parsed``);
- files are opened with mode ``"rb"`` only; the output directory must lie
  outside the data root; annexed files whose content is not fetched (broken
  symlinks) are logged and never followed.

Definedness rules mirror the pipeline contracts (``event_parsing`` patterns,
``mpc_metrics`` channels, with the pipeline's strict defaults): PDI
(repertoire, unlabelled), NAS (capacity) and IIM need a recording with its sampling
metadata (``RepetitionTime`` / ``SamplingFrequency``; no silent TR fallback);
NAS capacity also needs a hub (``workspace_nodes``) declared in the protocol
(``--protocol``, default ``protocols/mpc_default_v1.json``, which declares
none): without one it is ``NOT_DEFINABLE`` with the reason
``NO_DECLARED_WORKSPACE``, as the evidence layer records NAS as UNDEFINED
(``UNDEFINED:NAS:NO_DECLARED_WORKSPACE``); the audit does not check that a
declared hub fits the recording's grain;
PDI's legacy baseline needs a rest recording of the same subject and
modality; RAM needs an events table with ``trial_type`` goal, stimulus and
feedback levels and a numeric feedback value column
(``event_parsing.FEEDBACK_VALUE_COLUMNS``; strict RAM is undefined without
explicit feedback values), and typed channels also need an ``impact_channel``
column (perturbational and endogenous are not implemented); SRPI legacy needs
levels that ``event_parsing.classify_self_nonself`` classifies as self and as
non-self; SRPI agency needs ``self_caused``/``other_caused`` levels plus
``yoked_to`` and ``phase_bin`` columns. When the level set cannot be known
from metadata (no ``*_events.json`` level description) the cell is
``REQUIRES_EVENT_VALUES``: the audit does not read event rows. DEFINABLE is a
metadata-level statement; event counts and value ranges are checked only when
the estimators run.

Statuses: ``DEFINABLE``, ``REQUIRES_EVENT_VALUES``, ``NOT_DEFINABLE``,
``NOT_IMPLEMENTED``, ``METADATA_UNAVAILABLE``.

Prior-access levels: ds003171, ds005620 and ds006623 have been
analysed with the author's code and are exploratory / calibration only; the
other datasets are recorded as ``metadata_only`` (this audit read metadata;
confirmatory eligibility additionally requires a declaration that no time
series were analysed).

Outputs (``--out``): ``definedness_recordings.csv`` (one row per recording
and principle/channel), ``definedness_matrix.csv`` (dataset x modality x task x
principle x channel counts), ``definedness_access_log.csv`` and
``definedness_summary.json`` (provenance, prior access, file counts).

Example::

    python scripts/definedness_audit.py --data-root data/scratch \
        --out outputs/definedness
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline.event_parsing import (  # noqa: E402
    AGENCY_OTHER_LABEL,
    AGENCY_SELF_LABEL,
    FEEDBACK_RE,
    FEEDBACK_VALUE_COLUMNS,
    GOAL_RE,
    IMPACT_CHANNEL_COLUMN,
    IMPACT_CHANNELS,
    IMPLEMENTED_RAM_CHANNELS,
    SRPI_TEXT_COLUMNS,
    STIM_RE,
    _norm_trial_type,
    classify_self_nonself,
)

AUDIT_VERSION = "definedness-audit/1.1.0"
# NAS capacity needs a declared hub; the default protocol of empirical runs
# declares none (derived protocols in protocols/examples/ do).
DEFAULT_PROTOCOL = REPO_ROOT / "protocols" / "mpc_default_v1.json"
NO_DECLARED_WORKSPACE = "NO_DECLARED_WORKSPACE"
DEFAULT_DATASETS = (
    "ds003171", "ds005620", "ds006623", "ds002547",
    "ds004295", "ds005479", "ds002685", "ds002336",
)
EXPLORATORY_DATASETS = ("ds003171", "ds005620", "ds006623")
PRIOR_ACCESS = {d: "exploratory_calibration" for d in EXPLORATORY_DATASETS}
DEFAULT_PRIOR_ACCESS = "metadata_only"

DEFINABLE = "DEFINABLE"
REQUIRES_EVENT_VALUES = "REQUIRES_EVENT_VALUES"
NOT_DEFINABLE = "NOT_DEFINABLE"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
METADATA_UNAVAILABLE = "METADATA_UNAVAILABLE"
STATUSES = (DEFINABLE, REQUIRES_EVENT_VALUES, NOT_DEFINABLE, NOT_IMPLEMENTED,
            METADATA_UNAVAILABLE)

# Recording data files (listed, never opened) and the metadata files that may
# be opened. Anything else is never opened.
RECORDING_PATTERNS = {
    "fmri": re.compile(r"_bold\.nii(\.gz)?$"),
    "eeg": re.compile(r"_eeg\.(edf|bdf|set|vhdr|fif)$"),
}
METADATA_OPEN_RE = re.compile(r"(\.json|_events\.tsv)$")
SKIP_DIRS = {"derivatives", "sourcedata", "code", "stimuli", ".git", ".datalad",
             ".heudiconv", "log", "logs"}
REST_TASK_RE = re.compile(r"rest|baseline", re.I)
MAX_HEADER_BYTES = 65536


class MetadataReader:
    """Opens only metadata files, read-only, and logs every access."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self.log = []
        self._json_cache = {}
        self._header_cache = {}

    def _check(self, path: Path) -> Path:
        path = Path(path)
        if not METADATA_OPEN_RE.search(path.name):
            raise PermissionError(f"refusing to open a non-metadata file: {path}")
        resolved = path.resolve() if path.exists() else path
        return resolved

    def _record(self, path, kind, status, data=None, bytes_parsed=0, error=None):
        rel = os.path.relpath(path, self.data_root)
        entry = {
            "order": len(self.log),
            "path": rel,
            "kind": kind,
            "status": status,
            "size": None,
            "mtime_ns": None,
            "sha256": None,
            "bytes_parsed": int(bytes_parsed),
            "error": error,
            "is_symlink": os.path.islink(path),
        }
        if data is not None:
            st = os.stat(path)
            entry.update(size=len(data), mtime_ns=int(st.st_mtime_ns),
                         sha256=hashlib.sha256(data).hexdigest())
        self.log.append(entry)

    def _read(self, path: Path, kind: str):
        self._check(path)
        if not os.path.exists(path):  # broken symlink: annex content not fetched
            self._record(path, kind, "annex_not_fetched" if os.path.islink(path)
                         else "missing")
            return None
        with open(path, "rb") as fh:
            data = fh.read()
        return data

    def json(self, path: Path):
        key = str(path)
        if key in self._json_cache:
            return self._json_cache[key]
        data = self._read(path, "json")
        obj = None
        if data is not None:
            try:
                obj = json.loads(data.decode("utf-8-sig"))
                self._record(path, "json", "read", data, len(data))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._record(path, "json", "parse_error", data, 0, str(exc))
        self._json_cache[key] = obj
        return obj

    def events_header(self, path: Path):
        key = str(path)
        if key in self._header_cache:
            return self._header_cache[key]
        data = self._read(path, "events_header")
        cols = None
        if data is not None:
            first = data[:MAX_HEADER_BYTES].split(b"\n", 1)[0]
            try:
                line = first.decode("utf-8-sig").rstrip("\r")
                cols = [c.strip() for c in line.split("\t") if c.strip()]
                self._record(path, "events_header", "read", data, len(first))
            except UnicodeDecodeError as exc:
                self._record(path, "events_header", "parse_error", data, 0, str(exc))
        self._header_cache[key] = cols
        return cols


# --------------------------------------------------------------------------
# BIDS file discovery (listing only)
# --------------------------------------------------------------------------
_ENTITY_RE = re.compile(r"([a-zA-Z0-9]+)-([a-zA-Z0-9]+)")


def parse_bids_name(name: str):
    """``(entities, suffix)`` of a BIDS file name (extension stripped)."""
    stem = name.split(".", 1)[0]
    parts = stem.split("_")
    suffix = parts[-1] if parts and "-" not in parts[-1] else None
    ents = {}
    for p in parts[:-1] if suffix else parts:
        m = _ENTITY_RE.fullmatch(p)
        if m:
            ents[m.group(1)] = m.group(2)
    return ents, suffix


def discover(dataset_root: Path):
    """Recordings (listing only) and the metadata files per directory."""
    recs = []
    meta_by_dir = {}
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith("."))
        meta = [f for f in sorted(filenames) if METADATA_OPEN_RE.search(f)]
        if meta:
            meta_by_dir[Path(dirpath)] = meta
        seen = set()
        for f in sorted(filenames):
            for modality, pat in RECORDING_PATTERNS.items():
                if pat.search(f):
                    stem = pat.sub("", f)
                    if (modality, stem) in seen:
                        continue
                    seen.add((modality, stem))
                    p = Path(dirpath) / f
                    ents, suffix = parse_bids_name(f)
                    recs.append({
                        "path": p, "modality": modality, "entities": ents,
                        "suffix": suffix, "stem": stem,
                        "data_fetched": os.path.exists(p),
                    })
    return recs, meta_by_dir


def _applicable(meta_by_dir, rec, dataset_root, suffix, ext):
    """Inheritance chain (least to most specific) of sidecars for ``rec``."""
    chain = []
    rec_dir = rec["path"].parent
    dirs = [dataset_root]
    rel = rec_dir.relative_to(dataset_root)
    cur = dataset_root
    for part in rel.parts:
        cur = cur / part
        dirs.append(cur)
    for d in dirs:
        for f in meta_by_dir.get(d, []):
            if not f.endswith(ext):
                continue
            ents, suf = parse_bids_name(f)
            if suf != suffix:
                continue
            if all(rec["entities"].get(k) == v for k, v in ents.items()):
                chain.append((len(ents), d / f))
    chain.sort(key=lambda t: t[0])
    return [p for _, p in chain]


# --------------------------------------------------------------------------
# definedness rules
# --------------------------------------------------------------------------
def _levels(events_json):
    """``{column: [level labels]}`` described in an ``*_events.json`` sidecar."""
    out = {}
    if not isinstance(events_json, dict):
        return out
    for col, desc in events_json.items():
        if isinstance(desc, dict) and isinstance(desc.get("Levels"), dict):
            out[str(col).lower()] = [str(k) for k in desc["Levels"].keys()]
    return out


def _any(regex, levels):
    return any(regex.search(str(v)) for v in levels)


def _event_status(rec, cols, label_cols):
    """Events-level precondition for rules that read ``label_cols``."""
    if cols is None:
        if rec.get("events_path") is None:
            return (NOT_DEFINABLE, "no_events_file")
        return (METADATA_UNAVAILABLE, "events_not_readable")
    if not any(c in cols for c in label_cols):
        return (NOT_DEFINABLE, "no_label_column:" + "|".join(label_cols))
    return None


def _level_rule(rec, cols, levels_by_col, label_cols, checks, extra_cols=(),
                any_of_cols=None):
    """
    DEFINABLE when the described levels of the present label columns pass
    ``checks``; REQUIRES_EVENT_VALUES when they fail only because some present
    label column has no level description; NOT_DEFINABLE otherwise, and
    whenever a required column is absent from the header (every column of
    ``extra_cols``; at least one column of ``any_of_cols = (name, columns)``).
    """
    pre = _event_status(rec, cols, label_cols)
    if pre is not None:
        return pre
    missing_cols = [c for c in extra_cols if c not in cols]
    if missing_cols:
        return (NOT_DEFINABLE, "missing_columns:" + ",".join(missing_cols))
    if any_of_cols is not None and not any(c in cols for c in any_of_cols[1]):
        return (NOT_DEFINABLE, f"missing_columns:{any_of_cols[0]}")
    present = [c for c in label_cols if c in cols]
    described = [lv for c in present for lv in levels_by_col.get(c, [])]
    undescribed = [c for c in present if c not in levels_by_col]
    failed = [name for name, ok in checks(described) if not ok]
    if not failed:
        return (DEFINABLE, None)
    if undescribed:
        return (REQUIRES_EVENT_VALUES, "levels_not_described:" + ",".join(undescribed))
    return (NOT_DEFINABLE, "no_levels:" + ",".join(failed))


def evaluate_recording(rec, sidecar, columns, levels_by_col, subject_tasks,
                       nas_hub_declared=False):
    """
    Status and reason per (principle, channel) for one recording.
    ``columns``: events header (None without an events file);
    ``levels_by_col``: level labels described per column in events sidecars;
    ``nas_hub_declared``: whether the protocol declares the NAS hub
    (``workspace_nodes``); without it NAS capacity is ``NOT_DEFINABLE``
    (``NO_DECLARED_WORKSPACE``) whatever the recording.
    """
    out = {}
    levels_by_col = dict(levels_by_col or {})
    timing_key = "RepetitionTime" if rec["modality"] == "fmri" else "SamplingFrequency"
    if sidecar is None:
        timing = (METADATA_UNAVAILABLE, "no_sidecar")
    elif sidecar.get(timing_key) in (None, "", "n/a"):
        timing = (NOT_DEFINABLE, f"missing_{timing_key}")
    else:
        timing = (DEFINABLE, None)
    out[("PDI", "repertoire")] = timing
    out[("NAS", "capacity")] = (
        timing if (nas_hub_declared or timing[0] != DEFINABLE)
        else (NOT_DEFINABLE, NO_DECLARED_WORKSPACE)
    )
    out[("IIM", "default")] = timing
    has_rest = any(REST_TASK_RE.search(t or "") for t in subject_tasks)
    out[("PDI", "legacy_baseline")] = (
        timing if timing[0] != DEFINABLE else
        ((DEFINABLE, None) if has_rest else (NOT_DEFINABLE, "no_rest_recording"))
    )

    cols = None if columns is None else {c.lower() for c in columns}
    # Strict RAM (the pipeline default, require_explicit_feedback=True): the
    # feedback events are trial_type rows matching FEEDBACK_RE, and their
    # values come from a numeric column of FEEDBACK_VALUE_COLUMNS
    # (event_parsing._ram_fields); without such a column RAM is undefined
    # (missing_explicit_feedback) whatever the levels.
    fb_value = ("feedback_value", FEEDBACK_VALUE_COLUMNS)

    def _ram_checks(lv):
        return [("goal", _any(GOAL_RE, lv)), ("stimulus", _any(STIM_RE, lv)),
                ("feedback", _any(FEEDBACK_RE, lv))]

    def _srpi_checks(lv):
        # the estimator's own classifier (token rules, "non-self" is not self)
        self_m, nonself_m = classify_self_nonself(pd.Series([str(v) for v in lv],
                                                            dtype=object))
        return [("self", bool(self_m.any())), ("nonself", bool(nonself_m.any()))]

    def _agency_checks(lv):
        low = {_norm_trial_type(v) for v in lv}
        return [(AGENCY_SELF_LABEL, AGENCY_SELF_LABEL in low),
                (AGENCY_OTHER_LABEL, AGENCY_OTHER_LABEL in low)]

    def _gate(res):
        # an event-definable principle still needs the recording's timing
        return timing if (res[0] == DEFINABLE and timing[0] != DEFINABLE) else res

    tt = ("trial_type",)
    out[("RAM", "untyped")] = _gate(_level_rule(rec, cols, levels_by_col, tt,
                                                _ram_checks, any_of_cols=fb_value))
    ev_pre = _event_status(rec, cols, tt)
    for ch in IMPACT_CHANNELS:
        if ch not in IMPLEMENTED_RAM_CHANNELS:
            out[("RAM", ch)] = (NOT_IMPLEMENTED, "NOT_IMPLEMENTED")
        elif ev_pre is not None:
            out[("RAM", ch)] = ev_pre
        elif IMPACT_CHANNEL_COLUMN not in cols:
            out[("RAM", ch)] = (NOT_DEFINABLE, "no_impact_channel_column")
        else:
            lv = levels_by_col.get(IMPACT_CHANNEL_COLUMN)
            if lv is None:
                out[("RAM", ch)] = (REQUIRES_EVENT_VALUES,
                                    "levels_not_described:impact_channel")
            elif ch in {str(v).strip().lower() for v in lv}:
                out[("RAM", ch)] = _gate(_level_rule(rec, cols, levels_by_col, tt,
                                                     _ram_checks,
                                                     any_of_cols=fb_value))
            else:
                out[("RAM", ch)] = (NOT_DEFINABLE, "channel_not_in_levels")
    out[("SRPI", "legacy_self_other")] = _gate(_level_rule(
        rec, cols, levels_by_col, SRPI_TEXT_COLUMNS, _srpi_checks))
    out[("SRPI", "agency")] = _gate(_level_rule(
        rec, cols, levels_by_col, tt, _agency_checks,
        extra_cols=("yoked_to", "phase_bin")))
    return out


# --------------------------------------------------------------------------
# audit driver
# --------------------------------------------------------------------------
def protocol_nas_hub(protocol):
    """
    ``(info, hub_declared)`` of the protocol the audit applies (JSON path,
    dict, ``evidence.Protocol`` or None): its source and hash, and whether it
    declares the NAS hub (``workspace_nodes`` of its NAS options).
    """
    from impact_pipeline.evidence import resolve_protocol

    source = None
    if isinstance(protocol, (str, os.PathLike)):
        path = Path(protocol).resolve()
        try:
            source = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            source = path.name
    proto = resolve_protocol(protocol)
    if proto is None:
        return {"source": None, "name": None, "hash": None,
                "nas_hub_declared": False}, False
    declared = proto.estimator_options("NAS").get("workspace_nodes") is not None
    return {"source": source, "name": proto.name, "hash": proto.hash,
            "nas_hub_declared": declared}, declared


def audit_dataset(reader, dataset_root: Path, dataset_id: str,
                  nas_hub_declared=False):
    recs, meta_by_dir = discover(dataset_root)
    desc_path = dataset_root / "dataset_description.json"
    description = reader.json(desc_path) if desc_path.exists() or os.path.islink(
        desc_path) else None
    # rest recordings of the same subject *and modality* (the legacy PDI
    # baseline is a recording of the same kind as the evaluated run)
    tasks_by_subject = {}
    for r in recs:
        tasks_by_subject.setdefault((r["entities"].get("sub"), r["modality"]),
                                    set()).add(r["entities"].get("task"))
    rows = []
    for r in recs:
        sidecar_paths = _applicable(meta_by_dir, r, dataset_root, r["suffix"], ".json")
        sidecar = None
        for p in sidecar_paths:
            obj = reader.json(p)
            if isinstance(obj, dict):
                sidecar = {**(sidecar or {}), **obj}
        ev_paths = _applicable(meta_by_dir, r, dataset_root, "events", "_events.tsv")
        r["events_path"] = ev_paths[-1] if ev_paths else None
        columns = reader.events_header(r["events_path"]) if r["events_path"] else None
        evj_paths = _applicable(meta_by_dir, r, dataset_root, "events", ".json")
        levels = {}
        for p in evj_paths:  # least to most specific: later files override
            levels.update(_levels(reader.json(p)))
        cells = evaluate_recording(
            r, sidecar, columns, levels,
            tasks_by_subject.get((r["entities"].get("sub"), r["modality"]), set()),
            nas_hub_declared=nas_hub_declared)
        base = {
            "dataset": dataset_id,
            "recording": os.path.relpath(r["path"], dataset_root),
            "modality": r["modality"],
            "subject": r["entities"].get("sub"),
            "session": r["entities"].get("ses"),
            "task": r["entities"].get("task"),
            "run": r["entities"].get("run"),
            "data_fetched": bool(r["data_fetched"]),
            "sidecar_found": sidecar is not None,
            "events_file": (os.path.relpath(r["events_path"], dataset_root)
                            if r["events_path"] else None),
            "events_columns": ",".join(columns) if columns else None,
            "levels_described": ",".join(sorted(levels)) or None,
            "has_choice_reward": bool(columns and {"choice", "reward"} <= {
                c.lower() for c in columns}),
        }
        for (principle, channel), (status, reason) in cells.items():
            rows.append({**base, "principle": principle, "channel": channel,
                         "status": status, "reason": reason})
    return rows, {"n_recordings": len(recs), "description": {
        k: (description or {}).get(k) for k in ("Name", "BIDSVersion", "DatasetType")
    }}


def coverage_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["dataset", "modality", "task", "principle",
                                     "channel", "n_recordings", *STATUSES,
                                     "top_reason"])
    keys = ["dataset", "modality", "task", "principle", "channel"]
    counts = (rows.groupby(keys + ["status"]).size().unstack("status", fill_value=0)
              .reindex(columns=list(STATUSES), fill_value=0))
    counts.insert(0, "n_recordings", counts.sum(axis=1))
    reasons = (rows.dropna(subset=["reason"]).groupby(keys)["reason"]
               .agg(lambda s: s.value_counts().index[0]))
    out = counts.join(reasons.rename("top_reason")).reset_index()
    return out


def _provenance():
    try:
        from impact_pipeline.provenance import collect_code_version

        info = collect_code_version(REPO_ROOT)
    except Exception as exc:  # noqa: BLE001 - provenance must not fail a run
        info = {"code_version": "unknown", "error": str(exc)}
    info["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return info


def run_audit(data_root, out_dir, datasets=DEFAULT_DATASETS,
              protocol=DEFAULT_PROTOCOL) -> dict:
    data_root = Path(data_root).resolve()
    out = Path(out_dir).resolve()
    if out == data_root or data_root in out.parents:
        raise ValueError("the output directory must lie outside the data root")
    protocol_info, hub_declared = protocol_nas_hub(protocol)
    out.mkdir(parents=True, exist_ok=True)
    reader = MetadataReader(data_root)
    all_rows, ds_info = [], {}
    t0 = time.time()
    for ds in datasets:
        root = data_root / ds
        prior = PRIOR_ACCESS.get(ds, DEFAULT_PRIOR_ACCESS)
        if not root.is_dir():
            ds_info[ds] = {"present": False, "prior_access": prior}
            continue
        rows, info = audit_dataset(reader, root, ds, nas_hub_declared=hub_declared)
        all_rows.extend(rows)
        ds_info[ds] = {"present": True, "prior_access": prior, **info,
                       "confirmatory_eligible": prior != "exploratory_calibration"}
    rec = pd.DataFrame(all_rows)
    rec.to_csv(out / "definedness_recordings.csv", index=False)
    matrix = coverage_matrix(rec)
    matrix.to_csv(out / "definedness_matrix.csv", index=False)
    log = pd.DataFrame(reader.log)
    log.to_csv(out / "definedness_access_log.csv", index=False)
    opened = [e["path"] for e in reader.log]
    summary = {
        "version": AUDIT_VERSION,
        "data_root": str(data_root),
        "datasets": ds_info,
        "n_recordings": int(rec["recording"].nunique()) if not rec.empty else 0,
        "n_files_opened": int(sum(e["status"] not in ("annex_not_fetched", "missing")
                                  for e in reader.log)),
        "n_annex_not_fetched": int(sum(e["status"] == "annex_not_fetched"
                                       for e in reader.log)),
        "bytes_hashed": int(sum(e["size"] or 0 for e in reader.log)),
        "time_series_opened": False,
        "only_metadata_opened": all(METADATA_OPEN_RE.search(p) for p in opened),
        "open_policy": {"allowed": METADATA_OPEN_RE.pattern, "mode": "rb",
                        "events_tsv": "header line only"},
        "status_definitions": list(STATUSES),
        "protocol": protocol_info,
        "seconds": round(time.time() - t0, 3),
        "provenance": _provenance(),
    }
    with open(out / "definedness_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return {"summary": summary, "recordings": rec, "matrix": matrix, "log": log}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data" / "scratch"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument(
        "--protocol", default=str(DEFAULT_PROTOCOL),
        help="MPC protocol whose NAS hub (workspace_nodes) decides whether NAS "
             "capacity is definable (default: protocols/mpc_default_v1.json, "
             "which declares none; 'none': no protocol, no hub)")
    args = ap.parse_args(argv)
    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    protocol = (None if str(args.protocol).strip().lower() in ("none", "")
                else args.protocol)
    res = run_audit(args.data_root, args.out, datasets, protocol=protocol)
    s = res["summary"]
    keys = ("n_recordings", "n_files_opened", "n_annex_not_fetched",
            "only_metadata_opened")
    print(json.dumps({k: s[k] for k in keys}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
