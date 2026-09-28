import os
import csv
import logging
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np
from bids import BIDSLayout

log = logging.getLogger(__name__)

# Channel types kept as network nodes. Everything else (EOG, EMG, ECG, misc,
# stim, ...) is excluded before the per-subject channel intersection.
EEG_NODE_TYPE = "eeg"
_BIDS_CHANNEL_TYPE_MAP = {
    "EEG": "eeg",
    "EOG": "eog",
    "HEOG": "eog",
    "VEOG": "eog",
    "EMG": "emg",
    "ECG": "ecg",
    "EKG": "ecg",
    "MISC": "misc",
    "TRIG": "stim",
    "RESP": "resp",
    "GSR": "gsr",
    "TEMP": "temperature",
    "REF": "misc",
    "AUDIO": "misc",
    "PD": "misc",
    "EYEGAZE": "misc",
    "PUPIL": "misc",
    "SYSCLOCK": "misc",
    "ADC": "misc",
    "DAC": "misc",
}
# Some datasets (e.g. ds005620) declare ocular/muscle channels as type EEG in
# both the BrainVision header and channels.tsv. Names that unambiguously denote
# non-EEG sensors are therefore reclassified.
_NON_EEG_NAME_PATTERNS = (
    (re.compile(r"eog", re.I), "eog"),
    (re.compile(r"emg", re.I), "emg"),
    (re.compile(r"^(ecg|ekg)", re.I), "ecg"),
)
# Run selectors for session/rest rules: (task, acq[, selector]).
RUN_SELECTORS = ("all", "first", "after_first")


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-12
    return (x - mu) / sd


def _parse_rule(rule) -> Tuple[str, str, str]:
    if len(rule) == 2:
        task, acq = rule
        selector = "all"
    elif len(rule) == 3:
        task, acq, selector = rule
        selector = "all" if selector is None else str(selector)
    else:
        raise ValueError(
            f"EEG session rule must be (task, acq[, run_selector]); got {rule!r}"
        )
    if selector not in RUN_SELECTORS:
        raise ValueError(
            f"Unknown EEG run selector {selector!r}; expected one of {RUN_SELECTORS}."
        )
    return str(task), str(acq), selector


def _bids_run_id(fn: str) -> Optional[int]:
    m = re.search(r"_run-0*([0-9]+)", os.path.basename(str(fn)))
    return int(m.group(1)) if m else None


def _apply_run_selector(files: Sequence[str], selector: str) -> List[str]:
    ordered = sorted(files, key=lambda f: ((_bids_run_id(f) or 0), f))
    if selector == "first":
        return ordered[:1]
    if selector == "after_first":
        return ordered[1:]
    return ordered


def _collect_runs_for_rule(
    layout: BIDSLayout,
    subject: str,
    task: str,
    acq: str,
) -> List[str]:
    files = layout.get(
        subject=subject,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
        task=task,
        acquisition=acq,
        return_type="filename",
    )
    return sorted(files)


def _existing_files(
    files: Sequence[str],
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    missing_files: List[Dict[str, Any]],
) -> List[str]:
    keep: List[str] = []
    for fn in files:
        if os.path.exists(fn):
            keep.append(fn)
        else:
            log.warning("Skipping missing EEG file referenced by BIDS index: %s", fn)
            missing_files.append(
                {
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "acquisition": acquisition,
                    "file": fn,
                    "reason": "missing_on_disk",
                }
            )
    return keep


def _build_session_runs_with_rules(
    layout: BIDSLayout,
    subject: str,
    session_rules: Dict[str, Sequence[Tuple[str, ...]]],
    missing_files: List[Dict[str, Any]],
    exclude: Optional[Dict[str, Iterable[str]]] = None,
    required_task: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Tuple[str, str, str]]]:
    out: Dict[str, List[str]] = {k: [] for k in session_rules}
    matched: Dict[str, Tuple[str, str, str]] = {}
    for session, rules in session_rules.items():
        excluded = {os.path.abspath(f) for f in (exclude or {}).get(session, ())}
        for rule in rules:
            task, acq, selector = _parse_rule(rule)
            if (
                required_task is not None
                and session in required_task
                and task != required_task[session]
            ):
                # Baselines must come from the same state (task) as the analysed runs.
                continue
            hits = _collect_runs_for_rule(layout, subject=subject, task=task, acq=acq)
            hits = _existing_files(
                hits,
                subject=subject,
                session=session,
                task=task,
                acquisition=acq,
                missing_files=missing_files,
            )
            hits = [
                f
                for f in _apply_run_selector(hits, selector)
                if os.path.abspath(f) not in excluded
            ]
            if hits:
                # Use all selected runs of the first matching rule to avoid
                # discarding valid recordings from the selected session definition.
                out[session].extend(hits)
                matched[session] = (task, acq, selector)
                break
    return out, matched


def _build_session_runs(
    layout: BIDSLayout,
    subject: str,
    session_rules: Dict[str, Sequence[Tuple[str, ...]]],
    missing_files: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    out, _matched = _build_session_runs_with_rules(
        layout, subject, session_rules, missing_files
    )
    return out


def _channels_tsv_for(vhdr_path: str) -> Optional[str]:
    base = re.sub(r"_eeg\.vhdr$", "_channels.tsv", str(vhdr_path))
    if base != str(vhdr_path) and os.path.exists(base):
        return base
    return None


def read_bids_channels(vhdr_path: str) -> Dict[str, Dict[str, str]]:
    """Read the BIDS channels.tsv next to a recording (name -> {type, status})."""
    path = _channels_tsv_for(vhdr_path)
    if path is None:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            name = str(row.get("name") or "").strip()
            if name:
                out[name] = {
                    "type": str(row.get("type") or "").strip(),
                    "status": str(row.get("status") or "").strip().lower(),
                }
    return out


def classify_eeg_channels(
    ch_names: Sequence[str],
    mne_types: Sequence[str],
    bids_channels: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Decide which channels are scalp EEG nodes.

    The channel type is taken from BIDS channels.tsv when present, otherwise
    from the reader; channels whose names denote EOG/EMG/ECG sensors are
    reclassified even when declared as EEG. Channels marked ``status=bad`` in
    channels.tsv are excluded as well.
    """
    bids_channels = bids_channels or {}
    keep: List[str] = []
    excluded: List[Dict[str, str]] = []
    for name, mne_type in zip(ch_names, mne_types):
        ch_type = str(mne_type).lower()
        source = "reader"
        entry = bids_channels.get(name)
        if entry is not None and entry.get("type"):
            mapped = _BIDS_CHANNEL_TYPE_MAP.get(entry["type"].upper())
            if mapped is not None:
                ch_type = mapped
                source = "channels.tsv"
        if ch_type == EEG_NODE_TYPE:
            for pattern, pattern_type in _NON_EEG_NAME_PATTERNS:
                if pattern.search(str(name)):
                    ch_type = pattern_type
                    source = "name_pattern"
                    break
        if ch_type != EEG_NODE_TYPE:
            excluded.append({"channel": str(name), "type": ch_type, "source": source})
            continue
        if entry is not None and entry.get("status") == "bad":
            excluded.append(
                {"channel": str(name), "type": "bad", "source": "channels.tsv"}
            )
            continue
        keep.append(str(name))
    return {"eeg": keep, "excluded": excluded}


def _output_run_ids(files: Sequence[str]) -> Tuple[List[int], str]:
    """Keep BIDS run numbers in output names; fall back to 1..n only if ambiguous."""
    ordered = sorted(files, key=lambda f: ((_bids_run_id(f) or 0), f))
    ids = [_bids_run_id(f) for f in ordered]
    if len(ordered) == 1 and ids[0] is None:
        return [1], "single_run_without_run_entity"
    if all(i is not None for i in ids) and len(set(ids)) == len(ids):
        return [int(i) for i in ids], "bids_run_entity"
    return list(range(1, len(ordered) + 1)), "sequential"


def run_preprocessing_eeg(
    bids_root: str,
    out_root: str,
    subjects: Optional[Iterable[str]] = None,
    session_rules: Optional[Dict[str, Sequence[Tuple[str, ...]]]] = None,
    condition_label: str = "eeg",
    atlas_key: str = "eeg64",
    target_sfreq: float = 250.0,
    l_freq: float = 0.5,
    h_freq: float = 45.0,
    max_duration_sec: Optional[float] = 120.0,
    min_common_channels: int = 16,
    rest_rules: Optional[Dict[str, Sequence[Tuple[str, ...]]]] = None,
    rest_label: str = "rest",
) -> Dict[str, Any]:
    """
    Convert raw EEG-BIDS recordings into standardized time×channel arrays.

    Output layout (``<stem>`` = ``{subject}_run-{k}_{atlas_key}_ts.npy``):
        {out_root}/{subject}/{session}/{condition_label}/<stem>
        {out_root}/{subject}/{session}/{rest_label}/<stem>     (with rest_rules)

    ``k`` is the BIDS run number when the recordings carry one. Session and
    rest rules are ``(task, acq[, run_selector])`` with run_selector one of
    'all' (default), 'first', 'after_first'. Rest/baseline recordings must come
    from the same task (state) as the analysed runs of that session and never
    reuse an analysed recording; sessions without such data get no rest folder
    (PDI then reports an explicit missing-baseline reason).
    """
    # Ensure numba has a writable cache location before importing MNE.
    if not os.environ.get("NUMBA_CACHE_DIR", "").strip():
        default_cache = Path(tempfile.gettempdir()) / "numba_cache"
        default_cache.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(default_cache)

    try:
        import mne  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "EEG preprocessing requires MNE-Python. "
            "Install with: conda install -c conda-forge mne"
        ) from exc

    if session_rules is None:
        # Default mapping for ds005620:
        # awake/EC for wake baseline; sed2/rest preferred for deep sedation,
        # fallback to sed/rest when sed2 is unavailable.
        session_rules = {
            "awake": [("awake", "EC")],
            "deep": [("sed2", "rest"), ("sed", "rest")],
        }
    if rest_rules and str(rest_label) == str(condition_label):
        raise ValueError(
            "rest_label must differ from condition_label when rest_rules are given."
        )

    layout = BIDSLayout(bids_root, validate=False)
    all_subjects = sorted(layout.get(return_type="id", target="subject"))
    if not all_subjects:
        raise FileNotFoundError(
            f"No EEG subjects found under '{bids_root}'. "
            "Check --bids-root and ensure data files are present."
        )
    if subjects is None:
        subjects = all_subjects
    else:
        wanted = {str(s).strip().replace("sub-", "", 1) for s in subjects}
        subjects = [s for s in all_subjects if s in wanted]

    summary: Dict[str, Any] = {
        "subjects_requested": list(subjects),
        "missing_files": [],
        "skipped_subjects": [],
        "written_runs": [],
        "excluded_channels": [],
        "missing_rest": [],
        "channel_policy": (
            "scalp EEG only: channels typed EOG/EMG/ECG/misc/stim in the reader or "
            "BIDS channels.tsv, channels whose names denote EOG/EMG/ECG, and "
            "channels.tsv status=bad are excluded"
        ),
        "rest_policy": (
            None
            if not rest_rules
            else "rest/baseline = recordings of the same state (task) as the analysed "
            "runs, disjoint from them; written under <session>/" + str(rest_label)
        ),
    }

    def _read_header(fn, session):
        try:
            return mne.io.read_raw_brainvision(fn, preload=False, verbose="ERROR")
        except FileNotFoundError:
            log.warning("Missing EEG file at read time: %s", fn)
            reason = "missing_at_read_time"
        except Exception as exc:
            log.warning("Unreadable EEG file %s (%s)", fn, type(exc).__name__)
            reason = f"header_read_error:{type(exc).__name__}"
        summary["missing_files"].append(
            {
                "subject": subj,
                "session": session,
                "task": None,
                "acquisition": None,
                "file": fn,
                "reason": reason,
            }
        )
        return None

    for subj in subjects:
        run_map, matched_rules = _build_session_runs_with_rules(
            layout,
            subj,
            session_rules,
            missing_files=summary["missing_files"],
        )

        # Skip subjects without both comparison sessions.
        missing_sessions = [s for s, files in run_map.items() if len(files) == 0]
        if missing_sessions:
            log.warning(
                "Skipping sub-%s: missing EEG runs for sessions=%s",
                subj,
                ",".join(missing_sessions),
            )
            summary["skipped_subjects"].append(
                {
                    "subject": subj,
                    "reason": "missing_sessions",
                    "detail": ",".join(missing_sessions),
                }
            )
            continue

        rest_map: Dict[str, List[str]] = {}
        if rest_rules:
            rest_map, _rest_matched = _build_session_runs_with_rules(
                layout,
                subj,
                {k: v for k, v in rest_rules.items() if k in run_map},
                missing_files=summary["missing_files"],
                exclude=run_map,
                required_task={s: rule[0] for s, rule in matched_rules.items()},
            )

        # Build per-subject common EEG channel set across all selected runs.
        usable_map: Dict[str, List[str]] = {k: [] for k in run_map}
        eeg_sets = []
        excluded_seen = set()
        header_channels: Dict[str, set] = {}
        for session, files in run_map.items():
            for fn in files:
                raw = _read_header(fn, session)
                if raw is None:
                    continue
                cls = classify_eeg_channels(
                    raw.ch_names, raw.get_channel_types(), read_bids_channels(fn)
                )
                for rec in cls["excluded"]:
                    key = (rec["channel"], rec["type"], rec["source"])
                    if key not in excluded_seen:
                        excluded_seen.add(key)
                        summary["excluded_channels"].append({"subject": subj, **rec})
                eeg_sets.append(set(cls["eeg"]))
                usable_map[session].append(fn)
                raw.close()

        missing_sessions = [s for s, files in usable_map.items() if len(files) == 0]
        if missing_sessions:
            log.warning(
                "Skipping sub-%s: no readable EEG runs for sessions=%s",
                subj,
                ",".join(missing_sessions),
            )
            summary["skipped_subjects"].append(
                {
                    "subject": subj,
                    "reason": "no_readable_runs",
                    "detail": ",".join(missing_sessions),
                }
            )
            continue

        common_channels = sorted(set.intersection(*eeg_sets)) if eeg_sets else []
        if len(common_channels) < int(min_common_channels):
            log.warning(
                "Skipping sub-%s: only %d common EEG channels (<%d).",
                subj,
                len(common_channels),
                min_common_channels,
            )
            summary["skipped_subjects"].append(
                {
                    "subject": subj,
                    "reason": "insufficient_common_channels",
                    "detail": str(len(common_channels)),
                }
            )
            continue

        # Rest/baseline recordings must provide every analysed channel so the
        # baseline matrices have the same nodes as the analysed runs.
        usable_rest: Dict[str, List[str]] = {}
        for session in run_map:
            if not rest_rules or session not in rest_rules:
                continue
            for fn in rest_map.get(session, []):
                raw = _read_header(fn, f"{session}/{rest_label}")
                if raw is None:
                    continue
                cls = classify_eeg_channels(
                    raw.ch_names, raw.get_channel_types(), read_bids_channels(fn)
                )
                header_channels[fn] = set(cls["eeg"])
                raw.close()
                if not set(common_channels).issubset(header_channels[fn]):
                    summary["missing_rest"].append(
                        {
                            "subject": subj,
                            "session": session,
                            "file": fn,
                            "reason": "rest_missing_channels",
                        }
                    )
                    continue
                usable_rest.setdefault(session, []).append(fn)
            if not usable_rest.get(session):
                reason = (
                    "no_usable_rest_recording"
                    if rest_map.get(session)
                    else "no_rest_recording_for_state"
                )
                summary["missing_rest"].append(
                    {
                        "subject": subj,
                        "session": session,
                        "file": None,
                        "reason": reason,
                    }
                )

        segments = [
            (session, condition_label, files, "analysis")
            for session, files in usable_map.items()
        ]
        segments.extend(
            (session, rest_label, files, "rest")
            for session, files in usable_rest.items()
        )
        for session, folder, files, segment in segments:
            out_dir = os.path.join(out_root, subj, session, folder)
            os.makedirs(out_dir, exist_ok=True)
            ordered = sorted(files, key=lambda f: ((_bids_run_id(f) or 0), f))
            run_ids, run_id_source = _output_run_ids(ordered)
            for i, fn in zip(run_ids, ordered):
                try:
                    raw = mne.io.read_raw_brainvision(fn, preload=True, verbose="ERROR")
                except Exception as exc:
                    log.warning(
                        "Skipping run due read failure %s (%s)", fn, type(exc).__name__
                    )
                    summary["missing_files"].append(
                        {
                            "subject": subj,
                            "session": session,
                            "task": None,
                            "acquisition": None,
                            "file": fn,
                            "reason": f"run_read_error:{type(exc).__name__}",
                        }
                    )
                    continue

                raw.pick(common_channels)
                if target_sfreq and float(raw.info["sfreq"]) != float(target_sfreq):
                    raw.resample(float(target_sfreq), npad="auto", verbose="ERROR")
                if (l_freq is not None) or (h_freq is not None):
                    raw.filter(
                        l_freq=l_freq,
                        h_freq=h_freq,
                        fir_design="firwin",
                        verbose="ERROR",
                    )
                x = raw.get_data(reject_by_annotation="omit")
                if max_duration_sec is not None:
                    keep = int(float(max_duration_sec) * float(raw.info["sfreq"]))
                    if keep > 0:
                        x = x[:, : min(keep, x.shape[1])]

                x = _zscore_rows(x)
                ts_time_channel = x.T
                out_fn = os.path.join(out_dir, f"{subj}_run-{i}_{atlas_key}_ts.npy")
                np.save(out_fn, ts_time_channel)
                summary["written_runs"].append(
                    {
                        "subject": subj,
                        "session": session,
                        "segment": segment,
                        "source_file": fn,
                        "output_file": out_fn,
                        "run_index": int(i),
                        "bids_run": _bids_run_id(fn),
                        "run_index_source": run_id_source,
                        "n_channels": int(ts_time_channel.shape[1]),
                        "n_timepoints": int(ts_time_channel.shape[0]),
                        "sfreq_hz": float(raw.info["sfreq"]),
                    }
                )
                raw.close()

    requested = len(summary["subjects_requested"])
    skipped = len({r["subject"] for r in summary["skipped_subjects"]})
    processed = len({r["subject"] for r in summary["written_runs"]})
    summary["summary"] = {
        "subjects_requested": int(requested),
        "subjects_processed": int(processed),
        "subjects_skipped": int(skipped),
        "missing_file_records": int(len(summary["missing_files"])),
        "written_runs": int(len(summary["written_runs"])),
        "written_rest_runs": int(
            sum(1 for r in summary["written_runs"] if r.get("segment") == "rest")
        ),
        "excluded_channel_records": int(len(summary["excluded_channels"])),
        "missing_rest_records": int(len(summary["missing_rest"])),
    }
    return summary
