"""
Single source of truth for BIDS ``events.tsv`` parsing used by RAM and SRPI.

Both the compute path (``run_synergy_ci.load_onsets``) and the readiness check
(``mpc_readiness.check_mpc_readiness``) import the label patterns, the
events-file resolution and the table-to-bundle conversion from here, so a run
that readiness reports as ready is parsed identically when metrics are
computed.

Contract ("measured inputs only"):
  * no implicit stimuli: when no row matches the stimulus pattern the bundle
    has no stimulus onsets (RAM undefined) unless ``allow_implicit_stimuli``;
  * feedback values are taken only from feedback rows, and onsets/values are
    kept pairwise aligned (rows without a finite value are dropped together);
  * ``response_time`` is a behavioural latency, not environmental feedback,
    and is used as a feedback value only with ``allow_response_time_feedback``;
  * events files are resolved per session exactly; there is no cross-session
    glob fallback. The only other task labels accepted are the explicit
    per-subject aliases of ``dataset_catalog.DATASET_TASK_ALIASES`` (e.g.
    ds003171 sub-10JR's ``task-audio`` for ``audioawake``), the same table
    preprocessing uses, so event-based run selection matches the
    preprocessed tree.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from impact_pipeline.dataset_catalog import DATASET_TASK_ALIASES, task_labels_for_state

# Label patterns (case-insensitive). Stimulus/goal/feedback patterns match
# substrings of ``trial_type`` (e.g. "audio_stim", "goal_cue",
# "feedback_reward").
STIM_RE = re.compile(r"audio|stim|tone|target|event", re.I)
GOAL_RE = re.compile(r"goal|objective|intent|instruction|cue", re.I)
FEEDBACK_RE = re.compile(
    r"feedback|reward|error|outcome|correct|incorrect|result",
    re.I,
)

# Self/non-self tokens are matched as whole tokens, where '_', '-', spaces,
# punctuation and camelCase boundaries separate tokens ("self_name",
# "SelfName" -> self; "OtherName" -> non-self; "unknown", "time", "mother" ->
# no match).
_TOK_L = r"(?<![a-z0-9])"
_TOK_R = r"(?![a-z0-9])"
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
NONSELF_RE = re.compile(
    rf"{_TOK_L}(?:non[-_ ]?self|other|others|othername|another|stranger|"
    rf"strangers|third[-_ ]?person){_TOK_R}",
    re.I,
)
SELF_RE = re.compile(
    rf"{_TOK_L}(?:self|selfname|own|me|my|mine|myname|own[-_ ]?name|"
    rf"subject[-_ ]?name|participant[-_ ]?name){_TOK_R}",
    re.I,
)
_NONSELF_SELF_TOKEN_RE = re.compile(rf"{_TOK_L}non[-_ ]?self{_TOK_R}", re.I)

FEEDBACK_VALUE_COLUMNS = (
    "prediction_error",
    "pe",
    "reward",
    "outcome",
    "accuracy",
    "correct",
    "value",
)
RESPONSE_TIME_COLUMN = "response_time"
SRPI_TEXT_COLUMNS = ("trial_type", "condition", "stimulus", "stim_file", "value")

# Event-count contract shared with the RAM estimator
# (``mpc_metrics._RAM_MIN_GOAL_PAIRS`` / ``_RAM_MIN_FEEDBACK_EVENTS``).
RAM_MIN_GOAL_RESPONSE_PAIRS = 6
RAM_MIN_FEEDBACK_EVENTS = 3


def empty_event_bundle() -> Dict[str, object]:
    return {
        "onsets": [],
        "goal_onsets": [],
        "feedback_onsets": [],
        "feedback_values": None,
        "self_onsets": [],
        "nonself_onsets": [],
    }


def extract_run_id_from_name(fname: str) -> Optional[str]:
    m = re.search(r"_run-0*([0-9]+)", str(fname))
    if not m:
        return None
    return str(int(m.group(1)))


def _first_match(base_dir: Path, patterns: List[str]) -> Optional[Path]:
    if not base_dir.exists():
        return None
    for pat in patterns:
        hits = sorted(base_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _task_alias_labels(subject, state, condition, dataset_id=None) -> List[str]:
    """
    Raw BIDS task labels that the explicit alias table maps to
    ``<condition><state>`` for this subject (e.g. ds003171 sub-10JR: ``audio``
    for ``audioawake``). Without ``dataset_id`` every dataset's table is
    consulted for this subject label; nothing is inferred.
    """
    datasets = [dataset_id] if dataset_id else list(DATASET_TASK_ALIASES)
    labels: List[str] = []
    for ds in datasets:
        for label in task_labels_for_state(ds, subject, state, condition)[1:]:
            if label not in labels:
                labels.append(label)
    return labels


def resolve_events_file(
    bids_root, subject, session, condition="audio", dataset_id=None
) -> Optional[Path]:
    """
    Resolve the events.tsv of one subject/session (fMRI ``func`` first, then EEG).

    Patterns end in ``_*events.tsv`` after the full task (and acquisition)
    label, so ``task-sed`` never matches ``task-sed2`` and ``task-audio`` never
    matches ``task-audioawake``. There is no fallback to other sessions' files.
    BIDS session folders (``sub-X/ses-<session>/func``) are searched too.
    Explicit per-subject task aliases (``dataset_catalog.DATASET_TASK_ALIASES``,
    scoped to ``dataset_id`` when given) are tried right after the canonical
    ``<condition><session>`` label.
    """
    subject = str(subject)
    subj_root = Path(bids_root) / f"sub-{subject}"
    session_key = str(session).strip().lower()
    condition_key = str(condition or "").strip().lower()

    def _alias_patterns(cond):
        return [
            f"sub-{subject}_task-{label}_*events.tsv"
            for label in _task_alias_labels(subject, session_key, cond, dataset_id)
        ]

    fmri_patterns = []
    if condition_key and condition_key != "audio":
        fmri_patterns.extend(
            [
                f"sub-{subject}_task-{condition_key}{session_key}_*events.tsv",
                f"sub-{subject}_task-{condition_key}_ses-{session_key}_*events.tsv",
            ]
        )
        fmri_patterns.extend(_alias_patterns(condition_key))
    fmri_patterns.append(f"sub-{subject}_task-audio{session_key}_*events.tsv")
    fmri_patterns.extend(_alias_patterns("audio"))
    if session_key not in {"awake", "deep"}:
        fmri_patterns.append(f"sub-{subject}_task-{session_key}_*events.tsv")

    ses_label = session_key[4:] if session_key.startswith("ses-") else session_key
    ses_patterns = []
    if condition_key:
        ses_patterns.append(
            f"sub-{subject}_ses-{ses_label}_task-{condition_key}_*events.tsv"
        )

    if session_key == "deep":
        # Mirrors the EEG preprocessing session rule: sed2/rest, else sed/rest.
        eeg_patterns = [
            f"sub-{subject}_task-sed2_acq-rest_*events.tsv",
            f"sub-{subject}_task-sed_acq-rest_*events.tsv",
        ]
    elif session_key == "awake":
        eeg_patterns = [
            f"sub-{subject}_task-awake_acq-EC_*events.tsv",
            f"sub-{subject}_task-awake_acq-EO_*events.tsv",
            f"sub-{subject}_task-awake_*events.tsv",
        ]
    else:
        eeg_patterns = [f"sub-{subject}_task-{session_key}_*events.tsv"]

    for base_dir, patterns in (
        (subj_root / "func", fmri_patterns),
        (subj_root / f"ses-{ses_label}" / "func", ses_patterns),
        (subj_root / "eeg", eeg_patterns),
    ):
        hit = _first_match(base_dir, patterns)
        if hit is not None:
            return hit
    return None


def read_events_table(events_file) -> Optional[pd.DataFrame]:
    if events_file is None:
        return None
    events_file = Path(events_file)
    if not events_file.exists():
        return None
    df = pd.read_csv(events_file, sep="\t")
    if len(df.columns) > 0:
        # Normalize column labels (handles UTF-8 BOM in some EEG exports).
        df = df.rename(columns={c: str(c).strip().lstrip("﻿") for c in df.columns})
    return df


def _text_series(df: pd.DataFrame, cols_l: Dict[str, str], keys) -> pd.Series:
    txt = pd.Series([""] * len(df), index=df.index, dtype=object)
    for k in keys:
        if k in cols_l:
            txt = txt.str.cat(df[cols_l[k]].astype(str), sep=" ", na_rep="")
    # Case is kept: classify_self_nonself needs camelCase token boundaries.
    return txt


def classify_self_nonself(labels: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Classify event labels into self and non-self masks.

    A label is non-self when it contains a non-self token ("non-self",
    "other", "stranger", ...); it is self when it contains a self token and no
    non-self token. Labels carrying both kinds of token (e.g. "self_other")
    are ambiguous and assigned to neither class. camelCase boundaries split
    tokens, so "SelfName"/"OtherName" are classified like "self_name"/
    "other_name".
    """
    txt = labels.astype(str).str.replace(_CAMEL_BOUNDARY_RE, " ", regex=True)
    txt = txt.str.lower()
    has_nonself = txt.str.contains(NONSELF_RE, regex=True, na=False)
    # "non-self" must not also count as a self token.
    txt_wo_nonself = txt.str.replace(_NONSELF_SELF_TOKEN_RE, " ", regex=True)
    has_self = txt_wo_nonself.str.contains(SELF_RE, regex=True, na=False)
    self_mask = has_self & (~has_nonself)
    nonself_mask = has_nonself & (~has_self)
    return self_mask, nonself_mask


def events_table_to_bundle(
    df: Optional[pd.DataFrame],
    allow_implicit_stimuli: bool = False,
    allow_response_time_feedback: bool = False,
) -> Dict[str, object]:
    """
    Convert a BIDS events table into the RAM/SRPI event bundle.

    Returns a dict with ``onsets``, ``goal_onsets``, ``feedback_onsets``,
    ``feedback_values`` (aligned 1:1 with ``feedback_onsets`` or ``None``),
    ``self_onsets`` and ``nonself_onsets``.
    """
    bundle = empty_event_bundle()
    if df is None or df.empty:
        return bundle
    cols_l = {str(c).lower(): c for c in df.columns}
    onset_col = cols_l.get("onset")
    if onset_col is None:
        return bundle

    onset = pd.to_numeric(df[onset_col], errors="coerce")
    valid = onset.notna() & np.isfinite(onset)
    if "trial_type" in cols_l:
        trial = df[cols_l["trial_type"]].astype(str)
    else:
        trial = pd.Series([""] * len(df), index=df.index, dtype=object)

    stim_mask = trial.str.contains(STIM_RE, regex=True, na=False)
    goal_mask = trial.str.contains(GOAL_RE, regex=True, na=False)
    fb_mask = trial.str.contains(FEEDBACK_RE, regex=True, na=False)

    stim_onsets = onset[valid & stim_mask].astype(float).tolist()
    if not stim_onsets and allow_implicit_stimuli:
        stim_onsets = onset[valid].astype(float).tolist()
    bundle["onsets"] = stim_onsets
    bundle["goal_onsets"] = onset[valid & goal_mask].astype(float).tolist()

    fb_rows = valid & fb_mask
    feedback_onsets = onset[fb_rows].astype(float).tolist()
    feedback_values = None
    candidates = list(FEEDBACK_VALUE_COLUMNS)
    if allow_response_time_feedback:
        candidates.append(RESPONSE_TIME_COLUMN)
    if bool(fb_rows.any()):
        for cand in candidates:
            col = cols_l.get(cand)
            if col is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            has_val = fb_rows & vals.notna() & np.isfinite(vals)
            v = vals[has_val].astype(float)
            if v.shape[0] >= 2 and float(v.std(ddof=0)) > 0:
                # Keep only feedback events that carry a value, so onsets and
                # values stay pairwise aligned.
                feedback_onsets = onset[has_val].astype(float).tolist()
                feedback_values = v.tolist()
                break
    bundle["feedback_onsets"] = feedback_onsets
    bundle["feedback_values"] = feedback_values

    txt = _text_series(df, cols_l, SRPI_TEXT_COLUMNS)
    self_mask, nonself_mask = classify_self_nonself(txt)
    bundle["self_onsets"] = onset[valid & self_mask].astype(float).tolist()
    bundle["nonself_onsets"] = onset[valid & nonself_mask].astype(float).tolist()
    return bundle
