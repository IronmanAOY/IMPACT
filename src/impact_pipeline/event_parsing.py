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

Typed-event columns (paper-1 construct revisions; all optional):
  * ``impact_channel`` (behavioural_feedback | covert_neural | perturbational
    | endogenous): the RAM evidence channel of a row. The bundle then carries
    one RAM sub-bundle per channel in ``channel_bundles`` and the labels in
    ``impact_channels``; rows without a label belong to no channel.
  * ``choice`` and ``reward``: the logged choice/outcome of each trial for the
    prediction-error form of RAM-U. A row carrying both is one trial (at its
    onset); otherwise a ``reward`` row is paired with the most recent
    preceding ``choice`` row (trial onset = reward onset). Bundle keys:
    ``choice_onsets``, ``choices``, ``rewards``.
  * SRPI-agency rows have ``trial_type`` ``self_caused`` or ``other_caused``
    (case, '-' and spaces are normalised to '_'), a ``phase_bin`` label, and
    for ``other_caused`` rows ``yoked_to``: the self-caused event this
    replay copies, referenced by its ``event_id`` when the table has an
    ``event_id`` column, otherwise by its onset in seconds (matched within
    ``AGENCY_ONSET_TOLERANCE_SEC``). A stimulus identity column
    (``stim_id``, ``stimulus`` or ``stim_file``, first present) is checked
    when available. Bundle key: ``agency_events`` (column lists) plus
    ``self_caused_onsets``/``other_caused_onsets``; see
    :func:`validate_srpi_agency_contract`.
"""
import re
from collections.abc import Mapping
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

# Typed RAM channels (mirrors ``mpc_metrics.RAM_IMPACT_CHANNELS`` /
# ``RAM_IMPLEMENTED_CHANNELS``; a test keeps them identical).
IMPACT_CHANNEL_COLUMN = "impact_channel"
IMPACT_CHANNELS = (
    "behavioural_feedback",
    "covert_neural",
    "perturbational",
    "endogenous",
)
IMPLEMENTED_RAM_CHANNELS = ("behavioural_feedback", "covert_neural")
CHOICE_COLUMN = "choice"
REWARD_COLUMN = "reward"

# SRPI-agency contract.
AGENCY_SELF_LABEL = "self_caused"
AGENCY_OTHER_LABEL = "other_caused"
AGENCY_STIMULUS_COLUMNS = ("stim_id", "stimulus", "stim_file")
AGENCY_EVENT_ID_COLUMN = "event_id"
AGENCY_ONSET_TOLERANCE_SEC = 1e-3
# SRPI needs >= 3 events per class (cross-validated separability trains on
# >= 2 events per class; mirrors mpc_metrics.compute_SRPI).
SRPI_MIN_EVENTS_PER_CLASS = 3


def empty_event_bundle() -> Dict[str, object]:
    return {
        "onsets": [],
        "goal_onsets": [],
        "feedback_onsets": [],
        "feedback_values": None,
        "self_onsets": [],
        "nonself_onsets": [],
        "choice_onsets": [],
        "choices": [],
        "rewards": [],
        "impact_channels": [],
        "channel_bundles": {},
        "agency_events": None,
        "self_caused_onsets": [],
        "other_caused_onsets": [],
    }


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return True
    return str(v).strip().lower() in {"", "nan", "n/a", "none"}


def _norm_label(v) -> Optional[str]:
    """Canonical string of a label value (None when missing; 2.0 -> '2')."""
    if _is_missing(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else repr(float(v))
    s = str(v).strip()
    try:
        f = float(s)
    except ValueError:
        return s
    if np.isfinite(f) and f.is_integer():
        return str(int(f))
    return s


def _norm_trial_type(v) -> str:
    return re.sub(r"[\s\-]+", "_", str(v).strip().lower())


def _clean_value(v):
    """JSON-friendly cell value (missing -> None)."""
    if _is_missing(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


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
    ``self_onsets`` and ``nonself_onsets``, plus the typed-event keys (see the
    module docstring): ``choice_onsets``/``choices``/``rewards``,
    ``impact_channels`` and ``channel_bundles`` (one RAM sub-bundle per
    ``impact_channel`` label), ``agency_events`` (``None`` without
    self_caused/other_caused rows), ``self_caused_onsets`` and
    ``other_caused_onsets``.
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

    bundle.update(
        _ram_fields(
            df,
            cols_l,
            onset,
            valid,
            trial,
            allow_implicit_stimuli,
            allow_response_time_feedback,
        )
    )

    txt = _text_series(df, cols_l, SRPI_TEXT_COLUMNS)
    self_mask, nonself_mask = classify_self_nonself(txt)
    bundle["self_onsets"] = onset[valid & self_mask].astype(float).tolist()
    bundle["nonself_onsets"] = onset[valid & nonself_mask].astype(float).tolist()

    ch_col = cols_l.get(IMPACT_CHANNEL_COLUMN)
    if ch_col is not None:
        labels = df[ch_col].map(
            lambda v: None if _is_missing(v) else str(v).strip().lower()
        )
        channels = sorted({lab for lab in labels[valid].tolist() if lab is not None})
        bundle["impact_channels"] = channels
        bundle["channel_bundles"] = {
            ch: _ram_fields(
                df,
                cols_l,
                onset,
                valid & (labels == ch),
                trial,
                allow_implicit_stimuli,
                allow_response_time_feedback,
            )
            for ch in channels
        }

    tt_norm = trial.map(_norm_trial_type)
    agency_rows = valid & tt_norm.isin([AGENCY_SELF_LABEL, AGENCY_OTHER_LABEL])
    if bool(agency_rows.any()):
        sub = df[agency_rows]
        events = {
            "onset": onset[agency_rows].astype(float).tolist(),
            "trial_type": tt_norm[agency_rows].tolist(),
        }
        extra_cols = ["yoked_to", "phase_bin", AGENCY_EVENT_ID_COLUMN]
        extra_cols += list(AGENCY_STIMULUS_COLUMNS)
        for name in extra_cols:
            if name in cols_l:
                events[name] = [_clean_value(v) for v in sub[cols_l[name]].tolist()]
        bundle["agency_events"] = events
        bundle["self_caused_onsets"] = onset[
            agency_rows & (tt_norm == AGENCY_SELF_LABEL)
        ].astype(float).tolist()
        bundle["other_caused_onsets"] = onset[
            agency_rows & (tt_norm == AGENCY_OTHER_LABEL)
        ].astype(float).tolist()
    return bundle


def _ram_fields(
    df: pd.DataFrame,
    cols_l: Dict[str, str],
    onset: pd.Series,
    rows: pd.Series,
    trial: pd.Series,
    allow_implicit_stimuli: bool,
    allow_response_time_feedback: bool,
) -> Dict[str, object]:
    """RAM part of the bundle from the valid rows selected by ``rows``."""
    stim_mask = trial.str.contains(STIM_RE, regex=True, na=False)
    goal_mask = trial.str.contains(GOAL_RE, regex=True, na=False)
    fb_mask = trial.str.contains(FEEDBACK_RE, regex=True, na=False)

    out = {}
    stim_onsets = onset[rows & stim_mask].astype(float).tolist()
    if not stim_onsets and allow_implicit_stimuli:
        stim_onsets = onset[rows].astype(float).tolist()
    out["onsets"] = stim_onsets
    out["goal_onsets"] = onset[rows & goal_mask].astype(float).tolist()

    fb_rows = rows & fb_mask
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
    out["feedback_onsets"] = feedback_onsets
    out["feedback_values"] = feedback_values
    out.update(_choice_reward_log(df, cols_l, onset, rows))
    return out


def _choice_reward_log(df, cols_l, onset, rows) -> Dict[str, list]:
    """
    Logged trials ``(choice_onsets, choices, rewards)`` in onset order.

    A row with both a choice and a finite reward is one trial. A reward-only
    row is paired with the most recent preceding unpaired choice-only row
    (trial onset = the reward row's onset); unpaired rows are dropped.
    """
    out = {"choice_onsets": [], "choices": [], "rewards": []}
    c_col = cols_l.get(CHOICE_COLUMN)
    r_col = cols_l.get(REWARD_COLUMN)
    if c_col is None or r_col is None:
        return out
    reward = pd.to_numeric(df[r_col], errors="coerce")
    position = {label: k for k, label in enumerate(df.index)}
    order = sorted(df.index[rows], key=lambda i: (float(onset[i]), position[i]))
    pending = None
    for i in order:
        choice = _norm_label(df.at[i, c_col])
        rew = reward[i]
        has_r = bool(pd.notna(rew) and np.isfinite(rew))
        if choice is not None and has_r:
            trial = (float(onset[i]), choice, float(rew))
            pending = None
        elif choice is not None:
            pending = choice
            continue
        elif has_r and pending is not None:
            trial = (float(onset[i]), pending, float(rew))
            pending = None
        else:
            continue
        out["choice_onsets"].append(trial[0])
        out["choices"].append(trial[1])
        out["rewards"].append(trial[2])
    return out


def _agency_columns(agency_events) -> Optional[Dict[str, list]]:
    """Normalise agency events (DataFrame, column mapping or row records)."""
    if agency_events is None:
        return None
    if hasattr(agency_events, "columns") and hasattr(agency_events, "to_dict"):
        cols = {
            str(c).strip().lower(): list(agency_events[c])
            for c in agency_events.columns
        }
    elif isinstance(agency_events, Mapping):
        cols = {}
        for k, v in agency_events.items():
            if v is None:
                continue
            if isinstance(v, (str, bytes)) or not hasattr(v, "__len__"):
                raise ValueError(f"agency event column {k!r} must be a sequence")
            cols[str(k).strip().lower()] = list(v)
    elif isinstance(agency_events, (list, tuple)):
        if not all(isinstance(r, Mapping) for r in agency_events):
            raise ValueError("agency events given as a list must be row mappings")
        keys = []
        for r in agency_events:
            for k in r:
                kk = str(k).strip().lower()
                if kk not in keys:
                    keys.append(kk)
        cols = {k: [] for k in keys}
        for r in agency_events:
            rl = {str(k).strip().lower(): v for k, v in r.items()}
            for k in keys:
                cols[k].append(rl.get(k))
    else:
        raise ValueError(
            "agency_events must be a DataFrame, a mapping of columns or a list of rows"
        )
    lengths = {len(v) for v in cols.values()}
    if len(lengths) > 1:
        raise ValueError("agency event columns must have equal lengths")
    return cols


def validate_srpi_agency_contract(agency_events, tr: Optional[float] = None) -> Dict:
    """
    Check the SRPI-agency events contract (self-caused vs yoked replays).

    Rows with ``trial_type`` ``self_caused``/``other_caused`` are used. Every
    ``other_caused`` row must be yoked (``yoked_to``) to exactly one
    ``self_caused`` row, by ``event_id`` when that column exists, otherwise by
    onset (within ``AGENCY_ONSET_TOLERANCE_SEC``). Each yoked pair must share
    its ``phase_bin`` (phase-matched) and, when a stimulus identity column is
    present, its stimulus (stimulus-identical). Self-caused events without a
    replay are not part of the contrast (counted in ``n_self_unyoked``).

    Returns a report with ``valid``, ``violations`` (stable reason codes:
    ``missing_agency_events``, ``missing_column:<name>``,
    ``non_finite_onset``, ``missing_self_caused_events``,
    ``missing_other_caused_events``, ``missing_phase_bin``,
    ``missing_yoked_to``, ``unresolved_yoking``, ``ambiguous_yoking``,
    ``duplicate_event_id``, ``phase_bin_mismatch``,
    ``missing_stimulus_identity``, ``stimulus_mismatch``), counts, the
    matched ``pairs`` (row indices into ``events``) and the normalised
    ``events`` (onset, trial_type, phase_bin). ``tr`` is accepted for
    interface symmetry and recorded; matching does not depend on it.
    """
    report = {
        "valid": False,
        "violations": [],
        "n_self_caused": 0,
        "n_other_caused": 0,
        "n_pairs": 0,
        "n_self_unyoked": 0,
        "yoking_key": None,
        "stimulus_identity": None,
        "stimulus_column": None,
        "pairs": [],
        "events": {"onset": [], "trial_type": [], "phase_bin": []},
        "tr": None if tr is None else float(tr),
    }
    violations: List[str] = []

    def _done():
        seen = []
        for v in violations:
            if v not in seen:
                seen.append(v)
        report["violations"] = seen
        report["valid"] = (not seen) and report["n_pairs"] > 0
        return report

    cols = _agency_columns(agency_events)
    if cols is None or not cols:
        violations.append("missing_agency_events")
        return _done()
    for name in ("onset", "trial_type", "phase_bin", "yoked_to"):
        if name not in cols:
            violations.append(f"missing_column:{name}")
    if "onset" not in cols or "trial_type" not in cols:
        return _done()
    onset = np.asarray(
        [np.nan if _is_missing(v) else float(v) for v in cols["onset"]], dtype=float
    )
    ttype = [_norm_trial_type(v) for v in cols["trial_type"]]
    keep = [
        i for i, t in enumerate(ttype) if t in (AGENCY_SELF_LABEL, AGENCY_OTHER_LABEL)
    ]
    if any(not np.isfinite(onset[i]) for i in keep):
        violations.append("non_finite_onset")
    keep = [i for i in keep if np.isfinite(onset[i])]
    phase = [
        _norm_label(cols["phase_bin"][i]) if "phase_bin" in cols else None for i in keep
    ]
    events = {
        "onset": [float(onset[i]) for i in keep],
        "trial_type": [ttype[i] for i in keep],
        "phase_bin": phase,
    }
    report["events"] = events
    self_rows = [
        j for j, t in enumerate(events["trial_type"]) if t == AGENCY_SELF_LABEL
    ]
    other_rows = [
        j for j, t in enumerate(events["trial_type"]) if t == AGENCY_OTHER_LABEL
    ]
    report["n_self_caused"] = len(self_rows)
    report["n_other_caused"] = len(other_rows)
    if not self_rows:
        violations.append("missing_self_caused_events")
    if not other_rows:
        violations.append("missing_other_caused_events")
    if "phase_bin" in cols and any(p is None for p in phase):
        violations.append("missing_phase_bin")
    if not self_rows or not other_rows or "yoked_to" not in cols:
        return _done()

    yoked = [cols["yoked_to"][i] for i in keep]
    use_id = AGENCY_EVENT_ID_COLUMN in cols
    report["yoking_key"] = AGENCY_EVENT_ID_COLUMN if use_id else "onset"
    if use_id:
        ids = [_norm_label(cols[AGENCY_EVENT_ID_COLUMN][i]) for i in keep]
        self_ids = [ids[j] for j in self_rows if ids[j] is not None]
        if len(set(self_ids)) != len(self_ids):
            violations.append("duplicate_event_id")
    pairs = []
    for j in other_rows:
        target = yoked[j]
        if _is_missing(target):
            violations.append("missing_yoked_to")
            continue
        if use_id:
            key = _norm_label(target)
            hits = [s for s in self_rows if ids[s] is not None and ids[s] == key]
        else:
            try:
                t = float(target)
            except (TypeError, ValueError):
                violations.append("unresolved_yoking")
                continue
            hits = [
                s
                for s in self_rows
                if abs(events["onset"][s] - t) <= AGENCY_ONSET_TOLERANCE_SEC
            ]
        if not hits:
            violations.append("unresolved_yoking")
            continue
        if len(hits) > 1:
            violations.append("ambiguous_yoking")
            continue
        pairs.append((hits[0], j))

    stim_col = next((c for c in AGENCY_STIMULUS_COLUMNS if c in cols), None)
    report["stimulus_column"] = stim_col
    stim = [_norm_label(cols[stim_col][i]) for i in keep] if stim_col else None
    for s, o in pairs:
        if phase[s] is not None and phase[o] is not None and phase[s] != phase[o]:
            violations.append("phase_bin_mismatch")
        if stim is not None:
            if stim[s] is None or stim[o] is None:
                violations.append("missing_stimulus_identity")
            elif stim[s] != stim[o]:
                violations.append("stimulus_mismatch")
    report["stimulus_identity"] = "verified" if stim_col else "by_yoking"
    report["pairs"] = [(int(s), int(o)) for s, o in pairs]
    report["n_pairs"] = len(pairs)
    report["n_self_unyoked"] = len(set(self_rows) - {s for s, _ in pairs})
    return _done()
