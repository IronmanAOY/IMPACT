import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from impact_pipeline.event_parsing import (
    IMPACT_CHANNELS,
    IMPLEMENTED_RAM_CHANNELS,
    RAM_MIN_FEEDBACK_EVENTS,
    RAM_MIN_GOAL_RESPONSE_PAIRS,
    SRPI_MIN_EVENTS_PER_CLASS,
    events_table_to_bundle,
    extract_run_id_from_name,
    read_events_table,
    resolve_events_file,
    validate_srpi_agency_contract,
)


# Event parsing is shared with the compute path (run_synergy_ci.load_onsets)
# so that readiness and metric computation classify events identically.

# Reason of NAS_reason when NAS mode='capacity' has no declared hub; the same
# detail code as the evidence layer (UNDEFINED:NAS:NO_DECLARED_WORKSPACE).
NAS_NO_DECLARED_WORKSPACE = "NO_DECLARED_WORKSPACE"
# Reason of NAS_reason when the declared hub does not fit the recording (an
# index outside its node count, or a mask of another length): compute_NAS
# refuses such a hub (ValueError), so the run cannot define NAS.
NAS_INVALID_WORKSPACE = "INVALID_WORKSPACE"
# The empirical default protocol of run_pipeline.py in a source checkout (the
# command-line default here as well).
DEFAULT_EMPIRICAL_PROTOCOL = (
    Path(__file__).resolve().parents[2] / "protocols" / "mpc_default_v1.json"
)


def _events_to_ram_bundle(df: Optional[pd.DataFrame]) -> Dict[str, object]:
    bundle = events_table_to_bundle(df)
    return {
        "onsets": bundle["onsets"],
        "goal_onsets": bundle["goal_onsets"],
        "feedback_onsets": bundle["feedback_onsets"],
        "feedback_values": (
            [] if bundle["feedback_values"] is None else list(bundle["feedback_values"])
        ),
    }


def _events_to_srpi_onsets(df: Optional[pd.DataFrame]) -> Tuple[List[float], List[float]]:
    bundle = events_table_to_bundle(df)
    return list(bundle["self_onsets"]), list(bundle["nonself_onsets"])


def _pick_session_ts_paths(
    prep_root: Path,
    subject: str,
    session: str,
    condition: str,
    atlas: str,
    run_id: Optional[str],
) -> List[Path]:
    session_dir = prep_root / subject / session / condition
    cands = sorted(session_dir.glob(f"{subject}_run-*_{atlas}_ts.npy"))
    if not cands:
        return []
    if run_id is None:
        return cands

    picked = []
    try:
        run_clean = str(int(run_id))
    except Exception:
        run_clean = str(run_id)
    run_padded = str(run_id)
    for tok in (run_clean, run_padded):
        hits = sorted(session_dir.glob(f"{subject}_run-{tok}_{atlas}_ts.npy"))
        if hits:
            picked.extend(hits)
    if picked:
        return [picked[0]]

    tags = (f"run-{run_clean}_", f"run-{run_padded}_")
    filt = [p for p in cands if any(t in p.name for t in tags)]
    if filt:
        return [filt[0]]
    return cands


def _pdi_state_rest_runs(prep_root: Path, subject: str, session: str, atlas: str) -> List[Path]:
    return sorted(
        (prep_root / subject / session / "rest").glob(f"{subject}_run-*_{atlas}_ts.npy")
    )


def _pdi_deep_rest_runs(prep_root: Path, subject: str, atlas: str) -> List[Path]:
    return sorted(
        (prep_root / subject / "deep" / "rest").glob(f"{subject}_run-*_{atlas}_ts.npy")
    )


def _assess_ram(
    bundle: Dict[str, object],
    require_explicit_feedback: bool,
    require_explicit_goals: bool = True,
) -> Tuple[bool, str, int, int, int]:
    """
    Event-level RAM readiness, mirroring compute_RAM's definedness contract.

    Counts are raw (before windowing), so a ready run can still be undefined
    in compute when events fall too close to the run edges.
    """
    stim = np.asarray(bundle.get("onsets", []), dtype=float)
    stim = stim[np.isfinite(stim)]
    n_stim = int(stim.shape[0])
    if n_stim == 0:
        return False, "missing_stimulus_events", 0, 0, 0

    goal = np.asarray(bundle.get("goal_onsets", []), dtype=float)
    goal = goal[np.isfinite(goal)]
    fb = np.asarray(bundle.get("feedback_onsets", []), dtype=float)
    fb = fb[np.isfinite(fb)]
    n_fb = int(fb.shape[0])
    fv_raw = bundle.get("feedback_values")
    fvals = np.asarray([] if fv_raw is None else fv_raw, dtype=float).reshape(-1)
    fvals = fvals[np.isfinite(fvals)]
    n_fvals = int(fvals.shape[0])

    if require_explicit_feedback:
        if n_fb == 0:
            return False, "missing_feedback_events", n_stim, n_fb, n_fvals
        if n_fvals < int(RAM_MIN_FEEDBACK_EVENTS):
            return False, "missing_or_nonvarying_feedback_values", n_stim, n_fb, n_fvals
    if require_explicit_goals and int(goal.shape[0]) == 0:
        return False, "missing_goal_events", n_stim, n_fb, n_fvals
    if n_stim < int(RAM_MIN_GOAL_RESPONSE_PAIRS):
        return False, "insufficient_stimulus_events", n_stim, n_fb, n_fvals

    return True, "ok", n_stim, n_fb, n_fvals


def _nas_setup(protocol, nas_params) -> Dict[str, object]:
    """
    NAS mode and hub under the run's protocol and ``nas_params``, decided as
    ``compute_synergy_ci`` decides them: NAS ``mode='capacity'`` without a
    declared hub (``workspace_nodes``) is UNDEFINED in every run
    (``NO_DECLARED_WORKSPACE``), so no run is NAS-ready. ``nas_hub`` is the
    declared hub under NAS capacity (else None), checked per recording.
    """
    if protocol is None and not nas_params:
        return {"protocol_hash": None, "nas_mode": "legacy",
                "nas_hub_declared": None, "nas_hub_missing": False,
                "nas_hub": None}
    from impact_pipeline import evidence
    from impact_pipeline.synergy_ci import nas_hub_missing

    proto = evidence.resolve_protocol(protocol)
    opts = {**dict(nas_params or {}),
            **(proto.estimator_options("NAS") if proto is not None else {})}
    mode = str(opts.get("mode") or "legacy")
    missing = bool(nas_hub_missing(proto, nas_params))
    hub = opts.get("workspace_nodes") if mode == "capacity" else None
    return {
        "protocol_hash": None if proto is None else proto.hash,
        "nas_mode": mode,
        "nas_hub_declared": (
            None if mode != "capacity" else opts.get("workspace_nodes") is not None
        ),
        "nas_hub_missing": missing,
        "nas_hub": hub,
    }


def _nas_hub_fits(hub, n_regions: int) -> bool:
    """
    Whether the declared hub is a valid node set of a recording with
    ``n_regions`` nodes, by the rule ``compute_NAS`` applies to
    ``workspace_nodes`` (distinct integer indices in range, or a boolean
    mask of that length).
    """
    from impact_pipeline.mpc_metrics import _resolve_node_indices

    try:
        _resolve_node_indices(hub, int(n_regions), "workspace_nodes")
    except (TypeError, ValueError):
        return False
    return True


def _assess_nas(
    n_regions: int, n_time: int, hub_missing: bool = False, hub=None
) -> Tuple[bool, str]:
    if hub_missing:
        return False, NAS_NO_DECLARED_WORKSPACE
    if n_regions < 2:
        return False, "insufficient_regions"
    if n_time < 4:
        return False, "insufficient_timepoints"
    if hub is not None and not _nas_hub_fits(hub, n_regions):
        return False, NAS_INVALID_WORKSPACE
    return True, "ok"


def _assess_iim(
    n_regions: int,
    n_time: int,
    bins: int,
    lag_trs: int,
    max_state_space: int,
    max_nodes: Optional[int],
) -> Tuple[bool, str, int, int]:
    if n_regions < 2 or (n_time - int(lag_trs)) < 1:
        return False, "insufficient_shape", 0, 0

    eff_bins = int(bins)
    if max_nodes is None:
        n_sel = int(n_regions)
    else:
        n_sel = int(min(int(max_nodes), int(n_regions)))

    while n_sel >= 2 and (eff_bins ** n_sel) > int(max_state_space):
        if eff_bins > 2:
            eff_bins -= 1
        else:
            n_sel -= 1
    if n_sel < 2:
        return False, "state_space_too_large", int(n_sel), int(eff_bins)
    return True, "ok", int(n_sel), int(eff_bins)


def _assess_ram_channels(
    bundle: Dict[str, object],
    require_explicit_feedback: bool,
    require_explicit_goals: bool = True,
) -> Dict[str, Tuple[bool, str]]:
    """
    Per-``impact_channel`` RAM readiness: implemented channels use the same
    event contract as ``_assess_ram`` on the channel's events; declared but
    unimplemented channels report ``NOT_IMPLEMENTED``; labels outside
    ``IMPACT_CHANNELS`` report ``unknown_impact_channel``.
    """
    out: Dict[str, Tuple[bool, str]] = {}
    sub_bundles = bundle.get("channel_bundles") or {}
    for ch in bundle.get("impact_channels") or []:
        if ch not in IMPACT_CHANNELS:
            out[ch] = (False, "unknown_impact_channel")
        elif ch not in IMPLEMENTED_RAM_CHANNELS:
            out[ch] = (False, "NOT_IMPLEMENTED")
        else:
            ok, reason, *_ = _assess_ram(
                sub_bundles.get(ch) or {},
                require_explicit_feedback=require_explicit_feedback,
                require_explicit_goals=require_explicit_goals,
            )
            out[ch] = (bool(ok), reason)
    return out


def _assess_srpi_agency(
    bundle: Dict[str, object],
    min_events_per_class: int,
) -> Tuple[bool, str, int, int, int]:
    """
    SRPI-agency readiness from the events contract (raw counts, before
    windowing): ``(ready, reason, n_self_caused, n_other_caused, n_pairs)``.
    """
    events = bundle.get("agency_events")
    if events is None:
        return False, "missing_agency_events", 0, 0, 0
    report = validate_srpi_agency_contract(events)
    n_self = int(report["n_self_caused"])
    n_other = int(report["n_other_caused"])
    n_pairs = int(report["n_pairs"])
    if not report["valid"]:
        code = report["violations"][0] if report["violations"] else "no_yoked_pairs"
        return False, f"agency_contract_violation:{code}", n_self, n_other, n_pairs
    m = max(SRPI_MIN_EVENTS_PER_CLASS, int(min_events_per_class))
    n_self_paired = len({s for s, _ in report["pairs"]})
    if n_self_paired < m or n_pairs < m:
        return False, "insufficient_yoked_events", n_self, n_other, n_pairs
    return True, "ok", n_self, n_other, n_pairs


def _assess_srpi(
    self_onsets: List[float],
    nonself_onsets: List[float],
    min_events_per_class: int,
) -> Tuple[bool, str, int, int]:
    n_self = int(len(self_onsets))
    n_non = int(len(nonself_onsets))
    m = max(SRPI_MIN_EVENTS_PER_CLASS, int(min_events_per_class))
    if n_self == 0 and n_non == 0:
        return False, "missing_self_and_nonself_events", n_self, n_non
    if n_self == 0:
        return False, "missing_self_events", n_self, n_non
    if n_non == 0:
        return False, "missing_nonself_events", n_self, n_non
    if n_self < m:
        return False, "insufficient_self_events", n_self, n_non
    if n_non < m:
        return False, "insufficient_nonself_events", n_self, n_non
    return True, "ok", n_self, n_non


def _load_ts_shape(ts_path: Path) -> Tuple[int, int, Optional[str]]:
    try:
        arr = np.load(ts_path)
    except Exception as exc:
        return 0, 0, f"load_error:{type(exc).__name__}"
    if arr.ndim != 2:
        return 0, 0, f"invalid_ndim:{arr.ndim}"
    # stored as time x region in preprocessing outputs
    n_time = int(arr.shape[0])
    n_regions = int(arr.shape[1])
    return n_regions, n_time, None


def check_mpc_readiness(
    prep_root: str,
    bids_root: Optional[str],
    atlas: str,
    condition: str,
    sessions: Sequence[str],
    subjects: Optional[Sequence[str]] = None,
    require_explicit_feedback: bool = True,
    require_explicit_goals: bool = True,
    require_explicit_srpi: bool = True,
    srpi_min_events_per_class: int = 3,
    iim_bins: int = 3,
    iim_lag_trs: int = 1,
    iim_max_state_space: int = 1500,
    iim_max_nodes: Optional[int] = None,
    protocol=None,
    nas_params: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Per-run readiness of every MPC metric and the summary.

    ``protocol`` (JSON path, dict or ``evidence.Protocol``) and ``nas_params``
    are those of the run being planned; they decide the NAS mode. Under NAS
    ``mode='capacity'`` without a declared hub no run is NAS-ready
    (``NAS_reason`` ``NO_DECLARED_WORKSPACE``), as the evidence layer records
    NAS as UNDEFINED; a recording whose node count the declared hub does not
    fit is not NAS-ready either (``INVALID_WORKSPACE``), as ``compute_NAS``
    refuses that hub. Without either, NAS is checked in its legacy mode.
    """
    if not bool(require_explicit_srpi):
        raise ValueError("Neutral SRPI mode is disabled; explicit SRPI evidence is required.")
    if int(srpi_min_events_per_class) < SRPI_MIN_EVENTS_PER_CLASS:
        # Mirrors compute_SRPI: cross-validated separability needs >= 3 per class.
        raise ValueError(
            f"srpi_min_events_per_class must be >= {SRPI_MIN_EVENTS_PER_CLASS}"
        )
    prep = Path(prep_root)
    bids = Path(bids_root) if bids_root is not None else None
    if not prep.exists():
        raise FileNotFoundError(f"Preprocessed root not found: {prep}")
    nas_setup = _nas_setup(protocol, nas_params)

    prep_subjects = sorted(
        p.name for p in prep.iterdir() if p.is_dir() and (not p.name.startswith("."))
    )
    if subjects is None:
        use_subjects = prep_subjects
    else:
        req = [str(s).replace("sub-", "").strip() for s in subjects if str(s).strip()]
        use_subjects = [s for s in req if s in set(prep_subjects)]

    rows = []
    for subj in use_subjects:
        for ses in sessions:
            events_file = (
                resolve_events_file(bids, subj, ses, condition=condition)
                if bids is not None
                else None
            )
            run_id = (
                extract_run_id_from_name(events_file.name)
                if events_file is not None
                else None
            )
            df_events = read_events_table(events_file)
            ram_bundle = _events_to_ram_bundle(df_events)
            self_onsets, nonself_onsets = _events_to_srpi_onsets(df_events)
            full_bundle = events_table_to_bundle(df_events)
            channel_status = _assess_ram_channels(
                full_bundle,
                require_explicit_feedback=require_explicit_feedback,
                require_explicit_goals=require_explicit_goals,
            )
            agency_ok, agency_reason, n_sc, n_oc, n_pairs = _assess_srpi_agency(
                full_bundle, int(srpi_min_events_per_class)
            )

            def _typed_cols(ts_reason=None):
                # Typed-event readiness (additive columns): RAM channels and
                # SRPI-agency. Without a usable time series nothing is ready.
                return {
                    "RAM_channels": ";".join(full_bundle["impact_channels"]),
                    "RAM_channel_status": ";".join(
                        f"{ch}:{ts_reason or reason}"
                        for ch, (_ok, reason) in channel_status.items()
                    ),
                    "RAM_channels_ready": ";".join(
                        ch
                        for ch, (ok, _r) in channel_status.items()
                        if ok and ts_reason is None
                    ),
                    "n_choice_trials": int(len(full_bundle["choices"])),
                    "SRPI_agency_ready": bool(agency_ok and ts_reason is None),
                    "SRPI_agency_reason": ts_reason or agency_reason,
                    "n_self_caused": int(n_sc),
                    "n_other_caused": int(n_oc),
                    "n_yoked_pairs": int(n_pairs),
                }
            ts_paths = _pick_session_ts_paths(
                prep_root=prep,
                subject=subj,
                session=ses,
                condition=condition,
                atlas=atlas,
                run_id=run_id,
            )
            if not ts_paths:
                rows.append(
                    {
                        "subject": subj,
                        "session": ses,
                        "ts_path": None,
                        "events_file": None if events_file is None else str(events_file),
                        "run_id": run_id,
                        "ts_ready": False,
                        "ts_reason": "missing_timeseries",
                        "RAM_ready": False,
                        "RAM_reason": "missing_timeseries",
                        "PDI_ready": False,
                        "PDI_reason": "missing_timeseries",
                        "PDI_anchor_ready": False,
                        "PDI_anchor_reason": "missing_timeseries",
                        "PDI_task_ready": False,
                        "PDI_task_reason": "missing_timeseries",
                        "PDI_anchor_baseline_runs": 0,
                        "PDI_task_baseline_runs": 0,
                        "PDI_baseline_mode": "unknown",
                        "NAS_ready": False,
                        "NAS_reason": "missing_timeseries",
                        "IIM_ready": False,
                        "IIM_reason": "missing_timeseries",
                        "IIM_nodes_used": 0,
                        "IIM_bins_used": 0,
                        "SRPI_ready": False,
                        "SRPI_reason": "missing_timeseries",
                        "n_regions": 0,
                        "n_timepoints": 0,
                        "n_stim_onsets": 0,
                        "n_feedback_onsets": 0,
                        "n_feedback_values": 0,
                        "n_self_onsets": int(len(self_onsets)),
                        "n_nonself_onsets": int(len(nonself_onsets)),
                        "CI_ready": False,
                        **_typed_cols("missing_timeseries"),
                    }
                )
                continue

            for ts_path in ts_paths:
                n_regions, n_time, ts_err = _load_ts_shape(ts_path)
                ts_ready = ts_err is None
                if not ts_ready:
                    rows.append(
                        {
                            "subject": subj,
                            "session": ses,
                            "ts_path": str(ts_path),
                            "events_file": None if events_file is None else str(events_file),
                            "run_id": run_id,
                            "ts_ready": False,
                            "ts_reason": ts_err,
                            "RAM_ready": False,
                            "RAM_reason": ts_err,
                            "PDI_ready": False,
                            "PDI_reason": ts_err,
                            "PDI_anchor_ready": False,
                            "PDI_anchor_reason": ts_err,
                            "PDI_task_ready": False,
                            "PDI_task_reason": ts_err,
                            "PDI_anchor_baseline_runs": 0,
                            "PDI_task_baseline_runs": 0,
                            "PDI_baseline_mode": "unknown",
                            "NAS_ready": False,
                            "NAS_reason": ts_err,
                            "IIM_ready": False,
                            "IIM_reason": ts_err,
                            "IIM_nodes_used": 0,
                            "IIM_bins_used": 0,
                            "SRPI_ready": False,
                            "SRPI_reason": ts_err,
                            "n_regions": 0,
                            "n_timepoints": 0,
                            "n_stim_onsets": 0,
                            "n_feedback_onsets": 0,
                            "n_feedback_values": 0,
                            "n_self_onsets": int(len(self_onsets)),
                            "n_nonself_onsets": int(len(nonself_onsets)),
                            "CI_ready": False,
                            **_typed_cols(ts_err),
                        }
                    )
                    continue

                ram_ok, ram_reason, n_stim, n_fb, n_fvals = _assess_ram(
                    ram_bundle,
                    require_explicit_feedback=require_explicit_feedback,
                    require_explicit_goals=require_explicit_goals,
                )
                pdi_anchor_runs = _pdi_deep_rest_runs(prep, subj, atlas)
                pdi_task_runs = _pdi_state_rest_runs(prep, subj, ses, atlas)
                pdi_anchor_ok = bool(len(pdi_anchor_runs) > 0)
                pdi_task_ok = bool(len(pdi_task_runs) > 0)
                pdi_anchor_reason = "ok" if pdi_anchor_ok else "missing_deep_rest_baseline"
                pdi_task_reason = "ok" if pdi_task_ok else "missing_state_rest_baseline"
                pdi_ok = bool(pdi_anchor_ok and pdi_task_ok)
                if pdi_ok:
                    pdi_reason = "ok"
                elif (not pdi_anchor_ok) and (not pdi_task_ok):
                    pdi_reason = "missing_deep_and_state_rest_baselines"
                elif not pdi_anchor_ok:
                    pdi_reason = "missing_deep_rest_baseline"
                else:
                    pdi_reason = "missing_state_rest_baseline"
                pdi_mode = f"anchor_runs={len(pdi_anchor_runs)};task_runs={len(pdi_task_runs)}"
                nas_ok, nas_reason = _assess_nas(
                    n_regions, n_time,
                    hub_missing=bool(nas_setup["nas_hub_missing"]),
                    hub=nas_setup["nas_hub"],
                )
                iim_ok, iim_reason, iim_nodes, iim_bins_used = _assess_iim(
                    n_regions=n_regions,
                    n_time=n_time,
                    bins=iim_bins,
                    lag_trs=iim_lag_trs,
                    max_state_space=iim_max_state_space,
                    max_nodes=iim_max_nodes,
                )
                srpi_ok, srpi_reason, n_self, n_non = _assess_srpi(
                    self_onsets=self_onsets,
                    nonself_onsets=nonself_onsets,
                    min_events_per_class=int(srpi_min_events_per_class),
                )
                ci_ok = bool(ram_ok and pdi_ok and nas_ok and iim_ok and srpi_ok)

                rows.append(
                    {
                        "subject": subj,
                        "session": ses,
                        "ts_path": str(ts_path),
                        "events_file": None if events_file is None else str(events_file),
                        "run_id": run_id,
                        "ts_ready": True,
                        "ts_reason": "ok",
                        "RAM_ready": bool(ram_ok),
                        "RAM_reason": ram_reason,
                        "PDI_ready": bool(pdi_ok),
                        "PDI_reason": pdi_reason,
                        "PDI_anchor_ready": bool(pdi_anchor_ok),
                        "PDI_anchor_reason": pdi_anchor_reason,
                        "PDI_task_ready": bool(pdi_task_ok),
                        "PDI_task_reason": pdi_task_reason,
                        "PDI_anchor_baseline_runs": int(len(pdi_anchor_runs)),
                        "PDI_task_baseline_runs": int(len(pdi_task_runs)),
                        "PDI_baseline_mode": pdi_mode,
                        "NAS_ready": bool(nas_ok),
                        "NAS_reason": nas_reason,
                        "IIM_ready": bool(iim_ok),
                        "IIM_reason": iim_reason,
                        "IIM_nodes_used": int(iim_nodes),
                        "IIM_bins_used": int(iim_bins_used),
                        "SRPI_ready": bool(srpi_ok),
                        "SRPI_reason": srpi_reason,
                        "n_regions": int(n_regions),
                        "n_timepoints": int(n_time),
                        "n_stim_onsets": int(n_stim),
                        "n_feedback_onsets": int(n_fb),
                        "n_feedback_values": int(n_fvals),
                        "n_self_onsets": int(n_self),
                        "n_nonself_onsets": int(n_non),
                        "CI_ready": bool(ci_ok),
                        **_typed_cols(),
                    }
                )

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        summary = {
            "n_rows": 0,
            "metrics": {},
            "settings": {
                "require_explicit_feedback": bool(require_explicit_feedback),
                "require_explicit_goals": bool(require_explicit_goals),
                "require_explicit_srpi": True,
                "srpi_min_events_per_class": int(srpi_min_events_per_class),
                "pdi_baseline_policy": "strict_deep_rest_plus_state_rest",
                "nas_mode": nas_setup["nas_mode"],
                "nas_hub_declared": nas_setup["nas_hub_declared"],
                "protocol_hash": nas_setup["protocol_hash"],
            },
        }
        return df, summary

    metrics = [
        "RAM",
        "PDI",
        "PDI_anchor",
        "PDI_task",
        "NAS",
        "IIM",
        "SRPI",
        "SRPI_agency",
        "CI",
    ]
    metric_summary = {}
    for m in metrics:
        col = f"{m}_ready"
        metric_summary[m] = {
            "ready": int(df[col].sum()),
            "not_ready": int((~df[col]).sum()),
            "ready_fraction": float(df[col].mean()),
        }
    # Rows per RAM impact_channel label and rows where that channel is ready.
    channel_summary: Dict[str, Dict[str, int]] = {}
    for labels, ready in zip(df["RAM_channels"], df["RAM_channels_ready"]):
        ready_set = set(filter(None, str(ready).split(";")))
        for ch in filter(None, str(labels).split(";")):
            entry = channel_summary.setdefault(ch, {"rows": 0, "ready": 0})
            entry["rows"] += 1
            entry["ready"] += int(ch in ready_set)
    summary = {
        "n_rows": int(df.shape[0]),
        "n_subjects": int(df["subject"].nunique()),
        "sessions": sorted(df["session"].dropna().unique().tolist()),
        "metrics": metric_summary,
        "ram_channels": channel_summary,
        "settings": {
            "require_explicit_feedback": bool(require_explicit_feedback),
            "require_explicit_goals": bool(require_explicit_goals),
            "require_explicit_srpi": True,
            "srpi_min_events_per_class": int(srpi_min_events_per_class),
            "pdi_baseline_policy": "strict_deep_rest_plus_state_rest",
            "iim_bins": int(iim_bins),
            "iim_lag_trs": int(iim_lag_trs),
            "iim_max_state_space": int(iim_max_state_space),
            "iim_max_nodes": None if iim_max_nodes is None else int(iim_max_nodes),
            "nas_mode": nas_setup["nas_mode"],
            "nas_hub_declared": nas_setup["nas_hub_declared"],
            "protocol_hash": nas_setup["protocol_hash"],
        },
    }
    return df, summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check dataset readiness for all MPC metrics.")
    p.add_argument("--prep-root", required=True, help="Preprocessed root folder.")
    p.add_argument("--bids-root", required=False, default=None, help="BIDS root for events parsing.")
    p.add_argument("--atlas", default="schaefer400")
    p.add_argument("--condition", default="audio")
    p.add_argument("--sessions", nargs="+", default=["awake", "deep"])
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--out-csv", default=None, help="Optional output CSV path.")
    p.add_argument("--out-json", default=None, help="Optional summary JSON path.")
    p.add_argument("--allow-implicit-ram-feedback", action="store_true")
    p.add_argument("--allow-implicit-ram-goals", action="store_true")
    p.add_argument("--srpi-min-events-per-class", type=int, default=3)
    p.add_argument("--iim-bins", type=int, default=3)
    p.add_argument("--iim-lag-trs", type=int, default=1)
    p.add_argument("--iim-max-state-space", type=int, default=1500)
    p.add_argument("--iim-max-nodes", type=int, default=None)
    p.add_argument(
        "--protocol",
        default=None,
        help=(
            "MPC protocol of the planned run; it decides the NAS mode (NAS "
            "capacity without a declared hub is never ready: "
            "NO_DECLARED_WORKSPACE; a hub that does not fit a recording's "
            "node count: INVALID_WORKSPACE). Default: protocols/mpc_default_v1.json, "
            "the default of empirical runs; 'none' checks the flag-built "
            "legacy NAS mode (as for dummy data)."
        ),
    )
    return p


def resolve_protocol_arg(value: Optional[str]) -> Optional[str]:
    """``--protocol``: a path, ``none``/``flags`` (None) or the default."""
    if value is None:
        default = DEFAULT_EMPIRICAL_PROTOCOL
        return str(default) if default.is_file() else None
    if str(value).strip().lower() in ("none", "flags"):
        return None
    return str(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    df, summary = check_mpc_readiness(
        prep_root=args.prep_root,
        bids_root=args.bids_root,
        atlas=args.atlas,
        condition=args.condition,
        sessions=args.sessions,
        subjects=args.subjects,
        require_explicit_feedback=not bool(args.allow_implicit_ram_feedback),
        require_explicit_goals=not bool(args.allow_implicit_ram_goals),
        require_explicit_srpi=True,
        srpi_min_events_per_class=int(args.srpi_min_events_per_class),
        iim_bins=args.iim_bins,
        iim_lag_trs=args.iim_lag_trs,
        iim_max_state_space=args.iim_max_state_space,
        iim_max_nodes=args.iim_max_nodes,
        protocol=resolve_protocol_arg(args.protocol),
    )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
