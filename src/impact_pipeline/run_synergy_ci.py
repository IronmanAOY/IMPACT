#!/usr/bin/env python
import glob
import json
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from bids import BIDSLayout
from impact_pipeline.event_parsing import (
    NONSELF_RE,
    SELF_RE,
    empty_event_bundle,
    events_table_to_bundle,
    extract_run_id_from_name,
    read_events_table,
    resolve_events_file,
)
from impact_pipeline.provenance import (
    PROVENANCE_COLUMNS,
    REAL_DATA_ORIGIN,
)
from impact_pipeline.synergy_ci import RAM_PARAM_DEFAULTS, compute_synergy_ci
from pathlib import Path

log = logging.getLogger("pipeline")
# Label patterns live in impact_pipeline.event_parsing (single source shared
# with mpc_readiness); the private aliases are kept for backward compatibility.
_SELF_RE = SELF_RE
_NONSELF_RE = NONSELF_RE


def _infer_sample_interval_seconds(bids_root):
    """
    Infer sampling interval from BIDS sidecars.

    Priority:
      1) fMRI RepetitionTime from *_bold.json
      2) EEG SamplingFrequency from *_eeg.json
      3) no fallback
    """
    if bids_root is None:
        return None
    root = Path(bids_root)
    if not root.exists():
        return None

    fmri_sidecars = glob.glob(str(root / "sub-*/func/*_bold.json"))
    if fmri_sidecars:
        with open(fmri_sidecars[0], "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        tr = sidecar.get("RepetitionTime")
        if tr is not None:
            tr = float(tr)
            if np.isfinite(tr) and tr > 0:
                return tr

    eeg_sidecars = glob.glob(str(root / "sub-*/eeg/*_eeg.json"))
    for fn in eeg_sidecars:
        with open(fn, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        sfreq = sidecar.get("SamplingFrequency")
        if sfreq is None:
            continue
        sfreq = float(sfreq)
        if np.isfinite(sfreq) and sfreq > 0:
            return 1.0 / sfreq

    return None


# Modality presets for RAM hyperparameters, used by run_s_ci when no explicit
# ``ram_params`` are supplied. The fMRI preset is the canonical-HRF
# configuration of ``synergy_ci.RAM_PARAM_DEFAULTS``; the EEG preset replaces
# the HRF model (physiologically meaningless at sub-second sampling) by an
# event-locked boxcar response and a measured FIR latency.
RAM_PARAM_PRESETS = {
    "fmri": dict(RAM_PARAM_DEFAULTS),
    "eeg": {
        **RAM_PARAM_DEFAULTS,
        "response_model": "boxcar",
        "response_boxcar_width_sec": 0.30,
        "latency_method": "fir",
        "fir_window": 0.80,
        "goal_pre_window_sec": 0.20,
        "response_window_sec": 0.40,
        "goal_objective_window_sec": 0.20,
        "feedback_window_sec": 0.20,
        "quality_lag_sec": 0.0,
    },
}


def resolve_ram_params(ram_params=None, modality=None):
    """
    Return explicit RAM hyperparameters.

    ``ram_params`` wins when given. Otherwise the preset of the declared
    ``modality`` is used; RAM hyperparameters are modality-specific, so an
    unknown or missing modality raises instead of silently using fMRI
    defaults.
    """
    if ram_params is not None:
        return dict(ram_params)
    mode = str(modality or "").strip().lower()
    if mode not in RAM_PARAM_PRESETS:
        raise ValueError(
            "RAM hyperparameters are modality-specific: pass ram_params explicitly "
            f"or a modality in {sorted(RAM_PARAM_PRESETS)} (got {modality!r})."
        )
    log.info("RAM: using the '%s' hyperparameter preset (no explicit ram_params).",
             mode)
    return dict(RAM_PARAM_PRESETS[mode])


def _extract_run_id_from_name(fname: str):
    return extract_run_id_from_name(fname)


def _resolve_events_file(
    bids_root, subject, session, condition="audio", dataset_id=None
):
    """
    Resolve an events.tsv for either fMRI or EEG sessions.

    Delegates to :func:`impact_pipeline.event_parsing.resolve_events_file`
    (shared with the readiness check; session-exact, no cross-session
    fallback; explicit per-subject task aliases only).
    """
    return resolve_events_file(
        bids_root, subject, session, condition=condition, dataset_id=dataset_id
    )


def _events_to_ram_bundle(
    fn: Path,
    allow_implicit_stimuli: bool = False,
    allow_response_time_feedback: bool = False,
):
    """
    Convert BIDS events.tsv into a structured RAM/SRPI event bundle.

    See :func:`impact_pipeline.event_parsing.events_table_to_bundle`; the
    implicit-stimulus and response-time-as-feedback proxies are disabled
    unless explicitly enabled.
    """
    return events_table_to_bundle(
        read_events_table(fn),
        allow_implicit_stimuli=allow_implicit_stimuli,
        allow_response_time_feedback=allow_response_time_feedback,
    )


# For each session, read event timings and RAM/SRPI sub-components from BIDS events.tsv.
def load_onsets(
    bids_root,
    subject,
    session,
    condition="audio",
    allow_implicit_stimuli=False,
    allow_response_time_feedback=False,
    dataset_id=None,
):
    fn = _resolve_events_file(
        bids_root, subject, session, condition=condition, dataset_id=dataset_id
    )
    if fn is None:
        log.warning(
            (
                "No events.tsv found for sub-%s session=%s; returning empty event "
                "bundle (RAM/SRPI undefined)."
            ),
            subject,
            session,
        )
        return empty_event_bundle(), None

    bundle = _events_to_ram_bundle(
        fn,
        allow_implicit_stimuli=allow_implicit_stimuli,
        allow_response_time_feedback=allow_response_time_feedback,
    )
    run_id = _extract_run_id_from_name(fn.name)
    return bundle, run_id


def run_s_ci(
    prep_out,
    bids_root,
    figdir,
    atlas,
    sessions,
    thetas,
    thetas_fine,
    mpc_metrics=None,
    compute_ci=True,
    ci_reference=None,
    condition='audio',
    tr=None,
    onsets=None,
    load_onsets_fn=load_onsets,
    iim_n_parts=None,
    iim_max_timepoints=None,
    iim_max_nodes=None,
    iim_max_mechanism_size=None,
    iim_max_purview_size=None,
    iim_parallel_workers=None,
    iim_memory_target_ratio=0.90,
    iim_worker_mem_gb_estimate=3.0,
    iim_cpu_oversub_factor=3.0,
    iim_enable_parallel=True,
    iim_checkpoint_dir=None,
    iim_resume_checkpoint=True,
    iim_checkpoint_every_cuts=1,
    iim_progress_log_every_cuts=1,
    iim_use_shared_memory=True,
    iim_phase1_parallel_workers=None,
    iim_phase1_chunk_size=8,
    iim_phase1_shared_memory=True,
    pdi_params=None,
    pdi_require_explicit_params=False,
    pdi_require_strict_baseline=False,
    pdi_primary_endpoint="anchor",
    nas_params=None,
    srpi_params=None,
    srpi_require_explicit_params=True,
    subjects=None,
    iim_precomputed_by_path=None,
    dataset_id=None,
    data_origin=REAL_DATA_ORIGIN,
    dataset_role=None,
    provenance_label=None,
    modality=None,
    hardware_target="cpu",
    ram_params=None,
    event_options=None,
    null_surrogates=0,
    necessity_set=None,
    applicability_registry=None,
    null_seed=0,
    protocol=None,
    bootstrap_se=0,
    bootstrap_block_len=None,
):
    """
    Step 2: synergy S, MPC metrics and CI over subjects/sessions/runs.

    ``ram_params`` are passed to ``compute_RAM``; when omitted, the preset of
    ``modality`` is used (see :func:`resolve_ram_params`). ``event_options``
    are keyword arguments forwarded to ``load_onsets_fn`` (e.g.
    ``allow_implicit_stimuli``/``allow_response_time_feedback``, both off by
    default so RAM stays undefined without measured goal/feedback structure).
    ``null_surrogates``, ``necessity_set``, ``applicability_registry``,
    ``null_seed``, ``protocol`` (Protocol, dict or JSON path),
    ``bootstrap_se`` (block-bootstrap replicates for the sampling SE; 0 leaves
    every empirical component UNDEFINED, ``NO_SAMPLING_SE``) and
    ``bootstrap_block_len`` configure the MPC evidence layer (MPC verdict
    columns; see :func:`impact_pipeline.synergy_ci.compute_synergy_ci`).
    """
    if mpc_metrics is None and compute_ci:
        log.info("2/9 Computing Synergy & Consciousness Index (CI)")
    else:
        metric_label = "all" if mpc_metrics is None else ",".join(mpc_metrics)
        log.info("2/9 Computing Synergy and selected MPC metrics (%s)", metric_label)
    # 1) set up subjects & onsets
    discovered_subjects = []
    if bids_root is not None and Path(bids_root).exists():
        layout = BIDSLayout(bids_root, validate=False)
        discovered_subjects = sorted(layout.get(return_type='id', target='subject'))
    else:
        discovered_subjects = sorted(
            d for d in os.listdir(prep_out)
            if not d.startswith('.') and os.path.isdir(os.path.join(prep_out, d))
        )
    if subjects is not None:
        sel = {str(s).replace("sub-", "").strip() for s in subjects if str(s).strip()}
        discovered_subjects = [s for s in discovered_subjects if s in sel]

    needs_onsets = (mpc_metrics is None) or bool({"RAM", "SRPI"} & set(mpc_metrics))
    needs_ram = (mpc_metrics is None) or ("RAM" in set(mpc_metrics))
    if needs_ram:
        ram_params = resolve_ram_params(ram_params, modality=modality)
    event_kwargs = dict(event_options or {})
    if onsets is None and needs_onsets and load_onsets_fn is not None and bids_root is not None:
        def _load_one(subj, ses):
            try:
                return load_onsets_fn(
                    bids_root, subj, ses, condition=condition, **event_kwargs
                )
            except TypeError:
                if event_kwargs:
                    raise
                return load_onsets_fn(bids_root, subj, ses)

        onsets = {
            subj: {ses: _load_one(subj, ses) for ses in sessions}
            for subj in discovered_subjects
        }

    # 2) get TR/sample interval (no fallback defaults)
    real_tr = tr if tr is not None else _infer_sample_interval_seconds(bids_root)
    if real_tr is None or (not np.isfinite(real_tr)) or float(real_tr) <= 0:
        raise ValueError(
            "Could not determine a valid sample interval. "
            "Provide explicit --tr (or dataset metadata with RepetitionTime/SamplingFrequency)."
        )
    # 3) compute coarse & fine synergy+CI
    df = compute_synergy_ci(
        str(prep_out),
        atlas,
        thetas,
        sessions,
        condition=condition,
        tr=real_tr,
        stimulus_onsets=onsets,
        mpc_metrics=mpc_metrics,
        compute_ci=compute_ci,
        ci_reference=ci_reference,
        ram_params=ram_params,
        iim_n_parts=iim_n_parts,
        iim_max_timepoints=iim_max_timepoints,
        iim_max_nodes=iim_max_nodes,
        iim_max_mechanism_size=iim_max_mechanism_size,
        iim_max_purview_size=iim_max_purview_size,
        iim_parallel_workers=iim_parallel_workers,
        iim_memory_target_ratio=iim_memory_target_ratio,
        iim_worker_mem_gb_estimate=iim_worker_mem_gb_estimate,
        iim_cpu_oversub_factor=iim_cpu_oversub_factor,
        iim_enable_parallel=iim_enable_parallel,
        iim_checkpoint_dir=iim_checkpoint_dir,
        iim_resume_checkpoint=iim_resume_checkpoint,
        iim_checkpoint_every_cuts=iim_checkpoint_every_cuts,
        iim_progress_log_every_cuts=iim_progress_log_every_cuts,
        iim_use_shared_memory=iim_use_shared_memory,
        iim_phase1_parallel_workers=iim_phase1_parallel_workers,
        iim_phase1_chunk_size=iim_phase1_chunk_size,
        iim_phase1_shared_memory=iim_phase1_shared_memory,
        pdi_params=pdi_params,
        pdi_require_explicit_params=pdi_require_explicit_params,
        pdi_require_strict_baseline=pdi_require_strict_baseline,
        pdi_primary_endpoint=pdi_primary_endpoint,
        nas_params=nas_params,
        srpi_params=srpi_params,
        srpi_require_explicit_params=srpi_require_explicit_params,
        subjects=discovered_subjects,
        iim_precomputed_by_path=iim_precomputed_by_path,
        dataset_id=dataset_id,
        data_origin=data_origin,
        dataset_role=dataset_role,
        provenance_label=provenance_label,
        modality=modality,
        hardware_target=hardware_target,
        null_surrogates=null_surrogates,
        necessity_set=necessity_set,
        applicability_registry=applicability_registry,
        null_seed=null_seed,
        protocol=protocol,
        bootstrap_se=bootstrap_se,
        bootstrap_block_len=bootstrap_block_len,
    )
    df['session'] = df['session'].replace({'audioawake': 'awake', 'audiodeep': 'deep'})
    meta_cols = [
        c
        for c in (*PROVENANCE_COLUMNS, "hardware_target", "hardware_backend", "hardware_runtime")
        if c in df.columns
    ]
    agg_map = {'S': ('S', 'mean')}
    for meta_col in meta_cols:
        agg_map[meta_col] = (meta_col, 'first')
    for metric in (
        'CI', 'RAM', 'PDI', 'PDI_anchor', 'PDI_task',
        'NAS', 'IIM', 'SRPI', 'IIM_raw', 'IIM_raw_scaled'
    ):
        if metric in df.columns:
            agg_map[metric] = (metric, 'mean')
    df_mean = df.groupby(['subject', 'session']).agg(**agg_map).reset_index()
    df_S = df_mean.pivot(index='subject', columns='session', values='S')
    df_CI = (
        df_mean.pivot(index='subject', columns='session', values='CI')
        if 'CI' in df_mean.columns else None
    )
    # --- fine grid & supplemental figure ---
    df_fine = compute_synergy_ci(
        str(prep_out),
        atlas,
        thetas_fine,
        sessions,
        condition=condition,
        tr=real_tr,
        stimulus_onsets=onsets,
        compute_mpc=False,
        pdi_params=pdi_params,
        pdi_require_explicit_params=pdi_require_explicit_params,
        pdi_require_strict_baseline=pdi_require_strict_baseline,
        pdi_primary_endpoint=pdi_primary_endpoint,
        nas_params=nas_params,
        srpi_params=srpi_params,
        srpi_require_explicit_params=srpi_require_explicit_params,
        subjects=discovered_subjects,
        dataset_id=dataset_id,
        data_origin=data_origin,
        dataset_role=dataset_role,
        provenance_label=provenance_label,
        modality=modality,
        hardware_target=hardware_target,
    )
    df_fine['session'] = df_fine['session'].replace(
        {'audioawake': 'awake', 'audiodeep': 'deep'})
    means, sems = [], []
    theta_vals = []
    for theta, subdf in df_fine.groupby('theta'):
        theta_vals.append(float(theta))
        # If multiple runs per subject/session exist, average within each
        # subject/session at fixed theta before paired comparisons.
        sub_mean = (
            subdf.groupby(['subject', 'session'], as_index=False)['S']
            .mean()
        )
        piv = sub_mean.pivot(index='subject', columns='session', values='S')
        paired = piv[['awake', 'deep']].dropna() if {'awake', 'deep'}.issubset(piv.columns) else pd.DataFrame()
        if paired.empty:
            means.append(np.nan)
            sems.append(np.nan)
            continue
        diff = paired['awake'] - paired['deep']
        n = int(diff.notna().sum())
        means.append(float(diff.mean()))
        sems.append(float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan)
    theta_arr = np.asarray(theta_vals, dtype=float)
    mean_arr = np.asarray(means, dtype=float)
    sem_arr = np.asarray(sems, dtype=float)
    if mean_arr.size and np.isfinite(mean_arr).any():
        i_star = int(np.nanargmax(np.abs(mean_arr)))
        theta_star = float(theta_arr[i_star])
    else:
        theta_star = np.nan
    fig, ax = plt.subplots()
    ax.errorbar(theta_arr, mean_arr, yerr=sem_arr, marker='o')
    if np.isfinite(theta_star):
        ax.axvline(theta_star, linestyle='--')
    ax.set(xlabel='θ', ylabel='Mean S_awake–S_deep')
    fig.tight_layout()
    fig.savefig(figdir / 'supp_theta_curve.png')
    # --- stats by theta ---
    rows = []
    for theta, subdf in df.groupby('theta'):
        sub_mean = (
            subdf.groupby(['subject', 'session'], as_index=False)['S']
            .mean()
        )
        piv = sub_mean.pivot(index='subject', columns='session', values='S')
        paired = piv[['awake', 'deep']].dropna() if {'awake', 'deep'}.issubset(piv.columns) else pd.DataFrame()
        if len(paired) < 2:
            rows.append(
                {
                    'theta': theta,
                    't_S': np.nan,
                    'p_S': np.nan,
                    'd_S': np.nan,
                    'mean_diff_S': np.nan,
                }
            )
            continue
        t, p = stats.ttest_rel(paired['awake'].values, paired['deep'].values)
        diff = paired['awake'] - paired['deep']
        sd = diff.std(ddof=1)
        d = float(diff.mean() / sd) if np.isfinite(sd) and sd > 0 else np.nan
        rows.append(
            {
                'theta': theta,
                't_S': float(t),
                'p_S': float(p),
                'd_S': d,
                'mean_diff_S': float(diff.mean()),
            }
        )
    df_stats_by_theta = pd.DataFrame(rows).set_index('theta')
    return df, df_mean, df_S, df_CI, df_stats_by_theta
