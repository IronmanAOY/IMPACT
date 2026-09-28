#!/usr/bin/env python
import sys
import argparse
import logging
import random
import tempfile
import numpy as np
import subprocess, os
import pandas as pd
import json

from pathlib import Path
from bids import BIDSLayout

root = Path(__file__).resolve().parent
src_root = root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

mpl_cache_dir = Path(tempfile.gettempdir()) / "impact_mpl_cache"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

from impact_pipeline.preprocessing_eeg import run_preprocessing_eeg
from impact_pipeline.baseline_metrics import BASELINE_METRICS, compute_baseline_metrics
from impact_pipeline.analysis_bootstrap import (
    bootstrap_ci,
    definedness_summary,
    holm_adjust,
    paired_session_test,
    paired_tests_table,
    permutation_test_auc,
)
from impact_pipeline.motion_model import motion_covariate_analysis
from impact_pipeline.atlas_robustness import atlas_check
from impact_pipeline.replication import (
    _find_fmriprep_derivatives,
    run_replication,
)
from impact_pipeline.model_comparison import compare_models
from impact_pipeline.generate_word_doc import create_doc
from impact_pipeline.execution_profiles import get_execution_profile
from impact_pipeline.hardware_backend import (
    HardwareBackendError,
    backend_summary,
    configure_process_for_hardware,
    normalize_hardware_target,
)
from impact_pipeline.hunter_iim import (
    campaign_status,
    collect_iim_results_by_path,
    prepare_hunter_campaign,
    run_cut_reduce,
    run_cut_reduce_all,
    run_cut_shard,
    run_packed_shard,
    run_phase1_reduce,
    run_phase1_reduce_all,
    run_phase1_shard,
    run_reduce_all,
    summarize_campaign_timing,
    write_iim_results_table,
)
from impact_pipeline.provenance import (
    PROVENANCE_COLUMNS,
    REAL_DATA_ORIGIN,
    assert_origin_matches_dataset,
    collect_code_version,
    collect_runtime_versions,
    is_test_object_origin,
    resolve_dataset_provenance,
    resolve_repo_root,
    write_json,
)
from impact_pipeline.dataset_catalog import get_report_dataset

# ---------------------------------------------------------------------
### ── TOGGLE FULL-RUN STEPS ──────────────────────────────────────────────
RUN_FMRIPREP      = False   # set True to run step 0 (fMRIPrep)
RUN_PREPROCESSING = False   # set True to run step 1 (preprocessing)
RUN_REPLICATION   = False   # set True to run step 7 (replication)

random.seed(42)
np.random.seed(42)
STATS_SEED = 42  # explicit seed for all resampling statistics (steps 4 and 8)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pipeline")
EXPECTED_CONDA_ENV = os.environ.get("IMPACT_CONDA_ENV", "impact-synergy-clean")
HUNTER_STAGES = (
    "build-campaign",
    "phase1-shard",
    "phase1-reduce",
    "phase1-reduce-all",
    "cut-shard",
    "cut-reduce",
    "cut-reduce-all",
    "reduce-all",
    "finalize-pipeline",
    "status",
)
# Robustness atlases for fMRI step 6 (AAL-116 = SPM12 AAL; 'aal90' is the
# legacy file key, still recognised for existing outputs).
ROBUSTNESS_ATLASES = ("aal116", "shen268")
LEGACY_ATLAS_KEYS = {"aal116": "aal90"}
HARDWARE_COLUMNS = ("hardware_target", "hardware_backend", "hardware_runtime")

DATASET_CONFIGS = {
    "ds003171": {
        "modality": "fmri",
        "bids_candidates": (
            root / "data" / "scratch" / "ds003171",
            root / "data" / "ds003171",
        ),
        "atlas": "schaefer400",
        "sessions": ("awake", "deep"),
        "condition": "audio",
        "atlas_robustness": True,
        "supports_fmriprep": True,
        "supports_replication": True,
        "iim_n_parts": None,
        "iim_max_timepoints": None,
        "iim_max_nodes": None,
        "iim_max_mechanism_size": None,
        "iim_max_purview_size": None,
        "iim_parallel_workers": None,
        "iim_memory_target_ratio": 0.90,
        "iim_worker_mem_gb_estimate": 3.0,
        "iim_cpu_oversub_factor": 3.0,
        "iim_phase1_parallel_workers": None,
        "iim_phase1_chunk_size": 8,
        "iim_phase1_shared_memory": True,
        "pdi_params": {
            "bins": 10,
            "weighted": True,
            "normalize": False,
            "clip_negative": True,
            "stability_segments": 4,
            "noise_penalty_kappa": 1.0,
            "component_weights": (0.35, 0.25, 0.20, 0.20),
            "ordinal_order": 3,
            "multiscale_max_scale": 5,
            "eps": 1e-12,
        },
        "pdi_require_explicit_params": True,
        "pdi_require_strict_baseline": True,
        "pdi_primary_endpoint": "anchor",
        "nas_params": {
            "zthr": 1.0,
            "eps": 0.2,
            "tau": 0.2,
            "lambda_phase": 0.5,
            "alpha": 0.20,
            "beta": 0.16,
            "gamma": 0.14,
            "delta": 0.12,
            "eta": 0.16,
            "zeta": 0.12,
            "rho": 0.10,
            "bands": ((0.01, 0.10),),
            "band_weights": (1.0,),
            "window_len": 30,
            "step_len": 15,
            "max_triads": 5000,
            "random_state": 0,
            "workspace_nodes": None,
            "workspace_quantile": 0.2,
            "workspace_min_size": 4,
            "directed_lag": 1,
            "reverberation_lags": (2, 3, 4),
            "baseline_ts": None,
            "boost_against_baseline": False,
            "normalize": True,
        },
        "srpi_params": {
            "modality": "fmri",
            "pre_window_sec": 2.0,
            "response_lag_sec": 4.0,
            "response_window_sec": 6.0,
            "covariance_ridge": 1e-3,
            "component_weights": (0.35, 0.25, 0.20, 0.20),
            "min_events_per_class": 3,
            "sample_reliability_tau": 4.0,
            "eps": 1e-8,
        },
        "srpi_require_explicit_params": True,
        "eeg_session_rules": None,
    },
    "ds002547": {
        "modality": "fmri",
        "bids_candidates": (
            root / "data" / "scratch" / "ds002547",
            root / "data" / "ds002547",
        ),
        "atlas": "schaefer400",
        "sessions": ("awake", "deep"),
        "condition": "selfother",
        "atlas_robustness": False,
        "supports_fmriprep": False,
        "supports_replication": False,
        "iim_n_parts": None,
        "iim_max_timepoints": None,
        "iim_max_nodes": None,
        "iim_max_mechanism_size": None,
        "iim_max_purview_size": None,
        "iim_parallel_workers": None,
        "iim_memory_target_ratio": 0.90,
        "iim_worker_mem_gb_estimate": 3.0,
        "iim_cpu_oversub_factor": 3.0,
        "iim_phase1_parallel_workers": None,
        "iim_phase1_chunk_size": 8,
        "iim_phase1_shared_memory": True,
        "pdi_params": {
            "bins": 10,
            "weighted": True,
            "normalize": False,
            "clip_negative": True,
            "stability_segments": 4,
            "noise_penalty_kappa": 1.0,
            "component_weights": (0.35, 0.25, 0.20, 0.20),
            "ordinal_order": 3,
            "multiscale_max_scale": 5,
            "eps": 1e-12,
        },
        "pdi_require_explicit_params": True,
        "pdi_require_strict_baseline": True,
        "pdi_primary_endpoint": "anchor",
        "nas_params": {
            "zthr": 1.0,
            "eps": 0.2,
            "tau": 0.2,
            "lambda_phase": 0.5,
            "alpha": 0.20,
            "beta": 0.16,
            "gamma": 0.14,
            "delta": 0.12,
            "eta": 0.16,
            "zeta": 0.12,
            "rho": 0.10,
            "bands": ((0.01, 0.10),),
            "band_weights": (1.0,),
            "window_len": 30,
            "step_len": 15,
            "max_triads": 5000,
            "random_state": 0,
            "workspace_nodes": None,
            "workspace_quantile": 0.2,
            "workspace_min_size": 4,
            "directed_lag": 1,
            "reverberation_lags": (2, 3, 4),
            "baseline_ts": None,
            "boost_against_baseline": False,
            "normalize": True,
        },
        "srpi_params": {
            "modality": "fmri",
            "pre_window_sec": 2.0,
            "response_lag_sec": 4.0,
            "response_window_sec": 6.0,
            "covariance_ridge": 1e-3,
            "component_weights": (0.35, 0.25, 0.20, 0.20),
            "min_events_per_class": 3,
            "sample_reliability_tau": 4.0,
            "eps": 1e-8,
        },
        "srpi_require_explicit_params": True,
        "eeg_session_rules": None,
    },
    "ds005620": {
        "modality": "eeg",
        "bids_candidates": (
            root / "data" / "scratch" / "ds005620_annex",
            root / "data" / "scratch" / "ds005620",
            root / "data" / "ds005620",
        ),
        "atlas": "eeg64",
        "sessions": ("awake", "deep"),
        "condition": "eeg",
        "atlas_robustness": False,
        "supports_fmriprep": False,
        "supports_replication": False,
        "iim_n_parts": None,
        "iim_max_timepoints": None,
        "iim_max_nodes": None,
        "iim_max_mechanism_size": None,
        "iim_max_purview_size": None,
        "iim_parallel_workers": None,
        "iim_memory_target_ratio": 0.90,
        "iim_worker_mem_gb_estimate": 3.0,
        "iim_cpu_oversub_factor": 3.0,
        "iim_phase1_parallel_workers": None,
        "iim_phase1_chunk_size": 8,
        "iim_phase1_shared_memory": True,
        "pdi_params": {
            "bins": 10,
            "weighted": True,
            "normalize": False,
            "clip_negative": True,
            "stability_segments": 4,
            "noise_penalty_kappa": 1.0,
            "component_weights": (0.35, 0.25, 0.20, 0.20),
            "ordinal_order": 3,
            "multiscale_max_scale": 5,
            "eps": 1e-12,
        },
        "pdi_require_explicit_params": True,
        "pdi_require_strict_baseline": True,
        "pdi_primary_endpoint": "anchor",
        "nas_params": {
            "zthr": 1.0,
            "eps": 0.2,
            "tau": 0.2,
            "lambda_phase": 0.5,
            "alpha": 0.20,
            "beta": 0.16,
            "gamma": 0.14,
            "delta": 0.12,
            "eta": 0.16,
            "zeta": 0.12,
            "rho": 0.10,
            "bands": ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)),
            "band_weights": (0.2, 0.2, 0.2, 0.2, 0.2),
            "window_len": 500,
            "step_len": 250,
            "max_triads": 5000,
            "random_state": 0,
            "workspace_nodes": None,
            "workspace_quantile": 0.2,
            "workspace_min_size": 4,
            "directed_lag": 1,
            "reverberation_lags": (2, 3, 4),
            "baseline_ts": None,
            "boost_against_baseline": False,
            "normalize": True,
        },
        "srpi_params": {
            "modality": "eeg",
            "pre_window_sec": 0.20,
            "response_lag_sec": 0.05,
            "response_window_sec": 0.40,
            "covariance_ridge": 1e-3,
            "component_weights": (0.35, 0.25, 0.20, 0.20),
            "min_events_per_class": 3,
            "sample_reliability_tau": 4.0,
            "eps": 1e-8,
        },
        "srpi_require_explicit_params": True,
        # Analysed runs: awake eyes-closed rest; deep = first sed2 (else sed)
        # rest run, i.e. the run the events resolver selects as well, so the
        # analysed set does not depend on which metrics are requested.
        "eeg_session_rules": {
            "awake": [("awake", "EC")],
            "deep": [("sed2", "rest", "first"), ("sed", "rest", "first")],
        },
        # Rest/baseline recordings for PDI: same state, disjoint from the
        # analysed runs (awake eyes-open rest; remaining sed2/sed rest runs).
        "eeg_rest_rules": {
            "awake": [("awake", "EO")],
            "deep": [("sed2", "rest", "after_first"), ("sed", "rest", "after_first")],
        },
    },
}


def _first_existing_path(*candidates):
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _apply_metric_subset(df, df_mean, mpc_metrics=None, compute_ci=True):
    """
    Keep only S and the selected MPC metrics in run tables.
    Useful when reusing cached step-2 outputs produced with a wider metric set.
    """
    if mpc_metrics is None:
        selected = {"RAM", "PDI", "NAS", "IIM", "SRPI"}
    else:
        selected = set(mpc_metrics)

    keep_df = [c for c in PROVENANCE_COLUMNS if c in df.columns]
    keep_df.extend(c for c in HARDWARE_COLUMNS if c in df.columns)
    keep_df.extend(['subject', 'session', 'theta', 'S'])
    for metric in ("RAM", "PDI", "NAS", "IIM", "SRPI"):
        if metric in selected and metric in df.columns:
            keep_df.append(metric)
    if "PDI" in selected:
        for extra in (
            "PDI_anchor",
            "PDI_task",
            "PDI_anchor_defined",
            "PDI_task_defined",
            "PDI_anchor_reason",
            "PDI_task_reason",
            "PDI_primary_endpoint",
            "PDI_primary_source",
            "PDI_baseline_policy",
            "PDI_anchor_baseline_n_runs",
            "PDI_task_baseline_n_runs",
            "PDI_anchor_baseline_paths",
            "PDI_task_baseline_paths",
        ):
            if extra in df.columns:
                keep_df.append(extra)
    if "IIM" in selected:
        for extra in ("IIM_raw", "IIM_raw_scaled", "IIM_defined", "IIM_undefined_reason"):
            if extra in df.columns:
                keep_df.append(extra)
    if compute_ci and "CI" in df.columns and selected.issuperset({"RAM", "PDI", "NAS", "IIM", "SRPI"}):
        keep_df.append("CI")
        for extra in ("CI_defined", "CI_missing", "CI_reference",
                      "RAM_norm", "PDI_norm", "NAS_norm", "IIM_norm", "SRPI_norm"):
            if extra in df.columns:
                keep_df.append(extra)

    df_filtered = df.loc[:, [c for c in keep_df if c in df.columns]].copy()

    # Some cached df_mean tables may miss per-metric columns; recover from df.
    desired_means = []
    for m in ("RAM", "PDI", "PDI_anchor", "PDI_task", "NAS", "IIM", "SRPI", "IIM_raw", "IIM_raw_scaled"):
        include = (m in selected) or (m.startswith("IIM") and "IIM" in selected)
        if m in {"PDI_anchor", "PDI_task"} and "PDI" in selected:
            include = True
        if include:
            desired_means.append(m)
    missing_in_mean = [m for m in desired_means if m not in df_mean.columns and m in df_filtered.columns]
    if missing_in_mean:
        recovered = (
            df_filtered
            .groupby(["subject", "session"])[missing_in_mean]
            .mean()
            .reset_index()
        )
        df_mean = df_mean.merge(recovered, on=["subject", "session"], how="left")

    keep_mean = [c for c in PROVENANCE_COLUMNS if c in df_mean.columns]
    keep_mean.extend(c for c in HARDWARE_COLUMNS if c in df_mean.columns)
    keep_mean.extend(['subject', 'session', 'S'])
    for metric in ("RAM", "PDI", "PDI_anchor", "PDI_task", "NAS", "IIM", "SRPI", "IIM_raw", "IIM_raw_scaled"):
        include = (metric in selected) or (metric.startswith("IIM") and "IIM" in selected)
        if metric in {"PDI_anchor", "PDI_task"} and "PDI" in selected:
            include = True
        if metric in df_mean.columns and include:
            keep_mean.append(metric)
    if compute_ci and "CI" in df_mean.columns and selected.issuperset({"RAM", "PDI", "NAS", "IIM", "SRPI"}):
        keep_mean.append("CI")

    df_mean_filtered = df_mean.loc[:, [c for c in keep_mean if c in df_mean.columns]].copy()
    return df_filtered, df_mean_filtered


def _write_run_provenance_manifest(
    *,
    cache_dir: Path,
    provenance,
    modality: str,
    bids_root: Path | None,
    execution_mode: str,
    atlas: str,
    sessions,
    condition: str,
    parameters: dict | None = None,
    hardware=None,
    status: str | None = None,
    extra: dict | None = None,
) -> None:
    payload = provenance.as_manifest_dict()
    payload.update(
        {
            "modality": str(modality),
            "bids_root": (None if bids_root is None else str(Path(bids_root).resolve())),
            "execution_mode": str(execution_mode),
            "atlas": str(atlas),
            "sessions": [str(s) for s in sessions],
            "condition": str(condition),
            "status": (None if status is None else str(status)),
            "code_version": collect_code_version(root),
            "runtime": collect_runtime_versions(),
            "hardware_backend": (
                None
                if hardware is None
                else (hardware.to_dict() if hasattr(hardware, "to_dict") else hardware)
            ),
            "parameters": _json_safe(parameters or {}),
            "result_usage_note": (
                "Results retain explicit dataset-origin provenance. Synthetic validation outputs "
                "remain isolated from real-study outputs unless selected explicitly."
            ),
        }
    )
    if extra:
        payload.update(_json_safe(extra))
    write_json(cache_dir / "provenance_manifest.json", payload)


def _json_safe(value):
    """Convert tuples/Paths/numpy scalars so run parameters serialise to JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _portable_ci_reference(ci_reference):
    """--ci-reference as recorded/forwarded: a JSON path becomes absolute."""
    if ci_reference is None or isinstance(ci_reference, dict):
        return ci_reference
    text = str(ci_reference).strip()
    if text == "cohort_high_state":
        return text
    return str(Path(text).expanduser().resolve())


def _ensure_provenance_columns(df: pd.DataFrame, provenance) -> pd.DataFrame:
    out = df.copy()
    for key, value in provenance.as_result_metadata().items():
        if key not in out.columns:
            out[key] = value
        else:
            out[key] = out[key].replace("", pd.NA).fillna(value)
    return out


def _export_test_object_metric_bank(
    *,
    df: pd.DataFrame,
    df_mean: pd.DataFrame,
    cache_dir: Path,
    provenance,
    modality: str,
    atlas: str,
    sessions,
    condition: str,
) -> None:
    metric_bank_dir = provenance.metric_bank_dataset_dir
    if metric_bank_dir is None:
        return

    metric_bank_dir.mkdir(parents=True, exist_ok=True)
    per_metric_dir = metric_bank_dir / "per_metric"
    per_metric_dir.mkdir(parents=True, exist_ok=True)

    full_df_path = metric_bank_dir / "step2_df.csv"
    full_mean_path = metric_bank_dir / "step2_df_mean.csv"
    df.to_csv(full_df_path, index=False)
    df_mean.to_csv(full_mean_path, index=False)

    metric_files = {}
    for metric in ("RAM", "PDI", "NAS", "IIM", "SRPI", "CI"):
        if metric not in df.columns:
            continue
        keep_cols = [c for c in PROVENANCE_COLUMNS if c in df.columns]
        keep_cols.extend([c for c in ("subject", "session", "theta", "S", metric) if c in df.columns])
        if metric == "PDI":
            keep_cols.extend(
                [
                    c
                    for c in (
                        "PDI_anchor",
                        "PDI_task",
                        "PDI_anchor_defined",
                        "PDI_task_defined",
                        "PDI_anchor_reason",
                        "PDI_task_reason",
                        "PDI_primary_endpoint",
                        "PDI_primary_source",
                        "PDI_baseline_policy",
                        "PDI_anchor_baseline_n_runs",
                        "PDI_task_baseline_n_runs",
                        "PDI_anchor_baseline_paths",
                        "PDI_task_baseline_paths",
                    )
                    if c in df.columns
                ]
            )
        if metric == "IIM":
            keep_cols.extend(
                [c for c in ("IIM_raw", "IIM_raw_scaled", "IIM_defined", "IIM_undefined_reason") if c in df.columns]
            )
        out_path = per_metric_dir / f"{metric}.csv"
        df.loc[:, keep_cols].to_csv(out_path, index=False)
        metric_files[metric] = str(out_path)

    manifest = provenance.as_manifest_dict()
    manifest.update(
        {
            "source_dataset_id": str(provenance.dataset_id),
            "source_type": "synthetic_validation",
            "modality": str(modality),
            "atlas": str(atlas),
            "sessions": [str(s) for s in sessions],
            "condition": str(condition),
            "step2_cache_dir": str(cache_dir),
            "step2_df_path": str(full_df_path),
            "step2_df_mean_path": str(full_mean_path),
            "available_metrics": [m for m in ("RAM", "PDI", "NAS", "IIM", "SRPI", "CI") if m in df.columns],
            "metric_files": metric_files,
            "row_count": int(len(df)),
            "row_count_mean": int(len(df_mean)),
            "usage_note": (
                "This metric bank contains synthetic validation results. Mixed-source CI analyses "
                "must select these results explicitly and keep provenance labels intact."
            ),
        }
    )
    write_json(metric_bank_dir / "metric_bank_manifest.json", manifest)


def _assert_expected_runtime_env(expected_env: str = EXPECTED_CONDA_ENV) -> None:
    """
    Fail fast when the pipeline is launched from an unexpected Python runtime.
    """
    skip_check = os.environ.get("IMPACT_SKIP_ENV_CHECK", "").strip().lower()
    if skip_check in {"1", "true", "yes", "on"}:
        log.warning("Skipping runtime environment check (IMPACT_SKIP_ENV_CHECK=%s).", skip_check)
        return

    conda_default = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    exe = sys.executable
    tokens = [conda_default, conda_prefix, exe]
    ok = any(expected_env in t for t in tokens if t)

    if not ok:
        raise RuntimeError(
            "Invalid Python runtime for run_pipeline.py.\n"
            f"Expected conda env: '{expected_env}'\n"
            f"Detected CONDA_DEFAULT_ENV='{conda_default}', CONDA_PREFIX='{conda_prefix}', "
            f"sys.executable='{exe}'.\n"
            f"Run with: conda run -n {expected_env} python run_pipeline.py ..."
        )


def _write_eeg_preprocessing_summary(cache_dir: Path, summary_payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    (cache_dir / "preprocessing_eeg_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    table_map = {
        "missing_files": (
            "preprocessing_eeg_missing_files.csv",
            ["subject", "session", "task", "acquisition", "file", "reason"],
        ),
        "skipped_subjects": (
            "preprocessing_eeg_skipped_subjects.csv",
            ["subject", "reason", "detail"],
        ),
        "written_runs": (
            "preprocessing_eeg_written_runs.csv",
            [
                "subject",
                "session",
                "segment",
                "source_file",
                "output_file",
                "run_index",
                "bids_run",
                "run_index_source",
                "n_channels",
                "n_timepoints",
                "sfreq_hz",
            ],
        ),
        "excluded_channels": (
            "preprocessing_eeg_excluded_channels.csv",
            ["subject", "channel", "type", "source"],
        ),
        "missing_rest": (
            "preprocessing_eeg_missing_rest.csv",
            ["subject", "session", "file", "reason"],
        ),
    }
    for key, (name, cols) in table_map.items():
        rows = summary_payload.get(key, [])
        pd.DataFrame(rows).reindex(columns=cols).to_csv(cache_dir / name, index=False)


def _resolve_freesurfer_license_path() -> Path:
    env_path = os.environ.get("FS_LICENSE", "").strip()
    if env_path:
        lic = Path(env_path).expanduser().resolve()
        if not lic.exists():
            raise FileNotFoundError(
                f"FS_LICENSE points to '{lic}', but that file does not exist."
            )
        return lic

    local_default = root / "licenses" / "fs_license.txt"
    if local_default.exists():
        return local_default

    raise FileNotFoundError(
        "FreeSurfer license not found. Set FS_LICENSE to your local license file "
        "or place it at 'licenses/fs_license.txt' (see licenses/fs_license.txt.example)."
    )


FMRIPREP_IMAGE = "nipreps/fmriprep:25.1.3"
FMRIPREP_LICENSE_IN_CONTAINER = "/opt/freesurfer/license.txt"


def canonical_fmriprep_dir(bids_root) -> Path:
    """Canonical fMRIPrep derivatives location: <bids_root>/derivatives/fmriprep."""
    return Path(bids_root).expanduser().resolve() / "derivatives" / "fmriprep"


def _has_fmriprep_subjects(path: Path) -> bool:
    return path.is_dir() and any(p.is_dir() for p in path.glob("sub-*"))


def _resolve_fmriprep_dir(fmriprep_dir_override, bids_root, out) -> Path:
    """
    fMRIPrep derivatives used by preprocessing (and written by --run-fmriprep):
    --fmriprep-dir if given, else <bids_root>/derivatives/fmriprep. The legacy
    <out>/fmriprep location is still read when only it holds derivatives.
    """
    if fmriprep_dir_override is not None:
        return Path(fmriprep_dir_override).expanduser().resolve()
    canonical = canonical_fmriprep_dir(bids_root)
    legacy = Path(out) / "fmriprep"
    if not _has_fmriprep_subjects(canonical) and _has_fmriprep_subjects(legacy):
        log.warning(
            "Using legacy fMRIPrep derivatives at %s (canonical location: %s).",
            legacy,
            canonical,
        )
        return legacy.resolve()
    return canonical


def _fmriprep_subject_complete(fmriprep_out: Path, sub: str) -> bool:
    out = Path(fmriprep_out)
    report = out / f"sub-{sub}.html"
    bold = list((out / f"sub-{sub}").glob("**/func/*desc-preproc_bold.nii.gz"))
    return report.exists() and bool(bold)


def ensure_fmriprep(
    bids_dir,
    fmriprep_out,
    work_dir,
    fs_license,
    freesurf_out,
    subjects=None,
    *,
    nthreads=None,
    omp_nthreads=None,
    mem_mb=None,
    image=None,
    skip_existing=True,
):
    """
    Run fMRIPrep (docker) for subjects whose derivatives are not complete yet.

    The license file is mounted by its own path (any file name) at
    /opt/freesurfer/license.txt. Resource limits default to
    IMPACT_FMRIPREP_NTHREADS / IMPACT_FMRIPREP_OMP_NTHREADS /
    IMPACT_FMRIPREP_MEM_MB (16 / 8 / 96000) and the pinned image to
    IMPACT_FMRIPREP_IMAGE (nipreps/fmriprep:25.1.3).
    """
    bids_dir = Path(bids_dir).expanduser().resolve()
    fmriprep_out = Path(fmriprep_out).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()
    freesurf_out = Path(freesurf_out).expanduser().resolve()
    license_path = Path(fs_license).expanduser().resolve()
    nthreads = int(nthreads or os.environ.get("IMPACT_FMRIPREP_NTHREADS", 16))
    omp_nthreads = int(
        omp_nthreads or os.environ.get("IMPACT_FMRIPREP_OMP_NTHREADS", 8)
    )
    mem_mb = int(mem_mb or os.environ.get("IMPACT_FMRIPREP_MEM_MB", 96000))
    image = str(image or os.environ.get("IMPACT_FMRIPREP_IMAGE", FMRIPREP_IMAGE))

    layout = BIDSLayout(str(bids_dir), validate=False)
    all_subj = sorted(layout.get(return_type='id', target='subject'))
    if subjects:
        wanted = {str(s).strip().replace("sub-", "", 1) for s in subjects}
        subjects = [s for s in all_subj if s in wanted]
    else:
        subjects = all_subj

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(fmriprep_out, exist_ok=True)
    os.makedirs(freesurf_out, exist_ok=True)

    ran = []
    for sub in subjects:
        if skip_existing and _fmriprep_subject_complete(fmriprep_out, sub):
            log.info(
                "fMRIPrep derivatives for sub-%s already present in %s; skipping.",
                sub,
                fmriprep_out,
            )
            continue
        log.info("Running fMRIPrep on %s …", sub)
        cache_host = work_dir / "bids_db"
        cache_cont = '/bids_db'
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{bids_dir}:/data",
            "-v", f"{fmriprep_out}:/out",
            "-v", f"{work_dir}:/work",
            "-v", f"{freesurf_out}:/out_freesurfer",
            # the license is mounted by its own path, whatever its file name
            "-v", f"{license_path}:{FMRIPREP_LICENSE_IN_CONTAINER}:ro",
            "-v", f"{cache_host}:{cache_cont}",
            image,
            "/data", "/out", "participant",
            "--participant-label", sub,
            "--fs-license-file", FMRIPREP_LICENSE_IN_CONTAINER,
            "--fs-subjects-dir", "/out_freesurfer",
            "--bids-database-dir", cache_cont,
            "--work-dir", "/work",
            # "--clean-workdir",  # final full-dataset runs only (discards work)
            "--skip-bids-validation",
            "--nthreads", str(nthreads),
            "--omp-nthreads", str(omp_nthreads),
            "--mem", str(mem_mb),
        ]
        subprocess.run(cmd, check=True)
        ran.append(sub)
    return ran


def _persist_step2_outputs(
    cache_dir: Path,
    df: pd.DataFrame,
    df_mean: pd.DataFrame,
    df_stats_by_theta: pd.DataFrame,
    *,
    persist_primary: bool,
) -> None:
    if bool(persist_primary):
        step2_df_path = cache_dir / "step2_df.csv"
        step2_df_mean_path = cache_dir / "step2_df_mean.csv"
        step2_theta_stats_path = cache_dir / "step2_theta_stats.csv"
        df.to_csv(step2_df_path, index=False)
        df_mean.to_csv(step2_df_mean_path, index=False)
        df_stats_by_theta.reset_index().to_csv(step2_theta_stats_path, index=False)
    (cache_dir / "step2_df_active.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (cache_dir / "step2_df_mean_active.csv").write_text(df_mean.to_csv(index=False), encoding="utf-8")


def _jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return [_jsonable(r) for r in obj.to_dict("records")]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def _ci_reference_record(df):
    """Reference label and per-component means used for CI (for the stats outputs)."""
    from impact_pipeline.synergy_ci import resolve_ci_references

    if "CI_reference" not in df.columns or not df["CI_reference"].notna().any():
        return None
    label = str(df["CI_reference"].dropna().iloc[0])
    rec = {"ci_reference": label}
    try:
        if label == "cohort_high_state":
            rec["references"], _ = resolve_ci_references(df)
        elif label.startswith("external_json:"):
            ref_path = label.split(":", 1)[1]
            rec["references"], _ = resolve_ci_references(df, reference=ref_path)
    except Exception as exc:  # recorded, not fatal
        rec["references_error"] = f"{type(exc).__name__}: {exc}"
    return rec


def _resolve_replication_root() -> Path:
    return _first_existing_path(
        root / "data" / "scratch" / "melbourne",
        root / "data" / "scratch" / "melbourne_propofol",
        root / "data" / "melbourne",
        root / "data" / "melbourne_propofol",
    )


def _check_replication_inputs(melb_root: Path) -> None:
    """Fail before the expensive steps when replication cannot run at all."""
    if not melb_root.exists():
        raise FileNotFoundError(
            "--run-replication requested, but the replication dataset is missing at "
            f"'{melb_root}'. Download it or run without --run-replication."
        )
    # Same derivative search as run_replication (single source of truth).
    deriv_root, _n_files, searched = _find_fmriprep_derivatives(str(melb_root))
    if deriv_root is None:
        raise FileNotFoundError(
            "--run-replication requested, but no fMRIPrep desc-preproc BOLD files "
            f"were found (searched: {', '.join(searched)}). Run fMRIPrep for the "
            "replication dataset first or run without --run-replication."
        )


def _missing_atlas_runs(prep_out, atlas, sessions, condition, subjects=None, limit=10):
    """Subject/session folders lacking <subj>_run-*_<atlas>_ts.npy (step-6 inputs)."""
    prep_out = Path(prep_out)
    if not prep_out.exists():
        return ["<missing preprocessed directory>"]
    wanted = (
        None if subjects is None else {str(s).replace("sub-", "", 1) for s in subjects}
    )
    missing = []
    for subj_dir in sorted(
        p for p in prep_out.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        if wanted is not None and subj_dir.name not in wanted:
            continue
        for ses in sessions:
            if not list(
                (subj_dir / str(ses) / str(condition)).glob(
                    f"{subj_dir.name}_run-*_{atlas}_ts.npy"
                )
            ):
                missing.append(f"{subj_dir.name}/{ses}")
                if len(missing) >= int(limit):
                    return missing
    return missing


def _bids_repetition_time(bids_root):
    """First RepetitionTime in sub-*/func or sub-*/ses-*/func *_bold.json sidecars."""
    if bids_root is None:
        return None
    broot = Path(bids_root)
    sidecars = sorted(
        list(broot.glob("sub-*/func/*_bold.json"))
        + list(broot.glob("sub-*/ses-*/func/*_bold.json"))
    )
    for sidecar_path in sidecars:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            tr = json.load(f).get("RepetitionTime")
        if tr is not None and np.isfinite(float(tr)) and float(tr) > 0:
            return float(tr)
    return None


def _run_atlas_robustness(
    *,
    prep_out,
    bids_root,
    sessions,
    compute_ci,
    df,
    subjects,
    dataset_id,
    pdi_params,
    pdi_require_explicit_params,
    pdi_require_strict_baseline,
    pdi_primary_endpoint,
    nas_params,
    srpi_params,
    srpi_require_explicit_params,
    context=None,
):
    """
    Step 6: repeat the metrics on the robustness atlases with the same IIM
    settings, condition, TR and hardware as step 2. Atlases without
    time series are skipped with a recorded reason instead of crashing. In
    Hunter mode IIM is excluded (it is not distributed for these atlases).
    """
    ctx = dict(context or {})
    if not bool(ctx.get("enabled", True)):
        return {
            "skipped": {
                "reason": "atlas robustness disabled (--no-atlas-robustness)",
                "dataset": dataset_id,
            }
        }
    condition = ctx.get("condition") or "audio"
    atlases = tuple(ctx.get("atlases") or ROBUSTNESS_ATLASES)
    tr = ctx.get("tr")
    if tr is None:
        from impact_pipeline.run_synergy_ci import _infer_sample_interval_seconds

        # Same TR source as step 2, then any session-level BOLD sidecar.
        tr = _infer_sample_interval_seconds(bids_root)
        if tr is None:
            tr = _bids_repetition_time(bids_root)
    if tr is None:
        return {
            "skipped": {
                "reason": (
                    "no sample interval: no *_bold.json sidecar with "
                    "RepetitionTime under the BIDS root (pass --tr)"
                ),
                "dataset": dataset_id,
            }
        }

    results, usable, labels = {}, [], {}
    for atlas in atlases:
        key = str(atlas)
        missing = _missing_atlas_runs(prep_out, key, sessions, condition, subjects)
        legacy = LEGACY_ATLAS_KEYS.get(key)
        if (
            missing
            and legacy
            and not _missing_atlas_runs(prep_out, legacy, sessions, condition, subjects)
        ):
            key, missing = legacy, []
        if missing:
            log.warning(
                "6/9 Atlas robustness: skipping %s (no time series for %s)",
                atlas,
                ", ".join(missing),
            )
            results[str(atlas)] = {
                "skipped": {"reason": "missing_atlas_timeseries", "missing": missing}
            }
            continue
        usable.append(key)
        labels[key] = str(atlas)
    if not usable:
        return results

    robust_metrics = [
        m for m in ("RAM", "PDI", "NAS", "IIM", "SRPI") if m in df.columns
    ]
    notes = []
    if str(ctx.get("execution_mode", "local")) == "hunter" and "IIM" in robust_metrics:
        robust_metrics.remove("IIM")
        notes.append(
            "IIM omitted in Hunter finalize: robustness-atlas IIM is not distributed."
        )
    iim_settings = dict(ctx.get("iim_settings") or {})
    synergy_kwargs = {
        "condition": condition,
        "hardware_target": ctx.get("hardware_target", "cpu"),
        "dataset_id": dataset_id,
    }
    synergy_kwargs.update({k: v for k, v in iim_settings.items() if v is not None})
    # Provenance of the primary run is carried into the robustness rows.
    synergy_kwargs.update({
        k: str(df[k].dropna().iloc[0])
        for k in ("data_origin", "dataset_role", "provenance_label")
        if k in df.columns and df[k].notna().any()
    })
    log.info("6/9 Testing robustness across atlases: %s", ", ".join(labels.values()))
    atlas_res = atlas_check(
        str(prep_out),
        atlases=tuple(usable),
        sessions=sessions,
        thetas=np.arange(0.1, 1.0, 0.1),
        tr=float(tr),
        stimulus_onsets=None,
        mpc_metrics=robust_metrics,
        compute_ci=bool(
            compute_ci and ("CI" in df.columns) and "IIM" in robust_metrics
        ),
        pdi_params=pdi_params,
        pdi_require_explicit_params=pdi_require_explicit_params,
        pdi_require_strict_baseline=pdi_require_strict_baseline,
        pdi_primary_endpoint=pdi_primary_endpoint,
        nas_params=nas_params,
        srpi_params=srpi_params,
        srpi_require_explicit_params=srpi_require_explicit_params,
        subjects=subjects,
        **synergy_kwargs,
    )
    for key, payload in dict(atlas_res).items():
        if isinstance(payload, dict) and notes:
            payload = dict(payload)
            payload["notes"] = list(payload.get("notes") or []) + notes
        results[labels.get(key, key)] = payload
    return results


def _postprocess_after_step2(
    *,
    df,
    df_mean,
    prep_out,
    bids_root,
    figdir,
    atlas,
    sessions,
    compute_ci,
    modality,
    dataset_id,
    cfg,
    subjects,
    out,
    report_doc,
    pdi_params,
    pdi_require_explicit_params,
    pdi_require_strict_baseline,
    pdi_primary_endpoint,
    nas_params,
    srpi_params,
    srpi_require_explicit_params,
    run_replication_flag,
    df_stats_by_theta,
    condition=None,
    stats_seed=STATS_SEED,
    robustness_context=None,
):
    condition_eff = condition if condition is not None else (cfg or {}).get("condition")
    pair = tuple(str(s) for s in tuple(sessions)[:2])
    stats_dir = Path(out) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    log.info("3/9 Computing baseline graph metrics")
    df_baseline = compute_baseline_metrics(
        df_mean,
        data_dir=str(prep_out),
        atlas=atlas,
        condition=condition_eff,
    )
    log.debug("df_baseline columns: %s", df_baseline.columns.tolist())

    log.info("4/9 Paired statistics (subject bootstrap, two-sided permutation, Holm)")
    stats = {"seed": int(stats_seed), "sessions": list(pair)}
    stats["condition"] = condition_eff
    has_ci = 'CI' in df_mean.columns
    if has_ci:
        # Undefined CI rows are NaN and excluded (never zeros); report how many.
        stats['ci_definedness'] = definedness_summary(df, sessions=pair)
        stats['ci_reference'] = _ci_reference_record(df)
        cd = stats['ci_definedness']
        log.info(
            "CI defined in %s/%s %ss (%s undefined, excluded); "
            "%s subject(s) with defined CI in both sessions",
            cd.get('n_rows_defined'), cd.get('n_rows'), cd.get('count_unit', 'row'),
            cd.get('n_rows_undefined'), cd.get('n_subjects_complete_pairs', 'na'),
        )
    boot_S = bootstrap_ci(df_mean, 'S', sessions=pair, random_state=stats_seed)
    auc_S, p_S = permutation_test_auc(
        df_mean, 'S', sessions=pair, random_state=stats_seed
    )
    stats['auc_S'] = {'auc': auc_S, 'lo': boot_S[0], 'hi': boot_S[1], 'p': p_S}
    stats['s_test'] = paired_session_test(df_mean, 'S', pair)
    if has_ci:
        boot_CI = bootstrap_ci(df_mean, 'CI', sessions=pair, random_state=stats_seed)
        auc_CI, p_CI = permutation_test_auc(
            df_mean, 'CI', sessions=pair, random_state=stats_seed
        )
        stats['auc_CI'] = {'auc': auc_CI, 'lo': boot_CI[0], 'hi': boot_CI[1], 'p': p_CI}
        stats['ci_test'] = paired_session_test(df_mean, 'CI', pair)
    comp_metrics = [
        m for m in ("RAM", "PDI", "NAS", "IIM", "SRPI") if m in df_mean.columns
    ]
    stats['components'] = paired_tests_table(
        df_mean, comp_metrics, pair, family="mpc_components"
    )

    # S is theta-dependent (exploratory): every theta is tested, Holm across theta.
    theta_results = {}
    for theta in sorted(df['theta'].unique()):
        subdf = df[df['theta'] == theta]
        df_theta_mean = (
            subdf.groupby(['subject', 'session'])
                 .agg(S=('S', 'mean'))
                 .reset_index()
        )
        auc_S_theta, p_S_theta = permutation_test_auc(
            df_theta_mean, 'S', sessions=pair, random_state=stats_seed
        )
        theta_results[theta] = {'auc_S': auc_S_theta, 'p_S': p_S_theta}
    theta_keys = list(theta_results.keys())
    theta_p_holm = holm_adjust([theta_results[k]['p_S'] for k in theta_keys])
    for key, p_adj in zip(theta_keys, theta_p_holm):
        theta_results[key]['p_S_holm'] = float(p_adj)
    if df_stats_by_theta is not None and 'p_S' in df_stats_by_theta.columns:
        df_stats_by_theta = df_stats_by_theta.copy()
        p_theta = df_stats_by_theta['p_S'].to_numpy(dtype=float)
        df_stats_by_theta['p_S_holm'] = holm_adjust(p_theta)

    log.info("5/9 Motion covariate analysis")
    if log.isEnabledFor(logging.DEBUG):
        log.debug("df columns: %s", df.columns.tolist())
        log.debug("df_mean columns: %s", df_mean.columns.tolist())
        rows_per_group = (
            df.groupby(['subject', 'session']).size()
            .rename('rows').reset_index()
            .sort_values('rows', ascending=False)
            .head(10)
        )
        log.debug("rows per (subject, session):\n%s", rows_per_group.to_string(index=False))
        if 'theta' in df.columns:
            n_theta = (
                df.groupby(['subject', 'session'])['theta']
                .nunique().rename('n_theta').reset_index()
                .sort_values('n_theta', ascending=False)
                .head(10)
            )
            log.debug("n_theta per (subject, session):\n%s", n_theta.to_string(index=False))
        for cand in ['condition', 'task', 'run', 'acq', 'desc']:
            if cand in df.columns:
                log.debug("%s unique values: %s", cand, df[cand].unique()[:10])

    if (modality == "fmri") and has_ci:
        motion = motion_covariate_analysis(
            df_mean, str(prep_out), atlas=atlas, condition=condition_eff, sessions=pair
        )
    else:
        motion = {'skipped': f"motion~CI omitted for modality={modality} or missing CI."}

    if cfg.get("atlas_robustness", False) and modality == "fmri":
        # Same condition, TR, IIM settings and hardware as step 2; only the
        # analysed session pair is recomputed (the statistics compare that pair).
        robustness_ctx = dict(robustness_context or {})
        if not robustness_ctx.get("condition"):
            robustness_ctx["condition"] = condition_eff
        atlas_res = _run_atlas_robustness(
            prep_out=prep_out,
            bids_root=bids_root,
            sessions=pair,
            compute_ci=compute_ci,
            df=df,
            subjects=subjects,
            dataset_id=dataset_id,
            pdi_params=pdi_params,
            pdi_require_explicit_params=pdi_require_explicit_params,
            pdi_require_strict_baseline=pdi_require_strict_baseline,
            pdi_primary_endpoint=pdi_primary_endpoint,
            nas_params=nas_params,
            srpi_params=srpi_params,
            srpi_require_explicit_params=srpi_require_explicit_params,
            context=robustness_ctx,
        )
    else:
        atlas_res = {
            "skipped": {
                "reason": f"atlas robustness not configured for modality={modality}",
                "dataset": dataset_id,
            }
        }

    repl = None
    if run_replication_flag and cfg.get("supports_replication", False):
        log.info("7/9 Replication (Melbourne Propofol)")
        melb_root = _resolve_replication_root()
        melb_out = out / 'melbourne' / 'preprocessed'
        try:
            repl = run_replication(
                data_root=str(melb_root),
                out_dir=str(melb_out),
                atlas=atlas,
                sessions=pair,
                random_state=stats_seed,
            )
        except Exception as exc:
            # A failing optional replication must not discard the finished analysis.
            log.error(
                "7/9 Replication failed (%s: %s); continuing with the report.",
                type(exc).__name__,
                exc,
            )
            repl = {"dataset": "melbourne", "error": f"{type(exc).__name__}: {exc}"}
    elif run_replication_flag:
        log.info("7/9 Replication skipped: not configured for dataset '%s'", dataset_id)

    log.info("8/9 Comparing to baseline models (exploratory S; CI when defined)")
    mc = compare_models(
        df_baseline,
        metrics=BASELINE_METRICS,
        sessions=pair,
    )
    if has_ci and 'CI' in df_baseline.columns:
        mc_ci = compare_models(
            df_baseline, metrics=BASELINE_METRICS, sessions=pair, score_col='CI'
        )
        mc.update({f"{m} [CI]": res for m, res in mc_ci.items()})
    log.debug("baseline session counts:\n%s", df_baseline['session'].value_counts().to_string())

    # Persist every statistic (nothing computed here is discarded).
    stats['theta_permutation_S'] = theta_results
    stats['model_comparison'] = mc
    stats['motion'] = motion
    stats['atlas_robustness'] = atlas_res
    stats['replication'] = repl
    if isinstance(stats['components'], pd.DataFrame):
        stats['components'].to_csv(stats_dir / "component_tests.csv", index=False)
    if df_stats_by_theta is not None:
        theta_tab = df_stats_by_theta.reset_index()
        theta_perm = pd.DataFrame(
            [{'theta': k, **v} for k, v in theta_results.items()]
        ).rename(columns={
            'auc_S': 'perm_auc_S', 'p_S': 'perm_p_S', 'p_S_holm': 'perm_p_S_holm',
        })
        if 'theta' in theta_tab.columns and not theta_perm.empty:
            theta_tab = theta_tab.merge(theta_perm, on='theta', how='outer')
        theta_tab.to_csv(stats_dir / "theta_tests_S.csv", index=False)
    if isinstance(motion, pd.DataFrame):
        motion.to_csv(stats_dir / "motion_covariates.csv", index=False)
    (stats_dir / "statistics_summary.json").write_text(
        json.dumps(_jsonable(stats), indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Statistics written to %s", stats_dir)

    log.info("9/9 Generating Word report")
    report_path = out / report_doc
    create_doc(
        path=str(report_path),
        df=df,
        df_stats_by_theta=df_stats_by_theta,
        motion=motion,
        atlas_results=atlas_res,
        mc=mc,
        theta_results=theta_results,
        fig_dir=str(figdir),
        use_sem=True,
        S_SCALE=1e3,
        RAM_SCALE=1e3,
        CI_SCALE=1.0,
        LABEL_S="S×10³",
        LABEL_RAM="RAM×10³",
        LABEL_CI="CI (reference-normalised)",
        repl=repl,
        stats=stats,
        modality=modality,
        sessions=pair,
    )
    log.info("Pipeline complete! Outputs in %s", out)


def _run_hunter_stage(
    *,
    hunter_stage,
    hunter_campaign_dir,
    hunter_task_index,
    hunter_run_index,
    execution_profile,
    prep_out,
    bids_root,
    figdir,
    cache_dir,
    atlas,
    sessions,
    condition,
    metric_tr,
    subjects,
    mpc_metrics,
    compute_ci,
    iim_n_parts,
    iim_max_timepoints,
    iim_max_nodes,
    iim_max_mechanism_size,
    iim_max_purview_size,
    pdi_params,
    pdi_require_explicit_params,
    pdi_require_strict_baseline,
    pdi_primary_endpoint,
    nas_params,
    srpi_params,
    srpi_require_explicit_params,
    out,
    modality,
    dataset_id,
    data_origin,
    dataset_role,
    provenance_label,
    cfg,
    report_doc,
    run_replication_flag,
    hardware_target,
    hunter_array_index=None,
    hunter_shards_per_node=None,
    hunter_scheduler=None,
    hunter_settings_overrides=None,
    build_hardware_backend=None,
    iim_settings=None,
    run_parameters=None,
    atlas_robustness=True,
    ci_reference=None,
    hunter_iim_null_surrogates=0,
    repo_root=None,
):
    from impact_pipeline.run_synergy_ci import load_onsets, run_s_ci

    stage = str(hunter_stage or "build-campaign").strip().lower()
    if stage not in HUNTER_STAGES:
        raise ValueError(
            f"Unknown hunter stage '{hunter_stage}'. "
            f"Expected one of: {', '.join(HUNTER_STAGES)}."
        )
    campaign_dir = Path(
        hunter_campaign_dir if hunter_campaign_dir is not None else (cache_dir / "hunter_iim_campaign")
    ).resolve()

    if stage == "build-campaign":
        if mpc_metrics is not None and "IIM" not in set(mpc_metrics):
            raise ValueError(
                "Hunter mode distributes IIM only, but IIM is not in --mpc-metrics. "
                "Add IIM or use --execution-mode local for the other metrics."
            )
        onsets = None
        needs_onsets = (mpc_metrics is None) or bool({"RAM", "SRPI"} & set(mpc_metrics))
        if needs_onsets and bids_root is not None:
            discovered_subjects = []
            if subjects is not None:
                discovered_subjects = [str(s).replace("sub-", "").strip() for s in subjects if str(s).strip()]
            else:
                layout = BIDSLayout(bids_root, validate=False)
                discovered_subjects = sorted(layout.get(return_type='id', target='subject'))
            # Same condition as finalize (run_s_ci) so both select the same runs.
            onsets = {
                subj: {
                    ses: load_onsets(bids_root, subj, ses, condition=condition)
                    for ses in sessions
                }
                for subj in discovered_subjects
            }
        context = {
            "prep_out": str(prep_out),
            "bids_root": (None if bids_root is None else str(bids_root)),
            "figdir": str(figdir),
            "cache_dir": str(cache_dir),
            "out_dir": str(out),
            "atlas": str(atlas),
            "sessions": list(sessions),
            "condition": str(condition),
            "tr": (None if metric_tr is None else float(metric_tr)),
            "subjects": (None if subjects is None else list(subjects)),
            "mpc_metrics": (None if mpc_metrics is None else list(mpc_metrics)),
            "compute_ci": bool(compute_ci),
            # Finalize recomputes CI; it must use the reference of this build.
            "ci_reference": _portable_ci_reference(ci_reference),
            "dataset_id": str(dataset_id),
            "data_origin": str(data_origin),
            "dataset_role": str(dataset_role),
            "provenance_label": str(provenance_label),
            "report_doc": str(report_doc),
            "run_replication_flag": bool(run_replication_flag),
            "hardware_target": str(hardware_target),
            "pdi_params": pdi_params,
            "pdi_require_explicit_params": bool(pdi_require_explicit_params),
            "pdi_require_strict_baseline": bool(pdi_require_strict_baseline),
            "pdi_primary_endpoint": str(pdi_primary_endpoint),
            "nas_params": nas_params,
            "srpi_params": srpi_params,
            "srpi_require_explicit_params": bool(srpi_require_explicit_params),
            "iim_settings": _json_safe(dict(iim_settings or {})),
            "atlas_robustness": bool(atlas_robustness),
            "run_parameters": _json_safe(dict(run_parameters or {})),
        }
        manifest = prepare_hunter_campaign(
            data_dir=prep_out,
            atlas=atlas,
            sessions=sessions,
            condition=condition,
            stimulus_onsets=onsets,
            subjects=subjects,
            campaign_dir=campaign_dir,
            execution_profile=execution_profile,
            iim_bins=3,
            iim_lag_trs=1,
            iim_n_parts=iim_n_parts,
            iim_max_timepoints=iim_max_timepoints,
            iim_max_nodes=iim_max_nodes,
            iim_max_mechanism_size=iim_max_mechanism_size,
            iim_max_purview_size=iim_max_purview_size,
            hardware_target=hardware_target,
            step2_context=context,
            scheduler=hunter_scheduler,
            settings_overrides=hunter_settings_overrides,
            build_hardware_backend=build_hardware_backend,
            # K circular-shift surrogate runs per real run (IIM null calibration).
            iim_null_surrogates=int(hunter_iim_null_surrogates or 0),
            # Checkout the jobs run: --repo-root > IMPACT_REPO_ROOT > package
            # checkout > this script's directory (non-editable installs).
            repo_root=resolve_repo_root(repo_root, fallback=root),
        )
        sched = manifest.get("scheduler") or {}
        log.info(
            "Hunter campaign prepared: runs=%d phase1_tasks=%d cut_tasks=%d "
            "scheduler=%s shards_per_node=%s workers_per_task=%s dir=%s",
            len(manifest.get("runs", [])),
            len(manifest.get("phase1_tasks", [])),
            len(manifest.get("cut_tasks", [])),
            sched.get("scheduler"),
            sched.get("shards_per_node"),
            manifest["execution_profile"].get("hunter_phase1_workers_per_task"),
            campaign_dir,
        )
        log.info(
            "Submit on a Hunter login node with: bash %s",
            campaign_dir / sched.get("scheduler", "pbs") / "00_submit_all.sh",
        )
        return manifest

    if stage in {"phase1-shard", "cut-shard"}:
        if hunter_array_index is not None:
            run_packed_shard(
                campaign_dir,
                stage,
                int(hunter_array_index),
                # None: the packing the campaign was built with
                (
                    None
                    if hunter_shards_per_node is None
                    else int(hunter_shards_per_node)
                ),
                # The job's own target (Slurm phase-1 jobs run on CPU nodes).
                hardware_target=hardware_target,
            )
            return None
        if hunter_task_index is None:
            raise ValueError(
                "--hunter-task-index (or --hunter-array-index) is required for "
                f"hunter {stage}."
            )
        if stage == "phase1-shard":
            run_phase1_shard(
                campaign_dir, int(hunter_task_index), hardware_target=hardware_target
            )
        else:
            run_cut_shard(
                campaign_dir, int(hunter_task_index), hardware_target=hardware_target
            )
        return None
    if stage == "phase1-reduce":
        if hunter_run_index is None:
            raise ValueError("--hunter-run-index is required for hunter phase1-reduce.")
        run_phase1_reduce(campaign_dir, int(hunter_run_index))
        return None
    if stage == "cut-reduce":
        if hunter_run_index is None:
            raise ValueError("--hunter-run-index is required for hunter cut-reduce.")
        run_cut_reduce(campaign_dir, int(hunter_run_index))
        return None
    if stage == "phase1-reduce-all":
        run_phase1_reduce_all(campaign_dir)
        return None
    if stage == "cut-reduce-all":
        run_cut_reduce_all(campaign_dir)
        return None
    if stage == "reduce-all":
        run_reduce_all(campaign_dir)
        return None
    if stage == "status":
        status = campaign_status(campaign_dir)
        log.info("Hunter campaign status: %s", json.dumps(status["stages"]))
        return status

    # finalize-pipeline
    manifest = json.loads(
        (campaign_dir / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    ctx = manifest["step2_context"]
    ctx_dataset_id = str(ctx.get("dataset_id", dataset_id))
    ctx_cfg = DATASET_CONFIGS.get(ctx_dataset_id, cfg)
    ctx_modality = ctx_cfg["modality"] if ctx_cfg is not None else modality
    ctx_iim = dict(ctx.get("iim_settings") or {})

    def _ctx_iim(key, fallback):
        return ctx_iim.get(key, fallback)

    iim_precomputed_by_path = collect_iim_results_by_path(campaign_dir)
    # Per-run IIM with estimator/calibration provenance next to step-2 outputs.
    write_iim_results_table(
        campaign_dir, Path(ctx["cache_dir"]) / "hunter_iim_results.csv"
    )
    df, df_mean, _df_S, _df_CI, df_stats_by_theta = run_s_ci(
        prep_out=Path(ctx["prep_out"]),
        bids_root=(None if ctx["bids_root"] is None else Path(ctx["bids_root"])),
        figdir=Path(ctx["figdir"]),
        atlas=ctx["atlas"],
        sessions=tuple(ctx["sessions"]),
        thetas=np.arange(0.1, 1.0, 0.1),
        thetas_fine=np.arange(0.4, 0.81, 0.02),
        mpc_metrics=ctx["mpc_metrics"],
        compute_ci=bool(ctx["compute_ci"]),
        ci_reference=ctx.get("ci_reference", ci_reference),
        condition=ctx["condition"],
        tr=ctx["tr"],
        onsets=None,
        load_onsets_fn=load_onsets if ctx_modality in {"fmri", "eeg"} else None,
        iim_n_parts=_ctx_iim("iim_n_parts", iim_n_parts),
        iim_max_timepoints=_ctx_iim("iim_max_timepoints", iim_max_timepoints),
        iim_max_nodes=_ctx_iim("iim_max_nodes", iim_max_nodes),
        iim_max_mechanism_size=_ctx_iim(
            "iim_max_mechanism_size", iim_max_mechanism_size
        ),
        iim_max_purview_size=_ctx_iim("iim_max_purview_size", iim_max_purview_size),
        iim_parallel_workers=None,
        iim_memory_target_ratio=0.90,
        iim_worker_mem_gb_estimate=3.0,
        iim_cpu_oversub_factor=3.0,
        iim_enable_parallel=False,
        iim_checkpoint_dir=None,
        iim_resume_checkpoint=False,
        iim_checkpoint_every_cuts=1,
        iim_progress_log_every_cuts=1,
        iim_use_shared_memory=False,
        iim_phase1_parallel_workers=None,
        iim_phase1_chunk_size=8,
        iim_phase1_shared_memory=False,
        pdi_params=ctx["pdi_params"],
        pdi_require_explicit_params=bool(ctx["pdi_require_explicit_params"]),
        pdi_require_strict_baseline=bool(ctx["pdi_require_strict_baseline"]),
        pdi_primary_endpoint=ctx["pdi_primary_endpoint"],
        nas_params=ctx["nas_params"],
        srpi_params=ctx["srpi_params"],
        srpi_require_explicit_params=bool(ctx["srpi_require_explicit_params"]),
        subjects=ctx["subjects"],
        iim_precomputed_by_path=iim_precomputed_by_path,
        dataset_id=ctx_dataset_id,
        data_origin=ctx.get("data_origin", REAL_DATA_ORIGIN),
        dataset_role=ctx.get("dataset_role"),
        provenance_label=ctx.get("provenance_label"),
        modality=ctx_modality,
        # The finalize job's own target (e.g. 'cpu' on a CPU/pre queue).
        hardware_target=hardware_target,
    )
    df, df_mean = _apply_metric_subset(
        df,
        df_mean,
        mpc_metrics=ctx["mpc_metrics"],
        compute_ci=bool(ctx["compute_ci"]),
    )
    if "subject" in df.columns:
        df["subject"] = df["subject"].astype(str)
    if "subject" in df_mean.columns:
        df_mean["subject"] = df_mean["subject"].astype(str)
    _persist_step2_outputs(
        Path(ctx["cache_dir"]),
        df,
        df_mean,
        df_stats_by_theta,
        persist_primary=True,
    )
    provenance = resolve_dataset_provenance(
        repo_root=root,
        out_dir=ctx["out_dir"],
        dataset_id=ctx_dataset_id,
        data_origin=ctx.get("data_origin", REAL_DATA_ORIGIN),
    )
    df = _ensure_provenance_columns(df, provenance)
    df_mean = _ensure_provenance_columns(df_mean, provenance)
    manifest_kwargs = dict(
        cache_dir=Path(ctx["cache_dir"]),
        provenance=provenance,
        modality=ctx_modality,
        bids_root=(None if ctx["bids_root"] is None else Path(ctx["bids_root"])),
        execution_mode="hunter",
        atlas=ctx["atlas"],
        sessions=tuple(ctx["sessions"]),
        condition=ctx["condition"],
        parameters=ctx.get("run_parameters") or {},
        hardware={
            "finalize_target": str(hardware_target),
            "campaign_target": ctx.get("hardware_target"),
            "campaign_build": manifest.get("build_hardware_backend"),
        },
        extra={
            "hunter_campaign": {
                "campaign_dir": str(campaign_dir),
                "scheduler": manifest.get("scheduler"),
                "execution_profile": manifest.get("execution_profile"),
                "kernel_code_version": manifest.get("kernel_code_version"),
            }
        },
    )
    _write_run_provenance_manifest(status="finalize_started", **manifest_kwargs)
    if is_test_object_origin(provenance.data_origin):
        _export_test_object_metric_bank(
            df=df,
            df_mean=df_mean,
            cache_dir=Path(ctx["cache_dir"]),
            provenance=provenance,
            modality=ctx_modality,
            atlas=ctx["atlas"],
            sessions=tuple(ctx["sessions"]),
            condition=ctx["condition"],
        )
    _postprocess_after_step2(
        df=df,
        df_mean=df_mean,
        prep_out=Path(ctx["prep_out"]),
        bids_root=(None if ctx["bids_root"] is None else Path(ctx["bids_root"])),
        figdir=Path(ctx["figdir"]),
        atlas=ctx["atlas"],
        sessions=tuple(ctx["sessions"]),
        compute_ci=bool(ctx["compute_ci"]),
        modality=ctx_modality,
        dataset_id=ctx_dataset_id,
        cfg=ctx_cfg,
        subjects=ctx["subjects"],
        out=Path(ctx["out_dir"]),
        report_doc=ctx["report_doc"],
        pdi_params=ctx["pdi_params"],
        pdi_require_explicit_params=bool(ctx["pdi_require_explicit_params"]),
        pdi_require_strict_baseline=bool(ctx["pdi_require_strict_baseline"]),
        pdi_primary_endpoint=ctx["pdi_primary_endpoint"],
        nas_params=ctx["nas_params"],
        srpi_params=ctx["srpi_params"],
        srpi_require_explicit_params=bool(ctx["srpi_require_explicit_params"]),
        run_replication_flag=bool(ctx["run_replication_flag"]),
        df_stats_by_theta=df_stats_by_theta,
        condition=ctx["condition"],
        robustness_context={
            "enabled": bool(ctx.get("atlas_robustness", True)),
            "execution_mode": "hunter",
            "condition": ctx["condition"],
            "tr": ctx["tr"],
            "iim_settings": ctx_iim,
            "hardware_target": hardware_target,
        },
    )
    summarize_campaign_timing(campaign_dir)
    _write_run_provenance_manifest(status="completed", **manifest_kwargs)
    return None


def _normalize_subjects(subjects):
    """Accept subject IDs with or without the 'sub-' prefix everywhere."""
    if subjects is None:
        return None
    out = []
    for s in subjects:
        label = str(s).strip().replace("sub-", "", 1)
        if label and label not in out:
            out.append(label)
    return out or None


def main(
    out_dir,
    subjects=None,
    reuse_step2=False,
    mpc_metrics=None,
    compute_ci=True,
    ci_reference=None,
    dataset_id="ds003171",
    bids_root_override=None,
    run_fmriprep=False,
    run_preprocessing_flag=False,
    run_replication_flag=False,
    atlas_override=None,
    sessions_override=None,
    condition_override=None,
    tr_override=None,
    eeg_target_sfreq=250.0,
    eeg_l_freq=0.5,
    eeg_h_freq=45.0,
    eeg_max_duration_sec=120.0,
    iim_n_parts_override=None,
    iim_max_timepoints_override=None,
    iim_max_nodes_override=None,
    iim_max_mechanism_size_override=None,
    iim_max_purview_size_override=None,
    iim_parallel_workers_override=None,
    iim_memory_target_ratio_override=None,
    iim_worker_mem_gb_estimate_override=None,
    iim_cpu_oversub_factor_override=None,
    iim_phase1_parallel_workers_override=None,
    iim_phase1_chunk_size_override=None,
    iim_checkpoint_dir_override=None,
    disable_iim_checkpoint_resume=False,
    iim_checkpoint_every_cuts=1,
    iim_progress_log_every_cuts=1,
    disable_iim_shared_memory=False,
    disable_iim_phase1_shared_memory=False,
    disable_iim_parallel=False,
    execution_mode="local",
    hardware_target="cpu",
    data_origin=REAL_DATA_ORIGIN,
    hunter_stage=None,
    hunter_campaign_dir=None,
    hunter_task_index=None,
    hunter_run_index=None,
    assume_tr=None,
    fmriprep_dir=None,
    hunter_scheduler=None,
    hunter_array_index=None,
    hunter_shards_per_node=None,
    hunter_phase1_shards_per_run=None,
    hunter_cut_shards_per_run=None,
    hunter_workers_per_task=None,
    atlas_robustness=True,
    cli_argv=None,
    hunter_iim_null_surrogates=None,
    repo_root=None,
):
    from impact_pipeline.run_synergy_ci import load_onsets, run_s_ci

    _assert_expected_runtime_env()
    subjects = _normalize_subjects(subjects)

    catalog_entry = get_report_dataset(dataset_id)
    cfg = DATASET_CONFIGS.get(dataset_id)
    if cfg is None:
        if catalog_entry is not None:
            raise ValueError(
                f"Dataset '{dataset_id}' is available in the local dataset catalog "
                f"('{catalog_entry.title}') but does not have full pipeline mappings. "
                "Use compatible preprocessed outputs for modular analyses or add "
                "dataset-specific support before running the full pipeline."
            )
        raise ValueError(
            f"Unknown dataset_id='{dataset_id}'. "
            f"Known datasets: {sorted(DATASET_CONFIGS.keys())}"
        )
    if (
        catalog_entry is not None
        and not bool(catalog_entry.pipeline_ready)
        and (bool(run_preprocessing_flag) or bool(run_fmriprep))
    ):
        raise ValueError(
            f"Dataset '{dataset_id}' is cataloged locally ('{catalog_entry.title}') but is not "
            "pipeline-enabled for raw preprocessing/fMRIPrep in this repository yet. "
            "Reuse existing compatible preprocessed outputs for modular analyses, or add "
            "dataset-specific preprocessing/session support first."
        )
    modality = cfg["modality"]
    execution_profile = get_execution_profile(execution_mode)
    hunter_mode = execution_profile.name == "hunter"
    hunter_stage_key = str(hunter_stage or "build-campaign").strip().lower()
    if hunter_mode and hunter_stage_key not in HUNTER_STAGES:
        raise ValueError(
            f"Unknown hunter stage '{hunter_stage}'. "
            f"Expected one of: {', '.join(HUNTER_STAGES)}."
        )
    if not hunter_mode and hunter_stage is not None:
        log.warning(
            "--hunter-stage %s is ignored in execution mode '%s'.",
            hunter_stage,
            execution_mode,
        )
    try:
        requested_hardware_target = normalize_hardware_target(hardware_target)
    except HardwareBackendError as exc:
        raise RuntimeError(f"Hardware target unavailable: {exc}") from exc
    try:
        hardware_backend = configure_process_for_hardware(requested_hardware_target)
    except HardwareBackendError as exc:
        if not (hunter_mode and hunter_stage_key == "build-campaign"):
            raise RuntimeError(f"Hardware target unavailable: {exc}") from exc
        # Login nodes have no APU: prepare the campaign on the CPU (numerically
        # identical TPM estimation) and let the compute jobs request the target.
        log.warning(
            "Hardware target '%s' is not available on this build host (%s). The "
            "campaign is prepared on the CPU; its compute jobs will request '%s'.",
            hardware_target,
            exc,
            requested_hardware_target,
        )
        hardware_backend = configure_process_for_hardware("cpu")
    log.info("Hardware target: %s", backend_summary(hardware_backend))

    provenance = resolve_dataset_provenance(
        repo_root=root,
        out_dir=out_dir,
        dataset_id=dataset_id,
        data_origin=data_origin,
    )
    out = provenance.effective_out_dir
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / 'pipe_figures'
    figdir.mkdir(exist_ok=True)
    cache_dir = out / 'cache'
    cache_dir.mkdir(exist_ok=True)

    prep_out     = out / 'preprocessed'
    workdir      = out / 'work'
    bids_root = (
        Path(bids_root_override)
        if bids_root_override is not None
        else _first_existing_path(*cfg["bids_candidates"])
    )
    if hunter_mode and hunter_stage_key != "build-campaign":
        _run_hunter_stage(
            hunter_stage=hunter_stage_key,
            hunter_campaign_dir=hunter_campaign_dir,
            hunter_task_index=hunter_task_index,
            hunter_run_index=hunter_run_index,
            execution_profile=execution_profile,
            prep_out=prep_out,
            bids_root=bids_root,
            figdir=figdir,
            cache_dir=cache_dir,
            atlas=atlas_override or cfg["atlas"],
            sessions=(
                tuple(sessions_override)
                if sessions_override
                else tuple(cfg["sessions"])
            ),
            condition=condition_override or cfg["condition"],
            metric_tr=tr_override,
            subjects=subjects,
            mpc_metrics=mpc_metrics,
            compute_ci=compute_ci,
            iim_n_parts=iim_n_parts_override,
            iim_max_timepoints=iim_max_timepoints_override,
            iim_max_nodes=iim_max_nodes_override,
            iim_max_mechanism_size=iim_max_mechanism_size_override,
            iim_max_purview_size=iim_max_purview_size_override,
            pdi_params=dict(cfg.get("pdi_params", {})),
            pdi_require_explicit_params=bool(
                cfg.get("pdi_require_explicit_params", True)
            ),
            pdi_require_strict_baseline=bool(
                cfg.get("pdi_require_strict_baseline", True)
            ),
            pdi_primary_endpoint=str(cfg.get("pdi_primary_endpoint", "anchor")),
            nas_params=dict(cfg.get("nas_params", {})),
            srpi_params=dict(cfg.get("srpi_params", {})),
            srpi_require_explicit_params=bool(
                cfg.get("srpi_require_explicit_params", True)
            ),
            out=out,
            modality=modality,
            dataset_id=dataset_id,
            data_origin=provenance.data_origin,
            dataset_role=provenance.dataset_role,
            provenance_label=provenance.provenance_label,
            cfg=cfg,
            report_doc=f"IMPaCT_Empirical_Validation_{dataset_id}.docx",
            run_replication_flag=run_replication_flag,
            hardware_target=requested_hardware_target,
            hunter_array_index=hunter_array_index,
            hunter_shards_per_node=hunter_shards_per_node,
            ci_reference=ci_reference,
        )
        return
    if not bids_root.exists():
        raise FileNotFoundError(
            f"Missing {dataset_id} dataset at '{bids_root}'. "
            "Provide --bids-root explicitly or download the dataset first."
        )
    assert_origin_matches_dataset(bids_root, provenance.data_origin)
    replication_enabled = bool(run_replication_flag) and bool(
        cfg.get("supports_replication", False)
    )
    if replication_enabled:
        # Fail now rather than after hours of metric computation (step 7).
        _check_replication_inputs(_resolve_replication_root())

    atlas = atlas_override or cfg["atlas"]
    sessions = tuple(sessions_override) if sessions_override else tuple(cfg["sessions"])
    condition = condition_override or cfg["condition"]
    metric_tr = tr_override
    if metric_tr is None and modality == "eeg":
        if eeg_target_sfreq is None or float(eeg_target_sfreq) <= 0:
            raise ValueError("eeg_target_sfreq must be > 0 for EEG metric computation.")
        metric_tr = 1.0 / float(eeg_target_sfreq)
    def _opt_int(v):
        return None if v is None else int(v)

    iim_n_parts = (
        _opt_int(iim_n_parts_override)
        if iim_n_parts_override is not None
        else _opt_int(cfg.get("iim_n_parts", None))
    )
    iim_max_nodes = (
        _opt_int(iim_max_nodes_override)
        if iim_max_nodes_override is not None
        else _opt_int(cfg.get("iim_max_nodes", None))
    )
    iim_max_mechanism_size = (
        _opt_int(iim_max_mechanism_size_override)
        if iim_max_mechanism_size_override is not None
        else _opt_int(cfg.get("iim_max_mechanism_size", None))
    )
    iim_max_purview_size = (
        _opt_int(iim_max_purview_size_override)
        if iim_max_purview_size_override is not None
        else _opt_int(cfg.get("iim_max_purview_size", None))
    )
    iim_parallel_workers = (
        _opt_int(iim_parallel_workers_override)
        if iim_parallel_workers_override is not None
        else _opt_int(cfg.get("iim_parallel_workers", None))
    )
    iim_memory_target_ratio = (
        float(iim_memory_target_ratio_override)
        if iim_memory_target_ratio_override is not None
        else float(cfg.get("iim_memory_target_ratio", 0.90))
    )
    iim_worker_mem_gb_estimate = (
        float(iim_worker_mem_gb_estimate_override)
        if iim_worker_mem_gb_estimate_override is not None
        else float(cfg.get("iim_worker_mem_gb_estimate", 3.0))
    )
    iim_cpu_oversub_factor = (
        float(iim_cpu_oversub_factor_override)
        if iim_cpu_oversub_factor_override is not None
        else float(cfg.get("iim_cpu_oversub_factor", 3.0))
    )
    iim_phase1_parallel_workers = (
        _opt_int(iim_phase1_parallel_workers_override)
        if iim_phase1_parallel_workers_override is not None
        else _opt_int(cfg.get("iim_phase1_parallel_workers", None))
    )
    iim_phase1_chunk_size = (
        _opt_int(iim_phase1_chunk_size_override)
        if iim_phase1_chunk_size_override is not None
        else _opt_int(cfg.get("iim_phase1_chunk_size", 8))
    )
    if iim_phase1_chunk_size is None:
        iim_phase1_chunk_size = 8
    if int(iim_phase1_chunk_size) < 1:
        raise ValueError("iim_phase1_chunk_size must be >= 1")
    iim_enable_parallel = not bool(disable_iim_parallel)
    iim_use_shared_memory = not bool(disable_iim_shared_memory)
    iim_phase1_shared_memory = (
        bool(cfg.get("iim_phase1_shared_memory", True))
        and not bool(disable_iim_phase1_shared_memory)
    )
    pdi_params = dict(cfg.get("pdi_params", {}))
    pdi_require_explicit_params = bool(cfg.get("pdi_require_explicit_params", True))
    pdi_require_strict_baseline = bool(cfg.get("pdi_require_strict_baseline", True))
    pdi_primary_endpoint = str(cfg.get("pdi_primary_endpoint", "anchor"))
    nas_params = dict(cfg.get("nas_params", {}))
    srpi_params = dict(cfg.get("srpi_params", {}))
    srpi_require_explicit_params = bool(cfg.get("srpi_require_explicit_params", True))
    iim_resume_checkpoint = not bool(disable_iim_checkpoint_resume)
    iim_checkpoint_dir = (
        str(Path(iim_checkpoint_dir_override))
        if iim_checkpoint_dir_override is not None
        else str(cache_dir / "iim_checkpoints")
    )
    if int(iim_checkpoint_every_cuts) < 1:
        raise ValueError("iim_checkpoint_every_cuts must be >= 1")
    if int(iim_progress_log_every_cuts) < 1:
        raise ValueError("iim_progress_log_every_cuts must be >= 1")
    iim_max_timepoints = (
        None
        if iim_max_timepoints_override is None and cfg.get("iim_max_timepoints", None) is None
        else int(iim_max_timepoints_override if iim_max_timepoints_override is not None else cfg.get("iim_max_timepoints"))
    )
    report_doc = f"IMPaCT_Empirical_Validation_{dataset_id}.docx"
    cuts_mode = "all" if iim_n_parts is None else str(iim_n_parts)
    log.info(
        "IIM configuration: cuts=%s, max_nodes=%s, max_mechanism_size=%s, max_purview_size=%s, "
        "max_timepoints=%s, parallel=%s, workers=%s, mem_target=%.2f, worker_mem_est=%.2fGB",
        cuts_mode,
        "all" if iim_max_nodes is None else iim_max_nodes,
        "all" if iim_max_mechanism_size is None else iim_max_mechanism_size,
        "all" if iim_max_purview_size is None else iim_max_purview_size,
        "all" if iim_max_timepoints is None else iim_max_timepoints,
        iim_enable_parallel,
        "auto" if iim_parallel_workers is None else iim_parallel_workers,
        iim_memory_target_ratio,
        iim_worker_mem_gb_estimate,
    )
    log.info(
        "IIM planner oversub factor: %.2f (soft worker cap = cpu_count * factor)",
        iim_cpu_oversub_factor,
    )
    log.info(
        "IIM intra-task phase config: workers=%s chunk_size=%d shared_memory=%s",
        "off/auto" if iim_phase1_parallel_workers is None else str(int(iim_phase1_parallel_workers)),
        int(iim_phase1_chunk_size),
        bool(iim_phase1_shared_memory),
    )
    log.info(
        "IIM checkpoint config: dir=%s, resume=%s, checkpoint_every_cuts=%d, progress_every_cuts=%d, use_shared_memory=%s",
        iim_checkpoint_dir,
        iim_resume_checkpoint,
        int(iim_checkpoint_every_cuts),
        int(iim_progress_log_every_cuts),
        iim_use_shared_memory,
    )
    iim_settings = {
        "iim_n_parts": iim_n_parts,
        "iim_max_timepoints": iim_max_timepoints,
        "iim_max_nodes": iim_max_nodes,
        "iim_max_mechanism_size": iim_max_mechanism_size,
        "iim_max_purview_size": iim_max_purview_size,
        "iim_parallel_workers": iim_parallel_workers,
        "iim_memory_target_ratio": iim_memory_target_ratio,
        "iim_worker_mem_gb_estimate": iim_worker_mem_gb_estimate,
        "iim_cpu_oversub_factor": iim_cpu_oversub_factor,
        "iim_enable_parallel": iim_enable_parallel,
        "iim_checkpoint_dir": iim_checkpoint_dir,
        "iim_resume_checkpoint": iim_resume_checkpoint,
        "iim_checkpoint_every_cuts": int(iim_checkpoint_every_cuts),
        "iim_progress_log_every_cuts": int(iim_progress_log_every_cuts),
        "iim_use_shared_memory": iim_use_shared_memory,
        "iim_phase1_parallel_workers": iim_phase1_parallel_workers,
        "iim_phase1_chunk_size": int(iim_phase1_chunk_size),
        "iim_phase1_shared_memory": bool(iim_phase1_shared_memory),
    }
    fmriprep_out = (
        _resolve_fmriprep_dir(fmriprep_dir, bids_root, out)
        if modality == "fmri"
        else None
    )
    run_parameters = {
        "cli_argv": (None if cli_argv is None else list(cli_argv)),
        "dataset_id": dataset_id,
        "data_origin": provenance.data_origin,
        "subjects": subjects,
        "mpc_metrics": mpc_metrics,
        "compute_ci": bool(compute_ci),
        "ci_reference": _portable_ci_reference(ci_reference),
        "reuse_step2": bool(reuse_step2),
        "atlas": atlas,
        "sessions": sessions,
        "condition": condition,
        "tr_override": tr_override,
        "metric_tr": metric_tr,
        "assume_tr": assume_tr,
        "run_fmriprep": bool(run_fmriprep),
        "run_preprocessing": bool(run_preprocessing_flag),
        "run_replication": bool(run_replication_flag),
        "fmriprep_dir": (None if fmriprep_out is None else str(fmriprep_out)),
        "eeg": {
            "target_sfreq": eeg_target_sfreq,
            "l_freq": eeg_l_freq,
            "h_freq": eeg_h_freq,
            "max_duration_sec": eeg_max_duration_sec,
            "session_rules": cfg.get("eeg_session_rules"),
            "rest_rules": cfg.get("eeg_rest_rules"),
        },
        "iim": iim_settings,
        "pdi_params": pdi_params,
        "pdi_require_explicit_params": pdi_require_explicit_params,
        "pdi_require_strict_baseline": pdi_require_strict_baseline,
        "pdi_primary_endpoint": pdi_primary_endpoint,
        "nas_params": nas_params,
        "srpi_params": srpi_params,
        "srpi_require_explicit_params": srpi_require_explicit_params,
        "execution_mode": execution_mode,
        "hardware_target_requested": requested_hardware_target,
        "atlas_robustness": bool(atlas_robustness),
        "hunter": (
            None
            if not hunter_mode
            else {
                "stage": hunter_stage_key,
                "scheduler": hunter_scheduler,
                "phase1_shards_per_run": hunter_phase1_shards_per_run,
                "cut_shards_per_run": hunter_cut_shards_per_run,
                "workers_per_task": hunter_workers_per_task,
                "shards_per_node": hunter_shards_per_node,
                "iim_null_surrogates": int(hunter_iim_null_surrogates or 0),
                "repo_root": (None if repo_root is None else str(repo_root)),
            }
        ),
    }
    manifest_kwargs = dict(
        cache_dir=cache_dir,
        provenance=provenance,
        modality=modality,
        bids_root=bids_root,
        execution_mode=execution_mode,
        atlas=atlas,
        sessions=sessions,
        condition=condition,
        parameters=run_parameters,
        hardware=hardware_backend,
    )
    _write_run_provenance_manifest(status="started", **manifest_kwargs)

    # 0. RUN FMRIPrep ON MISSING SUBJECTS
    if run_fmriprep:
        if not cfg["supports_fmriprep"]:
            raise ValueError(
                f"fMRIPrep is not applicable to dataset '{dataset_id}' "
                f"(modality={modality})."
            )
        ensure_fmriprep(
            bids_dir=str(bids_root),
            fmriprep_out=str(fmriprep_out),
            work_dir=str(workdir),
            fs_license=str(_resolve_freesurfer_license_path()),
            freesurf_out=str(Path(fmriprep_out).parent / "freesurfer"),
            subjects=subjects,
        )

    # 1. PREPROCESSING
    if run_preprocessing_flag:
        if modality == "fmri":
            from impact_pipeline.preprocessing import run_preprocessing
            log.info(
                "1/9 Preprocessing %s (fMRI) from fMRIPrep derivatives at %s",
                dataset_id,
                fmriprep_out,
            )
            if not _has_fmriprep_subjects(Path(fmriprep_out)):
                raise FileNotFoundError(
                    f"No fMRIPrep derivatives (sub-*/) at '{fmriprep_out}'. Run "
                    f"fMRIPrep into '{canonical_fmriprep_dir(bids_root)}' (canonical), "
                    "use --run-fmriprep, or point --fmriprep-dir at existing "
                    "derivatives."
                )
            fmri_summary = run_preprocessing(
                bids_root=str(bids_root),
                fmriprep_deriv=str(fmriprep_out),
                out_root=str(prep_out),
                subjects=subjects,
                assume_tr=assume_tr,
                dataset_id=dataset_id,
            )
            write_json(
                cache_dir / "preprocessing_fmri_summary.json", _json_safe(fmri_summary)
            )
            log.info("fMRI preprocessing summary: %s", fmri_summary.get("summary"))
        elif modality == "eeg":
            log.info("1/9 Preprocessing %s (EEG)", dataset_id)
            eeg_summary = run_preprocessing_eeg(
                bids_root=str(bids_root),
                out_root=str(prep_out),
                subjects=subjects,
                session_rules=cfg.get("eeg_session_rules"),
                condition_label=condition,
                atlas_key=atlas,
                target_sfreq=float(eeg_target_sfreq),
                l_freq=float(eeg_l_freq) if eeg_l_freq is not None else None,
                h_freq=float(eeg_h_freq) if eeg_h_freq is not None else None,
                max_duration_sec=(
                    float(eeg_max_duration_sec)
                    if eeg_max_duration_sec is not None
                    else None
                ),
                rest_rules=cfg.get("eeg_rest_rules"),
            )
            _write_eeg_preprocessing_summary(cache_dir, eeg_summary)
            summ = eeg_summary.get("summary", {})
            log.info(
                "EEG preprocessing summary: subjects_requested=%s processed=%s skipped=%s missing_records=%s written_runs=%s",
                summ.get("subjects_requested", "na"),
                summ.get("subjects_processed", "na"),
                summ.get("subjects_skipped", "na"),
                summ.get("missing_file_records", "na"),
                summ.get("written_runs", "na"),
            )
        else:
            raise ValueError(f"Unsupported modality '{modality}'")

    # Step 2 depends on preprocessing outputs even when Step 1 is skipped.
    if not prep_out.exists() or not any(prep_out.iterdir()):
        raise FileNotFoundError(
            f"Missing preprocessing outputs at '{prep_out}'. "
            "Run with --run-preprocessing (and ensure required derivatives are available) "
            "or provide an out-dir that already contains a populated 'preprocessed' folder."
        )

    if hunter_mode:
        campaign_manifest = _run_hunter_stage(
            hunter_stage=hunter_stage_key,
            hunter_campaign_dir=hunter_campaign_dir,
            hunter_task_index=hunter_task_index,
            hunter_run_index=hunter_run_index,
            execution_profile=execution_profile,
            prep_out=prep_out,
            bids_root=bids_root,
            figdir=figdir,
            cache_dir=cache_dir,
            atlas=atlas,
            sessions=sessions,
            condition=condition,
            metric_tr=metric_tr,
            subjects=subjects,
            mpc_metrics=mpc_metrics,
            compute_ci=compute_ci,
            iim_n_parts=iim_n_parts,
            iim_max_timepoints=iim_max_timepoints,
            iim_max_nodes=iim_max_nodes,
            iim_max_mechanism_size=iim_max_mechanism_size,
            iim_max_purview_size=iim_max_purview_size,
            pdi_params=pdi_params,
            pdi_require_explicit_params=pdi_require_explicit_params,
            pdi_require_strict_baseline=pdi_require_strict_baseline,
            pdi_primary_endpoint=pdi_primary_endpoint,
            nas_params=nas_params,
            srpi_params=srpi_params,
            srpi_require_explicit_params=srpi_require_explicit_params,
            out=out,
            modality=modality,
            dataset_id=dataset_id,
            data_origin=provenance.data_origin,
            dataset_role=provenance.dataset_role,
            provenance_label=provenance.provenance_label,
            cfg=cfg,
            report_doc=report_doc,
            run_replication_flag=run_replication_flag,
            hardware_target=requested_hardware_target,
            hunter_scheduler=hunter_scheduler,
            hunter_settings_overrides={
                "phase1_shards_per_run": hunter_phase1_shards_per_run,
                "cut_shards_per_run": hunter_cut_shards_per_run,
                "workers_per_task": hunter_workers_per_task,
                "shards_per_node": hunter_shards_per_node,
            },
            build_hardware_backend=hardware_backend,
            iim_settings=iim_settings,
            run_parameters=run_parameters,
            atlas_robustness=atlas_robustness,
            ci_reference=ci_reference,
            hunter_iim_null_surrogates=int(hunter_iim_null_surrogates or 0),
            repo_root=repo_root,
        )
        _write_run_provenance_manifest(
            status="hunter_campaign_built",
            extra={
                "hunter_campaign": {
                    "campaign_dir": (campaign_manifest or {}).get("campaign_dir"),
                    "scheduler": (campaign_manifest or {}).get("scheduler"),
                    "n_runs": len((campaign_manifest or {}).get("runs", [])),
                }
            },
            **manifest_kwargs,
        )
        return

    # 2. SYNERGY & CI
    thetas      = np.arange(0.1, 1.0, 0.1)
    thetas_fine = np.arange(0.4, 0.81, 0.02)
    step2_df_path = cache_dir / "step2_df.csv"
    step2_df_mean_path = cache_dir / "step2_df_mean.csv"
    step2_theta_stats_path = cache_dir / "step2_theta_stats.csv"
    if reuse_step2:
        missing = [
            str(p)
            for p in (step2_df_path, step2_df_mean_path, step2_theta_stats_path)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Cannot reuse step 2 because cache files are missing: "
                + ", ".join(missing)
            )
        log.info("2/9 Reusing cached step-2 metric tables")
        df = pd.read_csv(step2_df_path)
        df_mean = pd.read_csv(step2_df_mean_path)
        if 'subject' in df.columns:
            df['subject'] = df['subject'].astype(str)
        if 'subject' in df_mean.columns:
            df_mean['subject'] = df_mean['subject'].astype(str)
        df, df_mean = _apply_metric_subset(
            df,
            df_mean,
            mpc_metrics=mpc_metrics,
            compute_ci=compute_ci,
        )
        df = _ensure_provenance_columns(df, provenance)
        df_mean = _ensure_provenance_columns(df_mean, provenance)
        df_stats_by_theta = pd.read_csv(step2_theta_stats_path).set_index('theta')
        df_S = df_mean.pivot(index='subject', columns='session', values='S')
        df_CI = df_mean.pivot(index='subject', columns='session', values='CI') if 'CI' in df_mean.columns else None
    else:
        df, df_mean, df_S, df_CI, df_stats_by_theta = run_s_ci(
            prep_out=prep_out,
            bids_root=bids_root,
            figdir=figdir,
            atlas=atlas,
            sessions=sessions,
            thetas=thetas,
            thetas_fine=thetas_fine,
            mpc_metrics=mpc_metrics,
            compute_ci=compute_ci,
            ci_reference=ci_reference,
            condition=condition,
            tr=metric_tr,
            onsets=None,
            load_onsets_fn=load_onsets if modality in {"fmri", "eeg"} else None,
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
            iim_checkpoint_every_cuts=int(iim_checkpoint_every_cuts),
            iim_progress_log_every_cuts=int(iim_progress_log_every_cuts),
            iim_use_shared_memory=iim_use_shared_memory,
            iim_phase1_parallel_workers=iim_phase1_parallel_workers,
            iim_phase1_chunk_size=int(iim_phase1_chunk_size),
            iim_phase1_shared_memory=bool(iim_phase1_shared_memory),
            pdi_params=pdi_params,
            pdi_require_explicit_params=pdi_require_explicit_params,
            pdi_require_strict_baseline=pdi_require_strict_baseline,
            pdi_primary_endpoint=pdi_primary_endpoint,
            nas_params=nas_params,
            srpi_params=srpi_params,
            srpi_require_explicit_params=srpi_require_explicit_params,
            subjects=subjects,
            dataset_id=dataset_id,
            data_origin=provenance.data_origin,
            dataset_role=provenance.dataset_role,
            provenance_label=provenance.provenance_label,
            modality=modality,
            hardware_target=hardware_backend.requested,
        )
        df, df_mean = _apply_metric_subset(
            df,
            df_mean,
            mpc_metrics=mpc_metrics,
            compute_ci=compute_ci,
        )
        df = _ensure_provenance_columns(df, provenance)
        df_mean = _ensure_provenance_columns(df_mean, provenance)
        if 'subject' in df.columns:
            df['subject'] = df['subject'].astype(str)
        if 'subject' in df_mean.columns:
            df_mean['subject'] = df_mean['subject'].astype(str)
        df_S = df_mean.pivot(index='subject', columns='session', values='S')
        df_CI = df_mean.pivot(index='subject', columns='session', values='CI') if 'CI' in df_mean.columns else None
        df.to_csv(step2_df_path, index=False)
        df_mean.to_csv(step2_df_mean_path, index=False)
        df_stats_by_theta.reset_index().to_csv(step2_theta_stats_path, index=False)

    _persist_step2_outputs(
        cache_dir,
        df,
        df_mean,
        df_stats_by_theta,
        persist_primary=not bool(reuse_step2),
    )
    if is_test_object_origin(provenance.data_origin):
        _export_test_object_metric_bank(
            df=df,
            df_mean=df_mean,
            cache_dir=cache_dir,
            provenance=provenance,
            modality=modality,
            atlas=atlas,
            sessions=sessions,
            condition=condition,
        )
    _postprocess_after_step2(
        df=df,
        df_mean=df_mean,
        prep_out=prep_out,
        bids_root=bids_root,
        figdir=figdir,
        atlas=atlas,
        sessions=sessions,
        compute_ci=compute_ci,
        modality=modality,
        dataset_id=dataset_id,
        cfg=cfg,
        subjects=subjects,
        out=out,
        report_doc=report_doc,
        pdi_params=pdi_params,
        pdi_require_explicit_params=pdi_require_explicit_params,
        pdi_require_strict_baseline=pdi_require_strict_baseline,
        pdi_primary_endpoint=pdi_primary_endpoint,
        nas_params=nas_params,
        srpi_params=srpi_params,
        srpi_require_explicit_params=srpi_require_explicit_params,
        run_replication_flag=run_replication_flag,
        df_stats_by_theta=df_stats_by_theta,
        condition=condition,
        robustness_context={
            "enabled": bool(atlas_robustness),
            "execution_mode": execution_mode,
            "condition": condition,
            "tr": metric_tr,
            "iim_settings": iim_settings,
            "hardware_target": requested_hardware_target,
        },
    )
    _write_run_provenance_manifest(status="completed", **manifest_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run IMPaCT pipeline")
    parser.add_argument('--out-dir', default='outputs/scratch',
                        help="Root folder for outputs. For non-ds003171 datasets, a dataset subfolder is auto-created.")
    parser.add_argument(
        '--data-origin',
        default=REAL_DATA_ORIGIN,
        choices=('real', 'dummy'),
        help=(
            "Mark the dataset as real study data or synthetic validation data. "
            "Synthetic validation runs are stored separately."
        ),
    )
    parser.add_argument(
        '--dataset-id',
        default='ds003171',
        help=f"Dataset identifier. Known: {', '.join(sorted(DATASET_CONFIGS.keys()))}",
    )
    parser.add_argument(
        '--bids-root',
        default=None,
        help="Optional explicit BIDS root override.",
    )
    parser.add_argument(
        '--subjects', nargs='+',
        help="Optional subject IDs to restrict preprocessing and (if enabled) fMRIPrep."
    )
    parser.add_argument(
        '--run-fmriprep',
        action='store_true',
        help="Run fMRIPrep step (only applicable to fMRI datasets).",
    )
    parser.add_argument(
        '--run-preprocessing',
        action='store_true',
        help="Run preprocessing step for the selected dataset modality.",
    )
    parser.add_argument(
        '--run-replication',
        action='store_true',
        help="Run replication step if configured for selected dataset.",
    )
    parser.add_argument(
        '--reuse-step2',
        action='store_true',
        help="Reuse cached step-2 tables from <out-dir>/cache and skip recomputing core metrics."
    )
    parser.add_argument(
        '--execution-mode',
        default='local',
        choices=('local', 'hunter'),
        help="Execution backend. 'local' keeps the current workstation pipeline; 'hunter' uses distributed IIM campaign stages.",
    )
    parser.add_argument(
        '--hardware-target',
        default='cpu',
        choices=('cpu', 'auto', 'gpu', 'hunter-apu'),
        help=(
            "Compute hardware for MPC kernels. "
            "'cpu' uses NumPy/SciPy, 'auto' uses CuPy when available and otherwise CPU, "
            "'gpu' requires a visible CuPy GPU, and 'hunter-apu' requires ROCm/HIP CuPy."
        ),
    )
    parser.add_argument(
        "--hunter-stage",
        default=None,
        choices=HUNTER_STAGES,
        help=(
            "Hunter stage selector (default: build-campaign). reduce-all merges the "
            "phase-1 and cut reductions of all runs; status reports missing shards."
        ),
    )
    parser.add_argument(
        "--hunter-scheduler",
        default=None,
        choices=("pbs", "slurm"),
        help=(
            "Batch system for generated Hunter scripts (default: "
            "IMPACT_HUNTER_SCHEDULER or 'pbs'; HLRS Hunter runs PBS Pro). "
            "'slurm' writes generic sbatch scripts."
        ),
    )
    parser.add_argument(
        "--hunter-array-index",
        type=int,
        default=None,
        help=(
            "PBS array index of a packed shard job; the shard is "
            "shards_per_node*index + PMI_LOCAL_RANK (set by PALS mpiexec)."
        ),
    )
    parser.add_argument(
        "--hunter-shards-per-node",
        type=int,
        default=None,
        help=(
            "Shards packed per node (build: default 4 for PBS mi300a nodes; "
            "shard stages: packing of this launch)."
        ),
    )
    parser.add_argument(
        "--hunter-phase1-shards-per-run",
        type=int,
        default=None,
        help=(
            "Phase-1 (mechanism) shards per run "
            "(default: IMPACT_HUNTER_PHASE1_SHARDS_PER_RUN or 16)."
        ),
    )
    parser.add_argument(
        "--hunter-cut-shards-per-run",
        type=int,
        default=None,
        help="Cut shards per run (default: IMPACT_HUNTER_CUT_SHARDS_PER_RUN or 256).",
    )
    parser.add_argument(
        "--hunter-workers-per-task",
        type=int,
        default=None,
        help=(
            "Worker processes per shard (default: cores per packed rank minus 2 "
            "for PBS; cpus-per-task for Slurm)."
        ),
    )
    parser.add_argument(
        "--hunter-iim-null-surrogates",
        type=int,
        default=None,
        help=(
            "build-campaign: add K circular-shift surrogate runs per real run "
            "(seeded as compute_IIM's null calibration); finalize then reports "
            "IIM_null_mean/IIM_null_sd/IIM_z and the calibrated IIM (default 0)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help=(
            "Pipeline checkout the Hunter jobs cd into and run (default: "
            "IMPACT_REPO_ROOT, the package's checkout, or this script's directory)."
        ),
    )
    parser.add_argument(
        "--assume-tr",
        type=float,
        default=None,
        help=(
            "Explicit TR (s) for fMRI preprocessing when neither the NIfTI header "
            "nor the BIDS sidecar provides a plausible TR. Without it such runs "
            "raise an error (no silent fallback)."
        ),
    )
    parser.add_argument(
        "--fmriprep-dir",
        default=None,
        help=(
            "fMRIPrep derivatives directory (read by --run-preprocessing, written "
            "by --run-fmriprep). Default: <bids-root>/derivatives/fmriprep."
        ),
    )
    parser.add_argument(
        "--no-atlas-robustness",
        action="store_true",
        help="Skip step 6 (metrics on the AAL-116 and Shen-268 robustness atlases).",
    )
    parser.add_argument(
        '--hunter-campaign-dir',
        default=None,
        help="Optional Hunter campaign directory override (default: <out-dir>/cache/hunter_iim_campaign).",
    )
    parser.add_argument(
        '--hunter-task-index',
        type=int,
        default=None,
        help="Array-task index for Hunter shard stages.",
    )
    parser.add_argument(
        '--hunter-run-index',
        type=int,
        default=None,
        help="Run index for Hunter reduction stages.",
    )
    parser.add_argument(
        '--mpc-metrics',
        nargs='+',
        default=None,
        help=(
            "Subset of MPC metrics to compute (choices: RAM PDI NAS IIM SRPI). "
            "If omitted, all MPC metrics are computed."
        ),
    )
    parser.add_argument(
        '--no-ci',
        action='store_true',
        help="Disable CI computation even if all MPC components are available.",
    )
    parser.add_argument(
        '--ci-reference',
        default=None,
        help=(
            "CI reference means: 'cohort_high_state' (default; awake-session "
            "cohort means, recorded as CI_reference) or a JSON file with "
            'per-component means, e.g. {"references": {"RAM": 1.0, "PDI": 1.0, '
            '"NAS": 1.0, "IIM": 1.0, "SRPI": 1.0}}. '
            "Non-finite or <=0 reference means make CI undefined."
        ),
    )
    parser.add_argument(
        '--atlas',
        default=None,
        help="Override atlas/time-series key used in preprocessed filenames.",
    )
    parser.add_argument(
        '--sessions',
        nargs='+',
        default=None,
        help="Override session labels (default from dataset config).",
    )
    parser.add_argument(
        '--condition',
        default=None,
        help="Override condition folder name under preprocessed/<subject>/<session>/<condition>.",
    )
    parser.add_argument(
        '--tr',
        type=float,
        default=None,
        help="Optional TR/sample interval override for metric computation.",
    )
    parser.add_argument('--eeg-target-sfreq', type=float, default=250.0)
    parser.add_argument('--eeg-l-freq', type=float, default=0.5)
    parser.add_argument('--eeg-h-freq', type=float, default=45.0)
    parser.add_argument('--eeg-max-duration-sec', type=float, default=120.0)
    parser.add_argument(
        '--iim-n-parts',
        type=int,
        default=None,
        help="Override number of evaluated system cuts for IIM MIP search (default per dataset; None means exhaustive).",
    )
    parser.add_argument(
        '--iim-max-timepoints',
        type=int,
        default=None,
        help="Override max timepoints used for IIM (uniform decimation; default per dataset).",
    )
    parser.add_argument(
        '--iim-max-nodes',
        type=int,
        default=None,
        help="Override max nodes used for IIM reduced subsystem (default per dataset).",
    )
    parser.add_argument(
        '--iim-max-mechanism-size',
        type=int,
        default=None,
        help="Override maximum mechanism size used in IIM Ψ computation.",
    )
    parser.add_argument(
        '--iim-max-purview-size',
        type=int,
        default=None,
        help="Override maximum purview size used in IIM Ψ computation.",
    )
    parser.add_argument(
        '--iim-parallel-workers',
        type=int,
        default=None,
        help="Override number of parallel IIM workers (default: auto planner).",
    )
    parser.add_argument(
        '--iim-memory-target-ratio',
        type=float,
        default=None,
        help="Target fraction of total system RAM for IIM worker planning (default per dataset).",
    )
    parser.add_argument(
        '--iim-worker-mem-gb-estimate',
        type=float,
        default=None,
        help="Estimated RAM footprint (GB) per IIM worker for auto planning.",
    )
    parser.add_argument(
        '--iim-cpu-oversub-factor',
        type=float,
        default=None,
        help="Soft multiplier on CPU count when auto-planning IIM workers (allows controlled oversubscription).",
    )
    parser.add_argument(
        '--iim-phase1-parallel-workers',
        type=int,
        default=None,
        help=(
            "Intra-task workers per IIM Ψ computation (phase-1 and per-cut Ψ in phase-2). "
            "None/1 disables intra-task parallelization."
        ),
    )
    parser.add_argument(
        '--iim-phase1-chunk-size',
        type=int,
        default=None,
        help="Mechanisms per intra-task work chunk (default from dataset config).",
    )
    parser.add_argument(
        '--iim-checkpoint-dir',
        default=None,
        help="Directory for per-run IIM checkpoint JSON files (default: <out-dir>/cache/iim_checkpoints).",
    )
    parser.add_argument(
        '--no-iim-resume-checkpoint',
        action='store_true',
        help="Disable resuming from existing IIM checkpoint files.",
    )
    parser.add_argument(
        '--iim-checkpoint-every-cuts',
        type=int,
        default=1,
        help="Write IIM checkpoint after every N newly completed cuts.",
    )
    parser.add_argument(
        '--iim-progress-log-every-cuts',
        type=int,
        default=1,
        help="Log IIM cut-level progress every N completed cuts.",
    )
    parser.add_argument(
        '--disable-iim-shared-memory',
        action='store_true',
        help="Disable shared-memory staging of IIM worker input arrays.",
    )
    parser.add_argument(
        '--disable-iim-phase1-shared-memory',
        action='store_true',
        help="Disable shared-memory staging for intra-task phase parallelization.",
    )
    parser.add_argument(
        '--disable-iim-parallel',
        action='store_true',
        help="Disable parallel IIM workers and force sequential IIM computation.",
    )
    args = parser.parse_args()
    main(
        args.out_dir,
        subjects=args.subjects,
        reuse_step2=args.reuse_step2,
        mpc_metrics=args.mpc_metrics,
        compute_ci=not args.no_ci,
        ci_reference=args.ci_reference,
        data_origin=args.data_origin,
        dataset_id=args.dataset_id,
        bids_root_override=args.bids_root,
        run_fmriprep=args.run_fmriprep or RUN_FMRIPREP,
        run_preprocessing_flag=args.run_preprocessing or RUN_PREPROCESSING,
        run_replication_flag=args.run_replication or RUN_REPLICATION,
        atlas_override=args.atlas,
        sessions_override=args.sessions,
        condition_override=args.condition,
        tr_override=args.tr,
        eeg_target_sfreq=args.eeg_target_sfreq,
        eeg_l_freq=args.eeg_l_freq,
        eeg_h_freq=args.eeg_h_freq,
        eeg_max_duration_sec=args.eeg_max_duration_sec,
        iim_n_parts_override=args.iim_n_parts,
        iim_max_timepoints_override=args.iim_max_timepoints,
        iim_max_nodes_override=args.iim_max_nodes,
        iim_max_mechanism_size_override=args.iim_max_mechanism_size,
        iim_max_purview_size_override=args.iim_max_purview_size,
        iim_parallel_workers_override=args.iim_parallel_workers,
        iim_memory_target_ratio_override=args.iim_memory_target_ratio,
        iim_worker_mem_gb_estimate_override=args.iim_worker_mem_gb_estimate,
        iim_cpu_oversub_factor_override=args.iim_cpu_oversub_factor,
        iim_phase1_parallel_workers_override=args.iim_phase1_parallel_workers,
        iim_phase1_chunk_size_override=args.iim_phase1_chunk_size,
        iim_checkpoint_dir_override=args.iim_checkpoint_dir,
        disable_iim_checkpoint_resume=args.no_iim_resume_checkpoint,
        iim_checkpoint_every_cuts=args.iim_checkpoint_every_cuts,
        iim_progress_log_every_cuts=args.iim_progress_log_every_cuts,
        disable_iim_shared_memory=args.disable_iim_shared_memory,
        disable_iim_phase1_shared_memory=args.disable_iim_phase1_shared_memory,
        disable_iim_parallel=args.disable_iim_parallel,
        execution_mode=args.execution_mode,
        hardware_target=args.hardware_target,
        hunter_stage=args.hunter_stage,
        hunter_campaign_dir=args.hunter_campaign_dir,
        hunter_task_index=args.hunter_task_index,
        hunter_run_index=args.hunter_run_index,
        assume_tr=args.assume_tr,
        fmriprep_dir=args.fmriprep_dir,
        hunter_scheduler=args.hunter_scheduler,
        hunter_array_index=args.hunter_array_index,
        hunter_shards_per_node=args.hunter_shards_per_node,
        hunter_phase1_shards_per_run=args.hunter_phase1_shards_per_run,
        hunter_cut_shards_per_run=args.hunter_cut_shards_per_run,
        hunter_workers_per_task=args.hunter_workers_per_task,
        atlas_robustness=not args.no_atlas_robustness,
        cli_argv=sys.argv,
        hunter_iim_null_surrogates=args.hunter_iim_null_surrogates,
        repo_root=args.repo_root,
    )
