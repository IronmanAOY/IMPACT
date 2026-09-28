#!/usr/bin/env python3
"""Generate and re-validate the real-data-derived synthetic smoke-test objects.

What these objects are
    Software smoke-test objects for the RAM, PDI, NAS, IIM, SRPI and CI code
    paths. Every run starts from a real OpenNeuro payload (fMRI voxel time
    series or scalp-EEG channels). The generator then plants known structure
    on top of it (state-dependent differentiation, directed module coupling,
    workspace broadcast and event-locked responses) and records every planted
    quantity, and its ground-truth direction, in the manifests.

What they are not
    They are not validation of the IMPaCT theory, and metric values computed
    on them are not evidence about consciousness or about metric validity on
    real data.

The validation report keeps three things apart:
    1. smoke-test checks (schema, BIDS/array consistency, readiness, metric
       definedness and documented bounds). Only these gate
       ``smoke_test_passed``.
    2. generator self-checks, computed without the metric code, that the
       planted structure is present in the arrays
       (``planted_structure_verified``).
    3. known-answer checks: whether each public metric recovers the direction
       of the planted state contrast, plus a null contrast with no planted
       difference. They are reported as observed and are never a pass gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg as sp_linalg
from scipy import stats

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional outside validation env
    nib = None

try:
    import mne
except Exception:  # pragma: no cover - optional outside validation env
    mne = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impact_pipeline import run_synergy_ci as _run_synergy_ci  # noqa: E402
from impact_pipeline.mpc_readiness import check_mpc_readiness  # noqa: E402
from impact_pipeline.mpc_metrics import compute_RAM, compute_SRPI  # noqa: E402
from impact_pipeline.provenance import DUMMY_DATA_ORIGIN  # noqa: E402
from impact_pipeline.synergy_ci import compute_synergy_ci  # noqa: E402

load_onsets = _run_synergy_ci.load_onsets

GENERATOR_VERSION = "2.0.0"
MANIFEST_VERSION = 2
OBJECT_KIND = "software_smoke_test_objects"
OBJECT_DISCLAIMER = (
    "Real-data-derived synthetic SOFTWARE SMOKE-TEST objects. The signals are "
    "engineered by this generator (planted structure recorded in the manifest). "
    "They check that the metric code runs and that planted structure can be "
    "recovered; they are not validation of the IMPaCT theory and their metric "
    "values are not evidence about consciousness or about metric validity on "
    "real data."
)
VALIDATION_SCOPE = (
    "compute_synergy_ci, compute_RAM and compute_SRPI (public metric functions) "
    "on the generator-written preprocessed arrays, with the configuration "
    "recorded in the manifest. run_pipeline orchestration, fMRIPrep/EEG "
    "preprocessing, run_s_ci and the statistics steps are NOT exercised."
)
PRIVACY_NOTE = (
    "Arrays are deterministic transforms of individual OpenNeuro recordings "
    "(CC0 sources) and keep the source subject IDs; they are participant-derived "
    "signals, not anonymous noise."
)

DATASETS_REL = Path("test_objects") / "datasets" / "real_derived_synth_completed"
RUNS_REL = Path("test_objects") / "runs" / "real_derived_synth_completed"
REPORTS_REL = Path("test_objects") / "real_derived_synth_completed" / "reports"
SUMMARY_NAME = "real_derived_synth_completed_summary.json"

SSD_ROOT = REPO_ROOT
INSPECTION_DIR = REPO_ROOT / REPORTS_REL
DATASETS_OUT = REPO_ROOT / DATASETS_REL
RUNS_OUT = REPO_ROOT / RUNS_REL
REPORTS_OUT = REPO_ROOT / REPORTS_REL
SOURCE_ROOT = REPO_ROOT / "data" / "scratch"


def _configure_paths(
    synth_root: str | Path | None = None,
    source_root: str | Path | None = None,
    inspection_dir: str | Path | None = None,
) -> None:
    """Resolve output/source roots (CLI value, then environment, then defaults)."""
    global SSD_ROOT, INSPECTION_DIR, DATASETS_OUT, RUNS_OUT, REPORTS_OUT, SOURCE_ROOT
    root = synth_root or os.environ.get("IMPACT_SYNTH_ROOT") or REPO_ROOT
    SSD_ROOT = Path(root).expanduser().resolve()
    DATASETS_OUT = SSD_ROOT / DATASETS_REL
    RUNS_OUT = SSD_ROOT / RUNS_REL
    REPORTS_OUT = SSD_ROOT / REPORTS_REL
    src = (
        source_root
        or os.environ.get("IMPACT_SOURCE_ROOT")
        or (SSD_ROOT / "data" / "scratch")
    )
    SOURCE_ROOT = Path(src).expanduser().resolve()
    INSPECTION_DIR = (
        Path(inspection_dir).expanduser().resolve() if inspection_dir else REPORTS_OUT
    )


_configure_paths()

TARGETS = ("ds003171", "ds002547", "ds005620")
# Event-timing donor actually read by the generator. ds003171 (rest donor for
# ds002547) and ds002547 (self/other timing) are targets and inspected anyway.
DONORS = ("ds005479",)

PDI_PARAMS = {
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
}
FMRI_RAM_PARAMS = {
    "epsilon": None,
    "magnitude_scale": 0.5,
    "response_model": "hrf",
    "response_boxcar_width_sec": None,
    "latency_method": "hrf_peak",
    "fir_window": 20.0,
    "xcorr_maxlag": 10,
    "quality_weights": (1.0, 1.0, 1.0),
    "goal_pre_window_sec": 2.0,
    "response_window_sec": 3.0,
    "goal_objective_window_sec": 2.0,
    "feedback_window_sec": 2.0,
    "quality_ridge": 1e-4,
    "require_explicit_feedback": True,
}
EEG_RAM_PARAMS = dict(FMRI_RAM_PARAMS)
EEG_RAM_PARAMS.update(
    {
        "response_model": "boxcar",
        "response_boxcar_width_sec": 0.30,
        "latency_method": "fir",
        "fir_window": 0.80,
        "goal_pre_window_sec": 0.20,
        "response_window_sec": 0.40,
        "goal_objective_window_sec": 0.20,
        "feedback_window_sec": 0.20,
    }
)
FMRI_NAS_PARAMS = {
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
}
EEG_NAS_PARAMS = dict(FMRI_NAS_PARAMS)
EEG_NAS_PARAMS.update(
    {
        "bands": ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)),
        "band_weights": (0.2, 0.2, 0.2, 0.2, 0.2),
        "window_len": 500,
        "step_len": 250,
    }
)
FMRI_SRPI_PARAMS = {
    "modality": "fmri",
    "pre_window_sec": 2.0,
    "response_lag_sec": 4.0,
    "response_window_sec": 6.0,
    "covariance_ridge": 1e-3,
    "component_weights": (0.35, 0.25, 0.20, 0.20),
    "min_events_per_class": 3,
    "sample_reliability_tau": 4.0,
    "eps": 1e-8,
}
EEG_SRPI_PARAMS = {
    "modality": "eeg",
    "pre_window_sec": 0.20,
    "response_lag_sec": 0.05,
    "response_window_sec": 0.40,
    "covariance_ridge": 1e-3,
    "component_weights": (0.35, 0.25, 0.20, 0.20),
    "min_events_per_class": 3,
    "sample_reliability_tau": 4.0,
    "eps": 1e-8,
}
# One IIM configuration, used for readiness AND computation and recorded in
# every manifest/report. It is a reduced smoke-test configuration, not the
# pipeline default (run_pipeline uses all nodes/sizes unless told otherwise).
IIM_VALIDATION_PARAMS = {
    "iim_bins": 2,
    "iim_lag_trs": 1,
    "iim_max_timepoints": 80,
    "iim_max_nodes": 6,
    "iim_max_mechanism_size": 2,
    "iim_max_purview_size": 2,
    "iim_max_state_space": 1500,
}
IIM_CONFIG_NOTES = (
    "Arrays longer than iim_max_timepoints are decimated by compute_synergy_ci "
    "with ts[:, ::ceil(T / iim_max_timepoints)] (no anti-aliasing); nodes are "
    "selected by compute_IIM's own rule from the full array; iim_max_state_space "
    "is the compute_IIM default and is passed explicitly to readiness only."
)
CI_THETAS = (0.5,)
GENERATION_DEFAULTS = {
    "fmri_nodes": 400,
    "eeg_seconds": 60.0,
    "eeg_sfreq": 250.0,
    "eeg_max_channels": None,
    "max_subjects": None,
}

# Session sets. The first two sessions of every target form the planted
# contrast and are allocated source payloads first.
SESSIONS = {
    "ds003171": ("awake", "deep", "light", "recovery"),
    "ds002547": ("awake", "deep", "ses-1", "ses-2"),
    # 'deep' is the pipeline's name for the deepest sedation level (task-sed2 in
    # ds005620; run_pipeline maps deep -> sed2). A separate 'sed2' pseudo-session
    # would duplicate it, so it is not generated.
    "ds005620": ("awake", "deep", "sed"),
}
# Planted state level g in [0, 1]. Every planted quantity scales with g, so the
# ground-truth direction for every metric is metric(high g) > metric(low g).
PLANTED_STATE_LEVELS = {
    "ds003171": {"awake": 1.0, "recovery": 1.0, "light": 0.5, "deep": 0.2},
    "ds002547": {"awake": 1.0, "deep": 0.2, "ses-1": 0.6, "ses-2": 0.6},
    "ds005620": {"awake": 1.0, "sed": 0.5, "deep": 0.2},
}
KNOWN_ANSWER_CONTRASTS = {
    "ds003171": {"planted": ("awake", "deep"), "null": ("awake", "recovery")},
    "ds002547": {"planted": ("awake", "deep"), "null": ("ses-1", "ses-2")},
    "ds005620": {"planted": ("awake", "deep"), "null": None},
}
KNOWN_ANSWER_METRICS = ("RAM", "PDI", "NAS", "IIM", "SRPI", "CI")
# Picked up automatically when the metric code reports them.
KNOWN_ANSWER_OPTIONAL_METRICS = ("IIM_raw", "IIM_z", "PDI_z", "NAS_z", "SRPI_z")
KNOWN_ANSWER_ALPHA = 0.05
METRIC_BOUNDS = {
    "RAM": (0.0, None),
    "PDI": (0.0, None),
    "NAS": (0.0, None),
    "IIM": (0.0, 1.0),
    "SRPI": (0.0, 1.0),
    "CI": (0.0, None),
}
# A-priori constants of the planted structure (fixed before looking at any
# metric output; not tuned to make metrics pass).
PLANTED_DESIGN = {
    # differentiation: share of variance kept node-specific,
    # d = floor + (1 - floor) * g; the rest is one real-derived global signal.
    "differentiation_floor": 0.25,
    # integration: latent modules with antisymmetric circulant coupling
    # kappa = max * g (directed lagged dependence, isotropic equal-time
    # covariance, so integration does not reduce differentiation).
    "coupling_modules": 4,
    "coupling_weight": 1.0,
    "coupling_kappa_max": 0.6,
    "coupling_tau_sec": {"fmri": 8.0, "eeg": 1.0},
    # broadcast: each receiver gets a fixed random mix of workspace nodes one
    # sample later with gain = max * g.
    "broadcast_fraction": 0.2,
    "broadcast_min_nodes": 4,
    "broadcast_inputs_per_receiver": 3,
    "broadcast_gain_max": 0.8,
    "broadcast_lag_samples": 1,
    # event-locked responses (peak amplitude in node-SD units, times g) in a
    # fixed fraction of nodes.
    "event_active_fraction": 0.1,
    "event_min_active_nodes": 3,
    "stimulus_peak_sd": 1.0,
    "goal_peak_sd": 0.75,
    "feedback_peak_sd": 0.75,
    "update_peak_sd": 0.5,
    "self_peak_sd": 1.0,
    "nonself_peak_sd": 0.5,
    # pre-event internal state (state-independent), and how strongly the self
    # response is modulated by it.
    "internal_state_sd": 1.0,
    "self_state_coupling": 0.5,
    "eeg_kernel_peak_sec": 0.15,
}
PLANTED_DESIGN_DESCRIPTION = [
    "Node payload x: real source payload, z-scored per node.",
    "Differentiation (PDI target): y = sqrt(d) x + sqrt(1-d) c(t), with c the "
    "leading principal component of x and d = floor + (1-floor) g.",
    "Integration (IIM target): nodes are assigned to latent modules; module "
    "latents follow an Ornstein-Uhlenbeck system dz = (-z + kappa (P - P^T) z) "
    "dt / tau + noise with ring permutation P, kappa = kappa_max g and exact "
    "discretisation. The antisymmetric coupling gives directed lagged "
    "dependence while the equal-time covariance stays isotropic.",
    "Broadcast (NAS target): a workspace (fixed node subset) is broadcast to "
    "every other node: receiver r gets gain_max g sum_j M[r, j] w_j(t - 1 "
    "sample), with M a fixed sparse random mix of workspace nodes w.",
    "Events (RAM target): goal, stimulus, feedback (value-scaled) and feedback-"
    "driven update responses convolved with a canonical kernel (double-gamma "
    "HRF for fMRI, alpha function for EEG), amplitude proportional to g.",
    "Self/non-self (SRPI target): pre-event internal state s ~ N(0,1) planted "
    "in [onset - pre_window, onset); self responses on a fixed pattern with "
    "amplitude self_peak g max(0, 1 + coupling s); non-self responses on a fresh "
    "random pattern with amplitude nonself_peak g.",
    "Rest arrays get the same differentiation/integration/broadcast structure "
    "for the same g, without events.",
    "Final arrays are z-scored per node (the preprocessing contract).",
]
MANIPULATION_CHECKS = (
    "participation_ratio",
    "module_coupling_directionality",
    "broadcast_directionality",
    "stimulus_evoked_projection",
    "self_specific_projection",
)
BIDS_EEG_MICROVOLTS_PER_UNIT = 10.0
_NON_EEG_NAME_RE = re.compile(r"eog|emg|ecg|ekg|resp|trig|status|misc|gsr|stim", re.I)
_NON_EEG_TYPES = {"eog", "emg", "ecg", "ekg", "misc", "trig", "stim", "resp", "gsr"}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path | str, base: Path) -> str:
    """Path relative to base (POSIX); manifests never store absolute paths.

    The lexical path is tried first so that symlinked files (git-annex) keep
    their BIDS name instead of the annex object path.
    """
    p = Path(path)
    for cand, root in ((p.absolute(), base.absolute()), (p.resolve(), base.resolve())):
        try:
            return cand.relative_to(root).as_posix()
        except ValueError:
            continue
    return p.name


def _source_rel(path: Path | str, dataset_id: str) -> str:
    """Source payload path as '<dataset_id>/<path inside the dataset>'."""
    rel = _rel(path, _source_dataset_root(dataset_id))
    return f"{dataset_id}/{rel}"


def _load_inspect_module():
    path = Path(__file__).resolve().with_name("inspect_real_sources_for_synth.py")
    spec = importlib.util.spec_from_file_location(
        "inspect_real_sources_for_synth", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_sources(dataset_ids: tuple[str, ...] | list[str]) -> list[str]:
    """Source datasets read when generating the given targets."""
    needed = list(dataset_ids) + ["ds002547", *DONORS]  # self/other + MID templates
    if "ds002547" in dataset_ids:
        needed.append("ds003171")  # rest donor for ds002547
    return [ds for ds in (*TARGETS, *DONORS) if ds in set(needed)]


def _load_source_inspections(
    dataset_ids: tuple[str, ...] | list[str] = TARGETS,
    *,
    run_missing: bool = True,
) -> dict[str, dict[str, Any]]:
    """Load <ds>_source_inspection.json; inspect missing sources when allowed."""
    wanted = _required_sources(dataset_ids)
    inspections = {}
    missing = []
    for ds in wanted:
        path = INSPECTION_DIR / f"{ds}_source_inspection.json"
        if path.exists():
            inspections[ds] = _read_json(path)
        else:
            missing.append(ds)
    if missing and run_missing:
        inspect_mod = _load_inspect_module()
        print(f"inspecting sources without inspection reports: {missing}", flush=True)
        inspect_mod.inspect_datasets(
            missing,
            output_dir=INSPECTION_DIR,
            source_root=SOURCE_ROOT,
            max_nifti=4,
            max_eeg=4,
        )
        for ds in missing:
            path = INSPECTION_DIR / f"{ds}_source_inspection.json"
            if path.exists():
                inspections[ds] = _read_json(path)
    not_available = [
        ds for ds in wanted if not inspections.get(ds, {}).get("available")
    ]
    if not_available:
        raise FileNotFoundError(
            "Source datasets not available under the source root "
            f"({SOURCE_ROOT}): {not_available}. Run "
            "scripts/inspect_real_sources_for_synth.py --source-root <root> "
            "--output-dir <inspection dir> first, or pass --source-root."
        )
    return inspections


def _first_existing(patterns: list[Path]) -> Path:
    for p in patterns:
        if p.exists():
            return p
    raise FileNotFoundError(f"No existing path among: {[str(p) for p in patterns]}")


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        expanded = path.expanduser()
        key = str(expanded)
        if key not in seen:
            seen.add(key)
            out.append(expanded)
    return out


def _source_dataset_candidates(dataset_id: str) -> list[Path]:
    root = SOURCE_ROOT.expanduser()
    candidates: list[Path] = []
    if dataset_id == "ds005620":
        if root.name in {"ds005620", "ds005620_annex"}:
            candidates.append(root)
        candidates.extend(
            [
                root / "ds005620_annex",
                root / "ds005620",
                root / "data" / "scratch" / "ds005620_annex",
                root / "data" / "scratch" / "ds005620",
                root / "data" / "ds005620",
            ]
        )
    else:
        if root.name == dataset_id:
            candidates.append(root)
        candidates.extend(
            [
                root / dataset_id,
                root / "data" / "scratch" / dataset_id,
                root / "data" / dataset_id,
            ]
        )
    return _unique_paths(candidates)


def _source_dataset_root(dataset_id: str) -> Path:
    candidates = _source_dataset_candidates(dataset_id)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def _load_mid_events() -> pd.DataFrame:
    root = _source_dataset_root("ds005479")
    path = _first_existing(sorted(root.glob("sub-*/func/*_task-MID_events.tsv")))
    df = pd.read_csv(path, sep="\t")
    label_to_reward = {
        "loss big": 0.25,
        "loss small": 0.55,
        "win small": 1.05,
        "win big": 1.55,
    }
    df["feedback_value"] = (
        df["trial_type"].astype(str).str.lower().map(label_to_reward).astype(float)
    )
    df["source_file"] = _source_rel(path, "ds005479")
    return df


def _load_self_other_templates() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = _source_dataset_root("ds002547")
    self_path = _first_existing(
        sorted(root.glob("sub-*/ses-*/func/*_task-self*_events.tsv"))
    )
    other_path = _first_existing(
        sorted(root.glob("sub-*/ses-*/func/*_task-other*_events.tsv"))
    )
    self_df = pd.read_csv(self_path, sep="\t")
    other_df = pd.read_csv(other_path, sep="\t")
    self_df["source_file"] = _source_rel(self_path, "ds002547")
    other_df["source_file"] = _source_rel(other_path, "ds002547")
    return self_df, other_df


def _normalize_nodes(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x -= x.mean(axis=0, keepdims=True)
    x /= x.std(axis=0, keepdims=True) + 1e-6
    return x.astype(np.float32)


def _acf1_summary(ts: np.ndarray) -> dict[str, Any]:
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        return {"n": 0}
    x0 = x[:-1] - x[:-1].mean(axis=0, keepdims=True)
    x1 = x[1:] - x[1:].mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(x0 * x0, axis=0) * np.sum(x1 * x1, axis=0))
    vals = np.divide(
        np.sum(x0 * x1, axis=0), denom, out=np.zeros(x.shape[1]), where=denom > 1e-12
    )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _corr_mean(ts: np.ndarray) -> float | None:
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        return None
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    keep = sd.reshape(-1) > 1e-8
    if int(keep.sum()) < 2:
        return None
    x = x[:, keep] / sd[:, keep]
    c = np.corrcoef(x, rowvar=False)
    tri = c[np.triu_indices_from(c, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return None
    return float(np.mean(tri))


def _extract_nifti_payload_timeseries(
    path: Path,
    *,
    n_nodes: int,
    rng: np.random.Generator,
    n_time: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample ``n_nodes`` real voxel time series (native length, no tiling)."""
    if nib is None:
        raise RuntimeError("nibabel is required for fMRI payload extraction")
    if not path.exists():
        raise FileNotFoundError(path)
    img = nib.load(str(path))
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D NIfTI, got {img.shape}: {path}")
    shape = tuple(int(v) for v in img.shape)
    # gzipped NIfTI proxies are very slow for random voxel access. Load once,
    # sample voxel rows in memory, then release the array before the next run.
    data = np.asarray(img.dataobj, dtype=np.float32)
    vol0 = data[..., 0]
    mask = np.isfinite(vol0) & (
        np.abs(vol0) > max(1e-6, float(np.nanpercentile(np.abs(vol0), 55)))
    )
    coords = np.argwhere(mask)
    if coords.shape[0] < int(n_nodes):
        mask = np.isfinite(vol0) & (np.abs(vol0) > 1e-8)
        coords = np.argwhere(mask)
    if coords.shape[0] < int(n_nodes):
        raise RuntimeError(
            f"Not enough nonzero voxels in {path}: {coords.shape[0]} < {n_nodes}"
        )
    rng.shuffle(coords)
    flat_indices = np.ravel_multi_index(coords.T, shape[:3])
    flat = data.reshape((-1, shape[3]))
    selected = None
    for mult in (2, 4, 8, 16, 32):
        cand_idx = flat_indices[: min(flat_indices.size, int(n_nodes) * mult)]
        cand = np.asarray(flat[cand_idx, :], dtype=np.float32)
        keep = np.isfinite(cand).all(axis=1) & (np.std(cand, axis=1) > 1e-5)
        if int(keep.sum()) >= int(n_nodes):
            selected = cand[keep][: int(n_nodes)]
            break
    if selected is None or selected.shape[0] < int(n_nodes):
        raise RuntimeError(
            f"Could not extract {n_nodes} valid voxel series from {path}"
        )
    out = selected.T.astype(np.float32)
    if n_time is not None and int(n_time) > 0:
        out = out[: int(n_time)]
    out = _normalize_nodes(out)
    del data
    meta = {
        "source_shape": shape,
        "source_zooms": tuple(float(v) for v in img.header.get_zooms()),
        "extracted_nodes": int(out.shape[1]),
        "extracted_timepoints": int(out.shape[0]),
        "acf1": _acf1_summary(out),
        "corr_mean": _corr_mean(out),
    }
    return out, meta


def _channel_types_from_sidecar(vhdr: Path) -> dict[str, str]:
    """Channel types from the BIDS channels.tsv next to a BrainVision header."""
    name = vhdr.name
    if not name.endswith("_eeg.vhdr"):
        return {}
    tsv = vhdr.with_name(name[: -len("_eeg.vhdr")] + "_channels.tsv")
    if not tsv.exists():
        return {}
    try:
        df = pd.read_csv(tsv, sep="\t")
    except Exception:
        return {}
    df = df.rename(columns={c: str(c).strip().lstrip("﻿") for c in df.columns})
    if "name" not in df.columns or "type" not in df.columns:
        return {}
    return {str(n): str(t).strip().lower() for n, t in zip(df["name"], df["type"])}


def _scalp_eeg_channels(raw, vhdr: Path) -> tuple[list[str], list[str]]:
    """Scalp EEG channels; EOG/EMG/ECG/misc excluded by type and by name."""
    sidecar_types = _channel_types_from_sidecar(vhdr)
    keep, excluded = [], []
    for name, typ in zip(raw.ch_names, raw.get_channel_types()):
        bad = (
            str(typ).lower() != "eeg"
            or sidecar_types.get(name, "eeg") in _NON_EEG_TYPES
            or bool(_NON_EEG_NAME_RE.search(name))
        )
        (excluded if bad else keep).append(name)
    return keep, excluded


def _brainvision_duration_sec(path: Path) -> float:
    if mne is None:
        raise RuntimeError("mne is required for EEG payload extraction")
    raw = mne.io.read_raw_brainvision(str(path), preload=False, verbose="ERROR")
    return float(raw.n_times) / float(raw.info["sfreq"])


def _extract_brainvision_payload_timeseries(
    path: Path,
    *,
    n_nodes: int | None = None,
    target_sfreq: float,
    max_seconds: float,
    start_sec: float = 0.0,
    channels: list[str] | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract a [start, start + max_seconds) scalp-EEG segment (time x channel)."""
    if mne is None:
        raise RuntimeError("mne is required for EEG payload extraction")
    if not path.exists():
        raise FileNotFoundError(path)
    raw = mne.io.read_raw_brainvision(str(path), preload=False, verbose="ERROR")
    orig_sfreq = float(raw.info["sfreq"])
    scalp, excluded = _scalp_eeg_channels(raw, path)
    picks = list(channels) if channels is not None else scalp
    missing = [c for c in picks if c not in raw.ch_names]
    if missing:
        raise RuntimeError(f"Channels {missing} not found in {path}")
    if n_nodes is not None:
        picks = picks[: int(n_nodes)]
    start = int(round(float(start_sec) * orig_sfreq))
    stop = min(int(raw.n_times), start + int(round(float(max_seconds) * orig_sfreq)))
    if stop - start < 2:
        raise RuntimeError(f"Segment [{start_sec}, +{max_seconds}) s is outside {path}")
    raw = (
        raw.copy()
        .pick(picks)
        .crop(
            tmin=start / orig_sfreq,
            tmax=(stop - 1) / orig_sfreq,
            include_tmax=True,
        )
    )
    raw.load_data(verbose="ERROR")
    if (
        float(target_sfreq) > 0
        and abs(float(raw.info["sfreq"]) - float(target_sfreq)) > 1e-6
    ):
        raw.resample(float(target_sfreq), npad="auto", verbose="ERROR")
    data = raw.get_data().T.astype(np.float64)
    sd = data.std(axis=0)
    ref = float(np.median(sd[sd > 0])) if np.any(sd > 0) else 0.0
    flat = [raw.ch_names[i] for i in np.flatnonzero(sd <= max(1e-20, 1e-6 * ref))]
    if normalize:
        data = _normalize_nodes(data)
    meta = {
        "source_sfreq": orig_sfreq,
        "target_sfreq": float(target_sfreq),
        "segment_start_sec": float(start_sec),
        "extracted_nodes": int(data.shape[1]),
        "extracted_timepoints": int(data.shape[0]),
        "duration_sec": float(data.shape[0] / float(target_sfreq)),
        "channels": list(raw.ch_names),
        "excluded_non_eeg_channels": excluded,
        "flat_channels": flat,
    }
    return np.asarray(data, dtype=np.float32), meta


class _SourceAllocator:
    """Hands out non-overlapping source payload segments within one subject.

    fMRI runs are used whole (``segment_sec=None``); EEG recordings are cut into
    consecutive non-overlapping segments. A payload is only reused once every
    candidate is exhausted, and the reuse is flagged in the manifest.
    """

    def __init__(self, segment_sec: float | None = None, duration_fn=None):
        self.segment_sec = segment_sec
        self.duration_fn = duration_fn
        self.used: dict[str, list[str]] = {}
        self._durations: dict[str, float] = {}

    def _duration(self, path: Path) -> float:
        key = str(path)
        if key not in self._durations:
            self._durations[key] = float(self.duration_fn(path))
        return self._durations[key]

    def allocate(
        self, candidates: list[tuple[Path, bool]], owner: str
    ) -> dict[str, Any]:
        existing = [(p, pref) for p, pref in candidates if p.exists()]
        for path, preferred in existing:
            users = self.used.setdefault(str(path), [])
            if self.segment_sec is None:
                if users:
                    continue
                start = 0.0
            else:
                start = len(users) * float(self.segment_sec)
                if start + float(self.segment_sec) > self._duration(path) + 1e-6:
                    continue
            users.append(owner)
            return {
                "path": path,
                "start_sec": float(start),
                "reused": False,
                "shared_with": [],
                "preferred_source": bool(preferred),
            }
        if existing:
            path, preferred = existing[0]
            users = self.used.setdefault(str(path), [])
            shared = list(users)
            users.append(owner)
            return {
                "path": path,
                "start_sec": 0.0,
                "reused": True,
                "shared_with": shared,
                "preferred_source": bool(preferred),
            }
        raise FileNotFoundError(
            f"No source payload for {owner}; candidates: "
            f"{[str(p) for p, _ in candidates]}"
        )


def _ds003171_bold_path(subj: str, session: str, *, rest: bool) -> Path:
    root = _source_dataset_root("ds003171")
    task = f"rest{session}" if rest else f"audio{session}"
    path = root / f"sub-{subj}" / "func" / f"sub-{subj}_task-{task}_run-01_bold.nii.gz"
    if path.exists():
        return path
    if (not rest) and session == "awake":
        fallback = (
            root / f"sub-{subj}" / "func" / f"sub-{subj}_task-audio_run-01_bold.nii.gz"
        )
        if fallback.exists():
            return fallback
    return path


_DS002547_TASK_PREFERENCE = {
    "awake": ("ses-1:self_run-1", "ses-1:self_run-2", "ses-1:other"),
    "deep": ("ses-1:other", "ses-2:other", "ses-1:self_run-2"),
    "ses-1": ("ses-1:self_run-2", "ses-1:self_run-1", "ses-1:other"),
    "ses-2": ("ses-2:self_run-1", "ses-2:self_run-2", "ses-2:other"),
}
_DS002547_ALL_RUNS = (
    "ses-1:self_run-1",
    "ses-1:self_run-2",
    "ses-1:other",
    "ses-2:self_run-1",
    "ses-2:self_run-2",
    "ses-2:other",
)
_DS002547_REST_DONOR_STATE = {
    "awake": "awake",
    "deep": "deep",
    "ses-1": "light",
    "ses-2": "recovery",
}


def _ds002547_run_path(subj: str, key: str) -> Path:
    ses, task = key.split(":")
    root = _source_dataset_root("ds002547")
    func = root / "derivatives" / "fmriprep" / f"sub-{subj}" / ses / "func"
    space = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    return func / f"sub-{subj}_{ses}_task-{task}_{space}"


def _ds002547_task_path(subj: str, session: str) -> Path:
    for key in (*_DS002547_TASK_PREFERENCE.get(session, ()), *_DS002547_ALL_RUNS):
        path = _ds002547_run_path(subj, key)
        if path.exists():
            return path
    return _ds002547_run_path(
        subj, _DS002547_TASK_PREFERENCE.get(session, _DS002547_ALL_RUNS)[0]
    )


def _ds005620_vhdr_path(subj: str, task: str) -> Path:
    eeg_dir = _source_dataset_root("ds005620") / f"sub-{subj}" / "eeg"
    return eeg_dir / f"sub-{subj}_task-{task}_eeg.vhdr"


def _source_candidates(
    dataset_id: str,
    subj: str,
    session: str,
    *,
    rest: bool,
    donor_subject: str | None = None,
) -> list[tuple[Path, bool]]:
    """Ordered (path, preferred) source candidates for one run."""
    if dataset_id == "ds003171":
        return [(_ds003171_bold_path(subj, session, rest=rest), True)]
    if dataset_id == "ds002547":
        if rest:
            first = _DS002547_REST_DONOR_STATE.get(session, "awake")
            states = [first] + [
                s for s in ("awake", "deep", "light", "recovery") if s != first
            ]
            return [
                (_ds003171_bold_path(donor_subject, s, rest=True), s == first)
                for s in states
            ]
        preferred = _DS002547_TASK_PREFERENCE.get(session, ())
        keys = list(preferred) + [k for k in _DS002547_ALL_RUNS if k not in preferred]
        return [(_ds002547_run_path(subj, k), k in preferred) for k in keys]
    if dataset_id == "ds005620":
        awake = (
            ["awake_acq-EO", "awake_acq-EC"]
            if rest
            else ["awake_acq-EC", "awake_acq-EO"]
        )
        if session == "awake":
            order = [(t, True) for t in awake]
            order += [(t, False) for t in ("sed_acq-rest_run-1", "sed2_acq-rest_run-1")]
        else:
            task = "sed2" if session == "deep" else "sed"
            other = "sed" if task == "sed2" else "sed2"
            runs = (2, 3, 1) if rest else (1, 2, 3)
            order = [(f"{task}_acq-rest_run-{r}", True) for r in runs]
            order += [(f"{other}_acq-rest_run-{r}", False) for r in runs]
            order += [(t, False) for t in awake]
        return [(_ds005620_vhdr_path(subj, t), pref) for t, pref in order]
    raise ValueError(dataset_id)


def _source_subjects(dataset_id: str) -> list[str]:
    root = _source_dataset_root(dataset_id)
    return [
        p.name.replace("sub-", "")
        for p in sorted(root.glob("sub-*"))
        if p.is_dir() and re.fullmatch(r"sub-[A-Za-z0-9]+", p.name)
    ]


def _sessions_to_generate(dataset_id: str) -> list[str]:
    if dataset_id not in SESSIONS:
        raise ValueError(dataset_id)
    return list(SESSIONS[dataset_id])


def _leading_components(x: np.ndarray, k: int) -> np.ndarray:
    """First k principal-component time courses of x (time x node), z-scored."""
    xc = np.asarray(x, dtype=np.float64)
    xc = xc - xc.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    comps = u[:, :k] * s[:k]
    if comps.shape[1] < k:
        pad = np.zeros((comps.shape[0], k - comps.shape[1]))
        comps = np.concatenate([comps, pad], axis=1)
    return _normalize_nodes(comps).astype(np.float64)


def _coupled_module_latents(
    n_time: int,
    n_modules: int,
    *,
    kappa: float,
    tau_sec: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Ornstein-Uhlenbeck modules with antisymmetric circulant coupling.

    Module m is driven by +kappa z[m-1] and -kappa z[m+1]. Because the coupling
    is antisymmetric the stationary equal-time covariance stays isotropic; the
    coupling only creates directed time-lagged dependence.
    """
    m = int(n_modules)
    ring = np.roll(np.eye(m), 1, axis=0)
    drift = (-np.eye(m) + float(kappa) * (ring - ring.T)) / float(tau_sec)
    step = sp_linalg.expm(drift * float(dt))
    burn = int(np.ceil(5.0 * float(tau_sec) / float(dt)))
    noise = rng.normal(size=(int(n_time) + burn, m))
    z = np.zeros(m)
    out = np.empty((int(n_time), m))
    for t in range(int(n_time) + burn):
        z = step @ z + noise[t]
        if t >= burn:
            out[t - burn] = z
    return _normalize_nodes(out).astype(np.float64)


def _node_pattern(
    rng: np.random.Generator,
    n_nodes: int,
    n_active: int,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    pool = np.arange(int(n_nodes))
    if exclude is not None and exclude.size:
        rest = np.setdiff1d(pool, exclude)
        if rest.size >= int(n_active):
            pool = rest
    idx = rng.choice(pool, size=int(n_active), replace=False)
    vec = np.zeros(int(n_nodes), dtype=np.float32)
    vec[idx] = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=int(n_active))
    return vec


def _subject_layout(
    n_nodes: int,
    rng: np.random.Generator,
    design: dict[str, Any] = PLANTED_DESIGN,
) -> dict[str, Any]:
    """Per-subject node roles shared by all sessions of that subject."""
    n_modules = int(min(int(design["coupling_modules"]), n_nodes))
    perm = rng.permutation(n_nodes)
    modules = np.empty(n_nodes, dtype=int)
    modules[perm] = np.arange(n_nodes) % n_modules
    n_ws = int(
        max(
            design["broadcast_min_nodes"], round(design["broadcast_fraction"] * n_nodes)
        )
    )
    n_ws = int(min(n_ws, n_nodes - 1))
    workspace = np.sort(rng.permutation(n_nodes)[:n_ws])
    n_recv = n_nodes - n_ws
    k_in = int(min(int(design["broadcast_inputs_per_receiver"]), n_ws))
    broadcast_mix = np.zeros((n_recv, n_ws), dtype=np.float64)
    for r in range(n_recv):
        cols = rng.choice(n_ws, size=k_in, replace=False)
        broadcast_mix[r, cols] = rng.choice([-1.0, 1.0], size=k_in) / math.sqrt(k_in)
    n_active = int(
        min(
            n_nodes,
            max(
                design["event_min_active_nodes"],
                round(design["event_active_fraction"] * n_nodes),
            ),
        )
    )
    patterns = {}
    used = np.zeros(0, dtype=int)
    for name in ("stimulus", "goal", "feedback", "self", "internal_state"):
        vec = _node_pattern(rng, n_nodes, n_active, exclude=used)
        patterns[name] = vec
        used = np.union1d(used, np.flatnonzero(vec))
    return {
        "modules": modules,
        "n_modules": n_modules,
        "workspace": workspace,
        "broadcast_mix": broadcast_mix,
        "patterns": patterns,
        "n_active": n_active,
    }


def _plant_state_structure(
    payload: np.ndarray,
    *,
    level: float,
    modality: str,
    sample_interval: float,
    layout: dict[str, Any],
    rng: np.random.Generator,
    design: dict[str, Any] = PLANTED_DESIGN,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Plant differentiation, module coupling and broadcast for state level g."""
    g = float(np.clip(level, 0.0, 1.0))
    x = _normalize_nodes(payload).astype(np.float64)
    n_time, n_nodes = x.shape
    common = _leading_components(x, 1)[:, 0]
    floor = float(design["differentiation_floor"])
    d = floor + (1.0 - floor) * g
    y = math.sqrt(d) * x + math.sqrt(1.0 - d) * common[:, None]

    kappa = float(design["coupling_kappa_max"]) * g
    tau = float(design["coupling_tau_sec"][modality])
    latents = _coupled_module_latents(
        n_time,
        layout["n_modules"],
        kappa=kappa,
        tau_sec=tau,
        dt=sample_interval,
        rng=rng,
    )
    y += float(design["coupling_weight"]) * latents[:, layout["modules"]]

    ws = np.asarray(layout["workspace"], dtype=int)
    recv = np.setdiff1d(np.arange(n_nodes), ws)
    lag = int(design["broadcast_lag_samples"])
    gain = float(design["broadcast_gain_max"]) * g
    if n_time > lag and recv.size:
        source = _normalize_nodes(y[:, ws]).astype(np.float64)
        y[lag:, recv] += gain * (source[:-lag] @ np.asarray(layout["broadcast_mix"]).T)
    truth = {
        "level": g,
        "differentiation_weight": d,
        "coupling_kappa": kappa,
        "coupling_tau_sec": tau,
        "broadcast_gain": gain,
        "broadcast_lag_samples": lag,
    }
    return y.astype(np.float32), truth


def _gamma_pdf(t: np.ndarray, shape: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = np.exp((shape - 1.0) * np.log(t[pos]) - t[pos] - math.lgamma(shape))
    return out


def _response_kernel(
    modality: str, dt: float, design: dict[str, Any] = PLANTED_DESIGN
) -> np.ndarray:
    """Double-gamma HRF (fMRI, peak ~5 s) or alpha-function ERP (EEG), peak 1."""
    if modality == "eeg":
        peak = float(design["eeg_kernel_peak_sec"])
        t = np.arange(0.0, 0.8, float(dt))
        k = (t / peak) * np.exp(1.0 - t / peak)
    else:
        t = np.arange(0.0, 32.0, float(dt))
        k = _gamma_pdf(t, 6.0) - _gamma_pdf(t, 16.0) / 6.0
    k = np.asarray(k, dtype=float)
    k /= max(float(np.max(k)), 1e-12)
    return k.astype(np.float32)


def _add_kernel(
    ts: np.ndarray, idx: int, pattern: np.ndarray, amp: float, kernel: np.ndarray
) -> None:
    n_time = ts.shape[0]
    if idx >= n_time:
        return
    start = max(0, int(idx))
    stop = min(n_time, start + len(kernel))
    if stop <= start:
        return
    ts[start:stop, :] += (
        float(amp) * kernel[: stop - start, None] * pattern[None, :]
    ).astype(ts.dtype)


def _build_events(
    *,
    rng: np.random.Generator,
    n_time: int,
    tr: float,
    modality: str,
    mid_events: pd.DataFrame,
    self_template: pd.DataFrame,
    other_template: pd.DataFrame,
    session: str,
) -> pd.DataFrame:
    run_stop = (int(n_time) - 1) * float(tr)
    rows: list[dict[str, Any]] = []
    if modality == "eeg":
        trial_onsets = np.arange(4.0, min(run_stop - 3.0, 54.0), 6.0)
    else:
        raw = mid_events.loc[mid_events["onset"].astype(float) + 8.0 < run_stop].head(
            18
        )
        trial_onsets = raw["onset"].astype(float).to_numpy()
    if trial_onsets.size < 6:
        trial_onsets = np.linspace(8.0, max(18.0, run_stop - 16.0), 8)

    reward_template = mid_events["feedback_value"].astype(float).to_numpy()
    reward_template = reward_template[np.isfinite(reward_template)]
    if reward_template.size == 0:
        reward_template = np.asarray([0.25, 0.55, 1.05, 1.55], dtype=float)
    reward_vals = np.resize(reward_template, trial_onsets.shape[0])
    trial_labels = np.resize(
        mid_events["trial_type"].astype(str).to_numpy(), trial_onsets.shape[0]
    )
    for k, onset in enumerate(trial_onsets):
        onset = float(onset)
        dur = (
            0.40
            if modality == "eeg"
            else float(mid_events["duration"].astype(float).median())
        )
        feedback_onset = onset + (0.8 if modality == "eeg" else dur + 1.0)
        rows.append(
            {
                "onset": max(0.0, onset - (0.6 if modality == "eeg" else 2.0)),
                "duration": 0.20 if modality == "eeg" else 1.0,
                "trial_type": "goal_cue",
                "reward": np.nan,
                "source_event_family": "ds005479_mid_goal_proxy",
            }
        )
        rows.append(
            {
                "onset": onset,
                "duration": dur,
                "trial_type": "stimulus_target",
                "reward": np.nan,
                "source_event_family": f"ds005479_mid_{trial_labels[k]}",
            }
        )
        rows.append(
            {
                "onset": feedback_onset,
                "duration": 0.20 if modality == "eeg" else 1.0,
                "trial_type": "feedback_reward",
                "reward": float(reward_vals[k]),
                "outcome": str(trial_labels[k]),
                "source_event_family": f"ds005479_mid_{trial_labels[k]}",
            }
        )

    if modality == "eeg":
        base_self = np.arange(3.0, min(run_stop - 1.0, 58.0), 5.0)
        base_non = base_self + 2.4
    else:
        self_on = (
            pd.to_numeric(self_template["onset"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        other_on = (
            pd.to_numeric(other_template["onset"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        source_on = np.sort(
            np.concatenate(
                [self_on[np.isfinite(self_on)], other_on[np.isfinite(other_on)]]
            )
        )
        source_diffs = np.diff(source_on)
        source_diffs = source_diffs[(source_diffs >= 4.0) & (source_diffs <= 30.0)]
        median_source_gap = float(np.median(source_diffs)) if source_diffs.size else 8.0
        # Keep fMRI self/nonself response windows separated.
        slot_gap = max(10.5, median_source_gap)
        n_slots = 20
        latest_start = max(12.0, run_stop - (slot_gap * (n_slots - 1) + 18.0))
        first_source = (
            float(source_on[source_on > 4.0][0]) if np.any(source_on > 4.0) else 10.0
        )
        start = max(first_source, latest_start)
        slots = start + np.arange(n_slots, dtype=float) * slot_gap
        slots = slots[slots < run_stop - 12.0]
        if slots.size < 12:
            slots = np.linspace(14.0, max(30.0, run_stop - 16.0), 12)
        base_self = slots[::2][:10]
        base_non = slots[1::2][:10]
    for onset in base_self[:10]:
        rows.append(
            {
                "onset": float(onset),
                "duration": 0.35 if modality == "eeg" else 6.0,
                "trial_type": "self",
                "reward": np.nan,
                "source_event_family": "ds002547_task-self",
            }
        )
    for onset in base_non[:10]:
        if float(onset) >= run_stop - (1.0 if modality == "eeg" else 12.0):
            continue
        rows.append(
            {
                "onset": float(onset),
                "duration": 0.35 if modality == "eeg" else 6.0,
                "trial_type": "nonself",
                "reward": np.nan,
                "source_event_family": "ds002547_task-other",
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["onset", "trial_type"]).reset_index(drop=True)
    df["synthetic_derivation"] = "real_data_derived_synthetic_smoke_test"
    jitter = rng.normal(0.0, 0.015 if modality == "eeg" else 0.08, size=len(df))
    df["onset"] = np.maximum(0.0, df["onset"].astype(float) + jitter)
    return df


def _prediction_errors(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    pred = np.empty_like(x)
    pred[0] = x[0]
    if x.size > 1:
        pred[1:] = np.cumsum(x[:-1]) / np.arange(1, x.size, dtype=float)
    return x - pred


def _plant_event_responses(
    ts: np.ndarray,
    events: pd.DataFrame,
    *,
    level: float,
    modality: str,
    sample_interval: float,
    layout: dict[str, Any],
    rng: np.random.Generator,
    pre_window_sec: float,
    design: dict[str, Any] = PLANTED_DESIGN,
) -> dict[str, Any]:
    """Add event-locked responses whose amplitude is proportional to level g."""
    g = float(np.clip(level, 0.0, 1.0))
    n_time, n_nodes = ts.shape
    dt = float(sample_interval)
    kernel = _response_kernel(modality, dt, design)
    pats = layout["patterns"]
    n_active = int(layout["n_active"])
    ev = events.sort_values("onset").reset_index(drop=True)

    def _idx(onset: float) -> int:
        return int(round(float(onset) / dt))

    goal_on = ev.loc[ev["trial_type"].eq("goal_cue"), "onset"].to_numpy(dtype=float)
    stim_on = ev.loc[ev["trial_type"].eq("stimulus_target"), "onset"].to_numpy(
        dtype=float
    )
    fb_rows = ev.loc[ev["trial_type"].eq("feedback_reward")]
    fb_on = fb_rows["onset"].to_numpy(dtype=float)
    fb_val = pd.to_numeric(fb_rows["reward"], errors="coerce").to_numpy(dtype=float)
    if fb_val.size and not np.all(np.isfinite(fb_val)):
        fb_val = np.where(np.isfinite(fb_val), fb_val, np.nanmean(fb_val))
    n_trials = int(min(goal_on.size, stim_on.size, fb_on.size))
    value_scale = (
        fb_val[:n_trials] / max(float(np.mean(fb_val[:n_trials])), 1e-12)
        if n_trials
        else fb_val
    )
    pe = _prediction_errors(fb_val[:n_trials])
    pe_scale = np.abs(pe) / max(float(np.mean(np.abs(pe))), 1e-12) if n_trials else pe
    for k in range(n_trials):
        _add_kernel(
            ts,
            _idx(goal_on[k]),
            pats["goal"],
            design["goal_peak_sd"] * g * value_scale[k],
            kernel,
        )
        _add_kernel(
            ts,
            _idx(stim_on[k]),
            pats["stimulus"],
            design["stimulus_peak_sd"] * g * value_scale[k],
            kernel,
        )
        if k > 0:
            update = _node_pattern(rng, n_nodes, n_active)
            amp = design["update_peak_sd"] * g * pe_scale[k - 1]
            _add_kernel(ts, _idx(stim_on[k]), update, amp, kernel)
        _add_kernel(
            ts,
            _idx(fb_on[k]),
            pats["feedback"],
            design["feedback_peak_sd"] * g * value_scale[k],
            kernel,
        )

    pre = max(1, int(round(float(pre_window_sec) / dt)))
    state_self, state_non = [], []
    for onset, trial_type in ev.loc[
        ev["trial_type"].isin(["self", "nonself"]), ["onset", "trial_type"]
    ].itertuples(index=False):
        i = _idx(onset)
        s = float(rng.normal())
        lo = max(0, i - pre)
        if i > lo and lo < n_time:
            ts[lo : min(i, n_time), :] += (
                design["internal_state_sd"] * s * pats["internal_state"]
            ).astype(ts.dtype)
        if trial_type == "self":
            amp = (
                design["self_peak_sd"]
                * g
                * max(0.0, 1.0 + design["self_state_coupling"] * s)
            )
            _add_kernel(ts, i, pats["self"], amp, kernel)
            state_self.append(s)
        else:
            pattern = _node_pattern(rng, n_nodes, n_active)
            _add_kernel(ts, i, pattern, design["nonself_peak_sd"] * g, kernel)
            state_non.append(s)
    return {
        "level": g,
        "n_trials": n_trials,
        "feedback_values": fb_val[:n_trials].tolist(),
        "internal_state_self": state_self,
        "internal_state_nonself": state_non,
        "kernel_peak_sec": float(np.argmax(kernel) * dt),
        "pre_window_samples": pre,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or b.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _event_projection(
    x: np.ndarray,
    onsets: np.ndarray,
    pattern: np.ndarray,
    *,
    dt: float,
    srpi_params: dict[str, Any],
) -> np.ndarray:
    pre = max(1, int(round(float(srpi_params["pre_window_sec"]) / dt)))
    lag = max(0, int(round(float(srpi_params["response_lag_sec"]) / dt)))
    win = max(1, int(round(float(srpi_params["response_window_sec"]) / dt)))
    weights = pattern / max(float(np.sum(np.abs(pattern))), 1e-12)
    vals = []
    for onset in onsets:
        i = int(round(float(onset) / dt))
        if i - pre < 0 or i + lag + win > x.shape[0]:
            continue
        diff = x[i + lag : i + lag + win].mean(axis=0) - x[i - pre : i].mean(axis=0)
        vals.append(float(diff @ weights))
    return np.asarray(vals, dtype=float)


def _manipulation_checks(
    task_ts: np.ndarray,
    events: pd.DataFrame,
    *,
    layout: dict[str, Any],
    modality: str,
    sample_interval: float,
    srpi_params: dict[str, Any],
    design: dict[str, Any] = PLANTED_DESIGN,
) -> dict[str, float]:
    """Metric-independent statistics that should increase with the planted level."""
    x = _normalize_nodes(task_ts).astype(np.float64)
    n_time, n_nodes = x.shape
    dt = float(sample_interval)
    eig = np.clip(np.linalg.eigvalsh(np.cov(x, rowvar=False)), 0.0, None)
    pr = float(np.sum(eig) ** 2 / max(float(np.sum(eig**2)), 1e-12)) / float(n_nodes)

    mods = np.asarray(layout["modules"], dtype=int)
    n_mod = int(layout["n_modules"])
    means = np.stack([x[:, mods == m].mean(axis=1) for m in range(n_mod)], axis=1)
    lag = max(1, int(round(0.5 * float(design["coupling_tau_sec"][modality]) / dt)))
    fwd, bwd = [], []
    if n_time > lag + 3 and n_mod >= 2:
        for m in range(n_mod):
            src = (m - 1) % n_mod
            fwd.append(_safe_corr(means[:-lag, src], means[lag:, m]))
            bwd.append(_safe_corr(means[:-lag, m], means[lag:, src]))
    coupling = float(np.nanmean(fwd) - np.nanmean(bwd)) if fwd else float("nan")

    ws = np.asarray(layout["workspace"], dtype=int)
    recv = np.setdiff1d(np.arange(n_nodes), ws)
    blag = int(design["broadcast_lag_samples"])
    wired = (
        x[:, ws] @ np.asarray(layout["broadcast_mix"]).T
    )  # planted input per receiver
    b_fwd, b_bwd = [], []
    if n_time > blag + 3:
        for j, r in enumerate(recv):
            b_fwd.append(_safe_corr(wired[:-blag, j], x[blag:, r]))
            b_bwd.append(_safe_corr(x[:-blag, r], wired[blag:, j]))
    broadcast = float(np.nanmean(b_fwd) - np.nanmean(b_bwd)) if b_fwd else float("nan")

    pats = layout["patterns"]
    on = {
        t: events.loc[events["trial_type"].eq(t), "onset"].to_numpy(dtype=float)
        for t in ("stimulus_target", "self", "nonself")
    }
    stim = _event_projection(
        x, on["stimulus_target"], pats["stimulus"], dt=dt, srpi_params=srpi_params
    )
    self_p = _event_projection(
        x, on["self"], pats["self"], dt=dt, srpi_params=srpi_params
    )
    non_p = _event_projection(
        x, on["nonself"], pats["self"], dt=dt, srpi_params=srpi_params
    )
    return {
        "participation_ratio": pr,
        "module_coupling_directionality": coupling,
        "broadcast_directionality": float(broadcast),
        "stimulus_evoked_projection": (
            float(np.mean(stim)) if stim.size else float("nan")
        ),
        "self_specific_projection": (
            float(np.mean(self_p) - np.mean(non_p))
            if self_p.size and non_p.size
            else float("nan")
        ),
    }


def _bids_label(text: str) -> str:
    """BIDS labels are alphanumeric ('selfother' + 'ses-1' -> 'selfotherses1')."""
    return re.sub(r"[^A-Za-z0-9]", "", str(text))


def _write_fmri_nifti(path: Path, ts: np.ndarray, tr: float) -> None:
    """Write the analysed node x time array as a (n_nodes, 1, 1, T) container."""
    if nib is None:
        raise RuntimeError("nibabel is required to write synthetic fMRI BIDS objects")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(ts, dtype=np.float32).T[:, None, None, :]
    img = nib.Nifti1Image(data, affine=np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0, float(tr)))
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(path))


def _write_brainvision_triplet(
    base: Path,
    ts: np.ndarray,
    sfreq: float,
    ch_names: list[str] | None = None,
) -> None:
    """Write the full analysed EEG array (no truncation) as BrainVision float32."""
    base.parent.mkdir(parents=True, exist_ok=True)
    n_ch = int(ts.shape[1])
    names = (
        list(ch_names)
        if ch_names is not None
        else [f"E{i:03d}" for i in range(1, n_ch + 1)]
    )
    eeg_path = base.with_suffix(".eeg")
    vhdr_path = base.with_suffix(".vhdr")
    vmrk_path = base.with_suffix(".vmrk")
    # BrainVision multiplexed float32, channel changes fastest per time sample.
    (np.asarray(ts, dtype=np.float64) * BIDS_EEG_MICROVOLTS_PER_UNIT).astype(
        "<f4"
    ).tofile(eeg_path)
    sampling_interval_us = 1_000_000.0 / float(sfreq)
    channels = "\n".join(f"Ch{i}={name},,1,uV" for i, name in enumerate(names, start=1))
    vhdr = f"""Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile={eeg_path.name}
MarkerFile={vmrk_path.name}
DataFormat=BINARY
DataOrientation=MULTIPLEXED
NumberOfChannels={n_ch}
SamplingInterval={sampling_interval_us:.6f}

[Binary Infos]
BinaryFormat=IEEE_FLOAT_32

[Channel Infos]
{channels}
"""
    vmrk = f"""Brain Vision Data Exchange Marker File, Version 1.0
[Common Infos]
DataFile={eeg_path.name}

[Marker Infos]
Mk1=New Segment,,1,1,0,synthetic
"""
    vhdr_path.write_text(vhdr, encoding="utf-8")
    vmrk_path.write_text(vmrk, encoding="utf-8")


def _write_dataset_description(root: Path, dataset_id: str, modality: str) -> None:
    _write_json(
        root / "dataset_description.json",
        {
            "Name": (
                "SYNTHETIC real-derived IMPaCT smoke-test objects for " f"{dataset_id}"
            ),
            "BIDSVersion": "1.10.0",
            "DatasetType": "raw",
            "GeneratedBy": [
                {
                    "Name": "generate_real_derived_synth_completed.py",
                    "Version": GENERATOR_VERSION,
                    "Description": OBJECT_DISCLAIMER,
                }
            ],
            "HowToAcknowledge": (
                "Cite the OpenNeuro source datasets listed in " "manifest.json."
            ),
            "SyntheticData": True,
            "Modality": modality,
        },
    )
    (root / "README_SYNTHETIC.md").write_text(
        "\n".join(
            [
                f"# {dataset_id} real-data-derived synthetic smoke-test objects",
                "",
                OBJECT_DISCLAIMER,
                "",
                PRIVACY_NOTE,
                "",
                "The analysed arrays live under test_objects/runs/; the BIDS files "
                "here hold exactly the same node x time data (fMRI as a node container "
                "NIfTI, not an anatomical image) plus the events that were planted.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_events(path: Path, events: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "onset",
        "duration",
        "trial_type",
        "reward",
        "outcome",
        "source_event_family",
        "synthetic_derivation",
    ]
    events = events.copy()
    for col in cols:
        if col not in events.columns:
            events[col] = np.nan
    events.loc[:, cols].to_csv(path, sep="\t", index=False, na_rep="n/a")


def _write_sidecar(path: Path, payload: dict[str, Any]) -> None:
    _write_json(
        path,
        payload
        | {
            "SyntheticData": True,
            "synthetic_derivation": "real_data_derived_synthetic_smoke_test",
        },
    )


def _eeg_bids_stem(subj: str, session: str) -> str:
    """Unique BrainVision stem per session ('deep' is the task-sed2 level)."""
    if session == "awake":
        return f"sub-{subj}_task-awake_acq-EC"
    if session == "deep":
        return f"sub-{subj}_task-sed2_acq-rest_run-1"
    return f"sub-{subj}_task-{_bids_label(session)}_acq-rest_run-1"


def _task_label_from_stem(stem: str) -> str:
    m = re.search(r"_task-([A-Za-z0-9]+)", stem)
    return m.group(1) if m else ""


def _source_record(
    dataset_id: str, alloc: dict[str, Any], duration_sec: float | None
) -> dict[str, Any]:
    return {
        "source_dataset": dataset_id,
        "source_path": _source_rel(alloc["path"], dataset_id),
        "segment_start_sec": float(alloc["start_sec"]),
        "segment_duration_sec": duration_sec,
        "reused": bool(alloc["reused"]),
        "shared_with": list(alloc["shared_with"]),
        "state_fallback": not bool(alloc["preferred_source"]),
    }


def _make_dataset(
    dataset_id: str,
    inspections: dict[str, dict[str, Any]],
    mid_events: pd.DataFrame,
    self_template: pd.DataFrame,
    other_template: pd.DataFrame,
    *,
    seed: int,
    params: dict[str, Any] | None = None,
    subjects: list[str] | None = None,
) -> dict[str, Any]:
    gen = dict(GENERATION_DEFAULTS)
    gen.update(params or {})
    iim_params = dict(IIM_VALIDATION_PARAMS)
    iim_params.update(gen.pop("iim_params", None) or {})
    dataset_root = DATASETS_OUT / dataset_id
    prep_root = RUNS_OUT / dataset_id / "preprocessed"
    truth_root = RUNS_OUT / dataset_id / "planted_truth"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    if prep_root.parent.exists():
        shutil.rmtree(prep_root.parent)
    dataset_root.mkdir(parents=True, exist_ok=True)
    prep_root.mkdir(parents=True, exist_ok=True)
    truth_root.mkdir(parents=True, exist_ok=True)

    if dataset_id == "ds005620":
        modality, atlas, condition = "eeg", "eeg64", "eeg"
        sfreq = float(gen["eeg_sfreq"])
        tr = 1.0 / sfreq
        srpi_params = EEG_SRPI_PARAMS
    else:
        modality, atlas = "fmri", "schaefer400"
        condition = "audio" if dataset_id == "ds003171" else "selfother"
        tr, sfreq = 2.0, None
        srpi_params = FMRI_SRPI_PARAMS
    all_subjects = _source_subjects(dataset_id)
    if subjects:
        wanted = [str(s).replace("sub-", "") for s in subjects]
        all_subjects = [s for s in all_subjects if s in wanted]
    if gen.get("max_subjects"):
        all_subjects = all_subjects[: int(gen["max_subjects"])]
    if not all_subjects:
        raise FileNotFoundError(
            f"No source subjects for {dataset_id} under "
            f"{_source_dataset_root(dataset_id)}"
        )
    sessions = _sessions_to_generate(dataset_id)
    levels = PLANTED_STATE_LEVELS[dataset_id]

    _write_dataset_description(dataset_root, dataset_id, modality)
    pd.DataFrame(
        {
            "participant_id": [f"sub-{s}" for s in all_subjects],
            "synthetic_data": [True] * len(all_subjects),
            "source_derivation": ["real_data_derived_synthetic_smoke_test"]
            * len(all_subjects),
        }
    ).to_csv(dataset_root / "participants.tsv", sep="\t", index=False)

    donor_subjects = _source_subjects("ds003171") if dataset_id == "ds002547" else []
    if dataset_id == "ds002547" and not donor_subjects:
        raise FileNotFoundError(
            "ds002547 rest baselines need ds003171 rest runs (donor), none found."
        )
    runs: list[dict[str, Any]] = []
    n_nodes_by_subject: dict[str, int] = {}
    donor_usage: dict[str, list[str]] = {}
    for subj_i, subj in enumerate(all_subjects):
        subj_rng = np.random.default_rng([int(seed), subj_i])
        donor = donor_subjects[subj_i % len(donor_subjects)] if donor_subjects else None
        if modality == "eeg":
            allocator = _SourceAllocator(
                float(gen["eeg_seconds"]), _brainvision_duration_sec
            )
        else:
            allocator = _SourceAllocator(None)
        rest_allocator = (
            _SourceAllocator(None) if dataset_id == "ds002547" else allocator
        )
        allocations = {}
        for ses in sessions:
            task_alloc = allocator.allocate(
                _source_candidates(dataset_id, subj, ses, rest=False), f"{ses}/task"
            )
            rest_alloc = rest_allocator.allocate(
                _source_candidates(
                    dataset_id, subj, ses, rest=True, donor_subject=donor
                ),
                f"{ses}/rest",
            )
            allocations[ses] = (task_alloc, rest_alloc)

        # Extract payloads.
        payloads: dict[str, dict[str, Any]] = {}
        if modality == "eeg":
            raw_segments = {}
            for ses, (task_alloc, rest_alloc) in allocations.items():
                for role, alloc in (("task", task_alloc), ("rest", rest_alloc)):
                    data, meta = _extract_brainvision_payload_timeseries(
                        alloc["path"],
                        target_sfreq=sfreq,
                        max_seconds=float(gen["eeg_seconds"]),
                        start_sec=alloc["start_sec"],
                        normalize=False,
                    )
                    raw_segments[(ses, role)] = (data, meta)
            # One channel set per subject: scalp EEG channels present and non-flat
            # in every segment used for this subject.
            first_meta = next(iter(raw_segments.values()))[1]
            common = [
                c
                for c in first_meta["channels"]
                if all(
                    c in m["channels"] and c not in m["flat_channels"]
                    for _, m in raw_segments.values()
                )
            ]
            if gen.get("eeg_max_channels"):
                common = common[: int(gen["eeg_max_channels"])]
            if len(common) < 4:
                raise RuntimeError(
                    f"sub-{subj}: fewer than 4 usable scalp EEG channels ({common})"
                )
            for (ses, role), (data, meta) in raw_segments.items():
                idx = [meta["channels"].index(c) for c in common]
                meta = dict(meta)
                meta["flat_channels_excluded"] = [
                    c for c in meta["flat_channels"] if c not in common
                ]
                meta["channels"] = common
                meta["extracted_nodes"] = len(common)
                payloads.setdefault(ses, {})[role] = (
                    _normalize_nodes(data[:, idx]),
                    meta,
                )
            channel_names = common
        else:
            channel_names = None
            for ses, (task_alloc, rest_alloc) in allocations.items():
                ses_rng = np.random.default_rng(
                    [int(seed), subj_i, sessions.index(ses)]
                )
                for role, alloc in (("task", task_alloc), ("rest", rest_alloc)):
                    data, meta = _extract_nifti_payload_timeseries(
                        alloc["path"], n_nodes=int(gen["fmri_nodes"]), rng=ses_rng
                    )
                    payloads.setdefault(ses, {})[role] = (data, meta)
        n_nodes = int(next(iter(payloads.values()))["task"][0].shape[1])
        n_nodes_by_subject[subj] = n_nodes
        layout = _subject_layout(n_nodes, subj_rng)

        for session_i, ses in enumerate(sessions):
            local_rng = np.random.default_rng([int(seed), subj_i, session_i, 1])
            level = float(levels[ses])
            task_alloc, rest_alloc = allocations[ses]
            (task_payload, task_meta), (rest_payload, rest_meta) = (
                payloads[ses]["task"],
                payloads[ses]["rest"],
            )
            events = _build_events(
                rng=local_rng,
                n_time=int(task_payload.shape[0]),
                tr=tr,
                modality=modality,
                mid_events=mid_events,
                self_template=self_template,
                other_template=other_template,
                session=ses,
            )
            ts, state_truth = _plant_state_structure(
                task_payload,
                level=level,
                modality=modality,
                sample_interval=tr,
                layout=layout,
                rng=local_rng,
            )
            event_truth = _plant_event_responses(
                ts,
                events,
                level=level,
                modality=modality,
                sample_interval=tr,
                layout=layout,
                rng=local_rng,
                pre_window_sec=float(srpi_params["pre_window_sec"]),
            )
            ts = _normalize_nodes(ts)
            rest, rest_truth = _plant_state_structure(
                rest_payload,
                level=level,
                modality=modality,
                sample_interval=tr,
                layout=layout,
                rng=local_rng,
            )
            rest = _normalize_nodes(rest)
            checks = _manipulation_checks(
                ts,
                events,
                layout=layout,
                modality=modality,
                sample_interval=tr,
                srpi_params=srpi_params,
            )

            task_dir = prep_root / subj / ses / condition
            rest_dir = prep_root / subj / ses / "rest"
            task_dir.mkdir(parents=True, exist_ok=True)
            rest_dir.mkdir(parents=True, exist_ok=True)
            task_npy = task_dir / f"{subj}_run-1_{atlas}_ts.npy"
            rest_npy = rest_dir / f"{subj}_run-1_{atlas}_ts.npy"
            np.save(task_npy, ts.astype(np.float32))
            np.save(rest_npy, rest.astype(np.float32))
            truth_path = truth_root / f"{subj}_{ses}.npz"
            np.savez_compressed(
                truth_path,
                level=level,
                modules=layout["modules"],
                workspace=layout["workspace"],
                broadcast_mix=layout["broadcast_mix"],
                internal_state_self=np.asarray(event_truth["internal_state_self"]),
                internal_state_nonself=np.asarray(
                    event_truth["internal_state_nonself"]
                ),
                **{f"pattern_{k}": v for k, v in layout["patterns"].items()},
            )

            task_ds = dataset_id
            rest_ds = "ds003171" if dataset_id == "ds002547" else dataset_id
            sources_used = sorted({task_ds, rest_ds, "ds005479", "ds002547"})
            if modality == "fmri":
                task = _bids_label(f"{condition}{ses}")
                func_dir = dataset_root / f"sub-{subj}" / "func"
                stem = f"sub-{subj}_task-{task}_run-1"
                events_path = func_dir / f"{stem}_events.tsv"
                data_path = func_dir / f"{stem}_bold.nii.gz"
                sidecar_path = func_dir / f"{stem}_bold.json"
                _write_events(events_path, events)
                _write_sidecar(
                    sidecar_path,
                    {
                        "TaskName": task,
                        "RepetitionTime": tr,
                        "SourcesInspected": sources_used,
                        "PlantedStateLevel": level,
                        "SyntheticNodeContainer": (
                            "Voxel i along the first axis is node i of the "
                            "analysed array; not an anatomical image and cannot "
                            "be parcellated."
                        ),
                    },
                )
                _write_fmri_nifti(data_path, ts, tr)
            else:
                eeg_dir = dataset_root / f"sub-{subj}" / "eeg"
                stem = _eeg_bids_stem(subj, ses)
                events_path = eeg_dir / f"{stem}_events.tsv"
                data_path = eeg_dir / f"{stem}_eeg.vhdr"
                sidecar_path = eeg_dir / f"{stem}_eeg.json"
                _write_events(events_path, events)
                _write_sidecar(
                    sidecar_path,
                    {
                        "TaskName": _task_label_from_stem(stem),
                        "SamplingFrequency": sfreq,
                        "RecordingDuration": float(ts.shape[0] / sfreq),
                        "EEGChannelCount": int(ts.shape[1]),
                        "EOGChannelCount": 0,
                        "EMGChannelCount": 0,
                        "EEGReference": "n/a (source reference; synthetic transform)",
                        "PowerLineFrequency": "n/a",
                        "SourcesInspected": sources_used,
                        "PlantedStateLevel": level,
                        "SyntheticMicrovoltsPerArrayUnit": BIDS_EEG_MICROVOLTS_PER_UNIT,
                    },
                )
                pd.DataFrame(
                    {
                        "name": channel_names,
                        "type": ["EEG"] * len(channel_names),
                        "units": ["uV"] * len(channel_names),
                    }
                ).to_csv(eeg_dir / f"{stem}_channels.tsv", sep="\t", index=False)
                _write_brainvision_triplet(
                    eeg_dir / f"{stem}_eeg", ts, sfreq, channel_names
                )
            seg = float(gen["eeg_seconds"]) if modality == "eeg" else None
            task_src = _source_record(task_ds, task_alloc, seg)
            rest_src = _source_record(rest_ds, rest_alloc, seg)
            if dataset_id == "ds002547":
                donor_usage.setdefault(rest_src["source_path"], []).append(
                    f"{subj}/{ses}"
                )
            runs.append(
                {
                    "subject": subj,
                    "session": ses,
                    "planted_level": level,
                    "task_source": task_src,
                    "rest_source": rest_src,
                    "task_array": _rel(task_npy, SSD_ROOT),
                    "rest_array": _rel(rest_npy, SSD_ROOT),
                    "task_shape": [int(v) for v in ts.shape],
                    "rest_shape": [int(v) for v in rest.shape],
                    "bids_events": _rel(events_path, SSD_ROOT),
                    "bids_data": _rel(data_path, SSD_ROOT),
                    "bids_sidecar": _rel(sidecar_path, SSD_ROOT),
                    "run_id": "1",
                    "planted_truth": _rel(truth_path, SSD_ROOT),
                    "planted_state_structure": state_truth,
                    "planted_rest_structure": rest_truth,
                    "planted_events": {
                        k: v
                        for k, v in event_truth.items()
                        if k not in {"feedback_values"}
                    },
                    "manipulation_checks": checks,
                    "n_events": int(len(events)),
                    "n_feedback": int(events["trial_type"].eq("feedback_reward").sum()),
                    "n_self": int(events["trial_type"].eq("self").sum()),
                    "n_nonself": int(events["trial_type"].eq("nonself").sum()),
                    "task_payload": task_meta,
                    "rest_payload": rest_meta,
                }
            )

    for rec in runs:
        if dataset_id == "ds002547":
            others = [
                u
                for u in donor_usage.get(rec["rest_source"]["source_path"], [])
                if not u.startswith(f"{rec['subject']}/")
            ]
            rec["rest_source"]["donor_payload_shared_with_other_subjects"] = others
    reuse = {
        "task_payload_reused": sum(r["task_source"]["reused"] for r in runs),
        "rest_payload_reused": sum(r["rest_source"]["reused"] for r in runs),
        "task_state_fallback": sum(r["task_source"]["state_fallback"] for r in runs),
        "rest_state_fallback": sum(r["rest_source"]["state_fallback"] for r in runs),
    }
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "generator_version": GENERATOR_VERSION,
        "object_kind": OBJECT_KIND,
        "disclaimer": OBJECT_DISCLAIMER,
        "privacy_note": PRIVACY_NOTE,
        "validation_scope": VALIDATION_SCOPE,
        "dataset_id": dataset_id,
        "synthetic": True,
        "derived_from_real_recordings": True,
        "paths_relative_to": "IMPACT_SYNTH_ROOT",
        "bids_root": _rel(dataset_root, SSD_ROOT),
        "preprocessed_root": _rel(prep_root, SSD_ROOT),
        "planted_truth_root": _rel(truth_root, SSD_ROOT),
        "source_paths_relative_to": "IMPACT_SOURCE_ROOT",
        "atlas": atlas,
        "atlas_label_note": (
            "'eeg64' is the pipeline's EEG file label; the node count is the number of "
            "scalp EEG channels kept (see n_nodes_by_subject)."
            if modality == "eeg"
            else (
                "'schaefer400' is the pipeline's file label; nodes are sampled real "
                "voxels, not parcels."
            )
        ),
        "condition": condition,
        "sessions": sessions,
        "validation_sessions": sessions,
        "subjects": all_subjects,
        "modality": modality,
        "sample_interval_seconds": tr,
        "n_nodes_by_subject": n_nodes_by_subject,
        "generation_parameters": {"seed": int(seed), **{k: v for k, v in gen.items()}},
        "metric_configuration": {
            "ram": EEG_RAM_PARAMS if modality == "eeg" else FMRI_RAM_PARAMS,
            "pdi": PDI_PARAMS,
            "pdi_primary_endpoint": "anchor",
            "pdi_require_strict_baseline": True,
            "nas": EEG_NAS_PARAMS if modality == "eeg" else FMRI_NAS_PARAMS,
            "srpi": srpi_params,
            "iim": iim_params,
            "iim_notes": IIM_CONFIG_NOTES,
            "ci": {"thetas": list(CI_THETAS), "ci_human_refs": None},
        },
        "planted_design": {
            "state_levels": levels,
            "ground_truth_direction": (
                "metric(session with higher planted level) > " "metric(lower level)"
            ),
            "known_answer_contrasts": KNOWN_ANSWER_CONTRASTS[dataset_id],
            "constants": PLANTED_DESIGN,
            "description": PLANTED_DESIGN_DESCRIPTION,
        },
        "runs": runs,
        "source_reuse_summary": reuse,
        "sources_used": sorted(
            {r["task_source"]["source_dataset"] for r in runs}
            | {r["rest_source"]["source_dataset"] for r in runs}
            | {"ds005479", "ds002547"}
        ),
        "source_inspection_report": _rel(
            INSPECTION_DIR / f"{dataset_id}_source_inspection.json", SSD_ROOT
        ),
        "donor_inspection_reports": [
            _rel(INSPECTION_DIR / f"{ds}_source_inspection.json", SSD_ROOT)
            for ds in DONORS
            if (INSPECTION_DIR / f"{ds}_source_inspection.json").exists()
        ],
        "derivation": {
            "base_timing": inspections.get(dataset_id, {}).get(
                "nifti" if modality == "fmri" else "eeg", {}
            ),
            "ram_event_donor": (
                "ds005479 MID onsets, labels and reward magnitudes (one "
                "events file)."
            ),
            "srpi_event_donor": (
                "ds002547 self/other event spacing (one events file " "per class)."
            ),
            "rest_donor": (
                "ds003171 rest runs of other subjects (ds002547 has no rest runs)"
                if dataset_id == "ds002547"
                else f"{dataset_id} rest runs of the same subject"
            ),
            "base_signal": (
                "Real NIfTI voxel time series (random voxels) or scalp BrainVision EEG "
                "channels; no generic AR/sinusoid fallback."
            ),
        },
        "limitations": [
            "Smoke-test objects: planted structure, not physiology, defines the state "
            "contrast.",
            "Metrics are computed with the reduced configuration in "
            "metric_configuration "
            "(notably IIM), not with run_pipeline defaults.",
            "fMRI nodes are randomly sampled voxels (spatial structure scrambled), "
            "labelled schaefer400.",
            "Events are a template from one ds005479 MID file and one ds002547 "
            "self/other layout, "
            "identical across subjects apart from jitter.",
            "ds002547 has no state design; awake/deep are pseudo-states defined only "
            "by the planted level.",
            "Source payload reuse and state fallbacks are flagged per run "
            "(source_reuse_summary).",
        ],
    }
    _write_json(dataset_root / "manifest.json", manifest)
    _write_json(REPORTS_OUT / f"{dataset_id}_manifest.json", manifest)
    return manifest


def _resolve_manifest_path(value: str | Path, synth_root: Path) -> Path:
    """Resolve a manifest path against the synth root.

    Relative paths (manifest v2) are joined to ``synth_root``. Legacy absolute
    paths are re-rooted at their 'test_objects' component under ``synth_root``
    first, so an extracted archive is validated even when the original
    machine's copy also exists; the absolute path is used only as a fallback.
    """
    p = Path(str(value))
    if not p.is_absolute():
        return (synth_root / p).resolve()
    parts = p.parts
    if "test_objects" in parts:
        i = parts.index("test_objects")
        rebased = synth_root.joinpath(*parts[i:])
        if rebased.exists():
            return rebased
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Manifest path {p} does not exist and cannot be re-rooted under {synth_root}. "
        "Set IMPACT_SYNTH_ROOT (or --synth-root) to the folder that contains "
        "test_objects/."
    )


def _runs_for_validation(
    manifest: dict[str, Any], synth_root: Path
) -> list[dict[str, Any]]:
    """Runs from a v2 manifest, or reconstructed from a legacy (v1) manifest."""
    if manifest.get("runs"):
        return [dict(r) for r in manifest["runs"]]
    prep_root = _resolve_manifest_path(manifest["preprocessed_root"], synth_root)
    atlas = str(manifest["atlas"])
    condition = str(manifest["condition"])
    sessions = manifest.get(
        "validation_sessions", manifest.get("sessions", ["awake", "deep"])
    )
    runs = []
    for subj in manifest["subjects"]:
        for ses in sessions:
            runs.append(
                {
                    "subject": str(subj),
                    "session": str(ses),
                    "task_array": prep_root
                    / str(subj)
                    / str(ses)
                    / condition
                    / f"{subj}_run-1_{atlas}_ts.npy",
                    "rest_array": prep_root
                    / str(subj)
                    / str(ses)
                    / "rest"
                    / f"{subj}_run-1_{atlas}_ts.npy",
                    "run_id": "1",
                }
            )
    return runs


def _events_bundle_from_file(events_path: Path) -> dict[str, Any]:
    """Parse one events.tsv with the pipeline's own event parser."""
    parser = getattr(_run_synergy_ci, "_events_to_ram_bundle", None)
    if parser is None:
        raise RuntimeError(
            "impact_pipeline.run_synergy_ci._events_to_ram_bundle is unavailable; "
            "cannot parse the exact events file of each run."
        )
    return parser(Path(events_path))


def _bids_array_consistency(
    run: dict[str, Any], ts: np.ndarray, synth_root: Path, tr: float, modality: str
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    events = _resolve_manifest_path(run["bids_events"], synth_root)
    checks["bids_events_exist"] = events.exists()
    data_path = _resolve_manifest_path(run["bids_data"], synth_root)
    sidecar = _read_json(_resolve_manifest_path(run["bids_sidecar"], synth_root))
    try:
        if modality == "eeg":
            raw = mne.io.read_raw_brainvision(
                str(data_path), preload=True, verbose="ERROR"
            )
            bids = raw.get_data().T * 1e6 / BIDS_EEG_MICROVOLTS_PER_UNIT
            checks["bids_sample_interval_matches"] = bool(
                np.isclose(1.0 / float(sidecar.get("SamplingFrequency", np.nan)), tr)
            )
        else:
            img = nib.load(str(data_path))
            bids = np.asarray(img.dataobj, dtype=np.float32)[:, 0, 0, :].T
            checks["bids_sample_interval_matches"] = bool(
                np.isclose(float(sidecar.get("RepetitionTime", np.nan)), tr)
            )
        checks["bids_data_matches_array"] = bool(
            bids.shape == ts.shape and np.allclose(bids, ts, atol=1e-4)
        )
    except Exception:
        checks["bids_data_matches_array"] = False
        checks.setdefault("bids_sample_interval_matches", False)
    stem_task = _task_label_from_stem(Path(run["bids_data"]).name)
    checks["bids_task_label_valid"] = bool(stem_task) and stem_task == str(
        sidecar.get("TaskName", "")
    )
    return checks


def _sign_test_p(n_pos: int, n_neg: int) -> float:
    n = int(n_pos) + int(n_neg)
    if n == 0:
        return float("nan")
    return float(stats.binomtest(int(n_pos), n, 0.5).pvalue)


def _paired_summary(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    arr = np.asarray(pairs, dtype=float).reshape(-1, 2)
    ok = np.all(np.isfinite(arr), axis=1)
    diffs = arr[ok, 0] - arr[ok, 1]
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    return {
        "n_pairs": int(arr.shape[0]),
        "n_defined_pairs": int(ok.sum()),
        "n_first_greater": n_pos,
        "n_second_greater": n_neg,
        "n_ties": int(ok.sum()) - n_pos - n_neg,
        "median_difference": float(np.median(diffs)) if diffs.size else None,
        "sign_test_p_two_sided": _sign_test_p(n_pos, n_neg),
    }


def _planted_outcome(summary: dict[str, Any], alpha: float = KNOWN_ANSWER_ALPHA) -> str:
    if summary["n_defined_pairs"] == 0:
        return "undefined"
    n_pos, n_neg = summary["n_first_greater"], summary["n_second_greater"]
    p = summary["sign_test_p_two_sided"]
    if n_pos > n_neg:
        return (
            "recovered"
            if (np.isfinite(p) and p < alpha)
            else "direction_only_not_significant"
        )
    if n_neg > n_pos and np.isfinite(p) and p < alpha:
        return "reversed"
    return "not_recovered"


def _null_outcome(summary: dict[str, Any], alpha: float = KNOWN_ANSWER_ALPHA) -> str:
    if summary["n_defined_pairs"] == 0:
        return "undefined"
    p = summary["sign_test_p_two_sided"]
    if np.isfinite(p) and p < alpha:
        return "systematic_difference_without_planted_difference"
    return "no_systematic_difference"


def _evaluate_known_answers(
    values: dict[str, dict[str, dict[str, float]]],
    *,
    levels: dict[str, float],
    contrasts: dict[str, Any],
    metrics: list[str],
    alpha: float = KNOWN_ANSWER_ALPHA,
) -> dict[str, Any]:
    """Recovery of planted direction per metric.

    ``values[metric][subject][session]`` holds metric values (NaN = undefined).
    """
    out: dict[str, Any] = {
        "available": True,
        "alpha": alpha,
        "test": (
            "exact two-sided sign test over subjects (ties and undefined pairs "
            "dropped)"
        ),
        "state_levels": levels,
        "contrasts": {},
        "monotonic": {},
    }
    planted = contrasts.get("planted")
    null = contrasts.get("null")
    for kind, pair in (("planted", planted), ("null", null)):
        if not pair:
            continue
        a, b = pair
        entry = {
            "sessions": [a, b],
            "planted_levels": [levels.get(a), levels.get(b)],
            "expected": (
                f"{a} > {b}"
                if kind == "planted"
                else "no systematic difference (equal planted level)"
            ),
            "metrics": {},
        }
        for metric in metrics:
            by_subj = values.get(metric, {})
            pairs = [
                (by_subj[s].get(a, np.nan), by_subj[s].get(b, np.nan))
                for s in sorted(by_subj)
            ]
            summ = _paired_summary(pairs)
            summ["outcome"] = (
                _planted_outcome(summ, alpha)
                if kind == "planted"
                else _null_outcome(summ, alpha)
            )
            entry["metrics"][metric] = summ
        out["contrasts"][kind] = entry
    ordered = sorted(levels)
    if len(set(levels.values())) >= 3:
        for metric in metrics:
            rhos = []
            for subj, by_ses in values.get(metric, {}).items():
                x = [levels[s] for s in ordered if np.isfinite(by_ses.get(s, np.nan))]
                y = [by_ses[s] for s in ordered if np.isfinite(by_ses.get(s, np.nan))]
                if len(x) >= 3 and len(set(x)) >= 2 and np.std(y) > 0:
                    rho = stats.spearmanr(x, y)[0]
                    if np.isfinite(rho):
                        rhos.append(float(rho))
            out["monotonic"][metric] = {
                "mean_spearman_level_vs_metric": float(np.mean(rhos)) if rhos else None,
                "n_subjects": len(rhos),
            }
    return out


def _validate_dataset(
    manifest: dict[str, Any],
    *,
    out_dir: Path | None = None,
    synth_root: Path | None = None,
    iim_params: dict[str, Any] | None = None,
    subjects: list[str] | None = None,
) -> dict[str, Any]:
    dataset_id = manifest["dataset_id"]
    synth_root = Path(synth_root or SSD_ROOT)
    bids_root = _resolve_manifest_path(manifest["bids_root"], synth_root)
    prep_root = _resolve_manifest_path(manifest["preprocessed_root"], synth_root)
    atlas = str(manifest["atlas"])
    condition = str(manifest["condition"])
    tr = float(manifest["sample_interval_seconds"])
    modality = str(manifest["modality"])
    legacy = int(manifest.get("manifest_version", 1)) < 2
    runs = _runs_for_validation(manifest, synth_root)
    if subjects:
        wanted = {str(s).replace("sub-", "") for s in subjects}
        runs = [r for r in runs if str(r["subject"]) in wanted]
    if not runs:
        raise ValueError(
            f"{dataset_id}: no runs to validate (subject filter {subjects})"
        )
    subj_list = list(dict.fromkeys(str(r["subject"]) for r in runs))
    sessions = tuple(dict.fromkeys(str(r["session"]) for r in runs))
    cfg = dict(
        manifest.get("metric_configuration", {}).get("iim") or IIM_VALIDATION_PARAMS
    )
    cfg.update(iim_params or {})
    out_dir = Path(out_dir or (REPORTS_OUT / dataset_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    readiness_csv = out_dir / "readiness_after.csv"
    readiness_json = out_dir / "readiness_after.json"
    df_ready, ready_summary = check_mpc_readiness(
        prep_root=str(prep_root),
        bids_root=str(bids_root),
        atlas=atlas,
        condition=condition,
        sessions=sessions,
        subjects=subj_list,
        require_explicit_feedback=True,
        require_explicit_srpi=True,
        srpi_min_events_per_class=3,
        iim_bins=int(cfg["iim_bins"]),
        iim_lag_trs=int(cfg["iim_lag_trs"]),
        iim_max_state_space=int(cfg.get("iim_max_state_space", 1500)),
        iim_max_nodes=cfg.get("iim_max_nodes"),
    )
    df_ready.to_csv(readiness_csv, index=False)
    _write_json(readiness_json, ready_summary)

    onsets: dict[str, dict[str, tuple]] = {}
    resolution = []
    resolver = getattr(_run_synergy_ci, "_resolve_events_file", None)
    for run in runs:
        subj, ses = str(run["subject"]), str(run["session"])
        resolved = (
            resolver(str(bids_root), subj, ses, condition=condition)
            if resolver
            else None
        )
        if legacy:
            # Legacy manifests do not record the events file of each run, so the
            # pipeline resolver is used and consistency cannot be checked.
            bundle, run_id = load_onsets(str(bids_root), subj, ses, condition=condition)
            expected = None
        else:
            expected = _resolve_manifest_path(run["bids_events"], synth_root)
            bundle, run_id = _events_bundle_from_file(expected), str(
                run.get("run_id", "1")
            )
        onsets.setdefault(subj, {})[ses] = (bundle, run_id)
        resolution.append(
            {
                "subject": subj,
                "session": ses,
                "expected_events": _rel(expected, synth_root) if expected else None,
                "pipeline_resolver_events": (
                    _rel(resolved, synth_root) if resolved else None
                ),
                "consistent": (
                    None
                    if expected is None
                    else bool(
                        resolved is not None
                        and Path(resolved).resolve() == Path(expected).resolve()
                    )
                ),
            }
        )

    nas_params = EEG_NAS_PARAMS if modality == "eeg" else FMRI_NAS_PARAMS
    srpi_params = EEG_SRPI_PARAMS if modality == "eeg" else FMRI_SRPI_PARAMS
    ram_params = EEG_RAM_PARAMS if modality == "eeg" else FMRI_RAM_PARAMS
    df_ci = compute_synergy_ci(
        str(prep_root),
        atlas,
        thetas=list(CI_THETAS),
        sessions=sessions,
        condition=condition,
        tr=tr,
        stimulus_onsets=onsets,
        subjects=subj_list,
        ram_params=ram_params,
        pdi_params=PDI_PARAMS,
        pdi_require_explicit_params=True,
        pdi_require_strict_baseline=True,
        pdi_primary_endpoint="anchor",
        nas_params=nas_params,
        srpi_params=srpi_params,
        srpi_require_explicit_params=True,
        iim_bins=int(cfg["iim_bins"]),
        iim_lag_trs=int(cfg["iim_lag_trs"]),
        iim_max_timepoints=cfg.get("iim_max_timepoints"),
        iim_max_nodes=cfg.get("iim_max_nodes"),
        iim_max_mechanism_size=cfg.get("iim_max_mechanism_size"),
        iim_max_purview_size=cfg.get("iim_max_purview_size"),
        iim_parallel_workers=1,
        iim_enable_parallel=False,
        iim_use_shared_memory=False,
        iim_phase1_parallel_workers=1,
        iim_phase1_chunk_size=4,
        iim_phase1_shared_memory=False,
        iim_checkpoint_dir=str(out_dir / "iim_checkpoints"),
        iim_resume_checkpoint=True,
        compute_mpc=True,
        compute_ci=True,
        dataset_id=dataset_id,
        data_origin=DUMMY_DATA_ORIGIN,
        dataset_role="real_data_derived_synthetic_smoke_test",
        provenance_label="synthetic_validation",
        modality=modality,
    )
    df_ci.to_csv(out_dir / "actual_metric_computation.csv", index=False)

    srpi_details, ram_details, rows = [], [], []
    optional = [m for m in KNOWN_ANSWER_OPTIONAL_METRICS if m in df_ci.columns]
    metrics_all = list(KNOWN_ANSWER_METRICS) + optional
    values: dict[str, dict[str, dict[str, float]]] = {m: {} for m in metrics_all}
    for run in runs:
        subj, ses = str(run["subject"]), str(run["session"])
        checks: dict[str, bool] = {}
        task_path = _resolve_manifest_path(run["task_array"], synth_root)
        rest_path = _resolve_manifest_path(run["rest_array"], synth_root)
        checks["task_array_exists"] = task_path.exists()
        checks["rest_array_exists"] = rest_path.exists()
        arr = np.load(task_path)
        rest = np.load(rest_path)
        checks["arrays_finite"] = bool(
            np.isfinite(arr).all() and np.isfinite(rest).all()
        )
        checks["no_zero_variance_nodes"] = bool(
            np.all(arr.std(axis=0) > 1e-8) and np.all(rest.std(axis=0) > 1e-8)
        )
        checks["task_rest_node_count_match"] = bool(arr.shape[1] == rest.shape[1])
        if not legacy:
            checks["task_shape_matches_manifest"] = list(arr.shape) == list(
                run["task_shape"]
            )
            checks.update(_bids_array_consistency(run, arr, synth_root, tr, modality))
        ts = arr.T
        bundle = onsets[subj][ses][0]
        srpi = compute_SRPI(
            ts,
            tr=tr,
            self_onsets=bundle.get("self_onsets", []),
            nonself_onsets=bundle.get("nonself_onsets", []),
            return_details=True,
            **srpi_params,
        )
        ram = compute_RAM(
            ts, tr=tr, stimulus_onsets=bundle, return_details=True, **ram_params
        )
        srpi_details.append({"subject": subj, "session": ses, **srpi})
        ram_details.append({"subject": subj, "session": ses, **ram})
        sel = df_ci[
            (df_ci["subject"].astype(str) == subj)
            & (df_ci["session"].astype(str) == ses)
        ]
        if "theta" in sel.columns and len(sel) > 1:
            sel = sel[np.isclose(sel["theta"].astype(float), float(CI_THETAS[0]))]
        rec = sel.iloc[0].to_dict() if len(sel) else {}
        checks["metric_row_present"] = bool(len(sel))
        metric_values = {}
        for metric in metrics_all:
            try:
                val = float(rec.get(metric, np.nan))
            except (TypeError, ValueError):
                val = float("nan")
            metric_values[metric] = val
            values[metric].setdefault(subj, {})[ses] = val
        for metric in KNOWN_ANSWER_METRICS:
            val = metric_values[metric]
            checks[f"{metric}_column_present"] = metric in df_ci.columns
            checks[f"{metric}_defined"] = bool(np.isfinite(val))
            lo, hi = METRIC_BOUNDS[metric]
            in_bounds = (not np.isfinite(val)) or (
                (lo is None or val >= lo - 1e-12) and (hi is None or val <= hi + 1e-12)
            )
            checks[f"{metric}_within_documented_bounds"] = bool(in_bounds)
        rows.append(
            {
                "subject": subj,
                "session": ses,
                "planted_level": run.get("planted_level"),
                "metric_values": metric_values,
                "checks": checks,
            }
        )

    _write_json(out_dir / "srpi_details.json", {"rows": srpi_details})
    _write_json(out_dir / "ram_details.json", {"rows": ram_details})
    ready_fraction = (
        ready_summary.get("metrics", {}).get("CI", {}).get("ready_fraction")
    )
    failed = sorted({k for row in rows for k, ok in row["checks"].items() if not ok})
    smoke = {
        "passed": bool(ready_fraction == 1.0 and not failed),
        "all_ready": bool(ready_fraction == 1.0),
        "ready_fraction_CI": ready_fraction,
        "n_rows": len(rows),
        "failed_checks": failed,
        "metric_definedness": {
            m: int(sum(np.isfinite(r["metric_values"][m]) for r in rows))
            for m in KNOWN_ANSWER_METRICS
        },
    }

    if legacy or "planted_design" not in manifest:
        known = {
            "available": False,
            "reason": (
                "legacy manifest without planted ground truth (generated before "
                "generator 2.0.0)"
            ),
        }
        planted_verified = None
        manip = None
    else:
        design = manifest["planted_design"]
        levels = {
            k: float(v) for k, v in design["state_levels"].items() if k in sessions
        }
        known = _evaluate_known_answers(
            values,
            levels=levels,
            contrasts=design["known_answer_contrasts"],
            metrics=metrics_all,
        )
        manip_values = {c: {} for c in MANIPULATION_CHECKS}
        for run in runs:
            for c in MANIPULATION_CHECKS:
                val = run.get("manipulation_checks", {}).get(c, np.nan)
                manip_values[c].setdefault(str(run["subject"]), {})[
                    str(run["session"])
                ] = float(val)
        manip = _evaluate_known_answers(
            manip_values,
            levels=levels,
            contrasts=design["known_answer_contrasts"],
            metrics=list(MANIPULATION_CHECKS),
        )
        planted_pair = manip["contrasts"].get("planted", {}).get("metrics", {})
        planted_verified = bool(planted_pair) and all(
            (s.get("median_difference") or 0.0) > 0 for s in planted_pair.values()
        )
    report = {
        "dataset_id": dataset_id,
        "object_kind": OBJECT_KIND,
        "disclaimer": OBJECT_DISCLAIMER,
        "validation_scope": VALIDATION_SCOPE,
        "manifest_version": int(manifest.get("manifest_version", 1)),
        "iim_configuration": cfg,
        "readiness_csv": _rel(readiness_csv, synth_root),
        "readiness_json": _rel(readiness_json, synth_root),
        "actual_metric_csv": _rel(
            out_dir / "actual_metric_computation.csv", synth_root
        ),
        "smoke_test": smoke,
        "planted_structure_verified": planted_verified,
        "manipulation_checks": manip,
        "known_answer": known,
        "pipeline_event_resolution": {
            "consistent": (
                None if legacy else all(r["consistent"] for r in resolution)
            ),
            "note": (
                "Metrics are computed from the exact events file of each run; this "
                "records whether run_synergy_ci._resolve_events_file picks the same "
                "file."
            ),
            "runs": resolution,
        },
        "rows": rows,
    }
    _write_json(out_dir / "actual_metric_report.json", report)
    return report


def _known_answer_digest(validation: dict[str, Any]) -> dict[str, Any]:
    digest = {}
    for ds, rep in validation.items():
        ka = rep.get("known_answer", {})
        if not ka.get("available"):
            digest[ds] = {"available": False}
            continue
        digest[ds] = {
            kind: {m: s["outcome"] for m, s in entry["metrics"].items()}
            for kind, entry in ka.get("contrasts", {}).items()
        }
    return digest


def _write_summary(
    path: Path,
    manifests: list[dict[str, Any]],
    validation: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    verified = [v.get("planted_structure_verified") for v in validation.values()]
    payload = {
        "object_kind": OBJECT_KIND,
        "disclaimer": OBJECT_DISCLAIMER,
        "validation_scope": VALIDATION_SCOPE,
        "generator_version": GENERATOR_VERSION,
        "mode": mode,
        "targets": [m["dataset_id"] for m in manifests],
        "manifests": [
            _rel(REPORTS_OUT / f"{m['dataset_id']}_manifest.json", SSD_ROOT)
            for m in manifests
        ],
        "validation": validation,
        "smoke_test_passed": (
            bool(validation)
            and all(v["smoke_test"]["passed"] for v in validation.values())
        ),
        "planted_structure_verified": (
            None
            if (not verified or any(v is None for v in verified))
            else all(verified)
        ),
        "known_answer_summary": _known_answer_digest(validation),
        "known_answer_note": (
            "Known-answer outcomes are observations of whether each metric recovers "
            "the planted direction; they are not a pass criterion and were not tuned."
        ),
        "synthetic": True,
        "derived_from_real_recordings": True,
    }
    _write_json(path, payload)
    return payload


def _write_donor_report(
    inspections: dict[str, dict[str, Any]], manifests: list[dict[str, Any]]
) -> None:
    used = sorted({ds for m in manifests for ds in m.get("sources_used", [])})
    payload = {
        "synthetic_targets": [m["dataset_id"] for m in manifests],
        "source_inspection_summary": _rel(
            INSPECTION_DIR / "source_inspection_summary.json", SSD_ROOT
        ),
        "sources_read_by_generator": used,
        "not_used_by_generator": ["ds004295", "ds002336", "ds006623", "ds002685"],
        "donor_statistics": {
            ds: {
                "file_counts": inspections.get(ds, {}).get("file_counts"),
                "n_subjects": inspections.get(ds, {}).get("n_subjects"),
                "tasks": inspections.get(ds, {}).get("tasks"),
                "events": inspections.get(ds, {})
                .get("events", {})
                .get("trial_type_counts_total"),
                "nifti": {
                    "tr": inspections.get(ds, {}).get("nifti", {}).get("tr"),
                    "n_volumes": inspections.get(ds, {})
                    .get("nifti", {})
                    .get("n_volumes"),
                    "acf1_means": inspections.get(ds, {})
                    .get("nifti", {})
                    .get("acf1_means"),
                },
                "eeg": {
                    "sampling_frequency": inspections.get(ds, {})
                    .get("eeg", {})
                    .get("sampling_frequency"),
                    "n_channels": inspections.get(ds, {})
                    .get("eeg", {})
                    .get("n_channels"),
                    "duration_sec": inspections.get(ds, {})
                    .get("eeg", {})
                    .get("duration_sec"),
                },
            }
            for ds in used
        },
        "limitations": [
            "The package is compact and is not an OpenNeuro mirror.",
            "Only the datasets in sources_read_by_generator are read.",
        ],
    }
    _write_json(REPORTS_OUT / "donor_statistics_report.json", payload)


def _link_repo_paths() -> None:
    link_specs = [
        (REPO_ROOT / DATASETS_REL, DATASETS_OUT),
        (REPO_ROOT / RUNS_REL, RUNS_OUT),
    ]
    for link, target in link_specs:
        link.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            link.unlink()
        if link.exists():
            continue
        link.symlink_to(target, target_is_directory=True)


def _load_manifest(dataset_id: str) -> dict[str, Any]:
    for path in (
        REPORTS_OUT / f"{dataset_id}_manifest.json",
        DATASETS_OUT / dataset_id / "manifest.json",
    ):
        if path.exists():
            return _read_json(path)
    raise FileNotFoundError(
        f"No manifest for {dataset_id} under {REPORTS_OUT} or "
        f"{DATASETS_OUT / dataset_id}. "
        "Set IMPACT_SYNTH_ROOT (or --synth-root) to the folder that contains "
        "test_objects/."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or re-validate the real-data-derived synthetic smoke-test "
            "objects. " + OBJECT_DISCLAIMER
        )
    )
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Re-validate existing objects under --synth-root/IMPACT_SYNTH_ROOT. "
            "Shipped reports "
            "are left untouched; results go to --validation-out."
        ),
    )
    parser.add_argument(
        "--synth-root",
        default=None,
        help=(
            "Folder that contains test_objects/ (default: IMPACT_SYNTH_ROOT, then the "
            "repository)."
        ),
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help=(
            "Parent directory containing downloaded source datasets, for example "
            "/data/openneuro_sources with ds003171, ds002547, and other dataset "
            "folders inside. "
            "Defaults to IMPACT_SOURCE_ROOT, then IMPACT_SYNTH_ROOT/data/scratch."
        ),
    )
    parser.add_argument(
        "--inspection-dir", default=None, help="Where <ds>_source_inspection.json live."
    )
    parser.add_argument(
        "--validation-out",
        default=None,
        help=(
            "Output folder for --validate-only (default: "
            "<synth-root>/test_objects/real_derived_synth_completed/revalidation/<UTC "
            "time>)."
        ),
    )
    parser.add_argument(
        "--datasets", nargs="*", default=list(TARGETS), choices=list(TARGETS)
    )
    parser.add_argument(
        "--subjects", nargs="*", default=None, help="Subject IDs to generate/validate."
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--fmri-nodes", type=int, default=GENERATION_DEFAULTS["fmri_nodes"]
    )
    parser.add_argument(
        "--eeg-seconds", type=float, default=GENERATION_DEFAULTS["eeg_seconds"]
    )
    parser.add_argument(
        "--eeg-sfreq", type=float, default=GENERATION_DEFAULTS["eeg_sfreq"]
    )
    parser.add_argument("--eeg-max-channels", type=int, default=None)
    parser.add_argument("--iim-bins", type=int, default=None)
    parser.add_argument("--iim-max-nodes", type=int, default=None)
    parser.add_argument("--iim-max-timepoints", type=int, default=None)
    parser.add_argument("--iim-max-mechanism-size", type=int, default=None)
    parser.add_argument("--iim-max-purview-size", type=int, default=None)
    parser.add_argument(
        "--no-link-repo-paths",
        action="store_true",
        help=(
            "Do not symlink the repository test_objects/ folders to an external "
            "--synth-root."
        ),
    )
    return parser.parse_args(argv)


def _iim_overrides(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for key in (
        "bins",
        "max_nodes",
        "max_timepoints",
        "max_mechanism_size",
        "max_purview_size",
    ):
        val = getattr(args, f"iim_{key}")
        if val is not None:
            out[f"iim_{key}"] = int(val)
    return out


def _print_outcome(summary: dict[str, Any]) -> None:
    print(f"smoke_test_passed={summary['smoke_test_passed']}")
    print(f"planted_structure_verified={summary['planted_structure_verified']}")
    for ds, rep in summary["validation"].items():
        failed = rep["smoke_test"]["failed_checks"]
        status = "passed" if rep["smoke_test"]["passed"] else "FAILED"
        print(f"  {ds}: smoke_test={status}")
        if failed:
            print(f"    failed checks: {', '.join(failed)}")
        for kind, outcomes in summary["known_answer_summary"].get(ds, {}).items():
            if isinstance(outcomes, dict):
                text = ", ".join(f"{m}={o}" for m, o in outcomes.items())
                print(f"    known-answer {kind}: {text}")
    print(summary["known_answer_note"])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_paths(args.synth_root, args.source_root, args.inspection_dir)
    datasets = [ds for ds in TARGETS if ds in set(args.datasets)]
    iim_over = _iim_overrides(args)

    if not SSD_ROOT.exists():
        raise FileNotFoundError(f"Synthetic output root not found: {SSD_ROOT}")
    if args.validate_only:
        manifests = [_load_manifest(ds) for ds in datasets]
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_root = (
            Path(args.validation_out).expanduser().resolve()
            if args.validation_out
            else SSD_ROOT / REPORTS_REL.parent / "revalidation" / stamp
        )
        validation = {
            m["dataset_id"]: _validate_dataset(
                m,
                out_dir=out_root / m["dataset_id"],
                synth_root=SSD_ROOT,
                iim_params=iim_over,
                subjects=args.subjects,
            )
            for m in manifests
        }
        summary = _write_summary(
            out_root / SUMMARY_NAME, manifests, validation, mode="validate_only"
        )
        print(f"re-validated synthetic smoke-test objects -> {out_root}")
        _print_outcome(summary)
        return 0 if summary["smoke_test_passed"] else 1

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source dataset root not found: {SOURCE_ROOT}")
    REPORTS_OUT.mkdir(parents=True, exist_ok=True)
    inspections = _load_source_inspections(datasets)
    mid_events = _load_mid_events()
    self_template, other_template = _load_self_other_templates()
    _write_json(
        REPORTS_OUT / "source_file_inspection_report.json",
        {
            "source_paths_relative_to": "IMPACT_SOURCE_ROOT",
            "inspections": {
                ds: _rel(INSPECTION_DIR / f"{ds}_source_inspection.json", SSD_ROOT)
                for ds in inspections
            },
            "templates_read": {
                "mid_events": str(mid_events["source_file"].iloc[0]),
                "self_events": str(self_template["source_file"].iloc[0]),
                "other_events": str(other_template["source_file"].iloc[0]),
            },
        },
    )
    params = {
        "fmri_nodes": int(args.fmri_nodes),
        "eeg_seconds": float(args.eeg_seconds),
        "eeg_sfreq": float(args.eeg_sfreq),
        "eeg_max_channels": args.eeg_max_channels,
        "max_subjects": args.max_subjects,
        "iim_params": iim_over,
    }
    manifests = []
    for dataset_id in datasets:
        offset = TARGETS.index(dataset_id)
        manifests.append(
            _make_dataset(
                dataset_id,
                inspections,
                mid_events,
                self_template,
                other_template,
                seed=int(args.seed) + 1000 * offset,
                params=params,
                subjects=args.subjects,
            )
        )
    _write_donor_report(inspections, manifests)
    if not args.no_link_repo_paths:
        _link_repo_paths()

    validation = {}
    if not args.skip_validation:
        for manifest in manifests:
            validation[manifest["dataset_id"]] = _validate_dataset(
                manifest, synth_root=SSD_ROOT
            )
    summary = _write_summary(
        REPORTS_OUT / SUMMARY_NAME, manifests, validation, mode="generate"
    )
    print(f"wrote synthetic datasets -> {DATASETS_OUT}")
    print(f"wrote synthetic runs -> {RUNS_OUT}")
    print(f"wrote reports -> {REPORTS_OUT}")
    if validation:
        _print_outcome(summary)
        return 0 if summary["smoke_test_passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
