import os
import glob
import json
import logging
import concurrent.futures
import subprocess
import hashlib
import gc
from collections import deque
from multiprocessing import shared_memory

import numpy as np
import pandas as pd

from impact_pipeline import evidence as mpc_evidence
from impact_pipeline import nulls
from impact_pipeline.hardware_backend import (
    backend_summary,
    configure_process_for_hardware,
)
from impact_pipeline.mpc_metrics import (  # noqa: F401 (compute_CI re-exported)
    SURROGATE_METHODS,
    compute_RAM,
    compute_PDI,
    compute_NAS,
    compute_IIM,
    compute_SRPI,
    compute_CI,
    _compute_ci_legacy,
)
from impact_pipeline.provenance import (
    PROVENANCE_COLUMNS,
    REAL_DATA_ORIGIN,
    dataset_role_for_origin,
    normalize_data_origin,
    provenance_label_for_origin,
)
from impact_pipeline.utils import HypergraphSynergy

# Keep raw IIM display in native (unscaled) units.
IIM_DISPLAY_SCALE_DEFAULT = 1.0
log = logging.getLogger(__name__)

RAM_PARAM_DEFAULTS = {
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
    "quality_ridge": 1.0,
    "require_explicit_feedback": True,
    "require_explicit_goals": True,
    "quality_response_estimate": "auto",
    "quality_lag_sec": None,
    "quality_cv_folds": 5,
    "quality_null_samples": 200,
    "quality_random_state": 0,
}
RAM_PARAM_KEYS = tuple(RAM_PARAM_DEFAULTS.keys())

PDI_PARAM_DEFAULTS = {
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
PDI_PARAM_KEYS = tuple(PDI_PARAM_DEFAULTS.keys())

NAS_PARAM_DEFAULTS = {
    "zthr": 1.0,
    "eps": 0.2,
    "tau": None,
    "lambda_phase": 0.5,
    "alpha": 0.20,
    "beta": 0.16,
    "gamma": 0.14,
    "delta": 0.12,
    "eta": 0.16,
    "zeta": 0.12,
    "rho": 0.10,
    "bands": None,
    "band_weights": None,
    "window_len": None,
    "step_len": None,
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
NAS_PARAM_KEYS = tuple(NAS_PARAM_DEFAULTS.keys())

SRPI_PARAM_DEFAULTS = {
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
SRPI_PARAM_KEYS = tuple(SRPI_PARAM_DEFAULTS.keys())

CI_COMPONENTS = ("RAM", "PDI", "NAS", "IIM", "SRPI")
# Default CI reference: per-component means of the cohort's high-state
# (awake) session, computed from the same call's rows and recorded per row.
CI_REFERENCE_COHORT_HIGH_STATE = "cohort_high_state"
CI_STATUS_COLUMNS = ("CI_defined", "CI_missing", "CI_reference")
CI_NORM_COLUMNS = tuple(f"{k}_norm" for k in CI_COMPONENTS)

# MPC evidence layer (see impact_pipeline.evidence). Null families used when
# null_surrogates > 0: PDI, NAS and IIM use their estimators' own surrogate
# calibration (methods of mpc_metrics.SURROGATE_METHODS); RAM and SRPI use
# impact_pipeline.nulls.component_null (kinds of nulls.SURROGATE_KINDS).
# The RAM null shifts the whole event train rigidly (goal/stimulus/feedback
# timing is kept, only its alignment with the recording is destroyed). For a
# strictly periodic design a shift by a multiple of the period realigns the
# train with itself, so this null is only informative for jittered designs.
MPC_NULL_KINDS_DEFAULT = {
    "RAM": "onset_jitter",  # rigid circular shift of the whole event train
    "PDI": "phase_randomize",  # multivariate amplitude-adjusted phase surrogate
    "NAS": "circular_shift",  # independent circular shift per node
    "IIM": "circular_shift",  # independent circular shift per node
    "SRPI": "label_permutation",  # self/non-self labels re-dealt (counts kept)
}
# Minimum shift of the RAM/SRPI event-train null, as a fraction of the run.
MPC_EVENT_NULL_MIN_SHIFT_FRACTION = 0.1
# MPC degree: capped power mean (p=0: geometric) of two-anchor components.
MPC_DEGREE_P = 0.0
MPC_DEGREE_CAP = 1.0
MPC_EVIDENCE_FIELDS = ("status", "margin", "estimate", "null_mean", "null_sd", "null_n")
MPC_VERDICT_COLUMNS = (
    "MPC_verdict",
    "MPC_reason",
    "MPC_degree",
    "MPC_necessity_set",
    "MPC_null_surrogates",
    "MPC_null_seed",
    "MPC_null_families",
)
MPC_EVIDENCE_COLUMNS = MPC_VERDICT_COLUMNS + tuple(
    f"{k}_{f}" for k in CI_COMPONENTS for f in MPC_EVIDENCE_FIELDS
)


def load_ci_reference(path) -> dict:
    """
    Load external per-component CI reference means from a JSON file.

    Accepted layouts: ``{"references": {"RAM": .., "PDI": .., ...}}`` or a flat
    ``{"RAM": .., "PDI": .., ...}``. Every CI component must be present; values
    may be null/non-finite/<=0, in which case CI is undefined for every row.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    refs_raw = payload.get("references", payload) if isinstance(payload, dict) else None
    if not isinstance(refs_raw, dict):
        raise ValueError(
            f"CI reference file '{path}' must contain a JSON object of component means."
        )
    return _coerce_ci_reference_dict(refs_raw, source=str(path))


def _coerce_ci_reference_dict(refs_raw, source="reference"):
    missing = [k for k in CI_COMPONENTS if k not in refs_raw]
    if missing:
        raise ValueError(
            f"CI {source} is missing reference means for components: {missing}"
        )
    refs = {}
    for k in CI_COMPONENTS:
        val = refs_raw[k]
        try:
            refs[k] = float("nan") if val is None else float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"CI {source}: reference for {k} is not numeric ({val!r})."
            ) from exc
    return refs


def resolve_ci_references(df, reference=None, high_state_session="awake"):
    """
    Resolve per-component CI reference means and a provenance label.

    reference:
      - None or ``"cohort_high_state"``: mean of each component over the rows of
        ``high_state_session`` (subject means first, then the cohort mean, using
        only rows where that component is defined). If the session is absent or a
        component is never defined there, that reference is NaN (no floor), which
        makes CI undefined.
      - dict: external per-component means (label ``"external"``).
      - str / os.PathLike: path to a JSON file (see ``load_ci_reference``).
    Returns (references: dict, label: str).
    """
    is_cohort = (
        isinstance(reference, str) and reference == CI_REFERENCE_COHORT_HIGH_STATE
    )
    if reference is None or is_cohort:
        refs = {}
        if "session" in df.columns:
            high = df[df["session"].astype(str) == str(high_state_session)]
        else:
            high = df.iloc[0:0]
        for k in CI_COMPONENTS:
            if high.empty or k not in high.columns:
                refs[k] = float("nan")
                continue
            vals = pd.to_numeric(high[k], errors="coerce")
            if k == "IIM" and "IIM_defined" in high.columns:
                # Same definedness rule as assemble_ci: a flagged-undefined IIM
                # does not contribute to the reference even if a value is stored.
                iim_ok = [
                    _as_bool(f, default=np.isfinite(v))
                    for f, v in zip(high["IIM_defined"], vals)
                ]
                vals = vals.where(np.asarray(iim_ok, dtype=bool))
            if "subject" in high.columns:
                vals = vals.groupby(high["subject"].astype(str)).mean()
            m = float(vals.mean(skipna=True)) if vals.notna().any() else float("nan")
            refs[k] = m if np.isfinite(m) else float("nan")
        return refs, CI_REFERENCE_COHORT_HIGH_STATE
    if isinstance(reference, dict):
        return _coerce_ci_reference_dict(reference, source="reference dict"), "external"
    if isinstance(reference, (str, os.PathLike)):
        refs = load_ci_reference(reference)
        return refs, f"external_json:{os.path.abspath(os.fspath(reference))}"
    raise TypeError(
        "ci reference must be None, 'cohort_high_state', a dict of component means, "
        "or a JSON path."
    )


def _as_float(val):
    try:
        return float("nan") if val is None else float(val)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(val, default):
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, str):
        txt = val.strip().lower()
        if txt in {"true", "1", "yes"}:
            return True
        if txt in {"false", "0", "no"}:
            return False
        return bool(default)
    try:
        if val is None or (not np.isfinite(float(val))):
            return bool(default)
        return bool(val)
    except (TypeError, ValueError):
        return bool(default)


def assemble_ci(
    df,
    reference=None,
    weights=None,
    eps=1e-12,
    high_state_session="awake",
):
    """
    Add reference-normalised CI columns to a per-run metric table.

    CI uses NAS directly (no HypergraphSynergy multiplier). Adds ``CI`` (NaN when
    undefined), ``CI_defined``, ``CI_missing`` (comma-separated undefined weighted
    components, or ``<component>_reference`` for unusable references),
    ``CI_reference`` (reference label) and ``<component>_norm`` columns.
    Returns (df_with_ci, references).
    """
    out = df.copy()
    refs, label = resolve_ci_references(
        out, reference=reference, high_state_session=high_state_session
    )
    ci_vals, ci_defined, ci_missing = [], [], []
    norms = {k: [] for k in CI_COMPONENTS}
    for _, row in out.iterrows():
        vals = {k: _as_float(row.get(k, np.nan)) for k in CI_COMPONENTS}
        defined = {k: bool(np.isfinite(vals[k])) for k in CI_COMPONENTS}
        if "IIM_defined" in out.columns:
            iim_flag = _as_bool(row.get("IIM_defined"), default=defined["IIM"])
            defined["IIM"] = defined["IIM"] and iim_flag
        det = _compute_ci_legacy(
            vals["RAM"],
            vals["PDI"],
            vals["NAS"],
            vals["IIM"],
            vals["SRPI"],
            references=refs,
            weights=weights,
            defined=defined,
            eps=eps,
            return_details=True,
        )
        ci_vals.append(det["value"])
        ci_defined.append(bool(det["defined"]))
        ci_missing.append(",".join(det["missing"]))
        for k in CI_COMPONENTS:
            norms[k].append(det["normalized_components"][k])
    out["CI"] = ci_vals
    out["CI_defined"] = ci_defined
    out["CI_missing"] = ci_missing
    out["CI_reference"] = label
    for k in CI_COMPONENTS:
        out[f"{k}_norm"] = norms[k]
    out.attrs["ci_references"] = dict(refs)
    out.attrs["ci_reference_label"] = label
    return out, refs


def assemble_mpc_degree(
    df,
    reference=None,
    weights=None,
    high_state_session="awake",
    p=MPC_DEGREE_P,
    cap=MPC_DEGREE_CAP,
):
    """
    Add ``MPC_degree`` to a per-run table that carries the evidence columns.

    For ATTRIBUTED rows each component of the row's necessity set is put on the
    two-anchor scale with the row's null mean as the origin and the D3
    reference as the unit: ``c = (estimate - null_mean) / reference``
    (``evidence.two_anchor_normalize(estimate, null_mean, null_mean +
    reference)``). ``reference`` is resolved as for CI
    (:func:`resolve_ci_references`: cohort high-state means of the metric
    columns, which under null calibration hold the excess over the null, or an
    external dict/JSON on that scale). The degree is the capped power mean
    (``evidence.degree``) with ``weights`` restricted to the necessity set;
    NaN for every other verdict. Returns ``(df_with_degree, references)``.
    """
    out = df.copy()
    refs, _label = resolve_ci_references(
        out, reference=reference, high_state_session=high_state_session
    )
    degrees = []
    for _, row in out.iterrows():
        if str(row.get("MPC_verdict")) != mpc_evidence.Verdict.ATTRIBUTED.value:
            degrees.append(float("nan"))
            continue
        nset = mpc_evidence.normalize_necessity_set(str(row.get("MPC_necessity_set")))
        comps = {}
        for k in nset:
            null_mean = _as_float(row.get(f"{k}_null_mean"))
            comps[k] = mpc_evidence.two_anchor_normalize(
                _as_float(row.get(f"{k}_estimate")),
                null_mean,
                null_mean + _as_float(refs.get(k)),
            )
        w = None
        if weights is not None:
            w = {k: float(weights.get(k, 0.0)) for k in nset}
            if not any(v > 0 for v in w.values()):
                degrees.append(float("nan"))
                continue
        degrees.append(mpc_evidence.degree(comps, w, p=p, cap=cap))
    out["MPC_degree"] = degrees
    return out, refs


def _finish_mpc_evidence(df, reference, weights, high_state_session, registry):
    """MPC degree for ATTRIBUTED rows, a verdict summary log and provenance."""
    if "MPC_verdict" not in df.columns:
        return df
    attrs = dict(df.attrs)
    out, refs = assemble_mpc_degree(
        df, reference=reference, weights=weights, high_state_session=high_state_session
    )
    counts = out["MPC_verdict"].value_counts().to_dict()
    log.info(
        "MPC verdicts (rows): %s; necessity set=%s; null surrogates=%s",
        {k: int(v) for k, v in counts.items()},
        out["MPC_necessity_set"].iloc[0] if len(out) else "na",
        out["MPC_null_surrogates"].iloc[0] if len(out) else "na",
    )
    out.attrs.update(attrs)
    out.attrs["mpc_evidence"] = {
        "degree_p": MPC_DEGREE_P,
        "degree_cap": MPC_DEGREE_CAP,
        "degree_references": {k: _as_float(v) for k, v in refs.items()},
        "applicability_registry": (
            None if registry is None else registry.to_dict()
        ),
    }
    return out


def _resolve_null_kinds(null_kinds):
    kinds = dict(MPC_NULL_KINDS_DEFAULT)
    if null_kinds:
        unknown = sorted(set(null_kinds) - set(CI_COMPONENTS))
        if unknown:
            raise ValueError(f"null_kinds has unknown components: {unknown}")
        kinds.update({str(k): str(v) for k, v in dict(null_kinds).items()})
    for k in ("PDI", "NAS", "IIM"):
        if kinds[k] not in SURROGATE_METHODS:
            raise ValueError(
                f"null kind for {k} must be one of {SURROGATE_METHODS} (estimator "
                f"calibration), got {kinds[k]!r}"
            )
    for k in ("RAM", "SRPI"):
        kinds[k] = nulls.normalize_kind(kinds[k])
    return kinds


def _resolve_applicability_registry(registry):
    if registry is None or isinstance(registry, mpc_evidence.ApplicabilityRegistry):
        return registry
    if isinstance(registry, (str, os.PathLike)):
        return mpc_evidence.ApplicabilityRegistry.from_json(os.fspath(registry))
    if isinstance(registry, (dict, list, tuple)):
        return mpc_evidence.ApplicabilityRegistry.from_dict(registry)
    raise TypeError(
        "applicability_registry must be None, an ApplicabilityRegistry, a JSON "
        "path or a registry dict."
    )


def _run_null_key(ts_path):
    """Location-independent run label (<subj>/<ses>/<cond>/<file>) for seeds."""
    return "/".join(os.path.normpath(str(ts_path)).split(os.sep)[-4:])


def _event_null_kwargs(kind, n_tp, tr):
    if kind == "onset_jitter":
        t_max = float(n_tp) * float(tr)
        return {
            "common": True,
            "t_max": t_max,
            "min_shift": MPC_EVENT_NULL_MIN_SHIFT_FRACTION * t_max,
        }
    return {}


def _as_details(out):
    """Estimator output as a details dict (a bare float counts as its value)."""
    if isinstance(out, dict):
        return out
    val = _as_float(out)
    return {"value": val, "raw": val, "undefined_reason": None}


def _component_record(
    estimate,
    reason=None,
    estimator=None,
    null_mean=np.nan,
    null_sd=np.nan,
    null_n=0,
    null_family=None,
    null_reason=None,
):
    """Per-run evidence record of one component (raw estimate + its null)."""
    est = _as_float(estimate)
    defined = bool(np.isfinite(est))
    return {
        "estimate": est,
        "defined": defined,
        "reason": None if defined else str(reason or "undefined"),
        "estimator": estimator,
        "null_mean": _as_float(null_mean),
        "null_sd": _as_float(null_sd),
        "null_n": int(null_n or 0),
        "null_family": null_family,
        "null_reason": null_reason,
    }


def _record_from_null_fields(prefix, det, estimator, family):
    """Evidence record from a PDI/NAS details dict (``<prefix>_null_*`` fields)."""
    if not isinstance(det, dict):
        return _component_record(det, estimator=estimator)
    n = int(det.get(f"{prefix}_null_n", 0) or 0)
    return _component_record(
        det.get("raw", det.get("value")),
        reason=det.get("undefined_reason"),
        estimator=estimator,
        null_mean=det.get(f"{prefix}_null_mean") if n else np.nan,
        null_sd=det.get(f"{prefix}_null_sd") if n else np.nan,
        null_n=n,
        null_family=family if n else None,
        null_reason=det.get(f"{prefix}_null_undefined_reason"),
    )


def _calibrated_value(rec):
    """Excess over the null, floored at 0 (NaN when uncalibrated)."""
    if not (np.isfinite(rec["estimate"]) and np.isfinite(rec["null_mean"])):
        return float("nan")
    return float(max(rec["estimate"] - rec["null_mean"], 0.0))


def _component_null_record(
    principle, estimator_fn, ts_region_time, events, kind, n, seed, rec, tr
):
    """Attach a nulls.component_null distribution to a RAM/SRPI record."""
    kwargs = {}
    if kind in nulls.EVENT_SURROGATE_KINDS:
        kwargs = _event_null_kwargs(kind, ts_region_time.shape[1], tr)
    res = nulls.component_null(
        estimator_fn, ts_region_time, events, kind=kind, n=int(n), seed=int(seed),
        **kwargs,
    )
    rec = dict(rec)
    rec.update(
        null_mean=float(res.null_mean),
        null_sd=float(res.null_sd),
        null_n=int(res.samples.size),
        null_family=kind,
        null_reason=(None if res.samples.size else "all_surrogates_undefined"),
    )
    log.debug(
        "%s null (%s, n=%d, failed=%d): estimate=%.6g null_mean=%.6g null_sd=%.6g",
        principle, kind, int(n), int(res.n_failed), rec["estimate"],
        rec["null_mean"], rec["null_sd"],
    )
    return rec


_NULL_NOT_RUN = "not_run"


def _iim_record(iim_info, null_kind):
    """IIM evidence on the integration-mass scale Delta_Psi (bits)."""
    defined = bool(iim_info.get("defined", False))
    null_n = int(iim_info.get("IIM_null_n", 0) or 0)
    delta_psi = iim_info.get("Delta_Psi")
    if delta_psi is None:
        delta_psi = _as_float(iim_info.get("Psi_full")) - _as_float(
            iim_info.get("Psi_mip_preserved")
        )
    return _component_record(
        delta_psi if defined else np.nan,
        reason=iim_info.get("undefined_reason"),
        estimator=f"compute_IIM:{iim_info.get('iim_algorithm_version', 'unknown')}",
        null_mean=iim_info.get("Delta_Psi_null_mean") if null_n else np.nan,
        null_sd=iim_info.get("Delta_Psi_null_sd") if null_n else np.nan,
        null_n=null_n,
        null_family=iim_info.get("IIM_null_method", null_kind) if null_n else None,
        # Results without any null fields (e.g. Hunter reductions) were never
        # calibrated: NO_NULL_CALIBRATION rather than a failed null.
        null_reason=(
            iim_info.get("IIM_null_undefined_reason")
            if "IIM_null_n" in iim_info
            else _NULL_NOT_RUN
        ),
    )


def _mpc_protocol_id(**config):
    """Short digest of the protocol configuration shared by a run's evidence."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return "protocol-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _mpc_run_columns(
    records,
    nset,
    null_k,
    null_seed,
    null_kinds,
    registry,
    bearer_id,
    protocol_id,
    substrate,
    grain,
    regime,
):
    """
    ComponentEvidence per computed principle, the MPC verdict and the per-run
    evidence columns (``MPC_degree`` is filled later by assemble_mpc_degree).
    """
    evidence = {}
    for k, rec in records.items():
        defined, reason = rec["defined"], rec["reason"]
        null_ran = rec.get("null_reason") != _NULL_NOT_RUN
        if defined and null_k > 0 and rec["null_n"] == 0 and null_ran:
            # Calibration was requested but the null family failed.
            defined = False
            reason = f"null_undefined:{rec.get('null_reason') or 'no_valid_surrogates'}"
        evidence[k] = [
            mpc_evidence.ComponentEvidence(
                principle=k,
                estimate=rec["estimate"],
                null_mean=rec["null_mean"],
                null_sd=rec["null_sd"],
                se=0.0,
                defined=defined,
                reason=reason,
                bearer_id=bearer_id,
                protocol_id=protocol_id,
                substrate=substrate,
                grain=grain,
                estimator=rec["estimator"],
                null_family=rec["null_family"],
                n_null=int(rec["null_n"]),
            )
        ]
    verdict = mpc_evidence.mpc_verdict(
        evidence, necessity_set=nset, registry=registry, regime=regime
    )
    families = ""
    if null_k > 0:
        families = ";".join(
            f"{k}:{null_kinds[k]}" for k in CI_COMPONENTS if k in records
        )
    cols = {
        "MPC_verdict": verdict.verdict.value,
        "MPC_reason": verdict.reason_string,
        "MPC_degree": float("nan"),
        "MPC_necessity_set": ",".join(nset),
        "MPC_null_surrogates": int(null_k),
        "MPC_null_seed": int(null_seed),
        "MPC_null_families": families,
    }
    undefined = mpc_evidence.ComponentStatus.UNDEFINED
    for k in CI_COMPONENTS:
        rec = records.get(k)
        cols[f"{k}_status"] = verdict.component_status.get(k, undefined).value
        cols[f"{k}_margin"] = float(verdict.margins.get(k, float("nan")))
        cols[f"{k}_estimate"] = float("nan") if rec is None else rec["estimate"]
        cols[f"{k}_null_mean"] = float("nan") if rec is None else rec["null_mean"]
        cols[f"{k}_null_sd"] = float("nan") if rec is None else rec["null_sd"]
        cols[f"{k}_null_n"] = 0 if rec is None else int(rec["null_n"])
    return cols


def _parse_vm_stat_pages(vm_stat_text: str) -> dict:
    pages = {}
    page_size = 4096
    for line in vm_stat_text.splitlines():
        line = line.strip()
        if "page size of" in line and "bytes" in line:
            try:
                page_size = int(line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            except Exception:
                page_size = 4096
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip().rstrip(".").replace(".", "")
        try:
            pages[key.strip()] = int(val)
        except Exception:
            continue
    pages["_page_size"] = int(page_size)
    return pages


def _memory_snapshot_bytes():
    """
    Best-effort memory snapshot.
    Returns (used_bytes, total_bytes) or (None, None) when unavailable.
    """
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        p = _parse_vm_stat_pages(out)
        ps = float(p.get("_page_size", 4096))
        free = float(p.get("Pages free", 0))
        active = float(p.get("Pages active", 0))
        inactive = float(p.get("Pages inactive", 0))
        spec = float(p.get("Pages speculative", 0))
        wired = float(p.get("Pages wired down", 0))
        comp = float(p.get("Pages occupied by compressor", 0))
        # Same accounting already used in this project for rough used/free tracking.
        used_pages = active + inactive + spec + wired + comp
        total_pages = used_pages + free
        if total_pages <= 0:
            return None, None
        return int(used_pages * ps), int(total_pages * ps)
    except Exception:
        return None, None


def _iim_worker_from_path(
    ts_source,
    iim_bins,
    iim_lag_trs,
    iim_n_parts,
    iim_max_timepoints,
    iim_max_nodes,
    iim_max_mechanism_size,
    iim_max_purview_size,
    iim_checkpoint_dir,
    iim_resume_checkpoint,
    iim_checkpoint_every_cuts,
    iim_progress_log_every_cuts,
    iim_phase1_parallel_workers,
    iim_phase1_chunk_size,
    iim_phase1_shared_memory,
    hardware_target,
    iim_null_surrogates=0,
    iim_null_seed=0,
    iim_null_method=MPC_NULL_KINDS_DEFAULT["IIM"],
):
    # Keep each worker single-threaded for predictable scaling when many workers are used.
    backend = configure_process_for_hardware(hardware_target)
    if not backend.accelerator:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    shm = None
    if isinstance(ts_source, dict) and ts_source.get("mode") == "shared":
        ts_path = str(ts_source["ts_path"])
        shm = shared_memory.SharedMemory(name=str(ts_source["shm_name"]))
        ts_time_region = np.ndarray(
            tuple(ts_source["shape"]),
            dtype=np.dtype(ts_source["dtype"]),
            buffer=shm.buf,
        )
    else:
        ts_path = str(ts_source)
        ts_time_region = np.load(ts_path)
    ts_iim = ts_time_region.T
    if iim_max_timepoints is not None and int(iim_max_timepoints) > 0:
        max_tp = int(iim_max_timepoints)
        if ts_iim.shape[1] > max_tp:
            step = int(np.ceil(ts_iim.shape[1] / max_tp))
            ts_iim = ts_iim[:, ::step]

    checkpoint_path = None
    if iim_checkpoint_dir:
        os.makedirs(iim_checkpoint_dir, exist_ok=True)
        stem = os.path.basename(ts_path).replace("_ts.npy", "")
        stem = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in stem)[:80]
        sig = (
            f"{os.path.abspath(ts_path)}|bins={iim_bins}|lag={iim_lag_trs}|n_parts={iim_n_parts}|"
            f"max_tp={iim_max_timepoints}|max_nodes={iim_max_nodes}|"
            f"max_mech={iim_max_mechanism_size}|max_purv={iim_max_purview_size}|"
            f"ph1_workers={iim_phase1_parallel_workers}|ph1_chunk={iim_phase1_chunk_size}|"
            f"ph1_shm={int(bool(iim_phase1_shared_memory))}"
        )
        if int(iim_null_surrogates) > 0:
            # Uncalibrated checkpoints keep their historical names.
            sig += (
                f"|null={int(iim_null_surrogates)}|null_seed={int(iim_null_seed)}|"
                f"null_method={iim_null_method}"
            )
        digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
        checkpoint_path = os.path.join(
            iim_checkpoint_dir,
            f"{stem}_{digest}.iim_checkpoint.json",
        )

    null_kwargs = {}
    if int(iim_null_surrogates) > 0:
        null_kwargs = {
            "null_surrogates": int(iim_null_surrogates),
            "null_method": str(iim_null_method),
            "null_seed": nulls.derive_seed(
                int(iim_null_seed), "IIM", _run_null_key(ts_path)
            ),
        }
    try:
        iim_info = compute_IIM(
            ts_iim,
            bins=iim_bins,
            lag_trs=iim_lag_trs,
            n_parts=iim_n_parts,
            max_nodes=iim_max_nodes,
            max_mechanism_size=iim_max_mechanism_size,
            max_purview_size=iim_max_purview_size,
            partition_mode="all",
            clamp=True,
            return_details=True,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=bool(iim_resume_checkpoint),
            checkpoint_every_cuts=int(iim_checkpoint_every_cuts),
            progress_log_every_cuts=int(iim_progress_log_every_cuts),
            progress_label=os.path.basename(ts_path),
            phase1_parallel_workers=iim_phase1_parallel_workers,
            phase1_chunk_size=iim_phase1_chunk_size,
            phase1_shared_memory=bool(iim_phase1_shared_memory),
            hardware_backend=backend,
            **null_kwargs,
        )
    finally:
        if shm is not None:
            shm.close()
    return ts_path, iim_info


def _resolve_pdi_kwargs(pdi_params, require_explicit):
    if pdi_params is None:
        if require_explicit:
            raise ValueError(
                "Explicit PDI hyperparameters are required for strict validation runs, "
                "but pdi_params was None."
            )
        params = dict(PDI_PARAM_DEFAULTS)
    else:
        params = {}
        for k in PDI_PARAM_KEYS:
            if k in pdi_params:
                params[k] = pdi_params[k]
        missing = [k for k in PDI_PARAM_KEYS if k not in params]
        if missing and require_explicit:
            raise ValueError(
                "Explicit PDI hyperparameters are required for strict validation runs. "
                f"Missing keys: {missing}"
            )
        for k in missing:
            params[k] = PDI_PARAM_DEFAULTS[k]
    # Ensure normalization is controlled at endpoint level.
    params["normalize"] = False
    return params


def _resolve_ram_kwargs(ram_params):
    if ram_params is None:
        return dict(RAM_PARAM_DEFAULTS)
    params = {}
    for k in RAM_PARAM_KEYS:
        if k in ram_params:
            params[k] = ram_params[k]
    for k in RAM_PARAM_KEYS:
        if k not in params:
            params[k] = RAM_PARAM_DEFAULTS[k]
    return params


def _resolve_nas_kwargs(nas_params):
    if nas_params is None:
        raise ValueError(
            "NAS hyperparameters must be provided explicitly; fallback defaults are disabled."
        )
    params = {}
    for k in NAS_PARAM_KEYS:
        if k not in nas_params:
            raise ValueError(
                "NAS hyperparameters must be fully explicit; "
                f"missing key: '{k}'."
            )
        params[k] = nas_params[k]
    required_non_null = ("tau", "bands", "band_weights", "window_len", "step_len")
    none_keys = [k for k in required_non_null if params.get(k) is None]
    if none_keys:
        raise ValueError(
            "NAS fallback behavior is disabled. "
            f"The following keys cannot be None: {none_keys}"
        )
    # NAS baseline subtraction is handled explicitly when enabled by caller.
    params["baseline_ts"] = None
    return params


def _resolve_srpi_kwargs(srpi_params, require_explicit):
    if srpi_params is None:
        if require_explicit:
            raise ValueError(
                "SRPI hyperparameters must be provided explicitly; "
                "fallback defaults are disabled."
            )
        params = dict(SRPI_PARAM_DEFAULTS)
    else:
        params = {}
        for k in SRPI_PARAM_KEYS:
            if k in srpi_params:
                params[k] = srpi_params[k]
        missing = [k for k in SRPI_PARAM_KEYS if k not in params]
        if missing and require_explicit:
            raise ValueError(
                "SRPI hyperparameters must be fully explicit; "
                f"missing keys: {missing}."
            )
        for k in missing:
            params[k] = SRPI_PARAM_DEFAULTS[k]

    mode = str(params.get("modality", "")).strip().lower()
    if mode not in {"fmri", "eeg"}:
        raise ValueError("SRPI modality must be one of {'fmri','eeg'}.")
    params["modality"] = mode
    return params


def discover_ci_subjects(data_dir, subjects=None):
    subj_filter = None
    subj_order_requested = None
    if subjects is not None:
        subj_order_requested = []
        for s in subjects:
            ss = str(s).replace("sub-", "").strip()
            if ss and ss not in subj_order_requested:
                subj_order_requested.append(ss)
        subj_filter = set(subj_order_requested)

    subj_dirs = [
        subj
        for subj in sorted(os.listdir(data_dir))
        if (not subj.startswith(".")) and os.path.isdir(os.path.join(data_dir, subj))
    ]
    if subj_filter is not None:
        available = set(subj_dirs)
        missing = [s for s in subj_order_requested if s not in available]
        if missing:
            log.warning("Requested subjects not found under %s: %s", data_dir, ",".join(missing))
        return [s for s in subj_order_requested if s in available]
    return subj_dirs


def build_ci_run_specs(
    data_dir,
    atlas,
    sessions,
    condition,
    stimulus_onsets=None,
    subjects=None,
):
    run_specs = []
    subj_iter = discover_ci_subjects(data_dir, subjects=subjects)
    for subj in subj_iter:
        for ses in sessions:
            log.debug("compute_synergy_ci: subject=%s session=%s", subj, ses)
            session_dir = os.path.join(data_dir, subj, ses, condition)
            pattern = os.path.join(session_dir, f"{subj}_run-*_{atlas}_ts.npy")
            all_cands = sorted(glob.glob(pattern))

            subj_onsets, run_id = (None, None)
            if isinstance(stimulus_onsets, dict):
                tup = stimulus_onsets.get(subj, {}).get(ses)
                if tup:
                    subj_onsets, run_id = tup

            session_ts_paths = []
            if run_id is not None:
                run_clean = str(int(run_id))
                run_padded = str(run_id)
                exact_candidates = []
                for run_token in (run_clean, run_padded):
                    exact_candidates.extend(
                        sorted(
                            glob.glob(
                                os.path.join(
                                    session_dir,
                                    f"{subj}_run-{run_token}_{atlas}_ts.npy",
                                )
                            )
                        )
                    )
                if exact_candidates:
                    session_ts_paths = [exact_candidates[0]]
                else:
                    run_tags = (f"run-{run_clean}_", f"run-{run_padded}_")
                    filt = [
                        c for c in all_cands
                        if any(tag in os.path.basename(c) for tag in run_tags)
                    ]
                    if filt:
                        session_ts_paths = [filt[0]]
            if not session_ts_paths:
                if not all_cands:
                    raise FileNotFoundError(
                        f"No time-series for {subj}/{ses} (searched {pattern})"
                    )
                session_ts_paths = all_cands

            for ts_path in session_ts_paths:
                run_specs.append(
                    {
                        "subject": subj,
                        "session": ses,
                        "ts_path": ts_path,
                        "stimulus_onsets": subj_onsets,
                    }
                )
    return run_specs


def compute_synergy_ci(
    data_dir,
    atlas,
    thetas,
    sessions=('awake', 'deep', 'recovery'),
    condition='audio',
    tr=None,
    stimulus_onsets=None,
    ci_human_refs=None,
    ci_weights=None,
    ci_eps=1e-12,
    ci_reference=None,
    ci_reference_session="awake",
    pdi_anchor_session="deep",
    iim_display_scale=IIM_DISPLAY_SCALE_DEFAULT,
    iim_bins=3,
    iim_lag_trs=1,
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
    compute_mpc=True,
    mpc_metrics=None,
    compute_ci=True,
    subjects=None,
    ram_params=None,
    pdi_params=None,
    pdi_require_explicit_params=False,
    pdi_require_strict_baseline=False,
    pdi_primary_endpoint="anchor",
    nas_params=None,
    srpi_params=None,
    srpi_require_explicit_params=True,
    iim_precomputed_by_path=None,
    dataset_id=None,
    data_origin=REAL_DATA_ORIGIN,
    dataset_role=None,
    provenance_label=None,
    modality=None,
    hardware_target="cpu",
    null_surrogates=0,
    necessity_set=None,
    applicability_registry=None,
    null_seed=0,
    null_kinds=None,
):
    """
    Per-run MPC metrics, the exploratory legacy statistic S (one row per theta)
    and, when all five metrics are computed, the reference-normalised CI.

    CI notes: NAS enters CI directly (S is reported separately and never enters
    CI, so CI does not depend on theta). ``ci_reference`` selects the reference
    means: None/``"cohort_high_state"`` (default; means of ``ci_reference_session``),
    a dict of component means, or a JSON path. ``ci_human_refs`` is a deprecated
    alias for a dict reference. Undefined CI is NaN with ``CI_defined=False``.
    ``CI`` is the legacy geometric-mean CI (not a gate).

    PDI baselines: the anchor baseline is ``<subj>/<pdi_anchor_session>/rest`` and
    the state-matched baseline is ``<subj>/<session>/rest``; the evaluated run is
    never used as its own baseline. Missing baselines give NaN with a reason.

    MPC evidence layer (``impact_pipeline.evidence``): every run gets one
    ComponentEvidence per computed principle and a three-valued verdict over
    ``necessity_set`` (default all five; a sequence or ``"RAM,PDI,..."``),
    emitted as ``MPC_verdict``, ``MPC_reason`` (``;``-joined reason codes),
    ``MPC_degree`` (only for ATTRIBUTED rows, see :func:`assemble_mpc_degree`),
    ``MPC_necessity_set``, ``MPC_null_surrogates``, ``MPC_null_seed``,
    ``MPC_null_families`` and, per principle ``<P>``, ``<P>_status``,
    ``<P>_margin`` (null-SD units), ``<P>_estimate`` (raw estimator value),
    ``<P>_null_mean``, ``<P>_null_sd`` and ``<P>_null_n`` (null moments on the
    scale of ``<P>_estimate``; for IIM the integration mass Delta_Psi in bits).

    - ``null_surrogates=0`` (default): no null family is run, so every defined
      component is UNDEFINED (``NO_NULL_CALIBRATION:<P>``) and the verdict is
      UNDETERMINED. All metric and CI columns keep their previous values.
    - ``null_surrogates=K > 0``: PDI, NAS and IIM are calibrated by their
      estimators (``null_surrogates=K``; PDI on the primary endpoint), RAM and
      SRPI by ``nulls.component_null`` (default event-train circular shift and
      self/non-self label permutation; see ``MPC_NULL_KINDS_DEFAULT``,
      overridable per component with ``null_kinds``). Seeds are derived from
      ``null_seed`` and the run path. The metric columns ``RAM``..``SRPI`` then
      hold the calibrated values (excess over the null mean, floored at 0; NaN
      when the null could not be computed, never the raw value), so the legacy
      CI and its reference are on the calibrated scale. Precomputed IIM results
      (Hunter) without null fields give NaN IIM and ``NO_NULL_CALIBRATION:IIM``.
    - ``applicability_registry`` (ApplicabilityRegistry, dict or JSON path):
      evidence from estimators not validated for the substrate (``modality``),
      grain (``atlas``) and regime (``modality``, ``n_time``, ``n_nodes``,
      ``tr``) is UNDEFINED (``ESTIMATOR_NOT_VALIDATED``).
    """
    if ci_reference is None and ci_human_refs is not None:
        ci_reference = dict(ci_human_refs)
    null_k = int(null_surrogates)
    if null_k < 0:
        raise ValueError("null_surrogates must be >= 0")
    null_seed = int(null_seed)
    mpc_nset = mpc_evidence.normalize_necessity_set(necessity_set)
    mpc_registry = _resolve_applicability_registry(applicability_registry)
    mpc_null_kinds = _resolve_null_kinds(null_kinds)
    hardware_backend = configure_process_for_hardware(hardware_target)
    log.info("MPC hardware backend: %s", backend_summary(hardware_backend))

    valid_mpc_metrics = ("RAM", "PDI", "NAS", "IIM", "SRPI")
    if not compute_mpc:
        selected_metrics = tuple()
    elif mpc_metrics is None:
        selected_metrics = valid_mpc_metrics
    else:
        selected_metrics = tuple(dict.fromkeys(mpc_metrics))
        unknown = sorted(set(selected_metrics) - set(valid_mpc_metrics))
        if unknown:
            raise ValueError(
                f"Unknown MPC metric(s): {unknown}. "
                f"Expected subset of {valid_mpc_metrics}."
            )
    do_ram = "RAM" in selected_metrics
    do_pdi = "PDI" in selected_metrics
    do_nas = "NAS" in selected_metrics
    do_iim = "IIM" in selected_metrics
    do_srpi = "SRPI" in selected_metrics
    data_origin_norm = normalize_data_origin(data_origin)
    dataset_role_txt = (
        dataset_role_for_origin(data_origin_norm)
        if dataset_role is None
        else str(dataset_role)
    )
    provenance_label_txt = (
        provenance_label_for_origin(data_origin_norm)
        if provenance_label is None
        else str(provenance_label)
    )
    result_metadata = {
        "dataset_id": "" if dataset_id is None else str(dataset_id),
        "data_origin": str(data_origin_norm),
        "dataset_role": str(dataset_role_txt),
        "provenance_label": str(provenance_label_txt),
        "hardware_target": str(hardware_backend.requested),
        "hardware_backend": str(hardware_backend.target),
        "hardware_runtime": str(hardware_backend.runtime),
    }

    pdi_endpoint = str(pdi_primary_endpoint).strip().lower()
    if pdi_endpoint not in {"anchor", "task"}:
        raise ValueError("pdi_primary_endpoint must be one of {'anchor','task'}")
    pdi_kwargs = _resolve_pdi_kwargs(
        pdi_params=pdi_params,
        require_explicit=bool(pdi_require_explicit_params and do_pdi),
    )
    ram_kwargs = _resolve_ram_kwargs(ram_params) if do_ram else {}
    pdi_kwargs_raw = dict(pdi_kwargs)
    pdi_kwargs_raw["clip_negative"] = False
    nas_kwargs = (
        _resolve_nas_kwargs(
            nas_params=nas_params,
        )
        if do_nas
        else {}
    )
    srpi_kwargs = (
        _resolve_srpi_kwargs(
            srpi_params=srpi_params,
            require_explicit=bool(srpi_require_explicit_params and do_srpi),
        )
        if do_srpi
        else {}
    )
    if do_ram or do_nas or do_srpi:
        if tr is None or (not np.isfinite(tr)) or float(tr) <= 0:
            raise ValueError(
                "Explicit positive sample interval 'tr' is required for RAM/NAS/SRPI; "
                "fallback TR resolution is disabled."
            )
        tr = float(tr)

    pdi_anchor_session = str(pdi_anchor_session)
    # All components of a run are measured on one declared bearer (the run's
    # node x time matrix) under one protocol (this call's configuration).
    mpc_protocol_id = _mpc_protocol_id(
        atlas=atlas,
        condition=condition,
        tr=tr,
        modality=modality,
        null_surrogates=null_k,
        null_seed=null_seed,
        null_kinds=mpc_null_kinds if null_k > 0 else None,
    )

    def _select_pdi_rest_runs(subj, session_name):
        # Rest baselines live under <subj>/<session>/rest (fMRI preprocessing and,
        # when rest recordings exist, EEG preprocessing). Accept any run naming.
        rest_dir = os.path.join(data_dir, subj, session_name, 'rest')
        pattern = os.path.join(rest_dir, f"{subj}_*_{atlas}_ts.npy")
        return sorted(set(glob.glob(pattern)))

    def _select_pdi_state_rest_runs(subj, session_name):
        return _select_pdi_rest_runs(subj, session_name)

    def _select_pdi_anchor_rest_runs(subj):
        return _select_pdi_rest_runs(subj, pdi_anchor_session)

    def _select_pdi_legacy_baseline_runs(subj, session_name):
        """
        Baseline policy for permissive runs.
        Priority:
          1) same-session rest
          2) any subject-level rest
          3) surrogate baseline inside compute_PDI (None)
        """
        same_session_rest = _select_pdi_state_rest_runs(subj, session_name)
        if same_session_rest:
            return same_session_rest
        subj_root = os.path.join(data_dir, subj)
        pattern = os.path.join(subj_root, '*', 'rest', f"{subj}_*_{atlas}_ts.npy")
        any_rest = sorted(set(glob.glob(pattern)))
        if any_rest:
            return any_rest
        return None

    def _load_pdi_baseline_ts(paths, task_ts_path, task_region_time, label):
        """
        Load baseline runs for one evaluated run. The evaluated run itself (same
        file or an identical copy) is never used as its own baseline, and runs with
        a different region count are rejected. Returns (ts_list, paths, reason).
        """
        ts_list = []
        keep_paths = []
        n_same = n_mismatch = n_unreadable = 0
        task_real = os.path.realpath(task_ts_path)
        for fn in paths:
            if os.path.realpath(fn) == task_real:
                n_same += 1
                continue
            try:
                arr = np.load(fn).T
            except Exception:
                n_unreadable += 1
                continue
            if arr.ndim != 2:
                n_unreadable += 1
                continue
            if int(arr.shape[0]) != int(task_region_time.shape[0]):
                n_mismatch += 1
                continue
            same_shape = arr.shape == task_region_time.shape
            if same_shape and np.array_equal(arr, task_region_time):
                n_same += 1
                continue
            ts_list.append(arr)
            keep_paths.append(str(fn))
        if ts_list:
            reason = "ok"
        elif n_mismatch:
            reason = f"{label}_baseline_region_mismatch"
        elif n_unreadable:
            reason = f"{label}_baseline_unreadable"
        elif n_same:
            reason = f"{label}_baseline_identical_to_task_run"
        else:
            reason = None
        return ts_list, keep_paths, reason

    run_specs = build_ci_run_specs(
        data_dir,
        atlas,
        sessions,
        condition,
        stimulus_onsets=stimulus_onsets,
        subjects=subjects,
    )

    records = []
    if not run_specs:
        cols_empty = list(PROVENANCE_COLUMNS) + [
            'hardware_target', 'hardware_backend', 'hardware_runtime',
            'subject', 'session', 'theta', 'S',
        ]
        if compute_mpc:
            cols_empty.extend(list(selected_metrics))
            if do_pdi:
                cols_empty.extend([
                    'PDI_anchor', 'PDI_task',
                    'PDI_anchor_defined', 'PDI_task_defined',
                    'PDI_anchor_reason', 'PDI_task_reason',
                    'PDI_primary_endpoint', 'PDI_primary_source',
                    'PDI_baseline_policy',
                    'PDI_anchor_baseline_n_runs', 'PDI_task_baseline_n_runs',
                    'PDI_anchor_baseline_paths', 'PDI_task_baseline_paths',
                ])
            if do_iim:
                cols_empty.extend([
                    'IIM_raw', 'IIM_raw_scaled',
                    'IIM_defined', 'IIM_undefined_reason',
                ])
            if compute_ci and set(valid_mpc_metrics).issubset(set(selected_metrics)):
                cols_empty.extend([
                    'CI', *CI_STATUS_COLUMNS,
                    'RAM_norm', 'PDI_norm', 'NAS_norm', 'IIM_norm', 'SRPI_norm',
                ])
            cols_empty.extend(MPC_EVIDENCE_COLUMNS)
        return pd.DataFrame(columns=cols_empty)

    iim_by_path = {}
    if compute_mpc and do_iim and iim_precomputed_by_path is not None:
        iim_by_path = {
            str(k): v
            for k, v in dict(iim_precomputed_by_path).items()
        }
    elif compute_mpc and do_iim:
        by_subject_paths = {}
        for spec in run_specs:
            subj = str(spec["subject"])
            ts_path = str(spec["ts_path"])
            paths = by_subject_paths.setdefault(subj, [])
            if ts_path not in paths:
                paths.append(ts_path)
        subject_order = list(by_subject_paths.keys())
        unique_paths = [p for subj in subject_order for p in by_subject_paths[subj]]
        ts_source_by_path = {p: p for p in unique_paths}
        shared_handles = []

        if iim_checkpoint_dir:
            os.makedirs(iim_checkpoint_dir, exist_ok=True)
            log.info(
                "IIM checkpointing: dir=%s resume=%s checkpoint_every_cuts=%d progress_every_cuts=%d",
                iim_checkpoint_dir,
                bool(iim_resume_checkpoint),
                int(iim_checkpoint_every_cuts),
                int(iim_progress_log_every_cuts),
            )
        log.info(
            "IIM intra-task config: phase_workers=%s chunk_size=%d phase_shared_memory=%s",
            "auto/off" if iim_phase1_parallel_workers is None else str(int(iim_phase1_parallel_workers)),
            int(iim_phase1_chunk_size),
            bool(iim_phase1_shared_memory),
        )

        if bool(iim_use_shared_memory) and len(unique_paths) > 0:
            shared_total_b = 0
            try:
                for ts_path in unique_paths:
                    arr = np.load(ts_path, mmap_mode="r")
                    arr_c = np.ascontiguousarray(arr)
                    shm = shared_memory.SharedMemory(create=True, size=int(arr_c.nbytes))
                    shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
                    shm_arr[...] = arr_c
                    shared_handles.append(shm)
                    ts_source_by_path[ts_path] = {
                        "mode": "shared",
                        "shm_name": str(shm.name),
                        "shape": tuple(int(x) for x in arr_c.shape),
                        "dtype": str(arr_c.dtype),
                        "ts_path": ts_path,
                    }
                    shared_total_b += int(arr_c.nbytes)
                    del shm_arr
                    del arr_c
                    del arr
                gc.collect()
                log.info(
                    "IIM shared-memory input staging: arrays=%d total=%.3fGB",
                    len(unique_paths),
                    shared_total_b / float(1024 ** 3),
                )
            except Exception as exc:
                log.warning(
                    "IIM shared-memory staging failed (%s). Falling back to direct file loads.",
                    exc,
                )
                for shm in shared_handles:
                    try:
                        shm.close()
                    except Exception:
                        pass
                    try:
                        shm.unlink()
                    except Exception:
                        pass
                shared_handles = []
                ts_source_by_path = {p: p for p in unique_paths}

        cpu_max = max(1, int(os.cpu_count() or 1))
        if not iim_enable_parallel:
            workers = 1
        elif iim_parallel_workers is not None:
            workers = max(1, int(iim_parallel_workers))
        else:
            oversub = max(1.0, float(iim_cpu_oversub_factor))
            soft_cpu_cap = max(1, int(np.ceil(cpu_max * oversub)))
            workers = min(soft_cpu_cap, len(unique_paths))
            used_b, total_b = _memory_snapshot_bytes()
            target_ratio = float(iim_memory_target_ratio)
            target_ratio = min(max(target_ratio, 0.10), 0.99)
            if used_b is not None and total_b is not None and total_b > 0:
                budget_b = max(0, int(total_b * target_ratio) - int(used_b))
                est_b = max(int(float(iim_worker_mem_gb_estimate) * (1024 ** 3)), 256 * 1024 * 1024)
                by_mem = max(1, int(budget_b // est_b))
                workers = min(workers, by_mem)
                used_ratio = float(used_b / total_b)
                log.info(
                    "IIM memory planner: used=%.1f%% target=%.1f%% est_worker_mem=%.2fGB "
                    "budget=%.2fGB oversub=%.2f cpu_cap=%d -> workers=%d",
                    used_ratio * 100.0,
                    target_ratio * 100.0,
                    est_b / float(1024 ** 3),
                    budget_b / float(1024 ** 3),
                    oversub,
                    soft_cpu_cap,
                    workers,
                )
        workers = max(1, min(workers, len(unique_paths)))
        log.info(
            "IIM execution: tasks=%d workers=%d (cpu_max=%d, target_mem_ratio=%.2f)",
            len(unique_paths),
            workers,
            cpu_max,
            float(iim_memory_target_ratio),
        )
        if hardware_backend.accelerator and iim_parallel_workers is None:
            workers_limited = max(1, min(int(workers), max(1, int(hardware_backend.device_count))))
            if workers_limited != workers:
                log.info(
                    "IIM accelerator scheduling: limiting auto workers %d -> %d for %d visible device(s)",
                    int(workers),
                    int(workers_limited),
                    int(hardware_backend.device_count),
                )
                workers = workers_limited
        if subject_order:
            init_slots = int(workers)
            init_alloc = []
            for subj in subject_order:
                if init_slots <= 0:
                    break
                n_take = min(len(by_subject_paths.get(subj, [])), init_slots)
                if n_take > 0:
                    init_alloc.append(f"{subj}:{n_take}")
                    init_slots -= n_take
            log.info(
                "IIM subject-priority scheduling: order=%s initial_allocation=%s",
                ",".join(subject_order),
                ",".join(init_alloc) if init_alloc else "none",
            )

        try:
            if workers == 1:
                for i, ts_path in enumerate(unique_paths, start=1):
                    _, iim_info = _iim_worker_from_path(
                        ts_source_by_path[ts_path],
                        iim_bins,
                        iim_lag_trs,
                        iim_n_parts,
                        iim_max_timepoints,
                        iim_max_nodes,
                        iim_max_mechanism_size,
                        iim_max_purview_size,
                        iim_checkpoint_dir,
                        iim_resume_checkpoint,
                        iim_checkpoint_every_cuts,
                        iim_progress_log_every_cuts,
                        iim_phase1_parallel_workers,
                        iim_phase1_chunk_size,
                        iim_phase1_shared_memory,
                        hardware_backend.requested,
                        null_k,
                        null_seed,
                        mpc_null_kinds["IIM"],
                    )
                    iim_by_path[ts_path] = iim_info
                    if (i == len(unique_paths)) or (i % max(1, len(unique_paths) // 20) == 0):
                        log.info("IIM progress: %d/%d completed", i, len(unique_paths))
            else:
                pending_by_subject = {
                    subj: deque(by_subject_paths.get(subj, []))
                    for subj in subject_order
                }
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = {}
                    done = 0
                    total = len(unique_paths)
                    while done < total:
                        free_slots = int(workers) - len(futures)
                        if free_slots > 0:
                            for subj in subject_order:
                                q = pending_by_subject[subj]
                                while q and free_slots > 0:
                                    ts_path = q.popleft()
                                    fut = ex.submit(
                                        _iim_worker_from_path,
                                        ts_source_by_path[ts_path],
                                        iim_bins,
                                        iim_lag_trs,
                                        iim_n_parts,
                                        iim_max_timepoints,
                                        iim_max_nodes,
                                        iim_max_mechanism_size,
                                        iim_max_purview_size,
                                        iim_checkpoint_dir,
                                        iim_resume_checkpoint,
                                        iim_checkpoint_every_cuts,
                                        iim_progress_log_every_cuts,
                                        iim_phase1_parallel_workers,
                                        iim_phase1_chunk_size,
                                        iim_phase1_shared_memory,
                                        hardware_backend.requested,
                                        null_k,
                                        null_seed,
                                        mpc_null_kinds["IIM"],
                                    )
                                    futures[fut] = ts_path
                                    free_slots -= 1
                                if free_slots <= 0:
                                    break

                        if not futures:
                            raise RuntimeError(
                                "IIM scheduler stalled with no active futures before completing all tasks."
                            )

                        done_now, _ = concurrent.futures.wait(
                            list(futures.keys()),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for fut in done_now:
                            ts_path = futures.pop(fut)
                            try:
                                _, iim_info = fut.result()
                            except Exception as exc:
                                raise RuntimeError(f"IIM worker failed for {ts_path}: {exc}") from exc
                            iim_by_path[ts_path] = iim_info
                            done += 1
                            if (done == total) or (done % max(1, total // 20) == 0):
                                log.info("IIM progress: %d/%d completed", done, total)
        finally:
            for shm in shared_handles:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass
            if shared_handles:
                log.info("IIM shared-memory staging: cleaned up %d segments", len(shared_handles))

    for spec in run_specs:
        subj = spec["subject"]
        ses = spec["session"]
        ts_path = spec["ts_path"]
        subj_onsets = spec["stimulus_onsets"]

        log.debug("compute_synergy_ci: using TS %s", ts_path)
        ts_time_region = np.load(ts_path)
        ts_region_time = ts_time_region.T

        run_key = _run_null_key(ts_path)
        mpc_records = {}
        if compute_mpc:
            if do_ram:
                ram_det = _as_details(compute_RAM(
                    ts_region_time,
                    tr=tr,
                    stimulus_onsets=subj_onsets,
                    **ram_kwargs,
                    hardware_backend=hardware_backend,
                    return_details=True,
                ))
                ram_rec = _component_record(
                    ram_det.get("value"),
                    reason=ram_det.get("undefined_reason"),
                    estimator="compute_RAM",
                )
                if null_k > 0 and ram_rec["defined"]:
                    def _ram_fn(ts_s, ev_s):
                        return compute_RAM(
                            ts_s,
                            tr=tr,
                            stimulus_onsets=ev_s,
                            **ram_kwargs,
                            hardware_backend=hardware_backend,
                        )

                    ram_rec = _component_null_record(
                        "RAM", _ram_fn, ts_region_time, subj_onsets,
                        mpc_null_kinds["RAM"], null_k,
                        nulls.derive_seed(null_seed, "RAM", run_key), ram_rec, tr,
                    )
                mpc_records["RAM"] = ram_rec
                ram = _calibrated_value(ram_rec) if null_k > 0 else ram_rec["estimate"]
            else:
                ram = np.nan
            if do_pdi:
                deep_rest_cands = _select_pdi_anchor_rest_runs(subj)
                state_rest_cands = _select_pdi_state_rest_runs(subj, ses)
                (
                    deep_rest_ts, deep_rest_paths, anchor_load_reason
                ) = _load_pdi_baseline_ts(
                    deep_rest_cands,
                    ts_path,
                    ts_region_time,
                    label="anchor",
                )
                (
                    state_rest_ts, state_rest_paths, state_load_reason
                ) = _load_pdi_baseline_ts(
                    state_rest_cands,
                    ts_path,
                    ts_region_time,
                    label="state",
                )

                pdi_anchor_raw = np.nan
                pdi_task_raw = np.nan
                pdi_anchor_reason = (
                    anchor_load_reason or f"missing_{pdi_anchor_session}_rest_baseline"
                )
                pdi_task_reason = state_load_reason or "missing_state_rest_baseline"
                pdi_primary_source = "undefined"
                pdi_baseline_policy = (
                    "strict_dual_baseline"
                    if bool(pdi_require_strict_baseline)
                    else "dual_baseline_with_legacy_fallback"
                )
                # Null calibration (K > 0) runs on the endpoint that enters PDI.
                pdi_null_kw = {}
                if null_k > 0:
                    pdi_null_kw = {
                        "null_surrogates": null_k,
                        "null_method": mpc_null_kinds["PDI"],
                        "null_seed": nulls.derive_seed(null_seed, "PDI", run_key),
                    }
                pdi_primary_det = None

                if deep_rest_ts:
                    pdi_anchor_det = _as_details(compute_PDI(
                        ts_region_time,
                        baseline_ts=deep_rest_ts,
                        hardware_backend=hardware_backend,
                        return_details=True,
                        **pdi_kwargs_raw,
                        **(pdi_null_kw if pdi_endpoint == "anchor" else {}),
                    ))
                    pdi_anchor_raw = float(pdi_anchor_det["raw"])
                    pdi_anchor_reason = (
                        "ok" if np.isfinite(pdi_anchor_raw)
                        else "anchor_pdi_estimator_undefined"
                    )
                    if pdi_endpoint == "anchor":
                        pdi_primary_det = pdi_anchor_det
                if state_rest_ts:
                    pdi_task_det = _as_details(compute_PDI(
                        ts_region_time,
                        baseline_ts=state_rest_ts,
                        hardware_backend=hardware_backend,
                        return_details=True,
                        **pdi_kwargs_raw,
                        **(pdi_null_kw if pdi_endpoint == "task" else {}),
                    ))
                    pdi_task_raw = float(pdi_task_det["raw"])
                    pdi_task_reason = (
                        "ok" if np.isfinite(pdi_task_raw)
                        else "state_pdi_estimator_undefined"
                    )
                    if pdi_endpoint == "task":
                        pdi_primary_det = pdi_task_det

                if pdi_endpoint == "anchor":
                    pdi_primary_raw = pdi_anchor_raw
                    pdi_primary_reason = pdi_anchor_reason
                else:
                    pdi_primary_raw = pdi_task_raw
                    pdi_primary_reason = pdi_task_reason

                if np.isfinite(pdi_primary_raw):
                    pdi_primary_source = pdi_endpoint
                    if null_k > 0:
                        excess = _as_float(pdi_primary_det.get("PDI_excess"))
                        pdi0 = max(excess, 0.0) if np.isfinite(excess) else np.nan
                    else:
                        pdi0 = max(float(pdi_primary_raw), 0.0)
                else:
                    pdi0 = np.nan
                    pdi_primary_det = None

                # The legacy fallback replaces an undefined primary endpoint
                # only (a failed null never triggers it).
                if (not np.isfinite(pdi_primary_raw)) and (
                    not bool(pdi_require_strict_baseline)
                ):
                    legacy_cands = _select_pdi_legacy_baseline_runs(subj, ses)
                    if legacy_cands is None:
                        pdi_primary_det = _as_details(compute_PDI(
                            ts_region_time,
                            baseline_ts=None,
                            hardware_backend=hardware_backend,
                            return_details=True,
                            **pdi_kwargs,
                            **pdi_null_kw,
                        ))
                        pdi0 = float(pdi_primary_det["value"])
                        pdi_primary_source = "legacy_surrogate"
                    else:
                        legacy_ts, _paths, _reason = _load_pdi_baseline_ts(
                            legacy_cands,
                            ts_path,
                            ts_region_time,
                            label="legacy",
                        )
                        if legacy_ts:
                            pdi_primary_det = _as_details(compute_PDI(
                                ts_region_time,
                                baseline_ts=legacy_ts,
                                hardware_backend=hardware_backend,
                                return_details=True,
                                **pdi_kwargs,
                                **pdi_null_kw,
                            ))
                            pdi0 = float(pdi_primary_det["value"])
                            # Legacy pool provenance is carried by the source
                            # label; the state-matched baseline columns keep
                            # describing PDI_task only.
                            pdi_primary_source = "legacy_rest_pool"
                        else:
                            pdi0 = np.nan
                if pdi_primary_det is not None:
                    mpc_records["PDI"] = _record_from_null_fields(
                        "PDI", pdi_primary_det, f"compute_PDI:{pdi_primary_source}",
                        mpc_null_kinds["PDI"],
                    )
                else:
                    mpc_records["PDI"] = _component_record(
                        np.nan, reason=pdi_primary_reason, estimator="compute_PDI"
                    )
            else:
                pdi0 = np.nan
                pdi_anchor_raw = np.nan
                pdi_task_raw = np.nan
                pdi_anchor_reason = "not_computed"
                pdi_task_reason = "not_computed"
                pdi_primary_source = "not_computed"
                pdi_baseline_policy = "not_computed"
                deep_rest_paths = []
                state_rest_paths = []
            if do_nas:
                nas_null_kw = {}
                if null_k > 0:
                    nas_null_kw = {
                        "null_surrogates": null_k,
                        "null_method": mpc_null_kinds["NAS"],
                        "null_seed": nulls.derive_seed(null_seed, "NAS", run_key),
                    }
                nas_det = _as_details(compute_NAS(
                    ts_region_time,
                    tr=tr,
                    hardware_backend=hardware_backend,
                    return_details=True,
                    **nas_kwargs,
                    **nas_null_kw,
                ))
                nas = float(nas_det["value"])
                mpc_records["NAS"] = _record_from_null_fields(
                    "NAS", nas_det, "compute_NAS", mpc_null_kinds["NAS"]
                )
            else:
                nas = np.nan
            if do_iim:
                iim_info = iim_by_path.get(ts_path)
                if iim_info is None:
                    raise RuntimeError(f"Missing precomputed IIM result for {ts_path}")
                iim_defined = bool(iim_info.get("defined", False))
                iim_undefined_reason = iim_info.get("undefined_reason")
                iim = float(iim_info["canonical"]) if iim_defined else np.nan
                iim_raw = float(iim_info["raw"]) if iim_defined else np.nan
                iim_raw_scaled = (
                    iim_raw * float(iim_display_scale) if np.isfinite(iim_raw) else np.nan
                )
                mpc_records["IIM"] = _iim_record(iim_info, mpc_null_kinds["IIM"])
                if null_k > 0 and iim_defined:
                    # Calibrated canonical IIM (bits); never the raw ratio.
                    iim = _as_float(iim_info.get("canonical_calibrated"))
                    if not (mpc_records["IIM"]["null_n"] and np.isfinite(iim)):
                        iim = np.nan
                        iim_defined = False
                        iim_undefined_reason = "null_calibration_unavailable:" + str(
                            iim_info.get("IIM_null_undefined_reason")
                            or "no_iim_null_fields"
                        )
            else:
                iim_defined = False
                iim_undefined_reason = "not_computed"
                iim = np.nan
                iim_raw = np.nan
                iim_raw_scaled = np.nan
            if do_srpi:
                srpi_events = subj_onsets if isinstance(subj_onsets, dict) else {}
                srpi_det = _as_details(compute_SRPI(
                    ts_region_time,
                    tr=tr,
                    self_onsets=srpi_events.get("self_onsets", []),
                    nonself_onsets=srpi_events.get("nonself_onsets", []),
                    hardware_backend=hardware_backend,
                    return_details=True,
                    **srpi_kwargs,
                ))
                srpi_rec = _component_record(
                    srpi_det.get("value"),
                    reason=srpi_det.get("undefined_reason"),
                    estimator="compute_SRPI",
                )
                if null_k > 0 and srpi_rec["defined"]:
                    def _srpi_fn(ts_s, ev_s):
                        return compute_SRPI(
                            ts_s,
                            tr=tr,
                            self_onsets=ev_s.get("self_onsets", []),
                            nonself_onsets=ev_s.get("nonself_onsets", []),
                            hardware_backend=hardware_backend,
                            **srpi_kwargs,
                        )

                    srpi_rec = _component_null_record(
                        "SRPI", _srpi_fn, ts_region_time, dict(srpi_events),
                        mpc_null_kinds["SRPI"], null_k,
                        nulls.derive_seed(null_seed, "SRPI", run_key), srpi_rec, tr,
                    )
                mpc_records["SRPI"] = srpi_rec
                srpi = (
                    _calibrated_value(srpi_rec) if null_k > 0 else srpi_rec["estimate"]
                )
            else:
                srpi = np.nan
            mpc_cols = _mpc_run_columns(
                mpc_records,
                nset=mpc_nset,
                null_k=null_k,
                null_seed=null_seed,
                null_kinds=mpc_null_kinds,
                registry=mpc_registry,
                bearer_id=f"{subj}/{ses}/{os.path.basename(ts_path)}",
                protocol_id=mpc_protocol_id,
                substrate=modality,
                grain=atlas,
                regime={
                    "modality": modality,
                    "n_time": int(ts_region_time.shape[1]),
                    "n_nodes": int(ts_region_time.shape[0]),
                    "tr": tr,
                },
            )
        else:
            ram = np.nan
            pdi0 = np.nan
            pdi_anchor_raw = np.nan
            pdi_task_raw = np.nan
            pdi_anchor_reason = "not_computed"
            pdi_task_reason = "not_computed"
            pdi_primary_source = "not_computed"
            pdi_baseline_policy = "not_computed"
            deep_rest_paths = []
            state_rest_paths = []
            nas = np.nan
            iim = np.nan
            iim_raw = np.nan
            iim_raw_scaled = np.nan
            iim_defined = False
            iim_undefined_reason = "not_computed"
            srpi = np.nan
            mpc_cols = {}

        for theta in thetas:
            S = HypergraphSynergy.compute(ts_time_region, theta)
            rec = dict(result_metadata)
            rec.update({
                'subject': subj,
                'session': ses,
                'theta': theta,
                'S': S,
            })
            if compute_mpc:
                if do_ram:
                    rec['RAM'] = ram
                if do_pdi:
                    rec.update({
                        'PDI': pdi0,
                        'PDI_anchor': pdi_anchor_raw,
                        'PDI_task': pdi_task_raw,
                        'PDI_anchor_defined': bool(np.isfinite(pdi_anchor_raw)),
                        'PDI_task_defined': bool(np.isfinite(pdi_task_raw)),
                        'PDI_anchor_reason': str(pdi_anchor_reason),
                        'PDI_task_reason': str(pdi_task_reason),
                        'PDI_primary_endpoint': str(pdi_endpoint),
                        'PDI_primary_source': str(pdi_primary_source),
                        'PDI_baseline_policy': str(pdi_baseline_policy),
                        'PDI_anchor_baseline_n_runs': int(len(deep_rest_paths)),
                        'PDI_task_baseline_n_runs': int(len(state_rest_paths)),
                        'PDI_anchor_baseline_paths': ";".join(deep_rest_paths),
                        'PDI_task_baseline_paths': ";".join(state_rest_paths),
                    })
                if do_nas:
                    rec['NAS'] = nas
                if do_iim:
                    rec.update({
                        'IIM': iim,
                        'IIM_raw': iim_raw,
                        'IIM_raw_scaled': iim_raw_scaled,
                        'IIM_defined': iim_defined,
                        'IIM_undefined_reason': iim_undefined_reason,
                    })
                if do_srpi:
                    rec['SRPI'] = srpi
                rec.update(mpc_cols)
            records.append(rec)

    hardware_cols = ['hardware_target', 'hardware_backend', 'hardware_runtime']
    cols_empty = list(PROVENANCE_COLUMNS) + hardware_cols + ['subject', 'session', 'theta', 'S']
    if compute_mpc:
        cols_empty = list(PROVENANCE_COLUMNS) + hardware_cols + ['subject', 'session', 'theta', 'S']
        cols_empty.extend(list(selected_metrics))
        if do_pdi:
            cols_empty.extend([
                'PDI_anchor', 'PDI_task',
                'PDI_anchor_defined', 'PDI_task_defined',
                'PDI_anchor_reason', 'PDI_task_reason',
                'PDI_primary_endpoint', 'PDI_primary_source',
                'PDI_baseline_policy',
                'PDI_anchor_baseline_n_runs', 'PDI_task_baseline_n_runs',
                'PDI_anchor_baseline_paths', 'PDI_task_baseline_paths',
            ])
        if do_iim:
            cols_empty.extend([
                'IIM_raw', 'IIM_raw_scaled',
                'IIM_defined', 'IIM_undefined_reason',
            ])
        if compute_ci and set(valid_mpc_metrics).issubset(set(selected_metrics)):
            cols_empty.extend([
                'CI', *CI_STATUS_COLUMNS,
                'RAM_norm', 'PDI_norm', 'NAS_norm', 'IIM_norm', 'SRPI_norm',
            ])
        cols_empty.extend(MPC_EVIDENCE_COLUMNS)
    if not records:
        return pd.DataFrame(columns=cols_empty)

    df = pd.DataFrame.from_records(records)
    if not compute_mpc:
        keep_cols = [
            c
            for c in (*PROVENANCE_COLUMNS, *hardware_cols, 'subject', 'session', 'theta', 'S')
            if c in df.columns
        ]
        return df[keep_cols]

    ci_required = set(valid_mpc_metrics)
    ci_computable = compute_ci and ci_required.issubset(set(df.columns))
    if not ci_computable:
        ordered_cols = [c for c in PROVENANCE_COLUMNS if c in df.columns]
        ordered_cols.extend([c for c in hardware_cols if c in df.columns])
        ordered_cols.extend(['subject', 'session', 'theta', 'S'])
        ordered_cols.extend([m for m in valid_mpc_metrics if m in df.columns])
        for pdi_extra in (
            'PDI_anchor', 'PDI_task',
            'PDI_anchor_defined', 'PDI_task_defined',
            'PDI_anchor_reason', 'PDI_task_reason',
            'PDI_primary_endpoint', 'PDI_primary_source',
            'PDI_baseline_policy',
            'PDI_anchor_baseline_n_runs', 'PDI_task_baseline_n_runs',
            'PDI_anchor_baseline_paths', 'PDI_task_baseline_paths',
        ):
            if pdi_extra in df.columns:
                ordered_cols.append(pdi_extra)
        for extra in ('IIM_raw', 'IIM_raw_scaled', 'IIM_defined', 'IIM_undefined_reason'):
            if extra in df.columns:
                ordered_cols.append(extra)
        df = _finish_mpc_evidence(
            df, ci_reference, ci_weights, ci_reference_session, mpc_registry
        )
        ordered_cols.extend(c for c in MPC_EVIDENCE_COLUMNS if c in df.columns)
        return df[ordered_cols]

    # CI uses NAS directly. The legacy HypergraphSynergy statistic S is reported
    # separately (exploratory) and does not enter CI, so CI is theta-invariant.
    # Reference means are explicit: cohort high-state means by default, or an
    # external reference (dict / JSON path); unusable references make CI undefined.
    df, ci_refs = assemble_ci(
        df,
        reference=ci_reference,
        weights=ci_weights,
        eps=ci_eps,
        high_state_session=ci_reference_session,
    )
    log.info(
        "CI reference=%s means=%s; CI defined in %d/%d rows",
        df['CI_reference'].iloc[0] if len(df) else "na",
        {k: (round(v, 6) if np.isfinite(v) else None) for k, v in ci_refs.items()},
        int(df['CI_defined'].sum()),
        int(len(df)),
    )
    ordered_cols = [
        *[c for c in PROVENANCE_COLUMNS if c in df.columns],
        *[c for c in hardware_cols if c in df.columns],
        'subject', 'session', 'theta', 'S', 'CI',
        *CI_STATUS_COLUMNS,
        'RAM', 'PDI', 'NAS', 'IIM', 'SRPI',
        'PDI_anchor', 'PDI_task',
        'PDI_anchor_defined', 'PDI_task_defined',
        'PDI_anchor_reason', 'PDI_task_reason',
        'PDI_primary_endpoint', 'PDI_primary_source',
        'PDI_baseline_policy',
        'PDI_anchor_baseline_n_runs', 'PDI_task_baseline_n_runs',
        'PDI_anchor_baseline_paths', 'PDI_task_baseline_paths',
        'IIM_raw', 'IIM_raw_scaled',
        'IIM_defined', 'IIM_undefined_reason',
        'RAM_norm', 'PDI_norm', 'NAS_norm', 'IIM_norm', 'SRPI_norm',
        *MPC_EVIDENCE_COLUMNS,
    ]
    df = _finish_mpc_evidence(
        df, ci_reference, ci_weights, ci_reference_session, mpc_registry
    )
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    return df[ordered_cols]
