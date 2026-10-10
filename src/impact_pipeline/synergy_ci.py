import os
import glob
import json
import logging
import concurrent.futures
import contextlib
import subprocess
import hashlib
import gc
import functools
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
from impact_pipeline.mpc_metrics import (
    ESTIMATOR_VERSIONS,
    SURROGATE_METHODS,
    compute_RAM,
    compute_PDI,
    compute_NAS,
    compute_IIM,
    compute_SRPI,
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

# MPC evidence layer (see impact_pipeline.evidence). Null families of the
# legacy estimator modes (run when null_surrogates > 0): PDI, NAS and IIM use
# their estimators' own surrogate calibration (methods of
# mpc_metrics.SURROGATE_METHODS); RAM and SRPI use
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
# Opt-in construct-revision modes of the estimators (mpc_metrics) and their
# options, selected through the Protocol's ``estimators`` or the params dicts
# (``bearer_nodes`` is handled separately). Defaults keep the legacy modes.
MPC_MODE_KEYS = {
    "RAM": ("update", "update_fallback", "impact_channel", "adaptation_locus"),
    "PDI": ("mode", "excess_components", "excess_weights", "excess_surrogate",
            "repertoire_window", "repertoire_features", "repertoire_folds",
            "repertoire_gap", "repertoire_components", "repertoire_max_states",
            "repertoire_null", "repertoire_criterion", "repertoire_valley",
            "repertoire_min_dwell"),
    "NAS": ("mode", "transfer_lags", "transfer_components", "workspace_nodes"),
    "IIM": ("cut_mode", "tpm_estimator", "node_selection", "state_budget_policy",
            "psi_kernel"),
    "SRPI": ("mode", "mode_fallback", "agency_null_permutations",
             "agency_components", "agency_pre_components", "agency_random_state"),
}
# Declared per-run fallbacks of modes that need inputs a run may lack (not a
# silent fallback: the rule is part of the protocol, and every row records the
# mode used in <P>_estimator and why in <P>_mode_reason). Keys: principle ->
# (mode key, fallback key); the requirement of each primary mode: (reason,
# test on the run's event bundle).
MPC_MODE_FALLBACK_KEYS = {
    "RAM": ("update", "update_fallback"),
    "SRPI": ("mode", "mode_fallback"),
}


def _has_choice_reward_log(bundle):
    b = bundle if isinstance(bundle, dict) else {}
    return any(len(b.get(k) or []) > 0 for k in ("choice_onsets", "choices", "rewards"))


def _has_agency_events(bundle):
    b = bundle if isinstance(bundle, dict) else {}
    return b.get("agency_events") is not None


MPC_MODE_REQUIREMENTS = {
    ("RAM", "prediction_error"): ("no_choice_reward_log", _has_choice_reward_log),
    ("SRPI", "agency"): ("no_agency_events", _has_agency_events),
}
# Modes that compute their own null family inside the estimator (used for the
# evidence whatever null_surrogates is; null_surrogates=0 selects the
# estimator's default size) and the family they declare.
MPC_MODE_NULL_FAMILIES = {
    ("NAS", "capacity"): "block_circular_shift",
    ("SRPI", "agency"): "yoked_label_permutation",
}
# Bootstrap replicates need only the evidence statistic. These internal null
# sizes keep them cheap without changing it (the raw PDI/SRPI statistics and
# the per-direction NAS transfer entropies do not depend on the null draws).
MPC_BOOTSTRAP_NULL_OVERRIDES = {
    ("PDI", "surrogate_excess"): {"null_surrogates": 2},
    ("PDI", "repertoire"): {"null_surrogates": 2},
    ("NAS", "capacity"): {"null_surrogates": 3},
    ("SRPI", "agency"): {"agency_null_permutations": 5},
}
# A bootstrap SE needs at least two valid replicates and at least this
# fraction of the replicates valid: when the estimator is undefined on most
# resamples, the SD of the few survivors is neither stable nor unbiased (the
# failures are not random), so the component has no sampling SE
# (NO_SAMPLING_SE). ``<P>_boot_failed`` reports the failed replicates.
MPC_BOOTSTRAP_MIN_VALID_FRACTION = 0.5
# MPC degree: capped power mean (p=0: geometric) of construct-scale components.
MPC_DEGREE_P = 0.0
MPC_DEGREE_CAP = 1.0
MPC_EVIDENCE_FIELDS = (
    "status",
    "margin",
    "estimate",
    "null_mean",
    "null_sd",
    "null_n",
    "se",
    "se_df",
    "boot_n",
    "boot_failed",
    "c",
    "c_se",
    "c_df",
    "c_lower",
    "c_upper",
    "margin_absent",
    "reference",
    "reference_se",
    "estimator",
    "estimator_version",
    "mode_reason",
    "channels",
)
MPC_VERDICT_COLUMNS = (
    "MPC_verdict",
    "MPC_reason",
    "MPC_degree",
    "MPC_necessity_set",
    "MPC_null_surrogates",
    "MPC_null_seed",
    "MPC_null_families",
    "MPC_bootstrap_se",
    "MPC_bootstrap_block_len",
    "MPC_protocol_hash",
    "MPC_joint_dependence",
    "MPC_joint_dependence_p",
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


def assemble_mpc_degree(df, weights=None, p=MPC_DEGREE_P, cap=MPC_DEGREE_CAP):
    """
    Add ``MPC_degree`` to a per-run table that carries the evidence columns.

    For MPC_CONSISTENT rows the degree is the capped power mean
    (``evidence.degree``) of the construct-scale components ``<P>_c`` over the
    row's necessity set (``MPC_necessity_set``), with ``weights`` (a dict)
    restricted to that set; NaN for every other verdict and when no weight in
    the set is positive. It is a reference-relative evidence summary, not a
    level of consciousness.
    """
    out = df.copy()
    consistent = mpc_evidence.Verdict.MPC_CONSISTENT.value
    degrees = []
    for _, row in out.iterrows():
        if str(row.get("MPC_verdict")) != consistent:
            degrees.append(float("nan"))
            continue
        nset = mpc_evidence.normalize_necessity_set(str(row.get("MPC_necessity_set")))
        comps = {k: _as_float(row.get(f"{k}_c")) for k in nset}
        w = None
        if weights is not None:
            w = {k: float(weights.get(k, 0.0)) for k in nset}
            if not any(v > 0 for v in w.values()):
                degrees.append(float("nan"))
                continue
        degrees.append(mpc_evidence.degree(comps, w, p=p, cap=cap))
    out["MPC_degree"] = degrees
    return out


def _resolve_null_kinds(null_kinds):
    kinds = dict(MPC_NULL_KINDS_DEFAULT)
    if null_kinds:
        unknown = sorted(set(null_kinds) - set(CI_COMPONENTS))
        if unknown:
            raise ValueError(f"null_kinds has unknown components: {unknown}")
        kinds.update({str(k): str(v) for k, v in dict(null_kinds).items()})
    return _check_null_kinds(kinds)


def _check_null_kinds(kinds):
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
    return mpc_evidence.resolve_registry(registry)


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
    channel="default",
    nodes=None,
    null_attempted=False,
):
    """Per-run evidence record of one component channel (estimate + its null)."""
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
        "null_attempted": bool(null_attempted),
        "channel": str(channel),
        "nodes": None if nodes is None else tuple(int(i) for i in nodes),
        "se": float("nan"),
        "boot_n": 0,
        "boot_failed": 0,
    }


def _record_from_null_fields(
    prefix, det, estimator, family, channel="default", nodes=None,
    null_attempted=False,
):
    """Evidence record from a details dict with ``<prefix>_null_*`` fields."""
    if not isinstance(det, dict):
        return _component_record(det, estimator=estimator, channel=channel,
                                 nodes=nodes)
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
        channel=channel,
        nodes=nodes,
        null_attempted=null_attempted,
    )


def _calibrated_value(rec):
    """Excess over the null, floored at 0 (NaN when uncalibrated)."""
    if not (np.isfinite(rec["estimate"]) and np.isfinite(rec["null_mean"])):
        return float("nan")
    return float(max(rec["estimate"] - rec["null_mean"], 0.0))


def _metric_value(recs, null_k):
    """
    Legacy metric column of a component: the excess over the null floored at
    0 when a null was run (always for estimator modes with their own null; at
    ``null_k > 0`` a failed null gives NaN, never the raw value), else the
    estimate. Several channels: the largest finite value.
    """
    vals = []
    for rec in recs:
        if rec["null_n"] > 0 or null_k > 0 or rec["null_attempted"]:
            vals.append(_calibrated_value(rec))
        else:
            vals.append(rec["estimate"])
    vals = [v for v in vals if np.isfinite(v)]
    return float(max(vals)) if vals else float("nan")


def _legacy_pdi_value(det, null_k):
    """
    PDI of the legacy-baseline fallback: the estimator value at K=0 (unchanged
    behaviour), and at K>0 the excess over the null floored at 0 like every
    calibrated metric column (``PDI_calibrated`` is not floored when the
    configured ``clip_negative`` is False).
    """
    if int(null_k) > 0:
        excess = _as_float(det.get("PDI_excess"))
        return float(max(excess, 0.0)) if np.isfinite(excess) else float("nan")
    return float(det["value"])


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
        null_attempted=True,
    )
    log.debug(
        "%s null (%s, n=%d, failed=%d): estimate=%.6g null_mean=%.6g null_sd=%.6g",
        principle, kind, int(n), int(res.n_failed), rec["estimate"],
        rec["null_mean"], rec["null_sd"],
    )
    return rec


def _se_df(rec):
    """
    Degrees of freedom of a record's bootstrap SE (valid replicates - 1), so
    that the evidence layer uses the Student-t quantile for SEs from few
    replicates; None without a usable SE.
    """
    if not np.isfinite(_as_float(rec.get("se"))) or int(rec.get("boot_n", 0)) < 2:
        return None
    return float(int(rec["boot_n"]) - 1)


def _bootstrap_se_usable(n_valid, n_failed):
    """At least two valid replicates and MPC_BOOTSTRAP_MIN_VALID_FRACTION."""
    n_valid, n_failed = int(n_valid), int(n_failed)
    total = n_valid + n_failed
    return n_valid >= 2 and n_valid >= MPC_BOOTSTRAP_MIN_VALID_FRACTION * total


def _attach_bootstrap(rec, fn, ts, events, n_boot, seed, block_len, tr,
                      statistic=None):
    """Block-bootstrap sampling SE of a defined record's statistic."""
    if int(n_boot) <= 0 or not rec["defined"]:
        return rec
    res = nulls.component_bootstrap_se(
        fn, ts, events, n=int(n_boot), seed=int(seed), block_len=block_len,
        tr=float(tr), statistic=statistic,
    )
    rec = dict(rec)
    se = float(res.se)
    if not _bootstrap_se_usable(res.samples.size, res.n_failed):
        log.warning(
            "bootstrap SE of %s withheld: %d of %d replicates failed "
            "(NO_SAMPLING_SE)", rec["estimator"], int(res.n_failed), int(n_boot),
        )
        se = float("nan")
    rec.update(se=se, boot_n=int(res.samples.size), boot_failed=int(res.n_failed))
    log.debug(
        "bootstrap SE (%s, n=%d, failed=%d, block=%d): se=%.6g",
        rec["estimator"], int(n_boot), int(res.n_failed), int(res.block_len),
        rec["se"],
    )
    return rec


def _raw_statistic(details):
    """Evidence statistic of PDI/NAS/SRPI details: the raw estimator value."""
    return _as_float(details.get("raw", details.get("value")))


def _transfer_statistic(details, direction):
    """NAS-capacity statistic: the transfer entropy of one direction (nats)."""
    return _as_float(details["transfer"][f"te_{direction}"])


def _iim_delta_psi(info):
    """Integration mass Delta_Psi (bits) of an IIM details dict."""
    delta_psi = info.get("Delta_Psi")
    if delta_psi is None:
        delta_psi = _as_float(info.get("Psi_full")) - _as_float(
            info.get("Psi_mip_preserved")
        )
    return _as_float(delta_psi)


_NULL_NOT_RUN = "not_run"


# Declared IIM options (protocol / params) and the details field that records
# the setting actually used; checked for every IIM result (a precomputed Hunter
# reduction is computed without the protocol's options).
_IIM_OPTION_FIELDS = {
    "cut_mode": "cut_mode",
    "tpm_estimator": "tpm_estimator",
    "node_selection": "node_selection_rule",
    "state_budget_policy": "state_budget_policy",
}


def _iim_option_mismatch(iim_info, declared, bearer):
    """First declared IIM option that the result was not computed with."""
    for key, field_name in _IIM_OPTION_FIELDS.items():
        want = dict(declared or {}).get(key)
        got = iim_info.get(field_name)
        if want is not None and got is not None and str(got) != str(want):
            return f"{key}={want}/{got}"
    if "bearer_nodes" in iim_info:
        got = iim_info.get("bearer_nodes")
        got = None if got is None else tuple(sorted(int(i) for i in got))
        want = None if bearer is None else tuple(sorted(int(i) for i in bearer))
        if got != want:
            return "bearer_nodes"
    return None


def _iim_record(iim_info, null_kind, cut_mode="bidirectional", nodes=None,
                declared=None, bearer=None, tr=None):
    """
    IIM evidence on the integration-mass scale Delta_Psi (bits). A result
    computed with other options than the declared ones (``declared``: IIM
    mode options, ``bearer``: IIM bearer nodes) is undefined
    (``iim_option_mismatch:<option>``), never judged under the protocol.
    """
    defined = bool(iim_info.get("defined", False))
    null_n = int(iim_info.get("IIM_null_n", 0) or 0)
    version = iim_info.get("iim_algorithm_version") or ESTIMATOR_VERSIONS["IIM"]
    cut = iim_info.get("cut_mode") or cut_mode
    reason = iim_info.get("undefined_reason")
    mismatch = _iim_option_mismatch(iim_info, declared, bearer) if defined else None
    if mismatch is not None:
        defined, reason = False, f"iim_option_mismatch:{mismatch}"
    rec = _component_record(
        _iim_delta_psi(iim_info) if defined else np.nan,
        reason=reason,
        estimator=mpc_evidence.estimator_id("IIM", cut, version),
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
        nodes=nodes,
    )
    # The registry regime of IIM is the subsystem and the series actually
    # scored: after a time subsampling of the IIM input (iim_max_timepoints)
    # its length is n_transitions + lag and its sample interval step * tr.
    selected = iim_info.get("selected_nodes")
    n_time = iim_info.get("n_time_used")
    if n_time is None and iim_info.get("n_transitions") is not None:
        n_time = int(iim_info["n_transitions"]) + int(iim_info.get("lag_trs") or 1)
    step = iim_info.get("iim_time_step")
    rec["regime"] = {
        k: v for k, v in (
            ("n_nodes", None if selected is None else len(selected)),
            ("bins", iim_info.get("bins_used")),
            ("n_time", n_time),
            ("tr", None if (tr is None or not step) else float(tr) * int(step)),
        ) if v is not None
    }
    rec["null_attempted"] = rec["null_reason"] != _NULL_NOT_RUN and bool(
        iim_info.get("IIM_null_undefined_reason") or null_n
    )
    se = _as_float(iim_info.get("Delta_Psi_bootstrap_se"))
    boot_n = int(iim_info.get("Delta_Psi_bootstrap_n", 0) or 0)
    boot_failed = int(iim_info.get("Delta_Psi_bootstrap_failed", 0) or 0)
    if boot_n or boot_failed:
        rec["boot_n"], rec["boot_failed"] = boot_n, boot_failed
    if np.isfinite(se) and (
        "Delta_Psi_bootstrap_n" not in iim_info
        or _bootstrap_se_usable(boot_n, boot_failed)
    ):
        rec["se"] = se
    return rec


# --------------------------------------------------------------------------
# protocol, modes and bearers of a compute_synergy_ci call
# --------------------------------------------------------------------------
def _mode_of(principle, opts):
    if principle == "RAM":
        return str(opts.get("update") or "feedback_magnitude")
    if principle == "IIM":
        return str(opts.get("cut_mode") or "bidirectional")
    return str(opts.get("mode") or "legacy")


def _nas_hub_missing(nas_kw, opts):
    """
    NAS ``mode='capacity'`` without a declared hub: ``workspace_nodes`` is
    neither in the protocol's NAS options nor in ``nas_params``. The
    estimator requires the hub (an invalid declaration still raises there);
    the pipeline records the component as UNDEFINED
    (``UNDEFINED:NAS:NO_DECLARED_WORKSPACE``) instead of calling it.
    """
    return (
        _mode_of("NAS", opts) == "capacity"
        and dict(nas_kw or {}).get("workspace_nodes") is None
    )


def nas_hub_missing(protocol=None, nas_params=None):
    """
    Whether :func:`compute_synergy_ci` records NAS as UNDEFINED
    (``UNDEFINED:NAS:NO_DECLARED_WORKSPACE``) in every run under this protocol
    and ``nas_params``: NAS ``mode='capacity'`` (from the protocol's NAS
    options or ``nas_params``) without a declared hub (``workspace_nodes``).
    Readiness and definedness reports use it so that they agree with the
    evidence layer. Conflicting options raise ValueError, as in the pipeline.
    """
    proto = mpc_evidence.resolve_protocol(protocol)
    params_opts, _ = _params_modes("NAS", nas_params)
    proto_opts = proto.estimator_options("NAS") if proto is not None else {}
    opts = _merge_estimator_options("NAS", params_opts, proto_opts)
    return _nas_hub_missing({**dict(nas_params or {}), **opts}, opts)


def _uses_internal_null(principle, opts):
    mode = _mode_of(principle, opts)
    return (
        (principle == "PDI" and mode != "legacy")
        or (principle, mode) in MPC_MODE_NULL_FAMILIES
    )


def _internal_null_family(principle, opts):
    mode = _mode_of(principle, opts)
    if (principle, mode) in MPC_MODE_NULL_FAMILIES:
        return MPC_MODE_NULL_FAMILIES[(principle, mode)]
    if principle == "PDI" and mode == "surrogate_excess":
        return str(opts.get("excess_surrogate") or "fourier")
    if principle == "PDI" and mode == "repertoire":
        # unlabelled repertoire: the declared state-count null ('both' reports
        # the binding family per run, so it is not pre-declared)
        fam = str(opts.get("repertoire_null") or "circular_shift")
        return None if fam == "both" else fam
    return None  # unknown mode: the family the estimator reports is recorded


# PDI mode='repertoire' options that need per-run inputs (the labelled
# variant); compute_synergy_ci runs the unlabelled variant only.
PDI_PER_RUN_OPTIONS = ("state_labels", "events", "tr")


def _run_mode_options(principle, opts, bundle):
    """
    Estimator options of one run: with a declared fallback
    (``update_fallback`` for RAM, ``mode_fallback`` for SRPI) the primary mode
    is replaced by the fallback when the run's events lack its inputs (RAM
    ``prediction_error``: a choice/reward log; SRPI ``agency``:
    self_caused/other_caused events). Returns ``(options without the
    fallback key, reason or None)``; the reason is
    ``<requirement>:<primary>-><fallback>``.
    """
    opts = dict(opts or {})
    spec = MPC_MODE_FALLBACK_KEYS.get(principle)
    if spec is None:
        return opts, None
    key, fb_key = spec
    fallback = opts.pop(fb_key, None)
    if fallback is None:
        return opts, None
    primary = _mode_of(principle, opts)
    req = MPC_MODE_REQUIREMENTS.get((principle, primary))
    if req is None or req[1](bundle):
        return opts, None
    opts[key] = str(fallback)
    return opts, f"{req[0]}:{primary}->{fallback}"


def _check_mode_fallback(principle, opts):
    """A declared fallback must be a valid mode and follow a mode with
    requirements (anything else would never be used)."""
    spec = MPC_MODE_FALLBACK_KEYS.get(principle)
    if spec is None or spec[1] not in dict(opts or {}):
        return
    from impact_pipeline.mpc_metrics import RAM_UPDATE_MODES, SRPI_MODES

    valid = RAM_UPDATE_MODES if principle == "RAM" else SRPI_MODES
    fallback = opts[spec[1]]
    if fallback not in valid:
        raise ValueError(f"{principle} {spec[1]} must be one of {valid}")
    primary = _mode_of(principle, opts)
    if (principle, primary) not in MPC_MODE_REQUIREMENTS:
        raise ValueError(
            f"{principle} {spec[1]} needs a primary mode with requirements "
            f"({sorted(m for p, m in MPC_MODE_REQUIREMENTS if p == principle)}), "
            f"not {primary!r}"
        )


def _check_pdi_options(opts):
    bad = sorted(k for k in PDI_PER_RUN_OPTIONS if k in dict(opts or {}))
    if bad:
        raise ValueError(
            f"PDI options {bad} are per-run inputs of the labelled repertoire "
            "variant; compute_synergy_ci runs mode='repertoire' unlabelled "
            "(state labels are not part of a protocol)"
        )


def _same_option(a, b):
    try:
        return json.dumps(a, sort_keys=True, default=str) == json.dumps(
            b, sort_keys=True, default=str
        )
    except TypeError:
        return a == b


def _merge_estimator_options(principle, from_params, from_protocol):
    """Mode options of one principle from the params dict and the Protocol."""
    out = {k: v for k, v in dict(from_params).items() if v is not None}
    for k, v in dict(from_protocol).items():
        if k in out and not _same_option(out[k], v):
            raise ValueError(
                f"{principle} option {k!r} differs between the params "
                f"({out[k]!r}) and the protocol ({v!r})"
            )
        out[k] = v
    return out


def _params_modes(principle, params):
    params = dict(params or {})
    opts = {k: params[k] for k in MPC_MODE_KEYS[principle] if k in params}
    bearer = params.get("bearer_nodes")
    return opts, (None if bearer is None else mpc_evidence._node_tuple(
        bearer, f"{principle} bearer_nodes"))


def _resolve_mpc_setup(
    protocol, necessity_set, null_kinds, ci_reference_session, params_by_p,
):
    """
    Effective Protocol, estimator mode options, bearer node sets and legacy
    null kinds of a compute_synergy_ci call. Without a protocol a default one
    is built from the keywords (so its hash covers the modes, bearers and null
    families in use); with a protocol, conflicting keywords raise ValueError.
    """
    proto_in = mpc_evidence.resolve_protocol(protocol)
    if proto_in is not None and necessity_set is not None:
        if mpc_evidence.normalize_necessity_set(necessity_set) != (
            proto_in.necessity_set
        ):
            raise ValueError("necessity_set differs from the protocol's necessity set")
    modes, bearers = {}, {}
    for p in CI_COMPONENTS:
        p_opts, p_bearer = _params_modes(p, params_by_p.get(p))
        pr_opts = proto_in.estimator_options(p) if proto_in is not None else {}
        modes[p] = _merge_estimator_options(p, p_opts, pr_opts)
        if p == "PDI":
            _check_pdi_options(modes[p])
        _check_mode_fallback(p, modes[p])
        pr_bearer = proto_in.bearer_nodes.get(p) if proto_in is not None else None
        if p_bearer is not None and pr_bearer is not None and p_bearer != pr_bearer:
            raise ValueError(f"{p} bearer_nodes differ between the params and protocol")
        bearers[p] = pr_bearer if pr_bearer is not None else p_bearer
    kinds = _resolve_null_kinds(null_kinds)
    if proto_in is not None:
        for p, fam in proto_in.null_families.items():
            if _uses_internal_null(p, modes[p]):
                continue
            if null_kinds and p in null_kinds and str(null_kinds[p]) != fam:
                raise ValueError(f"null_kinds[{p}] differs from the protocol's family")
            kinds[p] = fam
        kinds = _check_null_kinds(kinds)
    for p in CI_COMPONENTS:
        if _uses_internal_null(p, modes[p]) and null_kinds and p in null_kinds:
            raise ValueError(
                f"{p} mode {_mode_of(p, modes[p])!r} computes its own null family; "
                "null_kinds does not apply"
            )
    if proto_in is not None:
        return proto_in, modes, bearers, kinds
    families = {}
    for p in CI_COMPONENTS:
        spec = MPC_MODE_FALLBACK_KEYS.get(p)
        if spec is not None and spec[1] in modes[p]:
            continue  # the family depends on the mode each run uses
        fam = (
            _internal_null_family(p, modes[p])
            if _uses_internal_null(p, modes[p]) else kinds[p]
        )
        if fam is not None:
            families[p] = fam
    proto = mpc_evidence.Protocol(
        necessity_set=necessity_set,
        null_families=families,
        reference={"kind": "cohort_high_state", "session": str(ci_reference_session)},
        estimators={p: o for p, o in modes.items() if o},
        bearer_nodes={p: b for p, b in bearers.items() if b is not None},
    )
    return proto, modes, bearers, kinds


def _mpc_references(runs, proto):
    """
    Reference anchors per (principle, channel) from the protocol: the cohort
    high-state mean excess ``estimate - null_mean`` (subject means first; SE =
    SD of the subject means / sqrt(n) with >= 2 subjects, else unavailable) on
    the excess scale, or the protocol's external values.
    """
    ref = proto.reference
    keys = {
        (p, rec["channel"]) for run in runs for p, recs in run["records"].items()
        for rec in recs
    }
    out = {}
    if ref["kind"] == "external":
        for p, ch in keys:
            key = f"{p}:{ch}" if f"{p}:{ch}" in ref["values"] else p
            out[(p, ch)] = {
                "reference": ref["values"].get(key, float("nan")),
                "reference_se": ref["se"].get(key, float("nan")),
                "scale": ref["scale"],
                "n_subjects": 0,
            }
        return out
    session = str(ref["session"])
    for p, ch in keys:
        by_subject = {}
        for run in runs:
            if str(run["session"]) != session:
                continue
            for rec in run["records"].get(p, []):
                if rec["channel"] != ch or not rec["defined"]:
                    continue
                excess = rec["estimate"] - rec["null_mean"]
                if np.isfinite(excess):
                    by_subject.setdefault(str(run["subject"]), []).append(excess)
        means = np.asarray([np.mean(v) for v in by_subject.values()], dtype=float)
        n = int(means.size)
        out[(p, ch)] = {
            "reference": float(means.mean()) if n else float("nan"),
            "reference_se": (
                float(means.std(ddof=1) / np.sqrt(n)) if n >= 2 else float("nan")
            ),
            "scale": "excess",
            "n_subjects": n,
        }
    return out


def _joint_dependence_label(jd, n_sets):
    if n_sets <= 1:
        return ""
    if jd is None:
        return "untested"
    return "dependent" if jd.get("dependent") else str(
        jd.get("reason") or mpc_evidence.REASON_SOURCE_INCOHERENT
    )


def _item_regime(rec):
    """
    Per-component regime for the registry: the size of the component's bearer
    node set, or for IIM the subsystem and series actually scored (nodes,
    bins and, after time subsampling, n_time and tr).
    """
    regime = {} if rec["nodes"] is None else {"n_nodes": len(rec["nodes"])}
    regime.update(rec.get("regime") or {})
    return regime or None


def _estimator_version(estimator):
    """
    ``<P>_estimator_version``: the version recorded in the evidence estimator
    id ``compute_<P>:<mode>@<version>`` (ComponentEvidence.estimator), e.g.
    the ``iim_algorithm_version`` of a precomputed Hunter IIM result, not the
    current code's :data:`ESTIMATOR_VERSIONS` entry. Empty when the principle
    was not computed or the id carries no version.
    """
    return mpc_evidence.split_estimator(estimator)[1] or ""


def _mpc_run_columns(run, proto, registry, refs, null_k, null_seed, boot_k,
                     boot_block_len):
    """ComponentEvidence, the MPC verdict and the evidence columns of one run."""
    evidence = {}
    for p, recs in run["records"].items():
        items = []
        for rec in recs:
            defined, reason = rec["defined"], rec["reason"]
            if defined and rec["null_attempted"] and rec["null_n"] == 0:
                # A null family was requested but produced no valid surrogate.
                defined = False
                reason = (
                    f"null_undefined:{rec.get('null_reason') or 'no_valid_surrogates'}"
                )
            ref = refs.get((p, rec["channel"]), {})
            items.append(
                mpc_evidence.ComponentEvidence(
                    principle=p,
                    estimate=rec["estimate"],
                    null_mean=rec["null_mean"],
                    null_sd=rec["null_sd"],
                    se=rec["se"],
                    se_df=_se_df(rec),
                    channel=rec["channel"],
                    defined=defined,
                    reason=reason,
                    reference=ref.get("reference"),
                    reference_se=ref.get("reference_se"),
                    reference_scale=ref.get("scale", "excess"),
                    bearer_id=run["bearer_id"],
                    protocol_id=proto.protocol_id,
                    substrate=run["substrate"],
                    grain=run["grain"],
                    estimator=rec["estimator"],
                    null_family=rec["null_family"],
                    n_null=int(rec["null_n"]),
                    nodes=rec["nodes"],
                    regime=_item_regime(rec),
                )
            )
        evidence[p] = items
    verdict = mpc_evidence.mpc_verdict(
        evidence, proto, registry=registry, regime=run["regime"],
        joint_dependence=run["joint"],
    )
    families = ";".join(
        f"{p}:{rec['null_family']}"
        for p in CI_COMPONENTS
        for rec in run["records"].get(p, [])
        if rec["null_n"] > 0 and rec["null_family"]
    )
    jd = run["joint"]
    cols = {
        "MPC_verdict": verdict.verdict.value,
        "MPC_reason": verdict.reason_string,
        "MPC_degree": float("nan"),
        "MPC_necessity_set": ",".join(verdict.necessity_set),
        "MPC_null_surrogates": int(null_k),
        "MPC_null_seed": int(null_seed),
        "MPC_null_families": families,
        "MPC_bootstrap_se": int(boot_k),
        "MPC_bootstrap_block_len": (
            float("nan") if boot_block_len is None else int(boot_block_len)
        ),
        "MPC_protocol_hash": proto.hash,
        "MPC_joint_dependence": _joint_dependence_label(jd, run["n_node_sets"]),
        "MPC_joint_dependence_p": (
            float("nan") if jd is None else _as_float(jd.get("p"))
        ),
    }
    undefined = mpc_evidence.ComponentStatus.UNDEFINED
    nan = float("nan")
    for k in CI_COMPONENTS:
        recs = run["records"].get(k, [])
        a = verdict.principle_assessment.get(k)
        ch = next(
            (c for c, x in verdict.assessments.get(k, {}).items() if x is a), None
        )
        rec = next((r for r in recs if r["channel"] == ch), recs[0] if recs else None)
        ref = refs.get((k, ch), {}) if ch is not None else {}
        cols[f"{k}_status"] = verdict.component_status.get(k, undefined).value
        # presence margin of the deciding channel (the one c, c_lower, ...
        # describe), not the largest margin over the channels
        cols[f"{k}_margin"] = nan if a is None else float(a.margin_present)
        cols[f"{k}_estimate"] = nan if rec is None else rec["estimate"]
        cols[f"{k}_null_mean"] = nan if rec is None else rec["null_mean"]
        cols[f"{k}_null_sd"] = nan if rec is None else rec["null_sd"]
        cols[f"{k}_null_n"] = 0 if rec is None else int(rec["null_n"])
        cols[f"{k}_se"] = nan if rec is None else rec["se"]
        se_df = None if rec is None else _se_df(rec)
        cols[f"{k}_se_df"] = nan if se_df is None else se_df
        cols[f"{k}_boot_n"] = 0 if rec is None else int(rec["boot_n"])
        cols[f"{k}_boot_failed"] = 0 if rec is None else int(rec["boot_failed"])
        cols[f"{k}_c"] = nan if a is None else float(a.c)
        cols[f"{k}_c_se"] = nan if a is None else float(a.se)
        cols[f"{k}_c_df"] = nan if a is None else float(a.df)
        cols[f"{k}_c_lower"] = nan if a is None else float(a.lower)
        cols[f"{k}_c_upper"] = nan if a is None else float(a.upper)
        cols[f"{k}_margin_absent"] = nan if a is None else float(a.margin_absent)
        cols[f"{k}_reference"] = _as_float(ref.get("reference"))
        cols[f"{k}_reference_se"] = _as_float(ref.get("reference_se"))
        cols[f"{k}_estimator"] = "" if rec is None else str(rec["estimator"] or "")
        # the version part of that exact evidence id (compute_<P>:<mode>@<version>),
        # as scripts/run_predictions.py checks it against the registry
        cols[f"{k}_estimator_version"] = _estimator_version(
            None if rec is None else rec["estimator"]
        )
        cols[f"{k}_mode_reason"] = (
            "" if rec is None else str(rec.get("mode_reason") or "")
        )
        cols[f"{k}_channels"] = ",".join(
            f"{c}:{s.value}" for c, s in verdict.channels.get(k, {}).items()
        )
    return cols


def _channel_evidence_records(runs):
    """
    Every run's evidence record per principle and channel (JSON-safe dicts):
    what the ``<P>_*`` columns show only for the deciding channel. A requested
    null without a valid surrogate counts as undefined, as in the verdict.
    """
    out = []
    for run in runs:
        for p in CI_COMPONENTS:
            for rec in run["records"].get(p, []):
                defined, reason = bool(rec["defined"]), rec["reason"]
                if defined and rec["null_attempted"] and rec["null_n"] == 0:
                    defined = False
                    reason = (
                        "null_undefined:"
                        f"{rec.get('null_reason') or 'no_valid_surrogates'}"
                    )
                out.append({
                    "subject": str(run["subject"]),
                    "session": str(run["session"]),
                    "bearer_id": run["bearer_id"],
                    "principle": p,
                    "channel": str(rec["channel"]),
                    "estimate": _as_float(rec["estimate"]),
                    "null_mean": _as_float(rec["null_mean"]),
                    "null_sd": _as_float(rec["null_sd"]),
                    "null_n": int(rec["null_n"]),
                    "null_family": rec["null_family"],
                    "defined": defined,
                    "reason": None if reason is None else str(reason),
                    "estimator": (
                        None if rec["estimator"] is None else str(rec["estimator"])
                    ),
                    "mode_reason": str(rec.get("mode_reason") or ""),
                })
    return out


def _finish_mpc_evidence(
    df, runs, proto, registry, null_k, null_seed, boot_k, boot_block_len, weights,
    channel_evidence=False,
):
    """
    Verdict stage (after all runs, once the reference anchors are known): the
    evidence columns of every run, the MPC degree of MPC_CONSISTENT rows, a
    verdict summary log and the provenance in ``df.attrs['mpc_evidence']``
    (with ``channel_evidence``: every run's record per principle and channel).
    """
    if not runs:
        return df
    attrs = dict(df.attrs)
    refs = _mpc_references(runs, proto)
    rows = {}
    for run in runs:
        cols = _mpc_run_columns(run, proto, registry, refs, null_k, null_seed, boot_k,
                                boot_block_len)
        for i in run["rows"]:
            rows[i] = cols
    mpc_df = pd.DataFrame.from_dict(rows, orient="index")
    mpc_df = mpc_df.reindex(df.index)
    out = df.drop(columns=[c for c in mpc_df.columns if c in df.columns])
    out = pd.concat([out, mpc_df], axis=1)
    out = assemble_mpc_degree(out, weights=weights)
    counts = out["MPC_verdict"].value_counts().to_dict()
    log.info(
        "MPC verdicts (rows): %s; necessity set=%s; null surrogates=%s; "
        "bootstrap SE replicates=%s; protocol=%s",
        {k: int(v) for k, v in counts.items()},
        ",".join(proto.necessity_set), null_k, boot_k, proto.hash[:12],
    )
    out.attrs.update(attrs)
    out.attrs["mpc_evidence"] = {
        "protocol": proto.to_dict(),
        "protocol_hash": proto.hash,
        "degree_p": MPC_DEGREE_P,
        "degree_cap": MPC_DEGREE_CAP,
        "references": {
            f"{p}:{ch}": {k: (_as_float(v) if k != "scale" else v)
                          for k, v in r.items()}
            for (p, ch), r in sorted(refs.items())
        },
        "null_surrogates": int(null_k),
        "null_seed": int(null_seed),
        "bootstrap_se": int(boot_k),
        "bootstrap_block_len": boot_block_len,
        "estimator_versions": dict(ESTIMATOR_VERSIONS),
        "applicability_registry": (
            None if registry is None else registry.to_dict()
        ),
    }
    if channel_evidence:
        out.attrs["mpc_evidence"]["channel_evidence"] = (
            _channel_evidence_records(runs)
        )
    return out


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
        # macOS vm_stat: used = active + inactive + speculative + wired +
        # compressor pages, total = used + free (a rough figure for logging).
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
    iim_options=None,
    iim_bootstrap_n=0,
    iim_bootstrap_block_len=None,
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
    step = 1
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
        if iim_options:
            # Protocol-selected IIM options (cut mode, bearer, ...); default
            # checkpoints keep their historical names.
            sig += "|opts=" + json.dumps(iim_options, sort_keys=True, default=str)
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
    common = dict(
        bins=iim_bins,
        lag_trs=iim_lag_trs,
        n_parts=iim_n_parts,
        max_nodes=iim_max_nodes,
        max_mechanism_size=iim_max_mechanism_size,
        max_purview_size=iim_max_purview_size,
        partition_mode="all",
        clamp=True,
        return_details=True,
        phase1_parallel_workers=iim_phase1_parallel_workers,
        phase1_chunk_size=iim_phase1_chunk_size,
        phase1_shared_memory=bool(iim_phase1_shared_memory),
        hardware_backend=backend,
        **dict(iim_options or {}),
    )
    try:
        iim_info = compute_IIM(
            ts_iim,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=bool(iim_resume_checkpoint),
            checkpoint_every_cuts=int(iim_checkpoint_every_cuts),
            progress_log_every_cuts=int(iim_progress_log_every_cuts),
            progress_label=os.path.basename(ts_path),
            **common,
            **null_kwargs,
        )
        # the series IIM actually scored (for the registry regime)
        iim_info = dict(iim_info, iim_time_step=int(step),
                        n_time_used=int(ts_iim.shape[1]))
        if int(iim_bootstrap_n) > 0 and bool(iim_info.get("defined", False)):
            iim_info = _iim_bootstrap(
                iim_info, ts_iim, common, int(iim_bootstrap_n),
                nulls.derive_seed(int(iim_null_seed), "BOOT", "IIM",
                                  _run_null_key(ts_path)),
                iim_bootstrap_block_len, step,
            )
    finally:
        if shm is not None:
            shm.close()
    return ts_path, iim_info


def _release_shared_memory(handles):
    """Close and unlink shared-memory segments this process created."""
    for shm in handles:
        with contextlib.suppress(Exception):
            shm.close()
        with contextlib.suppress(Exception):
            shm.unlink()


def _iim_bootstrap(iim_info, ts_iim, common, n_boot, seed, block_len, step=1):
    """
    Block-bootstrap SE of Delta_Psi on the IIM input (after any time
    subsampling; ``block_len`` in run samples is rescaled accordingly). The
    subsystem is pinned to the nodes selected on the original data, so every
    replicate scores the same subsystem; replicates run without null or
    checkpoint.
    """
    kw = dict(common)
    selected = iim_info.get("selected_nodes")
    if selected is not None and kw.get("node_indices") is None:
        kw["node_indices"] = [int(i) for i in selected]
    if block_len is not None:
        block_len = max(1, int(np.ceil(int(block_len) / max(1, int(step)))))

    def _fn(x, _events):
        return compute_IIM(x, progress_log_every_cuts=10 ** 9, **kw)

    res = nulls.component_bootstrap_se(
        _fn, ts_iim, None, n=n_boot, seed=seed, block_len=block_len,
        statistic=lambda d: _iim_delta_psi(d) if d.get("defined") else float("nan"),
    )
    out = dict(iim_info)
    out.update(
        Delta_Psi_bootstrap_se=float(res.se),
        Delta_Psi_bootstrap_n=int(res.samples.size),
        Delta_Psi_bootstrap_failed=int(res.n_failed),
        Delta_Psi_bootstrap_block_len=int(res.block_len),
    )
    return out


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


def _resolve_nas_kwargs(nas_params, mode="legacy"):
    if str(mode) == "capacity":
        # NAS-capacity (receive-transform-return transfer between a declared
        # hub and the periphery) needs only the declared hub; the legacy
        # L/B/H settings are optional profile descriptors.
        params = {k: dict(nas_params or {}).get(k, NAS_PARAM_DEFAULTS[k])
                  for k in NAS_PARAM_KEYS}
        params["baseline_ts"] = None
        return params
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
    protocol=None,
    bootstrap_se=0,
    bootstrap_block_len=None,
    record_channel_evidence=False,
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
    ComponentEvidence per computed principle and channel, on the construct
    scale ``c = (m - nu) / (rho - nu)``, and a three-valued verdict (an
    exclusion rule: ``EXCLUDED`` / ``MPC_CONSISTENT`` / ``UNDETERMINED``) under
    a :class:`~impact_pipeline.evidence.Protocol`:

    - ``protocol`` (Protocol, dict or JSON path) declares the necessity set,
      channels, construct-scale cutoffs, alpha, null families, reference,
      source rule, estimator modes (``estimators``, e.g. SRPI
      ``mode='agency'``, NAS ``mode='capacity'``, PDI ``mode='repertoire'``
      (unlabelled repertoire of distinguishable states, options
      ``repertoire_*``; its evidence is ``raw`` in bits against the
      estimator's own state-count null, default ``circular_shift``), RAM
      ``update='prediction_error'``, with the declared fallbacks
      ``update_fallback`` / ``mode_fallback``) and per-component
      ``bearer_nodes``. Without it a default protocol is built
      from ``necessity_set`` (default all five), ``null_kinds``, the mode keys
      of the params dicts (:data:`MPC_MODE_KEYS`, plus ``bearer_nodes``) and
      the cohort reference of ``ci_reference_session``; conflicting keywords
      raise ValueError. Its SHA-256 is recorded in ``MPC_protocol_hash`` and
      ``df.attrs['mpc_evidence']``. RAM channels declared by the protocol are
      computed one by one (``impact_channel``).
    - ``null_surrogates=0`` (default): the legacy estimator modes run no null
      family (``NO_NULL_CALIBRATION:<P>``). With ``K > 0`` PDI, NAS and IIM
      are calibrated by their estimators, RAM and SRPI by
      ``nulls.component_null`` (see ``MPC_NULL_KINDS_DEFAULT``); modes with
      their own null (NAS capacity, SRPI agency, PDI repertoire and
      surrogate_excess) always use it (``K = 0`` selects the estimator's
      default size). Seeds are
      derived from ``null_seed`` and the run path. The metric columns
      ``RAM``..``SRPI`` hold the calibrated values (excess over the null mean,
      floored at 0; NaN when a requested null failed) whenever a null was
      run, so the legacy CI is then on the calibrated scale.
    - ``bootstrap_se=K_b`` (default 0): the sampling SE of each estimate from
      ``K_b`` moving-block bootstrap replicates of the run and its events
      (``nulls.component_bootstrap_se``; ``bootstrap_block_len`` samples,
      default ``ceil(sqrt(n_time))``), with ``B - 1`` degrees of freedom for
      ``B`` valid replicates (``<P>_se_df``; Student-t bounds of ``c``,
      ``<P>_c_df``). With ``K_b = 0`` every empirical
      component is UNDEFINED (``NO_SAMPLING_SE:<P>``) and the verdict is
      UNDETERMINED (the honest default). The SE also stays undefined when
      fewer than two replicates, or fewer than
      :data:`MPC_BOOTSTRAP_MIN_VALID_FRACTION` of them, are valid (the
      estimator is undefined on most resamples). Precomputed IIM results
      carry an SE only with ``Delta_Psi_bootstrap_se``.
    - Reference anchors come from the protocol only (``ci_reference`` is used
      for the legacy CI): by default the cohort high-state mean excess
      ``estimate - null_mean`` per component and channel (subject means
      first; its SE enters ``se_c`` with >= 2 subjects). The high-state runs
      are part of their own reference (their ``c`` averages 1 by
      construction, and a component that is not above its null in the high
      state has no construct scale, ``INVALID_ANCHORS``), so their verdicts
      are not independent tests of necessity: tests among report-positive
      (high-state) episodes need an external reference (``{"kind":
      "external", ...}``, e.g. from MPC-Bench).
    - Single-source constraint: when the components of the necessity set come
      from different node sets (``bearer_nodes``), ``evidence.joint_dependence``
      is run per run with ``null_surrogates`` circular-shift surrogates;
      without dependence above null the verdict is UNDETERMINED
      (``SOURCE_INCOHERENT``; ``:UNTESTED`` with ``K = 0``,
      ``:INSUFFICIENT_SURROGATES`` when ``1 / (K + 1)`` exceeds alpha).
    - ``applicability_registry`` (ApplicabilityRegistry, dict or JSON path):
      evidence from estimator versions not validated for the substrate
      (``modality``), grain (``atlas``) and regime (``modality``, ``n_time``,
      ``n_nodes`` of the component's bearer, ``tr``, ``bins``; for IIM the
      scored subsystem and series: selected nodes, bins used and, after
      ``iim_max_timepoints`` subsampling, its length and sample interval) is
      UNDEFINED (``ESTIMATOR_NOT_VALIDATED``).

    Columns: ``MPC_verdict``, ``MPC_reason`` (``;``-joined reason codes),
    ``MPC_degree`` (MPC_CONSISTENT rows only, see :func:`assemble_mpc_degree`),
    ``MPC_necessity_set``, ``MPC_null_surrogates``, ``MPC_null_seed``,
    ``MPC_null_families``, ``MPC_bootstrap_se``, ``MPC_bootstrap_block_len``,
    ``MPC_protocol_hash``, ``MPC_joint_dependence``/``_p`` and, per principle
    ``<P>`` (for the channel that decides its status), ``<P>_status``,
    ``<P>_margin`` (``c_lower - z``), ``<P>_margin_absent`` (``delta -
    c_upper``), ``<P>_estimate``, ``<P>_null_mean``, ``<P>_null_sd``,
    ``<P>_null_n``, ``<P>_se``, ``<P>_boot_n``, ``<P>_boot_failed``,
    ``<P>_c``, ``<P>_c_se``,
    ``<P>_c_lower``, ``<P>_c_upper``, ``<P>_reference`` (reference excess),
    ``<P>_reference_se``, ``<P>_estimator`` (``compute_<P>:<mode>@<version>``),
    ``<P>_estimator_version`` (the ``<version>`` of that id) and
    ``<P>_channels``. IIM evidence is on the integration-mass scale
    Delta_Psi (bits).

    NAS ``mode='capacity'`` (e.g. ``protocols/mpc_default_v1.json``, the
    default empirical protocol, which declares no hub) needs a declared hub
    (``workspace_nodes`` in the protocol's NAS options or in ``nas_params``,
    e.g. a derived protocol from ``protocols/examples/``); without one NAS is
    UNDEFINED (``UNDEFINED:NAS:NO_DECLARED_WORKSPACE``) in every run instead
    of raising.

    The ``<P>_*`` evidence columns describe only the channel that decides the
    principle's status. ``record_channel_evidence=True`` (default False; the
    pipeline's outputs are unchanged) also lists every run's evidence record
    per principle and declared channel in
    ``df.attrs['mpc_evidence']['channel_evidence']`` (subject, session,
    bearer, principle, channel, estimate, null moments, definedness, reason,
    estimator), e.g. for a per-channel reference anchor
    (``scripts/compute_empirical_reference.py``).
    """
    if ci_reference is None and ci_human_refs is not None:
        ci_reference = dict(ci_human_refs)
    null_k = int(null_surrogates)
    if null_k < 0:
        raise ValueError("null_surrogates must be >= 0")
    null_seed = int(null_seed)
    boot_k = int(bootstrap_se or 0)
    if boot_k < 0:
        raise ValueError("bootstrap_se must be >= 0")
    if bootstrap_block_len is not None and int(bootstrap_block_len) < 1:
        raise ValueError("bootstrap_block_len must be >= 1")
    boot_block_len = None if bootstrap_block_len is None else int(bootstrap_block_len)
    mpc_proto, mpc_modes, mpc_bearers, mpc_null_kinds = _resolve_mpc_setup(
        protocol, necessity_set, null_kinds, ci_reference_session,
        {"RAM": ram_params, "PDI": pdi_params, "NAS": nas_params,
         "SRPI": srpi_params},
    )
    mpc_nset = mpc_proto.necessity_set
    mpc_registry = _resolve_applicability_registry(applicability_registry)
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
            mode=_mode_of("NAS", mpc_modes["NAS"]),
        )
        if do_nas
        else {}
    )
    if do_nas and _nas_hub_missing({**nas_kwargs, **mpc_modes["NAS"]},
                                   mpc_modes["NAS"]):
        log.warning(
            "NAS mode='capacity' has no declared hub (workspace_nodes in the "
            "protocol or nas_params): NAS is UNDEFINED (%s) in every run. "
            "Declare the hub in a derived protocol (protocols/examples/).",
            mpc_evidence.REASON_NO_DECLARED_WORKSPACE,
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
    # node x time matrix) under one protocol (mpc_proto); the evidence of all
    # runs is collected first and judged once the reference anchors are known.
    mpc_runs = []
    boot_tr = 1.0 if tr is None else float(tr)
    ram_version = ESTIMATOR_VERSIONS["RAM"]
    pdi_version = ESTIMATOR_VERSIONS["PDI"]
    nas_version = ESTIMATOR_VERSIONS["NAS"]
    srpi_version = ESTIMATOR_VERSIONS["SRPI"]
    iim_options = dict(mpc_modes["IIM"])
    if mpc_bearers["IIM"] is not None:
        iim_options["bearer_nodes"] = list(mpc_bearers["IIM"])

    def _bearer_kw(principle):
        b = mpc_bearers[principle]
        return {} if b is None else {"bearer_nodes": list(b)}

    def _nodes(principle, n_nodes):
        b = mpc_bearers[principle]
        return tuple(range(int(n_nodes))) if b is None else tuple(b)

    def _select_pdi_rest_runs(subj, session_name):
        # Rest baselines live under <subj>/<session>/rest (fMRI preprocessing and,
        # when rest recordings exist, EEG preprocessing). Accept any run naming.
        rest_dir = os.path.join(data_dir, subj, session_name, 'rest')
        pattern = os.path.join(rest_dir, f"{subj}_*_{atlas}_ts.npy")
        return sorted(set(glob.glob(pattern)))

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
        same_session_rest = _select_pdi_rest_runs(subj, session_name)
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
                _release_shared_memory(shared_handles)
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
                        iim_options,
                        boot_k,
                        boot_block_len,
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
                                        iim_options,
                                        boot_k,
                                        boot_block_len,
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
            _release_shared_memory(shared_handles)
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
        n_nodes_run = int(ts_region_time.shape[0])
        mpc_records = {}
        run_joint = None
        n_node_sets = 0
        if compute_mpc:
            if do_ram:
                ram_opts, ram_mode_reason = _run_mode_options(
                    "RAM", mpc_modes["RAM"], subj_onsets
                )
                ram_declared = mpc_proto.channels_for("RAM")
                ram_channels = (
                    list(ram_declared) if ram_declared
                    else [str(ram_opts.get("impact_channel") or "default")]
                )
                ram_recs = []
                for ram_channel in ram_channels:
                    ram_kw = {**ram_kwargs, **ram_opts, **_bearer_kw("RAM")}
                    if ram_declared:
                        ram_kw["impact_channel"] = (
                            None if ram_channel == "default" else ram_channel
                        )
                    ram_det = _as_details(compute_RAM(
                        ts_region_time,
                        tr=tr,
                        stimulus_onsets=subj_onsets,
                        **ram_kw,
                        hardware_backend=hardware_backend,
                        return_details=True,
                    ))
                    ram_rec = _component_record(
                        ram_det.get("value"),
                        reason=ram_det.get("undefined_reason"),
                        estimator=mpc_evidence.estimator_id(
                            "RAM", _mode_of("RAM", ram_kw), ram_version
                        ),
                        channel=ram_channel,
                        nodes=_nodes("RAM", n_nodes_run),
                    )

                    def _ram_fn(ts_s, ev_s, _kw=ram_kw):
                        return compute_RAM(
                            ts_s,
                            tr=tr,
                            stimulus_onsets=ev_s,
                            **_kw,
                            hardware_backend=hardware_backend,
                        )

                    # Seeds of the default channel are those of earlier releases.
                    ch_key = () if ram_channel == "default" else (ram_channel,)
                    if null_k > 0 and ram_rec["defined"]:
                        ram_rec = _component_null_record(
                            "RAM", _ram_fn, ts_region_time, subj_onsets,
                            mpc_null_kinds["RAM"], null_k,
                            nulls.derive_seed(null_seed, "RAM", *ch_key, run_key),
                            ram_rec, tr,
                        )
                    ram_rec = _attach_bootstrap(
                        ram_rec, _ram_fn, ts_region_time, subj_onsets, boot_k,
                        nulls.derive_seed(null_seed, "BOOT", "RAM", *ch_key, run_key),
                        boot_block_len, boot_tr,
                    )
                    ram_rec["mode_reason"] = ram_mode_reason
                    ram_recs.append(ram_rec)
                mpc_records["RAM"] = ram_recs
                ram = _metric_value(ram_recs, null_k)
            else:
                ram = np.nan
            if do_pdi:
                pdi_opts = dict(mpc_modes["PDI"])
                pdi_mode = _mode_of("PDI", pdi_opts)
                pdi_nodes = _nodes("PDI", n_nodes_run)
                pdi_boot_seed = nulls.derive_seed(null_seed, "BOOT", "PDI", run_key)
            if do_pdi and pdi_mode != "legacy":
                # Construct-revision modes (surrogate_excess, ...): one call on
                # the run with the estimator's own multivariate surrogate null;
                # the rest baselines are not used.
                pdi_kw = {**pdi_kwargs_raw, **pdi_opts, **_bearer_kw("PDI")}
                pdi_seed = nulls.derive_seed(null_seed, "PDI", run_key)
                pdi_det = _as_details(compute_PDI(
                    ts_region_time,
                    baseline_ts=None,
                    hardware_backend=hardware_backend,
                    return_details=True,
                    null_surrogates=null_k,
                    null_seed=pdi_seed,
                    **pdi_kw,
                ))
                pdi_rec = _record_from_null_fields(
                    "PDI", pdi_det,
                    mpc_evidence.estimator_id("PDI", pdi_mode, pdi_version),
                    pdi_det.get("PDI_null_method"), nodes=pdi_nodes,
                    null_attempted="PDI_null_n" in pdi_det,
                )
                pdi_boot_kw = {
                    **pdi_kw,
                    **MPC_BOOTSTRAP_NULL_OVERRIDES.get(
                        ("PDI", pdi_mode), {"null_surrogates": null_k}
                    ),
                }

                def _pdi_boot(ts_s, _ev, _kw=pdi_boot_kw, _seed=pdi_seed):
                    return compute_PDI(
                        ts_s, baseline_ts=None, hardware_backend=hardware_backend,
                        return_details=True, null_seed=_seed, **_kw,
                    )

                pdi_rec = _attach_bootstrap(
                    pdi_rec, _pdi_boot, ts_region_time, None, boot_k, pdi_boot_seed,
                    boot_block_len, boot_tr, statistic=_raw_statistic,
                )
                mpc_records["PDI"] = [pdi_rec]
                pdi0 = _metric_value([pdi_rec], null_k)
                pdi_anchor_raw = np.nan
                pdi_task_raw = np.nan
                pdi_anchor_reason = f"not_used_by_mode:{pdi_mode}"
                pdi_task_reason = f"not_used_by_mode:{pdi_mode}"
                pdi_primary_source = pdi_mode
                pdi_baseline_policy = "none"
                deep_rest_paths = []
                state_rest_paths = []
            elif do_pdi:
                pdi_bearer = _bearer_kw("PDI")
                deep_rest_cands = _select_pdi_anchor_rest_runs(subj)
                state_rest_cands = _select_pdi_rest_runs(subj, ses)
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
                # (baseline, kwargs) of the primary endpoint, for its bootstrap
                pdi_primary_call = None

                if deep_rest_ts:
                    pdi_anchor_det = _as_details(compute_PDI(
                        ts_region_time,
                        baseline_ts=deep_rest_ts,
                        hardware_backend=hardware_backend,
                        return_details=True,
                        **pdi_kwargs_raw,
                        **pdi_bearer,
                        **(pdi_null_kw if pdi_endpoint == "anchor" else {}),
                    ))
                    pdi_anchor_raw = float(pdi_anchor_det["raw"])
                    pdi_anchor_reason = (
                        "ok" if np.isfinite(pdi_anchor_raw)
                        else "anchor_pdi_estimator_undefined"
                    )
                    if pdi_endpoint == "anchor":
                        pdi_primary_det = pdi_anchor_det
                        pdi_primary_call = (deep_rest_ts, pdi_kwargs_raw)
                if state_rest_ts:
                    pdi_task_det = _as_details(compute_PDI(
                        ts_region_time,
                        baseline_ts=state_rest_ts,
                        hardware_backend=hardware_backend,
                        return_details=True,
                        **pdi_kwargs_raw,
                        **pdi_bearer,
                        **(pdi_null_kw if pdi_endpoint == "task" else {}),
                    ))
                    pdi_task_raw = float(pdi_task_det["raw"])
                    pdi_task_reason = (
                        "ok" if np.isfinite(pdi_task_raw)
                        else "state_pdi_estimator_undefined"
                    )
                    if pdi_endpoint == "task":
                        pdi_primary_det = pdi_task_det
                        pdi_primary_call = (state_rest_ts, pdi_kwargs_raw)

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
                    pdi_primary_call = None

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
                            **pdi_bearer,
                            **pdi_null_kw,
                        ))
                        pdi0 = _legacy_pdi_value(pdi_primary_det, null_k)
                        pdi_primary_source = "legacy_surrogate"
                        pdi_primary_call = (None, pdi_kwargs)
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
                                **pdi_bearer,
                                **pdi_null_kw,
                            ))
                            pdi0 = _legacy_pdi_value(pdi_primary_det, null_k)
                            # Legacy pool provenance is carried by the source
                            # label; the state-matched baseline columns keep
                            # describing PDI_task only.
                            pdi_primary_source = "legacy_rest_pool"
                            pdi_primary_call = (legacy_ts, pdi_kwargs)
                        else:
                            pdi0 = np.nan
                pdi_estimator = mpc_evidence.estimator_id(
                    "PDI", f"legacy-{pdi_primary_source}", pdi_version
                )
                if pdi_primary_det is not None:
                    pdi_rec = _record_from_null_fields(
                        "PDI", pdi_primary_det, pdi_estimator, mpc_null_kinds["PDI"],
                        nodes=pdi_nodes, null_attempted=null_k > 0,
                    )
                    pdi_base, pdi_base_kw = pdi_primary_call

                    def _pdi_boot(ts_s, _ev, _base=pdi_base, _kw=pdi_base_kw):
                        return compute_PDI(
                            ts_s, baseline_ts=_base, hardware_backend=hardware_backend,
                            return_details=True, **_kw, **pdi_bearer,
                        )

                    pdi_rec = _attach_bootstrap(
                        pdi_rec, _pdi_boot, ts_region_time, None, boot_k,
                        pdi_boot_seed, boot_block_len, boot_tr,
                        statistic=_raw_statistic,
                    )
                else:
                    pdi_rec = _component_record(
                        np.nan, reason=pdi_primary_reason, estimator=pdi_estimator,
                        nodes=pdi_nodes,
                    )
                mpc_records["PDI"] = [pdi_rec]
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
            if do_nas and _nas_hub_missing(
                {**nas_kwargs, **mpc_modes["NAS"]}, mpc_modes["NAS"]
            ):
                # NAS capacity without a declared hub: UNDEFINED with a stable
                # reason, never a crash (the estimator requires the hub and is
                # not called; no null, no bootstrap).
                mpc_records["NAS"] = [_component_record(
                    np.nan,
                    reason=mpc_evidence.REASON_NO_DECLARED_WORKSPACE,
                    estimator=mpc_evidence.estimator_id(
                        "NAS", _mode_of("NAS", mpc_modes["NAS"]), nas_version
                    ),
                    nodes=_nodes("NAS", n_nodes_run),
                )]
                nas = np.nan
            elif do_nas:
                nas_opts = dict(mpc_modes["NAS"])
                nas_mode = _mode_of("NAS", nas_opts)
                nas_internal = _uses_internal_null("NAS", nas_opts)
                nas_kw = {**nas_kwargs, **nas_opts, **_bearer_kw("NAS")}
                nas_seed = nulls.derive_seed(null_seed, "NAS", run_key)
                nas_null_kw = {}
                if nas_internal:
                    nas_null_kw = {"null_surrogates": null_k, "null_seed": nas_seed}
                elif null_k > 0:
                    nas_null_kw = {
                        "null_surrogates": null_k,
                        "null_method": mpc_null_kinds["NAS"],
                        "null_seed": nas_seed,
                    }
                nas_det = _as_details(compute_NAS(
                    ts_region_time,
                    tr=tr,
                    hardware_backend=hardware_backend,
                    return_details=True,
                    **nas_kw,
                    **nas_null_kw,
                ))
                nas_rec = _record_from_null_fields(
                    "NAS", nas_det,
                    mpc_evidence.estimator_id("NAS", nas_mode, nas_version),
                    nas_det.get("NAS_null_method") if nas_internal
                    else mpc_null_kinds["NAS"],
                    nodes=_nodes("NAS", n_nodes_run),
                    null_attempted=nas_internal or null_k > 0,
                )
                nas_boot_kw = dict(nas_kw)
                nas_statistic = _raw_statistic
                if nas_internal:
                    # The statistic is the transfer entropy of the direction
                    # that limited the observed value (its null is nas_det's).
                    nas_boot_kw.update(
                        MPC_BOOTSTRAP_NULL_OVERRIDES.get(("NAS", nas_mode), {}),
                        null_seed=nas_seed,
                    )
                    nas_statistic = functools.partial(
                        _transfer_statistic,
                        direction=str(nas_det.get("limiting_direction")),
                    )

                def _nas_boot(ts_s, _ev, _kw=nas_boot_kw):
                    return compute_NAS(
                        ts_s, tr=tr, hardware_backend=hardware_backend,
                        return_details=True, **_kw,
                    )

                nas_rec = _attach_bootstrap(
                    nas_rec, _nas_boot, ts_region_time, None, boot_k,
                    nulls.derive_seed(null_seed, "BOOT", "NAS", run_key),
                    boot_block_len, boot_tr, statistic=nas_statistic,
                )
                mpc_records["NAS"] = [nas_rec]
                nas = (
                    _metric_value([nas_rec], null_k) if nas_internal
                    else float(nas_det["value"])
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
                mpc_records["IIM"] = [_iim_record(
                    iim_info, mpc_null_kinds["IIM"],
                    cut_mode=_mode_of("IIM", mpc_modes["IIM"]),
                    nodes=_nodes("IIM", n_nodes_run),
                    declared=mpc_modes["IIM"], bearer=mpc_bearers["IIM"], tr=tr,
                )]
                if null_k > 0 and iim_defined:
                    # Calibrated canonical IIM (bits); never the raw ratio.
                    iim = _as_float(iim_info.get("canonical_calibrated"))
                    if not (mpc_records["IIM"][0]["null_n"] and np.isfinite(iim)):
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
                srpi_opts, srpi_mode_reason = _run_mode_options(
                    "SRPI", mpc_modes["SRPI"], subj_onsets
                )
                srpi_mode = _mode_of("SRPI", srpi_opts)
                srpi_internal = _uses_internal_null("SRPI", srpi_opts)
                srpi_events = subj_onsets if isinstance(subj_onsets, dict) else {}
                srpi_kw = {**srpi_kwargs, **srpi_opts, **_bearer_kw("SRPI")}

                def _srpi_call(ts_s, ev_s, _kw=srpi_kw, details=False,
                               _agency=srpi_internal):
                    ev_s = ev_s if isinstance(ev_s, dict) else {}
                    extra = (
                        {"agency_events": ev_s.get("agency_events")} if _agency else {}
                    )
                    return compute_SRPI(
                        ts_s,
                        tr=tr,
                        self_onsets=ev_s.get("self_onsets", []),
                        nonself_onsets=ev_s.get("nonself_onsets", []),
                        hardware_backend=hardware_backend,
                        return_details=details,
                        **extra,
                        **_kw,
                    )

                srpi_det = _as_details(_srpi_call(ts_region_time, srpi_events,
                                                  details=True))
                srpi_estimator = mpc_evidence.estimator_id(
                    "SRPI", srpi_mode, srpi_version
                )
                srpi_nodes = _nodes("SRPI", n_nodes_run)
                if srpi_internal:
                    srpi_rec = _record_from_null_fields(
                        "SRPI", srpi_det, srpi_estimator,
                        srpi_det.get("SRPI_null_method"), nodes=srpi_nodes,
                        null_attempted=True,
                    )
                    srpi_boot_kw = {
                        **srpi_kw,
                        **MPC_BOOTSTRAP_NULL_OVERRIDES.get(("SRPI", srpi_mode), {}),
                    }
                    srpi_boot = functools.partial(
                        _srpi_call, _kw=srpi_boot_kw, details=True
                    )
                    srpi_statistic = _raw_statistic
                else:
                    srpi_rec = _component_record(
                        srpi_det.get("value"),
                        reason=srpi_det.get("undefined_reason"),
                        estimator=srpi_estimator,
                        nodes=srpi_nodes,
                    )
                    if null_k > 0 and srpi_rec["defined"]:
                        srpi_rec = _component_null_record(
                            "SRPI", _srpi_call, ts_region_time, dict(srpi_events),
                            mpc_null_kinds["SRPI"], null_k,
                            nulls.derive_seed(null_seed, "SRPI", run_key), srpi_rec, tr,
                        )
                    srpi_boot = _srpi_call
                    srpi_statistic = None
                srpi_rec = _attach_bootstrap(
                    srpi_rec, srpi_boot, ts_region_time, dict(srpi_events), boot_k,
                    nulls.derive_seed(null_seed, "BOOT", "SRPI", run_key),
                    boot_block_len, boot_tr, statistic=srpi_statistic,
                )
                srpi_rec["mode_reason"] = srpi_mode_reason
                mpc_records["SRPI"] = [srpi_rec]
                srpi = _metric_value([srpi_rec], null_k)
            else:
                srpi = np.nan
            # Single-source constraint: components of N measured on different
            # node sets need joint dependence above null (tested with the
            # run's null budget; untested at K = 0).
            gated_sets = {
                p: mpc_records[p][0]["nodes"]
                for p in mpc_nset if mpc_records.get(p)
            }
            n_node_sets = len(set(gated_sets.values()))
            if (
                mpc_proto.source_rule == "single_source" and n_node_sets > 1
                and null_k > 0
            ):
                try:
                    run_joint = mpc_evidence.joint_dependence(
                        ts_region_time,
                        {p: list(s) for p, s in gated_sets.items()},
                        lag=1,
                        n_surrogates=null_k,
                        seed=nulls.derive_seed(null_seed, "JOINT", run_key),
                        alpha=mpc_proto.alpha,
                    )
                except ValueError as exc:
                    run_joint = {
                        "dependent": False,
                        "reason": f"joint_dependence_failed:{exc}",
                        "sets": [list(s) for s in set(gated_sets.values())],
                        "p": float("nan"),
                    }
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

        run_rows_start = len(records)
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
            records.append(rec)
        if compute_mpc:
            mpc_runs.append({
                "rows": list(range(run_rows_start, len(records))),
                "records": mpc_records,
                "subject": subj,
                "session": ses,
                "bearer_id": f"{subj}/{ses}/{os.path.basename(ts_path)}",
                "substrate": modality,
                "grain": atlas,
                "regime": {
                    "modality": modality,
                    "n_time": int(ts_region_time.shape[1]),
                    "n_nodes": n_nodes_run,
                    "tr": tr,
                    "bins": iim_bins,
                },
                "joint": run_joint,
                "n_node_sets": n_node_sets,
            })

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
            df, mpc_runs, mpc_proto, mpc_registry, null_k, null_seed, boot_k,
            boot_block_len, ci_weights,
            channel_evidence=record_channel_evidence,
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
        df, mpc_runs, mpc_proto, mpc_registry, null_k, null_seed, boot_k,
        boot_block_len, ci_weights, channel_evidence=record_channel_evidence,
    )
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    return df[ordered_cols]
