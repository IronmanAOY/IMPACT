"""
MPC-Bench export and in-memory estimator runner.

Export layout (``root`` = one bench export):

    <root>/prep/<subj>/<session>/<condition>/<subj>_run-<k>_<atlas>_ts.npy
    <root>/prep/<subj>/<session>/<condition>/<subj>_run-<k>_<macro_atlas>_ts.npy
    <root>/prep/<subj>/<session>/rest/<subj>_run-<k>_<atlas>_ts.npy        (optional)
    <root>/prep/<subj>/<session>/<condition>/<subj>_run-<k>_<atlas>_meta.json
    <root>/bids/dataset_description.json
    <root>/bids/sub-<subj>/func/sub-<subj>_task-<condition><session>_run-<k>_events.tsv
    <root>/bids/sub-<subj>/func/sub-<subj>_task-<condition><session>_run-<k>_events.json
    <root>/bids/sub-<subj>/func/sub-<subj>_task-<condition><session>_run-<k>_bold.json
    <root>/oracle/<subj>/<session>/<subj>_run-<k>_oracle.npz / _oracle.json
    <root>/bench_manifest.json

The ``prep`` tree is the ``synergy_ci`` glob layout (arrays are time x nodes,
subject folders carry no ``sub-`` prefix), so ``compute_synergy_ci`` and
``run_synergy_ci.run_s_ci`` run unchanged with ``data_dir=<root>/prep`` and
``bids_root=<root>/bids`` (events are resolved by
``event_parsing.resolve_events_file``: task label ``<condition><session>``;
the sample interval comes from ``RepetitionTime`` in the ``_bold.json``
sidecar, which here is the simulation step ``dt``). The ``<macro_atlas>``
array holds the declared IIM grain (mean of each macro node), so IIM can be
computed on that grain through the same glob. Hidden oracle channels live in
``<root>/oracle``, outside both trees read by the estimators.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from impact_pipeline.bench.generators import (
    EVENT_COLUMNS,
    GENERATOR_VERSION,
    PRINCIPLES,
    BenchSystem,
)

EXPORT_LAYOUT_VERSION = "mpc-bench-export/1.0.0"
DEFAULT_CONDITION = "bench"
DEFAULT_ATLAS = "bench"
DEFAULT_MACRO_ATLAS = "benchmacro"
DEFAULT_DATASET_ID = "mpcbench"

# Declared estimator settings for family A/C (20 Hz sampling, 0.1 s unit time
# constant). Fixed before any confirmatory run; recorded with every result.
BENCH_ESTIMATOR_PARAMS = {
    "RAM": {
        "response_model": "boxcar",
        "response_boxcar_width_sec": 0.3,
        "latency_method": "fir",
        "fir_window": 0.8,
        "goal_pre_window_sec": 0.3,
        "response_window_sec": 0.4,
        "goal_objective_window_sec": 0.4,
        "feedback_window_sec": 0.3,
        "quality_lag_sec": 0.0,
        "quality_null_samples": 200,
        "quality_random_state": 0,
    },
    "PDI": {},
    "NAS": {
        "tau": 0.2,
        "bands": [[0.5, 4.0]],
        "band_weights": [1.0],
        "window_len": 100,
        "step_len": 50,
        "random_state": 0,
    },
    "IIM": {"bins": 2, "lag_trs": 2},
    "SRPI": {
        "modality": "eeg",
        "pre_window_sec": 0.5,
        "response_lag_sec": 0.0,
        "response_window_sec": 0.6,
        "min_events_per_class": 3,
    },
}

# Optional construct revisions (added by other streams). They are used when the
# installed estimator accepts them and recorded in ``estimator_modes``.
OPTIONAL_MODES = {
    "RAM": {"update": "prediction_error"},
    "PDI": {"mode": "surrogate_excess"},
    "NAS": {"mode": "capacity"},
    "SRPI": {"mode": "agency"},
}
EVENT_NULL_SURROGATE = "circular_shift"

_EVENT_COLUMN_DESCRIPTIONS = {
    "onset": "Event onset (s) from the start of the run.",
    "duration": "Event duration (s).",
    "trial_type": "goal_cue | stimulus | response | feedback | action | "
    "self_caused | other_caused.",
    "value": "Goal id (goal_cue), stimulus id (stimulus, self_caused, "
    "other_caused, action), choice (response), reward (feedback).",
    "choice": "Chosen arm (0/1) on response and feedback rows.",
    "reward": "Feedback value (+1/-1) on feedback rows.",
    "yoked_to": "event_id of the self_caused event that an other_caused replay "
    "reproduces (stimulus-identical, phase-matched).",
    "phase_bin": "Bin of the endogenous slow-rhythm phase at onset.",
    "impact_channel": "Evidence channel: behavioural_feedback (bandit) or "
    "agency (reafference phase).",
    "trial": "Bandit trial index.",
    "event_id": "Unique event identifier.",
}


def bids_label(text) -> str:
    """Alphanumeric BIDS label (other characters removed)."""
    label = re.sub(r"[^A-Za-z0-9]", "", str(text))
    if not label:
        raise ValueError(f"Cannot build a BIDS label from {text!r}")
    return label


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def macro_node_timeseries(
    system: BenchSystem, macro_nodes: Optional[dict] = None
) -> np.ndarray:
    """Mean activity of each declared IIM macro node (macro nodes x time)."""
    groups = macro_nodes if macro_nodes is not None else system.meta["iim_macro_nodes"]
    return np.stack(
        [system.ts[np.asarray(idx, dtype=int)].mean(axis=0) for idx in groups.values()]
    )


def write_dataset_description(
    bids_root: Path, dataset_id: str = DEFAULT_DATASET_ID
) -> Path:
    path = Path(bids_root) / "dataset_description.json"
    if not path.exists():
        _write_json(
            path,
            {
                "Name": f"MPC-Bench simulated systems ({dataset_id})",
                "BIDSVersion": "1.10.0",
                "DatasetType": "raw",
                "SyntheticData": True,
                "GeneratedBy": [
                    {
                        "Name": "impact_pipeline.bench",
                        "Version": GENERATOR_VERSION,
                        "Description": "White-box simulated systems with switchable "
                        "mechanisms; not human data.",
                    }
                ],
            },
        )
    return path


def export_system(
    system: BenchSystem,
    root,
    subject: str,
    session: str,
    condition: str = DEFAULT_CONDITION,
    run: int = 1,
    atlas: str = DEFAULT_ATLAS,
    macro_atlas: Optional[str] = DEFAULT_MACRO_ATLAS,
    write_rest: bool = True,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> Dict[str, str]:
    """Write one system in the export layout (module docstring); returns paths."""
    root = Path(root)
    subj = bids_label(subject)
    ses = bids_label(session)
    cond = bids_label(condition)
    run = int(run)
    prep_dir = root / "prep" / subj / ses / cond
    prep_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    ts_path = prep_dir / f"{subj}_run-{run}_{atlas}_ts.npy"
    np.save(ts_path, np.ascontiguousarray(system.ts.T))
    paths["ts"] = str(ts_path)
    if macro_atlas and system.meta.get("iim_macro_nodes"):
        macro_path = prep_dir / f"{subj}_run-{run}_{macro_atlas}_ts.npy"
        np.save(macro_path, np.ascontiguousarray(macro_node_timeseries(system).T))
        paths["macro_ts"] = str(macro_path)
    if write_rest and system.rest_ts is not None:
        rest_dir = root / "prep" / subj / ses / "rest"
        rest_dir.mkdir(parents=True, exist_ok=True)
        rest_path = rest_dir / f"{subj}_run-{run}_{atlas}_ts.npy"
        np.save(rest_path, np.ascontiguousarray(system.rest_ts.T))
        paths["rest_ts"] = str(rest_path)
    meta_path = prep_dir / f"{subj}_run-{run}_{atlas}_meta.json"
    _write_json(meta_path, system.meta)
    paths["meta"] = str(meta_path)

    bids_root = root / "bids"
    write_dataset_description(bids_root, dataset_id)
    func = bids_root / f"sub-{subj}" / "func"
    func.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{subj}_task-{cond}{ses}_run-{run}"
    ev = system.events.copy()
    for col in EVENT_COLUMNS:
        if col not in ev.columns:
            ev[col] = np.nan
    ev_path = func / f"{stem}_events.tsv"
    ev.loc[:, list(EVENT_COLUMNS)].to_csv(ev_path, sep="\t", index=False, na_rep="n/a")
    paths["events"] = str(ev_path)
    ev_json = func / f"{stem}_events.json"
    _write_json(
        ev_json, {k: {"Description": v} for k, v in _EVENT_COLUMN_DESCRIPTIONS.items()}
    )
    paths["events_sidecar"] = str(ev_json)
    sidecar = func / f"{stem}_bold.json"
    declared = {
        k: system.meta.get(k)
        for k in (
            "family",
            "seed",
            "knobs",
            "workspace_nodes",
            "bearer_nodes",
            "iim_macro_nodes",
            "iim_grain",
            "iim_lag_samples",
            "iim_bins",
            "module_order",
            "modules",
        )
    }
    _write_json(
        sidecar,
        {
            "TaskName": f"{cond}{ses}",
            "RepetitionTime": float(system.dt),
            "SamplingFrequency": float(1.0 / system.dt),
            "SyntheticData": True,
            "SyntheticNodeContainer": "Node i of the analysed array is unit i of the "
            "simulated system; no anatomical image.",
            "GeneratorVersion": system.meta.get("generator_version"),
            "MPCBenchDeclared": declared,
        },
    )
    paths["sidecar"] = str(sidecar)

    oracle_dir = root / "oracle" / subj / ses
    oracle_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        k: np.asarray(v) for k, v in system.oracle.items() if isinstance(v, np.ndarray)
    }
    scalars = {k: v for k, v in system.oracle.items() if not isinstance(v, np.ndarray)}
    npz = oracle_dir / f"{subj}_run-{run}_oracle.npz"
    np.savez_compressed(npz, **arrays)
    oj = oracle_dir / f"{subj}_run-{run}_oracle.json"
    _write_json(oj, scalars)
    paths["oracle_npz"] = str(npz)
    paths["oracle_json"] = str(oj)
    return paths


def export_bench(
    items: Iterable[Tuple[str, str, BenchSystem]],
    root,
    condition: str = DEFAULT_CONDITION,
    atlas: str = DEFAULT_ATLAS,
    macro_atlas: Optional[str] = DEFAULT_MACRO_ATLAS,
    dataset_id: str = DEFAULT_DATASET_ID,
    provenance: Optional[dict] = None,
) -> dict:
    """Export ``(subject, session, system)`` items and write ``bench_manifest.json``."""
    root = Path(root)
    runs = []
    for subject, session, system in items:
        paths = export_system(
            system,
            root,
            subject,
            session,
            condition=condition,
            atlas=atlas,
            macro_atlas=macro_atlas,
            dataset_id=dataset_id,
        )
        runs.append(
            {
                "subject": bids_label(subject),
                "session": bids_label(session),
                "family": system.meta.get("family"),
                "seed": system.meta.get("seed"),
                "knobs": system.meta.get("knobs"),
                "paths": {k: str(Path(v).relative_to(root)) for k, v in paths.items()},
            }
        )
    manifest = {
        "layout_version": EXPORT_LAYOUT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "condition": bids_label(condition),
        "atlas": atlas,
        "macro_atlas": macro_atlas,
        "dataset_id": dataset_id,
        "data_dir": "prep",
        "bids_root": "bids",
        "oracle_root": "oracle",
        "runs": runs,
        "provenance": provenance or {},
    }
    _write_json(root / "bench_manifest.json", manifest)
    return manifest


def load_export_onsets(
    root,
    subjects: Sequence[str],
    sessions: Sequence[str],
    condition: str = DEFAULT_CONDITION,
) -> dict:
    """``stimulus_onsets`` mapping for ``compute_synergy_ci`` built with the
    pipeline's own events resolver and parser."""
    from impact_pipeline.event_parsing import (
        events_table_to_bundle,
        extract_run_id_from_name,
        read_events_table,
        resolve_events_file,
    )

    out = {}
    bids_root = Path(root) / "bids"
    for subj in subjects:
        s = bids_label(subj)
        out[s] = {}
        for ses in sessions:
            fn = resolve_events_file(
                bids_root, s, bids_label(ses), condition=bids_label(condition)
            )
            if fn is None:
                continue
            out[s][bids_label(ses)] = (
                events_table_to_bundle(read_events_table(fn)),
                extract_run_id_from_name(fn.name),
            )
    return out


def synergy_ci_params(meta: dict, params: Optional[dict] = None) -> dict:
    """Fully explicit ``ram_params``/``pdi_params``/``nas_params``/``srpi_params``
    for ``compute_synergy_ci`` on an export (declared workspace nodes included)."""
    from impact_pipeline.synergy_ci import (
        NAS_PARAM_DEFAULTS,
        PDI_PARAM_DEFAULTS,
        RAM_PARAM_DEFAULTS,
        SRPI_PARAM_DEFAULTS,
    )

    p = _merged_params(params)
    nas = dict(NAS_PARAM_DEFAULTS)
    nas.update({k: v for k, v in p["NAS"].items() if k in NAS_PARAM_DEFAULTS})
    nas["bands"] = [tuple(b) for b in nas["bands"]]
    nas["workspace_nodes"] = list(meta.get("workspace_nodes") or []) or None
    srpi = dict(SRPI_PARAM_DEFAULTS)
    srpi.update({k: v for k, v in p["SRPI"].items() if k in SRPI_PARAM_DEFAULTS})
    ram = dict(RAM_PARAM_DEFAULTS)
    ram.update({k: v for k, v in p["RAM"].items() if k in RAM_PARAM_DEFAULTS})
    pdi = dict(PDI_PARAM_DEFAULTS)
    pdi.update({k: v for k, v in p["PDI"].items() if k in PDI_PARAM_DEFAULTS})
    return {
        "ram_params": ram,
        "pdi_params": pdi,
        "nas_params": nas,
        "srpi_params": srpi,
    }


def run_export_with_synergy_ci(
    root,
    sessions: Sequence[str],
    subjects=None,
    metrics=None,
    atlas: str = DEFAULT_ATLAS,
    condition: str = DEFAULT_CONDITION,
    params: Optional[dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """Run ``compute_synergy_ci`` unchanged on an export (``data_origin='dummy'``)."""
    from impact_pipeline.synergy_ci import compute_synergy_ci, discover_ci_subjects

    root = Path(root)
    manifest = json.loads((root / "bench_manifest.json").read_text(encoding="utf-8"))
    data_dir = root / manifest.get("data_dir", "prep")
    subj = discover_ci_subjects(str(data_dir), subjects=subjects)
    onsets = load_export_onsets(root, subj, sessions, condition=condition)
    first = manifest["runs"][0]
    meta = json.loads((root / first["paths"]["meta"]).read_text(encoding="utf-8"))
    dt = float(meta["dt"])
    call = dict(
        thetas=[0.5],
        sessions=tuple(bids_label(s) for s in sessions),
        condition=bids_label(condition),
        tr=dt,
        stimulus_onsets=onsets,
        mpc_metrics=metrics,
        compute_ci=metrics is None,
        subjects=subj,
        data_origin="dummy",
        dataset_id=manifest.get("dataset_id", DEFAULT_DATASET_ID),
        modality="eeg",
        iim_enable_parallel=False,
        iim_use_shared_memory=False,
        iim_bins=int(meta.get("iim_bins", 2)),
        iim_lag_trs=int(meta.get("iim_lag_samples", 2)),
    )
    call.update(synergy_ci_params(meta, params))
    if atlas != DEFAULT_ATLAS:
        # Declared workspace indices refer to units, not to macro nodes.
        call["nas_params"]["workspace_nodes"] = None
    call.update(kwargs)
    return compute_synergy_ci(str(data_dir), atlas, **call)


# ---------------------------------------------------------------------------
# In-memory runner
# ---------------------------------------------------------------------------


def _merged_params(params: Optional[dict]) -> dict:
    out = {k: dict(v) for k, v in BENCH_ESTIMATOR_PARAMS.items()}
    for k, v in (params or {}).items():
        if k not in out:
            raise ValueError(f"Unknown estimator block {k!r}; expected {sorted(out)}")
        out[k].update(v or {})
    return out


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _optional_kwargs(principle: str, fn, events: pd.DataFrame) -> Tuple[dict, dict]:
    """Keyword arguments of optional construct revisions the estimator supports."""
    kw, used = {}, {}
    for name, value in OPTIONAL_MODES.get(principle, {}).items():
        if _accepts(fn, name):
            kw[name] = value
            used[name] = value
    if _accepts(fn, "events"):
        kw["events"] = events
        used["events"] = "dataframe"
    # Bearer restriction is applied by slicing the node set (bearer_view), so
    # an estimator-level ``bearer_nodes`` keyword is never passed as well.
    return kw, used


def _details_value(d):
    if isinstance(d, dict):
        return d.get("value", np.nan)
    return d


def _circular_shift_null(fn, ts: np.ndarray, n: int, seed: int) -> list:
    """Event-locked null: independent circular shift of the whole node x time
    array relative to the (fixed) event times, >= 10% of the run."""
    rng = np.random.default_rng(int(seed))
    t = ts.shape[1]
    lo = max(1, int(np.ceil(0.1 * t)))
    vals = []
    for _ in range(int(n)):
        shift = int(rng.integers(lo, t - lo + 1))
        v = fn(np.roll(ts, shift, axis=1))
        vals.append(float(_details_value(v)) if v is not None else np.nan)
    return vals


def _null_moments(vals) -> Tuple[float, float, int]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, 0
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan
    return float(np.mean(arr)), sd, int(arr.size)


def _event_null(principle, fn, ts, events, n, seed):
    """RAM/SRPI null: ``impact_pipeline.nulls.component_null`` when available,
    otherwise the circular-shift fallback above."""
    try:
        from impact_pipeline import nulls  # optional (stream I1)

        res = nulls.component_null(
            fn, ts, events, surrogate=EVENT_NULL_SURROGATE, n=int(n), seed=int(seed)
        )
        vals = res.get("values") if isinstance(res, dict) else res
        return list(np.asarray(vals, dtype=float).reshape(-1)), "nulls.component_null"
    except Exception:
        return (
            _circular_shift_null(lambda x: fn(x, events), ts, n, seed),
            "circular_shift_fallback",
        )


BEARER_MODES = ("system", "principle")


def bearer_view(
    system: BenchSystem, principle: str, bearer_mode: str = "system"
) -> dict:
    """
    Node set on which ``principle`` is measured. ``'system'``: all declared
    bearer nodes (IIM on ``meta['iim_macro_nodes']``). ``'principle'``: the
    node set declared in ``meta['principle_bearers'][principle]`` (patchwork),
    with workspace and IIM macro nodes re-indexed into that subset; systems
    without per-principle bearers fall back to the system view.
    """
    if bearer_mode not in BEARER_MODES:
        raise ValueError(f"bearer_mode must be one of {BEARER_MODES}")
    meta = system.meta
    nodes = [int(i) for i in (meta.get("bearer_nodes") or range(system.n_nodes))]
    ws = [int(i) for i in (meta.get("workspace_nodes") or [])]
    macro = meta.get("iim_macro_nodes") or {}
    bearer_id = "system"
    per = meta.get("principle_bearers") or {}
    if bearer_mode == "principle" and principle in per:
        nodes = [int(i) for i in per[principle]]
        ws = [
            int(i)
            for i in (meta.get("principle_workspace_nodes") or {}).get(principle, [])
        ]
        macro = (meta.get("principle_macro_nodes") or {}).get(
            principle, {k: [i] for k, i in ((str(j), j) for j in nodes)}
        )
        bearer_id = f"principle:{principle}"
    pos = {g: k for k, g in enumerate(nodes)}
    return {
        "nodes": nodes,
        "ts": np.asarray(system.ts, dtype=float)[nodes],
        "workspace_nodes": [pos[i] for i in ws if i in pos] or None,
        "macro_nodes": {k: [pos[i] for i in v if i in pos] for k, v in macro.items()},
        "bearer_id": bearer_id,
    }


def run_in_memory(
    system: BenchSystem,
    metrics: Sequence[str] = PRINCIPLES,
    params: Optional[dict] = None,
    null_surrogates: int = 0,
    null_seed: int = 0,
    use_optional_modes: bool = True,
    bearer_mode: str = "system",
) -> dict:
    """
    Run the public estimators directly on one system (no files).

    Events are parsed by ``event_parsing.events_table_to_bundle`` (the
    pipeline's parser); NAS uses the declared workspace nodes; IIM runs on the
    declared macro-node grain; ``bearer_mode`` selects the node set per
    principle (:func:`bearer_view`). With ``null_surrogates > 0`` each
    component gets a null family: the estimators' own surrogate calibration
    for PDI (phase-randomised), NAS (circular shift) and IIM (circular shift of
    macro nodes; statistic Delta_Psi in bits), and an event-locked null for
    RAM/SRPI (circular shift of the time series against the fixed events).
    Returns per-component ``estimate`` (raw statistic), ``value`` (returned
    value), ``null_mean``, ``null_sd``, ``n_null``, ``defined``, ``reason``,
    ``bearer_id`` and ``seconds``, plus the optional ``estimator_modes`` used.
    """
    from impact_pipeline import mpc_metrics as mm
    from impact_pipeline.event_parsing import events_table_to_bundle

    unknown = sorted(set(metrics) - set(PRINCIPLES))
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}")
    p = _merged_params(params)
    meta = system.meta
    dt = float(system.dt)
    events = system.events
    bundle = events_table_to_bundle(events)
    k_null = int(null_surrogates)
    out = {
        "components": {},
        "estimator_modes": {},
        "params": p,
        "bearer_mode": bearer_mode,
    }

    def _record(
        principle,
        t0,
        details,
        estimate,
        view,
        null_mean=np.nan,
        null_sd=np.nan,
        n_null=0,
        null_family=None,
        statistic="value",
    ):
        value = float(_details_value(details)) if details is not None else np.nan
        reason = details.get("undefined_reason") if isinstance(details, dict) else None
        defined = bool(np.isfinite(estimate))
        out["components"][principle] = {
            "estimate": float(estimate) if np.isfinite(estimate) else np.nan,
            "value": value,
            "statistic": statistic,
            "null_mean": float(null_mean),
            "null_sd": float(null_sd),
            "n_null": int(n_null),
            "null_family": null_family,
            "defined": defined,
            "reason": None if defined else (reason or "undefined"),
            "bearer_id": view["bearer_id"],
            "n_bearer_nodes": len(view["nodes"]),
            "seconds": round(time.perf_counter() - t0, 4),
        }

    def _modes(principle, fn):
        if not use_optional_modes:
            return {}, {}
        return _optional_kwargs(principle, fn, events)

    if "RAM" in metrics:
        t0 = time.perf_counter()
        view = bearer_view(system, "RAM", bearer_mode)
        kw, used = _modes("RAM", mm.compute_RAM)
        out["estimator_modes"]["RAM"] = used

        def ram_fn(x, ev=None):
            return mm.compute_RAM(
                x, tr=dt, stimulus_onsets=bundle, return_details=True, **p["RAM"], **kw
            )

        d = ram_fn(view["ts"])
        est = float(_details_value(d))
        nm = nsd = np.nan
        nn = 0
        fam = None
        if k_null > 0 and np.isfinite(est):
            vals, fam = _event_null(
                "RAM", ram_fn, view["ts"], events, k_null, null_seed
            )
            nm, nsd, nn = _null_moments(vals)
        _record("RAM", t0, d, est, view, nm, nsd, nn, fam)

    if "PDI" in metrics:
        t0 = time.perf_counter()
        view = bearer_view(system, "PDI", bearer_mode)
        kw, used = _modes("PDI", mm.compute_PDI)
        kw.pop("events", None)
        out["estimator_modes"]["PDI"] = used
        baseline = None
        if system.rest_ts is not None:
            baseline = np.asarray(system.rest_ts, dtype=float)[view["nodes"]]
        d = mm.compute_PDI(
            view["ts"],
            baseline_ts=baseline,
            return_details=True,
            null_surrogates=k_null,
            null_seed=int(null_seed),
            **p["PDI"],
            **kw,
        )
        est = float(d.get("raw", np.nan)) if isinstance(d, dict) else float(d)
        _record(
            "PDI",
            t0,
            d,
            est,
            view,
            d.get("PDI_null_mean", np.nan),
            d.get("PDI_null_sd", np.nan),
            d.get("PDI_null_n", 0),
            d.get("PDI_null_method") if k_null else None,
            statistic="raw",
        )

    if "NAS" in metrics:
        t0 = time.perf_counter()
        view = bearer_view(system, "NAS", bearer_mode)
        kw, used = _modes("NAS", mm.compute_NAS)
        kw.pop("events", None)
        out["estimator_modes"]["NAS"] = used
        nas_kw = dict(p["NAS"])
        nas_kw["bands"] = [tuple(b) for b in nas_kw["bands"]]
        d = mm.compute_NAS(
            view["ts"],
            tr=dt,
            workspace_nodes=view["workspace_nodes"],
            return_details=True,
            null_surrogates=k_null,
            null_seed=int(null_seed),
            **nas_kw,
            **kw,
        )
        est = float(d.get("raw", np.nan))
        _record(
            "NAS",
            t0,
            d,
            est,
            view,
            d.get("NAS_null_mean", np.nan),
            d.get("NAS_null_sd", np.nan),
            d.get("NAS_null_n", 0),
            d.get("NAS_null_method") if k_null else None,
            statistic="raw",
        )

    if "IIM" in metrics:
        t0 = time.perf_counter()
        view = bearer_view(system, "IIM", bearer_mode)
        macro = np.stack(
            [
                view["ts"][np.asarray(v, dtype=int)].mean(axis=0)
                for v in view["macro_nodes"].values()
            ]
        )
        iim_kw = dict(p["IIM"])
        iim_kw.setdefault("bins", int(meta.get("iim_bins", 2)))
        iim_kw.setdefault("lag_trs", int(meta.get("iim_lag_samples", 2)))
        d = mm.compute_IIM(
            macro,
            return_details=True,
            null_surrogates=k_null,
            null_seed=int(null_seed),
            progress_log_every_cuts=10**9,
            **iim_kw,
        )
        out["estimator_modes"]["IIM"] = {
            "grain": meta.get("iim_grain"),
            "n_macro_nodes": int(macro.shape[0]),
        }
        if k_null > 0:
            est = float(d.get("Delta_Psi", np.nan))
            _record(
                "IIM",
                t0,
                d,
                est,
                view,
                d.get("Delta_Psi_null_mean", np.nan),
                d.get("Delta_Psi_null_sd", np.nan),
                d.get("IIM_null_n", 0),
                "circular_shift",
                statistic="Delta_Psi_bits",
            )
        else:
            est = (
                float(d.get("raw", np.nan)) if bool(d.get("defined", True)) else np.nan
            )
            _record("IIM", t0, d, est, view, statistic="raw")

    if "SRPI" in metrics:
        t0 = time.perf_counter()
        view = bearer_view(system, "SRPI", bearer_mode)
        kw, used = _modes("SRPI", mm.compute_SRPI)
        out["estimator_modes"]["SRPI"] = used

        def srpi_fn(x, ev=None):
            return mm.compute_SRPI(
                x,
                tr=dt,
                self_onsets=bundle["self_onsets"],
                nonself_onsets=bundle["nonself_onsets"],
                return_details=True,
                **p["SRPI"],
                **kw,
            )

        d = srpi_fn(view["ts"])
        est = float(_details_value(d))
        nm = nsd = np.nan
        nn = 0
        fam = None
        if k_null > 0 and np.isfinite(est):
            vals, fam = _event_null(
                "SRPI", srpi_fn, view["ts"], events, k_null, null_seed + 1
            )
            nm, nsd, nn = _null_moments(vals)
        _record("SRPI", t0, d, est, view, nm, nsd, nn, fam)

    return out


def evidence_verdict(
    result: dict, meta: dict, protocol_id: Optional[str] = None
) -> dict:
    """
    MPC verdict from an in-memory result through ``impact_pipeline.evidence``
    when that module is available (stream I1); otherwise
    ``{'verdict': None, 'reason': 'evidence_layer_unavailable'}``.
    """
    try:
        from impact_pipeline import evidence as ev
    except Exception:
        return {"verdict": None, "reasons": ["evidence_layer_unavailable"]}
    try:
        items = {}
        for principle, c in result["components"].items():
            items[principle] = [
                ev.ComponentEvidence(
                    principle=principle,
                    estimate=float(c["estimate"]),
                    null_mean=float(c["null_mean"]),
                    null_sd=float(c["null_sd"]),
                    se=0.0,
                    channel="default",
                    defined=bool(c["defined"]),
                    reason=c.get("reason"),
                    bearer_id=str(c.get("bearer_id", "system")),
                    protocol_id=protocol_id,
                    estimator=str(c.get("statistic")),
                    null_family=c.get("null_family"),
                    n_null=int(c.get("n_null", 0)),
                )
            ]
        v = ev.mpc_verdict(items)
        verdict = getattr(v, "verdict", None)
        return {
            "verdict": getattr(verdict, "value", verdict),
            "reasons": list(getattr(v, "reasons", []) or []),
            "component_status": {
                k: getattr(s, "value", s)
                for k, s in (getattr(v, "component_status", {}) or {}).items()
            },
            "margins": dict(getattr(v, "margins", {}) or {}),
        }
    except Exception as exc:  # pragma: no cover - depends on the other stream's API
        return {
            "verdict": None,
            "reasons": [f"evidence_layer_error:{type(exc).__name__}:{exc}"],
        }
