import logging
import numpy as np
import pandas as pd
from scipy import stats
from impact_pipeline.synergy_ci import compute_synergy_ci
from impact_pipeline.analysis_bootstrap import holm_adjust, definedness_summary

log = logging.getLogger(__name__)


def _theta_roughness(df: pd.DataFrame, session) -> tuple[float, int]:
    """
    Within-subject smoothness of S over theta for one session.

    S is first averaged over runs per (subject, theta); for each subject the RMS
    of successive differences along the sorted theta grid is computed, and the
    subject values are averaged. Returns (mean roughness, n_subjects).
    """
    sub = df[df['session'].astype(str) == str(session)]
    if sub.empty or 'theta' not in sub.columns:
        return np.nan, 0
    curves = sub.groupby(['subject', 'theta'])['S'].mean().reset_index()
    vals = []
    for _, grp in curves.groupby('subject'):
        s = grp.sort_values('theta')['S'].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if len(s) > 1:
            vals.append(float(np.sqrt(np.mean(np.diff(s) ** 2))))
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def _paired_summary(agg: pd.DataFrame, metric: str, sessions: tuple[str, ...]) -> dict:
    out = {
        "n": 0,
        "mean_a": np.nan,
        "mean_b": np.nan,
        "delta_a_minus_b": np.nan,
        "t": np.nan,
        "p": np.nan,
        "d_paired": np.nan,
    }
    if metric not in agg.columns or len(sessions) < 2:
        return out
    s0, s1 = str(sessions[0]), str(sessions[1])
    piv = agg.pivot(index="subject", columns="session", values=metric)
    if (s0 not in piv.columns) or (s1 not in piv.columns):
        return out
    paired = piv[[s0, s1]].dropna()
    n = int(len(paired))
    out["n"] = n
    if n == 0:
        return out
    a = paired[s0].to_numpy(dtype=float)
    b = paired[s1].to_numpy(dtype=float)
    diff = a - b
    out["mean_a"] = float(np.nanmean(a))
    out["mean_b"] = float(np.nanmean(b))
    out["delta_a_minus_b"] = float(np.nanmean(diff))
    if n < 2:
        return out
    try:
        t_val, p_val = stats.ttest_rel(a, b, nan_policy="omit")
        out["t"] = float(t_val)
        out["p"] = float(p_val)
    except Exception:
        pass
    sd = float(np.nanstd(diff, ddof=1))
    if np.isfinite(sd) and sd > 0:
        out["d_paired"] = float(np.nanmean(diff) / sd)
    return out


def atlas_check(
    data_dir,
    atlases=('aal116', 'shen268'),
    sessions=('awake', 'deep'),
    thetas=None,
    tr=None,
    stimulus_onsets=None,
    mpc_metrics=None,
    compute_ci=False,
    pdi_params=None,
    pdi_require_explicit_params=False,
    pdi_require_strict_baseline=False,
    pdi_primary_endpoint="anchor",
    nas_params=None,
    srpi_params=None,
    srpi_require_explicit_params=True,
    subjects=None,
    condition='audio',
    iim_kwargs=None,
    ci_reference=None,
    compute_kwargs=None,
    **synergy_kwargs,
):
    """
    Re-run the metric computation on alternative atlases and summarise paired
    session contrasts per atlas (Holm-adjusted across the metrics of each atlas).

    ``condition`` is forwarded (same condition folder as the primary analysis).
    ``iim_kwargs`` (e.g. iim_max_nodes, iim_bins) and ``compute_kwargs`` (e.g.
    provenance, hardware_target) are passed through to compute_synergy_ci so the
    robustness run uses the primary configuration. An atlas whose time series are
    missing is recorded as skipped with the reason instead of aborting.
    """
    out = {}
    thresholds = thetas if thetas is not None else [i * 0.1 for i in range(1, 10)]
    valid_mpc = ("RAM", "PDI", "NAS", "IIM", "SRPI")
    if mpc_metrics is None:
        # Default robustness set for event-sparse datasets.
        mpc_eff = ("PDI", "NAS", "IIM")
    else:
        mpc_eff = tuple(dict.fromkeys([m for m in mpc_metrics if m in valid_mpc]))

    notes = []
    # RAM/SRPI require event timing to be meaningful.
    if stimulus_onsets is None:
        dropped = [m for m in mpc_eff if m in {"RAM", "SRPI"}]
        mpc_eff = tuple(m for m in mpc_eff if m not in {"RAM", "SRPI"})
        if dropped or compute_ci:
            notes.append(
                "RAM/SRPI omitted in atlas robustness because no event timings "
                "were provided."
            )
    ci_enabled = bool(compute_ci and set(valid_mpc).issubset(set(mpc_eff)))
    if compute_ci and not ci_enabled:
        missing = sorted(set(valid_mpc) - set(mpc_eff))
        notes.append(
            "CI not assessed across atlases: components not computed here: "
            f"{','.join(missing)}."
        )
    # ``synergy_kwargs`` (e.g. condition, hardware_target, step-2 iim_* settings)
    # are accepted from the orchestration layer; explicit arguments win.
    synergy_kwargs = dict(synergy_kwargs)
    condition = synergy_kwargs.pop("condition", condition)
    ci_reference = synergy_kwargs.pop("ci_reference", ci_reference)
    extra = dict(synergy_kwargs)
    extra.update(dict(compute_kwargs or {}))
    extra.update(dict(iim_kwargs or {}))

    for atlas in atlases:
        try:
            df = compute_synergy_ci(
                data_dir,
                atlas=atlas,
                thetas=thresholds,
                sessions=sessions,
                condition=condition,
                tr=tr,
                stimulus_onsets=stimulus_onsets,
                compute_mpc=bool(len(mpc_eff) > 0),
                mpc_metrics=(None if len(mpc_eff) == 0 else mpc_eff),
                compute_ci=ci_enabled,
                ci_reference=ci_reference,
                pdi_params=pdi_params,
                pdi_require_explicit_params=pdi_require_explicit_params,
                pdi_require_strict_baseline=pdi_require_strict_baseline,
                pdi_primary_endpoint=pdi_primary_endpoint,
                nas_params=nas_params,
                srpi_params=srpi_params,
                srpi_require_explicit_params=srpi_require_explicit_params,
                subjects=subjects,
                **extra,
            )
        except FileNotFoundError as exc:
            log.warning("Atlas robustness: skipping atlas %s (%s)", atlas, exc)
            out[atlas] = {
                "skipped": f"missing time series for atlas '{atlas}': {exc}",
                "notes": list(notes),
            }
            continue
        if not isinstance(df, pd.DataFrame):
            raise TypeError("compute_synergy_ci must return DataFrame")
        rough = {}
        rough_n = {}
        for ses in sessions:
            rough[ses], rough_n[ses] = (
                _theta_roughness(df, ses) if not df.empty else (np.nan, 0)
            )
        agg_map = {"S": ("S", "mean")}
        for metric in ("CI", "RAM", "PDI", "NAS", "IIM", "SRPI", "IIM_raw", "IIM_raw_scaled"):
            if metric in df.columns:
                agg_map[metric] = (metric, "mean")
        agg = df.groupby(["subject", "session"]).agg(**agg_map).reset_index()

        metric_stats = {
            "S": _paired_summary(agg, "S", sessions),
        }
        for metric in ("PDI", "NAS", "IIM", "IIM_raw", "RAM", "SRPI", "CI"):
            if metric in agg.columns:
                metric_stats[metric] = _paired_summary(agg, metric, sessions)
        names = list(metric_stats.keys())
        p_holm = holm_adjust([metric_stats[k]["p"] for k in names])
        for name, p_adj in zip(names, p_holm):
            metric_stats[name]["p_holm"] = float(p_adj)

        payload = {
            # Legacy top-level fields retained for backward compatibility.
            str(sessions[0]) if len(sessions) > 0 else "session0": rough.get(sessions[0], np.nan) if len(sessions) > 0 else np.nan,
            str(sessions[1]) if len(sessions) > 1 else "session1": rough.get(sessions[1], np.nan) if len(sessions) > 1 else np.nan,
            "roughness": rough,
            "roughness_n_subjects": rough_n,
            "roughness_definition": (
                "mean over subjects of RMS successive S differences across sorted theta"
            ),
            "metrics": metric_stats,
            "mpc_metrics_used": list(mpc_eff),
            "ci_enabled": ci_enabled,
            "n_subjects": int(agg["subject"].nunique()) if not agg.empty else 0,
            "n_rows": int(len(df)),
            "notes": list(notes),
        }
        if ci_enabled and "CI" in df.columns:
            payload["ci_definedness"] = definedness_summary(df, sessions=sessions)
        out[atlas] = payload
    return out
