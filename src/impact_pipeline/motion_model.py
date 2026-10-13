import os
import glob
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

log = logging.getLogger(__name__)


def _read_fd_file(path: str) -> float:
    with open(path, 'r') as f:
        return float(f.read().strip())


def _n_timepoints(ts_paths) -> int:
    n_tp = 0
    for ts_path in ts_paths:
        try:
            ts = np.load(ts_path, mmap_mode='r')
            # TS shape is (n_time, n_regions); weight by timepoints.
            n_tp += int(ts.shape[0])
        except Exception:
            continue
    return n_tp


def _fd_entries_for_folder(cond_dir: str, subject: str, atlas: str):
    """
    (fd, n_timepoints, source) per run of one condition folder.

    Preprocessing writes one ``{subject}_run-{k}_mean_fd.txt`` per run (paired
    with ``{subject}_run-{k}_{atlas}_ts.npy``); these are used when present.
    Legacy folders only have ``mean_fd.txt`` (one value for the folder),
    weighted by all runs in it.
    """
    entries = []
    run_files = sorted(
        glob.glob(os.path.join(cond_dir, f"{subject}_run-*_mean_fd.txt"))
    )
    for fd_path in run_files:
        prefix = os.path.basename(fd_path)[: -len("_mean_fd.txt")]
        n_tp = _n_timepoints([os.path.join(cond_dir, f"{prefix}_{atlas}_ts.npy")])
        if n_tp <= 0:
            log.warning("No readable time series for %s; FD weight set to 1.", fd_path)
            n_tp = 1
        entries.append((_read_fd_file(fd_path), n_tp, fd_path))
    if entries:
        return entries
    legacy = os.path.join(cond_dir, "mean_fd.txt")
    if not os.path.exists(legacy):
        return entries
    # Match the TS file(s) for this condition folder
    ts_candidates = sorted(glob.glob(os.path.join(
        cond_dir, f"{subject}_run-*_{atlas}_ts.npy"
    )))
    if not ts_candidates:
        # fallback (handles slightly different naming styles)
        ts_candidates = sorted(glob.glob(os.path.join(
            cond_dir, f"{subject}_*_{atlas}_ts.npy"
        )))
    n_tp = _n_timepoints(ts_candidates)
    if n_tp <= 0:
        log.warning("No readable time series next to %s; FD weight set to 1.", legacy)
        n_tp = 1
    return [(_read_fd_file(legacy), n_tp, legacy)]


def _weighted_session_fd(
    data_dir: str, subject: str, session: str, atlas: str, condition=None
) -> float:
    """
    Timepoint-weighted mean FD for one subject/session.

    With ``condition`` only ``{data_dir}/{subject}/{session}/{condition}`` (and
    sub-folders) is used (the condition analysed for CI); otherwise all
    conditions of the session are pooled (legacy). Each run's own FD
    (``{subject}_run-{k}_mean_fd.txt``) is weighted by that run's number of
    timepoints; folders from older preprocessing that only have the folder-level
    ``mean_fd.txt`` are weighted by all their runs. Runs with undefined (NaN) FD
    are excluded, never imputed; the result is NaN when no run has a finite FD.
    Saved *_ts.npy arrays are (n_time, n_regions), so timepoints are
    ``shape[0]``.
    """
    root = os.path.join(data_dir, subject, session)
    if condition is not None:
        root = os.path.join(root, condition)
    fd_paths = sorted(
        glob.glob(os.path.join(root, "**", "*mean_fd.txt"), recursive=True)
    )
    if not fd_paths:
        raise FileNotFoundError(f"No mean_fd.txt under {root}")

    fds, weights = [], []
    for cond_dir in sorted({os.path.dirname(p) for p in fd_paths}):
        for fd, n_tp, _source in _fd_entries_for_folder(cond_dir, subject, atlas):
            if np.isfinite(fd):
                fds.append(fd)
                weights.append(n_tp)
    if not fds:
        return float("nan")
    return float(np.average(fds, weights=weights))


def _ols(y, x):
    X = sm.add_constant(np.asarray(x, dtype=float), has_constant='add')
    res = sm.OLS(np.asarray(y, dtype=float), X).fit()
    return res.params, res.pvalues


def motion_covariate_analysis(df_agg, data_dir: str, atlas: str = "schaefer400",
                              condition=None, score_col: str = "CI", sessions=None):
    """
    df_agg should be one row per (subject, session), e.g., df_mean from run_s_ci.

    FD per (subject, session) is the timepoint-weighted mean over the analysed
    ``condition`` (all conditions when None). Rows with an undefined score (NaN CI)
    or without FD are excluded and counted (never imputed). Fits
      - score ~ FD separately within each session (coef_{ses}, p_{ses}, n_{ses}), and
      - the paired state contrast Δscore = b0 + b1·ΔFD over subjects with both
        sessions (Δ = sessions[0] − sessions[1]); b0 (delta_intercept) is the
        FD-adjusted state difference.
    Returns a one-row DataFrame.
    """
    dfc = df_agg.copy()
    dfc['subject'] = dfc['subject'].astype(str)
    dfc['session'] = dfc['session'].astype(str)

    keys = dfc[['subject', 'session']].drop_duplicates()
    fd_map = {}
    n_missing_fd = 0
    for _, row in keys.iterrows():
        sub, ses = row['subject'], row['session']
        try:
            fd = _weighted_session_fd(data_dir, sub, ses, atlas, condition=condition)
        except FileNotFoundError as exc:
            log.warning("motion: %s; %s/%s excluded from the FD models.", exc, sub, ses)
            fd = np.nan
        if not np.isfinite(fd):
            # No file, or only undefined (NaN) FD: excluded and counted.
            n_missing_fd += 1
        fd_map[(sub, ses)] = fd

    dfc['FD'] = [fd_map[(r.subject, r.session)] for r in dfc.itertuples()]
    if score_col in dfc.columns:
        score = pd.to_numeric(dfc[score_col], errors='coerce')
    else:
        score = pd.Series(np.nan, index=dfc.index)
    dfc['__score'] = score
    n_undefined = int((~np.isfinite(score.to_numpy(dtype=float))).sum())
    usable = dfc[np.isfinite(dfc['__score']) & np.isfinite(dfc['FD'])]

    out = {
        'score': score_col,
        'fd_condition': condition if condition is not None else "all_conditions",
        'n_rows': int(len(dfc)),
        'n_rows_undefined_score': n_undefined,
        'n_subject_sessions_missing_fd': int(n_missing_fd),
    }
    # Regress within each session separately (awake vs deep)
    for ses, grp in usable.groupby('session'):
        coef, pval = np.nan, np.nan
        if len(grp) >= 3 and np.ptp(grp['FD'].to_numpy(dtype=float)) > 0:
            params, pvalues = _ols(grp['__score'], grp['FD'])
            coef, pval = float(params[1]), float(pvalues[1])
        out[f'coef_{ses}'] = coef
        out[f'p_{ses}'] = pval
        out[f'n_{ses}'] = int(len(grp))
    for ses in dfc['session'].unique():
        out.setdefault(f'coef_{ses}', np.nan)
        out.setdefault(f'p_{ses}', np.nan)
        out.setdefault(f'n_{ses}', 0)

    if sessions is not None:
        ses_pair = tuple(sessions)
    else:
        ses_pair = tuple(sorted(dfc['session'].unique()))[:2]
    out['delta_sessions'] = "-".join(str(s) for s in ses_pair)
    out['n_pairs'] = 0
    delta_keys = (
        'delta_intercept', 'p_delta_intercept', 'delta_fd_coef', 'p_delta_fd_coef'
    )
    for k in delta_keys:
        out[k] = np.nan
    if len(ses_pair) == 2:
        s0, s1 = str(ses_pair[0]), str(ses_pair[1])
        piv_kw = dict(index='subject', columns='session', aggfunc='mean')
        piv_s = usable.pivot_table(values='__score', **piv_kw)
        piv_f = usable.pivot_table(values='FD', **piv_kw)
        if {s0, s1}.issubset(piv_s.columns):
            both = piv_s[[s0, s1]].dropna().index
            both = both.intersection(piv_f[[s0, s1]].dropna().index)
            d_score = (piv_s.loc[both, s0] - piv_s.loc[both, s1]).to_numpy(dtype=float)
            d_fd = (piv_f.loc[both, s0] - piv_f.loc[both, s1]).to_numpy(dtype=float)
            out['n_pairs'] = int(len(both))
            if len(both) >= 3 and np.ptp(d_fd) > 0:
                params, pvalues = _ols(d_score, d_fd)
                out['delta_intercept'] = float(params[0])
                out['p_delta_intercept'] = float(pvalues[0])
                out['delta_fd_coef'] = float(params[1])
                out['p_delta_fd_coef'] = float(pvalues[1])

    return pd.DataFrame([out])
