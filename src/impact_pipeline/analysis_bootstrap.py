import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from impact_pipeline.utils import compute_midrank

# Default seed for all resampling statistics (reproducible p-values / intervals).
DEFAULT_STATS_SEED = 0


def _two_session_rows(df, score_col, session_col, sessions, subject_col, paired):
    """
    Restrict to the two compared sessions, drop undefined (non-finite) scores and,
    in paired mode, keep only subjects with a defined score in both sessions.

    Returns (rows, y, info). y = 1 for sessions[1], 0 for sessions[0].
    """
    s0, s1 = str(sessions[0]), str(sessions[1])
    sub = df[df[session_col].astype(str).isin([s0, s1])].copy()
    scores = pd.to_numeric(sub[score_col], errors="coerce")
    finite = np.isfinite(scores.to_numpy(dtype=float))
    info = {
        "n_rows_in_sessions": int(len(sub)),
        "n_rows_excluded_undefined": int((~finite).sum()),
        "n_subjects_excluded_incomplete": 0,
    }
    sub = sub.loc[finite].assign(__score=scores[finite].to_numpy(dtype=float))
    use_pairs = (
        bool(paired) and (subject_col is not None) and (subject_col in sub.columns)
    )
    if use_pairs:
        sub[subject_col] = sub[subject_col].astype(str)
        has = sub.groupby(subject_col)[session_col].agg(lambda s: set(s.astype(str)))
        complete = has[has.apply(lambda ss: {s0, s1}.issubset(ss))].index
        all_subj = (
            set(df[subject_col].astype(str)) if subject_col in df.columns else set()
        )
        info["n_subjects_excluded_incomplete"] = int(len(all_subj - set(complete)))
        sub = sub[sub[subject_col].isin(complete)]
    info["mode"] = "paired_subject" if use_pairs else "row"
    info["n_rows_used"] = int(len(sub))
    info["n_subjects_used"] = int(sub[subject_col].nunique()) if use_pairs else np.nan
    y = (sub[session_col].astype(str) == s1).astype(int).to_numpy()
    return sub, y, info


def rank_auc(y, scores):
    """AUC = P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney), as roc_auc_score."""
    y = np.asarray(y).astype(bool)
    return float(_auc_from_ranks(stats.rankdata(scores), y[None, :])[0])


def _auc_from_ranks(ranks, labels):
    """AUC = P(score_pos > score_neg) + 0.5 P(tie) for each row of a label matrix."""
    labels = np.atleast_2d(labels).astype(bool)
    m = labels.sum(axis=1).astype(float)
    n = labels.shape[1] - m
    r_pos = (labels * ranks[None, :]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (r_pos - m * (m + 1.0) / 2.0) / (m * n)


def bootstrap_ci(
    df,
    score_col,
    session_col="session",
    sessions=("awake", "deep"),
    n_boot=1000,
    random_state=DEFAULT_STATS_SEED,
    subject_col="subject",
    paired=True,
    alpha=0.05,
    return_details=False,
):
    """
    Percentile bootstrap interval of the AUC separating sessions[1] (positive) from
    sessions[0]. AUC = P(score_{sessions[1]} > score_{sessions[0]}).

    Paired design (default, needs ``subject_col``): subjects are resampled with
    replacement, keeping all of a subject's rows together; only subjects with a
    defined score in both sessions are used. Without a subject column, rows are
    resampled. Undefined (NaN) scores are excluded and counted.
    """
    rows, y, info = _two_session_rows(
        df, score_col, session_col, sessions, subject_col, paired
    )
    nan_out = {"auc": np.nan, "lo": np.nan, "hi": np.nan, "n_boot_valid": 0, **info}
    if np.unique(y).size < 2:
        warnings.warn(
            f"bootstrap_ci({score_col}): session labels contain <2 classes for sessions={sessions}; returning NaN CI.",
            RuntimeWarning,
        )
        return nan_out if return_details else (np.nan, np.nan)
    scores = rows["__score"].to_numpy(dtype=float)
    rng = np.random.RandomState(random_state)
    aucs = []
    attempts = 0
    max_attempts = max(10 * n_boot, 1000)
    if info["mode"] == "paired_subject":
        subj = rows[subject_col].to_numpy()
        uniq = np.unique(subj)
        idx_by_subj = [np.flatnonzero(subj == s) for s in uniq]
    while len(aucs) < n_boot and attempts < max_attempts:
        attempts += 1
        if info["mode"] == "paired_subject":
            pick = rng.randint(0, len(uniq), size=len(uniq))
            idx = np.concatenate([idx_by_subj[i] for i in pick])
        else:
            idx = rng.choice(len(rows), size=len(rows), replace=True)
        if np.unique(y[idx]).size < 2:
            continue
        aucs.append(rank_auc(y[idx], scores[idx]))
    if len(aucs) == 0:
        warnings.warn(
            f"bootstrap_ci({score_col}): no valid bootstrap resamples produced; returning NaN CI.",
            RuntimeWarning,
        )
        return nan_out if return_details else (np.nan, np.nan)
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if return_details:
        return {
            "auc": float(roc_auc_score(y, scores)),
            "lo": float(lo),
            "hi": float(hi),
            "n_boot_valid": int(len(aucs)),
            "random_state": random_state,
            **info,
        }
    return lo, hi


def permutation_test_auc(
    df,
    score_col,
    session_col="session",
    sessions=("awake", "deep"),
    n_perm=10000,
    random_state=DEFAULT_STATS_SEED,
    subject_col="subject",
    paired=True,
    alternative="two-sided",
    return_details=False,
):
    """
    Permutation test of the session AUC,
    AUC = P(score_{sessions[1]} > score_{sessions[0]}).

    alternative: ``"two-sided"`` (default; |AUC - 0.5| as statistic), ``"greater"``
    (sessions[1] higher) or ``"less"`` (sessions[1] lower, e.g. awake > deep for
    sessions=('awake','deep')).
    Paired design (default, needs ``subject_col``): session labels are swapped
    within subject (exact enumeration when 2**n_subjects <= n_perm). Without a
    subject column, labels are permuted across rows. Undefined scores are excluded.
    Returns (auc, p).
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    rows, y, info = _two_session_rows(
        df, score_col, session_col, sessions, subject_col, paired
    )
    if np.unique(y).size < 2:
        warnings.warn(
            f"permutation_test_auc({score_col}): session labels contain <2 classes for sessions={sessions}; returning NaNs.",
            RuntimeWarning,
        )
        if return_details:
            return {
                "auc": np.nan,
                "p": np.nan,
                "n_perm": 0,
                "alternative": alternative,
                **info,
            }
        return np.nan, np.nan
    scores = rows["__score"].to_numpy(dtype=float)
    ranks = compute_midrank(scores)
    obs_auc = float(_auc_from_ranks(ranks, y[None, :])[0])
    rng = np.random.RandomState(random_state)
    exact = False
    if info["mode"] == "paired_subject":
        subj = rows[subject_col].to_numpy()
        uniq, inv = np.unique(subj, return_inverse=True)
        n_subj = len(uniq)
        if 2**n_subj <= int(n_perm):
            flips = np.array(list(itertools.product((0, 1), repeat=n_subj)), dtype=bool)
            exact = True
        else:
            flips = rng.rand(int(n_perm), n_subj) < 0.5
        labels = np.where(flips[:, inv], 1 - y[None, :], y[None, :])
    else:
        labels = np.vstack([rng.permutation(y) for _ in range(int(n_perm))])
    perm_auc = _auc_from_ranks(ranks, labels)
    tol = 1e-12
    if alternative == "two-sided":
        hits = np.abs(perm_auc - 0.5) >= abs(obs_auc - 0.5) - tol
    elif alternative == "greater":
        hits = perm_auc >= obs_auc - tol
    else:
        hits = perm_auc <= obs_auc + tol
    if exact:
        p_val = float(np.mean(hits))
    else:
        p_val = float((hits.sum() + 1) / (len(perm_auc) + 1))
    if return_details:
        return {
            "auc": obs_auc,
            "p": p_val,
            "n_perm": int(len(perm_auc)),
            "exact": bool(exact),
            "alternative": alternative,
            "random_state": random_state,
            **info,
        }
    return obs_auc, p_val


def holm_adjust(pvals):
    """Holm step-down adjusted p-values; NaN entries are ignored and stay NaN."""
    p = np.asarray(pvals, dtype=float).ravel()
    out = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    m = int(mask.sum())
    if m == 0:
        return out
    pv = p[mask]
    order = np.argsort(pv, kind="mergesort")
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * pv[order])
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    out[mask] = adj
    return out


def _subject_session_pivot(
    df, metric, sessions, subject_col="subject", session_col="session"
):
    s0, s1 = str(sessions[0]), str(sessions[1])
    sub = df[df[session_col].astype(str).isin([s0, s1])].copy()
    sub["__val"] = pd.to_numeric(sub[metric], errors="coerce")
    sub.loc[~np.isfinite(sub["__val"].to_numpy(dtype=float)), "__val"] = np.nan
    sub[subject_col] = sub[subject_col].astype(str)
    sub[session_col] = sub[session_col].astype(str)
    # Mean over each subject/session's defined rows (undefined rows excluded).
    piv = sub.groupby([subject_col, session_col])["__val"].mean().unstack(session_col)
    for s in (s0, s1):
        if s not in piv.columns:
            piv[s] = np.nan
    return piv[[s0, s1]]


def paired_session_test(
    df, metric, sessions=("awake", "deep"), subject_col="subject", session_col="session"
):
    """
    Subject-level paired comparison sessions[0] - sessions[1] for one metric.

    Rows with undefined (NaN) values are excluded; subjects without a defined value
    in both sessions are excluded and counted. Descriptives are computed on the
    complete pairs used for the test. Effect size is Cohen's dz.
    """
    out = {
        "metric": str(metric),
        "session_a": str(sessions[0]),
        "session_b": str(sessions[1]),
        "n_pairs": 0,
        "n_subjects_total": 0,
        "n_subjects_excluded": 0,
        "mean_a": np.nan,
        "sd_a": np.nan,
        "sem_a": np.nan,
        "mean_b": np.nan,
        "sd_b": np.nan,
        "sem_b": np.nan,
        "mean_diff": np.nan,
        "sd_diff": np.nan,
        "t": np.nan,
        "df": np.nan,
        "p": np.nan,
        "dz": np.nan,
    }
    if metric not in df.columns or len(sessions) < 2:
        return out
    piv = _subject_session_pivot(df, metric, sessions, subject_col, session_col)
    paired = piv.dropna()
    n = int(len(paired))
    out["n_subjects_total"] = int(len(piv))
    out["n_subjects_excluded"] = int(len(piv) - n)
    out["n_pairs"] = n
    if n == 0:
        return out
    a = paired.iloc[:, 0].to_numpy(dtype=float)
    b = paired.iloc[:, 1].to_numpy(dtype=float)
    diff = a - b
    out["mean_a"] = float(a.mean())
    out["mean_b"] = float(b.mean())
    out["mean_diff"] = float(diff.mean())
    if n < 2:
        return out
    out["sd_a"] = float(a.std(ddof=1))
    out["sd_b"] = float(b.std(ddof=1))
    out["sem_a"] = out["sd_a"] / np.sqrt(n)
    out["sem_b"] = out["sd_b"] / np.sqrt(n)
    sd = float(diff.std(ddof=1))
    out["sd_diff"] = sd
    out["df"] = n - 1
    if np.isfinite(sd) and sd > 0:
        t_val, p_val = stats.ttest_rel(a, b)
        out["t"] = float(t_val)
        out["p"] = float(p_val)
        out["dz"] = float(diff.mean() / sd)
    return out


def paired_tests_table(
    df,
    metrics,
    sessions=("awake", "deep"),
    family="components",
    subject_col="subject",
    session_col="session",
):
    """Paired tests for several metrics with Holm correction within the family."""
    rows = [
        paired_session_test(
            df, m, sessions, subject_col=subject_col, session_col=session_col
        )
        for m in metrics
        if m in df.columns
    ]
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    tab["family"] = str(family)
    tab["p_holm"] = holm_adjust(tab["p"].to_numpy(dtype=float))
    return tab


def paired_bootstrap_mean_diff(
    df,
    metric,
    sessions=("awake", "deep"),
    n_boot=2000,
    random_state=DEFAULT_STATS_SEED,
    alpha=0.05,
    subject_col="subject",
    session_col="session",
):
    """Percentile bootstrap CI of the mean paired difference sessions[0]-sessions[1].

    Subjects (complete pairs) are resampled with replacement.
    """
    piv = _subject_session_pivot(
        df, metric, sessions, subject_col, session_col
    ).dropna()
    diff = (piv.iloc[:, 0] - piv.iloc[:, 1]).to_numpy(dtype=float)
    n = diff.size
    if n < 2:
        return {
            "mean_diff": float(diff.mean()) if n else np.nan,
            "lo": np.nan,
            "hi": np.nan,
            "n_pairs": int(n),
        }
    rng = np.random.RandomState(random_state)
    boots = diff[rng.randint(0, n, size=(int(n_boot), n))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_diff": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_pairs": int(n),
        "n_boot": int(n_boot),
        "random_state": random_state,
    }


def definedness_summary(
    df,
    value_col="CI",
    defined_col="CI_defined",
    missing_col="CI_missing",
    sessions=("awake", "deep"),
    subject_col="subject",
    session_col="session",
):
    """
    Count defined/undefined rows of a three-valued metric (e.g. CI) and how many
    subjects remain with a defined value in both compared sessions.
    """
    out = {"metric": value_col, "n_rows": int(len(df))}
    if value_col not in df.columns:
        out["available"] = False
        return out
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    if defined_col in df.columns:
        defined = df[defined_col].astype(str).str.lower().isin(
            {"true", "1"}
        ).to_numpy() & np.isfinite(vals)
    else:
        defined = np.isfinite(vals)
    out["available"] = True
    out["n_rows_defined"] = int(defined.sum())
    out["n_rows_undefined"] = int((~defined).sum())
    if missing_col in df.columns:
        counts = {}
        for txt in df.loc[~defined, missing_col].fillna("").astype(str):
            for comp in [c for c in txt.split(",") if c]:
                counts[comp] = counts.get(comp, 0) + 1
        out["missing_component_row_counts"] = counts
    if subject_col in df.columns and session_col in df.columns and len(sessions) >= 2:
        tmp = df[[subject_col, session_col]].astype(str).assign(__def=defined)
        by_ss = tmp.groupby([subject_col, session_col])["__def"].any()
        out["n_subject_sessions"] = int(len(by_ss))
        out["n_subject_sessions_defined"] = int(by_ss.sum())
        s0, s1 = str(sessions[0]), str(sessions[1])
        ok = by_ss.unstack(session_col)
        if {s0, s1}.issubset(ok.columns):
            both = ok[[s0, s1]].eq(True).all(axis=1)
            out["n_subjects_complete_pairs"] = int(both.sum())
            out["n_subjects_excluded"] = int((~both).sum())
    return out
