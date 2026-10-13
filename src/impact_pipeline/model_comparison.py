import numpy as np
import logging
from sklearn.metrics import roc_auc_score
from impact_pipeline.utils import delong_roc_test
from impact_pipeline.analysis_bootstrap import DEFAULT_STATS_SEED, holm_adjust, rank_auc

log = logging.getLogger(__name__)

DEFAULT_BASELINE_METRICS = ("mean_conn", "modularity", "lzc")


def _bootstrap_delta_auc(y, s1, s2, groups, n_boot, random_state, alpha=0.05):
    """
    Subject-cluster percentile intervals of (signed ΔAUC, Δ|AUC − 0.5|).
    """
    rng = np.random.RandomState(random_state)
    uniq = np.unique(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in uniq]
    deltas = []
    deltas_disc = []
    for _ in range(int(n_boot)):
        pick = rng.randint(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_group[i] for i in pick])
        if np.unique(y[idx]).size < 2:
            continue
        a1 = rank_auc(y[idx], s1[idx])
        a2 = rank_auc(y[idx], s2[idx])
        deltas.append(a1 - a2)
        deltas_disc.append(abs(a1 - 0.5) - abs(a2 - 0.5))
    if not deltas:
        return (np.nan, np.nan), (np.nan, np.nan)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    lo, hi = np.percentile(deltas, q)
    lo_d, hi_d = np.percentile(deltas_disc, q)
    return (float(lo), float(hi)), (float(lo_d), float(hi_d))


def compare_models(
    df,
    metrics=DEFAULT_BASELINE_METRICS,
    sessions=("awake", "deep"),
    score_col="S",
    subject_col="subject",
    n_boot=2000,
    random_state=DEFAULT_STATS_SEED,
):
    """
    Compare ``score_col`` (default S, the exploratory legacy statistic) against
    each baseline metric: ΔAUC = AUC_score – AUC_metric with
    AUC = P(value_{sessions[1]} > value_{sessions[0]}).

    Only rows of the two compared sessions are used, and for each metric only rows
    where both the score and the metric are defined, so both AUCs are estimated on
    identical cases. p_val is the paired DeLong test (Sun & Xu 2014, including the
    covariance between the two AUCs); p_holm is Holm-adjusted across metrics;
    delta_auc_ci is a subject-level bootstrap interval (cases resampled by subject
    when ``subject_col`` is present, which respects repeated measures).

    ΔAUC compares *signed* AUCs: two scores that separate the sessions equally well
    but in opposite directions (e.g. AUC 0.1 vs 0.9) give a large |ΔAUC| and a small
    p_val. Discriminability is therefore also compared directly:
    delta_discrimination = |AUC_score − 0.5| − |AUC_metric − 0.5| with a
    subject-bootstrap interval (delta_discrimination_ci) and p_discrimination, a
    paired DeLong test after orienting each score so that its AUC is >= 0.5
    (orientation taken from the data, so approximate when an AUC is near 0.5),
    Holm-adjusted across metrics (p_discrimination_holm).
    """
    neg, pos = sessions
    session_vals = df.session.astype(str).to_numpy()
    uniq_sessions = set(session_vals.tolist())
    if pos not in uniq_sessions and neg in uniq_sessions and len(uniq_sessions) == 2:
        # Backward-compatible fallback for datasets that use a different
        # positive label (e.g., "sedation" instead of "deep").
        pos = next(s for s in uniq_sessions if s != neg)
        log.debug("compare_models: inferred positive session label '%s'", pos)

    in_pair = np.isin(session_vals, [str(neg), str(pos)])
    y_all = (session_vals == str(pos)).astype(int)
    if np.unique(y_all[in_pair]).size < 2:
        raise ValueError(
            f"compare_models requires two classes after session mapping; got labels={sorted(uniq_sessions)} "
            f"with sessions={sessions}."
        )

    unique, counts = np.unique(y_all[in_pair], return_counts=True)
    log.debug("compare_models y counts: %s", dict(zip(unique, counts)))

    score_all = np.asarray(df[score_col], dtype=float)
    groups_all = (
        df[subject_col].astype(str).to_numpy()
        if subject_col in df.columns
        else np.arange(len(df)).astype(str)
    )
    out = {}
    for m in metrics:
        if m not in df.columns:
            out[m] = {
                "delta_auc": np.nan,
                "p_val": np.nan,
                "p_holm": np.nan,
                "delta_discrimination": np.nan,
                "p_discrimination": np.nan,
                "p_discrimination_holm": np.nan,
                "note": f"metric '{m}' not available",
            }
            continue
        metric_all = np.asarray(df[m], dtype=float)
        keep = in_pair & np.isfinite(score_all) & np.isfinite(metric_all)
        y = y_all[keep]
        s1 = score_all[keep]
        s2 = metric_all[keep]
        res = {
            "score": str(score_col),
            "n_rows": int(keep.sum()),
            "n_rows_excluded": int((in_pair & ~keep).sum()),
            "delta_auc": np.nan,
            "auc_score": np.nan,
            "auc_metric": np.nan,
            "p_val": np.nan,
            "delta_auc_ci": (np.nan, np.nan),
            "delta_discrimination": np.nan,
            "p_discrimination": np.nan,
            "delta_discrimination_ci": (np.nan, np.nan),
        }
        if np.unique(y).size < 2:
            res["note"] = "fewer than two classes after excluding undefined rows"
            out[m] = res
            continue
        res["auc_score"] = float(roc_auc_score(y, s1))
        res["auc_metric"] = float(roc_auc_score(y, s2))
        res["delta_auc"] = res["auc_score"] - res["auc_metric"]
        res["delta_discrimination"] = abs(res["auc_score"] - 0.5) - abs(
            res["auc_metric"] - 0.5
        )
        try:
            res["p_val"] = delong_roc_test(y, s1, s2)
            sign1 = 1.0 if res["auc_score"] >= 0.5 else -1.0
            sign2 = 1.0 if res["auc_metric"] >= 0.5 else -1.0
            res["p_discrimination"] = delong_roc_test(y, sign1 * s1, sign2 * s2)
        except ValueError as exc:
            res["note"] = f"DeLong undefined: {exc}"
        res["delta_auc_ci"], res["delta_discrimination_ci"] = _bootstrap_delta_auc(
            y, s1, s2, groups_all[keep], n_boot=n_boot, random_state=random_state
        )
        out[m] = res
    names = list(out.keys())
    p_adj = holm_adjust([out[k]["p_val"] for k in names])
    p_adj_disc = holm_adjust([out[k]["p_discrimination"] for k in names])
    for k, p, p_d in zip(names, p_adj, p_adj_disc):
        out[k]["p_holm"] = float(p)
        out[k]["p_discrimination_holm"] = float(p_d)
    return out
