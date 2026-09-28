import numpy as np
import networkx as nx
from scipy.stats import norm


class HypergraphSynergy:
    """
    Legacy exploratory statistic S (reported separately; not part of CI).

    With an edge wherever |corr| > theta, S = I * Bc * Bal where I is the mean
    variance of pairwise-mean signals over edges, Bc the mean size of connected
    components with more than 5 nodes divided by the node count, and
    Bal = 1 - |E - G| / (E + G) for E edges and G large components. Because
    E >> G in practice, Bal ~= 2G/E, so S behaves roughly like 2*I*Bc/E: an
    inverse suprathreshold edge count. S depends strongly on theta; report it
    for all thetas with multiplicity correction or at one pre-declared theta.
    """

    @staticmethod
    def compute(ts, theta):
        # ts is expected as (time, nodes)
        corr = np.corrcoef(ts, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0.0)

        adj_bool = abs_corr > theta
        edge_i, edge_j = np.where(np.triu(adj_bool, k=1))
        n_local = int(edge_i.size)

        adj = adj_bool.astype(int)
        G = nx.from_numpy_array(adj)
        glob_ = [c for c in nx.connected_components(G) if len(c) > 5]
        I_measure = HypergraphSynergy._integration(ts, edge_i, edge_j)
        Bc = HypergraphSynergy._broadcast(glob_, ts.shape[1])
        Bal = HypergraphSynergy._balance(n_local, len(glob_))
        return I_measure * Bc * Bal

    @staticmethod
    def _integration(ts, edge_i, edge_j):
        if edge_i.size == 0:
            return 0.0
        # Vectorized equivalent of mean over edge-wise var(mean(ts[:, i], ts[:, j])).
        pair_means = 0.5 * (ts[:, edge_i] + ts[:, edge_j])
        return float(np.var(pair_means, axis=0).mean())

    @staticmethod
    def _broadcast(glob_, n):
        sizes = [len(e) for e in glob_]
        return float(np.mean(sizes) / n) if sizes else 0.0

    @staticmethod
    def _balance(n_local, n_global):
        Wl, Wg = int(n_local), int(n_global)
        return 1 - abs(Wl - Wg) / (Wl + Wg) if (Wl + Wg) > 0 else 0.0


def compute_midrank(x):
    """1-based midranks (ties get the average rank), as scipy.stats.rankdata."""
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        # NaN != NaN would never close a tie group (the loop below would not end).
        raise ValueError("compute_midrank requires non-NaN input")
    sorted_idx = np.argsort(x, kind="mergesort")
    T = x[sorted_idx]
    n = len(x)
    mid = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and T[j] == T[i]:
            j += 1
        mid[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    ret = np.empty(n, dtype=float)
    ret[sorted_idx] = mid
    return ret


def _fast_delong_components(y_true, scores):
    """
    Fast DeLong (Sun & Xu 2014) for k score vectors on the same cases.

    Returns (aucs: (k,), cov: (k, k)) where cov is the DeLong covariance matrix
    of the AUC estimates, including the between-score covariance terms.
    AUC is P(score_pos > score_neg) + 0.5 P(tie), with positives = (y_true == 1).
    """
    y_raw = np.asarray(y_true).ravel()
    if not np.all(np.isin(y_raw, (0, 1))):
        # Any other label would be silently dropped from both classes.
        raise ValueError("DeLong requires binary labels coded 0/1 (or False/True)")
    y = y_raw.astype(int)
    S = np.atleast_2d(np.asarray(scores, dtype=float))
    if S.shape[1] != y.size:
        raise ValueError("scores and y_true must have the same number of cases")
    if not np.all(np.isfinite(S)):
        raise ValueError("DeLong requires finite scores (drop undefined rows first)")
    pos = S[:, y == 1]
    neg = S[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        raise ValueError("DeLong requires at least two cases per class")
    k = S.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = compute_midrank(pos[r])
        ty[r] = compute_midrank(neg[r])
        tz[r] = compute_midrank(np.concatenate([pos[r], neg[r]]))
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    # Structural components (placement values): v10 per positive, v01 per negative.
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    s10 = np.atleast_2d(np.cov(v10))
    s01 = np.atleast_2d(np.cov(v01))
    cov = s10 / m + s01 / n
    return aucs, cov


def fast_delong(y_true, y_score):
    """AUC and its DeLong variance for one score vector (positives = y_true == 1)."""
    scores = np.asarray(y_score, dtype=float)[None, :]
    aucs, cov = _fast_delong_components(y_true, scores)
    return float(aucs[0]), float(cov[0, 0])


def delong_roc_test(y_true, y1, y2, return_details=False):
    """
    Paired two-sided DeLong test of AUC(y1) == AUC(y2) on the same cases.

    Uses var(AUC1 - AUC2) = var1 + var2 - 2 cov12 (Sun & Xu 2014). DeLong
    treats cases as independent; with repeated measures per subject, prefer a
    subject-level bootstrap as a complement.
    """
    scores = np.vstack([np.asarray(y1, float), np.asarray(y2, float)])
    aucs, cov = _fast_delong_components(y_true, scores)
    delta = float(aucs[0] - aucs[1])
    var_delta = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var_delta > 0 and np.isfinite(var_delta):
        z = delta / np.sqrt(var_delta)
        p = float(2.0 * norm.sf(abs(z)))
    elif delta == 0:
        z, p = 0.0, 1.0
    else:
        z, p = np.nan, np.nan
    if return_details:
        return {
            "p": p,
            "z": float(z),
            "auc1": float(aucs[0]),
            "auc2": float(aucs[1]),
            "delta_auc": delta,
            "var_delta": var_delta,
            "cov": cov.tolist(),
        }
    return p
