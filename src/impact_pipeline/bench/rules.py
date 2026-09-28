"""
Rival attribution rules on component matrices (rule audit of MPC-Bench).

Every rule takes an ``(n_systems, 5)`` matrix of component values in
``PRINCIPLES`` order (RAM, PDI, NAS, IIM, SRPI); NaN marks an undefined
component. Unless stated otherwise, inputs are reference-normalised
components (0 = null anchor, 1 = reference anchor) and presence means
``value >= threshold`` (default 0.5). Rules return a :class:`RuleOutput` with a
rule-specific ``score`` and a ``decision`` in {MPC_CONSISTENT, EXCLUDED,
UNDETERMINED} (the v2 verdict names of the necessity-only stance: a rival
rule's positive decision is read as "not excluded", its negative decision as
"excluded"). Each rule's handling of missing components is fixed and listed
in ``RULE_PROPERTIES``:

``skip``       missing components are ignored (count / mean of the rest)
``zero``       missing components count as 0 (legacy geometric-mean CI)
``propagate``  any missing component makes the score NaN (UNDETERMINED)
``prior``      a missing component contributes its prior credence
``impute``     missing components are mean-imputed (learned classifier)
``kleene``     three-valued: missing is UNDEFINED under strong Kleene AND

The IMPaCT rule has two forms. :func:`rule_impact` (legacy v1 status rule,
a comparator only) works on null-standardised margins
``z = (estimate - null_mean) / null_sd``: PRESENT if
``z - z_{1-alpha} se > z_present``, ABSENT if
``|z| + z_{1-alpha} se <= delta_equiv`` (TOST), else UNDEFINED.
:func:`rule_impact_c` (v2, spec V2-2) works on the two-anchor construct scale
``c = (m - nu) / (rho - nu)`` with a genuine sampling SE of ``c``: PRESENT if
the one-sided ``1 - alpha`` lower bound of ``c`` exceeds ``z_j`` (default
0.25), ABSENT if the upper bound is below ``delta_j`` (default 0.10), else
UNDEFINED; a missing or zero SE, or invalid anchors, is UNDEFINED. An SE
from few replicates (delete-a-group jackknife with ``G`` groups) declares its
degrees of freedom (``G - 1``); the bounds then use the Student ``t`` quantile
with the Welch-Satterthwaite degrees of freedom of ``se_c``, exactly as
``impact_pipeline.evidence.component_assessment`` with ``se_df``. Both
verdicts are the strong-Kleene AND over the necessity set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import stdtrit
from scipy.stats import norm

MPC_CONSISTENT = "MPC_CONSISTENT"
EXCLUDED = "EXCLUDED"
UNDETERMINED = "UNDETERMINED"
VERDICTS = (MPC_CONSISTENT, EXCLUDED, UNDETERMINED)
# Construct-scale smallest effects of interest (spec v2, V2-2; initial
# defaults to be justified by MPC-Bench dose-response).
C_PRESENT_DEFAULT = 0.25
C_ABSENT_DEFAULT = 0.10
PRESENT, ABSENT, UNDEFINED = "PRESENT", "ABSENT", "UNDEFINED"
N_COMPONENTS = 5
PRINCIPLE_INDEX = {"RAM": 0, "PDI": 1, "NAS": 2, "IIM": 3, "SRPI": 4}

RULE_PROPERTIES = {
    "union": dict(
        compensatory=True,
        missingness="skip",
        veto=False,
        abstains=False,
        prior_dependent=False,
    ),
    "count_k": dict(
        compensatory=True,
        missingness="skip",
        veto=False,
        abstains=False,
        prior_dependent=False,
    ),
    "arithmetic_mean": dict(
        compensatory=True,
        missingness="skip",
        veto=False,
        abstains=True,
        prior_dependent=False,
    ),
    "geometric_mean_uncapped": dict(
        compensatory=True,
        missingness="zero",
        veto=True,
        abstains=False,
        prior_dependent=False,
    ),
    "geometric_mean_capped": dict(
        compensatory=True,
        missingness="propagate",
        veto=True,
        abstains=True,
        prior_dependent=False,
    ),
    "power_mean": dict(
        compensatory=True,
        missingness="propagate",
        veto=True,
        abstains=True,
        prior_dependent=False,
    ),
    "weakest_link": dict(
        compensatory=False,
        missingness="propagate",
        veto=True,
        abstains=True,
        prior_dependent=False,
    ),
    "dcm_naive_bayes": dict(
        compensatory=True,
        missingness="skip",
        veto=False,
        abstains=False,
        prior_dependent=True,
    ),
    "chalmers_product": dict(
        compensatory=True,
        missingness="prior",
        veto=False,
        abstains=False,
        prior_dependent=True,
    ),
    "logistic": dict(
        compensatory=True,
        missingness="impute",
        veto=False,
        abstains=False,
        prior_dependent=True,
    ),
    "single_marker": dict(
        compensatory=False,
        missingness="propagate",
        veto=False,
        abstains=True,
        prior_dependent=False,
    ),
    "impact": dict(
        compensatory=False,
        missingness="kleene",
        veto=True,
        abstains=True,
        prior_dependent=False,
    ),
    "impact_c": dict(
        compensatory=False,
        missingness="kleene",
        veto=True,
        abstains=True,
        prior_dependent=False,
    ),
}


@dataclass(frozen=True)
class RuleOutput:
    rule: str
    score: np.ndarray
    decision: np.ndarray

    def counts(self) -> Dict[str, int]:
        return {v: int(np.sum(self.decision == v)) for v in VERDICTS}

    def rates(self) -> Dict[str, float]:
        n = max(1, int(self.decision.size))
        return {v: c / n for v, c in self.counts().items()}


def as_component_matrix(C) -> np.ndarray:
    """Validate and return an ``(n, 5)`` float matrix (a single vector is one row)."""
    x = np.asarray(C, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != N_COMPONENTS:
        raise ValueError(f"component matrix must be (n, {N_COMPONENTS}), got {x.shape}")
    return x


def _thresholds(threshold, width=N_COMPONENTS) -> np.ndarray:
    t = np.asarray(threshold, dtype=float).reshape(-1)
    if t.size == 1:
        t = np.repeat(t, width)
    if t.size != width:
        raise ValueError(f"threshold must be a scalar or {width} values")
    return t


def presence(C, threshold=0.5) -> np.ndarray:
    """1.0 present, 0.0 absent, NaN undefined (per component)."""
    x = as_component_matrix(C)
    t = _thresholds(threshold)
    out = (x >= t[None, :]).astype(float)
    out[~np.isfinite(x)] = np.nan
    return out


def _decide(score: np.ndarray, threshold: float, rule: str) -> RuleOutput:
    score = np.asarray(score, dtype=float)
    dec = np.where(
        np.isfinite(score),
        np.where(score >= threshold, MPC_CONSISTENT, EXCLUDED),
        UNDETERMINED,
    ).astype(object)
    return RuleOutput(rule, score, dec)


def rule_union(C, threshold=0.5) -> RuleOutput:
    """MPC_CONSISTENT if any observed component is present (missing skipped)."""
    n_present = np.nansum(presence(C, threshold), axis=1)
    return _decide(n_present, 1.0, "union")


def rule_count_k(C, k: int, threshold=0.5) -> RuleOutput:
    """MPC_CONSISTENT if at least ``k`` observed components are present
    (Alkire-Foster style dual cut-off; missing skipped)."""
    k = int(k)
    if not 1 <= k <= N_COMPONENTS:
        raise ValueError("k must be in 1..5")
    n_present = np.nansum(presence(C, threshold), axis=1)
    return _decide(n_present, float(k), f"count_{k}")


def rule_arithmetic_mean(C, threshold=0.5) -> RuleOutput:
    """Additive rule: mean of the observed components (NaN if none observed)."""
    x = as_component_matrix(C)
    with np.errstate(invalid="ignore"):
        observed = np.isfinite(x).sum(axis=1)
        score = np.where(
            observed > 0, np.nansum(x, axis=1) / np.maximum(observed, 1), np.nan
        )
    return _decide(score, threshold, "arithmetic_mean")


def rule_geometric_mean_uncapped(C, threshold=0.5) -> RuleOutput:
    """Legacy geometric-mean CI: uncapped, missing and negative components
    count as 0 (so one zero gives 0; large components compensate small ones)."""
    x = as_component_matrix(C)
    x = np.where(np.isfinite(x), np.maximum(x, 0.0), 0.0)
    with np.errstate(divide="ignore"):
        score = np.where(
            np.all(x > 0, axis=1),
            np.exp(np.mean(np.log(np.where(x > 0, x, 1.0)), axis=1)),
            0.0,
        )
    return _decide(score, threshold, "geometric_mean_uncapped")


def power_mean(
    C, p: float = 0.0, cap: Optional[float] = 1.0, weights=None
) -> np.ndarray:
    """
    Weighted power mean of ``min(max(c, 0), cap)`` per row; NaN if any
    component is NaN. ``p = 0`` geometric, ``p = -inf`` minimum (weakest
    link), ``p = +inf`` maximum; for ``p <= 0`` any zero component gives 0.
    """
    x = as_component_matrix(C)
    w = (
        np.full(N_COMPONENTS, 1.0 / N_COMPONENTS)
        if weights is None
        else np.asarray(weights, float)
    )
    if w.shape != (N_COMPONENTS,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be 5 non-negative values with a positive sum")
    w = w / w.sum()
    y = np.maximum(x, 0.0)
    if cap is not None:
        y = np.minimum(y, float(cap))
    nan_rows = ~np.all(np.isfinite(x[:, w > 0]), axis=1)
    y = np.where(np.isfinite(y), y, 0.0)
    active = w > 0
    if np.isneginf(p):
        score = np.min(y[:, active], axis=1)
    elif np.isposinf(p):
        score = np.max(y[:, active], axis=1)
    elif p == 0:
        zero = np.any(y[:, active] <= 0, axis=1)
        with np.errstate(divide="ignore"):
            logs = np.log(np.where(y > 0, y, 1.0))
        score = np.where(zero, 0.0, np.exp(logs @ w))
    elif p < 0:
        zero = np.any(y[:, active] <= 0, axis=1)
        with np.errstate(divide="ignore"):
            powed = np.where(y > 0, y, 1.0) ** p
        score = np.where(zero, 0.0, (powed @ w) ** (1.0 / p))
    else:
        score = (y**p @ w) ** (1.0 / p)
    score = np.asarray(score, dtype=float)
    score[nan_rows] = np.nan
    return score


def rule_geometric_mean_capped(C, threshold=0.5, cap=1.0) -> RuleOutput:
    """Capped geometric mean (NaN propagates: UNDETERMINED when any is missing)."""
    return _decide(power_mean(C, 0.0, cap), threshold, "geometric_mean_capped")


def rule_power_mean(C, p: float, threshold=0.5, cap=1.0, weights=None) -> RuleOutput:
    return _decide(power_mean(C, p, cap, weights), threshold, f"power_mean_p{p:g}")


def rule_weakest_link(C, threshold=0.5) -> RuleOutput:
    """Minimum component (non-compensatory; NaN propagates)."""
    return _decide(power_mean(C, -np.inf, cap=None), threshold, "weakest_link")


def calibrate_likelihood_ratios(P, labels, alpha: float = 1.0):
    """
    Per-component sensitivity P(present | positive) and specificity
    P(absent | negative) from a calibration presence matrix (NaN skipped),
    with Laplace smoothing ``alpha``. Returns ``(sensitivity, specificity)``.
    """
    P = as_component_matrix(P)
    y = np.asarray(labels).astype(bool).reshape(-1)
    if y.size != P.shape[0]:
        raise ValueError("labels must have one entry per row")
    sens = np.zeros(N_COMPONENTS)
    spec = np.zeros(N_COMPONENTS)
    for j in range(N_COMPONENTS):
        col = P[:, j]
        pos = col[y & np.isfinite(col)]
        neg = col[~y & np.isfinite(col)]
        sens[j] = (np.sum(pos == 1) + alpha) / (pos.size + 2 * alpha)
        spec[j] = (np.sum(neg == 0) + alpha) / (neg.size + 2 * alpha)
    return sens, spec


def rule_dcm_naive_bayes(
    C,
    threshold=0.5,
    prior: float = 1.0 / 6.0,
    sensitivity=0.8,
    specificity=0.8,
    decision_threshold: float = 0.5,
) -> RuleOutput:
    """
    DCM-like naive Bayes: posterior odds = prior odds x prod_j LR_j over the
    observed indicators (missing skipped), LR = sens / (1 - spec) for a
    present and (1 - sens) / spec for an absent indicator. No veto: an absent
    indicator only lowers the odds. Score = posterior probability.
    """
    if not 0.0 < prior < 1.0:
        raise ValueError("prior must be in (0, 1)")
    P = presence(C, threshold)
    sens = _thresholds(sensitivity)
    spec = _thresholds(specificity)
    if np.any((sens <= 0) | (sens >= 1) | (spec <= 0) | (spec >= 1)):
        raise ValueError("sensitivity and specificity must be in (0, 1)")
    log_lr_pos = np.log(sens / (1.0 - spec))
    log_lr_neg = np.log((1.0 - sens) / spec)
    contrib = np.where(
        P == 1, log_lr_pos[None, :], np.where(P == 0, log_lr_neg[None, :], 0.0)
    )
    log_odds = np.log(prior / (1.0 - prior)) + contrib.sum(axis=1)
    post = 1.0 / (1.0 + np.exp(-log_odds))
    return _decide(post, decision_threshold, "dcm_naive_bayes")


def rule_chalmers_product(
    C,
    credence: Optional[Callable] = None,
    missing_credence: float = 0.5,
    threshold: float = 0.5,
    cap: float = 1.0,
) -> RuleOutput:
    """
    Product of necessity credences: P = prod_j q_j with q_j the credence that
    principle j's necessary condition is met (default ``clip(c_j / cap, 0,
    1)``; a callable maps the matrix to credences). Missing components
    contribute ``missing_credence``. Score = the product.
    """
    x = as_component_matrix(C)
    if credence is None:
        q = np.clip(x / float(cap), 0.0, 1.0)
    else:
        q = np.asarray(credence(x), dtype=float)
    q = np.where(np.isfinite(x), q, float(missing_credence))
    return _decide(np.prod(q, axis=1), threshold, "chalmers_product")


def rule_logistic(
    C, labels, cv: int = 5, seed: int = 0, C_reg: float = 1.0, C_test=None
) -> RuleOutput:
    """
    Learned classifier (mean imputation, standardisation, L2 logistic
    regression). Without ``C_test`` the score is the out-of-fold probability
    (stratified ``cv``-fold cross-validation on ``C``); with ``C_test`` the
    model is fitted on all of ``C`` and scores ``C_test``.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = as_component_matrix(C)
    y = np.asarray(labels).astype(int).reshape(-1)
    if y.size != x.shape[0]:
        raise ValueError("labels must have one entry per row")
    model = make_pipeline(
        SimpleImputer(strategy="mean", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=float(C_reg), max_iter=1000),
    )
    if C_test is None:
        folds = StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=int(seed))
        prob = cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]
    else:
        model.fit(x, y)
        prob = model.predict_proba(as_component_matrix(C_test))[:, 1]
    return _decide(prob, 0.5, "logistic")


def rule_single_marker(
    values, threshold: float, name: str = "single_marker"
) -> RuleOutput:
    """One marker (IIM-only, NAS-only, LZc, Gaussian Phi_R): present ->
    MPC_CONSISTENT."""
    v = np.asarray(values, dtype=float).reshape(-1)
    return _decide(v, float(threshold), name)


def component_status_z(
    z, se=0.0, z_present: float = 1.645, delta_equiv: float = 1.0, alpha: float = 0.05
) -> np.ndarray:
    """
    Three-valued component status from null-standardised margins ``z`` (and
    SEs in null-SD units): PRESENT / ABSENT (TOST) / UNDEFINED, with NaN
    undefined. This is the superseded v1 status rule (null-SD units, no
    reference anchor), kept only for the legacy ``impact`` comparator of the
    rule audit; the evidence layer uses the v2 construct-scale rule, mirrored
    by :func:`component_status_c`. Parameter checks: finite thresholds,
    ``0 <= delta_equiv <= z_present`` (PRESENT and ABSENT cannot overlap),
    ``0 < alpha <= 0.5``; a negative SE is UNDEFINED.
    """
    z_present, delta_equiv, alpha = float(z_present), float(delta_equiv), float(alpha)
    if not (np.isfinite(z_present) and np.isfinite(delta_equiv)):
        raise ValueError("z_present and delta_equiv must be finite")
    if not 0.0 <= delta_equiv <= z_present:
        raise ValueError("need 0 <= delta_equiv <= z_present")
    if not 0.0 < alpha <= 0.5:
        raise ValueError("alpha must be in (0, 0.5]")
    z = np.asarray(z, dtype=float)
    se = np.broadcast_to(np.asarray(se, dtype=float), z.shape)
    zc = norm.ppf(1.0 - alpha)
    with np.errstate(invalid="ignore"):
        present = (z - zc * se) > z_present
        absent = ~present & ((np.abs(z) + zc * se) <= delta_equiv)
    out = np.full(z.shape, UNDEFINED, dtype=object)
    out[absent] = ABSENT
    out[present] = PRESENT
    out[~np.isfinite(z) | ~np.isfinite(se) | (se < 0)] = UNDEFINED
    return out


def kleene_and_rows(status: np.ndarray) -> np.ndarray:
    """Strong-Kleene AND per row over PRESENT (T) / ABSENT (F) / UNDEFINED (U)."""
    s = np.asarray(status, dtype=object)
    any_f = np.any(s == ABSENT, axis=1)
    all_t = np.all(s == PRESENT, axis=1)
    return np.where(
        any_f, EXCLUDED, np.where(all_t, MPC_CONSISTENT, UNDETERMINED)
    ).astype(object)


def rule_impact(
    Z,
    se=0.0,
    necessity_set: Sequence[str] = tuple(PRINCIPLE_INDEX),
    z_present: float = 1.645,
    delta_equiv: float = 1.0,
    alpha: float = 0.05,
) -> RuleOutput:
    """
    IMPaCT rule on null-standardised margins: component statuses, strong-Kleene
    AND over ``necessity_set``. Score = number of PRESENT components in the
    necessity set.
    """
    z = as_component_matrix(Z)
    se_arr = np.broadcast_to(np.asarray(se, dtype=float), z.shape)
    idx = [PRINCIPLE_INDEX[p] for p in necessity_set]
    status = component_status_z(z, se_arr, z_present, delta_equiv, alpha)[:, idx]
    dec = kleene_and_rows(status)
    score = np.sum(status == PRESENT, axis=1).astype(float)
    return RuleOutput("impact", score, dec)


def construct_scale(
    estimate,
    null_mean,
    reference,
    se_estimate=None,
    null_sd=None,
    n_null=None,
    se_reference=None,
    se_df=None,
    return_df=False,
):
    """
    Two-anchor construct scale ``c = (m - nu) / (rho - nu)`` and its SE
    (spec v2, V2-2): the sampling SE of ``m``, the Monte-Carlo error of the
    null mean ``sigma_null / sqrt(K)`` and (optionally) the SE of the
    reference are propagated through the ratio by the delta method,
    ``se_c^2 = [se_m^2 + (1 - c)^2 se_nu^2 + c^2 se_rho^2] / (rho - nu)^2``.
    Element-wise; ``c`` is NaN (invalid anchors) unless every anchor is
    finite and ``rho > nu``; ``se_c`` is NaN when the sampling SE
    ``se_estimate`` is missing, non-finite, zero or negative (spec V2-2:
    ``NO_SAMPLING_SE``; the Monte-Carlo error of the anchors never stands in
    for it; exact known-TPM values use ``exact=True`` in
    :func:`component_status_c`). Returns ``(c, se_c)``, or ``(c, se_c, df)``
    with ``return_df``: the Welch-Satterthwaite degrees of freedom
    ``se_df (se_c^2 / s_m^2)^2`` of ``se_c`` (``s_m = se_m / (rho - nu)``;
    ``se_df`` = degrees of freedom of the sampling SE, e.g. jackknife groups
    - 1), infinite where ``se_df`` is missing (normal quantile).
    """
    m, nu, rho = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (estimate, null_mean, reference))
    )
    ok = np.isfinite(m) & np.isfinite(nu) & np.isfinite(rho) & (rho > nu)
    span = np.where(ok, rho - nu, 1.0)
    c = np.where(ok, (m - nu) / span, np.nan)
    shape = c.shape

    def _arr(v, fill):
        if v is None:
            return np.full(shape, fill)
        return np.broadcast_to(np.asarray(v, dtype=float), shape)

    se_m = _arr(se_estimate, np.nan)
    with np.errstate(invalid="ignore"):
        se_m = np.where(np.isfinite(se_m) & (se_m > 0), se_m, np.nan)
    sd0 = _arr(null_sd, 0.0)
    k = _arr(n_null, np.inf)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        se_nu = np.where(np.isfinite(k) & (k > 0), sd0 / np.sqrt(k), 0.0)
        se_rho = _arr(se_reference, 0.0)
        var = (se_m**2 + (1.0 - c) ** 2 * se_nu**2 + c**2 * se_rho**2) / span**2
        se_c = np.where(ok, np.sqrt(var), np.nan)
        if not return_df:
            return c, se_c
        dfs = _arr(se_df, np.nan)
        s_m2 = se_m**2 / span**2
        ratio = var / s_m2
        df = np.where(
            np.isfinite(dfs) & (dfs > 0) & np.isfinite(s_m2) & (s_m2 > 0),
            dfs * ratio * ratio,
            np.inf,
        )
    return c, se_c, df


def one_sided_quantile(alpha: float, df=None):
    """``1 - alpha`` quantile: Student ``t`` with ``df`` degrees of freedom
    (element-wise), the normal quantile where ``df`` is None or infinite."""
    zc = float(norm.ppf(1.0 - float(alpha)))
    if df is None:
        return zc
    d = np.asarray(df, dtype=float)
    q = np.full(d.shape, zc)
    fin = np.isfinite(d) & (d > 0)
    q[fin] = stdtrit(d[fin], 1.0 - float(alpha))
    return q


def component_status_c(
    c,
    se,
    z_present=C_PRESENT_DEFAULT,
    delta_absent=C_ABSENT_DEFAULT,
    alpha: float = 0.05,
    exact=False,
    df=None,
) -> np.ndarray:
    """
    v2 component status on the construct scale (spec V2-2): PRESENT if
    ``c - q se > z_present``, ABSENT if ``c + q se < delta_absent`` (this
    includes estimates credibly *below* the null), else UNDEFINED, with
    ``q = z_{1-alpha}`` or, where ``df`` (degrees of freedom of ``se``, see
    :func:`construct_scale`) is finite, ``q = t_{1-alpha, df}``. A missing,
    non-finite or zero SE is UNDEFINED (no sampling SE) unless ``exact``
    (known-TPM computations: the value is exact, SE 0). ``z_present`` /
    ``delta_absent`` may be per-column arrays (principles);
    ``delta_absent <= z_present`` is required (exclusivity). A scalar ``c``
    gives a scalar status.
    """
    c = np.asarray(c, dtype=float)
    scalar = c.ndim == 0
    c = np.atleast_1d(c)
    zp = np.broadcast_to(np.asarray(z_present, dtype=float), c.shape[-1:] or (1,))
    da = np.broadcast_to(np.asarray(delta_absent, dtype=float), c.shape[-1:] or (1,))
    if np.any(~np.isfinite(zp)) or np.any(~np.isfinite(da)) or np.any(da > zp):
        raise ValueError("need finite cut-offs with delta_absent <= z_present")
    if not 0.0 < float(alpha) <= 0.5:
        raise ValueError("alpha must be in (0, 0.5]")
    se = np.broadcast_to(np.asarray(se, dtype=float), c.shape).copy()
    ex = np.broadcast_to(np.asarray(exact, dtype=bool), c.shape)
    se = np.where(ex & np.isfinite(c), 0.0, se)
    valid = np.isfinite(c) & np.isfinite(se) & ((se > 0) | ex)
    zc = one_sided_quantile(
        alpha, None if df is None else np.broadcast_to(np.asarray(df, float), c.shape)
    )
    with np.errstate(invalid="ignore"):
        present = valid & ((c - zc * se) > zp)
        absent = valid & ~present & ((c + zc * se) < da)
    out = np.full(c.shape, UNDEFINED, dtype=object)
    out[absent] = ABSENT
    out[present] = PRESENT
    return out[0] if scalar else out


def rule_impact_c(
    C,
    se,
    necessity_set: Sequence[str] = tuple(PRINCIPLE_INDEX),
    z_present=C_PRESENT_DEFAULT,
    delta_absent=C_ABSENT_DEFAULT,
    alpha: float = 0.05,
    exact=False,
    df=None,
) -> RuleOutput:
    """
    IMPaCT rule, v2 (construct scale): statuses by
    :func:`component_status_c` (``df``: degrees of freedom of ``se``),
    strong-Kleene AND over ``necessity_set``. Score = the smallest one-sided
    lower bound ``c - q se`` over the necessity set (NaN when any is
    undefined), a confidence for risk-coverage curves.
    """
    c = as_component_matrix(C)
    se_arr = np.broadcast_to(np.asarray(se, dtype=float), c.shape)
    df_arr = None if df is None else np.broadcast_to(np.asarray(df, float), c.shape)
    idx = [PRINCIPLE_INDEX[p] for p in necessity_set]
    status = component_status_c(
        c, se_arr, z_present, delta_absent, alpha, exact, df=df_arr
    )
    status = status[:, idx]
    dec = kleene_and_rows(status)
    zc = one_sided_quantile(alpha, df_arr)
    with np.errstate(invalid="ignore"):
        lower = (c - zc * se_arr)[:, idx]
    score = np.where(np.all(np.isfinite(lower), axis=1), lower.min(axis=1), np.nan)
    return RuleOutput("impact_c", score, dec)


def apply_rules(
    C,
    Z=None,
    labels=None,
    markers: Optional[Dict[str, np.ndarray]] = None,
    threshold=0.5,
    marker_thresholds: Optional[Dict[str, float]] = None,
    powers: Sequence[float] = (-1.0, 1.0),
    cv: int = 5,
    seed: int = 0,
    dcm_kwargs: Optional[dict] = None,
    C_se=None,
    impact_c_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Decisions of every rival rule for each row of ``C`` (normalised
    components). ``Z`` (null-standardised margins) adds the v1 IMPaCT rule,
    ``C_se`` (SEs of ``C`` on the construct scale) the v2 IMPaCT rule
    (:func:`rule_impact_c`), ``labels`` the cross-validated logistic
    classifier, ``markers`` external single markers (e.g. ``{'LZc': ...,
    'PhiR': ...}`` with thresholds in ``marker_thresholds``). IIM-only and
    NAS-only use columns 3 and 2 of ``C``.
    """
    x = as_component_matrix(C)
    outs = [
        rule_union(x, threshold),
        *[rule_count_k(x, k, threshold) for k in range(1, N_COMPONENTS + 1)],
        rule_arithmetic_mean(x, 0.5 if np.ndim(threshold) else threshold),
        rule_geometric_mean_uncapped(x, 0.5 if np.ndim(threshold) else threshold),
        rule_geometric_mean_capped(x, 0.5 if np.ndim(threshold) else threshold),
        *[
            rule_power_mean(x, p, 0.5 if np.ndim(threshold) else threshold)
            for p in powers
        ],
        rule_weakest_link(x, 0.5 if np.ndim(threshold) else threshold),
        rule_dcm_naive_bayes(x, threshold, **(dcm_kwargs or {})),
        rule_chalmers_product(x),
        rule_single_marker(x[:, 3], _thresholds(threshold)[3], "IIM_only"),
        rule_single_marker(x[:, 2], _thresholds(threshold)[2], "NAS_only"),
    ]
    for name, values in (markers or {}).items():
        thr = (marker_thresholds or {}).get(name)
        if thr is None:
            raise ValueError(f"marker {name!r} needs a threshold in marker_thresholds")
        outs.append(rule_single_marker(values, thr, name))
    if labels is not None:
        outs.append(rule_logistic(x, labels, cv=cv, seed=seed))
    if Z is not None:
        outs.append(rule_impact(Z))
    if C_se is not None:
        outs.append(rule_impact_c(x, C_se, **(impact_c_kwargs or {})))
    data = {}
    for o in outs:
        data[o.rule] = o.decision
        data[f"{o.rule}_score"] = o.score
    return pd.DataFrame(data)
