"""
Necessity statistics: Necessary Condition Analysis (NCA), set-theoretic
(fsQCA) necessity, and the symmetric three-outcome necessity criteria used by
the paper-2 hypothesis registry.

The MPC stance claims only *necessity* (every principle of the necessity set
``N`` is required for consciousness), so its empirical tests are tests of
necessary conditions, not of correlation or sufficiency.

Necessary Condition Analysis (Dul 2016, *Organ. Res. Methods* 19:10-52; Dul
2020, *Conducting Necessary Condition Analysis*, Sage; Dul, van der Laan & Kuik
2020, *Organ. Res. Methods* 23:385-395):

- ``X`` is necessary for ``Y`` ("high Y requires high X") when the upper-left
  corner of the ``(X, Y)`` scatter is empty. The *scope* is the rectangle
  ``[x_min, x_max] x [y_min, y_max]`` (empirical by default, or a declared
  theoretical scope that contains the data), of area ``S``.
- CE-FDH ceiling (ceiling envelopment, free disposal hull): the step function
  ``c(x) = max{y_i : x_i <= x}`` (``y_min`` where no observation has
  ``x_i <= x``). Its *peers* are the upper-left corner points of the steps.
- CR-FDH ceiling (ceiling regression): the OLS line through the CE-FDH peers,
  clipped to the scope. With fewer than two distinct peers the line is flat at
  the single peer.
- Ceiling zone ``C``: the area of the scope above the ceiling; effect size
  ``d = C / S`` in ``[0, 1]`` (Dul's labels: < 0.1 small, < 0.3 medium,
  < 0.5 large, otherwise very large; ``d = 0`` none). Ceiling accuracy: the
  fraction of observations on or below the ceiling (1 for CE-FDH by
  construction).
- Permutation test of ``d`` (Dul, van der Laan & Kuik 2020): under the null of
  independence ``Y`` is permuted against ``X`` (both marginals, and hence the
  empirical scope, are kept); ``p = (1 + #{d_perm >= d}) / (1 + B)``.
- Bootstrap: percentile interval of ``d`` over case resamples (the scope is
  re-estimated per resample when it is empirical). FDH ceilings are boundary
  estimators, so the bootstrap distribution is biased towards smaller ``d``;
  the bias is reported and the permutation test is the primary inference.
- ``corner`` selects the empty corner: 1 upper-left (default, "X necessary for
  Y"), 2 upper-right (absence of X necessary for Y), 3 lower-left, 4
  lower-right. Other corners are computed by reflecting the axes; intercepts,
  slopes and peers are reported in the original coordinates.

Set-theoretic necessity (Ragin 2006, *Political Analysis* 14:291-310;
Schneider & Wagemann 2012), for fuzzy memberships in ``[0, 1]``:
consistency ``sum min(x, y) / sum y``, coverage ``sum min(x, y) / sum x`` and
relevance of necessity ``RoN = sum (1 - x) / sum (1 - min(x, y))`` (low RoN
flags a trivially necessary, near-constant condition).

Symmetric three-outcome necessity criteria (spec V2-7): with ``k`` ABSENT
statuses among ``n`` report-positive episodes and exact one-sided
Clopper-Pearson bounds at level ``1 - alpha``,

- FALSIFIED if the lower bound of the ABSENT rate exceeds ``tau``;
- SUPPORTED if the upper bound is below ``tau`` and the component varies
  (it is credibly ABSENT somewhere else, e.g. in report-negative episodes);
- INDETERMINATE otherwise.

Neither outcome is favoured: both use the same ``alpha`` and the same
margin ``tau``. Abstentions (UNDEFINED statuses) never help: with
``missing="determinate"`` they shrink ``n``; with ``missing="worst_case"``
(partial identification) they count as ABSENT for the upper bound and as
PRESENT for the lower bound.

Verdict names: both the v1 names (ATTRIBUTED / NOT_ATTRIBUTED) and the v2
names (MPC_CONSISTENT / EXCLUDED) are accepted everywhere and normalised to
the v2 names by :func:`normalize_verdict`.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta_dist

NCA_CEILINGS = ("ce_fdh", "cr_fdh")
NCA_CORNERS = (1, 2, 3, 4)
EFFECT_SIZE_BANDS = ((0.1, "small"), (0.3, "medium"), (0.5, "large"))

SUPPORTED = "SUPPORTED"
FALSIFIED = "FALSIFIED"
INDETERMINATE = "INDETERMINATE"
NECESSITY_OUTCOMES = (SUPPORTED, FALSIFIED, INDETERMINATE)
MISSING_POLICIES = ("determinate", "worst_case")

# v2 verdict names (spec V2-1) and the accepted v1 aliases.
EXCLUDED = "EXCLUDED"
MPC_CONSISTENT = "MPC_CONSISTENT"
UNDETERMINED = "UNDETERMINED"
VERDICT_NAMES = (EXCLUDED, MPC_CONSISTENT, UNDETERMINED)
VERDICT_ALIASES = {
    "EXCLUDED": EXCLUDED,
    "NOT_ATTRIBUTED": EXCLUDED,
    "MPC_CONSISTENT": MPC_CONSISTENT,
    "ATTRIBUTED": MPC_CONSISTENT,
    "UNDETERMINED": UNDETERMINED,
}
STATUS_NAMES = ("PRESENT", "ABSENT", "UNDEFINED")

_TOL = 1e-12


# --------------------------------------------------------------------------
# verdict / status names
# --------------------------------------------------------------------------
def _enum_text(value):
    value = getattr(value, "value", value)
    return str(value).strip().upper()


def normalize_verdict(value):
    """
    v2 verdict name (EXCLUDED / MPC_CONSISTENT / UNDETERMINED) for a v1 or v2
    verdict (string or enum). ``None``/NaN/empty give ``None``; an unknown
    name raises ``ValueError``.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    txt = _enum_text(value)
    if txt in ("", "NAN", "NONE", "<NA>"):
        return None
    if txt.startswith("VERDICT."):
        txt = txt.split(".", 1)[1]
    try:
        return VERDICT_ALIASES[txt]
    except KeyError:
        raise ValueError(f"unknown MPC verdict {value!r}") from None


def normalize_status(value):
    """Component status name (PRESENT / ABSENT / UNDEFINED); missing -> UNDEFINED."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "UNDEFINED"
    txt = _enum_text(value)
    if txt.startswith("COMPONENTSTATUS."):
        txt = txt.split(".", 1)[1]
    if txt in ("", "NAN", "NONE", "<NA>"):
        return "UNDEFINED"
    if txt not in STATUS_NAMES:
        raise ValueError(f"unknown component status {value!r}")
    return txt


# --------------------------------------------------------------------------
# exact binomial bounds
# --------------------------------------------------------------------------
def clopper_pearson(k, n, alpha=0.05, side="two-sided"):
    """
    Exact (Clopper-Pearson) bounds for a binomial proportion ``k / n``.

    ``side="lower"`` / ``"upper"`` give the one-sided ``1 - alpha`` bound in
    the corresponding slot (the other slot is 0 or 1); ``"two-sided"`` gives
    the equal-tailed ``1 - alpha`` interval. Works element-wise on arrays;
    ``n = 0`` gives ``(0, 1)``.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if side not in ("two-sided", "lower", "upper"):
        raise ValueError("side must be 'two-sided', 'lower' or 'upper'")
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    k, n = np.broadcast_arrays(k, n)
    if np.any((k < 0) | (k > n) | (n < 0)):
        raise ValueError("need 0 <= k <= n")
    a = float(alpha) / 2.0 if side == "two-sided" else float(alpha)
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.where(k > 0, _beta_dist.ppf(a, np.maximum(k, 1e-300), n - k + 1), 0.0)
        hi = np.where(k < n, _beta_dist.ppf(1.0 - a, k + 1, np.maximum(n - k, 1e-300)),
                      1.0)
    lo = np.where(n > 0, lo, 0.0)
    hi = np.where(n > 0, hi, 1.0)
    if side == "lower":
        hi = np.ones_like(hi)
    elif side == "upper":
        lo = np.zeros_like(lo)
    if lo.ndim == 0:
        return float(lo), float(hi)
    return lo, hi


def necessity_thresholds(n, tau=0.05, alpha=0.05):
    """
    Decision thresholds of the symmetric criteria for ``n`` determinate
    report-positive episodes: ``k_F`` = the smallest ABSENT count whose
    one-sided lower bound exceeds ``tau`` (FALSIFIED for ``k >= k_F``; ``n + 1``
    if impossible) and ``k_S`` = the largest count whose one-sided upper bound
    is below ``tau`` (SUPPORTED-eligible for ``k <= k_S``; ``-1`` if
    impossible). Returns ``(k_F, k_S)``.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    _check_tau_alpha(tau, alpha)
    ks = np.arange(n + 1)
    lo, _ = clopper_pearson(ks, np.full(ks.shape, n), alpha, "lower")
    _, hi = clopper_pearson(ks, np.full(ks.shape, n), alpha, "upper")
    lo, hi = np.atleast_1d(lo), np.atleast_1d(hi)
    fals = np.flatnonzero(lo > tau)
    supp = np.flatnonzero(hi < tau)
    k_f = int(fals[0]) if fals.size else n + 1
    k_s = int(supp[-1]) if supp.size else -1
    return k_f, k_s


def min_n_for_support(tau=0.05, alpha=0.05):
    """
    Smallest ``n`` at which SUPPORTED is reachable at all (``k = 0``:
    ``1 - alpha**(1/n) < tau``, i.e. ``n > ln(alpha) / ln(1 - tau)``).
    """
    _check_tau_alpha(tau, alpha)
    bound = math.log(float(alpha)) / math.log(1.0 - float(tau))
    n = int(math.floor(bound)) + 1
    while n > 1 and 1.0 - float(alpha) ** (1.0 / (n - 1)) < tau:
        n -= 1
    while 1.0 - float(alpha) ** (1.0 / n) >= tau:
        n += 1
    return n


def _check_tau_alpha(tau, alpha):
    if not 0.0 < float(tau) < 1.0:
        raise ValueError("tau must be in (0, 1)")
    if not 0.0 < float(alpha) < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")


def necessity_threshold_table(n_values, tau=0.05, alpha=0.05) -> pd.DataFrame:
    """
    ``k_F(n)`` and ``k_S(n)`` (see :func:`necessity_thresholds`) for many
    ``n`` at once, through the exact binomial duality of the Clopper-Pearson
    bounds: the one-sided lower bound exceeds ``tau`` iff
    ``P(Bin(n, tau) >= k) < alpha``, and the upper bound is below ``tau`` iff
    ``P(Bin(n, tau) <= k) < alpha``.
    """
    from scipy.stats import binom

    _check_tau_alpha(tau, alpha)
    rows = []
    for n in (int(v) for v in n_values):
        if n < 0:
            raise ValueError("n must be >= 0")
        ks = np.arange(n + 1)
        fals = np.flatnonzero(binom.sf(ks - 1, n, tau) < alpha)
        supp = np.flatnonzero(binom.cdf(ks, n, tau) < alpha)
        rows.append({
            "n": n,
            "k_F": int(fals[0]) if fals.size else n + 1,
            "k_S": int(supp[-1]) if supp.size else -1,
        })
    out = pd.DataFrame(rows)
    out["tau"] = float(tau)
    out["alpha"] = float(alpha)
    return out


def observed_absent_rate(
    pi_absent, pi_absent_other=0.0, label_noise=0.0, sensitivity=1.0, false_absent=0.0
) -> float:
    """
    Probability that a determinate status is ABSENT in a labelled episode
    class, under label noise and status misclassification: a fraction
    ``label_noise`` of the episodes in the class truly belong to the other
    class; the component is truly absent with probability ``pi_absent`` in
    the class and ``pi_absent_other`` in the other class; a truly absent
    component is classified ABSENT with probability ``sensitivity`` and a
    truly present one with probability ``false_absent`` (UNDEFINED statuses
    are assumed missing at random and only reduce ``n``).
    """
    vals = (pi_absent, pi_absent_other, label_noise, sensitivity, false_absent)
    if any(not 0.0 <= float(v) <= 1.0 for v in vals):
        raise ValueError("all rates must lie in [0, 1]")

    def _q(pi):
        return float(pi) * float(sensitivity) + (1.0 - float(pi)) * float(false_absent)

    lam = float(label_noise)
    return (1.0 - lam) * _q(pi_absent) + lam * _q(pi_absent_other)


def necessity_outcome_probabilities(
    n_positive, q_positive, tau=0.05, alpha=0.05, *, n_negative=0, q_negative=0.0,
    variation_floor=0.0,
) -> dict:
    """
    Exact probabilities of the three outcomes of :func:`symmetric_necessity`
    (``missing="determinate"``) when the ABSENT count among ``n_positive``
    determinate report-positive episodes is ``Bin(n_positive, q_positive)``
    and the variation check uses ``Bin(n_negative, q_negative)`` ABSENT
    among the determinate report-negative episodes (independent).
    """
    from scipy.stats import binom

    n, nn = int(n_positive), int(n_negative)
    k_f, k_s = necessity_thresholds(n, tau, alpha)
    p_f = float(binom.sf(k_f - 1, n, q_positive)) if k_f <= n else 0.0
    p_low = float(binom.cdf(k_s, n, q_positive)) if k_s >= 0 else 0.0
    if nn <= 0:
        p_var = 0.0
    elif float(variation_floor) <= 0.0:
        p_var = 1.0 - (1.0 - float(q_negative)) ** nn
    else:
        k_var, _ = necessity_thresholds(nn, variation_floor, alpha)
        p_var = float(binom.sf(k_var - 1, nn, q_negative)) if k_var <= nn else 0.0
    p_s = p_low * p_var
    return {
        "P_SUPPORTED": p_s,
        "P_FALSIFIED": p_f,
        "P_INDETERMINATE": max(0.0, 1.0 - p_s - p_f),
        "k_F": k_f,
        "k_S": k_s,
        "P_varies": p_var,
    }


# --------------------------------------------------------------------------
# symmetric three-outcome necessity criteria (V2-7)
# --------------------------------------------------------------------------
def symmetric_necessity(
    k_absent,
    n_positive,
    tau=0.05,
    alpha=0.05,
    *,
    n_undefined=0,
    missing="determinate",
    varies=None,
    k_absent_negative=None,
    n_negative=None,
    variation_floor=0.0,
) -> dict:
    """
    Symmetric three-outcome necessity decision for one principle.

    ``k_absent`` ABSENT statuses among ``n_positive`` determinate
    (PRESENT or ABSENT) report-positive episodes; ``n_undefined`` further
    report-positive episodes had an UNDEFINED status. ``missing``:

    - ``"determinate"``: bounds on ``k / n_positive`` (UNDEFINED excluded);
    - ``"worst_case"``: partial identification over the ``n_positive +
      n_undefined`` episodes: the upper bound counts every UNDEFINED as
      ABSENT, the lower bound counts every UNDEFINED as PRESENT.

    Variation (non-triviality of the necessity claim): ``varies`` may be given
    directly; otherwise it is evaluated from ``k_absent_negative`` ABSENT among
    ``n_negative`` determinate report-negative episodes: the component varies
    when the one-sided lower bound of that ABSENT rate exceeds
    ``variation_floor`` (with the default 0 this means at least one ABSENT
    outside the report-positive episodes). Without any variation information
    the component is not shown to vary and SUPPORTED is unavailable.

    Returns a dict with ``outcome`` (SUPPORTED / FALSIFIED / INDETERMINATE),
    ``rate``, ``lower``, ``upper``, ``varies``, the inputs and the reason.
    """
    _check_tau_alpha(tau, alpha)
    if missing not in MISSING_POLICIES:
        raise ValueError(f"missing must be one of {MISSING_POLICIES}")
    k, n, u = int(k_absent), int(n_positive), int(n_undefined)
    if min(k, n, u) < 0 or k > n:
        raise ValueError("need 0 <= k_absent <= n_positive and n_undefined >= 0")
    if missing == "determinate":
        lower, _ = clopper_pearson(k, n, alpha, "lower")
        _, upper = clopper_pearson(k, n, alpha, "upper")
        rate = k / n if n else float("nan")
    else:
        lower, _ = clopper_pearson(k, n + u, alpha, "lower")
        _, upper = clopper_pearson(k + u, n + u, alpha, "upper")
        rate = k / n if n else float("nan")
    var_lower = float("nan")
    if varies is None:
        if k_absent_negative is not None and n_negative is not None:
            kn, nn = int(k_absent_negative), int(n_negative)
            var_lower, _ = clopper_pearson(kn, nn, alpha, "lower")
            varies = bool(nn > 0 and var_lower > float(variation_floor))
        else:
            varies = False
    varies = bool(varies)
    if lower > tau:
        outcome, reason = FALSIFIED, "lower_bound_above_tau"
    elif upper < tau and varies:
        outcome, reason = SUPPORTED, "upper_bound_below_tau"
    elif upper < tau:
        outcome, reason = INDETERMINATE, "component_does_not_vary"
    else:
        outcome, reason = INDETERMINATE, "interval_contains_tau"
    return {
        "outcome": outcome,
        "reason": reason,
        "rate": float(rate),
        "lower": float(lower),
        "upper": float(upper),
        "k_absent": k,
        "n_positive": n,
        "n_undefined": u,
        "missing": missing,
        "tau": float(tau),
        "alpha": float(alpha),
        "varies": varies,
        "variation_lower": float(var_lower),
    }


def necessity_from_statuses(
    statuses,
    report_positive,
    tau=0.05,
    alpha=0.05,
    *,
    missing="determinate",
    variation_floor=0.0,
) -> dict:
    """
    :func:`symmetric_necessity` from per-episode component statuses
    (PRESENT / ABSENT / UNDEFINED, v1/v2 enums or strings) and a boolean
    report-positive label; variation is evaluated on the report-negative
    episodes.
    """
    st = np.asarray([normalize_status(s) for s in statuses], dtype=object)
    pos = np.asarray(report_positive, dtype=bool).reshape(-1)
    if st.size != pos.size:
        raise ValueError("statuses and report_positive must align")
    det_pos = pos & (st != "UNDEFINED")
    det_neg = ~pos & (st != "UNDEFINED")
    return symmetric_necessity(
        int(np.sum(pos & (st == "ABSENT"))),
        int(det_pos.sum()),
        tau,
        alpha,
        n_undefined=int(np.sum(pos & (st == "UNDEFINED"))),
        missing=missing,
        k_absent_negative=int(np.sum(~pos & (st == "ABSENT"))),
        n_negative=int(det_neg.sum()),
        variation_floor=variation_floor,
    )


def verdict_level_summary(
    verdicts,
    report_positive,
    epsilon=0.05,
    alpha=0.05,
    *,
    missing="determinate",
) -> dict:
    """
    Verdict-level analysis (spec V2-7): EXCLUDED rate among report-positive
    episodes with exact one-sided bounds, and the symmetric decision against
    the margin ``epsilon`` (FALSIFIED if the lower bound exceeds ``epsilon``;
    SUPPORTED if the upper bound is below it; no variation requirement at the
    verdict level). Coverage is the fraction of determinate verdicts, overall
    and among report-positive / report-negative episodes. Verdict names may be
    v1 or v2.
    """
    v = np.asarray([normalize_verdict(x) for x in verdicts], dtype=object)
    pos = np.asarray(report_positive, dtype=bool).reshape(-1)
    if v.size != pos.size:
        raise ValueError("verdicts and report_positive must align")
    det = (v == EXCLUDED) | (v == MPC_CONSISTENT)
    k = int(np.sum(pos & (v == EXCLUDED)))
    n = int(np.sum(pos & det))
    u = int(np.sum(pos & ~det))
    res = symmetric_necessity(
        k, n, epsilon, alpha, n_undefined=u, missing=missing, varies=True
    )
    res["epsilon"] = res.pop("tau")
    res.pop("varies", None)
    res.pop("variation_lower", None)
    res["k_excluded"] = res.pop("k_absent")

    def _cov(mask):
        m = int(mask.sum())
        return float(np.sum(mask & det) / m) if m else float("nan")

    res.update(
        n_episodes=int(v.size),
        n_report_positive=int(pos.sum()),
        coverage=_cov(np.ones(v.size, dtype=bool)),
        coverage_report_positive=_cov(pos),
        coverage_report_negative=_cov(~pos),
        n_excluded_report_negative=int(np.sum(~pos & (v == EXCLUDED))),
        n_consistent_report_negative=int(np.sum(~pos & (v == MPC_CONSISTENT))),
    )
    return res


def kleene_verdicts(statuses: pd.DataFrame, necessity_set) -> np.ndarray:
    """
    v2 verdict names of the strong-Kleene AND over ``necessity_set`` from
    per-principle status columns ``<P>_status`` (EXCLUDED if any is ABSENT,
    MPC_CONSISTENT if all are PRESENT, else UNDETERMINED). Global reasons of
    the full verdict (bearer / protocol / single-source) are not visible here,
    so these verdicts are determinate at least as often as the pipeline's.
    """
    cols = [f"{p}_status" for p in necessity_set]
    missing = [c for c in cols if c not in statuses.columns]
    if missing:
        raise KeyError(f"missing status columns {missing}")
    st = np.vectorize(normalize_status, otypes=[object])(
        statuses[cols].to_numpy(dtype=object))
    return np.where((st == "ABSENT").any(axis=1), EXCLUDED,
                    np.where((st == "PRESENT").all(axis=1), MPC_CONSISTENT,
                             UNDETERMINED)).astype(object)


def coverage_by_necessity_set(statuses: pd.DataFrame, necessity_sets,
                              report_positive=None) -> pd.DataFrame:
    """
    Coverage (fraction of determinate Kleene verdicts) and verdict rates for
    each necessity set (spec V2-7, "coverage per N"); with ``report_positive``
    also per report class.
    """
    rows = []
    pos = None if report_positive is None else np.asarray(report_positive, bool)
    for nset in necessity_sets:
        nset = tuple(nset)
        v = kleene_verdicts(statuses, nset)
        groups = [("all", np.ones(v.size, bool))]
        if pos is not None:
            groups += [("report_positive", pos), ("report_negative", ~pos)]
        for name, mask in groups:
            m = int(mask.sum())
            row = {"necessity_set": ",".join(nset), "population": name, "n": m}
            for verdict in VERDICT_NAMES:
                row[f"rate_{verdict}"] = (float(np.sum(v[mask] == verdict) / m)
                                          if m else float("nan"))
            row["coverage"] = (1.0 - row[f"rate_{UNDETERMINED}"]) if m else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# selective prediction (risk-coverage)
# --------------------------------------------------------------------------
def risk_coverage(confidence, correct) -> dict:
    """
    Risk-coverage curve of a selective classifier: cases are accepted in
    order of decreasing ``confidence`` (ties are accepted together); at each
    acceptance level the coverage is the accepted fraction and the risk is the
    error rate among the accepted cases. Returns ``coverage``, ``risk``
    (arrays) and ``aurc`` (area under the curve by the step rule over the
    accepted levels; lower is better). Non-finite confidences are never
    accepted (abstentions).
    """
    conf = np.asarray(confidence, dtype=float).reshape(-1)
    ok = np.asarray(correct, dtype=bool).reshape(-1)
    if conf.size != ok.size:
        raise ValueError("confidence and correct must align")
    n = conf.size
    fin = np.isfinite(conf)
    if n == 0 or not fin.any():
        return {"coverage": np.zeros(0), "risk": np.zeros(0), "aurc": float("nan")}
    order = np.argsort(-conf[fin], kind="mergesort")
    c_sorted = conf[fin][order]
    err = (~ok[fin][order]).astype(float)
    cum_err = np.cumsum(err)
    # last index of each tie block
    last = np.r_[np.flatnonzero(np.diff(c_sorted) != 0), c_sorted.size - 1]
    accepted = last + 1
    coverage = accepted / n
    risk = cum_err[last] / accepted
    widths = np.diff(np.r_[0.0, coverage])
    aurc = float(np.sum(widths * risk) / coverage[-1])
    return {"coverage": coverage, "risk": risk, "aurc": aurc}


def selective_metrics(decisions, truth, alpha=0.05) -> dict:
    """
    Coverage and selective accuracy of a three-valued decision against binary
    truth: ``decisions`` in {True, False, None/NaN (abstain)}; accuracy is
    computed on the non-abstained cases with a two-sided Clopper-Pearson
    interval.
    """
    d = list(decisions)
    t = np.asarray(truth, dtype=bool).reshape(-1)
    if len(d) != t.size:
        raise ValueError("decisions and truth must align")
    made = np.asarray(
        [x is not None and not (isinstance(x, float) and math.isnan(x)) for x in d]
    )
    pred = np.asarray([bool(x) if m else False for x, m in zip(d, made)])
    n_made = int(made.sum())
    k = int(np.sum(made & (pred == t)))
    lo, hi = clopper_pearson(k, n_made, alpha) if n_made else (float("nan"),) * 2
    return {
        "n": int(t.size),
        "n_decided": n_made,
        "coverage": n_made / t.size if t.size else float("nan"),
        "selective_accuracy": k / n_made if n_made else float("nan"),
        "lower": lo,
        "upper": hi,
    }


# --------------------------------------------------------------------------
# NCA: scope, ceilings, effect size
# --------------------------------------------------------------------------
def _as_xy(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok], int(np.sum(~ok))


def _reflect(x, y, corner):
    corner = int(corner)
    if corner not in NCA_CORNERS:
        raise ValueError(f"corner must be one of {NCA_CORNERS}")
    fx = -1.0 if corner in (2, 4) else 1.0
    fy = -1.0 if corner in (3, 4) else 1.0
    return fx * x, fy * y, fx, fy


def nca_scope(x, y, scope=None):
    """
    ``(x_min, x_max, y_min, y_max)``: the empirical scope, or the declared
    theoretical ``scope`` (which must contain every observation).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    emp = (float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y)))
    if scope is None:
        return emp
    s = tuple(float(v) for v in scope)
    if len(s) != 4 or not all(math.isfinite(v) for v in s):
        raise ValueError("scope must be (x_min, x_max, y_min, y_max)")
    if s[0] > emp[0] + _TOL or s[1] < emp[1] - _TOL or s[2] > emp[2] + _TOL or (
        s[3] < emp[3] - _TOL
    ):
        raise ValueError("the theoretical scope must contain every observation")
    return s


def _reflected_scope(scope, fx, fy):
    if scope is None:
        return None
    x0, x1, y0, y1 = (float(v) for v in scope)
    xs = sorted((fx * x0, fx * x1))
    ys = sorted((fy * y0, fy * y1))
    return (xs[0], xs[1], ys[0], ys[1])


def _groups(xs_sorted):
    """Start indices of runs of equal x in a sorted array."""
    return np.r_[0, np.flatnonzero(np.diff(xs_sorted) > 0) + 1]


def _clip_antiderivative(u, lo, hi):
    """H(u) = int_lo^u clip(s, lo, hi) ds (vectorised)."""
    return np.where(
        u <= lo,
        lo * (u - lo),
        np.where(
            u <= hi,
            0.5 * (u * u - lo * lo),
            0.5 * (hi * hi - lo * lo) + hi * (u - hi),
        ),
    )


def _clipped_line_integral(a, b, x0, x1, lo, hi):
    """int_{x0}^{x1} clip(a + b x, lo, hi) dx, element-wise over a and b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    flat = np.abs(b) < 1e-300
    b_safe = np.where(flat, 1.0, b)
    u0, u1 = a + b_safe * x0, a + b_safe * x1
    h1 = _clip_antiderivative(u1, lo, hi)
    sloped = (h1 - _clip_antiderivative(u0, lo, hi)) / b_safe
    return np.where(flat, np.clip(a, lo, hi) * (x1 - x0), sloped)


def _nca_batch(xs, Y, scope, ceilings):
    """
    NCA quantities for a batch of y-vectors ``Y`` (B x n, columns aligned with
    the sorted ``xs``) on a fixed scope, for corner 1. Returns a dict per
    ceiling with arrays ``zone``, ``d``, ``accuracy`` (and ``intercept``,
    ``slope``, ``n_peers`` for CR-FDH).
    """
    x0, x1, y0, y1 = scope
    area = (x1 - x0) * (y1 - y0)
    starts = _groups(xs)
    gx = xs[starts]
    gmax = np.maximum.reduceat(Y, starts, axis=1)
    prev = np.maximum.accumulate(gmax, axis=1)
    prev = np.concatenate([np.full((Y.shape[0], 1), -np.inf), prev[:, :-1]], axis=1)
    peer = gmax > prev
    run = np.maximum.accumulate(gmax, axis=1)
    out = {}
    if "ce_fdh" in ceilings:
        widths = np.diff(np.r_[gx, x1])
        zone = (y1 - run) @ widths + (gx[0] - x0) * (y1 - y0)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(area > 0, zone / area, np.nan)
        out["ce_fdh"] = {
            "zone": zone,
            "d": d,
            "accuracy": np.ones(Y.shape[0]),
            "n_peers": peer.sum(axis=1),
        }
    if "cr_fdh" in ceilings:
        w = peer.astype(float)
        m = w.sum(axis=1)
        sx = w @ gx
        sy = (w * gmax).sum(axis=1)
        sxx = w @ (gx * gx)
        sxy = (w * gmax * gx[None, :]).sum(axis=1)
        den = m * sxx - sx * sx
        ok = (m >= 2) & (den > _TOL * np.maximum(1.0, m * sxx))
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = np.where(ok, (m * sxy - sx * sy) / np.where(ok, den, 1.0), 0.0)
            first = gmax[np.arange(Y.shape[0]), np.argmax(peer, axis=1)]
            intercept = np.where(ok, (sy - slope * sx) / np.maximum(m, 1.0), first)
        below = _clipped_line_integral(intercept, slope, x0, x1, y0, y1)
        zone = y1 * (x1 - x0) - below
        zone = np.maximum(zone, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(area > 0, zone / area, np.nan)
            line = intercept[:, None] + slope[:, None] * xs[None, :]
        tol = 1e-9 * max(1.0, abs(y1 - y0))
        acc = np.mean(Y <= line + tol, axis=1)
        out["cr_fdh"] = {
            "zone": zone,
            "d": d,
            "accuracy": acc,
            "intercept": intercept,
            "slope": slope,
            "n_peers": m.astype(int),
        }
    return out


def ce_fdh_peers(x, y, corner=1) -> np.ndarray:
    """
    CE-FDH peers (``m x 2`` array of ``(x, y)``, sorted by x): the upper-left
    corner points of the ceiling step function (one per distinct x at most),
    in the original coordinates for ``corner != 1``.
    """
    x, y, _ = _as_xy(x, y)
    if x.size == 0:
        return np.zeros((0, 2))
    xr, yr, fx, fy = _reflect(x, y, corner)
    order = np.lexsort((-yr, xr))
    xs, ys = xr[order], yr[order]
    starts = _groups(xs)
    gx = xs[starts]
    gmax = np.maximum.reduceat(ys, starts)
    prev = np.r_[-np.inf, np.maximum.accumulate(gmax)[:-1]]
    keep = gmax > prev
    peers = np.column_stack([fx * gx[keep], fy * gmax[keep]])
    return peers[np.argsort(peers[:, 0], kind="mergesort")]


def nca_effect_size(x, y, ceiling="ce_fdh", scope=None, corner=1) -> float:
    """NCA effect size ``d`` of one ceiling (NaN for a zero-area scope)."""
    res = nca(x, y, ceilings=(ceiling,), scope=scope, corner=corner)
    return float(res[ceiling]["d"])


def effect_size_label(d) -> str:
    """Dul's (2016) verbal label of an NCA effect size."""
    d = float(d)
    if not math.isfinite(d):
        return "undefined"
    if d <= 0.0:
        return "none"
    for bound, label in EFFECT_SIZE_BANDS:
        if d < bound:
            return label
    return "very large"


def _prepare(x, y, scope, corner):
    x, y, n_dropped = _as_xy(x, y)
    if x.size < 2:
        raise ValueError("NCA needs at least two finite observations")
    xr, yr, fx, fy = _reflect(x, y, corner)
    sc = nca_scope(xr, yr, _reflected_scope(scope, fx, fy))
    order = np.argsort(xr, kind="mergesort")
    return xr[order], yr[order], sc, fx, fy, n_dropped


def _ceilings_arg(ceilings):
    ceilings = (ceilings,) if isinstance(ceilings, str) else tuple(ceilings)
    bad = sorted(set(ceilings) - set(NCA_CEILINGS))
    if bad or not ceilings:
        raise ValueError(f"ceilings must be a non-empty subset of {NCA_CEILINGS}")
    return ceilings


def nca_permutation_test(
    x, y, ceilings=NCA_CEILINGS, n_permutations=1999, seed=0, scope=None, corner=1,
    chunk=512,
) -> dict:
    """
    Permutation test of the NCA effect size (Dul, van der Laan & Kuik 2020):
    ``y`` is permuted against ``x`` ``n_permutations`` times
    (``numpy.random.default_rng(seed)``); per ceiling the result has ``d``,
    ``p_value = (1 + #{d_perm >= d}) / (1 + B)``, the null mean and the 95th
    percentile of the null (``d_threshold``), and the null draws.
    """
    ceilings = _ceilings_arg(ceilings)
    xs, ys, sc, _, _, _ = _prepare(x, y, scope, corner)
    obs = _nca_batch(xs, ys[None, :], sc, ceilings)
    rng = np.random.default_rng(seed)
    b = int(n_permutations)
    if b < 1:
        raise ValueError("n_permutations must be >= 1")
    draws = {c: [] for c in ceilings}
    done = 0
    while done < b:
        m = min(int(chunk), b - done)
        perm = np.argsort(rng.random((m, ys.size)), axis=1)
        res = _nca_batch(xs, ys[perm], sc, ceilings)
        for c in ceilings:
            draws[c].append(res[c]["d"])
        done += m
    out = {}
    for c in ceilings:
        null = np.concatenate(draws[c])
        d = float(obs[c]["d"][0])
        exceed = int(np.sum(null >= d - 1e-12))
        out[c] = {
            "d": d,
            "p_value": (1.0 + exceed) / (1.0 + null.size),
            "null_mean": float(np.mean(null)),
            "d_threshold": float(np.quantile(null, 0.95)),
            "n_permutations": int(null.size),
            "null": null,
        }
    return out


def nca_bootstrap(
    x, y, ceilings=NCA_CEILINGS, n_bootstrap=999, seed=0, level=0.95, scope=None,
    corner=1,
) -> dict:
    """
    Case-resampling bootstrap of the NCA effect size: per ceiling the
    percentile interval (``lo``, ``hi``), the bootstrap SE and the bias
    ``mean(d*) - d`` (FDH ceilings are boundary estimators; expect a negative
    bias). An empirical scope is re-estimated in every resample; resamples
    with a zero-area scope are skipped.
    """
    ceilings = _ceilings_arg(ceilings)
    if not 0.0 < float(level) < 1.0:
        raise ValueError("level must be in (0, 1)")
    x, y, _ = _as_xy(x, y)
    base = nca(x, y, ceilings=ceilings, scope=scope, corner=corner)
    rng = np.random.default_rng(seed)
    boot = {c: [] for c in ceilings}
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, x.size, x.size)
        xb, yb = x[idx], y[idx]
        if np.ptp(xb) <= 0 or np.ptp(yb) <= 0:
            continue
        res = nca(xb, yb, ceilings=ceilings, scope=scope, corner=corner)
        for c in ceilings:
            boot[c].append(res[c]["d"])
    q = (0.5 - 0.5 * float(level), 0.5 + 0.5 * float(level))
    out = {}
    for c in ceilings:
        arr = np.asarray(boot[c], dtype=float)
        arr = arr[np.isfinite(arr)]
        d = base[c]["d"]
        if arr.size < 2:
            out[c] = {"d": d, "lo": np.nan, "hi": np.nan, "se": np.nan,
                      "bias": np.nan, "n_bootstrap": int(arr.size)}
            continue
        lo, hi = np.quantile(arr, q)
        out[c] = {
            "d": d,
            "lo": float(lo),
            "hi": float(hi),
            "se": float(np.std(arr, ddof=1)),
            "bias": float(np.mean(arr) - d),
            "n_bootstrap": int(arr.size),
            "level": float(level),
        }
    return out


def nca(
    x,
    y,
    ceilings=NCA_CEILINGS,
    scope=None,
    corner=1,
    *,
    n_permutations=0,
    n_bootstrap=0,
    seed=0,
    level=0.95,
) -> dict:
    """
    Necessary Condition Analysis of condition ``x`` for outcome ``y``.

    Per ceiling (``"ce_fdh"``, ``"cr_fdh"``) the result holds ``d`` (effect
    size), ``label`` (Dul's verbal band), ``ceiling_zone``, ``scope_area``,
    ``accuracy`` and ``n_peers``; CR-FDH adds ``intercept``/``slope`` of the
    ceiling line in the original coordinates. ``peers`` (CE-FDH corner
    points), ``scope`` (original coordinates), ``n`` and ``n_dropped``
    (non-finite pairs) are top-level. With ``n_permutations > 0`` each ceiling
    gets ``p_value``, ``null_mean`` and ``d_threshold``; with
    ``n_bootstrap > 0`` the percentile interval ``lo``/``hi`` and ``bias``.
    Seeds: the permutation test uses ``seed``, the bootstrap ``seed + 1``.
    """
    ceilings = _ceilings_arg(ceilings)
    xs, ys, sc, fx, fy, n_dropped = _prepare(x, y, scope, corner)
    res = _nca_batch(xs, ys[None, :], sc, ceilings)
    x0, x1, y0, y1 = sc
    out = {
        "n": int(xs.size),
        "n_dropped": n_dropped,
        "corner": int(corner),
        "scope": _reflected_scope(sc, fx, fy),
        "scope_area": float((x1 - x0) * (y1 - y0)),
        "peers": ce_fdh_peers(x, y, corner),
    }
    for c in ceilings:
        r = res[c]
        entry = {
            "d": float(r["d"][0]),
            "ceiling_zone": float(r["zone"][0]),
            "scope_area": out["scope_area"],
            "accuracy": float(r["accuracy"][0]),
            "n_peers": int(r["n_peers"][0]),
        }
        entry["label"] = effect_size_label(entry["d"])
        if c == "cr_fdh":
            # y' = a + b x' with x' = fx x and y' = fy y -> y = fy a + fy fx b x
            entry["intercept"] = float(fy * r["intercept"][0])
            entry["slope"] = float(fy * fx * r["slope"][0])
        out[c] = entry
    if int(n_permutations) > 0:
        perm = nca_permutation_test(
            x, y, ceilings, n_permutations=n_permutations, seed=seed, scope=scope,
            corner=corner,
        )
        for c in ceilings:
            out[c].update(
                p_value=perm[c]["p_value"],
                null_mean=perm[c]["null_mean"],
                d_threshold=perm[c]["d_threshold"],
                n_permutations=perm[c]["n_permutations"],
            )
    if int(n_bootstrap) > 0:
        boot = nca_bootstrap(
            x, y, ceilings, n_bootstrap=n_bootstrap, seed=int(seed) + 1, level=level,
            scope=scope, corner=corner,
        )
        for c in ceilings:
            out[c].update(
                lo=boot[c]["lo"], hi=boot[c]["hi"], boot_se=boot[c]["se"],
                bias=boot[c]["bias"], n_bootstrap=boot[c]["n_bootstrap"],
            )
    return out


def bottleneck_table(
    x, y, ceiling="ce_fdh", levels=tuple(range(0, 101, 10)), scope=None
) -> pd.DataFrame:
    """
    NCA bottleneck table (corner 1): for each outcome level (percent of the
    scope's y range) the minimum condition level (percent of the x range and
    raw) the ceiling requires. ``required = False`` marks "not necessary" (NN:
    the ceiling does not constrain x at that level); an unreachable level has
    NaN.
    """
    if ceiling not in NCA_CEILINGS:
        raise ValueError(f"ceiling must be one of {NCA_CEILINGS}")
    res = nca(x, y, ceilings=(ceiling,), scope=scope)
    x0, x1, y0, y1 = res["scope"]
    rows = []
    for lev in levels:
        yl = y0 + float(lev) / 100.0 * (y1 - y0)
        if ceiling == "ce_fdh":
            pk = res["peers"]
            hit = np.flatnonzero(pk[:, 1] >= yl - 1e-12)
            xr = float(pk[hit[0], 0]) if hit.size else float("nan")
            # levels at or below y_min are never constrained
            if yl <= y0 + 1e-12:
                xr = x0
        else:
            a, b = res[ceiling]["intercept"], res[ceiling]["slope"]
            xr = (yl - a) / b if abs(b) > 1e-300 else (x0 if a >= yl else float("nan"))
            if math.isfinite(xr) and xr > x1 + 1e-12:
                xr = float("nan")
        required = bool(math.isfinite(xr) and xr > x0 + 1e-12)
        xr_c = max(xr, x0) if math.isfinite(xr) else float("nan")
        pct = 100.0 * (xr_c - x0) / (x1 - x0) if x1 > x0 else float("nan")
        rows.append(
            {"y_level_pct": float(lev), "y_level": yl, "x_required": xr_c,
             "x_required_pct": pct, "required": required}
        )
    return pd.DataFrame(rows)


def nca_corner_contrast(x, y, ceiling="ce_fdh", scope=None) -> dict:
    """
    Necessity asymmetry: the effect size of the upper-left corner (``x``
    necessary for ``y``) minus that of the lower-right corner (``y`` necessary
    for ``x``). A centrally symmetric dependence (e.g. a bivariate Gaussian)
    empties both corners equally, so a merely correlated condition has a
    contrast near 0, whereas a planted ceiling (``y <= f(x)``) empties only
    the upper-left corner. Descriptive; use the permutation test of ``d`` for
    inference against independence.
    """
    d1 = nca(x, y, ceilings=(ceiling,), scope=scope, corner=1)[ceiling]["d"]
    d4 = nca(x, y, ceilings=(ceiling,), scope=scope, corner=4)[ceiling]["d"]
    return {"d_upper_left": d1, "d_lower_right": d4, "contrast": d1 - d4,
            "ceiling": ceiling}


# --------------------------------------------------------------------------
# set-theoretic (fsQCA) necessity and the binary special case
# --------------------------------------------------------------------------
def fsqca_necessity(x, y) -> dict:
    """
    Fuzzy-set necessity of ``x`` for ``y`` (memberships in ``[0, 1]``):
    ``consistency = sum min(x, y) / sum y``, ``coverage = sum min(x, y) /
    sum x`` and ``relevance = sum (1 - x) / sum (1 - min(x, y))`` (relevance
    of necessity, RoN). NaN where a denominator is 0.
    """
    x, y, n_dropped = _as_xy(x, y)
    if np.any((x < -_TOL) | (x > 1 + _TOL) | (y < -_TOL) | (y > 1 + _TOL)):
        raise ValueError("fuzzy memberships must lie in [0, 1]")
    mn = np.minimum(x, y)

    def _ratio(a, b):
        return float(a / b) if b > 0 else float("nan")

    return {
        "consistency": _ratio(mn.sum(), y.sum()),
        "coverage": _ratio(mn.sum(), x.sum()),
        "relevance": _ratio((1 - x).sum(), (1 - mn).sum()),
        "n": int(x.size),
        "n_dropped": n_dropped,
    }


def binary_necessity(x, y, tau=0.05, alpha=0.05) -> dict:
    """
    Crisp (binary) special case of necessity of ``x`` for ``y`` (0/1 arrays):
    the 2x2 counts, the counterexample rate ``P(x = 0 | y = 1)`` with exact
    one-sided bounds, the symmetric decision at margin ``tau`` (variation:
    ``x = 0`` observed among ``y = 0`` cases), set-theoretic consistency,
    coverage and relevance, and the NCA effect size (which is 0 with any
    counterexample and 1 without one when both x levels and both y levels
    occur).
    """
    x, y, n_dropped = _as_xy(x, y)
    if not (np.all(np.isin(x, (0.0, 1.0))) and np.all(np.isin(y, (0.0, 1.0)))):
        raise ValueError("binary_necessity needs 0/1 data")
    xb, yb = x.astype(bool), y.astype(bool)
    n11 = int(np.sum(xb & yb))
    n10 = int(np.sum(xb & ~yb))
    n01 = int(np.sum(~xb & yb))
    n00 = int(np.sum(~xb & ~yb))
    dec = symmetric_necessity(
        n01, n11 + n01, tau, alpha, k_absent_negative=n00, n_negative=n00 + n10
    )
    fs = fsqca_necessity(x, y)
    d = float("nan")
    if np.ptp(x) > 0 and np.ptp(y) > 0:
        d = nca_effect_size(x, y, "ce_fdh")
    return {
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "counterexample_rate": dec["rate"],
        "lower": dec["lower"],
        "upper": dec["upper"],
        "outcome": dec["outcome"],
        "reason": dec["reason"],
        "consistency": fs["consistency"],
        "coverage": fs["coverage"],
        "relevance": fs["relevance"],
        "nca_d": d,
        "n": int(x.size),
        "n_dropped": n_dropped,
    }


def binary_outcome_nca_d(x, y) -> float:
    """
    Closed form of the CE-FDH effect size for a binary outcome ``y`` and a
    continuous condition ``x`` (empirical scope):
    ``d = (min{x_i : y_i = 1} - x_min) / (x_max - x_min)``.
    """
    x, y, _ = _as_xy(x, y)
    pos = y >= 0.5
    if not pos.any() or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return float("nan")
    return float((x[pos].min() - x.min()) / (x.max() - x.min()))


def nca_table(
    data: pd.DataFrame,
    conditions: Sequence[str],
    outcome: str,
    *,
    ceilings=NCA_CEILINGS,
    n_permutations=0,
    n_bootstrap=0,
    seed=0,
    scopes: Mapping | None = None,
) -> pd.DataFrame:
    """
    One row per condition and ceiling: NCA effect size, label, accuracy,
    ceiling zone, permutation p-value and bootstrap interval (when
    requested), for the outcome column ``outcome`` of ``data``. ``scopes``
    maps a condition to its theoretical scope.
    """
    rows = []
    for i, cond in enumerate(conditions):
        res = nca(
            data[cond].to_numpy(dtype=float),
            data[outcome].to_numpy(dtype=float),
            ceilings=ceilings,
            scope=(scopes or {}).get(cond),
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            seed=int(seed) + 7919 * i,
        )
        for c in _ceilings_arg(ceilings):
            row = {"condition": cond, "outcome": outcome, "ceiling": c,
                   "n": res["n"], "n_dropped": res["n_dropped"]}
            row.update({k: v for k, v in res[c].items() if np.isscalar(v)})
            rows.append(row)
    return pd.DataFrame(rows)
