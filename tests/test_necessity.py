"""Known-answer and property tests of impact_pipeline.necessity (NCA, fsQCA,
symmetric three-outcome necessity criteria, verdict-level summaries)."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from impact_pipeline import necessity as nc


# --------------------------------------------------------------------------
# helpers: brute-force ceilings
# --------------------------------------------------------------------------
def _brute_ce_zone(x, y, scope, n_grid=40001):
    x0, x1, y0, y1 = scope
    grid = np.linspace(x0, x1, n_grid)
    mid = 0.5 * (grid[1:] + grid[:-1])
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    run = np.maximum.accumulate(ys)
    idx = np.searchsorted(xs, mid, side="right") - 1
    c = np.where(idx >= 0, run[np.clip(idx, 0, None)], y0)
    return float(np.sum((y1 - c) * np.diff(grid)))


def _brute_line_zone(a, b, scope, n_grid=40001):
    x0, x1, y0, y1 = scope
    grid = np.linspace(x0, x1, n_grid)
    mid = 0.5 * (grid[1:] + grid[:-1])
    c = np.clip(a + b * mid, y0, y1)
    return float(np.sum((y1 - c) * np.diff(grid)))


def _planted(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    return x, x * rng.random(n)


def _correlated(n, seed, rho=0.65):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    z = (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0))
    return z[:, 0], z[:, 1]


# --------------------------------------------------------------------------
# NCA ceilings: exact geometry
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ce_fdh_zone_matches_brute_force_including_ties(seed):
    rng = np.random.default_rng(seed)
    x = np.round(rng.random(40) * 10) / 10.0  # ties in x
    y = rng.random(40)
    res = nc.nca(x, y, ceilings=("ce_fdh",))
    brute = _brute_ce_zone(x, y, res["scope"])
    assert res["ce_fdh"]["ceiling_zone"] == pytest.approx(brute, rel=1e-3, abs=1e-6)
    assert res["ce_fdh"]["accuracy"] == 1.0
    # theoretical scope (wider than the data): zone left of the first
    # observation counts in full
    scope = (-0.5, 1.5, -0.2, 1.2)
    res_t = nc.nca(x, y, ceilings=("ce_fdh",), scope=scope)
    assert res_t["ce_fdh"]["ceiling_zone"] == pytest.approx(
        _brute_ce_zone(x, y, scope), rel=1e-3, abs=1e-6)
    assert res_t["scope_area"] == pytest.approx(2.0 * 1.4)


@pytest.mark.parametrize("seed", [3, 4])
def test_cr_fdh_is_ols_through_the_peers_and_zone_is_exact(seed):
    x, y = _planted(120, seed)
    res = nc.nca(x, y)
    peers = nc.ce_fdh_peers(x, y)
    b, a = np.polyfit(peers[:, 0], peers[:, 1], 1)
    assert res["cr_fdh"]["slope"] == pytest.approx(b, rel=1e-9)
    assert res["cr_fdh"]["intercept"] == pytest.approx(a, rel=1e-9, abs=1e-12)
    zone = _brute_line_zone(a, b, res["scope"])
    assert res["cr_fdh"]["ceiling_zone"] == pytest.approx(zone, rel=1e-3)
    line = a + b * x
    assert res["cr_fdh"]["accuracy"] == pytest.approx(np.mean(y <= line + 1e-9))
    # peers are the running-maximum corners: strictly increasing in x and y
    assert np.all(np.diff(peers[:, 0]) > 0) and np.all(np.diff(peers[:, 1]) > 0)


def test_corner_reflection_and_scope_errors():
    x, y = _planted(80, 5)
    for corner, (fx, fy) in {2: (-1, 1), 3: (1, -1), 4: (-1, -1)}.items():
        d = nc.nca(x, y, corner=corner)["ce_fdh"]["d"]
        assert d == pytest.approx(nc.nca(fx * x, fy * y)["ce_fdh"]["d"])
    with pytest.raises(ValueError, match="contain"):
        nc.nca(x, y, scope=(0.2, 1.0, 0.0, 1.0))
    flat = nc.nca(np.ones(5), np.arange(5.0))
    assert math.isnan(flat["ce_fdh"]["d"])
    with pytest.raises(ValueError):
        nc.nca(x, y, corner=5)
    assert nc.nca(np.r_[x, np.nan], np.r_[y, 1.0])["n_dropped"] == 1


def test_bottleneck_table_follows_the_ceilings():
    x, y = _planted(200, 6)
    ce = nc.bottleneck_table(x, y, "ce_fdh")
    peers = nc.ce_fdh_peers(x, y)
    assert ce.iloc[-1]["x_required"] == pytest.approx(peers[-1, 0])
    assert not bool(ce.iloc[0]["required"])
    assert np.all(np.diff(ce["x_required"].to_numpy()) >= -1e-12)
    cr = nc.bottleneck_table(x, y, "cr_fdh")
    res = nc.nca(x, y)
    a, b = res["cr_fdh"]["intercept"], res["cr_fdh"]["slope"]
    mid = cr[(cr["y_level_pct"] == 50)].iloc[0]
    assert mid["x_required"] == pytest.approx(max((mid["y_level"] - a) / b,
                                                  res["scope"][0]))


# --------------------------------------------------------------------------
# NCA: planted necessary vs merely correlated conditions
# --------------------------------------------------------------------------
def test_planted_necessary_condition_is_recovered():
    x, y = _planted(400, 7)
    res = nc.nca(x, y, n_permutations=199, n_bootstrap=100, seed=1)
    assert res["cr_fdh"]["d"] == pytest.approx(0.5, abs=0.06)
    assert 0.40 < res["ce_fdh"]["d"] <= 0.5 + 1e-9
    assert res["ce_fdh"]["label"] == "large"
    assert res["ce_fdh"]["p_value"] <= 0.01 and res["cr_fdh"]["p_value"] <= 0.01
    assert res["ce_fdh"]["lo"] <= res["ce_fdh"]["hi"]
    assert res["ce_fdh"]["bias"] <= 0.01
    contrast = nc.nca_corner_contrast(x, y)
    assert contrast["contrast"] > 0.35
    fs = nc.fsqca_necessity(x, y)
    assert fs["consistency"] == pytest.approx(1.0)  # y <= x everywhere


def test_merely_correlated_condition_has_symmetric_corners():
    contrasts, ds, cons = [], [], []
    for seed in range(5):
        x, y = _correlated(400, seed)
        contrasts.append(nc.nca_corner_contrast(x, y)["contrast"])
        ds.append(nc.nca(x, y)["ce_fdh"]["d"])
        cons.append(nc.fsqca_necessity(x, y)["consistency"])
    xp, yp = _planted(400, 7)
    d_planted = nc.nca(xp, yp)["ce_fdh"]["d"]
    assert abs(np.mean(contrasts)) < 0.08
    assert np.mean(ds) < d_planted - 0.15
    assert max(cons) < 0.95


def test_permutation_test_is_calibrated_under_independence():
    rng = np.random.default_rng(11)
    rejections = 0
    n_sets = 200
    for i in range(n_sets):
        x, y = rng.random(40), rng.random(40)
        p = nc.nca_permutation_test(x, y, ("ce_fdh",), n_permutations=99, seed=i)
        rejections += p["ce_fdh"]["p_value"] <= 0.05
    rate = rejections / n_sets
    assert 0.01 <= rate <= 0.11, rate


def test_permutation_and_bootstrap_are_deterministic():
    x, y = _planted(60, 8)
    a = nc.nca(x, y, n_permutations=50, n_bootstrap=30, seed=3)
    b = nc.nca(x, y, n_permutations=50, n_bootstrap=30, seed=3)
    for c in nc.NCA_CEILINGS:
        assert a[c]["p_value"] == b[c]["p_value"]
        assert a[c]["lo"] == b[c]["lo"] and a[c]["hi"] == b[c]["hi"]


def test_effect_size_labels_follow_dul():
    assert nc.effect_size_label(0.0) == "none"
    assert nc.effect_size_label(0.05) == "small"
    assert nc.effect_size_label(0.2) == "medium"
    assert nc.effect_size_label(0.4) == "large"
    assert nc.effect_size_label(0.7) == "very large"
    assert nc.effect_size_label(float("nan")) == "undefined"


def test_nca_table_rows():
    x, y = _planted(50, 9)
    df = pd.DataFrame({"a": x, "b": 1 - x, "y": y})
    tab = nc.nca_table(df, ["a", "b"], "y", n_permutations=19)
    assert len(tab) == 4
    assert {"condition", "ceiling", "d", "p_value", "label"} <= set(tab.columns)
    da = tab[(tab.condition == "a") & (tab.ceiling == "ce_fdh")]["d"].iloc[0]
    db = tab[(tab.condition == "b") & (tab.ceiling == "ce_fdh")]["d"].iloc[0]
    assert da > db


# --------------------------------------------------------------------------
# binary special cases and fsQCA
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_binary_outcome_effect_size_closed_form(seed):
    rng = np.random.default_rng(seed)
    x = rng.random(60)
    y = (rng.random(60) < x).astype(float)
    if y.min() == y.max():
        pytest.skip("degenerate draw")
    assert nc.nca(x, y)["ce_fdh"]["d"] == pytest.approx(nc.binary_outcome_nca_d(x, y))


def test_binary_necessity_two_by_two():
    # no counterexample (x=0, y=1): perfect crisp necessity, NCA d = 1
    x = np.r_[np.ones(80), np.zeros(40), np.ones(20)]
    y = np.r_[np.ones(80), np.zeros(40), np.zeros(20)]
    res = nc.binary_necessity(x, y, tau=0.05)
    assert (res["n11"], res["n10"], res["n01"], res["n00"]) == (80, 20, 0, 40)
    assert res["nca_d"] == 1.0
    assert res["consistency"] == 1.0
    assert res["coverage"] == pytest.approx(80 / 100)
    assert res["relevance"] == pytest.approx(40 / 60)
    assert res["outcome"] == nc.SUPPORTED  # n = 80 > 59, x = 0 occurs when y = 0
    # counterexamples: d = 0, and a large rate is FALSIFIED
    x2 = np.r_[x, np.zeros(30)]
    y2 = np.r_[y, np.ones(30)]
    res2 = nc.binary_necessity(x2, y2, tau=0.05)
    assert res2["nca_d"] == 0.0
    assert res2["counterexample_rate"] == pytest.approx(30 / 110)
    assert res2["outcome"] == nc.FALSIFIED
    with pytest.raises(ValueError):
        nc.binary_necessity([0, 0.5], [1, 0])


def test_fsqca_formulas_and_range_check():
    x = np.array([0.9, 0.6, 0.2, 1.0])
    y = np.array([0.8, 0.7, 0.1, 0.4])
    mn = np.minimum(x, y)
    fs = nc.fsqca_necessity(x, y)
    assert fs["consistency"] == pytest.approx(mn.sum() / y.sum())
    assert fs["coverage"] == pytest.approx(mn.sum() / x.sum())
    assert fs["relevance"] == pytest.approx((1 - x).sum() / (1 - mn).sum())
    with pytest.raises(ValueError):
        nc.fsqca_necessity([1.2], [0.5])


# --------------------------------------------------------------------------
# exact bounds and thresholds
# --------------------------------------------------------------------------
def test_clopper_pearson_known_values_and_symmetry():
    _, hi = nc.clopper_pearson(0, 59, 0.05, "upper")
    assert hi == pytest.approx(1 - 0.05 ** (1 / 59))
    lo, _ = nc.clopper_pearson(0, 59, 0.05, "lower")
    assert lo == 0.0
    lo2, hi2 = nc.clopper_pearson(7, 30, 0.05)
    lo3, hi3 = nc.clopper_pearson(23, 30, 0.05)
    assert lo2 == pytest.approx(1 - hi3) and hi2 == pytest.approx(1 - lo3)
    assert nc.clopper_pearson(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize("tau", [0.05, 0.1, 0.2])
def test_threshold_table_matches_clopper_pearson_duality(tau):
    tab = nc.necessity_threshold_table(range(1, 151), tau, 0.05).set_index("n")
    for n in range(1, 151):
        assert (tab.loc[n, "k_F"], tab.loc[n, "k_S"]) == nc.necessity_thresholds(
            n, tau, 0.05)
    # decisions at the thresholds agree with the one-sided exact binomial test
    n = 120
    kf, ks = nc.necessity_thresholds(n, tau, 0.05)
    assert binom.sf(kf - 1, n, tau) < 0.05 <= binom.sf(kf - 2, n, tau)
    if ks >= 0:
        assert binom.cdf(ks, n, tau) < 0.05 <= binom.cdf(ks + 1, n, tau)


def test_minimum_n_for_support():
    assert nc.min_n_for_support(0.05, 0.05) == 59
    assert nc.min_n_for_support(0.10, 0.05) == 29
    for tau in (0.02, 0.05, 0.1, 0.3):
        n = nc.min_n_for_support(tau, 0.05)
        assert nc.necessity_thresholds(n, tau, 0.05)[1] == 0
        assert nc.necessity_thresholds(n - 1, tau, 0.05)[1] == -1


# --------------------------------------------------------------------------
# symmetric three-outcome criteria
# --------------------------------------------------------------------------
def test_symmetric_necessity_known_outcomes():
    s = nc.symmetric_necessity(0, 100, 0.05, k_absent_negative=10, n_negative=50)
    assert s["outcome"] == nc.SUPPORTED and s["varies"]
    s = nc.symmetric_necessity(0, 100, 0.05, k_absent_negative=0, n_negative=50)
    assert s["outcome"] == nc.INDETERMINATE
    assert s["reason"] == "component_does_not_vary"
    s = nc.symmetric_necessity(0, 30, 0.05, varies=True)
    assert s["outcome"] == nc.INDETERMINATE  # n below 59: support unreachable
    s = nc.symmetric_necessity(20, 100, 0.05)
    assert s["outcome"] == nc.FALSIFIED and s["lower"] > 0.05


def test_symmetric_necessity_properties_on_random_counts():
    rng = np.random.default_rng(0)
    n_checked = 0
    for _ in range(2000):
        n = int(rng.integers(1, 300))
        k = int(rng.integers(0, n + 1))
        u = int(rng.integers(0, 50))
        tau = float(rng.choice([0.02, 0.05, 0.1, 0.25]))
        base = nc.symmetric_necessity(k, n, tau, varies=True)
        assert base["lower"] <= base["upper"]
        # monotone in k
        if k < n:
            nxt = nc.symmetric_necessity(k + 1, n, tau, varies=True)
            if base["outcome"] == nc.FALSIFIED:
                assert nxt["outcome"] == nc.FALSIFIED
            if nxt["outcome"] == nc.SUPPORTED:
                assert base["outcome"] == nc.SUPPORTED
        # no variation, no support
        flat = nc.symmetric_necessity(k, n, tau, varies=False)
        assert flat["outcome"] != nc.SUPPORTED
        # worst-case missingness is never less conservative
        wc = nc.symmetric_necessity(k, n, tau, n_undefined=u, missing="worst_case",
                                    varies=True)
        det = nc.symmetric_necessity(k, n, tau, n_undefined=u, varies=True)
        assert wc["outcome"] in (det["outcome"], nc.INDETERMINATE)
        n_checked += 1
    assert n_checked == 2000


def test_undefining_a_status_never_creates_a_worst_case_decision():
    """Missingness safety of the worst-case policy (cf. property P1)."""
    rng = np.random.default_rng(1)
    for _ in range(1500):
        n = int(rng.integers(5, 200))
        st = rng.choice(["PRESENT", "ABSENT", "UNDEFINED"], size=n,
                        p=[0.85, 0.05, 0.10])
        pos = rng.random(n) < 0.6
        before = nc.necessity_from_statuses(st, pos, 0.05, missing="worst_case")
        present = np.flatnonzero(pos & (st == "PRESENT"))
        if present.size == 0:
            continue
        st2 = st.copy()
        st2[rng.choice(present)] = "UNDEFINED"
        after = nc.necessity_from_statuses(st2, pos, 0.05, missing="worst_case")
        if before["outcome"] == nc.INDETERMINATE:
            assert after["outcome"] == nc.INDETERMINATE
        else:
            assert after["outcome"] in (before["outcome"], nc.INDETERMINATE)


def test_worst_case_variation_counts_undefined_negatives():
    """Regression: under missing='worst_case' the UNDEFINED report-negative
    statuses must not be dropped from the variation bound (dropping them
    made the worst case as optimistic as the determinate policy)."""
    st = np.r_[np.full(100, "PRESENT"), np.full(10, "ABSENT"),
               np.full(10, "PRESENT"), np.full(200, "UNDEFINED")].astype(object)
    pos = np.r_[np.ones(100, bool), np.zeros(220, bool)]
    det = nc.necessity_from_statuses(st, pos, 0.05, variation_floor=0.2)
    wc = nc.necessity_from_statuses(st, pos, 0.05, missing="worst_case",
                                    variation_floor=0.2)
    assert det["outcome"] == nc.SUPPORTED and det["varies"]
    lo_det, _ = nc.clopper_pearson(10, 20, 0.05, "lower")
    lo_wc, _ = nc.clopper_pearson(10, 220, 0.05, "lower")
    assert det["variation_lower"] == pytest.approx(lo_det)
    assert wc["variation_lower"] == pytest.approx(lo_wc)
    assert not wc["varies"] and wc["outcome"] == nc.INDETERMINATE
    # with the default floor 0 one ABSENT suffices under both policies
    assert nc.necessity_from_statuses(st, pos, 0.05, missing="worst_case")[
        "outcome"] == nc.SUPPORTED


def test_outcome_probabilities_match_simulation():
    rng = np.random.default_rng(5)
    n, q, nn, qn = 80, 0.02, 60, 0.05
    exact = nc.necessity_outcome_probabilities(n, q, 0.05, 0.05, n_negative=nn,
                                               q_negative=qn)
    counts = {o: 0 for o in nc.NECESSITY_OUTCOMES}
    reps = 4000
    for _ in range(reps):
        k = int(rng.binomial(n, q))
        kn = int(rng.binomial(nn, qn))
        counts[nc.symmetric_necessity(k, n, 0.05, k_absent_negative=kn,
                                      n_negative=nn)["outcome"]] += 1
    for o in nc.NECESSITY_OUTCOMES:
        assert counts[o] / reps == pytest.approx(exact[f"P_{o}"], abs=0.03)
    assert sum(exact[f"P_{o}"] for o in nc.NECESSITY_OUTCOMES) == pytest.approx(1.0)


def test_observed_absent_rate_mixture():
    q = nc.observed_absent_rate(0.0, 0.6, label_noise=0.02, sensitivity=0.9,
                                false_absent=0.0)
    assert q == pytest.approx(0.02 * 0.6 * 0.9)
    assert nc.observed_absent_rate(0.3) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        nc.observed_absent_rate(1.2)


# --------------------------------------------------------------------------
# verdict-level analyses and selective prediction
# --------------------------------------------------------------------------
def test_verdict_names_are_normalised_from_v1_and_v2():
    from impact_pipeline.evidence import ComponentStatus, Verdict

    for v in Verdict:
        assert nc.normalize_verdict(v) in nc.VERDICT_NAMES
    assert nc.normalize_verdict("ATTRIBUTED") == "MPC_CONSISTENT"
    assert nc.normalize_verdict("NOT_ATTRIBUTED") == "EXCLUDED"
    assert nc.normalize_verdict("excluded") == "EXCLUDED"
    assert nc.normalize_verdict(None) is None and nc.normalize_verdict(np.nan) is None
    with pytest.raises(ValueError):
        nc.normalize_verdict("CONSCIOUS")
    assert nc.normalize_status(ComponentStatus.ABSENT) == "ABSENT"
    assert nc.normalize_status(float("nan")) == "UNDEFINED"


def test_verdict_level_summary_counts_and_coverage():
    verdicts = (["ATTRIBUTED"] * 150 + ["NOT_ATTRIBUTED"] * 2 + ["UNDETERMINED"] * 48
                + ["EXCLUDED"] * 30 + ["UNDETERMINED"] * 20)
    pos = np.r_[np.ones(200, bool), np.zeros(50, bool)]
    s = nc.verdict_level_summary(verdicts, pos, epsilon=0.05)
    assert s["k_excluded"] == 2 and s["n_positive"] == 152 and s["n_undefined"] == 48
    assert s["coverage_report_positive"] == pytest.approx(152 / 200)
    assert s["coverage_report_negative"] == pytest.approx(30 / 50)
    assert s["n_excluded_report_negative"] == 30
    assert s["outcome"] == nc.SUPPORTED  # upper bound of 2/152 is below 0.05
    wc = nc.verdict_level_summary(verdicts, pos, epsilon=0.05, missing="worst_case")
    assert wc["outcome"] == nc.INDETERMINATE  # 48 abstentions could be exclusions


def test_coverage_per_necessity_set():
    df = pd.DataFrame({
        "RAM_status": ["PRESENT", "PRESENT", "ABSENT", "UNDEFINED"],
        "PDI_status": ["PRESENT", "UNDEFINED", "PRESENT", "ABSENT"],
        "NAS_status": ["PRESENT", "PRESENT", "PRESENT", np.nan],
    })
    v = nc.kleene_verdicts(df, ("RAM", "PDI"))
    assert v.tolist() == ["MPC_CONSISTENT", "UNDETERMINED", "EXCLUDED", "EXCLUDED"]
    cov = nc.coverage_by_necessity_set(df, [("RAM",), ("RAM", "PDI"), ("NAS",)],
                                       report_positive=[True, True, False, False])
    allp = cov[cov["population"] == "all"].set_index("necessity_set")
    assert allp.loc["RAM", "coverage"] == pytest.approx(0.75)
    assert allp.loc["RAM,PDI", "coverage"] == pytest.approx(0.75)
    assert allp.loc["NAS", "coverage"] == pytest.approx(0.75)
    pos = cov[(cov["population"] == "report_positive")].set_index("necessity_set")
    assert pos.loc["RAM,PDI", "rate_MPC_CONSISTENT"] == pytest.approx(0.5)
    with pytest.raises(KeyError):
        nc.kleene_verdicts(df, ("IIM",))


def test_risk_coverage_known_answer():
    rc = nc.risk_coverage([3, 2, 2, 1, np.nan], [1, 0, 1, 0, 1])
    assert rc["coverage"].tolist() == [0.2, 0.6, 0.8]
    assert rc["risk"].tolist() == pytest.approx([0.0, 1 / 3, 0.5])
    perfect = nc.risk_coverage([4, 3, 2, 1], [1, 1, 0, 0])
    worst = nc.risk_coverage([1, 2, 3, 4], [1, 1, 0, 0])
    assert perfect["aurc"] < worst["aurc"]
    sm = nc.selective_metrics([True, False, None, True], [True, True, False, True])
    assert sm["coverage"] == 0.75 and sm["selective_accuracy"] == pytest.approx(2 / 3)
