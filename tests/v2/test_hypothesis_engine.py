# -*- coding: utf-8 -*-
"""The declarative v2 evaluator: every decision rule and statistic on
hand-computed fixtures; NOT_EVALUABLE, NOT_TESTABLE_BY_DESIGN, removal and
the tally; the name map between the v2 protocols and the estimators; the
hypotheses file; the operating characteristics of the synthesis check; and a
synthetic bench (development seeds only, no simulation) on which every
HCv2 specification gives its predicted outcome, while a targeted change of
the records flips each one."""

import copy
import hashlib
import json
import math

import numpy as np
import pytest
from scipy import stats

from impact_pipeline import evidence_v2 as E
from impact_pipeline import necessity as NC
from impact_pipeline.v2 import hypothesis_engine as HE
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import records as REC
from impact_pipeline.v2 import seeds as S

SUP, FAL, IND = HE.SUPPORTED, HE.FALSIFIED, HE.INDETERMINATE
NE, NTBD = HE.NOT_EVALUABLE, HE.NOT_TESTABLE_BY_DESIGN


# ==========================================================================
# statistics on hand-computed fixtures
# ==========================================================================
@pytest.mark.parametrize(
    "n,level", [(20, 0.05), (42, 0.05), (45, 0.05), (150, 0.05), (61, 0.0125)]
)
def test_clopper_pearson_closed_forms(n, level):
    # 0 events: the upper bound solves (1 - p)^n = level; n events: p^n = level
    assert HE.cp_upper(0, n, level) == pytest.approx(1 - level ** (1 / n), abs=1e-12)
    assert HE.cp_lower(n, n, level) == pytest.approx(level ** (1 / n), abs=1e-12)
    assert HE.cp_lower(0, n, level) == 0.0 and HE.cp_upper(n, n, level) == 1.0


@pytest.mark.parametrize("k,n", [(1, 20), (11, 20), (14, 20), (26, 40), (5, 20)])
def test_clopper_pearson_inverts_the_binomial_and_matches_v1(k, n):
    up, lo = HE.cp_upper(k, n, 0.05), HE.cp_lower(k, n, 0.05)
    # the defining identities: P(X <= k | upper) = P(X >= k | lower) = 0.05
    assert stats.binom.cdf(k, n, up) == pytest.approx(0.05, abs=1e-9)
    assert stats.binom.sf(k - 1, n, lo) == pytest.approx(0.05, abs=1e-9)
    assert up == pytest.approx(NC.clopper_pearson(k, n, 0.05, "upper")[1], abs=1e-15)
    assert lo == pytest.approx(NC.clopper_pearson(k, n, 0.05, "lower")[0], abs=1e-15)


def test_seeds_needed_for_a_demonstration_match_the_design():
    # design 4.9 / synth_oc A: 42 / 51 / 61 / 66 seeds for 0.07 at m = 1/2/4/6,
    # 59 for 0.05, 149 for 0.02, 299 for 0.01
    for bound, level, n in [
        (0.07, 0.05, 42),
        (0.07, 0.025, 51),
        (0.07, 0.0125, 61),
        (0.07, 0.05 / 6, 66),
        (0.05, 0.05, 59),
        (0.02, 0.05, 149),
        (0.01, 0.05, 299),
    ]:
        assert HE.n_needed_zero_events(bound, level) == n


def test_cluster_counts_share_one_trial_per_simulation():
    # R and H scorings of one simulation are one cluster: an event on one of
    # its two rows counts 0.5
    c = HE.count_events([True, False, False, None], ["a", "a", "b", "b"])
    assert (c.k_rows, c.n_rows, c.n_clusters, c.k_eff, c.k_any, c.n_missing) == (
        1,
        3,
        2,
        0.5,
        1,
        1,
    )
    assert c.rate == 0.25


def test_newcombe_hybrid_score_interval_reproduces_newcombe_1998():
    # Newcombe (1998), method 10: 56/70 - 48/80 = 0.20, 95 % CI 0.0524 to 0.3339
    d, lo, hi = HE.newcombe_difference(56, 70, 48, 80)
    assert d == pytest.approx(0.2)
    assert (round(lo, 4), round(hi, 4)) == (0.0524, 0.3339)


def test_signed_rank_distribution_and_wilcoxon_exact():
    assert np.allclose(HE.signed_rank_pmf(3) * 8, [1, 1, 1, 2, 1, 1, 1])
    w = HE.wilcoxon_signed_rank([1, 2, 3, 4, 5])
    assert (w["t_plus"], w["method"]) == (15.0, "exact") and w["p"] == pytest.approx(
        1 / 32
    )
    # T+ = 10: subsets of {1..5} with sum >= 10 = complements with sum <= 5 (10 of 32)
    w = HE.wilcoxon_signed_rank([1, 2, 3, 4, -5])
    assert w["t_plus"] == 10.0 and w["p"] == pytest.approx(10 / 32)
    # zeros are dropped (Wilcoxon's convention)
    assert HE.wilcoxon_signed_rank([0, 0, 1, 2, 3, 4, 5])["p"] == pytest.approx(1 / 32)


def test_wilcoxon_normal_approximation_with_ties():
    # six tied differences: ranks 3.5 each, T+ = 21; var = 6*7*13/24 - (216-6)/48
    w = HE.wilcoxon_signed_rank([1, 1, 1, 1, 1, 1])
    var = 6 * 7 * 13 / 24 - (6**3 - 6) / 48
    z = (21 - 10.5 - 0.5) / math.sqrt(var)
    assert w["method"] == "normal" and w["p"] == pytest.approx(stats.norm.sf(z))


def test_sign_test_exact():
    s = HE.sign_test([1, 2, 3, 4, 5, 6, 7, 8, 9, -1, 0])
    assert (s["n"], s["positive"]) == (10, 9)
    assert s["p"] == pytest.approx(11 / 1024)
    assert HE.sign_test([-1, -2, -3], "less")["p"] == pytest.approx(1 / 8)


def test_hodges_lehmann_estimate_and_lower_bound():
    # 1..5: Walsh averages have median 3; P(T+ >= 15) = 1/32 <= 0.05 < P(T+ >= 14),
    # so the bound is the smallest Walsh average
    hl = HE.hodges_lehmann([1, 2, 3, 4, 5])
    assert (hl["estimate"], hl["lower"], hl["t_alpha"]) == (3.0, 1.0, 15)
    # 1..10: one-sided 0.05 critical value T = 10 (P = 0.042), t_alpha = 45, so the
    # bound is the 11th Walsh average: nine averages are <= 3, three equal 3.5
    hl = HE.hodges_lehmann(np.arange(1, 11))
    assert hl["t_alpha"] == 45 and hl["lower"] == 3.5 and hl["median"] == 5.5
    # n = 4 cannot reach 0.05 (1/16 > 0.05): no lower bound
    assert HE.hodges_lehmann([1, 2, 3, 4])["lower"] == -math.inf


def test_spearman_and_unit_slope():
    s = HE.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
    assert s["rho"] == pytest.approx(0.8)  # 1 - 6 * 4 / (5 * 24)
    t = 0.8 * math.sqrt(3 / (1 - 0.64))
    assert s["p"] == pytest.approx(stats.t.sf(t, 3))
    assert HE.ols_slope_unit([0, 1, 2], [0, 2, 4]) == pytest.approx(4.0)
    assert math.isnan(HE.spearman([1, 1, 1], [1, 2, 3])["rho"])


def test_mann_whitney_and_jonckheere_terpstra_exact():
    assert np.allclose(HE.mann_whitney_pmf(2, 2) * 6, [1, 1, 2, 1, 1])
    # fully ordered groups: JT = 12 = its maximum; P = 1 / (6! / (2! 2! 2!)) = 1/90
    jt = HE.jonckheere_terpstra([[1, 2], [3, 4], [5, 6]])
    assert (jt["jt"], jt["method"]) == (12.0, "exact") and jt["p"] == pytest.approx(
        1 / 90
    )
    # two groups: JT is Mann-Whitney's U; U >= 3 in 2 of the 6 arrangements
    jt = HE.jonckheere_terpstra([[1, 3], [2, 4]])
    assert jt["jt"] == 3.0 and jt["p"] == pytest.approx(1 / 3)


def test_jonckheere_terpstra_normal_with_ties():
    # [1, 1] vs [1, 2]: JT = 1 (ties) + 2 = 3; mean (16 - 8) / 4 = 2; the
    # tie-corrected variance is (156 - 36 - 66) / 72 + 0 + 4 * 6 / 96 = 1
    jt = HE.jonckheere_terpstra([[1, 1], [1, 2]])
    assert jt["method"] == "normal" and jt["jt"] == 3.0
    assert jt["p"] == pytest.approx(stats.norm.sf(1.0))


def test_kappa_interval_and_pooled_within_sd():
    sd, df = HE.pooled_within_sd([[1, 2, 3], [2, 4]])
    assert df == 3 and sd == pytest.approx(math.sqrt(4 / 3))
    lo, hi = HE.kappa_interval(sd, df, 0.9)
    # chi-square table: chi2(0.95, 3) = 7.8147, chi2(0.05, 3) = 0.3518
    assert lo == pytest.approx(sd * math.sqrt(3 / 7.814728), rel=1e-6)
    assert hi == pytest.approx(sd * math.sqrt(3 / 0.3518463), rel=1e-6)


def test_cluster_bootstrap_rate():
    assert HE.cluster_bootstrap_rate([0] * 5, [2] * 5, b=200) == (0.0, 0.0)
    # one of ten equal clusters carries the events: the rate is (times drawn) / 10,
    # Binomial(10, 0.1) / 10; its 5 % quantile is 0 and its 95 % quantile 0.3
    lo, hi = HE.cluster_bootstrap_rate([1] + [0] * 9, [1] * 10, b=20000, seed=1)
    assert (lo, hi) == (0.0, 0.3)


def test_outcome_combination():
    c = HE.combine_outcomes
    assert c([SUP, SUP, NE]) == SUP
    assert c([SUP, FAL, NE]) == FAL
    assert c([SUP, IND]) == IND
    assert c([NE, NE]) == NE and c([]) == NE
    assert c([NTBD, NTBD]) == NTBD and c([NTBD, NE]) == NE and c([NTBD, SUP]) == SUP


# ==========================================================================
# rules on prepared rows
# ==========================================================================
def prep(cells, *, pool=None):
    """Prepared rows from ``{cell: [events]}`` (one cluster per row)."""
    rows = []
    for cell, events in cells.items():
        for i, e in enumerate(events):
            rows.append(
                {
                    "_cell": cell,
                    "_event": e,
                    "_cluster": f"{cell}:{i}",
                    "_pool": pool or cell.split("|")[-1],
                    "member": cell,
                }
            )
    return HE.Prepared(rows)


def ev(k, n):
    return [True] * k + [False] * (n - k)


def rule(name, cells, **params):
    return HE.RULES[name](prep(cells), params)


def test_three_zone_rule():
    assert rule("three_zone", {"c": ev(16, 20)}, x=0.8).outcome == SUP
    assert rule("three_zone", {"c": ev(11, 20)}, x=0.8).outcome == FAL  # upper 0.75
    assert rule("three_zone", {"c": ev(14, 20)}, x=0.8).outcome == IND  # upper 0.86
    assert HE.cp_upper(11, 20) < 0.8 < HE.cp_upper(14, 20)
    # Bonferroni over m cells widens the upper bound
    assert HE.cp_upper(12, 20, 0.05) < 0.8 < HE.cp_upper(12, 20, 0.05 / 10)
    assert rule("three_zone", {"c": ev(12, 20)}, x=0.8).outcome == FAL
    assert rule("three_zone", {"c": ev(12, 20)}, x=0.8, m=10).outcome == IND
    r = rule("three_zone", {"a": ev(20, 20), "b": ev(5, 20)}, x=0.8)
    assert r.outcome == FAL and [c["outcome"] for c in r.cells] == [SUP, FAL]
    assert rule("three_zone", {"c": ev(9, 9)}, x=0.8, min_n=10).outcome == NE


def test_reversed_and_at_most_three_zone_rules():
    assert rule("reversed_three_zone", {"c": ev(0, 20)}, x=0.8).outcome == SUP  # 0.139
    assert rule("reversed_three_zone", {"c": ev(16, 20)}, x=0.8).outcome == FAL
    assert rule("reversed_three_zone", {"c": ev(15, 20)}, x=0.8).outcome == IND
    assert rule("three_zone_at_most", {"c": ev(4, 20)}, x=0.2).outcome == SUP
    assert rule("three_zone_at_most", {"c": ev(12, 20)}, x=0.2).outcome == FAL
    assert rule("three_zone_at_most", {"c": ev(6, 20)}, x=0.2).outcome == IND


def test_count_rule_two_of_twenty():
    # SUPPORTED iff <= 2/20, FALSIFIED iff >= 5/20 (CP lower > 0.10), else INDETERMINATE
    out = [
        rule("count_at_most", {"c": ev(k, 20)}, max_rate=0.1, falsify_above=0.1).outcome
        for k in range(7)
    ]
    assert out == [SUP, SUP, SUP, IND, IND, FAL, FAL]


def test_rate_lower_bound_rule_needs_26_of_40():
    # SUPPORTED iff CP lower > 0.5 (>= 26/40); FALSIFIED iff CP upper < 0.5
    # (<= 14/40: upper 0.4919, while 15/40 gives 0.5172)
    out = {
        k: rule("rate_lower_bound", {"c": ev(k, 40)}, x=0.5).outcome
        for k in (14, 15, 20, 25, 26)
    }
    assert out == {14: FAL, 15: IND, 20: IND, 25: IND, 26: SUP}
    assert HE.cp_upper(14, 40) < 0.5 < HE.cp_upper(15, 40)


def test_demonstration_rule_with_curtailment():
    assert rule("demonstration", {"v": ev(0, 149)}, bound=0.02).outcome == SUP
    assert rule("demonstration", {"v": ev(0, 148)}, bound=0.02).outcome == FAL
    # incomplete: 0 events in 100 of 150 planned runs can still demonstrate
    assert (
        rule("demonstration", {"v": ev(0, 100)}, bound=0.02, n_planned=150).outcome
        == IND
    )
    # curtailed: one event already rules the demonstration out at 150 runs
    assert HE.cp_upper(1, 150, 0.05) >= 0.02
    assert (
        rule("demonstration", {"v": ev(1, 100)}, bound=0.02, n_planned=150).outcome
        == FAL
    )


def test_h0_cell_rule():
    # pooled per principle at 0.05/P (P = 2): 0/50 -> 0.0711 (not below 0.07),
    # 0/60 -> 0.0596 (below)
    p = {"bound": 0.07}
    cells50 = {"x|NAS": ev(0, 50), "x|IIM": ev(0, 50)}
    cells60 = {"x|NAS": ev(0, 60), "x|IIM": ev(0, 60)}
    assert HE.cp_upper(0, 50, 0.025) > 0.07 > HE.cp_upper(0, 60, 0.025)
    assert HE.RULES["h0_cell"](prep(cells50), p).outcome == IND
    assert HE.RULES["h0_cell"](prep(cells60), p).outcome == SUP
    bad = {"x|NAS": ev(10, 50), "x|IIM": ev(0, 60)}
    assert HE.cp_lower(10, 50, 0.025) > 0.07
    assert HE.RULES["h0_cell"](prep(bad), p).outcome == FAL
    # the mirrored rule of a predicted exceedance
    assert HE.RULES["h0_cell_exceeds"](prep(bad), p).outcome == SUP
    assert HE.RULES["h0_cell_exceeds"](prep(cells60), p).outcome == FAL


def test_h0_cell_rule_counts_rescorings_of_one_simulation_once():
    # 45 simulations scored under R and H: 90 rows but 45 clusters in the pool
    rows = [
        {"_cell": "c", "_event": False, "_cluster": f"s{i}", "_pool": "NAS"}
        for i in range(45)
        for _ in ("R", "H")
    ]
    r = HE.RULES["h0_cell"](HE.Prepared(rows), {"bound": 0.07})
    assert r.stats["pooled"]["NAS"]["clusters"] == 45
    assert r.stats["pooled"]["NAS"]["upper"] == pytest.approx(1 - 0.05 ** (1 / 45))


def test_any_event_cluster_rule():
    assert (
        rule("any_event_clusters", {"c": ev(0, 45)}, bound=0.07).outcome == SUP
    )  # 0.0644
    assert (
        rule("any_event_clusters", {"c": ev(0, 20)}, bound=0.07).outcome == IND
    )  # 0.139
    assert rule("any_event_clusters", {"c": ev(10, 20)}, bound=0.07).outcome == FAL


def test_usable_share_rule():
    r = HE.RULES["usable_share"](prep({"g_b": ev(36, 40)}), {"share": 0.9})
    assert r.outcome == SUP and r.cells[0]["required"] == 36
    assert HE.RULES["usable_share"](prep({"g_b": ev(35, 40)}), {}).outcome == FAL


def test_false_exclusion_rule():
    # 45 seed clusters, 0 events, 20 cells: CP upper at 0.05/20 = 0.1249 < 0.15
    cells = {f"c{i}": ev(0, 45) for i in range(20)}
    r = HE.RULES["false_exclusion"](prep(cells), {})
    assert r.outcome == SUP
    assert r.cells[0]["seed_upper"] == pytest.approx(1 - 0.0025 ** (1 / 45))
    cells["c0"] = ev(15, 45)
    assert HE.RULES["false_exclusion"](prep(cells), {}).outcome == FAL


def value_prep(cells):
    rows = []
    for cell, vals in cells.items():
        for i, v in enumerate(vals):
            rows.append({"_cell": cell, "_value": v, "_cluster": i, "_pool": cell})
    return HE.Prepared(rows)


def test_value_rules():
    d = value_prep({"a": list(np.arange(1, 11) / 10)})
    assert HE.RULES["median_threshold"](d, {"op": "ge", "x": 0.5}).outcome == SUP
    assert HE.RULES["median_threshold"](d, {"op": "lt", "x": 0.25}).outcome == FAL
    r = HE.RULES["hodges_lehmann"](
        d, {"median_op": "ge", "median_x": 0.5, "lower_gt": 0.25}
    )
    assert r.outcome == SUP and r.cells[0]["lower"] == pytest.approx(0.35)
    assert HE.RULES["hodges_lehmann"](d, {"lower_gt": 0.4}).outcome == FAL
    assert HE.RULES["wilcoxon"](d, {"alpha": 0.0125}).outcome == SUP  # p = 1/1024
    assert HE.RULES["sign_test"](d, {"alpha": 0.0125}).outcome == SUP
    neg = value_prep({"a": [-0.1, 0.2, -0.3, 0.1, -0.2]})
    assert HE.RULES["sign_test"](neg, {"alpha": 0.05}).outcome == FAL


def test_dose_and_group_rules():
    rows = [
        {
            "_cell": "A",
            "_dose": d,
            "_value": 0.5 * d + 0.01 * (i % 3),
            "_cluster": i,
            "_group": d,
            "_pool": "A",
        }
        for i, d in enumerate(np.repeat(np.arange(5), 4))
    ]
    r = HE.RULES["spearman_ols"](HE.Prepared(rows), {"alpha": 0.05})
    assert r.outcome == SUP and r.cells[0]["slope_unit_dose"] == pytest.approx(
        2.0, abs=0.02
    )
    for row in rows:
        row["_value"] = -row["_value"]
    assert HE.RULES["spearman_ols"](HE.Prepared(rows), {"alpha": 0.05}).outcome == FAL
    grp = [
        {"_cell": "x", "_group": g, "_value": v, "_cluster": 0, "_pool": "x"}
        for g, vals in (("R", [1, 2]), ("Q10", [3, 4]), ("Q25", [5, 6]))
        for v in vals
    ]
    p = {"order": ["R", "Q10", "Q25"], "alpha": 0.05, "medians": "strictly_increasing"}
    r = HE.RULES["jonckheere_terpstra"](HE.Prepared(grp), p)
    assert r.outcome == SUP and r.cells[0]["p"] == pytest.approx(1 / 90)
    assert (
        HE.RULES["jonckheere_terpstra"](HE.Prepared(grp), {**p, "alpha": 0.01}).outcome
        == FAL
    )
    grp[-1]["_group"] = "other"  # Q25 keeps one value
    grp[-2]["_group"] = "other"
    assert HE.RULES["jonckheere_terpstra"](HE.Prepared(grp), p).outcome == NE


def test_newcombe_rule():
    rows = [
        {"_cell": "all", "member": m, "_event": e, "_cluster": f"{m}{i}", "_pool": "x"}
        for m, k in (("scr", 30), ("pc", 32))
        for i, e in enumerate(ev(k, 40))
    ]
    p = {"a": "scr", "b": "pc"}
    assert HE.RULES["newcombe_includes_zero"](HE.Prepared(rows), p).outcome == SUP
    rows = [
        {"_cell": "all", "member": m, "_event": e, "_cluster": f"{m}{i}", "_pool": "x"}
        for m, k in (("scr", 5), ("pc", 38))
        for i, e in enumerate(ev(k, 40))
    ]
    assert HE.RULES["newcombe_includes_zero"](HE.Prepared(rows), p).outcome == FAL


def _z(n):
    """n values with mean 0 and SD 1 (ddof 1) and |z| < 2."""
    z = np.linspace(-1.0, 1.0, n)
    return (z - z.mean()) / z.std(ddof=1)


def test_kappa_twins_rule_and_its_sides():
    def rows(scale):
        out = []
        for net in range(10):
            for r, z in enumerate(_z(7)):
                out.append(
                    {
                        "_cell": "PC",
                        "_network": net,
                        "_value": net + scale * 0.02 * z,
                        "se_c": 0.02,
                        "df_c": 9.0,
                        "principle": "IIM",
                        "_cluster": net,
                        "_pool": "x",
                        "estimator_version": "iim-v5",
                        "se_method": "circular_block_bootstrap_10pct_B50",
                    }
                )
        return HE.Prepared(out)

    r = HE.RULES["kappa_twins"](rows(1.0), {})
    assert r.outcome == SUP and r.cells[0]["kappa"] == pytest.approx(1.0)
    assert r.cells[0]["df"] == 60
    r = HE.RULES["kappa_twins"](rows(2.0), {})
    assert r.outcome == FAL and r.cells[0]["side"] == HE.SIDE_ANTI
    r = HE.RULES["kappa_twins"](rows(0.5), {})
    assert r.outcome == FAL and r.cells[0]["side"] == HE.SIDE_CONSERVATIVE


def test_kappa_null_rule():
    rng = np.random.default_rng(0)
    z = stats.norm.ppf((np.arange(400) + 0.5) / 400)
    rng.shuffle(z)

    def rows(scale):
        return HE.Prepared(
            [
                {
                    "_cell": "NAS",
                    "_value": scale * 0.02 * v,
                    "se_c": 0.02,
                    "df_c": 9.0,
                    "_cluster": i,
                    "_pool": "NAS",
                }
                for i, v in enumerate(z)
            ]
        )

    r = HE.RULES["kappa_null"](rows(1.0), {})
    assert r.outcome == SUP and r.cells[0]["kappa"] == pytest.approx(1.0, abs=0.02)
    r = HE.RULES["kappa_null"](rows(0.5), {})
    assert r.outcome == FAL and r.cells[0]["side"] == HE.SIDE_CONSERVATIVE


def _twin_rows(cell, *, n_nets=5, n_sessions=7, se=0.02, method="m", scale=1.0):
    out = []
    for net in range(n_nets):
        for r, z in enumerate(_z(n_sessions)):
            out.append(
                {
                    "_cell": cell,
                    "_network": net,
                    "_value": net + scale * se * z,
                    "se_c": se,
                    "df_c": 9.0,
                    "principle": "PDI",
                    "estimator_version": "pdi-v3",
                    "se_method": method,
                }
            )
    return out


def test_kappa_rules_with_zero_rms_are_not_evaluable():
    # SEs that are all 0 (admitted concordant components carry no sampling
    # SE) leave kappa undefined: NOT_EVALUABLE with a reason, not a crash
    rows = [
        {"_cell": "PDI", "_value": 0.01 * i, "se_c": 0.0, "_cluster": i}
        for i in range(10)
    ]
    r = HE.RULES["kappa_null"](HE.Prepared(rows), {})
    assert r.outcome == NE
    assert r.cells[0]["reason"] == HE.ZERO_RMS_REASON and r.cells[0]["n"] == 10
    # the twin rule: pooled SD 0 over RMS 0 (one count per network)
    twins = _twin_rows("PDI", se=0.0, method="concordant")
    r = HE.RULES["kappa_twins"](HE.Prepared(twins), {})
    assert r.outcome == NE and r.cells[0]["reason"] == HE.ZERO_RMS_REASON
    # a zero-RMS cell does not hide another cell's outcome
    rows += [
        {"_cell": "NAS", "_value": 0.02 * z, "se_c": 0.02, "_cluster": 100 + i}
        for i, z in enumerate(stats.norm.ppf((np.arange(400) + 0.5) / 400))
    ]
    r = HE.RULES["kappa_null"](HE.Prepared(rows), {})
    assert r.outcome == SUP
    assert {c["cell"]: c["outcome"] for c in r.cells} == {"PDI": NE, "NAS": SUP}


def test_kappa_twins_per_se_method_counts_the_class_sessions():
    # a class of 5 twin networks x 7 sessions in which 33 sessions are
    # admitted concordant components (SE 0, no sampling SE) and 2 carry the
    # jackknife SE: the concordant sessions form no cell and enter no kappa,
    # and the jackknife cell has c defined in 2 of 35 sessions, so it is not
    # eligible (HCv2-4: >= 80 %)
    rows = _twin_rows("A:W|PDI", se=0.0, method="concordant")
    for row in rows[:2]:
        row.update(se_method="jackknife_contiguous_10", se_c=0.3)
    r = HE.RULES["kappa_twins"](HE.Prepared(rows), {"by_se_method": True})
    (cell,) = r.cells
    assert cell["cell"] == "A:W|PDI|jackknife_contiguous_10"
    assert cell["outcome"] == NE and cell["defined_share"] == pytest.approx(2 / 35)
    assert r.outcome == NE
    # undefined components (no SE method) count among the sessions too
    rows = _twin_rows("A:X|IIM", method="bootstrap")
    for row in rows[:5]:
        row.update(se_method=None, _value=None, se_c=None)
    r = HE.RULES["kappa_twins"](HE.Prepared(rows), {"by_se_method": True})
    (cell,) = r.cells
    assert cell["cell"] == "A:X|IIM|bootstrap"
    assert cell["defined_share"] == pytest.approx(30 / 35) and cell["outcome"] == SUP
    # without sessions lacking a sampling SE the per-method cells are the
    # cells of a selection labelled by the method (same labels, order, m)
    rows = (
        _twin_rows("A:Y|PDI", method="jk")
        + _twin_rows("A:Z|IIM", method="bootstrap", scale=2.0)
        + _twin_rows("A:Y|PDI", method="other", scale=0.5)
    )
    labelled = copy.deepcopy(rows)
    for row in labelled:
        row["_cell"] = f"{row['_cell']}|{row['se_method']}"
    a = HE.RULES["kappa_twins"](HE.Prepared(rows), {"by_se_method": True})
    b = HE.RULES["kappa_twins"](HE.Prepared(labelled), {})
    assert [c["cell"] for c in a.cells] == ["A:Y|PDI|jk", "A:Z|IIM|bootstrap",
                                            "A:Y|PDI|other"]
    assert a.cells == b.cells and a.outcome == b.outcome == FAL
    assert a.stats == b.stats == {"m": 6}


def test_admission_and_anchor_rules():
    reg = HE.Prepared(
        [
            {"_cell": v, "admitted_for_present": a, "anchor_valid": True}
            for v, a in (("source", "yes"), ("eeg64", "no"))
        ]
    )
    p = {
        "predictions": {
            "source": {"admitted_for_present": "yes"},
            "eeg64": {"admitted_for_present": "no"},
        },
        "anchor_cell": "source",
    }
    assert HE.RULES["admission_matches"](reg, p).outcome == SUP
    reg.rows[1]["admitted_for_present"] = "yes"
    assert HE.RULES["admission_matches"](reg, p).outcome == FAL
    reg.rows[0]["anchor_valid"] = False
    assert HE.RULES["admission_matches"](reg, p).outcome == NE

    def anchor_rows(lesion_mean, pred):
        rows = []
        for i in range(20):
            for member, mean in (("pc", 1.0), ("lesion", lesion_mean)):
                rows.append(
                    {
                        "_cell": "A-R|PDI",
                        "member": member,
                        "seed": 900 + i,
                        "_value": mean + 0.05 * math.sin(i),
                        "principle": "PDI",
                        "_protocol": "A-R",
                        "_predicted_anchor": pred,
                    }
                )
        return HE.Prepared(rows)

    assert (
        HE.RULES["anchor_replicates"](anchor_rows(0.0, "valid_specific"), {}).outcome
        == SUP
    )
    r = HE.RULES["anchor_replicates"](anchor_rows(0.9, "valid_specific"), {})
    assert r.outcome == FAL and r.cells[0]["observed"] == "valid_nonspecific"
    assert r.cells[0]["flag"] == R.ANCHOR_NOT_REPLICATED


# ==========================================================================
# predicates, fields and the part semantics
# ==========================================================================
def test_three_valued_predicates_and_fields():
    f = HE.Fields(
        {"z_in": "details.te_in.z"}, {"sig": {"all": [{"@z_in": {"gt": 1.645}}]}}
    )
    row = {
        "status": "PRESENT",
        "reason": None,
        "details": {"te_in": {"z": 3.0}},
        "config": {"cell_id": "b10110"},
        "principle": "NAS",
        "estimate": 0.3,
        "null_mean": 0.1,
    }
    assert HE.evaluate_predicate({"@sig": True}, row, f) is True
    assert HE.evaluate_predicate({"@z_in": {"lt": 1}}, {"details": {}}, f) is None
    assert (
        HE.evaluate_predicate({"any": [{"x": 1}, {"status": "PRESENT"}]}, row, f)
        is True
    )
    assert HE.evaluate_predicate({"not": {"x": 1}}, row, f) is None
    # derived fields
    assert f.get(row, "reason_code") == ""  # decided: no reason, not a missing value
    assert (
        f.get(row, "own_bit") == 1
        and f.get(row, "bit.RAM") == 1
        and f.get(row, "bit.PDI") == 0
    )
    assert f.get(row, "excess") == pytest.approx(0.2)
    assert HE.evaluate_predicate({"seed": {"min": 1, "max": 3}}, {"seed": 2}) is True
    # numbers are equal within 1e-6 (G_nom = 8/7 is written 1.142857)
    assert HE.evaluate_predicate({"G": 1.142857}, {"G": 8 / 7}) is True
    assert HE.evaluate_predicate({"G": 1.14}, {"G": 8 / 7}) is False
    with pytest.raises(HE.SpecError):
        HE.evaluate_predicate({"x": {"bogus": 1}}, {"x": 1})


def mini_spec(parts, **extra):
    """A valid hypotheses file with the shipped scaffold and ``parts`` as
    HCv2-1's parts."""
    spec = copy.deepcopy(HE.load_spec())
    for h in spec["hypotheses"]:
        if h["id"] == "HCv2-1":
            h["parts"] = parts
    spec.update(extra)
    return HE.validate_spec(spec)


def comp_row(
    task,
    system,
    status,
    c,
    *,
    principle="NAS",
    decl="R",
    family="A",
    protocol="A-R",
    design="witnesses",
    seed=0,
    **kw,
):
    return {
        "task_id": task,
        "design": design,
        "family": family,
        "system": system,
        "seed": seed,
        "declaration_id": decl,
        "principle": principle,
        "status": status,
        "reason": None if status != "UNDEFINED" else "INCONCLUSIVE",
        "c": c,
        "estimator_form": "primary",
        "protocol_id": protocol,
        "flags": [],
        "config": {},
        "details": {},
        **kw,
    }


def test_part_semantics_not_evaluable_gated_removed_and_the_tally():
    rows = [comp_row(f"t{i}", "W", "ABSENT", 0.0, seed=i) for i in range(20)]
    base = {
        "rule": "three_zone",
        "params": {"x": 0.8},
        "data": {"where": {"system": "W"}, "event": {"status": "ABSENT"}},
    }
    parts = [
        {"id": "p-ok", **base},
        {
            "id": "p-none",
            **base,
            "data": {"where": {"system": "nope"}, "event": {"status": "ABSENT"}},
        },
        {
            "id": "p-gated",
            **base,
            "gate": {
                "kind": "absent",
                "family": "A-R",
                "principle": "NAS",
                "witness": "W",
            },
            "replaced_by": "p-ok",
        },
        {
            "id": "p-gate-missing",
            **base,
            "gate": {
                "kind": "absent",
                "family": "A-R",
                "principle": "IIM",
                "witness": "W",
            },
        },
        {"id": "p-fixed", **base, "gating_outcome": "NOT_TESTABLE_BY_DESIGN"},
        {
            "id": "p-removed",
            "status": "removed",
            "removed": {"reason": "slip rule", "development_outcome": "SUPPORTED"},
        },
        {"id": "p-reported", **base, "role": "reported"},
        {"id": "p-oracle", **base, "requires": {"usable": ["g_b"]}},
    ]
    spec = mini_spec(parts)
    proto = {
        "A-R": {
            "necessity_set": list(HE.PRINCIPLES),
            "precision": {
                "rows": [
                    {
                        "family": "A-R",
                        "principle": "NAS",
                        "witness": "W",
                        "kind": "absent",
                        "pi": 0.4,
                        "decisive": False,
                        "replacement": "target not PRESENT in >= 80 % of seeds",
                        "predicted_absent_not_reachable_rate": 0.6,
                    }
                ]
            },
        }
    }
    ctx = HE.build_context(sources={"components": rows}, protocols=proto)
    ctx.usability = {("A", "g_b"): False}
    h = next(h for h in spec["hypotheses"] if h["id"] == "HCv2-1")
    res = HE.evaluate_hypothesis(h, spec, ctx)
    out = {p["id"]: p["outcome"] for p in res["parts"]}
    assert out == {
        "p-ok": SUP,
        "p-none": NE,
        "p-gated": NTBD,
        "p-gate-missing": NE,
        "p-fixed": NTBD,
        "p-removed": HE.REMOVED,
        "p-reported": SUP,
        "p-oracle": NE,
    }
    gated = next(p for p in res["parts"] if p["id"] == "p-gated")
    assert gated["replacement"].startswith("target not PRESENT")
    assert gated["gate"]["predicted_absent_not_reachable_rate"] == 0.6
    assert "ORACLE" in next(p for p in res["parts"] if p["id"] == "p-oracle")["reason"]
    assert res["outcome"] == SUP  # the only evaluable decisive part
    t = HE.tally([res])
    assert t["decisive_parts"] == {
        SUP: 1,
        FAL: 0,
        IND: 0,
        NE: 3,
        NTBD: 2,
        HE.REMOVED: 1,
    }
    assert t["removed_parts"] == [
        {
            "hypothesis": "HCv2-1",
            "part": "p-removed",
            "predicted_outcome": "SUPPORTED",
            "reason": "slip rule",
        }
    ]
    assert t["reported_parts"] == 1


def test_per_cell_gates_and_cell_predictions():
    rows = [
        comp_row(f"{p}{i}", "W", "ABSENT", 0.0, protocol=p, seed=i)
        for p in ("A-R", "A-H")
        for i in range(20)
    ]
    part = {
        "id": "p",
        "rule": "three_zone",
        "params": {"x": 0.8},
        "data": {
            "where": {"system": "W"},
            "event": {"status": "ABSENT"},
            "cell_label": "{protocol_id}",
        },
        "gate": {
            "per_cell": True,
            "kind": "absent",
            "family": "{protocol_id}",
            "principle": "NAS",
            "witness": "{system}",
        },
        "cell_predictions": [
            {"match": {"protocol_id": "A-R"}, "prediction": "SUPPORTED"}
        ],
    }
    spec = mini_spec([part])
    prec = {
        "rows": [
            {
                "family": f,
                "principle": "NAS",
                "witness": "W",
                "kind": "absent",
                "decisive": d,
            }
            for f, d in (("A-R", True), ("A-H", False))
        ]
    }
    ctx = HE.build_context(
        sources={"components": rows}, protocols={"A-R": {"precision": prec}}
    )
    res = HE.evaluate_part(part, spec, ctx)
    assert res["outcome"] == SUP
    cells = {c["cell"]: c for c in res["cells"]}
    assert cells["A-R"]["outcome"] == SUP and cells["A-R"]["prediction"] == SUP
    assert cells["A-H"]["outcome"] == NTBD
    prec["rows"][0]["decisive"] = False
    ctx = HE.build_context(
        sources={"components": rows}, protocols={"A-R": {"precision": prec}}
    )
    assert HE.evaluate_part(part, spec, ctx)["outcome"] == NTBD


def test_an_evaluator_error_is_recorded_not_hidden(monkeypatch):
    rows = [comp_row(f"t{i}", "W", "ABSENT", 0.0, seed=i) for i in range(5)]
    part = {
        "id": "p",
        "rule": "three_zone",
        "params": {"x": 0.8},
        "data": {"where": {"system": "W"}, "event": {"status": "ABSENT"}},
    }
    spec = mini_spec([part])

    def boom(prep, params):
        raise RuntimeError("defect")

    monkeypatch.setitem(HE.RULES, "three_zone", boom)
    rep = HE.evaluate(spec, HE.build_context(sources={"components": rows}))
    assert rep["evaluator_errors"] == [
        {
            "hypothesis": "HCv2-1",
            "part": "p",
            "reason": "EVALUATOR_ERROR: RuntimeError: defect",
        }
    ]
    h1 = next(h for h in rep["hypotheses"] if h["id"] == "HCv2-1")
    assert h1["parts"][0]["outcome"] == NE


def test_tier_b_items_are_not_run():
    spec = HE.load_spec()
    res = HE.evaluate(spec, HE.build_context())
    assert [h for h, o in res["outcomes"].items() if o == HE.NOT_RUN] == [
        f"HCv2-B{i}" for i in range(1, 9)
    ]
    assert res["tally"]["not_run"] == [f"HCv2-B{i}" for i in range(1, 9)]
    assert res["status"] == HE.DEVELOPMENT_FLAG


def test_reversion_rule_reports_verdict_level_hypotheses_both_ways():
    comps = [
        comp_row(
            f"t{i}",
            "PC_nominal",
            "ABSENT",
            0.0,
            principle="RAM",
            seed=i,
            estimator_version="ram-v3-2026.10",
            se_method="shift_null_sd",
        )
        for i in range(3)
    ]
    verds = [
        {
            "task_id": f"t{i}",
            "design": "witnesses",
            "protocol_id": "A-R",
            "verdict_value": "EXCLUDED",
            "verdict": {"verdict": "EXCLUDED"},
            "component_status": {"RAM": "ABSENT", "NAS": "PRESENT"},
            "component_method": {
                "RAM": ["ram-v3-2026.10", "shift_null_sd"],
                "NAS": ["nas-v3-2026.10", "jackknife_contiguous_10"],
            },
        }
        for i in range(3)
    ]
    ctx = HE.build_context(
        sources={"components": comps, "verdicts": verds},
        protocols={"A-R": {"necessity_set": ["RAM", "NAS"]}},
    )
    new = HE.apply_reversion(ctx, {("ram-v3-2026.10", "shift_null_sd")})
    assert {r["status"] for r in new.sources["components"]} == {"UNDEFINED"}
    assert {r["reason"] for r in new.sources["components"]} == {R.SE_NOT_CALIBRATED}
    assert {v["verdict_value"] for v in new.sources["verdicts"]} == {"UNDETERMINED"}
    assert {r["status"] for r in ctx.sources["components"]} == {"ABSENT"}  # untouched
    # an EXCLUDED verdict that also rests on a calibrated ABSENT stays EXCLUDED
    verds[0]["component_status"]["NAS"] = "ABSENT"
    new = HE.apply_reversion(ctx, {("ram-v3-2026.10", "shift_null_sd")})
    assert new.sources["verdicts"][0]["verdict_value"] == "EXCLUDED"


# ==========================================================================
# the hypotheses file
# ==========================================================================
def test_the_shipped_hypotheses_file_is_valid_and_complete():
    spec = HE.load_spec()
    ids = [h["id"] for h in spec["hypotheses"]]
    assert ids[:25] == [f"HCv2-{i}" for i in range(25)]
    assert [e["id"] for e in spec["integrity_audit"]] == [
        f"IA-{i}" for i in range(1, 11)
    ]
    for h in spec["hypotheses"]:
        if h["tier"] == "A":
            assert h["parts"], h["id"]
            assert any(p.get("role", "decisive") == "decisive" for p in h["parts"])
        else:
            assert h["admitted"] is False and not h["parts"]
    # the declared calibration items name existing parts or hypotheses
    pids = {p["id"] for h in spec["hypotheses"] for p in h["parts"]} | set(ids)
    for item in spec["pending_calibration"]:
        assert all(a == "all" or a in pids for a in item["affects"]), item
    # the decided CD-7 and CD-8 tables (copied from the decided build)
    assert spec["declared_dependencies"]["status"] == "final"
    assert spec["mechanism_on"]["status"] == "final"
    mech = spec["mechanism_on"]
    assert mech["entries"] and mech["dose_only_entries"]
    keys = [
        (e["family"], e["declaration"], e["principle"], json.dumps(e["where"]))
        for e in mech["entries"]
    ]
    assert len(keys) == len(set(keys))
    off = {k for k, e in zip(keys, mech["entries"]) if not e["on"]}
    for e in mech["dose_only_entries"]:
        assert e["on"] is True
        k = (e["family"], e["declaration"], e["principle"], json.dumps(e["where"]))
        assert k in off, k
    assert len(HE.spec_sha256(spec)) == 64


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda s: s["hypotheses"][1]["parts"][0].update(rule="no_such_rule"),
            "unknown rule",
        ),
        (lambda s: s["hypotheses"][1]["parts"][0]["params"].pop("bound"), "needs"),
        (lambda s: s["hypotheses"][1]["parts"][0]["data"].pop("event"), "event"),
        (
            lambda s: s["hypotheses"][1]["parts"][0]["data"].update(where={"@nope": 1}),
            "unknown reference '@nope'",
        ),
        (
            lambda s: s["seed_blocks"]["twin_networks"].update(
                confirmatory=[15000, 15009]
            ),
            "seed block",
        ),
        (
            lambda s: s["seed_blocks"]["twin_networks"].update(development=[990, 1010]),
            "seed block",
        ),
        (lambda s: s["hypotheses"].pop(3), "missing Tier-A"),
        (lambda s: s["integrity_audit"].pop(), "integrity audit"),
        (
            lambda s: s["hypotheses"][2]["parts"].append(
                copy.deepcopy(s["hypotheses"][2]["parts"][0])
            ),
            "duplicate",
        ),
        (lambda s: s.update(schema="x"), "schema"),
        (
            lambda s: s["hypotheses"][1]["parts"][0]["data"].update(bogus=1),
            "unknown data",
        ),
        (
            lambda s: s["hypotheses"][1]["parts"][0]["data"]["union"][0][
                "where"
            ].update(x={"between": 1}),
            "operator",
        ),
        # a misspelt vocabulary or alias reference would select nothing in silence
        (
            lambda s: s["hypotheses"][1]["parts"][0]["data"]["union"][0][
                "where"
            ].update(design="@design.witnes"),
            "unknown reference",
        ),
        (
            lambda s: s["hypotheses"][1]["parts"][0]["data"]["union"][0].update(
                cell_label="{@null_knd}"
            ),
            "unknown reference",
        ),
        (lambda s: s["vocabulary"].update(null_kind="x"), "names used twice"),
    ],
)
def test_malformed_hypotheses_files_are_refused(mutate, match):
    spec = copy.deepcopy(HE.load_spec())
    mutate(spec)
    with pytest.raises(HE.SpecError, match=match):
        HE.validate_spec(spec)


def test_spec_hash_is_canonical():
    spec = HE.load_spec()
    again = json.loads(json.dumps(spec, sort_keys=False, indent=4))
    assert HE.spec_sha256(again) == HE.spec_sha256(spec)


# ==========================================================================
# the one name map between the v2 protocol and the estimators
# ==========================================================================
def test_protocol_names_map_to_the_estimator_names():
    from impact_pipeline.v2 import pdi_v3, ram_v3

    tpl = E.load_protocol(HE.SPEC_PATH.parent / "mpc_bench_v2_template.json")
    ram_block = dict(tpl.estimators["RAM"])
    facets = HE.ram_facets_for_estimator(ram_block["facets_not_applicable"])
    assert facets == dict(ram_v3.FACETS_NOT_APPLICABLE)
    assert HE.ram_facets_for_protocol(facets) == dict(
        ram_block["facets_not_applicable"]
    )
    pdi_block = dict(tpl.estimators["PDI"])
    params = pdi_v3.PDIParams.from_mapping(HE.pdi_params_for_estimator(pdi_block))
    assert params.bearer == pdi_v3.BEARER_NON_WORKSPACE == pdi_block["pdi_bearer"]
    assert set(HE.PROTOCOL_ESTIMATOR_NAMES) == {"RAM", "PDI"}
    with pytest.raises(ValueError):
        HE.ram_facets_for_estimator({"goal_alignment": "x"})
    with pytest.raises(ValueError):
        pdi_v3.PDIParams.from_mapping(
            HE.pdi_params_for_estimator({"pdi_bearer": "nope"})
        )


# ==========================================================================
# operating characteristics (CD-11) and the synthesis check
# ==========================================================================
# the lines of the synthesis check's log (scratch/SYNTH/synth_oc.log),
# verbatim; long lines are split by implicit string concatenation
SYNTH_OC_LOG = (
    "== A. seeds needed for 'demonstrated' bounds with 0 events",
    '  bound 0.07 level 0.0500 (m=1): n >= 42',
    '  bound 0.07 level 0.0250 (m=2): n >= 51',
    '  bound 0.07 level 0.0125 (m=4): n >= 61',
    '  bound 0.07 level 0.0083 (m=6): n >= 66',
    '  bound 0.02 level 0.0500 (ABSENT admission 0.02): n >= 149',
    '  bound 0.05 level 0.0500 (ABSENT admission 0.05): n >= 59',
    '  bound 0.01 level 0.0500 (0.01): n >= 299',
    '',
    (
        '== B. TOST quantiles and reachability at alpha_A = 0.01 (n_A '
        'fixed at |N_decl| = 5)'
    ),
    (
        '  df   9: q_P 1.833  q_A 2.821  q_A/2 (NAS union) 3.250  s_A = '
        'delta/q_A 0.0354  s_A(NAS) 0.0308'
    ),
    (
        '  df  12: q_P 1.782  q_A 2.681  q_A/2 (NAS union) 3.055  s_A = '
        'delta/q_A 0.0373  s_A(NAS) 0.0327'
    ),
    (
        '  df  19: q_P 1.729  q_A 2.539  q_A/2 (NAS union) 2.861  s_A = '
        'delta/q_A 0.0394  s_A(NAS) 0.0350'
    ),
    (
        '  df  49: q_P 1.677  q_A 2.405  q_A/2 (NAS union) 2.680  s_A = '
        'delta/q_A 0.0416  s_A(NAS) 0.0373'
    ),
    (
        '  df  69: q_P 1.667  q_A 2.382  q_A/2 (NAS union) 2.649  s_A = '
        'delta/q_A 0.0420  s_A(NAS) 0.0378'
    ),
    (
        '  df 199: q_P 1.653  q_A 2.345  q_A/2 (NAS union) 2.601  s_A = '
        'delta/q_A 0.0426  s_A(NAS) 0.0385'
    ),
    '  pi0 (TOST power at c = 0, normal approx):',
    '    NAS A under R, per direction         se_c 0.020 df   9: pi0 0.920',
    '    NAS A under R, per direction         se_c 0.025 df   9: pi0 0.547',
    '    NAS C contiguous, per direction      se_c 0.030 df   9: pi0 0.067',
    '    NAS C interleaved, per direction     se_c 0.023 df   9: pi0 0.728',
    '    IIM-dir bootstrap                    se_c 0.050 df  12: pi0 0.000',
    '    SRPI v1 (30 pairs)                   se_c 0.320 df   9: pi0 0.000',
    '    SRPI v3 lesion 120 pairs             se_c 0.021 df  54: pi0 0.982',
    '    PDI jackknife (non-concordant)       se_c 0.100 df   9: pi0 0.000',
    '',
    '== C. HR1 (interval calibration at the null), restated rule',
    (
        '  n_pool 900 n_cell 50 m 180 true tail 0.03: P(SUPPORTED) 1.000  '
        'P(FALSIFIED) 0.000'
    ),
    (
        '  n_pool 900 n_cell 50 m 180 true tail 0.05: P(SUPPORTED) 0.176  '
        'P(FALSIFIED) 0.000'
    ),
    (
        '  n_pool 900 n_cell 50 m 180 true tail 0.06: P(SUPPORTED) 0.000  '
        'P(FALSIFIED) 0.000'
    ),
    (
        '  n_pool 600 n_cell 50 m 200 true tail 0.03: P(SUPPORTED) 0.986  '
        'P(FALSIFIED) 0.000'
    ),
    (
        '  n_pool 600 n_cell 50 m 200 true tail 0.05: P(SUPPORTED) 0.011  '
        'P(FALSIFIED) 0.000'
    ),
    (
        '  n_pool 600 n_cell 50 m 200 true tail 0.06: P(SUPPORTED) 0.000  '
        'P(FALSIFIED) 0.000'
    ),
    (
        '  n_pool 1500 n_cell 100 m 200 true tail 0.03: P(SUPPORTED) 1.000 '
        ' P(FALSIFIED) 0.000'
    ),
    (
        '  n_pool 1500 n_cell 100 m 200 true tail 0.05: P(SUPPORTED) 0.843 '
        ' P(FALSIFIED) 0.000'
    ),
    (
        '  n_pool 1500 n_cell 100 m 200 true tail 0.06: P(SUPPORTED) 0.012 '
        ' P(FALSIFIED) 0.000'
    ),
    '',
    '== D. Twin SE calibration: pooled within-network df = N_net*(n_twin-1)',
    (
        '  10 networks x 7 twins (df 60): 90% factor [0.871, 1.179]; '
        'P(SUPPORTED|kappa=1) 0.983  P(FALSIFIED|1) 0.0001  '
        'P(FALSIFIED|1.5) 0.662  P(FALSIFIED|0.6) 0.931'
    ),
    (
        '  10 networks x 8 twins (df 70): 90% factor [0.879, 1.163]; '
        'P(SUPPORTED|kappa=1) 0.990  P(FALSIFIED|1) 0.0001  '
        'P(FALSIFIED|1.5) 0.713  P(FALSIFIED|0.6) 0.960'
    ),
    (
        '  8 networks x 6 twins (df 40): 90% factor [0.847, 1.228]; '
        'P(SUPPORTED|kappa=1) 0.940  P(FALSIFIED|1) 0.0006  '
        'P(FALSIFIED|1.5) 0.527  P(FALSIFIED|0.6) 0.796'
    ),
    (
        '  5 networks x 8 twins (df 35): 90% factor [0.838, 1.248]; '
        'P(SUPPORTED|kappa=1) 0.908  P(FALSIFIED|1) 0.0008  '
        'P(FALSIFIED|1.5) 0.488  P(FALSIFIED|0.6) 0.739'
    ),
    '',
    (
        "== E. Three-zone '>= 80 %' rule: SUPPORTED iff point >= 0.8; "
        'FALSIFIED iff CP upper(0.05/m) < 0.8'
    ),
    '  n 20 true 0.8: P(SUPPORTED) 0.630  P(FALSIFIED, m=1) 0.032  (m=4) 0.010',
    '  n 20 true 0.85: P(SUPPORTED) 0.830  P(FALSIFIED, m=1) 0.006  (m=4) 0.001',
    '  n 20 true 0.9: P(SUPPORTED) 0.957  P(FALSIFIED, m=1) 0.000  (m=4) 0.000',
    '  n 20 true 0.95: P(SUPPORTED) 0.997  P(FALSIFIED, m=1) 0.000  (m=4) 0.000',
    '  n 40 true 0.8: P(SUPPORTED) 0.593  P(FALSIFIED, m=1) 0.043  (m=4) 0.008',
    '  n 40 true 0.85: P(SUPPORTED) 0.865  P(FALSIFIED, m=1) 0.004  (m=4) 0.000',
    '  n 40 true 0.9: P(SUPPORTED) 0.985  P(FALSIFIED, m=1) 0.000  (m=4) 0.000',
    '  n 40 true 0.95: P(SUPPORTED) 1.000  P(FALSIFIED, m=1) 0.000  (m=4) 0.000',
    '  n 45 true 0.8: P(SUPPORTED) 0.588  P(FALSIFIED, m=1) 0.025  (m=4) 0.011',
    '  n 45 true 0.85: P(SUPPORTED) 0.873  P(FALSIFIED, m=1) 0.002  (m=4) 0.001',
    '  n 45 true 0.9: P(SUPPORTED) 0.988  P(FALSIFIED, m=1) 0.000  (m=4) 0.000',
    '  n 45 true 0.95: P(SUPPORTED) 1.000  P(FALSIFIED, m=1) 0.000  (m=4) 0.000',
    '',
    '== F. RAM-PE PRESENT on PC: SUPPORTED iff CP lower (0.05) > 0.5',
    '  n 20: needs >= 15/20; true 0.6: P(SUPPORTED) 0.126',
    '  n 20: needs >= 15/20; true 0.7: P(SUPPORTED) 0.416',
    '  n 20: needs >= 15/20; true 0.8: P(SUPPORTED) 0.804',
    '  n 40: needs >= 26/40; true 0.6: P(SUPPORTED) 0.317',
    '  n 40: needs >= 26/40; true 0.7: P(SUPPORTED) 0.807',
    '  n 40: needs >= 26/40; true 0.8: P(SUPPORTED) 0.992',
    '',
    '== G. Verdict specificity HC5v2: CP upper (0.05) at 0 events by seed clusters',
    '  20 clusters: 0.1391',
    '  40 clusters: 0.0722',
    '  42 clusters: 0.0688',
    '  45 clusters: 0.0644',
    '',
    '== H. H0 cell rule for rates near alpha (rank exceedance, H-IIM-1 style)',
    (
        '  n_cell 200 m 14 pooled clusters 1400: P(SUPPORTED|0.05) 0.929 '
        'P(FALSIFIED|0.05) 0.000'
    ),
    (
        '  n_cell 200 m 14 pooled clusters 2800: P(SUPPORTED|0.05) 0.996 '
        'P(FALSIFIED|0.05) 0.000'
    ),
    (
        '  n_cell 100 m 12 pooled clusters 1200: P(SUPPORTED|0.05) 0.885 '
        'P(FALSIFIED|0.05) 0.002'
    ),
)


@pytest.fixture(scope="module")
def oc():
    from scripts.v2 import operating_characteristics as OC

    return OC


def test_operating_characteristics_reproduce_the_synthesis_log(oc):
    rep = oc.reproduce_synth_oc()
    assert rep["lines"] == list(SYNTH_OC_LOG)


def test_exact_operating_characteristics(oc):
    # three-zone at a true rate of 0.9: P(SUPPORTED) 0.957 (20) and 0.988 (45);
    # at 0.8, P(FALSIFIED) <= 0.043 (design 4.9)
    assert round(oc.oc_three_zone(20, 0.9)["supported"], 3) == 0.957
    assert round(oc.oc_three_zone(45, 0.9)["supported"], 3) == 0.988
    assert oc.oc_three_zone(40, 0.8)["falsified"] <= 0.043 + 5e-4
    # RAM-PE PRESENT rule: P(SUPPORTED) 0.99 at 0.8 and 0.81 at 0.7 with 40 seeds
    assert round(oc.oc_rate_lower_bound(40, 0.8)["supported"], 2) == 0.99
    assert round(oc.oc_rate_lower_bound(40, 0.7)["supported"], 2) == 0.81
    # the analytic SE-calibration OC agrees with the simulated synthesis values
    k = oc.oc_kappa(60, 1.0)
    assert k["supported"] == pytest.approx(0.983, abs=0.002)
    assert oc.oc_kappa(60, 1.5)["falsified"] == pytest.approx(0.662, abs=0.005)
    assert oc.oc_kappa(60, 0.6)["falsified"] == pytest.approx(0.931, abs=0.005)


def test_cd11_resizes_seeds_and_never_thresholds(oc):
    rows = oc.evaluate_rates(
        [
            {
                "part": "a",
                "rule": "three_zone",
                "params": {"x": 0.8},
                "n": 20,
                "development_rate": 0.95,
            },
            {
                "part": "b",
                "rule": "three_zone",
                "params": {"x": 0.8},
                "n": 20,
                "development_rate": 0.84,
            },
            {
                "part": "c",
                "rule": "three_zone",
                "params": {"x": 0.8},
                "n": 20,
                "development_rate": 0.7,
            },
        ],
        n_max=120,
    )
    by = {r["part"]: r for r in rows}
    assert by["a"]["targets_met"] and by["a"]["resized_n"] is None
    assert not by["b"]["targets_met"] and by["b"]["resized_n"] > 20
    resized = oc.part_oc("three_zone", {"x": 0.8}, by["b"]["resized_n"], 0.84)
    assert resized["supported"] >= 0.8 and resized["falsified_if_correct"] <= 0.05
    assert by["c"]["indeterminate_capable"]


# ==========================================================================
# the synthetic bench (development seeds; no simulation)
# ==========================================================================
P5 = HE.PRINCIPLES
VERSION = {
    "RAM": "ram-v3-2026.10",
    "PDI": "pdi-v3-2026.10",
    "NAS": "nas-v3-2026.10",
    "IIM": "iim-v5-2026.10",
    "SRPI": "srpi-v2-2026.09",
}
SE_METHOD = {
    "RAM": "shift_null_sd",
    "PDI": "jackknife_contiguous_10",
    "NAS": "jackknife_contiguous_10",
    "IIM": "circular_block_bootstrap_10pct_B50",
    "SRPI": "jackknife_pairs_10",
}
SE_DF = {"RAM": 39.0, "PDI": 9.0, "NAS": 9.0, "IIM": 12.0, "SRPI": 9.0}
SE = 0.02
NA, OCC, CONC, MRD = "NA", "OCC", "CONC", "MRD"
TARGETS = {
    "W_RAM_no_plasticity": "RAM",
    "W_PDI_single_attractor": "PDI",
    "W_PDI_no_multistability": "PDI",
    "W_NAS_no_workspace": "NAS",
    "W_NAS_broadcast_only": "NAS",
    "W_IIM_feedforward": "IIM",
    "W_SRPI_no_efference": "SRPI",
}
OWN_LESION = {
    "RAM": "W_RAM_no_plasticity",
    "PDI": "W_PDI_single_attractor",
    "NAS": "W_NAS_no_workspace",
    "IIM": "W_IIM_feedforward",
    "SRPI": "W_SRPI_no_efference",
}
SIX = [
    "W_RAM_no_plasticity",
    "W_PDI_single_attractor",
    "W_NAS_no_workspace",
    "W_NAS_broadcast_only",
    "W_IIM_feedforward",
    "W_SRPI_no_efference",
]
SEEDS45 = list(range(100, 145))
SEEDS20 = list(range(100, 120))
SEEDS10 = list(range(100, 110))
G_LEVELS = [4 * k / 7 for k in range(8)]
G_NOM = G_LEVELS[2]
SEED_BLOCKS = {
    "family_b_rank_calibration": {"development": [400, 409]},
    "family_b_exact_tracking": {"development": [410, 419]},
    "family_b_feedforward_star": {"development": [420, 424]},
    "family_b_driver_conditioning": {"development": [430, 449]},
    "family_b_non_monotone": {"development": [450, 459]},
    "family_b_occupancy_gate": {"development": [460, 464]},
    "twin_networks": {"development": [820, 824]},
    "witnesses_20": {"development": [100, 119]},
}


def h64(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unif(*key):
    """A deterministic uniform (0, 1) number from a key."""
    return (int(h64("|".join(map(str, key)))[:12], 16) + 0.5) / 16**12


def gauss(*key):
    return float(stats.norm.ppf(unif(*key)))


def status_of(c, se, df):
    a = E.status_c(c, se, df)
    flags = [R.PRESENT_NOT_REACHABLE] if bool(a.present_not_reachable) else []
    return str(a.status.item()), a.reason.item(), flags


def comp(p, c, *, se=SE, df=None, details=None, extra_flags=()):
    df = SE_DF[p] if df is None else df
    d = {
        "principle": p,
        "estimator_version": VERSION[p],
        "details": dict(details or {}),
        "flags": list(extra_flags),
        "se_method": SE_METHOD[p],
        "n_null": 19,
    }
    if c == NA:
        d.update(status="UNDEFINED", reason=R.NOT_APPLICABLE_OBSERVATION_MODEL)
    elif c == OCC:
        d.update(status="UNDEFINED", reason=R.INSUFFICIENT_OCCUPANCY)
    elif c == MRD:
        d.update(status="UNDEFINED", reason=R.MACRO_RANK_DEFICIENT)
    elif c == CONC:
        d.update(
            status="ABSENT",
            reason=None,
            c=0.0,
            se=0.0,
            se_c=0.0,
            estimate=0.0,
            null_mean=0.0,
            se_method="concordant",
            flags=[R.ABSENT_BY_CONCORDANCE] + list(extra_flags),
        )
    else:
        st, reason, flags = status_of(c, se, df)
        d.update(
            status=st,
            reason=reason,
            c=c,
            se=se,
            se_c=se,
            se_df=df,
            df_c=df,
            estimate=c,
            null_mean=0.0,
            null_sd=0.1,
        )
        d["flags"] = flags + list(extra_flags)
        if p == "NAS":
            # both directions carry the component's value (c is the smaller
            # direction) and the per-direction SE and df
            d["c_R"] = d["c_B"] = c
            for key in ("te_in", "te_out"):
                d["details"][key] = {
                    **d["details"].get(key, {}),
                    "se_c": se,
                    "df_c": df,
                }
    return d


# The synthetic bench is written compactly; runner_format turns each record
# into the layout the v2 runner and the family-B validation script write,
# which is what the hypotheses file binds in 'vocabulary' and 'fields'.
DESIGN_NAMES = {
    "witnesses": "{f}_witnesses",
    "sweep": "{f}_sweeps",
    "factorial": "{f}_factorial",
    "adversarial": "A_adversaries",
    "held_out": "A_heldout",
    "twins": "{f}_twins",
    "anchor_replication": "{f}_anchors",
    "ram_only": "RAM160",
    "ram_only_twins": "RAM160_twins",
    "ram_only_anchor_replication": "RAM160_anchors",
}
FAMILY_NAMES = {"C": "C1"}
B_FORMS = {"primary": "directional", "iim_bidirectional": "bidirectional"}
RAM_CLASSES = {
    "adversarial_reflex_arc": "reflex_arc",
    "adversarial_scrambled_feedback": "scrambled_feedback",
}


def _runner_details(p, det):
    """Component details in the runner's layout: the estimator's details
    under ``estimator``, the status rule's under ``assessment``."""
    det = dict(det or {})
    out, est = {}, {}
    if p == "NAS":
        dirs, members = {}, {}
        for old, new in (("te_in", "receive"), ("te_out", "return")):
            d = det.pop(old, None) or {}
            sub = {k: d[k] for k in ("z", "excess") if k in d}
            if sub:
                dirs[new] = sub
            m = {k2: d[k1] for k1, k2 in (("se_c", "se"), ("df_c", "df")) if k1 in d}
            if m:
                members[new] = m
        if dirs:
            est["directions"] = dirs
        if members:
            out["assessment"] = {"members": members}
        r1 = (det.pop("descriptors", None) or {}).get("rank1")
        if r1:
            est["descriptors"] = {
                "pooled_rank1": {
                    "te_in_z": r1["te_in"]["z"],
                    "te_out_z": r1["te_out"]["z"],
                }
            }
    if p == "IIM" and "p_ind" in det:
        est["p_ind"] = det.pop("p_ind")
    if p == "PDI":
        if "n_states" in det:
            est["counts"] = [det.pop("n_states")] * 4
        fb = det.pop("full_bearer", None)
        if fb is not None:
            est["full_bearer"] = (
                {"counts": [fb["n_states"]] * 4} if "n_states" in fb else {}
            )
            if "ami_ignition_gate" in fb:
                est["ami_ignition_full_bearer"] = fb["ami_ignition_gate"]
    if "exact_value" in det:
        out["exact_target"] = det.pop("exact_value")
    out.update(det)
    if est:
        out["estimator"] = est
    return out


def runner_format(rec):
    """A synthetic record in the layout of the v2 runner (designs, family
    C1, ``config.params`` / ``config.tags``, component details), of the
    family-B validation script (``config.cell``, the cut mode as the form,
    the scoring's null order and label-error rate) and of the forward arms
    (``config.forward``, view and form names)."""
    rec = copy.deepcopy(rec)
    design, cfg = rec["design"], dict(rec.get("config") or {})
    fam = FAMILY_NAMES.get(rec["family"], rec["family"])
    rec["family"] = fam
    if design in DESIGN_NAMES:
        rec["design"] = DESIGN_NAMES[design].format(f=fam)
    if design == "family_b":
        net = {"xor": "xor_loop"}.get(cfg.get("network"), cfg.get("network"))
        cell = {"system": net, "n_time": cfg.get("T"), "params": {}, "expected": {}}
        for k in ("coupling", "noise"):
            if k in cfg:
                cell["params"][k] = cfg[k]
        if "condition" in cfg:
            cell["condition"] = cfg["condition"]
        if cfg.get("occupancy_cell"):
            cell["expected"]["occupancy"] = {"low": "undefined", "high": "defined"}[
                cfg["occupancy_cell"]
            ]
        rec["system"] = net
        rec["config"] = {"cell": cell}
        for sc in rec["scorings"]:
            form = sc["estimator_form"]
            order = (
                "project_then_shift"
                if form == "iim_residualise_then_shift"
                else "shift_then_project"
            )
            sc["estimator_form"] = B_FORMS.get(form, "directional")
            sc["details"] = {
                **(sc.get("details") or {}),
                "scoring": {
                    "null_order": order,
                    "label_error_q": float(cfg.get("label_error_q") or 0.0),
                },
            }
    elif design == "whole_brain":
        rec["config"] = {
            "forward": {
                "arm": "hopf",
                "condition_spec": {"G": cfg["G"], "lesion": cfg.get("lesion", "none")},
            }
        }
        for sc in rec["scorings"]:
            sc["view"] = {"eeg_low": "eeglow"}.get(sc["view"], sc["view"])
            if sc["estimator_form"] == "iim_v1_quadrant":
                sc["estimator_form"] = "iim_v1_quadrants"
    else:
        labels = ("sweep_knob", "sweep_level", "null_kind")
        tags = {k: cfg[k] for k in labels if k in cfg}
        params = {"variant": cfg["variant"]} if "variant" in cfg else {}
        for k in ("n_time", "n_nodes"):
            if k in cfg:
                rec["simulation"][k] = cfg[k]
        rec["config"] = {"params": params, "tags": tags}
        if design == "ram_only":
            rec["system"] = RAM_CLASSES.get(rec["system"], rec["system"])
    for sc in rec["scorings"]:
        for p, c in sc["components"].items():
            c["details"] = _runner_details(p, c.get("details"))
    return rec


def with_estimator(r, **values):
    """A row whose estimator details carry ``values``."""
    det = dict(r.get("details") or {})
    det["estimator"] = {**(det.get("estimator") or {}), **values}
    return {**r, "details": det}


class World:
    """The synthetic bench in which every prediction of the hypotheses file
    holds (records of schema /3 as dicts, plus the auxiliary sources)."""

    def __init__(self):
        self.records = []
        self.protocols = self._protocols()

    # -------------------------------------------------------------- protocols
    @staticmethod
    def _protocols():
        def anchors(statuses):
            n = [p for p in P5 if statuses.get(p) == "valid_specific"]
            return {
                "necessity_set": n,
                "principles": {p: {"status": statuses.get(p, "invalid")} for p in P5},
            }

        def prec(rows):
            return {
                "rows": [
                    {
                        "family": f,
                        "principle": p,
                        "witness": w,
                        "kind": k,
                        "pi": pi,
                        "decisive": pi >= 0.9,
                        "replacement": None if pi >= 0.9 else "replacement",
                        "predicted_absent_not_reachable_rate": 1 - pi,
                    }
                    for f, p, w, k, pi in rows
                ]
            }

        vs, vn = "valid_specific", "valid_nonspecific"
        a_r = {p: vs for p in P5}
        a_h = {"RAM": vs, "PDI": vs, "SRPI": vs, "NAS": vn, "IIM": vn}
        c_r = {"NAS": vs, "IIM": vs}
        c_h = {"NAS": vs, "IIM": vn}
        route = [
            {
                "principle": "PDI",
                "substrate": "synthetic_rate",
                "observation_stage": "source",
                "view": "source",
                "bearer": "non_workspace",
                "absent": True,
                "present": False,
            }
        ]
        out = {
            "A-R": {
                "anchors": anchors(a_r),
                "concordance_route": route,
                "precision": prec(
                    [
                        ("A-R", "IIM", "W_IIM_feedforward", "absent", 0.0),
                        ("A-R", "IIM", "PC_nominal", "present", 0.95),
                        ("A-R", "NAS", "W_NAS_no_workspace", "absent", 0.92),
                        ("A-R", "NAS", "W_NAS_broadcast_only", "absent", 0.91),
                        ("A-R", "PDI", "W_PDI_single_attractor", "absent", 0.97),
                        ("A-R", "RAM", "W_RAM_no_plasticity", "absent", 0.0),
                        ("A-R", "SRPI", "W_SRPI_no_efference", "absent", 0.0),
                    ]
                ),
            },
            "A-H": {
                "anchors": anchors(a_h),
                "precision": prec(
                    [
                        ("A-H", "RAM", "W_RAM_no_plasticity", "absent", 0.0),
                        ("A-H", "PDI", "W_PDI_single_attractor", "absent", 0.97),
                        ("A-H", "SRPI", "W_SRPI_no_efference", "absent", 0.0),
                    ]
                ),
            },
            "C1-R": {
                "anchors": anchors(c_r),
                "precision": prec(
                    [
                        ("C1-R", "IIM", "W_IIM_feedforward", "absent", 0.0),
                        ("C1-R", "IIM", "PC_nominal", "present", 0.95),
                        ("C1-R", "NAS", "W_NAS_no_workspace", "absent", 0.07),
                        ("C1-R", "NAS", "W_NAS_broadcast_only", "absent", 0.05),
                    ]
                ),
            },
            "C1-H": {
                "anchors": anchors(c_h),
                "precision": prec(
                    [
                        ("C1-H", "NAS", "W_NAS_no_workspace", "absent", 0.07),
                        ("C1-H", "NAS", "W_NAS_broadcast_only", "absent", 0.05),
                    ]
                ),
            },
            "A-RAM160": {"anchors": anchors({"RAM": vs})},
            # the null-calibration classification protocol: A-R's anchors
            # with the declaration none
            "A-none": {"anchors": anchors(a_r)},
            "B": {
                "precision": prec([("B", "IIM", "feedforward_star", "absent", 0.95)])
            },
            "fwd-eeg64": {"anchors": anchors({"PDI": vs})},
        }
        for pid, d in out.items():
            d["name"] = pid
            d["hash"] = h64(pid)
        return out

    def n_anch(self, pid):
        a = (self.protocols.get(pid) or {}).get("anchors")
        return None if a is None else a["necessity_set"]

    # -------------------------------------------------------------- records
    def task(
        self,
        task_id,
        design,
        family,
        system,
        seed,
        scorings,
        *,
        config=None,
        replicate=0,
        simulation=None,
    ):
        sc = []
        for s in scorings:
            pid = s["protocol_id"]
            comps = {}
            for c in s["components"]:
                comps[c["principle"]] = {
                    **c,
                    "declaration_id": s["declaration_id"],
                    "observation_stage": s["observation_stage"],
                    "protocol_id": pid,
                    "protocol_hash": h64(pid),
                    "replicate": replicate,
                }
            verdict = None
            n = self.n_anch(pid)
            if n and all(p in comps for p in n) and s.get("verdict", True):
                st = [comps[p]["status"] for p in n]
                v = (
                    "MPC_CONSISTENT"
                    if all(x == "PRESENT" for x in st)
                    else "EXCLUDED" if "ABSENT" in st else "UNDETERMINED"
                )
                verdict = {"verdict": v}
            sc.append(
                {
                    "scoring_id": s["scoring_id"],
                    "declaration_id": s["declaration_id"],
                    "observation_stage": s["observation_stage"],
                    "view": s["view"],
                    "estimator_form": s["estimator_form"],
                    "protocol_id": pid,
                    "protocol_hash": h64(pid),
                    "components": comps,
                    "verdict": verdict,
                    "details": {},
                }
            )
        self.records.append(
            {
                "schema": "mpc-bench-result/3",
                "task_id": task_id,
                "design": design,
                "family": family,
                "system": system,
                "seed": seed,
                "replicate": replicate,
                "split": "development",
                "generator_version": "mpc-bench-generators/2.0.0",
                "config": dict(config or {}),
                "scorings": sc,
                "status": "ok",
                "error": None,
                "simulation": {
                    "ts_sha256": h64("ts:" + task_id),
                    "raw_ts_sha256": None,
                    "structural_hash": h64(f"struct:{family}:{system}:{seed}"),
                    "schedule_hash": h64(f"sched:{family}:{system}:{seed}:{replicate}"),
                    "n_nodes": 16,
                    "n_time": 1000,
                    "dt": 0.05,
                    "seconds": 1.0,
                    **dict(simulation or {}),
                },
                "timing": {},
                "provenance": {},
            }
        )

    @staticmethod
    def scoring(
        decl, pid, comps, *, form="primary", stage="source", view="source", verdict=True
    ):
        return {
            "scoring_id": f"{decl}/{view}/{form}",
            "declaration_id": decl,
            "protocol_id": pid,
            "components": comps,
            "estimator_form": form,
            "observation_stage": stage,
            "view": view,
            "verdict": verdict,
        }

    # -------------------------------------------------------------- truths
    @staticmethod
    def witness_c(fam, system, decl, p, i):
        if fam == "C" and p in ("RAM", "PDI", "SRPI"):
            return NA
        if system in ("N_independent_noise", "N_ar1", "N_uncoupled"):
            return "NOISE"
        if system == "O_hypersynchronous":
            return OCC if p == "IIM" else 0.0
        if system == "O_inert":
            return 0.0
        if system == "N_modules_disconnected":
            # disconnecting the modules leaves RAM, PDI and SRPI on (CD-8)
            if p in ("RAM", "PDI", "SRPI"):
                return 1.0
            return (0.9 if decl == "H" and fam == "A" else 0.0) if p == "NAS" else 0.0
        if system == "PC_half":
            return 0.5
        c = 1.0
        if TARGETS.get(system) == p:
            c = 0.0
        if system == "W_NAS_broadcast_only" and p == "NAS":
            c = 0.13
        if system == "W_NAS_common_input_control" and p == "NAS":
            c = 0.01
        if system == "W_IIM_feedforward" and p == "NAS":
            c = 0.5
        if system == "W_PDI_single_attractor" and p == "NAS" and fam == "A":
            c = 0.6 if i % 5 < 3 else 1.0
        if decl == "H" and fam == "A" and system == "W_NAS_no_workspace" and p == "NAS":
            c = 0.09
        if decl == "H" and system == "W_IIM_feedforward" and p == "IIM":
            c = 1.0
        if (
            system in ("W_PDI_single_attractor", "W_PDI_no_multistability")
            and p == "PDI"
        ):
            return CONC
        return c

    def comp_for(
        self, p, c, key, *, noise=0.01, significant=None, rank1=None, se=SE, pdi=None
    ):
        if c == "NOISE":
            scale = 0.5 if p == "SRPI" else 1.0
            c = scale * SE * gauss("null", *key, p)
        elif isinstance(c, float):
            c = c + noise * gauss("noise", *key, p)
        details = {}
        if p == "NAS" and isinstance(c, float):
            sig = (c >= 0.3) if significant is None else significant
            r1 = sig if rank1 is None else rank1
            z, z1 = (5.0 if sig else 0.5), (5.0 if r1 else 0.5)
            details = {
                "te_in": {"z": z, "excess": c},
                "te_out": {"z": z, "excess": c},
                "descriptors": {"rank1": {"te_in": {"z": z1}, "te_out": {"z": z1}}},
            }
        if p == "IIM" and isinstance(c, float):
            details = {"p_ind": 0.5}
        if p == "PDI":
            details = dict(
                pdi
                or {
                    "n_states": 6,
                    "full_bearer": {"n_states": 6, "ami_ignition_gate": 0.8},
                }
            )
        return comp(p, c, se=se, details=details)

    # -------------------------------------------------------------- designs
    def witnesses(self):
        systems45 = ["PC_nominal"] + SIX
        systems20 = [
            "PC_half",
            "W_PDI_no_multistability",
            "W_NAS_common_input_control",
            "N_modules_disconnected",
            "N_independent_noise",
            "N_ar1",
            "O_hypersynchronous",
            "O_inert",
        ]
        for fam in ("A", "C"):
            sys20 = [
                s
                for s in systems20
                if not (
                    fam == "C"
                    and s
                    in (
                        "W_PDI_no_multistability",
                        "N_independent_noise",
                        "N_ar1",
                        "O_hypersynchronous",
                    )
                )
            ] + (["N_uncoupled"] if fam == "C" else [])
            for system in systems45 + sys20:
                seeds = SEEDS45 if system in systems45 else SEEDS20
                for i, seed in enumerate(seeds):
                    tid = f"w-{fam}-{system}-{seed}"
                    scs = []
                    for decl in ("R", "H"):
                        pid = f"{'A' if fam == 'A' else 'C1'}-{decl}"
                        comps = []
                        for p in P5:
                            c = self.witness_c(fam, system, decl, p, i)
                            pdi = None
                            if p == "PDI" and system == "W_PDI_single_attractor":
                                pdi = {
                                    "n_states": 1,
                                    "full_bearer": {
                                        "n_states": 2,
                                        "ami_ignition_gate": 0.8,
                                    },
                                }
                            comps.append(self.comp_for(p, c, (tid, decl), pdi=pdi))
                        scs.append(self.scoring(decl, pid, comps))
                        if system in ("W_NAS_no_workspace", "N_modules_disconnected"):
                            sec = (
                                1.0
                                if (decl == "H" and fam == "A")
                                else (
                                    0.0
                                    if system == "W_NAS_no_workspace"
                                    else self.witness_c(fam, system, decl, "NAS", i)
                                )
                            )
                            scs.append(
                                self.scoring(
                                    decl,
                                    pid,
                                    [
                                        self.comp_for(
                                            "NAS", float(sec), (tid, decl, "sec")
                                        )
                                    ],
                                    form="nas_secondary",
                                    verdict=False,
                                )
                            )
                        if (
                            system == "W_PDI_single_attractor"
                            and fam == "A"
                            and decl == "R"
                        ):
                            k = 2 if i % 2 == 0 else 1
                            scs.append(
                                self.scoring(
                                    decl,
                                    pid,
                                    [comp("PDI", 0.3, details={"n_states": k})],
                                    form="pdi_misdeclared_access",
                                    verdict=False,
                                )
                            )
                    if (
                        fam == "A"
                        and seed in SEEDS20
                        and system
                        in (
                            "PC_nominal",
                            "W_NAS_no_workspace",
                            "N_modules_disconnected",
                            "W_IIM_feedforward",
                            "O_inert",
                        )
                    ):
                        for decl, shift in (
                            ("P", 0.1),
                            ("Q10", 0.1),
                            ("Q25", 0.25),
                            ("J", 0.05),
                        ):
                            comps = []
                            for p in ("NAS", "IIM"):
                                base = self.witness_c(fam, system, "R", p, i)
                                base = 0.0 if not isinstance(base, float) else base
                                comps.append(
                                    self.comp_for(
                                        p, base + shift, (tid, decl), noise=0.005
                                    )
                                )
                            scs.append(self.scoring(decl, "A-R", comps, verdict=False))
                    self.task(tid, "witnesses", fam, system, seed, scs)

    def sweeps(self):
        levels = {
            "g_b": [round(x, 6) for x in np.linspace(0, 2, 10)],
            "c_int": [round(x, 6) for x in np.linspace(0, 1.2, 10)],
            "K": [2, 3, 4, 6, 8, 12],
        }
        for fam in ("A", "C"):
            for knob, lv in levels.items():
                if fam == "C" and knob == "K":
                    continue
                for level in lv:
                    for seed in SEEDS10:
                        tid = f"s-{fam}-{knob}-{level}-{seed}"
                        scs = []
                        for decl in ("R", "H"):
                            pid = f"{'A' if fam == 'A' else 'C1'}-{decl}"
                            comps = []
                            for p in P5:
                                c = 1.0
                                pdi = None
                                if fam == "C" and p in ("RAM", "PDI", "SRPI"):
                                    c = NA
                                elif knob == "g_b" and p == "NAS":
                                    c = 1.0 if decl == "H" else float(level)
                                elif knob == "c_int" and p == "IIM":
                                    c = level / 0.6
                                elif knob == "K" and p == "PDI":
                                    c = math.log2(level) / math.log2(6)
                                if p == "PDI" and c != NA:
                                    k = level if knob == "K" else 6
                                    pdi = {
                                        "n_states": k,
                                        "full_bearer": {"n_states": k},
                                    }
                                cc = self.comp_for(p, c, (tid, decl), pdi=pdi)
                                if p == "PDI" and c != NA:
                                    k = level if knob == "K" else 6
                                    cc["estimate"] = math.log2(k) - 0.05
                                comps.append(cc)
                            scs.append(self.scoring(decl, pid, comps))
                        self.task(
                            tid,
                            "sweep",
                            fam,
                            "PC_nominal",
                            seed,
                            scs,
                            config={"sweep_knob": knob, "sweep_level": level},
                        )

    def factorial(self):
        for fam in ("A", "C"):
            for bits in range(32):
                b = [int(x) for x in format(bits, "05b")]
                cid = "b" + "".join(map(str, b))
                for seed in SEEDS10:
                    tid = f"f-{fam}-{cid}-{seed}"
                    scs = []
                    for decl in ("R", "H"):
                        pid = f"{'A' if fam == 'A' else 'C1'}-{decl}"
                        comps = []
                        for j, p in enumerate(P5):
                            c = float(b[j])
                            if fam == "C" and p in ("RAM", "PDI", "SRPI"):
                                c = NA
                            if decl == "H" and fam == "A" and p == "NAS":
                                c = 1.0 if b[1] else float(b[2])
                            comps.append(self.comp_for(p, c, (tid, decl)))
                        scs.append(self.scoring(decl, pid, comps))
                    self.task(
                        tid, "factorial", fam, cid, seed, scs, config={"cell_id": cid}
                    )

    def adversaries(self):
        for variant in ("hierarchical", "uniform", "reversed"):
            for seed in SEEDS20:
                tid = f"a-{variant}-{seed}"
                scs = [
                    self.scoring(
                        d,
                        f"A-{d}",
                        [self.comp_for("NAS", c, (tid, d), noise=0.002)],
                        verdict=False,
                    )
                    for d, c in (("R", 0.0), ("H", 0.9))
                ]
                self.task(
                    tid,
                    "adversarial",
                    "A",
                    "ADV_NAS_staggered_driver",
                    seed,
                    scs,
                    config={"variant": variant},
                )
        for i, seed in enumerate(SEEDS20):
            tid = f"ho-tau10-{seed}"
            c = 0.5 if i < 15 else 0.0
            self.task(
                tid,
                "held_out",
                "A",
                "ADV_NAS_staggered_tau10",
                seed,
                [
                    self.scoring(
                        "R", "A-R", [self.comp_for("NAS", c, (tid,))], verdict=False
                    )
                ],
                config={"variant": "tau10"},
            )
            tid = f"ho-sat-{seed}"
            self.task(
                tid,
                "held_out",
                "A",
                "ADV_NAS_staggered_sat",
                seed,
                [
                    self.scoring(
                        "R",
                        "A-R",
                        [self.comp_for("NAS", 0.05, (tid,), noise=0.002)],
                        verdict=False,
                    )
                ],
                config={"variant": "saturating"},
            )

    def twins(self):
        z7, z8 = _z(7), _z(8)
        sets = {
            "A": [
                "PC_nominal",
                "PC_half",
                "W_NAS_no_workspace",
                "W_IIM_feedforward",
                "W_PDI_single_attractor",
            ],
            "C": ["PC_nominal", "W_NAS_no_workspace", "W_IIM_feedforward"],
        }
        for fam, systems in sets.items():
            for system in systems:
                for seed in range(820, 825):
                    off = 0.05 * (seed - 822)
                    for r in range(7):
                        tid = f"tw-{fam}-{system}-{seed}-r{r}"
                        scs = []
                        for decl in ("R", "H"):
                            pid = f"{'A' if fam == 'A' else 'C1'}-{decl}"
                            comps = []
                            for p in P5:
                                c = self.witness_c(fam, system, decl, p, seed - 820)
                                if isinstance(c, float):
                                    comps.append(
                                        self.comp_for(
                                            p, c + off + SE * z7[r], (), noise=0.0
                                        )
                                    )
                                else:
                                    comps.append(self.comp_for(p, c, (tid, decl)))
                            scs.append(self.scoring(decl, pid, comps))
                        self.task(tid, "twins", fam, system, seed, scs, replicate=r)
        for level in (0.0, 0.1, 0.3):
            for seed in range(820, 825):
                for r in range(8):
                    tid = f"rtw-{level}-{seed}-r{r}"
                    c = level / 0.3 + 0.1 * (seed - 822) + 0.05 * z8[r]
                    self.task(
                        tid,
                        "ram_only_twins",
                        "A",
                        "PC_nominal",
                        seed,
                        [
                            self.scoring(
                                "R",
                                "A-RAM160",
                                [self.comp_for("RAM", c, (), noise=0.0, se=0.05)],
                                verdict=False,
                            )
                        ],
                        replicate=r,
                        config={"sweep_knob": "eta", "sweep_level": level},
                    )

    def anchor_replication(self):
        nonspec = {("A", "H", "NAS"), ("A", "H", "IIM"), ("C", "H", "IIM")}
        for fam in ("A", "C"):
            systems = ["PC_nominal"] + list(OWN_LESION.values())
            for system in systems:
                for i, seed in enumerate(range(900, 920)):
                    tid = f"ar-{fam}-{system}-{seed}"
                    scs = []
                    for decl in ("R", "H"):
                        pid = f"{'A' if fam == 'A' else 'C1'}-{decl}"
                        comps = []
                        for p in P5:
                            if fam == "C" and p in ("RAM", "PDI", "SRPI"):
                                comps.append(comp(p, NA))
                                continue
                            mean = 1.0
                            if OWN_LESION[p] == system:
                                mean = 0.9 if (fam, decl, p) in nonspec else 0.0
                            x = mean + 0.1 * gauss("anchor", tid, decl, p)
                            cc = self.comp_for(p, x, (), noise=0.0)
                            comps.append(cc)
                        scs.append(self.scoring(decl, pid, comps, verdict=False))
                    self.task(tid, "anchor_replication", fam, system, seed, scs)
        for system, mean in (("PC_nominal", 1.0), ("W_RAM_no_plasticity", 0.0)):
            for seed in range(900, 920):
                tid = f"ar-ram-{system}-{seed}"
                x = mean + 0.1 * gauss("anchor", tid)
                self.task(
                    tid,
                    "ram_only_anchor_replication",
                    "A",
                    system,
                    seed,
                    [
                        self.scoring(
                            "R",
                            "A-RAM160",
                            [self.comp_for("RAM", x, (), noise=0.0, se=0.05)],
                            verdict=False,
                        )
                    ],
                )
        for seed in range(900, 920):
            tid = f"ar-fwd-{seed}"
            x = 2.5 + 0.1 * gauss("anchor", tid)
            self.task(
                tid,
                "forward_anchor_replication",
                "A",
                "PC_nominal",
                seed,
                [
                    self.scoring(
                        "R",
                        "fwd-eeg64",
                        [self.comp_for("PDI", x, (), noise=0.0)],
                        stage="sensor",
                        view="eeg64",
                        verdict=False,
                    )
                ],
            )

    def null_calibration(self):
        for kind in ("ar1", "pink", "surrogate_iid", "surrogate_linear"):
            for n_time in (1200, 2400):
                for n_nodes in (8, 16):
                    for seed in range(200, 210):
                        tid = f"nc-{kind}-{n_time}-{n_nodes}-{seed}"
                        comps = [self.comp_for(p, "NOISE", (tid,)) for p in P5]
                        # as the null-calibration design writes them: the
                        # classification protocol A-none (A-R with the
                        # declaration none), the kind in the task tags and
                        # the grid in the simulation block
                        self.task(
                            tid,
                            "null_calibration",
                            "null",
                            kind,
                            seed,
                            [self.scoring("none", "A-none", comps, verdict=False)],
                            config={
                                "tags": {
                                    "null_kind": kind,
                                    "n_time": n_time,
                                    "n_nodes": n_nodes,
                                    "cell": f"{kind}:T{n_time}:N{n_nodes}",
                                }
                            },
                            simulation={"n_time": n_time, "n_nodes": n_nodes},
                        )

    def family_b(self):
        def b_task(tid, seed, cfg, scorings):
            self.task(tid, "family_b", "B", cfg["network"], seed, scorings, config=cfg)

        def iim(c, *, p_ind=0.5, exact=None, noise=0.01, key=()):
            cc = self.comp_for("IIM", c, key, noise=noise)
            if isinstance(c, float):
                cc["details"] = {"p_ind": p_ind}
                if exact is not None:
                    cc["details"]["exact_value"] = exact
            return cc

        for cond, Ts, decl in (
            ("independent", (1000, 3000, 10000, 30000), "none"),
            ("stratified", (3000, 10000, 30000), "recorded"),
            ("residualised_continuous", (12000,), "recorded"),
            ("residualised_switching", (12000,), "recorded"),
        ):
            for T in Ts:
                for seed in range(400, 410):
                    tid = f"b-rank-{cond}-{T}-{seed}"
                    pind = 0.06 + 0.94 * unif(tid)
                    scs = [
                        self.scoring(
                            decl,
                            "B",
                            [iim(0.0, p_ind=pind, key=(tid, f))],
                            form=f,
                            verdict=False,
                        )
                        for f in (
                            "primary",
                            "iim_bidirectional",
                            "iim_residualise_then_shift",
                        )
                    ]
                    b_task(
                        tid,
                        seed,
                        {"network": "independent_units", "condition": cond, "T": T},
                        scs,
                    )
        for net, levels in (
            ("ring", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            ("xor", [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]),
        ):
            for lv in levels:
                exact = 0.2 * lv if net == "ring" else 0.5 - lv
                for T in (10000, 30000):
                    for seed in range(410, 420):
                        tid = f"b-exact-{net}-{lv}-{T}-{seed}"
                        cfg = {
                            "network": net,
                            "T": T,
                            ("coupling" if net == "ring" else "noise"): lv,
                        }
                        scs = [
                            self.scoring(
                                "none",
                                "B",
                                [
                                    iim(
                                        2 * exact,
                                        exact=exact,
                                        noise=0.002,
                                        key=(tid, f),
                                    )
                                ],
                                form=f,
                                verdict=False,
                            )
                            for f in ("primary", "iim_bidirectional")
                        ]
                        b_task(tid, seed, cfg, scs)
        for cpl in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            for T in (10000, 30000):
                for seed in range(420, 425):
                    tid = f"b-ff-{cpl}-{T}-{seed}"
                    scs = [
                        self.scoring(
                            "none", "B", [iim(0.0, key=(tid, 1))], verdict=False
                        ),
                        self.scoring(
                            "none",
                            "B",
                            [iim(cpl, key=(tid, 2))],
                            form="iim_bidirectional",
                            verdict=False,
                        ),
                    ]
                    b_task(
                        tid,
                        seed,
                        {"network": "feedforward_star", "coupling": cpl, "T": T},
                        scs,
                    )
        exact = {
            "primary": {0.45: 0.005, 0.9: -0.0097, 1.5: -0.02},
            "iim_bidirectional": {0.45: 0.0451, 0.9: 0.0362, 1.5: 0.02},
        }
        for cpl in (0.45, 0.9, 1.5):
            for seed in range(450, 460):
                tid = f"b-nonmono-{cpl}-{seed}"
                scs = [
                    self.scoring(
                        "none",
                        "B",
                        [
                            iim(
                                5 * exact[f][cpl] + 0.001 * (seed - 450),
                                exact=exact[f][cpl],
                                noise=0.0,
                            )
                        ],
                        form=f,
                        verdict=False,
                    )
                    for f in exact
                ]
                b_task(tid, seed, {"network": "ring", "coupling": cpl, "T": 30000}, scs)
        for seed in range(460, 465):
            tid = f"b-occ-low-{seed}"
            scs = [
                self.scoring("none", "B", [iim(OCC)], form=f, verdict=False)
                for f in ("primary", "iim_bidirectional")
            ]
            b_task(
                tid,
                seed,
                {
                    "network": "all_to_all",
                    "coupling": 0.6,
                    "T": 1000,
                    "occupancy_cell": "low",
                },
                scs,
            )
            tid = f"b-occ-high-{seed}"
            scs = [
                self.scoring(
                    "none", "B", [iim(0.1, key=(tid, f))], form=f, verdict=False
                )
                for f in ("primary", "iim_bidirectional")
            ]
            b_task(
                tid,
                seed,
                {
                    "network": "ring",
                    "coupling": 0.2,
                    "T": 30000,
                    "occupancy_cell": "high",
                },
                scs,
            )
            tid = f"b-montage-{seed}"
            scs = [
                self.scoring("none", "B", [comp("IIM", MRD)], form=f, verdict=False)
                for f in ("primary", "iim_bidirectional")
            ]
            b_task(tid, seed, {"network": "hopf_eeg_quadrants", "T": 0}, scs)
        hidden = {3000: 0.69, 10000: 0.91, 30000: 1.10}
        for T in (3000, 10000, 30000):
            for seed in range(430, 450):
                tid = f"b-hd-{T}-{seed}"
                scs = []
                for f in ("primary", "iim_bidirectional"):
                    scs.append(
                        self.scoring(
                            "recorded",
                            "B",
                            [iim(0.0, key=(tid, f))],
                            form=f,
                            verdict=False,
                        )
                    )
                    scs.append(
                        self.scoring(
                            "hidden",
                            "B",
                            [iim(hidden[T], key=(tid, f, 2))],
                            form=f,
                            verdict=False,
                        )
                    )
                b_task(tid, seed, {"network": "hidden_driver", "T": T}, scs)
                if T == 10000:
                    for q, c in ((0.1, 0.27), (0.25, 0.67)):
                        tq = f"b-hd-q{q}-{seed}"
                        scs = [
                            self.scoring(
                                "label_error",
                                "B",
                                [iim(c, key=(tq, f))],
                                form=f,
                                verdict=False,
                            )
                            for f in ("primary", "iim_bidirectional")
                        ]
                        b_task(
                            tq,
                            seed,
                            {"network": "hidden_driver", "T": T, "label_error_q": q},
                            scs,
                        )

    def hopf(self):
        views = (
            ("source", "source"),
            ("eeg64", "sensor"),
            ("eeg64_noref", "sensor"),
            ("eeg_low", "sensor"),
            ("mne_template", "source_estimate"),
            ("bold", "bold"),
        )
        for k, G in enumerate(G_LEVELS):
            n = 61 if k == 0 else (150 if k == 2 else 20)
            for seed in range(500, 500 + n):
                tid = f"h-{k}-{seed}"
                scs = []
                for view, stage in views:
                    if view == "bold":
                        c = OCC
                    elif view == "source":
                        c = float(G)
                    else:
                        c = 0.2 + 0.05 * G
                    scs.append(
                        self.scoring(
                            "none",
                            f"hopf-{view}",
                            [self.comp_for("IIM", c, (tid, view))],
                            stage=stage,
                            view=view,
                            verdict=False,
                        )
                    )
                if k == 0:
                    scs.append(
                        self.scoring(
                            "none",
                            "hopf-eeg64",
                            [comp("IIM", MRD)],
                            stage="sensor",
                            view="eeg64",
                            form="iim_v1_quadrant",
                            verdict=False,
                        )
                    )
                    scs.append(
                        self.scoring(
                            "none",
                            "hopf-eeg64_noref",
                            [self.comp_for("IIM", 1.0, (tid, "quad"))],
                            stage="sensor",
                            view="eeg64_noref",
                            form="iim_v1_quadrant",
                            verdict=False,
                        )
                    )
                self.task(
                    tid, "whole_brain", "hopf", "hopf", seed, scs, config={"G": G}
                )

    def ram_only(self):
        rows = [
            ("PC_nominal", None, None, 1.0),
            ("W_RAM_no_plasticity", None, None, 0.0),
            ("adversarial_reflex_arc", None, None, 0.0),
            ("adversarial_scrambled_feedback", None, None, 1.0),
            ("PC_nominal", "K", 1, 1.4),
            ("PC_nominal", "g_b", 0, 1.16),
        ]
        rows += [
            ("PC_nominal", "eta", lv, lv / 0.3) for lv in (0.0, 0.1, 0.2, 0.3, 0.6)
        ]
        for system, knob, level, c in rows:
            for seed in range(300, 340):
                tid = f"ro-{system}-{knob}-{level}-{seed}"
                cfg = {} if knob is None else {"sweep_knob": knob, "sweep_level": level}
                self.task(
                    tid,
                    "ram_only",
                    "A",
                    system,
                    seed,
                    [
                        self.scoring(
                            "R",
                            "A-RAM160",
                            [self.comp_for("RAM", float(c), (tid,), se=0.05)],
                            verdict=False,
                        )
                    ],
                    config=cfg,
                )

    def manipulation(self):
        rows = []
        for fam in ("A", "C1"):
            for sw in ("eta", "K", "g_b", "ff_only", "c_int", "e"):
                rows += [
                    {"family": fam, "switch": sw, "seed": s, "passed": True}
                    for s in range(600, 640)
                ]
        checks = [
            ("A", "N_modules_disconnected", None, "no_hub_periphery_path"),
            ("C1", "N_modules_disconnected", None, "no_hub_periphery_path"),
            ("C1", "N_uncoupled", None, "no_coupling"),
            ("A", "W_PDI_no_multistability", None, "no_ignition"),
            ("A", "ADV_NAS_staggered_tau10", "tau10", "driver_reaches_every_module"),
            ("A", "ADV_NAS_staggered_sat", "saturating", "driver_reaches_every_module"),
        ]
        checks += [
            ("A", "ADV_NAS_staggered_driver", v, "driver_reaches_every_module")
            for v in ("hierarchical", "uniform", "reversed")
        ]
        checks += [
            ("A", s, v, "stated_time_constants")
            for s, v in (
                ("ADV_NAS_staggered_driver", "hierarchical"),
                ("ADV_NAS_staggered_driver", "uniform"),
                ("ADV_NAS_staggered_driver", "reversed"),
                ("ADV_NAS_staggered_tau10", "tau10"),
                ("ADV_NAS_staggered_sat", "saturating"),
            )
        ]
        checks.append(("A", "PC_nominal", "slow_context_bold", "slow_context_dwell"))
        real = [
            {
                "family": f,
                "system_id": s,
                "variant": v,
                "check": c,
                "seed": seed,
                "passed": True,
            }
            for f, s, v, c in checks
            for seed in range(600, 640)
        ]
        # the reported checks (gate False): PC_half per family and switch, and
        # the twins of PC_nominal
        reported = [
            {"family": fam, "switch": sw, "system_id": "PC_half", "variant": sw,
             "check": "between_off_and_nominal", "passed": sw != "K", "gate": False}
            for fam in ("A", "C1")
            for sw in ("eta", "K", "g_b", "ff_only", "c_int", "e")
        ]
        reported += [
            {"family": "A", "system_id": "PC_nominal", "check": "twin_hashes",
             "seed": seed, "replicate": r, "passed": True, "gate": False}
            for seed in range(600, 640)
            for r in range(1, 7)
        ]
        return HE.manipulation_rows(rows + reported[:12], real + reported[12:])

    @staticmethod
    def registry():
        out = [
            {
                "principle": "NAS",
                "arm": "hopf",
                "substrate": sub,
                "view": v,
                "admitted_for_present": pr,
                "admitted_for_absent": ab,
                "anchor_valid": True,
            }
            for v, sub, pr, ab in (
                ("source", "stuart_landau", "yes", "yes"),
                ("eeg64", "eeg_like_forward", "no", "vacuous"),
                ("eeglow", "eeg_like_forward", "no", "vacuous"),
                ("mne_template", "eeg_like_forward", "no", "vacuous"),
            )
        ]
        out += [
            {
                "principle": "PDI",
                "arm": arm,
                "substrate": sub,
                "view": v,
                "admitted_for_present": pr,
                "admitted_for_absent": ab,
                "anchor_valid": True,
            }
            # the corrected HCv2-21 pattern: without an admitted forward
            # concordance cell ABSENT is unreachable on every view (vacuous)
            for v, arm, sub, pr, ab in (
                ("eeg64", "forward_a_eeg", "eeg_like_forward", "yes", "vacuous"),
                ("eeglow", "forward_a_eeg", "eeg_like_forward", "yes", "vacuous"),
                ("bold", "forward_a_bold", "bold_like_forward", "no", "vacuous"),
            )
        ]
        # an entry of another arm with the same principle and view is not
        # read by the forward family-A or Hopf parts
        out.append({"principle": "PDI", "arm": "hopf", "substrate": "bold_like_forward",
                    "view": "bold", "admitted_for_present": "yes",
                    "admitted_for_absent": "yes", "anchor_valid": True})
        return HE.registry_rows({"entries": out})

    def build(self):
        for fn in (
            self.witnesses,
            self.sweeps,
            self.factorial,
            self.adversaries,
            self.twins,
            self.anchor_replication,
            self.null_calibration,
            self.family_b,
            self.hopf,
            self.ram_only,
        ):
            fn()
        self.records = [runner_format(r) for r in self.records]
        return self


@pytest.fixture(scope="module")
def world():
    return World().build()


@pytest.fixture(scope="module")
def spec():
    return HE.load_spec()


def make_ctx(world):
    return HE.build_context(
        world.records,
        sources={"manipulation": world.manipulation(), "registry": world.registry()},
        protocols=world.protocols,
        seed_blocks=SEED_BLOCKS,
    )


@pytest.fixture(scope="module")
def evaluated(world, spec):
    ctx = make_ctx(world)
    return ctx, HE.evaluate(spec, ctx)


def test_the_synthetic_bench_follows_the_record_schema(world):
    step = max(1, len(world.records) // 400)
    for rec in world.records[::step]:
        REC.TaskRecord.from_dict(rec)
    assert {r["split"] for r in world.records} == {"development"}
    assert max(r["seed"] for r in world.records) <= 999


TIER_A = [f"HCv2-{i}" for i in range(25)]


@pytest.mark.parametrize("hid", TIER_A)
def test_every_specification_gives_its_predicted_outcome(evaluated, spec, hid):
    _, rep = evaluated
    h = next(h for h in rep["hypotheses"] if h["id"] == hid)
    want = next(x for x in spec["hypotheses"] if x["id"] == hid).get("prediction", SUP)
    detail = [
        (p["id"], p["outcome"], p.get("reason"), p.get("notes"))
        for p in h["parts"]
        if p["outcome"] not in (SUP, HE.REPORTED, NTBD)
    ]
    assert h["outcome"] == want, detail
    for p in h["parts"]:
        if p["role"] == "decisive":
            assert p["outcome"] == p["prediction"], (
                p["id"],
                p["outcome"],
                p.get("cells"),
            )


def test_the_synthetic_tally_and_usability(evaluated):
    _, rep = evaluated
    t = rep["tally"]["decisive_parts"]
    assert t[NE] == 0 and t[IND] == 0 and t[HE.REMOVED] == 0
    assert t[NTBD] == 1  # HCv2-14(c), predicted not testable
    assert rep["usability"]["A:g_b"] is True
    assert rep["uncalibrated_se_methods"] == []


# --------------------------------------------------------------------------
# targeted perturbations: each specification can fail
# --------------------------------------------------------------------------
def restatus(row, c):
    if row.get("se_method") == "concordant":
        row = {
            **row,
            "se_method": SE_METHOD[row["principle"]],
            "se_c": SE,
            "df_c": 9.0,
            "flags": [],
        }
    st, reason, flags = status_of(c, row.get("se_c") or SE, row.get("df_c") or 9.0)
    out = {
        **row,
        "c": c,
        "estimate": c,
        "status": st,
        "reason": reason,
        "flags": flags,
    }
    if row.get("principle") == "NAS":
        out["c_R"] = out["c_B"] = c
    return out


SPEC_FIELDS = HE.Fields(HE.load_spec().get("fields"), HE.load_spec().get("derived"))


def perturbed(ctx, source, where, update):
    new = copy.copy(ctx)
    new.__dict__.pop("_design_index", None)
    new.sources = dict(ctx.sources)
    new.usability = dict(ctx.usability)
    rows = []
    for r in ctx.sources[source]:
        if all(
            HE.evaluate_predicate({k: v}, r, SPEC_FIELDS) is True
            for k, v in where.items()
        ):
            r = update(dict(r))
        rows.append(r)
    new.sources[source] = rows
    return new


def outcome_of(spec, ctx, hid):
    h = next(x for x in spec["hypotheses"] if x["id"] == hid)
    return HE.evaluate_hypothesis(h, spec, ctx)["outcome"]


def set_c(c):
    return lambda r: restatus(r, c)


PERTURBATIONS = {
    "HCv2-0": (
        "manipulation",
        {"check_id": "g_b", "family": "A", "seed": {"max": 609}},
        lambda r: {**r, "passed": False},
        FAL,
    ),
    "HCv2-1": ("components", {"system": "N_ar1", "principle": "RAM"}, set_c(1.0), FAL),
    "HCv2-2": (
        "components",
        {
            "design": "family_b",
            "config.cell.condition": "independent",
            "config.cell.n_time": 1000,
        },
        lambda r: with_estimator(r, p_ind=0.01),
        FAL,
    ),
    "HCv2-3": (
        "components",
        {
            "principle": "SRPI",
            "family": ["null", "A"],
            "design": ["null_calibration", "A_witnesses"],
        },
        lambda r: restatus(r, 2 * r["c"]),
        SUP,
    ),
    "HCv2-4": (
        "components",
        {
            "design": "A_twins",
            "family": "A",
            "system": "PC_nominal",
            "principle": "NAS",
            "declaration_id": "R",
        },
        lambda r: restatus(r, r["c"] + 0.06 * (r["replicate"] - 3)),
        FAL,
    ),
    "HCv2-5": (
        "components",
        {
            "design": "A_witnesses",
            "system": "PC_nominal",
            "principle": "NAS",
            "family": "A",
            "declaration_id": "R",
            "seed": {"max": 114},
        },
        set_c(0.0),
        FAL,
    ),
    "HCv2-6": (
        "components",
        {
            "design": "A_anchors",
            "family": "A",
            "system": "W_PDI_single_attractor",
            "principle": "PDI",
            "declaration_id": "R",
        },
        lambda r: restatus(r, r["c"] + 1.0),
        FAL,
    ),
    "HCv2-7": (
        "components",
        {
            "design": ["A_witnesses", "C1_witnesses"],
            "system": "W_NAS_no_workspace",
            "principle": "NAS",
            "declaration_id": "R",
            "estimator_form": "primary",
        },
        set_c(1.0),
        FAL,
    ),
    "HCv2-8": (
        "components",
        {
            "system": "N_modules_disconnected",
            "principle": "NAS",
            "family": "A",
            "declaration_id": "R",
        },
        lambda r: with_estimator(
            r, directions={"receive": {"z": 5.0}, "return": {"z": 5.0}}
        ),
        FAL,
    ),
    "HCv2-9": (
        "components",
        {"declaration_id": "Q25"},
        lambda r: restatus(r, r["c"] - 0.5),
        FAL,
    ),
    "HCv2-10": (
        "registry",
        {"principle": "NAS", "view": "eeg64"},
        lambda r: {**r, "admitted_for_present": "yes"},
        FAL,
    ),
    "HCv2-11": (
        "components",
        {
            "design": "family_b",
            "config.cell.system": "feedforward_star",
            "config.cell.params.coupling": 0.4,
            "estimator_form": "directional",
        },
        set_c(1.0),
        FAL,
    ),
    "HCv2-12": (
        "components",
        {"config.cell.expected.occupancy": "undefined"},
        set_c(0.1),
        FAL,
    ),
    "HCv2-13": (
        "components",
        {"scoring_details.scoring.label_error_q": 0.25},
        set_c(0.0),
        FAL,
    ),
    "HCv2-14": (
        "components",
        {
            "design": ["A_witnesses", "C1_witnesses"],
            "system": "W_IIM_feedforward",
            "principle": "IIM",
            "declaration_id": "H",
        },
        set_c(0.0),
        FAL,
    ),
    "HCv2-15": (
        "components",
        {
            "design": "whole_brain",
            "config.forward.condition_spec.G": G_NOM,
            "view": "eeg64",
            "seed": {"max": 504},
        },
        set_c(0.0),
        FAL,
    ),
    "HCv2-16": (
        "components",
        {"design": "RAM160", "system": "W_RAM_no_plasticity"},
        set_c(1.0),
        FAL,
    ),
    "HCv2-17": (
        "components",
        {"design": "RAM160", "system": "scrambled_feedback"},
        set_c(0.0),
        FAL,
    ),
    "HCv2-18": ("components", {"system": "N_ar1", "principle": "PDI"}, set_c(1.0), FAL),
    "HCv2-19": (
        "components",
        {
            "design": "A_witnesses",
            "system": "PC_nominal",
            "principle": "PDI",
            "family": "A",
        },
        set_c(0.0),
        FAL,
    ),
    "HCv2-20": (
        "components",
        {
            "system": "W_PDI_single_attractor",
            "principle": "PDI",
            "estimator_form": "primary",
        },
        lambda r: with_estimator(
            r, full_bearer={"counts": [3] * 4}, ami_ignition_full_bearer=0.8
        ),
        FAL,
    ),
    "HCv2-21": (
        "registry",
        {"principle": "PDI", "view": "bold"},
        lambda r: {**r, "admitted_for_absent": "yes"},
        FAL,
    ),
    "HCv2-23": (
        "verdicts",
        {"system": "W_SRPI_no_efference", "family": "A", "declaration_id": "R"},
        lambda r: {**r, "verdict_value": "MPC_CONSISTENT"},
        FAL,
    ),
    "HCv2-24": (
        "components",
        {
            "design": "A_factorial",
            "family": "A",
            "declaration_id": "R",
            "principle": "NAS",
            "own_bit": 0,
        },
        set_c(1.0),
        FAL,
    ),
}


@pytest.mark.parametrize(
    "hid", sorted(PERTURBATIONS, key=lambda h: int(h.split("-")[1]))
)
def test_a_targeted_change_flips_each_specification(evaluated, spec, hid):
    ctx, rep = evaluated
    source, where, update, want = PERTURBATIONS[hid]
    assert rep["outcomes"][hid] != want
    assert outcome_of(spec, perturbed(ctx, source, where, update), hid) == want


def test_hcv2_22_supported_when_the_predicted_failures_do_not_occur(evaluated, spec):
    ctx, rep = evaluated
    assert rep["outcomes"]["HCv2-22"] == FAL
    new = perturbed(
        ctx,
        "components",
        {
            "design": ["A_witnesses", "C1_witnesses"],
            "system": "W_NAS_broadcast_only",
            "principle": "NAS",
        },
        set_c(0.0),
    )
    new = perturbed(
        new,
        "components",
        {
            "design": ["A_witnesses", "C1_witnesses"],
            "system": "W_PDI_single_attractor",
            "principle": "NAS",
        },
        set_c(1.0),
    )
    assert outcome_of(spec, new, "HCv2-22") == SUP


def test_reversion_reports_hcv2_5_both_ways(world, spec):
    # an anti-conservative NAS SE (twin spread four times the SE) triggers the
    # reversion rule: every NAS ABSENT of that SE method becomes
    # UNDEFINED(SE_NOT_CALIBRATED) for the verdict-level hypotheses
    ctx = make_ctx(world)
    ctx = perturbed(
        ctx,
        "components",
        {"design": ["A_twins", "C1_twins"], "principle": "NAS"},
        lambda r: restatus(r, r["c"] + 0.08 * (r["replicate"] - 3)),
    )
    rep = HE.evaluate(spec, ctx)
    assert ["nas-v3-2026.10", "jackknife_contiguous_10"] in rep[
        "uncalibrated_se_methods"
    ]
    h5 = next(h for h in rep["hypotheses"] if h["id"] == "HCv2-5")
    assert h5["outcome_as_recorded"] == SUP and h5["reversion"]


# ==========================================================================
# the evaluator script
# ==========================================================================
def test_evaluator_script_on_development_records(world, tmp_path):
    import scripts.bench_hypotheses_v2 as BH

    rec_file = tmp_path / "records.jsonl"
    keep = [
        r
        for r in world.records
        if r["design"] in ("A_witnesses", "C1_witnesses", "RAM160")
    ]
    rec_file.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
    man = tmp_path / "switches.json"
    man.write_text(
        json.dumps(
            [
                {"family": f, "switch": s, "seed": i, "passed": True}
                for f in ("A", "C1")
                for s in ("g_b", "ff_only", "c_int", "K", "eta", "e")
                for i in range(600, 640)
            ]
        ),
        encoding="utf-8",
    )
    protos = tmp_path / "protocols"
    protos.mkdir()
    rep = BH.run(
        tmp_path / "out", records=[rec_file], manipulation=[man], development=True
    )
    assert rep["status"] == HE.DEVELOPMENT_FLAG and rep["n_records"] == len(keep)
    assert rep["outcomes"]["HCv2-17"] == SUP  # the RAM-only arm needs no protocol
    assert rep["outcomes"]["HCv2-7"] in (SUP, NE)
    for name in (
        "hypotheses_v2.json",
        "hypotheses_v2_parts.csv",
        "hypotheses_v2_tally.csv",
    ):
        assert (tmp_path / "out" / name).is_file()
    saved = json.loads((tmp_path / "out" / "hypotheses_v2.json").read_text())
    assert saved["outcomes"] == rep["outcomes"]
    assert saved["inputs"]["records"] and saved["spec_sha256"] == HE.spec_sha256(
        HE.load_spec()
    )


def test_evaluator_script_refusals(world, tmp_path):
    import scripts.bench_hypotheses_v2 as BH
    from scripts.v2 import integrity_audit as IA

    recs = world.records[:20]
    rec_file = tmp_path / "records.jsonl"
    rec_file.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    with pytest.raises(BH.EvaluationRefused, match="integrity audit"):
        BH.run(tmp_path / "o1", records=[rec_file])
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"ok": False, "checks": [{"id": "IA-5", "status": "FAIL"}]})
    )
    with pytest.raises(BH.EvaluationRefused, match="IA-5"):
        BH.run(tmp_path / "o2", records=[rec_file], audit=audit)
    good = {
        "ok": True,
        "checks": [],
        "excluded_task_ids": [],
        "not_run": [],
        "records_sha256": IA.records_digest(recs),
    }
    audit.write_text(json.dumps({**good, "not_run": ["IA-6"]}))
    with pytest.raises(BH.EvaluationRefused, match="not run"):
        BH.run(tmp_path / "o2b", records=[rec_file], audit=audit)
    audit.write_text(json.dumps({**good, "records_sha256": "0" * 64}))
    with pytest.raises(BH.EvaluationRefused, match="other records"):
        BH.run(tmp_path / "o2c", records=[rec_file], audit=audit)
    audit.write_text(json.dumps(good))
    # the hypotheses file must be the one frozen at the tag
    with pytest.raises(BH.EvaluationRefused, match="tag"):
        BH.check_frozen_hypotheses(None, "no-such-freeze-tag")
    # development records are refused by a confirmatory evaluation
    with pytest.raises(BH.EvaluationRefused, match="seed policy"):
        BH.run(tmp_path / "o3", records=[rec_file], audit=audit, check_code=False)
    broken = dict(world.records[0], status="error")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(BH.EvaluationRefused, match="schema"):
        BH.run(tmp_path / "o4", records=[bad], development=True)
    # an excluded record is not judged
    audit.write_text(
        json.dumps({"ok": True, "checks": [], "excluded_task_ids": [broken["task_id"]]})
    )
    rep = BH.run(tmp_path / "o5", records=[bad], audit=audit, development=True)
    assert rep["n_records"] == 0


# ==========================================================================
# edge cases of the rules and the per-row prerequisite M
# ==========================================================================
def test_jonckheere_terpstra_with_few_tied_values_does_not_raise():
    # two tied values: no ordering evidence (zero variance), p = 1
    out = HE.jonckheere_terpstra([[1.0], [1.0]])
    assert out["method"] == "normal" and out["p"] == 1.0
    # JT = 1.5, mean 1, tie-corrected variance 30/72 + 4/48 = 0.5
    out = HE.jonckheere_terpstra([[1.0], [1.0, 2.0]])
    assert out["jt"] == 1.5
    assert out["p"] == pytest.approx(stats.norm.sf(0.5 / math.sqrt(0.5)))


def test_stepwise_sign_needs_the_median_change_in_the_exact_direction():
    def rows(n_down):
        out = []
        for seed in range(40):
            for lv, exact, v in (
                (0.45, 0.0451, 0.5),
                (0.9, 0.0362, 0.49 if seed < n_down else 0.5),
            ):
                out.append(
                    {
                        "_cell": "x",
                        "_group": lv,
                        "_exact": exact,
                        "_value": v,
                        "_defined": True,
                        "seed": seed,
                        "_cluster": seed,
                        "_pool": "x",
                    }
                )
        return HE.Prepared(out)

    p = {"steps": [[0.45, 0.9]], "alpha": 0.025, "pair_on": ["seed"]}
    # 10 of 40 pairs fall, 30 do not change: the sign test is significant
    # (p = 2^-10) but the median change is 0, not the exact sign
    r = HE.RULES["stepwise_sign"](rows(10), p)
    step = r.cells[0]["steps"][0]
    assert step["p"] < 0.025 and step["median_change"] == 0.0
    assert r.outcome == FAL
    r = HE.RULES["stepwise_sign"](rows(40), p)
    assert r.outcome == SUP and r.cells[0]["steps"][0]["median_change"] < 0


def test_anchor_replication_without_own_lesion_runs_is_not_evaluable():
    rows = [
        {
            "_cell": "A-R|RAM",
            "principle": "RAM",
            "_protocol": "A-R",
            "_predicted_anchor": "valid_specific",
            "member": "pc",
            "seed": s,
            "_value": 1.0 + 0.01 * (s % 5),
            "_cluster": s,
            "_pool": "RAM",
        }
        for s in range(20)
    ]
    r = HE.RULES["anchor_replicates"](HE.Prepared(rows), {"block_size": 20})
    assert r.outcome == NE and "own-lesion" in r.cells[0]["reason"]
    # the forward views check validity only and need no lesion
    r = HE.RULES["anchor_replicates"](
        HE.Prepared(rows), {"block_size": 20, "validity_only": True}
    )
    assert r.outcome == SUP and r.cells[0]["observed"] == "valid"


def _oracle_ctx(rows, usability):
    ctx = HE.build_context(sources={"components": rows})
    ctx.usability = dict(usability)
    return ctx


def test_prerequisite_m_drops_only_the_rows_of_an_unrealised_system():
    rows = [
        comp_row(f"{s}{i}", s, "ABSENT", 0.0, seed=i, principle=p)
        for s, p in (("W_NAS_no_workspace", "NAS"), ("W_IIM_feedforward", "IIM"))
        for i in range(20)
    ]
    part = {
        "id": "p-oracle-rows",
        "rule": "three_zone",
        "params": {"x": 0.8},
        "requires": {"oracle": True},
        "data": {
            "union": [
                {"label": "nw", "where": {"system": "W_NAS_no_workspace"}},
                {"label": "ff", "where": {"system": "W_IIM_feedforward"}},
            ],
            "cell_label": "{member}",
            "event": {"status": "ABSENT"},
        },
    }
    spec = mini_spec([part])
    ctx = _oracle_ctx(rows, {("A", "g_b"): False, ("A", "c_int"): True})
    out = HE.evaluate_part(part, spec, ctx)
    cells = {c["cell"]: c for c in out["cells"]}
    assert out["outcome"] == SUP
    assert cells["ff"]["outcome"] == SUP
    assert cells["nw"]["outcome"] == NE and cells["nw"]["checks"] == ["A:g_b"]
    assert cells["nw"]["reason"].startswith("ORACLE")
    # every switch usable: both cells are judged
    ctx = _oracle_ctx(rows, {("A", "g_b"): True, ("A", "c_int"): True})
    out = HE.evaluate_part(part, spec, ctx)
    assert {c["cell"]: c["outcome"] for c in out["cells"]} == {"nw": SUP, "ff": SUP}


def test_prerequisite_m_is_not_named_when_the_selection_is_empty():
    # a selection without rows (here: no record of the system) has no input
    # rows; ORACLE is the reason only when the checks dropped the rows
    rows = [
        comp_row(f"nw{i}", "W_NAS_no_workspace", "ABSENT", 0.0, seed=i, principle="NAS")
        for i in range(20)
    ]
    part = {
        "id": "p-oracle-empty",
        "rule": "three_zone",
        "params": {"x": 0.8},
        "requires": {"oracle": True},
        "data": {
            "where": {"system": "W_IIM_feedforward"},
            "event": {"status": "ABSENT"},
        },
    }
    spec = mini_spec([part])
    for usable in (False, True):
        ctx = _oracle_ctx(rows, {("A", "g_b"): usable, ("A", "c_int"): usable})
        out = HE.evaluate_part(part, spec, ctx)
        assert out["outcome"] == NE
        assert out["reason"] == "no input rows"
    part["data"]["where"] = {"system": "W_NAS_no_workspace"}
    out = HE.evaluate_part(part, spec, _oracle_ctx(rows, {("A", "g_b"): False}))
    assert out["outcome"] == NE and out["reason"].startswith("ORACLE")
    assert "20 rows dropped" in out["notes"][-1]


def test_prerequisite_m_checks_both_systems_of_a_pair_and_variants():
    rows = [
        comp_row(f"{s}{i}", s, "PRESENT", c, seed=i, principle="IIM")
        for s, c in (("PC_nominal", 1.0), ("W_IIM_feedforward", 0.0))
        for i in range(20)
    ]
    part = {
        "id": "p-oracle-pair",
        "rule": "three_zone",
        "params": {"x": 0.8},
        "requires": {"oracle": True},
        "data": {
            "pair": {
                "on": ["seed", "principle"],
                "a": {"system": "PC_nominal"},
                "b": {"system": "W_IIM_feedforward"},
            },
            "event": {"delta": {"ge": 0.5}},
        },
    }
    spec = mini_spec([part])
    out = HE.evaluate_part(part, spec, _oracle_ctx(rows, {("A", "c_int"): False}))
    assert out["outcome"] == NE and out["reason"].startswith("ORACLE")
    out = HE.evaluate_part(part, spec, _oracle_ctx(rows, {("A", "c_int"): True}))
    assert out["outcome"] == SUP
    # check ids are rendered per row: the variant of a staggered driver
    adv = [
        comp_row(
            f"{v}{i}",
            "ADV_NAS_staggered_driver",
            "UNDEFINED",
            0.0,
            seed=i,
            design="A_adversaries",
            config={"params": {"variant": v}},
        )
        for v in ("hierarchical", "uniform")
        for i in range(20)
    ]
    part = {
        "id": "p-oracle-variant",
        "rule": "three_zone",
        "params": {"x": 0.8},
        "requires": {"oracle": True},
        "data": {"event": {"status": {"ne": "PRESENT"}}, "cells": ["@variant"]},
    }
    spec = mini_spec([part])
    ok = {
        ("A", f"ADV_NAS_staggered_driver:hierarchical/{c}"): True
        for c in ("driver_reaches_every_module", "stated_time_constants")
    }
    ok[("A", "ADV_NAS_staggered_driver:uniform/driver_reaches_every_module")] = True
    out = HE.evaluate_part(part, spec, _oracle_ctx(adv, ok))
    cells = {c["cell"]: c for c in out["cells"]}
    assert cells["variant=hierarchical"]["outcome"] == SUP
    assert cells["variant=uniform"]["outcome"] == NE
    assert cells["variant=uniform"]["checks"] == [
        "A:ADV_NAS_staggered_driver:uniform/stated_time_constants"
    ]


SLOW = "PC_nominal:slow_context_bold/slow_context_dwell"


def test_prerequisite_m_gates_the_slow_context_arm_only(spec):
    """The slow-context check is required by the rows of the forward BOLD
    arm (its records, anchor replications and registry entries) and by no
    other row, and is looked up in family A for registry entries."""
    (cond,) = spec["oracle_checks"]["conditions"]
    assert cond["checks"] == [SLOW]
    fwd = {"forward": {"arm": "forward_a_bold", "condition": "PC_nominal"}}
    eeg = {"forward": {"arm": "forward_a_eeg", "condition": "PC_nominal"}}
    rows = [
        comp_row(f"{lab}{i}", "PC_nominal", "PRESENT", 1.0, seed=i, principle="PDI",
                 design=design, config=cfg, view=view)
        for lab, design, cfg, view in (
            ("bold", "forward_family_a_bold", fwd, "bold"),
            ("anch", "forward_anchor_replication", fwd, "bold"),
            ("eeg", "forward_family_a", eeg, "eeg64"),
            ("wit", "witnesses", {}, None),
        )
        for i in range(20)
    ]
    part = {
        "id": "p-oracle-condition",
        "rule": "three_zone",
        "params": {"x": 0.8},
        "requires": {"oracle": True},
        "data": {"event": {"status": "PRESENT"}, "cells": ["design"]},
    }
    mspec = mini_spec([part], oracle_checks=spec["oracle_checks"])
    out = HE.evaluate_part(part, mspec, _oracle_ctx(rows, {("A", SLOW): False}))
    cells = {c["cell"]: c for c in out["cells"]}
    for d in ("forward_family_a_bold", "forward_anchor_replication"):
        assert cells[f"design={d}"]["outcome"] == NE
        assert cells[f"design={d}"]["checks"] == [f"A:{SLOW}"]
    assert cells["design=forward_family_a"]["outcome"] == SUP
    assert cells["design=witnesses"]["outcome"] == SUP
    out = HE.evaluate_part(part, mspec, _oracle_ctx(rows, {("A", SLOW): True}))
    assert {c["outcome"] for c in out["cells"]} == {SUP}
    # registry entries carry no family: the condition names family A
    reg = HE.registry_rows(
        {
            "entries": [
                {"principle": "PDI", "arm": arm, "substrate": "x", "view": v,
                 "admitted_for_present": "yes", "admitted_for_absent": "vacuous",
                 "anchor_valid": True}
                for v, arm in (("eeg64", "forward_a_eeg"), ("eeglow", "forward_a_eeg"),
                               ("bold", "forward_a_bold"))
            ]
        }
    )
    h21 = spec_part(spec, "HCv2-21")
    assert h21["requires"] == {"oracle": True}
    for usable, bold in ((True, SUP), (False, NE)):
        ctx = HE.build_context(sources={"components": [], "registry": reg})
        ctx.usability = {("A", SLOW): usable}
        out = HE.evaluate_part(h21, spec, ctx)
        cells = {c["cell"]: c for c in out["cells"]}
        assert out["outcome"] == SUP and cells["eeg64"]["outcome"] == SUP
        assert cells["bold"]["outcome"] == bold
        # an emptied cell is listed once, with the reason ORACLE
        assert [c["cell"] for c in out["cells"]].count("bold") == 1
        if not usable:
            assert cells["bold"]["reason"].startswith("ORACLE")
            assert cells["bold"]["checks"] == [f"A:{SLOW}"]
    assert spec_part(spec, "HCv2-6(b)")["requires"] == {"oracle": True}


def test_oracle_conditions_are_validated():
    spec = copy.deepcopy(HE.load_spec())
    good = copy.deepcopy(spec["oracle_checks"]["conditions"][0])
    for bad, msg in (
        ({**good, "systems": []}, "a condition takes"),
        ({**good, "where": {}}, "needs a 'where'"),
        ({**good, "checks": []}, "list of check ids"),
        ({**good, "checks": "x/y"}, "list of check ids"),
        ({**good, "family": 1}, "family label"),
        ({**good, "where": {"arm": {"bogus": 1}}}, "unknown operators"),
        ({**good, "family": "@no_such_family"}, "unknown reference"),
    ):
        spec["oracle_checks"]["conditions"] = [bad]
        with pytest.raises(HE.SpecError, match=msg):
            HE.validate_spec(spec)
    spec["oracle_checks"]["conditions"] = good
    with pytest.raises(HE.SpecError, match="is a list"):
        HE.validate_spec(spec)


def test_manipulation_rows_leave_out_summaries_and_keep_reported_checks_apart():
    switches = [
        {"family": "A", "switch": "g_b", "seed": 600, "passed": "True"},
        # a usability summary row and a per-seed PC_half value row: no result
        {"kind": "switch", "family": "A", "switch": "g_b", "n": 40, "passes": 40,
         "usable": "True", "system_id": "", "check": ""},
        {"family": "A", "seed": 600, "switch": "g_b", "nominal": 0.03, "half": 0.02,
         "off": 0.0},
        # the reported PC_half row (names its switch and its check)
        {"family": "A", "switch": "K", "system_id": "PC_half", "variant": "K",
         "check": "between_off_and_nominal", "between": "False", "passed": "False",
         "gate": "False"},
    ]
    real = [
        {"family": "A", "system_id": "PC_nominal", "variant": "slow_context_bold",
         "check": "slow_context_dwell", "seed": 600, "passed": "True", "gate": "True"},
        {"family": "A", "system_id": "PC_nominal", "check": "twin_hashes",
         "seed": 600, "replicate": 1, "passed": True, "gate": False},
        # PC_half has twins too: its twin rows are not PC_half rows of (c)
        {"family": "A", "system_id": "PC_half", "check": "twin_hashes",
         "seed": 600, "replicate": 1, "passed": True, "gate": False},
    ]
    rows = HE.manipulation_rows(switches, real)
    got = [(r["kind"], r["check_id"], r["passed"]) for r in rows]
    assert got == [
        ("switch", "g_b", True),
        ("reported", "PC_half:K/between_off_and_nominal", False),
        ("new_system", SLOW, True),
        ("reported", "PC_nominal/twin_hashes", True),
        ("reported", "PC_half/twin_hashes", True),
    ]
    # reported rows never decide usability
    ctx = HE.build_context(sources={"manipulation": rows})
    rep = HE.evaluate(HE.load_spec(), ctx)
    assert rep["usability"] == {"A:g_b": True, f"A:{SLOW}": True}
    h0 = next(h for h in rep["hypotheses"] if h["id"] == "HCv2-0")
    parts = {p["id"]: p for p in h0["parts"]}
    assert parts["HCv2-0(c)"]["role"] == "reported"
    assert parts["HCv2-0(c)"]["outcome"] == HE.REPORTED
    (cell,) = parts["HCv2-0(c)"]["cells"]
    assert cell["cell"] == "family=A|check_id=PC_half:K/between_off_and_nominal"
    assert cell["k"] == 0 and cell["n"] == 1
    assert [c["cell"] for c in parts["HCv2-0(d)"]["cells"]] == [
        "family=A|system_id=PC_nominal", "family=A|system_id=PC_half"]
    assert all(c["k"] == 1 for c in parts["HCv2-0(d)"]["cells"])


def test_a_check_row_with_a_blank_result_counts_as_not_passed():
    """Only a row without a ``passed`` field is left out: a check row whose
    result is blank (a CSV cell left empty) still counts, as a failure, so
    a missing result can never raise a pass rate."""
    rows = HE.manipulation_rows(
        [{"family": "A", "switch": "g_b", "seed": s, "passed": p}
         for s, p in ((600, "True"), (601, ""), (602, None))],
        [{"family": "A", "system_id": "PC_nominal", "variant": "slow_context_bold",
          "check": "slow_context_dwell", "seed": 600, "passed": "", "gate": "True"}],
    )
    got = [(r["kind"], r["check_id"], r["seed"], r["passed"]) for r in rows]
    assert got == [
        ("switch", "g_b", 600, True),
        ("switch", "g_b", 601, False),
        ("switch", "g_b", 602, False),
        ("new_system", SLOW, 600, False),
    ]


def test_the_evaluator_reads_the_oracle_directory_without_its_summaries(tmp_path):
    """Every table the oracle item writes is read (the evaluator is given the
    directory's CSV files): the usability summary and the per-seed PC_half
    values are no check results and add no row, so a check passed in 40 of
    40 seeds counts 40 rows, not 41."""
    import csv

    import scripts.bench_hypotheses_v2 as BH

    def write(name, rows):
        path = tmp_path / f"manipulation_{name}.csv"
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write("switches", [{"switch": "g_b", "seed": s, "passed": True, "family": "A"}
                       for s in range(320, 360)])
    write("realisation", [{"system_id": "PC_nominal", "family": "A", "seed": s,
                           "variant": "slow_context_bold",
                           "check": "slow_context_dwell", "value": 31.0,
                           "passed": s != 320, "gate": True}
                          for s in range(320, 360)])
    write("summary", [{"kind": "switch", "family": "A", "switch": "g_b", "n": 40,
                       "passes": 40, "usable": True, "system_id": "", "check": ""}])
    write("pc_half", [{"family": "A", "seed": 320, "switch": "g_b", "nominal": 0.03,
                       "half": 0.02, "off": 0.0}])
    write("pc_half_summary", [{"family": "A", "switch": "g_b", "between": True,
                               "gate": False, "system_id": "PC_half",
                               "variant": "g_b", "check": "between_off_and_nominal",
                               "passed": True}])
    write("twins", [{"system_id": "PC_nominal", "family": "A", "seed": 320,
                     "replicate": 1, "check": "twin_hashes", "passed": True,
                     "gate": False}])
    rows = BH.manipulation_source([str(p) for p in sorted(tmp_path.glob("*.csv"))])
    kinds = {}
    for r in rows:
        kinds.setdefault((r["kind"], r["check_id"]), []).append(r["passed"])
    assert {k: (sum(v), len(v)) for k, v in kinds.items()} == {
        ("switch", "g_b"): (40, 40),
        ("new_system", SLOW): (39, 40),
        ("reported", "PC_half:g_b/between_off_and_nominal"): (1, 1),
        ("reported", "PC_nominal/twin_hashes"): (1, 1),
    }


def test_the_synthetic_prerequisite_m_reports_pc_half_and_twins(evaluated):
    _, rep = evaluated
    h0 = next(h for h in rep["hypotheses"] if h["id"] == "HCv2-0")
    parts = {p["id"]: p for p in h0["parts"]}
    assert rep["usability"][f"A:{SLOW}"] is True
    assert parts["HCv2-0(b)"]["outcome"] == SUP
    assert any(c["check"] == SLOW for c in parts["HCv2-0(b)"]["cells"])
    pc = {c["cell"]: c for c in parts["HCv2-0(c)"]["cells"]}
    assert len(pc) == 12 and all(c["outcome"] == HE.REPORTED for c in pc.values())
    assert pc["family=A|check_id=PC_half:K/between_off_and_nominal"]["k"] == 0
    (tw,) = parts["HCv2-0(d)"]["cells"]
    assert tw["n"] == 240 and tw["k"] == 240
    # the reported parts are not counted with the decisive ones
    assert rep["tally"]["reported_parts"] >= 2


def test_requirements_are_validated():
    spec = copy.deepcopy(HE.load_spec())
    part = spec["hypotheses"][1]["parts"][0]
    part["requires"] = {"usable": "g_b"}
    with pytest.raises(HE.SpecError, match="list of check ids"):
        HE.validate_spec(spec)
    part["requires"] = {"oracle": True, "switch": ["g_b"]}
    with pytest.raises(HE.SpecError, match="requires takes"):
        HE.validate_spec(spec)
    part["requires"] = {"oracle": True}
    spec["oracle_checks"]["systems"]["W_RAM_no_plasticity"] = ["{@no_such_alias}/x"]
    with pytest.raises(HE.SpecError, match="unknown reference"):
        HE.validate_spec(spec)


def test_auxiliary_inputs_must_share_the_split_of_the_records():
    man = HE.manipulation_rows(
        [{"family": "A", "switch": "g_b", "seed": 600, "passed": True}]
    )
    with pytest.raises(S.SeedPolicyError):
        HE.build_context(sources={"manipulation": man}, split="confirmatory")
    conf = [comp_row("t", "W", "ABSENT", 0.0, seed=20000)]
    with pytest.raises(S.SeedPolicyError):
        HE.build_context(sources={"components": conf, "manipulation": man})
    ctx = HE.build_context(sources={"manipulation": man})
    assert ctx.split == "development"


def test_nas_interval_calibration_is_judged_per_direction():
    # two independent, exactly calibrated directions at the null: each is
    # calibrated, but their minimum (the component c) has a lower tail of
    # about 1 - 0.95^2 = 0.0975 > 0.075
    n = 800
    z_r = stats.norm.ppf((np.arange(n) + 0.5) / n)
    z_b = np.random.default_rng(0).permutation(z_r)
    rows = []
    for i in range(n):
        cr, cb = 0.02 * z_r[i], 0.02 * z_b[i]
        rows.append(
            comp_row(
                f"n{i}",
                "ar1",
                "UNDEFINED",
                min(cr, cb),
                seed=i,
                design="null_calibration",
                decl="none",
                family="null",
                protocol="A-none",
                c_R=cr,
                c_B=cb,
                se_c=0.02,
                df_c=None,
                details={
                    "assessment": {
                        "members": {
                            "receive": {"se": 0.02, "df": None},
                            "return": {"se": 0.02, "df": None},
                        }
                    }
                },
            )
        )
    proto = {
        "A-none": {
            "necessity_set": ["NAS"],
            "anchors": {"principles": {"NAS": {"status": "valid_specific"}}},
        }
    }
    spec = HE.load_spec()
    part = next(
        p
        for h in spec["hypotheses"]
        for p in h.get("parts") or ()
        if p["id"] == "HCv2-3"
    )
    ctx = HE.build_context(sources={"components": rows}, protocols=proto)
    out = HE.evaluate_part(part, spec, ctx)
    cells = {c["cell"]: c for c in out["cells"]}
    assert set(cells) == {"NAS:receive", "NAS:return"}
    for c in cells.values():
        assert c["outcome"] == SUP and c["prediction"] == SUP
        assert c["kappa"] == pytest.approx(1.0, abs=0.01)
        assert c["tails"]["lower"]["rate"] == pytest.approx(0.05, abs=0.002)
    # the smaller direction alone would not be judged calibrated
    mins = HE.Prepared(
        [
            {"_cell": "NAS", "_value": r["c"], "se_c": 0.02, "_cluster": i}
            for i, r in enumerate(rows)
        ]
    )
    r = HE.RULES["kappa_null"](mins, {})
    assert r.outcome != SUP and r.cells[0]["tails"]["lower"]["rate"] > 0.075


def test_the_twenty_seed_parts_read_the_twenty_seed_set(evaluated):
    _, rep = evaluated
    h9 = next(h for h in rep["hypotheses"] if h["id"] == "HCv2-9")
    jt = next(p for p in h9["parts"] if p["id"] == "HCv2-9(a)")
    assert jt["cells"] and all(c["sizes"] == [20, 20, 20] for c in jt["cells"])
    h3 = next(h for h in rep["hypotheses"] if h["id"] == "HCv2-3")
    cells = {c["cell"] for c in h3["parts"][0]["cells"]}
    assert {"NAS:receive", "NAS:return", "SRPI"} <= cells and "NAS" not in cells


def test_evaluator_script_refuses_unfrozen_or_cross_split_inputs(world, tmp_path):
    import scripts.bench_hypotheses_v2 as BH
    from scripts.v2 import integrity_audit as IA

    # confirmatory look-alike records (never simulated: a copy of synthetic
    # development records with the seed moved and the freeze tag set)
    recs = []
    for i, r in enumerate(world.records[:5]):
        recs.append(
            {
                **r,
                "seed": 20000 + i,
                "split": "confirmatory",
                "provenance": {"code": {"freeze_tag": "mpcbench-freeze-v2"}},
            }
        )
    rec_file = tmp_path / "records.jsonl"
    rec_file.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "ok": True,
                "checks": [],
                "excluded_task_ids": [],
                "not_run": [],
                "records_sha256": IA.records_digest(recs),
            }
        )
    )
    mech = tmp_path / "mechanism_on.json"
    mech.write_text(json.dumps([{"where": {"design": "witnesses"}, "on": True}]))
    with pytest.raises(BH.EvaluationRefused, match="mechanism-on"):
        BH.run(
            tmp_path / "m1",
            records=[rec_file],
            audit=audit,
            mechanism_on=mech,
            check_code=False,
        )
    # development manipulation checks in a confirmatory evaluation
    man = tmp_path / "switches.json"
    man.write_text(
        json.dumps([{"family": "A", "switch": "g_b", "seed": 600, "passed": True}])
    )
    with pytest.raises(BH.EvaluationRefused, match="auxiliary inputs"):
        BH.run(
            tmp_path / "m2",
            records=[rec_file],
            audit=audit,
            manipulation=[man],
            check_code=False,
        )
    # anchors, gates and routes come from the frozen family protocols
    with pytest.raises(BH.EvaluationRefused, match="frozen protocols"):
        BH.run(tmp_path / "m3", records=[rec_file], audit=audit, check_code=False)


def test_operating_characteristics_of_a_part_with_several_cells(oc):
    one = oc.oc_three_zone(20, 0.9, m=3)
    part = oc.part_oc("three_zone", {"x": 0.8}, 20, 0.9, n_cells=[20, 20, 20])
    # SUPPORTED needs every cell; FALSIFIED any cell; m = 3 cells as in the
    # evaluator
    assert part["supported"] == pytest.approx(one["supported"] ** 3)
    fal = oc.oc_three_zone(20, 0.8, m=3)["falsified"]
    assert part["falsified_if_correct"] == pytest.approx(1 - (1 - fal) ** 3)
    assert (
        part["supported"] < oc.part_oc("three_zone", {"x": 0.8}, 20, 0.9)["supported"]
    )
    rows = oc.evaluate_rates(
        [
            {
                "part": "k",
                "rule": "three_zone",
                "params": {"x": 0.8},
                "n": 20,
                "development_rate": 0.9,
                "n_cells": [20] * 8,
            }
        ],
        n_max=120,
    )
    # one cell at 0.9 meets the target at 20 seeds (0.957); eight cells do not
    # (0.70), and the part is resized per cell, never re-thresholded
    assert not rows[0]["targets_met"] and rows[0]["resized_n"] > 20
    again = oc.part_oc(
        "three_zone",
        {"x": 0.8},
        rows[0]["resized_n"],
        0.9,
        n_cells=[rows[0]["resized_n"]] * 8,
    )
    assert again["supported"] >= 0.8 and again["falsified_if_correct"] <= 0.05


# ==========================================================================
# the record names of the file are those the v2 runner and the family-B
# validation script write
# ==========================================================================
# vocabulary values bound to records that no merged producer writes yet
# (none: the forward arms and the null-calibration generator run through the
# v2 runner)
PENDING_VOCABULARY: set = set()


def _values(v):
    return list(v) if isinstance(v, list) else [v]


def test_vocabulary_names_the_designs_forms_and_views_of_the_producers():
    from impact_pipeline.bench import designs_v2 as D
    from impact_pipeline.bench import forward_v2 as F2
    from impact_pipeline.bench.designs_v2 import family_b as FB
    from impact_pipeline.bench.designs_v2 import forward as FW
    from impact_pipeline.bench.designs_v2 import null_calibration as NCV

    spec = HE.load_spec()
    vocab = spec["vocabulary"]
    runner = D.designs()
    assert NCV.DESIGN in runner and vocab["design.null_calibration"] == NCV.DESIGN
    designs = set(runner) | {FB.DESIGN}
    families = {d.family for d in runner.values()} | {FB.FAMILY}
    forms = set(D.ESTIMATOR_FORMS) | set(FB.CUT_MODES) | {FW.IIM_V1_QUADRANTS}
    views = {v for vs in FW.VIEWS_OF_ARM.values() for v in vs} | {D.SOURCE_VIEW}
    assert set(views) <= set(F2.VIEWS) | {D.SOURCE_VIEW}
    for key, val in vocab.items():
        if key in PENDING_VOCABULARY:
            continue
        kind = key.split(".")[0]
        want = {"design": designs, "family": families, "form": forms,
                "view": views, "arm": set(FW.ARMS)}.get(kind)
        if want is not None:
            assert set(_values(val)) <= want, (key, val)
    assert vocab["family.C1"] == "C1"
    # the forward arms: one table of record designs (the forward design
    # module), which the registry builder reads and the hypotheses select by
    assert vocab["design.whole_brain"] == FW.RECORD_DESIGN_OF_ARM[FW.ARM_HOPF]
    assert _values(vocab["design.forward"]) == [
        FW.RECORD_DESIGN_OF_ARM[FW.ARM_A_EEG], FW.RECORD_DESIGN_OF_ARM[FW.ARM_A_BOLD]]
    assert vocab["design.forward_anchor_replication"] == FW.ANCHOR_REPLICATION_DESIGN
    assert {vocab["design.whole_brain"], *_values(vocab["design.forward"])} == set(
        FW.ADMISSION_RECORD_DESIGNS)
    assert vocab["arm.hopf"] == FW.ARM_HOPF
    assert _values(vocab["arm.forward_family_a"]) == [FW.ARM_A_EEG, FW.ARM_A_BOLD]
    assert not any(k.startswith("substrate.") for k in vocab)
    # the registry parts select the entries of their arms
    for hid, principle, arm in (("HCv2-10", "NAS", "@arm.hopf"),
                                ("HCv2-21", "PDI", "@arm.forward_family_a")):
        h = next(h for h in spec["hypotheses"] if h["id"] == hid)
        for part in h["parts"]:
            assert part["data"]["source"] == "registry"
            assert part["data"]["where"] == {"principle": principle, "arm": arm}
    # the family-B seed blocks are the blocks the family-B design runs
    blocks = {
        "family_b_rank_calibration": "HCv2-2",
        "family_b_feedforward_star": "HCv2-11",
        "family_b_exact_tracking": "HCv2-12(a)",
        "family_b_non_monotone": "HCv2-12(b)",
        "family_b_occupancy_gate": "HCv2-12(c, d)",
        "family_b_driver_conditioning": "HCv2-13",
    }
    for name, block in blocks.items():
        for split in ("development", "confirmatory"):
            assert tuple(spec["seed_blocks"][name][split]) == tuple(
                FB.SEED_BLOCKS[block][split]
            ), (name, split)


def _runner_record(task, protocols, settings=None):
    from impact_pipeline.bench import run_bench_v2 as RB

    rec = RB.run_task(task, protocols, settings or RB.DEFAULT_SETTINGS)
    assert rec.status in (REC.TASK_OK, REC.TASK_OK_WITH_COMPONENT_ERRORS), rec.error
    return rec.to_dict()


def _anchored(keys, estimators=None):
    """Draft protocols with external anchors (and estimator options)."""
    from impact_pipeline.bench import run_bench_v2 as RB

    ref = {"kind": "external", "scale": "excess",
           "values": {"RAM": 1.0, "PDI": 1.0, "IIM": 1.0, "SRPI": 1.0,
                      "NAS:receive": 1.0, "NAS:return": 1.0}}
    out = {}
    for k in keys:
        d = RB.draft_protocol(k).to_dict()
        d["reference"] = ref
        for p, opts in (estimators or {}).items():
            d["estimators"][p].update(opts)
        out[k] = RB.ResolvedProtocol(k, E.ProtocolV3.from_dict(d), "override")
    return out


def test_estimator_fields_resolve_on_runner_records():
    import dataclasses

    from impact_pipeline.bench import designs_v2 as D
    from impact_pipeline.bench.designs_v2 import family_a as FA

    spec = HE.load_spec()
    fields = HE.Fields(spec["fields"], spec["derived"])
    # NAS and PDI (a small PDI null, wiring only) on the witness whose PDI
    # partitions and oracle windows the runner keeps
    task = FA.witnesses(
        D.DEVELOPMENT, seeds=[0], systems=["W_PDI_single_attractor"], forms=()
    )[0]
    task = dataclasses.replace(
        task, scorings=D.make_scorings({"R": "A-R"}, ("NAS", "PDI"))
    )
    protos = _anchored(["A-R"], {"PDI": {"n_null": 2, "n_init": 1}})
    rows = HE.component_rows([_runner_record(task, protos)])
    nas = next(r for r in rows if r["principle"] == "NAS")
    pdi = next(r for r in rows if r["principle"] == "PDI")
    for name in (
        "nas_z_receive",
        "nas_z_return",
        "nas_excess_receive",
        "nas_excess_return",
        "nas_rank1_z_receive",
        "nas_rank1_z_return",
        "nas_conditioning_delta",
        "nas_conditioning_delta_return",
        "nas_c_receive",
        "nas_c_return",
        "nas_se_c_receive",
        "nas_se_c_return",
        "nas_df_c_receive",
        "nas_df_c_return",
    ):
        assert HE._finite(fields.get(nas, "@" + name)) is not None, name
    assert fields.get(nas, "@nas_significant") in (True, False)
    assert fields.get(nas, "@nas_rank1_significant") in (True, False)
    for name in ("pdi_k_full", "pdi_k_content", "pdi_ami_ignition"):
        assert HE._finite(fields.get(pdi, "@" + name)) is not None, name
    # IIM v5's rank p-value (a small null and bootstrap, wiring only)
    task = dataclasses.replace(task, scorings=D.make_scorings({"R": "A-R"}, ("IIM",)))
    protos = _anchored(["A-R"], {"IIM": {"n_null": 19, "bootstrap_replicates": 2}})
    iim = HE.component_rows([_runner_record(task, protos)])[0]
    assert 0 < HE._finite(fields.get(iim, "@iim_p_ind")) <= 1


def test_configuration_fields_resolve_on_runner_records():
    from impact_pipeline.bench import designs_v2 as D
    from impact_pipeline.bench import run_bench_v2 as RB

    spec = HE.load_spec()
    fields = HE.Fields(spec["fields"], spec["derived"])
    # the simulations only: the configuration fields need no estimator
    only_ram = RB.RunSettings(principles=("RAM",))

    def record(design, system, variant=None):
        task = next(
            t
            for t in D.get_design(design).tasks(D.DEVELOPMENT)
            if t.system == system and t.params.get("variant") == variant
        )
        keys = {s.protocol_key for s in task.scorings}
        rec = _runner_record(task, _anchored(keys), only_ram)
        return {**rec, "details": {}}

    adv = record("A_adversaries", "ADV_NAS_staggered_driver", "hierarchical")
    assert fields.get(adv, "@variant") == "hierarchical"
    assert adv["design"] in spec["vocabulary"]["design.adversarial"]
    sweep = record("A_sweeps", "sweep_K_l03")
    assert (fields.get(sweep, "@sweep_knob"), fields.get(sweep, "@sweep_level")) == (
        "K",
        6.0,
    )
    cell = record("A_factorial", "b01011")
    assert fields.get(cell, "@cell_id") == "b01011"
    assert [fields.get(cell, f"bit.{p}") for p in HE.PRINCIPLES] == [0, 1, 0, 1, 1]
    eta0 = record("RAM160", "eta_0")
    assert (fields.get(eta0, "@sweep_knob"), fields.get(eta0, "@sweep_level")) == (
        "eta",
        0.0,
    )
    pc = record("RAM160", "PC_nominal")
    assert fields.get(pc, "@no_sweep") is True
    assert fields.get(pc, "@n_time") == pc["simulation"]["n_time"] > 0
    assert fields.get(pc, "@n_nodes") == 30
    # one configuration under several designs carries one identity
    digest = "system_config_sha256"
    assert pc["config"][digest] != eta0["config"][digest]


def test_family_b_fields_resolve_on_validation_records():
    from impact_pipeline.bench.designs_v2 import family_b as FB
    from scripts.v2 import iim_validation_v2 as IV

    spec = HE.load_spec()
    fields = HE.Fields(spec["fields"], spec["derived"])
    by_id = {}
    for c in FB.cells():
        by_id.setdefault(c.cell_id, c)
    want = {
        "independent": ("HCv2-2", ("i",)),
        "stratified": ("HCv2-2", ("ii",)),
    }
    conds = {c.condition for c in by_id.values() if c.hypothesis == "HCv2-2"}
    assert conds == {
        "independent",
        "stratified",
        "residualised_continuous",
        "residualised_switching",
    }
    for cond, (hyp, parts) in want.items():
        assert any(
            c.condition == cond and c.hypothesis == hyp and c.parts == parts
            for c in by_id.values()
        )
    star = next(c for c in by_id.values() if c.system == "feedforward_star")
    row = {"config": {"cell": star.to_dict()}}
    assert fields.get(row, "@b_network") == "feedforward_star"
    assert fields.get(row, "@b_coupling") == star.param_dict["coupling"]
    assert fields.get(row, "@b_T") == star.n_time
    occ = {
        c.to_dict()["expected"].get("occupancy")
        for c in by_id.values()
        if c.hypothesis == "HCv2-12" and "c" in c.parts
    }
    assert occ <= {"undefined", "defined"} and occ
    # one cheap task scored end to end: the scoring and component fields
    task = next(
        t
        for t in FB.tasks(D_SPLIT, hypotheses=("HCv2-2",))
        if t.cell.condition == "independent" and t.cell.n_time == 1000
    )
    rows = HE.component_rows([IV.score_task(task).to_dict()])
    assert {r["estimator_form"] for r in rows} == set(FB.CUT_MODES)
    for r in rows:
        assert fields.get(r, "@b_condition") == "independent"
        assert fields.get(r, "@b_null_order") == "shift_then_project"
        assert fields.get(r, "@b_label_q") == 0.0
        assert HE._finite(fields.get(r, "@b_exact")) is not None
        vocab = spec["vocabulary"]
        forms = vocab["form.primary"] + vocab["form.iim_bidirectional"]
        assert r["estimator_form"] in forms


D_SPLIT = S.DEVELOPMENT


def test_the_pdi_masking_window_holds_the_v1_window_levels():
    # v1 counted 0 bits at g_b 0.67-1.56: the sweep levels 2/3 to 14/9
    from impact_pipeline.bench import designs_v2 as D

    spec = HE.load_spec()
    part = next(
        p
        for h in spec["hypotheses"]
        for p in h.get("parts") or ()
        if p["id"] == "HCv2-19(b)"
    )
    window = next(m for m in part["data"]["union"] if m["label"] == "g_b_window")
    cond = window["where"]["@sweep_level"]
    levels = sorted(
        t.tags["sweep_level"]
        for t in D.get_design("A_sweeps").tasks(D.CONFIRMATORY)
        if t.tags["sweep_knob"] == "g_b" and t.seed == S.CONFIRMATORY_SEED_MIN
    )
    inside = [v for v in levels if HE.evaluate_predicate({"x": cond}, {"x": v})]
    assert inside == pytest.approx([2 / 3, 8 / 9, 10 / 9, 4 / 3, 14 / 9], abs=1e-6)


def test_the_montage_member_of_hcv2_12d_is_the_forward_comparator(evaluated, spec):
    # design HCv2-12 (d): "the v1 quadrant montage under the average reference
    # on Hopf G = 0 sources", the comparator of the Hopf arm's EEG-64 view
    # (HCv2-15 (b) reads the same comparator without a reference)
    ctx, _rep = evaluated
    part = next(
        p
        for h in spec["hypotheses"]
        for p in h.get("parts") or ()
        if p["id"] == "HCv2-12(d)"
    )
    label = "v1_quadrant_average_reference"
    member = next(m for m in part["data"]["union"] if m["label"] == label)
    assert member["where"] == {
        "design": "@design.whole_brain",
        "principle": "IIM",
        "@hopf_G": 0,
        "view": "@view.eeg64",
        "estimator_form": "@form.iim_v1_quadrant",
    }

    def cell(c):
        out = HE.evaluate_part(part, spec, c)
        return next(x for x in out["cells"] if x["cell"] == member["label"])

    base = cell(ctx)
    assert (base["n"], base["k"], base["outcome"]) == (61, 61, SUP)  # G = 0 seeds
    # the family-B montage cell does not enter it ...
    other = perturbed(
        ctx,
        "components",
        {"design": "family_b", "config.cell.system": "hopf_eeg_quadrants"},
        lambda r: {**r, "reason": R.INSUFFICIENT_OCCUPANCY},
    )
    assert cell(other)["k"] == 61
    # ... the forward comparator does
    moved = perturbed(
        ctx,
        "components",
        {"design": "whole_brain", "view": "eeg64",
         "estimator_form": "iim_v1_quadrants"},
        lambda r: {**r, "reason": R.INSUFFICIENT_OCCUPANCY},
    )
    assert cell(moved)["k"] == 0


def test_one_protocol_key_convention(tmp_path):
    """A record names its protocol by the key (``A-R``, ``hopf-eeg64``,
    ``A-R+nas_secondary``); the frozen file is mpc_bench_v2_<key>.json and a
    protocol named by the convention is mpc-bench-v2-<key>; every loader keys
    the protocols by that key."""
    for key in ("A-R", "A-H+nas_secondary", "hopf-eeg64+iim_v1_quadrants"):
        name = HE.protocol_file_name(key)
        assert name == f"mpc_bench_v2_{key}.json"
        assert HE.protocol_key_of_file(tmp_path / name) == key
        assert HE.protocol_key_of_name(f"mpc-bench-v2-{key}") == key
        assert HE.protocol_key_of_name(f"mpc-bench-v2-{key}-draft") == key
    assert HE.protocol_key_of_name("my protocol") is None
    with pytest.raises(ValueError):
        HE.protocol_key_of_file(tmp_path / "anchors.json")
    with pytest.raises(ValueError):
        HE.protocol_file_name("")
    proto = E.ProtocolV3()
    for key, name in (("A-R", "mpc-bench-v2-A-R"), ("C1-H", "mpc-bench-v2-C1-H-draft"),
                      ("hopf-eeg64", None)):
        d = proto.to_dict()
        d["name"] = name
        (tmp_path / HE.protocol_file_name(key)).write_text(json.dumps(d))
    # a registry or an anchor table beside them is not read as a protocol
    (tmp_path / "applicability_registry_v3.json").write_text("{}")
    got = HE.load_protocol_files([tmp_path])
    assert list(got) == ["A-R", "C1-H", "hopf-eeg64"]
    assert HE.protocol_key(got["A-R"]) == "A-R"
    with pytest.raises(ValueError, match="does not carry a key"):
        HE.protocol_key(got["hopf-eeg64"])
    # a name that contradicts its file is refused, as is a file outside the
    # convention or a key given twice
    bad = tmp_path / "bad"
    bad.mkdir()
    d = proto.to_dict()
    d["name"] = "mpc-bench-v2-A-H"
    (bad / HE.protocol_file_name("A-R")).write_text(json.dumps(d))
    with pytest.raises(ValueError, match="names the key 'A-R'"):
        HE.load_protocol_files([bad])
    with pytest.raises(ValueError, match="mpc_bench_v2_<key>"):
        HE.load_protocol_files([tmp_path / "applicability_registry_v3.json"])
    with pytest.raises(ValueError, match="given twice"):
        HE.load_protocol_files([tmp_path, tmp_path / HE.protocol_file_name("A-R")])
    # the context keys a list of protocols by the key in their names, so a
    # row's protocol_id finds its protocol
    named = []
    for key in ("A-R", "C1-H"):
        d = proto.to_dict()
        d["name"] = f"mpc-bench-v2-{key}"
        named.append(E.ProtocolV3.from_dict(d))
    ctx = HE.build_context(sources={"components": []}, protocols=named)
    assert set(ctx.protocols) == {"A-R", "C1-H"}


# ==========================================================================
# the specification after the calibration decisions
# ==========================================================================
def spec_part(spec, pid):
    return next(
        p for h in spec["hypotheses"] for p in h.get("parts") or () if p["id"] == pid
    )


def test_hcv2_4_leaves_concordant_twin_sessions_out_of_the_kappa(world, spec):
    part = spec_part(spec, "HCv2-4(a,b)")
    ctx = make_ctx(world)
    base = HE.evaluate_part(part, spec, ctx)
    assert not any("concordant" in c["cell"] for c in base["cells"])
    # the admitted concordant W_PDI_single_attractor sessions form no cell
    assert not any(
        "W_PDI_single_attractor" in c["cell"] and "|PDI|" in c["cell"]
        for c in base["cells"]
    )
    # two of its 35 A-R twin sessions with a jackknife SE: that cell has c
    # defined in 2 of 35 sessions and is not eligible
    two = perturbed(
        ctx,
        "components",
        {
            "design": "A_twins",
            "system": "W_PDI_single_attractor",
            "principle": "PDI",
            "declaration_id": "R",
            "seed": 820,
            "replicate": [0, 1],
        },
        lambda r: restatus(r, 0.0),
    )
    out = HE.evaluate_part(part, spec, two)
    (cell,) = [
        c
        for c in out["cells"]
        if "W_PDI_single_attractor" in c["cell"] and ":R|PDI|" in c["cell"]
    ]
    assert cell["cell"].endswith("|jackknife_contiguous_10")
    assert cell["outcome"] == NE
    assert cell["defined_share"] == pytest.approx(2 / 35)
    assert out["outcome"] == base["outcome"]
    others = [c for c in out["cells"] if c is not cell]
    assert others == base["cells"]


def test_hcv2_20a_takes_the_route_branch_on_the_builders_admitted_cell(spec):
    from scripts.v2 import build_protocols_v2 as BP

    part = spec_part(spec, "HCv2-20(a)")
    battery = {
        "substrate": BP.FAMILY_SUBSTRATE["A"],
        "observation_stage": "source",
        "view": "source",
        "bearer": "non_workspace",
        "rule_absent": True,
        "rule_present": True,
        "provisional": False,
        "content_on_runs": 450,
        "concordant_absent": 0,
        "no_content_runs": 360,
        "concordant_present": 0,
    }

    def chosen(p, evidence):
        cells = BP.concordance_cells(evidence, {"pdi_concordance": "rule"})
        ctx = HE.build_context(
            sources={"components": []},
            protocols={"A-R": {"concordance_route": cells}},
        )
        out, why = HE._apply_choose(dict(p), ctx, spec)
        assert why is None
        return out

    out = chosen(part, [battery])
    assert out["chosen"] == "then" and out["params"]["x"] == 0.8
    assert out["chosen_text"].startswith("ABSENT in >= 80 %")
    # a cell of another substrate does not admit the family-A route
    other = {**battery, "substrate": BP.FAMILY_SUBSTRATE["C1"]}
    assert chosen(part, [other])["chosen"] == "else"
    # the family id the gate compared before never matched the builder's cell
    old = copy.deepcopy(part)
    old["choose"]["if"]["concordance_admitted"]["substrate"] = "@family.A"
    assert chosen(old, [battery])["chosen"] == "else"


def _every_principle_anchored(*keys):
    statuses = {p: {"status": "valid_specific"} for p in HE.PRINCIPLES}
    return {
        k: {"necessity_set": list(HE.PRINCIPLES), "anchors": {"principles": statuses}}
        for k in keys
    }


def test_hcv2_1_and_hcv2_3_count_only_defined_ram_rows(spec):
    null = dict(
        design="null_calibration", decl="none", family="null", protocol="A-none"
    )
    bench = dict(design="A_witnesses", decl="R", family="A", protocol="A-R")
    rows = [
        # RAM-PE on the null-calibration generator: undefined by construction
        comp_row(
            "nc-ram",
            "ar1",
            "UNDEFINED",
            None,
            principle="RAM",
            reason=R.INSUFFICIENT_UPDATES,
            estimate=None,
            **null,
        ),
        # another principle's undefined row stays (a non-event in HCv2-1)
        comp_row(
            "nc-nas",
            "ar1",
            "UNDEFINED",
            None,
            principle="NAS",
            reason=R.INSUFFICIENT_TIMEPOINTS,
            estimate=None,
            **null,
        ),
        # a decision-type UNDEFINED with a finite c is a defined RAM-PE row
        comp_row(
            "w-ar1",
            "N_ar1",
            "UNDEFINED",
            0.05,
            principle="RAM",
            reason=R.ABSENT_NOT_REACHABLE,
            estimate=0.4,
            se_c=0.03,
            df_c=39.0,
            **bench,
        ),
        comp_row(
            "w-noise",
            "N_independent_noise",
            "ABSENT",
            0.0,
            principle="RAM",
            reason=None,
            estimate=0.35,
            se_c=0.02,
            df_c=39.0,
            **bench,
        ),
    ]
    ctx = HE.build_context(
        sources={"components": rows},
        protocols=_every_principle_anchored("A-R", "A-none"),
    )
    for pid in ("HCv2-1", "HCv2-3"):
        prep = HE.prepare(spec_part(spec, pid)["data"], spec, ctx)
        kept = {r["task_id"] for r in prep.rows}
        assert kept == {"nc-nas", "w-ar1", "w-noise"}, pid
    # the cell rule of HCv2-1 counts the defined RAM-PE rows only
    out = HE.evaluate_part(spec_part(spec, "HCv2-1"), spec, ctx)
    ram = [c for c in out["cells"] if c["cell"].endswith("|RAM")]
    assert sum(c["n"] for c in ram) == 2


def test_the_dose_only_rows_are_reported_beside_hcv2_5():
    mech = {
        "status": "final",
        "entries": [
            {"principle": "IIM", "where": {"system": "PC_half"}, "on": False},
            {"where": {"system": ["PC_half", "PC_nominal"]}, "on": True},
        ],
        "dose_only_entries": [
            {"principle": "IIM", "where": {"system": "PC_half"}, "on": True}
        ],
    }
    data = {
        "where": {"design": "witnesses"},
        "event": {"status": "ABSENT"},
        "cells": ["system", "principle"],
    }
    spec = mini_spec(
        [
            {"id": "on", "rule": "describe", "params": {},
             "data": {**data, "mechanism_on": "own"}},
            {"id": "dose", "rule": "describe", "params": {}, "role": "reported",
             "data": {**data, "mechanism_on": "dose_only"}},
        ],
        mechanism_on=mech,
    )
    rows = [
        comp_row(f"{s}-{p}", s, "ABSENT", 0.0, principle=p, seed=20000)
        for s in ("PC_half", "PC_nominal")
        for p in ("IIM", "NAS")
    ]
    ctx = HE.build_context(sources={"components": rows}, split=S.CONFIRMATORY)
    h = next(x for x in spec["hypotheses"] if x["id"] == "HCv2-1")
    on, dose = (HE.evaluate_part(p, spec, ctx) for p in h["parts"])
    assert {c["cell"] for c in on["cells"]} == {
        "system=PC_half|principle=NAS",
        "system=PC_nominal|principle=IIM",
        "system=PC_nominal|principle=NAS",
    }
    (cell,) = dose["cells"]
    assert cell["cell"] == "system=PC_half|principle=IIM" and cell["k"] == 1
    # an unknown mode is refused; labels that are not final are not used in a
    # confirmatory evaluation
    bad = copy.deepcopy(h["parts"])
    bad[1]["data"]["mechanism_on"] = "dose"
    with pytest.raises(HE.SpecError, match="mechanism_on must be one of"):
        mini_spec(bad, mechanism_on=mech)
    draft = mini_spec(h["parts"], mechanism_on={**mech, "status": "draft"})
    hd = next(x for x in draft["hypotheses"] if x["id"] == "HCv2-1")
    out = HE.evaluate_part(hd["parts"][1], draft, ctx)
    assert out["outcome"] == NE and "not final" in out["reason"]


def test_the_shipped_mechanism_table_is_the_cd8_rule():
    spec = HE.load_spec()
    entries = spec["mechanism_on"]["entries"]
    fields = HE.Fields(spec.get("fields"), spec.get("derived"))

    def on(family, decl, design, system, principle):
        row = {"family": family, "declaration_id": decl, "design": design,
               "system": system}
        return HE._mechanism_on(row, principle, entries, fields)

    # every witness with a dose is labelled, not only the eight mechanism
    # witnesses: W_PDI_no_multistability keeps RAM on and has PDI off
    assert on("A", "R", "A_witnesses", "W_PDI_no_multistability", "RAM")
    assert not on("A", "R", "A_witnesses", "W_PDI_no_multistability", "PDI")
    assert on("A", "R", "A_witnesses", "N_modules_disconnected", "RAM")
    # the median condition: A-R IIM with g_b off is off although its dose is on
    assert not on("A", "R", "A_witnesses", "PC_half", "IIM")
    # C1 IIM is labelled by the dose condition alone (HO-4)
    assert on("C1", "R", "C1_factorial", "b00010", "IIM")
    assert not on("C1", "R", "C1_witnesses", "W_IIM_feedforward", "IIM")
    # an unmatched row is not on
    assert not on("A", "R", "A_adversaries", "ADV_NAS_staggered_driver", "NAS")


def test_hcv2_21_corrected_admission_keeps_the_original_prediction(spec):
    part = spec_part(spec, "HCv2-21")
    preds, orig = part["params"]["predictions"], part["params"]["original_predictions"]
    assert {v: p["admitted_for_absent"] for v, p in preds.items()} == {
        "@view.eeg64": "vacuous",
        "@view.eeg_low": "vacuous",
        "@view.bold": "vacuous",
    }
    assert {v: p["admitted_for_absent"] for v, p in orig.items()} == {
        "@view.eeg64": "yes",
        "@view.eeg_low": "yes",
        "@view.bold": "no",
    }
    # the PRESENT predictions are not part of the correction
    for v in ("@view.eeg64", "@view.eeg_low"):
        assert preds[v]["admitted_for_present"] == orig[v]["admitted_for_present"]
    assert "logical error" in part["notes"]
    pending = [i for i in spec["pending_calibration"] if "(CD-5)" in i["item"]]
    assert "HCv2-21" in pending[0]["affects"]

    # the registry pattern of the original prediction now falsifies it
    def registry(absent):
        return HE.registry_rows(
            {
                "entries": [
                    {"principle": "PDI", "arm": arm, "substrate": "x", "view": v,
                     "admitted_for_present": "yes", "admitted_for_absent": a,
                     "anchor_valid": True}
                    for v, arm, a in zip(("eeg64", "eeglow", "bold"),
                                         ("forward_a_eeg", "forward_a_eeg",
                                          "forward_a_bold"), absent)
                ]
            }
        )

    for absent, want in ((("vacuous",) * 3, SUP), (("yes", "yes", "no"), FAL)):
        ctx = HE.build_context(sources={"components": [], "registry": registry(absent)})
        assert HE.evaluate_part(part, spec, ctx)["outcome"] == want


def test_the_anchor_hypotheses_follow_the_development_statuses(spec):
    part = spec_part(spec, "HCv2-6(a)")
    preds = {
        (e["protocol"], e["principle"]): e["status"]
        for e in part["params"]["predictions"]
    }
    assert preds[("@protocol.A-H", "NAS")] == "valid_specific"
    assert preds[("@protocol.A-H", "IIM")] == "valid_nonspecific"
    assert preds[("@protocol.C1-H", "IIM")] == "valid_specific"
    assert "held out (HO-4)" not in part["notes"]
    assert "20900-20939" in part["text"]
    # the K -> NAS cell of HCv2-22(iii) is predicted by declaration
    p22 = spec_part(spec, "HCv2-22(iii)")
    by_decl = {
        cp["match"]["declaration_id"]: cp["prediction"]
        for cp in p22["cell_predictions"]
    }
    assert by_decl == {"H": FAL, "R": SUP}
    assert "14/20" not in p22["notes"]


def test_the_cell_predictions_of_hcv2_22_iii_split_by_declaration(world, spec):
    # with the decided A-H anchors (NAS specific under H, CD-7) the K -> NAS
    # cell exists under both declarations
    ctx = make_ctx(world)
    ctx.protocols = copy.deepcopy(ctx.protocols)
    a_h = ctx.protocols["A-H"]
    a_h["anchors"]["necessity_set"].append("NAS")
    a_h["anchors"]["principles"]["NAS"]["status"] = "valid_specific"
    a_h["necessity_set"] = list(a_h["anchors"]["necessity_set"])
    p = HE.evaluate_part(spec_part(spec, "HCv2-22(iii)"), spec, ctx)
    cells = {
        c["cell"]: c.get("prediction")
        for c in p["cells"]
        if c["cell"].endswith("|W_PDI_single_attractor|NAS")
    }
    assert cells == {"A-H|W_PDI_single_attractor|NAS": FAL,
                     "A-R|W_PDI_single_attractor|NAS": SUP}


def test_the_new_reported_parts_are_reported(evaluated):
    _, rep = evaluated
    parts = {p["id"]: p for h in rep["hypotheses"] for p in h["parts"]}
    for pid in ("HCv2-1(definedness)", "HCv2-5(dose-only)", "HCv2-14(H-present)"):
        assert parts[pid]["role"] == "reported"
        assert parts[pid]["outcome"] == HE.REPORTED, pid
    h14 = {c["cell"]: c for c in parts["HCv2-14(H-present)"]["cells"]}
    assert set(h14) == {"A:O_inert", "A:W_IIM_feedforward"}


def test_the_held_out_predictions_are_unchanged_since_their_commit(spec):
    """HCv2-10, HCv2-15 and the held-out parts keep their committed text,
    rule, parameters, data and predictions (d3bcb09); HCv2-21 is the one
    documented correction (CD-5), and a held-out part may only gain an
    expectation note."""
    import subprocess
    from pathlib import Path

    root = Path(HE.SPEC_PATH).resolve().parents[2]
    proc = subprocess.run(
        ["git", "-C", str(root), "show", "d3bcb09:protocols/v2/hypotheses_v2.json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip("the commit of the held-out predictions is not available")
    old = {h["id"]: h for h in json.loads(proc.stdout)["hypotheses"]}
    new = {h["id"]: h for h in spec["hypotheses"]}
    for hid in ("HCv2-10", "HCv2-15"):
        assert new[hid] == old[hid], hid
    keys = ("text", "rule", "params", "data", "prediction", "cell_predictions",
            "gate", "choose", "role", "label", "replaced_by")
    for hid, h in old.items():
        for p in h.get("parts") or ():
            held_out = p.get("label") == "HO" or "HO" in (h.get("labels") or ()) and (
                p.get("label") is None
            )
            if not held_out or p["id"] == "HCv2-21":
                continue
            q = spec_part(spec, p["id"])
            assert {k: q.get(k) for k in keys} == {k: p.get(k) for k in keys}, p["id"]


def test_the_held_out_transcription_agrees_with_the_hypotheses(spec):
    """The transcribed admission table (protocols/v2/held_out_predictions_v2.json)
    states the flags the admission hypotheses decide on, the HCv2-21
    correction included, and names existing hypotheses."""
    from pathlib import Path

    path = Path(HE.SPEC_PATH).with_name("held_out_predictions_v2.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema"] == "mpc-bench-held-out-predictions/1"
    assert [e["id"] for e in doc["held_out_elements"]] == [
        f"HO-{i}" for i in range(1, 8)
    ]
    ids = {h["id"] for h in spec["hypotheses"]} | {
        e["id"] for e in spec["integrity_audit"]
    }
    for row in doc["held_out_elements"] + doc["admission_table"]:
        for t in row["tested_in"]:
            assert t.split(" ")[0].split("(")[0] in ids, t
    vocab = spec["vocabulary"]

    def flags(pid, flag):
        preds = HE.resolve(spec_part(spec, pid)["params"]["predictions"], spec)
        return {v: p[flag] for v, p in preds.items() if flag in p}

    def table(principle, flag):
        return {
            v: row[flag]
            for row in doc["admission_table"]
            if row["principle"] == principle and flag in row
            for v in row["views"]
        }

    nas_present = table("NAS", "admitted_for_present")
    assert flags("HCv2-10", "admitted_for_present") == nas_present
    assert flags("HCv2-10(absent)", "admitted_for_absent") == table(
        "NAS", "admitted_for_absent"
    )
    pdi = spec_part(spec, "HCv2-21")
    assert flags("HCv2-21", "admitted_for_absent") == table(
        "PDI", "admitted_for_absent"
    )
    assert flags("HCv2-21", "admitted_for_present") == table(
        "PDI", "admitted_for_present"
    )
    original = HE.resolve(pdi["params"]["original_predictions"], spec)
    for row in doc["admission_table"]:
        if row["principle"] == "PDI":
            assert row["corrected"] == "HCv2-21"
            for v in row["views"]:
                assert (
                    original[v]["admitted_for_absent"]
                    == row["original"]["admitted_for_absent"]
                )
    assert set(table("PDI", "admitted_for_absent")) == {
        vocab["view.eeg64"],
        vocab["view.eeg_low"],
        vocab["view.bold"],
    }
