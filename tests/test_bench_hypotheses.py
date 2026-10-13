# -*- coding: utf-8 -*-
"""The declarative hypothesis evaluator: every decision rule and statistic on
hand-computed fixtures; NOT_EVALUABLE, NOT_TESTABLE_BY_DESIGN, removal and
the tally; the manipulation checks per row; the example hypotheses file
(tests/fixtures/hypotheses_example.json); the record fields it names on
records of the runner and the family-B validation; and the evaluator script
on synthetic development records."""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from impact_pipeline import evidence as E
from impact_pipeline import necessity as NC
from impact_pipeline import reasons as R
from impact_pipeline.bench import hypotheses as HE
from impact_pipeline.bench import records as REC
from impact_pipeline.bench import runner as RB
from impact_pipeline.bench import seeds as S

SUP, FAL, IND = HE.SUPPORTED, HE.FALSIFIED, HE.INDETERMINATE
NE, NTBD = HE.NOT_EVALUABLE, HE.NOT_TESTABLE_BY_DESIGN
EXAMPLE = Path(__file__).resolve().parent / "fixtures" / "hypotheses_example.json"


def example_spec():
    return HE.load_spec(EXAMPLE)


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
def test_clopper_pearson_inverts_the_binomial_and_matches_necessity(k, n):
    up, lo = HE.cp_upper(k, n, 0.05), HE.cp_lower(k, n, 0.05)
    # the defining identities: P(X <= k | upper) = P(X >= k | lower) = 0.05
    assert stats.binom.cdf(k, n, up) == pytest.approx(0.05, abs=1e-9)
    assert stats.binom.sf(k - 1, n, lo) == pytest.approx(0.05, abs=1e-9)
    assert up == pytest.approx(NC.clopper_pearson(k, n, 0.05, "upper")[1], abs=1e-15)
    assert lo == pytest.approx(NC.clopper_pearson(k, n, 0.05, "lower")[0], abs=1e-15)


def test_seeds_needed_for_a_demonstration_match_the_design():
    # 42 / 51 / 61 / 66 seeds for 0.07 at m = 1/2/4/6, 59 for 0.05, 149 for
    # 0.02, 299 for 0.01
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
                        "estimator_version": "iim-macro",
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
                    "estimator_version": "pdi-declared",
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


def test_kappa_null_leaves_admitted_concordant_rows_out():
    # an admitted concordant component has SE 0 and no sampling SE (its
    # error control is the concordance battery): with exclude_concordant it
    # enters no kappa0 or tail, and its cell keeps its place in P
    z = stats.norm.ppf((np.arange(400) + 0.5) / 400)

    def row(cell, value, se, method, i):
        return {
            "_cell": cell,
            "_value": value,
            "se_c": se,
            "df_c": 9.0,
            "_cluster": f"{cell}{i}",
            "se_method": method,
        }

    sampled = [row("NAS", 0.02 * v, 0.02, "jackknife", i) for i, v in enumerate(z)]
    # concordant two-state counts: c > 0 with SE 0
    mixed = sampled + [row("NAS", 0.3, 0.0, "concordant", 400 + i) for i in range(40)]
    only = [row("PDI", 0.0, 0.0, "concordant", i) for i in range(30)]
    opt = {"exclude_concordant": True}
    r = HE.RULES["kappa_null"](HE.Prepared(copy.deepcopy(mixed + only)), opt)
    nas, pdi = r.cells
    assert nas["cell"] == "NAS" and nas["n"] == 400 and nas["n_concordant"] == 40
    assert nas["outcome"] == SUP and nas["kappa"] == pytest.approx(1.0, abs=0.02)
    assert pdi == {
        "cell": "PDI",
        "n": 0,
        "n_concordant": 30,
        "outcome": NE,
        "reason": HE.CONCORDANT_ONLY_REASON,
    }
    assert r.outcome == SUP and r.stats == {"P": 2}
    # the NAS cell is the cell of its sampled rows alone, at the same P
    alone = HE.RULES["kappa_null"](HE.Prepared(copy.deepcopy(sampled + only)), {})
    assert alone.stats == {"P": 2}
    assert {k: v for k, v in nas.items() if k != "n_concordant"} == alone.cells[0]
    # without the option the concordant rows enter as SE-0 rows: each is an
    # upper-tail event, and the cell is no longer calibrated
    old = HE.RULES["kappa_null"](HE.Prepared(copy.deepcopy(mixed + only)), {})
    assert old.cells[0]["n"] == 440 and "n_concordant" not in old.cells[0]
    up = r.cells[0]["tails"]["upper"]["k"]
    assert old.cells[0]["tails"]["upper"]["k"] == up + 40
    assert old.cells[0]["outcome"] != SUP
    assert old.cells[1]["reason"] == HE.ZERO_RMS_REASON
    # a cell without concordant rows reports 0 and is judged as before
    plain = HE.RULES["kappa_null"](HE.Prepared(copy.deepcopy(sampled)), opt)
    base = HE.RULES["kappa_null"](HE.Prepared(copy.deepcopy(sampled)), {})
    assert plain.cells[0].pop("n_concordant") == 0
    assert plain.cells == base.cells and plain.outcome == base.outcome


def test_kappa_twins_per_se_method_counts_the_class_sessions():
    # a class of 5 twin networks x 7 sessions in which 33 sessions are
    # admitted concordant components (SE 0, no sampling SE) and 2 carry the
    # jackknife SE: the concordant sessions form no cell and enter no kappa,
    # and the jackknife cell has c defined in 2 of 35 sessions, so it is not
    # eligible (>= 80 %)
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
    """A valid hypotheses file: the example file with ``parts`` as the parts
    of its hypothesis ``null_present``."""
    spec = copy.deepcopy(example_spec())
    for h in spec["hypotheses"]:
        if h["id"] == "null_present":
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
    h = next(h for h in spec["hypotheses"] if h["id"] == "null_present")
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
            "hypothesis": "null_present",
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
            "hypothesis": "null_present",
            "part": "p",
            "reason": "EVALUATOR_ERROR: RuntimeError: defect",
        }
    ]
    h1 = next(h for h in rep["hypotheses"] if h["id"] == "null_present")
    assert h1["parts"][0]["outcome"] == NE


def test_hypotheses_not_admitted_are_not_run():
    res = HE.evaluate(example_spec(), HE.build_context())
    assert [h for h, o in res["outcomes"].items() if o == HE.NOT_RUN] == [
        "not_admitted"
    ]
    assert res["tally"]["not_run"] == ["not_admitted"]
    (h,) = [h for h in res["hypotheses"] if h["id"] == "not_admitted"]
    assert h["admitted"] is False and h["reason"] == "not admitted"
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
            estimator_version="ram-pe-2026.10",
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
                "RAM": ["ram-pe-2026.10", "shift_null_sd"],
                "NAS": ["nas-declared-2026.10", "jackknife_contiguous_10"],
            },
        }
        for i in range(3)
    ]
    ctx = HE.build_context(
        sources={"components": comps, "verdicts": verds},
        protocols={"A-R": {"necessity_set": ["RAM", "NAS"]}},
    )
    new = HE.apply_reversion(ctx, {("ram-pe-2026.10", "shift_null_sd")})
    assert {r["status"] for r in new.sources["components"]} == {"UNDEFINED"}
    assert {r["reason"] for r in new.sources["components"]} == {R.SE_NOT_CALIBRATED}
    assert {v["verdict_value"] for v in new.sources["verdicts"]} == {"UNDETERMINED"}
    assert {r["status"] for r in ctx.sources["components"]} == {"ABSENT"}  # untouched
    # an EXCLUDED verdict that also rests on a calibrated ABSENT stays EXCLUDED
    verds[0]["component_status"]["NAS"] = "ABSENT"
    new = HE.apply_reversion(ctx, {("ram-pe-2026.10", "shift_null_sd")})
    assert new.sources["verdicts"][0]["verdict_value"] == "EXCLUDED"


# ==========================================================================
# the hypotheses file
# ==========================================================================
def test_the_example_hypotheses_file_is_valid():
    spec = example_spec()
    assert spec["schema"] == HE.SPEC_SCHEMA == "mpc-bench-hypotheses/1"
    for h in spec["hypotheses"]:
        if h.get("admitted", True):
            assert any(p.get("role", "decisive") == "decisive" for p in h["parts"])
        else:
            assert not h["parts"]
    assert len(HE.spec_sha256(spec)) == 64
    with pytest.raises(TypeError):
        HE.load_spec()


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
        (lambda s: s["hypotheses"][2].update(admitted="no"), "admitted is true or false"),
        (lambda s: s["hypotheses"][2].update(parts=[]), "an admitted hypothesis has parts"),
        (lambda s: s["hypotheses"][2].update(priority="A"), "unknown keys"),
        (lambda s: s.update(checks=[]), "unknown top-level keys"),
        (
            lambda s: s["hypotheses"][2]["parts"].append(
                copy.deepcopy(s["hypotheses"][2]["parts"][0])
            ),
            "duplicate",
        ),
        (lambda s: s.update(schema="x"), "schema"),
        (lambda s: s.update(schema="mpc-bench-hypotheses/2"), "schema"),
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
    spec = copy.deepcopy(example_spec())
    mutate(spec)
    with pytest.raises(HE.SpecError, match=match):
        HE.validate_spec(spec)


def test_spec_hash_is_canonical():
    spec = example_spec()
    again = json.loads(json.dumps(spec, sort_keys=False, indent=4))
    assert HE.spec_sha256(again) == HE.spec_sha256(spec)


# ==========================================================================
# edge cases of the rules and the manipulation checks per row
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


def test_manipulation_checks_drop_only_the_rows_of_an_unrealised_system():
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


def test_manipulation_checks_are_not_named_when_the_selection_is_empty():
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


def test_manipulation_checks_cover_both_systems_of_a_pair_and_variants():
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


def test_manipulation_checks_gate_the_slow_context_arm_only():
    """The slow-context check of the example file is required by the rows of
    the forward BOLD arm (its records, anchor replications and registry
    entries) and by no other row, and is looked up in family A for registry
    entries."""
    spec = example_spec()
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
    vacuous = {"admitted_for_present": "yes", "admitted_for_absent": "vacuous"}
    admission = {
        "id": "p-admission",
        "rule": "admission_matches",
        "params": {"predictions": {"eeg64": vacuous, "eeglow": vacuous,
                                   "bold": {"admitted_for_absent": "vacuous"}}},
        "data": {"source": "registry", "where": {"principle": "PDI"},
                 "cell_label": "{view}"},
        "requires": {"oracle": True},
    }
    aspec = mini_spec([admission])
    for usable, bold in ((True, SUP), (False, NE)):
        ctx = HE.build_context(sources={"components": [], "registry": reg})
        ctx.usability = {("A", SLOW): usable}
        out = HE.evaluate_part(admission, aspec, ctx)
        cells = {c["cell"]: c for c in out["cells"]}
        assert out["outcome"] == SUP and cells["eeg64"]["outcome"] == SUP
        assert cells["bold"]["outcome"] == bold
        # an emptied cell is listed once, with the reason ORACLE
        assert [c["cell"] for c in out["cells"]].count("bold") == 1
        if not usable:
            assert cells["bold"]["reason"].startswith("ORACLE")
            assert cells["bold"]["checks"] == [f"A:{SLOW}"]


def test_oracle_conditions_are_validated():
    spec = copy.deepcopy(example_spec())
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
        ("realisation", SLOW, True),
        ("reported", "PC_nominal/twin_hashes", True),
        ("reported", "PC_half/twin_hashes", True),
    ]
    # reported rows never decide usability
    ctx = HE.build_context(sources={"manipulation": rows})
    rep = HE.evaluate(example_spec(), ctx)
    assert rep["usability"] == {"A:g_b": True, f"A:{SLOW}": True}
    h0 = next(h for h in rep["hypotheses"] if h["id"] == "checks")
    parts = {p["id"]: p for p in h0["parts"]}
    assert parts["checks(c)"]["role"] == "reported"
    assert parts["checks(c)"]["outcome"] == HE.REPORTED
    (cell,) = parts["checks(c)"]["cells"]
    assert cell["cell"] == "family=A|check_id=PC_half:K/between_off_and_nominal"
    assert cell["k"] == 0 and cell["n"] == 1
    assert [c["cell"] for c in parts["checks(d)"]["cells"]] == [
        "family=A|system_id=PC_nominal", "family=A|system_id=PC_half"]
    assert all(c["k"] == 1 for c in parts["checks(d)"]["cells"])


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
        ("realisation", SLOW, 600, False),
    ]


def test_the_evaluator_reads_the_oracle_directory_without_its_summaries(tmp_path):
    """Every table the oracle item writes is read (the evaluator is given the
    directory's CSV files): the usability summary and the per-seed PC_half
    values are no check results and add no row, so a check passed in 40 of
    40 seeds counts 40 rows, not 41."""
    import csv

    import scripts.evaluate_hypotheses as BH

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
        ("realisation", SLOW): (39, 40),
        ("reported", "PC_half:g_b/between_off_and_nominal"): (1, 1),
        ("reported", "PC_nominal/twin_hashes"): (1, 1),
    }


def test_requirements_are_validated():
    spec = copy.deepcopy(example_spec())
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


# a kappa_null part per anchored principle with NAS per direction, on the
# null-calibration generator and the family-A null witnesses
KAPPA_NULL_WHERE = {
    "declaration_id": ["R", "none"],
    "estimator_form": "@form.primary",
    "any": [
        {"design": "@design.null_calibration"},
        {"design": "@design.witnesses", "family": "@family.A",
         "system": "@systems.nulls_A"},
    ],
    "all": [{"any": [{"principle": {"ne": "RAM"}}, {"defined": True}]}],
}
KAPPA_NULL = {
    "id": "p-kappa-null",
    "rule": "kappa_null",
    "params": {"point": [0.8, 1.25], "interval": [0.67, 1.5], "level": 0.9,
               "tail_max": 0.075, "tail_falsify": 0.1, "q_alpha": 0.05,
               "exclude_concordant": True},
    "data": {
        "where": KAPPA_NULL_WHERE,
        "union": [
            {"label": "principle", "where": {"principle": {"ne": "NAS"}},
             "cell_label": "{principle}"},
            {"label": "NAS_receive", "where": {"principle": "NAS"},
             "value": "@nas_c_receive", "se": "@nas_se_c_receive",
             "df": "@nas_df_c_receive", "cell_label": "NAS:receive"},
            {"label": "NAS_return", "where": {"principle": "NAS"},
             "value": "@nas_c_return", "se": "@nas_se_c_return",
             "df": "@nas_df_c_return", "cell_label": "NAS:return"},
        ],
        "principle_filter": "valid_anchor",
        "value": "c",
    },
    "cell_predictions": [
        {"match": {"principle": ["NAS", "IIM", "PDI", "RAM"]},
         "prediction": "SUPPORTED"}
    ],
}
KAPPA_NULL_CONCORDANT = {
    "id": "p-kappa-null-concordant",
    "role": "reported",
    "rule": "describe",
    "params": {},
    "data": {"where": KAPPA_NULL_WHERE, "principle_filter": "valid_anchor",
             "event": {"concordant": True}, "cell_label": "{principle}"},
}


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
    spec = mini_spec([KAPPA_NULL])
    part = KAPPA_NULL
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


# ==========================================================================
# the record fields of the example file on the records of the runner and
# of the family-B validation script
# ==========================================================================
def _values(v):
    return list(v) if isinstance(v, list) else [v]


def test_the_example_vocabulary_names_designs_forms_and_views_of_the_producers():
    from impact_pipeline.bench import designs as D
    from impact_pipeline.bench.designs import family_b as FB
    from impact_pipeline.bench.designs import null_calibration as NCV

    vocab = example_spec()["vocabulary"]
    runner = D.designs()
    assert NCV.DESIGN in runner and vocab["design.null_calibration"] == NCV.DESIGN
    designs = set(runner) | {FB.DESIGN}
    families = {d.family for d in runner.values()} | {FB.FAMILY}
    forms = set(D.ESTIMATOR_FORMS) | set(FB.CUT_MODES)
    for key, val in vocab.items():
        kind = key.split(".")[0]
        want = {"design": designs, "family": families, "form": forms,
                "view": {D.SOURCE_VIEW}}.get(kind)
        if want is not None:
            assert set(_values(val)) <= want, (key, val)


def _runner_record(task, protocols, settings=None):
    from impact_pipeline.bench import runner as RB

    rec = RB.run_task(task, protocols, settings or RB.DEFAULT_SETTINGS)
    assert rec.status in (REC.TASK_OK, REC.TASK_OK_WITH_COMPONENT_ERRORS), rec.error
    return rec.to_dict()


def _anchored(keys, estimators=None):
    """Draft protocols with external anchors (and estimator options)."""
    from impact_pipeline.bench import runner as RB

    ref = {"kind": "external", "scale": "excess",
           "values": {"RAM": 1.0, "PDI": 1.0, "IIM": 1.0, "SRPI": 1.0,
                      "NAS:receive": 1.0, "NAS:return": 1.0}}
    out = {}
    for k in keys:
        d = RB.draft_protocol(k).to_dict()
        d["reference"] = ref
        for p, opts in (estimators or {}).items():
            d["estimators"][p].update(opts)
        out[k] = RB.ResolvedProtocol(k, E.Protocol.from_dict(d), "override")
    return out


def test_estimator_fields_resolve_on_runner_records():
    import dataclasses

    from impact_pipeline.bench import designs as D
    from impact_pipeline.bench.designs import family_a as FA

    spec = example_spec()
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
    # Macro IIM's rank p-value (a small null and bootstrap, wiring only)
    task = dataclasses.replace(task, scorings=D.make_scorings({"R": "A-R"}, ("IIM",)))
    protos = _anchored(["A-R"], {"IIM": {"n_null": 19, "bootstrap_replicates": 2}})
    iim = HE.component_rows([_runner_record(task, protos)])[0]
    assert 0 < HE._finite(fields.get(iim, "@iim_p_ind")) <= 1


def test_configuration_fields_resolve_on_runner_records():
    from impact_pipeline.bench import designs as D
    from impact_pipeline.bench import runner as RB

    spec = example_spec()
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
    from impact_pipeline.bench.designs import family_b as FB
    from scripts import iim_validation as IV

    spec = example_spec()
    fields = HE.Fields(spec["fields"], spec["derived"])
    by_id = {}
    for c in FB.cells():
        by_id.setdefault(c.cell_id, c)
    want = {
        "independent": ("rank_calibration", ("i",)),
        "stratified": ("rank_calibration", ("ii",)),
    }
    conds = {c.condition for c in by_id.values()
             if c.hypothesis == "rank_calibration"}
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
        if c.hypothesis == "exact_targets" and "c" in c.parts
    }
    assert occ <= {"undefined", "defined"} and occ
    # one cheap task scored end to end: the scoring and component fields
    task = next(
        t
        for t in FB.tasks(S.DEVELOPMENT, hypotheses=("rank_calibration",))
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


def test_the_iim_rank_p_value_resolves_in_both_record_layouts():
    """The family-B validation writes ``p_ind`` in the component details
    only; the runner writes it there and in the estimator details."""
    spec = example_spec()
    assert spec["fields"]["iim_p_ind"] == "details.p_ind"
    fields = HE.Fields(spec["fields"], spec["derived"])
    family_b = {"principle": "IIM", "details": {"p_ind": 0.03}}
    runner = {"principle": "IIM",
              "details": {"p_ind": 0.6, "estimator": {"p_ind": 0.6}}}
    assert fields.get(family_b, "@iim_p_ind") == 0.03
    assert fields.get(runner, "@iim_p_ind") == 0.6


# ==========================================================================
# protocol keys, choices, definedness and concordant rows
# ==========================================================================
def test_one_protocol_key_convention(tmp_path):
    """A record names its protocol by the key (``A-R``, ``hopf-eeg64``,
    ``A-R+nas_secondary``); the file is protocols/bench/<key>.json and a
    protocol named by the convention is mpc-bench-<key>; every loader keys
    the protocols by that key."""
    for key in ("A-R", "A-H+nas_secondary", "hopf-eeg64+iim_quadrants"):
        name = RB.protocol_file_name(key)
        assert name == f"{key}.json"
        assert RB.protocol_key_of_file(tmp_path / name) == key
        assert RB.protocol_key_of_name(f"mpc-bench-{key}") == key
        assert RB.protocol_key_of_name(f"mpc-bench-{key}-draft") == key
    assert RB.protocol_key_of_name("my protocol") is None
    with pytest.raises(ValueError):
        RB.protocol_key_of_file(tmp_path / "anchors.csv")
    with pytest.raises(ValueError):
        RB.protocol_file_name("")
    proto = E.Protocol(status_rule="tost")
    for key, name in (("A-R", "mpc-bench-A-R"), ("C-H", "mpc-bench-C-H-draft"),
                      ("hopf-eeg64", None)):
        d = proto.to_dict()
        d["name"] = name
        (tmp_path / RB.protocol_file_name(key)).write_text(json.dumps(d))
    # the template beside them is not read as a family protocol
    d = proto.to_dict()
    d["name"] = "mpc-bench-template"
    (tmp_path / RB.PROTOCOL_TEMPLATE_FILE).write_text(json.dumps(d))
    got = RB.load_protocol_files([tmp_path])
    assert list(got) == ["A-R", "C-H", "hopf-eeg64"]
    assert RB.protocol_key(got["A-R"]) == "A-R"
    with pytest.raises(ValueError, match="does not carry a key"):
        RB.protocol_key(got["hopf-eeg64"])
    # a name that contradicts its file is refused, as is a file outside the
    # convention or a key given twice
    bad = tmp_path / "bad"
    bad.mkdir()
    d = proto.to_dict()
    d["name"] = "mpc-bench-A-H"
    (bad / RB.protocol_file_name("A-R")).write_text(json.dumps(d))
    with pytest.raises(ValueError, match="names the key 'A-R'"):
        RB.load_protocol_files([bad])
    (tmp_path / "notes.txt").write_text("{}")
    with pytest.raises(ValueError, match="<key>.json"):
        RB.load_protocol_files([tmp_path / "notes.txt"])
    with pytest.raises(ValueError, match="given twice"):
        RB.load_protocol_files([tmp_path, tmp_path / RB.protocol_file_name("A-R")])
    # the context keys a list of protocols by the key in their names, so a
    # row's protocol_id finds its protocol
    named = []
    for key in ("A-R", "C-H"):
        d = proto.to_dict()
        d["name"] = f"mpc-bench-{key}"
        named.append(E.Protocol.from_dict(d))
    ctx = HE.build_context(sources={"components": []}, protocols=named)
    assert set(ctx.protocols) == {"A-R", "C-H"}


ROUTE_PART = {
    "id": "p-route",
    "rule": "three_zone",
    "params": {"x": 0.8},
    "data": {
        "where": {"design": "@design.witnesses", "family": "@family.A",
                  "principle": "PDI", "system": "W_PDI_single_attractor"},
        "event": {"status": "ABSENT"},
    },
    "choose": {
        "if": {"concordance_admitted": {"principle": "PDI",
                                        "substrate": "synthetic_rate",
                                        "bearer": "non_workspace",
                                        "direction": "absent"}},
        "then": {"text": "ABSENT in >= 80 % (route admitted)"},
        "else": {
            "text": "not PRESENT in >= 95 % (route not admitted)",
            "params": {"x": 0.95},
            "data": {
                "where": {"design": "@design.witnesses", "family": "@family.A",
                          "principle": "PDI", "system": "W_PDI_single_attractor"},
                "event": {"status": {"ne": "PRESENT"}},
            },
        },
    },
}


def test_a_choice_takes_the_route_branch_on_an_admitted_cell():
    spec = mini_spec([ROUTE_PART])
    part = ROUTE_PART
    battery = {
        "substrate": "synthetic_rate",
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
        cells = [
            {"principle": "PDI", "substrate": e["substrate"],
             "observation_stage": e["observation_stage"], "view": e["view"],
             "bearer": e["bearer"], "absent": e["rule_absent"],
             "present": e["rule_present"], "provisional": e["provisional"],
             "battery": {k: e[k] for k in ("content_on_runs", "concordant_absent",
                                           "no_content_runs", "concordant_present")}}
            for e in evidence if e["rule_absent"] or e["rule_present"]
        ]
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
    other = {**battery, "substrate": "stuart_landau"}
    assert chosen(part, [other])["chosen"] == "else"
    # a family id does not name the admitted cell
    old = copy.deepcopy(part)
    old["choose"]["if"]["concordance_admitted"]["substrate"] = "@family.A"
    assert chosen(old, [battery])["chosen"] == "else"


def _every_principle_anchored(*keys):
    statuses = {p: {"status": "valid_specific"} for p in HE.PRINCIPLES}
    return {
        k: {"necessity_set": list(HE.PRINCIPLES), "anchors": {"principles": statuses}}
        for k in keys
    }


def test_the_null_parts_count_only_defined_ram_rows():
    spec = mini_spec([example_spec()["hypotheses"][1]["parts"][0],
                      KAPPA_NULL])
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
        # another principle's undefined row stays (a non-event of the H0 rule)
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
    for part in spec["hypotheses"][1]["parts"]:
        prep = HE.prepare(part["data"], spec, ctx)
        kept = {r["task_id"] for r in prep.rows}
        assert kept == {"nc-nas", "w-ar1", "w-noise"}, part["id"]
    # the H0 cell rule counts the defined RAM-PE rows only
    out = HE.evaluate_part(spec["hypotheses"][1]["parts"][0], spec, ctx)
    ram = [c for c in out["cells"] if c["cell"].endswith("|RAM")]
    assert sum(c["n"] for c in ram) == 2


def test_kappa_null_leaves_admitted_concordant_pdi_rows_out_and_reports_them():
    spec = mini_spec([KAPPA_NULL, KAPPA_NULL_CONCORDANT])
    null = dict(
        design="null_calibration", decl="none", family="null", protocol="A-none"
    )
    z = stats.norm.ppf((np.arange(200) + 0.5) / 200)
    sampled = [
        comp_row(
            f"j{i}",
            "ar1",
            "ABSENT",
            0.02 * v,
            principle="PDI",
            seed=i,
            se_c=0.02,
            df_c=None,
            se_method="jackknife_contiguous_10",
            **null,
        )
        for i, v in enumerate(z)
    ]
    concordant = [
        comp_row(
            f"k{i}",
            "ar1",
            "ABSENT",
            0.0,
            principle="PDI",
            seed=500 + i,
            se_c=0.0,
            df_c=None,
            se_method="concordant",
            **null,
        )
        for i in range(50)
    ]
    protocols = _every_principle_anchored("A-R", "A-none")
    part = KAPPA_NULL
    assert part["params"]["exclude_concordant"] is True

    def run(rows):
        ctx = HE.build_context(sources={"components": rows}, protocols=protocols)
        out = HE.evaluate_part(part, spec, ctx)
        rep = HE.evaluate_part(KAPPA_NULL_CONCORDANT, spec, ctx)
        return out, rep

    # mixed: the PDI cell is judged on its 200 rows with a sampling SE
    out, rep = run(sampled + concordant)
    (cell,) = out["cells"]
    assert cell["cell"] == "PDI" and cell["n"] == 200 and cell["n_concordant"] == 50
    assert cell["kappa"] == pytest.approx(1.0, abs=0.02)
    assert cell["outcome"] == SUP and cell["prediction"] == SUP
    assert out["stats"] == {"P": 1}
    assert rep["role"] == HE.REPORTED_ROLE and rep["outcome"] == HE.REPORTED
    (rc,) = rep["cells"]
    assert (rc["cell"], rc["k"], rc["n"]) == ("PDI", 50, 250)
    assert rc["rate"] == pytest.approx(0.2)
    # development: every PDI null row is concordant, so the cell stays (in
    # P) and is NOT_EVALUABLE, and the share reported is 1
    out, rep = run(concordant)
    (cell,) = out["cells"]
    assert cell["outcome"] == NE and cell["reason"] == HE.CONCORDANT_ONLY_REASON
    assert cell["n_concordant"] == 50 and out["outcome"] == NE
    assert out["stats"] == {"P": 1}
    assert [(c["k"], c["n"]) for c in rep["cells"]] == [(50, 50)]


def test_the_dose_only_rows_are_reported_beside_the_mechanism_on_rows():
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
    h = next(x for x in spec["hypotheses"] if x["id"] == "null_present")
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
    hd = next(x for x in draft["hypotheses"] if x["id"] == "null_present")
    out = HE.evaluate_part(hd["parts"][1], draft, ctx)
    assert out["outcome"] == NE and "not final" in out["reason"]


# ==========================================================================
# the evaluator script on synthetic records (no simulation behind them)
# ==========================================================================
VERSION = {
    "RAM": "ram-pe-2026.10",
    "PDI": "pdi-declared-2026.10",
    "NAS": "nas-declared-2026.10",
    "IIM": "iim-macro-2026.10",
    "SRPI": "srpi-2026.09",
}
A_R_FILE = RB.PROTOCOL_DIR / "A-R.json"
TAG = "bench-confirmatory-1"


def synthetic_record(system, seed, statuses, *, design="A_witnesses", family="A",
                     provenance=None):
    """A record of one A-R scoring; ``statuses`` maps a principle to its
    (status, c)."""
    proto_hash = E.resolve_protocol(A_R_FILE).hash
    comps = {
        p: REC.ComponentRecord(
            principle=p, status=status, estimator_version=VERSION[p],
            declaration_id="R", observation_stage="source", protocol_id="A-R",
            protocol_hash=proto_hash, c=c, se_c=0.02, df_c=9.0, estimate=c,
            null_mean=0.0, null_sd=0.1, n_null=19, se=0.02, se_df=9.0,
            se_method="jackknife_contiguous_10")
        for p, (status, c) in statuses.items()
    }
    scoring = REC.ScoringRecord(
        scoring_id="R/source/primary", declaration_id="R",
        observation_stage="source", view="source", estimator_form="primary",
        protocol_id="A-R", protocol_hash=proto_hash, components=comps)
    task_id = f"{design}:{system}:s{seed:05d}"
    return REC.TaskRecord(
        task_id=task_id, design=design, family=family, system=system, seed=seed,
        generator_version="mpc-bench-generators/1.0.0", status=REC.TASK_OK,
        scorings=(scoring,),
        simulation={"ts_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
                    "n_nodes": 16, "n_time": 1000, "dt": 0.05},
        provenance=dict(provenance or {})).to_dict()


def synthetic_bench(seeds=range(320, 345), **kw):
    """The two family-A null witnesses on 25 seeds each and the NAS lesion
    on 20, every NAS component ABSENT at c = 0."""
    seeds = list(seeds)
    recs = [synthetic_record(sid, s, {"NAS": ("ABSENT", 0.0)}, **kw)
            for sid in ("N_independent_noise", "N_ar1") for s in seeds]
    recs += [synthetic_record("W_NAS_no_workspace", s, {"NAS": ("ABSENT", 0.0)}, **kw)
             for s in seeds[:20]]
    return recs


def _jsonl(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return path


def _switch_table(path, seeds=range(600, 640)):
    path.write_text(json.dumps([
        {"family": f, "switch": s, "seed": i, "passed": True}
        for f in ("A", "C")
        for s in ("g_b", "ff_only", "c_int", "K", "eta", "e")
        for i in seeds
    ]), encoding="utf-8")
    return path


def test_evaluator_script_on_development_records(tmp_path):
    import scripts.evaluate_hypotheses as BH

    recs = synthetic_bench()
    rec_file = _jsonl(tmp_path / "records.jsonl", recs)
    man = _switch_table(tmp_path / "switches.json")
    rep = BH.run(tmp_path / "out", spec=EXAMPLE, records=[rec_file],
                 manipulation=[man], protocols=[A_R_FILE], development=True)
    assert rep["status"] == HE.DEVELOPMENT_FLAG and rep["n_records"] == len(recs)
    assert rep["outcomes"] == {"checks": SUP, "null_present": SUP,
                               "lesion_absent": SUP, "not_admitted": HE.NOT_RUN}
    assert rep["usability"]["A:g_b"] is True
    assert rep["protocol_hash_mismatches"] == [] and rep["evaluator_errors"] == []
    parts = {p["id"]: p for h in rep["hypotheses"] for p in h["parts"]}
    (cell,) = parts["lesion_absent(a)"]["cells"]
    assert cell["cell"] == "family=A|system=W_NAS_no_workspace"
    assert {c["cell"] for c in parts["null_present(a)"]["cells"]} == {
        "witness:A:N_independent_noise|NAS", "witness:A:N_ar1|NAS"}
    assert parts["lesion_absent(b)"]["outcome"] == HE.REMOVED
    for name in (BH.REPORT_JSON, BH.PARTS_CSV, BH.TALLY_CSV):
        assert (tmp_path / "out" / name).is_file()
    saved = json.loads((tmp_path / "out" / BH.REPORT_JSON).read_text())
    assert saved["outcomes"] == rep["outcomes"] and saved["inputs"]["records"]
    assert saved["spec_sha256"] == HE.spec_sha256(example_spec())
    assert saved["inputs"]["spec"]["sha256"] == hashlib.sha256(
        EXAMPLE.read_bytes()).hexdigest()
    # the command line takes the same inputs, and the hypotheses file is required
    assert BH.main(["--spec", str(EXAMPLE), "--out", str(tmp_path / "cli"),
                    "--records", str(rec_file), "--manipulation", str(man),
                    "--protocols", str(A_R_FILE), "--development"]) == 0
    with pytest.raises(SystemExit):
        BH.main(["--out", str(tmp_path / "no_spec")])


def test_evaluator_script_refusals(tmp_path):
    import scripts.evaluate_hypotheses as BH

    rec_file = _jsonl(tmp_path / "records.jsonl", synthetic_bench()[:20])
    # the hypotheses file must equal its version at the tag
    with pytest.raises(BH.EvaluationRefused, match="tag"):
        BH.check_spec_at_tag(EXAMPLE, "no-such-freeze-tag")
    outside = tmp_path / "spec.json"
    outside.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(BH.EvaluationRefused, match="not a file of this checkout"):
        BH.check_spec_at_tag(outside, "no-such-freeze-tag")
    # a confirmatory evaluation needs the tag
    with pytest.raises(BH.EvaluationRefused, match="needs --freeze-tag"):
        BH.run(tmp_path / "o1", spec=EXAMPLE, records=[rec_file], check_code=False)
    # development records are refused by a confirmatory evaluation
    with pytest.raises(BH.EvaluationRefused, match="seed policy"):
        BH.run(tmp_path / "o2", spec=EXAMPLE, records=[rec_file], freeze_tag=TAG,
               check_code=False)
    broken = dict(synthetic_bench()[0], status="error")
    bad = _jsonl(tmp_path / "bad.jsonl", [broken])
    with pytest.raises(BH.EvaluationRefused, match="schema"):
        BH.run(tmp_path / "o3", spec=EXAMPLE, records=[bad], development=True)


def test_evaluator_script_refuses_untagged_or_cross_split_inputs(tmp_path):
    import scripts.evaluate_hypotheses as BH

    # confirmatory look-alike records (never simulated)
    recs = synthetic_bench(range(20000, 20005), provenance={"freeze_tag": TAG})
    rec_file = _jsonl(tmp_path / "records.jsonl", recs)
    run = dict(spec=EXAMPLE, records=[rec_file], freeze_tag=TAG, check_code=False)
    mech = tmp_path / "mechanism_on.json"
    mech.write_text(json.dumps([{"where": {"design": "witnesses"}, "on": True}]))
    with pytest.raises(BH.EvaluationRefused, match="mechanism-on"):
        BH.run(tmp_path / "m1", mechanism_on=mech, **run)
    # development manipulation checks in a confirmatory evaluation
    man = _switch_table(tmp_path / "switches.json")
    with pytest.raises(BH.EvaluationRefused, match="auxiliary inputs"):
        BH.run(tmp_path / "m2", manipulation=[man], **run)
    # anchors, gates and routes come from the family protocols
    with pytest.raises(BH.EvaluationRefused, match="family protocols"):
        BH.run(tmp_path / "m3", **run)
    # records of another tag
    with pytest.raises(BH.EvaluationRefused, match="without freeze tag"):
        BH.run(tmp_path / "m4", **{**run, "freeze_tag": "another-tag"})


def test_the_evaluator_reads_the_freeze_tag_the_runners_write():
    import scripts.evaluate_hypotheses as BH
    from scripts import iim_validation as V

    code = {"git_sha": "a" * 40, "git_dirty": False, "trees": {},
            "freeze_tag": TAG, "freeze_tag_commit": "b" * 40}
    # the runner's record provenance (flat) and the family-B record
    # provenance (the same fields, built from the run's code identity)
    flat = {"runner_version": "x", "confirmatory": True, "label": None, **code}
    fam_b = V.record_provenance({"code": code}, "confirmatory")
    assert fam_b["freeze_tag"] == TAG and fam_b["confirmatory"] is True
    for prov in (flat, fam_b, {"code": {"freeze_tag": TAG}}):
        recs = synthetic_bench(range(20000, 20003), provenance=prov)
        assert BH.record_freeze_tag(recs[0]) == TAG
        assert BH.check_records(recs, development=False, freeze_tag=TAG) == (
            "confirmatory"
        )
    untagged = synthetic_bench(range(20000, 20003),
                               provenance={**flat, "freeze_tag": None})
    with pytest.raises(BH.EvaluationRefused, match="freeze tag"):
        BH.check_records(untagged, development=False, freeze_tag=TAG)


def test_evaluator_checks_records_against_the_family_protocols():
    import scripts.evaluate_hypotheses as BH
    from impact_pipeline.estimators import ram_pe

    h = "a" * 64
    comps = {
        "RAM": {"protocol_id": "A-R", "protocol_hash": h, "details": {
            "facets_not_applicable": dict(ram_pe.FACETS_NOT_APPLICABLE)}},
        "PDI": {"protocol_id": "A-R", "protocol_hash": h,
                "details": {"bearer": "non_workspace"}},
    }
    rec = {"task_id": "t1", "scorings": [{"scoring_id": "R", "components": comps}]}
    proto = {"A-R": {"hash": h, "estimators": {
        "RAM": {"facets_not_applicable": {
            "F": "no_graded_feedback_referent",
            "G": "no_goal_contingency_referent",
            "speed": "speed_not_in_construct"}},
        "PDI": {"mode": "repertoire", "pdi_bearer": "non_workspace"}}}}
    assert BH.check_protocol_hashes([rec], {}) == []
    assert BH.check_protocol_hashes([rec], proto) == []
    # another hash, or a protocol id the given protocols do not hold
    for other in ({"A-R": {"hash": "b" * 64}}, {"A-H": {"hash": h}}):
        fails = BH.check_protocol_hashes([rec], other)
        assert [f["kind"] for f in fails] == ["hash_mismatch"] * 2
        assert {f["task_id"] for f in fails} == {"t1"}
    # a declaration other than the protocol's
    comps["PDI"]["details"] = {"bearer": "full"}
    fails = BH.check_protocol_hashes([rec], proto)
    assert [f["kind"] for f in fails] == ["pdi_bearer_differs_from_protocol"]
