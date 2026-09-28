import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from impact_pipeline.analysis_bootstrap import (
    bootstrap_ci,
    definedness_summary,
    holm_adjust,
    paired_bootstrap_mean_diff,
    paired_session_test,
    paired_tests_table,
    permutation_test_auc,
)


def _paired_frame(n_subj, effect, between_sd=1.0, within_sd=0.3, seed=0, col="S"):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_subj):
        base = rng.randn() * between_sd
        rows.append(
            {
                "subject": f"s{i:02d}",
                "session": "awake",
                col: base + effect + rng.randn() * within_sd,
            }
        )
        rows.append(
            {
                "subject": f"s{i:02d}",
                "session": "deep",
                col: base + rng.randn() * within_sd,
            }
        )
    return pd.DataFrame(rows)


def test_permutation_two_sided_detects_awake_greater_than_deep():
    # Old code: one-sided in the wrong direction -> AUC 0.13, p ~ 1.0 for this effect.
    df = _paired_frame(15, effect=2.0, seed=1)
    auc, p = permutation_test_auc(df, "S", n_perm=5000)
    assert auc < 0.5  # AUC = P(S_deep > S_awake)
    assert p < 0.01
    _, p_less = permutation_test_auc(df, "S", n_perm=5000, alternative="less")
    _, p_greater = permutation_test_auc(df, "S", n_perm=5000, alternative="greater")
    assert p_less < 0.01 and p_greater > 0.9


def test_permutation_is_seeded_and_reproducible():
    df = _paired_frame(20, effect=0.3, seed=2)
    r1 = permutation_test_auc(df, "S", n_perm=3000)
    r2 = permutation_test_auc(df, "S", n_perm=3000)
    assert r1 == r2
    b1 = bootstrap_ci(df, "S", n_boot=300)
    b2 = bootstrap_ci(df, "S", n_boot=300)
    assert b1 == b2


def test_permutation_null_calibration():
    rejections = 0
    n_sim = 200
    for seed in range(n_sim):
        df = _paired_frame(14, effect=0.0, seed=1000 + seed)
        rejections += (
            permutation_test_auc(df, "S", n_perm=500, random_state=seed)[1] < 0.05
        )
    assert rejections / n_sim <= 0.09


def test_paired_permutation_beats_unpaired_with_large_between_subject_variance():
    # Small consistent within-subject shift hidden by large between-subject spread.
    df = _paired_frame(16, effect=0.5, between_sd=5.0, within_sd=0.2, seed=3)
    _, p_paired = permutation_test_auc(df, "S", n_perm=4000)
    _, p_rows = permutation_test_auc(df, "S", n_perm=4000, paired=False)
    assert p_paired < 0.01
    assert p_rows > 0.2


def test_exact_enumeration_for_small_samples():
    # 5 subjects, perfect separation: only identity and full swap are as extreme.
    df = pd.DataFrame(
        [
            {
                "subject": f"s{i}",
                "session": ses,
                "S": float(i + (10 if ses == "awake" else 0)),
            }
            for i in range(5)
            for ses in ("awake", "deep")
        ]
    )
    det = permutation_test_auc(df, "S", n_perm=10000, return_details=True)
    assert det["exact"] is True
    assert det["auc"] < 0.5
    assert det["p"] == pytest.approx(2 / 2**5)


def test_undefined_rows_and_extra_sessions_are_excluded():
    df = _paired_frame(10, effect=1.0, seed=4)
    df.loc[0, "S"] = (
        np.nan
    )  # subject s00 awake undefined -> subject excluded in paired mode
    extra = pd.DataFrame([{"subject": "s01", "session": "recovery", "S": 99.0}])
    df = pd.concat([df, extra], ignore_index=True)
    det = permutation_test_auc(df, "S", n_perm=2000, return_details=True)
    assert np.isfinite(det["p"])  # old code: ValueError on NaN
    assert det["n_rows_excluded_undefined"] == 1
    assert det["n_subjects_excluded_incomplete"] == 1
    assert det["n_subjects_used"] == 9
    assert det["n_rows_used"] == 18  # 'recovery' row not treated as a negative
    boot = bootstrap_ci(df, "S", n_boot=200, return_details=True)
    assert boot["mode"] == "paired_subject" and boot["n_subjects_used"] == 9
    assert boot["lo"] <= boot["auc"] <= boot["hi"]


def test_bootstrap_resamples_subjects_not_rows():
    df = _paired_frame(12, effect=0.8, seed=5)
    det = bootstrap_ci(df, "S", n_boot=200, return_details=True)
    assert det["mode"] == "paired_subject"
    row_det = bootstrap_ci(
        df.drop(columns="subject"), "S", n_boot=200, return_details=True
    )
    assert row_det["mode"] == "row"


def test_holm_matches_statsmodels_and_ignores_nan():
    p = np.array([0.01, 0.04, 0.03, 0.2, np.nan, 0.005])
    out = holm_adjust(p)
    ref = multipletests(p[np.isfinite(p)], method="holm")[1]
    np.testing.assert_allclose(out[np.isfinite(p)], ref)
    assert np.isnan(out[4])
    assert np.all(np.isnan(holm_adjust([np.nan, np.nan])))


def test_paired_session_test_matches_scipy_and_counts_exclusions():
    df = _paired_frame(12, effect=0.4, seed=6, col="CI")
    df.loc[df.index[-1], "CI"] = np.nan  # last subject's deep CI undefined
    st = paired_session_test(df, "CI")
    piv = df.pivot(index="subject", columns="session", values="CI").dropna()
    t, p = stats.ttest_rel(piv["awake"], piv["deep"])
    assert st["n_pairs"] == 11 and st["n_subjects_excluded"] == 1
    assert st["t"] == pytest.approx(t) and st["p"] == pytest.approx(p)
    diff = piv["awake"] - piv["deep"]
    assert st["dz"] == pytest.approx(diff.mean() / diff.std(ddof=1))
    assert st["mean_a"] == pytest.approx(piv["awake"].mean())


def test_paired_tests_table_holm_family():
    df = _paired_frame(12, effect=0.4, seed=7, col="A")
    df["B"] = _paired_frame(12, effect=0.0, seed=8, col="B")["B"]
    tab = paired_tests_table(df, ["A", "B", "missing"])
    assert list(tab["metric"]) == ["A", "B"]
    np.testing.assert_allclose(tab["p_holm"], multipletests(tab["p"], method="holm")[1])


def test_paired_bootstrap_mean_diff_contains_estimate():
    df = _paired_frame(15, effect=1.0, seed=9)
    res = paired_bootstrap_mean_diff(df, "S", n_boot=1000)
    assert res["lo"] < res["mean_diff"] < res["hi"]
    assert res["n_pairs"] == 15
    assert res == paired_bootstrap_mean_diff(df, "S", n_boot=1000)


def test_definedness_summary_counts():
    df = pd.DataFrame(
        {
            "subject": ["a", "a", "b", "b", "c", "c"],
            "session": ["awake", "deep"] * 3,
            "CI": [1.0, 0.5, np.nan, 0.2, 0.9, np.nan],
            "CI_defined": [True, True, False, True, True, False],
            "CI_missing": ["", "", "SRPI", "", "", "IIM,SRPI"],
        }
    )
    out = definedness_summary(df)
    assert out["n_rows_defined"] == 4 and out["n_rows_undefined"] == 2
    assert out["missing_component_row_counts"] == {"SRPI": 2, "IIM": 1}
    assert out["n_subjects_complete_pairs"] == 1 and out["n_subjects_excluded"] == 2


def test_definedness_counts_runs_not_theta_replicates():
    # compute_synergy_ci repeats every run at each theta; CI does not depend on
    # theta, so one undefined run must be counted once, not once per theta.
    rows = []
    for subj in ("a", "b"):
        for ses in ("awake", "deep"):
            undefined = subj == "b" and ses == "deep"
            for theta in (0.1, 0.5, 0.9):
                rows.append({
                    "subject": subj, "session": ses, "theta": theta,
                    "CI": np.nan if undefined else 1.0,
                    "CI_defined": not undefined,
                    "CI_missing": "SRPI" if undefined else "",
                })
    out = definedness_summary(pd.DataFrame(rows))
    assert out["count_unit"] == "run" and out["n_thetas"] == 3
    assert out["n_rows"] == 4 and out["n_table_rows"] == 12
    assert out["n_rows_defined"] == 3 and out["n_rows_undefined"] == 1
    assert out["missing_component_row_counts"] == {"SRPI": 1}
    assert out["n_subjects_complete_pairs"] == 1


def test_definedness_falls_back_to_rows_when_it_varies_with_theta():
    df = pd.DataFrame({
        "subject": ["a"] * 4, "session": ["awake", "awake", "deep", "deep"],
        "theta": [0.1, 0.9, 0.1, 0.9], "CI": [1.0, np.nan, 1.0, 1.0],
    })
    out = definedness_summary(df)
    assert out["count_unit"] == "row"
    assert out["n_rows"] == 4 and out["n_rows_undefined"] == 1


@pytest.mark.parametrize(
    "flags", [[1.0, 1.0, 0.0, 1.0], ["True", "true", "False", "1"], [1, 1, 0, 1]]
)
def test_definedness_accepts_numeric_and_text_flags(flags):
    df = pd.DataFrame({
        "subject": list("aabb"), "session": ["awake", "deep"] * 2,
        "CI": [1.0, 2.0, np.nan, 1.0], "CI_defined": flags,
    })
    out = definedness_summary(df)
    assert out["n_rows_defined"] == 3 and out["n_rows_undefined"] == 1
