import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from impact_pipeline.model_comparison import compare_models
from impact_pipeline.utils import delong_roc_test


def test_compare_models():
    df=pd.DataFrame({
        'session':['awake','sedation'],
        'S':[0.2,0.8],
        'mean_conn':[0.5,0.6],
        'modularity':[0.1,0.2],
        'pci_fmri':[0.3,0.4]
    })
    out=compare_models(df)
    assert 'mean_conn' in out


def _frame(n=20, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        base = rng.randn()
        for ses, shift in (("awake", 1.0), ("deep", 0.0)):
            rows.append({
                "subject": f"s{i}", "session": ses,
                "S": base + shift + rng.randn() * 0.5,
                "mean_conn": rng.randn(),
                "modularity": base + 0.3 * shift + rng.randn(),
                "lzc": rng.randn(),
            })
    return pd.DataFrame(rows)


def test_compare_models_uses_correct_paired_delong_and_holm():
    df = _frame()
    out = compare_models(df, n_boot=200)
    y = (df.session == "deep").astype(int).to_numpy()
    for m in ("mean_conn", "modularity", "lzc"):
        res = out[m]
        expected = roc_auc_score(y, df.S) - roc_auc_score(y, df[m])
        assert res["delta_auc"] == pytest.approx(expected)
        p_ref = delong_roc_test(y, df.S.values, df[m].values)
        assert res["p_val"] == pytest.approx(p_ref)
        lo, hi = res["delta_auc_ci"]
        assert lo <= res["delta_auc"] <= hi
    p = [out[m]["p_val"] for m in ("mean_conn", "modularity", "lzc")]
    p_holm = [out[m]["p_holm"] for m in ("mean_conn", "modularity", "lzc")]
    np.testing.assert_allclose(p_holm, multipletests(p, method="holm")[1])


def test_compare_models_excludes_other_sessions_and_undefined_rows():
    df = _frame()
    extra = df[df.session == "awake"].assign(session="recovery", S=100.0)
    df2 = pd.concat([df, extra], ignore_index=True)
    df2.loc[0, "S"] = np.nan
    out = compare_models(df2, n_boot=50)
    ref = compare_models(df.drop(index=0), n_boot=50)
    assert out["mean_conn"]["delta_auc"] == pytest.approx(ref["mean_conn"]["delta_auc"])
    assert out["mean_conn"]["n_rows_excluded"] == 1
    assert out["mean_conn"]["n_rows"] == len(df) - 1


def test_compare_models_missing_metric_is_reported():
    out = compare_models(_frame().drop(columns="lzc"), n_boot=20)
    assert np.isnan(out["lzc"]["p_val"]) and "not available" in out["lzc"]["note"]
    assert np.isfinite(out["mean_conn"]["p_val"])


def test_compare_models_alternate_score_column():
    df = _frame().assign(CI=lambda d: d["S"] * 2.0)
    out = compare_models(df, score_col="CI", n_boot=20)
    assert out["mean_conn"]["score"] == "CI"
    ref = compare_models(df, n_boot=20)["mean_conn"]["auc_score"]
    assert out["mean_conn"]["auc_score"] == pytest.approx(ref)


def _opposite_direction_frame(n=30, seed=0, s_effect=1.5, m_effect=-1.5):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        base = rng.randn()
        for ses, state in (("awake", 1.0), ("deep", 0.0)):
            rows.append({
                "subject": f"s{i}", "session": ses,
                "S": base + s_effect * state + rng.randn() * 0.5,
                "mean_conn": base + m_effect * state + rng.randn() * 0.5,
                "modularity": rng.randn(), "lzc": rng.randn(),
            })
    return pd.DataFrame(rows)


def test_equal_discriminability_in_opposite_directions_is_not_a_difference():
    # S higher when awake, mean_conn higher when deep, same strength: the signed
    # ΔAUC is large and "significant", but discriminability does not differ.
    res = compare_models(_opposite_direction_frame(), n_boot=400)["mean_conn"]
    assert res["auc_score"] < 0.25 and res["auc_metric"] > 0.75
    assert res["delta_auc"] < -0.5 and res["p_val"] < 1e-6
    assert abs(res["delta_discrimination"]) < 0.1
    lo, hi = res["delta_discrimination_ci"]
    assert lo < 0 < hi
    assert res["p_discrimination"] > 0.05
    # Oriented DeLong equals the DeLong test on sign-aligned scores.
    df = _opposite_direction_frame()
    y = (df.session == "deep").astype(int).to_numpy()
    p_ref = delong_roc_test(y, -df.S.to_numpy(), df.mean_conn.to_numpy())
    assert res["p_discrimination"] == pytest.approx(p_ref)


def test_discriminability_detects_a_better_score():
    res = compare_models(
        _opposite_direction_frame(s_effect=2.0, m_effect=-0.2, seed=1), n_boot=400
    )
    r = res["mean_conn"]
    assert r["delta_discrimination"] > 0.2
    assert r["delta_discrimination_ci"][0] > 0
    assert r["p_discrimination"] < 0.01
    p = [res[m]["p_discrimination"] for m in ("mean_conn", "modularity", "lzc")]
    np.testing.assert_allclose(
        [res[m]["p_discrimination_holm"] for m in ("mean_conn", "modularity", "lzc")],
        multipletests(p, method="holm")[1],
    )
