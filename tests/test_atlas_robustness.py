import numpy as np
import pandas as pd
import pytest
from impact_pipeline import atlas_robustness

pytestmark = pytest.mark.filterwarnings("ignore:Precision loss occurred:RuntimeWarning")


def test_atlas_robustness(monkeypatch):
    monkeypatch.setattr(
        atlas_robustness, "compute_synergy_ci",
        lambda *a, **k: pd.DataFrame(columns=['subject', 'session', 'theta', 'S']))
    res = atlas_robustness.atlas_check("dummy", ["aal90"])
    assert isinstance(res, dict)
    assert 'aal90' in res


def _curves():
    rows = []
    # Each subject's S is flat over theta; subjects differ in level (0 vs 10).
    for sidx, level in enumerate((0.0, 10.0)):
        for ses, shift in (("awake", 1.0), ("deep", 0.0)):
            for th in (0.1, 0.2, 0.3):
                for run in (1, 2):
                    rows.append({"subject": f"s{sidx}", "session": ses, "theta": th,
                                 "S": level + shift, "NAS": level + shift + 0.01 * run})
    return pd.DataFrame(rows)


def test_roughness_is_within_subject(monkeypatch):
    calls = {}

    def fake(*a, **k):
        calls.update(k)
        return _curves()

    monkeypatch.setattr(atlas_robustness, "compute_synergy_ci", fake)
    res = atlas_robustness.atlas_check(
        "dummy", atlases=("aal90",), mpc_metrics=("NAS",), condition="rest",
        iim_kwargs={"iim_max_nodes": 4}, compute_kwargs={"dataset_id": "dsX"},
    )["aal90"]
    # Old code sorted all subjects' rows together -> roughness 10 for flat curves.
    assert res["roughness"]["awake"] == pytest.approx(0.0)
    assert res["roughness_n_subjects"]["awake"] == 2
    assert calls["condition"] == "rest" and calls["iim_max_nodes"] == 4
    assert calls["dataset_id"] == "dsX"
    assert "p_holm" in res["metrics"]["S"]


def test_missing_atlas_is_skipped_with_reason(monkeypatch):
    def fake(data_dir, atlas, **k):
        if atlas == "shen268":
            raise FileNotFoundError("No time-series for s1/awake")
        return _curves()

    monkeypatch.setattr(atlas_robustness, "compute_synergy_ci", fake)
    res = atlas_robustness.atlas_check(
        "dummy", atlases=("aal90", "shen268"), mpc_metrics=("NAS",)
    )
    assert "roughness" in res["aal90"]
    assert "missing time series" in res["shen268"]["skipped"]


def test_ci_not_assessed_note_when_components_missing(monkeypatch):
    monkeypatch.setattr(
        atlas_robustness, "compute_synergy_ci", lambda *a, **k: _curves()
    )
    res = atlas_robustness.atlas_check(
        "dummy", atlases=("aal90",), mpc_metrics=("PDI", "NAS"), compute_ci=True
    )["aal90"]
    assert res["ci_enabled"] is False
    assert any("CI not assessed" in n for n in res["notes"])
    assert np.isfinite(res["metrics"]["S"]["delta_a_minus_b"])
