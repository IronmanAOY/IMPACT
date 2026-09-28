"""Tests of scripts/audit_aggregation.py: the legacy compute_CI is loaded from
git commit 21ce76a and its compensation / definedness failures are reproduced."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.audit_aggregation as aa
from impact_pipeline.bench.rules import power_mean

REPO = Path(__file__).resolve().parents[1]


def _has_legacy_commit():
    proc = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                           f"{aa.LEGACY_COMMIT}^{{commit}}"], capture_output=True)
    return proc.returncode == 0


needs_git = pytest.mark.skipif(not _has_legacy_commit(),
                               reason="legacy commit 21ce76a not in this checkout")


@pytest.fixture(scope="module")
def legacy():
    if not _has_legacy_commit():
        pytest.skip("legacy commit 21ce76a not in this checkout")
    return aa.load_legacy_compute_ci(mode="module")


@needs_git
def test_legacy_source_identity_is_verified(tmp_path):
    src, info = aa.read_legacy_source()
    assert info["git_blob_sha1"] == aa.LEGACY_BLOB_SHA1
    assert "def compute_CI(" in src
    copy = tmp_path / "legacy_mpc_metrics.py"
    copy.write_text(src, encoding="utf-8")
    _, info2 = aa.read_legacy_source(source_file=copy)
    assert info2["sha256"] == info["sha256"]
    bad = tmp_path / "tampered.py"
    bad.write_text(src.replace("hard criterion", "soft criterion"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        aa.read_legacy_source(source_file=bad)


def test_legacy_module_is_not_registered_and_loads_verbatim(legacy):
    fn, info = legacy
    assert info["load_mode"] == "module"
    assert info["module_name"] not in sys.modules
    assert info["git_blob_sha1"] == aa.LEGACY_BLOB_SHA1
    assert "hard criterion" in (fn.__doc__ or "")


@needs_git
def test_ast_mode_gives_the_same_function_values():
    fn_ast, info = aa.load_legacy_compute_ci(mode="ast")
    fn_mod, _ = aa.load_legacy_compute_ci(mode="module")
    assert info["load_mode"] == "ast"
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = rng.lognormal(size=5)
        v[rng.random(5) < 0.2] = 0.0
        assert fn_ast(*v) == fn_mod(*v)


def test_tolerant_import_stubs_missing_modules():
    report = {"stubbed_modules": [], "stubbed_names": []}
    src = ("import numpy as np\n"
           "from impact_nonexistent_pkg.sub import thing\n"
           "from impact_pipeline.hardware_backend import not_a_real_name\n"
           "def f():\n    return np.float64(2.0)\n")
    mod = aa._exec_module(src, "impact_legacy_test_mod", aa._tolerant_import(report))
    assert mod.f() == 2.0
    assert any("impact_nonexistent_pkg" in m for m in report["stubbed_modules"])
    assert "impact_pipeline.hardware_backend.not_a_real_name" in report["stubbed_names"]
    with pytest.raises(RuntimeError, match="not available"):
        mod.thing()


def test_compensation_and_definedness_are_reproduced(legacy):
    fn, _ = legacy
    # (0.01 * 10**4) ** (1/5) = 100 ** 0.2
    assert fn(0.01, 10, 10, 10, 10) == pytest.approx(100 ** 0.2, rel=1e-9)
    assert fn(0.01, 10, 10, 10, 10) > 1.0
    assert fn(1e-4, 10, 10, 10, 10) == pytest.approx(1.0, rel=1e-8)
    assert fn(float("nan"), 1, 1, 1, 1) == 0.0
    assert fn(0.0, 1, 1, 1, 1) == 0.0
    assert fn(1, 1, 1, 1, 1, defined={"RAM": False}) == 0.0
    assert fn(-0.5, 1, 1, 1, 1) == 0.0
    # eps discontinuity: 0 at 0, (1e-13 + 1e-12) ** 0.2 just above 0
    assert fn(1e-13, 1, 1, 1, 1) == pytest.approx((1.1e-12) ** 0.2, rel=1e-6)
    table = aa.compensation_table(fn).set_index("case")
    assert bool(table.loc["compensation", "above_reference"])
    assert table.loc["compensation", "capped_geometric"] == pytest.approx(0.01 ** 0.2)
    assert table.loc["compensation", "weakest_link"] == pytest.approx(0.01)
    assert math.isnan(table.loc["undefined_component", "capped_geometric"])
    assert (table.loc["undefined_component", "legacy_CI"]
            == table.loc["measured_zero", "legacy_CI"])


@pytest.mark.parametrize("p", [0.0, -1.0, 1.0, 2.0])
@pytest.mark.parametrize("w", [0.1, 0.2, 1.0 / 3.0])
@pytest.mark.parametrize("m", [1.0, 2.0, 10.0])
def test_implied_floor_is_the_exact_threshold(p, w, m):
    t = 0.5
    f = aa.implied_floor(t, w, m, p)
    weights = np.array([w] + [(1 - w) / 4] * 4)
    if f == 0.0:  # full compensation: the threshold holds with the component at 0
        assert power_mean(np.array([[0.0] + [m] * 4]), p, cap=None,
                          weights=weights)[0] >= t or p <= 0
        return
    assert math.isfinite(f)
    at = power_mean(np.array([[f] + [m] * 4]), p, cap=None, weights=weights)[0]
    assert at == pytest.approx(t, rel=1e-9)
    below = power_mean(np.array([[f * 0.99] + [m] * 4]), p, cap=None,
                       weights=weights)[0]
    assert below < t


def test_implied_floor_headline_values():
    assert aa.implied_floor(0.5, 0.2) == pytest.approx(0.5 ** 5)
    assert aa.implied_floor(0.5, 0.2, others=10.0) == pytest.approx(
        (0.5 / 10 ** 0.8) ** 5)
    assert aa.implied_floor(0.5, 0.2, others=1.0, p=1.0) == 0.0  # arithmetic
    tab = aa.implied_floor_table()
    assert tab["fully_compensable"].any()


def test_aggregation_grid_uses_the_legacy_function(legacy):
    fn, _ = legacy
    grid = aa.aggregation_grid(fn, n=9)
    assert set(grid["rule"]) == set(aa.GRID_RULES)
    leg = grid[grid["rule"] == "legacy_CI"]
    row = leg[(leg["x"] == 2.0) & (leg["y"] == 2.0)].iloc[0]
    assert row["value"] == pytest.approx(fn(2, 2, 1, 1, 1))
    capped = grid[(grid["rule"] == "capped_geometric") & (grid["x"] == 2.0)
                  & (grid["y"] == 2.0)].iloc[0]
    assert capped["value"] == pytest.approx(1.0)


def test_aggregate_panel_missing_policies():
    X = np.array([[1.0, 1.0, np.nan, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5, 0.5]])
    w = np.full(5, 0.2)
    assert aa.aggregate_panel(X, w, 0.0, missing="legacy_zero")[0] == 0.0
    assert np.isnan(aa.aggregate_panel(X, w, 0.0, missing="exclude")[0])
    assert aa.aggregate_panel(X, w, 0.0, missing="skip")[0] == pytest.approx(1.0)
    assert aa.aggregate_panel(X, w, 1.0, missing="skip")[1] == pytest.approx(0.5)


def test_sensitivity_analysis_is_deterministic_and_well_formed():
    panel = aa.synthetic_panel(15, seed=2)
    u1, i1, m1 = aa.sensitivity_analysis(panel, n_base=24, n_boot=5, seed=4)
    u2, i2, _ = aa.sensitivity_analysis(panel, n_base=24, n_boot=5, seed=4)
    pd.testing.assert_frame_equal(u1, u2)
    pd.testing.assert_frame_equal(i1, i2)
    assert list(i1["factor"]) == [g for g, _ in aa.FACTOR_GROUPS]
    assert m1["n_model_runs"] == 24 * (len(aa.FACTOR_GROUPS) + 2)
    assert (u1["decision_flip_rate"].between(0, 1)).all()
    assert (u1["rank_p05"] <= u1["rank_p95"]).all()
    # a unit with a missing component is undefined under 'exclude' runs
    miss = u1[u1["n_missing_components"] > 0]
    if len(miss):
        assert (miss["undefined_rate"] > 0).all()


def test_baseline_configuration_has_zero_rank_shift():
    panel = aa.synthetic_panel(10, seed=3)
    X = panel[list(aa.COMPONENTS)].to_numpy(float)
    base = aa.aggregate_panel(X, np.full(5, 0.2), 0.0, None, "legacy_zero", None)
    r = aa._ranks(base)
    assert aa._rank_shift(r, r) == 0.0


@needs_git
def test_run_audit_writes_outputs_with_provenance(tmp_path):
    summary = aa.run_audit(tmp_path, n_base=16, n_boot=5, n_units=8, grid_n=7)
    for name in ("legacy_compensation.csv", "implied_floors.csv",
                 "aggregation_grid.csv", "sensitivity_units.csv",
                 "sensitivity_indices.csv", "audit_aggregation.json"):
        assert (tmp_path / name).is_file()
    data = json.loads((tmp_path / "audit_aggregation.json").read_text())
    assert data["headline"]["undefined_equals_measured_zero"] is True
    assert data["headline"]["CI(0.01,10,10,10,10)"] == pytest.approx(100 ** 0.2)
    assert data["legacy"]["git_blob_sha1"] == aa.LEGACY_BLOB_SHA1
    assert data["provenance"]["script_sha256"]
    assert summary["panel"]["source"] == "synthetic_panel"
    # user panel
    panel = aa.synthetic_panel(6, seed=1)
    csv = tmp_path / "panel.csv"
    panel.to_csv(csv, index=False)
    out2 = tmp_path / "user"
    s2 = aa.run_audit(out2, components_csv=csv, n_base=8, n_boot=2, grid_n=5)
    assert s2["panel"]["sha256"]
