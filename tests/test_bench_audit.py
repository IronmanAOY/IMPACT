"""Rule audit on estimated component statuses (MPC-Bench v2): the v2
construct-scale status rule, missingness scenarios, label noise, coverage,
P(verdict | class) and risk-coverage curves on synthetic bench records with
known answers, plus a smoke run of scripts/benchmark_attribution_rules.py."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.bench import audit as A
from impact_pipeline.bench import rules as R

P = A.PRINCIPLES


def _record(i, bits, seed, rng, se=0.02, missing_se=False, gen="family_a"):
    """Estimates on a scale with null mean 0, null SD 0.1 and reference ~1:
    present mechanisms near 1, absent ones near 0."""
    comps = {}
    for j, p in enumerate(P):
        m = (1.0 if bits[j] else 0.0) + 0.02 * rng.standard_normal()
        comps[p] = {
            "estimate": m,
            "null_mean": 0.0,
            "null_sd": 0.1,
            "n_null": 100,
            "se": None if missing_se else se,
            "defined": True,
            "statistic": "raw",
        }
    return {
        "task_id": f"t{i:04d}",
        "status": "ok",
        "design": "factorial",
        "generator": gen,
        "cell_id": "b" + "".join(map(str, bits)),
        "seed": seed,
        "intended_bits": list(bits),
        "components": comps,
        "markers": {
            "LZc": float(sum(bits)) + 0.1 * rng.standard_normal(),
            "PhiR_bits": 0.1 * rng.standard_normal(),
        },
    }


def _records(n_seeds=6, **kw):
    rng = np.random.default_rng(0)
    cells = [(1, 1, 1, 1, 1)] + [
        tuple(0 if k == j else 1 for k in range(5)) for j in range(5)
    ]
    recs, i = [], 0
    for seed in range(n_seeds):
        for bits in cells:
            recs.append(_record(i, bits, seed, rng, **kw))
            i += 1
    return recs


def test_construct_scale_and_v2_status_known_answers():
    c, se = R.construct_scale(
        1.5, 0.5, 2.5, se_estimate=0.2, null_sd=0.4, n_null=16, se_reference=0.0
    )
    assert c == pytest.approx(0.5)
    # se_c^2 = (0.2^2 + 0.5^2 * (0.4/4)^2) / 2^2
    assert se == pytest.approx(math.sqrt((0.04 + 0.25 * 0.01) / 4.0))
    c_bad, _ = R.construct_scale(1.0, 1.0, 1.0)
    assert np.isnan(c_bad)  # reference not above the null: invalid anchors
    st = R.component_status_c(
        np.array([[0.9, 0.0, 0.2, -0.5, 0.9]]),
        np.array([[0.1, 0.02, 0.1, 0.1, 0.0]]),
    )
    assert list(st[0]) == [R.PRESENT, R.ABSENT, R.UNDEFINED, R.ABSENT, R.UNDEFINED]
    # A zero SE counts only for exact (known-TPM) computations.
    st = R.component_status_c(np.array([[0.9]]), np.array([[0.0]]), exact=True)
    assert st[0, 0] == R.PRESENT
    with pytest.raises(ValueError):
        R.component_status_c(np.zeros((1, 5)), 0.1, z_present=0.1, delta_absent=0.2)
    out = R.rule_impact_c(
        np.array([[0.9] * 5, [0.9, 0.9, 0.9, 0.9, 0.0], [0.9, 0.9, 0.9, 0.9, np.nan]]),
        0.05,
    )
    assert list(out.decision) == [R.MPC_CONSISTENT, R.EXCLUDED, R.UNDETERMINED]


def test_v2_rule_is_missingness_safe_on_random_vectors():
    rng = np.random.default_rng(3)
    n = 20000
    C = rng.uniform(-0.5, 1.5, size=(n, 5))
    S = rng.uniform(0.01, 0.4, size=(n, 5))
    full = R.rule_impact_c(C, S).decision
    Cm = C.copy()
    Cm[rng.random(C.shape) < 0.3] = np.nan
    miss = R.rule_impact_c(Cm, S).decision
    # Missingness never creates MPC_CONSISTENT and never flips a determinate
    # verdict into the other determinate verdict.
    assert not np.any((miss == R.MPC_CONSISTENT) & (full != R.MPC_CONSISTENT))
    flipped = (miss != R.UNDETERMINED) & (full != R.UNDETERMINED) & (miss != full)
    assert not flipped.any()


def test_audit_known_answers_without_missingness():
    res = A.audit(_records(), scenarios=("none",), label_noise=(0.0,))
    cov = res["coverage"].set_index("rule")
    assert cov.loc["impact_c", "coverage"] == pytest.approx(1.0)
    assert cov.loc["impact_c", "selective_risk"] == pytest.approx(0.0)
    assert cov.loc["impact_c", "p_excluded_given_positive"] == pytest.approx(0.0)
    # A compensatory rule accepts single deficits.
    assert cov.loc["count_4", "p_consistent_given_negative"] == pytest.approx(1.0)
    assert cov.loc["weakest_link", "selective_risk"] == pytest.approx(0.0)
    pvc = res["p_verdict_given_class"]
    sums = pvc.groupby(["rule", "scenario", "label_noise", "class"])["p"].sum()
    assert np.allclose(sums, 1.0)
    assert {"LZc", "PhiR_bits", "logistic", "dcm_naive_bayes"} <= set(cov.index)
    assert res["settings"]["n_with_truth"] == 36


def test_missingness_scenarios_make_the_v2_rule_abstain_not_err():
    res = A.audit(_records(), scenarios=A.SCENARIOS, label_noise=(0.0,))
    cov = res["coverage"].set_index(["rule", "scenario"])
    for scen in A.SCENARIOS:
        assert cov.loc[("impact_c", scen), "selective_risk"] in (0.0,) or np.isnan(
            cov.loc[("impact_c", scen), "selective_risk"]
        )
    dec = res["decisions"]
    sub = dec[(dec.rule == "impact_c") & (dec.scenario == "RAM+SRPI_missing")]
    # Positives cannot be confirmed without RAM/SRPI; RAM- and SRPI-deficit
    # systems become undetermined, PDI/NAS/IIM deficits stay excluded.
    assert set(sub.loc[sub["class"] == "all_present", "decision"]) == {R.UNDETERMINED}
    for p in ("RAM", "SRPI"):
        assert set(sub.loc[sub["class"] == f"single_deficit:{p}", "decision"]) == {
            R.UNDETERMINED
        }
    for p in ("PDI", "NAS", "IIM"):
        assert set(sub.loc[sub["class"] == f"single_deficit:{p}", "decision"]) == {
            R.EXCLUDED
        }
    assert cov.loc[("impact_c", "RAM+SRPI_missing"), "coverage"] == pytest.approx(0.5)
    # The union rule skips missing components and keeps accepting deficits.
    assert cov.loc[("union", "RAM+SRPI_missing"), "p_consistent_given_negative"] == 1.0


def test_missing_sampling_se_is_undetermined():
    res = A.audit(_records(missing_se=True), scenarios=("none",), label_noise=(0.0,))
    dec = res["decisions"]
    assert set(dec.loc[dec.rule == "impact_c", "decision"]) == {R.UNDETERMINED}


def test_risk_coverage_curve_properties():
    truth = np.array([1, 0, 1, 0, 1, np.nan])
    dec = np.array(
        [
            R.MPC_CONSISTENT,
            R.MPC_CONSISTENT,
            R.EXCLUDED,
            R.EXCLUDED,
            R.UNDETERMINED,
            R.MPC_CONSISTENT,
        ],
        dtype=object,
    )
    conf = np.array([3.0, 0.5, 0.1, 2.0, np.nan, 9.0])
    rc = A.risk_coverage(dec, conf, truth)
    # Ranked 3.0 (right), 2.0 (right), 0.5 (wrong), 0.1 (wrong).
    assert list(rc["risk"]) == [0.0, 0.0, 1 / 3, 0.5]
    assert list(rc["coverage"]) == [0.2, 0.4, 0.6, 0.8]
    assert A.risk_coverage(dec[:0], conf[:0], truth[:0]).empty


def test_label_noise_and_missing_masks_are_seeded():
    y = np.array([1.0, 0.0] * 50 + [np.nan])
    a = A.flip_labels(y, 0.2, seed=1)
    assert np.array_equal(a, A.flip_labels(y, 0.2, seed=1), equal_nan=True)
    assert 5 <= np.sum(a[:-1] != y[:-1]) <= 35 and np.isnan(a[-1])
    m = A.missing_mask("random20", (500, 5), seed=2)
    assert 0.15 < m.mean() < 0.25
    assert A.missing_mask("SRPI_missing", (3, 5))[:, 4].all()
    with pytest.raises(ValueError):
        A.missing_mask("half", (3, 5))
    res = A.audit(_records(), scenarios=("none",), label_noise=(0.0, 0.2))
    assert set(res["coverage"]["label_noise"]) == {0.0, 0.2}


def test_classes_and_truth():
    rec = {
        "design": "adversarial",
        "generator": "adversarial_reflex_arc",
        "cell_id": "reflex_arc",
        "intended_bits": [0, 0, 0, 0, 0],
    }
    assert A.class_of(rec) == "adversarial:reflex_arc" and A.truth_of(rec) == 0
    wb = {
        "design": "whole_brain",
        "generator": "whole_brain_eeg",
        "cell_id": "G1",
        "intended_bits": [],
    }
    assert A.class_of(wb).startswith("whole_brain:") and A.truth_of(wb) is None
    pw = {
        "generator": "patchwork",
        "sweep_level": 0.4,
        "bearer_mode": "principle",
        "intended_bits": [1] * 5,
    }
    assert A.class_of(pw) == "patchwork:lambda=0.4:principle"
    assert A.truth_of(pw) is None
    assert A.truth_of({"intended_bits": [1, 1, 1, 1, 0]}, ("RAM", "PDI")) == 1


def test_benchmark_script_on_existing_results(tmp_path):
    from scripts import benchmark_attribution_rules as script

    res_dir = tmp_path / "estimates"
    res_dir.mkdir()
    with open(res_dir / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in _records(n_seeds=4):
            r["provenance"] = {"code_version": {"git_sha": "x" * 40}}
            fh.write(json.dumps(r) + "\n")
    out = tmp_path / "audit"
    rc = script.main(
        ["--results", str(res_dir), "--out", str(out), "--label-noise", "0,0.1"]
    )
    assert rc == 0
    summary = json.loads((out / "audit_summary.json").read_text())
    assert summary["development_only"] is True
    assert "not a result" in summary["notice"]
    for name in (
        "audit_decisions.csv",
        "audit_p_verdict_given_class.csv",
        "audit_coverage.csv",
        "audit_risk_coverage.csv",
    ):
        assert (out / name).exists(), name
    cov = pd.read_csv(out / "audit_coverage.csv")
    assert set(cov["scenario"]) == set(A.SCENARIOS)
