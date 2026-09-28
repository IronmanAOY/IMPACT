"""Tests of scripts/necessity_power.py, scripts/simulate_rule_recovery.py and
scripts/null_calibration.py (known answers, determinism, the v2 evidence rule
under a declared protocol)."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as t_dist

import scripts.necessity_power as npw
import scripts.null_calibration as ncal
import scripts.simulate_rule_recovery as srr
from impact_pipeline import necessity as nc


# --------------------------------------------------------------------------
# necessity_power
# --------------------------------------------------------------------------
def _params(**kw):
    p = dict(npw.DEFAULTS)
    p.update(tau=(0.05,), label_noise=(0.0, 0.05), base_rate=(0.5,), coverage=(1.0,),
             false_absent=(0.0,), pi_viol=(0.25,), n_values=(20, 60, 100, 160))
    p.update(kw)
    return p


def test_power_table_matches_exact_outcome_probabilities():
    pw = npw.power_table(_params())
    for _, r in pw.sample(8, random_state=0).iterrows():
        ex = nc.necessity_outcome_probabilities(
            int(r["n"]), r["q_positive"], r["tau"], r["alpha"],
            n_negative=int(r["n_negative"]), q_negative=r["q_negative"])
        assert r["P_SUPPORTED"] == pytest.approx(ex["P_SUPPORTED"], abs=1e-12)
        assert r["P_FALSIFIED"] == pytest.approx(ex["P_FALSIFIED"], abs=1e-12)
    holds = pw[pw["truth"] == "necessity_holds"]
    assert (holds["power"] == holds["P_SUPPORTED"]).all()
    # the (almost) noise-free necessity cannot be supported below n = 59
    small = holds[(holds["n"] == 20) & (holds["label_noise"] == 0.0)]
    assert float(small["P_SUPPORTED"].iloc[0]) == 0.0


def test_min_n_table_and_unreachable_targets():
    params = _params(label_noise=(0.0, 0.2), n_values=tuple(range(10, 301, 10)))
    mn = npw.min_n_table(npw.power_table(params), (0.8,))
    ok = mn[(mn["truth"] == "necessity_holds") & (mn["label_noise"] == 0.0)].iloc[0]
    assert ok["n_first"] == 60 and ok["n_stable"] >= ok["n_first"]
    # label noise 0.2 x 0.6 x 0.9 = 0.108 > tau: support is asymptotically wrong
    bad = mn[(mn["truth"] == "necessity_holds") & (mn["label_noise"] == 0.2)].iloc[0]
    assert math.isnan(bad["n_stable"]) and not bad["asymptotically_correct"]


def test_run_writes_outputs_and_monte_carlo_agrees(tmp_path):
    res = npw.run(tmp_path, _params(label_noise=(0.02,), n_values=(40, 80)),
                  mc_reps=400, seed=1, mc_n=(80,))
    pw = res["power"].dropna(subset=["MC_SUPPORTED"])
    assert len(pw) == 2
    for o in nc.NECESSITY_OUTCOMES:
        assert np.allclose(pw[f"P_{o}"], pw[f"MC_{o}"], atol=0.08)
    for name in ("necessity_thresholds.csv", "necessity_power.csv",
                 "necessity_min_n.csv", "necessity_power.json"):
        assert (tmp_path / name).is_file()
    assert res["summary"]["min_n_support_reachable"]["0.05"] == 59


def test_cli_parsing(tmp_path):
    assert npw.main(["--out", str(tmp_path), "--tau", "0.1", "--label-noise", "0",
                     "--base-rate", "0.5", "--coverage", "1", "--false-absent", "0",
                     "--pi-viol", "0.3", "--n-values", "10:50:10,75"]) == 0
    pw = pd.read_csv(tmp_path / "necessity_power.csv")
    assert sorted(pw["n"].unique()) == [10, 20, 30, 40, 50, 75]
    with pytest.raises(ValueError, match="positive"):
        npw.power_table(_params(n_values=(0, 10)))


# --------------------------------------------------------------------------
# simulate_rule_recovery
# --------------------------------------------------------------------------
def test_power_means_known_values_and_stability():
    X = np.array([[0.2, 0.4, 0.8, 0.5, 1.0]])
    pm = srr.power_means(X, [-math.inf, -1, 0, 1, 50, -50], cap=None)[:, 0]
    assert pm[0] == pytest.approx(0.2)
    assert pm[1] == pytest.approx(5 / np.sum(1 / X))
    assert pm[2] == pytest.approx(np.exp(np.mean(np.log(X))))
    assert pm[3] == pytest.approx(X.mean())
    assert 0.2 < pm[5] < 0.25 and 0.9 < pm[4] <= 1.0
    assert srr.power_means(X * 3, [1], cap=1.0)[0, 0] == pytest.approx(0.92)


def test_component_sd_quadrature():
    rng = np.random.default_rng(0)
    sd = srr.component_sd(1.0)
    assert sd == pytest.approx(np.std(1 / (1 + np.exp(-rng.standard_normal(400000)))),
                               rel=0.01)


@pytest.mark.parametrize("p_true", [-1.0, 0.0, 1.0])
def test_exponent_is_recovered_without_measurement_error(p_true):
    cell, _ = srr.run_cell(p_true, 1.0, 0.3, 500, reps=12, seed=3)
    assert abs(cell["median_p_hat"] - p_true) <= 0.5
    assert cell["coverage"] >= 0.75


def test_measurement_error_biases_towards_compensation_and_n_helps():
    noisy, _ = srr.run_cell(-1.0, 0.6, 0.3, 300, reps=12, seed=4)
    clean, _ = srr.run_cell(-1.0, 1.0, 0.3, 300, reps=12, seed=4)
    assert noisy["bias"] > clean["bias"] + 0.5
    assert noisy["class_accuracy"] < clean["class_accuracy"]
    small, _ = srr.run_cell(0.0, 1.0, 0.3, 50, reps=16, seed=5)
    large, _ = srr.run_cell(0.0, 1.0, 0.3, 500, reps=16, seed=5)
    assert large["rmse"] < small["rmse"]


def test_binary_outcome_and_run_outputs(tmp_path):
    cell, rep = srr.run_cell(1.0, 1.0, 0.0, 200, reps=3, seed=1, outcome="binary")
    assert np.isfinite(rep["p_hat_finite"]).all()
    a = srr.run(tmp_path / "a", p_true=(0.0,), reliability=(1.0,), rho=(0.3,), n=(60,),
                reps=3, seed=2)
    b = srr.run(tmp_path / "b", p_true=(0.0,), reliability=(1.0,), rho=(0.3,), n=(60,),
                reps=3, seed=2)
    pd.testing.assert_frame_equal(a["cells"], b["cells"])
    assert (tmp_path / "a" / "rule_recovery_confusion.csv").is_file()
    assert srr.p_class(-math.inf) == "min_like" and srr.p_class(0.2) == "geometric_like"


# --------------------------------------------------------------------------
# null_calibration: null data, design and evidence adapters
# --------------------------------------------------------------------------
def test_null_timeseries_properties():
    rng = np.random.default_rng(0)
    x = ncal.null_timeseries("ar1", 4, 6000, rng, ar_coef=0.5)
    assert x.shape == (4, 6000)
    assert np.allclose(x.mean(axis=1), 0, atol=1e-9) and np.allclose(x.std(axis=1), 1)
    r1 = np.mean([np.corrcoef(v[:-1], v[1:])[0, 1] for v in x])
    assert r1 == pytest.approx(0.5, abs=0.05)
    p = ncal.null_timeseries("pink", 4, 8192, rng, pink_beta=1.0)
    f = np.fft.rfftfreq(8192)[1:2000]
    psd = np.mean(np.abs(np.fft.rfft(p, axis=1)[:, 1:2000]) ** 2, axis=0)
    slope = np.polyfit(np.log(f), np.log(psd), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.3)
    s = ncal.null_timeseries("surrogate_iid", 6, 4000, rng)
    c = np.corrcoef(s)
    assert np.max(np.abs(c[np.triu_indices(6, 1)])) < 0.15
    with pytest.raises(ValueError):
        ncal.null_timeseries("white", 2, 100, rng)


def test_null_events_satisfy_the_task_and_agency_contracts():
    from impact_pipeline.event_parsing import (
        events_table_to_bundle,
        validate_srpi_agency_contract,
    )

    rng = np.random.default_rng(2)
    ev = ncal.null_events(2400, 0.05, rng)
    b = events_table_to_bundle(ev)
    assert len(b["goal_onsets"]) >= 6 and len(b["feedback_onsets"]) >= 6
    rep = validate_srpi_agency_contract(b["agency_events"], tr=0.05)
    assert rep["valid"], rep["violations"]
    assert rep["n_pairs"] >= 3
    other = ev[ev["trial_type"] == "other_caused"]
    selfs = ev.set_index("event_id")
    for _, r in other.iterrows():
        s = selfs.loc[r["yoked_to"]]
        assert s["value"] == r["value"] and s["phase_bin"] == r["phase_bin"]


def test_expected_exchangeable_rate_and_local_rule_agree():
    k = 19
    exp = ncal.expected_exchangeable_rate(k)
    assert exp == pytest.approx(t_dist.sf(1.645 / math.sqrt(1 + 1 / k), k - 1))
    assert ncal.expected_exchangeable_rate(10000) == pytest.approx(0.05, abs=0.002)
    rng = np.random.default_rng(0)
    hits = 0
    reps = 20000
    for _ in range(reps):
        draws = rng.standard_normal(k + 1)
        null = draws[1:]
        st = ncal.local_status(draws[0], null.mean(), null.std(ddof=1))
        hits += st == "PRESENT"
    assert hits / reps == pytest.approx(exp, abs=0.006)


NCAL_FAMILIES = {"RAM": "onset_jitter", "PDI": "circular_shift",
                 "NAS": "block_circular_shift", "IIM": "circular_shift",
                 "SRPI": "yoked_label_permutation"}


def _ncal_protocol(scale="excess", value=1.0):
    from impact_pipeline import evidence as ev

    return ev.Protocol(
        null_families=NCAL_FAMILIES,
        reference={"kind": "external", "scale": scale,
                   "values": {p: value for p in ev.PRINCIPLES},
                   "se": {p: 0.05 for p in ev.PRINCIPLES}},
        name="ncal-test")


def test_classify_is_the_v2_evidence_rule_under_the_protocol():
    from impact_pipeline import evidence as ev

    proto = _ncal_protocol()
    rng = np.random.default_rng(11)
    n_t = 0
    for _ in range(3000):
        comp = {"estimate": rng.normal(0.5, 0.6), "null_mean": rng.normal(0, 0.2),
                "null_sd": float(rng.exponential(0.2)),
                "n_null": int(rng.integers(0, 30)),
                "se": float(rng.choice([0.0, rng.exponential(0.2)])),
                "se_n": int(rng.choice([0, 3, 5])), "defined": True,
                "null_family": "block_circular_shift"}
        st, reason, impl, a = ncal.classify("NAS", comp, proto)
        se_df = comp["se_n"] - 1 if comp["se_n"] >= 2 else None
        ref = ev.component_assessment(ev.ComponentEvidence(
            "NAS", comp["estimate"], comp["null_mean"], comp["null_sd"],
            se=comp["se"], n_null=comp["n_null"], se_df=se_df, reference=1.0,
            reference_se=0.05, reference_scale="excess"))
        assert (st, reason, impl) == (ref.status.value, ref.reason, "evidence")
        assert a.to_dict() == ref.to_dict()
        n_t += se_df is not None and comp["se"] > 0
    assert n_t > 500


def test_classify_needs_a_sampling_se_an_anchor_and_the_declared_family():
    proto = _ncal_protocol()
    comp = {"estimate": 3.0, "null_mean": 0.0, "null_sd": 1.0, "n_null": 19,
            "defined": True, "statistic": "raw", "null_family": "circular_shift",
            "se": 0.1, "se_n": 5}
    st, reason, impl, a = ncal.classify("PDI", comp, proto)
    assert (st, impl) == ("PRESENT", "evidence") and a.df > 4.0
    assert ncal.classify("PDI", {**comp, "estimate": -0.5}, proto)[0] == "ABSENT"
    # no sampling SE: UNDEFINED whatever the estimate (no null-MC stand-in)
    for se in (None, 0.0):
        st, reason, _, _ = ncal.classify("PDI", {**comp, "se": se}, proto)
        assert (st, reason) == ("UNDEFINED", "NO_SAMPLING_SE")
    # a family other than the declared one is not judged
    st, reason, _, a = ncal.classify("PDI", {**comp, "null_family": "fourier"}, proto)
    assert st == "UNDEFINED" and reason.startswith("NULL_FAMILY_MISMATCH") and a is None
    # an anchor at or below the null is INVALID_ANCHORS
    st, reason, _, _ = ncal.classify("PDI", comp, _ncal_protocol(value=0.0))
    assert (st, reason) == ("UNDEFINED", "INVALID_ANCHORS")
    # the legacy v1 rule is an explicit, labelled diagnostic
    st, _, impl, a = ncal.classify("PDI", comp, proto, status_rule="legacy_v1")
    assert (st, impl, a) == ("PRESENT", "legacy_v1", None)
    assert ncal.classify("PDI", {**comp, "defined": False, "reason": "x"}, proto,
                         status_rule="legacy_v1")[:2] == ("UNDEFINED", "x")
    with pytest.raises(ValueError, match="status_rule"):
        ncal.classify("PDI", comp, proto, status_rule="v1")
    # null systems have no cohort: the protocol needs an external reference
    from impact_pipeline import evidence as ev

    with pytest.raises(ValueError, match="external reference"):
        ncal.resolve_calibration_protocol(ev.Protocol())
    assert ncal.kleene_verdict(["PRESENT", "PRESENT"]) == "MPC_CONSISTENT"
    assert ncal.kleene_verdict(["PRESENT", "ABSENT", "UNDEFINED"]) == "EXCLUDED"
    assert ncal.kleene_verdict(["PRESENT", "UNDEFINED"]) == "UNDETERMINED"


def test_expected_rate_uses_the_null_size_of_calibrated_replicates():
    """Regression: undefined replicates record n_null = 0; the median over all
    replicates then understated K (or gave K = 0 -> NaN)."""
    rows = []
    for rep, (st, n_null) in enumerate([("UNDEFINED", 0), ("UNDEFINED", 0),
                                        ("PRESENT", 19), ("ABSENT", 19),
                                        ("UNDEFINED", 0)]):
        rows.append({"null_kind": "ar1", "n_time": 1200, "n_nodes": 8,
                     "replicate": rep, "principle": "PDI", "status": st,
                     "n_null": n_null, "margin": 0.0, "defined": n_null > 0,
                     "status_impl": "evidence", "runner": "x",
                     "seconds_system": 0.1})
    rates, _ = ncal.summarise(pd.DataFrame(rows))
    assert int(rates["median_n_null"].iloc[0]) == 19
    assert rates["expected_rate_exchangeable_v1"].iloc[0] == pytest.approx(
        ncal.expected_exchangeable_rate(19))


def test_small_end_to_end_run_is_deterministic(tmp_path):
    import logging
    import warnings

    kw = dict(kinds=("ar1", "surrogate_linear"), n_times=(1200,), n_nodes=(8,),
              replicates=2, null_surrogates=5, metrics=("PDI", "NAS", "SRPI"),
              iim_macro_nodes=3, seed=7, se_groups=2, protocol=_ncal_protocol())
    disable_before = logging.root.manager.disable
    filters_before = list(warnings.filters)
    a = ncal.run(tmp_path / "a", **kw)
    # a serial run shares the caller's process: logging and warnings restored
    assert logging.root.manager.disable == disable_before
    assert list(warnings.filters) == filters_before
    b = ncal.run(tmp_path / "b", workers=2, **kw)
    cols = ["null_kind", "principle", "estimate", "null_mean", "null_sd", "status"]
    pd.testing.assert_frame_equal(a["replicates"][cols], b["replicates"][cols])
    rates = a["rates"]
    # NAS is not computed on the linear surrogate family (it is not a null for NAS)
    assert set(rates[rates["null_kind"] == "surrogate_linear"]["principle"]) == {
        "PDI", "SRPI"}
    assert rates["n"].eq(2).all()
    srpi = a["replicates"][a["replicates"]["principle"] == "SRPI"]
    assert srpi["defined"].all()
    assert set(a["verdicts"]["necessity_set"]) == {"NAS,PDI,SRPI", "PDI,SRPI"}
    for name in ("null_calibration_replicates.csv", "null_calibration_rates.csv",
                 "null_calibration_verdicts.csv", "null_calibration.json"):
        assert (tmp_path / "a" / name).is_file()
    assert a["summary"]["evidence_api"] == "v2"
    assert a["summary"]["protocol_hash"] == _ncal_protocol().hash
    rep = a["replicates"]
    assert set(rep["status_impl"]) == {"evidence"}
    # every defined component has a jackknife SE over 2 groups (1 df)
    ok = rep["defined"] & (rep["se_n"] == 2)
    assert ok.any() and np.isfinite(rep.loc[ok, "c"]).all()
