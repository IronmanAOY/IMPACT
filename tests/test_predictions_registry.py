"""Tests of the executable hypothesis registry (predictions/registry.yaml, its
schema) and scripts/run_predictions.py (refusals, strata, decisions)."""
import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

import scripts.necessity_power as npw
import scripts.run_predictions as rp
from impact_pipeline import necessity as nc

PRINCIPLES = rp.PRINCIPLES
VERSION = "1.1.0+gtest"
EST = {"RAM": "compute_RAM:prediction_error", "PDI": "compute_PDI:repertoire",
       "NAS": "compute_NAS:capacity", "IIM": "compute_IIM:delta_psi",
       "SRPI": "compute_SRPI:agency"}


@pytest.fixture(scope="module")
def registry():
    return rp.load_registry()


@pytest.fixture(scope="module")
def schema():
    return rp.load_schema()


def frozen(reg):
    r = copy.deepcopy(reg)
    r["status"] = "frozen"
    r["freeze_tag"] = "paper2-freeze-test"
    p = r["protocols"][0]
    p["hash_algorithm"] = "canonical_json_sha256"
    p["hash"] = rp.canonical_json_sha256(p["spec"])
    for e in r["estimators"]:
        e["version"] = VERSION
        e["registration"] = "registered"
    for h in r["hypotheses"]:
        if h.get("minimum_n_status") == "provisional":
            h["minimum_n_status"] = "frozen"
    return r


def episodes(n_pos=200, n_neg=100, dataset="dream_mediated", seed=0, reg=None,
             absent_pos=None, absent_neg=None, undefined_pos=None, v1_names=False):
    """Synthetic episode table; per-principle ABSENT/UNDEFINED rates."""
    rng = np.random.default_rng(seed)
    absent_pos = absent_pos or {}
    absent_neg = absent_neg or {}
    undefined_pos = undefined_pos or {}
    n = n_pos + n_neg
    pos = np.r_[np.ones(n_pos, bool), np.zeros(n_neg, bool)]
    df = pd.DataFrame({"dataset": dataset, "episode_id": [f"e{i}" for i in range(n)],
                       "report_positive": pos})
    for p in PRINCIPLES:
        st = np.full(n, "PRESENT", dtype=object)
        u = rng.random(n)
        k_abs = int(round(absent_pos.get(p, 0.0) * n_pos))
        st[:k_abs] = "ABSENT"
        k_und = int(round(undefined_pos.get(p, 0.0) * n_pos))
        st[k_abs:k_abs + k_und] = "UNDEFINED"
        neg = ~pos & (u < absent_neg.get(p, 0.5))
        st[neg] = "ABSENT"
        df[f"{p}_status"] = st
        df[f"{p}_estimator"] = EST[p]
        df[f"{p}_estimator_version"] = VERSION
    stat = df[[f"{p}_status" for p in PRINCIPLES]].to_numpy()
    verdict = np.where((stat == "ABSENT").any(axis=1), "EXCLUDED",
                       np.where((stat == "PRESENT").all(axis=1), "MPC_CONSISTENT",
                                "UNDETERMINED"))
    if v1_names:
        verdict = np.vectorize({"EXCLUDED": "NOT_ATTRIBUTED",
                                "MPC_CONSISTENT": "ATTRIBUTED",
                                "UNDETERMINED": "UNDETERMINED"}.get)(verdict)
    df["MPC_verdict"] = verdict
    df["MPC_reason"] = ""
    reg = reg if reg is not None else frozen(rp.load_registry())
    df["protocol_hash"] = reg["protocols"][0]["hash"]
    return df


def outcomes(res, stratum="confirmatory"):
    return {r["hypothesis"]: r for r in res["rows"] if r["stratum"] == stratum}


# --------------------------------------------------------------------------
# registry and schema
# --------------------------------------------------------------------------
def test_shipped_registry_is_valid(registry, schema):
    assert rp.validate_registry(registry, schema) == []
    js = rp.jsonschema_errors(registry, schema)
    if js is not None:
        assert js == []
    assert registry["status"] == "draft"
    ids = [h["id"] for h in registry["hypotheses"]]
    assert ids == list(rp.REQUIRED_HYPOTHESES)
    h2 = next(h for h in registry["hypotheses"] if h["id"] == "H2")
    assert h2["counts_for_stance"] is False and h2["level"] == "auxiliary"
    roles = {d["id"]: (d["prior_access"], d["role"]) for d in registry["datasets"]}
    for ds in ("ds003171", "ds005620", "ds006623"):
        assert roles[ds] == ("exploratory_calibration", "exploratory")
    assert all(pa == "never_accessed" for pa, role in roles.values()
               if role == "confirmatory")


@pytest.mark.parametrize("mutate, fragment", [
    (lambda r: r["datasets"][0].update(role="confirmatory"),
     "confirmatory role requires never_accessed"),
    (lambda r: r["datasets"][3].update(role="confirmatory"),
     "confirmatory role requires never_accessed"),
    (lambda r: r["datasets"][0].update(prior_access="metadata_only"),
     "ds003171 must have prior_access exploratory_calibration"),
    (lambda r: r["hypotheses"][2].update(counts_for_stance=True),
     "H2 (aggregation exponent) must be auxiliary"),
    (lambda r: r["hypotheses"].pop(7), "missing hypotheses ['H7']"),
    (lambda r: r["hypotheses"][3].update(decision="non_inferiority"),
     "needs decision symmetric_necessity"),
    (lambda r: r.update(status="frozen"), "a frozen registry needs a freeze_tag"),
    (lambda r: r.update(unexpected=1), "unexpected property 'unexpected'"),
    (lambda r: r["protocols"][0].update(hash="XYZ"), "does not match"),
    (lambda r: r["defaults"].update(alpha=0.5), "> 0.2"),
    (lambda r: r["estimators"][0].update(principle="PHI"), "not in"),
])
def test_registry_mutations_are_rejected(registry, schema, mutate, fragment):
    reg = copy.deepcopy(registry)
    mutate(reg)
    errs = rp.validate_registry(reg, schema)
    assert any(fragment in e for e in errs), errs


def test_builtin_validator_agrees_with_jsonschema(registry, schema):
    jsonschema = pytest.importorskip("jsonschema")
    assert jsonschema is not None
    variants = [copy.deepcopy(registry) for _ in range(6)]
    variants[1]["registry_version"] = "v1"
    variants[2]["hypotheses"][0]["statistic"] = "p_value"
    variants[3]["datasets"][0]["prior_access"] = "sometimes"
    variants[4]["protocols"][0]["necessity_set"] = ["RAM", "RAM"]
    variants[5]["hypotheses"][4]["power_assumptions"]["extra"] = 1
    for v in variants:
        assert (rp.validate_schema(v, schema) == []) == (
            rp.jsonschema_errors(v, schema) == [])


def test_registered_minimum_n_is_reproduced_by_the_power_analysis(registry):
    checked = {"component": 0, "verdict": 0}
    for h in registry["hypotheses"]:
        pa = h.get("power_assumptions")
        if not pa:
            continue
        level = pa.get("level", "component")
        if level == "verdict":
            assert h["statistic"] == "excluded_rate_report_positive", h["id"]
            assert npw.verdict_minimum_n(pa) == h["minimum_n"], h["id"]
            checked[level] += 1
            continue
        params = {
            "tau": (pa["tau"],), "alpha": pa["alpha"], "pi_nec": (pa["pi_nec"],),
            "pi_viol": (pa["pi_viol"],), "pi_unconscious": (pa["pi_unconscious"],),
            "label_noise": (pa["label_noise"],), "label_noise_negative": 0.0,
            "base_rate": (pa["base_rate"],), "coverage": (pa["coverage"],),
            "sensitivity": (pa["sensitivity"],), "false_absent": (pa["false_absent"],),
            "target_power": (pa["target_power"],), "n_values": tuple(range(5, 601, 5)),
        }
        mn = npw.min_n_table(npw.power_table(params), (pa["target_power"],))
        assert int(mn["n_stable"].max()) == h["minimum_n"], h["id"]
        assert h["statistic"] == "absent_rate_report_positive", h["id"]
        checked[level] += 1
    # H3-H7: component criterion; H1: verdict-level EXCLUDED rate
    assert checked == {"component": 5, "verdict": 1}


def test_h1_minimum_n_comes_from_the_verdict_power_model(registry):
    """H1's minimum n is the value scripts/necessity_power.py --level verdict
    produces at the declared assumptions (286), not the component-level 155,
    at which P(SUPPORTED) was only about 0.62; it stays provisional until the
    freeze."""
    from scipy.stats import binom

    h1 = next(h for h in registry["hypotheses"] if h["id"] == "H1")
    pa = h1["power_assumptions"]
    assert pa["level"] == "verdict" and pa["tau"] == h1["epsilon"]
    assert h1["minimum_n"] == npw.verdict_minimum_n(pa) == 286
    assert h1["minimum_n_status"] == "provisional"
    # the EXCLUDED rate at the assumptions (necessity holds) is the label noise
    sc = {**pa, "n_principles": 5}
    assert npw.excluded_rate(sc, "necessity_holds")[1] == pytest.approx(0.02)
    _, k_s = nc.necessity_thresholds(155, h1["epsilon"], 0.05)
    assert binom.cdf(k_s, 155, 0.02) == pytest.approx(0.62, abs=0.01)
    # a provisional minimum n cannot be frozen
    bad = frozen(registry)
    next(h for h in bad["hypotheses"] if h["id"] == "H1")[
        "minimum_n_status"] = "provisional"
    assert any("provisional minimum_n" in e for e in rp.validate_registry(bad))


def test_verdict_power_model_known_answers(tmp_path):
    params = dict(npw.DEFAULTS)
    params.update(tau=(0.05,), pi_nec=(0.0,), pi_viol=(0.25,), label_noise=(0.02,),
                  base_rate=(0.5,), coverage=(1.0,), sensitivity=(0.9,),
                  false_absent=(0.0, 0.01), pi_unconscious=(0.6,),
                  n_principles=(5,), excluded_unconscious=(1.0, 0.0),
                  target_power=(0.8,), n_values=(50, 150, 300))
    pw = npw.verdict_power_table(params)
    sub = pw[(pw["false_absent"] == 0.0) & (pw["excluded_unconscious"] == 0.0)]
    # f = 0 and no mislabelled exclusions: nothing is ever EXCLUDED when
    # necessity holds; one violated principle -> (1 - lambda) * 0.225
    holds = sub[sub["truth"] == "necessity_holds"]
    assert (holds["q_excluded"] == 0.0).all()
    viol = sub[sub["truth"] == "necessity_violated"]["q_excluded"].iloc[0]
    assert viol == pytest.approx(0.98 * 0.25 * 0.9)
    f = pw[(pw["false_absent"] == 0.01) & (pw["excluded_unconscious"] == 1.0)
           & (pw["truth"] == "necessity_holds")]["q_excluded"].iloc[0]
    assert f == pytest.approx(0.98 * (1 - 0.99 ** 5) + 0.02)
    res = npw.run_verdict(tmp_path, params)
    assert (tmp_path / "verdict_min_n.csv").is_file()
    assert set(res["min_n"]["truth"]) == {"necessity_holds", "necessity_violated"}
    assert npw.main(["--out", str(tmp_path / "cli"), "--level", "verdict",
                     "--tau", "0.05", "--label-noise", "0.02", "--pi-viol", "0.25",
                     "--base-rate", "0.5", "--coverage", "0.8",
                     "--false-absent", "0.0", "--target-power", "0.8",
                     "--n-values", "250:300:1"]) == 0


def test_frozen_registry_rules(registry, schema):
    reg = frozen(registry)
    assert rp.validate_registry(reg, schema) == []
    bad = copy.deepcopy(reg)
    bad["protocols"][0]["hash"] = "0" * 64
    assert any("does not match its spec" in e for e in rp.validate_registry(bad))
    bad = copy.deepcopy(reg)
    bad["estimators"][4]["registration"] = "pending_validation"
    assert any("no registered estimator for SRPI" in e
               for e in rp.validate_registry(bad))


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------
def test_draft_registry_refuses_confirmatory_and_unregistered_results(registry):
    df = episodes(reg=frozen(registry))
    with pytest.raises(rp.RegistryRefusal, match="draft"):
        rp.evaluate(registry, df)
    with pytest.raises(rp.RegistryRefusal, match="unregistered protocol hashes"):
        rp.evaluate(registry, df, exploratory=True)
    res = rp.evaluate(registry, df, exploratory=True, allow_unregistered=True)
    assert set(r["stratum"] for r in res["rows"]) == {"exploratory"}
    assert all(r["unregistered"] for r in res["rows"])


def _bump_one_ram_version(d):
    d.loc[0, "RAM_estimator_version"] = "9.9.9"


@pytest.mark.parametrize("change, fragment", [
    (_bump_one_ram_version,
     "unregistered RAM estimator versions ['compute_RAM:prediction_error@9.9.9']"),
    (lambda d: d.__setitem__("protocol_hash", "f" * 64), "unregistered protocol"),
    (lambda d: d.__setitem__("dataset", "ds999999"), "undeclared datasets"),
    (lambda d: d.drop(columns=["IIM_estimator_version"], inplace=True),
     "results lack IIM_estimator/IIM_estimator_version"),
])
def test_unregistered_results_are_refused(registry, change, fragment):
    reg = frozen(registry)
    df = episodes(reg=reg)
    change(df)
    with pytest.raises(rp.RegistryRefusal) as exc:
        rp.evaluate(reg, df)
    assert any(fragment in r for r in exc.value.reasons), exc.value.reasons


# --------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------
def test_confirmatory_outcomes_on_planted_results(registry):
    reg = frozen(registry)
    df = episodes(n_pos=450, n_neg=225, reg=reg, absent_pos={"PDI": 0.3},
                  absent_neg={"SRPI": 0.0}, undefined_pos={"NAS": 0.66})
    res = rp.evaluate(reg, df)
    out = outcomes(res)
    assert out["H3"]["outcome"] == nc.SUPPORTED  # RAM never ABSENT, varies
    assert out["H4"]["outcome"] == nc.FALSIFIED  # PDI ABSENT in 30%
    assert out["H5"]["outcome"] == nc.INDETERMINATE  # NAS: 153 determinate < 155
    assert out["H5"]["reason"] == "below_minimum_n"
    assert out["H1"]["n"] == 288  # 135 EXCLUDED + 153 MPC_CONSISTENT >= 286
    assert out["H6"]["outcome"] == nc.SUPPORTED
    assert out["H7"]["reason"] == "component_does_not_vary"
    assert out["H1"]["outcome"] == nc.FALSIFIED  # 30% EXCLUDED
    assert out["H8"]["outcome"] == nc.SUPPORTED  # no SOURCE_INCOHERENT
    assert out["H0"]["outcome"] == rp.NOT_EVALUABLE
    assert out["H2"]["outcome"] == rp.NOT_EVALUABLE
    assert res["stance"]["confirmatory"]["stance"] == "FALSIFIED"
    assert set(res["stance"]["confirmatory"]["falsified_by"]) == {"H1", "H4"}
    # worst-case missingness sensitivity is reported next to the primary outcome
    assert out["H3"]["sensitivity_missing"] == "worst_case"
    assert out["H3"]["sensitivity_outcome"] == nc.SUPPORTED
    assert out["H5"]["sensitivity_outcome"] == nc.INDETERMINATE


def test_freeze_tag_verification_refuses_unknown_tags(registry, tmp_path):
    reg = frozen(registry)
    probs = rp.verify_freeze_tag(reg, rp.DEFAULT_REGISTRY)
    assert probs and "not found" in probs[0]
    assert rp.verify_freeze_tag(registry, rp.DEFAULT_REGISTRY) == [
        "registry is not frozen"]


def test_exploratory_datasets_never_enter_the_confirmatory_stratum(registry):
    reg = frozen(registry)
    conf = episodes(reg=reg)
    expl = episodes(reg=reg, dataset="ds003171", absent_pos={"RAM": 0.5}, seed=1)
    res = rp.evaluate(reg, pd.concat([conf, expl], ignore_index=True))
    assert outcomes(res, "confirmatory")["H3"]["outcome"] == nc.SUPPORTED
    assert outcomes(res, "exploratory")["H3"]["outcome"] == nc.FALSIFIED
    assert res["stance"]["confirmatory"]["stance"] in ("SUPPORTED", "INDETERMINATE")


def test_v1_verdict_names_give_the_same_outcomes(registry):
    reg = frozen(registry)
    kw = dict(reg=reg, absent_pos={"PDI": 0.02}, undefined_pos={"IIM": 0.1})
    a = outcomes(rp.evaluate(reg, episodes(**kw)))
    b = outcomes(rp.evaluate(reg, episodes(v1_names=True, **kw)))
    for h in a:
        assert a[h]["outcome"] == b[h]["outcome"], h


def test_coverage_and_reason_rate_hypotheses(registry):
    reg = frozen(registry)
    no_neg_absent = {p: 0.0 for p in PRINCIPLES}
    df = episodes(reg=reg, undefined_pos={"NAS": 0.9}, absent_neg=no_neg_absent)
    df.loc[: 149, "MPC_reason"] = "SOURCE_INCOHERENT"
    out = outcomes(rp.evaluate(reg, df))
    # 180 of 200 positives undetermined, negatives all consistent: coverage 120/300
    assert out["H9"]["rate"] == pytest.approx(120 / 300)
    assert out["H9"]["outcome"] == nc.FALSIFIED  # upper bound below 0.5
    assert out["H8"]["outcome"] == nc.FALSIFIED  # 75% SOURCE_INCOHERENT


def _null_rates(rate, n, cells=1, principles=PRINCIPLES, impl="evidence"):
    rows = []
    for p in principles:
        for c in range(cells):
            k = int(round(rate * n))
            lo, hi = nc.clopper_pearson(k, n, 0.05)
            rows.append({"null_kind": f"k{c}", "n_time": 1200, "n_nodes": 8,
                         "principle": p, "n": n, "n_present": k,
                         "false_present_rate": k / n, "false_present_lo": lo,
                         "false_present_hi": hi, "status_impl": impl})
    return pd.DataFrame(rows)


def test_null_calibration_hypothesis(registry):
    reg = frozen(registry)
    df = episodes(reg=reg)
    ok = outcomes(rp.evaluate(reg, df, null_rates=_null_rates(0.01, 500)))
    assert ok["H0"]["outcome"] == nc.SUPPORTED
    assert ok["H0"]["reason"] == "pooled_upper_bounds_below_alpha_plus_band"
    small = outcomes(rp.evaluate(reg, df, null_rates=_null_rates(0.01, 100)))
    assert small["H0"]["reason"] == "below_minimum_n"
    bad = outcomes(rp.evaluate(reg, df, null_rates=_null_rates(0.2, 500)))
    assert bad["H0"]["outcome"] == nc.FALSIFIED
    # at exactly alpha a single cell of 500 cannot show the rate below 0.07
    edge = outcomes(rp.evaluate(reg, df, null_rates=_null_rates(0.05, 500)))
    assert edge["H0"]["outcome"] == nc.INDETERMINATE
    # one anti-conservative cell among many falsifies (family-wise)
    many = _null_rates(0.01, 500, cells=8)
    many.loc[3, ["n_present", "false_present_rate"]] = (100, 0.2)
    out = outcomes(rp.evaluate(reg, df, null_rates=many))["H0"]
    assert out["outcome"] == nc.FALSIFIED and out["n_cells_outside"] == 1


def test_null_calibration_rule_does_not_degrade_with_the_number_of_cells():
    """The former all-cells-within-band rule was SUPPORTED with probability
    about 0.95^cells under exact calibration; the pooled rule is not."""
    h0 = {"id": "H0", "statistic": "null_false_present_rate", "level": "measurement",
          "counts_for_stance": False, "minimum_n": 460}
    defaults = {"alpha": 0.05, "null_band": 0.02, "tau": 0.05, "epsilon": 0.05,
                "missing": "determinate"}
    rng = np.random.default_rng(0)
    for rate, want in ((0.02, nc.SUPPORTED), (0.15, nc.FALSIFIED)):
        hits = 0
        for _ in range(40):
            rows = _null_rates(0.0, 460, cells=8)
            k = rng.binomial(460, rate, size=len(rows))
            rows["n_present"] = k
            rows["false_present_rate"] = k / 460
            r = rp.eval_hypothesis(h0, pd.DataFrame(), defaults, null_rates=rows)
            hits += r["outcome"] == want
        assert hits >= 38, (rate, hits)


def _h10(reg, **kw):
    return {**next(h for h in reg["hypotheses"] if h["id"] == "H10"), **kw}


def test_selective_exclusion_accuracy_gap(registry):
    reg = frozen(registry)
    rng = np.random.default_rng(3)
    df = episodes(reg=reg, absent_neg={p: 0.0 for p in PRINCIPLES})
    pos = df["report_positive"].to_numpy()
    h10 = _h10(reg, n_boot=300)
    # IMPaCT excludes half of the report-negative episodes and is MPC_CONSISTENT
    # on the others (not an error: "not excluded" is no attribution)
    half = np.arange(len(df)) % 2 == 0
    df["MPC_verdict"] = np.where(~pos & half, "EXCLUDED", "MPC_CONSISTENT")
    df["comparator_LZc"] = rng.random(len(df)) < 0.5  # excludes at random
    r = rp.eval_hypothesis(h10, df, reg["defaults"])
    assert r["outcome"] == nc.SUPPORTED and r["rate"] > 0.3
    assert r["excluded_rate_report_positive"]["impact"] == 0.0
    assert r["excluded_rate_report_negative"]["impact"] == pytest.approx(0.5)
    # the same exclusions with UNDETERMINED instead of MPC_CONSISTENT: same gap
    df2 = df.copy()
    df2["MPC_verdict"] = np.where(~pos & half, "EXCLUDED", "UNDETERMINED")
    assert rp.eval_hypothesis(h10, df2, reg["defaults"])["rate"] == pytest.approx(
        r["rate"])
    # exclusions of report-positive episodes are the errors
    df["MPC_verdict"] = np.where(pos, "EXCLUDED", "MPC_CONSISTENT")
    df["comparator_LZc"] = pos
    r = rp.eval_hypothesis(h10, df, reg["defaults"])
    assert r["outcome"] == nc.FALSIFIED
    # no exclusion claims at all: nothing to compare
    df["MPC_verdict"] = "MPC_CONSISTENT"
    r = rp.eval_hypothesis(h10, df, reg["defaults"])
    assert r["outcome"] == nc.INDETERMINATE and r["reason"] == "too_few_exclusions"
    df = df.drop(columns=["comparator_LZc"])
    assert rp.eval_hypothesis(h10, df, reg["defaults"])["outcome"] == rp.NOT_EVALUABLE


def test_comparator_columns_read_back_from_csv(registry, tmp_path):
    """Regression: a 0/1 comparator column with abstentions is read by pandas
    as float (1.0/0.0/NaN); 1.0 must count as 'not excluded', not as False."""
    reg = frozen(registry)
    df = episodes(reg=reg, absent_neg={p: 0.0 for p in PRINCIPLES})
    pos = df["report_positive"].to_numpy()
    df["MPC_verdict"] = np.where(pos, "MPC_CONSISTENT", "EXCLUDED")
    comp = pos.astype(float)
    comp[::7] = np.nan  # abstentions
    df["comparator_LZc"] = comp  # a perfect comparator on its decided cases
    path = tmp_path / "episodes.csv"
    df.to_csv(path, index=False)
    back = pd.read_csv(path)
    assert back["comparator_LZc"].dtype == float
    h10 = _h10(reg, n_boot=200)
    r = rp.eval_hypothesis(h10, back, reg["defaults"])
    assert r["rate"] == pytest.approx(0.0)  # both perfect: gap 0
    assert r["outcome"] == nc.SUPPORTED
    assert rp._decisions_from_column([1.0, 0.0, np.nan, True, "EXCLUDED", ""]) == [
        True, False, None, True, False, None]
    back["comparator_LZc"] = back["comparator_LZc"].fillna(2.0)
    bad = rp.eval_hypothesis(h10, back, reg["defaults"])
    assert bad["outcome"] == rp.NOT_EVALUABLE
    assert bad["reason"].startswith("invalid_comparator_column")


def test_best_comparator_ignores_comparators_without_decisions(registry):
    """Regression: max() over accuracies with a NaN (a comparator that
    abstains everywhere) depended on the column order."""
    reg = frozen(registry)
    df = episodes(reg=reg, absent_neg={p: 0.0 for p in PRINCIPLES})
    pos = df["report_positive"].to_numpy()
    df["MPC_verdict"] = np.where(pos, "MPC_CONSISTENT", "EXCLUDED")
    h10 = {**next(h for h in reg["hypotheses"] if h["id"] == "H10"), "n_boot": 200}
    rng = np.random.default_rng(0)
    good = np.where(rng.random(len(df)) < 0.9, pos, ~pos)
    res = []
    for order in (("LZc", "PhiR"), ("PhiR", "LZc")):
        d = df.copy()
        d[f"comparator_{order[0]}"] = [None] * len(d)  # abstains everywhere
        d[f"comparator_{order[1]}"] = good
        res.append(rp.eval_hypothesis(h10, d, reg["defaults"]))
    assert res[0]["rate"] == pytest.approx(res[1]["rate"])
    assert res[0]["outcome"] == res[1]["outcome"] == nc.SUPPORTED
    assert math.isfinite(res[0]["rate"])


def test_missing_report_labels_are_excluded_not_negative(registry):
    """Regression: a missing / unrecognised report label was counted as
    report-negative (it fed the variation check and H10's truth)."""
    reg = frozen(registry)
    df = episodes(reg=reg, absent_neg={p: 0.0 for p in PRINCIPLES})
    base = outcomes(rp.evaluate(reg, df))
    assert base["H3"]["reason"] == "component_does_not_vary"
    # unknown-report episodes with RAM ABSENT must not make RAM 'vary'
    extra = episodes(n_pos=0, n_neg=30, reg=reg, absent_neg={"RAM": 1.0}, seed=5)
    extra["report_positive"] = [np.nan] * 15 + ["unknown"] * 15
    both = pd.concat([df, extra], ignore_index=True)
    out = outcomes(rp.evaluate(reg, both))
    assert out["H3"]["reason"] == "component_does_not_vary"
    assert out["H3"]["n_report_unknown"] == 30
    assert out["H1"]["n"] == base["H1"]["n"]
    labels = rp.report_labels(pd.DataFrame({"report": ["positive", "Negative", "",
                                                       "maybe", 1, 0.0]}))
    assert labels[:2].tolist() == [1.0, 0.0] and np.isnan(labels[2:4]).all()
    assert labels[4:].tolist() == [1.0, 0.0]
    with pytest.raises(ValueError, match="missing or unrecognised"):
        rp.report_positive(pd.DataFrame({"report_positive": [True, np.nan]}))


def test_v1_coherence_reason_counts_as_source_incoherent(registry):
    reg = frozen(registry)
    df = episodes(reg=reg)
    df.loc[:149, "MPC_reason"] = "INCONCLUSIVE:PDI;BEARER_MISMATCH:COHERENCE"
    out = outcomes(rp.evaluate(reg, df))
    assert out["H8"]["outcome"] == nc.FALSIFIED
    assert not rp._has_reason(np.nan, "SOURCE_INCOHERENT")
    assert rp._has_reason("SOURCE_INCOHERENT:RAM|PDI", "SOURCE_INCOHERENT")


def test_null_calibration_from_the_legacy_rule_is_not_evaluable(registry):
    reg = frozen(registry)
    df = episodes(reg=reg)
    rates = _null_rates(0.01, 500)
    assert outcomes(rp.evaluate(reg, df, null_rates=rates))["H0"]["outcome"] == (
        nc.SUPPORTED)
    rates.loc[2, "status_impl"] = "legacy_v1"
    h0 = outcomes(rp.evaluate(reg, df, null_rates=rates))["H0"]
    assert h0["outcome"] == rp.NOT_EVALUABLE
    assert "legacy_v1" in h0["reason"]


def test_aggregation_exponent_is_auxiliary(registry):
    reg = frozen(registry)
    df = episodes(reg=reg, absent_neg={p: 0.0 for p in PRINCIPLES})
    rng = np.random.default_rng(0)
    C = rng.uniform(0.2, 1.0, size=(len(df), 5))
    for j, p in enumerate(PRINCIPLES):
        df[f"{p}_c"] = C[:, j]
    df["outcome_graded"] = C.mean(axis=1) + 0.01 * rng.standard_normal(len(df))
    res = rp.evaluate(reg, df)
    h2 = outcomes(res)["H2"]
    assert h2["outcome"] == rp.AUXILIARY_REPORTED
    assert h2["counts_for_stance"] is False
    assert h2["rate"] == pytest.approx(1.0, abs=0.5)  # arithmetic mean planted
    assert "H2" not in res["stance"]["confirmatory"].get("supported", [])
    # the registered minimum n (200 MPC_CONSISTENT episodes) applies to H2 too
    small = outcomes(rp.evaluate(reg, df.iloc[:150]))["H2"]
    assert small["outcome"] == nc.INDETERMINATE
    assert small["reason"] == "below_minimum_n" and small["n"] == 150
    assert small["outcome_ignoring_minimum_n"] == rp.AUXILIARY_REPORTED


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_validate_refuse_and_run(tmp_path, registry, capsys):
    import yaml

    assert rp.main(["--validate-only"]) == 0
    reg = frozen(registry)
    df = episodes(reg=reg)
    res_csv = tmp_path / "episodes.csv"
    df.to_csv(res_csv, index=False)
    out = tmp_path / "draft_out"
    assert rp.main(["--results", str(res_csv), "--out", str(out)]) == rp.REFUSAL_EXIT
    refusal = json.loads((out / "predictions_refusal.json").read_text())
    assert any("draft" in r for r in refusal["reasons"])
    reg_path = tmp_path / "registry_frozen.yaml"
    reg_path.write_text(yaml.safe_dump(reg), encoding="utf-8")
    out_tag = tmp_path / "tag_out"
    assert rp.main(["--registry", str(reg_path), "--results", str(res_csv),
                    "--out", str(out_tag)]) == rp.REFUSAL_EXIT
    refusal = json.loads((out_tag / "predictions_refusal.json").read_text())
    assert any("paper2-freeze-test" in r for r in refusal["reasons"])
    out2 = tmp_path / "frozen_out"
    assert rp.main(["--registry", str(reg_path), "--results", str(res_csv),
                    "--out", str(out2), "--skip-freeze-tag-check"]) == 0
    report = json.loads((out2 / "predictions_report.json").read_text())
    assert report["freeze_tag_checked"] is False
    assert report["registry_sha256"] == rp.file_sha256(reg_path)
    assert report["results_sha256"] == rp.file_sha256(res_csv)
    assert "confirmatory" in report["stance"]
    table = pd.read_csv(out2 / "predictions_results.csv")
    assert set(table["hypothesis"]) == set(rp.REQUIRED_HYPOTHESES)
    spec = tmp_path / "protocol.json"
    spec.write_text(json.dumps(reg["protocols"][0]["spec"]))
    capsys.readouterr()
    assert rp.main(["--hash-protocol", str(spec)]) == 0
    assert capsys.readouterr().out.strip() == reg["protocols"][0]["hash"]
