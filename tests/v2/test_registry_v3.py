# -*- coding: utf-8 -*-
"""
The applicability registry v3 and the forward-model admission procedure:

* Clopper-Pearson demonstration bounds (42 / 51 / 61 seeds for 0.07 at
  m = 1 / 2 / 4, 149 runs for 0.02) and the curtailment logic, which only
  ever stops towards non-admission and otherwise equals the full-sample
  decision;
* the FM criteria on synthetic records (FM0, FMa, FMb1, FMb2, FMd, FMabs,
  ``pi0`` and the vacuous flag, not_observable, estimator errors);
* registry lookup per status direction, through the evidence layer v2, to
  ``ESTIMATOR_NOT_VALIDATED:not_admitted`` / ``:not_observable``;
* regime-key matching and the regime constraints of an entry;
* entry validation, the v1 registry read as /3, round trips;
* the builder script on synthetic task records of the forward design.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from impact_pipeline import evidence as v1
from impact_pipeline import evidence_v2 as E
from impact_pipeline.bench import forward_v2 as F2
from impact_pipeline.bench.designs_v2 import forward as D
from impact_pipeline.v2 import records as REC
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import registry_v3 as RV
from scripts.v2 import build_registry_v3 as B

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_V1 = REPO_ROOT / "protocols" / "applicability_registry_v1.json"
P_, A_, U_ = "PRESENT", "ABSENT", "UNDEFINED"


# --------------------------------------------------------------------------
# demonstration bounds
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bound, level, n", [
    (0.07, 0.05, 42), (0.07, 0.025, 51), (0.07, 0.0125, 61), (0.02, 0.05, 149),
    (0.05, 0.05, 59), (0.01, 0.05, 299), (0.07, 0.05 / 6, 66),
])
def test_minimum_runs_for_a_demonstration(bound, level, n):
    """The seed numbers of design 3.6 and 4.9 (synth_oc A)."""
    assert RV.min_runs_for_demonstration(bound, level) == n
    assert RV.cp_upper(0, n, level) < bound <= RV.cp_upper(0, n - 1, level)
    assert RV.demonstration(0, n, bound, level)["demonstrated"]
    assert not RV.demonstration(0, n - 1, bound, level)["demonstrated"]


def test_clopper_pearson_bounds():
    # 0 events: 1 - level^(1/n)
    assert RV.cp_upper(0, 42, 0.05) == pytest.approx(1 - 0.05 ** (1 / 42), rel=1e-10)
    assert RV.cp_lower(0, 42) == 0.0 and RV.cp_upper(5, 5) == 1.0
    assert RV.cp_upper(0, 0) == 1.0
    # all events: the lower bound is level^(1/n)
    assert RV.cp_lower(10, 10, 0.05) == pytest.approx(0.05 ** (1 / 10), rel=1e-10)
    from scipy.stats import binom
    up = RV.cp_upper(3, 40, 0.05)
    assert binom.cdf(3, 40, up) == pytest.approx(0.05, rel=1e-8)
    lo = RV.cp_lower(3, 40, 0.05)
    assert binom.sf(2, 40, lo) == pytest.approx(0.05, rel=1e-8)
    with pytest.raises(RV.RegistryV3Error):
        RV.cp_upper(5, 4)
    with pytest.raises(RV.RegistryV3Error):
        RV.cp_upper(1, 4, 1.5)
    assert RV.max_demonstrable_events(150, 0.02) == 0
    assert RV.max_demonstrable_events(148, 0.02) == -1
    k = RV.max_demonstrable_events(600, 0.02)
    assert RV.cp_upper(k, 600) < 0.02 <= RV.cp_upper(k + 1, 600)


def test_curtailment_stops_at_the_first_false_absent():
    cur = RV.Curtailment(150, 0.02)
    assert cur.k_max == 0 and not cur.stopped
    for _ in range(9):
        assert not cur.update(False)
    assert cur.update(True)  # the tenth run makes 150-run admission impossible
    res = cur.result()
    assert (res["stopped"], res["stop_index"], res["n"], res["events"]) == (
        True, 9, 10, 1)
    assert not res["demonstrated"]
    assert cur.update(False) and cur.result()["n"] == 10  # nothing after a stop
    assert RV.Curtailment(100, 0.02).stopped  # 100 runs can never demonstrate 0.02
    full = RV.curtailed_demonstration([False] * 150, 150, 0.02)
    assert full["demonstrated"] and full["n"] == 150 and not full["stopped"]


@pytest.mark.parametrize("n_planned, bound, p", [
    (150, 0.02, 0.003), (150, 0.02, 0.0), (300, 0.02, 0.004), (61, 0.07, 0.01),
    (400, 0.05, 0.02)])
def test_curtailment_equals_the_full_sample_decision(n_planned, bound, p):
    """When all planned runs exist, curtailing never changes the decision;
    a stop is always a non-admission; runs after the stop are not needed."""
    rng = np.random.default_rng(int(n_planned * 1000 * (p + 1)))
    for _ in range(300):
        events = rng.random(n_planned) < p
        full = RV.demonstration(int(events.sum()), n_planned, bound)["demonstrated"]
        res = RV.curtailed_demonstration(events, n_planned, bound)
        assert res["demonstrated"] == full
        if res["stopped"]:
            assert not full
            assert events[: res["n"]].sum() == RV.max_demonstrable_events(
                n_planned, bound) + 1
            assert events[res["n"] - 1]  # the stopping run is an event
        else:
            assert res["n"] == n_planned


def test_three_zone_rule():
    assert RV.three_zone(16, 20)["zone"] == "SUPPORTED"  # 0.8 point rate
    z = RV.three_zone(15, 20)
    assert z["zone"] == "INDETERMINATE" and not z["pass"]
    assert z["upper"] == pytest.approx(RV.cp_upper(15, 20, 0.05))
    assert RV.three_zone(10, 20)["zone"] == "FALSIFIED"  # CP upper 0.70 < 0.8
    assert RV.three_zone(0, 0)["zone"] == "NOT_EVALUABLE"
    # with m the upper bound is taken at level / m
    assert RV.three_zone(12, 20, m=4)["upper"] == pytest.approx(
        RV.cp_upper(12, 20, 0.0125))


def test_spearman_dose_lesion_and_concordance():
    dose = np.repeat([0.0, 1.0, 2.0, 3.0], 5)
    c = dose + np.random.default_rng(1).normal(0, 0.3, dose.size)
    res = RV.spearman_dose(c, dose)
    assert res["pass"] and res["rho"] > 0.8 and res["p"] < 1e-4
    assert not RV.spearman_dose(-c, dose)["pass"]  # one-sided: rho must be > 0
    assert not RV.spearman_dose(np.ones(20), dose)["pass"]  # constant c
    c_nan = c.copy()
    c_nan[:3] = np.nan
    assert RV.spearman_dose(c_nan, dose)["n"] == 17
    lz = RV.lesion_contrast([0.1] * 17 + [0.9] * 3, [0.5] * 20)
    assert lz["zone"] == "SUPPORTED" and lz["successes"] == 17
    assert not RV.lesion_contrast([0.6] * 20, [0.5] * 20)["pass"]
    # FMd: same sign and at least half the size
    fd = RV.source_concordance([1.0, 1.0, 1.0, 0.4, -1.0], [0.0] * 5,
                               [1.0, 2.0, 2.5, 1.0, 1.0], [0.0] * 5)
    assert fd["successes"] == 2 and fd["n"] == 5  # 1>=0.5, 1>=1; 1<1.25; 0.4<0.5; sign
    with pytest.raises(RV.RegistryV3Error):
        RV.source_concordance([1.0], [0.0], [1.0, 2.0], [0.0, 0.0])


# --------------------------------------------------------------------------
# the FM criteria on synthetic records
# --------------------------------------------------------------------------
NAS_DESIGN = RV.AdmissionDesign(
    principle="NAS", arm="hopf", views=("source", "eeg64"),
    null_conditions=("G0",), on_conditions=("Gnom",),
    dose_conditions={"G0": 0.0, "G1": 0.5, "Gnom": 1.0, "G4": 4.0},
    concordance=("Gnom", "G0"), lesion=("hub", "random"), n_planned_on=150)


def _runs(view, cond, seeds, status=P_, c=0.0, dose=None, reason=None,
          principle="NAS"):
    out = []
    for i, s in enumerate(seeds):
        st = status(i) if callable(status) else status
        cv = c(i) if callable(c) else c
        rs = reason if st == U_ else None
        if st == U_ and rs is None:
            rs = R.INCONCLUSIVE
        out.append(RV.AdmissionRun(principle, view, cond, s, st, rs, cv, dose))
    return out


def nas_world(view="eeg64", null_present=0, on_absent_at=None, null_absent=10,
              lesion_hits=18, fwd_scale=1.0, error_at=None):
    """A Hopf-like arm that meets every NAS criterion unless told otherwise."""
    seeds61, seeds150, seeds20 = range(61), range(150), range(20)
    runs = []
    pairs = ((view, fwd_scale), ("source", 1.0)) if view != "source" else (
        ("source", 1.0),)
    for v, scale in pairs:
        runs += _runs(v, "G0", seeds61,
                      lambda i: P_ if i < null_present else (A_ if i < null_present
                                                             + null_absent else U_),
                      c=lambda i: 0.01 * (i % 3))
        runs += _runs(v, "Gnom", seeds150,
                      lambda i: A_ if on_absent_at is not None and i == on_absent_at
                      else P_, c=lambda i: scale * (1.0 + 0.01 * (i % 5)))
        runs += _runs(v, "G1", seeds20, U_, c=lambda i: scale * (0.4 + 0.01 * i))
        runs += _runs(v, "G4", seeds20, P_, c=lambda i: scale * (2.5 + 0.01 * i))
        runs += _runs(v, "hub", seeds20, P_,
                      c=lambda i: 0.2 if i < lesion_hits else 0.9)
        runs += _runs(v, "random", seeds20, P_, c=0.6)
    if error_at is not None:
        runs = [RV.AdmissionRun(r.principle, r.view, r.condition, r.seed, U_,
                                R.estimator_error(KeyError), None)
                if (r.view == view and r.condition == "G0" and r.seed == error_at)
                else r for r in runs]
    return [r for r in runs if r.view == view], [r for r in runs if r.view == "source"]


def evaluate(view_runs, src_runs, anchor=True, stage="sensor", design=NAS_DESIGN,
             view="eeg64"):
    return RV.evaluate_view(design, view, view_runs, stage=stage,
                            anchor={"valid": anchor}, source_runs=src_runs)


def test_a_view_that_meets_every_criterion_is_admitted():
    res = evaluate(*nas_world())
    assert (res.admitted_for_present, res.admitted_for_absent) == ("yes", "yes")
    assert res.failing == ()
    c = res.criteria
    assert c["FMa"]["classes"]["G0"]["demonstrated"] and c["FMa"]["m"] == 1
    assert c["FMb1"]["pass"] and c["FMb1"]["seeds"] == 20
    assert c["FMb2"]["zone"] == "SUPPORTED" and c["FMb2"]["successes"] == 18
    assert c["FMd"]["zone"] == "SUPPORTED" and c["FMd"]["n"] == 61
    assert c["FMabs"]["demonstrated"] and c["FMabs"]["n"] == 150
    assert res.pi0 == pytest.approx(10 / 61)


def test_one_false_present_at_the_null_fails_fma_only():
    res = evaluate(*nas_world(null_present=1))  # 1/61: CP upper 0.076 > 0.07
    assert res.criteria["FMa"]["classes"]["G0"]["upper"] > 0.07
    assert (res.admitted_for_present, res.admitted_for_absent) == ("no", "yes")
    assert res.failing == ("FMa",)


def test_fma_level_follows_m():
    d4 = RV.AdmissionDesign(**{**NAS_DESIGN.__dict__, "fma_m": 4})
    res = evaluate(*nas_world(), design=d4)
    assert res.criteria["FMa"]["level"] == pytest.approx(0.0125)
    assert res.criteria["FMa"]["classes"]["G0"]["demonstrated"]  # 61 seeds at m = 4
    short = [r for r in nas_world()[0] if not (r.condition == "G0" and r.seed >= 60)]
    res = evaluate(short, nas_world()[1], design=d4)
    assert not res.criteria["FMa"]["pass"]  # 60 seeds are one too few


def test_a_false_absent_among_on_runs_stops_the_view():
    res = evaluate(*nas_world(on_absent_at=30))
    fm = res.criteria["FMabs"]
    assert fm["stopped"] and fm["stop_index"] == 30 and fm["n"] == 31
    assert fm["n_available"] == 150
    assert (res.admitted_for_present, res.admitted_for_absent) == ("yes", "no")
    assert res.failing == ("FMabs",)


def test_vacuous_absent_admission():
    res = evaluate(*nas_world(null_absent=2))  # pi0 = 2/61 < 0.1
    assert res.pi0 == pytest.approx(2 / 61)
    assert res.admitted_for_absent == "vacuous"
    assert res.criteria["FMabs"]["pi0"] == res.pi0


def test_nas_needs_the_lesion_contrast_and_fmd():
    res = evaluate(*nas_world(lesion_hits=14))
    assert res.failing == ("FMb2",) and res.admitted_for_present == "no"
    assert res.admitted_for_absent == "yes"
    res = evaluate(*nas_world(fwd_scale=0.3))  # forward contrast < half the source's
    assert "FMd" in res.failing and res.admitted_for_present == "no"
    # the source view is its own source contrast
    view_runs, src = nas_world(view="source")
    res = evaluate(view_runs, src, stage="source", view="source")
    assert res.criteria["FMd"]["trivial"] and res.admitted_for_present == "yes"


def test_anchor_failure_blocks_both_directions():
    res = evaluate(*nas_world(), anchor=False)
    assert (res.admitted_for_present, res.admitted_for_absent) == ("no", "no")
    assert res.failing == ("FM0",)
    res = RV.evaluate_view(NAS_DESIGN, "eeg64", nas_world()[0], stage="sensor",
                           anchor=None, source_runs=nas_world()[1])
    assert res.failing == ("FM0",) and res.criteria["FM0"]["valid"] is None
    with pytest.raises(RV.RegistryV3Error):
        RV.evaluate_view(NAS_DESIGN, "eeg64", nas_world()[0], stage="sensor",
                         anchor={"lower": 0.1})


def test_estimator_errors_leave_the_denominators():
    res = evaluate(*nas_world(error_at=60))
    assert res.n_errors == 1
    assert res.criteria["FMa"]["classes"]["G0"]["n"] == 60
    assert res.n_runs == len(nas_world()[0])
    # an error is not an observation: 60 runs, 0 events still demonstrate at
    # m = 1 (42 needed), but not at m = 4 (61 needed)
    assert res.criteria["FMa"]["pass"]
    d4 = RV.AdmissionDesign(**{**NAS_DESIGN.__dict__, "fma_m": 4})
    assert not evaluate(*nas_world(error_at=60), design=d4).criteria["FMa"]["pass"]


def test_not_observable_view():
    runs = []
    for cond, seeds in (("G0", range(61)), ("Gnom", range(150)), ("G1", range(20)),
                        ("G4", range(20)), ("hub", range(20)), ("random", range(20))):
        runs += _runs("bold", cond, seeds, U_, c=None, reason=R.SAMPLING_UNRESOLVED)
    res = RV.evaluate_view(NAS_DESIGN, "bold", runs, stage="bold",
                           anchor={"valid": False})
    assert (res.admitted_for_present, res.admitted_for_absent) == (
        "not_observable", "not_observable")
    assert RV.is_not_observable(runs)
    mixed = runs[:-1] + _runs("bold", "random", [19], U_,
                              reason=R.INSUFFICIENT_OCCUPANCY)
    assert not RV.is_not_observable(mixed)
    assert not RV.is_not_observable([])


def test_duplicate_runs_are_refused():
    runs = _runs("eeg64", "G0", [1, 1])
    with pytest.raises(RV.RegistryV3Error, match="two runs"):
        RV.evaluate_view(NAS_DESIGN, "eeg64", runs, stage="sensor", anchor=True)


def test_design_validation():
    kw = {**NAS_DESIGN.__dict__}
    with pytest.raises(RV.RegistryV3Error):
        RV.AdmissionDesign(**{**kw, "lesion": None})  # NAS needs FMb2
    with pytest.raises(RV.RegistryV3Error):
        RV.AdmissionDesign(**{**kw, "principle": "PDI"})  # FMb2 is NAS only
    with pytest.raises(RV.RegistryV3Error):
        RV.AdmissionDesign(**{**kw, "dose_conditions": {"G0": 0.0}})


# --------------------------------------------------------------------------
# registry lookup per status direction -> ESTIMATOR_NOT_VALIDATED
# --------------------------------------------------------------------------
EEG_REGIME = {"observation_stage": "sensor", "n_sensors": 64, "reference": "average",
              "sensor_filter": "bandpass_1_40", "fs": 250.0, "duration_s": 60.0,
              "inputs_declared": "none"}


def entry(principle="RAM", present="no", absent="vacuous", *, version="ram-v3-2026.10",
          substrate="eeg_like_forward", regime=None, pi0=0.05, **kw):
    crit = {k: {"pass": True} for k in ("FM0", "FMa", "FMb1", "FMd", "FMabs")}
    if principle == "NAS":
        crit["FMb2"] = {"pass": True}
    if present == "no":
        crit["FMa"] = {"pass": False}
    if absent == "no":
        crit["FMabs"] = {"pass": False}
    crit["FMabs"]["pi0"] = pi0 if absent != "yes" else 0.5
    raw = {"principle": principle, "estimator": f"compute_{principle}:*",
           "version": version, "substrate": substrate,
           "regime": dict(EEG_REGIME if regime is None else regime),
           "admitted_for_present": present, "admitted_for_absent": absent,
           "criteria": crit,
           "failing": [k for k in RV.CRITERIA if k in crit and not crit[k]["pass"]],
           "evidence": {"run_id": "test-run"}}
    raw.update(kw)
    return raw


REF = {"kind": "external", "scale": "excess", "values": {"RAM": 1.0}}
PROTO = E.ProtocolV3(reference=REF)


def item(c, se=0.02, **kw):
    base = dict(principle="RAM", estimate=c, null_mean=0.0, null_sd=0.0, n_null=0,
                se=se, se_df=9.0, reference=1.0, reference_scale="excess",
                estimator="compute_RAM:pe_readout@ram-v3-2026.10",
                substrate="eeg", regime=dict(EEG_REGIME))
    base.update(kw)
    return E.ComponentEvidenceV2(**base)


@pytest.mark.parametrize("present, absent, c, se, status, reason", [
    ("no", "vacuous", 0.6, 0.1, U_, "ESTIMATOR_NOT_VALIDATED:not_admitted"),
    ("no", "vacuous", 0.0, 0.02, A_, None),
    ("no", "yes", 0.0, 0.02, A_, None),
    ("yes", "no", 0.0, 0.02, U_, "ESTIMATOR_NOT_VALIDATED:not_admitted"),
    ("yes", "no", 0.6, 0.1, P_, None),
    ("no", "no", 0.0, 0.02, U_, "ESTIMATOR_NOT_VALIDATED:not_admitted"),
    ("no", "no", 0.6, 0.1, U_, "ESTIMATOR_NOT_VALIDATED:not_admitted"),
    ("not_observable", "not_observable", 0.0, 0.02, U_,
     "ESTIMATOR_NOT_VALIDATED:not_observable"),
    ("yes", "no", 0.3, 0.05, U_, "ABSENT_NOT_REACHABLE"),  # undecided: its own reason
])
def test_registry_licenses_each_status_direction(present, absent, c, se, status,
                                                 reason):
    reg = RV.RegistryV3([entry(present=present, absent=absent)])
    a = E.assess_item(item(c, se), PROTO, registry=reg)
    assert (a.status.value, a.reason) == (status, reason)


def test_no_matching_entry_is_not_admitted():
    reg = RV.RegistryV3([entry(present="yes", absent="yes")])
    for kw in ({"estimator": "compute_RAM:pe_readout@ram-v2-2026.09"},  # other version
               {"substrate": "fmri"},                                  # other substrate
               {"regime": {**EEG_REGIME, "n_sensors": 32}},            # other regime
               {"regime": {k: v for k, v in EEG_REGIME.items() if k != "fs"}}):
        a = E.assess_item(item(0.0, **kw), PROTO, registry=reg)
        assert a.reason == "ESTIMATOR_NOT_VALIDATED:not_admitted", kw
    a = E.assess_item(item(0.0), PROTO, registry=reg)
    assert a.status.value == A_
    out = reg.admission("RAM", "compute_RAM:x@ram-v3-2026.10", "eeg", **EEG_REGIME)
    assert (out["present"], out["absent"]) == ("yes", "yes") and out["why"] is None
    out = reg.admission("RAM", "compute_RAM:x@ram-v3-2026.10", "eeg",
                        **{**EEG_REGIME, "n_sensors": 32})
    assert out == {"present": "no", "absent": "no", "entries": [],
                   "why": "REGIME_MISMATCH:n_sensors"}
    assert reg.admission("RAM", "compute_RAM:x@ram-v3-2026.10", "bold",
                         **EEG_REGIME)["why"] == "NO_ENTRY"
    assert reg.admission("RAM", None)["why"] == "ESTIMATOR_UNDECLARED"


def test_the_most_restrictive_matching_entry_wins():
    loose = entry(present="yes", absent="yes")
    strict = entry(present="no", absent="vacuous",
                   regime={"observation_stage": "sensor", "n_sensors_min": 32})
    reg = RV.RegistryV3([loose, strict])
    out = reg.admission("RAM", "compute_RAM:x@ram-v3-2026.10", "eeg_like_forward",
                        **EEG_REGIME)
    assert (out["present"], out["absent"]) == ("no", "vacuous")
    assert len(out["entries"]) == 2
    reg = RV.RegistryV3([loose, entry(present="not_observable",
                                      absent="not_observable")])
    out = reg.admission("RAM", "compute_RAM:x@ram-v3-2026.10", "eeg", **EEG_REGIME)
    assert (out["present"], out["absent"]) == ("not_observable", "not_observable")
    assert reg.is_validated("RAM", "compute_RAM:x@ram-v3-2026.10", "eeg",
                            **EEG_REGIME)[0] is False
    ok = RV.RegistryV3([loose]).is_validated(
        "RAM", "compute_RAM:x@ram-v3-2026.10", "eeg", **EEG_REGIME)
    assert ok == (True, None)


def test_verdict_v3_uses_the_registry():
    reg = RV.RegistryV3([entry(present="no", absent="vacuous")])
    proto = E.ProtocolV3(reference=REF, necessity_set=("RAM",))
    v = E.mpc_verdict({"RAM": [item(0.0)]}, proto, registry=reg)
    assert getattr(v.verdict, "value", v.verdict) == "EXCLUDED"
    v = E.mpc_verdict({"RAM": [item(0.9, se=0.05)]}, proto, registry=reg)
    assert getattr(v.verdict, "value", v.verdict) == "UNDETERMINED"
    assert v.component_status["RAM"].value == U_


def test_vocabularies_agree():
    assert RV.ADMISSION_VALUES == E.ADMISSION_VALUES
    assert RV.REGIME_KEYS == F2.REGIME_KEYS
    assert set(RV.PROVENANCE_KEYS) | set(RV.CONSTRAINT_KEYS) == set(RV.REGIME_KEYS)
    for code in RV.NOT_OBSERVABLE_REASONS:
        assert R.status_for_reason(code) == U_
    assert R.not_validated(R.NOT_OBSERVABLE) == "ESTIMATOR_NOT_VALIDATED:not_observable"


# --------------------------------------------------------------------------
# regime keys
# --------------------------------------------------------------------------
@pytest.mark.parametrize("constraints, query, failure", [
    ({"n_sensors": 64}, {"n_sensors": 64}, None),
    ({"n_sensors": 64}, {"n_sensors": 64.0000000001}, None),  # numeric tolerance
    ({"n_sensors": 64}, {"n_sensors": 32}, "n_sensors"),
    ({"n_sensors": 64}, {}, "n_sensors"),                     # omitted key
    ({"n_sensors": 64}, {"n_sensors": None}, "n_sensors"),
    ({"reference": "average"}, {"reference": "AVERAGE"}, None),
    ({"reference": ["average", "none"]}, {"reference": "none"}, None),
    ({"duration_s": {"min": 300, "max": 400}}, {"duration_s": 350.5}, None),
    ({"duration_s": {"min": 300, "max": 400}}, {"duration_s": 401}, "duration_s"),
    ({"n_sensors_min": 32}, {"n_sensors": 48}, None),
    ({"n_sensors_min": 32}, {"n_sensors": 24}, "n_sensors_min"),
    ({"tr_max": 2.0}, {"tr": 2.5}, "tr_max"),
    ({"duration_min_s": 60}, {"duration_s": 60}, None),
    ({"fs_min": 200, "observation_stage": "sensor"},
     {"fs": 250, "observation_stage": "bold"}, "observation_stage"),
    ({"T_min": 2000}, {"n_time": 1999}, "T_min"),             # v1 named bound
    ({"fs_or_tr": {"min": 0.002, "max": 0.01}}, {"tr": 0.004}, None),
    ({"inputs_declared": "none"}, {"inputs_declared": "complete"}, "inputs_declared"),
])
def test_regime_key_matching(constraints, query, failure):
    assert RV.regime_failure(constraints, query) == failure


def test_regime_constraints_from_the_runs():
    a = dict(EEG_REGIME, leadfield_family="spherical_gauss", leadfield_seed=20261001,
             conduction_width=0.6, tr=None, snr=None)
    b = dict(a, duration_s=75.5)
    c = dict(a, duration_s=61.25)
    cons, keys = RV.regime_constraints([a, b, c])
    assert keys["duration_s"] == {"min": 60.0, "max": 75.5}
    assert cons["duration_s"] == {"min": 60.0, "max": 75.5}
    for k in RV.PROVENANCE_KEYS:
        assert k not in cons and keys[k] == a[k]
    assert "tr" not in cons and keys["tr"] is None
    assert cons["n_sensors"] == 64 and cons["observation_stage"] == "sensor"
    assert RV.regime_failure(cons, dict(EEG_REGIME, duration_s=70)) is None
    cons2, keys2 = RV.regime_constraints([a, dict(a, reference="none")])
    assert cons2["reference"] == ["average", "none"]
    with pytest.raises(RV.RegistryV3Error, match="unknown"):
        RV.regime_constraints([dict(a, montage="x")])
    with pytest.raises(RV.RegistryV3Error, match="missing"):
        RV.regime_constraints([a, dict(a, n_sensors=None)])
    # a paper-2 recording matches by what it can declare (no lead-field seed)
    assert RV.regime_failure(cons, {k: v for k, v in EEG_REGIME.items()}) is None


# --------------------------------------------------------------------------
# entries, v1 registry, round trips
# --------------------------------------------------------------------------
@pytest.mark.parametrize("change, msg", [
    ({"evidence": {}}, "run_id"),
    ({"version": "ram-v3-*"}, "pinned"),
    ({"substrate": "eeg"}, "empirical"),
    ({"substrate": "*"}, "explicit"),
    ({"failing": []}, "failing"),
    ({"admitted_for_present": "maybe"}, "admission values"),
    ({"admitted_for_present": "vacuous"}, "ABSENT direction"),
    ({"admitted_for_present": "not_observable"}, "both directions"),
    ({"regime": {"montage": "low"}}, "unknown keys"),
    ({"criteria": {"FMz": {"pass": True}}}, "unknown criterion"),
    ({"extra": 1}, "unknown keys"),
])
def test_entry_validation(change, msg):
    raw = entry(present="yes", absent="no")
    raw.update(change)
    with pytest.raises(RV.RegistryV3Error, match=msg):
        RV.check_entry(raw)


def test_admission_needs_the_criteria_behind_it():
    raw = entry(present="yes", absent="vacuous")
    raw["criteria"]["FMd"] = {"pass": False}
    raw["failing"] = ["FMd"]
    with pytest.raises(RV.RegistryV3Error, match="PRESENT admission"):
        RV.check_entry(raw)
    raw = entry(present="no", absent="vacuous", pi0=0.4)  # vacuous needs pi0 < 0.1
    with pytest.raises(RV.RegistryV3Error, match="pi0"):
        RV.check_entry(raw)
    raw = entry(principle="NAS", present="yes", absent="yes", version="nas-v3-2026.10")
    del raw["criteria"]["FMb2"]
    with pytest.raises(RV.RegistryV3Error, match="FMb2"):
        RV.check_entry(raw)
    # a non-admitting entry needs no evidence
    raw = entry(present="no", absent="no", evidence={})
    assert RV.check_entry(raw).admitted_for_absent == "no"


def test_the_v1_registry_reads_as_v3():
    reg1 = v1.ApplicabilityRegistry.from_json(str(REGISTRY_V1))
    reg = RV.RegistryV3.from_json(str(REGISTRY_V1))
    assert len(reg.entries) == len(reg1.entries) and all(e.legacy for e in reg.entries)
    srpi = reg1.entries[0]
    regime = {"n_time": 12000, "n_nodes": 30, "fs_or_tr": 0.05}
    ok1, _ = reg1.is_validated("SRPI", f"{srpi.estimator}@{srpi.version}",
                               srpi.substrate, **regime)
    out = reg.admission("SRPI", f"{srpi.estimator}@{srpi.version}", srpi.substrate,
                        **regime)
    assert ok1 and (out["present"], out["absent"]) == ("yes", "yes")
    out = reg.admission("SRPI", f"{srpi.estimator}@{srpi.version}", srpi.substrate,
                        **dict(regime, n_time=100))
    assert out["present"] == "no" and out["why"] == "REGIME_MISMATCH:T_min"
    again = RV.RegistryV3.from_dict(reg.to_dict())
    assert again.hash() == reg.hash()
    assert RV.resolve_registry_v3(reg1).hash() == reg.hash()


def test_round_trip_and_hash(tmp_path):
    reg = RV.RegistryV3([entry(present="yes", absent="vacuous"),
                         entry(principle="NAS", version="nas-v3-2026.10",
                               present="not_observable", absent="not_observable",
                               substrate="bold_like_forward")], version="t1")
    path = tmp_path / "reg.json"
    reg.to_json(path)
    back = RV.RegistryV3.from_json(path)
    assert back.hash() == reg.hash() and back.to_dict() == reg.to_dict()
    assert RV.resolve_registry_v3(str(path)).hash() == reg.hash()
    assert reg.to_dict()["schema"] == "impact-mpc-registry/3"
    assert all(e["scope_note"] == RV.SCOPE_NOTE for e in reg.to_dict()["entries"])
    with pytest.raises(RV.RegistryV3Error):
        RV.RegistryV3.from_dict({"schema": "impact-mpc-registry/9", "entries": []})
    with pytest.raises(RV.RegistryV3Error):
        RV.RegistryV3([], criteria={"fma_bound_typo": 0.1})


# --------------------------------------------------------------------------
# the builder on synthetic task records of the forward design
# --------------------------------------------------------------------------
H64 = "a" * 64


def _component(p, status, c, reason=None, version="pdi-v3-2026.10"):
    if status == U_ and reason is None:
        reason = R.INCONCLUSIVE
    return REC.ComponentRecord(
        principle=p, status=status, reason=reason, estimator_version=version,
        declaration_id="R", observation_stage="sensor", protocol_id="fwd",
        protocol_hash=H64, c=c)


def _record(task, statuses, regimes):
    """A task record of the forward design; family-A run lengths differ by
    seed, so the regime's duration does too."""
    scorings = []
    for plan in D.scoring_plan(task):
        if plan["estimator_form"] != D.PRIMARY:
            continue
        view, stage = plan["view"], plan["observation_stage"]
        comps = {}
        for p in plan["principles"]:
            if p != "PDI":
                continue
            st, c = statuses(task, view)
            comps[p] = REC.ComponentRecord(
                principle=p, status=st, reason=(R.INCONCLUSIVE if st == U_ else None),
                estimator_version="pdi-v3-2026.10", declaration_id="R",
                observation_stage=stage, protocol_id=f"fwd-{view}", protocol_hash=H64,
                c=c)
        scorings.append(REC.ScoringRecord(
            scoring_id=plan["scoring_id"], declaration_id="R", observation_stage=stage,
            view=view, estimator_form=plan["estimator_form"],
            protocol_id=f"fwd-{view}", protocol_hash=H64, components=comps,
            details={**regimes[view], "regime": {
                **regimes[view]["regime"],
                "duration_s": regimes[view]["regime"]["duration_s"]
                + 0.05 * (task.seed % 5)}}))
    return REC.TaskRecord(
        task_id=task.task_id, design=D.record_design(task.arm, task.purpose),
        family=task.family,
        system=task.system, seed=task.seed,
        generator_version="mpc-bench-generators/2.0.0",
        status="ok", config=task.to_config(),
        simulation={"ts_sha256": H64, "n_nodes": 30, "n_time": 100, "dt": 0.05},
        scorings=tuple(scorings))


@pytest.fixture(scope="module")
def eeg_details():
    """The scoring details of a realised dry-run task of the EEG arm."""
    task = [t for t in D.build_tasks(D.DRY_RUN, arms=[D.ARM_A_EEG])
            if t.condition == "N_ar1"][0]
    real = D.realise(task)
    return {v: F2.scoring_details(s) for v, s in real.views.items()}


def _statuses(task, view):
    """PDI on the EEG views: specific, monotone, concordant, safe."""
    k = {"PC_nominal": 6, "PC_nominal_K1": 1, "PC_nominal_K2": 2,
         "PC_nominal_K3": 3}.get(task.condition)
    if k is None:  # null classes
        return A_, 0.0
    c = (k - 1) / 5.0 + 0.001 * (task.seed % 7)
    if view == "source":
        c = 1.2 * c
    return (P_ if k >= 2 else U_), c


def test_builder_on_synthetic_records(tmp_path, eeg_details):
    tasks = D.build_tasks(D.DRY_RUN, arms=[D.ARM_A_EEG])
    recs = [_record(t, _statuses, eeg_details) for t in tasks]
    res = tmp_path / "results.jsonl"
    REC.write_jsonl(recs, res)
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps({"anchors": [
        {"arm": D.ARM_A_EEG, "view": "eeg64", "principle": "PDI", "valid": True},
        {"arm": D.ARM_A_EEG, "view": "eeglow", "principle": "PDI", "valid": False}]}))
    out, rep = tmp_path / "reg.json", tmp_path / "report.json"
    rc = B.main(["--results", str(res), "--anchors", str(anchors), "--run-id", "dry-1",
                 "--purpose", D.DRY_RUN, "--out", str(out), "--report", str(rep)])
    assert rc == 0
    reg = RV.RegistryV3.from_json(out)
    by_view = {e.view: e for e in reg.entries}
    assert set(by_view) == {"eeg64", "eeglow"}
    e64 = by_view["eeg64"]
    # dry-run scale: 10 seeds per null class cannot demonstrate 0.07, and
    # 22 on-runs cannot demonstrate 0.02: neither direction is admitted
    assert (e64.admitted_for_present, e64.admitted_for_absent) == ("no", "no")
    assert set(e64.failing) == {"FMa", "FMabs"}
    assert e64.criteria["FMb1"]["pass"] and e64.criteria["FMd"]["pass"]
    assert e64.substrate == "eeg_like_forward" and e64.version == "pdi-v3-2026.10"
    assert e64.estimator == "compute_PDI:*" and e64.arm == D.ARM_A_EEG
    assert e64.regime["n_sensors"] == 64 and e64.regime["sensor_filter"] == "none"
    assert "leadfield_seed" not in e64.regime
    assert e64.regime_keys["leadfield_seed"] == 20260928
    assert isinstance(e64.regime["duration_s"], dict)  # agent runs differ in length
    assert by_view["eeglow"].failing[0] == "FM0"
    assert by_view["eeglow"].regime["n_sensors"] == 32
    assert reg.provenance["purpose"] == D.DRY_RUN
    assert reg.provenance["split"] == "development"
    report = json.loads(rep.read_text())
    assert report["plan"]["missing_by_arm"] == {
        D.ARM_HOPF: 50, D.ARM_A_EEG: 0, D.ARM_A_BOLD: 35}
    assert report["plan"]["n_records"] == len(tasks)
    assert report["plan"]["unexpected"] == []
    assert report["registry_sha256"] == reg.hash()


def test_builder_refusals(tmp_path, eeg_details):
    tasks = D.build_tasks(D.DRY_RUN, arms=[D.ARM_A_EEG])[:3]
    recs = [_record(t, _statuses, eeg_details) for t in tasks]
    with pytest.raises(B.BuildError, match="registry is built"):
        B.admission_runs(recs, D.REFERENCE)
    with pytest.raises(Exception, match="confirmatory"):
        B.admission_runs(recs, D.CONFIRMATORY)  # development seeds
    with pytest.raises(B.BuildError, match="purpose"):
        B.admission_runs(recs, D.DEV_REGIME)
    with pytest.raises(B.BuildError, match="run id"):
        B.build_registry(recs, {}, run_id=" ", purpose=D.DRY_RUN)
    bad = REC.TaskRecord.from_dict({**recs[0].to_dict(), "config": {}})
    with pytest.raises(B.BuildError, match="config"):
        B.admission_runs([bad], D.DRY_RUN)
    res = tmp_path / "r.jsonl"
    res.write_text("".join(REC.dumps(r) + "\n" for r in recs + recs[:1]))
    with pytest.raises(B.BuildError, match="duplicate"):
        B.read_records([res])
    # records outside the frozen plan never enter an admission: an extra seed
    # of a planned condition, or a view observed at another regime
    k1 = [t for t in D.build_tasks(D.DRY_RUN, arms=[D.ARM_A_EEG])
          if t.condition == "PC_nominal_K1"][0]  # 3 dry-run seeds: 384-386
    extra = D.ForwardTask(**{**k1.to_dict(), "views": k1.views, "seed": 399,
                             "task_id": k1.task_id.replace("s00384", "s00399")})
    assert extra.task_id not in {t.task_id for t in D.build_tasks(D.DRY_RUN)}
    with pytest.raises(B.BuildError, match="not a task of the dry_run plan"):
        B.admission_runs(recs + [_record(extra, _statuses, eeg_details)], D.DRY_RUN)
    moved = REC.TaskRecord.from_dict({**recs[0].to_dict(), "seed": 390})
    with pytest.raises(B.BuildError, match="differ from the plan"):
        B.admission_runs([moved], D.DRY_RUN)
    held = {v: {**d, "regime_name": "held_out"} for v, d in eeg_details.items()}
    with pytest.raises(B.BuildError, match="held_out regime"):
        B.admission_runs([_record(tasks[0], _statuses, held)], D.DRY_RUN)
    assert B.load_anchors({"hopf/eeg64/NAS": True}) == {
        ("hopf", "eeg64", "NAS"): {"valid": True}}
    with pytest.raises(B.BuildError):
        B.load_anchors({"hopf/eeg64": True})
    rc = B.main(["--results", str(tmp_path / "missing"), "--anchors", "{}",
                 "--run-id", "x", "--out", str(tmp_path / "o.json")])
    assert rc == 1


def test_the_builder_reads_the_arms_by_their_record_designs(tmp_path, eeg_details):
    """One table names the records of each arm (the forward design module's
    RECORD_DESIGN_OF_ARM): the builder reads exactly those designs, ignores
    the anchor replication and every other design, and refuses a record
    filed under another arm's design."""
    assert dict(D.RECORD_DESIGN_OF_ARM) == {
        D.ARM_HOPF: "whole_brain", D.ARM_A_EEG: "forward_family_a",
        D.ARM_A_BOLD: "forward_family_a_bold"}
    assert D.ADMISSION_RECORD_DESIGNS == frozenset(D.RECORD_DESIGN_OF_ARM.values())
    assert D.ANCHOR_REPLICATION_DESIGN not in D.ADMISSION_RECORD_DESIGNS
    for purpose in D.ARM_PURPOSES:
        assert D.record_design(D.ARM_A_EEG, purpose) == "forward_family_a"
    for purpose in D.ANCHOR_PURPOSES:
        assert D.record_design(D.ARM_HOPF, purpose) == D.ANCHOR_REPLICATION_DESIGN
    tasks = D.build_tasks(D.DRY_RUN, arms=[D.ARM_A_EEG])[:3]
    recs = [_record(t, _statuses, eeg_details) for t in tasks]
    other = [REC.TaskRecord.from_dict({**recs[0].to_dict(), "design": d,
                                       "task_id": f"{d}-x"})
             for d in ("forward", D.ANCHOR_REPLICATION_DESIGN, "A_witnesses")]
    res = tmp_path / "r.jsonl"
    REC.write_jsonl(recs + other, res)
    got, _ = B.read_records([res])
    assert [r.task_id for r in got] == [r.task_id for r in recs]
    filed = REC.TaskRecord.from_dict({**recs[0].to_dict(),
                                      "design": "forward_family_a_bold"})
    with pytest.raises(B.BuildError, match="the arm forward_a_eeg records"):
        B.admission_runs([filed], D.DRY_RUN)
