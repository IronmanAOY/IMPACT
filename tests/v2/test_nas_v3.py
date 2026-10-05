# -*- coding: utf-8 -*-
"""NAS v3 (nas-v3-2026.10): equality with the frozen v1 statistic on the v1
geometry, the resolvability gate against each substrate's coupling time scale
(families A and C resolved at every sensitivity time scale, BOLD-like data
unresolved), descriptors isolated from the gated value, the lag sets, the
per-direction statistics with their own anchors and the per-block z, the
removal of a planted shared input by conditioning while a planted edge stays
detectable, the shift-then-project null, the jackknife groups, the budget by
the effective rank of the input basis and the secondary rank caps, the
observation gate and the identifiability record. Simulations use
development seeds only."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_pipeline import evidence_v2 as E
from impact_pipeline import mpc_metrics as mm
from impact_pipeline.bench import forward as FWD
from impact_pipeline.bench import generators as G
from impact_pipeline.bench import whole_brain as WB
from impact_pipeline.v2 import ESTIMATOR_VERSIONS_V2
from impact_pipeline.v2 import declared_inputs as DI
from impact_pipeline.v2 import nas_v3 as N
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import records as REC

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "protocols" / "v2" / "mpc_bench_v2_template.json"
DT = 0.05
TAU_C = 0.1
P = {"coupling_timescale_sec": TAU_C}
FAST = {"coupling_timescale_sec": TAU_C, "descriptors": ()}
SEED_A = 5  # development seeds (0-999) only
SEED_C = 5
BLOCK_TAUS = {"W": 0.3, "P1": 0.1, "P2": 0.1, "P3": 1.0}  # inside the basis taus


# --------------------------------------------------------------------------
# synthetic systems
# --------------------------------------------------------------------------
def switching_driver(rng, n_time, dt=DT):
    """A six-level switching driver with dwell U(1, 3) s."""
    u = np.zeros(n_time)
    levels = rng.standard_normal(6)
    t = 0
    while t < n_time:
        dur = max(1, int(round(rng.uniform(1.0, 3.0) / dt)))
        u[t:t + dur] = levels[rng.integers(0, 6)]
        t += dur
    return u


def varx(seed, n_time=6000, gain=0.0, loop=0.0, ret=0.0, units=3, unit_sd=0.3,
         dt=DT):
    """
    Hub W and periphery P1-P3 (``units`` nodes each). Every block carries a
    first-order low-pass of one shared switching input ``u`` (time constant
    per block from BLOCK_TAUS, gain ``gain``; no edge) plus its own AR(1)
    process; ``loop`` couples W and P1 in both directions, ``ret`` couples
    W -> P2 only. Returns ``(ts, u, hub, blocks)``.
    """
    rng = np.random.default_rng(seed)
    u = switching_driver(rng, n_time, dt)
    w = rng.standard_normal((4, n_time))
    e = np.zeros((4, n_time))
    for k in range(1, n_time):
        prev = e[:, k - 1]
        e[:, k] = 0.5 * prev + w[:, k]
        e[0, k] += loop * prev[1]
        e[1, k] += loop * prev[0]
        e[2, k] += ret * prev[0]
    rows, blocks = [], {}
    for b, name in enumerate(BLOCK_TAUS):
        y = gain * DI.lowpass(u, dt, BLOCK_TAUS[name]) + e[b]
        rows.append(y[None, :] + unit_sd * rng.standard_normal((units, n_time)))
        blocks[name] = list(range(b * units, (b + 1) * units))
    hub = blocks.pop("W")
    return np.vstack(rows), u, hub, blocks


def system(seed, **kw):
    """``(ts, hub, blocks)`` of :func:`varx`."""
    ts, _u, hub, blocks = varx(seed, **kw)
    return ts, hub, blocks


def declared_channels(channels, n_time, dt=DT, declaration_id="R",
                      shared="complete"):
    """Declared continuous inputs (no events) on the sample grid."""
    return DI.DeclaredInputs(
        declaration_id=declaration_id, shared_inputs=shared, n_time=n_time, dt=dt,
        seed=None, events=pd.DataFrame(columns=list(DI.DECLARED_EVENT_COLUMNS)),
        cue_alphabets={}, channels=dict(channels))


def run(ts, hub, blocks, inputs=None, params=P, seed=0, dt=DT, **kw):
    return N.compute_nas_v3(ts, dt=dt, hub=hub, blocks=blocks, inputs=inputs,
                            params=params, seed=seed, hub_name="W", **kw)


def zs(result):
    return {d: result["directions"][d]["z"] for d in N.DIRECTIONS}


def excess(result):
    return {d: result["directions"][d]["excess"] for d in N.DIRECTIONS}


@pytest.fixture(scope="module")
def family_a():
    return G.simulate_family_a(seed=SEED_A)


@pytest.fixture(scope="module")
def family_c():
    return G.simulate_family_c(seed=SEED_C)


@pytest.fixture(scope="module")
def template():
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------
def test_declarations_match_the_status_rule_and_the_template(template):
    assert N.ESTIMATOR_VERSION == "nas-v3-2026.10"
    assert N.ESTIMATOR_VERSION in ESTIMATOR_VERSIONS_V2["NAS"]
    assert N.ESTIMATOR_ID == "compute_NAS:conditional_capacity@nas-v3-2026.10"
    assert template["estimator_versions"]["NAS"] == N.ESTIMATOR_VERSION
    assert template["null_families"]["NAS"] == N.NULL_FAMILY
    assert tuple(template["directions"]["NAS"]) == N.DIRECTIONS == E.NAS_DIRECTIONS
    assert sorted(template["se_methods"]["NAS"]) == sorted(N.SE_METHODS)
    for name in N.SE_METHODS:
        m = E.SE_METHODS[name]
        assert "NAS" in m.principles and m.sampling_se
        assert m.df_values == (float(int(name.rsplit("_", 1)[1]) - 1),)
    assert N.SE_METHOD_DEFAULT == "jackknife_contiguous_10"
    assert E.is_quadratic(N.ESTIMATOR_VERSION)
    assert N.NULL_ORDER == DI.NULL_ORDER == "shift_then_project"
    assert N.LAG_TIMESCALE_SENSITIVITY_SEC == (0.05, 0.2)
    # the template's NAS block is read by the estimator under its own names
    p = N.NASParams.from_mapping(template["estimators"]["NAS"])
    assert (p.mode, p.block_representation, p.coupling_timescale_sec) == (
        "conditional_capacity", "block_mean", 0.1)
    assert (p.n_null, p.null_min_shift_fraction, p.se_method) == (
        19, 0.1, "jackknife_contiguous_10")
    assert p.secondary_dimensions == (8, 4, 2, 1)
    assert (p.rank_fraction, p.budget_factor) == (0.8, 10.0)
    assert p.lag_timescale() == p.coupling_timescale() == 0.1
    assert N.NASParams.from_mapping(json.loads(json.dumps(p.to_dict()))) == p


@pytest.mark.parametrize("bad", [
    {"mode": "capacity"}, {"block_representation": "pca"}, {"n_null": 1},
    {"se_method": "jackknife_delete_group_10"}, {"coupling_timescale_sec": 0},
    {"lag_timescale_sec": -0.1}, {"secondary_dimensions": (4, 8)},
    {"descriptors": ("pooled_v2",)}, {"rank_fraction": 1.5}, {"bands": []},
])
def test_malformed_parameters_are_refused(bad):
    with pytest.raises(N.NASError):
        N.NASParams.from_mapping(bad)


def test_the_coupling_time_scale_is_a_declaration():
    ts, _u, hub, blocks = varx(1, n_time=400)
    with pytest.raises(N.NASError, match="coupling_timescale_sec"):
        run(ts, hub, blocks, params={})


# --------------------------------------------------------------------------
# v1 equality on the v1 geometry (integrity audit IA-8)
# --------------------------------------------------------------------------
V1_KEYS = ("te_in", "te_out", "te_in_null_mean", "te_out_null_mean",
           "te_in_null_sd", "te_out_null_sd", "te_in_excess", "te_out_excess")


def _v1(ts, hub, lags, comps, null_seed):
    return mm.compute_NAS(ts, tr=DT, workspace_nodes=hub, mode="capacity",
                          null_surrogates=19, null_seed=null_seed,
                          transfer_lags=lags, transfer_components=comps,
                          return_details=True)


def _assert_v1_equal(v1, v3):
    tr = v1["transfer"]
    for k in V1_KEYS:
        assert abs(tr[k] - v3[k]) <= 1e-12, (k, tr[k], v3[k])
    for k in ("te_in_z", "te_out_z"):
        assert v3[k] == pytest.approx(tr[k], rel=1e-9, abs=1e-12)
    assert (tr["te_in_p"], tr["te_out_p"]) == (v3["te_in_p"], v3["te_out_p"])
    assert v1["limiting_direction"] == v3["limiting_direction"]
    assert abs(v1["value"] - v3["value"]) <= 1e-12
    assert abs(v1["raw"] - v3["raw"]) <= 1e-12
    assert abs(v1["NAS_null_mean"] - v3["null_mean"]) <= 1e-12


def test_v3_reproduces_the_frozen_v1_statistic_on_the_v1_geometry(family_a):
    s = family_a
    hub = s.meta["workspace_nodes"]
    per = np.setdiff1d(np.arange(s.n_nodes), hub)
    seed = N.null_seed_for(SEED_A)
    v1 = _v1(s.ts, hub, (1, 2), 5, seed)
    v3 = N.pooled_statistic(s.ts, hub, per, (1, 2), n_components=5, null_seed=seed)
    _assert_v1_equal(v1, v3)
    # the descriptor of a v3 run is the same computation with the same seeds
    r = N.nas_v3_system(s, "H", params={"descriptors": ("pooled_v1",)})
    d = r["details"]["descriptors"]["pooled_v1"]
    _assert_v1_equal(v1, d)
    assert d["null_shifts"] == r["details"]["null_shifts"]


@pytest.mark.parametrize("lags,comps", [((1, 2), 5), ((1, 3, 9), 2), ((2,), 1)])
def test_v3_reproduces_v1_on_synthetic_geometries(lags, comps):
    ts, _u, hub, blocks = varx(3, n_time=2500, gain=1.0, loop=0.3)
    per = sorted(i for v in blocks.values() for i in v)
    v1 = _v1(ts, hub, lags, comps, 77)
    v3 = N.pooled_statistic(ts, hub, per, lags, n_components=comps, null_seed=77)
    _assert_v1_equal(v1, v3)


# --------------------------------------------------------------------------
# resolvability gate, lag sets and descriptors
# --------------------------------------------------------------------------
def test_lag_sets_at_20_250_and_500_hz():
    for dt, lags in ((0.05, [1, 2]), (0.004, [1, 3, 9, 25]), (0.002, [1, 4, 14, 50])):
        plan = N.lag_plan(TAU_C, dt)
        assert plan["resolved"] and plan["lags"] == lags
        assert plan["l_max"] == lags[-1] and plan["basis_lags"] == [0] + lags
        assert plan["basis_taus"] == [0.1, 0.3, 1.0]
    # the sensitivity time scales change the lags and the basis, not the gate
    assert N.lag_plan(TAU_C, 0.05, 0.05)["lags"] == [1]
    assert N.lag_plan(TAU_C, 0.05, 0.2)["lags"] == [1, 2, 3, 4]
    assert N.lag_plan(TAU_C, 0.05, 0.05)["basis_taus"] == [0.05, 0.15, 0.5]
    for tl in (0.05, 0.1, 0.2):
        assert N.lag_plan(TAU_C, 0.05, tl)["resolved"]
        assert not N.lag_plan(TAU_C, 2.0, tl)["resolved"]
    # a run at 250 Hz uses the 250 Hz lag set and sizes the jackknife blocks
    ts, _u, hub, blocks = varx(4, n_time=4000, loop=0.3, dt=0.004)
    r = run(ts, hub, blocks, dt=0.004, params=FAST)
    assert r["defined"] and r["details"]["lags"] == [1, 3, 9, 25]
    assert N.interleave_block_samples(0.004, TAU_C) == 1250
    assert N.interleave_block_samples(0.05, TAU_C) == 100
    assert N.interleave_block_samples(0.05, 0.3) == 300


@pytest.mark.parametrize("lag_scale", [None, 0.05, 0.2])
def test_families_a_and_c_are_resolved_at_every_sensitivity_time_scale(
        family_a, family_c, lag_scale):
    """The gate reads each substrate's coupling time scale (family A rate
    units tau = 0.1 s, family C Stuart-Landau amplitude rate 1 / sl_rate =
    0.1 s) against its sampling interval (0.05 s): NAS is never undefined on
    the bench by construction, whatever lag time scale is reported."""
    for s, constant in ((family_a, "AgentConfig.tau"),
                        (family_c, "AgentConfig.sl_rate")):
        tc = DI.coupling_timescale(s)
        assert (tc.tau_c_sec, tc.constant, float(s.dt)) == (0.1, constant, 0.05)
        assert N.system_coupling_timescale(s) == 0.1
        r = N.nas_v3_system(s, "R", params={"lag_timescale_sec": lag_scale,
                                            "descriptors": ()})
        assert r["defined"] and r["reason"] is None, r["reason"]
        assert r["details"]["lag_plan"]["coupling_timescale_sec"] == 0.1
        plan = N.lag_plan(0.1, 0.05, lag_scale)
        assert r["details"]["lags"] == plan["lags"]
        # the filtered input copies follow the lag time scale as well
        assert r["details"]["basis"]["taus"] == plan["basis_taus"]
        assert all(math.isfinite(r["directions"][d]["estimate"]) for d in N.DIRECTIONS)
    # the family-C carriers (about 1.2 Hz) are far below the Nyquist frequency
    # of the 20 Hz recording, so the carrier does not limit resolvability
    f = np.asarray(family_c.oracle["oscillator_frequency_hz"], dtype=float)
    assert f.max() < 0.25 / float(family_c.dt)


def test_a_declared_coupling_time_scale_that_contradicts_the_generator_raises(
        family_a):
    with pytest.raises(N.NASError, match="lag_timescale_sec"):
        N.nas_v3_system(family_a, "R", params={"coupling_timescale_sec": 0.05})
    # a substrate whose coupling is faster than the sampling is unresolved
    ts, _u, hub, blocks = varx(2, n_time=1000, loop=0.3)
    r = run(ts, hub, blocks, params={"coupling_timescale_sec": 0.05})
    assert (r["defined"], r["reason"]) == (False, R.SAMPLING_UNRESOLVED)
    assert run(ts, hub, blocks, params={"coupling_timescale_sec": 0.1,
                                        "descriptors": ()})["defined"]


def test_sampling_unresolved_at_tr_2s_without_exception(family_a):
    """A BOLD-like view at TR = 2 s: UNDEFINED(SAMPLING_UNRESOLVED) at every
    lag time scale, never ABSENT, no exception; the v1 bench band is above
    Nyquist and only marks the descriptor. The frozen v1 estimator still
    raises its band error on the same data."""
    b = FWD.bold_forward(family_a, tr=2.0, seed=SEED_A)
    assert float(b.dt) == 2.0 and N.observation_of(b.meta) == "hemodynamic"
    for lag_scale in (None, 0.05, 0.2):
        r = N.nas_v3_system(b, "R", params={"lag_timescale_sec": lag_scale})
        assert (r["defined"], r["reason"]) == (False, R.SAMPLING_UNRESOLVED)
        assert R.status_for_reason(r["reason"]) == R.UNDEFINED
        meta = r["details"]["descriptors"]["metastability"]
        assert meta["reason"] == N.BAND_ABOVE_NYQUIST and meta["value"] is None
        assert r["identifiability"]["observation"] == "hemodynamic"
        items = N.evidence_items(r)
        assert all(not ev.defined and ev.reason == R.SAMPLING_UNRESOLVED
                   for ev in items)
        pa = E.assess_principle("NAS", items, E.ProtocolV3(
            reference={"kind": "external", "scale": "excess",
                       "values": {"NAS:receive": 1.0, "NAS:return": 1.0}},
            directions={"NAS": list(N.DIRECTIONS)}))
        assert (pa.status.value, pa.reason) == (R.UNDEFINED, R.SAMPLING_UNRESOLVED)
        fields = N.component_fields(r)
        assert fields["estimate"] is None and fields["se"] is None
    with pytest.raises(ValueError, match="Nyquist"):
        mm.compute_NAS(b.ts, tr=b.dt, workspace_nodes=b.meta["workspace_nodes"],
                       mode="capacity", null_surrogates=19, null_seed=1,
                       return_details=True, **N.PROFILE_PARAMETERS)
    # the same gate on a synthetic recording sampled every 2 s
    ts, _u, hub, blocks = varx(6, n_time=300, loop=0.3, dt=2.0)
    r = run(ts, hub, blocks, dt=2.0)
    assert (r["defined"], r["reason"]) == (False, R.SAMPLING_UNRESOLVED)


@pytest.mark.parametrize("tr", [0.72, 2.0])
def test_every_bold_view_of_the_hopf_arm_is_sampling_unresolved(tr):
    """IA-3 on the Hopf arm: the BOLD view of the whole-brain model keeps the
    source's coupling time scale (1 / rate = 0.1 s) and is UNDEFINED
    (SAMPLING_UNRESOLVED) at a multiband and a conventional TR, while the
    source view of the same run is defined. Only the signal descriptors run
    for the undefined value."""
    w = WB.simulate_whole_brain(WB.WholeBrainConfig(duration_sec=40.0), seed=4)
    b = FWD.bold_forward(w, tr=tr, seed=4)
    assert N.system_coupling_timescale(b) == N.system_coupling_timescale(w) == 0.1
    assert N.observation_of(b.meta) == "hemodynamic"
    r = N.nas_v3_system(b, "none")
    assert (r["defined"], r["reason"]) == (False, R.SAMPLING_UNRESOLVED)
    assert set(r["details"]["descriptors"]) == {"metastability"}
    meta = r["details"]["descriptors"]["metastability"]
    assert meta["reason"] == N.BAND_ABOVE_NYQUIST
    assert N.nas_v3_system(w, "none", params={"descriptors": ()})["defined"]


def test_descriptors_never_touch_the_gated_value(monkeypatch):
    ts, _u, hub, blocks = varx(7, n_time=2000, loop=0.3)
    params = dict(P, metastability_bands=((0.5, 12.0),))  # above Nyquist (10 Hz)
    ref = run(ts, hub, blocks, params=params)
    assert ref["defined"]
    meta = ref["details"]["descriptors"]["metastability"]
    assert meta["reason"] == N.BAND_ABOVE_NYQUIST

    def boom(*_a, **_k):
        raise RuntimeError("descriptor failure")

    for name in ("_metastability", "_bic_order", "_zero_lag", "pooled_statistic"):
        monkeypatch.setattr(N, name, boom)
    r = run(ts, hub, blocks, params=dict(params, descriptors=N.ALL_DESCRIPTORS))
    assert r["defined"] and r["directions"] == ref["directions"]
    desc = r["details"]["descriptors"]
    for name in ("metastability", "bic_order", "zero_lag_coupling", "pooled_v1",
                 "pooled_rank1"):
        assert desc[name]["error"] == "RuntimeError", name
    assert "receive" in desc["conditioning_delta"]  # unaffected descriptor


def test_descriptors_are_reported(family_a):
    r = N.nas_v3_system(family_a, "R")
    d = r["details"]["descriptors"]
    assert set(d) == set(N.DEFAULT_DESCRIPTORS)
    assert d["pooled_v1"]["n_components"] == 5 and d["pooled_v1"]["defined"]
    assert d["pooled_rank1"]["n_components"] == 1
    rank1 = d["pooled_rank1"]
    assert rank1["components_hub"] == rank1["components_periphery"] == 1
    assert isinstance(d["pooled_rank1"]["significant"], bool)
    assert 0.0 <= d["zero_lag_coupling"]["value"] <= 1.0
    assert d["bic_order"]["order"] in (1, 2)
    assert [row["lags"] for row in d["bic_order"]["table"]] == [[1], [1, 2]]
    assert d["metastability"]["reason"] is None and d["metastability"]["value"] > 0
    delta = d["conditioning_delta"]
    for k in N.DIRECTIONS:
        assert delta[k] == pytest.approx(
            r["directions"][k]["excess"] - delta["excess_without_inputs"][k])
    # the same simulation without inputs has no conditioning correction
    r0 = N.nas_v3_system(family_a, "none", params={"descriptors": (
        "conditioning_delta",)})
    assert r0["details"]["descriptors"]["conditioning_delta"] == {
        "receive": 0.0, "return": 0.0, "inputs": False}
    assert r0["directions"][N.RECEIVE]["excess"] == pytest.approx(
        delta["excess_without_inputs"][N.RECEIVE], rel=1e-9)
    # the legacy profile descriptor is opt-in
    rp = N.nas_v3_system(family_a, "none", params={"descriptors": ("profile",)})
    prof = rp["details"]["descriptors"]["profile"]
    assert prof["reason"] is None and all(math.isfinite(prof[k]) for k in "LBH")


def test_the_profile_parameters_are_the_v1_bench_parameters():
    from impact_pipeline.bench import export

    v1 = export.BENCH_ESTIMATOR_PARAMS["NAS"]
    for k, v in N.PROFILE_PARAMETERS.items():
        if k == "bands":
            assert [tuple(b) for b in v1[k]] == v
        else:
            assert v1[k] == v
    assert tuple(tuple(b) for b in v1["bands"]) == N.METASTABILITY_BANDS


# --------------------------------------------------------------------------
# per-direction statistics, anchors and per-block z
# --------------------------------------------------------------------------
def test_per_direction_statistics_and_per_block_z():
    """W <-> P1 coupled both ways, W -> P2 only, P3 independent: per block,
    P1 carries receive and return, P2 return only, P3 neither; R and B are
    the block means; every z is recomputed from the stored null draws."""
    ts, _u, hub, blocks = varx(11, loop=0.4, ret=0.4)
    r = run(ts, hub, blocks)
    assert r["defined"] and r["significant"]
    pb = r["details"]["per_block"]
    assert list(pb) == ["P1", "P2", "P3"]
    assert pb["P1"]["receive"]["z"] > 10 and pb["P1"]["return"]["z"] > 10
    assert pb["P2"]["return"]["z"] > 10 and abs(pb["P2"]["receive"]["z"]) < 3
    assert abs(pb["P3"]["receive"]["z"]) < 3 and abs(pb["P3"]["return"]["z"]) < 3
    assert pb["P1"]["bidirectional"] and not pb["P3"]["bidirectional"]
    n_bidir = sum(v["bidirectional"] for v in pb.values())
    assert r["details"]["n_bidirectional"] == n_bidir
    assert r["details"]["coverage"] == pytest.approx(n_bidir / 3)
    for d, key in ((N.RECEIVE, "receive"), (N.RETURN, "return")):
        dd = r["directions"][d]
        assert dd["statistic"] == N.DIRECTION_STATISTICS[d]
        observed = [pb[k][key]["observed"] for k in pb]
        assert dd["estimate"] == pytest.approx(np.mean(observed))
        null = np.asarray(dd["null_values"])
        assert null.size == dd["n_null"] == 19
        assert dd["null_mean"] == pytest.approx(null.mean())
        assert dd["null_sd"] == pytest.approx(null.std(ddof=1))
        assert dd["excess"] == pytest.approx(dd["estimate"] - null.mean())
        assert dd["z"] == pytest.approx(dd["excess"] / null.std(ddof=1))
        assert dd["p"] == pytest.approx((1 + np.sum(null >= dd["estimate"])) / 20)
        for k in pb:
            c = pb[k][key]
            pn = np.asarray(c["null_values"])
            assert c["z"] == pytest.approx((c["observed"] - pn.mean()) / pn.std(ddof=1))
        # the block means of the null draws are the direction's null draws
        per_null = np.mean([pb[k][key]["null_values"] for k in pb], axis=0)
        assert np.allclose(per_null, null, rtol=1e-12, atol=0)
    assert r["limiting_direction"] == min(N.DIRECTIONS, key=lambda d: zs(r)[d])


def test_a_pseudo_hub_reuses_inputs_lags_null_shifts_and_jackknife_groups():
    ts, u, hub, blocks = varx(15, n_time=2000, gain=1.0, loop=0.3)
    decl = declared_channels({"u": u}, ts.shape[1])
    r = run(ts, hub, blocks, inputs=decl, seed=15, params=FAST)
    pseudo_blocks = {"W": hub, **{k: v for k, v in blocks.items() if k != "P1"}}
    q = N.compute_nas_v3(ts, dt=DT, hub=blocks["P1"], blocks=pseudo_blocks,
                         inputs=decl, params=FAST, seed=15, hub_name="P1")
    assert q["defined"] and list(q["details"]["blocks"]) == ["W", "P2", "P3"]
    for key in ("null_shifts", "lags", "jackknife"):
        assert q["details"][key] == r["details"][key], key
    assert q["details"]["basis"]["sha256"] == r["details"]["basis"]["sha256"]
    for d in N.DIRECTIONS:
        n_q = len(q["directions"][d]["jackknife"])
        assert n_q == len(r["directions"][d]["jackknife"]) == 10
    with pytest.raises(N.NASError, match="hub_name"):
        N.compute_nas_v3(ts, dt=DT, hub=hub, blocks=blocks, params=FAST,
                         hub_name="P1")


def test_each_direction_is_scored_on_its_own_anchor(template):
    pcs = [run(*system(s, n_time=3000, loop=0.3, ret=0.2), seed=s, params=FAST)
           for s in (20, 21, 22)]
    ref = N.anchor_reference(pcs)
    assert set(ref["values"]) == {"NAS:receive", "NAS:return"}
    ex = N.direction_excesses(pcs)
    for d in N.DIRECTIONS:
        assert ref["values"][f"NAS:{d}"] == pytest.approx(ex[d].mean())
        assert ref["se"][f"NAS:{d}"] == pytest.approx(ex[d].std(ddof=1) / math.sqrt(3))
    assert ref["values"]["NAS:receive"] != pytest.approx(ref["values"]["NAS:return"])
    payload = dict(template)
    payload["reference"] = {"kind": "external", "scale": "excess",
                            "values": ref["values"], "se": ref["se"]}
    proto = E.load_protocol(payload)
    ts, _u, hub, blocks = varx(23, n_time=3000, loop=0.3, ret=0.2)
    r = run(ts, hub, blocks, seed=23, params=FAST)
    items = N.evidence_items(r, protocol_id=proto.protocol_id)
    assert [ev.direction for ev in items] == list(N.DIRECTIONS)
    pa = E.assess_principle("NAS", items, proto)
    rec = pa.record_fields()
    c_r = r["directions"]["receive"]["excess"] / ref["values"]["NAS:receive"]
    c_b = r["directions"]["return"]["excess"] / ref["values"]["NAS:return"]
    assert rec["c_R"] == pytest.approx(c_r, rel=1e-9)
    assert rec["c_B"] == pytest.approx(c_b, rel=1e-9)
    assert rec["c"] == pytest.approx(min(c_r, c_b), rel=1e-9)
    assert pa.status.value == R.PRESENT
    # explicit per-direction references give the same values
    items2 = N.evidence_items(r, references={
        d: (ref["values"][f"NAS:{d}"], ref["se"][f"NAS:{d}"]) for d in N.DIRECTIONS})
    assert [ev.reference for ev in items2] == [ref["values"]["NAS:receive"],
                                               ref["values"]["NAS:return"]]
    share = N.anchor_attributability(0.0025, 0.0127)
    assert share == pytest.approx(1 - 0.0025 / 0.0127)
    assert math.isnan(N.anchor_attributability(0.1, 0.0))


# --------------------------------------------------------------------------
# conditioning on declared inputs
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [11, 12])
def test_a_planted_shared_input_is_removed_while_a_planted_edge_stays(seed):
    """Synthetic VARX: a shared switching input reaches every block through
    block-specific low-passes and no edge. Undeclared, it makes both
    directions significant; declared, both directions fall to the null. With
    a planted W <-> P1 loop on top, the declared statistic keeps the edge."""
    ts, u, hub, blocks = varx(seed, gain=2.0)
    T = ts.shape[1]
    hidden = run(ts, hub, blocks, seed=seed)
    decl = declared_channels({"u": u}, T)
    shown = run(ts, hub, blocks, inputs=decl, seed=seed)
    assert hidden["significant"] and min(zs(hidden).values()) > 10
    assert hidden["identifiability"]["shared_inputs"] == "none"
    assert shown["identifiability"]["shared_inputs"] == "complete"
    assert not shown["significant"]
    for d in N.DIRECTIONS:
        assert abs(excess(shown)[d]) < 0.05 * excess(hidden)[d]
        assert shown["details"]["descriptors"]["conditioning_delta"][d] < 0
    assert shown["details"]["basis"]["channels"] == ["u"]
    ts2, u2, hub2, blocks2 = varx(seed, gain=2.0, loop=0.3)
    edge = run(ts2, hub2, blocks2, inputs=declared_channels({"u": u2}, T), seed=seed)
    assert edge["significant"] and min(zs(edge).values()) > 10
    p1 = edge["details"]["per_block"]["P1"]
    assert p1["bidirectional"] and p1["receive"]["z"] > 10 and p1["return"]["z"] > 10


def test_the_budget_counts_the_effective_rank_of_the_basis():
    """The basis is rank-deficient by construction (filtered copies at
    contiguous lags are combinations of each other): the budget uses its
    rank. Here the raw column count would fail the T - l_max >= 10 n_par
    rule while the effective rank passes it."""
    T = 180
    ts, u, hub, blocks = varx(8, n_time=T, gain=1.0)
    decl = declared_channels({"u": u}, T)
    basis = DI.input_basis(decl, tau_c=TAU_C, estimator="NAS")
    rank = basis.rank(start=2)
    assert basis.n_columns == 12 and rank < basis.n_columns
    n_par = 1 + rank + 2 * 4
    assert T - 2 >= 10 * n_par and T - 2 < 10 * (1 + basis.n_columns + 2 * 4)
    r = run(ts, hub, blocks, inputs=decl, params=FAST)
    assert r["defined"], r["reason"]
    assert r["details"]["budget"]["rank_u"] == rank
    assert r["details"]["budget"]["n_par"] == n_par
    assert r["details"]["basis"]["n_columns"] == 12
    assert r["details"]["rank_full_model"] <= r["details"]["n_columns"]
    # a prebuilt basis with the NAS lags is accepted; other lags are refused
    r2 = run(ts, hub, blocks, inputs=basis, params=FAST)
    assert r2["directions"] == r["directions"]
    with pytest.raises(N.NASError, match="lags"):
        run(ts, hub, blocks, params=FAST,
            inputs=DI.input_basis(decl, tau_c=TAU_C, estimator="IIM", lags=(0, 1)))


def test_family_a_declarations_and_effective_rank(family_a):
    s = family_a
    rec = DI.record_inputs(s)
    out = {}
    for did, shared in (("R", "complete"), ("H", "partial"), ("none", "none")):
        r = N.nas_v3_system(s, did, recorded=rec, params={"descriptors": ()})
        assert r["defined"] and r["identifiability"] == {
            "shared_inputs": shared, "observation": "direct",
            "hub_privileged": "not_tested"}
        b = r["details"]["budget"]
        rank = DI.system_basis(s, did, recorded=rec).rank() if did != "none" else 0
        assert b["rank_u"] == rank and b["n_par"] == 1 + rank + 2 * 5
        out[did] = r
    details_r = out["R"]["details"]
    assert details_r["basis"]["n_columns"] > details_r["budget"]["rank_u"]
    assert out["none"]["details"]["basis"]["n_columns"] == 0
    assert out["R"]["details"]["declaration"] == "R"
    assert out["R"]["details"]["hub_name"] == "W"
    assert list(out["R"]["details"]["blocks"]) == ["S", "C", "M", "V"]


# --------------------------------------------------------------------------
# null: shift, then project
# --------------------------------------------------------------------------
def test_the_null_shifts_the_hub_against_periphery_and_inputs_then_projects():
    ts, u, hub, blocks = varx(9, n_time=1500, gain=1.5, loop=0.2)
    T = ts.shape[1]
    basis = DI.input_basis(declared_channels({"u": u}, T), tau_c=TAU_C)
    span = basis.orthonormal_span(start=2)
    reps = {"W": N.block_representation(ts[hub])}
    reps.update({k: N.block_representation(ts[v]) for k, v in blocks.items()})
    design = N.GramDesign(reps, (1, 2), span)
    shift = 400
    G_s = design.shifted("W", shift)
    rolled = dict(reps, W=np.roll(reps["W"], shift, axis=1))
    fresh = N.GramDesign(rolled, (1, 2), span)
    assert np.allclose(G_s, fresh.G, rtol=1e-12, atol=1e-9)
    # U and the periphery are not moved: their Gram block is unchanged
    keep = np.r_[design.idx["one"], design.idx["U"]].tolist()
    for k in blocks:
        keep += np.r_[design.idx[("past", k)], design.idx[("cur", k)]].tolist()
    assert np.array_equal(G_s[np.ix_(keep, keep)], design.G[np.ix_(keep, keep)])
    per = list(blocks)
    a = N.nas_statistics(design, "W", per, G=G_s)
    b = N.nas_statistics(fresh, "W", per)
    assert np.allclose(a["receive"] + a["return"], b["receive"] + b["return"],
                       rtol=1e-9, atol=1e-12)
    # the projection on U is refitted on the shifted data: removing U from
    # the design changes the null statistic
    bare = N.GramDesign(reps, (1, 2))
    c = N.nas_statistics(bare, "W", per, G=bare.shifted("W", shift))
    assert not np.allclose(a["receive"], c["receive"])
    # the shifts are the v1 draws: RandomState(seed * 1000 + 17)
    shifts = N.null_shifts(T, 19, N.null_seed_for(9))
    rng = np.random.RandomState(9017)
    lo = int(math.ceil(0.1 * T))
    assert shifts == [int(rng.randint(lo, T - lo + 1)) for _ in range(19)]
    assert all(lo <= s <= T - lo for s in shifts)
    r = run(ts, hub, blocks, inputs=declared_channels({"u": u}, T), seed=9, params=FAST)
    assert r["details"]["null_shifts"] == shifts
    assert r["details"]["null_order"] == "shift_then_project"


def _lagged(y, lags, l_max):
    T = y.shape[1]
    return np.vstack([y[:, l_max - lag:T - lag] for lag in lags]).T


def _ols_logdet(Y, X):
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    res = Y - X @ beta
    return np.linalg.slogdet(res.T @ res / Y.shape[0])[1]


def _direct_statistics(reps, hub, periphery, lags, span, rows=None):
    """receive_j and return_j by direct least squares from their definition
    (Z_j = the other periphery blocks' pasts plus U)."""
    l_max = max(lags)
    n = reps[hub].shape[1] - l_max
    rows = np.ones(n, dtype=bool) if rows is None else rows
    base = [np.ones((n, 1))] + ([span] if span is not None else [])
    past = {k: _lagged(y, lags, l_max) for k, y in reps.items()}
    cur = {k: y[:, l_max:].T for k, y in reps.items()}
    receive, ret = [], []
    for j in periphery:
        z_j = base + [past[k] for k in periphery if k != j]
        x_r = np.hstack(z_j + [past[hub]])
        receive.append(0.5 * (_ols_logdet(cur[hub][rows], x_r[rows])
                              - _ols_logdet(cur[hub][rows],
                                            np.hstack([x_r, past[j]])[rows])))
        x_b = np.hstack(z_j + [past[j]])
        ret.append(0.5 * (_ols_logdet(cur[j][rows], x_b[rows])
                          - _ols_logdet(cur[j][rows],
                                        np.hstack([x_b, past[hub]])[rows])))
    return np.asarray(receive), np.asarray(ret)


@pytest.mark.parametrize("rep", N.REPRESENTATIONS)
def test_the_statistic_its_null_and_its_jackknife_equal_direct_least_squares(rep):
    """The Gram/Schur code against the definition of receive_j and return_j
    fitted by least squares on the multi-block, input-conditioned geometry:
    the observed per-block values, a hub-shifted null draw (shift, then
    project) and a jackknife replicate (rows touching the group dropped)."""
    ts, u, hub, blocks = varx(31, n_time=2500, gain=1.5, loop=0.3, ret=0.2)
    T = ts.shape[1]
    decl = declared_channels({"u": u}, T)
    r = run(ts, hub, blocks, inputs=decl, seed=31,
            params=dict(FAST, block_representation=rep))
    assert r["defined"], r["reason"]
    d = r["details"]["budget"]["d_max"]
    span = DI.input_basis(decl, tau_c=TAU_C).orthonormal_span(start=2)
    reps = {"W": N.block_representation(ts[hub], rep, d)}
    reps.update({k: N.block_representation(ts[v], rep, d) for k, v in blocks.items()})
    per = list(blocks)
    rec, ret = _direct_statistics(reps, "W", per, (1, 2), span)
    pb = r["details"]["per_block"]
    assert np.allclose(rec, [pb[k]["receive"]["observed"] for k in per],
                       rtol=0, atol=1e-12)
    assert np.allclose(ret, [pb[k]["return"]["observed"] for k in per],
                       rtol=0, atol=1e-12)
    s = r["details"]["null_shifts"][0]
    rec_s, ret_s = _direct_statistics(dict(reps, W=np.roll(reps["W"], s, axis=1)),
                                      "W", per, (1, 2), span)
    assert abs(rec_s.mean() - r["directions"]["receive"]["null_values"][0]) < 1e-12
    assert abs(ret_s.mean() - r["directions"]["return"]["null_values"][0]) < 1e-12
    keep = N.jackknife_keep(N.jackknife_groups(T, 10), 2, 10)[3]
    rec_j, ret_j = _direct_statistics(reps, "W", per, (1, 2), span, rows=keep)
    assert abs(rec_j.mean() - r["directions"]["receive"]["jackknife"][3]) < 1e-12
    assert abs(ret_j.mean() - r["directions"]["return"]["jackknife"][3]) < 1e-12


def test_a_degenerate_null_is_undefined(monkeypatch):
    """Identical null draws make the component UNDEFINED with the v1 reason,
    never ABSENT. Their SD is zero only up to the rounding of the mean (about
    1e-20, a z of about 1e17), so the gate must not read it as a valid null;
    the per-block nulls are marked degenerate and carry no z."""
    ts, _u, hub, blocks = varx(16, n_time=1500, loop=0.3)
    one = N.null_shifts(ts.shape[1], 1, N.null_seed_for(16))
    monkeypatch.setattr(N, "null_shifts", lambda n_time, n_null, *a, **k: one * n_null)
    r = run(ts, hub, blocks, seed=16, params=FAST)
    degenerate = "NOT_DEFINED:transfer_null_degenerate"
    assert (r["defined"], r["reason"]) == (False, degenerate)
    assert R.status_for_reason(r["reason"]) == R.UNDEFINED
    assert not r["significant"]
    assert all(math.isnan(r["directions"][d]["estimate"]) for d in N.DIRECTIONS)
    pb = r["details"]["per_block"]
    assert set(pb) == set(blocks) and r["details"]["coverage"] == 0.0
    for k in pb:
        for d in N.DIRECTIONS:
            assert pb[k][d]["null_degenerate"] and math.isnan(pb[k][d]["z"])
    assert N.component_fields(r)["estimate"] is None
    assert N.null_degenerate([1.0, 1.0], 0.0) and N.null_degenerate([1.0], 0.5)
    assert N.null_degenerate([0.1, 0.2], float("nan"))
    assert not N.null_degenerate([1e-4, 3e-4], float(np.std([1e-4, 3e-4], ddof=1)))


def test_malformed_node_and_basis_declarations_are_refused():
    ts, u, hub, blocks = varx(17, n_time=600, loop=0.3)
    mask = np.zeros(ts.shape[0], dtype=bool)
    mask[hub] = True
    with pytest.raises(N.NASError, match="boolean"):
        run(ts, mask, blocks, params=FAST)
    with pytest.raises(N.NASError, match="boolean"):
        run(ts, [True, False], blocks, params=FAST)
    with pytest.raises(N.NASError, match="integer"):
        run(ts, [0.5, 1.0], blocks, params=FAST)
    with pytest.raises(N.NASError, match="distinct"):
        run(ts, hub, {1: blocks["P1"], "1": blocks["P2"]}, params=FAST)
    # a prebuilt basis must carry the lag plan's filter time constants
    decl = declared_channels({"u": u}, ts.shape[1])
    other = DI.input_basis(decl, lags=(0, 1, 2), taus=(0.08, 0.24, 0.8))
    with pytest.raises(N.NASError, match="filter time constants"):
        run(ts, hub, blocks, inputs=other, params=FAST)
    # ... which a lag time scale of 0.08 s (same lags at 20 Hz) accepts
    p = dict(FAST, lag_timescale_sec=0.08)
    assert run(ts, hub, blocks, inputs=other, params=p)["defined"]


def test_the_gram_solve_equals_least_squares_and_handles_collinear_columns():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(500), rng.standard_normal((500, 4))])
    X = np.column_stack([X, X[:, 1] + 2 * X[:, 2]])  # an exactly collinear column
    y = rng.standard_normal((500, 2)) + X[:, 1:3] @ np.array([[1.0, 0.5], [0.2, -1.0]])
    A = np.column_stack([X, y])
    Gm = A.T @ A
    S, rank = N.residual_covariance(Gm, 500, np.arange(6), np.arange(6, 8))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    assert rank == 5
    assert np.allclose(S, res.T @ res / 500, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------
# jackknife
# --------------------------------------------------------------------------
def test_contiguous_and_interleaved_jackknife_groups():
    T, l_max = 1000, 2
    g = N.jackknife_groups(T, 10, "contiguous")
    edges = np.linspace(0, T, 11).round().astype(int)
    for k in range(10):
        assert np.all(g[edges[k]:edges[k + 1]] == k)
    gi = N.jackknife_groups(T, 10, "interleaved", block_samples=30)
    assert np.array_equal(gi, (np.arange(T) // 30) % 10)
    assert set(np.unique(gi)) == set(range(10))
    for gid in (g, gi):
        keeps = N.jackknife_keep(gid, l_max, 10)
        assert keeps.shape == (10, T - l_max)
        for k in range(10):
            for i, t in enumerate(range(l_max, T)):
                touched = np.any(gid[t - l_max:t + 1] == k)
                assert keeps[k, i] == (not touched)
        dropped = (~keeps).sum(axis=0)
        assert dropped.min() >= 1 and dropped.max() <= 2
    with pytest.raises(N.NASError):
        N.jackknife_groups(T, 1)
    with pytest.raises(N.NASError):
        N.jackknife_groups(T, 10, "interleaved")
    reps = [1.0, 2.0, 4.0]
    assert N.jackknife_se(reps) == pytest.approx(
        math.sqrt(2 / 3 * sum((x - 7 / 3) ** 2 for x in reps)))
    assert math.isnan(N.jackknife_se([1.0, float("nan")]))


def test_jackknife_masks_cover_the_whole_lag_window_at_250_hz():
    """With the 250 Hz lag set (l_max = 25) a replicate drops every row whose
    window [t - 25, t] touches the deleted group, i.e. the group's rows plus
    the 25 rows after each of its blocks."""
    T, l_max, b = 6000, 25, 1250
    gid = N.jackknife_groups(T, 4, "interleaved", block_samples=b)
    keeps = N.jackknife_keep(gid, l_max, 4)
    rows = np.arange(l_max, T)
    for g in range(4):
        expect = np.array([not np.any(gid[t - l_max:t + 1] == g) for t in rows])
        assert np.array_equal(keeps[g], expect)
        n_blocks = int(np.sum((np.arange(0, T, b) // b) % 4 == g))
        dropped = int((~keeps[g]).sum())
        assert np.sum(gid == g) <= dropped <= np.sum(gid == g) + n_blocks * l_max


@pytest.mark.parametrize("method", N.SE_METHODS)
def test_every_se_method_feeds_the_status_rule(method):
    ts, _u, hub, blocks = varx(10, n_time=4000, loop=0.3)
    r = run(ts, hub, blocks, params=dict(FAST, se_method=method))
    groups = int(method.rsplit("_", 1)[1])
    scheme = method.split("_")[1]
    jk = r["details"]["jackknife"]
    assert (jk["scheme"], jk["groups"]) == (scheme, groups)
    assert jk["block_samples"] == (100 if scheme == "interleaved" else None)
    for d in N.DIRECTIONS:
        dd = r["directions"][d]
        assert dd["se_method"] == method and dd["se_df"] == groups - 1
        assert len(dd["jackknife"]) == groups and dd["se"] > 0
        assert dd["se"] == pytest.approx(N.jackknife_se(dd["jackknife"]))
    proto = E.ProtocolV3(
        reference={"kind": "external", "scale": "excess",
                   "values": {"NAS:receive": 0.01, "NAS:return": 0.01}},
        directions={"NAS": list(N.DIRECTIONS)}, se_methods={"NAS": list(N.SE_METHODS)})
    for ev in N.evidence_items(r):
        a = E.assess_item(ev, proto)
        assert a.reason != R.INVALID_SE and a.se_method == method


def test_an_interleaved_scheme_without_enough_blocks_has_no_se():
    ts, _u, hub, blocks = varx(10, n_time=1500, loop=0.3)
    r = run(ts, hub, blocks, params=dict(FAST, se_method="jackknife_interleaved_10"))
    # 1500 samples are 15 blocks of 5 s: every group is populated
    assert math.isfinite(r["directions"]["receive"]["se"])
    r = run(ts, hub, blocks, params=dict(FAST, se_method="jackknife_interleaved_20"))
    assert r["details"]["jackknife"]["reason"] == "jackknife_groups_empty"
    assert math.isnan(r["directions"]["receive"]["se"])
    assert r["directions"]["receive"]["se_df"] == 19.0
    ev = N.evidence_items(r)[0]
    a = E.assess_item(ev, E.ProtocolV3(
        reference={"kind": "external", "scale": "excess",
                   "values": {"NAS:receive": 0.01, "NAS:return": 0.01}},
        directions={"NAS": list(N.DIRECTIONS)}))
    assert (a.status.value, a.reason) == (R.UNDEFINED, R.NO_SAMPLING_SE)


# --------------------------------------------------------------------------
# block representation, rank caps and INSUFFICIENT_TIMEPOINTS
# --------------------------------------------------------------------------
def mixed_blocks(seed, n_sources=6, n_blocks=4, units=8, n_time=3000):
    """Every node a mixture of ``n_sources`` AR(1) sources (rank n_sources)."""
    rng = np.random.default_rng(seed)
    src = np.zeros((n_sources, n_time))
    w = rng.standard_normal((n_sources, n_time))
    for t in range(1, n_time):
        src[:, t] = 0.6 * src[:, t - 1] + w[:, t]
    ts = rng.standard_normal((n_blocks * units, n_sources)) @ src
    names = ["W"] + [f"P{k}" for k in range(1, n_blocks)]
    blocks = {nm: list(range(k * units, (k + 1) * units)) for k, nm in enumerate(names)}
    return ts, blocks.pop("W"), blocks


def test_secondary_estimator_rank_caps():
    ts, hub, blocks = mixed_blocks(1)
    r = run(ts, hub, blocks, params=dict(FAST, block_representation="all_units"))
    b = r["details"]["budget"]
    assert b["observation_rank"] == 6
    tried = b["tried"]
    assert [t["d_max"] for t in tried] == [8, 4, 2, 1]
    assert [t["rank_ok"] for t in tried] == [False, False, False, True]
    assert b["d_max"] == 1 and b["total_dims"] == 4
    assert r["defined"], r["reason"]
    rng = np.random.default_rng(2)
    # full-rank blocks larger than 8 nodes: 8 leading components each, and
    # 4 x 8 = 0.8 x rank 40 passes at the first step
    big = rng.standard_normal((40, 3000))
    hub2 = list(range(10))
    blocks2 = {f"P{k}": list(range(10 * k, 10 * k + 10)) for k in (1, 2, 3)}
    r2 = run(big, hub2, blocks2, params=dict(FAST, block_representation="all_units"))
    b2 = r2["details"]["budget"]
    assert (b2["observation_rank"], b2["d_max"]) == (40, 8)
    assert b2["block_dims"] == {"__hub__": 8, "P1": 8, "P2": 8, "P3": 8}
    # every node of a direct full-rank observation exceeds 0.8 x its rank:
    # the rule lowers the per-block dimension (5 nodes per block -> 4 PCs)
    full = rng.standard_normal((20, 3000))
    hub3 = list(range(5))
    blocks3 = {f"P{k}": list(range(5 * k, 5 * k + 5)) for k in (1, 2, 3)}
    r3 = run(full, hub3, blocks3, params=dict(FAST, block_representation="all_units"))
    b3 = r3["details"]["budget"]
    assert [t["rank_ok"] for t in b3["tried"]] == [False, True]
    assert b3["block_dims"] == {"__hub__": 4, "P1": 4, "P2": 4, "P3": 4}
    # a block larger than the cap is reduced to its leading components
    y = N.block_representation(full[:12], "all_units", 4)
    assert y.shape == (4, 3000)
    assert np.allclose(y, mm._block_components(full[:12], 4))
    m = N.block_representation(full[:5], "block_mean")
    z, _ = N.zscore_rows(full[:5])
    assert np.allclose(m, z.mean(axis=0, keepdims=True))


def test_insufficient_timepoints():
    ts, _u, hub, blocks = varx(12, n_time=80, loop=0.3)
    for rep in N.REPRESENTATIONS:
        r = run(ts, hub, blocks, params=dict(FAST, block_representation=rep))
        assert (r["defined"], r["reason"]) == (False, R.INSUFFICIENT_TIMEPOINTS), rep
        assert R.status_for_reason(r["reason"]) == R.UNDEFINED
        tried = r["details"]["budget"]["tried"]
        assert tried and not any(t["timepoints_ok"] for t in tried)
    # 1 + 2 x 4 parameters need T - 2 >= 90
    assert run(*system(12, n_time=92, loop=0.3), params=FAST)["defined"]


def test_shape_gates_are_undefined_not_absent():
    ts, _u, hub, blocks = varx(13, n_time=600, loop=0.3)
    bad = ts.copy()
    bad[0, 5] = np.nan
    cases = [
        (run(bad, hub, blocks, params=FAST), "NOT_DEFINED:non_finite_timeseries"),
        (run(ts, [], blocks, params=FAST), "NOT_DEFINED:hub_or_periphery_empty"),
        (run(ts, hub, {}, params=FAST), "NOT_DEFINED:hub_or_periphery_empty"),
        (run(ts, hub, {"P1": [0, 1]}, params=FAST), "NOT_DEFINED:invalid_workspace"),
        (run(ts, [99], blocks, params=FAST), "NOT_DEFINED:invalid_workspace"),
    ]
    flat = ts.copy()
    flat[hub] = 1.0
    cases.append((run(flat, hub, blocks, params=FAST), "NOT_DEFINED:no_variance"))
    for r, reason in cases:
        assert (r["defined"], r["reason"]) == (False, reason)
        assert R.status_for_reason(r["reason"]) == R.UNDEFINED


# --------------------------------------------------------------------------
# observation gate and identifiability record
# --------------------------------------------------------------------------
def test_observation_mixed_not_admitted_gate():
    ts, _u, hub, blocks = varx(14, n_time=1500, loop=0.3)
    proto = E.ProtocolV3(reference={"kind": "external", "scale": "excess",
                                    "values": {"NAS:receive": 0.01,
                                               "NAS:return": 0.01}},
                         directions={"NAS": list(N.DIRECTIONS)})
    for obs in N.OBSERVATIONS_MIXED:
        r = run(ts, hub, blocks, params=FAST, observation=obs)
        assert (r["defined"], r["reason"]) == (False, R.OBSERVATION_MIXED_NOT_ADMITTED)
        assert r["details"]["observation_gate"] == "not_admitted"
        assert r["identifiability"]["observation"] == obs
        pa = E.assess_principle("NAS", N.evidence_items(r), proto)
        assert pa.status.value == R.UNDEFINED
        admitted = run(ts, hub, blocks, params=FAST, observation=obs,
                       observation_admitted=True)
        forced = run(ts, hub, blocks, params=FAST, observation=obs,
                     override_observation_gate=True)
        assert admitted["defined"]
        assert admitted["details"]["observation_gate"] == "admitted"
        assert forced["defined"]
        assert forced["details"]["observation_gate"] == "overridden"
        assert forced["directions"] == admitted["directions"]
    direct = run(ts, hub, blocks, params=FAST)
    assert direct["details"]["observation_gate"] == "not_applicable"
    assert direct["directions"] == admitted["directions"]
    # the resolvability gate comes first
    r = run(*system(14, n_time=300, dt=2.0), dt=2.0, observation="sensor_mixing")
    assert r["reason"] == R.SAMPLING_UNRESOLVED
    with pytest.raises(N.NASError):
        run(ts, hub, blocks, params=FAST, observation="eeg")


def test_forward_views_carry_their_observation():
    w = WB.simulate_whole_brain(WB.WholeBrainConfig(duration_sec=12.0), seed=3)
    assert N.observation_of(w.meta) == "direct"
    e = FWD.eeg_forward(w, seed=3)
    assert N.observation_of(e.meta) == "sensor_mixing"
    r = N.nas_v3_system(e, "none", params={"descriptors": ()})
    assert (r["defined"], r["reason"]) == (False, R.OBSERVATION_MIXED_NOT_ADMITTED)
    assert r["identifiability"] == {"shared_inputs": "none",
                                    "observation": "sensor_mixing",
                                    "hub_privileged": "not_tested"}
    src = N.nas_v3_system(w, "none", params={"descriptors": ()})
    assert src["defined"] and src["details"]["lags"] == [1, 3, 9, 25]
    assert N.observation_of({"observation": "source_estimate"}) == "source_estimate"
    assert [N.observation_stage(o) for o in REC.OBSERVATIONS] == [
        "source", "source_estimate", "sensor", "bold"]


def test_whole_brain_blocks_merge_small_groups_within_a_hemisphere():
    conn = WB.load_connectome()
    meta = {"workspace_nodes": conn.hubs(6), "modules": conn.modules(),
            "module_order": sorted(conn.modules())}
    hub_name, hub, blocks = N.declared_blocks(meta)
    assert hub_name == "hub" and hub == conn.hubs(6)
    assert all(len(v) >= N.MIN_BLOCK_SIZE for v in blocks.values())
    assert not any(k.endswith("insula") or k.startswith("M_") for k in blocks)
    covered = sorted(i for v in blocks.values() for i in v)
    assert covered == sorted(set(range(conn.n)) - set(hub))
    lab = conn.labels
    for k, v in blocks.items():
        hemis = {conn.hemisphere[i] for i in v} - {"M"}
        assert hemis == {k.split("_")[0]}, (k, [lab[i] for i in v])


def test_identifiability_record_and_component_record(family_a):
    r = N.nas_v3_system(family_a, "R", params={"descriptors": ("bic_order",)})
    ident = r["identifiability"]
    assert ident == REC.validate_identifiability(ident)
    assert ident == {"shared_inputs": "complete", "observation": "direct",
                     "hub_privileged": "not_tested"}
    fields = N.component_fields(r)
    assert fields["identifiability"] == ident
    assert fields["se_method"] == N.SE_METHOD_DEFAULT and fields["se_df"] == 9.0
    assert fields["details"]["directions"]["receive"]["statistic"] == "te_in"
    proto = E.ProtocolV3(reference={"kind": "external", "scale": "excess",
                                    "values": {"NAS:receive": 0.002,
                                               "NAS:return": 0.02}},
                         directions={"NAS": list(N.DIRECTIONS)})
    pa = E.assess_principle("NAS", N.evidence_items(r), proto)
    status = pa.record_fields()
    comp = REC.ComponentRecord(
        principle="NAS", estimator_version=N.ESTIMATOR_VERSION, declaration_id="R",
        observation_stage=N.observation_stage(ident["observation"]),
        protocol_id="mpc-bench-v2-test", protocol_hash="0" * 64, **fields,
        **{k: v for k, v in status.items() if k != "flags"},
        flags=tuple(status["flags"]))
    again = REC.ComponentRecord.from_dict(json.loads(json.dumps(comp.to_dict())))
    assert again.identifiability == ident and again.c_R == status["c_R"]
    assert again.details["directions"]["return"]["se_method"] == N.SE_METHOD_DEFAULT
