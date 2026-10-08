# -*- coding: utf-8 -*-
"""IIM v5 (iim-v5-2026.10): the rank p-value and the rank gate, the
occupancy gate, exact directional cuts on family-B networks, the stratified
pair rotation and the shift-then-project order, the circular block
bootstrap, the rank condition on montages and ZCA, the observation gate,
the exact path and its SE method, and the untouched v1 kernel path.
Fixtures are family-B networks, synthetic series and development seeds."""
import math
import os

import numpy as np
import pytest

from impact_pipeline import evidence_v2 as EV
from impact_pipeline import iim_xp
from impact_pipeline import mpc_metrics as mm
from impact_pipeline.bench import forward
from impact_pipeline.bench.generators import (
    family_b_network,
    make_system,
    sample_binary_trajectory,
)
from impact_pipeline.v2 import ESTIMATOR_VERSIONS_V2
from impact_pipeline.v2 import declared_inputs as DI
from impact_pipeline.v2 import iim_v5 as IIM
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import records as REC

NO_SE = {"n_null": 19, "se_method": None}


def binary_run(kind, n=3, T=2000, seed=7, **kw):
    net = family_b_network(kind, n=n, **kw)
    return net, sample_binary_trajectory(net, T, seed=seed).T.astype(float)


def protocol(cut="directional", se_methods=("circular_block_bootstrap_10pct_B50",
                                            "exact", "jackknife_contiguous_10"),
             null_family="circular_shift", anchor=0.03):
    payload = {
        "schema": EV.PROTOCOL_SCHEMA_V3, "name": f"test-iim-{cut}",
        "necessity_set": ["IIM"], "channels": {"IIM": ["default"]},
        "cutoffs": {"IIM": [0.25, 0.10]}, "alpha": 0.05,
        "null_families": {"IIM": null_family},
        "reference": {"kind": "external", "scale": "excess",
                      "values": {"IIM": anchor}, "se": {"IIM": 0.0}},
        "status_rule": EV.DEFAULT_STATUS_RULE.to_dict(),
        "rank_gates": {"IIM": 0.05},
        "estimator_versions": {"IIM": IIM.ESTIMATOR_VERSION},
        "se_methods": {"IIM": list(se_methods)} if se_methods else {},
    }
    return EV.ProtocolV3.from_dict(payload)


def fake_result(m, nu, se, p_ind, cut="directional", n_null=19, null_sd=1e-4,
                se_method="circular_block_bootstrap_10pct_B50", se_df=12.0):
    """A defined result in the compute_iim_v5 format (for status tests)."""
    view = {"cut_mode": cut, "estimate": m, "value": m - nu, "null_mean": nu,
            "null_sd": null_sd, "n_null": n_null, "null_family": "circular_shift",
            "null_values": [nu] * n_null, "p_ind": p_ind, "se": se, "se_df": se_df,
            "se_method": se_method, "estimator_id": IIM.estimator_id(cut)}
    return {"principle": "IIM", "estimator_version": IIM.ESTIMATOR_VERSION,
            "cut_mode": cut, "defined": True, "reason": None, "exact": False,
            "null_family": "circular_shift", "cuts": {cut: view}, "details": {},
            **{k: view[k] for k in ("estimate", "value", "null_mean", "null_sd",
                                    "n_null", "p_ind", "se", "se_df", "se_method")}}


# --------------------------------------------------------------------------
# identity with v1 and the kernel path
# --------------------------------------------------------------------------
def test_version_and_estimator_ids():
    assert IIM.ESTIMATOR_VERSION in ESTIMATOR_VERSIONS_V2["IIM"]
    assert IIM.estimator_id("directional") == "compute_IIM:directional@iim-v5-2026.10"
    assert IIM.PRIMARY_CUT_MODE == "directional"
    assert IIM.IIMParams().cut_modes == ("directional", "bidirectional")
    assert len(IIM.system_cuts(4, "directional")) == 14
    assert len(IIM.system_cuts(4, "bidirectional")) == 7
    assert IIM.system_cuts(4, "bidirectional") == mm._iim_all_system_cuts(4, "all")
    assert IIM.system_cuts(4, "directional") == mm._iim_all_system_cuts(
        4, "all", "directional")


def test_v5_equals_v1_on_the_v1_geometry():
    """Without strata, basis or pre-processing the v5 estimate and every null
    draw equal v1 compute_IIM (bins 2, node shrinkage) for both cut modes."""
    _, ts = binary_run("ring", coupling=0.45)
    r = IIM.compute_iim_v5(ts, lag=1, params=NO_SE, null_seed=17)
    assert r["defined"] and r["null_family"] == "circular_shift"
    for cut in IIM.CUT_MODES:
        for kernel in ("xp", "numba"):
            d = mm.compute_IIM(ts, bins=2, lag_trs=1, return_details=True,
                               psi_kernel=kernel, cut_mode=cut,
                               progress_log_every_cuts=10**9)
            assert abs(d["Delta_Psi"] - r["cuts"][cut]["estimate"]) < 1e-12
        d = mm.compute_IIM(ts, bins=2, lag_trs=1, return_details=True, psi_kernel="xp",
                           cut_mode=cut, null_surrogates=19, null_seed=17,
                           progress_log_every_cuts=10**9)
        np.testing.assert_allclose(r["cuts"][cut]["null_values"], d["Delta_Psi_null"],
                                   rtol=0, atol=1e-12)
        assert r["cuts"][cut]["null_mean"] == pytest.approx(d["Delta_Psi_null_mean"],
                                                            abs=1e-12)
    sur = IIM.circular_shift_surrogates(ts, 19, 17, IIM.null_min_shift(ts.shape[1], 1))
    ref = mm.iim_null_surrogate_series(ts, 19, "circular_shift", 17,
                                       mm.iim_null_min_shift(None, 1, ts.shape[1]))
    for a, b in zip(sur, ref):
        np.testing.assert_array_equal(a, b)


def test_v1_iim_kernel_path_untouched(monkeypatch):
    """v5 evaluates Psi with the xp kernel only; it never touches the v1 host
    (numba) kernel or the kernel selection of v1 compute_IIM."""
    env_before = os.environ.get(mm.IIM_PSI_KERNEL_ENV)
    kernel_before = mm.resolve_iim_psi_kernel("auto")
    calls = {"xp": 0}
    real = iim_xp.psi_contribution

    def spy(*a, **k):
        calls["xp"] += 1
        return real(*a, **k)

    def forbidden(*a, **k):
        raise AssertionError("v5 must not use the v1 host Psi kernel")

    monkeypatch.setattr(iim_xp, "psi_contribution", spy)
    monkeypatch.setattr(mm, "_iim_phase1_chunk_contribution", forbidden)
    _, ts = binary_run("ring", coupling=0.45, T=1500)
    r = IIM.compute_iim_v5(ts, lag=1, params=NO_SE, null_seed=3)
    assert r["defined"] and calls["xp"] > 0
    assert r["details"]["psi_kernel"] == "xp"
    IIM.exact_delta_psi(family_b_network("ring", n=3, coupling=0.45).tpm, "directional")
    assert os.environ.get(mm.IIM_PSI_KERNEL_ENV) == env_before
    assert mm.resolve_iim_psi_kernel("auto") == kernel_before
    monkeypatch.undo()
    # the v1 path still resolves and runs its own kernel
    d = mm.compute_IIM(ts, bins=2, lag_trs=1, return_details=True,
                       progress_log_every_cuts=10**9)
    assert d["psi_kernel"] == kernel_before
    assert abs(d["Delta_Psi"] - r["cuts"]["bidirectional"]["estimate"]) < 1e-12


# --------------------------------------------------------------------------
# rank p-value and rank gate
# --------------------------------------------------------------------------
def test_rank_p_value_formula():
    assert IIM.rank_p_value(1.0, [0.0] * 19) == pytest.approx(1 / 20)
    assert IIM.rank_p_value(1.0, [1.0] + [0.0] * 18) == pytest.approx(2 / 20)  # ties count
    assert IIM.rank_p_value(-1.0, np.zeros(19)) == pytest.approx(1.0)
    assert IIM.rank_p_value(0.5, [np.nan, 0.0, 1.0]) == pytest.approx(2 / 3)
    assert math.isnan(IIM.rank_p_value(np.nan, [0.0]))
    assert math.isnan(IIM.rank_p_value(0.0, [np.nan]))


def test_rank_p_value_exact_on_exchangeable_data():
    """With the estimate exchangeable with K = 19 draws, p_ind is uniform on
    {1/20, ..., 1}: P(p_ind <= 0.05) = 1/20 exactly (continuous draws) and at
    most 1/20 with ties."""
    rng = np.random.default_rng(20)
    n_rep, k = 8000, 19
    x = rng.normal(size=(n_rep, k + 1))
    p = np.array([IIM.rank_p_value(row[0], row[1:]) for row in x])
    grid = np.arange(1, k + 2) / (k + 1)
    assert np.all(np.isin(np.round(p, 12), np.round(grid, 12)))
    counts = np.array([np.sum(np.isclose(p, g)) for g in grid])
    se = math.sqrt(n_rep * (1 / 20) * (19 / 20))
    assert np.all(np.abs(counts - n_rep / 20) < 4.5 * se)
    rate = np.mean(p <= 0.05)
    assert abs(rate - 0.05) < 4.5 * math.sqrt(0.05 * 0.95 / n_rep)
    ties = rng.integers(0, 3, size=(n_rep, k + 1)).astype(float)
    pt = np.array([IIM.rank_p_value(row[0], row[1:]) for row in ties])
    assert np.mean(pt <= 0.05) <= 0.05 + 4.5 * math.sqrt(0.05 * 0.95 / n_rep)
    # on the estimator: an independent-unit run gives p_ind on the grid of K
    _, ts = binary_run("independent", T=1500, seed=11)
    r = IIM.compute_iim_v5(ts, lag=1, params=NO_SE, null_seed=5)
    for cut in IIM.CUT_MODES:
        assert any(np.isclose(r["cuts"][cut]["p_ind"], g) for g in grid)


def test_failed_rank_gate_is_undefined_never_absent():
    proto = protocol()
    # c = (0.033 - 0.003) / 0.03 = 1.0 with a small SE: PRESENT unless the gate fails
    present = EV.assess_item(IIM.evidence(fake_result(0.033, 0.003, 0.001, 0.05)), proto)
    assert present.status is EV.PRESENT
    gated = EV.assess_item(IIM.evidence(fake_result(0.033, 0.003, 0.001, 0.30)), proto)
    assert gated.status is EV.UNDEFINED
    assert gated.reason == "INCONCLUSIVE:NULL_NOT_EXCEEDED"
    assert R.status_for_reason(gated.reason) == R.UNDEFINED
    # the gate only removes PRESENT: a credibly negligible c is ABSENT whatever p_ind
    absent = EV.assess_item(IIM.evidence(fake_result(0.0031, 0.003, 0.0002, 1.0)), proto)
    assert absent.status is EV.ABSENT
    # no p_ind at all: PRESENT is not reachable
    nop = fake_result(0.033, 0.003, 0.001, None)
    assert EV.assess_item(IIM.evidence(nop), proto).status is EV.UNDEFINED


# --------------------------------------------------------------------------
# occupancy gate
# --------------------------------------------------------------------------
def test_occupancy_function_counts_rows_per_stratum():
    curr = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 30, dtype=np.int16)
    occ = IIM.occupancy(curr, None, n_min=25)
    assert occ["passes"] and occ["rarest_row"] == 30
    assert occ["strata"][0]["counts"] == [30, 30, 30, 30]
    strata = np.r_[np.zeros(100, int), np.ones(20, int)]
    occ = IIM.occupancy(curr, strata, n_min=25)
    assert not occ["passes"]
    assert [s["passes"] for s in occ["strata"]] == [True, False]
    occ = IIM.occupancy(curr[curr[:, 0] == 0], None, n_min=25)
    assert not occ["passes"] and occ["n_unvisited"] == 2


def test_occupancy_diagnostics_and_shrinkage_intensity():
    """The shrinkage intensities reproduce the v1 node-shrinkage TPM, and the
    occupancy diagnostics are recorded with every run."""
    _, ts = binary_run("ring", coupling=0.45, T=1500, seed=9)
    curr, nxt = IIM.transition_pairs(IIM.binarise(ts), 1)
    lam = IIM.shrinkage_intensity(curr, nxt)
    assert lam.shape == (8, 3) and np.all((lam >= 0) & (lam <= 1))
    states = iim_xp.state_table(3, 2).astype(int)
    keys = iim_xp.subset_keys(curr, (0, 1, 2), 2)
    n_obs = np.bincount(keys, minlength=8).astype(float)
    tpm = np.ones((8, 8))
    for i in range(3):
        counts = np.zeros((8, 2))
        np.add.at(counts, (keys, nxt[:, i].astype(int)), 1.0)
        own = np.zeros((2, 2))
        np.add.at(own, (curr[:, i].astype(int), nxt[:, i].astype(int)), 1.0)
        own = (own + 1e-3) / (own.sum(axis=1, keepdims=True) + 2e-3)
        target = own[states[:, i]]
        theta = np.where(n_obs[:, None] > 0, counts / np.maximum(n_obs[:, None], 1), target)
        p_i = lam[:, [i]] * target + (1 - lam[:, [i]]) * theta
        p_i /= p_i.sum(axis=1, keepdims=True)
        tpm *= p_i[:, states[:, i]]
    tpm /= tpm.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(tpm, IIM.node_shrinkage_tpm(curr, nxt), atol=1e-12)
    r = IIM.compute_iim_v5(ts, lag=1, params={"n_null": 0, "se_method": None})
    row = r["details"]["occupancy"]["strata"][0]
    assert row["mean_shrinkage"] == pytest.approx(float(lam.mean()))
    assert 0 <= row["split_half_tv"] < 0.2 and row["effective_states"] > 1


def test_family_a_system_with_the_complete_declaration():
    """On a family-A system the common input basis of declaration R is
    rank-deficient by construction; the residualisation counts its effective
    rank and the grain, lag and stage come from the system's meta."""
    from impact_pipeline.bench.generators import simulate_family_a

    system = simulate_family_a(None, None, 940)
    basis = DI.system_basis(system, "R", estimator="IIM")
    r = IIM.compute_iim_v5_system(system, basis=basis,
                                  params={"n_null": 0, "se_method": None})
    assert r["reason"] == R.NO_NULL_CALIBRATION  # every gate before the null passed
    d = r["details"]
    assert d["lag"] == 2 and d["n_macro"] == 4 and d["observation_stage"] == "source"
    assert d["basis"]["declaration"] == "R" and d["basis"]["shared_inputs"] == "complete"
    assert d["basis"]["rank"] == basis.rank() < d["basis"]["n_columns"]
    assert d["n_analysed"] == system.ts.shape[1] - basis.first_complete_sample
    assert d["occupancy"]["passes"]


def test_occupancy_gate_insufficient_occupancy():
    """Trapped all-to-all dynamics (coupling 0.6, T = 1000), a constant macro
    node and a stratum without some macro state are UNDEFINED, never ABSENT;
    the gate fires before any null is computed."""
    net = family_b_network("all_to_all", n=4, coupling=0.6)
    pi_min = float(net.stationary.min())
    assert 1000 * pi_min <= 12.5
    ts = sample_binary_trajectory(net, 1000, seed=430).T.astype(float)
    r = IIM.compute_iim_v5(ts, lag=1, params={"n_null": 19}, null_seed=1)
    assert r["reason"] == R.INSUFFICIENT_OCCUPANCY and not r["defined"]
    assert "null_values" not in r["details"]
    assert r["details"]["occupancy"]["passes"] is False
    _, ts = binary_run("ring", coupling=0.45)
    ts[1] = 1.0
    r = IIM.compute_iim_v5(ts, lag=1, params=NO_SE)
    assert r["reason"] == R.INSUFFICIENT_OCCUPANCY
    assert r["details"]["occupancy"]["constant_nodes"] == [1]
    # a stratum in which node 0 never takes the value 1
    _, ts = binary_run("independent", T=3000, seed=3)
    strata = (ts[0] > 0).astype(int)
    r = IIM.compute_iim_v5(ts, lag=1, params=NO_SE, strata=strata)
    assert r["reason"] == R.INSUFFICIENT_OCCUPANCY
    ev = IIM.evidence(r)
    a = EV.assess_item(ev, protocol(null_family="stratified_pair_rotation"))
    assert a.status is EV.UNDEFINED and a.reason == R.INSUFFICIENT_OCCUPANCY
    with pytest.raises(ValueError):
        IIM.IIMParams(n_min=10)  # N_min may only be raised (25, 50, 100)
    assert IIM.IIMParams(n_min=50).n_min == 50


# --------------------------------------------------------------------------
# exact path
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind,kw", [
    ("ring", {"coupling": 0.45}),
    ("all_to_all", {"coupling": 0.4}),
    ("xor_loop", {"noise": 0.2}),
    ("feedforward_star", {"coupling": 0.6}),
])
def test_directional_cuts_equal_the_exact_tpm_values(kind, kw):
    net = family_b_network(kind, n=4, **kw)
    for cut in IIM.CUT_MODES:
        mine = IIM.exact_iim(net, cut)["delta_psi"]
        ref = mm.compute_IIM_from_tpm(net.tpm, cut_mode=cut, psi_kernel="xp", clamp=False)
        assert mine == pytest.approx(ref, abs=1e-12)
    if kind == "feedforward_star":
        assert abs(IIM.exact_iim(net, "directional")["delta_psi"]) < 1e-12
        assert IIM.exact_iim(net, "bidirectional")["delta_psi"] > 0.01


def test_exact_anchors_and_the_hidden_driver_targets():
    ring = family_b_network("ring", n=4, coupling=0.45)
    assert IIM.exact_iim(ring, "bidirectional")["delta_psi"] == pytest.approx(0.04513,
                                                                              abs=5e-6)
    assert IIM.exact_iim(ring, "directional")["delta_psi"] == pytest.approx(0.02909,
                                                                            abs=5e-6)
    hd = family_b_network("hidden_driver", n=4, driver_weight=1.0,
                          driver_persistence=0.9)
    tpms, p_d, w = IIM.hidden_driver_parts(hd)
    assert p_d == pytest.approx([0.5, 0.5]) and np.allclose(w.sum(axis=0), 1.0)
    for t in tpms:
        assert iim_xp.state_by_node_deviation(t, 4, 2) < 1e-12
    for cut in IIM.CUT_MODES:
        cond = IIM.exact_iim(hd, cut, conditioned=True)
        assert abs(cond["delta_psi"]) < 1e-12 and cond["target"] == "conditioned"
    obs = IIM.exact_iim(hd, "bidirectional", observational=True)["delta_psi"]
    assert obs > 0.01
    with pytest.raises(ValueError):  # the observational TPM is not state-by-node
        IIM.exact_iim(hd, "directional", observational=True)


def test_sampled_directional_star_tracks_its_exact_zero():
    net = family_b_network("feedforward_star", n=4, coupling=0.6)
    ts = sample_binary_trajectory(net, 30000, seed=416).T
    curr, nxt = IIM.transition_pairs(IIM.binarise(ts), 1)
    m = IIM.conditioned_delta_psi(curr, nxt, None)["m"]
    assert abs(m["directional"]) < 0.002
    assert m["bidirectional"] > 0.01


def test_recorded_driver_removes_the_hidden_driver():
    net = family_b_network("hidden_driver", n=4, driver_weight=1.0, driver_persistence=0.9)
    traj, drv = sample_binary_trajectory(net, 30000, seed=437, return_driver=True)
    curr, nxt = IIM.transition_pairs(IIM.binarise(traj.T), 1)
    strata = (np.asarray(drv) > 0).astype(int)[: curr.shape[0]]
    cond = IIM.conditioned_delta_psi(curr, nxt, strata)
    hidden = IIM.conditioned_delta_psi(curr, nxt, None)
    for cut in IIM.CUT_MODES:
        assert abs(cond["m"][cut]) < 0.002
        assert hidden["m"][cut] > 0.01
    assert [s["share"] for s in cond["strata"]] == pytest.approx(
        [np.mean(strata == 0), np.mean(strata == 1)])


def test_exact_result_needs_the_exact_se_method():
    """An exact known-TPM value is decided on c alone, provided the protocol
    admits the SE method 'exact'; otherwise it is INVALID_SE."""
    ok = protocol(anchor=0.03)
    for value, want in ((0.03, EV.PRESENT), (0.0001, EV.ABSENT),
                        (0.005, EV.UNDEFINED)):
        a = EV.assess_item(IIM.evidence(IIM.exact_result(value)), ok)
        assert a.status is want and a.route == "exact"
    strict = protocol(se_methods=("circular_block_bootstrap_10pct_B50",))
    a = EV.assess_item(IIM.evidence(IIM.exact_result(0.03)), strict)
    assert a.status is EV.UNDEFINED and a.reason == R.INVALID_SE
    res = IIM.exact_result(0.03)
    assert res["exact"] and res["se_method"] == "exact" and res["se_df"] is None


# --------------------------------------------------------------------------
# conditioning and null orders
# --------------------------------------------------------------------------
def test_stratified_pair_rotation_preserves_strata():
    rng = np.random.default_rng(0)
    n_pairs, n = 600, 3
    curr = rng.integers(0, 2, size=(n_pairs, n)).astype(np.int16)
    nxt = rng.integers(0, 2, size=(n_pairs, n)).astype(np.int16)
    strata = rng.integers(0, 2, size=n_pairs)
    strata_before = strata.copy()
    c2, n2, shifts = IIM.stratified_pair_rotation(curr, nxt, strata,
                                                  np.random.default_rng(5))
    np.testing.assert_array_equal(strata, strata_before)
    np.testing.assert_array_equal(c2[:, 0], curr[:, 0])
    np.testing.assert_array_equal(n2[:, 0], nxt[:, 0])
    for s in (0, 1):
        idx = np.flatnonzero(strata == s)
        lo = math.ceil(0.1 * idx.size)
        assert all(lo <= r <= idx.size - lo for r in shifts[s])
        for i in range(n):
            pairs = sorted(zip(curr[idx, i], nxt[idx, i]))
            pairs2 = sorted(zip(c2[idx, i], n2[idx, i]))
            assert pairs == pairs2  # each node's own transitions per stratum
            j = i
            if i:  # the pair sequence is a rotation within the stratum
                r = shifts[s][i - 1]
                np.testing.assert_array_equal(c2[idx, j], np.roll(curr[idx, j], r))
                np.testing.assert_array_equal(n2[idx, j], np.roll(nxt[idx, j], r))
    again = IIM.stratified_pair_rotation(curr, nxt, strata, np.random.default_rng(5))
    np.testing.assert_array_equal(again[0], c2)
    # a stratum of one pair cannot be rotated: the draw fails
    assert IIM.stratified_pair_rotation(curr[:1], nxt[:1], strata[:1] * 0,
                                        np.random.default_rng(1)) is None


def test_stratified_null_keeps_each_node_and_its_strata():
    _, ts = binary_run("independent", T=3000, seed=21)
    driver = np.random.default_rng(2).integers(0, 2, size=ts.shape[1])
    r = IIM.compute_iim_v5(ts, lag=1, params=NO_SE, strata=driver, null_seed=9)
    assert r["defined"] and r["null_family"] == IIM.NULL_FAMILY_STRATIFIED
    assert r["details"]["conditioning"] == "stratify"
    assert len(r["details"]["strata"]) == 2
    assert len(r["details"]["null_rotations"]) == 19
    with pytest.raises(ValueError):  # more than four joint states: residualise
        IIM.joint_strata(np.arange(10) % 5)
    codes, alphabet = IIM.joint_strata(np.vstack([driver, 1 - driver]))
    assert len(alphabet) == 2 and codes.size == driver.size
    with pytest.raises(ValueError):
        IIM.compute_iim_v5(ts, lag=1, params=NO_SE, strata=driver,
                           basis=object())


def ar1_with_driver(n_nodes=3, T=2400, seed=4, dt=0.05):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 0.9, size=n_nodes)
    x = np.zeros((n_nodes, T))
    e = rng.normal(size=(n_nodes, T))
    for t in range(1, T):
        x[:, t] = a * x[:, t - 1] + e[:, t]
    phi = math.exp(-dt / 0.3)
    d = np.zeros(T)
    for t in range(1, T):
        d[t] = phi * d[t - 1] + math.sqrt(1 - phi * phi) * rng.normal()
    system = type("S", (), {})()
    system.ts, system.events, system.oracle = x + d, None, {}
    system.meta = {"dt": dt, "seed": None}
    rec = DI.record_inputs(system, extra_channels={"driver": d})
    basis = DI.input_basis(DI.declare(rec, "R"), tau_c=0.1, estimator="IIM")
    return x + d, basis


def test_shift_then_project_order_for_residualisation():
    """With an input basis the null shifts the pre-processed series and then
    fits every conditioning regression on the surrogate; the other order is
    computed on request, from the residuals."""
    y, basis = ar1_with_driver()
    assert basis.lags == (0, 1, 2)
    r = IIM.compute_iim_v5(y, lag=2, params=NO_SE, basis=basis, null_seed=31)
    assert r["defined"] and r["details"]["null_order"] == "shift_then_project"
    b = r["details"]["basis"]
    assert b["rank"] < b["n_columns"]  # the basis is rank-deficient by construction
    s0 = basis.first_complete_sample
    span = basis.orthonormal_span(s0)
    y_c = y[:, s0:]
    sur = IIM.circular_shift_surrogates(y_c, 19, 31, IIM.null_min_shift(y_c.shape[1], 2))
    want = [IIM._statistic(IIM.residualise(s, span), None, 2, ("directional",))
            ["directional"] for s in sur]
    np.testing.assert_allclose(r["cuts"]["directional"]["null_values"], want,
                               rtol=0, atol=1e-15)
    # projection after the shift: every surrogate residual is orthogonal to U
    for s in sur[:3]:
        res = IIM.residualise(s, span)
        assert np.max(np.abs(res @ span)) < 1e-8
    # the non-adopted order shifts the residuals (not orthogonal to U)
    resid = IIM.residualise(y_c, span)
    sur_r = IIM.circular_shift_surrogates(resid, 19, 31,
                                          IIM.null_min_shift(y_c.shape[1], 2))
    assert np.max(np.abs(sur_r[0] @ span)) > 1e-3
    r2 = IIM.compute_iim_v5(y, lag=2, params={**NO_SE, "null_order": "project_then_shift"},
                            basis=basis, null_seed=31)
    want2 = [IIM._statistic(s, None, 2, ("directional",))["directional"] for s in sur_r]
    np.testing.assert_allclose(r2["cuts"]["directional"]["null_values"], want2,
                               rtol=0, atol=1e-15)
    assert r2["cuts"]["directional"]["estimate"] == r["cuts"]["directional"]["estimate"]


def test_an_empty_basis_conditions_on_nothing():
    """Declaration ``none`` gives a basis without columns: the run is scored
    unconditioned and uncropped (estimate and null draws equal the run
    without a basis), and the declaration is still recorded."""
    y, _ = ar1_with_driver(T=1500, seed=5)
    system = type("S", (), {})()
    system.ts, system.events, system.oracle = y, None, {}
    system.meta = {"dt": 0.05, "seed": None}
    empty = DI.input_basis(DI.declare(DI.record_inputs(system), "none"), tau_c=0.1,
                           estimator="IIM")
    assert empty.n_columns == 0 and empty.first_complete_sample == 2
    r = IIM.compute_iim_v5(y, lag=2, params=NO_SE, basis=empty, null_seed=12)
    plain = IIM.compute_iim_v5(y, lag=2, params=NO_SE, null_seed=12)
    assert r["defined"] and r["details"]["conditioning"] == "none"
    assert r["details"]["n_analysed"] == y.shape[1]
    assert r["details"]["basis"]["declaration"] == "none"
    assert r["details"]["basis"]["rank"] == 0 and "null_order" not in r["details"]
    for cut in IIM.CUT_MODES:
        assert r["cuts"][cut]["estimate"] == plain["cuts"][cut]["estimate"]
        assert r["cuts"][cut]["null_values"] == plain["cuts"][cut]["null_values"]
    # strata may accompany an empty basis (nothing else is declared)
    driver = np.random.default_rng(1).integers(0, 2, size=y.shape[1])
    both = IIM.compute_iim_v5(y, lag=2, params=NO_SE, basis=empty, strata=driver)
    assert both["details"]["conditioning"] == "stratify"


def test_residualisation_removes_a_shared_driver():
    y, basis = ar1_with_driver(T=6000, seed=8)
    plain = IIM.compute_iim_v5(y, lag=2, params=NO_SE, null_seed=1)
    cond = IIM.compute_iim_v5(y, lag=2, params=NO_SE, basis=basis, null_seed=1)
    for cut in IIM.CUT_MODES:
        assert plain["cuts"][cut]["p_ind"] <= 0.05  # the shared driver reads as dependence
        assert cond["cuts"][cut]["value"] < plain["cuts"][cut]["value"]


# --------------------------------------------------------------------------
# sampling SE
# --------------------------------------------------------------------------
@pytest.mark.parametrize("T", [1000, 3000, 10000, 11800, 12000, 30000])
def test_circular_block_bootstrap_block_count(T):
    assert IIM.n_blocks(T) == 10
    length = IIM.block_length(T)
    assert length == math.ceil(0.1 * T)
    idx = IIM.bootstrap_indices(T, np.random.default_rng(T))
    assert idx.size == T and idx.min() >= 0 and idx.max() < T
    blocks = idx[: 9 * length].reshape(9, length)
    assert np.all((np.diff(blocks, axis=1) % T) == 1)  # contiguous, wrapping


def test_bootstrap_se_df_and_the_se_contract():
    p = IIM.IIMParams()
    assert p.se_method_name == "circular_block_bootstrap_10pct_B50"
    assert p.se_df == IIM.BOOTSTRAP_SE_DF == 9.0  # lowered from 12 at CD-3
    contract = EV.SE_METHODS[p.se_method_name]
    assert contract.principles == ("IIM",)
    assert set(IIM.BOOTSTRAP_SE_DF_ALLOWED) == set(contract.df_values)
    assert IIM.IIMParams(bootstrap_se_df=9).se_df == 9.0
    assert IIM.IIMParams(bootstrap_se_df=12.0).se_df == 12.0  # the pre-data value
    with pytest.raises(ValueError):
        IIM.IIMParams(bootstrap_se_df=49)  # never B - 1
    jk = IIM.IIMParams(se_method=IIM.SE_METHOD_JACKKNIFE)
    assert jk.se_method_name == "jackknife_contiguous_10" and jk.se_df == 9.0
    assert jk.se_method_name in EV.SE_METHODS
    off = IIM.IIMParams(bootstrap_replicates=8)
    assert off.se_method_name == "circular_block_bootstrap_10pct_B8"
    assert off.se_method_name not in EV.SE_METHODS


def test_bootstrap_run_reports_se_and_df():
    _, ts = binary_run("ring", coupling=0.45, T=3000, seed=12)
    r = IIM.compute_iim_v5(ts, lag=1, params={"n_null": 19}, null_seed=2, se_seed=3)
    assert r["defined"] and r["se_method"] == "circular_block_bootstrap_10pct_B50"
    assert r["se_df"] == IIM.BOOTSTRAP_SE_DF and r["se"] > 0
    boot = r["details"]["bootstrap"]
    assert boot["n_blocks"] == 10 and boot["block_length"] == 300
    for cut in IIM.CUT_MODES:
        reps = np.asarray(boot["replicates"][cut])
        assert reps.size == 50
        assert r["cuts"][cut]["se"] == pytest.approx(np.std(reps, ddof=1))
    again = IIM.compute_iim_v5(ts, lag=1, params={"n_null": 19}, null_seed=2, se_seed=3)
    assert again["se"] == r["se"]
    jk = IIM.compute_iim_v5(ts, lag=1, params={"n_null": 19, "se_method":
                                               IIM.SE_METHOD_JACKKNIFE}, null_seed=2)
    assert jk["se_df"] == 9.0 and len(jk["details"]["jackknife"]["replicates"]
                                      ["directional"]) == 10
    a = EV.assess_item(IIM.evidence(r), protocol(anchor=0.03))
    assert a.reason not in (R.INVALID_SE, R.NO_SAMPLING_SE)
    # the status rule reads the record's se_df (Welch-Satterthwaite never goes
    # below it)
    assert a.df >= IIM.BOOTSTRAP_SE_DF - 1e-9 and math.isfinite(a.df)
    # an explicit pre-data df of 12 is carried through to the record and the rule
    r12 = IIM.compute_iim_v5(ts, lag=1, params={"n_null": 19, "bootstrap_se_df": 12.0},
                             null_seed=2, se_seed=3)
    assert r12["se_df"] == 12.0 and r12["se"] == r["se"]
    a12 = EV.assess_item(IIM.evidence(r12), protocol(anchor=0.03))
    assert a12.df >= 12.0 - 1e-9 and math.isfinite(a12.df)


@pytest.mark.parametrize("se_method", [IIM.SE_METHOD_BOOTSTRAP, IIM.SE_METHOD_JACKKNIFE])
def test_resampling_keeps_the_strata_aligned_with_the_series(monkeypatch, se_method):
    """The bootstrap and the jackknife resample the analysed series and the
    strata labels with the same sample indices, so every replicate keeps
    each sample's own driver state (a misaligned replicate would read the
    driver as integration)."""
    _, ts = binary_run("independent", n=3, T=2000, seed=23)
    driver = np.random.default_rng(6).integers(0, 2, size=ts.shape[1])
    calls, picks = [], []
    real_stat, real_boot, real_keep = IIM._statistic, IIM.bootstrap_indices, IIM.jackknife_keep

    def spy_stat(y, strata, lag, cut_modes):
        calls.append((np.array(y, copy=True), np.array(strata, copy=True)))
        return real_stat(y, strata, lag, cut_modes)

    def spy_boot(*a, **k):
        picks.append(real_boot(*a, **k))
        return picks[-1]

    def spy_keep(*a, **k):
        picks.append(real_keep(*a, **k))
        return picks[-1]

    monkeypatch.setattr(IIM, "_statistic", spy_stat)
    monkeypatch.setattr(IIM, "bootstrap_indices", spy_boot)
    monkeypatch.setattr(IIM, "jackknife_keep", spy_keep)
    params = {"n_null": 19, "se_method": se_method, "bootstrap_replicates": 6}
    r = IIM.compute_iim_v5(ts, lag=1, params=params, strata=driver, null_seed=2, se_seed=3)
    assert r["defined"] and r["null_family"] == IIM.NULL_FAMILY_STRATIFIED
    want = 6 if se_method == IIM.SE_METHOD_BOOTSTRAP else IIM.JACKKNIFE_GROUPS
    assert len(calls) == len(picks) == want
    for (y, s), idx in zip(calls, picks):
        np.testing.assert_array_equal(y, ts[:, idx])
        np.testing.assert_array_equal(s, driver[idx])
    assert math.isfinite(r["se"]) and r["se"] > 0


def test_too_few_null_draws_is_no_null_calibration():
    _, ts = binary_run("ring", coupling=0.45, T=1500)
    r = IIM.compute_iim_v5(ts, lag=1, params={"n_null": 10, "se_method": None})
    assert r["reason"] == R.NO_NULL_CALIBRATION and not r["defined"]


# --------------------------------------------------------------------------
# sensor level: rank condition and ZCA
# --------------------------------------------------------------------------
def montage(reference="average", T=4000, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.normal(size=(6, T))
    for t in range(1, T):
        src[:, t] += 0.8 * src[:, t - 1]
    mix = rng.normal(size=(64, 6))
    x = mix @ src + 0.1 * rng.normal(size=(64, T))
    if reference == "average":
        x = x - x.mean(axis=0, keepdims=True)
    return x


def v1_quadrants(n_sensors=64):
    s = forward.sensor_positions(n_sensors)
    out = {}
    for name, sx, sy in IIM.QUADRANTS:
        out[name] = [i for i in range(n_sensors)
                     if np.sign(s[i, 0]) == sx and np.sign(s[i, 1]) == sy]
    return out


def test_macro_rank_deficient_on_an_average_referenced_full_montage():
    quads = v1_quadrants()
    assert sorted(i for v in quads.values() for i in v) == list(range(64))
    x = montage("average")
    rank = IIM.zero_lag_rank(IIM.macro_signals(x, quads))
    assert not rank["ok"] and rank["ratio"] < 1e-6
    r = IIM.compute_iim_v5(x, lag=2, macro_nodes=quads, params=NO_SE,
                           observation_stage="sensor")
    assert r["reason"] == R.MACRO_RANK_DEFICIENT  # before the observation gate
    r = IIM.compute_iim_v5(montage("none"), lag=2, macro_nodes=quads, params=NO_SE,
                           observation_stage="sensor", observation_admitted=True)
    assert r["details"]["rank_condition"]["ok"]
    clusters = IIM.rank_safe_clusters(forward.sensor_positions(64))
    members = [i for v in clusters.values() for i in v]
    assert [len(v) for v in clusters.values()] == [8, 8, 8, 8]
    assert len(set(members)) == 32  # disjoint, and they leave half the montage out
    assert IIM.zero_lag_rank(IIM.macro_signals(x, clusters))["ok"]
    assert [len(v) for v in IIM.rank_safe_clusters(
        forward.sensor_positions(32)).values()] == [4, 4, 4, 4]


def test_rank_safe_clusters_skip_midline_electrodes():
    """Electrodes on a midline (a zero coordinate) belong to no quadrant;
    each cluster holds the k electrodes nearest its quadrant's centroid, and
    a quadrant with fewer than k electrodes is refused."""
    grid = np.array([(x, y) for x in (-2, -1, 0, 1, 2) for y in (-2, -1, 0, 1, 2)],
                    dtype=float)
    clusters = IIM.rank_safe_clusters(grid, n_per_cluster=4)
    members = [i for v in clusters.values() for i in v]
    assert len(members) == len(set(members)) == 16
    assert not any(grid[i, 0] == 0 or grid[i, 1] == 0 for i in members)
    for name, sx, sy in IIM.QUADRANTS:
        assert all(np.sign(grid[i, 0]) == sx and np.sign(grid[i, 1]) == sy
                   for i in clusters[name])
    # one electrode per quadrant nearest its centroid at (+-1.5, +-1.5)
    assert len(IIM.rank_safe_clusters(grid, n_per_cluster=1)["R_ant"]) == 1
    with pytest.raises(ValueError):
        IIM.rank_safe_clusters(grid, n_per_cluster=5)
    with pytest.raises(ValueError):
        IIM.rank_safe_clusters(grid[:, :1])


def test_v1_quadrant_montage_on_hopf_sources_is_rank_deficient():
    eeg = make_system("whole_brain_eeg", seed=3, whole_brain_config={"G": 0.0})
    r = IIM.compute_iim_v5_system(eeg, params=NO_SE)
    assert r["details"]["observation_stage"] == "sensor"
    assert r["reason"] == R.MACRO_RANK_DEFICIENT


def test_zca_orthogonalisation():
    x = montage("average", T=3000, seed=4)
    y = IIM.macro_signals(x, IIM.rank_safe_clusters(forward.sensor_positions(64)))
    yt, w = IIM.zca(y)
    np.testing.assert_allclose(w, w.T, atol=1e-12)
    np.testing.assert_allclose(yt @ yt.T / yt.shape[1], np.eye(4), atol=1e-9)
    yc = y - y.mean(axis=1, keepdims=True)
    cov = yc @ yc.T / yc.shape[1]
    np.testing.assert_allclose(w @ cov @ w, np.eye(4), atol=1e-9)
    # the magnitude-preserving variant binarises to the same process
    scaled = yt * yc.std(axis=1, keepdims=True)
    np.testing.assert_array_equal(IIM.binarise(yt), IIM.binarise(scaled))
    with pytest.raises(np.linalg.LinAlgError):
        IIM.zca(np.vstack([y, y[0] + y[1]]))
    # without null draws the run stops at NO_NULL_CALIBRATION, after the
    # pre-processing and the gates
    r = IIM.compute_iim_v5(y, lag=2, params={"n_null": 0, "se_method": None,
                                             "preprocess": "zca"},
                           observation_stage="sensor", observation_admitted=True)
    assert r["reason"] == R.NO_NULL_CALIBRATION
    assert r["details"]["preprocess"] == "zca"
    assert r["details"]["occupancy"]["passes"]
    np.testing.assert_allclose(np.asarray(r["details"]["zca_matrix"]), w, atol=1e-12)


def test_observation_gate():
    x = montage("average", T=3000, seed=5)
    y = IIM.macro_signals(x, IIM.rank_safe_clusters(forward.sensor_positions(64)))
    fast = {"n_null": 0, "se_method": None, "preprocess": "zca"}
    for stage in ("sensor", "source_estimate"):
        r = IIM.compute_iim_v5(y, lag=2, params=fast, observation_stage=stage)
        assert r["reason"] == R.OBSERVATION_MIXED_NOT_ADMITTED
        assert r["details"]["occupancy"]["passes"]  # the data gates came first
    # admitted (the forward-arm admission runs): the gate is passed
    r = IIM.compute_iim_v5(y, lag=2, params=fast, observation_stage="sensor",
                           observation_admitted=True)
    assert r["reason"] == R.NO_NULL_CALIBRATION
    for stage in ("source", "bold"):
        r = IIM.compute_iim_v5(y, lag=2, params=fast, observation_stage=stage)
        assert r["reason"] == R.NO_NULL_CALIBRATION
    with pytest.raises(ValueError):
        IIM.compute_iim_v5(y, lag=2, params=fast, observation_stage="scalp")


# --------------------------------------------------------------------------
# reasons, params, records
# --------------------------------------------------------------------------
def test_every_v5_reason_maps_to_undefined():
    for reason in (R.MACRO_RANK_DEFICIENT, R.INSUFFICIENT_OCCUPANCY,
                   R.NO_NULL_CALIBRATION, R.OBSERVATION_MIXED_NOT_ADMITTED,
                   "INCONCLUSIVE:NULL_NOT_EXCEEDED",
                   IIM._not_defined("macro_nodes_1_outside_2_6")):
        assert R.status_for_reason(reason) == R.UNDEFINED
    r = IIM.compute_iim_v5(np.ones((1, 100)), lag=1, params=NO_SE)
    assert r["reason"].startswith("NOT_DEFINED:") and not r["defined"]
    x = np.zeros((3, 100))
    x[0, 5] = np.nan
    assert IIM.compute_iim_v5(x, lag=1, params=NO_SE)["reason"].startswith("NOT_DEFINED")


def test_params_validation():
    with pytest.raises(ValueError):
        IIM.IIMParams.from_mapping({"bogus": 1})
    with pytest.raises(ValueError):
        IIM.IIMParams(cut_mode="mixed")
    with pytest.raises(ValueError):
        IIM.IIMParams(null_order="residualise_then_shift")
    with pytest.raises(ValueError):
        IIM.IIMParams(se_method="jackknife_contiguous_20")
    p = IIM.IIMParams(report_cut_modes=("directional", "bidirectional", "bidirectional"))
    assert p.report_cut_modes == ("bidirectional",)
    assert IIM.IIMParams.from_mapping(p.to_dict()) == p


def test_component_fields_and_records():
    _, ts = binary_run("ring", coupling=0.45, T=1500)
    r = IIM.compute_iim_v5(ts, lag=1, params=NO_SE, null_seed=4)
    proto = protocol()
    for cut in IIM.CUT_MODES:
        fields = IIM.component_fields(r, cut)
        assert fields["details"]["p_ind"] == r["cuts"][cut]["p_ind"]
        assert fields["details"]["estimator_id"] == IIM.estimator_id(cut)
        ev = IIM.evidence(r, cut)
        assert ev.p_ind == r["cuts"][cut]["p_ind"] and ev.se_method is None
        a = EV.assess_item(ev, protocol(se_methods=None))
        assert a.reason == R.NO_SAMPLING_SE
        comp = REC.ComponentRecord(
            principle="IIM", status=a.status.value, reason=a.reason,
            estimator_version=IIM.ESTIMATOR_VERSION, declaration_id="none",
            observation_stage="source", protocol_id=proto.protocol_id,
            protocol_hash=proto.hash, c=a.c, **fields)
        assert REC.ComponentRecord.from_dict(comp.to_dict()) == comp
    undefined = IIM.compute_iim_v5(np.ones((1, 50)), lag=1, params=NO_SE)
    fields = IIM.component_fields(undefined)
    assert fields["estimate"] is None and fields["details"]["defined"] is False
    with pytest.raises(KeyError):
        IIM.component_fields(fake_result(0.03, 0.0, 0.01, 0.05), "bidirectional")


def _same_result(a, b):
    """Two results of compute_iim_v5 hold the same values (NaN equals NaN)."""
    import json

    def norm(x):
        return json.loads(json.dumps(x, sort_keys=True, default=str).replace(
            "NaN", "null"))

    return norm(a) == norm(b)


@pytest.mark.parametrize("basis_decl", [None, "R"])
def test_a_computed_cut_mode_is_read_from_a_run_that_reports_it(basis_decl):
    """The values of a cut mode do not depend on the other cut modes of the
    run (same pairs, surrogates and resamples): select_cut_modes gives the
    result of a run that computes the requested cut modes alone."""
    sysm = make_system("family_a", seed=3)
    basis = None
    if basis_decl is not None:
        dec = DI.declare(DI.record_inputs(sysm), basis_decl)
        basis = DI.input_basis(dec, tau_c=0.1, estimator="IIM")
    small = {"n_null": 19, "bootstrap_replicates": 3}
    both = IIM.compute_iim_v5_system(sysm, basis=basis, null_seed=5, params=small)
    assert IIM.cut_modes_computed(both) == ("directional", "bidirectional")
    for cut in ("bidirectional", "directional"):
        want = dict(small, cut_mode=cut, report_cut_modes=())
        alone = IIM.compute_iim_v5_system(sysm, basis=basis, null_seed=5, params=want)
        got = IIM.select_cut_modes(both, want)
        assert _same_result(got, alone), cut
    # the primary result itself
    assert _same_result(IIM.select_cut_modes(both, small), both)
    # nothing else may differ, and an uncomputed cut cannot be read
    with pytest.raises(ValueError, match="more than the cut modes"):
        IIM.select_cut_modes(both, dict(small, n_null=20))
    alone = IIM.compute_iim_v5_system(sysm, basis=basis, null_seed=5,
                                      params=dict(small, report_cut_modes=()))
    with pytest.raises(ValueError, match="not computed"):
        IIM.select_cut_modes(alone, dict(small, cut_mode="bidirectional",
                                         report_cut_modes=()))


def test_reading_a_cut_mode_from_an_undefined_run():
    flat = np.ones((4, 400))
    flat[0] += np.arange(400) % 2  # three constant macro nodes
    res = IIM.compute_iim_v5(flat, lag=1, params=NO_SE)
    assert res["reason"] == R.INSUFFICIENT_OCCUPANCY
    got = IIM.select_cut_modes(res, dict(NO_SE, cut_mode="bidirectional"))
    assert got["reason"] == R.INSUFFICIENT_OCCUPANCY
    assert got["estimator_id"] == IIM.estimator_id("bidirectional")
    assert _same_result(got, IIM.compute_iim_v5(
        flat, lag=1, params=dict(NO_SE, cut_mode="bidirectional")))
    # too few null draws counts the draws of every computed cut: computed anew
    few = IIM.compute_iim_v5(np.random.default_rng(1).standard_normal((3, 600)),
                             lag=1, params={"n_null": 5, "se_method": None})
    assert few["reason"] == R.NO_NULL_CALIBRATION
    with pytest.raises(ValueError, match="depends on its cut modes"):
        IIM.select_cut_modes(few, {"n_null": 5, "se_method": None,
                                   "cut_mode": "bidirectional"})
