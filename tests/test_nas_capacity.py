"""Known-answer and null tests for NAS broadcast capacity (mode='capacity')."""

import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm

N_NODES, N_HUB = 24, 4
HUB = np.arange(N_HUB)


def _hub_network(kind, seed, n_time=1500, gain=0.4, n_hub=N_HUB):
    """
    Linear AR(1) network (numpy only). ``kind``:
      - 'none': independent nodes (no hub);
      - 'ff': the hub drives the periphery, nothing returns (feedforward only);
      - 'bidir': the hub receives from the periphery and returns to it;
      - 'common': an external AR(1) driver feeds hub and periphery alike
        (hypersynchrony without any hub-periphery coupling); also returned.
    """
    rng = np.random.default_rng(seed)
    hub = np.arange(n_hub)
    per = np.arange(n_hub, N_NODES)
    w_in = rng.standard_normal((n_hub, per.size)) / np.sqrt(per.size)
    w_out = rng.standard_normal((per.size, n_hub)) / np.sqrt(n_hub)
    x = np.zeros((N_NODES, n_time))
    drive = np.zeros(n_time)
    for t in range(1, n_time):
        prev = x[:, t - 1]
        new = 0.6 * prev + rng.standard_normal(N_NODES)
        if kind in ("ff", "bidir"):
            new[per] += gain * w_out @ prev[hub]
        if kind == "bidir":
            new[hub] += gain * w_in @ prev[per]
        if kind == "common":
            drive[t] = 0.9 * drive[t - 1] + rng.standard_normal()
            new += 1.5 * drive[t - 1]
        x[:, t] = new
    return x, drive


def _cap(x, seed=0, **kw):
    return mm.compute_NAS(
        x,
        tr=1.0,
        mode="capacity",
        workspace_nodes=HUB,
        null_seed=seed,
        return_details=True,
        **kw,
    )


def test_hub_that_receives_and_returns_has_positive_excess():
    for seed in range(3):
        x, _ = _hub_network("bidir", seed)
        d = _cap(x, seed)
        tr = d["transfer"]
        assert d["defined"] and d["mode"] == "capacity"
        assert tr["te_in_z"] > 3.0 and tr["te_out_z"] > 3.0
        assert d["value"] > 0.0 and d["NAS_z"] > 3.0
        # Gated value = excess of the weaker direction (intersection-union).
        assert d["NAS_z"] == pytest.approx(min(tr["te_in_z"], tr["te_out_z"]))
        lim = d["limiting_direction"]
        assert d["value"] == pytest.approx(tr[f"te_{lim}_excess"])
        assert d["raw"] == pytest.approx(tr[f"te_{lim}"])
        assert d["NAS_excess"] == pytest.approx(d["value"])


@pytest.mark.parametrize("kind", ["ff", "none"])
def test_feedforward_only_or_no_hub_is_about_zero(kind):
    zs, vals = [], []
    for seed in range(4):
        x, _ = _hub_network(kind, seed)
        d = _cap(x, seed)
        zs.append(d["NAS_z"])
        vals.append(d["value"])
        if kind == "ff":
            # Broadcast without return: a huge hub -> periphery transfer...
            assert d["transfer"]["te_out_z"] > 50.0
            # ...but the receive direction limits the gated value.
            assert d["limiting_direction"] == "in"
    print(kind, "NAS capacity z:", np.round(zs, 2))
    assert np.all(np.asarray(zs) < 2.5)
    assert abs(float(np.mean(vals))) < 0.01


def test_common_driver_without_return_is_partly_inflated_and_fixed_by_conditioning():
    # Reported behaviour (not the intended construct): a common driver of hub
    # and periphery inflates the receive direction because the periphery's
    # past is a better estimate of the driver than the hub's own past. The
    # weaker (return) direction mostly stays near its null, so the gated value
    # is rarely inflated here (4-node hub vs 20-node periphery); with larger,
    # symmetric hubs it is inflated more often. Conditioning on the measured
    # driver removes the effect.
    zin, zgate, zcond = [], [], []
    for seed in range(4):
        x, drive = _hub_network("common", seed)
        d = _cap(x, seed)
        zin.append(d["transfer"]["te_in_z"])
        zgate.append(d["NAS_z"])
        c = _cap(x, seed, confounds=drive)
        zcond.append(c["NAS_z"])
        assert c["transfer"]["n_confounds"] == 1
        assert abs(c["transfer"]["te_in_z"]) < 3.0
    print(
        "common driver: receive z",
        np.round(zin, 1),
        "gated z",
        np.round(zgate, 1),
        "conditioned z",
        np.round(zcond, 1),
    )
    assert np.mean(zin) > 3.0
    assert np.all(np.asarray(zgate) < np.asarray(zin))
    assert np.all(np.abs(zcond) < 3.0)


def test_metastability_and_synchrony_descriptors_do_not_enter_the_value():
    x, _ = _hub_network("bidir", 0, n_time=800)
    plain = _cap(x)
    with_bands = _cap(x, bands=[(0.05, 0.2)])
    profile = _cap(
        x, bands=[(0.05, 0.2)], band_weights=[1.0], tau=0.2, window_len=100, step_len=50
    )
    assert plain["metastability"]["reason"] == "no_bands_declared"
    assert np.isnan(plain["metastability"]["value"])
    meta = with_bands["metastability"]
    assert meta["reason"] is None and np.isfinite(meta["z"]) and meta["value"] > 0
    assert plain["profile_descriptors"]["reason"] == "profile_parameters_missing"
    desc = profile["profile_descriptors"]
    assert desc["reason"] is None
    assert all(np.isfinite(desc[k]) for k in ("L", "B", "H"))
    assert "D" not in desc
    assert plain["value"] == with_bands["value"] == profile["value"]


def test_metastability_is_high_for_common_driver_but_not_for_the_linear_hub():
    # Why metastability is reported but not gated: in these linear systems it
    # is large for driver-induced hypersynchrony and about null for genuine
    # receive-and-return broadcast.
    common, _ = _hub_network("common", 0, n_time=1000)
    bidir, _ = _hub_network("bidir", 0, n_time=1000)
    m_common = _cap(common, bands=[(0.05, 0.2)])["metastability"]
    m_bidir = _cap(bidir, bands=[(0.05, 0.2)])["metastability"]
    print(
        "metastability z: common",
        round(m_common["z"], 1),
        "bidir",
        round(m_bidir["z"], 1),
    )
    assert m_common["z"] > 5.0
    assert m_bidir["z"] < 3.0


def test_capacity_contract_and_validation():
    x, _ = _hub_network("bidir", 0, n_time=300)
    with pytest.raises(ValueError, match="workspace_nodes"):
        mm.compute_NAS(x, tr=1.0, mode="capacity")
    with pytest.raises(ValueError, match="positive tr"):
        mm.compute_NAS(x, mode="capacity", workspace_nodes=HUB)
    with pytest.raises(ValueError, match="confounds"):
        _cap(x, confounds=np.zeros(10))
    with pytest.raises(ValueError, match="null_surrogates >= 2"):
        _cap(x, null_surrogates=1)
    with pytest.raises(ValueError, match="transfer_lags"):
        _cap(x, transfer_lags=(0,))
    with pytest.raises(ValueError, match="mode must be one of"):
        mm.compute_NAS(x, tr=1.0, mode="broadcast")
    all_hub = mm.compute_NAS(
        x,
        tr=1.0,
        mode="capacity",
        workspace_nodes=np.arange(N_NODES),
        return_details=True,
    )
    assert np.isnan(all_hub["value"])
    assert all_hub["undefined_reason"] == "hub_or_periphery_empty"
    short = mm.compute_NAS(
        x[:, :20], tr=1.0, mode="capacity", workspace_nodes=HUB, return_details=True
    )
    assert short["undefined_reason"] == "insufficient_timepoints_for_transfer_model"
    bad = x.copy()
    bad[3, 7] = np.nan
    assert np.isnan(mm.compute_NAS(bad, tr=1.0, mode="capacity", workspace_nodes=HUB))


def test_capacity_is_deterministic_and_scalar_matches_details():
    x, _ = _hub_network("bidir", 2, n_time=600)
    a = _cap(x, seed=5)
    b = _cap(x, seed=5)
    assert a["value"] == b["value"] and a["NAS_null_sd"] == b["NAS_null_sd"]
    assert a["n_surrogates_requested"] == mm.NAS_CAPACITY_DEFAULT_SURROGATES
    assert a["NAS_null_n"] == mm.NAS_CAPACITY_DEFAULT_SURROGATES
    assert a["NAS_null_method"] == "block_circular_shift"
    scalar = mm.compute_NAS(
        x, tr=1.0, mode="capacity", workspace_nodes=HUB, null_seed=5
    )
    assert scalar == a["value"]


def test_gaussian_transfer_entropy_known_answer():
    # Bivariate VAR(1): y_t = a x_{t-1} + e_t with unit-variance white x and e.
    # TE(x -> y) = 0.5 * log(1 + a^2) nats; TE(y -> x) = 0.
    rng = np.random.default_rng(0)
    n, a = 20000, 0.8
    x = rng.standard_normal(n)
    y = np.r_[0.0, a * x[:-1]] + rng.standard_normal(n)
    te_xy = mm._gaussian_transfer_entropy(x[None], y[None], (1,))
    te_yx = mm._gaussian_transfer_entropy(y[None], x[None], (1,))
    assert te_xy == pytest.approx(0.5 * np.log(1 + a * a), abs=0.01)
    assert abs(te_yx) < 0.002


def test_legacy_default_is_unchanged():
    x, _ = _hub_network("bidir", 0, n_time=200)
    kw = dict(
        tr=1.0,
        tau=0.2,
        bands=[(0.05, 0.2)],
        band_weights=[1.0],
        window_len=50,
        step_len=25,
    )
    assert mm.compute_NAS(x, **kw) == mm.compute_NAS(x, mode="legacy", **kw)
    d = mm.compute_NAS(x, return_details=True, **kw)
    assert d["mode"] == "legacy" and d["bearer_nodes"] is None
