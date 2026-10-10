# -*- coding: utf-8 -*-
"""
The staggered hidden-driver adversaries of MPC-Bench v2: construction (no
edges, six-level switching driver with one pattern per level and module,
per-module low-pass, AR(1) unit noise), the time constants and transform of
each variant (hierarchical, uniform, reversed and the held-out tau10 and
saturating variants), the reserved random stream, and the v2 adversary
dispatcher.
"""
import math

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.bench import adversarial_v2 as A2
from impact_pipeline.bench import generators as g
from impact_pipeline.bench import manipulation_v2 as MV
from impact_pipeline.v2 import GENERATOR_VERSION_V2
from impact_pipeline.v2 import seeds as S

VARIANTS = tuple(A2.STAGGERED_VARIANTS)
SEED = 3


@pytest.fixture(scope="module")
def systems():
    return {v: A2.staggered_driver_system(v, SEED) for v in VARIANTS}


# --------------------------------------------------------------------------
# stated time constants and transforms
# --------------------------------------------------------------------------
STATED = {
    "hierarchical": {"S": 0.05, "C": 0.05, "W": 0.3, "M": 0.8, "V": 0.8},
    "uniform": {"S": 0.3, "C": 0.3, "W": 0.3, "M": 0.3, "V": 0.3},
    "reversed": {"S": 0.8, "C": 0.8, "W": 0.3, "M": 0.05, "V": 0.05},
    "tau10": {"S": 0.5, "C": 0.5, "W": 3.0, "M": 8.0, "V": 8.0},
    "saturating": {"S": 0.05, "C": 0.05, "W": 0.3, "M": 0.8, "V": 0.8},
}


def test_the_variant_table_is_the_design():
    assert set(STATED) == set(VARIANTS)
    for v, taus in STATED.items():
        assert A2.staggered_module_taus(v, ("S", "C", "W", "M", "V")) == taus
    tau10 = A2.staggered_module_taus("tau10", ("S", "C", "W", "M", "V"))
    hier = A2.staggered_module_taus("hierarchical", ("S", "C", "W", "M", "V"))
    assert all(math.isclose(tau10[k], 10 * hier[k]) for k in hier)
    held = {v for v, spec in A2.STAGGERED_VARIANTS.items() if spec["held_out"]}
    assert held == {"tau10", "saturating"}
    assert A2.STAGGERED_SYSTEM_VARIANTS == {
        "ADV_NAS_staggered_driver": ("hierarchical", "uniform", "reversed"),
        "ADV_NAS_staggered_tau10": ("tau10",),
        "ADV_NAS_staggered_sat": ("saturating",),
    }
    # other layouts: upstream / hub / downstream by position
    assert A2.staggered_module_taus("hierarchical", ("S", "C", "A", "W", "M", "V")) == {
        "S": 0.05, "C": 0.05, "A": 0.05, "W": 0.3, "M": 0.8, "V": 0.8}
    with pytest.raises(ValueError):
        A2.staggered_module_taus("nope", ("S", "W"))
    with pytest.raises(ValueError):
        A2.staggered_module_taus("uniform", ("S", "M"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_realise_their_stated_time_constants(systems, variant):
    s = systems[variant]
    assert s.oracle["module_tau_sec"] == STATED[variant]
    real = MV.realised_time_constants(s)
    for mod, tau in STATED[variant].items():
        assert real[mod] == pytest.approx(tau, rel=1e-9), mod
    # the step response of every unit: after a switch held for k samples,
    # the filtered drive has covered 1 - a**k of the jump
    y = s.oracle["drive_filtered"]
    u = A2.staggered_drive_input(s)
    sw = s.oracle["driver_switch_idx"]
    for mod, tau in STATED[variant].items():
        a = math.exp(-s.dt / tau)
        i = int(s.meta["modules"][mod][0])
        for j in range(1, min(len(sw) - 1, 40)):
            t0, t1 = sw[j], sw[j + 1]
            if t1 - t0 < 3 or u[i, t0] == u[i, t0 - 1]:
                continue
            start, target = y[i, t0 - 1], u[i, t0]
            k = np.arange(1, t1 - t0 + 1)
            covered = (y[i, t0 - 1 + k] - start) / (target - start)
            assert np.allclose(covered, 1.0 - a ** k, atol=1e-9), (mod, j)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_apply_their_stated_transform(systems, variant):
    s = systems[variant]
    y, x = s.oracle["drive_filtered"], s.oracle["hidden_signal"]
    if variant == "saturating":
        assert s.oracle["transform"] == "tanh"
        assert np.array_equal(x, np.tanh(2.0 * y))
        assert np.max(np.abs(x)) < 1.0
    else:
        assert s.oracle["transform"] == "linear"
        assert np.array_equal(x, y)


def test_hierarchical_upstream_leads_the_hub_and_the_hub_leads_downstream(systems):
    """With one shared driver, a faster module tracks it sooner: the
    cross-correlation peaks of the module drives put S before W before M."""
    s = systems["hierarchical"]
    y = s.oracle["drive_filtered"]
    mods = s.meta["modules"]
    sig = {m: y[np.asarray(mods[m])].mean(axis=0) for m in ("S", "W", "M")}

    def lag(a, b, max_lag=40):
        a = (a - a.mean()) / a.std()
        b = (b - b.mean()) / b.std()
        cc = [np.mean(a[: len(a) - k] * b[k:]) for k in range(max_lag)]
        return int(np.argmax(cc))

    assert lag(sig["S"], sig["W"]) > 0
    assert lag(sig["W"], sig["M"]) > 0
    assert lag(sig["M"], sig["S"]) == 0


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def test_no_edges_and_the_declared_template(systems):
    like = g.simulate_family_a(None, None, SEED)
    for v, s in systems.items():
        assert s.ts.shape == like.ts.shape
        assert not np.any(s.oracle["unit_adjacency"])
        assert not np.any(s.oracle["module_adjacency"])
        p = MV.hub_periphery_paths(s)
        assert not (p["hub_to_periphery"] or p["periphery_to_hub"])
        pd.testing.assert_frame_equal(s.events, like.events)
        assert s.meta["modules"] == like.meta["modules"]
        assert s.meta["workspace_nodes"] == like.meta["workspace_nodes"]
        assert s.meta["family"] == "adversarial_staggered_driver"
        assert s.meta["generator_version"] == GENERATOR_VERSION_V2
        assert s.meta["adversarial_version"] == A2.ADVERSARIAL_V2_VERSION
        assert s.meta["knobs"] is None and s.meta["staggered_variant"] == v
        assert s.oracle["held_out"] == A2.STAGGERED_VARIANTS[v]["held_out"]
        assert s.oracle["intended_bits"] == [0, None, 0, 0, 0]
        assert s.oracle["mechanisms"]["NAS"] is False
        assert s.oracle["designed_to_fool"] == "NAS"
        assert "hidden_envelope" not in s.oracle and "slow_phase" not in s.oracle


def test_driver_levels_dwells_and_patterns(systems):
    s = systems["hierarchical"]
    level = s.oracle["context_state"]
    sw, lv = s.oracle["driver_switch_idx"], s.oracle["driver_levels"]
    assert level.shape == (s.n_time,) and level.dtype == np.int64
    assert set(np.unique(level)) <= set(range(6)) and len(np.unique(level)) == 6
    assert sw[0] == 0
    dwell = np.diff(sw) * s.dt
    assert dwell.min() >= 1.0 - 1e-9 and dwell.max() <= 3.0 + 1e-9
    for j in range(len(sw)):
        end = sw[j + 1] if j + 1 < len(sw) else s.n_time
        assert np.all(level[sw[j]:end] == lv[j])
    pat = s.oracle["driver_patterns"]
    assert pat.shape == (6, 5, 6)
    # one pattern per level and module: every unit receives its module's
    # pattern of the active level
    u = A2.staggered_drive_input(s)
    for k, mod in enumerate(s.meta["module_order"]):
        idx = np.asarray(s.meta["modules"][mod])
        assert np.array_equal(u[idx], pat[level][:, k, :].T)


def test_the_unit_noise_is_the_stated_ar1(systems):
    s = systems["uniform"]
    resid = s.ts - s.oracle["hidden_signal"]
    assert resid.std() == pytest.approx(0.5, rel=0.03)
    r1 = np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in resid])
    assert r1 == pytest.approx(0.7, abs=0.02)
    # unit noises are independent of each other
    c = np.corrcoef(resid)
    assert np.max(np.abs(c[np.triu_indices_from(c, 1)])) < 0.08


def test_draws_come_from_the_reserved_stream_and_are_shared_by_variants(systems):
    sched, pattern, _noise = S.stream_seed_sequence(SEED, "staggered_driver").spawn(3)
    rng = np.random.default_rng(sched)
    first_dwell = max(1, int(round(rng.uniform(1.0, 3.0) / 0.05)))
    first_level = int(rng.integers(0, 6))
    s = systems["reversed"]
    assert s.oracle["driver_switch_idx"][1] == first_dwell
    assert s.oracle["driver_levels"][0] == first_level
    pat = np.random.default_rng(pattern).standard_normal((6, 5, 6))
    for v in VARIANTS:
        assert np.array_equal(systems[v].oracle["driver_patterns"], pat)
        assert np.array_equal(systems[v].oracle["context_state"],
                              systems["hierarchical"].oracle["context_state"])
        assert np.allclose(systems[v].ts - systems[v].oracle["hidden_signal"],
                           s.ts - s.oracle["hidden_signal"], rtol=0, atol=1e-12)
    other = A2.staggered_driver_system("reversed", SEED + 1)
    assert not np.array_equal(other.oracle["context_state"], s.oracle["context_state"])
    again = A2.staggered_driver_system("reversed", SEED)
    assert np.array_equal(again.ts, s.ts)


def test_the_driver_reaches_every_module(systems):
    for v, s in systems.items():
        ci = MV.module_context_information(s)
        assert set(ci) == set(s.meta["module_order"])
        assert min(ci.values()) >= MV.CONTEXT_FLOOR, (v, ci)


def test_parameters_and_template(systems):
    small = g.AgentConfig(n_trials=8, n_reafference_pairs=4)
    s = A2.staggered_driver_system("uniform", 1, config=small, drive_gain=2.0,
                                   noise_sd=0.0)
    like = g.simulate_family_a(None, small, 1)
    assert s.ts.shape == like.ts.shape
    assert np.array_equal(s.ts, s.oracle["hidden_signal"])  # no noise
    unit_pat = s.oracle["driver_unit_patterns"]
    assert np.allclose(A2.staggered_drive_input(s),
                       2.0 * unit_pat[s.oracle["context_state"]].T)
    with pytest.raises(ValueError):
        A2.staggered_driver_system("nope", 0)
    for bad in ({"n_levels": 1}, {"drive_gain": 0.0}, {"noise_ar": 1.0},
                {"dwell_sec": (2.0, 1.0)}):
        with pytest.raises(ValueError):
            A2.staggered_driver_system("uniform", 0, like=like, **bad)
    with pytest.raises(ValueError):
        A2.staggered_drive_input(like)


@pytest.mark.parametrize("n_modules", [4, 6])
def test_other_module_layouts(n_modules):
    """Time constants follow the module's position relative to the hub in
    every family-A layout."""
    cfg = g.AgentConfig(n_trials=16, n_reafference_pairs=6, n_modules=n_modules)
    s = A2.staggered_driver_system("hierarchical", 2, config=cfg)
    order = s.meta["module_order"]
    assert len(order) == n_modules and s.ts.shape[0] == s.meta["n_nodes"]
    k = order.index("W")
    want = {m: 0.05 if i < k else 0.3 if i == k else 0.8 for i, m in enumerate(order)}
    assert s.oracle["module_tau_sec"] == want
    real = MV.realised_time_constants(s)
    assert all(real[m] == pytest.approx(want[m], rel=1e-9) for m in order)
    assert s.oracle["driver_patterns"].shape == (6, n_modules, 6)
    rows = MV.check_system(s, "ADV_NAS_staggered_driver", 2)
    assert {r["check"] for r in rows} == {"driver_reaches_every_module",
                                          "stated_time_constants"}
    assert rows[1]["passed"]


@pytest.mark.parametrize("sid,variants", sorted(A2.STAGGERED_SYSTEM_VARIANTS.items()))
def test_catalogue_builds_every_staggered_variant(sid, variants):
    for v in variants:
        s = A2.build_system(sid, 4, variant=v)
        assert s.meta["adversary_id"] == sid and s.meta["adversary_variant"] == v
        assert s.oracle["adversary_intended_pattern"] == [0, None, 0, 0, 0]
        assert s.oracle["adversary_expected_verdict"] == "EXCLUDED"
    assert A2.build_system(sid, 4).meta["adversary_variant"] == variants[0]
    with pytest.raises(ValueError):
        A2.build_system(sid, 4, variant="nope")
    with pytest.raises(ValueError):
        A2.build_system(sid, 4, replicate=1)
    with pytest.raises(ValueError):
        A2.build_system(sid, 4, family="C")
