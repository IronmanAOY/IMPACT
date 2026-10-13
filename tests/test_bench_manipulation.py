# -*- coding: utf-8 -*-
"""
Manipulation checks of the MPC-Bench switches on oracle channels (never on
MPC estimators): known-answer tests of the signatures, families A and C pass
every check on development seeds, and a weakly coupled family-C
configuration fails them.

Realisation checks of the configuration-defined systems: the oracle
signatures on hand-built fixtures, the gated checks on the real systems (and
their failure on systems that do not realise the mechanism), the usability
rule (36 of 40 seeds), the reported PC_half and twin checks, and the
structural and schedule hashes.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.bench import catalogue as CAT
from impact_pipeline.bench import generators as g
from impact_pipeline.bench import manipulation as MP
from impact_pipeline.bench import seeds as S

CHECK_CFG = g.AgentConfig(n_trials=40, n_reafference_pairs=15)


def _var(n, T, A, seed, noise=1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T))
    for t in range(1, T):
        x[:, t] = A @ x[:, t - 1] + noise * rng.standard_normal(n)
    return x


def test_granger_gain_known_answers():
    # Independent AR(1) processes: no gain in either direction (df-adjusted,
    # so short runs are not biased upwards).
    for seed in range(5):
        x = _var(2, 1500, np.diag([0.8, 0.5]), seed)
        assert MP.granger_gain(x[0], x[1]) < 0.01
    # x1 drives x0 (one direction only).
    A = np.array([[0.5, 0.6], [0.0, 0.5]])
    x = _var(2, 4000, A, 1)
    fwd, bwd = MP.granger_gain(x[0], x[1]), MP.granger_gain(x[1], x[0])
    assert fwd > 0.15 and bwd < 0.01
    # Complex signals: a rotated copy of the source's past is recovered.
    rng = np.random.default_rng(3)
    s = rng.standard_normal(3000) + 1j * rng.standard_normal(3000)
    tgt = np.r_[0, 0.8 * np.exp(1j) * s[:-1]] + 0.3 * (
        rng.standard_normal(3000) + 1j * rng.standard_normal(3000)
    )
    assert MP.granger_gain(tgt, s) > 0.8
    assert np.isnan(MP.granger_gain(np.arange(8.0), np.arange(8.0)))


def test_signature_known_answers():
    s = g.simulate_family_a(None, CHECK_CFG, seed=1)
    # One context pattern carries no context information by definition.
    k1 = g.simulate_family_a(g.NOMINAL_KNOBS.replace(K=1), CHECK_CFG, seed=1)
    assert MP.context_information(k1) == 0.0
    assert 0.0 < MP.context_information(s) < 1.0
    # Identical self and other responses give a contrast of about 0: replace
    # the hidden state by a copy in which each replay window repeats the
    # response of its self-caused event.
    ts = s.ts.copy()
    ev = s.events
    selfs = ev[ev.trial_type == "self_caused"].set_index("event_id")["onset"]
    pre = int(round(MP.PRE_WINDOW_SEC / s.dt))
    post = int(round(MP.RESPONSE_WINDOW_SEC / s.dt))
    for _, row in ev[ev.trial_type == "other_caused"].iterrows():
        i_o = int(round(row["onset"] / s.dt))
        i_s = int(round(selfs.loc[row["yoked_to"]] / s.dt))
        ts[:, i_o - pre : i_o + post] = ts[:, i_s - pre : i_s + post]
    same = g.BenchSystem(ts=ts, events=ev, meta=s.meta, oracle=s.oracle)
    assert MP.self_other_contrast(same) == pytest.approx(0.0, abs=1e-12)
    assert MP.self_other_contrast(s) > 0.3
    # Reversal tracking is the correlation of the hidden choice probability
    # with the rewarded arm.
    o = dict(s.oracle)
    o["p_choice1"] = o["good_arm"].astype(float)
    tracked = g.BenchSystem(ts=s.ts, events=ev, meta=s.meta, oracle=o)
    assert MP.reversal_tracking(tracked) == pytest.approx(1.0)
    assert MP.relative_change(0.0, 0.0) != MP.relative_change(0.0, 0.0)  # NaN
    with pytest.raises(ValueError):
        MP.hidden_state(s, "rotating")


@pytest.mark.parametrize(
    "simulate", [g.simulate_family_a, g.simulate_family_c], ids=["A", "C"]
)
def test_every_switch_passes_its_check(simulate):
    rep = MP.manipulation_report(simulate, (0, 1, 2), CHECK_CFG)
    assert set(rep["switch"]) == set(MP.SWITCHES)
    assert len(rep) == 3 * len(MP.SWITCHES)
    failed = rep[~rep["passed"]]
    assert failed.empty, failed.to_string()
    assert (rep["relative_change"] >= rep["threshold"]).all()
    assert (rep["nominal"] >= rep["floor"]).all()
    assert set(rep["check_version"]) == {MP.MANIPULATION_CHECK_VERSION}


def test_weakly_coupled_family_c_fails_the_checks():
    """Weak coupling against a strong additive-like drive and detuned
    oscillators does not pass the NAS and IIM checks: the checks can fail."""
    weak = CHECK_CFG.replace(sl_coupling=0.2, sl_input_gain=4.0, sl_freq_hz=(0.5, 1.0))
    rep = MP.manipulation_report(g.simulate_family_c, (0, 1), weak, ("g_b", "c_int"))
    assert not rep["passed"].any()


def test_checks_read_oracle_channels_only(monkeypatch):
    """The signatures never call an MPC estimator."""
    import sys

    monkeypatch.setitem(sys.modules, "impact_pipeline.mpc_metrics", None)
    s = g.simulate_family_c(None, CHECK_CFG, seed=0)
    sig = MP.oracle_signatures(s)
    assert set(sig) == set(MP.SIGNATURES)
    assert all(np.isfinite(v) for v in sig.values())
    assert isinstance(
        MP.manipulation_check(g.simulate_family_a, 0, CHECK_CFG, ("K",)), pd.DataFrame
    )


# --------------------------------------------------------------------------
# realisation checks
# --------------------------------------------------------------------------
SMALL = g.AgentConfig(n_trials=16, n_reafference_pairs=6)


def fake_system(ts=None, modules=None, oracle=None, meta=None, dt=0.05):
    modules = modules or {"S": [0], "W": [1], "M": [2]}
    n = sum(len(v) for v in modules.values())
    ts = np.zeros((n, 10)) if ts is None else np.asarray(ts, dtype=float)
    base = {"dt": dt, "modules": modules, "module_order": list(modules),
            "workspace_nodes": list(modules.get("W", [])), "n_nodes": n,
            "dynamics": "rate", "knobs": {"g_b": 1.0}, "config": {"w_ign": 1.5}}
    base.update(meta or {})
    events = pd.DataFrame({"onset": [0.0, 0.1], "duration": [0.1, 0.1],
                           "trial_type": ["goal_cue", "response"], "value": [1, 0]})
    return g.BenchSystem(ts=ts, events=events, meta=base, oracle=dict(oracle or {}))


def test_versions_and_reused_thresholds():
    assert MP.MANIPULATION_CHECK_VERSION == "mpc-bench-manipulation/1.0.0"
    assert MP.CONTEXT_FLOOR == MP.NOMINAL_FLOOR["K"] == 0.05
    assert MP.IGNITION_OCCUPANCY_MAX == 0.05 and MP.USABLE_PASS_SHARE == 0.9
    cat = CAT.load_catalogue()
    assert set(MP.CHECKS_BY_SYSTEM) <= set(CAT.system_ids(cat))
    assert set(MP.CHECKS_BY_SYSTEM) == {
        "N_modules_disconnected", "N_uncoupled", "W_PDI_no_multistability",
        "ADV_NAS_staggered_driver", "ADV_NAS_staggered_tau10", "ADV_NAS_staggered_sat"}


# --------------------------------------------------------------------------
# signatures on fixtures
# --------------------------------------------------------------------------
def test_hub_periphery_paths_follow_directed_edges():
    adj = np.zeros((3, 3))  # adjacency[source, target]; S=0, W=1, M=2
    s = fake_system(oracle={"unit_adjacency": adj})
    p = MP.hub_periphery_paths(s)
    assert p == {"hub_to_periphery": False, "periphery_to_hub": False,
                 "connected_pairs": 0, "max_abs_direct_coupling": 0.0}
    adj[1, 2] = 0.3  # W -> M
    p = MP.hub_periphery_paths(s)
    assert p["hub_to_periphery"] and not p["periphery_to_hub"]
    assert p["max_abs_direct_coupling"] == 0.3
    adj[:] = 0
    adj[0, 2] = adj[2, 1] = 1.0  # S -> M -> W: an indirect path into the hub
    p = MP.hub_periphery_paths(s)
    assert p["periphery_to_hub"] and not p["hub_to_periphery"]
    assert p["connected_pairs"] == 2
    adj[:] = 0
    adj[0, 2] = adj[2, 0] = 1.0  # periphery loop without the hub
    assert MP.hub_periphery_paths(s)["connected_pairs"] == 0


def test_ignition_occupancy_counts_the_amplification_only():
    gate = np.r_[np.full(40, 0.9), np.full(60, 0.1)]
    s = fake_system(oracle={"ignition_gate": gate})
    occ = MP.ignition_occupancy(s)
    assert occ["ignition_occupancy"] == pytest.approx(0.4)
    assert occ["hub_above_threshold_share"] == pytest.approx(0.4)
    s.meta["config"] = {"w_ign": 0.0}
    occ = MP.ignition_occupancy(s)
    assert occ["ignition_occupancy"] == 0.0
    assert occ["hub_above_threshold_share"] == pytest.approx(0.4)
    s.meta["config"] = {"w_ign": 1.5}
    s.meta["knobs"] = {"g_b": 0.0}
    assert MP.ignition_occupancy(s)["ignition_occupancy"] == 0.0
    s.meta.update({"knobs": {"g_b": 0.5}, "dynamics": "stuart_landau",
                   "config": {"w_ign": 0.0, "sl_ignition_gain": 2.5}})
    s.oracle["ignition_gate"] = 0.5 * gate
    assert MP.ignition_occupancy(s)["ignition_occupancy"] == pytest.approx(0.4)


def test_module_context_information_is_the_between_label_share():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 100)
    labels = np.tile(labels, 50)
    pattern = rng.standard_normal((4, 3))
    ts = pattern[labels].T  # all three units follow the label exactly
    s = fake_system(ts=ts, oracle={"context_state": labels},
                    modules={"S": [0, 1], "W": [2]})
    ci = MP.module_context_information(s)
    assert set(ci) == {"S", "W"} and min(ci.values()) > 0.9
    s.ts = rng.standard_normal(ts.shape)
    assert max(MP.module_context_information(s).values()) < 0.02
    assert MP.module_context_information(s, labels=np.zeros_like(labels)) == {
        "S": 0.0, "W": 0.0}


@pytest.mark.parametrize("simulate", [g.simulate_family_a, g.simulate_family_c])
def test_module_context_information_on_one_module_is_context_information(simulate):
    s = simulate(None, SMALL, 4)
    one = copy.deepcopy(s)
    one.meta["modules"] = {"ALL": list(range(s.n_nodes))}
    one.meta["module_order"] = ["ALL"]
    assert MP.module_context_information(one)["ALL"] == pytest.approx(
        MP.context_information(s), rel=1e-12)
    ci = MP.module_context_information(s)
    assert set(ci) == set(s.meta["module_order"])
    assert all(0.0 <= v <= 1.0 for v in ci.values())


def test_context_runs_without_the_last_run():
    s = fake_system(oracle={"context_state": np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])},
                    dt=0.5)
    assert MP.context_runs_sec(s).tolist() == [1.5, 1.0]


def test_hashes_on_fixtures():
    adj = np.zeros((3, 3))
    a = fake_system(oracle={"unit_adjacency": adj, "context_state": np.zeros(10)})
    b = copy.deepcopy(a)
    assert MP.structural_hash(a) == MP.structural_hash(b)
    assert MP.schedule_hash(a) == MP.schedule_hash(b)
    b.oracle["unit_adjacency"][0, 1] = 1e-12
    assert MP.structural_hash(a) != MP.structural_hash(b)
    b = copy.deepcopy(a)
    b.events.loc[1, "value"] = 1  # a response (choice): not part of the schedule
    assert MP.schedule_hash(a) == MP.schedule_hash(b)
    b.events.loc[0, "value"] = 0  # a goal cue: part of the schedule
    assert MP.schedule_hash(a) != MP.schedule_hash(b)
    b = copy.deepcopy(a)
    b.oracle["context_state"][3] = 1
    assert MP.schedule_hash(a) != MP.schedule_hash(b)
    assert MP.structural_hash(a) == MP.structural_hash(b)


# --------------------------------------------------------------------------
# gated checks on the real systems
# --------------------------------------------------------------------------
def test_realisation_report_covers_every_realised_system_and_passes():
    df = MP.realisation_report([0])
    got = {(r.system_id, r.family, r.variant or "", r.check) for r in df.itertuples()}
    want = {("N_modules_disconnected", "A", "", "no_hub_periphery_path"),
            ("N_modules_disconnected", "C", "", "no_hub_periphery_path"),
            ("N_uncoupled", "C", "", "no_coupling"),
            ("W_PDI_no_multistability", "A", "", "no_ignition")}
    for sid, vs in CAT.STAGGERED_SYSTEM_VARIANTS.items():
        for v in vs:
            want |= {(sid, "A", v, "driver_reaches_every_module"),
                     (sid, "A", v, "stated_time_constants")}
    assert got == want
    assert df["passed"].all() and df["gate"].all()
    assert set(df["check_version"]) == {MP.MANIPULATION_CHECK_VERSION}
    summ = MP.summarise_realisation(df)
    assert summ["usable"].all() and (summ["n"] == 1).all()
    only = MP.realisation_report([1, 2], system_ids=["N_uncoupled"], families=("A",))
    assert only.empty  # N_uncoupled is not defined in family A


def test_checks_fail_on_systems_that_do_not_realise_the_mechanism():
    pc = CAT.build_system("PC_nominal", 0, "A")
    (row,) = MP.check_system(pc, "N_modules_disconnected", 0)
    assert not row["passed"] and row["value"] > 0
    (row,) = MP.check_system(pc, "N_uncoupled", 0)
    assert not row["passed"]
    (row,) = MP.check_system(pc, "W_PDI_no_multistability", 0)
    assert not row["passed"] and row["value"] >= 0.05
    st = CAT.build_system("ADV_NAS_staggered_driver", 0, variant="uniform")
    rows = {r["check"]: r for r in MP.check_system(st, "ADV_NAS_staggered_driver", 0)}
    assert rows["stated_time_constants"]["passed"]
    bad = copy.deepcopy(st)
    bad.oracle["module_tau_sec"]["M"] = 0.8  # stated, not realised
    rows = {r["check"]: r for r in MP.check_system(bad, "ADV_NAS_staggered_driver", 0)}
    assert not rows["stated_time_constants"]["passed"]
    bad = copy.deepcopy(st)
    bad.oracle["transform"] = "tanh"
    bad.oracle["saturation_gain"] = 2.0
    rows = {r["check"]: r for r in MP.check_system(bad, "ADV_NAS_staggered_driver", 0)}
    assert not rows["stated_time_constants"]["passed"]
    assert rows["stated_time_constants"]["details"]["transform_max_error"] > 0
    flat = copy.deepcopy(st)  # a driver that reaches no module
    flat.ts = np.random.default_rng(1).standard_normal(st.ts.shape)
    rows = {r["check"]: r for r in MP.check_system(flat, "ADV_NAS_staggered_driver", 0)}
    assert not rows["driver_reaches_every_module"]["passed"]
    with pytest.raises(ValueError, match="no realisation check"):
        MP.realisation_check("PC_nominal", 0)


def test_usability_is_36_of_40():
    rows = []
    for passes, sid in ((36, "X"), (35, "Y")):
        for i in range(40):
            rows.append({"system_id": sid, "variant": None, "family": "A",
                         "check": "c", "value": 0.0, "passed": i < passes})
    summ = MP.summarise_realisation(pd.DataFrame(rows)).set_index("system_id")
    assert summ.loc["X", "required"] == 36 and summ.loc["X", "usable"]
    assert summ.loc["Y", "passes"] == 35 and not summ.loc["Y", "usable"]
    assert summ.loc["X", "near_threshold"] and summ.loc["Y", "near_threshold"]
    small = MP.summarise_realisation(pd.DataFrame(rows[:3]))
    assert small["required"].iloc[0] == 3 and not small["near_threshold"].iloc[0]


def test_oracle_checks_collect_switches_and_realised_systems():
    out = MP.oracle_checks([0], config=SMALL.replace(n_trials=40))
    sw = out["switches"]
    assert set(zip(sw["family"], sw["switch"])) == {
        (f, s) for f in ("A", "C") for s in MP.SWITCHES}
    assert set(sw["check_version"]) == {MP.MANIPULATION_CHECK_VERSION}
    summ = out["summary"]
    assert set(summ["kind"]) == {"switch", "realisation"}
    assert (summ.loc[summ.kind == "switch", "n"] == 1).all()
    new = summ[summ.kind == "realisation"]
    assert set(new["system_id"]) == set(MP.CHECKS_BY_SYSTEM) | {"PC_nominal"}
    assert out["complete"] and out["seeds"] == [0]
    assert out["split"] == "development"
    # 16 trials hold no reversal: the reversal-tracking signature is 0 at the
    # nominal dose, its relative change undefined, and the data incomplete
    only_a = MP.oracle_checks([1], families=("A",), config=SMALL, reported=False)
    assert set(only_a["switches"]["family"]) == {"A"}
    assert "N_uncoupled" not in set(only_a["realisation"]["system_id"])
    assert not only_a["complete"]
    assert "pc_half" not in only_a and "twins" not in only_a


def test_oracle_checks_gate_the_slow_context_preset_and_report_the_rest():
    """The slow-context check is a gated realisation row of family A (after
    the new systems, the switch and system rows unchanged); PC_half and the
    twins are reported, with gate False; more workers give the same tables."""
    cfg = SMALL.replace(n_trials=40)
    out = MP.oracle_checks([0, 1], config=cfg)
    real = out["realisation"]
    ref = MP.realisation_report([0, 1], config=cfg)
    pd.testing.assert_frame_equal(real.iloc[: len(ref)].reset_index(drop=True), ref)
    slow = real.iloc[len(ref):]
    assert list(slow.columns) == list(MP.REALISATION_COLUMNS)
    assert set(slow["family"]) == {"A"} and list(slow["seed"]) == [0, 1]
    assert set(slow["system_id"]) == {"PC_nominal"}
    assert set(slow["variant"]) == {"slow_context_bold"}
    assert set(slow["check"]) == {"slow_context_dwell"}
    assert slow["gate"].all() and slow["passed"].all()
    assert (slow["value"] >= MP.SLOW_CONTEXT_MIN_DWELL_SEC - 0.05).all()
    summ = out["summary"].set_index("check")
    assert summ.loc["slow_context_dwell", "usable"]
    assert summ.loc["slow_context_dwell", "family"] == "A"
    sw = pd.concat([MP.manipulation_report(f, [0, 1], cfg)
                    for f in (g.simulate_family_a, g.simulate_family_c)],
                   ignore_index=True)
    pd.testing.assert_frame_equal(out["switches"], sw)
    # reported
    pc = out["pc_half"]
    assert len(pc) == 2 * 2 * len(MP.SWITCHES)
    ps = out["pc_half_summary"]
    assert len(ps) == 2 * len(MP.SWITCHES) and not ps["gate"].any()
    assert (ps["passed"] == ps["between"]).all()
    assert set(ps["system_id"]) == {"PC_half"}
    assert list(ps["variant"]) == list(ps["switch"])
    tw = out["twins"]
    assert not tw["gate"].any() and set(tw["check"]) == {MP.TWIN_CHECK}
    assert set(zip(tw["family"], tw["system_id"])) == {
        (f, s) for f in ("A", "C") for s in MP.twin_systems(f)}
    assert sorted(set(tw["replicate"])) == list(MP.TWIN_REPLICATES)
    assert MP.TWIN_REPLICATES == (1, 2, 3, 4, 5, 6)
    assert tw["passed"].all()
    one = MP.oracle_checks([0], families=("A",), config=cfg)
    two = MP.oracle_checks([0], families=("A",), config=cfg, workers=2)
    for key in ("switches", "realisation", "summary", "pc_half", "pc_half_summary",
                "twins"):
        pd.testing.assert_frame_equal(one[key], two[key])


def test_twin_systems_are_those_of_the_twin_designs():
    from impact_pipeline.bench.designs import twins as T

    assert tuple(MP.twin_systems("A")) == T.twin_classes("A")
    assert tuple(MP.twin_systems("C")) == T.twin_classes("C")
    assert MP.TWIN_REPLICATES == tuple(T.REPLICATES["A"][S.CONFIRMATORY])
    assert MP.TWIN_REPLICATES == tuple(T.REPLICATES["C"][S.CONFIRMATORY])


def test_oracle_checks_keep_the_seed_policy():
    """Seeds outside the policy (the unused block 10000-19999, the
    unassigned block) are refused before any run."""
    for seeds in ([10000], [0, 19999], [1000]):
        with pytest.raises(S.SeedPolicyError):
            MP.oracle_checks(seeds, config=SMALL)


def test_hashes_of_object_arrays_are_stable():
    """An object array is hashed by value, not by its pointers."""
    def system():
        vals = [[float(str(0.25 * (i + j))) for j in range(3)] for i in range(3)]
        return fake_system(oracle={"unit_adjacency": np.array(vals, dtype=object),
                                   "context_state": np.array(
                                       [str(k) for k in range(10)], dtype=object)})

    a, b = system(), system()
    assert MP.structural_hash(a) == MP.structural_hash(b)
    assert MP.schedule_hash(a) == MP.schedule_hash(b)
    b.oracle["unit_adjacency"][0, 0] = 9.0
    assert MP.structural_hash(a) != MP.structural_hash(b)


# --------------------------------------------------------------------------
# reported checks
# --------------------------------------------------------------------------
def test_pc_half_rows_and_summary():
    df = MP.pc_half_check(0, "A", config=SMALL)
    assert list(df["switch"]) == list(MP.SWITCHES)
    assert set(df["signature"]) == set(MP.SIGNATURES)
    assert np.isfinite(df[["nominal", "half", "off"]].to_numpy()).all()
    # the nominal signatures are those of the switch check on the same seed
    ref = MP.manipulation_check(g.simulate_family_a, 0, SMALL)
    assert np.allclose(df["nominal"].to_numpy(), ref["nominal"].to_numpy())
    assert np.allclose(df["off"].to_numpy(), ref["off"].to_numpy())
    synth = pd.DataFrame([
        {"family": "A", "seed": s, "switch": "eta", "signature": "x",
         "off": 0.0, "half": h, "nominal": 1.0}
        for s, h in enumerate([0.4, 0.5, 0.6])
    ] + [
        {"family": "A", "seed": s, "switch": "K", "signature": "y",
         "off": 0.0, "half": h, "nominal": 1.0}
        for s, h in enumerate([1.2, 1.3, 0.2])
    ])
    summ = MP.pc_half_summary(synth).set_index("switch")
    assert summ.loc["eta", "between"] and not summ.loc["K", "between"]
    assert summ.loc["eta", "median_half"] == 0.5 and not summ["gate"].any()
    # each summary row is a reported check row
    assert summ.loc["eta", "passed"] and not summ.loc["K", "passed"]
    assert set(summ["system_id"]) == {"PC_half"}
    assert set(summ["check"]) == {"between_off_and_nominal"}
    assert summ.loc["K", "variant"] == "K"


def test_twin_check():
    df = MP.twin_check("PC_nominal", 2, (1, 3), "A", config=SMALL)
    assert list(df["replicate"]) == [1, 3]
    assert df["passed"].all() and df["same_structure"].all()
    assert df["new_schedule"].all() and df["new_series"].all()
    assert df["r0_equals_reference"].isna().all()
    assert set(df["check"]) == {"twin_hashes"} and not df["gate"].any()
    ref = g.simulate_family_a(None, SMALL, 2)
    ok = MP.twin_check("PC_nominal", 2, (1,), "A", config=SMALL, reference=ref)
    assert ok["r0_equals_reference"].iloc[0] and ok["passed"].iloc[0]
    other = g.simulate_family_a(None, SMALL, 3)
    bad = MP.twin_check("PC_nominal", 2, (1,), "A", config=SMALL, reference=other)
    assert not bad["r0_equals_reference"].iloc[0] and not bad["passed"].iloc[0]
    c = MP.twin_check("W_IIM_feedforward", 2, (1,), "C", config=SMALL)
    assert c["passed"].all()


def test_slow_context_check_fails_without_slow_contexts():
    cat = copy.deepcopy(CAT.load_catalogue())
    cat["config_presets"]["slow_context_bold"]["config"]["ctx_dwell"] = [1.0, 3.0]
    row = MP.slow_context_check(1, catalogue=cat).iloc[0]
    assert row["config_in_force"] and row["n_runs"] > 0
    assert row["shortest_run_sec"] < MP.SLOW_CONTEXT_MIN_DWELL_SEC
    assert not row["passed"]
    # a gated realisation row of the preset (check id
    # PC_nominal:slow_context_bold/slow_context_dwell)
    assert row["gate"] and row["variant"] == "slow_context_bold"
    assert row["check"] == "slow_context_dwell" and row["system_id"] == "PC_nominal"
    assert row["value"] == row["shortest_run_sec"]
    assert row["threshold"] == MP.SLOW_CONTEXT_MIN_DWELL_SEC
    assert row["details"]["n_runs"] == row["n_runs"]
    assert MP.SLOW_CONTEXT_CHECK_ID == "PC_nominal:slow_context_bold/slow_context_dwell"
    # the nominal preset is realised; without context changes the check fails
    ok = MP.slow_context_check(1).iloc[0]
    assert ok["passed"] and ok["value"] >= MP.SLOW_CONTEXT_MIN_DWELL_SEC - 0.05
    cat["config_presets"]["slow_context_bold"]["config"]["ctx_dwell"] = [1e6, 2e6]
    none = MP.slow_context_check(1, catalogue=cat).iloc[0]
    assert none["n_runs"] == 0 and none["value"] == 0.0 and not none["passed"]
    # the preset's timing must be in force
    cat = copy.deepcopy(CAT.load_catalogue())
    row = MP.slow_context_check(1, catalogue=cat, config={"trial_sec": 6.0}).iloc[0]
    assert row["passed"]  # the preset overrides the caller's trial length


def test_real_hashes_of_twins_seeds_and_knobs():
    a0 = g.simulate_family_a(None, SMALL, 7)
    a1 = g.simulate_family_a(None, SMALL, 7, replicate=1)
    b0 = g.simulate_family_a(None, SMALL, 8)
    assert MP.structural_hash(a0) == MP.structural_hash(a1) != MP.structural_hash(b0)
    assert len({MP.schedule_hash(x) for x in (a0, a1, b0)}) == 3
    ff = g.simulate_family_a(g.Knobs(c_int=0.0), SMALL, 7)
    assert MP.schedule_hash(ff) == MP.schedule_hash(a0)
    assert MP.structural_hash(ff) != MP.structural_hash(a0)
    st = CAT.build_system("ADV_NAS_staggered_driver", 7, config=SMALL)
    st2 = CAT.build_system("ADV_NAS_staggered_driver", 7, config=SMALL)
    assert MP.structural_hash(st) == MP.structural_hash(st2)
    assert MP.schedule_hash(st) == MP.schedule_hash(st2)
    nul = CAT.build_system("N_ar1", 7, config=SMALL)
    assert isinstance(MP.structural_hash(nul), str) and len(MP.schedule_hash(nul)) == 64
