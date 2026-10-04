# -*- coding: utf-8 -*-
"""
Realisation checks of the v2 systems (mpc-bench-manipulation/1.1.0): the
oracle signatures on hand-built fixtures, the gated checks on the real
systems (and their failure on systems that do not realise the mechanism),
the usability rule (36 of 40 seeds), the reported PC_half and twin checks,
and the structural and schedule hashes.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.bench import adversarial_v2 as A2
from impact_pipeline.bench import generators as g
from impact_pipeline.bench import manipulation as M
from impact_pipeline.bench import manipulation_v2 as MV
from impact_pipeline.v2 import seeds as S

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
    assert MV.MANIPULATION_CHECK_VERSION_V2 == "mpc-bench-manipulation/1.1.0"
    assert M.MANIPULATION_CHECK_VERSION == "mpc-bench-manipulation/1.0.0"
    assert MV.BASE_CHECK_VERSION == M.MANIPULATION_CHECK_VERSION
    assert MV.CONTEXT_FLOOR == M.NOMINAL_FLOOR["K"] == 0.05
    assert MV.IGNITION_OCCUPANCY_MAX == 0.05 and MV.USABLE_PASS_SHARE == 0.9
    cat = A2.load_catalogue_v2()
    assert set(MV.CHECKS_BY_SYSTEM) <= set(A2.system_ids(cat))
    assert set(MV.CHECKS_BY_SYSTEM) == {
        "N_modules_disconnected", "N_uncoupled", "W_PDI_no_multistability",
        "ADV_NAS_staggered_driver", "ADV_NAS_staggered_tau10", "ADV_NAS_staggered_sat"}


# --------------------------------------------------------------------------
# signatures on fixtures
# --------------------------------------------------------------------------
def test_hub_periphery_paths_follow_directed_edges():
    adj = np.zeros((3, 3))  # adjacency[source, target]; S=0, W=1, M=2
    s = fake_system(oracle={"unit_adjacency": adj})
    p = MV.hub_periphery_paths(s)
    assert p == {"hub_to_periphery": False, "periphery_to_hub": False,
                 "connected_pairs": 0, "max_abs_direct_coupling": 0.0}
    adj[1, 2] = 0.3  # W -> M
    p = MV.hub_periphery_paths(s)
    assert p["hub_to_periphery"] and not p["periphery_to_hub"]
    assert p["max_abs_direct_coupling"] == 0.3
    adj[:] = 0
    adj[0, 2] = adj[2, 1] = 1.0  # S -> M -> W: an indirect path into the hub
    p = MV.hub_periphery_paths(s)
    assert p["periphery_to_hub"] and not p["hub_to_periphery"]
    assert p["connected_pairs"] == 2
    adj[:] = 0
    adj[0, 2] = adj[2, 0] = 1.0  # periphery loop without the hub
    assert MV.hub_periphery_paths(s)["connected_pairs"] == 0


def test_ignition_occupancy_counts_the_amplification_only():
    gate = np.r_[np.full(40, 0.9), np.full(60, 0.1)]
    s = fake_system(oracle={"ignition_gate": gate})
    occ = MV.ignition_occupancy(s)
    assert occ["ignition_occupancy"] == pytest.approx(0.4)
    assert occ["hub_above_threshold_share"] == pytest.approx(0.4)
    s.meta["config"] = {"w_ign": 0.0}
    occ = MV.ignition_occupancy(s)
    assert occ["ignition_occupancy"] == 0.0
    assert occ["hub_above_threshold_share"] == pytest.approx(0.4)
    s.meta["config"] = {"w_ign": 1.5}
    s.meta["knobs"] = {"g_b": 0.0}
    assert MV.ignition_occupancy(s)["ignition_occupancy"] == 0.0
    s.meta.update({"knobs": {"g_b": 0.5}, "dynamics": "stuart_landau",
                   "config": {"w_ign": 0.0, "sl_ignition_gain": 2.5}})
    s.oracle["ignition_gate"] = 0.5 * gate
    assert MV.ignition_occupancy(s)["ignition_occupancy"] == pytest.approx(0.4)


def test_module_context_information_is_the_between_label_share():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 100)
    labels = np.tile(labels, 50)
    pattern = rng.standard_normal((4, 3))
    ts = pattern[labels].T  # all three units follow the label exactly
    s = fake_system(ts=ts, oracle={"context_state": labels},
                    modules={"S": [0, 1], "W": [2]})
    ci = MV.module_context_information(s)
    assert set(ci) == {"S", "W"} and min(ci.values()) > 0.9
    s.ts = rng.standard_normal(ts.shape)
    assert max(MV.module_context_information(s).values()) < 0.02
    assert MV.module_context_information(s, labels=np.zeros_like(labels)) == {
        "S": 0.0, "W": 0.0}


@pytest.mark.parametrize("simulate", [g.simulate_family_a, g.simulate_family_c])
def test_module_context_information_on_one_module_is_the_v1_signature(simulate):
    s = simulate(None, SMALL, 4)
    one = copy.deepcopy(s)
    one.meta["modules"] = {"ALL": list(range(s.n_nodes))}
    one.meta["module_order"] = ["ALL"]
    assert MV.module_context_information(one)["ALL"] == pytest.approx(
        M.context_information(s), rel=1e-12)
    ci = MV.module_context_information(s)
    assert set(ci) == set(s.meta["module_order"])
    assert all(0.0 <= v <= 1.0 for v in ci.values())


def test_context_runs_without_the_last_run():
    s = fake_system(oracle={"context_state": np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])},
                    dt=0.5)
    assert MV.context_runs_sec(s).tolist() == [1.5, 1.0]


def test_hashes_on_fixtures():
    adj = np.zeros((3, 3))
    a = fake_system(oracle={"unit_adjacency": adj, "context_state": np.zeros(10)})
    b = copy.deepcopy(a)
    assert MV.structural_hash(a) == MV.structural_hash(b)
    assert MV.schedule_hash(a) == MV.schedule_hash(b)
    b.oracle["unit_adjacency"][0, 1] = 1e-12
    assert MV.structural_hash(a) != MV.structural_hash(b)
    b = copy.deepcopy(a)
    b.events.loc[1, "value"] = 1  # a response (choice): not part of the schedule
    assert MV.schedule_hash(a) == MV.schedule_hash(b)
    b.events.loc[0, "value"] = 0  # a goal cue: part of the schedule
    assert MV.schedule_hash(a) != MV.schedule_hash(b)
    b = copy.deepcopy(a)
    b.oracle["context_state"][3] = 1
    assert MV.schedule_hash(a) != MV.schedule_hash(b)
    assert MV.structural_hash(a) == MV.structural_hash(b)


# --------------------------------------------------------------------------
# gated checks on the real systems
# --------------------------------------------------------------------------
def test_realisation_report_covers_every_new_system_and_passes():
    df = MV.realisation_report([0])
    got = {(r.system_id, r.family, r.variant or "", r.check) for r in df.itertuples()}
    want = {("N_modules_disconnected", "A", "", "no_hub_periphery_path"),
            ("N_modules_disconnected", "C", "", "no_hub_periphery_path"),
            ("N_uncoupled", "C", "", "no_coupling"),
            ("W_PDI_no_multistability", "A", "", "no_ignition")}
    for sid, vs in A2.STAGGERED_SYSTEM_VARIANTS.items():
        for v in vs:
            want |= {(sid, "A", v, "driver_reaches_every_module"),
                     (sid, "A", v, "stated_time_constants")}
    assert got == want
    assert df["passed"].all() and df["gate"].all()
    assert set(df["check_version"]) == {MV.MANIPULATION_CHECK_VERSION_V2}
    summ = MV.summarise_realisation(df)
    assert summ["usable"].all() and (summ["n"] == 1).all()
    only = MV.realisation_report([1, 2], system_ids=["N_uncoupled"], families=("A",))
    assert only.empty  # N_uncoupled is not defined in family A


def test_checks_fail_on_systems_that_do_not_realise_the_mechanism():
    pc = A2.build_system("PC_nominal", 0, "A")
    (row,) = MV.check_system(pc, "N_modules_disconnected", 0)
    assert not row["passed"] and row["value"] > 0
    (row,) = MV.check_system(pc, "N_uncoupled", 0)
    assert not row["passed"]
    (row,) = MV.check_system(pc, "W_PDI_no_multistability", 0)
    assert not row["passed"] and row["value"] >= 0.05
    st = A2.build_system("ADV_NAS_staggered_driver", 0, variant="uniform")
    rows = {r["check"]: r for r in MV.check_system(st, "ADV_NAS_staggered_driver", 0)}
    assert rows["stated_time_constants"]["passed"]
    bad = copy.deepcopy(st)
    bad.oracle["module_tau_sec"]["M"] = 0.8  # stated, not realised
    rows = {r["check"]: r for r in MV.check_system(bad, "ADV_NAS_staggered_driver", 0)}
    assert not rows["stated_time_constants"]["passed"]
    bad = copy.deepcopy(st)
    bad.oracle["transform"] = "tanh"
    bad.oracle["saturation_gain"] = 2.0
    rows = {r["check"]: r for r in MV.check_system(bad, "ADV_NAS_staggered_driver", 0)}
    assert not rows["stated_time_constants"]["passed"]
    assert rows["stated_time_constants"]["details"]["transform_max_error"] > 0
    flat = copy.deepcopy(st)  # a driver that reaches no module
    flat.ts = np.random.default_rng(1).standard_normal(st.ts.shape)
    rows = {r["check"]: r for r in MV.check_system(flat, "ADV_NAS_staggered_driver", 0)}
    assert not rows["driver_reaches_every_module"]["passed"]
    with pytest.raises(ValueError, match="no realisation check"):
        MV.realisation_check("PC_nominal", 0)


def test_usability_is_36_of_40():
    rows = []
    for passes, sid in ((36, "X"), (35, "Y")):
        for i in range(40):
            rows.append({"system_id": sid, "variant": None, "family": "A",
                         "check": "c", "value": 0.0, "passed": i < passes})
    summ = MV.summarise_realisation(pd.DataFrame(rows)).set_index("system_id")
    assert summ.loc["X", "required"] == 36 and summ.loc["X", "usable"]
    assert summ.loc["Y", "passes"] == 35 and not summ.loc["Y", "usable"]
    assert summ.loc["X", "near_threshold"] and summ.loc["Y", "near_threshold"]
    small = MV.summarise_realisation(pd.DataFrame(rows[:3]))
    assert small["required"].iloc[0] == 3 and not small["near_threshold"].iloc[0]


def test_prerequisite_m_collects_switches_and_new_systems():
    out = MV.prerequisite_m([0], config=SMALL.replace(n_trials=40))
    sw = out["switches"]
    assert set(zip(sw["family"], sw["switch"])) == {
        (f, s) for f in ("A", "C") for s in M.SWITCHES}
    assert set(sw["check_version"]) == {M.MANIPULATION_CHECK_VERSION}
    summ = out["summary"]
    assert set(summ["kind"]) == {"switch", "new_system"}
    assert (summ.loc[summ.kind == "switch", "n"] == 1).all()
    new = summ[summ.kind == "new_system"]
    assert set(new["system_id"]) == set(MV.CHECKS_BY_SYSTEM)
    assert out["complete"] and out["seeds"] == [0]
    assert out["split"] == "development"
    # 16 trials hold no reversal: the reversal-tracking signature is 0 at the
    # nominal dose, its relative change undefined, and the data incomplete
    only_a = MV.prerequisite_m([1], families=("A",), config=SMALL)
    assert set(only_a["switches"]["family"]) == {"A"}
    assert "N_uncoupled" not in set(only_a["realisation"]["system_id"])
    assert not only_a["complete"]


def test_prerequisite_m_keeps_the_seed_policy():
    """Seeds outside the v2 policy (the v1 confirmatory block, the
    unassigned block) are refused before any run."""
    for seeds in ([10000], [0, 19999], [1000]):
        with pytest.raises(S.SeedPolicyError):
            MV.prerequisite_m(seeds, config=SMALL)


def test_hashes_of_object_arrays_are_stable():
    """An object array is hashed by value, not by its pointers."""
    def system():
        vals = [[float(str(0.25 * (i + j))) for j in range(3)] for i in range(3)]
        return fake_system(oracle={"unit_adjacency": np.array(vals, dtype=object),
                                   "context_state": np.array(
                                       [str(k) for k in range(10)], dtype=object)})

    a, b = system(), system()
    assert MV.structural_hash(a) == MV.structural_hash(b)
    assert MV.schedule_hash(a) == MV.schedule_hash(b)
    b.oracle["unit_adjacency"][0, 0] = 9.0
    assert MV.structural_hash(a) != MV.structural_hash(b)


# --------------------------------------------------------------------------
# reported checks
# --------------------------------------------------------------------------
def test_pc_half_rows_and_summary():
    df = MV.pc_half_check(0, "A", config=SMALL)
    assert list(df["switch"]) == list(M.SWITCHES)
    assert set(df["signature"]) == set(M.SIGNATURES)
    assert np.isfinite(df[["nominal", "half", "off"]].to_numpy()).all()
    # the nominal signatures are those of the 1.0.0 check on the same seed
    ref = M.manipulation_check(g.simulate_family_a, 0, SMALL)
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
    summ = MV.pc_half_summary(synth).set_index("switch")
    assert summ.loc["eta", "between"] and not summ.loc["K", "between"]
    assert summ.loc["eta", "median_half"] == 0.5 and not summ["gate"].any()


def test_twin_check():
    df = MV.twin_check("PC_nominal", 2, (1, 3), "A", config=SMALL)
    assert list(df["replicate"]) == [1, 3]
    assert df["passed"].all() and df["same_structure"].all()
    assert df["new_schedule"].all() and df["new_series"].all()
    assert df["r0_equals_reference"].isna().all()
    ref = g.simulate_family_a(None, SMALL, 2)
    ok = MV.twin_check("PC_nominal", 2, (1,), "A", config=SMALL, reference=ref)
    assert ok["r0_equals_reference"].iloc[0] and ok["passed"].iloc[0]
    other = g.simulate_family_a(None, SMALL, 3)
    bad = MV.twin_check("PC_nominal", 2, (1,), "A", config=SMALL, reference=other)
    assert not bad["r0_equals_reference"].iloc[0] and not bad["passed"].iloc[0]
    c = MV.twin_check("W_IIM_feedforward", 2, (1,), "C", config=SMALL)
    assert c["passed"].all()


def test_slow_context_check_fails_without_slow_contexts():
    cat = copy.deepcopy(A2.load_catalogue_v2())
    cat["config_presets"]["slow_context_bold"]["config"]["ctx_dwell"] = [1.0, 3.0]
    row = MV.slow_context_check(1, catalogue=cat).iloc[0]
    assert row["config_in_force"] and row["n_runs"] > 0
    assert row["shortest_run_sec"] < MV.SLOW_CONTEXT_MIN_DWELL_SEC
    assert not row["passed"]


def test_real_hashes_of_twins_seeds_and_knobs():
    a0 = g.simulate_family_a(None, SMALL, 7)
    a1 = g.simulate_family_a(None, SMALL, 7, replicate=1)
    b0 = g.simulate_family_a(None, SMALL, 8)
    assert MV.structural_hash(a0) == MV.structural_hash(a1) != MV.structural_hash(b0)
    assert len({MV.schedule_hash(x) for x in (a0, a1, b0)}) == 3
    ff = g.simulate_family_a(g.Knobs(c_int=0.0), SMALL, 7)
    assert MV.schedule_hash(ff) == MV.schedule_hash(a0)
    assert MV.structural_hash(ff) != MV.structural_hash(a0)
    st = A2.build_system("ADV_NAS_staggered_driver", 7, config=SMALL)
    st2 = A2.build_system("ADV_NAS_staggered_driver", 7, config=SMALL)
    assert MV.structural_hash(st) == MV.structural_hash(st2)
    assert MV.schedule_hash(st) == MV.schedule_hash(st2)
    nul = A2.build_system("N_ar1", 7, config=SMALL)
    assert isinstance(MV.structural_hash(nul), str) and len(MV.schedule_hash(nul)) == 64
