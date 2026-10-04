# -*- coding: utf-8 -*-
"""
The v2 system catalogue (``witnesses_v2.yaml``) and its builder: v1 entries
inherited unchanged, the new configuration-only systems (PC_half,
W_PDI_no_multistability, W_NAS_common_input_control, N_modules_disconnected,
C1 N_uncoupled), configuration presets (RAM-only arm, slow-context agents),
the relabelled and pruned adversaries, twin and held-out flags, recorded
inputs, and the whole-brain lesions at G_nom.
"""
import copy

import numpy as np
import pytest

from impact_pipeline.bench import adversarial
from impact_pipeline.bench import adversarial_v2 as A2
from impact_pipeline.bench import generators as g
from impact_pipeline.bench import manipulation_v2 as MV
from impact_pipeline.bench import whole_brain as wb
from impact_pipeline.bench import witnesses

SMALL = g.AgentConfig(n_trials=16, n_reafference_pairs=6)
NEW_WITNESSES = ("PC_half", "W_PDI_no_multistability", "W_NAS_common_input_control",
                 "N_modules_disconnected", "N_uncoupled")


@pytest.fixture(scope="module")
def cat():
    return A2.load_catalogue_v2()


@pytest.fixture(scope="module")
def raw():
    return A2._read_yaml(A2.CATALOGUE_V2_PATH)


def test_the_catalogue_loads_and_validates(cat):
    assert A2.validate_catalogue_v2(cat) == []
    assert cat["schema_version"] == 3
    assert cat["catalogue_version"] == "mpc-bench-systems/2.0.0"
    assert cat["generator_version"] == "mpc-bench-generators/2.0.0"
    assert cat["source"].endswith("witnesses_v2.yaml")
    ids = A2.system_ids(cat)
    assert len(ids) == len(set(ids))


def test_every_v1_witness_is_carried_over_unchanged(cat):
    v1 = {w["id"]: w for w in witnesses.load_witnesses()["witnesses"]}
    v2 = {w["id"]: w for w in cat["witnesses"]}
    assert set(v1) <= set(v2)
    assert set(v2) - set(v1) == set(NEW_WITNESSES)
    for wid, w in v1.items():
        assert v2[wid]["origin"] == "v1"
        for key in ("class", "generator", "knobs", "target", "mechanism_removed",
                    "intended_pattern", "counterexample", "expected_verdict"):
            assert v2[wid][key] == w[key], (wid, key)
        assert v2[wid]["config"] == {}
        assert set(v2[wid]["families"]) <= set(w["families"])
        if "notes" in w:
            assert v2[wid]["notes"] == w["notes"]
    for wid in NEW_WITNESSES:
        assert v2[wid]["origin"] == "v2"


def test_family_c_drops_the_duplicated_family_a_nulls(cat):
    c_ids = set(A2.system_ids(cat, kind="witness", family="C"))
    assert {"N_independent_noise", "N_ar1", "O_hypersynchronous"}.isdisjoint(c_ids)
    assert {"N_uncoupled", "N_modules_disconnected"} <= c_ids
    assert "N_uncoupled" not in A2.system_ids(cat, family="A")
    assert "W_PDI_no_multistability" not in c_ids
    for sid, fam in (("N_uncoupled", "A"), ("N_ar1", "C"),
                     ("W_PDI_no_multistability", "C")):
        with pytest.raises(ValueError, match="not defined for family"):
            A2.build_system(sid, 0, fam, config=SMALL, catalogue=cat)


# --------------------------------------------------------------------------
# new configuration-only systems
# --------------------------------------------------------------------------
def test_pc_half_knobs_are_half_the_nominal_dose(cat):
    e = A2.get_entry("PC_half", cat)
    half = g.knobs_from_dict(e["knobs"])
    nom = g.NOMINAL_KNOBS
    for k in ("eta", "g_b", "c_int", "e"):
        assert getattr(half, k) == pytest.approx(0.5 * getattr(nom, k)), k
    assert half.K == nom.K // 2 == 3
    assert half.k_gain == nom.k_gain and half.ff_only is False
    assert list(half.bits()) == [1, 1, 1, 1, 1]
    s = A2.build_system("PC_half", 1, "A", config=SMALL, catalogue=cat)
    assert s.meta["knobs"] == half.to_dict()
    assert s.oracle["witness_intended_pattern"] == [1, 1, 1, 1, 1]
    # paired with PC_nominal: same network draws and schedule (K maps the
    # same context draws to three patterns instead of six)
    pc = A2.build_system("PC_nominal", 1, "A", config=SMALL, catalogue=cat)
    assert np.array_equal(s.events["onset"], pc.events["onset"])
    assert np.array_equal(s.oracle["good_arm"], pc.oracle["good_arm"])
    assert np.array_equal(s.oracle["context_state"], pc.oracle["context_state"] // 2)
    assert np.array_equal(np.abs(s.oracle["unit_adjacency"]) > 0,
                          np.abs(pc.oracle["unit_adjacency"]) > 0)


@pytest.mark.parametrize("family", ["A", "C"])
def test_modules_disconnected_has_no_hub_periphery_path(cat, family):
    e = A2.get_entry("N_modules_disconnected", cat)
    assert e["knobs"] == {"g_b": 0.0, "c_int": 0.0} and e["config"] == {"w_ff": 0.0}
    for seed in (0, 1):
        s = A2.build_system("N_modules_disconnected", seed, family, config=SMALL,
                            catalogue=cat)
        p = MV.hub_periphery_paths(s)
        assert not p["hub_to_periphery"] and not p["periphery_to_hub"]
        assert p["connected_pairs"] == 0 and p["max_abs_direct_coupling"] == 0.0
        assert s.oracle["module_adjacency"].sum() == 0
        # the module-local recurrence is kept
        adj = s.oracle["unit_adjacency"]
        for idx in s.meta["modules"].values():
            assert np.any(adj[np.ix_(idx, idx)])
        # inputs as nominal: the hub's input (context drive and slow rhythm)
        # is the positive control's, sample for sample
        pc = A2.build_system("PC_nominal", seed, family, config=SMALL, catalogue=cat)
        hub = s.meta["workspace_nodes"]
        assert np.array_equal(s.oracle["inputs"][hub], pc.oracle["inputs"][hub])
        assert np.any(s.oracle["inputs"][hub])
        assert MV.hub_periphery_paths(pc)["hub_to_periphery"]
        assert MV.hub_periphery_paths(pc)["periphery_to_hub"]


def test_no_multistability_has_no_ignition(cat):
    e = A2.get_entry("W_PDI_no_multistability", cat)
    assert e["knobs"] == {"K": 1} and e["config"] == {"w_ign": 0.0, "w_adapt": 0.0}
    assert e["class"] == "single_deficit" and e["target"] == "PDI"
    nominal = []
    for seed in range(4):
        s = A2.build_system("W_PDI_no_multistability", seed, "A", catalogue=cat)
        occ = MV.ignition_occupancy(s)
        assert occ["ignition_occupancy"] < 0.05
        assert s.meta["config"]["w_ign"] == 0.0 and s.meta["knobs"]["g_b"] == 1.0
        # receive-and-return edges kept
        assert MV.hub_periphery_paths(s)["periphery_to_hub"]
        # the check is not vacuous: the nominal agent of the same seed
        # ignites, and the occupancy is the v1 one there
        pc = A2.build_system("PC_nominal", seed, "A", catalogue=cat)
        occ_pc = MV.ignition_occupancy(pc)["ignition_occupancy"]
        assert occ_pc == g.summarise_oracle(pc)["ignition_occupancy"]
        nominal.append(occ_pc)
    assert max(nominal) > 0.05


def test_common_input_control_and_uncoupled(cat):
    e = A2.get_entry("W_NAS_common_input_control", cat)
    assert e["knobs"] == {"K": 1, "g_b": 0.0}
    assert e["intended_pattern"] == [1, 0, 0, 1, 1]
    assert e["class"] == "targeted_null" and e["target"] == "NAS"
    assert "not counted as a new" in e["notes"]
    u = A2.get_entry("N_uncoupled", cat)
    assert g.knobs_from_dict(u["knobs"]) == g.OFF_KNOBS
    s = A2.build_system("N_uncoupled", 2, "C", config=SMALL, catalogue=cat)
    assert s.meta["dynamics"] == "stuart_landau"
    assert not np.any(s.oracle["unit_adjacency"])
    assert s.oracle["intended_bits"] == [0, 0, 0, 0, 0]
    # task on: events and task input pulses are delivered
    assert (s.events.trial_type == "stimulus").sum() == SMALL.n_trials
    stim_input = s.oracle["inputs"][s.meta["modules"]["S"]]
    assert np.any(stim_input != stim_input[:, :1])


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------
def test_configuration_presets(cat):
    assert A2.config_preset("ram_only_160", cat) == {"n_trials": 160}
    slow = A2.config_preset("slow_context_bold", cat)
    assert slow == {"ctx_dwell": [30.0, 60.0], "trial_sec": 12.0, "iti_jitter_sec": 2.0}
    for p in cat["config_presets"].values():
        assert set(p["config"]) <= set(g.AgentConfig.__dataclass_fields__)
    ram = A2.build_system("W_RAM_no_plasticity", 1, "A", preset="ram_only_160",
                          catalogue=cat)
    assert (ram.events.trial_type == "goal_cue").sum() == 160
    assert ram.meta["config"]["n_reafference_pairs"] == 30
    sc = A2.build_system("PC_nominal", 1, "A", preset="slow_context_bold",
                         catalogue=cat)
    runs = MV.context_runs_sec(sc)
    assert runs.min() >= 30.0 - sc.dt
    assert sc.meta["config"]["ctx_dwell"] == [30.0, 60.0]
    assert sc.meta["config"]["trial_sec"] == 12.0
    row = MV.slow_context_check(1, catalogue=cat).iloc[0]
    assert row["passed"] and row["config_in_force"]
    with pytest.raises(KeyError):
        A2.config_preset("nope", cat)
    # an adversary takes the preset too (RAM-only arm)
    sf = A2.build_system("adversarial_scrambled_feedback", 1, preset="ram_only_160",
                         catalogue=cat)
    assert sf.meta["config"]["n_trials"] == 160
    assert sf.meta["config"]["feedback_mode"] == "scrambled"


def test_a_preset_may_not_contradict_a_system(cat):
    bad = copy.deepcopy(cat)
    bad["config_presets"]["clash"] = {"description": "x", "config": {"w_ff": 1.0}}
    with pytest.raises(ValueError, match="contradicts"):
        A2.build_system("N_modules_disconnected", 0, "A", preset="clash", catalogue=bad)
    bad["config_presets"]["agree"] = {"description": "x", "config": {"w_ff": 0.0,
                                                                     "n_trials": 8}}
    s = A2.build_system("N_modules_disconnected", 0, "A", preset="agree", catalogue=bad)
    assert s.meta["config"]["n_trials"] == 8 and s.meta["config"]["w_ff"] == 0.0


def test_configuration_is_applied_in_order(cat):
    """Base configuration, then preset, then the system's own overrides."""
    s = A2.build_system("W_PDI_no_multistability", 0, "A",
                        config={"n_trials": 12, "w_ign": 9.0}, preset="ram_only_160",
                        catalogue=cat)
    assert s.meta["config"]["n_trials"] == 160 and s.meta["config"]["w_ign"] == 0.0
    s = A2.build_system("PC_nominal", 0, "A", config=SMALL, catalogue=cat)
    assert s.meta["config"] == SMALL.to_dict()


# --------------------------------------------------------------------------
# adversaries
# --------------------------------------------------------------------------
def test_v1_adversaries_are_kept_relabelled_or_pruned(cat):
    adv = {a["id"]: a for a in cat["adversaries"]}
    kept = {"adversarial_common_driver", "adversarial_reflex_arc",
            "adversarial_random_label_self_other", "adversarial_scrambled_feedback"}
    assert kept <= set(adv)
    for aid in kept:
        v1 = adversarial.ADVERSARIAL_CATALOGUE[aid[len("adversarial_"):]]
        assert adv[aid]["mechanism"] == v1["mechanism"]
        assert adv[aid]["designed_to_fool"] == v1["designed_to_fool"]
    sf = adv["adversarial_scrambled_feedback"]
    assert sf["intended_pattern"] == [1, 1, 1, 1, 1]
    assert sf["expected_verdict"] == "MPC_CONSISTENT"
    assert sf["v1_intended_pattern"] == [0, 1, 1, 1, 1]
    assert sf["v1_expected_verdict"] == "EXCLUDED"
    assert sf["role"] == "construct_boundary"
    rf = adv["adversarial_reflex_arc"]
    assert rf["intended_pattern"] == [0, 0, 0, 0, 0] and "v1_intended_pattern" not in rf
    s = A2.build_system("adversarial_scrambled_feedback", 3, config=SMALL,
                        catalogue=cat)
    assert s.oracle["adversary_intended_pattern"] == [1, 1, 1, 1, 1]
    assert s.oracle["intended_bits"] == [0, 1, 1, 1, 1]  # the v1 label stays in v1 code
    for aid in ("adversarial_parity_grid", "adversarial_hypersynchrony"):
        assert aid not in adv
        with pytest.raises(KeyError, match="removed"):
            A2.get_entry(aid, cat)
    with pytest.raises(KeyError, match="Unknown"):
        A2.get_entry("nope", cat)


def test_held_out_and_twin_flags(cat):
    held = {e["id"] for e in cat["witnesses"] + cat["adversaries"] if e["held_out"]}
    assert held == {"ADV_NAS_staggered_tau10", "ADV_NAS_staggered_sat"}
    twins = {(e["id"], f) for e in cat["witnesses"] for f in e["twin_families"]}
    assert twins == {
        ("PC_nominal", "A"), ("PC_half", "A"), ("W_NAS_no_workspace", "A"),
        ("W_IIM_feedforward", "A"), ("W_PDI_single_attractor", "A"),
        ("PC_nominal", "C"), ("W_NAS_no_workspace", "C"), ("W_IIM_feedforward", "C"),
    }
    for wid, fam in twins:
        s = A2.build_system(wid, 5, fam, replicate=2, config=SMALL, catalogue=cat)
        assert s.meta["replicate"] == 2 and s.meta["witness_id"] == wid
    for sid in ("N_ar1", "PW_patchwork", "adversarial_reflex_arc"):
        with pytest.raises(ValueError, match="twins"):
            A2.build_system(sid, 5, "A", replicate=1, config=SMALL, catalogue=cat)


@pytest.mark.parametrize("family", ["A", "C"])
def test_every_entry_builds_with_its_recorded_inputs(cat, family):
    for sid in A2.system_ids(cat, family=family):
        e = A2.get_entry(sid, cat)
        for v in (e.get("variants") or [None]):
            s = A2.build_system(sid, 6, family, variant=v, config=SMALL, catalogue=cat)
            label = s.meta.get("witness_id") or s.meta.get("adversary_id")
            assert label == sid
            ch = A2.recorded_channels(s, catalogue=cat)
            assert set(ch) == set(e["recorded_inputs"]), sid
            for c in ch.values():
                assert c["value"].shape[-1] == s.n_time, sid
    pc = A2.build_system("PC_nominal", 6, family, config=SMALL, catalogue=cat)
    pc.oracle.pop("slow_phase")
    with pytest.raises(KeyError, match="slow_phase"):
        A2.recorded_channels(pc, catalogue=cat)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def _with(cat, kind, sid, **changes):
    bad = copy.deepcopy(cat)
    for e in bad[kind]:
        if e["id"] == sid:
            e.update(changes)
    return bad


@pytest.mark.parametrize("kind,sid,changes,message", [
    ("witnesses", "PC_half", {"knobs": {"eta": 0.0}}, "knobs realise"),
    ("witnesses", "W_PDI_no_multistability", {"intended_pattern": [1, 1, 1, 1, 0]},
     "co-atom"),
    ("witnesses", "N_modules_disconnected", {"intended_pattern": [1, 1, 1, 0, 1]},
     "target off"),
    ("witnesses", "N_uncoupled", {"intended_pattern": [1, 0, 0, 0, 0]}, "no mechanism"),
    ("witnesses", "PC_half", {"config": {"nope": 1}}, "bad knobs or config"),
    ("witnesses", "PC_half", {"families": ["B"]}, "families"),
    ("witnesses", "PC_half", {"twin_families": ["C", "A", "X"]}, "twin_families"),
    ("witnesses", "N_ar1", {"twin_families": ["A"]}, "agent generator"),
    ("witnesses", "PC_half", {"recorded_inputs": ["eeg"]}, "recorded inputs"),
    ("witnesses", "PC_half", {"class": "hero"}, "unknown class"),
    ("witnesses", "PC_half", {"expected_verdict": "ATTRIBUTED"}, "expected_verdict"),
    ("witnesses", "PC_half", {"counterexample": {"claim": "x", "formalised_as": "y",
                                                 "references": ["nobody2026"]}},
     "without a DOI"),
    ("adversaries", "ADV_NAS_staggered_tau10", {"held_out": False}, "held_out"),
    ("adversaries", "ADV_NAS_staggered_driver", {"variants": ["uniform"]}, "variants"),
    ("adversaries", "adversarial_reflex_arc", {"variants": ["uniform"]},
     "only the staggered"),
    ("adversaries", "adversarial_reflex_arc", {"families": ["A", "C"]}, "family A"),
    ("adversaries", "ADV_NAS_staggered_sat", {"generator": "slow_drift"},
     "unknown v2 generator"),
    ("adversaries", "adversarial_common_driver", {"generator": "adversarial_x"},
     "unknown v1 generator"),
])
def test_validation_reports_each_error(cat, kind, sid, changes, message):
    errors = A2.validate_catalogue_v2(_with(cat, kind, sid, **changes))
    assert any(message in e and sid in e for e in errors), errors


def test_validation_of_the_catalogue_level(cat):
    bad = copy.deepcopy(cat)
    bad["schema_version"] = 2
    bad["references"]["x"] = "doi.org/1"
    bad["removed"].append({"id": "PC_half", "reason": ""})
    bad["whole_brain_presets"]["lesions_g_nom"]["G"] = 1.0
    bad["witnesses"].append(copy.deepcopy(bad["witnesses"][0]))
    errors = " | ".join(A2.validate_catalogue_v2(bad))
    for msg in ("schema_version", "not a DOI", "removed and as active",
                "removal without a reason", "is not the sweep level", "duplicate id"):
        assert msg in errors, msg
    raw_bad = copy.deepcopy(A2._read_yaml(A2.CATALOGUE_V2_PATH))
    raw_bad["witnesses"].append({"id": "NOT_IN_V1", "origin": "v1"})
    with pytest.raises(ValueError, match="no v1 witness"):
        A2.resolve_catalogue(raw_bad)


def test_v2_references_have_dois(raw):
    refs = raw["references"]
    used = set()
    for w in raw["witnesses"]:
        if w["origin"] == "v2":
            used |= set(w["counterexample"]["references"])
    for a in raw["adversaries"]:
        if a["origin"] == "v2":
            used |= set(a["references"])
    assert used == set(refs)
    assert all(doi.startswith("10.") for doi in refs.values())


# --------------------------------------------------------------------------
# whole-brain lesions at G_nom
# --------------------------------------------------------------------------
def test_lesions_at_g_nom_are_on_the_sweep_grid(cat):
    levels = wb.g_sweep_levels()
    assert A2.g_nom(cat) == levels[2] == pytest.approx(8.0 / 7.0, abs=1e-6)
    conds = A2.lesion_conditions_g_nom(catalogue=cat)
    assert [c["name"] for c in conds] == ["lesion_hub", "lesion_random_matched_hub"]
    assert all(c["config"].G == A2.g_nom(cat) for c in conds)
    hub, rnd = (c["config"] for c in conds)
    assert hub.lesion == "hub" and rnd.lesion == "random"
    conn = wb.load_connectome(grain=hub.grain)
    n_hub = int(np.count_nonzero(np.triu(wb.lesion_mask(conn, "hub",
                                                        n_hubs=hub.n_hubs), 1)))
    assert rnd.n_lesion_edges == n_hub > 0
    other = A2.lesion_conditions_g_nom(base={"duration_sec": 30.0}, catalogue=cat)
    assert all(c["config"].duration_sec == 30.0 for c in other)
