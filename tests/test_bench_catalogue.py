# -*- coding: utf-8 -*-
"""
The system catalogue (``systems.yaml``) and its builder: a self-contained
catalogue, the configuration-only systems (PC_half, W_PDI_no_multistability,
W_NAS_common_input_control, N_modules_disconnected, family-C N_uncoupled),
configuration presets (RAM-only arm, slow-context agents), the adversaries
and their labels, twin and held-out flags, recorded inputs, the witnesses
and their mechanisms, the import rules of the catalogue and the checks, and
the whole-brain lesions at G_nom.
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from impact_pipeline.bench import adversarial
from impact_pipeline.bench import catalogue as CAT
from impact_pipeline.bench import generators as g
from impact_pipeline.bench import manipulation as MP
from impact_pipeline.bench import whole_brain as wb

SRC = Path(__file__).resolve().parents[1] / "src"
SMALL = g.AgentConfig(n_trials=16, n_reafference_pairs=6)


@pytest.fixture(scope="module")
def cat():
    return CAT.load_catalogue()


@pytest.fixture(scope="module")
def raw():
    return CAT._read_yaml(CAT.CATALOGUE_PATH)


def test_the_catalogue_loads_and_validates(cat):
    assert CAT.validate_catalogue(cat) == []
    assert cat["schema_version"] == 1
    assert cat["catalogue_version"] == "mpc-bench-systems/1.0.0"
    assert cat["generator_version"] == g.GENERATOR_VERSION
    assert cat["source"].endswith("systems.yaml")
    ids = CAT.system_ids(cat)
    assert len(ids) == len(set(ids))


def test_every_entry_is_complete_in_the_file(raw):
    """No entry inherits fields from elsewhere: every required field is in
    the file itself."""
    for w in raw["witnesses"]:
        assert set(CAT.WITNESS_REQUIRED) <= set(w), w["id"]
    for a in raw["adversaries"]:
        assert set(CAT.ADVERSARY_REQUIRED) <= set(a), a["id"]
    assert CAT.validate_catalogue(CAT.resolve_catalogue(raw)) == []


def test_family_c_drops_the_duplicated_family_a_nulls(cat):
    c_ids = set(CAT.system_ids(cat, kind="witness", family="C"))
    assert {"N_independent_noise", "N_ar1", "O_hypersynchronous"}.isdisjoint(c_ids)
    assert {"N_uncoupled", "N_modules_disconnected"} <= c_ids
    assert "N_uncoupled" not in CAT.system_ids(cat, family="A")
    assert "W_PDI_no_multistability" not in c_ids
    for sid, fam in (("N_uncoupled", "A"), ("N_ar1", "C"),
                     ("W_PDI_no_multistability", "C")):
        with pytest.raises(ValueError, match="not defined for family"):
            CAT.build_system(sid, 0, fam, config=SMALL, catalogue=cat)


# --------------------------------------------------------------------------
# configuration-only systems
# --------------------------------------------------------------------------
def test_pc_half_knobs_are_half_the_nominal_dose(cat):
    e = CAT.get_entry("PC_half", cat)
    half = g.knobs_from_dict(e["knobs"])
    nom = g.NOMINAL_KNOBS
    for k in ("eta", "g_b", "c_int", "e"):
        assert getattr(half, k) == pytest.approx(0.5 * getattr(nom, k)), k
    assert half.K == nom.K // 2 == 3
    assert half.k_gain == nom.k_gain and half.ff_only is False
    assert list(half.bits()) == [1, 1, 1, 1, 1]
    s = CAT.build_system("PC_half", 1, "A", config=SMALL, catalogue=cat)
    assert s.meta["knobs"] == half.to_dict()
    assert s.oracle["witness_intended_pattern"] == [1, 1, 1, 1, 1]
    # paired with PC_nominal: same network draws and schedule (K maps the
    # same context draws to three patterns instead of six)
    pc = CAT.build_system("PC_nominal", 1, "A", config=SMALL, catalogue=cat)
    assert np.array_equal(s.events["onset"], pc.events["onset"])
    assert np.array_equal(s.oracle["good_arm"], pc.oracle["good_arm"])
    assert np.array_equal(s.oracle["context_state"], pc.oracle["context_state"] // 2)
    assert np.array_equal(np.abs(s.oracle["unit_adjacency"]) > 0,
                          np.abs(pc.oracle["unit_adjacency"]) > 0)


@pytest.mark.parametrize("family", ["A", "C"])
def test_modules_disconnected_has_no_hub_periphery_path(cat, family):
    e = CAT.get_entry("N_modules_disconnected", cat)
    assert e["knobs"] == {"g_b": 0.0, "c_int": 0.0} and e["config"] == {"w_ff": 0.0}
    for seed in (0, 1):
        s = CAT.build_system("N_modules_disconnected", seed, family, config=SMALL,
                             catalogue=cat)
        p = MP.hub_periphery_paths(s)
        assert not p["hub_to_periphery"] and not p["periphery_to_hub"]
        assert p["connected_pairs"] == 0 and p["max_abs_direct_coupling"] == 0.0
        assert s.oracle["module_adjacency"].sum() == 0
        # the module-local recurrence is kept
        adj = s.oracle["unit_adjacency"]
        for idx in s.meta["modules"].values():
            assert np.any(adj[np.ix_(idx, idx)])
        # inputs as nominal: the hub's input (context drive and slow rhythm)
        # is the positive control's, sample for sample
        pc = CAT.build_system("PC_nominal", seed, family, config=SMALL, catalogue=cat)
        hub = s.meta["workspace_nodes"]
        assert np.array_equal(s.oracle["inputs"][hub], pc.oracle["inputs"][hub])
        assert np.any(s.oracle["inputs"][hub])
        assert MP.hub_periphery_paths(pc)["hub_to_periphery"]
        assert MP.hub_periphery_paths(pc)["periphery_to_hub"]


def test_no_multistability_has_no_ignition(cat):
    e = CAT.get_entry("W_PDI_no_multistability", cat)
    assert e["knobs"] == {"K": 1} and e["config"] == {"w_ign": 0.0, "w_adapt": 0.0}
    assert e["class"] == "single_deficit" and e["target"] == "PDI"
    nominal = []
    for seed in range(4):
        s = CAT.build_system("W_PDI_no_multistability", seed, "A", catalogue=cat)
        occ = MP.ignition_occupancy(s)
        assert occ["ignition_occupancy"] < 0.05
        assert s.meta["config"]["w_ign"] == 0.0 and s.meta["knobs"]["g_b"] == 1.0
        # receive-and-return edges kept
        assert MP.hub_periphery_paths(s)["periphery_to_hub"]
        # the check is not vacuous: the nominal agent of the same seed
        # ignites, and the occupancy is the generator summary's there
        pc = CAT.build_system("PC_nominal", seed, "A", catalogue=cat)
        occ_pc = MP.ignition_occupancy(pc)["ignition_occupancy"]
        assert occ_pc == g.summarise_oracle(pc)["ignition_occupancy"]
        nominal.append(occ_pc)
    assert max(nominal) > 0.05


def test_common_input_control_and_uncoupled(cat):
    e = CAT.get_entry("W_NAS_common_input_control", cat)
    assert e["knobs"] == {"K": 1, "g_b": 0.0}
    assert e["intended_pattern"] == [1, 0, 0, 1, 1]
    assert e["class"] == "targeted_null" and e["target"] == "NAS"
    assert "not counted as a new" in e["notes"]
    u = CAT.get_entry("N_uncoupled", cat)
    assert g.knobs_from_dict(u["knobs"]) == g.OFF_KNOBS
    s = CAT.build_system("N_uncoupled", 2, "C", config=SMALL, catalogue=cat)
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
    assert CAT.config_preset("ram_only_160", cat) == {"n_trials": 160}
    slow = CAT.config_preset("slow_context_bold", cat)
    assert slow == {"ctx_dwell": [30.0, 60.0], "trial_sec": 12.0, "iti_jitter_sec": 2.0}
    for p in cat["config_presets"].values():
        assert set(p["config"]) <= set(g.AgentConfig.__dataclass_fields__)
    ram = CAT.build_system("W_RAM_no_plasticity", 1, "A", preset="ram_only_160",
                           catalogue=cat)
    assert (ram.events.trial_type == "goal_cue").sum() == 160
    assert ram.meta["config"]["n_reafference_pairs"] == 30
    sc = CAT.build_system("PC_nominal", 1, "A", preset="slow_context_bold",
                          catalogue=cat)
    runs = MP.context_runs_sec(sc)
    assert runs.min() >= 30.0 - sc.dt
    assert sc.meta["config"]["ctx_dwell"] == [30.0, 60.0]
    assert sc.meta["config"]["trial_sec"] == 12.0
    row = MP.slow_context_check(1, catalogue=cat).iloc[0]
    assert row["passed"] and row["config_in_force"]
    with pytest.raises(KeyError):
        CAT.config_preset("nope", cat)
    # an adversary takes the preset too (RAM-only arm)
    sf = CAT.build_system("adversarial_scrambled_feedback", 1, preset="ram_only_160",
                          catalogue=cat)
    assert sf.meta["config"]["n_trials"] == 160
    assert sf.meta["config"]["feedback_mode"] == "scrambled"


def test_a_preset_may_not_contradict_a_system(cat):
    bad = copy.deepcopy(cat)
    bad["config_presets"]["clash"] = {"description": "x", "config": {"w_ff": 1.0}}
    with pytest.raises(ValueError, match="contradicts"):
        CAT.build_system("N_modules_disconnected", 0, "A", preset="clash", catalogue=bad)
    bad["config_presets"]["agree"] = {"description": "x", "config": {"w_ff": 0.0,
                                                                     "n_trials": 8}}
    s = CAT.build_system("N_modules_disconnected", 0, "A", preset="agree", catalogue=bad)
    assert s.meta["config"]["n_trials"] == 8 and s.meta["config"]["w_ff"] == 0.0


def test_configuration_is_applied_in_order(cat):
    """Base configuration, then preset, then the system's own overrides."""
    s = CAT.build_system("W_PDI_no_multistability", 0, "A",
                         config={"n_trials": 12, "w_ign": 9.0}, preset="ram_only_160",
                         catalogue=cat)
    assert s.meta["config"]["n_trials"] == 160 and s.meta["config"]["w_ign"] == 0.0
    s = CAT.build_system("PC_nominal", 0, "A", config=SMALL, catalogue=cat)
    assert s.meta["config"] == SMALL.to_dict()


# --------------------------------------------------------------------------
# adversaries
# --------------------------------------------------------------------------
def test_the_adversaries_and_their_labels(cat):
    adv = {a["id"]: a for a in cat["adversaries"]}
    agents = {f"adversarial_{k}" for k in adversarial.ADVERSARIAL_KINDS}
    assert agents <= set(adv)
    assert set(adv) - agents == set(adversarial.STAGGERED_SYSTEM_VARIANTS)
    for aid in agents:
        lab = adversarial.ADVERSARIAL_CATALOGUE[aid[len("adversarial_"):]]
        assert adv[aid]["designed_to_fool"] == lab["designed_to_fool"]
    sf = adv["adversarial_scrambled_feedback"]
    assert sf["intended_pattern"] == [1, 1, 1, 1, 1]
    assert sf["expected_verdict"] == "MPC_CONSISTENT"
    assert sf["role"] == "construct_boundary"
    rf = adv["adversarial_reflex_arc"]
    assert rf["intended_pattern"] == [0, 0, 0, 0, 0]
    s = CAT.build_system("adversarial_scrambled_feedback", 3, config=SMALL,
                         catalogue=cat)
    assert s.oracle["adversary_intended_pattern"] == [1, 1, 1, 1, 1]
    # the generator labels the construction RAM-off
    assert s.oracle["intended_bits"] == [0, 1, 1, 1, 1]
    for aid in ("adversarial_parity_grid", "adversarial_hypersynchrony", "nope"):
        assert aid not in adv
        with pytest.raises(KeyError, match="Unknown"):
            CAT.get_entry(aid, cat)


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
        s = CAT.build_system(wid, 5, fam, replicate=2, config=SMALL, catalogue=cat)
        assert s.meta["replicate"] == 2 and s.meta["witness_id"] == wid
    for sid in ("N_ar1", "PW_patchwork", "adversarial_reflex_arc"):
        with pytest.raises(ValueError, match="twins"):
            CAT.build_system(sid, 5, "A", replicate=1, config=SMALL, catalogue=cat)


@pytest.mark.parametrize("family", ["A", "C"])
def test_every_entry_builds_with_its_recorded_inputs(cat, family):
    for sid in CAT.system_ids(cat, family=family):
        e = CAT.get_entry(sid, cat)
        for v in (e.get("variants") or [None]):
            s = CAT.build_system(sid, 6, family, variant=v, config=SMALL, catalogue=cat)
            label = s.meta.get("witness_id") or s.meta.get("adversary_id")
            assert label == sid
            ch = CAT.recorded_channels(s, catalogue=cat)
            assert set(ch) == set(e["recorded_inputs"]), sid
            for c in ch.values():
                assert c["value"].shape[-1] == s.n_time, sid
    pc = CAT.build_system("PC_nominal", 6, family, config=SMALL, catalogue=cat)
    pc.oracle.pop("slow_phase")
    with pytest.raises(KeyError, match="slow_phase"):
        CAT.recorded_channels(pc, catalogue=cat)


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
                                                 "references": []}},
     "no references"),
    ("adversaries", "ADV_NAS_staggered_driver", {"references": []}, "no references"),
    ("adversaries", "ADV_NAS_staggered_tau10", {"held_out": False}, "held_out"),
    ("adversaries", "ADV_NAS_staggered_driver", {"variants": ["uniform"]}, "variants"),
    ("adversaries", "adversarial_reflex_arc", {"variants": ["uniform"]},
     "only the staggered"),
    ("adversaries", "adversarial_reflex_arc", {"families": ["A", "C"]}, "family A"),
    ("adversaries", "ADV_NAS_staggered_sat", {"generator": "slow_drift"},
     "unknown generator"),
    ("adversaries", "adversarial_common_driver", {"generator": "adversarial_x"},
     "unknown generator"),
])
def test_validation_reports_each_error(cat, kind, sid, changes, message):
    errors = CAT.validate_catalogue(_with(cat, kind, sid, **changes))
    assert any(message in e and sid in e for e in errors), errors


def test_validation_of_the_catalogue_level(cat):
    bad = copy.deepcopy(cat)
    bad["schema_version"] = 2
    bad["references"]["x"] = "doi.org/1"
    bad["whole_brain_presets"]["lesions_g_nom"]["G"] = 1.0
    bad["witnesses"].append(copy.deepcopy(bad["witnesses"][0]))
    bad["witnesses"].append({"id": "INCOMPLETE"})
    errors = " | ".join(CAT.validate_catalogue(bad))
    for msg in ("schema_version", "not a DOI", "is not the sweep level",
                "duplicate id", "INCOMPLETE: missing"):
        assert msg in errors, msg


def test_the_doi_table_is_cited(raw):
    refs = raw["references"]
    used = set()
    for w in raw["witnesses"]:
        used |= set(w["counterexample"]["references"])
    for a in raw["adversaries"]:
        used |= set(a["references"])
    assert set(refs) <= used
    assert all(doi.startswith("10.") for doi in refs.values())


def test_catalogue_witness_contents():
    cat = CAT.load_catalogue()
    ids = {w["id"] for w in cat["witnesses"]}
    classes = {w["class"] for w in cat["witnesses"]}
    assert classes == set(CAT.WITNESS_CLASSES)
    targets = {w["target"] for w in cat["witnesses"] if w["class"] == "single_deficit"}
    assert targets == set(g.PRINCIPLES)
    assert {
        "PC_nominal",
        "N_independent_noise",
        "N_ar1",
        "O_hypersynchronous",
        "PW_patchwork",
    } <= ids
    for w in cat["witnesses"]:
        assert w["counterexample"]["references"], w["id"]
    bad = json.loads(json.dumps(cat))
    bad["witnesses"][2]["intended_pattern"] = [1, 1, 1, 1, 1]
    assert CAT.validate_catalogue(bad)


def test_witnesses_realise_their_mechanisms():
    for wid in CAT.system_ids(kind="witness", family="A"):
        w = CAT.get_entry(wid)
        s = CAT.build_system(wid, 2, "A", config=SMALL)
        assert s.meta["witness_id"] == wid
        if w["generator"] == "family_a":
            assert s.oracle["intended_bits"] == w["intended_pattern"], wid
        else:
            assert s.oracle["witness_intended_pattern"] == w["intended_pattern"]
    s = CAT.build_system("PW_patchwork", 2, "C", config=SMALL)
    assert s.meta["dynamics"] == "stuart_landau" and s.meta["patchwork"]
    with pytest.raises(KeyError):
        CAT.get_entry("nope")


def test_catalogue_and_checks_keep_the_import_rules():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from impact_pipeline.bench import catalogue, manipulation\n"
        "cat = catalogue.load_catalogue()\n"
        "catalogue.build_system('ADV_NAS_staggered_driver', 0)\n"
        "catalogue.build_system('PC_nominal', 0, replicate=1)\n"
        "banned = ('impact_pipeline.mpc_metrics', 'impact_pipeline.synergy_ci')\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('mne', 'sklearn')\n"
        "             or m in banned)\n"
        "print('BAD', bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------
# whole-brain lesions at G_nom
# --------------------------------------------------------------------------
def test_lesions_at_g_nom_are_on_the_sweep_grid(cat):
    levels = wb.g_sweep_levels()
    assert CAT.g_nom(cat) == levels[2] == pytest.approx(8.0 / 7.0, abs=1e-6)
    conds = CAT.lesion_conditions_g_nom(catalogue=cat)
    assert [c["name"] for c in conds] == ["lesion_hub", "lesion_random_matched_hub"]
    assert all(c["config"].G == CAT.g_nom(cat) for c in conds)
    hub, rnd = (c["config"] for c in conds)
    assert hub.lesion == "hub" and rnd.lesion == "random"
    conn = wb.load_connectome(grain=hub.grain)
    n_hub = int(np.count_nonzero(np.triu(wb.lesion_mask(conn, "hub",
                                                        n_hubs=hub.n_hubs), 1)))
    assert rnd.n_lesion_edges == n_hub > 0
    other = CAT.lesion_conditions_g_nom(base={"duration_sec": 30.0}, catalogue=cat)
    assert all(c["config"].duration_sec == 30.0 for c in other)
