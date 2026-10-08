# -*- coding: utf-8 -*-
"""
The v2 design builders and their registry: the run plan of every design is
what its builder returns, its task counts equal those of the design
document, seeds follow the v2 seed policy and seed map, held-out conditions
exist only where they may, and the scorings carry the declarations, forms
and protocols the design names. No task is simulated here.
"""
import dataclasses

import pytest

from impact_pipeline.bench import adversarial_v2 as A2
from impact_pipeline.bench import designs_v2 as D
from impact_pipeline.bench import run_bench_v2 as RB
from impact_pipeline.bench.designs_v2 import anchors as AN
from impact_pipeline.bench.designs_v2 import family_a as FA
from impact_pipeline.bench.designs_v2 import family_c1 as FC
from impact_pipeline.bench.designs_v2 import ram_only as RO
from impact_pipeline.bench.designs_v2 import twins as TW
from impact_pipeline.v2 import seeds as S

OWN_MODULES = ("family_a", "family_c1", "ram_only", "twins", "anchors", "forward")
CONF, DEV = D.CONFIRMATORY, D.DEVELOPMENT
P5 = ("RAM", "PDI", "NAS", "IIM", "SRPI")

# Task counts of the design document (3.2 and 3.7), per design and split.
DOCUMENT_COUNTS = {
    CONF: {
        # the two null witnesses on 46 seeds (the HCv2-1 resize) and the
        # family-A anchor replication block on 40 (CD-11)
        "A_witnesses": 16 * 20 + 7 * 25 + 2 * 26, "A_sweeps": 260, "A_factorial": 320,
        "A_adversaries": 140, "A_heldout": 40, "A_twins": 300 + 10,
        "A_anchors": 7 * 40, "RAM160": 440, "RAM160_twins": 210,
        "C1_witnesses": 13 * 20 + 4 * 25, "C1_sweeps": 200, "C1_factorial": 320,
        "C1_twins": 180, "C1_anchors": 80, "RAM160_anchors": 2 * 20,
        # the forward arms (3.2; the BOLD arm's 271 before curtailment) and
        # the forward views' anchor replication (3 arms x 20)
        "whole_brain": 371, "forward_family_a": 332, "forward_family_a_bold": 271,
        "forward_anchor_replication": 3 * 20,
    },
    DEV: {
        "A_witnesses": 16 * 12, "A_sweeps": 26 * 4, "A_factorial": 32 * 4,
        "A_adversaries": 7 * 12, "A_heldout": 8 * 5, "A_twins": 5 * 5 * 7,
        "A_anchors": 7 * 40, "RAM160": 11 * 20, "RAM160_twins": 3 * 5 * 8,
        "C1_witnesses": 13 * 12, "C1_sweeps": 20 * 4, "C1_factorial": 32 * 4,
        "C1_twins": 3 * 5 * 7, "C1_anchors": 4 * 40, "RAM160_anchors": 2 * 40,
    },
}


@pytest.fixture(scope="module")
def plans():
    return {split: {name: D.get_design(name).tasks(split)
                    for name in DOCUMENT_COUNTS[split]} for split in D.SPLITS}


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def test_every_design_module_is_pre_declared_and_ours_are_available():
    assert set(OWN_MODULES) <= set(D.DESIGN_MODULES)
    for later in ("family_b", "forward", "null_calibration", "srpi_only",
                  "patchwork_v2", "tier_b_arms"):
        assert later in D.DESIGN_MODULES
    assert set(OWN_MODULES) <= set(D.available_modules())
    names = set(D.designs())
    assert set(DOCUMENT_COUNTS[CONF]) <= names


def test_a_missing_module_is_not_merged_yet(monkeypatch):
    real = D.importlib.import_module

    def fake(name, *a, **k):
        if name == f"{D.PACKAGE}.forward":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real(name, *a, **k)

    monkeypatch.setattr(D.importlib, "import_module", fake)
    monkeypatch.delitem(D._MODULE_CACHE, "forward", raising=False)
    with pytest.raises(D.DesignNotAvailableError, match="not merged yet"):
        D.load_module("forward")
    with pytest.raises(D.DesignError, match="unknown design module"):
        D.load_module("family_z")
    with pytest.raises(D.DesignError, match="not merged yet: .*forward"):
        D.get_design("F_eeg64")


def test_merged_modules_without_runner_designs_are_reported_not_skipped(
        capsys, monkeypatch):
    # family B runs through its own validation script; a merged module whose
    # runner adapter is not written yet is reported as such (a stand-in for
    # a Tier-B module here), and the forward arms run through the runner
    import types

    assert D.RUN_ELSEWHERE["family_b"] == "scripts/v2/iim_validation_v2.py"
    with pytest.raises(D.DesignNotRunnableError,
                       match="family_b.*runs? through scripts/v2/iim_validation_v2"):
        D.load_module("family_b")
    fake = types.ModuleType(f"{D.PACKAGE}.tier_b_arms")
    real = D.importlib.import_module
    monkeypatch.setattr(D.importlib, "import_module",
                        lambda name, *a, **k: fake if name == fake.__name__
                        else real(name, *a, **k))
    monkeypatch.delitem(D._MODULE_CACHE, "tier_b_arms", raising=False)
    with pytest.raises(D.DesignNotRunnableError, match="runner adapter is not written"):
        D.load_module("tier_b_arms")
    unrunnable = D.unrunnable_modules()
    assert {"family_b", "tier_b_arms"} <= set(unrunnable)
    assert "forward" in D.available_modules()
    assert not {"family_b", "tier_b_arms"} & set(D.available_modules())
    assert not {"family_b", "tier_b_arms"} & set(D.missing_modules())
    with pytest.raises(D.DesignError,
                       match="no designs for this runner: .*tier_b_arms"):
        D.get_design("Z_tier_b")
    # the plan scripts read a non-zero status instead of an empty list
    assert RB.main(["designs", "--module", "tier_b_arms"]) == 3
    assert "runner adapter is not written" in capsys.readouterr().err
    assert RB.main(["designs", "--module", "family_b"]) == 3
    assert RB.main(["designs", "--module", "forward"]) == 0
    assert capsys.readouterr().out.split() == [
        "whole_brain,forward_family_a,forward_family_a_bold,"
        "forward_anchor_replication"]
    assert RB.main(["designs"]) == 0
    out = capsys.readouterr().out
    assert "family_b: design module 'family_b'" in out
    assert "anchors: A_anchors" in out
    assert "forward: whole_brain, forward_family_a" in out


def test_a_module_with_malformed_designs_is_still_an_error(monkeypatch):
    import types

    fake = types.ModuleType(f"{D.PACKAGE}.forward")
    fake.DESIGNS = ["not a tuple of Design"]
    real = D.importlib.import_module
    monkeypatch.setattr(D.importlib, "import_module",
                        lambda name, *a, **k: fake if name == fake.__name__
                        else real(name, *a, **k))
    monkeypatch.delitem(D._MODULE_CACHE, "forward", raising=False)
    with pytest.raises(D.DesignError, match="must define DESIGNS as a tuple") as exc:
        D.load_module("forward")
    assert not isinstance(exc.value, D.DesignNotRunnableError)


def test_design_names_are_unique_and_designs_know_their_module():
    ds = D.designs(OWN_MODULES)
    for name, d in ds.items():
        assert d.module in OWN_MODULES
        assert d.name == name


# --------------------------------------------------------------------------
# run plan equals the builders, counts equal the design document
# --------------------------------------------------------------------------
@pytest.mark.parametrize("split", D.SPLITS)
def test_plan_table_is_generated_from_the_builders(plans, split):
    rows = {r["design"]: r for r in D.plan_table(split, list(DOCUMENT_COUNTS[split]))}
    for name, tasks in plans[split].items():
        r = rows[name]
        assert r["n_tasks"] == len(tasks)
        assert r["n_scorings"] == sum(len(t.scorings) for t in tasks)
        assert r["expected"] == DOCUMENT_COUNTS[split][name]
        assert r["matches"] is True, name


@pytest.mark.parametrize("split", D.SPLITS)
def test_task_ids_are_unique_and_the_runner_plans_the_same_tasks(plans, split):
    ids = [t.task_id for tasks in plans[split].values() for t in tasks]
    assert len(ids) == len(set(ids))
    names = list(DOCUMENT_COUNTS[split])
    assert [t.task_id for t in D.build_plan(names, split)] == ids
    assert [t.task_id for t in RB.build_tasks(names, split)] == ids
    for t in D.build_plan(names, split):
        assert t.design_module == D.get_design(t.design).module


def test_every_design_carries_an_ademp_statement():
    rows = D.plan_table(CONF, list(DOCUMENT_COUNTS[CONF]))
    for r in rows:
        assert tuple(r["ademp"]) == D.ADEMP_KEYS, r["design"]
        assert all(len(v) > 20 for v in r["ademp"].values()), r["design"]
    with pytest.raises(D.DesignError, match="ADEMP"):
        D.Design("X", "A", "x", lambda split: [], ademp={"aims": "only aims"})


def test_build_plan_passes_each_builder_the_options_it_accepts(monkeypatch):
    tasks = D.build_plan(["A_factorial", "A_witnesses"], DEV, seeds=[336],
                         systems=["PC_nominal"], cells=None)
    fac = [t for t in tasks if t.design == "A_factorial"]
    wit = [t for t in tasks if t.design == "A_witnesses"]
    assert len(fac) == 32 and {t.seed for t in fac} == {336}  # no systems option
    assert [t.system for t in wit] == ["PC_nominal"]
    # a task whose builder left its design module out gets it from the registry
    d = D.get_design("A_sweeps")

    def bare(split, **kw):
        return [dataclasses.replace(t, design_module=None)
                for t in d.build(split, **kw)]

    monkeypatch.setitem(D._MODULE_CACHE, "family_a", type(
        "M", (), {"DESIGNS": tuple(dataclasses.replace(x, build=bare)
                                   if x.name == "A_sweeps" else x
                                   for x in FA.DESIGNS)}))
    got = RB.build_tasks(["A_sweeps"], DEV, seeds=[332])
    assert got and all(t.design_module == "family_a" for t in got)


def test_cli_list_and_plan_equal_the_builders(capsys, tmp_path):
    assert RB.main(["list", "A_sweeps,C1_factorial", "--split", "development"]) == 0
    out = capsys.readouterr().out.split()
    want = [t.task_id for t in D.build_plan(["A_sweeps", "C1_factorial"], DEV)]
    assert out == want
    path = tmp_path / "plan.json"
    assert RB.main(["plan", "--split", "confirmatory", "--json", str(path)]) == 0
    import json

    assert json.loads(path.read_text()) == D.plan_table(CONF)


@pytest.mark.parametrize("split", D.SPLITS)
def test_seeds_follow_the_policy_and_the_seed_map(plans, split):
    smap = S.load_seed_map()
    for name, tasks in plans[split].items():
        seeds = {t.seed for t in tasks}
        assert S.check_seeds(seeds) == split
        for s in seeds:
            uses = S.seed_uses(s, smap)
            assert uses, (name, s)
        if split == DEV:
            assert not any(10000 <= s <= 19999 for s in seeds)


def test_development_seed_blocks_of_the_dry_run():
    blocks = {
        "A_witnesses": (320, 331), "A_sweeps": (332, 335), "A_factorial": (336, 339),
        "A_adversaries": (340, 351), "RAM160": (352, 371), "C1_witnesses": (372, 383),
        "C1_sweeps": (372, 375), "C1_factorial": (376, 379),
        "A_twins": (820, 824), "C1_twins": (820, 824), "RAM160_twins": (820, 824),
        "A_anchors": (900, 939), "C1_anchors": (900, 939), "RAM160_anchors": (900, 939),
        "A_heldout": (980, 984),
    }
    for name, (lo, hi) in blocks.items():
        seeds = {t.seed for t in D.get_design(name).tasks(DEV)}
        assert seeds == set(range(lo, hi + 1)), name


# --------------------------------------------------------------------------
# systems, levels, forms and protocols
# --------------------------------------------------------------------------
def test_family_a_witnesses_extended_seeds_and_order(plans):
    tasks = plans[CONF]["A_witnesses"]
    by_sys = {}
    for t in tasks:
        by_sys.setdefault(t.system, set()).add(t.seed)
    want = set(A2.system_ids(kind="witness", family="A"))
    assert set(by_sys) == want and len(want) == 16
    for sid, seeds in by_sys.items():
        n = (45 if sid in FA.EXTENDED_SYSTEMS else 46 if sid in FA.NULL_WITNESSES
             else 20)
        assert seeds == set(range(20000, 20000 + n)), sid
    # PC_nominal and the six single deficits run first
    order = list(dict.fromkeys(t.system for t in tasks))
    assert order[:7] == list(FA.EXTENDED_SYSTEMS)
    assert set(FA.EXTENDED_SYSTEMS) == {"PC_nominal"} | {
        sid for sid in A2.system_ids(kind="witness", family="A")
        if A2.get_entry(sid).get("class") == "single_deficit"
        and A2.get_entry(sid).get("origin") == "v1"}
    # the development witnesses keep one block for every system
    dev = {t.seed for t in plans[DEV]["A_witnesses"] if t.system in FA.NULL_WITNESSES}
    assert dev == set(range(320, 332))
    # a seed restriction replaces every block, the extensions included
    only = FA.witnesses(CONF, seeds=[20030], systems=list(FA.NULL_WITNESSES))
    assert [(t.system, t.seed) for t in only] == [("N_independent_noise", 20030),
                                                   ("N_ar1", 20030)]


def test_the_null_witnesses_are_sized_for_one_tolerated_event(repo_root):
    """HCv2-1 (CD-11 resize): the two family-A null witnesses hold the RAM-PE
    rows of the pooled H0 cell rule; 2 x 46 seed clusters is the smallest
    count whose pooled Clopper-Pearson bound at 0.05 / 5 stays below 0.07
    with one PRESENT event (seeds resized, the bound unchanged)."""
    import json

    from impact_pipeline.v2 import hypothesis_engine as HE
    from impact_pipeline.v2 import registry_v3 as RV

    spec = json.loads((repo_root / "protocols/v2/hypotheses_v2.json").read_text())
    assert tuple(spec["vocabulary"]["systems.nulls_A"]) == FA.NULL_WITNESSES
    level = 0.05 / len(P5)
    n = len(FA.NULL_WITNESSES) * len(D.seeds_of(FA.SEEDS["witnesses"], CONF)
                                     + D.seeds_of(FA.SEEDS["null_witnesses_extended"],
                                                  CONF))
    assert n == 92 == RV.min_runs_for_demonstration(0.07, level, events=1)
    assert HE.cp_upper(1, n, level) < 0.07 <= HE.cp_upper(1, n - 1, level)


@pytest.mark.parametrize("family", ["A", "C"])
def test_the_patchwork_iim_grain_is_the_iim_modules_three_subgroups(family):
    """The patchwork's declared IIM grain in the principle-bearer mode, the
    only mode it is recorded in: the IIM module's three sub-groups. Its
    system grain (one macro node per module, five nodes, about 28 times a
    four-node IIM call) is never scored."""
    from impact_pipeline.bench import export as X

    pw = A2.build_system("PW_patchwork", 320, family)
    iim = X.bearer_view(pw, "IIM", "principle")
    assert iim["bearer_id"] == "principle:IIM"
    assert len(iim["macro_nodes"]) == 3
    assert sorted(i for v in iim["macro_nodes"].values() for i in v) == list(
        range(len(iim["nodes"])))
    assert len(X.bearer_view(pw, "IIM", "system")["macro_nodes"]) == 5
    for p in ("NAS", "PDI", "RAM", "SRPI"):
        assert X.bearer_view(pw, p, "principle")["bearer_id"] == f"principle:{p}"


def test_family_a_scorings_declarations_forms_and_protocols(plans):
    t = next(t for t in plans[DEV]["A_witnesses"] if t.system == "PC_nominal")
    ids = [s.scoring_id for s in t.scorings]
    assert ids == ["R", "H", "R+nas_secondary", "H+nas_secondary",
                   "R+iim_bidirectional", "H+iim_bidirectional", "R+nas_tau_0.05",
                   "R+nas_tau_0.2"]
    prim = {s.scoring_id: s for s in t.scorings}
    assert prim["R"].protocol_key == "A-R" and prim["H"].protocol_key == "A-H"
    assert prim["R"].principles == P5 and prim["R"].verdict
    assert prim["H+nas_secondary"].protocol_key == "A-H+nas_secondary"
    assert prim["H+nas_secondary"].principles == ("NAS",)
    assert not prim["H+nas_secondary"].verdict
    for s in t.scorings:
        base, form = D.split_protocol_key(s.protocol_key)
        assert form == s.estimator_form
        assert RB.FAMILY_PROTOCOLS[base] == ("A", s.declaration_id)
    # the patchwork is recorded in the principle-bearer mode only (design 3.2):
    # its system grain would give IIM five macro nodes, which no Tier-A
    # hypothesis reads
    for name in ("A_witnesses", "C1_witnesses"):
        for split in (DEV, CONF):
            pws = [t for t in plans[split][name] if t.system == "PW_patchwork"]
            assert pws, (name, split)
            for pw in pws:
                assert {s.bearer_mode for s in pw.scorings} == {"principle"}
                ids = {s.scoring_id for s in pw.scorings}
                assert {"R@principle", "H@principle"} <= ids and "R" not in ids
    assert FA.bearer_modes("PW_patchwork") == ("principle",)
    assert FA.bearer_modes("PC_nominal") == ("system",)


def test_held_out_conditions_exist_only_where_they_may(plans):
    conf = plans[CONF]["A_witnesses"]
    ho = [(t.system, t.seed, s.scoring_id) for t in conf for s in t.scorings
          if s.held_out]
    rescored = [x for x in ho if x[2] in ("P", "Q10", "Q25", "J")]
    assert len(rescored) == FA.EXPECTED_HELD_OUT_RESCORINGS[CONF] == 400
    assert {x[0] for x in rescored} == set(FA.HELD_OUT_RESCORED)
    assert {x[1] for x in rescored} == set(range(20000, 20020))
    mis = [x for x in ho if x[2] == "R+pdi_misdeclared_access"]
    assert {x[0] for x in mis} == {"W_PDI_single_attractor"} and len(mis) == 45
    for t in conf:
        for s in t.scorings:
            if s.declaration_id in ("P", "Q10", "Q25", "J"):
                assert s.held_out and s.principles == ("NAS", "IIM")
                assert s.protocol_key == f"A-{s.declaration_id}"
    # development: no held-out scoring outside the smoke seeds
    for name, tasks in plans[DEV].items():
        for t in tasks:
            if t.has_held_out:
                assert S.is_smoke_seed(t.seed), (name, t.task_id)
    smoke = plans[DEV]["A_heldout"]
    assert all(t.smoke and t.has_held_out for t in smoke)
    assert {(t.system, t.params.get("variant")) for t in smoke} >= {
        ("ADV_NAS_staggered_tau10", "tau10"), ("ADV_NAS_staggered_sat", "saturating")}
    conf_ho = plans[CONF]["A_heldout"]
    assert {(t.system, t.params.get("variant")) for t in conf_ho} == {
        ("ADV_NAS_staggered_tau10", "tau10"), ("ADV_NAS_staggered_sat", "saturating")}
    assert all(s.principles == ("NAS",) and s.declaration_id == "R"
               and not s.verdict for t in conf_ho for s in t.scorings)
    with pytest.raises(D.DesignError, match="smoke"):
        FA.held_out(DEV, seeds=[320])


def test_sweeps_factorial_and_adversaries(plans):
    sw = plans[CONF]["A_sweeps"]
    levels = {}
    for t in sw:
        levels.setdefault(t.tags["sweep_knob"], set()).add(t.tags["sweep_level"])
    assert sorted(levels["K"]) == [2, 3, 4, 6, 8, 12]
    assert len(levels["g_b"]) == 10
    assert (min(levels["g_b"]), max(levels["g_b"])) == (0, 2)
    assert len(levels["c_int"]) == 10 and min(levels["c_int"]) == 0
    for t in sw:
        if t.tags["sweep_knob"] == "g_b":
            assert t.params["knobs"]["K"] == 6
    fac = plans[CONF]["A_factorial"]
    assert len({t.system for t in fac}) == 32
    adv = plans[CONF]["A_adversaries"]
    # a variant is a parameter of its catalogue system, not part of its name
    assert {(t.system, t.params.get("variant")) for t in adv} == {
        ("adversarial_common_driver", None), ("adversarial_reflex_arc", None),
        ("adversarial_random_label_self_other", None),
        ("adversarial_scrambled_feedback", None),
        ("ADV_NAS_staggered_driver", "hierarchical"),
        ("ADV_NAS_staggered_driver", "uniform"),
        ("ADV_NAS_staggered_driver", "reversed")}
    assert len({t.task_id for t in adv}) == len(adv)
    assert not any(t.has_held_out for t in adv)


def test_c1_leaves_iim_out_before_the_freeze_except_the_anchors(plans):
    for name in ("C1_witnesses", "C1_sweeps", "C1_factorial", "C1_twins"):
        for t in plans[DEV][name]:
            assert all("IIM" not in s.principles for s in t.scorings), name
        assert any("IIM" in s.principles for t in plans[CONF][name]
                   for s in t.scorings), name
    assert any("IIM" in s.principles for t in plans[DEV]["C1_anchors"]
               for s in t.scorings)
    t = plans[CONF]["C1_witnesses"][0]
    assert {s.protocol_key.split("+")[0] for s in t.scorings} == {"C1-R", "C1-H"}
    assert t.params["family"] == "C" and t.family == "C1"
    assert len({t.system for t in plans[CONF]["C1_witnesses"]}) == 13
    assert set(FC.EXTENDED_SYSTEMS) == {
        t.system for t in plans[CONF]["C1_witnesses"] if t.seed >= 20020}


def test_ram_only_arm(plans):
    tasks = plans[CONF]["RAM160"]
    assert {t.system for t in tasks} == set(RO.CLASS_IDS) and len(RO.CLASS_IDS) == 11
    for t in tasks:
        assert t.params["preset"] == "ram_only_160"
        (s,) = t.scorings
        assert (s.declaration_id, s.protocol_key, s.principles) == (
            "R", "A-RAM160", ("RAM",))
        assert not s.verdict  # one principle by design: no verdict
    eta = {t.tags.get("eta") for t in tasks if t.system.startswith("eta_")}
    assert eta == set(RO.ETA_LEVELS)
    assert {t.tags["same_system_as"] for t in tasks if "same_system_as" in t.tags} == {
        "W_RAM_no_plasticity", "PC_nominal"}


def test_twin_sets(plans):
    a = plans[CONF]["A_twins"]
    sess = [t for t in a if t.scorings]
    assert {t.system for t in sess} == set(TW.twin_classes("A")) == {
        "PC_nominal", "PC_half", "W_PDI_single_attractor", "W_NAS_no_workspace",
        "W_IIM_feedforward"}
    assert {t.replicate for t in sess} == set(range(1, 7))
    checks = [t for t in a if not t.scorings]
    assert len(checks) == 10 and all(t.replicate == 0 for t in checks)
    assert {t.tags["identity_check_of"] for t in checks} == {
        f"A_witnesses-PC_nominal-s{s:05d}" for s in range(20000, 20010)}
    witness_ids = {t.task_id for t in plans[CONF]["A_witnesses"]}
    assert {t.tags["identity_check_of"] for t in checks} <= witness_ids
    dev = plans[DEV]["A_twins"]
    assert {t.replicate for t in dev} == set(range(0, 7))
    assert {t.tags["twin_network"] for t in dev} == set(range(820, 825))
    assert {t.replicate for t in plans[CONF]["RAM160_twins"]} == set(range(1, 8))
    assert {t.replicate for t in plans[DEV]["RAM160_twins"]} == set(range(0, 8))
    assert {t.system for t in plans[CONF]["RAM160_twins"]} == set(TW.RAM_TWIN_CLASSES)
    assert {t.system for t in plans[CONF]["C1_twins"]} == set(TW.twin_classes("C"))
    with pytest.raises(D.DesignError, match="not part of the A twin set"):
        TW.a_twins(CONF, replicates=[0])


def test_anchor_blocks(plans):
    for name, systems in (("A_anchors", AN.A_SYSTEMS), ("C1_anchors", AN.C1_SYSTEMS)):
        dev = plans[DEV][name]
        assert {t.seed for t in dev} == set(S.REFERENCE_BLOCK)
        assert {t.system for t in dev} == set(systems)
        conf = plans[CONF][name]
        # CD-11 extended family A to 20900-20939 and no other design
        want = AN.REPLICATION_SEEDS_EXTENDED if name == "A_anchors" else (
            AN.REPLICATION_SEEDS)
        assert {t.seed for t in conf} == set(want), name
    assert {t.seed for t in plans[CONF]["RAM160_anchors"]} == set(AN.REPLICATION_SEEDS)
    assert set(AN.OWN_LESION.values()) <= set(AN.A_SYSTEMS)
    ext = AN.a_anchors(CONF, replication_extended=True)
    assert {t.seed for t in ext} == set(range(20900, 20940))
    assert {t.seed for t in AN.a_anchors(CONF, replication_extended=False)} == set(
        range(20900, 20920))
    # the extension is a calibration decision declared once (CD-11)
    assert AN.CALIBRATION_PENDING["replication_extended"]["value"] == {
        "A_anchors": True, "C1_anchors": False, "RAM160_anchors": False}
    assert "decided at CD-11" in AN.CALIBRATION_PENDING["replication_extended"][
        "meaning"]
    assert [AN.n_replication_seeds(d) for d in AN.ANCHOR_DESIGNS] == [40, 20, 20]
    assert {d.name: d.expected_tasks[CONF] for d in AN.DESIGNS} == {
        "A_anchors": 7 * 40, "C1_anchors": 4 * 20, "RAM160_anchors": 2 * 20}
    # the development reference block is never resized
    assert {t.seed for t in AN.a_anchors(DEV)} == set(S.REFERENCE_BLOCK)
    roles = {t.system: t.tags["anchor_role"] for t in plans[DEV]["A_anchors"]}
    assert roles["PC_nominal"] == "positive_control"
    assert roles["W_NAS_no_workspace"] == "own_lesion:NAS"
    assert roles["W_NAS_broadcast_only"] == "single_deficit"
    ram = plans[DEV]["RAM160_anchors"]
    assert {t.system for t in ram} == {"PC_nominal", "W_RAM_no_plasticity"}
    assert all(t.params["preset"] == "ram_only_160" for t in ram)


# --------------------------------------------------------------------------
# the task vocabulary
# --------------------------------------------------------------------------
def test_task_spec_validation_and_round_trip():
    t = FA.witnesses(DEV, seeds=[320], systems=["PC_nominal"])[0]
    again = D.TaskSpec.from_dict(t.to_dict())
    assert again == t and again.to_dict() == t.to_dict()
    with pytest.raises(D.DesignError, match="v1 confirmatory block"):
        dataclasses.replace(t, seed=10000)
    with pytest.raises(D.DesignError, match="replicate"):
        dataclasses.replace(t, replicate=31)
    with pytest.raises(D.DesignError, match="duplicate scoring ids"):
        dataclasses.replace(t, scorings=t.scorings + t.scorings[:1])
    with pytest.raises(D.DesignError, match="declaration"):
        D.ScoringSpec("X", "XYZ", "A-R")
    with pytest.raises(D.DesignError, match="unknown"):
        D.make_scorings({"R": "A-R"}, P5, ("no_such_form",))
    held = dataclasses.replace(t, held_out=True)
    assert all(s.held_out for s in held.scorings)


def test_make_scorings_and_protocol_keys():
    sc = D.make_scorings({"R": "A-R", "H": "A-H"}, ("NAS", "RAM"),
                         ("nas_secondary", "iim_bidirectional", "nas_tau_0.2"),
                         form_declarations={"nas_tau_0.2": ("R",)})
    assert [s.scoring_id for s in sc] == [
        "R", "H", "R+nas_secondary", "H+nas_secondary", "R+nas_tau_0.2"]
    assert D.protocol_key("A-R", "nas_secondary") == "A-R+nas_secondary"
    assert D.split_protocol_key("C1-H+iim_bidirectional") == ("C1-H",
                                                              "iim_bidirectional")
    assert D.scoring_id("R", "primary", "eeg64", "principle") == "R/eeg64@principle"
    only_form = D.make_scorings({"R": "A-R"}, ("PDI",), ("pdi_misdeclared_access",),
                                primary=False)
    assert [(s.scoring_id, s.held_out) for s in only_form] == [
        ("R+pdi_misdeclared_access", True)]


def test_a_declared_replication_extension_reaches_builders_and_counts(monkeypatch):
    value = dict(AN.CALIBRATION_PENDING["replication_extended"]["value"],
                 A_anchors=False, C1_anchors=True)
    monkeypatch.setitem(AN.CALIBRATION_PENDING["replication_extended"], "value",
                        value)
    assert {t.seed for t in AN.c1_anchors(CONF)} == set(range(20900, 20940))
    assert {t.seed for t in AN.a_anchors(CONF)} == set(range(20900, 20920))
    assert AN.n_replication_seeds("C1_anchors") == 40
    assert AN.n_replication_seeds("A_anchors") == 20
    assert RB.calibration_pending()["anchors.replication_extended"]["value"][
        "C1_anchors"] is True


def test_the_forward_replication_extension_is_declared_per_arm(monkeypatch):
    from impact_pipeline.bench.designs_v2 import forward as FW

    # one arm list, declared in both modules
    assert AN.FORWARD_ARMS == FW.ARMS
    pending = AN.CALIBRATION_PENDING["forward_replication_extended"]
    # decided after the held-out release: provisional, no arm extended
    assert pending["value"] == {arm: False for arm in FW.ARMS}
    assert "CD-11" in pending["meaning"] and "provisional" in pending["meaning"]
    assert all(AN.forward_replication_seeds(a) == AN.REPLICATION_SEEDS
               for a in FW.ARMS)
    assert RB.calibration_pending()["anchors.forward_replication_extended"] is pending
    monkeypatch.setitem(pending, "value", dict(pending["value"], hopf=True))
    assert AN.is_forward_replication_extended("hopf")
    assert AN.forward_replication_seeds("hopf") == AN.REPLICATION_SEEDS_EXTENDED
    assert not AN.is_forward_replication_extended("forward_a_eeg")
