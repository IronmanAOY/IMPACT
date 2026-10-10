# -*- coding: utf-8 -*-
"""Family-B validation v2: the cells of HCv2-2, HCv2-11, HCv2-12 and HCv2-13,
their seed blocks (confirmatory blocks of the seed map, development mirrors
in 400-439), cut-mode clusters and trajectory seeds, the family-B protocols
and exact targets, the label-error and driver constructions, and the
validation script end to end on development seeds (confirmatory runs are
refused before anything is simulated)."""
import json
import re

import numpy as np
import pytest

from impact_pipeline import evidence_v2 as EV
from impact_pipeline.bench.designs_v2 import family_b as FB
from impact_pipeline.v2 import hypothesis_engine as HE
from impact_pipeline.v2 import iim_v5 as IIM
from impact_pipeline.v2 import provenance as PV
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import records as REC
from impact_pipeline.v2 import seeds as S
from scripts.v2 import iim_validation_v2 as V

TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/=@-]*$")


def by_parts(cells, part):
    return [c for c in cells if part in c.parts]


def dev_task(hypothesis, predicate, seed):
    (task,) = [t for t in FB.tasks(S.DEVELOPMENT, hypotheses=[hypothesis], seeds=[seed])
               if predicate(t.cell)]
    return task


def small_ring_task():
    """A three-unit ring scored with the v5 SE (a fast plumbing fixture on a
    development seed of the HCv2-11 mirror)."""
    cell = FB.Cell("HCv2-11", ("smoke",), "HCv2-11", FB.GEN_BINARY, "ring",
                   (("coupling", 0.45), ("n", 3)), 2000, 1,
                   (FB.Scoring(FB.DECL_NONE, se=True),))
    return FB.Task(cell, 415, S.DEVELOPMENT)


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------
def test_family_b_cells_per_hypothesis():
    cells = FB.cells()
    ids = [c.cell_id for c in cells]
    assert len(set(ids)) == len(ids)
    assert all(TOKEN.match(i) for i in ids)
    assert all(c.n_null == 19 for c in cells)  # K = 19 in family-B cells
    assert all(c.cut_modes == ("directional", "bidirectional") for c in cells)

    h2 = FB.cells(["HCv2-2"])
    assert [len(by_parts(h2, p)) for p in ("i", "ii", "iii")] == [4, 3, 2]
    assert sorted(c.n_time for c in by_parts(h2, "i")) == [1000, 3000, 10000, 30000]
    assert sorted(c.n_time for c in by_parts(h2, "ii")) == [3000, 10000, 30000]
    decisive = sum(len(c.cut_modes) for c in h2 for s in c.scorings if s.role == "decisive")
    assert decisive == 18
    for c in by_parts(h2, "ii"):
        assert [s.conditioning for s in c.scorings] == ["stratify"]
    for c in by_parts(h2, "iii"):
        assert c.n_time == 12000 and c.lag == 2
        orders = {s.null_order: s.role for s in c.scorings}
        assert orders == {"shift_then_project": "decisive",
                          "project_then_shift": "reported"}
    assert all(not s.se for c in h2 for s in c.scorings)

    h11 = FB.cells(["HCv2-11"])
    assert len(h11) == 12
    assert sorted({c.param_dict["coupling"] for c in h11}) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert {c.n_time for c in h11} == {10000, 30000}
    assert all(c.system == "feedforward_star" and c.scorings[0].se for c in h11)

    h12 = FB.cells(["HCv2-12"])
    a = by_parts(h12, "a")
    assert len(a) == 24
    assert {(c.n_time, c.role) for c in a} == {(30000, "decisive"), (10000, "reported")}
    assert sorted(c.param_dict["coupling"] for c in by_parts(h12, "b")) == [0.45, 0.9, 1.5]
    for c in by_parts(h12, "c"):
        exp = dict(c.expected)
        assert exp["occupancy"] == FB.occupancy_designation(c.system, c.params, c.n_time)
    d = by_parts(h12, "d")
    assert sorted(dict(c.expected)["reason"] for c in d) == [
        "INSUFFICIENT_OCCUPANCY", "INSUFFICIENT_OCCUPANCY", "MACRO_RANK_DEFICIENT"]
    (a2a,) = [c for c in d if c.system == "all_to_all"]
    assert a2a.parts == ("c", "d") and a2a.n_time == 1000

    h13 = FB.cells(["HCv2-13"])
    assert [c.n_time for c in h13] == [3000, 10000, 30000]
    for c in h13:
        keys = [s.key for s in c.scorings]
        assert keys[:2] == ["recorded", "hidden"]
        assert c.scorings[0].se  # (a): statuses of the recorded driver at every T
        assert c.scorings[1].se == (c.n_time == 30000)  # (b): PRESENT at T = 30000
        if c.n_time == 10000:
            assert keys[2:] == ["label_error=0.1", "label_error=0.25"]
        else:
            assert len(keys) == 2


def test_seed_blocks_match_the_seed_map():
    smap = S.load_seed_map()
    (fb,) = [e for e in smap["confirmatory"]["assignments"] if e["use"] == "family B"]
    parts = {p["use"]: (p["min"], p["max"]) for p in fb["parts"]}
    assert {k: v[S.CONFIRMATORY] for k, v in FB.SEED_BLOCKS.items()} == parts
    sizes = {k: v[S.CONFIRMATORY][1] - v[S.CONFIRMATORY][0] + 1
             for k, v in FB.SEED_BLOCKS.items()}
    assert sizes == {"HCv2-2": 200, "HCv2-11": 40, "HCv2-12(a)": 40, "HCv2-12(b)": 40,
                     "HCv2-12(c, d)": 40, "HCv2-13": 100}
    (dry,) = [e for e in smap["development"]["assignments"]
              if (e["min"], e["max"]) == (400, 439)]
    assert "family-B" in dry["use"]
    dev = sorted(v[S.DEVELOPMENT] for v in FB.SEED_BLOCKS.values())
    assert dev[0][0] == 400 and dev[-1][1] == 439
    for (lo, hi), (lo2, _) in zip(dev, dev[1:]):
        assert hi < lo2  # disjoint mirrors
    for v in FB.SEED_BLOCKS.values():
        assert S.check_seeds(range(v[S.DEVELOPMENT][0], v[S.DEVELOPMENT][1] + 1)) == \
            S.DEVELOPMENT


def test_tasks_respect_the_seed_policy():
    dev = FB.tasks(S.DEVELOPMENT)
    assert dev and all(0 <= t.seed <= 999 and t.split == S.DEVELOPMENT for t in dev)
    plan = FB.plan(S.DEVELOPMENT)
    assert sum(p["tasks"] for p in plan.values()) == len(dev)
    with pytest.raises(S.SeedPolicyError):
        FB.tasks(S.DEVELOPMENT, seeds=[20000])
    with pytest.raises(S.SeedPolicyError):
        FB.tasks(S.CONFIRMATORY, seeds=[400])
    with pytest.raises(S.SeedPolicyError):
        FB.tasks(S.DEVELOPMENT, seeds=[12000])
    # the confirmatory plan is a list of descriptors; nothing is simulated
    conf = FB.tasks(S.CONFIRMATORY, hypotheses=["HCv2-11"])
    assert len(conf) == 12 * 40 and min(t.seed for t in conf) == 20280
    assert FB.plan(S.CONFIRMATORY)["HCv2-2"] == {
        "cells": 9, "cut_mode_cells": 18, "tasks": 1800, "scorings": 4400,
        "blocks": ["HCv2-2"]}
    only = FB.tasks(S.DEVELOPMENT, seeds=[400, 415])
    assert {t.seed for t in only} == {400, 415}
    with pytest.raises(ValueError):
        FB.cells(["HCv2-99"])


def test_cut_mode_cluster_ids_and_trajectory_seeds():
    tasks = FB.tasks(S.DEVELOPMENT)
    t = tasks[0]
    assert t.cluster_id == t.task_id and TOKEN.match(t.task_id)
    want = FB.hash_seed("iim_validation_v2", t.cell.generator, t.cell.system,
                        json.dumps(dict(t.cell.params), sort_keys=True,
                                   separators=(",", ":"), default=str),
                        t.cell.n_time, t.seed)
    assert t.trajectory_seed == want
    assert FB.Task(t.cell, t.seed, t.split).trajectory_seed == t.trajectory_seed
    # every family-B and synthetic task has its own trajectory
    seeds = [x.trajectory_seed for x in tasks
             if x.cell.generator not in (FB.GEN_HYPERSYNCHRONOUS, FB.GEN_HOPF_EEG)]
    assert len(set(seeds)) == len(seeds)
    hyp = [x for x in tasks if x.cell.generator == FB.GEN_HYPERSYNCHRONOUS][0]
    assert hyp.simulation_seed == hyp.seed
    # the declarations of a task share the trajectory, not their null seeds
    h13 = dev_task("HCv2-13", lambda c: c.n_time == 10000, 435)
    nulls = {s.key: h13.scoring_seeds(s) for s in h13.cell.scorings}
    assert len({v[0] for v in nulls.values()}) == len(nulls)


def test_occupancy_designation_rule():
    ring = (("coupling", 0.45),)
    assert FB.expected_rarest_count("ring", ring, 1000) == pytest.approx(12.04, abs=0.01)
    assert FB.occupancy_designation("ring", ring, 1000) == "undefined"
    assert FB.occupancy_designation("ring", ring, 3000) is None
    assert FB.occupancy_designation("ring", ring, 10000) == "defined"
    assert FB.occupancy_designation("ring", (("coupling", 0.9),), 30000) is None
    # raising N_min at calibration moves the designation, deterministically
    assert FB.occupancy_designation("ring", ring, 10000, n_min=50) is None
    c25 = {c.cell_id for c in by_parts(FB.cells(["HCv2-12"]), "c")}
    c50 = {c.cell_id for c in by_parts(FB.cells(["HCv2-12"], n_min=50), "c")}
    assert c25 != c50


# --------------------------------------------------------------------------
# protocols and exact targets
# --------------------------------------------------------------------------
def test_family_b_protocol():
    p = FB.family_b_protocol("directional")
    assert p.schema == EV.PROTOCOL_SCHEMA_V3
    assert p.necessity_set == ("IIM",)
    assert p.rank_gate_for("IIM") == 0.05
    assert p.estimator_version_for("IIM") == IIM.ESTIMATOR_VERSION
    assert set(p.se_methods_for("IIM")) == {"circular_block_bootstrap_10pct_B50",
                                            "exact", "jackknife_contiguous_10"}
    assert p.null_families["IIM"] == "circular_shift"
    ref = p.reference_for("IIM")
    assert ref["reference"] == pytest.approx(FB.anchor_value("directional"))
    assert ref["reference_se"] == 0.0 and ref["reference_scale"] == "excess"
    rebuilt = EV.ProtocolV3.from_dict(json.loads(p.to_json()))
    assert rebuilt.hash == p.hash
    assert p.status_rule.alpha_absent == 0.01
    values = FB.family_b_protocol("directional", sampling_se=False)
    assert values.name.endswith("-values") and values.se_methods_for("IIM") is None
    assert values.hash != p.hash
    strat = FB.family_b_protocol("directional", null_family="stratified_pair_rotation")
    assert strat.null_families["IIM"] == "stratified_pair_rotation"
    second = FB.family_b_protocol("bidirectional", anchor=FB.SECOND_ANCHOR)
    assert second.reference_for("IIM")["reference"] == pytest.approx(
        FB.anchor_value("bidirectional", FB.SECOND_ANCHOR))
    # exact known-TPM values are decided on c alone under the family-B protocol
    for cut in FB.CUT_MODES:
        a = EV.assess_item(IIM.evidence(IIM.exact_result(FB.anchor_value(cut), cut), cut),
                           FB.family_b_protocol(cut))
        assert a.status is EV.PRESENT and a.route == "exact" and a.c == pytest.approx(1)


def test_exact_targets_and_sweeps():
    for (anchor, cut), v in FB.ANCHOR_DESIGN_VALUES.items():
        assert FB.anchor_value(cut, anchor) == pytest.approx(v, abs=5e-6)
    mono = FB.sweep_monotonicity()
    for kind in ("ring", "xor_loop"):
        for cut in FB.CUT_MODES:
            assert mono[kind][cut]["strictly_monotone"]
    cells = {c.cell_id: c for c in FB.cells()}
    h13 = [c for c in cells.values() if c.hypothesis == "HCv2-13"][0]
    rec = FB.exact_targets(h13, h13.scorings[0])
    assert all(abs(rec[c]["value"]) < 1e-12 and rec[c]["target"] == "conditioned"
               for c in FB.CUT_MODES)
    hid = FB.exact_targets(h13, h13.scorings[1])
    assert hid["directional"] is None and hid["bidirectional"]["value"] > 0.01
    star = [c for c in cells.values() if c.system == "feedforward_star"
            and c.param_dict["coupling"] == 1.0][0]
    ex = FB.exact_targets(star)
    assert abs(ex["directional"]["c"]) < 1e-10 and ex["bidirectional"]["c"] > 1.0
    ring09 = [c for c in cells.values() if c.system == "ring"
              and c.param_dict["coupling"] == 0.9 and "b" in c.parts][0]
    assert FB.exact_targets(ring09)["directional"]["value"] == pytest.approx(-0.0097,
                                                                             abs=5e-5)
    assert FB.exact_targets(ring09)["bidirectional"]["value"] == pytest.approx(0.0362,
                                                                               abs=5e-5)
    hopf = [c for c in cells.values() if c.generator == FB.GEN_HOPF_EEG][0]
    assert FB.exact_targets(hopf) == {"directional": None, "bidirectional": None}


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def test_label_errors_are_nested_and_use_stream_41():
    states = np.random.default_rng(1).integers(0, 2, size=20000)
    lo, n_lo = FB.label_error_states(states, 0.1, seed=12345)
    hi, n_hi = FB.label_error_states(states, 0.25, seed=12345)
    flip_lo, flip_hi = lo != states, hi != states
    assert np.all(flip_hi[flip_lo])  # nested dose
    assert n_lo == flip_lo.sum() and n_hi == flip_hi.sum()
    assert abs(n_lo / states.size - 0.1) < 0.01 and abs(n_hi / states.size - 0.25) < 0.015
    u = np.random.default_rng(np.random.SeedSequence([12345, 41])).random(states.size)
    np.testing.assert_array_equal(flip_lo, u < 0.1)
    again, _ = FB.label_error_states(states, 0.1, seed=12345)
    np.testing.assert_array_equal(again, lo)


def test_markov_driver_and_switching_levels():
    d = FB.markov_binary(50000, 0.9, np.random.default_rng(3))
    assert set(np.unique(d)) == {0, 1}
    assert np.mean(d[1:] == d[:-1]) == pytest.approx(0.9, abs=0.01)
    lev = FB.switching_levels(20000, 0.05, np.random.default_rng(4))
    change = np.flatnonzero(np.diff(lev)) + 1
    runs = np.diff(np.r_[0, change, lev.size])[:-1]
    assert set(np.unique(lev)) == set(range(6))
    # dwells of 1-3 s (20-60 samples); a redrawn level merges two dwells, so
    # the mean run is 40 / (5/6) = 48 samples
    assert runs.min() >= 20 and 38 < runs.mean() < 60


@pytest.mark.parametrize("kind", FB.DRIVER_KINDS)
def test_ar1_driver_lies_inside_the_basis_span(kind):
    """The recorded driver of the residualised calibration cells is inside the
    span of the common input basis (lags 0-2, filtered copies at 0.1, 0.3 and
    1.0 s): residualising the drive itself leaves almost nothing."""
    y, rec = FB.ar1_driver_system(kind, seed=402)
    assert y.shape == (4, FB.AR1_N_TIME)
    sim = FB.Simulation(ts=y, lag=FB.AR1_LAG, dt=FB.AR1_DT, seed=402,
                        driver_states=rec.get("states"),
                        driver_channel=rec.get("channel"))
    basis = FB.input_basis(sim)
    assert basis.lags == (0, 1, 2) and basis.taus == (0.1, 0.3, 1.0)
    s0 = basis.first_complete_sample
    span = basis.orthonormal_span(s0)
    drive = rec["drive"][:, s0:]
    res = IIM.residualise(drive, span)
    # only the zero-start transient of the slowest filter (a few seconds) is
    # outside the span
    assert np.all(res.var(axis=1) / drive.var(axis=1) < 5e-3)
    tail = slice(1000, None)
    assert np.all(res[:, tail].var(axis=1) / drive[:, tail].var(axis=1) < 1e-3)
    assert np.all(np.abs(np.asarray(rec["coefficients"]) - 0.7) <= 0.2)


# --------------------------------------------------------------------------
# the script
# --------------------------------------------------------------------------
def gate_tasks():
    return [dev_task("HCv2-12", lambda c: c.system == "all_to_all"
                     and c.param_dict.get("coupling") == 0.6 and c.n_time == 1000, 430),
            dev_task("HCv2-12", lambda c: c.system == "ring"
                     and c.param_dict.get("coupling") == 0.9 and c.n_time == 1000, 430)]


def test_run_end_to_end_and_resume(tmp_path):
    tasks = gate_tasks() + [small_ring_task()]
    out = tmp_path / "fb"
    res = V.run(out, split=S.DEVELOPMENT, task_list=tasks, write_exact=False)
    recs = {r.task_id: r for r in res["records"]}
    assert set(recs) == {t.task_id for t in tasks}
    for t in tasks:
        rec = recs[t.task_id]
        assert rec.status == REC.TASK_OK and rec.split == S.DEVELOPMENT
        assert REC.loads(REC.dumps(rec)).to_dict() == rec.to_dict()
        assert rec.config["cluster_id"] == t.task_id
        assert rec.config["trajectory_seed"] == t.trajectory_seed
        # the cost of the task: its wall and CPU seconds
        assert rec.timing["cpu_s"] >= 0 and rec.timing["total_s"] > 0
        # one scoring per declaration and cut mode, in one cluster
        assert sorted(s.scoring_id for s in rec.scorings) == sorted(
            f"{sc.key}/{cut}" for sc in t.cell.scorings for cut in FB.CUT_MODES)
        for s in rec.scorings:
            comp = s.components["IIM"]
            assert comp.details["cluster_id"] == t.task_id
            assert comp.estimator_version == IIM.ESTIMATOR_VERSION
            proto = V.protocol_for(t.cell.scorings[0], s.estimator_form,
                                   {"null_family": comp.null_family})
            assert comp.protocol_hash == proto.hash
            # the record names its protocol by the key the evaluator and the
            # audit look protocols up by
            assert comp.protocol_id == s.protocol_id == HE.protocol_key(proto)
            assert comp.protocol_id == proto.name[len("mpc-bench-v2-"):]
    meta = json.loads((out / V.SUMMARY_JSON).read_text())
    for key, entry in meta["protocols"].items():
        assert HE.protocol_key_of_name(entry["name"]) == key
    for t in gate_tasks():
        for s in recs[t.task_id].scorings:
            comp = s.components["IIM"]
            assert comp.status == R.UNDEFINED and comp.reason == R.INSUFFICIENT_OCCUPANCY
            assert comp.details["defined"] is False
    ring = recs[small_ring_task().task_id]
    for s in ring.scorings:
        comp = s.components["IIM"]
        assert comp.se_method == "circular_block_bootstrap_10pct_B50"
        assert comp.se_df == IIM.BOOTSTRAP_SE_DF
        assert comp.status in (R.PRESENT, R.ABSENT, R.UNDEFINED) and comp.c is not None
        assert comp.reason not in (R.INVALID_SE, R.NO_SAMPLING_SE)
        assert 0 < comp.details["p_ind"] <= 1
        assert comp.details["second_anchor"]["anchor"] == FB.SECOND_ANCHOR
    comps = res["components"]
    assert len(comps) == sum(len(r.scorings) for r in recs.values())
    for col in ("hypothesis", "cell_id", "cluster_id", "p_ind", "c_exact", "status"):
        assert col in comps.columns
    meta = json.loads((out / V.SUMMARY_JSON).read_text())
    assert meta["split"] == S.DEVELOPMENT and meta["n_records"] == 3
    assert meta["provenance"]["split"] == S.DEVELOPMENT
    assert meta["provenance"]["code"]["confirmatory"] is False
    assert all(len(v["hash"]) == 64 for v in meta["protocols"].values())
    assert (out / V.COMPONENTS_CSV).exists() and (out / V.SUMMARY_CSV).exists()
    again = V.run(out, split=S.DEVELOPMENT, task_list=tasks, write_exact=False)
    assert again["summary"]["n_run"] == 0 and len(again["records"]) == 3


def test_records_carry_provenance_and_the_plan_is_written(tmp_path):
    from scripts.v2 import integrity_audit as IA

    task = small_ring_task()
    out = tmp_path / "fb"
    res = V.run(out, split=S.DEVELOPMENT, task_list=[task], write_exact=False)
    (rec,) = res["records"]
    # the code identity of the run, in the fields of the v2 runner's records
    prov = rec.provenance
    assert prov["git_sha"] == res["summary"]["provenance"]["code"]["git_sha"]
    assert prov["confirmatory"] is False and prov["freeze_tag"] is None
    assert set(prov) == {"runner_version", "git_sha", "git_dirty", "trees",
                         "confirmatory", "freeze_tag", "freeze_tag_commit", "label"}
    # the plan of task ids, read by the integrity audit (IA-7)
    plan = out / V.PLAN_JSON
    assert IA._load_plan(plan) == [task.task_id]
    assert IA._load_smoke([plan]) == []
    recs, bad = IA.read_raw_records([out])
    check = IA.ia7_plan_and_seeds(recs, IA._load_plan(plan), S.DEVELOPMENT, bad)
    assert check["status"] == IA.PASS and check["details"]["plan_checked"]


def test_resume_drops_an_interrupted_line_and_failed_tasks(tmp_path):
    from scripts.v2 import integrity_audit as IA

    task = small_ring_task()
    out = tmp_path / "fb"
    V.run(out, split=S.DEVELOPMENT, task_list=[task], write_exact=False)
    jsonl = out / V.RESULTS_JSONL
    good = jsonl.read_text(encoding="utf-8")
    # a run cut off while writing a record
    jsonl.write_text(good + '{"task_id": "B:cut', encoding="utf-8")
    again = V.run(out, split=S.DEVELOPMENT, task_list=[task], write_exact=False)
    assert again["summary"]["n_run"] == 0
    assert jsonl.read_text(encoding="utf-8") == good
    # a failed task runs again and its error record leaves the file
    err = REC.TaskRecord(task_id=task.task_id, design=FB.DESIGN, family=FB.FAMILY,
                         system=task.cell.system, seed=int(task.seed),
                         generator_version=FB.GENERATOR_VERSION,
                         status=REC.TASK_ERROR, split=task.split,
                         config={"cell": task.cell.to_dict()}, error="planted")
    jsonl.write_text(REC.dumps(err) + "\n", encoding="utf-8")
    third = V.run(out, split=S.DEVELOPMENT, task_list=[task], write_exact=False)
    assert third["summary"]["n_run"] == 1
    recs, bad = IA.read_raw_records([jsonl])
    assert bad == [] and [r["status"] for r in recs] == [REC.TASK_OK]
    # a confirmatory output directory keeps its plan
    other = small_ring_task()
    (tmp_path / "p").mkdir()
    V.write_plan(tmp_path / "p", [other], S.CONFIRMATORY)
    with pytest.raises(ValueError, match="another plan"):
        V.write_plan(tmp_path / "p", [other, *gate_tasks()], S.CONFIRMATORY)


def test_a_raising_scoring_leaves_the_others_intact(tmp_path, monkeypatch):
    task = dev_task("HCv2-13", lambda c: c.n_time == 3000, 435)
    real = IIM.compute_iim_v5

    def flaky(*a, **k):
        if k.get("strata") is None:  # the hidden-driver scoring
            raise RuntimeError("planted failure")
        return real(*a, **k)

    monkeypatch.setattr(IIM, "compute_iim_v5", flaky)
    rec = V.run_task(task)
    assert rec.status == REC.TASK_OK_WITH_COMPONENT_ERRORS
    by = {s.scoring_id: s.components["IIM"] for s in rec.scorings}
    for cut in FB.CUT_MODES:
        assert by[f"hidden/{cut}"].reason == "ESTIMATOR_ERROR:RuntimeError"
        assert by[f"recorded/{cut}"].reason != "ESTIMATOR_ERROR:RuntimeError"
        assert by[f"recorded/{cut}"].status == R.UNDEFINED  # gated at T = 3000


def test_confirmatory_run_is_guarded_before_anything_runs(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("nothing may be simulated")

    monkeypatch.setattr(FB, "simulate", forbidden)
    monkeypatch.setattr(V, "run_task", forbidden)
    conf = FB.tasks(S.CONFIRMATORY, hypotheses=["HCv2-12"], seeds=[20460])
    out = tmp_path / "conf"
    with pytest.raises(PV.ConfirmatoryGuardError):
        V.run(out, split=S.CONFIRMATORY, task_list=conf,
              freeze_tag="mpcbench-no-such-tag")
    assert not out.exists()
    with pytest.raises(S.SeedPolicyError):
        V.run(out, split=S.DEVELOPMENT, task_list=conf)
    with pytest.raises(S.SeedPolicyError):
        V.run(out, split=S.CONFIRMATORY, task_list=gate_tasks())
    assert not out.exists()
    # the command line refuses a development run on confirmatory seeds
    assert V.main(["--out", str(out), "--seeds", "20460"]) == 2
    assert not out.exists()


def test_exact_table_and_the_plan_listing(capsys):
    tab = V.exact_table(["HCv2-11"])
    anchors = tab[tab["row"] == "anchor"]
    assert len(anchors) == 4
    cells = tab[tab["row"] == "cell"]
    star_dir = cells[(cells["system"] == "feedforward_star/coupling=1")
                     & (cells["cut_mode"] == "directional")].iloc[0]
    assert star_dir["exact_status"] == "ABSENT" and star_dir["route"] == "exact"
    star_bid = cells[(cells["system"] == "feedforward_star/coupling=1")
                     & (cells["cut_mode"] == "bidirectional")].iloc[0]
    assert star_bid["exact_status"] == "PRESENT"
    sweeps = tab[tab["row"] == "sweep"]
    assert sweeps["strictly_monotone"].astype(bool).all() and len(sweeps) == 4
    assert V.main(["--list"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert set(plan) == set(FB.HYPOTHESES)
    assert plan["HCv2-13"]["tasks"] == 15
