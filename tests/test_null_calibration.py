# -*- coding: utf-8 -*-
"""
The null-calibration design: the null systems and their seeds; the cell
grid and the seed base of each split; the hub partition (first quarter of
the nodes, three equal periphery blocks) as the estimators read it; the
agency-event path of SRPI equals the in-memory path; the classification
protocol A-none (A-R with the declaration none); the records carry what the
evaluator reads; the script runs and summarises a development grid.

Only development seeds are simulated (seed base 400); confirmatory tasks are
built as plan entries and never run.
"""
import dataclasses
import json
import math
import os
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as t_dist

from impact_pipeline import evidence as E
from impact_pipeline.bench import designs as D
from impact_pipeline.bench import export as X
from impact_pipeline.bench import hypotheses as HE
from impact_pipeline.bench import records as REC
from impact_pipeline.bench import runner as RB
from impact_pipeline.bench import seeds as S
from impact_pipeline.bench.designs import null_calibration as NCV

import scripts.null_calibration as CLI

EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                       "hypotheses_example.json")

CONF, DEV = D.CONFIRMATORY, D.DEVELOPMENT
DEV_BASE = 400


def _dev_task(kind="ar1", n_time=1200, n_nodes=8, rep=0):
    return NCV.cell_task(kind, n_time, n_nodes, rep, DEV_BASE)


def _plain_system(kind="ar1", n_time=1200, n_nodes=8, rep=0):
    """The null system of a replicate without the hub partition."""
    return NCV.null_system(kind, n_nodes, n_time, NCV._seed(DEV_BASE, kind, n_time,
                                                            n_nodes, rep))


def _run(task, principles):
    protos = RB.resolve_protocols(RB.protocol_keys([task]))
    rec = RB.run_task(task, protos, RB.RunSettings(principles=tuple(principles)))
    assert rec.status in (REC.TASK_OK, REC.TASK_OK_WITH_COMPONENT_ERRORS), rec.error
    return rec


# --------------------------------------------------------------------------
# null systems
# --------------------------------------------------------------------------
def test_a_design_system_holds_the_replicate_recording_and_events():
    for kind, n_time, n_nodes, rep in (("ar1", 1200, 8, 0), ("pink", 1200, 16, 3),
                                       ("surrogate_iid", 2400, 8, 1),
                                       ("surrogate_linear", 1200, 8, 7)):
        task = _dev_task(kind, n_time, n_nodes, rep)
        system = NCV.build_null_system(task)
        plain = _plain_system(kind, n_time, n_nodes, rep)
        assert system.ts.shape == (n_nodes, n_time)
        assert np.array_equal(system.ts, plain.ts)
        pd.testing.assert_frame_equal(system.events, plain.events)
        want = NCV._seed(DEV_BASE, kind, n_time, n_nodes, rep)
        assert system.meta["seed"] == plain.meta["seed"] == want
        assert system.meta["family"] == "null_calibration"
        assert system.meta["null_kind"] == kind
        assert (system.meta["seed_base"] == DEV_BASE
                and system.meta["null_replicate"] == rep)


def test_the_replicate_seed_is_pinned():
    # the SHA-256 derivation of the replicate seed (no label string)
    assert NCV._seed(400, "ar1", 1200, 8, 0) == int(
        __import__("hashlib").sha256(b"400|ar1|1200|8|0").hexdigest()[:8], 16)


# --------------------------------------------------------------------------
# cell grid and seed base
# --------------------------------------------------------------------------
@pytest.mark.parametrize("split,base,n_rep", [(CONF, 20000, 50), (DEV, 400, 8)])
def test_cell_grid_and_seed_base(split, base, n_rep):
    design = D.get_design(NCV.DESIGN)
    assert design.module == NCV.MODULE and design.family == "null"
    tasks = design.tasks(split)
    assert len(tasks) == 16 * n_rep == design.expected_tasks[split]
    assert len({t.task_id for t in tasks}) == len(tasks)
    cells = {}
    for t in tasks:
        cells.setdefault((t.tags["null_kind"], t.tags["n_time"], t.tags["n_nodes"]),
                         []).append(t.tags["null_replicate"])
        assert t.seed == base and t.params["seed_base"] == base
        assert t.split == split and t.replicate == 0
        assert t.design == "null_calibration" and t.system == t.tags["null_kind"]
        assert t.builder == NCV.BUILDER and t.design_module == NCV.MODULE
        assert t.tags["cell"] == f"{t.system}:T{t.tags['n_time']}:N{t.tags['n_nodes']}"
    assert set(cells) == {(k, t, n) for k in ("ar1", "pink", "surrogate_iid",
                                              "surrogate_linear")
                          for t in (1200, 2400) for n in (8, 16)}
    assert all(sorted(r) == list(range(n_rep)) for r in cells.values())
    # the runner plans the same tasks and the plan table matches the document
    assert D.build_plan([NCV.DESIGN], split) == tasks
    (row,) = D.plan_table(split, [NCV.DESIGN])
    assert row["matches"] is True and row["n_seeds"] == 1
    assert set(row["ademp"]) == set(D.ADEMP_KEYS)


def test_scorings_follow_the_applicability_with_declaration_none():
    for t in NCV.cells(DEV, replicates=1):
        (s,) = t.scorings
        assert (s.declaration_id, s.protocol_key, s.view, s.estimator_form) == (
            "none", "A-none", "source", "primary")
        assert s.verdict is False and s.held_out is False
        assert set(s.principles) == set(NCV.NULL_KINDS[t.system])
    lin = NCV.cells(DEV, kinds=["surrogate_linear"], replicates=1)[0]
    assert lin.scorings[0].principles == ("RAM", "PDI", "SRPI")


def test_system_seeds_are_derivations_of_the_seed_base():
    seeds = {NCV.system_seed(DEV_BASE, k, t, n, r) for k in NCV.KINDS
             for t in NCV.N_TIMES for n in NCV.N_NODES for r in range(8)}
    assert len(seeds) == 16 * 8
    assert NCV.system_seed(DEV_BASE, "pink", 2400, 16, 5) == NCV._seed(
        DEV_BASE, "pink", 2400, 16, 5)
    # the plan carries the base, never a derived seed
    t = _dev_task("pink", 2400, 16, 5)
    assert t.seed == DEV_BASE and "seed" not in t.tags and "seed" not in t.params


def test_builder_options_and_refusals():
    t = NCV.cells(DEV, seeds=[401, 402], systems=["pink"], n_times=[1200],
                  n_nodes=[16], replicates=3)
    assert len(t) == 2 * 3 and {x.seed for x in t} == {401, 402}
    assert {x.system for x in t} == {"pink"}
    # the runner passes --seeds and --systems to the builder
    assert len(D.build_plan([NCV.DESIGN], DEV, seeds=[400], systems=["ar1"])) == 4 * 8
    with pytest.raises(NCV.NullCalibrationError, match="not a development seed"):
        NCV.cells(DEV, seeds=[20000])
    with pytest.raises(NCV.NullCalibrationError, match="unknown null kinds"):
        NCV.cells(DEV, systems=["white"])
    with pytest.raises(NCV.NullCalibrationError, match="replicates"):
        NCV.cells(DEV, replicates=0)
    with pytest.raises(NCV.NullCalibrationError, match="multiple of 4"):
        NCV.cells(DEV, n_nodes=[10])
    with pytest.raises(D.DesignError, match="seed"):
        NCV.cell_task("ar1", 1200, 8, 0, 1500)  # neither split
    bad = dataclasses.replace(_dev_task(), replicate=1)
    with pytest.raises(NCV.NullCalibrationError, match="no twins"):
        NCV.build_null_system(bad)
    bad = dataclasses.replace(_dev_task(), seed=401)
    with pytest.raises(NCV.NullCalibrationError, match="seed base"):
        NCV.build_null_system(bad)
    with pytest.raises(NCV.NullCalibrationError, match="twice"):
        NCV.cells(DEV, systems=["ar1"], kinds=["pink"])
    assert NCV.cells(DEV, systems=["ar1"], kinds=["ar1"], replicates=1) == NCV.cells(
        DEV, kinds=["ar1"], replicates=1)


def test_seed_bases_outside_the_split_are_refused_before_any_task(
        monkeypatch, tmp_path):
    # the unused block and the confirmatory block never reach a development plan,
    # and nothing is derived or simulated on the way to the refusal
    def refuse(*args, **kwargs):
        raise AssertionError("a seed was derived or a system built")

    monkeypatch.setattr(NCV, "_seed", refuse)
    monkeypatch.setattr(NCV, "null_system", refuse)
    for base in (10000, 15000, 19999):
        with pytest.raises(S.SeedPolicyError):
            NCV.cells(DEV, seeds=[base])
    with pytest.raises(NCV.NullCalibrationError, match="not a development seed"):
        NCV.cells(DEV, seeds=[400, 20000])
    with pytest.raises(NCV.NullCalibrationError, match="not a confirmatory seed"):
        NCV.cells(CONF, seeds=[400])
    for base in ("15000", "20000"):
        out = tmp_path / f"run{base}"
        assert CLI.main(["run", "--out", str(out), "--seed-base", base, "--kinds",
                         "ar1", "--T", "1200", "--nodes", "8", "--replicates",
                         "1"]) == 2
        assert not out.exists()


def test_the_confirmatory_plan_derives_no_seed_and_builds_no_system(
        monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("a confirmatory seed was derived or a system built")

    monkeypatch.setattr(NCV, "_seed", refuse)
    monkeypatch.setattr(NCV, "null_system", refuse)
    tasks = NCV.cells(CONF)
    assert len(tasks) == 800 and {t.seed for t in tasks} == {20000}
    keys = {"kind", "n_time", "n_nodes", "null_replicate", "seed_base", "dt",
            "iim_macro_nodes", "ar_coef", "pink_beta"}
    assert all(set(t.params) == keys for t in tasks)
    assert {t.params["seed_base"] for t in tasks} == {20000}
    assert CLI.main(["plan", "--split", "confirmatory"]) == 0
    assert "16 cells, 800 tasks" in capsys.readouterr().out


# --------------------------------------------------------------------------
# the hub partition
# --------------------------------------------------------------------------
def test_hub_partition_is_the_first_quarter_and_three_equal_blocks():
    p8 = NCV.hub_partition(8)
    assert p8["workspace_nodes"] == [0, 1]
    assert p8["modules"] == {"hub": [0, 1], "P1": [2, 3], "P2": [4, 5], "P3": [6, 7]}
    p16 = NCV.hub_partition(16)
    assert p16["modules"] == {"hub": [0, 1, 2, 3], "P1": [4, 5, 6, 7],
                              "P2": [8, 9, 10, 11], "P3": [12, 13, 14, 15]}
    assert p16["module_order"] == ["hub", "P1", "P2", "P3"]
    for n in (8, 12, 16, 32):
        mods = NCV.hub_partition(n)["modules"]
        assert sorted(i for v in mods.values() for i in v) == list(range(n))
        assert {len(v) for v in mods.values()} == {n // 4}
    for n in (4, 6, 10, 18):
        with pytest.raises(NCV.NullCalibrationError, match="multiple of 4"):
            NCV.hub_partition(n)


def test_iim_grain_rule_outside_the_hub():
    # the rule: k equal groups of the nodes outside the workspace, at most
    # one group per node and at least two groups
    macro, grain = NCV.iim_macro_nodes(16, [0, 1, 2, 3])
    assert grain == "null_macro_4"
    assert list(macro.values()) == [[4, 5, 6], [7, 8, 9], [10, 11, 12],
                                    [13, 14, 15]]
    macro, grain = NCV.iim_macro_nodes(8, [0, 1], k=10)
    assert grain == "null_macro_6" and len(macro) == 6
    macro, grain = NCV.iim_macro_nodes(8, [0, 1], k=1)
    assert grain == "null_macro_2" and list(macro.values()) == [[2, 3, 4],
                                                                [5, 6, 7]]


@pytest.mark.parametrize("n_nodes", [8, 16])
def test_estimators_read_the_hub_partition(n_nodes):
    from impact_pipeline.estimators import nas_declared

    system = NCV.build_null_system(_dev_task("ar1", 1200, n_nodes))
    part = NCV.hub_partition(n_nodes)
    hub_name, hub, blocks = nas_declared.declared_blocks(system.meta)
    assert (hub_name, hub) == ("hub", part["workspace_nodes"])
    assert dict(blocks) == {k: v for k, v in part["modules"].items() if k != "hub"}
    # PDI's content bearer leaves the hub out; IIM's macro nodes lie outside it
    view = X.bearer_view(system, "PDI")
    assert view["workspace_nodes"] == part["workspace_nodes"]
    macro = system.meta["iim_macro_nodes"]
    assert len(macro) == 4 and system.meta["iim_grain"] == "null_macro_4"
    outside = sorted(i for v in macro.values() for i in v)
    assert outside == list(range(n_nodes // 4, n_nodes))
    plain = _plain_system("ar1", 1200, n_nodes)
    if n_nodes == 8:
        # with 8 nodes the hub and the IIM grain are those of the plain system
        assert plain.meta["workspace_nodes"] == part["workspace_nodes"]
        assert plain.meta["iim_macro_nodes"] == macro
    else:
        assert plain.meta["workspace_nodes"] == [0, 1, 2]


def test_nas_declared_runs_on_the_hub_and_the_three_blocks():
    rec = _run(_dev_task("pink", 1200, 8, 2), ["NAS"])
    (sc,) = rec.scorings
    nas = sc.components["NAS"]
    est = nas.details["estimator"]
    assert est["hub_name"] == "hub" and est["hub_nodes"] == [0, 1]
    assert est["blocks"] == {"P1": [2, 3], "P2": [4, 5], "P3": [6, 7]}
    assert nas.identifiability["shared_inputs"] == "none"
    # declaration none: no input basis to condition on
    assert est["basis"]["declaration"] == "none" and est["basis"]["n_columns"] == 0


# --------------------------------------------------------------------------
# the agency-event path
# --------------------------------------------------------------------------
def test_agency_events_are_the_plain_systems_and_satisfy_the_contract():
    from impact_pipeline.event_parsing import (
        events_table_to_bundle,
        validate_srpi_agency_contract,
    )

    for kind, n_time in (("ar1", 1200), ("surrogate_linear", 2400)):
        system = NCV.build_null_system(_dev_task(kind, n_time, 8, 4))
        plain = _plain_system(kind, n_time, 8, 4)
        b2 = events_table_to_bundle(system.events)
        b1 = events_table_to_bundle(plain.events)
        a2, a1 = b2["agency_events"], b1["agency_events"]
        assert a2 is not None
        assert json.dumps(a2, sort_keys=True, default=str) == json.dumps(
            a1, sort_keys=True, default=str)
        rep = validate_srpi_agency_contract(a2, tr=system.dt)
        assert rep["valid"] and rep["n_pairs"] > 0, rep["violations"]


def test_srpi_takes_the_in_memory_path():
    task = _dev_task("surrogate_linear", 1200, 8, 1)
    rec = _run(task, ["SRPI"])
    srpi = rec.scorings[0].components["SRPI"]
    assert srpi.estimator_version == "srpi-2026.09"
    est = srpi.details["estimator"]
    assert est["path"] == "impact_pipeline.bench.export.run_in_memory"
    assert est["estimator_modes"] == {"mode": "agency", "agency_events": "bundle"}
    # the in-memory path on the plain system, with the runner's null seed,
    # null size and jackknife groups, gives the same component
    direct = X.run_in_memory(_plain_system("surrogate_linear", 1200, 8, 1),
                             metrics=["SRPI"],
                             null_surrogates=RB.IN_MEMORY_NULL_SURROGATES,
                             null_seed=RB.null_seed(DEV_BASE),
                             se_groups=RB.IN_MEMORY_SE_GROUPS)["components"]["SRPI"]
    assert np.isfinite(direct["estimate"])
    assert srpi.estimate == direct["estimate"]
    assert (srpi.null_mean, srpi.null_sd, srpi.n_null) == (
        direct["null_mean"], direct["null_sd"], direct["n_null"])
    assert srpi.se == direct["se"] and srpi.null_family == direct["null_family"]
    # the agency fallback is not triggered, and the summary agrees
    assert CLI.AGENCY_FIX_REASON not in str(direct.get("reason") or "")
    assert not CLI.agency_fix_needed(srpi)


def test_the_summary_flags_a_component_with_the_agency_fallback_trigger():
    trigger = f"agency_contract_violation:{CLI.AGENCY_FIX_REASON}"
    for details, reason in (({"estimator_reason": trigger}, None),
                            ({"estimator": {"reason": trigger}}, None),
                            (None, trigger)):
        comp = types.SimpleNamespace(details=details, reason=reason)
        assert CLI.agency_fix_needed(comp), (details, reason)
    for details, reason in (({}, None), (None, "INVALID_ANCHORS"),
                            ({"estimator_reason": "undefined"}, None)):
        comp = types.SimpleNamespace(details=details, reason=reason)
        assert not CLI.agency_fix_needed(comp), (details, reason)


# --------------------------------------------------------------------------
# the classification protocol A-none
# --------------------------------------------------------------------------
def test_classification_protocol_is_a_r_with_declaration_none():
    a_r = RB.draft_protocol("A-R").to_dict()
    a_none = NCV.classification_protocol(a_r)
    assert a_none["shared_inputs_declaration"] == {"id": "none",
                                                   "shared_inputs": "none"}
    assert a_none["name"] == "mpc-bench-A-none-draft"
    same = {k for k in a_r if k not in ("name", "shared_inputs_declaration")}
    assert {k: a_none[k] for k in same} == {k: a_r[k] for k in same}
    proto = E.Protocol.from_dict(a_none)
    assert proto.hash != RB.draft_protocol("A-R").hash
    with pytest.raises(NCV.NullCalibrationError, match="declaration R"):
        NCV.classification_protocol(RB.draft_protocol("A-H").to_dict())
    for name, want in (("A-R", "A-none"), ("mpc-bench-A-R", "mpc-bench-A-none"),
                       ("other", "other:A-none")):
        assert NCV.classification_protocol({**a_r, "name": name})["name"] == want


def test_classification_protocol_copies_its_base_and_refuses_non_a_r(tmp_path):
    a_r = RB.draft_protocol("A-R").to_dict()
    before = json.dumps(a_r, sort_keys=True)
    a_none = NCV.classification_protocol(a_r)
    assert json.dumps(a_r, sort_keys=True) == before
    a_none["estimators"]["NAS"]["n_null"] = -1
    assert json.dumps(a_r, sort_keys=True) == before
    # no declaration, or A-none itself, is not an A-R protocol
    with pytest.raises(NCV.NullCalibrationError, match="declaration R"):
        NCV.classification_protocol({k: v for k, v in a_r.items()
                                     if k != "shared_inputs_declaration"})
    with pytest.raises(NCV.NullCalibrationError, match="declaration R"):
        NCV.classification_protocol(NCV.classification_protocol(a_r))
    # the script refuses to write A-none from another family protocol
    RB.draft_protocol("A-H").to_json(tmp_path / "a_h.json")
    out = tmp_path / "A-none.json"
    assert CLI.main(["protocol", "--base", str(tmp_path / "a_h.json"),
                     "--out", str(out)]) == 2
    assert not out.exists()


def test_the_shipped_a_none_is_the_script_output_without_the_reported_cut(tmp_path):
    out = tmp_path / "A-none.json"
    assert CLI.main(["protocol", "--base", str(RB.protocol_path("A-R")),
                     "--out", str(out)]) == 0
    written = E.Protocol.from_json(out).to_dict()
    assert written["estimators"]["IIM"]["report_cut_modes"]
    written["estimators"]["IIM"]["report_cut_modes"] = []
    assert written == E.Protocol.from_json(RB.protocol_path("A-none")).to_dict()


def test_a_fresh_runner_resolves_the_draft_in_plan_order(repo_root, tmp_path):
    # importing the design module (building the plan) registers nothing in
    # the runner; collecting the plan's protocol keys registers the plan's
    # design modules, so their drafts resolve in a fresh process (from an
    # empty protocol directory: the repository's protocol files would
    # be found first). A runner started with src/ only (as the run scripts
    # start it) builds the null systems without any script
    no_protocols = tmp_path / "no_protocols"
    no_protocols.mkdir()
    code = (
        "import sys\n"
        "from impact_pipeline.bench import runner as RB\n"
        "assert 'scripts' not in sys.modules\n"
        "tasks = RB.build_tasks(['null_calibration'], 'development')\n"
        "assert 'impact_pipeline.bench.designs.null_calibration' in sys.modules\n"
        "assert 'A-none' not in RB.PROTOCOL_DRAFTS\n"
        f"protos = RB.resolve_protocols(RB.protocol_keys(tasks), directory={str(no_protocols)!r})\n"
        "print(sorted(protos), protos['A-none'].source,\n"
        "      dict(protos['A-none'].protocol.shared_inputs_declaration)['id'])\n"
        "print(RB.check_plan(tasks, protos)['n_tasks'])\n"
        "RB.SYSTEM_BUILDERS['null_calibration'](tasks[0])\n"
        "print('scripts' in sys.modules)\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(repo_root / "src"), PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "['A-none'] draft none"
    assert lines[1] == "128"
    assert lines[2] == "False"


def test_a_fresh_runner_command_line_runs_the_design(repo_root, tmp_path):
    # the run command of a fresh process registers the design module before
    # it resolves the plan's protocols: the A-none draft is found without
    # any registration at import (one task of the plan, one principle)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(repo_root / "src"), PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1")
    out = tmp_path / "run"
    no_protocols = tmp_path / "no_protocols"
    no_protocols.mkdir()
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_bench.py"), "run",
         NCV.DESIGN, "--split", "development", "--n-shards", "128",
         "--shard-index", "0", "--principles", "RAM", "--out", str(out),
         "--protocol-dir", str(no_protocols)],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    man = json.loads((out / RB.MANIFEST).read_text())
    assert man["n_run"] == 1 and man["n_errors"] == 0
    assert man["protocols"]["A-none"]["source"] == "draft"
    (rec,) = REC.read_jsonl(out / RB.RESULTS_JSONL)
    assert rec.design == NCV.DESIGN
    assert {c.protocol_id for s in rec.scorings for c in s.components.values()} == {
        NCV.PROTOCOL_KEY}


def test_collecting_the_protocol_keys_registers_the_draft(monkeypatch, tmp_path):
    # (the fresh-process test above shows that the import registers nothing)
    monkeypatch.setattr(RB, "PROTOCOL_DIR", tmp_path)
    monkeypatch.setattr(RB, "PROTOCOL_DRAFTS", {})
    monkeypatch.setattr(RB, "_LOADED_MODULES", set())
    keys = RB.protocol_keys([_dev_task()])
    assert keys == [NCV.PROTOCOL_KEY]
    assert set(RB.PROTOCOL_DRAFTS) == {NCV.PROTOCOL_KEY}
    assert RB.resolve_protocols(keys)[NCV.PROTOCOL_KEY].source == "draft"


def test_the_runner_plan_check_accepts_the_design(monkeypatch, tmp_path):
    monkeypatch.setattr(RB, "PROTOCOL_DIR", tmp_path)
    tasks = D.build_plan([NCV.DESIGN], DEV)
    protos = RB.resolve_protocols(RB.protocol_keys(tasks))
    summary = RB.check_plan(tasks, protos)
    assert summary["n_tasks"] == 128 and summary["n_scorings"] == 128
    assert summary["protocols"]["A-none"]["source"] == "draft"


def test_the_draft_follows_an_a_r_protocol_file(tmp_path, monkeypatch):
    a_r = RB.draft_protocol("A-R").to_dict()
    a_r["name"] = "A-R"
    a_r["reference"] = {"kind": "external", "scale": "excess",
                        "values": {"RAM": 1.0, "PDI": 1.0, "IIM": 1.0, "SRPI": 1.0,
                                   "NAS:receive": 1.0, "NAS:return": 1.0}}
    E.Protocol.from_dict(a_r).to_json(tmp_path / "A-R.json")
    monkeypatch.setattr(RB, "PROTOCOL_DIR", tmp_path)
    draft = NCV.draft_protocol()
    assert draft["name"] == "A-none" and draft["reference"] == E.Protocol.from_dict(
        a_r).to_dict()["reference"]
    # a run without drafts needs the protocol file, which the script writes
    with pytest.raises(RB.RunPolicyError, match="no protocol file"):
        RB.resolve_protocols(["A-none"], directory=tmp_path, allow_drafts=False)
    assert CLI.main(["protocol", "--base", str(tmp_path / "A-R.json"),
                     "--out", str(tmp_path / "A-none.json")]) == 0
    got = RB.resolve_protocols(["A-none"], directory=tmp_path, allow_drafts=False)
    proto = got["A-none"].protocol
    assert got["A-none"].source == "file"
    assert proto.name == "A-none" and proto.shared_inputs_declaration["id"] == "none"
    assert proto.to_dict() == E.Protocol.from_dict(draft).to_dict()


# --------------------------------------------------------------------------
# what the evaluator reads
# --------------------------------------------------------------------------
def test_records_carry_the_names_the_evaluator_reads():
    spec = HE.load_spec(EXAMPLE)
    assert spec["vocabulary"]["design.null_calibration"] == NCV.DESIGN
    assert spec["fields"]["null_kind"] == f"config.tags.{NCV.NULL_KIND_TAG}"
    # the example's null part selects this design and admits its declaration
    part = next(h for h in spec["hypotheses"] if h["id"] == "null_present")["parts"][0]
    assert "@design.null_calibration" in json.dumps(part)
    assert NCV.DECLARATION in part["data"]["where"]["declaration_id"]

    tasks = [_dev_task("ar1", 1200, 8, 0), _dev_task("surrogate_linear", 2400, 16, 1)]
    recs = [_run(t, ["RAM"]).to_dict() for t in tasks]
    for rec in recs:
        REC.TaskRecord.from_dict(rec)
        assert rec["design"] == NCV.DESIGN and rec["family"] == "null"
        assert rec["config"]["system_generator_version"] == "mpc-bench-generators/1.0.0"
        # the surrogate stream of every task of a seed base is the base's
        # (module docstring: replicates are independent given these draws)
        assert rec["config"]["null_seed"] == RB.null_seed(DEV_BASE)
    fields = HE.Fields(spec["fields"], spec["derived"])
    rows = HE.component_rows(recs)
    assert [r["principle"] for r in rows] == ["RAM", "RAM"]
    labels = [HE.render_template(
        "null_calibration:{@null_kind}:T{@n_time}:N{@n_nodes}|{principle}", r, fields)
        for r in rows]
    assert labels == ["null_calibration:ar1:T1200:N8|RAM",
                      "null_calibration:surrogate_linear:T2400:N16|RAM"]
    assert {r["declaration_id"] for r in rows} == {"none"}
    assert {r["protocol_id"] for r in rows} == {"A-none"}
    # no duplicate simulations, and the records are the plan's tasks
    parsed = [REC.TaskRecord.from_dict(r) for r in recs]
    assert REC.duplicate_simulations(parsed) == []
    assert REC.plan_differences(parsed, [t.task_id for t in tasks])["ok"]


def test_the_run_scripts_find_the_design(capsys):
    assert NCV.MODULE in D.available_modules()
    assert RB.main(["designs", "--module", NCV.MODULE]) == 0
    assert capsys.readouterr().out.strip() == NCV.DESIGN


# --------------------------------------------------------------------------
# the script
# --------------------------------------------------------------------------
def test_plan_command(capsys):
    assert CLI.main(["plan", "--split", "development"]) == 0
    out = capsys.readouterr().out
    assert "16 cells, 128 tasks" in out
    assert "surrogate_linear:T2400:N16" in out and "RAM,PDI,SRPI" in out
    rows = CLI.plan_rows(DEV)
    assert {r["n_tasks"] for r in rows} == {8}
    assert {r["seed_base"] for r in rows} == {400}


def test_confirmatory_runs_are_the_full_design_and_go_through_the_guard(tmp_path):
    with pytest.raises(CLI.RunRefused, match="full confirmatory design"):
        CLI.build_run_tasks(CONF, replicates=2, confirmatory=True)
    with pytest.raises(CLI.RunRefused, match="full confirmatory design"):
        CLI.build_run_tasks(DEV, confirmatory=True)
    with pytest.raises(CLI.RunRefused, match="--confirmatory"):
        CLI.build_run_tasks(CONF)
    assert CLI.main(["run", "--out", str(tmp_path / "x"), "--split", "confirmatory",
                     "--confirmatory", "--kinds", "ar1"]) == 2
    assert not (tmp_path / "x").exists()


def test_run_and_summarise_a_development_grid(tmp_path):
    out = tmp_path / "run"
    rc = CLI.main(["run", "--out", str(out), "--kinds", "ar1,surrogate_linear",
                   "--T", "1200", "--nodes", "8", "--replicates", "2",
                   "--principles", "SRPI"])
    assert rc == 0
    plan = json.loads((out / RB.PLAN_JSON).read_text())
    assert plan["split"] == DEV and len(plan["task_ids"]) == 4
    assert plan["protocols"].keys() == {"A-none"}
    summ = tmp_path / "summary"
    assert CLI.main(["summarise", str(out), "--out", str(summ)]) == 0
    rep = pd.read_csv(summ / CLI.REPLICATES_CSV)
    # the columns of the replicate table
    cols = {"null_kind", "n_time", "n_nodes", "replicate", "seed", "principle",
            "estimate", "null_mean", "null_sd", "n_null", "null_family",
            "statistic", "defined", "estimator_reason", "margin", "se", "c",
            "c_se", "c_df", "c_lower", "c_upper", "status", "status_reason",
            "status_impl", "runner", "seconds_system"}
    assert cols <= set(rep.columns)
    assert set(rep["statistic"]) == {"raw"}
    assert len(rep) == 4 and set(rep["principle"]) == {"SRPI"}
    assert set(rep["seed"]) == {NCV._seed(DEV_BASE, k, 1200, 8, r)
                                for k in ("ar1", "surrogate_linear") for r in (0, 1)}
    assert not rep["agency_fix_needed"].any()
    rates = pd.read_csv(summ / CLI.RATES_CSV)
    assert len(rates) == 2 and set(rates["n"]) == {2}
    assert set(rates["status_impl"]) == {"tost"}
    verdicts = pd.read_csv(summ / CLI.VERDICTS_CSV)
    assert set(verdicts["necessity_set"]) == {"SRPI"}
    s = json.loads((summ / CLI.SUMMARY_JSON).read_text())
    assert (s["n_tasks"], s["n_components"], s["n_srpi_agency_fix_needed"]) == (4, 4, 0)
    assert s["split"] == DEV and s["declarations"] == ["none"]
    assert [p["protocol_id"] for p in s["protocols"]] == ["A-none"]
    assert s["cells"] == ["ar1:T1200:N8", "surrogate_linear:T1200:N8"]
    # the summary refuses an empty input
    assert CLI.main(["summarise", str(tmp_path / "summary"), "--out",
                     str(tmp_path / "s2")]) == 2


# --------------------------------------------------------------------------
# null data, events and rate tables
# --------------------------------------------------------------------------
def test_null_timeseries_properties():
    rng = np.random.default_rng(0)
    x = NCV.null_timeseries("ar1", 4, 6000, rng, ar_coef=0.5)
    assert x.shape == (4, 6000)
    assert np.allclose(x.mean(axis=1), 0, atol=1e-9) and np.allclose(x.std(axis=1), 1)
    r1 = np.mean([np.corrcoef(v[:-1], v[1:])[0, 1] for v in x])
    assert r1 == pytest.approx(0.5, abs=0.05)
    p = NCV.null_timeseries("pink", 4, 8192, rng, pink_beta=1.0)
    f = np.fft.rfftfreq(8192)[1:2000]
    psd = np.mean(np.abs(np.fft.rfft(p, axis=1)[:, 1:2000]) ** 2, axis=0)
    slope = np.polyfit(np.log(f), np.log(psd), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.3)
    s = NCV.null_timeseries("surrogate_iid", 6, 4000, rng)
    c = np.corrcoef(s)
    assert np.max(np.abs(c[np.triu_indices(6, 1)])) < 0.15
    with pytest.raises(ValueError):
        NCV.null_timeseries("white", 2, 100, rng)


def test_null_events_satisfy_the_task_and_agency_contracts():
    from impact_pipeline.event_parsing import (
        events_table_to_bundle,
        validate_srpi_agency_contract,
    )

    rng = np.random.default_rng(2)
    ev = NCV.null_events(2400, 0.05, rng)
    b = events_table_to_bundle(ev)
    assert len(b["goal_onsets"]) >= 6 and len(b["feedback_onsets"]) >= 6
    rep = validate_srpi_agency_contract(b["agency_events"], tr=0.05)
    assert rep["valid"], rep["violations"]
    assert rep["n_pairs"] >= 3
    other = ev[ev["trial_type"] == "other_caused"]
    selfs = ev.set_index("event_id")
    for _, r in other.iterrows():
        s = selfs.loc[r["yoked_to"]]
        assert s["value"] == r["value"] and s["phase_bin"] == r["phase_bin"]


def test_expected_exchangeable_rate():
    k = 19
    exp = CLI.expected_exchangeable_rate(k)
    assert exp == pytest.approx(t_dist.sf(1.645 / math.sqrt(1 + 1 / k), k - 1))
    assert CLI.expected_exchangeable_rate(10000) == pytest.approx(0.05, abs=0.002)


def test_kleene_verdict_of_component_statuses():
    assert CLI.kleene_verdict(["PRESENT", "PRESENT"]) == "MPC_CONSISTENT"
    assert CLI.kleene_verdict(["PRESENT", "ABSENT", "UNDEFINED"]) == "EXCLUDED"
    assert CLI.kleene_verdict(["PRESENT", "UNDEFINED"]) == "UNDETERMINED"


def test_expected_rate_uses_the_null_size_of_calibrated_replicates():
    """Regression: undefined replicates record n_null = 0; the median over all
    replicates then understated K (or gave K = 0 -> NaN)."""
    rows = []
    for rep, (st, n_null) in enumerate([("UNDEFINED", 0), ("UNDEFINED", 0),
                                        ("PRESENT", 19), ("ABSENT", 19),
                                        ("UNDEFINED", 0)]):
        rows.append({"null_kind": "ar1", "n_time": 1200, "n_nodes": 8,
                     "replicate": rep, "principle": "PDI", "status": st,
                     "n_null": n_null, "margin": 0.0, "defined": n_null > 0,
                     "status_impl": "evidence", "runner": "x",
                     "seconds_system": 0.1})
    rates, _ = CLI.rate_tables(pd.DataFrame(rows))
    assert int(rates["median_n_null"].iloc[0]) == 19
    assert rates["expected_rate_exchangeable"].iloc[0] == pytest.approx(
        CLI.expected_exchangeable_rate(19))
