# -*- coding: utf-8 -*-
"""
The development calibration runner and the protocol builder of MPC-Bench v2:

* the builder is reproducible from fixed inputs (same bytes, same hashes);
* the anchor rule (>= 36 of 40 finite excesses, one-sided t lower bound
  > 0) and the specificity gate on synthetic records, and ``N_anch``;
* the testability table (empirical ``pi0`` and ``pi1``) and the precision
  block on synthetic records;
* every protocol of the confirmatory plan is written: copies of A-R for the
  held-out declarations and ``A-none`` without the reported IIM cut, the
  family-B protocols exactly as the validation builds them, the ``B``
  carrier of the family-B testability rows;
* decisions: pending, the code that runs them, deviations from the rules;
* the development policy: seeds of 1000 or more are refused, held-out
  conditions only as smoke tests (outputs discarded), held-out-regime
  anchors only after a logged release of committed predictions, IIM on
  the Hopf sensor views only at G = 0 (the G sweep is held out); a
  reference block without its own-lesion runs blocks the freeze;
* a stored component re-judged by the builder equals the runner's own
  judgment under the same protocol.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from impact_pipeline import evidence_v2 as E
from impact_pipeline.bench import designs_v2 as D
from impact_pipeline.bench import run_bench_v2 as RB
from impact_pipeline.v2 import PRINCIPLES
from impact_pipeline.v2 import hypothesis_engine as HE
from impact_pipeline.v2 import records as REC
from impact_pipeline.v2 import testability as T
from scripts.v2 import build_protocols_v2 as BP
from scripts.v2 import dev_calibration as DC

PH = "0" * 64
VERSIONS = {"NAS": "nas-v3-2026.10", "IIM": "iim-v5-2026.10", "PDI": "pdi-v3-2026.10",
            "RAM": "ram-v3-2026.10", "SRPI": "srpi-v2-2026.09"}
MODES = {"NAS": "conditional_capacity", "IIM": "directional",
         "PDI": "repertoire_discriminant", "RAM": "pe_readout", "SRPI": "agency"}
NULLS = {"NAS": "block_circular_shift", "IIM": "circular_shift",
         "PDI": "circular_shift", "RAM": "trial_circular_shift",
         "SRPI": "yoked_label_permutation"}
SE_METHODS = {"NAS": "jackknife_contiguous_10",
              "IIM": "circular_block_bootstrap_10pct_B50",
              "PDI": "jackknife_contiguous_10", "RAM": "shift_null_sd",
              "SRPI": "jackknife_pairs_10"}
SE_DF = {"NAS": 9.0, "IIM": 12.0, "PDI": 9.0, "RAM": 69.0, "SRPI": 9.0}
ANCHOR = {"NAS": 0.05, "IIM": 0.02, "PDI": 2.0, "RAM": 1.0, "SRPI": 0.2}
OWN = {"NAS": "W_NAS_no_workspace", "IIM": "W_IIM_feedforward",
       "PDI": "W_PDI_single_attractor", "RAM": "W_RAM_no_plasticity",
       "SRPI": "W_SRPI_no_efference"}


# --------------------------------------------------------------------------
# synthetic development records
# --------------------------------------------------------------------------
def _member(p, excess, se, direction=None, defined=True, se_method=None):
    return {"direction": direction, "estimate": float(excess) + 0.001,
            "null_mean": 0.001, "null_sd": 1e-4 * ANCHOR[p], "n_null": 19,
            "null_family": NULLS[p], "se": float(se), "se_df": SE_DF[p],
            "se_method": se_method or SE_METHODS[p], "defined": bool(defined),
            "reason": None,
            "exact": False, "p_ind": 0.01 if p == "IIM" else None,
            "content_bearer": "non_workspace" if p == "PDI" else None}


def component(p, excess, se, *, protocol="A-R", declaration="R", defined=True,
              se_method=None, replicate=0):
    """An UNDEFINED(INVALID_ANCHORS) component as a draft run stores it."""
    det = {"estimator_id": f"compute_{p}:{MODES[p]}@{VERSIONS[p]}",
           "defined": bool(defined), "estimator_reason": None, "scorer": "x"}
    if p == "NAS":
        ex = excess if isinstance(excess, (tuple, list)) else (excess, excess)
        det["members"] = {d: _member(p, e, se, d, defined, se_method)
                          for d, e in zip(E.NAS_DIRECTIONS, ex)}
        m = det["members"]["receive"]
    else:
        m = _member(p, excess, se, None, defined, se_method)
        if p == "IIM":
            det["p_ind"] = 0.01
        if p == "PDI":
            det["estimator"] = {"counted_bearer": "non_workspace"}
    status, reason = ("UNDEFINED", "INVALID_ANCHORS")
    return REC.ComponentRecord(
        principle=p, status=status, reason=reason, estimator_version=VERSIONS[p],
        declaration_id=declaration, observation_stage="source", protocol_id=protocol,
        protocol_hash=PH, estimate=m["estimate"], null_mean=m["null_mean"],
        null_sd=m["null_sd"], n_null=19, null_family=NULLS[p], se=m["se"],
        se_df=m["se_df"], se_method=m["se_method"], replicate=replicate,
        details=det)


def record(design, system, seed, comps, *, protocol="A-R", declaration="R",
           family="A", config=None, task_id=None, replicate=0):
    sc = REC.ScoringRecord(
        scoring_id=declaration, declaration_id=declaration, observation_stage="source",
        view="source", estimator_form="primary", protocol_id=protocol,
        protocol_hash=PH, components={c.principle: c for c in comps})
    cfg = {"held_out": False, "tags": {}, "settings": RB.DEFAULT_SETTINGS.to_dict()}
    cfg.update(config or {})
    return REC.TaskRecord(
        task_id=task_id or f"{design}-{system}-r{replicate}-s{seed:05d}",
        design=design, family=family, system=system, seed=int(seed),
        replicate=int(replicate),
        generator_version="mpc-bench-generators/2.0.0", status="ok", config=cfg,
        simulation={"ts_sha256": f"{seed:064x}", "n_nodes": 4, "n_time": 100,
                    "dt": 0.05},
        scorings=(sc,))


def reference_block(rng, *, n_ram_finite=35):
    """PC_nominal and the five own lesions on 900-939: NAS, IIM and PDI valid
    and specific; SRPI valid but not specific (its lesion keeps 90 %); RAM
    with 35 of 40 finite excesses (invalid)."""
    recs = []
    for i, seed in enumerate(range(900, 940)):
        for system in ("PC_nominal",) + tuple(OWN.values()):
            comps = []
            for p in PRINCIPLES:
                a = ANCHOR[p]
                lesioned = system == OWN[p]
                level = (0.9 if p == "SRPI" else 0.0) if lesioned else 1.0
                sd = 0.01 if level == 0.0 else 0.1
                excess = a * (level + sd * rng.standard_normal())
                se = (1.0 if (p == "SRPI" and lesioned) else 0.004) * a
                defined = not (p == "RAM" and i >= n_ram_finite)
                if p == "NAS":
                    excess = (excess, a * (level + sd * rng.standard_normal()))
                comps.append(component(p, excess, se, defined=defined))
            recs.append(record("A_anchors", system, seed, comps))
    return recs


def write_root(root: Path, item: str, run: str, recs, name=RB.RESULTS_JSONL):
    d = Path(root) / item / run
    d.mkdir(parents=True, exist_ok=True)
    REC.write_jsonl(recs, d / name)
    return d


@pytest.fixture(scope="module")
def dev_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("devroot")
    write_root(root, "anchors_A", "A_anchors",
               reference_block(np.random.default_rng(7)))
    return root


@pytest.fixture(scope="module")
def built(dev_root):
    return BP.build(None, dev_root, keys=["A-R"])


# --------------------------------------------------------------------------
# anchors, N_anch, testability
# --------------------------------------------------------------------------
def test_anchor_rule_and_specificity_gate(built):
    proto = built["protocols"]["A-R"]
    status = {p: proto.anchor_status(p) for p in PRINCIPLES}
    assert status == {"RAM": T.ANCHOR_INVALID, "PDI": T.ANCHOR_VALID_SPECIFIC,
                      "NAS": T.ANCHOR_VALID_SPECIFIC, "IIM": T.ANCHOR_VALID_SPECIFIC,
                      "SRPI": T.ANCHOR_VALID_NONSPECIFIC}
    assert proto.necessity_set == ("PDI", "NAS", "IIM")
    assert proto.fallback == "F1"
    ram = proto.anchors["principles"]["RAM"]["validity"]
    assert ram["n_finite"] == 35 and not ram["valid"]
    ref = proto.reference
    assert ref["kind"] == "external" and ref["scale"] == "excess"
    # every valid anchor (specific or not) carries its reference; NAS per direction
    assert set(ref["values"]) == {"PDI", "IIM", "SRPI", "NAS:receive", "NAS:return"}
    assert ref["values"]["PDI"] == pytest.approx(2.0, rel=0.1)
    assert ref["values"]["NAS:receive"] == pytest.approx(0.05, rel=0.1)
    spec = proto.anchors["principles"]["SRPI"]["specificity"]
    assert spec["mean_contrast"] < 0.5 * spec["anchor"]


@pytest.mark.parametrize("n_finite, mean, valid", [
    (36, 1.0, True), (35, 1.0, False), (40, 0.0, False), (40, 1.0, True)])
def test_anchor_rule_thresholds(n_finite, mean, valid):
    """>= 36 of 40 finite excesses and a positive one-sided t lower bound."""
    rng = np.random.default_rng(3)
    pc = mean + 0.1 * rng.standard_normal(40)
    pc[n_finite:] = np.nan
    entries, ref = BP.anchor_entries({"RAM": {"pc": pc, "lesion": np.zeros(40)}})
    assert entries["RAM"]["valid"] is valid
    assert ("RAM" in ref["values"]) is valid


def test_testability_rows_and_precision_block(built):
    rows = {(r["principle"], r["witness"], r["kind"]): r
            for r in built["protocols"]["A-R"].precision["rows"]}
    for p, w in (("NAS", OWN["NAS"]), ("IIM", OWN["IIM"]), ("PDI", OWN["PDI"])):
        r = rows[(p, w, T.KIND_ABSENT)]
        assert r["n"] == 40 and r["pi"] == 1.0 and r["decisive"]
    # NAS's TOST is a union over two directions: members 2 (alpha_A / 2)
    assert rows[("NAS", OWN["NAS"], T.KIND_ABSENT)]["members"] == 2
    srpi = rows[("SRPI", OWN["SRPI"], T.KIND_ABSENT)]
    assert srpi["pi"] == 0.0 and srpi["outcome"] == T.NOT_TESTABLE_BY_DESIGN
    assert srpi["rates"]["ABSENT_NOT_REACHABLE"] == 1.0
    ram = rows[("RAM", OWN["RAM"], T.KIND_ABSENT)]
    # RAM has no anchor: every run UNDEFINED, the defined ones INVALID_ANCHORS
    assert ram["pi"] == 0.0 and ram["rates"]["UNDEFINED"] == 1.0
    assert ram["rates"]["INVALID_ANCHORS"] == pytest.approx(35 / 40)
    for p in ("NAS", "IIM", "PDI"):
        r = rows[(p, "PC_nominal", T.KIND_PRESENT)]
        assert r["pi"] == 1.0 and r["decisive"]
    table = built["files"][BP.TESTABILITY]["rows"]
    short = [t for t in table if t["family"] == "A-R" and t["row"] is None]
    assert short and all(t["n"] < T.MIN_RUNS for t in short)


def test_rejudging_reads_the_stored_estimator_fields(built, dev_root):
    data = BP.DevData(dev_root)
    proto = built["protocols"]["A-R"]
    _src, rec = next((s, r) for s, r in data.records if r.system == OWN["NAS"])
    s = rec.scorings[0]
    j = BP.rejudge(rec, s, s.components["NAS"], proto)
    assert j.status == "ABSENT" and set(j.members) == set(E.NAS_DIRECTIONS)
    j = BP.rejudge(rec, s, s.components["RAM"], proto)
    assert j.status == "UNDEFINED" and j.reason == "INVALID_ANCHORS"


# --------------------------------------------------------------------------
# reproducibility and the protocol set
# --------------------------------------------------------------------------
def test_the_build_is_reproducible(dev_root, tmp_path):
    a = BP.write(BP.build(None, dev_root, keys=["A-R"]), tmp_path / "a")
    b = BP.write(BP.build(None, dev_root, keys=["A-R"]), tmp_path / "b")
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n
    assert BP.check(a)["ok"]
    man = json.loads((a / BP.MANIFEST).read_text())
    assert man["freeze_ready"] is False and man["blocking"]
    assert "created" not in json.dumps(man) and str(dev_root) not in json.dumps(man)
    # a changed file no longer matches the manifest
    f = a / HE.protocol_file_name("A-R")
    f.write_text(f.read_text().replace("A-R", "A-R "), encoding="utf-8")
    assert not BP.check(a)["ok"]


@pytest.fixture(scope="module")
def full_build(dev_root):
    return BP.build(None, dev_root)


def test_every_protocol_of_the_confirmatory_plan_is_written(full_build):
    planned = BP.confirmatory_protocol_keys()
    protos = full_build["protocols"]
    assert set(planned) <= set(protos)
    assert "B" in protos and "A-none" in planned and "A-P" in planned
    for key, p in protos.items():
        assert HE.protocol_key_of_name(p.name) == key
    files = full_build["files"]
    for key in planned:
        assert HE.protocol_file_name(key) in files


def test_copies_of_a_r_drop_the_reported_cut(full_build):
    protos = full_build["protocols"]
    a_r = protos["A-R"].to_dict()
    assert a_r["estimators"]["IIM"]["report_cut_modes"] == ["bidirectional"]
    for key, decl in BP.DERIVED_FROM_A_R.items():
        d = protos[key].to_dict()
        assert d["shared_inputs_declaration"]["id"] == decl
        assert d["estimators"]["IIM"]["report_cut_modes"] == []
        for field in ("reference", "anchors", "necessity_set", "se_methods",
                      "precision", "concordance_route", "cutoffs"):
            assert d[field] == a_r[field], (key, field)
    for key, p in protos.items():
        if key.startswith(("hopf-", "fwdA")):
            assert p.estimator_options("IIM")["report_cut_modes"] == []


def test_family_b_protocols_equal_the_validation_protocols(full_build):
    fb = BP.family_b_protocols()
    assert len(fb) == 16
    for key, p in fb.items():
        assert key.startswith("B-")
        got = E.ProtocolV3.from_dict(full_build["files"][HE.protocol_file_name(key)])
        assert got.hash == p.hash
    b = full_build["protocols"]["B"]
    assert b.precision is not None and b.necessity_set == ("IIM",)


def test_forward_and_inherited_protocols(full_build):
    protos = full_build["protocols"]
    ev = full_build["evidence"]
    # without forward anchors the forward views are pending, never invented
    assert protos["hopf-eeg64"].reference["kind"] == E.REFERENCE_PENDING
    assert any("anchors pending" in b for b in full_build["manifest"]["blocking"])
    # a held-out form inherits the anchors of its base
    assert ev["inherited_anchors"]["A-R+pdi_misdeclared_access"] == "A-R"
    mis = protos["A-R+pdi_misdeclared_access"]
    assert mis.reference == protos["A-R"].reference
    assert mis.estimator_options("PDI")["access_module"] == "S"


def test_decisions_reach_the_estimator_blocks(dev_root):
    dec = {"schema": BP.DECISIONS_SCHEMA, "status": "draft",
           "decisions": {"nas_se_method": {"value": "jackknife_interleaved_20",
                                           "deviation": "test"},
                         "ram_se_method": {"value": "jackknife_trials_10"}}}
    res = BP.build(dec, dev_root, keys=["A-R"])
    p = res["protocols"]["A-R"]
    assert p.estimator_options("NAS")["se_method"] == "jackknife_interleaved_20"
    assert p.se_methods_for("NAS") == ("jackknife_interleaved_20",)
    assert p.estimator_options("RAM")["se_method"] == "jackknife_trials_10"
    assert p.estimator_options("IIM")["n_min"] == 25
    # the testability rows were computed with another NAS SE method
    assert any("not the decided one" in b for b in res["manifest"]["blocking"])
    # the runner accepts every option the builder writes
    for prin in ("NAS", "IIM", "RAM", "PDI"):
        RB.scorer_for(p.estimator_version_for(prin), prin).check_options(
            p.estimator_options(prin))


# --------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------
def test_pending_decisions_use_the_code_values():
    dec = BP.load_decisions(None)
    assert set(dec.pending) == set(BP.DECISION_NAMES)
    assert dec["iim_max_macro_nodes"] == RB.IIM_MAX_MACRO_NODES == 4
    assert dec["pdi_kmeans_seed"] == RB.CALIBRATION_PENDING["pdi_kmeans_seed"]["value"]
    rec, blocking = BP.decisions_record(dec, {})
    assert all(e["pending"] for e in rec["decisions"].values())
    assert all(e.get("code", {}).get("consistent", True)
               for e in rec["decisions"].values())


def test_a_decision_must_match_the_code_that_runs_it():
    dec = BP.load_decisions({"schema": BP.DECISIONS_SCHEMA, "status": "final",
                             "decisions": {"iim_n_min": {"value": 50},
                                           "iim_max_macro_nodes": {"value": 5},
                                           "pdi_kmeans_seed": {"value": None,
                                                               "decided": True}}})
    assert "iim_n_min" not in dec.pending and "pdi_kmeans_seed" not in dec.pending
    rec, blocking = BP.decisions_record(dec, {})
    assert not rec["decisions"]["iim_n_min"]["code"]["consistent"]
    assert any("iim_n_min" in b and "N_MIN_DEFAULT" in b for b in blocking)
    assert any("iim_max_macro_nodes" in b for b in blocking)
    assert any("pdi_kmeans_seed" in b for b in blocking)  # null differs from 0


def test_a_decision_against_the_rule_needs_a_deviation_note():
    sugg = {"nas_se_method": {"value": "jackknife_contiguous_20"}}
    base = {"schema": BP.DECISIONS_SCHEMA, "status": "final"}
    dec = BP.load_decisions({**base, "decisions": {
        "nas_se_method": {"value": "jackknife_contiguous_10"}}})
    _rec, blocking = BP.decisions_record(dec, sugg)
    assert any("nas_se_method" in b and "deviation" in b for b in blocking)
    dec = BP.load_decisions({**base, "decisions": {
        "nas_se_method": {"value": "jackknife_contiguous_10", "deviation": "why"}}})
    _rec, blocking = BP.decisions_record(dec, sugg)
    assert not any("nas_se_method" in b for b in blocking if "pending" not in b)


@pytest.mark.parametrize("bad", [
    {"schema": "x"},
    {"schema": BP.DECISIONS_SCHEMA, "decisions": {"no_such": {"value": 1}}},
    {"schema": BP.DECISIONS_SCHEMA, "decisions": {"nas_se_method": {"value": "x"}}},
    {"schema": BP.DECISIONS_SCHEMA, "decisions": {"iim_n_min": {"value": 30}}},
    {"schema": BP.DECISIONS_SCHEMA,
     "decisions": {"null_calibration_ram": {"value": "maybe"}}},
    {"schema": BP.DECISIONS_SCHEMA, "status": "frozen"},
])
def test_malformed_decisions_are_refused(bad):
    with pytest.raises(BP.BuildError):
        BP.load_decisions(bad)


def test_the_template_lists_every_decision_pending(tmp_path):
    t = BP.template(None)
    assert t["schema"] == BP.DECISIONS_SCHEMA and t["status"] == "draft"
    assert set(t["decisions"]) == set(BP.DECISION_NAMES)
    assert all(v["value"] is None for v in t["decisions"].values())
    assert set(BP.load_decisions(t).pending) == set(BP.DECISION_NAMES)


# --------------------------------------------------------------------------
# the development policy of the builder
# --------------------------------------------------------------------------
def _single(tmp_path, rec, item="anchors_A", run="A_anchors"):
    root = tmp_path / "root"
    write_root(root, item, run, [rec])
    return root


def _pc(seed, **kw):
    return record("A_anchors", "PC_nominal", seed,
                  [component("PDI", 1.0, 0.01)], **kw)


def test_the_builder_refuses_seeds_of_1000_or_more(tmp_path):
    with pytest.raises(BP.BuildError, match="development seeds"):
        BP.DevData(_single(tmp_path, _pc(20000)))


def test_the_builder_refuses_held_out_outputs(tmp_path):
    with pytest.raises(BP.BuildError, match="held-out"):
        BP.DevData(_single(tmp_path, _pc(900, config={"held_out": True})))


def test_held_out_regime_anchors_need_a_logged_release(tmp_path):
    tags = {"purpose": "reference", "regime": "held_out", "anchor": True,
            "held_out_release": "abc123"}
    rec = _pc(900, config={"tags": tags})
    root = _single(tmp_path, rec, "anchors_forward_held_out", "forward_reference")
    with pytest.raises(BP.BuildError, match="release"):
        BP.DevData(root)
    (root / DC.RELEASE_LOG).write_text(json.dumps({"release_id": "abc123",
                                                   "predictions": []}) + "\n")
    assert len(BP.DevData(root).records) == 1


def test_held_out_family_b_cells_are_refused(tmp_path):
    cell = {"system": "ring", "params": {"coupling": 0.9}, "hypothesis": "HCv2-12"}
    rec = record("family_b", "ring", 425, [component("IIM", 0.01, 0.001)],
                 family="B", config={"cell": cell})
    with pytest.raises(BP.BuildError, match="held-out regime"):
        BP.DevData(_single(tmp_path, rec, "dry_run", "family_b"))


# --------------------------------------------------------------------------
# the calibration runner
# --------------------------------------------------------------------------
def test_the_plan_lists_every_item_with_projected_cost():
    p = DC.plan(workers=12)
    names = [r["item"] for r in p["items"]]
    assert names == list(DC.ITEM_NAMES)
    for item in ("constants", "oracle_checks", "anchors_A", "anchors_C1",
                 "anchors_RAM160", "anchors_forward_held_out", "twins",
                 "twins_nas_se", "pdi_concordance", "dry_run", "smoke"):
        row = p["items"][names.index(item)]
        assert not row["optional"] and row["command"].endswith("--workers 12")
    assert p["totals"]["planned"]["cpu_h"] > 0
    assert p["totals"]["planned"]["wall_h"] == pytest.approx(
        p["totals"]["planned"]["cpu_h"] * DC.CONTENTION / 12, abs=0.02)
    for row in p["items"]:
        for r in row["runs"]:
            assert "unpriced_designs" not in r, (row["item"], r["run"])


@pytest.mark.parametrize("item", [i.name for i in DC.ITEMS
                                  if any(r.kind in (DC.RUNNER, DC.BATTERY)
                                         and not r.released for r in i.runs)])
def test_every_planned_run_uses_development_seeds_and_no_held_out_output(item):
    for run in DC.get_item(item).runs:
        if run.kind not in (DC.RUNNER, DC.BATTERY):
            continue
        tasks = DC.runner_tasks(run)
        assert all(t.seed <= 999 for t in tasks)
        assert all(t.smoke for t in tasks if t.has_held_out)


def test_seeds_of_1000_or_more_are_refused():
    with pytest.raises(DC.CalibrationError, match="0-999"):
        DC.check_seed_list([900, 20000])
    task = D.TaskSpec(task_id="x-s20000", design="A_anchors", family="A",
                      system="PC_nominal", seed=20000)
    with pytest.raises(DC.CalibrationError, match="0-999"):
        DC.check_tasks([task])


def test_the_command_line_refuses_seeds_of_1000_or_more(tmp_path, capsys):
    for item in ("anchors_RAM160", "dry_run", "oracle_checks"):
        rc = DC.main(["run", item, "--root", str(tmp_path), "--seeds", "20000"])
        assert rc == 2
        assert "0-999" in capsys.readouterr().err
    assert not any(tmp_path.glob("**/results.jsonl"))


def test_held_out_conditions_only_as_smoke_tests():
    sc = D.make_scorings({"R": "A-R"}, ("NAS",), verdict=False)
    held = D.TaskSpec(task_id="h-s00900", design="A_heldout", family="A",
                      system="PC_nominal", seed=900, scorings=sc, held_out=True)
    with pytest.raises(DC.CalibrationError, match="smoke"):
        DC.check_tasks([held])
    smoke = dataclasses.replace(held, task_id="h-s00980", seed=980)
    DC.check_tasks([smoke])
    tasks = DC.runner_tasks(DC.get_item("smoke").runs[0])
    assert tasks and all(t.smoke and t.has_held_out for t in tasks)


def test_released_anchors_need_the_release(git_repo, tmp_path):
    run = DC.get_item("anchors_forward_held_out").runs[0]
    with pytest.raises(DC.CalibrationError, match="release"):
        DC.runner_tasks(run, seeds=[900])
    with pytest.raises(DC.CalibrationError, match="release"):
        DC.run_item("anchors_forward_held_out", root=tmp_path / "root")
    pred = git_repo.write("predictions.md", "held-out predictions\n")
    with pytest.raises(DC.CalibrationError, match="committed"):
        DC.release_held_out_anchors([pred], root=tmp_path / "root",
                                    repo_root=git_repo.root)
    git_repo.commit("predictions")
    entry = DC.release_held_out_anchors([pred], root=tmp_path / "root",
                                        repo_root=git_repo.root, note="test")
    assert entry["predictions"][0]["commit"] and entry["predictions"][0]["clean"]
    assert DC.current_release(tmp_path / "root", git_repo.root)["release_id"] == (
        entry["release_id"])
    tasks = DC.runner_tasks(run, seeds=[900], release=entry)
    assert len(tasks) == 3
    for t in tasks:
        assert not t.has_held_out
        assert t.tags["held_out_release"] == entry["release_id"]
        assert t.tags["regime"] == "held_out" and t.tags["purpose"] == "reference"
    pred.write_text("changed after the release\n", encoding="utf-8")
    with pytest.raises(DC.CalibrationError, match="changed"):
        DC.current_release(tmp_path / "root", git_repo.root)


def test_the_battery_and_the_family_b_development_cells():
    tasks = DC.battery_tasks()
    assert len(tasks) == len(DC.BATTERY_CLASSES) * len(DC.BATTERY_SEEDS)
    assert {t.seed for t in tasks} == set(range(870, 900))
    assert {t.tags["content"] for t in tasks} == {DC.CONTENT_ON, DC.CONTENT_OFF}
    for t in tasks:
        (s,) = t.scorings
        assert s.protocol_key == "A-R" and s.principles == ("PDI",) and not s.verdict
    keep, dropped = DC.family_b_tasks()
    assert dropped and not any(DC.family_b_held_out(t.cell) for t in keep)
    assert all("ring/coupling=0.9" in c or "ring/coupling=1.5" in c for c in dropped)


def test_alternative_se_methods_override_the_protocols(tmp_path):
    run = DC.get_item("twins_nas_se").runs[0]
    tasks = DC.runner_tasks(run, seeds=[820])[:3]
    over = DC.protocol_overrides(tasks, run.estimator_options, tmp_path)
    assert over and all(p.estimator_options("NAS")["se_method"]
                        == "jackknife_contiguous_20" for p in over.values())


def test_a_tiny_calibration_run_resumes_with_provenance(tmp_path):
    out = DC.run_item("anchors_RAM160", root=tmp_path, seeds=[900])
    assert out["RAM160_anchors"]["n_run"] == 2
    d = DC.run_dir(tmp_path, "anchors_RAM160", "RAM160_anchors")
    prov = json.loads((d / DC.PROVENANCE_JSON).read_text())
    assert prov["iim_max_macro_nodes"] == 4
    assert prov["settings"]["iim_max_macro_nodes"] == 4
    assert "pdi_kmeans_seed" in prov["calibration_pending"]
    assert "anchors.replication_extended" in prov["calibration_pending"]
    again = DC.run_item("anchors_RAM160", root=tmp_path, seeds=[900])
    assert again["RAM160_anchors"]["n_run"] == 0
    (row,) = DC.status(tmp_path)
    assert row["records"] == 2 and row["task_errors"] == 0


# --------------------------------------------------------------------------
# the builder re-judges a stored component as the runner judges it
# --------------------------------------------------------------------------
def _anchored(key, values, se):
    payload = RB.draft_protocol(key).to_dict()
    payload["reference"] = {"kind": "external", "scale": "excess",
                            "values": values, "se": se}
    return E.ProtocolV3.from_dict(payload)


@pytest.mark.parametrize("design, seed, principles, key, values", [
    ("RAM160", 352, ("RAM",), "A-RAM160", {"RAM": 0.4}),
    ("C1_witnesses", 372, ("NAS",), "C1-R",
     {"NAS:receive": 0.02, "NAS:return": 0.02}),
])
def test_rejudging_equals_the_runner(design, seed, principles, key, values, tmp_path):
    tasks = RB.build_tasks([design], "development", seeds=[seed])
    task = next(t for t in tasks if t.system == "PC_nominal")
    proto = _anchored(key, values, {k: 0.001 for k in values})
    protos = RB.resolve_protocols(RB.protocol_keys([task]), directory=tmp_path,
                                  overrides={key: proto})
    rec = RB.run_task(task, protos, RB.RunSettings(principles=principles))
    checked = 0
    for s in rec.scorings:
        if s.protocol_id != key:
            continue
        for p, comp in s.components.items():
            j = BP.rejudge(rec, s, comp, proto)
            assert (j.status, j.reason) == (comp.status, comp.reason)
            assert j.c == pytest.approx(comp.c, nan_ok=True)
            assert j.se_c == pytest.approx(comp.se_c, nan_ok=True)
            checked += 1
    assert checked >= 1


# --------------------------------------------------------------------------
# SE calibration on twins, the concordance rule, mechanism-on labels
# --------------------------------------------------------------------------
def _twins(rng, se_factor, n_networks=40, sessions=7):
    """NAS twins: per network a true level, sessions scattered with SD 0.1 x
    anchor around it; the recorded SE is ``se_factor`` x that SD."""
    recs = []
    a = ANCHOR["NAS"]
    for net in range(n_networks):
        level = 1.0 + 0.3 * rng.standard_normal()
        for r in range(sessions):
            ex = tuple(a * (level + 0.1 * rng.standard_normal()) for _ in range(2))
            comp = component("NAS", ex, se_factor * 0.1 * a, replicate=r)
            recs.append(record("A_twins", "PC_nominal", 500 + net, [comp],
                               replicate=r))
    return recs


@pytest.mark.parametrize("se_factor, inside, outside", [
    (1.0, True, False), (0.5, False, True)])
def test_se_calibration_on_twins(built, tmp_path, se_factor, inside, outside):
    root = tmp_path / "root"
    write_root(root, "twins", "A_twins", _twins(np.random.default_rng(11), se_factor))
    res = BP.se_calibration(BP.DevData(root), built["protocols"],
                            BP.load_decisions(None))
    cells = [c for c in res["cells"] if c["principle"] == "NAS"]
    assert {c["direction"] for c in cells} == set(E.NAS_DIRECTIONS)
    for c in cells:
        assert c["df"] == 40 * 6 and c["networks"] == 40
        assert c["inside"] is inside and c["outside"] is outside
        assert c["hcv2_4_a"] is inside
        assert (c["kappa"] > 1.25) is outside
    sug = res["suggestions"]["nas_se_method"]
    assert sug["value"] == "jackknife_contiguous_10"  # calibrated or the fallback
    assert sug["calibrated"] == (["jackknife_contiguous_10"] if inside else [])
    assert sug["calibrated_by_the_hcv2_4_a_statement"] == sug["calibrated"]


def test_se_calibration_at_the_development_twin_size(built, tmp_path):
    """5 networks x 7 sessions (df 30): the 90 % interval of a calibrated SE
    spans a factor of about 1.54, so the CD-2 reading (interval inside
    [0.8, 1.25]) fails where the HCv2-4 (a) statement holds; both are
    reported, the rule's suggestion follows CD-2."""
    root = tmp_path / "root"
    write_root(root, "twins", "A_twins",
               _twins(np.random.default_rng(11), 1.0, n_networks=5))
    res = BP.se_calibration(BP.DevData(root), built["protocols"],
                            BP.load_decisions(None))
    cells = [c for c in res["cells"] if c["principle"] == "NAS"]
    assert cells and all(c["df"] == 30 for c in cells)
    for c in cells:
        lo, hi = c["interval"]
        assert hi / lo == pytest.approx(1.539, abs=0.01)
        assert c["inside"] is (lo >= 0.8 and hi <= 1.25)
        assert c["hcv2_4_a"] is (0.8 <= c["kappa"] <= 1.25 and lo >= 0.67 and hi <= 1.5)
    assert any(c["hcv2_4_a"] and not c["inside"] for c in cells)


def _battery(n_on, n_events, n_off=0):
    """PDI battery runs with concordant counts: content-on runs at c = 1 and
    ``n_events`` of them at c = 0 (a concordant ABSENT)."""
    recs = []
    for i in range(n_on + n_off):
        on = i < n_on
        excess = 0.0 if (on and i < n_events) else ANCHOR["PDI"] * (1.0 if on else 0.0)
        comp = component("PDI", excess, 0.0, se_method=E.SE_METHOD_CONCORDANT)
        cls = "K2" if on else "N_ar1"
        tags = {"battery_class": cls, "content": "on" if on else "off"}
        recs.append(record(DC.BATTERY_DESIGN, f"battery_{cls}", 870 + i % 30,
                           [comp], config={"tags": tags},
                           task_id=f"battery-{i:04d}"))
    return recs


@pytest.mark.parametrize("n_on, n_events, admitted", [
    (300, 0, True), (300, 1, False), (299, 0, False)])
def test_the_concordance_rule(built, tmp_path, n_on, n_events, admitted):
    root = tmp_path / "root"
    write_root(root, "pdi_concordance", "battery", _battery(n_on, n_events, 10))
    rows = BP.judged_rows(BP.DevData(root), built["protocols"], [DC.BATTERY_DESIGN],
                          dec=BP.load_decisions(None))
    (cell,) = BP.concordance_evidence(rows)
    assert (cell["substrate"], cell["view"], cell["bearer"]) == (
        "synthetic_rate", "source", "non_workspace")
    assert cell["min_runs"] == BP.CONCORDANCE_MIN_RUNS == 300
    assert cell["content_on_runs"] == n_on and cell["concordant_absent"] == n_events
    assert cell["rule_absent"] is admitted and cell["rule_present"] is False
    rule = BP._with(BP.load_decisions(None), pdi_concordance="rule")
    cells = BP.concordance_cells([cell], rule)
    assert bool(cells) is admitted
    if admitted:
        payload = built["protocols"]["A-R"].to_dict()
        payload["concordance_route"] = cells
        proto = E.ProtocolV3.from_dict(payload)
        assert proto.concordance_admission("PDI", "synthetic_rate", "source",
                                           "source", "non_workspace") == (True, False)
    assert BP.concordance_cells([cell], BP.load_decisions(None)) == []  # 'none'


def test_mechanism_on_labels(built, tmp_path):
    rng = np.random.default_rng(5)
    recs = []
    for level_idx, level in ((1, 0.222222), (3, 0.666667)):
        for seed in range(332, 336):
            nas = ANCHOR["NAS"] * (0.1 if level < 0.5 else 0.6)
            pdi = ANCHOR["PDI"] * (1 + 0.01 * rng.standard_normal())
            comps = [component("NAS", (nas, nas), 0.001),
                     component("PDI", pdi, 0.01)]
            tags = {"sweep_knob": "g_b", "sweep_level": level}
            recs.append(record("A_sweeps", f"sweep_g_b_l{level_idx:02d}", seed, comps,
                               config={"tags": tags}))
    root = tmp_path / "root"
    write_root(root, "dry_run", "A_sweeps_factorial", recs)
    rows = BP.judged_rows(BP.DevData(root), built["protocols"], ["A_sweeps"],
                          dec=BP.load_decisions(None))
    entries, evidence = BP.mechanism_on(rows)
    on = {(e["principle"], e["where"]["system"]): e["on"] for e in entries}
    # NAS: g_b below half its nominal dose is off whatever c is; at 0.67 the
    # development median c (about 0.6) is >= 2 delta
    assert on[("NAS", "sweep_g_b_l01")] is False
    assert on[("NAS", "sweep_g_b_l03")] is True
    # PDI keeps its nominal dose in a g_b sweep: on by its median c
    assert on[("PDI", "sweep_g_b_l01")] is True
    ev = {(e["principle"], e["system"]): e for e in evidence}
    assert ev[("NAS", "sweep_g_b_l01")]["dose_ratio"] == pytest.approx(0.222222)
    assert ev[("PDI", "sweep_g_b_l01")]["dose_ratio"] == 1.0


def test_smoke_outputs_are_discarded(tmp_path):
    """A held-out condition on a smoke seed runs (plumbing) and leaves a
    status line only: no record, no estimator output."""
    item = DC.get_item("smoke")
    run = item.runs[0]
    out = DC.run_one(item, run, root=tmp_path, limit=1)
    assert out["n_smoke"] == 1 and out["n_run"] == 0 and out["n_errors"] == 0
    d = DC.run_dir(tmp_path, "smoke", run.name)
    assert (d / RB.RESULTS_JSONL).read_text() == ""
    (line,) = (d / RB.SMOKE_JSONL).read_text().splitlines()
    status = json.loads(line)
    assert status["discarded"] is True and status["seed"] in range(980, 985)
    assert set(status) >= {"task_id", "status", "n_components"}
    assert "components" not in status and "scorings" not in status


# --------------------------------------------------------------------------
# twin sessions once, paired potency, raised N_min, development re-judging
# --------------------------------------------------------------------------
def test_a_twin_session_enters_the_calibration_once(built, tmp_path):
    """A tagged re-run of the twins (with the decided protocols) repeats
    the same tasks: each session is counted once per SE method."""
    root = tmp_path / "root"
    recs = _twins(np.random.default_rng(11), 1.0, n_networks=10)
    write_root(root, "twins", "A_twins", recs)
    once = BP.se_calibration(BP.DevData(root), built["protocols"],
                             BP.load_decisions(None))
    write_root(root, "twins@decided", "A_twins", recs)
    twice = BP.se_calibration(BP.DevData(root), built["protocols"],
                              BP.load_decisions(None))
    assert len(BP.DevData(root).sources) == 2
    assert [(c["df"], c["sessions"], c["kappa"]) for c in twice["cells"]] == [
        (c["df"], c["sessions"], c["kappa"]) for c in once["cells"]]
    assert all(c["df"] == 10 * 6 for c in once["cells"])


def test_adversary_potency_is_the_paired_ratio(built, tmp_path):
    """CD-9: the excess of the adversary over the positive control's on the
    same seed (common random numbers), not c against the anchor."""
    a = ANCHOR["PDI"]
    recs = []
    for seed in range(340, 352):
        pc_level = 2.0 if seed < 346 else 0.5   # the PC excess varies by seed
        recs.append(record("A_witnesses", "PC_nominal", seed,
                           [component("PDI", a * pc_level, 0.01, protocol="A-H",
                                      declaration="H")],
                           protocol="A-H", declaration="H"))
        # the adversary has 0.6 x the anchor: c = 0.6 on every seed; the
        # paired ratio is 0.3 on seeds with a strong PC, 1.2 on the others
        recs.append(record("A_adversaries", "adversarial_common_driver", seed,
                           [component("PDI", a * 0.6, 0.01, protocol="A-H",
                                      declaration="H")],
                           protocol="A-H", declaration="H"))
    root = tmp_path / "root"
    write_root(root, "dry_run", "A_adversaries", recs)
    rows = BP.judged_rows(BP.DevData(root), {"A-H": built["protocols"]["A-R"]},
                          ["A_witnesses", "A_adversaries"],
                          dec=BP.load_decisions(None))
    (pot,) = BP.adversary_potency(rows)
    assert pot["n_paired"] == 12 and pot["under_h"] is True
    assert pot["share_paired_ratio_ge_0.5"] == pytest.approx(0.5)
    assert pot["median_paired_ratio"] == pytest.approx(0.75)
    assert pot["potent"] is True
    # without the positive control on the adversary seeds nothing is paired;
    # the adversary is potent through its PRESENT rate only
    root2 = tmp_path / "root2"
    write_root(root2, "dry_run", "A_adversaries",
               [r for r in recs if r.design == "A_adversaries"])
    rows = BP.judged_rows(BP.DevData(root2), {"A-H": built["protocols"]["A-R"]},
                          ["A_witnesses", "A_adversaries"],
                          dec=BP.load_decisions(None))
    (pot,) = BP.adversary_potency(rows)
    assert pot["n_paired"] == 0 and pot["share_paired_ratio_ge_0.5"] is None
    assert pot["present_rate"] == 1.0 and pot["potent"] is True


def test_the_dry_run_pairs_the_adversaries_with_the_positive_control():
    run = next(r for r in DC.get_item("dry_run").runs
               if r.name == "A_adversary_controls")
    tasks = DC.runner_tasks(run)
    adv = DC.runner_tasks(next(r for r in DC.get_item("dry_run").runs
                               if r.name == "A_adversaries"))
    assert {t.system for t in tasks} == {"PC_nominal"}
    assert {t.seed for t in tasks} == {t.seed for t in adv} == set(range(340, 352))
    assert not any(t.has_held_out for t in tasks)


def _occupancy(n_min, designation, ok, seeds=range(400, 410)):
    reason = "INSUFFICIENT_OCCUPANCY"
    cell = {"cell_id": "HCv2-12.c/ring/coupling=0.3/T1000", "hypothesis": "HCv2-12",
            "parts": ["c"], "system": "ring", "params": {"coupling": 0.3},
            "expected": {"occupancy": designation, "n_min": n_min, "t_pi_min": 10.0}}
    out = []
    for i, seed in enumerate(seeds):
        good = i < ok
        undefined = (designation == "undefined") == good
        comp = component("IIM", 0.01, 0.001, defined=not undefined, protocol="B",
                         declaration="none")
        if undefined:
            comp = dataclasses.replace(comp, reason=reason)
        sc = REC.ScoringRecord(
            scoring_id="none", declaration_id="none", observation_stage="source",
            view="source", estimator_form="directional", protocol_id="B",
            protocol_hash=PH, components={"IIM": comp})
        out.append(REC.TaskRecord(
            task_id=f"fb-{n_min}-{seed}", design="family_b", family="B",
            system="ring", seed=seed, generator_version="g", status="ok",
            config={"cell": cell}, scorings=(sc,),
            simulation={"ts_sha256": f"{seed:064x}", "n_nodes": 3, "n_time": 1000,
                        "dt": 1.0}))
    return out


def test_the_occupancy_gate_suggests_the_smallest_passing_n_min(tmp_path):
    root = tmp_path / "root"
    write_root(root, "dry_run", "family_b", _occupancy(25, "undefined", 8),
               name=DC.FAMILY_B_JSONL)
    ev = BP.occupancy_evidence(BP.DevData(root))
    assert ev["all_pass"] is False and not isinstance(ev["suggestion"], int)
    assert "occupancy_n_min" in ev["suggestion"]
    write_root(root, "occupancy_n_min", "n_min_50", _occupancy(50, "undefined", 10),
               name=DC.FAMILY_B_JSONL)
    write_root(root, "occupancy_n_min", "n_min_100", _occupancy(100, "undefined", 10),
               name=DC.FAMILY_B_JSONL)
    ev = BP.occupancy_evidence(BP.DevData(root))
    assert ev["all_pass_by_n_min"] == {"25": False, "50": True, "100": True}
    assert ev["suggestion"] == 50
    root2 = tmp_path / "root2"
    write_root(root2, "dry_run", "family_b", _occupancy(25, "defined", 9),
               name=DC.FAMILY_B_JSONL)
    ev = BP.occupancy_evidence(BP.DevData(root2))
    assert ev["all_pass"] is True and ev["suggestion"] == 25


def test_the_raised_n_min_runs_the_occupancy_cells_only():
    item = DC.get_item("occupancy_n_min")
    assert item.optional and [r.name for r in item.runs] == ["n_min_50", "n_min_100"]
    for run, n in zip(item.runs, DC.OCCUPANCY_N_MIN_ALTERNATIVES):
        tasks, _dropped = DC.family_b_tasks(n_min=n)
        assert tasks and all("c" in t.cell.parts for t in tasks)
        assert all(dict(t.cell.expected)["n_min"] == n for t in tasks)
        assert not any(DC.family_b_held_out(t.cell) for t in tasks)
        assert all(t.seed <= 999 for t in tasks)
        assert DC.plan_run(run)["n_tasks"] == len(tasks)


@pytest.mark.parametrize("design, seed, principles, key, values", [
    ("RAM160", 352, ("RAM",), "A-RAM160", {"RAM": 0.4}),
    ("C1_witnesses", 372, None, "C1-R",
     {"NAS:receive": 0.02, "NAS:return": 0.02, "SRPI": 0.2}),
])
def test_a_rejudged_record_equals_the_runner_record(design, seed, principles, key,
                                                    values, tmp_path):
    """The development evaluation re-judges records run under the draft
    protocols (anchors pending) under the generated ones: components,
    reported NAS member, assessment and verdict as the runner writes them
    under those protocols."""
    tasks = RB.build_tasks([design], "development", seeds=[seed])
    task = next(t for t in tasks if t.system == "PC_nominal")
    settings = RB.RunSettings(principles=principles)
    keys = RB.protocol_keys([task])
    draft = RB.resolve_protocols(keys, directory=tmp_path)
    rec0 = RB.run_task(task, draft, settings)
    proto = _anchored(key, values, {k: 0.001 for k in values})
    anchored = RB.resolve_protocols(keys, directory=tmp_path, overrides={key: proto})
    want = RB.run_task(task, anchored, settings)
    got = BP.rejudge_record(rec0, {key: proto})
    compared = 0
    for s_got, s_want in zip(got.scorings, want.scorings):
        if s_want.protocol_id != key:
            continue
        assert s_got.protocol_hash == s_want.protocol_hash == proto.hash
        assert s_got.details["rejudged_from"] != proto.hash
        assert s_got.verdict == s_want.verdict
        for p, cw in s_want.components.items():
            cg = s_got.components[p]
            for f in ("status", "reason", "flags", "c", "c_R", "c_B", "se_c", "df_c",
                      "estimate", "null_mean", "se", "se_df", "se_method",
                      "protocol_hash"):
                assert getattr(cg, f) == pytest.approx(getattr(cw, f), nan_ok=True), (
                    p, f)
            for f in ("assessment", "defined", "estimator_reason", "p_ind"):
                assert cg.details.get(f) == cw.details.get(f), (p, f)
            compared += 1
    assert compared >= 1
    files = tmp_path / "recs.jsonl"
    REC.write_jsonl([rec0], files)
    recs, summary = DC.rejudged_records([files], {key: proto})
    assert summary["n_rejudged"] == 1 and summary["n_not_rejudged"] == 0
    assert recs[0].to_dict()["scorings"] == got.to_dict()["scorings"]


def test_the_development_evaluation_needs_the_generated_protocols(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DC.CalibrationError, match="no generated protocols"):
        DC.run_item("dry_run_evaluation", root=tmp_path / "root",
                    protocol_dir=tmp_path / "empty")


# --------------------------------------------------------------------------
# gates by the systems of their part, smoke seeds, the release at the build,
# writing, the IIM macro-node cap, file order
# --------------------------------------------------------------------------
def _seen(witness, principle, kind=T.KIND_ABSENT, family="A-R", n=12):
    return {"family": family, "principle": principle, "witness": witness,
            "kind": kind, "n": n}


def test_a_per_cell_gate_covers_only_the_systems_of_its_part():
    """HCv2-22 (ii) selects the six single deficits: a witness target outside
    them (W_PDI_no_multistability, dry run only) is no cell of the part."""
    missing = BP._gates_without_rows([_seen("W_PDI_no_multistability", "PDI")], [])
    assert not any("W_PDI_no_multistability" in m for m in missing)
    missing = BP._gates_without_rows([_seen("W_PDI_single_attractor", "PDI")], [])
    assert any(m.startswith("HCv2-22(ii)") and "W_PDI_single_attractor" in m
               for m in missing)
    vocab = BP._hypotheses_spec()["vocabulary"]
    part = {"data": {"where": {"system": "@systems.single_deficits"}}}
    assert BP._part_systems(part, vocab) == list(vocab["systems.single_deficits"])
    assert BP._part_systems({"data": {"where": {}}}, vocab) is None


def test_the_builder_refuses_smoke_seed_records(tmp_path):
    with pytest.raises(BP.BuildError, match="smoke"):
        BP.DevData(_single(tmp_path, _pc(980)))


def test_a_release_whose_predictions_changed_blocks_the_freeze(git_repo, tmp_path):
    pred = git_repo.write("predictions.md", "held-out predictions\n")
    git_repo.commit("predictions")
    root = tmp_path / "root"
    entry = DC.release_held_out_anchors([pred], root=root, repo_root=git_repo.root)
    tags = {"purpose": "reference", "regime": "held_out", "anchor": True,
            "held_out_release": entry["release_id"]}
    write_root(root, "anchors_forward_held_out", "forward_reference",
               [_pc(900, config={"tags": tags})])
    data = BP.DevData(root)
    assert BP.release_problems(data, git_repo.root) == []
    pred.write_text("changed after the release\n", encoding="utf-8")
    (problem,) = BP.release_problems(data, git_repo.root)
    assert entry["release_id"] in problem and "changed since the release" in problem


def test_write_touches_only_the_files_of_builds(dev_root, tmp_path):
    res = BP.build(None, dev_root, keys=["A-R"])
    out = tmp_path / "generated"
    out.mkdir()
    foreign = out / HE.protocol_file_name("template")
    foreign.write_text("{}", encoding="utf-8")
    with pytest.raises(BP.BuildError, match="no build wrote"):
        BP.write(res, out)
    assert foreign.read_text() == "{}" and not (out / BP.MANIFEST).exists()
    foreign.unlink()
    BP.write(res, out)
    assert BP.check(out)["ok"]
    # a later build without a file of the earlier one removes it; a file that
    # no build wrote stays
    (out / "notes.txt").write_text("kept", encoding="utf-8")
    smaller = dict(res, files={k: v for k, v in res["files"].items()
                               if k != BP.MECHANISM_ON})
    BP.write(smaller, out)
    assert not (out / BP.MECHANISM_ON).exists()
    assert (out / "notes.txt").read_text() == "kept"
    assert (out / HE.protocol_file_name("A-R")).exists()


def test_records_under_another_iim_cap_block_the_freeze(tmp_path):
    recs = reference_block(np.random.default_rng(7))
    settings = dict(RB.DEFAULT_SETTINGS.to_dict(), iim_max_macro_nodes=5)
    recs[0] = dataclasses.replace(recs[0], config=dict(recs[0].config,
                                                       settings=settings))
    root = tmp_path / "root"
    write_root(root, "anchors_A", "A_anchors", recs)
    res = BP.build(None, root, keys=["A-R"])
    assert res["evidence"]["iim_grain_cap"]["caps_in_records"] == [4, 5]
    assert any("macro-node cap [5]" in b for b in res["manifest"]["blocking"])
    assert res["manifest"]["run_settings"]["iim_max_macro_nodes"] == 4


def test_the_build_does_not_depend_on_the_order_files_are_found(tmp_path):
    root = tmp_path / "root"
    write_root(root, "anchors_A", "A_anchors",
               reference_block(np.random.default_rng(7)))
    write_root(root, "pdi_concordance", "battery", _battery(12, 1, 6))
    files = DC.result_files(root)
    assert len(files) == 2
    a = BP.build(None, root, keys=["A-R"], files=files)
    b = BP.build(None, root, keys=["A-R"], files=list(reversed(files)))
    assert a["manifest"] == b["manifest"]
    assert {k: p.hash for k, p in a["protocols"].items()} == {
        k: p.hash for k, p in b["protocols"].items()}


# --------------------------------------------------------------------------
# the runner: the release confined to the reference block, restricted runs,
# the concordance battery's run counts
# --------------------------------------------------------------------------
def test_only_reference_block_anchors_are_released():
    task = RB.build_tasks(["forward_anchor_replication"], "development", seeds=[900],
                          purpose="reference")[0]
    assert task.has_held_out
    assert DC.declassify(task, "r1").tags["held_out_release"] == "r1"
    with pytest.raises(DC.CalibrationError, match="reference block"):
        DC.declassify(dataclasses.replace(task, seed=950), "r1")
    dev = RB.build_tasks(["forward_anchor_replication"], "development", seeds=[900],
                         purpose="reference_development")[0]
    with pytest.raises(DC.CalibrationError, match="released"):
        DC.declassify(dev, "r1")


def test_the_release_refuses_existing_outputs_of_a_tagged_run(git_repo, tmp_path):
    pred = git_repo.write("predictions.md", "held-out predictions\n")
    git_repo.commit("predictions")
    root = tmp_path / "root"
    write_root(root, "anchors_forward_held_out@early", "forward_reference",
               [_pc(900)])
    with pytest.raises(DC.CalibrationError, match="exist before the release"):
        DC.release_held_out_anchors([pred], root=root, repo_root=git_repo.root)
    assert not (root / DC.RELEASE_LOG).exists()


def test_a_restriction_applies_to_every_design():
    """Every task of a restricted run is on the requested seeds and systems
    (the null-calibration builder reads seeds as seed bases); a run left
    without tasks is NoTasks, also for a forward design that refuses seeds
    outside its block or a system the design does not have."""
    run = next(r for r in DC.get_item("dry_run").runs if r.name == "null_calibration")
    assert {t.seed for t in DC.runner_tasks(run, seeds=[400])} == {400}
    assert {t.seed for t in DC.runner_tasks(run, seeds=[352])} == {352}
    ram = next(r for r in DC.get_item("dry_run").runs if r.name == "RAM160")
    assert {t.system for t in DC.runner_tasks(ram, systems=["PC_nominal"])} == {
        "PC_nominal"}
    with pytest.raises(DC.NoTasks):
        DC.runner_tasks(ram, systems=["no_such_system"])
    # the sweeps and the factorial take no systems option: their tasks are
    # filtered by the system name (before, the restriction ran all of them)
    sf = next(r for r in DC.get_item("dry_run").runs if r.name == "A_sweeps_factorial")
    assert {t.system for t in DC.runner_tasks(sf, systems=["b00000"])} == {"b00000"}
    with pytest.raises(DC.NoTasks):
        DC.runner_tasks(sf, systems=["PC_nominal"])
    fwd = next(r for r in DC.get_item("dry_run").runs if r.name == "forward_dry_run")
    with pytest.raises(DC.NoTasks):
        DC.runner_tasks(fwd, seeds=[352])
    tasks = DC.runner_tasks(fwd, seeds=[384])
    assert tasks and {t.seed for t in tasks} == {384}


def test_restricted_items_skip_runs_without_tasks(tmp_path, monkeypatch, capsys):
    calls = []

    def fake(item, run, **kw):
        calls.append(run.name)
        if run.name != "RAM160":
            raise DC.NoTasks(f"run {run.name}: no task")
        return {"n_run": 1, "n_errors": 0}

    monkeypatch.setattr(DC, "run_one", fake)
    out = DC.run_item("dry_run", root=tmp_path, seeds=[352])
    assert out["RAM160"] == {"n_run": 1, "n_errors": 0}
    assert all("skipped" in v for k, v in out.items() if k != "RAM160")
    assert calls == [r.name for r in DC.get_item("dry_run").runs]
    with pytest.raises(DC.NoTasks, match="no run has a task"):
        DC.run_item("anchors_C1", root=tmp_path, seeds=[352])
    # without a restriction a run without tasks is an error
    with pytest.raises(DC.NoTasks):
        DC.run_item("anchors_C1", root=tmp_path)
    monkeypatch.undo()
    assert DC.main(["run", "anchors_forward_dev_regime", "--root", str(tmp_path),
                    "--seeds", "352"]) == 2
    assert "no run has a task" in capsys.readouterr().err


def test_the_concordance_battery_run_counts():
    p = DC.plan(["pdi_concordance", "pdi_concordance_extension"], workers=12)
    (base,) = p["items"][0]["runs"]
    (ext,) = p["items"][1]["runs"]
    assert (base["content_on_runs"], base["no_content_runs"]) == (150, 120)
    assert (ext["content_on_runs"], ext["no_content_runs"]) == (200, 160)
    assert base["content_on_runs"] < DC.CONCORDANCE_MIN_RUNS
    assert p["items"][1]["optional"] and not p["items"][0]["optional"]
    tasks = DC.runner_tasks(DC.get_item("pdi_concordance_extension").runs[0])
    assert {t.seed for t in tasks} == set(range(940, 980))
    assert not any(t.smoke or t.has_held_out for t in tasks)


# --------------------------------------------------------------------------
# the held-out G sweep of the Hopf sensor pipeline, own-lesion coverage of
# the reference block, the policy of the development evaluation
# --------------------------------------------------------------------------
def test_iim_on_the_hopf_sensor_sweep_is_left_out_of_the_development_runs():
    """HO-6: IIM on the v2 sensor views of the Hopf arm only at G = 0 in the
    development purposes; the source and BOLD views keep it at every G, and
    the released reference anchors at the held-out regime keep it."""
    vocab = BP._hypotheses_spec()["vocabulary"]
    assert set(DC.HO6_VIEWS) == set(vocab["view.sensor_set"])
    runs = [r for r in DC.get_item("dry_run").runs if r.purpose is not None]
    runs += list(DC.get_item("anchors_forward_dev_regime").runs)
    for run in runs:
        withheld = []
        tasks = DC.runner_tasks(run, withheld=withheld)
        hopf = [t for t in tasks if t.tags.get("arm") == DC.HO6_ARM]
        assert hopf and withheld
        for t in hopf:
            g = float(t.tags["dose"])
            sensor = {s.view for s in t.scorings if "IIM" in s.principles
                      and s.view in DC.HO6_VIEWS}
            assert not sensor if g != 0.0 else sensor == set(DC.HO6_VIEWS)
        # the source view keeps IIM above G = 0
        assert any(float(t.tags["dose"]) != 0.0 and any(
            "IIM" in s.principles and s.view == "source" for s in t.scorings)
            for t in hopf)
        assert DC.plan_run(run)["held_out_iim_withheld"] == len(withheld)
    released = DC.runner_tasks(DC.get_item("anchors_forward_held_out").runs[0],
                               seeds=[900], release={"release_id": "r1"})
    assert any("IIM" in s.principles and s.view in DC.HO6_VIEWS
               for t in released for s in t.scorings)


def _forward_record(view, dose, purpose="dry_run", principle="IIM"):
    stage = "source" if view == "source" else "sensor"
    comp = dataclasses.replace(
        component(principle, 0.01, 0.001, protocol=f"hopf-{view}", declaration="none"),
        observation_stage=stage)
    sc = REC.ScoringRecord(
        scoring_id=f"{view}:primary", declaration_id="none",
        observation_stage=stage, view=view, estimator_form="primary",
        protocol_id=f"hopf-{view}", protocol_hash=PH, components={principle: comp})
    tags = {"arm": "hopf", "purpose": purpose, "regime": "development",
            "dose": dose, "anchor": False}
    return REC.TaskRecord(
        task_id=f"hopf-{purpose}-{dose}-s00384", design="whole_brain",
        family="whole_brain", system=f"hopf_G{dose}", seed=384,
        generator_version="g", status="ok",
        config={"held_out": False, "tags": tags}, scorings=(sc,),
        simulation={"ts_sha256": "0" * 64, "n_nodes": 76, "n_time": 100, "dt": 0.004})


def test_the_builder_refuses_iim_on_the_hopf_sensor_sweep(tmp_path):
    with pytest.raises(BP.BuildError, match="HO-6"):
        BP.DevData(_single(tmp_path, _forward_record("eeg64", 1.142857),
                           "dry_run", "forward_dry_run"))
    # G = 0, the source view and NAS on a sensor view are development data
    for i, rec in enumerate((_forward_record("eeg64", 0.0),
                             _forward_record("source", 1.142857),
                             _forward_record("eeglow", 1.142857, principle="NAS"))):
        root = _single(tmp_path / str(i), rec, "dry_run", "forward_dry_run")
        assert len(BP.DevData(root).records) == 1


def test_a_reference_block_without_its_own_lesions_blocks_the_freeze(tmp_path):
    recs = [r for r in reference_block(np.random.default_rng(7))
            if not (r.system == OWN["PDI"] and r.seed >= 930)]
    root = tmp_path / "root"
    write_root(root, "anchors_A", "A_anchors", recs)
    res = BP.build(None, root, keys=["A-R"])
    info = res["evidence"]["anchors"]["A-R"]
    assert info["n_reference_seeds"] == 40
    assert info["n_lesion_seeds"]["PDI"] == 30 and info["n_lesion_seeds"]["NAS"] == 40
    assert any("own lesion of PDI on 30 of 40" in b
               for b in res["manifest"]["blocking"])


def test_the_development_evaluation_applies_the_development_policy(built, tmp_path):
    protos = BP.write(built, tmp_path / "generated")
    root = tmp_path / "root"
    write_root(root, "dry_run", "A_witnesses", [_pc(981)])
    with pytest.raises(DC.CalibrationError, match="smoke"):
        DC.run_item("dry_run_evaluation", root=root, protocol_dir=protos)
    assert not (root / "dry_run_evaluation").exists()


def test_a_system_restriction_keeps_the_designs_that_have_the_system():
    """The forward arms are one run of three designs: a Hopf condition
    selects the Hopf tasks although the family-A arms do not know it."""
    fwd = next(r for r in DC.get_item("dry_run").runs if r.name == "forward_dry_run")
    tasks = DC.runner_tasks(fwd, systems=["hopf_G0"])
    assert tasks and {t.system for t in tasks} == {"hopf_G0"}
    assert {t.tags["arm"] for t in tasks} == {"hopf"}
    tasks = DC.runner_tasks(fwd, systems=["PC_nominal"])
    assert {t.tags["arm"] for t in tasks} == {"forward_a_eeg", "forward_a_bold"}
    with pytest.raises(DC.NoTasks, match="unknown conditions"):
        DC.runner_tasks(fwd, systems=["no_such_condition"])
