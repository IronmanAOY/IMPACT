# -*- coding: utf-8 -*-
"""Result schema mpc-bench-result/3: one simulation, several scorings,
exact round trips, and the run-hygiene fields that go into a record."""
import copy
import json
import math

import numpy as np
import pytest

from impact_pipeline.v2 import RESULT_SCHEMA
from impact_pipeline.v2 import provenance as P
from impact_pipeline.v2 import records as REC

H_R = "a" * 64
H_H = "b" * 64
TS = np.arange(12.0).reshape(3, 4)


def component(principle, status="PRESENT", reason=None, declaration="R",
              protocol_id="A-R", protocol_hash=H_R, **kw):
    base = dict(
        principle=principle, status=status, reason=reason,
        estimator_version={"NAS": "nas-v3-2026.10", "IIM": "iim-v5-2026.10",
                           "RAM": "ram-v3-2026.10", "PDI": "pdi-v3-2026.10",
                           "SRPI": "srpi-v2-2026.09"}[principle],
        declaration_id=declaration, observation_stage="source",
        protocol_id=protocol_id, protocol_hash=protocol_hash,
        estimate=0.0123, null_mean=0.001, null_sd=0.0004, n_null=19,
        null_family="circular_shift", se=0.002, se_df=9.0,
        se_method="jackknife_contiguous_10", c=0.61, se_c=0.05, df_c=9.4,
        identifiability={
            "shared_inputs": "complete" if declaration == "R" else "partial",
            "observation": "direct", "hub_privileged": "not_tested"},
        seconds=1.25, load_average=(3.5, 2.25, 1.0),
        details={"per_block": [0.1, 0.2], "note": "x"},
    )
    if principle == "NAS":
        base.update(c_R=0.7, c_B=0.61)
    base.update(kw)
    return REC.ComponentRecord(**base)


def scoring(declaration="R", protocol_id="A-R", protocol_hash=H_R, error_in=None):
    comps = {}
    for p in ("RAM", "PDI", "NAS", "IIM", "SRPI"):
        if p == error_in:
            comps[p] = REC.component_error(
                p, ZeroDivisionError("division by zero"),
                estimator_version="iim-v5-2026.10", declaration_id=declaration,
                observation_stage="source", protocol_id=protocol_id,
                protocol_hash=protocol_hash)
        elif p == "SRPI":
            comps[p] = component(p, "UNDEFINED", "ABSENT_NOT_REACHABLE", declaration,
                                 protocol_id, protocol_hash,
                                 flags=("PRESENT_NOT_REACHABLE",))
        elif p == "PDI":
            comps[p] = component(p, "ABSENT", None, declaration, protocol_id,
                                 protocol_hash, flags=("ABSENT_BY_CONCORDANCE",),
                                 se=0.0)
        else:
            comps[p] = component(p, "PRESENT", None, declaration, protocol_id,
                                 protocol_hash)
    return REC.ScoringRecord(
        scoring_id=f"{declaration}/source/primary", declaration_id=declaration,
        observation_stage="source", view="source", estimator_form="primary",
        protocol_id=protocol_id, protocol_hash=protocol_hash, components=comps,
        verdict={"verdict": "UNDETERMINED",
                 "reasons": ["UNDEFINED:SRPI:ABSENT_NOT_REACHABLE"]},
    )


def task(error_in=None, **kw):
    base = dict(
        task_id="witness-A-PC_nominal-s00321", design="witnesses", family="A",
        system="PC_nominal", seed=321, generator_version="mpc-bench-generators/2.0.0",
        status="ok_with_component_errors" if error_in else "ok",
        config={"knobs": {"g_b": 1.0}, "n_trials": 80},
        simulation={"ts_sha256": P.array_sha256(TS), "raw_ts_sha256": None,
                    "structural_hash": "c" * 64, "schedule_hash": "d" * 64,
                    "n_nodes": 30, "n_time": 11552, "dt": 0.05, "seconds": 0.3},
        scorings=(scoring("R", "A-R", H_R, error_in), scoring("H", "A-H", H_H)),
        timing={"total_s": 101.9, "load_average": [4.0, 3.0, 2.0]},
        provenance={"code": {"git_sha": "e" * 40}},
    )
    base.update(kw)
    return REC.TaskRecord(**base)


def test_schema_name_and_v1_schema_untouched():
    from impact_pipeline.bench import run_bench as RB

    assert RESULT_SCHEMA == "mpc-bench-result/3"
    assert RB.RESULT_SCHEMA == "mpc-bench-result/2"
    assert task().to_dict()["schema"] == RESULT_SCHEMA


@pytest.mark.parametrize("error_in", [None, "IIM"])
def test_round_trip_is_exact(error_in):
    rec = task(error_in)
    d = rec.to_dict()
    assert REC.TaskRecord.from_dict(d) == rec
    assert REC.TaskRecord.from_dict(d).to_dict() == d
    text = REC.dumps(rec)
    back = REC.loads(text)
    assert back == rec
    assert REC.dumps(back) == text
    assert json.loads(text) == json.loads(json.dumps(d, sort_keys=True))


def test_round_trip_through_jsonl(tmp_path):
    recs = [task(), task("NAS", task_id="witness-A-PC_nominal-s00322", seed=322)]
    path = tmp_path / "results.jsonl"
    REC.write_jsonl(recs, path)
    assert REC.read_jsonl(path) == recs
    path.write_text(path.read_text() + "\n{\"schema\": \"mpc-bench-result/2\"}\n")
    with pytest.raises(REC.RecordSchemaError, match=":4:"):
        REC.read_jsonl(path)
    # malformed JSON and wrongly typed fields name their line as well
    bad = tmp_path / "bad.jsonl"
    bad.write_text(REC.dumps(recs[0]) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(REC.RecordSchemaError, match=":2:"):
        REC.read_jsonl(bad)
    d = recs[0].to_dict()
    d["scorings"][0]["components"]["NAS"]["flags"] = None
    bad.write_text(json.dumps(d) + "\n", encoding="utf-8")
    with pytest.raises(REC.RecordSchemaError, match=":1:"):
        REC.read_jsonl(bad)
    d = recs[0].to_dict()
    d["scorings"][0]["components"] = ["NAS"]
    bad.write_text(json.dumps(d) + "\n", encoding="utf-8")
    with pytest.raises(REC.RecordSchemaError, match=":1:"):
        REC.read_jsonl(bad)


def test_one_simulation_several_scorings():
    rec = task()
    assert [s.declaration_id for s in rec.scorings] == ["R", "H"]
    shared = [s.components["NAS"].identifiability["shared_inputs"]
              for s in rec.scorings]
    assert shared == ["complete", "partial"]
    assert list(rec.scorings[0].components) == ["RAM", "PDI", "NAS", "IIM", "SRPI"]
    assert rec.split == "development"


def test_component_error_is_isolated_and_sets_the_task_status():
    rec = task("IIM")
    assert rec.status == "ok_with_component_errors"
    iim = rec.scorings[0].components["IIM"]
    assert (iim.status, iim.reason) == (
        "UNDEFINED", "ESTIMATOR_ERROR:ZeroDivisionError")
    assert iim.details["error"] == "ZeroDivisionError: division by zero"
    assert iim.is_estimator_error
    # the other components and the other scoring are untouched
    assert rec.scorings[0].components["NAS"].status == "PRESENT"
    assert rec.scorings[1].components["IIM"].status == "PRESENT"
    assert REC.derive_task_status(rec.scorings) == "ok_with_component_errors"
    assert REC.derive_task_status(task().scorings) == "ok"
    assert REC.derive_task_status((), "RuntimeError: x") == "error"


def test_task_status_must_match_the_record():
    with pytest.raises(REC.RecordSchemaError, match="inconsistent"):
        task("IIM", status="ok")
    with pytest.raises(REC.RecordSchemaError, match="inconsistent"):
        task(status="ok_with_component_errors")
    with pytest.raises(REC.RecordSchemaError, match="inconsistent"):
        task(status="error")
    failed = task(status="error", error="ValueError: bad", scorings=(), simulation=None)
    assert failed.status == "error"
    assert REC.loads(REC.dumps(failed)) == failed
    with pytest.raises(REC.RecordSchemaError, match="simulation"):
        task(simulation=None)


def test_non_finite_numbers_become_null_and_numpy_scalars_are_accepted():
    c = component("NAS", "UNDEFINED", "NON_FINITE_ESTIMATE",
                  estimate=np.float64("nan"), se=np.float32(0.5),
                  n_null=np.int64(19), c=float("inf"),
                  details={"arr": np.array([1.0, np.nan]), "b": np.bool_(True)})
    assert c.estimate is None and c.c is None
    assert c.se == 0.5 and c.n_null == 19 and isinstance(c.n_null, int)
    assert c.details == {"arr": [1.0, None], "b": True}
    assert json.dumps(c.to_dict(), allow_nan=False)


@pytest.mark.parametrize("kw, match", [
    ({"status": "ABSENT", "reason": "INCONCLUSIVE"}, "carries no reason"),
    ({"status": "UNDEFINED", "reason": None}, "no reason given"),
    ({"status": "UNDEFINED", "reason": "WHATEVER"}, "unknown reason"),
    ({"status": "MAYBE"}, "status must be"),
    ({"flags": ("PRESENT_NOT_REACHABLE",)}, "UNDEFINED components only"),
    ({"flags": ("HUB_NOT_PRIVILEGED", "HUB_NOT_PRIVILEGED")}, "duplicate flags"),
    ({"observation_stage": "eeg"}, "observation_stage"),
    ({"protocol_hash": "ABC"}, "SHA-256"),
    ({"protocol_hash": "A" * 64}, "SHA-256"),
    ({"replicate": -1}, "replicate"),
    ({"n_null": 1.5}, "integer"),
    ({"se": -0.1}, "se must be"),
    ({"se_df": 0}, "se_df must be"),
    ({"estimator_version": ""}, "non-empty"),
    ({"identifiability": {"shared_inputs": "all", "observation": "direct",
                          "hub_privileged": True}}, "shared_inputs"),
    ({"identifiability": {"shared_inputs": "none", "observation": "eeg",
                          "hub_privileged": True}}, "observation"),
    ({"identifiability": {"shared_inputs": "none", "observation": "direct",
                          "hub_privileged": 1}}, "hub_privileged"),
    ({"identifiability": {"shared_inputs": "none"}}, "missing keys"),
    ({"load_average": (1.0, 2.0)}, "triple"),
    ({"details": {1: "x"}}, "string"),
    ({"estimate": "0.1"}, "number"),
    ({"replicate": 31}, "twin replicate range"),
    ({"flags": "PRESENT_NOT_REACHABLE"}, "sequence of flag codes"),
    ({"flags": None}, "sequence of flag codes"),
    ({"load_average": 3.0}, "triple"),
    ({"load_average": "1 2 3"}, "triple"),
    # a decided status is a statement about c
    ({"status": "PRESENT", "c": None}, "finite construct value"),
    ({"status": "ABSENT", "c": float("nan")}, "finite construct value"),
])
def test_component_validation(kw, match):
    with pytest.raises(REC.RecordSchemaError, match=match):
        component("NAS", **kw)


def test_per_direction_values_are_nas_only():
    component("NAS", c_R=0.2, c_B=0.3)
    with pytest.raises(REC.RecordSchemaError, match="NAS fields"):
        component("IIM", c_R=0.2)
    d = component("IIM").to_dict()
    d["principle"] = "PHI"
    with pytest.raises(REC.RecordSchemaError, match="principle"):
        REC.ComponentRecord.from_dict(d)


def test_scoring_and_component_must_agree():
    s = scoring()
    d = s.to_dict()
    bad = copy.deepcopy(d)
    bad["components"]["NAS"]["protocol_hash"] = H_H
    with pytest.raises(REC.RecordSchemaError, match="protocol_hash differs"):
        REC.ScoringRecord.from_dict(bad)
    bad = copy.deepcopy(d)
    bad["components"]["IIM"] = bad["components"].pop("NAS")
    with pytest.raises(REC.RecordSchemaError, match="differs from its principle"):
        REC.ScoringRecord.from_dict(bad)
    bad = copy.deepcopy(d)
    bad["verdict"] = {"verdict": "ABSENT"}
    with pytest.raises(REC.RecordSchemaError, match="verdict"):
        REC.ScoringRecord.from_dict(bad)


def test_undefined_components_may_lack_a_construct_value():
    c = component("IIM", "UNDEFINED", "INSUFFICIENT_OCCUPANCY", c=None, se_c=None)
    assert c.c is None
    assert REC.ComponentRecord.from_dict(c.to_dict()) == c
    # the concordance route: ABSENT with se = 0 but a finite c
    d = component("PDI", "ABSENT", None, flags=("ABSENT_BY_CONCORDANCE",), se=0.0,
                  se_c=0.0, c=0.0)
    assert d.c == 0.0 and d.se_c == 0.0


def test_twin_replicates_are_carried_through_the_record():
    comps = {p: component(p, replicate=8) for p in ("NAS", "IIM")}
    s = REC.ScoringRecord(
        scoring_id="R/source/primary", declaration_id="R",
        observation_stage="source", view="source", estimator_form="primary",
        protocol_id="A-R", protocol_hash=H_R, components=comps)
    rec = task(replicate=8, scorings=(s,))
    assert REC.loads(REC.dumps(rec)) == rec
    with pytest.raises(REC.RecordSchemaError, match="twin replicate range"):
        task(replicate=41, scorings=())


def test_task_level_validation():
    with pytest.raises(REC.RecordSchemaError, match="duplicate scoring ids"):
        task(scorings=(scoring(), scoring()))
    with pytest.raises(REC.RecordSchemaError, match="replicate differs"):
        task(replicate=1)
    with pytest.raises(REC.RecordSchemaError, match="v1 confirmatory block"):
        task(seed=10005)
    with pytest.raises(REC.RecordSchemaError, match="unassigned"):
        task(seed=5000)
    with pytest.raises(REC.RecordSchemaError, match="differs from the seed policy"):
        task(split="confirmatory")
    assert task(seed=20001).split == "confirmatory"
    with pytest.raises(REC.RecordSchemaError, match="schema"):
        task(schema="mpc-bench-result/2")
    sim = dict(task().simulation, ts_sha256="z")
    with pytest.raises(REC.RecordSchemaError, match="ts_sha256"):
        task(simulation=sim)
    sim = dict(task().simulation, dt=0)
    with pytest.raises(REC.RecordSchemaError, match="dt"):
        task(simulation=sim)
    with pytest.raises(REC.RecordSchemaError, match="load average"):
        task(timing={"load_average": [1.0]})


@pytest.mark.parametrize("level", ["task", "scoring", "component", "simulation"])
def test_unknown_keys_are_refused(level):
    d = task().to_dict()
    target = {"task": d, "scoring": d["scorings"][0],
              "component": d["scorings"][0]["components"]["RAM"],
              "simulation": d["simulation"]}[level]
    target["surprise"] = 1
    with pytest.raises(REC.RecordSchemaError, match="unknown keys"):
        REC.TaskRecord.from_dict(d)


@pytest.mark.parametrize("key", ["task_id", "seed", "status", "schema"])
def test_missing_required_task_keys_are_refused(key):
    d = task().to_dict()
    del d[key]
    with pytest.raises(REC.RecordSchemaError, match="missing keys"):
        REC.TaskRecord.from_dict(d)


def test_component_rows_are_flat_and_complete():
    rec = task("IIM")
    rows = list(rec.component_rows())
    assert len(rows) == 10
    r = next(x for x in rows
             if x["scoring_id"] == "R/source/primary" and x["principle"] == "SRPI")
    assert r["flags"] == "PRESENT_NOT_REACHABLE"
    assert r["reason"] == "ABSENT_NOT_REACHABLE"
    assert r["identifiability_shared_inputs"] == "complete"
    assert (r["load_1m"], r["load_5m"], r["load_15m"]) == (3.5, 2.25, 1.0)
    assert r["task_status"] == "ok_with_component_errors"
    for key in ("estimate", "null_mean", "null_sd", "null_family", "se", "se_df",
                "se_method", "c", "c_R", "c_B", "status", "declaration_id",
                "observation_stage", "estimator_version", "protocol_id",
                "protocol_hash", "replicate", "seconds"):
        assert key in r
    assert "details" not in r and "identifiability" not in r


# --------------------------------------------------------------------------
# record-level integrity checks
# --------------------------------------------------------------------------
def _sim(ts, raw=None):
    return {"ts_sha256": P.array_sha256(ts),
            "raw_ts_sha256": None if raw is None else P.array_sha256(raw),
            "structural_hash": None, "schedule_hash": None, "n_nodes": 3,
            "n_time": 4, "dt": 0.05, "seconds": None}


def test_duplicate_simulations_are_found_across_families():
    other = TS + 1.0
    a = task(task_id="t-A-1", simulation=_sim(TS))
    b = task(task_id="t-C-1", family="C1", simulation=_sim(TS))
    c = task(task_id="t-A-2", simulation=_sim(other, raw=TS))
    d = task(task_id="t-A-3", simulation=_sim(TS * 3))
    groups = REC.duplicate_simulations([a, b, c, d])
    assert len(groups) == 1
    g = groups[0]
    assert g["task_ids"] == ["t-A-1", "t-C-1"] and g["cross_family"] is True
    assert g["hash_kind"] == "ts_sha256"
    # the raw series of one task equal to the observed series of another is
    # not a duplicate (different hash kinds); equal raw series are
    e = task(task_id="t-A-4", simulation=_sim(TS * 5, raw=TS))
    groups = REC.duplicate_simulations([c, e])
    assert [(g["hash_kind"], g["task_ids"], g["cross_family"]) for g in groups] == [
        ("raw_ts_sha256", ["t-A-2", "t-A-4"], False)]
    failed = task(task_id="t-x", status="error", error="E: x", scorings=(),
                  simulation=None)
    assert REC.duplicate_simulations([failed, d]) == []


def test_protocol_hash_mismatches_are_reported():
    rec = task()
    assert REC.protocol_hash_mismatches([rec], {"A-R": H_R, "A-H": H_H}) == []
    bad = REC.protocol_hash_mismatches([rec], {"A-R": H_R, "A-H": "f" * 64})
    assert {(m["scoring_id"], m["principle"]) for m in bad} == {
        ("H/source/primary", p) for p in ("RAM", "PDI", "NAS", "IIM", "SRPI")}
    unknown = REC.protocol_hash_mismatches([rec], {"A-R": H_R})
    assert {m["protocol_id"] for m in unknown} == {"A-H"}
    assert all(m["frozen_hash"] is None for m in unknown)
    # a scoring without components is checked through its own protocol
    empty = REC.ScoringRecord(
        scoring_id="H/sensor/primary", declaration_id="H",
        observation_stage="sensor", view="eeg64", estimator_form="primary",
        protocol_id="A-H-eeg", protocol_hash=H_H)
    rec2 = task(task_id="t-empty", scorings=(empty,))
    assert REC.protocol_hash_mismatches([rec2], {"A-H-eeg": H_H}) == []
    (m,) = REC.protocol_hash_mismatches([rec2], {"A-H-eeg": "f" * 64})
    assert (m["scoring_id"], m["principle"], m["frozen_hash"]) == (
        "H/sensor/primary", None, "f" * 64)


def test_plan_differences():
    recs = [task(task_id="a"), task(task_id="b"), task(task_id="b"),
            task(task_id="z")]
    res = REC.plan_differences(recs, ["a", "b", "c"])
    assert res == {"missing": ["c"], "unexpected": ["z"], "duplicated": ["b"],
                   "duplicated_in_plan": [], "ok": False}
    assert REC.plan_differences(recs[:2], ["a", "b"])["ok"]
    assert REC.plan_differences([], ["a", "a"])["duplicated_in_plan"] == ["a"]
    # a plan given as a generator is read once
    res = REC.plan_differences(recs[:2], (t for t in ["a", "b", "b"]))
    assert res["duplicated_in_plan"] == ["b"] and res["missing"] == []


def test_plan_differences_scale_to_a_full_run():
    ids = [f"task-{i:06d}" for i in range(200_000)]
    recs = [task(task_id=ids[0]), task(task_id=ids[1])]
    res = REC.plan_differences(recs, ids)
    assert len(res["missing"]) == len(ids) - 2
    assert res["duplicated_in_plan"] == [] and not res["ok"]


# --------------------------------------------------------------------------
# run-hygiene fields of a record
# --------------------------------------------------------------------------
def test_array_hash_is_a_duplicate_detector():
    a = np.random.default_rng(0).normal(size=(5, 7))
    assert P.array_sha256(a) == P.array_sha256(a.copy())
    assert P.array_sha256(a[:, ::2]) == P.array_sha256(np.ascontiguousarray(a[:, ::2]))
    assert P.array_sha256(a.T) == P.array_sha256(np.ascontiguousarray(a.T))
    assert P.array_sha256(a) != P.array_sha256(a.reshape(7, 5))
    assert P.array_sha256(a) != P.array_sha256(a.astype(np.float32))
    b = a.copy()
    b[2, 3] = np.nextafter(b[2, 3], 1.0)
    assert P.array_sha256(a) != P.array_sha256(b)
    assert len(P.array_sha256(a)) == 64


def test_timed_records_seconds_and_load_average():
    with P.timed() as t:
        sum(range(1000))
    assert t["seconds"] is not None and t["seconds"] >= 0
    la = t["load_average"]
    assert la is None or (len(la) == 3 and all(x >= 0 for x in la))
    with pytest.raises(RuntimeError):
        with P.timed() as t2:
            raise RuntimeError("x")
    assert t2["seconds"] is not None


def test_environment_record_names_the_locked_packages():
    env = P.environment_versions()
    assert set(env["packages"]) == {"numpy", "scipy", "numba", "pandas"}
    assert env["packages"]["numpy"] == np.__version__
    assert env["python"].count(".") == 2
    assert math.isfinite(len(env["platform"]))


def test_protocol_file_record_hashes_v1_protocols(repo_root):
    rec = P.protocol_file_record(repo_root / "protocols" / "mpc_bench_v1.json")
    assert rec["protocol_hash"] == (
        "855f6a444b77030d33faa68fcb45e8576b931d2d681cf215d4dacdb57a6b2520")
    assert rec["name"] == "mpc-bench-v1"
    assert rec["sha256"] == P.file_sha256(repo_root / "protocols" / "mpc_bench_v1.json")
    seed_map = P.protocol_file_record(
        repo_root / "protocols" / "v2" / "seed_map_v2.json")
    assert seed_map["protocol_hash"] is None and len(seed_map["sha256"]) == 64
