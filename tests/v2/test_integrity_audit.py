# -*- coding: utf-8 -*-
"""The integrity audit IA-1 to IA-10: each check passes on clean records
and detects its planted defect (duplicates, hash mismatches, missing tasks,
an ABSENT with an UNDEFINED reason, ...); the failing records are excluded
from the evaluation; the command line writes the report."""

import copy
import hashlib
import json

import pytest

from impact_pipeline.v2 import hypothesis_engine as HE
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import records as REC
from scripts.v2 import integrity_audit as IA

PASS, FAIL, NOT_RUN = IA.PASS, IA.FAIL, IA.NOT_RUN
VERSION = {
    "RAM": "ram-v3-2026.10",
    "PDI": "pdi-v3-2026.10",
    "NAS": "nas-v3-2026.10",
    "IIM": "iim-v5-2026.10",
    "SRPI": "srpi-v2-2026.09",
}
SE_METHOD = {
    "RAM": "shift_null_sd",
    "PDI": "jackknife_contiguous_10",
    "NAS": "jackknife_contiguous_10",
    "IIM": "circular_block_bootstrap_10pct_B50",
    "SRPI": "jackknife_pairs_10",
}


def h64(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cp(p, status="PRESENT", c=1.0, reason=None, se_c=0.02, df_c=9.0, **kw):
    d = {
        "principle": p,
        "status": status,
        "reason": reason,
        "flags": [],
        "estimator_version": VERSION[p],
        "se_method": SE_METHOD[p],
        "details": {},
    }
    if status != "UNDEFINED" or reason in (
        R.INCONCLUSIVE,
        R.ABSENT_NOT_REACHABLE,
        R.NULL_MODEL_VIOLATED,
    ):
        d.update(
            c=c, estimate=c, null_mean=0.0, se=se_c, se_c=se_c, se_df=df_c, df_c=df_c
        )
    d.update(kw)
    return d


def sc(comps, decl="R", protocol="A-R", stage="source", view="source", form="primary"):
    return {
        "scoring_id": f"{decl}/{view}/{form}",
        "declaration_id": decl,
        "observation_stage": stage,
        "view": view,
        "estimator_form": form,
        "protocol_id": protocol,
        "protocol_hash": h64(protocol),
        "components": {
            c["principle"]: {
                **c,
                "declaration_id": decl,
                "observation_stage": stage,
                "protocol_id": protocol,
                "protocol_hash": h64(protocol),
            }
            for c in comps
        },
        "verdict": None,
        "details": {},
    }


def rec(
    task_id,
    scorings=None,
    *,
    family="A",
    system="PC_nominal",
    seed=100,
    design="A_witnesses",
    replicate=0,
    config=None,
    ts=None,
    struct=None,
    sched=None,
    status="ok",
):
    if scorings is None:
        scorings = [sc([cp(p) for p in HE.PRINCIPLES])]
    for s in scorings:
        for c in s["components"].values():
            c["replicate"] = replicate
    return {
        "schema": "mpc-bench-result/3",
        "task_id": task_id,
        "design": design,
        "family": family,
        "system": system,
        "seed": seed,
        "replicate": replicate,
        "split": "development",
        "generator_version": "mpc-bench-generators/2.0.0",
        "config": dict(config or {}),
        "scorings": scorings,
        "status": status,
        "error": None,
        "timing": {},
        "provenance": {},
        "simulation": {
            "ts_sha256": ts or h64("ts:" + task_id),
            "raw_ts_sha256": None,
            "structural_hash": struct or h64(f"st:{family}:{system}:{seed}"),
            "schedule_hash": sched or h64(f"sc:{task_id}"),
            "n_nodes": 16,
            "n_time": 1000,
            "dt": 0.05,
            "seconds": 1.0,
        },
    }


def clean_records():
    out = [rec(f"w-A-{s}", seed=s) for s in range(100, 106)]
    out += [
        rec(
            f"w-C-{s}",
            family="C1",
            design="C1_witnesses",
            seed=s,
            scorings=[
                sc(
                    [cp("NAS"), cp("IIM")]
                    + [
                        cp(p, "UNDEFINED", reason=R.NOT_APPLICABLE_OBSERVATION_MODEL)
                        for p in ("RAM", "PDI", "SRPI")
                    ],
                    protocol="C1-R",
                )
            ],
        )
        for s in range(100, 103)
    ]
    bold = sc(
        [
            cp("NAS", "UNDEFINED", reason=R.SAMPLING_UNRESOLVED),
            cp("IIM", "UNDEFINED", reason=R.INSUFFICIENT_OCCUPANCY),
        ],
        decl="none",
        protocol="hopf-bold",
        stage="bold",
        view="bold",
    )
    src = sc([cp("NAS"), cp("IIM")], decl="none", protocol="hopf-source")
    out += [
        rec(
            f"h-{s}",
            [src, bold],
            family="hopf",
            system="hopf",
            design="whole_brain",
            seed=s,
            config={"G": 1.0},
        )
        for s in range(500, 503)
    ]
    return out


@pytest.fixture
def spec():
    return HE.load_spec()


def ids(check):
    return sorted(check["excluded_task_ids"])


def kinds(check):
    return sorted({f.get("kind") for f in check["failures"]})


def test_the_clean_records_follow_the_schema():
    for r in clean_records():
        REC.TaskRecord.from_dict(r)


# --------------------------------------------------------------------------
def test_ia1_reads_or_runs_the_regression_gate(tmp_path):
    assert IA.ia1_regression_gate()["status"] == NOT_RUN
    ok = {
        "ok": True,
        "quick": True,
        "version": "g",
        "checks": [{"name": "read_only_files", "status": "PASS"}],
    }
    c = IA.ia1_regression_gate(ok)
    assert c["status"] == PASS and "quick" in c["summary"]
    p = tmp_path / "gate.json"
    p.write_text(
        json.dumps(
            {
                **ok,
                "ok": False,
                "checks": [
                    {
                        "name": "rejudge_records",
                        "status": "FAIL",
                        "summary": "1 difference",
                    }
                ],
            }
        )
    )
    c = IA.ia1_regression_gate(p)
    assert c["status"] == FAIL and c["failures"][0]["check"] == "rejudge_records"


def test_ia2_detects_a_lost_component_and_the_error_rate():
    recs = clean_records()
    assert IA.ia2_component_isolation(recs)["status"] == PASS
    bad = copy.deepcopy(recs[0])
    comps = bad["scorings"][0]["components"]
    comps["IIM"] = {
        **comps["IIM"],
        "status": "UNDEFINED",
        "reason": "ESTIMATOR_ERROR:ValueError",
        "c": None,
    }
    del comps["RAM"]  # the error took another component with it
    c = IA.ia2_component_isolation([bad] + recs[1:])
    assert c["status"] == FAIL and "lost_components" in kinds(c)
    assert bad["task_id"] in ids(c)
    assert "error_rate" in kinds(c)  # 1 error among few components > 1 %
    errored = copy.deepcopy(recs[0])
    errored["status"] = "error"
    errored["error"] = "RuntimeError: boom"
    assert "task_error" in kinds(IA.ia2_component_isolation([errored]))
    # a held-out re-scoring that scores NAS and IIM only loses nothing
    ho = copy.deepcopy(recs[0])
    ho["scorings"].append(sc([cp("NAS"), cp("IIM")], decl="Q10"))
    assert IA.ia2_component_isolation([ho] + recs[1:])["status"] == PASS


def test_ia2_compares_a_scoring_with_the_principles_it_declares():
    """A scoring that declares fewer principles than others of its kind (a
    Hopf lesion run scores NAS alone; a null kind its applicable
    principles) loses nothing; one that declares a principle and lacks it
    does; principles the run's settings leave out are not expected."""
    recs = clean_records()
    hopf = [r for r in recs if r["design"] == "whole_brain"]
    assert hopf
    lesion = copy.deepcopy(hopf[0])
    lesion["task_id"] = "h-lesion"
    src = lesion["scorings"][0]
    del src["components"]["IIM"]
    # without the declaration the scoring is compared with its kind
    c = IA.ia2_component_isolation(recs + [lesion])
    assert kinds(c) == ["missing_components"]
    src["details"] = {"principles": ["NAS"]}
    assert IA.ia2_component_isolation(recs + [lesion])["status"] == PASS
    src["details"] = {"principles": ["NAS", "IIM"]}
    c = IA.ia2_component_isolation(recs + [lesion])
    assert kinds(c) == ["missing_components"] and "h-lesion" in ids(c)
    lesion["config"] = {**(lesion.get("config") or {}),
                        "settings": {"principles": ["NAS"]}}
    assert IA.ia2_component_isolation(recs + [lesion])["status"] == PASS


def test_ia3_nas_sampling_gate(spec):
    recs = clean_records()
    assert IA.ia3_nas_sampling(recs, spec)["status"] == PASS
    bad = copy.deepcopy(recs[-1])
    bad["scorings"][1]["components"]["NAS"] = {
        **bad["scorings"][1]["components"]["NAS"],
        "status": "ABSENT",
        "reason": None,
        "c": 0.0,
    }
    c = IA.ia3_nas_sampling(recs[:-1] + [bad], spec)
    assert c["status"] == FAIL and kinds(c) == ["bold_not_sampling_unresolved"]
    # never undefined by construction on the bench (family A at 20 Hz)
    bench = copy.deepcopy(recs[0])
    bench["scorings"][0]["components"]["NAS"] = {
        **bench["scorings"][0]["components"]["NAS"],
        "status": "UNDEFINED",
        "reason": R.SAMPLING_UNRESOLVED,
        "c": None,
        "estimate": None,
    }
    c = IA.ia3_nas_sampling([bench], spec)
    assert c["status"] == FAIL and kinds(c) == ["nas_undefined_on_bench"]


def test_ia4_detects_an_absent_with_an_undefined_reason(spec):
    recs = clean_records()
    assert IA.ia4_undefined_never_absent(recs, spec)["status"] == PASS
    bad = copy.deepcopy(recs[0])
    bad["scorings"][0]["components"]["PDI"].update(
        status="ABSENT", reason="ABSENT_NOT_REACHABLE", c=0.0
    )
    c = IA.ia4_undefined_never_absent([bad], spec)
    assert c["status"] == FAIL and "ABSENT_with_reason" in kinds(c)
    assert ids(c) == [bad["task_id"]]
    # and the record no longer follows the schema (IA-7 reports it as well)
    with pytest.raises(REC.RecordSchemaError):
        REC.TaskRecord.from_dict(bad)


def test_ia4_degenerate_tpms_not_applicable_and_registry(spec):
    hyper = rec(
        "w-hyper",
        system="O_hypersynchronous",
        scorings=[sc([cp("IIM", "ABSENT", c=0.0)])],
    )
    c1 = copy.deepcopy(clean_records()[6])
    c1["scorings"][0]["components"]["SRPI"].update(status="PRESENT", reason=None, c=1.0)
    unknown = copy.deepcopy(clean_records()[0])
    unknown["scorings"][0]["components"]["RAM"].update(
        status="UNDEFINED", reason="MADE_UP", c=None
    )
    eeg = rec(
        "h-eeg",
        [
            sc(
                [cp("NAS", "ABSENT", c=0.0)],
                decl="none",
                protocol="hopf-eeg64",
                stage="sensor",
                view="eeg64",
            )
        ],
        family="hopf",
        system="hopf",
        design="whole_brain",
        seed=600,
    )
    c = IA.ia4_undefined_never_absent([hyper, c1, unknown, eeg], spec)
    assert {
        "absent_on_degenerate_tpm",
        "c1_not_applicable_violated",
        "reason_not_in_vocabulary",
    } <= set(kinds(c))
    reg = [
        {
            "principle": "NAS",
            "view": "eeg64",
            "admitted_for_present": "no",
            "admitted_for_absent": "no",
        }
    ]
    # the forward arms' records are the admission runs: judged without a
    # registry (they decide it), so the registry direction is not checked
    # on them, whatever their status
    for design in ("whole_brain", "forward_family_a", "forward_family_a_bold",
                   "forward_anchor_replication"):
        adm = {**copy.deepcopy(eeg), "design": design}
        c = IA.ia4_undefined_never_absent([adm], spec, registry=reg)
        assert c["status"] == PASS, design
        assert c["details"]["n_admission_run_components"] == 1
    # a sensor component of any other record is judged under the registry
    other = {**copy.deepcopy(eeg), "design": "paper2_like_sensor_data"}
    c = IA.ia4_undefined_never_absent([other], spec, registry=reg)
    assert kinds(c) == ["ABSENT_without_admitted_direction"]
    assert c["details"]["n_admission_run_components"] == 0
    reg[0]["admitted_for_absent"] = "yes"
    assert IA.ia4_undefined_never_absent([other], spec, registry=reg)["status"] == PASS


def test_ia5_detects_planted_duplicates():
    recs = clean_records()
    assert IA.ia5_duplicates(recs)["status"] == PASS
    same = h64("same")
    cross = [rec("x-A", family="A", ts=same), rec("x-C", family="C1", ts=same)]
    seeds = [rec("s1", seed=101, ts=h64("s")), rec("s2", seed=102, ts=h64("s"))]
    twins = [
        rec("t0", replicate=0, ts=h64("t"), design="A_twins"),
        rec("t1", replicate=1, ts=h64("t"), design="A_twins"),
    ]
    cfg = [
        rec("k1", config={"knobs": {"g_b": 1.0}}, ts=h64("k")),
        rec("k2", config={"knobs": {"g_b": 0.0}}, ts=h64("k")),
    ]
    allowed = [
        rec("a1", design="A_witnesses", ts=h64("a"), config={"knobs": {"g_b": 1.0}}),
        rec(
            "a2",
            design="A_sweeps",
            ts=h64("a"),
            config={"knobs": {"g_b": 1.0}, "sweep_knob": "g_b", "sweep_level": 1.0},
        ),
    ]
    c = IA.ia5_duplicates(recs + cross + seeds + twins + cfg + allowed)
    assert c["status"] == FAIL
    assert kinds(c) == [
        "cross_family",
        "different_configuration",
        "different_seeds",
        "identical_twin_replicates",
    ]
    assert ids(c) == ["k1", "k2", "s1", "s2", "t0", "t1", "x-A", "x-C"]
    assert c["details"]["allowed"][0]["task_ids"] == ["a1", "a2"]


def test_ia6_detects_hash_mismatches_and_declaration_names():
    recs = clean_records()
    assert IA.ia6_protocol_hashes(recs, None)["status"] == NOT_RUN
    frozen = {p: {"hash": h64(p)} for p in ("A-R", "C1-R", "hopf-source", "hopf-bold")}
    assert IA.ia6_protocol_hashes(recs, frozen)["status"] == PASS
    bad = copy.deepcopy(recs[1])
    bad["scorings"][0]["components"]["NAS"]["protocol_hash"] = "0" * 64
    c = IA.ia6_protocol_hashes([bad], frozen)
    assert (
        c["status"] == FAIL
        and kinds(c) == ["hash_mismatch"]
        and ids(c) == [bad["task_id"]]
    )
    # the estimators' declarations equal the protocol's, through the one name map
    from impact_pipeline.v2 import ram_v3

    proto = {
        "A-R": {
            "hash": h64("A-R"),
            "estimators": {
                "RAM": {
                    "facets_not_applicable": {
                        "F": "no_graded_feedback_referent",
                        "G": "no_goal_contingency_referent",
                        "speed": "speed_not_in_construct",
                    }
                },
                "PDI": {"mode": "repertoire", "pdi_bearer": "non_workspace"},
            },
        }
    }
    good = copy.deepcopy(recs[0])
    good["scorings"][0]["components"]["RAM"]["details"] = {
        "facets_not_applicable": dict(ram_v3.FACETS_NOT_APPLICABLE)
    }
    good["scorings"][0]["components"]["PDI"]["details"] = {"bearer": "non_workspace"}
    assert IA.ia6_protocol_hashes([good], proto)["status"] == PASS
    good["scorings"][0]["components"]["PDI"]["details"] = {"bearer": "full"}
    c = IA.ia6_protocol_hashes([good], proto)
    assert kinds(c) == ["pdi_bearer_differs_from_protocol"]


def test_ia7_detects_missing_tasks_and_the_seed_policy():
    recs = clean_records()
    plan = [r["task_id"] for r in recs]
    assert IA.ia7_plan_and_seeds(recs, plan)["status"] == PASS
    c = IA.ia7_plan_and_seeds(recs[1:] + [recs[2]], plan + ["planned-not-run"])
    assert c["status"] == FAIL
    by = {f["kind"]: f for f in c["failures"]}
    assert set(by) == {"missing", "duplicated"}
    assert sorted(f["task_id"] for f in c["failures"] if f["kind"] == "missing") == [
        "planned-not-run",
        recs[0]["task_id"],
    ]
    extra = rec("unplanned", seed=104)
    assert "unexpected" in kinds(IA.ia7_plan_and_seeds(recs + [extra], plan))
    v1 = copy.deepcopy(recs[0])
    v1["seed"] = 10005
    c = IA.ia7_plan_and_seeds([v1])
    assert "seed_policy" in kinds(c) and "schema" in kinds(c)
    c = IA.ia7_plan_and_seeds(recs, split="confirmatory")
    assert kinds(c) == ["mixed_or_wrong_split"]
    c = IA.ia7_plan_and_seeds(
        recs, bad_lines=[{"file": "x", "line": 3, "error": "bad"}]
    )
    assert kinds(c) == ["unreadable_line"]


def test_ia8_identities():
    ep = IA.status_rule_entry_points_agree()
    assert ep["n"] > 400 and ep["disagreements"] == 0
    recs = clean_records()
    st = IA.stored_status_identity(recs)
    assert st["checked"] > 0 and st["mismatches"] == []
    assert IA.ia8_identities(recs)["status"] == PASS
    bad = copy.deepcopy(recs[0])
    bad["scorings"][0]["components"]["RAM"].update(
        c=0.0, estimate=0.0
    )  # stored PRESENT
    c = IA.ia8_identities([bad])
    assert c["status"] == FAIL and kinds(c) == ["stored_status_differs"]
    # the IIM rank gate removes a PRESENT decision: consistent
    gated = copy.deepcopy(recs[0])
    gated["scorings"][0]["components"]["IIM"].update(
        status="UNDEFINED", reason=f"{R.INCONCLUSIVE}:{R.NULL_NOT_EXCEEDED}"
    )
    assert IA.ia8_identities([gated])["status"] == PASS
    assert IA.ia8_identities(recs, {"max_abs_diff": 3.3e-14})["status"] == PASS
    c = IA.ia8_identities(recs, {"max_abs_diff": 1e-9})
    assert kinds(c) == ["nas_v3_v1_identity"]


def test_ia9_v1_srpi_under_the_v2_rule():
    recs = clean_records()
    c = IA.ia9_srpi(recs)
    assert c["status"] == PASS and c["details"]["n_srpi_absent"] == 0
    bad = copy.deepcopy(recs[0])
    bad["scorings"][0]["components"]["SRPI"].update(status="ABSENT", c=0.2)
    sub = copy.deepcopy(recs[1])
    sub["scorings"][0]["components"]["SRPI"].update(status="PRESENT", c=-0.3)
    c = IA.ia9_srpi([bad, sub])
    assert kinds(c) == ["absent_outside_margin", "sub_null_decided"]
    assert c["details"]["n_srpi_absent"] == 1


def test_ia10_twins(spec):
    def twin_set(**over):
        base = rec(
            "w-PC-820", seed=820, struct=h64("net"), sched=h64("s0"), ts=h64("ts0")
        )
        twins = [
            rec(
                f"tw-PC-820-r{r}",
                seed=820,
                design="A_twins",
                replicate=r,
                struct=h64("net"),
                sched=h64(f"s{r}"),
                ts=h64(f"ts{r}"),
            )
            for r in range(0, 4)
        ]
        for k, v in over.items():
            twins[int(k[1:])]["simulation"].update(v)
        return [base] + twins

    assert IA.ia10_twins(twin_set(), spec)["status"] == PASS
    c = IA.ia10_twins(twin_set(r2={"structural_hash": h64("other")}), spec)
    assert kinds(c) == ["structural_hash"]
    c = IA.ia10_twins(twin_set(r3={"schedule_hash": h64("s1")}), spec)
    assert kinds(c) == ["schedule_hash"]
    c = IA.ia10_twins(twin_set(r0={"ts_sha256": h64("other")}), spec)
    assert kinds(c) == ["r0_differs_from_witness"]


# --------------------------------------------------------------------------
def test_run_audit_collects_the_excluded_records(spec):
    recs = clean_records()
    rep = IA.run_audit(
        recs,
        spec=spec,
        planned_task_ids=[r["task_id"] for r in recs],
        gate_report={"ok": True, "quick": False, "checks": []},
    )
    assert rep["ok"] and rep["excluded_task_ids"] == []
    assert rep["not_run"] == ["IA-6"]
    assert rep["records_sha256"] == IA.records_digest(list(reversed(recs)))
    assert [c["id"] for c in rep["checks"]] == [f"IA-{i}" for i in range(1, 11)]
    dup = rec("dup", family="C1", ts=recs[0]["simulation"]["ts_sha256"], seed=100)
    rep = IA.run_audit(recs + [dup], spec=spec)
    assert not rep["ok"]
    assert rep["excluded_task_ids"] == sorted([recs[0]["task_id"], "dup"])


def test_failures_that_name_their_records_do_not_block(spec):
    recs = clean_records()
    gate = {"ok": True, "quick": False, "checks": []}
    planned = [r["task_id"] for r in recs]
    # a task that still failed after its rerun: listed, excluded, and the
    # evaluation goes on without it
    failed = rec("w-A-failed", seed=110, status="error", scorings=[])
    failed["error"] = "LinAlgError: SVD did not converge"
    rep = IA.run_audit(recs + [failed], spec=spec, gate_report=gate,
                       planned_task_ids=planned + ["w-A-failed"])
    assert not rep["ok"] and rep["blocking_failures"] == []
    assert "w-A-failed" in rep["excluded_task_ids"]
    ia2 = next(c for c in rep["checks"] if c["id"] == "IA-2")
    assert ia2["failure_kinds"]["task_error"] == 1
    # a planned task without a record cannot be excluded: it blocks
    rep = IA.run_audit(recs, spec=spec, gate_report=gate,
                       planned_task_ids=planned + ["never-ran"])
    assert rep["blocking_failures"] == [{"id": "IA-7", "kind": "missing", "n": 1}]
    # every failure of the regression gate blocks
    rep = IA.run_audit(recs, spec=spec, planned_task_ids=planned,
                       gate_report={"ok": False, "checks": [
                           {"name": "rejudge_records", "status": "FAIL"}]})
    assert [b["id"] for b in rep["blocking_failures"]] == ["IA-1"]
    # a design's component error rate above the bound is reported only
    c = IA.ia2_component_isolation(recs, max_rate=-1.0)
    assert c["status"] == IA.FAIL and IA.blocking_failures([c]) == []
    # a record judged under another protocol than the frozen one blocks, and
    # so does a seed outside the seed policy
    c = IA.ia6_protocol_hashes(recs, {"A-R": {"hash": "0" * 64}})
    assert c["status"] == IA.FAIL
    assert {b["id"] for b in IA.blocking_failures([c])} == {"IA-6"}
    c = IA.ia7_plan_and_seeds(recs + [rec("v1-seed", seed=10001)])
    assert {(b["id"], b["kind"]) for b in IA.blocking_failures([c])} >= {
        ("IA-7", "seed_policy")}


def test_audit_command_line(tmp_path):
    recs = clean_records()
    f = tmp_path / "records.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([r["task_id"] for r in recs] + ["missing-task"]))
    out = tmp_path / "audit.json"
    rc = IA.main(["--records", str(f), "--plan", str(plan), "--out", str(out)])
    rep = json.loads(out.read_text())
    assert rc == 1 and not rep["ok"]
    ia7 = next(c for c in rep["checks"] if c["id"] == "IA-7")
    assert ia7["failures"] == [{"task_id": "missing-task", "kind": "missing"}]
    plan.write_text(json.dumps([r["task_id"] for r in recs]))
    assert IA.main(["--records", str(f), "--plan", str(plan), "--out", str(out)]) == 0
    assert IA.main(["--records", str(tmp_path / "none")]) == 2


def test_ia7_expects_the_tasks_a_run_curtailed(tmp_path):
    recs = clean_records()
    plan = [r["task_id"] for r in recs] + ["cut-1", "cut-2"]
    c = IA.ia7_plan_and_seeds(recs, plan)
    assert c["status"] == FAIL
    assert sorted(f["task_id"] for f in c["failures"]) == ["cut-1", "cut-2"]
    c = IA.ia7_plan_and_seeds(recs, plan, curtailed_task_ids=["cut-2", "cut-1"])
    assert c["status"] == PASS and c["details"]["n_curtailed"] == 2
    assert "2 tasks curtailed by design" in c["summary"]
    # a curtailed task must be planned and must not have run
    c = IA.ia7_plan_and_seeds(recs, plan, curtailed_task_ids=["cut-1", "cut-2",
                                                              "stray"])
    assert kinds(c) == ["curtailed_outside_plan"]
    c = IA.ia7_plan_and_seeds(recs, plan,
                              curtailed_task_ids=["cut-1", "cut-2",
                                                  recs[0]["task_id"]])
    assert kinds(c) == ["curtailed_but_recorded"]
    # the command line joins the plans of several runs and reads the
    # curtailed tasks from their manifests
    runs = []
    for k, (rs, cut) in enumerate(((recs[:4], ["cut-1"]), (recs[4:], ["cut-2"]))):
        d = tmp_path / f"run{k}"
        d.mkdir()
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
        (d / "run_plan.json").write_text(json.dumps(
            {"task_ids": [r["task_id"] for r in rs] + cut}))
        (d / "run_manifest.json").write_text(json.dumps({"curtailed_task_ids": cut}))
        runs.append(d)
    out = tmp_path / "audit.json"
    rc = IA.main(["--records", *map(str, runs), "--out", str(out),
                  "--plan", *(str(d / "run_plan.json") for d in runs)])
    assert rc == 0
    ia7 = next(c for c in json.loads(out.read_text())["checks"] if c["id"] == "IA-7")
    assert ia7["status"] == PASS and ia7["details"]["n_curtailed"] == 2


def test_ia7_expects_the_smoke_tasks_of_a_development_plan(tmp_path):
    """Smoke tasks (seeds 980-984) are planned, but the runner discards their
    outputs: they are expected to be missing, a record of one fails, and the
    runner's smoke status lines in a run directory are not read as
    records."""
    recs = clean_records()
    plan = [r["task_id"] for r in recs] + ["A_heldout-x-s00980"]
    c = IA.ia7_plan_and_seeds(recs, plan, smoke_task_ids=["A_heldout-x-s00980"])
    assert c["status"] == PASS and c["details"]["n_smoke"] == 1
    assert "1 smoke tasks discarded by design" in c["summary"]
    leaked = copy.deepcopy(recs[0])
    c = IA.ia7_plan_and_seeds(recs, [r["task_id"] for r in recs],
                              smoke_task_ids=[leaked["task_id"]])
    assert kinds(c) == ["smoke_output_recorded"]
    assert leaked["task_id"] in c["excluded_task_ids"]
    # the command line: the plan's smoke tasks come from its seeds, and the
    # smoke status file of the run directory is no record file
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (run / "smoke.jsonl").write_text(json.dumps(
        {"task_id": "A_heldout-x-s00980", "status": "ok", "discarded": True}) + "\n")
    tasks = [{"task_id": r["task_id"], "seed": r["seed"]} for r in recs]
    tasks.append({"task_id": "A_heldout-x-s00980", "seed": 980})
    (run / "run_plan.json").write_text(json.dumps(
        {"task_ids": [t["task_id"] for t in tasks], "tasks": tasks}))
    assert IA._paths([run]) == [run / "results.jsonl"]
    # the evaluator reads the run directory the same way
    from scripts import bench_hypotheses_v2 as BH

    got, bad, files = BH.read_records([run])
    assert files == [run / "results.jsonl"] and not bad and len(got) == len(recs)
    with pytest.raises(FileNotFoundError):
        BH.read_records([tmp_path / "missing"])
    out = tmp_path / "audit.json"
    rc = IA.main(["--records", str(run), "--plan", str(run / "run_plan.json"),
                  "--out", str(out)])
    assert rc == 0
    ia7 = next(c for c in json.loads(out.read_text())["checks"] if c["id"] == "IA-7")
    assert ia7["status"] == PASS and ia7["details"]["n_smoke"] == 1


def test_runner_records_find_their_frozen_protocols_by_key(tmp_path, monkeypatch):
    """The runner writes the protocol key as protocol_id; the evaluator and
    the audit load the frozen files mpc_bench_v2_<key>.json by that key, so
    IA-6 compares every record with its own frozen protocol."""
    import dataclasses

    from impact_pipeline import evidence_v2 as E
    from impact_pipeline.bench import designs_v2 as D
    from impact_pipeline.bench import run_bench_v2 as RB
    from impact_pipeline.bench.designs_v2 import ram_only as RO
    from scripts import bench_hypotheses_v2 as BH

    gen = tmp_path / "generated"
    gen.mkdir()
    for key in ("A-RAM160",):
        d = RB.draft_protocol(key).to_dict()
        d["name"] = f"mpc-bench-v2-{key}"
        RB.protocol_path(key, gen).write_text(json.dumps(d))
    # an anchor table beside the protocols is not a protocol
    (gen / "forward_anchors.json").write_text(json.dumps({"anchors": []}))
    task = RO.arm(D.DEVELOPMENT, seeds=[352], systems=["PC_nominal"])[0]
    task = dataclasses.replace(task, scorings=task.scorings)
    man = RB.run_tasks([task], tmp_path / "run", protocol_dir=gen)
    assert man["protocols"]["A-RAM160"]["source"] == "generated"
    recs, bad = IA.read_raw_records([tmp_path / "run" / RB.RESULTS_JSONL])
    assert not bad
    assert {c["protocol_id"] for r in recs for s in r["scorings"]
            for c in s["components"].values()} == {"A-RAM160"}
    frozen = IA._load_protocols([gen])
    assert list(frozen) == ["A-RAM160"]
    assert IA.ia6_protocol_hashes(recs, frozen)["status"] == PASS
    assert list(BH.load_protocols([gen])) == ["A-RAM160"]
    ctx = HE.build_context(recs, protocols=frozen)
    row = ctx.sources["components"][0]
    assert ctx.protocols[row["protocol_id"]]["hash"] == frozen["A-RAM160"].hash
    # a protocol loaded under another key is a hash mismatch, never a match
    wrong = {"RAM160": frozen["A-RAM160"]}
    assert IA.ia6_protocol_hashes(recs, wrong)["status"] == FAIL
    with pytest.raises(BH.EvaluationRefused, match="mpc_bench_v2_<key>"):
        BH.load_protocols([gen / "forward_anchors.json"])
    assert isinstance(frozen["A-RAM160"], E.ProtocolV3)
