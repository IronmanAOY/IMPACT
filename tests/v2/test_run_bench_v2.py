# -*- coding: utf-8 -*-
"""
The v2 bench runner: per-estimator isolation, one simulation scored under
several declarations and views, twin plumbing, dispatch by estimator version
(v1 SRPI through the unchanged v1 in-memory path), the protocol option
mapping, the confirmatory guard, held-out and smoke handling, schema /3
records with duplicate hashes, timing and load, and resume from partial
results. Development seeds only; confirmatory task specs are built (never
simulated) to show that the guard refuses them.
"""
import contextlib
import dataclasses
import json
import sys
import types

import numpy as np
import pytest

from impact_pipeline import evidence_v2 as E
from impact_pipeline.bench import adversarial_v2 as A2
from impact_pipeline.bench import designs_v2 as D
from impact_pipeline.bench import run_bench_v2 as RB
from impact_pipeline.bench.designs_v2 import anchors as AN
from impact_pipeline.bench.designs_v2 import family_a as FA
from impact_pipeline.bench.designs_v2 import family_c1 as FC
from impact_pipeline.bench.designs_v2 import ram_only as RO
from impact_pipeline.bench.designs_v2 import twins as TW
from impact_pipeline.bench.generators import BenchSystem
from impact_pipeline.v2 import provenance as PV
from impact_pipeline.v2 import records as REC

P5 = ("RAM", "PDI", "NAS", "IIM", "SRPI")
DEV, CONF = D.DEVELOPMENT, D.CONFIRMATORY
REF = {"kind": "external", "scale": "excess",
       "values": {"RAM": 1.0, "PDI": 1.0, "IIM": 1.0, "SRPI": 1.0,
                  "NAS:receive": 1.0, "NAS:return": 1.0}}
VERSIONS = {"RAM": "ram-v3-2026.10", "PDI": "pdi-v3-2026.10", "NAS": "nas-v3-2026.10",
            "IIM": "iim-v5-2026.10", "SRPI": "srpi-v2-2026.09"}
# Estimator fields each stub reports: the null family and SE contract of the
# protocol template for its principle.
STUB_FIELDS = {
    "RAM": dict(null_family="trial_circular_shift", se_method="shift_null_sd",
                n_null=69, se_df=68.0),
    "PDI": dict(null_family="circular_shift", se_method="jackknife_contiguous_10",
                n_null=19, se_df=9.0),
    "NAS": dict(null_family="block_circular_shift",
                se_method="jackknife_contiguous_10", n_null=19, se_df=9.0),
    "IIM": dict(null_family="circular_shift",
                se_method="circular_block_bootstrap_10pct_B50", n_null=39,
                se_df=12.0, p_ind=0.025),
    "SRPI": dict(null_family="yoked_label_permutation",
                 se_method="jackknife_pairs_10", n_null=200, se_df=9.0),
}


class Stub(RB.Scorer):
    """A scorer with a fixed output (or a fixed exception) that records its
    calls; NAS and IIM read the declared inputs and the basis."""

    kind = "stub"

    def __init__(self, principle, *, estimate=0.8, raise_exc=None, fields=None):
        self.principle, self.version = principle, VERSIONS[principle]
        self.directions = ("receive", "return") if principle == "NAS" else None
        self.uses_declaration = principle in ("NAS", "IIM")
        self.estimate, self.raise_exc = estimate, raise_exc
        self.fields = dict(STUB_FIELDS[principle], **(fields or {}))
        self.calls = []

    def score(self, ctx):
        self.calls.append((ctx.scoring.scoring_id, ctx.view.name,
                           ctx.scoring.declaration_id, ctx.task.task_id))
        if self.raise_exc is not None:
            raise self.raise_exc
        details = {"declaration": ctx.scoring.declaration_id,
                   "observation_model": ctx.observation_model,
                   "null_seed": ctx.null_seed}
        if self.uses_declaration:
            b = ctx.basis(self.principle)
            details.update(basis_columns=b.n_columns, basis_lags=list(b.lags),
                           state_types=list(ctx.declared().state_types),
                           blocks=sorted(ctx.blocks()))
        members = [RB.Member(direction=d, estimate=self.estimate, null_mean=0.0,
                             null_sd=0.1, se=0.05, **self.fields)
                   for d in (self.directions or (None,))]
        return RB.ScorerOutput(f"compute_{self.principle}:stub@{self.version}",
                               members, details)


@contextlib.contextmanager
def stubs(**overrides):
    with contextlib.ExitStack() as st:
        out = {}
        for p in P5:
            out[p] = overrides.get(p) or Stub(p)
            st.enter_context(RB.scorer_override(out[p]))
        yield out


def anchored(keys):
    """Draft protocols of the keys with the external test anchors."""
    out = {}
    for k in keys:
        d = RB.draft_protocol(k).to_dict()
        d["reference"] = REF
        out[k] = RB.ResolvedProtocol(k, E.ProtocolV3.from_dict(d), "override")
    return out


def protocols_for(tasks):
    return anchored(RB.protocol_keys(tasks))


def witness(system="PC_nominal", seed=0, forms=()):
    return FA.witnesses(DEV, seeds=[seed], systems=[system], forms=forms)[0]


def scoring(rec, sid):
    return next(s for s in rec.scorings if s.scoring_id == sid)


def strip_timing(comp):
    d = comp.to_dict()
    d.pop("seconds")
    d.pop("load_average")
    return d


# --------------------------------------------------------------------------
# isolation of every estimator
# --------------------------------------------------------------------------
def test_a_raising_estimator_leaves_the_other_components_intact():
    task = witness()
    protos = protocols_for([task])
    with stubs():
        ok = RB.run_task(task, protos)
    with stubs(NAS=Stub("NAS", raise_exc=ValueError("boom"))) as st:
        bad = RB.run_task(task, protos)
    assert ok.status == REC.TASK_OK
    assert bad.status == REC.TASK_OK_WITH_COMPONENT_ERRORS
    assert len(st["NAS"].calls) == 2  # R and H: each declaration ran it
    for sid in ("R", "H"):
        so, sb = scoring(ok, sid), scoring(bad, sid)
        nas = sb.components["NAS"]
        assert (nas.status, nas.reason) == ("UNDEFINED", "ESTIMATOR_ERROR:ValueError")
        assert nas.is_estimator_error and "boom" in nas.details["error"]
        assert nas.c is None and nas.estimate is None
        for p in ("RAM", "PDI", "IIM", "SRPI"):
            assert strip_timing(so.components[p]) == strip_timing(sb.components[p]), p
            assert sb.components[p].status == "PRESENT"
        assert so.verdict["verdict"] == "MPC_CONSISTENT"
        assert sb.verdict["verdict"] == "UNDETERMINED"
        assert "ESTIMATOR_ERROR:NAS:ValueError" in sb.verdict["reasons"]
    assert ok.scorings[0].components["NAS"].c_R == pytest.approx(0.8)


def test_an_invalid_estimator_output_is_a_component_error_too():
    task = witness()
    with stubs(IIM=Stub("IIM", fields={"p_ind": 2.0})):
        rec = RB.run_task(task, protocols_for([task]))
    assert rec.status == REC.TASK_OK_WITH_COMPONENT_ERRORS
    iim = scoring(rec, "R").components["IIM"]
    assert iim.reason == "ESTIMATOR_ERROR:ValueError"
    assert "p_ind" in iim.details["error"]
    assert scoring(rec, "R").components["RAM"].status == "PRESENT"


@pytest.mark.parametrize("fields, what", [
    ({"se_df": 0.0}, "se_df"),
    ({"null_family": "circular shift"}, "null_family"),
    ({"n_null": -3}, "n_null"),
])
def test_an_output_the_record_refuses_costs_only_its_own_component(fields, what):
    # fields the evidence layer accepts but the /3 record does not
    task = witness()
    with stubs(IIM=Stub("IIM", fields=fields)):
        rec = RB.run_task(task, protocols_for([task]))
    assert rec.status == REC.TASK_OK_WITH_COMPONENT_ERRORS and rec.error is None
    for sid in ("R", "H"):
        s = scoring(rec, sid)
        iim = s.components["IIM"]
        assert iim.reason == "ESTIMATOR_ERROR:RecordSchemaError"
        assert what in iim.details["error"]
        for p in ("RAM", "PDI", "NAS", "SRPI"):
            assert s.components[p].status == "PRESENT", p
        # the verdict was taken again with the component as an error
        assert s.verdict["component_status"]["IIM"] == "UNDEFINED"
        assert "ESTIMATOR_ERROR:IIM:RecordSchemaError" in s.verdict["reasons"]
    REC.loads(REC.dumps(rec))


@pytest.mark.parametrize("directions", [("receive", "receive"),
                                        ("receive", "sideways")])
def test_malformed_directions_cost_only_their_own_component(directions):
    nas = Stub("NAS")
    nas.directions = directions
    task = witness()
    with stubs(NAS=nas):
        rec = RB.run_task(task, protocols_for([task]))
    assert rec.status == REC.TASK_OK_WITH_COMPONENT_ERRORS
    s = scoring(rec, "R")
    assert s.components["NAS"].reason == "ESTIMATOR_ERROR:ValueError"
    assert all(s.components[p].status == "PRESENT" for p in ("RAM", "PDI", "IIM"))


def test_evidence_the_status_rule_refuses_costs_only_its_own_component(monkeypatch):
    real = E.assess_principle

    class Refused(Exception):
        pass

    def picky(principle, items, *a, **k):
        if any(getattr(ev, "estimate", None) == 0.777 for ev in items):
            raise Refused("not assessable")
        return real(principle, items, *a, **k)

    monkeypatch.setattr(E, "assess_principle", picky)
    task = witness()
    with stubs(PDI=Stub("PDI", estimate=0.777)):
        rec = RB.run_task(task, protocols_for([task]))
    assert rec.status == REC.TASK_OK_WITH_COMPONENT_ERRORS
    for sid in ("R", "H"):
        s = scoring(rec, sid)
        assert s.components["PDI"].reason == "ESTIMATOR_ERROR:Refused"
        assert all(s.components[p].status == "PRESENT"
                   for p in ("RAM", "NAS", "IIM", "SRPI"))


def test_an_output_is_reused_only_under_the_same_anchor():
    from impact_pipeline.v2 import ram_v3

    task = dataclasses.replace(
        witness(), scorings=D.make_scorings(FA.PROTOCOLS, ("RAM", "SRPI")))
    protos = protocols_for([task])
    d = protos["A-H"].protocol.to_dict()
    d["reference"] = {**REF, "values": {**REF["values"], "RAM": 100.0}}
    protos["A-H"] = RB.ResolvedProtocol("A-H", E.ProtocolV3.from_dict(d), "override")
    with RB.scorer_override(Stub("SRPI")) as srpi:
        rec = RB.run_task(task, protos)
    assert len(srpi.calls) == 1  # the v1 SRPI reads no anchor: reused under H
    assert scoring(rec, "H").components["SRPI"].details["computed_in_scoring"] == "R"
    r_ram, h_ram = (scoring(rec, s).components["RAM"] for s in ("R", "H"))
    assert "computed_in_scoring" not in h_ram.details  # RAM-PE reads the anchor
    assert r_ram.estimate == h_ram.estimate
    se, df = r_ram.se, r_ram.se_df
    for comp, ref in ((r_ram, 1.0), (h_ram, 100.0)):
        assert comp.details["estimator"]["absent_reachable"] == (
            ram_v3.absent_reachable(se, ref, df))


def test_a_failing_simulation_is_a_task_error():
    def boom(task):
        raise RuntimeError("generator failed")

    RB.register_system_builder("test_boom", boom, replace=True)
    task = dataclasses.replace(witness(), builder="test_boom")
    with stubs():
        rec = RB.run_task(task, protocols_for([task]))
    assert rec.status == REC.TASK_ERROR and "generator failed" in rec.error
    assert rec.simulation is None and rec.scorings == ()
    REC.loads(REC.dumps(rec))


# --------------------------------------------------------------------------
# one simulation, several declarations and views
# --------------------------------------------------------------------------
def _mixing(system, task):
    rng = np.random.default_rng(0)
    n = system.n_nodes
    mix = np.eye(n) + 0.2 * rng.standard_normal((n, n))
    return BenchSystem(ts=mix @ system.ts, events=system.events, meta=system.meta,
                       oracle=system.oracle)


def test_one_simulation_is_scored_under_several_declarations_and_views(monkeypatch):
    built = []
    real = A2.build_system

    def counting(*a, **k):
        built.append(a)
        return real(*a, **k)

    monkeypatch.setattr(A2, "build_system", counting)
    RB.register_view(D.ViewSpec("test_sensor", "sensor", "sensor_mixing",
                                transform=_mixing, regime={"n_sensors": 30}),
                     replace=True)
    base = witness()
    sc = (D.make_scorings(FA.PROTOCOLS, P5)
          + D.make_scorings(FA.PROTOCOLS, ("NAS", "RAM"), view="test_sensor"))
    task = dataclasses.replace(base, scorings=sc)
    with stubs() as st:
        rec = RB.run_task(task, protocols_for([task]))
    assert rec.status == REC.TASK_OK
    assert len(built) == 1  # one simulation
    assert [s.scoring_id for s in rec.scorings] == [
        "R", "H", "R/test_sensor", "H/test_sensor"]
    # declaration-dependent estimators ran per declaration and view; the
    # others once per view, reused by the other declaration
    assert len(st["NAS"].calls) == 4
    assert [c[:2] for c in st["RAM"].calls] == [("R", "source"),
                                                ("R/test_sensor", "test_sensor")]
    assert scoring(rec, "H").components["RAM"].details["computed_in_scoring"] == "R"
    assert "computed_in_scoring" not in scoring(rec, "R").components["RAM"].details
    r_nas = scoring(rec, "R").components["NAS"].details["estimator"]
    h_nas = scoring(rec, "H").components["NAS"].details["estimator"]
    assert r_nas["state_types"] == ["context_cue"] and h_nas["state_types"] == []
    assert r_nas["basis_columns"] > h_nas["basis_columns"]
    assert r_nas["basis_lags"] == [0, 1, 2]
    assert r_nas["blocks"] == ["C", "M", "S", "V"]  # the hub module W is no block
    decl = scoring(rec, "R").details["declaration"]
    assert "slow_phase_sin" in decl["channels"]
    assert scoring(rec, "H").details["declaration"]["shared_inputs"] == "partial"
    view = scoring(rec, "R/test_sensor")
    assert view.observation_stage == "sensor" and view.view == "test_sensor"
    assert view.details["view_ts_sha256"] != rec.simulation["ts_sha256"]
    assert set(view.components) == {"RAM", "NAS"}
    ident = view.components["NAS"].identifiability
    assert ident == {"shared_inputs": "complete", "observation": "sensor_mixing",
                     "hub_privileged": "not_tested"}
    assert scoring(rec, "H").components["NAS"].identifiability["shared_inputs"] == (
        "partial")
    assert scoring(rec, "R").verdict is not None


def test_the_recording_device_logs_only_the_catalogue_channels():
    task = FA.adversaries(DEV, seeds=[340], systems=["adversarial_common_driver"],
                          forms=())[0]
    system = RB.build_catalogue_system(task)
    rec = RB.recorded_inputs(system, task)
    assert rec.driver_groups == ("common_driver",)
    assert not rec.has_context_cues and not rec.has_slow_phase
    w = witness()
    rec_w = RB.recorded_inputs(RB.build_catalogue_system(w), w)
    assert rec_w.has_context_cues and rec_w.has_slow_phase


def test_each_scoring_records_the_bases_its_estimators_used():
    sc = (D.make_scorings(FA.PROTOCOLS, P5, ("iim_bidirectional", "nas_tau_0.2"),
                          form_declarations={"nas_tau_0.2": ("R",)})
          + D.make_scorings({"R": "A-R"}, ("RAM",), bearer_modes=("principle",)))
    task = dataclasses.replace(witness(), scorings=sc)
    with stubs():
        rec = RB.run_task(task, protocols_for([task]))
    bases = {s.scoring_id: s.details.get("basis") for s in rec.scorings}
    assert set(bases["R"]) == set(bases["H"]) == {"NAS:tau_c=0.1", "IIM:tau_c=0.1"}
    assert set(bases["R+iim_bidirectional"]) == {"IIM:tau_c=0.1"}
    assert set(bases["R+nas_tau_0.2"]) == {"NAS:tau_c=0.2"}
    assert bases["R@principle"] is None  # RAM-PE reads no input basis
    nas = bases["R"]["NAS:tau_c=0.1"]
    # the basis is rank-deficient by construction: its effective rank, not its
    # column count, is the conditioning budget
    assert 0 < nas["rank"] < nas["n_columns"] and nas["lags"] == [0, 1, 2]
    assert bases["R+nas_tau_0.2"]["NAS:tau_c=0.2"]["taus"] == [0.2, 0.6, 2.0]
    assert bases["H"]["NAS:tau_c=0.1"]["sha256"] != nas["sha256"]


# --------------------------------------------------------------------------
# twins
# --------------------------------------------------------------------------
def test_twin_sessions_share_the_network_and_differ_in_the_schedule():
    tasks = TW.a_twins(DEV, seeds=[820], systems=["PC_nominal"], replicates=[0, 1, 2])
    run0 = witness(seed=820, forms=("iim_bidirectional",))
    protos = protocols_for(tasks + [run0])
    with stubs() as st:
        recs = [RB.run_task(t, protos) for t in tasks]
        ref = RB.run_task(run0, protos)
    assert [r.replicate for r in recs] == [0, 1, 2]
    assert len({r.simulation["structural_hash"] for r in recs}) == 1
    assert len({r.simulation["schedule_hash"] for r in recs}) == 3
    assert len({r.simulation["ts_sha256"] for r in recs}) == 3
    # replicate 0 is the witness run bit for bit
    assert recs[0].simulation["ts_sha256"] == ref.simulation["ts_sha256"]
    seeds = [r.config["null_seed"] for r in recs]
    assert seeds[0] == 820 * 1000 + 17 and len(set(seeds)) == 3
    assert seeds == [RB.null_seed(820, r) for r in (0, 1, 2)]
    for rec in recs:
        assert rec.config["tags"]["twin_network"] == 820
        for s in rec.scorings:
            for c in s.components.values():
                assert c.replicate == rec.replicate
        nas = scoring(rec, "R").components["NAS"].details["estimator"]
        assert nas["null_seed"] == rec.config["null_seed"]
    assert {c[3] for c in st["RAM"].calls} == {t.task_id for t in tasks} | {
        run0.task_id}
    # RAM-only twins: the agent builder at 160 trials
    rt = TW.ram_twins(DEV, seeds=[820], systems=["eta_0.1"], replicates=[0, 1])
    with stubs():
        rr = [RB.run_task(t, protocols_for(rt)) for t in rt]
    assert rr[0].simulation["structural_hash"] == rr[1].simulation["structural_hash"]
    assert rr[0].simulation["schedule_hash"] != rr[1].simulation["schedule_hash"]
    assert rr[0].simulation["n_time"] > recs[0].simulation["n_time"]
    assert rr[1].config["params"]["preset"] == "ram_only_160"


# --------------------------------------------------------------------------
# dispatch by estimator version
# --------------------------------------------------------------------------
def test_dispatch_by_estimator_version():
    kinds = {v: RB.scorer_for(v, p).kind for p, v in VERSIONS.items()}
    assert kinds == {"ram-v3-2026.10": "ram_v3", "pdi-v3-2026.10": "pdi_v3",
                     "nas-v3-2026.10": "module", "iim-v5-2026.10": "module",
                     "srpi-v2-2026.09": "v1_in_memory"}
    for p, v in (("NAS", "nas-v2-2026.09"), ("IIM", "iim-v4-2026.09"),
                 ("PDI", "pdi-v2-2026.09"), ("RAM", "ram-v2-2026.09")):
        assert RB.scorer_for(v, p).kind == "v1_in_memory"
    tmpl = E.ProtocolV3.from_json(RB.TEMPLATE_PATH)
    for p in P5:
        assert RB.scorer_for(tmpl.estimator_version_for(p), p).principle == p
    with pytest.raises(RB.RunPolicyError, match="no scorer"):
        RB.scorer_for("nas-v9-2030.01", "NAS")
    with pytest.raises(RB.RunPolicyError, match="not NAS"):
        RB.scorer_for("ram-v3-2026.10", "NAS")
    with pytest.raises(RB.RunPolicyError, match="no estimator version"):
        RB.scorer_for(None, "NAS")


def test_v1_srpi_runs_through_the_unchanged_v1_in_memory_path_and_ram_v3_dispatches():
    from impact_pipeline import evidence as v1
    from impact_pipeline.bench import export as X
    from impact_pipeline.v2 import ram_v3

    task = dataclasses.replace(
        witness(), scorings=D.make_scorings({"R": "A-R"}, ("RAM", "SRPI")))
    rec = RB.run_task(task, protocols_for([task]))
    system = RB.build_catalogue_system(task)
    want = X.run_in_memory(system, metrics=["SRPI"], null_surrogates=19,
                           null_seed=0 * 1000 + 17, se_groups=10)
    c = want["components"]["SRPI"]
    # one principle at a time gives the component of a v1 run over several
    # principles (each block draws from its own seed)
    joint = X.run_in_memory(system, metrics=["NAS", "SRPI"], null_surrogates=19,
                            null_seed=0 * 1000 + 17, se_groups=10)
    for k, v in c.items():
        if k not in ("seconds", "se_seconds"):
            assert joint["components"]["SRPI"][k] == v, k
    got = scoring(rec, "R").components["SRPI"]
    assert got.estimator_version == "srpi-v2-2026.09"
    for name in ("estimate", "null_mean", "null_sd", "n_null", "null_family", "se"):
        assert getattr(got, name) == c[name], name
    assert (got.se_df, got.se_method) == (9.0, "jackknife_pairs_10")
    assert got.details["estimator_id"] == v1.estimator_id(
        "SRPI", "agency", "srpi-v2-2026.09")
    assert got.details["estimator"]["se_replicates"] == c["se_replicates"]
    # the v1 instrument judged by the v2 rule on the protocol's anchor
    assert got.c == pytest.approx((c["estimate"] - c["null_mean"]) / 1.0)
    r = ram_v3.compute_ram_v3(system.ts, system.dt, system.events,
                              observation_model="A")
    ram = scoring(rec, "R").components["RAM"]
    assert (ram.estimate, ram.null_mean, ram.se, ram.se_df, ram.se_method) == (
        r["estimate"], r["null_mean"], r["se"], r["se_df"], r["se_method"])
    assert ram.details["estimator_id"] == ram_v3.ESTIMATOR_ID


def test_a_v1_fallback_needs_a_protocol_without_nas_directions():
    task = dataclasses.replace(witness(),
                               scorings=D.make_scorings({"R": "A-R"}, ("NAS",)))
    d = RB.draft_protocol("A-R").to_dict()
    d["estimator_versions"]["NAS"] = "nas-v2-2026.09"
    d["estimators"]["NAS"] = {"mode": "capacity"}
    directional = {"A-R": RB.ResolvedProtocol("A-R", E.ProtocolV3.from_dict(d),
                                              "override")}
    with pytest.raises(RB.RunPolicyError, match="directions"):
        RB.check_plan([task], directional)
    d["directions"] = {}
    plain = {"A-R": RB.ResolvedProtocol("A-R", E.ProtocolV3.from_dict(d), "override")}
    assert RB.check_plan([task], plain)["n_tasks"] == 1
    d["estimators"]["NAS"] = {"mode": "conditional_capacity"}
    wrong = {"A-R": RB.ResolvedProtocol("A-R", E.ProtocolV3.from_dict(d), "override")}
    with pytest.raises(RB.RunPolicyError, match="the v1 path computes"):
        RB.check_plan([task], wrong)


def _fake_nas_module(name, captured, *, hook=False, extra_required=False):
    mod = types.ModuleType(name)

    def direction(c):
        return {"estimate": c, "null_mean": 0.0, "null_sd": 0.1, "n_null": 19,
                "null_family": "block_circular_shift", "se": 0.05, "se_df": 9.0,
                "se_method": "jackknife_contiguous_10", "defined": True,
                "details": {"te": c}}

    if extra_required:
        def compute_nas_v3(ts, dt, mystery_input):  # noqa: ARG001
            raise AssertionError("not reached")
    else:
        def compute_nas_v3(ts, dt, workspace_nodes, blocks, basis, params,
                           null_seed, observation="direct"):
            captured.update(ts_shape=ts.shape, dt=dt, workspace=list(workspace_nodes),
                            blocks=blocks, basis=basis, params=params,
                            null_seed=null_seed, observation=observation)
            return {"estimator_id": "compute_NAS:conditional_capacity@nas-v3-2026.10",
                    "directions": {"R": direction(0.7), "B": direction(0.9)},
                    "details": {"coverage": 1.0}}
    mod.compute_nas_v3 = compute_nas_v3
    if hook:
        def score_bench_v2(ctx):
            captured["hook"] = ctx.scoring.scoring_id
            return RB.ScorerOutput(
                "compute_NAS:hook@nas-v3-2026.10",
                [RB.Member(direction=d, estimate=0.5, null_mean=0.0, null_sd=0.1,
                           **STUB_FIELDS["NAS"]) for d in ("receive", "return")],
                {"via": "hook"})
        mod.score_bench_v2 = score_bench_v2
    return mod


def test_a_module_estimator_is_called_with_the_inputs_its_signature_names(monkeypatch):
    name = "impact_pipeline_test_fake_nas_v3"
    captured = {}
    monkeypatch.setitem(sys.modules, name, _fake_nas_module(name, captured))
    scorer = RB.ModuleScorer("NAS", "nas-v3-2026.10", name, "compute_nas_v3",
                             uses_declaration=True, directions=("receive", "return"),
                             basis_estimator="NAS")
    task = dataclasses.replace(witness(),
                               scorings=D.make_scorings(FA.PROTOCOLS, ("NAS",)))
    with RB.scorer_override(scorer):
        RB.check_plan([task], protocols_for([task]))
        rec = RB.run_task(task, protocols_for([task]))
    nas = scoring(rec, "R").components["NAS"]
    assert nas.status == "PRESENT"
    assert (nas.c_R, nas.c_B, nas.c) == pytest.approx((0.7, 0.9, 0.7))
    assert nas.estimate == 0.7  # the fields of the deciding (smaller) direction
    assert set(nas.details["members"]) == {"receive", "return"}
    assert nas.details["estimator"]["directions"]["return"] == {"te": 0.9}
    assert captured["ts_shape"][0] == 30 and captured["dt"] == 0.05
    assert len(captured["workspace"]) == 6
    assert sorted(captured["blocks"]) == ["C", "M", "S", "V"]
    assert captured["basis"].lags == (0, 1, 2)
    assert captured["basis"].declaration_id == "H"  # the last scoring
    assert captured["params"]["block_representation"] == "block_mean"
    assert captured["null_seed"] == 17 and captured["observation"] == "direct"
    # a module hook takes precedence
    captured.clear()
    monkeypatch.setitem(sys.modules, name, _fake_nas_module(name, captured, hook=True))
    with RB.scorer_override(scorer):
        rec = RB.run_task(task, protocols_for([task]))
    assert captured["hook"] == "H"
    assert scoring(rec, "R").components["NAS"].details["estimator"] == {"via": "hook"}
    # a required input the runner does not know is an estimator error
    monkeypatch.setitem(sys.modules, name,
                        _fake_nas_module(name, captured, extra_required=True))
    with RB.scorer_override(scorer):
        rec = RB.run_task(task, protocols_for([task]))
    nas = scoring(rec, "R").components["NAS"]
    assert nas.reason == "ESTIMATOR_ERROR:TypeError"
    assert "mystery_input" in nas.details["error"]


def test_an_estimator_module_not_merged_yet_is_refused_or_recorded():
    scorer = RB.ModuleScorer("NAS", "nas-v3-2026.10",
                             "impact_pipeline.v2._no_such_estimator_for_tests",
                             "compute_nas_v3", uses_declaration=True,
                             directions=("receive", "return"))
    task = dataclasses.replace(witness(),
                               scorings=D.make_scorings({"R": "A-R"}, ("NAS", "RAM")))
    protos = protocols_for([task])
    with RB.scorer_override(scorer):
        with pytest.raises(RB.RunPolicyError, match="not merged yet: nas-v3"):
            RB.check_plan([task], protos)
        allow = RB.RunSettings(allow_unavailable_estimators=True)
        assert RB.check_plan([task], protos, settings=allow)[
            "unavailable_estimators"] == ["nas-v3-2026.10"]
        only_ram = RB.RunSettings(principles=("RAM",))
        RB.check_plan([task], protos, settings=only_ram)
        rec = RB.run_task(task, protos, allow)
    nas = scoring(rec, "R").components["NAS"]
    assert nas.reason == "ESTIMATOR_ERROR:EstimatorUnavailableError"
    assert "not merged yet" in nas.details["error"]
    assert scoring(rec, "R").components["RAM"].status in ("PRESENT", "UNDEFINED")
    assert not scoring(rec, "R").components["RAM"].is_estimator_error


def test_a_tau_c_form_keeps_the_generator_time_scale_available(monkeypatch):
    # the NAS tau_c sensitivity changes the basis and lags, while the
    # system's own coupling time scale (family A: AgentConfig.tau = 0.1 s)
    # stays available to the estimator's sampling-resolution gate
    name = "impact_pipeline_test_fake_nas_tau"
    seen = []
    mod = types.ModuleType(name)

    def compute_nas_v3(ts, tau_c, system_tau_c, lag_plan, basis):
        seen.append((tau_c, system_tau_c, lag_plan.resolved,
                     lag_plan.nas_lags, basis.taus))
        member = {"estimate": 0.5, "null_mean": 0.0, "null_sd": 0.1, "n_null": 19,
                  "null_family": "block_circular_shift", "se": 0.05, "se_df": 9.0,
                  "se_method": "jackknife_contiguous_10"}
        return {"directions": {"receive": member, "return": member}}

    mod.compute_nas_v3 = compute_nas_v3
    monkeypatch.setitem(sys.modules, name, mod)
    scorer = RB.ModuleScorer("NAS", "nas-v3-2026.10", name, "compute_nas_v3",
                             uses_declaration=True, directions=("receive", "return"))
    task = dataclasses.replace(witness(), scorings=D.make_scorings(
        {"R": "A-R"}, ("NAS",), ("nas_tau_0.05",)))
    with RB.scorer_override(scorer):
        rec = RB.run_task(task, protocols_for([task]))
    assert rec.status == REC.TASK_OK
    (t_r, s_r, ok_r, lags_r, taus_r), (t_f, s_f, ok_f, lags_f, taus_f) = seen
    assert (t_r, s_r, ok_r, lags_r) == (0.1, 0.1, True, (1, 2))
    assert (t_f, s_f, ok_f, lags_f) == (0.05, 0.1, False, (1,))
    assert taus_r == (0.1, 0.3, 1.0) and taus_f == (0.05, 0.15, 0.5)


def test_twin_null_seeds_are_distinct_and_in_the_legacy_seed_range():
    seeds = {(s, r): RB.null_seed(s, r) for s in (0, 820, 999, 20000, 20044)
             for r in range(0, 31)}
    assert all(0 <= v < 2 ** 31 for v in seeds.values())
    assert len(set(seeds.values())) == len(seeds)
    assert RB.null_seed(20044, 0) == 20044 * 1000 + 17  # the v1 rule at r = 0


def test_workers_start_with_one_blas_thread(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    with RB._single_threaded_children():
        assert all(RB.os.environ[v] == "1" for v in RB._THREAD_VARS)
    assert RB.os.environ["OMP_NUM_THREADS"] == "8"
    assert "MKL_NUM_THREADS" not in RB.os.environ


def test_c1_declares_srpi_ram_and_pdi_not_applicable():
    task = FC.witnesses(DEV, seeds=[372], systems=["PC_nominal"], forms=())[0]
    with stubs() as st:
        rec = RB.run_task(task, protocols_for([task]))
    assert rec.status == REC.TASK_OK and rec.family == "C1"
    for p in ("SRPI", "RAM", "PDI"):
        c = scoring(rec, "R").components[p]
        assert (c.status, c.reason) == ("UNDEFINED", "NOT_APPLICABLE_OBSERVATION_MODEL")
        assert "carrier" in c.details["not_applicable"]
        assert st[p].calls == []
    assert set(scoring(rec, "R").components) == {"RAM", "PDI", "NAS", "SRPI"}
    assert scoring(rec, "R").components["NAS"].status == "PRESENT"
    # RAM-PE v3 declares C1 itself (with its descriptive readout)
    with contextlib.ExitStack() as es:
        for p in ("PDI", "NAS", "IIM", "SRPI"):
            es.enter_context(RB.scorer_override(Stub(p)))
        rec = RB.run_task(task, protocols_for([task]))
    ram = scoring(rec, "R").components["RAM"]
    assert ram.reason == "NOT_APPLICABLE_OBSERVATION_MODEL"
    assert ram.details["scorer"] == "ram_v3"
    assert "descriptive" in ram.details["estimator"]


# --------------------------------------------------------------------------
# protocols and their option names
# --------------------------------------------------------------------------
def test_protocol_option_names_map_onto_the_estimators():
    from impact_pipeline.v2 import pdi_v3, ram_v3

    tmpl = E.ProtocolV3.from_json(RB.TEMPLATE_PATH)
    ram = tmpl.estimator_options("RAM")
    assert set(RB.RAM_FACET_NAMES) == set(ram["facets_not_applicable"])
    assert set(RB.RAM_FACET_NAMES.values()) == set(ram_v3.FACETS_NOT_APPLICABLE)
    assert RB.translate_facets(ram["facets_not_applicable"]) == (
        ram_v3.FACETS_NOT_APPLICABLE)
    assert RB.ram_v3_params(ram) == {}
    params, runner = RB.pdi_v3_params(tmpl.estimator_options("PDI"))
    assert params == {"bearer": "non_workspace"} and runner == {"access_module": None}
    assert pdi_v3.PDIParams.from_mapping(params).bearer == pdi_v3.BEARER_NON_WORKSPACE
    with pytest.raises(RB.ProtocolOptionError, match="facets"):
        RB.ram_v3_params({"facets_not_applicable": {"G": "other_reason"}})
    with pytest.raises(RB.ProtocolOptionError, match="unknown RAM facet"):
        RB.translate_facets({"X": "y"})
    with pytest.raises(RB.ProtocolOptionError, match="computes"):
        RB.ram_v3_params({"update": "prediction_error"})
    with pytest.raises(RB.ProtocolOptionError, match="computes"):
        RB.pdi_v3_params({"mode": "labelled"})
    with pytest.raises(ValueError, match="unknown PDI parameters"):
        RB.pdi_v3_params({"pdi_bearer_typo": "full"})
    assert RB.v1_params("SRPI", tmpl.estimator_options("SRPI")) == {}
    assert RB.v1_params("IIM", tmpl.estimator_options("IIM")) == {
        "cut_mode": "directional", "tpm_estimator": "node_shrinkage"}
    # the plan check refuses contradicting options before anything runs
    task = dataclasses.replace(witness(),
                               scorings=D.make_scorings({"R": "A-R"}, ("RAM",)))
    d = RB.draft_protocol("A-R").to_dict()
    d["estimators"]["RAM"]["update"] = "prediction_error"
    bad = {"A-R": RB.ResolvedProtocol("A-R", E.ProtocolV3.from_dict(d), "override")}
    with pytest.raises(RB.RunPolicyError, match="RAM options"):
        RB.check_plan([task], bad)


def test_drafts_follow_the_template_and_admit_exact_known_tpm_values():
    tmpl = E.ProtocolV3.from_json(RB.TEMPLATE_PATH)
    draft = RB.draft_protocol("A-H+nas_secondary")
    assert draft.shared_inputs_declaration == {"id": "H", "shared_inputs": "partial"}
    assert draft.estimator_options("NAS")["block_representation"] == "secondary"
    assert draft.reference["kind"] == "pending"
    assert draft.estimator_versions == tmpl.estimator_versions
    assert E.SE_METHOD_EXACT in draft.se_methods["IIM"]
    assert E.SE_METHOD_EXACT not in tmpl.se_methods["IIM"]
    assert RB.draft_protocol("A-R").hash != RB.draft_protocol("C1-R").hash
    with pytest.raises(RB.RunPolicyError, match="no draft rule"):
        RB.draft_protocol("Z-R")
    exact = E.ComponentEvidenceV2(
        principle="IIM", estimate=0.6, null_mean=0.0, null_sd=0.0, n_null=0, se=0.0,
        exact=True, se_method=E.SE_METHOD_EXACT, null_family="circular_shift",
        estimator="compute_IIM:directional@iim-v5-2026.10")
    anchored_draft = anchored(["A-R"])["A-R"].protocol
    a = E.assess_item(exact, anchored_draft)
    assert (a.status.value, a.route) == ("PRESENT", "exact")
    anchored_tmpl = E.ProtocolV3.from_dict({**tmpl.to_dict(), "reference": REF})
    assert E.assess_item(exact, anchored_tmpl).reason == "INVALID_SE"


def test_generated_protocol_files_take_precedence_over_drafts(tmp_path):
    proto = RB.draft_protocol("A-R")
    d = proto.to_dict()
    d["name"] = "mpc-bench-v2-A-R"
    path = RB.protocol_path("A-R", tmp_path)
    path.write_text(json.dumps(d))
    got = RB.resolve_protocols(["A-R", "A-H"], directory=tmp_path)
    assert got["A-R"].source == "generated" and got["A-R"].path == str(path)
    assert got["A-H"].source == "draft"
    with pytest.raises(RB.RunPolicyError, match="no generated file"):
        RB.resolve_protocols(["A-H"], directory=tmp_path, allow_drafts=False)
    payload = got["A-R"].to_payload()
    assert RB.ResolvedProtocol.from_payload(payload).hash == got["A-R"].hash
    # the run records the file it read (byte hash) beside the content hash
    task = dataclasses.replace(witness(),
                               scorings=D.make_scorings({"R": "A-R"}, ("RAM",)))
    with stubs():
        man = RB.run_tasks([task], tmp_path / "run", protocol_dir=tmp_path)
    (pf,) = man["provenance"]["protocol_files"]
    assert pf["path"] == str(path) and pf["sha256"] == PV.file_sha256(path)
    assert man["protocols"]["A-R"] == {"hash": got["A-R"].hash,
                                       "source": "generated", "path": str(path)}


def test_calibration_pending_parameters_are_declared_once(monkeypatch):
    from impact_pipeline.v2 import pdi_v3

    assert RB.RunSettings().pdi_kmeans_seed == RB.CALIBRATION_PENDING[
        "pdi_kmeans_seed"]["value"]
    assert set(RB.calibration_pending()) == {"pdi_kmeans_seed",
                                             "anchors.replication_extended"}
    seen = []

    def fake(ts, workspace_nodes=None, **kw):
        seen.append(dict(kw, workspace_nodes=workspace_nodes, n=ts.shape[0]))
        return {"defined": True, "reason": None, "estimate": 1.0, "null_mean": 0.0,
                "null_sd": 0.1, "n_null": 19, "null_family": "circular_shift",
                "se": 0.1, "se_df": 9.0, "se_method": "jackknife_contiguous_10",
                "concordant": False, "estimator_id": pdi_v3.ESTIMATOR_ID,
                "details": {"counted_bearer": "non_workspace"}}

    monkeypatch.setattr(pdi_v3, "compute_pdi_v3", fake)
    task = dataclasses.replace(witness("W_PDI_single_attractor"), scorings=(
        D.make_scorings({"R": "A-R"}, ("PDI",), ("pdi_misdeclared_access",))))
    protos = protocols_for([task])
    source = RB.build_catalogue_system(task)
    cache = {}

    def ctx(spec, settings=RB.DEFAULT_SETTINGS):
        return RB.ScoringContext(task, spec, source, source, RB.VIEWS["source"],
                                 protos[spec.protocol_key], "PDI", settings, None,
                                 cache)

    primary, mis = task.scorings
    out = RB.PdiV3Scorer().score(ctx(primary))
    assert seen[-1]["seed"] == 0 and seen[-1]["null_seed"] == 17
    assert seen[-1]["params"] == {"bearer": "non_workspace"}
    assert seen[-1]["observation_model"] == "A"
    assert seen[-1]["workspace_nodes"] == source.meta["workspace_nodes"]
    assert seen[-1]["return_partition"] is True  # tagged for the oracle comparison
    win = out.details["oracle_windows"]
    assert win["n_windows"] == source.n_time // 5 and len(win["ignition"]) == win[
        "n_windows"]
    assert out.members[0].content_bearer == "non_workspace"
    RB.PdiV3Scorer().score(ctx(primary, RB.RunSettings(pdi_kmeans_seed=None)))
    assert seen[-1]["seed"] == 17  # the task's null seed
    out = RB.PdiV3Scorer().score(ctx(mis))
    assert seen[-1]["workspace_nodes"] == source.meta["modules"]["S"]
    assert out.details["declared_access_module"] == "S"


def test_pdi_oracle_windows_follow_the_estimator_windows_and_the_ignition_check():
    from impact_pipeline.bench import manipulation_v2 as MV
    from impact_pipeline.v2 import pdi_v3

    on = RB.build_catalogue_system(witness("W_PDI_single_attractor"))
    off = RB.build_catalogue_system(witness("W_PDI_no_multistability"))
    w = pdi_v3.PDIParams().window
    win = RB.pdi_oracle_windows(on, {"bearer": "non_workspace"})
    assert win["window_samples"] == w and win["n_windows"] == on.n_time // w
    n = win["n_windows"]
    gate = np.asarray(on.oracle["ignition_gate"])[: n * w] > 0.5 * on.meta["knobs"][
        "g_b"]
    assert win["ignition"] == [int(v) for v in gate.reshape(n, w).mean(1) > 0.5]
    assert 0 < sum(win["ignition"]) < n
    assert len(win["context"]) == n
    assert RB.pdi_oracle_windows(on, {"window": 10})["window_samples"] == 10
    # no ignition without the amplification, as the manipulation check says
    assert MV.ignition_occupancy(off)["ignition_occupancy"] == 0.0
    assert not any(RB.pdi_oracle_windows(off, {})["ignition"])


# --------------------------------------------------------------------------
# held-out conditions, smoke tests and the confirmatory guard
# --------------------------------------------------------------------------
def test_held_out_conditions_on_development_seeds_only_as_smoke_tests(tmp_path):
    smoke = [t for t in FA.held_out(DEV, seeds=[980])
             if t.system in ("ADV_NAS_staggered_tau10.tau10", "W_NAS_no_workspace")]
    assert len(smoke) == 2 and all(t.smoke for t in smoke)
    with stubs():
        man = RB.run_tasks(smoke, tmp_path / "smoke")
        again = RB.run_tasks(smoke, tmp_path / "smoke")
    assert man["n_smoke"] == 2 and man["n_run"] == 0
    assert again["n_skipped_done"] == 2 and again["n_smoke"] == 0
    lines = (tmp_path / "smoke" / RB.SMOKE_JSONL).read_text().splitlines()
    rows = [json.loads(x) for x in lines]
    assert {r["task_id"] for r in rows} == {t.task_id for t in smoke}
    assert all(r["discarded"] and r["status"] == "ok" for r in rows)
    text = "\n".join(lines)
    assert "estimate" not in text and '"c"' not in text
    res = tmp_path / "smoke" / RB.RESULTS_JSONL
    assert not res.exists() or res.read_text() == ""
    leaky = FA.catalogue_task("A_heldout", "A", "A", "ADV_NAS_staggered_tau10", 320,
                              D.make_scorings({"R": "A-R"}, ("NAS",)),
                              variant="tau10", held_out=True)
    with pytest.raises(RB.RunPolicyError, match="smoke"):
        RB.check_plan([leaky], protocols_for([leaky]))
    with pytest.raises(RB.RunPolicyError, match="smoke"):
        RB.run_task(leaky, protocols_for([leaky]))


def _frozen_repo(git_repo):
    git_repo.write("src/pkg.py", "x = 1\n")
    git_repo.write("scripts/run.py", "y = 1\n")
    git_repo.commit("code")
    git_repo.git("tag", "-a", PV.FREEZE_TAG_V2, "-m", "freeze")
    return git_repo


def test_the_confirmatory_guard_refuses_a_dirty_tree_a_wrong_tag_or_early_seeds(
        git_repo, tmp_path):
    repo = _frozen_repo(git_repo)
    dev = FA.witnesses(DEV, seeds=[320], systems=["PC_nominal"], forms=())
    conf = FA.witnesses(CONF, seeds=[20000], systems=["PC_nominal"], forms=())
    kw = dict(confirmatory=True, repo_root=repo.root, freeze_tag=PV.FREEZE_TAG_V2,
              protocol_dir=tmp_path / "none")
    with pytest.raises(RB.RunPolicyError, match="seed"):
        RB.run_tasks(dev, tmp_path / "a", **kw)
    with pytest.raises(RB.RunPolicyError, match="not found"):
        RB.run_tasks(conf, tmp_path / "b", **{**kw, "freeze_tag": "no-such-tag"})
    # the guard passes on the clean tagged tree; the run is still refused,
    # because confirmatory runs use generated protocols only
    with pytest.raises(RB.RunPolicyError, match="no generated file"):
        RB.run_tasks(conf, tmp_path / "c", **kw)
    with pytest.raises(RB.RunPolicyError, match="generated protocols only"):
        RB.run_tasks(conf, tmp_path / "c", **{**kw, "allow_drafts": True})
    repo.write("src/pkg.py", "x = 2\n")
    with pytest.raises(RB.RunPolicyError, match="clean tree"):
        RB.run_tasks(conf, tmp_path / "d", **kw)
    repo.commit("after the freeze")
    with pytest.raises(RB.RunPolicyError, match="differ from freeze tag"):
        RB.run_tasks(conf, tmp_path / "e", **kw)
    # a confirmatory task never runs outside the guard, and a development run
    # refuses confirmatory seeds before anything is simulated
    with pytest.raises(RB.RunPolicyError, match="confirmatory guard"):
        RB.run_task(conf[0], protocols_for(conf))
    with pytest.raises(ValueError, match="development run uses seeds"):
        RB.check_plan(conf, protocols_for(conf))
    # a confirmatory run scores every principle with merged estimators and the
    # frozen runner settings
    for settings in (RB.RunSettings(principles=("RAM", "PDI")),
                     RB.RunSettings(allow_unavailable_estimators=True),
                     RB.RunSettings(pdi_kmeans_seed=None)):
        with pytest.raises(RB.RunPolicyError, match="default runner settings"):
            RB.check_plan(conf, protocols_for(conf), confirmatory=True,
                          settings=settings)
    assert not any((tmp_path / x / RB.RESULTS_JSONL).exists() for x in "abcde")


# --------------------------------------------------------------------------
# records, duplicates, timing, resume
# --------------------------------------------------------------------------
def test_schema_3_records_with_duplicate_hashes_timing_and_load(tmp_path):
    tasks = (FA.witnesses(DEV, seeds=[0, 1], systems=["PC_nominal"], forms=())
             + AN.a_anchors(DEV, seeds=[0], systems=["PC_nominal"])
             + FC.witnesses(DEV, seeds=[0], systems=["PC_nominal"], forms=()))
    out = tmp_path / "run"
    with stubs():
        man = RB.run_tasks(tasks, out, protocol_overrides={
            k: v.protocol for k, v in protocols_for(tasks).items()})
    recs = REC.read_jsonl(out / RB.RESULTS_JSONL)
    assert [r.task_id for r in recs] == [t.task_id for t in tasks]
    for r in recs:
        assert r.schema == "mpc-bench-result/3" and r.status == REC.TASK_OK
        assert len(r.simulation["ts_sha256"]) == 64
        assert r.simulation["structural_hash"] and r.simulation["schedule_hash"]
        assert r.timing["total_s"] > 0 and r.timing["simulate_s"] >= 0
        la = r.timing["load_average"]
        assert la is None or len(la) == 3
        assert r.provenance["runner_version"] == RB.RUNNER_VERSION
        assert r.generator_version == "mpc-bench-generators/2.0.0"
        for s in r.scorings:
            for c in s.components.values():
                assert c.protocol_hash == s.protocol_hash
                assert (c.seconds is not None) and (c.load_average is None
                                                    or len(c.load_average) == 3)
    (dup,) = man["duplicate_simulations"]
    assert set(dup["task_ids"]) == {"A_witnesses-PC_nominal-s00000",
                                    "A_anchors-PC_nominal-s00000"}
    assert dup["cross_family"] is False and dup["hash_kind"] == "ts_sha256"
    assert man["plan_differences"]["ok"] and man["n_run"] == 4
    assert man["calibration_pending"] == json.loads(json.dumps(
        RB.calibration_pending()))
    assert man["components_by_design"]["A_anchors"]["n_component_errors"] == 0
    assert man["components_by_design"]["A_witnesses"]["n_components"] == 2 * 2 * 5
    plan = json.loads((out / RB.PLAN_JSON).read_text())
    assert plan["task_ids"] == [t.task_id for t in tasks]
    assert [D.TaskSpec.from_dict(t) for t in plan["tasks"]] == tasks
    rows = (out / RB.COMPONENTS_CSV).read_text().splitlines()
    assert len(rows) - 1 == sum(len(s.components) for r in recs for s in r.scorings)
    assert set(man["protocols"]) == set(RB.protocol_keys(tasks))
    assert REC.protocol_hash_mismatches(
        recs, {k: v["hash"] for k, v in man["protocols"].items()}) == []
    a_r = man["protocols"]["A-R"]["hash"]
    c_r = man["protocols"]["C1-R"]["hash"]
    assert a_r != c_r
    assert REC.protocol_hash_mismatches(recs, {**{k: v["hash"] for k, v in
                                                  man["protocols"].items()},
                                               "C1-R": a_r})


def test_resume_from_partial_results(tmp_path):
    tasks = FA.witnesses(DEV, seeds=[0, 1, 2, 3], systems=["PC_nominal"], forms=())
    out = tmp_path / "run"
    overrides = {k: v.protocol for k, v in protocols_for(tasks).items()}
    with stubs():
        RB.run_tasks(tasks, out, protocol_overrides=overrides)
    path = out / RB.RESULTS_JSONL
    lines = path.read_text().splitlines()
    assert len(lines) == 4
    failed = REC.TaskRecord(task_id=tasks[2].task_id, design="A_witnesses", family="A",
                            system="PC_nominal", seed=2,
                            generator_version="mpc-bench-generators/2.0.0",
                            status=REC.TASK_ERROR, error="RuntimeError: interrupted")
    # task 0 finished, task 1 was cut mid-line, task 2 failed, task 3 never ran
    path.write_text(lines[0] + "\n" + REC.dumps(failed) + "\n" + lines[1][:300])
    with stubs() as st:
        man = RB.run_tasks(tasks, out, protocol_overrides=overrides)
    assert sorted(c[3] for c in st["RAM"].calls) == [t.task_id for t in tasks[1:]]
    assert man["n_skipped_done"] == 1 and man["n_run"] == 3
    assert man["resume"]["n_bad_lines"] == 1
    assert man["resume"]["rerun_failed"] == [tasks[2].task_id]
    assert man["resume"]["rewritten"] is True
    recs = REC.read_jsonl(path)
    assert sorted(r.task_id for r in recs) == sorted(t.task_id for t in tasks)
    assert all(r.status == REC.TASK_OK for r in recs)
    assert man["plan_differences"]["ok"]
    # a finished run resumes to nothing
    with stubs() as st:
        man = RB.run_tasks(tasks, out, protocol_overrides=overrides)
    assert st["RAM"].calls == [] and man["n_skipped_done"] == 4
    # another plan, or starting over, needs another directory: other tasks,
    # the same task ids with other scorings, other settings or other
    # protocols would mix incomparable records
    with stubs(), pytest.raises(RB.RunPolicyError, match="another plan"):
        RB.run_tasks(tasks[:2], out, protocol_overrides=overrides)
    fewer = [dataclasses.replace(t, scorings=t.scorings[:1]) for t in tasks]
    with stubs(), pytest.raises(RB.RunPolicyError, match=r"another plan \(tasks"):
        RB.run_tasks(fewer, out, protocol_overrides=overrides)
    with stubs(), pytest.raises(RB.RunPolicyError, match=r"another plan \(settings"):
        RB.run_tasks(tasks, out, protocol_overrides=overrides,
                     settings=RB.RunSettings(principles=("RAM", "PDI")))
    other = dict(overrides)
    other["A-H"] = other["A-H"].replace(name="another-anchor-set")
    with stubs(), pytest.raises(RB.RunPolicyError, match=r"another plan \(protocols"):
        RB.run_tasks(tasks, out, protocol_overrides=other)
    with stubs(), pytest.raises(RB.RunPolicyError, match="exists"):
        RB.run_tasks(tasks, out, protocol_overrides=overrides, resume=False)


def test_parallel_workers_give_the_serial_records(tmp_path):
    tasks = RO.arm(DEV, seeds=[352], systems=["PC_nominal", "eta_0.1"])
    serial = RB.run_tasks(tasks, tmp_path / "serial")
    parallel = RB.run_tasks(tasks, tmp_path / "parallel", n_workers=2)
    assert serial["n_run"] == parallel["n_run"] == 2
    a = {r.task_id: r for r in REC.read_jsonl(tmp_path / "serial" / RB.RESULTS_JSONL)}
    b = {r.task_id: r for r in REC.read_jsonl(tmp_path / "parallel" / RB.RESULTS_JSONL)}
    assert set(a) == set(b)
    for tid in a:
        assert a[tid].simulation["ts_sha256"] == b[tid].simulation["ts_sha256"]
        ca = a[tid].scorings[0].components["RAM"]
        cb = b[tid].scorings[0].components["RAM"]
        assert (ca.estimate, ca.null_mean, ca.se) == (cb.estimate, cb.null_mean, cb.se)
        assert ca.estimate is not None


def test_the_cli_runs_a_small_development_plan(tmp_path, capsys):
    rc = RB.main(["run", "RAM160", "--split", "development", "--seeds", "352",
                  "--systems", "PC_nominal", "--out", str(tmp_path / "cli")])
    assert rc == 0
    recs = REC.read_jsonl(tmp_path / "cli" / RB.RESULTS_JSONL)
    assert [r.task_id for r in recs] == ["RAM160-PC_nominal-s00352"]
    man = json.loads((tmp_path / "cli" / RB.MANIFEST).read_text())
    assert man["protocols"]["A-RAM160"]["source"] == "draft"
    assert "1 task(s) run" in capsys.readouterr().out
    assert RB.main(["run", "RAM160", "--split", "development", "--seeds", "20000",
                    "--out", str(tmp_path / "x")]) == 2


def test_the_manipulation_runner_reports_completeness_apart_from_pass_fail(tmp_path):
    out = tmp_path / "manipulation"
    assert RB.main(["manipulation", "--seeds", "0", "--families", "A",
                    "--out", str(out)]) == 0
    man = json.loads((out / "manipulation_manifest.json").read_text())
    assert man["complete"] is True and man["split"] == "development"
    assert man["confirmatory"] is False and man["seeds"] == [0]
    assert isinstance(man["all_usable"], bool)
    for name in ("switches", "realisation", "summary"):
        assert (out / f"manipulation_{name}.csv").stat().st_size > 0
    with pytest.raises(ValueError, match="development run uses seeds"):
        RB.run_manipulation([20000], tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
