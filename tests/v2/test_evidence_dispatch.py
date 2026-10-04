# -*- coding: utf-8 -*-
"""
The dispatcher between the v1 and the v2 evidence layer: a /2 protocol (or
no protocol) is judged by the frozen v1 functions with every argument
unchanged, so v1 results come back bit for bit, including the stored v1
verdicts re-judged through the dispatcher; a /3 protocol is judged by the
v2 status rule, and the v1 functions refuse it.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from impact_pipeline import evidence as v1
from impact_pipeline import evidence_v2 as E

REPO = Path(__file__).resolve().parents[2]
V1_FILES = [REPO / "protocols" / name for name in (
    "mpc_bench_v1.json", "mpc_bench_v1_anchored.json", "mpc_default_v1.json")]
REGISTRY_V1 = REPO / "protocols" / "applicability_registry_v1.json"
P5 = ("RAM", "PDI", "NAS", "IIM", "SRPI")
FAMILIES = {"RAM": "onset_jitter", "PDI": "circular_shift",
            "NAS": "block_circular_shift", "IIM": "circular_shift",
            "SRPI": "yoked_label_permutation"}


def _dumps(verdict) -> str:
    return json.dumps(verdict.to_dict(), sort_keys=True)


def random_evidence(rng, cls=v1.ComponentEvidence, protocol_id=None):
    ev = {}
    for p in P5:
        if rng.random() < 0.1:
            continue
        channels = ["default"]
        if p == "RAM" and rng.random() < 0.4:
            channels.append(str(rng.choice(["perturbational", "covert_neural"])))
        items = []
        for ch in channels:
            items.append(cls(
                principle=p,
                estimate=float(rng.normal(0.3, 0.6)) if rng.random() > 0.03 else np.nan,
                null_mean=float(rng.normal(0.0, 0.2)),
                null_sd=float(abs(rng.normal(0.1, 0.05))),
                n_null=int(rng.choice([0, 19, 39])),
                se=float(rng.choice([0.0, abs(rng.normal(0.08, 0.1))])),
                se_df=rng.choice([None, 9.0, 19.0]),
                channel=ch,
                defined=bool(rng.random() > 0.05),
                reason=str(rng.choice(["below_null", "NO_DECLARED_WORKSPACE"])),
                reference=float(rng.normal(1.0, 0.4)),
                reference_se=float(abs(rng.normal(0.02, 0.02))),
                reference_scale=str(rng.choice(["estimate", "excess"])),
                bearer_id=str(rng.choice(["system", "system", "other"])),
                protocol_id=protocol_id,
                substrate="synthetic_rate",
                estimator=f"compute_{p}:mode@{p.lower()}-v2-2026.09",
                null_family=(FAMILIES[p] if rng.random() > 0.05 else "other"),
                exact=bool(rng.random() < 0.05),
            ))
        ev[p] = items
    return ev


@pytest.mark.parametrize("path", V1_FILES, ids=lambda p: p.name)
def test_v2_protocols_are_judged_by_v1_bit_for_bit(path):
    rng = np.random.default_rng(11)
    proto = v1.Protocol.from_json(path)
    forms = [proto, str(path), path, json.loads(path.read_text())]
    for _ in range(150):
        ev = random_evidence(rng)
        want = v1.mpc_verdict(ev, proto, require_same_protocol=False)
        for form in forms:
            got = E.mpc_verdict(ev, form, require_same_protocol=False)
            assert type(got) is v1.MPCVerdict
            assert _dumps(got) == _dumps(want)


def test_keyword_arguments_pass_through_unchanged():
    rng = np.random.default_rng(12)
    registry = v1.resolve_registry(str(REGISTRY_V1))
    proto = v1.Protocol.from_json(V1_FILES[0])
    for _ in range(80):
        ev = random_evidence(rng, protocol_id=proto.protocol_id)
        kw = dict(registry=registry, regime={"n_time": 2400, "n_nodes": 32},
                  joint_dependence=bool(rng.random() < 0.5))
        assert _dumps(E.mpc_verdict(ev, proto, **kw)) == _dumps(
            v1.mpc_verdict(ev, proto, **kw))
        kw = dict(necessity_set=("RAM", "NAS", "IIM"), cutoffs=(0.3, 0.1), alpha=0.1,
                  require_same_bearer=False)
        assert _dumps(E.mpc_verdict(ev, None, **kw)) == _dumps(v1.mpc_verdict(ev, **kw))
        assert _dumps(E.mpc_verdict(ev)) == _dumps(v1.mpc_verdict(ev))
    with pytest.raises(TypeError):  # v2-only keywords are not v1 keywords
        E.mpc_verdict({}, proto, uncalibrated_se_methods=())


def test_v2_items_under_a_v1_protocol_use_the_v1_rule():
    """The extra fields of a v2 item (SE method, direction, rank p) never
    reach the v1 path."""
    rng = np.random.default_rng(13)
    proto = v1.Protocol.from_json(V1_FILES[0])
    for _ in range(80):
        seed = int(rng.integers(0, 999))
        plain = random_evidence(np.random.default_rng(seed))
        rich = random_evidence(np.random.default_rng(seed), cls=E.ComponentEvidenceV2)
        got = E.mpc_verdict(rich, proto, require_same_protocol=False)
        want = v1.mpc_verdict(plain, proto, require_same_protocol=False)
        assert _dumps(got) == _dumps(want)


def test_component_assessment_dispatch():
    rng = np.random.default_rng(14)
    proto = v1.Protocol.from_json(V1_FILES[1])
    for _ in range(200):
        ev = random_evidence(rng)
        for items in ev.values():
            for item in items:
                got = E.assess_component(item, proto)
                want = v1.component_assessment(
                    item, cutoff=proto.cutoff_for(item.principle), alpha=proto.alpha)
                assert type(got) is v1.ComponentAssessment
                assert json.dumps(got.to_dict(), sort_keys=True) == json.dumps(
                    want.to_dict(), sort_keys=True)
                assert E.assess_component(item, None, alpha=0.1).to_dict() == (
                    v1.component_assessment(item, alpha=0.1).to_dict())
    with pytest.raises(TypeError):
        E.assess_component(item, proto, alpha=0.1)


def test_v3_protocols_are_judged_by_the_v2_rule():
    ref = {"kind": "external", "scale": "excess", "values": {p: 1.0 for p in P5}}
    p3 = E.ProtocolV3(reference=ref)
    p2 = v1.Protocol(reference=ref)
    ev = v1.ComponentEvidence(principle="RAM", estimate=-0.3, null_mean=0.0, se=0.05,
                              se_df=9.0, reference=1.0, reference_scale="excess")
    assert E.assess_component(ev, p2).status is v1.ComponentStatus.ABSENT
    a = E.assess_component(ev, p3)
    assert isinstance(a, E.AssessmentV2) and a.reason == "NULL_MODEL_VIOLATED"
    assert E.mpc_verdict({"RAM": ev}, p2).component_status["RAM"].value == "ABSENT"
    v = E.mpc_verdict({"RAM": ev}, p3)
    assert isinstance(v, E.MPCVerdictV2)
    assert v.component_status["RAM"].value == "UNDEFINED"
    assert E.mpc_verdict({"RAM": ev}, p3.to_dict()).to_dict() == v.to_dict()
    assert E.is_v3(p3) and E.is_v3(p3.to_dict()) and not E.is_v3(p2)
    assert not E.is_v3(None)


def test_v1_functions_refuse_v3_protocols(tmp_path):
    p3 = E.load_protocol(REPO / "protocols" / "v2" / "mpc_default_v2.json")
    path = tmp_path / "p3.json"
    p3.to_json(path)
    with pytest.raises(TypeError):
        v1.mpc_verdict({}, p3)
    for form in (str(path), p3.to_dict()):
        with pytest.raises(ValueError):
            v1.mpc_verdict({}, form)
    with pytest.raises(TypeError, match="impact-mpc-protocol/3"):
        E.verdict_v3({}, v1.Protocol())
    with pytest.raises(TypeError, match="impact-mpc-protocol/3"):
        E.assess_item(v1.ComponentEvidence(principle="RAM", estimate=0.0), None)


def test_stored_v1_verdicts_are_reproduced_through_the_dispatcher(
        v1_outputs, monkeypatch):
    """The stored C2 bench verdicts, re-judged by the v1 bench code with
    ``evidence.mpc_verdict`` routed through the dispatcher, are unchanged."""
    from scripts.v2 import regression_gate as G

    calls = []

    def routed(evidence, protocol=None, **kw):
        calls.append(type(protocol).__name__)
        return E.mpc_verdict(evidence, protocol, **kw)

    monkeypatch.setattr(v1, "mpc_verdict", routed)
    n = 0
    for rel in ("c2/bench/A/witnesses/results.jsonl",
                "c2/bench/C/witnesses/results.jsonl",
                "c2/bench/A/factorial/results.jsonl"):
        for rec in G.load_jsonl(v1_outputs / rel)[:120]:
            stored = rec.get("verdict")
            if rec.get("status") != "ok" or not stored:
                continue
            new = G.rejudge_bench_record(rec)
            assert G.diff(new, stored) == [], rec["task_id"]
            n += 1
    assert n > 200 and len(calls) == n
    assert set(calls) == {"Protocol"}
