# -*- coding: utf-8 -*-
"""
Protocol schema ``impact-mpc-protocol/3``: the two shipped /3 protocols load
with stable hashes and are stored canonically, every frozen v1 /2 protocol
keeps its hash and is still read by the v1 code, the /3 fields are validated
and hash-covered, and the anchor and precision blocks follow the rules of
the design (anchor validity and specificity, N_anch and its fallback, the
testability table).
"""
import copy
import hashlib
import json
import math
from dataclasses import replace as dataclasses_replace
from pathlib import Path

import numpy as np
import pytest

from impact_pipeline import evidence as v1
from impact_pipeline import evidence_v2 as E
from impact_pipeline.v2 import ESTIMATOR_VERSIONS_V2
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import testability as T

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "protocols" / "v2" / "mpc_bench_v2_template.json"
DEFAULT = REPO / "protocols" / "v2" / "mpc_default_v2.json"
HASHES_V3 = {
    TEMPLATE: "00a6dd9cbdf91219aa96f078d7411028d4c8131777ec34e372236e5cb081f1ad",
    DEFAULT: "c72c2c7d3a1922b448a5ef7ff2288c4b4987a7d7f1d5931e17e348b736eb549e",
}
# frozen with mpcbench-freeze-v1 (also checked by the v1 regression gate)
HASHES_V1 = {
    "protocols/mpc_bench_v1.json":
        "855f6a444b77030d33faa68fcb45e8576b931d2d681cf215d4dacdb57a6b2520",
    "protocols/mpc_bench_v1_anchored.json":
        "780581f57d24251fc8ed8c39565f0a89a96740908f2ada3f65f8961cf1543f4c",
    "protocols/mpc_default_v1.json":
        "383eb310cf4479d3260bc5f43f8972bd0b12104a221579fea17c40e410357267",
}
P5 = ("RAM", "PDI", "NAS", "IIM", "SRPI")
REF = {"kind": "external", "scale": "excess",
       "values": {"RAM": 1.0, "PDI": 1.0, "IIM": 1.0, "SRPI": 1.0,
                  "NAS:receive": 1.0, "NAS:return": 1.0}}
CELL = {"principle": "PDI", "substrate": "family_a", "observation_stage": "source",
        "view": "source", "bearer": "non_workspace", "absent": True, "present": False}


# --------------------------------------------------------------------------
# shipped files and hashes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", [TEMPLATE, DEFAULT], ids=["template", "default"])
def test_v3_protocol_files_have_stable_hashes(path):
    proto = E.load_protocol(path)
    assert isinstance(proto, E.ProtocolV3)
    assert E.protocol_schema(path) == "impact-mpc-protocol/3"
    assert proto.hash == HASHES_V3[path]
    assert proto.protocol_id == f"sha256:{HASHES_V3[path]}"
    # stored canonically: exactly what to_json writes
    assert path.read_text(encoding="utf-8") == proto.to_json() + "\n"
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == proto.hash


@pytest.mark.parametrize("rel, want", sorted(HASHES_V1.items()))
def test_every_v1_protocol_hash_is_unchanged(rel, want):
    path = REPO / rel
    assert v1.Protocol.from_json(path).hash == want
    loaded = E.load_protocol(path)
    assert type(loaded) is v1.Protocol and loaded.hash == want
    assert E.protocol_schema(path) == "impact-mpc-protocol/2"
    assert E.load_protocol(json.loads(path.read_text())).hash == want


def test_v3_hash_is_canonical_and_round_trips():
    proto = E.load_protocol(TEMPLATE)
    d = proto.to_dict()
    shuffled = {k: d[k] for k in reversed(list(d))}
    assert E.ProtocolV3.from_dict(shuffled).hash == proto.hash
    assert E.ProtocolV3.from_dict(json.loads(proto.to_json())) == proto
    assert proto.replace(name=proto.name).hash == proto.hash
    assert proto.replace(name="other").hash != proto.hash
    assert hash(proto) == hash(E.ProtocolV3.from_dict(d))
    assert proto != E.load_protocol(DEFAULT)
    without_schema = {k: v for k, v in d.items() if k != "schema"}
    assert E.ProtocolV3.from_dict(without_schema) == proto


def test_v1_code_refuses_v3_protocols():
    with pytest.raises(ValueError, match="unsupported protocol schema"):
        v1.Protocol.from_json(TEMPLATE)
    with pytest.raises(TypeError):
        v1.mpc_verdict({}, E.load_protocol(TEMPLATE))
    with pytest.raises(ValueError):
        v1.mpc_verdict({}, str(TEMPLATE))


def test_schema_detection():
    d2 = json.loads((REPO / "protocols" / "mpc_bench_v1.json").read_text())
    with pytest.raises(ValueError, match="not an impact-mpc-protocol/3"):
        E.ProtocolV3.from_dict(d2)
    no_schema = {k: v for k, v in d2.items() if k != "schema"}
    assert E.protocol_schema(no_schema) == "impact-mpc-protocol/2"
    with_rule = {**no_schema, "status_rule": {}}
    assert E.protocol_schema(with_rule) == "impact-mpc-protocol/3"
    with pytest.raises(ValueError, match="status_rule block has schema"):
        E.protocol_schema({**d2, "status_rule": E.StatusRule().to_dict()})
    with pytest.raises(ValueError, match="unsupported"):
        E.protocol_schema({"schema": "impact-mpc-protocol/9"})
    d3 = json.loads(TEMPLATE.read_text())
    with pytest.raises(ValueError, match="needs a status_rule"):
        E.ProtocolV3.from_dict({k: v for k, v in d3.items() if k != "status_rule"})
    with pytest.raises(ValueError, match="unknown fields"):
        E.ProtocolV3.from_dict({**d3, "colour": "red"})
    with pytest.raises(TypeError):
        E.load_protocol(42)


def test_template_contents():
    t = E.load_protocol(TEMPLATE)
    assert t.status_rule == E.StatusRule()
    assert t.necessity_set == P5 and t.declared_necessity_set == P5
    assert t.directions_for("NAS") == ("receive", "return")
    assert t.rank_gate_for("IIM") == 0.05 and t.rank_gate_for("NAS") is None
    for p in P5:
        assert t.estimator_version_for(p) == ESTIMATOR_VERSIONS_V2[p][0]
        assert t.cutoff_for(p) == (0.25, 0.10)
        assert t.channels_for(p) == ("default",)
        for m in t.se_methods_for(p):
            assert p in E.SE_METHODS[m].principles
    assert t.reference["kind"] == "pending" and t.reference_for("RAM") == {}
    assert t.null_families["RAM"] == "trial_circular_shift"
    assert t.estimator_options("PDI")["pdi_bearer"] == "non_workspace"
    assert t.estimator_options("RAM")["facets_not_applicable"] == {
        "F": "no_graded_feedback_referent", "G": "no_goal_contingency_referent",
        "speed": "speed_not_in_construct"}
    assert t.concordance_route == () and t.anchors is None and t.precision is None
    assert t.shared_inputs_declaration is None and t.fallback is None


def test_default_contents_keep_the_v1_stance():
    d = E.load_protocol(DEFAULT)
    old = v1.Protocol.from_json(REPO / "protocols" / "mpc_default_v1.json")
    assert d.status_rule == E.StatusRule()
    for name in ("necessity_set", "channels", "cutoffs", "alpha", "source_rule"):
        assert getattr(d, name) == getattr(old, name), name
    assert dict(d.reference) == dict(old.reference)
    assert d.channels_for("RAM") == ("default", "perturbational", "endogenous")
    for p in ("IIM", "NAS", "PDI"):
        assert d.null_families[p] == old.null_families[p]
    assert dict(d.shared_inputs_declaration) == {"id": "none", "shared_inputs": "none"}
    assert d.estimator_version_for("SRPI") == "srpi-v2-2026.09"


def test_template_gives_no_status_without_anchors():
    t = E.load_protocol(TEMPLATE)
    ev = {p: E.ComponentEvidenceV2(principle=p, estimate=0.0, null_mean=0.0, se=0.01,
                                   se_df=9.0, reference=1.0, reference_scale="excess",
                                   direction=("receive" if p == "NAS" else None),
                                   protocol_id=t.protocol_id) for p in P5}
    v = E.mpc_verdict(ev, t, require_same_protocol=False)
    assert v.verdict is v1.Verdict.UNDETERMINED
    assert all(s is v1.ComponentStatus.UNDEFINED for s in v.component_status.values())
    assert v.principle_reasons["RAM"] in (R.INVALID_ANCHORS,
                                          "ESTIMATOR_NOT_VALIDATED:not_admitted")


def test_default_protocol_never_excludes_through_behavioural_ram():
    d = E.load_protocol(DEFAULT)
    ev = E.ComponentEvidenceV2(principle="RAM", estimate=0.0, null_mean=0.0, se=0.02,
                               se_df=69.0, n_null=70, null_sd=0.0, reference=1.0,
                               reference_scale="excess", se_method="shift_null_sd",
                               estimator="compute_RAM:pe_readout@ram-v3-2026.10")
    pa = E.assess_principle("RAM", [ev], d)
    assert pa.channels["default"] is v1.ComponentStatus.ABSENT
    assert pa.status is v1.ComponentStatus.UNDEFINED
    assert pa.reasons == ("MISSING_CHANNEL:RAM:perturbational",
                          "MISSING_CHANNEL:RAM:endogenous")


# --------------------------------------------------------------------------
# validation and hash coverage
# --------------------------------------------------------------------------
BAD = [
    dict(status_rule=E.StatusRule(alpha=0.04, alpha_absent=0.008)),
    dict(status_rule=E.StatusRule(alpha_absent=0.0125)),
    dict(declared_necessity_set=("RAM", "PDI")),
    dict(directions={"NAS": ["receive"]}),
    dict(directions={"NAS": ["Receive", "return"]}),
    dict(directions={"NAS": ["default", "return"]}),
    dict(directions={"XYZ": ["a", "b"]}),
    dict(rank_gates={"IIM": 0.0}),
    dict(rank_gates={"IIM": 0.6}),
    dict(estimator_versions={"NAS": "nas-v9-2027.01"}),
    dict(estimator_versions={"NAS": "iim-v5-2026.10"}),
    dict(se_methods={"NAS": ["jackknife_pairs_10"]}),
    dict(se_methods={"NAS": []}),
    dict(se_methods={"PDI": ["bogus"]}),
    dict(concordance_route=[{**CELL, "principle": "NAS"}]),
    dict(concordance_route=[{**CELL, "absent": False}]),
    dict(concordance_route=[CELL, dict(CELL)]),
    dict(concordance_route=[{k: v for k, v in CELL.items() if k != "view"}]),
    dict(concordance_route=[{**CELL, "observation_stage": "scalp"}]),
    dict(concordance_route=[{**CELL, "absent": 1}]),
    dict(concordance_route=[{**CELL, "colour": "red"}]),
    dict(concordance_route=[CELL], se_methods={"PDI": ["jackknife_contiguous_10"]}),
    dict(concordance_route=CELL),
    dict(shared_inputs_declaration={"id": "R", "shared_inputs": "partial"}),
    dict(shared_inputs_declaration={"id": "Z", "shared_inputs": "none"}),
    dict(shared_inputs_declaration={"id": "H", "shared_inputs": "partial", "x": 1}),
    dict(shared_inputs_declaration={"id": "R", "shared_inputs": "complete",
                                    "inputs": ["a", "a"]}),
    dict(reference={**REF, "values": {**REF["values"], "NAS:sideways": 1.0}}),
    dict(reference={**REF, "values": {**REF["values"], "NAS:default:x": 1.0}}),
    dict(reference={**REF, "values": {**REF["values"], "NAS:a:b:c": 1.0}}),
    dict(reference={**REF, "values": {**REF["values"], "NAS": 1.0}}),  # pooled
    dict(reference={**REF, "values": {**REF["values"], "NAS:default": 1.0}}),
    dict(reference={**REF, "se": {"NAS": 0.1}}),
    dict(reference={**REF, "values": {**REF["values"], "RAM:default:receive": 1.0}}),
    dict(reference={"kind": "pending", "values": {}}),
    dict(reference={"kind": "somewhere"}),
]


@pytest.mark.parametrize("bad", BAD)
def test_v3_fields_are_validated(bad):
    kw = dict(reference=REF, directions={"NAS": ["receive", "return"]})
    kw.update(bad)
    with pytest.raises(ValueError):
        E.ProtocolV3(**kw)


@pytest.mark.parametrize("nset", [("RAM", "PDI", "NAS", "IIM"), ("RAM", "PDI"),
                                  ("RAM",)])
def test_alpha_absent_is_one_hundredth_for_every_anchored_set(nset):
    """alpha_A = alpha / |N_decl| = 0.01 in every protocol, whatever N_anch;
    alpha / |N_anch| is a sensitivity analysis (``status_rule=``), never the
    rule of a protocol."""
    p = E.ProtocolV3(necessity_set=nset)
    assert p.declared_necessity_set == P5
    assert p.status_rule.alpha_absent == 0.01
    with pytest.raises(ValueError, match="alpha_absent"):
        E.ProtocolV3(necessity_set=nset,
                     status_rule=E.StatusRule(alpha_absent=0.05 / len(nset)))


@pytest.mark.parametrize("declared", [("RAM", "PDI", "NAS", "IIM"), ("RAM",)])
def test_the_declared_set_is_the_five_principles(declared):
    """A smaller declared set would raise alpha_A above 0.01 (0.0125 for
    four, 0.05 for one) and break the transfer of statuses between
    protocols; it is refused."""
    with pytest.raises(ValueError, match="five principles"):
        E.ProtocolV3(declared_necessity_set=declared, necessity_set=declared)
    d = E.load_protocol(TEMPLATE).to_dict()
    d["declared_necessity_set"] = list(declared)
    d["necessity_set"] = list(declared)
    d["status_rule"] = {**d["status_rule"], "alpha_absent": 0.05 / len(declared)}
    with pytest.raises(ValueError, match="five principles"):
        E.ProtocolV3.from_dict(d)


def test_every_v3_field_is_hash_covered():
    base = dict(reference=REF, directions={"NAS": ["receive", "return"]})
    variants = [
        {},
        dict(status_rule=E.StatusRule(null_violation=False)),
        dict(status_rule=E.StatusRule(present_reachability_flag=False)),
        dict(necessity_set=("RAM", "PDI", "NAS")),
        dict(necessity_set=("RAM", "PDI", "NAS", "IIM")),
        dict(directions={"NAS": ["return", "receive"]}),
        dict(rank_gates={"IIM": 0.05}),
        dict(estimator_versions={"NAS": "nas-v3-2026.10"}),
        dict(estimator_versions={"NAS": "nas-v2-2026.09"}),
        dict(se_methods={"NAS": ["jackknife_contiguous_10"]}),
        dict(shared_inputs_declaration={"id": "R", "shared_inputs": "complete"}),
        dict(shared_inputs_declaration={"id": "H", "shared_inputs": "partial"}),
        dict(concordance_route=[CELL]),
        dict(concordance_route=[{**CELL, "provisional": True}]),
        dict(reference={**REF, "values": {**REF["values"], "RAM": 2.0}}),
        dict(reference={"kind": "pending"}),
        dict(name="x"),
    ]
    hashes = [E.ProtocolV3(**{**base, **v}).hash for v in variants]
    assert len(set(hashes)) == len(hashes)


def test_normalisation_of_v3_fields():
    p = E.ProtocolV3(
        reference=REF, directions={"nas": ("receive", "return")},
        se_methods={"NAS": ["jackknife_contiguous_20", "jackknife_contiguous_10"]},
        shared_inputs_declaration={"id": "R", "shared_inputs": "complete",
                                   "inputs": ["stimulus", "context_cue"]},
        concordance_route=[{**CELL, "view": "eeg64"}, CELL])
    d = p.to_dict()
    assert d["directions"] == {"NAS": ["receive", "return"]}
    assert d["se_methods"]["NAS"] == [
        "jackknife_contiguous_10", "jackknife_contiguous_20"]
    assert d["shared_inputs_declaration"]["inputs"] == ["context_cue", "stimulus"]
    assert [c["view"] for c in d["concordance_route"]] == ["eeg64", "source"]
    assert d["concordance_route"][0]["provisional"] is False
    assert d["concordance_route"][0]["battery"] == {}
    with pytest.raises(TypeError):
        p.directions["RAM"] = ("a", "b")  # read-only


# --------------------------------------------------------------------------
# anchors of the protocol
# --------------------------------------------------------------------------
def test_reference_lookup_and_replacement():
    ref = {"kind": "external", "scale": "excess", "se": {"NAS:receive": 0.1},
           "values": {"NAS:receive": 2.0, "NAS:return": 3.0,
                      "NAS:default:return": 4.0, "RAM": 5.0, "RAM:b": 6.0}}
    p = E.ProtocolV3(reference=ref, directions={"NAS": ["receive", "return"]},
                     channels={"RAM": ["a", "b"]})
    assert p.reference_for("NAS", direction="receive") == {
        "reference": 2.0, "reference_se": 0.1, "reference_scale": "excess"}
    assert p.reference_for("NAS", direction="return")["reference"] == 4.0
    assert p.reference_for("RAM", "a")["reference"] == 5.0
    assert p.reference_for("RAM", "b")["reference"] == 6.0
    assert p.reference_for("SRPI") == {}
    ev = E.ComponentEvidenceV2(principle="SRPI", estimate=1.0, null_mean=0.0, se=0.1,
                               reference=1.0, reference_scale="excess")
    assert E.with_protocol_reference(ev, p).reference is None
    ram = E.ComponentEvidenceV2(principle="RAM", estimate=1.0, null_mean=0.0, se=0.1,
                                channel="b", reference=1.0)
    assert E.with_protocol_reference(ram, p).reference == 6.0
    cohort = E.ProtocolV3(reference={"kind": "cohort_high_state", "session": "awake"})
    assert E.with_protocol_reference(ev, cohort) is ev
    assert E.with_protocol_reference(ev, E.ProtocolV3()).reference is None


def test_anchor_validity_rule():
    x = np.array([0.4, 0.6] * 20)  # mean 0.5, sd 0.1013, se 0.0160
    a = T.anchor_validity(x)
    assert a["valid"] and a["n_finite"] == 40
    assert a["lower"] == pytest.approx(0.5 - 1.6849 * 0.1013 / math.sqrt(40), abs=1e-3)
    y = x.copy()
    y[:5] = np.nan  # 35 finite < 36
    assert not T.anchor_validity(y)["valid"]
    y = x.copy()
    y[:4] = np.nan  # 36 finite
    assert T.anchor_validity(y)["valid"]
    assert not T.anchor_validity(x - 0.5)["valid"]  # lower bound < 0
    with pytest.raises(T.TestabilityError):
        T.anchor_validity(x[:39])


def test_anchor_specificity_gate():
    pc = np.array([1.0, 1.2] * 20)  # anchor 1.1
    spec = T.anchor_specificity(pc, pc - 0.6)
    assert spec["specific"] and spec["mean_contrast"] == pytest.approx(0.6)
    assert spec["anchor"] == pytest.approx(1.1)
    assert not T.anchor_specificity(pc, pc - 0.5)["specific"]  # 0.5 < 0.55
    noisy = pc - np.array([0.6, -0.6] * 20)  # mean contrast 0, lower bound < 0
    assert not T.anchor_specificity(pc, noisy)["specific"]
    with pytest.raises(T.TestabilityError):
        T.anchor_specificity(pc, pc[:-1])


def _entries(statuses):
    good_v = {"valid": True}
    out = {}
    for p, st in statuses.items():
        v = {"valid": st != "invalid"}
        s = None if st == "invalid" else {"specific": st == "valid_specific"}
        out[p] = T.anchor_entry({**good_v, **v}, s)
    return out


@pytest.mark.parametrize("statuses, n_anch, fallback", [
    ({p: "valid_specific" for p in P5}, P5, "F0"),
    ({**{p: "valid_specific" for p in P5}, "NAS": "valid_nonspecific",
      "IIM": "invalid"}, ("RAM", "PDI", "SRPI"), "F1"),
    ({**{p: "invalid" for p in P5}, "RAM": "valid_specific"}, ("RAM",), "F2"),
    ({p: "valid_nonspecific" for p in P5}, (), "F3"),
])
def test_anchors_block_sets_n_anch_and_fallback(statuses, n_anch, fallback):
    block = T.anchors_block(_entries(statuses))
    assert block["necessity_set"] == list(n_anch) and block["fallback"] == fallback
    nset = n_anch or P5
    p = E.ProtocolV3(reference=REF, directions={"NAS": ["receive", "return"]},
                     necessity_set=nset, anchors=block)
    assert p.fallback == fallback and p.status_rule.alpha_absent == 0.01
    for q, st in statuses.items():
        assert p.anchor_status(q) == st
    other = P5 if n_anch else ("RAM",)
    if tuple(other) != tuple(nset):
        with pytest.raises(ValueError, match="necessity_set must be"):
            E.ProtocolV3(reference=REF, directions={"NAS": ["receive", "return"]},
                         necessity_set=other, anchors=block)


def test_anchors_block_is_checked():
    block = T.anchors_block(_entries({p: "valid_specific" for p in P5}))
    for change in (
        {"fallback": "F1"}, {"necessity_set": ["RAM"]}, {"schema": "x"},
        {"reference_seeds": [20900, 20919]}, {"reference_seeds": [939, 900]},
        {"rule": {**T.ANCHOR_RULE, "min_finite": 30}},
        {"principles": {p: e for p, e in block["principles"].items() if p != "RAM"}},
    ):
        bad = {**copy.deepcopy(block), **change}
        with pytest.raises(ValueError):
            T.validate_anchors_block(bad, P5, P5)
    bad = copy.deepcopy(block)
    bad["principles"]["RAM"]["specific"] = False
    with pytest.raises(ValueError, match="contradict"):
        T.validate_anchors_block(bad, P5, P5)


def test_directional_anchor_entries_and_the_nonspecific_flag():
    ok = T.anchor_entry({"valid": True}, {"specific": True})
    weak = T.anchor_entry({"valid": True}, {"specific": False})
    assert T.combine_anchor_entries({"receive": ok, "return": ok})["status"] == (
        "valid_specific")
    nas = T.combine_anchor_entries({"receive": ok, "return": weak})
    assert nas["status"] == "valid_nonspecific"
    bad = T.anchor_entry({"valid": False})
    mixed = T.combine_anchor_entries({"receive": ok, "return": bad})
    assert mixed["status"] == "invalid"
    entries = {p: ok for p in P5}
    entries["NAS"] = nas
    block = T.anchors_block(entries)
    p = E.ProtocolV3(reference=REF, directions={"NAS": ["receive", "return"]},
                     necessity_set=("RAM", "PDI", "IIM", "SRPI"), anchors=block)
    ev = [E.ComponentEvidenceV2(principle="NAS", estimate=0.6, null_mean=0.0, se=0.05,
                                se_df=9.0, direction=d) for d in ("receive", "return")]
    pa = E.assess_principle("NAS", ev, p)
    assert pa.status is v1.ComponentStatus.PRESENT
    assert pa.flags == (R.NONSPECIFIC_ANCHOR,)
    assert T.fallback_level(5, 5) == "F0" and T.fallback_level(0, 5) == "F3"
    with pytest.raises(ValueError):
        T.fallback_level(6, 5)


# --------------------------------------------------------------------------
# precision block
# --------------------------------------------------------------------------
def _rows():
    def runs(statuses, se):
        return [{"status": s, "reason": r, "flags": [], "se_c": se, "df_c": 9.0,
                 "seed": 900 + i} for i, (s, r) in enumerate(statuses)]

    a = T.testability_row(family="A-R", principle="NAS", witness="W_NAS_no_workspace",
                          kind="absent", members=2,
                          runs=runs([("ABSENT", None)] * 38
                                    + [("UNDEFINED", "INCONCLUSIVE")] * 2, 0.02))
    b = T.testability_row(family="A-R", principle="IIM", witness="W_IIM_feedforward",
                          kind="absent",
                          runs=runs([("UNDEFINED", "ABSENT_NOT_REACHABLE")] * 40, 0.05))
    c = T.testability_row(family="A-R", principle="IIM", witness="PC_nominal",
                          kind="present", runs=runs([("PRESENT", None)] * 40, 0.1))
    return [a, b, c]


def test_precision_block_holds_the_testability_table():
    block = T.precision_block(_rows())
    keys = [(r["principle"], r["witness"]) for r in block["rows"]]
    assert keys == sorted(keys)
    by = {(r["principle"], r["witness"]): r for r in block["rows"]}
    assert by[("NAS", "W_NAS_no_workspace")]["outcome"] == "decisive"
    iim = by[("IIM", "W_IIM_feedforward")]
    assert iim["outcome"] == "NOT_TESTABLE_BY_DESIGN"
    assert iim["predicted_absent_not_reachable_rate"] == 1.0
    assert iim["pi_analytic"] == 0.0
    assert by[("IIM", "PC_nominal")]["decisive"]
    p = E.ProtocolV3(reference=REF, directions={"NAS": ["receive", "return"]},
                     precision=block)
    assert E.ProtocolV3.from_dict(json.loads(p.to_json())) == p
    q = E.ProtocolV3(reference=REF, directions={"NAS": ["receive", "return"]},
                     precision=T.precision_block(_rows()[:2]))
    assert p.hash != q.hash


def test_precision_block_is_checked():
    rows = _rows()
    with pytest.raises(ValueError, match="duplicate"):
        T.precision_block(rows + [rows[0]])
    with pytest.raises(ValueError, match="threshold"):
        T.precision_block(rows, threshold=0.99)  # decisive rows below 0.99
    bad = dict(rows[1], outcome="decisive")
    with pytest.raises(ValueError):
        T.precision_block([bad])
    bad = dict(rows[0], n=12)
    with pytest.raises(ValueError):
        T.precision_block([bad])
    bad = {k: v for k, v in rows[0].items() if k != "s_A"}
    with pytest.raises(ValueError):
        T.precision_block([bad])
    with pytest.raises(ValueError):
        T.validate_precision_block({"schema": "x", "threshold": 0.9, "min_runs": 40,
                                    "rows": []})


# --------------------------------------------------------------------------
# a generated family protocol: replace, JSON and pickle
# --------------------------------------------------------------------------
def _populated():
    entries = {p: T.anchor_entry({"valid": True}, {"specific": True}) for p in P5}
    entries["IIM"] = T.anchor_entry({"valid": True}, {"specific": False})
    return E.ProtocolV3(
        name="A-R", reference=REF, directions={"NAS": ["receive", "return"]},
        necessity_set=("RAM", "PDI", "NAS", "SRPI"), anchors=T.anchors_block(entries),
        precision=T.precision_block(_rows()), rank_gates={"IIM": 0.05},
        estimator_versions={"NAS": "nas-v3-2026.10"},
        se_methods={"PDI": ["concordant", "jackknife_contiguous_10"]},
        shared_inputs_declaration={"id": "R", "shared_inputs": "complete",
                                   "inputs": ["context_cue", "goal_cue"]},
        concordance_route=[{**CELL, "battery": {"runs": 300, "events": 0}}])


def test_a_populated_protocol_survives_replace_json_and_pickle():
    import pickle

    p = _populated()
    assert p.fallback == "F1"
    assert p.replace().hash == p.hash
    assert p.replace(name="A-H").hash != p.hash
    assert E.ProtocolV3.from_dict(json.loads(p.to_json())).hash == p.hash
    back = pickle.loads(pickle.dumps(p))
    assert back == p and back.hash == p.hash
    assert pickle.loads(pickle.dumps(E.StatusRule())) == E.StatusRule()


def test_assessments_and_verdicts_pickle():
    import pickle

    p = _populated()
    ev = {q: E.ComponentEvidenceV2(principle=q, estimate=0.6, null_mean=0.0, se=0.05,
                                   se_df=9.0, protocol_id=p.protocol_id)
          for q in ("RAM", "PDI", "SRPI", "IIM")}
    ev["IIM"] = dataclasses_replace(ev["IIM"], p_ind=0.01)
    # the protocol declares the admitted SE methods of PDI: its items name one
    ev["PDI"] = dataclasses_replace(ev["PDI"], se_method="jackknife_contiguous_10")
    ev["NAS"] = [E.ComponentEvidenceV2(principle="NAS", estimate=0.6, null_mean=0.0,
                                       se=0.05, se_df=9.0, direction=d,
                                       estimator="compute_NAS:x@nas-v3-2026.10",
                                       protocol_id=p.protocol_id)
                 for d in ("receive", "return")]
    v = E.mpc_verdict(ev, p)
    assert v.verdict is v1.Verdict.MPC_CONSISTENT
    assert v.flags["IIM"] == (R.NONSPECIFIC_ANCHOR,)
    assert v.component_status["IIM"] is v1.ComponentStatus.PRESENT
    assert pickle.loads(pickle.dumps(v)).to_dict() == v.to_dict()
    pa = E.assess_principle("NAS", ev["NAS"], p)
    assert pickle.loads(pickle.dumps(pa)).record_fields() == pa.record_fields()
