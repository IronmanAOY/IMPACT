"""Reference anchor of the MPC-Bench construct scale (positive control on
development seeds, recorded in the bench protocol) and its use by the
bench verdicts."""

import json

import numpy as np
import pytest

from impact_pipeline import evidence as ev
from impact_pipeline.bench import export, reference as ref
from impact_pipeline.bench.generators import PRINCIPLES
from impact_pipeline.bench.run_bench import RESULTS_JSONL

FAMILIES = {
    "RAM": "onset_jitter",
    "PDI": "circular_shift",
    "NAS": "block_circular_shift",
    "IIM": "circular_shift",
    "SRPI": "yoked_label_permutation",
}


def _pc_record(seed, excess, null_mean=0.2, witness="PC_nominal", split="development",
               modes=None, undefined=()):
    comps = {}
    for p in PRINCIPLES:
        comps[p] = {
            "estimate": null_mean + excess[p],
            "null_mean": null_mean,
            "null_sd": 0.05,
            "n_null": 19,
            "null_family": FAMILIES[p],
            "se": 0.05,
            "se_n": 5,
            "defined": p not in undefined,
            "statistic": "raw",
            "bearer_id": "system",
        }
    return {
        "task_id": f"witness-A-{witness}-s{seed:05d}",
        "status": "ok",
        "witness_id": witness,
        "family": "A",
        "seed": seed,
        "split": split,
        "null_surrogates": 19,
        "se_groups": 5,
        "bench_version": "2.0.0",
        "generator_version": "g",
        "components": comps,
        "estimator_modes": modes or dict(export.OPTIONAL_MODES),
        "provenance": {"code_version": {"git_sha": "abc"}},
    }


def _records(n=6, undefined_pdi=False):
    rng = np.random.default_rng(0)
    recs = []
    for i in range(n):
        exc = {p: 1.0 + 0.1 * rng.standard_normal() for p in PRINCIPLES}
        und = ("PDI",) if undefined_pdi and i > 0 else ()
        recs.append(_pc_record(900 + i, exc, undefined=und))
    return recs


def _template():
    return ev.Protocol(null_families=FAMILIES, estimators={
        "RAM": {"update": "prediction_error"}, "PDI": {"mode": "repertoire"},
        "NAS": {"mode": "capacity"}, "SRPI": {"mode": "agency"}},
        name="template")


def test_reference_tasks_are_development_positive_controls():
    tasks = ref.reference_tasks([900, 901])
    assert [t.witness_id for t in tasks] == ["PC_nominal", "PC_nominal"]
    assert {t.family for t in tasks} == {"A"}
    with pytest.raises(ValueError, match="held out"):
        ref.reference_tasks([900], family="C")
    with pytest.raises(ValueError, match="development seeds"):
        ref.reference_tasks([10000])


def test_reference_from_records_known_answers():
    recs = _records()
    values, summary = ref.reference_from_records(recs)
    for p in PRINCIPLES:
        exc = np.array([r["components"][p]["estimate"]
                        - r["components"][p]["null_mean"] for r in recs])
        assert values["values"][p] == pytest.approx(exc.mean())
        assert values["se"][p] == pytest.approx(exc.std(ddof=1) / np.sqrt(exc.size))
    assert values["kind"] == "external" and values["scale"] == "excess"
    assert "PC_nominal" in values["source"] and "900-905" in values["source"]
    assert summary["per_principle"]["NAS"]["n"] == 6
    est, _ = ref.reference_from_records(recs, scale="estimate")
    assert est["values"]["IIM"] == pytest.approx(values["values"]["IIM"] + 0.2)
    # a principle with fewer than min_n finite values gets no anchor
    part, summ = ref.reference_from_records(_records(undefined_pdi=True))
    assert "PDI" not in part["values"] and summ["per_principle"]["PDI"]["n"] == 1
    # only development positive controls can anchor
    with pytest.raises(ValueError, match="no successful"):
        ref.reference_from_records([_pc_record(900, {p: 1 for p in PRINCIPLES},
                                               witness="W_x")])
    with pytest.raises(ValueError, match="held-out"):
        ref.reference_from_records([_pc_record(900, {p: 1 for p in PRINCIPLES},
                                               split="confirmatory")])
    with pytest.raises(ValueError, match="scale"):
        ref.reference_from_records(recs, scale="raw")


def test_modes_of_the_records_must_match_the_protocol():
    recs = _records()
    ref.check_modes(recs, _template().to_dict()["estimators"])
    bad = [_pc_record(900, {p: 1.0 for p in PRINCIPLES},
                      modes={**export.OPTIONAL_MODES, "PDI": {"mode": "legacy"}})]
    with pytest.raises(ValueError, match="PDI mode='legacy'"):
        ref.check_modes(bad, _template().to_dict()["estimators"])


def test_cli_writes_the_protocol_and_the_bench_verdict_uses_it(tmp_path):
    recs = _records()
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    with open(res_dir / RESULTS_JSONL, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    tpl = tmp_path / "template.json"
    _template().to_json(tpl)
    out = tmp_path / "protocols" / "bench.json"
    assert ref.main(["--results", str(res_dir), "--template", str(tpl),
                     "--out", str(out), "--name", "bench-provisional"]) == 0
    proto = ev.Protocol.from_json(out)
    assert proto.name == "bench-provisional"
    assert proto.reference["kind"] == "external"
    assert proto.estimators["PDI"]["mode"] == "repertoire"
    summary = json.loads((out.parent / "bench_reference_summary.json").read_text())
    assert summary["protocol_hash"] == proto.hash
    # a system exactly at the positive-control mean excess has c = 1
    mean = proto.reference["values"]
    comps = {p: dict(recs[0]["components"][p]) for p in PRINCIPLES}
    for p in PRINCIPLES:
        comps[p]["estimate"] = comps[p]["null_mean"] + mean[p]
    out_v = export.evidence_verdict(
        {"components": comps, "estimator_modes": dict(export.OPTIONAL_MODES)},
        {}, protocol=str(out))
    for p in PRINCIPLES:
        assert out_v["c"][p] == pytest.approx(1.0)
    assert out_v["verdict"] == "MPC_CONSISTENT"
    assert out_v["protocol_hash"] == proto.hash
