"""
1.1.0 post-freeze fix: ``scripts/compute_empirical_reference.py``, the
external empirical reference anchor (per-principle rho and SE) from the
high-state runs of a declared held-out subset of participants, on tiny
synthetic layouts (hub networks: NAS capacity is above its null in the awake
runs and at its null in the deep runs; behavioural events: RAM's behavioural
channel responds in the awake runs only).

Under the default protocol ``mpc_default_v1.json`` RAM declares three
channels of which only ``default`` has an estimator: ``--data-dir`` mode
computes the reference per channel (``RAM:default``) and lists the
unimplemented channels as ``NOT_IMPLEMENTED``; ``--step2-table`` mode cannot
identify the channel of the RAM columns and gives RAM no reference
(``CHANNEL_NOT_IDENTIFIABLE``).
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

import scripts.build_example_protocols as bep
import scripts.compute_empirical_reference as cer
from impact_pipeline import evidence as E
from impact_pipeline import mpc_metrics as mm
from impact_pipeline import synergy_ci as sc
from impact_pipeline.run_synergy_ci import load_onsets
from test_default_empirical_run import (
    BEHAVIOURAL_RAM,
    TR as EVENT_TR,
    _behavioural_layout,
    install_behavioural_ram,
)
from test_nas_capacity import _hub_network

SUBJECTS = ("s1", "s2", "s3", "s4", "s5")
REFERENCE = ["s1", "s2", "s3"]
EVALUATION = ["s4", "s5"]


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    prep = tmp_path_factory.mktemp("layout") / "prep"
    for i, s in enumerate(SUBJECTS):
        for ses, kind, seed in (("awake", "bidir", i), ("deep", "none", 100 + i)):
            x, _ = _hub_network(kind, seed, n_time=1200)
            d = prep / s / ses / "audio"
            d.mkdir(parents=True)
            np.save(d / f"{s}_run-1_toy_ts.npy", x.T)
    return prep


@pytest.fixture(scope="module")
def protocol():
    # the default protocol (v1) with a declared hub (the example builder's
    # derivation)
    return bep.derived_protocol(cer.DEFAULT_PROTOCOL, [0, 1, 2, 3], "test-v1-hub")


def test_default_protocol_is_v1():
    assert cer.DEFAULT_PROTOCOL == cer.REPO_ROOT / "protocols" / "mpc_default_v1.json"
    assert E.Protocol.from_json(cer.DEFAULT_PROTOCOL).hash == (
        "383eb310cf4479d3260bc5f43f8972bd0b12104a221579fea17c40e410357267")
    assert cer.build_parser().parse_args(
        ["--step2-table", "x", "--reference-subjects", "a", "--out", "o"]
    ).protocol == str(cer.DEFAULT_PROTOCOL)


@pytest.fixture(scope="module")
def table(layout, protocol):
    return cer.compute_high_state_table(
        layout, "toy", protocol, REFERENCE, session="awake", condition="audio",
        tr=1.0, params=cer._dataset_params(), metrics=("NAS",), null_surrogates=19)


def test_computes_only_the_reference_participants_high_state_runs(table, protocol):
    assert set(table["subject"]) == set(REFERENCE)
    assert set(table["session"]) == {"awake"}
    assert (table["MPC_protocol_hash"] == protocol.hash).all()


def test_reference_is_the_pipeline_cohort_reference_of_the_held_out_subset(
        table, protocol):
    res = cer.build_reference(table, protocol, REFERENCE, session="awake",
                              evaluation_subjects=EVALUATION, dataset_id="toy")
    assert res["schema"] == "impact-empirical-reference/1"
    ref = res["reference"]
    assert ref["kind"] == "external" and ref["scale"] == "excess"
    # identical to the pipeline's own cohort reference on the same runs
    pipe = table.attrs["mpc_evidence"]["references"]["NAS:default"]
    assert ref["values"]["NAS"] == pytest.approx(pipe["reference"], rel=1e-12)
    assert ref["se"]["NAS"] == pytest.approx(pipe["reference_se"], rel=1e-12)
    # and by hand: participant means first, SE = SD / sqrt(n)
    ex = (table["NAS_estimate"] - table["NAS_null_mean"]).groupby(
        table["subject"]).mean().to_numpy()
    per = res["summary"]["per_principle"]["NAS"]
    assert per["n"] == 3 and per["n_runs"] == 3
    assert per["mean"] == pytest.approx(ex.mean())
    assert per["se"] == pytest.approx(ex.std(ddof=1) / math.sqrt(3))
    assert per["lower_bound"] > 0 and "anchor" not in per
    # principles without evidence get no anchor (never a made-up one)
    assert set(ref["values"]) == {"NAS"}
    assert res["summary"]["no_anchor"] == ["RAM", "PDI", "IIM", "SRPI"]
    # not computed (--metrics NAS): the same reason code as in --data-dir mode
    assert res["summary"]["per_principle"]["IIM"]["reason"] == "NOT_COMPUTED"
    assert res["summary"]["per_principle"]["IIM"]["anchor"].startswith(
        "none (NOT_COMPUTED:")
    assert res["summary"]["channels_without_reference"]["SRPI"] == "NOT_COMPUTED"
    # a step-2 table: RAM (three declared channels under v1) cannot be
    # attributed to a channel
    assert res["summary"]["evidence"] == "step2_columns"
    assert res["summary"]["channels_without_reference"]["RAM:default"] == (
        "CHANNEL_NOT_IDENTIFIABLE")
    # the external reference is a valid Protocol reference
    derived = cer.protocol_with_reference(protocol, res)
    assert derived.reference["values"] == {"NAS": ref["values"]["NAS"]}
    assert derived.name == "test-v1-hub+external-reference"
    prov = res["provenance"]
    assert prov["protocol_hash"] == protocol.hash
    assert prov["reference_subjects_used"] == REFERENCE
    assert prov["reference_subjects_missing"] == []
    assert prov["evaluation_subjects_declared"] == EVALUATION
    assert prov["reference_subjects_sha256"] == cer._sha256_text(REFERENCE)
    assert prov["estimators_seen"]["NAS"] == {
        "estimator": [f"compute_NAS:capacity@{mm.ESTIMATOR_VERSIONS['NAS']}"],
        "estimator_version": [mm.ESTIMATOR_VERSIONS["NAS"]]}
    assert prov["null_surrogates"] == [19]
    assert prov["code_version"]["package_version"]


def test_declaration_is_required_and_must_be_held_out(table, protocol, tmp_path):
    with pytest.raises(ValueError, match="declare the held-out"):
        cer.build_reference(table, protocol, [])
    with pytest.raises(ValueError, match="held out of the evaluation"):
        cer.build_reference(table, protocol, REFERENCE, evaluation_subjects=["s2"])
    other = E.Protocol.from_json(cer.DEFAULT_PROTOCOL)
    with pytest.raises(ValueError, match="computed under protocol hashes"):
        cer.build_reference(table, other, REFERENCE)
    with pytest.raises(ValueError, match="no 'deep' rows"):
        cer.build_reference(table, protocol, REFERENCE, session="deep")
    f = tmp_path / "heldout.txt"
    f.write_text("# reference subset\nsub-s1\ns2  # comment\n\ns3\n")
    assert cer.parse_subjects(f"@{f}") == REFERENCE
    assert cer.parse_subjects("sub-s1, s2") == ["s1", "s2"]
    with pytest.raises(ValueError, match="duplicate"):
        cer.parse_subjects("s1,sub-s1")


def test_anchor_rule_and_participant_means():
    rows = []
    for s, runs in {"a": [1.0, 1.2], "b": [0.9], "c": [1.1]}.items():
        rows += [{"subject": s, "session": "awake", "NAS_estimate": v + 0.5,
                  "NAS_null_mean": 0.5, "IIM_estimate": w, "IIM_null_mean": 0.0}
                 for v, w in zip(runs, (0.1, -0.1))]
    rows.append({"subject": "c", "session": "awake", "NAS_estimate": np.nan,
                 "NAS_null_mean": 0.5, "IIM_estimate": 0.05, "IIM_null_mean": 0.0})
    df = pd.DataFrame(rows)
    values, ses, per = cer.reference_from_table(df, ["a", "b", "c"], "awake",
                                                principles=("NAS", "IIM"))
    subj = np.array([1.1, 0.9, 1.1])  # participant means first
    assert values["NAS"] == pytest.approx(subj.mean())
    assert ses["NAS"] == pytest.approx(subj.std(ddof=1) / math.sqrt(3))
    assert per["NAS"]["n_runs"] == 4
    # IIM excess not credibly above its null: no anchor
    assert "IIM" not in values
    assert per["IIM"]["anchor"].startswith("none (high-state excess not credibly")
    # a principle with several declared channels: the step-2 columns hold the
    # deciding channel only, so no channel gets a reference from a table
    values, _, per = cer.reference_from_table(
        df, ["a", "b", "c"], "awake", principles=("NAS",),
        channels={"NAS": ("default", "other")})
    assert values == {}
    assert per["NAS"]["anchor"] == "none (no declared channel has an anchor)"
    chans = per["NAS"]["channels"]
    assert chans["default"]["reason"] == "CHANNEL_NOT_IDENTIFIABLE"
    assert chans["default"]["anchor"].startswith("none (CHANNEL_NOT_IDENTIFIABLE:")
    assert chans["other"]["reason"] == "NOT_IMPLEMENTED"
    assert cer.channels_without_reference(per) == {
        "NAS:default": "CHANNEL_NOT_IDENTIFIABLE", "NAS:other": "NOT_IMPLEMENTED"}
    with pytest.raises(ValueError, match="min_n"):
        cer.reference_from_table(df, ["a"], "awake", min_n=1)


def test_reference_keys_and_implemented_channels():
    assert cer.reference_key("RAM", "default", 1) == "RAM"
    assert cer.reference_key("RAM", "default", 3) == "RAM:default"
    assert cer.channel_has_estimator("RAM", "default")
    assert all(cer.channel_has_estimator("RAM", c) for c in mm.RAM_IMPLEMENTED_CHANNELS)
    assert not cer.channel_has_estimator("RAM", "perturbational")
    assert not cer.channel_has_estimator("RAM", "endogenous")
    assert cer.channel_has_estimator("NAS", "default")
    assert not cer.channel_has_estimator("NAS", "other")


def test_channel_evidence_aggregation_per_channel():
    """Per-channel records: RAM:default from the implemented channel, the
    unimplemented channels without reference (NOT_IMPLEMENTED), single-channel
    principles keyed P."""
    recs = []
    for s, (ram, nas) in {"a": (1.0, 0.5), "b": (1.2, 0.7), "c": (0.8, 0.6)}.items():
        recs.append({"subject": s, "session": "awake", "principle": "RAM",
                     "channel": "default", "estimate": ram + 0.1, "null_mean": 0.1,
                     "reason": None})
        recs += [{"subject": s, "session": "awake", "principle": "RAM", "channel": ch,
                  "estimate": float("nan"), "null_mean": float("nan"),
                  "reason": "NOT_IMPLEMENTED"}
                 for ch in ("perturbational", "endogenous")]
        recs.append({"subject": s, "session": "awake", "principle": "NAS",
                     "channel": "default", "estimate": nas, "null_mean": 0.0,
                     "reason": None})
        # other sessions and participants never count
        recs.append({"subject": s, "session": "deep", "principle": "RAM",
                     "channel": "default", "estimate": 99.0, "null_mean": 0.0,
                     "reason": None})
    recs.append({"subject": "z", "session": "awake", "principle": "RAM",
                 "channel": "default", "estimate": 99.0, "null_mean": 0.0,
                 "reason": None})
    chans = {"RAM": ("default", "perturbational", "endogenous"), "NAS": ("default",),
             "PDI": ("default",)}
    values, ses, per = cer.reference_from_channel_evidence(
        recs, ["a", "b", "c"], "awake", principles=("RAM", "NAS", "PDI"),
        channels=chans)
    assert set(values) == {"RAM:default", "NAS"}
    assert values["RAM:default"] == pytest.approx(1.0)
    assert ses["RAM:default"] == pytest.approx(
        np.std([1.0, 1.2, 0.8], ddof=1) / math.sqrt(3))
    assert values["NAS"] == pytest.approx(0.6)
    ram = per["RAM"]
    assert ram["reference_keys"] == ["RAM:default"]
    assert ram["channels"]["default"]["n"] == 3
    assert ram["channels"]["default"]["reference_key"] == "RAM:default"
    for ch in ("perturbational", "endogenous"):
        assert ram["channels"][ch]["reason"] == "NOT_IMPLEMENTED"
        assert "reference_key" not in ram["channels"][ch]
    assert per["NAS"]["reference_key"] == "NAS" and per["NAS"]["channel"] == "default"
    assert per["PDI"]["reason"] == "NOT_COMPUTED"
    assert cer.channels_without_reference(per) == {
        "RAM:perturbational": "NOT_IMPLEMENTED", "RAM:endogenous": "NOT_IMPLEMENTED",
        "PDI": "NOT_COMPUTED"}
    # the reference is a valid Protocol reference (keys P and P:channel)
    proto = E.Protocol.from_json(cer.DEFAULT_PROTOCOL).replace(reference={
        "kind": "external", "scale": "excess", "values": values, "se": ses})
    assert set(proto.reference["values"]) == {"RAM:default", "NAS"}
    # RAM not computed at all: the unimplemented channels are still NOT_IMPLEMENTED
    _, _, per = cer.reference_from_channel_evidence(
        [r for r in recs if r["principle"] != "RAM"], ["a", "b", "c"], "awake",
        principles=("RAM",), channels=chans)
    assert {ch: st["reason"] for ch, st in per["RAM"]["channels"].items()} == {
        "default": "NOT_COMPUTED", "perturbational": "NOT_IMPLEMENTED",
        "endogenous": "NOT_IMPLEMENTED"}


def test_cli_from_a_step2_table(table, protocol, tmp_path):
    proto_path = tmp_path / "protocol.json"
    protocol.to_json(proto_path)
    # a step-2 table has one row per theta and run, other participants and
    # other sessions: only the reference participants' high-state runs count
    distractor = table.copy()
    distractor["subject"] = "s4"
    distractor["NAS_estimate"] = distractor["NAS_estimate"] + 100.0
    deep = table.copy()
    deep["session"] = "deep"
    deep["NAS_estimate"] = deep["NAS_estimate"] - 100.0
    theta2 = table.assign(theta=0.9)
    csv = tmp_path / "step2_df.csv"
    pd.concat([table, theta2, distractor, deep]).to_csv(csv, index=False)
    out, derived = tmp_path / "ref.json", tmp_path / "derived.json"
    rc = cer.main(["--step2-table", str(csv), "--protocol", str(proto_path),
                   "--reference-subjects", ",".join(REFERENCE),
                   "--evaluation-subjects", ",".join(EVALUATION),
                   "--dataset-id", "toy", "--out", str(out),
                   "--write-protocol", str(derived)])
    assert rc == 0
    res = json.loads(out.read_text())
    direct = cer.build_reference(table, protocol, REFERENCE)
    assert res["reference"]["values"] == pytest.approx(direct["reference"]["values"])
    assert res["reference"]["se"] == pytest.approx(direct["reference"]["se"])
    assert res["provenance"]["evidence_source"]["kind"] == "step2_table"
    assert res["provenance"]["evidence_source"]["sha256"] == cer._sha256_file(csv)
    assert res["provenance"]["n_runs"] == 3
    d = E.Protocol.from_json(derived)
    assert res["provenance"]["derived_protocol"]["hash"] == d.hash
    assert d.reference["kind"] == "external"
    assert d.replace(reference=protocol.reference,
                     name=protocol.name).hash == protocol.hash
    # no valid anchor: exit 1, JSON with the reasons, no protocol written
    rc = cer.main(["--step2-table", str(csv), "--protocol", str(proto_path),
                   "--reference-subjects", ",".join(REFERENCE), "--min-n", "4",
                   "--out", str(tmp_path / "none.json"),
                   "--write-protocol", str(tmp_path / "none_protocol.json")])
    assert rc == 1
    assert json.loads((tmp_path / "none.json").read_text())["reference"] is None
    assert not (tmp_path / "none_protocol.json").exists()


def test_derived_protocol_anchors_the_evaluation_participants(
        layout, table, protocol):
    res = cer.build_reference(table, protocol, REFERENCE,
                              evaluation_subjects=EVALUATION)
    derived = cer.protocol_with_reference(protocol, res)
    ev = sc.compute_synergy_ci(
        str(layout), "toy", [0.5], sessions=("awake", "deep"), tr=1.0,
        mpc_metrics=("NAS",), compute_ci=False, subjects=EVALUATION,
        protocol=derived, bootstrap_se=10)
    rho = res["reference"]["values"]["NAS"]
    # the per-channel records are opt-in: the pipeline's outputs are unchanged
    assert "channel_evidence" not in ev.attrs["mpc_evidence"]
    assert set(ev["subject"]) == set(EVALUATION)
    assert np.allclose(ev["NAS_reference"], rho)
    assert np.allclose(ev["NAS_reference_se"], res["reference"]["se"]["NAS"])
    assert (ev["MPC_protocol_hash"] == derived.hash).all()
    np.testing.assert_allclose(
        ev["NAS_c"], (ev["NAS_estimate"] - ev["NAS_null_mean"]) / rho)
    awake = ev[ev["session"] == "awake"]
    deep = ev[ev["session"] == "deep"]
    assert (awake["NAS_status"] == "PRESENT").all()
    # no hub coupling: never PRESENT (ABSENT, or INCONCLUSIVE at this length)
    assert not (deep["NAS_status"] == "PRESENT").any()
    assert (deep["NAS_status"] == "ABSENT").any()


def test_script_runs_standalone_with_dataset_parameters(layout, protocol, tmp_path):
    """Run as ``python scripts/compute_empirical_reference.py`` (sys.path[0]
    is scripts/, not the repository root) from another directory, with the
    estimator parameters of a known dataset (``run_pipeline.DATASET_CONFIGS``)."""
    import os
    import subprocess
    import sys

    proto_path = tmp_path / "protocol.json"
    protocol.to_json(proto_path)
    out = tmp_path / "ref.json"
    script = cer.REPO_ROOT / "scripts" / "compute_empirical_reference.py"
    proc = subprocess.run(
        [sys.executable, str(script),
         "--data-dir", str(layout), "--atlas", "toy", "--tr", "1.0",
         "--protocol", str(proto_path), "--reference-subjects", "s1,s2,s3",
         "--metrics", "NAS", "--null-surrogates", "5", "--dataset-id", "ds003171",
         "--out", str(out)],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=600,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    res = json.loads(out.read_text())
    assert res["provenance"]["dataset_id"] == "ds003171"
    assert res["provenance"]["evidence_source"]["kind"] == "computed"
    assert res["provenance"]["reference_subjects_used"] == REFERENCE


def test_unknown_dataset_never_falls_back_to_default_parameters(tmp_path):
    with pytest.raises(ValueError, match="unknown --dataset-id"):
        cer._dataset_params("no_such_dataset")
    pj = tmp_path / "params.json"
    pj.write_text(json.dumps({"modality": "fmri"}))
    assert cer._dataset_params("no_such_dataset", str(pj))["modality"] == "fmri"
    assert cer._dataset_params("ds003171")["nas_params"] is not None


def test_cli_data_dir_mode(layout, protocol, tmp_path):
    proto_path = tmp_path / "protocol.json"
    protocol.to_json(proto_path)
    out = tmp_path / "ref.json"
    rc = cer.main(["--data-dir", str(layout), "--atlas", "toy", "--tr", "1.0",
                   "--protocol", str(proto_path), "--reference-subjects",
                   "s1,s2,s3,s9", "--metrics", "NAS", "--null-surrogates", "9",
                   "--out", str(out)])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["provenance"]["reference_subjects_missing"] == ["s9"]
    assert res["provenance"]["evidence_source"]["kind"] == "computed"
    assert res["provenance"]["null_surrogates"] == [9]
    assert set(res["reference"]["values"]) == {"NAS"}


# --------------------------------------------------------------------------
# RAM under the default protocol v1: three declared channels, one implemented
# --------------------------------------------------------------------------
RAM_SUBJECTS = ("r1", "r2", "r3", "r4", "r5")
RAM_REFERENCE = ["r1", "r2", "r3"]
RAM_EVALUATION = ["r4", "r5"]


@pytest.fixture
def ram_layout(tmp_path, monkeypatch):
    """Awake runs respond behaviourally, deep runs are behaviourally null
    (tests/test_default_empirical_run.py); the behavioural RAM channel is a
    known-answer estimator, the typed channels the real compute_RAM."""
    bids, out = _behavioural_layout(tmp_path, subjects=RAM_SUBJECTS, atlas="toy",
                                    seed=11)
    calls = install_behavioural_ram(monkeypatch)
    return bids, out / "preprocessed", calls


def _ram_table(prep, bids, proto, subjects=RAM_REFERENCE):
    return cer.compute_high_state_table(
        prep, "toy", proto, subjects, session="awake", condition="audio",
        tr=EVENT_TR, bids_root=bids, params=cer._dataset_params(),
        metrics=("RAM",), null_surrogates=19)


def test_data_dir_mode_gives_the_implemented_ram_channel_its_reference(ram_layout):
    bids, prep, calls = ram_layout
    v1 = E.Protocol.from_json(cer.DEFAULT_PROTOCOL)
    table = _ram_table(prep, bids, v1)
    assert {"perturbational", "endogenous", None} <= set(calls)
    ev = cer.channel_evidence_of(table)
    assert ev is not None
    assert {(r["principle"], r["channel"]) for r in ev} == {
        ("RAM", "default"), ("RAM", "perturbational"), ("RAM", "endogenous")}
    assert all(r["reason"] == "NOT_IMPLEMENTED" for r in ev
               if r["channel"] != "default")
    res = cer.build_reference(table, v1, RAM_REFERENCE, session="awake",
                              evaluation_subjects=RAM_EVALUATION,
                              channel_evidence=ev)
    summary = res["summary"]
    assert summary["evidence"] == "per_channel_evidence"
    assert summary["declared_channels"]["RAM"] == [
        "default", "perturbational", "endogenous"]
    # the key the Protocol expects for a principle with several channels
    assert set(res["reference"]["values"]) == {"RAM:default"}
    assert summary["reference_keys"] == ["RAM:default"]
    assert summary["channels_without_reference"]["RAM:perturbational"] == (
        "NOT_IMPLEMENTED")
    assert summary["channels_without_reference"]["RAM:endogenous"] == (
        "NOT_IMPLEMENTED")
    assert "RAM" not in summary["no_anchor"]
    # identical to the pipeline's cohort reference of the same channel and runs
    pipe = table.attrs["mpc_evidence"]["references"]["RAM:default"]
    assert res["reference"]["values"]["RAM:default"] == pytest.approx(
        pipe["reference"], rel=1e-12)
    assert res["reference"]["se"]["RAM:default"] == pytest.approx(
        pipe["reference_se"], rel=1e-12)
    st = summary["per_principle"]["RAM"]["channels"]["default"]
    assert st["n"] == 3 and st["lower_bound"] > 0
    assert "RAM:default n=3" in res["reference"]["source"]
    assert "RAM:perturbational NOT_IMPLEMENTED" in res["reference"]["source"]


def test_step2_table_mode_never_guesses_the_ram_channel(ram_layout, tmp_path):
    """The same runs from a step-2 table: the RAM columns describe the
    deciding channel, so RAM gets no reference (CHANNEL_NOT_IDENTIFIABLE for
    the implemented channel, NOT_IMPLEMENTED for the others)."""
    bids, prep, _ = ram_layout
    v1 = E.Protocol.from_json(cer.DEFAULT_PROTOCOL)
    table = _ram_table(prep, bids, v1)
    csv = tmp_path / "step2_df.csv"
    table.to_csv(csv, index=False)
    out = tmp_path / "ref.json"
    proto_path = tmp_path / "v1.json"
    v1.to_json(proto_path)
    rc = cer.main(["--step2-table", str(csv), "--protocol", str(proto_path),
                   "--reference-subjects", ",".join(RAM_REFERENCE),
                   "--out", str(out)])
    assert rc == 1  # no principle has an anchor
    res = json.loads(out.read_text())
    assert res["reference"] is None
    summary = res["summary"]
    assert summary["evidence"] == "step2_columns"
    assert summary["channels_without_reference"]["RAM:default"] == (
        "CHANNEL_NOT_IDENTIFIABLE")
    for ch in ("perturbational", "endogenous"):
        assert summary["channels_without_reference"][f"RAM:{ch}"] == "NOT_IMPLEMENTED"
    text = summary["per_principle"]["RAM"]["channels"]["default"]["anchor"]
    assert "--data-dir" in text
    # the behavioural-RAM protocol declares one channel: a table suffices
    bram = E.Protocol.from_json(BEHAVIOURAL_RAM)
    table = _ram_table(prep, bids, bram)
    res = cer.build_reference(table, bram, RAM_REFERENCE)
    assert set(res["reference"]["values"]) == {"RAM"}
    assert "RAM" not in res["summary"]["channels_without_reference"]


def test_cli_data_dir_mode_and_the_derived_protocol_under_v1(ram_layout, tmp_path):
    """--data-dir under the default protocol writes RAM:default; the derived
    protocol anchors the evaluation participants' behavioural channel, and
    their behaviourally null runs are still never RAM ABSENT."""
    bids, prep, _ = ram_layout
    out, derived_path = tmp_path / "ref.json", tmp_path / "derived.json"
    rc = cer.main(["--data-dir", str(prep), "--atlas", "toy", "--tr", str(EVENT_TR),
                   "--bids-root", str(bids), "--metrics", "RAM",
                   "--reference-subjects", ",".join(RAM_REFERENCE),
                   "--evaluation-subjects", ",".join(RAM_EVALUATION),
                   "--null-surrogates", "19", "--out", str(out),
                   "--write-protocol", str(derived_path)])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["provenance"]["protocol_hash"] == E.Protocol.from_json(
        cer.DEFAULT_PROTOCOL).hash
    assert set(res["reference"]["values"]) == {"RAM:default"}
    rho = res["reference"]["values"]["RAM:default"]
    derived = E.Protocol.from_json(derived_path)
    assert derived.channels_for("RAM") == ("default", "perturbational", "endogenous")
    ev = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake", "deep"), tr=EVENT_TR,
        stimulus_onsets={s: {ses: load_onsets(bids, s, ses, condition="audio")
                             for ses in ("awake", "deep")} for s in RAM_EVALUATION},
        ram_params=cer._dataset_params()["ram_params"], mpc_metrics=("RAM",),
        compute_ci=False, subjects=RAM_EVALUATION, protocol=derived,
        null_surrogates=19, bootstrap_se=20)
    awake = ev[ev["session"] == "awake"]
    deep = ev[ev["session"] == "deep"]
    assert np.allclose(awake["RAM_reference"], rho)
    assert (awake["RAM_status"] == "PRESENT").all()
    assert deep["RAM_channels"].str.startswith("default:ABSENT").all()
    assert (deep["RAM_status"] == "UNDEFINED").all()
    assert not (ev["MPC_verdict"] == "EXCLUDED").any()
