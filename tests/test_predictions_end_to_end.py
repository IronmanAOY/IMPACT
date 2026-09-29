"""
The hypothesis registry against what the pipeline records: the estimator
names of predictions/registry.yaml are the ids the evidence layer writes
(``compute_<P>:<mode>@<version>``, ComponentEvidence.estimator) under the
registry's protocol, and
scripts/run_predictions.py accepts a step-2 table in the pipeline's column
format (``<P>_estimator`` = the recorded id, ``<P>_estimator_version``,
``MPC_protocol_hash``) when these versions are registered, and refuses it
when they are not.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.run_predictions as rp
from impact_pipeline import evidence as E
from impact_pipeline import mpc_metrics as mm
from impact_pipeline import synergy_ci as sc

REPO = Path(__file__).resolve().parents[1]
VERSIONS = mm.ESTIMATOR_VERSIONS


def _registry_protocol(reg):
    entry = reg["protocols"][0]
    return entry, E.Protocol.from_json(REPO / entry["file"])


def _recorded_id(proto, p, version=None):
    """The id the pipeline records for principle ``p`` under ``proto`` (the
    primary mode, as synergy_ci selects it)."""
    mode = sc._mode_of(p, proto.estimator_options(p))
    return E.estimator_id(p, mode, VERSIONS[p] if version is None else version)


def _frozen(reg, proto, necessity_set=None, versions=VERSIONS):
    """The shipped registry frozen for a test: ``proto``'s evidence.Protocol
    hash and the given estimator versions registered."""
    r = copy.deepcopy(reg)
    r.update(status="frozen", freeze_tag="paper2-freeze-e2e",
             registry_version=r["registry_version"].split("-")[0])
    entry = r["protocols"][0]
    entry.update(hash=proto.hash, hash_algorithm="evidence.Protocol")
    if necessity_set is not None:
        entry["necessity_set"] = list(necessity_set)
    for e in r["estimators"]:
        e.update(version=versions[e["principle"]], registration="registered")
    for h in r["hypotheses"]:
        if h.get("minimum_n_status") == "provisional":
            h["minimum_n_status"] = "frozen"
    assert rp.validate_registry(r) == []
    return r


def _step2_table(proto, version_form, n_pos=200, n_neg=100, seed=0):
    """Episode table with the identification columns in the pipeline's
    format: ``<P>_estimator`` as synergy_ci writes it (the recorded id) and
    ``<P>_estimator_version`` as the version of that id (``version``) or as
    the id itself (``id``)."""
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    pos = np.r_[np.ones(n_pos, bool), np.zeros(n_neg, bool)]
    df = pd.DataFrame({
        "dataset": "dream_mediated",
        "episode_id": [f"sub-{i // 4:03d}_awk-{i % 4}" for i in range(n)],
        "report_positive": pos,
        "MPC_protocol_hash": proto.hash,
        "MPC_necessity_set": ",".join(proto.necessity_set),
    })
    for p in E.PRINCIPLES:
        st = np.full(n, "PRESENT", dtype=object)
        st[~pos & (rng.random(n) < 0.5)] = "ABSENT"
        df[f"{p}_status"] = st
        rec = _recorded_id(proto, p)
        df[f"{p}_estimator"] = rec
        df[f"{p}_estimator_version"] = (
            E.split_estimator(rec)[1] if version_form == "version" else rec)
    stat = df[[f"{p}_status" for p in E.PRINCIPLES]].to_numpy()
    df["MPC_verdict"] = np.where((stat == "ABSENT").any(axis=1), "EXCLUDED",
                                 "MPC_CONSISTENT")
    df["MPC_reason"] = ""
    return df


def _set_id(df, row, p, rec, version_form):
    df.loc[row, f"{p}_estimator"] = rec
    df.loc[row, f"{p}_estimator_version"] = (
        E.split_estimator(rec)[1] if version_form == "version" else rec)


def test_registry_estimators_are_the_ids_the_evidence_layer_records():
    """Regression: registry 0.2.0 named IIM compute_IIM:delta_psi, while the
    evidence layer records compute_IIM:<cut_mode>@<version>, so every IIM
    row of the pipeline would have been refused."""
    reg = rp.load_registry()
    entry, proto = _registry_protocol(reg)
    assert entry["file"] == "protocols/mpc_default_v1.json"
    names = {e["principle"]: e["estimator"] for e in reg["estimators"]}
    assert set(names) == set(entry["necessity_set"]) == set(E.PRINCIPLES)
    for p in entry["necessity_set"]:
        name, version = E.split_estimator(_recorded_id(proto, p))
        assert names[p] == name, p
        assert version == VERSIONS[p], p
    assert names["IIM"] == "compute_IIM:bidirectional"
    # the declared fallbacks are recorded under their own names and stay
    # unregistered, so their rows are refused
    ram_fb = "compute_RAM:" + proto.estimator_options("RAM")["update_fallback"]
    srpi_fb = "compute_SRPI:" + proto.estimator_options("SRPI")["mode_fallback"]
    assert {ram_fb, srpi_fb} == {"compute_RAM:feedback_magnitude",
                                 "compute_SRPI:legacy"}
    assert not {ram_fb, srpi_fb} & set(names.values())


@pytest.mark.parametrize("version_form", ["version", "id"])
def test_step2_table_in_pipeline_format_end_to_end(tmp_path, version_form):
    reg = rp.load_registry()
    _, proto = _registry_protocol(reg)
    frozen = _frozen(reg, proto)
    reg_path = tmp_path / "registry_frozen.yaml"
    reg_path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    df = _step2_table(proto, version_form)
    assert "protocol_hash" not in df.columns  # the pipeline's column name only

    def run(table, name, *extra):
        path = tmp_path / f"{name}.csv"
        table.to_csv(path, index=False)
        out = tmp_path / name
        code = rp.main(["--registry", str(reg_path), "--results", str(path),
                        "--out", str(out), "--skip-freeze-tag-check", *extra])
        return code, out

    # registered versions: accepted, every outcome confirmatory
    code, out = run(df, "accepted")
    assert code == 0
    report = json.loads((out / "predictions_report.json").read_text())
    assert report["registration_problems"] == []
    assert list(report["stance"]) == ["confirmatory"]
    table = pd.read_csv(out / "predictions_results.csv")
    assert set(table["stratum"]) == {"confirmatory"}
    assert not table["unregistered"].any()
    assert set(table["hypothesis"]) == set(rp.REQUIRED_HYPOTHESES)

    # unregistered: a later IIM version, the directional cut, the RAM fallback
    cases = {
        "later_version": ("IIM", _recorded_id(proto, "IIM", "iim-v5-2026.10")),
        "directional": ("IIM", E.estimator_id("IIM", "directional", VERSIONS["IIM"])),
        "fallback": ("RAM", E.estimator_id("RAM", "feedback_magnitude",
                                           VERSIONS["RAM"])),
    }
    for name, (p, rec) in cases.items():
        bad = df.copy()
        _set_id(bad, 7, p, rec, version_form)
        code, out = run(bad, name)
        assert code == rp.REFUSAL_EXIT, name
        reasons = json.loads((out / "predictions_refusal.json").read_text())["reasons"]
        assert reasons == [
            f"results not registered: unregistered {p} estimator versions ['{rec}']"
        ], name
    # the refused table may still be evaluated as an explicitly labelled
    # exploratory run
    code, out = run(bad, "fallback_exploratory", "--allow-unregistered")
    assert code == 0
    table = pd.read_csv(out / "predictions_results.csv")
    assert set(table["stratum"]) == {"exploratory"} and table["unregistered"].all()

    # an id whose version contradicts the version column is refused
    if version_form == "version":
        bad = df.copy()
        bad.loc[3, "NAS_estimator"] = _recorded_id(proto, "NAS", "nas-v1-2026.08")
        code, out = run(bad, "inconsistent")
        assert code == rp.REFUSAL_EXIT
        reasons = json.loads((out / "predictions_refusal.json").read_text())["reasons"]
        assert "inconsistent NAS_estimator / NAS_estimator_version" in reasons[0]

    # the registry 0.2.0 IIM name refuses every pipeline row
    old = copy.deepcopy(frozen)
    next(e for e in old["estimators"] if e["principle"] == "IIM")[
        "estimator"] = "compute_IIM:delta_psi"
    with pytest.raises(rp.RegistryRefusal, match="unregistered IIM estimator"):
        rp.evaluate(old, df)


def test_pipeline_output_is_accepted_end_to_end(tmp_path):
    """compute_synergy_ci on a tiny synthetic layout under a protocol derived
    from the registry's protocol file (its IIM options; necessity set
    {IIM}), assembled into a step-2 table, is accepted by run_predictions
    when its IIM version is registered and refused otherwise."""
    from test_verdict_wiring import IIM_TOY, TR, _layout

    reg = rp.load_registry()
    _, default = _registry_protocol(reg)
    proto = default.replace(necessity_set=("IIM",))
    prep, _ = _layout(tmp_path)
    df = sc.compute_synergy_ci(str(prep), "toy", [0.5], sessions=("awake", "deep"),
                               tr=TR, mpc_metrics=("IIM",), compute_ci=False,
                               protocol=proto, **IIM_TOY)
    assert len(df) >= 2
    assert (df["MPC_protocol_hash"] == proto.hash).all()
    assert (df["IIM_estimator"] == _recorded_id(proto, "IIM")).all()
    # the pipeline writes the version of the recorded id next to it
    assert (df["IIM_estimator_version"]
            == E.split_estimator(_recorded_id(proto, "IIM"))[1]).all()
    # step-2 assembly: dataset, episode and report label
    table = df.assign(dataset="dream_mediated",
                      episode_id=[f"e{i}" for i in range(len(df))],
                      report_positive=(df["session"] == "awake").to_numpy())
    frozen = _frozen(reg, proto, necessity_set=("IIM",))
    assert rp.registration_problems(table, frozen) == []
    res = rp.evaluate(frozen, table)
    assert res["registration_problems"] == []
    assert {r["stratum"] for r in res["rows"]} == {"confirmatory"}
    # the same output against a registry that registers another IIM version
    other = _frozen(reg, proto, necessity_set=("IIM",),
                    versions={**VERSIONS, "IIM": "iim-v3-2026.08"})
    with pytest.raises(rp.RegistryRefusal) as exc:
        rp.evaluate(other, table)
    assert exc.value.reasons == [
        "results not registered: unregistered IIM estimator versions "
        f"['{_recorded_id(proto, 'IIM')}']"]
    # and against another protocol (the full default necessity set)
    with pytest.raises(rp.RegistryRefusal, match="unregistered protocol hashes"):
        rp.evaluate(_frozen(reg, default), table)
