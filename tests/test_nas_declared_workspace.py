"""
Regression (1.1.0 post-freeze fix): NAS ``mode='capacity'`` without a
declared hub. The frozen code raised in ``compute_NAS`` (``requires declared
workspace_nodes``), so an empirical run under ``mpc_default_v1.json`` (NAS
capacity, no hub) crashed. The pipeline now records NAS as UNDEFINED with the
reason ``NO_DECLARED_WORKSPACE`` and never calls the estimator; a derived
protocol that declares the hub computes NAS as before.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.build_example_protocols as bep
from impact_pipeline import evidence as E
from impact_pipeline import mpc_metrics as mm
from impact_pipeline import synergy_ci as sc
from test_nas_capacity import _hub_network

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "protocols" / "mpc_default_v1.json"  # the default empirical protocol
BRAM = REPO / "protocols" / "mpc_behavioural_ram_v1.json"  # opt-in
HUB = [0, 1, 2, 3]


def _prep(tmp_path, n_time=1500):
    prep = tmp_path / "prep"
    for ses, (kind, seed) in {"awake": ("bidir", 0), "deep": ("none", 1)}.items():
        x, _ = _hub_network(kind, seed, n_time=n_time)
        d = prep / "s1" / ses / "audio"
        d.mkdir(parents=True)
        np.save(d / "s1_run-1_toy_ts.npy", x.T)
    return prep


def _nas_run(prep, protocol, **kw):
    kw.setdefault("bootstrap_se", 5)
    return sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake", "deep"), tr=1.0,
        mpc_metrics=("NAS",), compute_ci=False, protocol=protocol, **kw)


def test_estimator_still_requires_the_hub():
    """The estimator is unchanged (frozen): it raises without a hub; only the
    pipeline wiring changed."""
    x, _ = _hub_network("bidir", 0, n_time=400)
    with pytest.raises(ValueError, match="requires declared workspace_nodes"):
        mm.compute_NAS(x, tr=1.0, mode="capacity", return_details=True)


@pytest.mark.parametrize("protocol", [V1, BRAM], ids=["v1", "behavioural-ram"])
def test_default_protocols_give_undefined_nas_instead_of_a_crash(
        tmp_path, monkeypatch, protocol):
    prep = _prep(tmp_path, n_time=300)
    calls = []
    real = sc.compute_NAS

    def spy(*a, **k):
        calls.append(k)
        return real(*a, **k)

    monkeypatch.setattr(sc, "compute_NAS", spy)
    df = _nas_run(prep, str(protocol))
    assert not calls  # the estimator is not called without a hub (no bootstrap)
    assert (df["NAS_status"] == "UNDEFINED").all()
    assert df["NAS"].isna().all() and df["NAS_estimate"].isna().all()
    assert (df["NAS_null_n"] == 0).all() and (df["NAS_boot_n"] == 0).all()
    assert df["MPC_reason"].str.contains(
        "UNDEFINED:NAS:NO_DECLARED_WORKSPACE", regex=False).all()
    nas_id = f"compute_NAS:capacity@{mm.ESTIMATOR_VERSIONS['NAS']}"
    assert (df["NAS_estimator"] == nas_id).all()
    assert (df["MPC_protocol_hash"] == E.Protocol.from_json(protocol).hash).all()
    # never EXCLUDED through NAS: UNDEFINED is not ABSENT
    assert not (df["MPC_verdict"] == "EXCLUDED").any()


def test_reason_is_a_stable_detail_code():
    assert E.REASON_NO_DECLARED_WORKSPACE == "NO_DECLARED_WORKSPACE"
    kind, principle, detail = E.parse_reason("UNDEFINED:NAS:NO_DECLARED_WORKSPACE")
    assert (kind, principle, detail) == ("UNDEFINED", "NAS", "NO_DECLARED_WORKSPACE")
    text = (REPO / "docs" / "metrics.md").read_text(encoding="utf-8")
    assert "NO_DECLARED_WORKSPACE" in text


def test_a_declared_hub_computes_nas_capacity(tmp_path):
    """The other path: a derived protocol (v1 + declared hub, built by the
    example builder) runs the capacity estimator and judges its evidence."""
    prep = _prep(tmp_path)
    proto = bep.derived_protocol(V1, HUB, "test-v1-hub")
    assert proto.estimator_options("NAS") == {"mode": "capacity",
                                              "workspace_nodes": HUB}
    df = _nas_run(prep, proto.replace(necessity_set=("NAS",)), bootstrap_se=10)
    awake = df[df["session"] == "awake"].iloc[0]
    deep = df[df["session"] == "deep"].iloc[0]
    assert "NO_DECLARED_WORKSPACE" not in "".join(df["MPC_reason"])
    assert (df["NAS_null_n"] == mm.NAS_CAPACITY_DEFAULT_SURROGATES).all()
    assert np.isfinite(df["NAS_estimate"]).all() and np.isfinite(df["NAS_se"]).all()
    assert awake["NAS_status"] == "PRESENT"
    assert deep["NAS_status"] == "ABSENT" and deep["MPC_verdict"] == "EXCLUDED"


def test_hub_through_nas_params_is_also_a_declaration(tmp_path):
    prep = _prep(tmp_path, n_time=600)
    proto = E.Protocol.from_json(V1).replace(necessity_set=("NAS",))
    df = _nas_run(prep, proto, nas_params={"workspace_nodes": HUB}, bootstrap_se=0)
    assert "NO_DECLARED_WORKSPACE" not in "".join(df["MPC_reason"])
    assert np.isfinite(df["NAS_estimate"]).all()


def test_invalid_hub_declarations_still_raise(tmp_path):
    prep = _prep(tmp_path, n_time=300)
    proto = E.Protocol.from_json(V1).replace(necessity_set=("NAS",))
    with pytest.raises(ValueError, match="out of range"):
        _nas_run(prep, proto, nas_params={"workspace_nodes": [0, 99]})


def test_default_protocol_runs_all_five_principles_through_the_pipeline(tmp_path):
    """The default protocol runs through the pipeline with all five
    principles, NAS included."""
    from test_verdict_wiring import _layout, _run

    from impact_pipeline.run_synergy_ci import RAM_PARAM_PRESETS

    prep, onsets = _layout(tmp_path, with_events=True)
    ram = {**RAM_PARAM_PRESETS["eeg"], "quality_null_samples": 5}
    df = _run(prep, sessions=("awake",), stimulus_onsets=onsets, ram_params=ram,
              nas_params=None, protocol=str(V1), compute_ci=False)
    row = df.iloc[0]
    assert row["MPC_necessity_set"] == "RAM,PDI,NAS,IIM,SRPI"
    assert "UNDEFINED:NAS:NO_DECLARED_WORKSPACE" in row["MPC_reason"]
    assert row["MPC_verdict"] == "UNDETERMINED"
    assert json.loads(json.dumps(df.attrs["mpc_evidence"]["protocol"])) == (
        E.Protocol.from_json(V1).to_dict())
    # v1 keeps RAM's unimplemented channels: RAM is never ABSENT
    assert "NOT_IMPLEMENTED:RAM:perturbational" in row["MPC_reason"]
    assert row["RAM_status"] != "ABSENT"
