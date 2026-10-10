from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_pipeline import evidence as E
from impact_pipeline import mpc_readiness
from impact_pipeline.mpc_readiness import check_mpc_readiness
from impact_pipeline.synergy_ci import nas_hub_missing

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "protocols" / "mpc_default_v1.json"
HUB_EXAMPLE = (REPO / "protocols" / "examples"
               / "mpc_default_v1_schaefer400_7networks_hub.json")


def _write_events(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)


def _write_ts(path, n_time=80, n_regions=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    arr = rng.randn(n_time, n_regions)
    np.save(path, arr)


def _all_ready_layout(tmp_path):
    prep = tmp_path / "preprocessed"
    bids = tmp_path / "bids"
    subj = "01"
    ses = "awake"

    ts_path = prep / subj / ses / "audio" / f"{subj}_run-1_schaefer400_ts.npy"
    _write_ts(ts_path)
    _write_ts(prep / subj / ses / "rest" / f"{subj}_run-1_schaefer400_ts.npy")
    _write_ts(prep / subj / "deep" / "rest" / f"{subj}_run-1_schaefer400_ts.npy")

    ev_path = bids / f"sub-{subj}" / "func" / f"sub-{subj}_task-audioawake_run-01_events.tsv"
    # RAM needs goal, stimulus and valued feedback events (>= 6 stimulus
    # events for the cross-validated goal alignment, >= 3 feedback values).
    rows = []
    for i in range(6):
        t0 = 0.2 + 1.8 * i
        for dt, label, reward in (
            (0.0, "goal_cue", np.nan),
            (0.6, "audio_stim", np.nan),
            (1.2, "feedback", float(i % 3)),
        ):
            row = {"onset": t0 + dt, "duration": 0.1, "trial_type": label}
            rows.append(dict(row, reward=reward))
    for i in range(3):
        t0 = 11.0 + 1.2 * i
        rows.append({"onset": t0, "duration": 0.1, "trial_type": "self_name"})
        rows.append({"onset": t0 + 0.6, "duration": 0.1, "trial_type": "other_name"})
    _write_events(ev_path, rows)
    return prep, bids, ses


def _check(prep, bids, ses, **kwargs):
    return check_mpc_readiness(
        prep_root=str(prep),
        bids_root=str(bids),
        atlas="schaefer400",
        condition="audio",
        sessions=[ses],
        **kwargs,
    )


def test_readiness_all_mpcs_ready(tmp_path):
    prep, bids, ses = _all_ready_layout(tmp_path)
    df, summary = _check(prep, bids, ses)
    assert df.shape[0] == 1
    row = df.iloc[0]
    assert bool(row["RAM_ready"])
    assert bool(row["PDI_ready"])
    assert bool(row["PDI_anchor_ready"])
    assert bool(row["PDI_task_ready"])
    assert bool(row["NAS_ready"])
    assert bool(row["IIM_ready"])
    assert bool(row["SRPI_ready"])
    assert bool(row["CI_ready"])
    assert summary["metrics"]["CI"]["ready"] == 1


# --------------------------------------------------------------------------
# NAS capacity needs a declared hub, as in the evidence layer
# --------------------------------------------------------------------------
def _v1_with_hub(nodes):
    payload = E.Protocol.from_json(V1).to_dict()
    payload["estimators"]["NAS"] = {**payload["estimators"]["NAS"],
                                    "workspace_nodes": list(nodes)}
    return payload


def test_nas_capacity_without_a_declared_hub_is_never_ready(tmp_path):
    prep, bids, ses = _all_ready_layout(tmp_path)
    df, summary = _check(prep, bids, ses, protocol=str(V1))
    row = df.iloc[0]
    assert not bool(row["NAS_ready"])
    assert row["NAS_reason"] == "NO_DECLARED_WORKSPACE"
    assert row["NAS_reason"] == E.REASON_NO_DECLARED_WORKSPACE
    assert not bool(row["CI_ready"])
    # the other metrics do not depend on the hub
    assert bool(row["IIM_ready"]) and bool(row["PDI_ready"])
    assert summary["metrics"]["NAS"]["ready"] == 0
    settings = summary["settings"]
    assert settings["nas_mode"] == "capacity"
    assert settings["nas_hub_declared"] is False
    assert settings["protocol_hash"] == E.Protocol.from_json(V1).hash
    # the same through nas_params alone (no protocol)
    df, _ = _check(prep, bids, ses, nas_params={"mode": "capacity"})
    assert df.iloc[0]["NAS_reason"] == "NO_DECLARED_WORKSPACE"


def test_nas_capacity_with_a_declared_hub_is_ready(tmp_path):
    prep, bids, ses = _all_ready_layout(tmp_path)
    df, summary = _check(prep, bids, ses, protocol=_v1_with_hub([0, 1, 2]))
    row = df.iloc[0]
    assert bool(row["NAS_ready"]) and row["NAS_reason"] == "ok"
    assert summary["settings"]["nas_hub_declared"] is True
    df, _ = _check(prep, bids, ses,
                   nas_params={"mode": "capacity", "workspace_nodes": [0, 1]})
    assert bool(df.iloc[0]["NAS_ready"])
    # without a protocol NAS is checked in its legacy mode
    df, summary = _check(prep, bids, ses)
    assert bool(df.iloc[0]["NAS_ready"])
    assert summary["settings"]["nas_mode"] == "legacy"
    assert summary["settings"]["nas_hub_declared"] is None


def test_nas_capacity_with_a_hub_that_does_not_fit_is_not_ready(tmp_path):
    # The layout's recordings have 6 nodes; compute_NAS refuses a hub outside
    # them, so the run could not define NAS and readiness must not say ready.
    from impact_pipeline.mpc_metrics import compute_NAS

    prep, bids, ses = _all_ready_layout(tmp_path)
    ts = np.random.default_rng(0).standard_normal((6, 80))
    for protocol, nas_params in (
        (str(HUB_EXAMPLE), None),  # a Schaefer-400 hub on 6 nodes
        (_v1_with_hub([0, 6]), None),
        (None, {"mode": "capacity", "workspace_nodes": [10]}),
        (None, {"mode": "capacity", "workspace_nodes": [True, False]}),
    ):
        df, summary = _check(prep, bids, ses, protocol=protocol,
                             nas_params=nas_params)
        row = df.iloc[0]
        assert not bool(row["NAS_ready"]), (protocol, nas_params)
        assert row["NAS_reason"] == "INVALID_WORKSPACE"
        assert row["NAS_reason"] == mpc_readiness.NAS_INVALID_WORKSPACE
        assert not bool(row["CI_ready"])
        assert bool(row["IIM_ready"]) and bool(row["PDI_ready"])
        assert summary["metrics"]["NAS"]["ready"] == 0
        assert summary["settings"]["nas_hub_declared"] is True
        hub = (nas_params or {}).get("workspace_nodes")
        if hub is None:
            hub = E.resolve_protocol(protocol).estimator_options("NAS")[
                "workspace_nodes"]
        with pytest.raises(ValueError, match="workspace_nodes"):
            compute_NAS(ts, tr=0.5, mode="capacity", workspace_nodes=hub)
    # a hub inside the node count is ready (the boundary index included)
    df, _ = _check(prep, bids, ses, protocol=_v1_with_hub([0, 5]))
    assert bool(df.iloc[0]["NAS_ready"]) and df.iloc[0]["NAS_reason"] == "ok"
    # legacy NAS drops declared nodes outside the recording and stays defined
    df, summary = _check(prep, bids, ses, nas_params={"workspace_nodes": [99]})
    assert summary["settings"]["nas_mode"] == "legacy"
    assert bool(df.iloc[0]["NAS_ready"])


def test_readiness_nas_rule_is_the_evidence_layer_rule():
    assert nas_hub_missing(V1) is True
    assert nas_hub_missing(HUB_EXAMPLE) is False
    assert nas_hub_missing(None) is False
    assert nas_hub_missing(None, {"mode": "capacity"}) is True
    assert nas_hub_missing(None, {"mode": "capacity", "workspace_nodes": [0]}) is False
    assert nas_hub_missing(_v1_with_hub([0, 1])) is False
    with pytest.raises(ValueError, match="differs"):
        nas_hub_missing(V1, {"mode": "legacy"})


def test_readiness_cli_defaults_to_the_empirical_protocol():
    resolve = mpc_readiness.resolve_protocol_arg
    assert Path(mpc_readiness.DEFAULT_EMPIRICAL_PROTOCOL) == V1
    assert resolve(None) == str(V1)
    assert resolve("none") is None and resolve("FLAGS") is None
    assert resolve(str(HUB_EXAMPLE)) == str(HUB_EXAMPLE)


def test_readiness_ram_not_ready_without_feedback(tmp_path):
    prep = tmp_path / "preprocessed"
    bids = tmp_path / "bids"
    subj = "02"
    ses = "awake"

    ts_path = prep / subj / ses / "audio" / f"{subj}_run-1_schaefer400_ts.npy"
    _write_ts(ts_path)

    ev_path = bids / f"sub-{subj}" / "func" / f"sub-{subj}_task-audioawake_run-01_events.tsv"
    _write_events(
        ev_path,
        [
            {"onset": 0.30, "duration": 0.1, "trial_type": "audio_stim"},
            {"onset": 1.30, "duration": 0.1, "trial_type": "audio_stim"},
        ],
    )

    df, _ = check_mpc_readiness(
        prep_root=str(prep),
        bids_root=str(bids),
        atlas="schaefer400",
        condition="audio",
        sessions=[ses],
    )
    row = df.iloc[0]
    assert not bool(row["RAM_ready"])
    assert not bool(row["PDI_ready"])
    assert not bool(row["PDI_anchor_ready"])
    assert not bool(row["PDI_task_ready"])
    assert row["RAM_reason"] in {"missing_feedback_events", "missing_or_nonvarying_feedback_values"}
    assert not bool(row["CI_ready"])
