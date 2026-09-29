"""
End-to-end regression (1.1.0 post-freeze fixes): the default
empirical path of ``run_pipeline.py`` (real data, no ``--protocol``) on tiny
synthetic layouts.

- The command-line default is the preregistered
  ``protocols/mpc_default_v1.json`` (hash as at tag ``mpcbench-freeze-v1``).
  At the freeze an empirical run under it raised in step 2 (``compute_NAS
  mode='capacity' requires declared workspace_nodes``); now it completes,
  records NAS as ``UNDEFINED:NAS:NO_DECLARED_WORKSPACE`` and writes
  ``<P>_estimator_version`` for every principle.
- RAM under the default can be PRESENT but never ABSENT: on a behaviourally
  null run its behavioural (``default``) channel is ABSENT, and RAM stays
  UNDEFINED through its unimplemented ``perturbational`` and ``endogenous``
  channels, so behavioural non-response alone never excludes. The opt-in
  ``protocols/mpc_behavioural_ram_v1.json`` makes the same run's RAM ABSENT
  (verdict EXCLUDED).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_pipeline
from impact_pipeline import evidence as E
from impact_pipeline import mpc_metrics as mm
from impact_pipeline import synergy_ci as sc

PRINCIPLES = E.PRINCIPLES
REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "protocols" / "mpc_default_v1.json"
BEHAVIOURAL_RAM = REPO / "protocols" / "mpc_behavioural_ram_v1.json"
# the default protocol at tag mpcbench-freeze-v1 (must never change)
V1_FROZEN_HASH = "383eb310cf4479d3260bc5f43f8972bd0b12104a221579fea17c40e410357267"
BEHAVIOURAL_RAM_HASH = (
    "531c15b90ea0338402f3eb2df97ae27ab9a9d9a5ffb22e17c93413231ff220b8")


def _layout(tmp_path, n_time=160, n_nodes=6):
    bids, out = tmp_path / "bids", tmp_path / "out"
    rng = np.random.RandomState(1)
    for subj in ("01", "02", "03"):
        func = bids / f"sub-{subj}" / "func"
        func.mkdir(parents=True)
        (func / f"sub-{subj}_task-audioawake_run-01_bold.json").write_text(
            json.dumps({"RepetitionTime": 2.0}))
        for ses in ("awake", "deep"):
            d = out / "preprocessed" / subj / ses / "audio"
            d.mkdir(parents=True)
            np.save(d / f"{subj}_run-1_schaefer400_ts.npy",
                    rng.standard_normal((n_time, n_nodes)))
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "tiny", "BIDSVersion": "1.8.0"}))
    return bids, out


def test_default_empirical_run_completes_and_writes_the_new_columns(
        tmp_path, monkeypatch):
    bids, out = _layout(tmp_path)
    monkeypatch.setattr(run_pipeline, "create_doc", lambda *a, **k: None)
    protocol = run_pipeline.resolve_cli_protocol(None, "real", None)
    assert protocol == str(run_pipeline.DEFAULT_EMPIRICAL_PROTOCOL) == str(V1)
    run_pipeline.main(
        str(out), dataset_id="ds003171", bids_root_override=str(bids),
        protocol=protocol, iim_max_nodes_override=3, disable_iim_parallel=True,
        atlas_robustness=False, null_surrogates=3, bootstrap_se=2,
    )
    df = pd.read_csv(out / "cache" / "step2_df.csv", dtype={"subject": str})
    v1 = E.Protocol.from_json(run_pipeline.DEFAULT_EMPIRICAL_PROTOCOL)
    assert v1.hash == V1_FROZEN_HASH
    assert set(df["MPC_protocol_hash"]) == {V1_FROZEN_HASH}
    assert set(df["MPC_necessity_set"]) == {"RAM,PDI,NAS,IIM,SRPI"}
    assert (df["NAS_status"] == "UNDEFINED").all()
    assert df["MPC_reason"].str.contains(
        "UNDEFINED:NAS:NO_DECLARED_WORKSPACE", regex=False).all()
    # v1 declares RAM's unimplemented channels: RAM is never ABSENT
    for ch in ("perturbational", "endogenous"):
        assert df["MPC_reason"].str.contains(
            f"NOT_IMPLEMENTED:RAM:{ch}", regex=False).all()
    assert not (df["RAM_status"] == "ABSENT").any()
    assert not (df["MPC_verdict"] == "EXCLUDED").any()
    for p in PRINCIPLES:
        col = f"{p}_estimator_version"
        assert col in df.columns, col
        assert set(df[col]) == {mm.ESTIMATOR_VERSIONS[p]}, col
        assert all(e.endswith("@" + mm.ESTIMATOR_VERSIONS[p])
                   for e in df[f"{p}_estimator"])
    prov = json.loads((out / "cache" / "provenance_manifest.json").read_text())
    assert prov["status"] == "completed"
    assert prov["parameters"]["mpc_protocol"] == {
        "source": str(run_pipeline.DEFAULT_EMPIRICAL_PROTOCOL), "hash": v1.hash}


# --------------------------------------------------------------------------
# RAM under the default protocol: PRESENT possible, ABSENT never
# --------------------------------------------------------------------------
TR = 0.1
N_TIME = 1200


def _behavioural_layout(tmp_path, amp=2.0, noise=0.2, subjects=("01", "02", "03"),
                        atlas="schaefer400", seed=3):
    """
    Participants with an awake run whose node 0 responds to every stimulus
    (``amp`` over 3 samples) and a behaviourally null deep run (the same kind
    of event train, no response): ``<out>/preprocessed/<subj>/<ses>/audio/
    <subj>_run-1_<atlas>_ts.npy`` and BIDS events.tsv with goal cues, stimuli
    and valued feedback (the RAM event contract).
    """
    bids, out = tmp_path / "bids", tmp_path / "out"
    rng = np.random.default_rng(seed)
    for subj in subjects:
        func = bids / f"sub-{subj}" / "func"
        func.mkdir(parents=True)
        (func / f"sub-{subj}_task-audioawake_run-01_bold.json").write_text(
            json.dumps({"RepetitionTime": TR}))
        for ses in ("awake", "deep"):
            # jittered inter-event intervals (a rigid shift of a periodic
            # train could realign it with itself)
            stim = 3.0 + np.cumsum(rng.uniform(1.4, 2.6, 50))
            x = noise * rng.standard_normal((4, N_TIME))
            if ses == "awake":
                for on in stim:
                    k = int(round(on / TR))
                    x[0, k + 1:k + 4] += amp
            rows = []
            for on in stim:
                rows += [
                    {"onset": on - 0.5, "duration": 0.1, "trial_type": "goal_cue"},
                    {"onset": on, "duration": 0.1, "trial_type": "audio_stim"},
                    {"onset": on + 0.8, "duration": 0.1, "trial_type": "feedback",
                     "value": float(rng.choice([-1.0, 1.0]))},
                ]
            pd.DataFrame(rows).to_csv(
                func / f"sub-{subj}_task-audio{ses}_run-01_events.tsv",
                sep="\t", index=False)
            d = out / "preprocessed" / subj / ses / "audio"
            d.mkdir(parents=True)
            np.save(d / f"{subj}_run-1_{atlas}_ts.npy", x.T)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "tiny-behavioural", "BIDSVersion": "1.8.0"}))
    return bids, out


def _evoked(ts, tr, onsets, node=0):
    idx = np.rint(np.asarray(onsets, dtype=float) / tr).astype(int)
    idx = idx[(idx >= 1) & (idx + 4 < ts.shape[1])]
    if idx.size < 3:
        return float("nan")
    return float(np.mean([ts[node, k + 1:k + 4].mean() - ts[node, k - 1]
                          for k in idx]))


def install_behavioural_ram(monkeypatch):
    """
    The behavioural (untyped ``default``) RAM channel as a known-answer
    evoked-response estimator, so that "responsive" and "behaviourally null"
    are known by construction; typed channels go to the real ``compute_RAM``
    (``perturbational``/``endogenous``: ``NOT_IMPLEMENTED``). Everything else
    (protocol resolution, channel loop, onset-jitter null, bootstrap SE,
    cohort reference, verdict) is the pipeline's own code. Returns the list of
    requested channels (None = untyped).
    """
    real = sc.compute_RAM
    calls = []

    def ram(ts, tr=None, stimulus_onsets=None, impact_channel=None,
            return_details=False, **kw):
        calls.append(impact_channel)
        if impact_channel is not None:
            return real(ts, tr=tr, stimulus_onsets=stimulus_onsets,
                        impact_channel=impact_channel,
                        return_details=return_details, **kw)
        val = _evoked(ts, tr, (stimulus_onsets or {}).get("onsets", []))
        return {"value": val, "undefined_reason": None} if return_details else val

    monkeypatch.setattr(sc, "compute_RAM", ram)
    return calls


@pytest.fixture
def behavioural_ram(monkeypatch):
    monkeypatch.setattr(run_pipeline, "create_doc", lambda *a, **k: None)
    return install_behavioural_ram(monkeypatch)


def _run_ram(bids, out, protocol):
    run_pipeline.main(
        str(out), dataset_id="ds003171", bids_root_override=str(bids),
        protocol=protocol, mpc_metrics=["RAM"], compute_ci=False,
        atlas_robustness=False, null_surrogates=19, bootstrap_se=20,
    )
    df = pd.read_csv(out / "cache" / "step2_df.csv", dtype={"subject": str})
    return one_row_per_run(df)


def one_row_per_run(df):
    return df[np.isclose(df["theta"], df["theta"].min())].set_index(
        ["subject", "session"]).sort_index()


def test_ram_under_the_default_is_present_but_never_absent(
        tmp_path, behavioural_ram):
    bids, out = _behavioural_layout(tmp_path)
    protocol = run_pipeline.resolve_cli_protocol(None, "real", None)
    assert protocol == str(V1)
    df = _run_ram(bids, out, protocol)
    assert set(df["MPC_protocol_hash"]) == {V1_FROZEN_HASH}
    # the typed channels were asked for (and are not implemented)
    assert {"perturbational", "endogenous"} <= set(behavioural_ram)
    awake, deep = df.xs("awake", level="session"), df.xs("deep", level="session")
    assert len(awake) == len(deep) == 3
    # responsive runs: RAM PRESENT through the behavioural channel
    assert (awake["RAM_status"] == "PRESENT").all()
    assert awake["RAM_channels"].str.startswith("default:PRESENT").all()
    # behaviourally null runs: the behavioural channel is credibly ABSENT ...
    assert deep["RAM_channels"].str.startswith("default:ABSENT").all()
    # ... and RAM is still not ABSENT: its unimplemented channels are UNDEFINED
    assert (deep["RAM_status"] == "UNDEFINED").all()
    for ch in ("perturbational", "endogenous"):
        assert deep["RAM_channels"].str.contains(f"{ch}:UNDEFINED").all()
        assert deep["MPC_reason"].str.contains(f"NOT_IMPLEMENTED:RAM:{ch}").all()
    assert not (df["MPC_verdict"] == "EXCLUDED").any()
    assert not df["MPC_reason"].str.contains("ABSENT:RAM").any()
    # the RAM_* columns then describe the deciding (unimplemented) channel, not
    # the behavioural one: why compute_empirical_reference.py refuses a per-
    # channel reference from a step-2 table (CHANNEL_NOT_IDENTIFIABLE)
    assert deep["RAM_estimate"].isna().all()
    assert np.isfinite(awake["RAM_estimate"]).all()


def test_behavioural_ram_protocol_makes_the_same_runs_absent(
        tmp_path, behavioural_ram):
    bids, out = _behavioural_layout(tmp_path)
    protocol = run_pipeline.resolve_cli_protocol(str(BEHAVIOURAL_RAM), "real", None)
    assert protocol == str(BEHAVIOURAL_RAM)
    df = _run_ram(bids, out, protocol)
    assert set(df["MPC_protocol_hash"]) == {BEHAVIOURAL_RAM_HASH}
    # only the behavioural channel is computed
    assert set(behavioural_ram) == {None}
    awake, deep = df.xs("awake", level="session"), df.xs("deep", level="session")
    assert (awake["RAM_status"] == "PRESENT").all()
    assert (deep["RAM_channels"] == "default:ABSENT").all()
    assert (deep["RAM_status"] == "ABSENT").all()
    assert (deep["MPC_verdict"] == "EXCLUDED").all()
    assert deep["MPC_reason"].str.contains("ABSENT:RAM", regex=False).all()
    assert not deep["MPC_reason"].str.contains("NOT_IMPLEMENTED").any()
    assert np.isfinite(deep["RAM_estimate"]).all()
    assert (deep["RAM_c_upper"] < E.Protocol.from_json(
        BEHAVIOURAL_RAM).cutoff_for("RAM")[1]).all()
