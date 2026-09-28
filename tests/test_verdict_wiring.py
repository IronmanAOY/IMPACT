"""
MPC verdict wiring (evidence layer v2): compute_synergy_ci evidence columns on
the construct scale, the honest defaults (no null: NO_NULL_CALIBRATION; no
bootstrap: NO_SAMPLING_SE), determinate verdicts on planted-structure toys,
protocol-selected estimator modes, declared RAM channels, the single-source
constraint, the applicability registry and the run_s_ci / run_pipeline
plumbing (local and Hunter finalize).
"""
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as t_dist

import run_pipeline
from impact_pipeline import evidence as E
from impact_pipeline import mpc_metrics as mm
from impact_pipeline import synergy_ci as sc
from impact_pipeline.event_parsing import events_table_to_bundle, read_events_table
from impact_pipeline.run_synergy_ci import RAM_PARAM_PRESETS
from test_hunter_pbs import _tiny_bids_and_prep
from test_nas_capacity import _hub_network
from test_srpi_agency import _agency_task
from test_synergy_ci import _NAS_PARAMS, _PDI_PARAMS, _SRPI_PARAMS

REPO = Path(__file__).resolve().parents[1]
TR = 0.1
N_NODES, N_TIME = 8, 480
NAS_TOY = {**_NAS_PARAMS, "tau": 0.5}  # >= 3 broadcast nodes of 8
IIM_TOY = dict(iim_bins=2, iim_max_nodes=3, iim_enable_parallel=False,
               iim_use_shared_memory=False)
V = E.Verdict
VERSIONS = mm.ESTIMATOR_VERSIONS


def _var(rng, coupling, common, n_time=N_TIME):
    """Stable VAR(1) ring (spectral radius <= 0.9) plus a shared AR(1) drive."""
    a = np.eye(N_NODES) * 0.3
    for i in range(N_NODES):
        a[(i + 1) % N_NODES, i] += coupling
    x = np.zeros((N_NODES, n_time))
    drive = np.zeros(n_time)
    for k in range(1, n_time):
        drive[k] = 0.8 * drive[k - 1] + rng.standard_normal()
        x[:, k] = a @ x[:, k - 1] + rng.standard_normal(N_NODES) + common * drive[k]
    return x


def _events_table(rng, n_time, planted_x=None):
    """Self/other name events plus goal-stimulus-feedback triplets (BIDS table)."""
    rows = []
    pat_self, pat_other = rng.standard_normal(N_NODES), rng.standard_normal(N_NODES)
    for i, on in enumerate(np.arange(2.0, n_time * TR - 4.0, 3.0)):
        lab = "self_name" if i % 2 == 0 else "other_name"
        rows.append({"onset": on, "duration": 0.1, "trial_type": lab})
        rows.append({"onset": on - 0.5, "duration": 0.1, "trial_type": "goal_cue"})
        rows.append({"onset": on + 0.2, "duration": 0.1, "trial_type": "audio_stim"})
        rows.append({"onset": on + 1.2, "duration": 0.1, "trial_type": "feedback",
                     "value": float(rng.choice([-1.0, 1.0]))})
        if planted_x is not None:
            k = int(round(on / TR))
            pat = 2.0 * pat_self if lab == "self_name" else 0.7 * pat_other
            planted_x[:, k + 2:k + 6] += pat[:, None]
            ks = int(round((on + 0.2) / TR))
            planted_x[:3, ks + 1:ks + 4] += 1.5
    return pd.DataFrame(rows).sort_values("onset", kind="mergesort")


def _layout(tmp_path, with_events=False, seed=0, n_time=N_TIME):
    """
    <prep>/s1/<ses>/audio/s1_run-1_toy_ts.npy (+ rest runs for PDI) and, with
    events, a BIDS-like events.tsv per session converted to the event bundle.
    awake: coupled VAR + shared drive (+ planted evoked responses);
    deep: independent AR noise.
    """
    rng = np.random.default_rng(seed)
    prep = tmp_path / "prep"
    bids = tmp_path / "bids" / "sub-s1" / "func"
    bids.mkdir(parents=True)
    onsets = {"s1": {}}
    for ses, coupling in (("awake", 0.4), ("deep", 0.0)):
        x = _var(rng, coupling, common=2.5 * coupling, n_time=n_time)
        if with_events:
            ev = _events_table(rng, n_time, planted_x=x if coupling else None)
            fn = bids / f"sub-s1_task-audio{ses}_run-1_events.tsv"
            ev.to_csv(fn, sep="\t", index=False)
            onsets["s1"][ses] = (events_table_to_bundle(read_events_table(fn)), "1")
        d = prep / "s1" / ses / "audio"
        d.mkdir(parents=True)
        np.save(d / "s1_run-1_toy_ts.npy", x.T)
        r = prep / "s1" / ses / "rest"
        r.mkdir(parents=True)
        np.save(r / "s1_run-2_toy_ts.npy", rng.standard_normal((n_time, N_NODES)))
    return prep, (onsets if with_events else None)


def _run(prep, **kw):
    kw.setdefault("sessions", ("awake", "deep"))
    kw.setdefault("nas_params", NAS_TOY)
    kw.setdefault("srpi_params", _SRPI_PARAMS)
    kw.setdefault("pdi_params", _PDI_PARAMS)
    return sc.compute_synergy_ci(str(prep), "toy", [0.3, 0.6], tr=TR, **IIM_TOY, **kw)


def _row(df, session):
    return df[df["session"] == session].iloc[0]


# --------------------------------------------------------------------------
# honest defaults: no null (K = 0), no sampling SE (K_b = 0)
# --------------------------------------------------------------------------
def test_k0_emits_verdict_columns_and_is_undetermined_everywhere(tmp_path):
    prep, onsets = _layout(tmp_path, with_events=True)
    ram = {**RAM_PARAM_PRESETS["eeg"], "quality_null_samples": 5}
    mm._COMPUTE_CI_DEPRECATION_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = _run(prep, stimulus_onsets=onsets, ram_params=ram)
    assert not [w for w in caught if "compute_CI is deprecated" in str(w.message)]
    for col in ("CI", "CI_defined", "CI_missing", *sc.MPC_EVIDENCE_COLUMNS):
        assert col in df.columns, col
    assert (df["MPC_verdict"] == "UNDETERMINED").all()
    assert df["MPC_degree"].isna().all()
    assert (df["MPC_null_surrogates"] == 0).all()
    assert (df["MPC_bootstrap_se"] == 0).all()
    assert (df["MPC_null_families"] == "").all()
    assert (df["MPC_necessity_set"] == "RAM,PDI,NAS,IIM,SRPI").all()
    proto = E.Protocol.from_dict(df.attrs["mpc_evidence"]["protocol"])
    assert (df["MPC_protocol_hash"] == proto.hash).all()
    assert df.attrs["mpc_evidence"]["estimator_versions"] == VERSIONS
    for _, row in df.iterrows():
        reasons = row["MPC_reason"].split(";")
        for p in E.PRINCIPLES:
            assert row[f"{p}_status"] == "UNDEFINED"
            assert row[f"{p}_null_n"] == 0 and np.isnan(row[f"{p}_null_mean"])
            assert row[f"{p}_estimator"].startswith(f"compute_{p}:")
            assert row[f"{p}_estimator"].endswith("@" + VERSIONS[p])
            if np.isfinite(row[f"{p}_estimate"]):
                assert f"NO_NULL_CALIBRATION:{p}" in reasons
            else:
                assert any(r.startswith(f"UNDEFINED:{p}:") for r in reasons)
        assert E.verdict_from_reasons(reasons) == V.UNDETERMINED
    # the metric columns keep their uncalibrated meaning
    awake = _row(df, "awake")
    for p in ("PDI", "NAS", "IIM", "SRPI"):
        assert np.isfinite(awake[f"{p}_estimate"]), p
    np.testing.assert_allclose(df["NAS"], df["NAS_estimate"])
    np.testing.assert_allclose(df["PDI"], np.maximum(df["PDI_estimate"], 0.0))
    np.testing.assert_allclose(df["PDI_anchor"], df["PDI_estimate"])
    for _, grp in df.groupby("session"):  # evidence is per run
        assert grp["MPC_reason"].nunique() == 1


def test_null_without_bootstrap_is_no_sampling_se(tmp_path):
    prep, _ = _layout(tmp_path)
    df = _run(prep, mpc_metrics=("NAS",), compute_ci=False, null_surrogates=5,
              necessity_set=("NAS",))
    for ses in ("awake", "deep"):
        row = _row(df, ses)
        assert row["MPC_reason"] == "NO_SAMPLING_SE:NAS"
        assert np.isfinite(row["NAS_c"]) and np.isnan(row["NAS_c_lower"])
        assert row["NAS_boot_n"] == 0 and np.isnan(row["NAS_se"])
    # the awake run is its own cohort reference: c = 1 there
    assert _row(df, "awake")["NAS_c"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# K > 0 and K_b > 0 on a planted-structure toy
# --------------------------------------------------------------------------
PLANTED = dict(mpc_metrics=("PDI", "NAS", "IIM"), compute_ci=False,
               null_surrogates=20, null_seed=0, bootstrap_se=20)


@pytest.fixture(scope="module")
def planted_runs(tmp_path_factory):
    prep, _ = _layout(tmp_path_factory.mktemp("planted"), n_time=1500)
    two = _run(prep, necessity_set=("NAS", "IIM"), **PLANTED)
    three = _run(prep, necessity_set="PDI,NAS,IIM", **PLANTED)
    return two, three


def test_k_positive_gives_determinate_verdicts_on_planted_structure(planted_runs):
    df, _ = planted_runs
    awake, deep = _row(df, "awake"), _row(df, "deep")
    # coupled network with a shared broadcast drive: NAS and IIM credibly above
    # the cutoff z = 0.25 of the construct scale
    assert awake["NAS_status"] == "PRESENT" and awake["IIM_status"] == "PRESENT"
    assert awake["MPC_verdict"] == "MPC_CONSISTENT" and awake["MPC_reason"] == ""
    # independent noise: IIM credibly below delta = 0.10 -> excluded
    assert deep["IIM_status"] == "ABSENT"
    assert deep["MPC_verdict"] == "EXCLUDED" and "ABSENT:IIM" in deep["MPC_reason"]
    # the awake run is the (single-subject) cohort reference: c = 1, degree 1
    for p in ("NAS", "IIM"):
        assert awake[f"{p}_c"] == pytest.approx(1.0)
        assert awake[f"{p}_reference"] == pytest.approx(
            awake[f"{p}_estimate"] - awake[f"{p}_null_mean"])
        assert np.isnan(awake[f"{p}_reference_se"])  # one subject: unavailable
    assert awake["MPC_degree"] == pytest.approx(1.0)
    assert np.isnan(deep["MPC_degree"])
    # the metric columns hold the calibrated (excess over null) values
    for p in ("NAS", "IIM"):
        assert (df[f"{p}_null_n"] == 20).all() and (df[f"{p}_boot_n"] > 15).all()
        np.testing.assert_allclose(
            df[p], np.maximum(df[f"{p}_estimate"] - df[f"{p}_null_mean"], 0.0),
            rtol=1e-9, atol=1e-12,
        )
    assert (df["MPC_null_families"] ==
            "PDI:phase_randomize;NAS:circular_shift;IIM:circular_shift").all()
    assert (df["MPC_bootstrap_se"] == 20).all()


def test_construct_scale_columns_follow_the_propagation_formula(planted_runs):
    df, _ = planted_runs
    for _, row in df.drop_duplicates("session").iterrows():
        for p in ("NAS", "IIM"):
            ref = row[f"{p}_reference"]
            c = (row[f"{p}_estimate"] - row[f"{p}_null_mean"]) / ref
            assert row[f"{p}_c"] == pytest.approx(c, rel=1e-12)
            se_nu = row[f"{p}_null_sd"] / math.sqrt(row[f"{p}_null_n"])
            # excess-scale reference: dc/dm = 1/ref, dc/dnu = -1/ref
            se_c = math.hypot(row[f"{p}_se"] / ref, se_nu / ref)
            assert row[f"{p}_c_se"] == pytest.approx(se_c, rel=1e-12)
            # bootstrap SE from B valid replicates: B - 1 degrees of freedom,
            # Welch-Satterthwaite df of se_c, Student-t quantile
            assert row[f"{p}_se_df"] == row[f"{p}_boot_n"] - 1
            s_samp = row[f"{p}_se"] / ref
            df_eff = row[f"{p}_se_df"] * (se_c ** 2 / s_samp ** 2) ** 2
            assert row[f"{p}_c_df"] == pytest.approx(df_eff, rel=1e-9)
            z = t_dist.ppf(0.95, df_eff)
            assert z > 1.6448536269514722
            assert row[f"{p}_c_lower"] == pytest.approx(c - z * se_c, rel=1e-12)
            assert row[f"{p}_margin"] == pytest.approx(c - z * se_c - 0.25, abs=1e-12)
            assert row[f"{p}_margin_absent"] == pytest.approx(
                0.10 - (c + z * se_c), abs=1e-12)
            assert row[f"{p}_channels"] == f"default:{row[f'{p}_status']}"


def test_necessity_set_changes_only_the_verdict(planted_runs):
    two, three = planted_runs
    # the null families, seeds and bootstrap do not depend on N
    for col in ("NAS_null_mean", "IIM_null_mean", "PDI_null_mean", "IIM_estimate",
                "IIM_se", "NAS_c", "PDI_c"):
        np.testing.assert_array_equal(two[col].to_numpy(), three[col].to_numpy())
    awake, deep = _row(three, "awake"), _row(three, "deep")
    # a linear-Gaussian VAR is reproduced by its spectrum-matched PDI null, so
    # the cohort's PDI excess is not above the null: PDI has no construct scale
    # (INVALID_ANCHORS) and nothing can be concluded about PDI ...
    assert awake["PDI_status"] == "UNDEFINED"
    assert awake["MPC_verdict"] == "UNDETERMINED"
    assert awake["MPC_reason"] == "INVALID_ANCHORS:PDI"
    # ... while the deep run stays excluded by IIM (veto; an undefined PDI
    # never rescues it)
    assert deep["MPC_verdict"] == "EXCLUDED"
    assert deep["MPC_reason"].endswith("ABSENT:IIM")


# --------------------------------------------------------------------------
# RAM / SRPI event nulls and bootstrap (simple known-answer estimators)
# --------------------------------------------------------------------------
def _evoked(ts, tr, onsets, node=0):
    idx = np.rint(np.asarray(onsets, dtype=float) / tr).astype(int)
    idx = idx[(idx >= 1) & (idx + 4 < ts.shape[1])]
    if idx.size < 3:
        return float("nan")
    return float(np.mean([ts[node, k + 1:k + 4].mean() - ts[node, k - 1]
                          for k in idx]))


def _fake_ram(ts, tr=None, stimulus_onsets=None, return_details=False, **_kw):
    val = _evoked(ts, tr, (stimulus_onsets or {}).get("onsets", []))
    return {"value": val, "undefined_reason": None} if return_details else val


def _fake_srpi(ts, tr=None, self_onsets=(), nonself_onsets=(), return_details=False,
               **_kw):
    val = _evoked(ts, tr, self_onsets, node=1) - _evoked(ts, tr, nonself_onsets, node=1)
    return {"value": val, "undefined_reason": None} if return_details else val


def _event_layout(tmp_path, rng):
    prep = tmp_path / "prep"
    onsets = {"s1": {}}
    for ses, planted in (("awake", True), ("deep", False)):
        x = rng.standard_normal((4, 1200))
        # jittered inter-event intervals: a rigid shift of a strictly periodic
        # train could realign it with itself (see MPC_NULL_KINDS_DEFAULT)
        stim = 3.0 + np.cumsum(rng.uniform(1.4, 2.6, 50))
        self_on, other_on = stim[0::2] + 0.7, stim[1::2] + 0.7
        if planted:
            for on in stim:
                k = int(round(on / TR))
                x[0, k + 1:k + 4] += 1.0  # event-locked response
            for on in self_on:
                k = int(round(on / TR))
                x[1, k + 1:k + 4] += 2.5  # self-specific response
        d = prep / "s1" / ses / "audio"
        d.mkdir(parents=True)
        np.save(d / "s1_run-1_toy_ts.npy", x.T)
        bundle = {"onsets": stim.tolist(), "self_onsets": self_on.tolist(),
                  "nonself_onsets": other_on.tolist()}
        onsets["s1"][ses] = (bundle, "1")
    return prep, onsets


def test_ram_and_srpi_get_event_nulls_and_bootstrap_se(tmp_path, monkeypatch):
    prep, onsets = _event_layout(tmp_path, np.random.default_rng(5))
    monkeypatch.setattr(sc, "compute_RAM", _fake_ram)
    monkeypatch.setattr(sc, "compute_SRPI", _fake_srpi)
    common = dict(tr=TR, stimulus_onsets=onsets, srpi_params=_SRPI_PARAMS,
                  mpc_metrics=("RAM", "SRPI"), compute_ci=False)
    df = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake", "deep"), null_surrogates=40,
        bootstrap_se=40, bootstrap_block_len=100, necessity_set=("RAM", "SRPI"),
        **common,
    ).set_index("session")
    assert (df["RAM_null_n"] == 40).all() and (df["SRPI_null_n"] == 40).all()
    assert (df["RAM_boot_n"] >= 38).all() and (df["SRPI_boot_n"] >= 38).all()
    for p in ("RAM", "SRPI"):
        assert (df[f"{p}_boot_n"] + df[f"{p}_boot_failed"] == 40).all()
    assert (df["MPC_null_families"] == "RAM:onset_jitter;SRPI:label_permutation").all()
    assert (df["MPC_bootstrap_block_len"] == 100).all()
    assert df.loc["awake", "MPC_verdict"] == "MPC_CONSISTENT"
    assert df.loc["awake", "RAM_status"] == df.loc["awake", "SRPI_status"] == "PRESENT"
    # 25 events: the bootstrap SE of the evoked response is about 1/sqrt(3*25)
    assert 0.03 < df.loc["awake", "RAM_se"] < 0.3
    assert df.loc["deep", "MPC_verdict"] != "MPC_CONSISTENT"
    assert abs(df.loc["deep", "RAM_c"]) < 0.5
    np.testing.assert_allclose(
        df["RAM"], np.maximum(df["RAM_estimate"] - df["RAM_null_mean"], 0.0))
    # a different base seed gives a different (but still valid) null and SE
    other = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake",), null_surrogates=40,
        null_seed=1, bootstrap_se=10, null_kinds={"SRPI": "onset_jitter"}, **common,
    ).iloc[0]
    assert other["RAM_null_mean"] != df.loc["awake", "RAM_null_mean"]
    assert other["RAM_se"] != df.loc["awake", "RAM_se"]
    assert other["MPC_null_families"] == "RAM:onset_jitter;SRPI:onset_jitter"
    assert other["MPC_null_seed"] == 1


def test_real_ram_estimator_is_calibrated_and_bootstrapped(tmp_path):
    from test_ram_ground_truth import _simulate_goal_task

    ts, tr, bundle = _simulate_goal_task(seed=0, coupled=True, n_regions=20)
    d = tmp_path / "prep" / "s1" / "awake" / "audio"
    d.mkdir(parents=True)
    np.save(d / "s1_run-1_toy_ts.npy", ts.T)
    ram = {**RAM_PARAM_PRESETS["fmri"], "quality_null_samples": 20}
    df = sc.compute_synergy_ci(
        str(tmp_path / "prep"), "toy", [0.5], sessions=("awake",), tr=tr,
        stimulus_onsets={"s1": {"awake": (bundle, "1")}}, ram_params=ram,
        mpc_metrics=("RAM",), compute_ci=False, null_surrogates=4, bootstrap_se=3,
        necessity_set=("RAM",),
    )
    row = df.iloc[0]
    # Integration check with the real estimator (not a known-answer test).
    assert np.isfinite(row["RAM_estimate"]) and row["RAM_null_n"] == 4
    assert row["RAM_null_sd"] > 0 and row["RAM_boot_n"] + 0 <= 3
    assert row["RAM"] == pytest.approx(
        max(row["RAM_estimate"] - row["RAM_null_mean"], 0.0))
    assert row["RAM_estimator"] == f"compute_RAM:feedback_magnitude@{VERSIONS['RAM']}"
    assert row["MPC_verdict"] in {v.value for v in V}


# --------------------------------------------------------------------------
# protocol-selected estimator modes, channels and bearers
# --------------------------------------------------------------------------
def _save_runs(tmp_path, runs, name="toy"):
    prep = tmp_path / "prep"
    for ses, x in runs.items():
        d = prep / "s1" / ses / "audio"
        d.mkdir(parents=True)
        np.save(d / f"s1_run-1_{name}_ts.npy", np.asarray(x).T)
    return prep


def test_nas_capacity_mode_through_the_protocol_and_the_params(tmp_path):
    x_hub, _ = _hub_network("bidir", 0, n_time=1500)
    x_none, _ = _hub_network("none", 1, n_time=1500)
    prep = _save_runs(tmp_path, {"awake": x_hub, "deep": x_none})
    proto = E.Protocol(necessity_set=("NAS",),
                       estimators={"NAS": {"mode": "capacity",
                                           "workspace_nodes": [0, 1, 2, 3]}})
    common = dict(sessions=("awake", "deep"), tr=1.0, mpc_metrics=("NAS",),
                  compute_ci=False, bootstrap_se=10)
    df = sc.compute_synergy_ci(str(prep), "toy", [0.5], protocol=proto, **common)
    awake, deep = _row(df, "awake"), _row(df, "deep")
    # the mode's own block circular-shift null runs even with null_surrogates=0
    assert (df["NAS_estimator"] == f"compute_NAS:capacity@{VERSIONS['NAS']}").all()
    assert (df["NAS_null_n"] == mm.NAS_CAPACITY_DEFAULT_SURROGATES).all()
    assert (df["MPC_null_families"] == "NAS:block_circular_shift").all()
    assert (df["MPC_protocol_hash"] == proto.hash).all()
    assert (df["NAS_boot_n"] == 10).all() and np.isfinite(df["NAS_se"]).all()
    # a hub that receives and returns is credited; no hub is excluded
    assert awake["MPC_verdict"] == "MPC_CONSISTENT"
    assert deep["MPC_verdict"] == "EXCLUDED" and deep["NAS_status"] == "ABSENT"
    np.testing.assert_allclose(
        df["NAS"], np.maximum(df["NAS_estimate"] - df["NAS_null_mean"], 0.0))
    # the same mode through nas_params gives the same evidence
    via_params = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], necessity_set=("NAS",),
        nas_params={"mode": "capacity", "workspace_nodes": [0, 1, 2, 3]}, **common)
    for col in ("NAS_estimate", "NAS_null_mean", "NAS_se", "NAS_c", "MPC_verdict"):
        assert via_params[col].tolist() == df[col].tolist()
    params_proto = E.Protocol.from_dict(via_params.attrs["mpc_evidence"]["protocol"])
    assert params_proto.estimator_options("NAS")["mode"] == "capacity"
    assert params_proto.null_families["NAS"] == "block_circular_shift"
    with pytest.raises(ValueError, match="differs"):
        sc.compute_synergy_ci(str(prep), "toy", [0.5], protocol=proto,
                              nas_params={"mode": "legacy"}, **common)
    with pytest.raises(ValueError, match="own null"):
        sc.compute_synergy_ci(str(prep), "toy", [0.5], protocol=proto,
                              null_kinds={"NAS": "circular_shift"}, **common)


def test_srpi_agency_mode_through_the_protocol(tmp_path):
    x_tag, rows_tag = _agency_task(0, tag=1.0)
    x_none, rows_none = _agency_task(1, tag=0.0)
    prep = _save_runs(tmp_path, {"awake": x_tag, "deep": x_none})
    onsets = {"s1": {
        "awake": (events_table_to_bundle(pd.DataFrame(rows_tag)), "1"),
        "deep": (events_table_to_bundle(pd.DataFrame(rows_none)), "1"),
    }}
    proto = E.Protocol(necessity_set=("SRPI",), estimators={
        "SRPI": {"mode": "agency", "agency_null_permutations": 50}})
    srpi = {**_SRPI_PARAMS, "modality": "fmri", "pre_window_sec": 2.0,
            "response_lag_sec": 2.0, "response_window_sec": 4.0}
    df = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake", "deep"), tr=1.0,
        stimulus_onsets=onsets, srpi_params=srpi, mpc_metrics=("SRPI",),
        compute_ci=False, protocol=proto, bootstrap_se=10, bootstrap_block_len=64,
    )
    assert (df["SRPI_estimator"] == f"compute_SRPI:agency@{VERSIONS['SRPI']}").all()
    assert (df["SRPI_null_n"] == 50).all()
    assert (df["MPC_null_families"] == "SRPI:yoked_label_permutation").all()
    # yoked pairs survive the block bootstrap (replays move with their event)
    assert (df["SRPI_boot_n"] == 10).all() and np.isfinite(df["SRPI_se"]).all()
    awake, deep = _row(df, "awake"), _row(df, "deep")
    assert awake["SRPI_status"] == "PRESENT"
    assert awake["MPC_verdict"] == "MPC_CONSISTENT"
    assert deep["SRPI_c"] < 0.5 and deep["MPC_verdict"] != "MPC_CONSISTENT"


def test_pdi_surrogate_excess_and_iim_cut_mode_through_the_modes(tmp_path):
    prep, _ = _layout(tmp_path)
    proto = E.Protocol(necessity_set=("PDI", "IIM"), estimators={
        "PDI": {"mode": "surrogate_excess"}, "IIM": {"cut_mode": "directional"}})
    df = _run(prep, sessions=("awake",), mpc_metrics=("PDI", "IIM"),
              compute_ci=False, protocol=proto, bootstrap_se=4)
    row = df.iloc[0]
    # PDI: the mode's own multivariate Fourier null (19 surrogates at K = 0);
    # the rest baselines are not used by this mode
    assert row["PDI_estimator"] == f"compute_PDI:surrogate_excess@{VERSIONS['PDI']}"
    assert row["PDI_null_n"] == mm.PDI_EXCESS_DEFAULT_SURROGATES
    assert row["MPC_null_families"] == "PDI:fourier"
    assert row["PDI_primary_source"] == "surrogate_excess"
    assert np.isnan(row["PDI_anchor"])
    assert row["PDI_anchor_reason"] == "not_used_by_mode:surrogate_excess"
    assert row["PDI_boot_n"] == 4 and np.isfinite(row["PDI_se"])
    assert row["PDI"] == pytest.approx(
        max(row["PDI_estimate"] - row["PDI_null_mean"], 0.0))
    # IIM: directional cuts reach compute_IIM (no null at K = 0)
    assert row["IIM_estimator"] == f"compute_IIM:directional@{VERSIONS['IIM']}"
    assert "NO_NULL_CALIBRATION:IIM" in row["MPC_reason"]
    assert row["IIM_boot_n"] == 4 and np.isfinite(row["IIM_se"])


def test_declared_ram_channels_are_computed_one_by_one(tmp_path):
    prep, onsets = _layout(tmp_path, with_events=True)
    ram = {**RAM_PARAM_PRESETS["eeg"], "quality_null_samples": 5}
    proto = E.Protocol(
        necessity_set=("RAM",),
        channels={"RAM": ("behavioural_feedback", "perturbational")},
        estimators={"RAM": {"update": "prediction_error"}},
    )
    df = _run(prep, stimulus_onsets=onsets, ram_params=ram, mpc_metrics=("RAM",),
              compute_ci=False, protocol=proto, null_surrogates=3)
    for _, row in df.iterrows():
        # no impact_channel labels in these events: the behavioural channel is
        # undefined and the perturbational channel is not implemented, so RAM
        # can never be ABSENT and the verdict is UNDETERMINED
        assert row["MPC_verdict"] == "UNDETERMINED"
        assert row["RAM_status"] == "UNDEFINED"
        assert row["RAM_channels"] == (
            "behavioural_feedback:UNDEFINED,perturbational:UNDEFINED")
        assert "NOT_IMPLEMENTED:RAM:perturbational" in row["MPC_reason"]
        assert "UNDEFINED:RAM:" in row["MPC_reason"]
        assert row["RAM_estimator"] == f"compute_RAM:prediction_error@{VERSIONS['RAM']}"
        assert row["RAM_null_n"] == 0  # undefined estimates get no null


def test_single_source_constraint_in_the_pipeline(tmp_path):
    prep, _ = _layout(tmp_path)
    proto = E.Protocol(necessity_set=("NAS", "IIM"),
                       bearer_nodes={"NAS": [0, 1, 2, 3], "IIM": [4, 5, 6, 7]})
    common = dict(mpc_metrics=("NAS", "IIM"), compute_ci=False, protocol=proto,
                  nas_params={**NAS_TOY, "tau": 0.75})
    common_run = {k: v for k, v in common.items() if k != "nas_params"}
    df = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake", "deep"), tr=TR,
        nas_params=common["nas_params"], srpi_params=_SRPI_PARAMS,
        pdi_params=_PDI_PARAMS, null_surrogates=19, **IIM_TOY, **common_run)
    awake, deep = _row(df, "awake"), _row(df, "deep")
    # the coupled ring links the two halves; independent noise does not
    assert awake["MPC_joint_dependence"] == "dependent"
    assert awake["MPC_joint_dependence_p"] <= 0.05
    assert not any(r.startswith("SOURCE_INCOHERENT")
                   for r in awake["MPC_reason"].split(";"))
    assert deep["MPC_joint_dependence"] == "SOURCE_INCOHERENT"
    assert deep["MPC_reason"].split(";")[0] == "SOURCE_INCOHERENT"
    assert deep["MPC_verdict"] == "UNDETERMINED"
    # bearer node sets reach the estimators (NAS on 4 nodes, IIM within 4..7)
    assert set(df["NAS_estimator"]) == {f"compute_NAS:legacy@{VERSIONS['NAS']}"}
    # without a null budget the dependence is untested
    df0 = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake",), tr=TR,
        nas_params=common["nas_params"], srpi_params=_SRPI_PARAMS,
        pdi_params=_PDI_PARAMS, **IIM_TOY, **common_run)
    assert df0.iloc[0]["MPC_joint_dependence"] == "untested"
    assert df0.iloc[0]["MPC_reason"].startswith("SOURCE_INCOHERENT:UNTESTED")


# --------------------------------------------------------------------------
# registry, precomputed IIM, validation
# --------------------------------------------------------------------------
def _iim_entry(**kw):
    entry = {
        "estimator": "compute_IIM:bidirectional",
        "version": VERSIONS["IIM"],
        "substrate": "bold_like_forward",
        "grain": "toy",
        # IIM's regime is the scored subsystem: 3 nodes, 2 bins
        "regime": {"T_min": 400, "nodes_min": 3, "nodes_max": 3, "bins": 2},
        "evidence": {"run_id": "bench-test", "null_false_present_rate": 0.05,
                     "recovery_slope": 1.0},
        "status": "validated",
    }
    entry.update(kw)
    return entry


def test_applicability_registry_is_applied(tmp_path):
    prep, _ = _layout(tmp_path)
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema": E.REGISTRY_SCHEMA,
                               "entries": [_iim_entry()]}))
    df = _run(prep, sessions=("awake",), mpc_metrics=("NAS", "IIM"),
              compute_ci=False, applicability_registry=str(reg), modality="fmri")
    row = df.iloc[0]
    assert row["NAS_status"] == "UNDEFINED"
    assert (f"ESTIMATOR_NOT_VALIDATED:NAS:compute_NAS:legacy@{VERSIONS['NAS']}"
            in row["MPC_reason"])
    # IIM is validated for fMRI through its BOLD-like forward-model entry (and
    # then lacks only its null calibration)
    assert "NO_NULL_CALIBRATION:IIM" in row["MPC_reason"]
    assert df.attrs["mpc_evidence"]["applicability_registry"]["entries"]
    # the regime is checked (here: T_min, and the scored IIM subsystem size)
    for regime in ({"T_min": 481}, {"nodes_min": 4}):
        reg.write_text(json.dumps({"entries": [_iim_entry(regime=regime)]}))
        row = _run(prep, sessions=("awake",), mpc_metrics=("IIM",), compute_ci=False,
                   applicability_registry=str(reg), modality="fmri",
                   necessity_set=("IIM",)).iloc[0]
        assert row["MPC_reason"] == (
            f"ESTIMATOR_NOT_VALIDATED:IIM:compute_IIM:bidirectional@{VERSIONS['IIM']}")
    # Regression (review of stream E2): with iim_max_timepoints IIM scores a
    # subsampled series (here 240 samples at 2 * TR), and that is the regime
    # the registry checks, not the 480-sample run at TR
    for regime, ok in (({"T_min": 400}, False), ({"T_min": 240}, True),
                       ({"fs_or_tr": TR}, False), ({"fs_or_tr": 2 * TR}, True)):
        reg.write_text(json.dumps({"entries": [_iim_entry(regime=regime)]}))
        row = _run(prep, sessions=("awake",), mpc_metrics=("IIM",), compute_ci=False,
                   applicability_registry=str(reg), modality="fmri",
                   necessity_set=("IIM",), iim_max_timepoints=240).iloc[0]
        assert (row["MPC_reason"] == "NO_NULL_CALIBRATION:IIM") is ok, regime
    # entries that miss the entry criteria are refused before any computation
    bad = _iim_entry(evidence={"run_id": "x", "null_false_present_rate": 0.2,
                               "recovery_slope": 1.0})
    reg.write_text(json.dumps({"entries": [bad]}))
    with pytest.raises(ValueError, match="false-PRESENT"):
        _run(prep, sessions=("awake",), mpc_metrics=("IIM",), compute_ci=False,
             applicability_registry=str(reg))


def test_precomputed_iim_without_null_fields_is_not_calibrated(tmp_path):
    prep, _ = _layout(tmp_path)
    ts_path = str(prep / "s1" / "awake" / "audio" / "s1_run-1_toy_ts.npy")
    info = {"defined": True, "undefined_reason": None, "canonical": 0.4,
            "raw": 0.4, "Psi_full": 1.0, "Psi_mip_preserved": 0.6,
            "iim_algorithm_version": "hunter"}
    df = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake",), tr=TR,
        mpc_metrics=("IIM",), compute_ci=False, null_surrogates=3,
        iim_precomputed_by_path={ts_path: info}, necessity_set=("IIM",),
    )
    row = df.iloc[0]
    assert np.isnan(row["IIM"]) and not row["IIM_defined"]
    assert row["IIM_undefined_reason"].startswith("null_calibration_unavailable")
    assert row["IIM_estimate"] == pytest.approx(0.4)
    assert row["IIM_estimator"] == "compute_IIM:bidirectional@hunter"
    assert row["MPC_reason"] == "NO_NULL_CALIBRATION:IIM"
    assert row["MPC_verdict"] == "UNDETERMINED"
    # a precomputed result with null fields and a bootstrap SE is judged
    info.update(Delta_Psi=0.4, Delta_Psi_null_mean=0.1, Delta_Psi_null_sd=0.05,
                IIM_null_n=20, IIM_null_method="circular_shift",
                canonical_calibrated=0.3, Delta_Psi_bootstrap_se=0.02,
                Delta_Psi_bootstrap_n=30)
    row = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake",), tr=TR,
        mpc_metrics=("IIM",), compute_ci=False, null_surrogates=3,
        iim_precomputed_by_path={ts_path: info}, necessity_set=("IIM",),
    ).iloc[0]
    assert row["MPC_verdict"] == "MPC_CONSISTENT" and row["IIM_c"] == 1.0
    assert row["IIM_se"] == 0.02 and row["IIM_boot_n"] == 30


def test_precomputed_iim_with_other_options_is_not_judged(tmp_path):
    """Regression (review of stream E2): a precomputed (Hunter) IIM result is
    computed without the protocol's IIM options; a result whose recorded
    cut mode or bearer differs from the declared ones is undefined
    (iim_option_mismatch), never judged under the protocol."""
    prep, _ = _layout(tmp_path)
    ts_path = str(prep / "s1" / "awake" / "audio" / "s1_run-1_toy_ts.npy")
    info = {"defined": True, "undefined_reason": None, "canonical": 0.4,
            "raw": 0.4, "Delta_Psi": 0.4, "Delta_Psi_null_mean": 0.1,
            "Delta_Psi_null_sd": 0.05, "IIM_null_n": 20,
            "IIM_null_method": "circular_shift", "canonical_calibrated": 0.3,
            "Delta_Psi_bootstrap_se": 0.02, "Delta_Psi_bootstrap_n": 30,
            "iim_algorithm_version": VERSIONS["IIM"], "cut_mode": "bidirectional",
            "bearer_nodes": None, "tpm_estimator": "node_shrinkage"}
    common = dict(sessions=("awake",), tr=TR, mpc_metrics=("IIM",), compute_ci=False,
                  null_surrogates=3, iim_precomputed_by_path={ts_path: info})

    def run(proto):
        return sc.compute_synergy_ci(str(prep), "toy", [0.5], protocol=proto,
                                     **common).iloc[0]

    ok = run(E.Protocol(necessity_set=("IIM",)))
    assert ok["MPC_verdict"] == "MPC_CONSISTENT"
    directional = run(E.Protocol(necessity_set=("IIM",),
                                 estimators={"IIM": {"cut_mode": "directional"}}))
    assert directional["MPC_verdict"] == "UNDETERMINED"
    assert directional["MPC_reason"] == (
        "UNDEFINED:IIM:iim_option_mismatch:cut_mode=directional/bidirectional")
    assert directional["IIM_estimator"] == (
        f"compute_IIM:bidirectional@{VERSIONS['IIM']}")  # what was computed
    bearer = run(E.Protocol(necessity_set=("IIM",), bearer_nodes={"IIM": [0, 1, 2]}))
    assert bearer["MPC_reason"] == "UNDEFINED:IIM:iim_option_mismatch:bearer_nodes"
    # in-process results carry the declared options and are judged
    df = _run(prep, sessions=("awake",), mpc_metrics=("IIM",), compute_ci=False,
              protocol=E.Protocol(necessity_set=("IIM",),
                                  estimators={"IIM": {"cut_mode": "directional"}},
                                  bearer_nodes={"IIM": [0, 1, 2, 3]}))
    assert "iim_option_mismatch" not in df.iloc[0]["MPC_reason"]
    assert df.iloc[0]["MPC_reason"] == "NO_NULL_CALIBRATION:IIM"


def test_bootstrap_se_needs_a_majority_of_valid_replicates():
    """Regression (review of stream E2): an SE from the few replicates on
    which the estimator is defined is withheld (NO_SAMPLING_SE) when most
    replicates fail; the failures are reported in <P>_boot_failed."""
    x = np.random.default_rng(0).standard_normal((2, 400))
    rec = sc._component_record(0.3, estimator="compute_NAS:legacy@v")
    calls = {"n": 0}

    def flaky(ts, _ev, fail_every):
        calls["n"] += 1
        if calls["n"] % fail_every:
            raise ValueError("undefined on this resample")
        return float(ts[0].mean())

    def attach(fail_every, n_boot=20):
        calls["n"] = 0
        return sc._attach_bootstrap(
            rec, lambda t, e: flaky(t, e, fail_every), x, None, n_boot, 0, None, 1.0)

    all_ok = attach(1)  # every replicate valid
    assert all_ok["boot_n"] == 20 and all_ok["boot_failed"] == 0
    assert np.isfinite(all_ok["se"])
    half = attach(2)  # every second replicate fails: 10 of 20 valid -> kept
    assert half["boot_n"] == 10 and half["boot_failed"] == 10
    assert np.isfinite(half["se"])
    most = attach(4)  # 5 of 20 valid -> withheld
    assert most["boot_n"] == 5 and most["boot_failed"] == 15
    assert np.isnan(most["se"])
    ev = E.ComponentEvidence("NAS", 0.3, 0.0, 0.1, se=most["se"], n_null=10,
                             reference=0.3)
    assert E.component_status(ev)[2] == E.REASON_NO_SAMPLING_SE
    assert np.isnan(attach(1, n_boot=1)["se"])  # a single replicate has no SD
    # precomputed IIM: the same rule on its bootstrap counts
    info = {"defined": True, "Delta_Psi": 0.4, "Delta_Psi_bootstrap_se": 0.02,
            "Delta_Psi_bootstrap_n": 4, "Delta_Psi_bootstrap_failed": 16}
    iim = sc._iim_record(info, "circular_shift")
    assert np.isnan(iim["se"]) and iim["boot_n"] == 4 and iim["boot_failed"] == 16
    info.update(Delta_Psi_bootstrap_n=16, Delta_Psi_bootstrap_failed=4)
    assert sc._iim_record(info, "circular_shift")["se"] == 0.02


def test_margin_column_describes_the_deciding_channel():
    """Regression (review of stream E2): <P>_margin is the presence margin of
    the channel that decides the principle (the channel <P>_c and
    <P>_c_lower describe), not the largest margin over the channels."""
    proto = E.Protocol(necessity_set=("RAM",),
                       channels={"RAM": ("behavioural_feedback", "covert_neural")})

    def rec(estimate, se, channel):
        r = sc._component_record(estimate, estimator="compute_RAM:x@v",
                                 null_mean=0.0, null_sd=0.1, null_n=50,
                                 null_family="onset_jitter", channel=channel)
        r["se"] = se
        return r

    # behavioural: inconclusive (c = 0.5, se_c = 0.5: margin -0.57);
    # covert: absent (c = 0.0, se_c = 0.01: margin -0.27, the larger one)
    run = {"records": {"RAM": [rec(0.5, 0.5, "behavioural_feedback"),
                               rec(0.0, 0.01, "covert_neural")]},
           "bearer_id": "b", "substrate": None, "grain": None, "regime": {},
           "joint": None, "n_node_sets": 1}
    refs = {("RAM", ch): {"reference": 1.0, "reference_se": float("nan"),
                          "scale": "excess"}
            for ch in ("behavioural_feedback", "covert_neural")}
    cols = sc._mpc_run_columns(run, proto, None, refs, 50, 0, 20, None)
    assert cols["RAM_status"] == "UNDEFINED" and cols["MPC_reason"] == (
        "INCONCLUSIVE:RAM")
    assert cols["RAM_c"] == pytest.approx(0.5)
    assert cols["RAM_margin"] == pytest.approx(cols["RAM_c_lower"] - 0.25)
    assert cols["RAM_margin"] < -0.5
    assert cols["RAM_channels"] == (
        "behavioural_feedback:UNDEFINED,covert_neural:ABSENT")


def test_evidence_options_are_validated(tmp_path):
    prep, _ = _layout(tmp_path)
    kw = dict(sessions=("awake",), mpc_metrics=("NAS",), compute_ci=False)
    with pytest.raises(ValueError, match="null_surrogates"):
        _run(prep, null_surrogates=-1, **kw)
    with pytest.raises(ValueError, match="bootstrap_se"):
        _run(prep, bootstrap_se=-1, **kw)
    with pytest.raises(ValueError, match="bootstrap_block_len"):
        _run(prep, bootstrap_block_len=0, **kw)
    with pytest.raises(ValueError, match="necessity set"):
        _run(prep, necessity_set=("NAS", "PHI"), **kw)
    with pytest.raises(ValueError, match="null kind"):
        _run(prep, null_kinds={"NAS": "label_permutation"}, **kw)
    with pytest.raises(ValueError, match="unknown surrogate"):
        _run(prep, null_kinds={"RAM": "bogus"}, **kw)
    proto = E.Protocol(necessity_set=("NAS",))
    with pytest.raises(ValueError, match="necessity"):
        _run(prep, protocol=proto, necessity_set=("IIM",), **kw)
    with pytest.raises(ValueError, match="bearer_nodes"):
        _run(prep, protocol=E.Protocol(bearer_nodes={"NAS": [0, 1]}),
             nas_params={**NAS_TOY, "bearer_nodes": [0, 2]}, **kw)
    with pytest.raises(ValueError, match="null_kinds"):
        _run(prep, protocol=E.Protocol(null_families={"NAS": "phase_randomize"}),
             null_kinds={"NAS": "circular_shift"}, **kw)
    # an unselected principle in N is MISSING, not silently ignored
    df = _run(prep, necessity_set=("NAS", "RAM"), **kw)
    assert "MISSING:RAM" in df.iloc[0]["MPC_reason"]
    # a protocol file is accepted and its hash recorded
    path = tmp_path / "protocol.json"
    proto.to_json(path)
    df = _run(prep, protocol=str(path), **kw)
    assert (df["MPC_protocol_hash"] == proto.hash).all()
    assert df.attrs["mpc_evidence"]["protocol_hash"] == proto.hash


def test_empty_layout_has_the_evidence_columns(tmp_path):
    df = sc.compute_synergy_ci(
        str(tmp_path), "toy", [0.5], sessions=("awake",), tr=TR,
        nas_params=NAS_TOY, srpi_params=_SRPI_PARAMS,
    )
    assert df.empty and set(sc.MPC_EVIDENCE_COLUMNS) <= set(df.columns)


# --------------------------------------------------------------------------
# compute_CI deprecation alias
# --------------------------------------------------------------------------
def test_compute_ci_warns_once_and_is_unchanged(monkeypatch):
    monkeypatch.setattr(mm, "_COMPUTE_CI_DEPRECATION_WARNED", False)
    args = (0.5, 0.2, 0.3, 0.4, 0.6)
    with pytest.warns(DeprecationWarning, match="evidence.mpc_verdict"):
        val = mm.compute_CI(*args)
    assert val == mm._compute_ci_legacy(*args)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        again = mm.compute_CI(*args, return_details=True)
    assert again["value"] == val
    assert np.isnan(mm.compute_CI(*args[:4], float("nan")))


# --------------------------------------------------------------------------
# run_s_ci and run_pipeline pass-through
# --------------------------------------------------------------------------
class _Stop(Exception):
    pass


def test_run_s_ci_forwards_the_evidence_options(tmp_path, monkeypatch):
    from impact_pipeline import run_synergy_ci

    captured = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(run_synergy_ci, "compute_synergy_ci", fake)
    (tmp_path / "s1").mkdir()
    proto = E.Protocol(necessity_set=("NAS",)).to_dict()
    with pytest.raises(_Stop):
        run_synergy_ci.run_s_ci(
            tmp_path, None, tmp_path, "toy", ("awake",), [0.5], [0.5],
            mpc_metrics=("NAS",), tr=TR, null_surrogates=7,
            necessity_set=("NAS",), applicability_registry="reg.json", null_seed=3,
            protocol=proto, bootstrap_se=11, bootstrap_block_len=50,
        )
    assert captured["null_surrogates"] == 7 and captured["null_seed"] == 3
    assert captured["necessity_set"] == ("NAS",)
    assert captured["applicability_registry"] == "reg.json"
    assert captured["protocol"] == proto
    assert captured["bootstrap_se"] == 11 and captured["bootstrap_block_len"] == 50


def test_cli_exposes_the_evidence_flags():
    helptext = subprocess.run(
        [sys.executable, str(REPO / "run_pipeline.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    ).stdout
    for flag in ("--null-surrogates", "--necessity-set", "--applicability-registry",
                 "--protocol", "--bootstrap-se", "--bootstrap-block-len"):
        assert flag in helptext


def test_mpc_evidence_options_validation(tmp_path):
    opts = run_pipeline._mpc_evidence_options(5, "iim, nas", None)
    assert opts == {"null_surrogates": 5, "necessity_set": ["NAS", "IIM"],
                    "applicability_registry": None, "protocol": None,
                    "bootstrap_se": 0, "bootstrap_block_len": None}
    with pytest.raises(ValueError):
        run_pipeline._mpc_evidence_options(-1)
    with pytest.raises(ValueError, match="bootstrap-se"):
        run_pipeline._mpc_evidence_options(0, bootstrap_se=-2)
    with pytest.raises(FileNotFoundError):
        run_pipeline._mpc_evidence_options(0, None, tmp_path / "missing.json")
    reg = tmp_path / "reg.json"
    reg.write_text("[]")
    assert run_pipeline._mpc_evidence_options(0, None, reg)[
        "applicability_registry"] == str(reg.resolve())
    # registry entries are checked against the entry criteria up front
    reg.write_text(json.dumps([{"principle": "NAS", "estimator": "compute_NAS"}]))
    with pytest.raises(ValueError, match="entry criteria"):
        run_pipeline._mpc_evidence_options(0, None, reg)
    # a protocol is carried as its canonical dict; N comes from it
    proto = E.Protocol(necessity_set=("NAS", "IIM"), alpha=0.025)
    path = tmp_path / "protocol.json"
    proto.to_json(path)
    opts = run_pipeline._mpc_evidence_options(3, None, None, protocol=path,
                                              bootstrap_se=9, bootstrap_block_len=40)
    assert opts["protocol"] == proto.to_dict() and opts["necessity_set"] is None
    assert opts["bootstrap_se"] == 9 and opts["bootstrap_block_len"] == 40
    assert run_pipeline._mpc_protocol_provenance(opts, path) == {
        "source": str(path.resolve()), "hash": proto.hash}
    with pytest.raises(ValueError, match="necessity"):
        run_pipeline._mpc_evidence_options(0, "RAM", None, protocol=path)
    with pytest.raises(FileNotFoundError):
        run_pipeline._mpc_evidence_options(0, protocol=tmp_path / "nope.json")


def test_metric_subset_keeps_the_evidence_columns():
    df = pd.DataFrame([{"subject": "1", "session": "awake", "theta": 0.5, "S": 0.1,
                        "NAS": 0.2, "MPC_verdict": "UNDETERMINED",
                        "MPC_reason": "NO_NULL_CALIBRATION:NAS", "NAS_status":
                        "UNDEFINED", "NAS_c": 0.3, "MPC_protocol_hash": "h",
                        "unrelated": 1}])
    mean = df[["subject", "session", "S"]]
    out, _ = run_pipeline._apply_metric_subset(df, mean, mpc_metrics=["NAS"],
                                               compute_ci=False)
    assert {"MPC_verdict", "MPC_reason", "NAS_status", "NAS_c",
            "MPC_protocol_hash"} <= set(out.columns)
    assert "unrelated" not in out.columns


def test_main_forwards_the_evidence_options_to_step2(tmp_path, monkeypatch):
    from impact_pipeline import run_synergy_ci

    bids, out = _tiny_bids_and_prep(tmp_path)
    captured = {}

    def fake_run_s_ci(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(run_synergy_ci, "run_s_ci", fake_run_s_ci)
    with pytest.raises(_Stop):
        run_pipeline.main(
            str(out), dataset_id="ds003171", bids_root_override=str(bids),
            mpc_metrics=["NAS"], compute_ci=False, null_surrogates=12,
            necessity_set="NAS", bootstrap_se=4,
        )
    assert captured["null_surrogates"] == 12
    assert captured["necessity_set"] == ["NAS"]
    assert captured["applicability_registry"] is None
    assert captured["protocol"] is None and captured["bootstrap_se"] == 4


def test_evidence_options_reach_the_hunter_finalize_stage(tmp_path, monkeypatch):
    from impact_pipeline import run_synergy_ci

    bids, out = _tiny_bids_and_prep(tmp_path)
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"entries": []}))
    proto = E.Protocol(necessity_set=("IIM",))
    proto_path = tmp_path / "protocol.json"
    proto.to_json(proto_path)
    common = dict(dataset_id="ds003171", bids_root_override=str(bids),
                  execution_mode="hunter", mpc_metrics=["IIM"], compute_ci=False,
                  iim_max_nodes_override=3, iim_n_parts_override=2)
    run_pipeline.main(str(out), hunter_stage="build-campaign", null_surrogates=9,
                      applicability_registry=str(reg), protocol=str(proto_path),
                      bootstrap_se=6, **common)
    campaign = out / "cache" / "hunter_iim_campaign"
    ctx = json.loads((campaign / "campaign_manifest.json").read_text())[
        "step2_context"]
    expected = {"null_surrogates": 9, "necessity_set": None,
                "applicability_registry": str(reg.resolve()),
                "protocol": proto.to_dict(), "bootstrap_se": 6,
                "bootstrap_block_len": None}
    assert ctx["mpc_evidence"] == expected
    prov = json.loads((out / "cache" / "provenance_manifest.json").read_text())
    assert prov["parameters"]["mpc_evidence"] == expected
    assert prov["parameters"]["mpc_protocol"]["hash"] == proto.hash

    captured = {}

    def fake_run_s_ci(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(run_synergy_ci, "run_s_ci", fake_run_s_ci)
    monkeypatch.setattr(run_pipeline, "collect_iim_results_by_path", lambda d: {})
    with pytest.raises(_Stop):
        # the finalize job's command line does not repeat the evidence flags
        run_pipeline.main(str(out), hunter_stage="finalize-pipeline",
                          hunter_campaign_dir=str(campaign), **common)
    assert {k: captured[k] for k in expected} == expected


# --------------------------------------------------------------------------
# MPC degree known answer
# --------------------------------------------------------------------------
def test_assemble_mpc_degree_known_answer():
    # Row 0 is MPC_CONSISTENT with c_NAS = 0.5 and c_IIM = 1.5 (capped at 1):
    # geometric mean 0.5 ** 0.5; the degree is never computed for other rows.
    df = pd.DataFrame([
        {"MPC_verdict": "MPC_CONSISTENT", "MPC_necessity_set": "NAS,IIM",
         "NAS_c": 0.5, "IIM_c": 1.5, "PDI_c": 0.01},
        {"MPC_verdict": "EXCLUDED", "MPC_necessity_set": "NAS,IIM",
         "NAS_c": 0.9, "IIM_c": 0.0, "PDI_c": 0.9},
        {"MPC_verdict": "UNDETERMINED", "MPC_necessity_set": "NAS,IIM",
         "NAS_c": 0.9, "IIM_c": 0.9, "PDI_c": 0.9},
    ])
    out = sc.assemble_mpc_degree(df)
    assert out.loc[0, "MPC_degree"] == pytest.approx(0.5 ** 0.5)
    assert out.loc[1:, "MPC_degree"].isna().all()
    # weights restricted to N; arithmetic mean on request
    out_w = sc.assemble_mpc_degree(df, weights={"NAS": 3.0, "IIM": 1.0, "PDI": 5.0},
                                   p=1.0)
    assert out_w.loc[0, "MPC_degree"] == pytest.approx(0.75 * 0.5 + 0.25 * 1.0)
    no_weight = sc.assemble_mpc_degree(df, weights={"PDI": 1.0})
    assert np.isnan(no_weight.loc[0, "MPC_degree"])


def test_ci_reference_is_not_read_for_the_evidence(tmp_path):
    # The evidence anchors come from the protocol; a CI reference that is not
    # used (compute_ci=False) is never read.
    prep, _ = _layout(tmp_path)
    df = _run(prep, sessions=("awake",), mpc_metrics=("NAS",), compute_ci=False,
              ci_reference=str(tmp_path / "missing_reference.json"))
    assert df["MPC_degree"].isna().all()
    assert (df["MPC_verdict"] == "UNDETERMINED").all()


def test_legacy_pdi_fallback_is_on_the_floored_excess_scale(tmp_path):
    rng = np.random.default_rng(4)
    d = tmp_path / "prep" / "s1" / "awake" / "audio"
    d.mkdir(parents=True)
    np.save(d / "s1_run-1_toy_ts.npy", _var(rng, 0.0, 0.0).T)
    row = sc.compute_synergy_ci(
        str(tmp_path / "prep"), "toy", [0.5], sessions=("awake",), tr=TR,
        pdi_params={**_PDI_PARAMS, "clip_negative": False}, mpc_metrics=("PDI",),
        compute_ci=False, null_surrogates=6, necessity_set=("PDI",),
    ).iloc[0]
    assert row["PDI_primary_source"] == "legacy_surrogate"
    assert row["PDI_estimator"] == (
        f"compute_PDI:legacy-legacy_surrogate@{VERSIONS['PDI']}")
    assert row["PDI_null_n"] == 6
    excess = row["PDI_estimate"] - row["PDI_null_mean"]
    assert row["PDI"] == pytest.approx(max(excess, 0.0), abs=1e-12)
    assert row["PDI"] >= 0.0
