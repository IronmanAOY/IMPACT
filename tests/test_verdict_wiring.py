"""
MPC verdict wiring: compute_synergy_ci evidence columns, the honest K=0
default, determinate verdicts on planted-structure toys at K>0, run_s_ci /
run_pipeline pass-through (local and Hunter finalize) and the compute_CI
deprecation alias.
"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_pipeline
from impact_pipeline import evidence as E
from impact_pipeline import mpc_metrics as mm
from impact_pipeline import synergy_ci as sc
from impact_pipeline.event_parsing import events_table_to_bundle, read_events_table
from impact_pipeline.run_synergy_ci import RAM_PARAM_PRESETS
from test_hunter_pbs import _tiny_bids_and_prep
from test_synergy_ci import _NAS_PARAMS, _PDI_PARAMS, _SRPI_PARAMS

REPO = Path(__file__).resolve().parents[1]
TR = 0.1
N_NODES, N_TIME = 8, 480
NAS_TOY = {**_NAS_PARAMS, "tau": 0.5}  # >= 3 broadcast nodes of 8
IIM_TOY = dict(iim_bins=2, iim_max_nodes=3, iim_enable_parallel=False,
               iim_use_shared_memory=False)


def _var(rng, coupling, common):
    """Stable VAR(1) ring (spectral radius <= 0.9) plus a shared AR(1) drive."""
    a = np.eye(N_NODES) * 0.3
    for i in range(N_NODES):
        a[(i + 1) % N_NODES, i] += coupling
    x = np.zeros((N_NODES, N_TIME))
    drive = np.zeros(N_TIME)
    for k in range(1, N_TIME):
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


def _layout(tmp_path, with_events=False, seed=0):
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
        x = _var(rng, coupling, common=2.5 * coupling)
        if with_events:
            ev = _events_table(rng, N_TIME, planted_x=x if coupling else None)
            fn = bids / f"sub-s1_task-audio{ses}_run-1_events.tsv"
            ev.to_csv(fn, sep="\t", index=False)
            onsets["s1"][ses] = (events_table_to_bundle(read_events_table(fn)), "1")
        d = prep / "s1" / ses / "audio"
        d.mkdir(parents=True)
        np.save(d / "s1_run-1_toy_ts.npy", x.T)
        r = prep / "s1" / ses / "rest"
        r.mkdir(parents=True)
        np.save(r / "s1_run-2_toy_ts.npy", rng.standard_normal((N_TIME, N_NODES)))
    return prep, (onsets if with_events else None)


def _run(prep, **kw):
    kw.setdefault("sessions", ("awake", "deep"))
    return sc.compute_synergy_ci(
        str(prep), "toy", [0.3, 0.6], tr=TR, nas_params=NAS_TOY,
        srpi_params=_SRPI_PARAMS, pdi_params=_PDI_PARAMS, **IIM_TOY, **kw,
    )


# --------------------------------------------------------------------------
# end to end: K = 0 (honest default)
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
    assert (df["MPC_null_families"] == "").all()
    assert (df["MPC_necessity_set"] == "RAM,PDI,NAS,IIM,SRPI").all()
    for _, row in df.iterrows():
        reasons = row["MPC_reason"].split(";")
        for p in E.PRINCIPLES:
            assert row[f"{p}_status"] == "UNDEFINED"
            assert row[f"{p}_null_n"] == 0 and np.isnan(row[f"{p}_null_mean"])
            if np.isfinite(row[f"{p}_estimate"]):
                assert f"NO_NULL_CALIBRATION:{p}" in reasons
            else:
                assert any(r.startswith(f"UNDEFINED:{p}:") for r in reasons)
        assert E.verdict_from_reasons(reasons) == E.Verdict.UNDETERMINED
    # most components are measured on the toy; the metric columns keep their
    # uncalibrated meaning (NAS value, PDI = max(raw anchor PDI, 0))
    awake = df[df["session"] == "awake"].iloc[0]
    for p in ("PDI", "NAS", "IIM", "SRPI"):
        assert np.isfinite(awake[f"{p}_estimate"]), p
    np.testing.assert_allclose(df["NAS"], df["NAS_estimate"])
    np.testing.assert_allclose(df["PDI"], np.maximum(df["PDI_estimate"], 0.0))
    np.testing.assert_allclose(df["PDI_anchor"], df["PDI_estimate"])
    # evidence is per run: identical across theta rows
    for _, grp in df.groupby("session"):
        assert grp["MPC_reason"].nunique() == 1


# --------------------------------------------------------------------------
# end to end: K > 0 on a planted-structure toy
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def planted_runs(tmp_path_factory):
    prep, _ = _layout(tmp_path_factory.mktemp("planted"))
    common = dict(mpc_metrics=("PDI", "NAS", "IIM"), compute_ci=False,
                  null_surrogates=20, null_seed=0)
    two = _run(prep, necessity_set=("NAS", "IIM"), **common)
    three = _run(prep, necessity_set="PDI,NAS,IIM", **common)
    return two, three


def test_k_positive_gives_determinate_verdicts_on_planted_structure(planted_runs):
    df, _ = planted_runs
    awake = df[df["session"] == "awake"].iloc[0]
    deep = df[df["session"] == "deep"].iloc[0]
    # coupled network with a shared broadcast drive: NAS and IIM above null
    assert awake["NAS_status"] == "PRESENT" and awake["IIM_status"] == "PRESENT"
    assert awake["MPC_verdict"] == "ATTRIBUTED" and awake["MPC_reason"] == ""
    # independent noise: credibly null
    assert deep["MPC_verdict"] == "NOT_ATTRIBUTED"
    assert "ABSENT:" in deep["MPC_reason"]
    # degree only when ATTRIBUTED; the awake row is its own (cohort) reference
    assert awake["MPC_degree"] == pytest.approx(1.0)
    assert np.isnan(deep["MPC_degree"])
    # the metric columns now hold the calibrated (excess over null) values
    for p in ("NAS", "IIM"):
        assert (df[f"{p}_null_n"] == 20).all()
        np.testing.assert_allclose(
            df[p], np.maximum(df[f"{p}_estimate"] - df[f"{p}_null_mean"], 0.0),
            rtol=1e-9, atol=1e-12,
        )
    np.testing.assert_allclose(
        df["NAS_margin"], (df["NAS_estimate"] - df["NAS_null_mean"]) / df["NAS_null_sd"]
    )
    assert (df["MPC_null_families"] ==
            "PDI:phase_randomize;NAS:circular_shift;IIM:circular_shift").all()


def test_k_positive_veto_and_theory_dependence_of_the_necessity_set(planted_runs):
    two, three = planted_runs
    # the null families (and seeds) do not depend on N: identical evidence
    for col in ("NAS_null_mean", "IIM_null_mean", "PDI_null_mean", "IIM_estimate"):
        np.testing.assert_array_equal(two[col].to_numpy(), three[col].to_numpy())
    awake = three[three["session"] == "awake"].iloc[0]
    # a linear-Gaussian VAR is reproduced by its spectrum-matched PDI null:
    # PDI is credibly absent and vetoes the attribution (no compensation by
    # the strong NAS/IIM evidence)
    assert awake["PDI_status"] == "ABSENT"
    assert awake["MPC_verdict"] == "NOT_ATTRIBUTED"
    assert awake["MPC_reason"] == "ABSENT:PDI"
    assert np.isnan(awake["MPC_degree"])


# --------------------------------------------------------------------------
# RAM / SRPI event nulls (simple known-answer estimators patched in)
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


def test_ram_and_srpi_get_event_nulls(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
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
                x[1, k + 1:k + 4] += 1.0  # self-specific response
        d = prep / "s1" / ses / "audio"
        d.mkdir(parents=True)
        np.save(d / "s1_run-1_toy_ts.npy", x.T)
        bundle = {"onsets": stim.tolist(), "self_onsets": self_on.tolist(),
                  "nonself_onsets": other_on.tolist()}
        onsets["s1"][ses] = (bundle, "1")
    monkeypatch.setattr(sc, "compute_RAM", _fake_ram)
    monkeypatch.setattr(sc, "compute_SRPI", _fake_srpi)
    df = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake", "deep"), tr=TR,
        stimulus_onsets=onsets, srpi_params=_SRPI_PARAMS,
        mpc_metrics=("RAM", "SRPI"), compute_ci=False,
        null_surrogates=40, necessity_set=("RAM", "SRPI"),
    ).set_index("session")
    assert (df["RAM_null_n"] == 40).all() and (df["SRPI_null_n"] == 40).all()
    assert (df["MPC_null_families"] == "RAM:onset_jitter;SRPI:label_permutation").all()
    assert df.loc["awake", "MPC_verdict"] == "ATTRIBUTED"
    assert df.loc["awake", "RAM_status"] == df.loc["awake", "SRPI_status"] == "PRESENT"
    assert df.loc["awake", "RAM_margin"] > 1.645
    assert df.loc["awake", "SRPI_margin"] > 1.645
    assert df.loc["deep", "MPC_verdict"] != "ATTRIBUTED"
    assert abs(df.loc["deep", "RAM_margin"]) < 3
    np.testing.assert_allclose(
        df["RAM"], np.maximum(df["RAM_estimate"] - df["RAM_null_mean"], 0.0))
    # a different base seed gives a different (but still valid) null
    other = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake",), tr=TR, stimulus_onsets=onsets,
        srpi_params=_SRPI_PARAMS, mpc_metrics=("RAM", "SRPI"), compute_ci=False,
        null_surrogates=40, null_seed=1, null_kinds={"SRPI": "onset_jitter"},
    ).iloc[0]
    assert other["RAM_null_mean"] != df.loc["awake", "RAM_null_mean"]
    assert other["MPC_null_families"] == "RAM:onset_jitter;SRPI:onset_jitter"
    assert other["MPC_null_seed"] == 1


def test_real_ram_estimator_is_calibrated_by_the_onset_null(tmp_path):
    from test_ram_ground_truth import _simulate_goal_task

    ts, tr, bundle = _simulate_goal_task(seed=0, coupled=True, n_regions=20)
    d = tmp_path / "prep" / "s1" / "awake" / "audio"
    d.mkdir(parents=True)
    np.save(d / "s1_run-1_toy_ts.npy", ts.T)
    ram = {**RAM_PARAM_PRESETS["fmri"], "quality_null_samples": 20}
    df = sc.compute_synergy_ci(
        str(tmp_path / "prep"), "toy", [0.5], sessions=("awake",), tr=tr,
        stimulus_onsets={"s1": {"awake": (bundle, "1")}}, ram_params=ram,
        mpc_metrics=("RAM",), compute_ci=False, null_surrogates=4,
        necessity_set=("RAM",),
    )
    row = df.iloc[0]
    # Integration check with the real estimator (not a known-answer test):
    # every shifted-train surrogate is scored and the null is not degenerate.
    assert np.isfinite(row["RAM_estimate"]) and row["RAM_null_n"] == 4
    assert np.isfinite(row["RAM_margin"]) and row["RAM_null_sd"] > 0
    assert row["RAM"] == pytest.approx(
        max(row["RAM_estimate"] - row["RAM_null_mean"], 0.0))
    assert row["MPC_verdict"] in {v.value for v in E.Verdict}


# --------------------------------------------------------------------------
# registry, precomputed IIM, validation
# --------------------------------------------------------------------------
def test_applicability_registry_is_applied(tmp_path):
    prep, _ = _layout(tmp_path)
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"entries": [
        {"principle": "IIM", "estimator": "compute_IIM:*", "grain": "toy",
         "regime": {"n_time": {"min": 400}}},
    ]}))
    df = _run(prep, sessions=("awake",), mpc_metrics=("NAS", "IIM"),
              compute_ci=False, applicability_registry=str(reg))
    row = df.iloc[0]
    assert row["NAS_status"] == "UNDEFINED"
    assert "ESTIMATOR_NOT_VALIDATED:NAS:compute_NAS" in row["MPC_reason"]
    # IIM is validated (and then lacks only its null calibration)
    assert "NO_NULL_CALIBRATION:IIM" in row["MPC_reason"]
    assert df.attrs["mpc_evidence"]["applicability_registry"]["entries"]


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
    assert row["MPC_reason"] == "NO_NULL_CALIBRATION:IIM"
    assert row["MPC_verdict"] == "UNDETERMINED"


def test_evidence_options_are_validated(tmp_path):
    prep, _ = _layout(tmp_path)
    kw = dict(sessions=("awake",), mpc_metrics=("NAS",), compute_ci=False)
    with pytest.raises(ValueError, match="null_surrogates"):
        _run(prep, null_surrogates=-1, **kw)
    with pytest.raises(ValueError, match="necessity set"):
        _run(prep, necessity_set=("NAS", "PHI"), **kw)
    with pytest.raises(ValueError, match="null kind"):
        _run(prep, null_kinds={"NAS": "label_permutation"}, **kw)
    with pytest.raises(ValueError, match="unknown surrogate"):
        _run(prep, null_kinds={"RAM": "bogus"}, **kw)
    # an unselected principle in N is MISSING, not silently ignored
    df = _run(prep, necessity_set=("NAS", "RAM"), **kw)
    assert "MISSING:RAM" in df.iloc[0]["MPC_reason"]


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
    with pytest.raises(_Stop):
        run_synergy_ci.run_s_ci(
            tmp_path, None, tmp_path, "toy", ("awake",), [0.5], [0.5],
            mpc_metrics=("NAS",), tr=TR, null_surrogates=7,
            necessity_set=("NAS",), applicability_registry="reg.json", null_seed=3,
        )
    assert captured["null_surrogates"] == 7 and captured["null_seed"] == 3
    assert captured["necessity_set"] == ("NAS",)
    assert captured["applicability_registry"] == "reg.json"


def test_cli_exposes_the_evidence_flags():
    helptext = subprocess.run(
        [sys.executable, str(REPO / "run_pipeline.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    ).stdout
    for flag in ("--null-surrogates", "--necessity-set", "--applicability-registry"):
        assert flag in helptext


def test_mpc_evidence_options_validation(tmp_path):
    opts = run_pipeline._mpc_evidence_options(5, "iim, nas", None)
    assert opts == {"null_surrogates": 5, "necessity_set": ["NAS", "IIM"],
                    "applicability_registry": None}
    with pytest.raises(ValueError):
        run_pipeline._mpc_evidence_options(-1)
    with pytest.raises(FileNotFoundError):
        run_pipeline._mpc_evidence_options(0, None, tmp_path / "missing.json")
    reg = tmp_path / "reg.json"
    reg.write_text("[]")
    assert run_pipeline._mpc_evidence_options(0, None, reg)[
        "applicability_registry"] == str(reg.resolve())


def test_metric_subset_keeps_the_evidence_columns():
    df = pd.DataFrame([{"subject": "1", "session": "awake", "theta": 0.5, "S": 0.1,
                        "NAS": 0.2, "MPC_verdict": "UNDETERMINED",
                        "MPC_reason": "NO_NULL_CALIBRATION:NAS", "NAS_status":
                        "UNDEFINED", "unrelated": 1}])
    mean = df[["subject", "session", "S"]]
    out, _ = run_pipeline._apply_metric_subset(df, mean, mpc_metrics=["NAS"],
                                               compute_ci=False)
    assert {"MPC_verdict", "MPC_reason", "NAS_status"} <= set(out.columns)
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
            necessity_set="NAS",
        )
    assert captured["null_surrogates"] == 12
    assert captured["necessity_set"] == ["NAS"]
    assert captured["applicability_registry"] is None


def test_evidence_options_reach_the_hunter_finalize_stage(tmp_path, monkeypatch):
    from impact_pipeline import run_synergy_ci

    bids, out = _tiny_bids_and_prep(tmp_path)
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"entries": []}))
    common = dict(dataset_id="ds003171", bids_root_override=str(bids),
                  execution_mode="hunter", mpc_metrics=["IIM"], compute_ci=False,
                  iim_max_nodes_override=3, iim_n_parts_override=2)
    run_pipeline.main(str(out), hunter_stage="build-campaign", null_surrogates=9,
                      necessity_set=["IIM"], applicability_registry=str(reg),
                      **common)
    campaign = out / "cache" / "hunter_iim_campaign"
    ctx = json.loads((campaign / "campaign_manifest.json").read_text())[
        "step2_context"]
    expected = {"null_surrogates": 9, "necessity_set": ["IIM"],
                "applicability_registry": str(reg.resolve())}
    assert ctx["mpc_evidence"] == expected
    prov = json.loads((out / "cache" / "provenance_manifest.json").read_text())
    assert prov["parameters"]["mpc_evidence"] == expected

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
# MPC degree known answer (review regression)
# --------------------------------------------------------------------------
def test_assemble_mpc_degree_known_answer():
    # Two awake subjects; calibrated metric columns hold the excess over the
    # null (the D3 reference is their cohort high-state mean): NAS ref =
    # mean(2, 6) = 4, IIM ref = mean(4, 4) = 4. Row 0 is ATTRIBUTED with
    # c_NAS = (3 - 1) / 4 = 0.5 and c_IIM = (5 - 1) / 4 = 1 -> geometric 0.5**0.5.
    df = pd.DataFrame([
        {"subject": "a", "session": "awake", "MPC_verdict": "ATTRIBUTED",
         "MPC_necessity_set": "NAS,IIM", "NAS": 2.0, "NAS_estimate": 3.0,
         "NAS_null_mean": 1.0, "IIM": 4.0, "IIM_estimate": 5.0, "IIM_null_mean": 1.0},
        {"subject": "b", "session": "awake", "MPC_verdict": "UNDETERMINED",
         "MPC_necessity_set": "NAS,IIM", "NAS": 6.0, "NAS_estimate": 7.0,
         "NAS_null_mean": 1.0, "IIM": 4.0, "IIM_estimate": 6.0, "IIM_null_mean": 2.0},
    ])
    out, refs = sc.assemble_mpc_degree(df)
    assert refs["NAS"] == pytest.approx(4.0) and refs["IIM"] == pytest.approx(4.0)
    assert out.loc[0, "MPC_degree"] == pytest.approx(0.5 ** 0.5)
    assert np.isnan(out.loc[1, "MPC_degree"])
    # weights restricted to N; arithmetic mean on request
    out_w, _ = sc.assemble_mpc_degree(df, weights={"NAS": 3.0, "IIM": 1.0}, p=1.0)
    assert out_w.loc[0, "MPC_degree"] == pytest.approx(0.75 * 0.5 + 0.25 * 1.0)


def test_degree_reference_is_not_read_without_attributed_rows(tmp_path):
    # K=0 (all UNDETERMINED) with compute_ci=False must not start reading the
    # CI reference: before the evidence layer such calls never touched it.
    prep, _ = _layout(tmp_path)
    df = _run(prep, sessions=("awake",), mpc_metrics=("NAS",), compute_ci=False,
              ci_reference=str(tmp_path / "missing_reference.json"))
    assert df["MPC_degree"].isna().all()
    assert (df["MPC_verdict"] == "UNDETERMINED").all()


def test_legacy_pdi_fallback_is_on_the_floored_excess_scale(tmp_path):
    # No rest runs -> legacy surrogate baseline. With clip_negative=False the
    # estimator's PDI_calibrated is the signed excess; the metric column must
    # still be the excess over the null floored at 0 (as for every component).
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
    assert row["PDI_null_n"] == 6
    excess = row["PDI_estimate"] - row["PDI_null_mean"]
    assert row["PDI"] == pytest.approx(max(excess, 0.0), abs=1e-12)
    assert row["PDI"] >= 0.0
