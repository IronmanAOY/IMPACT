"""
Dashboard parity with the pipeline (stream I3).

- The mixed-source combined index is assembled by synergy_ci.assemble_ci (no
  NAS*S, no 1e-12 reference floor, NaN semantics) and, because different
  datasets are different bearers, is labelled exploratory with an
  UNDETERMINED (BEARER_MISMATCH) MPC verdict.
- MPC_verdict / MPC_reason are carried through when the step-2 tables have them.
- 'aal90' is the 116-label AAL image: shown and requested as 'aal116'.
- The IIM checkpoint mirrors follow synergy_ci's file naming and compute_IIM's
  D6 checkpoint signature.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

COMPONENTS = ("RAM", "PDI", "NAS", "IIM", "SRPI")


@pytest.fixture()
def dash_state(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("IMPACT_SYNTH_ROOT", raising=False)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="dsA",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    return dash, dash.DashboardState(cfg)


def _write_step2(out_dir: Path, rows: list[dict]) -> None:
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "cache" / "step2_df.csv", index=False)


def _context(anchor_out, aux_out, metric_map, mapping_aux=None, **extra):
    ctx = {
        "plan": [
            {"dataset_id": "dsA", "out_dir": str(anchor_out), "data_origin": "real"},
            {"dataset_id": "dsB", "out_dir": str(aux_out), "data_origin": "dummy"},
        ],
        "anchor_dataset_id": "dsA",
        "primary_dataset_id": "dsA",
        "metric_dataset_map": metric_map,
        "subject_mapping_effective": {
            "dsA": {s: s for s in ("01", "02", "03", "04")},
            "dsB": mapping_aux or {s: s for s in ("01", "02", "03", "04")},
        },
    }
    ctx.update(extra)
    return ctx


def _random_tables(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    anchor_out = tmp_path / "outputs" / "scratch" / "dsA"
    aux_out = tmp_path / "test_objects" / "runs" / "dsB"
    rows_a, rows_b = [], []
    for sub in ("01", "02", "03", "04"):
        for ses in ("awake", "deep"):
            vals = {k: float(rng.uniform(0.1, 2.0)) for k in COMPONENTS}
            if sub == "03" and ses == "deep":
                vals["NAS"] = np.nan  # undefined component -> CI NaN
            if sub == "04" and ses == "awake":
                vals["RAM"] = 0.0  # measured zero -> CI 0, defined
            for theta in (0.3, 0.5):
                base = {
                    "subject": sub,
                    "session": ses,
                    "theta": theta,
                    "S": rng.random(),
                }
                rows_a.append({**base, **{k: vals[k] for k in ("RAM", "PDI", "NAS")}})
                rows_b.append(
                    {
                        **base,
                        "IIM": vals["IIM"],
                        "SRPI": vals["SRPI"],
                        "IIM_defined": True,
                    }
                )
    _write_step2(anchor_out, rows_a)
    _write_step2(aux_out, rows_b)
    return anchor_out, aux_out


MIXED_MAP = {"RAM": "dsA", "PDI": "dsA", "NAS": "dsA", "IIM": "dsB", "SRPI": "dsB"}


def test_mixed_source_ci_equals_synergy_ci_assemble_ci(dash_state, tmp_path):
    from impact_pipeline.synergy_ci import assemble_ci

    dash, state = dash_state
    anchor_out, aux_out = _random_tables(tmp_path)
    manifest = state._compose_mixed_source_ci(_context(anchor_out, aux_out, MIXED_MAP))
    mixed = pd.read_csv(manifest["step2_df_path"], dtype={"subject": str})
    # Known answer: the pipeline's own assembly on the same component table.
    expected, refs = assemble_ci(mixed[["subject", "session", *COMPONENTS]])
    np.testing.assert_allclose(mixed["CI"], expected["CI"], rtol=1e-12, equal_nan=True)
    assert list(mixed["CI_defined"]) == list(expected["CI_defined"])
    assert list(mixed["CI_missing"].fillna("")) == list(expected["CI_missing"])
    assert manifest["ci_reference_means"] == pytest.approx(refs)
    assert manifest["ci_assembly"] == "impact_pipeline.synergy_ci.assemble_ci"
    by = {(r.subject, r.session): r for r in mixed.itertuples()}
    assert math.isnan(by[("03", "deep")].CI) and by[("03", "deep")].CI_missing == "NAS"
    assert by[("04", "awake")].CI == 0.0 and bool(by[("04", "awake")].CI_defined)
    # No HypergraphSynergy multiplier: S differs per theta and must not matter.
    subjects = ("01", "02", "03", "04")
    ref = {
        k: np.mean([getattr(by[(s, "awake")], k) for s in subjects])
        for k in COMPONENTS
    }
    row = by[("01", "deep")]
    geo = np.exp(np.mean([np.log(getattr(row, k) / ref[k]) for k in COMPONENTS]))
    assert row.CI == pytest.approx(geo, rel=1e-9)


def test_external_reference_without_floor(dash_state, tmp_path):
    dash, state = dash_state
    anchor_out, aux_out = _random_tables(tmp_path, seed=1)
    refs = {"RAM": 1.0, "PDI": 1.0, "NAS": 1e-15, "IIM": 1.0}  # SRPI missing
    manifest = state._compose_mixed_source_ci(
        _context(anchor_out, aux_out, MIXED_MAP, ci_reference_means=refs)
    )
    assert manifest["ci_reference"] == "external"
    assert manifest["ci_reference_invalid_components"] == ["SRPI"]
    mixed = pd.read_csv(manifest["step2_df_path"], dtype={"subject": str})
    assert mixed["CI"].isna().all()
    ok = mixed[mixed["NAS"].notna()]
    assert all("SRPI_reference" in m for m in ok["CI_missing"])
    # A tiny positive reference is used as is (a floor would have changed it).
    assert manifest["ci_reference_means"]["NAS"] == pytest.approx(1e-15)


def test_cross_dataset_composition_is_exploratory_and_undetermined(
    dash_state, tmp_path
):
    dash, state = dash_state
    anchor_out, aux_out = _random_tables(tmp_path, seed=2)
    manifest = state._compose_mixed_source_ci(_context(anchor_out, aux_out, MIXED_MAP))
    assert manifest["exploratory"] is True
    assert manifest["ci_composition"] == "cross_dataset_exploratory"
    assert manifest["source_datasets"] == ["dsA", "dsB"]
    assert "EXPLORATORY" in manifest["interpretation_note"]
    assert "different bearers" in manifest["interpretation_note"]
    mixed = pd.read_csv(manifest["step2_df_path"], dtype={"subject": str})
    assert set(mixed["MPC_verdict"]) == {"UNDETERMINED"}
    assert set(mixed["MPC_reason"]) == {"BEARER_MISMATCH"}
    assert set(mixed["CI_composition"]) == {"cross_dataset_exploratory"}
    assert manifest["mpc_verdict_counts"] == {"UNDETERMINED": len(mixed)}
    df_mean = pd.read_csv(manifest["step2_df_mean_path"], dtype={"subject": str})
    assert {"MPC_verdict", "MPC_reason", "CI_composition"} <= set(df_mean.columns)
    summary = {r["anchor_subject"]: r for r in manifest["subject_metric_summary"]}
    assert summary["01"]["MPC_verdict"] == "awake: UNDETERMINED; deep: UNDETERMINED"
    assert summary["01"]["MPC_reason"] == (
        "awake: BEARER_MISMATCH; deep: BEARER_MISMATCH"
    )


def test_single_source_carries_mpc_verdict_columns(dash_state, tmp_path):
    dash, state = dash_state
    anchor_out = tmp_path / "outputs" / "scratch" / "dsA"
    aux_out = tmp_path / "test_objects" / "runs" / "dsB"
    comps = {k: 1.0 for k in COMPONENTS}
    rows = []
    for sub, ses, runs in (
        ("01", "awake", [("ATTRIBUTED", "")]),
        ("01", "deep", [("NOT_ATTRIBUTED", "ABSENT:NAS")]),
        # Two runs of one session disagree: no determinate verdict holds.
        ("02", "awake", [("ATTRIBUTED", ""), ("UNDETERMINED", "INCONCLUSIVE:IIM")]),
        ("02", "deep", [("UNDETERMINED", "NO_NULL_CALIBRATION:RAM")]),
    ):
        for verdict, reason in runs:
            for theta in (0.3, 0.5):  # repeated per theta
                rows.append(
                    {"subject": sub, "session": ses, "theta": theta, **comps,
                     "MPC_verdict": verdict, "MPC_reason": reason}
                )
    _write_step2(anchor_out, rows)
    _write_step2(aux_out, [{"subject": "01", "session": "awake", **comps}])
    anchor_only = {k: "dsA" for k in COMPONENTS}
    manifest = state._compose_mixed_source_ci(
        _context(anchor_out, aux_out, anchor_only)
    )
    assert manifest["exploratory"] is False
    mixed = pd.read_csv(manifest["step2_df_path"], dtype={"subject": str})
    by = {(r.subject, r.session): r for r in mixed.itertuples()}
    assert by[("01", "awake")].MPC_verdict == "ATTRIBUTED"
    assert by[("01", "deep")].MPC_verdict == "NOT_ATTRIBUTED"
    assert by[("01", "deep")].MPC_reason == "ABSENT:NAS"
    assert by[("02", "awake")].MPC_verdict == "UNDETERMINED"
    assert by[("02", "awake")].MPC_reason == (
        "RUN_VERDICTS_DISAGREE:ATTRIBUTED/UNDETERMINED"
    )
    assert set(mixed["CI_composition"]) == {"single_source"}


def test_single_source_verdict_comes_from_the_source_dataset(dash_state, tmp_path):
    """All components from a non-anchor dataset: its verdict, mapped subject."""
    dash, state = dash_state
    anchor_out = tmp_path / "outputs" / "scratch" / "dsA"
    aux_out = tmp_path / "test_objects" / "runs" / "dsB"
    comps = {k: 1.0 for k in COMPONENTS}
    sessions = ("awake", "deep")
    # The anchor's own verdicts must not be used: none of its values are.
    _write_step2(
        anchor_out,
        [
            {"subject": s, "session": ses, **comps, "MPC_verdict": "ATTRIBUTED"}
            for s in ("01", "02")
            for ses in sessions
        ],
    )
    verdicts_b = {"11": ("NOT_ATTRIBUTED", "ABSENT:NAS"), "12": ("ATTRIBUTED", "")}
    _write_step2(
        aux_out,
        [
            {"subject": s, "session": ses, **comps,
             "MPC_verdict": verdicts_b[s][0], "MPC_reason": verdicts_b[s][1]}
            for s in ("11", "12")
            for ses in sessions
        ],
    )
    manifest = state._compose_mixed_source_ci(
        _context(
            anchor_out,
            aux_out,
            {k: "dsB" for k in COMPONENTS},
            mapping_aux={"01": "12", "02": "11"},
        )
    )
    assert manifest["exploratory"] is False and manifest["source_datasets"] == ["dsB"]
    mixed = pd.read_csv(manifest["step2_df_path"], dtype={"subject": str})
    by = {(r.subject, r.session): r for r in mixed.itertuples()}
    for ses in sessions:
        assert by[("01", ses)].MPC_verdict == "ATTRIBUTED"  # dsB subject 12
        assert by[("02", ses)].MPC_verdict == "NOT_ATTRIBUTED"  # dsB subject 11
        assert by[("02", ses)].MPC_reason == "ABSENT:NAS"
    assert manifest["mpc_verdict_counts"] == {"ATTRIBUTED": 2, "NOT_ATTRIBUTED": 2}


def test_merge_run_verdicts_known_answers():
    import scripts.live_dashboard as dash

    assert dash._merge_run_verdicts([]) == (None, None)
    assert dash._merge_run_verdicts([np.nan, None, ""]) == (None, None)
    assert dash._merge_run_verdicts(["ATTRIBUTED"] * 3, ["", np.nan, ""]) == (
        "ATTRIBUTED",
        None,
    )
    assert dash._merge_run_verdicts(
        ["UNDETERMINED", "UNDETERMINED"], ["MISSING:RAM", "INCONCLUSIVE:IIM"]
    ) == ("UNDETERMINED", "MISSING:RAM;INCONCLUSIVE:IIM")
    assert dash._merge_run_verdicts(["NOT_ATTRIBUTED", "ATTRIBUTED"]) == (
        "UNDETERMINED",
        "RUN_VERDICTS_DISAGREE:ATTRIBUTED/NOT_ATTRIBUTED",
    )
    # A run without a verdict is unknown: it never lets the other runs decide
    # (dropping it would turn a disagreement into a determinate verdict).
    assert dash._merge_run_verdicts(["ATTRIBUTED", np.nan, "ATTRIBUTED"]) == (
        "UNDETERMINED",
        "RUN_VERDICT_MISSING",
    )
    assert dash._merge_run_verdicts(["ATTRIBUTED", pd.NA]) == (
        "UNDETERMINED",
        "RUN_VERDICT_MISSING",
    )
    assert dash._merge_run_verdicts(["NOT_ATTRIBUTED", None], ["ABSENT:NAS", None]) == (
        "UNDETERMINED",
        "RUN_VERDICT_MISSING",
    )
    assert dash._merge_run_verdicts(
        ["UNDETERMINED", ""], ["INCONCLUSIVE:IIM", ""]
    ) == ("UNDETERMINED", "INCONCLUSIVE:IIM;RUN_VERDICT_MISSING")
    assert dash._merge_run_verdicts(["ATTRIBUTED", "NOT_ATTRIBUTED", None]) == (
        "UNDETERMINED",
        "RUN_VERDICTS_DISAGREE:ATTRIBUTED/NOT_ATTRIBUTED;RUN_VERDICT_MISSING",
    )
    # Missingness safety over random run sets: hiding run verdicts never
    # yields a determinate verdict that the full set does not have.
    rng = np.random.default_rng(0)
    codes = np.array(["ATTRIBUTED", "NOT_ATTRIBUTED", "UNDETERMINED"])
    for _ in range(2000):
        full = list(codes[rng.integers(0, 3, size=int(rng.integers(1, 5)))])
        hidden = [None if rng.random() < 0.4 else v for v in full]
        v_full, _ = dash._merge_run_verdicts(full)
        v_hidden, _ = dash._merge_run_verdicts(hidden)
        if v_hidden not in (None, "UNDETERMINED"):
            assert v_hidden == v_full and None not in hidden


def test_aal90_is_the_116_node_aal116_atlas(dash_state, tmp_path):
    dash, state = dash_state
    assert dash._canonical_atlas_name("aal90") == "aal116"
    assert dash._canonical_atlas_name("AAL90") == "aal116"
    assert dash._canonical_atlas_name("schaefer400") == "schaefer400"
    assert dash._atlas_size_from_name("aal90") == 116
    assert dash._atlas_size_from_name("aal116") == 116
    assert dash._atlas_size_from_name("shen268") == 268
    root = tmp_path / "data" / "scratch" / "ds003171"
    (root / "sub-01" / "func").mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "x"}))
    cmd, resolved = state._build_run_command(
        {
            "dataset_id": "ds003171",
            "bids_root": str(root),
            "atlas": "aal90",
            "run_preprocessing": True,
        }
    )
    assert cmd[cmd.index("--atlas") + 1] == "aal116"
    assert "aal90" not in cmd


def test_run_inputs_show_legacy_aal90_files_as_aal116(dash_state, tmp_path):
    dash, state = dash_state
    d = state.cfg.out_dir / "preprocessed" / "01" / "awake" / "audio"
    d.mkdir(parents=True)
    np.save(d / "01_run-1_aal90_ts.npy", np.zeros((10, 116)))
    # A legacy run requested with --atlas aal90: files keep their on-disk name.
    out = state._collect_run_inputs(
        {
            "subjects": ["01"],
            "sessions": ["awake"],
            "condition": "audio",
            "atlas": "aal90",
        }
    )
    assert out["by_atlas"] == {"aal116": {"count": 1, "size": 116}}
    assert out["inputs"][0]["atlas"] == "aal116"
    assert out["inputs"][0]["atlas_size"] == 116


def test_checkpoint_filename_mirror_matches_synergy_ci(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash
    from impact_pipeline import synergy_ci

    ts_dir = tmp_path / "prep" / "01" / "awake" / "audio"
    ts_path = ts_dir / "01_run-1_schaefer400_ts.npy"
    ts_path.parent.mkdir(parents=True)
    np.save(ts_path, np.random.default_rng(0).normal(size=(40, 5)))
    seen = {}

    def fake_compute_iim(ts, **kwargs):
        seen["checkpoint_path"] = kwargs.get("checkpoint_path")
        return {"value": 0.0}

    monkeypatch.setattr(synergy_ci, "compute_IIM", fake_compute_iim)
    params = dict(
        iim_bins=2,
        iim_lag_trs=1,
        iim_n_parts=None,
        iim_max_timepoints=30,
        iim_max_nodes=4,
        iim_max_mechanism_size=2,
        iim_max_purview_size=3,
    )
    synergy_ci._iim_worker_from_path(
        str(ts_path),
        **params,
        iim_checkpoint_dir=str(tmp_path / "ck"),
        iim_resume_checkpoint=True,
        iim_checkpoint_every_cuts=1,
        iim_progress_log_every_cuts=0,
        iim_phase1_parallel_workers=2,
        iim_phase1_chunk_size=8,
        iim_phase1_shared_memory=False,
        hardware_target="cpu",
    )
    mirrored = dash._checkpoint_filename_for_ts(
        ts_path,
        bins=params["iim_bins"],
        lag_trs=params["iim_lag_trs"],
        n_parts=params["iim_n_parts"],
        max_timepoints=params["iim_max_timepoints"],
        max_nodes=params["iim_max_nodes"],
        max_mechanism_size=params["iim_max_mechanism_size"],
        max_purview_size=params["iim_max_purview_size"],
        phase1_parallel_workers=2,
        phase1_chunk_size=8,
        phase1_shared_memory=False,
    )
    assert Path(seen["checkpoint_path"]).name == mirrored


def test_checkpoint_signature_mirror_follows_compute_iim_d6(dash_state, tmp_path):
    from impact_pipeline.mpc_metrics import IIM_ALGORITHM_VERSION, compute_IIM

    dash, state = dash_state
    ck_dir = tmp_path / "ck"
    ck = ck_dir / "01_run-1_schaefer400_0123456789abcdef.iim_checkpoint.json"
    ck_dir.mkdir()
    ts = np.random.default_rng(3).normal(size=(4, 120))
    compute_IIM(
        ts,
        bins=2,
        lag_trs=1,
        max_mechanism_size=2,
        max_purview_size=2,
        return_details=True,
        checkpoint_path=str(ck),
    )
    payload = json.loads(ck.read_text())
    sig = payload["signature"]
    # Every mirrored field exists in compute_IIM's real checkpoint signature.
    assert set(dash.IIM_CHECKPOINT_SIGNATURE_FIELDS) <= set(sig)
    assert sig["iim_algorithm_version"] == IIM_ALGORITHM_VERSION
    assert dash._checkpoint_signature_status(sig) == "current"
    state._ck_dir = ck_dir
    rows = state._read_checkpoints()
    assert rows[0]["signature_status"] == "current" and rows[0]["resumable"]
    assert rows[0]["sig_tpm_alpha"] == sig["tpm_alpha"]
    assert rows[0]["progress_estimate"] == pytest.approx(1.0)
    # Written by another algorithm version (or before D6): not resumable, so
    # its progress does not count.
    for stale_sig, status in (
        ({**sig, "iim_algorithm_version": "iim-v3-old"}, "stale_algorithm_version"),
        ({k: v for k, v in sig.items() if k != "iim_algorithm_version"},
         "legacy_unversioned"),
    ):
        ck.write_text(json.dumps({**payload, "signature": stale_sig}))
        rows = state._read_checkpoints()
        assert rows[0]["signature_status"] == status
        assert rows[0]["resumable"] is False
        assert rows[0]["progress_estimate"] == 0.0
