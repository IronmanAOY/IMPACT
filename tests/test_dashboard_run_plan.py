import json
import math
import os
import signal
import sys
import threading
import time

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_minimal_bids_root(root: Path, name: str) -> None:
    (root / "sub-01" / "func").mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": name}),
        encoding="utf-8",
    )


def test_preview_run_plan_groups_metrics_by_dataset(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds_primary",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    real_root = tmp_path / "data" / "managed" / "ds_primary"
    dummy_root = tmp_path / "test_objects" / "datasets" / "dummy_aux"
    _write_minimal_bids_root(real_root, "Primary")
    _write_minimal_bids_root(dummy_root, "Dummy")

    state._dataset_registry = {
        "ds_primary": {
            "bids_root": str(real_root),
            "data_origin": "real",
            "modality_profile": "fmri",
        },
        "dummy_aux": {
            "bids_root": str(dummy_root),
            "data_origin": "dummy",
            "modality_profile": "fmri",
        },
    }

    plan = state.preview_run_plan(
        {
            "selected_datasets": [
                {
                    "dataset_id": "ds_primary",
                    "bids_root": str(real_root),
                    "out_dir": str(tmp_path / "outputs" / "scratch"),
                    "data_origin": "real",
                    "modality_profile": "fmri",
                },
                {
                    "dataset_id": "dummy_aux",
                    "bids_root": str(dummy_root),
                    "out_dir": str(tmp_path / "outputs" / "scratch"),
                    "data_origin": "dummy",
                    "modality_profile": "fmri",
                },
            ],
            "primary_dataset_id": "ds_primary",
            "mpc_metrics": ["RAM", "PDI", "NAS", "IIM", "SRPI"],
            "metric_dataset_map": {
                "RAM": "ds_primary",
                "PDI": "ds_primary",
                "NAS": "ds_primary",
                "IIM": "dummy_aux",
                "SRPI": "dummy_aux",
            },
            "sessions": ["awake", "deep"],
            "condition": "audio",
            "atlas": "schaefer400",
            "execution_mode": "local",
            "hardware_target": "cpu",
            "run_preprocessing": True,
            "run_fmriprep": False,
            "run_replication": False,
            "reuse_step2": False,
            "no_ci": False,
        }
    )

    assert plan["ci_requested"] is True
    assert plan["ci_allowed"] is True
    assert plan["ci_mode"] == "mixed_source"
    assert "scientifically fragile" in str(plan["ci_problem_note"])
    assert plan["subject_mapping_auto"]["dummy_aux"]["01"] == "01"
    assert len(plan["plan"]) == 2
    assert plan["plan"][0]["dataset_id"] == "ds_primary"
    assert plan["plan"][0]["hardware_target"] == "cpu"
    assert plan["plan"][0]["metrics"] == ["RAM", "PDI", "NAS"]
    assert plan["plan"][0]["compute_ci"] is False
    assert plan["plan"][1]["dataset_id"] == "dummy_aux"
    assert plan["plan"][1]["metrics"] == ["IIM", "SRPI"]
    assert plan["plan"][1]["compute_ci"] is False


def test_preview_run_plan_preserves_manual_subject_mapping(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds_primary",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    real_root = tmp_path / "data" / "managed" / "ds_primary"
    dummy_root = tmp_path / "test_objects" / "datasets" / "dummy_aux"
    _write_minimal_bids_root(real_root, "Primary")
    _write_minimal_bids_root(dummy_root, "Dummy")

    state._dataset_registry = {
        "ds_primary": {
            "bids_root": str(real_root),
            "data_origin": "real",
            "modality_profile": "fmri",
        },
        "dummy_aux": {
            "bids_root": str(dummy_root),
            "data_origin": "dummy",
            "modality_profile": "fmri",
        },
    }

    plan = state.preview_run_plan(
        {
            "selected_datasets": [
                {
                    "dataset_id": "ds_primary",
                    "bids_root": str(real_root),
                    "out_dir": str(tmp_path / "outputs" / "scratch"),
                    "data_origin": "real",
                    "modality_profile": "fmri",
                },
                {
                    "dataset_id": "dummy_aux",
                    "bids_root": str(dummy_root),
                    "out_dir": str(tmp_path / "outputs" / "scratch"),
                    "data_origin": "dummy",
                    "modality_profile": "fmri",
                },
            ],
            "primary_dataset_id": "ds_primary",
            "mpc_metrics": ["RAM", "PDI", "NAS", "IIM", "SRPI"],
            "metric_dataset_map": {
                "RAM": "ds_primary",
                "PDI": "ds_primary",
                "NAS": "ds_primary",
                "IIM": "dummy_aux",
                "SRPI": "dummy_aux",
            },
            "subject_mapping": {
                "dummy_aux": {
                    "01": None,
                }
            },
            "sessions": ["awake", "deep"],
            "condition": "audio",
            "atlas": "schaefer400",
            "execution_mode": "local",
            "no_ci": False,
        }
    )

    assert plan["subject_mapping_auto"]["dummy_aux"]["01"] == "01"
    assert plan["subject_mapping_effective"]["dummy_aux"]["01"] is None


def test_compose_mixed_source_ci_records_subject_metric_summary(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds_primary",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    primary_out = tmp_path / "outputs" / "scratch" / "ds_primary"
    aux_out = tmp_path / "test_objects" / "runs" / "dummy_aux"
    for out_dir, ram_shift in ((primary_out, 0.0), (aux_out, 0.1)):
        cache_dir = out_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "subject": "01",
                    "session": "awake",
                    "theta": 0.5,
                    "S": 1.0,
                    "RAM": 1.0 + ram_shift,
                    "PDI": 1.1 + ram_shift,
                    "NAS": 1.2 + ram_shift,
                    "IIM": 1.3 + ram_shift,
                    "SRPI": 1.4 + ram_shift,
                    "IIM_defined": True,
                },
                {
                    "subject": "01",
                    "session": "deep",
                    "theta": 0.5,
                    "S": 0.8,
                    "RAM": 0.9 + ram_shift,
                    "PDI": 1.0 + ram_shift,
                    "NAS": 1.1 + ram_shift,
                    "IIM": 1.2 + ram_shift,
                    "SRPI": 1.3 + ram_shift,
                    "IIM_defined": True,
                },
            ]
        ).to_csv(cache_dir / "step2_df.csv", index=False)

    manifest = state._compose_mixed_source_ci(
        {
            "plan": [
                {
                    "dataset_id": "ds_primary",
                    "out_dir": str(primary_out),
                    "data_origin": "real",
                },
                {
                    "dataset_id": "dummy_aux",
                    "out_dir": str(aux_out),
                    "data_origin": "dummy",
                },
            ],
            "primary_dataset_id": "ds_primary",
            "anchor_dataset_id": "ds_primary",
            "metric_dataset_map": {
                "RAM": "ds_primary",
                "PDI": "ds_primary",
                "NAS": "ds_primary",
                "IIM": "dummy_aux",
                "SRPI": "dummy_aux",
            },
            "subject_mapping_effective": {
                "ds_primary": {"01": "01"},
                "dummy_aux": {"01": "01"},
            },
            "ci_problem_note": "Mixed-source CI is scientifically fragile.",
        }
    )

    summary = manifest["subject_metric_summary"]
    assert len(summary) == 1
    row = summary[0]
    assert row["anchor_subject"] == "01"
    assert row["RAM_source_dataset_id"] == "ds_primary"
    assert row["RAM_source_subject"] == "01"
    assert row["IIM_source_dataset_id"] == "dummy_aux"
    assert row["IIM_source_origin"] == "dummy"
    assert row["IIM_source_subject"] == "01"
    assert Path(manifest["step2_df_path"]).exists()
    assert Path(manifest["subject_metric_summary_path"]).exists()
    manifest_path = Path(manifest["out_dir"]) / "cache" / "mixed_source_manifest.json"
    assert manifest_path.exists()


def test_dataset_library_exposes_report_catalog_metadata(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds006623",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    root = tmp_path / "data" / "managed" / "ds006623"
    _write_minimal_bids_root(root, "Mirror")
    state._dataset_registry = {
        "ds006623": {
            "bids_root": str(root),
            "data_origin": "real",
            "modality_profile": "fmri",
        }
    }

    lib = {row["dataset_id"]: row for row in state.dataset_library()}
    rec = lib["ds006623"]
    assert rec["catalog_role"] == "state_switch_anchor_fmri"
    assert rec["pipeline_ready"] is False
    assert rec["target_metrics"] == ["PDI", "NAS", "IIM"]


def test_dataset_library_prefers_catalog_annex_root_without_alias_entry(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds005620",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    plain_root = tmp_path / "data" / "scratch" / "ds005620"
    annex_root = tmp_path / "data" / "scratch" / "ds005620_annex"
    _write_minimal_bids_root(plain_root, "Plain")
    _write_minimal_bids_root(annex_root, "Annex")

    lib = {row["dataset_id"]: row for row in state.dataset_library()}
    assert "ds005620_annex" not in lib
    assert lib["ds005620"]["bids_root"] == str(annex_root.resolve())
    assert state._detect_dataset_root_for_id("ds005620") == annex_root.resolve()


def test_guardrails_block_preprocessing_for_catalog_only_dataset(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds006623",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    resolved = {
        "dataset_id": "ds006623",
        "data_origin": "real",
        "out_dir": str(tmp_path / "outputs" / "scratch"),
        "run_preprocessing": True,
        "run_fmriprep": False,
        "no_ci": False,
        "mpc_metrics": ["PDI", "NAS", "IIM"],
    }

    try:
        state._apply_run_guardrails(resolved)
    except ValueError as exc:
        assert "dataset-specific preprocessing/session mappings" in str(exc)
    else:
        raise AssertionError("Expected preprocessing guardrail to reject catalog-only dataset.")


# ---------------------------------------------------------------------------
# Regression tests: run plans, datasets, mixed-source CI and run control.
# ---------------------------------------------------------------------------


def _state(
    tmp_path, monkeypatch, dataset_id="ds003171", data_origin="real", history_points=120
):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("IMPACT_SYNTH_ROOT", raising=False)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id=dataset_id,
        data_origin=data_origin,
        subject_filter=None,
        refresh_sec=2.0,
        history_points=history_points,
    )
    return dash, dash.DashboardState(cfg)


def test_library_out_dirs_do_not_follow_the_active_dataset(tmp_path, monkeypatch):
    """Regression: activating ds005620 gave ds003171 ds005620's folder."""
    dash, state = _state(tmp_path, monkeypatch)
    base = (tmp_path / "outputs" / "scratch").resolve()
    r1 = tmp_path / "data" / "scratch" / "ds003171"
    r2 = tmp_path / "data" / "scratch" / "ds005620"
    _write_minimal_bids_root(r1, "A")
    _write_minimal_bids_root(r2, "B")

    state.register_dataset({"dataset_id": "ds005620", "bids_root": str(r2)})
    assert state.cfg.dataset_id == "ds005620"
    lib = {row["library_key"]: row for row in state.dataset_library()}
    assert lib["ds003171"]["out_dir"] == str(base)
    assert lib["ds005620"]["out_dir"] == str(base / "ds005620")

    # A payload for another dataset never inherits the active effective folder.
    resolved = state._resolve_selection_from_payload({"dataset_id": "ds003171"})
    assert Path(resolved["out_dir"]) == base
    assert Path(resolved["bids_root"]) == r1.resolve()

    plan = state.preview_run_plan(
        {
            "selected_datasets": [
                {"dataset_id": "ds003171", "data_origin": "real"},
                {"dataset_id": "ds005620", "data_origin": "real"},
            ],
            "primary_dataset_id": "ds003171",
            "mpc_metrics": ["RAM", "PDI"],
            "metric_dataset_map": {"RAM": "ds003171", "PDI": "ds005620"},
            "run_preprocessing": True,
        }
    )
    outs = {
        item["dataset_id"]: item["support"]["effective_out_dir"]
        for item in plan["plan"]
    }
    assert outs["ds003171"] != outs["ds005620"]
    assert not any("same output directory" in b for b in plan["plan_blockers"])


def test_plan_blocks_two_datasets_writing_to_one_folder(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    r1 = tmp_path / "data" / "scratch" / "ds003171"
    r2 = tmp_path / "data" / "scratch" / "ds005620"
    _write_minimal_bids_root(r1, "A")
    _write_minimal_bids_root(r2, "B")
    # ds003171 has no per-dataset suffix, so this explicit folder collides.
    shared = str(tmp_path / "outputs" / "scratch" / "ds005620")
    plan = state.preview_run_plan(
        {
            "selected_datasets": [
                {
                    "dataset_id": "ds003171",
                    "data_origin": "real",
                    "out_dir": shared,
                    "bids_root": str(r1),
                },
                {
                    "dataset_id": "ds005620",
                    "data_origin": "real",
                    "out_dir": shared,
                    "bids_root": str(r2),
                },
            ],
            "primary_dataset_id": "ds003171",
            "mpc_metrics": ["RAM", "PDI"],
            "metric_dataset_map": {"RAM": "ds003171", "PDI": "ds005620"},
            "run_preprocessing": True,
        }
    )
    assert plan["plan_ready"] is False
    assert any("same output directory" in b for b in plan["plan_blockers"])


def test_multi_dataset_preview_keeps_monitoring_paths(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    r1 = tmp_path / "data" / "scratch" / "ds003171"
    r2 = tmp_path / "data" / "scratch" / "ds005620"
    _write_minimal_bids_root(r1, "A")
    _write_minimal_bids_root(r2, "B")
    ck_before, docs_before = state._ck_dir, list(state._doc_files)
    registry_before = dict(state._dataset_registry)
    state.preview_run_plan(
        {
            "selected_datasets": [
                {"dataset_id": "ds003171", "bids_root": str(r1), "data_origin": "real"},
                {"dataset_id": "ds005620", "bids_root": str(r2), "data_origin": "real"},
            ],
            "primary_dataset_id": "ds003171",
            "run_preprocessing": True,
        }
    )
    assert state._ck_dir == ck_before
    assert state._doc_files == docs_before
    assert state.cfg.dataset_id == "ds003171"
    assert (
        state._dataset_registry == registry_before
    )  # reviews do not register datasets


def test_synthetic_datasets_are_discovered_without_shadowing_real(
    tmp_path, monkeypatch
):
    dash, state = _state(tmp_path, monkeypatch)
    real = tmp_path / "data" / "scratch" / "ds003171"
    _write_minimal_bids_root(real, "Real")
    nested = tmp_path / "test_objects" / "datasets" / "real_derived_synth_completed"
    for ds in ("ds002547", "ds003171", "ds005620"):
        _write_minimal_bids_root(nested / ds, f"Synth {ds}")
    lib = {row["library_key"]: row for row in state.dataset_library()}
    assert "real_derived_synth_completed" not in {
        row["dataset_id"] for row in lib.values()
    }
    assert lib["ds003171"]["data_origin"] == "real"
    assert lib["ds003171"]["bids_root"] == str(real.resolve())
    for ds in ("ds002547", "ds003171", "ds005620"):
        rec = lib[f"{ds}@synthetic"]
        assert rec["data_origin"] == "dummy"
        assert rec["bids_root"] == str((nested / ds).resolve())
        assert "test_objects" in rec["out_dir"]

    # A flat synthetic copy with the same ID still does not replace the real entry.
    _write_minimal_bids_root(
        tmp_path / "test_objects" / "datasets" / "ds003171", "Flat synth"
    )
    lib = {row["library_key"]: row for row in state.dataset_library()}
    assert lib["ds003171"]["bids_root"] == str(real.resolve())
    assert lib["ds003171@synthetic"]["bids_root"].endswith(
        "test_objects/datasets/ds003171"
    )

    # Real origin never resolves to synthetic data, and vice versa.
    assert state._detect_dataset_root_for_id("ds002547", "real") is None
    assert (
        state._detect_dataset_root_for_id("ds002547", "dummy")
        == (nested / "ds002547").resolve()
    )

    # One plan cannot use both copies of the same dataset ID.
    with pytest.raises(ValueError, match="real and the synthetic copy"):
        state._dataset_selection_list(
            {
                "selected_datasets": [
                    {"dataset_id": "ds003171", "data_origin": "real"},
                    {"dataset_id": "ds003171", "data_origin": "dummy"},
                ]
            }
        )


def test_synthetic_archive_root_from_environment_is_discovered(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    synth_root = tmp_path / "external_ssd"
    _write_minimal_bids_root(
        synth_root
        / "test_objects"
        / "datasets"
        / "real_derived_synth_completed"
        / "ds005620",
        "S",
    )
    monkeypatch.setenv("IMPACT_SYNTH_ROOT", str(synth_root))
    lib = {row["library_key"]: row for row in state.dataset_library()}
    assert lib["ds005620@synthetic"]["data_origin"] == "dummy"
    assert lib["ds005620@synthetic"]["source"] == "synthetic_archive"


def test_register_refuses_to_label_test_objects_data_as_real(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    root = tmp_path / "test_objects" / "datasets" / "dsx"
    _write_minimal_bids_root(root, "Synth")
    with pytest.raises(ValueError, match="synthetic"):
        state.register_dataset(
            {"dataset_id": "dsx", "bids_root": str(root), "data_origin": "real"}
        )
    res = state.register_dataset(
        {"dataset_id": "dsx", "bids_root": str(root), "data_origin": "dummy"}
    )
    assert res["library_key"] == "dsx@synthetic"


def test_legacy_registry_keys_are_rekeyed_and_unsafe_ids_dropped(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    root = tmp_path / "data" / "managed" / "dsok"
    _write_minimal_bids_root(root, "ok")
    state._dataset_registry = {
        "dsok": {"bids_root": str(root), "data_origin": "dummy"},
        "..": {"bids_root": str(tmp_path / "data"), "data_origin": "real"},
    }
    lib = {row["library_key"]: row for row in state.dataset_library()}
    assert "dsok@synthetic" in lib
    assert ".." not in {row["dataset_id"] for row in lib.values()}


def test_subject_mapping_auto_pairs_identical_ids_only(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    ra = tmp_path / "a"
    rb = tmp_path / "b"
    for s in ("01", "02"):
        (ra / f"sub-{s}").mkdir(parents=True)
    for s in ("02", "77"):
        (rb / f"sub-{s}").mkdir(parents=True)
    reviews = {
        "dsA": {"inventory": state._build_dataset_inventory_for_root(ra)},
        "dsB": {"inventory": state._build_dataset_inventory_for_root(rb)},
    }
    mapping = state._subject_mapping_auto(
        reviews_by_dataset=reviews, anchor_dataset_id="dsA"
    )
    assert mapping["dsA"] == {"01": "01", "02": "02"}
    assert mapping["dsB"] == {"01": None, "02": "02"}  # never 01 -> 02 by list position


def _write_step2(out_dir: Path, rows: list[dict]) -> None:
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "cache" / "step2_df.csv", index=False)


def _rows(subject, session, thetas=(0.5, 0.7), runs=(None,), **metrics):
    """step2-style rows: metrics repeated across thetas, S varies (must not matter)."""
    out = []
    for run_idx, run_override in enumerate(runs):
        vals = dict(metrics)
        if run_override:
            vals.update(run_override)
        for k, theta in enumerate(thetas):
            out.append(
                {
                    "subject": subject,
                    "session": session,
                    "theta": theta,
                    "S": 10.0 * (k + 1) + run_idx,
                    **vals,
                }
            )
    return out


def _read_mixed(manifest):
    cols = [
        "subject",
        *[f"{m}_source_subject" for m in ("RAM", "PDI", "NAS", "IIM", "SRPI")],
    ]
    return pd.read_csv(manifest["step2_df_path"], dtype={c: str for c in cols})


def _compose_context(anchor_out, aux_out, mapping_aux):
    return {
        "plan": [
            {"dataset_id": "dsA", "out_dir": str(anchor_out), "data_origin": "real"},
            {"dataset_id": "dsB", "out_dir": str(aux_out), "data_origin": "dummy"},
        ],
        "anchor_dataset_id": "dsA",
        "primary_dataset_id": "dsA",
        "metric_dataset_map": {
            "RAM": "dsA",
            "PDI": "dsA",
            "NAS": "dsA",
            "IIM": "dsB",
            "SRPI": "dsB",
        },
        "subject_mapping_effective": {
            "dsA": {"01": "01", "02": "02"},
            "dsB": mapping_aux,
        },
    }


def test_mixed_source_ci_ground_truth_nan_semantics_and_pairing(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch, dataset_id="dsA")
    anchor_out = tmp_path / "outputs" / "scratch" / "dsA"
    aux_out = tmp_path / "test_objects" / "runs" / "dsB"
    _write_step2(
        anchor_out,
        # sub-01 awake has two runs (RAM 1 and 3 -> subject-session mean 2).
        _rows("01", "awake", runs=({"RAM": 1.0}, {"RAM": 3.0}), PDI=4.0, NAS=1.0)
        + _rows("01", "deep", RAM=1.0, PDI=2.0, NAS=0.5)
        + _rows("02", "awake", RAM=4.0, PDI=2.0, NAS=3.0)
        + _rows("02", "deep", RAM=1.0, PDI=1.0, NAS=1.0),
    )
    _write_step2(
        aux_out,
        # Different theta grid, and an IIM run flagged undefined that must be ignored.
        _rows(
            "01",
            "awake",
            thetas=(0.1,),
            runs=(None, {"IIM": 99.0, "IIM_defined": False}),
            IIM=3.0,
            SRPI=1.0,
            IIM_defined=True,
        )
        + _rows("01", "deep", thetas=(0.1,), IIM=1.5, SRPI=0.5, IIM_defined=True),
    )
    manifest = state._compose_mixed_source_ci(
        _compose_context(anchor_out, aux_out, {"01": "01", "02": None})
    )
    mixed = _read_mixed(manifest)
    by = {(r.subject, r.session): r for r in mixed.itertuples()}

    # Reference = cohort high-state (awake) means per component.
    refs = manifest["ci_reference_means"]
    assert manifest["ci_reference"] == "cohort_high_state"
    assert refs == pytest.approx(
        {"RAM": 3.0, "PDI": 3.0, "NAS": 2.0, "IIM": 3.0, "SRPI": 1.0}
    )
    # Ground truth: equal-weight geometric mean of reference-normalised components,
    # NAS used directly (S varies across theta/runs and must not matter).
    expected_awake = ((2 / 3) * (4 / 3) * (1 / 2) * (3 / 3) * (1 / 1)) ** 0.2
    expected_deep = ((1 / 3) * (2 / 3) * (0.5 / 2) * (1.5 / 3) * (0.5 / 1)) ** 0.2
    assert by[("01", "awake")].CI == pytest.approx(expected_awake, rel=1e-9)
    assert by[("01", "deep")].CI == pytest.approx(expected_deep, rel=1e-9)
    assert bool(by[("01", "awake")].CI_defined) is True

    # Unmapped participant: undefined (NaN), never 0, with the reason recorded.
    for ses in ("awake", "deep"):
        row = by[("02", ses)]
        assert math.isnan(row.CI)
        assert bool(row.CI_defined) is False
        assert row.CI_missing == "IIM,SRPI"
    assert manifest["ci_rows_total"] == 4
    assert manifest["ci_rows_defined"] == 2
    assert manifest["ci_rows_excluded_undefined"] == 2
    df_mean = pd.read_csv(manifest["step2_df_mean_path"], dtype={"subject": str})
    assert int(df_mean["CI"].isna().sum()) == 2
    assert not (df_mean["CI"] == 0).any()
    summary = {row["anchor_subject"]: row for row in manifest["subject_metric_summary"]}
    assert summary["02"]["IIM_source_subject"] is None
    assert summary["02"]["CI_defined_sessions"] == 0


def test_mixed_source_ci_explicit_mapping_session_match_and_bad_reference(
    tmp_path, monkeypatch
):
    dash, state = _state(tmp_path, monkeypatch, dataset_id="dsA")
    anchor_out = tmp_path / "outputs" / "scratch" / "dsA"
    aux_out = tmp_path / "test_objects" / "runs" / "dsB"
    _write_step2(
        anchor_out,
        _rows("01", "awake", RAM=1.0, PDI=1.0, NAS=1.0)
        + _rows("01", "deep", RAM=1.0, PDI=1.0, NAS=1.0)
        + _rows("02", "awake", RAM=1.0, PDI=1.0, NAS=1.0),
    )
    # sub-77 is explicitly mapped to anchor 02; sub-01 has no 'deep' session.
    _write_step2(
        aux_out,
        _rows("01", "awake", IIM=2.0, SRPI=2.0)
        + _rows("77", "awake", IIM=2.0, SRPI=2.0),
    )
    manifest = state._compose_mixed_source_ci(
        _compose_context(anchor_out, aux_out, {"01": "01", "02": "77"})
    )
    mixed = _read_mixed(manifest)
    by = {(r.subject, r.session): r for r in mixed.itertuples()}
    assert by[("02", "awake")].CI == pytest.approx(1.0)
    assert by[("02", "awake")].IIM_source_subject == "77"
    assert math.isnan(by[("01", "deep")].CI)  # joined on session, not position
    assert by[("01", "deep")].CI_missing == "IIM,SRPI"

    # Non-positive reference mean -> every CI undefined (no 1e-12 floor).
    _write_step2(
        aux_out,
        _rows("01", "awake", IIM=2.0, SRPI=0.0)
        + _rows("77", "awake", IIM=2.0, SRPI=0.0),
    )
    manifest = state._compose_mixed_source_ci(
        _compose_context(anchor_out, aux_out, {"01": "01", "02": "77"})
    )
    mixed = _read_mixed(manifest)
    assert mixed["CI"].isna().all()
    # synergy_ci.assemble_ci semantics: an undefined component is listed as
    # such; the reference is reported for rows whose components are defined.
    by = {(r.subject, r.session): r for r in mixed.itertuples()}
    assert by[("01", "deep")].CI_missing == "IIM,SRPI"
    for key in (("01", "awake"), ("02", "awake")):
        assert by[key].CI_missing == "SRPI_reference"
    assert manifest["ci_reference_invalid_components"] == ["SRPI"]


def test_plan_blocks_mixed_ci_without_matched_participants(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch, dataset_id="ds_primary")
    ra = tmp_path / "data" / "managed" / "ds_primary"
    rb = tmp_path / "data" / "managed" / "ds_other"
    _write_minimal_bids_root(ra, "A")
    (rb / "sub-99" / "func").mkdir(parents=True)
    (rb / "dataset_description.json").write_text(
        json.dumps({"Name": "B"}), encoding="utf-8"
    )
    for r in (
        tmp_path / "outputs" / "scratch" / "ds_primary",
        tmp_path / "outputs" / "scratch" / "ds_other",
    ):
        (r / "preprocessed").mkdir(parents=True)
    payload = {
        "selected_datasets": [
            {"dataset_id": "ds_primary", "bids_root": str(ra), "data_origin": "real"},
            {"dataset_id": "ds_other", "bids_root": str(rb), "data_origin": "real"},
        ],
        "primary_dataset_id": "ds_primary",
        "mpc_metrics": ["RAM", "PDI", "NAS", "IIM", "SRPI"],
        "metric_dataset_map": {
            "RAM": "ds_primary",
            "PDI": "ds_primary",
            "NAS": "ds_primary",
            "IIM": "ds_other",
            "SRPI": "ds_other",
        },
        "run_preprocessing": False,
    }
    plan = state.preview_run_plan(payload)
    assert plan["ci_mode"] == "mixed_source"
    assert plan["subject_mapping_auto"]["ds_other"]["01"] is None
    assert any(
        "No ds_primary participant is matched" in b for b in plan["plan_blockers"]
    )

    plan = state.preview_run_plan(
        {**payload, "subject_mapping": {"ds_other": {"01": "99"}}}
    )
    assert not any("participant is matched" in b for b in plan["plan_blockers"])

    plan = state.preview_run_plan(
        {
            **payload,
            "subject_mapping": {"ds_other": {"01": "99"}},
            "execution_mode": "hunter",
        }
    )
    assert any("Hunter" in b for b in plan["plan_blockers"])
    assert "only BUILDS" in plan["execution_note"]


def test_hunter_build_is_not_reported_as_a_completed_run(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    state._launch_managed_process(
        cmd=[sys.executable, "-c", "pass"],
        resolved={
            "out_dir": str(tmp_path / "o"),
            "dataset_id": "ds003171",
            "execution_mode": "hunter",
        },
        queue_remaining=0,
    )
    assert "Hunter campaign build started" in state._last_run_message
    state._managed_proc.wait(timeout=30)
    state._refresh_managed_process()
    msg = state.control_state_snapshot()["managed_run"]["display_message"]
    assert "NOT submitted" in msg and "qsub" in msg
    assert "finished successfully" not in msg


def test_replication_needs_melbourne_data(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    resolved = {
        "dataset_id": "ds003171",
        "data_origin": "real",
        "out_dir": str(tmp_path / "o"),
        "run_preprocessing": True,
        "run_replication": True,
        "mpc_metrics": ["PDI"],
    }
    with pytest.raises(ValueError, match="Melbourne"):
        state._apply_run_guardrails(dict(resolved))
    (tmp_path / "data" / "scratch" / "melbourne").mkdir(parents=True)
    state._apply_run_guardrails(dict(resolved))


def test_history_is_bounded_and_deduplicated(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch, history_points=100)
    for i in range(5000):
        state._append_history_point(
            {"t": 1000.0 + 2.0 * i, "system_cpu_percent": 1.23456789}
        )
    assert len(state.history) <= 100
    assert state.history[0]["t"] == 1000.0  # the run start stays visible
    assert state.history[-1]["t"] == 1000.0 + 2.0 * 4999
    assert state.history[-1]["system_cpu_percent"] == 1.235
    n = len(state.history)
    state._append_history_point(
        {"t": state.history[-1]["t"] + 0.2}
    )  # a second tab polling
    assert len(state.history) == n
    for _ in range(50):  # idle polls no longer grow the timeline without bound
        state.snapshot()
    assert len(state.history) <= 100


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signals")
def test_stop_of_paused_run_is_graceful_and_queue_is_cancelled(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    marker = tmp_path / "graceful.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib, signal, sys, time\n"
        f"m = pathlib.Path({str(marker)!r})\n"
        "def h(*a):\n    m.write_text('ok')\n    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, h)\n"
        "print('ready', flush=True)\n"
        "while True:\n    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    state._launch_managed_process(
        cmd=[sys.executable, str(child)],
        resolved={"out_dir": str(tmp_path / "o"), "dataset_id": "ds003171"},
        queue_remaining=1,
    )
    state._managed_queue = [
        {"cmd": [sys.executable, "-c", "pass"], "resolved": {"dataset_id": "never"}}
    ]
    deadline = time.time() + 20
    log = Path(state._managed_log_file)
    while "ready" not in log.read_text(encoding="utf-8") and time.time() < deadline:
        time.sleep(0.05)
    state.pause_run()
    t0 = time.time()
    res = state.stop_run(grace_seconds=8.0)
    assert time.time() - t0 < 6.0
    assert marker.exists()  # the SIGTERM handler ran although the group was SIGSTOPped
    assert res["message"] == "Managed run stopped."
    assert state._managed_queue == []
    assert state._last_error is None  # a user stop is not recorded as a failure
    assert not state._managed_run_file.exists()


def test_failed_run_cancels_queued_follow_ups(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    state._launch_managed_process(
        cmd=[sys.executable, "-c", "import sys; sys.exit(3)"],
        resolved={"out_dir": str(tmp_path / "o"), "dataset_id": "ds003171"},
        queue_remaining=1,
    )
    state._managed_queue = [{"cmd": ["true"], "resolved": {"dataset_id": "never"}}]
    state._managed_proc.wait(timeout=30)
    state._refresh_managed_process()
    assert state._managed_status == "error"
    assert state._managed_queue == []
    assert "cancelled" in state._last_run_message


def test_status_polls_are_not_blocked_while_a_start_is_planning(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    started = threading.Event()
    errors = []

    def slow_plan(payload):
        started.set()
        time.sleep(2.0)
        raise RuntimeError("planning aborted for the test")

    def run_start():
        try:
            state.start_run({})
        except RuntimeError as exc:
            errors.append(str(exc))

    monkeypatch.setattr(state, "preview_run_plan", slow_plan)
    worker = threading.Thread(target=run_start)
    worker.start()
    assert started.wait(10)
    t0 = time.time()
    state.control_state_snapshot()
    assert time.time() - t0 < 1.0
    with pytest.raises(RuntimeError, match="already in progress"):
        state.start_run({})
    worker.join(10)
    assert errors == ["planning aborted for the test"]


def test_windows_run_control_uses_process_groups_without_posix_calls(
    tmp_path, monkeypatch
):
    dash, state = _state(tmp_path, monkeypatch)
    captured = {}

    class FakePopen:
        pid = 424242

        def __init__(self, cmd, **kwargs):
            captured.update(kwargs)
            self.signals = []
            self.rc = None

        def poll(self):
            return self.rc

        def send_signal(self, sig):
            self.signals.append(sig)
            self.rc = 0

        def terminate(self):
            self.rc = 0

        def wait(self, timeout=None):
            return self.rc

    def forbidden(*_args):
        raise AssertionError("POSIX process-group call on Windows")

    monkeypatch.setattr(dash, "_IS_WINDOWS", True)
    monkeypatch.setattr(dash.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(dash.signal, "CTRL_BREAK_EVENT", 1, raising=False)
    monkeypatch.setattr(dash.os, "getpgid", forbidden, raising=False)
    monkeypatch.setattr(dash.os, "killpg", forbidden, raising=False)
    state._launch_managed_process(
        cmd=["python", "run_pipeline.py"],
        resolved={"out_dir": str(tmp_path / "o"), "dataset_id": "ds003171"},
        queue_remaining=0,
    )
    assert "creationflags" in captured
    assert "preexec_fn" not in captured and "start_new_session" not in captured
    snap = state.control_state_snapshot()["managed_run"]
    assert snap["pause_supported"] is False and snap["can_pause"] is False
    with pytest.raises(RuntimeError, match="not supported on Windows"):
        state.pause_run()
    res = state.stop_run(grace_seconds=1.0)
    assert res["message"].startswith("Managed run stopped")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_restarted_dashboard_reattaches_to_running_pipeline(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch)
    fake_pipeline = tmp_path / "run_pipeline.py"
    fake_pipeline.write_text(
        "import time\nwhile True:\n    time.sleep(0.05)\n", encoding="utf-8"
    )
    state._launch_managed_process(
        cmd=[sys.executable, str(fake_pipeline)],
        resolved={"out_dir": str(tmp_path / "o"), "dataset_id": "ds003171"},
        queue_remaining=0,
    )
    proc = state._managed_proc
    try:
        assert state._managed_run_file.exists()
        _, state2 = _state(tmp_path, monkeypatch)  # a new dashboard session
        snap = state2.control_state_snapshot()["managed_run"]
        assert (
            snap["active"] is True
            and snap["adopted"] is True
            and snap["pid"] == proc.pid
        )
        state2.stop_run(grace_seconds=5.0)
        proc.wait(timeout=10)
        assert state2.control_state_snapshot()["managed_run"]["active"] is False
        assert not state2._managed_run_file.exists()
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)


def test_powermetrics_feed_falls_back_to_base_cache(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch, dataset_id="ds005620")
    assert state._powermetrics_cache_file.parent.parent.name == "ds005620"
    cache = tmp_path / "outputs" / "scratch" / "cache" / "powermetrics_telemetry.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(
        json.dumps({"ok": True, "timestamp_unix": time.time(), "cpu_temp_c": 50.0})
    )
    feed = state._read_powermetrics_cache()
    assert feed["available"] is True
    assert np.isfinite(feed["cpu_temp_c"])


def test_compose_reads_each_runs_effective_folder_not_the_requested_base(
    tmp_path, monkeypatch
):
    """
    A selection may carry the base --out-dir (e.g. typed in the review card).
    ds005620's run then writes to <base>/ds005620, while <base>/cache holds
    ds003171's table; the combined index must read the folder the run used.
    """
    dash, state = _state(tmp_path, monkeypatch)
    base = tmp_path / "outputs" / "scratch"
    _write_minimal_bids_root(tmp_path / "data" / "scratch" / "ds003171", "A")
    _write_minimal_bids_root(tmp_path / "data" / "scratch" / "ds005620", "B")
    plan = state.preview_run_plan(
        {
            "selected_datasets": [
                {"dataset_id": "ds003171", "data_origin": "real", "out_dir": str(base)},
                {"dataset_id": "ds005620", "data_origin": "real", "out_dir": str(base)},
            ],
            "primary_dataset_id": "ds003171",
            "mpc_metrics": ["RAM", "PDI", "NAS", "IIM", "SRPI"],
            "metric_dataset_map": {
                "RAM": "ds003171",
                "PDI": "ds003171",
                "NAS": "ds003171",
                "IIM": "ds005620",
                "SRPI": "ds005620",
            },
            "run_preprocessing": True,
        }
    )
    assert plan["plan_ready"] is True, plan["plan_blockers"]
    eff = {item["dataset_id"]: item["effective_out_dir"] for item in plan["plan"]}
    assert Path(eff["ds005620"]) == (base / "ds005620").resolve()

    # ds003171's own table (with IIM/SRPI from an earlier full run) is in <base>.
    _write_step2(
        base,
        _rows("01", "awake", RAM=1.0, PDI=1.0, NAS=1.0, IIM=111.0, SRPI=111.0)
        + _rows("01", "deep", RAM=1.0, PDI=1.0, NAS=1.0, IIM=111.0, SRPI=111.0),
    )
    _write_step2(
        base / "ds005620",
        _rows("01", "awake", IIM=2.0, SRPI=2.0)
        + _rows("01", "deep", IIM=1.0, SRPI=1.0),
    )
    # Exactly the context start_run stores for the composition.
    manifest = state._compose_mixed_source_ci(dash.json_ready(dict(plan)))
    mixed = _read_mixed(manifest)
    assert sorted(set(mixed["IIM"])) == [1.0, 2.0]  # never ds003171's 111.0
    assert manifest["source_out_dirs"]["ds005620"] == str((base / "ds005620").resolve())
    assert Path(manifest["out_dir"]).parent == (base / "mixed_source_ci").resolve()


def test_plan_blocks_one_participant_matched_to_several(tmp_path, monkeypatch):
    dash, state = _state(tmp_path, monkeypatch, dataset_id="ds_primary")
    ra = tmp_path / "data" / "managed" / "ds_primary"
    rb = tmp_path / "data" / "managed" / "ds_other"
    _write_minimal_bids_root(ra, "A")
    (ra / "sub-02" / "func").mkdir(parents=True)
    (rb / "sub-99" / "func").mkdir(parents=True)
    (rb / "dataset_description.json").write_text(
        json.dumps({"Name": "B"}), encoding="utf-8"
    )
    for r in ("ds_primary", "ds_other"):
        (tmp_path / "outputs" / "scratch" / r / "preprocessed").mkdir(parents=True)
    payload = {
        "selected_datasets": [
            {"dataset_id": "ds_primary", "bids_root": str(ra), "data_origin": "real"},
            {"dataset_id": "ds_other", "bids_root": str(rb), "data_origin": "real"},
        ],
        "primary_dataset_id": "ds_primary",
        "mpc_metrics": ["RAM", "PDI", "NAS", "IIM", "SRPI"],
        "metric_dataset_map": {
            "RAM": "ds_primary",
            "PDI": "ds_primary",
            "NAS": "ds_primary",
            "IIM": "ds_other",
            "SRPI": "ds_other",
        },
        "run_preprocessing": False,
    }
    plan = state.preview_run_plan(
        {**payload, "subject_mapping": {"ds_other": {"01": "99", "02": "99"}}}
    )
    assert plan["plan_ready"] is False
    assert any("at most once" in b for b in plan["plan_blockers"])
    plan = state.preview_run_plan(
        {**payload, "subject_mapping": {"ds_other": {"01": "99"}}}
    )
    assert not any("at most once" in b for b in plan["plan_blockers"])


def test_plan_blocks_case_variant_ids_that_share_a_folder(tmp_path, monkeypatch):
    """On case-insensitive file systems (macOS default) dsX and DSX collide."""
    dash, state = _state(tmp_path, monkeypatch)
    root = tmp_path / "data" / "managed" / "dscase"
    _write_minimal_bids_root(root, "x")
    for name in ("dscase", "DSCASE"):
        (tmp_path / "outputs" / "scratch" / name / "preprocessed").mkdir(
            parents=True, exist_ok=True
        )
    plan = state.preview_run_plan(
        {
            "selected_datasets": [
                {"dataset_id": "dscase", "data_origin": "real", "bids_root": str(root)},
                {"dataset_id": "DSCASE", "data_origin": "real", "bids_root": str(root)},
            ],
            "primary_dataset_id": "dscase",
            "mpc_metrics": ["RAM", "PDI"],
            "metric_dataset_map": {"RAM": "dscase", "PDI": "DSCASE"},
            "run_preprocessing": False,
        }
    )
    assert plan["plan_ready"] is False
    assert any("same output directory" in b for b in plan["plan_blockers"])


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_reattach_signals_the_live_process_group_not_the_recorded_one(
    tmp_path, monkeypatch
):
    dash, state = _state(tmp_path, monkeypatch)
    fake_pipeline = tmp_path / "run_pipeline.py"
    fake_pipeline.write_text(
        "import time\nwhile True:\n    time.sleep(0.05)\n", encoding="utf-8"
    )
    state._launch_managed_process(
        cmd=[sys.executable, str(fake_pipeline)],
        resolved={"out_dir": str(tmp_path / "o"), "dataset_id": "ds003171"},
        queue_remaining=0,
    )
    proc = state._managed_proc
    try:
        rec = json.loads(state._managed_run_file.read_text(encoding="utf-8"))
        rec["pgid"] = os.getpgid(0)  # edited record: the test runner's own group
        state._managed_run_file.write_text(json.dumps(rec), encoding="utf-8")
        _, state2 = _state(tmp_path, monkeypatch)
        assert state2._managed_adopted is True
        assert state2._managed_pgid == os.getpgid(proc.pid)
        assert state2._managed_pgid != os.getpgid(0)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)
