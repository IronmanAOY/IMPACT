import json
from pathlib import Path

import pytest

from impact_pipeline.provenance import (
    assert_origin_matches_dataset,
    collect_code_version,
    collect_runtime_versions,
    dataset_declares_synthetic,
    resolve_dataset_provenance,
)


def test_synthetic_provenance_uses_validation_output_root(tmp_path, monkeypatch):
    # The README tells users to export IMPACT_SYNTH_ROOT; do not depend on it.
    monkeypatch.delenv("IMPACT_SYNTH_ROOT", raising=False)
    prov = resolve_dataset_provenance(
        repo_root=tmp_path,
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="dummy_ds",
        data_origin="dummy",
    )
    assert prov.data_origin == "dummy"
    assert prov.dataset_role == "synthetic_validation"
    assert prov.effective_out_dir == (tmp_path / "test_objects" / "runs" / "dummy_ds").resolve()
    assert prov.metric_bank_dataset_dir == (tmp_path / "test_objects" / "metric_bank" / "dummy_ds").resolve()


def test_synthetic_provenance_can_use_external_root(tmp_path, monkeypatch):
    external_root = tmp_path / "external"
    monkeypatch.setenv("IMPACT_SYNTH_ROOT", str(external_root))

    prov = resolve_dataset_provenance(
        repo_root=tmp_path / "repo",
        out_dir=external_root / "test_objects" / "runs" / "real_derived_synth_completed" / "ds003171",
        dataset_id="ds003171",
        data_origin="dummy",
    )

    assert prov.effective_out_dir == (
        external_root / "test_objects" / "runs" / "real_derived_synth_completed" / "ds003171"
    ).resolve()
    assert prov.metric_bank_dataset_dir == (external_root / "test_objects" / "metric_bank" / "ds003171").resolve()


def test_real_provenance_keeps_standard_output_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("IMPACT_SYNTH_ROOT", raising=False)
    prov = resolve_dataset_provenance(
        repo_root=tmp_path,
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds005620",
        data_origin="real",
    )
    assert prov.data_origin == "real"
    assert prov.dataset_role == "study_data"
    assert prov.effective_out_dir == (tmp_path / "outputs" / "scratch" / "ds005620").resolve()
    assert prov.metric_bank_dataset_dir is None
    # real runs no longer create synthetic scaffolding as a side effect
    assert not (tmp_path / "test_objects").exists()


def test_code_version_records_git_commit_of_this_checkout():
    repo = Path(__file__).resolve().parents[1]
    info = collect_code_version(repo)
    if info["source"] == "git":
        assert len(info["git_sha"]) == 40
        assert isinstance(info["git_dirty"], bool)
    else:  # e.g. an exported tarball: explicit, not a silent guess
        assert info["git_sha"] is None


def test_code_version_without_git_uses_declared_version(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPACT_CODE_VERSION", "1.1.0+hunter")
    info = collect_code_version(tmp_path)
    assert info["git_sha"] is None
    assert info["declared_version"] == "1.1.0+hunter"
    assert info["source"] == "IMPACT_CODE_VERSION"


def test_runtime_versions_list_installed_packages():
    info = collect_runtime_versions()
    assert info["python"].count(".") >= 1
    assert "numpy" in info["packages"]


def test_synthetic_dataset_cannot_be_labelled_real(tmp_path):
    (tmp_path / "dataset_description.json").write_text(
        json.dumps(
            {"Name": "synthetic", "SyntheticData": True, "DatasetType": "synthetic"}
        )
    )
    assert dataset_declares_synthetic(tmp_path)
    with pytest.raises(ValueError, match="--data-origin dummy"):
        assert_origin_matches_dataset(tmp_path, "real")
    assert_origin_matches_dataset(tmp_path, "dummy")
    (tmp_path / "dataset_description.json").write_text(json.dumps({"Name": "real"}))
    assert not dataset_declares_synthetic(tmp_path)
    assert_origin_matches_dataset(tmp_path, "real")
