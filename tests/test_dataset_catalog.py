from pathlib import Path

from impact_pipeline.dataset_catalog import (
    PIPELINE_ENABLED_DATASET_IDS,
    get_report_dataset,
    iter_report_datasets,
    resolve_local_dataset_root,
    resolve_representative_payload_paths,
)


def test_report_dataset_lookup_exposes_pipeline_flag():
    entry = get_report_dataset("ds006623")
    assert entry is not None
    assert entry.dataset_id == "ds006623"
    assert entry.pipeline_ready is False
    assert "ds003171" in PIPELINE_ENABLED_DATASET_IDS
    assert "ds006623" not in PIPELINE_ENABLED_DATASET_IDS


def test_resolve_local_dataset_root_prefers_annex_candidate(tmp_path: Path):
    annex_root = tmp_path / "data" / "scratch" / "ds005620_annex"
    plain_root = tmp_path / "data" / "scratch" / "ds005620"
    plain_root.mkdir(parents=True, exist_ok=True)
    annex_root.mkdir(parents=True, exist_ok=True)

    resolved = resolve_local_dataset_root("ds005620", tmp_path)
    assert resolved == annex_root.resolve()


def test_every_report_dataset_defines_representative_payloads():
    for entry in iter_report_datasets():
        assert entry.representative_payload_paths, entry.dataset_id


def test_resolve_representative_payload_paths_uses_selected_local_root(tmp_path: Path):
    root = tmp_path / "data" / "scratch" / "ds005620_annex"
    root.mkdir(parents=True, exist_ok=True)

    resolved = resolve_representative_payload_paths("ds005620", tmp_path)
    assert resolved
    assert resolved[0] == root / "sub-1010/eeg/sub-1010_task-awake_acq-EC_eeg.vhdr"


def test_resolve_local_dataset_root_follows_an_external_symlinked_root(
    tmp_path: Path,
):
    repo_root = tmp_path / "repo"
    local_link = repo_root / "data" / "scratch" / "ds006623"
    real_root = tmp_path / "external" / "data" / "scratch" / "ds006623"
    real_root.mkdir(parents=True, exist_ok=True)
    local_link.parent.mkdir(parents=True, exist_ok=True)
    local_link.symlink_to(real_root, target_is_directory=True)

    assert resolve_local_dataset_root("ds006623", repo_root) == real_root.resolve()
