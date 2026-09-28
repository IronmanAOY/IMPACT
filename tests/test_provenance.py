import json
from pathlib import Path

import pytest

import impact_pipeline
from impact_pipeline import provenance
from impact_pipeline.provenance import (
    assert_origin_matches_dataset,
    collect_code_version,
    collect_runtime_versions,
    dataset_declares_synthetic,
    format_code_version,
    resolve_dataset_provenance,
    resolve_repo_root,
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


# ---------------------------------------------------------------------------
# Package version, code version and checkout resolution (stream I2)
# ---------------------------------------------------------------------------


def _pyproject_version():
    import re

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    return re.search(r'(?m)^version\s*=\s*"([^"]+)"', text).group(1)


def test_package_version_is_the_declared_version():
    from importlib import metadata

    try:
        installed = metadata.version("impact-synergy-pipeline")
    except metadata.PackageNotFoundError:
        installed = None
    assert impact_pipeline.__version__ in {_pyproject_version(), installed}
    assert collect_runtime_versions()["impact_pipeline_version"] == (
        impact_pipeline.__version__
    )


def test_package_version_falls_back_without_distribution_metadata(monkeypatch):
    from importlib import metadata

    def missing(_name):
        raise metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(metadata, "version", missing)
    assert impact_pipeline._resolve_version() == _pyproject_version()
    monkeypatch.setattr(impact_pipeline, "_version_from_pyproject", lambda: None)
    assert impact_pipeline._resolve_version() == "0+unknown"


def test_checkout_version_wins_over_an_unrelated_installed_wheel(monkeypatch):
    from importlib import metadata

    # A checkout run via PYTHONPATH next to an older installed wheel.
    monkeypatch.setattr(metadata, "version", lambda _name: "0.9.0")
    assert impact_pipeline._resolve_version() == _pyproject_version()
    # No checkout next to the package (regular install): the metadata.
    monkeypatch.setattr(impact_pipeline, "_version_from_pyproject", lambda: None)
    assert impact_pipeline._resolve_version() == "0.9.0"


def test_code_version_is_package_version_plus_commit(tmp_path, monkeypatch):
    base = impact_pipeline.__version__.split("+", 1)[0]
    sha = "a" * 40
    assert (
        format_code_version({"package_version": base, "git_sha": sha})
        == f"{base}+g{sha}"
    )
    assert (
        format_code_version(
            {"package_version": base, "git_sha": sha, "git_dirty": True}
        )
        == f"{base}+g{sha}.dirty"
    )
    assert format_code_version({"package_version": base}) == f"{base}+unknown"
    monkeypatch.setenv("IMPACT_CODE_VERSION", "hunter-2026-09")
    info = collect_code_version(tmp_path)  # not a git checkout
    assert info["package_version"] == impact_pipeline.__version__
    assert info["code_version"] == f"{base}+hunter-2026-09"
    repo = Path(__file__).resolve().parents[1]
    info = collect_code_version(repo)
    if info["source"] == "git":
        assert info["code_version"].startswith(f"{base}+g{info['git_sha']}")


def test_repo_root_resolution(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.delenv(provenance.REPO_ROOT_ENV, raising=False)
    # source checkout / editable install: the package-relative root
    assert resolve_repo_root() == repo
    other = tmp_path / "checkout"
    other.mkdir()
    (other / "run_pipeline.py").write_text("# entry point\n")
    assert resolve_repo_root(other) == other.resolve()
    monkeypatch.setenv(provenance.REPO_ROOT_ENV, str(other))
    assert resolve_repo_root() == other.resolve()
    with pytest.raises(FileNotFoundError, match="--repo-root"):
        resolve_repo_root(tmp_path)  # explicit but not a checkout: no substitution
    monkeypatch.setenv(provenance.REPO_ROOT_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match=provenance.REPO_ROOT_ENV):
        resolve_repo_root()


def test_repo_root_for_a_non_editable_install(tmp_path, monkeypatch):
    site = tmp_path / "site-packages" / "impact_pipeline"
    site.mkdir(parents=True)
    monkeypatch.setattr(provenance, "__file__", str(site / "provenance.py"))
    monkeypatch.delenv(provenance.REPO_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_repo_root(required=False) is None
    with pytest.raises(FileNotFoundError, match="non-editable"):
        resolve_repo_root()
    checkout = tmp_path / "ws" / "impact-synergy-pipeline"
    checkout.mkdir(parents=True)
    (checkout / "run_pipeline.py").write_text("# entry point\n")
    # run_pipeline.py passes its own directory as the fallback
    assert resolve_repo_root(fallback=checkout) == checkout.resolve()
    monkeypatch.setenv(provenance.REPO_ROOT_ENV, str(checkout))
    assert resolve_repo_root() == checkout.resolve()
