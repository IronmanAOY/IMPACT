# -*- coding: utf-8 -*-
"""Regression tests for packaging, environment and helper-script fixes."""
import importlib.metadata
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: tomli ships with pytest
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if not (ROOT / ".github" / "workflows" / "ci.yml").is_file():
    # e.g. tests mounted into the runtime container, which holds no repo files
    pytest.skip("packaging tests need a full source checkout", allow_module_level=True)
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")

# pip distribution name -> conda-forge package name, where they differ.
CONDA_NAME = {"mne": "mne-base", "matplotlib": "matplotlib-base"}


def _pyproject():
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def _env_specs():
    yaml = pytest.importorskip("yaml")
    env = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
    specs = {}
    for dep in env["dependencies"]:
        if not isinstance(dep, str):
            continue
        name = re.split(r"[=<>!~ ]", dep, maxsplit=1)[0]
        specs[name] = dep[len(name):]
    return env, specs


def _run_script(args, env=None, cwd=None):
    full_env = dict(os.environ)
    full_env.update(env or {})
    return subprocess.run(
        [BASH] + [str(a) for a in args],
        cwd=str(cwd or ROOT),
        env=full_env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _last_command(stdout):
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    return shlex.split(lines[-1])


# --- packaging / environment ------------------------------------------------

def test_pyproject_declares_installable_src_package():
    meta = _pyproject()
    project = meta["project"]
    assert project["name"] == "impact-synergy-pipeline"
    assert project["requires-python"] == ">=3.10"
    assert meta["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert (ROOT / "src" / "impact_pipeline" / "__init__.py").is_file()
    assert set(project["optional-dependencies"]) >= {"hunter", "dashboard", "dev"}
    hunter = " ".join(project["optional-dependencies"]["hunter"]).lower()
    assert "cupy" not in hunter


def test_pyproject_dependencies_are_in_environment_yml():
    _, specs = _env_specs()
    for dep in _pyproject()["project"]["dependencies"]:
        req = Requirement(dep)
        conda_name = CONDA_NAME.get(req.name, req.name)
        assert conda_name in specs, f"{req.name} missing from environment.yml"
        pin = specs[conda_name]
        match = re.fullmatch(r"=([0-9.]+?)(\.\*)?", pin)
        if match:  # minor pin such as numpy=2.2.* must fall inside the range
            assert req.specifier.contains(Version(match.group(1)), prereleases=True), (
                f"environment.yml pins {conda_name}{pin} outside pyproject {req}"
            )


def test_environment_yml_is_pinned_for_python310():
    env, specs = _env_specs()
    assert env["name"] == "impact-synergy-clean"
    assert specs["python"] == "=3.10"
    for pkg in ("numpy", "scipy", "pandas", "nilearn", "mne-base", "numba", "pybids"):
        assert specs.get(pkg, "").startswith("="), f"{pkg} is not pinned"
    for dropped in ("pyinstaller", "black", "mne"):
        assert dropped not in specs


def test_installed_versions_satisfy_pyproject_ranges():
    for dep in _pyproject()["project"]["dependencies"]:
        req = Requirement(dep)
        try:
            installed = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            pytest.fail(f"{req.name} is not installed")
        assert req.specifier.contains(installed, prereleases=True), (
            f"installed {req.name} {installed} outside {req.specifier}"
        )


def test_citation_and_license_metadata():
    yaml = pytest.importorskip("yaml")
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert str(cff["version"]) == _pyproject()["project"]["version"] == "1.1.0"
    assert str(cff["date-released"]) == "2026-09-28"
    assert cff["doi"] == "10.5281/zenodo.15306740"
    assert cff["license"] == "MIT"
    repo = "https://github.com/IronmanAOY/impact-synergy-pipeline"
    assert cff["repository-code"] == repo
    assert cff["authors"][0]["orcid"] == "https://orcid.org/0009-0005-7698-8304"
    license_text = (ROOT / "licenses" / "MIT_LICENSE").read_text()
    assert (ROOT / "LICENSE").read_text() == license_text


def test_ci_workflow_uses_login_shell_and_installs_package():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    job = wf["jobs"]["test"]
    assert job["defaults"]["run"]["shell"] == "bash -el {0}"
    runs = [step.get("run", "") for step in job["steps"]]
    assert any("pip install" in r and "-e ." in r for r in runs)
    assert any("pytest" in r and "--timeout" in r for r in runs)


def test_container_recipes_do_not_copy_data():
    ignore = [
        ln.strip()
        for ln in (ROOT / ".dockerignore").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert ignore[0] == "*"  # allowlist: nothing enters the context by default
    for pattern in ("data/", "test_objects/", "dist/", "outputs/", "docs/manuscript/",
                    ".tmp.drive*", "licenses/fs_license*.txt"):
        assert pattern in ignore
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert not re.search(r"^COPY\s+\.\s", dockerfile, flags=re.M)
    assert "setup_14.x" not in dockerfile
    assert re.search(r"^FROM \S+:\S+@sha256:[0-9a-f]{64}$", dockerfile, flags=re.M)
    singularity = (ROOT / "Singularity").read_text()
    assert "setup_14.x" not in singularity
    assert "source activate" not in singularity
    files = singularity.split("%files", 1)[1].split("\n%", 1)[0]
    for line in files.strip().splitlines():
        src = line.split()[0]
        assert src != ".", "Singularity must not copy the whole tree"
        assert (ROOT / src).exists(), f"Singularity %files source missing: {src}"


@needs_bash
def test_singularity_sections_are_valid_shell():
    text = (ROOT / "Singularity").read_text()
    for section in ("post", "environment", "runscript", "test"):
        body = text.split(f"%{section}\n", 1)[1].split("\n%", 1)[0]
        proc = subprocess.run([BASH, "-n"], input=body, capture_output=True, text=True)
        assert proc.returncode == 0, f"%{section}: {proc.stderr}"


def test_gitignore_protects_manuscripts_and_scratch():
    lines = set((ROOT / ".gitignore").read_text().splitlines())
    for pattern in ("docs/manuscript/", ".tmp.driveupload/", ".tmp.drivedownload/",
                    "*.aux", "*.fls", "*.fdb_latexmk", "__pycache__/", ".DS_Store"):
        assert pattern in lines


# --- helper scripts ---------------------------------------------------------

@needs_bash
@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("*.sh")))
def test_shell_scripts_parse(script):
    proc = _run_script(["-n", SCRIPTS / script])
    assert proc.returncode == 0, proc.stderr


@needs_bash
def test_download_atlases_verifies_tracked_atlases_offline():
    proc = _run_script([SCRIPTS / "download_atlases.sh", "--verify"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "mismatch" not in proc.stdout and "missing" not in proc.stdout


@needs_bash
def test_download_atlases_detects_modified_and_missing_files(tmp_path):
    dest = tmp_path / "atlases"
    shutil.copytree(ROOT / "atlases", dest)
    shen = dest / "shen_1mm_268_parcellation.nii.gz"
    shen.write_bytes(shen.read_bytes() + b"x")
    (dest / "aal_SPM12" / "aal" / "atlas" / "AAL.nii").unlink()
    proc = _run_script([SCRIPTS / "download_atlases.sh", "--verify", "--dest", dest])
    assert proc.returncode == 1
    assert "mismatch: atlases/shen_1mm_268_parcellation.nii.gz" in proc.stdout
    assert "missing: atlases/aal_SPM12/aal/atlas/AAL.nii" in proc.stdout
    # Without --force a modified local file is never overwritten.
    proc = _run_script([SCRIPTS / "download_atlases.sh", "--dest", dest])
    assert proc.returncode == 1
    assert "not overwriting" in proc.stderr


def _fake_bids(tmp_path, name="ds003171"):
    bids = tmp_path / "bids" / name
    (bids / "sub-02CB").mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        '{"Name": "x", "BIDSVersion": "1.6.0"}'
    )
    return bids


@needs_bash
def test_fetch_fmriprep_command_is_pinned_and_writes_canonical_derivatives(tmp_path):
    bids = _fake_bids(tmp_path)
    license_file = tmp_path / "my license.txt"
    license_file.write_text("fake")
    proc = _run_script(
        [SCRIPTS / "fetch_fmriprep_ds003171.sh", "--bids-root", bids,
         "--skip-reconall", "--participant-label", "sub-02CB", "04HD", "--dry-run",
         "--", "--skip-bids-validation"],
        env={"FS_LICENSE": str(license_file)},
    )
    assert proc.returncode == 0, proc.stderr
    cmd = _last_command(proc.stdout)
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "nipreps/fmriprep:25.1.3" in cmd
    mounts = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-v"]
    assert f"{bids.resolve()}:/data:ro" in mounts
    assert f"{bids.resolve()}/derivatives/fmriprep:/out" in mounts
    assert f"{license_file.resolve()}:/opt/freesurfer/license.txt:ro" in mounts
    fmriprep_args = cmd[cmd.index("nipreps/fmriprep:25.1.3") + 1:]
    assert fmriprep_args[:3] == ["/data", "/out", "participant"]
    labels = fmriprep_args[fmriprep_args.index("--participant-label") + 1:][:2]
    assert labels == ["02CB", "04HD"]
    assert "all" not in fmriprep_args
    assert "--fs-no-reconall" in fmriprep_args
    assert fmriprep_args[fmriprep_args.index("--fs-license-file") + 1] == (
        "/opt/freesurfer/license.txt"
    )
    assert fmriprep_args[-1] == "--skip-bids-validation"


@needs_bash
def test_fetch_fmriprep_without_labels_processes_all_subjects(tmp_path):
    bids = _fake_bids(tmp_path)
    license_file = tmp_path / "license.txt"
    license_file.write_text("fake")
    proc = _run_script(
        [SCRIPTS / "fetch_fmriprep.sh", "--bids-root", bids,
         "--fs-license", license_file, "--dry-run"],
    )
    assert proc.returncode == 0, proc.stderr
    cmd = _last_command(proc.stdout)
    assert "--participant-label" not in cmd
    assert "--fs-no-reconall" not in cmd


@needs_bash
@pytest.mark.skipif((ROOT / "licenses" / "fs_license.txt").exists(),
                    reason="a local FreeSurfer license is present")
def test_fetch_fmriprep_requires_license_even_without_reconall(tmp_path):
    bids = _fake_bids(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "FS_LICENSE"}
    proc = subprocess.run(
        [BASH, str(SCRIPTS / "fetch_fmriprep.sh"), "--bids-root", str(bids),
         "--skip-reconall", "--dry-run"],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1
    assert "FreeSurfer license is required" in proc.stderr


@needs_bash
def test_fetch_fmriprep_rejects_non_bids_root(tmp_path):
    proc = _run_script([SCRIPTS / "fetch_fmriprep.sh", "--bids-root", tmp_path / "nope",
                        "--dry-run"])
    assert proc.returncode == 1
    assert "Not a BIDS dataset root" in proc.stderr


@needs_bash
def test_download_data_uses_pinned_snapshot_and_absolute_path(tmp_path):
    proc = _run_script(
        [SCRIPTS / "download_data.sh", "--dry-run", "ds005620", "", "rel/ds005620"],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    target = f"{tmp_path.resolve()}/rel/ds005620"
    assert f"Target:   {target}" in out
    assert "https://github.com/OpenNeuroDatasets/ds005620.git" in out
    assert "refs/tags/1.0.0" in out
    assert "openneuro" not in out.replace("OpenNeuroDatasets", "")


@needs_bash
def test_download_data_ds003171_does_not_clone_melbourne(tmp_path):
    proc = _run_script(
        [SCRIPTS / "download_data.sh", "--dry-run", "--no-get", "ds003171", "",
         tmp_path / "d"],
    )
    assert proc.returncode == 0, proc.stderr
    assert "refs/tags/2.0.1" in proc.stdout
    assert "melbourne" not in proc.stdout.lower()


@needs_bash
@pytest.mark.parametrize(
    "args",
    [["ds999999"], ["not-a-dataset", "1.0.0"]],
)
def test_download_data_rejects_unpinned_or_invalid_ids(tmp_path, args):
    proc = _run_script([SCRIPTS / "download_data.sh", "--dry-run"] + args, cwd=tmp_path)
    assert proc.returncode == 2


@needs_bash
def test_run_all_does_not_require_melbourne_and_passes_fmriprep_dir(tmp_path):
    data_root = tmp_path / "scratch"
    proc = _run_script(
        [SCRIPTS / "run_all.sh", "--dry-run", "ds003171"],
        env={"IMPACT_DATA_ROOT": str(data_root), "IMPACT_PYTHON": "python",
             "IMPACT_OUT_DIR": str(tmp_path / "out")},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "melbourne" not in out.lower()
    bids = f"{data_root}/ds003171"
    derivatives = f"{bids}/derivatives/fmriprep"
    lines = out.splitlines()
    pipeline = shlex.split(next(ln for ln in lines if "run_pipeline.py" in ln))
    assert pipeline[pipeline.index("--bids-root") + 1] == bids
    assert pipeline[pipeline.index("--fmriprep-dir") + 1] == derivatives
    assert "--run-replication" not in pipeline
    fetch = shlex.split(next(ln for ln in lines if "fetch_fmriprep.sh" in ln))
    assert fetch[fetch.index("--out-dir") + 1] == derivatives


def _stub(bin_dir, name, body):
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


@needs_bash
def test_run_all_end_to_end_with_stubbed_tools(tmp_path):
    """Real (non-dry) control flow with git/docker/python replaced by stubs."""
    log = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # git: pretend the dataset clone exists at the pinned tag; annex calls succeed.
    _stub(bin_dir, "git", f'echo "git $*" >> "{log}"\n'
          'case "$*" in *describe*) echo 2.0.1 ;; esac\nexit 0\n')
    _stub(bin_dir, "docker", f'echo "docker $*" >> "{log}"\nexit 0\n')
    _stub(bin_dir, "fakepython", f'echo "python $*" >> "{log}"\nexit 0\n')
    data_root = tmp_path / "scratch"
    bids = data_root / "ds003171"
    (bids / ".git").mkdir(parents=True)
    (bids / "sub-02CB").mkdir()
    (bids / "dataset_description.json").write_text('{"Name": "x"}')
    license_file = tmp_path / "license.txt"
    license_file.write_text("fake")
    path = os.pathsep.join([str(bin_dir), "/usr/bin", "/bin", "/usr/sbin", "/sbin"])
    proc = _run_script(
        [SCRIPTS / "run_all.sh", "ds003171"],
        env={"PATH": path, "IMPACT_DATA_ROOT": str(data_root),
             "IMPACT_PYTHON": str(bin_dir / "fakepython"),
             "IMPACT_OUT_DIR": str(tmp_path / "out"), "FS_LICENSE": str(license_file)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = log.read_text().splitlines()
    assert any("describe --tags --exact-match" in c for c in calls)
    assert any("annex get" in c for c in calls)
    docker = next(c for c in calls if c.startswith("docker run"))
    assert "nipreps/fmriprep:25.1.3" in docker
    assert "--participant-label 02CB" in docker
    assert f"{bids}/derivatives/fmriprep:/out" in docker
    pipeline = next(c for c in calls if c.startswith("python "))
    assert f"--fmriprep-dir {bids}/derivatives/fmriprep" in pipeline
    assert "melbourne" not in proc.stdout.lower() + proc.stderr.lower()


@needs_bash
def test_run_all_eeg_skips_fmriprep(tmp_path):
    proc = _run_script(
        [SCRIPTS / "run_all.sh", "--dry-run", "ds005620"],
        env={"IMPACT_DATA_ROOT": str(tmp_path), "IMPACT_PYTHON": "python"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "fetch_fmriprep" not in proc.stdout
    assert "download_atlases" not in proc.stdout
    assert "--fmriprep-dir" not in proc.stdout


@needs_bash
def test_run_all_rejects_unsupported_dataset(tmp_path):
    proc = _run_script([SCRIPTS / "run_all.sh", "--dry-run", "ds000001"],
                       env={"IMPACT_DATA_ROOT": str(tmp_path)})
    assert proc.returncode == 2


def test_third_party_notice_lists_atlas_checksums():
    notice = (ROOT / "licenses" / "THIRD_PARTY_NOTICES.md").read_text()
    script = (SCRIPTS / "download_atlases.sh").read_text()
    sums = re.findall(r"\|([0-9a-f]{64})\"", script)
    assert len(sums) == 5
    for digest in sums:
        if digest == "1a0f2bb952b700fa94f5156b544cf238f2e8d0e0e5ebedaefac835938b7609f3":
            continue  # AAL.xml label file, listed with the AAL folder
        assert digest in notice
