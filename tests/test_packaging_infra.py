# -*- coding: utf-8 -*-
"""Regression tests for packaging, environment and helper-script fixes."""
import importlib.metadata
import os
import re
import shlex
import shutil
import subprocess
import sys
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
    # Only the tracked license texts may enter the context: a FreeSurfer
    # license stored under any other name must not reach an image layer.
    reincluded = [p[1:] for p in ignore if p.startswith("!licenses")]
    assert "licenses/" not in reincluded and "licenses" not in reincluded
    assert sorted(reincluded) == [
        "licenses/MIT_LICENSE",
        "licenses/THIRD_PARTY_NOTICES.md",
        "licenses/fs_license.txt.example",
    ]
    for path in reincluded:
        assert (ROOT / path).is_file(), path
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


def test_container_recipes_ship_the_default_protocol():
    """run_pipeline.py reads protocols/mpc_default_v1.json by default on
    empirical data (1.1.0 post-freeze fixes), so both images must contain
    protocols/ next to run_pipeline.py (else every default run stops with
    'MPC protocol not found')."""
    import run_pipeline

    default = Path(run_pipeline.DEFAULT_EMPIRICAL_PROTOCOL)
    assert default == ROOT / "protocols" / "mpc_default_v1.json" and default.is_file()
    ignore = [ln.strip() for ln in (ROOT / ".dockerignore").read_text().splitlines()]
    assert "!protocols/" in ignore
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert re.search(r"^COPY protocols/ protocols/$", dockerfile, flags=re.M)
    singularity = (ROOT / "Singularity").read_text()
    files = singularity.split("%files", 1)[1].split("\n%", 1)[0]
    lines = [ln.strip() for ln in files.splitlines()]
    assert "protocols /opt/impact/protocols" in lines
    assert "/opt/impact/run_pipeline.py" in files


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


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_gitignore_rules_take_effect_without_hiding_tracked_files():
    git = ["git", "-C", str(ROOT)]
    inside = subprocess.run(git + ["rev-parse", "--is-inside-work-tree"],
                            capture_output=True, text=True)
    if inside.returncode != 0:
        pytest.skip("not a git checkout")
    paths = [
        "docs/manuscript/paper_2/main.tex",
        ".tmp.driveupload/123",
        ".tmp.drivedownload/456",
        "docs/notes/build/paper.aux",
        "paper.synctex.gz",
        "data/ds003171/sub-01/func/bold.nii.gz",
        "data/melbourne/sub-01/x.nii.gz",
        "data/scratch/ds005620/sub-1/eeg/x.vhdr",
        "licenses/fs_license.txt",
        "licenses/fs_license_2026.txt",
        "licenses/license.txt",
        "impact.sif",
    ]
    proc = subprocess.run(git + ["check-ignore", "--no-index", *paths],
                          capture_output=True, text=True)
    ignored = set(proc.stdout.split())
    assert ignored == set(paths), f"not ignored: {sorted(set(paths) - ignored)}"
    # No tracked file may be hidden by an over-broad pattern (e.g. *.out).
    tracked_ignored = subprocess.run(
        git + ["ls-files", "-ci", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert tracked_ignored == []


# --- helper scripts ---------------------------------------------------------

@needs_bash
@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("*.sh")))
def test_shell_scripts_parse(script):
    proc = _run_script(["-n", SCRIPTS / script])
    assert proc.returncode == 0, proc.stderr


def test_impact_pipeline_is_imported_from_this_checkout():
    import impact_pipeline

    if os.environ.get("IMPACT_TEST_INSTALLED_PACKAGE") == "1":
        pytest.skip("testing an installed copy on request")
    origin = Path(impact_pipeline.__file__).resolve()
    assert (ROOT / "src") in origin.parents, origin


def test_conftest_prefers_this_checkout_over_another_importable_copy(tmp_path):
    # Another copy of the package importable first (another checkout or
    # worktree installed into the shared env, a stale install): the suite must
    # still test this checkout's src/, not silently the other copy.
    other = tmp_path / "other_checkout"
    (other / "impact_pipeline").mkdir(parents=True)
    (other / "impact_pipeline" / "__init__.py").write_text("OTHER_COPY = True\n")
    env = {k: v for k, v in os.environ.items() if k != "IMPACT_TEST_INSTALLED_PACKAGE"}
    env["PYTHONPATH"] = str(other)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    node = (f"{Path(__file__).resolve()}"
            "::test_impact_pipeline_is_imported_from_this_checkout")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed" in proc.stdout


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


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@needs_bash
def test_download_atlases_never_installs_an_unverified_download(tmp_path):
    # Upstream content changed (or a proxy returned an error page): nothing
    # that fails the pinned checksum may land where preprocessing reads it,
    # and a good local AAL.nii must survive a bad AAL archive.
    dest = tmp_path / "atlases"
    shutil.copytree(ROOT / "atlases", dest)
    (dest / "shen_1mm_268_parcellation.nii.gz").unlink()
    (dest / "aal_SPM12" / "aal" / "atlas" / "AAL.xml").unlink()
    aal_nii = dest / "aal_SPM12" / "aal" / "atlas" / "AAL.nii"
    aal_before = _sha256(aal_nii)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "curl", (
        'out=""; url=""\n'
        'while [ $# -gt 0 ]; do case "$1" in --output) out="$2"; shift ;; '
        'http*) url="$1" ;; esac; shift; done\n'
        'case "$url" in\n'
        '  *.tar.gz) d="$(mktemp -d)"; mkdir -p "$d/aal/atlas"\n'
        '    echo altered > "$d/aal/atlas/AAL.nii"\n'
        '    echo altered > "$d/aal/atlas/AAL.xml"\n'
        '    tar -czf "$out" -C "$d" aal ;;\n'
        '  *) echo altered > "$out" ;;\n'
        'esac\n'
    ))
    proc = _run_script(
        [SCRIPTS / "download_atlases.sh", "--dest", dest],
        env={"PATH": os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")])},
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    shen = "atlases/shen_1mm_268_parcellation.nii.gz"
    assert f"Checksum mismatch for downloaded {shen}" in proc.stderr
    assert not (dest / "shen_1mm_268_parcellation.nii.gz").exists()
    assert _sha256(aal_nii) == aal_before
    assert not (dest / "aal_SPM12" / "aal" / "atlas" / "AAL.xml").exists()


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
@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_download_data_reuses_clone_whose_head_carries_several_tags(tmp_path):
    # OpenNeuro can tag one commit with several snapshots (the local ds002547
    # clone has 1.0.1 and 1.1.0 on HEAD); `git describe` reports only one.
    repo = tmp_path / "ds002547"
    git = ["git", "-C", str(repo), "-c", "user.name=t",
           "-c", "user.email=t@example.org",
           "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "snapshot"],
                   check=True)
    subprocess.run(git + ["tag", "1.0.1"], check=True)
    subprocess.run(git + ["tag", "1.1.0"], check=True)
    proc = _run_script([SCRIPTS / "download_data.sh", "--dry-run", "--no-get",
                        "ds002547", "", repo])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Reusing existing clone at snapshot 1.1.0." in proc.stdout
    proc = _run_script([SCRIPTS / "download_data.sh", "--dry-run", "--no-get",
                        "ds002547", "2.0.0", repo])
    assert proc.returncode == 1
    assert "not snapshot 2.0.0" in proc.stderr


def test_download_data_pins_match_dataset_catalog():
    from impact_pipeline.dataset_catalog import REPORT_DATASETS

    script = (SCRIPTS / "download_data.sh").read_text()
    pins = dict(re.findall(r"^\s+(ds\d{6})\) echo (\S+) ;;$", script, flags=re.M))
    assert pins == {ds: meta.snapshot for ds, meta in REPORT_DATASETS.items()}
    for ds, meta in REPORT_DATASETS.items():
        assert meta.mirror_git_url == f"https://github.com/OpenNeuroDatasets/{ds}.git"


@needs_bash
@pytest.mark.parametrize("snapshot", ["/abs/path/ds005620", "1.0.0/../x", " 1.0.0"])
def test_download_data_rejects_path_like_snapshot(tmp_path, snapshot):
    proc = _run_script([SCRIPTS / "download_data.sh", "--dry-run", "ds005620",
                        snapshot, tmp_path / "d"])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Invalid snapshot tag" in proc.stderr
    assert "clone" not in proc.stdout


@needs_bash
def test_download_data_double_dash_ends_options(tmp_path):
    # run_all.sh passes `--` so that nothing positional is parsed as an option.
    proc = _run_script([SCRIPTS / "download_data.sh", "--dry-run", "--no-get", "--",
                        "ds005620", "", tmp_path / "d"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "refs/tags/1.0.0" in proc.stdout
    proc = _run_script([SCRIPTS / "download_data.sh", "--dry-run", "--",
                        "ds005620", "--no-get", tmp_path / "d"])
    assert proc.returncode == 2
    assert "Invalid snapshot tag: '--no-get'" in proc.stderr


@needs_bash
@pytest.mark.parametrize("dataset", ["ds003171", "ds005620"])
def test_run_all_dry_run_anywhere_executes_nothing(tmp_path, dataset):
    # `run_all.sh ds003171 --dry-run` used to treat --dry-run as the snapshot:
    # the download was dry, but atlases, fMRIPrep and run_pipeline ran for real.
    log = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("docker", "git", "datalad", "curl", "fakepython"):
        _stub(bin_dir, tool, f'echo "{tool} $*" >> "{log}"\nexit 0\n')
    data_root = tmp_path / "scratch"  # not downloaded yet
    license_file = tmp_path / "license.txt"
    license_file.write_text("fake")
    env = {
        "PATH": os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")]),
        "IMPACT_DATA_ROOT": str(data_root),
        "IMPACT_OUT_DIR": str(tmp_path / "out"),
        "IMPACT_PYTHON": str(bin_dir / "fakepython"),
        "FS_LICENSE": str(license_file),
    }
    proc = _run_script([SCRIPTS / "run_all.sh", dataset, "--dry-run"], env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not log.exists(), log.read_text()
    assert "run_pipeline.py" in proc.stdout
    assert "refs/tags/--dry-run" not in proc.stdout
    assert not (tmp_path / "out").exists() and not data_root.exists()


@needs_bash
def test_run_all_rejects_unknown_options(tmp_path):
    proc = _run_script([SCRIPTS / "run_all.sh", "ds005620", "--no-get"],
                       env={"IMPACT_DATA_ROOT": str(tmp_path),
                            "IMPACT_PYTHON": "python"})
    assert proc.returncode == 2
    assert "Unknown option: --no-get" in proc.stderr


@needs_bash
def test_run_all_resolves_relative_env_paths_against_caller(tmp_path):
    proc = _run_script(
        [SCRIPTS / "run_all.sh", "--dry-run", "ds003171"],
        env={"IMPACT_DATA_ROOT": "rel/data", "IMPACT_OUT_DIR": "rel/out",
             "IMPACT_PYTHON": "python"},
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    base = tmp_path.resolve()
    pipeline = shlex.split(next(ln for ln in proc.stdout.splitlines()
                                if "run_pipeline.py" in ln))
    assert pipeline[pipeline.index("--bids-root") + 1] == f"{base}/rel/data/ds003171"
    assert pipeline[pipeline.index("--out-dir") + 1] == f"{base}/rel/out"


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
          'case "$*" in *"tag --points-at HEAD"*) echo 2.0.1 ;; esac\nexit 0\n')
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
    assert any("tag --points-at HEAD" in c for c in calls)
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
