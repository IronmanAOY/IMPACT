"""Hunter helper scripts (scripts/hunter) and pinned requirement files."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HUNTER = REPO / "scripts" / "hunter"
SCRIPTS = ("hunter_pbs_setup.sh", "install_hunter_env.sh", "hunter_smoke_test.sh")


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_are_valid_bash(name):
    path = HUNTER / name
    assert path.read_text().startswith("#!/bin/bash\n")
    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_setup_template_follows_hunter_policy():
    text = (HUNTER / "hunter_pbs_setup.sh").read_text()
    assert "module load cray-python" in text
    assert "ws_find" in text
    assert "venvs/impact-hunter" in text and "IMPACT_HUNTER_PYTHON" in text
    assert "CUPY_CACHE_DIR" in text
    # sourced before `set -u` by the job scripts; it must not change shell options
    assert "set -u" not in text.replace("`set -u`", "")
    assert "conda" not in text


def test_setup_failure_returns_when_sourced_and_aborts_jobs(tmp_path):
    """The setup file is sourced: an error must not close a login shell."""
    stub = (
        "module() { :; }\n"
        "ws_find() { echo ''; }\n"
        f"source {HUNTER / 'hunter_pbs_setup.sh'}\n"
    )
    interactive = subprocess.run(
        ["bash", "-c", stub + 'echo "alive rc=$?"'],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "alive rc=1" in interactive.stdout
    assert "workspace 'impact' not found" in interactive.stderr
    job = subprocess.run(
        ["bash", "-c", "set -eo pipefail\n" + stub + "echo continued"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert job.returncode != 0 and "continued" not in job.stdout


def test_login_node_commands_run_from_the_sourced_venv(tmp_path):
    """
    After `source hunter_pbs_setup.sh`, the documented login-node commands
    (build-campaign, status) run from the cray-python venv, whose path does not
    contain the workstation conda env name.
    """
    ws = tmp_path / "ws"
    venv = ws / "venvs" / "impact-hunter"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            "--without-pip",
            str(venv),
        ],
        check=True,
        timeout=120,
    )
    script = (
        "module() { :; }\n"
        f"export IMPACT_WS={ws}\n"
        f"source {HUNTER / 'hunter_pbs_setup.sh'}\n"
        'exec "$IMPACT_HUNTER_PYTHON" '
        f"{REPO / 'run_pipeline.py'} --execution-mode hunter --hunter-stage status "
        f"--hunter-campaign-dir {tmp_path / 'no_campaign'} "
        f"--out-dir {tmp_path / 'out'}\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"CONDA_DEFAULT_ENV", "CONDA_PREFIX", "IMPACT_SKIP_ENV_CHECK"}
        and not k.startswith("IMPACT_CONDA_ENV")
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=180, env=env
    )
    assert "Invalid Python runtime" not in proc.stderr, proc.stderr[-1500:]
    # it got past the runtime check and failed only on the missing campaign
    assert "campaign_manifest.json" in proc.stderr


def test_install_notes_default_to_dry_run():
    proc = subprocess.run(
        ["bash", str(HUNTER / "install_hunter_env.sh"), "--cupy", "source-13.6"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "mode: print" in out
    assert "python3 -m venv --system-site-packages" in out
    assert '-c "' in out and "constraints-hunter.txt" in out
    assert "CUPY_INSTALL_USE_HIP=1" in out and "HCC_AMDGPU_TARGET=gfx942" in out
    assert "cupy==13.6.0" in out
    assert "qsub" not in out.split("# 8.")[0]  # nothing is submitted by the installer


def test_requirement_files_pin_cray_numpy_and_exclude_dev_tools():
    constraints = (REPO / "constraints-hunter.txt").read_text()
    assert "numpy==1.24.4" in constraints and "scipy==1.10.1" in constraints
    reqs = [
        line.split("==")[0].strip().lower()
        for line in (REPO / "requirements-hunter.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert {"mne", "nilearn", "pybids", "numba", "statsmodels", "python-docx"}.issubset(
        reqs
    )
    assert not {
        "numpy",
        "scipy",
        "cupy",
        "pytest",
        "flake8",
    } & set(reqs)
    for line in (REPO / "requirements-hunter.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            assert "==" in line, line  # fully pinned


def test_smoke_helper_builds_a_tiny_campaign(tmp_path):
    synth_root = tmp_path / "synth"
    bids = synth_root / "test_objects" / "datasets" / "ds003171"
    out = synth_root / "test_objects" / "runs" / "ds003171"
    bids.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps(
            {"Name": "synthetic", "SyntheticData": True, "DatasetType": "synthetic"}
        )
    )
    rng = np.random.RandomState(0)
    for subj in ("01", "02"):
        (bids / f"sub-{subj}" / "func").mkdir(parents=True)
        for ses in ("awake", "deep"):
            d = out / "preprocessed" / subj / ses / "audio"
            d.mkdir(parents=True)
            np.save(d / f"{subj}_run-1_schaefer400_ts.npy", rng.rand(40, 6))
    env = {
        **os.environ,
        "IMPACT_SYNTH_ROOT": str(synth_root),
        "IMPACT_SKIP_ENV_CHECK": "1",
    }
    env.pop("IMPACT_HUNTER_SETUP_FILE", None)
    proc = subprocess.run(
        [
            "bash",
            str(HUNTER / "hunter_smoke_test.sh"),
            "--bids-root",
            str(bids),
            "--out-dir",
            str(out),
            "--subjects",
            "01",
            "--hardware-target",
            "cpu",
            "--python",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    campaign = out / "cache" / "hunter_iim_smoke"
    manifest = json.loads((campaign / "campaign_manifest.json").read_text())
    assert {r["subject"] for r in manifest["runs"]} == {"01"}
    assert all(r["n_nodes_used"] == 4 for r in manifest["runs"])
    assert (campaign / "pbs" / "90_smoke_all_in_one.pbs").exists()
    assert "qsub" in proc.stdout
    assert "IMPACT_HUNTER_SETUP_FILE is not set" in proc.stderr


def test_requirements_match_the_cray_python_route_and_document_2026_2():
    reqs = {
        line.split("==")[0].strip().lower(): line.split("==")[1].strip()
        for line in (REPO / "requirements-hunter.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    # numba 0.61.x supports numpy < 2.3 (stack 2026.1: numpy 1.24.4)
    assert reqs["numba"].startswith("0.61.")
    # CuPy's runtime dependency, so a locally built CuPy installs with --no-deps
    assert "fastrlock" in reqs
    constraints = (REPO / "constraints-hunter.txt").read_text()
    for token in ("testing-2026.2", "numpy==2.3.5", "scipy==1.16.3", "numba==0.62.1"):
        assert token in constraints
    smoke = (HUNTER / "hunter_smoke_test.sh").read_text()
    assert "--hunter-iim-null-surrogates" in smoke
    readme = (HUNTER / "README.md").read_text()
    assert "iim_psi_xp_parity" in readme and "--hunter-iim-null-surrogates" in readme
