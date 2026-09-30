# -*- coding: utf-8 -*-
"""
Shared fixtures of the v2 tests.

* ``slow`` marker: tests that re-run stored v1 tasks from scratch (the full
  v1 regression gate). They run only with ``MPCBENCH_RUN_SLOW=1``.
* ``work_plan``: the v2 work plan (``work_packages.json``); the test is
  skipped where the plan is not available (it is not versioned).
* ``v1_outputs``: the stored v1 outputs (``outputs/paper1_mpcbench``); the
  test is skipped where they are not available (for example on CI).
* ``git_repo``: a throw-away git repository for guard tests (identity passed
  per command, no configuration is written).
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SLOW_ENV = "MPCBENCH_RUN_SLOW"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"slow: re-runs stored v1 tasks from scratch; runs only with {RUN_SLOW_ENV}=1",
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("slow") and os.environ.get(RUN_SLOW_ENV) != "1":
        pytest.skip(f"slow test; set {RUN_SLOW_ENV}=1 to run it")


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def work_plan():
    from impact_pipeline.v2 import ownership as own

    path = own.locate_work_packages(REPO_ROOT)
    if path is None:
        pytest.skip("v2 work plan (work_packages.json) not available")
    return own.load_work_packages(path)


@pytest.fixture(scope="session")
def v1_outputs():
    from scripts.v2 import regression_gate as G

    out = G.locate_v1_outputs()
    if out is None:
        pytest.skip("stored v1 outputs (outputs/paper1_mpcbench) not available")
    return out


class GitRepo:
    """A scratch git repository; commits pass their identity per command."""

    IDENTITY = ("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false")

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q")

    def git(self, *args, check=True) -> str:
        env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_OPTIONAL_LOCKS": "0"}
        proc = subprocess.run(["git", *self.IDENTITY, *args], cwd=self.root,
                              capture_output=True, text=True, env=env, check=False)
        if check and proc.returncode != 0:
            raise RuntimeError(f"git {args}: {proc.stderr}")
        return proc.stdout.strip()

    def write(self, rel: str, text: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def git_repo(tmp_path):
    return GitRepo(tmp_path / "repo")
