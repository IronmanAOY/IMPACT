# -*- coding: utf-8 -*-
# tests/conftest.py
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The tests must exercise this checkout's src/. If impact_pipeline is not
# importable, or resolves to another copy (a different checkout or worktree
# installed into the same env, or a stale site-packages install), put this
# checkout's src/ first. An editable install of this checkout already resolves
# here and is left alone. Set IMPACT_TEST_INSTALLED_PACKAGE=1 to test an
# installed copy (e.g. a built wheel) instead. The repository root (for
# run_pipeline.py and scripts/) is added by pytest's `pythonpath` setting in
# pyproject.toml.
REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"
_spec = importlib.util.find_spec("impact_pipeline")
_origin = Path(_spec.origin).resolve() if _spec is not None and _spec.origin else None
if os.environ.get("IMPACT_TEST_INSTALLED_PACKAGE") != "1" and (
    _origin is None or _SRC not in _origin.parents
):
    sys.path.insert(0, str(_SRC))

# Unit tests import and call run_pipeline.main() directly from arbitrary
# interpreters. Skip strict runtime env checks in test context.
os.environ.setdefault("IMPACT_SKIP_ENV_CHECK", "1")


@pytest.fixture
def clean_hunter_env(monkeypatch, tmp_path):
    """Hide the caller's Hunter, IIM and PBS environment from a test and give
    it a private IIM cache directory."""
    for name in list(os.environ):
        if name.startswith(("IMPACT_HUNTER_", "IMPACT_IIM_", "PMI_LOCAL_RANK", "PBS_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("IMPACT_REPO_ROOT", raising=False)
    monkeypatch.setenv("IMPACT_IIM_CACHE_DIR", str(tmp_path / "node_local"))


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


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
    """A throw-away git repository for the confirmatory-guard tests (identity
    passed per command, no configuration is written)."""
    return GitRepo(tmp_path / "repo")
