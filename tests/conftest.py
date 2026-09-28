# -*- coding: utf-8 -*-
# tests/conftest.py
import importlib.util
import os
import sys
from pathlib import Path

# The tests must exercise this checkout's src/. If impact_pipeline is not
# importable, or resolves to another copy (a different checkout or worktree
# installed into the same env, or a stale site-packages install), put this
# checkout's src/ first. An editable install of this checkout already resolves
# here and is left alone. Set IMPACT_TEST_INSTALLED_PACKAGE=1 to test an
# installed copy (e.g. a built wheel) instead. The repository root (for
# run_pipeline.py and scripts/) is added by pytest's `pythonpath` setting in
# pyproject.toml.
_SRC = Path(__file__).resolve().parent.parent / "src"
_spec = importlib.util.find_spec("impact_pipeline")
_origin = Path(_spec.origin).resolve() if _spec is not None and _spec.origin else None
if os.environ.get("IMPACT_TEST_INSTALLED_PACKAGE") != "1" and (
    _origin is None or _SRC not in _origin.parents
):
    sys.path.insert(0, str(_SRC))

# Unit tests import and call run_pipeline.main() directly from arbitrary
# interpreters. Skip strict runtime env checks in test context.
os.environ.setdefault("IMPACT_SKIP_ENV_CHECK", "1")
