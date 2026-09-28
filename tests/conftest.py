# -*- coding: utf-8 -*-
# tests/conftest.py
import importlib.util
import os
import sys
from pathlib import Path

# The package is normally importable because it is installed
# (`pip install -e .`). From a plain source checkout without installation,
# fall back to src/ so the suite still runs. The repository root (for
# run_pipeline.py and scripts/) is added by pytest's `pythonpath` setting in
# pyproject.toml.
if importlib.util.find_spec("impact_pipeline") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Unit tests import and call run_pipeline.main() directly from arbitrary
# interpreters. Skip strict runtime env checks in test context.
os.environ.setdefault("IMPACT_SKIP_ENV_CHECK", "1")
