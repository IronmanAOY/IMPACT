#!/usr/bin/env python
"""
Reference anchor of the MPC-Bench construct scale: run (or read) the nominal
positive control on development reference seeds and write the bench protocol
with its external reference. See ``impact_pipeline.bench.reference``.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline.bench.reference import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
