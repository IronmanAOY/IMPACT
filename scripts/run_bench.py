#!/usr/bin/env python
"""
MPC-Bench command line (thin wrapper around ``impact_pipeline.bench.runner``).

Examples:
    python scripts/run_bench.py designs
    python scripts/run_bench.py plan --split confirmatory
    python scripts/run_bench.py list A_witnesses --split development
    python scripts/run_bench.py run A_anchors,C_anchors --split development \
        --workers 12 --out outputs/mpcbench/development/anchors
    python scripts/run_bench.py run A_witnesses --split development \
        --seeds 320-323 --systems PC_nominal --principles RAM,PDI,SRPI \
        --out outputs/mpcbench/development/check
    python scripts/run_bench.py manipulation --seeds 0-4 \
        --out outputs/mpcbench/development/manipulation
    python scripts/run_bench.py run A_witnesses --split confirmatory \
        --confirmatory --freeze-tag <tag> --workers 12 \
        --out outputs/mpcbench/confirmatory/A_witnesses

Development runs use seeds 0-999 and may use draft protocols derived from
the template; confirmatory runs need a freeze tag (the git tag of the code
fixed before the run), a clean tree, seeds >= 20000 and the family protocol
files (protocols/bench/). Held-out conditions run on development seeds only
as smoke tests (seeds 980-984), whose outputs are discarded unread.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# One BLAS / OpenMP thread, as in the spawned workers, unless set explicitly
# (before numpy is imported).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline.bench.runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
