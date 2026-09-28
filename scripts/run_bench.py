#!/usr/bin/env python
"""
MPC-Bench command line (thin wrapper around ``impact_pipeline.bench.run_bench``).

Examples:
    python scripts/run_bench.py timing --seeds 0-2
    python scripts/run_bench.py factorial --seeds 0-19 --workers 8 \
        --null-surrogates 20 --out outputs/bench/factorial_A
    python scripts/run_bench.py sweep --knobs eta,g_b --seeds 0-19 \
        --out outputs/bench/sweep
    python scripts/run_bench.py witnesses --seeds 0-19 --out outputs/bench/witnesses
    python scripts/run_bench.py factorial --family C --seeds 10000-10019 \
        --confirmatory --freeze-tag <tag> --out <workspace>/bench/factorial_C
    python scripts/run_bench.py factorial --seeds 0-19 --n-shards 8 \
        --pbs-template bench_factorial.pbs
    python scripts/run_bench.py patchwork_sweep --seeds 0-19 --levels 6 \
        --out outputs/bench/patchwork_sweep
    python scripts/run_bench.py manipulation --seeds 0-4 --out outputs/bench/manip
    python scripts/run_bench.py adversarial --seeds 10000-10019 \
        --confirmatory --freeze-tag <tag> --out <workspace>/bench/adversarial
    python scripts/run_bench.py whole_brain --observations source,eeg,bold \
        --seeds 10000-10004 --confirmatory --freeze-tag <tag> \
        --out <workspace>/bench/whole_brain
Add ``--se-groups 10`` (the frozen protocol's G) for jackknife component SEs (needed by the v2 rule
audit, scripts/benchmark_attribution_rules.py, and by determinate verdicts).
Verdicts use ``--protocol`` (default: protocols/mpc_bench_v1.json with its
positive-control reference anchor, see scripts/bench_reference.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline.bench.run_bench import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
