"""
MPC-Bench: white-box validation systems for the IMPaCT estimators.

Submodules are imported lazily by callers; importing the package does not
import the metric code (``impact_pipeline.mpc_metrics``), so the generators
stay independent of the estimators they validate.
"""

BENCH_VERSION = "1.0.0"
