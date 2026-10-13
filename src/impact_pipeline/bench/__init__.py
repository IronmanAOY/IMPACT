"""
MPC-Bench: white-box validation systems for the IMPaCT estimators.

Submodules are imported lazily by callers; importing the package does not
import the metric code (``impact_pipeline.mpc_metrics``), so the generators
stay independent of the estimators they validate.

Systems: ``generators`` (families A, B, C, nulls), ``patchwork`` (incl.
graded patchworks), ``adversarial`` (constructions designed to fool one
estimator, incl. the staggered hidden driver), ``whole_brain`` (Hopf model on
the empirical connectome), ``forward`` (EEG-like and BOLD-like observation
models and the views of the forward-model arms) and ``catalogue`` (loader,
validator and builder of the system catalogue ``systems.yaml``).
Validation: ``manipulation`` (oracle switch and realisation checks) and
``export`` (the synergy_ci layout and the in-memory estimator path).
Runs: ``designs`` (designs, task vocabulary and run-plan tables),
``runner`` (runner), ``records`` (result records), ``seeds`` (seed
policy and random streams), ``run_hygiene`` (code identity and the
confirmatory guard) and ``hypotheses`` (declarative hypothesis evaluator).
"""

BENCH_VERSION = "1.0.0"
