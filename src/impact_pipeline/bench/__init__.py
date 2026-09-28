"""
MPC-Bench: white-box validation systems for the IMPaCT estimators.

Submodules are imported lazily by callers; importing the package does not
import the metric code (``impact_pipeline.mpc_metrics``), so the generators
stay independent of the estimators they validate.

Generators: ``generators`` (families A, B, C), ``patchwork`` (incl. graded
patchworks), ``adversarial`` (constructions designed to fool one estimator),
``whole_brain`` (Hopf model on the empirical connectome) and ``forward``
(EEG-like and BOLD-like observation models). Validation and analysis:
``manipulation`` (preregistered oracle manipulation checks), ``export`` (the
synergy_ci layout and the in-memory estimator runner), ``rules`` (rival
attribution rules and the v2 construct-scale rule), ``audit`` (rule audit on
estimated statuses), ``run_bench`` (runner and development / confirmatory
split) and ``compat`` (evidence-layer verdict names, v1 and v2).
"""

BENCH_VERSION = "2.0.0"
