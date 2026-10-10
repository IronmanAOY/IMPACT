"""
MPC-Bench: white-box validation systems for the IMPaCT estimators.

Submodules are imported lazily by callers; importing the package does not
import the metric code (``impact_pipeline.mpc_metrics``), so the generators
stay independent of the estimators they validate.

Round v1
--------
Generators: ``generators`` (families A, B, C), ``patchwork`` (incl. graded
patchworks), ``adversarial`` (constructions designed to fool one estimator),
``whole_brain`` (Hopf model on the empirical connectome) and ``forward``
(EEG-like and BOLD-like observation models). Designs: ``factorial`` (the 2^5
mechanism cells), ``sweeps`` (dose-response sweeps) and ``witnesses`` (loader
of the witness catalogue ``witnesses.yaml``). Validation and analysis:
``manipulation`` (preregistered oracle manipulation checks), ``export`` (the
synergy_ci layout and the in-memory estimator runner), ``reference`` (the
reference anchor of the construct scale), ``analysis`` (construct-scale
assessment, cutoff grids and dose-response), ``rules`` (rival attribution
rules and the construct-scale rule of the rule audit), ``audit`` (rule audit
on estimated statuses), ``lz`` and ``phiid_gaussian`` (the LZc and Gaussian
Phi_R markers), ``run_bench`` (runner and development / confirmatory split)
and ``compat`` (verdict names, including those of older result files).

MPC-Bench v2
------------
``designs_v2`` (designs, task vocabulary and run-plan tables),
``run_bench_v2`` (runner, records of schema ``mpc-bench-result/3``),
``adversarial_v2`` (catalogue and builder of the v2 witnesses and adversaries,
``witnesses_v2.yaml``), ``forward_v2`` (observation views of the forward-model
arms) and ``manipulation_v2`` (realisation checks of the v2 systems).
"""

BENCH_VERSION = "2.0.0"
