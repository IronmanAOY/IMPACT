"""
MPC-Bench v2: infrastructure shared by the v2 estimators, the v2 runner and
the v2 evaluator.

The v2 code lives in new modules next to the frozen v1 code and imports v1
helpers read-only; nothing here changes a v1 code path, protocol or result.
The v1 path (``evidence``, ``mpc_metrics``, ``bench.run_bench`` and the
``protocols/*_v1*.json`` files) stays selectable and byte-identical, which
``scripts/v2/regression_gate.py`` checks.

Modules of this package (each imported on demand, so importing
``impact_pipeline.v2`` is cheap):

- :mod:`~impact_pipeline.v2.reasons`: the central vocabulary of UNDEFINED
  reasons and of flags. Every reason maps to UNDEFINED, never to ABSENT.
- :mod:`~impact_pipeline.v2.records`: the result schema
  ``mpc-bench-result/3`` (one simulation, several scorings per task).
- :mod:`~impact_pipeline.v2.seeds`: the v2 seed policy (development 0-999,
  confirmatory >= 20000, the v1 block 10000-19999 never reused), the seed map
  of ``protocols/v2/seed_map_v2.json`` and the reserved random-stream keys.
- :mod:`~impact_pipeline.v2.provenance`: run hygiene (code identity by git
  tree, the v2 confirmatory guard, array hashes for the duplicate detector,
  timing with the load average, the environment record).
- :mod:`~impact_pipeline.v2.declared_inputs`: recording device, input
  declarations and the common input basis of NAS v3 and IIM v5.
- :mod:`~impact_pipeline.v2.numerics`: SVD-based linear algebra with a
  fallback driver.
- :mod:`~impact_pipeline.v2.ram_v3`: the RAM-PE v3 estimator.
- :mod:`~impact_pipeline.v2.pdi_v3`: the PDI v3 estimator.
- :mod:`~impact_pipeline.v2.nas_v3`: the NAS v3 estimator.
- :mod:`~impact_pipeline.v2.iim_v5`: the IIM v5 estimator.
- :mod:`~impact_pipeline.v2.registry_v3`: applicability registry v3 and the
  forward-model admission procedure.
- :mod:`~impact_pipeline.v2.testability`: testability gating, anchors and
  necessity sets.
- :mod:`~impact_pipeline.v2.hypothesis_engine`: declarative evaluator of the
  hypotheses in ``protocols/v2/hypotheses_v2.json``.
"""

from __future__ import annotations

RESULT_SCHEMA = "mpc-bench-result/3"
GENERATOR_VERSION_V2 = "mpc-bench-generators/2.0.0"
FREEZE_TAG_V1 = "mpcbench-freeze-v1"
FREEZE_TAG_V2 = "mpcbench-freeze-v2"
PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")

# Estimator versions a v2 record may carry, per principle. The v2 versions
# are frozen with v2; the v1 versions stay admissible as fallbacks (a v1
# estimator under the v2 status rule is always an admissible v2 component).
ESTIMATOR_VERSIONS_V2 = {
    "RAM": ("ram-v3-2026.10",),
    "PDI": ("pdi-v3-2026.10",),
    "NAS": ("nas-v3-2026.10",),
    "IIM": ("iim-v5-2026.10",),
    "SRPI": ("srpi-v2-2026.09", "srpi-v3-2026.10"),
}
ESTIMATOR_VERSIONS_V1 = {
    "RAM": "ram-v2-2026.09",
    "PDI": "pdi-v2-2026.09",
    "NAS": "nas-v2-2026.09",
    "IIM": "iim-v4-2026.09",
    "SRPI": "srpi-v2-2026.09",
}

__all__ = [
    "ESTIMATOR_VERSIONS_V1",
    "ESTIMATOR_VERSIONS_V2",
    "FREEZE_TAG_V1",
    "FREEZE_TAG_V2",
    "GENERATOR_VERSION_V2",
    "PRINCIPLES",
    "RESULT_SCHEMA",
    "provenance",
    "reasons",
    "records",
    "seeds",
]
