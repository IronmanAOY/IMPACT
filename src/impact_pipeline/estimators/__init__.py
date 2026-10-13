"""
Benchmark variants of the MPC estimators, next to the estimators of
:mod:`impact_pipeline.mpc_metrics` (their versions are listed together in
:data:`impact_pipeline.evidence.ESTIMATOR_VERSIONS`).

Modules (each imported on demand):

- :mod:`~impact_pipeline.estimators.ram_pe`: RAM-PE, the prediction-error
  readout of the plasticity channel of RAM.
- :mod:`~impact_pipeline.estimators.pdi_declared`: PDI on the content bearer
  of declared access nodes.
- :mod:`~impact_pipeline.estimators.nas_declared`: NAS between a declared hub
  and declared periphery blocks, beyond declared inputs.
- :mod:`~impact_pipeline.estimators.iim_macro`: IIM on 2 to 6 declared macro
  nodes.
- :mod:`~impact_pipeline.estimators.declared_inputs`: recording device, input
  declarations and the common input basis of NAS and IIM.
"""
