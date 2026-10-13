# Architecture

This document describes how the IMPaCT Synergy Pipeline 1.2.0 is organised:
the layers, the modules, the data flow of a run, the HLRS Hunter campaign, the
evidence layer, the benchmark MPC-Bench and the design rules the code follows.
Definitions of the estimators, the evidence layer and the protocols are in
[`metrics.md`](metrics.md); the Hunter procedure is in
[`HLRS_HUNTER_RUNBOOK.md`](HLRS_HUNTER_RUNBOOK.md).

## 1. Layers

```text
 entry points      run_pipeline.py · scripts/*.py|sh · python -m impact_pipeline.hardware_selftest
      │
 orchestration     run_pipeline.main (steps 0-9) · hunter_iim (campaign, stages, PBS writers)
      │
 assembly          run_synergy_ci.run_s_ci · synergy_ci.compute_synergy_ci (per-run metrics,
                   null calibration, evidence wiring, legacy CI, MPC degree)
      │
 evidence          evidence (interval and TOST rules, Kleene logic, verdict, degree) · registry · nulls
      │
 estimators        mpc_metrics (RAM, PDI, NAS, IIM, SRPI, legacy CI) · iim_xp (Ψ kernel) · lz
                   estimators.* (benchmark variants and declared inputs)
      │
 infrastructure    hardware_backend (NumPy/CuPy) · event_parsing · provenance · execution_profiles
```

Each layer only calls the layers below it. `impact_pipeline.bench` (MPC-Bench)
sits beside the pipeline: its generators never import `mpc_metrics`; its runner
calls the public estimators and the evidence layer like any other user.

## 2. Modules

| Module | Responsibility |
|---|---|
| `run_pipeline.py` | CLI and `main()`: dataset configuration (`DATASET_CONFIGS`), output resolution, steps 0-9, Hunter stage dispatch, provenance manifest |
| `impact_pipeline.mpc_metrics` | the estimators `compute_RAM`, `compute_RAM_by_channel`, `compute_PDI`, `compute_NAS`, `compute_IIM`, `compute_IIM_from_tpm`, `compute_SRPI`; surrogate generation for their own null calibration; `prepare_iim_problem`; legacy `compute_CI` (deprecated) |
| `impact_pipeline.estimators` | benchmark variants of the estimators: `ram_pe` (RAM-PE), `pdi_declared` (declared PDI), `nas_declared` (declared NAS), `iim_macro` (macro IIM); `declared_inputs` (recording device, input declarations, common input basis) |
| `impact_pipeline.iim_xp` | array-module (NumPy/CuPy) Ψ contribution and cut-TPM kernels |
| `impact_pipeline.lz` | the LZ76 phrase count shared by `mpc_metrics` and `baseline_metrics` |
| `impact_pipeline.nulls` | generic surrogates (circular shift, phase randomisation, IAAFT, label permutation, onset jitter), `component_null` and the moving-block bootstrap |
| `impact_pipeline.evidence` | `ComponentEvidence`, `Protocol` (schema `impact-mpc-protocol/1`), the interval rule (`component_assessment`, `interval_verdict`), the TOST rule (`assess_item`, `tost_verdict`), `mpc_verdict` (the protocol's rule), strong-Kleene logic, reason codes, two-anchor scale, estimator versions, MPC degree and intervals, joint dependence, `bearer_coherence` |
| `impact_pipeline.registry` | applicability registry (`ApplicabilityRegistry`, schema `impact-mpc-registry/1`): validation entries and the forward-model admission entries of the benchmark |
| `impact_pipeline.reasons`, `testability`, `numerics` | the vocabulary of UNDEFINED reasons and flags; testability gating, protocol anchors and necessity sets; SVD-based linear algebra with a fallback driver |
| `impact_pipeline.synergy_ci` | `compute_synergy_ci` (per-run loop over the preprocessed tree), `assemble_ci`, `assemble_mpc_degree`, CI reference resolution, IIM worker planning |
| `impact_pipeline.run_synergy_ci` | `run_s_ci` (step 2 driver: events, RAM presets, subject means, S by theta), `load_onsets`, `RAM_PARAM_PRESETS` |
| `impact_pipeline.event_parsing` | the single source for `events.tsv` parsing: label patterns, session-exact file resolution, event bundles, SRPI-agency contract |
| `impact_pipeline.mpc_readiness` | per-run readiness report (which metrics can be defined, and why not), using `event_parsing`; NAS against the run's protocol (capacity without a declared hub: `NO_DECLARED_WORKSPACE`, the rule of `synergy_ci.nas_hub_missing`; a declared hub that does not fit the recording's node count: `INVALID_WORKSPACE`) |
| `impact_pipeline.preprocessing`, `preprocessing_eeg` | fMRI atlas time series from fMRIPrep derivatives; EEG channel time series from BrainVision; rest baselines |
| `impact_pipeline.dataset_catalog` | dataset metadata, task/state grammar, explicit per-subject task aliases |
| `impact_pipeline.analysis_bootstrap`, `model_comparison`, `baseline_metrics`, `motion_model`, `atlas_robustness`, `replication` | steps 3-8: paired statistics (bootstrap, permutation, DeLong, Holm), baseline comparators, motion covariates, robustness atlases, Melbourne replication |
| `impact_pipeline.necessity` | symmetric three-outcome necessity criteria, verdict-level summaries and Clopper-Pearson bounds; NCA ceilings and fsQCA necessity |
| `impact_pipeline.generate_word_doc` | step 9: Word report |
| `impact_pipeline.hunter_iim`, `hunter_cost` | Hunter campaign: preparation, shard/reduce/finalize stages, idempotency, timing, PBS script writers; the cost model and size guard of the campaign |
| `impact_pipeline.execution_profiles` | `local` and `hunter` profiles, `HunterPBSProfile` |
| `impact_pipeline.hardware_backend` | target resolution (`cpu`, `auto`, `gpu`, `hunter-apu`), CuPy loading, thread limits, accelerated linear algebra with the eigh parity check |
| `impact_pipeline.hardware_selftest` | CuPy-against-NumPy self-test (`python -m`) |
| `impact_pipeline.provenance` | data origin (`real`/`dummy`) and output routing, code version, runtime versions, repository root |
| `impact_pipeline.utils` | HypergraphSynergy (the exploratory statistic S) and helpers |
| `impact_pipeline.bench.generators`, `patchwork`, `whole_brain`, `forward`, `adversarial`, `catalogue` | MPC-Bench systems: families A, B and C, null and hypersynchronous systems, patchworks, the whole-brain Hopf model, forward models and views, adversarial constructions, the system catalogue `systems.yaml` |
| `impact_pipeline.bench.manipulation`, `export` | oracle switch and realisation checks; the synergy_ci export layout and the in-memory estimator path |
| `impact_pipeline.bench.designs` | the designs and run plans (families A, B and C, the RAM-only arm, twins, anchors, forward-model arms, null calibration) |
| `impact_pipeline.bench.runner`, `records`, `seeds`, `run_hygiene` | the runner (one simulation, several scorings, each estimator in its own try block); result schema `mpc-bench-result/1`; seed split and random streams; code identity by git tree and the confirmatory guard |
| `impact_pipeline.bench.hypotheses` | declarative hypothesis evaluator (schema `mpc-bench-hypotheses/1`) |
| `protocols/` | declared MPC protocols (JSON, hashed): `mpc_default.json` (the default of empirical runs; RAM is never ABSENT under it; no NAS hub, so NAS is UNDEFINED, `NO_DECLARED_WORKSPACE`, unless a derived protocol declares one), the opt-in [`mpc_behavioural_ram.json`](../protocols/README.md#mpc_behavioural_ramjson-opt-in), `examples/` (derived from the default with a declared NAS hub) and `bench/` (the benchmark protocols); [`protocols/README.md`](../protocols/README.md) gives the reasons for each declaration |
| `scripts/run_bench.py`, `null_calibration.py`, `iim_validation.py`, `evaluate_hypotheses.py` | MPC-Bench: runner command line (designs, run plans, runs, manipulation checks), null calibration, family-B IIM validation, the declarative hypothesis evaluator |
| `scripts/definedness_audit.py`, `compute_empirical_reference.py`, `build_example_protocols.py` | definedness audit, external empirical reference anchor, example derived protocols (NAS hub) |
| `scripts/live_dashboard.py`, `impact_desktop_app.py`, `powermetrics_telemetry.py` | browser dashboard, desktop launcher, macOS thermal telemetry for the dashboard |
| `scripts/generate_real_derived_synth_completed.py`, `inspect_real_sources_for_synth.py` | real-data-derived synthetic smoke-test objects |
| `scripts/download_data.sh`, `download_atlases.sh`, `fetch_fmriprep.sh`, `run_all.sh` | data, atlases, fMRIPrep, end-to-end local run |
| `scripts/hunter/` | Hunter setup-file template, install notes, smoke-test helper |

## 3. A local run

```text
 BIDS dataset ──(0) fMRIPrep (optional, Docker)──► <bids-root>/derivatives/fmriprep
      │
      └──(1) preprocessing (--run-preprocessing)──► <out>/preprocessed/<subj>/<ses>/<cond>/*_ts.npy
                                                    <out>/preprocessed/<subj>/<ses>/rest/*_ts.npy
 (2) run_s_ci → compute_synergy_ci, per run:
       events.tsv ─► event_parsing ─► RAM / SRPI bundles
       time series ─► RAM, PDI, NAS, IIM, SRPI  (modes from --protocol, default
                      protocols/mpc_default.json on real data; + K null surrogates
                      each when --null-surrogates K; + B bootstrap replicates when --bootstrap-se B)
                  ─► ComponentEvidence per principle ─► evidence.mpc_verdict ─► MPC_* columns
                  ─► S for every theta (exploratory)
     then, per table: assemble_ci (legacy CI) and assemble_mpc_degree (MPC_CONSISTENT rows)
       ─► <out>/cache/step2_df.csv, step2_df_mean.csv, step2_theta_stats.csv
 (3) baseline graph metrics   (4) paired statistics   (5) motion covariates (fMRI)
 (6) robustness atlases (fMRI) (7) replication (optional)  (8) model comparison
       ─► <out>/stats/*.json|csv, <out>/pipe_figures/
 (9) Word report ─► <out>/IMPaCT_Empirical_Validation_<dataset>.docx
 provenance manifest ─► <out>/cache/provenance_manifest.json (status started → completed)
```

- Output routing (`provenance.resolve_dataset_provenance`): real data write to
  `--out-dir` (plus a `<dataset>` subfolder for datasets other than ds003171);
  `--data-origin dummy` (synthetic) routes everything under `test_objects/`, and
  a synthetic dataset is refused as real.
- Step 2 is the only step that computes the estimators. `--reuse-step2` reads its
  tables back and reruns steps 3-9.
- Per-run seeds of the null families are derived from the base seed, the
  component and the run path, so a run gives the same result wherever the data
  live.

## 4. The Hunter campaign

`--execution-mode hunter` replaces the IIM part of step 2 by a campaign of PBS
jobs; everything else is computed by the finalize stage on one node.

```text
 build-campaign (login node)
   size guard (hunter_cost): Ψ evaluations of the campaign, closed form from each run's region
     count, and an order-of-magnitude runtime; above the ceiling HunterCostError, nothing written
   prepare_hunter_campaign: for every run (and every IIM surrogate run):
     node selection, discretisation, TPM ─► runs/<run>/{tpm_full,states_full,curr_obs}.npy,
     mechanisms.json, purviews.json, cuts.json, meta.json
   task lists: phase-1 tasks (mechanism chunks) and cut tasks (cut slices)
   write_hunter_scheduler_scripts ─► pbs/00_submit_all.sh, 01..03 *.pbs, 90_smoke_all_in_one.pbs,
                                     campaign_plan.json
 01 phase1-shard  (array; 4 ranks/node; shard = 4*PBS_ARRAY_INDEX + PMI_LOCAL_RANK)
     Ψ contributions of a mechanism chunk of the intact TPM ─► runs/<run>/phase1_shards/shard_NNNN.json
 02 cut-shard     (array, concurrent with 01)
     Ψ of a slice of system cuts, checkpoint per cut ─► runs/<run>/cut_shards/shard_NNNN.json
 03 reduce-all + finalize-pipeline (one job, afterok on 01 and 02)
     reduce: Ψ_full, max_κ Ψ^κ, ΔΨ, IIM null calibration from the surrogate runs
       ─► runs/<run>/final_result.json, iim_results.json/.csv, cost_calibration.json
     finalize: run_s_ci with the campaign's IIM results (iim_precomputed_by_path), steps 3-9
       ─► the same outputs as a local run, plus cache/hunter_iim_results.csv, timing_summary.json
```

Design points:

- **Equivalence with the local path.** The campaign uses the same problem
  preparation, the same Ψ definitions and the same calibration function as
  `compute_IIM`; the tests compare the results to 1e-9, including the surrogate
  calibration (same `RandomState` stream, same minimum shift, same cut sample).
- **Identity and idempotency.** Every shard result carries an identity (problem
  digest, shard range, code version). A shard whose result matches is skipped;
  reducers refuse mixed code versions or results from another build; a rebuild
  removes the previous reduced results. `IMPACT_HUNTER_FORCE=1` recomputes.
- **Packing.** Nodes are allocated and charged whole, so 4 shards run per
  `mi300a` node, one per APU, with PALS CPU/GPU binding. Arrays need at least 2
  subjobs; a single-node stage is a plain job.
- **Kernels.** On accelerator targets the shards use the `iim_xp` kernel on the
  device (one process per APU); on `cpu` the numba kernel with worker processes
  and node-local SQLite kernel caches.
- **Scheduler backend.** PBS Pro (HLRS Hunter) is the only backend.
- **Environment.** Generated scripts start with `#!/bin/bash`, source the site
  setup file before `set -u`, `cd` into the checkout recorded at build time and
  export `IMPACT_REPO_ROOT`. `qsub` does not forward the login environment.

## 5. The evidence layer in the data flow

```text
 estimator details ──► _component_record (estimate, null moments, bootstrap SE, reason,
                                           estimator id)
                   ──► reference anchors of the protocol (cohort high state or external)
                   ──► ComponentEvidence (+ se_df, reference, bearer_id, protocol_id,
                                           substrate = modality, grain = atlas, nodes, regime)
                   ──► ApplicabilityRegistry.is_validated (optional)
                   ──► construct-scale status per channel ─► Kleene OR per principle
                   ──► Kleene AND over the necessity set (+ single-source check)
                                                         ─► verdict + reasons
                   ──► MPC degree (MPC_CONSISTENT rows; construct scale; capped power mean)
```

- Missing evidence is `MISSING`/UNDEFINED, never 0, and can only make the verdict
  `UNDETERMINED`. The verdict is recoverable from its reason codes.
- The evidence layer does not depend on how an estimate was obtained: the
  pipeline, the Hunter finalize stage and MPC-Bench build the same
  `ComponentEvidence` objects.
- The protocol selects the status rule: without a `status_rule` block the
  interval rule (the empirical pipeline, which refuses the other kind), with
  the block the TOST rule (the benchmark protocols in `protocols/bench/`).

## 6. MPC-Bench

```text
 catalogue (systems.yaml) / generators / patchwork / whole_brain / adversarial
     ──► BenchSystem(ts, events, meta, oracle)         oracle stored apart, never read by estimators
           │                                           manipulation: switch and realisation checks
           ├─ forward.observe ──► views (source, eeg64, eeg64_noref, eeglow, mne_template, bold)
           └─ declared_inputs.record_inputs ──► declarations (R, H, P, Q10, Q25, J, none)
 designs (task specs, run plans) ──► runner.run_tasks (process pool, resume, shards):
     one simulation per task, several scorings (declaration, view, estimator form, protocol);
     each principle dispatched by the estimator version of its family protocol:
       ram_pe, pdi_declared, nas_declared, iim_macro, or the in-memory mpc_metrics path
     each estimator in its own try block ─► evidence (TOST rule, protocols/bench/<key>.json)
       ─► results.jsonl (mpc-bench-result/1), components.csv, run_plan.json, run_manifest.json
 records + hypotheses file ──► scripts/evaluate_hypotheses.py (bench.hypotheses)
```

- **Seed split** (`bench.seeds`): development seeds 0-999, confirmatory seeds
  20000 and above; the seeds in between are refused and one run never mixes
  the two splits.
- **Confirmatory guard** (`bench.run_hygiene.confirmatory_guard`): a run
  with `--confirmatory --freeze-tag <tag>` is refused for a dirty tree, a tag
  that is not given or not found, and `src/` or `scripts/` trees that differ
  from those of the tag.
- **Family B** has its own task model: its cells run through
  `scripts/iim_validation.py`, which builds the family-B protocols in code.

## 7. Design rules

1. **Measured inputs only.** No silent fallbacks (TR, events, baselines,
   feedback). A quantity that cannot be measured is NaN with a reason string; a
   measured 0 is a value.
2. **Three-valued logic.** Undefined never becomes a number; determinate verdicts
   never change when evidence goes missing.
3. **Null anchoring.** Every component is interpreted against a declared null
   family computed with the same estimator and configuration.
4. **Backward compatibility.** Existing estimator defaults are unchanged and
   pinned by tests; new behaviour is a new mode, keyword or estimator variant.
   Deprecations warn instead of breaking.
5. **Determinism and provenance.** Explicit seeds everywhere; every writer
   records the code version (package version + commit), parameters and seeds.
6. **No circularity in validation.** Bench generators do not import the
   estimators; witnesses are defined by mechanism; oracle channels are stored
   apart from the data the estimators read.
7. **Local and HPC parity.** Hunter results equal the local computation up to
   rounding; the hardware self-test checks the device kernels against NumPy.

## 8. Tests

`tests/` holds regression, ground-truth, null and property tests (run with
`python -m pytest -q`). Some groups:

| Area | Test files (examples) |
|---|---|
| estimators, known answers | `test_ram_ground_truth.py`, `test_srpi_ground_truth.py`, `test_srpi_agency.py`, `test_iim_ground_truth.py`, `test_iim_exact_tpm.py`, `test_iim_exact_reference.py`, `test_nas_capacity.py`, `test_pdi_repertoire.py`, `test_pdi_surrogate_excess.py`, `test_pdi_nas_null.py` |
| legacy defaults pinned | `test_construct_modes_legacy_pins.py` |
| estimator variants | `test_ram_pe.py`, `test_pdi_declared.py`, `test_nas_declared.py`, `test_iim_macro.py`, `test_declared_inputs.py`, `test_numerics.py` |
| evidence layer | `test_evidence.py`, `test_evidence_properties.py`, `test_verdict_wiring.py`, `test_nulls.py`, `test_status_rule.py`, `test_status_rule_error_control.py`, `test_protocol_schema.py`, `test_registry.py`, `test_reasons.py` |
| CI and statistics | `test_ci_assembly.py`, `test_ci_undefined.py`, `test_analysis_bootstrap.py`, `test_model_comparison.py` |
| Hunter | `test_hunter_pbs.py`, `test_hunter_iim.py`, `test_hunter_cost.py`, `test_hunter_calibration.py`, `test_hunter_scripts.py`, `test_hardware_selftest.py`, `test_iim_xp_kernel.py` |
| MPC-Bench | `test_bench_generators.py`, `test_bench_catalogue.py`, `test_bench_adversarial.py`, `test_bench_manipulation.py`, `test_bench_whole_brain.py`, `test_bench_forward.py`, `test_bench_designs.py`, `test_bench_runner.py`, `test_bench_records.py`, `test_bench_seeds.py`, `test_bench_hypotheses.py`, `test_bench_export.py`, `test_null_calibration.py`, `test_iim_validation.py` |
| protocols and analysis | `test_protocols.py`, `test_protocol_examples.py`, `test_nas_declared_workspace.py`, `test_estimator_version_columns.py`, `test_empirical_reference.py`, `test_default_empirical_run.py`, `test_necessity.py`, `test_definedness_audit.py` |
| preprocessing, events, orchestration | `test_preprocessing_fmri.py`, `test_preprocessing_eeg.py`, `test_event_parsing.py`, `test_run_pipeline_orchestration.py`, `test_tr_fallback.py` |
| dashboard, packaging, documentation, synthetic objects | `test_dashboard_security.py`, `test_dashboard_run_plan.py`, `test_packaging_infra.py`, `test_docs_consistency.py`, `test_synthetic_generator.py` |
