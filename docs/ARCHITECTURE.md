# Architecture

This document describes how the IMPaCT Synergy Pipeline 1.1.0 is organised:
the layers, the data flow of a run, the HLRS Hunter campaign, MPC-Bench and its
v2 round, and the design rules the code follows. Definitions of the estimators
and of the evidence layer are in [`metrics.md`](metrics.md) (v2:
[`metrics_v2.md`](metrics_v2.md)); the Hunter procedure is in
[`HLRS_HUNTER_RUNBOOK.md`](HLRS_HUNTER_RUNBOOK.md).

## 1. Layers

```text
 entry points      run_pipeline.py · scripts/*.py|sh · python -m impact_pipeline.hardware_selftest
      │
 orchestration     run_pipeline.main (steps 0-9) · hunter_iim (campaign, stages, PBS/Slurm writers)
      │
 assembly          run_synergy_ci.run_s_ci · synergy_ci.compute_synergy_ci (per-run metrics,
                   null calibration, evidence wiring, legacy CI, MPC degree)
      │
 evidence          evidence (status, Kleene logic, verdict, degree, registry) · nulls
      │
 estimators        mpc_metrics (RAM, PDI, NAS, IIM, SRPI, legacy CI) · iim_xp (Ψ kernel)
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
| `impact_pipeline.iim_xp` | array-module (NumPy/CuPy) Ψ contribution and cut-TPM kernels |
| `impact_pipeline.nulls` | generic surrogates (circular shift, phase randomisation, IAAFT, label permutation, onset jitter) and `component_null` |
| `impact_pipeline.evidence` | `ComponentEvidence`, component status, strong-Kleene logic, `mpc_verdict`, reason codes, two-anchor scale, MPC degree and intervals, `ApplicabilityRegistry`, `bearer_coherence` |
| `impact_pipeline.synergy_ci` | `compute_synergy_ci` (per-run loop over the preprocessed tree), `assemble_ci`, `assemble_mpc_degree`, CI reference resolution, IIM worker planning |
| `impact_pipeline.run_synergy_ci` | `run_s_ci` (step 2 driver: events, RAM presets, subject means, S by theta), `load_onsets`, `RAM_PARAM_PRESETS` |
| `impact_pipeline.event_parsing` | the single source for `events.tsv` parsing: label patterns, session-exact file resolution, event bundles, SRPI-agency contract |
| `impact_pipeline.mpc_readiness` | per-run readiness report (which metrics can be defined, and why not), using `event_parsing`; NAS against the run's protocol (capacity without a declared hub: `NO_DECLARED_WORKSPACE`, the rule of `synergy_ci.nas_hub_missing`; a declared hub that does not fit the recording's node count: `INVALID_WORKSPACE`) |
| `impact_pipeline.preprocessing`, `preprocessing_eeg` | fMRI atlas time series from fMRIPrep derivatives; EEG channel time series from BrainVision; rest baselines |
| `impact_pipeline.dataset_catalog` | dataset metadata, task/state grammar, explicit per-subject task aliases |
| `impact_pipeline.analysis_bootstrap`, `model_comparison`, `baseline_metrics`, `motion_model`, `atlas_robustness`, `replication` | steps 3-8: paired statistics (bootstrap, permutation, DeLong, Holm), baseline comparators, motion covariates, robustness atlases, Melbourne replication |
| `impact_pipeline.generate_word_doc` | step 9: Word report |
| `impact_pipeline.hunter_iim` | Hunter campaign: preparation, shard/reduce/finalize stages, idempotency, timing, PBS and Slurm script writers |
| `impact_pipeline.execution_profiles` | `local` and `hunter` profiles, `HunterPBSProfile`, `HunterSlurmProfile` |
| `impact_pipeline.hardware_backend` | target resolution (`cpu`, `auto`, `gpu`, `hunter-apu`), CuPy loading, thread limits, accelerated linear algebra with the eigh parity check |
| `impact_pipeline.hardware_selftest` | CuPy-against-NumPy self-test (`python -m`) |
| `impact_pipeline.provenance` | data origin (`real`/`dummy`) and output routing, code version, runtime versions, repository root |
| `impact_pipeline.utils` | HypergraphSynergy (the exploratory statistic S) and helpers |
| `impact_pipeline.bench.*` | MPC-Bench: generators (families A, B, C), patchwork, whole-brain Hopf model and forward models, adversarial constructions, manipulation checks, witnesses, export and in-memory runner, factorial, sweeps, runner, reference anchor (`reference`), rival rules, rule audit, LZ76, Gaussian Φ_R |
| `impact_pipeline.necessity` | symmetric three-outcome necessity criteria, verdict-level summaries and Clopper-Pearson bounds (paper-2 hypotheses, bench evaluators); NCA ceilings and fsQCA necessity, which no script calls yet, are kept for the planned paper-2 analysis |
| `protocols/`, `predictions/` | declared MPC protocols (JSON, hashed): `mpc_default_v1.json` (the preregistered default of empirical runs; RAM is never ABSENT under it; no NAS hub, so NAS is UNDEFINED, `NO_DECLARED_WORKSPACE`, unless a derived protocol declares one), the opt-in [`mpc_behavioural_ram_v1.json`](../protocols/README.md#mpc_behavioural_ram_v1json-opt-in), the frozen bench protocols, `examples/` (derived from v1 with a declared NAS hub) and `v2/`; [`protocols/README.md`](../protocols/README.md) gives the reasons for each declaration. The paper-2 hypothesis registry and its schema |
| `impact_pipeline.evidence_v2` | MPC-Bench v2: the status rule `tost-v2`, protocol schema `impact-mpc-protocol/3` (`ProtocolV3`) and the dispatch between the v1 and the v2 layer (a protocol without a `status_rule` block goes to the v1 layer unchanged) |
| `impact_pipeline.v2.*` | MPC-Bench v2: estimators `nas_v3`, `iim_v5`, `ram_v3`, `pdi_v3`; `declared_inputs` (recording device, input declarations, common input basis); `records` (result schema `mpc-bench-result/3`), `reasons` (UNDEFINED vocabulary), `seeds` (v2 seed policy and seed map), `provenance` (code identity by git tree, confirmatory guard); `testability`, `registry_v3` (applicability registry v3, forward-model admission), `hypothesis_engine` (declarative evaluator), `numerics` |
| `impact_pipeline.bench.run_bench_v2`, `designs_v2`, `forward_v2`, `adversarial_v2`, `manipulation_v2` | the v2 runner (one simulation, several scorings, each estimator in its own try block), the v2 designs and run plans, the forward-model views, the v2 witnesses and adversaries (`witnesses_v2.yaml`) and the realisation checks |
| `scripts/run_bench.py`, `bench_reference.py`, `benchmark_attribution_rules.py`, `null_calibration.py`, `calibrate_bench.py`, `iim_validation.py`, `bench_hypotheses.py`, `mpcbench_confirmatory.sh`, `necessity_power.py`, `simulate_rule_recovery.py`, `audit_aggregation.py`, `definedness_audit.py`, `run_predictions.py`, `compute_empirical_reference.py`, `build_example_protocols.py`, `figures/` | MPC-Bench runs, bench reference, rule audit, null calibration, development calibration, family-B IIM validation, the preregistered hypotheses HC1-HC10 and their run plan, power and recovery simulations, aggregation audit, definedness audit, registry evaluation, external empirical reference anchor, example derived protocols (NAS hub), figures |
| `scripts/live_dashboard.py`, `impact_desktop_app.py` | browser dashboard and desktop launcher |
| `scripts/generate_real_derived_synth_completed.py`, `inspect_real_sources_for_synth.py` | real-data-derived synthetic smoke-test objects |
| `scripts/download_data.sh`, `download_atlases.sh`, `fetch_fmriprep.sh`, `run_all.sh` | data, atlases, fMRIPrep, end-to-end local run |
| `scripts/hunter/` | Hunter setup-file template, install notes, smoke-test helper |
| `scripts/run_bench_v2.py`, `scripts/v2/`, `bench_hypotheses_v2.py` | MPC-Bench v2: runner command line, development calibration, protocol builder, family-B IIM validation, null calibration, operating characteristics, environment lock, v1 regression gate, registry v3, integrity audit, the confirmatory run plan and the evaluator |

## 3. A local run

```text
 BIDS dataset ──(0) fMRIPrep (optional, Docker)──► <bids-root>/derivatives/fmriprep
      │
      └──(1) preprocessing (--run-preprocessing)──► <out>/preprocessed/<subj>/<ses>/<cond>/*_ts.npy
                                                    <out>/preprocessed/<subj>/<ses>/rest/*_ts.npy
 (2) run_s_ci → compute_synergy_ci, per run:
       events.tsv ─► event_parsing ─► RAM / SRPI bundles
       time series ─► RAM, PDI, NAS, IIM, SRPI  (modes from --protocol, default
                      protocols/mpc_default_v1.json on real data; + K null surrogates
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
- **Scheduler backends.** PBS Pro is the default (HLRS Hunter). A generic Slurm
  backend is kept (`--hunter-scheduler slurm`) with the same stages.
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

## 6. MPC-Bench

```text
 generators / patchwork / witnesses ──► BenchSystem(ts, events, meta, oracle)
     (knobs: eta, K, g_b + ff_only, c_int, e; families A, B, C; seeds)
         │                                   │
         │ export_system                     │ oracle written separately, never read by estimators
         ▼                                   ▼
 synergy_ci layout on disk            run_in_memory (public estimators, optional modes,
 (compute_synergy_ci unchanged)       null calibration, jackknife SEs, bearer views)
                                             ─► evidence_verdict (bench protocol: external
                                                reference = positive control, dev seeds)
                                             │
 factorial / sweeps / witness / patchwork / adversarial / whole-brain tasks
                                      ──► run_bench.run_tasks (process pool, resume, shards)
                                             ─► results.jsonl, results.csv, run_manifest.json
 rules.py + audit.py: rival decision rules and the rule audit on estimated statuses
```

Seed policy (round v1): development seeds 0-999 (1000-9999 are refused);
confirmatory seeds from 10000 and family C only with `--confirmatory
--freeze-tag mpcbench-freeze-v1`, which requires a clean checkout that
descends from the code-freeze tag with unchanged `src/` and `scripts/`, and
records the tag and commit.

## 7. MPC-Bench v2

The v2 round adds new modules next to the frozen v1 code and never edits a v1
code path, protocol or result ([`metrics_v2.md`](metrics_v2.md), the
[v2 preregistration](preregistration/MPC_BENCH_PREREGISTRATION_V2.md)):

```text
 declared inputs (v2.declared_inputs) ──► estimators NAS v3, IIM v5, RAM-PE v3, PDI v3 (+ v1 SRPI)
 designs_v2 / adversarial_v2 / forward_v2 ──► run_bench_v2: one simulation per task, scored under
                                               several declarations, views and estimator forms
                                             ─► evidence_v2 (tost-v2, protocols /3 from
                                                protocols/v2/generated/) ─► records mpc-bench-result/3
 dev_calibration.py (development seeds) + decisions ──► build_protocols_v2.py ──► protocols/v2/generated/
 confirmatory run plan ──► build_registry_v3.py ──► integrity_audit.py ──► bench_hypotheses_v2.py
                                                                         (hypothesis_engine)
```

- **Seed policy (round v2)**: development seeds 0-999, confirmatory seeds from
  20000; the v1 block 10000-19999 is never reused (`v2.seeds`,
  `protocols/v2/seed_map_v2.json`).
- **Confirmatory guard**: `v2.provenance.confirmatory_guard` refuses a dirty
  tree, a missing tag `mpcbench-freeze-v2`, and `src/` or `scripts/` trees
  that differ from the tag. Commits after the tag change those trees, so a
  confirmatory run, the integrity audit and the evaluator run from a checkout
  of the tag.
- **v1 regression gate**: `scripts/v2/regression_gate.py` pins the read-only
  v1 files by SHA-256 and re-judges and re-runs stored v1 records (it needs
  the stored v1 outputs, which are not versioned).

**Naming.** v1 and v2 are the two freeze rounds of MPC-Bench. Within round
v1, "v2" also names the second evidence-layer revision of 1.1.0 (protocol
schema `/2`), and bench 2.0.0 is the v1 benchmark; the `_v2` modules and
`v2/` folders belong to round v2. A suffix such as `_v3` or `_v5` on an
estimator numbers that estimator's revision, not the round.

## 8. Design rules

1. **Measured inputs only.** No silent fallbacks (TR, events, baselines,
   feedback). A quantity that cannot be measured is NaN with a reason string; a
   measured 0 is a value.
2. **Three-valued logic.** Undefined never becomes a number; determinate verdicts
   never change when evidence goes missing.
3. **Null anchoring.** Every component is interpreted against a declared null
   family computed with the same estimator and configuration.
4. **Backward compatibility.** Existing estimator defaults are unchanged and
   pinned by tests against values from the pre-revision code; new behaviour is a
   new mode or keyword. Deprecations warn instead of breaking.
5. **Determinism and provenance.** Explicit seeds everywhere; every writer
   records the code version (package version + commit), parameters and seeds.
6. **No circularity in validation.** Bench generators do not import the
   estimators; witnesses are defined by mechanism; oracle channels are stored
   apart from the data the estimators read.
7. **Local and HPC parity.** Hunter results equal the local computation up to
   rounding; the hardware self-test checks the device kernels against NumPy.

## 9. Tests

`tests/` holds regression, ground-truth, null and property tests (run with
`python -m pytest -q`); `tests/v2/` holds the tests of the v2 round. Some
groups:

| Area | Test files (examples) |
|---|---|
| estimators, known answers | `test_ram_ground_truth.py`, `test_srpi_ground_truth.py`, `test_srpi_agency.py`, `test_iim_ground_truth.py`, `test_iim_exact_tpm.py`, `test_iim_exact_reference.py`, `test_nas_capacity.py`, `test_pdi_repertoire.py`, `test_pdi_surrogate_excess.py`, `test_pdi_nas_null.py` |
| legacy defaults pinned | `test_construct_modes_legacy_pins.py` |
| evidence layer | `test_evidence.py`, `test_evidence_properties.py`, `test_verdict_wiring.py`, `test_nulls.py` |
| CI and statistics | `test_ci_assembly.py`, `test_ci_undefined.py`, `test_analysis_bootstrap.py`, `test_model_comparison.py` |
| Hunter | `test_hunter_pbs.py`, `test_hunter_iim.py`, `test_hunter_cost.py`, `test_hunter_calibration.py`, `test_hunter_scripts.py`, `test_hardware_selftest.py`, `test_iim_xp_kernel.py` |
| MPC-Bench | `test_bench_generators.py`, `test_bench_export.py`, `test_bench_rules.py`, `test_bench_audit.py`, `test_bench_reference.py`, `test_bench_runner_v2.py`, `test_bench_calibration.py`, `test_phiid_gaussian.py` |
| protocols, analysis and registry | `test_protocols.py`, `test_protocol_examples.py`, `test_nas_declared_workspace.py`, `test_estimator_version_columns.py`, `test_empirical_reference.py`, `test_default_empirical_run.py`, `test_analysis_scripts.py`, `test_necessity.py`, `test_predictions_registry.py`, `test_definedness_audit.py`, `test_figures.py` |
| preprocessing, events, orchestration | `test_preprocessing_fmri.py`, `test_preprocessing_eeg.py`, `test_event_parsing.py`, `test_run_pipeline_orchestration.py`, `test_tr_fallback.py` |
| dashboard, packaging, synthetic objects | `test_dashboard_security.py`, `test_dashboard_run_plan.py`, `test_packaging_infra.py`, `test_synthetic_generator.py` |
| MPC-Bench v2 (`tests/v2/`) | `test_status_rule_v2.py`, `test_nas_v3.py`, `test_iim_v5.py`, `test_ram_v3.py`, `test_pdi_v3.py`, `test_run_bench_v2.py`, `test_designs_v2.py`, `test_seed_policy_v2.py`, `test_build_protocols_v2.py`, `test_hypothesis_engine.py`, `test_integrity_audit.py`, `test_v1_regression_gate.py`, `test_prereg_v2_consistency.py`, `test_docs_v2.py` |
