# Architecture

This document describes how the IMPaCT Synergy Pipeline 1.1.0 is organised:
the layers, the data flow of a run, the HLRS Hunter campaign, MPC-Bench and the
design rules the code follows. Definitions of the estimators and of the evidence
layer are in [`metrics.md`](metrics.md); the Hunter procedure is in
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
| `impact_pipeline.mpc_readiness` | per-run readiness report (which metrics can be defined, and why not), using `event_parsing` |
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
| `impact_pipeline.bench.*` | MPC-Bench: generators, patchwork, witnesses, export, factorial, sweeps, runner, rival rules, LZ76, Gaussian Φ_R |
| `scripts/live_dashboard.py`, `impact_desktop_app.py` | browser dashboard and desktop launcher |
| `scripts/generate_real_derived_synth_completed.py`, `inspect_real_sources_for_synth.py` | real-data-derived synthetic smoke-test objects |
| `scripts/download_data.sh`, `download_atlases.sh`, `fetch_fmriprep*.sh`, `run_all.sh` | data, atlases, fMRIPrep, end-to-end local run |
| `scripts/hunter/` | Hunter setup-file template, install notes, smoke-test helper |

## 3. A local run

```text
 BIDS dataset ──(0) fMRIPrep (optional, Docker)──► <bids-root>/derivatives/fmriprep
      │
      └──(1) preprocessing (--run-preprocessing)──► <out>/preprocessed/<subj>/<ses>/<cond>/*_ts.npy
                                                    <out>/preprocessed/<subj>/<ses>/rest/*_ts.npy
 (2) run_s_ci → compute_synergy_ci, per run:
       events.tsv ─► event_parsing ─► RAM / SRPI bundles
       time series ─► RAM, PDI, NAS, IIM, SRPI  (+ K null surrogates each when --null-surrogates K)
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
       ─► runs/<run>/final_result.json, iim_results.json/.csv
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
 estimator details ──► _component_record (estimate, null moments, reason, estimator id)
                   ──► ComponentEvidence (+ bearer_id, protocol_id, substrate = modality,
                                           grain = atlas, regime = {modality, n_time, n_nodes, tr})
                   ──► ApplicabilityRegistry.is_validated (optional)
                   ──► component status per channel ─► Kleene OR per principle
                   ──► Kleene AND over the necessity set ─► verdict + reasons
                   ──► MPC degree (MPC_CONSISTENT rows; two-anchor scale; capped power mean)
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
 (compute_synergy_ci unchanged)       null calibration, bearer views) ─► evidence_verdict
                                             │
 factorial / sweeps / witness tasks ──► run_bench.run_tasks (process pool, resume, shards)
                                             ─► results.jsonl, results.csv, run_manifest.json
 rules.py: rival attribution rules on component matrices (rule audit)
```

Seed policy: development seeds 0-999; confirmatory seeds from 10000 and family C
only with `--confirmatory`, which requires a clean checkout descending from the
code-freeze tag and records it.

## 7. Design rules

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

## 8. Tests

`tests/` holds regression, ground-truth, null and property tests (run with
`python -m pytest -q`). Some groups:

| Area | Test files (examples) |
|---|---|
| estimators, known answers | `test_ram_ground_truth.py`, `test_srpi_ground_truth.py`, `test_srpi_agency.py`, `test_iim_ground_truth.py`, `test_iim_exact_tpm.py`, `test_iim_exact_reference.py`, `test_nas_capacity.py`, `test_pdi_surrogate_excess.py`, `test_pdi_nas_null.py` |
| legacy defaults pinned | `test_construct_modes_legacy_pins.py` |
| evidence layer | `test_evidence.py`, `test_evidence_properties.py`, `test_verdict_wiring.py`, `test_nulls.py` |
| CI and statistics | `test_ci_assembly.py`, `test_ci_undefined.py`, `test_analysis_bootstrap.py`, `test_model_comparison.py` |
| Hunter | `test_hunter_pbs.py`, `test_hunter_iim.py`, `test_hunter_calibration.py`, `test_hunter_scripts.py`, `test_hardware_selftest.py`, `test_iim_xp_kernel.py` |
| MPC-Bench | `test_bench_generators.py`, `test_bench_export.py`, `test_bench_rules.py`, `test_phiid_gaussian.py` |
| preprocessing, events, orchestration | `test_preprocessing_fmri.py`, `test_preprocessing_eeg.py`, `test_event_parsing.py`, `test_run_pipeline_orchestration.py`, `test_tr_fallback.py` |
| dashboard, packaging, synthetic objects | `test_dashboard_security.py`, `test_dashboard_run_plan.py`, `test_packaging_infra.py`, `test_synthetic_generator.py` |
