# Changelog

All notable changes to the IMPaCT Synergy Pipeline. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). Archived versions:
https://doi.org/10.5281/zenodo.15306740.

## [1.1.0] - 2026-09-28

Release 1.1.0 repositions the pipeline as a measurement framework:
null-anchored estimators, a three-valued, missingness-safe verdict rule (the MPC
verdict), a white-box benchmark (MPC-Bench) and a PBS Pro backend for HLRS
Hunter. It also fixes the scientific and engineering defects found in a full
review of 1.0.0. Every behavioural fix has a regression test; estimator fixes
have ground-truth or null tests. Defaults of the existing estimator modes are
unchanged unless listed under "Changed".

### Added

- **Evidence layer** (`impact_pipeline.evidence`). `ComponentEvidence`, the
  component status PRESENT / ABSENT / UNDEFINED, strong-Kleene OR over the
  evidence channels of a principle and AND over the necessity set, and the MPC
  verdict `EXCLUDED` / `MPC_CONSISTENT` / `UNDETERMINED` with stable reason codes
  (`MISSING`, `NO_NULL_CALIBRATION`, `INCONCLUSIVE`, `UNDEFINED`,
  `NOT_IMPLEMENTED`, `ESTIMATOR_NOT_VALIDATED`, `ABSENT`, `BEARER_MISMATCH`,
  `PROTOCOL_MISMATCH`). The verdict is recoverable from its reasons.
  Two-anchor normalisation, the MPC degree (capped weighted power mean,
  geometric by default, only for `MPC_CONSISTENT` rows), `weakest_link`,
  `degree_interval` (delta method and bootstrap), `verdict_stability`, the
  JSON `ApplicabilityRegistry` of validated estimator configurations and the
  `bearer_coherence` diagnostic. Property tests on seeded random configurations
  cover missingness safety, monotone resolution, determinacy iff all completions
  agree, veto, permutation symmetry, channel disjunction and reason-code
  decomposability.
- **Generic nulls** (`impact_pipeline.nulls`): circular shift, uni- and
  multivariate phase randomisation, IAAFT, label permutation (optionally
  stratified by phase bin), onset jitter and `component_null`, with seeds derived
  per run.
- **Verdict wiring**: `compute_synergy_ci(..., null_surrogates, necessity_set,
  applicability_registry, null_seed, null_kinds)` and the CLI flags
  `--null-surrogates K`, `--necessity-set`, `--applicability-registry`. New
  step-2 columns `MPC_verdict`, `MPC_reason`, `MPC_degree`, `MPC_necessity_set`,
  `MPC_null_surrogates`, `MPC_null_seed`, `MPC_null_families` and per principle
  `<P>_status`, `<P>_margin`, `<P>_estimate`, `<P>_null_mean`, `<P>_null_sd`,
  `<P>_null_n`. With K = 0 every verdict is `UNDETERMINED`
  (`NO_NULL_CALIBRATION`).
- **Null calibration of PDI, NAS and IIM** (`null_surrogates=K`): excess over
  surrogates, z-score, one-sided p-value and calibrated value. IIM reports the
  integration mass `Delta_Psi` (bits) and `canonical_calibrated`, which is about
  0 for independent nodes and increases with coupling.
- **Construct-revision modes** (defaults unchanged):
  - `compute_SRPI(mode="agency")`: self-caused events against yoked,
    stimulus-identical, phase-matched other-caused replays; events contract
    validator; pre-state partialling refitted inside each CV fold; two-sided,
    chance-corrected terms; arithmetic aggregation without a hard zero.
  - `compute_NAS(mode="capacity")` (Network Availability Score): receive and
    return Gaussian transfer entropy between a declared hub and the periphery
    against block circular-shift surrogates; the weaker direction is gated;
    metastability and L/B/H are profile descriptors; `confounds` option.
  - `compute_PDI(mode="surrogate_excess")`: repertoire entropy, LZ76 diversity
    and binarised effective dimensionality as signed excess over multivariate
    spectrum-preserving surrogates.
  - `compute_RAM`: typed `impact_channel` evidence channels and
    `compute_RAM_by_channel`; `update="prediction_error"` (Rescorla-Wagner fit to
    logged choices and rewards, permutation null); declared `adaptation_locus`.
  - `bearer_nodes` for RAM, PDI, NAS, SRPI and IIM.
- **IIM**: `compute_IIM_from_tpm` (exact IIM of a known TPM; value
  `Delta_Psi` in bits), directional (IIT unidirectional) cuts
  (`cut_mode="directional"`), the `per_unit` TPM estimator, explicit node
  selection and state-budget policy with logged adjustments, and the
  array-module Ψ kernel `impact_pipeline.iim_xp` (NumPy/CuPy; parity with the
  numba kernel to 1e-10).
- **MPC-Bench** (`impact_pipeline.bench`, `scripts/run_bench.py`): family A
  modular rate-network agent with five mechanism switches, family B
  binary/kinetic-Ising networks with exact TPMs, held-out family C
  (Stuart-Landau), patchwork, null and hypersynchronous systems, the witness
  catalogue (`witnesses.yaml`), export to the pipeline layout, an in-memory
  runner, factorial and dose-response designs, a process-pool runner with
  resume, sharding, provenance and a PBS Pro array template, a development /
  confirmatory seed policy with a code-freeze guard, rival attribution rules
  (union, count-k, means, weakest link, naive Bayes, product of credences,
  logistic classifier, single markers) and comparators (LZ76, closed-form
  Gaussian Φ_R for VAR(1)).
- **HLRS Hunter PBS Pro backend** (default for `--execution-mode hunter`;
  `--hunter-scheduler pbs|slurm`): `*.pbs` scripts with
  `select=1:node_type=mi300a`, walltime checks (24 h; 25 min on `test`), arrays
  only for two or more subjobs, `-W group_list`, `-l ws13=True`; 4 shards per
  node through PALS `mpiexec` with CPU/GPU binding; `00_submit_all.sh` with
  whole-array `afterok` dependencies; phase-1 and cut shards run concurrently
  with one merged reduce + finalize job; single-job smoke test for the `test`
  queue; skip-if-complete shards, per-cut checkpoints, atomic writes, per-task
  timing JSON and `timing_summary.json`; `--hunter-stage status`;
  `--hunter-iim-null-surrogates K` (surrogate runs in the campaign, reduced with
  the same function as the local path); `iim_results.csv` and
  `cache/hunter_iim_results.csv`; `--repo-root` / `IMPACT_REPO_ROOT`;
  configurable packing, shard and worker counts; campaigns can be built on login
  nodes without an APU.
- `python -m impact_pipeline.hardware_selftest`: CuPy against NumPy for matmul,
  eigh, scatter-add, bincount, SVD, pinv, the IIM TPM kernels and the IIM Ψ
  kernel (`iim_psi_xp_parity`); JSON report; exit codes 0/1/2.
- `scripts/hunter/`: PBS setup-file template, dry-run-by-default install notes,
  smoke-test helper; `requirements-hunter.txt` and `constraints-hunter.txt` for
  the cray-python route.
- Provenance: code version = package version + commit (`1.1.0+g<sha>[.dirty]`)
  or `IMPACT_CODE_VERSION`; `impact_pipeline.__version__`; provenance manifest
  with runtime versions, run parameters and hardware backend.
- `impact_pipeline.event_parsing`: single source for `events.tsv` parsing and
  events-file resolution, shared by computation and readiness checks.
- `impact_pipeline.replication` with `--replication-root` /
  `IMPACT_REPLICATION_ROOT` and a fail-fast check before any computation.
- `--assume-tr`, `--fmriprep-dir`, `--no-atlas-robustness`, `--ci-reference`.
- Packaging: `pyproject.toml` (installable package, extras `dashboard`,
  `hunter`, `dev`), `LICENSE`, `licenses/THIRD_PARTY_NOTICES.md`,
  `.dockerignore`, `.flake8`, pre-commit and GitHub Actions workflows that work.
- Documentation: `docs/HLRS_HUNTER_RUNBOOK.md`, `docs/ARCHITECTURE.md`, this
  changelog; `docs/metrics.md` rewritten against the code.

### Changed

- **Verdict names** follow the necessity-only stance: `EXCLUDED` (a principle is
  credibly absent), `MPC_CONSISTENT` (all principles present; not an
  attribution of consciousness) and `UNDETERMINED`. The component status is
  judged on a two-anchor construct scale with a sampling SE and declared
  smallest effects of interest (`docs/metrics.md`, section 8).
- **Three-valued CI (D1).** An undefined component is NaN, never 0; CI is NaN
  when a weighted component or its reference is unusable, with `CI_defined`,
  `CI_missing` and `CI_reference` columns; statistics exclude undefined rows and
  report how many.
- **CI uses NAS directly (D2).** The HypergraphSynergy multiplier was removed, so
  CI no longer depends on theta. S is reported separately, at every theta with
  Holm correction.
- **Reference-normalised CI (D3)** instead of "human-normalised": default
  reference = cohort high-state (awake) means, or an external JSON; no 1e-12
  floor.
- **RAM (D4).** Canonical HRF evaluated analytically on the true time axis (peak
  about 5 s instead of about 252 s); goal and feedback events are nuisance
  regressors of the magnitude GLM; FIR/xcorr latencies without clipping to 0 and
  undefined at the search edge; G is a cross-validated ridge-CCA held-out
  correlation, chance-corrected against phase-randomised surrogates; F and U are
  chance-corrected; feedback values stay aligned with their events; strict
  goal/feedback contract (undefined instead of proxies); undefined on
  rank-deficient designs and non-finite input; EEG uses its own preset.
- **SRPI (D5).** Pre-event window strictly before onset; separability is the
  cross-validated shrinkage-LDA AUC, `[2(AUC - 0.5)]_+`, instead of an in-sample
  distance that saturated; `min_events_per_class` must be at least 3.
- **IIM (D6).** Default TPM estimator `node_shrinkage` (state-by-node,
  James-Stein shrinkage) instead of the joint Laplace estimate, which inflated
  Ψ for noise (about 17-fold); the MIP search evaluates both mechanism/purview
  pairings; checkpoint and cache signatures include every result-changing
  parameter and `IIM_ALGORITHM_VERSION = "iim-v4-2026.09"`; node selection and
  bin/node budget are explicit and logged.
- **PDI/NAS (D7).** Undefined inputs return NaN with a reason instead of 0.0.
- **Statistics (D8).** Fast DeLong with the covariance term; two-sided, seeded
  permutation tests with within-subject swaps; subject-level bootstrap; Holm
  correction across families; all statistics written to `<out>/stats/`;
  discriminability contrast in model comparison; motion FD weighted per run.
- **Preprocessing (D10).** No silent TR fallback (error unless `--assume-tr`);
  ds003171 `task-audio` (sub-10JR) and `light` sessions handled explicitly;
  EEG excludes EOG/EMG/ECG/misc channels and writes rest baselines for PDI; the
  AAL atlas is AAL-116 (`aal90` accepted as a legacy key); atlas root from
  `IMPACT_ATLAS_DIR` or the repository; fMRIPrep derivatives at
  `<bids-root>/derivatives/fmriprep`.
- **Dashboard (D11).** Loopback bind by default, Host/Origin checks, CSRF token,
  escaped output, validated dataset IDs, no argv flag injection; mixed-source CI
  is labelled exploratory with an `UNDETERMINED` (`BEARER_MISMATCH`) verdict.
- **Synthetic smoke-test generator** (2.2.0): honest reporting (smoke gate
  separate from known-answer observations), planted structure recorded, relative
  paths, `--validate-only` on relocated archives; the smoke gate follows D1 (a
  NaN value with a recorded reason passes).
- With `--null-surrogates K > 0` the metric columns hold calibrated values
  (excess over the null mean, floored at 0) and CI uses them.
- Hardware: the accelerator `eigh` is used only after a NumPy parity check,
  with a CPU fallback (`IMPACT_EIGH_BACKEND`); Hunter shards run the Ψ kernel on
  the APU.
- Environment: `environment.yml` pinned to the verified minor versions;
  `mne-base`/`matplotlib-base`; Dockerfile and Singularity definitions rebuilt
  (non-root user, no data in the build context).
- Download scripts: pinned OpenNeuro snapshots through DataLad/git-annex,
  checksum-verified atlas downloads, fMRIPrep 25.1.3 with the license mounted by
  path.

### Fixed

- Hunter `build-campaign` raised `TypeError` (missing provenance arguments).
- Hunter scripts targeted Slurm, which HLRS Hunter does not run.
- Parallel IIM Ψ double-counted in the fallback path; a kernel-cache miss
  disabled parallelism permanently; SQLite kernel caches were left behind and
  raced when shared by pool workers.
- DeLong variance without the covariance term; one-sided permutation tests.
- CI reference: a component missing from a supplied reference silently became
  1.0; a `None` reference raised `TypeError`.
- PDI baseline routing could use the evaluated run as its own baseline.
- Event resolution fell back across sessions (sed vs sed2, task-audio*);
  camelCase self/other labels were classified asymmetrically.
- Dashboard: out-dir collisions between datasets, form reverted by polling,
  stored XSS, upload error handling, participant mappings that reused one source
  subject.
- `run_all.sh`: a trailing `--dry-run` ran the real pipeline; fMRIPrep
  completeness check failed under a `func` ancestor folder.
- `download_data.sh` refused a clone whose HEAD carried several tags.
- Synthetic generator: overlapping EEG events, a stale RAM parameter copy
  (`quality_ridge=1e-4`), subset-dependent seeds.

### Deprecated

- `compute_CI`: emits a `DeprecationWarning` once per process; the `CI` column is
  the legacy geometric-mean CI and is not a gate. Use `evidence.mpc_verdict` and
  `evidence.degree`.
- `ci_human_refs` argument of `compute_synergy_ci` (use `ci_reference`).
- Environment variable names `IMPACT_HUNTER_SLURM_SETUP_FILE`,
  `IMPACT_HUNTER_SLURM_SETUP`, `IMPACT_HUNTER_SLURM_ACCOUNT` (use
  `IMPACT_HUNTER_SETUP_FILE`, `IMPACT_HUNTER_SETUP`,
  `IMPACT_HUNTER_PBS_GROUP_LIST`); they are still accepted.
- Atlas key `aal90` (use `aal116`; still recognised for existing outputs).

### Removed

- The `ATTRIBUTED` / `NOT_ATTRIBUTED` verdict names of the unreleased first
  evidence-layer build (no aliases).
- HypergraphSynergy multiplier inside CI; the degenerate `pci_fmri` baseline
  comparator (replaced by epoch-wise normalised LZ76 complexity, `lzc`).
- Silent fallbacks: the 2.0 s default TR, all-events-as-stimuli, `response_time`
  as feedback, zero-filled undefined metrics.
- The npm OpenNeuro CLI download path and the Melbourne repository clone (the
  source returns HTTP 404; replication needs a local copy).
- Unused imports and dead helpers in `mpc_metrics` (for example
  `_safe_mutual_info_score`).

## [1.0.0] - 2025-04-29

First archived release (https://doi.org/10.5281/zenodo.15306741).
