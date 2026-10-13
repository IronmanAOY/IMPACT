# Changelog

All notable changes to the IMPaCT Synergy Pipeline. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). Archived versions:
https://doi.org/10.5281/zenodo.23339786.

## [1.2.0] - 2026-10-13

Release 1.2.0 turns the pipeline into a measurement framework: five
null-anchored estimators, a three-valued, missingness-safe exclusion verdict
under declared protocols, a PBS Pro backend for HLRS Hunter, and the
white-box benchmark MPC-Bench with variants of the estimators that read
declared inputs. Definitions: [`docs/metrics.md`](docs/metrics.md).

### Added

- **Evidence layer** (`impact_pipeline.evidence`). `ComponentEvidence`, the
  component status PRESENT / ABSENT / UNDEFINED on the construct scale
  `c = (m - nu)/(rho - nu)` with a sampling SE (bootstrap or jackknife, plus
  the Monte-Carlo error of the null mean and the reference SE) and declared
  cutoffs `(z, delta)`; strong-Kleene OR over the channels of a principle and
  AND over the necessity set; the MPC verdict `EXCLUDED` / `MPC_CONSISTENT` /
  `UNDETERMINED` with stable reason codes, recoverable from its reasons; the
  MPC degree (capped weighted power mean, only for `MPC_CONSISTENT` rows),
  `degree_interval` and `verdict_stability`; `joint_dependence` (the
  single-source constraint). Property tests cover missingness safety,
  monotone resolution, veto, permutation symmetry, channel disjunction and
  reason-code decomposability.
- **Two status rules**, selected by the protocol: the interval rule (the
  empirical pipeline) and the TOST rule (the benchmark protocols), in which
  ABSENT is a two-sided equivalence test at `alpha_A = alpha / 5`, undecided
  components carry named reasons (`impact_pipeline.reasons`), NAS is judged
  per direction, IIM has a rank gate and every SE method has a declared
  degrees-of-freedom rule. Testability gating, protocol anchors and
  necessity sets: `impact_pipeline.testability`.
- **Protocols** (`protocols/`, schema `impact-mpc-protocol/1`, SHA-256 hash
  recorded with every verdict): `mpc_default.json`, the default of
  `run_pipeline.py` on empirical data (RAM prediction-error update, PDI
  repertoire, NAS capacity, SRPI agency, each with a declared fallback); the
  opt-in `mpc_behavioural_ram.json`; two example protocols with a declared NAS
  hub (`protocols/examples/`, `scripts/build_example_protocols.py`); the
  benchmark protocols (`protocols/bench/`).
- **Applicability registry** (`impact_pipeline.registry`, schema
  `impact-mpc-registry/1`, `--applicability-registry`): validation entries
  with entry criteria and the forward-model admission entries of the
  benchmark. No registry file is shipped.
- **Null calibration and sampling SEs.** Generic nulls
  (`impact_pipeline.nulls`: circular shift, phase randomisation, IAAFT, label
  permutation, onset jitter) with per-run seeds; null calibration of PDI, NAS
  and IIM; the moving-block bootstrap. Command-line flags `--protocol`,
  `--null-surrogates`, `--bootstrap-se`, `--bootstrap-block-len`,
  `--necessity-set`, `--applicability-registry`, and the evidence columns of
  `step2_df.csv` (`MPC_*` and `<P>_*`).
- **Construct-revision modes** (defaults unchanged): RAM typed evidence
  channels and `update="prediction_error"`; PDI `mode="repertoire"` (the
  repertoire of distinguishable states, labelled or unlabelled) and
  `mode="surrogate_excess"` (a documented negative result); NAS
  `mode="capacity"` (receive and return transfer between a declared hub and
  the periphery); SRPI `mode="agency"` (self-caused events against yoked
  replays); `bearer_nodes` for every estimator.
- **IIM**: `compute_IIM_from_tpm` (exact IIM of a known TPM), directional
  cuts, the `per_unit` TPM estimator, explicit node selection and state budget
  with logged adjustments, and the array-module Ψ kernel `impact_pipeline.iim_xp`
  (NumPy/CuPy).
- **Estimator variants** (`impact_pipeline.estimators`): RAM-PE
  (`ram-pe-2026.10`), declared PDI (`pdi-declared-2026.10`), declared NAS
  (`nas-declared-2026.10`) and macro IIM (`iim-macro-2026.10`), with the
  recording device, input declarations and common input basis of
  `estimators.declared_inputs`. One table of estimator versions
  (`evidence.ESTIMATOR_VERSIONS`).
- **MPC-Bench** (`impact_pipeline.bench`): families A (rate-network agent with
  five mechanism switches), B (binary networks with exact TPMs) and C
  (Stuart-Landau network); patchworks; a whole-brain Hopf model on the shipped
  connectome with EEG-like and BOLD-like forward models and views;
  adversarial constructions; null systems; the system catalogue
  (`bench/systems.yaml`); oracle manipulation checks; designs and run plans;
  the runner (`scripts/run_bench.py`, records `mpc-bench-result/1`, each
  estimator in its own try block, resume and shards); the development /
  confirmatory seed split and the confirmatory guard; the null-calibration
  design (`scripts/null_calibration.py`); the family-B IIM validation
  (`scripts/iim_validation.py`); the declarative hypothesis evaluator
  (`scripts/evaluate_hypotheses.py`, schema `mpc-bench-hypotheses/1`).
- **HLRS Hunter PBS Pro backend** (`--execution-mode hunter`): array jobs with
  4 shards per `mi300a` node, concurrent phase-1 and cut shards, one reduce and
  finalize job, a smoke test for the `test` queue, idempotent shards with
  checkpoints, timing records, `--hunter-iim-null-surrogates` and
  `--hunter-iim-bootstrap-se`, and a cost model with a size guard
  (`impact_pipeline.hunter_cost`). `python -m impact_pipeline.hardware_selftest`
  checks CuPy against NumPy; `scripts/hunter/` holds the setup-file template,
  install notes and smoke-test helper.
- Provenance (package version + commit, runtime versions, run parameters,
  hardware backend), `impact_pipeline.event_parsing`,
  `impact_pipeline.replication`, `--assume-tr`, `--fmriprep-dir`,
  `--no-atlas-robustness`, `--ci-reference`.
- Analysis tools: `scripts/definedness_audit.py`,
  `scripts/compute_empirical_reference.py` and the necessity statistics of
  `impact_pipeline.necessity`.
- Packaging and documentation: installable package (`pyproject.toml`),
  `LICENSE`, third-party notices, `.dockerignore`, `.flake8`, pre-commit and
  CI workflows, Dockerfile and Singularity definitions, the metrics
  reference, the architecture document and the Hunter runbook.

### Changed

- **Verdict names** follow the necessity-only stance: `EXCLUDED`,
  `MPC_CONSISTENT` (not an attribution of consciousness) and `UNDETERMINED`.
- **Three-valued CI.** An undefined component is NaN, never 0; CI is NaN when
  a weighted component or its reference is unusable (`CI_defined`,
  `CI_missing`, `CI_reference`). CI uses NAS directly (no HypergraphSynergy
  multiplier) and is reference-normalised (cohort high state or an external
  JSON). S is reported separately at every theta with Holm correction.
- **RAM.** Canonical HRF evaluated analytically; goal and feedback events as
  nuisance regressors; measured latencies undefined at the search edge;
  chance-corrected, cross-validated quality terms; a strict goal/feedback
  contract; modality presets.
- **SRPI.** Pre-event window strictly before onset; cross-validated
  separability; `min_events_per_class` at least 3.
- **IIM.** Default TPM estimator `node_shrinkage`; the MIP search evaluates
  both mechanism/purview pairings; checkpoint signatures include every
  result-changing parameter and `IIM_ALGORITHM_VERSION`.
- **PDI and NAS.** Undefined inputs return NaN with a reason instead of 0.0.
- **Statistics, preprocessing and dashboard.** DeLong with the covariance
  term, two-sided seeded permutation tests and Holm correction; no silent TR
  fallback; EEG channel typing and rest baselines; AAL-116; a dashboard bound
  to loopback with Host/Origin checks and a CSRF token.
- With `--null-surrogates K > 0` the metric columns hold calibrated values.
- Environment, containers and downloads: pinned minor versions in
  `environment.yml`, rebuilt container definitions, pinned OpenNeuro
  snapshots and checksum-verified atlas downloads.

### Fixed

- Hunter `build-campaign` raised `TypeError`; the Hunter scripts targeted
  Slurm, which HLRS Hunter does not run.
- Parallel IIM Ψ double-counted in the fallback path; SQLite kernel caches were
  left behind and raced.
- DeLong variance without the covariance term; one-sided permutation tests.
- CI reference: a missing component silently became 1.0.
- PDI baseline routing could use the evaluated run as its own baseline.
- Event resolution fell back across sessions; camelCase self/other labels
  were classified asymmetrically.
- Dashboard: out-dir collisions, stored XSS, upload error handling.
- `run_all.sh --dry-run` ran the real pipeline; `download_data.sh` refused a
  clone whose HEAD carried several tags.
- numpy's SVD can fail on an exactly rank-deficient matrix; the estimator
  variants and the forward models repeat the computation with SciPy's
  `gesvd` only when numpy raises (`impact_pipeline.numerics`).

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

- HypergraphSynergy multiplier inside CI; the degenerate `pci_fmri` baseline
  comparator (replaced by epoch-wise normalised LZ76 complexity, `lzc`).
- Silent fallbacks: the 2.0 s default TR, all-events-as-stimuli, `response_time`
  as feedback, zero-filled undefined metrics.
- The npm OpenNeuro CLI download path and the Melbourne repository clone (the
  source returns HTTP 404; replication needs a local copy).
