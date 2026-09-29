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

### 1.1.0 — post-freeze fixes

Changes made after the code-freeze tag `mpcbench-freeze-v1` (commit f2cf249).
None of them changes an estimator, a null, the evidence rule or a bench
protocol, so the MPC-Bench results stay reproducible from the tag. They fix
the empirical pipeline and the interface between its outputs and the
hypothesis registry, make the default protocol explicit, guard Hunter
campaigns against infeasible IIM sizes, and correct two descriptive numbers
in the preregistration. Hypotheses, statistics, margins, minimum n and
decision rules are unchanged. Each change has a regression test.

#### Fixed

- **NAS capacity without a declared hub no longer crashes an empirical run.**
  `compute_NAS(mode="capacity")` requires `workspace_nodes` and raises without
  it (unchanged); `protocols/mpc_default_v1.json` declares NAS capacity but no
  hub, so step 2 of an empirical run under the default protocol stopped with
  `ValueError`. `compute_synergy_ci` now records NAS as UNDEFINED with the
  reason `UNDEFINED:NAS:NO_DECLARED_WORKSPACE` (new detail code
  `evidence.REASON_NO_DECLARED_WORKSPACE`) without calling the estimator, and
  logs one warning; an invalid hub declaration still raises. A run that is
  meant to measure NAS (e.g. a Hunter campaign) passes a derived protocol
  that declares the hub (see the example protocols under "Added"). Tests:
  `tests/test_nas_declared_workspace.py`,
  `tests/test_default_empirical_run.py`.
- **`<P>_estimator_version` columns.** The step-2 outputs (and
  `synergy_ci.MPC_EVIDENCE_COLUMNS`) carry `<P>_estimator_version` for every
  principle: the version part of the exact evidence id in `<P>_estimator`
  (`compute_<P>:<mode>@<version>`, as recorded in
  `ComponentEvidence.estimator`; for a precomputed Hunter IIM result its own
  `iim_algorithm_version`). `scripts/run_predictions.py` requires the
  column; before, every pipeline table was refused (`results lack
  <P>_estimator/<P>_estimator_version`). Test:
  `tests/test_estimator_version_columns.py`.
- **`run_predictions.py` refused pipeline output.** The pipeline writes the
  recorded id `compute_<P>:<mode>@<version>` into `<P>_estimator` and the
  protocol hash as `MPC_protocol_hash`. The evaluator compared
  `<P>_estimator` with the bare registry name and required a
  `protocol_hash` column, so it refused every principle of every pipeline
  row even with all versions registered. It now splits the recorded id
  (`evidence.split_estimator`), reads `<P>_estimator_version` as the version
  or as the recorded id, refuses rows whose two columns state different
  names or versions (never guessed), and reads `MPC_protocol_hash` when
  `protocol_hash` is absent (when both are present they must agree). Tests:
  `tests/test_predictions_end_to_end.py` feeds a synthetic step-2 table in
  the pipeline's column format through the command line, and
  `compute_synergy_ci` output on a tiny synthetic layout through `evaluate`:
  registered versions are accepted (confirmatory stratum); a later IIM
  version, the directional cut, the RAM fallback, an inconsistent id and the
  registry-0.2.0 IIM name are refused; `--allow-unregistered` gives a
  labelled exploratory run. `tests/test_estimator_version_columns.py` runs
  the same check on the pipeline's own step-2 columns.
- **IIM estimator id in the hypothesis registry**
  (`predictions/registry.yaml`). The IIM entry was named
  `compute_IIM:delta_psi`, but the evidence layer records
  `compute_IIM:<cut_mode>@<version>`
  (`compute_IIM:bidirectional@iim-v4-2026.09` under
  `protocols/mpc_default_v1.json`), so `scripts/run_predictions.py` would
  have refused every IIM row. The entry is now `compute_IIM:bidirectional`;
  the RAM, PDI, NAS and SRPI names already matched. The registry validator
  now refuses an estimator name that is not `compute_<P>:<mode>` of its own
  principle (the version belongs in `version`). Tests:
  `tests/test_predictions_end_to_end.py` (registry names equal the ids
  `synergy_ci` records under the registry's protocol file; the declared RAM
  and SRPI fallbacks stay unregistered) and
  `tests/test_predictions_registry.py` (malformed names are refused).
- **Container images ship `protocols/`.** The Docker and Apptainer images
  contained no protocol file, so a default empirical run in a container
  stopped with `MPC protocol not found`. Test:
  `tests/test_packaging_infra.py`.
- **No absolute local paths in the dataset reports.**
  `data/managed/report_dataset_inventory.json` and
  `data/managed/report_dataset_snapshot_status.json` recorded the dataset
  roots as absolute paths on the machine that wrote them. They now hold
  repository-relative paths (e.g. `data/scratch/ds003171`), and
  `dataset_catalog.build_inventory` writes `local_root` and
  `annex_objects_root` relative to the repository root: a dataset linked in
  from another disk is recorded under its path in the repository, while
  reading still resolves the link (`resolve_local_dataset_root`). Test:
  `tests/test_dataset_catalog.py`.

#### Changed

- **Default protocol for empirical runs: the preregistered
  `protocols/mpc_default_v1.json`**, byte-identical to the freeze (hash
  `383eb310…`). `run_pipeline.py` now uses it when `--protocol` is not given
  for real data (not with `--necessity-set`) and logs its source and hash;
  before, the command line built a protocol from the flags unless
  `--protocol` was given. `--protocol none` builds the protocol from the flags
  as before, and dummy data always do. Under v1 RAM declares its
  unimplemented `perturbational` and `endogenous` channels, so RAM can be
  PRESENT but never ABSENT: behavioural non-response alone never excludes,
  because covert, perturbational and endogenous responsiveness are not
  measured (the preregistered stance, confirmed at the freeze). NAS capacity
  has no hub in v1 and is UNDEFINED (`NO_DECLARED_WORKSPACE`) unless a
  derived protocol declares one. Tests: `tests/test_protocol_examples.py`,
  `tests/test_default_empirical_run.py` (end to end on tiny synthetic
  layouts: under the default, RAM is PRESENT on responsive runs and, on
  behaviourally null runs, its behavioural channel is ABSENT while RAM stays
  UNDEFINED), `tests/test_protocols.py`, `tests/test_docs_consistency.py`
  (the documents name v1 as the default and the behavioural-RAM caveat,
  documented Hunter campaigns pass a protocol with a declared hub, and no
  file outside this changelog refers to the withdrawn draft default).
- **Why v1 and not a behavioural-only RAM protocol.** An intermediate draft
  of these fixes had shipped the behavioural-only RAM protocol (see "Added")
  as `protocols/mpc_default_v1.1.json` (name `mpc-default-v1.1`, hash
  `4a94a79c…`) and made it the command-line default, on the grounds that
  under v1 RAM can never contribute an exclusion. That reversed the
  preregistered, freeze-confirmed stance (preregistration section 4,
  "Default protocol"; `protocols/README.md`) on which the accompanying
  papers rest: behavioural non-response is not evidence that responsiveness
  is absent, and undefined is not absent. It was undone before release: v1
  is the default, the draft's file was renamed and made opt-in, and the
  example protocols were re-derived from v1 (the draft's
  `protocols/examples/mpc_default_v1.1_*` files are removed).
- **Stance summary: a FALSIFIED outcome must survive the worst-case
  sensitivity analysis** (`scripts/run_predictions.py`, now
  `run-predictions/1.1.0`). A FALSIFIED outcome of H1 or H3-H7 counts
  against the stance only if the registered worst-case sensitivity analysis
  (UNDEFINED statuses / UNDETERMINED verdicts of report-positive episodes
  counted as not ABSENT / not EXCLUDED for the lower bound) also returns
  FALSIFIED; otherwise it is listed under `falsified_not_confirmed` and does
  not count. H8 has no missingness dimension and counts as evaluated. Every
  result row records `counts_against_stance`. Reason: INCONCLUSIVE statuses
  are UNDEFINED, so a weakly expressed capacity can inflate the primary
  ABSENT rate when necessity holds; this is the rule stated in the companion
  article (Sections 3 and 9, supplementary preregistration). The registry
  declares it (`defaults.stance_falsification: sensitivity_confirmed`,
  required by the schema), and the validator refuses a registry in which a
  hypothesis on the EXCLUDED or ABSENT rate among report-positive episodes
  (H1, H3-H7) counts for the stance without `sensitivity_missing:
  worst_case`. Hypothesis outcomes, statistics, margins, minimum n and
  decision rules are unchanged. Tests: `tests/test_predictions_registry.py`
  (primary FALSIFIED with an unconfirming and a confirming worst case; H8;
  rule edge cases; registry mutations; command-line report fields).
- **Hypothesis registry 0.3.0-draft** (from 0.2.0). Besides the IIM id and
  the stance rule above, the protocol note no longer says the cutoffs are
  "to be justified by MPC-Bench dose-response": it states the preregistered
  keep-unless rule (keep `(z, delta, alpha) = (0.25, 0.10, 0.05)` unless an
  anchored principle's development false-PRESENT rate exceeds `alpha + 0.02`
  at `z`, or the nominal positive control is ever ABSENT; neither occurred)
  and the frozen values z = 0.25, δ = 0.10, α = 0.05. The schema's
  `registry_version` admits a pre-release suffix, and a frozen registry must
  carry a release version (the suffix is dropped at the freeze).

#### Added

- **Example protocols with a declared NAS hub** (`protocols/examples/`), two
  **examples** derived from `mpc_default_v1.json`:
  `mpc_default_v1_schaefer400_7networks_hub.json` (Schaefer-400 / 7
  networks: the `Cont`, `DorsAttn` and `SalVentAttn` parcels read from the
  atlas order file) and `mpc_default_v1_eeg64_hub.json` (a 64-channel 10-20
  montage: fronto-parietal sensors in the pipeline's sorted channel order),
  each with a `.derivation.json` sidecar (base protocol, its name, hash and
  RAM channels, hub rule, source file and SHA-256, node order). They are
  built and checked by `scripts/build_example_protocols.py` (`--check` also
  fails on a protocol file in `protocols/examples/` that is not part of a
  fresh build; `--base protocols/mpc_behavioural_ram_v1.json --out-dir
  <dir>` derives them from the opt-in behavioural-RAM protocol, named after
  it, with its interpretation caveat in the sidecars). Both must be
  preregistered before confirmatory use. The EEG example uses the BioSemi-64
  labels and does not apply to ds005620 (62 scalp channels, another 10-10
  layout; its indices would be in range but name other channels); the
  builder derives a hub from a dataset's `channels.tsv` (`--eeg-channels`,
  classified as the EEG preprocessing does: BOM-safe, EOG/EMG/ECG names and
  `status=bad` excluded) under the distinct name
  `EXAMPLE-mpc-default-v1-eeg-custom-hub`. Test:
  `tests/test_protocol_examples.py`.
- **`protocols/mpc_behavioural_ram_v1.json` (opt-in only)**: v1 with RAM
  declared on its behavioural channel only (`channels.RAM = ["default"]`,
  name `mpc-behavioural-ram-v1`, hash `531c15b9…`), everything else
  identical. Under it RAM can be ABSENT, so behavioural evidence alone can
  exclude. It is never selected automatically (`--protocol
  protocols/mpc_behavioural_ram_v1.json`), selecting it is a substantive
  choice to preregister, and results under it are behavioural-RAM results
  (an ABSENT RAM means no responsiveness-and-adaptation above the null in
  the recorded behaviour, not absence of responsiveness). Test:
  `tests/test_default_empirical_run.py` (the behaviourally null runs that
  are RAM UNDEFINED under v1 are RAM ABSENT and EXCLUDED under it).
- **`scripts/compute_empirical_reference.py`**: an external reference anchor
  (per-principle rho and SE on the excess scale) from the high-state runs of
  a declared held-out subset of participants (`--reference-subjects`, checked
  to be disjoint from `--evaluation-subjects`), from a pipeline step-2 table
  (`--step2-table`, protocol hash checked) or computed from a preprocessed
  layout (`--data-dir`). Participant means first, SE = SD / sqrt(n), and the
  bench anchor rule (a positive one-sided 95% t lower bound); principles
  without an anchor are listed with the reason. Default `--protocol`:
  `protocols/mpc_default_v1.json`. Channels: a principle with one declared
  channel gets the key `P`; one with several gets `P:channel` per channel
  with an estimator (under v1 `RAM:default`), and channels without an
  estimator get none (`channels_without_reference`, reason
  `NOT_IMPLEMENTED`). Per-channel evidence comes from `--data-dir` mode (new
  opt-in `compute_synergy_ci(record_channel_evidence=True)`, which lists
  every run's record per principle and channel in
  `df.attrs['mpc_evidence']['channel_evidence']`; the pipeline's outputs are
  unchanged). A step-2 table cannot identify the channel of its `<P>_*`
  columns (they describe the deciding channel; under v1 an unimplemented RAM
  channel whenever the behavioural one is ABSENT), so in `--step2-table`
  mode such a principle gets no reference (`CHANNEL_NOT_IDENTIFIABLE`),
  never a guess. A principle that was not computed is `NOT_COMPUTED` in both
  modes. The JSON records the provenance (evidence source and SHA-256,
  evidence mode, protocol hash, declared subset and its SHA-256,
  participants used and missing, estimator ids and versions, null settings,
  code version); `--write-protocol` writes the protocol with the external
  reference. Test: `tests/test_empirical_reference.py` (tiny synthetic
  layouts; equals the pipeline's cohort reference of the same channel and
  runs).
- **Hunter IIM cost guard** (`impact_pipeline.hunter_cost`). Exhaustive IIM
  grows roughly 30x per added subsystem node (one core, T = 1000: 1.7 s at 4
  nodes, 46 s at 5, about 27 min at 6), and the default subsystem of 10
  nodes needs about 4.2e11 Psi evaluations (about 7 core-years) per IIM run,
  but the campaign build accepted any configuration without saying so.
  `--execution-mode hunter --hunter-stage build-campaign` and
  `prepare_hunter_campaign` now count the work in Psi evaluations
  (mechanisms x purviews x bipartitions x (1 + cuts) x runs x (1 + K_null +
  K_boot); one evaluation is one mechanism/purview/bipartition-pair term of
  Psi under one TPM). The count is made in closed form before anything is
  prepared (from each run's region count and the state-budget rule of
  `prepare_iim_problem`; an upper bound) and again from the enumerated
  problems afterwards. The build logs it with an order-of-magnitude runtime:
  the reference single-core rate of 5.6e-4 s per evaluation, or a measured
  rate from `--hunter-seconds-per-psi-eval` /
  `IMPACT_HUNTER_SECONDS_PER_PSI_EVAL`. It warns when the longest shard would
  exceed its walltime and records everything in `campaign_manifest.json`
  (`iim_cost_estimate`, with `preflight`) and `pbs/campaign_plan.json`.
  Configurations above the ceiling (`--hunter-max-psi-evals` >
  `IMPACT_HUNTER_MAX_PSI_EVALS` > 1e11) are refused before anything is
  written (`HunterCostError`), unless `--hunter-allow-large` is given.
  `python -m impact_pipeline.hunter_cost` prints the same counts per
  subsystem size without data. Tests: `tests/test_hunter_cost.py`
  (closed-form counts equal the `prepare_iim_problem` enumerations, refusal
  with no files written, environment ceiling, `--hunter-allow-large`,
  command-line flags, manifest records).
- **Measured cost calibration.** Phase-1 and cut shards record the Psi
  evaluations they computed next to their wall time
  (`timing/<stage>/<task>.json`; resumed cuts are not counted). `reduce-all`
  (the last step of the smoke job) and `status` write
  `cost_calibration.json` with `shard_wall_seconds_per_psi_evaluation`,
  per-stage totals, the per-shard spread and an `overhead_dominated` flag
  below 1e6 timed evaluations. `scripts/hunter/hunter_smoke_test.sh` takes
  `--iim-max-nodes`, `--iim-max-mechanism-size`, `--iim-max-purview-size`
  and `--iim-n-parts` (`all` = exhaustive), so the same helper runs the
  calibration.

#### Documentation

- `docs/HLRS_HUNTER_RUNBOOK.md`: a new section 10a, "Size the IIM
  configuration before the full campaign", linked from the top of the
  runbook. It covers why (growth per node, the infeasible 10-node default),
  what the guard does, the calibration smoke test, extrapolation, how the
  author chooses `iim_max_nodes`, mechanism/purview sizes and `n_parts`, and
  what to report back. Sections 8.2, 9, 14-17 and 19 were updated to match
  (the preprocessing-only build of section 8.2 now sets `--iim-max-nodes 4`;
  without it the guard refuses the 10-node default after preprocessing).
  `scripts/hunter/README.md` has a new section, "IIM cost guard and sizing",
  and `docs/ARCHITECTURE.md` shows the guard in the campaign flow. The
  runbook, `scripts/hunter/README.md` and `README.md` also say which
  protocol a Hunter campaign passes (one derived from v1 with a declared NAS
  hub), and the runbook shows how to check its hash in the build log.
- `docs/preregistration/MPC_BENCH_PREREGISTRATION.md`: a dated section,
  "Errata (documentation only; no change to hypotheses or decision rules)".
  The whole-brain grain is 76 regions / 265 edges; 209 is the minimum edge
  confidence in the connectome file name. There are 260 witness tasks per
  family (13 x 20 seeds), not 280. The frozen text itself is unchanged
  (`tests/test_hunter_cost.py` compares it with the tag).

### Added

- **Evidence layer** (`impact_pipeline.evidence`). `ComponentEvidence`, the
  component status PRESENT / ABSENT / UNDEFINED, strong-Kleene OR over the
  evidence channels of a principle and AND over the necessity set, and the MPC
  verdict `EXCLUDED` / `MPC_CONSISTENT` / `UNDETERMINED` with stable reason codes
  (`MISSING`, `MISSING_CHANNEL`, `NO_NULL_CALIBRATION`, `NO_SAMPLING_SE`,
  `INVALID_ANCHORS`, `INCONCLUSIVE`, `UNDEFINED` (e.g. `DEGENERATE_NULL`,
  `INVALID_SE`, `NULL_FAMILY_MISMATCH`), `NOT_IMPLEMENTED`,
  `ESTIMATOR_NOT_VALIDATED`, `ABSENT`, `BEARER_MISMATCH`, `PROTOCOL_MISMATCH`,
  `SOURCE_INCOHERENT`). The verdict is recoverable from its reasons. The
  component status is judged on the construct scale `c = (m - nu)/(rho - nu)`
  with a sampling SE (bootstrap/jackknife, plus the Monte-Carlo error of the
  null mean and the reference SE), declared cutoffs `(z, delta)` and
  Student-t bounds for SEs from few replicates (`se_df`, Welch-Satterthwaite
  degrees of freedom). `Protocol` (JSON schema `impact-mpc-protocol/2`, SHA-256
  hash): necessity set, declared channels, cutoffs, alpha, null families,
  reference (cohort high state or external), source rule, estimator modes and
  bearer nodes. `joint_dependence` (single-source constraint: Gaussian total
  correlation of the node sets' summaries against independent circular
  shifts). Applicability registry schema `impact-mpc-registry/2` with entry
  criteria (null false-PRESENT rate, recovery slope, forward-model substrates
  for human EEG/fMRI). The MPC degree (capped weighted power mean of the
  construct-scale components, geometric by default, only for `MPC_CONSISTENT`
  rows), `weakest_link`, `degree_interval` (delta method and bootstrap),
  `verdict_stability` and the `bearer_coherence` diagnostic. Property tests on
  seeded random configurations cover missingness safety, monotone resolution,
  determinacy iff all completions agree, veto, permutation symmetry, channel
  disjunction, reason-code decomposability, declared channels without items
  and missing SEs.
- **Generic nulls** (`impact_pipeline.nulls`): circular shift, uni- and
  multivariate phase randomisation, IAAFT, label permutation (optionally
  stratified by phase bin), onset jitter and `component_null`, with seeds derived
  per run; the moving-block bootstrap `block_bootstrap` /
  `component_bootstrap_se` (events move with their blocks, yoked pairs stay
  intact).
- **Verdict wiring**: `compute_synergy_ci(..., null_surrogates, necessity_set,
  applicability_registry, null_seed, null_kinds, protocol, bootstrap_se,
  bootstrap_block_len)` and the CLI flags `--protocol`, `--null-surrogates K`,
  `--bootstrap-se B`, `--bootstrap-block-len`, `--necessity-set`,
  `--applicability-registry`. New step-2 columns `MPC_verdict`, `MPC_reason`,
  `MPC_degree`, `MPC_necessity_set`, `MPC_null_surrogates`, `MPC_null_seed`,
  `MPC_null_families`, `MPC_bootstrap_se`, `MPC_bootstrap_block_len`,
  `MPC_protocol_hash`, `MPC_joint_dependence(_p)` and per principle
  `<P>_status`, `<P>_margin`, `<P>_margin_absent`, `<P>_estimate`,
  `<P>_null_mean`, `<P>_null_sd`, `<P>_null_n`, `<P>_se`, `<P>_se_df`,
  `<P>_boot_n`, `<P>_boot_failed`, `<P>_c`, `<P>_c_se`, `<P>_c_df`,
  `<P>_c_lower`, `<P>_c_upper`, `<P>_reference(_se)`, `<P>_estimator`,
  `<P>_mode_reason`, `<P>_channels`. Estimator modes come from the protocol
  (or the params dicts), with declared per-run fallbacks for modes that need
  inputs a run may lack (RAM `update_fallback`, SRPI `mode_fallback`). With
  K = 0 the legacy modes are `NO_NULL_CALIBRATION` and with B = 0 every
  component is `NO_SAMPLING_SE`, so every verdict is `UNDETERMINED`.
- **Protocols** (`protocols/`): `mpc_default_v1.json` (default for empirical
  data; selected automatically by `run_pipeline.py` since the post-freeze
  fixes) and the MPC-Bench protocols `mpc_bench_v1.json` (all five principles)
  and `mpc_bench_v1_anchored.json` (necessity set NAS, IIM, SRPI), frozen at
  the local tag `mpcbench-freeze-v1` with their rationale and hashes. The bench
  reference anchor (development positive control, seeds 900-919) follows an
  anchor rule: a principle is anchored only if its mean excess over the null is
  credibly positive (one-sided 95% t bound); RAM and PDI are not, so they are
  `INVALID_ANCHORS` on the bench.
- **Preregistration of the MPC-Bench hypotheses** (`docs/preregistration/`):
  HC1-HC10 with decision rules, the calibration decisions and development
  findings, and the confirmatory run plan (`scripts/mpcbench_confirmatory.sh`);
  not registered publicly yet. `scripts/bench_hypotheses.py` evaluates the
  hypotheses on the confirmatory runs (refuses development records, records
  without the freeze tag and unfrozen protocol hashes);
  `scripts/calibrate_bench.py` produces the development calibration evidence
  (development records only); `scripts/iim_validation.py` compares the sampled
  IIM with the exact TPMs of family B; `impact_pipeline.bench.analysis` holds the
  shared construct-scale helpers. The bench runner, the reference and the null
  calibration apply the protocol's declared estimator options (e.g. the IIM cut
  mode); bench records keep the jackknife replicates (`se_replicates`) and the
  IIM grain and cut mode; `null_calibration.py` has a development / confirmatory
  seed policy (`--confirmatory --freeze-tag`). Default jackknife groups are
  G = 10 (was 5) in the bench scripts, as in the frozen protocol.
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
  - `compute_PDI(mode="repertoire")`: the repertoire of distinguishable
    states in bits, labelled (cross-validated decoding of declared states,
    Miller-Madow mutual information, block label-permutation null) or
    unlabelled (held-out count of recurring, persisting, separated states
    against circular-shift or Fourier surrogates); selectable in the pipeline
    through the protocol (unlabelled) and used by MPC-Bench.
  - `compute_PDI(mode="surrogate_excess")`: repertoire entropy, LZ76 diversity
    and binarised effective dimensionality as signed excess over multivariate
    spectrum-preserving surrogates; a documented negative result (the Gaussian
    null is maximum-entropy for the given spectra), kept for reference.
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
  confirmatory seed policy with a code-freeze guard, rival decision rules
  (union, count-k, means, weakest link, naive Bayes, product of credences,
  logistic classifier, single markers) and comparators (LZ76, closed-form
  Gaussian Φ_R for VAR(1)). MPC-Bench v2 adds family C manipulation checks, the
  whole-brain Hopf generator on the shipped connectome with EEG-like and
  BOLD-like forward models, adversarial constructions, graded patchworks, a
  rule audit on estimated statuses with risk-coverage curves
  (`scripts/benchmark_attribution_rules.py`), jackknife SEs (`--se-groups`,
  G - 1 degrees of freedom), verdicts under the bench protocol (`--protocol`)
  whose external reference is the positive control on development seeds
  (`scripts/bench_reference.py`), and PDI in `repertoire` mode.
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
  the same function as the local path); `--hunter-iim-bootstrap-se B`
  (block-bootstrap replicate runs, drawn as the local pipeline draws them, so
  the reducer reports `Delta_Psi_bootstrap_se` and Hunter IIM evidence has a
  sampling SE); the protocol's IIM options and bearer nodes reach the
  campaign; `iim_results.csv` and
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
- **Analysis and preregistration tooling**: `impact_pipeline.necessity` (NCA
  ceilings, symmetric three-outcome necessity criteria, verdict-level
  summaries), `scripts/audit_aggregation.py`, `scripts/necessity_power.py`
  (component and verdict level), `scripts/simulate_rule_recovery.py`,
  `scripts/definedness_audit.py`, `scripts/null_calibration.py` (v2 evidence
  rule under a protocol), `scripts/run_predictions.py` with the draft
  hypothesis registry `predictions/registry.yaml` (H0-H10) and its schema,
  and one script per figure (`scripts/figures/`).
- Documentation: `docs/HLRS_HUNTER_RUNBOOK.md`, `docs/ARCHITECTURE.md`, this
  changelog; `docs/metrics.md` rewritten against the code.
- `protocols/applicability_registry_v1.json`, the first applicability
  registry built from benchmark evidence, and
  `scripts/build_applicability_registry.py`, which derives it from the
  confirmatory MPC-Bench results with the entry criteria of the preregistration
  (valid anchor, null false-PRESENT rate, dose-response, specificity without
  the mechanism, cross-talk). Validated: SRPI-agency on the rate agents,
  NAS-capacity and bidirectional IIM on the oscillator agents; every other
  estimator and substrate is listed as excluded with the failed criteria. No
  forward-modelled EEG or BOLD entry exists, so human EEG and fMRI components
  are `ESTIMATOR_NOT_VALIDATED` under this registry.

### Changed

- **Verdict names** follow the necessity-only stance: `EXCLUDED` (a principle is
  credibly absent), `MPC_CONSISTENT` (all principles present; not an
  attribution of consciousness) and `UNDETERMINED`. The component status is
  judged on a two-anchor construct scale with a sampling SE and declared
  smallest effects of interest (`docs/metrics.md`, section 8). `<P>_margin`
  is the construct-scale presence margin `c_lower - z`;
  `assemble_mpc_degree(df, weights, p, cap)` reads the `<P>_c` columns and
  returns the DataFrame; the evidence reference comes from the protocol only
  (`--ci-reference` affects the legacy CI); a tie in `verdict_stability` gives
  `UNDETERMINED`.
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
