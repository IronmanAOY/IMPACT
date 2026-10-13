[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23339786.svg)](https://doi.org/10.5281/zenodo.23339786)

# IMPaCT Synergy Pipeline

A measurement framework for the Minimal Principles for Consciousness (MPC) of
the IMPaCT framework, for EEG and fMRI data and for simulated systems. It
computes five null-anchored estimators and combines them with a three-valued,
missingness-safe rule:

- **MPC profile**: the five components, each calibrated against a declared null
  family and put on a construct scale with an interval:

  | Code | Estimator | Principle |
  |---|---|---|
  | RAM | Responsiveness-Adaptation Metric | goal-directed responsiveness and feedback-driven adaptation |
  | PDI | Pattern Differentiation Index | differentiation of the state repertoire |
  | NAS | Network Availability Score | receive-and-return broadcast through a workspace |
  | IIM | Integrated-Information Measure | integration (an IIT-inspired proxy from an estimated causal TPM; not IIT's Φ) |
  | SRPI | Self-Referential Processing Index | self/other and self-caused/other-caused processing |

- **MPC verdict**: `EXCLUDED`, `MPC_CONSISTENT` or `UNDETERMINED`, with reason
  codes. The principles are treated as necessary conditions, which license
  exclusion only. `EXCLUDED` means some principle of the necessity set is
  credibly absent. `MPC_CONSISTENT` means every principle is credibly present,
  so the MPC stance does not exclude consciousness; it is **not** an attribution
  of consciousness. Anything else, including missing or inconclusive evidence,
  is `UNDETERMINED`. Missing evidence never becomes a zero and never flips a
  determinate verdict.
- **MPC degree**: a capped power mean of the construct-scale components,
  computed only for `MPC_CONSISTENT` rows. It is a reference-relative evidence
  summary, not a level of consciousness.

Everything a verdict depends on besides the data (necessity set, channels,
construct-scale cutoffs, null families, reference anchor, source rule and
estimator modes) is declared in a [protocol](#protocols) whose SHA-256 is
recorded with every verdict.

The empirical pipeline (`run_pipeline.py`) computes the estimators of
`impact_pipeline.mpc_metrics` and judges them with the interval rule. The
benchmark [MPC-Bench](#benchmark-mpc-bench) validates the estimators and the
exclusion rule on white-box simulated systems; it also runs variants of RAM,
PDI, NAS and IIM that read declared inputs (`impact_pipeline.estimators`) and
judges them with the TOST rule, in which absence is a two-sided equivalence
test.

The legacy geometric-mean Consciousness Index (`CI` column, `compute_CI`) is
kept for backward compatibility. It is deprecated and is not a gate. The
exploratory HypergraphSynergy statistic `S` is reported separately and never
enters CI or the verdict.

Documentation: [`docs/metrics.md`](docs/metrics.md) defines the estimators and
their variants, the null calibration, both status rules, the protocols and the
benchmark; [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes the code
structure; [`protocols/README.md`](protocols/README.md) the shipped protocols;
[`CHANGELOG.md`](CHANGELOG.md) the changes.

## Installation

The current release is 1.2.0, tagged `v1.2.0`:

```bash
git clone --branch v1.2.0 https://github.com/IronmanAOY/IMPACT.git impact-synergy-pipeline
cd impact-synergy-pipeline
```

Python 3.10 with conda (the reference environment the tests run on):

```bash
conda env create -f environment.yml
conda activate impact-synergy-clean
python -m pip install --no-deps --no-build-isolation -e .
```

Or with pip into any Python >= 3.10 environment (dependency ranges from
`pyproject.toml`; `dev` adds pytest, flake8 and pre-commit):

```bash
python -m pip install -e ".[dev]"
```

The pre-commit hooks (`.pre-commit-config.yaml`) are optional. `pre-commit`
is not part of the locked conda environment; install it separately
(`python -m pip install pre-commit`, then `pre-commit install`).

`run_pipeline.py` checks that it runs in the conda env `impact-synergy-clean`
(override the name with `IMPACT_CONDA_ENV`, or skip the check with
`IMPACT_SKIP_ENV_CHECK=1`). CuPy is not a dependency: install a build that
matches your CUDA/ROCm stack to use `--hardware-target gpu|auto`. For HLRS Hunter
see [`docs/HLRS_HUNTER_RUNBOOK.md`](docs/HLRS_HUNTER_RUNBOOK.md).

External tools for some workflows:

- DataLad or git + git-annex, for `scripts/download_data.sh` (pinned OpenNeuro
  snapshots);
- Docker and a FreeSurfer license, for fMRIPrep (`scripts/fetch_fmriprep.sh`).
  Set `FS_LICENSE=/absolute/path/to/license.txt` or place the file at
  `licenses/fs_license.txt` (template: `licenses/fs_license.txt.example`).

## Quickstart

### 1. Tests

```bash
python -m pytest -q
```

The full suite takes about 15 minutes on a workstation and needs no data
outside the repository.

### 2. Benchmark smoke test (no downloads)

MPC-Bench simulates white-box systems with switchable mechanisms and scores them
with the estimators. The positive control and a null witness of family A on one
development seed, three principles (about a minute on a laptop):

```bash
python scripts/run_bench.py run A_witnesses --split development --seeds 320 \
    --systems PC_nominal,N_independent_noise --principles RAM,PDI,SRPI \
    --workers 2 --out outputs/mpcbench/development/smoke
```

It writes `run_plan.json`, `results.jsonl` (one record per task: one
simulation scored under the input declarations R and H), `components.csv`
(one row per scoring and principle) and `run_manifest.json` (code identity,
environment, protocol hashes) to `outputs/mpcbench/development/smoke/`. The
components are judged by the family protocols `protocols/bench/A-R.json` and
`A-H.json`: on the positive control RAM, PDI and SRPI are PRESENT; on the
null witness PDI is ABSENT, and RAM and SRPI are UNDEFINED
(`ABSENT_NOT_REACHABLE`).

### 3. Real-data-derived synthetic objects

`scripts/generate_real_derived_synth_completed.py` plants known structure on
real OpenNeuro recordings and re-validates the objects with the metric code.
These are software smoke-test objects, not validation of the theory. See
[`docs/synthetic_data.md`](docs/synthetic_data.md) for generation from the
OpenNeuro sources. To re-validate existing objects (for example one EEG
subject):

```bash
export IMPACT_SYNTH_ROOT=/absolute/path/that/contains/test_objects
python scripts/generate_real_derived_synth_completed.py --validate-only \
    --datasets ds005620 --subjects 1010
```

The exit code is 0 when the smoke gate passes. Results go to
`$IMPACT_SYNTH_ROOT/test_objects/real_derived_synth_completed/revalidation/<UTC time>/`
(or `--validation-out`); the objects are never modified.

### 4. Local runs on OpenNeuro data

Download the data, then run the pipeline. `--run-preprocessing` extracts the
time series first (fMRI: from fMRIPrep derivatives at
`<bids-root>/derivatives/fmriprep` or `--fmriprep-dir`; EEG: from the BrainVision
files). A determinate MPC verdict needs a null family and a sampling SE for
every component: `--null-surrogates K` calibrates the legacy estimator modes
against K surrogates (default 0: `NO_NULL_CALIBRATION:<P>`) and
`--bootstrap-se B` adds B moving-block bootstrap replicates per run and
component (default 0: `NO_SAMPLING_SE:<P>`); with either at 0 every verdict is
`UNDETERMINED`. Runtime grows roughly (K+1)-fold and (B+1)-fold per component.
On empirical data the pipeline uses `protocols/mpc_default.json` unless
`--protocol` says otherwise (see [Protocols](#protocols)). Under it RAM can be
PRESENT but never ABSENT, and NAS is UNDEFINED in every run
(`UNDEFINED:NAS:NO_DECLARED_WORKSPACE`) unless a protocol derived from the
default declares the hub for your grain. Use `--iim-max-nodes` to bound the
IIM subsystem on a workstation.

fMRI (ds003171, propofol sedation; sessions `awake` and `deep`):

```bash
bash scripts/download_data.sh ds003171 2.0.1 /data/openneuro/ds003171
export FS_LICENSE=/absolute/path/to/license.txt
bash scripts/fetch_fmriprep.sh --bids-root /data/openneuro/ds003171   # Docker, hours per subject

python run_pipeline.py --dataset-id ds003171 \
    --bids-root /data/openneuro/ds003171 --out-dir outputs/ds003171 \
    --run-preprocessing --iim-max-nodes 5 --null-surrogates 19
```

EEG (ds005620; outputs go to `<out-dir>/ds005620`):

```bash
bash scripts/download_data.sh ds005620 1.0.0 /data/openneuro/ds005620

python run_pipeline.py --dataset-id ds005620 \
    --bids-root /data/openneuro/ds005620 --out-dir outputs \
    --run-preprocessing --subjects 1010 1016 --iim-max-nodes 4 --null-surrogates 3
```

This two-subject example takes several minutes on a laptop. ds005620 has no
goal/feedback or self/other events, so RAM and SRPI are undefined
(`UNDEFINED:RAM:missing_stimulus_events`,
`UNDEFINED:SRPI:missing_self_and_nonself_events`), the legacy CI is undefined,
NAS is UNDEFINED as well (`UNDEFINED:NAS:NO_DECLARED_WORKSPACE`: the default
protocol declares no hub, and the 64-channel example in `protocols/examples/`
does not fit ds005620's montage), and no verdict can be `MPC_CONSISTENT`.
Three surrogates only exercise the code; use at least 19 for inference.

Useful options (`python run_pipeline.py --help` lists all):

| Option | Effect |
|---|---|
| `--subjects A B` | restrict to some subjects |
| `--mpc-metrics RAM PDI NAS IIM SRPI` | subset of estimators (default all) |
| `--protocol protocol.json` | MPC protocol (`evidence.Protocol` JSON: necessity set, channels, cutoffs, null families, reference, source rule, estimator modes, bearer nodes); its hash is recorded. Default on real data: `protocols/mpc_default.json` (not with `--necessity-set`; declares no NAS hub, see `protocols/examples/`); `none`: built from the flags |
| `--null-surrogates K` | null calibration per run and component (0 = none) |
| `--bootstrap-se B`, `--bootstrap-block-len L` | block-bootstrap sampling SE per run and component (0 = none; block default ceil(sqrt(n_time)) samples) |
| `--necessity-set RAM,PDI,NAS,IIM,SRPI` | principles the verdict requires (default all five; must match `--protocol`) |
| `--applicability-registry registry.json` | evidence from unvalidated estimators becomes UNDEFINED |
| `--ci-reference cohort_high_state\|file.json` | reference means of the legacy CI (the evidence reference comes from the protocol) |
| `--no-ci` | skip the legacy CI |
| `--reuse-step2` | reuse `<out-dir>/cache/step2_*.csv` and rerun only the statistics and report |
| `--hardware-target cpu\|auto\|gpu\|hunter-apu` | NumPy or CuPy kernels (explicit GPU/APU targets fail early if no device) |
| `--assume-tr SECONDS` | TR for fMRI runs without a plausible TR in header or sidecar (otherwise an error) |
| `--no-atlas-robustness` | skip step 6 (AAL-116 and Shen-268 atlases) |
| `--data-origin dummy` | synthetic data; outputs are kept under `test_objects/` |

`scripts/run_all.sh [--dry-run] [ds003171|ds005620]` chains download, atlases,
fMRIPrep and `run_pipeline.py` for one dataset.

## Outputs

A local run writes to `<out-dir>` (for datasets other than ds003171 a
`<dataset>` subfolder is added unless `--out-dir` already ends with it):

```text
preprocessed/<subject>/<session>/<condition>/<subject>_run-<k>_<atlas>_ts.npy   time series (time x nodes)
preprocessed/<subject>/<session>/rest/...                                    PDI baselines
cache/step2_df.csv              one row per run and theta: metrics, legacy CI, MPC verdict and evidence
cache/step2_df_mean.csv         subject x session means of the metric columns
cache/step2_theta_stats.csv     exploratory S by theta
cache/provenance_manifest.json  code version (package + commit), runtime versions, parameters, hardware
stats/statistics_summary.json   paired tests (subject bootstrap, two-sided permutation, Holm), CI definedness
stats/component_tests.csv, theta_tests_S.csv, motion_covariates.csv (fMRI)
pipe_figures/                   figures
IMPaCT_Empirical_Validation_<dataset>.docx   Word report
```

Main columns of `step2_df.csv` (full list: `docs/metrics.md`, section 13):

| Column | Content |
|---|---|
| `subject`, `session`, `theta` | run identity; theta of S |
| `RAM`, `PDI`, `NAS`, `IIM`, `SRPI` | metric values; with `--null-surrogates K > 0` the excess over the null mean, floored at 0; NaN when undefined |
| `MPC_verdict` | `EXCLUDED`, `MPC_CONSISTENT` or `UNDETERMINED` |
| `MPC_reason` | `;`-joined reason codes, e.g. `MISSING:RAM`, `NO_NULL_CALIBRATION:IIM`, `NO_SAMPLING_SE:PDI`, `INVALID_ANCHORS:NAS`, `ABSENT:NAS`, `UNDEFINED:SRPI:<reason>` |
| `MPC_degree` | MPC degree (`MPC_CONSISTENT` rows only) |
| `<P>_status` | PRESENT, ABSENT or UNDEFINED per principle |
| `<P>_estimate`, `<P>_null_mean`, `<P>_null_sd`, `<P>_null_n` | raw estimate and null moments (IIM on the ΔΨ scale, bits) |
| `<P>_se`, `<P>_se_df`, `<P>_boot_n`, `<P>_boot_failed` | bootstrap sampling SE, its degrees of freedom, valid and failed replicates |
| `<P>_c`, `<P>_c_se`, `<P>_c_df`, `<P>_c_lower`, `<P>_c_upper`, `<P>_margin`, `<P>_margin_absent` | construct scale `c = (m - nu)/(rho - nu)`, its SE, degrees of freedom, one-sided bounds and the margins to the cutoffs |
| `<P>_reference`, `<P>_reference_se`, `<P>_estimator`, `<P>_estimator_version`, `<P>_mode_reason`, `<P>_channels` | reference anchor, estimator id `compute_<P>:<mode>@<version>` and its version, fallback reason, per-channel statuses |
| `MPC_necessity_set`, `MPC_null_surrogates`, `MPC_null_seed`, `MPC_null_families`, `MPC_bootstrap_se`, `MPC_bootstrap_block_len`, `MPC_protocol_hash`, `MPC_joint_dependence`, `MPC_joint_dependence_p` | evidence configuration and the single-source test |
| `CI`, `CI_defined`, `CI_missing`, `CI_reference` | legacy CI (deprecated, not a gate); NaN with the missing components listed when undefined |
| `PDI_anchor`, `PDI_task`, `*_reason`; `IIM_raw`, `IIM_defined`, `IIM_undefined_reason` | endpoints and reasons |
| `S` | exploratory HypergraphSynergy at `theta` |

## Protocols

A protocol is an `evidence.Protocol` stored as JSON (schema
`impact-mpc-protocol/1`); its hash is recorded with every verdict
(`MPC_protocol_hash`). A protocol without a `status_rule` block is judged by
the interval rule, which the empirical pipeline uses; a protocol with the
block is judged by the TOST rule of the benchmark. The shipped protocols, with
their hashes and the reasons for each declaration, are described in
[`protocols/README.md`](protocols/README.md):

- `protocols/mpc_default.json`, the default of `run_pipeline.py` on empirical
  data: the declared estimator modes (RAM prediction-error update, PDI
  repertoire, NAS capacity, SRPI agency, each with its declared fallback).
  RAM also declares two channels that have no estimator yet, so RAM can be
  PRESENT but never ABSENT; NAS declares no hub, so it is UNDEFINED
  (`NO_DECLARED_WORKSPACE`) unless a derived protocol declares one
  ([`protocols/README.md`](protocols/README.md#mpc_defaultjson-default-for-empirical-data)
  explains both).
- `protocols/examples/`: two protocols derived from the default with a
  declared NAS hub (Schaefer-400 / 7 networks and 64-channel EEG). They are
  examples; a hub has to be fixed with the protocol before confirmatory use
  ([`protocols/examples/README.md`](protocols/examples/README.md)).
- `protocols/mpc_behavioural_ram.json`, opt-in only (`--protocol
  protocols/mpc_behavioural_ram.json`): RAM declared on its behavioural
  channel only, so RAM can be ABSENT; how to read results under it is in
  [`protocols/README.md`](protocols/README.md#mpc_behavioural_ramjson-opt-in).
- `--protocol none` builds the protocol from the flags.
- `protocols/bench/`: the benchmark protocols (TOST rule), one per family
  protocol key, and their template.

## Running on HLRS Hunter

`--execution-mode hunter` distributes the IIM computation over PBS Pro array jobs
on HLRS Hunter (AMD MI300A APUs, 4 shards per node, Ψ kernels on the APU) and
then runs the other estimators, the evidence layer, the statistics and the report
in one job. The complete procedure (account, workspace, cray-python venv, CuPy
for ROCm, hardware self-test, data staging, sizing of the IIM configuration,
smoke test on the `test` queue, submission, monitoring, troubleshooting,
hand-back, open questions for HLRS) is in
[`docs/HLRS_HUNTER_RUNBOOK.md`](docs/HLRS_HUNTER_RUNBOOK.md). The helper
scripts and the configuration variables are described in
[`scripts/hunter/README.md`](scripts/hunter/README.md).

```bash
python3 run_pipeline.py --execution-mode hunter --hunter-stage build-campaign \
    --hardware-target hunter-apu --dataset-id ds003171 \
    --bids-root <BIDS> --out-dir <OUT> --iim-max-nodes <N> \
    --protocol <PROTOCOL> \
    --hunter-iim-null-surrogates <K> --null-surrogates <K> \
    --hunter-iim-bootstrap-se <B> --bootstrap-se <B>
bash <OUT>/cache/hunter_iim_campaign/pbs/00_submit_all.sh    # on a Hunter login node
```

`<PROTOCOL>` is the protocol derived from `protocols/mpc_default.json` that
declares the NAS hub for the dataset's grain (the author gives it; for
`schaefer400` the example
`protocols/examples/mpc_default_schaefer400_7networks_hub.json` shows the
form). Without `--protocol` the campaign uses `mpc_default.json`, which
declares no hub, and NAS is UNDEFINED (`NO_DECLARED_WORKSPACE`) in every run.
`--hunter-iim-null-surrogates K` and `--hunter-iim-bootstrap-se B` add the
surrogate and bootstrap replicate runs that give the IIM evidence a null family
and a sampling SE (runbook, section 9.1).

## Benchmark (MPC-Bench)

`impact_pipeline.bench` provides white-box systems for validating the
estimators and the exclusion rule: a modular rate-network agent with five
mechanism switches (family A), binary networks with exact TPMs (family B), a
Stuart-Landau network (family C), graded patchworks of five single-principle
modules, a whole-brain Hopf model on the shipped structural connectome with
EEG-like and BOLD-like forward models (`bench/whole_brain.py`,
`bench/forward.py`), adversarial constructions that fool single estimators
(`bench/adversarial.py`), null systems and a system catalogue
(`src/impact_pipeline/bench/systems.yaml`). Oracle manipulation checks
(`bench/manipulation.py`) test that every switch changes what it should.
Generators never import the estimators; hidden ground truth stays in the
system's oracle.

The runner (`bench/runner.py`, command line `scripts/run_bench.py`) simulates
each task once and scores it under several input declarations, views and
estimator forms. Each estimator runs in its own try block, and the records
(schema `mpc-bench-result/1`) carry the evidence and the status under the
TOST rule of the family protocol (`protocols/bench/`). The estimator variants
and the rule are defined in
[`docs/metrics.md`](docs/metrics.md#14-benchmark-mpc-bench).

```bash
python scripts/run_bench.py designs                     # design modules and designs
python scripts/run_bench.py plan --split development    # tasks and scorings per design
python scripts/run_bench.py list A_sweeps --split development
python scripts/run_bench.py run A_witnesses,A_sweeps --split development \
    --workers 8 --out outputs/mpcbench/development/family_a
python scripts/run_bench.py manipulation --seeds 0-39 \
    --out outputs/mpcbench/development/manipulation
python scripts/null_calibration.py run --split development --kinds ar1 --T 1200 \
    --nodes 8 --replicates 2 --out outputs/mpcbench/development/null_calibration
python scripts/null_calibration.py summarise outputs/mpcbench/development/null_calibration \
    --out outputs/mpcbench/development/null_calibration/summary
python scripts/iim_validation.py --hypotheses exact_targets --workers 6 \
    --out outputs/mpcbench/development/iim_family_b
```

A run resumes from partial results (with `--no-resume` it refuses an existing
results file), and `--n-shards N --shard-index K` runs one shard of a plan.
Without a protocol file, a development run derives a draft protocol from
`protocols/bench/template.json`. `scripts/evaluate_hypotheses.py --spec
<file>` evaluates a declarative hypotheses file (schema
`mpc-bench-hypotheses/1`, rules in `bench/hypotheses.py`; a small example is
`tests/fixtures/hypotheses_example.json`) on the records.

Development and confirmatory runs use separate seeds: development seeds
0-999 and confirmatory seeds 20000 and above (the seeds in between are
refused); one run never mixes the two. A confirmatory run needs
`--confirmatory --freeze-tag <tag>`: a clean checkout whose `src/` and
`scripts/` trees equal those of the tag you name (the code fixed before the
confirmatory data), the family protocol files and the default runner
settings; the outputs record the tag and its commit.

## Analysis tools

Command-line tools beside the pipeline and the benchmark runner (inputs and
outputs in each script's docstring):

| Script | Purpose |
|---|---|
| `scripts/definedness_audit.py` | which principle / channel is definable on which dataset, from BIDS metadata only, with an access log (NAS capacity only with a hub declared in `--protocol`; the default `protocols/mpc_default.json` declares none: `NO_DECLARED_WORKSPACE`) |
| `scripts/compute_empirical_reference.py` | external empirical reference anchor (per-principle rho and SE) from the high-state runs of a declared held-out subset of participants, with provenance; optionally writes the protocol with that reference |
| `scripts/build_example_protocols.py` | the example derived protocols with a declared NAS hub (`protocols/examples/`; `--check` verifies them) |
| `scripts/null_calibration.py` | the null-calibration design of MPC-Bench: plan, run, rate tables (`summarise`) and the A-none classification protocol (`protocol`) |
| `scripts/iim_validation.py` | macro IIM on the family-B cells with exact TPMs |
| `scripts/evaluate_hypotheses.py` | the declarative hypothesis evaluator on MPC-Bench records |

## Dashboard

```bash
python scripts/live_dashboard.py --out-dir outputs/scratch --dataset-id ds003171 --port 8765
```

Then open `http://127.0.0.1:8765`. The dashboard binds to the loopback interface
and has no user authentication; `--host` with a non-loopback address requires
`--allow-remote`. The dashboard selects datasets, configures runs, shows live
metrics and verdicts, and can build (not submit) a Hunter campaign. Synthetic
datasets are routed to `test_objects/`. A desktop launcher is
`scripts/impact_desktop_app.py` (double-click: `scripts/start_impact_desktop.command`
on macOS, `scripts/start_impact_desktop.bat` on Windows).

## Repository layout

```text
run_pipeline.py            command-line entry point (local and Hunter modes)
src/impact_pipeline/       package: estimators, evidence layer, nulls, assembly, statistics,
                           preprocessing, Hunter backend
src/impact_pipeline/estimators/  benchmark variants of the estimators and their declared inputs
src/impact_pipeline/bench/       MPC-Bench: systems, forward models, designs, runner, evaluator
protocols/                 MPC protocols (default, opt-in, examples, bench/) with their rationale
scripts/                   downloads, fMRIPrep, run_all, dashboard, synthetic objects, benchmark,
                           analysis tools, hunter/ (setup, install, smoke test)
tests/                     regression, ground-truth and property tests (pipeline and MPC-Bench)
docs/                      metrics, architecture, Hunter runbook, synthetic data
data/managed/              small versioned reference files (structural connectome)
atlases/                   atlas files (scripts/download_atlases.sh)
licenses/                  third-party notices and the FreeSurfer license template
test_objects/              synthetic validation data and runs (generated locally, not versioned)
environment.yml            reference conda environment (pyproject.toml has the pip ranges)
Dockerfile, Singularity    container definitions for the empirical pipeline
CITATION.cff               citation metadata
```

## Containers

`Dockerfile` and `Singularity` build an image for the empirical pipeline
(`run_pipeline.py`), protocols included; build and run examples are in the
header of each file. Confirmatory benchmark runs compare the git trees of the
checkout with a tag, so they run from a git clone, not inside the images.

## Citation

Please cite the software through its concept DOI, which resolves to the latest
archived version: https://doi.org/10.5281/zenodo.23339786 (metadata in
`CITATION.cff`; version 1.2.0, tag `v1.2.0`).

## License

MIT (`LICENSE`). The atlas files under `atlases/` and the structural
connectome under `data/managed/structural/` are third-party data under their
own terms (`licenses/THIRD_PARTY_NOTICES.md`).
