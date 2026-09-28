[![DOI](https://zenodo.org/badge/975114052.svg)](https://doi.org/10.5281/zenodo.15306740)

# IMPaCT Synergy Pipeline

A measurement pipeline for the Minimal Principles for Consciousness (MPC) of the
IMPaCT framework, for EEG and fMRI data and for simulated systems. It computes
five null-anchored estimators and combines them with a three-valued,
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
- **MPC degree**: a capped power mean of the reference-normalised components,
  computed only for `MPC_CONSISTENT` rows. It is a reference-relative evidence
  summary, not a level of consciousness.

The legacy geometric-mean Consciousness Index (`CI` column, `compute_CI`) is
kept for backward compatibility. It is deprecated and is not a gate. The
exploratory HypergraphSynergy statistic `S` is reported separately and never
enters CI or the verdict.

Definitions: [`docs/metrics.md`](docs/metrics.md). Code structure:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Changes in 1.1.0:
[`CHANGELOG.md`](CHANGELOG.md).

## Installation

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

### 2. Synthetic smoke tests (no downloads)

MPC-Bench simulates white-box systems with switchable mechanisms and scores them
with the public estimators. Two witness systems, three estimators, 3 null
surrogates (about 15 s on a laptop):

```bash
python scripts/run_bench.py witnesses --seeds 0 \
    --witnesses PC_nominal,N_independent_noise \
    --metrics PDI,NAS,IIM --null-surrogates 3 --no-markers \
    --out outputs/bench_smoke
```

It writes `results.csv`, `results.jsonl` and `run_manifest.json` (with the code
version) to `outputs/bench_smoke/`, one row per system with the estimates, null
moments, z-scores, the verdict and its reasons. With only three of the five
principles computed, RAM and SRPI are `MISSING` and no verdict can be
`MPC_CONSISTENT`.

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
files). `--null-surrogates K` calibrates every component against K surrogates,
which the MPC verdict needs; with the default 0 every verdict is `UNDETERMINED`
(`NO_NULL_CALIBRATION`). Runtime grows roughly (K+1)-fold. Use
`--iim-max-nodes` to bound the IIM subsystem on a workstation.

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
and no verdict can be `MPC_CONSISTENT`. Three surrogates only exercise the code;
use at least 19 for inference.

Useful options (`python run_pipeline.py --help` lists all):

| Option | Effect |
|---|---|
| `--subjects A B` | restrict to some subjects |
| `--mpc-metrics RAM PDI NAS IIM SRPI` | subset of estimators (default all) |
| `--null-surrogates K` | null calibration per run and component (0 = none) |
| `--necessity-set RAM,PDI,NAS,IIM,SRPI` | principles the verdict requires (default all five) |
| `--applicability-registry registry.json` | evidence from unvalidated estimators becomes UNDEFINED |
| `--ci-reference cohort_high_state\|file.json` | reference means of the legacy CI and the MPC degree |
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

Main columns of `step2_df.csv` (full list: `docs/metrics.md`, section 11):

| Column | Content |
|---|---|
| `subject`, `session`, `theta` | run identity; theta of S |
| `RAM`, `PDI`, `NAS`, `IIM`, `SRPI` | metric values; with `--null-surrogates K > 0` the excess over the null mean, floored at 0; NaN when undefined |
| `MPC_verdict` | `EXCLUDED`, `MPC_CONSISTENT` or `UNDETERMINED` |
| `MPC_reason` | `;`-joined reason codes, e.g. `MISSING:RAM`, `NO_NULL_CALIBRATION:IIM`, `ABSENT:NAS` |
| `MPC_degree` | MPC degree (`MPC_CONSISTENT` rows only) |
| `<P>_status` | PRESENT, ABSENT or UNDEFINED per principle |
| `<P>_estimate`, `<P>_null_mean`, `<P>_null_sd`, `<P>_null_n`, `<P>_margin` | raw estimate, null moments and margin (IIM on the ΔΨ scale, bits) |
| `MPC_necessity_set`, `MPC_null_surrogates`, `MPC_null_seed`, `MPC_null_families` | evidence configuration |
| `CI`, `CI_defined`, `CI_missing`, `CI_reference` | legacy CI (deprecated, not a gate); NaN with the missing components listed when undefined |
| `PDI_anchor`, `PDI_task`, `*_reason`; `IIM_raw`, `IIM_defined`, `IIM_undefined_reason` | endpoints and reasons |
| `S` | exploratory HypergraphSynergy at `theta` |

## Running on HLRS Hunter

`--execution-mode hunter` distributes the IIM computation over PBS Pro array jobs
on HLRS Hunter (AMD MI300A APUs, 4 shards per node, Ψ kernels on the APU) and
then runs the other estimators, the evidence layer, the statistics and the report
in one job. The complete procedure (account, workspace, cray-python venv, CuPy
for ROCm, hardware self-test, data staging, smoke test on the `test` queue,
submission, monitoring, troubleshooting, hand-back, open questions for HLRS) is
in [`docs/HLRS_HUNTER_RUNBOOK.md`](docs/HLRS_HUNTER_RUNBOOK.md). The helper
scripts and the configuration variables are described in
[`scripts/hunter/README.md`](scripts/hunter/README.md).

```bash
python3 run_pipeline.py --execution-mode hunter --hunter-stage build-campaign \
    --hardware-target hunter-apu --dataset-id ds003171 \
    --bids-root <BIDS> --out-dir <OUT> --iim-max-nodes <N> \
    --hunter-iim-null-surrogates <K> --null-surrogates <K>
bash <OUT>/cache/hunter_iim_campaign/pbs/00_submit_all.sh    # on a Hunter login node
```

Check a GPU/APU node first with
`PYTHONPATH=src python3 -m impact_pipeline.hardware_selftest --target hunter-apu`
from the checkout (the package is not pip-installed on Hunter; runbook,
section 7).

## MPC-Bench

`impact_pipeline.bench` provides white-box systems for validating the
estimators and the attribution rule: a modular rate-network agent with five
mechanism switches (family A), binary/kinetic-Ising networks with exact TPMs
(family B), a held-out Stuart-Landau network (family C), a patchwork of five
disconnected single-principle modules, null and hypersynchronous systems, a
witness catalogue (`src/impact_pipeline/bench/witnesses.yaml`) and rival
attribution rules (`bench/rules.py`). Generators never import the estimators;
hidden ground truth is written to a separate oracle file.

```bash
python scripts/run_bench.py --help
python scripts/run_bench.py witnesses --seeds 0-9 --out outputs/bench_witnesses --workers 8
python scripts/run_bench.py factorial --seeds 0-19 --out outputs/bench_factorial --workers 8
python scripts/run_bench.py sweep --knobs eta,g_b --levels 10 --seeds 0-9 --out outputs/bench_sweep
python scripts/run_bench.py timing --seeds 0
```

Designs: `factorial` (2^5 mechanism cells x seeds), `sweep` (dose-response),
`witnesses` (catalogue x seeds), `timing`. Results resume by default
(`--no-resume` to recompute). Seed policy: development seeds are 0-999;
confirmatory seeds start at 10000 (1000-9999 are refused), and they and family
C run only with `--confirmatory --freeze-tag <tag>` on a clean checkout that
descends from the code-freeze tag and has the same `src/` and `scripts/` as the
tag. `--n-shards N --pbs-template bench.pbs` writes a PBS Pro array
script for Hunter (set `BENCH_VENV` and `BENCH_OUT_DIR`; the bench is CPU-bound).

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
                           preprocessing, Hunter backend, MPC-Bench (bench/)
scripts/                   downloads, fMRIPrep, run_all, dashboard, synthetic objects, run_bench,
                           hunter/ (setup, install, smoke test)
tests/                     regression, ground-truth and property tests
docs/                      metrics, architecture, Hunter runbook, synthetic data
data/managed/              small versioned reference files (dataset inventory, structural connectome)
atlases/                   atlas files (scripts/download_atlases.sh)
```

## Citation

Please cite the software through its concept DOI, which resolves to the latest
archived version: https://doi.org/10.5281/zenodo.15306740 (metadata in
`CITATION.cff`; this release is version 1.1.0).

The real-data-derived synthetic smoke-test archive has the reserved DOI
10.5281/zenodo.20786673. It is **not yet published**; until it is, regenerate the
objects from the OpenNeuro sources (`docs/synthetic_data.md`).

## License

MIT (`LICENSE`). Atlas files are covered by their own licenses
(`licenses/THIRD_PARTY_NOTICES.md`).
