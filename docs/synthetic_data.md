# Real-Data-Derived Synthetic Smoke-Test Objects

## What These Objects Are

These are **software smoke-test objects** for the RAM, PDI, NAS, IIM, SRPI and
CI code paths. Each run starts from a real OpenNeuro recording (fMRI voxel time
series or scalp-EEG channels). `scripts/generate_real_derived_synth_completed.py`
then plants known structure on top of it and records every planted quantity,
with its ground-truth direction, in the manifests.

What they are **not**:

- They are not validation of the IMPaCT theory. Metric values computed on them
  are not evidence about consciousness, and not evidence that the metrics are
  valid on real data. The state contrast is defined by the planted structure,
  not by physiology.
- They are not anonymous noise. The arrays are deterministic transforms of
  individual recordings from CC0 OpenNeuro datasets, and they keep the source
  subject IDs.
- They do not exercise the whole pipeline. Validation calls the public metric
  functions `compute_synergy_ci`, `compute_RAM` and `compute_SRPI` on the
  generator-written preprocessed arrays. `run_pipeline.py` orchestration,
  fMRIPrep/EEG preprocessing, `run_s_ci` and the statistics steps are not run.

## Targets, Sessions And Planted Levels

Every planted quantity scales with a state level `g` in [0, 1]. For every metric
the ground-truth direction is: higher `g` gives a higher metric value.

| Target | Modality | Sessions (planted `g`) | Planted contrast | Null contrast (equal `g`) |
|---|---|---|---|---|
| `ds003171` | fMRI | awake 1.0, recovery 1.0, light 0.5, deep 0.2 | awake > deep | awake vs recovery |
| `ds002547` | fMRI | awake 1.0, ses-1 0.6, ses-2 0.6, deep 0.2 | awake > deep | ses-1 vs ses-2 |
| `ds005620` | EEG | awake 1.0, sed 0.5, deep 0.2 | awake > deep | none |

Notes:

- `ds005620` `deep` is the pipeline's name for the deepest sedation level: it
  is built from `task-sed2` recordings and written as `task-sed2` in BIDS. No
  separate `sed2` pseudo-session is generated, because it would duplicate
  `deep`.
- `ds002547` has no state design. Its `awake`/`deep` labels are pseudo-states
  defined only by the planted level. Its rest baselines are ds003171 rest runs
  from other subjects, because ds002547 has no rest runs.

## What Is Planted (Generator 2.0.0)

For node payload `x` (the real recording, z-scored per node):

- **Differentiation (PDI target).** `y = sqrt(d) x + sqrt(1-d) c(t)`, where `c`
  is the leading principal component of `x` and `d = 0.25 + 0.75 g`.
- **Integration (IIM target).** Nodes are assigned to 4 latent modules. The
  module latents follow an Ornstein-Uhlenbeck system with antisymmetric
  circulant coupling `kappa = 0.6 g` (time constant 8 s for fMRI, 1 s for EEG).
  This coupling gives directed, time-lagged dependence while the equal-time
  covariance stays isotropic, so integration does not reduce differentiation.
- **Broadcast (NAS target).** 20% of nodes form a workspace. Every other node
  receives a fixed sparse random mix of workspace nodes one sample later, with
  gain `0.8 g`.
- **Events (RAM target).** Goal, stimulus and feedback responses (scaled by
  reward value) plus feedback-driven update responses. They are convolved with
  a canonical kernel (a double-gamma HRF with a ~5 s peak for fMRI, a 0.15 s
  alpha function for EEG). Amplitude is proportional to `g`, in node-SD units,
  in 10% of nodes.
- **Self/non-self (SRPI target).** A pre-event internal state `s ~ N(0,1)` is
  planted in `[onset - pre_window, onset)`. Self responses use a fixed pattern
  with amplitude `g * max(0, 1 + 0.5 s)`. Non-self responses use a fresh random
  pattern with amplitude `0.5 g`.
- Rest arrays get the same differentiation, integration and broadcast structure
  for the same `g`, without events. The final arrays are z-scored per node,
  which is the preprocessing contract.

All constants are in `PLANTED_DESIGN` in the generator and in each manifest
under `planted_design`. They were fixed before any metric output was looked at.
Per run, the node roles, patterns and internal states are saved in
`test_objects/runs/real_derived_synth_completed/<ds>/planted_truth/`.

Source payloads are allocated per subject so that no two runs share a source
segment. fMRI runs are used whole. EEG recordings are cut into consecutive
60 s segments. When a subject has fewer source runs than sessions, the reuse is
flagged per run (`task_source.reused`, `shared_with`). When a session has to use
a recording of a different real state, that is flagged too (`state_fallback`).
`source_reuse_summary` counts both. EEG nodes are the scalp EEG channels only.
EOG, EMG, ECG and misc channels are excluded by type and by name, and channels
that are flat in any segment of a subject are dropped. The file label `eeg64` is
kept for pipeline compatibility.

## Validation Report

`actual_metric_report.json` (one per dataset) and the summary JSON keep three
things separate.

1. **Smoke test (`smoke_test_passed`).** This is the only pass gate. It checks:
   the arrays exist, are finite and have no zero-variance nodes; array shapes
   match the manifest; the BIDS data (NIfTI node container or BrainVision)
   equal the analysed arrays; sidecar sample intervals and task labels match;
   readiness is 1.0 for CI; and RAM, PDI, NAS, IIM, SRPI and CI are present,
   defined and within their documented bounds.
2. **Generator self-check (`planted_structure_verified`).** It uses simple
   statistics computed without the metric code: participation ratio, directed
   module coupling, directed workspace broadcast, evoked projection and
   self-specific projection. It confirms that the planted structure is larger
   in the high-level session than in the low-level session.
3. **Known-answer checks (`known_answer`).** For each metric, and for optional
   columns such as `IIM_z` when the metric code reports them, the report gives
   paired subject-level differences for the planted and null contrasts and an
   exact two-sided sign test. The outcomes are `recovered`,
   `direction_only_not_significant`, `not_recovered`, `reversed` or
   `undefined`. For the null contrast they are `no_systematic_difference` or
   `systematic_difference_without_planted_difference`. A per-subject Spearman
   correlation between the planted level and the metric is also reported.
   These outcomes are observations. They are not a pass criterion, and nothing
   was tuned to make them pass.

The report also has `pipeline_event_resolution`. Metrics are computed from the
exact events file of each run. This section records whether the pipeline's own
events resolver (`run_synergy_ci._resolve_events_file`) would pick the same
file. With the resolver as of this writing, ds005620 `sed` resolves to the
`task-sed2` events and ds002547 `ses-1`/`ses-2` resolve to the `awake` events.

## Metric Configuration Used

The full configuration is recorded in every manifest under
`metric_configuration` and in every report. The IIM configuration is a reduced
smoke-test configuration. The same values are used for readiness and for
computation:

| Parameter | Value |
|---|---|
| `iim_bins` | 2 |
| `iim_lag_trs` | 1 |
| `iim_max_nodes` | 6 (selected by `compute_IIM`'s own rule) |
| `iim_max_mechanism_size` / `iim_max_purview_size` | 2 / 2 |
| `iim_max_timepoints` | 80 (arrays are decimated with `ts[:, ::ceil(T/80)]`, no anti-aliasing) |
| `iim_max_state_space` | 1500 |

This is not the `run_pipeline.py` default. As of this writing, that default uses
all mechanism and purview sizes over up to 10 nodes at 2 bins. Use `--iim-bins`, `--iim-max-nodes`,
`--iim-max-timepoints`, `--iim-max-mechanism-size` and `--iim-max-purview-size`
to validate with another configuration; the configuration used is recorded. The
RAM, PDI, NAS and SRPI parameters for fMRI and EEG are the `*_PARAMS`
dictionaries in the generator.

## Using The Published Archive

The open archive is at:

```text
https://doi.org/10.5281/zenodo.20786673
```

It contains `impact-synergy-real-derived-synthetic-test-objects.tar.gz`,
`release_manifest.json` and `SHA256SUMS.txt`.

> The archive was built on 2026-06-21 by generator 1.x. Its shipped reports use
> the old criterion ("every metric finite and > 0") on objects engineered to
> satisfy it, so they are not validation. It also has known defects: ds005620
> `deep` duplicates `sed2`, its EEG BIDS files are truncated to 10 s,
> VEOG/HEOG are used as nodes, some rest arrays have zero-variance channels,
> and ds002547 rest baselines are tiled. Re-validating it with the current code
> runs the smoke checks. Known-answer checks are reported as unavailable,
> because the legacy manifests have no planted ground truth. Regenerate the
> objects (see below) to get them.

The tarball has a single top-level folder,
`impact-synergy-real-derived-synthetic-test-objects/`. `SHA256SUMS.txt` lists
the extracted files relative to that folder and has no entry for the tarball
itself. Download the three files into one folder, then run:

```bash
export IMPACT_SYNTH_DOWNLOAD=/absolute/path/to/downloaded/files
export IMPACT_SYNTH_EXTRACT=/absolute/path/for/synthetic_package

# Optional: compare with the MD5 that Zenodo lists for the tarball
# (use md5sum on Linux).
md5 "$IMPACT_SYNTH_DOWNLOAD/impact-synergy-real-derived-synthetic-test-objects.tar.gz"

mkdir -p "$IMPACT_SYNTH_EXTRACT"
tar -xzf "$IMPACT_SYNTH_DOWNLOAD/impact-synergy-real-derived-synthetic-test-objects.tar.gz" \
  -C "$IMPACT_SYNTH_EXTRACT"
export IMPACT_SYNTH_ROOT="$IMPACT_SYNTH_EXTRACT/impact-synergy-real-derived-synthetic-test-objects"

# Verify the extracted files (use sha256sum -c on Linux).
(cd "$IMPACT_SYNTH_ROOT" && shasum -a 256 -c "$IMPACT_SYNTH_DOWNLOAD/SHA256SUMS.txt")

# Re-validate, from the root of this repository.
conda run -n impact-synergy-clean python scripts/generate_real_derived_synth_completed.py \
  --validate-only
```

For reference, the locally built 2026-06-21 tarball is 903,159,868 bytes, with
MD5 `c90fa8173eb38ba38a3f6aa2ebea4c90` and SHA-256
`135aef8c8c8398d5583cb39da17aec3521d162823fc9f263ccf501fb716f4607`. These
values have not been checked against the copy hosted on Zenodo.

`--validate-only` works on an archive extracted anywhere. It finds the objects
through `IMPACT_SYNTH_ROOT` (or `--synth-root`). Relative manifest paths are
joined to that root. The absolute paths in legacy manifests are re-rooted at
their `test_objects/` component. The shipped reports are never modified.
Results go to
`$IMPACT_SYNTH_ROOT/test_objects/real_derived_synth_completed/revalidation/<UTC time>/`
(override with `--validation-out`). The exit code is 0 when
`smoke_test_passed` is true and 1 otherwise. Use `--datasets` and `--subjects`
for a quick partial run.

## Regenerating From OpenNeuro Sources

The generator reads four source datasets. ds004295 and ds002336 are no longer
needed.

- `ds003171`: fMRI task and rest runs
- `ds005620`: BrainVision EEG awake, sed and sed2 recordings
- `ds002547`: fMRIPrep derivatives (`derivatives/fmriprep`) and self/other
  events
- `ds005479`: MID events (the goal/stimulus/feedback timing template)

Download them into one source root:

```bash
export IMPACT_SOURCE_ROOT=/absolute/path/to/openneuro_sources

bash scripts/download_data.sh ds003171 2.0.1 "$IMPACT_SOURCE_ROOT/ds003171"
bash scripts/download_data.sh ds005620 1.0.0 "$IMPACT_SOURCE_ROOT/ds005620"
bash scripts/download_data.sh ds002547 1.1.0 "$IMPACT_SOURCE_ROOT/ds002547"
bash scripts/download_data.sh ds005479 1.1.1 "$IMPACT_SOURCE_ROOT/ds005479"
```

Then inspect the sources, generate and validate:

```bash
export IMPACT_SYNTH_ROOT=/absolute/path/for/generated_outputs
mkdir -p "$IMPACT_SYNTH_ROOT"

conda run -n impact-synergy-clean python scripts/inspect_real_sources_for_synth.py \
  --source-root "$IMPACT_SOURCE_ROOT" \
  --output-dir "$IMPACT_SYNTH_ROOT/test_objects/real_derived_synth_completed/reports"

conda run -n impact-synergy-clean python scripts/generate_real_derived_synth_completed.py \
  --source-root "$IMPACT_SOURCE_ROOT"
```

The generation step validates the new objects unless you pass
`--skip-validation`, and writes the reports next to the manifests. Its exit
code is 1 when the smoke test fails. It runs the
source inspection itself for any source that has no
`<ds>_source_inspection.json`. An explicit `--source-root` is authoritative: the
repository's `data/` folders are not searched as a fallback. When
`IMPACT_SYNTH_ROOT` is not the repository, the generator symlinks
`test_objects/datasets|runs/real_derived_synth_completed` in the repository to
it. Pass `--no-link-repo-paths` to skip this. If `IMPACT_SYNTH_ROOT` is omitted,
outputs go under this repository's `test_objects/` tree. Useful options:
`--datasets`, `--subjects`, `--max-subjects`, `--fmri-nodes`, `--eeg-seconds`
and `--eeg-max-channels`.

## Outputs

```text
$IMPACT_SYNTH_ROOT/test_objects/
  datasets/real_derived_synth_completed/<ds>/           BIDS (DatasetType raw) + manifest.json
  runs/real_derived_synth_completed/<ds>/preprocessed/  <subj>/<session>/<condition|rest>/*_ts.npy
  runs/real_derived_synth_completed/<ds>/planted_truth/ <subj>_<session>.npz
  real_derived_synth_completed/reports/                 manifests, source inspections, validation
```

Manifests store paths relative to `IMPACT_SYNTH_ROOT`, and source paths as
`<dataset>/<path inside the dataset>`. They contain no absolute local paths.

## Known Limitations

- fMRI nodes are randomly sampled real voxels, so spatial structure is
  scrambled. They carry the `schaefer400` label but are not parcels. The BIDS
  NIfTI is a node container of shape (nodes, 1, 1, time), not an anatomical
  image.
- The event timing is one template, from one ds005479 MID file and one ds002547
  self/other layout. It is identical across subjects apart from jitter.
- ds002547 donor rest runs are shared across subjects, because there are fewer
  ds003171 donors than ds002547 subjects. This is flagged per run
  (`donor_payload_shared_with_other_subjects`).
- CI is normalised within the validated cohort by the metric code, so its value
  depends on which subjects are validated together.
