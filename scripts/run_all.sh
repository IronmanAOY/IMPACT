#!/usr/bin/env bash
# End-to-end local run for one supported dataset:
#   download (pinned snapshot) -> [fMRI: atlases + fMRIPrep] -> run_pipeline.py
#
# Usage: bash scripts/run_all.sh [--dry-run] [DATASET_ID] [SNAPSHOT]
#   DATASET_ID  ds003171 (fMRI, default) or ds005620 (EEG)
#   SNAPSHOT    defaults to the snapshot pinned in scripts/download_data.sh
#
#   --dry-run   print every command instead of running it (may appear anywhere)
#
# Environment (relative paths are resolved against the current directory):
#   IMPACT_CONDA_ENV          conda env name (default impact-synergy-clean)
#   IMPACT_PYTHON             python to use instead of `conda run -n <env> python`
#   IMPACT_DATA_ROOT          folder holding the BIDS datasets (default <repo>/data/scratch)
#   IMPACT_OUT_DIR            output folder (default <repo>/outputs/scratch)
#   IMPACT_FMRIPREP_RECONALL  1 = run FreeSurfer recon-all in fMRIPrep (default 0)
#   FS_LICENSE                FreeSurfer license (required for fMRIPrep)
#
# The Melbourne replication data are not needed: replication (step 7) is not
# requested here.
set -euo pipefail

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

DRY_RUN=0
POSITIONAL=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) POSITIONAL+=("$1") ;;
  esac
  shift
done
if [ "${#POSITIONAL[@]}" -gt 2 ]; then
  usage >&2
  exit 2
fi

CALLER_DIR="$(pwd)"
abs_from_caller() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${CALLER_DIR}" "$1" ;;
  esac
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_ID="${POSITIONAL[0]:-ds003171}"
SNAPSHOT="${POSITIONAL[1]:-}"
CONDA_ENV="${IMPACT_CONDA_ENV:-impact-synergy-clean}"
OUT_DIR="$(abs_from_caller "${IMPACT_OUT_DIR:-${REPO_ROOT}/outputs/scratch}")"
DATA_ROOT="$(abs_from_caller "${IMPACT_DATA_ROOT:-${REPO_ROOT}/data/scratch}")"
BIDS_ROOT="${DATA_ROOT}/${DATASET_ID}"
FMRIPREP_DIR="${BIDS_ROOT}/derivatives/fmriprep"

run() {
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [ -n "${IMPACT_PYTHON:-}" ]; then
  case "${IMPACT_PYTHON}" in
    */*) PYTHON_CMD=("$(abs_from_caller "${IMPACT_PYTHON}")") ;;
    *) PYTHON_CMD=("${IMPACT_PYTHON}") ;;  # a command name looked up in PATH
  esac
elif [ "${CONDA_DEFAULT_ENV:-}" = "${CONDA_ENV}" ]; then
  PYTHON_CMD=(python)
else
  CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
  if [ -z "${CONDA_BIN}" ] && [ "${DRY_RUN}" -eq 0 ]; then
    echo "conda not found in PATH. Install/initialize conda or set IMPACT_PYTHON." >&2
    exit 1
  fi
  PYTHON_CMD=("${CONDA_BIN:-conda}" run --no-capture-output -n "${CONDA_ENV}" python)
fi

run_pipeline() {
  MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl}" \
  NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba_cache}" \
  run "${PYTHON_CMD[@]}" run_pipeline.py "$@"
}

case "${DATASET_ID}" in
  ds003171|ds005620) ;;
  *) echo "Unsupported DATASET_ID=${DATASET_ID} (use ds003171 or ds005620)" >&2; exit 2 ;;
esac

# `--` keeps a snapshot or path that starts with '-' from being read as an option.
if [ "${DRY_RUN}" -eq 1 ]; then
  bash scripts/download_data.sh --dry-run -- "${DATASET_ID}" "${SNAPSHOT}" "${BIDS_ROOT}"
else
  bash scripts/download_data.sh -- "${DATASET_ID}" "${SNAPSHOT}" "${BIDS_ROOT}"
fi

# Same rule as run_pipeline._fmriprep_subject_complete: a subject is complete
# only with its report (sub-<label>.html) AND at least one
# sub-<label>/**/func/*desc-preproc_bold.nii.gz. A run that stopped after
# writing the report is therefore re-run here as well.
fmriprep_subject_complete() {
  local label="$1"
  [ -f "${FMRIPREP_DIR}/${label}.html" ] || return 1
  [ -d "${FMRIPREP_DIR}/${label}" ] || return 1
  local hit
  # Search relative to the subject folder, so a 'func' component in the
  # derivatives path itself cannot match or exclude anything.
  hit="$(cd "${FMRIPREP_DIR}/${label}" 2>/dev/null \
    && find . -path '*/func/*desc-preproc_bold.nii.gz' ! -path '*/func/*/*' \
      -print -quit 2>/dev/null || true)"
  [ -n "${hit}" ]
}

if [ "${DATASET_ID}" = "ds003171" ]; then
  run bash scripts/download_atlases.sh

  # fMRIPrep only for subjects without complete derivatives (see above).
  missing=()
  for sub_dir in "${BIDS_ROOT}"/sub-*; do
    [ -d "${sub_dir}" ] || continue
    label="$(basename "${sub_dir}")"
    if ! fmriprep_subject_complete "${label}"; then
      missing+=("${label#sub-}")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ] || [ "${DRY_RUN}" -eq 1 ]; then
    FMRIPREP_ARGS=(--bids-root "${BIDS_ROOT}" --out-dir "${FMRIPREP_DIR}")
    if [ "${IMPACT_FMRIPREP_RECONALL:-0}" != "1" ]; then
      FMRIPREP_ARGS+=(--skip-reconall)
    fi
    if [ "${#missing[@]}" -gt 0 ]; then
      FMRIPREP_ARGS+=(--participant-label "${missing[@]}")
    fi
    run bash scripts/fetch_fmriprep.sh "${FMRIPREP_ARGS[@]}"
  else
    echo "fMRIPrep derivatives complete in ${FMRIPREP_DIR}; skipping fMRIPrep."
  fi

  run_pipeline \
    --execution-mode local \
    --dataset-id ds003171 \
    --bids-root "${BIDS_ROOT}" \
    --fmriprep-dir "${FMRIPREP_DIR}" \
    --out-dir "${OUT_DIR}" \
    --run-preprocessing \
    --mpc-metrics PDI NAS IIM \
    --no-ci
else
  run_pipeline \
    --execution-mode local \
    --dataset-id ds005620 \
    --bids-root "${BIDS_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --run-preprocessing \
    --mpc-metrics PDI NAS IIM \
    --no-ci
fi
