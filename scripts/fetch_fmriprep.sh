#!/usr/bin/env bash
# Run fMRIPrep (Docker, pinned version) on a BIDS dataset and write the
# derivatives where the pipeline reads them:
#
#   <bids-root>/derivatives/fmriprep   (run_pipeline.py --fmriprep-dir)
#
# Usage:
#   bash scripts/fetch_fmriprep.sh --bids-root /abs/path/ds003171 [options] [-- extra fMRIPrep args]
#   bash scripts/fetch_fmriprep.sh --dataset-id ds003171 [options]
#
# Options:
#   --bids-root PATH          BIDS dataset root
#   --dataset-id ID           default BIDS root <repo>/data/scratch/<ID>
#   --out-dir PATH            derivatives folder (default <bids-root>/derivatives/fmriprep)
#   --work-dir PATH           scratch folder (default <bids-root>/../fmriprep_work/<name>)
#   --participant-label L ... subjects without the "sub-" prefix (default: all)
#   --fs-license PATH         FreeSurfer license (else $FS_LICENSE, else
#                             licenses/fs_license.txt). Always required:
#                             fMRIPrep refuses to run without one, also with
#                             --skip-reconall.
#   --skip-reconall           pass --fs-no-reconall to fMRIPrep
#   --nthreads N              (default 8)
#   --omp-nthreads N          (default 4)
#   --mem-mb N                (default 16000)
#   --dry-run                 print the docker command instead of running it
#
# The image is nipreps/fmriprep:${FMRIPREP_VERSION:-25.1.3}, the version
# run_pipeline.py --run-fmriprep also uses. It is an amd64 image (emulated on
# Apple silicon).
set -euo pipefail

usage() { sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FMRIPREP_VERSION="${FMRIPREP_VERSION:-25.1.3}"
IMAGE="nipreps/fmriprep:${FMRIPREP_VERSION}"

BIDS_ROOT=""
DATASET_ID=""
OUT_DIR=""
WORK_DIR=""
FS_LICENSE_ARG=""
SKIP_RECONALL=0
NTHREADS=8
OMP_NTHREADS=4
MEM_MB=16000
DRY_RUN=0
PARTICIPANTS=()
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bids-root) BIDS_ROOT="${2:?--bids-root needs a path}"; shift ;;
    --dataset-id) DATASET_ID="${2:?--dataset-id needs an ID}"; shift ;;
    --out-dir) OUT_DIR="${2:?--out-dir needs a path}"; shift ;;
    --work-dir) WORK_DIR="${2:?--work-dir needs a path}"; shift ;;
    --fs-license) FS_LICENSE_ARG="${2:?--fs-license needs a path}"; shift ;;
    --skip-reconall) SKIP_RECONALL=1 ;;
    --nthreads) NTHREADS="${2:?}"; shift ;;
    --omp-nthreads) OMP_NTHREADS="${2:?}"; shift ;;
    --mem-mb) MEM_MB="${2:?}"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    --participant-label)
      shift
      while [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; do
        PARTICIPANTS+=("${1#sub-}")
        shift
      done
      if [ "${#PARTICIPANTS[@]}" -eq 0 ]; then
        echo "--participant-label needs at least one label" >&2
        exit 2
      fi
      continue
      ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

abspath() {
  local p="$1"
  if [ -d "${p}" ]; then
    (cd "${p}" && pwd)
  else
    case "${p}" in
      /*) printf '%s\n' "${p}" ;;
      *) printf '%s/%s\n' "$(pwd)" "${p}" ;;
    esac
  fi
}

if [ -z "${BIDS_ROOT}" ]; then
  if [ -z "${DATASET_ID}" ]; then
    echo "Pass --bids-root or --dataset-id." >&2
    exit 2
  fi
  BIDS_ROOT="${REPO_ROOT}/data/scratch/${DATASET_ID}"
fi
if [ ! -f "${BIDS_ROOT}/dataset_description.json" ]; then
  echo "Not a BIDS dataset root (no dataset_description.json): ${BIDS_ROOT}" >&2
  exit 1
fi
BIDS_ROOT="$(abspath "${BIDS_ROOT}")"
OUT_DIR="$(abspath "${OUT_DIR:-${BIDS_ROOT}/derivatives/fmriprep}")"
WORK_DIR="$(abspath "${WORK_DIR:-$(dirname "${BIDS_ROOT}")/fmriprep_work/$(basename "${BIDS_ROOT}")}")"

# FreeSurfer license: --fs-license, then $FS_LICENSE, then licenses/fs_license.txt.
LICENSE="${FS_LICENSE_ARG:-${FS_LICENSE:-}}"
if [ -z "${LICENSE}" ] && [ -f "${REPO_ROOT}/licenses/fs_license.txt" ]; then
  LICENSE="${REPO_ROOT}/licenses/fs_license.txt"
fi
if [ -z "${LICENSE}" ]; then
  echo "A FreeSurfer license is required (fMRIPrep also needs it with --skip-reconall)." >&2
  echo "Use --fs-license PATH, set FS_LICENSE, or copy it to licenses/fs_license.txt." >&2
  exit 1
fi
LICENSE="${LICENSE/#\~/${HOME}}"
if [ ! -f "${LICENSE}" ]; then
  echo "FreeSurfer license file not found: ${LICENSE}" >&2
  exit 1
fi
LICENSE="$(abspath "${LICENSE}")"

FMRIPREP_ARGS=(
  /data /out participant
  -w /work
  --fs-license-file /opt/freesurfer/license.txt
  --output-spaces MNI152NLin2009cAsym
  --nthreads "${NTHREADS}"
  --omp-nthreads "${OMP_NTHREADS}"
  --mem-mb "${MEM_MB}"
)
if [ "${#PARTICIPANTS[@]}" -gt 0 ]; then
  FMRIPREP_ARGS+=(--participant-label "${PARTICIPANTS[@]}")
fi
if [ "${SKIP_RECONALL}" -eq 1 ]; then
  FMRIPREP_ARGS+=(--fs-no-reconall)
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  FMRIPREP_ARGS+=("${EXTRA_ARGS[@]}")
fi

DOCKER_CMD=(
  docker run --rm
  --user "$(id -u):$(id -g)"
  -v "${BIDS_ROOT}:/data:ro"
  -v "${OUT_DIR}:/out"
  -v "${WORK_DIR}:/work"
  -v "${LICENSE}:/opt/freesurfer/license.txt:ro"
  "${IMAGE}"
  "${FMRIPREP_ARGS[@]}"
)

echo "fMRIPrep ${FMRIPREP_VERSION}: ${BIDS_ROOT} -> ${OUT_DIR}"
if [ "${DRY_RUN}" -eq 1 ]; then
  printf '%q ' "${DOCKER_CMD[@]}"
  printf '\n'
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found; fMRIPrep runs as a Docker container." >&2
  exit 1
fi
mkdir -p "${OUT_DIR}" "${WORK_DIR}"
"${DOCKER_CMD[@]}"
