#!/usr/bin/env bash
# Download a pinned OpenNeuro snapshot as a DataLad/git-annex dataset.
#
# Usage: bash scripts/download_data.sh [options] DATASET_ID [SNAPSHOT] [OUT_DIR]
#
#   DATASET_ID  OpenNeuro accession, e.g. ds003171
#   SNAPSHOT    snapshot tag; defaults to the version pinned below. Datasets
#               without a pinned version require an explicit snapshot.
#   OUT_DIR     target folder (default: <repo>/data/scratch/<DATASET_ID>);
#               relative paths are resolved against the current directory
#               and the absolute path is printed.
#
# Options:
#   --no-get    clone and check out the snapshot only (file content is
#               fetched later with `datalad get` / `git annex get`)
#   --jobs N    parallel annex downloads (default 4)
#   --dry-run   print the commands instead of running them
#   --          end of options (the remaining arguments are positional)
#
# Snapshots are the git tags of https://github.com/OpenNeuroDatasets/<id>;
# file content comes from OpenNeuro's public S3 annex remote (no login).
# Requires git + git-annex, or DataLad (preferred when installed).
# Re-running is safe: an existing clone at the requested snapshot is reused
# and only missing content is fetched.
set -euo pipefail

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GET_CONTENT=1
JOBS=4
DRY_RUN=0
POSITIONAL=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-get) GET_CONTENT=0 ;;
    --jobs) JOBS="${2:?--jobs needs a number}"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    --) shift; POSITIONAL+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1") ;;
  esac
  shift
done
if [ "${#POSITIONAL[@]}" -lt 1 ] || [ "${#POSITIONAL[@]}" -gt 3 ]; then
  usage >&2
  exit 2
fi
case "${JOBS}" in
  ''|*[!0-9]*|0) echo "--jobs needs a positive integer, got '${JOBS}'" >&2; exit 2 ;;
esac

DATASET_ID="${POSITIONAL[0]}"
SNAPSHOT="${POSITIONAL[1]:-}"
OUT_DIR="${POSITIONAL[2]:-${REPO_ROOT}/data/scratch/${DATASET_ID}}"

case "${DATASET_ID}" in
  ds[0-9][0-9][0-9][0-9][0-9][0-9]) ;;
  *) echo "Invalid OpenNeuro accession: ${DATASET_ID}" >&2; exit 2 ;;
esac

# Snapshot versions used by the pipeline and the synthetic-data generator;
# they match the snapshots in src/impact_pipeline/dataset_catalog.py.
pinned_snapshot() {
  case "$1" in
    ds003171) echo 2.0.1 ;;
    ds005620) echo 1.0.0 ;;
    ds002547) echo 1.1.0 ;;
    ds005479) echo 1.1.1 ;;
    ds004295) echo 1.0.0 ;;
    ds002336) echo 2.0.2 ;;
    ds006623) echo 1.0.0 ;;
    ds002685) echo 1.3.1 ;;
    *) echo "" ;;
  esac
}
if [ -z "${SNAPSHOT}" ]; then
  SNAPSHOT="$(pinned_snapshot "${DATASET_ID}")"
  if [ -z "${SNAPSHOT}" ]; then
    echo "No pinned snapshot for ${DATASET_ID}; pass one explicitly." >&2
    exit 2
  fi
fi
case "${SNAPSHOT}" in
  -*|*/*|*..*|*[[:space:]]*)
    echo "Invalid snapshot tag: '${SNAPSHOT}'" >&2
    exit 2
    ;;
esac

# Absolute output path (the parent is created if needed).
OUT_PARENT="$(dirname "${OUT_DIR}")"
if [ "${DRY_RUN}" -eq 0 ]; then
  mkdir -p "${OUT_PARENT}"
  OUT_DIR="$(cd "${OUT_PARENT}" && pwd)/$(basename "${OUT_DIR}")"
else
  case "${OUT_DIR}" in
    /*) ;;
    *) OUT_DIR="$(pwd)/${OUT_DIR}" ;;
  esac
fi

URL="https://github.com/OpenNeuroDatasets/${DATASET_ID}.git"

run() {
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

USE_DATALAD=0
if command -v datalad >/dev/null 2>&1; then
  USE_DATALAD=1
elif [ "${DRY_RUN}" -eq 0 ]; then
  if ! command -v git >/dev/null 2>&1 || ! git annex version >/dev/null 2>&1; then
    echo "Need DataLad or git + git-annex (e.g. 'conda install -c conda-forge datalad'" >&2
    echo "on Linux, or 'brew install git-annex' and 'pip install datalad' on macOS)." >&2
    exit 1
  fi
fi

echo "Dataset:  ${DATASET_ID} (snapshot ${SNAPSHOT})"
echo "Source:   ${URL}"
echo "Target:   ${OUT_DIR}"

if [ -e "${OUT_DIR}/.git" ]; then
  # All tags on HEAD: one commit can carry several snapshot tags (e.g. an
  # unchanged re-release), and `git describe` would report only one of them.
  head_tags="$(git -C "${OUT_DIR}" tag --points-at HEAD 2>/dev/null || true)"
  if ! printf '%s\n' "${head_tags}" | grep -Fxq -- "${SNAPSHOT}"; then
    current="$(printf '%s' "${head_tags}" | tr '\n' ' ')"
    echo "Existing clone at ${OUT_DIR} is at '${current:-untagged revision}'," >&2
    echo "not snapshot ${SNAPSHOT}. Use another OUT_DIR or check out the tag yourself." >&2
    exit 1
  fi
  echo "Reusing existing clone at snapshot ${SNAPSHOT}."
elif [ -e "${OUT_DIR}" ] && [ -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]; then
  echo "${OUT_DIR} exists and is not a dataset clone; refusing to overwrite." >&2
  exit 1
else
  if [ "${USE_DATALAD}" -eq 1 ]; then
    run datalad clone "${URL}" "${OUT_DIR}"
  else
    run git clone --quiet "${URL}" "${OUT_DIR}"
  fi
  run git -C "${OUT_DIR}" -c advice.detachedHead=false checkout --quiet "refs/tags/${SNAPSHOT}"
  if [ "${USE_DATALAD}" -eq 0 ]; then
    run git -C "${OUT_DIR}" annex init --quiet
  fi
fi

if [ "${GET_CONTENT}" -eq 1 ]; then
  if [ "${USE_DATALAD}" -eq 1 ]; then
    run datalad get --jobs "${JOBS}" --dataset "${OUT_DIR}" "${OUT_DIR}"
  else
    run git -C "${OUT_DIR}" annex get --jobs="${JOBS}" .
  fi
fi

echo "Done: ${OUT_DIR}"
