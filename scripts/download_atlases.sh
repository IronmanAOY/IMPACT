#!/usr/bin/env bash
# Fetch the exact atlas files that fMRI preprocessing (src/impact_pipeline/
# preprocessing.py) reads, from pinned upstream sources, and verify them by
# SHA-256. Idempotent: files that already match are left untouched and no
# network access is needed.
#
#   atlases/schaefer_2018/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz
#   atlases/schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt
#   atlases/aal_SPM12/aal/atlas/AAL.nii   (AAL for SPM12, 116 labels)
#   atlases/shen_1mm_268_parcellation.nii.gz
#
# Sources, licenses and checksums: licenses/THIRD_PARTY_NOTICES.md.
# The download uses curl only (no nilearn), so it does not depend on the
# nilearn fetcher API, which changed between releases.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/download_atlases.sh [--verify] [--force] [--dest DIR]

  --verify    only check the local files against the pinned SHA-256 values
  --force     replace local files whose checksum does not match
  --dest DIR  atlas folder (default: <repo>/atlases)
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/atlases"
VERIFY_ONLY=0
FORCE=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --verify) VERIFY_ONLY=1 ;;
    --force) FORCE=1 ;;
    --dest) DEST="${2:?--dest needs a directory}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
mkdir -p "${DEST}"
DEST="$(cd "${DEST}" && pwd)"

SCHAEFER_URL="https://raw.githubusercontent.com/ThomasYeoLab/CBIG/v0.14.3-Update_Yeo2011_Schaefer2018_labelname/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI"
AAL_URL="https://www.gin.cnrs.fr/AAL_files/aal_for_SPM12.tar.gz"
SHEN_URL="https://raw.githubusercontent.com/canlab/Neuroimaging_Pattern_Masks/788c2108df4145e4abb6218ebda08e406437006e/Atlases_and_parcellations/2013_Shen_Constable_NIMG_268_parcellation/shen_1mm_268_parcellation.nii.gz"

# relative path | sha256
ATLAS_FILES=(
  "schaefer_2018/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz|abb8032840af30603995fd59634fd6bcd816088870de9061bbf90ad42c408d4d"
  "schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt|f62cdc9de9696e11bda9fe1cd51fc013b1da27ba180d2dcead8f3fd94bf3cbc1"
  "aal_SPM12/aal/atlas/AAL.nii|91e8ec62a293c2a28c8e7343f37abb978ab941c5e26fbc496945c66d073a2cd8"
  "aal_SPM12/aal/atlas/AAL.xml|1a0f2bb952b700fa94f5156b544cf238f2e8d0e0e5ebedaefac835938b7609f3"
  "shen_1mm_268_parcellation.nii.gz|675f9174c0c418e30f95e7e9fd84be4fdf3b705656fbb22164aeca8546f0c21d"
)

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# Prints ok / missing / mismatch for one relative path.
file_state() {
  local rel="$1" want="$2" path="${DEST}/$1"
  if [ ! -f "${path}" ]; then
    echo missing
  elif [ "$(sha256_of "${path}")" = "${want}" ]; then
    echo ok
  else
    echo mismatch
  fi
}

download() {
  local url="$1" out="$2"
  curl --fail --location --silent --show-error --retry 3 --output "${out}" "${url}"
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

need_fetch=()
bad=0
for entry in "${ATLAS_FILES[@]}"; do
  rel="${entry%%|*}"
  want="${entry##*|}"
  state="$(file_state "${rel}" "${want}")"
  echo "${state}: atlases/${rel}"
  case "${state}" in
    ok) ;;
    missing) need_fetch+=("${rel}") ;;
    mismatch)
      bad=1
      if [ "${FORCE}" -eq 1 ]; then need_fetch+=("${rel}"); fi
      ;;
  esac
done

if [ "${VERIFY_ONLY}" -eq 1 ]; then
  if [ "${bad}" -eq 0 ] && [ "${#need_fetch[@]}" -eq 0 ]; then
    echo "All atlas files present and verified in ${DEST}."
    exit 0
  fi
  echo "Atlas verification failed (see missing/mismatch above)." >&2
  exit 1
fi

if [ "${bad}" -eq 1 ] && [ "${FORCE}" -eq 0 ]; then
  echo "Local atlas files differ from the pinned versions; not overwriting." >&2
  echo "Re-run with --force to replace them." >&2
  exit 1
fi

if [ "${#need_fetch[@]}" -eq 0 ]; then
  echo "All atlas files present and verified in ${DEST}; nothing to download."
  exit 0
fi

fetched_aal=0
for rel in "${need_fetch[@]}"; do
  case "${rel}" in
    schaefer_2018/*)
      mkdir -p "${DEST}/schaefer_2018"
      download "${SCHAEFER_URL}/$(basename "${rel}")" "${TMP_DIR}/$(basename "${rel}")"
      mv -f "${TMP_DIR}/$(basename "${rel}")" "${DEST}/${rel}"
      ;;
    aal_SPM12/*)
      if [ "${fetched_aal}" -eq 0 ]; then
        download "${AAL_URL}" "${TMP_DIR}/aal_for_SPM12.tar.gz"
        mkdir -p "${TMP_DIR}/aal" "${DEST}/aal_SPM12"
        tar -xzf "${TMP_DIR}/aal_for_SPM12.tar.gz" -C "${TMP_DIR}/aal"
        rm -rf "${DEST}/aal_SPM12/aal"
        mv "${TMP_DIR}/aal/aal" "${DEST}/aal_SPM12/aal"
        fetched_aal=1
      fi
      ;;
    shen_1mm_268_parcellation.nii.gz)
      download "${SHEN_URL}" "${TMP_DIR}/shen.nii.gz"
      mv -f "${TMP_DIR}/shen.nii.gz" "${DEST}/${rel}"
      ;;
  esac
done

# Verify everything after download; a changed upstream file is an error.
failed=0
for entry in "${ATLAS_FILES[@]}"; do
  rel="${entry%%|*}"
  want="${entry##*|}"
  if [ "$(file_state "${rel}" "${want}")" != ok ]; then
    echo "Checksum mismatch after download: atlases/${rel}" >&2
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  exit 1
fi
echo "Atlas files downloaded and verified in ${DEST}."
