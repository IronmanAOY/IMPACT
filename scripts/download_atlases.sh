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
if [ "${VERIFY_ONLY}" -eq 1 ] && [ ! -d "${DEST}" ]; then
  echo "Atlas folder not found: ${DEST}" >&2
  exit 1
fi
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

# Pinned SHA-256 of one relative path.
pinned_sha() {
  local entry
  for entry in "${ATLAS_FILES[@]}"; do
    if [ "${entry%%|*}" = "$1" ]; then
      echo "${entry##*|}"
      return 0
    fi
  done
  return 1
}

# Succeeds only if a downloaded (not yet installed) file has the pinned
# checksum; a changed upstream file must never replace anything in DEST.
download_ok() {
  local file="$1" rel="$2" want got
  want="$(pinned_sha "${rel}")"
  if [ ! -f "${file}" ]; then
    echo "Download did not contain atlases/${rel}; local files left unchanged." >&2
    return 1
  fi
  got="$(sha256_of "${file}")"
  if [ "${got}" != "${want}" ]; then
    echo "Checksum mismatch for downloaded atlases/${rel} (got ${got});" \
      "local files left unchanged." >&2
    return 1
  fi
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

# Every download is checked in TMP_DIR first and only then moved into DEST,
# so a changed or truncated upstream file never replaces a local atlas.
fetched_aal=0
failed=0
for rel in "${need_fetch[@]}"; do
  case "${rel}" in
    schaefer_2018/*)
      tmp_file="${TMP_DIR}/$(basename "${rel}")"
      download "${SCHAEFER_URL}/$(basename "${rel}")" "${tmp_file}"
      if download_ok "${tmp_file}" "${rel}"; then
        mkdir -p "${DEST}/schaefer_2018"
        mv -f "${tmp_file}" "${DEST}/${rel}"
      else
        failed=1
      fi
      ;;
    aal_SPM12/*)
      # One archive holds all AAL files: verify every pinned AAL file in the
      # extracted copy before the local aal/ folder is replaced as a whole.
      if [ "${fetched_aal}" -eq 0 ]; then
        fetched_aal=1
        download "${AAL_URL}" "${TMP_DIR}/aal_for_SPM12.tar.gz"
        mkdir -p "${TMP_DIR}/aal"
        tar -xzf "${TMP_DIR}/aal_for_SPM12.tar.gz" -C "${TMP_DIR}/aal"
        aal_ok=1
        for entry in "${ATLAS_FILES[@]}"; do
          aal_rel="${entry%%|*}"
          case "${aal_rel}" in
            aal_SPM12/aal/*)
              download_ok "${TMP_DIR}/aal/${aal_rel#aal_SPM12/}" "${aal_rel}" || aal_ok=0
              ;;
          esac
        done
        if [ "${aal_ok}" -eq 1 ]; then
          mkdir -p "${DEST}/aal_SPM12"
          rm -rf "${DEST}/aal_SPM12/aal"
          mv "${TMP_DIR}/aal/aal" "${DEST}/aal_SPM12/aal"
        else
          failed=1
        fi
      fi
      ;;
    shen_1mm_268_parcellation.nii.gz)
      download "${SHEN_URL}" "${TMP_DIR}/shen.nii.gz"
      if download_ok "${TMP_DIR}/shen.nii.gz" "${rel}"; then
        mv -f "${TMP_DIR}/shen.nii.gz" "${DEST}/${rel}"
      else
        failed=1
      fi
      ;;
  esac
done

# Final check of the installed tree.
for entry in "${ATLAS_FILES[@]}"; do
  rel="${entry%%|*}"
  want="${entry##*|}"
  state="$(file_state "${rel}" "${want}")"
  if [ "${state}" != ok ]; then
    echo "Not verified after download (${state}): atlases/${rel}" >&2
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  exit 1
fi
echo "Atlas files downloaded and verified in ${DEST}."
