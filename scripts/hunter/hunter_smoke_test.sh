#!/bin/bash
# hunter_smoke_test.sh -- build a tiny IMPaCT IIM campaign and (optionally) submit its single-job
# smoke test (pbs/90_smoke_all_in_one.pbs) to the PBS 'test' queue (25 min, 1 job per user) [KB].
#
# The smoke job runs the hardware self-test (CuPy vs NumPy), every phase-1 and cut shard packed
# 4 per node through PALS mpiexec, the merged reduce and finalize -- all inside one job.
# Run it by hand on a Hunter login node after sourcing the setup file (see hunter_pbs_setup.sh).
#
# Usage:
#   bash scripts/hunter/hunter_smoke_test.sh --bids-root DIR --out-dir DIR [options]
# Options:
#   --dataset-id ID          (default: ds003171)
#   --data-origin real|dummy (default: dummy)
#   --subjects "A B"         restrict to 1-2 subjects to stay within 25 minutes (recommended)
#   --hardware-target T      cpu|auto|gpu|hunter-apu (default: hunter-apu)
#   --campaign-dir DIR       (default: <out-dir>/cache/hunter_iim_smoke)
#   --python PY              interpreter (default: $IMPACT_HUNTER_PYTHON or python3)
#   --submit                 qsub the smoke job (default: only print the command)
# Output location: with --data-origin dummy, --out-dir is used only if it lies under
# ${IMPACT_SYNTH_ROOT:-<repo>}/test_objects; otherwise the preprocessed inputs are read from and
# the results written to test_objects/runs/<dataset> there. With real, use a separate --out-dir
# (a copy or symlink of preprocessed/): finalize overwrites <out-dir>/cache/step2_*.
set -eo pipefail

DATASET_ID=ds003171
ORIGIN=dummy
SUBJECTS=""
TARGET=hunter-apu
BIDS_ROOT=""
OUT_DIR=""
CAMPAIGN_DIR=""
PY="${IMPACT_HUNTER_PYTHON:-python3}"
SUBMIT=0

usage() { sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --bids-root) BIDS_ROOT="${2:?}"; shift ;;
    --out-dir) OUT_DIR="${2:?}"; shift ;;
    --dataset-id) DATASET_ID="${2:?}"; shift ;;
    --data-origin) ORIGIN="${2:?}"; shift ;;
    --subjects) SUBJECTS="${2:?}"; shift ;;
    --hardware-target) TARGET="${2:?}"; shift ;;
    --campaign-dir) CAMPAIGN_DIR="${2:?}"; shift ;;
    --python) PY="${2:?}"; shift ;;
    --submit) SUBMIT=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if [ -z "${BIDS_ROOT}" ] || [ -z "${OUT_DIR}" ]; then
  usage >&2
  exit 2
fi
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-${OUT_DIR}/cache/hunter_iim_smoke}"
if [ -z "${IMPACT_HUNTER_SETUP_FILE:-}" ]; then
  echo "WARNING: IMPACT_HUNTER_SETUP_FILE is not set; the generated jobs will not load modules/venv." >&2
fi
if [ -z "${SUBJECTS}" ]; then
  echo "WARNING: no --subjects given; the smoke job may exceed the 25 min 'test' queue limit." >&2
fi

subject_args=()
if [ -n "${SUBJECTS}" ]; then
  # shellcheck disable=SC2206
  subject_args=(--subjects ${SUBJECTS})
fi

# Tiny, fast IIM problem: 4 nodes, <=2-node mechanisms/purviews, 4 cuts, one shard of each kind per run.
"${PY}" "${repo_root}/run_pipeline.py" \
  --execution-mode hunter \
  --hunter-stage build-campaign \
  --hunter-campaign-dir "${CAMPAIGN_DIR}" \
  --dataset-id "${DATASET_ID}" \
  --data-origin "${ORIGIN}" \
  --bids-root "${BIDS_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --hardware-target "${TARGET}" \
  --mpc-metrics IIM \
  --no-ci \
  --iim-max-nodes 4 \
  --iim-n-parts 4 \
  --iim-max-mechanism-size 2 \
  --iim-max-purview-size 2 \
  --hunter-phase1-shards-per-run 1 \
  --hunter-cut-shards-per-run 1 \
  ${subject_args[@]+"${subject_args[@]}"}

smoke="${CAMPAIGN_DIR}/pbs/90_smoke_all_in_one.pbs"
if [ ! -f "${smoke}" ]; then
  echo "ERROR: ${smoke} was not generated (is the PBS scheduler selected?)" >&2
  exit 1
fi
cmd="cd \"${CAMPAIGN_DIR}/pbs/logs\" && qsub \"${smoke}\""
if [ "${SUBMIT}" -eq 1 ]; then
  (cd "${CAMPAIGN_DIR}/pbs/logs" && qsub "${smoke}")
else
  echo "Smoke campaign built in ${CAMPAIGN_DIR}. Submit with:"
  echo "  ${cmd}"
fi
echo "Afterwards: python3 run_pipeline.py --execution-mode hunter --hunter-stage status --hunter-campaign-dir ${CAMPAIGN_DIR} --dataset-id ${DATASET_ID} --data-origin ${ORIGIN} --out-dir ${OUT_DIR}"
