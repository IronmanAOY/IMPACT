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
#   --null-surrogates K      add K circular-shift surrogate runs per run (IIM null calibration;
#                            default 0; each surrogate costs one more IIM run)
#   --iim-max-nodes N        IIM subsystem size (default 4)
#   --iim-max-mechanism-size S|all, --iim-max-purview-size S|all, --iim-n-parts P|all
#                            (defaults 2, 2, 4; 'all' = exhaustive). A calibration run for
#                            sizing the production campaign uses e.g. --iim-max-nodes 6 and
#                            'all' for the other three (runbook section 10a)
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
NULL_SURROGATES=0
IIM_MAX_NODES=4
IIM_MAX_MECH=2
IIM_MAX_PURVIEW=2
IIM_N_PARTS=4

usage() { sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

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
    --null-surrogates) NULL_SURROGATES="${2:?}"; shift ;;
    --iim-max-nodes) IIM_MAX_NODES="${2:?}"; shift ;;
    --iim-max-mechanism-size) IIM_MAX_MECH="${2:?}"; shift ;;
    --iim-max-purview-size) IIM_MAX_PURVIEW="${2:?}"; shift ;;
    --iim-n-parts) IIM_N_PARTS="${2:?}"; shift ;;
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

# Default: tiny, fast IIM problem (4 nodes, <=2-node mechanisms/purviews, 4 cuts); one shard of
# each kind per run. 'all' leaves a size flag out (exhaustive). The shard jobs record their wall
# time and Psi evaluations (timing/); reduce-all writes cost_calibration.json from them.
iim_args=(--iim-max-nodes "${IIM_MAX_NODES}")
[ "${IIM_MAX_MECH}" = "all" ] || iim_args+=(--iim-max-mechanism-size "${IIM_MAX_MECH}")
[ "${IIM_MAX_PURVIEW}" = "all" ] || iim_args+=(--iim-max-purview-size "${IIM_MAX_PURVIEW}")
[ "${IIM_N_PARTS}" = "all" ] || iim_args+=(--iim-n-parts "${IIM_N_PARTS}")
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
  "${iim_args[@]}" \
  --hunter-phase1-shards-per-run 1 \
  --hunter-cut-shards-per-run 1 \
  --hunter-iim-null-surrogates "${NULL_SURROGATES}" \
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
echo "Measured rate for sizing the production campaign: ${CAMPAIGN_DIR}/cost_calibration.json"
echo "  (shard_wall_seconds_per_psi_evaluation -> --hunter-seconds-per-psi-eval; runbook section 10a)"
