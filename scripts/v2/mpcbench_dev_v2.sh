#!/bin/bash
# MPC-Bench v2 development runs (development seeds 0-999 only).
#
# Steps, in order (select with STEPS="anchors twins"):
#   manipulation  prerequisite M and the realisation checks of the new
#                 systems on development seeds (oracle channels only)
#   anchors       reference blocks 900-939: PC_nominal and the own-lesion
#                 witnesses of A-R, A-H, C1-R, C1-H and A-RAM160 (anchors,
#                 validity, specificity, testability), and the reference
#                 conditions of the forward views at the development regime
#   twins         development twin networks 820-824 (SE calibration)
#   dry_run       every Tier-A design of this runner at about 15 % scale
#                 (seeds 320-383; the forward arms 384-399 at the
#                 development regime)
#   smoke         held-out conditions on seeds 980-984 (family A, and the
#                 forward arms' anchor conditions at the held-out regime);
#                 the runner keeps status lines only (smoke.jsonl) and
#                 discards the outputs
#   modules       the other v2 modules that are merged: family B through
#                 its validation script (development mirrors 400-439) and
#                 the null-calibration generator through the runner when it
#                 defines designs for it
#
# Every run step resumes: completed task ids are skipped, failed tasks and a
# line cut by an interruption are run again. A step that ends with task
# errors does not stop the later steps; the script then exits with 1. Development runs may use draft
# protocols (anchors pending) until the generated protocols exist.
#
#   OUT      output root (default outputs/paper1_mpcbench/v2_dev)
#   WORKERS  process-pool size (default 12)
#   PY       Python interpreter (default python3)
#   EXTRA    extra run options, e.g. "--principles RAM,PDI,SRPI" or
#            "--allow-unavailable-estimators"
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-outputs/paper1_mpcbench/v2_dev}"
WORKERS="${WORKERS:-12}"
PY="${PY:-python3}"
STEPS="${STEPS:-manipulation anchors twins dry_run smoke modules}"
EXTRA="${EXTRA:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

FAILED=""
has() { [[ " $STEPS " == *" $1 "* ]]; }
outcome() {  # outcome <exit status> <step>
  # exit 1 (task errors or plan differences) is recorded and the next step
  # runs; any other failure (a refused run) stops the script
  if [[ $1 -eq 1 ]]; then
    echo "step $2: task errors or plan differences (see its manifest)"
    FAILED="$FAILED $2"
  elif [[ $1 -ne 0 ]]; then
    echo "step $2: run refused (exit $1)"
    exit "$1"
  fi
}
run() {  # run <designs> <out-subdir> [extra run options]
  local rc=0
  # shellcheck disable=SC2086
  "$PY" scripts/run_bench_v2.py run "$1" --split development \
      --workers "$WORKERS" --out "$OUT/$2" "${@:3}" $EXTRA || rc=$?
  outcome "$rc" "$2"
}

if has manipulation; then
  "$PY" scripts/run_bench_v2.py manipulation --seeds 320-359 \
      --out "$OUT/manipulation" || echo "prerequisite M: some check not computed"
fi
if has anchors; then
  run A_anchors,C1_anchors,RAM160_anchors,forward_anchor_replication anchors
fi
if has twins; then
  run A_twins,C1_twins,RAM160_twins twins
fi
if has dry_run; then
  run A_witnesses dry_run/A_witnesses
  run A_sweeps,A_factorial dry_run/A_sweeps_factorial
  run A_adversaries dry_run/A_adversaries
  run RAM160 dry_run/RAM160
  run C1_witnesses,C1_sweeps,C1_factorial dry_run/C1
  run whole_brain,forward_family_a,forward_family_a_bold dry_run/forward
fi
if has smoke; then
  run A_heldout smoke
  # the forward arms' anchor conditions at the held-out regime
  run whole_brain,forward_family_a,forward_family_a_bold smoke/forward --purpose smoke
fi
if has modules; then
  rc=0
  "$PY" scripts/v2/iim_validation_v2.py --split development --workers "$WORKERS" \
      --out "$OUT/modules/family_b" || rc=$?
  outcome "$rc" modules/family_b
  for MOD in null_calibration; do
    rc=0
    NAMES="$("$PY" scripts/run_bench_v2.py designs --module "$MOD")" || rc=$?
    if [[ $rc -ne 0 ]]; then
      # merged, but no designs for the runner yet: reported, not skipped quietly
      echo "design module $MOD: no designs for the runner (exit $rc); skipped"
      FAILED="$FAILED modules/$MOD"
    elif [[ -n "$NAMES" ]]; then
      run "$NAMES" "modules/$MOD"
    else
      echo "design module $MOD is not merged yet; skipped"
    fi
  done
fi
if [[ -n "$FAILED" ]]; then
  echo "steps with task errors or plan differences:$FAILED"
  exit 1
fi
