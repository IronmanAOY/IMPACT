#!/bin/bash
# MPC-Bench v2 confirmatory run plan (seeds >= 20000; design section 7.4).
#
# Runs only on a clean checkout of the freeze tag: every step goes through
# the v2 confirmatory guard (clean tree, src/ and scripts/ trees of the tag,
# seeds >= 20000) and uses the generated family protocols only
# (protocols/v2/generated/; no drafts). Every run step resumes (completed
# task ids are skipped), so the script can be re-run after an interruption.
# A step that ends with task errors does not stop the later steps (the
# script then exits with 1); a refused step stops the plan.
#
# Order (most decision-relevant first): manipulation checks; anchor
# replication blocks; family-A witnesses (PC_nominal and the six single
# deficits first); C1 witnesses; null-calibration generator; Hopf arm and
# forward family A; family B; twins; RAM-only arm; sweeps and factorial;
# adversaries and held-out conditions; then the admitted Tier-B arms
# (TIER_B=1). Select steps with STEPS="anchor_replication A_witnesses".
#
#   OUT      output root (a workspace; default outputs/paper1_mpcbench/v2_confirmatory)
#   WORKERS  process-pool size per step (default 12)
#   PY       Python interpreter (default python3)
#   TIER_B   1 to run the Tier-B design modules that are merged (admitted
#            at the checkpoint)
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG="${FREEZE_TAG:-mpcbench-freeze-v2}"
OUT="${OUT:-outputs/paper1_mpcbench/v2_confirmatory}"
WORKERS="${WORKERS:-12}"
PY="${PY:-python3}"
TIER_B="${TIER_B:-0}"
STEPS="${STEPS:-manipulation anchor_replication A_witnesses C1_witnesses null_calibration forward family_b twins ram_only sweeps_factorial adversaries_heldout tier_b}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

FAILED=""
has() { [[ " $STEPS " == *" $1 "* ]]; }
outcome() {  # outcome <exit status> <step>
  # exit 1 (task errors or plan differences) is recorded and the next step
  # runs; any other failure (the guard or the plan check refused the run)
  # stops the plan
  if [[ $1 -eq 1 ]]; then
    echo "step $2: task errors or plan differences (see its manifest)"
    FAILED="$FAILED $2"
  elif [[ $1 -ne 0 ]]; then
    echo "step $2: run refused (exit $1); the confirmatory plan stops here"
    exit "$1"
  fi
}
run() {  # run <designs> <out-subdir>
  local rc=0
  "$PY" scripts/run_bench_v2.py run "$1" --split confirmatory --confirmatory \
      --freeze-tag "$TAG" --no-drafts --workers "$WORKERS" --out "$OUT/$2" || rc=$?
  outcome "$rc" "$2"
}
run_module() {  # run_module <design module>
  # a module that is merged but defines no designs for the runner stops the
  # plan (exit 3), so that no arm of the frozen plan is skipped silently
  local NAMES rc=0
  NAMES="$("$PY" scripts/run_bench_v2.py designs --module "$1")" || rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "design module $1: no designs for the runner (exit $rc); the confirmatory plan stops here"
    exit "$rc"
  fi
  if [[ -n "$NAMES" ]]; then
    run "$NAMES" "$1"
  else
    echo "design module $1 is not in this tree; step skipped"
  fi
}

if has manipulation; then
  # prerequisite M (oracle checks only); exit status 1 means a check was
  # not computed, which is reported and does not stop the plan
  "$PY" scripts/run_bench_v2.py manipulation --seeds 20000-20039 \
      --confirmatory --freeze-tag "$TAG" --out "$OUT/manipulation" \
      || echo "prerequisite M: some check was not computed (see the manifest)"
fi
if has anchor_replication; then
  run A_anchors,C1_anchors,RAM160_anchors anchor_replication
fi
if has A_witnesses; then
  run A_witnesses A_witnesses
fi
if has C1_witnesses; then
  run C1_witnesses C1_witnesses
fi
if has null_calibration; then
  run_module null_calibration
fi
if has forward; then
  # Hopf arm, forward family A (EEG, then BOLD with curtailed sampling)
  run_module forward
fi
if has family_b; then
  # family B has its own task model and runs through its validation script,
  # behind the same confirmatory guard
  rc=0
  "$PY" scripts/v2/iim_validation_v2.py --split confirmatory --freeze-tag "$TAG" \
      --workers "$WORKERS" --out "$OUT/family_b" || rc=$?
  outcome "$rc" family_b
fi
if has twins; then
  run A_twins,C1_twins,RAM160_twins twins
fi
if has ram_only; then
  run RAM160 RAM160
fi
if has sweeps_factorial; then
  run A_sweeps,A_factorial A_sweeps_factorial
  run C1_sweeps,C1_factorial C1_sweeps_factorial
fi
if has adversaries_heldout; then
  run A_adversaries,A_heldout A_adversaries_heldout
fi
if has tier_b && [[ "$TIER_B" == "1" ]]; then
  for MOD in srpi_only patchwork_v2 tier_b_arms; do
    run_module "$MOD"
  done
fi
if [[ -n "$FAILED" ]]; then
  echo "steps with task errors or plan differences:$FAILED"
fi
echo "confirmatory runs done; the integrity audit and the evaluator run next, from the tag, on $OUT"
if [[ -n "$FAILED" ]]; then
  exit 1
fi
