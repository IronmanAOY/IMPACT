#!/bin/bash
# MPC-Bench confirmatory run plan of paper 1 (preregistered; see
# docs/preregistration/MPC_BENCH_PREREGISTRATION.md, section 7).
#
# Runs only on a clean checkout of the code-freeze tag (every step checks it:
# run_bench.py / iim_validation.py / null_calibration.py --confirmatory
# --freeze-tag). Every run_bench step is resumable (completed task ids are
# skipped), so the script can be re-run after an interruption. Steps can be
# selected: STEPS="bench_A bench_C" scripts/mpcbench_confirmatory.sh
#
#   OUT      output root (a workspace; default outputs/paper1_mpcbench/confirmatory)
#   WORKERS  process-pool size per step (default 14)
#   PY       Python interpreter (default: python3)
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${FREEZE_TAG:-mpcbench-freeze-v1}"
OUT="${OUT:-outputs/paper1_mpcbench/confirmatory}"
WORKERS="${WORKERS:-14}"
PY="${PY:-python3}"
PROTOCOL="protocols/mpc_bench_v1.json"
NANCH="$($PY -c 'import json; print(",".join(json.load(open("protocols/mpc_bench_v1_anchored.json"))["necessity_set"]))')"
STEPS="${STEPS:-manipulation bench_A bench_C reference_C adversarial whole_brain null_calibration iim_validation audit hypotheses}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COMMON=(--null-surrogates 19 --se-groups 10 --protocol "$PROTOCOL"
        --confirmatory --freeze-tag "$TAG" --workers "$WORKERS")
EVAL_SEEDS="10000-10019"      # witnesses, sweeps, adversarial
SMALL_SEEDS="10000-10009"     # factorial, patchwork sweep, whole-brain
REF_C_SEEDS="19000-19019"     # family-C positive-control reference block

has() { [[ " $STEPS " == *" $1 "* ]]; }

if has manipulation; then
  # oracle manipulation checks, families A and C (no estimator)
  $PY scripts/run_bench.py manipulation --seeds 10000-10039 \
      --confirmatory --freeze-tag "$TAG" --out "$OUT/manipulation" || true
fi
for FAM in A C; do
  if has "bench_$FAM"; then
    $PY scripts/run_bench.py witnesses --family "$FAM" --seeds "$EVAL_SEEDS" \
        "${COMMON[@]}" --out "$OUT/bench/$FAM/witnesses"
    $PY scripts/run_bench.py sweep --family "$FAM" --seeds "$EVAL_SEEDS" --levels 10 \
        "${COMMON[@]}" --out "$OUT/bench/$FAM/sweep"
    $PY scripts/run_bench.py factorial --family "$FAM" --seeds "$SMALL_SEEDS" \
        "${COMMON[@]}" --out "$OUT/bench/$FAM/factorial"
    $PY scripts/run_bench.py patchwork_sweep --family "$FAM" --seeds "$SMALL_SEEDS" \
        --levels 6 "${COMMON[@]}" --out "$OUT/bench/$FAM/patchwork_sweep"
  fi
done
if has reference_C; then
  # family C is anchored on its own positive control (disjoint seed block)
  $PY scripts/run_bench.py witnesses --family C --witnesses PC_nominal \
      --seeds "$REF_C_SEEDS" --no-markers "${COMMON[@]}" --out "$OUT/reference_C"
fi
if has adversarial; then
  $PY scripts/run_bench.py adversarial --seeds "$EVAL_SEEDS" "${COMMON[@]}" \
      --out "$OUT/bench/adversarial"
fi
if has whole_brain; then
  # descriptive (no confirmatory hypothesis): G sweep and lesions, 3 observations
  $PY scripts/run_bench.py whole_brain --seeds "$SMALL_SEEDS" "${COMMON[@]}" \
      --out "$OUT/bench/whole_brain"
fi
if has null_calibration; then
  $PY scripts/null_calibration.py --out "$OUT/null_calibration" \
      --protocol "$PROTOCOL" --kinds ar1,pink,surrogate_iid,surrogate_linear \
      --T 1200,2400 --nodes 8,16 --replicates 100 --null-surrogates 19 \
      --se-groups 10 --seed 10000 --workers "$WORKERS" \
      --confirmatory --freeze-tag "$TAG"
fi
if has iim_validation; then
  $PY scripts/iim_validation.py --out "$OUT/iim_validation" \
      --T 1000,3000,10000,30000 --seeds "$EVAL_SEEDS" --null-surrogates 19 \
      --workers "$WORKERS" --confirmatory --freeze-tag "$TAG"
fi
if has audit; then
  # primary: the anchored necessity set N_anch; secondary: all five principles
  $PY scripts/benchmark_attribution_rules.py --results "$OUT/bench" \
      --necessity-set "$NANCH" --out "$OUT/audit"
  $PY scripts/benchmark_attribution_rules.py --results "$OUT/bench" \
      --necessity-set RAM,PDI,NAS,IIM,SRPI --out "$OUT/audit_all5"
fi
if has hypotheses; then
  $PY scripts/bench_hypotheses.py --out "$OUT/hypotheses" --freeze-tag "$TAG" \
      --bench-a "$OUT/bench/A" --bench-c "$OUT/bench/C" \
      --reference-c "$OUT/reference_C" --adversarial "$OUT/bench/adversarial" \
      --null-calibration "$OUT/null_calibration" \
      --iim-validation "$OUT/iim_validation" --audit "$OUT/audit" \
      --manipulation "$OUT/manipulation"
fi
