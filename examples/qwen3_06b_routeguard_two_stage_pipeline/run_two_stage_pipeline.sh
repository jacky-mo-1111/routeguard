#!/usr/bin/env bash
# Pipeline: train/eval stage-2 expert router, then score baseline-gated RouteGuard.
#
# Stage 1 is not trained here. It uses BASELINE_PRED as the safe/unsafe gate:
#   baseline predicts safe       -> final safe
#   baseline predicts categories -> stage2 router selects experts, then union expert outputs
#
# Usage:
#   bash examples/qwen3_06b_routeguard_two_stage_pipeline/run_two_stage_pipeline.sh
#   SKIP_TRAIN=1 bash ...
#   SKIP_EVAL=1 bash ...
#   ONLY_METRICS=1 bash ...
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash ...

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate dl 2>/dev/null || true
fi

export WANDB_DISABLED=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PIPE="examples/qwen3_06b_routeguard_two_stage_pipeline"
RESULT_ROOT="${RESULT_ROOT:-/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_two_stage}"
EXPERT_ROOT="${EXPERT_ROOT:-/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_local_expert}"
BASELINE_PRED="${BASELINE_PRED:-/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard/baseline/test_eval_category_label/generated_predictions.jsonl}"
STAGE2_EVAL_DATA="${STAGE2_EVAL_DATA:-/nas02/jacky/data/route_guard_final/category_label/router_stage2_baseline_gate/test_eval_baseline_unsafe.json}"
mkdir -p "${RESULT_ROOT}" logs

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo; echo "================================ $(ts)  $* ================================"; }
run() {
  banner "$1"
  shift
  if ! "$@"; then
    echo "ERROR: command failed: $*" >&2
    exit 1
  fi
}

if [[ "${ONLY_METRICS:-0}" != "1" ]]; then
  if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    run "TRAIN [stage2_router] routeguard_stage2_router_train" \
      llamafactory-cli train "${PIPE}/train_stage2_router.yaml"
  else
    banner "SKIP_TRAIN=1: skipping stage2 router training"
  fi

  if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    run "EVAL  [stage2_router] routeguard_stage2_router_test_eval_baseline_unsafe" \
      llamafactory-cli train "${PIPE}/eval_stage2_router.yaml"
  else
    banner "SKIP_EVAL=1: skipping stage2 router eval"
  fi
fi

run "METRICS [baseline gate + stage2 router + local experts]" \
  python "${PIPE}/evaluate_two_stage_routeguard.py" \
    --stage2-pred "${RESULT_ROOT}/stage2_router/generated_predictions.jsonl" \
    --stage2-eval-data "${STAGE2_EVAL_DATA}" \
    --baseline-pred "${BASELINE_PRED}" \
    --expert-root "${EXPERT_ROOT}" \
    --out-dir "${RESULT_ROOT}" \
    --empty-stage2-fallback "${EMPTY_STAGE2_FALLBACK:-safe}"

banner "Done. Final report: ${RESULT_ROOT}/two_stage_routeguard_result.txt"
