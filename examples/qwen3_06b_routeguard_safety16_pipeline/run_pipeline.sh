#!/usr/bin/env bash
# Full Safety16 RouteGuard pipeline.
#
# Trains one new safety expert and one new router. Existing five unsafe experts
# are reused from results/qwen3_06b_routeguard_local_expert for test inference.
#
# Usage:
#   bash examples/qwen3_06b_routeguard_safety16_pipeline/run_pipeline.sh
#   SKIP_SAFETY_TRAIN=1 bash ...   # if safety checkpoint exists
#   SKIP_ROUTER_TRAIN=1 bash ...   # if router checkpoint exists
#   SKIP_EVAL=1 bash ...           # train only
#   ONLY_METRICS=1 bash ...        # metrics only from existing predictions
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ...

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PIPE="examples/qwen3_06b_routeguard_safety16_pipeline"
RESULT_ROOT="/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_safety16"
EXPERT_ROOT="${EXPERT_ROOT:-/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_local_expert}"
BASELINE_PRED="${BASELINE_PRED:-/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard/baseline/test_eval_category_label/generated_predictions.jsonl}"
mkdir -p "${RESULT_ROOT}" logs

ts(){ date -u +"%Y-%m-%dT%H:%M:%SZ"; }
run(){ echo; echo "================================ $(ts)  $1 ================================"; shift; "$@"; }

if [[ "${ONLY_METRICS:-0}" != "1" ]]; then
  if [[ "${SKIP_SAFETY_TRAIN:-0}" != "1" ]]; then
    run "TRAIN safety expert" llamafactory-cli train "${PIPE}/train_safety.yaml"
  fi
  if [[ "${SKIP_ROUTER_TRAIN:-0}" != "1" ]]; then
    run "TRAIN safety16 router" llamafactory-cli train "${PIPE}/train_router.yaml"
  fi
  if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    run "EVAL safety expert" llamafactory-cli train "${PIPE}/eval_safety.yaml"
    run "EVAL safety16 router" llamafactory-cli train "${PIPE}/eval_router.yaml"
  fi
fi

run "METRICS safety16 routeguard" \
  python "${PIPE}/evaluate_safety16_routeguard.py" \
    --router-pred "${RESULT_ROOT}/router/generated_predictions.jsonl" \
    --safety-pred "${RESULT_ROOT}/safety/generated_predictions.jsonl" \
    --expert-root "${EXPERT_ROOT}" \
    --baseline-pred "${BASELINE_PRED}" \
    --out-dir "${RESULT_ROOT}"

echo "$(ts) Done: ${RESULT_ROOT}/safety16_result.txt"
