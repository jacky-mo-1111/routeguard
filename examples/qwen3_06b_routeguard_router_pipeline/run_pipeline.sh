#!/usr/bin/env bash
# Pipeline: train/eval RouteGuard router, then score router + precomputed local experts.
#
# Router behavior:
#   SAFE                  -> final safe
#   ROUTE = expert(s)     -> union outputs from those local expert prediction files
#   empty expert union    -> final safe
#
# Usage:
#   bash examples/qwen3_06b_routeguard_router_pipeline/run_pipeline.sh
#   SKIP_TRAIN=1 bash ...       # only router eval + metrics
#   SKIP_EVAL=1 bash ...        # only train
#   ONLY_METRICS=1 bash ...     # only compute metrics from existing router/expert predictions
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

PIPE="examples/qwen3_06b_routeguard_router_pipeline"
RESULT_ROOT="/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_router"
EXPERT_ROOT="${EXPERT_ROOT:-/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_local_expert}"
BASELINE_PRED="${BASELINE_PRED:-/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard/baseline/test_eval_category_label/generated_predictions.jsonl}"
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
    run "TRAIN [router] routeguard_router_train" \
      llamafactory-cli train "${PIPE}/train_router.yaml"
  else
    banner "SKIP_TRAIN=1: skipping router training"
  fi

  if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    run "EVAL  [router] routeguard_router_test_eval" \
      llamafactory-cli train "${PIPE}/eval_router.yaml"
  else
    banner "SKIP_EVAL=1: skipping router eval"
  fi
fi

run "METRICS [router + local experts]" \
  python "${PIPE}/evaluate_router_routeguard.py" \
    --router-pred "${RESULT_ROOT}/router/generated_predictions.jsonl" \
    --expert-root "${EXPERT_ROOT}" \
    --baseline-pred "${BASELINE_PRED}" \
    --out-dir "${RESULT_ROOT}"

banner "Done. Final report: ${RESULT_ROOT}/router_routeguard_result.txt"
