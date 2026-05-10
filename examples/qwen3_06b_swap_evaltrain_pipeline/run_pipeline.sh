#!/usr/bin/env bash
# Swap train/eval pipeline:
# - Combined model: train on all eval splits
# - Separate models: one model per eval split
# - Evaluate every model on all training splits

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

PIPE="examples/qwen3_06b_swap_evaltrain_pipeline"
RESULT_ROOT="/nas02/jacky/Debug_LM/results/qwen3_06b_swap_evaltrain"
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
    run "TRAIN [1] combined (all eval splits)" \
      llamafactory-cli train "${PIPE}/train_combined.yaml"
    run "TRAIN [2.1] separate (test_eval)" \
      llamafactory-cli train "${PIPE}/train_test_eval.yaml"
    run "TRAIN [2.2] separate (actorattack_guardrail_eval)" \
      llamafactory-cli train "${PIPE}/train_actorattack_eval.yaml"
    run "TRAIN [2.3] separate (or_bench_ob_eval)" \
      llamafactory-cli train "${PIPE}/train_orbench_eval.yaml"
    run "TRAIN [2.4] separate (policyguard_guardrail_eval)" \
      llamafactory-cli train "${PIPE}/train_policyguard_eval.yaml"
    run "TRAIN [2.5] separate (redcoder_guardrail_eval)" \
      llamafactory-cli train "${PIPE}/train_redcoder_eval.yaml"
  else
    banner "SKIP_TRAIN=1: skipping all training stages"
  fi

  if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    run "EVAL [1] combined model on all training splits" \
      llamafactory-cli train "${PIPE}/eval_combined.yaml"
    run "EVAL [2.1] test_eval model on all training splits" \
      llamafactory-cli train "${PIPE}/eval_test_eval_model.yaml"
    run "EVAL [2.2] actorattack_eval model on all training splits" \
      llamafactory-cli train "${PIPE}/eval_actorattack_model.yaml"
    run "EVAL [2.3] orbench_eval model on all training splits" \
      llamafactory-cli train "${PIPE}/eval_orbench_model.yaml"
    run "EVAL [2.4] policyguard_eval model on all training splits" \
      llamafactory-cli train "${PIPE}/eval_policyguard_model.yaml"
    run "EVAL [2.5] redcoder_eval model on all training splits" \
      llamafactory-cli train "${PIPE}/eval_redcoder_model.yaml"
  else
    banner "SKIP_EVAL=1: skipping all eval stages"
  fi
fi

run "METRICS -> ${RESULT_ROOT}/result.txt" \
  python "${PIPE}/compute_metrics.py" --root "${RESULT_ROOT}"

banner "Done. Final report: ${RESULT_ROOT}/result.txt"
