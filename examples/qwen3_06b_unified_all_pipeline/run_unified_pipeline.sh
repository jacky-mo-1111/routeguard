#!/usr/bin/env bash
# Train one Qwen3-Guard-0.6B on: train_all + all four specialist guardrail trains,
# evaluate on every eval split from the attackactor / orbench / policyguard / redcoder pipelines,
# then merge those metrics into each pipeline's existing result.txt.
#
# Usage:
#   bash examples/qwen3_06b_unified_all_pipeline/run_unified_pipeline.sh
#   SKIP_TRAIN=1 bash ...                # eval + metrics only (expects saves/qwen3_06b_unified_all)
#   SKIP_EVAL=1 bash ...                 # train + metrics merge only if predictions exist
#   ONLY_MERGE_METRICS=1 bash ...        # run apply script only (expects eval predictions)

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

PIPE="${ROOT}/examples/qwen3_06b_unified_all_pipeline"
RESULT_EVAL="${ROOT}/results/qwen3_06b_unified_all/eval"
mkdir -p "${ROOT}/results/qwen3_06b_unified_all" logs

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

if [[ "${ONLY_MERGE_METRICS:-0}" == "1" ]]; then
  run "METRICS merge into per-experiment result.txt" \
    python "${PIPE}/apply_unified_eval_to_experiment_results.py" \
      --repo-root "${ROOT}" --pred-root "${RESULT_EVAL}"
  banner "Done. Also see: ${ROOT}/results/qwen3_06b_unified_all/unified_metrics.txt"
  exit 0
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  run "TRAIN unified (all training sets)" \
    llamafactory-cli train "${PIPE}/train_unified.yaml"
else
  banner "SKIP_TRAIN=1: skipping training"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  run "EVAL unified ckpt on all splits → ${RESULT_EVAL}" \
    llamafactory-cli train "${PIPE}/eval_unified.yaml"
else
  banner "SKIP_EVAL=1: skipping eval"
fi

run "Merge unified eval into qwen3_06b_{attackactor,orbench,policyguard,redcoder}/result.txt" \
  python "${PIPE}/apply_unified_eval_to_experiment_results.py" \
    --repo-root "${ROOT}" --pred-root "${RESULT_EVAL}"

banner "Done. Unified summary: ${ROOT}/results/qwen3_06b_unified_all/unified_metrics.txt"
