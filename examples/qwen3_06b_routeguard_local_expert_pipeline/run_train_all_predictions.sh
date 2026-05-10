#!/usr/bin/env bash
# Precompute each local expert on the full original train_all split, aligned row-by-row.
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PIPE="examples/qwen3_06b_routeguard_local_expert_pipeline"
RESULT_ROOT="/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_local_expert_train_all"
mkdir -p "${RESULT_ROOT}" logs
EXPERTS=("agent" "cyber" "harm" "non_violent" "social")
IFS=',' read -ra GPU_LIST <<<"${GPUS:-0,1,2,3,4}"
if ((${#GPU_LIST[@]} < ${#EXPERTS[@]})); then
  echo "Need at least 5 GPUs in GPUS" >&2
  exit 1
fi
ts(){ date -u +"%Y-%m-%dT%H:%M:%SZ"; }
pids=()
for idx in "${!EXPERTS[@]}"; do
  e="${EXPERTS[$idx]}"
  gpu="${GPU_LIST[$idx]}"
  log="logs/qwen3_06b_routeguard_local_expert_train_all_${e}.log"
  echo "$(ts) START full train_all ${e} on GPU ${gpu}; log=${log}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    llamafactory-cli train "${PIPE}/eval_train_all_${e}.yaml"
  ) >"${log}" 2>&1 &
  pids+=("$!")
done
failed=0
for idx in "${!pids[@]}"; do
  e="${EXPERTS[$idx]}"
  if wait "${pids[$idx]}"; then
    echo "$(ts) DONE ${e}"
  else
    echo "$(ts) FAIL ${e}" >&2
    failed=1
  fi
done
exit "${failed}"
