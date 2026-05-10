#!/usr/bin/env bash
# Run train-split inference for the five RouteGuard local experts.
#
# This does not train models. It uses the existing checkpoints under:
#   saves/qwen3_06b_routeguard_local_expert/<expert>
# and writes predictions to:
#   results/qwen3_06b_routeguard_local_expert_train/<expert>/generated_predictions.jsonl
#
# Fast path: run one expert per GPU concurrently. With five 0.6B models, this is
# faster than sequential 8-GPU DDP inference and avoids overwriting test_eval results.
#
# Usage:
#   bash examples/qwen3_06b_routeguard_local_expert_pipeline/run_train_predictions.sh
#   GPUS=0,1,2,3,4 bash ...       # default
#   GPUS=0,1,2,3,4,5,6,7 bash ... # also OK; only first 5 are used
#   ONLY_EXPERTS=agent,cyber bash ...
#   EVAL_BATCH_SIZE=32 bash ...    # if 64 OOMs, edit yaml or set with sed yourself

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
RESULT_ROOT="/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_local_expert_train"
mkdir -p "${RESULT_ROOT}" logs

EXPERTS=("agent" "cyber" "harm" "non_violent" "social")
if [[ -n "${ONLY_EXPERTS:-}" ]]; then
  IFS=',' read -ra _filter <<<"${ONLY_EXPERTS}"
  declare -a FILTERED=()
  for e in "${EXPERTS[@]}"; do
    for keep in "${_filter[@]}"; do
      if [[ "${e}" == "${keep}" ]]; then
        FILTERED+=("${e}")
      fi
    done
  done
  EXPERTS=("${FILTERED[@]}")
fi

IFS=',' read -ra GPU_LIST <<<"${GPUS:-0,1,2,3,4}"
if ((${#GPU_LIST[@]} < ${#EXPERTS[@]})); then
  echo "ERROR: Need at least ${#EXPERTS[@]} GPUs in GPUS=${GPUS:-0,1,2,3,4}; got ${#GPU_LIST[@]}" >&2
  exit 1
fi

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "$(ts) Running train-split predictions for experts: ${EXPERTS[*]}"
echo "$(ts) GPU assignment: ${GPU_LIST[*]}"

pids=()
for idx in "${!EXPERTS[@]}"; do
  e="${EXPERTS[$idx]}"
  gpu="${GPU_LIST[$idx]}"
  log="logs/qwen3_06b_routeguard_local_expert_train_${e}.log"
  echo "$(ts) START ${e} on GPU ${gpu}; log=${log}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    llamafactory-cli train "${PIPE}/eval_train_${e}.yaml"
  ) >"${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  e="${EXPERTS[$idx]}"
  if wait "${pid}"; then
    echo "$(ts) DONE  ${e} -> ${RESULT_ROOT}/${e}/generated_predictions.jsonl"
  else
    echo "$(ts) FAIL  ${e}; see logs/qwen3_06b_routeguard_local_expert_train_${e}.log" >&2
    failed=1
  fi
done

if ((failed)); then
  exit 1
fi

echo "$(ts) All train-split predictions complete: ${RESULT_ROOT}"
