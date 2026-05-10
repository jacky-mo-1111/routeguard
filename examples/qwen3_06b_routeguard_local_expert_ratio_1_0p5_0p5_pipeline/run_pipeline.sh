#!/usr/bin/env bash
# Pipeline: train + eval five RouteGuard local experts with own:other:safe = 1:0.5:0.5.
#
# Data construction:
#   - Start from old local expert train.json.
#   - Keep all own positive rows.
#   - Downsample other unsafe to 0.5 * own.
#   - Downsample safe to 0.5 * own.
#
# Usage:
#   bash examples/qwen3_06b_routeguard_local_expert_ratio_1_0p5_0p5_pipeline/run_pipeline.sh
#   SKIP_TRAIN=1 bash ...
#   SKIP_EVAL=1 bash ...
#   ONLY_EXPERTS=agent,cyber bash ...
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

PIPE="examples/qwen3_06b_routeguard_local_expert_ratio_1_0p5_0p5_pipeline"
RESULT_ROOT="/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_local_expert_ratio_1_0p5_0p5"
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
  banner "Filtered experts -> ${EXPERTS[*]:-<none>}"
fi

if [[ ${#EXPERTS[@]} -eq 0 ]]; then
  echo "No experts selected." >&2
  exit 1
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  for e in "${EXPERTS[@]}"; do
    run "TRAIN [ratio_1_0p5_0p5/${e}]"       llamafactory-cli train "${PIPE}/train_${e}.yaml"
  done
else
  banner "SKIP_TRAIN=1: skipping all training stages"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  for e in "${EXPERTS[@]}"; do
    run "EVAL  [ratio_1_0p5_0p5/${e}]"       llamafactory-cli train "${PIPE}/eval_${e}.yaml"
  done
else
  banner "SKIP_EVAL=1: skipping all eval stages"
fi

banner "Done. Predictions under: ${RESULT_ROOT}/<expert>/generated_predictions.jsonl"
