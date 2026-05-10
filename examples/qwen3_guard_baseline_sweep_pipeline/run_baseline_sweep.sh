#!/usr/bin/env bash
# Train/eval a sweep of category-label guardrail baselines on train_all -> test_eval.
#
# Usage:
#   bash examples/qwen3_guard_baseline_sweep_pipeline/run_baseline_sweep.sh
#   SKIP_TRAIN=1 bash examples/qwen3_guard_baseline_sweep_pipeline/run_baseline_sweep.sh
#   SKIP_EVAL=1 bash examples/qwen3_guard_baseline_sweep_pipeline/run_baseline_sweep.sh
#   ONLY_METRICS=1 bash examples/qwen3_guard_baseline_sweep_pipeline/run_baseline_sweep.sh
#   ONLY_MODELS=qwen3guard_0_6b,llamaguard3_1b bash examples/qwen3_guard_baseline_sweep_pipeline/run_baseline_sweep.sh
#
# Notes:
#   - Runs one full-SFT job at a time with DeepSpeed ZeRO-3 over all visible GPUs.
#   - Default GPUs are 0,1,2,3.
#   - Results are written to /nas02/jacky/Debug_LM/results/qwen3_guard_baseline_sweep.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate dl 2>/dev/null || true
fi

export WANDB_DISABLED=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PIPE="examples/qwen3_guard_baseline_sweep_pipeline"
RESULT_ROOT="/nas02/jacky/Debug_LM/results/qwen3_guard_baseline_sweep"
SAVE_ROOT="/nas02/jacky/Debug_LM/saves/qwen3_guard_baseline_sweep"
mkdir -p "${RESULT_ROOT}" "${SAVE_ROOT}" logs

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

MODELS=(
  qwen3guard_0_6b
  qwen3guard_4b
  qwen3guard_8b
  llamaguard2_8b
  llamaguard3_1b
  llamaguard3_8b
  shieldgemma_2b
  shieldgemma_9b
)

if [[ -n "${ONLY_MODELS:-}" ]]; then
  IFS=',' read -ra _filter <<<"${ONLY_MODELS}"
  declare -a FILTERED=()
  for m in "${MODELS[@]}"; do
    for keep in "${_filter[@]}"; do
      if [[ "${m}" == "${keep}" ]]; then
        FILTERED+=("${m}")
      fi
    done
  done
  MODELS=("${FILTERED[@]}")
  banner "Filtered models -> ${MODELS[*]:-<none>}"
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No models selected." >&2
  exit 1
fi

if [[ "${ONLY_METRICS:-0}" != "1" ]]; then
  for m in "${MODELS[@]}"; do
    if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
      run "TRAIN [${m}] on train_all"         llamafactory-cli train "${PIPE}/train_${m}.yaml"
    else
      banner "SKIP_TRAIN=1: skipping train for ${m}"
    fi

    if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
      run "EVAL  [${m}] on test_eval"         llamafactory-cli train "${PIPE}/eval_${m}.yaml"
    else
      banner "SKIP_EVAL=1: skipping eval for ${m}"
    fi
  done
fi

run "METRICS -> ${RESULT_ROOT}/summary_result.txt"   python "${PIPE}/compute_baseline_sweep_metrics.py"     --root "${RESULT_ROOT}"     --slugs "${MODELS[@]}"

banner "Done. Final report: ${RESULT_ROOT}/summary_result.txt"
