#!/usr/bin/env bash
# Pipeline: train + eval the RouteGuard expert family vs. a label-only SFT baseline,
# then score every (config, split) with both binary safety detection metrics
# and multi-label unsafe-category metrics.
#
# Configurations
#   Baseline (label-only SFT)
#     train: train_all_category_label
#     eval : test_eval_category_label,
#            ood_category_eval_category_label,
#            ood_dataset_eval_category_label
#
#   RouteGuard experts (one full FT each)
#     agent       : train=agent_category_label
#     cyber       : train=cyber_category_label
#     harm        : train=harm_category_label
#     non_violent : train=non_violent_category_label
#     social      : train=social_category_label
#     eval (each) : test_eval_category_label,
#                   ood_category_eval_category_label,
#                   ood_dataset_eval_category_label,
#                   train_all_category_label
#
# All training stages run sequentially on the visible GPU set so DeepSpeed ZeRO-3
# can use every GPU for one job at a time (highest end-to-end utilization).
#
# Usage:
#   bash examples/qwen3_06b_routeguard_pipeline/run_pipeline.sh
#   SKIP_TRAIN=1     bash ...   # only eval + metrics (expects existing ckpts)
#   SKIP_EVAL=1      bash ...   # only train
#   ONLY_METRICS=1   bash ...   # just rebuild result.txt from existing predictions
#   SKIP_BASELINE=1  bash ...   # skip baseline train+eval
#   SKIP_EXPERTS=1   bash ...   # skip all expert train+eval
#   ONLY_EXPERTS=...           # comma list (e.g. "agent,cyber") to limit experts
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PIPE="examples/qwen3_06b_routeguard_pipeline"
RESULT_ROOT="/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard"
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

# Optional filter via ONLY_EXPERTS=agent,cyber
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

if [[ "${ONLY_METRICS:-0}" != "1" ]]; then
  if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    if [[ "${SKIP_BASELINE:-0}" != "1" ]]; then
      run "TRAIN [baseline] train_all_category_label" \
        llamafactory-cli train "${PIPE}/train_baseline.yaml"
    fi
    if [[ "${SKIP_EXPERTS:-0}" != "1" ]]; then
      for e in "${EXPERTS[@]}"; do
        run "TRAIN [routeguard/${e}] ${e}_category_label" \
          llamafactory-cli train "${PIPE}/train_${e}.yaml"
      done
    fi
  else
    banner "SKIP_TRAIN=1: skipping all training stages"
  fi

  if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    if [[ "${SKIP_BASELINE:-0}" != "1" ]]; then
      run "EVAL  [baseline] -> 3 eval splits" \
        llamafactory-cli train "${PIPE}/eval_baseline.yaml"
    fi
    if [[ "${SKIP_EXPERTS:-0}" != "1" ]]; then
      for e in "${EXPERTS[@]}"; do
        run "EVAL  [routeguard/${e}] -> 4 splits" \
          llamafactory-cli train "${PIPE}/eval_${e}.yaml"
      done
    fi
  else
    banner "SKIP_EVAL=1: skipping all eval stages"
  fi
fi

run "METRICS  ->  ${RESULT_ROOT}/result.txt" \
  python "${PIPE}/compute_metrics.py" --root "${RESULT_ROOT}"

banner "Done. Final report: ${RESULT_ROOT}/result.txt"
