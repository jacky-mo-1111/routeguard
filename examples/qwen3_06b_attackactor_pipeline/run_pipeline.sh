#!/usr/bin/env bash
# Pipeline: train + eval qwen3_guard_gen_0.6b under three configs and write result.txt
#   [1]   combined : train=train_all + actorattack_guardrail_train , eval=test_eval + actorattack_guardrail_eval
#   [2.1] separate : train=train_all                               , eval=test_eval
#   [2.2] separate : train=actorattack_guardrail_train             , eval=actorattack_guardrail_eval
#
# Why sequential on all 8 GPUs (instead of 4+4 parallel)?
#   With ZeRO-3 data-parallel, total throughput scales ~linearly with N_GPU.
#   Sum of work is fixed, so 1 job on 8 GPUs ≈ 2 jobs of 4 GPUs each in wall time,
#   but sequential keeps all 8 GPUs at 100% utilization end-to-end (no idle when
#   one parallel branch finishes earlier) and is far simpler / more robust.
#
# Usage:
#   bash examples/qwen3_06b_attackactor_pipeline/run_pipeline.sh
#   SKIP_TRAIN=1 bash ...        # only eval + metrics (expects ckpts already there)
#   SKIP_EVAL=1  bash ...        # only train
#   ONLY_METRICS=1 bash ...      # just rebuild result.txt from existing predictions
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash ...   # restrict GPUs

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

PIPE="examples/qwen3_06b_attackactor_pipeline"
RESULT_ROOT="/nas02/jacky/Debug_LM/results/qwen3_06b_attackactor"
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
    # Trainings — sequential on all 8 GPUs. Largest first so the GPUs are warm.
    run "TRAIN [1] combined  (train_all + actorattack_guardrail_train)" \
      llamafactory-cli train "${PIPE}/train_combined.yaml"
    run "TRAIN [2.1] train_all only" \
      llamafactory-cli train "${PIPE}/train_train_all.yaml"
    run "TRAIN [2.2] actorattack_guardrail_train only" \
      llamafactory-cli train "${PIPE}/train_attackactor.yaml"
  else
    banner "SKIP_TRAIN=1: skipping all 3 training stages"
  fi

  if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    run "EVAL  [1] combined ckpt  → test_eval + actorattack_guardrail_eval" \
      llamafactory-cli train "${PIPE}/eval_combined.yaml"
    run "EVAL  [2.1] train_all ckpt → test_eval" \
      llamafactory-cli train "${PIPE}/eval_train_all.yaml"
    run "EVAL  [2.2] attackactor ckpt → actorattack_guardrail_eval" \
      llamafactory-cli train "${PIPE}/eval_attackactor.yaml"
  else
    banner "SKIP_EVAL=1: skipping all 3 evaluation stages"
  fi
fi

run "METRICS  →  ${RESULT_ROOT}/result.txt" \
  python "${PIPE}/compute_metrics.py" --root "${RESULT_ROOT}"

banner "Done. Final report: ${RESULT_ROOT}/result.txt"
