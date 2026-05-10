#!/usr/bin/env bash
# Full optimal-router pipeline.
# 1) optionally precompute local expert predictions on full train_all (aligned)
# 2) build optimal router train/test labels by enumerating expert subsets
# 3) train/eval Qwen3-0.6B router on 8 GPUs
# 4) evaluate router + existing test expert predictions
#
# Usage:
#   bash examples/qwen3_06b_routeguard_router_optimal_pipeline/run_pipeline.sh
#   SKIP_EXPERT_TRAIN_ALL=1 bash ...   # if full train_all expert predictions already exist
#   SKIP_TRAIN=1 bash ...              # skip router training
#   SKIP_EVAL=1 bash ...               # skip router eval
#   ONLY_METRICS=1 bash ...            # only score existing router predictions
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
PIPE="examples/qwen3_06b_routeguard_router_optimal_pipeline"
RESULT_ROOT="/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_router_optimal"
DATA_DIR="/nas02/jacky/data/route_guard_final/category_label"
EXPERT_TRAIN_ROOT="/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_local_expert_train_all"
EXPERT_TEST_ROOT="/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard_local_expert"
BASELINE_PRED="/nas02/jacky/Debug_LM/results/qwen3_06b_routeguard/baseline/test_eval_category_label/generated_predictions.jsonl"
mkdir -p "${RESULT_ROOT}" logs

ts(){ date -u +"%Y-%m-%dT%H:%M:%SZ"; }
run(){ echo; echo "================================ $(ts)  $1 ================================"; shift; "$@"; }

if [[ "${ONLY_METRICS:-0}" != "1" ]]; then
  if [[ "${SKIP_EXPERT_TRAIN_ALL:-0}" != "1" ]]; then
    run "EXPERT PREDICT full train_all (parallel 5 GPUs)" \
      env GPUS="${EXPERT_GPUS:-0,1,2,3,4}" bash "examples/qwen3_06b_routeguard_local_expert_pipeline/run_train_all_predictions.sh"
  fi

  run "BUILD optimal router train/test data" \
    python "${DATA_DIR}/router_optimal/build_optimal_router_dataset.py" \
      --data-dir "${DATA_DIR}" \
      --train-expert-root "${EXPERT_TRAIN_ROOT}" \
      --test-expert-root "${EXPERT_TEST_ROOT}" \
      --alpha 2.0 --beta 1.0

  python - <<'PYINNER'
import json
from pathlib import Path
p=Path('/nas02/jacky/data/route_guard_final/category_label/dataset_info.json')
info=json.loads(p.read_text())
info['routeguard_router_optimal_train']={'file_name':'/nas02/jacky/data/route_guard_final/category_label/router_optimal/train.json'}
info['routeguard_router_optimal_test_eval']={'file_name':'/nas02/jacky/data/route_guard_final/category_label/router_optimal/test_eval.json'}
p.write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n')
PYINNER

  if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    run "TRAIN optimal router" llamafactory-cli train "${PIPE}/train_router.yaml"
  fi
  if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    run "EVAL optimal router" llamafactory-cli train "${PIPE}/eval_router.yaml"
  fi
fi

run "METRICS optimal router + local experts" \
  python "${PIPE}/evaluate_router_routeguard.py" \
    --router-pred "${RESULT_ROOT}/router/generated_predictions.jsonl" \
    --expert-root "${EXPERT_TEST_ROOT}" \
    --baseline-pred "${BASELINE_PRED}" \
    --out-dir "${RESULT_ROOT}"

echo "$(ts) Done: ${RESULT_ROOT}/router_routeguard_result.txt"
