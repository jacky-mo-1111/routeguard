#!/usr/bin/env bash
# Gemma 3 1B IT: train (lineage SFT) then eval (MCQ + tag-debug stop), using LLaMA-Factory.
#
# References:
#   - examples/train_full/llama3_full_sft.yaml  (dataset mix)
#   - examples/inference/eval_tag_debug_stop_2.yaml  (eval + tag_debug_force_eos_after_candidate)
#
# Requires:
#   - `llamafactory-cli` on PATH (this repo)
#   - `eval_dataset` keys present in ${DATASET_DIR}/dataset_info.json
#   - `transformers` with Gemma 3 support (`google/gemma-3-1b-it`)
#
# Usage:
#   bash examples/pipeline/run_gemma4_e2b_train_eval.sh
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/pipeline/run_gemma4_e2b_train_eval.sh
#   MODEL_NAME=google/gemma-3-1b-it bash examples/pipeline/run_gemma4_e2b_train_eval.sh
#   SKIP_TRAIN=1 bash examples/pipeline/run_gemma4_e2b_train_eval.sh
#   EVAL_DATASET=tofu_dev_lineage EVAL_OUTPUT=results/gemma3_1b_lineage_tofu bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TRAIN_YAML="${TRAIN_YAML:-examples/train_full/gemma4_e2b_sft.yaml}"
EVAL_YAML="${EVAL_YAML:-examples/inference/eval_gemma4_e2b_tag_debug_stop_2.yaml}"

MODEL_NAME="${MODEL_NAME:-google/gemma-3-1b-it}"

TRAIN_OUTPUT="${TRAIN_OUTPUT:-${REPO_ROOT}/saves/gemma3_1b_debuglm}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${REPO_ROOT}/results/gemma3_1b_lineage_dev}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data}"
EVAL_DATASET="${EVAL_DATASET:-tofu_dev_lineage,chatdoctor_dev_lineage,bever_dev_lineage,tqa_dev_lineage,wmdp_dev_lineage}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"

banner() {
  echo ""
  echo "================================================================================"
  echo "$*"
  echo "================================================================================"
}

banner "REPO_ROOT=${REPO_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "TRAIN_OUTPUT=${TRAIN_OUTPUT}"
echo "EVAL_OUTPUT=${EVAL_OUTPUT}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "EVAL_DATASET=${EVAL_DATASET}"

if [[ ! -f "${DATASET_DIR}/dataset_info.json" ]]; then
  echo "ERROR: missing ${DATASET_DIR}/dataset_info.json" >&2
  exit 1
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  banner "TRAIN"
  llamafactory-cli train "${TRAIN_YAML}" \
    model_name_or_path="${MODEL_NAME}" \
    output_dir="${TRAIN_OUTPUT}" \
    dataset_dir="${DATASET_DIR}"
else
  banner "SKIP_TRAIN=1 — using existing checkpoint at ${TRAIN_OUTPUT}"
fi

if [[ ! -d "${TRAIN_OUTPUT}" ]]; then
  echo "ERROR: train output not found: ${TRAIN_OUTPUT}" >&2
  exit 1
fi

banner "EVAL (predict)"
llamafactory-cli train "${EVAL_YAML}" \
  model_name_or_path="${TRAIN_OUTPUT}" \
  output_dir="${EVAL_OUTPUT}" \
  dataset_dir="${DATASET_DIR}" \
  eval_dataset="${EVAL_DATASET}"

banner "Done. Predictions: ${EVAL_OUTPUT}/generated_predictions.jsonl"
