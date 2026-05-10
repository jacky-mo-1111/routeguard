#!/usr/bin/env bash
# Benchmark sequential generation latency on TOFU dev splits.
#
# Default models (Hugging Face; same max_new_tokens / device / N for fair timing):
#   jackysnake/llama_debug_lm
#   jackysnake/qwen_debug_lm
#   meta-llama/Llama-3.2-1B-Instruct  (open baseline; gated — need HF_TOKEN / huggingface-cli login)
#
# Default datasets (paths from data/dataset_info.json):
#   tofu_dev_lineage
#   tofu_dev
#
# Compare: tofu_dev_lineage vs tofu_dev × {debug LMs, Llama-3.2-1B-Instruct}.
#
# Metrics per (model × dataset): wall_total_sec, mean_latency_sec, samples_per_sec
#
# Usage:
#   bash scripts/bench_hf_debug_lm_latency.sh
#
# Overrides:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/bench_hf_debug_lm_latency.sh
#   NUM_SAMPLES=100 MAX_NEW_TOKENS=256 bash scripts/bench_hf_debug_lm_latency.sh
#   RANDOM_SAMPLE=1 bash scripts/bench_hf_debug_lm_latency.sh
#   # base 1B instead of Instruct:
#   MODELS="jackysnake/llama_debug_lm jackysnake/qwen_debug_lm meta-llama/Llama-3.2-1B" bash scripts/bench_hf_debug_lm_latency.sh
#   DATASETS="tofu_dev_lineage tofu_dev" MODELS="..." bash scripts/bench_hf_debug_lm_latency.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASET_INFO="${DATASET_INFO:-${ROOT}/data/dataset_info.json}"
PY="${SCRIPT_DIR}/_bench_hf_debug_lm_latency.py"
OUT_JSON="${OUT_JSON:-${ROOT}/results/debug_lm_latency/benchmark_tofu_dev_latency_$(date +%Y%m%d_%H%M%S).json}"

# Space-separated lists (no spaces inside IDs)
MODELS="${MODELS:-jackysnake/llama_debug_lm jackysnake/qwen_debug_lm meta-llama/Llama-3.2-1B-Instruct}"
DATASETS="${DATASETS:-tofu_dev_lineage tofu_dev}"
read -r -a MODELS_ARR <<< "${MODELS}"
read -r -a DATASETS_ARR <<< "${DATASETS}"

NUM_SAMPLES="${NUM_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

EXTRA=()
if [[ "${RANDOM_SAMPLE:-0}" == "1" ]]; then
  EXTRA+=(--random_sample)
fi

mkdir -p "$(dirname "${OUT_JSON}")"

echo "ROOT=${ROOT}"
echo "DATASET_INFO=${DATASET_INFO}"
echo "NUM_SAMPLES=${NUM_SAMPLES} MAX_NEW_TOKENS=${MAX_NEW_TOKENS} SEED=${SEED} DEVICE=${DEVICE} DTYPE=${DTYPE}"
echo "MODELS=${MODELS}"
echo "DATASETS=${DATASETS}"
echo "OUT_JSON=${OUT_JSON}"
echo

python "${PY}" \
  --dataset_info "${DATASET_INFO}" \
  --datasets "${DATASETS_ARR[@]}" \
  --models "${MODELS_ARR[@]}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --out_json "${OUT_JSON}" \
  "${EXTRA[@]}"
