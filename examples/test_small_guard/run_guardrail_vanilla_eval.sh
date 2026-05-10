#!/usr/bin/env bash
# Zero-shot (vanilla) guardrail eval on HuggingFace base checkpoints.
# Spelling: vanilla (two l's), not "vanila".
# Eval splits: test_eval, ood_dataset_eval, ood_category_eval only (train_all is for training, not scored here).
#
# Usage:
#   bash examples/test_small_guard/run_guardrail_vanilla_eval.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash ...            # 本机只有 4 卡时保持默认
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ...   # 8 卡机器上再设满
#   RUN_ONLY=qwen3_guard_gen_0_6b bash ...   # single slug
#
# Gated Meta Llama models need: huggingface-cli login  OR  export HF_TOKEN=...

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

VANILLA_DIR="examples/test_small_guard/guardrail_vanilla"
ALL_SLUGS=(
  # qwen3_guard_gen_8b
  # qwen3_guard_gen_4b
  # qwen3_guard_gen_0_6b
  # llama_guard_3_8b
  # llama_guard_3_1b
  # meta_llama_guard_2_8b
  llama_guard_4_12b

)

if [[ -n "${RUN_ONLY:-}" ]]; then
  SLUGS=("${RUN_ONLY}")
else
  SLUGS=("${ALL_SLUGS[@]}")
fi

for slug in "${SLUGS[@]}"; do
  cfg="${VANILLA_DIR}/${slug}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "SKIP missing ${cfg}" >&2
    continue
  fi
  echo "========================================"
  echo "$(date -u)  ${cfg}"
  llamafactory-cli train "${cfg}"
done

echo "$(date -u) Done. Outputs under results/guardrail_vanilla/<slug>/"
