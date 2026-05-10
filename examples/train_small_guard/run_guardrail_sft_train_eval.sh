#!/usr/bin/env bash
# SFT on train_all (ZeRO-3), then eval test_eval + OOD splits only (no train_all).
#
# Usage:
#   bash examples/train_small_guard/run_guardrail_sft_train_eval.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash ...
#   RUN_ONLY=qwen3_guard_gen_0_6b bash ...
#   SKIP_TRAIN=1 bash ...   # only run eval (expects checkpoint under saves/rg_final/sft_gr/<slug>/)
#
# One slug failing does not stop the rest; the script exits 1 if any slug failed.
#
# Requires HF auth for gated meta-llama/* checkpoints.

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

TRAIN_DIR="examples/train_small_guard/sft_gr"
EVAL_DIR="examples/test_small_guard/sft_gr"

ALL_SLUGS=(
  llama_guard_3_1b
  qwen3_guard_gen_8b
  qwen3_guard_gen_4b
  qwen3_guard_gen_0_6b
  llama_guard_3_8b
  meta_llama_guard_2_8b
  llama_guard_4_12b
)

if [[ -n "${RUN_ONLY:-}" ]]; then
  SLUGS=("${RUN_ONLY}")
else
  SLUGS=("${ALL_SLUGS[@]}")
fi

FAILED_SLUGS=()
for slug in "${SLUGS[@]}"; do
  train_cfg="${TRAIN_DIR}/${slug}.yaml"
  eval_cfg="${EVAL_DIR}/${slug}.yaml"
  if [[ ! -f "${train_cfg}" ]] || [[ ! -f "${eval_cfg}" ]]; then
    echo "SKIP missing train=${train_cfg} eval=${eval_cfg}" >&2
    continue
  fi

  if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    echo "======================================== TRAIN ${slug}"
    echo "$(date -u)  ${train_cfg}"
    if ! llamafactory-cli train "${train_cfg}"; then
      echo "ERROR: TRAIN failed for ${slug}; skipping eval and continuing." >&2
      FAILED_SLUGS+=("${slug}")
      continue
    fi
  else
    echo "======================================== SKIP TRAIN ${slug} (SKIP_TRAIN=1)"
  fi

  echo "======================================== EVAL ${slug}"
  echo "$(date -u)  ${eval_cfg}"
  if ! llamafactory-cli train "${eval_cfg}"; then
    echo "ERROR: EVAL failed for ${slug}; continuing to next slug." >&2
    FAILED_SLUGS+=("${slug}")
  fi
done

echo "$(date -u) Done. Checkpoints: saves/rg_final/sft_gr/<slug>/  predictions: results/rg_final/sft_gr/<slug>/"
if ((${#FAILED_SLUGS[@]} > 0)); then
  echo "$(date -u) Failed slugs: ${FAILED_SLUGS[*]}" >&2
  exit 1
fi
