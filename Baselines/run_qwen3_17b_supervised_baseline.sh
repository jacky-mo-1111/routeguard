#!/usr/bin/env bash
set -euo pipefail
#
# Train + eval Qwen3-1.7B supervised data-attribution baseline (MCQ logits → domain label).
#
# Requires: transformers, peft, torch (same env you use for other baselines).
#
# Outputs:
#   Baselines/saves/qwen3_17b_llm_classifier/        (LoRA adapter)
#   Baselines/qwen3_17b_supervised/qwen3_17b_supervised_results.{txt,json}
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export ROOT

exec bash "${SCRIPT_DIR}/run_qwen3_06b_supervised_baseline.sh" "$@"
