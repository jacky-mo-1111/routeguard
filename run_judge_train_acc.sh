#!/usr/bin/env bash
set -euo pipefail

ROOT="/nas02/jacky/Debug_LM"
PY="${ROOT}/examples/test_small_guard/compute_guard_confidence.py"

# 只用两张卡，避免和 0–1 上 vLLM 冲突；按你机器情况可改成别的卡
export CUDA_VISIBLE_DEVICES=2,3
# confidence 前向 batch；显存不够就调小（如 32）
BATCH_SIZE="${BATCH_SIZE:-32}"
# 断点续跑：RESUME=1 且同目录已有 generated_predictions_conf.jsonl 时，按已写行数跳过输入并追加
RESUME="${RESUME:-1}"

source /home/jackymo/anaconda3/etc/profile.d/conda.sh
conda activate dl

cd "${ROOT}"

# 五个 train_all：目录 -> 对应 checkpoint
declare -A PAIRS=(
  ["${ROOT}/results/small_guard_eval/rg_33k/crime_machine_agree_dev/train_all/generated_predictions.jsonl"]="${ROOT}/saves/routeguard/rg_33k/crime"
  ["${ROOT}/results/small_guard_eval/rg_33k/agent_machine_agree_dev/train_all/generated_predictions.jsonl"]="${ROOT}/saves/routeguard/rg_33k/agent"
  ["${ROOT}/results/small_guard_eval/rg_33k/info_machine_agree_dev/train_all/generated_predictions.jsonl"]="${ROOT}/saves/routeguard/rg_33k/info"
  ["${ROOT}/results/small_guard_eval/rg_33k/violence_machine_agree_dev/train_all/generated_predictions.jsonl"]="${ROOT}/saves/routeguard/rg_33k/violence"
  ["${ROOT}/results/small_guard_eval/rg_33k/hate_machine_agree_dev/train_all/generated_predictions.jsonl"]="${ROOT}/saves/routeguard/rg_33k/hate"
)

for jsonl in "${!PAIRS[@]}"; do
  model="${PAIRS[$jsonl]}"
  echo "========================================"
  echo "$(date)  model=${model}"
  echo "  input: ${jsonl}"
  extra=()
  if [ "${RESUME}" = "1" ]; then extra+=(--resume); fi
  python "${PY}" --input-jsonl "${jsonl}" --model "${model}" --output-suffix _conf --force \
    --batch-size "${BATCH_SIZE}" "${extra[@]}"
done

echo "$(date) All five done. Outputs: generated_predictions_conf.jsonl next to each generated_predictions.jsonl"