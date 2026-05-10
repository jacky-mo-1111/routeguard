#!/bin/bash
# Re-run Qwen/Qwen3-8B (base) on MMLU + ARC with max_new_tokens=4096

source /home/jackymo/anaconda3/etc/profile.d/conda.sh
conda activate dl
echo "Active env: $CONDA_DEFAULT_ENV"

export WANDB_DISABLED=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FORCE_TORCHRUN=0
export CUDA_VISIBLE_DEVICES=4,5,6,7

cd /nas02/jacky/Debug_LM

echo ""
echo "===== [$(date)] qwen_base / mmlu ====="
llamafactory-cli train examples/inference/quick/eval_mmlu_qwen_base.yaml

echo ""
echo "===== [$(date)] qwen_base / arc_challenge ====="
llamafactory-cli train examples/inference/quick/eval_arc_challenge_qwen_base.yaml

echo ""
echo "===== [$(date)] Computing Accuracy ====="
python eval_benchmarks.py \
  --result_dirs results/qwen_base_mmlu_quick results/qwen_base_arc_challenge_quick \
  --output results/qwen_base_rerun.txt

echo "[$(date)] Done. See results/qwen_base_rerun.txt"
