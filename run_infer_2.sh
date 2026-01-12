#!/bin/bash
#SBATCH --job-name=qwen2-5-vl-72b-infer
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=18:00:00
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.log

# Create directories
mkdir -p logs

echo "[`date`] Node: $SLURMD_NODENAME"
echo "[`date`] Job ID: $SLURM_JOB_ID"
echo "[`date`] GPUs: $CUDA_VISIBLE_DEVICES"

# Activate conda environment if available
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate dl || true
fi

# Environment variables
export WANDB_DISABLED=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to the repo root
cd /data/wenjie_jacky_mo/Debug_LM

# Ensure local sources are on PYTHONPATH so llamafactory imports resolve
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

# Use LLaMA-Factory's built-in inference instead of vLLM

# llamafactory-cli train examples/inference/qwen.yaml
# llamafactory-cli train examples/inference/llama.yaml

llamafactory-cli train examples/inference/eval_tag_debug_2.yaml

# llamafactory-cli chat examples/inference/chat.yaml

echo "[`date`] Finished."
