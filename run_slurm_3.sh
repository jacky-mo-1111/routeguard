#!/bin/bash
#SBATCH --job-name=emergent_value
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
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
export FORCE_TORCHRUN=1

# Change to the LLaMA-Factory directory
cd /nas02/jacky/Debug_LM

export CUDA_VISIBLE_DEVICES=0,1,2,3




llamafactory-cli train examples/train_small_guard/router_variety/router_train_v1_new.yaml
llamafactory-cli train examples/test_small_guard/router_variety/router_train_v1_new.yaml

llamafactory-cli train examples/train_small_guard/router_variety/router_train_v2_new.yaml
llamafactory-cli train examples/test_small_guard/router_variety/router_train_v2_new.yaml




echo "[`date`] Finished."




