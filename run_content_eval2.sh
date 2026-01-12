#!/bin/bash
#SBATCH --job-name=llama3-eval
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/%j.eval.log
#SBATCH --error=logs/%j.eval.log

mkdir -p logs
mkdir -p /data/wenjie_jacky_mo/LLaMA-Factory/results

echo "[`date`] Node: $SLURMD_NODENAME"
echo "[`date`] Job ID: $SLURM_JOB_ID"
echo "[`date`] GPUs: $CUDA_VISIBLE_DEVICES"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate dl || true
fi

export WANDB_DISABLED=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/wenjie_jacky_mo/Debug_LM


# llamafactory-cli train examples/inference/finegrained.yaml




# Evaluate multiple datasets sequentially
DATA_DIRS=(
"/data/wenjie_jacky_mo/Debug_LM/results/qwen_debug_quote_sum/quote_sum_eval"

)

for DATA_DIR in "${DATA_DIRS[@]}"; do
  echo "[`date`] Evaluating: $DATA_DIR"
  python /data/wenjie_jacky_mo/Debug_LM/eval_judge.py \
    --data-dir "$DATA_DIR" \
    --batch-size 8 \
    --max-new-tokens 20 \
    --judge-model /data/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
done

echo "[`date`] Finished."


