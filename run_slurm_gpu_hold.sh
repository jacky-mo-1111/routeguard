#!/bin/bash
# 只占 Slurm 分给你的 GPU，不跑算子；到时间或 scancel 才释放。
# 这样别人在调度器层面抢不到这几张卡；不需要真的 load 模型占显存（除非你另有需求）。
#
# 用法:
#   sbatch run_slurm_gpu_hold.sh
#
# 想「占着卡」同时自己进节点交互用卡（聊天/调试），用交互式更省事，二选一:
#   salloc --partition=cais --nodes=1 --gres=gpu:4 --cpus-per-task=8 --mem=0 --time=12:00:00
#   拿到 shell 后 ssh 到节点（看集群文档）或直接在 salloc 的 shell 里跑你的命令
#
#SBATCH --job-name=gpu_hold
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_gpu_hold.log
#SBATCH --error=logs/%j_gpu_hold.log

set -euo pipefail
mkdir -p logs

echo "[$(date -Is)] job=$SLURM_JOB_ID node=$SLURM_NODELIST gpus=$CUDA_VISIBLE_DEVICES"
echo "Ctrl+C 无效；要停: scancel $SLURM_JOB_ID"

# 每隔一段时间打一行，方便你在 log 里确认还活着
while true; do
  echo "[$(date -Is)] still holding (sleep 5m) ..."
  sleep 300
done
