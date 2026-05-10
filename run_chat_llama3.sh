#!/bin/bash
# Interactive CLI chat with Llama3-8B-Instruct on GPUs 0,1,2,3
# Usage:
#   bash run_chat_llama3.sh            # 直接在当前 shell 运行 (要交互)
#   srun --gres=gpu:4 --pty bash run_chat_llama3.sh   # 通过 slurm 抢到交互式节点再跑
#
# 注意: llamafactory-cli chat 是交互式的, 不要用 sbatch 提交 (stdin 会被关掉).

set -euo pipefail

# Activate conda env if available
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate dl || true
fi

# CUDA toolkit for packages that inspect nvcc at import time (e.g. DeepSpeed).
if [[ -x /usr/local/cuda-12.4/bin/nvcc ]]; then
  export CUDA_HOME=/usr/local/cuda-12.4
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
fi

# 4 GPUs: huggingface backend 会自动 device_map=auto 把层切到这 4 张卡上
export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_DISABLED=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_autotune}"
mkdir -p "$TRITON_CACHE_DIR"

cd /nas02/jacky/Debug_LM

echo "[`date`] Launching interactive chat on GPUs=$CUDA_VISIBLE_DEVICES"
echo "Tips:"
echo "  - 输入 'clear' 清空对话历史"
echo "  - 输入 'exit' 退出"
echo

llamafactory-cli chat examples/inference/chat_llama3_8b.yaml
