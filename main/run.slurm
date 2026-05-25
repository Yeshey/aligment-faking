#!/bin/bash
#SBATCH --job-name=alignment-faking
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joaofilipe.silvade@studio.unibo.it

set -euo pipefail

# squeue -u joaofilipe.silvade
# tail -f logs/slurm-JOBID.log logs/slurm-JOBID.err

SCR=/scratch.hpc/joaofilipe.silvade
ROOT=$SCR/alignment-faking
cd "$ROOT"

export HOME=$SCR
export XDG_CACHE_HOME=$SCR/.cache
export XDG_CONFIG_HOME=$SCR/.config
export XDG_STATE_HOME=$SCR/.local/state
export TMPDIR=$SCR/tmp
export HF_HOME=$SCR/.cache/huggingface
export UV_PROJECT_ENVIRONMENT=".venv"
export UV_CACHE_DIR=$SCR/.cache/uv
export PATH="$SCR/.local/bin:$PATH"

source "$ROOT/.hf_token"

mkdir -p logs data evaluated_data "$TMPDIR"

uv sync --frozen
source .venv/bin/activate

source main/common.sh

VLLM_PID=""
trap cleanup EXIT

vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --port "$VLLM_PORT" \
  --download-dir "$HF_HOME/hub" \
  > logs/vllm.log 2>&1 &
VLLM_PID=$!

wait_for_vllm 120
run_pipeline src
