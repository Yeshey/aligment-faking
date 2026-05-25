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
SCR=/scratch.hpc/joaofilipe.silvade
ROOT=$SCR/alignment-faking
cd "$ROOT"

# squeue -u joaofilipe.silvade
# tail -f logs/slurm-JOBID.log logs/slurm-JOBID.err

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
trap cleanup EXIT

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo "GPU detected"
  start_vllm "--dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 8192 --download-dir $HF_HOME/hub"
else
  echo "No GPU — Exit"
  exit
  # echo "CPU fallback (slow)"
  # start_vllm "--device cpu --dtype float32 --max-model-len 2048"
fi

wait_for_vllm 120
run_pipeline src