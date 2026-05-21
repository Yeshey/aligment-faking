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

# Know that one job running non stop for 8h gets killed
# commands
# squeue -u joaofilipe.silvade
# tail -f logs/slurm-JOBID.log logs/slurm-JOBID.err

SCR=/scratch.hpc/joaofilipe.silvade
cd $SCR/alignment-faking

export HOME=$SCR
export XDG_CACHE_HOME=$SCR/.cache
export XDG_CONFIG_HOME=$SCR/.config
export XDG_STATE_HOME=$SCR/.local/state
export TMPDIR=$SCR/tmp
export HF_HOME=$SCR/.cache/huggingface
export UV_PROJECT_ENVIRONMENT=".venv"
export UV_CACHE_DIR=$SCR/.cache/uv

export PATH="$SCR/.local/bin:$PATH"   # uv installed here

source $SCR/.hf_token

mkdir -p logs data "$TMPDIR"

# Sync locked env (fast no-op if already up to date)
uv sync --frozen

source .venv/bin/activate

# L40 serve args
VLLM_SERVE_ARGS="--dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 16384"

VLLM_PID=""
cleanup() {
  echo "Cleanup: killing vllm (PID=${VLLM_PID:-none})"
  [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  $VLLM_SERVE_ARGS \
  --port 8000 \
  --download-dir "$HF_HOME/hub" &
VLLM_PID=$!

echo "Waiting for vllm server (up to 10 min)..."
TRIES=0; MAX_TRIES=120
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
  sleep 5
  TRIES=$((TRIES + 1))
  if [[ $TRIES -ge $MAX_TRIES ]]; then
    echo "ERROR: vllm not ready after $((MAX_TRIES * 5))s" >&2; exit 1
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vllm process died" >&2; exit 1
  fi
done
echo "Server ready"

python src/gen.py
python src/eval.py
