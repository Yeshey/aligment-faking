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

set -e

# Note that jobs ran here have a 8h timeout limit

SCR=/scratch.hpc/joaofilipe.silvade
cd $SCR/alignment-faking

# Env (mirrors bootstrap.sh)
export HOME=$SCR
export NP_LOCATION=$SCR/.nix-portable
export NP_RUNTIME=proot
export XDG_CACHE_HOME=$SCR/.cache
export XDG_CONFIG_HOME=$SCR/.config
export XDG_STATE_HOME=$SCR/.local/state
export TMPDIR=$SCR/tmp
export HF_HOME=$SCR/.cache/huggingface
export NIX_USER_PROFILE_DIR=$SCR/.local/state/nix/profiles

# Use HM-managed nix if available, else nix-portable
if [[ -f $SCR/.nix-profile/bin/nix ]]; then
  export PATH="$SCR/.nix-profile/bin:$PATH"
  NIX="nix"
else
  NIX="$SCR/.local/bin/nix-portable nix"
fi

mkdir -p logs

# Enter devshell and run — nix develop runs shellHook then exits unless we pass -c
$SCR/.local/bin/nix-portable nix develop .#vllm-cuda-l40 --command bash -c '

  # Start vllm server in background
  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    $VLLM_SERVE_ARGS \
    --port 8000 \
    --download-dir $HF_HOME/hub &
  VLLM_PID=$!

  # Wait for server ready
  echo "Waiting for vllm server..."
  until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done
  echo "Server ready"

  python src/gen.py
  python src/eval.py

  kill $VLLM_PID
'
