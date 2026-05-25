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

source main/common.sh
trap cleanup EXIT

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo "GPU detected"
  # override MODEL for local (AWQ fits 6 GB)
  MODEL="casperhansen/llama-3-8b-instruct-awq"
  export MODEL_NAME="$MODEL"
  start_vllm "--quantization awq --dtype float16 --gpu-memory-utilization 0.92 --max-model-len 2048"
else
  echo "No GPU — Exit"
  exit
  # echo "CPU fallback (slow)"
  # start_vllm "--device cpu --dtype float32 --max-model-len 2048"
fi

wait_for_vllm 120
run_pipeline src