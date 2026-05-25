#!/usr/bin/env bash
# Local run — RTX 2060 6 GB
# Uses 4-bit AWQ model to fit VRAM; falls back to CPU if GPU unavailable.
#
# Deps: uv, curl, nvidia-smi (optional)
# HF token: export HUGGING_FACE_HUB_TOKEN or put in .hf_token
#
# Ollama alternative (simpler, handles memory itself):
#   ollama pull llama3          # pulls llama-3-8b-q4
#   VLLM_PORT=11434 ollama serve &
#   # set CLIENT_URL="http://localhost:11434/v1" in src/gen.py + src/eval.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ -f .hf_token ]] && source .hf_token

mkdir -p logs data evaluated_data

uv sync --frozen
source .venv/bin/activate

source main/common.sh

# 4-bit AWQ — Llama-3-8B weights ~4.5 GB, leaves room for KV cache at 2048 ctx
MODEL="casperhansen/llama-3-8b-instruct-awq"

# Auto-detect GPU
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo "GPU detected — running AWQ 4-bit on GPU"
  DEVICE_ARGS="--quantization awq --dtype float16 --gpu-memory-utilization 0.92 --max-model-len 2048"
else
  echo "No GPU — falling back to CPU (slow, expect ~10 min/query)"
  # CPU mode: use original fp16 model, no quantization flag needed
  MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
  DEVICE_ARGS="--device cpu --dtype float32 --max-model-len 2048"
fi

VLLM_PID=""
trap cleanup EXIT

vllm serve "$MODEL" \
  $DEVICE_ARGS \
  --port "$VLLM_PORT" &
VLLM_PID=$!

wait_for_vllm 120   # CPU load can take several minutes
run_pipeline src
