#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a; source "$ROOT/config.env"; set +a   # ← explicit, before common.sh
[[ -f .hf_token ]] && source .hf_token
mkdir -p logs data evaluated_data
uv sync --frozen
source .venv/bin/activate
source main/common.sh
trap cleanup EXIT

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo "GPU detected"
  start_vllm "--quantization bitsandbytes --load-format bitsandbytes --dtype float16 --gpu-memory-utilization 0.92 --max-model-len 2048"
else
  echo "No GPU — CPU fallback (slow)"
  start_vllm "--device cpu --dtype float32 --max-model-len 2048"
fi

wait_for_vllm 120
run_pipeline src