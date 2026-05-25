source main/common.sh
trap cleanup EXIT

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo "GPU detected"
  # override MODEL for local (AWQ fits 6 GB)
  MODEL="casperhansen/llama-3-8b-instruct-awq"
  export MODEL_NAME="$MODEL"
  start_vllm "--quantization awq --dtype float16 --gpu-memory-utilization 0.92 --max-model-len 2048"
else
  echo "No GPU — CPU fallback (slow)"
  start_vllm "--device cpu --dtype float32 --max-model-len 2048"
fi

wait_for_vllm 120
run_pipeline src