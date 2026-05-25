#!/usr/bin/env bash

# config.env already sourced by caller; this is a fallback
if [[ -z "${MODEL:-}" ]]; then
  set -a; source "$(git rev-parse --show-toplevel)/config.env"; set +a
fi
export MODEL_NAME="$MODEL"

VLLM_LOG="${VLLM_LOG:-logs/vllm.log}"
VLLM_PID=""

cleanup() {
  echo "Cleanup: killing vllm (PID=${VLLM_PID:-none})"
  [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true
}

# Usage: start_vllm "--dtype bfloat16 --max-model-len 8192 ..."
start_vllm() {
  local extra_args="${1:-}"
  echo "Freeing port ${VLLM_PORT}..."
  fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
  sleep 2
  echo "Starting vllm: $MODEL  (log → $VLLM_LOG)"
  # shellcheck disable=SC2086
  vllm serve "$MODEL" \
    $extra_args \
    --port "$VLLM_PORT" \
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
}

wait_for_vllm() {
  local max_tries="${1:-120}"
  local tries=0
  echo "Waiting for vllm on :${VLLM_PORT} (up to $((max_tries * 5))s)..."
  until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 5
    tries=$((tries + 1))
    if [[ $tries -ge $max_tries ]]; then
      echo "ERROR: vllm not ready — check $VLLM_LOG" >&2; exit 1
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "ERROR: vllm died — check $VLLM_LOG" >&2; exit 1
    fi
  done
  echo "Server ready on :${VLLM_PORT}"
}

run_pipeline() {
  local src_dir="${1:-src}"
  python "${src_dir}/gen.py"
  python "${src_dir}/eval.py"
}