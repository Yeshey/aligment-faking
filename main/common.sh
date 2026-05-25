#!/usr/bin/env bash
# Shared helpers — sourced by run.slurm and run.sh
# Assumes caller has set: VLLM_PID (empty string), VLLM_PORT (default 8000)

VLLM_PORT="${VLLM_PORT:-8000}"

cleanup() {
  echo "Cleanup: killing vllm (PID=${VLLM_PID:-none})"
  [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true
}

wait_for_vllm() {
  local max_tries="${1:-120}"
  local tries=0
  echo "Waiting for vllm on :${VLLM_PORT} (up to $((max_tries * 5))s)..."
  until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 5
    tries=$((tries + 1))
    if [[ $tries -ge $max_tries ]]; then
      echo "ERROR: vllm not ready after $((max_tries * 5))s" >&2; exit 1
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "ERROR: vllm process died" >&2; exit 1
    fi
  done
  echo "Server ready on :${VLLM_PORT}"
}

run_pipeline() {
  local src_dir="${1:-src}"
  python "${src_dir}/gen.py"
  python "${src_dir}/eval.py"
}
