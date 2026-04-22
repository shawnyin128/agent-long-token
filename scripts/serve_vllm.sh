#!/usr/bin/env bash
# Launch (or stop) a local vLLM server for agentdiet.
#
# Usage:
#   scripts/serve_vllm.sh           # start server in background, write pid
#   scripts/serve_vllm.sh stop      # kill running server via pid file
#   scripts/serve_vllm.sh status    # print pid/port info
#
# Environment overrides (all optional):
#   AGENTDIET_MODEL                default: Qwen/Qwen2.5-7B-Instruct
#   VLLM_PORT                      default: 8000
#   VLLM_DTYPE                     default: bfloat16
#   VLLM_MAX_MODEL_LEN             default: 8192
#   VLLM_GPU_MEM_UTIL              default: 0.9
#   VLLM_LOG                       default: artifacts/logs/vllm.log
#   VLLM_PIDFILE                   default: artifacts/logs/vllm.pid

set -euo pipefail

MODEL="${AGENTDIET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_MEM_UTIL:-0.9}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${VLLM_LOG:-$ROOT/artifacts/logs/vllm.log}"
PIDFILE="${VLLM_PIDFILE:-$ROOT/artifacts/logs/vllm.pid}"
mkdir -p "$(dirname "$LOG")"

cmd="${1:-start}"

case "$cmd" in
  start)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "vLLM already running (pid $(cat "$PIDFILE")) on port $PORT" >&2
      exit 1
    fi
    echo "starting vllm serve $MODEL (dtype=$DTYPE max-len=$MAX_LEN port=$PORT)" >&2
    nohup vllm serve "$MODEL" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --port "$PORT" \
      --served-model-name "$MODEL" \
      > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "pid=$(cat "$PIDFILE") log=$LOG" >&2
    echo "next: scripts/wait_healthy.py --timeout 180" >&2
    ;;
  stop)
    if [[ ! -f "$PIDFILE" ]]; then
      echo "no pid file at $PIDFILE" >&2
      exit 0
    fi
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "sent TERM to pid $pid" >&2
      for _ in 1 2 3 4 5; do
        if kill -0 "$pid" 2>/dev/null; then sleep 1; else break; fi
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" || true
        echo "forced kill -9 pid $pid" >&2
      fi
    else
      echo "no process at pid $pid" >&2
    fi
    rm -f "$PIDFILE"
    ;;
  status)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "running pid=$(cat "$PIDFILE") port=$PORT log=$LOG"
    else
      echo "not running"
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 [start|stop|status]" >&2
    exit 2
    ;;
esac
