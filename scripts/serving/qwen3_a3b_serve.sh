#!/usr/bin/env bash
# Qwen3-30B-A3B vLLM serve launcher (HPC).
#
# Override defaults via env:
#   VLLM_PORT          — default 8000
#   VLLM_TP_SIZE       — default = number of GPUs visible to CUDA
#   VLLM_MAX_MODEL_LEN — default 32768
#   VLLM_HOST          — default 0.0.0.0
#   VLLM_API_KEY       — default EMPTY (vLLM-OpenAI compat)
#
# Hybrid thinking: Qwen3 supports enable_thinking via chat-template
# kwargs, set per-request by OpenAIBackend(model_family="qwen3").
# No serve-time flag is required for hybrid thinking on Qwen3.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
API_KEY="${VLLM_API_KEY:-EMPTY}"

# Auto-detect tensor-parallel size from CUDA_VISIBLE_DEVICES if not set.
if [[ -z "${VLLM_TP_SIZE:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        VLLM_TP_SIZE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    else
        VLLM_TP_SIZE=1
    fi
fi

cmd=(
    vllm serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --tensor-parallel-size "$VLLM_TP_SIZE"
    --api-key "$API_KEY"
    --trust-remote-code
)

echo "[serve] launching: ${cmd[*]}"
exec "${cmd[@]}"
