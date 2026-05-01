#!/usr/bin/env bash
# GPT-OSS-20B vLLM serve launcher (HPC).
#
# GPT-OSS supports per-request reasoning_effort=low|high via the OpenAI
# chat-completions API; OpenAIBackend(model_family="gpt-oss") sets it
# from the uniform thinking:bool flag.
#
# Override defaults via env:
#   VLLM_PORT          — default 8000
#   VLLM_TP_SIZE       — default = number of GPUs visible to CUDA
#   VLLM_MAX_MODEL_LEN — default 65536 (AIME debate accumulates context)
#   VLLM_HOST          — default 0.0.0.0
#   VLLM_API_KEY       — default EMPTY
set -euo pipefail

MODEL="${MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"  # gpt-oss-20b natively supports 128k
API_KEY="${VLLM_API_KEY:-EMPTY}"

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
