#!/usr/bin/env bash
# Qwen3-30B-A3B vLLM serve launcher (HPC).
#
# Override defaults via env:
#   VLLM_PORT          — default 8000
#   VLLM_TP_SIZE       — default = number of GPUs visible to CUDA
#   VLLM_MAX_MODEL_LEN — default 65536 (uses YaRN over native 32k)
#   VLLM_HOST          — default 0.0.0.0
#   VLLM_API_KEY       — default EMPTY (vLLM-OpenAI compat)
#   VLLM_DISABLE_YARN  — set non-empty to keep native 32k context
#
# Hybrid thinking: Qwen3 supports enable_thinking via chat-template
# kwargs, set per-request by OpenAIBackend(model_family="qwen3").
# No serve-time flag is required for hybrid thinking on Qwen3.
#
# Long-context: AIME debate at 3 rounds accumulates ~25-40k tokens of
# history per agent's round-3 prompt. We default to 65536 with YaRN
# rope scaling to keep AIME from blowing the context window.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
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

# YaRN rope scaling for >32k contexts (Qwen3 native max is 32768).
# Skip if user opts out or asks for native length.
if [[ -z "${VLLM_DISABLE_YARN:-}" ]] && [[ "$MAX_MODEL_LEN" -gt 32768 ]]; then
    YARN_FACTOR=$(awk "BEGIN {print $MAX_MODEL_LEN / 32768}")
    cmd+=(
        --rope-scaling '{"rope_type":"yarn","factor":'"$YARN_FACTOR"',"original_max_position_embeddings":32768}'
    )
fi

echo "[serve] launching: ${cmd[*]}"
exec "${cmd[@]}"
