# Qwen3-30B-A3B on vLLM (HPC) runbook

## What this runbook does

Brings up Qwen3-30B-A3B as an OpenAI-compatible HTTP endpoint via vLLM,
then runs `smoke_qwen3.py` to verify the `enable_thinking` toggle
reaches the model.

## Prereqs

- HPC node with at least 1× A100 80G (Qwen3-30B-A3B is MoE; the
  3B-active footprint fits in one A100, but 2× speeds prefill)
- conda env `course` (or your project env) with the project installed
  in editable mode (`pip install -e .`)
- vLLM ≥ 0.6 (Qwen3 hybrid thinking template support)

## Stand up

In one shell (tmux/screen so it survives logout):

```bash
module load cuda/12.1            # or whatever your cluster uses
conda activate course

# Defaults are fine; override if you want a non-8000 port etc.
export VLLM_PORT=8000
# export VLLM_TP_SIZE=2          # auto-detected from $CUDA_VISIBLE_DEVICES otherwise
# export VLLM_MAX_MODEL_LEN=32768

bash scripts/serving/qwen3_a3b_serve.sh
```

Wait for `Application startup complete` in the log. The endpoint will
be at `http://<node>:8000/v1`.

## Smoke test

In a second shell (same node, or any node that can reach the endpoint):

```bash
conda activate course
export AGENTDIET_BASE_URL=http://localhost:8000/v1
# export AGENTDIET_BASE_URL=http://<node>:8000/v1   # if cross-node

python scripts/serving/smoke_qwen3.py
```

Expected output:

```
smoke artifact -> artifacts/serving/qwen3_smoke.json
  mean_tokens_on  = 350-700  (thinking traces add tokens)
  mean_tokens_off = 80-150
  delta           = +200-500
  passed          = True
```

If `delta` is near zero or negative, the `enable_thinking` toggle is
not reaching the model. Likely causes:

- vLLM version too old to honor `chat_template_kwargs.enable_thinking`
- Qwen3 chat template not picked up — confirm
  `--trust-remote-code` is in the serve flags (it is, by default)

## Commit the artifact

```bash
git add artifacts/serving/qwen3_smoke.json
git commit -m "[serving]: qwen3 smoke artifact"
```

The schema validator at
`tests/cross_model_grid_hpc_serving/test_smoke_artifacts.py` turns
from skip to pass once this file is committed.

## Tear down

```bash
# In the serve shell:
Ctrl-C   # or kill the tmux pane
```

vLLM frees GPU memory on shutdown. Confirm with `nvidia-smi`.
