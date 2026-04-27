# GPT-OSS-20B on vLLM (HPC) runbook

## What this runbook does

Brings up GPT-OSS-20B as an OpenAI-compatible HTTP endpoint via vLLM,
then runs `smoke_gpt_oss.py` to verify the `reasoning_effort` toggle
reaches the model.

GPT-OSS uses `reasoning_effort=low` vs `=high` per request (no
serve-time flag needed, similar to OpenAI's API).
`OpenAIBackend(model_family="gpt-oss")` translates the project's
uniform `thinking:bool` into `reasoning_effort=high|low`.

## Prereqs

- HPC node with at least 1× A100 80G (gpt-oss-20b fits comfortably
  on one A100; 2× helps prefill on long contexts)
- conda env `course` with the project installed in editable mode
- vLLM ≥ 0.6 with reasoning-parser support for GPT-OSS

## Stand up

In one shell (tmux/screen so it survives logout):

```bash
module load cuda/12.1
conda activate course

# Defaults are fine; override if you want a non-8000 port etc.
export VLLM_PORT=8000
# export VLLM_TP_SIZE=2
# export VLLM_MAX_MODEL_LEN=32768

bash scripts/serving/gpt_oss_20b_serve.sh
```

Wait for `Application startup complete` in the log. The endpoint will
be at `http://<node>:8000/v1`.

## Smoke test

In a second shell:

```bash
conda activate course
export AGENTDIET_BASE_URL=http://localhost:8000/v1

python scripts/serving/smoke_gpt_oss.py
```

Expected output:

```
smoke artifact -> artifacts/serving/gpt_oss_smoke.json
  mean_tokens_on (high)  = 400-1500   (high effort produces long CoT)
  mean_tokens_off (low)  = 100-300
  delta                  = +300-1200
  passed                 = True
```

If `delta` is near zero or negative, the `reasoning_effort` flag is
not reaching the model. Likely causes:

- vLLM version too old to support per-request `reasoning_effort`
- Reasoning parser not configured on the serve side — check vLLM
  release notes for whether GPT-OSS reasoning parsing is built-in
  or needs a `--reasoning-parser` flag

## Commit the artifact

```bash
git add artifacts/serving/gpt_oss_smoke.json
git commit -m "[serving]: gpt-oss smoke artifact"
```

The schema validator at
`tests/cross_model_grid_hpc_serving/test_smoke_artifacts.py` turns
from skip to pass once this file is committed.

## Tear down

```bash
# In the serve shell:
Ctrl-C
```
