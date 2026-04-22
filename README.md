# agentdiet — Multi-Agent Debate Claim-Level Analysis

Research code for the CSCI3033 LLM Reasoner final project. Spec:
[`docs/design-docs/2026-04-21-multi-agent-debate-claim-analysis-design.md`](docs/design-docs/2026-04-21-multi-agent-debate-claim-analysis-design.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

For analysis features later in the pipeline:

```bash
.venv/bin/pip install -e ".[analysis]"
```

## Gate 1 Runbook

Run these in order. Each step is idempotent; failures at one step do not
invalidate earlier ones thanks to the replay cache (spec §2).

### 1. Start vLLM

```bash
make serve
```

Launches `vllm serve Qwen/Qwen2.5-7B-Instruct` (bf16, 8192 context, port
8000) in the background. pid → `artifacts/logs/vllm.pid`, logs →
`artifacts/logs/vllm.log`.

Override via env: `VLLM_DTYPE`, `VLLM_MAX_MODEL_LEN`, `VLLM_PORT`,
`VLLM_GPU_MEM_UTIL`, `AGENTDIET_MODEL`.

### 2. Wait for readiness

```bash
make health
```

Polls `http://localhost:8000/v1/models` every 2s up to 180s. Exits 0
when the endpoint returns 200.

### 3. Run the pilot

```bash
make pilot
```

Loads 30 GSM8K test questions (seed=42) and runs:
- **single** — one solver agent, round 1 only
- **debate** — 3 agents × 3 rounds

Per-qid artifacts: `artifacts/pilot/{single,debate}/{model_slug}/{qid}.json`.
Manifest: `artifacts/pilot/manifest.json`. Resumable — re-running skips
existing artifacts and uses the LLM cache for free replays.

Partial runs:

```bash
.venv/bin/python -m agentdiet.cli.pilot --no-debate      # only single
.venv/bin/python -m agentdiet.cli.pilot --no-single      # only debate
.venv/bin/python -m agentdiet.cli.pilot --n 10           # subset
```

### 4. Gate 1 decision

```bash
make gate
```

Writes `artifacts/pilot_report.md` with:
- Accuracy for single and debate
- Delta (pp)
- Three sample dialogues (both correct / debate flip / both wrong)
- Verdict

Exit codes:
- `0` — delta ≥ 3pp → PASS, proceed to next feature
- `10` — 0 ≤ delta < 3pp → SOFT FAIL, switch model and re-pilot
- `20` — delta < 0 or insufficient parsed answers → HARD FAIL, inspect

### 5. If Gate 1 fails

Switch to an API model and re-pilot:

```bash
make stop
export AGENTDIET_MODEL=gpt-4o-mini
export AGENTDIET_BASE_URL=https://api.openai.com/v1
export AGENTDIET_API_KEY=sk-...
# skip `make serve` / `make health` since API is already up
make pilot
make gate
```

The cache partitions artifacts by model name, so Qwen and GPT-4o-mini
runs coexist without conflict.

### One-shot

```bash
make pilot-full    # serve -> health -> pilot -> gate
```

### Clean up

```bash
make stop          # stop vLLM
make pilot-clean   # remove pilot artifacts (keeps LLM cache)
```

## Running tests

```bash
make test          # full suite, no network required
make smoke         # fast smoke tests only
```

Network-gated HF dataset tests require `AGENTDIET_ALLOW_NETWORK=1`.

## Features

See `.claude/features.json` for the in-progress feature list. Current
status: `core-infrastructure`, `debate-runner`, `pilot-gate1` complete.
