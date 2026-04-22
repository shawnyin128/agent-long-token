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

## Full Collection on HPC

The 100-question collection runs on the NYU HPC (Greene). You submit the
`sbatch`/`srun` job yourself; the instructions below are what to do
**after you have a compute node allocated and a shell on it**.

### 1. Module + environment

```bash
module purge
module load python/3.11 cuda/12.1     # adjust to your Greene module names
cd $SCRATCH/final-project               # or wherever you cloned the repo
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### 2. Start vLLM and wait for readiness

```bash
make serve     # backgrounds vLLM; pid -> artifacts/logs/vllm.pid
make health    # polls /v1/models, waits up to 180s
```

### 3. Run the 100-question collection

```bash
make collect
```

Default `n=100`, `seed=42`, `n_agents=3`, `n_rounds=3`. Per-qid dialogues
land in `artifacts/dialogues/{model_slug}/{qid}.json`; failures in
`artifacts/failures/debate/{qid}.json`; manifest in
`artifacts/dialogues/manifest.json`.

Collection is **resumable**: re-running `make collect` skips qids whose
dialogue file already exists and re-attempts only failed/missing ones.
The LLM cache (`artifacts/llm_cache.jsonl`) makes even a full re-run
cheap if nothing changed.

### 4. Inspect the manifest

```bash
make collect-report
```

Prints counts per outcome:

```
  ok         96
  cached      0
  unparsed    2
  failed      2
```

Target: `ok + cached >= 95 / 100`. If below, re-run `make collect` — it
will only retry the failed/unparsed ones (resume), which is often enough
to clear transient issues.

### 5. Stop vLLM

```bash
make stop
```

### 6. Copy artifacts back (optional)

If you want to inspect locally before running the analysis phase on HPC:

```bash
# from your laptop:
rsync -avz hpc.nyu.edu:$SCRATCH/final-project/artifacts/ ./artifacts/
```

### Troubleshooting

- `make health` timing out — check `artifacts/logs/vllm.log` for OOM,
  port conflicts, or CUDA errors.
- `make collect` reports many `failed` outcomes — inspect
  `artifacts/failures/debate/*.json`; traceback usually points at
  connection refused (health check passed but server died) or context
  overflow (rare on 8k with 3×3 debate).
- Cached LLM calls survive vLLM restarts but are keyed by `model`; if
  you switch model names mid-run, new cache entries are generated and
  old dialogues remain valid under the previous model-slug directory.

## Running tests

```bash
make test          # full suite, no network required
make smoke         # fast smoke tests only
```

Network-gated HF dataset tests require `AGENTDIET_ALLOW_NETWORK=1`.

## Features

See `.claude/features.json` for the in-progress feature list. Current
status: `core-infrastructure`, `debate-runner`, `pilot-gate1` complete.
