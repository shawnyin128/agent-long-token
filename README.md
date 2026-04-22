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

## Claim extraction

Once dialogues exist under `artifacts/dialogues/{model_slug}/`, run claim
extraction on the same node (it still needs the vLLM server). The
pipeline is resumable per-qid and the LLM cache makes reruns cheap.

### 1. Extract claims

```bash
make extract
```

Default reads every `*.json` in `artifacts/dialogues/{model_slug}/` and
writes a per-qid artifact to `artifacts/claims/{model_slug}/{qid}.json`
with fields `qid`, `claims[]` (6-type taxonomy per spec §3.4),
`per_message_status[]`, and `extraction_failed`. Failures land in
`artifacts/failures/claim_extraction/`. Manifest:
`artifacts/claims/manifest.json`.

### 2. Inspect the manifest

```bash
make extract-report
```

Prints counts per outcome:

```
  ok         94
  partial     5     # at least one message failed extraction
  cached      0
  failed      1     # whole-dialogue failure (IO or parse error)
```

Target: `ok + partial >= 95 / 100`. `partial` is tolerated — those
dialogues still have usable claims; the failed messages are skipped by
downstream analysis via `extraction_failed` flags.

### 3. 10-dialogue spot check (spec §9.3)

```bash
make spot-check
```

Samples 10 dialogues (seeded with `AGENTDIET_SEED`, default 42) and
writes:

- `artifacts/spot_check.csv` — one row per claim with blank
  `manual_pass` and `notes` columns for you to fill
- `artifacts/spot_check_notes.md` — companion with each dialogue
  printed next to its claims for faster reading

Open the markdown, skim each dialogue, then mark yes/no in the CSV.
Target: ≥ 70 % manual_pass rate — that meets spec §9.3's "not trivially
wrong" bar and unblocks Day 2 attribution work.

### Clean up

```bash
make extract-clean    # drops artifacts/claims/ (keeps cache + dialogues)
```

## Analysis: flip points + per-claim signals

Once claim artifacts exist, compute flip events and four independent
per-claim signals. Install the optional analysis extras first:

```bash
.venv/bin/pip install -e ".[analysis]"
```

This pulls sentence-transformers, scikit-learn, hdbscan, pyarrow. Then:

```bash
make analyze
```

Reads dialogues + claims from the paths established by the earlier
steps and writes to `artifacts/analysis/`:

- `flip_events.jsonl` — one JSON object per line with `qid`, `round`,
  `triggering_claim_id`, `pre_flip_answers`, `post_flip_answers`.
- `signal_scores.parquet` — 6 columns: `qid`, `claim_id`,
  `flip_coincidence`, `novelty`, `referenced_later`, `position`.
  Stored as **independent fields** with no composite score (spec §5.2).
- `manifest.json` — counts summary.

Embedder selection:

```bash
.venv/bin/python -m agentdiet.cli.analyze --embedder real   # default; needs [analysis]
.venv/bin/python -m agentdiet.cli.analyze --embedder fake   # offline HashingFakeEmbedder
```

If `sentence-transformers` is missing the CLI prints a loud stderr
warning and falls back to the hashing embedder — good for smoke, not
for the real 100-dialogue run.

```bash
make analyze-report   # print counts from the manifest
make analyze-clean    # drop artifacts/analysis/ only
```

## Type-level ablation (Gate 2)

Causal per-type Δ_t = acc(with) − acc(without) on the
`single_wrong ∧ debate_right` subset. Requires prior `make pilot`
(single-agent artifacts), `make collect` (debate dialogues), and
`make extract` (claim artifacts).

```bash
make ablate
```

Samples `--n` dialogues (default 20, seeded) from the eligible
subset and for each of 6 claim types removes all type-t claim spans
from rounds 1..N-1 of history, then replays the final round through
the cached LLM client. A hard `--max-calls` cap (default **500**)
enforces spec §6.1's new-call budget — cache hits are free.

Artifacts in `artifacts/analysis/`:

- `ablation.jsonl` — one row per (qid, drop_type) with
  `correct_with`, `correct_without`, `skipped`, `skip_reason`
- `ablation_summary.json` — per-type aggregates `{n_used, acc_with,
  acc_without, delta}`
- `ablation_manifest.json` — run metadata with call counts

```bash
make ablate-report   # print summary
make gate2           # emit gate2_report.md; exit code encodes verdict
```

Gate-2 exit codes (from `make gate2`):

- `0` — **PASS** (at least one |Δ| ≥ 0.05 → data-supported rule viable)
- `10` — **NULL_RESULT** (all |Δ| ≤ 0.03 → Day-3 switches to
  descriptive comparison; spec §5.4)
- `20` — **INCONCLUSIVE** (between thresholds — gather more dialogues
  or adjust thresholds)

Thresholds (module constants in `cli/ablate.py`): LIKELY 0.10 · PASS
0.05 · NOISE 0.03. Adjust before Gate-2 decision if n is very small.

```bash
make ablate-clean    # drop ablation artifacts only
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
