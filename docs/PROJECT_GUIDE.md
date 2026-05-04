# Project Guide — CSCI3033 LLM Reasoner Final Project

A single-document map of **what we're trying to learn**, **how the
experiment is structured**, and **where every number lives**. Written
to be the front door — read this first, then drill into the design
docs / code / artifacts as needed.

---

## 1. What this project is

We measure whether **multi-agent debate (MAD)** offers genuine
benefit over a **token-matched majority-voting (self-consistency, SC)**
baseline, across a phase grid of `(model × dataset × thinking
mode)`. The motivation is that the canonical +6.7pp MAD claim from
Du et al. (2023) was reproduced in our own RQ0 work, but a drop-all
control showed the gain is *not* attributable to inter-agent
information transfer. That made the *original* compression-rule
question moot until we re-pin where (if anywhere) MAD actually wins.

Stack: Python + vLLM-served HF models + bootstrap CI + (only as a
contingent later step) HDBSCAN/sentence-transformers for token
clustering. Models served on a single HPC node via vLLM; client
talks OpenAI-compatible API.

---

## 2. Research-question lineage

| # | Status | Question |
|---|---|---|
| **RQ0** | done | Does the +6.7pp 3×3 MAD gain over SA on GSM8K + Qwen2.5-7B reproduce, and is it driven by inter-agent information transfer? |
| **RQ1** | **current** | Across `(model × dataset × thinking)`, where does MAD's collapse to SC end? Is there *any* cell where Δ(D−V) is meaningfully positive? |
| **RQ4** | **contingent** | If RQ1 finds a positive cell, *that* cell becomes the setting for resuming the token-pruning / compression-rule line. If RQ1 is universally null, RQ4 retires. |

RQ0 result (locked in `docs/reports/check-in-2026-04-22.tex`):
- 3×3 MAD reproduces +6.7pp over SA on GSM8K + Qwen2.5-7B.
- A drop-all control (peers can't see each other) preserves the gain
  → RQ0 conclusion: the gain is **majority-vote at-temperature-0**,
  not debate-specific.
- Therefore, debate as canonically described offers no marginal
  benefit *on this configuration*.

The pivot to RQ1 is in `docs/design-docs/2026-04-27-debate-phase-mapping-design.md` (current authoritative spec; supersedes the 2026-04-21 doc for RQ1+).

---

## 3. What we're investigating (RQ1 phase grid)

### 3.1 Cells

**Main grid: 16 cells** = 2 models × 4 datasets × 2 thinking states.

| | gsm8k | aime | humaneval+ | livecodebench |
|---|---|---|---|---|
| **Qwen3-30B-A3B**, t=0 | ✓ | ✓ | ✓ | ✓ |
| **Qwen3-30B-A3B**, t=1 | ✓ | ✓ | ✓ | ✓ |
| **gpt-oss-20b**, t=0 | ✓ | ✓ | ✓ | ✓ |
| **gpt-oss-20b**, t=1 | ✓ | ✓ | ✓ | ✓ |

`t=0/1` toggles each model's "thinking" mode (Qwen3:
`enable_thinking`; gpt-oss: `reasoning_effort=low|high`).

**Prompt-robustness sub-grid: 4 cells** = Qwen3 × gsm8k × {`adversarial-strict`, `symmetric`} × {t=0, t=1}. (cooperative variant overlaps with the main grid.)

Per cell: n=40 questions sampled with seed=42.

### 3.2 Three conditions per cell

For each question we run all three:

1. **SA** — single answer at T=0 (1 call).
2. **Voting (SC)** — N independent samples (T=0.7) clustered by parsed answer; the majority answer wins. N is **calibrated per cell** so total tokens roughly match the debate condition (`agentdiet/voting.py:calibrate_n`).
3. **Debate** — 3 agents × 3 rounds, peer messages broadcast each round with thinking traces stripped. Final answer = majority over round-3 per-agent answers.

The token-matched calibration is what lets us claim Δd-v is a *fair*
comparison rather than "debate spent more tokens".

### 3.3 Variables we hold fixed

- temperature = 0.0 for SA & debate; 0.7 for voting samples
- 3 agents, 3 rounds for debate (cooperative prompts; spec §4.3)
- thinking traces stripped from broadcast (`strip_thinking_trace` in `agentdiet/debate/__init__.py`)
- max_tokens caps: only `adversarial-strict` variant is capped (2048 / 8192 for thinking off/on); cooperative & symmetric let vLLM auto-allocate.

---

## 4. What we want to analyze

Six analyses, all driven by `make analyze-phase` →
`agentdiet/cli/analyze_phase.py`.

### 4.1 Per-cell Δd-v with paired bootstrap 95% CI

The headline statistic. For each cell:

```
Δd-v_i = correct_debate(q_i) - correct_voting(q_i)
mean_Δ = (1/n) Σ Δd-v_i
```

Paired bootstrap over the n=40 question vector → percentile CI.
**A cell is "positive" only if its CI lower bound > 0.**

Reported in `docs/reports/data/analysis.json` (`cells[].delta_ci_low/high`) and `docs/reports/tables/phase_summary.tex`.

### 4.2 O1 — thinking-axis effect

For each (model, dataset) pair we compute Δd-v at t=0 and at t=1 and
their difference. Is enabling thinking systematically helpful or
harmful for *debate-vs-voting*?

Output: `docs/reports/tables/thinking_o1.tex`.

### 4.3 O2 — debate-off vs SA-on

Tests whether SA + thinking already captures most of debate's headroom. If `acc(SA, t=1) >= acc(D, t=0)` in many cells, it cheapens the case for debate.

Output: `docs/reports/tables/thinking_o2.tex`.

### 4.4 AIME per-year stratification

AIME has known contamination concerns for older years (≤2023). We split AIME by contest year (2024-sample / 2025 / 2026) and report per-year accuracy + Δd-v. If 2026 (newest, least contaminated) cells flip the sign, contamination is plausibly responsible for older-year results.

Output: `docs/reports/tables/aime_per_year.tex`.

### 4.5 Cross-model comparison

Side-by-side Qwen3 vs gpt-oss for the same dataset+thinking.
Output: `docs/reports/tables/cross_model.tex`.

### 4.6 Phase diagram

Heatmap of Δd-v across the 16-cell main grid, with significance
marked. `docs/reports/figs/phase_diagram.pdf`.

---

## 5. Current finding (snapshot, 2026-05-04)

**Phase map is decisively null.**

- 0 / 20 cells reach significant positive Δd-v (CI > 0).
- 1 / 20 reaches significant *negative* Δd-v: gpt-oss × aime × t=1, Δd-v = −0.225, CI95 = [−0.350, −0.100]. Voting beats debate by 22.5pp on this cell — group-think pathology.
- The single largest positive point estimate (gpt-oss × LCB × t=0, Δd-v = +0.100) has CI lower bound exactly at 0 → not significant.
- Thinking on (t=1) systematically pushes Δd-v *more negative* in non-ceiling cells (e.g. gpt-oss aime: −.05 → −.225; gpt-oss LCB: +.10 → −.10).

**Implication for RQ4**: contingency not satisfied → token-pruning question retires for this regime.

The framing for the final report writes itself: a clean null result
with one named pathology (gpt-oss aime t=1 group-think) and one
near-positive cell (gpt-oss LCB t=0) that doesn't quite clear
significance.

---

## 6. Data index map

### 6.1 Source-of-truth design docs

| Path | Status |
|---|---|
| `docs/design-docs/2026-04-27-debate-phase-mapping-design.md` | **current** (RQ1 phase mapping; RQ4 contingent) |
| `docs/design-docs/2026-04-21-multi-agent-debate-claim-analysis-design.md` | superseded for RQ1+; RQ0 sections still authoritative |

### 6.2 Per-cell experimental artifacts

`artifacts/grid/<cell_dir>/` — one subdirectory per cell, naming convention:
- main grid: `{model_safe}__{dataset}__t{0|1}` (e.g. `Qwen__Qwen3-30B-A3B__gsm8k__t0`)
- sub-grid: `…__pv-{adversarial-strict|symmetric}` suffix

Each cell directory has 5 files:

| File | Schema | Notes |
|---|---|---|
| `sa.json` | `ConditionRecord` (see `agentdiet/grid/types.py:ConditionRecord`) | One QuestionResult per question for the SA condition |
| `voting.json` | `ConditionRecord` | Voting condition; N is per-cell calibrated |
| `debate.json` | `ConditionRecord` | 3×3 debate dialogue summary |
| `sc_calibration.json` | dict | Per-cell calibration: chosen N, mean tokens, over-budget factor |
| `summary.json` | `CellSummary` | Aggregate accuracies, Δd-v, Δd-sa, total tokens |

Each `QuestionResult` carries `final_answer` (parsed answer / extracted code), `correct: bool`, `prompt_tokens`, `completion_tokens`, and `meta` (per-condition extras: voting parsed answers, debate per-agent finals, code judge pass-counts).

**One historical wart**: `Qwen__Qwen3-30B-A3B__gsm8k__t0/sa.json` was originally run at n=80 then truncated to n=40 to align with voting/debate (commit `1df617d`). All other cells are uniform at n=40.

### 6.3 Aggregated analysis outputs (regenerated by `make analyze-phase`)

| Path | Content |
|---|---|
| `docs/reports/data/analysis.json` | Everything: per-cell stats, bootstrap CIs, O1/O2 tables, AIME per-year, cross-model |
| `docs/reports/figs/phase_diagram.pdf` | Δd-v heatmap with significance markers |
| `docs/reports/tables/phase_summary.tex` | Per-cell Δd-v + CI95 |
| `docs/reports/tables/thinking_o1.tex` | Thinking-axis effect on Δd-v |
| `docs/reports/tables/thinking_o2.tex` | SA-on-thinking vs Debate-off comparison |
| `docs/reports/tables/aime_per_year.tex` | AIME 2024-sample/2025/2026 split |
| `docs/reports/tables/cross_model.tex` | Qwen3 vs gpt-oss side-by-side |

### 6.4 Reports

| Path | Content |
|---|---|
| `docs/reports/check-in-2026-04-22.tex` | RQ0 final write-up (locked) |
| `docs/reports/final-report.tex` | RQ1 phase-mapping IMRAD report |
| `docs/reports/final-report.pdf` | Compiled PDF (regenerated via `make phase-report`) |

### 6.5 Code map

| Module | Responsibility |
|---|---|
| `agentdiet/dataset.py` | GSM8K loader |
| `agentdiet/eval/datasets.py` | LCB, HumanEval+, AIME loaders |
| `agentdiet/eval/judges.py` | SubprocessJudge (sandboxed code-execution grader) |
| `agentdiet/llm_client.py` | OpenAI-compatible client w/ caching, retry, threading lock |
| `agentdiet/agents.py`, `agentdiet/prompts.py` | Solver / debate-agent prompts and variants |
| `agentdiet/voting.py` | N-sample voting + per-cell N calibration |
| `agentdiet/debate/` | 3×3 debate runner; thinking-trace stripping; code-debate protocol |
| `agentdiet/grid/runner.py` | Per-cell math/code condition runners (SA / voting / debate) |
| `agentdiet/grid/orchestrator.py` | Per-cell driver: load, run conditions, calibrate, aggregate |
| `agentdiet/cli/grid.py` | Run one or more cells from the command line |
| `agentdiet/cli/rejudge.py` | Re-grade existing code-cell artifacts (used after LCB loader fix; doesn't re-call the LLM) |
| `agentdiet/cli/analyze_phase.py` | Aggregate cells → tables, figure, JSON |
| `agentdiet/analysis_phase/bootstrap.py` | Paired bootstrap CI implementation |

### 6.6 Operational scripts

| Path | Use |
|---|---|
| `scripts/serving/qwen3_a3b_serve.sh` | Start vLLM serving Qwen3-30B-A3B (env-tunable: `VLLM_MAX_MODEL_LEN`, `VLLM_HF_OVERRIDES` for YaRN) |
| `scripts/serving/gpt_oss_20b_serve.sh` | Start vLLM serving gpt-oss-20b |
| `scripts/wait_healthy.py` | Block until vLLM `/health` is green |
| `scripts/run_grid_cells.sh` | Run a sequence of cells, auto-commit + push artifacts after each |

---

## 7. End-to-end runbook (sanity reference)

```
# 1. Start vLLM for whichever model family you're running
bash scripts/serving/qwen3_a3b_serve.sh           # in tmux/separate shell
python scripts/wait_healthy.py --port 8000

# 2. Run one or more cells (commits + pushes after each)
bash scripts/run_grid_cells.sh \
    qwen3:gsm8k:0 qwen3:gsm8k:1 \
    qwen3:aime:0 qwen3:aime:1 \
    qwen3:humaneval_plus:0 qwen3:humaneval_plus:1 \
    qwen3:livecodebench:0 qwen3:livecodebench:1

# 3. After all cells are in artifacts/grid/, run analysis + report
PYTHON=python3 make analyze-phase
PYTHON=python3 make phase-report   # needs pdflatex / tectonic
```

For long-context cells (Qwen3 AIME thinking-on debate), serve with
YaRN to extend max_model_len above the 40960 native limit:

```
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_MAX_MODEL_LEN=65536 \
VLLM_HF_OVERRIDES='{"rope_scaling":{"rope_type":"yarn","factor":1.6,"original_max_position_embeddings":40960},"max_position_embeddings":65536}' \
bash scripts/serving/qwen3_a3b_serve.sh
```
