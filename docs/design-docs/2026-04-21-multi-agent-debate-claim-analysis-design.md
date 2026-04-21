# Multi-Agent Debate — Claim-Level Gain Attribution & Compression

**Date:** 2026-04-21
**Status:** Approved (design phase complete)
**Duration:** 3 days, high-intensity single-developer execution
**Superseded artifacts:** none (new project)

---

## 1. Problem & Research Framing

Multi-agent LLM debate protocols (Du et al. 2023, "Multi-Agent Debate" / MAD) accumulate communication history across rounds. Prompt length grows as O(N × R × L), which raises inference cost. The original project proposal (`PROPOSAL.md`) planned a bottom-up structural analysis on MMLU. Professor feedback flagged two risks:

1. Multi-agent debate often does not meaningfully outperform single-agent on MMLU; heavyweight structural analysis on a protocol that barely helps would dissect noise.
2. The project should first identify a task where debate yields robust gains, then analyze *where* the gains come from, and only then design a compression scheme around the gain-bearing structure.

This spec adopts that feedback in full. The pivot is:

- **Task:** GSM8K (grade-school math reasoning), where MAD-style debate has documented, reproducible gains over single-agent baselines.
- **Research question:** Among the claims exchanged during debate, which ones causally contribute to the accuracy gain?
- **Compression scheme (P1):** *data-driven*. The form of the compression rule is not pre-specified — it is chosen at the start of Day 3 based on the empirical findings of Day 2.

### 1.1 Explicit design constraints

- **Architecture type:** pure code. All control flow is deterministic Python. LLM calls are function-like and stateless on the server side.
- **Model:** single open-source model (Qwen2.5-7B-Instruct) served via vLLM on an OpenAI-compatible endpoint. One model instance, three client-side "agent" objects that each carry their own system prompt and conversation history.
- **No training.** LLM-only throughout (agents, claim extraction, embedding).
- **Time budget:** 3 days. Scope has been cut aggressively — see §7.

### 1.2 Research questions answered by this spec

1. On GSM8K, does 3-agent × 3-round debate yield ≥ 3pp accuracy gain over single-agent? (Gate — §7.)
2. When debate flips a wrong single-agent answer to a correct one, which claim types are on the flip path?
3. For each claim type, what is the causal effect on final accuracy of removing that type from the debate history? (Type-level ablation.)
4. Given the above findings, can a data-supported claim-level compression rule reduce token cost while preserving accuracy, outperforming sliding-window and random-drop baselines?

---

## 2. Architecture

Two-layer, replay-based pipeline. All LLM calls flow through a single cache layer; all analysis reads from cached artifacts and does not re-issue LLM calls except where semantically required (e.g. ablation replays the final round under modified context).

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1 — LLM Cache & Artifacts (底层)                 │
│  ├─ llm_cache/ : JSONL, key = hash(model, prompt)       │
│  ├─ dialogues/ : {question_id}.json (1 file per debate) │
│  └─ claims/    : {question_id}.json (抽取结果)           │
└─────────────────────────────────────────────────────────┘
               ▲                    ▲
               │ writes             │ reads
┌──────────────┴──────┐    ┌────────┴───────────────────┐
│  Collection Stage    │    │  Analysis & Eval Stage      │
│  (Day 1 PM)          │    │  (Day 2–3)                  │
│  ├─ debate_runner    │    │  ├─ claim_extractor         │
│  └─ dataset_loader   │    │  ├─ flip_attributor         │
│                      │    │  ├─ ablator                 │
│                      │    │  ├─ compressor (P1)         │
│                      │    │  └─ evaluator (baselines)   │
└──────────────────────┘    └─────────────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │  reports/       │
                          │  figs + tables  │
                          └─────────────────┘
```

### 2.1 Design principles

1. **Single cache layer.** Every LLM call (debate turns, claim extraction, ablation replays) is keyed by `sha256(model || temperature || messages_json)`. Re-runs are free on cache hits; rebuilds are free on analysis-only changes.
2. **Artifacts are ground truth.** Once a dialogue is written to `dialogues/{qid}.json`, downstream stages are read-only with respect to it. Changing the claim schema requires re-running only `extract_claims`, not the debate.
3. **No shared runtime state.** Each stage is a CLI: reads N artifacts, writes M artifacts. State lives on disk.
4. **Model switching is a prefix change.** Artifacts are partitioned by model name (`dialogues/{model}/{qid}.json`), so gating from Qwen to an API model is a config flip, not a code change.

### 2.2 Model serving

- vLLM server, single Qwen2.5-7B-Instruct instance, OpenAI-compatible endpoint (`localhost:8000`).
- The "3 agents" are client-side state: `Agent(id, system_prompt, conversation_history)`. Each debate turn sends the agent's full history to the server; the server is stateless.
- Prefix caching at the vLLM level naturally accelerates repeated shared prefixes across agents in the same debate.

---

## 3. Components

### 3.1 Project layout

```
agentdiet/
├── __init__.py
├── config.py            # global config (model, paths, seeds)
├── llm_client.py        # OpenAI-compat client + cache layer
├── dataset.py           # GSM8K loader + answer parser
├── agents.py            # Agent class (system prompt + history)
├── debate.py            # debate scheduler (N agents × R rounds)
├── extract_claims.py    # schema-guided claim extraction
├── analysis/
│   ├── flip.py          # flip-point localization
│   ├── signals.py       # per-claim independent signals
│   ├── ablate.py        # claim-type ablation
│   └── cluster.py       # HDBSCAN clustering (OPTIONAL, time-permitting)
├── compress.py          # P1 compression policy (implemented Day 3)
├── evaluate.py          # baselines + ours × metrics
└── cli/
    ├── collect.py
    ├── extract.py
    ├── analyze.py
    ├── evaluate.py
    └── report.py
```

### 3.2 Component contracts

| Component | Input | Output | LLM calls? |
|---|---|---|---|
| `llm_client.chat` | `messages`, `model`, `temperature` | `response: str` | cache-first; real call on miss |
| `dataset.load_gsm8k` | `split`, `n`, `seed` | `List[Question]` | none |
| `debate.run_debate` | `Question`, `n_agents=3`, `n_rounds=3` | `Dialogue` | N × R per debate |
| `extract_claims.run` | `Dialogue` | `List[Claim]` | 1 per message (or 1 per dialogue, TBD at impl) |
| `analysis.flip.locate` | `Dialogue`, `final_answer` | `List[FlipEvent]` | none |
| `analysis.signals.compute` | `Dialogue`, `List[Claim]` | `SignalScores` (4 independent fields per claim) | none (embedding model local) |
| `analysis.ablate.run` | `Dialogue`, `claim_type` | `Δ_accuracy` | 1 final-round replay per dialogue per type |
| `compress.apply` | `Dialogue`, `Policy` | compressed history string | none |
| `evaluate.run` | `method`, `n_questions` | `{acc, avg_tokens, acc_per_1k}` | 1 final-round replay per question per method |

### 3.3 Core data types (pydantic)

```python
class Claim:
    id: str                   # f"{qid}_{round}_{agent}_{idx}"
    text: str
    agent_id: int
    round: int
    type: Literal["proposal","evidence","correction",
                  "agreement","question","other"]
    source_message_span: tuple[int, int]  # char offsets in original message

class Dialogue:
    question_id: str
    question: str
    gold_answer: str
    messages: list[Message]    # Message = {agent_id, round, text}
    final_answer: str | None   # majority vote at last round
    meta: dict                 # model, temperature, timestamp, seed

class FlipEvent:
    question_id: str
    triggering_claim_id: str   # claim in the round when majority flipped
    pre_flip_answers: dict[int, str]    # {agent_id: answer}
    post_flip_answers: dict[int, str]

class SignalScores:
    claim_id: str
    flip_coincidence: bool
    novelty: float             # 1 - max cos sim to prior claims
    referenced_later: bool     # later claim with cos sim > 0.7
    position: int              # round index
```

### 3.4 Claim taxonomy (6 types)

- `proposal` — introduces a new candidate answer or reasoning path
- `evidence` — supports a position with computation, derivation, or citation
- `correction` — identifies an error in a previous claim and offers a fix
- `agreement` — affirms a previous claim without adding information
- `question` — directed at another agent
- `other` — pleasantries, meta-discourse, off-topic

The extractor uses schema-guided JSON mode (vLLM structured output) with 3 few-shot examples covering typical cases.

### 3.5 Agent role differentiation

Three agents in the debate use differentiated system prompts drawn from the MAD literature pattern: one baseline solver, one skeptic, one synthesizer. Exact wording fixed in `agents.py` and logged in `Dialogue.meta`.

---

## 4. Data Flow

```
[GSM8K HF] --dataset.sample(n=100, seed=42)--> dataset/gsm8k_sample.json
                                                       │
                                                       ▼
                            debate_runner × 3 agents × 3 rounds
                          ┌────────────────────────────────────┐
                          │ each turn:                          │
                          │   prompt = sys + history            │
                          │   → llm_client.chat(...)            │
                          │      ↳ cache hit? return : call LLM │
                          └────────────────────────────────────┘
                                                       │
                                                       ▼
                                         dialogues/{model}/{qid}.json
                                                       │
                           ┌───────────────────────────┼───────────────────────────┐
                           ▼                           ▼                           ▼
                  extract_claims              flip_analyzer                 (baseline eval
                  (schema, JSON mode)         (locate round flips)           reads dialogues
                           │                           │                      directly)
                           ▼                           │
                   claims/{qid}.json                   │
                           │                           │
                           ├──── signals.compute ──────┤
                           │   (4 independent fields;  │
                           │    NO composite score)    │
                           │                           │
                           ▼                           ▼
                  signal_scores.parquet        flip_events.jsonl
                           │
                           └───────────┬───────────────┘
                                       ▼
                               ablator (type-level)
                     ┌────────────────────────────────┐
                     │ for each type t:               │
                     │   remove all claims of type t  │
                     │   reconstruct history          │
                     │   llm_client.chat(final round) │
                     │   Δ_t = acc(with) − acc(w/o)   │
                     └────────────────────────────────┘
                                       │
                                       ▼
                                ablation.jsonl
                                       │
                                       ▼
        ┌────── Day 3 start: human reads report, picks policy.json ─────┐
        │   example: {"drop_types": ["agreement", "other"]}             │
        └───────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                                compress.apply
                                       │
                                       ▼
                           compressed/{qid}.json
                                       │
                                       ▼
                  evaluate: 5 methods × 100 questions
                  (b1/b2/b3/b5 + ours) × (acc, tokens, acc_per_1k)
                                       │
                                       ▼
                          evaluation/results.json
                                       │
                                       ▼
                         report.render → figs + tables
```

### 4.1 Artifact directory

```
artifacts/
├── llm_cache.jsonl                  # append-only
├── dataset/gsm8k_sample.json
├── dialogues/{model}/{qid}.json
├── claims/{model}/{qid}.json
├── analysis/
│   ├── flip_events.jsonl
│   ├── signal_scores.parquet
│   ├── ablation.jsonl
│   └── clusters.json                # optional
├── compression/
│   ├── policy.json
│   └── compressed/{qid}.json
├── evaluation/results.json
└── report/
    ├── figs/*.png
    └── tables/*.tex
```

### 4.2 Baselines as view transforms

All baselines and the candidate method are implemented as transformations over the same `Dialogue` artifact, feeding a common `compress.apply(dialogue, policy) → history_string` interface:

- **b1 full-history:** no transformation; upper bound.
- **b2 single-agent:** keep only round 1, agent 1; lower bound.
- **b3 sliding-window:** keep only the last k rounds.
- **b5 random-drop:** drop x% of claims uniformly at random (seeded).
- **ours:** apply `policy.json` derived from Day 2 analysis.

### 4.3 Handling gate failure (model switch)

If the Day 1 pilot gate fails and we switch to an API model, the `model` field changes → cache keys change → dialogues are re-collected under `dialogues/{new_model}/`. Downstream stages reference the model path prefix. No code changes required.

---

## 5. Operationalizing "Importance" — Methodological Stance

The analysis stage intentionally does **not** construct a hand-weighted composite importance score. The reason: arbitrary weights in a score formula make the analysis appear to produce findings that are actually artifacts of the weight choice. The analysis phase is strictly descriptive; the only "importance" quantity named as such is the causal effect measured by type-level ablation.

### 5.1 What the analysis produces

**Per-claim independent measurements** (no composite):
- `flip_coincidence`: does this claim's round coincide with an answer flip?
- `novelty`: how semantically distinct is this claim from earlier claims?
- `referenced_later`: is this claim semantically echoed in later rounds?
- `position`: in what round does it appear?

**Per-type causal measurement** (the only "importance"):
- `Δ_t` for each of the 6 claim types, from type-level ablation on the subset where `single_wrong ∧ debate_right`.

**Descriptive tables:**
- Distribution of claim types × agents × rounds.
- Correlation of each independent signal with being on the flip path (reported separately, not combined).
- Ranking of types by `Δ_t`.

### 5.2 What the analysis does *not* do

- No formula of the shape `score = w1·S1 + w2·S2 + …`.
- No ranking of individual claims by a composite "importance".
- No hand-tuned threshold on a composite score.

### 5.3 How P1 compression is chosen (Day 3 morning)

The compression rule is selected as the simplest rule that follows from the Day 2 evidence:

- If some type has `Δ_t ≈ 0` → drop claims of that type.
- If `novelty == 0` claims can be ablated without accuracy loss → drop low-novelty claims.
- If `referenced_later == False` claims can be ablated without accuracy loss → drop un-referenced claims.
- If no single signal suffices, a weighted combination may be fit by logistic regression on a held-out set of ablation outcomes — in that case weights are data-derived, not hand-picked.

### 5.4 Null-result contingency

It is a legitimate outcome that no independent signal significantly predicts ablation-derived importance, and no claim type has a large `Δ_t`. If this happens, the project reports a negative finding: *"on GSM8K 3×3 debate, we could not isolate a claim subset whose removal preserves accuracy at comparable rates to random drop."* This is scientifically honest and consistent with Day 3 delivery; it does not require re-scoping.

### 5.5 Clustering as conditional add-on

HDBSCAN clustering on claim embeddings runs **only if** (i) the claim-type analysis produces a signal worth layering on top of, and (ii) Day 2 has spare time. Clustering is not part of the critical path. When it runs, it provides a second cut at the data (gain distribution across semantic topics) but is not required for any downstream decision.

---

## 6. Error Handling

Research-grade, not production-grade. Core principles: fail visible, resumable, no silent swallowing.

### 6.1 Failure classes

| Failure | Stage | Handling |
|---|---|---|
| vLLM server unavailable | `llm_client` | 3 exponential retries (1s/4s/16s); on final failure, record `qid` to `failed.jsonl`, continue next question |
| LLM returns malformed JSON / schema mismatch | `extract_claims`, `ablate` | 1 stricter-prompt retry; on failure, record raw output to `failures/`, mark item `extraction_failed=True`, do not abort pipeline |
| Structured output timeout | `extract_claims` | same as above |
| Answer parser fails to extract numeric answer | final aggregation | `final_answer = None`; question excluded from accuracy denominator; count reported in metadata |
| Disk write fails / corrupt JSON | any | fail-fast; do not write partial artifact (atomic `.tmp` + rename) |
| Cache JSONL corruption | startup | on load, validate final line; truncate if incomplete |
| Interrupt / OOM | any | per-qid artifact files; on restart, `skip-if-exists` resume |
| Ablation LLM-call budget overrun | `ablate` | hard cap `MAX_NEW_LLM_CALLS=500`; abort on overrun |

### 6.2 Resumability

Every CLI supports `--resume` (default on):

```python
for qid in questions:
    out = f"artifacts/dialogues/{model}/{qid}.json"
    if os.path.exists(out): continue
    dialogue = run_debate(...)
    atomic_write(out, dialogue.json())
```

### 6.3 Logging

- `logging` module; level from `AGENTDIET_LOG` env var.
- Each CLI entry logs config snapshot (model, seed, n, paths) to stderr.
- `failures/{stage}/{qid}.json` stores raw prompt + raw response + traceback for post-hoc debugging.

### 6.4 Explicitly NOT done

- No circuit breakers, no jittered retry.
- No semantic validation of LLM outputs at runtime (that is Day 2 manual spot-check).
- No automatic model fallback. Gate decisions are human.
- No broad `except Exception: pass`.

### 6.5 Evaluation robustness

- All seeds fixed (Python random, numpy, torch, vLLM).
- Fixed question subset (`dataset/gsm8k_sample.json` generated once).
- Single evaluation run per method (no bootstrap) — acknowledged as limitation in write-up.

---

## 7. Timeline & Gates

Three days, high-intensity single-developer execution.

### Day 1 (~12h) — Infrastructure + pilot + collection

- **AM (4h):** GSM8K loader, debate runner, Qwen2.5-7B serving, dialogue JSON schema, smoke tests.
- **Midday (1h):** Pilot on 30 questions, single vs 3×3 debate.
  - **GATE 1:** if `acc_debate − acc_single < 3pp`, stop; switch to API model (GPT-4o-mini ≈ $2–5 for full experiment); re-pilot. If still <3pp, escalate to user for task-change decision.
- **PM (5–6h):** Full collection of 100 dialogues (background). In parallel: finalize claim extraction prompt + schema; validate on 5 dialogues.

### Day 2 (~12h) — Extraction + attribution

- **AM (3h):** Run full claim extraction; 10-dialogue manual spot check (sanity: not trivially wrong).
- **Midday (3h):** Flip-point localization over all dialogues; compute independent per-claim signals.
- **PM (4–6h):** Type-level ablation on ~20 sampled dialogues from `single_wrong ∧ debate_right`. Clustering as time-permitting bonus.
  - **GATE 2:** if all `Δ_t` are within noise and no signal correlates with flip path, accept null result; Day 3 compression becomes a descriptive comparison rather than a data-supported rule.

### Day 3 (~12h) — Compression design + evaluation + report

- **AM (3h):** Read Day 2 report; write `policy.json` implementing simplest data-supported rule; implement `compress.apply`.
- **Midday (3–4h):** Evaluation sweep: b1/b2/b3/b5 + ours × (acc, tokens, acc_per_1k) × 100 questions.
- **PM (4–5h):** Figures (Pareto curve), tables, write IMRAD report (~6–8 pages).

### Scope cuts already made

- Human validation 30 → 10 dialogues (sanity check only; no NMI/ARI computation).
- Dataset 200–300 → 100 questions.
- Clustering: conditional, not critical path.
- Summarization baseline (b4) cut.
- LLMLingua cross-method comparison (b6) cut.
- Multi-seed evaluation cut.
- Buffer days: zero — gates are the only safety mechanism.

---

## 8. Deliverables

- **d1** Code repository with README that reproduces every artifact from `make all`.
- **d2** Dialogue JSON + claim JSON + 10-dialogue spot-check CSV.
- **d3** Analysis notebook generating all figures and tables.
- **d4** IMRAD report PDF.

---

## 9. Testing

Three layers, minimal surface area.

### 9.1 Smoke tests (`tests/test_smoke.py`, < 30s, dummy LLM backend)

- Cache hit on repeated prompt (no extra call).
- 2×1 minimal debate produces schema-valid dialogue.
- Dataset loader yields non-empty `question`/`gold_answer`.
- GSM8K answer parser handles `#### 42`, `42.0`, `$42` variants.

### 9.2 Dataflow invariants (inline assertions at end of each CLI)

| Script | Invariant |
|---|---|
| `collect` | `n_collected == n_requested − n_failed`; each dialogue has exactly `n_agents × n_rounds` messages |
| `extract_claims` | `source_message_span` within the referenced message; `type` in enum |
| `flip` | every `triggering_claim_id` exists in the claims artifact |
| `ablate` | for each type, at least 1 claim of that type exists; else skip and mark |
| `evaluate` | all methods evaluated on the same `qid` set; `tokens > 0` per question per method |

### 9.3 Experiment sanity checklist (manual, Day 1 PM / Day 2 AM / Day 3 PM)

**Day 1 PM:**
- [ ] 3 random dialogues, read — agents engage, not monologue
- [ ] `final_answer` not overwhelmingly `None`
- [ ] pilot gap within MAD-literature range (3–10pp)

**Day 2 AM:**
- [ ] 10-dialogue claim spot check
- [ ] claim-type distribution not collapsed to one type
- [ ] random 5 claims traceable back to source message text

**Day 3 PM:**
- [ ] `acc(b1) ≥ acc(b2)` — debate actually helps
- [ ] `acc(ours) ≥ acc(b5)` — our selection is better than random
- [ ] token ordering: `b1 > b3 > b5 ≈ ours`

### 9.4 NOT done

No pytest/CI setup. No mocks beyond Layer 1 dummy. No property-based tests. No testing of HF / vLLM internals.

---

## 10. Divergence Risk Analysis

### 10.1 Sources of non-determinism

| Source type | Instance | Notes |
|---|---|---|
| LLM calls | Debate turn generation | temperature=0 but vLLM sampling can still vary across GPU / batch composition |
| LLM calls | Claim extraction JSON output | schema-guided but may still fail or produce unexpected values |
| LLM calls | Ablation final-round replay | same as debate turns |
| External | GSM8K via HF datasets | version-pinned; low risk |
| State | `llm_cache.jsonl` append concurrency | single-process; low risk, but corruption on crash possible |
| Embeddings | sentence-transformer model | deterministic given pinned weights; low risk |
| Clustering | HDBSCAN parameters | non-deterministic if `min_cluster_size` not fixed; optional path |

### 10.2 Risk matrix

| Source | Probability | Impact scope | Risk |
|---|---|---|---|
| Debate turn generation | medium | local (per-turn) | medium |
| Claim extraction JSON | medium | chain (analysis input) | high |
| Ablation replay | medium | local | medium |
| Cache corruption | low | global | medium |
| HF dataset shift | low | global | low |
| HDBSCAN | medium | local (optional path) | low |
| Answer parser ambiguity | medium | chain (ground-truth label) | high |

### 10.3 Divergence trees (medium+ risks)

**Claim extraction JSON malformed**
```
LLM returns non-JSON or missing required field
  → pydantic validation fails
    → extract_claims skips this message
      → downstream signals.compute sees missing claim
        → flip analysis missing context
          → type-level ablation sample biased
            → Δ_t estimates unreliable
              → compression policy chosen on unreliable data
```
Mitigation surfaces: extract_claims retry + `extraction_failed` flag; ablation reports sample size and skip count; manual spot check at Day 2 AM.

**Debate turn stochasticity**
```
Same question, different run produces different agents' final answers
  → accuracy metric varies across runs
    → pilot gate decision may flip near 3pp threshold
      → false negative (gate fails on a protocol that actually works)
```
Mitigation: temperature=0; single evaluation run acknowledged as limitation; gate threshold is a decision point requiring human judgment, not a strict 3pp cut.

**Answer parser ambiguity**
```
LLM outputs "the answer is 42 dollars" where gold is "42"
  → parser returns 42 or None depending on regex
    → accuracy over- or under-estimated
      → gate decision compromised
        → policy evaluation compromised
```
Mitigation: parser tested against known answer variants in Layer 1 smoke tests; parser logs unparseable outputs to a visible file for manual inspection during Day 1 sanity check.

**Cache corruption on crash**
```
Crash mid-write to llm_cache.jsonl
  → on restart, last line is truncated
    → JSON parse error on cache load
      → all subsequent runs fail to start
        → Day 2+ blocked
```
Mitigation: startup validates final line, truncates if incomplete; cache writes are atomic per-entry (full line flush).

### 10.4 Fallback summary

The two critical mitigations for Day 2 analysis integrity are:
1. **Extraction failure handling** — skip + log + spot-check, rather than crash or silent pass.
2. **Answer parser visibility** — unparseable answers are counted and inspected, not silently zeroed.

Other risks either have low impact or are handled by the cache/artifact layer's resumability.
