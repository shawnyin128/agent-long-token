# Multi-Agent Debate — Phase Mapping of When Debate Beats Self-Consistency

**Date:** 2026-04-27
**Status:** Design pivot in progress (replaces 2026-04-21 spec for everything past §5)
**Trigger:** 2026-04-22 check-in feedback (reviewer noted three things: clarify the §3-4 perfect-and-unchanged-accuracy result; add a token-matched self-consistency baseline at the same compute as 3×3 debate; characterize the negative result in detail).
**Supersedes (in part):** `2026-04-21-multi-agent-debate-claim-analysis-design.md` §6–§8 (compression rule, evaluation sweep, original Day-3 plan).

---

## 1. Story line (one-paragraph thesis)

In multi-agent debate (MAD), agents converse over multiple rounds and the literature attributes the resulting accuracy gain to information exchange — proposals, corrections, syntheses. Our preliminary attribution analysis on GSM8K with Qwen2.5-7B-Instruct showed that this attribution does not survive a causal test: type-level ablations were all zero, and a drop-all control (every inter-agent message replaced by the empty string) preserved accuracy at 100% on the *single-wrong ∧ debate-right* subset. The simplest explanation is that "debate" on this regime reduces to **self-consistency (independent-sample majority voting) with extra inference cost**: three independent re-solves followed by a majority vote, dressed up as a dialogue. **This is also why the project's original compression-rule research question (token-level pruning of debate communication, see 2026-04-21 spec §1.2 RQ4) cannot be answered as posed: there is no debate-gain-bearing structure to compress.** This pivoted project therefore asks **where the collapse to voting ends** — over which combinations of model capability, problem difficulty, task domain, and reasoning mode (thinking vs. non-thinking) does multi-agent debate provide measurable benefit over a token-matched majority-voting baseline? The result serves two purposes: (i) as a standalone contribution, a **phase diagram** identifying the regime in which debate is genuinely useful (if any), with a token-matched voting baseline supplying the missing methodological control that prior MAD work omits; (ii) as a **gating experiment for the original compression research question** — any cell that shows positive debate benefit becomes a valid setting in which to revisit token-pruning, while a universally null phase diagram retires that question for this project's regime.

## 2. What changed from the 2026-04-21 spec

| Item | Old (2026-04-21) | New (2026-04-27) |
|---|---|---|
| Output | Data-supported claim-level compression rule + Pareto figure | Phase diagram: when does debate benefit > 0 over token-matched majority-voting baseline? Compression rule itself recast as contingent follow-up (RQ4). |
| Main RQ | Which claim types are causally important for the debate gain? | Across (model × difficulty × thinking) what regime gives debate genuine benefit? Compression question recast as gated by this answer. |
| Baselines | single-agent, sliding-window, random-drop, "ours" heuristic | single-agent (SA), **token-matched SC** (new core baseline), 3×3 debate |
| Models | Qwen2.5-7B only | Qwen3-30B-A3B + GPT-OSS-20B (open-source main); GPT API optional probe |
| Datasets | GSM8K only | GSM8K, AIME 2025, HumanEval+, LiveCodeBench (4 datasets, 2 domains) |
| Thinking axis | not studied | toggled per cell (thinking on / off) |
| Claim extraction + type-level ablation | main result | demoted to methodological appendix that *explains* why debate collapses |
| Compression rule | required deliverable | dropped (logically vacuous if information transfer is null) |

The §1–§5 content of the 2026-04-21 spec (debate protocol, claim schema, gates, drop-all control, preliminary results) remains valid and is treated as completed RQ0; this document specifies RQ1–RQ3.

## 3. Research questions

- **RQ0 (done — see check-in 2026-04-22).** On GSM8K + Qwen2.5-7B, can the +6.7pp 3×3 debate gain be attributed to specific claim types? **Result:** no — null result confirmed by drop-all control.
- **RQ1 (main).** Across {Qwen3-30B-A3B, GPT-OSS-20B} × {GSM8K, AIME 2024–2026 pool, HumanEval+, LiveCodeBench} × {thinking on, thinking off}, what is Δ = acc(3×3 debate) − acc(token-matched majority-voting baseline)? At which (model, dataset, thinking) cells is Δ statistically distinguishable from 0?
- **RQ2 (thinking axis).** Two observation questions, both answerable from RQ1 data:
  - **O1 — does thinking absorb debate's marginal value?** For each (model, dataset), compare Δ(thinking-off) vs. Δ(thinking-on). If Δ(on) is reliably smaller than Δ(off), thinking is taking on the work that debate was doing in the off regime.
  - **O2 — is debate-without-thinking equivalent to single-agent-with-thinking?** For each (model, dataset), compare acc(debate, thinking-off) vs. acc(SA, thinking-on). If they match closely, external dialogue and internal CoT are interchangeable forms of test-time compute, regardless of whether either is also a self-consistency surrogate.
  
  These observations are not mutually exclusive — both can be true, false, or partially true on different cells. We report both as descriptive findings rather than as competing hypotheses, and discuss the resulting joint pattern in the report.

- **RQ3 (prompt robustness).** Does the collapse-to-voting depend on a specific prompt design? On Qwen3-30B-A3B + GSM8K × thinking on/off, replace the cooperative (solver / skeptic / synthesizer) role prompts with adversarial-strict (skeptic must produce ≥1 concrete disagreement) or symmetric (3 identical agents, no role differentiation) prompts. Does Δ become non-zero under any prompt variant?
- **RQ4 (contingent — original compression-rule question, gated by RQ1 outcome).** *Only pursued if RQ1 identifies at least one cell with statistically distinguishable Δ > 0.* In such a cell, which tokens / claims / message-segments inside the debate communication carry the gain, and can a data-supported pruning rule reduce token cost while preserving accuracy? This is the original 2026-04-21 spec's central question, which has been recast as the dependent payoff of the gating experiment: if no cell shows debate gain, there is nothing to compress; if a cell does, that cell is the natural setting in which to resume the compression analysis. RQ4 is **not** in scope for this project's deliverable — it is the **next-iteration follow-up** that the phase diagram either unlocks or formally retires.

## 4. Experimental design

### 4.1 Models

| Model | Source | Hosting | Thinking toggle |
|---|---|---|---|
| Qwen3-30B-A3B | Qwen3 (MoE, 30B total / 3B active) | vLLM on HPC A100 | native `enable_thinking={True,False}` |
| GPT-OSS-20B | OpenAI open-weights | vLLM on HPC A100 | `reasoning_effort=low` (off) vs `high` (on) |
| GPT (API) | closed | API | `reasoning_effort` toggle (optional probe; not required for main results) |

Qwen3.5 was considered but does not preserve hybrid thinking toggle in the same checkpoint (model line was split into thinking-only and instruct-only) — using it would degrade the thinking comparison from a within-model toggle to a between-checkpoint comparison, so we keep Qwen3.

### 4.2 Datasets

| Dataset | Domain | Difficulty | N (cap) | Notes |
|---|---|---|---|---|
| GSM8K (test) | math | easy | 80 | Already used in RQ0; same seed (42) |
| AIME 2024+2025+2026 (mixed) | math | hard | 80 (sampled, see below) | Multi-year pool to reach a statistically-readable n; per-year contamination stratification — see §4.2.1 |
| HumanEval+ (evalplus) | code | easy | 80 (HumanEval+ has 164 total) | Strengthened test cases vs. original HumanEval |
| LiveCodeBench | code | hard | 80 | Filter `contest_date ≥ 2024-08` for contamination control; cap at 80 if available, else use what passes the date filter |

The math difficulty range has a known **gap between GSM8K (easy) and AIME (hard)**; MATH-500 was considered for the middle band but cut to keep the grid at four datasets. The gap is acknowledged as a limitation (§7) — coding partially compensates by spanning easy → hard within its own domain.

#### 4.2.1 AIME multi-year sampling

AIME 2025 alone (n=30) gives 95% bootstrap CIs of width ≈ ±15-18pp on Δ — wide enough that most plausible effect sizes are statistically indistinguishable from zero. To reach a readable n, we pool three contests:

| Year | n included | Contamination expectation |
|---|---|---|
| AIME 2026 (Feb 2026) | 30 (all) | **None** — both open-source models have train cutoffs before Feb 2026; even web crawls of 2026-02 solutions are too recent to enter pretraining |
| AIME 2025 (Feb 2025) | 30 (all) | **Low** — Qwen3 (released Apr 2025) and GPT-OSS-20B (released Aug 2025) have cutoffs likely before Feb 2025; web solutions are scrapeable but unlikely amplified into training |
| AIME 2024 (Feb 2024) | 20 (random sample, seed 42) | **Possible** — both models' training windows likely include AIME 2024 solutions from the web |

Total: 80 questions. The 20-of-30 sample for AIME 2024 is by `random.seed(42); sample(range(30), 20)` for reproducibility.

Per-year stratified analysis is part of Phase F (§6): we report Δ separately for each year. If 2024 differs systematically from 2026, it is flagged as contamination-suspected and excluded from primary phase-mapping conclusions; if all three years agree, contamination is not driving the signal and the pooled n=80 result stands.

### 4.3 Experimental conditions

For every (model, dataset, thinking) cell, run three conditions:

1. **SA — single-agent.** One agent, no debate, no voting. One inference per question. **Temperature: 0** (deterministic baseline).
2. **Token-matched majority-voting baseline (self-consistency, Wang et al. 2023).** N independent SA samples + majority vote on the parsed answer. No inter-agent dialogue — each sample is solved from scratch on the question alone. **Temperature: 0.7, top_p: 0.95** (Wang et al. 2023 standard — diversity must come from sampling because there are no role prompts to differentiate samples). **N is calibrated per cell** to total-token parity with 3×3 debate, with a floor of N=3 to preserve voting semantics. Full procedure in §5.3.
3. **3×3 debate.** 3 agents × 3 rounds, **temperature 0** (Du et al. 2023 standard — diversity comes from differentiated role prompts, not sampling), full message history shared between agents, **but agents' thinking traces are NOT shared** — each agent's thinking is internal; only its final per-turn message is broadcast. **The role assignments are domain-specific** — math uses the protocol from RQ0, code uses a code-native protocol (§4.3a, §4.3b).

**Methodological asymmetry (acknowledged):** the majority-voting baseline and debate derive sample diversity from different sources — temperature for voting, role prompts for debate. We adopt each literature's default rather than equalizing, because (i) temp-0 voting degenerates (all samples identical, vote = SA), and (ii) temp-0.7 debate departs from MAD literature and would not be comparable to prior reported gains. Listed as a known confound in §7.

#### 4.3a. Math-domain debate protocol (GSM8K, AIME pool)

Carry over from RQ0:
- **Roles:** solver (proposes a solution), skeptic (probes for errors), synthesizer (resolves disagreements, produces final answer)
- **Round 1:** all 3 agents produce an initial solution in their role's voice
- **Rounds 2–3:** each agent sees the other 2 agents' previous-round messages and revises in role
- **Per-message format:** natural language reasoning + a final answer line (existing parser extracts the numeric answer)
- **Aggregation:** majority vote on the parsed answer from each agent's round-3 message

#### 4.3b. Code-domain debate protocol (HumanEval+, LiveCodeBench)

The math role triplet (solver / skeptic / synthesizer) does not transfer cleanly to code: "skeptic" on a code chain reduces to either bug-hunting or empty agreement, and "synthesizer" without something to synthesize defaults to copying. We therefore define a code-native role triplet:

- **Proposer.** Writes a complete code solution. Defends and refines its approach across rounds. Per-round output: short rationale (≤ 200 words) + full code in a fenced block.
- **Reviewer.** Reads the other agents' code from the previous round, identifies concrete issues (bugs, missing edge cases, complexity problems), and **also produces its own code** that addresses what it found. Per-round output: critique notes (≤ 200 words, structured by which agent's code each note targets) + full code.
- **Integrator.** Reads all prior-round outputs and produces the best synthesis it can. Per-round output: short integration rationale (≤ 200 words, naming which prior ideas it kept and which it dropped) + full code.

**Round structure.**
- **Round 1:** All 3 agents produce code from the question alone (no inter-agent visibility yet).
- **Rounds 2–3:** Each agent sees the other 2 agents' previous-round outputs and produces a new (rationale + code) pair in role.

**Per-message schema** (uniform across all 3 roles, parsed by the runner):
```
## Notes
<≤200 words; role-specific content>

## Code
```python
<full solution>
```
```

The 200-word notes cap is to keep prompt growth manageable across rounds. Code blocks are not capped — full solution every round.

**Aggregation.** All 3 agents produce code at round 3. Apply functional clustering (§5.4) to the 3 final-round code samples; the answer is a representative from the largest cluster (ties broken by integrator's solution). This makes math and code use the same final aggregation rule (majority on functional/parsed equivalence).

**Why every agent always emits code (including reviewer).** If the reviewer were critique-only, we would have 2 code samples per round instead of 3, breaking the parallel with the majority-voting baseline (which votes over N independent code samples). Forcing every agent to commit code keeps the comparison clean: debate is "3 agents talk while writing code" vs. voting is "N agents write code in silence."

**Distinction from the prompt-robustness "symmetric" sub-grid (§4.5).** The symmetric sub-grid runs only on Qwen3-30B-A3B + GSM8K, not on code. So the code-domain protocol does not collide with the symmetric prompt variant. If we later want to ask "does role differentiation matter on code?", that is a follow-up ablation explicitly out of scope for this project.

### 4.4 Main grid

| Model \ Dataset | GSM8K | AIME 2025 | HumanEval+ | LiveCodeBench |
|---|---|---|---|---|
| Qwen3-30B-A3B (T+) | ✓ | ✓ | ✓ | ✓ |
| Qwen3-30B-A3B (T−) | ✓ | ✓ | ✓ | ✓ |
| GPT-OSS-20B (T+) | ✓ | ✓ | ✓ | ✓ |
| GPT-OSS-20B (T−) | ✓ | ✓ | ✓ | ✓ |

= **16 cells × 3 conditions = 48 condition-runs**, each at the per-dataset N cap above.

### 4.5 Prompt robustness sub-grid

Run only on Qwen3-30B-A3B + GSM8K:

| Prompt variant | T+ | T− |
|---|---|---|
| cooperative (current) | ✓ | ✓ |
| adversarial-strict (skeptic must produce ≥1 concrete disagreement; synthesizer must enumerate disagreement points before resolving) | ✓ | ✓ |
| symmetric (3 identical agents, no role; "review the other agents' work and produce your final answer") | ✓ | ✓ |

= **6 cells × 3 conditions = 18 condition-runs.**

### 4.6 Optional API probe

Reserved budget cap **$20**. If main grid completes on schedule, run gpt-5-mini on GSM8K + HumanEval+ × thinking on/off (4 cells, 80 questions each). This is a **frontier probe**, not a primary result — used in discussion to comment on whether the open-source phase pattern generalizes upward.

### 4.7 Statistical treatment

For each (model, dataset, thinking) cell, compute Δ = acc(debate) − acc(SC). Use **paired bootstrap (10000 resamples)** over the question set to derive a 95% CI for Δ. Report Δ as "non-zero" only if the CI excludes 0. AIME 2025 (n=30) will have wide CIs; this is acknowledged.

## 5. Method details

### 5.1 Backend integration

Two open-source backends behind a unified `LLMClient` interface (already partially exists in `agentdiet/llm_client.py`):

- vLLM OpenAI-compatible endpoint for Qwen3-30B-A3B and GPT-OSS-20B (one server per model)
- thin client adapter exposes `enable_thinking` / `reasoning_effort` as a single `thinking: bool` parameter so the rest of the pipeline is backend-agnostic
- existing JSONL cache layer (`artifacts/llm_cache.jsonl`, currently ~1.1 MB) is **partitioned by (model, thinking)** — same partition scheme already used for model partitioning. No code changes for partitioning.

### 5.2 Eval framework: evalscope

Use evalscope for **dataset loading and answer evaluation** (especially code execution sandbox + pass@1 for HumanEval+ / LiveCodeBench). Wrap evalscope's per-question evaluator behind a `Judge` adapter so the debate runner can call it uniformly. Math judging stays on our own answer parser (existing GSM8K parser; AIME parser is a thin extension — boxed integer answer).

We do **not** use evalscope's debate / multi-agent runners — we keep our own debate orchestration so that thinking-trace handling and turn structure are under our control.

### 5.3 Token-matched majority-voting baseline procedure

**Purpose.** This baseline answers a single question raised by reviewer feedback: at the same compute budget as 3×3 debate, can independent-sample majority voting (Wang et al. 2023 self-consistency) match or exceed debate accuracy? If yes, debate's "dialogue" is decorative — it's just compute. If no, dialogue carries information.

The match should be **conservative against debate** (give the voting baseline at least as much budget as debate, never less), so that "voting did not beat debate" cannot be dismissed as "voting was under-resourced."

**"Token" definition: total tokens (input + output).** Not output-only. Reasons:
- API/compute cost is total tokens, not output tokens
- Debate's prompts grow O(N×R×L) — the input cost is real and must be counted
- Output-only would systematically under-count debate's compute and over-resource the voting baseline

**Per-cell calibration:**

1. Run 3×3 debate on the first 10 questions; record total tokens (input+output, summed across agents and rounds) per question.
2. Run SA on the same 10 questions; record total tokens per question.
3. Compute raw match `N_raw = ⌈mean(debate_total_tokens) / mean(SA_total_tokens)⌉`.
4. Apply floor: `N = max(N_raw, 3)`. Rationale: N<3 collapses majority voting to either pass-through (N=1) or near-tie (N=2), at which point the baseline is no longer doing what its name says. The floor preserves "voting" as a meaningful operation.
5. Run the voting baseline with this N on the full dataset (80 or dataset cap).
6. Log calibration in `artifacts/sc_calibration_{cell}.json`, including: `N_raw`, `N` (post-floor), `mean_debate_tokens`, `mean_sa_tokens`, and `over_budget_factor = (N × mean_sa_tokens) / mean_debate_tokens` (>1.0 means voting got more compute than debate).

**Cells with `over_budget_factor > 1` are flagged in the phase diagram** (e.g., open vs filled markers). If voting still loses to debate in an over-budgeted cell, the conclusion is *strengthened* — debate beat voting even when voting was given a compute advantage.

Expected behavior: thinking-off cells will have `N_raw ≈ 8-12`, floor inactive. Thinking-on cells will likely have `N_raw ≤ 3` (because thinking-on SA already consumes much of the budget per call), floor active, `over_budget_factor` 1.5-3×.

### 5.4 Code execution sandbox

evalscope ships an execution harness for HumanEval+ / LiveCodeBench. Per generated code sample:

- 10s wall-clock timeout, fresh subprocess per test
- runs in a dedicated temp dir; no network, no shared filesystem outside the dir
- pass@1 = fraction of test cases that pass for the single generated solution

**Functional clustering** (used by both the voting baseline and the debate aggregation rule for code):
- For each generated code sample, run it against the **public** test cases (the ones bundled in the prompt — for HumanEval+ these are the docstring examples; for LiveCodeBench these are the listed sample I/O pairs)
- Compute a pass/fail signature: a tuple `(pass, fail, fail, ...)` indexed by test case
- Cluster samples by exact signature equality
- Return a representative from the largest cluster; ties broken by lexical order of the cluster keys (deterministic)
- The returned representative is then evaluated against the **hidden** test cases (the actual eval set) to compute pass@1

This is the standard self-consistency-on-code definition (Wang et al. 2023 adapted by Chen et al. 2023 for code). Public-test signatures avoid leaking the hidden eval into the clustering step.

### 5.5 Phase diagram (the main figure)

Scatter:
- X axis: SA accuracy on the cell (model × dataset × thinking) — a composite "single-agent capability"
- Y axis: Δ = acc(debate) − acc(majority-voting baseline), with 95% CI bars
- Color: thinking on vs off
- Marker shape: domain (math vs code)
- Marker fill: filled if voting baseline was at parity with debate budget (`over_budget_factor ≈ 1`), open if voting got more compute than debate (`over_budget_factor > 1.2`) — open markers mean "voting baseline was given the advantage"
- Annotations: cell label

Three readable patterns:

- All points cluster around Δ ≈ 0 → universal SC collapse, RQ0's negative result is a task-level property, not Qwen2.5-specific
- Inverted-U (Δ peaks in middle X range) → debate helps where SA is "competent but not saturated"
- Monotone decreasing → debate is a substitute for capability (helps weak SA, useless for strong SA)

Whichever shape comes out, the figure is the punchline.

## 6. Step-by-step execution plan

Phases are dependency-ordered. Each phase ends with a concrete artifact.

### Phase A — Backend & infra (≈ 2.5 days)

1. **A1. Stand up Qwen3-30B-A3B on vLLM (HPC).** Verify `enable_thinking` flag end-to-end via 5 GSM8K test calls (toggle on/off, confirm output structure differs). Artifact: smoke-test log.
2. **A2. Stand up GPT-OSS-20B on vLLM (HPC).** Verify `reasoning_effort` low/high differs in output token count and CoT presence. Artifact: smoke-test log.
3. **A3. Extend `llm_client.py`** with a `thinking: bool` parameter and per-(model, thinking) cache partition. Existing tests must still pass; add 2 new tests covering toggle behavior.
4. **A4. evalscope integration.** Install, wrap dataset loaders behind a `Dataset` interface. Wrap code-execution sandbox behind a `Judge` interface. Coverage: GSM8K / HumanEval+ / LiveCodeBench through evalscope; **AIME 2024+2025+2026 pool** loaded directly from public sources (MAA / AoPS) since evalscope only ships AIME 2024 — write a thin custom loader producing the 80-question pool per §4.2.1 (deterministic seed 42 sampling on AIME 2024). Artifact: `agentdiet/eval/` module + 1 test per dataset confirming a known-correct answer scores 1.0; AIME loader has an additional test confirming year-stratified question count (30/30/20).
5. **A5. Code-domain debate protocol.** Implement the proposer / reviewer / integrator role prompts for code (§4.3b). Implement the per-message schema parser (Notes / Code split). Implement functional clustering for both the voting baseline and debate aggregation (§5.4). Artifact: `agentdiet/debate/code_protocol.py` + unit tests covering schema parse and 3-way clustering.
6. **A6. Token-matched voting baseline implementation.** Add `agentdiet/voting.py` implementing the per-cell calibration procedure (§5.3) — total-token match with N≥3 floor, plus over-budget logging. Output: `artifacts/sc_calibration_{cell}.json`. Test: on a synthetic LLM client, calibration produces deterministic N and correctly logs over-budget cases.

### Phase B — Pilot (≈ 1 day)

7. **B1. Math pilot.** Run all three conditions on Qwen3-30B-A3B + GSM8K × thinking off (replicate the prior Qwen2.5-7B regime as a sanity check on the new backend and the math protocol from RQ0).
8. **B2. Code pilot.** Run all three conditions on Qwen3-30B-A3B + HumanEval+ × thinking off. **This is the first end-to-end test of §4.3b**; expect to find protocol bugs (schema parsing, code execution timeouts, clustering ties). Budget 0.5 day for fix-and-rerun cycles.
9. **B3. Sanity gate.** Confirm: math pilot reproduces a non-trivial SA accuracy (≥80%); code pilot reproduces a non-trivial pass@1 (≥30% for HumanEval+ on this model); the voting baseline calibration produces well-defined N for both pilots; no condition produces empty/un-parseable outputs at >5% rate. Failure on the **math pilot** means a backend or pipeline bug — investigate before scaling. Failure on the **code pilot** is more likely a §4.3b protocol bug — iterate the protocol on this pilot only, before extending to LiveCodeBench. Note: the gate does NOT require debate − SA ≥ 3pp — that would presume the result we're trying to measure.
10. **B4. Calibration log.** Document N, `over_budget_factor`, and any protocol patches made during pilot.

### Phase C — Main grid (≈ 2 days)

9. **C1. Open-source main grid (16 cells).** Run all three conditions on each cell. Order: thinking-off cells first (cheaper, faster) then thinking-on. Within each, math before code (code sandbox is the riskier dependency; do it second so any sandbox bugs are caught after we already have a partial story).
10. **C2. Per-cell artifact.** `artifacts/grid/{model}__{dataset}__t{0,1}/` containing `sa.json`, `sc.json`, `debate.json`, `summary.json` (acc, CI, token usage).

### Phase D — Prompt robustness (≈ 0.5 day)

11. **D1. Implement two new prompt variants** in `agentdiet/prompts.py`: `adversarial_strict`, `symmetric`. Existing `cooperative` stays as default.
12. **D2. Run** Qwen3-30B-A3B + GSM8K × {3 prompts} × {thinking on, off} (6 cells × 3 conditions). Same artifact structure as Phase C.

### Phase E — Optional API probe (≈ 0.5 day; budget-gated)

13. **E1.** If main grid completes within Phase C's time budget AND $20 API budget is intact, run gpt-5-mini on GSM8K + HumanEval+ × thinking on/off. Skip if either condition fails. Document the skip.

### Phase F — Analysis (≈ 1 day)

14. **F1. Phase diagram.** Generate scatter (§5.5). Save to `docs/reports/figs/phase_diagram.pdf`.
15. **F2. Per-axis characterization.**
    - Per-dataset stratification (e.g., GSM8K bucketed by reasoning-step count; LiveCodeBench bucketed by problem difficulty rating)
    - **AIME per-year stratification:** report Δ separately for AIME 2024 / 2025 / 2026 to bound contamination effects (§4.2.1). If 2024 disagrees with 2026, pooled AIME numbers are flagged contamination-mixed.
    - Cross-model agreement table (do both open models tell the same story per dataset?)
    - Round-by-round trajectory (after round 1, 2, 3, what is the majority vote? Does it move?)
    - Voting-wrong ∧ debate-right reverse-attribution case study (find any such question, present the dialogue)
16. **F3. Thinking-axis observations.** Compute O1 (Δ on vs Δ off, per model × dataset) and O2 (acc(debate, off) vs acc(SA, on), per model × dataset) tables. Optionally interpret the joint pattern using the H_A / H_B / H_C framings (debate-as-CoT-substitute / collapse-persists-under-thinking / thinking-dominates) — these are explanatory frames for the discussion, not pre-registered tests.
17. **F4. Methodological appendix.** Re-state RQ0's claim-extraction + type-level ablation results as the *mechanism* explaining the SC collapse on GSM8K (claim distribution skew → no inter-agent information to transfer). This is the role of the old direction in the new paper.

### Phase G — Write-up (≈ 1 day)

18. **G1. IMRAD report draft** → LaTeX, citing Du et al. 2023, Wang et al. 2023 (self-consistency), and our drop-all control as setup.
19. **G2. Final figures + tables** via `make report`.
20. **G3. PDF via pandoc** (or direct LaTeX → PDF).

**Total estimated effort:** ~8 days (Phase A 2.5 + B 1 + C 2 + D 0.5 + E 0.5 + F 1 + G 1 = 8.5; rounded down assuming some overlap). Buffer is now negative — see §7 limitations on schedule pressure. The 6.5-day figure in earlier drafts assumed a math-only pipeline; adding the code-domain debate protocol (§4.3b) and an honest backend-stand-up estimate adds ~1.5 days.

## 7. Limitations and known gaps

- **Math middle-difficulty band uncovered.** GSM8K (easy) → AIME 2025 (hard) skips MATH-500. If the phase diagram shows action only in middle X range, we have to caveat that math is under-sampled in that range.
- **AIME contamination layered.** AIME 2024 questions (20 of 80) are likely present in both open models' pretraining; AIME 2025 (30) is borderline; AIME 2026 (30) is contamination-clean. We mitigate by reporting per-year Δ in Phase F. If the per-year cuts disagree, the pooled n=80 figure must be re-interpreted as a contamination-mixed estimate, not a clean hard-math estimate.
- **Two open models is a small basis.** If they disagree on a cell, we can't tell whether one is the outlier without a third model.
- **Single seed per condition.** No multi-seed bootstrap on the *generation* side; only resampling-bootstrap over the question set. Adding seed variance would multiply runtime.
- **Diversity-source asymmetry between SC and debate.** SC's diversity comes from temperature 0.7 sampling (Wang et al. 2023 default); debate's diversity comes from differentiated role prompts at temperature 0 (Du et al. 2023 default). Equalizing these would force one of the two baselines off its literature-standard configuration; we keep the literature defaults so prior reported numbers remain comparable. The consequence is that any Δ(debate − SC) we report contains a small unidentified component attributable to "role-prompt diversity vs. temperature diversity" rather than purely to information exchange. We treat this as a known methodological caveat and not as a confound that invalidates the phase-mapping result.
- **Thinking-trace sharing decision is fixed (§4.3 condition 3).** Sharing thinking traces between agents was not explored; we note this as a future ablation, not in scope.
- **Closed-source generalization is optional probe-only.** No frontier-only claim is made.
- **"Thinking on" is not the same operator across models.** Qwen3-30B-A3B's `enable_thinking` is the result of hybrid reasoning training; GPT-OSS-20B's `reasoning_effort=high` is RL-based reasoning. Within a model, on-vs-off is a clean within-checkpoint toggle; *across* models, "thinking on" represents different mechanisms. Consequently the thinking axis is interpreted within-model only — cross-model averaging of thinking effects is not reported.
- **Cross-model token comparisons are nominal, not semantic.** Different tokenizers carry different content per token. Within-cell comparisons (same model, same dataset, same thinking) are unaffected; cross-model comparisons of "matched compute" are nominal alignments at the token-count level, not at the information-content level.
- **The result is for one specific debate protocol (3 agents × 3 rounds, role-differentiated, per §4.3a/b).** The phase diagram is not a claim about all multi-agent debate. Other configurations (more agents, more rounds, judge-arbiter, tournament-style, thinking-trace shared) are out of scope and may behave differently.

## 8. Out of scope (explicitly deferred from the 2026-04-21 direction)

- **Compression rule** (`compression-policy` feature, evaluation sweep with b1/b2/b3/b5/ours methods, Pareto figure) — recast as **contingent RQ4** (§3). The compression question is not dead; it is gated on the phase diagram outcome. If RQ1 finds at least one cell with positive debate benefit, that cell becomes the natural setting for resuming the compression analysis as a follow-up. If RQ1 is universally null, RQ4 is formally retired for this regime.
- Claim clustering, ICL-guided compression (P0/P1 of the original proposal). Same gating: contingent on a positive RQ1 cell. Otherwise stays in future work.
- Cross-seed generation variance. Out of scope; mentioned as limitation.

## 9. Deliverable

- IMRAD final report (LaTeX → PDF) as the course final.
- Reproducible artifact: code in `agentdiet/`, all per-cell raw outputs in `artifacts/grid/`, calibration logs, and figure-generation script.
- Public repo at https://github.com/shawnyin128/agent-long-token .

## 10. Mapping to features.json (to be applied later — NOT in this commit)

The features-list update is intentionally deferred per user direction. When applied, the changes will be:

- **Mark superseded:** `compression-policy`, `evaluation-sweep` (deliverables no longer needed).
- **Reframe:** `report-generation` (now generates the phase diagram instead of compression Pareto).
- **Add:** `cross-model-grid` (Phase A1+A2+A3, Phase C), `coding-eval-infra` (Phase A4), `thinking-axis` (covered by A3+C), `prompt-robustness` (Phase D), `phase-mapping-analysis` (Phase F).
- **Keep as done (re-purposed as appendix):** `claim-extraction`, `flip-and-signals`, `type-level-ablation`.

---

**End of design.** Approval gate: user reviews this document and confirms the story line, the phase plan, and the limitation list. After approval, `features.json` will be updated to match §10 in a separate step.
