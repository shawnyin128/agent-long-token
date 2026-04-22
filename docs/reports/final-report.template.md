---
title: "Multi-Agent Debate Claim-Level Gain Attribution & Compression on GSM8K"
author: "TODO: name"
date: "TODO: date"
---

# Abstract

**TODO (~150 words):** State the problem (debate prompt growth),
the pivot from MMLU to GSM8K, the research question (which claim
types cause the gain?), method (type-level ablation + compression),
and headline result (Δ_t ranking, acc-per-1k ordering).

# 1. Introduction

**TODO:** Motivate debate, cost growth, the professor's feedback
that shifted MMLU → GSM8K. State the four research questions (spec
§1.2). One paragraph on why attribution before compression matters.

# 2. Methods

## 2.1 Dataset and debate protocol

GSM8K test split, 100 questions (seed=42), 3 agents × 3 rounds.
Solver / Skeptic / Synthesizer role prompts.

## 2.2 Claim extraction

Schema-guided extraction with 6-type taxonomy (proposal, evidence,
correction, agreement, question, other). Per-message LLM call, one
retry on schema failure.

## 2.3 Flip points + independent signals

Flip = round where majority answer turns from wrong to right.
Four independent per-claim signals (flip_coincidence, novelty,
referenced_later, position) — **no composite score** (spec §5.2).

## 2.4 Type-level ablation (Gate 2)

Sample `single_wrong ∧ debate_right`, span-mask type-t claims,
replay final round. Report Δ_t per type. Hard cap
MAX_NEW_LLM_CALLS=500.

## 2.5 Compression and evaluation

5 methods: b1 full / b2 single-agent / b3 sliding-window /
b5 random-drop / ours (data-driven). Single-synthesizer replay per
(qid, method). Metrics: accuracy, total tokens, acc-per-1k.

# 3. Results

**TODO: insert figure** — claim type distribution by agent × round:
`figs/claim_type_distribution.png`

**TODO: insert figure** — signal correlations with flip:
`figs/signal_correlations.png`

**TODO: insert figure** — Δ_t ranking:
`figs/delta_ranking.png`

**TODO: insert table** — baseline comparison:
include contents of `tables/baselines.tex`

**TODO: insert figure** — accuracy vs tokens Pareto:
`figs/pareto.png`

**TODO: insert table** — claim-type statistics:
include contents of `tables/claim_stats.tex`

# 4. Discussion

**TODO:** Interpret Δ_t ranking. If Gate-2 NULL_RESULT triggered,
pivot to descriptive comparison (spec §5.4). Discuss whether ours
beats b5 and why.

# 5. Limitations

- n=100, single seed, no bootstrap confidence intervals.
- Token counts approximate (char/4 heuristic).
- 10-dialogue manual spot-check only; no NMI/ARI.
- Cluster analysis (HDBSCAN) not in critical path.

# 6. Conclusion

**TODO (~80 words):** Summarize contribution (attribution-first
workflow), main finding (data-supported rule OR honest null),
future work (multi-seed, harder-math tasks).

---

## Reproduction

See the project `README.md`. `make pilot-full && make collect &&
make extract && make analyze && make ablate && make gate2 &&
make evaluate && make report` reproduces every artifact this paper
references.
