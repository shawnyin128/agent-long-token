# Session Memory (short-term, pre-triage)

<!--
SCOPE: undecided observations only (bug/hypothesis/concern/note).
Decided items go elsewhere: ideas → todos.json, fixes → features.json
(fix_feature), patterns → sp-feedback, design → docs/design-docs/.

TRIAGE BEFORE ADDING:
  1. git log --since="<ts>" -- <file> — resolved?
  2. grep todos.json / features.json — tracked?
  3. Still undecided? Keep. Else remove.

HARD RULE: no duplication with todos.json / features.json / agent-memory.
Keep under 30 lines.
-->

## Observations

<!-- Confirmed findings (Gate-1 +6.7pp; Gate-2 null; control Δ=0;
cooperative not adversarial) are codified in
docs/reports/check-in-2026-04-22.tex (§5) and don't need to live
here. Memory holds only in-flight state. -->

## In-flight

- **Investigating (L1):** does "debate gain = voting, not dialogue"
  replicate cross-model? First model meta-llama/Llama-3.1-8B-Instruct.
  Possible second GPT-OSS-20B (deferred). Decisions: serial L1→L2;
  minimal-pipeline L1 (skip extract); thresholds per check-in §7
  (control Δ ≤ 0.03 confirm; ≥ 0.10 reject).

- **HPC flow — pull and run L1:** make stop → export AGENTDIET_MODEL=
  meta-llama/Llama-3.1-8B-Instruct → make serve → make health →
  git pull → make pilot --n 100 → make gate → make collect (cache
  hits, ~5-10 min IO) → make ablate-control → paste numbers back.
  Then decide L2.

- **L2 deferred decision:** prompt-variant choice (A "skeptic must
  disagree" / B "synth must ask" / C "uniform role" / D "A+B").
  Re-decide once L1 result lands.
