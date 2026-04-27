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

- **L1 Llama-3.1-8B replication retired (2026-04-27).** Pivoting
  directly to RQ1 phase-mapping chain (Qwen3-30B-A3B + GPT-OSS-20B
  with thinking axis) per docs/design-docs/2026-04-27-debate-phase-
  mapping-design.md. cross-model-grid is the next active feature.
