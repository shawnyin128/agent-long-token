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

- **4 pre-existing test failures in RQ0 subsystems** (surfaced by
  hygiene scan 2026-04-27, not regressions): tolerate_latex_escapes
  ::test_fixes_latex_command_{frac,times} (regex edge case);
  type_level_ablation::test_reconstruct_emits_empty_string_for_
  fully_masked_message (expects span-level, default switched to
  message-level in f7e5315); type_level_ablation::test_different_
  drop_types_produce_fresh_calls (cache-key invariant under
  message-granularity). Undecided whether to fix — RQ0 pipeline
  may be retired entirely under phase-mapping pivot.

- **LCB cell yields 0/40 across all conditions (2026-05-03).**
  gpt-oss:livecodebench:t0 just landed (commit 63784dd) with
  SA=V=D=0.000 despite 55k/945k/741k tokens generated. Either the
  hf_hub_download loader pulled rows whose test_cases decode/templating
  is wrong, or _extract_code mis-parses gpt-oss's responses, or the
  SubprocessJudge harness runs the test scripts incorrectly. Need to
  inspect sa.json + one hidden_test script + raw model response
  before running any more LCB cells (qwen3 LCB and gpt-oss LCB t1
  blocked on this triage).
