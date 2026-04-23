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

- [2026-04-22] [finding] Qwen 100-dialogue claim-type distribution
  confirmed real (not a prompt bug): proposal 44% + evidence 39% +
  agreement 15% = 98%; correction 0.3% (9), question 0.0% (0),
  other 1.5%. Skeptic-correction-keyword sanity on 20 dialogues:
  1/20, and that 1 is "all steps appear to be correct" — not a
  correction. Interpretation: Qwen 3×3 debate on GSM8K is
  cooperative, not adversarial; simple arithmetic leaves little
  room to disagree. Impact on report: (a) Gate-2 Δ_t for
  correction/question will be undefined/unreliable (N too small);
  spec §5.4 null-result path applies for those types; (b) main
  attribution story will hinge on proposal/evidence/agreement;
  (c) a legitimate finding to report — "3×3 cooperative not
  adversarial on GSM8K" — rather than a bug.

- [2026-04-22] [finding-confirmed] Gate-2 type-level ablation: Δ=0.000
  for all 6 types on n=9 `single_wrong ∧ debate_right` subset even
  under message-level masking (whole messages blanked). Sanity check
  on gsm8k-test-1034 confirms mask is real: rounds 1-2 fully blanked,
  only agent-1 round-2 survived, round-3 replay still produces correct
  answer via 3-agent majority vote. Interpretation: Qwen2.5-7B +
  GSM8K + 3-agent debate → +6.7pp gain comes from majority voting
  over 3 independent re-solves (each agent re-reads the question),
  NOT from information transfer between agents. spec §5.4 null-result
  path applies. Report main thesis should be "debate is voting, not
  dialogue" on simple arithmetic tasks.

- [2026-04-22] [finding-confirmed-via-control] Control experiment
  resolved the Gate-2 concern: blanking ALL messages in rounds 1..N-1
  (keep only question) on the same n=9 subset gives acc(without)=1.000
  matching acc(with)=1.000, Δ=+0.000. No claim type matters because
  no claim CONTENT matters — debate gain is ensembling+voting over
  3 independent re-solves. Spec §1.2 RQ#2 and RQ#3 answer: "no type
  is causally important on this task". Spec §1.2 RQ#4 (compression)
  remains open and now has a sharper prior: any compression
  preserving 3-agent voting structure should recover ~full accuracy.
  Day-3 Pareto will test this.

## In-flight

<!-- Replaced as investigation progresses. -->
