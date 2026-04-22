# sp-evaluator Memory

## Active Patterns

### 2026-04-22 — decision-id-reversal-closure-check

- **Observed in**: type-level-ablation (D1 silently reversed by generator
  with evaluator accepting), corroborated across 10-feature audit where
  `unplanned_rejected: []` every time
- **Rule**: When an entry in `unplanned_changes` references or contradicts
  a plan decision with id `Dn`, the closure_check section MUST either
  (a) re-classify the change as a plan revision that needs user
  confirmation (add it back to `decisions[]` as a new ask_user:true
  question), or (b) cite in `unplanned_accepted` why the contradiction is
  safe (user precedent, scope deferral explicitly agreed, lower risk than
  original plan). Silent acceptance of a decision reversal is a
  closure_check gap, not a PASS.
- **Context**: Applies whenever Generator's `unplanned_changes[].what`
  names a decision id, mentions a plan field, or deviates from an
  `ask_user:false` decision's `planner_view`. Especially important when
  Dn was a research-integrity argument (e.g. subset completeness, sample
  size).
- **Status**: active
- **Last triggered**: type-level-ablation (2026-04-22)

## Archive

(none)
