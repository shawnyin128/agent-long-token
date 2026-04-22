# sp-planner Memory

## Active Patterns

### 2026-04-22 — late-phase-ask-user-collapse

- **Observed in**: claim-extraction (D4 CSV vs CSV+markdown shipped as
  ask_user:false @ 72%), type-level-ablation (D3 Gate-2 threshold
  values shipped as ask_user:false @ 70%), plus pattern across
  features 5-10 having 0 ask_user:true vs features 1-4 having 4
- **Rule**: Late-in-project (analysis/report) features still have
  user-visible policy choices — thresholds, output schemas, file
  formats, column sets. Before marking `ask_user: false` with
  confidence > 70 on a decision that introduces a NEW threshold,
  NEW file format, or NEW output schema (not a direct derivative of
  spec wording), reconsider asking. Low-confidence unasked decisions
  are worse than asked ones because they bypass user review silently
  and pollute the audit trail.
- **Context**: Applies to analysis-phase and report-phase features.
  Infrastructure features usually have genuinely high-confidence
  defaults (paths, seeds, cache semantics). Analysis features
  introduce new artifacts whose shape is a research decision, not a
  wiring decision.
- **Status**: active
- **Last triggered**: type-level-ablation (2026-04-22)

## Archive

(none)
