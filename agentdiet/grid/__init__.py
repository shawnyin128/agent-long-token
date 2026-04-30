"""Per-cell grid execution: types, runners, orchestrator.

A "cell" is one (model, dataset, thinking) point in the phase-mapping
grid. Each cell produces four JSON artifacts:
  - sa.json — single-agent results
  - voting.json — token-matched majority-voting results
  - debate.json — 3x3 debate results
  - summary.json — aggregated accuracies, deltas, and calibration
"""
from agentdiet.grid.types import (
    CellSpec,
    CellSummary,
    ConditionRecord,
    QuestionResult,
    cell_dir,
    load_record,
    load_summary,
    save_record,
    save_summary,
)

__all__ = [
    "CellSpec",
    "CellSummary",
    "ConditionRecord",
    "QuestionResult",
    "cell_dir",
    "load_record",
    "load_summary",
    "save_record",
    "save_summary",
]
