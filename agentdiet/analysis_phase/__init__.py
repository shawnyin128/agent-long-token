"""Phase-mapping analysis: bootstrap CIs, phase diagram, characterizations.

Reads per-cell artifacts under artifacts/grid/{cell_dir}/ written by
agentdiet.cli.grid + agentdiet.grid.orchestrator. Does not call any LLM
or judge — pure post-hoc analysis.
"""
from agentdiet.analysis_phase.bootstrap import (
    BootstrapResult,
    CellAnalysis,
    compute_per_cell_analysis,
    load_cell_summary,
    paired_bootstrap_delta,
)

__all__ = [
    "BootstrapResult",
    "CellAnalysis",
    "compute_per_cell_analysis",
    "load_cell_summary",
    "paired_bootstrap_delta",
]
