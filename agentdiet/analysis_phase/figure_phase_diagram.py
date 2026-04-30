"""Phase diagram scatter renderer (spec §5.5).

X axis: SA accuracy (single-agent capability proxy).
Y axis: Delta = acc(debate) - acc(voting), with vertical CI bars.
Color: thinking on / off.
Marker shape: math (circle) / code (square).
Marker fill: filled if over_budget_factor <= 1.2, open otherwise.
Annotation: short cell label per marker.

matplotlib is lazy-imported; raises ImportError pointing at the
agentdiet[analysis] extras when missing.
"""
from __future__ import annotations

from pathlib import Path

from agentdiet.analysis_phase.bootstrap import CellAnalysis


_MATH_DATASETS = {"gsm8k", "aime"}
_CODE_DATASETS = {"humaneval_plus", "livecodebench"}

_MODEL_FAMILY_CODE = {
    "qwen3": "Q3",
    "gpt-oss": "GO",
    "generic": "GE",
}

_DATASET_CODE = {
    "gsm8k": "GS",
    "aime": "AI",
    "humaneval_plus": "HE",
    "livecodebench": "LC",
}


def short_label(cell: CellAnalysis) -> str:
    """Compress a cell into a 4-6 char annotation, e.g. 'Q3-GS'."""
    fam = _MODEL_FAMILY_CODE.get(cell.model_family, cell.model_family[:2].upper())
    ds = _DATASET_CODE.get(cell.dataset_name, cell.dataset_name[:2].upper())
    label = f"{fam}-{ds}"
    if cell.thinking:
        label += "+"
    if cell.prompt_variant != "cooperative":
        label += f"({cell.prompt_variant[:3]})"
    return label


def _ensure_matplotlib():
    try:
        import matplotlib  # type: ignore  # noqa: F401
        import matplotlib.pyplot as plt  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib not installed; install agentdiet[analysis]"
        ) from exc


def _domain(dataset_name: str) -> str:
    if dataset_name in _MATH_DATASETS:
        return "math"
    if dataset_name in _CODE_DATASETS:
        return "code"
    return "unknown"


def render_phase_diagram(
    cells: list[CellAnalysis],
    output_path: Path,
    *,
    over_budget_threshold: float = 1.2,
    title: str = "Phase diagram: Δ = acc(debate) − acc(voting) vs SA accuracy",
) -> Path:
    """Render the phase diagram and save to output_path.

    Returns the resolved path to the saved figure.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Reference line at Delta = 0
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    for cell in cells:
        x = cell.sa_accuracy
        y = cell.delta_debate_voting
        yerr_low = y - cell.delta_ci_low
        yerr_high = cell.delta_ci_high - y

        color = "C0" if cell.thinking else "C1"
        domain = _domain(cell.dataset_name)
        marker = "o" if domain == "math" else "s"
        is_open = cell.over_budget_factor > over_budget_threshold
        facecolor = "none" if is_open else color

        ax.errorbar(
            x, y,
            yerr=[[max(0, yerr_low)], [max(0, yerr_high)]],
            fmt=marker,
            markersize=8,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=1.2,
        )
        ax.annotate(
            short_label(cell),
            (x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("SA accuracy")
    ax.set_ylabel("Δ accuracy (debate − voting), with 95% CI")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)

    # Legend (manual)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C1",
               markeredgecolor="C1", markersize=8, label="math · thinking off"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0",
               markeredgecolor="C0", markersize=8, label="math · thinking on"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="C1",
               markeredgecolor="C1", markersize=8, label="code · thinking off"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="C0",
               markeredgecolor="C0", markersize=8, label="code · thinking on"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="black", markersize=8,
               label=f"open: over_budget > {over_budget_threshold}"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
