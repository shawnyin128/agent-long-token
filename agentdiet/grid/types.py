"""On-disk schema for one phase-mapping grid cell.

Naming: cell_dir = "{model_slug}__{dataset_name}__t{thinking_int}"
where model_slug is HF id with "/" -> "__" (matches Config.model_slug).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


ConditionName = Literal["sa", "voting", "debate"]


@dataclass(frozen=True)
class CellSpec:
    """Identifies one (model, dataset, thinking, prompt_variant) cell.

    prompt_variant defaults to "cooperative" so existing CellSpec callers
    are unaffected; the cell_dir naming function preserves the legacy
    format when prompt_variant=="cooperative" so main-grid artifacts
    keep their on-disk paths.
    """
    model: str           # full HF id, e.g. "Qwen/Qwen3-30B-A3B"
    model_family: str    # "qwen3" | "gpt-oss" | "generic"
    dataset_name: str    # "gsm8k" | "aime" | "humaneval_plus" | "livecodebench"
    thinking: bool
    prompt_variant: str = "cooperative"

    @property
    def model_slug(self) -> str:
        return self.model.replace("/", "__")


def cell_dir(cell: CellSpec) -> str:
    base = f"{cell.model_slug}__{cell.dataset_name}__t{int(cell.thinking)}"
    if cell.prompt_variant == "cooperative":
        return base
    return f"{base}__pv-{cell.prompt_variant}"


@dataclass(frozen=True)
class QuestionResult:
    qid: str
    gold: str
    final_answer: Optional[str]
    correct: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionRecord:
    condition: ConditionName
    cell: CellSpec
    questions: list[QuestionResult]
    n_evaluated: int
    accuracy: float
    total_tokens: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CellSummary:
    cell: CellSpec
    sa_accuracy: float
    voting_accuracy: float
    debate_accuracy: float
    sa_total_tokens: int
    voting_total_tokens: int
    debate_total_tokens: int
    delta_debate_voting: float
    delta_debate_sa: float
    calibration: dict[str, Any]
    n_questions: int


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def save_record(path: Path, record: ConditionRecord) -> None:
    payload = dataclasses.asdict(record)
    _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False))


def load_record(path: Path) -> ConditionRecord:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cell = CellSpec(**raw["cell"])
    questions = [QuestionResult(**q) for q in raw["questions"]]
    return ConditionRecord(
        condition=raw["condition"],
        cell=cell,
        questions=questions,
        n_evaluated=raw["n_evaluated"],
        accuracy=raw["accuracy"],
        total_tokens=raw["total_tokens"],
        meta=raw.get("meta", {}),
    )


def save_summary(path: Path, summary: CellSummary) -> None:
    payload = dataclasses.asdict(summary)
    _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False))


def load_summary(path: Path) -> CellSummary:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cell = CellSpec(**raw["cell"])
    return CellSummary(
        cell=cell,
        sa_accuracy=raw["sa_accuracy"],
        voting_accuracy=raw["voting_accuracy"],
        debate_accuracy=raw["debate_accuracy"],
        sa_total_tokens=raw["sa_total_tokens"],
        voting_total_tokens=raw["voting_total_tokens"],
        debate_total_tokens=raw["debate_total_tokens"],
        delta_debate_voting=raw["delta_debate_voting"],
        delta_debate_sa=raw["delta_debate_sa"],
        calibration=raw["calibration"],
        n_questions=raw["n_questions"],
    )
