from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agentdiet.config import Config, get_config
from agentdiet.types import Dialogue


GATE_THRESHOLD_PP = 3.0
EXIT_PASS = 0
EXIT_SOFT_FAIL = 10
EXIT_HARD_FAIL = 20


@dataclass
class MethodStats:
    method: str
    total: int
    answered: int
    correct: int

    @property
    def accuracy(self) -> Optional[float]:
        if self.answered == 0:
            return None
        return self.correct / self.answered

    @property
    def unparsed(self) -> int:
        return self.total - self.answered


def _model_slug(model: str) -> str:
    return model.replace("/", "__")


def _load_dialogue(path: Path) -> Dialogue:
    return Dialogue.model_validate_json(path.read_text(encoding="utf-8"))


def _load_method_stats(cfg: Config, method: str) -> tuple[MethodStats, list[Dialogue]]:
    dir_ = cfg.artifacts_dir / "pilot" / method / _model_slug(cfg.model)
    paths = sorted(dir_.glob("*.json"))
    dialogues = [_load_dialogue(p) for p in paths]
    total = len(dialogues)
    answered = sum(1 for d in dialogues if d.final_answer is not None)
    correct = sum(1 for d in dialogues if d.final_answer is not None and d.final_answer == d.gold_answer)
    return MethodStats(method, total, answered, correct), dialogues


def _pp(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _sample_sections(single_d: list[Dialogue], debate_d: list[Dialogue]) -> list[str]:
    by_qid_s = {d.question_id: d for d in single_d}
    by_qid_d = {d.question_id: d for d in debate_d}
    common = [q for q in by_qid_s if q in by_qid_d]

    def correct(d: Dialogue) -> bool:
        return d.final_answer is not None and d.final_answer == d.gold_answer

    picks: list[tuple[str, str, Dialogue, Dialogue]] = []
    # first passing both
    for q in common:
        if correct(by_qid_s[q]) and correct(by_qid_d[q]):
            picks.append(("both correct", q, by_qid_s[q], by_qid_d[q])); break
    # first single wrong but debate correct (flip)
    for q in common:
        if not correct(by_qid_s[q]) and correct(by_qid_d[q]):
            picks.append(("debate flip (single wrong, debate right)", q, by_qid_s[q], by_qid_d[q])); break
    # first both wrong
    for q in common:
        if not correct(by_qid_s[q]) and not correct(by_qid_d[q]):
            picks.append(("both wrong", q, by_qid_s[q], by_qid_d[q])); break

    sections = []
    for label, qid, ds, dd in picks:
        sections.append(
            f"### Sample — {label} (qid={qid})\n\n"
            f"**Question:** {ds.question}\n\n"
            f"**Gold:** {ds.gold_answer}\n\n"
            f"**Single answer:** {ds.final_answer}\n\n"
            f"**Debate answer:** {dd.final_answer}\n\n"
            f"<details><summary>Single-agent response</summary>\n\n```\n{ds.messages[0].text}\n```\n</details>\n\n"
            f"<details><summary>Debate final round</summary>\n\n"
            + "\n\n".join(f"**Agent {m.agent_id} (round {m.round}):**\n```\n{m.text}\n```"
                          for m in dd.messages if m.round == dd.messages[-1].round)
            + "\n</details>"
        )
    return sections


def build_report(cfg: Config) -> tuple[str, int]:
    s_stats, s_d = _load_method_stats(cfg, "single")
    d_stats, d_d = _load_method_stats(cfg, "debate")

    if s_stats.accuracy is None or d_stats.accuracy is None:
        verdict_line = "HARD FAIL — not enough parsed answers on one or both methods"
        rc = EXIT_HARD_FAIL
        delta_pp = None
    else:
        delta_pp = 100 * (d_stats.accuracy - s_stats.accuracy)
        if delta_pp < 0:
            verdict_line = f"HARD FAIL — debate worse than single ({delta_pp:.1f}pp)"
            rc = EXIT_HARD_FAIL
        elif delta_pp < GATE_THRESHOLD_PP:
            verdict_line = (
                f"SOFT FAIL — delta {delta_pp:.1f}pp < {GATE_THRESHOLD_PP}pp threshold. "
                f"Switch model (AGENTDIET_MODEL=gpt-4o-mini) and re-pilot, or escalate."
            )
            rc = EXIT_SOFT_FAIL
        else:
            verdict_line = f"PASS — delta {delta_pp:.1f}pp >= {GATE_THRESHOLD_PP}pp threshold"
            rc = EXIT_PASS

    header = f"# Pilot Report — Gate 1\n\nModel: `{cfg.model}`  \nN requested (pilot): {s_stats.total}\n\n"
    table = (
        "| Method | n | answered | correct | accuracy | unparsed |\n"
        "|---|---|---|---|---|---|\n"
        f"| single | {s_stats.total} | {s_stats.answered} | {s_stats.correct} | {_pp(s_stats.accuracy)} | {s_stats.unparsed} |\n"
        f"| debate | {d_stats.total} | {d_stats.answered} | {d_stats.correct} | {_pp(d_stats.accuracy)} | {d_stats.unparsed} |\n"
    )
    if delta_pp is not None:
        table += f"\n**Delta (debate − single):** {delta_pp:.1f}pp\n"
    verdict = f"\n## Verdict\n\n**{verdict_line}**\n"
    samples = "\n## Samples\n\n" + "\n\n---\n\n".join(_sample_sections(s_d, d_d)) if s_d and d_d else ""
    return header + table + verdict + samples, rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate-1 pilot decision report")
    parser.add_argument("--report", action="store_true", help="write and print pilot_report.md")
    args = parser.parse_args(argv)

    cfg = get_config()
    cfg.ensure_dirs()
    text, rc = build_report(cfg)
    out = cfg.artifacts_dir / "pilot_report.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    if args.report:
        print(text)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
