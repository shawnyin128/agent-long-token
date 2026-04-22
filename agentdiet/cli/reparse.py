"""Re-parse existing pilot + dialogue artifacts in place.

Use case: parser bug fix (see ``fix-answer-parser-for-qwen-literal-N``)
where ``final_answer`` fields were computed by an older regex. This
CLI walks the artifact tree, reruns ``parse_answer`` + ``majority_vote``
on the stored message texts, and rewrites ``final_answer`` /
``meta.per_agent_final_answers`` atomically. Zero LLM calls — the
message texts themselves don't change.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agentdiet.aggregate import majority_vote
from agentdiet.config import Config, get_config
from agentdiet.dataset import parse_answer
from agentdiet.types import Dialogue


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _reparse_single_dialogue(d: Dialogue) -> tuple[Dialogue, bool]:
    """For a single-message dialogue (baseline), recompute final_answer
    from the one message. Returns (new_dialogue, changed)."""
    if not d.messages:
        return d, False
    new_final = parse_answer(d.messages[0].text)
    if new_final == d.final_answer:
        return d, False
    new = d.model_copy(update={"final_answer": new_final})
    return new, True


def _reparse_debate_dialogue(d: Dialogue) -> tuple[Dialogue, bool]:
    """For a multi-message dialogue, recompute per_agent_final_answers
    (all messages) and final_answer (majority of last round)."""
    if not d.messages:
        return d, False

    rounds = sorted({m.round for m in d.messages})
    last_round = rounds[-1]
    last_msgs = [m for m in d.messages if m.round == last_round]
    new_final, new_per_agent = majority_vote(last_msgs)

    per_agent_str = {str(k): v for k, v in new_per_agent.items()}
    old_meta = dict(d.meta or {})
    old_per_agent = old_meta.get("per_agent_final_answers")
    old_per_agent_str = (
        {str(k): v for k, v in old_per_agent.items()} if old_per_agent else None
    )

    changed = (new_final != d.final_answer) or (old_per_agent_str != per_agent_str)
    if not changed:
        return d, False

    new_meta = dict(old_meta)
    new_meta["per_agent_final_answers"] = per_agent_str
    new = d.model_copy(update={"final_answer": new_final, "meta": new_meta})
    return new, True


def _process_dir(
    directory: Path, *, reparse_fn, dry_run: bool,
) -> dict[str, int]:
    counts = {"visited": 0, "changed": 0, "unchanged": 0, "errored": 0}
    if not directory.is_dir():
        return counts
    for path in sorted(directory.glob("*.json")):
        counts["visited"] += 1
        try:
            raw = path.read_text(encoding="utf-8")
            d = Dialogue.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            counts["errored"] += 1
            print(f"  ERROR {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        new_d, changed = reparse_fn(d)
        if changed and not dry_run:
            _atomic_write(path, new_d.model_dump_json())
        counts["changed" if changed else "unchanged"] += 1
    return counts


def reparse_pilot_single(cfg: Config, *, dry_run: bool = False) -> dict[str, int]:
    directory = cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug
    return _process_dir(directory, reparse_fn=_reparse_single_dialogue, dry_run=dry_run)


def reparse_pilot_debate(cfg: Config, *, dry_run: bool = False) -> dict[str, int]:
    directory = cfg.artifacts_dir / "pilot" / "debate" / cfg.model_slug
    return _process_dir(directory, reparse_fn=_reparse_debate_dialogue, dry_run=dry_run)


def reparse_dialogues(cfg: Config, *, dry_run: bool = False) -> dict[str, int]:
    return _process_dir(
        cfg.dialogues_dir, reparse_fn=_reparse_debate_dialogue, dry_run=dry_run
    )


def _print_counts(label: str, counts: dict[str, int]) -> None:
    print(
        f"  {label:24s} visited={counts['visited']:4d} "
        f"changed={counts['changed']:4d} "
        f"unchanged={counts['unchanged']:4d} "
        f"errored={counts['errored']:4d}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None, *, cfg: Config | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-parse final_answer fields in existing artifacts "
                    "(no LLM calls)"
    )
    parser.add_argument("--pilot", action="store_true",
                        help="Re-parse pilot single + debate only")
    parser.add_argument("--dialogues", action="store_true",
                        help="Re-parse dialogues only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts without writing changes")
    args = parser.parse_args(argv)

    if cfg is None:
        cfg = get_config()
    cfg.ensure_dirs()

    do_pilot = args.pilot or not args.dialogues        # default: all
    do_dialogues = args.dialogues or not args.pilot

    print(
        f"reparse model={cfg.model} dry_run={args.dry_run}",
        file=sys.stderr,
    )

    if do_pilot:
        c = reparse_pilot_single(cfg, dry_run=args.dry_run)
        _print_counts("pilot/single", c)
        c = reparse_pilot_debate(cfg, dry_run=args.dry_run)
        _print_counts("pilot/debate", c)

    if do_dialogues:
        c = reparse_dialogues(cfg, dry_run=args.dry_run)
        _print_counts("dialogues", c)

    print("next: make gate (to re-run Gate-1 decision with corrected final_answers)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
