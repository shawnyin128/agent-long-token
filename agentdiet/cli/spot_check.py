"""Claim-extraction spot check scaffold.

Samples K dialogues (seeded) that already have matching claim artifacts
and writes:

  * ``artifacts/spot_check.csv`` — one row per claim with blank
    ``manual_pass`` and ``notes`` columns for the human reviewer.
  * ``artifacts/spot_check_notes.md`` — a companion that prints each
    sampled dialogue side-by-side with its claims for faster reading.

Usage (typically invoked via ``make spot-check``)::

    python -m agentdiet.cli.spot_check --k 10
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from agentdiet.config import Config, get_config
from agentdiet.types import Dialogue


SPOT_CHECK_COLUMNS = (
    "qid",
    "message_idx",
    "agent_id",
    "round",
    "claim_id",
    "type",
    "text",
    "span_start",
    "span_end",
    "quoted_source",
    "manual_pass",
    "notes",
)


def _eligible_qids(cfg: Config) -> list[str]:
    claim_qids = {p.stem for p in cfg.claims_dir.glob("*.json")}
    dialogue_qids = {p.stem for p in cfg.dialogues_dir.glob("*.json")}
    return sorted(claim_qids & dialogue_qids)


def _load_dialogue(cfg: Config, qid: str) -> Dialogue:
    path = cfg.dialogues_dir / f"{qid}.json"
    return Dialogue.model_validate_json(path.read_text(encoding="utf-8"))


def _load_claims(cfg: Config, qid: str) -> dict:
    path = cfg.claims_dir / f"{qid}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _message_index_by_key(dialogue: Dialogue) -> dict[tuple[int, int], int]:
    return {(m.agent_id, m.round): i for i, m in enumerate(dialogue.messages)}


def sample_and_write(cfg: Config, *, k: int) -> Path:
    qids = _eligible_qids(cfg)
    if len(qids) < k:
        raise ValueError(
            f"only {len(qids)} qids have both dialogue + claim artifacts, need {k}"
        )
    rng = random.Random(cfg.seed)
    sampled = sorted(rng.sample(qids, k))

    csv_rows: list[dict[str, str]] = []
    md_blocks: list[str] = []
    for qid in sampled:
        dialogue = _load_dialogue(cfg, qid)
        claim_doc = _load_claims(cfg, qid)
        idx_by_key = _message_index_by_key(dialogue)

        claim_lines: list[str] = []
        for c in claim_doc.get("claims", []):
            agent_id = int(c["agent_id"])
            round_ = int(c["round"])
            msg_idx = idx_by_key.get((agent_id, round_))
            span = c["source_message_span"]
            span_start = int(span[0])
            span_end = int(span[1])
            if msg_idx is not None:
                quoted = dialogue.messages[msg_idx].text[span_start:span_end]
            else:
                quoted = ""
            csv_rows.append({
                "qid": qid,
                "message_idx": str(msg_idx) if msg_idx is not None else "",
                "agent_id": str(agent_id),
                "round": str(round_),
                "claim_id": c["id"],
                "type": c["type"],
                "text": c["text"],
                "span_start": str(span_start),
                "span_end": str(span_end),
                "quoted_source": quoted,
                "manual_pass": "",
                "notes": "",
            })
            claim_lines.append(
                f"  - [{c['type']}] {c['text']}  ← {quoted!r}"
            )

        md_block = [f"## {qid}", f"**Question:** {dialogue.question}", ""]
        md_block.append(f"**Gold answer:** {dialogue.gold_answer}   "
                        f"**Final:** {dialogue.final_answer}")
        md_block.append("")
        for i, msg in enumerate(dialogue.messages):
            md_block.append(f"### Message {i} (agent {msg.agent_id}, round {msg.round})")
            md_block.append("```")
            md_block.append(msg.text)
            md_block.append("```")
        md_block.append("")
        md_block.append("**Extracted claims:**")
        md_block.extend(claim_lines or ["  (none)"])
        md_block.append("")
        md_blocks.append("\n".join(md_block))

    csv_path = cfg.artifacts_dir / "spot_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SPOT_CHECK_COLUMNS))
        writer.writeheader()
        writer.writerows(csv_rows)

    md_path = cfg.artifacts_dir / "spot_check_notes.md"
    md_header = (
        "# Claim-extraction spot check\n\n"
        f"Sampled {k} dialogues with seed={cfg.seed} from "
        f"{len(qids)} eligible qids. Fill `manual_pass` (yes/no) and "
        "`notes` in `spot_check.csv`.\n"
    )
    md_path.write_text(md_header + "\n---\n\n".join(md_blocks), encoding="utf-8")

    return csv_path


def main(argv: list[str] | None = None, *, cfg: Config | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sample K dialogues for manual claim spot-check"
    )
    parser.add_argument("--k", type=int, default=10,
                        help="Number of dialogues to sample (default 10)")
    args = parser.parse_args(argv)

    if cfg is None:
        cfg = get_config()
    cfg.ensure_dirs()

    try:
        csv_path = sample_and_write(cfg, k=args.k)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"wrote {csv_path}", file=sys.stderr)
    print(f"wrote {cfg.artifacts_dir / 'spot_check_notes.md'}", file=sys.stderr)
    print("next: open spot_check_notes.md, read each dialogue, "
          "fill manual_pass in spot_check.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
