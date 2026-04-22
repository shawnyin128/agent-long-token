"""Reparse CLI recomputes final_answer in place without calling the LLM."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.cli.reparse import (
    main, reparse_dialogues, reparse_pilot_single, reparse_pilot_debate,
)
from agentdiet.config import Config
from agentdiet.types import Dialogue, Message


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path, hf_cache_dir=tmp_path / "hf",
        model="test-model",
    )


def _write_single_pilot(cfg: Config, qid: str, text: str, gold: str = "66",
                        stored_final: str | None = "16") -> Path:
    """Mimic baseline.run_single_agent output (single message, round 1)."""
    d = Dialogue(
        question_id=qid, question="Q", gold_answer=gold,
        messages=[Message(agent_id=0, round=1, text=text)],
        final_answer=stored_final,
        meta={"method": "single", "model": cfg.model},
    )
    pilot_dir = cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug
    pilot_dir.mkdir(parents=True, exist_ok=True)
    path = pilot_dir / f"{qid}.json"
    path.write_text(d.model_dump_json())
    return path


def _write_debate_pilot(cfg: Config, qid: str, agent_texts: dict[int, str],
                        gold: str = "66",
                        stored_final: str | None = "16",
                        stored_per_agent: dict | None = None) -> Path:
    msgs = [
        Message(agent_id=aid, round=1, text=t)
        for aid, t in sorted(agent_texts.items())
    ]
    d = Dialogue(
        question_id=qid, question="Q", gold_answer=gold, messages=msgs,
        final_answer=stored_final,
        meta={
            "method": "debate", "model": cfg.model,
            "per_agent_final_answers": stored_per_agent or {"0": "16"},
        },
    )
    debate_dir = cfg.artifacts_dir / "pilot" / "debate" / cfg.model_slug
    debate_dir.mkdir(parents=True, exist_ok=True)
    path = debate_dir / f"{qid}.json"
    path.write_text(d.model_dump_json())
    return path


# ---------------------------------------------------------------------------
# reparse_pilot_single
# ---------------------------------------------------------------------------

def test_reparse_single_updates_mis_parsed_final(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    # Text with #### N66 and $16 distractor — old parser stored '16'.
    text = "Each costs $16. Total 66. #### N66"
    path = _write_single_pilot(cfg, "qa", text, stored_final="16")

    counts = reparse_pilot_single(cfg)
    assert counts["visited"] == 1
    assert counts["changed"] == 1
    assert counts["unchanged"] == 0

    updated = json.loads(path.read_text())
    assert updated["final_answer"] == "66"


def test_reparse_single_already_correct_not_touched(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_single_pilot(cfg, "qa", "#### 66", stored_final="66")
    counts = reparse_pilot_single(cfg)
    assert counts["changed"] == 0
    assert counts["unchanged"] == 1


# ---------------------------------------------------------------------------
# reparse_pilot_debate
# ---------------------------------------------------------------------------

def test_reparse_debate_recomputes_majority(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    # 3 agents all write #### N66 (new parser → all 66, majority 66)
    _write_debate_pilot(cfg, "qa", {
        0: "Each book $16. total 66. #### N66",
        1: "Same calc. #### N66",
        2: "$66 total. #### N66",
    }, stored_final="16", stored_per_agent={"0": "16", "1": "16", "2": "66"})

    counts = reparse_pilot_debate(cfg)
    assert counts["changed"] == 1

    data = json.loads(next((cfg.artifacts_dir / "pilot" / "debate" / cfg.model_slug).glob("*.json")).read_text())
    assert data["final_answer"] == "66"
    per_agent = data["meta"]["per_agent_final_answers"]
    assert per_agent == {"0": "66", "1": "66", "2": "66"}


def test_reparse_debate_uses_last_round_only_for_majority(tmp_path):
    """majority_vote is computed on the LAST round of messages, not all."""
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    # Round 1 messages say 10, round 2 messages say 42 — last round wins.
    d = Dialogue(
        question_id="q", question="Q", gold_answer="42",
        messages=[
            Message(agent_id=0, round=1, text="#### 10"),
            Message(agent_id=1, round=1, text="#### 10"),
            Message(agent_id=0, round=2, text="#### 42"),
            Message(agent_id=1, round=2, text="#### 42"),
        ],
        final_answer="10",
    )
    p = cfg.artifacts_dir / "pilot" / "debate" / cfg.model_slug
    p.mkdir(parents=True, exist_ok=True)
    (p / "q.json").write_text(d.model_dump_json())

    reparse_pilot_debate(cfg)
    out = json.loads((p / "q.json").read_text())
    assert out["final_answer"] == "42"


# ---------------------------------------------------------------------------
# reparse_dialogues (full collection artifacts)
# ---------------------------------------------------------------------------

def test_reparse_dialogues_walks_model_slug_dir(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    d = Dialogue(
        question_id="q1", question="Q", gold_answer="42",
        messages=[
            Message(agent_id=a, round=r, text="#### N42")
            for r in (1, 2) for a in (0, 1, 2)
        ],
        final_answer="wrong",
    )
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / "q1.json").write_text(d.model_dump_json())

    counts = reparse_dialogues(cfg)
    assert counts["changed"] == 1
    data = json.loads((cfg.dialogues_dir / "q1.json").read_text())
    assert data["final_answer"] == "42"


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def test_cli_dry_run_does_not_write(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_single_pilot(cfg, "qa", "#### N66", stored_final="16")
    rc = main(["--dry-run"], cfg=cfg)
    assert rc == 0
    # final_answer untouched.
    data = json.loads(next((cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug).glob("*.json")).read_text())
    assert data["final_answer"] == "16"


def test_cli_all_default_reparses_both(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_single_pilot(cfg, "qa", "#### N66", stored_final="16")
    # Also a dialogue.
    d = Dialogue(
        question_id="qb", question="Q", gold_answer="42",
        messages=[Message(agent_id=0, round=1, text="#### N42")],
        final_answer="wrong",
    )
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / "qb.json").write_text(d.model_dump_json())

    rc = main([], cfg=cfg)
    assert rc == 0

    # Both updated.
    assert json.loads(next((cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug).glob("*.json")).read_text())["final_answer"] == "66"
    assert json.loads((cfg.dialogues_dir / "qb.json").read_text())["final_answer"] == "42"


def test_cli_corrupt_file_reported_not_fatal(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _write_single_pilot(cfg, "qa", "#### N66", stored_final="16")
    # Corrupt sibling file.
    bad = cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug / "broken.json"
    bad.write_text("{not json")

    counts = reparse_pilot_single(cfg)
    assert counts["errored"] >= 1
    # Good file still got reparsed.
    assert counts["changed"] == 1
