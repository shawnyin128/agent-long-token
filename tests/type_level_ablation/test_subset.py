from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.analysis.ablate import (
    is_single_wrong_debate_right,
    load_dialogue,
    load_dialogue_and_claims,
    select_subset,
)
from agentdiet.config import Config
from agentdiet.types import Dialogue, Message


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path, hf_cache_dir=tmp_path / "hf",
        model="test-model", n_agents=3, n_rounds=2,
        seed=42,
    )


def _single_doc(qid: str, final: str, gold: str = "7") -> dict:
    d = Dialogue(
        question_id=qid, question="Q", gold_answer=gold,
        messages=[Message(agent_id=0, round=1, text=f"ans #### {final}")],
        final_answer=final,
    )
    return json.loads(d.model_dump_json())


def _debate_dialogue(qid: str, final: str, gold: str = "7") -> Dialogue:
    return Dialogue(
        question_id=qid, question="Q", gold_answer=gold,
        messages=[
            Message(agent_id=a, round=r, text=f"#### {final}")
            for r in (1, 2) for a in (0, 1, 2)
        ],
        final_answer=final,
    )


def test_predicate_single_wrong_debate_right():
    single = _single_doc("qa", final="3", gold="7")
    debate = _debate_dialogue("qa", final="7", gold="7")
    assert is_single_wrong_debate_right(single, debate, gold="7") is True


def test_predicate_single_right_false():
    single = _single_doc("qa", final="7")
    debate = _debate_dialogue("qa", final="7")
    assert is_single_wrong_debate_right(single, debate, gold="7") is False


def test_predicate_debate_wrong_false():
    single = _single_doc("qa", final="3")
    debate = _debate_dialogue("qa", final="3")
    assert is_single_wrong_debate_right(single, debate, gold="7") is False


def test_predicate_tolerates_missing_final_answers():
    single = _single_doc("qa", final="3")
    single["final_answer"] = None
    debate = _debate_dialogue("qa", final="7")
    # No single answer -> cannot assert single_wrong -> False
    assert is_single_wrong_debate_right(single, debate, gold="7") is False


def _seed_triplet(cfg: Config, qid: str, *, single: str, debate: str, gold: str = "7") -> None:
    # Dialogue
    d = _debate_dialogue(qid, final=debate, gold=gold)
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / f"{qid}.json").write_text(d.model_dump_json())
    # Claims
    claim = {
        "id": f"{qid}_r1_a0_c0", "text": "x",
        "agent_id": 0, "round": 1, "type": "proposal",
        "source_message_span": [0, 3],
    }
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    (cfg.claims_dir / f"{qid}.json").write_text(json.dumps({
        "qid": qid, "claims": [claim], "per_message_status": [],
        "extraction_failed": False,
    }))
    # Pilot single artifact
    pilot_dir = cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / f"{qid}.json").write_text(json.dumps(_single_doc(qid, final=single, gold=gold)))


def test_select_subset_deterministic_seeded(tmp_path):
    cfg_a = _cfg(tmp_path / "a")
    cfg_b = _cfg(tmp_path / "b")
    cfg_a.ensure_dirs()
    cfg_b.ensure_dirs()
    for qid in ("q1", "q2", "q3", "q4", "q5"):
        _seed_triplet(cfg_a, qid, single="3", debate="7")
        _seed_triplet(cfg_b, qid, single="3", debate="7")

    a = select_subset(cfg_a, target_size=3)
    b = select_subset(cfg_b, target_size=3)
    assert a == b
    assert len(a) == 3
    assert set(a).issubset({"q1", "q2", "q3", "q4", "q5"})


def test_select_subset_caps_at_target_size(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    for qid in ("q1", "q2", "q3", "q4"):
        _seed_triplet(cfg, qid, single="3", debate="7")
    picked = select_subset(cfg, target_size=2)
    assert len(picked) == 2


def test_select_subset_skips_non_eligible(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_triplet(cfg, "q1", single="3", debate="7")   # eligible
    _seed_triplet(cfg, "q2", single="7", debate="7")   # single_right -> skip
    _seed_triplet(cfg, "q3", single="3", debate="3")   # debate_wrong -> skip

    picked = select_subset(cfg, target_size=10)
    assert picked == ["q1"]


def test_select_subset_skips_qid_missing_claims(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_triplet(cfg, "q1", single="3", debate="7")
    # Remove claim artifact for q1
    (cfg.claims_dir / "q1.json").unlink()
    picked = select_subset(cfg, target_size=10)
    assert picked == []


def test_load_dialogue_and_claims_returns_tuple(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_triplet(cfg, "q1", single="3", debate="7")
    d, cd = load_dialogue_and_claims(cfg, "q1")
    assert d.question_id == "q1"
    assert cd["qid"] == "q1"
    assert cd["claims"][0]["type"] == "proposal"


def test_load_dialogue_and_claims_raises_when_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    with pytest.raises(FileNotFoundError):
        load_dialogue_and_claims(cfg, "nope")


def _seed_dialogue_only(cfg: Config, qid: str, *, single: str, debate: str, gold: str = "7") -> None:
    """Seed dialogue + pilot single but NO claims artifact."""
    d = _debate_dialogue(qid, final=debate, gold=gold)
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / f"{qid}.json").write_text(d.model_dump_json())
    pilot_dir = cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / f"{qid}.json").write_text(json.dumps(_single_doc(qid, final=single, gold=gold)))


def test_select_subset_no_claims_required_finds_eligible(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_dialogue_only(cfg, "q1", single="3", debate="7")
    _seed_dialogue_only(cfg, "q2", single="3", debate="7")
    picked = select_subset(cfg, target_size=10, require_claims=False)
    assert set(picked) == {"q1", "q2"}


def test_select_subset_no_claims_still_filters_non_eligible(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_dialogue_only(cfg, "q1", single="3", debate="7")   # eligible
    _seed_dialogue_only(cfg, "q2", single="7", debate="7")   # single right -> skip
    picked = select_subset(cfg, target_size=10, require_claims=False)
    assert picked == ["q1"]


def test_load_dialogue_returns_dialogue(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_dialogue_only(cfg, "q1", single="3", debate="7")
    d = load_dialogue(cfg, "q1")
    assert d.question_id == "q1"


def test_load_dialogue_raises_when_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    with pytest.raises(FileNotFoundError):
        load_dialogue(cfg, "nope")
