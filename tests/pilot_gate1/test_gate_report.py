from __future__ import annotations

from pathlib import Path

from agentdiet.cli.gate import (
    EXIT_HARD_FAIL,
    EXIT_PASS,
    EXIT_SOFT_FAIL,
    build_report,
)
from agentdiet.config import Config
from agentdiet.types import Dialogue, Message


def _write_d(path: Path, qid: str, gold: str, final: str | None, method: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = Dialogue(
        question_id=qid,
        question="Q",
        gold_answer=gold,
        messages=[Message(agent_id=0, round=1, text=f"... #### {final or 'x'}")],
        final_answer=final,
        meta={"method": method, "model": "test-model"},
    )
    path.write_text(d.model_dump_json(), encoding="utf-8")


def _make_cfg(tmp_path: Path) -> Config:
    return Config(artifacts_dir=tmp_path, hf_cache_dir=tmp_path / "hf", model="test-model")


def _seed_pilot(tmp_path: Path, single_correct: int, debate_correct: int, n: int = 10):
    cfg = _make_cfg(tmp_path)
    s_dir = tmp_path / "pilot" / "single" / "test-model"
    d_dir = tmp_path / "pilot" / "debate" / "test-model"
    for i in range(n):
        qid = f"q{i}"
        _write_d(s_dir / f"{qid}.json", qid, gold="1", final="1" if i < single_correct else "2", method="single")
        _write_d(d_dir / f"{qid}.json", qid, gold="1", final="1" if i < debate_correct else "2", method="debate")
    return cfg


def test_pass_above_threshold(tmp_path):
    cfg = _seed_pilot(tmp_path, single_correct=3, debate_correct=7)
    text, rc = build_report(cfg)
    assert rc == EXIT_PASS
    assert "PASS" in text
    assert "40.0pp" in text  # delta


def test_soft_fail_below_threshold(tmp_path):
    cfg = _seed_pilot(tmp_path, single_correct=5, debate_correct=5)
    text, rc = build_report(cfg)
    assert rc == EXIT_SOFT_FAIL
    assert "SOFT FAIL" in text


def test_pass_with_large_n(tmp_path):
    cfg = _seed_pilot(tmp_path, single_correct=50, debate_correct=55, n=100)
    _text, rc = build_report(cfg)
    assert rc == EXIT_PASS  # 5pp >= 3pp threshold


def test_soft_fail_at_edge(tmp_path):
    cfg = _seed_pilot(tmp_path, single_correct=50, debate_correct=52, n=100)
    text, rc = build_report(cfg)
    assert rc == EXIT_SOFT_FAIL  # 2pp < 3pp
    assert "2.0pp" in text


def test_hard_fail_negative_delta(tmp_path):
    cfg = _seed_pilot(tmp_path, single_correct=7, debate_correct=3)
    text, rc = build_report(cfg)
    assert rc == EXIT_HARD_FAIL
    assert "HARD FAIL" in text


def test_unparsed_excluded_from_accuracy(tmp_path):
    cfg = _make_cfg(tmp_path)
    s_dir = tmp_path / "pilot" / "single" / "test-model"
    d_dir = tmp_path / "pilot" / "debate" / "test-model"
    # 5 single parsed (3 correct), 5 unparsed
    for i in range(5):
        _write_d(s_dir / f"q{i}.json", f"q{i}", "1", "1" if i < 3 else "2", "single")
    for i in range(5, 10):
        _write_d(s_dir / f"q{i}.json", f"q{i}", "1", None, "single")
    # 10 debate, 10 correct
    for i in range(10):
        _write_d(d_dir / f"q{i}.json", f"q{i}", "1", "1", "debate")
    text, rc = build_report(cfg)
    # single acc = 3/5 = 60%, debate acc = 10/10 = 100%, delta = 40pp -> PASS
    assert rc == EXIT_PASS
    assert "60.0%" in text
    assert "100.0%" in text


def test_report_includes_table_and_samples(tmp_path):
    cfg = _seed_pilot(tmp_path, single_correct=3, debate_correct=7)
    text, _ = build_report(cfg)
    assert "| Method |" in text
    assert "## Verdict" in text
    assert "## Samples" in text


def test_hard_fail_when_all_unparsed(tmp_path):
    cfg = _make_cfg(tmp_path)
    s_dir = tmp_path / "pilot" / "single" / "test-model"
    d_dir = tmp_path / "pilot" / "debate" / "test-model"
    for i in range(5):
        _write_d(s_dir / f"q{i}.json", f"q{i}", "1", None, "single")
        _write_d(d_dir / f"q{i}.json", f"q{i}", "1", None, "debate")
    _text, rc = build_report(cfg)
    assert rc == EXIT_HARD_FAIL
