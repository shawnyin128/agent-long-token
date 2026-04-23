from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdiet.cli.ablate import _manifest_path, run_ablation_cli
from agentdiet.config import Config
from agentdiet.llm_client import DummyBackend
from agentdiet.types import Dialogue, Message


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path, hf_cache_dir=tmp_path / "hf",
        model="test-model", n_agents=2, n_rounds=2, seed=42,
    )


def _seed_eligible(cfg: Config, qid: str) -> None:
    d = Dialogue(
        question_id=qid, question="Q", gold_answer="7",
        messages=[
            Message(agent_id=0, round=1, text="agent 0 r1 reasoning"),
            Message(agent_id=1, round=1, text="agent 1 r1 I agree"),
            Message(agent_id=0, round=2, text="#### 7"),
            Message(agent_id=1, round=2, text="#### 7"),
        ],
        final_answer="7",
    )
    cfg.dialogues_dir.mkdir(parents=True, exist_ok=True)
    (cfg.dialogues_dir / f"{qid}.json").write_text(d.model_dump_json())

    claims = [
        {"id": f"{qid}_r1_a1_c0", "text": "agrees",
         "agent_id": 1, "round": 1, "type": "agreement",
         "source_message_span": [12, 19]},   # "I agree"
    ]
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    (cfg.claims_dir / f"{qid}.json").write_text(json.dumps({
        "qid": qid, "claims": claims, "per_message_status": [],
        "extraction_failed": False,
    }))

    # single_wrong ∧ debate_right: single returns "3", gold "7".
    pilot_dir = cfg.artifacts_dir / "pilot" / "single" / cfg.model_slug
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / f"{qid}.json").write_text(json.dumps({
        "question_id": qid, "question": "Q", "gold_answer": "7",
        "messages": [{"agent_id": 0, "round": 1, "text": "#### 3"}],
        "final_answer": "3", "meta": {},
    }))


def test_manifest_records_granularity(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_eligible(cfg, "qa")
    backend = DummyBackend(lambda m, mo, t: "#### 7")
    run_ablation_cli(
        cfg=cfg, target_size=1, max_new_llm_calls=500, backend=backend,
        granularity="message",
    )
    m = json.loads(_manifest_path(cfg).read_text())
    assert m["granularity"] == "message"


def test_message_vs_span_produce_different_prompts(tmp_path):
    """Same fixture, different granularity → different cache keys →
    backend sees different payloads for the 'agreement' drop_type."""
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_eligible(cfg, "qa")

    seen_prompts = []

    def capture(messages, model, temperature):
        # Capture the final-round user message (the one containing the
        # masked prior-round transcript).
        seen_prompts.append(messages[-1]["content"])
        return "#### 7"

    # First, message-level.
    backend_m = DummyBackend(capture)
    run_ablation_cli(
        cfg=cfg, target_size=1, max_new_llm_calls=500, backend=backend_m,
        granularity="message",
    )
    message_mode_prompts = list(seen_prompts)
    seen_prompts.clear()

    # Now span-level.
    backend_s = DummyBackend(capture)
    run_ablation_cli(
        cfg=cfg, target_size=1, max_new_llm_calls=500, backend=backend_s,
        granularity="span",
    )
    span_mode_prompts = list(seen_prompts)

    # For drop_type=agreement, the agent-1 round-1 text differs between
    # modes — whole blanked vs just 'I agree' removed.
    assert any(
        m_prompt != s_prompt
        for m_prompt, s_prompt in zip(message_mode_prompts, span_mode_prompts)
    )


def test_default_granularity_is_message_in_cli(tmp_path):
    from agentdiet.cli.ablate import main
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_eligible(cfg, "qa")
    backend = DummyBackend(lambda m, mo, t: "#### 7")
    # No --granularity flag: should default to 'message'.
    rc = main(["--n", "1"], cfg=cfg, backend=backend)
    assert rc == 0
    m = json.loads(_manifest_path(cfg).read_text())
    assert m["granularity"] == "message"


def test_cli_explicit_span_flag(tmp_path):
    from agentdiet.cli.ablate import main
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    _seed_eligible(cfg, "qa")
    backend = DummyBackend(lambda m, mo, t: "#### 7")
    rc = main(["--n", "1", "--granularity", "span"], cfg=cfg, backend=backend)
    assert rc == 0
    m = json.loads(_manifest_path(cfg).read_text())
    assert m["granularity"] == "span"
