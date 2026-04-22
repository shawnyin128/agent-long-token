from __future__ import annotations

import subprocess
import sys

import pytest

from agentdiet.dataset import Question
from agentdiet.debate import run_debate
from agentdiet.llm_client import DummyBackend, LLMClient
from agentdiet.types import Dialogue


pytestmark = [pytest.mark.smoke, pytest.mark.timeout(30)]


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    import openai

    class Guard:
        def __init__(self, *a, **k):
            raise RuntimeError("Smoke tests must not hit the network")

    monkeypatch.setattr(openai, "OpenAI", Guard)


def test_2x1_minimal_schema_valid(tmp_path):
    backend = DummyBackend(lambda msgs, m, t: "reasoning. #### 42")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    q = Question(qid="q1", question="What is 6 * 7?", gold_answer="42")

    d = run_debate(q, client, model="m", n_agents=2, n_rounds=1)

    # schema validation: must be a Dialogue and roundtrip via JSON
    assert isinstance(d, Dialogue)
    s = d.model_dump_json()
    d2 = Dialogue.model_validate_json(s)
    assert d2 == d
    assert d.final_answer == "42"


def test_dialogue_bytes_roundtrip(tmp_path):
    backend = DummyBackend(lambda msgs, m, t: "#### 42")
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    q = Question(qid="q1", question="Q", gold_answer="42")
    d = run_debate(q, client, model="m", n_agents=2, n_rounds=1)

    out = tmp_path / "d.json"
    out.write_text(d.model_dump_json(), encoding="utf-8")
    # Strip the time-varying timestamp before comparing
    d_loaded = Dialogue.model_validate_json(out.read_text(encoding="utf-8"))
    d_norm = d.model_copy(update={"meta": {**d.meta, "timestamp": 0}})
    d_loaded_norm = d_loaded.model_copy(update={"meta": {**d_loaded.meta, "timestamp": 0}})
    assert d_norm == d_loaded_norm


def test_cli_dry_run_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDIET_ARTIFACTS_DIR", str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "agentdiet.cli.collect", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "dry-run OK" in result.stdout


def test_cli_dry_run_in_process_for_coverage(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENTDIET_ARTIFACTS_DIR", str(tmp_path))
    from agentdiet.cli.collect import main

    rc = main(["--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "dry-run OK" in captured.out


def test_cli_no_dry_run_exits_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENTDIET_ARTIFACTS_DIR", str(tmp_path))
    from agentdiet.cli.collect import main

    rc = main([])
    captured = capsys.readouterr()
    assert rc != 0
    assert "not yet implemented" in captured.err
