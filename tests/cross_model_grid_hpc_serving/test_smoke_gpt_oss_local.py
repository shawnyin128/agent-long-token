"""Local lint-style tests for the GPT-OSS smoke runner."""
from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path

import pytest

from agentdiet.llm_client import ChatResult


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "serving" / "gpt_oss_20b_serve.sh"
SMOKE_RUNNER = REPO_ROOT / "scripts" / "serving" / "smoke_gpt_oss.py"
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "gpt-oss-vllm-serving.md"


class _SimulatedGptOssBackend:
    """Returns longer responses when thinking=True (high reasoning_effort)."""

    def chat_full(self, messages, model, temperature, *,
                  thinking=False, top_p=1.0):
        if thinking:
            return ChatResult(
                response="<long CoT>" * 100 + "#### 18",
                prompt_tokens=80, completion_tokens=600,
            )
        return ChatResult(
            response="The answer is 18. #### 18",
            prompt_tokens=80, completion_tokens=80,
        )


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_gpt_oss", SMOKE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def test_serve_script_exists_and_executable():
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & stat.S_IXUSR


def test_serve_script_references_gpt_oss_model():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "vllm serve" in text
    assert "gpt-oss-20b" in text


def test_runbook_mentions_reasoning_effort():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "reasoning_effort" in text
    assert "smoke_gpt_oss.py" in text


def test_smoke_runner_writes_well_formed_artifact():
    smoke = _load_smoke_module()
    artifact = smoke.run_smoke(
        backend=_SimulatedGptOssBackend(), model="openai/gpt-oss-20b",
    )
    for key in ("model", "tested_at", "thinking_on", "thinking_off", "summary"):
        assert key in artifact
    # Per-call schema also tags reasoning_effort
    assert artifact["thinking_on"][0]["reasoning_effort"] == "high"
    assert artifact["thinking_off"][0]["reasoning_effort"] == "low"


def test_smoke_runner_distinguishes_high_low_effort():
    smoke = _load_smoke_module()
    artifact = smoke.run_smoke(
        backend=_SimulatedGptOssBackend(), model="openai/gpt-oss-20b",
    )
    s = artifact["summary"]
    assert s["mean_tokens_on"] > s["mean_tokens_off"]
    assert s["passed"] is True


def test_smoke_runner_main_writes_to_file(tmp_path):
    smoke = _load_smoke_module()
    output = tmp_path / "gpt_oss_smoke.json"
    rc = smoke.main(
        argv=["--model", "openai/gpt-oss-20b"],
        backend=_SimulatedGptOssBackend(),
        output_path=output,
    )
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["passed"] is True


def test_smoke_runner_returns_nonzero_when_no_difference(tmp_path):
    class _FlatBackend:
        def chat_full(self, messages, model, temperature, **kw):
            return ChatResult(
                response="X", prompt_tokens=5, completion_tokens=5,
            )

    smoke = _load_smoke_module()
    output = tmp_path / "gpt_oss_smoke.json"
    rc = smoke.main(
        argv=["--model", "openai/gpt-oss-20b"],
        backend=_FlatBackend(),
        output_path=output,
    )
    assert rc == 1
