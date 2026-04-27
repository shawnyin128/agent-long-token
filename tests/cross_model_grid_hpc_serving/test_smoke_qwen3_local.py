"""Local lint-style tests for the Qwen3 smoke runner.

Runs the runner against a DummyBackend that returns known
completion_tokens so the runner's artifact format + summary math
can be exercised without GPU.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from agentdiet.llm_client import ChatResult


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "serving" / "qwen3_a3b_serve.sh"
SMOKE_RUNNER = REPO_ROOT / "scripts" / "serving" / "smoke_qwen3.py"
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "qwen3-vllm-serving.md"


class _SimulatedQwenBackend:
    """Returns longer responses when thinking=True (simulating a real
    Qwen3 hybrid model)."""

    def chat_full(self, messages, model, temperature, *,
                  thinking=False, top_p=1.0):
        if thinking:
            return ChatResult(
                response="<thinking trace>" * 50 + "#### 18",
                prompt_tokens=80, completion_tokens=400,
            )
        return ChatResult(
            response="The answer is 18. #### 18",
            prompt_tokens=80, completion_tokens=80,
        )


def test_serve_script_exists_and_executable():
    assert SCRIPT.is_file()
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, f"{SCRIPT} not user-executable"


def test_serve_script_references_model_and_vllm():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "vllm serve" in text
    assert "Qwen3-30B-A3B" in text
    # Tensor-parallel auto-detection
    assert "CUDA_VISIBLE_DEVICES" in text


def test_runbook_exists_and_mentions_smoke_command():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "smoke_qwen3.py" in text
    assert "enable_thinking" in text or "thinking" in text.lower()


def test_smoke_runner_writes_well_formed_artifact(tmp_path, monkeypatch):
    """Run the smoke main() against a simulated backend and check the
    artifact JSON structure."""
    # Import without going through CLI argparse: call run_smoke directly.
    import importlib.util
    spec = importlib.util.spec_from_file_location("smoke_qwen3", SMOKE_RUNNER)
    smoke_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke_module)  # type: ignore

    backend = _SimulatedQwenBackend()
    artifact = smoke_module.run_smoke(backend=backend, model="Qwen/Qwen3-30B-A3B")

    # Top-level keys
    for key in ("model", "tested_at", "thinking_on", "thinking_off", "summary"):
        assert key in artifact, f"missing key {key}"
    # 5 questions × 2 toggles
    assert len(artifact["thinking_on"]) == 5
    assert len(artifact["thinking_off"]) == 5
    # Per-call schema
    for row in artifact["thinking_on"] + artifact["thinking_off"]:
        for k in ("qid", "thinking", "response_chars",
                  "response_preview", "prompt_tokens", "completion_tokens"):
            assert k in row
    # Summary
    s = artifact["summary"]
    assert s["mean_tokens_on"] > s["mean_tokens_off"]
    assert s["passed"] is True
    assert s["delta"] == pytest.approx(s["mean_tokens_on"] - s["mean_tokens_off"])


def test_smoke_runner_main_writes_to_file(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("smoke_qwen3", SMOKE_RUNNER)
    smoke_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke_module)  # type: ignore

    output = tmp_path / "qwen3_smoke.json"
    rc = smoke_module.main(
        argv=["--model", "Qwen/Qwen3-30B-A3B"],
        backend=_SimulatedQwenBackend(),
        output_path=output,
    )
    assert rc == 0
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["passed"] is True


def test_smoke_runner_returns_nonzero_when_thinking_doesnt_help(tmp_path):
    """If thinking=True does NOT increase output, exit code 1."""
    class _FlatBackend:
        def chat_full(self, messages, model, temperature, **kw):
            return ChatResult(
                response="same response", prompt_tokens=10, completion_tokens=10,
            )

    import importlib.util
    spec = importlib.util.spec_from_file_location("smoke_qwen3", SMOKE_RUNNER)
    smoke_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke_module)  # type: ignore

    output = tmp_path / "qwen3_smoke.json"
    rc = smoke_module.main(
        argv=["--model", "Qwen/Qwen3-30B-A3B"],
        backend=_FlatBackend(),
        output_path=output,
    )
    assert rc == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["passed"] is False
