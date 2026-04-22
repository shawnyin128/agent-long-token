from __future__ import annotations

from pathlib import Path

import pytest

from agentdiet.cli.collect import run_collection
from agentdiet.config import Config
from agentdiet.dataset import Question
from agentdiet.llm_client import DummyBackend


def _cfg(tmp_path: Path) -> Config:
    return Config(
        artifacts_dir=tmp_path,
        hf_cache_dir=tmp_path / "hf",
        model="test-model",
        n_agents=2,
        n_rounds=1,
    )


def _qs(n: int) -> list[Question]:
    return [Question(qid=f"q{i}", question=f"problem {i}", gold_answer=str(i)) for i in range(n)]


def test_manifest_count_equals_questions(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    qs = _qs(5)
    backend = DummyBackend(lambda m, mo, t: "#### 0")
    manifest = run_collection(cfg, qs, backend=backend)
    assert len(manifest["outcomes"]) == len(qs)
    assert sum(manifest["counts"].values()) == len(qs)


def test_counts_bucket_sum_matches(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    qs = _qs(4)

    def mixed(msgs, model, temp):
        text = msgs[-1]["content"]
        if "problem 0" in text:
            return "#### 0"
        if "problem 1" in text:
            return "no number"
        if "problem 2" in text:
            raise RuntimeError("sim fail")
        return "#### 3"

    backend = DummyBackend(mixed)
    manifest = run_collection(cfg, qs, backend=backend)
    counts = manifest["counts"]
    assert counts["ok"] + counts["unparsed"] + counts["failed"] + counts["cached"] == 4


def test_all_failed_exit_code_via_main(tmp_path, monkeypatch):
    import sys
    import types

    # Fake HF dataset so main() can load questions without network
    fake_mod = types.ModuleType("datasets")

    def fake_load_dataset(name, subset, split, **kwargs):
        return [{"question": f"P{i}", "answer": f"...\n#### {i}"} for i in range(3)]

    fake_mod.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    # Fake backend that always fails
    from agentdiet.cli import collect as collect_mod

    class AlwaysFail:
        def __init__(self, *a, **k):
            pass

        def chat(self, messages, model, temperature):
            raise RuntimeError("simulated vLLM down")

    monkeypatch.setattr(collect_mod, "OpenAIBackend", AlwaysFail)
    monkeypatch.setenv("AGENTDIET_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTDIET_HF_CACHE_DIR", str(tmp_path / "hf"))
    monkeypatch.setenv("AGENTDIET_MODEL", "test-model")
    monkeypatch.setenv("AGENTDIET_N_QUESTIONS", "3")
    monkeypatch.setenv("AGENTDIET_MAX_RETRIES", "1")

    rc = collect_mod.main(["--n", "3"])
    assert rc == 1


def test_partial_failure_exits_zero(tmp_path, monkeypatch):
    import sys
    import types
    fake_mod = types.ModuleType("datasets")

    def fake_load_dataset(name, subset, split, **kwargs):
        return [{"question": f"P{i}", "answer": f"...\n#### {i}"} for i in range(3)]

    fake_mod.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    from agentdiet.cli import collect as collect_mod

    class MostlyOk:
        def __init__(self, *a, **k):
            self.n = 0

        def chat(self, messages, model, temperature):
            self.n += 1
            if "P1" in messages[-1]["content"]:
                raise RuntimeError("fail one")
            return "#### 0"

    monkeypatch.setattr(collect_mod, "OpenAIBackend", MostlyOk)
    monkeypatch.setenv("AGENTDIET_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTDIET_HF_CACHE_DIR", str(tmp_path / "hf"))
    monkeypatch.setenv("AGENTDIET_MODEL", "test-model")
    monkeypatch.setenv("AGENTDIET_N_QUESTIONS", "3")
    monkeypatch.setenv("AGENTDIET_MAX_RETRIES", "1")
    monkeypatch.setenv("AGENTDIET_N_AGENTS", "2")
    monkeypatch.setenv("AGENTDIET_N_ROUNDS", "1")

    rc = collect_mod.main(["--n", "3"])
    assert rc == 0  # partial failure still returns 0


def test_stderr_summary_matches_manifest(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    qs = _qs(3)
    backend = DummyBackend(lambda m, mo, t: "#### 0")
    manifest = run_collection(cfg, qs, backend=backend)
    # Per-qid progress lines go to stderr
    captured = capsys.readouterr()
    for q in qs:
        assert q.qid in captured.err
    assert manifest["counts"]["ok"] == 3
