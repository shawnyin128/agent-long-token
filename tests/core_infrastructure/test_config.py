from __future__ import annotations

import os
from pathlib import Path

from agentdiet.config import Config


def test_defaults_match_spec():
    cfg = Config()
    assert cfg.model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.temperature == 0.0
    assert cfg.n_agents == 3
    assert cfg.n_rounds == 3
    assert cfg.n_questions == 100
    assert cfg.n_pilot == 30
    assert cfg.seed == 42


def test_env_override(monkeypatch):
    monkeypatch.setenv("AGENTDIET_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("AGENTDIET_N_QUESTIONS", "50")
    cfg = Config()
    assert cfg.model == "gpt-4o-mini"
    assert cfg.n_questions == 50


def test_derived_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDIET_ARTIFACTS_DIR", str(tmp_path))
    cfg = Config()
    assert cfg.cache_path == tmp_path / "llm_cache.jsonl"
    assert "Qwen__Qwen2.5-7B-Instruct" in str(cfg.dialogues_dir)


def test_ensure_dirs_creates_all(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDIET_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTDIET_HF_CACHE_DIR", str(tmp_path / "hf"))
    cfg = Config()
    cfg.ensure_dirs()
    for d in [
        cfg.artifacts_dir,
        cfg.hf_cache_dir,
        cfg.dialogues_dir,
        cfg.claims_dir,
        cfg.analysis_dir,
        cfg.compression_dir,
        cfg.evaluation_dir,
        cfg.failures_dir,
    ]:
        assert Path(d).is_dir()
