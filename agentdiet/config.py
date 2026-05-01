from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTDIET_",
        env_file=".env",
        extra="ignore",
    )

    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.0
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    request_timeout_s: float = 600.0  # AIME / thinking-on can take minutes
    max_retries: int = 3

    seed: int = 42
    n_agents: int = 3
    n_rounds: int = 3
    n_questions: int = 100
    n_pilot: int = 30

    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts")
    hf_cache_dir: Path = Field(default=PROJECT_ROOT / "artifacts" / "dataset" / "hf_cache")

    @property
    def model_slug(self) -> str:
        return self.model.replace("/", "__")

    @property
    def cache_path(self) -> Path:
        return self.artifacts_dir / "llm_cache.jsonl"

    @property
    def dataset_sample_path(self) -> Path:
        return self.artifacts_dir / "dataset" / "gsm8k_sample.json"

    @property
    def dialogues_dir(self) -> Path:
        return self.artifacts_dir / "dialogues" / self.model_slug

    @property
    def claims_dir(self) -> Path:
        return self.artifacts_dir / "claims" / self.model_slug

    @property
    def analysis_dir(self) -> Path:
        return self.artifacts_dir / "analysis"

    @property
    def compression_dir(self) -> Path:
        return self.artifacts_dir / "compression"

    @property
    def evaluation_dir(self) -> Path:
        return self.artifacts_dir / "evaluation"

    @property
    def failures_dir(self) -> Path:
        return self.artifacts_dir / "failures"

    def ensure_dirs(self) -> None:
        for d in [
            self.artifacts_dir,
            self.artifacts_dir / "dataset",
            self.hf_cache_dir,
            self.dialogues_dir,
            self.claims_dir,
            self.analysis_dir,
            self.compression_dir,
            self.compression_dir / "compressed",
            self.evaluation_dir,
            self.failures_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    return Config()
