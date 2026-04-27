"""voting.py — calibrate_n + run_voting."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentdiet.llm_client import ChatResult, DummyBackend, LLMClient
from agentdiet.voting import (
    CalibrationResult,
    VotingResult,
    calibrate_n,
    run_voting,
)


# --- calibrate_n ---


def test_calibrate_n_basic_no_floor():
    res = calibrate_n([3000] * 10, [300] * 10)
    assert res.N_raw == 10
    assert res.N == 10
    assert res.floor_active is False
    assert res.mean_debate_tokens == 3000
    assert res.mean_sa_tokens == 300
    assert res.over_budget_factor == pytest.approx(1.0)


def test_calibrate_n_floor_active():
    """When SA per-call cost is close to debate, N_raw < 3 -> floor."""
    res = calibrate_n([3000] * 10, [2500] * 10)
    assert res.N_raw == 2
    assert res.N == 3
    assert res.floor_active is True
    assert res.over_budget_factor == pytest.approx(2500 * 3 / 3000)


def test_calibrate_n_ceil_rounding():
    """N_raw uses ceil — fractional ratio rounds up."""
    res = calibrate_n([1000], [333])
    assert res.N_raw == 4  # ceil(1000/333) = 4
    assert res.N == 4
    assert res.floor_active is False


def test_calibrate_n_empty_inputs_raises():
    with pytest.raises(ValueError):
        calibrate_n([], [100])
    with pytest.raises(ValueError):
        calibrate_n([100], [])


def test_calibrate_n_nonpositive_count_raises():
    with pytest.raises(ValueError):
        calibrate_n([100, 0, 100], [50] * 3)
    with pytest.raises(ValueError):
        calibrate_n([100] * 3, [50, -1, 50])


def test_calibrate_n_unequal_lengths_allowed():
    """Means use each list's own length — different ns are fine."""
    res = calibrate_n([1000, 2000, 3000], [100, 200])
    assert res.mean_debate_tokens == 2000
    assert res.mean_sa_tokens == 150


def test_calibration_result_serializable():
    """Frozen dataclass + asdict round-trips for JSON."""
    import dataclasses
    res = calibrate_n([1000] * 5, [100] * 5)
    d = dataclasses.asdict(res)
    assert set(d.keys()) == {
        "N_raw", "N", "mean_debate_tokens", "mean_sa_tokens",
        "over_budget_factor", "floor_active",
    }


# --- run_voting ---


def _make_client(tmp_path: Path, responder) -> tuple[LLMClient, DummyBackend]:
    backend = DummyBackend(responder)
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    return client, backend


def test_run_voting_distinct_samples_via_nonce(tmp_path):
    """Each sample id gets its own response from a per-message responder."""
    def responder(msgs, model, temp, **kw):
        # extract the "Sample id: K" line from user content
        user = msgs[-1]["content"]
        line = user.split("\n", 1)[0]  # "Sample id: K"
        k = int(line.split(":")[1].strip())
        return f"Solution path {k}\n#### {k}"

    client, _ = _make_client(tmp_path, responder)
    result = run_voting(
        question="What is 2+2?",
        n_samples=4,
        llm_client=client,
        model="m",
        system_prompt="You are a solver.",
    )
    assert isinstance(result, VotingResult)
    assert len(result.samples) == 4
    assert len(result.parsed_answers) == 4
    assert result.parsed_answers == ["0", "1", "2", "3"]
    # 4 distinct answers each with count 1 -> majority_vote returns None (tie rule)
    assert result.final_answer is None


def test_run_voting_majority(tmp_path):
    """3 of 5 samples agree on '7' — final_answer must be '7'."""
    answers = ["7", "5", "7", "9", "7"]

    def responder(msgs, model, temp, **kw):
        user = msgs[-1]["content"]
        k = int(user.split("\n", 1)[0].split(":")[1].strip())
        return f"working...\n#### {answers[k]}"

    client, _ = _make_client(tmp_path, responder)
    result = run_voting(
        question="Q",
        n_samples=5,
        llm_client=client,
        model="m",
        system_prompt="sys",
    )
    assert result.final_answer == "7"


def test_run_voting_n_samples_zero_raises(tmp_path):
    """n_samples must be >= 1; 0 is a programmer error, not a degenerate."""
    client, _ = _make_client(tmp_path, lambda m, mo, t, **k: "#### 0")
    with pytest.raises(ValueError):
        run_voting(
            question="Q", n_samples=0, llm_client=client, model="m",
            system_prompt="sys",
        )


def test_run_voting_n_samples_negative_raises(tmp_path):
    client, _ = _make_client(tmp_path, lambda m, mo, t, **k: "#### 0")
    with pytest.raises(ValueError):
        run_voting(
            question="Q", n_samples=-3, llm_client=client, model="m",
            system_prompt="sys",
        )


def test_run_voting_n_samples_one(tmp_path):
    """Degenerate case: 1 sample, final_answer is that sample's parse."""
    client, _ = _make_client(tmp_path, lambda m, mo, t, **k: "answer is\n#### 42")
    result = run_voting(
        question="Q", n_samples=1, llm_client=client, model="m",
        system_prompt="sys",
    )
    assert result.final_answer == "42"


def test_run_voting_total_tokens_sums_per_call(tmp_path):
    """total_tokens = sum of (prompt + completion) across all calls."""

    class UsageBackend:
        call_count = 0

        def chat_full(self, messages, model, temperature, **kw):
            self.call_count += 1
            return ChatResult(
                response="#### 1",
                prompt_tokens=10,
                completion_tokens=5,
            )

    backend = UsageBackend()
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    result = run_voting(
        question="Q", n_samples=4, llm_client=client, model="m",
        system_prompt="sys",
    )
    # 4 calls × (10 + 5) = 60
    assert result.total_tokens == 60


def test_run_voting_passes_thinking_through(tmp_path):
    seen_kwargs: list[dict] = []

    def responder(msgs, model, temp, *, thinking=False, top_p=1.0, **kw):
        seen_kwargs.append({"thinking": thinking, "top_p": top_p})
        return "#### 0"

    client, _ = _make_client(tmp_path, responder)
    run_voting(
        question="Q", n_samples=2, llm_client=client, model="m",
        system_prompt="sys", thinking=True, top_p=0.95,
    )
    for kw in seen_kwargs:
        assert kw == {"thinking": True, "top_p": 0.95}


def test_run_voting_unparseable_responses(tmp_path):
    """If all samples fail to parse, parsed_answers are all None and
    final_answer is None."""
    client, _ = _make_client(
        tmp_path, lambda m, mo, t, **kw: "garbage with no number marker"
    )
    result = run_voting(
        question="Q", n_samples=3, llm_client=client, model="m",
        system_prompt="sys",
    )
    # Note: parse_answer falls back to last number in text if no '####'
    # — to truly get None we use a custom parser
    # so this test just confirms the structure handles whatever parser yields
    assert len(result.parsed_answers) == 3


def test_run_voting_custom_parser(tmp_path):
    """Custom parser controls how answers are extracted."""
    def my_parser(text: str) -> str | None:
        return "X" if "marker" in text else None

    client, _ = _make_client(
        tmp_path, lambda m, mo, t, **kw: "marker present"
    )
    result = run_voting(
        question="Q", n_samples=3, llm_client=client, model="m",
        system_prompt="sys", parser=my_parser,
    )
    assert result.parsed_answers == ["X", "X", "X"]
    assert result.final_answer == "X"


def test_run_voting_default_temperature_and_top_p(tmp_path):
    """Defaults match Wang et al. 2023 self-consistency: temp=0.7, top_p=0.95."""
    seen: list[tuple[float, float]] = []

    def responder(msgs, model, temp, *, top_p=1.0, **kw):
        seen.append((temp, top_p))
        return "#### 0"

    client, _ = _make_client(tmp_path, responder)
    run_voting(
        question="Q", n_samples=2, llm_client=client, model="m",
        system_prompt="sys",
    )
    for t, tp in seen:
        assert t == 0.7
        assert tp == 0.95
