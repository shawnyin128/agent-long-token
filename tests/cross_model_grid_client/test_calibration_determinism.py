"""calibrate_n is referentially transparent; voting + calibrate
composition recovers expected over_budget_factor."""
from __future__ import annotations

import json

from agentdiet.llm_client import ChatResult, LLMClient
from agentdiet.voting import calibrate_n, run_voting


def test_calibrate_n_deterministic_over_100_invocations():
    debate = [3000, 3500, 2800, 3200, 2900, 3100, 3300, 3000, 2950, 3050]
    sa = [320, 290, 310, 305, 295, 315, 300, 308, 298, 302]
    first = calibrate_n(debate, sa)
    for _ in range(100):
        result = calibrate_n(debate, sa)
        assert result == first


def test_calibrate_n_independent_of_input_ordering():
    """Means are order-invariant — shuffling inputs gives identical result."""
    a = calibrate_n([1000, 2000, 3000], [100, 200, 300])
    b = calibrate_n([3000, 1000, 2000], [300, 100, 200])
    assert a == b


def test_voting_then_calibrate_composition_recovers_over_budget(tmp_path):
    """Run voting on a fixed-usage backend, sum tokens externally,
    feed into calibrate_n, and verify over_budget_factor matches what
    we calculate by hand."""
    PROMPT_TOK = 50
    COMPLETION_TOK = 30

    class FixedUsageBackend:
        call_count = 0

        def chat_full(self, messages, model, temperature, **kw):
            self.call_count += 1
            return ChatResult(
                response="#### 1",
                prompt_tokens=PROMPT_TOK,
                completion_tokens=COMPLETION_TOK,
            )

    backend = FixedUsageBackend()
    client = LLMClient(backend, cache_path=tmp_path / "cache.jsonl")
    voting_result = run_voting(
        question="Q",
        n_samples=5,
        llm_client=client,
        model="m",
        system_prompt="sys",
    )
    assert voting_result.total_tokens == 5 * (PROMPT_TOK + COMPLETION_TOK)

    # Feed N independent SA token totals (per-sample) into calibrate_n
    # alongside synthetic debate budgets.
    sa_per_call = [PROMPT_TOK + COMPLETION_TOK] * 5  # one per voting sample
    debate_per_q = [320] * 5  # synthetic — what 3x3 debate cost would be

    cal = calibrate_n(debate_per_q, sa_per_call)

    # Verify over_budget_factor matches manual computation:
    expected_mean_debate = 320.0
    expected_mean_sa = float(PROMPT_TOK + COMPLETION_TOK)
    import math
    expected_n_raw = math.ceil(expected_mean_debate / expected_mean_sa)
    expected_n = max(expected_n_raw, 3)
    expected_over = (expected_n * expected_mean_sa) / expected_mean_debate
    assert cal.over_budget_factor == expected_over
    assert cal.N_raw == expected_n_raw
    assert cal.N == expected_n


def test_calibration_result_json_serializable():
    """asdict + json.dumps round-trip works for downstream artifact write."""
    import dataclasses

    cal = calibrate_n([3000] * 10, [300] * 10)
    payload = json.dumps(dataclasses.asdict(cal))
    parsed = json.loads(payload)
    assert parsed["N"] == 10
    assert parsed["floor_active"] is False
    assert parsed["over_budget_factor"] == 1.0
