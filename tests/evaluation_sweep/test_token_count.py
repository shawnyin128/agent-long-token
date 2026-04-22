from __future__ import annotations

import pytest

from agentdiet.evaluate import TOKENS_PER_CHAR, count_tokens


def test_empty_string_is_zero_tokens():
    assert count_tokens("") == 0


def test_count_tokens_monotone_in_length():
    assert count_tokens("hi") < count_tokens("hello world")


def test_count_tokens_formula():
    # Heuristic: ceil(len(s) * TOKENS_PER_CHAR) truncated/approximated to int.
    s = "a" * 40
    expected = int(round(len(s) * TOKENS_PER_CHAR))
    assert count_tokens(s) == expected


def test_tokens_per_char_sensible_value():
    # Rough char-to-token ratio for English GPT-family is ~0.25.
    assert 0.2 <= TOKENS_PER_CHAR <= 0.5


def test_count_tokens_deterministic():
    assert count_tokens("abcdef") == count_tokens("abcdef")
