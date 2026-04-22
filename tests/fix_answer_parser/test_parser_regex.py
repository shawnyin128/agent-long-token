"""Parser regex fix for Qwen's literal-N output."""
from __future__ import annotations

import pytest

from agentdiet.dataset import parse_answer


def test_plain_hash_number_still_works():
    assert parse_answer("blah blah\n#### 42") == "42"


def test_hash_with_literal_N_prefix():
    # The bug: Qwen outputs '#### N66' and parser used to miss.
    assert parse_answer("#### N66") == "66"


def test_hash_with_N_space_number():
    assert parse_answer("#### N 66") == "66"


def test_hash_with_ans_prefix():
    assert parse_answer("#### ANS 42") == "42"


def test_hash_with_answer_prefix():
    assert parse_answer("#### answer 42") == "42"


def test_hash_negative_number():
    assert parse_answer("#### -5") == "-5"


def test_hash_decimal_number():
    assert parse_answer("#### 3.14") == "3.14"


def test_hash_with_letters_and_negative():
    assert parse_answer("#### N -5") == "-5"


def test_multiple_hash_markers_uses_last():
    # Existing contract: last '#### N' wins.
    assert parse_answer("#### 10\n\nMore work\n\n#### N66") == "66"


def test_full_qwen_response_picks_hash_not_first_dollar():
    """This is the bug in production: response has $16 (a price) AND #### N66.
    Before fix: parser returned '16' via dollar-fallback. After: returns '66'."""
    response = """Each book costs $16. Total books: 3 * 16 = 48 dollars.
Each pencil costs $6. Total pencils: 3 * 6 = 18 dollars.
Total spent: 48 + 18 = 66 dollars.

Thus, Ted spent a total of \\(\\boxed{66}\\) dollars.

#### N66"""
    assert parse_answer(response) == "66"


def test_hash_no_digits_falls_through_to_dollar_or_number():
    # '#### forty-two' has no digit → regex fails, falls back.
    # Behavior depends on later fallbacks; at minimum should not crash.
    out = parse_answer("spent $42 on food\n#### forty-two")
    # Dollar fallback picks $42.
    assert out == "42"


def test_hash_empty_marker_falls_through():
    # '####' with nothing after should not match.
    out = parse_answer("answer 42\n####")
    assert out == "42"  # falls to last-number regex


def test_dollar_only_fallback_still_works():
    # No #### → dollar regex picks first $<num>.
    assert parse_answer("I paid $99 for it.") == "99"


def test_multiple_hash_N_picks_last_one():
    """Debate transcripts have multiple '#### N<num>' in history; last wins."""
    text = "Agent 0: #### N10\nAgent 1: #### N20\nFinal: #### N30"
    assert parse_answer(text) == "30"
