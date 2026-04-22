from __future__ import annotations

import json

import pytest

from agentdiet.extract_claims import _fix_json_escapes


def test_fixes_latex_open_paren():
    assert _fix_json_escapes(r"\(") == r"\\("


def test_fixes_latex_open_bracket():
    assert _fix_json_escapes(r"\[") == r"\\["


def test_fixes_latex_command_frac():
    assert _fix_json_escapes(r"\frac") == r"\\frac"


def test_fixes_latex_command_times():
    assert _fix_json_escapes(r"\times") == r"\\times"


def test_leaves_legal_escape_backslash_quote():
    # JSON "\"" → Python r'\"' (2 chars: backslash, quote)
    assert _fix_json_escapes(r'\"') == r'\"'


def test_leaves_legal_escape_backslash_backslash():
    # JSON "\\" → 2 chars.
    assert _fix_json_escapes(r"\\") == r"\\"


def test_leaves_legal_escape_n_t_r():
    for esc in (r"\n", r"\t", r"\r", r"\b", r"\f", r"\/"):
        assert _fix_json_escapes(esc) == esc, f"broke legal {esc!r}"


def test_leaves_legal_escape_unicode():
    assert _fix_json_escapes(r"é") == r"é"


def test_real_failure_sample_parses_after_fix():
    raw = (
        r'[{"type": "evidence", "text": "Cost is 20", '
        r'"quote": "Total cost: \( 16 + 4 = 20 \) dollars"}]'
    )
    fixed = _fix_json_escapes(raw)
    data = json.loads(fixed)
    assert data[0]["quote"] == r"Total cost: \( 16 + 4 = 20 \) dollars"


def test_plain_json_without_backslashes_unchanged():
    raw = '[{"type": "proposal", "text": "x", "quote": "hello"}]'
    assert _fix_json_escapes(raw) == raw


def test_mixed_latex_and_legal_escapes():
    # Has both \( (invalid) and \n (valid) — both must survive correctly.
    raw = r'{"quote": "line\n\( x \) end"}'
    fixed = _fix_json_escapes(raw)
    obj = json.loads(fixed)
    # '\n' stays a newline; '\(' becomes literal \(
    assert obj["quote"] == "line\n\\( x \\) end"


def test_double_backslash_latex_command_unchanged():
    # Already-correctly-escaped '\\frac' should NOT become '\\\\frac'.
    raw = r'{"quote": "\\frac{1}{2}"}'
    fixed = _fix_json_escapes(raw)
    obj = json.loads(fixed)
    # JSON '\\frac' decodes to literal \frac (single backslash).
    assert obj["quote"] == r"\frac{1}{2}"


def test_consecutive_bad_escapes():
    raw = r'"\( x \) \[ y \]"'
    fixed = _fix_json_escapes(raw)
    obj = json.loads(fixed)
    assert obj == r"\( x \) \[ y \]"
