from __future__ import annotations

import pytest

from agentdiet.dataset import Question, load_gsm8k, parse_answer


class TestParseAnswer:
    def test_hash_marker(self):
        assert parse_answer("reasoning...\n#### 42") == "42"

    def test_hash_marker_negative(self):
        assert parse_answer("#### -7") == "-7"

    def test_hash_marker_with_comma(self):
        assert parse_answer("#### 1,234") == "1234"

    def test_hash_marker_decimal(self):
        assert parse_answer("#### 3.14") == "3.14"

    def test_dollar_sign(self):
        assert parse_answer("The total cost is $42.") == "42"

    def test_dollar_sign_with_comma(self):
        assert parse_answer("We owe $1,250") == "1250"

    def test_trailing_dollars_word(self):
        assert parse_answer("He earned 42 dollars in total.") == "42"

    def test_sentence_final(self):
        assert parse_answer("So the final answer is 15.") == "15"

    def test_decimal_variant(self):
        assert parse_answer("The ratio is 42.0") == "42"

    def test_unicode_minus_not_captured_as_number_alone(self):
        assert parse_answer("no numbers here") is None

    def test_empty(self):
        assert parse_answer("") is None

    def test_none(self):
        assert parse_answer(None) is None

    def test_whitespace_only(self):
        assert parse_answer("   \n  ") is None

    def test_hash_takes_precedence_over_inline_numbers(self):
        assert parse_answer("I thought 99 but recomputed. #### 42") == "42"

    def test_multiple_hash_markers_last_wins(self):
        assert parse_answer("Initially #### 99 but on reflection #### 42") == "42"

    def test_three_hash_markers_last_wins(self):
        assert parse_answer("#### 1 then #### 2 then #### 3") == "3"


class TestLoadGSM8KOffline:
    """Exercises load_gsm8k with a mocked HF backend (no network)."""

    FAKE_ROWS = [
        {"question": f"Q{i}", "answer": f"solution steps for {i}\n#### {i}"}
        for i in range(20)
    ]

    def _install_fake_hf(self, monkeypatch):
        import agentdiet.dataset as ds_mod
        import sys
        import types

        fake_mod = types.ModuleType("datasets")

        def fake_load_dataset(name, subset, split, **kwargs):
            assert name == "gsm8k"
            assert subset == "main"
            assert split in {"train", "test"}
            return list(self.FAKE_ROWS)

        fake_mod.load_dataset = fake_load_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    def test_load_returns_requested_n(self, monkeypatch):
        self._install_fake_hf(monkeypatch)
        qs = load_gsm8k(split="test", n=5, seed=42)
        assert len(qs) == 5
        assert all(isinstance(q, Question) for q in qs)

    def test_same_seed_reproduces(self, monkeypatch):
        self._install_fake_hf(monkeypatch)
        a = load_gsm8k(split="test", n=5, seed=42)
        b = load_gsm8k(split="test", n=5, seed=42)
        assert [q.qid for q in a] == [q.qid for q in b]

    def test_different_seed_changes(self, monkeypatch):
        self._install_fake_hf(monkeypatch)
        a = load_gsm8k(split="test", n=5, seed=42)
        c = load_gsm8k(split="test", n=5, seed=99)
        assert [q.qid for q in a] != [q.qid for q in c]

    def test_gold_answer_parsed(self, monkeypatch):
        self._install_fake_hf(monkeypatch)
        qs = load_gsm8k(split="test", n=3, seed=42)
        for q in qs:
            assert q.gold_answer
            assert q.gold_answer.isdigit() or q.gold_answer.lstrip("-").replace(".", "").isdigit()

    def test_n_larger_than_pool_returns_all(self, monkeypatch):
        self._install_fake_hf(monkeypatch)
        qs = load_gsm8k(split="test", n=999, seed=42)
        assert len(qs) == len(self.FAKE_ROWS)

    def test_n_none_returns_all(self, monkeypatch):
        self._install_fake_hf(monkeypatch)
        qs = load_gsm8k(split="test", n=None, seed=42)
        assert len(qs) == len(self.FAKE_ROWS)

    def test_malformed_gold_row_is_skipped(self, monkeypatch):
        import sys
        import types
        fake_mod = types.ModuleType("datasets")

        def fake_load_dataset(name, subset, split, **kwargs):
            return [
                {"question": "ok1", "answer": "solution\n#### 5"},
                {"question": "bad", "answer": "no marker here"},
                {"question": "ok2", "answer": "solution\n#### 7"},
            ]

        fake_mod.load_dataset = fake_load_dataset
        monkeypatch.setitem(sys.modules, "datasets", fake_mod)
        qs = load_gsm8k(split="test", n=None, seed=42)
        assert len(qs) == 2
        assert [q.gold_answer for q in qs] == ["5", "7"]


@pytest.mark.skipif(
    __import__("os").environ.get("AGENTDIET_ALLOW_NETWORK") != "1",
    reason="requires network + HF download; set AGENTDIET_ALLOW_NETWORK=1",
)
class TestLoadGSM8KNetwork:
    def test_load_returns_requested_n(self, tmp_path):
        qs = load_gsm8k(split="test", n=10, seed=42, cache_dir=tmp_path)
        assert len(qs) == 10
        assert all(isinstance(q, Question) for q in qs)

    def test_same_seed_reproduces(self, tmp_path):
        a = load_gsm8k(split="test", n=10, seed=42, cache_dir=tmp_path)
        b = load_gsm8k(split="test", n=10, seed=42, cache_dir=tmp_path)
        assert [q.qid for q in a] == [q.qid for q in b]

    def test_different_seed_changes(self, tmp_path):
        a = load_gsm8k(split="test", n=10, seed=42, cache_dir=tmp_path)
        c = load_gsm8k(split="test", n=10, seed=99, cache_dir=tmp_path)
        assert [q.qid for q in a] != [q.qid for q in c]

    def test_gold_answer_non_empty(self, tmp_path):
        qs = load_gsm8k(split="test", n=3, seed=42, cache_dir=tmp_path)
        for q in qs:
            assert q.gold_answer
            assert parse_answer(f"#### {q.gold_answer}") == q.gold_answer
