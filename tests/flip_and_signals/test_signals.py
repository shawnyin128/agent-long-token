from __future__ import annotations

import numpy as np
import pytest

from agentdiet.analysis.signals import (
    HashingFakeEmbedder,
    REFERENCED_LATER_THRESHOLD,
    SIGNAL_KEYS,
    compute_signals,
)
from agentdiet.types import FlipEvent


def _claim(cid, text, agent_id, round_, type_="proposal"):
    return {
        "id": cid, "text": text, "agent_id": agent_id, "round": round_,
        "type": type_, "source_message_span": [0, 3],
    }


def test_hashing_fake_embedder_returns_unit_norm_vectors():
    emb = HashingFakeEmbedder(dim=16)
    vecs = emb.encode(["hello", "world", "hello world"])
    assert vecs.shape == (3, 16)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_hashing_fake_embedder_is_deterministic():
    a = HashingFakeEmbedder(dim=16).encode(["x", "y"])
    b = HashingFakeEmbedder(dim=16).encode(["x", "y"])
    assert np.array_equal(a, b)


def test_compute_signals_returns_exact_keys():
    claims = [_claim("c0", "first claim", 0, 1)]
    rows = compute_signals(claims, flip_events=[], embedder=HashingFakeEmbedder())
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(SIGNAL_KEYS)


def test_signal_keys_contain_exactly_5_names():
    assert set(SIGNAL_KEYS) == {
        "claim_id", "flip_coincidence", "novelty",
        "referenced_later", "position",
    }


def test_no_composite_score_key_present():
    claims = [_claim(f"c{i}", f"t{i}", 0, 1) for i in range(3)]
    rows = compute_signals(claims, flip_events=[], embedder=HashingFakeEmbedder())
    for r in rows:
        # Ensure no forbidden composite/weighted/importance key snuck in.
        for forbidden in ("composite_score", "importance", "weighted_score",
                          "combined_score", "score"):
            assert forbidden not in r


def test_flip_coincidence_true_for_claim_in_flip_round():
    claims = [
        _claim("c0", "wrong", 0, 1),
        _claim("c1", "right", 0, 2),
    ]
    fe = FlipEvent(
        question_id="q", round=2, triggering_claim_id="c1",
        pre_flip_answers={}, post_flip_answers={},
    )
    rows = compute_signals(claims, flip_events=[fe], embedder=HashingFakeEmbedder())
    by_id = {r["claim_id"]: r for r in rows}
    assert by_id["c0"]["flip_coincidence"] is False
    assert by_id["c1"]["flip_coincidence"] is True


def test_position_equals_round_index():
    claims = [_claim("a", "x", 0, 1), _claim("b", "y", 0, 3)]
    rows = compute_signals(claims, flip_events=[], embedder=HashingFakeEmbedder())
    by_id = {r["claim_id"]: r for r in rows}
    assert by_id["a"]["position"] == 1
    assert by_id["b"]["position"] == 3


def test_first_claim_has_novelty_1():
    claims = [_claim("c0", "only claim", 0, 1)]
    rows = compute_signals(claims, flip_events=[], embedder=HashingFakeEmbedder())
    assert rows[0]["novelty"] == pytest.approx(1.0)


def test_near_duplicate_later_claim_has_low_novelty():
    claims = [
        _claim("c0", "three plus four equals seven", 0, 1),
        _claim("c1", "three plus four equals seven", 1, 1),  # identical
    ]
    rows = compute_signals(claims, flip_events=[], embedder=HashingFakeEmbedder())
    by_id = {r["claim_id"]: r for r in rows}
    assert by_id["c0"]["novelty"] == pytest.approx(1.0)
    # Identical text → cos sim == 1 → novelty == 0.
    assert by_id["c1"]["novelty"] == pytest.approx(0.0, abs=1e-6)


def test_distinct_later_claim_has_high_novelty():
    claims = [
        _claim("c0", "alpha beta", 0, 1),
        _claim("c1", "gamma delta epsilon zeta eta theta", 1, 1),
    ]
    rows = compute_signals(claims, flip_events=[], embedder=HashingFakeEmbedder())
    by_id = {r["claim_id"]: r for r in rows}
    assert by_id["c1"]["novelty"] > 0.3  # lax bound for hashing embedder


def test_referenced_later_true_when_near_duplicate_follows():
    claims = [
        _claim("c0", "reusable claim text here", 0, 1),
        _claim("c1", "reusable claim text here", 1, 2),
    ]
    rows = compute_signals(claims, flip_events=[], embedder=HashingFakeEmbedder())
    by_id = {r["claim_id"]: r for r in rows}
    # c0 has c1 later with identical text → cos_sim ≥ 0.7 → True.
    assert by_id["c0"]["referenced_later"] is True
    # c1 is the last one → nothing after → False.
    assert by_id["c1"]["referenced_later"] is False


def test_referenced_later_uses_strict_later_ordering():
    """A claim is NOT considered referenced by itself."""
    claims = [_claim("c0", "solo claim", 0, 1)]
    rows = compute_signals(claims, flip_events=[], embedder=HashingFakeEmbedder())
    assert rows[0]["referenced_later"] is False


def test_referenced_later_threshold_is_0_7():
    # Module-constant check for auditability.
    assert REFERENCED_LATER_THRESHOLD == 0.7


def test_claim_id_field_preserved():
    claims = [_claim("weird_id_123", "x", 0, 1)]
    rows = compute_signals(claims, flip_events=[], embedder=HashingFakeEmbedder())
    assert rows[0]["claim_id"] == "weird_id_123"


def test_empty_input_returns_empty_list():
    rows = compute_signals([], flip_events=[], embedder=HashingFakeEmbedder())
    assert rows == []


def test_signal_order_by_round_then_agent_then_id():
    claims = [
        _claim("r1_a1", "x1", 1, 1),
        _claim("r2_a0", "x2", 0, 2),
        _claim("r1_a0", "x3", 0, 1),
    ]
    rows = compute_signals(claims, flip_events=[], embedder=HashingFakeEmbedder())
    ids = [r["claim_id"] for r in rows]
    # Stable order that we rely on for novelty (earlier claims come first).
    assert ids == ["r1_a0", "r1_a1", "r2_a0"]
