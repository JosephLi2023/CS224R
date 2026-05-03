"""Unit tests for the judge cache and score-normalization invariants.

These tests are runnable today against the scaffold (no LLM dependency).
"""

from __future__ import annotations

import os
import tempfile

from src.judge.backend import JudgeRequest, JudgeTurn, TurnScore
from src.judge.cache import JudgeCache, prefix_hash
from src.judge.prompts import normalize_scores, to_turn_scores


def _make_request(reward: float = 1.0) -> JudgeRequest:
    return JudgeRequest(
        task_id="task-1",
        env_name="webshop",
        turns=[
            JudgeTurn(turn_idx=0, observation_text="page A", action_text="search[laptop]"),
            JudgeTurn(turn_idx=1, observation_text="page B", action_text="click[item-3]"),
            JudgeTurn(turn_idx=2, observation_text="page C", action_text="buy"),
        ],
        final_reward=reward,
    )


def test_normalize_scores_sums_to_final_reward() -> None:
    raw = [2.0, 5.0, 3.0]
    out = normalize_scores(raw, final_reward=1.0)
    assert abs(sum(out) - 1.0) < 1e-9


def test_normalize_scores_zero_input_distributes_uniformly() -> None:
    out = normalize_scores([0.0, 0.0, 0.0], final_reward=0.6)
    assert all(abs(x - 0.2) < 1e-9 for x in out)
    assert abs(sum(out) - 0.6) < 1e-9


def test_normalize_scores_handles_negative_reward() -> None:
    raw = [1.0, 1.0, 2.0]
    out = normalize_scores(raw, final_reward=-1.0)
    assert abs(sum(out) - (-1.0)) < 1e-9


def test_to_turn_scores_preserves_indices() -> None:
    scores = to_turn_scores([1.0, 2.0, 3.0], final_reward=1.0)
    assert [s.turn_idx for s in scores] == [0, 1, 2]
    assert abs(sum(s.normalized for s in scores) - 1.0) < 1e-9


def test_prefix_hash_deterministic_and_grows_per_turn() -> None:
    req = _make_request()
    h0 = prefix_hash("webshop", req.turns, 0)
    h1 = prefix_hash("webshop", req.turns, 1)
    assert h0 != h1
    # Repeating the call yields the same hash.
    assert prefix_hash("webshop", req.turns, 0) == h0


def test_judge_cache_round_trip_and_partition_by_model_tag() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "judge.sqlite")
        cache = JudgeCache(path)
        try:
            req = _make_request(reward=1.0)
            score = TurnScore(turn_idx=0, raw_score=4.0, normalized=0.4)
            h = prefix_hash("webshop", req.turns, 0)

            assert cache.get("task-1", 0, h, "gpt-4o-mini") is None
            cache.put(
                task_id="task-1",
                prefix_hash_=h,
                model_tag="gpt-4o-mini",
                final_reward=1.0,
                score=score,
            )
            hit = cache.get("task-1", 0, h, "gpt-4o-mini")
            assert hit is not None
            assert abs(hit.normalized - 0.4) < 1e-9
            # A different model_tag must not hit.
            assert cache.get("task-1", 0, h, "qwen2.5-7b-instruct") is None
        finally:
            cache.close()


def test_judge_cache_get_or_miss_returns_aligned_lists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "judge.sqlite")
        cache = JudgeCache(path)
        try:
            req = _make_request(reward=1.0)
            cached, hashes = cache.get_or_miss(req, model_tag="gpt-4o-mini")
            assert len(cached) == len(req.turns)
            assert len(hashes) == len(req.turns)
            assert all(c is None for c in cached)
            assert len(set(hashes)) == len(req.turns)  # each turn hashes uniquely
        finally:
            cache.close()
