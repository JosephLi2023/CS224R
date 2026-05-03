"""Integration tests for the judge backend factory + sqlite cache.

These exercise the full read-through cache path that the trainer will use on
Day 11 (`src/algorithms/hgpo/decomposers/judge.py`):

    cached, hashes = cache.get_or_miss(request, model_tag)
    missing_indices = [i for i, c in enumerate(cached) if c is None]
    if missing_indices:
        scores = backend.score_turns(request)        # only invoked on cache miss
        for i in missing_indices:
            cache.put(task_id=..., prefix_hash_=hashes[i], model_tag=model_tag,
                      final_reward=request.final_reward, score=scores[i])

We use a FakeBackend implementing the JudgeBackend Protocol so the test never
touches OpenAI or vLLM, while still proving the integration shape is correct.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from src.judge.backend import (
    JudgeBackend,
    JudgeRequest,
    JudgeTurn,
    TurnScore,
    build_judge,
)
from src.judge.cache import JudgeCache, prefix_hash
from src.judge.openai_backend import OpenAIJudge
from src.judge.prompts import to_turn_scores
from src.judge.vllm_backend import VLLMJudge


# ---------- helpers ----------


class FakeBackend:
    """Deterministic JudgeBackend used to drive the cache integration loop.

    Returns raw scores `[1.0, 2.0, 3.0, ...]` for any request, normalized so
    `Σ normalized == final_reward`. Counts calls for cache-shortcut assertions.
    """

    model_tag: str

    def __init__(self, model_tag: str = "fake-judge-v1") -> None:
        self.model_tag = model_tag
        self.call_count = 0

    def score_turns(self, request: JudgeRequest) -> list[TurnScore]:
        self.call_count += 1
        raw = [float(i + 1) for i in range(len(request.turns))]
        return to_turn_scores(raw, request.final_reward)

    async def score_turns_async(self, request: JudgeRequest) -> list[TurnScore]:
        return self.score_turns(request)


def _request(task_id: str = "task-1", reward: float = 1.0) -> JudgeRequest:
    return JudgeRequest(
        task_id=task_id,
        env_name="webshop",
        turns=[
            JudgeTurn(turn_idx=0, observation_text="page A", action_text="search[laptop]"),
            JudgeTurn(turn_idx=1, observation_text="page B", action_text="click[item-3]"),
            JudgeTurn(turn_idx=2, observation_text="page C", action_text="buy"),
        ],
        final_reward=reward,
    )


def _read_through(
    backend: FakeBackend,
    cache: JudgeCache,
    request: JudgeRequest,
) -> list[TurnScore]:
    """Trainer-side cache loop the integration test is validating."""
    cached, hashes = cache.get_or_miss(request, model_tag=backend.model_tag)
    missing = [i for i, c in enumerate(cached) if c is None]
    if missing:
        fresh = backend.score_turns(request)
        for i in missing:
            cache.put(
                task_id=request.task_id,
                prefix_hash_=hashes[i],
                model_tag=backend.model_tag,
                final_reward=request.final_reward,
                score=fresh[i],
            )
            cached[i] = fresh[i]
    # All slots are filled now; pyright-friendly final list.
    return [c for c in cached if c is not None]


# ---------- factory tests ----------


def _openai_cfg() -> dict:
    return {
        "judge": {
            "backend": "openai",
            "openai": {"model": "gpt-4o-mini", "max_concurrency": 4},
        }
    }


def _vllm_cfg() -> dict:
    return {
        "judge": {
            "backend": "vllm",
            "vllm": {
                "endpoint": "https://example.modal.run/score_turns",
                "model": "qwen2.5-7b-instruct",
            },
        }
    }


def test_build_judge_openai_returns_openai_backend() -> None:
    judge = build_judge(_openai_cfg())
    assert isinstance(judge, OpenAIJudge)
    assert isinstance(judge, JudgeBackend)  # Protocol runtime check
    assert judge.model_tag == "gpt-4o-mini"


def test_build_judge_vllm_returns_vllm_backend() -> None:
    judge = build_judge(_vllm_cfg())
    assert isinstance(judge, VLLMJudge)
    assert isinstance(judge, JudgeBackend)
    assert judge.model_tag == "qwen2.5-7b-instruct"


def test_build_judge_unknown_backend_raises() -> None:
    cfg = {"judge": {"backend": "anthropic"}}
    with pytest.raises(ValueError, match="Unsupported judge backend"):
        build_judge(cfg)


def test_build_judge_backend_field_is_lowercased() -> None:
    cfg = {"judge": {"backend": "OPENAI", "openai": {"model": "gpt-4o-mini"}}}
    judge = build_judge(cfg)
    assert isinstance(judge, OpenAIJudge)


def test_factory_backends_satisfy_jud_backend_protocol_signature() -> None:
    """Both real backends expose `model_tag`, `score_turns`, `score_turns_async`."""
    for cfg in (_openai_cfg(), _vllm_cfg()):
        judge = build_judge(cfg)
        assert hasattr(judge, "model_tag") and isinstance(judge.model_tag, str)
        assert callable(getattr(judge, "score_turns", None))
        assert callable(getattr(judge, "score_turns_async", None))


# ---------- cache + backend integration ----------


def test_cache_short_circuits_repeated_request() -> None:
    """Second identical request must be served entirely from cache."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = JudgeCache(os.path.join(tmp, "judge.sqlite"))
        try:
            backend = FakeBackend()
            req = _request(reward=0.8)

            first = _read_through(backend, cache, req)
            assert backend.call_count == 1
            assert len(first) == len(req.turns)
            assert pytest.approx(sum(s.normalized for s in first), abs=1e-9) == 0.8

            second = _read_through(backend, cache, req)
            # Backend MUST NOT be called again — every turn was cached.
            assert backend.call_count == 1
            # Cached values match what the backend originally produced.
            assert [s.normalized for s in second] == [s.normalized for s in first]
        finally:
            cache.close()


def test_model_tag_partitioning_prevents_cross_backend_hits() -> None:
    """Switching backends must NOT serve stale scores from a different model."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = JudgeCache(os.path.join(tmp, "judge.sqlite"))
        try:
            openai_like = FakeBackend(model_tag="gpt-4o-mini")
            qwen_like = FakeBackend(model_tag="qwen2.5-7b-instruct")
            req = _request(reward=1.0)

            _read_through(openai_like, cache, req)
            assert openai_like.call_count == 1

            # Fresh backend with a different model_tag → must miss and re-query.
            _read_through(qwen_like, cache, req)
            assert qwen_like.call_count == 1

            # Hitting the openai_like path again still short-circuits.
            _read_through(openai_like, cache, req)
            assert openai_like.call_count == 1
        finally:
            cache.close()


def test_cache_persists_across_instances_same_db_path() -> None:
    """A new JudgeCache pointing at the same sqlite file sees prior entries."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "judge.sqlite")
        backend = FakeBackend()
        req = _request(reward=0.5)

        cache_a = JudgeCache(db_path)
        try:
            _read_through(backend, cache_a, req)
            assert backend.call_count == 1
        finally:
            cache_a.close()

        cache_b = JudgeCache(db_path)
        try:
            _read_through(backend, cache_b, req)
            # Persistence: still no extra backend call.
            assert backend.call_count == 1
        finally:
            cache_b.close()


def test_distinct_tasks_do_not_collide_in_cache() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = JudgeCache(os.path.join(tmp, "judge.sqlite"))
        try:
            backend = FakeBackend()
            req_a = _request(task_id="task-A", reward=1.0)
            req_b = _request(task_id="task-B", reward=1.0)

            _read_through(backend, cache, req_a)
            _read_through(backend, cache, req_b)
            # Different task_id keys → both miss, two backend calls.
            assert backend.call_count == 2

            _read_through(backend, cache, req_a)
            _read_through(backend, cache, req_b)
            # Both now cached.
            assert backend.call_count == 2
        finally:
            cache.close()


def test_end_to_end_normalization_invariant_holds_on_cache_hit() -> None:
    """`Σ normalized == final_reward` must survive a round-trip through sqlite."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = JudgeCache(os.path.join(tmp, "judge.sqlite"))
        try:
            backend = FakeBackend()
            for reward in (1.0, 0.0, -0.4, 2.5):
                req = _request(task_id=f"task-{reward}", reward=reward)
                first = _read_through(backend, cache, req)
                second = _read_through(backend, cache, req)
                assert pytest.approx(sum(s.normalized for s in first), abs=1e-9) == reward
                assert pytest.approx(sum(s.normalized for s in second), abs=1e-9) == reward
        finally:
            cache.close()


def test_prefix_hash_changes_on_action_perturbation_invalidates_cache() -> None:
    """Mutating the action text yields a different prefix_hash → cache miss."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = JudgeCache(os.path.join(tmp, "judge.sqlite"))
        try:
            backend = FakeBackend()
            req_a = _request(reward=1.0)
            _read_through(backend, cache, req_a)
            assert backend.call_count == 1

            # Same task_id but turn-0 action changed → different prefix hash on turn 0+.
            mutated = JudgeRequest(
                task_id=req_a.task_id,
                env_name=req_a.env_name,
                turns=[
                    JudgeTurn(turn_idx=0, observation_text="page A", action_text="search[laptop pro]"),
                    *req_a.turns[1:],
                ],
                final_reward=req_a.final_reward,
            )
            _read_through(backend, cache, mutated)
            assert backend.call_count == 2  # backend re-queried for the mutated trajectory

            # Sanity: prefix_hash on turn 0 differs.
            assert prefix_hash("webshop", req_a.turns, 0) != prefix_hash(
                "webshop", mutated.turns, 0
            )
        finally:
            cache.close()
