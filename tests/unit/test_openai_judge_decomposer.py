"""Unit tests for OpenAIJudge backend + JudgeDecomposer.

These tests never touch the OpenAI network. The OpenAI client is monkeypatched
via `OpenAIJudge._ensure_clients` so we can drive the parse / retry / async
paths deterministically.

Coverage:
- `test_openai_judge_score_turns_parses_and_normalizes`
- `test_openai_judge_score_turns_raises_on_count_mismatch`
- `test_openai_judge_score_turns_retries_on_rate_limit`
- `test_openai_judge_async_runs_concurrently_under_semaphore`
- `test_judge_decomposer_returns_correct_shape_and_invariant`
- `test_judge_decomposer_max_calls_cap_falls_back_to_uniform`
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, RateLimitError

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord
from src.algorithms.hgpo.decomposers.judge import JudgeDecomposer
from src.judge.backend import JudgeRequest, JudgeTurn, TurnScore
from src.judge.cache import JudgeCache
from src.judge.openai_backend import OpenAIJudge
from src.judge.prompts import to_turn_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request(reward: float = 1.0, n_turns: int = 3) -> JudgeRequest:
    return JudgeRequest(
        task_id="task-1",
        env_name="webshop",
        turns=[
            JudgeTurn(
                turn_idx=i,
                observation_text=f"obs-{i}",
                action_text=f"act-{i}",
            )
            for i in range(n_turns)
        ],
        final_reward=reward,
    )


def _completion_with_content(content: str) -> Any:
    """Build a minimal mock that mirrors the chat.completions.create return."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock


def _rate_limit_error(message: str = "throttled") -> RateLimitError:
    """Construct a real RateLimitError instance (the lib's __init__ requires
    a populated httpx.Response object)."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError(message, response=response, body=None)


def _api_connection_error() -> APIConnectionError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return APIConnectionError(request=request)


def _install_fake_clients(
    judge: OpenAIJudge,
    sync_create: Any,
    async_create: Any,
) -> None:
    """Patch `_ensure_clients` so it installs the provided fake create() callables.

    `sync_create(**kwargs)` is invoked by `score_turns`, `async_create(**kwargs)`
    (an awaitable callable) by `score_turns_async`.
    """
    sync_client = MagicMock()
    sync_client.chat.completions.create = sync_create
    async_client = MagicMock()
    async_client.chat.completions.create = async_create

    def _ensure_clients_patched(self: OpenAIJudge) -> None:
        self._client = sync_client
        self._async_client = async_client

    judge._ensure_clients = _ensure_clients_patched.__get__(judge, OpenAIJudge)  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# OpenAIJudge.score_turns: parse + normalize
# ---------------------------------------------------------------------------


def test_openai_judge_score_turns_parses_and_normalizes() -> None:
    judge = OpenAIJudge({"model": "gpt-4o-mini", "max_retries": 0, "backoff_base_s": 0.0})
    req = _request(reward=0.8, n_turns=3)
    payload = json.dumps(
        {
            "scores": [
                {"turn": 0, "score": 2.0},
                {"turn": 1, "score": 4.0},
                {"turn": 2, "score": 6.0},
            ]
        }
    )

    def _create(**_kwargs: Any) -> Any:
        return _completion_with_content(payload)

    async def _async_create(**_kwargs: Any) -> Any:
        return _completion_with_content(payload)

    _install_fake_clients(judge, _create, _async_create)

    out = judge.score_turns(req)
    assert isinstance(out, list)
    assert len(out) == 3
    assert [s.turn_idx for s in out] == [0, 1, 2]
    assert [s.raw_score for s in out] == [2.0, 4.0, 6.0]
    # §3.2 invariant: Σ normalized == final_reward
    assert sum(s.normalized for s in out) == pytest.approx(0.8, abs=1e-9)


def test_openai_judge_score_turns_handles_unsorted_turn_indices() -> None:
    """Out-of-order responses should be resorted by `turn` before parsing."""
    judge = OpenAIJudge({"model": "gpt-4o-mini", "max_retries": 0, "backoff_base_s": 0.0})
    req = _request(reward=1.0, n_turns=3)
    payload = json.dumps(
        {
            "scores": [
                {"turn": 2, "score": 3.0},
                {"turn": 0, "score": 1.0},
                {"turn": 1, "score": 2.0},
            ]
        }
    )

    def _create(**_kwargs: Any) -> Any:
        return _completion_with_content(payload)

    async def _async_create(**_kwargs: Any) -> Any:
        return _completion_with_content(payload)

    _install_fake_clients(judge, _create, _async_create)

    out = judge.score_turns(req)
    assert [s.turn_idx for s in out] == [0, 1, 2]
    assert [s.raw_score for s in out] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# OpenAIJudge.score_turns: structural failure modes
# ---------------------------------------------------------------------------


def test_openai_judge_score_turns_raises_on_count_mismatch() -> None:
    judge = OpenAIJudge({"model": "gpt-4o-mini", "max_retries": 0, "backoff_base_s": 0.0})
    req = _request(reward=1.0, n_turns=3)
    payload = json.dumps({"scores": [{"turn": 0, "score": 1.0}, {"turn": 1, "score": 2.0}]})

    def _create(**_kwargs: Any) -> Any:
        return _completion_with_content(payload)

    async def _async_create(**_kwargs: Any) -> Any:
        return _completion_with_content(payload)

    _install_fake_clients(judge, _create, _async_create)

    with pytest.raises(ValueError, match="expected 3 score entries, got 2"):
        judge.score_turns(req)


def test_openai_judge_score_turns_raises_on_invalid_json() -> None:
    judge = OpenAIJudge({"model": "gpt-4o-mini", "max_retries": 0, "backoff_base_s": 0.0})
    req = _request(reward=1.0, n_turns=2)

    def _create(**_kwargs: Any) -> Any:
        return _completion_with_content("not json at all")

    async def _async_create(**_kwargs: Any) -> Any:
        return _completion_with_content("not json at all")

    _install_fake_clients(judge, _create, _async_create)

    with pytest.raises(ValueError, match="not valid JSON"):
        judge.score_turns(req)


# ---------------------------------------------------------------------------
# OpenAIJudge.score_turns: retry on transient errors
# ---------------------------------------------------------------------------


def test_openai_judge_score_turns_retries_on_rate_limit() -> None:
    """RateLimitError once, then success — verifies retry path with backoff=0."""
    judge = OpenAIJudge(
        {"model": "gpt-4o-mini", "max_retries": 3, "backoff_base_s": 0.0}
    )
    req = _request(reward=1.0, n_turns=2)
    payload = json.dumps({"scores": [{"turn": 0, "score": 1.0}, {"turn": 1, "score": 1.0}]})
    call_log: list[int] = []

    def _create(**_kwargs: Any) -> Any:
        call_log.append(1)
        if len(call_log) == 1:
            raise _rate_limit_error("first call throttled")
        return _completion_with_content(payload)

    async def _async_create(**_kwargs: Any) -> Any:
        return _completion_with_content(payload)

    _install_fake_clients(judge, _create, _async_create)

    out = judge.score_turns(req)
    assert len(call_log) == 2  # one failed call + one successful retry
    assert sum(s.normalized for s in out) == pytest.approx(1.0, abs=1e-9)


def test_openai_judge_score_turns_exhausts_retries_then_raises() -> None:
    """All attempts fail → final RateLimitError surfaces to the caller."""
    judge = OpenAIJudge(
        {"model": "gpt-4o-mini", "max_retries": 2, "backoff_base_s": 0.0}
    )
    req = _request(reward=1.0, n_turns=1)
    call_log: list[int] = []

    def _create(**_kwargs: Any) -> Any:
        call_log.append(1)
        raise _rate_limit_error()

    async def _async_create(**_kwargs: Any) -> Any:
        raise _rate_limit_error()

    _install_fake_clients(judge, _create, _async_create)

    with pytest.raises(RateLimitError):
        judge.score_turns(req)
    # max_retries=2 ⇒ initial attempt + 2 retries = 3 calls
    assert len(call_log) == 3


def test_openai_judge_constructs_clients_with_sdk_retries_disabled() -> None:
    """Clients must be constructed with `max_retries=0` so the in-house
    retry loop is the single source of truth (review item I1: SDK's own
    retry layer would otherwise compound, blowing through the budget)."""
    constructed: dict[str, dict[str, Any]] = {}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            constructed.setdefault("sync", {}).update(kwargs)

    class _FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            constructed.setdefault("async", {}).update(kwargs)

    import src.judge.openai_backend as backend_mod

    monkeypatch_attrs: dict[str, Any] = {}

    def _ensure_clients_real(self: OpenAIJudge) -> None:
        # Mirror the production lazy import path but inject our fakes.
        if self._client is None:
            self._client = _FakeClient(timeout=self.timeout_s, max_retries=0)
        if self._async_client is None:
            self._async_client = _FakeAsyncClient(timeout=self.timeout_s, max_retries=0)

    judge = OpenAIJudge({"model": "gpt-4o-mini", "timeout_s": 9.5})
    judge._ensure_clients = _ensure_clients_real.__get__(judge, OpenAIJudge)  # type: ignore[method-assign]
    judge._ensure_clients()

    assert "sync" in constructed and "async" in constructed
    assert constructed["sync"]["max_retries"] == 0
    assert constructed["sync"]["timeout"] == 9.5
    assert constructed["async"]["max_retries"] == 0
    assert constructed["async"]["timeout"] == 9.5
    # Silence "unused" noise for backend_mod / monkeypatch_attrs (kept for grep).
    del backend_mod, monkeypatch_attrs


def test_openai_judge_score_turns_does_not_retry_non_transient_errors() -> None:
    """Generic RuntimeError (e.g. auth failure) should NOT be retried."""
    judge = OpenAIJudge(
        {"model": "gpt-4o-mini", "max_retries": 5, "backoff_base_s": 0.0}
    )
    req = _request(reward=1.0, n_turns=1)
    call_log: list[int] = []

    def _create(**_kwargs: Any) -> Any:
        call_log.append(1)
        raise RuntimeError("auth failure")

    async def _async_create(**_kwargs: Any) -> Any:
        raise RuntimeError("auth failure")

    _install_fake_clients(judge, _create, _async_create)

    with pytest.raises(RuntimeError, match="auth failure"):
        judge.score_turns(req)
    assert len(call_log) == 1  # no retry


def test_openai_judge_score_turns_propagates_keyboard_interrupt() -> None:
    """KeyboardInterrupt is a BaseException; it MUST escape the retry loop
    untouched (review item I2: narrowed `except BaseException` to `except
    Exception` precisely so Ctrl-C and asyncio.CancelledError propagate)."""
    judge = OpenAIJudge(
        {"model": "gpt-4o-mini", "max_retries": 5, "backoff_base_s": 0.0}
    )
    req = _request(reward=1.0, n_turns=1)
    call_log: list[int] = []

    def _create(**_kwargs: Any) -> Any:
        call_log.append(1)
        raise KeyboardInterrupt("user pressed Ctrl-C")

    async def _async_create(**_kwargs: Any) -> Any:
        return _completion_with_content('{"scores": [{"turn": 0, "score": 1.0}]}')

    _install_fake_clients(judge, _create, _async_create)

    with pytest.raises(KeyboardInterrupt):
        judge.score_turns(req)
    # Crucially: no retry — the BaseException blew straight through.
    assert len(call_log) == 1


def test_openai_judge_score_turns_async_propagates_cancellation() -> None:
    """asyncio.CancelledError (BaseException in 3.8+) must escape the
    retry loop so `gather(...)` cancellation cascades work (review I2)."""
    judge = OpenAIJudge(
        {"model": "gpt-4o-mini", "max_retries": 5, "backoff_base_s": 0.0}
    )
    req = _request(reward=1.0, n_turns=1)
    call_log: list[int] = []

    def _create(**_kwargs: Any) -> Any:
        return _completion_with_content('{"scores": [{"turn": 0, "score": 1.0}]}')

    async def _async_create(**_kwargs: Any) -> Any:
        call_log.append(1)
        raise asyncio.CancelledError()

    _install_fake_clients(judge, _create, _async_create)

    async def _run() -> None:
        await judge.score_turns_async(req)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_run())
    assert len(call_log) == 1  # no retry on cancellation


# ---------------------------------------------------------------------------
# OpenAIJudge.score_turns_async: smoke + concurrency under semaphore
# ---------------------------------------------------------------------------


def test_openai_judge_async_runs_concurrently_under_semaphore() -> None:
    """K concurrent score_turns_async calls; verify all complete and the
    semaphore-gated decomposer never exceeds its concurrency cap."""
    judge = OpenAIJudge(
        {"model": "gpt-4o-mini", "max_retries": 0, "backoff_base_s": 0.0, "max_concurrency": 2}
    )
    payload = json.dumps({"scores": [{"turn": 0, "score": 1.0}, {"turn": 1, "score": 1.0}]})

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    def _create(**_kwargs: Any) -> Any:
        # Sync client is unused on the async path.
        return _completion_with_content(payload)

    async def _async_create(**_kwargs: Any) -> Any:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            if in_flight > max_in_flight:
                max_in_flight = in_flight
        # Yield so other coroutines can interleave; this exercises the
        # caller-owned semaphore (here we use the decomposer's).
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return _completion_with_content(payload)

    _install_fake_clients(judge, _create, _async_create)

    # Drive concurrency through JudgeDecomposer.decompose_async, which is the
    # production caller that owns the asyncio.Semaphore (max_concurrency=2).
    with tempfile.TemporaryDirectory() as tmp:
        cache = JudgeCache(os.path.join(tmp, "judge.sqlite"))
        try:
            decomposer = JudgeDecomposer(backend=judge, cache=cache)
            group = TrajectoryGroup(
                task_id="async-task",
                env_name="webshop",
                trajectories=[
                    Trajectory(
                        task_id="async-task",
                        env_name="webshop",
                        turns=[
                            TurnRecord(turn_idx=0, observation_text="o0", action_text="a0"),
                            TurnRecord(turn_idx=1, observation_text="o1", action_text="a1"),
                        ],
                        final_reward=1.0,
                    )
                    for _ in range(5)
                ],
            )
            out = asyncio.run(decomposer.decompose_async(group))
        finally:
            cache.close()

    assert len(out) == 5
    for traj_scores in out:
        assert len(traj_scores) == 2
        assert sum(traj_scores) == pytest.approx(1.0, abs=1e-9)
    # Semaphore bound: max_concurrency=2 ⇒ at most 2 in flight simultaneously.
    assert 1 <= max_in_flight <= 2


# ---------------------------------------------------------------------------
# JudgeDecomposer integration: shape, invariant, cache short-circuit
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Deterministic JudgeBackend stand-in for tests.

    Returns raw scores `[1.0, 2.0, 3.0, ...]` for any request, normalized so
    `Σ normalized == final_reward`. Counts calls so the cache-shortcut
    assertion can verify second `decompose` is fully cached.
    """

    model_tag: str

    def __init__(self, model_tag: str = "fake-judge-v1", max_concurrency: int = 4) -> None:
        self.model_tag = model_tag
        self.max_concurrency = max_concurrency
        self.call_count = 0

    def score_turns(self, request: JudgeRequest) -> list[TurnScore]:
        self.call_count += 1
        raw = [float(i + 1) for i in range(len(request.turns))]
        return to_turn_scores(raw, request.final_reward)

    async def score_turns_async(self, request: JudgeRequest) -> list[TurnScore]:
        return self.score_turns(request)


def _make_group(rewards: list[float], n_turns: int = 3) -> TrajectoryGroup:
    trajs = []
    for i, r in enumerate(rewards):
        trajs.append(
            Trajectory(
                task_id="task-X",
                env_name="webshop",
                turns=[
                    TurnRecord(
                        turn_idx=t,
                        observation_text=f"obs-{i}-{t}",
                        # Vary action text per K-sample so each trajectory has
                        # a distinct prefix_hash (the cache keys on prefix).
                        action_text=f"act-i{i}-t{t}",
                    )
                    for t in range(n_turns)
                ],
                final_reward=r,
            )
        )
    return TrajectoryGroup(task_id="task-X", env_name="webshop", trajectories=trajs)


def test_judge_decomposer_returns_correct_shape_and_invariant() -> None:
    backend = _FakeBackend()
    rewards = [1.0, 0.5, -0.4]
    group = _make_group(rewards, n_turns=4)

    with tempfile.TemporaryDirectory() as tmp:
        cache = JudgeCache(os.path.join(tmp, "judge.sqlite"))
        try:
            decomposer = JudgeDecomposer(backend=backend, cache=cache)
            out = decomposer.decompose(group)

            # Shape: K trajectories, each with T_i turns.
            assert len(out) == len(rewards)
            for per_traj in out:
                assert len(per_traj) == 4

            # §3.2 invariant: Σ_t out[i][t] == final_reward.
            for i, r in enumerate(rewards):
                assert sum(out[i]) == pytest.approx(r, abs=1e-9)

            # K backend calls on first invocation (one per trajectory).
            assert backend.call_count == len(rewards)

            # Re-run: every turn is now cached → 0 fresh backend calls.
            out2 = decomposer.decompose(group)
            assert backend.call_count == len(rewards)  # unchanged
            for i in range(len(rewards)):
                assert out2[i] == out[i]
        finally:
            cache.close()


def test_judge_decomposer_max_calls_cap_falls_back_to_uniform() -> None:
    """`max_judge_calls_per_run=0` → never call backend, return uniform splits."""
    backend = _FakeBackend()
    rewards = [1.0, 0.5]
    group = _make_group(rewards, n_turns=3)

    with tempfile.TemporaryDirectory() as tmp:
        cache = JudgeCache(os.path.join(tmp, "judge.sqlite"))
        try:
            decomposer = JudgeDecomposer(
                backend=backend, cache=cache, max_judge_calls_per_run=0
            )
            out = decomposer.decompose(group)

            assert backend.call_count == 0
            assert len(out) == 2
            for i, r in enumerate(rewards):
                assert len(out[i]) == 3
                # Uniform split: each turn = R / T.
                expected = r / 3
                for v in out[i]:
                    assert v == pytest.approx(expected, abs=1e-9)
                assert sum(out[i]) == pytest.approx(r, abs=1e-9)
        finally:
            cache.close()


def test_judge_decomposer_handles_empty_trajectory() -> None:
    """Trajectories with zero turns return empty per-turn lists; no backend call."""
    backend = _FakeBackend()
    group = TrajectoryGroup(
        task_id="task-empty",
        env_name="webshop",
        trajectories=[
            Trajectory(task_id="task-empty", env_name="webshop", turns=[], final_reward=0.0)
        ],
    )

    with tempfile.TemporaryDirectory() as tmp:
        cache = JudgeCache(os.path.join(tmp, "judge.sqlite"))
        try:
            decomposer = JudgeDecomposer(backend=backend, cache=cache)
            out = decomposer.decompose(group)
            assert out == [[]]
            assert backend.call_count == 0
        finally:
            cache.close()


def test_judge_decomposer_falls_back_when_backend_raises() -> None:
    """Non-recoverable backend error → uniform-split fallback (logged warning)."""

    class _BrokenBackend:
        model_tag = "broken-v1"
        max_concurrency = 1

        def score_turns(self, request: JudgeRequest) -> list[TurnScore]:  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated backend outage")

        async def score_turns_async(self, request: JudgeRequest) -> list[TurnScore]:  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated backend outage")

    backend = _BrokenBackend()
    rewards = [0.7]
    group = _make_group(rewards, n_turns=4)

    with tempfile.TemporaryDirectory() as tmp:
        cache = JudgeCache(os.path.join(tmp, "judge.sqlite"))
        try:
            decomposer = JudgeDecomposer(backend=backend, cache=cache)  # type: ignore[arg-type]
            out = decomposer.decompose(group)

            assert len(out) == 1
            assert len(out[0]) == 4
            # Σ-invariant still holds via uniform fallback.
            assert sum(out[0]) == pytest.approx(0.7, abs=1e-9)
        finally:
            cache.close()
