"""LLM-as-judge per-turn reward decomposer (Method A).

For each `Trajectory τ_i` in a `TrajectoryGroup`, builds a `JudgeRequest`
with one `JudgeTurn` per `TurnRecord`, hits the SQLite read-through cache
to skip already-scored turns, and asks the `JudgeBackend` to score any
missing turns. The returned per-turn rewards satisfy the invariant
`Σ_t r̂_t = R` (within ~1e-9) by construction (`to_turn_scores` does the
rescaling — see `src/judge/prompts.py:81`).

Cost guardrail: if the per-run hard cap on judge calls
(`judge.limits.max_judge_calls_per_run` from the run config) would be
exceeded by an upcoming call, `decompose` falls back to a uniform split
(`R / T_i` per turn) and logs a warning rather than silently spending more
$$ than the run budget allows.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup
from src.judge.backend import JudgeBackend, JudgeRequest, JudgeTurn, TurnScore
from src.judge.cache import JudgeCache

_logger = logging.getLogger(__name__)


def _build_request(group_task_id: str, env_name: str, traj: Trajectory, k_index: int) -> JudgeRequest:
    """Build a JudgeRequest for one trajectory.

    `task_id` is qualified as `{group_task_id}#k{k_index}` for a
    correctness reason (NOT defensive uniqueness): the cache stores
    `normalized` scores that are pre-rescaled against THIS trajectory's
    `final_reward`. If two K-samples in a group shared cache entries
    (because their action prefixes match) and they had different `R`s,
    reading the stored `normalized` values would silently produce
    per-turn rewards that DO NOT sum to that trajectory's `R`,
    violating the invariant the cache exists to preserve.
    Per-K qualification makes the entries disjoint so this can't happen.

    Side effect: this defeats the cross-K prefix-sharing that the cache's
    `prefix_hash` was designed to enable (`src/judge/cache.py::prefix_hash`).
    That's acceptable today because `OpenAIJudge` sends the FULL trajectory
    to the model (so per-turn raw scores are not actually prefix-conditioned
    anyway, and cross-K sharing was aspirational). If future judges become
    truly prefix-conditioned, the right fix is to cache `raw_score` only and
    re-normalize at read time using `request.final_reward` — then the
    `#k{i}` qualifier can be dropped.
    """
    qualified_task_id = f"{group_task_id}#k{k_index}"
    turns = [
        JudgeTurn(
            turn_idx=t.turn_idx,
            observation_text=t.observation_text,
            action_text=t.action_text,
        )
        for t in traj.turns
    ]
    return JudgeRequest(
        task_id=qualified_task_id,
        env_name=env_name,
        turns=turns,
        final_reward=float(traj.final_reward),
    )


def _uniform_split(traj: Trajectory) -> list[float]:
    """Fallback when the judge can't be called: uniform-split `R` over T turns."""
    n = len(traj.turns)
    if n == 0:
        return []
    share = float(traj.final_reward) / n
    return [share] * n


class JudgeDecomposer:
    """Per-turn reward decomposer that delegates to a `JudgeBackend`.

    Plugs into `HGPOTrainer(decomposer=JudgeDecomposer(...).decompose)`.
    """

    def __init__(
        self,
        backend: JudgeBackend,
        cache: JudgeCache,
        *,
        max_judge_calls_per_run: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.cache = cache
        # None ⇒ unlimited. 0 ⇒ never call the backend (always fall back
        # to uniform split). >0 ⇒ hard cap.
        self.max_judge_calls_per_run = max_judge_calls_per_run
        self._calls_used: int = 0

    # ------------------------------------------------------------------
    # Internal cache helpers
    # ------------------------------------------------------------------

    def _normalized_from_cached(self, cached: list[TurnScore | None]) -> list[float]:
        # All entries non-None at this point.
        out: list[float] = []
        for c in cached:
            assert c is not None
            out.append(float(c.normalized))
        return out

    def _read_through_sync(self, request: JudgeRequest) -> list[float]:
        """Cache read-through using the sync backend path."""
        cached, hashes = self.cache.get_or_miss(request, model_tag=self.backend.model_tag)
        missing = [i for i, c in enumerate(cached) if c is None]
        if not missing:
            return self._normalized_from_cached(cached)

        if not self._can_make_call():
            _logger.warning(
                "JudgeDecomposer: max_judge_calls_per_run=%s reached; falling back "
                "to uniform-split for task_id=%s.",
                self.max_judge_calls_per_run,
                request.task_id,
            )
            n = len(request.turns)
            share = float(request.final_reward) / n if n else 0.0
            return [share] * n

        self._calls_used += 1
        fresh = self.backend.score_turns(request)
        for i in missing:
            self.cache.put(
                task_id=request.task_id,
                prefix_hash_=hashes[i],
                model_tag=self.backend.model_tag,
                final_reward=request.final_reward,
                score=fresh[i],
            )
            cached[i] = fresh[i]
        return self._normalized_from_cached(cached)

    async def _read_through_async(
        self, request: JudgeRequest, sem: asyncio.Semaphore
    ) -> list[float]:
        """Cache read-through using the async backend path under a semaphore."""
        cached, hashes = self.cache.get_or_miss(request, model_tag=self.backend.model_tag)
        missing = [i for i, c in enumerate(cached) if c is None]
        if not missing:
            return self._normalized_from_cached(cached)

        if not self._can_make_call():
            _logger.warning(
                "JudgeDecomposer: max_judge_calls_per_run=%s reached; falling back "
                "to uniform-split for task_id=%s.",
                self.max_judge_calls_per_run,
                request.task_id,
            )
            n = len(request.turns)
            share = float(request.final_reward) / n if n else 0.0
            return [share] * n

        self._calls_used += 1
        async with sem:
            fresh = await self.backend.score_turns_async(request)
        for i in missing:
            self.cache.put(
                task_id=request.task_id,
                prefix_hash_=hashes[i],
                model_tag=self.backend.model_tag,
                final_reward=request.final_reward,
                score=fresh[i],
            )
            cached[i] = fresh[i]
        return self._normalized_from_cached(cached)

    def _can_make_call(self) -> bool:
        cap = self.max_judge_calls_per_run
        if cap is None:
            return True
        return self._calls_used < int(cap)

    # ------------------------------------------------------------------
    # Public API matching the PerTurnDecomposer signature
    # ------------------------------------------------------------------

    def decompose(self, group: TrajectoryGroup) -> list[list[float]]:
        """Score every trajectory in `group` and return list[K] of list[T_i].

        Uses the sync backend path (one call per trajectory; cache hits
        short-circuit). For concurrent scoring across many groups, prefer
        `decompose_async`.
        """
        out: list[list[float]] = []
        for k, traj in enumerate(group.trajectories):
            if not traj.turns:
                out.append([])
                continue
            req = _build_request(group.task_id, group.env_name, traj, k)
            try:
                per_turn = self._read_through_sync(req)
            except Exception as exc:
                _logger.warning(
                    "JudgeDecomposer: backend.score_turns raised %r for task_id=%s; "
                    "falling back to uniform-split.",
                    exc,
                    req.task_id,
                )
                per_turn = _uniform_split(traj)
            out.append(per_turn)
        return out

    async def decompose_async(self, group: TrajectoryGroup) -> list[list[float]]:
        """Concurrent variant: gates calls with `asyncio.Semaphore`.

        The semaphore concurrency is sourced from `backend.max_concurrency`
        when present, defaulting to 8 otherwise. Per-trajectory results
        retain ordering despite gather() so the returned list aligns with
        `group.trajectories`.
        """
        max_conc = int(getattr(self.backend, "max_concurrency", 8) or 8)
        sem = asyncio.Semaphore(max_conc)

        async def _one(k: int, traj: Trajectory) -> list[float]:
            if not traj.turns:
                return []
            req = _build_request(group.task_id, group.env_name, traj, k)
            try:
                return await self._read_through_async(req, sem)
            except Exception as exc:
                _logger.warning(
                    "JudgeDecomposer: backend.score_turns_async raised %r for "
                    "task_id=%s; falling back to uniform-split.",
                    exc,
                    req.task_id,
                )
                return _uniform_split(traj)

        results = await asyncio.gather(
            *[_one(k, traj) for k, traj in enumerate(group.trajectories)]
        )
        return list(results)
