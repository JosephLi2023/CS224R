"""JudgeBackend Protocol and factory.

Both backends consume the same trajectory representation and return per-turn
scores normalized so that `sum(scores) == final_reward` (the proposal's
`sum_t r_t = R` invariant). The `model_tag` field is included in cache keys so
OpenAI and Qwen entries never collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class JudgeTurn:
    """One turn of a trajectory presented to the judge."""

    turn_idx: int
    observation_text: str
    action_text: str


@dataclass(frozen=True)
class JudgeRequest:
    """A single trajectory the judge will score per-turn."""

    task_id: str
    env_name: str  # "webshop" | "alfworld"
    turns: list[JudgeTurn]
    final_reward: float


@dataclass(frozen=True)
class TurnScore:
    """Per-turn judge score after normalization."""

    turn_idx: int
    raw_score: float        # judge's raw 0-10 (or env-specific) score
    normalized: float       # rescaled so sum(normalized) == final_reward


@runtime_checkable
class JudgeBackend(Protocol):
    """Common interface for all judge implementations.

    Backends produce raw per-turn scores; normalization to the sum invariant
    is handled by callers, so backends stay stateless and swappable.
    """

    model_tag: str  # short identifier baked into cache keys, e.g. "gpt-4o-mini" or "qwen2.5-7b-instruct"

    def score_turns(self, request: JudgeRequest) -> list[TurnScore]:
        """Return a list[TurnScore] aligned with `request.turns`.

        Must produce exactly `len(request.turns)` entries; raise on parse
        failure rather than returning zeros.
        """
        ...

    async def score_turns_async(self, request: JudgeRequest) -> list[TurnScore]:
        """Async variant for batched concurrent calls during rollout."""
        ...


def build_judge(cfg: dict[str, Any]) -> JudgeBackend:
    """Instantiate the backend named in `cfg["judge"]["backend"]` ("openai" or "vllm")."""
    backend = str(cfg["judge"]["backend"]).lower()
    if backend == "openai":
        from src.judge.openai_backend import OpenAIJudge
        return OpenAIJudge(cfg["judge"]["openai"])
    if backend == "vllm":
        from src.judge.vllm_backend import VLLMJudge
        return VLLMJudge(cfg["judge"]["vllm"])
    raise ValueError(f"Unsupported judge backend: {backend!r} (expected 'openai' or 'vllm')")
