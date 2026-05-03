"""JudgeBackend Protocol and factory.

Both backends consume the same trajectory representation and return per-turn
scores normalized so that `sum(scores) == final_reward` (the proposal's
`Σ_t r̂_t = R` invariant). The `model_tag` field is included in cache keys so
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
    normalized: float       # rescaled so Σ normalized == final_reward


@runtime_checkable
class JudgeBackend(Protocol):
    """Common interface for all judge implementations.

    Implementations are responsible only for producing raw per-turn scores;
    normalization to `Σ r̂_t = R` is handled in `src.judge.prompts.normalize_scores`
    by callers, so backends remain stateless and easy to swap.
    """

    model_tag: str  # short identifier baked into cache keys, e.g. "gpt-4o-mini" or "qwen2.5-7b-instruct"

    def score_turns(self, request: JudgeRequest) -> list[TurnScore]:
        """Return a list[TurnScore] aligned with `request.turns`.

        Must produce exactly `len(request.turns)` entries with `turn_idx`
        matching the input ordering. Implementations should raise on parse
        failure rather than silently returning zeros.
        """
        ...

    async def score_turns_async(self, request: JudgeRequest) -> list[TurnScore]:
        """Async variant for batched concurrent calls during rollout."""
        ...


def build_judge(cfg: dict[str, Any]) -> JudgeBackend:
    """Factory: instantiate the backend named in `cfg["judge"]["backend"]`.

    Expected config shape:
      {
        "judge": {
          "backend": "openai" | "vllm",
          "openai": { "model": "gpt-4o-mini", "max_retries": 3, ... },
          "vllm":   { "endpoint": "https://...modal.run/score_turns", "model": "qwen2.5-7b-instruct", ... },
          "cache":  { "path": "/vol/cache/judge.sqlite" }
        }
      }
    """
    backend = str(cfg["judge"]["backend"]).lower()
    if backend == "openai":
        from src.judge.openai_backend import OpenAIJudge
        return OpenAIJudge(cfg["judge"]["openai"])
    if backend == "vllm":
        from src.judge.vllm_backend import VLLMJudge
        return VLLMJudge(cfg["judge"]["vllm"])
    raise ValueError(f"Unsupported judge backend: {backend!r} (expected 'openai' or 'vllm')")
