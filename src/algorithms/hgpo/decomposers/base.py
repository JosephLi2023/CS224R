"""Decomposer Protocol + factory.

A decomposer maps a `TrajectoryGroup` to per-turn rewards `r_hat_t^i`, matching
the trainer's `PerTurnDecomposer` contract in `src/algorithms/grpo/trainer.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, runtime_checkable

from src.algorithms.grpo.rollout import TrajectoryGroup
from src.algorithms.grpo.trainer import PerTurnDecomposer, progress_decomposer
from src.judge.backend import JudgeBackend
from src.judge.cache import JudgeCache

if TYPE_CHECKING:
    import torch

    from src.algorithms.hgpo.decomposers.counterfactual import (
        ActionParser,
        EnvFactory,
        PromptRenderer,
        SamplingFactory,
        _RunnerLike,
    )
    from src.turnrd.model import TurnRD


@runtime_checkable
class TurnRewardDecomposer(Protocol):
    """Common shape for any per-turn reward decomposer.

    `decompose(group)` returns floats shaped `[K][T_i]`. The judge/turnrd
    methods satisfy `sum_t out[i][t] == final_reward` (within ~1e-9); Method C
    (progress) does not, since raw progress signals don't sum to the reward.
    """

    def decompose(self, group: TrajectoryGroup) -> list[list[float]]:
        ...


def build_decomposer(
    cfg: dict[str, Any],
    *,
    backend: JudgeBackend | None = None,
    cache: JudgeCache | None = None,
    model: "TurnRD | None" = None,
    embedder: Optional[Callable[..., "torch.Tensor"]] = None,
    device: str | None = None,
    runner: Optional["_RunnerLike"] = None,
    env_factory: Optional["EnvFactory"] = None,
    prompt_renderer: Optional["PromptRenderer"] = None,
    action_parser: Optional["ActionParser"] = None,
    sampling_factory: Optional["SamplingFactory"] = None,
) -> PerTurnDecomposer:
    """Dispatch to the decomposer named in `cfg["hgpo"]["decomposer"]`.

    Branches and required kwargs: "progress" (none), "judge" (`backend` +
    `cache`), "turnrd" (`model` + `embedder`), "counterfactual" (`runner`,
    `env_factory`, `prompt_renderer`, `action_parser`, `sampling_factory`).
    Returns a callable matching the `PerTurnDecomposer` signature.
    """
    name = str(cfg.get("hgpo", {}).get("decomposer", "progress")).lower()
    if name == "progress":
        return progress_decomposer
    if name == "judge":
        if backend is None or cache is None:
            raise ValueError(
                "build_decomposer(decomposer='judge'): both `backend` and `cache` "
                "must be provided."
            )
        from src.algorithms.hgpo.decomposers.judge import JudgeDecomposer

        judge_cfg = cfg.get("judge", {})
        limits = judge_cfg.get("limits", {}) if isinstance(judge_cfg, dict) else {}
        max_calls = int(limits.get("max_judge_calls_per_run", 0)) if isinstance(limits, dict) else 0
        decomposer = JudgeDecomposer(
            backend=backend, cache=cache, max_judge_calls_per_run=max_calls or None
        )
        return decomposer.decompose
    if name == "turnrd":
        if model is None or embedder is None:
            raise ValueError(
                "build_decomposer(decomposer='turnrd'): both `model` and `embedder` "
                "must be provided."
            )
        from src.algorithms.hgpo.decomposers.turnrd import build_turnrd_decomposer

        return build_turnrd_decomposer(cfg, model=model, embedder=embedder, device=device)
    if name == "counterfactual":
        if (
            runner is None
            or env_factory is None
            or prompt_renderer is None
            or action_parser is None
            or sampling_factory is None
        ):
            raise ValueError(
                "build_decomposer(decomposer='counterfactual'): all of "
                "`runner`, `env_factory`, `prompt_renderer`, `action_parser`, "
                "and `sampling_factory` must be provided."
            )
        from src.algorithms.hgpo.decomposers.counterfactual import (
            build_counterfactual_decomposer,
        )

        return build_counterfactual_decomposer(
            cfg,
            runner=runner,
            env_factory=env_factory,
            prompt_renderer=prompt_renderer,
            action_parser=action_parser,
            sampling_factory=sampling_factory,
        )
    raise ValueError(
        f"Unknown decomposer name {name!r}; expected 'progress' | 'judge' | "
        "'turnrd' | 'counterfactual'."
    )
