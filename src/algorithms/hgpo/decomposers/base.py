"""Decomposer Protocol + factory.

A decomposer maps a `TrajectoryGroup` to per-turn rewards `r̂_t^i` (one float
per turn per trajectory). This is the contract the trainer's `compute_loss`
consumes — see `src/algorithms/grpo/trainer.py::PerTurnDecomposer` (line 74)
which is structurally identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, runtime_checkable

from src.algorithms.grpo.rollout import TrajectoryGroup
from src.algorithms.grpo.trainer import PerTurnDecomposer, progress_decomposer
from src.judge.backend import JudgeBackend
from src.judge.cache import JudgeCache

if TYPE_CHECKING:
    import torch

    from src.turnrd.model import TurnRD


@runtime_checkable
class TurnRewardDecomposer(Protocol):
    """Common shape for any per-turn reward decomposer.

    `decompose(group)` must return a list shape `[K][T_i]` of floats with
    the §3.2 invariant `Σ_t out[i][t] == group.trajectories[i].final_reward`
    (within ~1e-9) for the judge/turnrd methods. Method C (progress) does
    NOT enforce that invariant since raw env-progress signals don't sum to
    the final reward by construction.
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
) -> PerTurnDecomposer:
    """Dispatch to the decomposer named in `cfg["hgpo"]["decomposer"]`.

    - "progress": returns the existing `progress_decomposer` callable
      (re-exported from `src.algorithms.grpo.trainer`).
    - "judge":    requires `backend` + `cache`; returns a callable wrapping
      `JudgeDecomposer.decompose`.
    - "turnrd":   requires `model` (a `TurnRD` nn.Module) + `embedder` (a
      `Callable[[Trajectory], torch.Tensor]` returning per-turn embeddings of
      shape `[T_i, D]`); returns a callable wrapping
      `TurnRDDecomposer.decompose`. HGPOTrainer integration + refresh hook
      lands Day 14 — today's surface is the model + adapter + tests only
      (see `MEDIUM_FIXES.md::M1` and the
      `~/.llms/plans/cs224r_hgpo_method_b_turnrd_m1.plan.md` plan).

    Returns a callable matching the `PerTurnDecomposer` signature so the
    trainer can plug it in directly.
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

        # Read the optional per-run hard cap from the judge config block.
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
    if name == "residual":
        if model is None or embedder is None:
            raise ValueError(
                "build_decomposer(decomposer='residual'): both `model` and `embedder` "
                "must be provided."
            )
        from src.algorithms.hgpo.decomposers.residual import build_residual_decomposer

        return build_residual_decomposer(cfg, model=model, embedder=embedder, device=device)
    raise ValueError(
        f"Unknown decomposer name {name!r}; expected 'progress' | 'judge' | 'turnrd' | 'residual'."
    )
