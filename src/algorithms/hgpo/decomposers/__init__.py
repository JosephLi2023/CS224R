"""Per-turn reward decomposers for H-GRPO.

Methods supported (proposal §3):
- "progress" (Method C): per-turn reward = `TurnRecord.raw_env_reward`.
- "judge"    (Method A): per-turn reward = LLM-as-judge normalized score.
- "turnrd"   (Method B): per-turn reward = learned TurnRD model output
  (`r̂_t = α_t · R` from a [CLS] cross-attention head). Day 13 added the
  full learnable surface (`has_learnable_params`, `parameters`,
  `decompose_with_grad`, `state_dict`/`load_state_dict`) plus the
  HGPOTrainer refresh hook + C3 consistency-loss reattach.

Trainer plugs in via `HGPOTrainer(decomposer=...)`. Use `build_decomposer(cfg)`
to instantiate the right decomposer from a method config. For the
`"turnrd"` branch, the factory now returns the `TurnRDDecomposer` *object*
(rather than its bound `.decompose`) so the trainer can reach the
learnable surface; the object's `__call__` keeps the existing
`PerTurnDecomposer` callable contract intact.
"""

from __future__ import annotations

# Re-export the shared progress decomposer so all decomposers live under one
# namespace. We do NOT move it from `src.algorithms.grpo.trainer` to avoid
# breaking existing imports + the H-GRPO Method C unit tests.
from src.algorithms.grpo.trainer import progress_decomposer

from src.algorithms.hgpo.decomposers.base import (
    TurnRewardDecomposer,
    build_decomposer,
)
from src.algorithms.hgpo.decomposers.judge import JudgeDecomposer
from src.algorithms.hgpo.decomposers.turnrd import (
    TurnRDDecomposer,
    build_turnrd_decomposer,
)

__all__ = [
    "TurnRewardDecomposer",
    "build_decomposer",
    "progress_decomposer",
    "JudgeDecomposer",
    "TurnRDDecomposer",
    "build_turnrd_decomposer",
]
