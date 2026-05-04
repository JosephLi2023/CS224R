"""Per-turn reward decomposers for H-GRPO.

Methods supported (proposal §3):
- "progress" (Method C): per-turn reward = `TurnRecord.raw_env_reward`.
- "judge"    (Method A): per-turn reward = LLM-as-judge normalized score.
- "turnrd"   (Method B): per-turn reward = learned TurnRD model output
  (`r̂_t = α_t · R` from a [CLS] cross-attention head).
  HGPOTrainer integration + refresh hook lands Day 14 per
  `~/.llms/plans/cs224r_hgpo_method_b_turnrd_m1.plan.md`.

Trainer plugs in via `HGPOTrainer(decomposer=...)`. Use `build_decomposer(cfg)`
to instantiate the right decomposer from a method config.
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
from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer

__all__ = [
    "TurnRewardDecomposer",
    "build_decomposer",
    "progress_decomposer",
    "JudgeDecomposer",
    "TurnRDDecomposer",
]
