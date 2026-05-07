"""Per-turn reward decomposers for H-GRPO.

Methods supported:
- "progress" (Method C): per-turn reward = `TurnRecord.raw_env_reward`.
- "judge"    (Method A): per-turn reward = LLM-as-judge normalized score.
- "turnrd"   (Method B): per-turn reward = learned TurnRD model output
  (`r̂_t = α_t · R` from a [CLS] cross-attention head). Provides the
  full learnable surface (`has_learnable_params`, `parameters`,
  `decompose_with_grad`, `state_dict`/`load_state_dict`) plus the
  HGPOTrainer refresh hook + C3 consistency-loss reattach.
- "counterfactual" (Method D): per-turn reward = `R − R_baseline_t` from
  short re-rollouts that replace each turn's action with `N` alt samples
  from the policy. See `src/algorithms/hgpo/decomposers/counterfactual.py`.

Trainer plugs in via `HGPOTrainer(decomposer=...)`. Use `build_decomposer(cfg)`
to instantiate the right decomposer from a method config. For the
`"turnrd"` branch, the factory returns the `TurnRDDecomposer` *object*
(rather than its bound `.decompose`) so the trainer can reach the
learnable surface; the object's `__call__` keeps the existing
`PerTurnDecomposer` callable contract intact.

torch-dependent decomposers (`TurnRDDecomposer`) are imported lazily via
`__getattr__` so this package can be imported on hosts without torch
(pure-Python decomposers — judge, progress, counterfactual — still work).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export the shared progress decomposer so all decomposers live under one
# namespace. We do NOT move it from `src.algorithms.grpo.trainer` to avoid
# breaking existing imports + the H-GRPO Method C unit tests.
from src.algorithms.grpo.trainer import progress_decomposer

from src.algorithms.hgpo.decomposers.base import (
    TurnRewardDecomposer,
    build_decomposer,
)
from src.algorithms.hgpo.decomposers.counterfactual import (
    CounterFactualDecomposer,
    build_counterfactual_decomposer,
)
from src.algorithms.hgpo.decomposers.judge import JudgeDecomposer

if TYPE_CHECKING:
    # Type-checker can see TurnRD symbols; runtime imports are lazy below.
    from src.algorithms.hgpo.decomposers.turnrd import (  # noqa: F401
        TurnRDDecomposer,
        build_turnrd_decomposer,
    )


def __getattr__(name: str):
    """Lazy re-export of torch-dependent decomposers.

    Importing `TurnRDDecomposer` requires torch (it's defined in a module
    that does `import torch` at the top). On Mac dev hosts torch is not
    installed; deferring the import lets the rest of the package
    (judge, progress, counterfactual) still work for unit tests.
    """
    if name in ("TurnRDDecomposer", "build_turnrd_decomposer"):
        from src.algorithms.hgpo.decomposers import turnrd as _turnrd

        return getattr(_turnrd, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TurnRewardDecomposer",
    "build_decomposer",
    "progress_decomposer",
    "JudgeDecomposer",
    "TurnRDDecomposer",
    "build_turnrd_decomposer",
    "CounterFactualDecomposer",
    "build_counterfactual_decomposer",
]
