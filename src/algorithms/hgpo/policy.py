from __future__ import annotations

from src.algorithms.baseline.policy import SoftmaxPolicy
from src.algorithms.hgpo.objective import hgpo_action_bonus


class HGPOSoftmaxPolicy(SoftmaxPolicy):
    def __init__(self, n_actions: int, seed: int, groups: dict[int, list[int]], alpha: float) -> None:
        super().__init__(n_actions=n_actions, seed=seed)
        self.groups = groups
        self.alpha = alpha

    def update(self, action_counts: list[float], action_returns: list[float], lr: float) -> None:
        bonus = hgpo_action_bonus(
            groups=self.groups,
            action_returns=action_returns,
            alpha=self.alpha,
        )
        shaped_returns = [r + b for r, b in zip(action_returns, bonus)]
        super().update(action_counts=action_counts, action_returns=shaped_returns, lr=lr)
