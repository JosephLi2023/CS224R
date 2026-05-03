"""GRPO trainer building blocks: rollout collection, advantage computation,
loss, and the main HGPOTrainer class. See proposal §3.1 for the math."""

from src.algorithms.grpo.advantage import (
    combine,
    compute_traj_advantages,
    compute_turn_advantages,
    consistency_loss,
)
from src.algorithms.grpo.rollout import TrajectoryGroup, TurnRecord

__all__ = [
    "TrajectoryGroup",
    "TurnRecord",
    "compute_traj_advantages",
    "compute_turn_advantages",
    "combine",
    "consistency_loss",
]
