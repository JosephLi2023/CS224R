"""GRPO trainer building blocks: rollout collection, advantage computation,
loss, KL controller, and the HGPOTrainer class."""

from src.algorithms.grpo.advantage import (
    combine,
    compute_traj_advantages,
    compute_turn_advantages,
    consistency_loss,
)
from src.algorithms.grpo.collectors import (
    CollectStats,
    RolloutCollector,
    RolloutCollectorConfig,
)
from src.algorithms.grpo.kl import AdaptiveKLConfig, AdaptiveKLController
from src.algorithms.grpo.loss import (
    clipped_ppo_term,
    importance_ratio,
    kl_k3_per_token,
    kl_per_token,
    mask_mean,
)
from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord
from src.algorithms.grpo.trainer import (
    HGPOTrainer,
    HGPOTrainerConfig,
    PerTurnDecomposer,
    TrainStepStats,
    progress_decomposer,
)

__all__ = [
    "AdaptiveKLConfig",
    "AdaptiveKLController",
    "CollectStats",
    "HGPOTrainer",
    "HGPOTrainerConfig",
    "PerTurnDecomposer",
    "RolloutCollector",
    "RolloutCollectorConfig",
    "TrainStepStats",
    "Trajectory",
    "TrajectoryGroup",
    "TurnRecord",
    "clipped_ppo_term",
    "combine",
    "compute_traj_advantages",
    "compute_turn_advantages",
    "consistency_loss",
    "importance_ratio",
    "kl_k3_per_token",
    "kl_per_token",
    "mask_mean",
    "progress_decomposer",
]
