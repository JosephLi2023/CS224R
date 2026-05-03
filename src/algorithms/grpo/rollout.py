"""Rollout-collection dataclasses for H-GRPO.

These are pure Python (no torch) so the advantage math + decomposers can be
unit-tested without a GPU. The actual rollout collector that materializes
these from a vLLM policy + env will land Week 1 Day 4 in
`src/algorithms/grpo/collectors.py`.

Conventions match proposal §3.1:
- A "task" is one environment instance (one WebShop product query, one
  ALFWorld household goal).
- For each task we sample K trajectories τ_1, ..., τ_K from π_θ.
- Each trajectory is a sequence of turns. A turn = (observation, action).
- Each trajectory has a single scalar `final_reward` R_i (sparse outcome).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TurnRecord:
    """One turn in a single trajectory.

    Pure Python view; the trainer will additionally carry token ids /
    log-probs / hidden-state pointers in a parallel torch.Tensor structure.
    """

    turn_idx: int
    observation_text: str
    action_text: str
    # Per-turn intermediate reward as observed from the env (often 0 for
    # sparse-reward tasks; populated by `progress` decomposer for Method C).
    raw_env_reward: float = 0.0
    # ----- token-level fields populated by the rollout collector (Day 4) -----
    # Token ids of the model-generated action (matches `action_text` after detokenization).
    action_token_ids: tuple[int, ...] = ()
    # Per-token log-probs under the rollout-time policy (used by the PPO
    # importance-weight ratio at training time).
    action_token_logprobs: tuple[float, ...] = ()
    # Number of tokens in the prompt at this turn (for context-budget tracking).
    prompt_token_count: int = 0


@dataclass(frozen=True)
class Trajectory:
    """One sampled trajectory τ_i."""

    task_id: str
    env_name: str  # "webshop" | "alfworld"
    turns: list[TurnRecord]
    final_reward: float
    # The seed used when sampling π_θ(τ | task) — kept for reproducibility.
    sample_seed: int = 0

    @property
    def n_turns(self) -> int:
        return len(self.turns)


@dataclass(frozen=True)
class TrajectoryGroup:
    """A K-sized group of trajectories sampled for the same task.

    All `Trajectory.task_id` MUST match `task_id`. Group sizes can be uneven
    in turn count across trajectories — advantage math handles padding via
    explicit per-position masking.
    """

    task_id: str
    env_name: str
    trajectories: list[Trajectory] = field(default_factory=list)

    @property
    def K(self) -> int:
        return len(self.trajectories)

    @property
    def max_turns(self) -> int:
        return max((t.n_turns for t in self.trajectories), default=0)

    def final_rewards(self) -> list[float]:
        return [t.final_reward for t in self.trajectories]

    def per_turn_rewards(self) -> list[list[float]]:
        """List[K] of List[T_i] of `raw_env_reward` from each turn.

        Useful for Method C (progress) decomposers that read environment
        score deltas straight off the trajectory.
        """
        return [[turn.raw_env_reward for turn in traj.turns] for traj in self.trajectories]

    def __post_init__(self) -> None:
        for t in self.trajectories:
            if t.task_id != self.task_id:
                raise ValueError(
                    f"Trajectory task_id={t.task_id!r} does not match group task_id={self.task_id!r}"
                )
            if t.env_name != self.env_name:
                raise ValueError(
                    f"Trajectory env_name={t.env_name!r} does not match group env_name={self.env_name!r}"
                )
