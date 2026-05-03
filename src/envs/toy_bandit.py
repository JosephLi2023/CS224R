from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class ToyBanditConfig:
    n_actions: int
    episode_length: int
    reward_noise_std: float
    action_means: list[float]


class ToyBanditEnv:
    """A tiny stationary bandit-like environment with fixed horizon episodes."""

    def __init__(self, cfg: ToyBanditConfig, seed: int) -> None:
        if cfg.n_actions != len(cfg.action_means):
            raise ValueError("n_actions must match len(action_means)")
        self.cfg = cfg
        self.rng = random.Random(seed)
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def step(self, action: int) -> tuple[float, bool]:
        if not (0 <= action < self.cfg.n_actions):
            raise ValueError(f"action {action} out of range")

        base = self.cfg.action_means[action]
        noise = self.rng.gauss(0.0, self.cfg.reward_noise_std)
        reward = float(base + noise)
        self._t += 1
        done = self._t >= self.cfg.episode_length
        return reward, done
