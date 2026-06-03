"""Adaptive KL controller (Schulman PPO style).

Multiplies the KL-to-reference penalty by `coef`, nudging it up/down toward
`target_kl` after each step. `min_coef`/`max_coef` bound it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveKLConfig:
    init_coef: float = 0.04
    target_kl: float = 0.04
    min_coef: float = 1e-4
    max_coef: float = 1.0
    # Multiplicative steps applied when observed KL is far from target.
    increase_factor: float = 1.5
    decrease_factor: float = 0.5
    high_threshold: float = 1.5  # observed > high*target -> increase coef
    low_threshold: float = 0.5   # observed < low*target -> decrease coef


class AdaptiveKLController:
    """Stateful KL coefficient updater. Read `coef` after `update(observed_kl)`."""

    def __init__(self, cfg: AdaptiveKLConfig | None = None) -> None:
        self.cfg = cfg or AdaptiveKLConfig()
        self.coef: float = float(self.cfg.init_coef)
        self.last_observed: float = 0.0
        self.steps: int = 0

    def update(self, observed_kl: float) -> float:
        if observed_kl < 0:
            # KL can be slightly negative due to k1 estimator noise; clamp at 0.
            observed_kl = 0.0
        self.last_observed = float(observed_kl)
        target = self.cfg.target_kl
        high = self.cfg.high_threshold * target
        low = self.cfg.low_threshold * target
        if observed_kl > high:
            self.coef *= self.cfg.increase_factor
        elif observed_kl < low:
            self.coef *= self.cfg.decrease_factor
        self.coef = min(self.cfg.max_coef, max(self.cfg.min_coef, self.coef))
        self.steps += 1
        return self.coef

    def state_dict(self) -> dict:
        return {
            "coef": self.coef,
            "last_observed": self.last_observed,
            "steps": self.steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.coef = float(state.get("coef", self.cfg.init_coef))
        self.last_observed = float(state.get("last_observed", 0.0))
        self.steps = int(state.get("steps", 0))
