from __future__ import annotations

import math
import random
from typing import Any


def _softmax(logits: list[float]) -> list[float]:
    mx = max(logits)
    exps = [math.exp(x - mx) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


class SoftmaxPolicy:
    def __init__(self, n_actions: int, seed: int) -> None:
        self.n_actions = n_actions
        self.logits = [0.0 for _ in range(n_actions)]
        self.rng = random.Random(seed)

    def probs(self) -> list[float]:
        return _softmax(self.logits)

    def sample_action(self) -> int:
        p = self.probs()
        x = self.rng.random()
        c = 0.0
        for i, pi in enumerate(p):
            c += pi
            if x <= c:
                return i
        return self.n_actions - 1

    def greedy_action(self) -> int:
        best = 0
        best_val = self.logits[0]
        for i in range(1, self.n_actions):
            if self.logits[i] > best_val:
                best = i
                best_val = self.logits[i]
        return best

    def sample_text_action(self, state: Any, fallback: str) -> tuple[int, str]:
        idx = self.sample_action()
        valid_actions = getattr(state, "valid_actions", [])
        if valid_actions:
            return idx, str(valid_actions[idx % len(valid_actions)])
        return idx, fallback

    def greedy_text_action(self, state: Any, fallback: str) -> tuple[int, str]:
        idx = self.greedy_action()
        valid_actions = getattr(state, "valid_actions", [])
        if valid_actions:
            return idx, str(valid_actions[idx % len(valid_actions)])
        return idx, fallback

    def update(self, action_counts: list[float], action_returns: list[float], lr: float) -> None:
        # REINFORCE-style bandit surrogate update over averaged batch returns.
        p = self.probs()
        expected_return = sum(pi * ri for pi, ri in zip(p, action_returns))
        for i in range(self.n_actions):
            advantage = action_returns[i] - expected_return
            grad = action_counts[i] * advantage
            self.logits[i] += lr * grad
