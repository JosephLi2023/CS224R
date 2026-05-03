from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalResult:
    avg_return: float


def _pick_webshop_action(policy, state: Any, greedy: bool) -> str:
    idx = policy.greedy_action() if greedy else policy.sample_action()
    valid_actions = getattr(state, "valid_actions", [])
    if valid_actions:
        return str(valid_actions[idx % len(valid_actions)])
    return "search[noop]"


def evaluate_policy(env, policy, episodes: int, env_name: str, greedy: bool = True) -> EvalResult:
    returns = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            if env_name == "webshop":
                action = _pick_webshop_action(policy=policy, state=state, greedy=greedy)
                state, reward, done, _ = env.step(action)
            else:
                action = policy.greedy_action() if greedy else policy.sample_action()
                reward, done = env.step(action)
            total += reward
        returns.append(total)
    return EvalResult(avg_return=sum(returns) / len(returns))
