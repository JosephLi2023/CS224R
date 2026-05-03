from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalResult:
    avg_return: float


def evaluate_policy(env, policy, episodes: int, env_name: str, greedy: bool = True) -> EvalResult:
    returns = []
    text_envs = {"webshop", "alfworld"}

    for _ in range(episodes):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            if env_name in text_envs:
                fallback = "search[noop]" if env_name == "webshop" else "look"
                _, action = (
                    policy.greedy_text_action(state=state, fallback=fallback)
                    if greedy
                    else policy.sample_text_action(state=state, fallback=fallback)
                )
                state, reward, done, _ = env.step(action)
            else:
                action = policy.greedy_action() if greedy else policy.sample_action()
                reward, done = env.step(action)
            total += reward
        returns.append(total)
    return EvalResult(avg_return=sum(returns) / len(returns))
