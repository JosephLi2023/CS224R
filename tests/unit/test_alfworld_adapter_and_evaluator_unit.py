from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.envs.alfworld_adapter import ALFWorldAdapter
from src.trainers.evaluator import evaluate_policy


class _FakeALFWorldEnv:
    def __init__(self) -> None:
        self.last_action = None

    def reset(self, split: str = "train"):
        return (
            [f"{split} room"],
            {"admissible_commands": ["look", "open fridge"]},
        )

    def step(self, action: str):
        self.last_action = action
        return (
            {"state": f"after {action}"},
            0.75,
            False,
            {"admissible_actions": ["go north"]},
        )


class _TestableALFWorldAdapter(ALFWorldAdapter):
    def _build_alfworld_env(self):
        return _FakeALFWorldEnv()


class TestALFWorldAdapter(unittest.TestCase):
    def test_reset_and_step_shapes(self) -> None:
        adapter = _TestableALFWorldAdapter(max_steps=4)

        state = adapter.reset()
        next_state, reward, done, info = adapter.step(1)

        self.assertEqual(state.observation_text, "train room")
        self.assertEqual(state.valid_actions, ["look", "open fridge"])
        self.assertEqual(next_state.observation_text, "after open fridge")
        self.assertEqual(next_state.valid_actions, ["go north"])
        self.assertEqual(reward, 0.75)
        self.assertFalse(done)
        self.assertEqual(info["resolved_action"], "open fridge")

    def test_normalize_step_handles_five_tuple(self) -> None:
        adapter = _TestableALFWorldAdapter(max_steps=5)

        obs, reward, done, info = adapter._normalize_step(("o", 2.0, True, False, {"k": "v"}))

        self.assertEqual(obs, "o")
        self.assertEqual(reward, 2.0)
        self.assertTrue(done)
        self.assertEqual(info["k"], "v")


@dataclass
class _EvalState:
    valid_actions: list[str]


class _EvalWebShopEnv:
    def __init__(self) -> None:
        self.steps = 0
        self.actions_seen: list[str] = []

    def reset(self):
        self.steps = 0
        return _EvalState(valid_actions=["search[a]", "search[b]"])

    def step(self, action: str):
        self.steps += 1
        self.actions_seen.append(action)
        done = self.steps >= 2
        next_state = _EvalState(valid_actions=["search[a]", "search[b]"])
        return next_state, 1.0, done, {}


class _EvalPolicy:
    def __init__(self) -> None:
        self.greedy_calls = 0

    def greedy_action(self) -> int:
        self.greedy_calls += 1
        return 1

    def sample_action(self) -> int:
        return 0


class TestEvaluatorWebShop(unittest.TestCase):
    def test_evaluate_policy_uses_text_actions_for_webshop(self) -> None:
        env = _EvalWebShopEnv()
        policy = _EvalPolicy()

        result = evaluate_policy(env=env, policy=policy, episodes=2, env_name="webshop", greedy=True)

        self.assertEqual(result.avg_return, 2.0)
        self.assertTrue(all(a == "search[b]" for a in env.actions_seen))
        self.assertGreater(policy.greedy_calls, 0)


if __name__ == "__main__":
    unittest.main()
