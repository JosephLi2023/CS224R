from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.envs.alfworld_adapter import ALFWorldAdapter
from src.trainers.evaluator import evaluate_policy


class _FakeALFWorldEnv:
    def __init__(self) -> None:
        self.last_action = None
        # Game-files list + pointer used by the adapter's deterministic
        # task_id → game-index mapping. The adapter sets `next_game_idx`
        # before calling `reset()`; we expose it so tests can verify
        # selection determinism.
        self.game_files = [f"game_{i}.tw" for i in range(64)]
        self.next_game_idx: int = 0

    def reset(self, split: str = "train"):
        return (
            [f"{split} room (game_idx={self.next_game_idx})"],
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

        # Without task_id, the fake env's `next_game_idx` stays at 0.
        self.assertEqual(state.observation_text, "train room (game_idx=0)")
        self.assertEqual(state.valid_actions, ["look", "open fridge"])
        self.assertEqual(next_state.observation_text, "after open fridge")
        self.assertEqual(next_state.valid_actions, ["go north"])
        self.assertEqual(reward, 0.75)
        self.assertFalse(done)
        self.assertEqual(info["resolved_action"], "open fridge")

    def test_reset_honors_task_id(self) -> None:
        """`reset(task_id=42)` must point the underlying env at game-index
        42 BEFORE calling env.reset() — without this, K parallel adapter
        instances in a GRPO group would each hit a different randomized
        game (breaks the K-trajectories-per-task invariant)."""
        adapter = _TestableALFWorldAdapter(max_steps=4)
        state = adapter.reset(task_id=42)
        # 42 % 64 (the fake's game_files length) == 42.
        self.assertEqual(adapter._last_task_idx, 42)
        # The fake env reflects `next_game_idx` in its observation, so
        # the adapter's hand-off through reset() is end-to-end verifiable.
        self.assertEqual(state.observation_text, "train room (game_idx=42)")

    def test_reset_task_id_wraps_modulo_game_pool(self) -> None:
        """task_id larger than the game-files list wraps via `%`."""
        adapter = _TestableALFWorldAdapter(max_steps=4)
        # 64 game files in the fake; 130 % 64 == 2.
        adapter.reset(task_id=130)
        self.assertEqual(adapter._last_task_idx, 2)

    def test_observation_mode_forwarded_to_constructor(self) -> None:
        """`observation_mode` must reach the adapter (was previously dropped)
        — parallel to WebShop. Verifies via the adapter's stored attribute
        since the fake `_build_alfworld_env` ignores constructor kwargs."""
        adapter = _TestableALFWorldAdapter(
            max_steps=4, observation_mode="text+image"
        )
        self.assertEqual(adapter.observation_mode, "text+image")

    def test_task_split_injected_into_env_kwargs(self) -> None:
        """`task_split` must land in env_kwargs as `train_eval` (the
        upstream `AlfredTWEnv` constructor's canonical key) — not as a
        runtime `reset(split=...)` kwarg, which the prior implementation
        relied on (and which a real `AlfredTWEnv.reset()` doesn't accept)."""
        adapter = _TestableALFWorldAdapter(
            max_steps=4, task_split="eval_in_distribution"
        )
        self.assertEqual(
            adapter.env_kwargs.get("train_eval"), "eval_in_distribution"
        )

    def test_factory_forwards_observation_mode_to_alfworld_adapter(self) -> None:
        """Smoke: `make_env({...}, seed=...)` must thread observation_mode
        through to the adapter (was dropped before this fix)."""
        from src.envs.factory import make_env

        # Patch the build so the factory call doesn't try to import the
        # real alfworld package.
        import src.envs.alfworld_adapter as adapter_mod

        original_build = adapter_mod.ALFWorldAdapter._build_alfworld_env
        adapter_mod.ALFWorldAdapter._build_alfworld_env = lambda self: _FakeALFWorldEnv()
        try:
            env = make_env(
                {
                    "name": "alfworld",
                    "max_steps": 8,
                    "observation_mode": "text+image",
                    "task_split": "train",
                    "env_kwargs": {},
                },
                seed=11,
            )
        finally:
            adapter_mod.ALFWorldAdapter._build_alfworld_env = original_build
        self.assertEqual(env.observation_mode, "text+image")
        # `seed` must be threaded through into env_kwargs (parity with
        # WebShop's reproducible task ordering).
        self.assertEqual(env.env_kwargs.get("seed"), 11)

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

    def greedy_text_action(self, state, fallback: str) -> tuple[int, str]:
        self.greedy_calls += 1
        valid_actions = getattr(state, "valid_actions", [])
        if valid_actions:
            return 1, valid_actions[1]
        return 0, fallback

    def sample_text_action(self, state, fallback: str) -> tuple[int, str]:
        valid_actions = getattr(state, "valid_actions", [])
        if valid_actions:
            return 0, valid_actions[0]
        return 0, fallback


class TestEvaluatorTextEnvs(unittest.TestCase):
    def test_evaluate_policy_uses_text_actions_for_webshop(self) -> None:
        env = _EvalWebShopEnv()
        policy = _EvalPolicy()

        result = evaluate_policy(env=env, policy=policy, episodes=2, env_name="webshop", greedy=True)

        self.assertEqual(result.avg_return, 2.0)
        self.assertTrue(all(a == "search[b]" for a in env.actions_seen))
        self.assertGreater(policy.greedy_calls, 0)

    def test_evaluate_policy_uses_text_actions_for_alfworld(self) -> None:
        env = _EvalWebShopEnv()
        policy = _EvalPolicy()

        result = evaluate_policy(env=env, policy=policy, episodes=1, env_name="alfworld", greedy=True)

        self.assertEqual(result.avg_return, 2.0)
        self.assertEqual(env.actions_seen, ["search[b]", "search[b]"])


if __name__ == "__main__":
    unittest.main()
