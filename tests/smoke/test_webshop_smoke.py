from __future__ import annotations

import unittest

from src.envs.factory import make_env


class TestWebShopSmoke(unittest.TestCase):
    def test_webshop_reset_and_step_shape(self) -> None:
        env_cfg = {
            "name": "webshop",
            "max_steps": 5,
            "observation_mode": "text",
            "task_split": "train",
            "env_kwargs": {},
        }

        try:
            env = make_env(env_cfg, seed=0)
        except ImportError as exc:
            self.skipTest(f"WebShop dependency not available: {exc}")
            return

        state = env.reset()
        self.assertIsInstance(state.observation_text, str)
        self.assertIsInstance(state.valid_actions, list)

        # Use a valid action if available; otherwise the adapter fallback should handle noop.
        action = state.valid_actions[0] if state.valid_actions else "search[noop]"
        next_state, reward, done, info = env.step(action)

        self.assertIsInstance(next_state.observation_text, str)
        self.assertIsInstance(next_state.valid_actions, list)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn("resolved_action", info)
        self.assertIn("timeout", info)


if __name__ == "__main__":
    unittest.main()
