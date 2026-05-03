from __future__ import annotations

import unittest

from src.envs.webshop_adapter import WebShopAdapter


class _FakeWebShopEnv:
    def __init__(self) -> None:
        self.last_action = None

    def reset(self):
        return (
            {"obs": "initial page"},
            {"valid_actions": ["search[laptop]", "click[item_1]"]},
        )

    def step(self, action: str):
        self.last_action = action
        return (
            {"text": f"after {action}"},
            1.5,
            False,
            {"available_actions": ["buy[now]", "back"]},
        )


class _TestableWebShopAdapter(WebShopAdapter):
    def _build_webshop_env(self):
        return _FakeWebShopEnv()


class TestWebShopAdapter(unittest.TestCase):
    def test_reset_builds_state_with_text_and_actions(self) -> None:
        adapter = _TestableWebShopAdapter(max_steps=3)

        state = adapter.reset()

        self.assertEqual(state.observation_text, "initial page")
        self.assertEqual(state.valid_actions, ["search[laptop]", "click[item_1]"])
        self.assertEqual(state.step_index, 0)

    def test_step_resolves_index_and_sets_info_fields(self) -> None:
        adapter = _TestableWebShopAdapter(max_steps=3)
        adapter.reset()

        next_state, reward, done, info = adapter.step(1)

        self.assertEqual(next_state.observation_text, "after click[item_1]")
        self.assertEqual(reward, 1.5)
        self.assertFalse(done)
        self.assertEqual(info["resolved_action"], "click[item_1]")
        self.assertFalse(info["timeout"])

    def test_step_enforces_timeout(self) -> None:
        adapter = _TestableWebShopAdapter(max_steps=1)
        state = adapter.reset()

        action = state.valid_actions[0]
        _, _, done, info = adapter.step(action)

        self.assertTrue(done)
        self.assertTrue(info["timeout"])

    def test_index_action_without_valid_actions_raises(self) -> None:
        class _NoActionEnv(_FakeWebShopEnv):
            def reset(self):
                return ("obs", {})

        class _NoActionAdapter(WebShopAdapter):
            def _build_webshop_env(self):
                return _NoActionEnv()

        adapter = _NoActionAdapter(max_steps=2)
        adapter.reset()

        with self.assertRaises(ValueError):
            adapter.step(0)


if __name__ == "__main__":
    unittest.main()
