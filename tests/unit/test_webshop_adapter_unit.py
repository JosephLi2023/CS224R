from __future__ import annotations

import unittest

from src.envs.webshop_adapter import WebShopAdapter


class _FakeWebShopEnv:
    def __init__(self, n_goals: int = 0) -> None:
        self.last_action = None
        self.session = None
        if n_goals > 0:
            self.server = type("Server", (), {"goals": [{}] * n_goals})()

    def reset(self, session=None, **_kwargs):
        self.session = session
        return (
            {"obs": f"initial page session={session}"},
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
    def __init__(self, *args, n_goals: int = 0, **kwargs) -> None:
        self._n_goals = n_goals
        super().__init__(*args, **kwargs)

    def _build_webshop_env(self):
        return _FakeWebShopEnv(n_goals=self._n_goals)


class TestWebShopAdapter(unittest.TestCase):
    def test_reset_builds_state_with_text_and_actions(self) -> None:
        adapter = _TestableWebShopAdapter(max_steps=3)

        state = adapter.reset()

        self.assertIn("initial page", state.observation_text)
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

    def test_reset_task_id_wraps_modulo_goal_pool(self) -> None:
        """Large per-seed offsets (e.g. 32800) must not IndexError upstream."""
        adapter = _TestableWebShopAdapter(max_steps=3, n_goals=6910)
        state = adapter.reset(task_id=32800)
        self.assertEqual(adapter._last_session, 32800 % 6910)
        self.assertIn(f"session={adapter._last_session}", state.observation_text)

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
