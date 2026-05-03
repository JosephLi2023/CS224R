from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.algorithms.baseline.policy import SoftmaxPolicy
from src.algorithms.hgpo.policy import HGPOSoftmaxPolicy


@dataclass
class _State:
    valid_actions: list[str]


class TestPolicyTextActions(unittest.TestCase):
    def test_baseline_sample_text_action_uses_valid_action_mapping(self) -> None:
        policy = SoftmaxPolicy(n_actions=3, seed=1)
        state = _State(valid_actions=["a", "b"])

        idx, action = policy.sample_text_action(state=state, fallback="noop")

        self.assertIn(idx, [0, 1, 2])
        self.assertIn(action, ["a", "b"])

    def test_baseline_text_action_fallback_when_no_candidates(self) -> None:
        policy = SoftmaxPolicy(n_actions=2, seed=1)
        state = _State(valid_actions=[])

        idx, action = policy.greedy_text_action(state=state, fallback="look")

        self.assertIn(idx, [0, 1])
        self.assertEqual(action, "look")

    def test_hgpo_inherits_text_action_behavior(self) -> None:
        policy = HGPOSoftmaxPolicy(
            n_actions=4,
            seed=3,
            groups={0: [0, 1], 1: [2, 3]},
            alpha=0.1,
        )
        state = _State(valid_actions=["x", "y", "z"])

        idx, action = policy.greedy_text_action(state=state, fallback="noop")

        self.assertIn(idx, [0, 1, 2, 3])
        self.assertIn(action, ["x", "y", "z"])


if __name__ == "__main__":
    unittest.main()
