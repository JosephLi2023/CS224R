"""Unit tests for `src/envs/prompts/react_alfworld.py`.

Pure-Python: no torch / vllm dependency, runs on Mac without any heavy install.

Verification matrix:
1. `test_render_includes_alfworld_system_prompt`
2. `test_render_ends_with_thought_marker`
3. `test_render_includes_instruction_when_provided`
4. `test_render_truncates_history_above_cap`
5. `test_render_lists_valid_actions_with_truncation`
6. `test_parse_react_action_extracts_action_line`
7. `test_parse_react_action_falls_back_to_first_nonempty_line`
8. `test_parse_react_action_handles_empty_generation`
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.envs.prompts.react_alfworld import (
    ALFWORLD_SYSTEM_PROMPT,
    parse_react_action,
    render_alfworld_turn_prompt,
)


@dataclass
class _State:
    observation_text: str = ""
    valid_actions: list[str] = None  # type: ignore[assignment]
    instruction: str = ""

    def __post_init__(self) -> None:
        if self.valid_actions is None:
            self.valid_actions = []


@dataclass
class _Turn:
    observation_text: str
    action_text: str


class TestRenderAlfWorldTurnPrompt(unittest.TestCase):
    def test_render_includes_alfworld_system_prompt(self) -> None:
        state = _State(observation_text="You are in a kitchen.")
        prompt = render_alfworld_turn_prompt(state)
        self.assertIn(ALFWORLD_SYSTEM_PROMPT, prompt)
        # AlfWorld-specific verbs must be in the system prompt so the
        # policy sees them at every turn.
        self.assertIn("go to", prompt)
        self.assertIn("take", prompt)

    def test_render_ends_with_thought_marker(self) -> None:
        state = _State(observation_text="You are in a kitchen.")
        prompt = render_alfworld_turn_prompt(state)
        self.assertTrue(
            prompt.rstrip().endswith("Thought:"),
            f"prompt should end with `Thought:` so the policy is "
            f"prompted to produce a Thought + Action block; got tail "
            f"{prompt[-50:]!r}",
        )

    def test_render_includes_instruction_when_provided(self) -> None:
        state = _State(
            observation_text="You see a fridge.",
            instruction="Put a hot apple in the fridge.",
        )
        prompt = render_alfworld_turn_prompt(state)
        self.assertIn("Put a hot apple in the fridge.", prompt)
        self.assertIn("Task instruction:", prompt)

    def test_render_truncates_history_above_cap(self) -> None:
        state = _State(observation_text="current obs")
        history = [
            _Turn(observation_text=f"obs{i}", action_text=f"act{i}")
            for i in range(10)
        ]
        prompt = render_alfworld_turn_prompt(state, history, max_history_turns=3)
        # Only the last 3 turns should be present.
        self.assertIn("obs9", prompt)
        self.assertIn("obs7", prompt)
        self.assertNotIn("obs0", prompt)
        self.assertIn("7 earlier turns omitted", prompt)

    def test_render_lists_valid_actions_with_truncation(self) -> None:
        valid = [f"go to room {i}" for i in range(20)]
        state = _State(observation_text="hallway", valid_actions=valid)
        prompt = render_alfworld_turn_prompt(state)
        # First 16 should appear; remaining count summarized.
        self.assertIn("Valid actions:", prompt)
        self.assertIn("go to room 0", prompt)
        self.assertIn("go to room 15", prompt)
        self.assertNotIn("go to room 16", prompt)
        self.assertIn("4 more", prompt)


class TestParseReactAction(unittest.TestCase):
    def test_parse_react_action_extracts_action_line(self) -> None:
        gen = (
            "Thought: I should grab the apple.\n"
            "Action: take apple from table\n"
            "Observation: ignored"
        )
        self.assertEqual(parse_react_action(gen), "take apple from table")

    def test_parse_react_action_extracts_complex_alfworld_command(self) -> None:
        gen = "Thought: hot first\nAction: heat apple 1 with microwave 2"
        self.assertEqual(parse_react_action(gen), "heat apple 1 with microwave 2")

    def test_parse_react_action_falls_back_to_first_nonempty_line(self) -> None:
        # No `Action:` prefix anywhere — falls back to the first
        # non-empty line so early-training drift doesn't blow up the
        # rollout collector.
        gen = "\n\n  go to fridge 1\n"
        self.assertEqual(parse_react_action(gen), "go to fridge 1")

    def test_parse_react_action_handles_empty_generation(self) -> None:
        self.assertEqual(parse_react_action(""), "")
        self.assertEqual(parse_react_action("   \n\n"), "")


if __name__ == "__main__":
    unittest.main()
