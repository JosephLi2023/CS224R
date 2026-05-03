"""Deterministic in-memory fake WebShop env for collector unit tests.

Mirrors the `WebShopAdapter` reset/step shape but does NOT depend on the real
WebShop install (pyserini, spaCy, BM25 index). Behavior:

- `reset(task_id)` → initial state with `instruction` + `observation_text` +
  `valid_actions`.
- `step(action_text)` → progresses through a fixed branching script:
    turn 0: only `search[*]` actions are valid; any `search` advances.
    turn 1: must `click[item-N]`; clicking item-0 yields the highest reward.
    turn 2: must `click[buy]`; rewards finalize.
- Episode terminates after the buy step or after `max_steps`.

Reward model:
- Clicking the canonical "best" item then buying ⇒ reward = 1.0.
- Clicking a wrong item then buying    ⇒ reward = 0.4 (partial match).
- Failing to buy by `max_steps`        ⇒ reward = 0.0.

Determinism: derived purely from `task_id` (an int). Two FakeWebShopEnv
instances with the same task_id reset to byte-identical states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeWebShopState:
    observation_text: str
    valid_actions: list[str] = field(default_factory=list)
    instruction: str = ""
    step_index: int = 0
    raw_info: dict[str, Any] = field(default_factory=dict)


_INSTRUCTIONS = [
    "Find a black laptop bag under $30.",
    "Buy a wireless mouse rated at least 4 stars.",
    "Find a women's running shoe in size 8.",
    "Get a USB-C cable longer than 6 feet.",
]


def _instruction_for(task_id: int) -> str:
    return _INSTRUCTIONS[task_id % len(_INSTRUCTIONS)]


class FakeWebShopEnv:
    """Deterministic 3-turn WebShop-like env for collector smoke tests."""

    def __init__(self, max_steps: int = 8) -> None:
        self.max_steps = max_steps
        self._task_id: int = 0
        self._steps: int = 0
        self._stage: str = "search"  # search → click → buy → done
        self._clicked_correct: bool = False

    # ---- gym-ish API -----------------------------------------------

    def reset(self, task_id: int = 0) -> FakeWebShopState:
        self._task_id = int(task_id)
        self._steps = 0
        self._stage = "search"
        self._clicked_correct = False
        return self._make_state()

    def step(self, action: str) -> tuple[FakeWebShopState, float, bool, dict[str, Any]]:
        action_text = (action or "").strip().lower()
        reward = 0.0
        done = False
        info: dict[str, Any] = {"resolved_action": action_text, "stage": self._stage}

        if self._stage == "search":
            if action_text.startswith("search["):
                self._stage = "click"
            # Any other action wastes a step but doesn't terminate.
        elif self._stage == "click":
            if action_text.startswith("click[item-0"):
                self._clicked_correct = True
                self._stage = "buy"
            elif action_text.startswith("click[item-"):
                self._clicked_correct = False
                self._stage = "buy"
            # Other actions also waste a step.
        elif self._stage == "buy":
            if action_text == "click[buy]":
                reward = 1.0 if self._clicked_correct else 0.4
                done = True
                self._stage = "done"

        self._steps += 1
        if not done and self._steps >= self.max_steps:
            done = True
            info["timeout"] = True

        return self._make_state(), reward, done, info

    # ---- internals -------------------------------------------------

    def _make_state(self) -> FakeWebShopState:
        if self._stage == "search":
            obs = (
                f"You are on the search page. Task: {_instruction_for(self._task_id)} "
                "Issue a search query."
            )
            valid = [
                "search[laptop bag]",
                "search[wireless mouse]",
                "search[running shoe]",
                "search[usb-c cable]",
                "think[narrow it down]",
            ]
        elif self._stage == "click":
            obs = (
                "Search results: 4 items. item-0 (best match), item-1, item-2, item-3."
            )
            valid = [
                "click[item-0]",
                "click[item-1]",
                "click[item-2]",
                "click[item-3]",
                "think[compare items]",
            ]
        elif self._stage == "buy":
            obs = "On product page. Click [buy] to purchase."
            valid = ["click[buy]", "click[back]"]
        else:
            obs = "Episode complete."
            valid = []
        return FakeWebShopState(
            observation_text=obs,
            valid_actions=valid,
            instruction=_instruction_for(self._task_id),
            step_index=self._steps,
            raw_info={"task_id": self._task_id, "stage": self._stage},
        )
