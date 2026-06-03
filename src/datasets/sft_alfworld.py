"""SFT dataset loader for ALFWorld expert-trajectory JSONL files.

Each row's `prompt` is pre-rendered by the same `render_alfworld_turn_prompt`
the runtime collector uses, so the loader reads it verbatim and only
synthesizes the SFT target. Pure stdlib, so it's unit-testable without
torch / transformers / Modal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SFTExample:
    """One (prompt, target_action) training pair; matches sft_webshop.SFTExample."""

    prompt: str
    action: str
    instruction: str
    step_idx: int
    trajectory_id: str
    final_reward: float


# Action -> Thought template


def _action_to_thought(action: str) -> str:
    """Synthesize a one-sentence ReAct 'Thought' from an ALFWorld action.

    The expert plan is action-only, so `synthesize_sft_target` uses a
    leading-verb dispatch to build a full Thought/Action SFT label.
    """
    a = action.strip()
    low = a.lower()
    if not low:
        return "Let me decide what to do next."
    if low.startswith("go to "):
        target = a[len("go to "):]
        return f"I should go to the {target}."
    if low.startswith("take "):
        body = a[len("take "):]
        if " from " in body:
            obj, _, src = body.partition(" from ")
            return f"I'll take the {obj.strip()} from the {src.strip()}."
        return f"I'll pick up the {body.strip()}."
    if low.startswith("put "):
        body = a[len("put "):]
        # ALFWorld uses both "in" and "on" - handle either.
        for sep in (" in/on ", " in ", " on "):
            if sep in body:
                obj, _, dst = body.partition(sep)
                return f"I'll place the {obj.strip()} on the {dst.strip()}."
        return f"I'll place the {body.strip()}."
    if low.startswith("open "):
        target = a[len("open "):]
        return f"I'll open the {target} to see what's inside."
    if low.startswith("close "):
        target = a[len("close "):]
        return f"I'll close the {target}."
    if low.startswith("examine "):
        target = a[len("examine "):]
        return f"Let me examine the {target} more closely."
    if low.startswith("use "):
        target = a[len("use "):]
        return f"I'll use the {target}."
    if low.startswith("clean "):
        body = a[len("clean "):]
        return f"I'll clean the {body.strip()}."
    if low.startswith("heat "):
        body = a[len("heat "):]
        return f"I'll heat the {body.strip()}."
    if low.startswith("cool "):
        body = a[len("cool "):]
        return f"I'll cool the {body.strip()}."
    if low.startswith("slice "):
        body = a[len("slice "):]
        return f"I'll slice the {body.strip()}."
    if low == "look":
        return "Let me look around the room."
    if low == "inventory":
        return "Let me check what I'm carrying."
    return f"I'll {low}."


def synthesize_sft_target(action: str) -> str:
    """Build the SFT label emitted after the prompt's trailing `Thought:`.

    Returns ` <thought>\\nAction: <action>`; the leading space concatenates
    cleanly with the prompt's `Thought:` ending. Mirrors the WebShop version.
    """
    return f" {_action_to_thought(action)}\nAction: {action}"


# JSONL loader


def _row_to_example(row: dict[str, Any]) -> SFTExample | None:
    """Convert one JSONL row into an SFTExample, or None if malformed.

    Requires `prompt` and `action` strings; other fields default safely.
    """
    prompt = row.get("prompt")
    action = row.get("action")
    if not isinstance(prompt, str) or not isinstance(action, str):
        return None
    if not prompt or not action:
        return None
    instruction = row.get("instruction")
    if not isinstance(instruction, str):
        instruction = ""
    step_idx_raw = row.get("step_idx", 0)
    try:
        step_idx = int(step_idx_raw)
    except (TypeError, ValueError):
        step_idx = 0
    trajectory_id = row.get("trajectory_id")
    if not isinstance(trajectory_id, str):
        trajectory_id = ""
    reward_raw = row.get("final_reward", 0.0)
    try:
        final_reward = float(reward_raw)
    except (TypeError, ValueError):
        final_reward = 0.0
    return SFTExample(
        prompt=prompt,
        action=action,
        instruction=instruction,
        step_idx=step_idx,
        trajectory_id=trajectory_id,
        final_reward=final_reward,
    )


def load_sft_examples_from_jsonl(
    path: str,
    *,
    min_reward: float = 0.0,
    max_examples: int | None = None,
) -> list[SFTExample]:
    """Load SFTExamples from a single ALFWorld trajectory JSONL file.

    `min_reward` drops examples whose `final_reward` is below it (default
    0.0 keeps all rows); `max_examples` optionally caps the returned rows.
    Returns examples in file order and skips malformed rows.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SFT JSONL not found: {path}")
    examples: list[SFTExample] = []
    with open(path) as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines but keep loading.
                continue
            if not isinstance(row, dict):
                continue
            ex = _row_to_example(row)
            if ex is None:
                continue
            if ex.final_reward < min_reward:
                continue
            examples.append(ex)
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def summarize_sft_dataset(examples: Iterable[SFTExample]) -> dict[str, Any]:
    """Quick stats for printing after a load; mirrors the WebShop version."""
    examples = list(examples)
    n = len(examples)
    if n == 0:
        return {"n_examples": 0}
    by_action: dict[str, int] = {}
    rewards: list[float] = []
    for ex in examples:
        head = ex.action.split(" ", 1)[0].lower() if ex.action else ""
        by_action[head] = by_action.get(head, 0) + 1
        rewards.append(ex.final_reward)
    return {
        "n_examples": n,
        "n_trajectories": len({ex.trajectory_id for ex in examples}),
        "by_action_kind": by_action,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "reward_mean": sum(rewards) / n,
    }
