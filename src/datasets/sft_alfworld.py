"""SFT dataset loader for ALFWorld expert-trajectory JSONL files.

Each `.jsonl` file is one **whole trajectory** consisting of multiple
`{prompt, action, ...}` rows produced by `infra/app_alfworld_sft_gen.py`.
Unlike the WebShop loader (`src/datasets/sft_webshop.py`), each row's
`prompt` field is **pre-rendered** by the same
`render_alfworld_turn_prompt` function the runtime ReAct collector uses.
The SFT loader therefore reads the prompt verbatim and ONLY synthesizes
the SFT target (`Thought: ... \\nAction: ...`).

Why pre-render: the WebShop SFT pipeline duplicated its prompt template
into `src/datasets/sft_webshop.py::default_render_prompt`, and a v3
template-drift bug (see comments at `sft_webshop.py:218-227`) sent
post-SFT R back to zero. By having the trajectory generator call the
runtime renderer directly we *guarantee* the SFT prompt and the GRPO
runtime prompt are byte-identical for every row.

Schema: each JSONL row is a JSON object with at minimum
    {
        "prompt": str,        # pre-rendered ReAct prompt ending with `Thought:`
        "action": str,        # the expert action string (e.g. "go to fridge 1")
        "step_idx": int,      # 0-based step within the trajectory
        "trajectory_id": str, # unique per trajectory (typically the game-file path basename)
        "instruction": str,   # the task instruction (mostly for debugging / summary stats)
        "final_reward": float # 1.0 for successful expert demos, 0.0 otherwise
    }

This module is pure Python (stdlib only) so it's unit-testable locally
without torch / transformers / Modal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SFTExample:
    """One (prompt, target_action) training pair.

    Field shape is intentionally identical to
    `src.datasets.sft_webshop.SFTExample` so downstream tokenizer code
    in the SFT trainer (`infra/app_sft_train_alfworld.py`) can be a
    one-line import swap.
    """

    prompt: str
    action: str
    instruction: str
    step_idx: int
    trajectory_id: str
    final_reward: float


# --------- Action → Thought template ------------------------------------


def _action_to_thought(action: str) -> str:
    """Synthesize a one-sentence ReAct 'Thought' from an ALFWorld action.

    AlfWorld's verb surface is small and regular, so a leading-verb
    dispatch produces consistent natural-language thoughts. Used by
    `synthesize_sft_target` so the SFT label is a full
    `Thought: <reason>\\nAction: <body>` block matching the runtime
    ReAct template exactly. The expert plan is action-only (no
    natural-language thoughts), so we synthesize one consistent
    with each action verb.
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
        # ALFWorld uses both "in" and "on" — handle either.
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
    """Build the multi-line SFT label matching the prompt's `Thought:` ending.

    Returns the string the model should emit AFTER the prompt ends — i.e.
    ` <synthesized thought>\\nAction: <action body>`. The leading space
    is deliberate so it concatenates cleanly with the prompt's trailing
    `Thought:` (which has no trailing whitespace). Mirrors
    `src.datasets.sft_webshop.synthesize_sft_target` exactly.
    """
    return f" {_action_to_thought(action)}\nAction: {action}"


# --------- JSONL loader -------------------------------------------------


def _row_to_example(row: dict[str, Any]) -> SFTExample | None:
    """Convert one JSONL row dict into an SFTExample, or None if malformed.

    Required keys: `prompt` (str) and `action` (str). All other fields
    have safe defaults so the loader is resilient to schema additions.
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

    Args:
        path: filesystem path to the `.jsonl` file produced by
              `infra/app_alfworld_sft_gen.py`. Each line is one
              (prompt, action) pair plus metadata.
        min_reward: drop examples whose `final_reward < min_reward`.
              Default 0.0 keeps all rows. Set to e.g. 0.5 to keep
              only successful expert trajectories (the SFT-gen app
              already filters non-`won` trajectories so any row in
              the file should have `final_reward == 1.0` in practice,
              but the filter is preserved as a defensive guard).
        max_examples: optional cap on returned rows (for smoke tests).

    Returns:
        list of `SFTExample` in the order they appear in the file.
        Skips malformed rows silently (logged via print to stderr).
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
                # Skip malformed lines but keep loading — partial files
                # from a crashed SFT-gen run are still usable.
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
    """Quick stats for printing after a load.

    Mirrors `src.datasets.sft_webshop.summarize_sft_dataset` so the
    trainer's summary-printing code is shape-compatible across envs.
    """
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
