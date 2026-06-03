"""Pure-Python AlfWorld goal-text extractor for the TurnRDv2 goal-conditioning ablation.

AlfWorld's Turn 0 observation follows the TextWorld convention, ending with a
`Your task is to: <goal text>.` line. `extract_goal_text(turn0_obs)` returns the
trimmed goal (or `None`). Torch-free + dep-free so the producer can import it.
"""
from __future__ import annotations

import re
from typing import Optional

# Capture after "Your task is to:" up to the first newline; `[ \t]*` skips
# whitespace after the colon.
_GOAL_RE = re.compile(
    r"your task is to:[ \t]*(.+?)\s*(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def extract_goal_text(turn0_obs: str) -> Optional[str]:
    """Return the trimmed AlfWorld goal substring from a Turn 0 observation,
    or `None` when no `"Your task is to:"` line is found."""
    if not isinstance(turn0_obs, str) or not turn0_obs:
        return None
    # `re.DOTALL` lets `.+?` cross newlines, but the `\s*(?:\n|$)` anchor stops
    # at the first line break, scoping the match to a single line.
    m = _GOAL_RE.search(turn0_obs)
    if not m:
        return None
    goal = m.group(1).strip()
    return goal if goal else None
