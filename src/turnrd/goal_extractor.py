"""Pure-Python AlfWorld goal-text extractor for the TurnRDv2 goal-conditioning ablation.

AlfWorld's Turn 0 observation follows the TextWorld convention:

    -= Welcome to TextWorld, ALFRED! =-

    You are in the middle of a room. Looking quickly around you, you see ...

    Your task is to: <goal text>.

This module provides a single function `extract_goal_text(turn0_obs)` that returns
the trimmed goal substring (e.g. "examine the cd with the desklamp.") or `None` when
no match is found. It is intentionally torch-free + dep-free so the producer side
(in `RolloutCollector._emit_turnrd_records`) can import it without any heavy stack.

Format variants accepted:
- Period vs no period at end of goal.
- Optional trailing whitespace / blank lines.
- Multi-line preamble before the task line (TextWorld decorations).
- Casing of the literal "Your task is to:" — case-insensitive match.
"""
from __future__ import annotations

import re
from typing import Optional

# `Your task is to:` — anchored with non-greedy capture up to a newline (or
# end-of-string). The `[ \t]*` after the colon tolerates extra whitespace
# without consuming the goal text itself.
_GOAL_RE = re.compile(
    r"your task is to:[ \t]*(.+?)\s*(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def extract_goal_text(turn0_obs: str) -> Optional[str]:
    """Return the AlfWorld goal substring from a Turn 0 observation.

    Args:
        turn0_obs: the full text of the Turn 0 observation (typically the
            "Welcome to TextWorld" preamble + room layout + "Your task is
            to: <goal>." line).

    Returns:
        The trimmed goal text (e.g. ``"examine the cd with the desklamp."``)
        or `None` when no `"Your task is to:"` line is found.
    """
    if not isinstance(turn0_obs, str) or not turn0_obs:
        return None
    # `re.DOTALL` lets `.+?` cross newlines, but our `\s*(?:\n|$)` anchor
    # ensures we stop at the first line break — keeps the match scoped to a
    # single line even when the preamble contains other colon-prefixed
    # phrases.
    m = _GOAL_RE.search(turn0_obs)
    if not m:
        return None
    goal = m.group(1).strip()
    return goal if goal else None
