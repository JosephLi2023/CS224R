"""ReAct-style prompt rendering for WebShop trajectories.

Pure-Python (no torch / transformers / vLLM) so the prompt format is
unit-testable without a GPU. Uses the `Thought:`/`Action:` ReAct cadence with
the standard WebShop action vocabulary.
"""

from __future__ import annotations

from typing import Any, Iterable

WEBSHOP_SYSTEM_PROMPT = (
    "You are an online shopping agent. Your goal is to find and buy a product "
    "matching the user's instruction. On every turn you must respond in the "
    "ReAct format with exactly one of:\n"
    "  Thought: <one short reasoning sentence>\n"
    "  Action: search[<query>]\n"
    "  Action: click[<item or button>]\n"
    "  Action: think[<short note>]\n"
    "Pick a single Action per turn; do not output more than one Action."
)


def _format_history(history: Iterable[Any]) -> str:
    """Render previous (observation, action) turns into a chat-style transcript.

    Accepts any objects exposing `observation_text` / `action_text`.
    """
    lines: list[str] = []
    for t in history:
        obs = getattr(t, "observation_text", "").strip()
        act = getattr(t, "action_text", "").strip()
        if obs:
            lines.append(f"Observation: {obs}")
        if act:
            lines.append(f"Action: {act}")
    return "\n".join(lines)


def render_webshop_turn_prompt(
    state: Any,
    history: Iterable[Any] = (),
    *,
    instruction: str | None = None,
    valid_actions: list[str] | None = None,
    max_history_turns: int = 3,
) -> str:
    """Build the prompt for the agent's next turn.

    Reads `observation_text`, `instruction`, and `valid_actions` from `state`
    (overridable via kwargs), appends the most recent `max_history_turns`
    turns, and ends with `Thought:` to prompt a Thought + Action block.
    """
    obs_text = getattr(state, "observation_text", "") or ""
    if instruction is None:
        instruction = getattr(state, "instruction", "") or ""
    if valid_actions is None:
        valid_actions = list(getattr(state, "valid_actions", []) or [])

    parts: list[str] = [WEBSHOP_SYSTEM_PROMPT, ""]
    if instruction:
        parts.append(f"User instruction: {instruction}")
        parts.append("")

    history_list = list(history)
    if max_history_turns > 0 and len(history_list) > max_history_turns:
        dropped = len(history_list) - max_history_turns
        parts.append(f"... ({dropped} earlier turns omitted) ...")
        history_list = history_list[-max_history_turns:]
    history_text = _format_history(history_list)
    if history_text:
        parts.append(history_text)

    parts.append(f"Observation: {obs_text.strip()}")
    if valid_actions:
        cap = 16
        head = ", ".join(valid_actions[:cap])
        if len(valid_actions) > cap:
            head += f", … ({len(valid_actions) - cap} more)"
        parts.append(f"Valid actions: {head}")
    parts.append("Thought:")
    return "\n".join(parts)


def parse_react_action(generation: str) -> str:
    """Extract the `Action: ...` body from a model generation.

    Falls back to the first non-empty line when no `Action:` prefix is found.
    """
    for raw in generation.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("action:"):
            return line[len("action:") :].strip() or line
    # Fallback: first non-empty line.
    for raw in generation.splitlines():
        line = raw.strip()
        if line:
            return line
    return ""
