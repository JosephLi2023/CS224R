"""ReAct-style prompt rendering for ALFWorld trajectories.

Pure-Python: no torch / transformers / vLLM dependency, so the prompt format
is unit-testable without a GPU.

Mirrors `src/envs/prompts/react_webshop.py` — same `render_*_turn_prompt`
signature, same `_format_history` truncation pattern, same `parse_react_action`
body. Only the system prompt + action vocabulary differs.

ALFWorld's action surface is a small fixed verb set (`go to`, `take`, `put`,
`open`, `close`, `examine`, `look`, `inventory`, `use`, etc.) typically
combined with object names from the current room. The system prompt below
keeps the ReAct cadence used elsewhere in the project so the rollout
collector + parser are env-agnostic.
"""

from __future__ import annotations

from typing import Any, Iterable

ALFWORLD_SYSTEM_PROMPT = (
    "You are a household task agent in a simulated environment. Your goal is "
    "to complete the task described in the instruction by interacting with "
    "objects in the rooms (navigation, manipulation, inspection). On every "
    "turn you must respond in the ReAct format with exactly one of:\n"
    "  Thought: <one short reasoning sentence>\n"
    "  Action: <a single command, e.g. `go to fridge 1`, `take apple 1 from "
    "table 1`, `put apple 1 in/on fridge 1`, `open drawer 1`, `close "
    "cabinet 1`, `examine countertop 1`, `look`, `inventory`, `use "
    "lamp 1`>\n"
    "Pick a single Action per turn; do not output more than one Action. "
    "When admissible commands are listed in the observation, prefer one of "
    "them verbatim."
)


def _format_history(history: Iterable[Any]) -> str:
    """Render previous (observation, action) turns into a chat-style transcript.

    `history` is an iterable of `TurnRecord` (or any object exposing
    `observation_text` and `action_text` attributes). We avoid importing
    `TurnRecord` directly to keep this module dependency-free.
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


def render_alfworld_turn_prompt(
    state: Any,
    history: Iterable[Any] = (),
    *,
    instruction: str | None = None,
    valid_actions: list[str] | None = None,
    max_history_turns: int = 3,
) -> str:
    """Build the prompt for the agent's next ALFWorld turn.

    Args:
        state: object exposing `observation_text` (str) and optionally
               `valid_actions` (list[str]) and `instruction` (str).
        history: completed turns (TurnRecord-like) so far in this trajectory.
        instruction: explicit goal string; defaults to `state.instruction`
                     if available, else the empty string.
        valid_actions: explicit action whitelist; defaults to
                       `state.valid_actions` if available, else None.
        max_history_turns: keep only the most recent N turns of history in
                           the prompt to bound vLLM context growth (default 3
                           keeps prompts well under the 2048-token cap; ALFWorld
                           trajectories can hit 30+ turns on multi-stage tasks
                           so trimming is essential).

    Returns:
        Prompt string ending with `Thought:` so the model is prompted to
        produce a Thought + Action block.
    """
    obs_text = getattr(state, "observation_text", "") or ""
    if instruction is None:
        instruction = getattr(state, "instruction", "") or ""
    if valid_actions is None:
        valid_actions = list(getattr(state, "valid_actions", []) or [])

    parts: list[str] = [ALFWORLD_SYSTEM_PROMPT, ""]
    if instruction:
        parts.append(f"Task instruction: {instruction}")
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
    """Extract the `Action: ...` line from a model generation.

    Identical body to `react_webshop.parse_react_action` — duplicated here
    rather than imported so the env-dispatch test in
    `tests/unit/test_train_loop_env_dispatch.py` can verify each env's
    parser is independently importable.

    Returns the action body (everything after `Action:` on the first matching
    line). Falls back to the first non-empty line if no `Action:` prefix is
    found, which makes the function robust to the policy's early-training
    drift before it has internalized the ReAct format.
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
