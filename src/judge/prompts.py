"""Judge prompt templates + score normalization (Σ r̂_t = R invariant).

Templates ask the judge to assign each turn a 0-10 score reflecting that turn's
contribution to the final outcome. We rescale so that the per-turn scores sum
exactly to the trajectory's final reward.
"""

from __future__ import annotations

from src.judge.backend import JudgeRequest, TurnScore

_SYSTEM_PROMPT = """\
You are a careful evaluator of an AI agent's decision-making in a multi-turn task.
Given the full trajectory and the final outcome, assign each turn a score from 0 to 10
reflecting how much that turn contributed to (or detracted from) the final outcome.
Return ONLY a JSON object with the schema:
{"scores": [{"turn": <int>, "score": <float>}, ...]}
The list MUST contain exactly one entry per input turn, in order.
"""


_WEBSHOP_TASK_PRIMER = (
    "Task: WebShop product purchase. Score informative actions (good search queries, "
    "selecting matching products) higher than redundant navigation."
)

_ALFWORLD_TASK_PRIMER = (
    "Task: ALFWorld household task. Score actions that progress toward subgoals higher "
    "than exploration that does not change world state."
)


def _env_primer(env_name: str) -> str:
    if env_name == "webshop":
        return _WEBSHOP_TASK_PRIMER
    if env_name == "alfworld":
        return _ALFWORLD_TASK_PRIMER
    return f"Task env: {env_name}."


def render_user_prompt(request: JudgeRequest) -> str:
    """Render the per-trajectory user message sent to the judge."""
    parts = [_env_primer(request.env_name), ""]
    for t in request.turns:
        parts.append(f"Turn {t.turn_idx}:")
        parts.append(f"  Observation: {t.observation_text}")
        parts.append(f"  Action:      {t.action_text}")
    parts.append("")
    parts.append(f"Final reward (trajectory outcome): {request.final_reward:.4f}")
    parts.append("")
    parts.append(
        "Return JSON: "
        '{"scores": [{"turn": 0, "score": 0.0}, ...]} '
        f"with exactly {len(request.turns)} entries."
    )
    return "\n".join(parts)


def system_prompt() -> str:
    return _SYSTEM_PROMPT


def normalize_scores(raw_scores: list[float], final_reward: float) -> list[float]:
    """Rescale `raw_scores` so they sum to `final_reward` (Σ invariant).

    Edge cases:
    - All-zero raw scores: distribute final_reward uniformly.
    - Sum is non-zero: scale by `final_reward / sum(raw_scores)`.
    """
    if not raw_scores:
        return []
    total = sum(raw_scores)
    n = len(raw_scores)
    if abs(total) < 1e-9:
        share = final_reward / n
        return [share for _ in raw_scores]
    factor = final_reward / total
    return [r * factor for r in raw_scores]


def to_turn_scores(raw_scores: list[float], final_reward: float) -> list[TurnScore]:
    normalized = normalize_scores(raw_scores, final_reward)
    return [
        TurnScore(turn_idx=i, raw_score=float(r), normalized=float(n))
        for i, (r, n) in enumerate(zip(raw_scores, normalized))
    ]
