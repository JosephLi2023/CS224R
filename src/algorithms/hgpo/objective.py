from __future__ import annotations


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def group_mean_returns(groups: dict[int, list[int]], action_returns: list[float]) -> list[float]:
    means: list[float] = []
    for gid in sorted(groups.keys()):
        members = groups[gid]
        means.append(_mean([action_returns[m] for m in members]))
    return means


def hgpo_action_bonus(
    groups: dict[int, list[int]],
    action_returns: list[float],
    alpha: float,
) -> list[float]:
    """
    Simple HGPO-like shaping term:
    action bonus = alpha * (group_mean - global_mean)
    """
    global_mean = _mean(action_returns)
    bonus = [0.0 for _ in action_returns]
    for _, members in groups.items():
        gm = _mean([action_returns[m] for m in members])
        delta = alpha * (gm - global_mean)
        for m in members:
            bonus[m] = delta
    return bonus
