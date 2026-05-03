from __future__ import annotations


def validate_groups(groups: dict[int, list[int]], n_actions: int) -> None:
    covered: set[int] = set()
    for _, members in groups.items():
        for action in members:
            if action in covered:
                raise ValueError(f"action {action} appears in multiple groups")
            if action < 0 or action >= n_actions:
                raise ValueError(f"action {action} out of range")
            covered.add(action)

    if len(covered) != n_actions:
        raise ValueError("groups must cover every action exactly once")
