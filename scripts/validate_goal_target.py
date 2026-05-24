"""Ad-hoc diagnostic for the goal-aware per-turn score rubric.

Loads the existing v3 R9 probe (/tmp/v3_R9_probe.jsonl), runs
`parse_goal_object` + `score_action_against_goal` over every (goal,
action) pair, and prints a histogram of the resulting per-turn scores.

Used to tune the score rubric before committing to the producer; not
run on Modal.

Usage:
    python scripts/validate_goal_target.py
        [--probe /tmp/v3_R9_probe.jsonl]
        [--csv-out /tmp/goal_target_score_histogram.csv]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

# Make the local src/ importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.turnrd.goal_target import (  # noqa: E402
    parse_goal_object,
    score_action_against_goal,
)

_GOAL_RE = re.compile(
    r"your task is to:[ \t]*(.+?)\s*(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_goal(turn0_obs: str) -> str | None:
    m = _GOAL_RE.search(turn0_obs or "")
    return m.group(1).strip() if m else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="/tmp/v3_R9_probe.jsonl")
    parser.add_argument(
        "--csv-out", default="/tmp/goal_target_score_histogram.csv"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.probe):
        sys.exit(f"missing probe file: {args.probe}")

    n_traj = 0
    n_traj_with_goal = 0
    n_traj_parsed = 0
    n_turns = 0
    score_hist = Counter()  # bucketed score → count
    per_traj_positive: list[bool] = []  # is at least one turn > 0?
    rows: list[tuple[str, str, float]] = []  # for CSV

    with open(args.probe) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not d.get("turns"):
                continue
            n_traj += 1
            turn0 = d["turns"][0].get("observation_text", "")
            goal_text = _extract_goal(turn0)
            if goal_text is None:
                per_traj_positive.append(False)
                continue
            n_traj_with_goal += 1
            target, secondary = parse_goal_object(goal_text)
            if target is None:
                per_traj_positive.append(False)
                continue
            n_traj_parsed += 1
            any_positive = False
            for turn in d["turns"]:
                action = turn.get("action_text", "")
                score = score_action_against_goal(action, target, secondary)
                # Bucket to 2-decimal increments for histogram.
                bucket = round(score, 2)
                score_hist[bucket] += 1
                n_turns += 1
                if score > 0:
                    any_positive = True
                rows.append((goal_text, action, score))
            per_traj_positive.append(any_positive)

    print(f"=== goal-target validation ===")
    print(f"trajectories total:                    {n_traj}")
    print(f"trajectories with extracted goal text: {n_traj_with_goal}")
    print(f"trajectories with parseable goal:      {n_traj_parsed}")
    print(f"total (goal, action) pairs scored:     {n_turns}")
    n_pos = sum(1 for x in per_traj_positive if x)
    print(
        f"trajectories with ≥1 positive turn:    {n_pos} "
        f"({100 * n_pos / max(1, len(per_traj_positive)):.1f}%)"
    )

    print()
    print("=== score histogram (per-turn) ===")
    for bucket in sorted(score_hist):
        count = score_hist[bucket]
        bar = "#" * max(1, int(60 * count / max(score_hist.values())))
        print(f"  {bucket:.2f}  {count:5d}  {bar}")

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        with open(args.csv_out, "w") as fh:
            fh.write("goal_text,action_text,score\n")
            for g, a, s in rows:
                # CSV-escape commas / quotes
                g_safe = g.replace('"', '""')
                a_safe = a.replace('"', '""')
                fh.write(f'"{g_safe}","{a_safe}",{s:.4f}\n')
        print(f"\nCSV: {args.csv_out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
