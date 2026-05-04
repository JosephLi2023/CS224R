#!/usr/bin/env python
"""End-to-end smoke for OpenAIJudge + JudgeDecomposer (Method A).

Gated on `OPENAI_API_KEY`. Builds 10 synthetic 3-turn TrajectoryGroups (no
WebShop required), runs `JudgeDecomposer.decompose(...)` against the real
gpt-4o-mini backend with a temp JudgeCache, then re-runs and verifies the
cache short-circuits (zero fresh API calls).

Acceptance bullets validated (plan-doc Day 10):
- ≥10 trajectories scored end-to-end via OpenAI.
- Σ-invariant `Σ_t r̂_t == R` per trajectory holds on real outputs.
- Cache hits on rerun (`fresh_calls == 0`).

Cost: ~$0.005 (10 trajectories × 3 turns × ~500 prompt + ~50 completion
tokens at gpt-4o-mini rates). Hard cap: 50 calls.

Usage:
  python scripts/smoke_openai_judge.py
  # → skip with a clear message if OPENAI_API_KEY is unset
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Allow running from repo root without setup.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord
from src.algorithms.hgpo.decomposers.judge import JudgeDecomposer
from src.judge.cache import JudgeCache
from src.judge.openai_backend import OpenAIJudge


def _build_synthetic_groups(n_groups: int = 10, n_turns: int = 3) -> list[TrajectoryGroup]:
    groups: list[TrajectoryGroup] = []
    for g in range(n_groups):
        traj = Trajectory(
            task_id=f"smoke-task-{g}",
            env_name="webshop",
            turns=[
                TurnRecord(
                    turn_idx=t,
                    observation_text=(
                        f"[Smoke obs g{g} t{t}] You are on the search results page "
                        f"for 'lightweight laptop under $800'. Top result: ASUS Vivobook Go 14."
                    ),
                    action_text=(
                        f"think[The Vivobook matches budget and weight constraints; click it]; "
                        f"click[asus_vivobook_go_14]"
                        if t == n_turns - 1
                        else f"think[Need to compare]; search[lightweight laptop under 800]"
                    ),
                )
                for t in range(n_turns)
            ],
            # Vary final reward so the §3.2 invariant is exercised at multiple scales.
            final_reward=float(0.5 + 0.05 * g),
        )
        groups.append(
            TrajectoryGroup(task_id=traj.task_id, env_name="webshop", trajectories=[traj])
        )
    return groups


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "[smoke_openai_judge] OPENAI_API_KEY not set in env — skipping. "
            "Export the key or run inside a Modal container that mounts the "
            "openai-secret Modal Secret."
        )
        return 0

    print("[smoke_openai_judge] Starting OpenAI judge end-to-end smoke …")
    n_groups = 10
    n_turns = 3
    groups = _build_synthetic_groups(n_groups=n_groups, n_turns=n_turns)
    expected_calls = n_groups  # one trajectory per group, one call per trajectory

    backend = OpenAIJudge(
        {
            "model": "gpt-4o-mini",
            "max_retries": 3,
            "temperature": 0.0,
            "timeout_s": 30.0,
            "max_concurrency": 4,
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        cache = JudgeCache(os.path.join(tmp, "judge.sqlite"))
        try:
            decomposer = JudgeDecomposer(
                backend=backend,
                cache=cache,
                max_judge_calls_per_run=50,  # hard cap for cost safety
            )

            t0 = time.time()
            first = [decomposer.decompose(g) for g in groups]
            t1 = time.time()
            calls_first = decomposer._calls_used
            print(
                f"[smoke_openai_judge] First pass: {calls_first} OpenAI calls in "
                f"{t1 - t0:.2f}s (expected {expected_calls})."
            )

            # Σ-invariant check
            invariants: list[bool] = []
            for g, per_group in zip(groups, first):
                for traj, per_turn in zip(g.trajectories, per_group):
                    invariants.append(
                        abs(sum(per_turn) - traj.final_reward) < 1e-9
                    )
            print(
                "[smoke_openai_judge] Σ-invariant per-trajectory: "
                f"{sum(invariants)}/{len(invariants)} pass; values={invariants}"
            )
            assert all(invariants), "§3.2 invariant violated on at least one trajectory"

            # Re-run: should hit cache exclusively (zero fresh calls).
            calls_before = decomposer._calls_used
            t0 = time.time()
            second = [decomposer.decompose(g) for g in groups]
            t1 = time.time()
            fresh_calls = decomposer._calls_used - calls_before
            print(
                f"[smoke_openai_judge] Second pass: {fresh_calls} fresh OpenAI calls "
                f"(expected 0) in {t1 - t0:.2f}s."
            )
            assert fresh_calls == 0, (
                f"Cache short-circuit failed: {fresh_calls} fresh calls on re-run."
            )

            # Sanity: cached values agree with the originally-returned ones.
            for a, b in zip(first, second):
                for ai, bi in zip(a, b):
                    assert ai == bi, "Cached values diverged from initial scoring."

            est_cost_usd = calls_first * 0.0005  # ~ gpt-4o-mini per-call estimate
            print(
                f"[smoke_openai_judge] OK. Estimated spend: ~${est_cost_usd:.4f}. "
                "Method A end-to-end smoke passed."
            )
            return 0
        finally:
            cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
