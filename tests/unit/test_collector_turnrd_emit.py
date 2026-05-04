"""Unit tests for the Day-14 TurnRD replay-buffer producer in
`src.algorithms.grpo.collectors.RolloutCollector._emit_turnrd_records`.

Verification matrix:
1. `test_emit_writes_one_jsonl_row_per_non_empty_trajectory`
2. `test_emit_skips_empty_trajectories`
3. `test_emit_disabled_when_path_is_none`  (default behavior preserved)
4. `test_emit_mode_2_writes_judge_labels_from_decomposer`
5. `test_emit_validates_embedder_required_when_path_set`

Skipped cleanly on hosts without torch (the producer round-trips through
`TurnRDRecord` whose `pad_collate` companion needs torch — but the
record dataclass itself is pure Python, so we still gate at module top).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.algorithms.grpo.collectors import RolloutCollector, RolloutCollectorConfig  # noqa: E402
from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup  # noqa: E402
from src.envs.fake_webshop import FakeWebShopEnv  # noqa: E402
from src.envs.prompts.react_webshop import parse_react_action, render_webshop_turn_prompt  # noqa: E402
from src.turnrd.dataset import TurnRDReplayDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Reused fakes (copied minimally from test_rollout_collector.py to keep
# this test file self-contained — those fakes are not exported as helpers).
# ---------------------------------------------------------------------------


@dataclass
class _FakeGen:
    text: str
    token_ids: tuple
    token_logprobs: tuple
    prompt_token_count: int
    prompt_token_ids: tuple = ()
    finish_reason: str = "stop"


class _FakeRunner:
    def __init__(self, recipe: list[str], ptc: int = 100) -> None:
        self.recipe = list(recipe)
        self.ptc = ptc
        self._cur = 0

    def generate_rich(self, prompts: list[str], sampling) -> list[list[_FakeGen]]:
        n = getattr(sampling, "n", 1)
        t = self.recipe[self._cur % len(self.recipe)]
        self._cur += 1
        ids = tuple(range(10, 10 + max(1, len(t.split()))))
        lps = tuple(-0.1 * (k + 1) for k in range(len(ids)))
        return [[_FakeGen(t, ids, lps, self.ptc) for _ in range(n)] for _ in prompts]


@dataclass
class _S:
    n: int = 1
    temperature: float = 1.0


# Deterministic embedder: returns torch.arange(T*D).view(T, D).float() per
# trajectory. Easy to reason about + cheap to call. CPU fp32 — matches the
# producer's expected output dtype/device.
INPUT_DIM = 8


def _embedder(traj: Trajectory) -> torch.Tensor:
    T = len(traj.turns)
    return torch.arange(T * INPUT_DIM, dtype=torch.float32).view(T, INPUT_DIM)


def _make_collector(
    tmp_path: Path,
    *,
    runner: _FakeRunner,
    emit_path: str | None,
    judge_decomposer=None,
    embedder=_embedder,
) -> RolloutCollector:
    return RolloutCollector(
        runner=runner,
        env_factory=lambda: FakeWebShopEnv(max_steps=8),
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        cfg=RolloutCollectorConfig(max_turns=4),
        turnrd_emit_path=emit_path,
        turnrd_embedder=embedder if emit_path is not None else None,
        judge_decomposer=judge_decomposer,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_emit_writes_one_jsonl_row_per_non_empty_trajectory(tmp_path: Path) -> None:
    """3 trajectories from `FakeWebShopEnv` → 3 JSONL rows. Each row
    parses cleanly through `TurnRDReplayDataset` (which validates schema
    via `TurnRDRecord.__post_init__`)."""
    runner = _FakeRunner(["Action: search[bag]", "Action: click[item-0]", "Action: click[buy]"])
    emit_path = tmp_path / "replay.jsonl"
    collector = _make_collector(tmp_path, runner=runner, emit_path=str(emit_path))

    group, _ = collector.collect_group(task_id=42, env_name="webshop", K=3, sampling=_S())

    assert emit_path.is_file()
    rows = [json.loads(line) for line in emit_path.read_text().splitlines() if line.strip()]
    assert len(rows) == len(group.trajectories)  # K=3 non-empty
    for row in rows:
        assert row["task_id"] == "42"
        assert isinstance(row["turn_embeds"], list) and len(row["turn_embeds"]) > 0
        assert all(len(turn) == INPUT_DIM for turn in row["turn_embeds"])
        assert row["judge_labels"] is None  # Mode 1: no judge_decomposer

    # Round-trip: dataset reader loads what the producer wrote.
    ds = TurnRDReplayDataset(emit_path, mode=1)
    assert len(ds) == 3


def test_emit_skips_empty_trajectories(tmp_path: Path) -> None:
    """A patched group containing one empty trajectory + two non-empty ones
    → only 2 JSONL rows. Matches the dataset reader's drop-and-warn semantics.
    """
    runner = _FakeRunner(["Action: search[m]", "Action: click[item-0]", "Action: click[buy]"])
    emit_path = tmp_path / "replay.jsonl"
    collector = _make_collector(tmp_path, runner=runner, emit_path=str(emit_path))

    # Build a group, then directly call the producer with a synthetic group
    # where one trajectory has 0 turns. (Driving through collect_group + a
    # fake env to land an empty trajectory is awkward; this targets the
    # method directly with a constructed group.)
    real_group, _ = collector.collect_group(task_id=7, env_name="webshop", K=2, sampling=_S())
    # Replace the first trajectory with an empty-turns clone and rewrite.
    empty_traj = Trajectory(task_id="7", env_name="webshop", turns=[], final_reward=0.0)
    mixed_group = TrajectoryGroup(
        task_id="7",
        env_name="webshop",
        trajectories=[empty_traj, real_group.trajectories[0]],
    )
    # Truncate the file so we can re-emit cleanly.
    emit_path.write_text("")
    collector._emit_turnrd_records(mixed_group)

    rows = [json.loads(line) for line in emit_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert len(rows[0]["turn_embeds"]) == len(real_group.trajectories[0].turns)


def test_emit_disabled_when_path_is_none(tmp_path: Path) -> None:
    """`turnrd_emit_path=None` (the default) → no file is created and the
    embedder is never invoked. Preserves Methods A/C behavior + the
    existing flag-driven `infra/app_train_loop.py` path."""
    runner = _FakeRunner(["Action: search[bag]", "Action: click[item-0]", "Action: click[buy]"])

    embedder_calls: list = []

    def tracked_embedder(traj):
        embedder_calls.append(traj.task_id)
        return _embedder(traj)

    # NOTE: tracked_embedder is passed to the collector with emit_path=None,
    # so it should NEVER be called. We pass it via the helper as a regression
    # guard against future refactors that accidentally call the embedder.
    collector = RolloutCollector(
        runner=runner,
        env_factory=lambda: FakeWebShopEnv(max_steps=8),
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        cfg=RolloutCollectorConfig(max_turns=4),
        turnrd_emit_path=None,
        turnrd_embedder=tracked_embedder,
    )

    group, _ = collector.collect_group(task_id=0, env_name="webshop", K=2, sampling=_S())

    assert group.K == 2
    assert embedder_calls == []
    # No replay file dropped anywhere in tmp_path either.
    assert not list(tmp_path.glob("**/*.jsonl"))


def test_emit_mode_2_writes_judge_labels_from_decomposer(tmp_path: Path) -> None:
    """Replaces the dropped `JudgeCache.get_trajectory` test family.

    A stub `judge_decomposer.decompose(group)` returns known per-trajectory
    labels; the producer must write those labels element-wise into each
    row's `judge_labels` field. Verifies the Mode-2 producer pipes the
    labels through unchanged (modulo float casts)."""
    runner = _FakeRunner(["Action: search[m]", "Action: click[item-0]", "Action: click[buy]"])
    emit_path = tmp_path / "replay.jsonl"

    captured_groups: list = []

    class _StubJudgeDecomposer:
        def decompose(self, group: TrajectoryGroup) -> list[list[float]]:
            captured_groups.append(group)
            # One label per turn; deterministic based on (k, t) so we can
            # assert the round-trip below.
            return [
                [0.1 * (i + 1) + 0.01 * t for t in range(len(traj.turns))]
                for i, traj in enumerate(group.trajectories)
            ]

    collector = _make_collector(
        tmp_path,
        runner=runner,
        emit_path=str(emit_path),
        judge_decomposer=_StubJudgeDecomposer(),
    )

    group, _ = collector.collect_group(task_id=99, env_name="webshop", K=2, sampling=_S())

    # The judge decomposer was called exactly once for the whole group.
    assert len(captured_groups) == 1
    assert captured_groups[0] is group

    rows = [json.loads(line) for line in emit_path.read_text().splitlines() if line.strip()]
    assert len(rows) == len(group.trajectories)
    for i, row in enumerate(rows):
        T_i = len(group.trajectories[i].turns)
        expected = [0.1 * (i + 1) + 0.01 * t for t in range(T_i)]
        assert row["judge_labels"] == pytest.approx(expected, abs=1e-6)
        assert len(row["turn_embeds"]) == T_i

    # Mode-2 dataset reader keeps every row (none have None labels).
    ds = TurnRDReplayDataset(emit_path, mode=2)
    assert len(ds) == len(group.trajectories)


def test_emit_validates_embedder_required_when_path_set(tmp_path: Path) -> None:
    """`turnrd_emit_path` without `turnrd_embedder` → ValueError at
    construction time (fail-fast, not at first emit)."""
    runner = _FakeRunner(["Action: search[m]"])
    with pytest.raises(ValueError, match=r"turnrd_embedder"):
        RolloutCollector(
            runner=runner,
            env_factory=lambda: FakeWebShopEnv(max_steps=8),
            prompt_renderer=render_webshop_turn_prompt,
            action_parser=parse_react_action,
            cfg=RolloutCollectorConfig(max_turns=4),
            turnrd_emit_path=str(tmp_path / "x.jsonl"),
            turnrd_embedder=None,
        )
