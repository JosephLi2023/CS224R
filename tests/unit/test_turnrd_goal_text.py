"""Unit tests for the AlfWorld goal-text extraction + plumbing.

These tests survive the cleanup of the (failed) goal-conditioning V-head
ablation. They cover the pieces that remain useful for the goal-aware
supervision proposal:
- `src.turnrd.goal_extractor.extract_goal_text` — pure-Python parser.
- `TurnRDRecord.goal_text` — optional field, round-trips through JSONL.
- `RolloutCollector(turnrd_emit_goal_text=True)` — producer populates
  `goal_text` on emitted rows.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict

import pytest

# Goal-extractor is pure-Python — no torch gate.
from src.turnrd.goal_extractor import extract_goal_text


# ---------------------------------------------------------------------------
# 1. Goal extractor
# ---------------------------------------------------------------------------


def test_goal_extractor_alfworld_canonical() -> None:
    obs = (
        "-= Welcome to TextWorld, ALFRED! =-\n"
        "\n"
        "You are in the middle of a room.\n"
        "\n"
        "Your task is to: examine the cd with the desklamp.\n"
    )
    assert extract_goal_text(obs) == "examine the cd with the desklamp."


def test_goal_extractor_no_period() -> None:
    assert extract_goal_text("Your task is to: find apple") == "find apple"


def test_goal_extractor_case_insensitive() -> None:
    assert extract_goal_text("YOUR TASK IS TO: HELLO.\n") == "HELLO."


def test_goal_extractor_trailing_whitespace() -> None:
    assert extract_goal_text("Your task is to:    foo  \n") == "foo"


def test_goal_extractor_missing_returns_none() -> None:
    assert extract_goal_text("no goal anywhere here") is None
    assert extract_goal_text("") is None
    assert extract_goal_text(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. TurnRDRecord.goal_text round-trip
# ---------------------------------------------------------------------------


torch = pytest.importorskip("torch")  # noqa: E402

from src.turnrd.dataset import (  # noqa: E402
    TurnRDRecord,
    TurnRDReplayDataset,
)


def test_record_validates_goal_text_type() -> None:
    with pytest.raises(ValueError, match="goal_text must be str"):
        TurnRDRecord(
            task_id="t",
            turn_embeds=[[1.0, 2.0]],
            final_reward=0.0,
            goal_text=123,  # type: ignore[arg-type]
        )


def test_record_allows_goal_text_or_none() -> None:
    r1 = TurnRDRecord(
        task_id="t",
        turn_embeds=[[1.0, 2.0]],
        final_reward=0.0,
        goal_text="find apple",
    )
    assert r1.goal_text == "find apple"
    r2 = TurnRDRecord(task_id="t", turn_embeds=[[1.0]], final_reward=0.0)
    assert r2.goal_text is None


def test_dataset_loader_roundtrips_goal_text() -> None:
    D = 4
    recs = [
        TurnRDRecord(
            task_id="a",
            turn_embeds=[[0.1] * D, [0.2] * D],
            final_reward=1.0,
            goal_text="goal A",
        ),
        TurnRDRecord(
            task_id="b",
            turn_embeds=[[0.3] * D],
            final_reward=0.0,
        ),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in recs:
            f.write(json.dumps(asdict(r)) + "\n")
        path = f.name
    try:
        ds = TurnRDReplayDataset(path, mode=1)
        assert len(ds) == 2
        assert ds[0].goal_text == "goal A"
        assert ds[1].goal_text is None
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 3. Producer: collector emits goal_text when enabled
# ---------------------------------------------------------------------------


from src.algorithms.grpo.collectors import (  # noqa: E402
    RolloutCollector,
    RolloutCollectorConfig,
)


class _StubRunner:
    def generate_rich(self, prompts, sampling):
        class _Gen:
            text = "look 1"
            token_ids = (1, 2)
            token_logprobs = (-0.1, -0.2)
            prompt_token_count = 16
            prompt_token_ids = tuple(range(16))
            finish_reason = "stop"
        return [[_Gen] for _ in prompts]


class _StubEnv:
    def __init__(self, goal_text: str | None) -> None:
        self.goal_text = goal_text

    def reset(self, task_id=None):
        if self.goal_text is None:
            obs = "you are in a room with no canonical goal line"
        else:
            obs = (
                "-= Welcome to TextWorld, ALFRED! =-\n"
                "\n"
                f"Your task is to: {self.goal_text}\n"
            )

        class _S:
            observation_text = obs
        return _S

    def step(self, action_text: str):
        class _S:
            observation_text = "done"
        return _S, 1.0, True, {"intermediate_reward": 0.5}


def _stub_renderer(state, turns):
    return state.observation_text + "\n" + "\n".join(t.action_text for t in turns)


def _stub_parser(text: str) -> str:
    return text.strip().split("\n")[0]


def test_collector_emits_goal_text_when_flag_set() -> None:
    """Producer populates goal_text when turnrd_emit_goal_text=True."""
    D = 4
    goal_text = "find the apple."

    def turn_emb(traj):
        return torch.zeros(len(traj.turns), D, dtype=torch.float32)

    out_dir = tempfile.mkdtemp()
    replay_path = os.path.join(out_dir, "replay.jsonl")

    collector = RolloutCollector(
        runner=_StubRunner(),
        env_factory=lambda: _StubEnv(goal_text),
        prompt_renderer=_stub_renderer,
        action_parser=_stub_parser,
        cfg=RolloutCollectorConfig(max_turns=3),
        turnrd_emit_path=replay_path,
        turnrd_embedder=turn_emb,
        turnrd_emit_goal_text=True,
        round_idx=42,
    )

    class _SamplingStub:
        n = 1
        temperature = 1.0
    collector.collect_group(
        task_id="t0", env_name="alfworld", K=2, sampling=_SamplingStub()
    )

    with open(replay_path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 2
    for row in rows:
        assert row["goal_text"] == goal_text
        # goal_embed field should NOT be present after cleanup.
        assert "goal_embed" not in row
        assert row["round_idx"] == 42

    os.unlink(replay_path)
    os.rmdir(out_dir)


def test_collector_skips_goal_text_when_no_match() -> None:
    """Goal extractor returning None → goal_text stays None."""
    D = 4

    def turn_emb(traj):
        return torch.zeros(len(traj.turns), D, dtype=torch.float32)

    out_dir = tempfile.mkdtemp()
    replay_path = os.path.join(out_dir, "replay.jsonl")

    collector = RolloutCollector(
        runner=_StubRunner(),
        env_factory=lambda: _StubEnv(None),  # obs has no goal line
        prompt_renderer=_stub_renderer,
        action_parser=_stub_parser,
        cfg=RolloutCollectorConfig(max_turns=3),
        turnrd_emit_path=replay_path,
        turnrd_embedder=turn_emb,
        turnrd_emit_goal_text=True,
    )

    class _SamplingStub:
        n = 1
        temperature = 1.0
    collector.collect_group(
        task_id="t0", env_name="alfworld", K=1, sampling=_SamplingStub()
    )

    with open(replay_path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 1
    assert rows[0]["goal_text"] is None

    os.unlink(replay_path)
    os.rmdir(out_dir)


def test_collector_backcompat_when_flag_off() -> None:
    """Flag default False → goal_text stays None even with alfworld obs."""
    D = 4

    def turn_emb(traj):
        return torch.zeros(len(traj.turns), D, dtype=torch.float32)

    out_dir = tempfile.mkdtemp()
    replay_path = os.path.join(out_dir, "replay.jsonl")

    collector = RolloutCollector(
        runner=_StubRunner(),
        env_factory=lambda: _StubEnv("find apple"),
        prompt_renderer=_stub_renderer,
        action_parser=_stub_parser,
        cfg=RolloutCollectorConfig(max_turns=3),
        turnrd_emit_path=replay_path,
        turnrd_embedder=turn_emb,
        # turnrd_emit_goal_text NOT set ⇒ default False ⇒ no goal text emitted.
    )

    class _SamplingStub:
        n = 1
        temperature = 1.0
    collector.collect_group(
        task_id="t0", env_name="alfworld", K=1, sampling=_SamplingStub()
    )

    with open(replay_path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 1
    assert rows[0]["goal_text"] is None

    os.unlink(replay_path)
    os.rmdir(out_dir)
