"""Unit tests for the `goal_emb` schema + per-row masking in
`src.turnrd.dataset`.

Plan: `turnrd_goal_conditioned_v_head` Step 10.

Covers:
1. `TurnRDRecord` accepts a goal_emb of length == input_dim and rejects
   a wrong-width goal_emb.
2. `pad_collate` with mixed `goal_emb` presence emits a
   `goal_emb` tensor + `goal_emb_mask` flag and does NOT drop non-goal
   records.
3. JSONL round-trip: writing a record with goal_emb and reading it
   back via `TurnRDReplayDataset` preserves the field.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.turnrd.dataset import TurnRDRecord, TurnRDReplayDataset, pad_collate


D = 4  # tiny input_dim for tests


def _mk_record(
    task_id: str = "t0",
    T: int = 3,
    with_goal_emb: bool = True,
) -> TurnRDRecord:
    return TurnRDRecord(
        task_id=task_id,
        turn_embeds=[[float(t + d * 0.1) for d in range(D)] for t in range(T)],
        final_reward=1.0,
        goal_emb=[0.1 * (d + 1) for d in range(D)] if with_goal_emb else None,
    )


def test_record_accepts_goal_emb_at_correct_width() -> None:
    rec = _mk_record(with_goal_emb=True)
    assert rec.goal_emb is not None
    assert len(rec.goal_emb) == D


def test_record_rejects_wrong_width_goal_emb() -> None:
    with pytest.raises(ValueError, match="goal_emb"):
        TurnRDRecord(
            task_id="bad",
            turn_embeds=[[0.0] * D] * 2,
            final_reward=0.0,
            goal_emb=[0.0] * (D + 1),  # wrong width
        )


def test_record_rejects_non_list_goal_emb() -> None:
    with pytest.raises(ValueError, match="goal_emb"):
        TurnRDRecord(
            task_id="bad2",
            turn_embeds=[[0.0] * D] * 2,
            final_reward=0.0,
            goal_emb="not a list",  # type: ignore[arg-type]
        )


def test_pad_collate_mixed_goal_emb_emits_mask() -> None:
    """Mix one record WITH goal_emb and one WITHOUT. The collated batch
    must emit a `goal_emb` [B, D] tensor (zero-filled on the absent
    row) plus a `goal_emb_mask` [B] flag — and crucially must NOT drop
    either record."""
    rec_with = _mk_record(task_id="a", with_goal_emb=True)
    rec_without = _mk_record(task_id="b", with_goal_emb=False)
    out = pad_collate([rec_with, rec_without])
    # Both records present in the batch (no drop).
    assert out["turn_embeds"].shape[0] == 2
    # goal_emb emitted with the right shape.
    assert "goal_emb" in out
    assert out["goal_emb"].shape == (2, D)
    # Mask flags which rows are real.
    assert "goal_emb_mask" in out
    assert out["goal_emb_mask"].tolist() == [1.0, 0.0]
    # Row without goal_emb is zero-filled.
    assert torch.allclose(out["goal_emb"][1], torch.zeros(D))


def test_pad_collate_all_records_lack_goal_emb_drops_field() -> None:
    rec1 = _mk_record(task_id="a", with_goal_emb=False)
    rec2 = _mk_record(task_id="b", with_goal_emb=False)
    out = pad_collate([rec1, rec2])
    assert "goal_emb" not in out
    assert "goal_emb_mask" not in out


def test_jsonl_round_trip_with_goal_emb() -> None:
    rec = _mk_record(task_id="rt", with_goal_emb=True)
    payload = {
        "task_id": rec.task_id,
        "turn_embeds": rec.turn_embeds,
        "final_reward": rec.final_reward,
        "goal_emb": rec.goal_emb,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps(payload) + "\n")
        path = Path(fh.name)
    try:
        ds = TurnRDReplayDataset(path, mode=1)
        assert len(ds) == 1
        loaded = ds[0]
        assert loaded.goal_emb is not None
        assert loaded.goal_emb == rec.goal_emb
    finally:
        path.unlink(missing_ok=True)


def test_jsonl_round_trip_legacy_record_without_goal_emb() -> None:
    """A legacy row (no goal_emb field at all) loads cleanly with
    goal_emb=None."""
    payload = {
        "task_id": "legacy",
        "turn_embeds": [[0.0] * D for _ in range(2)],
        "final_reward": 0.5,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps(payload) + "\n")
        path = Path(fh.name)
    try:
        ds = TurnRDReplayDataset(path, mode=1)
        assert len(ds) == 1
        loaded = ds[0]
        assert loaded.goal_emb is None
    finally:
        path.unlink(missing_ok=True)
