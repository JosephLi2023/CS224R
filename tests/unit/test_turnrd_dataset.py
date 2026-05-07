"""Unit tests for `src.turnrd.dataset` (replay-buffer reader).

Verification matrix:
1. `test_dataset_loads_all_records_round_trip_shapes`
2. `test_mode_2_filters_rows_lacking_judge_labels`
3. `test_pad_collate_shapes_and_mask_pattern`
4. `test_dataset_rejects_malformed_jsonl`
5. `test_dataset_max_records_clamps_length`

Skipped cleanly on hosts without torch (the dataset module imports torch
unconditionally for `pad_collate`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.turnrd.dataset import (  # noqa: E402  (after importorskip)
    TurnRDRecord,
    TurnRDReplayDataset,
    pad_collate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embedding(T: int, D: int, *, seed: int) -> list[list[float]]:
    """Deterministic per-test embedding."""
    g = torch.Generator().manual_seed(seed)
    arr = torch.randn(T, D, generator=g)
    return arr.tolist()


def _write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _make_fixture(tmp_path: Path, *, with_judge: bool, n: int = 4, D: int = 8) -> Path:
    rows = []
    for i in range(n):
        T = 2 + (i % 3)  # T_i ∈ {2, 3, 4}
        rec = {
            "task_id": f"task-{i}",
            "turn_embeds": _embedding(T, D, seed=100 + i),
            "final_reward": 0.1 * (i + 1),
        }
        if with_judge:
            rec["judge_labels"] = [0.05 * (j + 1) for j in range(T)]
        rows.append(rec)
    path = tmp_path / "replay.jsonl"
    _write_jsonl(path, rows)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_loads_all_records_round_trip_shapes(tmp_path: Path) -> None:
    """4 records, varying T_i, fixed D — Mode 1 loader returns all 4 with the
    correct per-record shapes."""
    path = _make_fixture(tmp_path, with_judge=False, n=4, D=8)
    ds = TurnRDReplayDataset(path, mode=1)
    assert len(ds) == 4
    for i, rec in enumerate(ds):
        assert isinstance(rec, TurnRDRecord)
        assert rec.task_id == f"task-{i}"
        assert len(rec.turn_embeds) == 2 + (i % 3)
        assert all(len(row) == 8 for row in rec.turn_embeds)
        assert rec.judge_labels is None
    assert ds.skipped_empty == 0
    assert ds.skipped_missing_judge == 0


def test_mode_2_filters_rows_lacking_judge_labels(tmp_path: Path) -> None:
    """Two rows with judge_labels, one without → mode=2 keeps 2, increments
    skipped_missing_judge counter."""
    rows = [
        {
            "task_id": "with-1",
            "turn_embeds": _embedding(3, 4, seed=1),
            "final_reward": 0.5,
            "judge_labels": [0.1, 0.2, 0.2],
        },
        {
            "task_id": "without",
            "turn_embeds": _embedding(2, 4, seed=2),
            "final_reward": 0.3,
            # judge_labels intentionally omitted (loaded as None)
        },
        {
            "task_id": "with-2",
            "turn_embeds": _embedding(4, 4, seed=3),
            "final_reward": 0.7,
            "judge_labels": [0.1, 0.1, 0.2, 0.3],
        },
    ]
    path = tmp_path / "mixed.jsonl"
    _write_jsonl(path, rows)

    ds = TurnRDReplayDataset(path, mode=2)
    assert len(ds) == 2
    assert ds.skipped_missing_judge == 1
    task_ids = [rec.task_id for rec in ds]
    assert task_ids == ["with-1", "with-2"]


def test_pad_collate_shapes_and_mask_pattern(tmp_path: Path) -> None:
    """pad_collate produces correctly-shaped tensors with the right mask pattern.

    Two records, T_i ∈ {2, 4}, D=3 → turn_embeds:[2, 4, 3], mask:[2, 4]
    with row 0 = [1, 1, 0, 0] and row 1 = [1, 1, 1, 1].
    """
    rec_a = TurnRDRecord(
        task_id="a",
        turn_embeds=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        final_reward=0.5,
        judge_labels=[0.1, 0.2],
    )
    rec_b = TurnRDRecord(
        task_id="b",
        turn_embeds=[
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
            [70.0, 80.0, 90.0],
            [100.0, 110.0, 120.0],
        ],
        final_reward=-0.3,
        judge_labels=[0.0, 0.1, 0.2, -0.1],
    )

    out = pad_collate([rec_a, rec_b])

    assert out["turn_embeds"].shape == (2, 4, 3)
    assert out["attention_mask"].shape == (2, 4)
    assert out["final_reward"].shape == (2,)
    assert out["judge_labels"].shape == (2, 4)

    # Mask pattern: row 0 = [1, 1, 0, 0], row 1 = [1, 1, 1, 1].
    assert out["attention_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    # Padded positions in turn_embeds are exactly 0.
    assert torch.equal(out["turn_embeds"][0, 2:], torch.zeros(2, 3))
    # Real values preserved.
    assert torch.equal(out["turn_embeds"][0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(out["turn_embeds"][1, 3], torch.tensor([100.0, 110.0, 120.0]))
    # final_reward + judge_labels carry their values.
    assert pytest.approx(out["final_reward"].tolist()) == [0.5, -0.3]
    assert pytest.approx(out["judge_labels"][0].tolist()) == [0.1, 0.2, 0.0, 0.0]
    assert pytest.approx(out["judge_labels"][1].tolist()) == [0.0, 0.1, 0.2, -0.1]


def test_pad_collate_omits_judge_labels_when_any_record_lacks_them() -> None:
    """If at least one record has judge_labels=None, pad_collate must NOT
    emit a judge_labels key (the trainer would otherwise feed zeros that
    look like real teacher labels)."""
    rec_a = TurnRDRecord(
        task_id="a",
        turn_embeds=[[1.0, 2.0]],
        final_reward=0.5,
        judge_labels=[0.1],
    )
    rec_b = TurnRDRecord(
        task_id="b",
        turn_embeds=[[3.0, 4.0]],
        final_reward=0.6,
        judge_labels=None,
    )
    out = pad_collate([rec_a, rec_b])
    assert "judge_labels" not in out


def test_dataset_rejects_malformed_jsonl(tmp_path: Path) -> None:
    """A non-JSON line raises ValueError (with the file path + line number)."""
    path = tmp_path / "bad.jsonl"
    with open(path, "w") as fh:
        fh.write(
            json.dumps(
                {"task_id": "ok", "turn_embeds": [[1.0]], "final_reward": 0.0}
            )
            + "\n"
        )
        fh.write("this is not json\n")
    with pytest.raises(ValueError, match=r"malformed JSON"):
        TurnRDReplayDataset(path, mode=1)


def test_dataset_max_records_clamps_length(tmp_path: Path) -> None:
    """`max_records=2` returns the first 2 records even if the file has more."""
    path = _make_fixture(tmp_path, with_judge=False, n=5, D=4)
    ds = TurnRDReplayDataset(path, mode=1, max_records=2)
    assert len(ds) == 2


def test_record_post_init_rejects_inconsistent_d() -> None:
    """A record with rows of different D raises ValueError at construction."""
    with pytest.raises(ValueError, match=r"D="):
        TurnRDRecord(
            task_id="bad",
            turn_embeds=[[1.0, 2.0], [3.0, 4.0, 5.0]],
            final_reward=0.0,
        )


def test_record_post_init_rejects_judge_label_length_mismatch() -> None:
    """judge_labels length must match T."""
    with pytest.raises(ValueError, match=r"length"):
        TurnRDRecord(
            task_id="bad",
            turn_embeds=[[1.0, 2.0], [3.0, 4.0]],
            final_reward=0.0,
            judge_labels=[0.1],
        )


# ---------------------------------------------------------------------------
# Method D — `progress` field tests
# ---------------------------------------------------------------------------


def test_pad_collate_pads_progress_to_T_max() -> None:
    """When all records have `progress`, pad_collate emits a [B, T_max]
    `progress` tensor padded with 0.0 at masked positions."""
    rec_a = TurnRDRecord(
        task_id="a",
        turn_embeds=[[1.0, 2.0]],
        final_reward=0.5,
        progress=[0.3],
    )
    rec_b = TurnRDRecord(
        task_id="b",
        turn_embeds=[[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        final_reward=0.7,
        progress=[0.1, 0.2, 0.4],
    )

    out = pad_collate([rec_a, rec_b])

    assert "progress" in out
    assert out["progress"].shape == (2, 3)
    assert pytest.approx(out["progress"][0].tolist()) == [0.3, 0.0, 0.0]
    assert pytest.approx(out["progress"][1].tolist()) == [0.1, 0.2, 0.4]


def test_pad_collate_omits_progress_when_any_record_lacks_it() -> None:
    """If at least one record has progress=None, pad_collate must NOT
    emit a `progress` key (zeros would look like real env signal)."""
    rec_a = TurnRDRecord(
        task_id="a",
        turn_embeds=[[1.0, 2.0]],
        final_reward=0.5,
        progress=[0.3],
    )
    rec_b = TurnRDRecord(
        task_id="b",
        turn_embeds=[[3.0, 4.0]],
        final_reward=0.6,
        progress=None,  # legacy row
    )
    out = pad_collate([rec_a, rec_b])
    assert "progress" not in out


def test_dataset_handles_records_without_progress_field(tmp_path: Path) -> None:
    """Backward compat: legacy JSONL rows without `progress` field load
    cleanly with `progress=None`."""
    rows = [
        {
            "task_id": "legacy-a",
            "turn_embeds": _embedding(2, 4, seed=10),
            "final_reward": 0.5,
            # NO progress, NO raw_env_rewards
        },
        {
            "task_id": "new-b",
            "turn_embeds": _embedding(3, 4, seed=11),
            "final_reward": 0.7,
            "progress": [0.1, 0.2, 0.4],
        },
        {
            "task_id": "raw-c",
            "turn_embeds": _embedding(2, 4, seed=12),
            "final_reward": 0.3,
            # producer wrote `raw_env_rewards` (alias accepted)
            "raw_env_rewards": [0.0, 0.3],
        },
    ]
    path = tmp_path / "mixed_progress.jsonl"
    _write_jsonl(path, rows)
    ds = TurnRDReplayDataset(path, mode=1)
    assert len(ds) == 3
    recs = list(ds)
    assert recs[0].progress is None
    assert recs[1].progress == [0.1, 0.2, 0.4]
    assert recs[2].progress == [0.0, 0.3]


def test_record_post_init_rejects_progress_length_mismatch() -> None:
    """progress length must match T."""
    with pytest.raises(ValueError, match=r"length"):
        TurnRDRecord(
            task_id="bad",
            turn_embeds=[[1.0, 2.0], [3.0, 4.0]],
            final_reward=0.0,
            progress=[0.1],
        )
