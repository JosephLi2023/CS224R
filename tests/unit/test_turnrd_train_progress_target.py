"""Unit tests for the v2 per-turn progress-target wiring (Tier-3 Phase A).

Verifies the four touchpoints that turn the v2 value-head's `R/T_i`
placeholder target into the env-side per-turn progress signal:

1. `pad_collate` exposes `progress` as a `[B, T_max]` tensor when ALL
   records carry the field.
2. `pad_collate` OMITS `progress` whenever any record has `progress=None`
   (so the train loop's R/T_i fallback fires safely).
3. `train_turnrd` (v2 path) computes `loss_v2_value` against the new
   progress target whenever `lambda_value > 0` AND every record in the
   batch carries `progress` — the loss is non-zero and matches the
   reference MSE we compute analytically from `model(...).predicted_per_turn_R`.
4. `TurnRDRecord(progress=...)` rejects a length mismatch at construction.

Skipped cleanly on hosts without torch.
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
from src.turnrd.model import TurnRDv2, TurnRDv2Config, loss_v2_value  # noqa: E402
from src.turnrd.train import train_turnrd  # noqa: E402


INPUT_DIM = 8


def _make_v2(seed: int = 0) -> TurnRDv2:
    torch.manual_seed(seed)
    cfg = TurnRDv2Config(n_layers=2, hidden_size=16, n_heads=2, max_turns=8, dropout=0.0)
    return TurnRDv2(cfg, input_dim=INPUT_DIM)


def _write_replay(path: Path, *, with_progress: bool, n: int = 4, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        T = 2 + (i % 3)  # T_i ∈ {2, 3, 4}
        embeds = torch.randn(T, INPUT_DIM, generator=g).tolist()
        # Simulate AlfWorld's binary success: per-turn rewards are 0 except
        # the final turn carries the +1 success signal for half the rows.
        success = (i % 2) == 0
        per_turn = [0.0] * T
        if success:
            per_turn[-1] = 1.0
        R = float(sum(per_turn))
        row: dict = {
            "task_id": f"task-{i}",
            "turn_embeds": embeds,
            "final_reward": R,
        }
        if with_progress:
            row["progress"] = per_turn
        rows.append(row)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Test 1: pad_collate exposes `progress` when all records carry it.
# ---------------------------------------------------------------------------


def test_pad_collate_emits_progress_when_all_records_have_it() -> None:
    rec_a = TurnRDRecord(
        task_id="a",
        turn_embeds=[[1.0, 2.0]],
        final_reward=0.0,
        progress=[0.0],
    )
    rec_b = TurnRDRecord(
        task_id="b",
        turn_embeds=[[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        final_reward=1.0,
        progress=[0.0, 0.0, 1.0],
    )
    out = pad_collate([rec_a, rec_b])
    assert "progress" in out
    assert out["progress"].shape == (2, 3)
    assert out["progress"][0].tolist() == [0.0, 0.0, 0.0]
    assert out["progress"][1].tolist() == [0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Test 2: pad_collate omits `progress` when any record has progress=None.
# ---------------------------------------------------------------------------


def test_pad_collate_omits_progress_when_any_record_lacks_it() -> None:
    rec_a = TurnRDRecord(
        task_id="a",
        turn_embeds=[[1.0, 2.0]],
        final_reward=0.0,
        progress=[0.0],
    )
    rec_b = TurnRDRecord(
        task_id="b",
        turn_embeds=[[3.0, 4.0]],
        final_reward=1.0,
        progress=None,  # legacy
    )
    out = pad_collate([rec_a, rec_b])
    assert "progress" not in out


# ---------------------------------------------------------------------------
# Test 3: v2 trainer uses progress as the value-head target when present.
# ---------------------------------------------------------------------------


def test_train_v2_uses_progress_as_value_target(tmp_path: Path) -> None:
    """When `progress` is present in the batch, `loss_v2_value` is computed
    against it (not against R/T_i). We verify by recomputing the reference
    MSE analytically from the model's per-turn predictions."""
    replay = tmp_path / "replay.jsonl"
    _write_replay(replay, with_progress=True, n=4, seed=7)
    model = _make_v2(seed=7)

    # Snapshot the per-turn predictions BEFORE optimizer.step() so the
    # reference MSE matches what the trainer's first-batch loss saw.
    ds = TurnRDReplayDataset(replay, mode=1)
    batch = pad_collate([ds[0], ds[1], ds[2], ds[3]])
    assert "progress" in batch  # producer wrote it; collator forwarded it

    model.eval()
    with torch.no_grad():
        out = model(batch["turn_embeds"], batch["attention_mask"])
        target_v = batch["progress"] * batch["attention_mask"].to(
            dtype=batch["progress"].dtype
        )
        ref_mse = loss_v2_value(out, target_v, batch["attention_mask"])
    assert ref_mse.item() > 0.0, "Synthetic progress fixture produced zero MSE"

    # Now train one micro-step at lambda_value=1.0 (no rank, no
    # progress-prior to keep the breakdown clean).
    model = _make_v2(seed=7)
    summary = train_turnrd(
        replay,
        mode=1,
        model=model,
        n_epochs=1,
        batch_size=4,
        lr=0.0,  # no parameter update — preserves the loss snapshot
        log_every=0,
        version="v2",
        lambda_value=1.0,
        lambda_rank=0.0,
        lambda_progress=0.0,
    )
    # The breakdown's `value_loss` slot should match our analytic ref MSE
    # to within float32 precision.
    assert summary["final_loss_breakdown"]["value_loss"] == pytest.approx(
        float(ref_mse.item()), rel=1e-4, abs=1e-5
    )


def test_train_v2_falls_back_to_R_over_T_when_progress_absent(tmp_path: Path) -> None:
    """Legacy replays (no `progress` field) keep working — R/T_i fallback
    fires and the value loss is still computed (non-zero)."""
    replay = tmp_path / "replay.jsonl"
    _write_replay(replay, with_progress=False, n=4, seed=8)
    model = _make_v2(seed=8)

    summary = train_turnrd(
        replay,
        mode=1,
        model=model,
        n_epochs=1,
        batch_size=4,
        lr=0.0,
        log_every=0,
        version="v2",
        lambda_value=1.0,
        lambda_rank=0.0,
        lambda_progress=0.0,
    )
    assert summary["n_steps"] > 0
    # Value loss MUST still be tracked even though we used the fallback
    # target — otherwise we silently lost the supervision signal.
    assert summary["final_loss_breakdown"]["value_loss"] > 0.0


# ---------------------------------------------------------------------------
# Test 4: TurnRDRecord rejects mismatched-length progress.
# ---------------------------------------------------------------------------


def test_record_post_init_rejects_progress_length_mismatch() -> None:
    with pytest.raises(ValueError, match=r"length"):
        TurnRDRecord(
            task_id="bad",
            turn_embeds=[[1.0, 2.0], [3.0, 4.0]],  # T=2
            final_reward=0.0,
            progress=[0.1],  # T=1 → mismatch
        )
