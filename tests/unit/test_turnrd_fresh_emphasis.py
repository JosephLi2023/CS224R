"""Unit test for the fresh-emphasis pass in train_turnrd.

Plan: `turnrd_v2_continual_larger` Step 2.

Covers:
1. With `fresh_emphasis_window_rounds=1, fresh_emphasis_n_epochs=2`, the
   trainer's main pass runs on ALL rounds, then the fresh-emphasis pass
   runs ONLY on the latest round's records (max_round_idx).
2. The returned summary's `fresh_emphasis` block reports n_rows == count
   of records with `round_idx == max_round_idx` and `n_steps` ≈ 2 epochs
   × ceil(n_rows / batch_size).
3. Default kwargs (window=0, n_epochs=0) preserve byte-for-byte legacy
   behaviour: no fresh-emphasis pass, summary['fresh_emphasis'] is None.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.turnrd.model import TurnRDv2, TurnRDv2Config
from src.turnrd.train import train_turnrd


def _write_synthetic_replay(
    path: Path,
    *,
    n_records_per_round: int = 4,
    rounds: list[int] = (0, 1, 2),
    D: int = 8,
    T: int = 5,
) -> int:
    """Write a 3-round synthetic JSONL. Returns total number of records."""
    import random
    rng = random.Random(0)
    n_total = 0
    with path.open("w") as fh:
        for r in rounds:
            for i in range(n_records_per_round):
                turn_embeds = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(T)]
                progress_signal = [0.0] * (T - 1) + [1.0 if i % 2 == 0 else 0.0]
                rec = {
                    "task_id": int(r * 1000 + i),
                    "turn_embeds": turn_embeds,
                    "final_reward": progress_signal[-1],
                    "progress": progress_signal,
                    "progress_signal": progress_signal,
                    "judge_labels": None,
                    "round_idx": int(r),
                }
                fh.write(json.dumps(rec) + "\n")
                n_total += 1
    return n_total


def _mk_model(D: int = 8) -> TurnRDv2:
    cfg = TurnRDv2Config(
        n_layers=1, hidden_size=16, n_heads=2,
        max_turns=8, dropout=0.0,
        causal=False, progress_prior_strength=1.0,
        goal_conditioned_value_head=False,
    )
    torch.manual_seed(0)
    return TurnRDv2(cfg, input_dim=D)


def test_fresh_emphasis_trains_only_on_latest_window() -> None:
    """Window=1 should restrict the fresh-emphasis pass to round_idx==2 records."""
    n_per_round = 4
    rounds = [0, 1, 2]
    with tempfile.TemporaryDirectory() as td:
        replay_path = Path(td) / "replay.jsonl"
        n_total = _write_synthetic_replay(
            replay_path, n_records_per_round=n_per_round, rounds=rounds, D=8, T=5
        )
        assert n_total == len(rounds) * n_per_round

        model = _mk_model(D=8)
        summary = train_turnrd(
            str(replay_path),
            mode=1,
            model=model,
            n_epochs=1,
            batch_size=2,
            lr=1e-4,
            version="v2",
            lambda_value=1.0,
            lambda_rank=0.0,
            lambda_progress=0.0,
            fresh_emphasis_window_rounds=1,
            fresh_emphasis_n_epochs=2,
        )

    # Main pass should have trained on all records: 12 records / batch 2 = 6 batches
    expected_main_steps = math.ceil(n_total / 2) * 1  # 1 epoch
    assert summary["n_steps"] == expected_main_steps, (
        f"main n_steps expected {expected_main_steps}, got {summary['n_steps']}"
    )

    # Fresh emphasis should be populated
    fe = summary["fresh_emphasis"]
    assert fe is not None, "fresh_emphasis block missing from summary"
    assert fe["window_rounds"] == 1
    assert fe["n_epochs"] == 2
    # window=1 means only round_idx >= max_round_idx (=2) - 1 + 1 = 2; so only round 2 rows
    assert fe["min_round_idx"] == 2
    assert fe["n_rows"] == n_per_round, (
        f"fresh n_rows expected {n_per_round} (round_idx=2 only), got {fe['n_rows']}"
    )
    # Steps: 2 epochs × ceil(4/2) = 4
    expected_fresh_steps = math.ceil(n_per_round / 2) * 2
    assert fe["n_steps"] == expected_fresh_steps, (
        f"fresh n_steps expected {expected_fresh_steps}, got {fe['n_steps']}"
    )
    assert summary["n_steps_fresh"] == expected_fresh_steps


def test_fresh_emphasis_disabled_by_default_preserves_legacy_behavior() -> None:
    """Default (window=0, n_epochs=0): no fresh pass, summary['fresh_emphasis'] is None."""
    n_per_round = 4
    rounds = [0, 1, 2]
    with tempfile.TemporaryDirectory() as td:
        replay_path = Path(td) / "replay.jsonl"
        _write_synthetic_replay(
            replay_path, n_records_per_round=n_per_round, rounds=rounds, D=8, T=5
        )
        model = _mk_model(D=8)
        summary = train_turnrd(
            str(replay_path),
            mode=1, model=model,
            n_epochs=1, batch_size=2, lr=1e-4,
            version="v2", lambda_value=1.0, lambda_rank=0.0, lambda_progress=0.0,
            # NO fresh_emphasis_* kwargs → both default to 0 → disabled.
        )
    assert summary["fresh_emphasis"] is None
    assert summary["n_steps_fresh"] == 0
