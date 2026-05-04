"""Unit tests for `scripts/merge_turnrd_round_logs.py`.

Verification matrix:
1. `test_merge_rounds_concatenates_global_episodes`
2. `test_merge_rounds_preserves_local_episode_and_round_idx`
3. `test_merge_rounds_orders_by_round_index_not_filesystem`
4. `test_merge_rounds_rejects_duplicate_rounds`
5. `test_merge_rounds_raises_on_empty_input`
6. `test_find_round_dirs_filters_by_prefix`
7. `test_main_writes_output_file_and_stdout`
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest


# Load the script as a module (it lives in scripts/ which isn't a package).
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "merge_turnrd_round_logs.py"


def _load_merger():
    spec = importlib.util.spec_from_file_location("merge_turnrd_round_logs", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so @dataclass(frozen=True) can resolve
    # cls.__module__ via sys.modules during class creation.
    sys.modules["merge_turnrd_round_logs"] = mod
    spec.loader.exec_module(mod)
    return mod


merger = _load_merger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_round_dir(
    manifests_dir: Path,
    *,
    prefix: str,
    round_idx: int,
    timestamp: str,
    rows: list[dict],
    cfg_extras: dict | None = None,
) -> Path:
    """Create a synthetic round dir matching the orchestrator's layout."""
    name = f"{prefix}_round{round_idx:02d}_{timestamp}"
    rd = manifests_dir / name
    rd.mkdir(parents=True, exist_ok=True)
    cfg = {
        "n_episodes": len(rows),
        "K": 4,
        "max_turns": 6,
        "task_id_offset": round_idx * len(rows),
        "num_products": 1000,
        "sync_every": 1,
        "run_name": name,
        "sft_adapter": "",
    }
    if cfg_extras:
        cfg.update(cfg_extras)
    (rd / "train_log.json").write_text(json.dumps({"rows": rows, "config": cfg}))
    return rd


def _row(episode: int, mean_reward: float, **extras) -> dict:
    """Synthetic train_log row mirroring infra/app_train_loop.py output."""
    base = {
        "episode": episode,
        "task_id": episode,
        "mean_reward": mean_reward,
        "std_reward": 0.0,
        "completed": 4,
        "truncated": 0,
        "n_turns": 24,
        "n_action_tokens": 200,
        "policy_loss": 0.01,
        "kl_term": 0.0,
        "consistency": 0.0,
        "total_loss": 0.01,
        "observed_kl": 0.0,
        "kl_coef": 0.04,
        "grad_norm": 0.5,
        "mean_traj_adv": 0.0,
        "elapsed_s": 1.0,
    }
    base.update(extras)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_merge_rounds_concatenates_global_episodes(tmp_path: Path) -> None:
    """Two rounds × 3 episodes each → 6 merged rows with `episode` running 0..5."""
    prefix = "test_run_seed11"
    _write_round_dir(
        tmp_path, prefix=prefix, round_idx=0, timestamp="20260101_000000",
        rows=[_row(0, 0.1), _row(1, 0.2), _row(2, 0.3)],
    )
    _write_round_dir(
        tmp_path, prefix=prefix, round_idx=1, timestamp="20260101_000100",
        rows=[_row(0, 0.4), _row(1, 0.5), _row(2, 0.6)],
    )

    round_dirs = merger._find_round_dirs(tmp_path, prefix)
    merged = merger.merge_rounds(round_dirs)

    assert len(merged["rows"]) == 6
    assert [r["episode"] for r in merged["rows"]] == [0, 1, 2, 3, 4, 5]
    assert [r["mean_reward"] for r in merged["rows"]] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert merged["config"]["merged_rounds"] == 2
    assert merged["config"]["total_episodes"] == 6
    # Per-round-only knobs were dropped from the merged config.
    assert "task_id_offset" not in merged["config"]
    assert "run_name" not in merged["config"]


def test_merge_rounds_preserves_local_episode_and_round_idx(tmp_path: Path) -> None:
    """`local_episode` keeps the original within-round number; `round_idx`
    stamps which round the row came from."""
    prefix = "test_seed11"
    _write_round_dir(
        tmp_path, prefix=prefix, round_idx=0, timestamp="20260101_000000",
        rows=[_row(0, 0.1), _row(1, 0.2)],
    )
    _write_round_dir(
        tmp_path, prefix=prefix, round_idx=1, timestamp="20260101_000100",
        rows=[_row(0, 0.3), _row(1, 0.4)],
    )

    round_dirs = merger._find_round_dirs(tmp_path, prefix)
    merged = merger.merge_rounds(round_dirs)

    # Round 0: local_episode 0, 1 (round_idx=0).
    # Round 1: local_episode 0, 1 (round_idx=1) — note local resets, but global doesn't.
    assert [r["local_episode"] for r in merged["rows"]] == [0, 1, 0, 1]
    assert [r["round_idx"] for r in merged["rows"]] == [0, 0, 1, 1]
    assert [r["episode"] for r in merged["rows"]] == [0, 1, 2, 3]


def test_merge_rounds_orders_by_round_index_not_filesystem(tmp_path: Path) -> None:
    """Even if filesystem returns dirs out of order (e.g. round 2 listed
    before round 0 due to alphabetical ts), the merger sorts by
    parsed round_idx."""
    prefix = "test_seed11"
    _write_round_dir(
        tmp_path, prefix=prefix, round_idx=2, timestamp="20260101_000000",
        rows=[_row(0, 0.5)],  # written first; ts=earlier
    )
    _write_round_dir(
        tmp_path, prefix=prefix, round_idx=0, timestamp="20260101_999999",
        rows=[_row(0, 0.1)],  # written second; ts=later
    )
    _write_round_dir(
        tmp_path, prefix=prefix, round_idx=1, timestamp="20260101_500000",
        rows=[_row(0, 0.2)],
    )

    round_dirs = merger._find_round_dirs(tmp_path, prefix)
    merged = merger.merge_rounds(round_dirs)

    # Order should be round 0, 1, 2 regardless of timestamp string.
    assert [r["mean_reward"] for r in merged["rows"]] == [0.1, 0.2, 0.5]
    assert [r["round_idx"] for r in merged["rows"]] == [0, 1, 2]


def test_merge_rounds_rejects_duplicate_rounds(tmp_path: Path) -> None:
    """Two dirs with the same round_idx → ValueError (would corrupt
    global episode numbering)."""
    prefix = "test_seed11"
    _write_round_dir(
        tmp_path, prefix=prefix, round_idx=0, timestamp="20260101_000000",
        rows=[_row(0, 0.1)],
    )
    _write_round_dir(
        tmp_path, prefix=prefix, round_idx=0, timestamp="20260101_111111",
        rows=[_row(0, 0.2)],
    )
    round_dirs = merger._find_round_dirs(tmp_path, prefix)
    with pytest.raises(ValueError, match=r"duplicate round_idx"):
        merger.merge_rounds(round_dirs)


def test_merge_rounds_raises_on_empty_input() -> None:
    """Empty round list → ValueError with a clear message."""
    with pytest.raises(ValueError, match=r"empty round list"):
        merger.merge_rounds([])


def test_find_round_dirs_filters_by_prefix(tmp_path: Path) -> None:
    """Only dirs starting with `<prefix>_round` are returned. Unrelated
    runs (Methods A/C) under the same manifests dir are ignored."""
    _write_round_dir(
        tmp_path, prefix="test_seed11", round_idx=0, timestamp="20260101_000000",
        rows=[_row(0, 0.1)],
    )
    # Unrelated runs that should NOT match.
    _write_round_dir(
        tmp_path, prefix="other_seed11", round_idx=0, timestamp="20260101_000000",
        rows=[_row(0, 0.9)],
    )
    (tmp_path / "method_hgpo_progress_seed11_20260101_000000").mkdir()  # no _round in name
    (tmp_path / "test_seed11_round_no_index_20260101").mkdir()  # malformed round suffix

    round_dirs = merger._find_round_dirs(tmp_path, "test_seed11")
    assert len(round_dirs) == 1
    assert round_dirs[0].round_idx == 0


def test_main_writes_output_file_and_stdout(tmp_path: Path) -> None:
    """End-to-end: --out to a file, then --out '-' to stdout, both produce
    identical merged JSON."""
    prefix = "test_seed11"
    _write_round_dir(
        tmp_path, prefix=prefix, round_idx=0, timestamp="20260101_000000",
        rows=[_row(0, 0.1), _row(1, 0.2)],
    )

    out_file = tmp_path / "merged.json"
    rc = merger.main([
        "--manifests-dir", str(tmp_path),
        "--prefix", prefix,
        "--out", str(out_file),
    ])
    assert rc == 0
    assert out_file.is_file()
    file_payload = json.loads(out_file.read_text())
    assert len(file_payload["rows"]) == 2

    # Now --out '-' (stdout).
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = merger.main([
            "--manifests-dir", str(tmp_path),
            "--prefix", prefix,
            "--out", "-",
        ])
    assert rc == 0
    stdout_payload = json.loads(buf.getvalue())
    assert stdout_payload == file_payload
