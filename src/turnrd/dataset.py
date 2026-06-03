"""TurnRD replay-buffer reader (Method B).

Schema (one JSON object per line):
```
{
  "task_id": str,
  "turn_embeds": float[][T_i][D],   # pre-computed mean-pooled hidden states
  "final_reward": float,
  "judge_labels": float[T_i] | null # judge scores per turn (Mode 2)
  "progress":     float[T_i] | null # per-turn env signal; v2 V-head target
                                    # (alias `raw_env_rewards`)
  "progress_signal": float[T_i] | null # dense shaping signal; preferred over
                                    # `progress` for the V-head target
}
```

Embeddings are pre-computed producer-side so the trainer never re-tokenizes.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# Record dataclass


@dataclass(frozen=True)
class TurnRDRecord:
    """One row of the replay JSONL. `turn_embeds` is `list[list[float]]`;
    tensors are built in `pad_collate`.
    """

    task_id: str
    turn_embeds: list[list[float]]  # [T_i][D]
    final_reward: float
    judge_labels: Optional[list[float]] = None  # [T_i] or None
    progress: Optional[list[float]] = None  # [T_i] or None - per-turn env signal
    # Dense per-turn shaping signal (e.g. ALFWorld expert-plan reduction);
    # preferred over `progress` for the V-head target. None on non-ALFWorld.
    progress_signal: Optional[list[float]] = None  # [T_i] or None
    # Round index (0-based), read by the trainer's recency-decay path. None on
    # legacy rows.
    round_idx: Optional[int] = None
    # Per-trajectory AlfWorld goal text (Turn 0 obs); consumed with `goal_emb`
    # by the FiLM V-head. None for non-AlfWorld envs.
    goal_text: Optional[str] = None
    # Per-trajectory `goal_text` embedding (`[input_dim]`) for the FiLM V-head.
    # None on legacy rows; `pad_collate` emits a `goal_emb_mask` for them.
    goal_emb: Optional[list[float]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("TurnRDRecord.task_id must be a non-empty string.")
        if not isinstance(self.turn_embeds, list) or len(self.turn_embeds) == 0:
            raise ValueError(
                "TurnRDRecord.turn_embeds must be a non-empty list[list[float]]."
            )
        T = len(self.turn_embeds)
        D = len(self.turn_embeds[0])
        if D == 0:
            raise ValueError("TurnRDRecord.turn_embeds[0] must be non-empty.")
        for t, row in enumerate(self.turn_embeds):
            if not isinstance(row, list):
                raise ValueError(
                    f"TurnRDRecord.turn_embeds[{t}] must be list[float]; "
                    f"got {type(row).__name__}."
                )
            if len(row) != D:
                raise ValueError(
                    f"TurnRDRecord.turn_embeds[{t}] has D={len(row)}; "
                    f"expected D={D} (mismatched embedding width)."
                )
        if not isinstance(self.final_reward, (int, float)):
            raise ValueError(
                f"TurnRDRecord.final_reward must be a number; got {type(self.final_reward).__name__}."
            )
        if self.judge_labels is not None:
            if not isinstance(self.judge_labels, list):
                raise ValueError(
                    f"TurnRDRecord.judge_labels must be list[float] or None; "
                    f"got {type(self.judge_labels).__name__}."
                )
            if len(self.judge_labels) != T:
                raise ValueError(
                    f"TurnRDRecord.judge_labels has length {len(self.judge_labels)}; "
                    f"expected T={T} (must match turn_embeds)."
                )
        if self.progress is not None:
            if not isinstance(self.progress, list):
                raise ValueError(
                    f"TurnRDRecord.progress must be list[float] or None; "
                    f"got {type(self.progress).__name__}."
                )
            if len(self.progress) != T:
                raise ValueError(
                    f"TurnRDRecord.progress has length {len(self.progress)}; "
                    f"expected T={T} (must match turn_embeds)."
                )
        if self.progress_signal is not None:
            if not isinstance(self.progress_signal, list):
                raise ValueError(
                    f"TurnRDRecord.progress_signal must be list[float] or None; "
                    f"got {type(self.progress_signal).__name__}."
                )
            if len(self.progress_signal) != T:
                raise ValueError(
                    f"TurnRDRecord.progress_signal has length {len(self.progress_signal)}; "
                    f"expected T={T} (must match turn_embeds)."
                )
        if self.goal_text is not None and not isinstance(self.goal_text, str):
            raise ValueError(
                f"TurnRDRecord.goal_text must be str or None; "
                f"got {type(self.goal_text).__name__}."
            )
        if self.goal_emb is not None:
            if not isinstance(self.goal_emb, list):
                raise ValueError(
                    f"TurnRDRecord.goal_emb must be list[float] or None; "
                    f"got {type(self.goal_emb).__name__}."
                )
            if len(self.goal_emb) != D:
                raise ValueError(
                    f"TurnRDRecord.goal_emb has width {len(self.goal_emb)}; "
                    f"expected input_dim={D} (must match turn_embeds[0] width)."
                )


# Dataset


class TurnRDReplayDataset:
    """JSONL-backed replay buffer, parsed on construction. `mode` 2 drops rows
    lacking `judge_labels`. Does not filter by `round_idx` (the trainer applies
    recency decay at loss time).
    """

    def __init__(
        self,
        jsonl_path: str | os.PathLike[str],
        mode: int,
        *,
        max_records: int | None = None,
    ) -> None:
        if mode not in (1, 2):
            raise ValueError(f"TurnRDReplayDataset: mode must be 1 or 2; got {mode}.")
        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"TurnRDReplayDataset: file not found: {path}")
        self.path = path
        self.mode = int(mode)
        self.skipped_empty = 0
        self.skipped_missing_judge = 0

        records: list[TurnRDRecord] = []
        with open(path) as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"TurnRDReplayDataset: malformed JSON at {path}:{lineno}: {e}"
                    ) from e
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"TurnRDReplayDataset: line {lineno} is not a JSON object."
                    )
                # Filter empty trajectories early.
                turn_embeds = obj.get("turn_embeds", [])
                if not isinstance(turn_embeds, list) or len(turn_embeds) == 0:
                    self.skipped_empty += 1
                    continue
                try:
                    # Backward-compat alias: `raw_env_rewards` -> `progress`.
                    raw_progress = obj.get("progress", obj.get("raw_env_rewards", None))
                    raw_progress_signal = obj.get("progress_signal", None)
                    raw_round_idx = obj.get("round_idx", None)
                    raw_goal_text = obj.get("goal_text", None)
                    raw_goal_emb = obj.get("goal_emb", None)
                    rec = TurnRDRecord(
                        task_id=str(obj["task_id"]),
                        turn_embeds=turn_embeds,
                        final_reward=float(obj["final_reward"]),
                        judge_labels=obj.get("judge_labels", None),
                        progress=(
                            [float(x) for x in raw_progress]
                            if raw_progress is not None
                            else None
                        ),
                        progress_signal=(
                            [float(x) for x in raw_progress_signal]
                            if raw_progress_signal is not None
                            else None
                        ),
                        round_idx=(
                            int(raw_round_idx) if raw_round_idx is not None else None
                        ),
                        goal_text=(
                            str(raw_goal_text) if raw_goal_text is not None else None
                        ),
                        goal_emb=(
                            [float(x) for x in raw_goal_emb]
                            if raw_goal_emb is not None
                            else None
                        ),
                    )
                except (KeyError, TypeError, ValueError) as e:
                    raise ValueError(
                        f"TurnRDReplayDataset: malformed record at {path}:{lineno}: {e}"
                    ) from e
                if self.mode == 2 and rec.judge_labels is None:
                    self.skipped_missing_judge += 1
                    continue
                records.append(rec)
                if max_records is not None and len(records) >= int(max_records):
                    break
        if self.skipped_missing_judge > 0:
            logger.warning(
                "TurnRDReplayDataset(mode=2): dropped %d row(s) lacking judge_labels (path=%s).",
                self.skipped_missing_judge,
                path,
            )
        if self.skipped_empty > 0:
            logger.warning(
                "TurnRDReplayDataset: dropped %d empty-trajectory row(s) (path=%s).",
                self.skipped_empty,
                path,
            )

        self._records: list[TurnRDRecord] = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> TurnRDRecord:
        return self._records[index]

    def __iter__(self):
        return iter(self._records)


# Padded collate


def pad_collate(batch: list[TurnRDRecord]) -> dict[str, torch.Tensor]:
    """Pad `TurnRDRecord`s to fixed-shape tensors.

    Returns: `turn_embeds [B,T,D]`, `attention_mask [B,T]`, `final_reward [B]`,
    plus `judge_labels`/`progress`/`progress_signal [B,T]` (each present only
    if every record has it) and `goal_emb [B,D]` + `goal_emb_mask [B]` (if any
    record carries a goal embedding).
    """
    if not batch:
        raise ValueError("pad_collate: batch must be non-empty.")
    B = len(batch)
    D = len(batch[0].turn_embeds[0])
    for i, rec in enumerate(batch):
        if len(rec.turn_embeds[0]) != D:
            raise ValueError(
                f"pad_collate: record {i} has D={len(rec.turn_embeds[0])}; "
                f"batch[0] had D={D}. Embedding widths must match across the batch."
            )
    T_max = max(len(rec.turn_embeds) for rec in batch)

    turn_embeds = torch.zeros(B, T_max, D, dtype=torch.float32)
    attention_mask = torch.zeros(B, T_max, dtype=torch.long)
    final_reward = torch.zeros(B, dtype=torch.float32)
    have_all_judge = all(rec.judge_labels is not None for rec in batch)
    judge_labels: torch.Tensor | None = (
        torch.zeros(B, T_max, dtype=torch.float32) if have_all_judge else None
    )
    have_all_progress = all(rec.progress is not None for rec in batch)
    progress: torch.Tensor | None = (
        torch.zeros(B, T_max, dtype=torch.float32) if have_all_progress else None
    )
    have_all_progress_signal = all(rec.progress_signal is not None for rec in batch)
    progress_signal: torch.Tensor | None = (
        torch.zeros(B, T_max, dtype=torch.float32) if have_all_progress_signal else None
    )
    # Goal embedding [B, D] (zeros on absent rows) + `goal_emb_mask [B]` so the
    # model skips FiLM on missing-goal rows.
    any_goal_emb = any(rec.goal_emb is not None for rec in batch)
    goal_emb: torch.Tensor | None = (
        torch.zeros(B, D, dtype=torch.float32) if any_goal_emb else None
    )
    goal_emb_mask: torch.Tensor | None = (
        torch.zeros(B, dtype=torch.float32) if any_goal_emb else None
    )
    # Round index per record (always emitted; -1 for legacy rows).
    round_idx_t = torch.full((B,), -1, dtype=torch.long)

    for i, rec in enumerate(batch):
        T_i = len(rec.turn_embeds)
        turn_embeds[i, :T_i] = torch.tensor(rec.turn_embeds, dtype=torch.float32)
        attention_mask[i, :T_i] = 1
        final_reward[i] = float(rec.final_reward)
        if judge_labels is not None:
            assert rec.judge_labels is not None
            judge_labels[i, :T_i] = torch.tensor(rec.judge_labels, dtype=torch.float32)
        if progress is not None:
            assert rec.progress is not None
            progress[i, :T_i] = torch.tensor(rec.progress, dtype=torch.float32)
        if progress_signal is not None:
            assert rec.progress_signal is not None
            progress_signal[i, :T_i] = torch.tensor(
                rec.progress_signal, dtype=torch.float32
            )
        if goal_emb is not None:
            if rec.goal_emb is not None:
                goal_emb[i] = torch.tensor(rec.goal_emb, dtype=torch.float32)
                assert goal_emb_mask is not None
                goal_emb_mask[i] = 1.0
        if rec.round_idx is not None:
            round_idx_t[i] = int(rec.round_idx)

    out: dict[str, torch.Tensor] = {
        "turn_embeds": turn_embeds,
        "attention_mask": attention_mask,
        "final_reward": final_reward,
        "round_idx": round_idx_t,
    }
    if judge_labels is not None:
        out["judge_labels"] = judge_labels
    if progress is not None:
        out["progress"] = progress
    if progress_signal is not None:
        out["progress_signal"] = progress_signal
    if goal_emb is not None:
        out["goal_emb"] = goal_emb
        assert goal_emb_mask is not None
        out["goal_emb_mask"] = goal_emb_mask
    return out
