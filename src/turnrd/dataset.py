"""TurnRD replay-buffer reader (Method B).

Schema (one JSON object per line in the replay JSONL file):
```
{
  "task_id": str,
  "turn_embeds": float[][T_i][D],   # pre-computed mean-pooled hidden states
                                    # (one [D]-vector per turn, length T_i)
  "final_reward": float,
  "judge_labels": float[T_i] | null # cached normalized judge scores per turn
                                    # (Mode 2 only; null when not available)
  "progress":     float[T_i] | null # OPTIONAL per-turn environment progress
                                    # signal (e.g., AlfWorld per-step reward).
                                    # When present and ALL records in a batch
                                    # carry it, `pad_collate` exposes it as
                                    # `progress` and the v2 trainer uses it
                                    # as the per-turn value-head target
                                    # (replaces the placeholder R/T_i target).
                                    # JSONL alias `raw_env_rewards` is also
                                    # accepted for legacy producers.
  "progress_signal": float[T_i] | null # OPTIONAL dense per-turn shaping
                                    # signal sourced from the env adapter
                                    # (e.g., ALFWorld expert-plan-length
                                    # reduction). When present and ALL
                                    # records in a batch carry it, the v2
                                    # trainer prefers it over `progress`
                                    # for the V-head target — captures
                                    # "did this turn move closer to the
                                    # goal?" rather than the near-degenerate
                                    # terminal-only `raw_env_reward`.
}
```

Why pre-compute embeddings on the producer side rather than ship raw text?
The producer runs once during a parent H-GRPO rollout and already has the
policy + tokenizer in
memory; pushing the cost there means the standalone TurnRD trainer never
re-tokenizes or re-forwards through the policy, and per-step train cost
stays cheap (pure TurnRD-only forward + backward). A `text` variant can
be added later if useful.

torch is imported at module top — this module is consumed only on
torch-enabled hosts (Modal A100 + Mac with the heavy stack); Mac-side
tests gate via `pytest.importorskip("torch")`.
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


# ---------------------------------------------------------------------------
# Record dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnRDRecord:
    """One row of the replay JSONL — the producer's contract with this reader.

    `turn_embeds` is `list[list[float]]` after `json.loads`; this dataclass
    keeps that shape (it's the cheapest representation to pad later in
    `pad_collate`). All conversion to tensors happens in `pad_collate`.
    """

    task_id: str
    turn_embeds: list[list[float]]  # [T_i][D]
    final_reward: float
    judge_labels: Optional[list[float]] = None  # [T_i] or None
    progress: Optional[list[float]] = None  # [T_i] or None — per-turn env signal
    # Optional dense per-turn shaping signal sourced from the env adapter
    # (e.g., ALFWorld's `info["intermediate_reward"]` = expert-plan-length
    # reduction). When present and ALL records in a batch carry it, the
    # v2 trainer's V-head target preference chain prefers it over the
    # legacy `progress` (raw_env_reward) field. None on non-ALFWorld
    # envs (no expert plan available) keeps backward compat for legacy
    # replay buffers.
    progress_signal: Optional[list[float]] = None  # [T_i] or None
    # Round index (0-based) under the orchestrator's protocol. Producer
    # writes the parent `train_loop`'s `--round-idx` value here so the
    # `TurnRDReplayDataset` can apply a recency window at load time
    # (keep only rows from the last N rounds; see
    # `TurnRDReplayDataset.__init__(..., recency_window_rounds=N)`).
    # `None` on legacy replay buffers produced before this field existed;
    # those rows are dropped when a recency window is requested (treated
    # as "older than any tracked round"), and kept when no window is set.
    round_idx: Optional[int] = None
    # Optional per-trajectory AlfWorld goal text, populated by the rollout
    # collector when configured with `turnrd_emit_goal_text=True`. Parsed
    # from the Turn 0 observation via `src.turnrd.goal_extractor
    # .extract_goal_text`. None for non-AlfWorld envs (WebShop, FakeWebShop)
    # and for AlfWorld trajectories whose Turn 0 observation didn't contain
    # the canonical "Your task is to: ..." pattern. Consumed (together
    # with `goal_emb`) by the FiLM goal-conditioned V-head.
    goal_text: Optional[str] = None
    # Optional per-trajectory embedding of `goal_text` in the same
    # representation space as `turn_embeds[t]` (i.e. `[input_dim]`),
    # produced by the rollout collector via the same `turnrd_embedder`
    # callable used for per-turn embeddings. Consumed by the FiLM
    # goal-conditioned V-head when
    # `turnrd.goal_conditioned_value_head=True` is set in the JSON
    # config. `None` on legacy rows and on records produced without
    # the new flag — the dataset's `pad_collate` emits a per-row
    # `goal_emb_mask` so the V-head can zero-mask the contribution of
    # missing-goal rows without dropping them from the batch.
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TurnRDReplayDataset:
    """JSONL-backed replay buffer for the standalone TurnRD trainer.

    Loads + parses on construction so per-step iteration is cheap. We do
    NOT depend on `torch.utils.data` — a plain `list` + a separate
    `pad_collate` helper keeps the test surface trivial and matches the
    rest of the repo's style (see `src/datasets/sft_webshop.py`).

    Args:
        jsonl_path: path to a JSONL file written by the rollout producer.
        mode: 1 (predict R) or 2 (distill judge labels). For Mode 2 the
              dataset additionally filters out rows whose `judge_labels`
              are `None`.
        max_records: optional cap on the number of records to load FROM
              THE START (useful for unit tests + smoke runs).

    NOTE on recency: this dataset deliberately does NOT filter by
    `round_idx`. The replay buffer is append-only by design — stale rows
    are kept and the trainer downweights them at loss time via
    `train_turnrd(..., recency_decay_half_life=N)`. See
    `src.turnrd.train.train_turnrd` for the per-batch decay-weight
    implementation. Producers that emit `round_idx` enable that path;
    legacy rows without `round_idx` get a neutral default weight.
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
                # Filter empty trajectories early — no model call would
                # happen for them anyway.
                turn_embeds = obj.get("turn_embeds", [])
                if not isinstance(turn_embeds, list) or len(turn_embeds) == 0:
                    self.skipped_empty += 1
                    continue
                try:
                    # Backward-compat: legacy producers wrote
                    # `raw_env_rewards`; new producers write `progress`
                    # (the per-turn env signal — already a delta in
                    # AlfWorld where env.step returns step-rewards).
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


# ---------------------------------------------------------------------------
# Padded collate
# ---------------------------------------------------------------------------


def pad_collate(batch: list[TurnRDRecord]) -> dict[str, torch.Tensor]:
    """Pad a list of `TurnRDRecord` to fixed-shape tensors.

    Returns a dict with:
    - `turn_embeds`:    `[B, T_max, D]` float32
    - `attention_mask`: `[B, T_max]` long (1 == real, 0 == padded)
    - `final_reward`:   `[B]` float32
    - `judge_labels`:   `[B, T_max]` float32 — present iff EVERY record in
                        the batch has non-None `judge_labels`. Padded
                        positions are 0.0; the trainer's `loss_mode_2`
                        masks them via `attention_mask`.
    - `progress`:       `[B, T_max]` float32 — present iff EVERY record in
                        the batch has non-None `progress`. Same all-or-
                        nothing semantics as `judge_labels` so the
                        trainer's progress-target path can fire safely
                        without zero-padding looking like real env signal.
    - `progress_signal`:`[B, T_max]` float32 — present iff EVERY record in
                        the batch has non-None `progress_signal`. Same
                        all-or-nothing gate as `progress`. The v2
                        trainer prefers this over `progress` when both
                        are present (ALFWorld dense expert-plan deltas
                        are denser than the near-degenerate raw env
                        reward).
    - `goal_emb`:       `[B, input_dim]` float32 — ALWAYS present when
                        any record in the batch carries a goal embedding;
                        paired with `goal_emb_mask: [B]` bool indicating
                        which rows are real. Absent (key not in dict)
                        when no record in the batch carries `goal_emb`.

    Pure-torch — no `torch.nn.utils.rnn.pad_sequence` dep. Matches the
    style of `TurnRDDecomposer.decompose` which also pads inline.
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
    # Per-trajectory goal embedding (FiLM goal-conditioned V-head). When
    # at least one row carries the field, the [B, D] tensor is always
    # emitted (zeros on absent rows) and a parallel `goal_emb_mask: [B]`
    # flags which rows actually carry a goal embedding so the model can
    # skip the FiLM modulation on missing-goal rows.
    any_goal_emb = any(rec.goal_emb is not None for rec in batch)
    goal_emb: torch.Tensor | None = (
        torch.zeros(B, D, dtype=torch.float32) if any_goal_emb else None
    )
    goal_emb_mask: torch.Tensor | None = (
        torch.zeros(B, dtype=torch.float32) if any_goal_emb else None
    )
    # Round index per record (always emitted; -1 sentinel for legacy rows
    # that lack it). The trainer's recency-decay path reads this to
    # compute per-batch loss weights; the all-or-nothing gate is
    # intentionally absent so mixed legacy + fresh batches still produce
    # a valid (mostly-fresh-weighted) update instead of crashing.
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
