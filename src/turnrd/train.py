"""Standalone TurnRD trainer (Method B; Day 13).

`train_turnrd(replay_path, mode, model, ...)` runs SGD over a replay
buffer of trajectory records (see `src/turnrd/dataset.py` for the schema).
Two modes:
- Mode 1: regress `predicted_R` against ground-truth `final_reward` (MSE
  via `loss_mode_1`).
- Mode 2: distill cached judge labels into the [CLS] attention head
  (MSE via `loss_mode_2`).

This entrypoint is pure-Python; any Modal `@app.function` can wrap it.
The corresponding Modal app (`infra/app_train_turnrd.py`) is a
follow-up — see plan "Deliberately NOT included".

torch is imported at module top — same gating pattern as
`src/turnrd/model.py`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterator

import torch

from src.turnrd.dataset import TurnRDRecord, TurnRDReplayDataset, pad_collate
from src.turnrd.model import (
    TurnRD,
    alpha_entropy,
    loss_contrastive,
    loss_mode_1,
    loss_mode_2,
    loss_value_head,
)

logger = logging.getLogger(__name__)


def _iter_batches(
    dataset: TurnRDReplayDataset, batch_size: int
) -> Iterator[list[TurnRDRecord]]:
    """Sequential batching — no shuffling. Trainer-level shuffling can be
    added later by passing a pre-shuffled dataset; keeping this simple
    matches the producer's "stream as written" semantics.
    """
    bs = max(1, int(batch_size))
    n = len(dataset)
    for start in range(0, n, bs):
        end = min(start + bs, n)
        yield [dataset[i] for i in range(start, end)]


def train_turnrd(
    replay_path: str | os.PathLike[str],
    mode: int,
    *,
    model: TurnRD,
    n_epochs: int = 1,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str | torch.device | None = None,
    log_every: int = 50,
    ckpt_path: str | os.PathLike[str] | None = None,
    max_records: int | None = None,
    # Post-WebShop-attempt aux losses (Mode 1 only — Mode 2 has direct
    # judge-label supervision so doesn't need either).
    lambda_value: float = 0.5,
    gamma: float = 0.95,
    lambda_entropy: float = 0.01,
    # v8 Tier 1: contrastive aux loss on per-turn encoder hidden states.
    # Pulls success-trajectory turn embeddings together; pushes them
    # apart from failure-trajectory turn embeddings. Self-supervised
    # — uses only the binary R>0 vs R==0 signal we already have.
    # Forces the encoder to learn discriminative features that α can
    # use to identify causal turns.
    lambda_contrastive: float = 0.1,
    contrastive_temperature: float = 0.1,
) -> dict[str, Any]:
    """Train TurnRD on a replay JSONL.

    Args:
        replay_path: path to the JSONL written by the rollout producer.
        mode: 1 (regress R) or 2 (distill judge labels).
        model: a `TurnRD` instance whose `input_dim` matches the
            embedding width in the JSONL.
        n_epochs: number of full passes over the dataset.
        batch_size: sequential micro-batch size.
        lr: AdamW learning rate.
        device: optional device override; default = the model's parameter
            device.
        log_every: print loss every N optimizer steps (no W&B dep).
        ckpt_path: if provided, write `model.state_dict()` here after the
            last epoch.
        max_records: optional cap forwarded to `TurnRDReplayDataset`.
        lambda_value: weight on the per-turn V-head MSE loss
            `(V(h_t) - γ^(T-t-1)·R)²`. Only effective in Mode 1 AND
            when the model was built with `cfg.value_head=True`. Set
            to 0 to disable. Default 0.5.
        gamma: discount factor for the per-turn return target. Default
            0.95. With sparse final R, smaller γ skews credit toward
            later turns.
        lambda_entropy: NEGATIVE entropy regularization strength on α
            (subtracts β·H from the Mode-1 loss). Encourages α to
            commit to a small set of high-credit turns rather than
            collapse to uniform. Set to 0 to disable. Default 0.01.

    Returns:
        Dict with `final_loss`, `initial_loss`, `n_steps`, `ckpt_path`,
        `skipped_records`, plus `final_loss_breakdown` showing the
        last batch's component losses (cls / value / entropy).
    """
    if mode not in (1, 2):
        raise ValueError(f"train_turnrd: mode must be 1 or 2; got {mode}.")
    if n_epochs <= 0:
        raise ValueError(f"train_turnrd: n_epochs must be positive; got {n_epochs}.")
    if batch_size <= 0:
        raise ValueError(f"train_turnrd: batch_size must be positive; got {batch_size}.")

    dataset = TurnRDReplayDataset(replay_path, mode=mode, max_records=max_records)
    if len(dataset) == 0:
        raise ValueError(
            f"train_turnrd: dataset at {replay_path} has 0 usable records "
            f"(mode={mode}, skipped_empty={dataset.skipped_empty}, "
            f"skipped_missing_judge={dataset.skipped_missing_judge})."
        )

    if device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)
    model.to(target_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

    initial_loss: float | None = None
    final_loss: float = float("nan")
    n_steps = 0

    for epoch in range(int(n_epochs)):
        model.train()
        for batch in _iter_batches(dataset, batch_size):
            collated = pad_collate(batch)
            turn_embeds = collated["turn_embeds"].to(target_device)
            attention_mask = collated["attention_mask"].to(target_device)
            final_reward = collated["final_reward"].to(target_device)

            optimizer.zero_grad(set_to_none=True)
            out = model(turn_embeds, attention_mask)
            # Track component losses for the last batch's breakdown
            # (returned in the train_turnrd summary for diagnostics).
            cls_loss_v = 0.0
            value_loss_v = 0.0
            entropy_v = 0.0
            contrast_loss_v = 0.0
            if mode == 1:
                cls_loss = loss_mode_1(out, final_reward)
                # Aux per-turn value loss: trains V(h_t) to predict
                # γ^(T-t-1)·R per real turn. Gives the encoder a
                # credit-relevant signal under sparse R alone.
                value_loss = loss_value_head(
                    out, final_reward, attention_mask, gamma=float(gamma)
                )
                # Negative-entropy reg: subtract β·H(α) so the loss
                # PENALIZES uniform decompositions. Without this the
                # standalone R-prediction objective (which is satisfied
                # by ANY α since Σα=1 → CLS gets the same R) leaves α
                # at uniform-init.
                H = alpha_entropy(out, attention_mask)
                # v8 Tier 1: contrastive on per-turn encoder hidden states.
                # Discriminate success vs failure trajectories at the
                # encoder level → α has discriminative features to attend over.
                contrast_loss = loss_contrastive(
                    out,
                    final_reward,
                    attention_mask,
                    temperature=float(contrastive_temperature),
                )
                loss = (
                    cls_loss
                    + float(lambda_value) * value_loss
                    - float(lambda_entropy) * H
                    + float(lambda_contrastive) * contrast_loss
                )
                cls_loss_v = float(cls_loss.detach().item())
                value_loss_v = float(value_loss.detach().item())
                entropy_v = float(H.detach().item())
                contrast_loss_v = float(contrast_loss.detach().item())
            else:  # mode == 2
                if "judge_labels" not in collated:
                    raise RuntimeError(
                        "train_turnrd(mode=2): pad_collate did not produce "
                        "judge_labels — at least one record in the batch had "
                        "judge_labels=None. The dataset filter should have "
                        "dropped it."
                    )
                judge_labels = collated["judge_labels"].to(target_device)
                loss = loss_mode_2(out, judge_labels, final_reward, attention_mask)
                cls_loss_v = float(loss.detach().item())

            loss.backward()
            optimizer.step()
            n_steps += 1

            loss_value = float(loss.detach().item())
            if initial_loss is None:
                initial_loss = loss_value
            final_loss = loss_value

            if log_every > 0 and n_steps % int(log_every) == 0:
                logger.info(
                    "[turnrd train] epoch=%d step=%d loss=%.6f",
                    epoch,
                    n_steps,
                    loss_value,
                )
                # Mirror to stdout so pure-script invocations also see progress.
                print(
                    f"[turnrd train] epoch={epoch} step={n_steps} loss={loss_value:.6f}",
                    flush=True,
                )

    saved_ckpt: str | None = None
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        saved_ckpt = str(ckpt_path)

    return {
        "final_loss": float(final_loss),
        "initial_loss": float(initial_loss if initial_loss is not None else float("nan")),
        "n_steps": int(n_steps),
        "ckpt_path": saved_ckpt,
        "skipped_records": int(dataset.skipped_empty + dataset.skipped_missing_judge),
        "final_loss_breakdown": {
            "cls_loss": cls_loss_v,
            "value_loss": value_loss_v,
            "alpha_entropy": entropy_v,
            "contrast_loss": contrast_loss_v,
            "lambda_value": float(lambda_value),
            "lambda_entropy": float(lambda_entropy),
            "lambda_contrastive": float(lambda_contrastive),
            "contrastive_temperature": float(contrastive_temperature),
            "gamma": float(gamma),
        },
    }
