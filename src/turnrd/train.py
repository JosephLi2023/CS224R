"""Standalone TurnRD trainer (Method B).

`train_turnrd(replay_path, mode, model, ...)` runs SGD over a replay
buffer of trajectory records (see `src/turnrd/dataset.py` for the schema).
Two modes:
- Mode 1: regress `predicted_R` against ground-truth `final_reward` (MSE
  via `loss_mode_1`).
- Mode 2: distill cached judge labels into the [CLS] attention head
  (MSE via `loss_mode_2`).

This entrypoint is pure-Python; any Modal `@app.function` can wrap it.
The corresponding Modal app is `infra/app_train_turnrd.py`.

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
    TurnRDv2,
    alpha_entropy,
    loss_contrastive,
    loss_mode_1,
    loss_mode_2,
    loss_v2_pred,
    loss_v2_progress_prior,
    loss_v2_rank,
    loss_v2_value,
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
    model: "TurnRD | TurnRDv2",
    n_epochs: int = 1,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str | torch.device | None = None,
    log_every: int = 50,
    ckpt_path: str | os.PathLike[str] | None = None,
    max_records: int | None = None,
    # Architecture selector. "v1" runs the legacy
    # `loss_mode_1 + value_head + entropy + contrastive` mix.
    # "v2" runs the new
    # `loss_v2_pred + λ_rank·loss_v2_rank + λ_progress·loss_v2_progress_prior`
    # mix (Mode 2 is rejected for v2 — v2 doesn't have a mode-2
    # distillation path in this prototype).
    version: str = "v1",
    # v1 Mode-1 aux losses (Mode 2 has direct judge-label supervision so
    # doesn't need either).
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
    # v2 loss-mix knobs (effective only when version=="v2").
    lambda_rank: float = 0.1,
    lambda_progress: float = 0.01,
    rank_margin: float = 0.1,
    # Optional replay-buffer **recency decay**. When set to H > 0, each
    # batch's loss is multiplied by a per-batch scalar equal to the
    # mean over its records of `0.5 ** ((max_round_idx - round_idx) / H)`,
    # where `max_round_idx` is the highest `round_idx` present in the
    # full dataset. Concretely: a record from the current round gets
    # weight 1.0, one H rounds older gets weight 0.5, one 2H rounds
    # older gets 0.25, etc. The buffer is NEVER physically trimmed —
    # stale rows are kept and downweighted, preserving information
    # while reducing their gradient contribution. Legacy rows missing
    # `round_idx` (None / -1 sentinel) receive a fixed default weight
    # of `legacy_decay_weight` so they neither dominate nor disappear.
    # `None` or a non-positive value disables the decay entirely (every
    # batch contributes unscaled) — preserves backward compat.
    recency_decay_half_life: float | None = None,
    legacy_decay_weight: float = 0.5,
    # Numerical floor on the per-batch weight so a batch of pure-stale
    # records still produces a small (but non-zero) gradient. Prevents
    # the optimizer from no-op'ing on highly-stale batches when the
    # half-life is aggressive.
    min_batch_weight: float = 1e-3,
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
    version_norm = str(version).lower()
    if version_norm not in ("v1", "v2"):
        raise ValueError(
            f"train_turnrd: version must be 'v1' or 'v2'; got {version!r}."
        )
    if version_norm == "v2" and mode != 1:
        # v2 has no mode-2 distillation path — its primary credit
        # signal is the identifiable Σα·v R-prediction loss + ranking +
        # progress prior. Reject mode=2 loudly so misconfigured runs
        # don't silently fall back to v1 semantics.
        raise ValueError(
            "train_turnrd: version='v2' currently only supports mode=1; "
            f"got mode={mode}."
        )
    if n_epochs <= 0:
        raise ValueError(f"train_turnrd: n_epochs must be positive; got {n_epochs}.")
    if batch_size <= 0:
        raise ValueError(f"train_turnrd: batch_size must be positive; got {batch_size}.")

    dataset = TurnRDReplayDataset(
        replay_path, mode=mode,
        max_records=max_records,
    )
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

    # ---- Recency-decay setup --------------------------------------------
    # Compute `max_round_idx` once over the dataset; reused per batch.
    # Disabled (decay_half_life=None) means every batch carries unit
    # weight (no behavior change).
    decay_half_life: float | None = (
        float(recency_decay_half_life)
        if recency_decay_half_life is not None and float(recency_decay_half_life) > 0.0
        else None
    )
    decay_max_round_idx: int | None = None
    decay_n_with_round_idx = 0
    decay_n_without_round_idx = 0
    if decay_half_life is not None:
        tracked = [r.round_idx for r in dataset if r.round_idx is not None]
        decay_n_with_round_idx = len(tracked)
        decay_n_without_round_idx = len(dataset) - decay_n_with_round_idx
        if tracked:
            decay_max_round_idx = max(tracked)
            print(
                f"[turnrd train] recency-decay enabled: half_life={decay_half_life} "
                f"rounds, max_round_idx={decay_max_round_idx}, "
                f"{decay_n_with_round_idx} rows w/ round_idx, "
                f"{decay_n_without_round_idx} legacy rows (weight={legacy_decay_weight}).",
                flush=True,
            )
        else:
            # All-legacy buffer: cannot compute a meaningful per-round
            # weight. Disable the decay path entirely to avoid a misleading
            # uniform scaling.
            print(
                "[turnrd train] recency-decay requested but no rows carry "
                "`round_idx` (legacy replay buffer); decay disabled.",
                flush=True,
            )
            decay_half_life = None
    decay_batch_weights: list[float] = []
    # ---------------------------------------------------------------------

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
            # v2-specific component losses (default to 0 on the v1 path).
            v2_pred_v = 0.0
            v2_rank_v = 0.0
            v2_progress_v = 0.0
            if version_norm == "v2":
                # v2 loss mix: identifiable R-prediction + within-batch
                # ranking hinge + KL pull toward the progress prior +
                # (when lambda_value > 0) per-turn value-head MSE against
                # the env-side progress signal (Tier-3 Phase A: collector
                # emits `progress` per turn; pad_collate forwards; this
                # loop prefers it over the legacy R/T_i fallback target).
                pred_loss = loss_v2_pred(out, final_reward)
                rank_loss = loss_v2_rank(out, final_reward, margin=float(rank_margin))
                progress_loss = loss_v2_progress_prior(out, attention_mask)
                loss = (
                    pred_loss
                    + float(lambda_rank) * rank_loss
                    + float(lambda_progress) * progress_loss
                )
                if float(lambda_value) != 0.0:
                    # Per-turn value-head target. Preference chain
                    # (highest priority first):
                    #   1. `progress_signal` \u2014 dense per-turn shaping
                    #      signal sourced from the env adapter (e.g.
                    #      ALFWorld's expert-plan-length reduction \u2014
                    #      "did this turn move strictly closer to the
                    #      goal?"). Strictly more informative than the
                    #      raw env reward on sparse-terminal envs.
                    #   2. `progress` \u2014 legacy per-turn raw_env_reward
                    #      list. On ALFWorld this is near-degenerate
                    #      (zero everywhere except the terminal turn);
                    #      preserved for WebShop where step rewards are
                    #      already meaningful deltas.
                    #   3. R/T_i \u2014 uniform fallback for legacy replays
                    #      that carry neither field.
                    fmask = attention_mask.to(dtype=final_reward.dtype)
                    if "progress_signal" in collated:
                        target_v = collated["progress_signal"].to(target_device).to(
                            dtype=final_reward.dtype
                        ) * fmask
                        if not getattr(train_turnrd, "_progress_signal_seen", False):
                            train_turnrd._progress_signal_seen = True  # type: ignore[attr-defined]
                            print(
                                "[turnrd train] v2 value-head target = "
                                "expert-plan progress signal "
                                "(progress_signal field present in batch).",
                                flush=True,
                            )
                    elif "progress" in collated:
                        target_v = collated["progress"].to(target_device).to(
                            dtype=final_reward.dtype
                        ) * fmask
                        # One-time activation log (the silent-fallback
                        # mode is the riskiest failure \u2014 make it
                        # impossible to miss in stdout).
                        if not getattr(train_turnrd, "_progress_seen", False):
                            train_turnrd._progress_seen = True  # type: ignore[attr-defined]
                            print(
                                "[turnrd train] v2 value-head target = "
                                "per-turn progress signal (progress field "
                                "present in batch).",
                                flush=True,
                            )
                    else:
                        T_i = fmask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                        target_v = (final_reward.unsqueeze(-1) / T_i) * fmask
                        if not getattr(train_turnrd, "_fallback_seen", False):
                            train_turnrd._fallback_seen = True  # type: ignore[attr-defined]
                            print(
                                "[turnrd train] v2 value-head target = "
                                "R/T_i fallback (no `progress` field in "
                                "batch \u2014 legacy replay buffer).",
                                flush=True,
                            )
                    value_loss = loss_v2_value(out, target_v, attention_mask)
                    loss = loss + float(lambda_value) * value_loss
                    value_loss_v = float(value_loss.detach().item())
                v2_pred_v = float(pred_loss.detach().item())
                v2_rank_v = float(rank_loss.detach().item())
                v2_progress_v = float(progress_loss.detach().item())
                cls_loss_v = v2_pred_v  # report the headline R-loss in cls slot too
            elif mode == 1:
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

            # ---- Apply recency decay (per-batch scalar) ----------------
            # When enabled, scale the batch's total loss by the mean
            # decay weight over its records. Recent rows pull the mean
            # toward 1.0; old rows pull it toward 0. Legacy rows (no
            # round_idx) get a fixed `legacy_decay_weight`. Final
            # multiplier is floored at `min_batch_weight` so a batch of
            # pure-stale records still contributes a tiny gradient.
            if decay_half_life is not None and decay_max_round_idx is not None:
                round_idx_batch = collated["round_idx"].to(target_device)
                # legacy (-1 sentinel) → use `legacy_decay_weight`; otherwise
                # 0.5 ** ((max - round_idx) / half_life).
                legacy_mask = (round_idx_batch < 0)
                age = (
                    torch.full_like(round_idx_batch, decay_max_round_idx)
                    - round_idx_batch
                ).to(dtype=loss.dtype)
                # Avoid negative ages (a fresh row in a later round): clamp.
                age = age.clamp(min=0.0)
                weights = torch.pow(
                    torch.tensor(0.5, dtype=loss.dtype, device=target_device),
                    age / float(decay_half_life),
                )
                weights = torch.where(
                    legacy_mask,
                    torch.full_like(weights, float(legacy_decay_weight)),
                    weights,
                )
                batch_weight = weights.mean()
                batch_weight = torch.clamp(batch_weight, min=float(min_batch_weight))
                decay_batch_weights.append(float(batch_weight.detach().item()))
                loss = loss * batch_weight
            # -------------------------------------------------------------

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
        "version": version_norm,
        "recency_decay": {
            "half_life": (
                float(decay_half_life) if decay_half_life is not None else None
            ),
            "max_round_idx": decay_max_round_idx,
            "legacy_decay_weight": float(legacy_decay_weight),
            "min_batch_weight": float(min_batch_weight),
            "n_rows_with_round_idx": int(decay_n_with_round_idx),
            "n_rows_legacy": int(decay_n_without_round_idx),
            "batch_weight_mean": (
                float(sum(decay_batch_weights) / len(decay_batch_weights))
                if decay_batch_weights else None
            ),
            "batch_weight_min": (
                float(min(decay_batch_weights)) if decay_batch_weights else None
            ),
            "batch_weight_max": (
                float(max(decay_batch_weights)) if decay_batch_weights else None
            ),
        },
        "final_loss_breakdown": {
            "cls_loss": cls_loss_v,
            "value_loss": value_loss_v,
            "alpha_entropy": entropy_v,
            "contrast_loss": contrast_loss_v,
            "v2_pred_loss": v2_pred_v,
            "v2_rank_loss": v2_rank_v,
            "v2_progress_loss": v2_progress_v,
            "lambda_value": float(lambda_value),
            "lambda_entropy": float(lambda_entropy),
            "lambda_contrastive": float(lambda_contrastive),
            "lambda_rank": float(lambda_rank),
            "lambda_progress": float(lambda_progress),
            "rank_margin": float(rank_margin),
            "contrastive_temperature": float(contrastive_temperature),
            "gamma": float(gamma),
        },
    }
