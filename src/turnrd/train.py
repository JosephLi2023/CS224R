"""Standalone TurnRD trainer (Method B).

`train_turnrd(replay_path, mode, model, ...)` runs SGD over a replay
buffer of trajectory records (see `src/turnrd/dataset.py` for the schema).
Two modes:
- Mode 1: regress `predicted_R` against ground-truth `final_reward` (MSE
  via `loss_mode_1`).
- Mode 2: distill cached judge labels into the [CLS] attention head
  (MSE via `loss_mode_2`).

The corresponding Modal app is `infra/app_train_turnrd.py`.
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
    """Sequential batching - no shuffling (pass a pre-shuffled dataset if needed)."""
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
    # Architecture selector. "v1": loss_mode_1 + value_head + entropy +
    # contrastive. "v2": loss_v2_pred + lambda_rank*loss_v2_rank +
    # lambda_progress*loss_v2_progress_prior (Mode 2 is rejected for v2).
    version: str = "v1",
    # v1 Mode-1 aux losses (Mode 2 has direct judge-label supervision so
    # doesn't need either).
    lambda_value: float = 0.5,
    gamma: float = 0.95,
    lambda_entropy: float = 0.01,
    # Contrastive aux loss on per-turn encoder hidden states: pulls
    # success-trajectory turn embeddings together, pushes failure ones apart,
    # giving alpha discriminative features to identify causal turns.
    lambda_contrastive: float = 0.1,
    contrastive_temperature: float = 0.1,
    # v2 loss-mix knobs (effective only when version=="v2").
    lambda_rank: float = 0.1,
    lambda_progress: float = 0.01,
    rank_margin: float = 0.1,
    # Optional learning-rate schedule. The default constant schedule preserves
    # the original optimizer behavior.
    warmup_steps: int = 0,
    lr_schedule: str = "constant",
    # Optional extra pass over recent replay rows. Disabled by default.
    fresh_emphasis_window_rounds: int = 0,
    fresh_emphasis_n_epochs: int = 0,
    # Optional replay-buffer recency decay. When set to H > 0, each batch's
    # loss is scaled by the mean over its records of
    # `0.5 ** ((max_round_idx - round_idx) / H)` (current round -> 1.0, H
    # rounds older -> 0.5, ...). The buffer is never trimmed; stale rows are
    # downweighted. Legacy rows missing `round_idx` get `legacy_decay_weight`.
    # `None` or non-positive disables the decay.
    recency_decay_half_life: float | None = None,
    legacy_decay_weight: float = 0.5,
    # Floor on the per-batch weight so a pure-stale batch still produces a
    # small non-zero gradient.
    min_batch_weight: float = 1e-3,
) -> dict[str, Any]:
    """Train TurnRD on a replay JSONL.

    Args:
        replay_path: path to the JSONL written by the rollout producer.
        mode: 1 (regress R) or 2 (distill judge labels).
        model: a `TurnRD` instance whose `input_dim` matches the JSONL
            embedding width.
        n_epochs: full passes over the dataset.
        batch_size: sequential micro-batch size.
        lr: AdamW learning rate.
        device: optional device override; default = the model's device.
        log_every: log loss every N optimizer steps.
        ckpt_path: if set, write `model.state_dict()` here after the last epoch.
        max_records: optional cap forwarded to `TurnRDReplayDataset`.
        lambda_value: weight on the per-turn V-head MSE
            `(V(h_t) - gamma^(T-t-1)*R)^2`. Mode 1 + `cfg.value_head=True`
            only; 0 disables. Default 0.5.
        gamma: discount factor for the per-turn return target. Default 0.95.
        lambda_entropy: negative-entropy regularization on alpha (subtracts
            beta*H from the Mode-1 loss to keep alpha non-uniform). Default 0.01.

    Returns:
        Dict with `final_loss`, `initial_loss`, `n_steps`, `ckpt_path`,
        `skipped_records`, and `final_loss_breakdown` (component losses).
    """
    if mode not in (1, 2):
        raise ValueError(f"train_turnrd: mode must be 1 or 2; got {mode}.")
    version_norm = str(version).lower()
    if version_norm not in ("v1", "v2"):
        raise ValueError(
            f"train_turnrd: version must be 'v1' or 'v2'; got {version!r}."
        )
    if version_norm == "v2" and mode != 1:
        # v2 has no mode-2 distillation path; reject mode=2 loudly rather than
        # silently fall back to v1 semantics.
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

    # Recency-decay setup. Compute `max_round_idx` once; reused per batch.
    # Disabled (decay_half_life=None) means every batch carries unit weight.
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
            # All-legacy buffer: no meaningful per-round weight, so disable.
            print(
                "[turnrd train] recency-decay requested but no rows carry "
                "`round_idx` (legacy replay buffer); decay disabled.",
                flush=True,
            )
            decay_half_life = None
    decay_batch_weights: list[float] = []


    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

    # Learning-rate schedule setup. The constant schedule is a no-op.
    _n_batches_per_epoch = max(1, (len(dataset) + int(batch_size) - 1) // int(batch_size))
    _total_steps_main = int(n_epochs) * _n_batches_per_epoch
    _total_steps_fresh_est = 0
    if int(fresh_emphasis_window_rounds) > 0 and int(fresh_emphasis_n_epochs) > 0:
        # Over-estimate using the full-buffer batch count; cosine decay reaches
        # the floor a bit early, which is fine for a tail pass.
        _total_steps_fresh_est = int(fresh_emphasis_n_epochs) * _n_batches_per_epoch
    _total_steps_for_schedule = max(1, _total_steps_main + _total_steps_fresh_est)
    scheduler = None
    _lr_schedule_norm = str(lr_schedule).lower().strip()
    if _lr_schedule_norm == "warmup_cosine":
        import math as _math
        _ws = max(0, int(warmup_steps))
        _total_for_decay = max(1, _total_steps_for_schedule - _ws)

        def _lr_lambda(step: int) -> float:
            if step < _ws:
                return float(step + 1) / max(1, _ws)
            progress = float(step - _ws) / float(_total_for_decay)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + _math.cos(_math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        print(
            f"[turnrd train] LR schedule = warmup_cosine: warmup_steps={_ws}, "
            f"total_steps={_total_steps_for_schedule} (main {_total_steps_main} + "
            f"fresh_est {_total_steps_fresh_est})",
            flush=True,
        )
    elif _lr_schedule_norm != "constant":
        raise ValueError(
            f"train_turnrd: lr_schedule must be 'constant' or 'warmup_cosine'; "
            f"got {lr_schedule!r}."
        )

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
            # Pass goal embeddings when the model and replay batch both provide them.
            _flag_goal_cond = bool(
                getattr(getattr(model, "cfg", None), "goal_conditioned_value_head", False)
            )
            if _flag_goal_cond and "goal_emb" in collated:
                _goal_emb_t = collated["goal_emb"].to(target_device)
                _goal_emb_mask_t = collated.get("goal_emb_mask")
                if _goal_emb_mask_t is not None:
                    _goal_emb_mask_t = _goal_emb_mask_t.to(target_device)
                out = model(
                    turn_embeds,
                    attention_mask,
                    goal_emb=_goal_emb_t,
                    goal_emb_mask=_goal_emb_mask_t,
                )
            else:
                out = model(turn_embeds, attention_mask)
            # Track component losses for the last batch's breakdown.
            cls_loss_v = 0.0
            value_loss_v = 0.0
            entropy_v = 0.0
            contrast_loss_v = 0.0
            # v2-specific component losses (default to 0 on the v1 path).
            v2_pred_v = 0.0
            v2_rank_v = 0.0
            v2_progress_v = 0.0
            if version_norm == "v2":
                # v2 loss mix: R-prediction + ranking hinge + progress-prior KL
                # + (when lambda_value > 0) per-turn value-head MSE.
                pred_loss = loss_v2_pred(out, final_reward)
                rank_loss = loss_v2_rank(out, final_reward, margin=float(rank_margin))
                progress_loss = loss_v2_progress_prior(out, attention_mask)
                loss = (
                    pred_loss
                    + float(lambda_rank) * rank_loss
                    + float(lambda_progress) * progress_loss
                )
                if float(lambda_value) != 0.0:
                    # Per-turn value-head target, preference order:
                    #   1. `progress_signal` - dense env-adapter shaping signal
                    #      (e.g. ALFWorld expert-plan-length reduction).
                    #   2. `progress` - legacy per-turn raw_env_reward (dense on
                    #      WebShop, near-degenerate on ALFWorld).
                    #   3. R/T_i - uniform fallback for legacy replays.
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
                        # One-time activation log.
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
                    # Per-row goal_emb masking on V-head loss: with a
                    # goal-conditioned V-head and a `goal_emb_mask`, restrict
                    # the MSE to rows that carry a goal embedding so missing-goal
                    # rows don't pollute the gradient.
                    value_attention_mask = attention_mask
                    if _flag_goal_cond and "goal_emb_mask" in collated:
                        row_mask = collated["goal_emb_mask"].to(
                            target_device
                        ).to(dtype=attention_mask.dtype).unsqueeze(-1)  # [B, 1]
                        # rows with mask=0 get all positions zeroed -> no loss.
                        value_attention_mask = attention_mask * row_mask.to(
                            attention_mask.dtype
                        )
                    value_loss = loss_v2_value(out, target_v, value_attention_mask)
                    loss = loss + float(lambda_value) * value_loss
                    value_loss_v = float(value_loss.detach().item())
                v2_pred_v = float(pred_loss.detach().item())
                v2_rank_v = float(rank_loss.detach().item())
                v2_progress_v = float(progress_loss.detach().item())
                cls_loss_v = v2_pred_v  # report the headline R-loss in cls slot too
            elif mode == 1:
                cls_loss = loss_mode_1(out, final_reward)
                # Aux per-turn value loss: trains V(h_t) to predict
                # gamma^(T-t-1)*R per real turn.
                value_loss = loss_value_head(
                    out, final_reward, attention_mask, gamma=float(gamma)
                )
                # Negative-entropy reg: subtract beta*H(alpha) so the loss
                # penalizes uniform decompositions (otherwise R-prediction is
                # satisfied by any alpha and stays at uniform-init).
                H = alpha_entropy(out, attention_mask)
                # Contrastive on per-turn encoder hidden states: discriminate
                # success vs failure so alpha has features to attend over.
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

            # Apply recency decay (per-batch scalar): scale the batch loss by
            # the mean decay weight over its records (legacy rows get
            # `legacy_decay_weight`), floored at `min_batch_weight`.
            if decay_half_life is not None and decay_max_round_idx is not None:
                round_idx_batch = collated["round_idx"].to(target_device)
                # legacy (-1 sentinel) -> `legacy_decay_weight`; otherwise
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

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
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

    # Optional fresh-emphasis pass: a second pass over the last
    # `fresh_emphasis_window_rounds` rounds for `fresh_emphasis_n_epochs`
    # epochs. Recency decay is off here (rows are already fresh); the
    # scheduler keeps decaying.
    n_steps_fresh = 0
    fresh_emphasis_info: dict[str, Any] | None = None
    if int(fresh_emphasis_window_rounds) > 0 and int(fresh_emphasis_n_epochs) > 0:
        # Compute max_round_idx independently of the decay path (fresh-emphasis
        # works regardless of the decay setting).
        _all_round_idx = [r.round_idx for r in dataset if r.round_idx is not None]
        if _all_round_idx:
            _max_round = max(_all_round_idx)
            _min_recent = int(_max_round) - int(fresh_emphasis_window_rounds) + 1
            _fresh_records = [
                r for r in dataset
                if r.round_idx is not None and r.round_idx >= _min_recent
            ]
        else:
            _min_recent = None
            _fresh_records = []
        if _fresh_records:
            class _ListShim:
                def __init__(self, records): self.records = records
                def __len__(self): return len(self.records)
                def __getitem__(self, i): return self.records[i]
            _fresh_ds = _ListShim(_fresh_records)
            print(
                f"[turnrd train] fresh-emphasis pass: {len(_fresh_records)} rows "
                f"(round_idx >= {_min_recent}), {fresh_emphasis_n_epochs} epochs, "
                f"decay=OFF",
                flush=True,
            )
            for fe_epoch in range(int(fresh_emphasis_n_epochs)):
                model.train()
                for batch in _iter_batches(_fresh_ds, batch_size):
                    collated = pad_collate(batch)
                    turn_embeds = collated["turn_embeds"].to(target_device)
                    attention_mask = collated["attention_mask"].to(target_device)
                    final_reward = collated["final_reward"].to(target_device)
                    optimizer.zero_grad(set_to_none=True)
                    _flag_goal_cond = bool(
                        getattr(getattr(model, "cfg", None), "goal_conditioned_value_head", False)
                    )
                    if _flag_goal_cond and "goal_emb" in collated:
                        _goal_emb_t = collated["goal_emb"].to(target_device)
                        _goal_emb_mask_t = collated.get("goal_emb_mask")
                        if _goal_emb_mask_t is not None:
                            _goal_emb_mask_t = _goal_emb_mask_t.to(target_device)
                        out = model(
                            turn_embeds, attention_mask,
                            goal_emb=_goal_emb_t, goal_emb_mask=_goal_emb_mask_t,
                        )
                    else:
                        out = model(turn_embeds, attention_mask)
                    # v2-only loss (matches the main loop's v2 branch).
                    if version_norm != "v2":
                        raise RuntimeError(
                            "train_turnrd: fresh_emphasis pass only supports "
                            "version='v2' (current SOTA recipe)."
                        )
                    pred_loss = loss_v2_pred(out, final_reward)
                    rank_loss = loss_v2_rank(out, final_reward, margin=float(rank_margin))
                    progress_loss = loss_v2_progress_prior(out, attention_mask)
                    loss = (
                        pred_loss
                        + float(lambda_rank) * rank_loss
                        + float(lambda_progress) * progress_loss
                    )
                    if float(lambda_value) != 0.0:
                        fmask = attention_mask.to(dtype=final_reward.dtype)
                        if "progress_signal" in collated:
                            target_v = collated["progress_signal"].to(target_device).to(
                                dtype=final_reward.dtype
                            ) * fmask
                        elif "progress" in collated:
                            target_v = collated["progress"].to(target_device).to(
                                dtype=final_reward.dtype
                            ) * fmask
                        else:
                            T_i = fmask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                            target_v = (final_reward.unsqueeze(-1) / T_i) * fmask
                        value_attention_mask = attention_mask
                        if _flag_goal_cond and "goal_emb_mask" in collated:
                            row_mask = collated["goal_emb_mask"].to(
                                target_device
                            ).to(dtype=attention_mask.dtype).unsqueeze(-1)
                            value_attention_mask = attention_mask * row_mask.to(
                                attention_mask.dtype
                            )
                        value_loss = loss_v2_value(out, target_v, value_attention_mask)
                        loss = loss + float(lambda_value) * value_loss
                    # NO recency-decay multiplication (this is the fresh pass).
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    n_steps_fresh += 1
                    final_loss = float(loss.detach().item())
            fresh_emphasis_info = {
                "window_rounds": int(fresh_emphasis_window_rounds),
                "n_epochs": int(fresh_emphasis_n_epochs),
                "n_rows": len(_fresh_records),
                "min_round_idx": _min_recent,
                "n_steps": n_steps_fresh,
            }
        else:
            print(
                f"[turnrd train] fresh-emphasis: NO records with round_idx >= "
                f"{_min_recent} — skipping (likely all-legacy dataset).",
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
        "n_steps_fresh": int(n_steps_fresh),
        "ckpt_path": saved_ckpt,
        "skipped_records": int(dataset.skipped_empty + dataset.skipped_missing_judge),
        "version": version_norm,
        "lr_schedule": _lr_schedule_norm,
        "warmup_steps": int(warmup_steps),
        "fresh_emphasis": fresh_emphasis_info,
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
