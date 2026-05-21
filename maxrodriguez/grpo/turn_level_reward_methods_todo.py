"""Torch turn-level reward methods for Max's ALFWorld alpha-GRPO runs.

Each method returns one reward per real turn:

    list[K][T_i]

No method pads trajectories with fake turns. The GRPO loss later converts
these rewards into turn advantages and combines them with trajectory GRPO:

    A_H = alpha * A_turn + (1 - alpha) * A_traj
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import torch
from torch.nn import functional as F
from torch import Tensor, nn

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup
from src.datasets.sft_alfworld import synthesize_sft_target


TurnRewards = list[list[float]]


class TurnRewardMethod(Protocol):
    def decompose(self, group: TrajectoryGroup) -> TurnRewards:
        """Return one float reward per real turn."""


def _assert_ragged_shape(group: TrajectoryGroup, rewards: TurnRewards) -> None:
    if len(rewards) != len(group.trajectories):
        raise ValueError("turn reward rows must match number of trajectories")
    for traj, row in zip(group.trajectories, rewards):
        if len(row) != len(traj.turns):
            raise ValueError("each reward row must match that trajectory's turns")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _turn_progress(turn: object) -> float:
    intermediate = getattr(turn, "intermediate_reward", None)
    if intermediate is not None:
        return _safe_float(intermediate)
    return _safe_float(getattr(turn, "raw_env_reward", 0.0))


def _trajectory_feature_tensor(
    traj: Trajectory,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build lightweight per-turn features when hidden embeddings are absent."""
    rows: list[list[float]] = []
    n_turns = max(len(traj.turns), 1)
    for turn in traj.turns:
        progress = _turn_progress(turn)
        raw = _safe_float(getattr(turn, "raw_env_reward", 0.0))
        position = _safe_float(getattr(turn, "turn_idx", len(rows))) / max(n_turns - 1, 1)
        action_tokens = float(len(getattr(turn, "action_token_ids", ()) or ()))
        prompt_tokens = _safe_float(getattr(turn, "prompt_token_count", 0.0))
        rows.append([progress, raw, position, action_tokens, prompt_tokens])
    return torch.as_tensor(rows, device=device, dtype=dtype)


def _to_float_rows(rows: list[Tensor]) -> TurnRewards:
    return [row.detach().cpu().tolist() for row in rows]


@dataclass
class ProgressDeltaTODO:
    """Deterministic ALFWorld progress reward using Torch tensors internally."""

    terminal_bonus: float = 1.0
    device: str = "cpu"

    def decompose(self, group: TrajectoryGroup) -> TurnRewards:
        reward_rows: list[Tensor] = []
        device = torch.device(self.device)

        for traj in group.trajectories:
            values = [_turn_progress(turn) for turn in traj.turns]
            row = torch.as_tensor(values, device=device, dtype=torch.float32)
            if row.numel() and float(traj.final_reward) > 0.0:
                bonus = torch.zeros_like(row)
                bonus[-1] = float(self.terminal_bonus)
                row = row + bonus
            reward_rows.append(row)

        rewards = _to_float_rows(reward_rows)
        _assert_ragged_shape(group, rewards)
        return rewards


ALFWorldProgressDeltaTODO = ProgressDeltaTODO


class SignedAttentionTransformer(nn.Module):
    """Transformer scorer that produces centered signed turn rewards.

    Softmax gives positive weights. We center them with:

        centered_weight_t = T * softmax(score)_t - 1

    so the average centered weight is zero and individual turns can be negative.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.score_head = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor, mask: Tensor | None = None) -> Tensor:
        if features.dim() != 3:
            raise ValueError(f"features must be [B,T,D], got {tuple(features.shape)}")

        hidden = self.input_proj(features)
        key_padding_mask = None if mask is None else ~mask.bool()
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        scores = self.score_head(encoded).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
            lengths = mask.sum(dim=1).clamp_min(1).to(dtype=scores.dtype)
        else:
            lengths = torch.full(
                (scores.shape[0],),
                scores.shape[1],
                device=scores.device,
                dtype=scores.dtype,
            )

        attention = torch.softmax(scores, dim=1)
        centered = lengths.unsqueeze(1) * attention - 1.0
        if mask is not None:
            centered = centered.masked_fill(~mask.bool(), 0.0)
        return centered


def save_signed_attention_transformer_checkpoint(
    model: SignedAttentionTransformer,
    ckpt_path: str,
    *,
    hidden_size: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
) -> None:
    payload = {
        "model_config": {
            "input_size": 5,
            "hidden_size": int(hidden_size),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
            "dropout": float(dropout),
        },
        "state_dict": model.state_dict(),
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(payload, ckpt_path)


def load_signed_attention_transformer_checkpoint(
    ckpt_path: str,
    *,
    device: str = "cpu",
) -> SignedAttentionTransformer:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = dict(payload.get("model_config", {}) or {})
    model = SignedAttentionTransformer(
        input_size=int(model_config.get("input_size", 5)),
        hidden_size=int(model_config.get("hidden_size", 128)),
        n_heads=int(model_config.get("n_heads", 4)),
        n_layers=int(model_config.get("n_layers", 2)),
        dropout=float(model_config.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model


@dataclass
class SignedAttentionTODO:
    """Learned signed attention over turns.

    This is the alpha-GRPO transformer option. It is Torch-native and centered,
    so a softmax attention distribution can still produce negative turn rewards.
    """

    model: SignedAttentionTransformer | None = None
    hidden_size: int = 128
    outcome_scale: float = 1.0
    device: str = "cpu"

    def decompose(self, group: TrajectoryGroup) -> TurnRewards:
        device = torch.device(self.device)
        reward_rows: list[Tensor] = []
        if self.model is None:
            for traj in group.trajectories:
                if not traj.turns:
                    reward_rows.append(torch.empty(0, device=device))
                    continue

                progress = torch.tensor(
                    [_turn_progress(turn) for turn in traj.turns],
                    device=device,
                    dtype=torch.float32,
                )
                centered_progress = progress - progress.mean()
                scale = centered_progress.abs().max().clamp_min(1e-8)
                normalized_progress = centered_progress / scale
                positions = torch.linspace(
                    0.0,
                    1.0,
                    steps=progress.numel(),
                    device=device,
                    dtype=torch.float32,
                )
                scores = normalized_progress + 0.1 * positions
                attention = torch.softmax(scores, dim=0)
                centered = progress.numel() * attention - 1.0
                outcome_sign = 1.0 if float(traj.final_reward) > 0.0 else -1.0
                reward_rows.append(float(self.outcome_scale) * outcome_sign * centered)
        else:
            model = self.model.to(device)
            model.eval()
            with torch.no_grad():
                for traj in group.trajectories:
                    if not traj.turns:
                        reward_rows.append(torch.empty(0, device=device))
                        continue

                    features = _trajectory_feature_tensor(
                        traj,
                        device=device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    mask = torch.ones(
                        (1, features.shape[1]),
                        device=device,
                        dtype=torch.bool,
                    )
                    centered = model(features, mask=mask).squeeze(0)
                    outcome_sign = 1.0 if float(traj.final_reward) > 0.0 else -1.0
                    reward_rows.append(float(self.outcome_scale) * outcome_sign * centered)

        rewards = _to_float_rows(reward_rows)
        _assert_ragged_shape(group, rewards)
        return rewards


def train_signed_attention_transformer(
    model: SignedAttentionTransformer,
    groups: Sequence[TrajectoryGroup],
    *,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    device: str = "cuda",
) -> list[float]:
    """Train the signed transformer on centered ALFWorld progress targets.

    Target per trajectory:

        target_t = sign(R) * center(progress_t)

    where `center(progress_t)` subtracts the per-trajectory mean and rescales by
    max absolute centered value. That gives a direct signed turn-level training
    signal instead of training a positive-only softmax decomposition.
    """
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    losses: list[float] = []

    for _epoch in range(int(epochs)):
        for group in groups:
            for traj in group.trajectories:
                if not traj.turns:
                    continue

                features = _trajectory_feature_tensor(
                    traj,
                    device=target_device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                mask = torch.ones(
                    (1, features.shape[1]),
                    device=target_device,
                    dtype=torch.bool,
                )
                progress = torch.tensor(
                    [_turn_progress(turn) for turn in traj.turns],
                    device=target_device,
                    dtype=torch.float32,
                )
                centered_progress = progress - progress.mean()
                scale = centered_progress.abs().max().clamp_min(1e-8)
                outcome_sign = 1.0 if float(traj.final_reward) > 0.0 else -1.0
                target = (float(outcome_sign) * centered_progress / scale).unsqueeze(0)

                pred = model(features, mask=mask)
                loss = F.mse_loss(pred[mask], target[mask])

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

    return losses


@dataclass
class AdmissibleActionMarginTODO:
    """Margin between chosen action score and best admissible alternative."""

    max_actions_to_score: int = 32
    normalize_margin: bool = True
    policy: object | None = None
    tokenizer: object | None = None
    max_seq_len: int = 2048
    score_normalization: str = "mean"
    device: str = "cpu"

    def _score_candidates(
        self,
        prompt_token_ids: Sequence[int],
        actions: list[str],
    ) -> dict[str, float]:
        if self.policy is None or self.tokenizer is None or not actions:
            return {}

        model = getattr(self.policy, "model", None)
        tokenizer = self.tokenizer
        if model is None:
            return {}

        unique_actions = list(dict.fromkeys(str(action).strip() for action in actions if str(action).strip()))
        if not unique_actions:
            return {}

        rows: list[dict[str, object]] = []
        eos_token = getattr(tokenizer, "eos_token", None) or ""
        for action in unique_actions:
            prompt_ids = list(prompt_token_ids)
            target_text = synthesize_sft_target(action) + eos_token
            target_ids = list(tokenizer(target_text, add_special_tokens=False).input_ids)
            if len(prompt_ids) + len(target_ids) > self.max_seq_len:
                if len(target_ids) >= self.max_seq_len:
                    target_ids = target_ids[-self.max_seq_len :]
                    prompt_ids = []
                else:
                    prompt_ids = prompt_ids[-(self.max_seq_len - len(target_ids)) :]
            rows.append(
                {
                    "action": action,
                    "input_ids": prompt_ids + target_ids,
                    "n_prompt": len(prompt_ids),
                }
            )

        device = next(model.parameters()).device
        pad_id = int(getattr(tokenizer, "pad_token_id"))
        max_len = max(len(row["input_ids"]) for row in rows)
        batch_ids: list[list[int]] = []
        batch_attn: list[list[int]] = []
        for row in rows:
            ids = list(row["input_ids"])
            pad = max_len - len(ids)
            batch_ids.append(ids + [pad_id] * pad)
            batch_attn.append([1] * len(ids) + [0] * pad)

        ids_t = torch.tensor(batch_ids, dtype=torch.long, device=device)
        attn_t = torch.tensor(batch_attn, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ids_t, attention_mask=attn_t).logits[:, :-1, :].to(torch.float32)
            labels = ids_t[:, 1:].contiguous()
            token_logp = torch.log_softmax(logits, dim=-1).gather(
                -1, labels.unsqueeze(-1)
            ).squeeze(-1)

        scores: dict[str, float] = {}
        for i, row in enumerate(rows):
            real_len = int(sum(batch_attn[i]))
            label_len = max(0, real_len - 1)
            target_start = max(0, int(row["n_prompt"]) - 1)
            selected = token_logp[i, target_start:label_len]
            total = float(selected.sum().item()) if selected.numel() else float("-inf")
            mean = total / max(1, int(selected.numel()))
            score = mean if self.score_normalization == "mean" else total
            scores[str(row["action"])] = score
        return scores

    def decompose(self, group: TrajectoryGroup) -> TurnRewards:
        device = torch.device(self.device)
        reward_rows: list[Tensor] = []

        for traj in group.trajectories:
            values: list[float] = []
            for turn in traj.turns:
                chosen_action = str(getattr(turn, "action_text", "") or "").strip()
                valid_actions = [
                    str(action).strip()
                    for action in list(getattr(turn, "valid_actions", ()) or ())
                    if str(action).strip()
                ]
                if chosen_action and chosen_action not in valid_actions:
                    valid_actions.insert(0, chosen_action)
                if len(valid_actions) > self.max_actions_to_score:
                    trimmed = valid_actions[: self.max_actions_to_score]
                    if chosen_action and chosen_action not in trimmed:
                        trimmed[-1] = chosen_action
                    valid_actions = list(dict.fromkeys(trimmed))

                score_map = self._score_candidates(
                    prompt_token_ids=list(getattr(turn, "prompt_token_ids", ()) or ()),
                    actions=valid_actions,
                )

                if chosen_action not in score_map:
                    values.append(0.0)
                    continue
                chosen_lp = float(score_map[chosen_action])
                alt_lps = [
                    float(score_map[action])
                    for action in valid_actions
                    if action != chosen_action and action in score_map
                ]
                if not alt_lps:
                    values.append(0.0)
                    continue

                chosen = torch.tensor(float(chosen_lp), device=device)
                alternatives = torch.tensor(alt_lps, device=device)
                margin = chosen - alternatives.max()
                if self.normalize_margin:
                    denom = torch.maximum(
                        torch.maximum(chosen.abs(), alternatives.max().abs()),
                        torch.tensor(1.0, device=device),
                    )
                    margin = margin / denom
                values.append(float(margin.detach().cpu()))

            reward_rows.append(torch.as_tensor(values, device=device, dtype=torch.float32))

        rewards = _to_float_rows(reward_rows)
        _assert_ragged_shape(group, rewards)
        return rewards


@dataclass
class CounterfactualDeltaTODO:
    """Causal turn reward by comparing actual reward to replay alternatives."""

    n_alternatives: int = 2
    counterfactual_runner: Callable[[Trajectory, int, int], list[float]] | None = None
    device: str = "cpu"

    def decompose(self, group: TrajectoryGroup) -> TurnRewards:
        device = torch.device(self.device)
        reward_rows: list[Tensor] = []

        for traj in group.trajectories:
            values: list[float] = []
            for t, _turn in enumerate(traj.turns):
                if self.counterfactual_runner is None:
                    values.append(0.0)
                    continue

                alt_rewards = self.counterfactual_runner(
                    traj,
                    t,
                    int(self.n_alternatives),
                )
                if not alt_rewards:
                    values.append(0.0)
                    continue

                alternatives = torch.as_tensor(
                    alt_rewards,
                    device=device,
                    dtype=torch.float32,
                )
                actual = torch.tensor(float(traj.final_reward), device=device)
                values.append(float((actual - alternatives.mean()).detach().cpu()))

            reward_rows.append(torch.as_tensor(values, device=device, dtype=torch.float32))

        rewards = _to_float_rows(reward_rows)
        _assert_ragged_shape(group, rewards)
        return rewards


TURN_REWARD_METHOD_GRID: dict[str, list[object]] = {
    "method": [
        "progress_delta",
        "signed_attention",
        "admissible_margin",
    ],
    "terminal_bonus": [0.5, 1.0],
    "signed_attention_hidden_size": [64, 128],
    "signed_attention_outcome_scale": [0.5, 1.0],
    "admissible_margin_weight": [0.0, 0.25],
}


POST_MILESTONE_TURN_REWARD_METHODS: dict[str, list[object]] = {
    "method": ["counterfactual_delta"],
    "counterfactual_n_alternatives": [1, 2],
}


BEST_TURN_REWARD_METHOD_AFTER_SWEEP: dict[str, object] = {
    "source_run_name": None,
    "method": None,
    "alpha": None,
    "selection_metric": None,
    "seen_success": None,
    "unseen_success": None,
    "method_hyperparameters": None,
}
