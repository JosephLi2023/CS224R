"""Torch implementation scaffold for Max's alpha-weighted GRPO loss.

This file stays inside `maxrodriguez/grpo/` while the proposal-specific loss
is still being tested. It is intentionally Torch-based: `new_action_logprobs`
must remain connected to the current policy forward pass so
`total_loss.backward()` can update the policy.

Core proposal math:

    A_traj_i = (R_i - mean_j R_j) / (std_j R_j + eps)

    A_turn_i,t = (r_i,t - mean_j r_j,t) / (std_j r_j,t + eps)

    A_H_i,t = alpha * A_turn_i,t + (1 - alpha) * A_traj_i

    rho = exp(log pi_theta(a|s) - log pi_old(a|s))

    L_policy = -mean(min(rho * A_H, clip(rho, 1-eps, 1+eps) * A_H))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor


RaggedTensorLike = list[list[float] | Tensor]
RaggedTensor = list[Tensor]


@dataclass(frozen=True)
class GRPOLossBatchTODO:
    """Batch shape expected by the Max alpha-GRPO trainer.

    `new_action_logprobs` should be produced by the current model forward pass.
    Everything else can be detached rollout/reference data.
    """

    trajectory_rewards: list[float] | Tensor
    turn_rewards: RaggedTensorLike
    action_token_counts: list[list[int]]
    old_action_logprobs: RaggedTensorLike
    new_action_logprobs: RaggedTensorLike
    ref_action_logprobs: RaggedTensorLike | None = None


@dataclass(frozen=True)
class GRPOLossOutput:
    """Loss tensor plus detached logging stats."""

    total_loss: Tensor
    policy_loss: Tensor
    kl_loss: Tensor
    mean_token_advantage: float
    num_action_tokens: int

    def logs(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total_loss.detach().cpu()),
            "policy_loss": float(self.policy_loss.detach().cpu()),
            "kl_loss": float(self.kl_loss.detach().cpu()),
            "mean_token_advantage": self.mean_token_advantage,
            "num_action_tokens": float(self.num_action_tokens),
        }


def _first_tensor(rows: RaggedTensorLike) -> Tensor | None:
    for row in rows:
        if isinstance(row, Tensor):
            return row
    return None


def _target_device_dtype(
    *ragged: RaggedTensorLike,
) -> tuple[torch.device, torch.dtype]:
    for rows in ragged:
        tensor = _first_tensor(rows)
        if tensor is not None:
            return tensor.device, tensor.dtype
    return torch.device("cpu"), torch.float32


def _as_1d_tensor(
    values: list[float] | Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    detach: bool,
) -> Tensor:
    if isinstance(values, Tensor):
        tensor = values.to(device=device, dtype=dtype).reshape(-1)
        return tensor.detach() if detach else tensor
    return torch.as_tensor(values, device=device, dtype=dtype).reshape(-1)


def _as_ragged_tensors(
    rows: RaggedTensorLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
    detach: bool,
) -> RaggedTensor:
    return [
        _as_1d_tensor(row, device=device, dtype=dtype, detach=detach)
        for row in rows
    ]


def _check_same_ragged_shape(name: str, a: RaggedTensor, b: RaggedTensor) -> None:
    if len(a) != len(b):
        raise ValueError(f"{name}: outer length mismatch: {len(a)} vs {len(b)}")
    for i, (left, right) in enumerate(zip(a, b)):
        if left.numel() != right.numel():
            raise ValueError(
                f"{name}: row {i} length mismatch: "
                f"{left.numel()} vs {right.numel()}"
            )


def _zero_like_reference(rows: RaggedTensor) -> Tensor:
    ref = _first_tensor(rows)
    if ref is None:
        return torch.tensor(0.0)
    return ref.sum() * 0.0


def _cat_nonempty(rows: RaggedTensor) -> Tensor:
    nonempty = [row for row in rows if row.numel() > 0]
    if not nonempty:
        return _zero_like_reference(rows).reshape(0)
    return torch.cat(nonempty, dim=0)


def compute_trajectory_advantages_todo(
    trajectory_rewards: list[float] | Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-8,
) -> Tensor:
    """TODO(trainer-1): group-normalize final trajectory rewards with Torch."""
    if isinstance(trajectory_rewards, Tensor):
        rewards = trajectory_rewards.to(
            device=device or trajectory_rewards.device,
            dtype=dtype,
        ).reshape(-1).detach()
    else:
        rewards = torch.as_tensor(
            trajectory_rewards,
            device=device or torch.device("cpu"),
            dtype=dtype,
        ).reshape(-1)

    if rewards.numel() == 0:
        return rewards

    std = rewards.std(unbiased=False)
    if bool(std <= eps):
        return torch.zeros_like(rewards)

    return (rewards - rewards.mean()) / (std + eps)


def compute_turn_advantages_todo(
    turn_rewards: RaggedTensorLike,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-8,
) -> RaggedTensor:
    """TODO(trainer-2): normalize per-turn rewards without padded fake turns."""
    if device is None:
        inferred_device, inferred_dtype = _target_device_dtype(turn_rewards)
        device = inferred_device
        dtype = inferred_dtype

    rows = _as_ragged_tensors(
        turn_rewards,
        device=device,
        dtype=dtype,
        detach=True,
    )
    max_turns = max((row.numel() for row in rows), default=0)
    out = [torch.zeros_like(row) for row in rows]

    for t in range(max_turns):
        present = [row[t] for row in rows if row.numel() > t]
        if not present:
            continue
        values = torch.stack(present)
        std = values.std(unbiased=False)
        if bool(std <= eps):
            normalized = torch.zeros_like(values)
        else:
            normalized = (values - values.mean()) / (std + eps)

        j = 0
        for i, row in enumerate(rows):
            if row.numel() > t:
                out[i][t] = normalized[j]
                j += 1

    return out


def combine_alpha_advantages_todo(
    trajectory_advantages: Tensor,
    turn_advantages: RaggedTensor,
    alpha: float,
) -> RaggedTensor:
    """TODO(trainer-3): apply A_H = alpha*A_turn + (1-alpha)*A_traj."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if trajectory_advantages.numel() != len(turn_advantages):
        raise ValueError(
            "trajectory_advantages length must match number of trajectories"
        )

    return [
        float(alpha) * row + (1.0 - float(alpha)) * trajectory_advantages[i]
        for i, row in enumerate(turn_advantages)
    ]


def expand_turn_advantages_to_action_tokens_todo(
    alpha_advantages: RaggedTensor,
    action_token_counts: list[list[int]],
) -> RaggedTensor:
    """TODO(trainer-4): repeat each turn advantage over its action tokens."""
    if len(alpha_advantages) != len(action_token_counts):
        raise ValueError("alpha_advantages and action_token_counts lengths differ")

    token_rows: RaggedTensor = []
    for i, (adv_row, count_row) in enumerate(zip(alpha_advantages, action_token_counts)):
        if adv_row.numel() != len(count_row):
            raise ValueError(
                f"row {i}: advantage turns and token counts differ: "
                f"{adv_row.numel()} vs {len(count_row)}"
            )
        pieces: list[Tensor] = []
        for turn_adv, n_tokens in zip(adv_row, count_row):
            if n_tokens < 0:
                raise ValueError(f"row {i}: token count cannot be negative")
            if n_tokens:
                pieces.append(turn_adv.repeat(int(n_tokens)))
        token_rows.append(
            torch.cat(pieces) if pieces else adv_row.new_zeros((0,))
        )

    return token_rows


def compute_clipped_grpo_policy_loss_todo(
    new_action_logprobs: RaggedTensorLike,
    old_action_logprobs: RaggedTensorLike,
    token_advantages: RaggedTensor,
    clip_eps: float,
) -> Tensor:
    """TODO(loss-1): clipped GRPO/PPO policy loss over real action tokens."""
    if clip_eps < 0:
        raise ValueError(f"clip_eps must be nonnegative, got {clip_eps}")

    device, dtype = _target_device_dtype(new_action_logprobs)
    new_rows = _as_ragged_tensors(
        new_action_logprobs,
        device=device,
        dtype=dtype,
        detach=False,
    )
    old_rows = _as_ragged_tensors(
        old_action_logprobs,
        device=device,
        dtype=dtype,
        detach=True,
    )
    adv_rows = [
        row.to(device=device, dtype=dtype).detach()
        for row in token_advantages
    ]

    _check_same_ragged_shape("new vs old logprobs", new_rows, old_rows)
    _check_same_ragged_shape("new logprobs vs token advantages", new_rows, adv_rows)

    new_flat = _cat_nonempty(new_rows)
    old_flat = _cat_nonempty(old_rows)
    adv_flat = _cat_nonempty(adv_rows)
    if new_flat.numel() == 0:
        return _zero_like_reference(new_rows)

    ratio = torch.exp(new_flat - old_flat)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    unclipped = ratio * adv_flat
    clipped = clipped_ratio * adv_flat
    return -torch.minimum(unclipped, clipped).mean()


def compute_reference_kl_penalty_todo(
    new_action_logprobs: RaggedTensorLike,
    ref_action_logprobs: RaggedTensorLike,
    kl_coeff: float,
) -> Tensor:
    """TODO(loss-2): lightweight action-token KL against the reference policy."""
    if kl_coeff < 0:
        raise ValueError(f"kl_coeff must be nonnegative, got {kl_coeff}")

    device, dtype = _target_device_dtype(new_action_logprobs)
    new_rows = _as_ragged_tensors(
        new_action_logprobs,
        device=device,
        dtype=dtype,
        detach=False,
    )
    ref_rows = _as_ragged_tensors(
        ref_action_logprobs,
        device=device,
        dtype=dtype,
        detach=True,
    )
    _check_same_ragged_shape("new vs ref logprobs", new_rows, ref_rows)

    new_flat = _cat_nonempty(new_rows)
    ref_flat = _cat_nonempty(ref_rows)
    if new_flat.numel() == 0:
        return _zero_like_reference(new_rows)

    diff = new_flat - ref_flat
    ratio = torch.exp(diff)
    kl_k3 = (ratio - 1.0) - diff
    return float(kl_coeff) * kl_k3.mean()


def compute_total_grpo_loss_todo(
    batch: GRPOLossBatchTODO,
    alpha: float,
    clip_eps: float,
    kl_coeff: float,
) -> GRPOLossOutput:
    """TODO(loss-3): full differentiable alpha-GRPO loss path."""
    device, dtype = _target_device_dtype(batch.new_action_logprobs)

    trajectory_advantages = compute_trajectory_advantages_todo(
        batch.trajectory_rewards,
        device=device,
        dtype=dtype,
    )
    turn_advantages = compute_turn_advantages_todo(
        batch.turn_rewards,
        device=device,
        dtype=dtype,
    )
    alpha_advantages = combine_alpha_advantages_todo(
        trajectory_advantages=trajectory_advantages,
        turn_advantages=turn_advantages,
        alpha=alpha,
    )
    token_advantages = expand_turn_advantages_to_action_tokens_todo(
        alpha_advantages=alpha_advantages,
        action_token_counts=batch.action_token_counts,
    )

    policy_loss = compute_clipped_grpo_policy_loss_todo(
        new_action_logprobs=batch.new_action_logprobs,
        old_action_logprobs=batch.old_action_logprobs,
        token_advantages=token_advantages,
        clip_eps=clip_eps,
    )

    if batch.ref_action_logprobs is None or kl_coeff == 0.0:
        kl_loss = policy_loss.detach() * 0.0
    else:
        kl_loss = compute_reference_kl_penalty_todo(
            new_action_logprobs=batch.new_action_logprobs,
            ref_action_logprobs=batch.ref_action_logprobs,
            kl_coeff=kl_coeff,
        )

    total_loss = policy_loss + kl_loss
    flat_adv = _cat_nonempty(token_advantages)
    mean_adv = float(flat_adv.detach().mean().cpu()) if flat_adv.numel() else 0.0

    return GRPOLossOutput(
        total_loss=total_loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        mean_token_advantage=mean_adv,
        num_action_tokens=int(flat_adv.numel()),
    )


class MaxAlphaGRPOTrainerTODO:
    """Thin trainer shell that wires rollout collection to the Torch loss."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        rollout_collector: Callable[[dict[str, Any]], Any] | None = None,
        turn_reward_scorer: Callable[[Any], RaggedTensorLike] | None = None,
        batch_builder: Callable[[Any, RaggedTensorLike], GRPOLossBatchTODO] | None = None,
        optimizer_step: Callable[[Tensor], None] | None = None,
    ) -> None:
        self.config = config
        self.rollout_collector = rollout_collector
        self.turn_reward_scorer = turn_reward_scorer
        self.batch_builder = batch_builder
        self.optimizer_step = optimizer_step

    def collect_rollout_group_todo(self) -> Any:
        """TODO(trainer-5): plug in the ALFWorld K-trajectory collector."""
        if self.rollout_collector is None:
            raise RuntimeError("pass rollout_collector=... to MaxAlphaGRPOTrainerTODO")
        return self.rollout_collector(self.config)

    def score_turn_rewards_todo(self, rollout_group: Any) -> RaggedTensorLike:
        """TODO(trainer-6): call the selected turn-level reward method."""
        if self.turn_reward_scorer is None:
            raise RuntimeError("pass turn_reward_scorer=... to MaxAlphaGRPOTrainerTODO")
        return self.turn_reward_scorer(rollout_group)

    def train_step_todo(self) -> GRPOLossOutput:
        """TODO(trainer-7): collect, score, build Torch batch, and step."""
        if self.batch_builder is None:
            raise RuntimeError("pass batch_builder=... to MaxAlphaGRPOTrainerTODO")

        rollout_group = self.collect_rollout_group_todo()
        turn_rewards = self.score_turn_rewards_todo(rollout_group)
        batch = self.batch_builder(rollout_group, turn_rewards)

        train_cfg = self.config.get("train", {})
        hgpo_cfg = self.config.get("hgpo", {})
        output = compute_total_grpo_loss_todo(
            batch=batch,
            alpha=float(hgpo_cfg.get("alpha", 0.5)),
            clip_eps=float(train_cfg.get("clip_eps", 0.2)),
            kl_coeff=float(train_cfg.get("kl_coeff", 0.0)),
        )

        if self.optimizer_step is not None:
            self.optimizer_step(output.total_loss)

        return output
