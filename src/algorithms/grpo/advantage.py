"""H-GRPO advantage math.

Pure-Python (list[float] / list[list[float]]) so it's unit-testable without
torch and shared between the correctness tests and the trainer.

Notation:
- K = number of trajectories per task (default K=4).
- For trajectory i, R_i is the final scalar reward.
- For turn t in trajectory i, r_hat_t^i is the per-turn reward from a
  decomposer (Method A judge / Method B TurnRD / Method C progress).
- sigma_floor (=1e-8) is added inside std() to keep gradients finite when a
  group is degenerate (all rewards equal).

Formulas:
    A_traj(tau_i)        = (R_i - R_bar) / sigma_R
    A_turn(t, tau_i)     = (r_hat_t^i - r_bar_t) / sigma_{r_hat_t}   per position t
    A_H(t, tau_i)        = alpha * A_traj + (1 - alpha) * A_turn
    L_consistency(tau_i) = lambda * norm(sum_t A_turn(t, tau_i) - A_traj(tau_i))^2

alpha=1, lambda=0 reduces H-GRPO to flat GRPO; unit-tested.
"""

from __future__ import annotations

import math

SIGMA_FLOOR = 1e-8


def _mean(xs: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    return sum(xs) / n


def _std(xs: list[float], mean: float | None = None) -> float:
    """Population std (divide by N), with a small floor to avoid div-by-zero."""
    n = len(xs)
    if n == 0:
        return SIGMA_FLOOR
    m = mean if mean is not None else _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / n
    return math.sqrt(var + SIGMA_FLOOR**2)


def compute_traj_advantages(final_rewards: list[float]) -> list[float]:
    """Group-normalized trajectory advantage: (R_i - R_bar) / sigma_R."""
    if not final_rewards:
        return []
    mean = _mean(final_rewards)
    std = _std(final_rewards, mean)
    return [(r - mean) / std for r in final_rewards]


def compute_turn_advantages(per_turn_rewards: list[list[float]]) -> list[list[float]]:
    """Position-wise group-normalized turn advantage.

    For each absolute position t, normalize r_hat_t^i across the K
    trajectories that reach position t. Uneven lengths are fine; missing
    positions stay 0.0.
    """
    K = len(per_turn_rewards)
    if K == 0:
        return []

    max_t = max((len(traj) for traj in per_turn_rewards), default=0)
    if max_t == 0:
        return [[] for _ in per_turn_rewards]

    # Per-position mean/std across the K trajectories that reach position t.
    pos_mean: list[float] = []
    pos_std: list[float] = []
    for t in range(max_t):
        col = [traj[t] for traj in per_turn_rewards if t < len(traj)]
        if not col:
            pos_mean.append(0.0)
            pos_std.append(SIGMA_FLOOR)
            continue
        m = _mean(col)
        pos_mean.append(m)
        pos_std.append(_std(col, m))

    out: list[list[float]] = []
    for traj in per_turn_rewards:
        row = [(traj[t] - pos_mean[t]) / pos_std[t] for t in range(len(traj))]
        out.append(row)
    return out


def combine(
    alpha: float,
    traj_advantages: list[float],
    turn_advantages: list[list[float]],
) -> list[list[float]]:
    """Combine advantages: A_H = alpha * A_traj + (1 - alpha) * A_turn.

    alpha=1 -> flat GRPO; alpha=0 -> pure turn-level signal.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if len(traj_advantages) != len(turn_advantages):
        raise ValueError(
            f"length mismatch: traj_advantages={len(traj_advantages)} "
            f"turn_advantages={len(turn_advantages)}"
        )

    out: list[list[float]] = []
    for traj_a, turn_row in zip(traj_advantages, turn_advantages):
        out.append([alpha * traj_a + (1.0 - alpha) * t for t in turn_row])
    return out


def consistency_loss(
    lam: float,
    traj_advantages: list[float],
    turn_advantages: list[list[float]],
) -> float:
    """L_consistency = lambda * mean_i (sum_t A_turn(t, tau_i) - A_traj(tau_i))^2.

    Aligns turn- and trajectory-level signal scales; 0 when they match.
    """
    if lam == 0.0:
        return 0.0
    if len(traj_advantages) != len(turn_advantages):
        raise ValueError(
            f"length mismatch: traj_advantages={len(traj_advantages)} "
            f"turn_advantages={len(turn_advantages)}"
        )
    K = len(traj_advantages)
    if K == 0:
        return 0.0
    sq = 0.0
    for traj_a, turn_row in zip(traj_advantages, turn_advantages):
        delta = sum(turn_row) - traj_a
        sq += delta * delta
    return lam * (sq / K)


def consistency_loss_tensor(
    lam: float,
    traj_adv,
    turn_adv,
    attention_mask,
):
    """Torch-tensor twin of `consistency_loss` so gradients reach a learnable
    decomposer (Method B/TurnRD). Returns a scalar tensor
    lam * mean_i (sum_t A_turn * m_t - A_traj)^2. Only called when
    decomposer.has_learnable_params is True.

    attention_mask is [B, T] (1 = real, 0 = padded).
    """
    # Local import: keep module load torch-free for pure-Python tests.
    import torch  # type: ignore[import-not-found]

    if traj_adv.dim() != 1:
        raise ValueError(
            f"consistency_loss_tensor: traj_adv must be [B], got shape "
            f"{tuple(traj_adv.shape)}."
        )
    if turn_adv.dim() != 2:
        raise ValueError(
            f"consistency_loss_tensor: turn_adv must be [B, T], got shape "
            f"{tuple(turn_adv.shape)}."
        )
    if attention_mask.dim() != 2:
        raise ValueError(
            f"consistency_loss_tensor: attention_mask must be [B, T], got shape "
            f"{tuple(attention_mask.shape)}."
        )
    if traj_adv.shape[0] != turn_adv.shape[0]:
        raise ValueError(
            f"consistency_loss_tensor: B mismatch: traj_adv={traj_adv.shape[0]}, "
            f"turn_adv={turn_adv.shape[0]}."
        )
    if turn_adv.shape != attention_mask.shape:
        raise ValueError(
            f"consistency_loss_tensor: turn_adv shape {tuple(turn_adv.shape)} "
            f"!= attention_mask shape {tuple(attention_mask.shape)}."
        )

    if turn_adv.shape[0] == 0:
        # Empty batch: keep the result dependent on traj_adv so .backward() works.
        return lam * (traj_adv.sum() * 0.0)

    mask_f = attention_mask.to(dtype=turn_adv.dtype)
    summed_per_traj = (turn_adv * mask_f).sum(dim=-1)  # [B]
    diff = (summed_per_traj - traj_adv.to(dtype=turn_adv.dtype)) ** 2  # [B]
    # Cast lam via tensor so the result keeps turn_adv.dtype (not float64).
    return torch.as_tensor(lam, dtype=turn_adv.dtype, device=turn_adv.device) * diff.mean()
