"""H-GRPO advantage math (proposal §3.1).

All functions are pure-Python (operate on `list[float]` / `list[list[float]]`)
so they're trivially unit-testable without torch and so the same code paths
run in numerical-correctness tests AND in the trainer (which adapts via
`torch.tensor(_).tolist()` at the boundary). When perf becomes the
bottleneck we'll add a torch-native path; until then clarity wins.

Notation:
- K = number of trajectories per task (proposal default: K=4).
- For trajectory i ∈ {0..K-1}, R_i is the final scalar reward.
- For turn t in trajectory i, r̂_t^i is the per-turn reward produced by a
  decomposer (Method A judge / Method B TurnRD / Method C progress).
- σ_floor (=1e-8) is added inside std() to keep gradients finite when a
  group is degenerate (all rewards equal).

Formulas (proposal §3.1):

    Â_traj(τ_i)        = (R_i − R̄) / σ_R
    Â_turn(t, τ_i)     = (r̂_t^i − r̄_t) / σ_{r̂_t}        per position t
    Â_H(t, τ_i)        = α · Â_traj(τ_i) + (1 − α) · Â_turn(t, τ_i)
    L_consistency(τ_i) = λ · ‖ Σ_t Â_turn(t, τ_i) − Â_traj(τ_i) ‖²

Setting α=1 and λ=0 reduces H-GRPO exactly to flat GRPO; this is unit-tested.
"""

from __future__ import annotations

import math

SIGMA_FLOOR = 1e-8


# ---------- low-level helpers ----------


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


# ---------- trajectory-level advantage ----------


def compute_traj_advantages(final_rewards: list[float]) -> list[float]:
    """Standard GRPO trajectory-level group-normalized advantage.

    Args:
        final_rewards: length-K list of scalar trajectory rewards R_i.

    Returns:
        Length-K list of `Â_traj(τ_i) = (R_i − R̄) / σ_R`.
    """
    if not final_rewards:
        return []
    mean = _mean(final_rewards)
    std = _std(final_rewards, mean)
    return [(r - mean) / std for r in final_rewards]


# ---------- turn-level advantage ----------


def compute_turn_advantages(per_turn_rewards: list[list[float]]) -> list[list[float]]:
    """Position-wise group-normalized turn advantage.

    For each absolute turn position t, normalize r̂_t^i across the K
    trajectories that have a turn at position t (uneven trajectory lengths
    are handled gracefully — turns missing at position t are skipped in the
    mean/std but the advantage entry is still 0.0 for that trajectory).

    Args:
        per_turn_rewards: list[K] of list[T_i] of decomposed per-turn rewards
                          r̂_t^i. T_i may differ across trajectories.

    Returns:
        list[K] of list[T_i], same shape as input, with each entry replaced
        by `Â_turn(t, τ_i) = (r̂_t^i − r̄_t) / σ_{r̂_t}`.
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


# ---------- combination + consistency ----------


def combine(
    alpha: float,
    traj_advantages: list[float],
    turn_advantages: list[list[float]],
) -> list[list[float]]:
    """Combine trajectory- and turn-level advantages per proposal §3.1.

    Â_H(t, τ_i) = α · Â_traj(τ_i) + (1 − α) · Â_turn(t, τ_i)

    Args:
        alpha: weight on trajectory-level advantage in [0, 1].
               α=1 ⇒ flat GRPO (turn signal dropped).
               α=0 ⇒ pure turn-level signal (no group baseline).
        traj_advantages: length-K list of `Â_traj(τ_i)`.
        turn_advantages: list[K] of list[T_i] of `Â_turn(t, τ_i)`.

    Returns:
        list[K] of list[T_i] of combined per-turn advantages, ready to be
        broadcast over action tokens for the PPO surrogate loss.
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
    """L_consistency = λ · mean_i ‖ Σ_t Â_turn(t, τ_i) − Â_traj(τ_i) ‖²

    Aligns the turn-level signal scale with the trajectory-level scale.
    Returns 0 exactly when, for every i, Σ_t Â_turn(t, τ_i) = Â_traj(τ_i).

    Args:
        lam: weight on the regularizer (0 disables it).
        traj_advantages: length-K list of `Â_traj(τ_i)`.
        turn_advantages: list[K] of list[T_i] of `Â_turn(t, τ_i)`.

    Returns:
        Scalar loss (mean over the K trajectories of the squared deviation).
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
