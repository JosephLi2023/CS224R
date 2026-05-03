"""Unit tests for H-GRPO advantage math (proposal §3.1).

Includes the verification-gate test from the plan: with α=1 and λ=0, the
combined advantage and loss must reduce exactly to flat GRPO.
"""

from __future__ import annotations

import math

import pytest

from src.algorithms.grpo.advantage import (
    SIGMA_FLOOR,
    combine,
    compute_traj_advantages,
    compute_turn_advantages,
    consistency_loss,
)
from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord


# ---------- compute_traj_advantages ----------


def test_traj_advantages_zero_mean_unit_std() -> None:
    """Group-normalized advantages must have mean ≈ 0 and std ≈ 1."""
    final_rewards = [0.2, 0.5, 0.9, 0.4]
    a = compute_traj_advantages(final_rewards)
    assert pytest.approx(sum(a) / len(a), abs=1e-9) == 0.0
    var = sum(x * x for x in a) / len(a)
    assert math.isclose(math.sqrt(var), 1.0, abs_tol=1e-6)


def test_traj_advantages_degenerate_group_no_nan() -> None:
    """All-equal rewards should produce all-zero advantages (not NaN)."""
    a = compute_traj_advantages([0.5, 0.5, 0.5, 0.5])
    assert all(abs(x) < 1e-3 for x in a)
    assert all(not math.isnan(x) for x in a)


def test_traj_advantages_empty_input() -> None:
    assert compute_traj_advantages([]) == []


def test_traj_advantages_two_traj_known_value() -> None:
    """Hand-computed: rewards [0.0, 1.0] → mean=0.5, std≈0.5
    → advantages ≈ [-1.0, +1.0]."""
    a = compute_traj_advantages([0.0, 1.0])
    assert math.isclose(a[0], -1.0, abs_tol=1e-4)
    assert math.isclose(a[1], +1.0, abs_tol=1e-4)


# ---------- compute_turn_advantages ----------


def test_turn_advantages_uniform_shape() -> None:
    """Even trajectories: per-position normalization yields zero-mean rows."""
    # K=3 trajectories, T=4 turns each.
    per_turn = [
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
        [0.2, 0.2, 0.2, 0.2],
    ]
    turn_adv = compute_turn_advantages(per_turn)
    assert len(turn_adv) == 3
    assert all(len(row) == 4 for row in turn_adv)
    # Per-position means across K trajectories should be ~0.
    for t in range(4):
        col = [turn_adv[i][t] for i in range(3)]
        assert pytest.approx(sum(col) / 3, abs=1e-9) == 0.0


def test_turn_advantages_uneven_lengths() -> None:
    """Variable trajectory lengths should still produce well-formed output."""
    per_turn = [
        [0.1, 0.2, 0.3],   # len 3
        [0.4, 0.3],        # len 2
        [0.2, 0.5, 0.7, 0.9],  # len 4
    ]
    turn_adv = compute_turn_advantages(per_turn)
    assert [len(r) for r in turn_adv] == [3, 2, 4]
    # Position 3 only has 1 trajectory → its advantage is 0 (mean of 1).
    assert math.isclose(turn_adv[2][3], 0.0, abs_tol=1e-3)


def test_turn_advantages_empty_input() -> None:
    assert compute_turn_advantages([]) == []
    assert compute_turn_advantages([[], []]) == [[], []]


def test_turn_advantages_degenerate_position_no_nan() -> None:
    """Position where all K rewards are equal → advantages are 0 not NaN."""
    per_turn = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    turn_adv = compute_turn_advantages(per_turn)
    for row in turn_adv:
        for x in row:
            assert not math.isnan(x)
            assert abs(x) < 1e-3


# ---------- combine ----------


def test_combine_alpha_one_drops_turn_signal() -> None:
    """α=1 ⇒ combined = Â_traj broadcast across all turns of the trajectory."""
    traj_a = [0.7, -0.7]
    turn_a = [[1.0, -2.0, 3.0], [-1.0, 2.0]]
    out = combine(alpha=1.0, traj_advantages=traj_a, turn_advantages=turn_a)
    assert out[0] == [pytest.approx(0.7)] * 3
    assert out[1] == [pytest.approx(-0.7)] * 2


def test_combine_alpha_zero_drops_traj_signal() -> None:
    """α=0 ⇒ combined = Â_turn unchanged."""
    traj_a = [99.0, -99.0]
    turn_a = [[0.5, -0.5], [1.0, -1.0]]
    out = combine(alpha=0.0, traj_advantages=traj_a, turn_advantages=turn_a)
    for row, expected in zip(out, turn_a):
        assert row == pytest.approx(expected)


def test_combine_alpha_half_blends() -> None:
    out = combine(0.5, [1.0], [[2.0, 4.0]])
    # 0.5 * 1.0 + 0.5 * 2.0 = 1.5; 0.5 * 1.0 + 0.5 * 4.0 = 2.5
    assert out == [[pytest.approx(1.5), pytest.approx(2.5)]]


def test_combine_alpha_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="alpha must be"):
        combine(-0.1, [0.0], [[0.0]])
    with pytest.raises(ValueError, match="alpha must be"):
        combine(1.1, [0.0], [[0.0]])


def test_combine_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        combine(0.5, [0.0, 0.0], [[0.0]])


# ---------- consistency_loss ----------


def test_consistency_loss_zero_when_turn_sum_matches_traj() -> None:
    """If Σ_t Â_turn(t, τ_i) == Â_traj(τ_i) for every i, loss = 0."""
    traj_a = [0.6, -0.6]
    turn_a = [[0.2, 0.4], [-0.3, -0.3]]  # sums match traj_a
    assert consistency_loss(lam=1.0, traj_advantages=traj_a, turn_advantages=turn_a) == pytest.approx(0.0, abs=1e-12)


def test_consistency_loss_positive_when_perturbed() -> None:
    traj_a = [0.6, -0.6]
    turn_a = [[0.2, 0.4], [-0.3, -0.4]]  # i=1 sums to -0.7 != -0.6
    loss = consistency_loss(lam=2.0, traj_advantages=traj_a, turn_advantages=turn_a)
    # i=0 contributes 0; i=1 contributes (-0.7 - -0.6)^2 = 0.01
    # mean over K=2 is 0.005; lam=2 → 0.01
    assert math.isclose(loss, 0.01, abs_tol=1e-9)


def test_consistency_loss_lam_zero_short_circuits() -> None:
    """λ=0 should bypass even the length check (trainer's hot path optimization)."""
    assert consistency_loss(0.0, [], []) == 0.0
    assert consistency_loss(0.0, [1.0, 2.0], [[1.0]]) == 0.0  # mismatch is OK when lam=0


def test_consistency_loss_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        consistency_loss(1.0, [0.0, 0.0], [[0.0]])


# ---------- alpha=1, lambda=0 reduces to flat GRPO (verification gate #1) ----------


def test_alpha_one_lambda_zero_reduces_to_flat_grpo() -> None:
    """Verification gate #1 from the plan: with α=1 and λ=0, the H-GRPO
    combined advantage equals the trajectory-level advantage broadcast over
    all turns, and the consistency loss vanishes. This is the bit-exact
    flat-GRPO recovery property required by proposal §3.1."""
    final_rewards = [0.1, 0.5, 0.9, 0.3]  # K=4
    per_turn = [
        [0.0, 0.05, 0.05],
        [0.1, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.05, 0.1, 0.15],
    ]

    traj_a = compute_traj_advantages(final_rewards)
    turn_a = compute_turn_advantages(per_turn)

    combined_hgpo = combine(alpha=1.0, traj_advantages=traj_a, turn_advantages=turn_a)
    cons = consistency_loss(lam=0.0, traj_advantages=traj_a, turn_advantages=turn_a)

    # Flat GRPO uses traj_a broadcast to every turn.
    flat_grpo = [[traj_a[i] for _ in per_turn[i]] for i in range(len(traj_a))]

    assert cons == 0.0
    for row_h, row_f in zip(combined_hgpo, flat_grpo):
        assert row_h == pytest.approx(row_f, abs=1e-12)


# ---------- TrajectoryGroup integration sanity ----------


def _toy_group() -> TrajectoryGroup:
    """K=3 trajectory group on a fake WebShop task."""
    def _t(rewards: list[float], R: float, tid: str) -> Trajectory:
        return Trajectory(
            task_id="task-1",
            env_name="webshop",
            turns=[
                TurnRecord(turn_idx=i, observation_text=f"o-{tid}-{i}",
                           action_text=f"a-{tid}-{i}", raw_env_reward=r)
                for i, r in enumerate(rewards)
            ],
            final_reward=R,
        )

    return TrajectoryGroup(
        task_id="task-1",
        env_name="webshop",
        trajectories=[
            _t([0.0, 0.1, 0.2], R=0.3, tid="A"),
            _t([0.0, 0.0, 0.5], R=0.5, tid="B"),
            _t([0.1, 0.1, 0.1], R=0.3, tid="C"),
        ],
    )


def test_integration_trajectory_group_through_advantage_pipeline() -> None:
    g = _toy_group()
    assert g.K == 3
    assert g.max_turns == 3
    rewards = g.final_rewards()
    per_turn = g.per_turn_rewards()

    traj_a = compute_traj_advantages(rewards)
    turn_a = compute_turn_advantages(per_turn)
    combined = combine(0.5, traj_a, turn_a)

    assert len(combined) == 3
    assert all(len(combined[i]) == g.trajectories[i].n_turns for i in range(3))


def test_trajectory_group_rejects_mismatched_task_ids() -> None:
    bad = Trajectory(task_id="task-other", env_name="webshop",
                     turns=[TurnRecord(0, "o", "a", 0.0)], final_reward=0.0)
    with pytest.raises(ValueError, match="task_id"):
        TrajectoryGroup(task_id="task-1", env_name="webshop", trajectories=[bad])


def test_trajectory_group_rejects_mismatched_env_names() -> None:
    bad = Trajectory(task_id="task-1", env_name="alfworld",
                     turns=[TurnRecord(0, "o", "a", 0.0)], final_reward=0.0)
    with pytest.raises(ValueError, match="env_name"):
        TrajectoryGroup(task_id="task-1", env_name="webshop", trajectories=[bad])


# ---------- numerical hygiene ----------


def test_sigma_floor_exposed() -> None:
    """Trainer reads SIGMA_FLOOR for logging/debugging — keep public."""
    assert SIGMA_FLOOR > 0
    assert SIGMA_FLOOR < 1e-3
