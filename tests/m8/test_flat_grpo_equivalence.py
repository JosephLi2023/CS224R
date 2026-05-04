"""M8 verification gate: independent flat-GRPO reference vs trainer helpers."""
from __future__ import annotations

import math
import random

import pytest

from src.algorithms.grpo.advantage import compute_traj_advantages
from src.algorithms.grpo.loss import (
    clipped_ppo_term,
    importance_ratio,
    mask_mean,
)
from tests.m8.flat_grpo_reference import (
    reference_flat_grpo_loss,
    reference_traj_advantages,
)


def _trainer_equivalent_loss(
    *,
    final_rewards,
    per_trajectory_new_logprobs,
    per_trajectory_old_logprobs,
    clip_eps,
):
    advantages = compute_traj_advantages(final_rewards)
    flat_new, flat_old, flat_adv = [], [], []
    for i, (n_lps, o_lps) in enumerate(
        zip(per_trajectory_new_logprobs, per_trajectory_old_logprobs)
    ):
        for n, o in zip(n_lps, o_lps):
            flat_new.append(n)
            flat_old.append(o)
            flat_adv.append(advantages[i])
    if not flat_new:
        return 0.0
    ratios = importance_ratio(flat_new, flat_old)
    per_token = clipped_ppo_term(ratios, flat_adv, clip_eps)
    return mask_mean(per_token, [1] * len(per_token))


def test_advantage_construction_matches_reference():
    """Step 1 of the loss: traj advantages from group of 4 distinct rewards."""
    rewards = [0.0, 1.0, 0.5, -0.5]
    ref = reference_traj_advantages(rewards)
    trn = compute_traj_advantages(rewards)
    assert len(ref) == len(trn) == 4
    for r_v, t_v in zip(ref, trn):
        assert math.isclose(r_v, t_v, abs_tol=1e-12)


def test_degenerate_group_no_nan():
    """All-equal rewards → zero variance → both sides return zero advantages
    (sigma floor prevents NaN)."""
    rewards = [0.7, 0.7, 0.7]
    ref = reference_traj_advantages(rewards)
    trn = compute_traj_advantages(rewards)
    assert all(abs(v) < 1e-6 for v in ref)
    assert all(abs(v) < 1e-6 for v in trn)


def test_loss_zero_when_advantages_zero():
    """Equal rewards → zero advantages → zero loss."""
    rewards = [0.5, 0.5]
    new = [[-0.1, -0.2], [-0.3]]
    old = [[-0.15, -0.18], [-0.31]]
    ref = reference_flat_grpo_loss(
        final_rewards=rewards,
        per_trajectory_new_logprobs=new,
        per_trajectory_old_logprobs=old,
        clip_eps=0.2,
    )
    trn = _trainer_equivalent_loss(
        final_rewards=rewards,
        per_trajectory_new_logprobs=new,
        per_trajectory_old_logprobs=old,
        clip_eps=0.2,
    )
    assert math.isclose(ref, 0.0, abs_tol=1e-9)
    assert math.isclose(trn, 0.0, abs_tol=1e-9)


def test_loss_matches_reference_on_canonical_4_traj_group():
    """K=4 trajectories with distinct rewards + small (new−old) drift."""
    rewards = [0.0, 1.0, 0.5, -0.5]
    new = [[-1.0, -0.8], [-0.6, -0.7, -0.9], [-0.5], [-0.4, -0.5]]
    old = [[-1.05, -0.82], [-0.61, -0.69, -0.91], [-0.49], [-0.42, -0.51]]
    ref = reference_flat_grpo_loss(
        final_rewards=rewards,
        per_trajectory_new_logprobs=new,
        per_trajectory_old_logprobs=old,
        clip_eps=0.2,
    )
    trn = _trainer_equivalent_loss(
        final_rewards=rewards,
        per_trajectory_new_logprobs=new,
        per_trajectory_old_logprobs=old,
        clip_eps=0.2,
    )
    assert math.isclose(ref, trn, rel_tol=1e-10, abs_tol=1e-12), (
        f"reference={ref}, trainer={trn}"
    )


def test_loss_matches_reference_when_clip_active():
    """Force ratio outside the clip band on positive AND negative advantages
    to exercise BOTH branches of the min(unclipped, clipped) selection."""
    rewards = [-1.0, 1.0]
    # First traj: very negative new−old → ratio ≈ exp(−2) ≈ 0.135 (below clip)
    # Second traj: very positive new−old → ratio ≈ exp(+2) ≈ 7.39 (above clip)
    new = [[-3.0], [-1.0]]
    old = [[-1.0], [-3.0]]
    ref = reference_flat_grpo_loss(
        final_rewards=rewards,
        per_trajectory_new_logprobs=new,
        per_trajectory_old_logprobs=old,
        clip_eps=0.2,
    )
    trn = _trainer_equivalent_loss(
        final_rewards=rewards,
        per_trajectory_new_logprobs=new,
        per_trajectory_old_logprobs=old,
        clip_eps=0.2,
    )
    assert math.isclose(ref, trn, rel_tol=1e-10, abs_tol=1e-12), (
        f"reference={ref}, trainer={trn}"
    )


@pytest.mark.parametrize("seed", list(range(50)))
def test_loss_matches_reference_random_fuzz(seed):
    """50 randomly-generated TrajectoryGroups: reference and trainer-helper
    composition must agree to 1e-9 on every one."""
    rng = random.Random(seed)
    K = rng.randint(2, 6)
    rewards = [rng.uniform(-1.0, 1.0) for _ in range(K)]
    new_lps, old_lps = [], []
    for _ in range(K):
        T = rng.randint(1, 6)
        n = [rng.uniform(-3.0, -0.05) for _ in range(T)]
        # Drift between rollout-time (old) and current-policy (new) in [-0.3, +0.3]
        o = [v + rng.uniform(-0.3, 0.3) for v in n]
        new_lps.append(n)
        old_lps.append(o)
    clip_eps = rng.choice([0.1, 0.2, 0.3])
    ref = reference_flat_grpo_loss(
        final_rewards=rewards,
        per_trajectory_new_logprobs=new_lps,
        per_trajectory_old_logprobs=old_lps,
        clip_eps=clip_eps,
    )
    trn = _trainer_equivalent_loss(
        final_rewards=rewards,
        per_trajectory_new_logprobs=new_lps,
        per_trajectory_old_logprobs=old_lps,
        clip_eps=clip_eps,
    )
    assert math.isclose(ref, trn, rel_tol=1e-9, abs_tol=1e-10), (
        f"seed={seed}: reference={ref}, trainer={trn}"
    )
