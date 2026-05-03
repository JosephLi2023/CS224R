"""Unit tests for src.algorithms.grpo.loss helpers (pure-Python; no torch)."""
from __future__ import annotations

import math
import pytest

from src.algorithms.grpo.loss import (
    clipped_ppo_term,
    importance_ratio,
    kl_k3_per_token,
    kl_per_token,
    mask_mean,
)


def test_importance_ratio_returns_exp_of_diff():
    out = importance_ratio([0.0, 1.0, -1.0], [0.0, 0.5, -0.5])
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(math.exp(0.5))
    assert out[2] == pytest.approx(math.exp(-0.5))


def test_importance_ratio_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        importance_ratio([0.0], [0.0, 0.0])


def test_clipped_ppo_term_below_clip():
    """With ratio inside [1-eps, 1+eps], unclipped == clipped."""
    out = clipped_ppo_term([1.0, 1.05], [1.0, 1.0], clip_eps=0.2)
    # loss = -min(1*1, clamp(1, 0.8, 1.2)*1) = -1
    assert out[0] == pytest.approx(-1.0)
    assert out[1] == pytest.approx(-1.05)


def test_clipped_ppo_term_clips_high_ratio_when_advantage_positive():
    out = clipped_ppo_term([2.0], [1.0], clip_eps=0.2)
    # unclipped = 2*1=2, clipped = 1.2*1=1.2; min = 1.2; loss = -1.2
    assert out[0] == pytest.approx(-1.2)


def test_clipped_ppo_term_keeps_unclipped_when_advantage_negative():
    """Negative advantage flips the inequality; the unclipped (more negative)
    contribution should win the `min`."""
    out = clipped_ppo_term([2.0], [-1.0], clip_eps=0.2)
    # unclipped = -2, clipped = -1.2; min = -2; loss = +2
    assert out[0] == pytest.approx(2.0)


def test_clipped_ppo_term_negative_eps_raises():
    with pytest.raises(ValueError, match="non-negative"):
        clipped_ppo_term([1.0], [1.0], clip_eps=-0.1)


def test_mask_mean_ignores_zero_mask_positions():
    assert mask_mean([1.0, 100.0, 3.0], [1, 0, 1]) == pytest.approx(2.0)


def test_mask_mean_returns_zero_when_no_active_tokens():
    assert mask_mean([1.0, 2.0], [0, 0]) == 0.0


def test_mask_mean_float_positive_values_treat_as_binary():
    """Non-binary masks: any m > 0 activates the position with equal weight."""
    # mean of values at positions [0, 1] (both masks > 0) = (1 + 3) / 2 = 2.0
    assert mask_mean([1.0, 3.0], [1.0, 0.5]) == pytest.approx(2.0)


def test_kl_per_token_is_simple_diff():
    out = kl_per_token([0.5, 0.0], [0.0, 0.5])
    assert out == pytest.approx([0.5, -0.5])


def test_kl_k3_is_nonnegative_and_zero_at_match():
    # k3 = (rho - 1) - log(rho); zero when new == old (rho=1).
    out = kl_k3_per_token([0.0, 0.5, -0.5], [0.0, 0.0, 0.0])
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    # k3 ≥ 0 always (Jensen inequality)
    assert out[1] > 0
    assert out[2] > 0
