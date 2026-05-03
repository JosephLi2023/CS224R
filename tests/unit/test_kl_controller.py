"""Unit tests for AdaptiveKLController."""
from __future__ import annotations

import pytest

from src.algorithms.grpo.kl import AdaptiveKLConfig, AdaptiveKLController


def _ctrl(**kw) -> AdaptiveKLController:
    return AdaptiveKLController(AdaptiveKLConfig(**kw))


def test_init_coef_set_from_config():
    c = _ctrl(init_coef=0.1, target_kl=0.04)
    assert c.coef == pytest.approx(0.1)


def test_high_observed_kl_increases_coef():
    c = _ctrl(init_coef=0.04, target_kl=0.04, increase_factor=1.5)
    c.update(observed_kl=0.10)  # > 1.5 * 0.04 = 0.06
    assert c.coef == pytest.approx(0.04 * 1.5)


def test_low_observed_kl_decreases_coef():
    c = _ctrl(init_coef=0.04, target_kl=0.04, decrease_factor=0.5)
    c.update(observed_kl=0.005)  # < 0.5 * 0.04 = 0.02
    assert c.coef == pytest.approx(0.04 * 0.5)


def test_in_band_observed_leaves_coef_unchanged():
    c = _ctrl(init_coef=0.04, target_kl=0.04)
    c.update(observed_kl=0.04)  # in [0.02, 0.06]
    assert c.coef == pytest.approx(0.04)


def test_min_max_clipping():
    c = _ctrl(init_coef=0.5, target_kl=0.04, max_coef=0.6, increase_factor=10.0)
    c.update(observed_kl=1.0)  # would push to 5.0; clipped to 0.6
    assert c.coef == pytest.approx(0.6)

    c2 = _ctrl(init_coef=0.001, target_kl=0.04, min_coef=0.0005, decrease_factor=0.01)
    c2.update(observed_kl=0.0)  # would push to 1e-5; clipped to 5e-4
    assert c2.coef == pytest.approx(0.0005)


def test_negative_observed_kl_clamped_to_zero_then_decreases():
    c = _ctrl(init_coef=0.04, target_kl=0.04)
    c.update(observed_kl=-0.01)  # noisy negative; treated as 0 < 0.02
    assert c.coef < 0.04
    assert c.last_observed == 0.0  # clamped


def test_state_dict_round_trip():
    c = _ctrl()
    c.update(0.10)
    s = c.state_dict()
    c2 = _ctrl()
    c2.load_state_dict(s)
    assert c2.coef == c.coef
    assert c2.last_observed == c.last_observed
    assert c2.steps == c.steps


def test_steps_counter_increments():
    c = _ctrl()
    assert c.steps == 0
    c.update(0.04)
    c.update(0.04)
    assert c.steps == 2
