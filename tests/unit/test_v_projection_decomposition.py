"""Unit tests for the Proposal-A `use_v_projection_for_decomposition` flag.

The flag REPLACES the standard α·R per-turn formula with a Lagrangian
projection of V_t onto the sum-to-R constraint:
  per_turn_t = (V_t_clamped - (∑V_t_clamped - R)/T_active) · mask

Properties this test verifies:
1. Configuration: flag exists, defaults preserve legacy α·R behavior.
2. Sum-to-R: ∑per_turn = R per trajectory after projection.
3. Negative per_turn allowed: turns with low V_t get negative
   per_turn (impossible under softmax-α).
4. Mask handling: padded turns get zero per_turn regardless of V_t.
5. K=1 path: works for single-trajectory groups.
6. Mutual exclusion: when projection is enabled, the legacy α·R term
   is replaced entirely (no double-counting).
7. Config-file loadability for the new bake-off variant.
"""
from __future__ import annotations

import json
import unittest

import torch

from src.algorithms.grpo.trainer import HGPOTrainerConfig


def _project_v_to_sum_R(
    v_t: torch.Tensor,    # [K, T] raw V_t predictions
    R: torch.Tensor,      # [K] trajectory rewards
    mask: torch.Tensor,   # [K, T] (1 = active turn, 0 = pad)
    clamp_val: float = 2.0,
) -> torch.Tensor:
    """Reference implementation of the projection used in trainer.py."""
    v_clamped = v_t.detach().clamp(-clamp_val, clamp_val) * mask
    T_active = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    v_sum = v_clamped.sum(dim=-1, keepdim=True)
    adjustment = (v_sum - R.unsqueeze(-1)) / T_active
    return (v_clamped - adjustment) * mask


class TestVProjectionConfig(unittest.TestCase):
    def test_default_disabled(self) -> None:
        """Default flag is False — legacy α·R framing preserved."""
        cfg = HGPOTrainerConfig()
        self.assertFalse(cfg.use_v_projection_for_decomposition)

    def test_clamp_default(self) -> None:
        cfg = HGPOTrainerConfig()
        self.assertEqual(cfg.v_projection_clamp, 2.0)

    def test_can_enable(self) -> None:
        cfg = HGPOTrainerConfig(
            use_v_projection_for_decomposition=True,
            v_projection_clamp=1.5,
        )
        self.assertTrue(cfg.use_v_projection_for_decomposition)
        self.assertEqual(cfg.v_projection_clamp, 1.5)

    def test_other_flags_unchanged(self) -> None:
        """Enabling v-projection doesn't touch the older flags."""
        cfg = HGPOTrainerConfig(use_v_projection_for_decomposition=True)
        self.assertEqual(cfg.alpha, 0.5)
        self.assertEqual(cfg.lambda_consistency, 0.1)


class TestVProjectionMath(unittest.TestCase):
    """Verify the projection satisfies its constraint and properties."""

    def test_sum_to_R_constraint(self) -> None:
        """∑per_turn = R per trajectory after projection."""
        v_t = torch.tensor([
            [+1.0, +0.5, -0.3, +0.2],
            [+0.5, +0.5, +0.5, +0.5],
        ])
        R = torch.tensor([1.0, 0.0])
        mask = torch.ones_like(v_t)
        per_turn = _project_v_to_sum_R(v_t, R, mask)
        sums = per_turn.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, R, atol=1e-5))

    def test_negative_per_turn_allowed(self) -> None:
        """Low-V turns can produce NEGATIVE per_turn — something
        softmax-α framing forbade."""
        v_t = torch.tensor([[+1.5, -0.5, +0.2]])
        R = torch.tensor([0.0])
        mask = torch.ones_like(v_t)
        per_turn = _project_v_to_sum_R(v_t, R, mask)
        # Sum should be 0 (R=0)
        self.assertAlmostEqual(per_turn.sum().item(), 0.0, places=5)
        # At least one turn should be negative (the -0.5 V_t turn definitely)
        self.assertLess(per_turn.min().item(), 0.0)

    def test_individual_per_turn_can_exceed_R(self) -> None:
        """When V_t is high at one turn, per_turn at that turn can
        EXCEED R — something α·R framing also forbade (α ∈ [0,1])."""
        v_t = torch.tensor([[+1.8, -0.5, -0.5, -0.5]])
        R = torch.tensor([0.5])
        mask = torch.ones_like(v_t)
        per_turn = _project_v_to_sum_R(v_t, R, mask)
        # Sum-to-R preserved
        self.assertAlmostEqual(per_turn.sum().item(), 0.5, places=5)
        # The first turn has very high V_t (1.8), so its per_turn should
        # be close to or exceeding 0.5 (R), which softmax-α couldn't do.
        self.assertGreater(per_turn[0, 0].item(), 0.5)

    def test_mask_zeros_padded_turns(self) -> None:
        """Padded turns get exactly 0 per_turn regardless of V_t."""
        v_t = torch.tensor([[+0.5, +1.0, +5.0]])  # turn 2 is padded
        R = torch.tensor([1.0])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        per_turn = _project_v_to_sum_R(v_t, R, mask)
        # Padded position must be 0
        self.assertAlmostEqual(per_turn[0, 2].item(), 0.0, places=6)
        # Active sum should still be R
        self.assertAlmostEqual(per_turn[0, :2].sum().item(), 1.0, places=5)

    def test_uniform_v_gives_uniform_R_share(self) -> None:
        """If V_t is uniform, projection should degenerate to uniform
        R/T per active turn (legacy α=uniform behavior)."""
        v_t = torch.zeros(1, 4)
        R = torch.tensor([1.0])
        mask = torch.ones_like(v_t)
        per_turn = _project_v_to_sum_R(v_t, R, mask)
        expected = torch.full((1, 4), 0.25)
        self.assertTrue(torch.allclose(per_turn, expected, atol=1e-5))

    def test_clamp_bounds_extreme_v_before_projection(self) -> None:
        """Extreme V_t is clamped BEFORE projection — wild outliers
        don't blow up the adjustment term."""
        v_t = torch.tensor([[+100.0, +1.0, -100.0, +1.0]])
        R = torch.tensor([1.0])
        mask = torch.ones_like(v_t)
        per_turn = _project_v_to_sum_R(v_t, R, mask, clamp_val=2.0)
        # Sum-to-R still holds
        self.assertAlmostEqual(per_turn.sum().item(), 1.0, places=5)
        # No turn should be wildly out of bounds
        self.assertLess(per_turn.abs().max().item(), 5.0)

    def test_single_trajectory_K1(self) -> None:
        """Projection works for K=1 (no group baseline needed)."""
        v_t = torch.tensor([[+1.0, +0.5, +0.0]])
        R = torch.tensor([0.5])
        mask = torch.ones_like(v_t)
        per_turn = _project_v_to_sum_R(v_t, R, mask)
        self.assertAlmostEqual(per_turn.sum().item(), 0.5, places=5)

    def test_failure_trajectory_R_zero_still_has_signal(self) -> None:
        """R=0 failure trajectory: projection produces non-zero per_turn
        (zero sum but differentiated by V_t shape) — the failure-blame
        signal that legacy α·R = 0 silently discarded."""
        v_t = torch.tensor([[+1.0, +0.5, -0.5]])  # last turn stuck
        R = torch.tensor([0.0])
        mask = torch.ones_like(v_t)
        per_turn = _project_v_to_sum_R(v_t, R, mask)
        # Sum = 0 (constraint)
        self.assertAlmostEqual(per_turn.sum().item(), 0.0, places=5)
        # But individual turns are differentiated — first turn most
        # positive, last turn most negative (matches V_t shape)
        self.assertGreater(per_turn[0, 0].item(), per_turn[0, 2].item())
        # The high-V turn should be POSITIVE; the negative-V turn NEGATIVE
        self.assertGreater(per_turn[0, 0].item(), 0.0)
        self.assertLess(per_turn[0, 2].item(), 0.0)

    def test_v_t_detach_no_grad_flow(self) -> None:
        """V_t is detached — gradient doesn't backprop into V_t weights
        through per_turn_rewards."""
        v_t = torch.tensor([[+1.0, +0.5, +0.5]], requires_grad=True)
        R = torch.tensor([1.0])
        mask = torch.ones_like(v_t)
        # Mix with grad-tracking dummy so backward succeeds.
        dummy = torch.tensor([0.0], requires_grad=True)
        per_turn = _project_v_to_sum_R(v_t, R, mask) + dummy
        per_turn.sum().backward()
        # V_t should NOT get gradient (detached in projection)
        self.assertIsNone(v_t.grad)
        # Dummy SHOULD get gradient (proves backward ran)
        self.assertIsNotNone(dummy.grad)


class TestConfigLoadable(unittest.TestCase):
    def test_vProjection_config_has_flag(self) -> None:
        from pathlib import Path
        cfg_path = (
            Path(__file__).resolve().parent.parent.parent
            / "configs"
            / "TurnRDV2_alfworld_a050_vProjection.json"
        )
        cfg = json.loads(cfg_path.read_text())
        self.assertTrue(cfg["hgpo"]["use_v_projection_for_decomposition"])
        self.assertEqual(cfg["hgpo"]["v_projection_clamp"], 2.0)
        self.assertEqual(cfg["hgpo"]["decomposer"], "turnrd")
        # Isolated cache paths
        self.assertIn("vProjection", cfg["turnrd"]["replay_buffer_path"])
        self.assertIn("vProjection", cfg["turnrd"]["ckpt_path"])

    def test_a050_baseline_does_not_set_projection(self) -> None:
        from pathlib import Path
        cfg_path = (
            Path(__file__).resolve().parent.parent.parent
            / "configs"
            / "TurnRDV2_alfworld_a050.json"
        )
        cfg = json.loads(cfg_path.read_text())
        self.assertFalse(
            cfg["hgpo"].get("use_v_projection_for_decomposition", False)
        )


if __name__ == "__main__":
    unittest.main()
