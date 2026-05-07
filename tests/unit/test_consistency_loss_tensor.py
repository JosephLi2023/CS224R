"""Unit tests for `src.algorithms.grpo.advantage.consistency_loss_tensor`.

Twin of the pure-Python `consistency_loss`; verifies:
1. Numerical equivalence on identical inputs (within 1e-6).
2. Backward through the tensor variant populates α_t.grad.

Skipped cleanly on hosts without torch.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.algorithms.grpo.advantage import (  # noqa: E402
    consistency_loss,
    consistency_loss_tensor,
)


def test_consistency_loss_tensor_matches_pure_python() -> None:
    """Same inputs ⇒ same output (within fp32 noise)."""
    lam = 0.137
    traj_adv_list = [0.5, -0.2, 0.7, 0.0]
    turn_adv_list = [
        [0.1, 0.2, 0.2],     # sums to 0.5 (matches traj_adv → 0 contribution)
        [0.05, 0.1],         # sums to 0.15
        [0.3, 0.4],          # sums to 0.7 (matches → 0)
        [-0.1, 0.05, 0.04],  # sums to -0.01
    ]
    py_value = consistency_loss(lam, traj_adv_list, turn_adv_list)

    # Pad turn_adv to a tensor with mask.
    T_max = max(len(r) for r in turn_adv_list)
    B = len(turn_adv_list)
    turn_t = torch.zeros(B, T_max, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.long)
    for i, row in enumerate(turn_adv_list):
        turn_t[i, : len(row)] = torch.tensor(row, dtype=torch.float32)
        mask[i, : len(row)] = 1
    traj_t = torch.tensor(traj_adv_list, dtype=torch.float32)

    tensor_value = consistency_loss_tensor(lam, traj_t, turn_t, mask)
    assert isinstance(tensor_value, torch.Tensor)
    assert tensor_value.shape == ()
    assert abs(float(tensor_value.item()) - float(py_value)) < 1e-6, (
        f"tensor variant differs: tensor={tensor_value.item()} vs python={py_value}"
    )


def test_consistency_loss_tensor_backward_populates_alpha_grad() -> None:
    """A grad-tracking α-tensor receives a non-trivial gradient from
    `.backward()` through `consistency_loss_tensor`."""
    lam = 0.5
    B, T = 3, 4
    alpha = torch.full((B, T), 1.0 / T, requires_grad=True)
    R = torch.tensor([1.0, -0.5, 0.3])
    mask = torch.ones(B, T, dtype=torch.long)
    # turn_adv = α · R (per-row); traj_adv = R per row (gradient sink).
    turn_adv = alpha * R.unsqueeze(-1)
    traj_adv = R.detach()

    loss = consistency_loss_tensor(lam, traj_adv, turn_adv, mask)
    loss.backward()

    assert alpha.grad is not None
    assert alpha.grad.shape == alpha.shape
    # Σ_t α_t = 1 + R = R, but the loss target is `traj_adv = R`, so the
    # consistency term is exactly 0 — but the gradient flowing through the
    # tensor path is still computable. With Σ_t α_t·R = R, diff=0, so the
    # mean-of-squares is 0 and the analytic gradient is 0. Build a less
    # degenerate case to confirm gradient is non-zero.
    alpha2 = torch.full((B, T), 0.1, requires_grad=True)
    R2 = torch.tensor([1.0, -0.5, 0.3])
    turn_adv_2 = alpha2 * R2.unsqueeze(-1)
    loss2 = consistency_loss_tensor(0.5, R2, turn_adv_2, mask)
    loss2.backward()
    assert alpha2.grad is not None
    assert alpha2.grad.abs().sum().item() > 0.0


def test_consistency_loss_tensor_respects_mask_zero_positions() -> None:
    """Padded positions should NOT contribute to the per-trajectory sum.

    Build B=2, T_max=3 where row 0 has T_real=1 and row 1 has T_real=3.
    Padded values are deliberately set to large numbers to expose any
    mask-leakage.
    """
    traj = torch.tensor([0.5, 0.6])
    turn = torch.tensor([[0.5, 99.0, 99.0], [0.1, 0.2, 0.3]])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.long)
    loss_t = consistency_loss_tensor(1.0, traj, turn, mask)
    # Per-traj sums: row0 = 0.5 (matches → 0); row1 = 0.6 (matches → 0).
    assert abs(float(loss_t.item())) < 1e-6
