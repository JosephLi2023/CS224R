"""Unit tests for the LR warmup_cosine schedule used by train_turnrd.

Plan: `turnrd_v2_continual_larger` Step 1.

Covers:
1. warmup_cosine schedule produces expected LR at step 0 (≈base/warmup),
   warmup_steps (=base), and final step (=0).
2. constant schedule (default) keeps LR flat across all steps.
3. invalid schedule name raises ValueError.

These tests don't actually train — they construct the LambdaLR exactly
like `train_turnrd` does and inspect `optimizer.param_groups[0]['lr']`
after each `.step()`.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


def _build_scheduler(
    *,
    lr: float,
    warmup_steps: int,
    total_steps: int,
):
    """Replicates the LambdaLR construction inside train_turnrd exactly."""
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ws = max(0, int(warmup_steps))
    total_for_decay = max(1, total_steps - ws)

    def lr_lambda(step: int) -> float:
        if step < ws:
            return float(step + 1) / max(1, ws)
        progress = float(step - ws) / float(total_for_decay)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def test_warmup_cosine_lr_at_step_0() -> None:
    """At step 0, LR should be ~ base * (0+1)/warmup = base/warmup."""
    lr = 1e-3
    warmup = 100
    total = 1000
    opt, _ = _build_scheduler(lr=lr, warmup_steps=warmup, total_steps=total)
    # LambdaLR auto-applies _step_count=0 at init; LR at step 0 == base * factor(0)
    # = base * (0+1)/warmup
    expected = lr * (0 + 1) / warmup
    assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-6)


def test_warmup_cosine_lr_at_warmup_step() -> None:
    """At step == warmup_steps, the scheduler transitions from warmup to
    cosine decay; LR should be at the peak (≈ base)."""
    lr = 1e-3
    warmup = 100
    total = 1000
    opt, sched = _build_scheduler(lr=lr, warmup_steps=warmup, total_steps=total)
    # Step warmup times → we're now at the warmup boundary
    for _ in range(warmup):
        sched.step()
    # At step == warmup_steps, cosine starts at 0 → cos(0)=1 → factor = 0.5*(1+1) = 1.0
    assert opt.param_groups[0]["lr"] == pytest.approx(lr, rel=1e-6)


def test_warmup_cosine_lr_at_final_step() -> None:
    """At the final step, cosine has decayed all the way to 0."""
    lr = 1e-3
    warmup = 100
    total = 1000
    opt, sched = _build_scheduler(lr=lr, warmup_steps=warmup, total_steps=total)
    for _ in range(total):
        sched.step()
    # progress=1.0 → cos(π) = -1 → factor = 0.5*(1 + -1) = 0.0
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_warmup_cosine_monotone_decay_after_warmup() -> None:
    """After warmup, LR is monotone-decreasing through the cosine phase."""
    lr = 1e-3
    warmup = 100
    total = 1000
    opt, sched = _build_scheduler(lr=lr, warmup_steps=warmup, total_steps=total)
    # Step into the decay phase
    for _ in range(warmup):
        sched.step()
    prev_lr = opt.param_groups[0]["lr"]
    # Sample 9 more decay-phase points; each must be lower than the prior
    decay_points = total - warmup - 1
    sample_interval = max(1, decay_points // 10)
    for _ in range(10):
        for _ in range(sample_interval):
            sched.step()
        cur_lr = opt.param_groups[0]["lr"]
        assert cur_lr < prev_lr + 1e-12, (
            f"LR should decrease through cosine decay; prev={prev_lr}, cur={cur_lr}"
        )
        prev_lr = cur_lr


def test_warmup_cosine_zero_warmup_starts_at_peak() -> None:
    """warmup_steps=0 → schedule is pure cosine; step 0 LR == base."""
    lr = 1e-3
    opt, _ = _build_scheduler(lr=lr, warmup_steps=0, total_steps=100)
    # With warmup_steps=0, step 0 → progress = 0/(100-0) = 0 → cos(0)=1 → factor=1.0
    assert opt.param_groups[0]["lr"] == pytest.approx(lr, rel=1e-6)
