"""Unit tests for `TurnRDv2` (architecturally simplified credit-assignment).

Test matrix:
1. forward output shapes match v1 contract (TurnRDOutput)
2. §3.2 invariant: Σ_t r̂_t == R (per row, within fp tolerance)
3. mask honored: padded positions get α == 0 and v == 0
4. bidirectional attention: a perturbation at turn T-1 reaches α at turn 0
   (verifies causal=False is actually wired through the encoder)
5. progress-prior init produces monotone-increasing α at init time
6. loss_v2_pred decreases on a tiny synthetic regression task
7. loss_v2_rank zeros out cleanly when there are no contrastive pairs
8. loss_v2_progress_prior is zero when α already matches softmax(t/T)
9. R-prediction is identifiable for α: gradient flows through α (not just v)

Skipped cleanly on torch-less hosts via `pytest.importorskip("torch")`.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.turnrd.model import (  # noqa: E402  (after importorskip gate)
    TurnRDOutput,
    TurnRDv2,
    TurnRDv2Config,
    loss_v2_pred,
    loss_v2_progress_prior,
    loss_v2_rank,
    loss_v2_value,
)


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _make_v2(input_dim: int = 32, max_turns: int = 16, **cfg_overrides) -> TurnRDv2:
    _seed()
    cfg = TurnRDv2Config(
        n_layers=2,
        hidden_size=32,
        n_heads=4,
        max_turns=max_turns,
        dropout=0.0,
        **cfg_overrides,
    )
    return TurnRDv2(cfg, input_dim=input_dim)


def test_v2_forward_output_shapes() -> None:
    """Same TurnRDOutput surface as v1; α sums to 1; predicted_R is finite."""
    B, T, D = 4, 6, 32
    model = _make_v2(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)

    out = model(embeds, mask)

    assert isinstance(out, TurnRDOutput)
    assert out.predicted_R.shape == (B,)
    assert out.cls_attn_weights.shape == (B, T)  # this is α in v2
    assert out.predicted_per_turn_R is not None
    assert out.predicted_per_turn_R.shape == (B, T)
    assert torch.isfinite(out.predicted_R).all()
    assert torch.isfinite(out.cls_attn_weights).all()
    row_sums = out.cls_attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5)


def test_v2_decompose_invariant() -> None:
    """§3.2: Σ_t (α_t · R) == R per row."""
    B, T, D = 3, 5, 32
    model = _make_v2(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    R = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    out = model(embeds, mask)
    per_turn = out.decompose(R)  # uses α_t · R, same as v1

    assert per_turn.shape == (B, T)
    summed = per_turn.sum(dim=-1)
    assert torch.allclose(summed, R, atol=1e-5)


def test_v2_mask_zeros_padded_alpha_and_value() -> None:
    """Padded positions must have α == 0 AND v == 0 (so r̂_t == 0 there)."""
    B, T, D = 2, 6, 32
    model = _make_v2(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.long)

    out = model(embeds, mask)

    # α == 0 at padded positions
    assert torch.all(out.cls_attn_weights[mask == 0] == 0)
    # And rows still sum to 1 over the unmasked positions
    row_sums = out.cls_attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5)
    # v == 0 at padded positions
    assert out.predicted_per_turn_R is not None
    assert torch.all(out.predicted_per_turn_R[mask == 0] == 0)


def test_v2_is_bidirectional_perturbation_propagates_back() -> None:
    """Verifies causal=False is wired through: perturbing turn T-1's input
    changes α at turn 0. With causal=True (v1 default) it would NOT.
    """
    B, T, D = 1, 5, 32
    model = _make_v2(input_dim=D)
    model.eval()
    mask = torch.ones(B, T, dtype=torch.long)
    embeds_a = torch.randn(B, T, D)
    embeds_b = embeds_a.clone()
    embeds_b[0, T - 1] = embeds_b[0, T - 1] + 5.0  # perturb only LAST turn

    with torch.no_grad():
        alpha_a = model(embeds_a, mask).cls_attn_weights[0, 0].item()
        alpha_b = model(embeds_b, mask).cls_attn_weights[0, 0].item()

    # Bidirectional: perturbing turn 4 must reach turn 0's α via attention.
    # Use a generous tolerance — we just need to confirm the dependency
    # exists, not that it has a particular magnitude.
    assert abs(alpha_a - alpha_b) > 1e-4, (
        "TurnRDv2 appears to be effectively causal: perturbing turn T-1 did "
        "not change α at turn 0."
    )


def test_v2_progress_prior_init_is_monotone_at_init() -> None:
    """At init time, with progress_prior_strength > 0, α should INCREASE in
    expectation across t — matching the Method-C "later turns matter more"
    prior. We check on multiple random batches to average out the random
    encoder/score-head init noise.
    """
    B, T, D = 16, 6, 32
    model = _make_v2(input_dim=D, progress_prior_strength=2.0)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)

    with torch.no_grad():
        alpha = model(embeds, mask).cls_attn_weights  # [B, T]
    mean_alpha = alpha.mean(dim=0)  # [T]
    # We don't require strict monotonicity per-position (random init noise
    # in the score head can flip adjacent positions); we just require the
    # last turn to outweigh the first turn on average.
    assert mean_alpha[-1] > mean_alpha[0], (
        f"Progress prior should make late turns outweigh early ones at init; "
        f"got mean_alpha = {mean_alpha.tolist()}"
    )


def test_v2_progress_prior_off_is_uniform_in_expectation() -> None:
    """With progress_prior_strength = 0, untrained α is approximately
    uniform on average (no monotonicity bias). Sanity-checks the knob.
    """
    B, T, D = 32, 4, 32
    model = _make_v2(input_dim=D, progress_prior_strength=0.0)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)

    with torch.no_grad():
        alpha = model(embeds, mask).cls_attn_weights
    mean_alpha = alpha.mean(dim=0)
    # All within 0.1 of uniform 1/T = 0.25 — generous tolerance.
    assert torch.allclose(mean_alpha, torch.full((T,), 1.0 / T), atol=0.10)


def test_v2_loss_pred_decreases_on_synthetic() -> None:
    """SGD on `loss_v2_pred` should reduce MSE(R̂, R) on a fixed batch."""
    B, T, D = 8, 4, 16
    _seed(42)
    model = TurnRDv2(
        TurnRDv2Config(n_layers=1, hidden_size=16, n_heads=2, max_turns=8, dropout=0.0),
        input_dim=D,
    )
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    R = torch.randn(B)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    initial_loss = float(loss_v2_pred(model(embeds, mask), R).detach())
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        loss = loss_v2_pred(model(embeds, mask), R)
        loss.backward()
        opt.step()
    final_loss = float(loss_v2_pred(model(embeds, mask), R).detach())
    assert final_loss < initial_loss * 0.5, (
        f"loss_v2_pred didn't drop: initial={initial_loss}, final={final_loss}"
    )


def test_v2_loss_pred_gradient_flows_into_alpha() -> None:
    """Identifiability check: the R-prediction loss must produce a non-zero
    gradient through the SCORE head (which controls α), not only through
    the value head. This is the key fix vs v1's `loss_mode_1`.
    """
    B, T, D = 4, 5, 16
    _seed(7)
    model = _make_v2(input_dim=D)
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    R = torch.randn(B)

    out = model(embeds, mask)
    loss = loss_v2_pred(out, R)
    loss.backward()

    # The first linear of score_head must have a non-trivial grad.
    score_grad = model.score_head[0].weight.grad
    assert score_grad is not None
    score_grad_norm = float(score_grad.detach().abs().sum())
    assert score_grad_norm > 1e-6, (
        f"Score-head grad is ~0 ({score_grad_norm}); R-prediction loss is "
        "not identifiable for α — same v1 bug."
    )


def test_v2_loss_rank_zero_on_constant_R() -> None:
    """No contrastive pairs ⇒ ranking loss returns 0 (not NaN, not raise)."""
    B, T, D = 4, 3, 16
    model = _make_v2(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    R_const = torch.zeros(B)

    out = model(embeds, mask)
    loss = loss_v2_rank(out, R_const)
    assert torch.isfinite(loss).all()
    assert float(loss.detach()) == 0.0


def test_v2_loss_rank_pushes_high_R_above_low_R() -> None:
    """SGD on `loss_v2_rank` should rank R̂ for high-R rows above low-R rows."""
    B, T, D = 6, 3, 16
    _seed(11)
    model = _make_v2(input_dim=D)
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    # Three low-R, three high-R rows.
    R = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    initial_loss = float(loss_v2_rank(model(embeds, mask), R, margin=0.5).detach())
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        loss = loss_v2_rank(model(embeds, mask), R, margin=0.5)
        loss.backward()
        opt.step()
    with torch.no_grad():
        R_pred = model(embeds, mask).predicted_R
    # High-R rows should have higher R̂ than low-R rows on average.
    assert R_pred[3:].mean() > R_pred[:3].mean(), (
        f"Ranking loss didn't separate the two groups: "
        f"low-R̂ mean={R_pred[:3].mean().item()}, high-R̂ mean={R_pred[3:].mean().item()}"
    )


def test_v2_loss_value_decreases_on_synthetic() -> None:
    """SGD on `loss_v2_value` should reduce MSE(v, target) — confirms the
    value head is plumbed correctly and supervisable.
    """
    B, T, D = 8, 4, 16
    _seed(3)
    model = TurnRDv2(
        TurnRDv2Config(n_layers=1, hidden_size=16, n_heads=2, max_turns=8, dropout=0.0),
        input_dim=D,
    )
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    target_v = torch.randn(B, T)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    initial_loss = float(loss_v2_value(model(embeds, mask), target_v, mask).detach())
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        loss = loss_v2_value(model(embeds, mask), target_v, mask)
        loss.backward()
        opt.step()
    final_loss = float(loss_v2_value(model(embeds, mask), target_v, mask).detach())
    assert final_loss < initial_loss * 0.5, (
        f"loss_v2_value didn't drop: initial={initial_loss}, final={final_loss}"
    )


def test_v2_progress_prior_loss_is_finite_and_nonneg() -> None:
    """KL(α || progress_prior) must be ≥ 0 and finite on random α."""
    B, T, D = 4, 5, 16
    model = _make_v2(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    out = model(embeds, mask)
    loss = loss_v2_progress_prior(out, mask)
    assert torch.isfinite(loss).all()
    assert float(loss.detach()) >= -1e-6  # KL ≥ 0 (small fp slack)


def test_v2_raises_on_T_exceeding_max_turns() -> None:
    """forward() must reject turn-counts > cfg.max_turns rather than silently
    truncate or NaN out the positional embedding lookup.
    """
    B, T, D = 1, 9, 16
    model = _make_v2(input_dim=D, max_turns=8)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds cfg.max_turns"):
        model(embeds, mask)


def test_v2_raises_on_fully_padded_row() -> None:
    """forward() must reject all-padding rows (no real turns) loudly."""
    B, T, D = 2, 4, 16
    model = _make_v2(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="at least one real"):
        model(embeds, mask)
