"""Unit tests for the new `prior_bias` arg on `TurnRD.forward` (Method D).

The `prior_bias` arg adds a per-turn additive bias to the pre-softmax
logits inside the [CLS] cross-attention pool. Shape `[B, T]`. Used by
the `ResidualDecomposer` to inject `gamma_prior · raw_env_reward` so
TurnRD only needs to learn the residual correction.

Verification matrix:
1. `test_prior_bias_zero_matches_baseline_alpha`
2. `test_prior_bias_huge_collapses_to_progress`
3. `test_prior_bias_finite_blends`
4. `test_prior_bias_mask_respected`
5. `test_prior_bias_grad_flows_through_gamma`

Skipped cleanly on hosts without torch.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.turnrd.model import TurnRD, TurnRDConfig  # noqa: E402


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _make_model(input_dim: int = 16, max_turns: int = 16) -> TurnRD:
    _seed()
    cfg = TurnRDConfig(
        n_layers=2,
        hidden_size=32,
        n_heads=4,
        max_turns=max_turns,
        dropout=0.0,
        causal=True,
        value_head=True,
    )
    return TurnRD(cfg, input_dim=input_dim)


def test_prior_bias_zero_matches_baseline_alpha() -> None:
    """`prior_bias=zeros(...)` produces α identical to `prior_bias=None`."""
    B, T, D = 2, 5, 16
    model = _make_model(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)

    with torch.no_grad():
        out_no_bias = model(embeds, mask)
        out_zero_bias = model(embeds, mask, prior_bias=torch.zeros(B, T))

    assert torch.allclose(
        out_no_bias.cls_attn_weights, out_zero_bias.cls_attn_weights, atol=1e-6
    )


def test_prior_bias_huge_collapses_to_progress() -> None:
    """A very large positive bias on a single turn concentrates α there
    (>0.99 mass), regardless of the transformer logits."""
    B, T, D = 2, 5, 16
    model = _make_model(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)

    # One-hot at position 2 with huge magnitude.
    bias = torch.zeros(B, T)
    bias[:, 2] = 1e6

    with torch.no_grad():
        out = model(embeds, mask, prior_bias=bias)

    # α at position 2 should dominate.
    assert (out.cls_attn_weights[:, 2] > 0.99).all(), (
        f"prior_bias=1e6 at t=2 should give α[:,2] > 0.99; got "
        f"{out.cls_attn_weights[:, 2].tolist()}"
    )


def test_prior_bias_finite_blends() -> None:
    """A finite bias on one position should INCREASE α at that position
    relative to the no-bias baseline (but not collapse all mass)."""
    B, T, D = 2, 4, 16
    model = _make_model(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)

    bias = torch.zeros(B, T)
    bias[:, 1] = 1.0

    with torch.no_grad():
        out_no = model(embeds, mask)
        out_yes = model(embeds, mask, prior_bias=bias)

    # α at position 1 should be strictly larger with the bias.
    assert (out_yes.cls_attn_weights[:, 1] > out_no.cls_attn_weights[:, 1]).all(), (
        f"prior_bias=1.0 at t=1 should increase α[:,1]; got "
        f"no_bias={out_no.cls_attn_weights[:, 1].tolist()}, "
        f"yes_bias={out_yes.cls_attn_weights[:, 1].tolist()}"
    )
    # And it should still be a valid distribution (sum to 1, no NaN).
    sums = out_yes.cls_attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5)


def test_prior_bias_mask_respected() -> None:
    """Padded positions stay 0 in α even when their bias is huge — the
    `key_padding_mask` blocks them inside MHA, and the post-pool
    re-mask also zeros them defensively."""
    B, T, D = 2, 5, 16
    T_real = 3
    model = _make_model(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[:, :T_real] = 1

    # Huge bias at a PADDED position (t=4).
    bias = torch.zeros(B, T)
    bias[:, 4] = 1e6

    with torch.no_grad():
        out = model(embeds, mask, prior_bias=bias)

    padded = out.cls_attn_weights[:, T_real:]
    assert torch.equal(padded, torch.zeros_like(padded)), (
        f"padded positions should stay 0 even with huge bias; got {padded.tolist()}"
    )
    # Real positions still sum to 1.
    real_sums = out.cls_attn_weights[:, :T_real].sum(dim=-1)
    assert torch.allclose(real_sums, torch.ones(B), atol=1e-5)


def test_prior_bias_grad_flows_through_gamma() -> None:
    """Backward through α populates `gamma.grad`. This proves the
    residual decomposer's `gamma_prior` will receive gradients during
    training."""
    B, T, D = 2, 4, 16
    model = _make_model(input_dim=D)
    model.train()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)

    # gamma is a learnable scalar that scales the per-turn raw_env_reward prior.
    gamma = torch.tensor(1.0, requires_grad=True)
    progress = torch.tensor([[0.1, 0.3, 0.0, 0.6], [0.2, 0.0, 0.5, 0.3]])

    out = model(embeds, mask, prior_bias=gamma * progress)

    # Position-weighted sum gives a non-trivial scalar (α.sum() == B is
    # constant by softmax).
    weights = torch.arange(T, dtype=out.cls_attn_weights.dtype)
    loss = (out.cls_attn_weights * weights).sum()
    loss.backward()

    assert gamma.grad is not None, "gamma should receive a gradient"
    assert gamma.grad.abs().item() > 0.0, (
        f"gamma.grad should be non-zero; got {gamma.grad.item()}"
    )
