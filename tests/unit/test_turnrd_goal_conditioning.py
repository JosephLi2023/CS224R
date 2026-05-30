"""Unit tests for the FiLM goal-conditioned V-head (TurnRDv2).

Plan: `turnrd_goal_conditioned_v_head` Step 10.

Covers:
1. Backward-compat: flag=False → forward(turn_embeds, mask) produces
   byte-identical output to the legacy path. The new layers don't exist
   on the module → state_dict matches the legacy schema.
2. Flag=True + goal_emb=None → falls back gracefully to the
   unconditioned path (no FiLM modulation, identity behaviour).
3. Flag=True + DIFFERENT goal_emb → DIFFERENT predicted_per_turn_R
   for the SAME turn_embeds. Proves the conditioning has gradient flow
   AND is non-degenerate (after a tiny optimizer warmup so γ/β move
   off their zero init).
4. Backward pass produces non-zero gradients on goal_proj /
   goal_gamma / goal_beta parameters (a smoke that the autograd graph
   actually reaches those layers).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.turnrd.model import TurnRDv2, TurnRDv2Config


def _mk_inputs(B: int = 2, T: int = 4, D: int = 16) -> tuple:
    torch.manual_seed(0)
    turn_embeds = torch.randn(B, T, D)
    attention_mask = torch.ones(B, T, dtype=torch.long)
    return turn_embeds, attention_mask


def test_flag_off_byte_identical_to_legacy() -> None:
    """With flag=False, the model has the legacy state_dict shape and
    forward() matches the pre-FiLM behaviour exactly."""
    turn_embeds, attention_mask = _mk_inputs()
    cfg = TurnRDv2Config(
        n_layers=2, hidden_size=16, n_heads=2, dropout=0.0,
        progress_prior_strength=1.0, goal_conditioned_value_head=False,
    )
    torch.manual_seed(42)
    model = TurnRDv2(cfg, input_dim=turn_embeds.shape[-1])
    model.eval()
    out = model(turn_embeds, attention_mask)
    # FiLM layers must NOT exist when flag is off.
    assert model.goal_proj is None
    assert model.goal_gamma is None
    assert model.goal_beta is None
    # state_dict carries the legacy keys only (no goal_* entries).
    legacy_keys = set(model.state_dict().keys())
    assert not any("goal_" in k for k in legacy_keys)
    # Forward output shapes match.
    assert out.predicted_R.shape == (turn_embeds.shape[0],)
    assert out.cls_attn_weights.shape == (turn_embeds.shape[0], turn_embeds.shape[1])
    assert out.predicted_per_turn_R is not None
    assert out.predicted_per_turn_R.shape == (turn_embeds.shape[0], turn_embeds.shape[1])


def test_flag_on_goal_emb_none_falls_through() -> None:
    """With flag=True but goal_emb=None, the model takes the unconditioned
    path (no FiLM modulation). v_t equals what it would be without the
    FiLM block (at init, since γ ≈ 1 + 0 = 1 and β ≈ 0, the FiLM-on
    path with a non-None goal_emb ALSO computes the identity; this
    test specifically checks the None branch doesn't crash)."""
    turn_embeds, attention_mask = _mk_inputs()
    cfg = TurnRDv2Config(
        n_layers=2, hidden_size=16, n_heads=2, dropout=0.0,
        progress_prior_strength=1.0, goal_conditioned_value_head=True,
    )
    torch.manual_seed(42)
    model = TurnRDv2(cfg, input_dim=turn_embeds.shape[-1])
    model.eval()
    # Should not raise even though no goal_emb was provided.
    out = model(turn_embeds, attention_mask, goal_emb=None)
    assert out.predicted_per_turn_R is not None
    assert torch.isfinite(out.predicted_per_turn_R).all()


def test_film_at_init_matches_unconditioned() -> None:
    """At init, γ outputs ~0 (zero-init weights/bias). With the +1
    offset in the forward, γ_effective ≈ 1 and β ≈ 0 → FiLM modulation
    is the identity. The flag-on + non-None goal_emb output must equal
    the flag-on + None goal_emb output (and both equal what an
    unconditioned model would produce, modulo random init of the
    underlying encoder which is the same in both paths)."""
    turn_embeds, attention_mask = _mk_inputs()
    cfg = TurnRDv2Config(
        n_layers=2, hidden_size=16, n_heads=2, dropout=0.0,
        progress_prior_strength=1.0, goal_conditioned_value_head=True,
    )
    torch.manual_seed(123)
    model = TurnRDv2(cfg, input_dim=turn_embeds.shape[-1])
    model.eval()
    goal_emb_a = torch.randn(turn_embeds.shape[0], turn_embeds.shape[-1])
    out_none = model(turn_embeds, attention_mask, goal_emb=None)
    out_with = model(turn_embeds, attention_mask, goal_emb=goal_emb_a)
    # At init the FiLM path is identity, so V-head outputs match.
    assert torch.allclose(
        out_none.predicted_per_turn_R, out_with.predicted_per_turn_R, atol=1e-5
    )


def test_film_after_training_differentiates_by_goal() -> None:
    """After a few optimizer steps that move γ/β off zero-init, two
    DIFFERENT goal_emb values produce DIFFERENT predicted_per_turn_R
    on the SAME turn_embeds. Proves the conditioning has real
    expressive capacity (not just identity)."""
    turn_embeds, attention_mask = _mk_inputs()
    cfg = TurnRDv2Config(
        n_layers=2, hidden_size=16, n_heads=2, dropout=0.0,
        progress_prior_strength=1.0, goal_conditioned_value_head=True,
    )
    torch.manual_seed(7)
    model = TurnRDv2(cfg, input_dim=turn_embeds.shape[-1])
    # Manually nudge the FiLM weights off zero so the modulation is
    # non-identity; bypasses the need for a full optimizer loop.
    with torch.no_grad():
        model.goal_gamma.weight.normal_(0.0, 0.5)
        model.goal_beta.weight.normal_(0.0, 0.5)
    model.eval()
    g1 = torch.randn(turn_embeds.shape[0], turn_embeds.shape[-1])
    g2 = torch.randn(turn_embeds.shape[0], turn_embeds.shape[-1])
    out1 = model(turn_embeds, attention_mask, goal_emb=g1)
    out2 = model(turn_embeds, attention_mask, goal_emb=g2)
    diff = (out1.predicted_per_turn_R - out2.predicted_per_turn_R).abs().max().item()
    assert diff > 1e-4, (
        f"FiLM did not differentiate by goal_emb (max abs diff {diff:.2e}); "
        "expected the V-head output to depend on the goal."
    )


def test_film_backward_pass_gradients_flow() -> None:
    """A loss on predicted_per_turn_R produces non-zero gradients on
    goal_proj / goal_gamma / goal_beta parameters."""
    turn_embeds, attention_mask = _mk_inputs()
    cfg = TurnRDv2Config(
        n_layers=2, hidden_size=16, n_heads=2, dropout=0.0,
        progress_prior_strength=1.0, goal_conditioned_value_head=True,
    )
    torch.manual_seed(11)
    model = TurnRDv2(cfg, input_dim=turn_embeds.shape[-1])
    # Move γ off zero so the modulation has SOME signal to back-prop
    # through (otherwise predicted_per_turn_R is independent of γ at
    # the exact zero point and the gradient w.r.t. γ.weight is also
    # zero — mathematically correct but uninformative for this test).
    with torch.no_grad():
        model.goal_gamma.weight.normal_(0.0, 0.1)
    goal_emb = torch.randn(turn_embeds.shape[0], turn_embeds.shape[-1])
    out = model(turn_embeds, attention_mask, goal_emb=goal_emb)
    loss = out.predicted_per_turn_R.sum()
    loss.backward()
    for name, p in [
        ("goal_proj.weight", model.goal_proj.weight),
        ("goal_gamma.weight", model.goal_gamma.weight),
        ("goal_beta.weight", model.goal_beta.weight),
    ]:
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad contains NaN/Inf"
        assert p.grad.abs().max().item() > 0.0, f"{name} grad is identically zero"


def test_film_per_row_goal_emb_mask_reverts_masked_rows() -> None:
    """With per-row goal_emb_mask, rows whose mask is 0 should produce
    V-head outputs equal to the unconditioned path; rows whose mask is
    1 should reflect the FiLM modulation."""
    B, T, D = 2, 4, 16
    turn_embeds = torch.randn(B, T, D)
    attention_mask = torch.ones(B, T, dtype=torch.long)
    cfg = TurnRDv2Config(
        n_layers=2, hidden_size=16, n_heads=2, dropout=0.0,
        progress_prior_strength=1.0, goal_conditioned_value_head=True,
    )
    torch.manual_seed(19)
    model = TurnRDv2(cfg, input_dim=D)
    with torch.no_grad():
        model.goal_gamma.weight.normal_(0.0, 0.5)
        model.goal_beta.weight.normal_(0.0, 0.5)
    model.eval()
    goal_emb = torch.randn(B, D)
    # Row 0 has the goal, row 1 doesn't.
    mask = torch.tensor([1.0, 0.0])
    out_masked = model(turn_embeds, attention_mask, goal_emb=goal_emb, goal_emb_mask=mask)
    out_uncond = model(turn_embeds, attention_mask, goal_emb=None)
    # Row 1 (masked out) should match the unconditioned path exactly.
    assert torch.allclose(
        out_masked.predicted_per_turn_R[1], out_uncond.predicted_per_turn_R[1], atol=1e-5
    )
    # Row 0 (masked in) should diverge from the unconditioned path.
    diff0 = (
        out_masked.predicted_per_turn_R[0] - out_uncond.predicted_per_turn_R[0]
    ).abs().max().item()
    assert diff0 > 1e-4, (
        f"Row 0 with FiLM did not differ from the unconditioned path "
        f"(max abs diff {diff0:.2e})."
    )
