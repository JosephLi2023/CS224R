"""Unit tests for TurnRDv2 ckpt warm-start round-trip.

Plan: `turnrd_v2_continual_larger` Step 1.

Covers:
1. Save → load round-trip preserves all parameters bitwise.
2. `strict=False` graceful fallback: loading a no-FiLM ckpt into a
   FiLM-enabled model leaves the FiLM-specific keys at their zero-init
   and reports the missing keys (so callers can log them).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.turnrd.model import TurnRDv2, TurnRDv2Config


def _mk_model(*, goal_cond: bool, hidden: int = 16, layers: int = 2) -> TurnRDv2:
    cfg = TurnRDv2Config(
        n_layers=layers,
        hidden_size=hidden,
        n_heads=2,
        max_turns=8,
        dropout=0.0,
        causal=False,
        progress_prior_strength=1.0,
        goal_conditioned_value_head=goal_cond,
    )
    torch.manual_seed(0)
    return TurnRDv2(cfg, input_dim=32)


def test_warm_start_preserves_all_tensors_exactly() -> None:
    """Save → load on a FiLM-enabled model: all tensors match the source."""
    src = _mk_model(goal_cond=True)
    # Perturb the model so the saved ckpt is NOT identical to zero-init —
    # otherwise the test trivially passes by coincidence.
    with torch.no_grad():
        for p in src.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "ckpt.pt"
        torch.save(src.state_dict(), ckpt_path)

        # Build a fresh model and warm-start from the saved state dict.
        dst = _mk_model(goal_cond=True)
        # Pre-load divergence check: fresh model SHOULD NOT match the perturbed src.
        for k, v in src.state_dict().items():
            if not torch.equal(v, dst.state_dict()[k]):
                break
        else:
            pytest.fail("Pre-load: src and dst already match — invalid test fixture")

        result = dst.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True),
            strict=False,
        )
        assert result.missing_keys == [], (
            f"unexpected missing keys after warm-start: {result.missing_keys}"
        )
        assert result.unexpected_keys == [], (
            f"unexpected extra keys: {result.unexpected_keys}"
        )

        # Post-load: every tensor must match bitwise.
        for k, v_src in src.state_dict().items():
            v_dst = dst.state_dict()[k]
            assert torch.equal(v_src, v_dst), f"tensor {k!r} mismatch after warm-start"


def test_warm_start_strict_false_tolerates_missing_film_keys() -> None:
    """Loading a no-FiLM ckpt into a FiLM-enabled model:
    - strict=False must not raise
    - missing_keys must list goal_proj/goal_gamma/goal_beta tensors
    - FiLM tensors stay at zero-init (γ=β=0) since the source ckpt had none
    """
    # Source: no-FiLM model (smaller state dict)
    src_no_film = _mk_model(goal_cond=False)
    with torch.no_grad():
        for p in src_no_film.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # Destination: FiLM-enabled model (state dict has goal_proj/gamma/beta)
    dst_film = _mk_model(goal_cond=True)
    # Snapshot FiLM tensors BEFORE load — they should be at zero-init for γ/β
    pre_gamma_w = dst_film.goal_gamma.weight.clone()
    pre_beta_w = dst_film.goal_beta.weight.clone()
    assert pre_gamma_w.abs().sum().item() == 0.0, "γ.weight should be zero-init"
    assert pre_beta_w.abs().sum().item() == 0.0, "β.weight should be zero-init"

    result = dst_film.load_state_dict(src_no_film.state_dict(), strict=False)
    # strict=False so this returned without raising; check the missing-keys report.
    missing_set = set(result.missing_keys)
    for expected in ("goal_proj.weight", "goal_gamma.weight", "goal_beta.weight"):
        assert expected in missing_set, (
            f"expected {expected!r} in missing_keys, got {missing_set}"
        )

    # FiLM weights should STILL be at zero-init (the load did not touch them).
    assert torch.equal(dst_film.goal_gamma.weight, pre_gamma_w)
    assert torch.equal(dst_film.goal_beta.weight, pre_beta_w)

    # The non-FiLM keys (input_proj, encoder, score_head, value_head)
    # SHOULD now match the perturbed source.
    for k, v_src in src_no_film.state_dict().items():
        v_dst = dst_film.state_dict()[k]
        assert torch.equal(v_src, v_dst), (
            f"non-FiLM tensor {k!r} should have been loaded from src"
        )
