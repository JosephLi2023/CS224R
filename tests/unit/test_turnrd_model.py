"""Unit tests for `src.turnrd.model` (TurnRD; Method B; M1 surface).

Verification matrix per `MEDIUM_FIXES.md::M1` + the
`~/.llms/plans/cs224r_hgpo_method_b_turnrd_m1.plan.md` plan:

1. `test_turnrd_forward_output_shapes`
2. `test_turnrd_mask_honored_padded_turns_zero_attention`
3. `test_turnrd_decompose_invariant`
4. `test_turnrd_mode_1_loss_decreases_on_synthetic`
5. `test_turnrd_mode_2_loss_decreases_on_synthetic`
6. `test_turnrd_raises_on_T_exceeding_max_turns`

Skipped cleanly on hosts without torch (e.g. lean Mac envs that haven't
installed the modal heavy stack) via module-level `pytest.importorskip`.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.turnrd.model import (  # noqa: E402  (after importorskip gate)
    TurnRD,
    TurnRDConfig,
    TurnRDOutput,
    loss_mode_1,
    loss_mode_2,
)


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _make_model(input_dim: int = 768, max_turns: int = 64) -> TurnRD:
    _seed()
    cfg = TurnRDConfig(
        n_layers=2, hidden_size=64, n_heads=4, max_turns=max_turns, dropout=0.0
    )
    return TurnRD(cfg, input_dim=input_dim)


def test_turnrd_forward_output_shapes() -> None:
    """B=4, T=8, input_dim=768 → predicted_R: [4], cls_attn_weights: [4, 8],
    Σ_t α_t == 1 per row, no NaN/Inf."""
    B, T, D = 4, 8, 768
    model = _make_model(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)

    out = model(embeds, mask)

    assert isinstance(out, TurnRDOutput)
    assert out.predicted_R.shape == (B,)
    assert out.cls_attn_weights.shape == (B, T)
    assert torch.isfinite(out.predicted_R).all()
    assert torch.isfinite(out.cls_attn_weights).all()
    row_sums = out.cls_attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5)


def test_turnrd_mask_honored_padded_turns_zero_attention() -> None:
    """T_real=3, T_padded=5: padded positions are exactly 0.0 in α; unmasked
    weights still sum to 1 per row."""
    B, T, D = 2, 8, 64
    T_real = 3
    model = _make_model(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[:, :T_real] = 1

    out = model(embeds, mask)

    # Padded positions are exactly 0.
    padded = out.cls_attn_weights[:, T_real:]
    assert torch.equal(padded, torch.zeros_like(padded))
    # Real positions sum to 1.
    real_sums = out.cls_attn_weights[:, :T_real].sum(dim=-1)
    assert torch.allclose(real_sums, torch.ones(B), atol=1e-5)


def test_turnrd_decompose_invariant() -> None:
    """Σ_t r̂_t == R per row (within 1e-6) for arbitrary final rewards."""
    B, T, D = 4, 6, 64
    model = _make_model(input_dim=D)
    model.eval()
    embeds = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)

    out = model(embeds, mask)
    R = torch.tensor([1.0, 0.5, 0.0, -0.4])
    per_turn = out.decompose(R)

    assert per_turn.shape == (B, T)
    sums = per_turn.sum(dim=-1)
    assert torch.allclose(sums, R, atol=1e-6)


def test_turnrd_mode_1_loss_decreases_on_synthetic() -> None:
    """50 optimizer steps on a synthetic (turn_embeds, R = simple-linear-of-embeds)
    target. Mode 1 MSE-on-R loss must drop ≥30% (final ≤ 0.7 * initial).

    Uses Adam(lr=1e-3) rather than the literal SGD-lr-1e-3 the M1 spec
    sketched: with a small Transformer + softmax pool, plain SGD at 1e-3 is
    too slow to clear the 30% bar in 50 steps; Adam reaches it comfortably
    and is the realistic choice for the standalone TurnRD trainer that
    Day 13 will land. The verification gate is the >=30% drop, not the
    optimizer choice.
    """
    _seed(123)
    N, T, D = 64, 4, 128
    model = _make_model(input_dim=D, max_turns=16)

    # Synthetic: R = mean over turns of (w · embed) + tiny noise.
    w = torch.randn(D)
    embeds = torch.randn(N, T, D)
    target_R = (embeds @ w).mean(dim=-1) + 0.01 * torch.randn(N)
    mask = torch.ones(N, T, dtype=torch.long)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    initial = loss_mode_1(model(embeds, mask), target_R).item()
    for _ in range(50):
        opt.zero_grad()
        out = model(embeds, mask)
        loss = loss_mode_1(out, target_R)
        loss.backward()
        opt.step()
    final = loss_mode_1(model(embeds, mask), target_R).item()

    assert final <= 0.7 * initial, (
        f"Mode 1 loss did not decrease enough: initial={initial:.4f}, "
        f"final={final:.4f}, expected final <= {0.7 * initial:.4f}"
    )


def test_turnrd_mode_2_loss_decreases_on_synthetic() -> None:
    """50 optimizer steps where the per-turn target is `α* · R` for a
    fixed-random α* in the simplex Δ^{T-1}. Mode 2 MSE-per-turn loss must
    drop ≥30%. Uses Adam(lr=1e-3) per the same rationale as the Mode 1 test.
    """
    _seed(456)
    N, T, D = 64, 4, 128
    model = _make_model(input_dim=D, max_turns=16)

    embeds = torch.randn(N, T, D)
    R = torch.randn(N)
    # Fixed teacher α* ∈ Δ^{T-1}.
    alpha_star = torch.softmax(torch.randn(T), dim=-1)
    target_per_turn = alpha_star.unsqueeze(0) * R.unsqueeze(-1)  # [N, T]
    mask = torch.ones(N, T, dtype=torch.long)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    initial = loss_mode_2(model(embeds, mask), target_per_turn, R, mask).item()
    for _ in range(50):
        opt.zero_grad()
        out = model(embeds, mask)
        loss = loss_mode_2(out, target_per_turn, R, mask)
        loss.backward()
        opt.step()
    final = loss_mode_2(model(embeds, mask), target_per_turn, R, mask).item()

    assert final <= 0.7 * initial, (
        f"Mode 2 loss did not decrease enough: initial={initial:.4f}, "
        f"final={final:.4f}, expected final <= {0.7 * initial:.4f}"
    )


def test_turnrd_raises_on_T_exceeding_max_turns() -> None:
    """Building with max_turns=8 then passing T=9 must raise ValueError."""
    cfg = TurnRDConfig(n_layers=2, hidden_size=32, n_heads=4, max_turns=8, dropout=0.0)
    model = TurnRD(cfg, input_dim=16)
    embeds = torch.randn(1, 9, 16)
    mask = torch.ones(1, 9, dtype=torch.long)
    with pytest.raises(ValueError, match=r"max_turns"):
        model(embeds, mask)


def test_turnrd_raises_on_fully_padded_row() -> None:
    """A batch row with `attention_mask.sum(-1) == 0` must raise rather than
    return NaN. `nn.TransformerEncoder` softmax produces NaN for an all-True
    `src_key_padding_mask` row; the post-pool clamp can't recover it. The
    decomposer adapter short-circuits empty trajectories before they reach
    forward, so this guard only fires for direct callers.
    """
    model = _make_model(input_dim=16, max_turns=8)
    model.eval()
    embeds = torch.randn(2, 4, 16)
    mask = torch.ones(2, 4, dtype=torch.long)
    mask[1] = 0  # row 1 is fully padded
    with pytest.raises(ValueError, match=r"fully-padded|at least one real"):
        model(embeds, mask)


# ---------------------------------------------------------------------------
# v8 Tier 1: contrastive aux loss
# ---------------------------------------------------------------------------


def test_loss_contrastive_returns_zero_when_all_success():
    """No failures in batch → no negative pool → loss returns 0 (not NaN)."""
    from src.turnrd.model import loss_contrastive
    model = _make_model(input_dim=16, max_turns=4)
    model.eval()
    embeds = torch.randn(3, 4, 16)
    mask = torch.ones(3, 4, dtype=torch.long)
    out = model(embeds, mask)
    R = torch.tensor([0.5, 0.7, 0.3])  # all > 0
    L = loss_contrastive(out, R, mask, temperature=0.1)
    assert L.item() == 0.0


def test_loss_contrastive_returns_zero_when_all_failure():
    """No successes in batch → loss returns 0 (not NaN)."""
    from src.turnrd.model import loss_contrastive
    model = _make_model(input_dim=16, max_turns=4)
    model.eval()
    embeds = torch.randn(3, 4, 16)
    mask = torch.ones(3, 4, dtype=torch.long)
    out = model(embeds, mask)
    R = torch.zeros(3)  # all == 0
    L = loss_contrastive(out, R, mask, temperature=0.1)
    assert L.item() == 0.0


def test_loss_contrastive_produces_grad_when_mixed_batch():
    """Mixed success+failure → real positive scalar with gradient."""
    from src.turnrd.model import loss_contrastive
    model = _make_model(input_dim=16, max_turns=4)
    model.train()
    embeds = torch.randn(4, 4, 16)
    mask = torch.ones(4, 4, dtype=torch.long)
    out = model(embeds, mask)
    R = torch.tensor([1.0, 1.0, 0.0, 0.0])  # 2 success + 2 failure
    L = loss_contrastive(out, R, mask, temperature=0.1)
    assert L.item() > 0.0
    # Verify gradient flows back to the encoder.
    L.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    assert has_grad, "loss_contrastive should backprop to encoder params"


def test_loss_contrastive_pulls_success_together():
    """After several SGD steps with contrastive ONLY, success-turn embeddings
    cluster more tightly than at init (cosine sim within-success increases)."""
    from src.turnrd.model import loss_contrastive
    torch.manual_seed(0)
    model = _make_model(input_dim=16, max_turns=4)
    embeds = torch.randn(6, 4, 16)
    mask = torch.ones(6, 4, dtype=torch.long)
    R = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    # Initial within-success cosine sim.
    model.eval()
    with torch.no_grad():
        out0 = model(embeds, mask)
        succ_h0 = out0.encoder_hidden[R > 0]  # [3, T, H]
        flat0 = succ_h0.reshape(-1, 16)
        n0 = torch.nn.functional.normalize(flat0, dim=-1)
        sim_init = (n0 @ n0.T).fill_diagonal_(0).mean().item()

    # Train 50 steps with contrastive only.
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(50):
        opt.zero_grad()
        out = model(embeds, mask)
        L = loss_contrastive(out, R, mask, temperature=0.1)
        L.backward()
        opt.step()

    # Final within-success cosine sim.
    model.eval()
    with torch.no_grad():
        outF = model(embeds, mask)
        succ_hF = outF.encoder_hidden[R > 0]
        flatF = succ_hF.reshape(-1, 16)
        nF = torch.nn.functional.normalize(flatF, dim=-1)
        sim_final = (nF @ nF.T).fill_diagonal_(0).mean().item()

    assert sim_final > sim_init, (
        f"contrastive should pull success-turn embeddings together; "
        f"sim went {sim_init:.4f} → {sim_final:.4f} (no growth)"
    )


# ---------------------------------------------------------------------------
# v8 review fix M2: loss_value_head per-row T_i discount
# ---------------------------------------------------------------------------


def test_loss_value_head_uses_per_row_T_not_batch_T_max():
    """For mixed-length batches, the discount target at the LAST real
    turn of every trajectory should equal R (γ^0), regardless of batch
    T_max. Pre-fix, a length-3 row in a length-5 batch had final-turn
    target γ^2·R = 0.9025·R instead of R."""
    from src.turnrd.model import TurnRDOutput, loss_value_head

    # Build a synthetic output where:
    # row 0 has T_i=5 (full length); row 1 has T_i=3 (padded to T=5).
    B, T = 2, 5
    # V predicts EXACTLY the correct (post-fix) target; loss should be 0.
    R = torch.tensor([1.0, 1.0])
    gamma = 0.95
    # Per-row targets:
    # row 0: γ^4·R, γ^3·R, γ^2·R, γ^1·R, γ^0·R = [.815, .857, .9025, .95, 1]
    # row 1: γ^2·R, γ^1·R, γ^0·R, 0,    0     = [.9025, .95, 1, 0, 0]
    pred_v = torch.tensor([
        [gamma**4, gamma**3, gamma**2, gamma**1, gamma**0],
        [gamma**2, gamma**1, gamma**0,        0,        0],
    ])
    mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
    ], dtype=torch.long)
    out = TurnRDOutput(
        predicted_R=R,
        cls_attn_weights=torch.zeros(B, T),
        predicted_per_turn_R=pred_v,
        encoder_hidden=torch.zeros(B, T, 4),
    )
    loss = loss_value_head(out, R, mask, gamma=gamma)
    # Post-fix: targets match exactly → loss == 0.
    assert loss.item() < 1e-6, (
        f"loss_value_head should compute per-row T_i targets; got loss={loss.item()}. "
        "Pre-fix bug would compute target γ^(T_max-1-t)·R using global T_max=5, "
        "making row 1's last-turn target γ^(5-1-2)=γ^2 instead of γ^0=1."
    )


def test_loss_value_head_pre_fix_would_have_failed():
    """Independent verification: with full-length rows ONLY (T_i==T_max),
    the loss should also be 0 with pre-fix and post-fix code (sanity)."""
    from src.turnrd.model import TurnRDOutput, loss_value_head
    B, T = 2, 4
    R = torch.tensor([1.0, 1.0])
    gamma = 0.95
    pred_v = torch.tensor([
        [gamma**3, gamma**2, gamma**1, gamma**0],
        [gamma**3, gamma**2, gamma**1, gamma**0],
    ])
    mask = torch.ones(B, T, dtype=torch.long)
    out = TurnRDOutput(
        predicted_R=R,
        cls_attn_weights=torch.zeros(B, T),
        predicted_per_turn_R=pred_v,
        encoder_hidden=torch.zeros(B, T, 4),
    )
    loss = loss_value_head(out, R, mask, gamma=gamma)
    assert loss.item() < 1e-6
