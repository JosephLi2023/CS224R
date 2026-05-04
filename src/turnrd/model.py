"""TurnRD model — learned per-turn reward decomposer (Method B; proposal §3.2).

Architecture:
- `input_proj`: learned linear `[input_dim → hidden_size]` so we can run
  TurnRD natively at `hidden_size=256` even when the upstream policy hidden
  state is wider (e.g. 1536 for Qwen2.5-1.5B). See `MEDIUM_FIXES.md::M1`
  "open design question" — the default.
- `pos_embed`: learned positional embedding for turn index `0..max_turns-1`.
- `encoder`: stacked self-attention layers over the turn sequence (turn↔turn
  mixing). The encoder does NOT contain a [CLS] token; the [CLS] role is
  performed by a separate cross-attention pool below.
- `cls_pool`: a single cross-attention layer where a learned `cls_query`
  attends over the encoder outputs. Its softmax `attn` weights ARE the
  `α_t` of proposal §3.2 (non-negative, sum-to-1 along T, mask-respecting).
  This is the cleanest way to extract attention weights from PyTorch:
  the built-in `nn.TransformerEncoder` does not surface per-layer attention
  via its forward signature.
- `r_head`: linear `[hidden_size → 1]` from the pooled [CLS] vector to
  `predicted_R`.

Per-turn rewards are constructed as `r̂_t = α_t · R` via
`TurnRDOutput.decompose(R)`. Because `Σ_t α_t = 1` by softmax (with
masking), the §3.2 invariant `Σ_t r̂_t = R` holds exactly.

torch is imported at module top — this module is consumed only on
torch-enabled hosts (Modal A100); Mac-side tests gate via
`pytest.importorskip("torch")`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TurnRDConfig:
    """Hyperparameters for the TurnRD model.

    Defaults match `MEDIUM_FIXES.md::M1`. They size a small Transformer
    (~1–2 M params at hidden_size=256) that can train comfortably on a
    single A100 alongside the LoRA policy.
    """

    n_layers: int = 4
    hidden_size: int = 256
    n_heads: int = 4
    max_turns: int = 64
    dropout: float = 0.1


@dataclass(frozen=True)
class TurnRDOutput:
    """Forward-pass outputs of `TurnRD.forward`.

    `predicted_R: [B]`         — scalar `R` predicted from the pooled [CLS] vec.
    `cls_attn_weights: [B, T]` — softmax probabilities, sum to 1 along T per
                                 row (within fp32 tol), padded positions == 0.
    `per_turn_rewards`         — None at construction; populated by callers via
                                 `decompose(R)`. Kept in the dataclass so the
                                 spec API surface is self-documenting.
    """

    predicted_R: torch.Tensor
    cls_attn_weights: torch.Tensor
    per_turn_rewards: Optional[torch.Tensor] = None

    def decompose(self, final_reward: torch.Tensor) -> torch.Tensor:
        """Return `[B, T]` per-turn rewards `r̂_t = α_t · R`, masked.

        `final_reward: [B]`. Returns a fresh tensor so callers can `.cpu()`
        / `.tolist()` / store it without aliasing `cls_attn_weights`.

        The §3.2 invariant `Σ_t r̂_t = R` holds by construction because
        `Σ_t α_t = 1` (softmax) and padded positions were already zeroed
        in `cls_attn_weights`.
        """
        if final_reward.dim() != 1:
            raise ValueError(
                f"decompose: final_reward must be [B], got shape {tuple(final_reward.shape)}."
            )
        if final_reward.shape[0] != self.cls_attn_weights.shape[0]:
            raise ValueError(
                "decompose: final_reward batch dim "
                f"({final_reward.shape[0]}) must match cls_attn_weights batch dim "
                f"({self.cls_attn_weights.shape[0]})."
            )
        return self.cls_attn_weights * final_reward.to(
            dtype=self.cls_attn_weights.dtype, device=self.cls_attn_weights.device
        ).unsqueeze(-1)


class TurnRD(nn.Module):
    """Small Transformer + [CLS] cross-attention head over per-turn embeddings.

    `forward(turn_embeds: [B, T, input_dim], attention_mask: [B, T])
        -> TurnRDOutput`.

    Two training modes (see `loss_mode_1`, `loss_mode_2` below):
    - Mode 1: regress `predicted_R` against ground-truth final reward `R`
      (MSE). Trains the encoder + r_head end-to-end via the [CLS] pool.
    - Mode 2: supervise `decompose(R)` against externally-supplied per-turn
      labels (e.g. cached judge scores from a prior Method A run). Trains
      the [CLS] attention to match a teacher's decomposition.
    """

    def __init__(self, cfg: TurnRDConfig, input_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, cfg.hidden_size)
        self.pos_embed = nn.Embedding(cfg.max_turns, cfg.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.hidden_size,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            # `norm_first=True` triggers a UserWarning otherwise; it doesn't
            # change behavior, just suppresses the noise on import.
            enable_nested_tensor=False,
        )

        # Learned [CLS]-style query, single-vector per batch row.
        self.cls_query = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        nn.init.normal_(self.cls_query, mean=0.0, std=0.02)

        # Single cross-attention layer: cls_query attends over encoded turns.
        # The returned attention weights ARE α_t (softmax-normalized,
        # mask-respecting via key_padding_mask).
        self.cls_pool = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.r_head = nn.Linear(cfg.hidden_size, 1)

    def forward(
        self,
        turn_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> TurnRDOutput:
        if turn_embeds.dim() != 3:
            raise ValueError(
                f"forward: turn_embeds must be [B, T, input_dim], got shape "
                f"{tuple(turn_embeds.shape)}."
            )
        if attention_mask.dim() != 2:
            raise ValueError(
                f"forward: attention_mask must be [B, T], got shape "
                f"{tuple(attention_mask.shape)}."
            )
        B, T, D = turn_embeds.shape
        if D != self.input_dim:
            raise ValueError(
                f"forward: turn_embeds last dim {D} != configured input_dim {self.input_dim}."
            )
        if attention_mask.shape != (B, T):
            raise ValueError(
                f"forward: attention_mask shape {tuple(attention_mask.shape)} "
                f"must equal turn_embeds [B, T] = ({B}, {T})."
            )
        if T > self.cfg.max_turns:
            raise ValueError(
                f"forward: T={T} exceeds cfg.max_turns={self.cfg.max_turns}; "
                "either truncate the trajectory or rebuild the model with a larger max_turns."
            )
        # Reject fully-padded rows. `nn.TransformerEncoder` with an all-True
        # `src_key_padding_mask` row produces NaN in the self-attention softmax
        # (well-known PyTorch behavior), and the post-pool clamp_min(1e-12) won't
        # rescue an already-NaN attention weight tensor. The TurnRDDecomposer
        # adapter short-circuits empty trajectories before they reach forward,
        # so this only fires for direct callers (Day-13 standalone trainer,
        # ad-hoc debugging) — fail loudly rather than silently emit NaN rewards.
        if not bool((attention_mask.sum(dim=-1) > 0).all().item()):
            raise ValueError(
                "forward: every row of attention_mask must have at least one real "
                "(unmasked) turn. A fully-padded row produces NaN in the encoder "
                "softmax. Drop empty trajectories before batching."
            )

        # 1. Project to internal hidden_size.
        h = self.input_proj(turn_embeds)  # [B, T, H]

        # 2. Add positional embedding for turn indices 0..T-1.
        positions = torch.arange(T, device=h.device, dtype=torch.long)
        h = h + self.pos_embed(positions).unsqueeze(0)  # [B, T, H]

        # 3. PyTorch convention: True == "mask out" (padded). attention_mask is
        #    1 for real turns, so invert.
        key_padding_mask = ~attention_mask.bool()  # [B, T]

        # 4. Self-attention over turns. Padded positions are masked out.
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # [B, T, H]

        # 5. Build the [CLS] query: replicate the learned scalar query per row.
        cls_q = self.cls_query.expand(B, -1, -1)  # [B, 1, H]

        # 6. Single cross-attention pool. Returns:
        #    - pooled: [B, 1, H]
        #    - attn:   [B, 1, T]   (averaged over heads via average_attn_weights=True)
        #    The softmax attention weights ARE α_t (proposal §3.2).
        pooled, attn = self.cls_pool(
            cls_q,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        # 7. Squeeze the cls-query dim.
        cls_attn_weights = attn.squeeze(1)  # [B, T]

        # 7a. Re-mask defensively. PyTorch's MHA can leak tiny float noise into
        #     masked positions; enforce exact zeros and re-normalize so
        #     Σ_t α_t == 1 holds along the unmasked positions.
        float_mask = attention_mask.to(dtype=cls_attn_weights.dtype)
        cls_attn_weights = cls_attn_weights * float_mask
        # If a row has zero unmasked positions (degenerate; shouldn't happen
        # in production since the decomposer skips empty trajectories) the
        # row sum is 0 — clamp to avoid NaN.
        row_sum = cls_attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cls_attn_weights = cls_attn_weights / row_sum
        # Re-zero padded positions in case the renormalization above drifted
        # them (it shouldn't, since they were 0 before the divide).
        cls_attn_weights = cls_attn_weights * float_mask

        # 8. Predicted final reward from the pooled [CLS] vector.
        predicted_R = self.r_head(pooled.squeeze(1)).squeeze(-1)  # [B]

        return TurnRDOutput(
            predicted_R=predicted_R,
            cls_attn_weights=cls_attn_weights,
            per_turn_rewards=None,
        )


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def loss_mode_1(out: TurnRDOutput, target_R: torch.Tensor) -> torch.Tensor:
    """Mode 1 (regress final reward): MSE between `predicted_R` and ground-truth `R`.

    `target_R: [B]`. Returns a scalar tensor.
    """
    return F.mse_loss(out.predicted_R, target_R.to(dtype=out.predicted_R.dtype))


def loss_mode_2(
    out: TurnRDOutput,
    target_per_turn: torch.Tensor,
    R: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mode 2 (supervise decomposition): MSE between `α·R` and an external
    per-turn target (e.g. judge labels from a prior Method A cache).

    `target_per_turn: [B, T]`, `R: [B]`, `attention_mask: [B, T]`.
    Loss is averaged over real (unmasked) entries only.
    """
    pred = out.decompose(R)  # [B, T]
    float_mask = attention_mask.to(dtype=pred.dtype)
    sq = (pred - target_per_turn.to(dtype=pred.dtype)) ** 2
    sq = sq * float_mask
    n = float_mask.sum().clamp_min(1.0)
    return sq.sum() / n
