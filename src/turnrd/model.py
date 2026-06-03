"""TurnRD model - learned per-turn reward decomposer (Method B).

Architecture:
- `input_proj`: linear `[input_dim -> hidden_size]`.
- `pos_embed`: positional embedding for turn index `0..max_turns-1`.
- `encoder`: stacked self-attention over the turn sequence. No [CLS] token;
  the [CLS] role is done by a separate cross-attention pool.
- `cls_pool`: a single cross-attention layer where a learned `cls_query`
  attends over the encoder outputs. Its softmax weights ARE the `alpha_t`
  (non-negative, sum-to-1 along T, mask-respecting). nn.TransformerEncoder
  does not surface per-layer attention, hence the separate pool.
- `r_head`: linear `[hidden_size -> 1]` from the pooled [CLS] vector to
  `predicted_R`.

Per-turn rewards are `r_hat_t = alpha_t * R` via `TurnRDOutput.decompose(R)`.
Since `sum_t alpha_t = 1` (masked softmax), `sum_t r_hat_t = R` holds exactly.
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

    Defaults size a small Transformer trainable alongside the LoRA policy on
    one A100.
    - `value_head=True`: adds an auxiliary `V(h_t) -> scalar` head trained
      against the discounted future return `gamma^(T-t-1)*R` per turn, giving
      credit-relevant supervision from sparse R alone.
    """

    n_layers: int = 4
    hidden_size: int = 256
    n_heads: int = 4
    max_turns: int = 64
    dropout: float = 0.1
    causal: bool = True
    value_head: bool = True


@dataclass(frozen=True)
class TurnRDOutput:
    """Forward-pass outputs of `TurnRD.forward`.

    `predicted_R: [B]`         - scalar R from the pooled [CLS] vector.
    `cls_attn_weights: [B, T]` - softmax weights, sum to 1 along T per row,
                                 padded positions == 0.
    `predicted_per_turn_R`     - `[B, T]` per-turn V-head predictions; `None`
                                 when `cfg.value_head=False`. Trained against
                                 `gamma^(T-t-1)*R` per turn.
    `encoder_hidden`           - `[B, T, H]` per-turn encoder outputs, for the
                                 contrastive aux loss. Always populated.
    `per_turn_rewards`         - None at construction; populated by callers via
                                 `decompose(R)`.
    """

    predicted_R: torch.Tensor
    cls_attn_weights: torch.Tensor
    predicted_per_turn_R: Optional[torch.Tensor] = None
    encoder_hidden: Optional[torch.Tensor] = None
    per_turn_rewards: Optional[torch.Tensor] = None

    def decompose(self, final_reward: torch.Tensor) -> torch.Tensor:
        """Return `[B, T]` per-turn rewards `r_hat_t = alpha_t * R`, masked.

        `final_reward: [B]`. Returns a fresh tensor (no aliasing of
        `cls_attn_weights`). `sum_t r_hat_t = R` holds since `sum_t alpha_t = 1`
        and padded positions are zeroed.
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
            # enable_nested_tensor=False avoids a UserWarning under norm_first=True.
            enable_nested_tensor=False,
        )

        # Learned [CLS]-style query, single-vector per batch row.
        self.cls_query = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        nn.init.normal_(self.cls_query, mean=0.0, std=0.02)

        # Single cross-attention layer: cls_query attends over encoded turns.
        # The returned attention weights ARE alpha_t (softmax-normalized,
        # mask-respecting via key_padding_mask).
        self.cls_pool = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.r_head = nn.Linear(cfg.hidden_size, 1)

        # Auxiliary per-turn value head. Trained against gamma^(T-t-1)*R per
        # turn - a credit-relevant signal under sparse R alone.
        if cfg.value_head:
            self.value_head = nn.Linear(cfg.hidden_size, 1)
        else:
            self.value_head = None  # type: ignore[assignment]

    def forward(
        self,
        turn_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> TurnRDOutput:
        """Forward pass. `turn_embeds: [B, T, input_dim]`, `attention_mask:
        [B, T]` (1 real, 0 pad)."""
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
        # Reject fully-padded rows: an all-True src_key_padding_mask row makes
        # the encoder softmax produce NaN. Fail loudly rather than emit NaN.
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

        # 3. PyTorch convention: True == masked out. attention_mask is 1 for
        #    real turns, so invert. Build both masks as boolean (True = block).
        key_padding_mask = ~attention_mask.bool()  # [B, T] bool
        causal_attn_mask: torch.Tensor | None = None
        if self.cfg.causal:
            # [T, T] bool: True above diagonal = block future positions.
            causal_attn_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=h.device),
                diagonal=1,
            )

        # 4. Self-attention over turns. Padded positions are masked out.
        h = self.encoder(
            h,
            mask=causal_attn_mask,
            src_key_padding_mask=key_padding_mask,
        )  # [B, T, H]

        # 5. Build the [CLS] query: replicate the learned scalar query per row.
        cls_q = self.cls_query.expand(B, -1, -1)  # [B, 1, H]

        # 6. Single cross-attention pool. Returns:
        #    - pooled: [B, 1, H]
        #    - attn:   [B, 1, T]   (averaged over heads via average_attn_weights=True)
        #    The softmax attention weights ARE alpha_t.
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

        # 7a. Re-mask defensively: MHA can leak float noise into masked
        #     positions. Zero them and re-normalize so sum_t alpha_t == 1.
        float_mask = attention_mask.to(dtype=cls_attn_weights.dtype)
        cls_attn_weights = cls_attn_weights * float_mask
        # Clamp row sum to avoid NaN on degenerate all-masked rows.
        row_sum = cls_attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cls_attn_weights = cls_attn_weights / row_sum
        # Re-zero padded positions after the divide.
        cls_attn_weights = cls_attn_weights * float_mask

        # 8. Predicted final reward from the pooled [CLS] vector.
        predicted_R = self.r_head(pooled.squeeze(1)).squeeze(-1)  # [B]

        # 9. Optional per-turn V(h_t) predictions. Padded positions zeroed.
        predicted_per_turn_R: Optional[torch.Tensor] = None
        if self.value_head is not None:
            v = self.value_head(h).squeeze(-1)  # [B, T]
            predicted_per_turn_R = v * float_mask

        return TurnRDOutput(
            predicted_R=predicted_R,
            cls_attn_weights=cls_attn_weights,
            predicted_per_turn_R=predicted_per_turn_R,
            encoder_hidden=h,  # [B, T, H] - for contrastive aux loss
            per_turn_rewards=None,
        )


# Loss helpers


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
    """Mode 2 (supervise decomposition): MSE between `alpha*R` and an external
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


def loss_value_head(
    out: TurnRDOutput,
    R: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    gamma: float = 0.95,
) -> torch.Tensor:
    """Auxiliary per-turn value loss.

    Trains `V(h_t)` to predict the discounted future return `gamma^(T-t-1)*R`
    per real (unmasked) turn. `out` must have `predicted_per_turn_R` populated
    (`cfg.value_head=True`); `R: [B]`, `attention_mask: [B, T]`. Returns a
    scalar MSE over real turns, or `0` when built without `value_head`.
    """
    if out.predicted_per_turn_R is None:
        # value_head=False: return 0 so callers can multiply by a coefficient.
        return torch.zeros((), dtype=R.dtype, device=R.device)
    pred_v = out.predicted_per_turn_R  # [B, T]
    float_mask = attention_mask.to(dtype=pred_v.dtype)  # [B, T]
    B, T = pred_v.shape
    # Discount uses per-row trajectory length T_i, not batch-global T_max, so
    # short trajectories get gamma^0*R = R at their real final turn. Padded
    # positions are masked out after target construction.
    t_idx = torch.arange(T, device=pred_v.device, dtype=pred_v.dtype)  # [T]
    T_i = float_mask.sum(dim=-1, keepdim=True)  # [B, 1] real turn count per row
    # exponent[b, t] = max(0, T_i[b] - 1 - t); clamp guards padded positions.
    exponent = (T_i - 1.0 - t_idx.unsqueeze(0)).clamp_min(0.0)  # [B, T]
    discount = gamma ** exponent  # [B, T]
    target = R.to(dtype=pred_v.dtype).unsqueeze(-1) * discount  # [B, T]
    sq = (pred_v - target) ** 2 * float_mask
    n = float_mask.sum().clamp_min(1.0)
    return sq.sum() / n


def alpha_entropy(out: TurnRDOutput, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average entropy of alpha across the batch.

    Returns `Hbar = mean_b (-sum_t alpha_{b,t} * log alpha_{b,t})` over real
    positions only. Lower entropy = alpha concentrating credit on fewer turns.
    Also used as the negative-entropy regularization target (subtract
    beta*entropy from the trainer loss to discourage uniform alpha).
    """
    alpha = out.cls_attn_weights  # [B, T]
    float_mask = attention_mask.to(dtype=alpha.dtype)
    # Epsilon before log avoids -inf on zero (padded) entries.
    log_alpha = torch.log(alpha.clamp_min(1e-12))
    H = -(alpha * log_alpha * float_mask).sum(dim=-1)  # [B]
    return H.mean()


# TurnRDv2 - simplified credit-assignment decomposer.
#
# Differences from v1:
#   1. Bidirectional self-attention (credit is retrospective: turn t's value
#      depends on what happens after t).
#   2. Predicts `R_hat = sum_t alpha_t * v_t` directly (no CLS-pool
#      bottleneck), so dloss/dalpha_t is identifiable.
#   3. Pluggable V-target slot (callers can feed counterfactual targets).
#   4. Score head inits to a Method-C "progress" prior (linear in t/T) so the
#      untrained alpha is sensible rather than uniform.
#
# Architecture:
#   input_proj : Linear(D -> H)
#   pos_embed  : Embedding(max_turns, H)
#   encoder    : 2-layer bidirectional TransformerEncoder
#   score_head : MLP(H -> 1)   # per-turn alpha-score; alpha = softmax(scores)
#   value_head : MLP(H -> 1)   # per-turn v_t; R_hat = sum alpha*v
# Same `TurnRDOutput` shape, so `TurnRDDecomposer` is unchanged.


@dataclass(frozen=True)
class TurnRDv2Config:
    """Hyperparameters for `TurnRDv2`. Smaller defaults than v1: the direct
    score/value heads need no wide hidden_size, and the bidirectional encoder
    is sample-efficient at short sequence lengths (T <= 10).
    """

    n_layers: int = 2
    hidden_size: int = 128
    n_heads: int = 4
    max_turns: int = 64
    dropout: float = 0.1
    # Non-causal attention is the default because TurnRD analyzes completed trajectories.
    causal: bool = False
    # Strength of the progress-prior initialization bias on the score head.
    # A value of 0 disables the prior.
    progress_prior_strength: float = 1.0
    # Optional FiLM goal conditioning. When enabled, `forward(...)` takes a
    # per-trajectory goal embedding to modulate encoder states before the
    # score/value heads. FiLM layers are zero-initialized so the conditioned
    # path starts identical to the unconditioned one.
    goal_conditioned_value_head: bool = False


class TurnRDv2(nn.Module):
    """v2: bidirectional encoder + per-turn score head, no CLS bottleneck.

    Forward signature matches v1 so `TurnRDDecomposer` is unchanged. Two
    semantic differences:
      * `predicted_R` is `sum_t alpha_t * v_t` (identifiable for alpha).
      * `predicted_per_turn_R` is `v_t` directly (always populated), the slot
        for counterfactual supervision.
    """

    def __init__(self, cfg: TurnRDv2Config, input_dim: int) -> None:
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
            enable_nested_tensor=False,
        )

        # Per-turn alpha-score head: MLP(H -> 1), two-layer with GELU.
        self.score_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, 1),
        )

        # Per-turn value head: MLP(H -> 1). Kept separate from score_head so
        # alpha and v don't share a final layer (avoids v1's coupling).
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, 1),
        )

        # Progress prior: bias scores by ~t/T at init so the untrained alpha is
        # Method-C-like (linear in t/T after softmax) rather than uniform.
        if cfg.progress_prior_strength != 0.0:
            with torch.no_grad():
                # Prior is injected as a per-turn additive bias on scores
                # (see `_progress_prior_bias`), not on the head itself.
                self._has_progress_bias = True
        else:
            self._has_progress_bias = False
        # The per-turn bias is computed on the fly at forward time, not stored.

        # Optional FiLM goal conditioning. The FiLM layers exist only when
        # enabled, and zero initialization makes the conditioned path start
        # as the unconditioned model.
        if cfg.goal_conditioned_value_head:
            self.goal_proj = nn.Linear(input_dim, cfg.hidden_size)
            self.goal_gamma = nn.Linear(cfg.hidden_size, cfg.hidden_size)
            self.goal_beta = nn.Linear(cfg.hidden_size, cfg.hidden_size)
            # Zero initialization makes the initial FiLM modulation an identity.
            with torch.no_grad():
                nn.init.zeros_(self.goal_gamma.weight)
                nn.init.zeros_(self.goal_gamma.bias)
                nn.init.zeros_(self.goal_beta.weight)
                nn.init.zeros_(self.goal_beta.bias)
        else:
            self.goal_proj = None  # type: ignore[assignment]
            self.goal_gamma = None  # type: ignore[assignment]
            self.goal_beta = None  # type: ignore[assignment]

    def _progress_prior_bias(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return a [T] bias added to per-turn scores, linear in t/T so
        softmax(scores+bias) ~= softmax(t/T) for an untrained model."""
        if not self._has_progress_bias:
            return torch.zeros(T, device=device, dtype=dtype)
        positions = torch.arange(T, device=device, dtype=dtype)
        # Normalize by actual sequence length (not max_turns) so the bias
        # spans [0, prior_strength] for short trajectories.
        denom = max(1.0, float(T - 1))
        return (positions / denom) * float(self.cfg.progress_prior_strength)

    def forward(
        self,
        turn_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        goal_emb: Optional[torch.Tensor] = None,
        goal_emb_mask: Optional[torch.Tensor] = None,
    ) -> TurnRDOutput:
        if turn_embeds.dim() != 3:
            raise ValueError(
                f"TurnRDv2.forward: turn_embeds must be [B, T, input_dim]; "
                f"got shape {tuple(turn_embeds.shape)}."
            )
        if attention_mask.dim() != 2:
            raise ValueError(
                f"TurnRDv2.forward: attention_mask must be [B, T]; "
                f"got shape {tuple(attention_mask.shape)}."
            )
        B, T, D = turn_embeds.shape
        if D != self.input_dim:
            raise ValueError(
                f"TurnRDv2.forward: turn_embeds last dim {D} != input_dim {self.input_dim}."
            )
        if attention_mask.shape != (B, T):
            raise ValueError(
                f"TurnRDv2.forward: attention_mask shape {tuple(attention_mask.shape)} "
                f"!= turn_embeds [B, T] = ({B}, {T})."
            )
        if T > self.cfg.max_turns:
            raise ValueError(
                f"TurnRDv2.forward: T={T} exceeds cfg.max_turns={self.cfg.max_turns}."
            )
        if not bool((attention_mask.sum(dim=-1) > 0).all().item()):
            raise ValueError(
                "TurnRDv2.forward: every row must have at least one real (unmasked) turn."
            )

        # 1. Project to internal hidden_size.
        h = self.input_proj(turn_embeds)  # [B, T, H]

        # 2. Add positional embedding.
        positions = torch.arange(T, device=h.device, dtype=torch.long)
        h = h + self.pos_embed(positions).unsqueeze(0)  # [B, T, H]

        # 3. Bidirectional self-attention (or causal if explicitly enabled).
        key_padding_mask = ~attention_mask.bool()  # True == block
        causal_attn_mask: torch.Tensor | None = None
        if self.cfg.causal:
            causal_attn_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=h.device),
                diagonal=1,
            )

        h = self.encoder(
            h,
            mask=causal_attn_mask,
            src_key_padding_mask=key_padding_mask,
        )  # [B, T, H]

        # 4. Per-turn scores -> alpha via masked softmax.
        # Apply FiLM before the score head so alpha can depend on both the
        # turn representation and the goal.
        if (
            self.cfg.goal_conditioned_value_head
            and goal_emb is not None
            and self.goal_proj is not None
        ):
            if goal_emb.dim() != 2:
                raise ValueError(
                    f"TurnRDv2.forward: goal_emb must be [B, input_dim]; "
                    f"got shape {tuple(goal_emb.shape)}."
                )
            if goal_emb.shape[0] != B or goal_emb.shape[1] != self.input_dim:
                raise ValueError(
                    f"TurnRDv2.forward: goal_emb shape {tuple(goal_emb.shape)} "
                    f"must equal [B={B}, input_dim={self.input_dim}]."
                )
            g = self.goal_proj(goal_emb.to(dtype=h.dtype, device=h.device))
            gamma = self.goal_gamma(g)
            beta = self.goal_beta(g)
            gamma_eff = gamma.unsqueeze(1) + 1.0
            beta_b = beta.unsqueeze(1)
            h_cond = h * gamma_eff + beta_b
            if goal_emb_mask is not None:
                if goal_emb_mask.dim() != 1 or goal_emb_mask.shape[0] != B:
                    raise ValueError(
                        f"TurnRDv2.forward: goal_emb_mask must be [B={B}]; "
                        f"got shape {tuple(goal_emb_mask.shape)}."
                    )
                mask_b = goal_emb_mask.to(dtype=h.dtype, device=h.device).view(B, 1, 1)
                h_cond = h_cond * mask_b + h * (1.0 - mask_b)
        else:
            h_cond = h

        scores = self.score_head(h_cond).squeeze(-1)  # [B, T]
        # Add the progress-prior bias - broadcasts across batch.
        scores = scores + self._progress_prior_bias(T, scores.device, scores.dtype).unsqueeze(0)
        # Mask before softmax: set padded positions to a large negative value
        # (not -inf, for bf16/fp16 safety) so they get ~0 mass.
        neg_large = torch.tensor(-1e9, dtype=scores.dtype, device=scores.device)
        scores = torch.where(attention_mask.bool(), scores, neg_large)
        alpha = torch.softmax(scores, dim=-1)  # [B, T], sums to 1 along T

        # Defensive: zero padded positions explicitly (softmax gives ~0, not 0).
        float_mask = attention_mask.to(dtype=alpha.dtype)
        alpha = alpha * float_mask
        # Re-normalize (numerical safety; shouldn't shift values noticeably).
        row_sum = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        alpha = alpha / row_sum
        alpha = alpha * float_mask  # re-zero in case the divide drifted

        # 5. Per-turn value v_t. Reuses h_cond so alpha and v_t share the FiLM
        #    modulation; h_cond == h when goal conditioning is off.
        v = self.value_head(h_cond).squeeze(-1)  # [B, T]
        v = v * float_mask

        # 6. R_hat = sum_t alpha_t * v_t. Identifiable: dL/dalpha_t = 2(R_hat-R)*v_t.
        predicted_R = (alpha * v).sum(dim=-1)  # [B]

        return TurnRDOutput(
            predicted_R=predicted_R,
            cls_attn_weights=alpha,
            predicted_per_turn_R=v,
            encoder_hidden=h,
            per_turn_rewards=None,
        )


# v2 loss helpers


def loss_v2_pred(out: TurnRDOutput, target_R: torch.Tensor) -> torch.Tensor:
    """v2 R-prediction loss: MSE(sum alpha*v, R).

    The gradient flows directly into alpha (no `r_head` to absorb scale), so
    alpha upweights turns whose v_t is informative about R. Primary
    credit-assignment signal for v2.
    """
    return F.mse_loss(out.predicted_R, target_R.to(dtype=out.predicted_R.dtype))


def loss_v2_value(
    out: TurnRDOutput,
    target_per_turn: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """v2 per-turn value loss: MSE(v_t, target_t) over real turns.

    `target_per_turn` is pluggable (e.g. counterfactual per-turn deltas, or an
    R/T uniform fallback). Skip with a coefficient of 0.
    """
    if out.predicted_per_turn_R is None:
        # v2 always populates this; return 0 (not raise) if a v1 output is passed.
        return torch.zeros((), dtype=target_per_turn.dtype, device=target_per_turn.device)
    pred_v = out.predicted_per_turn_R  # [B, T]
    float_mask = attention_mask.to(dtype=pred_v.dtype)
    sq = (pred_v - target_per_turn.to(dtype=pred_v.dtype)) ** 2 * float_mask
    n = float_mask.sum().clamp_min(1.0)
    return sq.sum() / n


def loss_v2_rank(
    out: TurnRDOutput,
    R: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """Within-batch pairwise hinge: rank R_hat of high-R rows above low-R.

    For pairs (i, j) with R_i > R_j, push `R_hat_i >= R_hat_j + margin`. A
    relative-ordering complement to `loss_v2_pred`, robust to R-scale drift.
    Returns 0 when the batch has no contrastive pair.
    """
    R_pred = out.predicted_R  # [B]
    R_true = R.to(dtype=R_pred.dtype, device=R_pred.device)  # [B]
    B = R_pred.shape[0]
    if B < 2:
        return torch.zeros((), dtype=R_pred.dtype, device=R_pred.device)

    # diff[i, j] = R_true[i] - R_true[j]
    diff_true = R_true.unsqueeze(1) - R_true.unsqueeze(0)  # [B, B]
    diff_pred = R_pred.unsqueeze(1) - R_pred.unsqueeze(0)  # [B, B]
    # Pair (i, j) is positive iff R_true[i] > R_true[j] (by margin).
    pos_mask = (diff_true > 1e-6).to(dtype=R_pred.dtype)  # [B, B]
    n_pairs = pos_mask.sum().clamp_min(1.0)
    # Hinge: max(0, margin - (R_hat_i - R_hat_j)) when the pair is positive.
    hinge = torch.clamp_min(margin - diff_pred, 0.0) * pos_mask
    return hinge.sum() / n_pairs


def loss_v2_progress_prior(
    out: TurnRDOutput,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """KL(alpha || softmax(t/T)) - a soft pull toward the Method-C prior.

    Keeps alpha anchored near the progress prior (beta ~= 0.01) unless the
    data demands otherwise. Returns 0 on degenerate (T=0) rows.
    """
    alpha = out.cls_attn_weights  # [B, T]
    float_mask = attention_mask.to(dtype=alpha.dtype)  # [B, T]
    B, T = alpha.shape
    if T == 0:
        return torch.zeros((), dtype=alpha.dtype, device=alpha.device)
    # Build the progress prior: softmax(t/T_i) per row, masked.
    t_idx = torch.arange(T, device=alpha.device, dtype=alpha.dtype)  # [T]
    T_i = float_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [B, 1]
    raw = (t_idx.unsqueeze(0) / T_i)  # [B, T] in [0, 1)
    neg_large = torch.tensor(-1e9, dtype=alpha.dtype, device=alpha.device)
    raw_masked = torch.where(attention_mask.bool(), raw, neg_large)
    prior = torch.softmax(raw_masked, dim=-1) * float_mask
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    # KL(alpha || prior) summed over real positions, averaged over batch.
    eps = 1e-12
    kl_per_pos = alpha * (torch.log(alpha.clamp_min(eps)) - torch.log(prior.clamp_min(eps)))
    kl_per_pos = kl_per_pos * float_mask
    return kl_per_pos.sum(dim=-1).mean()


def loss_contrastive(
    out: TurnRDOutput,
    R: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    temperature: float = 0.1,
    success_threshold: float = 0.0,
) -> torch.Tensor:
    """InfoNCE-style contrastive loss on per-turn encoder hidden states.

    Pulls together turn embeddings from successful trajectories
    (R > `success_threshold`) and pushes them apart from unsuccessful ones,
    so the encoder learns outcome-discriminative features for alpha to attend
    over. `out.encoder_hidden` must be `[B, T, H]`; `R: [B]`;
    `attention_mask: [B, T]`. `temperature` (default 0.1) scales the InfoNCE
    log-likelihood. Returns 0 when no valid success/failure pair exists.
    """
    if out.encoder_hidden is None:
        return torch.zeros((), dtype=R.dtype, device=R.device)
    H_state = out.encoder_hidden  # [B, T, H]
    float_mask = attention_mask.to(dtype=H_state.dtype)  # [B, T]
    B, T, _ = H_state.shape

    # Trajectory-level success label.
    is_success = (R > success_threshold)  # [B] bool

    # Need at least 1 success AND 1 failure to have positive + negative pairs.
    if int(is_success.sum().item()) < 1 or int((~is_success).sum().item()) < 1:
        return torch.zeros((), dtype=R.dtype, device=R.device)

    # Flatten (B, T) -> (B*T) and filter unmasked turns only.
    H_flat = H_state.reshape(B * T, -1)  # [B*T, H]
    mask_flat = float_mask.reshape(B * T)  # [B*T]
    # Per-turn success label (broadcast trajectory R to all its turns).
    succ_flat = (
        is_success.unsqueeze(-1).expand(-1, T).reshape(B * T)
    )  # [B*T] bool

    # Keep only unmasked entries.
    keep = mask_flat > 0
    if int(keep.sum().item()) < 2:
        return torch.zeros((), dtype=R.dtype, device=R.device)

    H_keep = H_flat[keep]  # [N, H]
    succ_keep = succ_flat[keep]  # [N] bool

    n_succ = int(succ_keep.sum().item())
    n_fail = int((~succ_keep).sum().item())
    if n_succ < 2 or n_fail < 1:
        # Need >=2 successful turns (anchor + >=1 positive) and >=1 failure turn.
        return torch.zeros((), dtype=R.dtype, device=R.device)

    # L2-normalize for cosine similarity.
    H_norm = torch.nn.functional.normalize(H_keep, dim=-1)  # [N, H]

    # Pairwise similarity matrix [N, N], temperature-scaled.
    sim = H_norm @ H_norm.T / float(temperature)  # [N, N]
    # Mask out self-similarities (diagonal) so an anchor can't be its own positive.
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, float("-inf"))

    # For each success-anchor i:
    #   numerator   = sum over success j != i of exp(sim[i, j])
    #   denominator = sum over ALL j != i of exp(sim[i, j])
    #   L_i = -log(numerator / denominator)
    # Use logsumexp for numerical stability.
    succ_idx = succ_keep.nonzero(as_tuple=True)[0]  # indices of successes
    sim_anchors = sim[succ_idx]  # [n_succ, N]

    # Numerator mask: only success-success pairs (exclude self via the eye mask above).
    succ_row_mask = succ_keep.unsqueeze(0).expand(n_succ, -1)  # [n_succ, N]
    # Set non-positive entries to -inf for the numerator's logsumexp.
    sim_pos = sim_anchors.masked_fill(~succ_row_mask, float("-inf"))

    log_num = torch.logsumexp(sim_pos, dim=-1)  # [n_succ]
    log_den = torch.logsumexp(sim_anchors, dim=-1)  # [n_succ]
    loss_per_anchor = -(log_num - log_den)
    # Filter out anchors whose log_num was -inf (no positives left after self-mask).
    valid = ~torch.isinf(log_num)
    if int(valid.sum().item()) == 0:
        return torch.zeros((), dtype=R.dtype, device=R.device)
    return loss_per_anchor[valid].mean()
