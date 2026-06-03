"""TurnRD model - learned per-turn reward decomposer (Method B).

A small Transformer encodes per-turn embeddings; a [CLS] cross-attention pool
produces softmax weights `alpha_t` and a predicted reward. Per-turn rewards are
`r_hat_t = alpha_t * R` (via `decompose`), so `sum_t r_hat_t = R`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TurnRDConfig:
    """Hyperparameters for the TurnRD model. `value_head=True` adds an auxiliary
    per-turn `V(h_t)` head trained against `gamma^(T-t-1)*R`.
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

    - `predicted_R: [B]` scalar reward.
    - `cls_attn_weights: [B, T]` softmax weights (sum to 1 along T; pad == 0).
    - `predicted_per_turn_R: [B, T]` V-head predictions, or `None`.
    - `encoder_hidden: [B, T, H]` per-turn encoder outputs (for contrastive loss).
    - `per_turn_rewards`: `None` until set by `decompose(R)`.
    """

    predicted_R: torch.Tensor
    cls_attn_weights: torch.Tensor
    predicted_per_turn_R: Optional[torch.Tensor] = None
    encoder_hidden: Optional[torch.Tensor] = None
    per_turn_rewards: Optional[torch.Tensor] = None

    def decompose(self, final_reward: torch.Tensor) -> torch.Tensor:
        """Return `[B, T]` masked per-turn rewards `r_hat_t = alpha_t * R`
        (`final_reward: [B]`). Fresh tensor; `sum_t r_hat_t = R`.
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

    Mode 1 regresses `predicted_R` against `R`; Mode 2 supervises `decompose(R)`
    against external per-turn labels (see `loss_mode_1`, `loss_mode_2`).
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

        # cls_query attends over encoded turns; the returned weights ARE alpha_t.
        self.cls_pool = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.r_head = nn.Linear(cfg.hidden_size, 1)

        # Auxiliary per-turn value head; trained against gamma^(T-t-1)*R.
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
        # Reject fully-padded rows: they make the encoder softmax produce NaN.
        if not bool((attention_mask.sum(dim=-1) > 0).all().item()):
            raise ValueError(
                "forward: every row of attention_mask must have at least one real "
                "(unmasked) turn. A fully-padded row produces NaN in the encoder "
                "softmax. Drop empty trajectories before batching."
            )

        h = self.input_proj(turn_embeds)  # [B, T, H]

        positions = torch.arange(T, device=h.device, dtype=torch.long)
        h = h + self.pos_embed(positions).unsqueeze(0)  # [B, T, H]

        # PyTorch convention: True == masked out, so invert attention_mask.
        key_padding_mask = ~attention_mask.bool()  # [B, T] bool
        causal_attn_mask: torch.Tensor | None = None
        if self.cfg.causal:
            # [T, T] bool: True above diagonal = block future positions.
            causal_attn_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=h.device),
                diagonal=1,
            )

        h = self.encoder(
            h,
            mask=causal_attn_mask,
            src_key_padding_mask=key_padding_mask,
        )  # [B, T, H]

        cls_q = self.cls_query.expand(B, -1, -1)  # [B, 1, H]

        # Cross-attention pool; the returned attn weights ARE alpha_t.
        pooled, attn = self.cls_pool(
            cls_q,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        cls_attn_weights = attn.squeeze(1)  # [B, T]

        # Re-mask defensively (MHA can leak noise into masked positions) and
        # re-normalize so sum_t alpha_t == 1.
        float_mask = attention_mask.to(dtype=cls_attn_weights.dtype)
        cls_attn_weights = cls_attn_weights * float_mask
        # Clamp row sum to avoid NaN on degenerate all-masked rows.
        row_sum = cls_attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cls_attn_weights = cls_attn_weights / row_sum
        # Re-zero padded positions after the divide.
        cls_attn_weights = cls_attn_weights * float_mask

        # 8. Predicted final reward from the pooled [CLS] vector.
        predicted_R = self.r_head(pooled.squeeze(1)).squeeze(-1)  # [B]

        # Optional per-turn V(h_t) predictions (padded positions zeroed).
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
    """Mode 1: MSE between `predicted_R` and ground-truth `R` (`target_R: [B]`)."""
    return F.mse_loss(out.predicted_R, target_R.to(dtype=out.predicted_R.dtype))


def loss_mode_2(
    out: TurnRDOutput,
    target_per_turn: torch.Tensor,
    R: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mode 2: MSE between `alpha*R` and an external per-turn target, averaged
    over real turns. `target_per_turn: [B, T]`, `R: [B]`, `attention_mask: [B, T]`.
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
    """Auxiliary per-turn value loss: MSE of `V(h_t)` against `gamma^(T-t-1)*R`
    over real turns. Returns `0` when the model has no `value_head`.
    """
    if out.predicted_per_turn_R is None:
        # value_head=False: return 0 so callers can multiply by a coefficient.
        return torch.zeros((), dtype=R.dtype, device=R.device)
    pred_v = out.predicted_per_turn_R  # [B, T]
    float_mask = attention_mask.to(dtype=pred_v.dtype)  # [B, T]
    B, T = pred_v.shape
    # Discount uses per-row length T_i (not batch T_max), so a row's final real
    # turn gets gamma^0*R = R.
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
    """Mean entropy of alpha over real positions. Lower entropy = credit on
    fewer turns; also used as a negative-entropy regularizer.
    """
    alpha = out.cls_attn_weights  # [B, T]
    float_mask = attention_mask.to(dtype=alpha.dtype)
    # Epsilon before log avoids -inf on zero (padded) entries.
    log_alpha = torch.log(alpha.clamp_min(1e-12))
    H = -(alpha * log_alpha * float_mask).sum(dim=-1)  # [B]
    return H.mean()


# TurnRDv2 - simplified credit-assignment decomposer. Bidirectional encoder
# with per-turn score + value heads (no CLS bottleneck), so
# `predicted_R = sum_t alpha_t * v_t` is identifiable for alpha. The score head
# inits to a progress prior (alpha linear in t/T) rather than uniform. Same
# `TurnRDOutput` shape as v1, so `TurnRDDecomposer` is unchanged.


@dataclass(frozen=True)
class TurnRDv2Config:
    """Hyperparameters for `TurnRDv2` (smaller defaults than v1)."""

    n_layers: int = 2
    hidden_size: int = 128
    n_heads: int = 4
    max_turns: int = 64
    dropout: float = 0.1
    # Non-causal by default: TurnRD analyzes completed trajectories.
    causal: bool = False
    # Progress-prior init bias strength on the score head; 0 disables.
    progress_prior_strength: float = 1.0
    # Optional FiLM goal conditioning: modulate encoder states by a goal
    # embedding before the heads. Zero-init so it starts as a no-op.
    goal_conditioned_value_head: bool = False


class TurnRDv2(nn.Module):
    """v2: bidirectional encoder + per-turn score head, no CLS bottleneck.
    `predicted_R = sum_t alpha_t * v_t` (identifiable for alpha) and
    `predicted_per_turn_R = v_t`. Forward signature matches v1.
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

        # Per-turn alpha-score head.
        self.score_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, 1),
        )

        # Per-turn value head, kept separate from score_head so alpha and v
        # don't share a final layer.
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, 1),
        )

        # Progress prior: bias scores by ~t/T so untrained alpha is linear in
        # t/T rather than uniform.
        if cfg.progress_prior_strength != 0.0:
            with torch.no_grad():
                # Injected as an additive score bias (see `_progress_prior_bias`).
                self._has_progress_bias = True
        else:
            self._has_progress_bias = False
        # The per-turn bias is computed on the fly at forward time, not stored.

        # FiLM goal-conditioning layers (zero-init -> identity at start).
        if cfg.goal_conditioned_value_head:
            self.goal_proj = nn.Linear(input_dim, cfg.hidden_size)
            self.goal_gamma = nn.Linear(cfg.hidden_size, cfg.hidden_size)
            self.goal_beta = nn.Linear(cfg.hidden_size, cfg.hidden_size)
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
        # Normalize by actual length (not max_turns) so the bias spans
        # [0, prior_strength].
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

        h = self.input_proj(turn_embeds)  # [B, T, H]

        positions = torch.arange(T, device=h.device, dtype=torch.long)
        h = h + self.pos_embed(positions).unsqueeze(0)  # [B, T, H]

        # Bidirectional self-attention (causal if enabled).
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
        # FiLM before the score head so alpha depends on (turn, goal).
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
        # Add the progress-prior bias.
        scores = scores + self._progress_prior_bias(T, scores.device, scores.dtype).unsqueeze(0)
        # Mask before softmax: padded positions -> large negative (bf16-safe).
        neg_large = torch.tensor(-1e9, dtype=scores.dtype, device=scores.device)
        scores = torch.where(attention_mask.bool(), scores, neg_large)
        alpha = torch.softmax(scores, dim=-1)  # [B, T], sums to 1 along T

        # Defensive: zero padded positions explicitly (softmax gives ~0, not 0).
        float_mask = attention_mask.to(dtype=alpha.dtype)
        alpha = alpha * float_mask
        # Re-normalize (numerical safety).
        row_sum = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        alpha = alpha / row_sum
        alpha = alpha * float_mask  # re-zero in case the divide drifted

        # Per-turn value v_t (reuses h_cond; h_cond == h when goal cond. off).
        v = self.value_head(h_cond).squeeze(-1)  # [B, T]
        v = v * float_mask

        # R_hat = sum_t alpha_t * v_t (identifiable for alpha).
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
    """v2 R-prediction loss: MSE(sum alpha*v, R). Gradient flows directly into
    alpha (no `r_head` to absorb scale).
    """
    return F.mse_loss(out.predicted_R, target_R.to(dtype=out.predicted_R.dtype))


def loss_v2_value(
    out: TurnRDOutput,
    target_per_turn: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """v2 per-turn value loss: MSE(v_t, target_t) over real turns
    (`target_per_turn` is pluggable).
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
    """Within-batch pairwise hinge: for pairs R_i > R_j, push
    `R_hat_i >= R_hat_j + margin`. Returns 0 when no pair exists.
    """
    R_pred = out.predicted_R  # [B]
    R_true = R.to(dtype=R_pred.dtype, device=R_pred.device)  # [B]
    B = R_pred.shape[0]
    if B < 2:
        return torch.zeros((), dtype=R_pred.dtype, device=R_pred.device)

    # diff[i, j] = R_true[i] - R_true[j]
    diff_true = R_true.unsqueeze(1) - R_true.unsqueeze(0)  # [B, B]
    diff_pred = R_pred.unsqueeze(1) - R_pred.unsqueeze(0)  # [B, B]
    # Positive pair iff R_true[i] > R_true[j].
    pos_mask = (diff_true > 1e-6).to(dtype=R_pred.dtype)  # [B, B]
    n_pairs = pos_mask.sum().clamp_min(1.0)
    # Hinge: max(0, margin - (R_hat_i - R_hat_j)) when the pair is positive.
    hinge = torch.clamp_min(margin - diff_pred, 0.0) * pos_mask
    return hinge.sum() / n_pairs


def loss_v2_progress_prior(
    out: TurnRDOutput,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """KL(alpha || softmax(t/T)): a soft pull toward the progress prior.
    Returns 0 on degenerate (T=0) rows.
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
    # KL summed over real positions, averaged over batch.
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
    """InfoNCE-style contrastive loss on per-turn encoder hidden states: pulls
    success-trajectory turns together and pushes failures apart, so the encoder
    learns outcome-discriminative features. Returns 0 when no success/failure
    pair exists.
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
    # Per-turn success label (broadcast from trajectory R).
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
    # Mask the diagonal so an anchor isn't its own positive.
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, float("-inf"))

    # Per success-anchor: L_i = -log(sum_{success j!=i} exp(sim) / sum_{all j!=i} exp(sim)),
    # via logsumexp for stability.
    succ_idx = succ_keep.nonzero(as_tuple=True)[0]  # indices of successes
    sim_anchors = sim[succ_idx]  # [n_succ, N]

    # Numerator: success-success pairs only.
    succ_row_mask = succ_keep.unsqueeze(0).expand(n_succ, -1)  # [n_succ, N]
    # -inf on non-positive entries for the numerator logsumexp.
    sim_pos = sim_anchors.masked_fill(~succ_row_mask, float("-inf"))

    log_num = torch.logsumexp(sim_pos, dim=-1)  # [n_succ]
    log_den = torch.logsumexp(sim_anchors, dim=-1)  # [n_succ]
    loss_per_anchor = -(log_num - log_den)
    # Drop anchors with no positives left (log_num == -inf).
    valid = ~torch.isinf(log_num)
    if int(valid.sum().item()) == 0:
        return torch.zeros((), dtype=R.dtype, device=R.device)
    return loss_per_anchor[valid].mean()
