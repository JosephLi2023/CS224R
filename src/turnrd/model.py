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

    Improvements (post-WebShop-attempt):
    - `causal=True`: self-attn is causal so turn-t's α can only depend on
      h_0..h_t. Bidirectional attention encourages "focus on the
      high-info turn regardless of position" rather than "this turn
      caused that downstream success" — a credit-assignment red flag.
    - `value_head=True`: adds an auxiliary `V(h_t) → scalar` head trained
      against the discounted future return γ^(T-t-1)·R per turn. Gives
      each turn's representation direct credit-relevant supervision
      from sparse R alone (no env signal, no judge required) — the
      only Mode-1 way to push α off uniform.
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

    `predicted_R: [B]`         — scalar `R` predicted from the pooled [CLS] vec.
    `cls_attn_weights: [B, T]` — softmax probabilities, sum to 1 along T per
                                 row (within fp32 tol), padded positions == 0.
    `predicted_per_turn_R`     — `[B, T]` per-turn value predictions from
                                 the auxiliary V-head; `None` when
                                 `cfg.value_head=False`. Trained against
                                 `γ^(T-t-1) · R` per turn so the encoder
                                 learns features that predict per-turn
                                 discounted future return — gives α
                                 a credit-relevant signal to attend to.
    `encoder_hidden`           — `[B, T, H]` per-turn encoder outputs
                                 (post-self-attn, pre-CLS-pool). Exposed
                                 for the contrastive auxiliary loss
                                 (`loss_contrastive`); required for
                                 InfoNCE-style discriminative training
                                 between success/failure trajectories.
                                 Always populated.
    `per_turn_rewards`         — None at construction; populated by callers via
                                 `decompose(R)`. Kept in the dataclass so the
                                 spec API surface is self-documenting.
    """

    predicted_R: torch.Tensor
    cls_attn_weights: torch.Tensor
    predicted_per_turn_R: Optional[torch.Tensor] = None
    encoder_hidden: Optional[torch.Tensor] = None
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

        # Auxiliary per-turn value head (post-WebShop-attempt improvement).
        # Trained against γ^(T-t-1)·R per turn — gives the encoder a
        # credit-relevant signal under sparse R alone.
        if cfg.value_head:
            self.value_head = nn.Linear(cfg.hidden_size, 1)
        else:
            self.value_head = None  # type: ignore[assignment]

    def forward(
        self,
        turn_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        prior_bias: Optional[torch.Tensor] = None,
    ) -> TurnRDOutput:
        """Forward pass.

        Args:
            turn_embeds: `[B, T, input_dim]` per-turn embeddings.
            attention_mask: `[B, T]` long, 1 for real turns, 0 for padding.
            prior_bias: optional `[B, T]` float additive bias on the
                pre-softmax logits inside the [CLS] cross-attention
                pool. Used by Method D (Residual Decomposer) to inject
                a per-turn prior (e.g. `γ · raw_env_reward`) before
                the α softmax. The bias is added to `Q·K^T / √d`
                inside `nn.MultiheadAttention.attn_mask`, so larger
                values make a turn more likely to receive credit.
                When `None` the model behaves exactly like Method B.
        """
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
        if prior_bias is not None:
            if prior_bias.dim() != 2 or prior_bias.shape != (B, T):
                raise ValueError(
                    f"forward: prior_bias shape {tuple(prior_bias.shape)} "
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
        # 3a. Optional causal mask (post-WebShop-attempt improvement).
        # PyTorch deprecated mismatched-dtype combinations of `mask` +
        # `src_key_padding_mask`; build BOTH as boolean (True = block)
        # so they unify cleanly.
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
        #    The softmax attention weights ARE α_t (proposal §3.2).
        #
        #    Method D: when `prior_bias` is supplied, we add it to the
        #    pre-softmax logits via `attn_mask`. PyTorch's MHA expects
        #    `attn_mask` shape `[B*n_heads, L=1, S=T]`; we broadcast the
        #    same `[B, T]` bias across heads (no per-head differentiation).
        #    The bias is added to `Q·K^T / √d` BEFORE the softmax, so
        #    larger values make the corresponding turn more likely to
        #    receive credit. The bias is cast to the model's compute
        #    dtype (it may carry `gamma_prior` grad from upstream).
        cls_attn_mask: Optional[torch.Tensor] = None
        cls_key_padding_mask: torch.Tensor = key_padding_mask
        if prior_bias is not None:
            n_heads = self.cls_pool.num_heads
            cls_attn_mask = (
                prior_bias.to(dtype=h.dtype)
                .unsqueeze(1)  # [B, 1, T]
                .expand(B, n_heads, T)
                .reshape(B * n_heads, 1, T)
            )
            # PyTorch ≥2.1 deprecates passing a bool key_padding_mask
            # alongside a float attn_mask. Convert key_padding_mask to a
            # float-additive form (0 for keep, -inf for block) so both
            # masks share the same type. This is mathematically
            # equivalent — the bool form was being canonicalized to
            # exactly this internally.
            cls_key_padding_mask = torch.zeros_like(
                key_padding_mask, dtype=h.dtype
            ).masked_fill(key_padding_mask, float("-inf"))

        pooled, attn = self.cls_pool(
            cls_q,
            h,
            h,
            key_padding_mask=cls_key_padding_mask,
            attn_mask=cls_attn_mask,
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

        # 9. Optional per-turn V(h_t) predictions (for the discounted-future-
        #    return aux loss). Padded positions kept as 0 — the loss helper
        #    masks them anyway, but keep the tensor mask-clean for callers
        #    that want to compute correlations / per-position diagnostics.
        predicted_per_turn_R: Optional[torch.Tensor] = None
        if self.value_head is not None:
            v = self.value_head(h).squeeze(-1)  # [B, T]
            predicted_per_turn_R = v * float_mask

        return TurnRDOutput(
            predicted_R=predicted_R,
            cls_attn_weights=cls_attn_weights,
            predicted_per_turn_R=predicted_per_turn_R,
            encoder_hidden=h,  # [B, T, H] — for contrastive aux loss
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


def loss_value_head(
    out: TurnRDOutput,
    R: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    gamma: float = 0.95,
) -> torch.Tensor:
    """Auxiliary per-turn value loss (post-WebShop-attempt improvement).

    For sparse final R, the discounted future return at turn t in a
    trajectory of length T is `γ^(T-t-1) · R`. We train V(h_t) to
    predict that quantity per real (unmasked) turn.

    Args:
        out: TurnRDOutput from `model.forward`. Must have
            `predicted_per_turn_R` populated (`cfg.value_head=True`).
        R: `[B]` final scalar reward per trajectory.
        attention_mask: `[B, T]` 1 for real turns, 0 for padding.
        gamma: discount factor. Default 0.95.

    Returns:
        Scalar tensor — MSE averaged over real (unmasked) turn positions.
        `0` (no_grad) when the model was built without `value_head`.
    """
    if out.predicted_per_turn_R is None:
        # Compatibility: when `cfg.value_head=False`, return a zero
        # constant so callers can multiply by a coefficient without
        # branching. Use the same dtype/device as `R` for safety.
        return torch.zeros((), dtype=R.dtype, device=R.device)
    pred_v = out.predicted_per_turn_R  # [B, T]
    float_mask = attention_mask.to(dtype=pred_v.dtype)  # [B, T]
    B, T = pred_v.shape
    # M2 review fix: discount uses PER-ROW trajectory length T_i, not
    # the batch-global T_max. With γ=0.95 and a length-3 trajectory in
    # a length-5 batch, the OLD code computed target for real final
    # turn (t=2) as γ^(5-1-2)·R = 0.9025·R instead of γ^0·R = R, so
    # V was trained on systematically-biased targets for any
    # trajectory shorter than the batch max. Now compute T_i per row
    # from the mask and exponent (T_i - 1 - t_idx). Padded positions
    # would give a meaningless negative exponent; mask them out
    # explicitly via `float_mask` after target construction.
    t_idx = torch.arange(T, device=pred_v.device, dtype=pred_v.dtype)  # [T]
    T_i = float_mask.sum(dim=-1, keepdim=True)  # [B, 1] real turn count per row
    # exponent[b, t] = max(0, T_i[b] - 1 - t)
    # The clamp_min(0) is just for numerical safety on padded positions
    # (which are masked out anyway).
    exponent = (T_i - 1.0 - t_idx.unsqueeze(0)).clamp_min(0.0)  # [B, T]
    discount = gamma ** exponent  # [B, T]
    target = R.to(dtype=pred_v.dtype).unsqueeze(-1) * discount  # [B, T]
    sq = (pred_v - target) ** 2 * float_mask
    n = float_mask.sum().clamp_min(1.0)
    return sq.sum() / n


def alpha_entropy(out: TurnRDOutput, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute the average entropy of α across the batch.

    Returns a scalar tensor `H̄ = mean_b (-Σ_t α_{b,t} · log α_{b,t})`,
    summed over real (unmasked) positions only. Uniform α over T_b real
    turns gives H = log(T_b). Lower entropy = α concentrating credit on
    fewer turns. Used both as a diagnostic AND as the negative-entropy
    regularization target (subtract β·entropy from the standalone
    trainer's loss to encourage non-uniform decompositions).
    """
    alpha = out.cls_attn_weights  # [B, T]
    float_mask = attention_mask.to(dtype=alpha.dtype)
    # Add tiny epsilon before log to avoid -inf on exact-zero (padded)
    # entries, then mask them out with `float_mask`.
    log_alpha = torch.log(alpha.clamp_min(1e-12))
    H = -(alpha * log_alpha * float_mask).sum(dim=-1)  # [B]
    return H.mean()


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
    (final R > `success_threshold`); pushes them apart from turn
    embeddings in unsuccessful trajectories. Self-supervised — only
    needs the binary success/failure signal we already have. Forces
    the encoder to extract features that DISCRIMINATE by outcome,
    which is exactly the signal α needs to identify causal turns.

    Standard contrastive RL trick (CURL, ATC, DRIML).

    Args:
        out: TurnRDOutput with `encoder_hidden` populated `[B, T, H]`.
        R: `[B]` final scalar reward per trajectory.
        attention_mask: `[B, T]` 1 for real turns, 0 for padding.
        temperature: softmax temperature for the InfoNCE log-likelihood.
            Default 0.1. Lower = more peaked (harder positives + negatives).
        success_threshold: trajectories with R > this are "success",
            others are "failure". Default 0.0 (any positive reward).

    Returns:
        Scalar tensor — InfoNCE loss averaged over success-anchor turns.
        Returns 0 (no_grad) when there's no valid pair (e.g., all-success
        or all-failure batch with no contrast available).
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

    # Flatten (B, T) → (B*T) and filter unmasked turns only.
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
        # Need ≥2 successful turns (anchor + at least one positive)
        # and ≥1 failure turn (negative pool).
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
