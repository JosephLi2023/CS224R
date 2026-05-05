"""Production policy-hidden-state embedder for the TurnRD decomposer (Day 13).

`policy_hidden_state_embedder(policy, *, max_len_per_turn=512)` returns a
closure that satisfies the `TurnEmbedder` contract enforced by
`src/algorithms/hgpo/decomposers/turnrd.py` (the adapter):

    TurnEmbedder = Callable[[Trajectory], torch.Tensor]      # returns [T_i, D]

For each `Trajectory`:
1. Build per-turn span text `f"{turn.observation_text}\\n{turn.action_text}"`
   (the "what happened on this turn" span; we don't reuse
   `prompt_token_ids` because that's the cumulative React prompt, much
   wider than the per-turn span).
2. Tokenize all T spans with the policy's tokenizer (truncate to
   `max_len_per_turn`).
3. Forward through `policy.model` under `torch.no_grad()` + `eval()` with
   `output_hidden_states=True`; read the last-layer hidden state.
4. Mean-pool over the L axis with the attention mask (padded positions
   contribute zero, denominator clamped to ≥1 to avoid div-by-zero on
   single-token spans).
5. Return `[T_i, D]` on CPU as float32. The decomposer's adapter
   normalises dtype/device before forward (see `TurnRDDecomposer.__init__`),
   so the embedder is free to return CPU fp32 even when the policy lives
   on a CUDA bf16 device.

The LoRA adapter stays ENABLED — we want hiddens from the current trained
policy, not the frozen base. (Day 14+ may add a "frozen-backbone" variant
if a frozen-embedder experiment is desired.)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from src.algorithms.grpo.rollout import Trajectory

if TYPE_CHECKING:
    from src.policy.lora_policy import LoRAPolicy


# Match the type alias used by the decomposer adapter.
TurnEmbedder = Callable[[Trajectory], torch.Tensor]


def policy_hidden_state_embedder(
    policy: "LoRAPolicy",
    *,
    max_len_per_turn: int = 512,
) -> TurnEmbedder:
    """Build a `TurnEmbedder` that mean-pools last-layer hidden states.

    Args:
        policy: a `LoRAPolicy` whose `.tokenizer` and `.model` attributes
            are honored. `.model` must be a `transformers.AutoModelForCausalLM`
            (or PEFT wrapper) that supports `output_hidden_states=True`.
        max_len_per_turn: per-turn tokenization truncation bound. 512 is
            enough for typical WebShop turns (~100 tokens of obs + ~50
            tokens of action); raise if your env produces longer spans.

    Returns:
        A function that maps a `Trajectory` to a `[T_i, D]` CPU fp32
        tensor of mean-pooled last-layer hidden states (one row per turn).
    """
    if max_len_per_turn <= 0:
        raise ValueError(
            f"policy_hidden_state_embedder: max_len_per_turn must be positive; "
            f"got {max_len_per_turn}."
        )

    tokenizer = policy.tokenizer
    model = policy.model

    def _embed(traj: Trajectory) -> torch.Tensor:
        T = len(traj.turns)
        if T == 0:
            # Defensive: the decomposer adapter already short-circuits empty
            # trajectories, so this only runs if a direct caller calls the
            # embedder. Hidden size lookup needs SOMETHING to forward, and
            # an empty batch trips most HF models. Match the adapter's
            # contract by raising rather than fabricating a hidden_size.
            raise ValueError(
                f"policy_hidden_state_embedder: trajectory {traj.task_id!r} "
                "has no turns; the decomposer adapter normally filters these "
                "before calling the embedder."
            )

        spans = [
            f"{turn.observation_text}\n{turn.action_text}" for turn in traj.turns
        ]
        enc = tokenizer(
            spans,
            padding=True,
            truncation=True,
            max_length=max_len_per_turn,
            return_tensors="pt",
        )
        # `next(model.parameters())` is the canonical "where does this
        # model live?" idiom — it works for both bare HF models and PEFT
        # wrappers without needing to know the wrapper's internals.
        try:
            target_device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover (extremely unusual)
            target_device = torch.device("cpu")
        input_ids = enc["input_ids"].to(target_device)
        attention_mask = enc["attention_mask"].to(target_device)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        finally:
            if was_training:
                model.train()

        # `outputs.hidden_states` is a tuple `(layer_0, ..., layer_N)` of
        # tensors `[T, L, D]`. v6 improvement: average across multiple
        # layers (last + 3 earlier) instead of just the last layer.
        # Lower layers carry more positional/syntactic info that helps
        # TurnRD's encoder discriminate "search" from "click[item-N]"
        # — which the deepest layer's task-specialized features may have
        # smoothed over. Averaging keeps the output dim at D (no
        # downstream input_dim change required).
        all_hidden = outputs.hidden_states  # tuple of [T, L, D] tensors
        n_layers = len(all_hidden)
        # Pick layers from depth-quartiles: last (-1), then 3 earlier
        # at evenly-spaced positions. Skip the embedding layer (idx 0)
        # which is just token embeddings without contextualization.
        if n_layers >= 13:  # Qwen-1.5B has 28 layers + embedding = 29
            layer_idxs = [-1, -7, -14, -21]
        elif n_layers >= 5:
            # Smaller models / unit-test stubs: just last 4 layers.
            layer_idxs = [-1, -2, -3, -4]
        else:
            layer_idxs = [-1]
        # Stack picked layers along a new dim and average.
        hidden = torch.stack(
            [all_hidden[i] for i in layer_idxs], dim=0
        ).mean(dim=0)  # [T, L, D]

        # Mean-pool over L using the attention mask.
        # mask: [T, L] long → cast to hidden's dtype for the multiply.
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)  # [T, L, 1]
        summed = (hidden * mask).sum(dim=1)  # [T, D]
        denom = mask.sum(dim=1).clamp_min(1)  # [T, 1]
        pooled = summed / denom  # [T, D]

        # Return CPU fp32. The adapter casts to model dtype/device.
        return pooled.detach().to(device="cpu", dtype=torch.float32)

    return _embed
