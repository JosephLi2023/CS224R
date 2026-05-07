"""Production policy-hidden-state embedder for the TurnRD decomposer.

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
policy, not the frozen base. (A "frozen-backbone" variant could be added
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
                # H3 v11 memory fix: bypass the LM head entirely by
                # walking down to the bare transformer backbone. For
                # PEFT-wrapped CausalLM:
                #   PeftModel.base_model = LoraModel
                #   LoraModel.model      = AutoModelForCausalLM (LoRA-wrapped)
                #   CausalLM.model       = bare transformer (Qwen2Model)
                # The bare transformer returns BaseModelOutputWithPast
                # which has `.last_hidden_state` directly, so we don't
                # need `output_hidden_states=True` (which builds a
                # 29-layer tuple — ~0.5-2 GB transient that fragments
                # the allocator under PPO grad pressure). The LoRA
                # adapter is STILL applied because LoraModel only wraps
                # the linear submodules — calling .model.model still
                # routes through them. We don't disable_adapter() here;
                # we want hiddens from the trained policy, not the
                # frozen base.
                backbone = model
                # PEFT unwrap (no-op if model is already a bare HF model)
                backbone = getattr(backbone, "base_model", backbone)
                backbone = getattr(backbone, "model", backbone)
                # CausalLM unwrap — `.model` on AutoModelForCausalLM
                # points to the bare transformer.
                _bare = getattr(backbone, "model", None)
                if _bare is not None and hasattr(_bare, "forward"):
                    backbone = _bare
                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        finally:
            if was_training:
                model.train()

        # The bare transformer returns BaseModelOutputWithPast with
        # `.last_hidden_state` populated. Fall back to .hidden_states[-1]
        # only if a stub model returns the older format (test path).
        if getattr(outputs, "last_hidden_state", None) is not None:
            hidden = outputs.last_hidden_state  # [T, L, D]
        elif getattr(outputs, "hidden_states", None) is not None:
            hidden = outputs.hidden_states[-1]  # [T, L, D]
        else:
            # Last resort: outputs[0] — only safe when we're certain the
            # callee returned hidden states (e.g., test stub). Will be
            # the wrong tensor (logits) if we're wrong about the model
            # type. Guard with shape check downstream — input_dim
            # mismatch raises in TurnRD.forward().
            hidden = outputs[0]
        # Detach + ensure no autograd graph leaks into the no_grad path.
        hidden = hidden.detach() if hidden.requires_grad else hidden
        del outputs

        # Mean-pool over L using the attention mask.
        # mask: [T, L] long → cast to hidden's dtype for the multiply.
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)  # [T, L, 1]
        summed = (hidden * mask).sum(dim=1)  # [T, D]
        denom = mask.sum(dim=1).clamp_min(1)  # [T, 1]
        pooled = summed / denom  # [T, D]

        # Return CPU fp32. The adapter casts to model dtype/device.
        return pooled.detach().to(device="cpu", dtype=torch.float32)

    return _embed
