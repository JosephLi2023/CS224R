"""Production policy-hidden-state embedder for the TurnRD decomposer.

`policy_hidden_state_embedder(policy, *, max_len_per_turn=512)` returns a
closure satisfying the `TurnEmbedder` contract from
`src/algorithms/hgpo/decomposers/turnrd.py`:

    TurnEmbedder = Callable[[Trajectory], torch.Tensor]      # returns [T_i, D]

For each `Trajectory`:
1. Build per-turn span text `f"{turn.observation_text}\\n{turn.action_text}"`.
2. Tokenize all T spans (truncate to `max_len_per_turn`).
3. Forward through the policy backbone under `torch.no_grad()` + `eval()`;
   read the last-layer hidden state.
4. Mean-pool over the L axis with the attention mask (denominator clamped to
   >=1).
5. Return `[T_i, D]` on CPU as float32; the adapter normalizes dtype/device.

The LoRA adapter stays ENABLED - hiddens come from the trained policy, not the
frozen base.
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
        policy: a `LoRAPolicy` whose `.tokenizer` and `.model` are used;
            `.model` is an `AutoModelForCausalLM` (or PEFT wrapper).
        max_len_per_turn: per-turn tokenization truncation bound (default 512).

    Returns:
        A function mapping a `Trajectory` to a `[T_i, D]` CPU fp32 tensor of
        mean-pooled last-layer hidden states (one row per turn).
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
            # Defensive: empty trajectories are normally filtered by the adapter;
            # raise rather than forward an empty batch.
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
        # `next(model.parameters())` to find the model's device (works for bare
        # HF models and PEFT wrappers).
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
                # Bypass the LM head by walking to the bare transformer backbone,
                # which returns `.last_hidden_state` directly (avoids the
                # output_hidden_states tuple - a large transient). Unwrap chain:
                #   PeftModel.base_model = LoraModel
                #   LoraModel.model      = AutoModelForCausalLM (LoRA-wrapped)
                #   CausalLM.model       = bare transformer
                # LoRA stays applied (it wraps the linear submodules).
                backbone = model
                # PEFT unwrap (no-op if model is already a bare HF model)
                backbone = getattr(backbone, "base_model", backbone)
                backbone = getattr(backbone, "model", backbone)
                # CausalLM unwrap: `.model` points to the bare transformer.
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
            # Last resort: outputs[0] (test stubs). Wrong if the callee
            # returned logits; an input_dim mismatch then raises downstream.
            hidden = outputs[0]
        # Detach + ensure no autograd graph leaks into the no_grad path.
        hidden = hidden.detach() if hidden.requires_grad else hidden
        del outputs

        # Mean-pool over L using the attention mask.
        # mask: [T, L] long -> cast to hidden's dtype for the multiply.
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)  # [T, L, 1]
        summed = (hidden * mask).sum(dim=1)  # [T, D]
        denom = mask.sum(dim=1).clamp_min(1)  # [T, 1]
        pooled = summed / denom  # [T, D]

        # Return CPU fp32. The adapter casts to model dtype/device.
        return pooled.detach().to(device="cpu", dtype=torch.float32)

    return _embed
