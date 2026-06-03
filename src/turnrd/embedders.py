"""Production policy-hidden-state embedder for the TurnRD decomposer.

`policy_hidden_state_embedder(policy, *, max_len_per_turn=512)` returns a
`TurnEmbedder` (`Callable[[Trajectory], torch.Tensor]`). For each trajectory it
tokenizes per-turn observation+action spans, forwards them through the policy
backbone under no_grad, and mean-pools the last-layer hidden state over tokens
to a `[T_i, D]` CPU fp32 tensor. LoRA stays enabled (hiddens from the trained
policy).
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
    """Build a `TurnEmbedder` mapping a `Trajectory` to a `[T_i, D]` CPU fp32
    tensor of mean-pooled last-layer hidden states. `max_len_per_turn` bounds
    per-turn tokenization (default 512).
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
            # Empty trajectories are normally filtered upstream; raise here.
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
        # Model device (works for bare HF and PEFT wrappers).
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
                # Walk to the bare transformer backbone (returns last_hidden_state
                # directly, avoiding the output_hidden_states tuple). Unwrap:
                #   PeftModel.base_model -> LoraModel.model -> CausalLM.model.
                # LoRA stays applied (it wraps the linear submodules).
                backbone = model
                # PEFT unwrap (no-op for bare HF models).
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

        # Prefer `.last_hidden_state`; fall back to `.hidden_states[-1]` for
        # stub models (test path).
        if getattr(outputs, "last_hidden_state", None) is not None:
            hidden = outputs.last_hidden_state  # [T, L, D]
        elif getattr(outputs, "hidden_states", None) is not None:
            hidden = outputs.hidden_states[-1]  # [T, L, D]
        else:
            # Last resort: outputs[0] (test stubs); a wrong tensor raises downstream.
            hidden = outputs[0]
        # Detach (no autograd graph in the no_grad path).
        hidden = hidden.detach() if hidden.requires_grad else hidden
        del outputs

        # Mean-pool over L using the attention mask (cast to hidden dtype).
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)  # [T, L, 1]
        summed = (hidden * mask).sum(dim=1)  # [T, D]
        denom = mask.sum(dim=1).clamp_min(1)  # [T, 1]
        pooled = summed / denom  # [T, D]

        # Return CPU fp32 (adapter recasts).
        return pooled.detach().to(device="cpu", dtype=torch.float32)

    return _embed
