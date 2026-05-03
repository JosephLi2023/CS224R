"""Pure-Python helpers for LoRA → vLLM weight synchronization.

The PEFT-wrapped policy stores parameters under names like:

  base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
  base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
  base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight

The underlying base model (and vLLM's runtime model) expects:

  model.layers.0.self_attn.q_proj.weight

This module provides the prefix-stripping + LoRA-pair detection logic so the
trainer can convert a merged-LoRA state dict into a `(name, tensor)` iterator
that `vllm_engine.model.load_weights(...)` accepts.

Helpers here are torch-free so we can unit-test the renaming logic without a
GPU.
"""

from __future__ import annotations

# Standard PEFT prefix prepended when wrapping a base model with LoraModel.
_PEFT_PREFIX = "base_model.model."


def strip_peft_prefix(name: str) -> str:
    """Drop the leading `base_model.model.` PEFT wrapper prefix if present.

    Idempotent: `strip_peft_prefix(strip_peft_prefix(x)) == strip_peft_prefix(x)`.
    """
    if name.startswith(_PEFT_PREFIX):
        return name[len(_PEFT_PREFIX) :]
    return name


def is_lora_param_name(name: str) -> bool:
    """Return True for PEFT-LoRA-only parameters (lora_A / lora_B / lora_embedding_*).

    These should NOT be passed verbatim to vLLM; they need to be merged into
    the corresponding base-layer weight first.
    """
    s = name
    return (
        ".lora_A." in s
        or ".lora_B." in s
        or ".lora_embedding_A." in s
        or ".lora_embedding_B." in s
    )


def canonicalize_lora_target_name(name: str) -> str:
    """Convert a PEFT base-layer parameter name into the canonical model name.

    PEFT renames the wrapped Linear's underlying weight as `<module>.base_layer.weight`.
    vLLM expects `<module>.weight`. This helper handles both halves of the rename:

      base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
        →  model.layers.0.self_attn.q_proj.weight

      base_model.model.lm_head.weight
        →  lm_head.weight

    Returns the name unchanged if it doesn't carry the PEFT prefix or
    `.base_layer.` segment.
    """
    name = strip_peft_prefix(name)
    return name.replace(".base_layer.", ".")


def plan_weight_sync(state_dict_keys: list[str]) -> dict[str, list[str]]:
    """Group a PEFT-wrapped state-dict's keys for vLLM weight sync.

    Produces a manifest the trainer can use to validate its merge step:

        {
          "passthrough": [...]      # base layers — pass directly to vLLM with renamed key
          "lora_pair":   [...]      # lora_A / lora_B that need merging into the matching base layer
          "skipped":     [...]      # PEFT bookkeeping (e.g. modules_to_save) we ignore for sync
        }

    Pure inspection: does NOT modify tensors. Useful for asserting at runtime
    that `merge_and_unload()` produced the expected number of merged layers.
    """
    out: dict[str, list[str]] = {"passthrough": [], "lora_pair": [], "skipped": []}
    for key in state_dict_keys:
        if is_lora_param_name(key):
            out["lora_pair"].append(key)
        elif ".base_layer." in key or key.startswith(_PEFT_PREFIX):
            # Standard wrapped weight — passes through after canonicalization.
            out["passthrough"].append(key)
        else:
            # Anything else (e.g. modules_to_save buffers) is not part of the
            # base-model load_weights contract.
            out["skipped"].append(key)
    return out
