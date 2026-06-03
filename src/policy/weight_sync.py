"""Pure-Python (torch-free) helpers for LoRA -> vLLM weight synchronization.

PEFT stores params like base_model.model.<...>.q_proj.base_layer.weight (plus
lora_A/lora_B), while vLLM expects <...>.q_proj.weight. These helpers strip the
prefix and detect LoRA pairs so a merged state dict becomes a (name, tensor)
iterator for load_weights.
"""

from __future__ import annotations

# Standard PEFT prefix prepended when wrapping a base model with LoraModel.
_PEFT_PREFIX = "base_model.model."


def strip_peft_prefix(name: str) -> str:
    """Drop the leading `base_model.model.` PEFT prefix if present (idempotent)."""
    if name.startswith(_PEFT_PREFIX):
        return name[len(_PEFT_PREFIX) :]
    return name


def is_lora_param_name(name: str) -> bool:
    """True for PEFT-LoRA-only params (lora_A/lora_B/lora_embedding_*), which
    must be merged into the base weight before going to vLLM."""
    s = name
    return (
        ".lora_A." in s
        or ".lora_B." in s
        or ".lora_embedding_A." in s
        or ".lora_embedding_B." in s
    )


def canonicalize_lora_target_name(name: str) -> str:
    """Convert a PEFT base-layer param name to the canonical model name.

    Strips the PEFT prefix and the `.base_layer.` segment (e.g.
    base_model.model.<...>.q_proj.base_layer.weight -> <...>.q_proj.weight),
    returning the name unchanged when neither is present.
    """
    name = strip_peft_prefix(name)
    return name.replace(".base_layer.", ".")


def plan_weight_sync(state_dict_keys: list[str]) -> dict[str, list[str]]:
    """Group a PEFT-wrapped state-dict's keys into passthrough (base layers),
    lora_pair (lora_A/lora_B to merge), and skipped (everything else).

    Pure inspection; useful for asserting the expected merged-layer count.
    """
    out: dict[str, list[str]] = {"passthrough": [], "lora_pair": [], "skipped": []}
    for key in state_dict_keys:
        if is_lora_param_name(key):
            out["lora_pair"].append(key)
        elif ".base_layer." in key or key.startswith(_PEFT_PREFIX):
            # Standard wrapped weight - passes through after canonicalization.
            out["passthrough"].append(key)
        else:
            # Other keys (e.g. modules_to_save) aren't part of load_weights.
            out["skipped"].append(key)
    return out
