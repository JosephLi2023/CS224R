"""LoRA policy + vLLM runner for the H-GRPO trainer.

Layout:
  src/policy/lora_policy.py — Qwen2.5-1.5B + PEFT LoRA, used by the trainer.
  src/policy/vllm_runner.py — vLLM `LLM` instance for batched rollout sampling.
  src/policy/weight_sync.py — pure-Python helpers for the LoRA-merge → vLLM
                              load_weights pipeline (unit-testable without torch).
"""

from src.policy.weight_sync import (
    canonicalize_lora_target_name,
    is_lora_param_name,
    plan_weight_sync,
    strip_peft_prefix,
)

__all__ = [
    "canonicalize_lora_target_name",
    "is_lora_param_name",
    "plan_weight_sync",
    "strip_peft_prefix",
]
