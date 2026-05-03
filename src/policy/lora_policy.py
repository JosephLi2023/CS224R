"""LoRA-wrapped Qwen2.5-1.5B-Instruct policy for H-GRPO training.

Heavy module — torch / transformers / peft are required at runtime. Locally
on macOS arm64 these aren't installed; `import` is therefore guarded so the
rest of `src.policy.*` (pure-Python helpers) stays loadable for unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import AutoTokenizer, PreTrainedTokenizer  # noqa: F401  # type: ignore[import-not-found]


@dataclass
class LoRAPolicyConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"            # "float16" | "bfloat16" | "float32"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # device_map="auto" pushes everything onto a single GPU when only one is visible;
    # explicit "cuda:0" works too if you want to be explicit.
    device_map: str = "cuda:0"
    # Where HF caches downloaded weights. Setting this to a Modal Volume path
    # avoids re-downloading on every container start.
    cache_dir: str | None = None


class LoRAPolicy:
    """Wraps `transformers.AutoModelForCausalLM` + PEFT LoRA.

    Public surface:
      - `tokenizer` — HF tokenizer (PreTrainedTokenizer)
      - `model`    — peft.PeftModel (the trainable wrapper)
      - `parameters()` — yields trainable LoRA parameters (for AdamW)
      - `merged_state_dict()` — non-destructive merge → state-dict ready for vLLM
      - `save_adapter(path)` / `load_adapter(path)` — PEFT-style adapter persistence
    """

    def __init__(self, cfg: LoRAPolicyConfig) -> None:
        # Imports inline so the module loads on Mac (where these aren't installed).
        import torch  # noqa: F401
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        self.cfg = cfg

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[cfg.dtype]

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, cache_dir=cfg.cache_dir, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            # Qwen tokenizer ships without a pad token; reuse eos.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch_dtype,
            cache_dir=cfg.cache_dir,
            device_map=cfg.device_map,
        )
        # Keep base in train-eval-mode-aware state; gradients only flow into LoRA.
        base.requires_grad_(False)

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model: PeftModel = get_peft_model(base, lora_cfg)

    # ------------------------------------------------------------------
    # Trainer-facing API
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> list[Any]:
        """Yield only the LoRA parameters (what AdamW should touch)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def merged_state_dict(self) -> dict[str, "torch.Tensor"]:
        """Non-destructive merge of LoRA into base, returns a state-dict view.

        The merge happens on a *clone* so the live model is unaffected (we
        still need the LoRA params for the next optimizer step). Keys are
        canonicalized via `src.policy.weight_sync.canonicalize_lora_target_name`
        so the result is shaped exactly like a plain Qwen base-model state-dict
        and can be passed to `vllm_engine.model.load_weights(...)`.
        """
        from copy import deepcopy

        from src.policy.weight_sync import (
            canonicalize_lora_target_name,
            is_lora_param_name,
        )

        # `merge_and_unload` IS destructive, so do it on a deep copy.
        # NOTE: this duplicates the model in GPU memory briefly; for
        # Qwen2.5-1.5B (~3 GB bf16) this is comfortably under the A100-80GB
        # ceiling. If we move to larger models later, we'd switch to a
        # memory-efficient merge path.
        clone = deepcopy(self.model)
        merged = clone.merge_and_unload()
        raw = merged.state_dict()

        out: dict[str, Any] = {}
        for k, v in raw.items():
            if is_lora_param_name(k):
                # Should not happen after merge_and_unload, but guard.
                continue
            out[canonicalize_lora_target_name(k)] = v
        return out

    def save_adapter(self, path: str) -> None:
        self.model.save_pretrained(path)

    def load_adapter(self, path: str) -> None:
        self.model.load_adapter(path, adapter_name="default")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "model_name": self.cfg.model_name,
            "dtype": self.cfg.dtype,
            "trainable_params": self.trainable_param_count(),
            "total_params": self.total_param_count(),
            "trainable_pct": (
                100.0 * self.trainable_param_count() / max(1, self.total_param_count())
            ),
            "lora_r": self.cfg.lora_r,
            "lora_alpha": self.cfg.lora_alpha,
            "lora_target_modules": list(self.cfg.lora_target_modules),
        }
