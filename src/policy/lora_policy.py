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

        # H1 (v11) memory fix: enable HF gradient checkpointing on the
        # PEFT-wrapped policy. PPO under K=8 microbatched forwards in
        # `_batched_logprobs` retained 30-50 GiB of activations until
        # `backward()` (every microbatch's `grad_fn` chain stays alive
        # in the `out[]` collection). Gradient checkpointing trades
        # ~30% extra compute for re-running the forward during
        # backward, freeing those activations between forward and
        # backward. `use_reentrant=False` is required for PEFT
        # (the reentrant variant breaks LoRA gradient flow).
        # `enable_input_require_grads()` ensures the embedded input has
        # `requires_grad=True` so the recomputation chain reaches LoRA
        # params (the frozen base would otherwise short-circuit grad
        # through the input embeddings).
        try:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:  # older transformers signature
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

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
        """Materialize all merged-LoRA weights as a single dict.

        Backward-compat wrapper around `iter_merged_weights()`. Prefer the
        iterator form for vLLM weight sync — it streams one tensor at a
        time and avoids holding the full base+LoRA state-dict in memory
        simultaneously.
        """
        return dict(self.iter_merged_weights())

    def iter_merged_weights(self):
        """Lazy generator yielding `(canonical_name, merged_tensor)` pairs.

        For each PEFT-LoRA-wrapped Linear, computes the merged weight on
        the fly as `base + (lora_B @ lora_A) * scaling` WITHOUT modifying
        the live model and WITHOUT a full deepcopy. Each merged tensor
        exists only briefly during `yield`; vLLM's `model.load_weights`
        copies the data into its own buffer, then the tensor is freed
        before the next iteration.

        Memory profile: peak transient ≈ 3× one LoRA target's weight
        (~18 MB for a Qwen-1.5B q_proj at hidden=1536), instead of the
        ~3 GB transient of the deepcopy path.
        """
        import torch  # type: ignore[import-not-found]

        from src.policy.weight_sync import canonicalize_lora_target_name

        # Map module-name → PEFT LoRA layer (if any). PEFT wraps a Linear
        # with a LoraLayer that holds .base_layer (the original weight) and
        # .lora_A.<adapter>, .lora_B.<adapter>, plus .scaling[adapter].
        adapter_name = "default"

        # First pass: build a set of module names that ARE LoRA-wrapped, so
        # we can skip their .base_layer.weight in the second pass and yield
        # the merged version instead.
        lora_module_names: set[str] = set()
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base_layer"):
                lora_module_names.add(name)

        # Second pass: walk every parameter and decide what to yield.
        # PEFT puts everything under "base_model.model." prefix; we strip
        # that via canonicalize_lora_target_name.
        with torch.no_grad():
            for raw_name, param in self.model.named_parameters():
                # Skip pure-LoRA params (lora_A/lora_B); they are merged
                # into the base layer in the LoRA pass below.
                if ".lora_A." in raw_name or ".lora_B." in raw_name:
                    continue

                # Skip ANY .base_layer.weight whose parent module is LoRA-wrapped;
                # we yield the merged version below instead.
                # raw_name like 'base_model.model.<...>.q_proj.base_layer.weight'
                # → its parent module name is everything up to '.base_layer.<...>'
                if ".base_layer." in raw_name:
                    parent = raw_name.split(".base_layer.")[0]
                    parent_short = parent[len("base_model.model.") :] if parent.startswith("base_model.model.") else parent
                    parent_full = "base_model.model." + parent_short
                    if parent_full in {"base_model.model." + n for n in lora_module_names} or parent in lora_module_names:
                        continue

                # Untouched param (norms, embed, lm_head, mlp not in target_modules):
                # canonicalize the name and yield the param data directly (no copy).
                yield canonicalize_lora_target_name(raw_name), param.data

            # LoRA pass: for each wrapped module, compute the merged weight
            # and yield it. This is the only place where a transient tensor
            # is materialized.
            for mod_name in lora_module_names:
                module = self.model.get_submodule(mod_name)
                base_layer = module.base_layer
                base_weight = base_layer.weight  # [out, in]
                lora_A = module.lora_A[adapter_name].weight  # [r, in]
                lora_B = module.lora_B[adapter_name].weight  # [out, r]
                scaling = float(module.scaling[adapter_name])
                # delta = (B @ A) * scaling — one tensor of base_weight shape
                delta = (lora_B @ lora_A) * scaling
                merged = base_weight + delta
                # `mod_name` from named_modules() already carries the full
                # `base_model.model.<...>` PEFT prefix (since `self.model` IS
                # the wrapped PeftModel). Don't double-prefix here.
                full_name = f"{mod_name}.weight"
                yield canonicalize_lora_target_name(full_name), merged
                # `merged` and `delta` go out of scope here → freed before
                # next iteration.

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
