"""LoRA-wrapped Qwen2.5-1.5B-Instruct policy for H-GRPO training.

Heavy module - torch / transformers / peft are required at runtime. Locally
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


class LoRAMergeNonFiniteError(RuntimeError):
    """Raised by `iter_merged_weights` when a merged LoRA weight is non-finite
    (inf/NaN), so callers can fail the round instead of syncing a bad model.
    """


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
    # Single visible GPU; "cuda:0" or "auto" both work.
    device_map: str = "cuda:0"
    # HF weight cache; a Modal Volume path avoids re-downloads.
    cache_dir: str | None = None


class LoRAPolicy:
    """Wraps `transformers.AutoModelForCausalLM` + PEFT LoRA.

    Exposes `tokenizer`, `model`, trainable-parameter accessors, merged-weight
    export for vLLM (`merged_state_dict`/`iter_merged_weights`), and
    `save_adapter`/`load_adapter`.
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

        # Gradient checkpointing to save memory (~30% extra compute).
        # use_reentrant=False is required for PEFT; enable_input_require_grads
        # keeps the recomputation chain reaching the LoRA params.
        try:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:  # older transformers signature
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    # Trainer-facing API

    def trainable_parameters(self) -> list[Any]:
        """Yield only the LoRA parameters (what AdamW should touch)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def merged_state_dict(self) -> dict[str, "torch.Tensor"]:
        """Materialize all merged-LoRA weights as a dict.

        Prefer iter_merged_weights() for vLLM sync; it streams one tensor at
        a time instead of holding the full state-dict in memory.
        """
        return dict(self.iter_merged_weights())

    def iter_merged_weights(self):
        """Lazily yield `(canonical_name, merged_tensor)` pairs.

        For each LoRA-wrapped Linear, computes base + (lora_B @ lora_A) *
        scaling on the fly without mutating the model or deepcopying. Peak
        memory is bounded by one layer since transients free before the next.
        """
        import torch  # type: ignore[import-not-found]

        from src.policy.weight_sync import canonicalize_lora_target_name

        # PEFT wraps each Linear in a LoraLayer holding .base_layer,
        # .lora_A/.lora_B[adapter], and .scaling[adapter].
        adapter_name = "default"

        # First pass: collect LoRA-wrapped module names so we skip their
        # .base_layer.weight below and yield the merged version instead.
        lora_module_names: set[str] = set()
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base_layer"):
                lora_module_names.add(name)

        # Second pass: yield each param, stripping the PEFT prefix.
        with torch.no_grad():
            for raw_name, param in self.model.named_parameters():
                # Skip pure-LoRA params; merged in the LoRA pass below.
                if ".lora_A." in raw_name or ".lora_B." in raw_name:
                    continue

                # Skip a LoRA-wrapped module's .base_layer.weight; the merged
                # version is yielded in the LoRA pass below.
                if ".base_layer." in raw_name:
                    parent = raw_name.split(".base_layer.")[0]
                    parent_short = parent[len("base_model.model.") :] if parent.startswith("base_model.model.") else parent
                    parent_full = "base_model.model." + parent_short
                    if parent_full in {"base_model.model." + n for n in lora_module_names} or parent in lora_module_names:
                        continue

                # Untouched param: yield directly (no copy).
                yield canonicalize_lora_target_name(raw_name), param.data

            # LoRA pass: materialize and yield each merged weight.
            for mod_name in lora_module_names:
                module = self.model.get_submodule(mod_name)
                base_layer = module.base_layer
                base_weight = base_layer.weight  # [out, in]
                lora_A = module.lora_A[adapter_name].weight  # [r, in]
                lora_B = module.lora_B[adapter_name].weight  # [out, r]
                scaling = float(module.scaling[adapter_name])
                # fp32-promote the matmul + add before downcasting (mirrors
                # PEFT's get_delta_weight); bf16 otherwise overflows to
                # inf/NaN for large LoRA-B entries after ~10-20 RL steps.
                delta = (lora_B.to(torch.float32) @ lora_A.to(torch.float32)) * scaling
                merged = (base_weight.to(torch.float32) + delta).to(base_weight.dtype)
                # Finite-check here (the true bug site); on failure the
                # module name + abs-max pinpoints the blown-up layer.
                if not torch.isfinite(merged).all():
                    raise LoRAMergeNonFiniteError(
                        f"iter_merged_weights: non-finite merged weight at "
                        f"{mod_name} (abs-max={float(merged.abs().max())}); "
                        f"likely LoRA-B gradient explosion — tighten "
                        f"max_grad_norm or add per-tensor LoRA-B clip."
                    )
                # mod_name already carries the PEFT prefix; don't double-prefix.
                full_name = f"{mod_name}.weight"
                yield canonicalize_lora_target_name(full_name), merged

    def save_adapter(self, path: str) -> None:
        """Save the LoRA adapter to `path`.

        Tries PEFT's save_pretrained first; on a CUDA error, copies trainable
        tensors to CPU and writes them with safetensors. Re-raises if the CUDA
        context is already invalid rather than writing a partial directory.
        """
        import torch  # local import: torch is heavy and only present in the runtime image

        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            # Re-raise deferred CUDA errors at the synchronization site.
            raise

        try:
            self.model.save_pretrained(path)
            return
        except RuntimeError as e:
            if "CUDA" not in str(e) and "cuda" not in str(e):
                raise  # not the CUDA-save bug; bubble up untouched
            print(
                f"[save_adapter] FAST path failed with CUDA error: {e!r}. "
                "Falling back to per-tensor CPU copy."
            )

        # Fallback path: per-tensor manual CPU copy + direct safetensors write
        import os
        import json
        from safetensors.torch import save_file as _st_save_file
        os.makedirs(path, exist_ok=True)

        # Per-tensor CPU copy; detach->clone->contiguous->cpu order matters
        # (independent, contiguous storage that safetensors can write).
        cpu_state: dict[str, "torch.Tensor"] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # LoRA-A, LoRA-B, modules_to_save
                try:
                    cpu_state[name] = (
                        param.data.detach().clone().contiguous().cpu()
                    )
                except RuntimeError as e:
                    # Re-raise rather than writing a partial adapter directory.
                    raise RuntimeError(
                        f"[save_adapter] fallback per-tensor copy also "
                        f"failed at param '{name}': {e!r}"
                    ) from e

        # Persist via safetensors (matches the format PEFT writes).
        adapter_file = os.path.join(path, "adapter_model.safetensors")
        _st_save_file(cpu_state, adapter_file)

        # Write adapter_config.json so load_adapter sees a local adapter.
        try:
            self.model.peft_config["default"].save_pretrained(path)
        except Exception as e:
            print(
                f"[save_adapter] WARN: peft_config.save_pretrained failed "
                f"({e!r}); writing minimal adapter_config.json fallback."
            )
            # Minimal fallback so load_adapter doesn't fall back to HF Hub.
            minimal = {
                "peft_type": "LORA",
                "base_model_name_or_path": str(self.cfg.model_name),
                "r": int(self.cfg.lora_r),
                "lora_alpha": int(self.cfg.lora_alpha),
                "target_modules": list(self.cfg.lora_target_modules),
            }
            with open(os.path.join(path, "adapter_config.json"), "w") as fh:
                json.dump(minimal, fh)
        print(f"[save_adapter] fallback CPU-write succeeded → {path}")

    def load_adapter(self, path: str) -> None:
        self.model.load_adapter(path, adapter_name="default")

    # Diagnostics

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
