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


class LoRAMergeNonFiniteError(RuntimeError):
    """Raised by `iter_merged_weights` when a merged LoRA weight contains
    non-finite entries (inf/NaN).

    This is a distinct subclass of `RuntimeError` so callers can catch it
    by type and escalate (vs. silently swallowing it as a transient
    per-episode crash). In particular, the per-episode try/except in
    `infra/app_train_loop.py` re-raises this class to fail the entire
    round, because a partial vLLM `load_weights` leaves the engine in a
    half-updated "split-brain" state where the first N layers carry the
    new (bad) weights and the remaining layers carry stale weights —
    every subsequent rollout in the round would then run against a
    corrupted policy. Failing fast lets `app_aggregate_alfworld` skip the
    round cleanly via the missing-eval-block path.
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

        Memory profile: per-layer transient peaks around 5–6× one LoRA
        target's bf16 weight during the fp32-promoted merge expression
        (intermediate fp32 base + fp32 sum + bf16 downcast all alive
        briefly), then drops to ~3× across the `yield` (live: fp32
        `delta` + bf16 `merged` + the shared bf16 base). Concretely for a
        Qwen-1.5B `down_proj` (8960×1536, ~14M params): ~165 MB peak
        during the merge expression, ~83 MB held across `yield`. For a
        q_proj (1536×1536, ~2.4M params): ~28 MB peak, ~14 MB across
        yield. Compare to the ~3 GB transient of the deepcopy path.
        Transients are released before the next layer's merge, so the
        full-sync cost is bounded by ONE layer, not the sum.
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
                # Promote the matmul accumulator + add to fp32 BEFORE casting
                # back to the base dtype. Mirrors PEFT's own
                # `get_delta_weight()` and the codebase's fp32 promotion of
                # logits before log_softmax
                # (src/algorithms/grpo/trainer.py:363-365). Without this,
                # rank-32 MLP+attn produces inf/NaN entries after ~10-20 RL
                # steps because bf16's ~3 mantissa decimals can't represent
                # (B @ A) * scaling for large LoRA-B entries; vLLM then
                # silent-copies the bad tensor and the next forward pass
                # dies as "CUDA illegal memory access" far from the true
                # site (root cause of the AlfWorld SOTA 8-round R0 crashes
                # at ep=9/19/21, 2026-05-21).
                delta = (lora_B.to(torch.float32) @ lora_A.to(torch.float32)) * scaling
                merged = (base_weight.to(torch.float32) + delta).to(base_weight.dtype)
                # Finite-check at the true bug site (the LoRA layer), not
                # at vLLM's async forward. Cheap (~1µs on-GPU); will not
                # fire post-fp32 promotion for well-behaved weights. If it
                # DOES fire, the module name + abs-max value tells us
                # exactly which layer blew up — invaluable diagnostic for
                # any residual gradient-explosion issue.
                if not torch.isfinite(merged).all():
                    raise LoRAMergeNonFiniteError(
                        f"iter_merged_weights: non-finite merged weight at "
                        f"{mod_name} (abs-max={float(merged.abs().max())}); "
                        f"likely LoRA-B gradient explosion — tighten "
                        f"max_grad_norm or add per-tensor LoRA-B clip."
                    )
                # `mod_name` from named_modules() already carries the full
                # `base_model.model.<...>` PEFT prefix (since `self.model` IS
                # the wrapped PeftModel). Don't double-prefix here.
                full_name = f"{mod_name}.weight"
                yield canonicalize_lora_target_name(full_name), merged
                # `merged` and `delta` go out of scope here → freed before
                # next iteration.

    def save_adapter(self, path: str) -> None:
        """Save the LoRA adapter to `path`.

        Hardened against the R4 CUDA-illegal-memory bug seen during the
        AlfWorld SOTA 8-round run (TurnRDV2_alfworld_SOTA_8round, 2026-05-20):
        after 4 carry-policy rounds of merge-then-load LoRA, the default
        `model.save_pretrained` path crashes at
            safetensors.torch._tobytes → tensor.to("cpu")
        with `CUDA error: an illegal memory access was encountered`.

        The crash is async-reported (the real fault happened earlier in the
        round), so we:
          1. `torch.cuda.synchronize()` first — surfaces the real op that
             corrupted CUDA state in the traceback instead of the .to("cpu")
             at save time. (No effect if CUDA is clean.)
          2. `torch.cuda.empty_cache()` — releases inactive blocks; can
             defuse memory pressure that may be feeding the bug.
          3. Try `model.save_pretrained(path)` first (fast, the normal path).
          4. On CUDA failure, fall back to manually moving each LoRA
             parameter to CPU one-by-one with explicit
             `.detach().cpu().contiguous().clone()` and write a temporary
             CPU-only PeftModel state_dict via safetensors directly. This
             bypasses the bulk-tensor `_flatten` call that triggers the bug.

        The fallback is best-effort: if the underlying CUDA context is
        actually corrupted (not just memory-pressed), the per-tensor copy
        will also fail and the original exception is re-raised so the
        orchestrator's crash-detection patch can fail loudly instead of
        silently cascading to the next round (the bug we hit twice on R5+).
        """
        import torch  # local import: torch is heavy and only present in the runtime image

        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            # Synchronize itself can raise the deferred CUDA error here —
            # that's actually the desired diagnostic. Re-raise so the
            # caller sees the real source.
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

        # ---- Fallback path: per-tensor manual CPU copy + direct safetensors write
        import os
        import json
        from safetensors.torch import save_file as _st_save_file
        os.makedirs(path, exist_ok=True)

        # Collect LoRA + trainable params with explicit per-tensor moves.
        # `.detach().clone().contiguous().cpu()` order matters: detach first
        # (no grad fn), clone (independent storage so partial fault doesn't
        # corrupt others), contiguous (safetensors requires it), .cpu() last.
        cpu_state: dict[str, "torch.Tensor"] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # LoRA-A, LoRA-B, modules_to_save
                try:
                    cpu_state[name] = (
                        param.data.detach().clone().contiguous().cpu()
                    )
                except RuntimeError as e:
                    # If this also fails, the CUDA context is truly cooked.
                    # Re-raise so the orchestrator gets a clean failure
                    # instead of writing a half-baked adapter dir.
                    raise RuntimeError(
                        f"[save_adapter] fallback per-tensor copy also "
                        f"failed at param '{name}': {e!r}"
                    ) from e

        # Persist via safetensors (matches the format PEFT writes).
        adapter_file = os.path.join(path, "adapter_model.safetensors")
        _st_save_file(cpu_state, adapter_file)

        # Write adapter_config.json so peft.load_adapter() recognizes the
        # directory as a LoRA adapter (not an HF Hub repo id — which is
        # exactly the cascading failure mode that bit R5..R7).
        # peft writes this via PeftConfig.save_pretrained; we mimic the
        # JSON-only path here without touching CUDA again.
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
