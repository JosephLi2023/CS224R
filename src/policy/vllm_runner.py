"""vLLM runner: batched K-sample text generation + in-process weight sync.

Owns a `vllm.LLM` instance pointed at the same backbone the trainer uses
(Qwen2.5-1.5B-Instruct by default). After every optimizer step the trainer
calls `sync_weights(state_dict_iter)` to push the merged-LoRA weights into
the running engine so subsequent rollouts reflect the policy update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import torch
    from vllm import LLM  # type: ignore[import-not-found]


@dataclass
class VLLMRunnerConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.50  # leave room for trainer model on the same GPU
    seed: int = 0
    enforce_eager: bool = False
    download_dir: str | None = None  # HF download cache; align with LoRAPolicy.cache_dir


@dataclass
class SamplingParams:
    n: int = 4               # how many trajectories per prompt (K from proposal §3.1)
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 256
    stop: list[str] = field(default_factory=list)


class VLLMRunner:
    """Batched generator + weight-sync wrapper around `vllm.LLM`."""

    def __init__(self, cfg: VLLMRunnerConfig) -> None:
        from vllm import LLM  # type: ignore[import-not-found]

        self.cfg = cfg
        self.llm: LLM = LLM(
            model=cfg.model_name,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            seed=cfg.seed,
            enforce_eager=cfg.enforce_eager,
            download_dir=cfg.download_dir,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompts: list[str], sampling: SamplingParams) -> list[list[str]]:
        """Return list[len(prompts)] of list[sampling.n] generations."""
        from vllm import SamplingParams as VLLMSamplingParams  # type: ignore[import-not-found]

        params = VLLMSamplingParams(
            n=sampling.n,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            stop=sampling.stop or None,
            seed=self.cfg.seed,
        )
        outputs = self.llm.generate(prompts, params)
        # vLLM returns one RequestOutput per prompt; each has .outputs which
        # contains .n CompletionOutputs.
        return [[o.text for o in req.outputs] for req in outputs]

    # ------------------------------------------------------------------
    # Weight sync (the core Day-3 deliverable)
    # ------------------------------------------------------------------

    def sync_weights(self, state_dict: dict[str, "torch.Tensor"]) -> dict[str, int]:
        """Push a merged-LoRA state-dict into the running vLLM model.

        `state_dict` keys must match what the underlying base model expects
        (i.e. they should already have been canonicalized via
        `src.policy.weight_sync.canonicalize_lora_target_name`).
        """
        worker = self._get_driver_worker()
        model_runner = worker.model_runner  # type: ignore[attr-defined]
        model = model_runner.model
        # vLLM's load_weights expects an iterable of (name, tensor).
        items: Iterable[tuple[str, "torch.Tensor"]] = list(state_dict.items())
        model.load_weights(items)
        return {"loaded": len(state_dict)}

    def _get_driver_worker(self) -> Any:
        """Reach into the engine for the worker holding the model.

        vLLM has reorganized this path across versions; we try the canonical
        chain first and fall back if attribute names changed.
        """
        engine = self.llm.llm_engine
        # 0.6.x layout
        try:
            return engine.model_executor.driver_worker
        except AttributeError:
            pass
        # 0.5.x layout
        try:
            return engine.driver_worker
        except AttributeError:
            pass
        raise RuntimeError(
            "Could not locate the vLLM driver worker. The vLLM internal "
            "layout may have changed; pin a known-good vllm version in "
            "infra/image.py and update VLLMRunner._get_driver_worker."
        )
