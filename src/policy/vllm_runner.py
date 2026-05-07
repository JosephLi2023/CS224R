"""vLLM runner: batched K-sample text generation + in-process weight sync.

Owns a `vllm.LLM` instance pointed at the same backbone the trainer uses
(Qwen2.5-1.5B-Instruct by default). After every optimizer step the trainer
calls `sync_weights(state_dict_iter)` to push the merged-LoRA weights into
the running engine so subsequent rollouts reflect the policy update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import LLM  # type: ignore[import-not-found]


@dataclass
class VLLMRunnerConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.20  # H2 v11: was 0.50 — vLLM was over-reserving KV cache (which it never returns), pushing steady-state past A100-80 cap. K=8 with max_tokens=48 doesn't need much KV; the orchestrator typically passes 0.20 explicitly anyway.
    seed: int = 0
    enforce_eager: bool = False
    download_dir: str | None = None  # HF download cache; align with LoRAPolicy.cache_dir
    # vLLM CPU swap space in GiB. Default vLLM value is 4 GiB. We override
    # to 0 to disable CPU↔GPU KV-block swapping entirely. Rationale:
    # vLLM 0.6.3.post1 has a known bug where `blocks_to_swap_in` can
    # contain a non-int element under race conditions during weight
    # syncs / KV preemption, causing
    #   `RuntimeError: unknown parameter type` inside
    #   `worker.prepare_worker_input → torch.tensor(blocks_to_swap_in)`
    # This bug reliably fires after ~30 weight-sync cycles in our
    # heavy-rollout protocols (CF, TurnRDv2). Setting swap_space=0
    # means the KV scheduler never swaps blocks → the buggy code path
    # is never entered. Safe for our workload (K≤8, max_tokens≤48,
    # gpu_memory_utilization≤0.30 leaves >50% GPU headroom for KV).
    swap_space_gib: int = 0


@dataclass
class SamplingParams:
    n: int = 4               # how many trajectories per prompt (K from proposal §3.1)
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 256
    stop: list[str] = field(default_factory=list)
    # When True, generate_rich() pulls per-token logprobs back from vLLM. Costs
    # a small amount of extra CPU but is required by the GRPO trainer's
    # importance-weight calculation.
    return_logprobs: bool = True
    # If None, vLLM uses fresh randomness on every call (preferred for
    # diverse K-trajectory rollouts). Set to an int for deterministic tests.
    seed: int | None = None


@dataclass(frozen=True)
class GenerationOutput:
    """Token-level generation result returned by `VLLMRunner.generate_rich`."""

    text: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    prompt_token_count: int
    prompt_token_ids: tuple[int, ...]
    finish_reason: str


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
            swap_space=cfg.swap_space_gib,
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
            seed=sampling.seed,  # None ⇒ fresh randomness each call
        )
        outputs = self.llm.generate(prompts, params)
        return [[o.text for o in req.outputs] for req in outputs]

    def generate_rich(
        self, prompts: list[str], sampling: SamplingParams
    ) -> list[list[GenerationOutput]]:
        """Generate with per-token ids + log-probs preserved.

        Used by the rollout collector so the trainer has the rollout-time
        policy log-probs needed for the PPO importance-weight ratio.
        """
        from vllm import SamplingParams as VLLMSamplingParams  # type: ignore[import-not-found]

        params = VLLMSamplingParams(
            n=sampling.n,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            stop=sampling.stop or None,
            seed=sampling.seed,  # None ⇒ fresh randomness each call
            logprobs=1 if sampling.return_logprobs else None,
        )
        request_outputs = self.llm.generate(prompts, params)

        results: list[list[GenerationOutput]] = []
        for req in request_outputs:
            prompt_ids_tup = tuple(int(t) for t in (req.prompt_token_ids or ()))
            prompt_token_count = len(prompt_ids_tup)
            per_prompt: list[GenerationOutput] = []
            for o in req.outputs:
                token_ids = tuple(int(t) for t in (o.token_ids or ()))
                logprobs: tuple[float, ...] = ()
                if sampling.return_logprobs and getattr(o, "logprobs", None) is not None:
                    extracted: list[float] = []
                    for tok_id, dist in zip(token_ids, o.logprobs):
                        if dist is None:
                            extracted.append(0.0)
                            continue
                        entry = dist.get(tok_id) if hasattr(dist, "get") else None
                        if entry is None:
                            extracted.append(0.0)
                            continue
                        extracted.append(float(getattr(entry, "logprob", entry)))
                    logprobs = tuple(extracted)
                per_prompt.append(
                    GenerationOutput(
                        text=o.text,
                        token_ids=token_ids,
                        token_logprobs=logprobs,
                        prompt_token_count=prompt_token_count,
                        prompt_token_ids=prompt_ids_tup,
                        finish_reason=str(getattr(o, "finish_reason", "")),
                    )
                )
            results.append(per_prompt)
        return results

    # ------------------------------------------------------------------
    # Weight sync (the core Day-3 deliverable)
    # ------------------------------------------------------------------

    def sync_weights(self, state_dict) -> dict[str, int]:
        """Push merged-LoRA weights into the running vLLM model.

        `state_dict` can be either a dict OR an iterable of (name, tensor)
        pairs. The iterable form streams one tensor at a time so LoRA
        policies can use `iter_merged_weights()` without materializing the
        full merged state-dict in memory (fixes sync_weights OOM at scale).
        """
        worker = self._get_driver_worker()
        model_runner = worker.model_runner  # type: ignore[attr-defined]
        model = model_runner.model
        # Accept both dict and iterable-of-pairs.
        if hasattr(state_dict, "items") and not hasattr(state_dict, "__next__"):
            items_iter = iter(state_dict.items())
        else:
            items_iter = iter(state_dict)
        # vLLM's `load_weights` itself streams — we count as we go.
        count = 0

        def counted():
            nonlocal count
            for name, t in items_iter:
                count += 1
                yield name, t

        model.load_weights(counted())
        return {"loaded": count}

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
