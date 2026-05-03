"""Modal A100 smoke tests for the LoRA policy + vLLM weight-sync stack.

Three entrypoints, run incrementally to bound cost:

  modal run infra/app_policy.py::lora_load_smoke      # ~2 min, ~$0.10
  modal run infra/app_policy.py::vllm_generate_smoke  # ~3 min, ~$0.15
  modal run infra/app_policy.py::weight_sync_smoke    # ~5 min, ~$0.30 (full Day-3 deliverable)

`weight_sync_smoke` is the canonical Day-3 deliverable: 4 prompts → 4 vLLM
generations → perturb LoRA → sync to vLLM → 4 fresh generations → assert
they differ.
"""

from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

APP_NAME = "cs224r-hgpo-policy"
app = modal.App(APP_NAME)

PROMPTS = [
    "You are an online shopping agent. The user wants a black laptop bag under $30. Your first action: ",
    "You are an online shopping agent. Find a wireless mouse with at least 4-star rating. Your first action: ",
    "You are an online shopping agent. The user wants a women's running shoe size 8. Your first action: ",
    "You are an online shopping agent. Find a USB-C cable longer than 6 feet. Your first action: ",
]


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_MOUNT: volume},
    timeout=15 * 60,
)
def lora_load_smoke() -> dict:
    """Cheapest smoke: just instantiate LoRAPolicy and report the param split."""
    from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig

    cfg = LoRAPolicyConfig(cache_dir="/vol/hf_cache")
    policy = LoRAPolicy(cfg)
    info = policy.describe()
    print(info)
    volume.commit()  # persist HF cache for subsequent runs
    return info


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_MOUNT: volume},
    timeout=20 * 60,
)
def vllm_generate_smoke() -> dict:
    """Boot vLLM, generate 4 samples per prompt, return text + timing."""
    import time

    from src.policy.vllm_runner import SamplingParams, VLLMRunner, VLLMRunnerConfig

    cfg = VLLMRunnerConfig(
        gpu_memory_utilization=0.85,  # only model on this GPU during this smoke
        max_model_len=2048,
        download_dir="/vol/hf_cache",
        enforce_eager=True,            # cheaper compile, fine for smoke
    )
    runner = VLLMRunner(cfg)

    sampling = SamplingParams(n=4, temperature=1.0, top_p=0.95, max_tokens=64)

    t0 = time.time()
    out = runner.generate(PROMPTS, sampling)
    elapsed = round(time.time() - t0, 2)

    sample = {
        f"prompt_{i}": [s[:120] for s in samples] for i, samples in enumerate(out)
    }
    print(f"vllm generation took {elapsed}s for {len(PROMPTS)} prompts × n=4")
    volume.commit()
    return {"elapsed_s": elapsed, "n_prompts": len(PROMPTS), "n_per_prompt": 4, "sample": sample}


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_MOUNT: volume},
    timeout=30 * 60,
)
def weight_sync_smoke() -> dict:
    """Full Day-3 verification:

    1. Build LoRAPolicy (trainer-side).
    2. Build VLLMRunner pointing at the same backbone.
    3. Generate 4 samples for prompts → snapshot baseline.
    4. Perturb LoRA `lora_B.*` weights with deterministic random noise.
    5. Push merged state-dict into vLLM.
    6. Re-generate → assert ≥75% of (prompt, sample) pairs differ from baseline.
    """
    import time

    import torch  # type: ignore[import-not-found]

    from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig
    from src.policy.vllm_runner import SamplingParams, VLLMRunner, VLLMRunnerConfig

    # --- 1. Trainer model with LoRA -----------------------------------
    print(">>> Loading LoRAPolicy")
    policy = LoRAPolicy(LoRAPolicyConfig(cache_dir="/vol/hf_cache"))
    print(">>> Policy params:", policy.describe())

    # --- 2. vLLM runner -----------------------------------------------
    # Both models on one A100 → keep vLLM under 50% memory.
    print(">>> Loading VLLMRunner")
    runner = VLLMRunner(
        VLLMRunnerConfig(
            gpu_memory_utilization=0.45,
            max_model_len=2048,
            download_dir="/vol/hf_cache",
            enforce_eager=True,
            seed=42,
        )
    )

    sampling = SamplingParams(n=4, temperature=1.0, top_p=0.95, max_tokens=48)

    # --- 3. Baseline generation ---------------------------------------
    print(">>> Baseline generation")
    t0 = time.time()
    baseline = runner.generate(PROMPTS, sampling)
    t_base = round(time.time() - t0, 2)

    # --- 4. Perturb LoRA params ---------------------------------------
    print(">>> Perturbing LoRA params")
    rng = torch.Generator(device="cuda:0").manual_seed(0)
    perturbed_count = 0
    with torch.no_grad():
        for name, p in policy.model.named_parameters():
            if "lora_B" in name and p.requires_grad:
                p.add_(torch.randn(p.shape, generator=rng, device=p.device, dtype=p.dtype) * 0.5)
                perturbed_count += 1
    print(f">>> Perturbed {perturbed_count} lora_B parameter tensors")

    # --- 5. Sync to vLLM ----------------------------------------------
    print(">>> Building merged state dict + syncing into vLLM")
    state = policy.merged_state_dict()
    print(f">>> merged_state_dict has {len(state)} keys")
    sync_info = runner.sync_weights(state)
    print(">>> sync_weights returned:", sync_info)

    # Free the trainer-side merged copy before re-generating.
    del state
    torch.cuda.empty_cache()

    # --- 6. Re-generate ------------------------------------------------
    print(">>> Generation after weight sync")
    t0 = time.time()
    after = runner.generate(PROMPTS, sampling)
    t_after = round(time.time() - t0, 2)

    # --- compare ------------------------------------------------------
    pairs = 0
    differ = 0
    for b_row, a_row in zip(baseline, after):
        for b, a in zip(b_row, a_row):
            pairs += 1
            if b.strip() != a.strip():
                differ += 1
    diff_ratio = round(differ / max(1, pairs), 3)
    assert diff_ratio >= 0.75, (
        f"Weight sync did not propagate: only {differ}/{pairs} samples changed. "
        "Either sync_weights is a no-op, or perturbation magnitude is too small."
    )

    sample = {
        f"prompt_{i}": {
            "before": [s[:80] for s in baseline[i]],
            "after": [s[:80] for s in after[i]],
        }
        for i in range(len(PROMPTS))
    }
    volume.commit()
    return {
        "baseline_elapsed_s": t_base,
        "after_elapsed_s": t_after,
        "lora_tensors_perturbed": perturbed_count,
        "merged_state_dict_keys": len(state) if False else None,  # state already deleted
        "diff_pairs": f"{differ}/{pairs}",
        "diff_ratio": diff_ratio,
        "sample": sample,
    }


@app.local_entrypoint()
def main(action: str = "lora_load_smoke") -> None:
    import json as _json

    if action == "lora_load_smoke":
        result = lora_load_smoke.remote()
    elif action == "vllm_generate_smoke":
        result = vllm_generate_smoke.remote()
    elif action == "weight_sync_smoke":
        result = weight_sync_smoke.remote()
    else:
        raise ValueError(
            f"Unknown action: {action!r} (expected one of "
            "'lora_load_smoke' | 'vllm_generate_smoke' | 'weight_sync_smoke')"
        )
    print(_json.dumps(result, indent=2, default=str))
