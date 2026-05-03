"""Modal A100 smoke for the Day-4 rollout collector + FakeWebShopEnv."""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

app = modal.App("cs224r-hgpo-rollout")


@app.function(image=image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=20 * 60)
def collect_smoke(k: int = 4, max_turns: int = 6) -> dict:
    import time
    from src.algorithms.grpo.collectors import RolloutCollector, RolloutCollectorConfig
    from src.envs.fake_webshop import FakeWebShopEnv
    from src.envs.prompts.react_webshop import parse_react_action, render_webshop_turn_prompt
    from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig
    from src.policy.vllm_runner import SamplingParams, VLLMRunner, VLLMRunnerConfig

    print(">>> Loading LoRAPolicy")
    policy = LoRAPolicy(LoRAPolicyConfig(cache_dir="/vol/hf_cache"))
    print(">>> trainable params:", policy.describe()["trainable_params"])

    print(">>> Booting VLLMRunner")
    runner = VLLMRunner(
        VLLMRunnerConfig(
            gpu_memory_utilization=0.45,
            max_model_len=2048,
            download_dir="/vol/hf_cache",
            enforce_eager=True,
            seed=42,
        )
    )

    collector = RolloutCollector(
        runner=runner,
        env_factory=lambda: FakeWebShopEnv(max_steps=8),
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        cfg=RolloutCollectorConfig(max_turns=max_turns),
    )
    sampling = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=48, return_logprobs=True)

    print(f">>> Collecting K={k} trajectories on FakeWebShopEnv task_id=0")
    t0 = time.time()
    group, stats = collector.collect_group(task_id=0, env_name="webshop", K=k, sampling=sampling)
    elapsed = round(time.time() - t0, 2)

    assert group.K == k, f"expected K={k}, got {group.K}"
    assert all(t.task_id == "0" for t in group.trajectories)
    assert all(t.env_name == "webshop" for t in group.trajectories)
    assert all(len(t.turns) >= 1 for t in group.trajectories)
    for traj in group.trajectories:
        for turn in traj.turns:
            assert len(turn.action_token_ids) > 0, "no action tokens captured"
            assert len(turn.action_token_logprobs) == len(turn.action_token_ids), \
                "logprob/token length mismatch"
            assert all(lp <= 0.0 for lp in turn.action_token_logprobs), \
                "logprobs must be non-positive"

    sample_actions = [[t.action_text[:60] for t in g.turns] for g in group.trajectories]
    sample_token_lens = [[len(t.action_token_ids) for t in g.turns] for g in group.trajectories]
    sample_first_logprobs = [
        [round(t.action_token_logprobs[0], 4) if t.action_token_logprobs else None for t in g.turns]
        for g in group.trajectories
    ]

    volume.commit()
    return {
        "elapsed_s": elapsed,
        "K": k,
        "completed": stats.completed,
        "truncated": stats.truncated,
        "total_turns": stats.total_turns,
        "total_action_tokens": stats.total_action_tokens,
        "final_rewards": stats.final_rewards,
        "sample_actions": sample_actions,
        "sample_token_lens": sample_token_lens,
        "sample_first_logprobs": sample_first_logprobs,
    }


@app.local_entrypoint()
def main(k: int = 4, max_turns: int = 6) -> None:
    import json as _json
    print(_json.dumps(collect_smoke.remote(k=k, max_turns=max_turns), indent=2, default=str))


# ---------- Real WebShop variant (uses webshop_image + WebShopAdapter) ----------

from infra.image import webshop_image  # type: ignore[import-not-found]


@app.function(image=webshop_image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=30 * 60)
def collect_smoke_real_webshop(k: int = 4, max_turns: int = 6, num_products: int = 1000) -> dict:
    """End-to-end smoke: real WebShop env + LoRAPolicy + VLLMRunner + RolloutCollector.
    Uses the small (1000-product) split with the BM25 index built by Track B."""
    import sys, time
    sys.path.insert(0, "/vol/code/webshop")  # WebShop has no setup.py; PYTHONPATH-based access
    sys.path.insert(0, "/workspace")

    from src.algorithms.grpo.collectors import RolloutCollector, RolloutCollectorConfig
    from src.envs.webshop_adapter import WebShopAdapter
    from src.envs.prompts.react_webshop import parse_react_action, render_webshop_turn_prompt
    from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig
    from src.policy.vllm_runner import SamplingParams, VLLMRunner, VLLMRunnerConfig

    print(">>> Loading LoRAPolicy")
    policy = LoRAPolicy(LoRAPolicyConfig(cache_dir="/vol/hf_cache"))
    print(">>> trainable params:", policy.describe()["trainable_params"])

    print(">>> Booting VLLMRunner")
    runner = VLLMRunner(
        VLLMRunnerConfig(
            gpu_memory_utilization=0.45,
            max_model_len=2048,
            download_dir="/vol/hf_cache",
            enforce_eager=True,
            seed=42,
        )
    )

    def env_factory():
        return WebShopAdapter(
            max_steps=8,
            observation_mode="text",
            env_kwargs={"num_products": num_products},
        )

    collector = RolloutCollector(
        runner=runner,
        env_factory=env_factory,
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        cfg=RolloutCollectorConfig(max_turns=max_turns),
    )
    sampling = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=64, return_logprobs=True)

    print(f">>> Collecting K={k} trajectories on REAL WebShop task_id=0")
    t0 = time.time()
    group, stats = collector.collect_group(task_id=0, env_name="webshop", K=k, sampling=sampling)
    elapsed = round(time.time() - t0, 2)

    assert group.K == k
    assert all(t.task_id == "0" for t in group.trajectories)
    for traj in group.trajectories:
        for turn in traj.turns:
            assert len(turn.action_token_ids) > 0
            assert len(turn.action_token_logprobs) == len(turn.action_token_ids)

    sample = {
        f"traj_{i}": {
            "n_turns": len(g.turns),
            "actions": [t.action_text[:80] for t in g.turns],
            "rewards": [t.raw_env_reward for t in g.turns],
            "final_reward": g.final_reward,
        }
        for i, g in enumerate(group.trajectories)
    }
    volume.commit()
    return {
        "elapsed_s": elapsed,
        "K": k,
        "num_products": num_products,
        "completed": stats.completed,
        "truncated": stats.truncated,
        "total_turns": stats.total_turns,
        "total_action_tokens": stats.total_action_tokens,
        "final_rewards": stats.final_rewards,
        "trajectories": sample,
    }
