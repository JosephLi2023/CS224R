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
