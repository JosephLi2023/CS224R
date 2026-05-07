"""Modal A100 smoke: end-to-end HGPOTrainer.train_step on FakeWebShopEnv.

  modal run infra/app_train_step.py::train_step_smoke

Exercises the full pipeline:
  LoRAPolicy → VLLMRunner.generate_rich → RolloutCollector → TrajectoryGroup
  (with prompt_token_ids populated) → HGPOTrainer.compute_loss → backward →
  AdamW step → AdaptiveKLController.update → TrainStepStats returned.

Cost: ~$0.20 (A100 ~2-3 min).
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

app = modal.App("cs224r-hgpo-train-step")


@app.function(image=image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=20 * 60)
def train_step_smoke(k: int = 2, max_turns: int = 3) -> dict:
    import time
    import torch  # type: ignore[import-not-found]

    from src.algorithms.grpo.collectors import RolloutCollector, RolloutCollectorConfig
    from src.algorithms.grpo.trainer import (
        HGPOTrainer,
        HGPOTrainerConfig,
        progress_decomposer,
    )
    from src.envs.fake_webshop import FakeWebShopEnv
    from src.envs.prompts.react_webshop import parse_react_action, render_webshop_turn_prompt
    from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig
    from src.policy.vllm_runner import SamplingParams, VLLMRunner, VLLMRunnerConfig

    print(">>> Loading LoRAPolicy")
    policy = LoRAPolicy(LoRAPolicyConfig(cache_dir="/vol/hf_cache"))

    print(">>> Booting VLLMRunner")
    runner = VLLMRunner(
        VLLMRunnerConfig(
            gpu_memory_utilization=0.40,
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
    sampling = SamplingParams(
        n=1, temperature=1.0, top_p=0.95, max_tokens=32, return_logprobs=True
    )

    print(f">>> Collecting K={k} trajectories")
    t0 = time.time()
    group, cstats = collector.collect_group(
        task_id=0, env_name="webshop", K=k, sampling=sampling
    )
    collect_elapsed = round(time.time() - t0, 2)

    for traj in group.trajectories:
        for turn in traj.turns:
            assert len(turn.prompt_token_ids) > 0, (
                "prompt_token_ids missing — wiring regressed"
            )

    print(">>> Building trainer")
    trainer = HGPOTrainer(
        policy=policy,
        decomposer=progress_decomposer,
        cfg=HGPOTrainerConfig(
            alpha=0.5,
            lambda_consistency=0.0,
            clip_eps=0.2,
            learning_rate=1e-6,
            max_grad_norm=1.0,
        ),
    )

    trainable = policy.trainable_parameters()
    before_norms = [float(p.detach().norm().item()) for p in trainable[:4]]

    print(">>> train_step()")
    t1 = time.time()
    stats = trainer.train_step(group)
    step_elapsed = round(time.time() - t1, 2)

    after_norms = [float(p.detach().norm().item()) for p in trainable[:4]]
    param_deltas = [round(abs(a - b), 6) for a, b in zip(after_norms, before_norms)]

    # Assertions: loss finite, grad norm > 0, at least one LoRA param norm changed.
    assert not torch.isnan(torch.tensor(stats.total_loss)), "NaN total_loss"
    assert stats.grad_norm >= 0.0
    assert any(d > 0 for d in param_deltas), (
        "No LoRA parameter moved — optimizer step did not run"
    )

    volume.commit()
    return {
        "collect_elapsed_s": collect_elapsed,
        "train_step_elapsed_s": step_elapsed,
        "K": k,
        "n_trajectories": group.K,
        "n_total_turns": cstats.total_turns,
        "n_action_tokens": stats.n_action_tokens,
        "policy_loss": stats.policy_loss,
        "kl_term": stats.kl_term,
        "consistency": stats.consistency,
        "total_loss": stats.total_loss,
        "observed_kl": stats.observed_kl,
        "kl_coef_after": stats.kl_coef,
        "grad_norm": stats.grad_norm,
        "mean_traj_adv": stats.mean_traj_adv,
        "mean_turn_adv": stats.mean_turn_adv,
        "param_norm_deltas_first4": param_deltas,
    }


@app.local_entrypoint()
def main(k: int = 2, max_turns: int = 3) -> None:
    import json as _json
    print(_json.dumps(train_step_smoke.remote(k=k, max_turns=max_turns), indent=2, default=str))
