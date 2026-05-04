"""Modal A100 app: 50-episode flat-GRPO training loop on real WebShop.

  modal run infra/app_train_loop.py --n_episodes 50 --k 4 --max_turns 6

Per-episode loop:
  task_id = task_id_offset + episode_idx
  group, _ = collector.collect_group(task_id, K=k)
  stats = trainer.train_step(group)
  runner.sync_weights(policy.merged_state_dict())
  log.append({episode, mean_reward, policy_loss, kl_term, kl_coef, grad_norm})

Persists `train_log.json` to /vol/manifests/<run_name>/train_log.json so we
have a per-episode reward curve we can plot offline.

Cost: ~$3-5 for 50 episodes (~10-15 min on A100).
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import webshop_image

app = modal.App("cs224r-hgpo-train-loop")


@app.function(image=webshop_image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=60 * 60)
def train_loop_smoke(
    n_episodes: int = 50,
    k: int = 4,
    max_turns: int = 6,
    task_id_offset: int = 0,
    num_products: int = 1000,
    sync_every: int = 1,
    run_name: str = "flat_grpo_webshop_smoke",
    sft_adapter: str = "",
    use_sft_as_ref: bool = True,
    kl_warmup_episodes: int = 0,
    gpu_mem_util: float = 0.30,
    config: str = "",  # Day 14: when non-empty, build trainer from this JSON config.
) -> dict:
    import json
    import os
    import sys
    import time
    from datetime import datetime, timezone

    sys.path.insert(0, "/vol/code/webshop")
    sys.path.insert(0, "/workspace")

    from src.algorithms.grpo.collectors import RolloutCollector, RolloutCollectorConfig
    from src.algorithms.grpo.trainer import (
        HGPOTrainer,
        HGPOTrainerConfig,
        progress_decomposer,
    )
    from src.envs.webshop_adapter import WebShopAdapter
    from src.envs.prompts.react_webshop import parse_react_action, render_webshop_turn_prompt
    from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig
    from src.policy.vllm_runner import SamplingParams, VLLMRunner, VLLMRunnerConfig

    # Day 14: optional JSON config — overrides the per-flag trainer/decomposer
    # construction with a `build_trainer_from_config` call. When config is
    # empty, the legacy flag-driven path runs unchanged (preserves all
    # existing callers' behavior — Methods A/C and the flat-GRPO smoke tests).
    cfg_dict: dict | None = None
    if config:
        with open(config) as fh:
            cfg_dict = json.load(fh)
        # Per-config overrides for the loop knobs that the JSON owns.
        train_block = cfg_dict.get("train", {}) if cfg_dict else {}
        if "total_episodes" in train_block:
            n_episodes = int(train_block["total_episodes"])
        if "K_trajectories_per_task" in train_block:
            k = int(train_block["K_trajectories_per_task"])
        run_block = cfg_dict.get("run", {}) if cfg_dict else {}
        if "name" in run_block:
            run_name = str(run_block["name"])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = f"/vol/manifests/{run_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(">>> Loading LoRAPolicy")
    policy = LoRAPolicy(LoRAPolicyConfig(cache_dir="/vol/hf_cache"))

    sft_loaded = False
    if sft_adapter:
        print(">>> Loading SFT-warm-started LoRA adapter from", sft_adapter)
        policy.load_adapter(sft_adapter)
        sft_loaded = True

    print(">>> Booting VLLMRunner")
    runner = VLLMRunner(
        VLLMRunnerConfig(
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=2048,
            download_dir="/vol/hf_cache",
            enforce_eager=True,
            seed=0,
        )
    )

    if sft_loaded:
        # vLLM was just initialised with the BASE Qwen weights — push the
        # SFT-merged LoRA into it so the very first episode samples from the
        # warm-started policy. Without this, episode 0 would still be
        # base-Qwen.
        print(">>> Syncing SFT-merged weights to vLLM before first episode")
        runner.sync_weights(policy.iter_merged_weights())
        import gc as _gc
        import torch as _torch
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    def env_factory():
        return WebShopAdapter(
            max_steps=max_turns + 2,
            observation_mode="text",
            env_kwargs={"num_products": num_products},
        )

    # Day 14: choose between the legacy flag-driven trainer/collector
    # construction and the JSON-config-driven path.
    if cfg_dict is not None:
        from src.trainers.train_hgpo import build_trainer_from_config

        (
            trainer,
            _refresh_fn,
            turnrd_emit_path,
            turnrd_embedder,
            judge_decomposer,
        ) = build_trainer_from_config(cfg_dict, policy=policy)
        collector = RolloutCollector(
            runner=runner,
            env_factory=env_factory,
            prompt_renderer=render_webshop_turn_prompt,
            action_parser=parse_react_action,
            cfg=RolloutCollectorConfig(max_turns=max_turns),
            turnrd_emit_path=turnrd_emit_path,
            turnrd_embedder=turnrd_embedder,
            judge_decomposer=judge_decomposer,
        )
    else:
        collector = RolloutCollector(
            runner=runner,
            env_factory=env_factory,
            prompt_renderer=render_webshop_turn_prompt,
            action_parser=parse_react_action,
            cfg=RolloutCollectorConfig(max_turns=max_turns),
        )
        trainer = HGPOTrainer(
            policy=policy,
            decomposer=progress_decomposer,
            cfg=HGPOTrainerConfig(
                alpha=1.0,                 # flat GRPO: drop turn-level signal
                lambda_consistency=0.0,
                clip_eps=0.2,
                learning_rate=1e-6,
                max_grad_norm=1.0,
                kl_warmup_episodes=kl_warmup_episodes,
            ),
        )

    sampling = SamplingParams(
        n=1, temperature=1.0, top_p=0.95, max_tokens=48,
        return_logprobs=True, seed=None,  # fresh sampling each call → diverse K trajectories
    )

    if sft_loaded and use_sft_as_ref:
        n = trainer.snapshot_current_lora_as_ref()
        print(f">>> Snapshotted {n} LoRA modules as KL reference (RL-from-SFT)")

    log: list[dict] = []
    overall_start = time.time()

    for ep in range(n_episodes):
        ep_t0 = time.time()
        task_id = task_id_offset + ep
        try:
            group, cstats = collector.collect_group(
                task_id=task_id, env_name="webshop", K=k, sampling=sampling
            )
            stats = trainer.train_step(group)

            mean_reward = sum(cstats.final_rewards) / max(1, len(cstats.final_rewards))
            std_reward = (
                sum((r - mean_reward) ** 2 for r in cstats.final_rewards)
                / max(1, len(cstats.final_rewards))
            ) ** 0.5

            row = {
                "episode": ep,
                "task_id": task_id,
                "mean_reward": round(mean_reward, 4),
                "std_reward": round(std_reward, 4),
                "completed": cstats.completed,
                "truncated": cstats.truncated,
                "n_turns": cstats.total_turns,
                "n_action_tokens": stats.n_action_tokens,
                "policy_loss": round(stats.policy_loss, 6),
                "kl_term": round(stats.kl_term, 6),
                "consistency": round(stats.consistency, 6),
                "total_loss": round(stats.total_loss, 6),
                "observed_kl": round(stats.observed_kl, 6),
                "kl_coef": round(stats.kl_coef, 6),
                "grad_norm": round(stats.grad_norm, 6),
                "mean_traj_adv": round(stats.mean_traj_adv, 6),
                "elapsed_s": round(time.time() - ep_t0, 2),
            }
            log.append(row)
            print(
                f"ep={ep:03d} task={task_id} "
                f"R={row['mean_reward']:.3f}\u00b1{row['std_reward']:.3f} "
                f"loss={row['policy_loss']:+.4f} "
                f"kl={row['observed_kl']:+.4f} kl_coef={row['kl_coef']:.4f} "
                f"gn={row['grad_norm']:.3f} t={row['elapsed_s']}s"
            )

            if sync_every > 0 and (ep + 1) % sync_every == 0:
                # Stream merged LoRA weights one tensor at a time — avoids
                # the deepcopy(self.model) memory spike. Then explicitly
                # release any leftover buffers before next episode.
                runner.sync_weights(policy.iter_merged_weights())
                import gc
                import torch as _torch
                gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()

            # Persist log every episode so a crash or local timeout still leaves
            # a useful artifact on the Volume.
            with open(os.path.join(run_dir, "train_log.json"), "w") as f:
                json.dump({"rows": log, "config": {
                    "n_episodes": n_episodes, "K": k, "max_turns": max_turns,
                    "task_id_offset": task_id_offset, "num_products": num_products,
                    "sync_every": sync_every, "run_name": run_name,
                    "sft_adapter": sft_adapter,
                }}, f, indent=2)
            volume.commit()
        except Exception as exc:
            print(f"ep={ep} CRASHED: {exc!r}")
            log.append({"episode": ep, "task_id": task_id, "error": repr(exc)})
            continue

    total_elapsed = round(time.time() - overall_start, 2)

    rewards = [r["mean_reward"] for r in log if "mean_reward" in r]
    early = rewards[: max(1, len(rewards) // 10)]
    late = rewards[-max(1, len(rewards) // 10):]

    summary = {
        "run_dir": run_dir,
        "n_episodes": n_episodes,
        "K": k,
        "completed_episodes": len(rewards),
        "total_elapsed_s": total_elapsed,
        "mean_reward_first_10pct": round(sum(early) / max(1, len(early)), 4),
        "mean_reward_last_10pct": round(sum(late) / max(1, len(late)), 4),
        "mean_grad_norm": round(
            sum(r.get("grad_norm", 0) for r in log) / max(1, len(rewards)), 4
        ),
        "mean_kl_coef_final": round(log[-1].get("kl_coef", 0), 4) if log else 0.0,
        "log_path": os.path.join(run_dir, "train_log.json"),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    n_episodes: int = 50,
    k: int = 4,
    max_turns: int = 6,
    task_id_offset: int = 0,
    num_products: int = 1000,
    sync_every: int = 1,
    run_name: str = "flat_grpo_webshop_smoke",
    sft_adapter: str = "",
    use_sft_as_ref: bool = True,
    kl_warmup_episodes: int = 0,
    gpu_mem_util: float = 0.30,
    config: str = "",
) -> None:
    import json as _json
    print(_json.dumps(
        train_loop_smoke.remote(
            n_episodes=n_episodes, k=k, max_turns=max_turns,
            task_id_offset=task_id_offset, num_products=num_products,
            sync_every=sync_every, run_name=run_name,
            sft_adapter=sft_adapter,
            use_sft_as_ref=use_sft_as_ref,
            kl_warmup_episodes=kl_warmup_episodes,
            gpu_mem_util=gpu_mem_util,
            config=config,
        ),
        indent=2, default=str,
    ))
