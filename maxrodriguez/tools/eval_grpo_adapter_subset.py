from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import modal  # type: ignore[import-not-found]

os.environ.setdefault("CS224R_SKIP_OPENAI_SECRET", "1")

from infra.common import VOLUME_MOUNT, volume
from infra.image import alfworld_image


app = modal.App("maxrodriguez-grpo-adapter-eval")


@app.function(
    image=alfworld_image,
    gpu="A100-80GB",
    volumes={VOLUME_MOUNT: volume},
    timeout=2 * 60 * 60,
)
def evaluate_grpo_adapter_subset(
    *,
    adapter_path: str,
    config_path: str,
    run_name: str,
    eval_episodes: int = 20,
    eval_task_id_base: int = 6500,
    max_turns: int = 30,
    gpu_mem_util: float = 0.20,
) -> dict:
    sys.path.insert(0, "/workspace")
    volume.reload()

    from infra.app_train_loop import _resolve_env_bindings
    from src.algorithms.grpo.collectors import RolloutCollector, RolloutCollectorConfig
    from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig
    from src.policy.vllm_runner import SamplingParams, VLLMRunner, VLLMRunnerConfig

    with open(f"{adapter_path}/adapter_config.json") as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg["base_model_name_or_path"]

    with open(config_path) as f:
        cfg_dict = json.load(f)

    adapter_cls, prompt_renderer, action_parser = _resolve_env_bindings("alfworld")
    env_block = dict(cfg_dict.get("env", {}))
    cfg_env_kwargs = dict(env_block.get("env_kwargs", {}) or {})
    adapter_max_steps = int(env_block.get("max_steps", max_turns + 2))
    adapter_obs_mode = str(env_block.get("observation_mode", "text"))
    adapter_task_split = str(env_block.get("task_split", "train"))
    adapter_use_tw_ir = bool(env_block.get("use_textworld_intermediate_reward", False))
    adapter_use_facts_diff_ir = bool(env_block.get("use_facts_diff_intermediate_reward", False))

    def env_factory():
        return adapter_cls(
            max_steps=adapter_max_steps,
            observation_mode=adapter_obs_mode,
            task_split=adapter_task_split,
            env_kwargs=dict(cfg_env_kwargs),
            use_textworld_intermediate_reward=adapter_use_tw_ir,
            use_facts_diff_intermediate_reward=adapter_use_facts_diff_ir,
        )

    policy = LoRAPolicy(
        LoRAPolicyConfig(
            model_name=base_model_name,
            cache_dir="/vol/hf_cache",
        )
    )
    policy.load_adapter(adapter_path)

    runner = VLLMRunner(
        VLLMRunnerConfig(
            model_name=base_model_name,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=2048,
            download_dir="/vol/hf_cache",
            enforce_eager=True,
            seed=0,
        )
    )
    runner.sync_weights(policy.iter_merged_weights())

    eval_sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=48,
        return_logprobs=False,
        seed=42,
    )
    eval_collector = RolloutCollector(
        runner=runner,
        env_factory=env_factory,
        prompt_renderer=prompt_renderer,
        action_parser=action_parser,
        cfg=RolloutCollectorConfig(max_turns=max_turns),
    )

    eval_t0 = time.time()
    eval_returns: list[float] = []
    eval_turns: list[int] = []
    eval_completed = 0
    eval_truncated = 0
    eval_crashed = 0

    for j in range(eval_episodes):
        task_id = eval_task_id_base + j
        try:
            _, eval_cstats = eval_collector.collect_group(
                task_id=task_id,
                env_name="alfworld",
                K=1,
                sampling=eval_sampling,
            )
            eval_returns.extend(eval_cstats.final_rewards)
            eval_turns.append(eval_cstats.total_turns)
            eval_completed += eval_cstats.completed
            eval_truncated += eval_cstats.truncated
        except Exception as exc:
            print(f"eval ep={j} task={task_id} CRASHED: {exc!r}")
            eval_crashed += 1
            continue

    eval_elapsed = round(time.time() - eval_t0, 2)
    if eval_returns:
        mean_r = sum(eval_returns) / len(eval_returns)
        pct_success = sum(1 for r in eval_returns if r > 0) / len(eval_returns)
        std_r = (sum((r - mean_r) ** 2 for r in eval_returns) / len(eval_returns)) ** 0.5
    else:
        mean_r = 0.0
        pct_success = 0.0
        std_r = 0.0

    summary = {
        "run_name": run_name,
        "adapter_path": adapter_path,
        "config_path": config_path,
        "base_model_name_or_path": base_model_name,
        "eval_episodes": eval_episodes,
        "task_id_base": eval_task_id_base,
        "avg_return": round(mean_r, 4),
        "std_return": round(std_r, 4),
        "pct_success": round(pct_success, 4),
        "n_episodes_ok": len(eval_returns),
        "n_episodes_crashed": eval_crashed,
        "completed": eval_completed,
        "truncated": eval_truncated,
        "n_turns_avg": round(sum(eval_turns) / max(1, len(eval_turns)), 2),
        "elapsed_s": eval_elapsed,
    }
    print(
        f">>> Eval done: avg_R={summary['avg_return']:.4f} "
        f"(±{summary['std_return']:.4f}) | pct_success={summary['pct_success']:.3f} | "
        f"ok={summary['n_episodes_ok']}/{eval_episodes} | elapsed={eval_elapsed}s"
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = f"/vol/manifests/maxrodriguez_grpo_eval/{run_name}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    adapter_path: str,
    config_path: str,
    run_name: str,
    eval_episodes: int = 20,
    eval_task_id_base: int = 6500,
    max_turns: int = 30,
    gpu_mem_util: float = 0.20,
) -> None:
    result = evaluate_grpo_adapter_subset.remote(
        adapter_path=adapter_path,
        config_path=config_path,
        run_name=run_name,
        eval_episodes=eval_episodes,
        eval_task_id_base=eval_task_id_base,
        max_turns=max_turns,
        gpu_mem_util=gpu_mem_util,
    )
    print(json.dumps(result, indent=2, default=str))
