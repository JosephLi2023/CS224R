"""Modal A100 app: H-GRPO training loop on real WebShop OR AlfWorld.

Two image-bound `@app.function` entrypoints (`train_loop_webshop`,
`train_loop_alfworld`) both delegate to `_train_loop_impl`; the local entrypoint
dispatches on `--env-name`. Persists a per-episode `train_log.json` under
/vol/manifests/<run_name>/.
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, maybe_openai_secret, volume
from infra.image import alfworld_image, webshop_image

app = modal.App("cs224r-hgpo-train-loop")


# Returns the (adapter_cls, prompt_renderer, action_parser) triple per env.
def _resolve_env_bindings(env_name: str):
    """Lazily import the (adapter_cls, prompt_renderer, action_parser) triple
    for `env_name`. Torch-free so unit tests run without a GPU; raises
    ValueError on an unknown env_name.
    """
    if env_name == "webshop":
        from src.envs.prompts.react_webshop import (
            parse_react_action,
            render_webshop_turn_prompt,
        )
        from src.envs.webshop_adapter import WebShopAdapter

        return WebShopAdapter, render_webshop_turn_prompt, parse_react_action

    if env_name == "alfworld":
        from src.envs.alfworld_adapter import ALFWorldAdapter
        from src.envs.prompts.react_alfworld import (
            parse_react_action,
            render_alfworld_turn_prompt,
        )

        return ALFWorldAdapter, render_alfworld_turn_prompt, parse_react_action

    raise ValueError(
        f"_resolve_env_bindings: unknown env_name {env_name!r}; "
        "expected 'webshop' or 'alfworld'."
    )


def _train_loop_impl(
    *,
    env_name: str,
    n_episodes: int,
    k: int,
    max_turns: int,
    task_id_offset: int,
    num_products: int,
    sync_every: int,
    run_name: str,
    sft_adapter: str,
    use_sft_as_ref: bool,
    kl_warmup_episodes: int,
    gpu_mem_util: float,
    config: str,
    eval_episodes: int,
    eval_task_id_base: int,
    round_idx: int,
    save_adapter_out: str = "",
    rollout_temperature: float = 1.0,
) -> dict:
    """Env-agnostic training loop body; the two entrypoints differ only in
    image binding. `num_products` is WebShop-only (other envs read
    `env.env_kwargs` from the JSON config). `eval_task_id_base` defaults to 6500,
    kept disjoint from the per-seed training task ranges.
    """
    import json
    import os
    import sys
    import time
    from datetime import datetime, timezone

    sys.path.insert(0, "/vol/code/webshop")
    sys.path.insert(0, "/workspace")

    # Reload so we see the ckpt the previous round's train_turnrd committed.
    volume.reload()

    from src.algorithms.grpo.collectors import RolloutCollector, RolloutCollectorConfig
    from src.algorithms.grpo.trainer import (
        HGPOTrainer,
        HGPOTrainerConfig,
        progress_decomposer,
    )
    from src.policy.lora_policy import LoRAMergeNonFiniteError, LoRAPolicy, LoRAPolicyConfig
    from src.policy.vllm_runner import SamplingParams, VLLMRunner, VLLMRunnerConfig

    adapter_cls, prompt_renderer, action_parser = _resolve_env_bindings(env_name)

    # Optional JSON config drives `build_trainer_from_config`; empty config
    # keeps the legacy flag-driven path (Methods A/C, flat-GRPO smoke tests).
    cfg_dict: dict | None = None
    if config:
        with open(config) as fh:
            cfg_dict = json.load(fh)
        # Do NOT override n_episodes, k, or run_name from the JSON: those are
        # orchestrator-supplied per-round, and overriding would collapse the
        # per-round run names the log aggregator relies on.

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = f"/vol/manifests/{run_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(">>> Loading LoRAPolicy")
    # Plumb LoRA arch knobs from the JSON config (else the dataclass
    # defaults - rank 16, attention-only - silently win).
    _lora_kwargs: dict = {}
    if cfg_dict is not None:
        _pol = cfg_dict.get("policy") or {}
        if "lora_rank" in _pol:
            _r = int(_pol["lora_rank"])
            _lora_kwargs["lora_r"] = _r
            # Standard 2:1 alpha:rank convention (matches dataclass default 16/32).
            _lora_kwargs["lora_alpha"] = 2 * _r
        if "lora_target_modules" in _pol and _pol["lora_target_modules"]:
            _lora_kwargs["lora_target_modules"] = list(_pol["lora_target_modules"])
        if _lora_kwargs:
            print(f">>> Applying LoRA arch overrides from config: {_lora_kwargs}")
    policy = LoRAPolicy(LoRAPolicyConfig(cache_dir="/vol/hf_cache", **_lora_kwargs))

    sft_loaded = False
    if sft_adapter:
        print(">>> Loading SFT-warm-started LoRA adapter from", sft_adapter)
        policy.load_adapter(sft_adapter)
        sft_loaded = True

    # Pre-vLLM cleanup: free the ~14 GB the allocator holds after the
    # base+LoRA+adapter load so vLLM's init-time probe forward can grow memory
    # (else `_assert_memory_footprint_increased_during_profiling` fires).
    import gc as _gc_pre
    import torch as _torch_pre
    _gc_pre.collect()
    if _torch_pre.cuda.is_available():
        _torch_pre.cuda.synchronize()
        _torch_pre.cuda.empty_cache()
        _mem_alloc = _torch_pre.cuda.memory_allocated() / (1024 ** 3)
        _mem_reserv = _torch_pre.cuda.memory_reserved() / (1024 ** 3)
        print(
            f">>> Pre-vLLM CUDA memory snapshot: "
            f"allocated={_mem_alloc:.2f} GB, reserved={_mem_reserv:.2f} GB"
        )

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
        # vLLM booted with BASE Qwen weights; push the SFT-merged LoRA in so
        # episode 0 samples from the warm-started policy.
        print(">>> Syncing SFT-merged weights to vLLM before first episode")
        runner.sync_weights(policy.iter_merged_weights())
        import gc as _gc
        import torch as _torch
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    # Env-aware kwargs: WebShop falls back to {"num_products": num_products}
    # when the JSON has no `env.env_kwargs`; other envs read it verbatim.
    env_block: dict = {}
    if cfg_dict is not None and isinstance(cfg_dict.get("env"), dict):
        env_block = dict(cfg_dict["env"])

    cfg_env_kwargs = dict(env_block.get("env_kwargs", {}) or {})
    if env_name == "webshop" and not cfg_env_kwargs:
        cfg_env_kwargs = {"num_products": num_products}

    # Adapter kwargs from the env block; max_steps derives from max_turns
    # ("steps = turns + 2") to preserve the legacy flag semantics.
    adapter_max_steps = int(env_block.get("max_steps", max_turns + 2))
    adapter_obs_mode = str(env_block.get("observation_mode", "text"))
    adapter_task_split = str(env_block.get("task_split", "train"))
    # Phase 2 (AlfWorld only): opt-in to TextWorld's native intermediate
    # reward as the V-head signal. Default False; threaded only on alfworld.
    adapter_use_tw_ir = bool(env_block.get("use_textworld_intermediate_reward", False))
    # Phase 3 (AlfWorld only): opt-in to the PDDL-facts-diff per-turn signal.
    # Default False.
    adapter_use_facts_diff_ir = bool(
        env_block.get("use_facts_diff_intermediate_reward", False)
    )
    # WebShop dense-signal opt-in: synthesize a per-step attribute-progress
    # intermediate reward for the V-head. Default False; WebShop branch only.
    adapter_use_attr_progress_ir = bool(
        env_block.get("use_attribute_progress_intermediate_reward", False)
    )

    def env_factory():
        # Thread each adapter's env-specific intermediate-reward opt-ins.
        extra_kwargs: dict = {}
        if env_name == "alfworld":
            extra_kwargs["use_textworld_intermediate_reward"] = adapter_use_tw_ir
            extra_kwargs["use_facts_diff_intermediate_reward"] = adapter_use_facts_diff_ir
        elif env_name == "webshop":
            extra_kwargs["use_attribute_progress_intermediate_reward"] = (
                adapter_use_attr_progress_ir
            )
        return adapter_cls(
            max_steps=adapter_max_steps,
            observation_mode=adapter_obs_mode,
            task_split=adapter_task_split,
            env_kwargs=dict(cfg_env_kwargs),
            **extra_kwargs,
        )

    # JSON-config-driven vs legacy flag-driven trainer/collector.
    if cfg_dict is not None:
        from src.trainers.train_hgpo import build_trainer_from_config

        # Method D (counterfactual) needs runner/env/renderer/parser threaded
        # through for its alt rollouts; Methods A/B/C ignore these kwargs.
        (
            trainer,
            _refresh_fn,
            turnrd_emit_path,
            turnrd_embedder,
            judge_decomposer,
        ) = build_trainer_from_config(
            cfg_dict,
            policy=policy,
            runner=runner,
            env_factory=env_factory,
            prompt_renderer=prompt_renderer,
            action_parser=action_parser,
            sampling_factory=SamplingParams,
        )
        # Tell the trainer the round index for the V-baseline annealing schedule.
        trainer.cfg.v_baseline_round_idx = int(round_idx)
        # When set, the producer emits goal_text/goal_emb feeding the V-head's
        # FiLM path. Both default False.
        _emit_goal_text = bool(
            (cfg_dict.get("turnrd") or {}).get("emit_goal_text", False)
        )
        _emit_goal_emb = bool(
            (cfg_dict.get("turnrd") or {}).get("emit_goal_emb", False)
        )
        collector = RolloutCollector(
            runner=runner,
            env_factory=env_factory,
            prompt_renderer=prompt_renderer,
            action_parser=action_parser,
            cfg=RolloutCollectorConfig(max_turns=max_turns),
            turnrd_emit_path=turnrd_emit_path,
            turnrd_embedder=turnrd_embedder,
            judge_decomposer=judge_decomposer,
            round_idx=int(round_idx),
            turnrd_emit_goal_text=_emit_goal_text,
            turnrd_emit_goal_emb=_emit_goal_emb,
        )
    else:
        collector = RolloutCollector(
            runner=runner,
            env_factory=env_factory,
            prompt_renderer=prompt_renderer,
            action_parser=action_parser,
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

    # rollout_temperature affects only training rollouts; eval stays greedy.
    # Lower values (e.g. 0.7) reduce dead-K on saturated policies.
    sampling = SamplingParams(
        n=1, temperature=rollout_temperature, top_p=0.95,
        max_tokens=48,
        return_logprobs=True, seed=None,  # fresh sampling each call -> diverse K trajectories
    )

    if sft_loaded and use_sft_as_ref:
        n = trainer.snapshot_current_lora_as_ref()
        print(f">>> Snapshotted {n} LoRA modules as KL reference (RL-from-SFT)")

    # Contiguous task-id slice [task_id_offset, task_id_offset + n_episodes).
    task_ids = [int(task_id_offset) + i for i in range(int(n_episodes))]
    n_episodes_actual = len(task_ids)
    print(
        f">>> Iterating {n_episodes_actual} task ids "
        f"(contiguous slice [{task_id_offset}, {task_id_offset + n_episodes}))"
    )

    # Config snapshot for train_log.json.
    def _config_snapshot() -> dict:
        # Surface the TurnRD ckpt-load outcome so the manifest can verify the
        # round N-1 ckpt reached round N.
        _turnrd_refresh = None
        try:
            _decomposer = getattr(trainer, "decomposer", None)
            if _decomposer is not None:
                _turnrd_refresh = getattr(_decomposer, "_last_refresh_status", None)
        except Exception:
            _turnrd_refresh = None
        return {
            "env_name": env_name,
            "env_kwargs": cfg_env_kwargs,
            "n_episodes": n_episodes, "K": k, "max_turns": max_turns,
            "task_id_offset": task_id_offset,
            "sync_every": sync_every, "run_name": run_name,
            "sft_adapter": sft_adapter,
            "turnrd_refresh": _turnrd_refresh,
            "n_episodes_actual": n_episodes_actual,
        }

    log: list[dict] = []
    overall_start = time.time()

    for ep, task_id in enumerate(task_ids):
        ep_t0 = time.time()
        try:
            group, cstats = collector.collect_group(
                task_id=task_id, env_name=env_name, K=k, sampling=sampling
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
                "consistency": round(stats.consistency, 6),  # python residual (always 0)
                "consistency_t": round(stats.consistency_t, 6),  # actual gradient-bearing C3 loss
                "total_loss": round(stats.total_loss, 6),
                "observed_kl": round(stats.observed_kl, 6),
                "kl_coef": round(stats.kl_coef, 6),
                "grad_norm": round(stats.grad_norm, 6),
                "turnrd_grad_norm": round(stats.turnrd_grad_norm, 6),
                "mean_traj_adv": round(stats.mean_traj_adv, 6),
                "mean_turn_adv": round(stats.mean_turn_adv, 6),
                "cls_query_norm": round(stats.cls_query_norm, 6),
                "alpha_mean": round(stats.alpha_mean, 6),
                "alpha_var": round(stats.alpha_var, 6),
                "alpha_max": round(stats.alpha_max, 6),
                "alpha_entropy": round(stats.alpha_entropy, 6),
                "alpha_progress_corr": round(stats.alpha_progress_corr, 6),
                # Tier-4 real-signal columns (see TrainStepStats)
                "std_reward_group": round(stats.std_reward_group, 6),
                "dead_K_group": int(stats.dead_K_group),
                "mean_abs_traj_adv": round(stats.mean_abs_traj_adv, 6),
                "std_traj_adv": round(stats.std_traj_adv, 6),
                "mean_abs_adv_token": round(stats.mean_abs_adv_token, 6),
                "elapsed_s": round(time.time() - ep_t0, 2),
            }
            log.append(row)
            print(
                f"ep={ep:03d} task={task_id} "
                f"R={row['mean_reward']:.3f}\u00b1{row['std_reward']:.3f} "
                f"loss={row['policy_loss']:+.4f} "
                f"kl={row['observed_kl']:+.4f} kl_coef={row['kl_coef']:.4f} "
                f"gn={row['grad_norm']:.3f} "
                f"aT={row['mean_abs_adv_token']:.3f} dK={row['dead_K_group']} "
                f"t={row['elapsed_s']}s"
            )

            if sync_every > 0 and (ep + 1) % sync_every == 0:
                # Stream merged LoRA weights one tensor at a time to avoid a
                # deepcopy memory spike, then release buffers.
                runner.sync_weights(policy.iter_merged_weights())
                import gc
                import torch as _torch
                gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()

            # Persist every episode so a crash still leaves a useful artifact.
            with open(os.path.join(run_dir, "train_log.json"), "w") as f:
                json.dump({"rows": log, "config": _config_snapshot()}, f, indent=2)
            volume.commit()
        except Exception as exc:
            # A non-finite LoRA merge leaves vLLM half-updated (split-brain
            # policy), so re-raise to abort the round instead of silently
            # corrupting every later rollout.
            if isinstance(exc, LoRAMergeNonFiniteError):
                print(f"ep={ep} FATAL (LoRA merge): {exc!r}; aborting round.")
                log.append({"episode": ep, "task_id": task_id, "fatal_error": repr(exc)})
                with open(os.path.join(run_dir, "train_log.json"), "w") as f:
                    json.dump({"rows": log, "config": _config_snapshot()}, f, indent=2)
                volume.commit()
                raise
            print(f"ep={ep} CRASHED: {exc!r}")
            # Capture the full traceback into the log row; Modal truncates
            # `app logs`, so the print above is often dropped.
            import traceback as _tb
            _tb_str = _tb.format_exc()
            log.append({
                "episode": ep,
                "task_id": task_id,
                "error": repr(exc),
                "traceback": _tb_str,
            })
            # A mid-step crash skips the empty_cache above, leaving activations
            # cached and risking a cascading OOM; flush here so the next ep is clean.
            try:
                import gc as _gc
                import torch as _torch
                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            continue

    total_elapsed = round(time.time() - overall_start, 2)

    rewards = [r["mean_reward"] for r in log if "mean_reward" in r]
    early = rewards[: max(1, len(rewards) // 10)]
    late = rewards[-max(1, len(rewards) // 10):]

    # Held-out eval pass on a disjoint task range, greedy sampling
    eval_block: dict | None = None
    if eval_episodes > 0:
        print(
            f">>> Eval pass: {eval_episodes} held-out tasks at offset "
            f"{eval_task_id_base} (greedy sampling)"
        )
        # Modal-bug bypass: the train-phase vLLM accumulates CUDA corruption
        # after many sync_weights cycles, so don't reuse it for eval. Shut it
        # down FIRST (before save_adapter, which else inherits the poisoned
        # CUDA context), then save the adapter, merge LoRA into the base, and
        # boot a fresh vLLM on the merged model.
        print(">>> [pre-merge] Shutting down train-phase vLLM (FIRST, before save_adapter — un-poisons CUDA)")
        try:
            runner.shutdown()
        except Exception as exc:
            print(f">>> WARNING: vLLM shutdown raised {exc!r}; proceeding anyway")
        # Force GC + CUDA cache flush BEFORE we touch CUDA again for save.
        import gc
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
        except Exception:
            pass
        if save_adapter_out:
            print(f">>> [pre-merge] Saving trained LoRA adapter to {save_adapter_out}")
            os.makedirs(save_adapter_out, exist_ok=True)
            policy.save_adapter(save_adapter_out)
            volume.commit()  # persist to Modal volume so next round can load it
        # Merge LoRA into base for the eval vLLM (destroys policy.model's PEFT
        # structure; only safe because training is done this round).
        import shutil
        eval_merged_dir = "/tmp/eval_merged"
        if os.path.exists(eval_merged_dir):
            shutil.rmtree(eval_merged_dir)
        os.makedirs(eval_merged_dir, exist_ok=True)
        print(f">>> [pre-merge] Merging LoRA into base for eval-vLLM (fresh-state bypass)")
        merged_model = policy.model.merge_and_unload(progressbar=False)
        merged_model.save_pretrained(eval_merged_dir, safe_serialization=True)
        policy.tokenizer.save_pretrained(eval_merged_dir)
        print(f">>> [pre-merge] Merged model saved to {eval_merged_dir}")
        del merged_model
        # GC before booting the eval vLLM (merge+save materialized a full model).
        gc.collect()
        # Drop the shut-down train-vLLM ref so its GPU memory is reclaimed before
        # the new vLLM's profile_run (else its profile assertion trips).
        runner = None  # type: ignore[assignment]
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
                # vLLM 0.6.3.post1's profile_run asserts peak_memory > 0, which
                # fails when mem_get_info() is unchanged across its fake forward
                # (cached blocks). Reset stats + force a real alloc/free so the
                # next mem_get_info() registers a non-zero delta.
                _torch.cuda.reset_peak_memory_stats()
                _torch.cuda.reset_accumulated_memory_stats()
                _dummy = _torch.empty(64 * 1024 * 1024, dtype=_torch.uint8, device="cuda")
                del _dummy
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
                _free_b, _total_b = _torch.cuda.mem_get_info()
                print(
                    f">>> [pre-merge] post-cleanup CUDA "
                    f"free={_free_b / 1e9:.2f} GB / "
                    f"total={_total_b / 1e9:.2f} GB"
                )
        except Exception as _exc:
            print(f">>> [pre-merge] post-cleanup raised {_exc!r}; proceeding anyway")
        # Boot a fresh vLLM on the merged model. num_gpu_blocks_override=8000
        # bypasses vLLM's profile assertion on the train->eval handoff.
        print(">>> [pre-merge] Booting fresh eval-vLLM with merged weights")
        runner = VLLMRunner(
            VLLMRunnerConfig(
                model_name=eval_merged_dir,
                gpu_memory_utilization=gpu_mem_util,
                max_model_len=2048,
                download_dir="/vol/hf_cache",
                enforce_eager=True,
                seed=42,
                num_gpu_blocks_override=8000,
            )
        )

        eval_sampling = SamplingParams(
            n=1, temperature=0.0, top_p=1.0,
            max_tokens=48,
            return_logprobs=False, seed=42,
        )
        # One eval collector reused across episodes (env pool built once).
        # No producer hook so eval rollouts don't pollute the replay buffer.
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
                    task_id=task_id, env_name=env_name, K=1, sampling=eval_sampling
                )
                eval_returns.extend(eval_cstats.final_rewards)
                eval_turns.append(eval_cstats.total_turns)
                eval_completed += eval_cstats.completed
                eval_truncated += eval_cstats.truncated
            except Exception as exc:
                # Per-ep eval crashes don't abort the pass; log + skip.
                import traceback
                print(
                    f"eval ep={j} task={task_id} CRASHED: {exc!r}\n"
                    f"  traceback:\n{traceback.format_exc()}"
                )
                eval_crashed += 1
                continue
        eval_elapsed = round(time.time() - eval_t0, 2)
        if eval_returns:
            mean_R = sum(eval_returns) / len(eval_returns)
            pct_success = sum(1 for r in eval_returns if r > 0) / len(eval_returns)
            std_R = (
                sum((r - mean_R) ** 2 for r in eval_returns) / len(eval_returns)
            ) ** 0.5
        else:
            mean_R = 0.0
            pct_success = 0.0
            std_R = 0.0
        eval_block = {
            "n_episodes_attempted": eval_episodes,
            "n_episodes_ok": len(eval_returns),
            "n_episodes_crashed": eval_crashed,
            "task_id_base": eval_task_id_base,
            "task_id_range": [eval_task_id_base, eval_task_id_base + eval_episodes],
            "sampling": (
                f"greedy (T=0.0, top_p=1.0, max_tokens=48)"
            ),
            "K": 1,
            "elapsed_s": eval_elapsed,
            "avg_return": round(mean_R, 4),
            "std_return": round(std_R, 4),
            "pct_success": round(pct_success, 4),
            "completed": eval_completed,
            "truncated": eval_truncated,
            "n_turns_avg": round(sum(eval_turns) / max(1, len(eval_turns)), 2),
            "max_turns": max_turns,
        }
        print(
            f">>> Eval done: avg_R={eval_block['avg_return']:.4f} "
            f"(\u00b1{eval_block['std_return']:.4f}) | "
            f"pct_success={eval_block['pct_success']:.3f} | "
            f"ok={eval_block['n_episodes_ok']}/{eval_episodes} | "
            f"elapsed={eval_elapsed}s"
        )

        # Persist eval into the same train_log.json for post-hoc aggregation.
        with open(os.path.join(run_dir, "train_log.json"), "w") as f:
            json.dump(
                {"rows": log, "config": _config_snapshot(), "eval": eval_block},
                f, indent=2,
            )
        volume.commit()

    # Save the trained LoRA adapter so the next round can load it
    # (default "" = each round resets to the SFT warm-start).
    if save_adapter_out and not (eval_episodes > 0):
        # Only save here if the pre-merge path above didn't already (merge_and_unload
        # since destroyed policy.model's PEFT structure).
        print(f">>> Saving trained LoRA adapter to {save_adapter_out}")
        os.makedirs(save_adapter_out, exist_ok=True)
        policy.save_adapter(save_adapter_out)
        volume.commit()

    summary = {
        "run_dir": run_dir,
        "env_name": env_name,
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
    if eval_block is not None:
        summary["eval_avg_return"] = eval_block["avg_return"]
        summary["eval_pct_success"] = eval_block["pct_success"]
        summary["eval_n_episodes_ok"] = eval_block["n_episodes_ok"]
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()
    return summary


# Two image-bound entrypoints. Modal image binding is decorator-time so
# a single @app.function cannot switch images at call time.

@app.function(image=webshop_image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, secrets=maybe_openai_secret(), timeout=120 * 60)
def train_loop_webshop(
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
    eval_episodes: int = 50,
    eval_task_id_base: int = 6500,
    round_idx: int = 0,
    save_adapter_out: str = "",
    rollout_temperature: float = 1.0,
) -> dict:
    return _train_loop_impl(
        env_name="webshop",
        n_episodes=n_episodes, k=k, max_turns=max_turns,
        task_id_offset=task_id_offset, num_products=num_products,
        sync_every=sync_every, run_name=run_name,
        sft_adapter=sft_adapter, use_sft_as_ref=use_sft_as_ref,
        kl_warmup_episodes=kl_warmup_episodes, gpu_mem_util=gpu_mem_util,
        config=config,
        eval_episodes=eval_episodes, eval_task_id_base=eval_task_id_base,
        round_idx=round_idx,
        save_adapter_out=save_adapter_out,
        rollout_temperature=rollout_temperature,
    )


@app.function(image=alfworld_image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, secrets=maybe_openai_secret(), timeout=240 * 60)
def train_loop_alfworld(
    n_episodes: int = 50,
    k: int = 4,
    max_turns: int = 30,
    task_id_offset: int = 0,
    num_products: int = 0,  # unused for AlfWorld - kept for shared signature
    sync_every: int = 1,
    run_name: str = "method_alfworld_smoke",
    sft_adapter: str = "",
    use_sft_as_ref: bool = True,
    kl_warmup_episodes: int = 0,
    gpu_mem_util: float = 0.30,
    config: str = "",
    # AlfWorld wraps task_id with `% len(games)`, so any base is safe; 6500
    # (mirroring WebShop) keeps eval outside training ranges.
    eval_episodes: int = 50,
    eval_task_id_base: int = 6500,
    round_idx: int = 0,
    save_adapter_out: str = "",
    rollout_temperature: float = 1.0,
) -> dict:
    return _train_loop_impl(
        env_name="alfworld",
        n_episodes=n_episodes, k=k, max_turns=max_turns,
        task_id_offset=task_id_offset, num_products=num_products,
        sync_every=sync_every, run_name=run_name,
        sft_adapter=sft_adapter, use_sft_as_ref=use_sft_as_ref,
        kl_warmup_episodes=kl_warmup_episodes, gpu_mem_util=gpu_mem_util,
        config=config,
        eval_episodes=eval_episodes, eval_task_id_base=eval_task_id_base,
        round_idx=round_idx,
        save_adapter_out=save_adapter_out,
        rollout_temperature=rollout_temperature,
    )


# Backward-compat alias for `train_loop_smoke` (WebShop entrypoint).
train_loop_smoke = train_loop_webshop


@app.local_entrypoint()
def main(
    env_name: str = "webshop",
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
    eval_episodes: int = 50,
    eval_task_id_base: int = 6500,
    round_idx: int = 0,
    save_adapter_out: str = "",
    rollout_temperature: float = 1.0,
) -> None:
    """Local entrypoint: dispatch to the right `@app.function` by `--env-name` (default webshop)."""
    import json as _json

    if env_name == "webshop":
        fn = train_loop_webshop
    elif env_name == "alfworld":
        fn = train_loop_alfworld
    else:
        raise ValueError(
            f"--env-name must be 'webshop' or 'alfworld'; got {env_name!r}."
        )

    print(_json.dumps(
        fn.remote(
            n_episodes=n_episodes, k=k, max_turns=max_turns,
            task_id_offset=task_id_offset, num_products=num_products,
            sync_every=sync_every, run_name=run_name,
            sft_adapter=sft_adapter,
            use_sft_as_ref=use_sft_as_ref,
            kl_warmup_episodes=kl_warmup_episodes,
            gpu_mem_util=gpu_mem_util,
            config=config,
            eval_episodes=eval_episodes,
            eval_task_id_base=eval_task_id_base,
            round_idx=round_idx,
            save_adapter_out=save_adapter_out,
            rollout_temperature=rollout_temperature,
        ),
        indent=2, default=str,
    ))
