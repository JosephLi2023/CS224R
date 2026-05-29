"""Modal app: greedy rollouts + TurnRD v-projection per-turn credits (probe JSONL).

Used with `scripts/render_turnrd_credits_latex.py` for qualitative credit figures.

Example (WebShop attention R7, 85% peak):
  modal run infra/app_probe_turnrd.py::probe_turnrd_webshop \\
    --lora-adapter /vol/checkpoints/webshop_attention_v1_seed11_round07_adapter \\
    --turnrd-ckpt /vol/cache/TurnRDV2_webshop_SOTA_10round_mlpr32_v1/ckpt.pt \\
    --config /workspace/configs/TurnRDV2_webshop_SOTA_10round_mlpr32_v1.json \\
    --eval-task-id-base 6500 --n-tasks 40 --out-name webshop_attention_r07_probe
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import webshop_image

app = modal.App("cs224r-hgpo-probe-turnrd")


def _compute_v_projection_credits(
    v_t_raw,
    final_R: float,
    mask_row,
    *,
    v_projection_clamp: float,
):
    """Match HGPOTrainer v-projection (configs use v_projection_clamp=2.0)."""
    import torch

    R = torch.tensor([final_R], dtype=v_t_raw.dtype, device=v_t_raw.device)
    proj_mask = mask_row.to(dtype=v_t_raw.dtype)
    v_clamped = v_t_raw.clamp(-v_projection_clamp, v_projection_clamp) * proj_mask
    T_active = proj_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    v_sum = v_clamped.sum(dim=-1, keepdim=True)
    adjustment = (v_sum - R.unsqueeze(-1)) / T_active
    per_turn = (v_clamped - adjustment) * proj_mask
    return per_turn[0].detach().cpu().tolist()


@app.function(
    image=webshop_image,
    gpu="A100-80GB",
    volumes={VOLUME_MOUNT: volume},
    timeout=90 * 60,
)
def probe_turnrd_webshop(
    *,
    lora_adapter: str,
    turnrd_ckpt: str,
    config: str,
    eval_task_id_base: int = 6500,
    n_tasks: int = 40,
    max_turns: int = 15,
    gpu_mem_util: float = 0.35,
    out_name: str = "webshop_attention_probe",
    v_projection_clamp: float = 2.0,
) -> dict[str, Any]:
    import gc
    import hashlib
    import site
    import sys

    import torch

    # Match `infra/app_train_loop.py` WebShop import path (volume-resident install).
    sys.path.insert(0, "/vol/code/webshop")
    sys.path.insert(0, "/workspace")
    os.environ["PYTHONUSERBASE"] = "/vol/webshop_pyuser"
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    site.addsitedir(f"/vol/webshop_pyuser/lib/python{pyver}/site-packages")

    from src.algorithms.grpo.collectors import RolloutCollector, RolloutCollectorConfig
    from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer
    from src.envs.prompts.react_webshop import parse_react_action, render_webshop_turn_prompt
    from src.envs.webshop_adapter import WebShopAdapter
    from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig
    from src.policy.vllm_runner import SamplingParams, VLLMRunner, VLLMRunnerConfig
    from src.trainers.train_hgpo import build_trainer_from_config
    from src.turnrd.embedders import policy_hidden_state_embedder

    volume.reload()

    with open(config) as fh:
        cfg_dict = json.load(fh)

    env_block = dict(cfg_dict.get("env") or {})
    cfg_env_kwargs = dict(env_block.get("env_kwargs") or {})
    if not cfg_env_kwargs:
        cfg_env_kwargs = {"num_products": 1000}
    adapter_max_steps = int(env_block.get("max_steps", max_turns))
    use_attr_ir = bool(env_block.get("use_attribute_progress_intermediate_reward", False))

    pol = cfg_dict.get("policy") or {}
    lora_kwargs: dict = {}
    if "lora_rank" in pol:
        r = int(pol["lora_rank"])
        lora_kwargs = {"lora_r": r, "lora_alpha": 2 * r}
    if pol.get("lora_target_modules"):
        lora_kwargs["lora_target_modules"] = list(pol["lora_target_modules"])

    print(">>> Loading LoRAPolicy + adapter:", lora_adapter)
    policy = LoRAPolicy(LoRAPolicyConfig(cache_dir="/vol/hf_cache", **lora_kwargs))
    policy.load_adapter(lora_adapter)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(">>> Booting vLLM")
    runner = VLLMRunner(
        VLLMRunnerConfig(
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=2048,
            download_dir="/vol/hf_cache",
            enforce_eager=True,
            seed=42,
        )
    )
    runner.sync_weights(policy.iter_merged_weights())

    def env_factory():
        return WebShopAdapter(
            max_steps=adapter_max_steps,
            observation_mode=str(env_block.get("observation_mode", "text")),
            task_split=str(env_block.get("task_split", "train")),
            env_kwargs=cfg_env_kwargs,
            use_attribute_progress_intermediate_reward=use_attr_ir,
        )

    trainer, _, _, turnrd_embedder, _ = build_trainer_from_config(
        cfg_dict, policy=policy, runner=runner, env_factory=env_factory,
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        sampling_factory=SamplingParams,
    )
    decomposer = trainer.decomposer
    if not isinstance(decomposer, TurnRDDecomposer):
        raise TypeError(f"Expected TurnRDDecomposer, got {type(decomposer)}")

    print(">>> Loading TurnRD ckpt:", turnrd_ckpt)
    sd = torch.load(turnrd_ckpt, map_location=next(decomposer.model.parameters()).device, weights_only=True)
    decomposer.load_state_dict(sd)
    decomposer.model.eval()

    embedder = policy_hidden_state_embedder(policy)

    collector = RolloutCollector(
        runner=runner,
        env_factory=env_factory,
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        cfg=RolloutCollectorConfig(max_turns=max_turns),
    )
    sampling = SamplingParams(
        n=1, temperature=0.0, top_p=1.0, max_tokens=48, return_logprobs=False, seed=42,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = f"/vol/manifests/turnrd_probe"
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = f"{out_dir}/{out_name}_{ts}.jsonl"

    cfg_bytes = json.dumps(cfg_dict, sort_keys=True).encode()
    t0 = time.time()
    n_ok = n_fail = n_crash = 0
    max_drift = 0.0

    with open(jsonl_path, "w") as out_fh:
        for j in range(n_tasks):
            task_id = eval_task_id_base + j
            try:
                group, _ = collector.collect_group(
                    task_id=task_id, env_name="webshop", K=1, sampling=sampling,
                )
                if not group.trajectories or not group.trajectories[0].turns:
                    n_crash += 1
                    continue
                traj = group.trajectories[0]
                R = float(traj.final_reward)
                success = R > 0.0

                # TurnRD forward for credits
                per_traj_embeds = []
                with torch.no_grad():
                    e = embedder(traj)
                    per_traj_embeds.append(e.detach())
                T_i = per_traj_embeds[0].shape[0]
                D = per_traj_embeds[0].shape[1]
                target_device = next(decomposer.model.parameters()).device
                target_dtype = next(decomposer.model.parameters()).dtype
                stacked = torch.zeros(1, T_i, D, dtype=target_dtype, device=target_device)
                attn_mask = torch.ones(1, T_i, dtype=torch.long, device=target_device)
                stacked[0, :T_i] = per_traj_embeds[0].to(device=target_device, dtype=target_dtype)
                with torch.no_grad():
                    out = decomposer.model(stacked, attn_mask)
                credits = _compute_v_projection_credits(
                    out.predicted_per_turn_R[0, :T_i],
                    R,
                    attn_mask[0, :T_i].to(dtype=out.predicted_per_turn_R.dtype),
                    v_projection_clamp=v_projection_clamp,
                )
                credit_sum = sum(credits)
                drift = abs(credit_sum - R)
                max_drift = max(max_drift, drift)
                predicted_R = float(out.predicted_R[0].item())

                turns_out = []
                for t_idx, turn in enumerate(traj.turns):
                    ir = turn.intermediate_reward
                    turns_out.append({
                        "turn_idx": t_idx,
                        "observation_text": turn.observation_text,
                        "action_text": turn.action_text,
                        "raw_env_reward": float(turn.raw_env_reward),
                        "intermediate_reward": float(ir) if ir is not None else None,
                        "v_t_raw": float(out.predicted_per_turn_R[0, t_idx].item()),
                        "v_t_clamped": float(
                            max(-v_projection_clamp, min(v_projection_clamp, out.predicted_per_turn_R[0, t_idx].item()))
                        ),
                        "credit": float(credits[t_idx]),
                    })

                row = {
                    "task_id": task_id,
                    "env_name": "webshop",
                    "final_reward": R,
                    "success": success,
                    "n_turns": len(traj.turns),
                    "predicted_R_from_model": predicted_R,
                    "credit_sum": credit_sum,
                    "v_projection_clamp": v_projection_clamp,
                    "turns": turns_out,
                }
                out_fh.write(json.dumps(row) + "\n")
                if success:
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as exc:
                import traceback
                print(f"task {task_id} CRASHED: {exc!r}\n{traceback.format_exc()}")
                n_crash += 1

    try:
        runner.shutdown()
    except Exception:
        pass

    summary = {
        "lora_adapter_path": lora_adapter,
        "turnrd_ckpt_path": turnrd_ckpt,
        "config_path": config,
        "config_sha256_16": hashlib.sha256(cfg_bytes).hexdigest()[:16],
        "v_projection_clamp": v_projection_clamp,
        "eval_task_id_base": eval_task_id_base,
        "n_tasks_requested": n_tasks,
        "n_emitted": n_ok + n_fail,
        "n_success": n_ok,
        "n_failure": n_fail,
        "n_crashed": n_crash,
        "max_sum_to_R_drift": max_drift,
        "elapsed_s": round(time.time() - t0, 2),
        "jsonl_path": jsonl_path,
        "embedder_source": "policy_hidden_state_embedder (mean-pool last_hidden_state, LoRA enabled)",
    }
    summary_path = jsonl_path.replace(".jsonl", ".summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    volume.commit()
    print(">>> Wrote", jsonl_path)
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(
    lora_adapter: str = "/vol/checkpoints/webshop_attention_v1_seed11_round07_adapter",
    turnrd_ckpt: str = "/vol/cache/TurnRDV2_webshop_SOTA_10round_mlpr32_v1/ckpt.pt",
    config: str = "/workspace/configs/TurnRDV2_webshop_SOTA_10round_mlpr32_v1.json",
    eval_task_id_base: int = 6500,
    n_tasks: int = 40,
    max_turns: int = 15,
    gpu_mem_util: float = 0.35,
    out_name: str = "webshop_attention_r07_probe",
) -> None:
    summary = probe_turnrd_webshop.remote(
        lora_adapter=lora_adapter,
        turnrd_ckpt=turnrd_ckpt,
        config=config,
        eval_task_id_base=eval_task_id_base,
        n_tasks=n_tasks,
        max_turns=max_turns,
        gpu_mem_util=gpu_mem_util,
        out_name=out_name,
    )
    print(json.dumps(summary, indent=2))
