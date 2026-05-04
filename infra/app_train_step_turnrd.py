"""Modal A100 smoke: end-to-end Method-B HGPOTrainer.train_step on FakeWebShopEnv.

  modal run infra/app_train_step_turnrd.py::train_step_turnrd_smoke

Mirrors `infra/app_train_step.py` (the Day-5.5 flat-GRPO smoke) but
exercises the Day-13/14 Method-B path on real hardware:

  LoRAPolicy  → VLLMRunner.generate_rich → RolloutCollector
                                            ├─ TurnRD producer hook writes
                                            │  one JSONL row per non-empty
                                            │  trajectory (real
                                            │  policy_hidden_state_embedder,
                                            │  not the unit-test stub).
                                            └─ TrajectoryGroup → ...
            → HGPOTrainer (TurnRDDecomposer; lambda_consistency > 0)
              → compute_loss
                ├─ build_advantages (pure-Python; stats path)
                ├─ decompose_with_grad → α [K_real, T_max] (grad-tracking)
                ├─ consistency_loss_tensor on (α·R-derived) advantages
                └─ total = policy_loss + kl_term + cons_loss_t
              → backward
              → policy AdamW step + clip
              → TurnRD AdamW step + clip   ← Day-13 second optimizer
              → AdaptiveKLController.update
              → TrainStepStats

Assertions catch the new failure modes the unit tests can't reach on CPU
(dtype mismatch on bf16 hidden states, device mismatch between policy
and TurnRD, OOM from the second optimizer, gradient flow through
`cls_query` on real grads, replay JSONL written + parses).

Cost: ~$0.20 (A100 ~3-4 min — slightly slower than the Method-C smoke
because of the extra TurnRD forward/backward + the embedder pass).
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

app = modal.App("cs224r-hgpo-train-step-turnrd")


@app.function(
    image=image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=20 * 60
)
def train_step_turnrd_smoke(
    k: int = 2,
    max_turns: int = 3,
    lambda_consistency: float = 0.1,
    turnrd_lr: float = 1e-4,
    emit_replay: bool = True,
) -> dict:
    """One end-to-end Method-B train_step on real Qwen2.5-1.5B + A100.

    Args:
        k: number of trajectories per group (matches K in production).
        max_turns: cap turns per trajectory (small to keep cost low).
        lambda_consistency: > 0 to actually exercise the C3 reattach
            (the whole point of this smoke). 0 reduces it to the
            Method-C path with a TurnRDDecomposer instead of progress.
        turnrd_lr: separate AdamW lr for TurnRD parameters.
        emit_replay: when True, the producer writes a JSONL row per
            non-empty trajectory to /vol/cache/turnrd_replay_smoke.jsonl
            so we also exercise the embedder + dataset round-trip.
    """
    import os
    import time

    import torch  # type: ignore[import-not-found]

    from src.algorithms.grpo.collectors import RolloutCollector, RolloutCollectorConfig
    from src.algorithms.grpo.trainer import HGPOTrainer, HGPOTrainerConfig
    from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer
    from src.envs.fake_webshop import FakeWebShopEnv
    from src.envs.prompts.react_webshop import (
        parse_react_action,
        render_webshop_turn_prompt,
    )
    from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig
    from src.policy.vllm_runner import SamplingParams, VLLMRunner, VLLMRunnerConfig
    from src.turnrd.dataset import TurnRDReplayDataset
    from src.turnrd.embedders import policy_hidden_state_embedder
    from src.turnrd.model import TurnRD, TurnRDConfig

    # -----------------------------------------------------------------
    # 1. Policy + vLLM runner (same as the flat-GRPO smoke)
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # 2. TurnRD model + decomposer (real hidden_size from the policy)
    # -----------------------------------------------------------------

    input_dim = int(policy.model.config.hidden_size)  # 1536 for Qwen2.5-1.5B
    print(f">>> Building TurnRD (input_dim={input_dim} from policy.model.config)")

    torch.manual_seed(0)
    turnrd_model = TurnRD(
        TurnRDConfig(
            n_layers=2,           # small to keep this smoke cheap
            hidden_size=128,
            n_heads=4,
            max_turns=16,
            dropout=0.0,
        ),
        input_dim=input_dim,
    )
    # Place TurnRD on the same device + dtype the LoRAPolicy uses, so
    # the decomposer adapter's auto-cast logic doesn't have to fix
    # mismatches at every forward.
    policy_param = next(policy.model.parameters())
    turnrd_model.to(device=policy_param.device, dtype=torch.float32)
    embedder = policy_hidden_state_embedder(policy)
    decomposer = TurnRDDecomposer(model=turnrd_model, embedder=embedder)

    # Sentinel: capture cls_query BEFORE the train step. We assert later
    # that it CHANGES (proves cls_query.grad was non-zero AND the second
    # optimizer actually stepped — the C3 reattach end-to-end check).
    cls_query_before = turnrd_model.cls_query.detach().clone().cpu()

    # -----------------------------------------------------------------
    # 3. Collector with the Day-14 producer plumbing
    # -----------------------------------------------------------------

    replay_path = "/vol/cache/turnrd_replay_smoke.jsonl"
    if emit_replay and os.path.exists(replay_path):
        os.remove(replay_path)  # start fresh each smoke run

    collector = RolloutCollector(
        runner=runner,
        env_factory=lambda: FakeWebShopEnv(max_steps=8),
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        cfg=RolloutCollectorConfig(max_turns=max_turns),
        turnrd_emit_path=(replay_path if emit_replay else None),
        turnrd_embedder=(embedder if emit_replay else None),
        # Mode 1 in this smoke (no judge_decomposer wired).
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
                "prompt_token_ids missing — Day 5.5 wiring regressed"
            )

    # -----------------------------------------------------------------
    # 4. Build the Method-B trainer (the interesting bit: TurnRD object
    #    as decomposer + lambda_consistency > 0 → C3 reattach fires).
    # -----------------------------------------------------------------

    print(
        f">>> Building HGPOTrainer (decomposer=TurnRD, "
        f"lambda_consistency={lambda_consistency}, turnrd_lr={turnrd_lr})"
    )
    trainer = HGPOTrainer(
        policy=policy,
        decomposer=decomposer,  # the OBJECT, not .decompose
        cfg=HGPOTrainerConfig(
            alpha=0.5,
            lambda_consistency=lambda_consistency,
            clip_eps=0.2,
            learning_rate=1e-6,
            max_grad_norm=1.0,
            turnrd_lr=turnrd_lr,
        ),
    )
    assert trainer._decomposer_learnable is True, (
        "Trainer did not detect TurnRDDecomposer as learnable — "
        "Day-13 has_learnable_params getattr regressed"
    )

    trainable = policy.trainable_parameters()
    before_norms = [float(p.detach().norm().item()) for p in trainable[:4]]

    # -----------------------------------------------------------------
    # 5. train_step()
    # -----------------------------------------------------------------

    print(">>> train_step()")
    t1 = time.time()
    stats = trainer.train_step(group)
    step_elapsed = round(time.time() - t1, 2)

    after_norms = [float(p.detach().norm().item()) for p in trainable[:4]]
    param_deltas = [round(abs(a - b), 6) for a, b in zip(after_norms, before_norms)]

    # The Day-13 C3 reattach end-to-end check: cls_query MUST have moved.
    cls_query_after = turnrd_model.cls_query.detach().clone().cpu()
    cls_query_delta = float((cls_query_after - cls_query_before).abs().sum().item())

    # -----------------------------------------------------------------
    # 6. Assertions
    # -----------------------------------------------------------------

    assert not torch.isnan(torch.tensor(stats.total_loss)), "NaN total_loss"
    assert stats.grad_norm >= 0.0
    assert any(d > 0 for d in param_deltas), (
        "No LoRA parameter moved — optimizer step did not run"
    )
    # The C3 reattach must have flowed gradient through TurnRD AND the
    # second optimizer must have stepped. cls_query is the natural
    # gradient sink for the [CLS] cross-attention pool.
    assert cls_query_delta > 0.0, (
        f"cls_query did not move (delta={cls_query_delta}); "
        "C3 reattach is not flowing gradient back to TurnRD params, OR "
        "the second AdamW didn't step. Check "
        "compute_loss → decompose_with_grad → consistency_loss_tensor "
        "and the train_step() step-boundary block."
    )

    # Replay JSONL round-trip (only if we emitted).
    replay_rows = 0
    if emit_replay:
        ds = TurnRDReplayDataset(replay_path, mode=1)
        replay_rows = len(ds)
        # Embedder produced [T_i, D] with D == policy hidden_size.
        if replay_rows > 0:
            first = ds[0]
            assert len(first.turn_embeds[0]) == input_dim, (
                f"Producer emitted D={len(first.turn_embeds[0])} != "
                f"policy.hidden_size={input_dim}. The embedder + adapter "
                "dtype/device cast may have silently truncated."
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
        # Note: stats.consistency is the pure-Python value (always 0 for
        # the in-tensor reattach since it's reported from the
        # build_advantages stats path; the actual gradient-bearing
        # consistency term lives in `total_loss`).
        "consistency": stats.consistency,
        "total_loss": stats.total_loss,
        "observed_kl": stats.observed_kl,
        "kl_coef_after": stats.kl_coef,
        "grad_norm": stats.grad_norm,
        "mean_traj_adv": stats.mean_traj_adv,
        "mean_turn_adv": stats.mean_turn_adv,
        "param_norm_deltas_first4": param_deltas,
        # Day-13/14 extras:
        "cls_query_delta": round(cls_query_delta, 6),
        "input_dim_from_policy": input_dim,
        "replay_path": (replay_path if emit_replay else None),
        "replay_rows": replay_rows,
        "lambda_consistency": lambda_consistency,
        "turnrd_lr": turnrd_lr,
    }


@app.local_entrypoint()
def main(
    k: int = 2,
    max_turns: int = 3,
    lambda_consistency: float = 0.1,
    turnrd_lr: float = 1e-4,
    emit_replay: bool = True,
) -> None:
    import json as _json

    print(
        _json.dumps(
            train_step_turnrd_smoke.remote(
                k=k,
                max_turns=max_turns,
                lambda_consistency=lambda_consistency,
                turnrd_lr=turnrd_lr,
                emit_replay=emit_replay,
            ),
            indent=2,
            default=str,
        )
    )
