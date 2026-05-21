from __future__ import annotations

import itertools
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import modal  # type: ignore[import-not-found]
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer  # type: ignore[import-not-found]

from infra.app_alfworld_sft_gen import _build_alfworld_config_dict, _extract_expert_plan, _extract_won
from infra.common import VOLUME_MOUNT, volume
from infra.image import ALFWORLD_DATA_DIR, alfworld_image
from maxrodriguez.grpo.turn_level_reward_methods_todo import (
    SignedAttentionTransformer,
    load_signed_attention_transformer_checkpoint,
    save_signed_attention_transformer_checkpoint,
    train_signed_attention_transformer,
)
from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord
from src.datasets.sft_alfworld import synthesize_sft_target
from src.envs.alfworld_adapter import ALFWorldAdapter
from src.envs.prompts.react_alfworld import render_alfworld_turn_prompt

app = modal.App("maxrodriguez-signed-attention-transformer")

DEFAULT_BASE_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_ROOT = "/vol/checkpoints/maxrodriguez_milestone/signed_attention"
DEFAULT_MANIFEST_ROOT = "/vol/manifests/maxrodriguez/signed_attention"

SIGNED_ATTENTION_TRANSFORMER_GRID: dict[str, list[Any]] = {
    "epochs": [1],
    "learning_rate": [5.0e-5, 1.0e-4, 2.0e-4],
    "hidden_size": [64, 128],
    "n_layers": [1, 2],
    "n_heads": [4],
    "dropout": [0.0],
    "train_trajectories": [256],
    "val_trajectories": [64],
    "max_turns": [30],
    "seed": [42],
}

SIGNED_ATTENTION_TRANSFORMER_FINAL_SPACE: dict[str, list[Any]] = {
    "epochs": [3],
}

BEST_SIGNED_ATTENTION_TRANSFORMER_AFTER_SWEEP: dict[str, Any] = {
    "source_run_name": None,
    "selection_metric": None,
    "epochs": None,
    "learning_rate": None,
    "hidden_size": None,
    "n_layers": None,
    "n_heads": None,
    "dropout": None,
    "train_trajectories": None,
    "val_trajectories": None,
    "max_turns": None,
    "seed": None,
    "checkpoint_path": None,
}


@dataclass(frozen=True)
class SignedAttentionTrainSpec:
    run_name: str
    epochs: int
    learning_rate: float
    hidden_size: int
    n_layers: int
    n_heads: int
    dropout: float
    train_trajectories: int
    val_trajectories: int
    max_turns: int
    seed: int
    output_dir: str


def _name_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p").replace("-", "m")
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "p")


def make_signed_attention_run_name(spec: SignedAttentionTrainSpec) -> str:
    return (
        f"satf_{spec.run_name}"
        f"_e{spec.epochs}"
        f"_lr{_name_value(spec.learning_rate)}"
        f"_h{spec.hidden_size}"
        f"_L{spec.n_layers}"
        f"_H{spec.n_heads}"
        f"_tr{spec.train_trajectories}"
        f"_va{spec.val_trajectories}"
        f"_t{spec.max_turns}"
    )


def iter_signed_attention_train_specs(
    *,
    run_tag: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    max_specs: int | None = None,
) -> list[SignedAttentionTrainSpec]:
    keys = list(SIGNED_ATTENTION_TRANSFORMER_GRID.keys())
    specs: list[SignedAttentionTrainSpec] = []
    for values in itertools.product(*(SIGNED_ATTENTION_TRANSFORMER_GRID[key] for key in keys)):
        row = dict(zip(keys, values))
        base = SignedAttentionTrainSpec(
            run_name=run_tag,
            epochs=int(row["epochs"]),
            learning_rate=float(row["learning_rate"]),
            hidden_size=int(row["hidden_size"]),
            n_layers=int(row["n_layers"]),
            n_heads=int(row["n_heads"]),
            dropout=float(row["dropout"]),
            train_trajectories=int(row["train_trajectories"]),
            val_trajectories=int(row["val_trajectories"]),
            max_turns=int(row["max_turns"]),
            seed=int(row["seed"]),
            output_dir="",
        )
        run_name = make_signed_attention_run_name(base)
        specs.append(
            SignedAttentionTrainSpec(
                **{
                    **base.__dict__,
                    "run_name": run_name,
                    "output_dir": f"{output_root}/grid/{run_name}",
                }
            )
        )
        if max_specs is not None and len(specs) >= max_specs:
            break
    return specs


def build_final_signed_attention_train_spec(
    *,
    run_tag: str,
    learning_rate: float,
    hidden_size: int,
    n_layers: int,
    n_heads: int = 4,
    dropout: float = 0.0,
    train_trajectories: int = 256,
    val_trajectories: int = 64,
    max_turns: int = 30,
    seed: int = 42,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> SignedAttentionTrainSpec:
    base = SignedAttentionTrainSpec(
        run_name=f"{run_tag}_final",
        epochs=int(SIGNED_ATTENTION_TRANSFORMER_FINAL_SPACE["epochs"][0]),
        learning_rate=float(learning_rate),
        hidden_size=int(hidden_size),
        n_layers=int(n_layers),
        n_heads=int(n_heads),
        dropout=float(dropout),
        train_trajectories=int(train_trajectories),
        val_trajectories=int(val_trajectories),
        max_turns=int(max_turns),
        seed=int(seed),
        output_dir="",
    )
    run_name = make_signed_attention_run_name(base)
    return SignedAttentionTrainSpec(
        **{
            **base.__dict__,
            "run_name": run_name,
            "output_dir": f"{output_root}/final/{run_name}",
        }
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_alfworld_adapter(max_turns: int, split: str) -> ALFWorldAdapter:
    os.environ.setdefault("ALFWORLD_DATA", ALFWORLD_DATA_DIR)
    config = _build_alfworld_config_dict()
    config["dataset"]["num_eval_games"] = -1
    return ALFWorldAdapter(
        max_steps=max_turns,
        observation_mode="text",
        task_split=split,
        env_kwargs={"config": config},
        use_textworld_intermediate_reward=True,
        use_facts_diff_intermediate_reward=True,
    )


def _collect_expert_groups(
    *,
    tokenizer: Any,
    split: str,
    n_trajectories: int,
    max_turns: int,
    task_id_base: int,
) -> dict[str, Any]:
    adapter = _build_alfworld_adapter(max_turns=max_turns, split=split)
    groups: list[TrajectoryGroup] = []
    skipped = 0

    for ep in range(n_trajectories):
        state = adapter.reset(task_id=task_id_base + ep)
        history: list[Any] = []
        turns: list[TurnRecord] = []
        final_info = getattr(state, "raw_info", {}) or {}
        total_reward = 0.0

        for step_idx in range(max_turns):
            prompt = render_alfworld_turn_prompt(
                state,
                history,
                max_history_turns=3,
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            expert_plan = _extract_expert_plan(getattr(state, "raw_info", {}) or {})
            if not expert_plan:
                break

            action = str(expert_plan[0]).strip()
            target_ids = tokenizer(
                synthesize_sft_target(action),
                add_special_tokens=False,
            ).input_ids

            next_state, reward, done, info = adapter.step(action)
            final_info = info or {}
            total_reward += float(reward)
            intermediate_reward = (
                float(info["intermediate_reward"])
                if isinstance(info, dict) and info.get("intermediate_reward") is not None
                else None
            )

            turns.append(
                TurnRecord(
                    turn_idx=step_idx,
                    observation_text=getattr(state, "observation_text", "") or "",
                    action_text=action,
                    raw_env_reward=float(reward),
                    action_token_ids=tuple(target_ids),
                    action_token_logprobs=(),
                    prompt_token_count=len(prompt_ids),
                    prompt_token_ids=tuple(prompt_ids[-2048:]),
                    intermediate_reward=intermediate_reward,
                )
            )
            history.append(
                SimpleNamespace(
                    observation_text=getattr(state, "observation_text", "") or "",
                    action_text=action,
                )
            )
            state = next_state
            if done:
                break

        won = _extract_won(final_info)
        if not turns:
            skipped += 1
            continue

        traj = Trajectory(
            task_id=str(task_id_base + ep),
            env_name="alfworld",
            turns=turns,
            final_reward=1.0 if won else float(total_reward),
        )
        groups.append(
            TrajectoryGroup(
                task_id=str(task_id_base + ep),
                env_name="alfworld",
                trajectories=[traj],
            )
        )

    return {
        "groups": groups,
        "n_groups": len(groups),
        "n_skipped": skipped,
    }


def _evaluate_signed_attention_transformer(
    model: SignedAttentionTransformer,
    groups: list[TrajectoryGroup],
    *,
    device: str,
) -> dict[str, Any]:
    if not groups:
        return {"n_groups": 0, "mse": None}

    model = model.to(torch.device(device))
    model.eval()
    losses: list[float] = []

    with torch.no_grad():
        for group in groups:
            for traj in group.trajectories:
                if not traj.turns:
                    continue

                progress = torch.tensor(
                    [
                        float(turn.intermediate_reward)
                        if turn.intermediate_reward is not None
                        else float(turn.raw_env_reward)
                        for turn in traj.turns
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                centered_progress = progress - progress.mean()
                scale = centered_progress.abs().max().clamp_min(1e-8)
                target = (centered_progress / scale).unsqueeze(0)

                features = torch.tensor(
                    [
                        [
                            float(turn.intermediate_reward)
                            if turn.intermediate_reward is not None
                            else float(turn.raw_env_reward),
                            float(turn.raw_env_reward),
                            float(turn.turn_idx) / max(len(traj.turns) - 1, 1),
                            float(len(turn.action_token_ids)),
                            float(turn.prompt_token_count),
                        ]
                        for turn in traj.turns
                    ],
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                mask = torch.ones((1, features.shape[1]), device=device, dtype=torch.bool)
                pred = model(features, mask=mask)
                losses.append(float(F.mse_loss(pred[mask], target[mask]).detach().cpu()))

    return {
        "n_groups": len(groups),
        "mse": round(sum(losses) / max(1, len(losses)), 6),
    }


@app.function(image=alfworld_image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=8 * 60 * 60)
def train_signed_attention_transformer_model(
    *,
    epochs: int,
    learning_rate: float,
    hidden_size: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    train_trajectories: int,
    val_trajectories: int,
    max_turns: int,
    seed: int,
    run_name: str,
    output_dir: str,
    base_model_path: str = DEFAULT_BASE_MODEL_PATH,
) -> dict[str, Any]:
    sys.path.insert(0, "/workspace")
    torch.manual_seed(seed)
    random.seed(seed)

    ckpt_dir = output_dir or f"{DEFAULT_OUTPUT_ROOT}/{run_name}_{_timestamp()}"
    os.makedirs(ckpt_dir, exist_ok=True)
    manifest_dir = f"{DEFAULT_MANIFEST_ROOT}/{run_name}_{_timestamp()}"
    os.makedirs(manifest_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        cache_dir="/vol/hf_cache",
        use_fast=True,
    )

    train_result = _collect_expert_groups(
        tokenizer=tokenizer,
        split="train",
        n_trajectories=int(train_trajectories),
        max_turns=int(max_turns),
        task_id_base=0,
    )
    val_result = _collect_expert_groups(
        tokenizer=tokenizer,
        split="eval_in_distribution",
        n_trajectories=int(val_trajectories),
        max_turns=int(max_turns),
        task_id_base=0,
    )

    model = SignedAttentionTransformer(
        input_size=5,
        hidden_size=int(hidden_size),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        dropout=float(dropout),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    losses = train_signed_attention_transformer(
        model,
        train_result["groups"],
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        device=device,
    )
    val_metrics = _evaluate_signed_attention_transformer(
        model,
        val_result["groups"],
        device=device,
    )

    ckpt_path = os.path.join(ckpt_dir, "signed_attention_transformer.pt")
    save_signed_attention_transformer_checkpoint(
        model,
        ckpt_path,
        hidden_size=int(hidden_size),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        dropout=float(dropout),
    )

    summary = {
        "run_name": run_name,
        "checkpoint_path": ckpt_path,
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "hidden_size": int(hidden_size),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "dropout": float(dropout),
        "train_trajectories_requested": int(train_trajectories),
        "train_trajectories_collected": int(train_result["n_groups"]),
        "train_trajectories_skipped": int(train_result["n_skipped"]),
        "val_trajectories_requested": int(val_trajectories),
        "val_trajectories_collected": int(val_result["n_groups"]),
        "val_trajectories_skipped": int(val_result["n_skipped"]),
        "max_turns": int(max_turns),
        "seed": int(seed),
        "val_mse": val_metrics["mse"],
        "n_training_updates": len(losses),
        "final_train_loss": round(losses[-1], 6) if losses else None,
        "elapsed_s": round(time.time() - t0, 2),
        "manifest_dir": manifest_dir,
    }
    artifact = {
        "summary": summary,
        "val_metrics": val_metrics,
        "train_losses": losses[-200:],
    }

    with open(os.path.join(ckpt_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(ckpt_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    with open(os.path.join(manifest_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(manifest_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    volume.commit()
    return summary


@app.function(image=alfworld_image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=2 * 60 * 60)
def evaluate_signed_attention_transformer_checkpoint(
    *,
    checkpoint_path: str,
    val_trajectories: int,
    max_turns: int,
    base_model_path: str = DEFAULT_BASE_MODEL_PATH,
) -> dict[str, Any]:
    sys.path.insert(0, "/workspace")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        cache_dir="/vol/hf_cache",
        use_fast=True,
    )
    val_result = _collect_expert_groups(
        tokenizer=tokenizer,
        split="eval_in_distribution",
        n_trajectories=int(val_trajectories),
        max_turns=int(max_turns),
        task_id_base=0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_signed_attention_transformer_checkpoint(checkpoint_path, device=device)
    return _evaluate_signed_attention_transformer(model, val_result["groups"], device=device)
