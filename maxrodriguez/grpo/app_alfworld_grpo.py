"""Max Rodriguez ALFWorld GRPO config builder and launcher.

Everything in this file is intentionally kept inside `maxrodriguez/`.

Core proposal math to preserve:

    A_H = alpha * A_turn + (1 - alpha) * A_traj

where:
    A_turn comes from a selected turn-level reward method
    A_traj is the normal GRPO group-normalized trajectory reward

Do not put Max-only experimental decomposers into shared `src/` until the
experimental methods have tests and we decide they belong in the main factory.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal  # type: ignore[import-not-found]

# Max milestone GRPO does not use the OpenAI judge backend, so avoid requiring
# the shared Modal secret on import and manual launch paths.
os.environ.setdefault("CS224R_SKIP_OPENAI_SECRET", "1")

from infra.app_train_loop import train_loop_alfworld


app = modal.App("maxrodriguez-alfworld-grpo")

CONFIG_OUTPUT_DIR = Path("maxrodriguez/configs/generated_grpo")
VOLUME_NAME = "cs224r-hgpo-vol"
VOLUME_CONFIG_ROOT = "/maxrodriguez/generated_grpo"
DEFAULT_SFT_ADAPTER = "/vol/checkpoints/best_full_sft_alfworld"
DEFAULT_SIGNED_ATTENTION_TRANSFORMER_CKPT = ""
GRID_DATA_ROOT = "/vol/data/alfworld_grid_subset/json_2.1.1"
FULL_DATA_ROOT = "$ALFWORLD_DATA/json_2.1.1"
GRID_NUM_TRAIN_GAMES = 200
GRID_NUM_EVAL_GAMES = 0
FULL_NUM_GAMES = -1
GRID_TASK_ID_STRIDE = 7
FULL_TASK_ID_STRIDE = 37


# Only vary the few hyperparameters most likely to matter at milestone time.
# The sweep should stay tight: method, alpha, policy LR, and KL coefficient.
# Everything else is fixed below unless we have concrete evidence it needs a
# dedicated search.
GRPO_SWEEP_AXES: dict[str, list[Any]] = {
    "alpha": [0.0, 0.1, 0.5, 0.9],
    "turn_reward_method": [
        "trajectory_only",
        "progress_delta",
        "signed_attention",
        "admissible_margin",
    ],
    "learning_rate": [1e-6, 2e-6, 5e-6],
    "kl_coeff": [0.02, 0.05],
}


# Milestone GRPO fixed defaults. These are intentionally not swept right now.
GRPO_FIXED_DEFAULTS: dict[str, Any] = {
    "clip_eps": 0.2,
    "K_trajectories_per_task": 4,
    "grad_accum_steps": 1,
    "max_tokens_per_microbatch": 2048,
    "kl_warmup_episodes": 5,
    "n_episodes": 100,
    "max_turns": 30,
    "dataset_size_mode": "grid",
}


POST_MILESTONE_TURN_REWARD_METHODS: list[str] = ["counterfactual_delta"]


TURN_REWARD_DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "trajectory_only": {},
    "progress_delta": {
        "terminal_bonus": 1.0,
    },
    "signed_attention": {
        "hidden_size": 128,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.0,
        "outcome_scale": 1.0,
        "failure_scale": 0.5,
        "heuristic_bias_scale": 0.25,
        "transformer_ckpt_path": DEFAULT_SIGNED_ATTENTION_TRANSFORMER_CKPT,
    },
    "admissible_margin": {
        "max_actions_to_score": 32,
        "normalize_margin": True,
        "max_seq_len": 2048,
        "score_normalization": "mean",
    },
    "counterfactual_delta": {
        "n_alternatives": 2,
        "requires_counterfactual_runner": True,
    },
}


# Fill after sweep. Include the turn reward method and alpha value together
# because alpha only means something relative to the chosen turn signal.
BEST_GRPO_HYPERPARAMS_AFTER_SWEEP: dict[str, Any] = {
    "source_run_name": None,
    "selection_metric": None,
    "alpha": None,
    "turn_reward_method": None,
    "learning_rate": None,
    "kl_coeff": None,
    "clip_eps": None,
    "K_trajectories_per_task": None,
    "grad_accum_steps": None,
    "max_tokens_per_microbatch": None,
    "kl_warmup_episodes": None,
    "n_episodes": None,
    "max_turns": None,
    "dataset_size_mode": None,
    "eval_seen_success": None,
    "eval_unseen_success": None,
}


@dataclass(frozen=True)
class GRPORunSpec:
    """Concrete run spec produced by the sweep launcher."""

    run_name: str
    sft_adapter: str
    alpha: float
    turn_reward_method: str
    n_episodes: int
    k: int
    max_turns: int
    learning_rate: float
    kl_coeff: float
    clip_eps: float
    grad_accum_steps: int
    max_tokens_per_microbatch: int
    kl_warmup_episodes: int
    dataset_size_mode: str = "grid"
    eval_episodes: int = GRID_NUM_EVAL_GAMES
    save_adapter_out: str = ""
    signed_attention_transformer_ckpt: str = ""
    task_id_stride: int = 1


def _name_value(value: object) -> str:
    """Make a short filesystem-safe value for generated run names."""
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p").replace("-", "m")
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "p")


def make_run_name(spec: GRPORunSpec) -> str:
    """Stable run name containing the sweep choices that matter most."""
    return (
        f"max_grpo_{spec.turn_reward_method}"
        f"_a{_name_value(spec.alpha)}"
        f"_lr{_name_value(spec.learning_rate)}"
        f"_kl{_name_value(spec.kl_coeff)}"
        f"_k{spec.k}"
        f"_t{spec.max_turns}"
        f"_{spec.dataset_size_mode}"
    )


def iter_grid_run_specs(
    *,
    sft_adapter: str = DEFAULT_SFT_ADAPTER,
    max_specs: int | None = None,
    allowed_methods: list[str] | None = None,
    signed_attention_transformer_ckpt: str = DEFAULT_SIGNED_ATTENTION_TRANSFORMER_CKPT,
) -> list[GRPORunSpec]:
    """Expand `GRPO_HYPERPARAMETER_GRID` into concrete run specs.

    `max_specs` is useful for smoke-checking the config writer without
    materializing the full sweep.
    """
    keys = list(GRPO_SWEEP_AXES.keys())
    specs: list[GRPORunSpec] = []
    allowed_method_set = (
        {method.strip() for method in allowed_methods if method.strip()}
        if allowed_methods is not None
        else None
    )
    for values in itertools.product(*(GRPO_SWEEP_AXES[key] for key in keys)):
        row = dict(zip(keys, values))
        row.update(GRPO_FIXED_DEFAULTS)
        alpha = float(row["alpha"])
        method = str(row["turn_reward_method"])
        if allowed_method_set is not None and method not in allowed_method_set:
            continue
        if alpha == 0.0 and method != "trajectory_only":
            continue
        if alpha > 0.0 and method == "trajectory_only":
            continue
        spec = GRPORunSpec(
            run_name="",
            sft_adapter=sft_adapter,
            alpha=alpha,
            turn_reward_method=method,
            n_episodes=int(row["n_episodes"]),
            k=int(row["K_trajectories_per_task"]),
            max_turns=int(row["max_turns"]),
            learning_rate=float(row["learning_rate"]),
            kl_coeff=float(row["kl_coeff"]),
            clip_eps=float(row["clip_eps"]),
            grad_accum_steps=int(row["grad_accum_steps"]),
            max_tokens_per_microbatch=int(row["max_tokens_per_microbatch"]),
            kl_warmup_episodes=int(row["kl_warmup_episodes"]),
            dataset_size_mode=str(row["dataset_size_mode"]),
            signed_attention_transformer_ckpt=(
                signed_attention_transformer_ckpt if method == "signed_attention" else ""
            ),
            task_id_stride=GRID_TASK_ID_STRIDE,
        )
        spec = GRPORunSpec(**{**spec.__dict__, "run_name": make_run_name(spec)})
        specs.append(spec)
        if max_specs is not None and len(specs) >= max_specs:
            break
    return specs


def build_config_for_run(spec: GRPORunSpec) -> dict[str, Any]:
    """Build the config dict consumed by `train_loop_alfworld`.

    The returned config is fully determined by the `GRPORunSpec`; grid mode
    uses smaller ALFWorld pools for sweeps, while full mode removes those caps.
    """
    if not (0.0 <= spec.alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {spec.alpha}")
    if spec.turn_reward_method not in GRPO_SWEEP_AXES["turn_reward_method"]:
        raise ValueError(
            f"unknown turn_reward_method={spec.turn_reward_method!r}; "
            "expected one from GRPO_SWEEP_AXES['turn_reward_method']"
        )
    if spec.dataset_size_mode not in {"grid", "full"}:
        raise ValueError("dataset_size_mode must be 'grid' or 'full'")

    method_cfg = dict(TURN_REWARD_DEFAULT_CONFIGS[spec.turn_reward_method])
    if spec.turn_reward_method == "signed_attention":
        if not spec.signed_attention_transformer_ckpt:
            raise ValueError(
                "signed_attention runs require GRPORunSpec.signed_attention_transformer_ckpt "
                "so the live trainer uses the trained transformer instead of the heuristic fallback."
            )
        method_cfg["transformer_ckpt_path"] = spec.signed_attention_transformer_ckpt
        method_cfg["device"] = "cuda"
    if spec.dataset_size_mode == "grid":
        num_train_games = FULL_NUM_GAMES
        num_eval_games = FULL_NUM_GAMES
        data_root = GRID_DATA_ROOT
    else:
        num_train_games = FULL_NUM_GAMES
        num_eval_games = FULL_NUM_GAMES
        data_root = FULL_DATA_ROOT
    task_id_stride = max(1, int(spec.task_id_stride))

    return {
        "run": {
            "name": spec.run_name,
            "output_dir": "experiments/manifests",
            "seed": 42,
        },
        "env": {
            "name": "alfworld",
            "max_steps": spec.max_turns,
            "observation_mode": "text",
            "task_split": "train",
            "use_textworld_intermediate_reward": False,
            "use_facts_diff_intermediate_reward": True,
            "env_kwargs": {
                "config": {
                    "dataset": {
                        "data_path": f"{data_root}/train",
                        "eval_id_data_path": f"{data_root}/valid_seen",
                        "eval_ood_data_path": f"{data_root}/valid_unseen",
                        "num_train_games": num_train_games,
                        "num_eval_games": num_eval_games,
                    },
                    "env": {
                        "type": "AlfredTWEnv",
                        "regen_game_files": False,
                        "domain_randomization": False,
                        "task_types": [1, 2, 3, 4, 5, 6],
                        "expert_timeout_steps": 150,
                        "expert_type": "handcoded",
                        "goal_desc_human_anns_prob": 0.0,
                        "hybrid": {
                            "start_eps": 100000,
                            "thor_prob": 0.5,
                            "eval_mode": "tw",
                        },
                    },
                    "general": {
                        "random_seed": 42,
                        "use_cuda": False,
                        "visdom": False,
                        "task": "alfred",
                        "training_method": "dagger",
                        "save_path": "./training/",
                        "observation_pool_capacity": 3,
                        "hide_init_receptacles": False,
                    },
                    "controller": {
                        "type": "oracle",
                        "debug": False,
                        "load_receps": False,
                    },
                    "logic": {
                        "domain": "$ALFWORLD_DATA/logic/alfred.pddl",
                        "grammar": "$ALFWORLD_DATA/logic/alfred.twl2",
                    },
                    "dagger": {
                        "training": {
                            "max_nb_steps_per_episode": spec.max_turns,
                        },
                        "fraction_assist": {
                            "fraction_assist_anneal_episodes": 0,
                            "fraction_assist_anneal_from": 1.0,
                            "fraction_assist_anneal_to": 0.01,
                        },
                        "fraction_random": {
                            "fraction_random_anneal_episodes": 0,
                            "fraction_random_anneal_from": 0.0,
                            "fraction_random_anneal_to": 0.0,
                        },
                        "replay": {
                            "replay_memory_capacity": 0,
                            "replay_memory_priority_fraction": 0.0,
                            "update_per_k_game_steps": 1,
                            "replay_batch_size": 1,
                            "multi_step": 1,
                            "replay_sample_history_length": 1,
                            "replay_sample_update_from": 1,
                        },
                    },
                },
            },
        },
        "train": {
            "algorithm": "max_alpha_grpo",
            "total_episodes": spec.n_episodes,
            "batch_size": spec.k,
            "learning_rate": spec.learning_rate,
            "checkpoint_every": 50,
            "eval_every": 25,
            "K_trajectories_per_task": spec.k,
            "kl_coeff": spec.kl_coeff,
            "clip_eps": spec.clip_eps,
            "grad_accum_steps": spec.grad_accum_steps,
            "max_tokens_per_microbatch": spec.max_tokens_per_microbatch,
            "gpu_mem_util": 0.20,
            "kl_warmup_episodes": spec.kl_warmup_episodes,
            "task_id_stride": task_id_stride,
        },
        "policy": {
            "backbone": "Qwen2.5-1.5B-Instruct",
            "lora": True,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_rank": 16,
        },
        "hgpo": {
            "alpha": spec.alpha,
            "lambda_consistency": 0.0,
            "decomposer": spec.turn_reward_method,
        },
        "max_turn_reward": {
            "method": spec.turn_reward_method,
            "method_hyperparameters": dict(method_cfg),
        },
        "sft": {
            "adapter": spec.sft_adapter,
        },
        "dataset_size_mode": spec.dataset_size_mode,
        "logging": {
            "print_every": 10,
        },
        "_notes": (
            "Generated by build_config_for_run. Proposal math: "
            "A_H = alpha * A_turn + (1 - alpha) * A_traj."
        ),
    }


def build_manual_run_spec(
    *,
    sft_adapter: str,
    alpha: float,
    turn_reward_method: str,
    learning_rate: float,
    kl_coeff: float,
    n_episodes: int,
    k: int = 4,
    max_turns: int = 30,
    clip_eps: float = 0.2,
    grad_accum_steps: int = 1,
    max_tokens_per_microbatch: int = 2048,
    kl_warmup_episodes: int = 5,
    dataset_size_mode: str = "full",
    eval_episodes: int = FULL_NUM_GAMES,
    run_name_suffix: str = "final",
    signed_attention_transformer_ckpt: str = DEFAULT_SIGNED_ATTENTION_TRANSFORMER_CKPT,
    task_id_stride: int = 0,
) -> GRPORunSpec:
    """Build one explicit GRPO run spec outside the sweep grid."""
    resolved_task_id_stride = (
        int(task_id_stride)
        if int(task_id_stride) > 0
        else (FULL_TASK_ID_STRIDE if dataset_size_mode == "full" else GRID_TASK_ID_STRIDE)
    )
    base = GRPORunSpec(
        run_name="",
        sft_adapter=sft_adapter,
        alpha=alpha,
        turn_reward_method=turn_reward_method,
        n_episodes=n_episodes,
        k=k,
        max_turns=max_turns,
        learning_rate=learning_rate,
        kl_coeff=kl_coeff,
        clip_eps=clip_eps,
        grad_accum_steps=grad_accum_steps,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
        kl_warmup_episodes=kl_warmup_episodes,
        dataset_size_mode=dataset_size_mode,
        eval_episodes=eval_episodes,
        signed_attention_transformer_ckpt=(
            signed_attention_transformer_ckpt if turn_reward_method == "signed_attention" else ""
        ),
        task_id_stride=resolved_task_id_stride,
    )
    run_name = f"{make_run_name(base)}_{run_name_suffix}"
    return GRPORunSpec(**{**base.__dict__, "run_name": run_name})


def write_config_for_run(
    spec: GRPORunSpec,
    output_dir: Path = CONFIG_OUTPUT_DIR,
) -> str:
    """Write one generated config JSON and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{spec.run_name}.json"
    with path.open("w") as f:
        json.dump(build_config_for_run(spec), f, indent=2)
        f.write("\n")
    return str(path)


def upload_config_for_run(config_path: str | Path, retries: int = 4) -> str:
    """Upload a generated config to the Modal volume and return the container path."""
    local_path = Path(config_path)
    remote_path = f"{VOLUME_CONFIG_ROOT}/{local_path.name}"
    last_error = ""
    for attempt in range(1, max(1, int(retries)) + 1):
        result = subprocess.run(
            [
                "modal",
                "volume",
                "put",
                "--force",
                VOLUME_NAME,
                str(local_path),
                remote_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "KMP_DUPLICATE_LIB_OK": "TRUE",
                "CS224R_SKIP_OPENAI_SECRET": "1",
            },
            check=False,
        )
        if result.returncode == 0:
            return f"/vol{remote_path}"
        last_error = result.stderr or result.stdout
        if attempt < max(1, int(retries)):
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"modal volume put failed for {local_path.name}: {last_error}")


def launch_grpo_run(spec: GRPORunSpec) -> dict[str, Any]:
    """Write one config and launch the matching Modal ALFWorld GRPO run."""
    config_path = upload_config_for_run(write_config_for_run(spec))
    save_adapter_out = spec.save_adapter_out or f"/vol/checkpoints/grpo/{spec.run_name}"
    return train_loop_alfworld.remote(
        n_episodes=spec.n_episodes,
        k=spec.k,
        max_turns=spec.max_turns,
        run_name=spec.run_name,
        sft_adapter=spec.sft_adapter,
        use_sft_as_ref=True,
        kl_warmup_episodes=spec.kl_warmup_episodes,
        gpu_mem_util=0.20,
        config=config_path,
        eval_episodes=spec.eval_episodes,
        train_task_id_stride=spec.task_id_stride,
        save_adapter_out=save_adapter_out,
    )


def run_grid_search(
    *,
    sft_adapter: str = DEFAULT_SFT_ADAPTER,
    max_specs: int | None = None,
    allowed_methods: list[str] | None = None,
    signed_attention_transformer_ckpt: str = DEFAULT_SIGNED_ATTENTION_TRANSFORMER_CKPT,
) -> list[dict[str, Any]]:
    """Launch the GRPO sweep and return Modal result dictionaries."""
    results: list[dict[str, Any]] = []
    for spec in iter_grid_run_specs(
        sft_adapter=sft_adapter,
        max_specs=max_specs,
        allowed_methods=allowed_methods,
        signed_attention_transformer_ckpt=signed_attention_transformer_ckpt,
    ):
        results.append(launch_grpo_run(spec))
    return results


def write_grid_configs(
    *,
    sft_adapter: str = DEFAULT_SFT_ADAPTER,
    max_specs: int | None = None,
    allowed_methods: list[str] | None = None,
    signed_attention_transformer_ckpt: str = DEFAULT_SIGNED_ATTENTION_TRANSFORMER_CKPT,
) -> list[str]:
    """Generate config files for the grid without launching training."""
    return [
        write_config_for_run(spec)
        for spec in iter_grid_run_specs(
            sft_adapter=sft_adapter,
            max_specs=max_specs,
            allowed_methods=allowed_methods,
            signed_attention_transformer_ckpt=signed_attention_transformer_ckpt,
        )
    ]


@app.local_entrypoint()
def main(
    action: str = "show_grid",
    sft_adapter: str = DEFAULT_SFT_ADAPTER,
    max_specs: int = 0,
    signed_attention_transformer_ckpt: str = DEFAULT_SIGNED_ATTENTION_TRANSFORMER_CKPT,
    include_methods: str = "",
    alpha: float = 0.0,
    turn_reward_method: str = "trajectory_only",
    learning_rate: float = 1e-6,
    kl_coeff: float = 0.02,
    n_episodes: int = 100,
    k: int = 4,
    max_turns: int = 30,
    clip_eps: float = 0.2,
    grad_accum_steps: int = 1,
    max_tokens_per_microbatch: int = 2048,
    kl_warmup_episodes: int = 5,
    dataset_size_mode: str = "full",
    eval_episodes: int = FULL_NUM_GAMES,
    run_name_suffix: str = "final",
    task_id_stride: int = 0,
) -> None:
    """Local entrypoint for inspecting configs or launching GRPO runs."""
    limit = max_specs if max_specs > 0 else None
    allowed_methods = [m.strip() for m in include_methods.split(",") if m.strip()] or None
    if action == "show_grid":
        print(
            json.dumps(
                {
                    "sweep_axes": GRPO_SWEEP_AXES,
                    "fixed_defaults": GRPO_FIXED_DEFAULTS,
                },
                indent=2,
                default=str,
            )
        )
        return
    if action == "show_best":
        print(json.dumps(BEST_GRPO_HYPERPARAMS_AFTER_SWEEP, indent=2, default=str))
        return
    if action == "show_sample_config":
        spec = iter_grid_run_specs(
            sft_adapter=sft_adapter,
            max_specs=1,
            allowed_methods=allowed_methods,
            signed_attention_transformer_ckpt=signed_attention_transformer_ckpt,
        )[0]
        print(json.dumps(build_config_for_run(spec), indent=2, default=str))
        return
    if action == "write_sample_config":
        spec = iter_grid_run_specs(
            sft_adapter=sft_adapter,
            max_specs=1,
            allowed_methods=allowed_methods,
            signed_attention_transformer_ckpt=signed_attention_transformer_ckpt,
        )[0]
        print(write_config_for_run(spec))
        return
    if action == "write_grid_configs":
        print(
            json.dumps(
                write_grid_configs(
                    sft_adapter=sft_adapter,
                    max_specs=limit,
                    allowed_methods=allowed_methods,
                    signed_attention_transformer_ckpt=signed_attention_transformer_ckpt,
                ),
                indent=2,
            )
        )
        return
    if action == "launch_sample":
        spec = iter_grid_run_specs(
            sft_adapter=sft_adapter,
            max_specs=1,
            allowed_methods=allowed_methods,
            signed_attention_transformer_ckpt=signed_attention_transformer_ckpt,
        )[0]
        print(json.dumps(launch_grpo_run(spec), indent=2, default=str))
        return
    if action == "launch_grid":
        print(
            json.dumps(
                run_grid_search(
                    sft_adapter=sft_adapter,
                    max_specs=limit,
                    allowed_methods=allowed_methods,
                    signed_attention_transformer_ckpt=signed_attention_transformer_ckpt,
                ),
                indent=2,
                default=str,
            )
        )
        return
    if action == "launch_manual":
        spec = build_manual_run_spec(
            sft_adapter=sft_adapter,
            alpha=alpha,
            turn_reward_method=turn_reward_method,
            learning_rate=learning_rate,
            kl_coeff=kl_coeff,
            n_episodes=n_episodes,
            k=k,
            max_turns=max_turns,
            clip_eps=clip_eps,
            grad_accum_steps=grad_accum_steps,
            max_tokens_per_microbatch=max_tokens_per_microbatch,
            kl_warmup_episodes=kl_warmup_episodes,
            dataset_size_mode=dataset_size_mode,
            eval_episodes=eval_episodes,
            run_name_suffix=run_name_suffix,
            signed_attention_transformer_ckpt=signed_attention_transformer_ckpt,
            task_id_stride=task_id_stride,
        )
        print(json.dumps(launch_grpo_run(spec), indent=2, default=str))
        return
    raise ValueError(
        "action must be one of: show_grid, show_best, "
        "show_sample_config, write_sample_config, write_grid_configs, "
        "launch_sample, launch_grid, launch_manual"
    )
