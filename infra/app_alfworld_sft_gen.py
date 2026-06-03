"""Modal app: generate ALFWorld SFT trajectories using the handcoded expert.

Drives `ALFWorldAdapter` over the trimmed train games using the upstream PDDL
expert, emitting one JSONL row per step (prompt pre-rendered by the runtime
`render_alfworld_turn_prompt` to avoid template drift). Only `won` trajectories
are kept; partial ones are dropped. Schema: `src/datasets/sft_alfworld.py`.
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import ALFWORLD_DATA_DIR, alfworld_image

app = modal.App("cs224r-hgpo-alfworld-sft-gen")


# Single JSONL on the shared Volume (one row per step), unlike WebShop's
# directory-of-files layout.
SFT_OUTPUT_PATH = "/vol/data/alfworld/sft_trajs.jsonl"


def _build_alfworld_config_dict() -> dict:
    """Build the upstream `AlfredTWEnv` config dict.

    Mirrors `configs/env_alfworld.json::env_kwargs.config`, hard-coded here so
    the gen app has no external file dependency at Modal-call time.
    """
    return {
        "dataset": {
            "data_path": "$ALFWORLD_DATA/json_2.1.1/train",
            "eval_id_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_seen",
            "eval_ood_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_unseen",
            "num_train_games": -1,  # use everything available in trimmed dir
            "num_eval_games": -1,
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
        "controller": {"type": "oracle", "debug": False, "load_receps": True},
        "logic": {
            "domain": "$ALFWORLD_DATA/logic/alfred.pddl",
            "grammar": "$ALFWORLD_DATA/logic/alfred.twl2",
        },
        "dagger": {
            "training": {"max_nb_steps_per_episode": 50},
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
    }


def _extract_expert_plan(info: dict) -> list[str]:
    """Pull the next-action list from the env info dict.

    Checks `extra.expert_plan` then `expert_plan`, normalized to list[str];
    returns [] when no plan is available (caller drops the trajectory).
    """
    if not isinstance(info, dict):
        return []
    for key in ("extra.expert_plan", "expert_plan"):
        plan = info.get(key)
        if isinstance(plan, (list, tuple)) and plan:
            # Some adapters wrap once more in a per-batch list - peel it.
            first = plan[0]
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
                return [str(x) for x in first]
            if isinstance(first, str):
                return [str(x) for x in plan]
    return []


def _extract_won(info: dict) -> bool:
    """Check the upstream `won` flag; returns False when absent."""
    if not isinstance(info, dict):
        return False
    won = info.get("won")
    if isinstance(won, (list, tuple)) and won:
        won = won[0]
    return bool(won)


@app.function(
    image=alfworld_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 60,  # 1 h cap; expected runtime ~15-30 min on CPU
)
def generate_sft_trajectories(
    n_games: int = 200,
    output_path: str = SFT_OUTPUT_PATH,
    max_history_turns: int = 3,
    max_steps_per_episode: int = 50,
) -> dict:
    """Iterate game indices [0, n_games), drive the expert, write JSONL.

    Each game maps to `i % len(game_files)`; `max_history_turns` MUST match the
    runtime renderer or the SFT prompt drifts from inference time. Returns a
    manifest with per-status cardinalities.
    """
    import json
    import os
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, "/workspace")
    os.environ.setdefault("ALFWORLD_DATA", ALFWORLD_DATA_DIR)

    from src.envs.alfworld_adapter import ALFWorldAdapter
    from src.envs.prompts.react_alfworld import render_alfworld_turn_prompt

    config = _build_alfworld_config_dict()
    print(f">>> Building ALFWorldAdapter with {n_games} train games target")
    adapter = ALFWorldAdapter(
        max_steps=max_steps_per_episode,
        observation_mode="text",
        task_split="train",
        env_kwargs={"config": config},
    )

    # Make sure the parent dir exists; truncate output (full re-gen).
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(output_path, "w")

    # Surface the actual game-files cardinality so we don't iterate past it.
    games = adapter._game_files()
    n_available = len(games)
    n_to_try = min(n_games, n_available) if n_available > 0 else n_games
    print(f">>> game_files cardinality: {n_available}; attempting {n_to_try}")

    n_won = 0
    n_dropped_no_plan = 0
    n_dropped_lost = 0
    n_dropped_truncated = 0
    n_examples = 0
    t0 = time.time()

    for game_idx in range(n_to_try):
        try:
            state = adapter.reset(task_id=game_idx)
        except Exception as exc:  # pragma: no cover  (depends on alfworld)
            print(f"  [game {game_idx}] reset() raised {type(exc).__name__}: {exc}; skipping")
            n_dropped_no_plan += 1
            continue

        # Trajectory id from the resolved game index + file basename so re-runs
        # land on consistent IDs even if the trimmed-dir cardinality changes.
        last_idx = adapter._last_task_idx if adapter._last_task_idx is not None else game_idx
        if 0 <= last_idx < n_available:
            traj_id = f"{last_idx:04d}_{Path(str(games[last_idx])).name}"
        else:
            traj_id = f"game_{game_idx:04d}"

        info0 = state.raw_info if isinstance(state.raw_info, dict) else {}
        plan = _extract_expert_plan(info0)
        if not plan:
            print(f"  [game {game_idx}] no expert_plan in reset info; dropping")
            n_dropped_no_plan += 1
            continue

        # Buffer rows; only flush when won=True so partial demos never hit disk.
        instruction_text = state.observation_text or ""
        history: list = []
        traj_rows: list[dict] = []
        won = False
        truncated = False

        for step_idx in range(max_steps_per_episode):
            if not plan:
                # Expert exhausted before winning -> drop trajectory.
                truncated = True
                break

            # Render with the runtime renderer to guarantee zero template drift.
            prompt = render_alfworld_turn_prompt(
                state, history, max_history_turns=max_history_turns,
            )
            action = plan[0]

            # Record the (prompt, action) pair BEFORE stepping; flushed only on win.
            traj_rows.append({
                "prompt": prompt,
                "action": action,
                "step_idx": step_idx,
                "trajectory_id": traj_id,
                "instruction": instruction_text,
                # final_reward is a placeholder; overwritten on flush.
                "final_reward": 0.0,
            })

            # Append to history so the next render sees the prior (obs, action);
            # SimpleNamespace suffices since `_format_history` uses getattr.
            from types import SimpleNamespace
            history.append(SimpleNamespace(
                observation_text=state.observation_text,
                action_text=action,
            ))

            try:
                state, reward, done, info = adapter.step(action)
            except Exception as exc:  # pragma: no cover
                print(f"  [game {game_idx} step {step_idx}] step('{action}') raised "
                      f"{type(exc).__name__}: {exc}; dropping")
                truncated = True
                break

            won = _extract_won(info)
            if won or done:
                # Re-read plan from post-step info; flush below if won, else drop.
                break

            new_plan = _extract_expert_plan(info)
            if not new_plan:
                # Plan unexpectedly empty mid-trajectory -> upstream
                # re-planning failed. Drop.
                truncated = True
                break
            plan = new_plan

        if won and traj_rows:
            n_won += 1
            for row in traj_rows:
                row["final_reward"] = 1.0
                fh.write(json.dumps(row) + "\n")
                n_examples += 1
            fh.flush()
        elif truncated:
            n_dropped_truncated += 1
        else:
            n_dropped_lost += 1

        if (game_idx + 1) % 10 == 0:
            elapsed = round(time.time() - t0, 1)
            print(f"  [{game_idx + 1}/{n_to_try}] won={n_won} "
                  f"truncated={n_dropped_truncated} lost={n_dropped_lost} "
                  f"no_plan={n_dropped_no_plan} examples={n_examples} t={elapsed}s")

    fh.close()
    volume.commit()

    manifest = {
        "output_path": output_path,
        "n_games_attempted": n_to_try,
        "n_games_available": n_available,
        "n_games_won": n_won,
        "n_dropped_truncated": n_dropped_truncated,
        "n_dropped_lost": n_dropped_lost,
        "n_dropped_no_plan": n_dropped_no_plan,
        "n_examples_written": n_examples,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(">>> SFT-gen done:", manifest)
    return manifest


@app.function(
    image=alfworld_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=10 * 60,
)
def inspect_sft_dataset(path: str = SFT_OUTPUT_PATH) -> dict:
    """Peek at the SFT dataset: cardinality + first row's prompt+action."""
    import os
    import sys

    sys.path.insert(0, "/workspace")
    from src.datasets.sft_alfworld import (
        load_sft_examples_from_jsonl,
        summarize_sft_dataset,
    )

    if not os.path.isfile(path):
        return {"path": path, "exists": False}
    examples = load_sft_examples_from_jsonl(path)
    summary = summarize_sft_dataset(examples)
    first = examples[0] if examples else None
    return {
        "path": path,
        "exists": True,
        "summary": summary,
        "first_row": None if first is None else {
            "prompt_head": first.prompt[:600],
            "prompt_tail": first.prompt[-200:],
            "action": first.action,
            "trajectory_id": first.trajectory_id,
            "step_idx": first.step_idx,
            "instruction_head": first.instruction[:200],
            "final_reward": first.final_reward,
        },
    }


@app.local_entrypoint()
def main(
    action: str = "generate",
    n_games: int = 200,
    output_path: str = SFT_OUTPUT_PATH,
    max_history_turns: int = 3,
    max_steps_per_episode: int = 50,
) -> None:
    """Local entrypoint; dispatch on --action (generate | inspect)."""
    import json as _json
    if action == "generate":
        res = generate_sft_trajectories.remote(
            n_games=n_games,
            output_path=output_path,
            max_history_turns=max_history_turns,
            max_steps_per_episode=max_steps_per_episode,
        )
    elif action == "inspect":
        res = inspect_sft_dataset.remote(path=output_path)
    else:
        raise ValueError(
            f"Unknown action {action!r}. Expected one of: generate, inspect."
        )
    print(_json.dumps(res, indent=2, default=str))
