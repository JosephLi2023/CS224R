"""Modal app: generate ALFWorld SFT trajectories using the handcoded expert.

  modal run infra/app_alfworld_sft_gen.py                  # generate
  modal run infra/app_alfworld_sft_gen.py --action inspect # peek + cardinality

Drives `ALFWorldAdapter` over the (trimmed) train games using the
upstream handcoded PDDL expert. The expert exposes its full optimal
plan via `info["extra.expert_plan"]` after every reset/step. We pop one
action at a time (`expert_plan[0]`) so we always re-read the plan
after each `env.step()` — robust to upstream re-planning if a step
nondeterministically fails.

For every step we emit one JSONL row with:
  - prompt  — pre-rendered by `render_alfworld_turn_prompt` (the SAME
              renderer the runtime ReAct collector uses; this is the
              critical anti-template-drift guarantee — see plan
              `alfworld_sft_warm_start.plan.md` Risks section).
  - action  — the expert plan action string just executed.
  - step_idx, trajectory_id, instruction, final_reward — bookkeeping.

We KEEP only trajectories where the env reports `won=True` (or
equivalent: episode terminated with reward > 0). If the expert plan
ever exhausts before `won`, we DROP the partial trajectory entirely
rather than write half-grounded supervision.

Cost: ~$1 / ~15-30 min on a single CPU container (no GPU needed).

Schema reference: `src/datasets/sft_alfworld.py::SFTExample`. The
SFT trainer (`infra/app_sft_train_alfworld.py`) reads these rows via
`load_sft_examples_from_jsonl(...)` and tokenizes prompt+action with
the standard masked-CE loss, mirroring `infra/app_sft_train.py`.
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import ALFWORLD_DATA_DIR, alfworld_image

app = modal.App("cs224r-hgpo-alfworld-sft-gen")


# Where the gen app writes the SFT trajectories on the shared Volume.
# Path mirrors the WebShop pipeline's `/vol/data/webshop/human_trajs/`
# location but is a single JSONL (AlfWorld emits one row per step
# rather than one file per trajectory) so the loader is a one-shot read.
SFT_OUTPUT_PATH = "/vol/data/alfworld/sft_trajs.jsonl"


def _build_alfworld_config_dict() -> dict:
    """Build the upstream `AlfredTWEnv` config dict.

    Mirrors `configs/env_alfworld.json::env_kwargs.config` so the SFT
    gen env matches the runtime env exactly. Hard-coded here (rather
    than read from the JSON) so the gen app has no external file
    dependency at Modal-call time.
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

    AlfWorld's handcoded expert exposes its plan under several
    possible keys depending on upstream version / wrapper layer:
      - `extra.expert_plan` (TextWorld batched info convention)
      - `expert_plan`        (some non-batched wrappers)

    Returns the first non-empty list found, normalized to list[str].
    Returns [] when no plan is available (caller drops trajectory).
    """
    if not isinstance(info, dict):
        return []
    for key in ("extra.expert_plan", "expert_plan"):
        plan = info.get(key)
        if isinstance(plan, (list, tuple)) and plan:
            # Some adapters wrap once more in a per-batch list — peel it.
            first = plan[0]
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
                return [str(x) for x in first]
            if isinstance(first, str):
                return [str(x) for x in plan]
    return []


def _extract_won(info: dict) -> bool:
    """Check the upstream `won` flag (terminal success indicator).

    AlfWorld marks task completion via `info["won"]` (True/False).
    Returns False when the key is absent so callers err on the side of
    dropping trajectories rather than emitting partial demos.
    """
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

    Args:
        n_games: how many distinct games to attempt. Each iteration calls
                 `adapter.reset(task_id=i)` which (via `_select_task`)
                 maps to game-index `i % len(game_files)`. Cap at the
                 trimmed-train-set cardinality (default 200).
        output_path: where to write the JSONL on the shared Volume.
                     Truncated on entry — re-running this app produces a
                     fresh dataset.
        max_history_turns: forwarded to the runtime renderer to bound
                           prompt context. MUST match what the runtime
                           collector uses or the SFT prompt template
                           will drift from inference time. Default 3 is
                           the runtime default in
                           `render_alfworld_turn_prompt`.
        max_steps_per_episode: hard cap on adapter steps per episode.
                               If the expert hasn't won by then we drop
                               the trajectory.

    Returns:
        manifest dict with cardinalities (n_games_attempted,
        n_games_won, n_games_dropped, n_examples_written, etc.).
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

        # Trajectory id derived from the resolved game index AND the
        # game-file path basename (when available) so re-runs land on
        # consistent IDs even if the trimmed-dir cardinality changes.
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

        # Buffer rows for THIS trajectory; we only flush when `won=True`
        # so partial demos never reach disk.
        instruction_text = state.observation_text or ""
        history: list = []
        traj_rows: list[dict] = []
        won = False
        truncated = False

        for step_idx in range(max_steps_per_episode):
            if not plan:
                # Expert exhausted before winning → drop trajectory.
                truncated = True
                break

            # Render the prompt with the runtime renderer (same module
            # `infra/app_train_loop.py::_resolve_env_bindings` uses for
            # AlfWorld). This guarantees zero template drift between
            # SFT-time and rollout-time.
            prompt = render_alfworld_turn_prompt(
                state, history, max_history_turns=max_history_turns,
            )
            action = plan[0]

            # Record the (prompt, action) pair BEFORE stepping; we write
            # to disk only if the trajectory ultimately wins.
            traj_rows.append({
                "prompt": prompt,
                "action": action,
                "step_idx": step_idx,
                "trajectory_id": traj_id,
                "instruction": instruction_text,
                # final_reward is a placeholder; overwritten on flush.
                "final_reward": 0.0,
            })

            # Append a TurnRecord-shaped object to history so the next
            # render sees the prior (obs, action). Use a dataclass-lite
            # SimpleNamespace — `_format_history` uses getattr so any
            # object exposing observation_text + action_text works.
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
                # Re-read plan from the post-step info; if won, we'll
                # flush below. If done-but-not-won, we drop.
                break

            new_plan = _extract_expert_plan(info)
            if not new_plan:
                # Plan unexpectedly empty mid-trajectory → upstream
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
    """Peek at the SFT dataset: cardinality + first row's prompt+action.

    Useful smoke test after `generate_sft_trajectories` to confirm the
    file is non-trivial before kicking off the SFT trainer.
    """
    import json
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
    """Local entrypoint dispatching on `--action`.

    Actions:
      generate (default) — run `generate_sft_trajectories(...)`
      inspect            — pretty-print the dataset summary + first row
    """
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
