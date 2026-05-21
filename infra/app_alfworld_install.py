"""Modal app: download AlfWorld data into the shared Volume.

Mirrors `infra/app_webshop_install.py` but for AlfWorld. AlfWorld provides a
CLI (`alfworld-download`) that downloads the TextWorld game files +
optional THOR/embodied scene assets to `$ALFWORLD_DATA`. We set
`ALFWORLD_DATA=/vol/data/alfworld/` in the image so the download lands on
the persistent Modal Volume, then `volume.commit()` so subsequent training
containers see the populated data tree.

One-shot. Cost: ~$1, ~5 min.

  modal run infra/app_alfworld_install.py --action download

Verify with:
  modal volume ls cs224r-hgpo-vol /data/alfworld/json_2.1.1/train

Should return >100 game directories.
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import ALFWORLD_DATA_DIR, alfworld_image

app = modal.App("cs224r-hgpo-alfworld-install")


@app.function(image=alfworld_image, volumes={VOLUME_MOUNT: volume}, timeout=30 * 60)
def download_alfworld_data() -> dict:
    """Run `alfworld-download --extra` against `$ALFWORLD_DATA` and commit
    the result to the shared Volume.

    `--extra` brings the THOR scene assets in addition to the base text-only
    game files. Even though we only run the text branch (`AlfredTWEnv`), the
    extra assets ship the PDDL logic + grammar files referenced by
    `configs/env_alfworld.json::env_kwargs.config.logic.{domain,grammar}`,
    so they're not optional in practice.
    """
    import os
    import subprocess

    os.makedirs(ALFWORLD_DATA_DIR, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("ALFWORLD_DATA", ALFWORLD_DATA_DIR)
    cmd = ["alfworld-download", "--extra", "--force"]
    print(">>>", " ".join(cmd), "(ALFWORLD_DATA=" + env["ALFWORLD_DATA"] + ")")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("--- STDOUT ---")
    print(proc.stdout[-4000:])
    if proc.stderr:
        print("--- STDERR ---")
        print(proc.stderr[-4000:])
    if proc.returncode != 0:
        raise RuntimeError(
            f"alfworld-download failed: rc={proc.returncode}. "
            "See stderr above. Common cause: network failure mid-download "
            "(retry — the CLI is idempotent with --force)."
        )

    # Sanity check: the train split should have at least 100 game dirs.
    train_dir = os.path.join(ALFWORLD_DATA_DIR, "json_2.1.1", "train")
    n_games = 0
    if os.path.isdir(train_dir):
        n_games = sum(
            1 for entry in os.scandir(train_dir) if entry.is_dir()
        )
    volume.commit()
    return {
        "alfworld_data_dir": ALFWORLD_DATA_DIR,
        "train_split_dir": train_dir,
        "n_train_game_dirs": n_games,
        "stdout_tail": proc.stdout[-500:],
    }


@app.function(image=alfworld_image, volumes={VOLUME_MOUNT: volume}, timeout=20 * 60)
def trim_data_dir(
    keep_train: int = 200,
    keep_eval_seen: int = 50,
    keep_eval_unseen: int = 50,
) -> dict:
    """Physically remove all but the first N game directories from each
    split under `$ALFWORLD_DATA/json_2.1.1/`.

    Why this exists: AlfWorld's `AlfredTWEnv.__init__` walks the entire
    data tree on EVERY adapter construction (~8810 PDDL/JSON entries
    iterated at ~2-3 it/s = ~30-60 min per env). With K=4 parallel
    rollouts in H-GRPO, that's hours of pure setup before any episode
    runs. The `num_train_games` config knob does NOT short-circuit the
    walk — it only truncates the final list. The walk is bounded by
    the physical size of the data dir.

    Solution: trim the data dir once. Subsequent env constructions only
    iterate the kept set (~300 entries × 0.5 s/entry = ~3 min/env vs
    ~60 min/env at full size).

    The deletion is destructive but easily reversed via
    `download_alfworld_data` (re-runs `alfworld-download --extra
    --force`).

    Returns a manifest of what was kept.
    """
    import os
    import shutil

    base = os.path.join(ALFWORLD_DATA_DIR, "json_2.1.1")
    splits = {
        "train": keep_train,
        "valid_seen": keep_eval_seen,
        "valid_unseen": keep_eval_unseen,
    }
    manifest: dict = {"trimmed_data_dir": base, "kept": {}, "removed": {}}

    for split_name, keep_n in splits.items():
        split_dir = os.path.join(base, split_name)
        if not os.path.isdir(split_dir):
            print(f">>> {split_name}: dir missing at {split_dir}; skipping")
            manifest["kept"][split_name] = 0
            manifest["removed"][split_name] = 0
            continue
        # Walk one level deep — task-type dirs (e.g. pick_and_place_simple-...).
        task_dirs = sorted(
            entry.path for entry in os.scandir(split_dir) if entry.is_dir()
        )
        # Each task_dir contains many trial subdirs. Flatten + sort so we
        # keep deterministic which trials survive.
        all_trials: list[str] = []
        for tdir in task_dirs:
            for trial in sorted(
                entry.path for entry in os.scandir(tdir) if entry.is_dir()
            ):
                all_trials.append(trial)
        kept = set(all_trials[:keep_n])
        removed_count = 0
        for trial in all_trials:
            if trial in kept:
                continue
            shutil.rmtree(trial)
            removed_count += 1
        # Also drop any task_dir that's now empty.
        for tdir in task_dirs:
            try:
                if not any(os.scandir(tdir)):
                    os.rmdir(tdir)
            except FileNotFoundError:
                pass
        manifest["kept"][split_name] = len(kept)
        manifest["removed"][split_name] = removed_count
        print(f">>> {split_name}: kept {len(kept)}, removed {removed_count}")

    volume.commit()
    return manifest


@app.function(image=alfworld_image, volumes={VOLUME_MOUNT: volume}, timeout=30 * 60)
def prepare_subset_data_dir(
    keep_train: int = 200,
    keep_eval_seen: int = 50,
    keep_eval_unseen: int = 50,
    destination_root: str = "/vol/data/alfworld_grid_subset",
) -> dict:
    """Build a separate physically trimmed ALFWorld subset tree.

    Unlike `trim_data_dir`, this is non-destructive: it copies only the kept
    trials into `destination_root/json_2.1.1/*` and leaves the full dataset in
    place. This is the correct source tree for fast grid searches because
    upstream ALFWorld walks the whole directory before applying `num_*_games`.
    """
    import os
    import shutil

    source_base = os.path.join(ALFWORLD_DATA_DIR, "json_2.1.1")
    dest_base = os.path.join(destination_root, "json_2.1.1")
    os.makedirs(dest_base, exist_ok=True)

    splits = {
        "train": keep_train,
        "valid_seen": keep_eval_seen,
        "valid_unseen": keep_eval_unseen,
    }
    manifest: dict = {
        "source_data_dir": source_base,
        "subset_data_dir": dest_base,
        "kept": {},
    }

    for split_name, keep_n in splits.items():
        split_src = os.path.join(source_base, split_name)
        split_dst = os.path.join(dest_base, split_name)
        if os.path.isdir(split_dst):
            shutil.rmtree(split_dst)
        os.makedirs(split_dst, exist_ok=True)

        task_dirs = sorted(
            entry.path for entry in os.scandir(split_src) if entry.is_dir()
        )
        all_trials: list[str] = []
        for tdir in task_dirs:
            for trial in sorted(
                entry.path for entry in os.scandir(tdir) if entry.is_dir()
            ):
                all_trials.append(trial)

        kept_trials = all_trials[:keep_n]
        copied = 0
        for trial in kept_trials:
            rel = os.path.relpath(trial, split_src)
            trial_dst = os.path.join(split_dst, rel)
            os.makedirs(os.path.dirname(trial_dst), exist_ok=True)
            shutil.copytree(trial, trial_dst)
            copied += 1

        manifest["kept"][split_name] = copied
        print(f">>> subset {split_name}: copied {copied} trials to {split_dst}")

    volume.commit()
    return manifest


@app.function(image=alfworld_image, volumes={VOLUME_MOUNT: volume}, timeout=10 * 60)
def reset_smoke() -> dict:
    """Smoke: instantiate AlfredTWEnv (train split) and call reset().

    Confirms the install is usable end-to-end before we point the training
    loop at it. Reads the same config dict shape that
    `configs/env_alfworld.json::env_kwargs.config` carries.
    """
    import os

    os.environ.setdefault("ALFWORLD_DATA", ALFWORLD_DATA_DIR)

    # Minimal config dict — mirrors the upstream example YAML enough to
    # boot a single text-env.
    config = {
        "dataset": {
            "data_path": "$ALFWORLD_DATA/json_2.1.1/train",
            "eval_id_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_seen",
            "eval_ood_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_unseen",
            "num_train_games": -1,
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

    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # type: ignore[import-not-found]

    e = AlfredTWEnv(config, train_eval="train")
    try:
        wrapped = e.init_env(batch_size=1)
    except Exception as exc:
        import traceback
        return {
            "stage": "init_env",
            "exception_type": type(exc).__name__,
            "exception_repr": repr(exc),
            "traceback": traceback.format_exc(),
        }
    try:
        out = wrapped.reset()
    except Exception as exc:
        import traceback
        return {
            "stage": "reset",
            "exception_type": type(exc).__name__,
            "exception_repr": repr(exc),
            "traceback": traceback.format_exc(),
        }
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    obs_text = obs if isinstance(obs, str) else str(obs)[:300]
    return {
        "obs_preview": obs_text[:300],
        "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else None,
    }


@app.function(image=alfworld_image, volumes={VOLUME_MOUNT: volume}, timeout=15 * 60)
def multi_adapter_smoke(n_envs: int = 4) -> dict:
    """Smoke: build multiple ALFWorldAdapter instances in one process.

    This mirrors the GRPO collector's K-env acquisition path closely enough to
    catch per-process ALFWorld/TextWorld registration failures that a single
    reset smoke will miss.
    """
    import os
    import traceback

    os.environ.setdefault("ALFWORLD_DATA", ALFWORLD_DATA_DIR)

    from src.envs.alfworld_adapter import ALFWorldAdapter

    config = {
        "dataset": {
            "data_path": "/vol/data/alfworld_grid_subset/json_2.1.1/train",
            "eval_id_data_path": "/vol/data/alfworld_grid_subset/json_2.1.1/valid_seen",
            "eval_ood_data_path": "/vol/data/alfworld_grid_subset/json_2.1.1/valid_unseen",
            "num_train_games": -1,
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
        "controller": {"type": "oracle", "debug": False, "load_receps": False},
        "logic": {
            "domain": "$ALFWORLD_DATA/logic/alfred.pddl",
            "grammar": "$ALFWORLD_DATA/logic/alfred.twl2",
        },
        "dagger": {
            "training": {"max_nb_steps_per_episode": 32},
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

    built: list[dict] = []
    for idx in range(n_envs):
        try:
            adapter = ALFWorldAdapter(
                max_steps=32,
                observation_mode="text",
                task_split="train",
                env_kwargs={"config": config},
                use_textworld_intermediate_reward=False,
                use_facts_diff_intermediate_reward=True,
            )
            state = adapter.reset(task_id=idx)
            built.append(
                {
                    "idx": idx,
                    "ok": True,
                    "obs_preview": state.observation_text[:120],
                    "n_valid_actions": len(state.valid_actions),
                }
            )
        except Exception as exc:
            built.append(
                {
                    "idx": idx,
                    "ok": False,
                    "exception_type": type(exc).__name__,
                    "exception_repr": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            break

    return {"n_envs_requested": n_envs, "results": built}


@app.local_entrypoint()
def main(action: str = "download") -> None:
    import json as _json
    fn = {
        "download": download_alfworld_data,
        "trim_data_dir": trim_data_dir,
        "prepare_subset_data_dir": prepare_subset_data_dir,
        "reset_smoke": reset_smoke,
        "multi_adapter_smoke": multi_adapter_smoke,
    }.get(action)
    if fn is None:
        raise ValueError(
            f"Unknown action: {action!r}. Expected one of: download, "
            "trim_data_dir, prepare_subset_data_dir, reset_smoke, "
            "multi_adapter_smoke."
        )
    print(_json.dumps(fn.remote(), indent=2, default=str))
