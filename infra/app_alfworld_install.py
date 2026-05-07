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
    }

    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # type: ignore[import-not-found]

    e = AlfredTWEnv(config, train_eval="train")
    out = e.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    obs_text = obs if isinstance(obs, str) else str(obs)[:300]
    return {
        "obs_preview": obs_text[:300],
        "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else None,
    }


@app.local_entrypoint()
def main(action: str = "download") -> None:
    import json as _json
    fn = {
        "download": download_alfworld_data,
        "trim_data_dir": trim_data_dir,
        "reset_smoke": reset_smoke,
    }.get(action)
    if fn is None:
        raise ValueError(
            f"Unknown action: {action!r}. Expected one of: download, "
            "trim_data_dir, reset_smoke."
        )
    print(_json.dumps(fn.remote(), indent=2, default=str))
