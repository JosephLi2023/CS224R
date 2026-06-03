from __future__ import annotations

from src.envs.alfworld_adapter import ALFWorldAdapter
from src.envs.webshop_adapter import WebShopAdapter


def make_env(env_cfg: dict, seed: int):
    name = env_cfg["name"].lower()

    if name == "webshop":
        return WebShopAdapter(
            max_steps=int(env_cfg.get("max_steps", 40)),
            observation_mode=str(env_cfg.get("observation_mode", "text")),
            task_split=str(env_cfg.get("task_split", "train")),
            env_kwargs=dict(env_cfg.get("env_kwargs", {})),
        )

    if name == "alfworld":
        # seed is unused by the adapter but accepted for parity with WebShop.
        env_kwargs = dict(env_cfg.get("env_kwargs", {}))
        if seed is not None and "seed" not in env_kwargs:
            env_kwargs.setdefault("seed", int(seed))
        return ALFWorldAdapter(
            max_steps=int(env_cfg.get("max_steps", 40)),
            observation_mode=str(env_cfg.get("observation_mode", "text")),
            task_split=str(env_cfg.get("task_split", "train")),
            env_kwargs=env_kwargs,
        )

    raise ValueError(f"Unsupported env name: {env_cfg['name']}")
