from __future__ import annotations

from src.envs.alfworld_adapter import ALFWorldAdapter
from src.envs.toy_bandit import ToyBanditConfig, ToyBanditEnv
from src.envs.webshop_adapter import WebShopAdapter


def make_env(env_cfg: dict, seed: int):
    name = env_cfg["name"].lower()

    if name == "toy_bandit":
        cfg = ToyBanditConfig(
            n_actions=int(env_cfg["n_actions"]),
            episode_length=int(env_cfg["episode_length"]),
            reward_noise_std=float(env_cfg["reward_noise_std"]),
            action_means=[float(x) for x in env_cfg["action_means"]],
        )
        return ToyBanditEnv(cfg=cfg, seed=seed)

    if name == "webshop":
        return WebShopAdapter(
            max_steps=int(env_cfg.get("max_steps", 40)),
            observation_mode=str(env_cfg.get("observation_mode", "text")),
            task_split=str(env_cfg.get("task_split", "train")),
            env_kwargs=dict(env_cfg.get("env_kwargs", {})),
        )

    if name == "alfworld":
        # `seed` is currently unused by the ALFWorldAdapter constructor
        # but we accept it via factory signature for parity with WebShop
        # / future deterministic seed propagation. The adapter forwards
        # `train_eval=task_split` into env_kwargs internally so configs
        # don't need to set it explicitly.
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
