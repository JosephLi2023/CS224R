from __future__ import annotations

import argparse

from src.algorithms.baseline.policy import SoftmaxPolicy
from src.algorithms.hgpo.grouping import validate_groups
from src.algorithms.hgpo.policy import HGPOSoftmaxPolicy
from src.envs.factory import make_env
from src.trainers.evaluator import evaluate_policy
from src.trainers.io_utils import deep_merge, load_checkpoint, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_file = load_json(args.env_config)
    train_file = load_json(args.train_config)
    eval_file = load_json(args.eval_config)
    cfg = deep_merge(deep_merge(env_file, train_file), eval_file)

    seed = int(cfg["run"]["seed"])
    env_name = str(cfg["env"]["name"]).lower()
    n_actions = int(cfg["env"].get("n_actions", 16 if env_name == "webshop" else cfg["env"]["n_actions"]))
    algorithm = cfg["train"]["algorithm"]

    env = make_env(cfg["env"], seed=seed + 100)

    if algorithm == "baseline":
        policy = SoftmaxPolicy(n_actions=n_actions, seed=seed)
    elif algorithm == "hgpo":
        groups = {int(k): [int(x) for x in v] for k, v in cfg["hgpo"]["groups"].items()}
        validate_groups(groups=groups, n_actions=n_actions)
        policy = HGPOSoftmaxPolicy(
            n_actions=n_actions,
            seed=seed,
            groups=groups,
            alpha=float(cfg["hgpo"]["group_regularization_alpha"]),
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    ckpt = load_checkpoint(args.checkpoint)
    policy.logits = [float(x) for x in ckpt["logits"]]

    res = evaluate_policy(
        env=env,
        policy=policy,
        episodes=int(cfg["eval"]["episodes"]),
        env_name=env_name,
        greedy=bool(cfg["eval"]["greedy"]),
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"avg_return={res.avg_return:.6f}")


if __name__ == "__main__":
    main()
