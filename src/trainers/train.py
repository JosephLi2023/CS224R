from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from src.algorithms.baseline.policy import SoftmaxPolicy
from src.algorithms.hgpo.grouping import validate_groups
from src.algorithms.hgpo.policy import HGPOSoftmaxPolicy
from src.envs.factory import make_env
from src.trainers.evaluator import evaluate_policy
from src.trainers.io_utils import (
    deep_merge,
    dump_json,
    ensure_dir,
    load_checkpoint,
    load_json,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline or HGPO")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--resume-checkpoint", default="")
    return parser.parse_args()


def collect_batch_toy(env, policy, batch_size: int, n_actions: int) -> tuple[list[float], list[float], float]:
    action_counts = [0.0 for _ in range(n_actions)]
    action_return_sum = [0.0 for _ in range(n_actions)]
    batch_episode_returns: list[float] = []

    for _ in range(batch_size):
        env.reset()
        done = False
        total = 0.0
        actions: list[int] = []
        rewards: list[float] = []
        while not done:
            a = policy.sample_action()
            r, done = env.step(a)
            actions.append(a)
            rewards.append(r)
            total += r

        for a, r in zip(actions, rewards):
            action_counts[a] += 1.0
            action_return_sum[a] += r
        batch_episode_returns.append(total)

    action_returns = [0.0 for _ in range(n_actions)]
    for i in range(n_actions):
        if action_counts[i] > 0:
            action_returns[i] = action_return_sum[i] / action_counts[i]

    avg_batch_return = sum(batch_episode_returns) / max(1, len(batch_episode_returns))
    return action_counts, action_returns, avg_batch_return


def collect_batch_text(env, policy, batch_size: int, n_actions: int, fallback: str) -> tuple[list[float], list[float], float]:
    action_counts = [0.0 for _ in range(n_actions)]
    action_return_sum = [0.0 for _ in range(n_actions)]
    batch_episode_returns: list[float] = []

    for _ in range(batch_size):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            action_idx, action_cmd = policy.sample_text_action(state=state, fallback=fallback)
            state, reward, done, _ = env.step(action_cmd)
            action_counts[action_idx] += 1.0
            action_return_sum[action_idx] += reward
            total += reward
        batch_episode_returns.append(total)

    action_returns = [0.0 for _ in range(n_actions)]
    for i in range(n_actions):
        if action_counts[i] > 0:
            action_returns[i] = action_return_sum[i] / action_counts[i]

    avg_batch_return = sum(batch_episode_returns) / max(1, len(batch_episode_returns))
    return action_counts, action_returns, avg_batch_return


def main() -> None:
    args = parse_args()

    env_file = load_json(args.env_config)
    train_file = load_json(args.train_config)
    eval_file = load_json(args.eval_config)

    cfg = deep_merge(env_file, train_file)
    cfg = deep_merge(cfg, eval_file)

    run_name = cfg["run"]["name"]
    output_dir = cfg["run"]["output_dir"]
    seed = int(cfg["run"]["seed"])
    algorithm = cfg["train"]["algorithm"]
    env_name = str(cfg["env"]["name"]).lower()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{run_name}_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(str(ckpt_dir))

    dump_json(str(run_dir / "config_snapshot.json"), cfg)

    env = make_env(cfg["env"], seed=seed)

    text_envs = {"webshop", "alfworld"}
    n_actions = int(cfg["env"].get("n_actions", 16 if env_name in text_envs else cfg["env"]["n_actions"]))

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

    start_episode = 1
    if args.resume_checkpoint:
        ckpt = load_checkpoint(args.resume_checkpoint)
        policy.logits = [float(x) for x in ckpt["logits"]]
        start_episode = int(ckpt["episode"]) + 1

    total_episodes = int(cfg["train"]["total_episodes"])
    batch_size = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["learning_rate"])
    checkpoint_every = int(cfg["train"]["checkpoint_every"])
    eval_every = int(cfg["train"]["eval_every"])
    print_every = int(cfg["logging"]["print_every"])

    train_log = []
    eval_log = []

    for episode in range(start_episode, total_episodes + 1):
        if env_name in text_envs:
            fallback = "search[noop]" if env_name == "webshop" else "look"
            action_counts, action_returns, batch_return = collect_batch_text(
                env=env,
                policy=policy,
                batch_size=batch_size,
                n_actions=n_actions,
                fallback=fallback,
            )
        else:
            action_counts, action_returns, batch_return = collect_batch_toy(
                env=env,
                policy=policy,
                batch_size=batch_size,
                n_actions=n_actions,
            )
        policy.update(action_counts=action_counts, action_returns=action_returns, lr=lr)

        train_log.append({"episode": episode, "batch_return": batch_return})

        if episode % eval_every == 0 or episode == total_episodes:
            eval_env = make_env(cfg["env"], seed=seed + 1)
            er = evaluate_policy(
                env=eval_env,
                policy=policy,
                episodes=int(cfg["eval"]["episodes"]),
                env_name=env_name,
                greedy=bool(cfg["eval"]["greedy"]),
            )
            eval_log.append({"episode": episode, "avg_return": er.avg_return})

        if episode % checkpoint_every == 0 or episode == total_episodes:
            ckpt_path = ckpt_dir / f"episode_{episode}.json"
            save_checkpoint(
                path=str(ckpt_path),
                logits=policy.logits,
                episode=episode,
                seed=seed,
                algorithm=algorithm,
            )

        if episode % print_every == 0 or episode == total_episodes:
            latest_eval = eval_log[-1]["avg_return"] if eval_log else float("nan")
            print(
                f"episode={episode} algorithm={algorithm} env={env_name} "
                f"batch_return={batch_return:.4f} eval_avg_return={latest_eval:.4f}"
            )

    dump_json(str(run_dir / "train_log.json"), {"rows": train_log})
    dump_json(str(run_dir / "eval_log.json"), {"rows": eval_log})
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
