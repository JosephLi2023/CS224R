"""K-trajectory rollout collector for H-GRPO.

Drives a vLLM-style runner through K parallel env instances for one task,
returning a `TrajectoryGroup` populated with action token ids and
rollout-time logprobs ready for the GRPO trainer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Callable, Protocol

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord


class _RunnerLike(Protocol):
    def generate_rich(self, prompts: list[str], sampling: Any) -> list[list[Any]]: ...


PromptRenderer = Callable[[Any, list[TurnRecord]], str]
ActionParser = Callable[[str], str]
EnvFactory = Callable[[], Any]


@dataclass
class RolloutCollectorConfig:
    max_turns: int = 12
    soft_prompt_token_budget: int = 3500


@dataclass
class CollectStats:
    K: int
    completed: int = 0
    truncated: int = 0
    total_turns: int = 0
    total_action_tokens: int = 0
    over_budget_count: int = 0
    final_rewards: list[float] = field(default_factory=list)


def _override_n(sampling: Any, n: int) -> Any:
    """Return a copy of `sampling` with `n=n`. Works for dataclass-style params."""
    try:
        return _dc_replace(sampling, n=n)
    except TypeError:
        sampling.n = n
        return sampling


def _safe_reset(env: Any, task_id: Any) -> Any:
    """Call env.reset(task_id=...) when supported, else env.reset()."""
    try:
        return env.reset(task_id=task_id)
    except TypeError:
        return env.reset()


class RolloutCollector:
    def __init__(
        self,
        runner: _RunnerLike,
        env_factory: EnvFactory,
        prompt_renderer: PromptRenderer,
        action_parser: ActionParser,
        cfg: RolloutCollectorConfig | None = None,
    ) -> None:
        self.runner = runner
        self.env_factory = env_factory
        self.render = prompt_renderer
        self.parse = action_parser
        self.cfg = cfg or RolloutCollectorConfig()

    def collect_group(
        self,
        task_id: str | int,
        env_name: str,
        K: int,
        sampling: Any,
    ) -> tuple[TrajectoryGroup, CollectStats]:
        """K parallel rollouts on the same task → populated TrajectoryGroup."""
        if getattr(sampling, "n", 1) != 1:
            sampling = _override_n(sampling, 1)

        envs = [self.env_factory() for _ in range(K)]
        states: list[Any] = [_safe_reset(env, task_id) for env in envs]

        traj_turns: list[list[TurnRecord]] = [[] for _ in range(K)]
        rewards_so_far: list[float] = [0.0 for _ in range(K)]
        dones: list[bool] = [False for _ in range(K)]

        stats = CollectStats(K=K)

        for turn_idx in range(self.cfg.max_turns):
            live_idx = [i for i in range(K) if not dones[i]]
            if not live_idx:
                break

            prompts = [self.render(states[i], traj_turns[i]) for i in live_idx]
            outs = self.runner.generate_rich(prompts, sampling)

            for j, i in enumerate(live_idx):
                gen = outs[j][0]
                action_text = self.parse(getattr(gen, "text", "") or "")

                next_state, reward, done, _info = envs[i].step(action_text)

                token_ids = tuple(getattr(gen, "token_ids", ()) or ())
                token_logprobs = tuple(getattr(gen, "token_logprobs", ()) or ())
                prompt_token_count = int(getattr(gen, "prompt_token_count", 0) or 0)
                prompt_token_ids = tuple(getattr(gen, "prompt_token_ids", ()) or ())

                obs_text = getattr(states[i], "observation_text", "") or ""
                turn = TurnRecord(
                    turn_idx=turn_idx,
                    observation_text=obs_text,
                    action_text=action_text,
                    raw_env_reward=float(reward),
                    action_token_ids=token_ids,
                    action_token_logprobs=token_logprobs,
                    prompt_token_count=prompt_token_count,
                    prompt_token_ids=prompt_token_ids,
                )
                traj_turns[i].append(turn)
                rewards_so_far[i] += float(reward)
                stats.total_turns += 1
                stats.total_action_tokens += len(token_ids)
                if prompt_token_count > self.cfg.soft_prompt_token_budget:
                    stats.over_budget_count += 1

                states[i] = next_state
                if done:
                    dones[i] = True

        for i in range(K):
            if dones[i]:
                stats.completed += 1
            else:
                stats.truncated += 1
        stats.final_rewards = list(rewards_so_far)

        trajectories = [
            Trajectory(
                task_id=str(task_id),
                env_name=env_name,
                turns=list(traj_turns[i]),
                final_reward=float(rewards_so_far[i]),
            )
            for i in range(K)
        ]
        group = TrajectoryGroup(
            task_id=str(task_id), env_name=env_name, trajectories=trajectories
        )
        return group, stats
