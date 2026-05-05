"""K-trajectory rollout collector for H-GRPO.

Drives a vLLM-style runner through K parallel env instances for one task,
returning a `TrajectoryGroup` populated with action token ids and
rollout-time logprobs ready for the GRPO trainer.

Day 14: optional opt-in **TurnRD replay-buffer producer**. When the
collector is constructed with `turnrd_emit_path` + `turnrd_embedder`
(and, for Mode 2, `judge_decomposer`), each `collect_group` call
appends one JSONL row per non-empty trajectory using
`src.turnrd.dataset.TurnRDRecord` for schema validation. All Day-14
producer params default to `None`, so the existing flag-driven
`infra/app_train_loop.py` path is byte-for-byte unchanged.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord

if TYPE_CHECKING:
    import torch  # noqa: F401  (only for the TurnEmbedder protocol below)

logger = logging.getLogger(__name__)


class _RunnerLike(Protocol):
    def generate_rich(self, prompts: list[str], sampling: Any) -> list[list[Any]]: ...


# Duck-typed callables (avoid eager imports of src.turnrd / src.algorithms.hgpo
# so this module remains torch-free for pure-Python tests).
PromptRenderer = Callable[[Any, list[TurnRecord]], str]
ActionParser = Callable[[str], str]
EnvFactory = Callable[[], Any]
# `TurnEmbedder` is `Callable[[Trajectory], torch.Tensor]` per
# `src/algorithms/hgpo/decomposers/turnrd.py::TurnEmbedder`. We don't import
# the type alias here because it requires torch at module-load time.
TurnEmbedder = Callable[[Trajectory], Any]
# `JudgeDecomposer`-shaped object: anything with a `.decompose(group) ->
# list[K] of list[T_i]` method. Matches the structural Protocol at
# `src/algorithms/hgpo/decomposers/base.py::TurnRewardDecomposer`.
TurnRewardDecomposerLike = Any


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
        reuse_envs: bool = True,
        *,
        # Day 14: optional TurnRD replay-buffer producer hook.
        # All None ⇒ producer disabled, default behavior unchanged.
        turnrd_emit_path: str | os.PathLike[str] | None = None,
        turnrd_embedder: TurnEmbedder | None = None,
        # Mode-2 only — supplies per-trajectory normalized labels via a
        # JudgeDecomposer call. None ⇒ Mode 1 (no judge_labels emitted;
        # records have judge_labels=None which the dataset reader keeps
        # for Mode 1 and drops for Mode 2).
        judge_decomposer: TurnRewardDecomposerLike | None = None,
    ) -> None:
        """K-trajectory rollout collector.

        Args:
            reuse_envs: when True (default), env instances built on the first
                `collect_group` call are cached and reused across episodes
                via `env.reset()`. Skips the ~5–8 s of WebShop product/goal
                loading per episode (review M2). Set False for unit tests
                that require fresh env state on every call.
            turnrd_emit_path: optional path to a JSONL file. When set,
                `collect_group` appends one row per non-empty trajectory
                using the schema defined by
                `src.turnrd.dataset.TurnRDRecord`. The parent directory is
                created lazily on first emit. Append + flush + fsync per
                row so a crash never leaves a half-written line.
            turnrd_embedder: required when `turnrd_emit_path` is set. A
                `Callable[[Trajectory], torch.Tensor]` returning per-turn
                embeddings of shape `[T_i, D]` (CPU fp32 recommended; the
                producer calls `.tolist()` to serialize as JSON).
            judge_decomposer: optional. When set, each `collect_group` call
                runs `judge_decomposer.decompose(group)` ONCE per group
                and writes the returned `list[T_i]` of normalized labels
                into each row's `judge_labels` field — enabling Mode-2
                replay buffers. With a warm Method-A judge cache this is
                a pure read (no fresh judge calls).
        """
        self.runner = runner
        self.env_factory = env_factory
        self.render = prompt_renderer
        self.parse = action_parser
        self.cfg = cfg or RolloutCollectorConfig()
        self.reuse_envs = reuse_envs
        self._env_pool: list[Any] = []
        # Day 14: producer plumbing (validated lazily so the import-time
        # surface remains torch-free).
        self._turnrd_emit_path: Optional[Path] = (
            Path(turnrd_emit_path) if turnrd_emit_path is not None else None
        )
        if self._turnrd_emit_path is not None:
            if turnrd_embedder is None:
                raise ValueError(
                    "RolloutCollector: turnrd_emit_path was provided but "
                    "turnrd_embedder is None. The producer needs an embedder "
                    "to convert each trajectory's turns into [T_i, D] vectors."
                )
            self._turnrd_emit_path.parent.mkdir(parents=True, exist_ok=True)
        self._turnrd_embedder: TurnEmbedder | None = turnrd_embedder
        self._judge_decomposer: TurnRewardDecomposerLike | None = judge_decomposer

    def _acquire_envs(self, K: int) -> list[Any]:
        """Return K env instances, growing the pool lazily when reuse is on."""
        if not self.reuse_envs:
            return [self.env_factory() for _ in range(K)]
        while len(self._env_pool) < K:
            self._env_pool.append(self.env_factory())
        return self._env_pool[:K]

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

        envs = self._acquire_envs(K)
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
                # Defensive: vLLM can return an EMPTY completion list for a
                # prompt when greedy sampling (T=0) immediately predicts
                # EOS as the top-1 token — `outs[j] = []`, and the next
                # line would raise IndexError. Treat as a no-op turn:
                # empty action text + no tokens. The env will step on
                # "" and likely return done=True with reward 0, which is
                # the correct eval behavior (the policy failed to
                # produce an action). Same defensive guard for the very
                # rare case where len(outs) < len(live_idx).
                gen_list = outs[j] if j < len(outs) else []
                if not gen_list:
                    class _EmptyGen:  # noqa: D401  (defensive empty stub)
                        text = ""
                        token_ids: tuple = ()
                        token_logprobs: tuple = ()
                        prompt_token_count = 0
                        prompt_token_ids: tuple = ()
                        finish_reason = "empty"

                    gen = _EmptyGen
                    stats.empty_outputs = getattr(stats, "empty_outputs", 0) + 1
                else:
                    gen = gen_list[0]
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

        # Day 14: emit TurnRD replay-buffer rows (if the producer was
        # configured at construction time). Skipped entirely otherwise.
        if self._turnrd_emit_path is not None:
            self._emit_turnrd_records(group)

        return group, stats

    # -------------------------------------------------------------------
    # Day 14: TurnRD replay-buffer producer
    # -------------------------------------------------------------------

    def _emit_turnrd_records(self, group: TrajectoryGroup) -> None:
        """Append one JSONL row per non-empty trajectory in `group`.

        Schema matches `src.turnrd.dataset.TurnRDRecord`:
          {"task_id", "turn_embeds", "final_reward", "judge_labels"}.

        Mode 2 (when `judge_decomposer` is set) calls
        `judge_decomposer.decompose(group)` once per group; the returned
        `list[K] of list[T_i]` is sliced into per-row `judge_labels`.
        Mode 1 (no judge decomposer) writes `judge_labels: None` and the
        dataset reader keeps the row.

        Empty trajectories are skipped (matches the dataset reader's
        drop-and-warn semantics — keeps producer + reader aligned).

        Atomicity: append-mode write + flush + fsync per row so a crash
        between trajectories never leaves a half-written line.
        """
        # Local import keeps the collector module torch-free at import time.
        from src.turnrd.dataset import TurnRDRecord  # type: ignore[import-not-found]

        assert self._turnrd_emit_path is not None  # narrowed
        assert self._turnrd_embedder is not None   # narrowed (validated in __init__)

        # Mode 2: get all per-trajectory labels in one judge call.
        per_traj_judge_labels: list[list[float]] | None = None
        if self._judge_decomposer is not None:
            try:
                per_traj_judge_labels = self._judge_decomposer.decompose(group)
            except Exception as exc:  # pragma: no cover (network/cache failure)
                logger.warning(
                    "TurnRD producer: judge_decomposer.decompose() failed for "
                    "task_id=%s (%s); emitting Mode-1 rows (judge_labels=None) "
                    "for this group.",
                    group.task_id,
                    exc,
                )
                per_traj_judge_labels = None
            else:
                if len(per_traj_judge_labels) != len(group.trajectories):
                    raise ValueError(
                        "TurnRD producer: judge_decomposer.decompose returned "
                        f"{len(per_traj_judge_labels)} rows for K="
                        f"{len(group.trajectories)} trajectories."
                    )

        path = self._turnrd_emit_path
        with open(path, "a") as fh:
            for i, traj in enumerate(group.trajectories):
                if not traj.turns:
                    # Match dataset-reader behavior: empty trajectories are
                    # skipped (no row → no later warning at load time).
                    continue
                embed_t = self._turnrd_embedder(traj)
                # Defensive: validate shape before serialising. The
                # adapter's contract is `[T_i, D]`.
                if embed_t.dim() != 2 or embed_t.shape[0] != len(traj.turns):
                    raise ValueError(
                        "TurnRD producer: embedder returned shape "
                        f"{tuple(embed_t.shape)} for trajectory with "
                        f"{len(traj.turns)} turns (task_id={traj.task_id})."
                    )
                turn_embeds = embed_t.detach().to(device="cpu").tolist()
                judge_labels: list[float] | None
                if per_traj_judge_labels is not None:
                    raw = per_traj_judge_labels[i]
                    if len(raw) != len(traj.turns):
                        raise ValueError(
                            "TurnRD producer: judge_decomposer returned "
                            f"{len(raw)} labels for trajectory with "
                            f"{len(traj.turns)} turns (task_id={traj.task_id})."
                        )
                    judge_labels = [float(x) for x in raw]
                else:
                    judge_labels = None
                # Round-trip through TurnRDRecord so producer-side schema
                # bugs surface here, not at the next trainer launch.
                rec = TurnRDRecord(
                    task_id=str(traj.task_id),
                    turn_embeds=turn_embeds,
                    final_reward=float(traj.final_reward),
                    judge_labels=judge_labels,
                )
                fh.write(json.dumps(asdict(rec)) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
