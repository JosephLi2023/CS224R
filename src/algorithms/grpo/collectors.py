"""K-trajectory rollout collector for H-GRPO.

Drives a vLLM-style runner through K parallel env instances for one task,
returning a `TrajectoryGroup` populated with action token ids and
rollout-time logprobs ready for the GRPO trainer.

Optional opt-in **TurnRD replay-buffer producer**. When the
collector is constructed with `turnrd_emit_path` + `turnrd_embedder`
(and, for Mode 2, `judge_decomposer`), each `collect_group` call
appends one JSONL row per non-empty trajectory using
`src.turnrd.dataset.TurnRDRecord` for schema validation. All producer
params default to `None`, so the existing flag-driven
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
        # Optional TurnRD replay-buffer producer hook.
        # All None -> producer disabled, default behavior unchanged.
        turnrd_emit_path: str | os.PathLike[str] | None = None,
        turnrd_embedder: TurnEmbedder | None = None,
        # Mode-2 only - per-trajectory normalized labels via a
        # JudgeDecomposer call. None -> Mode 1 (judge_labels=None).
        judge_decomposer: TurnRewardDecomposerLike | None = None,
        # Round index (0-based); written to each emitted row as `round_idx`
        # for the recency-decay path. None -> legacy rows without the field.
        round_idx: int | None = None,
        # When True (and turnrd_emit_path set), parse the AlfWorld goal from
        # each trajectory's Turn 0 obs and write it as `goal_text`. No-op on
        # non-AlfWorld envs. Default False preserves legacy output.
        turnrd_emit_goal_text: bool = False,
        # When True (with goal-text emission), also embed the goal text and
        # write it as `goal_emb` for the FiLM V-head. Cached per goal_text
        # within a collect_group call. Default False.
        turnrd_emit_goal_emb: bool = False,
    ) -> None:
        """K-trajectory rollout collector.

        With reuse_envs=True (default), env instances are cached and reused
        across episodes via env.reset() (skips WebShop reload; review M2).
        When turnrd_emit_path is set, each collect_group appends one
        TurnRDRecord JSONL row per non-empty trajectory (flush + fsync per
        row); turnrd_embedder is required and judge_decomposer enables Mode-2
        labels.
        """
        self.runner = runner
        self.env_factory = env_factory
        self.render = prompt_renderer
        self.parse = action_parser
        self.cfg = cfg or RolloutCollectorConfig()
        self.reuse_envs = reuse_envs
        self._env_pool: list[Any] = []
        # Producer plumbing (validated lazily so the import-time
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
        self._round_idx: int | None = (
            int(round_idx) if round_idx is not None else None
        )
        self._turnrd_emit_goal_text: bool = bool(turnrd_emit_goal_text)
        self._turnrd_emit_goal_emb: bool = bool(turnrd_emit_goal_emb)
        # Per-round cache: goal_text -> goal_emb (list[float]). Reset at
        # the start of every `collect_group` call so we don't grow
        # unbounded across the (~80) episodes in a round. The same goal
        # text repeats across the K rollouts of one task, so caching
        # saves K-1 redundant embedder calls per group.
        self._goal_emb_cache: dict[str, list[float]] = {}

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
        """K parallel rollouts on the same task -> populated TrajectoryGroup."""
        if getattr(sampling, "n", 1) != 1:
            sampling = _override_n(sampling, 1)

        # Per-call cache reset (see __init__ for rationale).
        self._goal_emb_cache = {}

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
                # Defensive: vLLM may return an empty completion list (e.g.
                # greedy EOS as top-1) -> treat as a no-op turn (empty action,
                # no tokens). Also guards len(outs) < len(live_idx).
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

                next_state, reward, done, info = envs[i].step(action_text)

                token_ids = tuple(getattr(gen, "token_ids", ()) or ())
                token_logprobs = tuple(getattr(gen, "token_logprobs", ()) or ())
                prompt_token_count = int(getattr(gen, "prompt_token_count", 0) or 0)
                prompt_token_ids = tuple(getattr(gen, "prompt_token_ids", ()) or ())

                # Optional dense per-turn shaping signal from the env adapter
                # via info["intermediate_reward"] (ALFWorld expert-plan-length
                # delta). None on envs that don't surface it (WebShop).
                inter_raw = info.get("intermediate_reward") if isinstance(info, dict) else None
                intermediate_reward = float(inter_raw) if inter_raw is not None else None

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
                    intermediate_reward=intermediate_reward,
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

        # Emit TurnRD replay-buffer rows (if the producer was
        # configured at construction time). Skipped entirely otherwise.
        if self._turnrd_emit_path is not None:
            self._emit_turnrd_records(group)

        return group, stats

    # TurnRD replay-buffer producer

    def _emit_turnrd_records(self, group: TrajectoryGroup) -> None:
        """Append one JSONL row (TurnRDRecord) per non-empty trajectory.

        Mode 2 (judge_decomposer set) calls decompose(group) once and slices
        the result into per-row judge_labels; Mode 1 writes judge_labels=None.
        Empty trajectories are skipped. Each row is flushed + fsync'd so a
        crash never leaves a half-written line.
        """
        # Local import keeps the collector module torch-free at import time.
        from src.turnrd.dataset import TurnRDRecord  # type: ignore[import-not-found]

        # The goal extractor is pure-Python (no torch / no heavy deps); import
        # once up front when goal-text emission is enabled, rather than
        # re-importing for every trajectory in every group.
        extract_goal_text = None
        if self._turnrd_emit_goal_text:
            from src.turnrd.goal_extractor import (  # type: ignore[import-not-found]
                extract_goal_text as _extract_goal_text,
            )
            extract_goal_text = _extract_goal_text

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
                    # skipped (no row -> no later warning at load time).
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
                # Per-turn env signal: step rewards are already deltas, so
                # forward as-is (v2 trainer's per-turn value-head target).
                progress: list[float] = [float(t.raw_env_reward) for t in traj.turns]
                # Optional dense progress signal sourced from the env
                # adapter (e.g. ALFWorld expert-plan-length deltas). All-
                # or-nothing per-trajectory emission so the dataset
                # collator's per-batch all-or-nothing gate stays
                # well-defined: a trajectory either has a complete
                # signal or none at all (no partial coverage that would
                # masquerade as zeros at masked-out positions).
                progress_signal: list[float] | None
                # Emit if ANY turn has a non-None intermediate_reward,
                # zero-filling missing turns (a zero is a valid "no progress"
                # V-head target). Envs that never set it (WebShop) -> None.
                # Empty trajectories still produce None.
                if traj.turns and any(t.intermediate_reward is not None for t in traj.turns):
                    progress_signal = [
                        float(t.intermediate_reward) if t.intermediate_reward is not None else 0.0
                        for t in traj.turns
                    ]
                else:
                    progress_signal = None
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
                # Goal text for the FiLM goal-conditioned V-head. Populated
                # when goal-text emission is on and the extractor finds a goal
                # in the Turn 0 obs; otherwise None.
                goal_text_v: str | None = None
                goal_emb_v: list[float] | None = None
                if (
                    self._turnrd_emit_goal_text
                    and extract_goal_text is not None
                    and traj.turns
                ):
                    goal_text_v = extract_goal_text(
                        traj.turns[0].observation_text or ""
                    )
                    # Goal embedding: run the same embedder on a synthetic
                    # single-turn trajectory (obs = goal text), take row 0.
                    # Cached by goal_text so the K rollouts share one call.
                    if (
                        self._turnrd_emit_goal_emb
                        and goal_text_v is not None
                    ):
                        cache_key = goal_text_v
                        cached = self._goal_emb_cache.get(cache_key)
                        if cached is not None:
                            goal_emb_v = cached
                        else:
                            try:
                                synth_traj = Trajectory(
                                    task_id=str(traj.task_id),
                                    env_name=traj.env_name,
                                    turns=[
                                        TurnRecord(
                                            turn_idx=0,
                                            observation_text=goal_text_v,
                                            action_text="",
                                            raw_env_reward=0.0,
                                        )
                                    ],
                                    final_reward=0.0,
                                )
                                ge_t = self._turnrd_embedder(synth_traj)
                                if ge_t.dim() != 2 or ge_t.shape[0] < 1:
                                    raise ValueError(
                                        "turnrd_embedder on goal-only "
                                        f"trajectory returned shape "
                                        f"{tuple(ge_t.shape)}; expected [1, D]."
                                    )
                                goal_emb_v = (
                                    ge_t[0].detach().to(device="cpu").tolist()
                                )
                                self._goal_emb_cache[cache_key] = goal_emb_v
                            except Exception as exc:  # pragma: no cover
                                # Embedder failure must not crash the
                                # producer - fall back to goal_emb=None.
                                logger.warning(
                                    "TurnRD producer: goal_emb embedder failed "
                                    "for task_id=%s (%s); emitting goal_emb=None.",
                                    traj.task_id,
                                    exc,
                                )
                                goal_emb_v = None
                # Round-trip through TurnRDRecord so producer-side schema
                # bugs surface here, not at the next trainer launch.
                rec = TurnRDRecord(
                    task_id=str(traj.task_id),
                    turn_embeds=turn_embeds,
                    final_reward=float(traj.final_reward),
                    judge_labels=judge_labels,
                    progress=progress,
                    progress_signal=progress_signal,
                    round_idx=self._round_idx,
                    goal_text=goal_text_v,
                    goal_emb=goal_emb_v,
                )
                fh.write(json.dumps(asdict(rec)) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
