"""CounterFactual per-turn reward decomposer (Method A revival).

Estimates each turn's counterfactual contribution `r_hat_t = R_actual -
R_baseline_t`, where `R_baseline_t` is the mean return from replacing `a_t` with
`N` policy-sampled alternatives and completing greedily. State `s_t` is recovered
by deterministic replay, so the env must be deterministic given `(task_id,
action_sequence)` (FakeWebShop and real WebShop). Knobs are documented on
`__init__`; `has_learnable_params = False`.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Protocol

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord

logger = logging.getLogger(__name__)


# Duck-typed Protocol: avoid importing VLLMRunner (and torch + vllm) at module
# import time; accept any object with the structural shape we use.
class _RunnerLike(Protocol):
    def generate_rich(self, prompts: list[str], sampling: Any) -> list[list[Any]]: ...


# Same callable shapes used by `RolloutCollector`.
PromptRenderer = Callable[[Any, list[TurnRecord]], str]
ActionParser = Callable[[str], str]
EnvFactory = Callable[[], Any]
SamplingFactory = Callable[..., Any]
"""Builds a sampling-params object the runner understands. Production wires
``functools.partial(SamplingParams, return_logprobs=False)``; tests can pass
a no-arg dataclass factory.

We accept a factory rather than two pre-built sampling objects because
``SamplingParams`` lives in ``src.policy.vllm_runner`` (which imports
torch lazily); keeping the construction at the call site lets the
decomposer module remain torch-free for unit tests.
"""


def _safe_reset(env: Any, task_id: Any) -> Any:
    """Mirror of `RolloutCollector._safe_reset`: real WebShop only accepts
    `task_id` via `session=`. Duplicated here to keep deps minimal.
    """
    try:
        return env.reset(task_id=task_id)
    except TypeError:
        return env.reset()


class CounterFactualDecomposer:
    """Per-turn reward decomposer driven by short counterfactual rollouts.

    Built from the same pieces as `RolloutCollector` plus a `sampling_factory`
    (for the high-temp alt draw and near-greedy completion). `decompose` matches
    the `PerTurnDecomposer` contract and the object is also directly callable.
    """

    def __init__(
        self,
        runner: _RunnerLike,
        env_factory: EnvFactory,
        prompt_renderer: PromptRenderer,
        action_parser: ActionParser,
        sampling_factory: SamplingFactory,
        *,
        n_alt_actions: int = 2,
        max_completion_turns: int = 3,
        cf_temperature: float = 1.0,
        completion_temperature: float = 0.0,
        cf_max_tokens: int = 48,
        n_turns_per_traj: int = 0,
        skip_if_zero_R: bool = True,
        output_mode: str = "raw_delta",
        seed: int = 0,
        max_env_pool_size: int | None = None,
        check_state_consistency: bool = False,
    ) -> None:
        if n_alt_actions < 1:
            raise ValueError(f"n_alt_actions must be ≥ 1; got {n_alt_actions}.")
        if max_completion_turns < 0:
            raise ValueError(
                f"max_completion_turns must be ≥ 0; got {max_completion_turns}."
            )
        if output_mode not in ("raw_delta", "normalized"):
            raise ValueError(
                f"output_mode must be 'raw_delta' or 'normalized'; got {output_mode!r}."
            )
        if max_env_pool_size is not None and int(max_env_pool_size) < 1:
            raise ValueError(
                f"max_env_pool_size must be ≥ 1 or None; got {max_env_pool_size}."
            )

        self.runner = runner
        self.env_factory = env_factory
        self.render = prompt_renderer
        self.parse = action_parser
        self.sampling_factory = sampling_factory

        self.n_alt = int(n_alt_actions)
        self.max_completion = int(max_completion_turns)
        self.cf_temperature = float(cf_temperature)
        self.completion_temperature = float(completion_temperature)
        self.cf_max_tokens = int(cf_max_tokens)
        self.n_turns_per_traj = int(n_turns_per_traj)
        self.skip_if_zero_R = bool(skip_if_zero_R)
        self.output_mode = output_mode
        self.max_env_pool_size: int | None = (
            int(max_env_pool_size) if max_env_pool_size is not None else None
        )
        self.check_state_consistency = bool(check_state_consistency)

        # Seeded RNG for reproducible turn subsampling, independent of the
        # policy/env RNGs.
        self._rng = random.Random(int(seed))

        # Env pool, reused across decompose() calls; each CF rollout needs its
        # own stateful env. Grown on demand up to `max_env_pool_size`; overflow
        # uses ephemeral envs discarded after the call.
        self._env_pool: list[Any] = []

    # Internal helpers

    def _acquire_envs(self, n: int) -> list[Any]:
        """Return ``n`` env instances, drawing from the pool first.

        When ``max_env_pool_size`` caps the pool below ``n``, the extras are
        built per-call (not cached) to bound resident memory.
        """
        cap = self.max_env_pool_size
        if cap is None:
            target = n
        else:
            target = min(n, cap)
        while len(self._env_pool) < target:
            self._env_pool.append(self.env_factory())
        if n <= len(self._env_pool):
            return self._env_pool[:n]
        # Overflow path: build the extras ephemerally.
        extras = [self.env_factory() for _ in range(n - len(self._env_pool))]
        return self._env_pool[:] + extras

    def _select_turn_indices(self, T: int) -> list[int]:
        """Turn indices to run CF on: all turns when ``n_turns_per_traj == 0``,
        else a without-replacement sample of that size.
        """
        if T <= 0:
            return []
        if self.n_turns_per_traj == 0 or self.n_turns_per_traj >= T:
            return list(range(T))
        return sorted(self._rng.sample(range(T), self.n_turns_per_traj))

    def _build_alt_sampling(self) -> Any:
        """High-temperature sampling for the alt-action draw (n=N)."""
        return self.sampling_factory(
            n=self.n_alt,
            temperature=self.cf_temperature,
            max_tokens=self.cf_max_tokens,
            return_logprobs=False,
        )

    def _build_completion_sampling(self) -> Any:
        """Near-greedy single-sample for the completion rollout."""
        return self.sampling_factory(
            n=1,
            temperature=self.completion_temperature,
            max_tokens=self.cf_max_tokens,
            return_logprobs=False,
        )

    # Public API

    def __call__(self, group: TrajectoryGroup) -> list[list[float]]:
        return self.decompose(group)

    def decompose(self, group: TrajectoryGroup) -> list[list[float]]:
        K = len(group.trajectories)
        if K == 0:
            return []

        # Per-trajectory plan: which turns to CF on + the baseline buffer.
        plans: list[tuple[Trajectory, list[int], list[float]]] = []
        for traj in group.trajectories:
            T = len(traj.turns)
            if T == 0:
                plans.append((traj, [], []))
                continue
            # Default 0.0 so skipped turns contribute 0 to the advantage.
            per_turn = [0.0] * T
            if self.skip_if_zero_R and float(traj.final_reward) == 0.0:
                # Zero R: nothing to attribute (same as Method C).
                plans.append((traj, [], per_turn))
                continue
            turn_idxs = self._select_turn_indices(T)
            plans.append((traj, turn_idxs, per_turn))

        # Flat list of CF rollouts, one per (traj_idx, turn_idx, alt_idx).
        cf_units: list[dict[str, Any]] = []
        for i, (traj, turn_idxs, _) in enumerate(plans):
            for t in turn_idxs:
                # Prefix history for the renderer: turns before t.
                prefix_turns: list[TurnRecord] = list(traj.turns[:t])
                prefix_actions: list[str] = [tr.action_text for tr in prefix_turns]
                for k in range(self.n_alt):
                    cf_units.append(
                        {
                            "i": i,
                            "t": t,
                            "k": k,
                            "prefix_turns": prefix_turns,
                            "prefix_actions": prefix_actions,
                        }
                    )

        if not cf_units:
            # No CF units (empty / R==0 trajectories or fully subsampled out).
            return [p[2] for p in plans]

        # Phase A: replay every CF env to its starting state s_t. One env
        # per CF unit; the env pool reuses instances across decompose() calls.
        envs = self._acquire_envs(len(cf_units))
        states: list[Any] = []
        for unit, env in zip(cf_units, envs):
            state = _safe_reset(env, group.task_id)
            # Accumulate prefix-replay rewards so R_baseline_t shares R_actual's
            # accounting (sums all step rewards). No-op on sparse-terminal envs;
            # needed on shaping-reward envs or delta_t would be biased upward.
            prefix_R = 0.0
            for action_text in unit["prefix_actions"]:
                state, _r, _done, _info = env.step(action_text)
                prefix_R += float(_r)
            unit["prefix_R"] = prefix_R
            states.append(state)

        # Phase B: sample alt actions in one batched call, deduped by (i, t)
        # (one prompt per turn-position), then slice the N alts back per unit.
        groups_by_it: dict[tuple[int, int], list[int]] = {}
        for ci, unit in enumerate(cf_units):
            groups_by_it.setdefault((unit["i"], unit["t"]), []).append(ci)

        alt_prompts: list[str] = []
        alt_owners: list[tuple[int, int]] = []  # (i, t) per prompt
        for (i, t), indices in groups_by_it.items():
            # Units in `indices` share a prefix; render once from the first
            # unit's state. Deterministic envs replay identically; for stochastic
            # ones set `check_state_consistency=True` to catch divergence.
            first = indices[0]
            if self.check_state_consistency and len(indices) > 1:
                ref_obs = getattr(states[first], "observation_text", None)
                for ci in indices[1:]:
                    other_obs = getattr(states[ci], "observation_text", None)
                    if other_obs != ref_obs:
                        raise RuntimeError(
                            "CounterFactualDecomposer.check_state_consistency: "
                            f"states[{ci}].observation_text diverges from "
                            f"states[{first}].observation_text at (i={i}, t={t}). "
                            "The env appears to be non-deterministic on replay; "
                            "either disable check_state_consistency to accept "
                            "the variance or render per-unit (one prompt per CF "
                            "unit) by setting n_alt_actions=1 and looping "
                            "externally."
                        )
            alt_prompts.append(self.render(states[first], cf_units[first]["prefix_turns"]))
            alt_owners.append((i, t))

        alt_sampling = self._build_alt_sampling()
        alt_outs = self.runner.generate_rich(alt_prompts, alt_sampling)

        # Distribute alt actions back to CF units (alt_outs[p] is N gens).
        for p, (i, t) in enumerate(alt_owners):
            indices = groups_by_it[(i, t)]
            gens = alt_outs[p] if p < len(alt_outs) else []
            for slot, ci in enumerate(indices):
                gen = gens[slot] if slot < len(gens) else None
                action_text = (
                    self.parse(getattr(gen, "text", "") or "") if gen is not None else ""
                )
                cf_units[ci]["alt_action"] = action_text

        # Step every CF env with its alt action - the actual intervention.
        # Track per-unit (R_so_far, done, history) as Phase C's bookkeeping.
        for ci, unit in enumerate(cf_units):
            env = envs[ci]
            alt_action = unit.get("alt_action", "")
            # Seed `R` with the Phase-A prefix reward so the baseline sums all
            # step rewards (matches `traj.final_reward`).
            unit["R"] = float(unit.get("prefix_R", 0.0))
            unit["done"] = False
            unit["history"] = list(unit["prefix_turns"])
            try:
                next_state, reward, done, _info = env.step(alt_action)
            except Exception as exc:
                logger.warning(
                    "CounterFactualDecomposer: env.step(alt_action=%r) raised %r "
                    "for task_id=%s, traj_idx=%d, turn=%d; treating as terminal R=0.",
                    alt_action,
                    exc,
                    group.task_id,
                    unit["i"],
                    unit["t"],
                )
                next_state, reward, done = states[ci], 0.0, True
            unit["R"] += float(reward)
            # Append the alt step so the next renderer sees the full transcript.
            # Don't `... or ""` the obs: preserve a legitimately empty one.
            obs_text = getattr(states[ci], "observation_text", "")
            unit["history"].append(
                TurnRecord(
                    turn_idx=unit["t"],
                    observation_text=obs_text,
                    action_text=alt_action,
                )
            )
            states[ci] = next_state
            unit["done"] = bool(done)

        # Phase C: greedy completion, one batched call per depth. Only the final
        # R matters, so token-id / prompt-budget tracking are omitted.
        completion_sampling = self._build_completion_sampling()
        for _depth in range(self.max_completion):
            live = [ci for ci, u in enumerate(cf_units) if not u["done"]]
            if not live:
                break
            prompts = [
                self.render(states[ci], cf_units[ci]["history"]) for ci in live
            ]
            outs = self.runner.generate_rich(prompts, completion_sampling)
            for slot, ci in enumerate(live):
                gen_list = outs[slot] if slot < len(outs) else []
                gen = gen_list[0] if gen_list else None
                action_text = (
                    self.parse(getattr(gen, "text", "") or "") if gen is not None else ""
                )
                env = envs[ci]
                try:
                    next_state, reward, done, _info = env.step(action_text)
                except Exception as exc:
                    logger.warning(
                        "CounterFactualDecomposer: env.step during completion "
                        "raised %r for task_id=%s, traj_idx=%d, turn=%d; "
                        "treating as terminal R+=0.",
                        exc,
                        group.task_id,
                        cf_units[ci]["i"],
                        cf_units[ci]["t"],
                    )
                    next_state, reward, done = states[ci], 0.0, True
                cf_units[ci]["R"] += float(reward)
                # As in Phase B: keep a legitimately empty observation.
                obs_text = getattr(states[ci], "observation_text", "")
                cf_units[ci]["history"].append(
                    TurnRecord(
                        turn_idx=cf_units[ci]["t"] + 1 + _depth,
                        observation_text=obs_text,
                        action_text=action_text,
                    )
                )
                states[ci] = next_state
                cf_units[ci]["done"] = bool(done)

        # Phase D: per-(i, t) baseline = mean unit R -> delta_t -> optional rescale.
        baseline_R: dict[tuple[int, int], float] = {}
        for (i, t), indices in groups_by_it.items():
            rs = [cf_units[ci]["R"] for ci in indices]
            baseline_R[(i, t)] = sum(rs) / max(1, len(rs))

        out: list[list[float]] = []
        for i, (traj, turn_idxs, per_turn) in enumerate(plans):
            T = len(traj.turns)
            R_actual = float(traj.final_reward)
            for t in turn_idxs:
                per_turn[t] = R_actual - baseline_R.get((i, t), 0.0)
            if self.output_mode == "normalized":
                # Rescale so per-turn rewards sum to R, weighting by max(0, delta_t)
                # (negative deltas don't remove credit). Uniform R/T if all <= 0.
                weights = [max(0.0, x) for x in per_turn]
                total_w = sum(weights)
                if total_w > 1e-9 and T > 0:
                    per_turn = [R_actual * w / total_w for w in weights]
                elif T > 0:
                    per_turn = [R_actual / T] * T
            out.append(per_turn)
        return out

    # PerTurnDecomposer surface helpers

    @property
    def has_learnable_params(self) -> bool:
        """False: CF has no trainable params, so the trainer skips the second
        AdamW + the C3 consistency loss.
        """
        return False


def build_counterfactual_decomposer(
    cfg: dict[str, Any],
    *,
    runner: _RunnerLike,
    env_factory: EnvFactory,
    prompt_renderer: PromptRenderer,
    action_parser: ActionParser,
    sampling_factory: SamplingFactory,
) -> CounterFactualDecomposer:
    """Factory for `build_decomposer`'s `"counterfactual"` branch; reads knobs
    from `cfg["counterfactual"]` (see `CounterFactualDecomposer.__init__` for
    names and defaults).
    """
    cf_cfg = cfg.get("counterfactual", {}) or {}
    pool_cap = cf_cfg.get("max_env_pool_size", None)
    return CounterFactualDecomposer(
        runner=runner,
        env_factory=env_factory,
        prompt_renderer=prompt_renderer,
        action_parser=action_parser,
        sampling_factory=sampling_factory,
        n_alt_actions=int(cf_cfg.get("n_alt_actions", 2)),
        max_completion_turns=int(cf_cfg.get("max_completion_turns", 3)),
        cf_temperature=float(cf_cfg.get("cf_temperature", 1.0)),
        completion_temperature=float(cf_cfg.get("completion_temperature", 0.0)),
        cf_max_tokens=int(cf_cfg.get("cf_max_tokens", 48)),
        n_turns_per_traj=int(cf_cfg.get("n_turns_per_traj", 0)),
        skip_if_zero_R=bool(cf_cfg.get("skip_if_zero_R", True)),
        output_mode=str(cf_cfg.get("output_mode", "raw_delta")),
        seed=int(cf_cfg.get("seed", 0)),
        max_env_pool_size=(int(pool_cap) if pool_cap is not None else None),
        check_state_consistency=bool(cf_cfg.get("check_state_consistency", False)),
    )
