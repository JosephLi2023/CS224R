"""CounterFactual per-turn reward decomposer (Method A revival).

Idea
----
For each turn ``t`` in trajectory ``τ_i`` we estimate the *counterfactual*
contribution of the action the policy actually took:

    r̂_t = R_actual − R_baseline_t

where ``R_baseline_t`` is the mean episode return obtained by replacing
``a_t`` with one of ``N`` alternative actions sampled from the policy at
state ``s_t`` and completing the rollout greedily for up to
``max_completion_turns`` more turns.

This is the literal "counterfactual rollouts" signal the original
design listed as **Method A** but had to cut from the WebShop
plan because WebShop's *in-progress* state is not snapshot-able. The
trick we use here is **deterministic replay**: WebShop episodes are
deterministic given ``(task_id, action_sequence)``, so we can recover
``s_t`` by ``env.reset(task_id=...) → step(a_0) → … → step(a_{t-1})``.
This works for FakeWebShop (the unit-test env) and the real WebShop
text env (whose ``session=task_id`` reset is deterministic).

Why we want it
--------------
The learned TurnRD decomposer (Method B) currently underperforms the
trivial Method C (`progress_decomposer`) on our 4-method bake-off
(`experiments/manifests/4method_comparison.txt`). Counterfactual
rollouts are the only credit signal in our toolbox that has a defensible
*causal* interpretation rather than a learned proxy — exactly the kind
of strong baseline a "TurnRD ≥ CF" or "TurnRD ≤ CF" comparison needs.

Cost
----
For ``K`` trajectories × average ``T̄`` turns × ``N`` alt actions ×
``C`` completion turns, the worst-case env-step count per group is::

    K * N * [ T̄(T̄-1)/2  +  1  +  C ]
        ↑ prefix replay     ↑ alt   ↑ completion

The ``T̄(T̄-1)/2`` term comes from Phase A: the unit at turn index ``t``
replays exactly ``t`` env.step calls to reach ``s_t``, summed across
``t = 0..T̄-1``. Earlier drafts of this docstring used the loose bound
``K * T̄ * N * (T̄ + C)`` which over-counted the trivially-true
rectangle area; the triangular sum is what real WebShop will pay.

With the defaults (K=4, T̄=4, N=2, C=3) the per-group budget is
``4 · 2 · (6 + 1 + 3) = 80`` env steps. With the production WebShop
setting (K=8, T̄=5, N=2, C=3) it's ``8 · 2 · (10 + 1 + 3) ≈ 224`` env
steps when ``n_turns_per_traj`` runs all turns; subsampling 2 turns per
trajectory (the production config default) cuts it roughly in half.
The matching number of LLM calls is ``≤ K * T̄`` batched alt-action
samples (one prompt per (i, t) at ``n=N``) plus ``≤ C`` batched greedy
calls during completion. On FakeWebShop the env steps run in
milliseconds; on real WebShop (each step ~100 ms over HTTP) this adds
≈ 3-5× wall-clock per training group.

Resident memory: ``self._env_pool`` grows monotonically up to
``len(cf_units)`` env instances across the largest group seen. Each
WebShop instance carries the full product/goal corpus in memory, so a
full-turns K=8/T̄=5/N=2 run can keep ≈80 env instances resident for
the lifetime of the trainer, in addition to the ``K=8`` envs the
``RolloutCollector`` keeps. Cap with ``max_env_pool_size`` when running
on a constrained Modal worker; ``None`` (default) lets the pool grow
freely.

Knobs:
* ``n_alt_actions`` — N. 2 is the cheapest meaningful value.
* ``max_completion_turns`` — C. Cap at ``rollout.max_turns − t`` since
  beyond the original trajectory's horizon there is no signal.
* ``n_turns_per_traj`` — randomly sample this many turns per trajectory
  for CF instead of all of them. ``0`` (default) = all turns.
* ``skip_if_zero_R`` — when the trajectory's actual ``R`` is 0 the
  shape of α doesn't matter (any decomposition multiplies to 0); skip
  the CF rollouts entirely and emit ``[0.0] * T``.
* ``max_env_pool_size`` — cap on the resident env-pool size; ``None``
  (default) means unlimited (the pool grows to the largest group seen
  and never shrinks). Set this when running on a constrained Modal
  worker that can't afford ~80 WebShop instances live; values lower
  than ``len(cf_units)`` for a given call cause envs to be re-built
  per-call instead of reused (correctness preserved, wall-clock
  worse).
* ``check_state_consistency`` — debug-only knob. When ``True`` the
  decomposer asserts that the per-unit env states sharing the same
  ``(i, t)`` carry identical ``observation_text`` after the Phase-A
  replay. Default ``False`` (zero-overhead production path). Useful
  on stochastic envs (e.g., AlfWorld variants) where deterministic
  replay is not guaranteed; turn on to surface state divergence early
  rather than silently rendering a single state's prompt for all alt
  units in the (i, t) bucket.
* ``output_mode``:
   - ``"raw_delta"`` (default) — emit ``R_actual − R_baseline_t``
     directly. Matches the natural causal-attribution semantics; per-turn
     values may NOT sum to ``R``. The trainer's
     ``compute_turn_advantages`` normalizes per-position across the
     K-group so absolute scale matters less than ordering.
   - ``"normalized"`` — rescale so ``Σ_t r̂_t = R`` (preserves the
     sum invariant the C3 consistency loss expects). Falls back to
     uniform ``R/T`` when all Δ_t ≤ 0.

Reward accounting precondition
------------------------------
This decomposer assumes that ``traj.final_reward`` accumulates ALL
step rewards over the whole episode (the ``RolloutCollector`` contract
at ``src/algorithms/grpo/collectors.py``). To keep ``R_actual`` and
``R_baseline_t`` on the same accounting basis, the Phase-A prefix
replay reward is captured into ``unit["prefix_R"]`` and added back to
``unit["R"]`` BEFORE the alt step — i.e. the baseline includes the
same prefix step rewards the original rollout earned. This makes
Δ_t = R_actual − R_baseline_t correct on envs that emit non-terminal
(shaping) rewards, not just on the sparse-terminal WebShop /
FakeWebShop case.

Adapter contract
----------------
Callable matching ``PerTurnDecomposer = Callable[[TrajectoryGroup],
list[list[float]]]``. Signature parity with ``JudgeDecomposer.decompose``
and ``progress_decomposer``. ``has_learnable_params = False`` (the
trainer detects this via ``getattr`` and skips the second optimizer
+ the C3 consistency-loss reattach — same path as Methods A/C today).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Protocol

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord

logger = logging.getLogger(__name__)


# Duck-typed Protocols. We DO NOT import VLLMRunner here because that would
# pull torch + vllm at module-import time; instead we accept any object that
# matches the structural shape used by the rollout collector.
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
torch lazily) — keeping the construction at the call site lets the
decomposer module remain torch-free for unit tests.
"""


def _safe_reset(env: Any, task_id: Any) -> Any:
    """Mirror of `RolloutCollector._safe_reset` so we behave identically.

    The real WebShop env accepts `task_id` only via the `session=` kwarg
    (mapped by `WebShopAdapter.reset`). The collector's helper handles
    this; we duplicate the trivial logic here rather than import it to
    keep the decomposer's deps minimal.
    """
    try:
        return env.reset(task_id=task_id)
    except TypeError:
        return env.reset()


class CounterFactualDecomposer:
    """Per-turn reward decomposer driven by short counterfactual rollouts.

    Instantiated with the same building blocks as ``RolloutCollector``
    (runner, env_factory, prompt_renderer, action_parser); also accepts
    a ``sampling_factory`` so the decomposer can build the two distinct
    sampling configs it needs (high-temp for the alt action sample,
    near-greedy for the completion). See module docstring for the full
    list of knobs.

    Public API: ``decompose(group: TrajectoryGroup) -> list[list[float]]``
    matching the existing ``PerTurnDecomposer`` callable contract. The
    object is also directly callable via ``__call__`` so callers can
    pass either ``cf_decomposer`` or ``cf_decomposer.decompose`` to
    ``HGPOTrainer(decomposer=...)`` — same convention as
    ``TurnRDDecomposer``.
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

        # Seeded RNG for reproducible turn subsampling. Independent from
        # the policy/env RNGs; the runner's own seed governs alt-action
        # sampling stochasticity.
        self._rng = random.Random(int(seed))

        # Env pool: acquired lazily, reused across decompose() calls.
        # Each CF rollout needs its own env because envs are stateful
        # and we step them through replay + alt + completion. Pool is
        # grown on demand and capped at `max_env_pool_size` (None =
        # unlimited). Pool overflow falls back to ephemeral envs that
        # are discarded after each decompose() call — correctness is
        # preserved at the cost of extra factory invocations.
        self._env_pool: list[Any] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _acquire_envs(self, n: int) -> list[Any]:
        """Return ``n`` env instances, drawing from the pool first.

        When ``max_env_pool_size`` is set and ``n`` exceeds it, the
        pool is grown only up to the cap and the remainder are built
        via ``env_factory()`` per-call (NOT cached). This bounds the
        decomposer's resident memory at the cost of paying the env
        factory's setup time on every overflow call.
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
        """Return the turn indices we'll run CF on for one trajectory.

        ``n_turns_per_traj == 0`` ⇒ all turns ``[0..T-1]``.
        ``n_turns_per_traj > 0``  ⇒ sample without replacement.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, group: TrajectoryGroup) -> list[list[float]]:
        return self.decompose(group)

    def decompose(self, group: TrajectoryGroup) -> list[list[float]]:
        K = len(group.trajectories)
        if K == 0:
            return []

        # Per-trajectory work plan: which turns we'll CF on, and the
        # baseline buffer we'll fill in.
        plans: list[tuple[Trajectory, list[int], list[float]]] = []
        for traj in group.trajectories:
            T = len(traj.turns)
            if T == 0:
                plans.append((traj, [], []))
                continue
            # Default per-turn rewards = 0.0 (so turns we skip — either
            # because of zero-R short-circuit or because of subsample —
            # contribute 0 to the H-GRPO advantage).
            per_turn = [0.0] * T
            if self.skip_if_zero_R and float(traj.final_reward) == 0.0:
                # Zero R ⇒ no information to attribute; uniform 0 is
                # what Method C would produce too.
                plans.append((traj, [], per_turn))
                continue
            turn_idxs = self._select_turn_indices(T)
            plans.append((traj, turn_idxs, per_turn))

        # Build the flat list of CF rollouts: one per (traj_idx, turn_idx, alt_idx).
        # Each entry holds: the prefix turns to render with, the prefix actions
        # to replay through env, and the alt index slot it will fill.
        cf_units: list[dict[str, Any]] = []
        for i, (traj, turn_idxs, _) in enumerate(plans):
            for t in turn_idxs:
                # Prefix history we hand to the renderer: turns BEFORE t.
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
            # Either the whole group has empty / R==0 trajectories, or
            # n_turns_per_traj filtered everything out. Return whatever
            # plans hold (zeros).
            return [p[2] for p in plans]

        # --------------------------------------------------------------
        # Phase A: replay every CF env to its starting state s_t.
        #
        # We acquire one env per CF unit (each needs its own state).
        # On real WebShop this is the expensive step (~3 s reset);
        # the env pool reuses instances across decompose() calls so the
        # fixed per-task reset cost is only paid once.
        # --------------------------------------------------------------
        envs = self._acquire_envs(len(cf_units))
        states: list[Any] = []
        for unit, env in zip(cf_units, envs):
            state = _safe_reset(env, group.task_id)
            # Accumulate prefix-replay rewards so R_baseline_t is on the
            # same accounting basis as R_actual (which sums step rewards
            # across the WHOLE episode — see
            # `src/algorithms/grpo/collectors.py::collect_group` where
            # `rewards_so_far[i] += float(reward)` runs every step).
            # Sparse-terminal envs (WebShop, FakeWebShop) emit 0 here
            # and the bookkeeping is a no-op; shaping-reward envs need
            # this term or Δ_t = R_actual − R_baseline_t would be
            # biased upward by the prefix step rewards.
            prefix_R = 0.0
            for action_text in unit["prefix_actions"]:
                state, _r, _done, _info = env.step(action_text)
                prefix_R += float(_r)
            unit["prefix_R"] = prefix_R
            states.append(state)

        # --------------------------------------------------------------
        # Phase B: sample alt actions for every CF unit in ONE batched
        # call. The runner returns a list of n=N completions per prompt;
        # we deduplicate by (i, t) so we only build one prompt per
        # turn-position and slice the N alts back into the matching
        # CF units.
        # --------------------------------------------------------------
        # Map (i, t) → slice indices in cf_units for that group.
        groups_by_it: dict[tuple[int, int], list[int]] = {}
        for ci, unit in enumerate(cf_units):
            groups_by_it.setdefault((unit["i"], unit["t"]), []).append(ci)

        alt_prompts: list[str] = []
        alt_owners: list[tuple[int, int]] = []  # (i, t) per prompt
        for (i, t), indices in groups_by_it.items():
            # All units in `indices` share the same prefix; render once
            # using the first unit's state. For a deterministic env
            # (real WebShop, FakeWebShop) the per-unit replays produce
            # byte-identical states, so this is a perfect optimisation.
            # For stochastic envs (e.g., a future AlfWorld variant with
            # random seeds in `info`) the per-unit states may diverge
            # and rendering only one would silently bias the alt-action
            # sample — enable `check_state_consistency=True` to surface
            # such divergence early instead of letting it slip through.
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

        # Distribute alt actions back to CF units.
        # alt_outs[p] is list[N] of generations for prompt p.
        for p, (i, t) in enumerate(alt_owners):
            indices = groups_by_it[(i, t)]
            gens = alt_outs[p] if p < len(alt_outs) else []
            for slot, ci in enumerate(indices):
                gen = gens[slot] if slot < len(gens) else None
                action_text = (
                    self.parse(getattr(gen, "text", "") or "") if gen is not None else ""
                )
                cf_units[ci]["alt_action"] = action_text

        # --------------------------------------------------------------
        # Step every CF env with its alt action — this is the actual
        # *intervention*. We track per-unit (R_so_far, done, history)
        # as Phase C's bookkeeping.
        # --------------------------------------------------------------
        for ci, unit in enumerate(cf_units):
            env = envs[ci]
            alt_action = unit.get("alt_action", "")
            # Track the per-unit state for completion + final R.
            # Seed `R` with the prefix-replay reward captured in Phase A
            # so the baseline accumulates ALL step rewards (matches
            # `traj.final_reward`'s accounting). See the "Reward
            # accounting precondition" section of the module docstring.
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
            # Append the alt step to the history so the next-turn
            # renderer sees the full transcript. We intentionally do NOT
            # `... or ""` the obs — the getattr default already returns
            # "" for the missing-attribute case, and a legitimately empty
            # observation (which a well-behaved env may emit between
            # successful clicks) should be preserved verbatim.
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

        # --------------------------------------------------------------
        # Phase C: greedy completion. Loop up to `max_completion_turns`
        # batched generate_rich calls — one per completion depth — and
        # step each live unit's env. The heavy artillery from
        # `RolloutCollector` (token-id capture, prompt-budget tracking)
        # is intentionally absent because we ONLY need the final R per
        # CF rollout; tokens are not used downstream.
        # --------------------------------------------------------------
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
                # Same rationale as Phase B: drop the spurious `or ""`
                # collapse so a legitimately empty observation survives.
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

        # --------------------------------------------------------------
        # Phase D: aggregate per-(i, t) baseline ⇒ Δ_t ⇒ optional rescale.
        # --------------------------------------------------------------
        # baseline_R[(i, t)] = mean(unit["R"] across alt rollouts at (i, t))
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
                # Rescale so the per-turn rewards sum to R, preserving the
                # §3.2 invariant. Distribute weight by max(0, Δ_t) since
                # negative deltas (alt did better than actual) shouldn't
                # take credit AWAY from R. Fall back to uniform R/T when
                # all weights are non-positive.
                weights = [max(0.0, x) for x in per_turn]
                total_w = sum(weights)
                if total_w > 1e-9 and T > 0:
                    per_turn = [R_actual * w / total_w for w in weights]
                elif T > 0:
                    per_turn = [R_actual / T] * T
            out.append(per_turn)
        return out

    # ------------------------------------------------------------------
    # PerTurnDecomposer surface helpers
    # ------------------------------------------------------------------

    @property
    def has_learnable_params(self) -> bool:
        """Mirror of `JudgeDecomposer` / `progress_decomposer`. CF has no
        trainable parameters — the trainer's `compute_loss` will skip the
        second AdamW + the C3 consistency-loss reattach.
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
    """Factory used by `build_decomposer` for the `"counterfactual"` branch.

    Reads `cfg["counterfactual"]`:
      n_alt_actions: int (default 2)
      max_completion_turns: int (default 3)
      cf_temperature: float (default 1.0)
      completion_temperature: float (default 0.0)
      cf_max_tokens: int (default 48)
      n_turns_per_traj: int (default 0 = all turns)
      skip_if_zero_R: bool (default True)
      output_mode: "raw_delta" | "normalized" (default "raw_delta")
      seed: int (default 0)
      max_env_pool_size: int | None (default None = unlimited)
      check_state_consistency: bool (default False; debug-only)
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
