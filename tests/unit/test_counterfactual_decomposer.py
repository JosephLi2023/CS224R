"""Unit tests for `src.algorithms.hgpo.decomposers.counterfactual.CounterFactualDecomposer`.

Test plan
---------
1. `test_returns_correct_shape` — Δ_t array per trajectory matches T_i.
2. `test_replay_recovers_original_state` — env replay reproduces the same
   state at turn t given the same action prefix (deterministic FakeWebShop).
3. `test_correct_turn_has_largest_credit` — on FakeWebShopEnv the
   `click[item-0]` turn is the highest-reward causal step; CF baseline
   confirms it gets the largest delta when alt actions click wrong items.
4. `test_zero_R_short_circuit_emits_zeros` — `skip_if_zero_R=True` and a
   trajectory with R=0 returns `[0.0] * T` and SKIPS env replays.
5. `test_normalized_output_sums_to_R` — `output_mode="normalized"` ⇒
   `Σ_t r̂_t ≈ R` for non-zero deltas.
6. `test_normalized_falls_back_to_uniform_when_all_deltas_nonpositive`
   — when every alt action ≥ actual R, deltas ≤ 0 ⇒ uniform R/T fallback.
7. `test_n_turns_per_traj_subsamples` — only `n_turns_per_traj` turns get
   a non-zero CF delta; the rest stay 0.0.
8. `test_empty_trajectory_returns_empty_list` — mirrors JudgeDecomposer.
9. `test_factory_guard_rejects_missing_deps` — `build_decomposer(decomposer=
   "counterfactual")` without runner/env/render/parser/sampling raises.
10. `test_has_learnable_params_is_false` — trainer skips second optimizer.
11. `test_env_pool_reused_across_decompose_calls` — the env pool grows
    once and is not rebuilt on subsequent `decompose()` calls (matches
    the perf optimisation in `RolloutCollector`).

These tests do NOT require torch — the CF decomposer is pure-Python by
design (no model parameters). The fake runner / FakeWebShopEnv handle
the env-driven side; we only need to validate the orchestration logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.algorithms.grpo.rollout import (
    Trajectory,
    TrajectoryGroup,
    TurnRecord,
)
from src.algorithms.hgpo.decomposers import build_decomposer
from src.algorithms.hgpo.decomposers.counterfactual import CounterFactualDecomposer
from src.envs.fake_webshop import FakeWebShopEnv
from src.envs.prompts.react_webshop import (
    parse_react_action,
    render_webshop_turn_prompt,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Sampling:
    """Minimal stand-in for `src.policy.vllm_runner.SamplingParams` —
    accepts the same kwargs the CF decomposer's `sampling_factory`
    invocations pass.
    """

    n: int = 1
    temperature: float = 1.0
    max_tokens: int = 48
    return_logprobs: bool = False


@dataclass
class _Gen:
    """Mimics `src.policy.vllm_runner.GenerationOutput.text` access pattern."""

    text: str


class _RecipeRunner:
    """Test-only runner that returns a fixed action string sequence.

    The recipe maps `(prompt_substring, sample_index)` → action_text. We
    pattern-match against the most recent observation in the prompt so
    callers can script the alt action returned at each FakeWebShop stage
    independently.

    The runner discriminates ALT-SAMPLING calls (high temperature) from
    COMPLETION calls (near-greedy temperature) using `sampling.temperature`:
    only the alt-sampling phase consults the recipe; completion always
    falls back to the canonical "best" action so we can isolate the CF
    intervention's effect from completion noise.
    """

    def __init__(
        self,
        mapping: dict[tuple[str, int], str],
        *,
        completion_temperature: float = 0.0,
    ) -> None:
        self.mapping = mapping
        self.completion_temperature = completion_temperature
        self.calls: list[tuple[list[str], int, float]] = []

    def generate_rich(self, prompts, sampling):
        n = getattr(sampling, "n", 1)
        temp = float(getattr(sampling, "temperature", 1.0))
        self.calls.append((list(prompts), n, temp))
        is_alt_phase = temp > self.completion_temperature + 1e-9
        out = []
        for prompt in prompts:
            row = []
            for k in range(n):
                action = self._lookup(prompt, k, use_recipe=is_alt_phase)
                row.append(_Gen(text=f"Thought: t\nAction: {action}"))
            out.append(row)
        return out

    def _lookup(self, prompt: str, k: int, *, use_recipe: bool) -> str:
        # Use the LAST `Observation:` line (the current state) — earlier
        # observations are in the history and would mis-classify the stage.
        last_obs = ""
        for line in prompt.split("\n"):
            if line.startswith("Observation:"):
                last_obs = line
        if "search page" in last_obs or "search query" in last_obs.lower():
            stage = "search"
        elif "Search results" in last_obs:
            stage = "click"
        elif "On product page" in last_obs:
            stage = "buy"
        else:
            stage = "other"
        defaults = {
            "search": "search[laptop bag]",
            "click": "click[item-0]",
            "buy": "click[buy]",
            "other": "think[noop]",
        }
        if use_recipe and (stage, k) in self.mapping:
            return self.mapping[(stage, k)]
        return defaults[stage]


# ---------------------------------------------------------------------------
# Helpers — collect a real trajectory by stepping FakeWebShop with a script.
# ---------------------------------------------------------------------------


def _run_scripted_trajectory(
    actions: list[str],
    task_id: int = 0,
    final_reward_override: float | None = None,
) -> Trajectory:
    """Replay `actions` against FakeWebShopEnv to materialise a Trajectory.

    Returns a `Trajectory` whose turns carry the obs+action of the actual
    rollout (so the CF decomposer's replay path can reuse the same
    prefix). We use `final_reward_override` for tests that want to force
    a specific R independent of the env reward.
    """
    env = FakeWebShopEnv(max_steps=8)
    state = env.reset(task_id=task_id)
    turns: list[TurnRecord] = []
    R = 0.0
    for t, action in enumerate(actions):
        obs_text = state.observation_text
        next_state, reward, done, _info = env.step(action)
        R += float(reward)
        turns.append(
            TurnRecord(
                turn_idx=t,
                observation_text=obs_text,
                action_text=action,
            )
        )
        state = next_state
        if done:
            break
    return Trajectory(
        task_id=str(task_id),
        env_name="webshop",
        turns=turns,
        final_reward=R if final_reward_override is None else float(final_reward_override),
    )


def _make_decomposer(
    runner_mapping: dict[tuple[str, int], str] | None = None,
    *,
    n_alt_actions: int = 2,
    max_completion_turns: int = 3,
    n_turns_per_traj: int = 0,
    skip_if_zero_R: bool = True,
    output_mode: str = "raw_delta",
    seed: int = 0,
) -> tuple[CounterFactualDecomposer, _RecipeRunner]:
    runner = _RecipeRunner(runner_mapping or {})
    cf = CounterFactualDecomposer(
        runner=runner,
        env_factory=lambda: FakeWebShopEnv(max_steps=8),
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        sampling_factory=_Sampling,
        n_alt_actions=n_alt_actions,
        max_completion_turns=max_completion_turns,
        n_turns_per_traj=n_turns_per_traj,
        skip_if_zero_R=skip_if_zero_R,
        output_mode=output_mode,
        seed=seed,
    )
    return cf, runner


# ---------------------------------------------------------------------------
# 1. Shape contract
# ---------------------------------------------------------------------------


def test_returns_correct_shape():
    """Each trajectory's per-turn list has length T_i; group length == K."""
    actions_traj_a = ["search[laptop bag]", "click[item-0]", "click[buy]"]  # R=1.0, T=3
    actions_traj_b = ["search[laptop bag]", "click[item-2]", "click[buy]"]  # R=0.4, T=3
    traj_a = _run_scripted_trajectory(actions_traj_a, task_id=0)
    traj_b = _run_scripted_trajectory(actions_traj_b, task_id=0)
    group = TrajectoryGroup(
        task_id="0", env_name="webshop", trajectories=[traj_a, traj_b]
    )
    cf, _ = _make_decomposer()
    out = cf.decompose(group)
    assert len(out) == 2
    assert len(out[0]) == 3  # T_a
    assert len(out[1]) == 3  # T_b
    for row in out:
        assert all(isinstance(x, float) for x in row)


# ---------------------------------------------------------------------------
# 2. Replay correctness
# ---------------------------------------------------------------------------


def test_replay_recovers_original_state():
    """For a deterministic env, the CF replay reaches the exact same
    observation at turn t as the original rollout. We verify this by
    asserting the turn-2 alt prompt contains the 'On product page'
    observation that should follow `search → click[item-0]`.
    """
    actions = ["search[laptop bag]", "click[item-0]", "click[buy]"]
    traj = _run_scripted_trajectory(actions, task_id=0)
    group = TrajectoryGroup(task_id="0", env_name="webshop", trajectories=[traj])

    # Force the alt action at the buy stage to a no-op so the CF rollout
    # ends without buying — keeps the test focused on prompt content.
    cf, runner = _make_decomposer(
        runner_mapping={
            ("buy", 0): "click[back]",
            ("buy", 1): "click[back]",
        }
    )
    cf.decompose(group)

    # First call to runner = batched alt-action sample. Each prompt
    # corresponds to one turn position the CF runs on (T=3 turns ⇒ 3
    # prompts in the first batch). The third (buy stage) prompt should
    # carry the 'On product page' observation that comes AFTER the
    # search → click[item-0] prefix.
    first_call = runner.calls[0]
    prompts = first_call[0]
    assert any("On product page" in p for p in prompts), (
        f"Expected one of the alt prompts to be at the 'buy' stage; got prompts: {prompts}"
    )


# ---------------------------------------------------------------------------
# 3. Correct-turn credit
# ---------------------------------------------------------------------------


def test_correct_turn_has_largest_credit():
    """`click[item-0]` is the highest-credit action: replacing it with a
    wrong item drops R from 1.0 to 0.4. CF should produce the largest
    Δ at turn 1 (the click turn).
    """
    actions = ["search[laptop bag]", "click[item-0]", "click[buy]"]  # R=1.0
    traj = _run_scripted_trajectory(actions, task_id=0)
    group = TrajectoryGroup(task_id="0", env_name="webshop", trajectories=[traj])

    # Alt actions: at click stage, sample `click[item-2]` (wrong item)
    # for both alt slots. At search stage, alt search query still works.
    # At buy stage, alt is `click[buy]` (default greedy) — same as actual.
    cf, _ = _make_decomposer(
        runner_mapping={
            ("click", 0): "click[item-2]",
            ("click", 1): "click[item-2]",
        },
        n_alt_actions=2,
    )
    out = cf.decompose(group)
    deltas = out[0]
    # Turn 1 (click) should have the LARGEST delta because alt actions
    # there steer the rollout to R=0.4 (Δ ≈ 0.6) while the alt search
    # at turn 0 still completes with R=1.0 (Δ ≈ 0.0) and the alt buy at
    # turn 2 = same as actual (Δ ≈ 0.0).
    assert deltas[1] > deltas[0]
    assert deltas[1] > deltas[2]
    assert deltas[1] == pytest.approx(0.6, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Zero-R short-circuit
# ---------------------------------------------------------------------------


def test_zero_R_short_circuit_emits_zeros():
    """When `final_reward = 0` and skip_if_zero_R=True, no env steps are
    incurred and the per-turn list is `[0.0] * T`."""
    actions = ["search[laptop bag]", "click[item-0]", "click[buy]"]
    traj = _run_scripted_trajectory(
        actions, task_id=0, final_reward_override=0.0
    )
    group = TrajectoryGroup(task_id="0", env_name="webshop", trajectories=[traj])

    cf, runner = _make_decomposer(skip_if_zero_R=True)
    out = cf.decompose(group)
    assert out == [[0.0, 0.0, 0.0]]
    assert runner.calls == [], "no LLM calls should happen for R=0 trajectory"


# ---------------------------------------------------------------------------
# 5. Normalized output sums to R
# ---------------------------------------------------------------------------


def test_normalized_output_sums_to_R():
    actions = ["search[laptop bag]", "click[item-0]", "click[buy]"]
    traj = _run_scripted_trajectory(actions, task_id=0)
    group = TrajectoryGroup(task_id="0", env_name="webshop", trajectories=[traj])

    # Wrong-item alt at click ⇒ positive Δ at turn 1, ≤ 0 elsewhere.
    cf, _ = _make_decomposer(
        runner_mapping={
            ("click", 0): "click[item-2]",
            ("click", 1): "click[item-2]",
        },
        n_alt_actions=2,
        output_mode="normalized",
    )
    out = cf.decompose(group)
    assert sum(out[0]) == pytest.approx(traj.final_reward, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Normalized fallback to uniform
# ---------------------------------------------------------------------------


def test_normalized_falls_back_to_uniform_when_all_deltas_nonpositive():
    """If every alt action does AT LEAST as well as actual ⇒ all Δ ≤ 0
    ⇒ normalized mode falls back to uniform R/T per turn."""
    actions = ["search[laptop bag]", "click[item-0]", "click[buy]"]
    traj = _run_scripted_trajectory(actions, task_id=0)
    group = TrajectoryGroup(task_id="0", env_name="webshop", trajectories=[traj])

    # All alt actions match the original ⇒ deltas all ≤ 0 ⇒ uniform fallback.
    cf, _ = _make_decomposer(output_mode="normalized")
    out = cf.decompose(group)
    expected_share = traj.final_reward / 3
    assert all(x == pytest.approx(expected_share, abs=1e-6) for x in out[0])


# ---------------------------------------------------------------------------
# 7. Subsample turns
# ---------------------------------------------------------------------------


def test_n_turns_per_traj_subsamples():
    actions = ["search[laptop bag]", "click[item-0]", "click[buy]"]
    traj = _run_scripted_trajectory(actions, task_id=0)
    group = TrajectoryGroup(task_id="0", env_name="webshop", trajectories=[traj])

    cf, runner = _make_decomposer(
        runner_mapping={
            ("click", 0): "click[item-2]",
            ("click", 1): "click[item-2]",
        },
        n_alt_actions=2,
        n_turns_per_traj=1,  # only 1 turn per trajectory gets CF
        seed=0,  # seed=0 picks turn idx 1 (click stage) → delta = 0.6
    )
    out = cf.decompose(group)
    # Exactly ONE turn should have a non-zero delta; the other two stay 0.
    nonzero = [x for x in out[0] if x != 0.0]
    assert len(nonzero) == 1
    # That turn is turn 1 (click stage) per the seeded selection.
    assert out[0][1] == pytest.approx(0.6, abs=1e-6)
    # Total alt-prompts = n_turns_per_traj (one prompt per turn position
    # in the first batched call).
    first_call = runner.calls[0]
    assert len(first_call[0]) == 1


# ---------------------------------------------------------------------------
# 8. Empty trajectory
# ---------------------------------------------------------------------------


def test_empty_trajectory_returns_empty_list():
    empty = Trajectory(
        task_id="0", env_name="webshop", turns=[], final_reward=0.0
    )
    group = TrajectoryGroup(task_id="0", env_name="webshop", trajectories=[empty])
    cf, runner = _make_decomposer()
    out = cf.decompose(group)
    assert out == [[]]
    assert runner.calls == []


# ---------------------------------------------------------------------------
# 9. Factory guard
# ---------------------------------------------------------------------------


def test_factory_guard_rejects_missing_deps():
    """`build_decomposer(decomposer='counterfactual')` raises if any of
    runner/env_factory/prompt_renderer/action_parser/sampling_factory is
    missing."""
    cfg = {"hgpo": {"decomposer": "counterfactual"}}
    with pytest.raises(ValueError, match="counterfactual"):
        build_decomposer(cfg)
    # Providing only some deps still raises.
    with pytest.raises(ValueError, match="counterfactual"):
        build_decomposer(
            cfg,
            runner=_RecipeRunner({}),
            env_factory=lambda: FakeWebShopEnv(),
        )


def test_factory_with_all_deps_returns_callable():
    cfg = {
        "hgpo": {"decomposer": "counterfactual"},
        "counterfactual": {"n_alt_actions": 2, "max_completion_turns": 1},
    }
    dec = build_decomposer(
        cfg,
        runner=_RecipeRunner({}),
        env_factory=lambda: FakeWebShopEnv(max_steps=8),
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        sampling_factory=_Sampling,
    )
    assert callable(dec)
    assert isinstance(dec, CounterFactualDecomposer)


# ---------------------------------------------------------------------------
# 10. has_learnable_params
# ---------------------------------------------------------------------------


def test_has_learnable_params_is_false():
    cf, _ = _make_decomposer()
    assert cf.has_learnable_params is False


# ---------------------------------------------------------------------------
# 11. Env-pool reuse
# ---------------------------------------------------------------------------


def test_env_pool_reused_across_decompose_calls():
    """Pool grows on the first call; subsequent calls of equal-or-smaller
    size reuse the same env instances. Mirrors the optimisation in
    RolloutCollector + matches the docstring contract."""
    actions = ["search[laptop bag]", "click[item-0]", "click[buy]"]
    traj = _run_scripted_trajectory(actions, task_id=0)
    group = TrajectoryGroup(task_id="0", env_name="webshop", trajectories=[traj])

    built: list[FakeWebShopEnv] = []

    def factory() -> FakeWebShopEnv:
        e = FakeWebShopEnv(max_steps=8)
        built.append(e)
        return e

    runner = _RecipeRunner(
        {("click", 0): "click[item-2]", ("click", 1): "click[item-2]"}
    )
    cf = CounterFactualDecomposer(
        runner=runner,
        env_factory=factory,
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        sampling_factory=_Sampling,
        n_alt_actions=2,
        max_completion_turns=1,
        seed=0,
    )
    cf.decompose(group)
    n_first = len(built)
    cf.decompose(group)
    # Second call should not have built more envs (T=3 turns × N=2 alt
    # = 6 envs in the pool — same need on both calls).
    assert len(built) == n_first


# ---------------------------------------------------------------------------
# 12. max_env_pool_size cap (post-review hardening)
# ---------------------------------------------------------------------------


def test_env_pool_cap_falls_back_to_ephemeral_envs():
    """`max_env_pool_size` caps the resident pool. Calls needing more
    envs than the cap build the surplus ephemerally on each call \u2014
    correctness is preserved, the pool just doesn't grow past the cap.
    """
    actions = ["search[laptop bag]", "click[item-0]", "click[buy]"]
    traj = _run_scripted_trajectory(actions, task_id=0)
    group = TrajectoryGroup(task_id="0", env_name="webshop", trajectories=[traj])

    built: list[FakeWebShopEnv] = []

    def factory() -> FakeWebShopEnv:
        e = FakeWebShopEnv(max_steps=8)
        built.append(e)
        return e

    cf = CounterFactualDecomposer(
        runner=_RecipeRunner({}),
        env_factory=factory,
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        sampling_factory=_Sampling,
        n_alt_actions=2,
        max_completion_turns=1,
        max_env_pool_size=2,  # T=3 * N=2 = 6 needed; pool capped at 2
        seed=0,
    )
    cf.decompose(group)
    # Pool grew to its cap, surplus came from ephemeral factory calls.
    assert len(cf._env_pool) == 2
    n_after_first = len(built)
    cf.decompose(group)
    # Second call: pool 2 reused, but 4 ephemeral envs built fresh again.
    assert len(built) == n_after_first + 4


# ---------------------------------------------------------------------------
# 13. Prefix-reward accounting (post-review fix)
# ---------------------------------------------------------------------------


class _ShapingRewardEnv:
    """Toy env that emits a +0.1 shaping reward on EVERY step plus a
    +1.0 terminal bonus on the third step. Used to verify the CF
    decomposer accumulates prefix-replay rewards into the baseline so
    \u0394_t = R_actual - R_baseline_t is unbiased on shaping-reward envs.
    """

    class _State:
        observation_text = "shaping-env-state"
        valid_actions: list[str] = []
        instruction = ""

    def reset(self, task_id: int = 0) -> "_ShapingRewardEnv._State":
        self._steps = 0
        return self._State()

    def step(self, action: str):
        self._steps += 1
        done = self._steps >= 3
        # +0.1 shaping per step, +1.0 terminal bonus on the last step.
        reward = 0.1 + (1.0 if done else 0.0)
        return self._State(), reward, done, {}


def test_prefix_replay_rewards_accumulated_in_baseline():
    """On a shaping-reward env where every step yields +0.1 plus a +1.0
    terminal bonus, R_actual = 1.3. If the CF decomposer's baseline
    discarded prefix-replay rewards, \u0394_t at turn t=2 would be biased
    upward by 0.2 (the prefix shaping). The post-review fix accumulates
    prefix rewards into `unit['prefix_R']`, so deltas should be \u2248 0.0
    when alt actions reproduce the same episode return.
    """
    # Build a 3-turn trajectory with R_actual = 0.1 + 0.1 + 0.1 + 1.0 = 1.3.
    turns = [
        TurnRecord(turn_idx=t, observation_text="shaping-env-state", action_text="noop")
        for t in range(3)
    ]
    traj = Trajectory(
        task_id="0",
        env_name="shaping",
        turns=turns,
        final_reward=1.3,
    )
    group = TrajectoryGroup(task_id="0", env_name="shaping", trajectories=[traj])

    runner = _RecipeRunner({})  # default = "think[noop]" stage fallback
    cf = CounterFactualDecomposer(
        runner=runner,
        env_factory=_ShapingRewardEnv,
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        sampling_factory=_Sampling,
        n_alt_actions=1,
        max_completion_turns=3,
        skip_if_zero_R=False,
        seed=0,
    )
    out = cf.decompose(group)
    # Every alt rollout reaches the same episode return (R=1.3), so
    # \u0394_t \u2248 0 for all t. WITHOUT the prefix-reward fix, \u0394_t at
    # turn 1 would be ~0.1 (one prefix step's reward unaccounted-for)
    # and at turn 2 ~0.2 \u2014 the test would fail.
    for x in out[0]:
        assert abs(x) < 1e-6, (
            f"\u0394_t expected \u22480 with prefix accounting but got {x}; baseline "
            "is likely missing prefix-replay rewards (regression of the "
            "post-review fix)."
        )


# ---------------------------------------------------------------------------
# 14. check_state_consistency (post-review hardening)
# ---------------------------------------------------------------------------


class _NondeterministicEnv:
    """Env that returns a different observation_text on every reset call,
    simulating a stochastic env where deterministic replay is broken.
    """

    _counter = 0

    class _State:
        def __init__(self, obs: str) -> None:
            self.observation_text = obs
            self.valid_actions: list[str] = []
            self.instruction = ""

    def reset(self, task_id: int = 0):
        type(self)._counter += 1
        self._steps = 0
        return self._State(f"obs-call-{type(self)._counter}")

    def step(self, action: str):
        self._steps += 1
        done = self._steps >= 1
        return self._State(f"obs-step-{self._steps}"), 0.0, done, {}


def test_check_state_consistency_raises_on_state_divergence():
    """When `check_state_consistency=True` and the env is non-deterministic
    on replay (so two CF units at the same (i, t) reach different states),
    decompose() raises a clear RuntimeError instead of silently rendering
    only one state's prompt.
    """
    # Reset the counter so the test is deterministic across re-runs.
    _NondeterministicEnv._counter = 0
    turns = [
        TurnRecord(turn_idx=0, observation_text="obs", action_text="noop"),
    ]
    traj = Trajectory(
        task_id="0", env_name="nondet", turns=turns, final_reward=0.5
    )
    group = TrajectoryGroup(task_id="0", env_name="nondet", trajectories=[traj])

    cf = CounterFactualDecomposer(
        runner=_RecipeRunner({}),
        env_factory=_NondeterministicEnv,
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        sampling_factory=_Sampling,
        n_alt_actions=2,  # > 1 needed to compare states
        max_completion_turns=0,
        skip_if_zero_R=False,
        check_state_consistency=True,
        seed=0,
    )
    with pytest.raises(RuntimeError, match=r"check_state_consistency"):
        cf.decompose(group)


def test_check_state_consistency_default_off_does_not_raise():
    """The default (`check_state_consistency=False`) preserves the
    zero-overhead production fast-path \u2014 even with a non-deterministic
    env decompose() should NOT raise (it just silently renders one
    state's prompt for all alt units in the (i, t) bucket).
    """
    _NondeterministicEnv._counter = 0
    turns = [
        TurnRecord(turn_idx=0, observation_text="obs", action_text="noop"),
    ]
    traj = Trajectory(
        task_id="0", env_name="nondet", turns=turns, final_reward=0.5
    )
    group = TrajectoryGroup(task_id="0", env_name="nondet", trajectories=[traj])

    cf = CounterFactualDecomposer(
        runner=_RecipeRunner({}),
        env_factory=_NondeterministicEnv,
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        sampling_factory=_Sampling,
        n_alt_actions=2,
        max_completion_turns=0,
        skip_if_zero_R=False,
        # check_state_consistency defaults to False
        seed=0,
    )
    out = cf.decompose(group)
    # No exception; output shape is correct.
    assert len(out) == 1 and len(out[0]) == 1


def test_max_env_pool_size_validation():
    """`max_env_pool_size` must be \u2265 1 or None."""
    with pytest.raises(ValueError, match=r"max_env_pool_size"):
        CounterFactualDecomposer(
            runner=_RecipeRunner({}),
            env_factory=lambda: FakeWebShopEnv(),
            prompt_renderer=render_webshop_turn_prompt,
            action_parser=parse_react_action,
            sampling_factory=_Sampling,
            max_env_pool_size=0,
        )
