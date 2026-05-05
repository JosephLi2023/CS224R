"""Unit tests for RolloutCollector + FakeWebShopEnv + ReAct prompts."""
from __future__ import annotations
from dataclasses import dataclass

from src.algorithms.grpo.collectors import RolloutCollector, RolloutCollectorConfig
from src.algorithms.grpo.rollout import TrajectoryGroup, TurnRecord
from src.envs.fake_webshop import FakeWebShopEnv
from src.envs.prompts.react_webshop import parse_react_action, render_webshop_turn_prompt


@dataclass
class _FakeGen:
    text: str
    token_ids: tuple
    token_logprobs: tuple
    prompt_token_count: int
    prompt_token_ids: tuple = ()
    finish_reason: str = "stop"


class _FakeRunner:
    def __init__(self, recipe, ptc=100):
        self.recipe = list(recipe)
        self.ptc = ptc
        self.calls = []
        self._cur = 0

    def generate_rich(self, prompts, sampling):
        n = getattr(sampling, "n", 1)
        self.calls.append((list(prompts), n))
        # One recipe entry per CALL (i.e. per turn), broadcast to all prompts.
        # This matches the collector's batched-per-turn pattern: at turn t, all
        # live envs ask for an action, and the test wants all of them to
        # receive the same scripted action.
        t = self.recipe[self._cur % len(self.recipe)]
        self._cur += 1
        ids = tuple(range(10, 10 + max(1, len(t.split()))))
        lps = tuple(-0.1 * (k + 1) for k in range(len(ids)))
        out = []
        for _ in prompts:
            row = [_FakeGen(t, ids, lps, self.ptc) for _ in range(n)]
            out.append(row)
        return out


def _coll(runner, max_turns=10, soft_budget=3500):
    return RolloutCollector(
        runner=runner,
        env_factory=lambda: FakeWebShopEnv(max_steps=8),
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        cfg=RolloutCollectorConfig(max_turns=max_turns, soft_prompt_token_budget=soft_budget),
    )


@dataclass
class _S:
    n: int = 1
    temperature: float = 1.0


def test_returns_K_trajectories():
    r = _FakeRunner(["Action: search[bag]", "Action: click[item-0]", "Action: click[buy]"])
    g, s = _coll(r).collect_group(task_id=0, env_name="webshop", K=4, sampling=_S())
    assert isinstance(g, TrajectoryGroup)
    assert g.K == 4 and g.task_id == "0"
    assert s.K == 4


def test_preserves_token_ids_and_logprobs():
    r = _FakeRunner(["Action: search[m]", "Action: click[item-0]", "Action: click[buy]"])
    g, _ = _coll(r).collect_group(task_id=1, env_name="webshop", K=2, sampling=_S())
    for traj in g.trajectories:
        for t in traj.turns:
            assert isinstance(t, TurnRecord)
            assert len(t.action_token_ids) > 0
            assert len(t.action_token_logprobs) == len(t.action_token_ids)
            assert all(lp <= 0 for lp in t.action_token_logprobs)
            assert t.prompt_token_count == 100


def test_correct_buy_full_reward():
    r = _FakeRunner(["Action: search[bag]", "Action: click[item-0]", "Action: click[buy]"])
    g, s = _coll(r).collect_group(task_id=0, env_name="webshop", K=3, sampling=_S())
    assert s.completed == 3 and s.truncated == 0
    assert all(t.final_reward == 1.0 for t in g.trajectories)
    assert all(len(t.turns) == 3 for t in g.trajectories)


def test_wrong_item_partial_credit():
    r = _FakeRunner(["Action: search[bag]", "Action: click[item-2]", "Action: click[buy]"])
    g, _ = _coll(r).collect_group(task_id=0, env_name="webshop", K=1, sampling=_S())
    assert g.trajectories[0].final_reward == 0.4


def test_truncates_at_max_turns():
    r = _FakeRunner(["Action: think[stuck]"])
    g, s = _coll(r, max_turns=3).collect_group(task_id=2, env_name="webshop", K=2, sampling=_S())
    assert s.truncated == 2 and s.completed == 0
    assert all(len(t.turns) == 3 for t in g.trajectories)


def test_forces_n_to_1():
    r = _FakeRunner(["Action: think[ok]"])
    _coll(r, max_turns=1).collect_group(task_id=0, env_name="webshop", K=2, sampling=_S(n=4))
    assert r.calls and r.calls[0][1] == 1


def test_one_batched_call_per_turn():
    r = _FakeRunner(["Action: search[x]", "Action: click[item-0]", "Action: click[buy]"])
    _coll(r, max_turns=5).collect_group(task_id=0, env_name="webshop", K=3, sampling=_S())
    assert len(r.calls) == 3
    assert all(len(p) == 3 for p, _ in r.calls)


def test_completed_envs_drop_from_later_batches():
    r = _FakeRunner(["Action: search[bag]", "Action: click[item-0]", "Action: click[buy]", "Action: think[x]"])
    _coll(r, max_turns=5).collect_group(task_id=0, env_name="webshop", K=2, sampling=_S())
    assert len(r.calls) == 3


def test_over_budget_bumps_stats():
    r = _FakeRunner(["Action: search[x]", "Action: click[item-0]", "Action: click[buy]"], ptc=5000)
    _, s = _coll(r, soft_budget=1000).collect_group(task_id=0, env_name="webshop", K=2, sampling=_S())
    assert s.over_budget_count >= 6


def test_react_parser_extracts_action_line():
    raw = "Thought: I should search.\nAction: search[laptop bag]\n"
    assert parse_react_action(raw) == "search[laptop bag]"


def test_react_parser_falls_back_to_first_nonempty_line():
    raw = "search[mouse]\nfollow-up text"
    assert parse_react_action(raw) == "search[mouse]"


def test_render_includes_observation_and_valid_actions():
    env = FakeWebShopEnv()
    state = env.reset(task_id=0)
    prompt = render_webshop_turn_prompt(state, [])
    assert "search[laptop bag]" in prompt
    assert "Thought:" in prompt
    assert state.observation_text in prompt


def test_env_pool_is_reused_across_collect_group_calls():
    """M2 regression: with reuse_envs=True (default), the same env instances
    are reused on subsequent collect_group calls."""
    r = _FakeRunner(["Action: search[bag]", "Action: click[item-0]", "Action: click[buy]"])
    built: list[FakeWebShopEnv] = []

    def factory():
        e = FakeWebShopEnv(max_steps=8)
        built.append(e)
        return e

    coll = RolloutCollector(
        runner=r,
        env_factory=factory,
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        cfg=RolloutCollectorConfig(max_turns=5),
    )
    coll.collect_group(task_id=0, env_name="webshop", K=3, sampling=_S())
    assert len(built) == 3, "first call should build K=3 envs"
    coll.collect_group(task_id=1, env_name="webshop", K=3, sampling=_S())
    assert len(built) == 3, "second call should reuse, NOT build new envs"


def test_env_pool_grows_when_K_increases():
    r = _FakeRunner(["Action: search[bag]", "Action: click[item-0]", "Action: click[buy]"])
    built: list[FakeWebShopEnv] = []
    coll = RolloutCollector(
        runner=r,
        env_factory=lambda: (built.append(FakeWebShopEnv(max_steps=8)), built[-1])[1],
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        cfg=RolloutCollectorConfig(max_turns=5),
    )
    coll.collect_group(task_id=0, env_name="webshop", K=2, sampling=_S())
    coll.collect_group(task_id=1, env_name="webshop", K=4, sampling=_S())
    assert len(built) == 4, "pool should grow from 2 → 4 on the larger call"


def test_reuse_envs_off_builds_fresh_each_call():
    r = _FakeRunner(["Action: search[bag]", "Action: click[item-0]", "Action: click[buy]"])
    built: list[FakeWebShopEnv] = []
    coll = RolloutCollector(
        runner=r,
        env_factory=lambda: (built.append(FakeWebShopEnv(max_steps=8)), built[-1])[1],
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        cfg=RolloutCollectorConfig(max_turns=5),
        reuse_envs=False,
    )
    coll.collect_group(task_id=0, env_name="webshop", K=2, sampling=_S())
    coll.collect_group(task_id=1, env_name="webshop", K=2, sampling=_S())
    assert len(built) == 4, "with reuse_envs=False each call rebuilds K envs"


# ---------------------------------------------------------------------------
# Defensive: empty-completion-list handling (post-mortem from the
# `eval ep=32 task=10032 CRASHED: IndexError('list index out of range')`
# Modal-side report). vLLM with greedy sampling can return [] for some
# prompts; collect_group must NOT raise IndexError on this.
# ---------------------------------------------------------------------------


class _EmptyOutRunner:
    """Returns an empty completion list for every prompt — simulates
    vLLM greedy sampling immediately predicting EOS (T=0 + EOS top-1).

    `out` shape from a healthy runner is `list[K_prompts] of list[n] of Gen`.
    Here we deliberately return `[[], [], ...]` to exercise the defensive
    guard that wraps `outs[j][0]`.
    """

    def generate_rich(self, prompts, sampling):
        return [[] for _ in prompts]


def test_collect_group_handles_empty_completion_list():
    """K=1 greedy + empty vLLM output → no IndexError; turn recorded with
    empty action_text + zero tokens; env still gets stepped."""
    coll = _coll(_EmptyOutRunner(), max_turns=2)
    g, s = coll.collect_group(task_id=99, env_name="webshop", K=1, sampling=_S())
    assert isinstance(g, TrajectoryGroup)
    assert g.K == 1
    # The collector must have emitted at least one turn (with empty
    # action) before the env declared done OR the max_turns budget.
    traj = g.trajectories[0]
    assert len(traj.turns) >= 1
    for t in traj.turns:
        assert t.action_text == ""  # parser fallback for empty text
        assert t.action_token_ids == ()
    # CollectStats should record the empty-output count for diagnostics.
    assert getattr(s, "empty_outputs", 0) >= 1


class _MixedOutRunner:
    """Returns valid output for the first prompt, empty for the second.
    Exercises the per-prompt defensive guard inside the live_idx loop."""

    def generate_rich(self, prompts, sampling):
        out = []
        for idx, _ in enumerate(prompts):
            if idx == 0:
                out.append([_FakeGen("Action: search[bag]", (10, 11), (-0.1, -0.2), 100)])
            else:
                out.append([])
        return out


def test_collect_group_handles_partial_empty_in_batch():
    """K=2: prompt 0 gets a valid completion, prompt 1 gets []. Both
    trajectories should still progress; only #1's actions are empty."""
    coll = _coll(_MixedOutRunner(), max_turns=1)
    g, s = coll.collect_group(task_id=0, env_name="webshop", K=2, sampling=_S())
    assert g.K == 2
    assert len(g.trajectories[0].turns) >= 1
    assert g.trajectories[0].turns[0].action_text != ""
    assert len(g.trajectories[1].turns) >= 1
    assert g.trajectories[1].turns[0].action_text == ""
