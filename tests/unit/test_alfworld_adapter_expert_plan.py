"""Unit tests for the ALFWorld adapter's expert-plan-length shaping signal.

Verifies the dense per-turn `info["intermediate_reward"]` field surfaced
by `ALFWorldAdapter.step()` — the producer-side data wired into
`TurnRDv2`'s V-head supervision target via the
`progress_signal` JSONL field. See
`/Users/shoupeili/.llms/plans/turnrd_dense_progress_alfworld.plan.md`.

Coverage:
1. Δ = max(0, prev_len - curr_len) for shrinking plans.
2. The clamp prevents negative deltas when the plan transiently grows
   (off-plan re-planning).
3. Missing/empty plans mid-trajectory degrade to 0 (no spurious shaping).
4. `_prev_plan_len` resets on every `reset()` call (no leak across episodes).
5. `extra.expert_plan` and `expert_plan` keys are both honored
   (TextWorld batched / non-batched info conventions).
"""
from __future__ import annotations

import unittest

from src.envs.alfworld_adapter import ALFWorldAdapter, _extract_expert_plan, _extract_facts_set


class _FakeALFWorldEnv:
    """Stub env whose `reset()` / `step()` return canned info dicts.

    Drive sequence is set per-test by mutating `step_infos`. `reset_info`
    holds the single info dict returned from `reset()`.
    """

    def __init__(
        self,
        reset_info: dict,
        step_infos: list[dict],
    ) -> None:
        self.reset_info = reset_info
        self.step_infos = list(step_infos)
        # The adapter's `_select_task` walks these attrs; we don't care
        # about deterministic indexing here so leave the list short.
        self.game_files = ["g0.tw"]
        self.next_game_idx = 0
        self._cursor = 0
        self.last_action: str | None = None

    def reset(self):
        return ([self.reset_info.get("_obs", "obs0")], dict(self.reset_info))

    def step(self, action: str):
        self.last_action = action
        info = self.step_infos[min(self._cursor, len(self.step_infos) - 1)]
        self._cursor += 1
        # Strip private control keys (`_reward`, `_done`) so the
        # returned info dict mirrors what real ALFWorld emits. Note we
        # deliberately preserve `intermediate_reward` (Phase 2 upstream
        # injection) so the reconciliation path can read it.
        public_info = {
            k: v for k, v in info.items() if not k.startswith("_")
        }
        return (
            {"state": f"after-{action}"},
            float(info.get("_reward", 0.0)),
            bool(info.get("_done", False)),
            public_info,
        )


class _TestableALFWorldAdapter(ALFWorldAdapter):
    def __init__(self, env: _FakeALFWorldEnv, **kw) -> None:
        self._fake_env = env
        super().__init__(**kw)

    def _build_alfworld_env(self):
        return self._fake_env


class _Phase2TestableAdapter(_TestableALFWorldAdapter):
    """Variant for Phase 2 tests: stubs `_build_request_infos` so the
    import-guard in `__init__` passes even when `textworld` isn't
    installed in the test environment. Tests still set
    `_tw_registration_succeeded` explicitly to control reconciliation.
    """

    def _build_request_infos(self):  # type: ignore[override]
        return "SENTINEL_ENVINFOS"


class TestExpertPlanExtraction(unittest.TestCase):
    """Direct unit tests on the helper — both keys + edge shapes."""

    def test_extracts_extra_dot_expert_plan(self) -> None:
        info = {"extra.expert_plan": ["look", "open fridge", "take apple"]}
        self.assertEqual(
            _extract_expert_plan(info), ["look", "open fridge", "take apple"]
        )

    def test_extracts_bare_expert_plan(self) -> None:
        info = {"expert_plan": ["go north", "pick up key"]}
        self.assertEqual(_extract_expert_plan(info), ["go north", "pick up key"])

    def test_peels_per_batch_outer_list(self) -> None:
        # TextWorld batch wrapper occasionally wraps the plan in a
        # per-batch list of length 1.
        info = {"extra.expert_plan": [["pick up apple", "go to fridge"]]}
        self.assertEqual(
            _extract_expert_plan(info), ["pick up apple", "go to fridge"]
        )

    def test_returns_empty_when_missing(self) -> None:
        self.assertEqual(_extract_expert_plan({}), [])

    def test_returns_empty_for_non_dict(self) -> None:
        self.assertEqual(_extract_expert_plan(None), [])
        self.assertEqual(_extract_expert_plan("not a dict"), [])


class TestIntermediateRewardWiring(unittest.TestCase):
    def test_step_emits_max_zero_delta_for_shrinking_plan(self) -> None:
        """Plan goes 5 → 4 → 3 → 1 → 0 (success).

        Per-step deltas: 1, 1, 2, 1.
        """
        reset_info = {"extra.expert_plan": ["a", "b", "c", "d", "e"]}
        step_infos = [
            {"extra.expert_plan": ["b", "c", "d", "e"], "_reward": 0.0},
            {"extra.expert_plan": ["c", "d", "e"], "_reward": 0.0},
            {"extra.expert_plan": ["e"], "_reward": 0.0},
            {"extra.expert_plan": [], "_reward": 1.0, "_done": True},
        ]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(env, max_steps=10)

        adapter.reset()
        deltas: list[float] = []
        for _ in range(4):
            _, _, done, info = adapter.step("noop")
            deltas.append(info["intermediate_reward"])
            if done:
                break
        self.assertEqual(deltas, [1.0, 1.0, 2.0, 1.0])

    def test_negative_delta_is_clamped_to_zero(self) -> None:
        """Off-plan action lengthens plan transiently (re-plan): 3 → 5.

        max(0, 3 - 5) = 0 — the clamp prevents spurious negative
        shaping.
        """
        reset_info = {"extra.expert_plan": ["a", "b", "c"]}
        step_infos = [{"extra.expert_plan": ["a", "b", "c", "d", "e"]}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(env, max_steps=4)

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 0.0)

    def test_missing_plan_mid_trajectory_yields_zero(self) -> None:
        """Plan available at reset, then disappears mid-trajectory.

        Adapter conservatively emits 0 when curr_plan_len is undefined.
        """
        reset_info = {"extra.expert_plan": ["a", "b"]}
        step_infos = [
            {},  # plan key missing entirely
            {"extra.expert_plan": []},  # explicitly empty
        ]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(env, max_steps=4)

        adapter.reset()
        # When the plan is missing the helper returns []; the adapter
        # treats this as `curr_plan_len = 0`. With prev_plan_len = 2
        # this yields delta = 2 (the helper can't distinguish "we
        # finished the plan" from "we lost visibility into the plan",
        # so we lean toward giving credit). The plan's edge-case
        # contract is to default to 0 ONLY when the prev tracker was
        # never primed; otherwise we trust the deltas. But if both
        # adjacent steps lack plans, the second step's delta = 0.
        _, _, _, info1 = adapter.step("noop")
        # First step: prev=2 (primed at reset), curr=0 → 2.
        self.assertEqual(info1["intermediate_reward"], 2.0)
        _, _, _, info2 = adapter.step("noop")
        # Second step: prev=0 (last seen), curr=0 → 0.
        self.assertEqual(info2["intermediate_reward"], 0.0)

    def test_no_plan_at_reset_first_step_delta_is_zero(self) -> None:
        """Env that never exposes an expert plan (e.g. non-handcoded fork).

        `_prev_plan_len` stays None at reset, and on the first step
        falls back to `curr_plan_len` so delta = 0 — no spurious
        shaping reward for the first step on plan-less envs.
        """
        reset_info = {}  # no plan keys at all
        step_infos = [{}, {}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(env, max_steps=4)

        adapter.reset()
        _, _, _, info1 = adapter.step("noop")
        _, _, _, info2 = adapter.step("noop")
        self.assertEqual(info1["intermediate_reward"], 0.0)
        self.assertEqual(info2["intermediate_reward"], 0.0)

    def test_prev_plan_len_resets_between_episodes(self) -> None:
        """Calling `reset()` again must re-prime `_prev_plan_len` from the
        new episode's initial info — no leak from the previous episode.
        """
        reset_info = {"extra.expert_plan": ["a", "b", "c"]}
        step_infos = [{"extra.expert_plan": ["b", "c"]}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(env, max_steps=4)

        adapter.reset()
        self.assertEqual(adapter._prev_plan_len, 3)
        adapter.step("noop")
        self.assertEqual(adapter._prev_plan_len, 2)

        # Re-prime the fake env for a fresh episode with a different
        # starting plan length.
        env.reset_info = {"extra.expert_plan": ["x", "y", "z", "w"]}
        env.step_infos = [{"extra.expert_plan": ["y", "z", "w"]}]
        env._cursor = 0

        adapter.reset()
        self.assertEqual(adapter._prev_plan_len, 4)
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 1.0)

    def test_intermediate_reward_present_when_plan_unavailable(self) -> None:
        """The field is ALWAYS surfaced (even when 0) so downstream code
        never has to KeyError-guard. Producer's collector reads
        `info.get("intermediate_reward")` so a present-but-zero value
        keeps the all-or-nothing emission gate well-defined."""
        reset_info = {}
        step_infos = [{}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(env, max_steps=2)

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertIn("intermediate_reward", info)
        self.assertIsInstance(info["intermediate_reward"], float)


class TestPhase2TextWorldReconciliation(unittest.TestCase):
    """Phase 2: TextWorld native `intermediate_reward` reconciliation.

    See `~/.llms/plans/turnrd_textworld_intermediate_reward_phase2.plan.md`.
    """

    def test_optin_off_preserves_phase1_behavior(self) -> None:
        """Default opt-in False ⇒ source tag is `"expert_plan"` and
        the value matches the Phase 1 plan-length delta."""
        reset_info = {"extra.expert_plan": ["a", "b", "c"]}
        step_infos = [{"extra.expert_plan": ["b", "c"]}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(
            env, max_steps=4, use_textworld_intermediate_reward=False
        )

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 1.0)
        self.assertEqual(info["intermediate_reward_source"], "expert_plan")

    def test_optin_on_with_upstream_value_prefers_textworld(self) -> None:
        """Upstream populates `info["intermediate_reward"]` ⇒ we use it
        (after `max(0.0, ·)` clamp) and tag source `"textworld"`.
        """
        reset_info = {"extra.expert_plan": ["a", "b", "c"]}
        # Step shrinks plan by 1 (Phase 1 ⇒ 1.0) but injects upstream
        # ir = 3 (e.g. one action satisfied 3 PDDL fluents). The
        # reconciler should prefer the upstream 3.0.
        step_infos = [{"extra.expert_plan": ["b", "c"], "intermediate_reward": 3}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _Phase2TestableAdapter(
            env, max_steps=4, use_textworld_intermediate_reward=True
        )
        # Test fakes skip `_wrap_batch_env` (the env is already gym-shaped),
        # so registration never fires — manually mark it as succeeded.
        adapter._tw_registration_succeeded = True

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 3.0)
        self.assertEqual(info["intermediate_reward_source"], "textworld")

    def test_optin_on_with_upstream_missing_falls_back_to_expert_plan(self) -> None:
        """Upstream key absent ⇒ fall through to Phase 1 delta and tag
        `"expert_plan"`. The fallback path stays warm.
        """
        reset_info = {"extra.expert_plan": ["a", "b", "c"]}
        step_infos = [{"extra.expert_plan": ["b", "c"]}]  # no `intermediate_reward`
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _Phase2TestableAdapter(
            env, max_steps=4, use_textworld_intermediate_reward=True
        )
        adapter._tw_registration_succeeded = True

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 1.0)
        self.assertEqual(info["intermediate_reward_source"], "expert_plan")

    def test_optin_on_with_negative_upstream_clamps_to_zero(self) -> None:
        """TextWorld `intermediate_reward` can be negative when fluents
        revert; the V-head expects non-negative ⇒ clamp to 0 and keep
        source tag `"textworld"` (we DID consume the upstream signal).
        """
        reset_info = {"extra.expert_plan": ["a", "b"]}
        step_infos = [{"extra.expert_plan": ["a", "b"], "intermediate_reward": -2}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _Phase2TestableAdapter(
            env, max_steps=4, use_textworld_intermediate_reward=True
        )
        adapter._tw_registration_succeeded = True

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 0.0)
        self.assertEqual(info["intermediate_reward_source"], "textworld")

    def test_optin_on_with_registration_failed_uses_expert_plan(self) -> None:
        """Registration silently fell to Tier 3 ⇒
        `_tw_registration_succeeded == False` ⇒ always use Phase 1
        delta even when upstream injects a value (because we can't
        trust it).
        """
        reset_info = {"extra.expert_plan": ["a", "b", "c"]}
        step_infos = [{"extra.expert_plan": ["b", "c"], "intermediate_reward": 99}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _Phase2TestableAdapter(
            env, max_steps=4, use_textworld_intermediate_reward=True
        )
        adapter._tw_registration_succeeded = False  # explicit Tier-3 simulation

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 1.0)
        self.assertEqual(info["intermediate_reward_source"], "expert_plan")

    def test_optin_on_without_textworld_raises_at_construction(self) -> None:
        """If `use_textworld_intermediate_reward=True` but
        `_build_request_infos()` returns None (textworld not
        importable), `__init__` must fail loudly.
        """

        class _NoTextworldAdapter(_TestableALFWorldAdapter):
            def _build_request_infos(self):  # type: ignore[override]
                return None

        env = _FakeALFWorldEnv({}, [{}])
        with self.assertRaises(ImportError) as ctx:
            _NoTextworldAdapter(
                env, max_steps=4, use_textworld_intermediate_reward=True
            )
        self.assertIn("textworld", str(ctx.exception))

    def test_optin_off_without_textworld_does_not_raise(self) -> None:
        """Default opt-in False ⇒ missing `textworld` is a non-issue.
        Phase 1 callers (SFT generator, evaluator) keep working.
        """

        class _NoTextworldAdapter(_TestableALFWorldAdapter):
            def _build_request_infos(self):  # type: ignore[override]
                return None

        env = _FakeALFWorldEnv({"extra.expert_plan": ["a"]}, [{}])
        adapter = _NoTextworldAdapter(
            env, max_steps=4, use_textworld_intermediate_reward=False
        )
        self.assertFalse(adapter._tw_registration_succeeded)
        self.assertFalse(adapter._use_tw_intermediate_reward)

    def test_optin_on_with_batched_list_wrapped_upstream_value_is_unwrapped(self) -> None:
        """Regression: TextWorld batched env emits scalar info values
        wrapped as length-1 lists per batch slot (e.g. `[3]` for
        batch_size=1). `_unbatch_info` intentionally only unwraps when
        the inner element is list/tuple/dict, so the upstream IR
        scalar arrives wrapped. The reconciler must peel that wrapping
        before the type check or it silently falls back to Phase 1.
        """
        reset_info = {"extra.expert_plan": ["a", "b", "c"]}
        # Inject upstream as a length-1 list — the shape that survives
        # `_unbatch_info` in the production batched code path.
        step_infos = [{"extra.expert_plan": ["b", "c"], "intermediate_reward": [3]}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _Phase2TestableAdapter(
            env, max_steps=4, use_textworld_intermediate_reward=True
        )
        adapter._tw_registration_succeeded = True

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 3.0)
        self.assertEqual(info["intermediate_reward_source"], "textworld")

    def test_prev_plan_len_updated_even_when_upstream_fires(self) -> None:
        """The Phase 1 fallback tracker must stay warm so the adapter
        degrades gracefully if upstream goes None mid-trajectory.
        """
        reset_info = {"extra.expert_plan": ["a", "b", "c"]}
        step_infos = [
            {"extra.expert_plan": ["b", "c"], "intermediate_reward": 5},
            # Step 2: upstream missing ⇒ fallback should compute
            # delta from prev_plan_len=2 (updated last step) ⇒ 1.
            {"extra.expert_plan": ["c"]},
        ]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _Phase2TestableAdapter(
            env, max_steps=4, use_textworld_intermediate_reward=True
        )
        adapter._tw_registration_succeeded = True

        adapter.reset()
        _, _, _, info1 = adapter.step("noop")
        self.assertEqual(info1["intermediate_reward"], 5.0)
        self.assertEqual(info1["intermediate_reward_source"], "textworld")
        self.assertEqual(adapter._prev_plan_len, 2)
        _, _, _, info2 = adapter.step("noop")
        self.assertEqual(info2["intermediate_reward"], 1.0)
        self.assertEqual(info2["intermediate_reward_source"], "expert_plan")


class TestFactsDiffSignal(unittest.TestCase):
    """Phase 3: PDDL-facts-diff per-turn signal reconciliation.

    See `~/.llms/plans/turnrd_facts_diff_intermediate_reward.plan.md`.
    """

    def test_facts_diff_normal_growing(self) -> None:
        """Facts go 3 -> 5 ⇒ delta=2, source="facts_diff"."""
        reset_info = {"facts": ["f1", "f2", "f3"]}
        step_infos = [{"facts": ["f1", "f2", "f3", "f4", "f5"]}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(
            env, max_steps=4, use_facts_diff_intermediate_reward=True
        )

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 2.0)
        self.assertEqual(info["intermediate_reward_source"], "facts_diff")

    def test_facts_diff_with_reverted_fact(self) -> None:
        """5 -> 5 facts where 1 new + 1 removed ⇒ net delta=0."""
        reset_info = {"facts": ["f1", "f2", "f3", "f4", "f5"]}
        # Drop f5, add f6 → |new|=1, |removed|=1 → max(0, 0)=0.
        step_infos = [{"facts": ["f1", "f2", "f3", "f4", "f6"]}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(
            env, max_steps=4, use_facts_diff_intermediate_reward=True
        )

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 0.0)
        self.assertEqual(info["intermediate_reward_source"], "facts_diff")

    def test_facts_diff_negative_clamped(self) -> None:
        """5 -> 3 facts (2 removed, 0 added) ⇒ max(0, 0-2)=0."""
        reset_info = {"facts": ["f1", "f2", "f3", "f4", "f5"]}
        step_infos = [{"facts": ["f1", "f2", "f3"]}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(
            env, max_steps=4, use_facts_diff_intermediate_reward=True
        )

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 0.0)
        self.assertEqual(info["intermediate_reward_source"], "facts_diff")

    def test_facts_diff_initial_state(self) -> None:
        """No facts at reset ⇒ first-step facts_delta is None (don't
        reward "everything new" spuriously). Source falls through to
        "expert_plan" path which yields 0 (no plan either).
        """
        reset_info = {}  # no facts
        step_infos = [{"facts": ["f1", "f2", "f3"]}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(
            env, max_steps=4, use_facts_diff_intermediate_reward=True
        )

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 0.0)
        self.assertEqual(info["intermediate_reward_source"], "expert_plan")

    def test_facts_diff_priority_below_textworld(self) -> None:
        """When BOTH flags on AND TextWorld value present ⇒
        source="textworld" (TW priority 1 > facts_diff priority 2).
        """
        reset_info = {"facts": ["f1", "f2", "f3"]}
        step_infos = [
            {
                "facts": ["f1", "f2", "f3", "f4", "f5"],
                "intermediate_reward": 7,
            }
        ]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _Phase2TestableAdapter(
            env,
            max_steps=4,
            use_textworld_intermediate_reward=True,
            use_facts_diff_intermediate_reward=True,
        )
        adapter._tw_registration_succeeded = True

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 7.0)
        self.assertEqual(info["intermediate_reward_source"], "textworld")

    def test_facts_diff_priority_above_expert_plan(self) -> None:
        """Only facts-diff flag on AND no TextWorld ⇒ source="facts_diff",
        value matches the facts diff (NOT the plan-delta) even when an
        expert_plan is present.
        """
        reset_info = {
            "facts": ["f1", "f2", "f3"],
            "extra.expert_plan": ["a", "b", "c", "d", "e"],
        }
        # Facts grow by 2; plan shrinks by 1. Facts-diff (2) wins.
        step_infos = [
            {
                "facts": ["f1", "f2", "f3", "f4", "f5"],
                "extra.expert_plan": ["b", "c", "d", "e"],
            }
        ]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(
            env, max_steps=4, use_facts_diff_intermediate_reward=True
        )

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 2.0)
        self.assertEqual(info["intermediate_reward_source"], "facts_diff")

    def test_facts_diff_optin_off_preserves_phase1(self) -> None:
        """Flag default False ⇒ source="expert_plan", value matches
        plan-delta (Phase 1 byte-for-byte preserved).
        """
        reset_info = {
            "facts": ["f1", "f2", "f3"],
            "extra.expert_plan": ["a", "b", "c"],
        }
        step_infos = [
            {
                "facts": ["f1", "f2", "f3", "f4", "f5"],
                "extra.expert_plan": ["b", "c"],
            }
        ]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(env, max_steps=4)

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 1.0)
        self.assertEqual(info["intermediate_reward_source"], "expert_plan")

    def test_facts_diff_handles_per_batch_list_wrap(self) -> None:
        """`info["facts"] = [[fact1, fact2, fact3]]` (per-batch wrap from
        TextWorld batched env) ⇒ unwraps the outer list and diffs the
        inner list correctly.
        """
        # Per-batch wrap: outer length-1 list with the actual list of
        # propositions inside.
        reset_info = {"facts": [["f1", "f2"]]}
        step_infos = [{"facts": [["f1", "f2", "f3", "f4"]]}]
        env = _FakeALFWorldEnv(reset_info, step_infos)
        adapter = _TestableALFWorldAdapter(
            env, max_steps=4, use_facts_diff_intermediate_reward=True
        )

        adapter.reset()
        _, _, _, info = adapter.step("noop")
        self.assertEqual(info["intermediate_reward"], 2.0)
        self.assertEqual(info["intermediate_reward_source"], "facts_diff")


class TestFactsExtraction(unittest.TestCase):
    """Direct unit tests on the `_extract_facts_set` helper."""

    def test_extracts_simple_list(self) -> None:
        info = {"facts": ["f1", "f2", "f3"]}
        self.assertEqual(_extract_facts_set(info), {"'f1'", "'f2'", "'f3'"})

    def test_returns_empty_when_missing(self) -> None:
        self.assertEqual(_extract_facts_set({}), set())

    def test_returns_empty_for_non_dict(self) -> None:
        self.assertEqual(_extract_facts_set(None), set())
        self.assertEqual(_extract_facts_set("not a dict"), set())

    def test_returns_empty_for_empty_list(self) -> None:
        self.assertEqual(_extract_facts_set({"facts": []}), set())

    def test_peels_per_batch_outer_list(self) -> None:
        info = {"facts": [["f1", "f2"]]}
        self.assertEqual(_extract_facts_set(info), {"'f1'", "'f2'"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
