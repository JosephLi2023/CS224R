"""Unit tests for the WebShop adapter's attribute-progress dense signal.

Verifies the new per-step `info["intermediate_reward"]` field surfaced
by `WebShopAdapter.step()` when
`use_attribute_progress_intermediate_reward=True` — the producer-side
data wired into `TurnRDv2`'s V-head supervision target via the
`progress_signal` JSONL field. See
`/Users/shoupeili/.llms/plans/webshop_sota_recipe_transplant_dense_signal.plan.md`.

Coverage:
1. Flag default (False) ⇒ NO `intermediate_reward` key on any step
   (Phase 0 byte-for-byte preserved — protects existing WebShop runs).
2. Flag True + populated `env.goal` ⇒ per-step delta matches the
   fraction-of-target-attrs-newly-engaged formula, with
   `intermediate_reward_source == "attr_progress"` on every step.
3. The delta is clamped to ≥0 (un-clicking a previously-engaged option
   should not produce a negative shaping signal).
4. The +0.25 ASIN-landing bonus fires exactly once per episode, on the
   first `click[<target_asin>]`.
5. Per-trajectory state resets on every `reset()` (no leak across
   episodes: `_prev_overlap` zeroed, `_asin_bonus_fired` cleared,
   `_target_attrs` re-snapshotted).
6. Missing-goal-payload fallback: opt-in on but upstream exposes no
   recognisable goal ⇒ silent no-emit + RuntimeWarning emitted once.
7. Direct unit tests on the `_extract_target_attrs` /
   `_extract_target_asin` / `_extract_selected_attrs` helpers — both
   `env.goal` and `env.server.goals[session]` access paths.
"""
from __future__ import annotations

import unittest
import warnings

from src.envs.webshop_adapter import (
    WebShopAdapter,
    _extract_selected_attrs,
    _extract_target_asin,
    _extract_target_attrs,
)


class _FakeWebShopEnv:
    """Stub WebShop env with mutable `goal` + `cur_options` state.

    `reset()` returns a deterministic initial (obs, info) pair. `step()`
    pops a script of dicts; each script entry may carry a
    `_cur_options` key whose value is installed onto the env BEFORE the
    step() return — that's how individual tests drive the attribute-
    progress trajectory.
    """

    def __init__(
        self,
        goal: dict | None = None,
        step_script: list[dict] | None = None,
    ) -> None:
        self.goal = goal if goal is not None else {}
        self.session = 0
        self.cur_options: dict = {}
        self._step_script = list(step_script or [])
        self._cursor = 0
        self.last_action: str | None = None

    def reset(self, **kwargs):
        # Respect a `session` kwarg purely so the adapter's task_id →
        # session mapping flows through; the goal is fixed per-test.
        self.cur_options = {}
        self._cursor = 0
        return (
            {"obs": "initial page"},
            {"valid_actions": ["search[red shirt]", "click[B000TEST01]"]},
        )

    def step(self, action: str):
        self.last_action = action
        if self._cursor < len(self._step_script):
            entry = self._step_script[self._cursor]
        else:
            entry = {}
        self._cursor += 1
        # Drive the upstream env's cur_options BEFORE returning so the
        # adapter's step() sees the post-action state.
        if "_cur_options" in entry:
            self.cur_options = dict(entry["_cur_options"])
        info = {
            "valid_actions": entry.get(
                "valid_actions", ["click[Buy Now]", "click[Back to Search]"]
            ),
        }
        return (
            {"text": f"after {action}"},
            float(entry.get("_reward", 0.0)),
            bool(entry.get("_done", False)),
            info,
        )


class _TestableWebShopAdapter(WebShopAdapter):
    """WebShopAdapter that injects a `_FakeWebShopEnv` instead of trying
    to import the real upstream `web_agent_site` package (which isn't
    installed in the test container).
    """

    def __init__(self, fake_env: _FakeWebShopEnv, **kwargs) -> None:
        self._fake_env = fake_env
        super().__init__(**kwargs)

    def _build_webshop_env(self):  # type: ignore[override]
        return self._fake_env


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


class TestExtractHelpers(unittest.TestCase):
    def test_extract_target_attrs_from_env_goal(self) -> None:
        env = _FakeWebShopEnv(
            goal={
                "attributes": ["red", "Cotton"],
                "category": "shirts",
                "price_upper": 30,
            }
        )
        out = _extract_target_attrs(env)
        # Canonicalised: lowercased + trimmed.
        self.assertEqual(
            out, {"red", "cotton", "shirts", "price_under_30"}
        )

    def test_extract_target_attrs_includes_goal_options(self) -> None:
        """Selector option values (goal_options) must be in `target` so the
        attr-progress delta fires on item-page option clicks.

        Regression test for the dense-signal-quality fix landed after the
        `validate_dense_signal` smoke showed mean_ir_by_action_kind=
        {click_option: 0.0} — `goal["attributes"]` carries description
        tags ("easy care") while `env.cur_options.values()` carries
        selector values ("mint", "7"); without merging goal_options into
        the target set, |selected ∩ target| was identically 0 on every
        option-engagement turn and the signal collapsed to a 1-bit
        per-episode ASIN-landing indicator (useless for TurnRDv2 V-head
        supervision). See infra/app_webshop_sft_gen.py::validate_dense_signal."""
        env = _FakeWebShopEnv(
            goal={
                "attributes": ["easy care", "stretch fabric"],
                "category": "shirts",
                "price_upper": 30,
                "goal_options": {"color": "Mint", "size": "7"},
            }
        )
        out = _extract_target_attrs(env)
        self.assertEqual(
            out,
            {
                "easy care", "stretch fabric",
                "shirts", "price_under_30",
                # ← these are the goal_options values, now in the
                # target set so click[mint] / click[7] produce IR > 0
                "mint", "7",
            },
        )

    def test_extract_target_attrs_from_server_goals_dict(self) -> None:
        class _ServerStub:
            def __init__(self):
                self.goals = {7: {"attributes": ["blue"]}}

        class _EnvStub:
            def __init__(self):
                self.server = _ServerStub()
                self.session = 7
                self.goal = None  # force fallback

        out = _extract_target_attrs(_EnvStub())
        self.assertEqual(out, {"blue"})

    def test_extract_target_attrs_empty_on_missing_goal(self) -> None:
        env = _FakeWebShopEnv(goal=None)
        out = _extract_target_attrs(env)
        self.assertEqual(out, set())

    def test_extract_target_asin_from_goal(self) -> None:
        env = _FakeWebShopEnv(goal={"asin": "B000TEST01"})
        # Lowercased.
        self.assertEqual(_extract_target_asin(env), "b000test01")

    def test_extract_target_asin_returns_none_when_absent(self) -> None:
        env = _FakeWebShopEnv(goal={"attributes": ["red"]})
        self.assertIsNone(_extract_target_asin(env))

    def test_extract_selected_attrs_from_cur_options(self) -> None:
        env = _FakeWebShopEnv()
        env.cur_options = {"color": "Red", "size": "M"}
        out = _extract_selected_attrs(env)
        self.assertEqual(out, {"red", "m"})

    def test_extract_selected_attrs_from_user_sessions(self) -> None:
        """Canonical upstream path: env.user_sessions[session]["options"].

        Regression test for the dense-signal-quality fix. The
        princeton-nlp/WebShop env stores selected option clicks at
        `env.user_sessions[session_id]["options"]` (per
        `web_agent_text_env.py:410`), NOT at `env.cur_options` (which
        doesn't exist on the upstream env). Without this lookup path
        the IR signal degenerates to a 1-bit ASIN-landing indicator
        and TurnRDv2's V-head sees no per-turn credit; the live smoke
        `infra/app_webshop_sft_gen.py::validate_dense_signal` reported
        mean_ir_by_action_kind[click_option]=0.0 across 39 turns
        before this lookup was added.
        """
        class _UpstreamishEnv:
            session = "sess_abc"
            user_sessions = {
                "sess_abc": {"options": {"color": "Mint", "size": "7"}},
            }
        out = _extract_selected_attrs(_UpstreamishEnv())
        self.assertEqual(out, {"mint", "7"})

    def test_extract_selected_attrs_from_server_user_sessions(self) -> None:
        """Some forks split env vs server — try `env.server.user_sessions`."""
        class _ServerStub:
            user_sessions = {
                42: {"options": {"color": "Dark Blue"}},
            }
        class _EnvStub:
            session = 42
            server = _ServerStub()
        out = _extract_selected_attrs(_EnvStub())
        self.assertEqual(out, {"dark blue"})

    def test_extract_selected_attrs_empty_when_no_state_anywhere(self) -> None:
        """No cur_options, no user_sessions → empty set (safe fallback)."""
        class _BareEnv:
            session = None
        self.assertEqual(_extract_selected_attrs(_BareEnv()), set())

    def test_extract_selected_attrs_prefers_cur_options_when_both_present(self) -> None:
        """Tier-1 (cur_options) wins over tier-2 (user_sessions)."""
        class _DualEnv:
            session = "s"
            cur_options = {"color": "red"}
            user_sessions = {"s": {"options": {"color": "blue"}}}
        out = _extract_selected_attrs(_DualEnv())
        self.assertEqual(out, {"red"})


# ---------------------------------------------------------------------------
# Adapter-level tests
# ---------------------------------------------------------------------------


class TestOptInOffPreservesLegacyBehavior(unittest.TestCase):
    """Flag default False ⇒ no `intermediate_reward` key emitted.

    Critical for byte-for-byte preservation of existing WebShop runs.
    """

    def test_no_intermediate_reward_key_when_flag_off(self) -> None:
        env = _FakeWebShopEnv(
            goal={"attributes": ["red", "cotton"], "asin": "B000A"},
            step_script=[
                {"_cur_options": {"color": "red"}},
                {"_cur_options": {"color": "red", "material": "cotton"}},
                {"_done": True, "_reward": 1.0},
            ],
        )
        adapter = _TestableWebShopAdapter(env, max_steps=5)
        adapter.reset()
        for _ in range(3):
            _, _, _, info = adapter.step("search[red shirt]")
            self.assertNotIn(
                "intermediate_reward",
                info,
                "Flag is OFF — adapter must NOT add intermediate_reward.",
            )
            self.assertNotIn("intermediate_reward_source", info)


class TestAttrProgressDelta(unittest.TestCase):
    """Per-step attribute-progress delta arithmetic."""

    def test_first_attribute_engaged_yields_one_over_n(self) -> None:
        env = _FakeWebShopEnv(
            goal={"attributes": ["red", "cotton", "small"]},
            step_script=[
                # Step 1: no overlap yet.
                {"_cur_options": {}},
                # Step 2: engage "red" → overlap 0 → 1, delta 1/3.
                {"_cur_options": {"color": "red"}},
                # Step 3: engage "cotton" → overlap 1 → 2, delta 1/3.
                {"_cur_options": {"color": "red", "material": "cotton"}},
                # Step 4: same options → delta 0.
                {"_cur_options": {"color": "red", "material": "cotton"}},
            ],
        )
        adapter = _TestableWebShopAdapter(
            env, max_steps=5, use_attribute_progress_intermediate_reward=True
        )
        adapter.reset()

        deltas: list[float] = []
        sources: list[str] = []
        for _ in range(4):
            _, _, _, info = adapter.step("click[opt]")
            deltas.append(info["intermediate_reward"])
            sources.append(info["intermediate_reward_source"])

        self.assertAlmostEqual(deltas[0], 0.0, places=6)
        self.assertAlmostEqual(deltas[1], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(deltas[2], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(deltas[3], 0.0, places=6)
        self.assertEqual(sources, ["attr_progress"] * 4)

    def test_unclick_does_not_produce_negative_delta(self) -> None:
        env = _FakeWebShopEnv(
            goal={"attributes": ["red", "cotton"]},
            step_script=[
                {"_cur_options": {"color": "red", "material": "cotton"}},
                # Step 2: drop the cotton option (overlap 2 → 1).
                # Naive delta would be -1/2; we clamp to 0.
                {"_cur_options": {"color": "red"}},
            ],
        )
        adapter = _TestableWebShopAdapter(
            env, max_steps=5, use_attribute_progress_intermediate_reward=True
        )
        adapter.reset()

        _, _, _, info1 = adapter.step("click[red]")
        _, _, _, info2 = adapter.step("click[remove cotton]")
        # Step 1: overlap 0 → 2, delta 2/2 = 1.0.
        self.assertAlmostEqual(info1["intermediate_reward"], 1.0, places=6)
        # Step 2: overlap 2 → 1, clamped to 0.0 (NOT -0.5).
        self.assertAlmostEqual(info2["intermediate_reward"], 0.0, places=6)


class TestAsinBonus(unittest.TestCase):
    """+0.25 ASIN-landing bonus semantics."""

    def test_asin_bonus_fires_once_on_target_click(self) -> None:
        env = _FakeWebShopEnv(
            goal={"attributes": ["red"], "asin": "B000A"},
            step_script=[
                {"_cur_options": {}},  # generic step
                {"_cur_options": {}},  # click[B000A] — bonus turn
                {"_cur_options": {}},  # click[B000A] again — no bonus
            ],
        )
        adapter = _TestableWebShopAdapter(
            env, max_steps=5, use_attribute_progress_intermediate_reward=True
        )
        adapter.reset()

        _, _, _, info0 = adapter.step("search[red]")
        _, _, _, info1 = adapter.step("click[B000A]")
        _, _, _, info2 = adapter.step("click[B000A]")

        # No bonus before the ASIN click + no attr engaged.
        self.assertAlmostEqual(info0["intermediate_reward"], 0.0, places=6)
        # Bonus fires on the first matching click. No attrs newly
        # engaged, so the full IR equals the +0.25 bonus.
        self.assertAlmostEqual(info1["intermediate_reward"], 0.25, places=6)
        # Second click on same ASIN: bonus does NOT fire again.
        self.assertAlmostEqual(info2["intermediate_reward"], 0.0, places=6)

    def test_asin_bonus_case_insensitive(self) -> None:
        env = _FakeWebShopEnv(
            goal={"attributes": ["red"], "asin": "B000Lower"},
            step_script=[{"_cur_options": {}}],
        )
        adapter = _TestableWebShopAdapter(
            env, max_steps=5, use_attribute_progress_intermediate_reward=True
        )
        adapter.reset()
        # Click with different casing should still match (we lowercase
        # the target asin at snapshot time + compare on lowercase).
        _, _, _, info = adapter.step("click[B000LOWER]")
        self.assertAlmostEqual(info["intermediate_reward"], 0.25, places=6)


class TestResetClearsState(unittest.TestCase):
    """Per-episode state must reset cleanly (no leak across episodes)."""

    def test_state_resets_between_episodes(self) -> None:
        env = _FakeWebShopEnv(
            goal={"attributes": ["red"], "asin": "B000A"},
            step_script=[
                {"_cur_options": {"color": "red"}},  # engage "red"
            ],
        )
        adapter = _TestableWebShopAdapter(
            env, max_steps=5, use_attribute_progress_intermediate_reward=True
        )
        adapter.reset()
        _, _, _, info = adapter.step("click[red]")
        # Sanity: bonus state should now be primed (overlap=1).
        self.assertAlmostEqual(info["intermediate_reward"], 1.0, places=6)
        self.assertEqual(adapter._prev_overlap, 1)

        # Reset re-snapshots target_attrs + clears _prev_overlap +
        # clears _asin_bonus_fired.
        env._step_script = [
            {"_cur_options": {"color": "red"}},  # engage "red" again
        ]
        adapter.reset()
        self.assertEqual(adapter._prev_overlap, 0)
        self.assertFalse(adapter._asin_bonus_fired)
        # Re-run: same delta on first engagement.
        _, _, _, info = adapter.step("click[red]")
        self.assertAlmostEqual(info["intermediate_reward"], 1.0, places=6)


class TestIntrospectionFailure(unittest.TestCase):
    """Opt-in on + upstream missing goal payload ⇒ silent fallback."""

    def test_missing_goal_payload_emits_no_ir_and_warns_once(self) -> None:
        env = _FakeWebShopEnv(goal=None, step_script=[{"_cur_options": {}}])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapter = _TestableWebShopAdapter(
                env,
                max_steps=5,
                use_attribute_progress_intermediate_reward=True,
            )
            adapter.reset()
        # Exactly one RuntimeWarning about introspection failure.
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime_warnings), 1)
        self.assertIn("WebShop", str(runtime_warnings[0].message))

        # Step emits no intermediate_reward — Phase 0 byte-for-byte
        # behavior when introspection fails.
        _, _, _, info = adapter.step("click[anything]")
        self.assertNotIn("intermediate_reward", info)
        self.assertNotIn("intermediate_reward_source", info)


if __name__ == "__main__":
    unittest.main()
