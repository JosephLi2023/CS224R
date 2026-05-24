"""Unit tests for the WebShop oracle SFT trajectory generator.

Validates `infra/app_webshop_sft_gen.py::_oracle_episode` end-to-end
against a deterministic `_FakeWebShopEnv` fixture (same pattern as
`tests/unit/test_webshop_adapter_attr_progress.py`) so we don't need
the real upstream `web_agent_site` package installed.

Coverage matrix (high-quality oracle = passes ALL of these):
  1. HAPPY PATH    — search → click ASIN (page 1) → click attrs →
                     click Buy Now with terminal reward=1.0 ⇒ all rows
                     flushed, status="won", final_reward stamped on
                     every row.
  2. PAGINATION    — ASIN found on page 3 ⇒ trajectory contains 1
                     `search[…]` + 2 `click[Next >]` + 1 `click[<asin>]`
                     + attr clicks + `click[Buy Now]`, in order.
  3. ASIN MISS     — ASIN never appears in `max_result_pages=5`
                     scans ⇒ status="asin_not_found", traj_rows=[].
  4. LOW REWARD    — Buy Now returns reward < threshold ⇒ status=
                     "lost", traj_rows=[] (won-gate dropped them).
  5. NO GOAL       — env.goal is empty ⇒ status="no_goal", traj=[].
  6. NO ASIN       — goal lacks asin/asins ⇒ status="no_target_asin".
  7. PROMPT QUALITY — every rendered prompt ends with "Thought:",
                      contains the instruction, and the i-th prompt
                      replays the prior i (Observation, Action) pairs
                      (zero template drift vs runtime renderer).
  8. SCHEMA RT     — the flushed rows can be written to JSONL and
                     re-loaded by `load_sft_examples_from_jsonl`
                     without loss (the consumer's contract).
  9. MAX STEPS     — `max_steps_per_episode` cap fires when option-
                     click count blows past the budget ⇒ status=
                     "truncated", traj=[].

The fake env's `step(action)` returns observations that explicitly
mention the ASIN substring exactly when the test wants the oracle to
"see" the target on the current page, so we have surgical control
over the oracle's page-walk decision branch.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from infra.app_webshop_sft_gen import (
    _instruction_from_goal,
    _oracle_episode,
    _query_from_goal,
    _render_prompt_for_state,
    _resolve_env_goal,
)
from src.datasets.sft_webshop import load_sft_examples_from_jsonl
from src.envs.webshop_adapter import WebShopAdapter


# ---------------------------------------------------------------------------
# Fake env + adapter shim (mirrors tests/unit/test_webshop_adapter_attr_progress.py)
# ---------------------------------------------------------------------------


class _FakeWebShopEnv:
    """Stub WebShop env scripted to drive deterministic oracle trajectories.

    Each entry in `step_script` is consumed in order; it controls the
    observation text + reward + done flag for one `step()` return:
      {"obs": str, "_reward": float, "_done": bool,
       "_cur_options": dict, "valid_actions": list[str]}

    Setting `obs` to a string containing the target ASIN (case-
    insensitive) is how the test signals to the oracle that the ASIN
    is visible on the current search-results page.
    """

    def __init__(self, goal: dict, step_script: list[dict]) -> None:
        self.goal = dict(goal)
        self.session = 0
        self.cur_options: dict = {}
        self._step_script = list(step_script)
        self._cursor = 0
        self.actions_seen: list[str] = []
        self._initial_obs = "WebShop home. Search for products."

    def reset(self, **kwargs):
        self.cur_options = {}
        self._cursor = 0
        self.actions_seen = []
        return (self._initial_obs, {"valid_actions": ["search[anything]"]})

    def step(self, action: str):
        self.actions_seen.append(action)
        if self._cursor >= len(self._step_script):
            # Out-of-script: synthesize a benign no-op so the test fails
            # loudly instead of mysteriously hanging.
            entry = {"obs": "<EOS>", "_reward": 0.0, "_done": True}
        else:
            entry = self._step_script[self._cursor]
        self._cursor += 1
        if "_cur_options" in entry:
            self.cur_options = dict(entry["_cur_options"])
        obs = entry.get("obs", f"after {action}")
        reward = float(entry.get("_reward", 0.0))
        done = bool(entry.get("_done", False))
        info = {"valid_actions": entry.get("valid_actions", [])}
        return obs, reward, done, info


class _TestableWebShopAdapter(WebShopAdapter):
    """WebShopAdapter that injects a `_FakeWebShopEnv` (no real install)."""

    def __init__(self, fake_env: _FakeWebShopEnv, **kwargs) -> None:
        self._fake_env = fake_env
        kwargs.setdefault("max_steps", 30)
        kwargs.setdefault("observation_mode", "text")
        kwargs.setdefault("task_split", "train")
        super().__init__(**kwargs)

    def _build_webshop_env(self):  # type: ignore[override]
        return self._fake_env


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


class TestGoalIntrospection(unittest.TestCase):
    def test_resolve_env_goal_from_env_attr(self):
        env = _FakeWebShopEnv(
            goal={"query": "red shirt", "asin": "B000ABC"}, step_script=[],
        )
        self.assertEqual(_resolve_env_goal(env)["query"], "red shirt")

    def test_resolve_env_goal_returns_none_for_empty(self):
        class _Bare:
            goal = None
            server = None
            session = None
        self.assertIsNone(_resolve_env_goal(_Bare()))

    def test_instruction_prefers_instruction_text(self):
        goal = {"instruction_text": "Buy a red shirt under $30", "query": "red shirt"}
        self.assertEqual(
            _instruction_from_goal(goal), "Buy a red shirt under $30"
        )

    def test_instruction_falls_back_to_query(self):
        self.assertEqual(_instruction_from_goal({"query": "red shirt"}), "red shirt")
        self.assertEqual(_instruction_from_goal({}), "")
        self.assertEqual(_instruction_from_goal(None), "")

    def test_query_extracts_from_goal(self):
        # Precedence: name > instruction_text > query > "".
        # Pinpoint precision when `name` is present (target product
        # title → top-1 BM25 hit virtually always).
        self.assertEqual(
            _query_from_goal({
                "name": "Slim Fit Dark Blue Polo Shirt by Acme",
                "instruction_text": "Find me a slim fit dark blue polo",
                "query": "men's polos",
            }),
            "Slim Fit Dark Blue Polo Shirt by Acme",
        )
        # Falls back to instruction_text when `name` is absent.
        self.assertEqual(
            _query_from_goal({
                "instruction_text": "Find me a slim fit dark blue polo",
                "query": "men's polos",
            }),
            "Find me a slim fit dark blue polo",
        )
        # Then falls back to `query`.
        self.assertEqual(_query_from_goal({"query": "blue jeans"}), "blue jeans")
        self.assertEqual(_query_from_goal({"instruction_text": "find x"}), "find x")
        self.assertEqual(_query_from_goal({}), "")


# ---------------------------------------------------------------------------
# Oracle episode tests
# ---------------------------------------------------------------------------


GOAL_HAPPY = {
    "query": "red shirt",
    "instruction_text": "Buy a red cotton shirt size medium under $30",
    "asin": "B07ABCDEFG",
    "asins": ["B07ABCDEFG"],
    "attributes": ["red", "cotton", "medium"],
    "category": "shirts",
    "price_upper": 30.0,
}


def _build_adapter(goal: dict, script: list[dict], **kw) -> _TestableWebShopAdapter:
    return _TestableWebShopAdapter(
        fake_env=_FakeWebShopEnv(goal=goal, step_script=script),
        **kw,
    )


class TestOracleHappyPath(unittest.TestCase):
    def test_happy_path_yields_full_trajectory(self):
        # Script: page 1 of results contains target ASIN → oracle clicks
        # in immediately. Then 3 attr clicks (red/cotton/medium) +
        # Buy Now. All non-terminal returns reward=0, terminal=1.0.
        asin_visible = "[Result 1] B07ABCDEFG Red Cotton Shirt $25 — relevant"
        script = [
            # step 1: search[red shirt] → results page 1
            {"obs": asin_visible, "_reward": 0.0, "_done": False},
            # step 2: click[b07abcdefg] → item page (lowercase from _extract_target_asin)
            {"obs": "Red Cotton Shirt page. Options: color [red], material [cotton], size [medium]",
             "_reward": 0.0, "_done": False,
             "_cur_options": {}},
            # step 3-5: click[red], click[cotton], click[medium]
            {"obs": "Selected color=red", "_reward": 0.0, "_done": False,
             "_cur_options": {"color": "red"}},
            {"obs": "Selected material=cotton", "_reward": 0.0, "_done": False,
             "_cur_options": {"color": "red", "material": "cotton"}},
            {"obs": "Selected size=medium", "_reward": 0.0, "_done": False,
             "_cur_options": {"color": "red", "material": "cotton", "size": "medium"}},
            # step 6: click[Buy Now] → terminal reward=1.0
            {"obs": "Order placed. Thanks!", "_reward": 1.0, "_done": True},
        ]
        adapter = _build_adapter(GOAL_HAPPY, script)
        rows, reward, status = _oracle_episode(adapter, session_id=42)

        self.assertEqual(status, "won")
        self.assertEqual(reward, 1.0)
        # 6 turns total: search + click_asin + 3 attr clicks + Buy Now.
        self.assertEqual(len(rows), 6, f"expected 6 turns, got {len(rows)}")
        actions = [r["action"] for r in rows]
        self.assertEqual(actions, [
            "search[Buy a red cotton shirt size medium under $30]",
            "click[b07abcdefg]",  # _extract_target_asin lowercases
            "click[red]",
            "click[cotton]",
            "click[medium]",
            "click[Buy Now]",
        ])
        # Schema invariants on every row.
        for i, r in enumerate(rows):
            self.assertEqual(r["step_idx"], i)
            self.assertEqual(r["trajectory_id"], "oracle_000042")
            self.assertEqual(r["instruction"],
                             "Buy a red cotton shirt size medium under $30")
            self.assertEqual(r["final_reward"], 0.0,
                             "final_reward placeholder until generator backfills")
            self.assertIsInstance(r["prompt"], str)
            self.assertTrue(r["prompt"].rstrip().endswith("Thought:"),
                            f"prompt must end with 'Thought:' (turn {i})")


class TestOraclePagination(unittest.TestCase):
    def test_asin_found_on_page_3_walks_pages(self):
        # Pages 1+2 do NOT contain target ASIN → oracle clicks Next.
        # Page 3 surfaces it → oracle clicks ASIN.
        empty_page = "[Result] B000OTHER1 some other item. No matches yet."
        page_with_asin = "[Result 1] B07ABCDEFG Red Cotton Shirt page 3"
        script = [
            {"obs": empty_page, "_reward": 0.0, "_done": False},          # 1: search → p1
            {"obs": empty_page, "_reward": 0.0, "_done": False},          # 2: Next → p2
            {"obs": page_with_asin, "_reward": 0.0, "_done": False},      # 3: Next → p3 (visible)
            {"obs": "item page", "_reward": 0.0, "_done": False,
             "_cur_options": {}},                                          # 4: click ASIN
            # Now attr loop (3 attrs, all no-op on cur_options for brevity)
            {"obs": "selected", "_reward": 0.0, "_done": False},          # 5: click[red]
            {"obs": "selected", "_reward": 0.0, "_done": False},          # 6: click[cotton]
            {"obs": "selected", "_reward": 0.0, "_done": False},          # 7: click[medium]
            {"obs": "purchased", "_reward": 1.0, "_done": True},          # 8: Buy Now
        ]
        adapter = _build_adapter(GOAL_HAPPY, script)
        rows, reward, status = _oracle_episode(adapter, session_id=7,
                                               max_result_pages=5)
        self.assertEqual(status, "won")
        self.assertEqual(reward, 1.0)
        actions = [r["action"] for r in rows]
        self.assertEqual(actions[:4], [
            "search[Buy a red cotton shirt size medium under $30]",
            "click[Next >]",
            "click[Next >]",
            "click[b07abcdefg]",
        ])
        self.assertEqual(actions[-1], "click[Buy Now]")

    def test_asin_never_found_drops_trajectory(self):
        empty_page = "[Result] B000OTHER nothing here"
        script = [{"obs": empty_page, "_reward": 0.0, "_done": False}] * 8
        adapter = _build_adapter(GOAL_HAPPY, script)
        rows, reward, status = _oracle_episode(adapter, session_id=99,
                                               max_result_pages=5)
        self.assertEqual(status, "asin_not_found")
        self.assertEqual(rows, [])
        self.assertEqual(reward, 0.0)


class TestOracleDropConditions(unittest.TestCase):
    def test_low_terminal_reward_drops_trajectory(self):
        asin_visible = "[Result 1] B07ABCDEFG matches"
        # Buy Now returns reward=0.5, below default threshold=0.99.
        script = [
            {"obs": asin_visible, "_reward": 0.0, "_done": False},        # 1: search
            {"obs": "item page", "_reward": 0.0, "_done": False},         # 2: click ASIN
            {"obs": "ok", "_reward": 0.0, "_done": False},                # 3: click[red]
            {"obs": "ok", "_reward": 0.0, "_done": False},                # 4: click[cotton]
            {"obs": "ok", "_reward": 0.0, "_done": False},                # 5: click[medium]
            {"obs": "partial match", "_reward": 0.5, "_done": True},      # 6: Buy Now (low)
        ]
        adapter = _build_adapter(GOAL_HAPPY, script)
        rows, reward, status = _oracle_episode(adapter, session_id=11)
        self.assertEqual(status, "lost")
        self.assertEqual(rows, [])
        self.assertEqual(reward, 0.5)

    def test_no_goal_drops_immediately(self):
        adapter = _build_adapter(goal={}, script=[])
        rows, reward, status = _oracle_episode(adapter, session_id=0)
        # Empty goal lacks `query`, so falls through `no_goal` branch.
        self.assertEqual(status, "no_goal")
        self.assertEqual(rows, [])

    def test_no_target_asin_drops(self):
        goal_no_asin = {
            "query": "red shirt", "instruction_text": "buy red shirt",
            "attributes": ["red"], "category": "shirts",
            # NO asin / asins keys.
        }
        adapter = _build_adapter(goal_no_asin, [])
        rows, reward, status = _oracle_episode(adapter, session_id=0)
        self.assertEqual(status, "no_target_asin")
        self.assertEqual(rows, [])

    def test_truncated_when_max_steps_exceeded(self):
        # Force truncation: very low max_steps_per_episode, ASIN missing
        # so oracle keeps clicking Next.
        empty = "[Result] B000NOPE"
        script = [{"obs": empty, "_reward": 0.0, "_done": False}] * 10
        adapter = _build_adapter(GOAL_HAPPY, script, max_steps=10)
        rows, reward, status = _oracle_episode(
            adapter, session_id=0,
            max_result_pages=20,        # don't bail on page count
            max_steps_per_episode=3,    # force truncation
        )
        self.assertEqual(status, "truncated")
        self.assertEqual(rows, [])


# ---------------------------------------------------------------------------
# Prompt-quality tests (zero template drift vs runtime ReAct renderer)
# ---------------------------------------------------------------------------


class TestOraclePromptQuality(unittest.TestCase):
    def test_prompts_contain_instruction_and_replay_history(self):
        asin_visible = "[Result 1] B07ABCDEFG Red Cotton Shirt"
        script = [
            {"obs": asin_visible, "_reward": 0.0, "_done": False},
            {"obs": "Red Cotton Shirt item page", "_reward": 0.0, "_done": False},
            {"obs": "color=red selected", "_reward": 0.0, "_done": False},
            {"obs": "ok", "_reward": 0.0, "_done": False},
            {"obs": "ok", "_reward": 0.0, "_done": False},
            {"obs": "purchased", "_reward": 1.0, "_done": True},
        ]
        adapter = _build_adapter(GOAL_HAPPY, script)
        rows, _, status = _oracle_episode(adapter, session_id=1)
        self.assertEqual(status, "won")

        # Every prompt embeds the instruction.
        for r in rows:
            self.assertIn("Buy a red cotton shirt", r["prompt"])
            self.assertTrue(r["prompt"].rstrip().endswith("Thought:"))
            # System prompt's ReAct format hint always present.
            self.assertIn("ReAct", r["prompt"])

        # Turn 0's prompt has NO history; turn 1 replays turn 0's
        # (initial_obs, search[…]); turn 2 also replays turn 1.
        # Use the full emitted search-action body to avoid colliding with
        # the system prompt's `Action: search[<query>]` format hint.
        emitted_search = "Action: search[Buy a red cotton shirt size medium under $30]"
        self.assertNotIn(emitted_search, rows[0]["prompt"])
        self.assertIn(emitted_search, rows[1]["prompt"])
        self.assertIn("Action: click[b07abcdefg]", rows[2]["prompt"])
        # And critically NO synthesized Thought lines in history
        # (matches the runtime collector's parse-Thoughts-away rule).
        self.assertNotIn("Thought: I'll search", rows[1]["prompt"])

    def test_prompts_byte_identical_to_runtime_renderer(self):
        """The oracle's `_render_prompt_for_state` must produce the
        SAME prompt the runtime ReAct collector would render for the
        same (state, history, instruction). Pins the SFT-rollout
        prompt-parity contract on the gen side."""
        from types import SimpleNamespace

        from src.envs.prompts.react_webshop import render_webshop_turn_prompt

        state = SimpleNamespace(
            observation_text="Red Cotton Shirt page",
            valid_actions=[],
        )
        history = [
            SimpleNamespace(observation_text="home page",
                            action_text="search[red shirt]"),
            SimpleNamespace(observation_text="[Result] B07ABCDEFG",
                            action_text="click[b07abcdefg]"),
        ]
        instruction = "Buy a red cotton shirt"

        oracle_prompt = _render_prompt_for_state(state, history, instruction)
        runtime_prompt = render_webshop_turn_prompt(
            SimpleNamespace(
                observation_text="Red Cotton Shirt page",
                instruction=instruction,
                valid_actions=[],
            ),
            history,
        )
        self.assertEqual(
            oracle_prompt, runtime_prompt,
            "Oracle prompt drifted from runtime renderer — SFT will "
            "teach a template the rollout collector doesn't use.",
        )


# ---------------------------------------------------------------------------
# JSONL round-trip: oracle output → loader input → SFTExample fidelity
# ---------------------------------------------------------------------------


class TestOracleJsonlRoundTrip(unittest.TestCase):
    def test_won_rows_roundtrip_through_loader(self):
        """Simulate the generator's flush path: backfill final_reward,
        write JSONL, then load via the trainer's loader and verify the
        SFTExample fields match."""
        asin_visible = "[Result 1] B07ABCDEFG match"
        script = [
            {"obs": asin_visible, "_reward": 0.0, "_done": False},
            {"obs": "item page", "_reward": 0.0, "_done": False},
            {"obs": "ok", "_reward": 0.0, "_done": False},
            {"obs": "ok", "_reward": 0.0, "_done": False},
            {"obs": "ok", "_reward": 0.0, "_done": False},
            {"obs": "purchased", "_reward": 1.0, "_done": True},
        ]
        adapter = _build_adapter(GOAL_HAPPY, script)
        rows, final_reward, status = _oracle_episode(adapter, session_id=123)
        self.assertEqual(status, "won")

        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "oracle_trajs.jsonl"
            with open(out, "w") as fh:
                for r in rows:
                    r["final_reward"] = float(final_reward)
                    fh.write(json.dumps(r) + "\n")

            # Defensive: 0.99 threshold matches generator's reward_threshold.
            keep = load_sft_examples_from_jsonl(str(out), min_reward=0.99)
            self.assertEqual(len(keep), len(rows))
            for src, ex in zip(rows, keep):
                self.assertEqual(ex.prompt, src["prompt"])
                self.assertEqual(ex.action, src["action"])
                self.assertEqual(ex.step_idx, src["step_idx"])
                self.assertEqual(ex.trajectory_id, src["trajectory_id"])
                self.assertEqual(ex.instruction, src["instruction"])
                self.assertEqual(ex.final_reward, 1.0)


if __name__ == "__main__":
    unittest.main()
