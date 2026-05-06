"""End-to-end smoke test for the CounterFactualDecomposer config-loader flow.

Mirrors `test_method_b_config_loader_smoke.py` for Method D (CF):

    JSON config
      → build_trainer_from_config(cfg, policy=stub, runner=fake, env=fake, ...)
        → (HGPOTrainer wired with CFDecomposer, no refresh fn, no producer)
      → trainer.decomposer(group) returns per-turn deltas with the right
        shape and the §3.2 invariant we documented (raw_delta or normalized).
      → A second decompose() call exercises env-pool reuse so we know the
        decomposer can be driven repeatedly from the trainer's loop.

The actual `trainer.train_step(group)` is NOT exercised here because it
requires a real LoRA-policy logprob forward (which means an LLM). The
Modal app `infra/app_train_loop.py` covers that path on real hardware.
This smoke test verifies the wiring between the JSON config and the
decomposer's callable surface.

Skipped cleanly when torch isn't installed (the loader instantiates
HGPOTrainerConfig which transitively pulls torch via the trainer module).
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch
except ImportError:  # pragma: no cover (skipped via setUp below)
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Test doubles (mirror tests/unit/test_counterfactual_decomposer.py shape).
# ---------------------------------------------------------------------------


@dataclass
class _Sampling:
    n: int = 1
    temperature: float = 1.0
    max_tokens: int = 48
    return_logprobs: bool = False


@dataclass
class _Gen:
    text: str
    token_ids: tuple = ()
    token_logprobs: tuple = ()
    prompt_token_count: int = 0
    prompt_token_ids: tuple = ()
    finish_reason: str = "stop"


class _FakeRunner:
    """vLLM-runner-shaped stub. Returns a stage-aware action so the CF
    rollouts can complete on FakeWebShopEnv. Alt-action calls (high temp)
    deliberately pick a wrong item at the click stage to produce a
    non-zero baseline R; greedy-completion calls (temp ≤ 0) always do
    the canonical correct action.
    """

    def generate_rich(self, prompts, sampling):
        n = getattr(sampling, "n", 1)
        temp = float(getattr(sampling, "temperature", 1.0))
        is_alt = temp > 1e-6
        out = []
        for prompt in prompts:
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
            if is_alt and stage == "click":
                action = "click[item-2]"  # wrong-item alt → R drops to 0.4
            else:
                action = {
                    "search": "search[laptop bag]",
                    "click": "click[item-0]",
                    "buy": "click[buy]",
                    "other": "think[noop]",
                }[stage]
            row = [_Gen(text=f"Thought: t\nAction: {action}") for _ in range(n)]
            out.append(row)
        return out


# ---------------------------------------------------------------------------
# Stub policy (loader only needs `policy.model.config.hidden_size` for the
# turnrd branch; the CF branch never reads it but we keep the stub uniform
# with the Method-B smoke).
# ---------------------------------------------------------------------------


class _StubModelConfig:
    hidden_size: int = 16


class _StubModel:
    def __init__(self) -> None:
        self.config = _StubModelConfig()
        self._param = torch.nn.Parameter(torch.zeros(1))  # type: ignore[union-attr]
        self.training = True

    def parameters(self):
        yield self._param

    def named_modules(self):
        return iter([])

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class _StubPolicy:
    def __init__(self) -> None:
        self.model = _StubModel()
        self.tokenizer = None


# ---------------------------------------------------------------------------
# Smoke: full Method-D config-loader flow + repeated decompose() calls
# ---------------------------------------------------------------------------


class TestCounterFactualSmoke(unittest.TestCase):
    """End-to-end smoke for the CF decomposer wiring."""

    def setUp(self) -> None:
        if torch is None:
            self.skipTest("torch not installed (HGPOTrainerConfig transitively imports torch)")

    def test_method_d_config_to_repeated_decompose(self) -> None:
        """Full loop closure on FakeWebShopEnv:

        config → build_trainer_from_config(cfg, policy, runner, env_factory,
            prompt_renderer, action_parser, sampling_factory)
          → trainer + CFDecomposer wired
          → collector.collect_group → real K=4 group
          → trainer.decomposer(group) → list[K] of list[T_i] of Δ_t
          → second collect_group + decompose ⇒ env-pool reused (no factory
            churn).
        """
        from src.algorithms.grpo.collectors import (
            RolloutCollector,
            RolloutCollectorConfig,
        )
        from src.algorithms.grpo.trainer import HGPOTrainer
        from src.algorithms.hgpo.decomposers.counterfactual import (
            CounterFactualDecomposer,
        )
        from src.envs.fake_webshop import FakeWebShopEnv
        from src.envs.prompts.react_webshop import (
            parse_react_action,
            render_webshop_turn_prompt,
        )
        from src.trainers.train_hgpo import build_trainer_from_config

        # Track factory invocations so we can assert env-pool reuse.
        factory_calls: list[FakeWebShopEnv] = []

        def env_factory() -> FakeWebShopEnv:
            e = FakeWebShopEnv(max_steps=8)
            factory_calls.append(e)
            return e

        runner = _FakeRunner()

        # ---------- 1. config (mirrors method_hgpo_counterfactual.json) ----
        cfg = {
            "run": {"name": "smoke_cf"},
            "train": {
                "learning_rate": 1e-6,
                "clip_eps": 0.2,
                "grad_accum_steps": 1,
                "kl_coeff": 0.04,
            },
            "hgpo": {
                "alpha": 0.5,
                "lambda_consistency": 0.0,  # CF Δ doesn't sum to R; consistency off
                "decomposer": "counterfactual",
            },
            "counterfactual": {
                "n_alt_actions": 2,
                "max_completion_turns": 2,
                "cf_temperature": 1.0,
                "completion_temperature": 0.0,
                "n_turns_per_traj": 0,        # all turns
                "skip_if_zero_R": True,
                "output_mode": "raw_delta",
                "seed": 0,
            },
        }

        # ---------- 2. loader ---------------------------------------------
        policy = _StubPolicy()
        (
            trainer,
            refresh_fn,
            turnrd_emit_path,
            turnrd_embedder,
            judge_decomposer,
        ) = build_trainer_from_config(
            cfg,
            policy=policy,
            runner=runner,
            env_factory=env_factory,
            prompt_renderer=render_webshop_turn_prompt,
            action_parser=parse_react_action,
            sampling_factory=_Sampling,
        )

        self.assertIsInstance(trainer, HGPOTrainer)
        self.assertIsInstance(trainer.decomposer, CounterFactualDecomposer)
        # Method D shouldn't surface a refresh hook or any producer plumbing
        # (mirrors Methods A/C).
        self.assertIsNone(refresh_fn)
        self.assertIsNone(turnrd_emit_path)
        self.assertIsNone(turnrd_embedder)
        self.assertIsNone(judge_decomposer)
        # No learnable surface ⇒ trainer skips second AdamW + C3 reattach.
        self.assertFalse(trainer.decomposer.has_learnable_params)
        self.assertFalse(getattr(trainer, "_decomposer_learnable", False))

        # ---------- 3. drive K=4 rollouts on FakeWebShop ------------------
        collector = RolloutCollector(
            runner=runner,
            env_factory=env_factory,
            prompt_renderer=render_webshop_turn_prompt,
            action_parser=parse_react_action,
            cfg=RolloutCollectorConfig(max_turns=4),
        )
        group, cstats = collector.collect_group(
            task_id=0,
            env_name="webshop",
            K=4,
            sampling=_Sampling(n=1),
        )
        self.assertEqual(group.K, 4)
        self.assertEqual(cstats.K, 4)
        self.assertTrue(all(traj.final_reward > 0 for traj in group.trajectories))

        # ---------- 4. decompose() returns per-turn Δ_t -------------------
        n_envs_before = len(factory_calls)
        per_turn = trainer.decomposer(group)
        self.assertEqual(len(per_turn), 4)
        for traj, row in zip(group.trajectories, per_turn):
            self.assertEqual(len(row), len(traj.turns))
            self.assertTrue(all(isinstance(x, float) for x in row))
            # raw_delta mode → per-turn values are bounded by R in
            # magnitude (alt R is in [0, 1] for FakeWebShop, actual R
            # likewise). They CAN be negative when the policy's actual
            # action did worse than the alt baseline (e.g., the recipe
            # runner's high-temp click sample returns 'click[item-2]'
            # but the greedy completion picks 'click[item-0]'). That's
            # the correct CF semantic, just exercised in the test.
            for x in row:
                self.assertLessEqual(abs(x), 1.0 + 1e-6)
        # The CF decomposer should have built its own env pool, so factory
        # was called at least once during decompose().
        self.assertGreater(len(factory_calls), n_envs_before)

        # ---------- 5. second decompose() reuses the pool -----------------
        # Driving the trainer for a second "episode" should not blow up the
        # env factory call count beyond the first decompose's allocation.
        group2, _ = collector.collect_group(
            task_id=1, env_name="webshop", K=4, sampling=_Sampling(n=1)
        )
        n_after_first = len(factory_calls)
        per_turn_2 = trainer.decomposer(group2)
        self.assertEqual(len(per_turn_2), 4)
        # The CF decomposer's pool may need MORE envs if turn counts differ,
        # but a second call with the same K + similar T_i should NOT add
        # more than the original allocation. Specifically, the CF pool
        # must not rebuild from scratch each call.
        cf_pool_size = len(trainer.decomposer._env_pool)  # type: ignore[attr-defined]
        # The pool grew once during step 4 to N_alt × Σ T_i (≈ 24 envs for
        # K=4, T̄=3, N=2). Step 5 should reuse those.
        self.assertGreater(cf_pool_size, 0)
        # Allow the pool to grow if T_i varies, but it shouldn't double.
        self.assertLess(len(factory_calls), 2 * n_after_first)


if __name__ == "__main__":
    unittest.main()
