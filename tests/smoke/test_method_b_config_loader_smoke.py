"""End-to-end smoke test for the Day-14 Method-B config-loader flow.

Closes the full Method-B loop on CPU without any Modal/GPU/LLM:

    JSON config
      → build_trainer_from_config(cfg, policy=stub)
        → (HGPOTrainer, refresh_fn, emit_path, embedder, judge_decomposer)
      → RolloutCollector(... wired with the producer plumbing ...)
        → collect_group(...) drives a _FakeRunner through FakeWebShopEnv,
          emits one JSONL row per non-empty trajectory.
      → TurnRDReplayDataset(emit_path, mode=1) loads cleanly.
      → train_turnrd(emit_path, mode=1, model=...)  (1 epoch over 4 traj)
        writes a checkpoint via the trainer's standalone-fitting path.
      → refresh_fn() loads the checkpoint into the trainer's decomposer
        in place.

The actual `trainer.train_step(group)` is NOT exercised here because it
requires a real policy logits forward (needs an LLM). The Modal app
`infra/app_train_step.py` covers that path on real hardware. This smoke
test verifies that everything BETWEEN the config and the trainer.train_step
boundary is wired correctly.

Skipped cleanly when torch isn't installed (the loader and the producer
both need it).
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


@dataclass
class _FakeGen:
    text: str
    token_ids: tuple
    token_logprobs: tuple
    prompt_token_count: int
    prompt_token_ids: tuple = ()
    finish_reason: str = "stop"


class _FakeRunner:
    """Minimal vLLM-runner-shaped stub. Returns the same scripted action
    for every prompt at each turn, with a tiny token-id sequence so the
    collector populates `action_token_ids` + `action_token_logprobs`.
    """

    def __init__(self, recipe: list[str]) -> None:
        self.recipe = list(recipe)
        self._cur = 0

    def generate_rich(self, prompts: list[str], sampling) -> list[list[_FakeGen]]:
        n = getattr(sampling, "n", 1)
        t = self.recipe[self._cur % len(self.recipe)]
        self._cur += 1
        ids = tuple(range(10, 10 + max(1, len(t.split()))))
        lps = tuple(-0.1 * (k + 1) for k in range(len(ids)))
        return [[_FakeGen(t, ids, lps, 100) for _ in range(n)] for _ in prompts]


@dataclass
class _SamplingParams:
    n: int = 1
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Stub policy (matches the loader's contract: needs `.model.config.hidden_size`
# and `.model.parameters()` for the trainer's lazy AdamW init).
# ---------------------------------------------------------------------------


class _StubModelConfig:
    hidden_size: int = 16


class _StubModel:
    def __init__(self) -> None:
        self.config = _StubModelConfig()
        # One real parameter so HGPOTrainer's AdamW can be constructed.
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
        self.tokenizer = None  # not used in this smoke (no real embedder calls)

    def trainable_parameters(self):
        return [self.model._param]


# ---------------------------------------------------------------------------
# Smoke: full Method-B config-loader flow on CPU
# ---------------------------------------------------------------------------


class TestConfigLoaderSmoke(unittest.TestCase):
    """End-to-end smoke for the Day-14 wiring."""

    def setUp(self) -> None:
        if torch is None:
            self.skipTest("torch not installed")

    def test_method_b_config_to_refresh_loop(self) -> None:
        """Full loop closure:

        config → trainer + producer plumbing → collect_group emits JSONL →
        replay dataset reads it → standalone train_turnrd writes ckpt →
        refresh_fn loads ckpt into the live decomposer.
        """
        # Local imports gated behind the torch-installed setUp.
        from src.algorithms.grpo.collectors import (
            RolloutCollector,
            RolloutCollectorConfig,
        )
        from src.algorithms.grpo.trainer import HGPOTrainer
        from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer
        from src.envs.fake_webshop import FakeWebShopEnv
        from src.envs.prompts.react_webshop import (
            parse_react_action,
            render_webshop_turn_prompt,
        )
        from src.trainers.train_hgpo import build_trainer_from_config
        from src.turnrd.dataset import TurnRDReplayDataset
        from src.turnrd.model import TurnRD, TurnRDConfig
        from src.turnrd.train import train_turnrd

        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            replay_path = tmp_dir / "replay.jsonl"
            ckpt_path = tmp_dir / "turnrd_ckpt.pt"

            # ---------- 1. config ----------
            # Mirrors `configs/method_hgpo_turnrd.json` shape (Method B,
            # Mode 1) with a zeroed refresh cadence so the trainer's hook
            # never fires during this smoke (we exercise the refresh_fn
            # manually below).
            cfg = {
                "run": {"name": "smoke"},
                "train": {
                    "learning_rate": 1e-6,
                    "clip_eps": 0.2,
                    "grad_accum_steps": 1,
                    "kl_coeff": 0.04,
                    "kl_warmup_episodes": 0,
                },
                "hgpo": {
                    "alpha": 0.5,
                    "lambda_consistency": 0.1,
                    "decomposer": "turnrd",
                },
                "turnrd": {
                    "mode": 1,
                    "layers": 2,
                    "hidden_size": 32,
                    "n_heads": 4,
                    "max_turns": 16,
                    "dropout": 0.0,
                    "refresh_every_episodes": 0,
                    "replay_buffer_path": str(replay_path),
                    "ckpt_path": str(ckpt_path),
                },
            }

            # ---------- 2. loader ----------
            policy = _StubPolicy()
            (
                trainer,
                refresh_fn,
                turnrd_emit_path,
                turnrd_embedder,
                judge_decomposer,
            ) = build_trainer_from_config(cfg, policy=policy)

            self.assertIsInstance(trainer, HGPOTrainer)
            self.assertIsInstance(trainer.decomposer, TurnRDDecomposer)
            self.assertIsNotNone(refresh_fn)
            self.assertEqual(turnrd_emit_path, str(replay_path))
            self.assertIsNotNone(turnrd_embedder)
            self.assertIsNone(judge_decomposer)  # Mode 1
            # The trainer wired the refresh hook on its constructor.
            self.assertIs(trainer._refresh_fn, refresh_fn)
            self.assertTrue(trainer._decomposer_learnable)

            # ---------- 3. collector with producer plumbing ----------
            # Use the production embedder factory's *signature* by passing a
            # deterministic stub: the real embedder needs a HF model+tokenizer
            # which would balloon this smoke into a network/model download.
            # The loader-returned embedder works in production; here we
            # substitute a stub that matches its [T_i, D] CPU fp32 contract.
            INPUT_DIM = policy.model.config.hidden_size  # 16

            def stub_embedder(traj):
                T = len(traj.turns)
                return torch.arange(T * INPUT_DIM, dtype=torch.float32).view(T, INPUT_DIM)

            runner = _FakeRunner(
                [
                    "Action: search[bag]",
                    "Action: click[item-0]",
                    "Action: click[buy]",
                ]
            )
            collector = RolloutCollector(
                runner=runner,
                env_factory=lambda: FakeWebShopEnv(max_steps=8),
                prompt_renderer=render_webshop_turn_prompt,
                action_parser=parse_react_action,
                cfg=RolloutCollectorConfig(max_turns=4),
                turnrd_emit_path=turnrd_emit_path,
                turnrd_embedder=stub_embedder,
                judge_decomposer=judge_decomposer,
            )

            # Drive K=4 rollouts → producer writes 4 JSONL rows.
            group, cstats = collector.collect_group(
                task_id=0,
                env_name="webshop",
                K=4,
                sampling=_SamplingParams(),
            )
            self.assertEqual(group.K, 4)
            self.assertEqual(cstats.K, 4)
            self.assertTrue(replay_path.is_file())

            # ---------- 4. replay dataset reads it ----------
            ds = TurnRDReplayDataset(replay_path, mode=1)
            self.assertEqual(len(ds), len(group.trajectories))
            self.assertEqual(ds.skipped_empty, 0)
            for rec in ds:
                self.assertEqual(rec.task_id, "0")
                self.assertEqual(len(rec.turn_embeds[0]), INPUT_DIM)
                self.assertIsNone(rec.judge_labels)  # Mode 1

            # ---------- 5. standalone train_turnrd writes the ckpt ----------
            # 1 epoch is enough for the smoke — we only need the loop to
            # complete and the ckpt path to exist.
            standalone_model = TurnRD(
                TurnRDConfig(
                    n_layers=cfg["turnrd"]["layers"],
                    hidden_size=cfg["turnrd"]["hidden_size"],
                    n_heads=cfg["turnrd"]["n_heads"],
                    max_turns=cfg["turnrd"]["max_turns"],
                    dropout=cfg["turnrd"]["dropout"],
                ),
                input_dim=INPUT_DIM,
            )
            # Mark the standalone model with a sentinel so we can detect
            # the refresh fn's load below.
            with torch.no_grad():
                standalone_model.cls_query.fill_(0.137)
            summary = train_turnrd(
                replay_path,
                mode=1,
                model=standalone_model,
                n_epochs=1,
                batch_size=2,
                lr=1e-3,
                log_every=0,
                ckpt_path=ckpt_path,
            )
            self.assertEqual(summary["ckpt_path"], str(ckpt_path))
            self.assertGreater(summary["n_steps"], 0)
            self.assertTrue(ckpt_path.is_file())
            # Re-mark AFTER the train_turnrd run (which mutated cls_query)
            # so the marker we look for in the refresh-fn check is exactly
            # what the ckpt should contain.
            ckpt_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            ckpt_cls_query = ckpt_state["cls_query"].clone()

            # Sanity: the live trainer's decomposer.cls_query is a freshly
            # initialised tensor that should NOT match the ckpt's
            # cls_query (the trainer was constructed before any training).
            self.assertFalse(
                torch.allclose(
                    trainer.decomposer.model.cls_query.detach().cpu(),
                    ckpt_cls_query,
                )
            )

            # ---------- 6. refresh_fn loads the ckpt in place ----------
            refresh_fn()
            self.assertTrue(
                torch.allclose(
                    trainer.decomposer.model.cls_query.detach().cpu(),
                    ckpt_cls_query,
                ),
                "refresh_fn() did not load the checkpoint into the live decomposer.",
            )


if __name__ == "__main__":
    unittest.main()
