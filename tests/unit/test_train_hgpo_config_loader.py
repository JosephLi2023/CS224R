"""Unit tests for `src.trainers.train_hgpo.build_trainer_from_config`
(Day 14).

Verification matrix:
1. `test_loader_builds_progress_decomposer`
2. `test_loader_builds_judge_decomposer`
3. `test_loader_builds_turnrd_decomposer_with_refresh_fn`
4. `test_loader_turnrd_mode_2_returns_judge_decomposer`
5. `test_loader_judge_branch_requires_cache_path`
6. `test_loader_turnrd_branch_requires_turnrd_block`
7. `test_loader_unknown_decomposer_raises`

Skipped cleanly on hosts without torch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.algorithms.grpo.trainer import HGPOTrainer  # noqa: E402
from src.algorithms.hgpo.decomposers.judge import JudgeDecomposer  # noqa: E402
from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer  # noqa: E402
from src.trainers.train_hgpo import build_trainer_from_config  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs (can't construct a real LoRAPolicy locally — needs transformers + GPU)
# ---------------------------------------------------------------------------


class _StubModelConfig:
    hidden_size: int = 16


class _StubModel:
    """Minimal nn.Module-shaped object exposing `parameters()` + `.config`."""

    def __init__(self) -> None:
        self.config = _StubModelConfig()
        self._param = torch.nn.Parameter(torch.zeros(1))
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
        self.tokenizer = None  # not needed for the loader (only the producer's embedder hits it)

    def trainable_parameters(self):
        return [self.model._param]


def _judge_cfg(tmp_path: Path) -> dict[str, Any]:
    """Minimal valid judge config shape (matches `configs/judge_openai.json`)."""
    return {
        "judge": {
            "backend": "openai",
            "openai": {
                "model": "gpt-4o-mini",
                "max_retries": 1,
                "temperature": 0.0,
                "timeout_s": 5.0,
                "max_concurrency": 1,
            },
            "cache": {"path": str(tmp_path / "judge.sqlite")},
            "limits": {"max_judge_calls_per_run": 100},
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_loader_builds_progress_decomposer() -> None:
    cfg = {
        "train": {"learning_rate": 1e-6, "clip_eps": 0.2},
        "hgpo": {"alpha": 0.5, "lambda_consistency": 0.0, "decomposer": "progress"},
    }
    trainer, refresh_fn, emit_path, embedder, judge_dec = build_trainer_from_config(
        cfg, policy=_StubPolicy()
    )
    assert isinstance(trainer, HGPOTrainer)
    assert refresh_fn is None
    assert emit_path is None
    assert embedder is None
    assert judge_dec is None
    assert trainer.cfg.alpha == 0.5
    assert trainer.cfg.learning_rate == 1e-6
    assert trainer.cfg.clip_eps == 0.2


def test_loader_builds_judge_decomposer(tmp_path: Path) -> None:
    pytest.importorskip("openai")  # OpenAIJudge construction needs the SDK
    cfg = {
        "train": {"learning_rate": 1e-6},
        "hgpo": {"alpha": 0.5, "lambda_consistency": 0.1, "decomposer": "judge"},
        **_judge_cfg(tmp_path),
    }
    trainer, refresh_fn, emit_path, embedder, judge_dec = build_trainer_from_config(
        cfg, policy=_StubPolicy()
    )
    assert isinstance(trainer, HGPOTrainer)
    # The trainer wraps the JudgeDecomposer's bound .decompose method
    # (matches the existing build_decomposer behavior for "judge").
    assert callable(trainer.decomposer)
    assert refresh_fn is None
    assert emit_path is None
    assert embedder is None
    assert judge_dec is None  # only Mode-2 turnrd returns a judge_decomposer


def test_loader_builds_turnrd_decomposer_with_refresh_fn(tmp_path: Path) -> None:
    """Mode 1 turnrd: returns a TurnRDDecomposer object + a refresh fn that
    calls `load_state_dict` on the decomposer when invoked."""
    # Pre-write a checkpoint that the refresh fn can pick up.
    from src.turnrd.model import TurnRD, TurnRDConfig

    ckpt = tmp_path / "turnrd_ckpt.pt"
    pre_trained = TurnRD(
        TurnRDConfig(n_layers=2, hidden_size=32, n_heads=4, max_turns=16, dropout=0.0),
        input_dim=16,  # matches _StubModelConfig.hidden_size
    )
    # Mark the tensor with a unique value so we can detect the load.
    with torch.no_grad():
        pre_trained.cls_query.fill_(0.137)
    torch.save(pre_trained.state_dict(), ckpt)

    cfg = {
        "train": {"learning_rate": 1e-6},
        "hgpo": {"alpha": 0.5, "lambda_consistency": 0.1, "decomposer": "turnrd"},
        "turnrd": {
            "mode": 1,
            "layers": 2,
            "hidden_size": 32,
            "n_heads": 4,
            "max_turns": 16,
            "dropout": 0.0,
            "refresh_every_episodes": 5,
            "replay_buffer_path": str(tmp_path / "replay.jsonl"),
            "ckpt_path": str(ckpt),
        },
    }
    trainer, refresh_fn, emit_path, embedder, judge_dec = build_trainer_from_config(
        cfg, policy=_StubPolicy()
    )
    assert isinstance(trainer, HGPOTrainer)
    assert isinstance(trainer.decomposer, TurnRDDecomposer)
    assert refresh_fn is not None
    assert emit_path == str(tmp_path / "replay.jsonl")
    assert embedder is not None and callable(embedder)
    assert judge_dec is None  # Mode 1
    assert trainer.cfg.refresh_every_episodes == 5

    # Refresh fn should successfully load the marker checkpoint.
    refresh_fn()
    assert torch.allclose(
        trainer.decomposer.model.cls_query,
        torch.full_like(trainer.decomposer.model.cls_query, 0.137),
    )


def test_loader_turnrd_mode_2_returns_judge_decomposer(tmp_path: Path) -> None:
    """Mode 2 needs a judge config; loader returns a JudgeDecomposer for the
    producer to source per-turn labels from."""
    pytest.importorskip("openai")
    cfg = {
        "train": {"learning_rate": 1e-6},
        "hgpo": {"alpha": 0.5, "lambda_consistency": 0.1, "decomposer": "turnrd"},
        "turnrd": {
            "mode": 2,
            "layers": 2,
            "hidden_size": 32,
            "max_turns": 16,
            "refresh_every_episodes": 0,
            "replay_buffer_path": str(tmp_path / "replay.jsonl"),
        },
        **_judge_cfg(tmp_path),
    }
    _, _, emit_path, _, judge_dec = build_trainer_from_config(cfg, policy=_StubPolicy())
    assert isinstance(judge_dec, JudgeDecomposer)
    assert emit_path == str(tmp_path / "replay.jsonl")


def test_loader_judge_branch_requires_cache_path() -> None:
    cfg = {
        "train": {"learning_rate": 1e-6},
        "hgpo": {"decomposer": "judge"},
        "judge": {"backend": "openai", "openai": {}},  # no cache.path
    }
    with pytest.raises(ValueError, match=r"judge\.cache\.path"):
        build_trainer_from_config(cfg, policy=_StubPolicy())


def test_loader_turnrd_branch_requires_turnrd_block() -> None:
    cfg = {
        "train": {"learning_rate": 1e-6},
        "hgpo": {"decomposer": "turnrd"},
        # No "turnrd" block at all.
    }
    with pytest.raises(ValueError, match=r"'turnrd' config block"):
        build_trainer_from_config(cfg, policy=_StubPolicy())


def test_loader_unknown_decomposer_raises() -> None:
    cfg = {"train": {}, "hgpo": {"decomposer": "bogus"}}
    with pytest.raises(ValueError, match=r"unknown hgpo\.decomposer"):
        build_trainer_from_config(cfg, policy=_StubPolicy())


# ---------------------------------------------------------------------------
# Counterfactual (Method D) branch
# ---------------------------------------------------------------------------


class _CFFakeRunner:
    """Minimal vLLM-runner stub for the CF loader test."""

    def generate_rich(self, prompts, sampling):
        from dataclasses import dataclass

        @dataclass
        class _G:
            text: str = "Action: noop"

        n = getattr(sampling, "n", 1)
        return [[_G() for _ in range(n)] for _ in prompts]


def test_loader_builds_counterfactual_decomposer() -> None:
    """Method D: returns a callable CounterFactualDecomposer + None for
    the TurnRD-only producer plumbing slots (mirrors Methods A/C)."""
    from src.algorithms.hgpo.decomposers.counterfactual import (
        CounterFactualDecomposer,
    )
    from src.envs.fake_webshop import FakeWebShopEnv
    from src.envs.prompts.react_webshop import (
        parse_react_action,
        render_webshop_turn_prompt,
    )

    cfg = {
        "train": {"learning_rate": 1e-6, "clip_eps": 0.2},
        "hgpo": {
            "alpha": 0.5,
            "lambda_consistency": 0.0,
            "decomposer": "counterfactual",
        },
        "counterfactual": {
            "n_alt_actions": 2,
            "max_completion_turns": 1,
            "n_turns_per_traj": 0,
            "skip_if_zero_R": True,
            "output_mode": "raw_delta",
        },
    }

    class _SamplingFactory:
        def __call__(self, **kwargs):
            from dataclasses import make_dataclass

            DC = make_dataclass(
                "_S",
                [(k, type(v), v) for k, v in kwargs.items()],
            )
            return DC(**kwargs)

    trainer, refresh_fn, emit_path, embedder, judge_dec = build_trainer_from_config(
        cfg,
        policy=_StubPolicy(),
        runner=_CFFakeRunner(),
        env_factory=lambda: FakeWebShopEnv(max_steps=8),
        prompt_renderer=render_webshop_turn_prompt,
        action_parser=parse_react_action,
        sampling_factory=_SamplingFactory(),
    )
    assert isinstance(trainer, HGPOTrainer)
    assert isinstance(trainer.decomposer, CounterFactualDecomposer)
    assert refresh_fn is None
    assert emit_path is None
    assert embedder is None
    assert judge_dec is None
    assert trainer.cfg.alpha == 0.5
    assert trainer.cfg.lambda_consistency == 0.0
    # CF read its own block correctly.
    assert trainer.decomposer.n_alt == 2
    assert trainer.decomposer.max_completion == 1
    assert trainer.decomposer.skip_if_zero_R is True
    assert trainer.decomposer.output_mode == "raw_delta"


def test_loader_counterfactual_branch_requires_runner_deps() -> None:
    """Missing runner / env_factory / etc. ⇒ loader raises with a clear
    diagnostic so the orchestrator surfaces the wiring bug at startup,
    not at the first decompose() call."""
    cfg = {
        "train": {},
        "hgpo": {"decomposer": "counterfactual"},
        "counterfactual": {},
    }
    with pytest.raises(ValueError, match=r"counterfactual"):
        build_trainer_from_config(cfg, policy=_StubPolicy())
