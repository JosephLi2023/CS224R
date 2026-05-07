"""Unit tests for `src.trainers.train_hgpo.build_trainer_from_config`.

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


def test_loader_loads_method_hgpo_turnrd_v2_json(tmp_path: Path) -> None:
    """End-to-end load of `configs/method_hgpo_turnrd_v2.json` — the
    config that drives the Method-B v2 sweep. Must construct the
    trainer + TurnRDDecomposer wrapping a TurnRDv2 model without
    raising on any of the v2 keys."""
    import json
    import shutil

    from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer
    from src.turnrd.model import TurnRDv2

    src_cfg = Path(__file__).resolve().parent.parent.parent / "configs" / "method_hgpo_turnrd_v2.json"
    assert src_cfg.is_file(), f"v2 config missing: {src_cfg}"

    # Copy to tmp so we can scrub `/vol/...` paths to local ones (the
    # ckpt path's eager-startup load shouldn't fire on a missing file
    # — refresh_fn just logs a warning — but pointing at a writable tmp
    # location keeps the test hermetic).
    dst = tmp_path / "v2.json"
    shutil.copyfile(src_cfg, dst)
    cfg = json.loads(dst.read_text())
    cfg["turnrd"]["replay_buffer_path"] = str(tmp_path / "replay.jsonl")
    cfg["turnrd"]["ckpt_path"] = str(tmp_path / "ckpt.pt")

    trainer, refresh_fn, emit_path, embedder, judge_dec = build_trainer_from_config(
        cfg, policy=_StubPolicy()
    )
    assert isinstance(trainer.decomposer, TurnRDDecomposer)
    assert isinstance(trainer.decomposer.model, TurnRDv2), (
        "method_hgpo_turnrd_v2.json must build TurnRDv2 (not v1)."
    )
    # Sanity: v2 cfg knobs round-tripped through the loader.
    assert trainer.decomposer.model.cfg.causal is False
    # ckpt_path is set ⇒ refresh_fn must be wired up (even if the ckpt
    # itself doesn't exist yet — that's a no-op + warning, not an error).
    assert refresh_fn is not None
    assert emit_path == str(tmp_path / "replay.jsonl")
    assert embedder is not None
    assert judge_dec is None  # Mode 1


# ---------------------------------------------------------------------------
# AlfWorld method configs — same loader, different env block. Verifies the
# 3 new alfworld-flavored configs construct trainers correctly and the
# `env` block round-trips (the loader itself doesn't touch `env`, but the
# train_loop's _train_loop_impl reads `env.env_kwargs` and we want to
# verify the JSON is shaped as expected).
# ---------------------------------------------------------------------------


def _load_alfworld_method_cfg(tmp_path: Path, name: str) -> dict:
    """Copy the alfworld method config to tmp, scrub vol paths, return
    the parsed dict. Each method config contains a TurnRD block whose
    `replay_buffer_path` and `ckpt_path` point at `/vol/...`; we redirect
    to tmp so the eager refresh_fn at startup doesn't try to write there."""
    import json
    import shutil

    src_cfg = Path(__file__).resolve().parent.parent.parent / "configs" / name
    assert src_cfg.is_file(), f"alfworld method config missing: {src_cfg}"
    dst = tmp_path / name
    shutil.copyfile(src_cfg, dst)
    cfg = json.loads(dst.read_text())
    if "turnrd" in cfg:
        cfg["turnrd"]["replay_buffer_path"] = str(tmp_path / "replay.jsonl")
        cfg["turnrd"]["ckpt_path"] = str(tmp_path / "ckpt.pt")
    return cfg


def test_loader_loads_method_hgpo_progress_alfworld_json(tmp_path: Path) -> None:
    """Method C on AlfWorld: progress decomposer, no TurnRD plumbing.
    Trainer must build, `env.name` must be `alfworld`."""
    cfg = _load_alfworld_method_cfg(tmp_path, "method_hgpo_progress_alfworld.json")
    trainer, refresh_fn, emit_path, embedder, judge_dec = build_trainer_from_config(
        cfg, policy=_StubPolicy()
    )
    assert isinstance(trainer, HGPOTrainer)
    assert refresh_fn is None
    assert emit_path is None
    assert embedder is None
    assert judge_dec is None
    # Env block sanity (the loader doesn't touch env, but the JSON shape
    # is what _train_loop_impl needs).
    assert cfg["env"]["name"] == "alfworld"
    assert "config" in cfg["env"]["env_kwargs"], (
        "alfworld env_kwargs MUST include the upstream `config` dict — "
        "AlfredTWEnv fails to construct without it."
    )


def test_loader_loads_method_hgpo_turnrd_lean_alfworld_json(tmp_path: Path) -> None:
    """Method B v1 lean on AlfWorld: TurnRD v1 (CLS-bottlenecked causal)."""
    from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer
    from src.turnrd.model import TurnRD

    cfg = _load_alfworld_method_cfg(
        tmp_path, "method_hgpo_turnrd_lean_alfworld.json"
    )
    trainer, refresh_fn, emit_path, embedder, judge_dec = build_trainer_from_config(
        cfg, policy=_StubPolicy()
    )
    assert isinstance(trainer.decomposer, TurnRDDecomposer)
    assert isinstance(trainer.decomposer.model, TurnRD), (
        "method_hgpo_turnrd_lean_alfworld.json must build v1 TurnRD."
    )
    assert refresh_fn is not None
    assert emit_path == str(tmp_path / "replay.jsonl")
    assert embedder is not None
    assert judge_dec is None  # Mode 1
    assert cfg["env"]["name"] == "alfworld"


def test_loader_loads_method_hgpo_turnrd_v2_alfworld_json(tmp_path: Path) -> None:
    """Method B v2 on AlfWorld: TurnRDv2 (bidirectional + identifiable
    Σα·v R-loss + progress-prior init)."""
    from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer
    from src.turnrd.model import TurnRDv2

    cfg = _load_alfworld_method_cfg(
        tmp_path, "method_hgpo_turnrd_v2_alfworld.json"
    )
    trainer, refresh_fn, emit_path, embedder, judge_dec = build_trainer_from_config(
        cfg, policy=_StubPolicy()
    )
    assert isinstance(trainer.decomposer, TurnRDDecomposer)
    assert isinstance(trainer.decomposer.model, TurnRDv2), (
        "method_hgpo_turnrd_v2_alfworld.json must build TurnRDv2."
    )
    assert trainer.decomposer.model.cfg.causal is False  # v2 default — bidirectional
    assert refresh_fn is not None
    assert emit_path == str(tmp_path / "replay.jsonl")
    assert embedder is not None
    assert judge_dec is None  # Mode 1
    assert cfg["env"]["name"] == "alfworld"
    # Cache paths must be alfworld-specific so v1 ckpts don't get mistakenly
    # loaded by the alfworld v2 run (and vice versa).
    import json as _json
    src_cfg = Path(__file__).resolve().parent.parent.parent / "configs" / "method_hgpo_turnrd_v2_alfworld.json"
    raw = _json.loads(src_cfg.read_text())
    assert "method_b_v2_alfworld" in raw["turnrd"]["replay_buffer_path"]
    assert "method_b_v2_alfworld" in raw["turnrd"]["ckpt_path"]


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
