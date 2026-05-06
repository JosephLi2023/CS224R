"""Wiring tests for the TurnRDv2 selector.

Verifies the `turnrd.version` JSON-config selector + the corresponding
trainer/standalone-fitter branches keep v1 behavior intact AND build
the v2 architecture when requested.

Test matrix:
1. `_build_turnrd_branch(cfg with version="v2")` returns a
   `TurnRDDecomposer` whose `.model` is `TurnRDv2`.
2. `_build_turnrd_branch(cfg with version="v1")` (and no version key)
   still returns a v1 `TurnRD` — regression guard.
3. `_build_turnrd_branch` rejects unknown `version` values loudly.
4. `train_turnrd(version="v2", ...)` runs one step on a synthetic replay
   buffer + writes a ckpt that round-trips cleanly via
   `decomposer.load_state_dict(...)` (v2 state-dict compat sanity).
5. `train_turnrd(version="v2", mode=2, ...)` raises (v2 has no Mode-2
   distillation path).
6. The summary returned by `train_turnrd(version="v2")` reports the
   v2 component-loss breakdown keys.

Skipped cleanly on torch-less hosts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.algorithms.grpo.trainer import HGPOTrainer  # noqa: E402
from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer  # noqa: E402
from src.trainers.train_hgpo import build_trainer_from_config  # noqa: E402
from src.turnrd.model import TurnRD, TurnRDv2, TurnRDv2Config  # noqa: E402
from src.turnrd.train import train_turnrd  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs (mirror tests/unit/test_train_hgpo_config_loader.py — can't construct
# a real LoRAPolicy locally without transformers + GPU).
# ---------------------------------------------------------------------------


class _StubModelConfig:
    hidden_size: int = 16


class _StubModel:
    """Minimal nn.Module-shaped stub exposing `parameters()` + `.config`."""

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
        self.tokenizer = None  # producer-only; loader doesn't touch it

    def trainable_parameters(self):
        return [self.model._param]


def _v2_cfg(tmp_path: Path) -> dict:
    """A minimal valid Method-B v2 cfg (no judge block — Mode 1 only)."""
    return {
        "train": {"learning_rate": 1e-6},
        "hgpo": {
            "alpha": 0.5,
            "lambda_consistency": 0.0,
            "decomposer": "turnrd",
        },
        "turnrd": {
            "version": "v2",
            "mode": 1,
            "layers": 2,
            "hidden_size": 32,
            "n_heads": 4,
            "max_turns": 16,
            "dropout": 0.0,
            "causal": False,
            "progress_prior_strength": 1.0,
            "refresh_every_episodes": 5,
            "replay_buffer_path": str(tmp_path / "replay.jsonl"),
            # Intentionally no ckpt_path → refresh_fn is None, so the
            # eager-startup load is a no-op (avoids needing a pre-saved
            # ckpt for this wiring sanity check).
        },
    }


# ---------------------------------------------------------------------------
# 1. v2 branch builds TurnRDv2
# ---------------------------------------------------------------------------


def test_build_turnrd_branch_v2_returns_turnrd_v2(tmp_path: Path) -> None:
    cfg = _v2_cfg(tmp_path)
    trainer, refresh_fn, emit_path, embedder, judge_dec = build_trainer_from_config(
        cfg, policy=_StubPolicy()
    )
    assert isinstance(trainer, HGPOTrainer)
    assert isinstance(trainer.decomposer, TurnRDDecomposer)
    assert isinstance(trainer.decomposer.model, TurnRDv2), (
        f"version='v2' must build TurnRDv2; got {type(trainer.decomposer.model).__name__}"
    )
    # Confirm v2 hyperparams flowed through the JSON keys.
    assert trainer.decomposer.model.cfg.n_layers == 2
    assert trainer.decomposer.model.cfg.hidden_size == 32
    assert trainer.decomposer.model.cfg.causal is False
    assert trainer.decomposer.model.cfg.progress_prior_strength == 1.0
    # v2 doesn't ship a ckpt path here, so refresh_fn must be None.
    assert refresh_fn is None
    assert emit_path == str(tmp_path / "replay.jsonl")
    assert embedder is not None and callable(embedder)
    assert judge_dec is None


# ---------------------------------------------------------------------------
# 2. v1 (default + explicit) regression guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version_key", [None, "v1"])
def test_build_turnrd_branch_v1_default_returns_turnrd_v1(
    tmp_path: Path, version_key
) -> None:
    """Omitting `version` (or setting it to "v1" explicitly) MUST still
    construct a v1 `TurnRD` — protects the existing method_b_lean
    config + every other v1-using config from a silent architecture flip.
    """
    cfg = {
        "train": {"learning_rate": 1e-6},
        "hgpo": {
            "alpha": 0.5,
            "lambda_consistency": 0.0,
            "decomposer": "turnrd",
        },
        "turnrd": {
            "mode": 1,
            "layers": 2,
            "hidden_size": 32,
            "n_heads": 4,
            "max_turns": 16,
            "dropout": 0.0,
            "refresh_every_episodes": 5,
            "replay_buffer_path": str(tmp_path / "replay.jsonl"),
        },
    }
    if version_key is not None:
        cfg["turnrd"]["version"] = version_key
    trainer, _, _, _, _ = build_trainer_from_config(cfg, policy=_StubPolicy())
    assert isinstance(trainer.decomposer, TurnRDDecomposer)
    assert isinstance(trainer.decomposer.model, TurnRD), (
        f"v1 path must build TurnRD; got {type(trainer.decomposer.model).__name__}"
    )
    assert not isinstance(trainer.decomposer.model, TurnRDv2)


# ---------------------------------------------------------------------------
# 3. Unknown version is rejected
# ---------------------------------------------------------------------------


def test_build_turnrd_branch_unknown_version_raises(tmp_path: Path) -> None:
    cfg = _v2_cfg(tmp_path)
    cfg["turnrd"]["version"] = "v3"  # nonexistent
    with pytest.raises(ValueError, match=r"turnrd\.version"):
        build_trainer_from_config(cfg, policy=_StubPolicy())


# ---------------------------------------------------------------------------
# 4. train_turnrd(version="v2") + ckpt round-trip into a TurnRDDecomposer
# ---------------------------------------------------------------------------


def _write_synthetic_replay(path: Path, *, n: int = 8, D: int = 16, seed: int = 0) -> None:
    """Build a small synthetic replay JSONL — same shape as the producer's."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(D, generator=g)
    rows = []
    for i in range(n):
        T = 2 + (i % 4)  # T_i ∈ {2, 3, 4, 5}
        embeds = torch.randn(T, D, generator=g)
        R = float((embeds @ w).mean().item())
        rows.append({
            "task_id": f"task-{i}",
            "turn_embeds": embeds.tolist(),
            "final_reward": R,
        })
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _make_v2_model(input_dim: int = 16, seed: int = 0) -> TurnRDv2:
    torch.manual_seed(seed)
    return TurnRDv2(
        TurnRDv2Config(
            n_layers=2,
            hidden_size=32,
            n_heads=4,
            max_turns=16,
            dropout=0.0,
        ),
        input_dim=input_dim,
    )


def test_train_turnrd_v2_runs_and_ckpt_round_trips(tmp_path: Path) -> None:
    """v2 standalone-fitter end-to-end: synthetic replay → train one
    epoch → saved ckpt loads cleanly into a freshly-built decomposer."""
    replay = tmp_path / "replay.jsonl"
    _write_synthetic_replay(replay, n=8, D=16, seed=11)

    model = _make_v2_model(input_dim=16, seed=11)
    ckpt = tmp_path / "v2_ckpt.pt"
    summary = train_turnrd(
        replay,
        mode=1,
        model=model,
        version="v2",
        n_epochs=1,
        batch_size=4,
        lr=1e-3,
        log_every=0,
        ckpt_path=ckpt,
        # v2 first-sweep loss mix: pred + λ_rank·rank + λ_progress·progress.
        # value loss disabled until the per-turn target wires in.
        lambda_value=0.0,
        lambda_rank=0.1,
        lambda_progress=0.01,
        rank_margin=0.1,
    )
    assert summary["version"] == "v2"
    assert summary["n_steps"] > 0
    assert summary["ckpt_path"] == str(ckpt)
    # v2 component losses were tracked in the summary.
    breakdown = summary["final_loss_breakdown"]
    assert "v2_pred_loss" in breakdown
    assert "v2_rank_loss" in breakdown
    assert "v2_progress_loss" in breakdown

    # Round-trip: load the v2 ckpt into a freshly-built decomposer.
    fresh_model = _make_v2_model(input_dim=16, seed=99)  # different init
    decomposer = TurnRDDecomposer(
        model=fresh_model,
        embedder=lambda traj: torch.zeros(len(traj.turns), 16),
    )
    sd = torch.load(ckpt, weights_only=True)
    incompat = decomposer.load_state_dict(sd, strict=True)
    # `IncompatibleKeys(missing_keys=[], unexpected_keys=[])` ⇒ all keys aligned.
    assert list(incompat.missing_keys) == []
    assert list(incompat.unexpected_keys) == []
    # And the loaded weights actually replaced the fresh init.
    for p_loaded, p_orig in zip(fresh_model.parameters(), model.parameters()):
        assert torch.allclose(p_loaded, p_orig.detach()), (
            "Loaded v2 ckpt did not match the trained model's parameters byte-for-byte."
        )


# ---------------------------------------------------------------------------
# 5. v2 + mode=2 is rejected
# ---------------------------------------------------------------------------


def test_train_turnrd_v2_mode_2_raises(tmp_path: Path) -> None:
    """v2 doesn't have a Mode-2 distillation path; the trainer must
    reject it loudly rather than silently fall back to v1 semantics."""
    replay = tmp_path / "replay.jsonl"
    _write_synthetic_replay(replay, n=4, D=16, seed=0)
    model = _make_v2_model()
    with pytest.raises(ValueError, match=r"version='v2'.*mode=1"):
        train_turnrd(replay, mode=2, model=model, version="v2", n_epochs=1)
