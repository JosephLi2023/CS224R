"""Unit tests for `src.turnrd.train.train_turnrd` (Day 13 — standalone trainer).

Verification matrix:
1. `test_train_mode_1_loss_decreases_on_synthetic`
2. `test_train_mode_2_loss_decreases_on_synthetic`
3. `test_checkpoint_round_trip_into_fresh_model`

Skipped cleanly on hosts without torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.turnrd.model import TurnRD, TurnRDConfig  # noqa: E402
from src.turnrd.train import train_turnrd  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 16


def _make_model(seed: int = 0) -> TurnRD:
    torch.manual_seed(seed)
    cfg = TurnRDConfig(
        n_layers=2, hidden_size=32, n_heads=4, max_turns=16, dropout=0.0
    )
    return TurnRD(cfg, input_dim=INPUT_DIM)


def _write_synthetic_replay(
    path: Path, *, n: int = 8, mode: int, D: int = INPUT_DIM, seed: int = 0
) -> None:
    """Build a learnable synthetic replay buffer.

    Mode 1: R = mean over turns of (w · embed) — the model learns to map
    α_t and r_head such that predicted_R ≈ R.
    Mode 2: judge_labels = α* · R for a fixed-random α* in Δ^{T-1} — the
    model learns the [CLS] attention to match the teacher.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(D, generator=g)
    rows = []
    for i in range(n):
        T = 2 + (i % 4)  # T_i ∈ {2, 3, 4, 5}
        embeds = torch.randn(T, D, generator=g)
        R = float((embeds @ w).mean().item())
        row = {
            "task_id": f"task-{i}",
            "turn_embeds": embeds.tolist(),
            "final_reward": R,
        }
        if mode == 2:
            alpha_star = torch.softmax(torch.randn(T, generator=g), dim=-1)
            row["judge_labels"] = (alpha_star * R).tolist()
        rows.append(row)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_mode_1_loss_decreases_on_synthetic(tmp_path: Path) -> None:
    """5 epochs over 8 trajectories drops Mode 1 loss ≥ 30%."""
    replay = tmp_path / "replay.jsonl"
    _write_synthetic_replay(replay, n=8, mode=1, seed=11)
    model = _make_model(seed=11)
    out = train_turnrd(
        replay,
        mode=1,
        model=model,
        n_epochs=5,
        batch_size=4,
        lr=1e-3,
        log_every=0,
    )
    assert out["n_steps"] > 0
    assert out["final_loss"] <= 0.7 * out["initial_loss"], (
        f"Mode 1 loss did not decrease enough: "
        f"initial={out['initial_loss']:.4f}, final={out['final_loss']:.4f}"
    )
    assert out["ckpt_path"] is None


def test_train_mode_2_loss_decreases_on_synthetic(tmp_path: Path) -> None:
    """Same harness with Mode 2 (judge label distillation) — loss drops ≥ 30%."""
    replay = tmp_path / "replay.jsonl"
    _write_synthetic_replay(replay, n=8, mode=2, seed=22)
    model = _make_model(seed=22)
    out = train_turnrd(
        replay,
        mode=2,
        model=model,
        n_epochs=5,
        batch_size=4,
        lr=1e-3,
        log_every=0,
    )
    assert out["n_steps"] > 0
    assert out["final_loss"] <= 0.7 * out["initial_loss"], (
        f"Mode 2 loss did not decrease enough: "
        f"initial={out['initial_loss']:.4f}, final={out['final_loss']:.4f}"
    )


def test_checkpoint_round_trip_into_fresh_model(tmp_path: Path) -> None:
    """ckpt_path persists state_dict; loading into a fresh model preserves
    every parameter byte-for-byte."""
    replay = tmp_path / "replay.jsonl"
    ckpt = tmp_path / "turnrd.pt"
    _write_synthetic_replay(replay, n=4, mode=1, seed=33)

    model = _make_model(seed=33)
    out = train_turnrd(
        replay,
        mode=1,
        model=model,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        log_every=0,
        ckpt_path=ckpt,
    )
    assert out["ckpt_path"] == str(ckpt)
    assert ckpt.is_file()

    fresh = _make_model(seed=999)  # different init
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    fresh.load_state_dict(sd)

    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), fresh.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(v1.cpu(), v2.cpu()), f"param {k1} differs after reload"


# ---------------------------------------------------------------------------
# Method D — gamma_prior path
# ---------------------------------------------------------------------------


def _write_synthetic_replay_with_progress(
    path: Path, *, n: int = 8, D: int = INPUT_DIM, seed: int = 0
) -> None:
    """Mode-1 replay with per-turn `progress` (raw_env_reward) populated.

    Used to exercise the Method-D residual path through the standalone
    fitter end-to-end.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(D, generator=g)
    rows = []
    for i in range(n):
        T = 2 + (i % 4)
        embeds = torch.randn(T, D, generator=g)
        R = float((embeds @ w).mean().item())
        # Progress: a noisy non-uniform per-turn signal so the prior
        # actually shifts α away from uniform.
        progress = torch.softmax(torch.randn(T, generator=g), dim=-1).tolist()
        rows.append({
            "task_id": f"task-{i}",
            "turn_embeds": embeds.tolist(),
            "final_reward": R,
            "progress": progress,
        })
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_train_residual_updates_gamma_prior(tmp_path: Path) -> None:
    """When `gamma_prior` is passed AND the replay carries `progress`,
    the standalone fitter's AdamW must move `gamma_prior` away from
    its initial value across ≥1 step."""
    replay = tmp_path / "replay.jsonl"
    _write_synthetic_replay_with_progress(replay, n=8, seed=44)

    model = _make_model(seed=44)
    gamma = torch.nn.Parameter(torch.tensor(1.0))
    initial_gamma = float(gamma.detach().item())

    out = train_turnrd(
        replay,
        mode=1,
        model=model,
        n_epochs=2,
        batch_size=4,
        lr=1e-2,  # bigger lr so gamma definitely moves in 4 steps
        log_every=0,
        gamma_prior=gamma,
    )
    assert out["n_steps"] > 0
    final_gamma = float(gamma.detach().item())
    assert abs(final_gamma - initial_gamma) > 1e-5, (
        f"gamma_prior should move under SGD; initial={initial_gamma}, "
        f"final={final_gamma}"
    )
    # Summary surfaces the final gamma value.
    assert out["gamma_prior"] == pytest.approx(final_gamma, abs=1e-6)


def test_train_residual_ckpt_includes_gamma_prior(tmp_path: Path) -> None:
    """Saved ckpt bundles `gamma_prior` so `ResidualDecomposer.load_state_dict`
    can round-trip both the model and the scalar."""
    replay = tmp_path / "replay.jsonl"
    ckpt = tmp_path / "residual.pt"
    _write_synthetic_replay_with_progress(replay, n=4, seed=55)

    model = _make_model(seed=55)
    gamma = torch.nn.Parameter(torch.tensor(1.5))

    train_turnrd(
        replay,
        mode=1,
        model=model,
        n_epochs=1,
        batch_size=2,
        lr=1e-2,
        log_every=0,
        gamma_prior=gamma,
        ckpt_path=ckpt,
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert "gamma_prior" in sd
    assert torch.allclose(
        sd["gamma_prior"].detach().cpu(), gamma.detach().cpu(), atol=1e-6
    )


def test_train_residual_no_progress_in_data_acts_like_method_b(tmp_path: Path) -> None:
    """When `gamma_prior` is supplied but the replay rows lack `progress`,
    the fitter MUST NOT crash and gamma stays put (no signal to update it)."""
    replay = tmp_path / "replay.jsonl"
    _write_synthetic_replay(replay, n=4, mode=1, seed=66)  # NO progress field

    model = _make_model(seed=66)
    gamma = torch.nn.Parameter(torch.tensor(1.0))
    initial_gamma = float(gamma.detach().item())

    out = train_turnrd(
        replay,
        mode=1,
        model=model,
        n_epochs=1,
        batch_size=2,
        lr=1e-2,
        log_every=0,
        gamma_prior=gamma,
    )
    # Gamma should not have moved (no prior signal in batches).
    assert float(gamma.detach().item()) == pytest.approx(initial_gamma, abs=1e-6)
    assert out["gamma_prior"] is not None
