"""Unit tests for `src.algorithms.hgpo.decomposers.residual.ResidualDecomposer`.

Method D — Residual Decomposer. Mirrors test_turnrd_decomposer.py.

Verification matrix:
1. `test_residual_decomposer_returns_correct_shape_and_invariant`
2. `test_residual_decomposer_uses_raw_env_reward_per_turn`
3. `test_residual_decomposer_gamma_zero_matches_turnrd_with_zero_bias`
4. `test_residual_decomposer_gamma_grad_flows`
5. `test_residual_decomposer_handles_empty_trajectory`
6. `test_build_decomposer_residual_dispatches_to_residual_class`
7. `test_residual_decomposer_state_dict_round_trips_gamma`
8. `test_residual_decomposer_parameters_includes_gamma`

Skipped cleanly on hosts without torch.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.algorithms.grpo.rollout import (  # noqa: E402
    Trajectory,
    TrajectoryGroup,
    TurnRecord,
)
from src.algorithms.hgpo.decomposers import build_decomposer
from src.algorithms.hgpo.decomposers.residual import ResidualDecomposer
from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer
from src.turnrd.model import TurnRD, TurnRDConfig


INPUT_DIM = 16


def _make_model(input_dim: int = INPUT_DIM, max_turns: int = 32) -> TurnRD:
    torch.manual_seed(0)
    cfg = TurnRDConfig(
        n_layers=2,
        hidden_size=32,
        n_heads=4,
        max_turns=max_turns,
        dropout=0.0,
    )
    return TurnRD(cfg, input_dim=input_dim)


def _traj(
    task_id: str,
    n_turns: int,
    final_reward: float,
    *,
    prefix: str = "act",
    raw_rewards: list[float] | None = None,
) -> Trajectory:
    if raw_rewards is None:
        raw_rewards = [0.0] * n_turns
    assert len(raw_rewards) == n_turns
    turns = [
        TurnRecord(
            turn_idx=t,
            observation_text=f"obs-{prefix}-{t}",
            action_text=f"{prefix}-t{t}",
            raw_env_reward=float(raw_rewards[t]),
        )
        for t in range(n_turns)
    ]
    return Trajectory(
        task_id=task_id,
        env_name="webshop",
        turns=turns,
        final_reward=final_reward,
    )


def _make_group(
    n_turns_per_traj: list[int],
    rewards: list[float],
    *,
    raw_rewards_per_traj: list[list[float]] | None = None,
) -> TrajectoryGroup:
    assert len(n_turns_per_traj) == len(rewards)
    raws = raw_rewards_per_traj or [None] * len(rewards)
    trajs = [
        _traj("task-X", T_i, r, prefix=f"k{i}", raw_rewards=raws[i])
        for i, (T_i, r) in enumerate(zip(n_turns_per_traj, rewards))
    ]
    return TrajectoryGroup(task_id="task-X", env_name="webshop", trajectories=trajs)


def _deterministic_embedder():
    """Same shape as test_turnrd_decomposer's helper."""
    call_log: list[Trajectory] = []

    def embedder(traj: Trajectory) -> torch.Tensor:
        call_log.append(traj)
        T_i = len(traj.turns)
        offset = (
            abs(hash(traj.task_id + "|" + (traj.turns[0].action_text if traj.turns else "")))
            % 997
        ) * 1.0
        base = torch.arange(T_i * INPUT_DIM, dtype=torch.float32).view(T_i, INPUT_DIM)
        return base + offset

    return embedder, call_log


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_residual_decomposer_returns_correct_shape_and_invariant() -> None:
    """Σ_t r̂_t == R per row (within 1e-5)."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = ResidualDecomposer(model=model, embedder=embedder)

    n_turns = [2, 3, 5]
    rewards = [1.0, 0.5, -0.4]
    raws = [[0.1, 0.2], [0.0, 0.3, 0.0], [0.0, 0.0, 0.5, 0.0, 0.4]]
    group = _make_group(n_turns, rewards, raw_rewards_per_traj=raws)

    out = decomposer.decompose(group)

    assert isinstance(out, list)
    assert len(out) == len(rewards)
    for i, (T_i, r) in enumerate(zip(n_turns, rewards)):
        assert len(out[i]) == T_i
        for x in out[i]:
            assert isinstance(x, float)
            assert x == x  # NaN check
        s = sum(out[i])
        assert abs(s - r) < 1e-5, f"row {i}: Σ_t r̂_t={s} != R={r}"


def test_residual_decomposer_uses_raw_env_reward_per_turn() -> None:
    """Manipulating `turn.raw_env_reward` must change α (vs identical
    group with all raw_env_reward=0)."""
    embedder, _ = _deterministic_embedder()
    # Use a large gamma so the prior dominates and the difference is
    # easy to detect even with a tiny model.
    model_a = _make_model()
    decomp_a = ResidualDecomposer(model=model_a, embedder=embedder, init_gamma=10.0)

    # Same model state for the contrast group (zero raw rewards).
    embedder_b, _ = _deterministic_embedder()
    model_b = _make_model()
    decomp_b = ResidualDecomposer(model=model_b, embedder=embedder_b, init_gamma=10.0)

    n_turns = [4]
    rewards = [1.0]
    # Heavy concentration on t=2 so Method D should bias α toward t=2.
    group_with_progress = _make_group(
        n_turns, rewards, raw_rewards_per_traj=[[0.0, 0.0, 1.0, 0.0]]
    )
    group_zero_progress = _make_group(
        n_turns, rewards, raw_rewards_per_traj=[[0.0, 0.0, 0.0, 0.0]]
    )

    out_with = decomp_a.decompose(group_with_progress)
    out_zero = decomp_b.decompose(group_zero_progress)

    # α at t=2 should be larger when raw_env_reward[2]=1 than when it's 0.
    # The output is `α * R = α * 1 = α`.
    assert out_with[0][2] > out_zero[0][2], (
        f"raw_env_reward at t=2 should bias α[2] up; got "
        f"with={out_with[0][2]}, zero={out_zero[0][2]}"
    )


def test_residual_decomposer_gamma_zero_matches_turnrd_with_zero_bias() -> None:
    """`init_gamma=0` reduces Method D to Method B (same α as TurnRD)."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()

    # Build separate but identically-initialized models so internal state
    # (eval mode, etc.) doesn't drift between calls.
    decomp_residual = ResidualDecomposer(model=model, embedder=embedder, init_gamma=0.0)

    embedder_b, _ = _deterministic_embedder()
    model_b = _make_model()  # same seed → same weights
    decomp_turnrd = TurnRDDecomposer(model=model_b, embedder=embedder_b)

    n_turns = [3, 4]
    rewards = [1.0, 0.5]
    raws = [[0.1, 0.2, 0.3], [0.0, 0.5, 0.0, 0.4]]
    group = _make_group(n_turns, rewards, raw_rewards_per_traj=raws)

    out_res = decomp_residual.decompose(group)
    out_b = decomp_turnrd.decompose(group)

    for i in range(len(n_turns)):
        a = torch.tensor(out_res[i])
        b = torch.tensor(out_b[i])
        assert torch.allclose(a, b, atol=1e-6), (
            f"row {i}: gamma=0 Residual should match Method B TurnRD; "
            f"residual={a.tolist()}, turnrd={b.tolist()}"
        )


def test_residual_decomposer_gamma_grad_flows() -> None:
    """Backward through `decompose_with_grad` populates `gamma_prior.grad`."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = ResidualDecomposer(model=model, embedder=embedder, init_gamma=1.0)
    model.train()

    n_turns = [3, 2]
    rewards = [1.0, 0.5]
    raws = [[0.1, 0.5, 0.2], [0.3, 0.0]]
    group = _make_group(n_turns, rewards, raw_rewards_per_traj=raws)

    out = decomposer.decompose_with_grad(group)
    assert out["alpha"].requires_grad

    # Position-weighted sum (alpha.sum() is constant per softmax → 0 grad).
    weights = torch.arange(out["alpha"].shape[1], dtype=out["alpha"].dtype)
    (out["alpha"] * weights).sum().backward()

    assert decomposer.gamma_prior.grad is not None
    assert decomposer.gamma_prior.grad.abs().item() > 0.0, (
        f"gamma_prior should receive non-zero gradient; got "
        f"{decomposer.gamma_prior.grad.item()}"
    )


def test_residual_decomposer_handles_empty_trajectory() -> None:
    """Trajectory with `turns=[]` returns `[]` and the model is NOT called."""
    model = _make_model()
    embedder, call_log = _deterministic_embedder()
    decomposer = ResidualDecomposer(model=model, embedder=embedder)

    trajs = [
        _traj("task-X", 0, 0.0, prefix="k0-empty"),
        _traj("task-X", 3, 1.0, prefix="k1", raw_rewards=[0.1, 0.2, 0.3]),
        _traj("task-X", 2, -0.5, prefix="k2", raw_rewards=[0.0, 0.4]),
    ]
    group = TrajectoryGroup(task_id="task-X", env_name="webshop", trajectories=trajs)

    out = decomposer.decompose(group)

    assert out[0] == []
    assert len(out[1]) == 3
    assert len(out[2]) == 2
    assert len(call_log) == 2  # only the two non-empty trajs

    # Σ invariant still holds for non-empty rows.
    assert abs(sum(out[1]) - 1.0) < 1e-5
    assert abs(sum(out[2]) - (-0.5)) < 1e-5


def test_residual_decomposer_handles_all_empty_group() -> None:
    """K=0 trajectories returns []."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = ResidualDecomposer(model=model, embedder=embedder)
    group = TrajectoryGroup(task_id="task-X", env_name="webshop", trajectories=[])
    assert decomposer.decompose(group) == []


def test_build_decomposer_residual_dispatches_to_residual_class() -> None:
    """`build_decomposer({"hgpo": {"decomposer": "residual"}, ...}, model=..., embedder=...)`
    returns a `ResidualDecomposer` with `gamma_prior` exposed in `parameters()`."""
    cfg = {"hgpo": {"decomposer": "residual"}, "residual": {"init_gamma": 0.5}}

    with pytest.raises(ValueError, match=r"both `model` and `embedder`"):
        build_decomposer(cfg)

    embedder, _ = _deterministic_embedder()
    fn = build_decomposer(cfg, model=_make_model(), embedder=embedder)
    assert isinstance(fn, ResidualDecomposer)
    # Identity check: gamma_prior is yielded by parameters().
    assert any(p is fn.gamma_prior for p in fn.parameters())
    # init_gamma honored.
    assert abs(fn.gamma_prior.detach().item() - 0.5) < 1e-6

    # Sanity: callable contract still works.
    group = _make_group([2, 3], [0.7, -0.3], raw_rewards_per_traj=[[0.1, 0.2], [0.0, 0.0, 0.0]])
    out = fn(group)
    assert len(out) == 2
    assert abs(sum(out[0]) - 0.7) < 1e-5
    assert abs(sum(out[1]) - (-0.3)) < 1e-5


def test_residual_decomposer_state_dict_round_trips_gamma() -> None:
    """`state_dict()` includes `gamma_prior`; `load_state_dict()` restores it."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = ResidualDecomposer(model=model, embedder=embedder, init_gamma=2.5)

    sd = decomposer.state_dict()
    assert "gamma_prior" in sd
    assert abs(float(sd["gamma_prior"].item()) - 2.5) < 1e-6

    # Build a fresh decomposer with a different init, then load.
    model2 = _make_model()
    embedder2, _ = _deterministic_embedder()
    decomposer2 = ResidualDecomposer(model=model2, embedder=embedder2, init_gamma=0.0)
    decomposer2.load_state_dict(sd)
    assert abs(decomposer2.gamma_prior.detach().item() - 2.5) < 1e-6


def test_residual_decomposer_load_state_dict_back_compat() -> None:
    """Legacy TurnRD-only state_dict (no `gamma_prior` key) loads cleanly."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = ResidualDecomposer(model=model, embedder=embedder, init_gamma=1.0)

    legacy_sd = dict(model.state_dict())  # NO gamma_prior key
    decomposer.load_state_dict(legacy_sd)
    # gamma_prior should be left at its current value (1.0).
    assert abs(decomposer.gamma_prior.detach().item() - 1.0) < 1e-6


def test_residual_decomposer_parameters_includes_gamma() -> None:
    """`parameters()` yields TurnRD model params PLUS `gamma_prior`."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = ResidualDecomposer(model=model, embedder=embedder)

    decomp_params = list(decomposer.parameters())
    model_params = list(model.parameters())
    assert len(decomp_params) == len(model_params) + 1
    # The extra parameter is gamma_prior (last).
    assert decomp_params[-1] is decomposer.gamma_prior


def test_residual_decomposer_advertises_learnable_params() -> None:
    """`has_learnable_params` is True (mirrors TurnRDDecomposer)."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = ResidualDecomposer(model=model, embedder=embedder)
    assert decomposer.has_learnable_params is True


def test_residual_decomposer_is_directly_callable() -> None:
    """`__call__` forwards to `.decompose`."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = ResidualDecomposer(model=model, embedder=embedder)
    group = _make_group([2, 3], [0.5, -0.3], raw_rewards_per_traj=[[0.1, 0.2], [0.0, 0.3, 0.0]])
    out_a = decomposer(group)
    out_b = decomposer.decompose(group)
    assert out_a == out_b
