"""Unit tests for `src.algorithms.hgpo.decomposers.turnrd.TurnRDDecomposer`.

Verifies the inference-only adapter that the trainer plugs in:

1. `test_turnrd_decomposer_returns_correct_shape_and_invariant`
2. `test_turnrd_decomposer_handles_empty_trajectory`
3. `test_turnrd_decomposer_uses_pluggable_embedder`
4. `test_turnrd_decomposer_padding_does_not_affect_real_turns`
5. `test_build_decomposer_turnrd_requires_model_and_embedder`

Skipped cleanly on hosts without torch.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.algorithms.grpo.rollout import (  # noqa: E402  (after importorskip gate)
    Trajectory,
    TrajectoryGroup,
    TurnRecord,
)
from src.algorithms.hgpo.decomposers import build_decomposer
from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer
from src.turnrd.model import TurnRD, TurnRDConfig


# ---------------------------------------------------------------------------
# Helpers (adapted from tests/unit/test_openai_judge_decomposer.py::_make_group)
# ---------------------------------------------------------------------------

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


def _traj(task_id: str, n_turns: int, final_reward: float, prefix: str = "act") -> Trajectory:
    turns = [
        TurnRecord(
            turn_idx=t,
            observation_text=f"obs-{prefix}-{t}",
            action_text=f"{prefix}-t{t}",
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
) -> TrajectoryGroup:
    assert len(n_turns_per_traj) == len(rewards)
    trajs = [
        _traj("task-X", T_i, r, prefix=f"k{i}")
        for i, (T_i, r) in enumerate(zip(n_turns_per_traj, rewards))
    ]
    return TrajectoryGroup(task_id="task-X", env_name="webshop", trajectories=trajs)


def _deterministic_embedder() -> "tuple[callable, list[Trajectory]]":
    """Returns (embedder, call_log).

    The embedder produces `torch.arange(T_i * D).view(T_i, D).float()` per
    trajectory and appends each call's traj to the log so tests can assert
    call counts and ordering.
    """
    call_log: list[Trajectory] = []

    def embedder(traj: Trajectory) -> torch.Tensor:
        call_log.append(traj)
        T_i = len(traj.turns)
        # Distinct tensor per trajectory by mixing in the task_id hash so two
        # K-samples with the same T_i don't collide. INPUT_DIM defined above.
        offset = (abs(hash(traj.task_id + "|" + (traj.turns[0].action_text if traj.turns else ""))) % 997) * 1.0
        base = torch.arange(T_i * INPUT_DIM, dtype=torch.float32).view(T_i, INPUT_DIM)
        return base + offset

    return embedder, call_log


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_turnrd_decomposer_returns_correct_shape_and_invariant() -> None:
    """3-trajectory group with varying T_i and final_reward.

    Asserts:
    - Output is `list[K]` of `list[T_i]`.
    - `Σ_t out[i] == final_reward[i]` per row (within 1e-6).
    - No NaN.
    """
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = TurnRDDecomposer(model=model, embedder=embedder)

    n_turns = [2, 3, 5]
    rewards = [1.0, 0.5, -0.4]
    group = _make_group(n_turns, rewards)

    out = decomposer.decompose(group)

    assert isinstance(out, list)
    assert len(out) == len(rewards)
    for i, (T_i, r) in enumerate(zip(n_turns, rewards)):
        assert len(out[i]) == T_i, f"row {i}: expected len={T_i}, got {len(out[i])}"
        for x in out[i]:
            assert isinstance(x, float)
            assert x == x  # NaN check (NaN != NaN)
        s = sum(out[i])
        assert abs(s - r) < 1e-5, f"row {i}: Σ_t r̂_t={s} != R={r}"


def test_turnrd_decomposer_handles_empty_trajectory() -> None:
    """Trajectory with `turns=[]` returns `[]` and the model is NOT called
    for it (the embedder is also NOT invoked for that K-slot)."""
    model = _make_model()
    embedder, call_log = _deterministic_embedder()
    decomposer = TurnRDDecomposer(model=model, embedder=embedder)

    # Mix one empty traj with two non-empty ones.
    trajs = [
        _traj("task-X", 0, 0.0, prefix="k0-empty"),
        _traj("task-X", 3, 1.0, prefix="k1"),
        _traj("task-X", 2, -0.5, prefix="k2"),
    ]
    group = TrajectoryGroup(task_id="task-X", env_name="webshop", trajectories=trajs)

    out = decomposer.decompose(group)

    assert out[0] == []
    assert len(out[1]) == 3
    assert len(out[2]) == 2
    # Embedder must have been called exactly twice (for the two non-empty trajs).
    assert len(call_log) == 2
    assert call_log[0].turns[0].action_text.startswith("k1")
    assert call_log[1].turns[0].action_text.startswith("k2")
    # Invariant still holds for the non-empty rows.
    assert abs(sum(out[1]) - 1.0) < 1e-5
    assert abs(sum(out[2]) - (-0.5)) < 1e-5


def test_turnrd_decomposer_uses_pluggable_embedder() -> None:
    """Stub embedder counts calls and records traj IDs; assert one call per
    non-empty trajectory and that the recorded action_texts match input."""
    model = _make_model()
    embedder, call_log = _deterministic_embedder()
    decomposer = TurnRDDecomposer(model=model, embedder=embedder)

    n_turns = [2, 4, 1]
    rewards = [0.1, 0.2, 0.3]
    group = _make_group(n_turns, rewards)

    decomposer.decompose(group)

    assert len(call_log) == 3
    # Order of calls matches order of trajectories in the group.
    for i, traj in enumerate(call_log):
        assert traj.turns[0].action_text == f"k{i}-t0"


def test_turnrd_decomposer_padding_does_not_affect_real_turns() -> None:
    """Same trajectory yields identical per-turn rewards regardless of
    whether it is co-batched with a longer (padded-out) neighbor.

    Defensive check: confirms padding leakage is not contaminating the
    real-turn outputs of the shorter trajectory.
    """
    model = _make_model()
    model.eval()
    embedder, _ = _deterministic_embedder()
    decomposer = TurnRDDecomposer(model=model, embedder=embedder)

    # Two trajectories: short (T=2) and long (T=5), with the same final
    # reward so we can attribute any difference cleanly to padding effects.
    short = _traj("task-X", 2, 1.0, prefix="k0")
    long = _traj("task-X", 5, 1.0, prefix="k1")

    co_batched = TrajectoryGroup(
        task_id="task-X", env_name="webshop", trajectories=[short, long]
    )
    out_co = decomposer.decompose(co_batched)

    alone = TrajectoryGroup(
        task_id="task-X", env_name="webshop", trajectories=[short]
    )
    out_alone = decomposer.decompose(alone)

    # The short trajectory's per-turn rewards should match exactly (or
    # within fp32 numerical noise) regardless of co-batching.
    a = torch.tensor(out_co[0])
    b = torch.tensor(out_alone[0])
    assert torch.allclose(a, b, atol=1e-6), (
        f"padding leaked into short trajectory: co-batched={a.tolist()}, "
        f"alone={b.tolist()}"
    )


def test_build_decomposer_turnrd_requires_model_and_embedder() -> None:
    """build_decomposer({"hgpo": {"decomposer": "turnrd"}}) without kwargs
    raises ValueError mentioning both `model` and `embedder`. With both
    kwargs, returns a callable whose output respects the §3.2 invariant."""
    cfg = {"hgpo": {"decomposer": "turnrd"}}

    with pytest.raises(ValueError, match=r"both `model` and `embedder`"):
        build_decomposer(cfg)

    with pytest.raises(ValueError, match=r"both `model` and `embedder`"):
        build_decomposer(cfg, model=_make_model())

    embedder, _ = _deterministic_embedder()
    with pytest.raises(ValueError, match=r"both `model` and `embedder`"):
        build_decomposer(cfg, embedder=embedder)

    fn = build_decomposer(cfg, model=_make_model(), embedder=embedder)
    assert callable(fn)

    group = _make_group([2, 3], [0.7, -0.3])
    out = fn(group)
    assert len(out) == 2
    assert abs(sum(out[0]) - 0.7) < 1e-5
    assert abs(sum(out[1]) - (-0.3)) < 1e-5


# ---------------------------------------------------------------------------
# Hardening regressions (post-review)
# ---------------------------------------------------------------------------


def test_turnrd_decomposer_defaults_device_from_model_params() -> None:
    """When `device=None`, the adapter must read the device from the model's
    parameters rather than from the embedder's tensor. This prevents the
    `TurnRDDecomposer(cuda_model, cpu_embedder)` foot-gun from silently
    landing tensors on CPU and crashing inside `model.input_proj` with a
    device-mismatch error.
    """
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = TurnRDDecomposer(model=model, embedder=embedder, device=None)

    expected = next(model.parameters()).device
    assert decomposer.device == expected


def test_turnrd_decomposer_casts_stacked_to_model_dtype() -> None:
    """Embedder may return a different dtype than the model's parameters.
    The adapter must cast `stacked` to the model dtype before forward;
    otherwise `nn.Linear(input_proj)` would raise dtype-mismatch on the
    Day-14 production wiring (LoRAPolicy fp16/bf16 hidden states feeding
    a fp32 TurnRD).
    """
    model = _make_model()  # fp32 by default
    expected_dtype = next(model.parameters()).dtype
    assert expected_dtype == torch.float32

    # Embedder returns fp64 to make the cast visible.
    seen_dtypes: list[torch.dtype] = []

    def fp64_embedder(traj: Trajectory) -> torch.Tensor:
        T_i = len(traj.turns)
        return torch.arange(T_i * INPUT_DIM, dtype=torch.float64).view(T_i, INPUT_DIM)

    # Hook the model forward to record the dtype of the input tensor at
    # the moment forward runs.
    orig_forward = model.forward

    def recording_forward(turn_embeds: torch.Tensor, attention_mask: torch.Tensor):
        seen_dtypes.append(turn_embeds.dtype)
        return orig_forward(turn_embeds, attention_mask)

    model.forward = recording_forward  # type: ignore[method-assign]

    decomposer = TurnRDDecomposer(model=model, embedder=fp64_embedder)
    decomposer.decompose(_make_group([2, 3], [1.0, 0.5]))

    assert seen_dtypes == [expected_dtype], (
        f"adapter did not cast stacked to model dtype: saw {seen_dtypes}, "
        f"expected [{expected_dtype}]"
    )


def test_turnrd_decomposer_neutralises_grad_tracking_embedder() -> None:
    """A careless embedder that returns a `requires_grad=True` tensor must
    NOT propagate the autograd graph through to the adapter's outputs.
    The adapter wraps the embedder loop in `torch.no_grad()` and detaches
    each returned tensor as belt-and-suspenders against memory leaks
    (see `decomposers/turnrd.py` "Contract requirements (c)").
    """
    model = _make_model()

    seen_requires_grad: list[bool] = []

    def grad_tracking_embedder(traj: Trajectory) -> torch.Tensor:
        T_i = len(traj.turns)
        # Build outside no_grad — would normally produce a grad-tracking tensor.
        leaf = torch.arange(T_i * INPUT_DIM, dtype=torch.float32, requires_grad=True).view(T_i, INPUT_DIM)
        # A dependent computation that retains the graph IFF grad mode is on.
        out = leaf * 1.0
        seen_requires_grad.append(out.requires_grad)
        return out

    decomposer = TurnRDDecomposer(model=model, embedder=grad_tracking_embedder)
    out = decomposer.decompose(_make_group([2, 3], [1.0, 0.5]))

    # The embedder was invoked under `no_grad`, so `out.requires_grad` should
    # be False inside the embedder body.
    assert seen_requires_grad == [False, False], (
        f"embedder ran outside torch.no_grad(): saw requires_grad={seen_requires_grad}"
    )
    # Sanity: §3.2 invariant still holds.
    assert abs(sum(out[0]) - 1.0) < 1e-5
    assert abs(sum(out[1]) - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# Day-13 learnable surface
# ---------------------------------------------------------------------------


def test_turnrd_decomposer_advertises_learnable_params() -> None:
    """`has_learnable_params` is True for TurnRD; the trainer reads this via
    `getattr` so other decomposers (Methods A/C) inherit False."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = TurnRDDecomposer(model=model, embedder=embedder)
    assert decomposer.has_learnable_params is True
    # Sanity: the bare function `progress_decomposer` is the contrast case.
    from src.algorithms.grpo.trainer import progress_decomposer

    assert getattr(progress_decomposer, "has_learnable_params", False) is False


def test_turnrd_decomposer_parameters_match_model_parameters() -> None:
    """`decomposer.parameters()` yields the same tensors as `model.parameters()`."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = TurnRDDecomposer(model=model, embedder=embedder)
    a = list(decomposer.parameters())
    b = list(model.parameters())
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        # Same Parameter object, not just allclose.
        assert pa is pb


def test_turnrd_decomposer_decompose_with_grad_returns_grad_tracking_alpha() -> None:
    """`decompose_with_grad(group)["alpha"].requires_grad is True` and
    `.sum().backward()` populates non-zero `cls_query.grad`."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = TurnRDDecomposer(model=model, embedder=embedder)
    model.train()  # the trainer is responsible for this; mimic that here

    group = _make_group([2, 3], [1.0, 0.5])
    out = decomposer.decompose_with_grad(group)

    assert out["alpha"].requires_grad
    assert out["alpha"].shape == (2, 3)  # K_real=2, T_max=3
    assert out["attention_mask"].shape == (2, 3)
    assert out["nonempty_indices"] == [0, 1]
    assert out["final_R"].shape == (2,)

    # Backward through a non-trivial scalar must populate cls_query.grad.
    # Note: `alpha.sum()` is always ≈ K (each row sums to 1 by softmax),
    # so its gradient is 0. Use a position-weighted sum to inject a real
    # gradient signal into TurnRD's params.
    model.zero_grad(set_to_none=True)
    weights = torch.arange(out["alpha"].shape[1], dtype=out["alpha"].dtype)
    (out["alpha"] * weights).sum().backward()
    assert model.cls_query.grad is not None
    assert model.cls_query.grad.abs().sum().item() > 0.0


def test_turnrd_decomposer_is_directly_callable() -> None:
    """`__call__` forwards to `.decompose` so the trainer can pass the object
    directly as the `decomposer` arg and still call `self.decomposer(group)`."""
    model = _make_model()
    embedder, _ = _deterministic_embedder()
    decomposer = TurnRDDecomposer(model=model, embedder=embedder)
    group = _make_group([2, 3], [0.5, -0.3])
    out_a = decomposer(group)
    out_b = decomposer.decompose(group)
    assert out_a == out_b
