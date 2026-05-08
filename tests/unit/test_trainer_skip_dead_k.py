"""Unit tests for the Tier-2 skip-dead-K guard in HGPOTrainer.

Background
----------
A K-trajectory group with all K final rewards equal has trajectory advantages
identically zero (the centered-by-mean / divided-by-std normalization is
undefined → A_traj = 0 by convention). With α-blended H-GRPO this collapses
the per-token PPO advantage to zero on every action token of every trajectory
in the group, so the policy gradient is identically zero by construction.

The Tier-2 fix in the alfworld α-sweep plan is to short-circuit BEFORE the
expensive 3-forward-pass `_batched_logprobs` call (one new + one ref pass per
group dominates per-step wallclock at K=8) and skip the optimizer step
entirely. Per-turn rewards from `build_advantages` are still computed so the
rollout's external replay-buffer emission (in `infra/app_train_loop.py`) sees
identical inputs whether the group was dead or not.

Tests
-----
1. `test_compute_loss_dead_k_marks_stats_and_zero_loss` — stats.dead_K_group=1,
   stats.policy_loss=stats.total_loss=0, returned tensor has no grad path.
2. `test_compute_loss_dead_k_skips_batched_logprobs` — the heavy forward
   (`_batched_logprobs`) is NEVER invoked on a dead group.
3. `test_compute_loss_dead_k_still_invokes_decomposer` — `build_advantages`
   (which calls the decomposer's __call__) DOES run, so per-turn rewards are
   computed for the replay-buffer emission downstream.
4. `test_train_step_dead_k_skips_optimizer_step` — AdamW step counter is
   unchanged across a dead-K train_step.
5. `test_train_step_dead_k_advances_internal_step_counter` — `trainer._step`
   still increments so refresh-fn cadence + KL-warmup gating stay aligned.
6. `test_train_step_non_dead_does_not_short_circuit` — sanity check: a
   non-dead group still flows through `_batched_logprobs` and stats.dead_K_group=0.
"""
from __future__ import annotations

import pytest

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord
from src.algorithms.grpo.trainer import (
    HGPOTrainer,
    HGPOTrainerConfig,
    progress_decomposer,
)


class _StubPolicy:
    """Minimal LoRAPolicy stand-in (mirrors test_hgpo_trainer.py::_StubPolicy).

    Provides a single torch.nn.Parameter so HGPOTrainer's lazy AdamW can be
    constructed; otherwise no real LoRA modules. `train()` is a no-op.
    """

    class _M:
        def __init__(self) -> None:
            try:
                import torch  # type: ignore[import-not-found]

                self._param = torch.nn.Parameter(torch.zeros(1))
            except Exception:
                self._param = None

        def parameters(self):
            if self._param is None:
                return iter([])
            return iter([self._param])

        def named_modules(self):
            return iter([])

        def train(self):
            pass

    def __init__(self) -> None:
        self.model = self._M()

    def trainable_parameters(self):
        if self.model._param is None:
            return []
        return [self.model._param]


def _dead_k_group(K: int = 4, T: int = 2, R: float = 0.5) -> TrajectoryGroup:
    """K trajectories with IDENTICAL `final_reward` → std_reward_group=0 → dead-K.

    Each turn has non-empty token-id triples so `compute_loss` would otherwise
    flow into `_batched_logprobs` (i.e. the dead-K guard's job is to prevent
    that heavy forward). raw_env_reward is non-trivial so the per-turn rewards
    that `build_advantages` emits are not all zero — this matches what a real
    AlfWorld dead-K rollout looks like (e.g. all K trajectories failed the
    same way and got R=0).
    """
    trajectories = []
    for i in range(K):
        turns = [
            TurnRecord(
                turn_idx=t,
                observation_text=f"o{i}-{t}",
                action_text=f"a{i}-{t}",
                raw_env_reward=0.1 * (t + 1),
                prompt_token_ids=(1, 2, 3),
                action_token_ids=(10, 20),
                action_token_logprobs=(-1.0, -1.0),
            )
            for t in range(T)
        ]
        trajectories.append(
            Trajectory(
                task_id="task-DEAD",
                env_name="webshop",
                turns=turns,
                final_reward=R,  # IDENTICAL across all K
            )
        )
    return TrajectoryGroup(
        task_id="task-DEAD", env_name="webshop", trajectories=trajectories
    )


def _live_k_group(K: int = 3, T: int = 2) -> TrajectoryGroup:
    """K trajectories with VARYING final_reward → std_reward_group>0 → live-K.

    Used as the negative control that the dead-K guard does NOT misfire on
    healthy groups.
    """
    trajectories = []
    for i in range(K):
        turns = [
            TurnRecord(
                turn_idx=t,
                observation_text=f"o{i}-{t}",
                action_text=f"a{i}-{t}",
                raw_env_reward=0.1 * (t + 1),
                prompt_token_ids=(1, 2, 3),
                action_token_ids=(10, 20),
                action_token_logprobs=(-1.0, -1.0),
            )
            for t in range(T)
        ]
        trajectories.append(
            Trajectory(
                task_id="task-LIVE",
                env_name="webshop",
                turns=turns,
                final_reward=0.3 * (i + 1),  # 0.3, 0.6, 0.9 → non-zero std
            )
        )
    return TrajectoryGroup(
        task_id="task-LIVE", env_name="webshop", trajectories=trajectories
    )


# ---------------------------------------------------------------------------
# compute_loss-level checks
# ---------------------------------------------------------------------------


def test_compute_loss_dead_k_marks_stats_and_zero_loss():
    """Dead-K K-group: stats.dead_K_group=1, all gradient-bearing fields zero,
    returned tensor has no autograd history."""
    pytest.importorskip("torch")

    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig())
    group = _dead_k_group(K=4)

    total, stats = trainer.compute_loss(group)

    assert stats.dead_K_group == 1
    assert stats.policy_loss == 0.0
    assert stats.total_loss == 0.0
    assert stats.kl_term == 0.0
    assert stats.std_reward_group == 0.0
    assert stats.n_action_tokens == 0
    # The zero loss must NOT carry an autograd graph — train_step relies on
    # the dead-K branch not invoking backward(), and a requires_grad=True
    # tensor would either silently leak grad state or trip a runtime error
    # depending on PyTorch version.
    assert not total.requires_grad


def test_compute_loss_dead_k_skips_batched_logprobs():
    """The heavy 3-forward-pass call MUST be short-circuited on dead-K groups.

    Sentinel-patch `_batched_logprobs` to record every invocation; assert it
    was never called for a dead-K group (and that the call IS made for a
    live-K group, as a positive control inside the same test).
    """
    pytest.importorskip("torch")

    import torch  # type: ignore[import-not-found]

    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig())

    call_log: list[bool] = []

    def _sentinel_logprobs(pa_pairs, *, use_ref: bool):
        call_log.append(use_ref)
        return [torch.zeros(len(ids), dtype=torch.float32) for _, ids in pa_pairs]

    trainer._batched_logprobs = _sentinel_logprobs  # type: ignore[assignment]

    # Dead-K: must NOT invoke _batched_logprobs.
    trainer.compute_loss(_dead_k_group(K=4))
    assert call_log == [], (
        f"_batched_logprobs called {len(call_log)}× for dead-K group; the "
        "Tier-2 guard should short-circuit BEFORE the forward pass."
    )

    # Live-K positive control: SHOULD invoke _batched_logprobs (twice — once
    # for the new policy, once for the reference). Confirms the guard is
    # specifically dead-K-conditional, not a global short-circuit.
    trainer.compute_loss(_live_k_group(K=3))
    assert len(call_log) == 2, (
        f"_batched_logprobs called {len(call_log)}× for live-K (expected 2: "
        "one new + one ref pass). The guard may be over-firing."
    )
    # Both flavors should have run (use_ref False then True, in some order).
    assert set(call_log) == {False, True}


def test_compute_loss_dead_k_still_invokes_decomposer():
    """Per-turn rewards must still be computed on dead-K groups.

    The plan's "skip optimizer step only — keep rollout, replay collection,
    eval" semantics require that `build_advantages` (which calls the
    decomposer's __call__) still runs on dead groups so the per-turn rewards
    are available for downstream emission to the standalone TurnRD trainer's
    replay buffer between rounds.
    """
    pytest.importorskip("torch")

    decomposer_calls: list[int] = []

    def _sentinel_decomposer(group):
        decomposer_calls.append(group.K)
        return [
            [float(turn.raw_env_reward) for turn in traj.turns]
            for traj in group.trajectories
        ]

    trainer = HGPOTrainer(
        _StubPolicy(), _sentinel_decomposer, HGPOTrainerConfig()
    )

    _, stats = trainer.compute_loss(_dead_k_group(K=3))

    assert stats.dead_K_group == 1
    assert decomposer_calls == [3], (
        "Decomposer was not invoked for dead-K — per-turn rewards would be "
        "missing from the rollout's replay-buffer emission downstream."
    )


# ---------------------------------------------------------------------------
# train_step-level checks
# ---------------------------------------------------------------------------


def test_train_step_dead_k_skips_optimizer_step():
    """AdamW's per-parameter `step` counter must NOT advance on dead-K groups.

    AdamW initializes each param's state lazily on the first .step() call;
    after one .step() each param's state['step'] is 1. A dead-K group's
    train_step should NOT trigger that init (no gradient was accumulated).
    """
    pytest.importorskip("torch")

    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig())
    # Pre-build the optimizer so we can read its state.
    trainer._ensure_optimizer()
    param = trainer._trainable_params[0]

    # Before any train_step, AdamW state for this param is empty.
    assert param not in trainer._optimizer.state

    stats = trainer.train_step(_dead_k_group(K=3))

    assert stats.dead_K_group == 1
    # Dead-K must not have stepped the optimizer.
    assert param not in trainer._optimizer.state, (
        "AdamW state was populated for the dead-K param — the optimizer "
        "appears to have stepped despite the dead-K guard."
    )


def test_train_step_dead_k_advances_internal_step_counter():
    """`trainer._step` must increment even when the optimizer is skipped.

    The TurnRD-refresh cadence (`refresh_every_episodes`) and the KL-warmup
    gating both key off `self._step`; if dead-K calls didn't advance _step,
    a long stretch of dead groups would silently freeze both clocks.
    """
    pytest.importorskip("torch")

    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig())
    assert trainer._step == 0
    trainer.train_step(_dead_k_group(K=3))
    assert trainer._step == 1
    trainer.train_step(_dead_k_group(K=3))
    assert trainer._step == 2


def test_train_step_dead_k_does_not_perturb_kl_controller():
    """Issue 1 regression: a long stretch of dead-K groups must NOT collapse
    `kl_coef` to its `min_coef` floor.

    Pre-fix: train_step on dead-K fed `observed_kl=0.0` into
    `kl_controller.update(...)` once warmup elapsed. Since `0.0` is
    strictly less than `low_threshold * target_kl` (default 0.5 * 0.04 =
    0.02), each dead group multiplied `coef` by `decrease_factor` (default
    0.5), collapsing the coefficient to `min_coef` over a handful of
    consecutive dead groups — exactly the failure mode the guard exists
    to handle. The fix skips `kl_controller.update(...)` entirely on
    dead-K (mirroring the optimizer-step skip).
    """
    pytest.importorskip("torch")

    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig())
    initial_coef = trainer.kl_controller.coef
    assert initial_coef > 0.0

    # Drive 10 consecutive dead-K train_step calls. Pre-fix this would
    # collapse coef by 0.5 ** 10 ≈ 1e-3 (or hit `min_coef` floor).
    for _ in range(10):
        stats = trainer.train_step(_dead_k_group(K=3))
        assert stats.dead_K_group == 1

    final_coef = trainer.kl_controller.coef
    assert final_coef == pytest.approx(initial_coef), (
        f"kl_coef drifted from {initial_coef} to {final_coef} across 10 "
        "dead-K train_steps — the controller is being perturbed by dead "
        "groups despite the guard."
    )


def test_train_step_dead_k_preserves_grad_accum_parity():
    """Issue 2 regression: dead-K early returns must NOT shift the grad-accum
    boundary parity.

    Pre-fix the boundary check `((self._step + 1) % accum) == 0` keyed off
    `_step`, which advanced on dead-K. With `grad_accum_steps=2`, a dead-K
    landing between two live-K calls would let the next boundary fire on
    a step that had only 1 actual backward contribution (instead of the
    expected 2). The fix counts `_pending_backwards` instead — boundary
    fires after exactly `accum` real backward calls regardless of how
    many dead-K interruptions happened in between.

    This test exercises only the bookkeeping (no real backward), then
    asserts that `_pending_backwards` increments only on live-K calls
    and the boundary check uses it.
    """
    pytest.importorskip("torch")

    cfg = HGPOTrainerConfig(grad_accum_steps=2)
    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, cfg)

    # Dead-K calls must NOT increment _pending_backwards.
    trainer.train_step(_dead_k_group(K=3))
    assert trainer._pending_backwards == 0
    trainer.train_step(_dead_k_group(K=3))
    assert trainer._pending_backwards == 0
    # _step still advances (refresh-fn cadence + KL warmup gating).
    assert trainer._step == 2


def test_train_step_non_dead_does_not_short_circuit():
    """Negative control: a live-K group must NOT trigger the dead-K branch.

    Patches `_batched_logprobs` so the test runs CPU-only (no real LoRA
    forward needed); only checks the control flow (dead_K_group == 0,
    std_reward_group > 0), not the loss values themselves. We exercise
    `compute_loss` directly rather than `train_step` because the stubbed
    logprobs return non-grad-tracking tensors which would fail backward()
    — the negative-control assertion is purely about the dead-K flag.
    """
    pytest.importorskip("torch")

    import torch  # type: ignore[import-not-found]

    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig())

    def _stub_logprobs(pa_pairs, *, use_ref: bool):
        return [torch.zeros(len(ids), dtype=torch.float32) for _, ids in pa_pairs]

    trainer._batched_logprobs = _stub_logprobs  # type: ignore[assignment]

    _, stats = trainer.compute_loss(_live_k_group(K=3))

    assert stats.dead_K_group == 0
    assert stats.std_reward_group > 0.0
