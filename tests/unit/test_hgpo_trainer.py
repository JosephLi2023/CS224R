"""Unit tests for HGPOTrainer's pure-Python advantage-construction stage.

The torch training path runs on Modal only; here we verify `build_advantages`
produces correctly-shaped outputs and that `alpha=1, lambda=0` reduces to
flat GRPO (verification gate #1 from the plan)."""
from __future__ import annotations

import pytest

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord
from src.algorithms.grpo.trainer import (
    HGPOTrainer,
    HGPOTrainerConfig,
    progress_decomposer,
)


class _StubPolicy:
    """Stand-in for LoRAPolicy used so the trainer can instantiate in tests."""

    class _M:
        def __init__(self):
            # Provide a real (1-element) parameter so AdamW can be built when
            # tests exercise `train_step` (e.g. the refresh-hook
            # tests). Lazy-imported torch keeps the module load torch-free
            # for the pure-Python tests above.
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
            # No LoRA-shaped modules; snapshot_current_lora_as_ref filters
            # by `hasattr(module, "lora_A")` so an empty iterator is fine.
            return iter([])

        def train(self):
            pass

    def __init__(self):
        self.model = self._M()

    def trainable_parameters(self):
        if self.model._param is None:
            return []
        return [self.model._param]


def _group(K: int = 3) -> TrajectoryGroup:
    trajectories = []
    for i in range(K):
        turns = [
            TurnRecord(
                turn_idx=t,
                observation_text=f"o{i}-{t}",
                action_text=f"a{i}-{t}",
                raw_env_reward=0.1 * (t + 1) + 0.05 * i,
            )
            for t in range(3)
        ]
        trajectories.append(
            Trajectory(
                task_id="task-A",
                env_name="webshop",
                turns=turns,
                final_reward=0.3 * (i + 1),
            )
        )
    return TrajectoryGroup(task_id="task-A", env_name="webshop", trajectories=trajectories)


def test_build_advantages_shapes():
    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig(alpha=0.5))
    g = _group(K=3)
    out = trainer.build_advantages(g)
    assert len(out["traj_adv"]) == 3
    assert [len(r) for r in out["turn_adv"]] == [3, 3, 3]
    assert [len(r) for r in out["combined"]] == [3, 3, 3]
    assert isinstance(out["consistency"], float)


def test_alpha_one_lambda_zero_reduces_to_flat_grpo():
    """The verification gate: α=1, λ=0 must broadcast traj advantages over turns."""
    cfg = HGPOTrainerConfig(alpha=1.0, lambda_consistency=0.0)
    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, cfg)
    g = _group(K=4)
    out = trainer.build_advantages(g)
    assert out["consistency"] == 0.0
    for i, row in enumerate(out["combined"]):
        for v in row:
            assert v == pytest.approx(out["traj_adv"][i])


def test_progress_decomposer_reads_raw_env_rewards():
    g = _group(K=2)
    decomposed = progress_decomposer(g)
    assert [len(r) for r in decomposed] == [3, 3]
    assert decomposed[0][0] == pytest.approx(0.1)   # 0.1*(0+1) + 0.05*0
    assert decomposed[1][2] == pytest.approx(0.35)  # 0.1*(2+1) + 0.05*1


def test_trainer_carries_kl_controller():
    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig())
    assert hasattr(trainer, "kl_controller")
    assert trainer.kl_controller.coef > 0


def test_trainer_config_defaults_sanity():
    cfg = HGPOTrainerConfig()
    assert 0 <= cfg.alpha <= 1
    assert cfg.clip_eps > 0
    assert cfg.learning_rate > 0
    assert cfg.max_grad_norm > 0


# ------- grad-accum control flow (no torch: structural check only) -------


def test_grad_accum_default_is_one():
    """Default: optimizer steps every train_step. Regression for review M4."""
    cfg = HGPOTrainerConfig()
    assert cfg.grad_accum_steps == 1
    # accum == 1 → always a step boundary (never skips optimizer.step)
    assert ((0 + 1) % max(1, cfg.grad_accum_steps)) == 0


def test_grad_accum_skip_schedule_with_accum_4():
    """With grad_accum_steps=4, optimizer.step runs every 4 invocations
    (steps 3, 7, 11, ... in 0-indexed self._step counter)."""
    accum = 4
    boundaries = [((s + 1) % accum) == 0 for s in range(12)]
    expected = [False, False, False, True] * 3
    assert boundaries == expected


def test_trainable_params_snapshot_is_list():
    """Snapshot preserves ordering + materialisation — guards review M7."""
    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig())
    # The snapshot is materialised on first _ensure_optimizer; here we just
    # verify the attribute is initialised empty (lazy init).
    assert trainer._optimizer is None


# ----- KL warmup + snapshot-as-ref config tests --------------


def test_kl_warmup_default_zero():
    cfg = HGPOTrainerConfig()
    assert cfg.kl_warmup_episodes == 0


def test_kl_warmup_configurable():
    cfg = HGPOTrainerConfig(kl_warmup_episodes=5)
    assert cfg.kl_warmup_episodes == 5


def test_snapshot_lora_attribute_present():
    """Trainer must have the _ref_lora_snapshot attribute (initially None)
    and the snapshot_current_lora_as_ref method (callable signature)."""
    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig())
    assert hasattr(trainer, "_ref_lora_snapshot")
    assert trainer._ref_lora_snapshot is None
    assert callable(getattr(trainer, "snapshot_current_lora_as_ref", None))


def test_snapshot_returns_count_for_stub_policy():
    """Stub policy has no LoRA modules → snapshot returns 0 but doesn't crash.
    Skipped when torch isn't installed locally (the method requires it)."""
    pytest.importorskip("torch")
    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig())
    n = trainer.snapshot_current_lora_as_ref()
    assert n == 0
    assert trainer._ref_lora_snapshot == {}


# ---------------------------------------------------------------------------
# Refresh hook + C3 reattach
# ---------------------------------------------------------------------------


def test_refresh_every_episodes_default_zero():
    """Default disables the refresh hook (preserves prior behavior)."""
    cfg = HGPOTrainerConfig()
    assert cfg.refresh_every_episodes == 0
    assert cfg.turnrd_lr == 1e-4


def test_refresh_hook_fires_at_correct_cadence():
    """With `refresh_every_episodes=2`, the hook fires at _step=2 and _step=4
    over 5 train_step invocations (skipping _step=0). Uses a stub trainer
    that bypasses the heavy compute path by short-circuiting via the
    `n_action_tokens == 0` branch (the test group has no token ids).
    """
    pytest.importorskip("torch")
    refresh_count: list[int] = []

    def refresh_fn() -> None:
        refresh_count.append(1)

    cfg = HGPOTrainerConfig(refresh_every_episodes=2)
    trainer = HGPOTrainer(
        _StubPolicy(),
        progress_decomposer,
        cfg,
        refresh_decomposer_fn=refresh_fn,
    )

    # Build a trivial group whose turns have no token ids → compute_loss
    # short-circuits via the n_action_tokens==0 path. The refresh hook
    # runs BEFORE the short-circuit so the cadence test still works.
    group = _group(K=2)

    for _ in range(5):
        trainer.train_step(group)

    # _step ∈ {0..4} as it's incremented after each call. The hook fires
    # iff (_step > 0 AND _step % 2 == 0) → _step in {2, 4} → 2 fires.
    assert len(refresh_count) == 2


def test_refresh_hook_disabled_when_cadence_zero():
    """`refresh_every_episodes=0` disables the hook (default)."""
    pytest.importorskip("torch")
    refresh_count: list[int] = []

    cfg = HGPOTrainerConfig(refresh_every_episodes=0)
    trainer = HGPOTrainer(
        _StubPolicy(),
        progress_decomposer,
        cfg,
        refresh_decomposer_fn=lambda: refresh_count.append(1),
    )
    group = _group(K=2)
    for _ in range(5):
        trainer.train_step(group)
    assert refresh_count == []


def test_decomposer_optimizer_only_built_for_learnable_decomposer():
    """Non-learnable decomposer (progress_decomposer) → no second optimizer."""
    pytest.importorskip("torch")
    trainer = HGPOTrainer(_StubPolicy(), progress_decomposer, HGPOTrainerConfig())
    assert trainer._decomposer_learnable is False
    # train_step on a group with no token ids exercises _ensure_optimizer.
    trainer.train_step(_group(K=2))
    assert trainer._decomposer_optimizer is None


def test_v9_v_head_override_does_not_crash_and_actually_drives_ppo_loss():
    """Regression test for two structural bugs in an earlier V-head wiring.

    An earlier version of ``compute_loss`` had two bugs in the V-head
    override path that were both observable simultaneously and individually
    catastrophic:

    Bug A (UnboundLocalError):
        The override block at L807 referenced ``traj_adv[orig_i]`` (Python
        list), but ``traj_adv = adv["traj_adv"]`` was only assigned at
        L871, AFTER the override. Every TurnRD episode crashed with
        ``UnboundLocalError("cannot access local variable 'traj_adv' ...")``,
        which the train-loop ``except Exception:`` swallowed → 0 PPO
        updates → frozen policy → byte-identical eval across all rounds.

    Bug B (override is dead code):
        The override mutated ``combined[orig_i][t]`` at L806, but the
        actor's ``adv_t`` tensor was already snapshotted from
        ``combined[i][t]`` ~200 lines earlier (L605), so even if Bug A
        were fixed, the override would have had ZERO effect on the PPO
        loss — V was being trained but never wired into the policy
        gradient.

    This test exercises the override end-to-end with:
    - A real ``TurnRDDecomposer`` configured with ``value_head=True`` so
      ``value_per_turn`` is provided to the override path.
    - ``v_baseline_round_idx >= v_baseline_warmup_rounds`` so β=1 and the
      override is fully active (vs. β=0 where the M1 short-circuit skips
      the override entirely).
    - A monkey-patched ``_batched_logprobs`` returning deterministic
      tensors so we don't need a real LoRA forward (the test runs CPU-only
      in <1s).

    Assertions:
    1. ``compute_loss`` runs to completion at β=1 (catches Bug A).
    2. ``policy_loss`` at β=1 (override active) DIFFERS from
       ``policy_loss`` at β=0 (override skipped). If they're equal, the
       override is dead code (catches Bug B).
    """
    torch = pytest.importorskip("torch")

    from src.algorithms.grpo.rollout import Trajectory, TurnRecord
    from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer
    from src.turnrd.model import TurnRD, TurnRDConfig

    INPUT_DIM = 16
    torch.manual_seed(0)
    cfg_model = TurnRDConfig(
        n_layers=1,
        hidden_size=32,
        n_heads=4,
        max_turns=8,
        dropout=0.0,
        value_head=True,
        causal=True,
    )
    model = TurnRD(cfg_model, input_dim=INPUT_DIM)

    def embedder(traj):
        T_i = len(traj.turns)
        # Deterministic per-turn embeddings; offset by trajectory hash so
        # K samples don't collide.
        offset = (
            abs(hash(traj.task_id + "|" + (traj.turns[0].action_text if traj.turns else "")))
            % 997
        ) * 1.0
        base = torch.arange(T_i * INPUT_DIM, dtype=torch.float32).view(T_i, INPUT_DIM)
        return base + offset

    decomposer = TurnRDDecomposer(model=model, embedder=embedder)

    # Build a group where every turn has token ids → compute_loss runs
    # the full PPO surrogate path (no early `not pa_pairs` short-circuit).
    K, T = 3, 3
    trajectories = []
    for i in range(K):
        turns = [
            TurnRecord(
                turn_idx=t,
                observation_text=f"o{i}-{t}",
                action_text=f"a{i}-{t}",
                raw_env_reward=0.1 * (t + 1) + 0.05 * i,
                prompt_token_ids=(1, 2, 3),
                action_token_ids=(10, 20),
                action_token_logprobs=(-1.0, -1.0),
            )
            for t in range(T)
        ]
        trajectories.append(
            Trajectory(
                task_id="task-X",
                env_name="webshop",
                turns=turns,
                final_reward=0.3 * (i + 1),
            )
        )
    group = TrajectoryGroup(
        task_id="task-X", env_name="webshop", trajectories=trajectories
    )

    def _make_trainer(round_idx: int) -> HGPOTrainer:
        cfg = HGPOTrainerConfig(
            alpha=0.5,
            lambda_consistency=0.1,  # > 0 so the grad block runs
            v_baseline_round_idx=round_idx,
            v_baseline_warmup_rounds=2,
            kl_warmup_episodes=0,
        )
        trainer = HGPOTrainer(_StubPolicy(), decomposer, cfg)
        # Patch the batched-logprob forward so the test is CPU-only and
        # deterministic. Returns one zero-tensor per (prompt, action) pair.
        def _stub_logprobs(pa_pairs, use_ref=False):
            return [
                torch.zeros(len(ids), dtype=torch.float32) for _, ids in pa_pairs
            ]
        trainer._batched_logprobs = _stub_logprobs  # type: ignore[assignment]
        return trainer

    # β=1 → override fully active. Pre-v9: UnboundLocalError on traj_adv.
    trainer_full = _make_trainer(round_idx=2)
    total_full, stats_full = trainer_full.compute_loss(group)
    assert torch.isfinite(total_full).item(), (
        "compute_loss returned non-finite total at β=1"
    )

    # β=0 → override short-circuited. Pre-v9 (without M1 fix) would also
    # have entered the override loop and crashed; with the M1 fix this
    # path skips the override entirely.
    trainer_skip = _make_trainer(round_idx=0)
    total_skip, stats_skip = trainer_skip.compute_loss(group)

    # Bug B regression: at β=1 the override MUST influence policy_loss.
    # If equal, the override is dead code (the actual v6 behavior).
    assert stats_full.policy_loss != pytest.approx(stats_skip.policy_loss), (
        f"V-head override at β=1 produced policy_loss={stats_full.policy_loss}, "
        f"identical to β=0 ({stats_skip.policy_loss}). The override is dead "
        "code — v6's V-head was never actually wired into PPO."
    )
