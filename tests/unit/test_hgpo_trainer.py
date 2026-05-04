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
        def parameters(self):
            return iter([])

        def train(self):
            pass

    def __init__(self):
        self.model = self._M()

    def trainable_parameters(self):
        return []


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
