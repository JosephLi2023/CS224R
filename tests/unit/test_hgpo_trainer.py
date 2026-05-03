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
