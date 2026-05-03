from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.trainers import train as train_module


class _FakeWebShopEnv:
    def __init__(self) -> None:
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return (
            {"obs": "start"},
            {"valid_actions": ["search[a]", "click[item_1]", "buy[now]"]},
        )

    def step(self, action: str):
        self._steps += 1
        done = self._steps >= 2
        reward = 1.0 if action else 0.0
        info = {"valid_actions": ["search[a]", "click[item_1]", "buy[now]"]}
        return {"obs": f"after {action}"}, reward, done, info


class TestWebShopTrainingIntegration(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _run_training_once(self, temp_dir: Path, algorithm: str) -> Path:
        env_cfg = {
            "env": {
                "name": "webshop",
                "max_steps": 4,
                "observation_mode": "text",
                "task_split": "train",
                "n_actions": 4,
                "env_kwargs": {},
            }
        }
        train_cfg = {
            "run": {
                "name": f"{algorithm}_webshop_integration",
                "output_dir": str(temp_dir / "runs"),
                "seed": 5,
            },
            "train": {
                "algorithm": algorithm,
                "total_episodes": 2,
                "batch_size": 2,
                "learning_rate": 0.05,
                "checkpoint_every": 1,
                "eval_every": 1,
            },
            "logging": {"print_every": 1},
        }
        if algorithm == "hgpo":
            train_cfg["hgpo"] = {
                "groups": {"0": [0, 1], "1": [2, 3]},
                "group_regularization_alpha": 0.1,
            }

        eval_cfg = {"eval": {"episodes": 2, "greedy": True}}

        env_path = temp_dir / f"env_{algorithm}.json"
        train_path = temp_dir / f"train_{algorithm}.json"
        eval_path = temp_dir / f"eval_{algorithm}.json"
        self._write_json(env_path, env_cfg)
        self._write_json(train_path, train_cfg)
        self._write_json(eval_path, eval_cfg)

        argv = [
            "train.py",
            "--env-config",
            str(env_path),
            "--train-config",
            str(train_path),
            "--eval-config",
            str(eval_path),
        ]

        with mock.patch(
            "src.envs.webshop_adapter.WebShopAdapter._build_webshop_env",
            autospec=True,
            side_effect=lambda _self: _FakeWebShopEnv(),
        ):
            with mock.patch("sys.argv", argv):
                train_module.main()

        run_root = temp_dir / "runs"
        matching = sorted(run_root.glob(f"{algorithm}_webshop_integration_*"))
        self.assertTrue(matching, f"No run dir created for {algorithm}")
        return matching[-1]

    def test_end_to_end_webshop_training_for_baseline_and_hgpo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)

            for algorithm in ("baseline", "hgpo"):
                with self.subTest(algorithm=algorithm):
                    run_dir = self._run_training_once(temp_dir=temp_dir, algorithm=algorithm)

                    self.assertTrue((run_dir / "config_snapshot.json").exists())
                    self.assertTrue((run_dir / "train_log.json").exists())
                    self.assertTrue((run_dir / "eval_log.json").exists())
                    self.assertTrue((run_dir / "checkpoints" / "episode_1.json").exists())
                    self.assertTrue((run_dir / "checkpoints" / "episode_2.json").exists())

                    with open(run_dir / "train_log.json", "r", encoding="utf-8") as f:
                        train_log = json.load(f)
                    with open(run_dir / "eval_log.json", "r", encoding="utf-8") as f:
                        eval_log = json.load(f)

                    self.assertEqual(len(train_log["rows"]), 2)
                    self.assertEqual(train_log["rows"][0]["episode"], 1)
                    self.assertEqual(train_log["rows"][1]["episode"], 2)
                    self.assertEqual(len(eval_log["rows"]), 2)
                    self.assertEqual(eval_log["rows"][1]["episode"], 2)


if __name__ == "__main__":
    unittest.main()
