"""Unit tests for the bake-off post-hoc analysis script."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "posthoc_factsdiff_bakeoff.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("posthoc_factsdiff_bakeoff", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SAMPLE_LOG = """\
=== Done. 5 rounds × 40 episodes = 200 total H-GRPO episodes. ===
┌── Round 0: train_loop (40 eps)
│  $ modal run --detach infra/app_train_loop.py::train_loop_alfworld
some intermediate junk
>>> Eval done: avg_R=0.4700 (±0.4991) | pct_success=0.470 | ok=100/100 | elapsed=658.78s
✓ App completed. View run at https://example/round0
   ↻ polling ap-AAA0 (Round 0: train_loop (40 eps))…
(Round 0: train_loop (40 eps) exited 0 after 2440.6s)
┌── Round 0: train_turnrd (3 epochs)
✓ App completed.
(Round 0: train_turnrd (3 epochs) exited 0 after 96.4s)
┌── Round 1: train_loop (40 eps)
ep=4 CRASHED: IndexError('Cannot choose from an empty sequence')
eval_success_rate: 0.520
(Round 1: train_loop (40 eps) exited 0 after 2378.41s)
┌── Round 1: train_turnrd (3 epochs)
(Round 1: train_turnrd (3 epochs) exited 0 after 96.97s)
"""


class TestLogParser(unittest.TestCase):
    def setUp(self) -> None:
        self.diag = _load_module()
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.tmp.write(SAMPLE_LOG)
        self.tmp.close()
        self.log_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.log_path.unlink(missing_ok=True)

    def test_round_banners_detected(self) -> None:
        result = self.diag.parse_log(self.log_path)
        self.assertIn("round00_train_loop", result["rounds"])
        self.assertIn("round00_train_turnrd", result["rounds"])
        self.assertIn("round01_train_loop", result["rounds"])
        self.assertIn("round01_train_turnrd", result["rounds"])

    def test_round_completion_recorded(self) -> None:
        result = self.diag.parse_log(self.log_path)
        for key in [
            "round00_train_loop",
            "round00_train_turnrd",
            "round01_train_loop",
            "round01_train_turnrd",
        ]:
            self.assertTrue(result["rounds"][key]["completed"], key)
            self.assertEqual(result["rounds"][key]["exit_code"], 0, key)
            self.assertGreater(result["rounds"][key]["elapsed_s"], 0.0, key)

    def test_eval_rates_per_round(self) -> None:
        result = self.diag.parse_log(self.log_path)
        rates = result["eval_success_rate_per_round"]
        # Round 0 has `pct_success=0.470` format; Round 1 has `eval_success_rate: 0.520`.
        # Both regex patterns must match.
        self.assertAlmostEqual(rates[0], 0.470, places=3)
        self.assertAlmostEqual(rates[1], 0.520, places=3)

    def test_transient_errors_collected(self) -> None:
        result = self.diag.parse_log(self.log_path)
        self.assertEqual(result["n_transient_errors"], 1)
        self.assertIn("CRASHED", result["transient_errors_sample"][0])

    def test_missing_log_returns_error(self) -> None:
        result = self.diag.parse_log(Path("/tmp/does_not_exist.log"))
        self.assertIn("error", result)


class TestReplayAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.diag = _load_module()
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        records = [
            # Trajectory with all-zero progress_signal (broken-Phase-1 baseline shape).
            {"task_id": "t0", "turn_embeds": [[0.1] * 3] * 4, "final_reward": 0.0,
             "progress_signal": [0.0, 0.0, 0.0, 0.0]},
            # Trajectory with mixed nonzero signal (facts-diff working).
            {"task_id": "t1", "turn_embeds": [[0.1] * 3] * 5, "final_reward": 1.0,
             "progress_signal": [0.0, 1.0, 0.0, 2.0, 0.0]},
            # Trajectory missing the field entirely.
            {"task_id": "t2", "turn_embeds": [[0.1] * 3] * 3, "final_reward": 0.6,
             "progress_signal": None},
        ]
        for r in records:
            self.tmp.write(json.dumps(r) + "\n")
        self.tmp.close()
        self.replay_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.replay_path.unlink(missing_ok=True)

    def test_audit_counts(self) -> None:
        result = self.diag.audit_replay(self.replay_path)
        self.assertEqual(result["n_trajectories"], 3)
        self.assertEqual(result["n_with_progress_signal"], 2)  # t0 + t1 have lists
        self.assertEqual(result["n_with_nonzero_progress_signal"], 1)  # only t1
        # Per-step stats: 4 zeros + 5 mixed = 9 values, 2 nonzero (1.0, 2.0)
        sig = result["per_step_signal_stats"]
        self.assertEqual(sig["n_per_step_values"], 9)
        self.assertEqual(sig["n_nonzero"], 2)
        self.assertAlmostEqual(sig["max"], 2.0, places=6)

    def test_final_reward_stats(self) -> None:
        result = self.diag.audit_replay(self.replay_path)
        # 3 trajectories: 0.0, 1.0, 0.6 → 2 above 0.5 → 2/3.
        self.assertAlmostEqual(result["final_reward"]["frac_success_at_0_5"], 2.0 / 3.0, places=6)

    def test_missing_replay_returns_error(self) -> None:
        result = self.diag.audit_replay(Path("/tmp/does_not_exist.jsonl"))
        self.assertIn("error", result)


class TestMarkdownRendering(unittest.TestCase):
    def setUp(self) -> None:
        self.diag = _load_module()

    def test_renders_with_minimal_summary(self) -> None:
        summary = {
            "log": {
                "log_path": "/tmp/example.log",
                "rounds": {
                    "round00_train_loop": {"round_idx": 0, "phase": "train_loop",
                                           "completed": True, "elapsed_s": 100.0},
                },
                "eval_success_rate_per_round": {0: 0.48, 1: 0.58},
                "n_transient_errors": 0,
            },
            "replay_audit": {
                "n_trajectories": 100,
                "n_with_progress_signal": 100,
                "n_with_nonzero_progress_signal": 80,
                "frac_traj_with_nonzero_signal": 0.8,
                "per_step_signal_stats": {"frac_nonzero": 0.4, "mean": 0.3, "mean_nonzero": 0.7, "max": 5.0},
            },
        }
        md = self.diag.render_markdown(summary)
        self.assertIn("Plan verification PASSED", md)
        self.assertIn("Δ = +0.100", md)

    def test_renders_failed_verification(self) -> None:
        summary = {
            "log": {
                "log_path": "/tmp/x.log",
                "rounds": {},
                "eval_success_rate_per_round": {},
                "n_transient_errors": 0,
            },
            "replay_audit": {
                "n_trajectories": 50,
                "n_with_progress_signal": 50,
                "n_with_nonzero_progress_signal": 0,
                "frac_traj_with_nonzero_signal": 0.0,
                "per_step_signal_stats": {"frac_nonzero": 0.0, "mean": 0.0, "mean_nonzero": 0.0, "max": 0.0},
            },
        }
        md = self.diag.render_markdown(summary)
        self.assertIn("Plan verification FAILED", md)

    def test_omitted_replay_does_not_falsely_claim_signal_absent(self) -> None:
        """Regression: when --replay isn't passed, the recommendation
        used to misread the missing audit as 'signal absent'. Now it
        explicitly says the audit wasn't run."""
        summary = {
            "log": {
                "log_path": "/tmp/x.log",
                "rounds": {},
                "eval_success_rate_per_round": {},
                "n_transient_errors": 0,
            },
            # NO replay_audit key.
        }
        md = self.diag.render_markdown(summary)
        self.assertNotIn("Facts-diff signal absent", md)
        self.assertIn("Replay JSONL not provided", md)


if __name__ == "__main__":
    unittest.main()
