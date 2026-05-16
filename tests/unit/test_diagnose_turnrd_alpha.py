"""Unit tests for the TurnRD α-distribution post-hoc diagnostic.

Covers the pure-Python metric helpers (`_entropy`, `_pearson`,
`_per_traj_metrics`) so we can assert the diagnostic's verdicts are
computed correctly without spinning up a real TurnRDv2 ckpt.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnose_turnrd_alpha.py"


def _load_diag_module():
    """Load the diagnostic script as a module so we can poke its helpers.

    The script lives under `scripts/` (not `src/`) so it's not on the
    normal import path. We load it by file path.
    """
    spec = importlib.util.spec_from_file_location("diagnose_turnrd_alpha", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StubRecord:
    """Minimal stand-in for `TurnRDRecord` for the metric helper.

    `_per_traj_metrics` only reads `task_id`, `final_reward`, and
    `progress_signal`, so we don't need to construct the full dataclass
    (which has stricter validation).
    """

    def __init__(
        self,
        task_id: str = "task0",
        final_reward: float = 1.0,
        progress_signal: list[float] | None = None,
    ) -> None:
        self.task_id = task_id
        self.final_reward = final_reward
        self.progress_signal = progress_signal


class TestEntropy(unittest.TestCase):
    def setUp(self) -> None:
        self.diag = _load_diag_module()

    def test_uniform_alpha_matches_log_T(self) -> None:
        # Uniform distribution H = log(T).
        T = 4
        alpha = [1.0 / T] * T
        self.assertAlmostEqual(self.diag._entropy(alpha), math.log(T), places=6)

    def test_one_hot_has_zero_entropy(self) -> None:
        alpha = [1.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(self.diag._entropy(alpha), 0.0, places=6)

    def test_empty_returns_zero(self) -> None:
        self.assertEqual(self.diag._entropy([]), 0.0)


class TestPearson(unittest.TestCase):
    def setUp(self) -> None:
        self.diag = _load_diag_module()

    def test_perfect_positive(self) -> None:
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 2.0, 4.0, 6.0]
        self.assertAlmostEqual(self.diag._pearson(xs, ys), 1.0, places=6)

    def test_perfect_negative(self) -> None:
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [3.0, 2.0, 1.0, 0.0]
        self.assertAlmostEqual(self.diag._pearson(xs, ys), -1.0, places=6)

    def test_constant_returns_none(self) -> None:
        # Undefined corr when one side has zero variance.
        self.assertIsNone(self.diag._pearson([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]))

    def test_too_short_returns_none(self) -> None:
        self.assertIsNone(self.diag._pearson([1.0], [1.0]))


class TestPerTrajMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.diag = _load_diag_module()

    def test_uniform_alpha_yields_entropy_ratio_one(self) -> None:
        rec = _StubRecord(progress_signal=None)
        T = 5
        alpha = [1.0 / T] * T
        m = self.diag._per_traj_metrics(rec, alpha)
        self.assertAlmostEqual(m["entropy_ratio"], 1.0, places=6)
        self.assertEqual(m["T"], T)
        self.assertTrue(m["success"])

    def test_progress_prior_alpha_yields_high_prior_corr(self) -> None:
        # α directly proportional to position should yield prior_corr ≈ 1.
        rec = _StubRecord()
        T = 5
        # α_t = (t+1) normalized — strictly increasing with position.
        raw = [t + 1 for t in range(T)]
        s = sum(raw)
        alpha = [x / s for x in raw]
        m = self.diag._per_traj_metrics(rec, alpha)
        self.assertIsNotNone(m["prior_corr"])
        self.assertGreater(m["prior_corr"], 0.99)

    def test_signal_corr_and_concentration(self) -> None:
        rec = _StubRecord(progress_signal=[0.0, 1.0, 0.0, 2.0])
        # α concentrated on the two positive-signal turns.
        alpha = [0.05, 0.45, 0.05, 0.45]
        m = self.diag._per_traj_metrics(rec, alpha)
        self.assertIsNotNone(m["signal_corr"])
        self.assertGreater(m["signal_corr"], 0.7)
        # Concentration = α_1 + α_3 = 0.45 + 0.45 = 0.9.
        self.assertAlmostEqual(m["signal_concentration"], 0.9, places=6)

    def test_no_signal_yields_none_signal_metrics(self) -> None:
        rec = _StubRecord(progress_signal=None)
        alpha = [0.25, 0.25, 0.25, 0.25]
        m = self.diag._per_traj_metrics(rec, alpha)
        self.assertIsNone(m["signal_corr"])
        self.assertIsNone(m["signal_concentration"])
        self.assertFalse(m["has_progress_signal"])

    def test_failure_trajectory_flagged(self) -> None:
        rec = _StubRecord(final_reward=0.2)
        m = self.diag._per_traj_metrics(rec, [0.5, 0.5])
        self.assertFalse(m["success"])


class TestAggregate(unittest.TestCase):
    def setUp(self) -> None:
        self.diag = _load_diag_module()

    def test_aggregate_skips_nones(self) -> None:
        agg = self.diag._aggregate([1.0, None, 3.0, None, 5.0])
        self.assertEqual(agg["count"], 3)
        self.assertAlmostEqual(agg["mean"], 3.0, places=6)
        self.assertGreater(agg["std"], 0.0)

    def test_aggregate_all_nones(self) -> None:
        agg = self.diag._aggregate([None, None])
        self.assertEqual(agg["count"], 0)
        self.assertTrue(math.isnan(agg["mean"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
