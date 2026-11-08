"""Unit tests for src/core/risk_metrics.py (QuantStats-style definitions).

Expected values below are hand-computed, not copied from the
implementation:

- closes [100, 110, 99, 121] give returns [0.10, -0.10, 2/9];
  the peak is 110 and the trough after it is 99, so the maximum
  drawdown is 99/110 - 1 = -0.10 exactly. The compounded return is
  1.1 * 0.9 * (11/9) = 1.21, so the annualised return over 3 returns
  is 1.21 ** 84 - 1 and Calmar is that divided by 0.10.
- a monotonically rising series never declines: max drawdown 0.0 and
  Calmar undefined (None, not an exception).
- constant closes give zero volatility: Sharpe undefined (None).
"""

import math
import statistics
import unittest
from unittest import mock

from src.core import risk_metrics


class TestDailyReturns(unittest.TestCase):
    def test_simple_pct_change(self):
        returns = risk_metrics.daily_returns([100.0, 110.0, 99.0])
        self.assertEqual(len(returns), 2)
        self.assertAlmostEqual(returns[0], 0.1)
        self.assertAlmostEqual(returns[1], -0.1)

    def test_skips_non_positive_closes(self):
        self.assertEqual(risk_metrics.daily_returns([100.0, 0.0, 110.0]), [])


class TestSharpeRatio(unittest.TestCase):
    def test_known_sequence(self):
        closes = [100.0, 110.0, 99.0, 121.0]
        returns = risk_metrics.daily_returns(closes)
        # Hand-derived: mean 2/27, sample std of [0.1, -0.1, 2/9].
        expected = statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(252)
        self.assertAlmostEqual(risk_metrics.sharpe_ratio(returns), expected, places=9)
        self.assertAlmostEqual(risk_metrics.sharpe_ratio(returns), 7.2288, places=3)

    def test_none_when_constant_returns(self):
        self.assertIsNone(risk_metrics.sharpe_ratio([0.0, 0.0, 0.0]))

    def test_none_when_too_short(self):
        self.assertIsNone(risk_metrics.sharpe_ratio([]))
        self.assertIsNone(risk_metrics.sharpe_ratio([0.01]))


class TestMaxDrawdown(unittest.TestCase):
    def test_known_sequence(self):
        # Peak 110 -> trough 99: -0.10 exactly.
        self.assertAlmostEqual(
            risk_metrics.max_drawdown([100.0, 110.0, 99.0, 121.0]), -0.10, places=9
        )

    def test_monotonic_rise_is_zero(self):
        self.assertEqual(risk_metrics.max_drawdown([100.0, 101.0, 102.0, 103.0]), 0.0)

    def test_none_when_too_short(self):
        self.assertIsNone(risk_metrics.max_drawdown([]))
        self.assertIsNone(risk_metrics.max_drawdown([100.0]))


class TestCalmarRatio(unittest.TestCase):
    def test_known_sequence(self):
        closes = [100.0, 110.0, 99.0, 121.0]
        returns = risk_metrics.daily_returns(closes)
        # Compounded return is exactly 1.21 over 3 days.
        expected = (1.21 ** (252 / 3) - 1.0) / 0.10
        self.assertTrue(
            math.isclose(risk_metrics.calmar_ratio(returns, closes), expected, rel_tol=1e-9)
        )

    def test_none_when_no_drawdown(self):
        # Monotonic rise: max drawdown is 0, so Calmar is undefined --
        # and must not raise.
        closes = [100.0, 101.0, 102.0, 103.0]
        self.assertIsNone(
            risk_metrics.calmar_ratio(risk_metrics.daily_returns(closes), closes)
        )

    def test_none_when_too_short(self):
        self.assertIsNone(risk_metrics.calmar_ratio([], []))


class TestComputeRiskMetrics(unittest.TestCase):
    def test_uses_last_252_closes(self):
        closes = [100.0 + i * 0.1 for i in range(300)]
        metrics = risk_metrics.compute_risk_metrics(closes)
        self.assertEqual(metrics["trading_days"], 252.0)
        self.assertEqual(metrics["max_drawdown"], 0.0)
        self.assertIsNone(metrics["calmar_ratio"])

    def test_empty_series_is_all_none(self):
        metrics = risk_metrics.compute_risk_metrics([])
        self.assertIsNone(metrics["sharpe_ratio"])
        self.assertIsNone(metrics["max_drawdown"])
        self.assertIsNone(metrics["calmar_ratio"])
        self.assertEqual(metrics["trading_days"], 0.0)


class TestGetRiskMetricsForCode(unittest.TestCase):
    def setUp(self):
        risk_metrics.clear_risk_metrics_cache()

    def tearDown(self):
        risk_metrics.clear_risk_metrics_cache()

    def test_non_pilot_code_does_not_fetch(self):
        with mock.patch.object(
            risk_metrics, "fetch_daily_closes"
        ) as mock_fetch:
            self.assertIsNone(risk_metrics.get_risk_metrics_for_code("EWY"))
            mock_fetch.assert_not_called()

    def test_fetch_failure_returns_none(self):
        with mock.patch.object(
            risk_metrics, "fetch_daily_closes", side_effect=RuntimeError("boom")
        ):
            self.assertIsNone(risk_metrics.get_risk_metrics_for_code("QQQ"))

    def test_result_is_cached_per_code(self):
        closes = [(f"2026-01-{day:02d}", 100.0 + day) for day in range(1, 11)]
        with mock.patch.object(
            risk_metrics, "fetch_daily_closes", return_value=closes
        ) as mock_fetch:
            first = risk_metrics.get_risk_metrics_for_code("QQQ")
            second = risk_metrics.get_risk_metrics_for_code("QQQ")
            self.assertIs(first, second)
            mock_fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
