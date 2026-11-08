"""Unit tests for src/core/duckdb_history.py.

The moving-average expectations are hand-computed for the synthetic
series closes 1..40: MA5 at the 5th bar is (1+2+3+4+5)/5 = 3.0 and
MA20 at the 20th bar is (1+...+20)/20 = 10.5; earlier bars are NULL,
matching pandas ``rolling(window).mean()`` with the default
``min_periods=window``.
"""

import os
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

from src.core import duckdb_history


def _synthetic_bars(n: int = 40) -> pd.DataFrame:
    start = date(2026, 1, 5)  # a Monday
    dates = [start + timedelta(days=i) for i in range(n)]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "close": [float(i + 1) for i in range(n)],
            "volume": [1000.0] * n,
        }
    )


class TestStoreAndMovingAverages(unittest.TestCase):
    def test_sql_ma_matches_hand_computed_values(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            bars = _synthetic_bars()
            self.assertEqual(
                duckdb_history.store_daily_bars(db_path, "QQQ", bars), 40
            )
            rows = duckdb_history.moving_averages(db_path, "QQQ")
            self.assertEqual(len(rows), 40)
            # First bars have no full window yet.
            self.assertIsNone(rows[0]["ma5"])
            self.assertIsNone(rows[3]["ma5"])
            self.assertIsNone(rows[18]["ma20"])
            # Hand-computed full-window values.
            self.assertAlmostEqual(rows[4]["ma5"], 3.0)
            self.assertAlmostEqual(rows[9]["ma10"], 5.5)
            self.assertAlmostEqual(rows[19]["ma20"], 10.5)
            self.assertAlmostEqual(rows[39]["ma5"], 38.0)
            self.assertAlmostEqual(rows[39]["ma20"], 30.5)

    def test_store_replaces_previous_rows(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            duckdb_history.store_daily_bars(db_path, "QQQ", _synthetic_bars(10))
            duckdb_history.store_daily_bars(db_path, "QQQ", _synthetic_bars(5))
            self.assertEqual(
                len(duckdb_history.moving_averages(db_path, "QQQ")), 5
            )

    def test_missing_db_returns_empty(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(
                duckdb_history.moving_averages(Path(tmpdir) / "nope.duckdb", "QQQ"),
                [],
            )
            self.assertEqual(
                duckdb_history.load_closes(Path(tmpdir) / "nope.duckdb", "QQQ"), []
            )

    def test_load_closes_is_date_ordered(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            duckdb_history.store_daily_bars(db_path, "QQQ", _synthetic_bars(5))
            closes = duckdb_history.load_closes(db_path, "QQQ")
            self.assertEqual(
                [c for _d, c in closes], [1.0, 2.0, 3.0, 4.0, 5.0]
            )


class TestCompareWithPandas(unittest.TestCase):
    def test_identical_data_has_no_mismatches(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            bars = _synthetic_bars()
            duckdb_history.store_daily_bars(db_path, "QQQ", bars)
            result = duckdb_history.compare_with_pandas(db_path, "QQQ", bars)
            # MA20 needs 20 rows: (40-4) + (40-9) + (40-19) = 36+31+21 = 88.
            self.assertEqual(result["rows_compared"], 88)
            self.assertEqual(result["max_abs_diff"], 0.0)
            self.assertEqual(result["mismatches"], [])

    def test_tampered_data_is_reported(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            bars = _synthetic_bars()
            duckdb_history.store_daily_bars(db_path, "QQQ", bars)
            tampered = bars.copy()
            tampered.loc[30, "close"] = 999.0
            result = duckdb_history.compare_with_pandas(db_path, "QQQ", tampered)
            self.assertTrue(result["mismatches"])


class TestResolveDbPath(unittest.TestCase):
    def test_env_var_overrides_default(self):
        with mock.patch.dict(os.environ, {"DSA_DUCKDB_PATH": "/tmp/custom/history.duckdb"}):
            self.assertEqual(
                duckdb_history.resolve_db_path(),
                Path("/tmp/custom/history.duckdb"),
            )

    def test_explicit_arg_wins_over_env_var(self):
        with mock.patch.dict(os.environ, {"DSA_DUCKDB_PATH": "/tmp/custom/history.duckdb"}):
            self.assertEqual(
                duckdb_history.resolve_db_path("/tmp/other.duckdb"),
                Path("/tmp/other.duckdb"),
            )

    def test_default_is_under_repo_data_dir(self):
        with mock.patch.dict(os.environ, {"DSA_DUCKDB_PATH": ""}):
            path = duckdb_history.resolve_db_path()
            self.assertEqual(path.name, "history.duckdb")
            self.assertEqual(path.parent.name, "duckdb")


if __name__ == "__main__":
    unittest.main()
