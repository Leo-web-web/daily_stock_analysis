"""Unit tests for scripts/check_duckdb_ma.py (all-tickers rollout).

The per-code verification is tested with a real temporary DuckDB file
and synthetic bars (the store/compare path is local and fast); only
the network fetch is mocked. Exit-code semantics under test:

- a code whose MAs match pandas is ``"ok"``;
- a code with no data at all is ``"skipped"`` only for the known
  no-data codes (``^N225``/``^KS11``) -- for any other code a fetch
  failure is ``"failed"``;
- any mismatch is ``"failed"``.
"""

import importlib.util
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_duckdb_ma.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_duckdb_ma", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_bars(n: int = 40) -> pd.DataFrame:
    start = date(2026, 1, 5)  # a Monday
    dates = [start + timedelta(days=i) for i in range(n)]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "close": [100.0 + i for i in range(n)],
            "volume": [1000.0] * n,
        }
    )


class TestVerifyCode(unittest.TestCase):
    def setUp(self):
        self.script = _load_script()

    def test_matching_data_is_ok(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            # 60 bars -> 3*60-32 = 148 compared values, above the 100 minimum.
            with mock.patch.object(
                self.script, "fetch_history", return_value=_synthetic_bars(60)
            ):
                result = self.script.verify_code("FXI", db_path=str(db_path))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["rows_compared"], 148)
            self.assertEqual(result["max_abs_diff"], 0.0)
            self.assertEqual(result["mismatches"], [])
            self.assertIsNotNone(result["metrics"])

    def test_mismatch_is_failed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            with (
                mock.patch.object(
                    self.script, "fetch_history", return_value=_synthetic_bars()
                ),
                mock.patch.object(
                    self.script.duckdb_history,
                    "compare_with_pandas",
                    return_value={
                        "rows_compared": 88,
                        "max_abs_diff": 1.5,
                        "mismatches": [
                            {
                                "trade_date": "2026-02-01",
                                "window": 5,
                                "pandas": 130.0,
                                "duckdb": 131.5,
                            }
                        ],
                    },
                ),
            ):
                result = self.script.verify_code("FXI", db_path=str(db_path))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["mismatches"]), 1)

    def test_no_data_code_is_skipped(self):
        with mock.patch.object(
            self.script, "fetch_history", side_effect=RuntimeError("no data")
        ):
            result = self.script.verify_code("^N225")
        self.assertEqual(result["status"], "skipped")

    def test_fetch_failure_for_data_code_is_failed(self):
        with mock.patch.object(
            self.script, "fetch_history", side_effect=RuntimeError("no data")
        ):
            result = self.script.verify_code("FXI")
        self.assertEqual(result["status"], "failed")

    def test_short_frame_for_data_code_is_failed(self):
        with mock.patch.object(
            self.script, "fetch_history", return_value=_synthetic_bars(5)
        ):
            result = self.script.verify_code("EWY")
        self.assertEqual(result["status"], "failed")


class TestMainExitCodes(unittest.TestCase):
    def setUp(self):
        self.script = _load_script()

    def _canned(self, code, status):
        return {
            "code": code,
            "status": status,
            "rows_compared": 88 if status == "ok" else 0,
            "max_abs_diff": 0.0,
            "mismatches": [],
            "metrics": None,
        }

    def test_all_ok_exits_zero(self):
        with mock.patch.object(
            self.script,
            "verify_code",
            side_effect=lambda code, **_kwargs: self._canned(code, "ok"),
        ):
            self.assertEqual(self.script.main(["--all"]), 0)

    def test_skipped_codes_do_not_fail(self):
        def fake_verify(code, **_kwargs):
            return self._canned(code, "skipped" if code in ("^N225", "^KS11") else "ok")

        with mock.patch.object(self.script, "verify_code", side_effect=fake_verify):
            self.assertEqual(self.script.main(["--all"]), 0)

    def test_failed_code_exits_nonzero(self):
        def fake_verify(code, **_kwargs):
            return self._canned(code, "failed" if code == "FXI" else "ok")

        with mock.patch.object(self.script, "verify_code", side_effect=fake_verify):
            self.assertEqual(self.script.main(["--all"]), 1)

    def test_all_skipped_exits_nonzero(self):
        with mock.patch.object(
            self.script,
            "verify_code",
            side_effect=lambda code, **_kwargs: self._canned(code, "skipped"),
        ):
            self.assertEqual(self.script.main(["--all"]), 1)


if __name__ == "__main__":
    unittest.main()
