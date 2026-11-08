#!/usr/bin/env python3
"""Verify the DuckDB history store: SQL moving averages vs pandas.

Fetches real daily history for one code (``--code``, default QQQ) or
for all brief tickers (``--all``) through the repo's own
``DataFetcherManager``, stores each in the local DuckDB file, computes
MA5/MA10/MA20 with SQL window functions, and compares them row by row
against the pandas ``rolling(window).mean()`` results used by
``StockAnalyzer._calculate_mas``.

Exit code 0 means every code that returned data agrees to 2 decimals.
Codes with no data at all (currently the Nikkei 225 and KOSPI, whose
sources return nothing) are reported as SKIPPED -- explicitly not
counted as passed. Anything else (a fetch failure for a code that is
expected to have data, too little data, or any mismatch) exits
non-zero and prints the offending rows. Nothing here changes the
online report: the pandas numbers remain the source of truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from data_provider.base import DataFetcherManager  # noqa: E402
from src.core import duckdb_history  # noqa: E402
from src.core import risk_metrics  # noqa: E402

WINDOWS = (5, 10, 20)
MIN_COMPARED_ROWS = 100

#: Every ticker the daily brief covers, in STOCK_LIST order.
ALL_CODES = ("QQQ", "FXI", "EWY", "EWJ", "sh000001", "^N225", "^KS11")

#: Codes whose data sources are known to return nothing today. A fetch
#: failure or an empty frame for one of these is a skip, not a failure;
#: for any other code it is a failure.
EXPECTED_NO_DATA_CODES = frozenset({"^N225", "^KS11"})


def fetch_history(code: str, days: int) -> pd.DataFrame:
    """Fetch daily history through the repo's fetcher manager."""
    manager = DataFetcherManager()
    df, source = manager.get_daily_data(code, days=days)
    print(f"Fetched {len(df)} rows for {code} from {source}")
    frame = df[["date", "close"] + (["volume"] if "volume" in df.columns else [])].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    return frame


def print_recent_table(frame: pd.DataFrame, sql_rows, windows=WINDOWS) -> None:
    """Print the last 5 trading days: pandas MA vs SQL MA side by side."""
    pandas_ma = {
        window: frame["close"].rolling(window=window).mean() for window in windows
    }
    sql_by_date = {row["trade_date"]: row for row in sql_rows}
    dates = [d.strftime("%Y-%m-%d") for d in frame["date"]]
    print("\nLast 5 trading days (pandas MA / DuckDB MA):")
    header = "date       close   " + "  ".join(f"MA{w} pandas/sql" for w in windows)
    print(header)
    for position in range(max(0, len(frame) - 5), len(frame)):
        date_iso = dates[position]
        sql_row = sql_by_date.get(date_iso, {})
        cells = []
        for window in windows:
            pandas_value = pandas_ma[window].iloc[position]
            sql_value = sql_row.get(f"ma{window}")
            pandas_text = f"{pandas_value:.2f}" if pd.notna(pandas_value) else "n/a"
            sql_text = f"{sql_value:.2f}" if sql_value is not None else "n/a"
            cells.append(f"{pandas_text}/{sql_text}")
        print(f"{date_iso} {frame['close'].iloc[position]:8.2f}  " + "  ".join(cells))


def verify_code(
    code: str,
    days: int = 365,
    db_path=None,
    tolerance: float = 0.005,
    windows=WINDOWS,
) -> Dict[str, object]:
    """Verify one code: fetch, store, compare, and compute risk metrics.

    Returns ``{"code", "status", "rows_compared", "max_abs_diff",
    "mismatches", "metrics"}`` where ``status`` is ``"ok"``,
    ``"skipped"`` (no data at all -- only expected for
    ``EXPECTED_NO_DATA_CODES``), or ``"failed"``.
    """
    result: Dict[str, object] = {
        "code": code,
        "status": "failed",
        "rows_compared": 0,
        "max_abs_diff": 0.0,
        "mismatches": [],
        "metrics": None,
    }
    try:
        frame = fetch_history(code, days)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print(f"{code}: FAILED to fetch history: {exc}")
        result["status"] = (
            "skipped" if code in EXPECTED_NO_DATA_CODES else "failed"
        )
        return result
    if len(frame) < max(windows) + 1:
        print(f"{code}: only {len(frame)} rows fetched, need more than {max(windows)}")
        result["status"] = (
            "skipped" if code in EXPECTED_NO_DATA_CODES else "failed"
        )
        return result

    path = duckdb_history.resolve_db_path(db_path)
    stored = duckdb_history.store_daily_bars(path, code, frame)
    print(f"Stored {stored} rows in {path}")

    comparison = duckdb_history.compare_with_pandas(
        path, code, frame, windows=windows, tolerance=tolerance
    )
    sql_rows = duckdb_history.moving_averages(path, code, windows=windows)
    print_recent_table(frame, sql_rows, windows)

    # Same data, same window: also show this code's risk metrics
    # (QuantStats definitions) so the rollout's second half has a
    # real-number check per code.
    metrics = risk_metrics.compute_risk_metrics(frame["close"].tolist())
    result["metrics"] = metrics
    print("\nRisk metrics (last 252 trading days, QuantStats definitions):")
    print(f"  sharpe_ratio : {metrics['sharpe_ratio']}")
    print(f"  max_drawdown : {metrics['max_drawdown']}")
    print(f"  calmar_ratio : {metrics['calmar_ratio']}")
    print(f"  trading_days : {metrics['trading_days']}")

    rows_compared = comparison["rows_compared"]
    mismatches = comparison["mismatches"]
    result["rows_compared"] = rows_compared
    result["max_abs_diff"] = comparison["max_abs_diff"]
    result["mismatches"] = mismatches
    print(
        f"\nCompared {rows_compared} MA values, "
        f"max abs diff {comparison['max_abs_diff']:.6f}, "
        f"mismatches {len(mismatches)}"
    )
    if rows_compared < MIN_COMPARED_ROWS:
        print(f"FAILED: only {rows_compared} rows compared, need >= {MIN_COMPARED_ROWS}")
        return result
    if mismatches:
        print("FAILED: mismatches (first 10):")
        for mismatch in mismatches[:10]:
            print(f"  {mismatch}")
        return result
    print("OK: DuckDB moving averages match pandas to 2 decimals.")
    result["status"] = "ok"
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="QQQ", help="Ticker to verify (default: QQQ)")
    parser.add_argument(
        "--all", action="store_true", help="Verify all brief tickers"
    )
    parser.add_argument("--days", type=int, default=365, help="Calendar days to fetch")
    parser.add_argument("--db-path", default=None, help="DuckDB file path override")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Max allowed abs diff (0.005 = identical at 2 decimals)",
    )
    args = parser.parse_args(argv)

    codes = list(ALL_CODES) if args.all else [args.code]
    results = [
        verify_code(code, days=args.days, db_path=args.db_path, tolerance=args.tolerance)
        for code in codes
    ]
    print("\n===== Summary =====")
    failed = 0
    for result in results:
        status = result["status"]
        if status == "ok":
            print(
                f"  {result['code']}: OK "
                f"({result['rows_compared']} compared, "
                f"max diff {result['max_abs_diff']:.6f})"
            )
        elif status == "skipped":
            print(f"  {result['code']}: SKIPPED (no data -- not counted as passed)")
        else:
            print(f"  {result['code']}: FAILED")
            failed += 1
    if failed:
        print(f"FAILED: {failed} code(s) did not verify.")
        return 1
    if all(result["status"] == "skipped" for result in results):
        print("FAILED: every code was skipped, nothing was verified.")
        return 1
    print("OK: all codes with data verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
