#!/usr/bin/env python3
"""Verify the DuckDB history pilot: SQL moving averages vs pandas.

Fetches real daily history for one code (default QQQ) through the repo's
own ``DataFetcherManager``, stores it in the local DuckDB file, computes
MA5/MA10/MA20 with SQL window functions, and compares them row by row
against the pandas ``rolling(window).mean()`` results used by
``StockAnalyzer._calculate_mas``.

Exit code 0 means every comparable row agrees to 2 decimals; anything
else (fetch failure, too little data, or any mismatch) exits non-zero
and prints the offending rows. Nothing here changes the online report:
the pandas numbers remain the source of truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from data_provider.base import DataFetcherManager  # noqa: E402
from src.core import duckdb_history  # noqa: E402
from src.core import risk_metrics  # noqa: E402

WINDOWS = (5, 10, 20)
MIN_COMPARED_ROWS = 100


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="QQQ", help="Ticker to verify (default: QQQ)")
    parser.add_argument("--days", type=int, default=365, help="Calendar days to fetch")
    parser.add_argument("--db-path", default=None, help="DuckDB file path override")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Max allowed abs diff (0.005 = identical at 2 decimals)",
    )
    args = parser.parse_args()

    try:
        frame = fetch_history(args.code, args.days)
    except Exception as exc:  # noqa: BLE001 - report and fail loudly
        print(f"FAILED: could not fetch {args.code} history: {exc}")
        return 1
    if len(frame) < max(WINDOWS) + 1:
        print(f"FAILED: only {len(frame)} rows fetched, need more than {max(WINDOWS)}")
        return 1

    db_path = duckdb_history.resolve_db_path(args.db_path)
    stored = duckdb_history.store_daily_bars(db_path, args.code, frame)
    print(f"Stored {stored} rows in {db_path}")

    comparison = duckdb_history.compare_with_pandas(
        db_path, args.code, frame, windows=WINDOWS, tolerance=args.tolerance
    )
    sql_rows = duckdb_history.moving_averages(db_path, args.code, windows=WINDOWS)
    print_recent_table(frame, sql_rows)

    # Same data, same window: also show the QQQ risk metrics (QuantStats
    # definitions) so the pilot's second half has a real-number check.
    metrics = risk_metrics.compute_risk_metrics(frame["close"].tolist())
    print("\nRisk metrics (last 252 trading days, QuantStats definitions):")
    print(f"  sharpe_ratio : {metrics['sharpe_ratio']}")
    print(f"  max_drawdown : {metrics['max_drawdown']}")
    print(f"  calmar_ratio : {metrics['calmar_ratio']}")
    print(f"  trading_days : {metrics['trading_days']}")

    rows_compared = comparison["rows_compared"]
    mismatches = comparison["mismatches"]
    print(
        f"\nCompared {rows_compared} MA values, "
        f"max abs diff {comparison['max_abs_diff']:.6f}, "
        f"mismatches {len(mismatches)}"
    )
    if rows_compared < MIN_COMPARED_ROWS:
        print(f"FAILED: only {rows_compared} rows compared, need >= {MIN_COMPARED_ROWS}")
        return 1
    if mismatches:
        print("FAILED: mismatches (first 10):")
        for mismatch in mismatches[:10]:
            print(f"  {mismatch}")
        return 1
    print("OK: DuckDB moving averages match pandas to 2 decimals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
