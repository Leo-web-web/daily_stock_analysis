"""Local DuckDB store for daily price history (pilot: QQQ only).

The online report still uses the pandas moving averages from
``src/stock_analyzer.py``; this module is a *parallel verification* path:
history is stored once, moving averages are computed with SQL window
functions, and ``scripts/check_duckdb_ma.py`` checks them against the
pandas results row by row.

The database file is never committed: the default location is
``data/duckdb/history.duckdb`` under the repo root, which ``.gitignore``
already excludes via ``/data/``. Override it with the ``DSA_DUCKDB_PATH``
environment variable or an explicit ``db_path`` argument.

Moving-average semantics match ``StockAnalyzer._calculate_mas``: a value
is only produced once the full window is available (pandas
``rolling(window).mean()`` with the default ``min_periods=window``), so
the first ``window - 1`` rows are NULL in SQL just as they are NaN in
pandas.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Default database file, relative to the repo root. Never committed.
DEFAULT_DB_RELATIVE_PATH = Path("data") / "duckdb" / "history.duckdb"

#: Environment variable that overrides the default database path.
DUCKDB_PATH_ENV_VAR = "DSA_DUCKDB_PATH"

TABLE_NAME = "daily_bars"


def resolve_db_path(db_path: Optional[str] = None) -> Path:
    """Resolve the DuckDB file path: explicit arg > env var > repo default."""
    if db_path:
        return Path(db_path)
    env_path = os.environ.get(DUCKDB_PATH_ENV_VAR, "").strip()
    if env_path:
        return Path(env_path)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / DEFAULT_DB_RELATIVE_PATH


def _connect(db_path: Path):
    """Open a DuckDB connection, creating the parent directory first."""
    import duckdb

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def _ensure_table(con) -> None:
    con.execute(
        f"""CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            code VARCHAR,
            trade_date DATE,
            close DOUBLE,
            volume DOUBLE
        )"""
    )


def _to_date(value) -> object:
    """Normalise a date-like value to ``datetime.date`` for DuckDB."""
    import pandas as pd

    parsed = pd.to_datetime(value)
    return parsed.date()


def store_daily_bars(db_path, code: str, bars) -> int:
    """Store daily bars for ``code``, replacing any rows already stored.

    ``bars`` is a DataFrame with ``date`` and ``close`` columns (the
    standardised fetcher output; ``volume`` is optional). Returns the
    number of rows stored.
    """
    import pandas as pd

    path = resolve_db_path(db_path) if isinstance(db_path, str) else Path(db_path)
    frame = bars[["date", "close"] + (["volume"] if "volume" in bars.columns else [])].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    rows = [
        (
            code,
            _to_date(row.date),
            float(row.close),
            float(row.volume) if "volume" in frame.columns and pd.notna(row.volume) else None,
        )
        for row in frame.itertuples(index=False)
    ]
    with _connect(path) as con:
        _ensure_table(con)
        con.execute(f"DELETE FROM {TABLE_NAME} WHERE code = ?", [code])
        con.executemany(
            f"INSERT INTO {TABLE_NAME} (code, trade_date, close, volume) VALUES (?, ?, ?, ?)",
            rows,
        )
    logger.info("DuckDB: stored %d daily bars for %s in %s", len(rows), code, path)
    return len(rows)


def load_closes(db_path, code: str) -> List[Tuple[str, float]]:
    """Load stored closes for ``code`` as ``[(date_iso, close), ...]`` ascending."""
    path = resolve_db_path(db_path) if isinstance(db_path, str) else Path(db_path)
    if not path.exists():
        return []
    with _connect(path) as con:
        _ensure_table(con)
        rows = con.execute(
            f"SELECT trade_date, close FROM {TABLE_NAME} WHERE code = ? ORDER BY trade_date",
            [code],
        ).fetchall()
    return [(str(trade_date), float(close)) for trade_date, close in rows]


def moving_averages(
    db_path, code: str, windows: Sequence[int] = (5, 10, 20)
) -> List[Dict[str, Optional[float]]]:
    """Moving averages from the store, computed with SQL window functions.

    Returns one dict per stored row, ``{"trade_date", "close", "ma5",
    "ma10", "ma20", ...}`` in date order. A window value is ``None``
    until the full window is available (matching pandas).
    """
    for window in windows:
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"Invalid MA window: {window!r}")
    path = resolve_db_path(db_path) if isinstance(db_path, str) else Path(db_path)
    if not path.exists():
        return []
    select_parts = ["trade_date", "close"]
    for window in windows:
        frame = f"ORDER BY trade_date ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        select_parts.append(
            f"CASE WHEN COUNT(*) OVER ({frame}) >= {window} "
            f"THEN AVG(close) OVER ({frame}) END AS ma{window}"
        )
    query = (
        f"SELECT {', '.join(select_parts)} FROM {TABLE_NAME} "
        f"WHERE code = ? ORDER BY trade_date"
    )
    with _connect(path) as con:
        _ensure_table(con)
        rows = con.execute(query, [code]).fetchall()
    result: List[Dict[str, Optional[float]]] = []
    for row in rows:
        entry: Dict[str, Optional[float]] = {
            "trade_date": str(row[0]),
            "close": float(row[1]),
        }
        for index, window in enumerate(windows):
            value = row[2 + index]
            entry[f"ma{window}"] = None if value is None else float(value)
        result.append(entry)
    return result


def compare_with_pandas(
    db_path,
    code: str,
    bars,
    windows: Sequence[int] = (5, 10, 20),
    tolerance: float = 0.005,
) -> Dict[str, object]:
    """Compare SQL moving averages against pandas ``rolling().mean()``.

    ``bars`` is the same DataFrame that was stored (``date``/``close``
    columns). Only rows where pandas has a value are compared; rows
    where pandas is NaN must also be NULL in SQL. Two values agree when
    they differ by at most ``tolerance`` (0.005 means the 2-decimal
    roundings are identical).

    Returns ``{"rows_compared", "max_abs_diff", "mismatches"}`` where each
    mismatch is ``{"trade_date", "window", "pandas", "duckdb"}`` (either
    value may be ``None`` when one side is missing and the other is not).
    """
    import pandas as pd

    frame = bars[["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    pandas_ma: Dict[str, "pd.Series"] = {}
    for window in windows:
        pandas_ma[f"ma{window}"] = frame["close"].rolling(window=window).mean()
    sql_rows = {row["trade_date"]: row for row in moving_averages(db_path, code, windows)}
    dates = [d.strftime("%Y-%m-%d") for d in frame["date"]]
    mismatches: List[Dict[str, object]] = []
    rows_compared = 0
    max_abs_diff = 0.0
    for position, date_iso in enumerate(dates):
        sql_row = sql_rows.get(date_iso)
        if sql_row is None:
            mismatches.append(
                {"trade_date": date_iso, "window": "all", "pandas": "present", "duckdb": None}
            )
            continue
        for window in windows:
            key = f"ma{window}"
            pandas_value = pandas_ma[key].iloc[position]
            sql_value = sql_row[key]
            if pd.isna(pandas_value):
                if sql_value is not None:
                    mismatches.append(
                        {
                            "trade_date": date_iso,
                            "window": window,
                            "pandas": None,
                            "duckdb": sql_value,
                        }
                    )
                continue
            rows_compared += 1
            if sql_value is None:
                mismatches.append(
                    {
                        "trade_date": date_iso,
                        "window": window,
                        "pandas": float(pandas_value),
                        "duckdb": None,
                    }
                )
                continue
            diff = abs(float(pandas_value) - float(sql_value))
            max_abs_diff = max(max_abs_diff, diff)
            if diff > tolerance:
                mismatches.append(
                    {
                        "trade_date": date_iso,
                        "window": window,
                        "pandas": float(pandas_value),
                        "duckdb": float(sql_value),
                    }
                )
    return {
        "rows_compared": rows_compared,
        "max_abs_diff": max_abs_diff,
        "mismatches": mismatches,
    }


__all__ = [
    "DEFAULT_DB_RELATIVE_PATH",
    "DUCKDB_PATH_ENV_VAR",
    "compare_with_pandas",
    "load_closes",
    "moving_averages",
    "resolve_db_path",
    "store_daily_bars",
]
