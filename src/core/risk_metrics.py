"""Risk metrics for the daily brief (QuantStats-style definitions, no quantstats dependency).

Pilot scope: QQQ only (see ``RISK_METRICS_PILOT_CODES`` for the extension
point). Only the *definitions* are borrowed from QuantStats
(ranaroussi/quantstats); the library itself is intentionally not a
dependency, so the daily job stays light:

- Sharpe ratio: mean excess daily return / std of daily returns,
  annualised with ``sqrt(periods_per_year)`` (risk-free rate 0).
- Maximum drawdown: worst peak-to-trough decline of the price series.
- Calmar ratio: annualised (geometric) return / abs(max drawdown).

All functions are pure (plain ``float`` sequences in, ``float``/``None``
out) so they are easy to unit-test with hand-computed series. A metric
is ``None`` -- never NaN -- when it is undefined (too little data, zero
volatility, or zero drawdown for Calmar); the report renders ``None`` as
the missing-data label.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Trading days in one year (QuantStats default for daily data).
TRADING_DAYS_PER_YEAR = 252

#: Number of most recent trading days the brief's risk metrics cover.
RISK_METRICS_LOOKBACK_DAYS = TRADING_DAYS_PER_YEAR

#: Pilot scope: only these (normalised) codes get a risk-metrics block in
#: the brief. Add a code here to extend the pilot to more tickers.
RISK_METRICS_PILOT_CODES = frozenset({"QQQ"})

#: How many calendar days of history to fetch for one trading year of data.
RISK_METRICS_FETCH_DAYS = 365

#: Process-lifetime cache: one report run fetches each pilot code once.
_RISK_METRICS_CACHE: Dict[str, Optional[Dict[str, Optional[float]]]] = {}


def normalize_code(code: Optional[str]) -> str:
    """Normalise a stock code for pilot-scope checks (``" qqq "`` -> ``"QQQ"``)."""
    return (code or "").strip().upper()


def is_risk_metrics_pilot_code(code: Optional[str]) -> bool:
    """Whether ``code`` is in the risk-metrics pilot scope."""
    return normalize_code(code) in RISK_METRICS_PILOT_CODES


def clear_risk_metrics_cache() -> None:
    """Drop the process-lifetime cache (used by tests)."""
    _RISK_METRICS_CACHE.clear()


def daily_returns(closes: Sequence[float]) -> List[float]:
    """Simple daily returns ``close[i] / close[i-1] - 1``.

    Pairs with a non-positive or non-finite close are skipped.
    """
    returns: List[float] = []
    for previous, current in zip(closes, closes[1:]):
        if (
            previous is None
            or current is None
            or not math.isfinite(previous)
            or not math.isfinite(current)
            or previous <= 0
            or current <= 0
        ):
            continue
        returns.append(current / previous - 1.0)
    return returns


def sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """Annualised Sharpe ratio (QuantStats definition).

    ``mean(excess daily returns) / std(daily returns) * sqrt(periods)``
    with sample standard deviation (ddof=1). Returns ``None`` when there
    are fewer than two returns or volatility is zero.
    """
    excess = [r - risk_free_rate / periods_per_year for r in returns]
    if len(excess) < 2:
        return None
    volatility = statistics.stdev(excess)
    if volatility == 0:
        return None
    return statistics.mean(excess) / volatility * math.sqrt(periods_per_year)


def max_drawdown(closes: Sequence[float]) -> Optional[float]:
    """Worst peak-to-trough decline as a fraction (<= 0).

    ``0.0`` for a monotonically rising series (no decline at all);
    ``None`` when fewer than two valid closes are available.
    """
    valid = [c for c in closes if c is not None and math.isfinite(c) and c > 0]
    if len(valid) < 2:
        return None
    peak = valid[0]
    worst = 0.0
    for close in valid:
        if close > peak:
            peak = close
        drawdown = close / peak - 1.0
        if drawdown < worst:
            worst = drawdown
    return worst


def annualized_return(
    returns: Sequence[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """Annualised geometric return from daily returns."""
    if not returns:
        return None
    compounded = 1.0
    for r in returns:
        compounded *= 1.0 + r
    return compounded ** (periods_per_year / len(returns)) - 1.0


def calmar_ratio(
    returns: Sequence[float],
    closes: Sequence[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """Calmar ratio: annualised return / abs(max drawdown).

    Returns ``None`` when the annualised return is undefined or the
    maximum drawdown is zero (a never-declining series has no Calmar
    ratio); it never raises.
    """
    annual = annualized_return(returns, periods_per_year)
    drawdown = max_drawdown(closes)
    if annual is None or drawdown is None or drawdown == 0:
        return None
    return annual / abs(drawdown)


def compute_risk_metrics(
    closes: Sequence[float],
    lookback_days: int = RISK_METRICS_LOOKBACK_DAYS,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, Optional[float]]:
    """Risk metrics for a price series, using the most recent ``lookback_days`` closes.

    Returns ``{"sharpe_ratio": ..., "max_drawdown": ..., "calmar_ratio": ...,
    "trading_days": ...}`` where a metric is ``None`` when undefined.
    ``trading_days`` is the number of closes actually used.
    """
    valid = [c for c in closes if c is not None and math.isfinite(c) and c > 0]
    window = valid[-lookback_days:] if lookback_days > 0 else valid
    returns = daily_returns(window)
    return {
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(window),
        "calmar_ratio": calmar_ratio(returns, window, periods_per_year),
        "trading_days": float(len(window)),
    }


def fetch_daily_closes(
    code: str, days: int = RISK_METRICS_FETCH_DAYS
) -> Optional[List[Tuple[str, float]]]:
    """Fetch daily closes for ``code`` via the repo's fetcher manager.

    Returns ``[(date_iso, close), ...]`` sorted by date ascending, or
    ``None`` when the fetch fails or yields no usable rows. The import is
    lazy so this module stays importable without the data providers.
    """
    try:
        from data_provider.base import DataFetcherManager

        manager = DataFetcherManager()
        df, _source = manager.get_daily_data(code, days=days)
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.warning("Risk metrics: failed to fetch %s history: %s", code, exc)
        return None
    if df is None or df.empty or "close" not in df.columns or "date" not in df.columns:
        return None
    try:
        import pandas as pd

        frame = df[["date", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        closes = [
            (row.date.strftime("%Y-%m-%d"), float(row.close))
            for row in frame.itertuples(index=False)
            if math.isfinite(float(row.close)) and float(row.close) > 0
        ]
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.warning("Risk metrics: failed to parse %s history: %s", code, exc)
        return None
    return closes or None


def get_risk_metrics_for_code(
    code: Optional[str], days: int = RISK_METRICS_FETCH_DAYS
) -> Optional[Dict[str, Optional[float]]]:
    """Risk metrics for one code, or ``None`` when unavailable.

    Only pilot codes are fetched; anything else returns ``None`` without
    touching the network. Results are cached per code for the lifetime of
    the process so a report rendered more than once fetches only once.
    """
    normalized = normalize_code(code)
    if normalized not in RISK_METRICS_PILOT_CODES:
        return None
    if normalized in _RISK_METRICS_CACHE:
        return _RISK_METRICS_CACHE[normalized]
    try:
        closes = fetch_daily_closes(normalized, days=days)
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.warning("Risk metrics: failed to fetch %s history: %s", normalized, exc)
        closes = None
    if not closes:
        _RISK_METRICS_CACHE[normalized] = None
        return None
    metrics = compute_risk_metrics([close for _date, close in closes])
    _RISK_METRICS_CACHE[normalized] = metrics
    return metrics
