from __future__ import annotations

import re

import pandas as pd


_TIMEFRAME_PATTERN = re.compile(r"^(\d+)\s*(m|min|h|d|w)$", re.IGNORECASE)

_PANDAS_UNIT = {"min": "min", "h": "h", "d": "D", "w": "W"}

OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def normalize_timeframe(timeframe: str) -> tuple[int, str]:
    """Parse "1h", "2h", "6h", "30m", "1d", ... into (n, unit)."""
    match = _TIMEFRAME_PATTERN.match(str(timeframe).strip())
    if not match:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")

    n = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "m":
        unit = "min"
    return n, unit


def to_pandas_rule(timeframe: str) -> str:
    n, unit = normalize_timeframe(timeframe)
    return f"{n}{_PANDAS_UNIT[unit]}"


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(to_pandas_rule(timeframe))


def raw_bars_needed(
    target_bars: int,
    source_timeframe: str,
    target_timeframe: str,
    safety_factor: float = 1.5,
) -> int:
    """How many source-timeframe bars to pull to be confident we can build
    `target_bars` complete target-timeframe bars. Padded to survive
    overnight/weekend gaps in the raw data (real bar counts per session
    vary, unlike a plain calendar-time ratio)."""
    ratio = timeframe_to_timedelta(target_timeframe) / timeframe_to_timedelta(source_timeframe)
    return max(target_bars, int(target_bars * ratio * safety_factor) + 1)


def resample_ohlcv(
    bars: pd.DataFrame,
    timeframe: str,
    source_timeframe: str | None = None,
) -> pd.DataFrame:
    """Resample a finer-grained OHLCV frame (e.g. 1-minute bars) up to a
    coarser timeframe (e.g. "1h", "2h", "6h").

    `bars` must have a sorted DatetimeIndex and open/high/low/close/volume
    columns. The final bin is dropped if it isn't fully closed yet (i.e.
    the raw data doesn't extend to that bin's nominal end) -- otherwise a
    strategy could act on a still-forming bar as though it were complete.
    """
    if bars.empty:
        return bars

    rule = to_pandas_rule(timeframe)
    resampled = bars.resample(rule, label="left", closed="left").agg(OHLCV_AGG)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])

    if resampled.empty:
        return resampled

    if source_timeframe is not None:
        source_step = timeframe_to_timedelta(source_timeframe)
    else:
        source_step = pd.Series(bars.index).diff().median()

    last_raw_timestamp = bars.index[-1]
    bin_end = resampled.index[-1] + timeframe_to_timedelta(timeframe)

    if bin_end > last_raw_timestamp + source_step:
        resampled = resampled.iloc[:-1]

    return resampled
