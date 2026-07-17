from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


REQUIRED_OHLC_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
}


def normalize_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a copy with lowercase column names.

    This allows indicators to accept either:

        Open, High, Low, Close, Volume

    or:

        open, high, low, close, volume
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    frame = data.copy()

    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]

    return frame


def require_columns(
    data: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """
    Normalize columns and validate required fields.
    """
    frame = normalize_columns(data)

    required = {
        str(column).strip().lower()
        for column in columns
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    return frame


def require_ohlc(
    data: pd.DataFrame,
) -> pd.DataFrame:
    return require_columns(
        data,
        REQUIRED_OHLC_COLUMNS,
    )


def require_hlc(
    data: pd.DataFrame,
) -> pd.DataFrame:
    return require_columns(
        data,
        {
            "high",
            "low",
            "close",
        },
    )


def validate_period(
    period: int,
    name: str = "period",
) -> int:
    period = int(period)

    if period <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return period


def validate_series(
    series: pd.Series,
    name: str = "series",
) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"{name} must be a pandas Series."
        )

    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)