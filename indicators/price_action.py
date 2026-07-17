from __future__ import annotations

import pandas as pd

from .moving_average import sma
from .utils import (
    validate_period,
    validate_series,
)


def rolling_high(
    close: pd.Series,
    period: int = 252,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Rolling closing-price high.
    """
    period = validate_period(period)
    close = validate_series(close, "close")

    if min_periods is None:
        min_periods = period

    result = close.rolling(
        window=period,
        min_periods=int(min_periods),
    ).max()

    result.name = f"rolling_high_{period}"

    return result


def rolling_low(
    close: pd.Series,
    period: int = 252,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Rolling closing-price low.
    """
    period = validate_period(period)
    close = validate_series(close, "close")

    if min_periods is None:
        min_periods = period

    result = close.rolling(
        window=period,
        min_periods=int(min_periods),
    ).min()

    result.name = f"rolling_low_{period}"

    return result


def drawdown_from_rolling_high(
    close: pd.Series,
    period: int = 252,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Drawdown from a rolling closing-price high.
    """
    close = validate_series(close, "close")

    high = rolling_high(
        close,
        period=period,
        min_periods=min_periods,
    )

    drawdown = (
        close / high - 1.0
    )

    return pd.DataFrame(
        {
            "rolling_high": high,
            "drawdown": drawdown,
        },
        index=close.index,
    )


def recovery_moving_average(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Moving average used to confirm drawdown recovery.
    """
    result = sma(
        close,
        period=period,
    )

    result.name = (
        f"recovery_ma_{period}"
    )

    return result


def distance_from_high(
    close: pd.Series,
    period: int = 252,
) -> pd.Series:
    """
    Percentage distance from rolling high.
    """
    result = drawdown_from_rolling_high(
        close,
        period=period,
        min_periods=1,
    )["drawdown"]

    result.name = (
        f"distance_from_high_{period}"
    )

    return result