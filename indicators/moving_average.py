from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    validate_period,
    validate_series,
)


def sma(
    series: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Simple moving average.
    """
    period = validate_period(period)
    values = validate_series(series)

    result = values.rolling(
        window=period,
        min_periods=period,
    ).mean()

    result.name = f"sma_{period}"

    return result


def ema(
    series: pd.Series,
    period: int = 20,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Exponential moving average.
    """
    period = validate_period(period)
    values = validate_series(series)

    if min_periods is None:
        min_periods = period

    result = values.ewm(
        span=period,
        adjust=False,
        min_periods=int(min_periods),
    ).mean()

    result.name = f"ema_{period}"

    return result


def wilder_average(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Wilder-style exponential smoothing.

    Alpha = 1 / period
    """
    period = validate_period(period)
    values = validate_series(series)

    result = values.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    result.name = f"wilder_{period}"

    return result


def wma(
    series: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Linearly weighted moving average.
    """
    period = validate_period(period)
    values = validate_series(series)

    weights = np.arange(
        1,
        period + 1,
        dtype=float,
    )

    denominator = float(weights.sum())

    result = values.rolling(
        window=period,
        min_periods=period,
    ).apply(
        lambda window: float(
            np.dot(window, weights)
            / denominator
        ),
        raw=True,
    )

    result.name = f"wma_{period}"

    return result


def moving_average_crossover(
    series: pd.Series,
    fast_period: int = 50,
    slow_period: int = 200,
    method: str = "sma",
) -> pd.DataFrame:
    """
    Return fast MA, slow MA, spread, and direction.

    direction:
         1 = fast MA above slow MA
        -1 = fast MA below slow MA
    """
    fast_period = validate_period(
        fast_period,
        "fast_period",
    )

    slow_period = validate_period(
        slow_period,
        "slow_period",
    )

    if fast_period >= slow_period:
        raise ValueError(
            "fast_period must be smaller than "
            "slow_period."
        )

    method = method.lower()

    if method == "sma":
        fast = sma(series, fast_period)
        slow = sma(series, slow_period)

    elif method == "ema":
        fast = ema(series, fast_period)
        slow = ema(series, slow_period)

    elif method == "wma":
        fast = wma(series, fast_period)
        slow = wma(series, slow_period)

    else:
        raise ValueError(
            "method must be one of: "
            "'sma', 'ema', 'wma'."
        )

    spread = (
        fast - slow
    ) / slow.replace(0.0, np.nan)

    direction = pd.Series(
        np.where(
            fast > slow,
            1.0,
            -1.0,
        ),
        index=series.index,
        dtype=float,
    )

    direction = direction.where(
        fast.notna() & slow.notna()
    )

    return pd.DataFrame(
        {
            "fast_ma": fast,
            "slow_ma": slow,
            "spread": spread,
            "direction": direction,
        },
        index=series.index,
    )