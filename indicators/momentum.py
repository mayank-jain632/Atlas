from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    require_hlc,
    validate_period,
    validate_series,
)


def roc(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Rate of change.
    """
    period = validate_period(period)
    close = validate_series(close, "close")

    result = close.pct_change(
        periods=period,
        fill_method=None,
    )

    result.name = f"roc_{period}"

    return result


def rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Wilder RSI.
    """
    period = validate_period(period)
    close = validate_series(close, "close")

    change = close.diff()

    gains = change.clip(lower=0.0)
    losses = -change.clip(upper=0.0)

    average_gain = gains.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = (
        average_gain
        / average_loss.replace(
            0.0,
            np.nan,
        )
    )

    result = (
        100.0
        - 100.0
        / (
            1.0
            + relative_strength
        )
    )

    result = result.where(
        average_loss != 0.0,
        100.0,
    )

    # No gains and no losses means a flat series.
    both_zero = (
        (average_gain == 0.0)
        & (average_loss == 0.0)
    )

    result = result.where(
        ~both_zero,
        50.0,
    )

    result.name = f"rsi_{period}"

    return result


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.
    """
    fast_period = validate_period(
        fast_period,
        "fast_period",
    )

    slow_period = validate_period(
        slow_period,
        "slow_period",
    )

    signal_period = validate_period(
        signal_period,
        "signal_period",
    )

    if fast_period >= slow_period:
        raise ValueError(
            "fast_period must be smaller than "
            "slow_period."
        )

    close = validate_series(close, "close")

    fast_ema = close.ewm(
        span=fast_period,
        adjust=False,
        min_periods=fast_period,
    ).mean()

    slow_ema = close.ewm(
        span=slow_period,
        adjust=False,
        min_periods=slow_period,
    ).mean()

    macd_line = fast_ema - slow_ema

    signal_line = macd_line.ewm(
        span=signal_period,
        adjust=False,
        min_periods=signal_period,
    ).mean()

    histogram = (
        macd_line
        - signal_line
    )

    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        },
        index=close.index,
    )


def stochastic_oscillator(
    data: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Stochastic oscillator %K and %D.
    """
    frame = require_hlc(data)

    k_period = validate_period(
        k_period,
        "k_period",
    )

    d_period = validate_period(
        d_period,
        "d_period",
    )

    lowest_low = frame["low"].rolling(
        window=k_period,
        min_periods=k_period,
    ).min()

    highest_high = frame["high"].rolling(
        window=k_period,
        min_periods=k_period,
    ).max()

    percent_k = (
        100.0
        * (
            frame["close"]
            - lowest_low
        )
        / (
            highest_high
            - lowest_low
        ).replace(0.0, np.nan)
    )

    percent_d = percent_k.rolling(
        window=d_period,
        min_periods=d_period,
    ).mean()

    return pd.DataFrame(
        {
            "percent_k": percent_k,
            "percent_d": percent_d,
        },
        index=frame.index,
    )