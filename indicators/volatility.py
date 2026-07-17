from __future__ import annotations

import numpy as np
import pandas as pd

from .moving_average import ema
from .utils import (
    require_hlc,
    validate_period,
    validate_series,
)


def true_range(
    data: pd.DataFrame,
) -> pd.Series:
    """
    True Range:

        max(
            high - low,
            abs(high - previous close),
            abs(low - previous close),
        )
    """
    frame = require_hlc(data)

    previous_close = frame["close"].shift(1)

    result = pd.concat(
        [
            frame["high"] - frame["low"],
            (
                frame["high"]
                - previous_close
            ).abs(),
            (
                frame["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result.name = "true_range"

    return result


def atr(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Wilder Average True Range.
    """
    period = validate_period(period)

    tr = true_range(data)

    result = tr.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    result.name = f"atr_{period}"

    return result


def rolling_volatility(
    close: pd.Series,
    period: int = 20,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Annualized rolling volatility of close-to-close returns.
    """
    period = validate_period(period)
    close = validate_series(close, "close")

    returns = close.pct_change(
        fill_method=None
    )

    result = (
        returns.rolling(
            window=period,
            min_periods=period,
        ).std(ddof=1)
        * np.sqrt(annualization_factor)
    )

    result.name = f"volatility_{period}"

    return result


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    standard_deviations: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.
    """
    period = validate_period(period)
    close = validate_series(close, "close")

    middle = close.rolling(
        window=period,
        min_periods=period,
    ).mean()

    deviation = close.rolling(
        window=period,
        min_periods=period,
    ).std(ddof=0)

    upper = (
        middle
        + float(standard_deviations)
        * deviation
    )

    lower = (
        middle
        - float(standard_deviations)
        * deviation
    )

    width = (
        upper - lower
    ) / middle.replace(0.0, np.nan)

    percent_b = (
        close - lower
    ) / (
        upper - lower
    ).replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "width": width,
            "percent_b": percent_b,
        },
        index=close.index,
    )


def keltner_channels(
    data: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Keltner Channels.
    """
    frame = require_hlc(data)

    middle = ema(
        frame["close"],
        period=ema_period,
    )

    average_range = atr(
        frame,
        period=atr_period,
    )

    upper = (
        middle
        + float(multiplier)
        * average_range
    )

    lower = (
        middle
        - float(multiplier)
        * average_range
    )

    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "atr": average_range,
        },
        index=frame.index,
    )