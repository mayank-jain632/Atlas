from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    require_hlc,
    validate_period,
)


def donchian_channels(
    data: pd.DataFrame,
    period: int = 20,
    shift: int = 1,
) -> pd.DataFrame:
    """
    Donchian upper, lower, and midpoint.

    shift=1 uses only levels known before today's close.
    """
    frame = require_hlc(data)
    period = validate_period(period)

    upper = frame["high"].rolling(
        window=period,
        min_periods=period,
    ).max()

    lower = frame["low"].rolling(
        window=period,
        min_periods=period,
    ).min()

    if int(shift) != 0:
        upper = upper.shift(int(shift))
        lower = lower.shift(int(shift))

    middle = (
        upper + lower
    ) / 2.0

    return pd.DataFrame(
        {
            "upper": upper,
            "lower": lower,
            "middle": middle,
        },
        index=frame.index,
    )


def dual_donchian_channels(
    data: pd.DataFrame,
    exit_period: int = 50,
    entry_period: int = 20,
    shift: int = 1,
) -> pd.DataFrame:
    """
    Separate Donchian levels for exit and re-entry.

    Useful for intervention hysteresis:

        exit below prior exit-period low
        re-enter above prior entry-period high
    """
    frame = require_hlc(data)

    exit_period = validate_period(
        exit_period,
        "exit_period",
    )

    entry_period = validate_period(
        entry_period,
        "entry_period",
    )

    exit_low = frame["low"].rolling(
        window=exit_period,
        min_periods=exit_period,
    ).min()

    entry_high = frame["high"].rolling(
        window=entry_period,
        min_periods=entry_period,
    ).max()

    if int(shift) != 0:
        exit_low = exit_low.shift(int(shift))
        entry_high = entry_high.shift(
            int(shift)
        )

    return pd.DataFrame(
        {
            "exit_low": exit_low,
            "entry_high": entry_high,
        },
        index=frame.index,
    )


def choppiness_index(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Choppiness Index.

    Higher values indicate a more range-bound market.
    Lower values indicate a stronger directional trend.
    """
    from .volatility import true_range

    frame = require_hlc(data)
    period = validate_period(period)

    tr_sum = true_range(frame).rolling(
        window=period,
        min_periods=period,
    ).sum()

    highest_high = frame["high"].rolling(
        window=period,
        min_periods=period,
    ).max()

    lowest_low = frame["low"].rolling(
        window=period,
        min_periods=period,
    ).min()

    price_range = (
        highest_high
        - lowest_low
    )

    ratio = (
        tr_sum
        / price_range.replace(
            0.0,
            np.nan,
        )
    )

    result = (
        100.0
        * np.log10(ratio)
        / np.log10(float(period))
    )

    result.name = f"choppiness_{period}"

    return result