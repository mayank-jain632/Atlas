from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    require_hlc,
    validate_period,
)
from .volatility import atr


def adx(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Average Directional Index with +DI and -DI.
    """
    frame = require_hlc(data)
    period = validate_period(period)

    high_change = frame["high"].diff()
    low_change = -frame["low"].diff()

    positive_dm = pd.Series(
        np.where(
            (
                high_change
                > low_change
            )
            & (
                high_change
                > 0.0
            ),
            high_change,
            0.0,
        ),
        index=frame.index,
        dtype=float,
    )

    negative_dm = pd.Series(
        np.where(
            (
                low_change
                > high_change
            )
            & (
                low_change
                > 0.0
            ),
            low_change,
            0.0,
        ),
        index=frame.index,
        dtype=float,
    )

    average_true_range = atr(
        frame,
        period=period,
    )

    positive_smoothed = positive_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    negative_smoothed = negative_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    positive_di = (
        100.0
        * positive_smoothed
        / average_true_range.replace(
            0.0,
            np.nan,
        )
    )

    negative_di = (
        100.0
        * negative_smoothed
        / average_true_range.replace(
            0.0,
            np.nan,
        )
    )

    dx = (
        100.0
        * (
            positive_di
            - negative_di
        ).abs()
        / (
            positive_di
            + negative_di
        ).replace(
            0.0,
            np.nan,
        )
    )

    adx_value = dx.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return pd.DataFrame(
        {
            "adx": adx_value,
            "positive_di": positive_di,
            "negative_di": negative_di,
            "dx": dx,
        },
        index=frame.index,
    )


def supertrend(
    data: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Supertrend indicator.

    direction:
         1 = bullish
        -1 = bearish
    """
    frame = require_hlc(data)
    period = validate_period(period)

    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)

    average_true_range = atr(
        frame,
        period=period,
    )

    midpoint = (
        high + low
    ) / 2.0

    basic_upper = (
        midpoint
        + float(multiplier)
        * average_true_range
    )

    basic_lower = (
        midpoint
        - float(multiplier)
        * average_true_range
    )

    final_upper = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    final_lower = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    line = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    direction = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    valid_positions = np.flatnonzero(
        average_true_range.notna()
    )

    if len(valid_positions) == 0:
        return pd.DataFrame(
            {
                "supertrend": line,
                "direction": direction,
                "upper_band": final_upper,
                "lower_band": final_lower,
                "atr": average_true_range,
            },
            index=frame.index,
        )

    first = int(valid_positions[0])

    final_upper.iloc[first] = (
        basic_upper.iloc[first]
    )

    final_lower.iloc[first] = (
        basic_lower.iloc[first]
    )

    direction.iloc[first] = 1.0
    line.iloc[first] = (
        final_lower.iloc[first]
    )

    for index in range(
        first + 1,
        len(frame),
    ):
        previous = index - 1

        current_basic_upper = (
            basic_upper.iloc[index]
        )

        current_basic_lower = (
            basic_lower.iloc[index]
        )

        previous_final_upper = (
            final_upper.iloc[previous]
        )

        previous_final_lower = (
            final_lower.iloc[previous]
        )

        previous_close = close.iloc[
            previous
        ]

        current_close = close.iloc[
            index
        ]

        if (
            current_basic_upper
            < previous_final_upper
            or previous_close
            > previous_final_upper
        ):
            final_upper.iloc[index] = (
                current_basic_upper
            )
        else:
            final_upper.iloc[index] = (
                previous_final_upper
            )

        if (
            current_basic_lower
            > previous_final_lower
            or previous_close
            < previous_final_lower
        ):
            final_lower.iloc[index] = (
                current_basic_lower
            )
        else:
            final_lower.iloc[index] = (
                previous_final_lower
            )

        previous_direction = (
            direction.iloc[previous]
        )

        if (
            previous_direction == -1.0
            and current_close
            > final_upper.iloc[index]
        ):
            current_direction = 1.0

        elif (
            previous_direction == 1.0
            and current_close
            < final_lower.iloc[index]
        ):
            current_direction = -1.0

        else:
            current_direction = (
                previous_direction
            )

        direction.iloc[index] = (
            current_direction
        )

        if current_direction == 1.0:
            line.iloc[index] = (
                final_lower.iloc[index]
            )
        else:
            line.iloc[index] = (
                final_upper.iloc[index]
            )

    return pd.DataFrame(
        {
            "supertrend": line,
            "direction": direction,
            "upper_band": final_upper,
            "lower_band": final_lower,
            "atr": average_true_range,
        },
        index=frame.index,
    )


def parabolic_sar(
    data: pd.DataFrame,
    step: float = 0.02,
    max_step: float = 0.20,
) -> pd.DataFrame:
    """
    Parabolic SAR.

    direction:
         1 = bullish
        -1 = bearish
    """
    frame = require_hlc(data)

    if step <= 0:
        raise ValueError(
            "step must be positive."
        )

    if max_step < step:
        raise ValueError(
            "max_step must be >= step."
        )

    high = frame["high"].to_numpy(
        dtype=float
    )

    low = frame["low"].to_numpy(
        dtype=float
    )

    close = frame["close"].to_numpy(
        dtype=float
    )

    count = len(frame)

    sar_values = np.full(
        count,
        np.nan,
        dtype=float,
    )

    direction_values = np.full(
        count,
        np.nan,
        dtype=float,
    )

    extreme_point_values = np.full(
        count,
        np.nan,
        dtype=float,
    )

    acceleration_values = np.full(
        count,
        np.nan,
        dtype=float,
    )

    if count < 2:
        return pd.DataFrame(
            {
                "sar": sar_values,
                "direction": direction_values,
                "extreme_point": extreme_point_values,
                "acceleration_factor": acceleration_values,
            },
            index=frame.index,
        )

    bullish = (
        close[1] >= close[0]
    )

    acceleration = float(step)

    if bullish:
        sar_value = low[0]
        extreme_point = max(
            high[0],
            high[1],
        )
    else:
        sar_value = high[0]
        extreme_point = min(
            low[0],
            low[1],
        )

    sar_values[1] = sar_value
    direction_values[1] = (
        1.0 if bullish else -1.0
    )

    extreme_point_values[1] = (
        extreme_point
    )

    acceleration_values[1] = (
        acceleration
    )

    for index in range(2, count):
        sar_value = (
            sar_value
            + acceleration
            * (
                extreme_point
                - sar_value
            )
        )

        if bullish:
            sar_value = min(
                sar_value,
                low[index - 1],
                low[index - 2],
            )

            if low[index] < sar_value:
                bullish = False
                sar_value = extreme_point
                extreme_point = low[index]
                acceleration = float(step)

            else:
                if high[index] > extreme_point:
                    extreme_point = high[index]
                    acceleration = min(
                        acceleration + step,
                        max_step,
                    )

        else:
            sar_value = max(
                sar_value,
                high[index - 1],
                high[index - 2],
            )

            if high[index] > sar_value:
                bullish = True
                sar_value = extreme_point
                extreme_point = high[index]
                acceleration = float(step)

            else:
                if low[index] < extreme_point:
                    extreme_point = low[index]
                    acceleration = min(
                        acceleration + step,
                        max_step,
                    )

        sar_values[index] = sar_value

        direction_values[index] = (
            1.0 if bullish else -1.0
        )

        extreme_point_values[index] = (
            extreme_point
        )

        acceleration_values[index] = (
            acceleration
        )

    return pd.DataFrame(
        {
            "sar": sar_values,
            "direction": direction_values,
            "extreme_point": extreme_point_values,
            "acceleration_factor": acceleration_values,
        },
        index=frame.index,
    )


def aroon(
    data: pd.DataFrame,
    period: int = 25,
) -> pd.DataFrame:
    """
    Aroon Up, Aroon Down, and oscillator.
    """
    frame = require_hlc(data)
    period = validate_period(period)

    def periods_since_high(
        window: np.ndarray,
    ) -> float:
        return float(
            len(window)
            - 1
            - np.argmax(window)
        )

    def periods_since_low(
        window: np.ndarray,
    ) -> float:
        return float(
            len(window)
            - 1
            - np.argmin(window)
        )

    since_high = frame["high"].rolling(
        window=period + 1,
        min_periods=period + 1,
    ).apply(
        periods_since_high,
        raw=True,
    )

    since_low = frame["low"].rolling(
        window=period + 1,
        min_periods=period + 1,
    ).apply(
        periods_since_low,
        raw=True,
    )

    aroon_up = (
        100.0
        * (
            period - since_high
        )
        / period
    )

    aroon_down = (
        100.0
        * (
            period - since_low
        )
        / period
    )

    oscillator = (
        aroon_up - aroon_down
    )

    return pd.DataFrame(
        {
            "aroon_up": aroon_up,
            "aroon_down": aroon_down,
            "oscillator": oscillator,
        },
        index=frame.index,
    )