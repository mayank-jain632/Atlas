from __future__ import annotations

from typing import Any

from .base import IndicatorState

from .states import (
    ADXTrendState,
    ChoppinessState,
    DonchianState,
    DrawdownRecoveryState,
    MACDState,
    MovingAverageCrossoverState,
    MovingAverageState,
    ParabolicSARState,
    RSIState,
    SupertrendState,
)


STATE_CLASSES = {
    "moving_average": MovingAverageState,
    "ma": MovingAverageState,
    "sma": MovingAverageState,
    "ema": MovingAverageState,

    "moving_average_crossover": (
        MovingAverageCrossoverState
    ),
    "ma_crossover": (
        MovingAverageCrossoverState
    ),

    "rsi": RSIState,
    "macd": MACDState,
    "supertrend": SupertrendState,
    "donchian": DonchianState,

    "drawdown_recovery": (
        DrawdownRecoveryState
    ),

    "adx": ADXTrendState,
    "adx_trend": ADXTrendState,

    "psar": ParabolicSARState,
    "parabolic_sar": ParabolicSARState,

    "choppiness": ChoppinessState,
    "chop": ChoppinessState,
}


def create_indicator_state(
    state_type: str,
    parameters: dict[str, Any] | None = None,
) -> IndicatorState:
    """
    Create an IndicatorState evaluator from a configuration.

    Example:

        create_indicator_state(
            "moving_average",
            {
                "period": 200,
                "method": "sma",
            },
        )
    """
    state_type = str(
        state_type
    ).strip().lower()

    if state_type not in STATE_CLASSES:
        raise ValueError(
            f"Unsupported indicator state: "
            f"{state_type}. "
            f"Expected one of "
            f"{sorted(STATE_CLASSES)}."
        )

    parameters = dict(
        parameters or {}
    )

    # Convenience aliases choose the MA method.
    if state_type == "sma":
        parameters.setdefault(
            "method",
            "sma",
        )

    if state_type == "ema":
        parameters.setdefault(
            "method",
            "ema",
        )

    state_class = STATE_CLASSES[
        state_type
    ]

    return state_class(
        **parameters
    )