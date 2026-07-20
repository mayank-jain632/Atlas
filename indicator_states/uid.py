from __future__ import annotations

from typing import Any

from indicator_states import (
    create_indicator_state,
)


# indicator_states sits below strategies (strategies consume it, not the
# reverse), so these three helpers are inlined rather than imported from
# strategies.uid_utils -- importing that module would import the whole
# strategies package, which imports back into indicator_states.uid and
# creates a circular import.

def require(parameters: dict[str, str], key: str) -> str:
    if key not in parameters:
        raise ValueError(f"Missing UID parameter: {key}")
    return parameters[key]


def get_int(parameters: dict[str, str], key: str, default: int) -> int:
    return int(parameters.get(key, default))


def get_float(parameters: dict[str, str], key: str, default: float) -> float:
    return float(parameters.get(key, default))


def create_state_from_parameters(
    parameters: dict[str, str],
):
    state_type = require(
        parameters,
        "state",
    ).lower()

    if state_type in {
        "ma_crossover",
        "moving_average_crossover",
    }:
        state_parameters = {
            "fast_period": get_int(
                parameters,
                "fast",
                50,
            ),
            "slow_period": get_int(
                parameters,
                "slow",
                200,
            ),
            "method": parameters.get(
                "method",
                "sma",
            ),
        }

    elif state_type in {
        "sma",
        "ema",
        "moving_average",
    }:
        state_parameters = {
            "period": get_int(
                parameters,
                "period",
                get_int(
                    parameters,
                    "p",
                    200,
                ),
            ),
            "method": (
                state_type
                if state_type in {"sma", "ema"}
                else parameters.get(
                    "method",
                    "sma",
                )
            ),
            "bearish_buffer": get_float(
                parameters,
                "bearish_buffer",
                0.0,
            ),
            "bullish_buffer": get_float(
                parameters,
                "bullish_buffer",
                0.0,
            ),
        }

    elif state_type == "supertrend":
        state_parameters = {
            "period": get_int(
                parameters,
                "period",
                get_int(parameters, "p", 10),
            ),
            "multiplier": get_float(
                parameters,
                "multiplier",
                get_float(parameters, "mult", 3.0),
            ),
        }

    elif state_type == "rsi":
        state_parameters = {
            "period": get_int(
                parameters,
                "period",
                get_int(parameters, "p", 14),
            ),
            "bearish_threshold": get_float(
                parameters,
                "bearish",
                40.0,
            ),
            "bullish_threshold": get_float(
                parameters,
                "bullish",
                50.0,
            ),
        }

    elif state_type == "macd":
        state_parameters = {
            "fast_period": get_int(
                parameters,
                "fast",
                12,
            ),
            "slow_period": get_int(
                parameters,
                "slow",
                26,
            ),
            "signal_period": get_int(
                parameters,
                "signal_period",
                9,
            ),
        }

    elif state_type == "donchian":
        state_parameters = {
            "exit_period": get_int(
                parameters,
                "exit_period",
                50,
            ),
            "entry_period": get_int(
                parameters,
                "entry_period",
                20,
            ),
        }

    elif state_type in {"adx", "adx_trend"}:
        state_parameters = {
            "period": get_int(
                parameters,
                "period",
                14,
            ),
            "bearish_threshold": get_float(
                parameters,
                "bearish",
                15.0,
            ),
            "bullish_threshold": get_float(
                parameters,
                "bullish",
                20.0,
            ),
        }

    elif state_type in {
        "psar",
        "parabolic_sar",
    }:
        state_parameters = {
            "step": get_float(
                parameters,
                "step",
                0.02,
            ),
            "max_step": get_float(
                parameters,
                "max_step",
                0.20,
            ),
        }

    else:
        raise ValueError(
            f"Unsupported state type: {state_type}"
        )

    evaluator = create_indicator_state(
        state_type=state_type,
        parameters=state_parameters,
    )

    return (
        evaluator,
        state_type,
        state_parameters,
    )