from __future__ import annotations

from typing import Any


STANDARD_STRATEGY = "momentum"
DIVERSITY_STRATEGY = "momentum_diversity"
INDICATOR_STRATEGY = "momentum_indicator"

SUPPORTED_STRATEGIES = {
    STANDARD_STRATEGY,
    DIVERSITY_STRATEGY,
    INDICATOR_STRATEGY,
}

SUPPORTED_SIGNALS = {
    "price",
    "rsi",
    "ma_cross",
    "vol_adj",
    "low_vol",
    "trend_quality",
}

SUPPORTED_REBALANCE_PERIODS = {
    "half_monthly",
    "monthly",
    "two_monthly",
    "quarterly",
}

SUPPORTED_ALLOCATORS = {
    "score",
    "equal",
}

SUPPORTED_DIVERSITY_METHODS = {
    "graph_cut",
    "facility_location",
}


# ============================================================
# GENERIC UID TOKENIZATION
# ============================================================

def parse_uid_parts(
    uid: str,
) -> tuple[str, dict[str, str]]:
    """
    Split a momentum-family UID into its prefix and raw values.

    Example:

        momentum__u=sp500__sig=price__lb=90

    Returns:

        "momentum"

        {
            "u": "sp500",
            "sig": "price",
            "lb": "90",
        }
    """
    if not uid or not isinstance(uid, str):
        raise ValueError(
            "UID must be a non-empty string."
        )

    parts = uid.strip().split("__")

    strategy_type = (
        parts[0]
        .strip()
        .lower()
    )

    if strategy_type not in SUPPORTED_STRATEGIES:
        raise ValueError(
            "Unsupported momentum strategy type: "
            f"{strategy_type}"
        )

    raw_parameters: dict[str, str] = {}

    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(
                f"Invalid UID component '{part}'. "
                "Expected key=value."
            )

        key, value = part.split(
            "=",
            1,
        )

        key = key.strip().lower()
        value = value.strip()

        if not key or not value:
            raise ValueError(
                f"Invalid UID component '{part}'."
            )

        if key in raw_parameters:
            raise ValueError(
                f"Duplicate UID parameter: {key}"
            )

        raw_parameters[key] = value

    return strategy_type, raw_parameters


# Backward-compatible private name if another file imports it.
_parse_uid_parts = parse_uid_parts


# ============================================================
# TYPE HELPERS
# ============================================================

def parse_bool(
    value: str,
    parameter_name: str,
) -> bool:
    normalized = (
        str(value)
        .strip()
        .lower()
    )

    if normalized in {
        "true",
        "1",
        "yes",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"UID parameter '{parameter_name}' "
        "must be true or false."
    )


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

def default_parameters(
    strategy_type: str = STANDARD_STRATEGY,
) -> dict[str, Any]:
    if strategy_type not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported strategy type: {strategy_type}"
        )

    parameters: dict[str, Any] = {
        "strategy_type": strategy_type,
        "universe": "sp500",
        "signal": "price",
        "lookback": 90,
        "rebalance_period": "monthly",
        "top_n": 10,
        "allocator": "score",

        # Signal-specific defaults
        "rsi_window": 14,
        "rsi_threshold": 50.0,
        "ma_short_window": 40,
        "ma_long_window": 100,

        # General behavior
        "minimum_score": 0.0,
    }

    if strategy_type == DIVERSITY_STRATEGY:
        parameters.update(
            {
                "diversity_method": (
                    "graph_cut"
                ),
                "diversity_lookback": 60,
                "diversity_lambda": 0.25,
            }
        )

    return parameters


# ============================================================
# STANDARD MOMENTUM PARAMETER PARSING
# ============================================================

def parse_uid(
    uid: str,
) -> dict[str, Any]:
    """
    Parse standard momentum parameters from any momentum-family
    UID.

    Indicator-specific parameters remain the responsibility of
    MomentumIndicatorStrategy.
    """
    strategy_type, raw = parse_uid_parts(
        uid
    )

    parameters = default_parameters(
        strategy_type
    )

    key_map = {
        "u": "universe",
        "sig": "signal",
        "lb": "lookback",
        "rb": "rebalance_period",
        "n": "top_n",
        "alloc": "allocator",
        "div": "diversity_method",
        "dlb": "diversity_lookback",
        "lam": "diversity_lambda",
        "rsiw": "rsi_window",
        "rsit": "rsi_threshold",
        "short": "ma_short_window",
        "long": "ma_long_window",
        "minscore": "minimum_score",
    }

    integer_fields = {
        "lookback",
        "top_n",
        "diversity_lookback",
        "rsi_window",
        "ma_short_window",
        "ma_long_window",
    }

    float_fields = {
        "diversity_lambda",
        "rsi_threshold",
        "minimum_score",
    }

    for short_key, raw_value in raw.items():
        # Ignore indicator-filter parameters here. The
        # MomentumIndicatorStrategy will parse them.
        if short_key not in key_map:
            if strategy_type == INDICATOR_STRATEGY:
                continue

            raise ValueError(
                "Unsupported UID parameter: "
                f"{short_key}"
            )

        parameter_name = key_map[
            short_key
        ]

        if parameter_name in integer_fields:
            parameters[parameter_name] = int(
                raw_value
            )

        elif parameter_name in float_fields:
            parameters[parameter_name] = float(
                raw_value
            )

        else:
            parameters[parameter_name] = (
                raw_value
            )

    validate_parameters(parameters)

    return parameters


# ============================================================
# VALIDATION
# ============================================================

def validate_parameters(
    parameters: dict[str, Any],
) -> None:
    strategy_type = parameters[
        "strategy_type"
    ]

    if strategy_type not in SUPPORTED_STRATEGIES:
        raise ValueError(
            "Unsupported strategy_type: "
            f"{strategy_type}"
        )

    if parameters["signal"] not in SUPPORTED_SIGNALS:
        raise ValueError(
            f"Unsupported signal: "
            f"{parameters['signal']}. "
            f"Expected one of "
            f"{sorted(SUPPORTED_SIGNALS)}."
        )

    if (
        parameters["rebalance_period"]
        not in SUPPORTED_REBALANCE_PERIODS
    ):
        raise ValueError(
            "Unsupported rebalance_period: "
            f"{parameters['rebalance_period']}"
        )

    if (
        parameters["allocator"]
        not in SUPPORTED_ALLOCATORS
    ):
        raise ValueError(
            "Unsupported allocator: "
            f"{parameters['allocator']}"
        )

    if int(parameters["lookback"]) <= 1:
        raise ValueError(
            "lookback must be greater than 1."
        )

    if int(parameters["top_n"]) <= 0:
        raise ValueError(
            "top_n must be positive."
        )

    if int(parameters["rsi_window"]) <= 1:
        raise ValueError(
            "rsi_window must be greater than 1."
        )

    if int(
        parameters["ma_short_window"]
    ) <= 0:
        raise ValueError(
            "ma_short_window must be positive."
        )

    if int(
        parameters["ma_long_window"]
    ) <= 0:
        raise ValueError(
            "ma_long_window must be positive."
        )

    if (
        int(parameters["ma_short_window"])
        >= int(parameters["ma_long_window"])
    ):
        raise ValueError(
            "ma_short_window must be smaller "
            "than ma_long_window."
        )

    if strategy_type == DIVERSITY_STRATEGY:
        method = parameters[
            "diversity_method"
        ]

        if (
            method
            not in SUPPORTED_DIVERSITY_METHODS
        ):
            raise ValueError(
                "Unsupported diversity method: "
                f"{method}"
            )

        if int(
            parameters["diversity_lookback"]
        ) <= 1:
            raise ValueError(
                "diversity_lookback must be "
                "greater than 1."
            )

        if float(
            parameters["diversity_lambda"]
        ) < 0:
            raise ValueError(
                "diversity_lambda cannot be "
                "negative."
            )


# ============================================================
# UID BUILDING
# ============================================================

def build_uid(
    parameters: dict[str, Any],
) -> str:
    """
    Build a canonical standard or diversity momentum UID.

    MomentumIndicatorStrategy UIDs should generally be supplied
    directly because their state parameters differ by indicator.
    """
    merged = default_parameters(
        parameters.get(
            "strategy_type",
            STANDARD_STRATEGY,
        )
    )

    merged.update(parameters)

    validate_parameters(merged)

    parts = [
        merged["strategy_type"],
        f"u={merged['universe']}",
        f"sig={merged['signal']}",
        f"lb={int(merged['lookback'])}",
        f"rb={merged['rebalance_period']}",
        f"n={int(merged['top_n'])}",
    ]

    if merged["signal"] == "rsi":
        parts.extend(
            [
                f"rsiw={int(merged['rsi_window'])}",
                f"rsit={float(merged['rsi_threshold']):g}",
            ]
        )

    if merged["signal"] == "ma_cross":
        parts.extend(
            [
                f"short={int(merged['ma_short_window'])}",
                f"long={int(merged['ma_long_window'])}",
            ]
        )

    if (
        merged["strategy_type"]
        == DIVERSITY_STRATEGY
    ):
        parts.extend(
            [
                f"div={merged['diversity_method']}",
                f"dlb={int(merged['diversity_lookback'])}",
                f"lam={float(merged['diversity_lambda']):g}",
            ]
        )

    parts.append(
        f"alloc={merged['allocator']}"
    )

    if float(
        merged["minimum_score"]
    ) != 0.0:
        parts.append(
            "minscore="
            f"{float(merged['minimum_score']):g}"
        )

    return "__".join(parts)