from __future__ import annotations

from typing import Any


STANDARD_STRATEGY = "momentum"
DIVERSITY_STRATEGY = "momentum_diversity"

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


def _parse_uid_parts(uid: str) -> tuple[str, dict[str, str]]:
    """
    Parse a UID such as:

        momentum__u=sp500__sig=price__lb=90__rb=monthly__n=10__alloc=score

    Returns:
        strategy_type
        raw string parameters
    """
    if not uid or not isinstance(uid, str):
        raise ValueError("UID must be a non-empty string.")

    parts = uid.strip().split("__")

    strategy_type = parts[0].strip()

    if strategy_type not in {
        STANDARD_STRATEGY,
        DIVERSITY_STRATEGY,
    }:
        raise ValueError(
            f"Unsupported momentum strategy type: {strategy_type}"
        )

    raw_params: dict[str, str] = {}

    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(
                f"Invalid UID component '{part}'. "
                "Expected key=value."
            )

        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or not value:
            raise ValueError(
                f"Invalid UID component '{part}'."
            )

        if key in raw_params:
            raise ValueError(
                f"Duplicate UID parameter: {key}"
            )

        raw_params[key] = value

    return strategy_type, raw_params


def default_parameters(
    strategy_type: str = STANDARD_STRATEGY,
) -> dict[str, Any]:
    params: dict[str, Any] = {
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
        params.update(
            {
                "diversity_method": "graph_cut",
                "diversity_lookback": 60,
                "diversity_lambda": 0.25,
            }
        )

    return params


def parse_uid(uid: str) -> dict[str, Any]:
    strategy_type, raw = _parse_uid_parts(uid)

    params = default_parameters(strategy_type)

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
        if short_key not in key_map:
            raise ValueError(
                f"Unsupported UID parameter: {short_key}"
            )

        parameter_name = key_map[short_key]

        if parameter_name in integer_fields:
            params[parameter_name] = int(raw_value)
        elif parameter_name in float_fields:
            params[parameter_name] = float(raw_value)
        else:
            params[parameter_name] = raw_value

    validate_parameters(params)

    return params


def validate_parameters(params: dict[str, Any]) -> None:
    strategy_type = params["strategy_type"]

    if strategy_type not in {
        STANDARD_STRATEGY,
        DIVERSITY_STRATEGY,
    }:
        raise ValueError(
            f"Unsupported strategy_type: {strategy_type}"
        )

    if params["signal"] not in SUPPORTED_SIGNALS:
        raise ValueError(
            f"Unsupported signal: {params['signal']}. "
            f"Expected one of {sorted(SUPPORTED_SIGNALS)}."
        )

    if (
        params["rebalance_period"]
        not in SUPPORTED_REBALANCE_PERIODS
    ):
        raise ValueError(
            "Unsupported rebalance_period: "
            f"{params['rebalance_period']}"
        )

    if params["allocator"] not in SUPPORTED_ALLOCATORS:
        raise ValueError(
            f"Unsupported allocator: {params['allocator']}"
        )

    if int(params["lookback"]) <= 1:
        raise ValueError("lookback must be greater than 1.")

    if int(params["top_n"]) <= 0:
        raise ValueError("top_n must be positive.")

    if int(params["rsi_window"]) <= 1:
        raise ValueError("rsi_window must be greater than 1.")

    if int(params["ma_short_window"]) <= 0:
        raise ValueError(
            "ma_short_window must be positive."
        )

    if int(params["ma_long_window"]) <= 0:
        raise ValueError(
            "ma_long_window must be positive."
        )

    if (
        int(params["ma_short_window"])
        >= int(params["ma_long_window"])
    ):
        raise ValueError(
            "ma_short_window must be smaller than "
            "ma_long_window."
        )

    if strategy_type == DIVERSITY_STRATEGY:
        method = params["diversity_method"]

        if method not in SUPPORTED_DIVERSITY_METHODS:
            raise ValueError(
                f"Unsupported diversity method: {method}"
            )

        if int(params["diversity_lookback"]) <= 1:
            raise ValueError(
                "diversity_lookback must be greater than 1."
            )

        if float(params["diversity_lambda"]) < 0:
            raise ValueError(
                "diversity_lambda cannot be negative."
            )


def build_uid(params: dict[str, Any]) -> str:
    """
    Build a canonical UID from a parameter dictionary.
    """
    merged = default_parameters(
        params.get("strategy_type", STANDARD_STRATEGY)
    )
    merged.update(params)

    validate_parameters(merged)

    parts = [
        merged["strategy_type"],
        f"u={merged['universe']}",
        f"sig={merged['signal']}",
        f"lb={int(merged['lookback'])}",
        f"rb={merged['rebalance_period']}",
        f"n={int(merged['top_n'])}",
    ]

    # Include non-default signal parameters only when relevant.
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

    if merged["strategy_type"] == DIVERSITY_STRATEGY:
        parts.extend(
            [
                f"div={merged['diversity_method']}",
                f"dlb={int(merged['diversity_lookback'])}",
                f"lam={float(merged['diversity_lambda']):g}",
            ]
        )

    parts.append(f"alloc={merged['allocator']}")

    if float(merged["minimum_score"]) != 0.0:
        parts.append(
            f"minscore={float(merged['minimum_score']):g}"
        )

    return "__".join(parts)