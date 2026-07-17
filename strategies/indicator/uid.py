from __future__ import annotations

from typing import Any

from indicator_states.uid import (
    create_state_from_parameters,
)


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


def parse_uid_parts(
    uid: str,
) -> dict[str, str]:
    """
    Parse:

        indicator_basket
        __weights=SPY:0.25,QQQ:0.25
        __renorm=false
        __state=ma_crossover
        __fast=50
        __slow=200
        __method=sma
    """
    if not uid or not isinstance(uid, str):
        raise ValueError(
            "UID must be a non-empty string."
        )

    parts = uid.strip().split("__")

    prefix = (
        parts[0]
        .strip()
        .lower()
    )

    if prefix != "indicator_basket":
        raise ValueError(
            "IndicatorBasketStrategy requires "
            "a UID beginning with "
            "'indicator_basket'."
        )

    parameters: dict[str, str] = {}

    for token in parts[1:]:
        token = token.strip()

        if not token:
            continue

        if "=" not in token:
            raise ValueError(
                f"Invalid UID component: {token}"
            )

        key, value = token.split(
            "=",
            1,
        )

        key = key.strip().lower()
        value = value.strip()

        if not key or not value:
            raise ValueError(
                f"Invalid UID component: {token}"
            )

        if key in parameters:
            raise ValueError(
                f"Duplicate UID parameter: {key}"
            )

        parameters[key] = value

    return parameters


def parse_weight_string(
    value: str,
) -> dict[str, float]:
    """
    Parse:

        SPY:0.25,QQQ:0.25,GLD:0.25,TLT:0.25
    """
    weights: dict[str, float] = {}

    for token in str(value).split(","):
        token = token.strip()

        if not token:
            continue

        if ":" not in token:
            raise ValueError(
                f"Invalid basket weight: {token}"
            )

        symbol, weight_text = token.split(
            ":",
            1,
        )

        symbol = (
            symbol
            .strip()
            .upper()
        )

        if not symbol:
            raise ValueError(
                f"Invalid basket symbol in: {token}"
            )

        try:
            weight = float(
                weight_text
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid basket weight: {token}"
            ) from exc

        if weight <= 0:
            raise ValueError(
                "Basket weights must be positive."
            )

        if symbol in weights:
            raise ValueError(
                f"Duplicate basket symbol: {symbol}"
            )

        weights[symbol] = weight

    if not weights:
        raise ValueError(
            "Indicator basket cannot be empty."
        )

    total_weight = sum(
        weights.values()
    )

    if total_weight > 1.0 + 1e-9:
        raise ValueError(
            "Total basket weight cannot exceed 1.0."
        )

    return weights


def parse_indicator_basket_uid(
    uid: str,
) -> dict[str, Any]:
    raw_parameters = parse_uid_parts(
        uid
    )

    if "weights" not in raw_parameters:
        raise ValueError(
            "Indicator basket UID must contain "
            "'weights'."
        )

    if "state" not in raw_parameters:
        raise ValueError(
            "Indicator basket UID must contain "
            "'state'."
        )

    target_weights = parse_weight_string(
        raw_parameters["weights"]
    )

    renormalize = parse_bool(
        raw_parameters.get(
            "renorm",
            "false",
        ),
        "renorm",
    )

    (
        indicator_state,
        state_type,
        state_parameters,
    ) = create_state_from_parameters(
        raw_parameters
    )

    return {
        "target_weights": target_weights,
        "renormalize": renormalize,
        "state_type": state_type,
        "state_parameters": state_parameters,
        "indicator_state": indicator_state,
    }