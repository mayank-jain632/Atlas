from __future__ import annotations

from typing import Any


def split_uid(
    uid: str,
) -> tuple[str, dict[str, str]]:
    """
    Parse:

        strategy_name__key=value__key=value

    Values remain strings here. Each strategy converts its own
    values to the correct types.
    """
    uid = str(uid).strip()

    if not uid:
        raise ValueError("UID cannot be empty.")

    parts = uid.split("__")

    strategy_name = parts[0].strip().lower()

    parameters: dict[str, str] = {}

    for token in parts[1:]:
        token = token.strip()

        if not token:
            continue

        if "=" not in token:
            raise ValueError(
                f"Invalid UID token: {token}"
            )

        key, value = token.split("=", 1)

        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(
                f"Invalid empty UID key in: {token}"
            )

        parameters[key] = value

    return strategy_name, parameters


def require(
    parameters: dict[str, str],
    key: str,
) -> str:
    if key not in parameters:
        raise ValueError(
            f"Missing UID parameter: {key}"
        )

    return parameters[key]


def get_str(
    parameters: dict[str, str],
    key: str,
    default: str,
) -> str:
    return str(
        parameters.get(key, default)
    )


def get_int(
    parameters: dict[str, str],
    key: str,
    default: int,
) -> int:
    return int(
        parameters.get(key, default)
    )


def get_float(
    parameters: dict[str, str],
    key: str,
    default: float,
) -> float:
    return float(
        parameters.get(key, default)
    )


def get_bool(
    parameters: dict[str, str],
    key: str,
    default: bool,
) -> bool:
    if key not in parameters:
        return bool(default)

    value = parameters[key].strip().lower()

    if value in {"true", "1", "yes"}:
        return True

    if value in {"false", "0", "no"}:
        return False

    raise ValueError(
        f"UID parameter '{key}' must be true or false."
    )


def verify_prefix(
    uid: str,
    expected: str,
) -> dict[str, str]:
    strategy_name, parameters = split_uid(uid)

    if strategy_name != expected:
        raise ValueError(
            f"Expected UID prefix '{expected}', "
            f"received '{strategy_name}'."
        )

    return parameters