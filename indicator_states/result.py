from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


BULLISH = "BULLISH"
BEARISH = "BEARISH"
UNKNOWN = "UNKNOWN"

VALID_STATES = {
    BULLISH,
    BEARISH,
    UNKNOWN,
}


@dataclass(frozen=True)
class IndicatorStateResult:
    """
    Result returned by an IndicatorState evaluator.

    The result describes the market state only. It does not
    prescribe or execute any trade.
    """

    symbol: str
    state: str
    previous_state: str
    changed: bool
    reason: str
    values: dict[str, Any]

    def __post_init__(self) -> None:
        if self.state not in VALID_STATES:
            raise ValueError(
                f"Invalid state: {self.state}"
            )

        if self.previous_state not in VALID_STATES:
            raise ValueError(
                "Invalid previous_state: "
                f"{self.previous_state}"
            )

    @property
    def is_bullish(self) -> bool:
        return self.state == BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.state == BEARISH

    @property
    def is_unknown(self) -> bool:
        return self.state == UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)