from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .result import (
    BULLISH,
    BEARISH,
    UNKNOWN,
    VALID_STATES,
    IndicatorStateResult,
)


class IndicatorState(ABC):
    """
    Base class for all market-state evaluators.

    IndicatorState classes:
        - read historical market data,
        - calculate an indicator,
        - classify the result as bullish or bearish.

    They do not:
        - place orders,
        - inspect portfolio positions,
        - allocate capital,
        - write tradebooks.
    """

    name = "base_state"

    @abstractmethod
    def required_history(self) -> int:
        """
        Minimum number of historical bars needed.
        """

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        data: pd.DataFrame,
        previous_state: str = UNKNOWN,
    ) -> IndicatorStateResult:
        """
        Evaluate the state using the latest available bar.
        """

    @staticmethod
    def validate_previous_state(
        previous_state: str,
    ) -> str:
        previous_state = str(
            previous_state
        ).upper()

        if previous_state not in VALID_STATES:
            raise ValueError(
                f"Invalid previous state: {previous_state}"
            )

        return previous_state

    @staticmethod
    def build_result(
        symbol: str,
        state: str,
        previous_state: str,
        reason: str,
        values: dict[str, Any] | None = None,
    ) -> IndicatorStateResult:
        if state not in VALID_STATES:
            raise ValueError(
                f"Invalid state: {state}"
            )

        if previous_state not in VALID_STATES:
            raise ValueError(
                "Invalid previous state: "
                f"{previous_state}"
            )

        return IndicatorStateResult(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            changed=(
                state != previous_state
                and state != UNKNOWN
            ),
            reason=reason,
            values=values or {},
        )

    def insufficient_history_result(
        self,
        symbol: str,
        previous_state: str,
        available_history: int,
    ) -> IndicatorStateResult:
        """
        Preserve an existing known state when history is
        temporarily insufficient. Otherwise return UNKNOWN.
        """
        previous_state = self.validate_previous_state(
            previous_state
        )

        state = (
            previous_state
            if previous_state in {
                BULLISH,
                BEARISH,
            }
            else UNKNOWN
        )

        return self.build_result(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            reason="INSUFFICIENT_HISTORY",
            values={
                "available_history": int(
                    available_history
                ),
                "required_history": int(
                    self.required_history()
                ),
            },
        )


def normalize_ohlc(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize OHLC columns for state evaluators.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    frame = data.copy()

    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]

    required = {
        "high",
        "low",
        "close",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "Missing OHLC columns: "
            + ", ".join(sorted(missing))
        )

    for column in {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    return frame.sort_index()


def preserve_state_or_unknown(
    previous_state: str,
) -> str:
    if previous_state in {
        BULLISH,
        BEARISH,
    }:
        return previous_state

    return UNKNOWN