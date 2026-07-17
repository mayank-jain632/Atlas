from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .base import IndicatorState
from .result import (
    BULLISH,
    BEARISH,
    UNKNOWN,
    VALID_STATES,
    IndicatorStateResult,
)


class IndicatorStateEngine:
    """
    Evaluates and stores IndicatorState results per symbol.

    The engine does not trade.

    The supplied data provider must implement:

        history(
            symbol,
            field="close",
            bars=100,
        )

        get_current_timestamp()
    """

    def __init__(
        self,
        evaluator: IndicatorState,
        initial_state: str = UNKNOWN,
    ) -> None:
        initial_state = str(
            initial_state
        ).upper()

        if initial_state not in VALID_STATES:
            raise ValueError(
                f"Invalid initial state: {initial_state}"
            )

        self.evaluator = evaluator
        self.initial_state = initial_state

        self.states: dict[str, str] = {}
        self.latest_results: dict[
            str,
            IndicatorStateResult,
        ] = {}

        self.history_log: list[
            dict[str, Any]
        ] = []

    def reset(self) -> None:
        self.states = {}
        self.latest_results = {}
        self.history_log = []

    def get_state(
        self,
        symbol: str,
    ) -> str:
        return self.states.get(
            symbol,
            self.initial_state,
        )

    def is_bullish(
        self,
        symbol: str,
    ) -> bool:
        return (
            self.get_state(symbol)
            == BULLISH
        )

    def is_bearish(
        self,
        symbol: str,
    ) -> bool:
        return (
            self.get_state(symbol)
            == BEARISH
        )

    def is_unknown(
        self,
        symbol: str,
    ) -> bool:
        return (
            self.get_state(symbol)
            == UNKNOWN
        )

    def set_state(
        self,
        symbol: str,
        state: str,
    ) -> None:
        state = str(state).upper()

        if state not in VALID_STATES:
            raise ValueError(
                f"Invalid state: {state}"
            )

        self.states[symbol] = state

    def _get_ohlc_history(
        self,
        data_provider,
        symbol: str,
    ) -> pd.DataFrame:
        bars = self.evaluator.required_history()

        columns = {}

        for field in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            series = data_provider.history(
                symbol=symbol,
                field=field,
                bars=bars,
            )

            if not series.empty:
                columns[field] = series

        if "close" not in columns:
            return pd.DataFrame()

        frame = pd.concat(
            columns,
            axis=1,
        ).sort_index()

        return frame.dropna(
            subset=["close"]
        )

    def evaluate(
        self,
        data_provider,
        symbol: str,
    ) -> IndicatorStateResult:
        previous_state = self.get_state(
            symbol
        )

        data = self._get_ohlc_history(
            data_provider=data_provider,
            symbol=symbol,
        )
        if data is None or data.empty:
            return IndicatorStateResult(
                symbol=symbol,
                state=previous_state,
                previous_state=previous_state,
                changed=False,
                reason="NO_HISTORY",
                values={},
            )
        result = self.evaluator.evaluate(
            symbol=symbol,
            data=data,
            previous_state=previous_state,
        )

        if result.state != UNKNOWN:
            self.states[symbol] = result.state

        self.latest_results[symbol] = result

        timestamp = None

        if hasattr(
            data_provider,
            "get_current_timestamp",
        ):
            try:
                timestamp = (
                    data_provider
                    .get_current_timestamp()
                )
            except RuntimeError:
                timestamp = None

        log_row = result.to_dict()

        log_row["timestamp"] = timestamp
        log_row["evaluator"] = (
            self.evaluator.name
        )

        self.history_log.append(log_row)

        return result

    def evaluate_many(
        self,
        data_provider,
        symbols: Iterable[str],
    ) -> dict[
        str,
        IndicatorStateResult,
    ]:
        unique_symbols = list(
            dict.fromkeys(symbols)
        )

        return {
            symbol: self.evaluate(
                data_provider=data_provider,
                symbol=symbol,
            )
            for symbol in unique_symbols
        }

    def get_latest_result(
        self,
        symbol: str,
    ) -> IndicatorStateResult | None:
        return self.latest_results.get(
            symbol
        )

    def get_state_frame(self) -> pd.DataFrame:
        rows = []

        for symbol, state in sorted(
            self.states.items()
        ):
            result = self.latest_results.get(
                symbol
            )

            rows.append(
                {
                    "symbol": symbol,
                    "state": state,
                    "reason": (
                        result.reason
                        if result is not None
                        else None
                    ),
                    "changed": (
                        result.changed
                        if result is not None
                        else None
                    ),
                    "values": (
                        result.values
                        if result is not None
                        else {}
                    ),
                }
            )

        return pd.DataFrame(rows)

    def get_history_log(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self.history_log
        )