from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from indicator_states import (
    BEARISH,
    BULLISH,
    UNKNOWN,
    VALID_STATES,
    IndicatorStateEngine,
)

from indicator_states.uid import (
    create_state_from_parameters,
)

from strategies.base import BaseStrategy


class BuyHoldIndicatorStrategy(BaseStrategy):
    """
    Single-symbol strategy controlled by an IndicatorState.

    UID examples:

        buy_hold_indicator
        __s=QQQ
        __w=1
        __state=ma_crossover
        __fast=50
        __slow=200
        __method=sma

    Full UID:

        buy_hold_indicator__s=QQQ__w=1__state=ma_crossover__fast=50__slow=200__method=sma

    Behavior:

        BULLISH:
            Hold target allocation.

        BEARISH:
            Close the position and remain in cash.

        UNKNOWN:
            Take no action.
    """

    strategy_name = "buy_hold_indicator"

    def __init__(
        self,
        uid: str,
        capital: float,
        db_path: str | Path | None = None,
        timeframe: str = "1d",
        allow_fractional_shares: bool = True,
    ) -> None:
        # ----------------------------------------------------
        # UID parsing
        # ----------------------------------------------------

        raw_parameters = self.parse_uid(uid)

        (
            indicator_state,
            state_type,
            state_parameters,
        ) = create_state_from_parameters(
            raw_parameters
        )

        symbol = (
            raw_parameters["s"]
            .strip()
            .upper()
        )

        target_percent = float(
            raw_parameters.get(
                "w",
                "1.0",
            )
        )

        initial_state = (
            raw_parameters.get(
                "initial",
                UNKNOWN,
            )
            .strip()
            .upper()
        )

        if not symbol:
            raise ValueError(
                "UID parameter 's' cannot be empty."
            )

        if not 0.0 <= target_percent <= 1.0:
            raise ValueError(
                "UID parameter 'w' must be between 0 and 1."
            )

        if initial_state not in VALID_STATES:
            raise ValueError(
                "UID parameter 'initial' must be one of: "
                f"{sorted(VALID_STATES)}"
            )

        self.parameters = {
            "symbol": symbol,
            "target_percent": target_percent,
            "initial_state": initial_state,
            "state_type": state_type,
            "state_parameters": state_parameters,
        }

        # ----------------------------------------------------
        # Base strategy / EMS
        # ---------------------------------------------------
        self.state_engine = (
            IndicatorStateEngine(
                evaluator=indicator_state,
                initial_state=initial_state,
            )
        )

        self.state_log: list[
            dict[str, Any]
        ] = []

        super().__init__(
            uid=uid,
            capital=capital,
            db_path=db_path,
            timeframe=timeframe,
            allow_fractional_shares=(
                allow_fractional_shares
            ),
        )

        self.symbol = symbol
        self.target_allocation = (
            target_percent
        )

    # ========================================================
    # UID
    # ========================================================

    @staticmethod
    def parse_uid(
        uid: str,
    ) -> dict[str, str]:
        """
        Parse the complete strategy UID.

        Values intentionally remain strings here. The strategy
        and indicator-state parser convert them to appropriate
        types.
        """
        uid = str(uid).strip()

        if not uid:
            raise ValueError(
                "UID cannot be empty."
            )

        parts = uid.split("__")

        prefix = parts[0].strip().lower()

        if prefix != "buy_hold_indicator":
            raise ValueError(
                "BuyHoldIndicatorStrategy requires "
                "a UID beginning with "
                "'buy_hold_indicator'. "
                f"Received: {prefix}"
            )

        parameters: dict[str, str] = {}

        for token in parts[1:]:
            token = token.strip()

            if not token:
                continue

            if "=" not in token:
                raise ValueError(
                    f"Invalid UID token: {token}"
                )

            key, value = token.split(
                "=",
                1,
            )

            key = key.strip().lower()
            value = value.strip()

            if not key:
                raise ValueError(
                    f"Invalid UID token: {token}"
                )

            parameters[key] = value

        required_parameters = {
            "s",
            "state",
        }

        missing = (
            required_parameters
            - set(parameters)
        )

        if missing:
            raise ValueError(
                "Missing required buy-hold indicator "
                f"UID parameters: {sorted(missing)}"
            )

        return parameters

    # ========================================================
    # Required data
    # ========================================================

    def required_symbols(
        self,
    ) -> list[str]:
        return [self.symbol]

    @property
    def symbols(
        self,
    ) -> list[str]:
        """
        Backward-compatible alias for existing runners.
        """
        return self.required_symbols()

    # ========================================================
    # Reset
    # ========================================================

    def reset(self) -> None:
        """
        Reset EMS and indicator-state history.
        """
        super().reset()

        self.state_engine.reset()
        self.state_log = []

    # ========================================================
    # Strategy event
    # ========================================================

    def on_day_close(self) -> None:
        result = self.state_engine.evaluate(
            data_provider=self,
            symbol=self.symbol,
        )

        self.state_log.append(
            {
                "timestamp": (
                    self.get_current_timestamp()
                ),
                "indicator_type": (
                    self.parameters[
                        "state_type"
                    ]
                ),
                **result.to_dict(),
            }
        )

        notes = {
            "indicator_type": (
                self.parameters[
                    "state_type"
                ]
            ),
            "indicator_parameters": (
                self.parameters[
                    "state_parameters"
                ]
            ),
            "target_percent": (
                self.target_allocation
            ),
            **result.to_dict(),
        }

        # ----------------------------------------------------
        # Bullish: establish or restore position
        # ----------------------------------------------------

        if result.state == BULLISH:
            current_position = (
                self.get_position(
                    self.symbol
                )
            )

            if current_position <= 0:
                self.target_percent(
                    symbol=self.symbol,
                    target_percent=(
                        self.target_allocation
                    ),
                    reason=(
                        "INDICATOR_STATE_BULLISH"
                    ),
                    notes=notes,
                )

        # ----------------------------------------------------
        # Bearish: close position
        # ----------------------------------------------------

        elif result.state == BEARISH:
            if self.get_position(
                self.symbol
            ) > 0:
                self.close_position(
                    symbol=self.symbol,
                    reason=(
                        "INDICATOR_STATE_BEARISH"
                    ),
                    notes=notes,
                )

        # ----------------------------------------------------
        # Unknown: no action
        # ----------------------------------------------------

        elif result.state == UNKNOWN:
            return

    # ========================================================
    # Output
    # ========================================================

    def get_state_log(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self.state_log
        )

    def save_results(
        self,
        output_dir: str | Path,
    ) -> None:
        super().save_results(
            output_dir
        )

        output = Path(output_dir)

        self.get_state_log().to_csv(
            output
            / "indicator_state_log.csv",
            index=False,
        )