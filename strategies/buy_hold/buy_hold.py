from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from strategies.base import BaseStrategy


class BuyHoldStrategy(BaseStrategy):
    """
    Simple single-symbol buy-and-hold strategy.

    UID format:

        buy_hold__s=QQQ__w=1

    Parameters:
        s:
            Symbol to hold.

        w:
            Target portfolio allocation between 0 and 1.

    Behavior:
        1. Parse the symbol and allocation from the UID.
        2. Buy the configured allocation on the first
           available backtest date.
        3. Hold through the rest of the backtest.
        4. Do not automatically liquidate at the end.
    """

    strategy_name = "buy_hold"

    def __init__(
        self,
        uid: str,
        capital: float,
        db_path: str | Path | None = None,
        timeframe: str = "1d",
        allow_fractional_shares: bool = True,
    ) -> None:
        # ----------------------------------------------------
        # Parse and validate UID before initializing EMS
        # ----------------------------------------------------

        parameters = self.parse_uid(uid)

        self.parameters = parameters

        self.symbol = parameters["symbol"]

        self.target_allocation = parameters[
            "target_percent"
        ]

        # ----------------------------------------------------
        # Initialize common strategy / EMS layer
        # ----------------------------------------------------

        super().__init__(
            uid=uid,
            capital=capital,
            db_path=db_path,
            timeframe=timeframe,
            allow_fractional_shares=(
                allow_fractional_shares
            ),
        )

        self.has_entered = False

        self.entry_log: list[
            dict[str, Any]
        ] = []

    # ========================================================
    # UID
    # ========================================================

    @staticmethod
    def parse_uid(
        uid: str,
    ) -> dict[str, Any]:
        """
        Parse:

            buy_hold__s=QQQ__w=1

        Returns:

            {
                "symbol": "QQQ",
                "target_percent": 1.0,
            }
        """
        uid = str(uid).strip()

        if not uid:
            raise ValueError(
                "UID cannot be empty."
            )

        parts = uid.split("__")

        prefix = parts[0].strip().lower()

        if prefix != "buy_hold":
            raise ValueError(
                "BuyHoldStrategy requires a UID "
                "starting with 'buy_hold'. "
                f"Received: {prefix}"
            )

        raw_parameters: dict[str, str] = {}

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

            raw_parameters[key] = value

        if "s" not in raw_parameters:
            raise ValueError(
                "Buy-hold UID must contain the "
                "symbol parameter 's'."
            )

        symbol = (
            raw_parameters["s"]
            .strip()
            .upper()
        )

        if not symbol:
            raise ValueError(
                "UID symbol parameter 's' "
                "cannot be empty."
            )

        try:
            target_percent = float(
                raw_parameters.get(
                    "w",
                    "1.0",
                )
            )

        except ValueError as exc:
            raise ValueError(
                "UID parameter 'w' must be numeric."
            ) from exc

        if not 0.0 <= target_percent <= 1.0:
            raise ValueError(
                "UID parameter 'w' must be "
                "between 0 and 1."
            )

        allowed_parameters = {
            "s",
            "w",
        }

        unknown_parameters = (
            set(raw_parameters)
            - allowed_parameters
        )

        if unknown_parameters:
            raise ValueError(
                "Unsupported buy-hold UID "
                f"parameters: "
                f"{sorted(unknown_parameters)}"
            )

        return {
            "symbol": symbol,
            "target_percent": (
                target_percent
            ),
        }

    # ========================================================
    # Required data
    # ========================================================

    def required_symbols(
        self,
    ) -> list[str]:
        """
        Symbols required by the generic strategy runner.
        """
        return [self.symbol]

    @property
    def symbols(
        self,
    ) -> list[str]:
        """
        Backward-compatible alias for older runners.
        """
        return self.required_symbols()

    # ========================================================
    # Reset
    # ========================================================

    def reset(self) -> None:
        """
        Reset both common EMS state and buy-and-hold state.
        """
        super().reset()

        self.has_entered = False
        self.entry_log = []

    # ========================================================
    # Strategy event
    # ========================================================

    def on_day_close(self) -> None:
        """
        Enter once and hold thereafter.
        """
        if self.has_entered:
            return

        current_price = self.get_close(
            self.symbol
        )

        if (
            current_price is None
            or pd.isna(current_price)
            or float(current_price) <= 0
        ):
            return

        trade = self.target_percent(
            symbol=self.symbol,
            target_percent=(
                self.target_allocation
            ),
            reason="BUY_HOLD_INITIAL_ENTRY",
            notes={
                "symbol": self.symbol,
                "target_percent": (
                    self.target_allocation
                ),
                "entry_type": (
                    "initial_entry"
                ),
                "uid_parameters": (
                    self.parameters
                ),
            },
        )

        # Only mark entry complete when EMS actually creates
        # a trade.
        if trade is not None:
            self.has_entered = True

            self.entry_log.append(
                {
                    "timestamp": (
                        self.get_current_timestamp()
                    ),
                    "symbol": self.symbol,
                    "price": float(
                        current_price
                    ),
                    "target_percent": (
                        self.target_allocation
                    ),
                    "quantity": (
                        self.get_position(
                            self.symbol
                        )
                    ),
                    "cash_after": self.cash,
                    "portfolio_value": (
                        self.portfolio_value()
                    ),
                }
            )

    # ========================================================
    # Output
    # ========================================================

    def get_entry_log(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self.entry_log
        )

    def save_results(
        self,
        output_dir: str | Path,
    ) -> None:
        super().save_results(
            output_dir
        )

        output = Path(output_dir)

        self.get_entry_log().to_csv(
            output / "buy_hold_entry.csv",
            index=False,
        )