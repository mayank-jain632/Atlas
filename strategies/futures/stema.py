from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseFuturesStrategy
from indicators import ema, supertrend


class STEMAStrategy(BaseFuturesStrategy):
    """Supertrend flip + EMA confirmation + ATR stop."""

    strategy_name = "stema"

    def __init__(
        self,
        uid: str,
        capital: float,
        db_path: str | Path | None = None,
        timeframe: str = "1d",
        allow_fractional_shares: bool = False,
    ) -> None:
        super().__init__(
            uid=uid,
            capital=capital,
            db_path=db_path,
            timeframe=timeframe,
            allow_fractional_shares=False,
        )
        if self.parameters["strategy_type"] != self.strategy_name:
            raise ValueError("STEMAStrategy requires a 'stema' UID.")

    def strategy_required_history(self) -> int:
        return max(
            int(self.parameters["supertrend_period"]) + 3,
            int(self.parameters["ema_period"]) + 3,
        )

    def generate_desired_position(
        self,
        data: pd.DataFrame,
    ) -> tuple[int, dict[str, Any]]:
        st = supertrend(
            data=data,
            period=int(self.parameters["supertrend_period"]),
            multiplier=float(self.parameters["supertrend_multiplier"]),
        )
        ema_series = ema(
            data["close"],
            period=int(self.parameters["ema_period"]),
        )

        close = float(data["close"].iloc[-1])
        current_direction = st["direction"].iloc[-1]
        previous_direction = st["direction"].iloc[-2]
        st_value = st["supertrend"].iloc[-1]
        ema_value = ema_series.iloc[-1]

        diagnostics = {
            "close": close,
            "supertrend": None if pd.isna(st_value) else float(st_value),
            "supertrend_direction": (
                None if pd.isna(current_direction)
                else int(current_direction)
            ),
            "previous_supertrend_direction": (
                None if pd.isna(previous_direction)
                else int(previous_direction)
            ),
            "ema": None if pd.isna(ema_value) else float(ema_value),
        }

        if (
            pd.isna(current_direction)
            or pd.isna(previous_direction)
            or pd.isna(ema_value)
        ):
            diagnostics["decision"] = "INDICATORS_NOT_READY"
            return int(self.position_direction), diagnostics

        bullish_flip = (
            float(previous_direction) < 0
            and float(current_direction) > 0
        )
        bearish_flip = (
            float(previous_direction) > 0
            and float(current_direction) < 0
        )
        long_confirmed = close > float(ema_value)
        short_confirmed = close < float(ema_value)

        diagnostics.update(
            {
                "bullish_flip": bullish_flip,
                "bearish_flip": bearish_flip,
                "ema_long_confirmed": long_confirmed,
                "ema_short_confirmed": short_confirmed,
            }
        )

        if self.position_direction == 0:
            if bullish_flip and long_confirmed:
                diagnostics["decision"] = "ENTER_LONG"
                return 1, diagnostics
            if bearish_flip and short_confirmed:
                diagnostics["decision"] = "ENTER_SHORT"
                return -1, diagnostics
            diagnostics["decision"] = "REMAIN_FLAT"
            return 0, diagnostics

        if self.position_direction > 0:
            if bearish_flip:
                if short_confirmed:
                    diagnostics["decision"] = "REVERSE_LONG_TO_SHORT"
                    return -1, diagnostics
                diagnostics["decision"] = "EXIT_LONG"
                return 0, diagnostics
            diagnostics["decision"] = "HOLD_LONG"
            return 1, diagnostics

        if bullish_flip:
            if long_confirmed:
                diagnostics["decision"] = "REVERSE_SHORT_TO_LONG"
                return 1, diagnostics
            diagnostics["decision"] = "EXIT_SHORT"
            return 0, diagnostics

        diagnostics["decision"] = "HOLD_SHORT"
        return -1, diagnostics
