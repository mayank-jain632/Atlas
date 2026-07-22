from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseFuturesStrategy
from indicators import ema, parabolic_sar


class PSAREMAStrategy(BaseFuturesStrategy):
    """PSAR flip + EMA confirmation + ATR stop."""

    strategy_name = "psarema"

    def __init__(
        self,
        uid: str,
        capital: float,
        db_path: str | Path | None = None,
        timeframe: str = "1d",
        source_timeframe: str | None = None,
        allow_fractional_shares: bool = False,
    ) -> None:
        super().__init__(
            uid=uid,
            capital=capital,
            db_path=db_path,
            timeframe=timeframe,
            source_timeframe=source_timeframe,
            allow_fractional_shares=False,
        )
        if self.parameters["strategy_type"] != self.strategy_name:
            raise ValueError("PSAREMAStrategy requires a 'psarema' UID.")

    def strategy_required_history(self) -> int:
        return max(int(self.parameters["ema_period"]) + 3, 10)

    def generate_desired_position(
        self,
        data: pd.DataFrame,
    ) -> tuple[int, dict[str, Any]]:
        psar_frame = parabolic_sar(
            data=data,
            step=float(self.parameters["psar_step"]),
            max_step=float(self.parameters["psar_max"]),
        )
        ema_series = ema(
            data["close"],
            period=int(self.parameters["ema_period"]),
        )

        close = float(data["close"].iloc[-1])
        current_direction = psar_frame["direction"].iloc[-1]
        previous_direction = psar_frame["direction"].iloc[-2]
        psar_value = psar_frame["sar"].iloc[-1]
        ema_value = ema_series.iloc[-1]

        diagnostics = {
            "close": close,
            "psar": None if pd.isna(psar_value) else float(psar_value),
            "psar_direction": (
                None if pd.isna(current_direction)
                else int(current_direction)
            ),
            "previous_psar_direction": (
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
