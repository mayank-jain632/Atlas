from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseFuturesStrategy
from indicators import donchian_channels
from data.resample import resample_ohlcv, raw_bars_needed


class DonchianAsymmetricBreakoutStrategy(BaseFuturesStrategy):
    """Long-only asymmetric Donchian channel breakout on weekly candles."""

    strategy_name = "dcbreakout"

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
            raise ValueError("DonchianAsymmetricBreakoutStrategy requires a 'dcbreakout' UID.")

    def strategy_required_history(self) -> int:
        target_bars_needed = (
            max(int(self.parameters["upper_period"]), int(self.parameters["lower_period"])) + 5
        )
        return raw_bars_needed(
            target_bars=target_bars_needed,
            source_timeframe="1d",
            target_timeframe=self.parameters["target_timeframe"],
        )

    def generate_desired_position(
        self,
        data: pd.DataFrame,
    ) -> tuple[int, dict[str, Any]]:
        resampled = resample_ohlcv(
            data,
            timeframe=self.parameters["target_timeframe"],
            source_timeframe="1d",
        )

        min_bars = max(int(self.parameters["upper_period"]), int(self.parameters["lower_period"])) + 2
        if len(resampled) < min_bars:
            return int(self.position_direction), {
                "resampled_bars": len(resampled),
                "decision": "INSUFFICIENT_RESAMPLED_HISTORY",
            }

        upper_channel = donchian_channels(resampled, period=int(self.parameters["upper_period"]))
        lower_channel = donchian_channels(resampled, period=int(self.parameters["lower_period"]))

        close = float(resampled["close"].iloc[-1])
        upper = upper_channel["upper"].iloc[-1]
        lower = lower_channel["lower"].iloc[-1]

        diagnostics = {
            "close": close,
            "donchian_upper": None if pd.isna(upper) else float(upper),
            "donchian_lower": None if pd.isna(lower) else float(lower),
        }

        if pd.isna(upper) or pd.isna(lower):
            diagnostics["decision"] = "INDICATORS_NOT_READY"
            return int(self.position_direction), diagnostics

        breakout_long = close > float(upper)
        breakdown_exit = close < float(lower)

        diagnostics.update(
            {
                "breakout_long": breakout_long,
                "breakdown_exit": breakdown_exit,
            }
        )

        if self.position_direction == 0:
            if breakout_long:
                diagnostics["decision"] = "ENTER_LONG"
                return 1, diagnostics
            diagnostics["decision"] = "REMAIN_FLAT"
            return 0, diagnostics

        if self.position_direction > 0:
            if breakdown_exit:
                diagnostics["decision"] = "EXIT_LONG"
                return 0, diagnostics
            diagnostics["decision"] = "HOLD_LONG"
            return 1, diagnostics

        diagnostics["decision"] = "UNEXPECTED_SHORT_POSITION"
        return 0, diagnostics