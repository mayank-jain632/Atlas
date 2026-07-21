from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseFuturesStrategy
from indicators import bollinger_bands, ema
from data.resample import resample_ohlcv, raw_bars_needed


class BollingerTrendStrategy(BaseFuturesStrategy):
    """Bollinger Band breakout + 200 EMA trend filter + volume confirmation, long-only."""

    strategy_name = "bbtrend"

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
            raise ValueError("BollingerTrendStrategy requires a 'bbtrend' UID.")

    def strategy_required_history(self) -> int:
        target_bars_needed = (
            max(
                int(self.parameters["bb_period"]),
                int(self.parameters["trend_ema_period"]),
                int(self.parameters["volume_ma_period"]),
            )
            + 5
        )
        return raw_bars_needed(
            target_bars=target_bars_needed,
            source_timeframe="1h",
            target_timeframe=self.parameters["target_timeframe"],
        )

    def generate_desired_position(
        self,
        data: pd.DataFrame,
    ) -> tuple[int, dict[str, Any]]:
        resampled = resample_ohlcv(
            data,
            timeframe=self.parameters["target_timeframe"],
            source_timeframe="1h",
        )

        min_bars = (
            max(
                int(self.parameters["bb_period"]),
                int(self.parameters["trend_ema_period"]),
                int(self.parameters["volume_ma_period"]),
            )
            + 2
        )
        if len(resampled) < min_bars:
            return int(self.position_direction), {
                "resampled_bars": len(resampled),
                "decision": "INSUFFICIENT_RESAMPLED_HISTORY",
            }

        bb = bollinger_bands(
            resampled["close"],
            period=int(self.parameters["bb_period"]),
            standard_deviations=float(self.parameters["bb_std"]),
        )
        trend_ema_series = ema(resampled["close"], period=int(self.parameters["trend_ema_period"]))
        volume_ma_series = resampled["volume"].rolling(int(self.parameters["volume_ma_period"])).mean()

        close = float(resampled["close"].iloc[-1])
        volume = float(resampled["volume"].iloc[-1])
        mid = bb["middle"].iloc[-1]
        upper = bb["upper"].iloc[-1]
        trend_value = trend_ema_series.iloc[-1]
        volume_avg = volume_ma_series.iloc[-1]

        diagnostics = {
            "close": close,
            "bb_mid": None if pd.isna(mid) else float(mid),
            "bb_upper": None if pd.isna(upper) else float(upper),
            "trend_ema": None if pd.isna(trend_value) else float(trend_value),
            "volume_ma": None if pd.isna(volume_avg) else float(volume_avg),
        }

        required_values = [mid, upper, trend_value, volume_avg]
        if any(pd.isna(value) for value in required_values):
            diagnostics["decision"] = "INDICATORS_NOT_READY"
            return int(self.position_direction), diagnostics

        price_above_trend = close > float(trend_value)
        volume_confirmed = volume > float(volume_avg)
        breakout_long = close > float(upper)
        exit_signal = close < float(mid)

        diagnostics.update(
            {
                "price_above_trend": price_above_trend,
                "volume_confirmed": volume_confirmed,
                "breakout_long": breakout_long,
                "exit_signal": exit_signal,
            }
        )

        entry_signal = breakout_long and price_above_trend and volume_confirmed

        if self.position_direction == 0:
            if entry_signal:
                diagnostics["decision"] = "ENTER_LONG"
                return 1, diagnostics
            diagnostics["decision"] = "REMAIN_FLAT"
            return 0, diagnostics

        if self.position_direction > 0:
            if exit_signal:
                diagnostics["decision"] = "EXIT_LONG"
                return 0, diagnostics
            diagnostics["decision"] = "HOLD_LONG"
            return 1, diagnostics

        # Strategy is long-only; short positions should never occur.
        diagnostics["decision"] = "UNEXPECTED_SHORT_POSITION"
        return 0, diagnostics