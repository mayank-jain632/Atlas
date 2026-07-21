from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseFuturesStrategy
from indicators import ema, macd
from data.resample import resample_ohlcv, raw_bars_needed


class EMAMACDStrategy(BaseFuturesStrategy):
    """200 EMA trend filter + MACD cross momentum confirmation, long-only."""

    strategy_name = "emamacd"

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
            raise ValueError("EMAMACDStrategy requires an 'emamacd' UID.")

    def strategy_required_history(self) -> int:
        target_bars_needed = int(self.parameters["trend_ema_period"]) + 40
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

        min_bars = int(self.parameters["trend_ema_period"]) + 35
        if len(resampled) < min_bars:
            return int(self.position_direction), {
                "resampled_bars": len(resampled),
                "decision": "INSUFFICIENT_RESAMPLED_HISTORY",
            }

        trend_ema_series = ema(resampled["close"], period=int(self.parameters["trend_ema_period"]))
        macd_frame = macd(resampled["close"])

        close = float(resampled["close"].iloc[-1])
        trend_value = trend_ema_series.iloc[-1]

        macd_line = macd_frame["macd"]
        signal_line = macd_frame["signal"]

        # Check whether MACD crossed above signal within the last 3 bars.
        lookback = 3
        recent_macd = macd_line.iloc[-(lookback + 1):]
        recent_signal = signal_line.iloc[-(lookback + 1):]
        crossed_up_recently = False
        for i in range(1, len(recent_macd)):
            prev_below = recent_macd.iloc[i - 1] <= recent_signal.iloc[i - 1]
            now_above = recent_macd.iloc[i] > recent_signal.iloc[i]
            if prev_below and now_above:
                crossed_up_recently = True
                break

        macd_now = macd_line.iloc[-1]
        signal_now = signal_line.iloc[-1]
        macd_above_zero = macd_now > 0
        currently_bullish = macd_now > signal_now

        diagnostics = {
            "close": close,
            "trend_ema": None if pd.isna(trend_value) else float(trend_value),
            "macd": None if pd.isna(macd_now) else float(macd_now),
            "signal": None if pd.isna(signal_now) else float(signal_now),
        }

        required_values = [trend_value, macd_now, signal_now]
        if any(pd.isna(value) for value in required_values):
            diagnostics["decision"] = "INDICATORS_NOT_READY"
            return int(self.position_direction), diagnostics

        price_above_trend = close > float(trend_value)
        price_below_trend = close < float(trend_value)

        diagnostics.update(
            {
                "price_above_trend": price_above_trend,
                "crossed_up_recently": crossed_up_recently,
                "macd_above_zero": macd_above_zero,
                "currently_bullish": currently_bullish,
            }
        )

        entry_signal = price_above_trend and crossed_up_recently and macd_above_zero
        macd_exit_signal = not currently_bullish
        trend_exit_signal = price_below_trend

        if self.position_direction == 0:
            if entry_signal:
                diagnostics["decision"] = "ENTER_LONG"
                return 1, diagnostics
            diagnostics["decision"] = "REMAIN_FLAT"
            return 0, diagnostics

        if self.position_direction > 0:
            if trend_exit_signal:
                diagnostics["decision"] = "EXIT_LONG_TREND"
                return 0, diagnostics
            if macd_exit_signal:
                diagnostics["decision"] = "EXIT_LONG_MACD"
                return 0, diagnostics
            diagnostics["decision"] = "HOLD_LONG"
            return 1, diagnostics

        # Strategy is long-only; short positions should never occur.
        diagnostics["decision"] = "UNEXPECTED_SHORT_POSITION"
        return 0, diagnostics