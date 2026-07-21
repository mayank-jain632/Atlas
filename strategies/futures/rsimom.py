from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseFuturesStrategy
from indicators import ema, rsi
from data.resample import resample_ohlcv, raw_bars_needed


class RSIMomentumStrategy(BaseFuturesStrategy):
    """RSI momentum cross / 50-level retest + 200 EMA trend filter + EMA cross exit."""

    strategy_name = "rsimom"

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
            raise ValueError("RSIMomentumStrategy requires a 'rsimom' UID.")

    def strategy_required_history(self) -> int:
        target_bars_needed = (
            int(self.parameters["trend_ema_period"])
            + int(self.parameters["rsi_period"])
            + 20
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
            int(self.parameters["trend_ema_period"])
            + int(self.parameters["rsi_period"])
            + 5
        )

        if len(resampled) < min_bars:
            return int(self.position_direction), {
                "resampled_bars": len(resampled),
                "decision": "INSUFFICIENT_RESAMPLED_HISTORY",
            }

        rsi_series = rsi(resampled["close"], period=int(self.parameters["rsi_period"]))
        rsi_ema_series = ema(rsi_series, period=int(self.parameters["rsi_ema_period"]))
        trend_ema_series = ema(resampled["close"], period=int(self.parameters["trend_ema_period"]))
        exit_fast_series = ema(resampled["close"], period=int(self.parameters["exit_fast_period"]))
        exit_slow_series = ema(resampled["close"], period=int(self.parameters["exit_slow_period"]))

        close = float(resampled["close"].iloc[-1])
        volume = float(resampled["volume"].iloc[-1])

        rsi_now, rsi_prev = rsi_series.iloc[-1], rsi_series.iloc[-2]
        rsi_ema_now, rsi_ema_prev = rsi_ema_series.iloc[-1], rsi_ema_series.iloc[-2]
        rsi_recent_max = rsi_series.iloc[-10:].max()
        trend_value = trend_ema_series.iloc[-1]
        exit_fast_now, exit_fast_prev = exit_fast_series.iloc[-1], exit_fast_series.iloc[-2]
        exit_slow_now, exit_slow_prev = exit_slow_series.iloc[-1], exit_slow_series.iloc[-2]

        diagnostics = {
            "close": close,
            "rsi": None if pd.isna(rsi_now) else float(rsi_now),
            "rsi_ema": None if pd.isna(rsi_ema_now) else float(rsi_ema_now),
            "trend_ema": None if pd.isna(trend_value) else float(trend_value),
            "exit_fast_ema": None if pd.isna(exit_fast_now) else float(exit_fast_now),
            "exit_slow_ema": None if pd.isna(exit_slow_now) else float(exit_slow_now),
        }

        required_values = [
            rsi_now, rsi_prev, rsi_ema_now, rsi_ema_prev,
            trend_value, exit_fast_now, exit_slow_now,
            exit_fast_prev, exit_slow_prev,
        ]
        if any(pd.isna(value) for value in required_values):
            diagnostics["decision"] = "INDICATORS_NOT_READY"
            return int(self.position_direction), diagnostics

        price_above_trend = close > float(trend_value)
        volume_confirmed = volume > 0

        momentum_cross = (
            float(rsi_ema_now) > 50
            and float(rsi_prev) <= float(rsi_ema_prev)
            and float(rsi_now) > float(rsi_ema_now)
        )
        retest_signal = (
            float(rsi_recent_max) > 56
            and 44 <= float(rsi_now) <= 55
            and float(rsi_ema_now) > 50
        )
        exit_signal = (
            float(exit_fast_prev) >= float(exit_slow_prev)
            and float(exit_fast_now) < float(exit_slow_now)
        )

        diagnostics.update(
            {
                "price_above_trend": price_above_trend,
                "volume_confirmed": volume_confirmed,
                "momentum_cross": momentum_cross,
                "retest_signal": retest_signal,
                "exit_signal": exit_signal,
            }
        )

        entry_signal = (
            (momentum_cross or retest_signal)
            and price_above_trend
            and volume_confirmed
        )

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