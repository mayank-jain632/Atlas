from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseFuturesStrategy
from indicators import adx, donchian_channels, choppiness_index, ema


class DCEMACHOPStrategy(BaseFuturesStrategy):
    """
    Donchian breakout + EMA + ADX + Choppiness + ATR stop.

    Entries require a fresh breakout and all filters.
    Existing positions are held until an opposite breakout or ATR stop.
    """

    strategy_name = "dcemachop"

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
            raise ValueError(
                "DCEMACHOPStrategy requires a 'dcemachop' UID."
            )

    def strategy_required_history(self) -> int:
        return max(
            int(self.parameters["donchian_period"]) + 4,
            int(self.parameters["ema_period"]) + 3,
            2 * int(self.parameters["adx_period"]) + 3,
            int(self.parameters["chop_period"]) + 3,
        )

    def generate_desired_position(
        self,
        data: pd.DataFrame,
    ) -> tuple[int, dict[str, Any]]:
        dc = donchian_channels(
            data=data,
            period=int(self.parameters["donchian_period"]), shift = 1)
        adx_frame = adx(
            data=data,
            period=int(self.parameters["adx_period"]),
        )
        ema_series = ema(
            data["close"],
            period=int(self.parameters["ema_period"]),
        )
        chop_series = choppiness_index(
            data=data,
            period=int(self.parameters["chop_period"]),
        )

        close = float(data["close"].iloc[-1])
        previous_close = float(data["close"].iloc[-2])

        upper = dc["upper"].iloc[-1]
        lower = dc["lower"].iloc[-1]
        previous_upper = dc["upper"].iloc[-2]
        previous_lower = dc["lower"].iloc[-2]

        adx_value = adx_frame["adx"].iloc[-1]
        plus_di = adx_frame["positive_di"].iloc[-1]
        minus_di = adx_frame["negative_di"].iloc[-1]
        ema_value = ema_series.iloc[-1]
        chop_value = chop_series.iloc[-1]

        diagnostics = {
            "close": close,
            "previous_close": previous_close,
            "donchian_upper": None if pd.isna(upper) else float(upper),
            "donchian_lower": None if pd.isna(lower) else float(lower),
            "previous_donchian_upper": (
                None if pd.isna(previous_upper)
                else float(previous_upper)
            ),
            "previous_donchian_lower": (
                None if pd.isna(previous_lower)
                else float(previous_lower)
            ),
            "ema": None if pd.isna(ema_value) else float(ema_value),
            "adx": None if pd.isna(adx_value) else float(adx_value),
            "plus_di": None if pd.isna(plus_di) else float(plus_di),
            "minus_di": None if pd.isna(minus_di) else float(minus_di),
            "chop": None if pd.isna(chop_value) else float(chop_value),
        }

        required = (
            upper,
            lower,
            previous_upper,
            previous_lower,
            adx_value,
            plus_di,
            minus_di,
            ema_value,
            chop_value,
        )
        if any(pd.isna(value) for value in required):
            diagnostics["decision"] = "INDICATORS_NOT_READY"
            return int(self.position_direction), diagnostics

        upside_breakout = (
            close > float(upper)
            and previous_close <= float(previous_upper)
        )
        downside_breakout = (
            close < float(lower)
            and previous_close >= float(previous_lower)
        )

        adx_confirmed = (
            float(adx_value)
            >= float(self.parameters["adx_threshold"])
        )
        chop_confirmed = (
            float(chop_value)
            <= float(self.parameters["chop_threshold"])
        )
        long_confirmed = (
            close > float(ema_value)
            and float(plus_di) > float(minus_di)
            and adx_confirmed
            and chop_confirmed
        )
        short_confirmed = (
            close < float(ema_value)
            and float(minus_di) > float(plus_di)
            and adx_confirmed
            and chop_confirmed
        )

        diagnostics.update(
            {
                "upside_breakout": upside_breakout,
                "downside_breakout": downside_breakout,
                "adx_confirmed": adx_confirmed,
                "chop_confirmed": chop_confirmed,
                "long_filters_confirmed": long_confirmed,
                "short_filters_confirmed": short_confirmed,
            }
        )

        if self.position_direction == 0:
            if upside_breakout and long_confirmed:
                diagnostics["decision"] = "ENTER_LONG"
                return 1, diagnostics
            if downside_breakout and short_confirmed:
                diagnostics["decision"] = "ENTER_SHORT"
                return -1, diagnostics
            diagnostics["decision"] = "REMAIN_FLAT"
            return 0, diagnostics

        if self.position_direction > 0:
            if downside_breakout:
                if short_confirmed:
                    diagnostics["decision"] = "REVERSE_LONG_TO_SHORT"
                    return -1, diagnostics
                diagnostics["decision"] = "EXIT_LONG"
                return 0, diagnostics
            diagnostics["decision"] = "HOLD_LONG"
            return 1, diagnostics

        if upside_breakout:
            if long_confirmed:
                diagnostics["decision"] = "REVERSE_SHORT_TO_LONG"
                return 1, diagnostics
            diagnostics["decision"] = "EXIT_SHORT"
            return 0, diagnostics

        diagnostics["decision"] = "HOLD_SHORT"
        return -1, diagnostics
