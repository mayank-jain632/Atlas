from __future__ import annotations

from pathlib import Path

from strategies.base import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    """Minimal example showing the intended Atlas strategy style."""

    strategy_name = "ma_example"

    def __init__(
        self,
        uid: str,
        capital: float,
        symbol: str,
        ma_window: int,
        db_path: str | Path | None = None,
    ) -> None:
        super().__init__(uid=uid, capital=capital, db_path=db_path)
        self.symbol = symbol
        self.ma_window = int(ma_window)

    @staticmethod
    def build_uid(symbol: str, ma_window: int) -> str:
        return f"ma_{symbol}_{int(ma_window)}"

    @staticmethod
    def parse_uid(uid: str) -> dict:
        parts = uid.split("_")
        if len(parts) != 3 or parts[0] != "ma":
            raise ValueError(f"Invalid MA UID: {uid}")
        return {"symbol": parts[1], "ma_window": int(parts[2])}

    @classmethod
    def from_uid(
        cls,
        uid: str,
        capital: float,
        db_path: str | Path = "market_data/market_data.duckdb",
    ) -> "MovingAverageStrategy":
        params = cls.parse_uid(uid)
        return cls(uid=uid, capital=capital, db_path=db_path, **params)

    def on_day_close(self) -> None:
        closes = self.history(self.symbol, "close", self.ma_window)
        if len(closes) < self.ma_window:
            return

        close = float(closes.iloc[-1])
        moving_average = float(closes.mean())

        if close > moving_average:
            self.target_percent(
                self.symbol,
                1.0,
                reason="MA_LONG",
                notes={"close": close, "ma": moving_average, "ma_window": self.ma_window},
            )
        else:
            self.close_position(
                self.symbol,
                reason="MA_EXIT",
                notes={"close": close, "ma": moving_average, "ma_window": self.ma_window},
            )
