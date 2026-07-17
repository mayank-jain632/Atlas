from __future__ import annotations

from pathlib import Path

from ems.ems import EMS


class BaseStrategy(EMS):
    """All Atlas strategies inherit EMS directly.

    A concrete strategy only needs to:
      1. parse/store its UID parameters,
      2. implement ``on_day_close``.
    """

    strategy_name = "base"

    def __init__(
        self,
        uid: str,
        capital: float,
        db_path: str | Path | None = None,        timeframe: str = "1d",
        allow_fractional_shares: bool = True,
    ) -> None:
        super().__init__(
            uid=uid,
            capital=capital,
            db_path=db_path,
            timeframe=timeframe,
            allow_fractional_shares=allow_fractional_shares,
        )
