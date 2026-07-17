from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from data.interface import DataInterface


TRADEBOOK_COLUMNS = [
    "timestamp",
    "uid",
    "strategy",
    "symbol",
    "action",
    "quantity",
    "price",
    "value",
    "position_before",
    "position_after",
    "cash_before",
    "cash_after",
    "reason",
    "notes",
]


class EMS(DataInterface):
    """Simple daily event, portfolio, and tradebook layer."""

    strategy_name = "base"

    def __init__(
        self,
        uid: str,
        capital: float,
        db_path: str | Path | None = None,
        timeframe: str = "1d",
        allow_fractional_shares: bool = True,
        allow_short_positions: bool = False,
        allow_leverage: bool = False,
    ) -> None:
        super().__init__(
            duckdb_path=db_path,
            timeframe=timeframe,
            read_only=True,
        )
        if float(capital) <= 0:
            raise ValueError("capital must be positive.")

        self.uid = str(uid)
        self.initial_capital = float(capital)
        self.allow_fractional_shares = bool(allow_fractional_shares)
        self.allow_short_positions = bool(allow_short_positions)
        self.allow_leverage = bool(allow_leverage)
        self.reset()

    # ========================================================
    # Event hook / runner
    # ========================================================

    def on_day_close(self) -> None:
        """Override in each strategy."""

    def reset(self) -> None:
        self.cash = float(self.initial_capital)
        self.positions: dict[str, float] = {}
        self.tradebook: list[dict[str, Any]] = []
        self.equity_history: list[dict[str, Any]] = []
        self.clear_current_timestamp()

    def run(
        self,
        symbols: Iterable[str],
        start: Optional[str | pd.Timestamp] = None,
        end: Optional[str | pd.Timestamp] = None,
        reset: bool = True,
    ) -> dict[str, pd.DataFrame]:
        if reset:
            self.reset()

        unique_symbols = list(dict.fromkeys(symbols))
        dates = self.timestamps(
            symbols=unique_symbols,
            start=start,
            end=end,
        )

        for timestamp in dates:
            self.set_current_timestamp(timestamp)
            self.on_day_close()
            self._record_equity()

        return {
            "tradebook": self.get_tradebook(),
            "equity": self.get_equity_history(),
            "positions": self.get_positions_frame(),
        }

    # ========================================================
    # Portfolio helpers
    # ========================================================

    def get_position(self, symbol: str) -> float:
        return float(self.positions.get(symbol, 0.0))

    def get_positions(self) -> dict[str, float]:
        return {
            symbol: float(quantity)
            for symbol, quantity in self.positions.items()
            if abs(quantity) > 1e-12
        }

    def portfolio_value(self) -> float:
        value = float(self.cash)
        for symbol, quantity in self.get_positions().items():
            value += quantity * self.get_close(symbol)
        return float(value)

    def _quantity_from_dollars(self, dollars: float, price: float) -> float:
        if price <= 0 or dollars <= 0:
            return 0.0
        quantity = float(dollars) / float(price)
        if self.allow_fractional_shares:
            return quantity
        return float(math.floor(quantity))

    # ========================================================
    # Trading
    # ========================================================

    def place_trade(
        self,
        symbol: str,
        quantity: float,
        reason: str = "SIGNAL",
        notes: Optional[dict[str, Any] | str] = None,
        price_field: str = "close",
    ) -> Optional[dict[str, Any]]:
        """Execute a signed quantity at the current bar price."""
        self.get_current_timestamp()

        quantity = float(quantity)
        if abs(quantity) <= 1e-12:
            return None

        price = self.get_price(symbol, price_field)
        before_position = self.get_position(symbol)

        if quantity < 0 and not self.allow_short_positions:
            quantity = max(quantity, -before_position)
            if abs(quantity) <= 1e-12:
                return None

        if quantity > 0 and not self.allow_leverage:
            max_quantity = self._quantity_from_dollars(self.cash, price)
            quantity = min(quantity, max_quantity)
            if quantity <= 1e-12:
                return None

        after_position = before_position + quantity
        cash_before = float(self.cash)
        trade_value = abs(quantity) * price

        self.cash -= quantity * price
        self.positions[symbol] = after_position
        if abs(after_position) <= 1e-12:
            self.positions.pop(symbol, None)

        row = {
            "timestamp": self.get_current_timestamp(),
            "uid": self.uid,
            "strategy": self.strategy_name,
            "symbol": symbol,
            "action": "BUY" if quantity > 0 else "SELL",
            "quantity": abs(quantity),
            "price": price,
            "value": trade_value,
            "position_before": before_position,
            "position_after": after_position,
            "cash_before": cash_before,
            "cash_after": float(self.cash),
            "reason": reason,
            "notes": (
                notes
                if isinstance(notes, str)
                else json.dumps(notes or {}, sort_keys=True)
            ),
        }
        self.tradebook.append(row)
        return row

    def target_quantity(
        self,
        symbol: str,
        target_quantity: float,
        reason: str = "TARGET_QUANTITY",
        notes: Optional[dict[str, Any] | str] = None,
    ) -> Optional[dict[str, Any]]:
        if not self.allow_short_positions and float(target_quantity) < 0:
            raise ValueError(
                "Negative target quantities require allow_short_positions=True."
            )
        delta = float(target_quantity) - self.get_position(symbol)
        return self.place_trade(
            symbol,
            delta,
            reason=reason,
            notes=notes,
        )

    def target_percent(
        self,
        symbol: str,
        target_percent: float,
        reason: str = "TARGET_PERCENT",
        notes: Optional[dict[str, Any] | str] = None,
    ) -> Optional[dict[str, Any]]:
        target_percent = float(target_percent)
        if target_percent < 0 and not self.allow_short_positions:
            raise ValueError(
                "Negative target percentages require allow_short_positions=True."
            )
        if target_percent > 1.0 and not self.allow_leverage:
            raise ValueError(
                "Target percent above 1.0 requires allow_leverage=True."
            )

        equity = self.portfolio_value()
        price = self.get_close(symbol)
        target_dollars = equity * target_percent
        target_quantity = self._quantity_from_dollars(
            abs(target_dollars),
            price,
        )
        if target_dollars < 0:
            target_quantity = -target_quantity

        return self.target_quantity(
            symbol,
            target_quantity,
            reason=reason,
            notes=notes,
        )

    def close_position(
        self,
        symbol: str,
        reason: str = "CLOSE_POSITION",
        notes: Optional[dict[str, Any] | str] = None,
    ) -> Optional[dict[str, Any]]:
        return self.target_quantity(
            symbol,
            0.0,
            reason=reason,
            notes=notes,
        )

    def rebalance_to_weights(
        self,
        weights: dict[str, float],
        reason: str = "REBALANCE",
        notes_by_symbol: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        notes_by_symbol = notes_by_symbol or {}
        clean_weights = {
            symbol: float(weight)
            for symbol, weight in weights.items()
            if abs(float(weight)) > 1e-12
        }

        if not self.allow_short_positions and any(
            weight < 0 for weight in clean_weights.values()
        ):
            raise ValueError(
                "Negative weights require allow_short_positions=True."
            )

        gross_weight = sum(abs(weight) for weight in clean_weights.values())
        if gross_weight > 1.0 + 1e-9 and not self.allow_leverage:
            raise ValueError(
                f"Gross target weight exceeds 1.0: {gross_weight:.6f}"
            )

        equity = self.portfolio_value()
        all_symbols = set(self.get_positions()) | set(clean_weights)
        target_quantities: dict[str, float] = {}

        for symbol in all_symbols:
            target_weight = clean_weights.get(symbol, 0.0)
            price = self.get_close(symbol)
            target_dollars = equity * target_weight
            quantity = self._quantity_from_dollars(
                abs(target_dollars),
                price,
            )
            target_quantities[symbol] = (
                -quantity if target_dollars < 0 else quantity
            )

        # First reduce or close positions so cash is available.
        for symbol in sorted(all_symbols):
            current = self.get_position(symbol)
            target = target_quantities[symbol]
            if target < current:
                self.target_quantity(
                    symbol,
                    target,
                    reason=f"{reason}_SELL",
                    notes=notes_by_symbol.get(symbol, {}),
                )

        # Then increase or open positions.
        for symbol in sorted(all_symbols):
            current = self.get_position(symbol)
            target = target_quantities[symbol]
            if target > current:
                self.target_quantity(
                    symbol,
                    target,
                    reason=f"{reason}_BUY",
                    notes=notes_by_symbol.get(symbol, {}),
                )

    # ========================================================
    # Output
    # ========================================================

    def _record_equity(self) -> None:
        equity = self.portfolio_value()
        self.equity_history.append(
            {
                "timestamp": self.get_current_timestamp(),
                "uid": self.uid,
                "cash": float(self.cash),
                "market_value": equity - self.cash,
                "equity": equity,
            }
        )

    def get_tradebook(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.tradebook,
            columns=TRADEBOOK_COLUMNS,
        )

    def get_equity_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_history)

    def get_positions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "uid": self.uid,
                    "symbol": symbol,
                    "quantity": quantity,
                }
                for symbol, quantity in self.get_positions().items()
            ],
            columns=["uid", "symbol", "quantity"],
        )

    def save_results(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.get_tradebook().to_csv(output / "tradebook.csv", index=False)
        self.get_equity_history().to_csv(output / "equity.csv", index=False)
        self.get_positions_frame().to_csv(output / "positions.csv", index=False)
