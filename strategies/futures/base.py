from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from strategies.base import BaseStrategy

from indicators import atr
from indicators.utils import normalize_columns
from .instruments import FuturesInstrument, get_futures_instrument
from .uid import parse_futures_uid


FUTURES_TRADEBOOK_COLUMNS = [
    "timestamp",
    "uid",
    "strategy",
    "contract_symbol",
    "data_symbol",
    "action",
    "direction_before",
    "direction_after",
    "contracts",
    "price",
    "multiplier",
    "notional",
    "entry_timestamp",
    "entry_price",
    "entry_atr",
    "stop_price",
    "realized_trade_pnl",
    "account_equity",
    "reason",
    "notes",
]


class BaseFuturesStrategy(BaseStrategy, ABC):
    """
    Base class for a single-symbol futures strategy.

    Current assumptions
    -------------------
    1. One strategy trades one futures contract.
    2. Position size is fixed at one contract.
    3. Signals are evaluated at the daily close.
    4. Signal entries and exits occur at that close.
    5. Open positions are marked to market each day.
    6. The ATR stop is fixed using ATR at entry.
    7. If the market gaps through the stop, the fill occurs at the open.
    8. Commissions, slippage, roll costs and margin enforcement are not
       included yet.
    """

    strategy_name = "futures_base"

    def __init__(
        self,
        uid: str,
        capital: float,
        db_path: str | Path | None = None,
        timeframe: str = "1d",
        allow_fractional_shares: bool = False,
    ) -> None:
        parameters = parse_futures_uid(uid)

        instrument = get_futures_instrument(
            parameters["symbol"]
        )

        if float(capital) <= 0:
            raise ValueError(
                "capital must be greater than zero."
            )

        # Define futures-specific attributes before calling BaseStrategy.
        # This avoids initialization issues if BaseStrategy or EMS calls reset.
        self.parameters: dict[str, Any] = parameters

        self.symbol: str = instrument.symbol
        self.instrument: FuturesInstrument = instrument
        self.data_symbol: str = instrument.data_symbol

        # Fixed at one contract for the initial framework.
        self.contracts: int = 1

        kwargs: dict[str, Any] = {
            "uid": uid,
            "capital": float(capital),
            "timeframe": timeframe,
            "allow_fractional_shares": False,
        }

        if db_path is not None:
            kwargs["db_path"] = db_path

        super().__init__(**kwargs)

        self._reset_futures_state()

    # ========================================================
    # Strategy interface
    # ========================================================

    def required_symbols(self) -> list[str]:
        """
        Return the symbols required from the Atlas database.

        Example:
            MES strategy -> ES=F data
            MNQ strategy -> NQ=F data
            MGC strategy -> GC=F data
        """
        return [self.data_symbol]

    @property
    def symbols(self) -> list[str]:
        """
        Compatibility alias for runners that expect strategy.symbols.
        """
        return self.required_symbols()

    @abstractmethod
    def strategy_required_history(self) -> int:
        """
        Return the minimum number of bars needed by the concrete strategy.
        """

    def required_history(self) -> int:
        """
        Ensure there is sufficient history for both:

        1. Strategy indicators
        2. ATR stop calculation
        """
        return max(
            int(self.strategy_required_history()),
            int(self.parameters["atr_period"]) + 2,
        )

    @abstractmethod
    def generate_desired_position(
        self,
        data: pd.DataFrame,
    ) -> tuple[int, dict[str, Any]]:
        """
        Generate the desired position from historical data.

        Returns
        -------
        desired_direction:
             1 = long one contract
             0 = flat
            -1 = short one contract

        diagnostics:
            Dictionary containing indicator values and the strategy decision.
        """

    # ========================================================
    # State
    # ========================================================

    def _reset_futures_state(self) -> None:
        """
        Reset all futures-specific backtest state.
        """

        self.position_direction: int = 0

        self.entry_timestamp: pd.Timestamp | None = None
        self.entry_price: float | None = None
        self.entry_atr: float | None = None
        self.stop_price: float | None = None

        # Price from which the current position will next be marked to market.
        self.previous_settlement: float | None = None

        self.account_equity: float = float(
            self.initial_capital
        )

        self.cumulative_pnl: float = 0.0

        self.futures_tradebook: list[dict[str, Any]] = []
        self.futures_equity_history: list[dict[str, Any]] = []
        self.signal_log: list[dict[str, Any]] = []

        self._daily_pnl: float = 0.0
        self._daily_event: str = "NO_EVENT"

    # ========================================================
    # Market data
    # ========================================================

    def _history_ohlc(
        self,
        bars: int,
    ) -> pd.DataFrame:
        """
        Build a normal single-level OHLCV DataFrame from Atlas history.
        """

        history = pd.DataFrame(
            {
                field: self.history(
                    symbol=self.data_symbol,
                    field=field,
                    bars=bars,
                )
                for field in (
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                )
            }
        )

        return normalize_columns(history).sort_index()

    # ========================================================
    # Futures accounting
    # ========================================================

    def portfolio_value(self) -> float:
        """
        Futures equity is account capital plus marked-to-market P&L.

        It is not the contract notional value.
        """
        return float(self.account_equity)

    def current_notional(
        self,
        price: float,
    ) -> float:
        """
        Calculate absolute contract notional exposure.

        notional =
            contracts
            × futures price
            × contract multiplier
        """

        return float(
            abs(self.position_direction)
            * self.contracts
            * float(price)
            * self.instrument.multiplier
        )

    def current_effective_leverage(
        self,
        price: float,
    ) -> float:
        """
        Calculate current notional exposure divided by account equity.
        """

        if self.account_equity <= 0:
            return float("inf")

        return (
            self.current_notional(price)
            / self.account_equity
        )

    def _pnl_between(
        self,
        start_price: float,
        end_price: float,
        direction: int | None = None,
    ) -> float:
        """
        Calculate futures P&L between two prices.

        P&L =
            direction
            × contracts
            × price change
            × contract multiplier
        """

        position = (
            self.position_direction
            if direction is None
            else int(direction)
        )

        return float(
            position
            * self.contracts
            * (
                float(end_price)
                - float(start_price)
            )
            * self.instrument.multiplier
        )

    def _mark_to_market(
        self,
        settlement_price: float,
    ) -> float:
        """
        Mark the current position from the previous settlement to the supplied
        settlement price.

        A new position entered at today's close starts earning P&L from that
        close onward. It therefore has zero P&L on its entry day.
        """

        settlement_price = float(
            settlement_price
        )

        if self.previous_settlement is None:
            self.previous_settlement = settlement_price
            return 0.0

        pnl = self._pnl_between(
            start_price=self.previous_settlement,
            end_price=settlement_price,
        )

        self.account_equity += pnl
        self.cumulative_pnl += pnl
        self._daily_pnl += pnl

        self.previous_settlement = settlement_price

        return float(pnl)

    def _full_trade_pnl(
        self,
        exit_price: float,
    ) -> float:
        """
        Calculate full entry-to-exit trade P&L.

        This is recorded for trade analysis only. It must not be added to
        account equity because equity has already been updated through daily
        mark-to-market.
        """

        if (
            self.position_direction == 0
            or self.entry_price is None
        ):
            return 0.0

        return self._pnl_between(
            start_price=self.entry_price,
            end_price=float(exit_price),
            direction=self.position_direction,
        )

    # ========================================================
    # Tradebook
    # ========================================================

    @staticmethod
    def _serialize_notes(
        notes: dict[str, Any] | str | None,
    ) -> str:
        if notes is None:
            return "{}"

        if isinstance(notes, str):
            return notes

        return json.dumps(
            notes,
            sort_keys=True,
            default=str,
        )

    def _append_trade(
        self,
        *,
        action: str,
        direction_before: int,
        direction_after: int,
        price: float,
        reason: str,
        realized_trade_pnl: float = 0.0,
        notes: dict[str, Any] | str | None = None,
        entry_timestamp: pd.Timestamp | None = None,
        entry_price: float | None = None,
        entry_atr: float | None = None,
        stop_price: float | None = None,
    ) -> None:
        """
        Append one execution event to the futures tradebook.
        """

        price = float(price)

        self.futures_tradebook.append(
            {
                "timestamp": self.get_current_timestamp(),
                "uid": self.uid,
                "strategy": self.strategy_name,
                "contract_symbol": self.symbol,
                "data_symbol": self.data_symbol,
                "action": action,
                "direction_before": int(
                    direction_before
                ),
                "direction_after": int(
                    direction_after
                ),
                "contracts": int(self.contracts),
                "price": price,
                "multiplier": float(
                    self.instrument.multiplier
                ),
                "notional": float(
                    self.contracts
                    * price
                    * self.instrument.multiplier
                ),
                "entry_timestamp": entry_timestamp,
                "entry_price": entry_price,
                "entry_atr": entry_atr,
                "stop_price": stop_price,
                "realized_trade_pnl": float(
                    realized_trade_pnl
                ),
                "account_equity": float(
                    self.account_equity
                ),
                "reason": reason,
                "notes": self._serialize_notes(
                    notes
                ),
            }
        )

    # ========================================================
    # Position operations
    # ========================================================

    def _open_position(
        self,
        *,
        direction: int,
        price: float,
        current_atr: float,
        reason: str,
        notes: dict[str, Any] | None = None,
    ) -> None:
        """
        Open a new long or short futures position.
        """

        if direction not in {-1, 1}:
            raise ValueError(
                "Opening direction must be -1 or 1."
            )

        if self.position_direction != 0:
            raise RuntimeError(
                "Cannot open a new position while another position is active."
            )

        price = float(price)
        current_atr = float(current_atr)

        if current_atr <= 0:
            raise ValueError(
                "Entry ATR must be greater than zero."
            )

        self.position_direction = int(
            direction
        )

        self.entry_timestamp = (
            self.get_current_timestamp()
        )

        self.entry_price = price
        self.entry_atr = current_atr

        stop_distance = (
            float(
                self.parameters[
                    "stop_atr_multiple"
                ]
            )
            * current_atr
        )

        if direction > 0:
            self.stop_price = (
                price - stop_distance
            )
        else:
            self.stop_price = (
                price + stop_distance
            )

        # Entry occurs at today's close, so future mark-to-market begins from
        # this price.
        self.previous_settlement = price

        self._append_trade(
            action=(
                "BUY"
                if direction > 0
                else "SELL_SHORT"
            ),
            direction_before=0,
            direction_after=direction,
            price=price,
            reason=reason,
            notes=notes,
            entry_timestamp=self.entry_timestamp,
            entry_price=self.entry_price,
            entry_atr=self.entry_atr,
            stop_price=self.stop_price,
        )

    def _close_position(
        self,
        *,
        price: float,
        reason: str,
        notes: dict[str, Any] | None = None,
    ) -> None:
        """
        Close the currently active futures position.
        """

        if self.position_direction == 0:
            return

        direction_before = int(
            self.position_direction
        )

        entry_timestamp = self.entry_timestamp
        entry_price = self.entry_price
        entry_atr = self.entry_atr
        stop_price = self.stop_price

        realized_trade_pnl = (
            self._full_trade_pnl(price)
        )

        self._append_trade(
            action=(
                "SELL"
                if direction_before > 0
                else "BUY_TO_COVER"
            ),
            direction_before=direction_before,
            direction_after=0,
            price=float(price),
            reason=reason,
            realized_trade_pnl=realized_trade_pnl,
            notes=notes,
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            entry_atr=entry_atr,
            stop_price=stop_price,
        )

        self.position_direction = 0

        self.entry_timestamp = None
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None

    def _apply_desired_position(
        self,
        *,
        desired_direction: int,
        price: float,
        current_atr: float,
        diagnostics: dict[str, Any],
    ) -> None:
        """
        Apply the concrete strategy's requested position.

        Handles:

        flat -> long
        flat -> short
        long -> flat
        short -> flat
        long -> short
        short -> long
        """

        desired_direction = int(
            desired_direction
        )

        if desired_direction not in {
            -1,
            0,
            1,
        }:
            raise ValueError(
                "generate_desired_position() must return -1, 0 or 1."
            )

        current_direction = int(
            self.position_direction
        )

        # Nothing changes.
        if desired_direction == current_direction:
            return

        # Flat to long or short.
        if current_direction == 0:
            if desired_direction != 0:
                self._open_position(
                    direction=desired_direction,
                    price=price,
                    current_atr=current_atr,
                    reason="SIGNAL_ENTRY",
                    notes=diagnostics,
                )

            return

        # Long or short to flat.
        if desired_direction == 0:
            self._close_position(
                price=price,
                reason="SIGNAL_EXIT",
                notes=diagnostics,
            )

            return

        # Long to short or short to long.
        self._close_position(
            price=price,
            reason="SIGNAL_REVERSAL_EXIT",
            notes=diagnostics,
        )

        self._open_position(
            direction=desired_direction,
            price=price,
            current_atr=current_atr,
            reason="SIGNAL_REVERSAL_ENTRY",
            notes=diagnostics,
        )

    # ========================================================
    # ATR stop
    # ========================================================

    def _stop_fill_price(
        self,
        bar: dict[str, float],
    ) -> float | None:
        """
        Determine whether the fixed ATR stop was triggered.

        Long position
        -------------
        If the bar opens below the stop:
            fill at the open.

        Otherwise, if the low touches the stop:
            fill at the stop.

        Short position
        --------------
        If the bar opens above the stop:
            fill at the open.

        Otherwise, if the high touches the stop:
            fill at the stop.
        """

        if (
            self.position_direction == 0
            or self.stop_price is None
        ):
            return None

        stop = float(self.stop_price)

        open_price = float(
            bar["open"]
        )

        high = float(
            bar["high"]
        )

        low = float(
            bar["low"]
        )

        if self.position_direction > 0:
            if open_price <= stop:
                return open_price

            if low <= stop:
                return stop

            return None

        if open_price >= stop:
            return open_price

        if high >= stop:
            return stop

        return None

    # ========================================================
    # Daily strategy event
    # ========================================================

    def on_day_close(self) -> None:
        """
        Process one daily bar.

        Order of operations
        -------------------
        1. Verify sufficient history.
        2. Check whether an existing position hit its ATR stop intraday.
        3. If stopped, close it and remain flat for the rest of the day.
        4. Otherwise mark the existing position to today's close.
        5. Generate the close-based strategy decision.
        6. Apply entries, exits or reversals at today's close.
        """

        history_bars = self.required_history()

        if not self.has_history(
            symbol=self.data_symbol,
            bars=history_bars,
        ):
            self._daily_event = (
                "INSUFFICIENT_HISTORY"
            )
            return

        data = self._history_ohlc(
            history_bars
        )

        if (
            data.empty
            or len(data) < history_bars
        ):
            self._daily_event = (
                "INSUFFICIENT_HISTORY"
            )
            return

        current_bar = {
            "open": float(
                data["open"].iloc[-1]
            ),
            "high": float(
                data["high"].iloc[-1]
            ),
            "low": float(
                data["low"].iloc[-1]
            ),
            "close": float(
                data["close"].iloc[-1]
            ),
        }

        close = current_bar["close"]

        # ----------------------------------------------------
        # Check stop before close-generated signals
        # ----------------------------------------------------

        stop_fill = self._stop_fill_price(
            current_bar
        )

        if stop_fill is not None:
            stopped_direction = int(
                self.position_direction
            )

            stop_level = float(
                self.stop_price
            )

            # Mark only from yesterday's settlement to the actual stop fill.
            self._mark_to_market(
                stop_fill
            )

            stop_notes = {
                "stop_level": stop_level,
                "stop_fill": float(
                    stop_fill
                ),
                "gap_through_stop": (
                    abs(
                        float(stop_fill)
                        - stop_level
                    )
                    > 1e-12
                ),
                "stopped_direction": (
                    stopped_direction
                ),
            }

            self._close_position(
                price=stop_fill,
                reason="ATR_STOP",
                notes=stop_notes,
            )

            # The position was flat from the stop fill through the close.
            # Using today's close as the settlement reference prevents any
            # phantom P&L if another position is entered later.
            self.previous_settlement = close

            self._daily_event = "ATR_STOP"

            self._append_signal_log(
                desired_direction=0,
                close=close,
                diagnostics={
                    **stop_notes,
                    "decision": (
                        "STOPPED_AND_REMAIN_FLAT"
                    ),
                },
            )

            return

        # ----------------------------------------------------
        # Mark existing position to today's close
        # ----------------------------------------------------

        self._mark_to_market(
            close
        )

        # ----------------------------------------------------
        # Generate strategy decision
        # ----------------------------------------------------

        desired_direction, diagnostics = (
            self.generate_desired_position(
                data
            )
        )

        # ----------------------------------------------------
        # Calculate ATR for a possible new entry
        # ----------------------------------------------------

        atr_series = atr(
            data=data,
            period=int(
                self.parameters[
                    "atr_period"
                ]
            ),
        )

        current_atr = atr_series.iloc[-1]

        if (
            pd.isna(current_atr)
            or float(current_atr) <= 0
        ):
            diagnostics = {
                **diagnostics,
                "decision": "ATR_NOT_READY",
            }

            self._daily_event = (
                "ATR_NOT_READY"
            )

            self._append_signal_log(
                desired_direction=(
                    self.position_direction
                ),
                close=close,
                diagnostics=diagnostics,
            )

            return

        direction_before = int(
            self.position_direction
        )

        # ----------------------------------------------------
        # Apply strategy decision
        # ----------------------------------------------------

        self._apply_desired_position(
            desired_direction=int(
                desired_direction
            ),
            price=close,
            current_atr=float(
                current_atr
            ),
            diagnostics=diagnostics,
        )

        direction_after = int(
            self.position_direction
        )

        if direction_after == direction_before:
            if direction_after == 0:
                self._daily_event = "FLAT"
            else:
                self._daily_event = "HOLD"

        elif direction_before == 0:
            self._daily_event = "ENTRY"

        elif direction_after == 0:
            self._daily_event = "EXIT"

        else:
            self._daily_event = "REVERSAL"

        self._append_signal_log(
            desired_direction=int(
                desired_direction
            ),
            close=close,
            diagnostics=diagnostics,
        )

    # ========================================================
    # Signal and equity logs
    # ========================================================

    def _append_signal_log(
        self,
        *,
        desired_direction: int,
        close: float,
        diagnostics: dict[str, Any],
    ) -> None:
        """
        Record the strategy's decision and indicator values for the day.
        """

        row: dict[str, Any] = {
            "timestamp": (
                self.get_current_timestamp()
            ),
            "uid": self.uid,
            "strategy": self.strategy_name,
            "contract_symbol": self.symbol,
            "data_symbol": self.data_symbol,
            "desired_direction": int(
                desired_direction
            ),
            "actual_direction": int(
                self.position_direction
            ),
            "contracts": (
                self.contracts
                if self.position_direction != 0
                else 0
            ),
            "close": float(close),
            "daily_pnl": float(
                self._daily_pnl
            ),
            "cumulative_pnl": float(
                self.cumulative_pnl
            ),
            "account_equity": float(
                self.account_equity
            ),
            "entry_price": self.entry_price,
            "entry_atr": self.entry_atr,
            "stop_price": self.stop_price,
            "event": self._daily_event,
        }

        row.update(
            diagnostics
        )

        self.signal_log.append(
            row
        )

    def _record_equity(self) -> None:
        """
        Record one row in the daily futures equity curve.
        """

        try:
            close = float(
                self.get_close(
                    self.data_symbol
                )
            )
        except KeyError:
            return

        self.futures_equity_history.append(
            {
                "timestamp": (
                    self.get_current_timestamp()
                ),
                "uid": self.uid,
                "cash": float(
                    self.account_equity
                ),
                "market_value": 0.0,
                "equity": float(
                    self.account_equity
                ),
                "daily_pnl": float(
                    self._daily_pnl
                ),
                "cumulative_pnl": float(
                    self.cumulative_pnl
                ),
                "contract_symbol": self.symbol,
                "data_symbol": self.data_symbol,
                "position_direction": int(
                    self.position_direction
                ),
                "contracts": (
                    int(self.contracts)
                    if self.position_direction != 0
                    else 0
                ),
                "close": close,
                "notional_exposure": (
                    self.current_notional(
                        close
                    )
                ),
                "effective_leverage": (
                    self.current_effective_leverage(
                        close
                    )
                ),
                "entry_price": self.entry_price,
                "entry_atr": self.entry_atr,
                "stop_price": self.stop_price,
                "event": self._daily_event,
            }
        )

    # ========================================================
    # Backtest runner
    # ========================================================

    def run(
        self,
        symbols: Iterable[str] | None = None,
        start: Optional[
            str | pd.Timestamp
        ] = None,
        end: Optional[
            str | pd.Timestamp
        ] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Run the futures strategy over the specified date range.
        """

        required = self.required_symbols()

        if symbols is None:
            requested = required
        else:
            requested = list(
                dict.fromkeys(symbols)
            )

        if requested != required:
            raise ValueError(
                f"{self.strategy_name} requires exactly "
                f"{required}; received {requested}."
            )

        self._reset_futures_state()

        dates = self.timestamps(
            symbols=required,
            start=start,
            end=end,
        )

        for timestamp in dates:
            self.set_current_timestamp(
                timestamp
            )

            self._daily_pnl = 0.0
            self._daily_event = "NO_EVENT"

            self.on_day_close()
            self._record_equity()

        return {
            "tradebook": (
                self.get_tradebook()
            ),
            "equity": (
                self.get_equity_history()
            ),
            "positions": (
                self.get_positions_frame()
            ),
            "signals": (
                self.get_signal_log()
            ),
        }

    # ========================================================
    # Outputs
    # ========================================================

    def get_tradebook(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self.futures_tradebook,
            columns=FUTURES_TRADEBOOK_COLUMNS,
        )

    def get_equity_history(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self.futures_equity_history
        )

    def get_positions_frame(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "uid": self.uid,
                    "strategy": (
                        self.strategy_name
                    ),
                    "contract_symbol": (
                        self.symbol
                    ),
                    "data_symbol": (
                        self.data_symbol
                    ),
                    "direction": int(
                        self.position_direction
                    ),
                    "contracts": (
                        int(self.contracts)
                        if self.position_direction != 0
                        else 0
                    ),
                    "entry_timestamp": (
                        self.entry_timestamp
                    ),
                    "entry_price": (
                        self.entry_price
                    ),
                    "entry_atr": (
                        self.entry_atr
                    ),
                    "stop_price": (
                        self.stop_price
                    ),
                    "account_equity": float(
                        self.account_equity
                    ),
                }
            ]
        )

    def get_signal_log(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self.signal_log
        )

    def save_results(
        self,
        output_dir: str | Path,
    ) -> None:
        """
        Save the primary futures backtest outputs.
        """

        output_path = Path(
            output_dir
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.get_tradebook().to_csv(
            output_path / "tradebook.csv",
            index=False,
        )

        self.get_equity_history().to_csv(
            output_path / "equity.csv",
            index=False,
        )

        self.get_positions_frame().to_csv(
            output_path / "positions.csv",
            index=False,
        )

        self.get_signal_log().to_csv(
            output_path / "signal_log.csv",
            index=False,
        )