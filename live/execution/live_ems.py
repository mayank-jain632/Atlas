from __future__ import annotations

from typing import Any

import pandas as pd

from live.execution.account_state import AccountState, AccountStateProvider
from live.execution.run_log import RunLog
from live.execution.signal_book import SignalBook


class SignalBookLiveMixin:
    """
    Mix in ahead of any existing strategy class (MomentumStrategy,
    MomentumIndicatorStrategy, BaseFuturesStrategy subclasses, ...) to
    redirect every place_trade() this instance makes into a SignalBook,
    in addition to the normal simulated-ledger bookkeeping EMS already
    does. Nothing about the strategy's own signal/weight computation
    changes -- rebalance_to_weights(), target_quantity(), and
    on_day_close() all run exactly as they do in a backtest, reading
    this instance's own simulated cash/positions along the way. This
    class only taps place_trade()'s result.

    Trades made during one on_day_close() call are buffered and
    flushed as a single signal_book write per symbol touched, keyed by
    the strategy's current timestamp and the reason passed to
    place_trade() (e.g. "MOMENTUM_REBALANCE_SELL"/"..._BUY" -- see
    EMS.rebalance_to_weights()). A symbol whose position is closed to
    zero still produces a real place_trade() call (the executed
    quantity is nonzero even though position_after is 0.0), so it flows
    through this hook and reaches the signal book as an explicit
    target_quantity=0.0 row -- consistent with SignalBook's "no
    implicit absence means zero" rule.
    """

    def __init__(self, *args: Any, signal_book: SignalBook, **kwargs: Any) -> None:
        self._signal_book = signal_book
        self._pending_targets: dict[str, float] = {}
        self._pending_notes: dict[str, dict[str, Any]] = {}
        self._pending_reason: str | None = None
        super().__init__(*args, **kwargs)

    def place_trade(
        self,
        symbol: str,
        quantity: float,
        reason: str = "SIGNAL",
        notes: dict[str, Any] | str | None = None,
        price_field: str = "close",
    ) -> dict[str, Any] | None:
        row = super().place_trade(
            symbol,
            quantity,
            reason=reason,
            notes=notes,
            price_field=price_field,
        )

        if row is not None:
            self._pending_targets[symbol] = row["position_after"]
            self._pending_reason = reason
            if isinstance(notes, dict):
                self._pending_notes[symbol] = notes

        return row

    def flush_signal_book(self) -> bool:
        """
        Write everything place_trade() produced since the last flush as
        one signal_book batch, then clear the buffer. A no-op if
        nothing traded (e.g. should_rebalance() was False that day).

        Returns True if anything was written, False otherwise -- lets
        callers (run_live_day's run-log entry) record what actually
        happened without reaching into this instance's private buffer.
        """
        if not self._pending_targets:
            return False

        self._signal_book.write_targets(
            uid=self.uid,
            targets=dict(self._pending_targets),
            as_of=self.get_current_timestamp(),
            reason=self._pending_reason or "REBALANCE",
            notes_by_symbol=dict(self._pending_notes),
        )

        self._pending_targets.clear()
        self._pending_notes.clear()
        self._pending_reason = None
        return True

    def seed_account_state(self, account_state: AccountState) -> None:
        """
        Overwrite this instance's simulated cash/positions with the real
        account's current cash/positions, so rebalance_to_weights()'s
        target computation reflects actual buying power and actual
        current holdings instead of a flat starting capital and an empty
        book.

        Deliberately does not touch tradebook/equity_history: a live run
        only needs those to cover today's single on_day_close() call
        (flush_signal_book() reads them for that) -- the durable
        multi-day record is the signal book on disk, not the in-memory
        EMS ledger, so there's nothing from prior days to preserve here.
        """
        self.cash = float(account_state.cash)
        self.positions = dict(account_state.positions)


def make_live_strategy(
    strategy_class: type,
    *,
    signal_book: SignalBook,
    **kwargs: Any,
) -> Any:
    """
    Build a strategy instance with the signal-book hook mixed in ahead
    of `strategy_class`, without modifying `strategy_class` itself.
    """
    live_class = type(
        f"Live{strategy_class.__name__}",
        (SignalBookLiveMixin, strategy_class),
        {},
    )
    return live_class(signal_book=signal_book, **kwargs)


def run_live_day(
    strategy: Any,
    as_of: str | pd.Timestamp,
    account_state_provider: AccountStateProvider | None = None,
    run_log: RunLog | None = None,
    data_as_of: str | pd.Timestamp | None = None,
) -> None:
    """
    Evaluate one strategy for a single day, the way a live scheduler
    eventually will: optionally seed real account cash/positions, set
    the timestamp, let the strategy's own on_day_close() decide whether
    to act (should_rebalance(), signal computation, all unchanged), then
    flush anything it did to the signal book.

    `account_state_provider` is optional so existing flat-capital,
    empty-book callers keep working unchanged (a "no live account yet"
    scenario) -- pass one to seed real cash/positions before the
    strategy computes its targets.

    `run_log`, if given, gets exactly one entry for this call regardless
    of outcome: "REBALANCED" (targets were written), "NO_REBALANCE"
    (ran cleanly, nothing to do -- should_rebalance() was False, or the
    strategy didn't have enough history yet), or "ERROR" (on_day_close()
    or flush_signal_book() raised -- the exception is still re-raised
    after being logged, so callers/schedulers see the failure too, not
    just the dashboard). This is what makes a scheduled check provable
    from the outside: the signal book alone can't distinguish "correctly
    decided not to act" from "never ran."

    `data_as_of` records the timestamp of the market data actually used,
    when the caller knows it (e.g. from a live data source) -- defaults
    to `as_of` when not given, since today that's the only timestamp
    available (see run_log.py).
    """
    resolved_data_as_of = data_as_of if data_as_of is not None else as_of

    if account_state_provider is not None:
        strategy.seed_account_state(account_state_provider.get_account_state())

    strategy.set_current_timestamp(as_of)

    try:
        strategy.on_day_close()
        rebalanced = strategy.flush_signal_book()
    except Exception as exc:
        if run_log is not None:
            run_log.record(
                uid=strategy.uid,
                as_of=as_of,
                decision="ERROR",
                data_as_of=resolved_data_as_of,
                detail={"error": str(exc)},
            )
        raise

    if run_log is not None:
        run_log.record(
            uid=strategy.uid,
            as_of=as_of,
            decision="REBALANCED" if rebalanced else "NO_REBALANCE",
            data_as_of=resolved_data_as_of,
        )
