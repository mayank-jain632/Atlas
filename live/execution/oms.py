from __future__ import annotations

from dataclasses import dataclass

from live.execution.account_state import AccountStateProvider
from live.execution.signal_book import SignalBook


@dataclass(frozen=True)
class Order:
    """
    One order needed to move an actual broker position toward its
    signal-book target -- the reconciliation layer's output, not
    something placed yet. `quantity` is a signed delta: positive means
    buy, negative means sell, matching the exact convention EMS.place_trade()
    and EMS.target_quantity() already use (see ems/ems.py) -- so this
    slots into that same convention once real order placement exists,
    rather than inventing a second one.
    """

    symbol: str
    quantity: float

    @property
    def action(self) -> str:
        return "BUY" if self.quantity > 0 else "SELL"


def compute_orders(
    targets: dict[str, float],
    actual_positions: dict[str, float],
    min_quantity: float = 1e-6,
) -> list[Order]:
    """
    Diff target positions (from SignalBook.get_latest_targets()) against
    actual broker positions (from AccountStateProvider.get_account_state()
    .positions) and return the orders needed to close the gap for every
    symbol on either side.

    A symbol present in `actual_positions` but absent from `targets` is
    treated as an implicit target of 0 -- SignalBook's own convention is
    that a rebalance dropping a symbol still writes an explicit
    target_quantity=0 row for it, but this stays defensive against a
    stray position (e.g. a manual trade, or a holding from before this
    UID existed) that no rebalance ever wrote a target for at all.

    Orders whose absolute size is below `min_quantity` are dropped --
    avoids generating a "sell 1e-9 shares" order out of floating-point
    noise from fractional-share sizing, not a real position to close.
    """
    all_symbols = set(targets) | set(actual_positions)

    orders = []
    for symbol in sorted(all_symbols):
        delta = targets.get(symbol, 0.0) - actual_positions.get(symbol, 0.0)
        if abs(delta) >= min_quantity:
            orders.append(Order(symbol=symbol, quantity=delta))

    return orders


def reconcile_uid(
    uid: str,
    signal_book: SignalBook,
    account_state_provider: AccountStateProvider,
    min_quantity: float = 1e-6,
) -> list[Order]:
    """
    Convenience wrapper composing the two existing pieces reconciliation
    actually needs -- the latest signal-book targets for `uid` and the
    account's current actual positions -- into the order list
    compute_orders() would produce. No new I/O: both calls are the same
    ones SignalBook/AccountStateProvider already expose elsewhere.
    """
    targets = signal_book.get_latest_targets(uid)
    actual_positions = account_state_provider.get_account_state().positions
    return compute_orders(targets, actual_positions, min_quantity=min_quantity)
