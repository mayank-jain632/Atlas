from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class AccountState:
    """
    A snapshot of what a brokerage account actually holds, as of a point
    in time -- the real-world counterpart to a backtest's simulated
    cash/positions.

    `equity` is the broker's own reported NetLiquidation, kept alongside
    `cash`/`positions` purely as a cross-check: once a strategy is seeded
    with `cash` and `positions` (see SignalBookLiveMixin.seed_account_state
    in live_ems.py), its own portfolio_value() recomputes equity from
    `cash` plus current local prices, and that should land close to
    `equity` here. A large gap between the two means the local price data
    is stale or wrong, not that the seeding mechanism is broken.
    """

    cash: float
    positions: dict[str, float]
    equity: float
    as_of: pd.Timestamp


class AccountStateProvider(Protocol):
    """Anything that can answer "what does the account hold right now."""

    def get_account_state(self) -> AccountState: ...


@dataclass(frozen=True)
class FakeAccountStateProvider:
    """
    Test/dev stand-in for a real broker connection (e.g. IBKR). Always
    returns the same AccountState it was constructed with. A future
    IBKR-backed provider implements the same AccountStateProvider
    Protocol without any downstream code needing to change.
    """

    account_state: AccountState

    def get_account_state(self) -> AccountState:
        return self.account_state
