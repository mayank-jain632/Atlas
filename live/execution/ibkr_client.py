from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from live.execution.account_state import AccountState

logger = logging.getLogger(__name__)


class IBKRClient:
    """
    Thin wrapper around ib_async's IB() connection.

    Owns the connect/disconnect lifecycle and translates ib_async's
    return shapes (AccountValue, Position, Ticker, ...) into this
    codebase's own types -- AccountState, plain float prices -- so
    nothing else here needs to know ib_async's API at all. Satisfies the
    AccountStateProvider Protocol (live/execution/account_state.py)
    directly via get_account_state() -- structural typing means no
    separate adapter class is needed.

    ib_async is imported lazily, inside connect() and get_last_prices(),
    not at module level. That means importing this module -- or
    unit-testing its data-translation logic against a fake `_ib` -- does
    not require ib_async to be installed; only an actual connect() call
    does.

    Nothing in this codebase constructs or connects one of these yet
    (see account_state.py's FakeAccountStateProvider for what
    run_live_day() actually uses today). This class exists so the real
    connector is ready to point at a paper account once there is one --
    it does nothing to a real account merely by being instantiated.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        readonly: bool = True,
        reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 5.0,
    ) -> None:
        # Port defaults to TWS's paper-trading port. IB Gateway's paper
        # port is 4002 (live is 4001; TWS live is 7496) -- confirm
        # against your actual gateway config rather than assuming this
        # default matches your setup.
        self.host = host
        self.port = port
        self.client_id = client_id

        # Safe by default: a readonly connection cannot place orders.
        # Mirrors the reference system's own belt-and-suspenders pattern
        # of keeping the live gateway read-only even when a client's own
        # execution_mode says "live" -- must be turned off explicitly to
        # place real orders through this client.
        self.readonly = readonly

        # Used by ensure_connected() -- an unattended scheduler has no
        # one to notice a dropped connection and reconnect manually, so
        # it needs to retry a bounded number of times itself.
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds

        self._ib: Any = None

    def connect(self) -> None:
        from ib_async import IB

        self._ib = IB()
        self._ib.connect(
            self.host,
            self.port,
            clientId=self.client_id,
            readonly=self.readonly,
        )

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
            self._ib = None

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    def ensure_connected(self) -> None:
        """
        Connect if not already connected, retrying up to
        `reconnect_attempts` times with `reconnect_delay_seconds`
        between attempts. A no-op if already connected.

        Meant for a scheduler (see live/run_daily_check.py) to call
        before each use instead of connect() directly: a gateway
        restart or a brief network blip shouldn't require a human to
        notice and reconnect by hand before the next scheduled run.
        Does not itself run on a timer or in the background -- it only
        acts when called.
        """
        if self.is_connected():
            return

        last_error: Exception | None = None
        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                self.connect()
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "IBKR connect attempt %d/%d failed: %s",
                    attempt,
                    self.reconnect_attempts,
                    exc,
                )
                if attempt < self.reconnect_attempts:
                    time.sleep(self.reconnect_delay_seconds)

        raise ConnectionError(
            f"Failed to connect to IBKR at {self.host}:{self.port} after "
            f"{self.reconnect_attempts} attempt(s)"
        ) from last_error

    def __enter__(self) -> "IBKRClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def _require_connection(self) -> Any:
        if self._ib is None or not self._ib.isConnected():
            raise RuntimeError(
                "Not connected to IBKR -- call connect() first "
                "(or use `with IBKRClient(...) as client:`)."
            )
        return self._ib

    # ========================================================
    # Account state (AccountStateProvider)
    # ========================================================

    def get_account_state(self, account: str | None = None) -> AccountState:
        """
        Fetch this account's current cash, positions, and broker-
        reported equity from IBKR and translate them into this
        codebase's AccountState.
        """
        ib = self._require_connection()

        resolved_account = account or ib.managedAccounts()[0]

        summary = {
            item.tag: item.value for item in ib.accountSummary(resolved_account)
        }

        # positions() returns positions across every managed account, not
        # just the one requested -- filtered here rather than trusting an
        # account-scoped call, since that isn't confirmed to exist.
        positions = {
            position.contract.symbol: float(position.position)
            for position in ib.positions()
            if position.account == resolved_account
        }

        return AccountState(
            cash=float(summary.get("TotalCashValue", 0.0)),
            positions=positions,
            equity=float(summary.get("NetLiquidation", 0.0)),
            as_of=pd.Timestamp.now(),
        )

    # ========================================================
    # Price lookup (polled snapshot, not a streaming subscription --
    # matches the "polling is enough for daily momentum" decision)
    # ========================================================

    def get_last_price(self, symbol: str) -> float:
        """One-shot last-traded-price lookup for a single US stock."""
        return self.get_last_prices([symbol])[symbol]

    def get_last_prices(self, symbols: list[str]) -> dict[str, float]:
        """
        One-shot snapshot of last-traded price per symbol. Blocking, not
        a persistent market data subscription.
        """
        from ib_async import Stock

        ib = self._require_connection()

        contracts = [Stock(symbol, "SMART", "USD") for symbol in symbols]
        tickers = ib.reqTickers(*contracts)

        return {
            ticker.contract.symbol: float(ticker.last)
            for ticker in tickers
            # ib_async represents "no trade yet / no subscription" as
            # NaN, not None -- `!= itself` is the standard NaN check,
            # avoids importing math just for this one comparison.
            if ticker.last == ticker.last
        }
