"""
Tests IBKRClient's data-translation logic (accountSummary -> AccountState,
positions filtering, ticker -> price dict, NaN handling) against fake
ib_async-shaped objects, injected directly as `_ib` -- no real IBKR
connection, and no dependency on ib_async actually being installed,
since connect() is never called here. That's deliberate: this project
isn't logging into IBKR yet (see the module docstring in ibkr_client.py),
but the translation logic underneath connect() is real code that should
be correct now, not just plausible-looking.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live.execution.ibkr_client import IBKRClient


@dataclass
class _FakeAccountValue:
    tag: str
    value: str


@dataclass
class _FakeContract:
    symbol: str


@dataclass
class _FakePosition:
    account: str
    contract: _FakeContract
    position: float
    avgCost: float


@dataclass
class _FakeTicker:
    contract: _FakeContract
    last: float


class _FakeIB:
    """Mimics just the ib_async.IB surface IBKRClient actually calls."""

    def __init__(
        self,
        *,
        managed_accounts: list[str],
        account_summary: dict[str, list[_FakeAccountValue]],
        positions: list[_FakePosition],
        tickers_by_symbol: dict[str, _FakeTicker],
        connected: bool = True,
    ) -> None:
        self._managed_accounts = managed_accounts
        self._account_summary = account_summary
        self._positions = positions
        self._tickers_by_symbol = tickers_by_symbol
        self._connected = connected

    def isConnected(self) -> bool:
        return self._connected

    def managedAccounts(self) -> list[str]:
        return self._managed_accounts

    def accountSummary(self, account: str) -> list[_FakeAccountValue]:
        return self._account_summary[account]

    def positions(self) -> list[_FakePosition]:
        return self._positions

    def reqTickers(self, *contracts) -> list[_FakeTicker]:
        return [self._tickers_by_symbol[c.symbol] for c in contracts]

    def disconnect(self) -> None:
        self._connected = False


def _connected_client(fake_ib: _FakeIB) -> IBKRClient:
    """An IBKRClient wired to a fake IB, bypassing connect() entirely."""
    client = IBKRClient()
    client._ib = fake_ib
    return client


def test_not_connected_raises_a_clear_error() -> None:
    client = IBKRClient()

    with pytest.raises(RuntimeError, match="Not connected"):
        client.get_account_state()


def test_is_connected_reflects_the_underlying_ib() -> None:
    client = IBKRClient()
    assert client.is_connected() is False

    client._ib = _FakeIB(
        managed_accounts=["DU123"],
        account_summary={},
        positions=[],
        tickers_by_symbol={},
    )
    assert client.is_connected() is True

    client._ib.disconnect()
    assert client.is_connected() is False


def test_get_account_state_translates_summary_and_positions() -> None:
    fake_ib = _FakeIB(
        managed_accounts=["DU123"],
        account_summary={
            "DU123": [
                _FakeAccountValue(tag="NetLiquidation", value="7500.50"),
                _FakeAccountValue(tag="TotalCashValue", value="1000.25"),
                _FakeAccountValue(tag="BuyingPower", value="4000.00"),
            ]
        },
        positions=[
            _FakePosition(
                account="DU123",
                contract=_FakeContract(symbol="AAPL"),
                position=2.0,
                avgCost=150.0,
            ),
            _FakePosition(
                account="DU123",
                contract=_FakeContract(symbol="IBM"),
                position=5.0,
                avgCost=120.0,
            ),
        ],
        tickers_by_symbol={},
    )

    state = _connected_client(fake_ib).get_account_state()

    assert state.cash == 1000.25
    assert state.equity == 7500.50
    assert state.positions == {"AAPL": 2.0, "IBM": 5.0}


def test_get_account_state_defaults_to_the_first_managed_account() -> None:
    fake_ib = _FakeIB(
        managed_accounts=["DU123"],
        account_summary={
            "DU123": [
                _FakeAccountValue(tag="NetLiquidation", value="1000.0"),
                _FakeAccountValue(tag="TotalCashValue", value="1000.0"),
            ]
        },
        positions=[],
        tickers_by_symbol={},
    )

    state = _connected_client(fake_ib).get_account_state()
    assert state.cash == 1000.0


def test_get_account_state_filters_positions_to_the_requested_account() -> None:
    # positions() returns every managed account's positions -- only the
    # requested account's rows should end up in the AccountState.
    fake_ib = _FakeIB(
        managed_accounts=["DU123"],
        account_summary={
            "DU123": [
                _FakeAccountValue(tag="NetLiquidation", value="1000.0"),
                _FakeAccountValue(tag="TotalCashValue", value="1000.0"),
            ]
        },
        positions=[
            _FakePosition(
                account="DU123",
                contract=_FakeContract(symbol="AAPL"),
                position=2.0,
                avgCost=150.0,
            ),
            _FakePosition(
                account="DU999",
                contract=_FakeContract(symbol="MSFT"),
                position=9.0,
                avgCost=300.0,
            ),
        ],
        tickers_by_symbol={},
    )

    state = _connected_client(fake_ib).get_account_state(account="DU123")
    assert state.positions == {"AAPL": 2.0}


def test_get_account_state_defaults_missing_summary_fields_to_zero() -> None:
    fake_ib = _FakeIB(
        managed_accounts=["DU123"],
        account_summary={"DU123": []},
        positions=[],
        tickers_by_symbol={},
    )

    state = _connected_client(fake_ib).get_account_state()
    assert state.cash == 0.0
    assert state.equity == 0.0


def test_get_last_prices_returns_a_symbol_keyed_dict() -> None:
    fake_ib = _FakeIB(
        managed_accounts=["DU123"],
        account_summary={},
        positions=[],
        tickers_by_symbol={
            "AAPL": _FakeTicker(contract=_FakeContract(symbol="AAPL"), last=210.5),
            "MSFT": _FakeTicker(contract=_FakeContract(symbol="MSFT"), last=415.25),
        },
    )

    prices = _connected_client(fake_ib).get_last_prices(["AAPL", "MSFT"])
    assert prices == {"AAPL": 210.5, "MSFT": 415.25}


def test_get_last_price_single_symbol() -> None:
    fake_ib = _FakeIB(
        managed_accounts=["DU123"],
        account_summary={},
        positions=[],
        tickers_by_symbol={
            "AAPL": _FakeTicker(contract=_FakeContract(symbol="AAPL"), last=210.5),
        },
    )

    assert _connected_client(fake_ib).get_last_price("AAPL") == 210.5


def test_get_last_prices_drops_symbols_with_nan_last() -> None:
    fake_ib = _FakeIB(
        managed_accounts=["DU123"],
        account_summary={},
        positions=[],
        tickers_by_symbol={
            "AAPL": _FakeTicker(contract=_FakeContract(symbol="AAPL"), last=210.5),
            "ILLIQUID": _FakeTicker(
                contract=_FakeContract(symbol="ILLIQUID"), last=math.nan
            ),
        },
    )

    prices = _connected_client(fake_ib).get_last_prices(["AAPL", "ILLIQUID"])
    assert prices == {"AAPL": 210.5}


def test_readonly_defaults_to_true() -> None:
    # Safe-by-default: a fresh client can't place orders unless readonly
    # is explicitly turned off.
    assert IBKRClient().readonly is True
    assert IBKRClient(readonly=False).readonly is False


# ========================================================
# ensure_connected() -- reconnect logic, tested against a fake connect()
# rather than a real gateway. reconnect_delay_seconds=0 everywhere so
# these tests don't actually sleep.
# ========================================================


def test_ensure_connected_is_a_noop_when_already_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _connected_client(
        _FakeIB(managed_accounts=["DU123"], account_summary={}, positions=[], tickers_by_symbol={})
    )

    def _fail_if_called() -> None:
        raise AssertionError("connect() should not be called when already connected")

    monkeypatch.setattr(client, "connect", _fail_if_called)
    client.ensure_connected()  # must not raise


def test_ensure_connected_reconnects_when_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = IBKRClient(reconnect_delay_seconds=0)
    calls = {"count": 0}

    def _fake_connect() -> None:
        calls["count"] += 1
        client._ib = _FakeIB(
            managed_accounts=["DU123"], account_summary={}, positions=[], tickers_by_symbol={}
        )

    monkeypatch.setattr(client, "connect", _fake_connect)
    client.ensure_connected()

    assert calls["count"] == 1
    assert client.is_connected()


def test_ensure_connected_retries_before_succeeding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = IBKRClient(reconnect_attempts=3, reconnect_delay_seconds=0)
    calls = {"count": 0}

    def _fake_connect() -> None:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionRefusedError("gateway not up yet")
        client._ib = _FakeIB(
            managed_accounts=["DU123"], account_summary={}, positions=[], tickers_by_symbol={}
        )

    monkeypatch.setattr(client, "connect", _fake_connect)
    client.ensure_connected()

    assert calls["count"] == 3
    assert client.is_connected()


def test_ensure_connected_raises_after_exhausting_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = IBKRClient(reconnect_attempts=2, reconnect_delay_seconds=0)
    calls = {"count": 0}

    def _always_fail() -> None:
        calls["count"] += 1
        raise ConnectionRefusedError("gateway down")

    monkeypatch.setattr(client, "connect", _always_fail)

    with pytest.raises(ConnectionError, match="Failed to connect to IBKR"):
        client.ensure_connected()

    assert calls["count"] == 2
    assert not client.is_connected()
