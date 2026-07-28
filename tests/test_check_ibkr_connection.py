"""
Tests for live/check_ibkr_connection.py's call sequence, against a fake
client -- this script genuinely connects to a broker when run for real,
so it's never exercised here against an actual IBKRClient/gateway, only
against a duck-typed fake that proves the sequencing (connect, read
state, read a price, disconnect) and error handling are correct.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live.check_ibkr_connection import main, run_check
from live.execution.account_state import AccountState


class _FakeClient:
    def __init__(self, *, fail_at: str | None = None) -> None:
        self.host = "127.0.0.1"
        self.port = 7497
        self.client_id = 1
        self.calls: list[str] = []
        self._fail_at = fail_at

    def _maybe_fail(self, name: str) -> None:
        self.calls.append(name)
        if name == self._fail_at:
            raise RuntimeError(f"simulated failure at {name}")

    def ensure_connected(self) -> None:
        self._maybe_fail("ensure_connected")

    def get_account_state(self) -> AccountState:
        self._maybe_fail("get_account_state")
        return AccountState(
            cash=1_000.0, positions={"AAPL": 2.0}, equity=1_500.0, as_of="2026-01-05"
        )

    def get_last_price(self, symbol: str) -> float:
        self._maybe_fail("get_last_price")
        return 210.5

    def disconnect(self) -> None:
        self.calls.append("disconnect")


def test_run_check_calls_everything_in_order() -> None:
    client = _FakeClient()
    run_check(client, "AAPL")

    assert client.calls == [
        "ensure_connected",
        "get_account_state",
        "get_last_price",
        "disconnect",
    ]


@pytest.mark.parametrize(
    "fail_at", ["ensure_connected", "get_account_state", "get_last_price"]
)
def test_run_check_propagates_a_failure_at_any_step(fail_at: str) -> None:
    client = _FakeClient(fail_at=fail_at)

    with pytest.raises(RuntimeError, match=f"simulated failure at {fail_at}"):
        run_check(client, "AAPL")


def test_main_disconnects_and_returns_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _FailingClient(_FakeClient):
        def ensure_connected(self) -> None:
            calls.append("ensure_connected")
            raise ConnectionRefusedError("no gateway listening")

        def disconnect(self) -> None:
            calls.append("disconnect")

    monkeypatch.setattr(
        "live.check_ibkr_connection.IBKRClient",
        lambda **kwargs: _FailingClient(),
    )

    exit_code = main(["--symbol", "AAPL"])

    assert exit_code == 1
    assert calls == ["ensure_connected", "disconnect"]


def test_main_returns_zero_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "live.check_ibkr_connection.IBKRClient",
        lambda **kwargs: _FakeClient(),
    )

    assert main(["--symbol", "AAPL"]) == 0
