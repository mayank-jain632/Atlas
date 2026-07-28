"""
Tests for live/run_daily_check.py -- the unattended cron script.

Overrides the module's config constants (UIDS, STATE_DB_PATH, LOG_PATH,
...) directly on the imported module before calling main(), the same
pattern runners/run_futures_grid.py's own tests use for its module-level
config. Everything here runs against the real market_data.duckdb (same
as tests/test_live_ems.py) since the whole point of this script is
running a real, unmodified strategy -- skipped automatically if that
file isn't available on the machine running the tests.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import live.run_daily_check as daily_check
from live.execution.account_state import AccountState
from live.execution.kill_switch import KillSwitch
from live.execution.run_log import RunLog
from live.execution.signal_book import SignalBook

DB_PATH = PROJECT_ROOT / "duckdb" / "market_data.duckdb"
VALID_UID = "momentum__u=nasdaq100__sig=price__lb=90__rb=monthly__n=10__alloc=score"

requires_market_data = pytest.mark.skipif(
    not DB_PATH.exists(), reason=f"{DB_PATH} not available on this machine"
)


@pytest.fixture(autouse=True)
def _reset_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every module-level config constant at tmp_path so tests
    never touch the real state DB or the real log file, and restore the
    originals afterward regardless of pass/fail."""
    monkeypatch.setattr(daily_check, "STATE_DB_PATH", tmp_path / "state.sqlite")
    monkeypatch.setattr(daily_check, "LOG_PATH", tmp_path / "daily_check.log")
    monkeypatch.setattr(daily_check, "UIDS", [VALID_UID])
    monkeypatch.setattr(daily_check, "CAPITAL", 100_000.0)
    monkeypatch.setattr(daily_check, "DB_PATH", None)
    monkeypatch.setattr(daily_check, "ACCOUNT_STATE_PROVIDER", None)
    # Unset regardless of the real environment, so notifier_from_env()
    # inside main() reliably returns None here -- no test in this file
    # should ever be able to trigger a real Telegram send.
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


@requires_market_data
def test_successful_run_records_rebalanced_and_returns_zero() -> None:
    exit_code = daily_check.main()

    assert exit_code == 0

    with RunLog(db_path=daily_check.STATE_DB_PATH) as run_log:
        latest = run_log.get_latest(VALID_UID)
        assert latest is not None
        assert latest["decision"] == "REBALANCED"

    with SignalBook(db_path=daily_check.STATE_DB_PATH) as signal_book:
        assert signal_book.get_latest_targets(VALID_UID)


@requires_market_data
def test_per_uid_kill_switch_skips_that_uid_without_writing_targets() -> None:
    with KillSwitch(db_path=daily_check.STATE_DB_PATH) as kill_switch:
        kill_switch.set_killed(VALID_UID, True, reason="test")

    exit_code = daily_check.main()

    assert exit_code == 0  # a deliberate skip is not a failure

    with RunLog(db_path=daily_check.STATE_DB_PATH) as run_log:
        assert run_log.get_latest(VALID_UID)["decision"] == "KILLED"

    with SignalBook(db_path=daily_check.STATE_DB_PATH) as signal_book:
        assert signal_book.get_latest_targets(VALID_UID) == {}


@requires_market_data
def test_strategy_engine_kill_switch_skips_every_uid() -> None:
    with KillSwitch(db_path=daily_check.STATE_DB_PATH) as kill_switch:
        kill_switch.set_killed("strategy_engine", True, reason="test")

    exit_code = daily_check.main()

    assert exit_code == 0
    with RunLog(db_path=daily_check.STATE_DB_PATH) as run_log:
        assert run_log.get_latest(VALID_UID)["decision"] == "KILLED"


def test_unknown_strategy_in_uid_is_logged_as_error_and_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daily_check, "UIDS", ["not_a_real_strategy__x=1"])

    exit_code = daily_check.main()

    assert exit_code == 1
    with RunLog(db_path=daily_check.STATE_DB_PATH) as run_log:
        latest = run_log.get_latest("not_a_real_strategy__x=1")
        assert latest is not None
        assert latest["decision"] == "ERROR"


@requires_market_data
def test_one_failing_uid_does_not_stop_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daily_check, "UIDS", ["not_a_real_strategy__x=1", VALID_UID])

    exit_code = daily_check.main()

    assert exit_code == 1  # overall failure, since one UID errored
    with RunLog(db_path=daily_check.STATE_DB_PATH) as run_log:
        assert run_log.get_latest("not_a_real_strategy__x=1")["decision"] == "ERROR"
        # ...but the valid UID after it in the list still ran cleanly.
        assert run_log.get_latest(VALID_UID)["decision"] == "REBALANCED"


def test_empty_uids_list_returns_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daily_check, "UIDS", [])
    assert daily_check.main() == 1


@requires_market_data
def test_use_real_ibkr_flag_swaps_in_the_ibkr_client_and_always_disconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """USE_REAL_IBKR=True should route through IBKRClient instead of
    ACCOUNT_STATE_PROVIDER, and disconnect() must fire even though this
    never touches a real gateway -- IBKRClient itself is swapped for a
    fake here, since the point is proving the wiring, not the network
    call."""
    calls = {"ensure_connected": 0, "disconnect": 0, "get_account_state": 0}

    class _FakeIBKRClient:
        def __init__(self, *, host: str, port: int, client_id: int, readonly: bool) -> None:
            self.host, self.port, self.client_id, self.readonly = (
                host,
                port,
                client_id,
                readonly,
            )

        def ensure_connected(self) -> None:
            calls["ensure_connected"] += 1

        def disconnect(self) -> None:
            calls["disconnect"] += 1

        def get_account_state(self) -> AccountState:
            calls["get_account_state"] += 1
            return AccountState(
                cash=100_000.0, positions={}, equity=100_000.0, as_of=pd.Timestamp.now()
            )

    monkeypatch.setattr(daily_check, "USE_REAL_IBKR", True)
    monkeypatch.setattr(daily_check, "IBKRClient", _FakeIBKRClient)

    exit_code = daily_check.main()

    assert exit_code == 0
    assert calls["ensure_connected"] == 1
    assert calls["get_account_state"] >= 1
    assert calls["disconnect"] == 1


def test_use_real_ibkr_flag_still_disconnects_if_a_uid_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"disconnect": 0}

    class _FakeIBKRClient:
        def __init__(self, *, host: str, port: int, client_id: int, readonly: bool) -> None:
            pass

        def ensure_connected(self) -> None:
            pass

        def disconnect(self) -> None:
            calls["disconnect"] += 1

        def get_account_state(self) -> AccountState:
            raise AssertionError("unreachable -- the UID fails before seeding runs")

    monkeypatch.setattr(daily_check, "USE_REAL_IBKR", True)
    monkeypatch.setattr(daily_check, "IBKRClient", _FakeIBKRClient)
    monkeypatch.setattr(daily_check, "UIDS", ["not_a_real_strategy__x=1"])

    exit_code = daily_check.main()

    assert exit_code == 1
    assert calls["disconnect"] == 1


def test_a_uid_error_notifies_via_the_configured_notifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves the wiring, not the network call: notifier_from_env() is
    replaced with a fake that records what it was asked to send,
    regardless of TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID."""
    sent: list[tuple[str, str]] = []

    class _FakeNotifier:
        def notify_error(self, uid: str, error: str) -> bool:
            sent.append((uid, error))
            return True

    monkeypatch.setattr(daily_check, "UIDS", ["not_a_real_strategy__x=1"])
    monkeypatch.setattr(daily_check, "notifier_from_env", lambda kill_switch: _FakeNotifier())

    exit_code = daily_check.main()

    assert exit_code == 1
    assert len(sent) == 1
    assert sent[0][0] == "not_a_real_strategy__x=1"


def test_no_notifier_configured_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    # notifier_from_env() returning None (the default, no env vars set)
    # must not crash the failure path that would otherwise call it.
    monkeypatch.setattr(daily_check, "UIDS", ["not_a_real_strategy__x=1"])
    assert daily_check.main() == 1
