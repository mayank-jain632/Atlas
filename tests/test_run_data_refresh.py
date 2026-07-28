"""
Tests for live/run_data_refresh.py -- the unattended market-data cron
script.

YahooDownloader is always monkeypatched to a fake here: this script
genuinely hits the network (Yahoo Finance) and writes to the real
market_data.duckdb when run for real, so tests must never construct a
real one. Only load_universe_symbols() runs against real local data
(config/universes/*.csv) -- a plain file read, no network involved.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import live.run_data_refresh as data_refresh
from live.execution.kill_switch import KillSwitch
from live.execution.run_log import RunLog


def _fake_downloader_class(result: dict[str, Any] | Exception):
    """Builds a fake standing in for YahooDownloader: same constructor
    signature and context-manager protocol, but download() returns (or
    raises) whatever the test wants instead of touching Yahoo Finance."""

    class _FakeYahooDownloader:
        def __init__(self, *, duckdb_path: Any) -> None:
            self.duckdb_path = duckdb_path

        def __enter__(self) -> "_FakeYahooDownloader":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def download(self, **kwargs: Any) -> dict[str, Any]:
            if isinstance(result, Exception):
                raise result
            return result

    return _FakeYahooDownloader


def _summary(
    *,
    successful: list[str] | None = None,
    empty: list[str] | None = None,
    failed: list[dict[str, str]] | None = None,
    rows_processed: int = 0,
) -> dict[str, Any]:
    return {
        "successful": successful or [],
        "empty": empty or [],
        "failed": failed or [],
        "rows_processed": rows_processed,
        "database_summary": None,
    }


@pytest.fixture(autouse=True)
def _reset_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_refresh, "STATE_DB_PATH", tmp_path / "state.sqlite")
    monkeypatch.setattr(data_refresh, "LOG_PATH", tmp_path / "data_refresh.log")
    # Unset regardless of the real environment -- no test in this file
    # should ever be able to trigger a real Telegram send.
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


def _run_with_summary(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any] | Exception) -> int:
    monkeypatch.setattr(data_refresh, "YahooDownloader", _fake_downloader_class(result))
    return data_refresh.main()


def test_load_universe_symbols_returns_a_deduplicated_list() -> None:
    symbols = data_refresh.load_universe_symbols()
    assert len(symbols) > 0
    assert len(symbols) == len(set(symbols))


def test_new_rows_records_refreshed_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_code = _run_with_summary(
        monkeypatch, _summary(successful=["AAPL", "MSFT"], rows_processed=42)
    )

    assert exit_code == 0
    with RunLog(db_path=data_refresh.STATE_DB_PATH) as run_log:
        latest = run_log.get_latest(data_refresh.JOB_NAME)
        assert latest is not None
        assert latest["decision"] == "REFRESHED"


def test_no_new_rows_records_no_new_data_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_code = _run_with_summary(
        monkeypatch, _summary(successful=["AAPL", "MSFT"], rows_processed=0)
    )

    assert exit_code == 0
    with RunLog(db_path=data_refresh.STATE_DB_PATH) as run_log:
        assert run_log.get_latest(data_refresh.JOB_NAME)["decision"] == "NO_NEW_DATA"


def test_total_failure_records_error_and_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_code = _run_with_summary(
        monkeypatch,
        _summary(failed=[{"symbol": "AAPL", "error": "boom"}], rows_processed=0),
    )

    assert exit_code == 1
    with RunLog(db_path=data_refresh.STATE_DB_PATH) as run_log:
        assert run_log.get_latest(data_refresh.JOB_NAME)["decision"] == "ERROR"


def test_partial_failure_records_partial_failure_and_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_code = _run_with_summary(
        monkeypatch,
        _summary(
            successful=["MSFT"],
            failed=[{"symbol": "AAPL", "error": "boom"}],
            rows_processed=10,
        ),
    )

    assert exit_code == 1
    with RunLog(db_path=data_refresh.STATE_DB_PATH) as run_log:
        assert run_log.get_latest(data_refresh.JOB_NAME)["decision"] == "PARTIAL_FAILURE"


def test_unexpected_exception_is_caught_and_logged_as_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_code = _run_with_summary(monkeypatch, RuntimeError("duckdb file locked"))

    assert exit_code == 1
    with RunLog(db_path=data_refresh.STATE_DB_PATH) as run_log:
        latest = run_log.get_latest(data_refresh.JOB_NAME)
        assert latest["decision"] == "ERROR"
        assert "duckdb file locked" in latest["detail"]


def test_market_data_kill_switch_skips_refresh_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with KillSwitch(db_path=data_refresh.STATE_DB_PATH) as kill_switch:
        kill_switch.set_killed("market_data", True, reason="test")

    def _fail_if_called(**kwargs: Any) -> None:
        raise AssertionError("YahooDownloader should never be constructed when killed")

    monkeypatch.setattr(data_refresh, "YahooDownloader", _fail_if_called)

    exit_code = data_refresh.main()

    assert exit_code == 0
    with RunLog(db_path=data_refresh.STATE_DB_PATH) as run_log:
        assert run_log.get_latest(data_refresh.JOB_NAME)["decision"] == "KILLED"


def test_system_kill_switch_also_skips_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    with KillSwitch(db_path=data_refresh.STATE_DB_PATH) as kill_switch:
        kill_switch.set_killed("system", True, reason="test")

    def _fail_if_called(**kwargs: Any) -> None:
        raise AssertionError("YahooDownloader should never be constructed when killed")

    monkeypatch.setattr(data_refresh, "YahooDownloader", _fail_if_called)

    exit_code = data_refresh.main()

    assert exit_code == 0
    with RunLog(db_path=data_refresh.STATE_DB_PATH) as run_log:
        assert run_log.get_latest(data_refresh.JOB_NAME)["decision"] == "KILLED"


def test_total_failure_notifies_via_the_configured_notifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[str, str]] = []

    class _FakeNotifier:
        def notify_error(self, uid: str, error: str) -> bool:
            sent.append((uid, error))
            return True

    monkeypatch.setattr(data_refresh, "notifier_from_env", lambda kill_switch: _FakeNotifier())

    exit_code = _run_with_summary(
        monkeypatch,
        _summary(failed=[{"symbol": "AAPL", "error": "boom"}], rows_processed=0),
    )

    assert exit_code == 1
    assert len(sent) == 1
    assert sent[0][0] == data_refresh.JOB_NAME


def test_unexpected_exception_notifies_via_the_configured_notifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[str, str]] = []

    class _FakeNotifier:
        def notify_error(self, uid: str, error: str) -> bool:
            sent.append((uid, error))
            return True

    monkeypatch.setattr(data_refresh, "notifier_from_env", lambda kill_switch: _FakeNotifier())

    exit_code = _run_with_summary(monkeypatch, RuntimeError("duckdb file locked"))

    assert exit_code == 1
    assert len(sent) == 1
    assert "duckdb file locked" in sent[0][1]


def test_success_does_not_notify(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, str]] = []

    class _FakeNotifier:
        def notify_error(self, uid: str, error: str) -> bool:
            sent.append((uid, error))
            return True

    monkeypatch.setattr(data_refresh, "notifier_from_env", lambda kill_switch: _FakeNotifier())

    exit_code = _run_with_summary(
        monkeypatch, _summary(successful=["AAPL"], rows_processed=10)
    )

    assert exit_code == 0
    assert sent == []


def test_no_notifier_configured_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_code = _run_with_summary(
        monkeypatch, _summary(failed=[{"symbol": "AAPL", "error": "boom"}])
    )
    assert exit_code == 1
