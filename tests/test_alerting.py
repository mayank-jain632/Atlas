"""
Tests for live/execution/alerting.py's TelegramNotifier.

_post() (the one method that actually calls the Telegram API) is
always monkeypatched here -- this module genuinely sends a real
message when used for real, so no test may let that happen. Mirrors
IBKRClient's tests never calling connect() for real.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live.execution.alerting import ALERTING_SWITCH, TelegramNotifier, notifier_from_env
from live.execution.kill_switch import KillSwitch


def _notifier_with_fake_post(**kwargs) -> tuple[TelegramNotifier, list[str]]:
    sent: list[str] = []
    notifier = TelegramNotifier(bot_token="token", chat_id="chat", **kwargs)
    notifier._post = sent.append  # type: ignore[method-assign]
    return notifier, sent


# ========================================================
# send() / notify_error() / notify_kill_switch()
# ========================================================


def test_send_calls_post_and_returns_true() -> None:
    notifier, sent = _notifier_with_fake_post()
    assert notifier.send("hello") is True
    assert sent == ["hello"]


def test_notify_error_formats_uid_and_error() -> None:
    notifier, sent = _notifier_with_fake_post()
    notifier.notify_error("uid-1", "boom")
    assert len(sent) == 1
    assert "uid-1" in sent[0]
    assert "boom" in sent[0]


def test_notify_kill_switch_formats_killed_state() -> None:
    notifier, sent = _notifier_with_fake_post()
    notifier.notify_kill_switch("system", True, reason="manual test")
    assert "KILLED" in sent[0]
    assert "system" in sent[0]
    assert "manual test" in sent[0]


def test_notify_kill_switch_formats_unkilled_state() -> None:
    notifier, sent = _notifier_with_fake_post()
    notifier.notify_kill_switch("system", False)
    assert "un-killed" in sent[0]


def test_a_real_network_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    notifier = TelegramNotifier(bot_token="token", chat_id="chat")

    def _raise(text: str) -> None:
        raise ConnectionError("no network")

    notifier._post = _raise  # type: ignore[method-assign]

    with pytest.raises(ConnectionError):
        notifier.send("hello")


# ========================================================
# Muting via the "alerting" kill switch
# ========================================================


def test_send_is_a_noop_when_alerting_is_killed(tmp_path: Path) -> None:
    with KillSwitch(db_path=tmp_path / "ks.sqlite") as kill_switch:
        kill_switch.set_killed(ALERTING_SWITCH, True, reason="test")

        notifier, sent = _notifier_with_fake_post(kill_switch=kill_switch)
        result = notifier.send("hello")

        assert result is False
        assert sent == []


def test_send_works_normally_when_alerting_is_not_killed(tmp_path: Path) -> None:
    with KillSwitch(db_path=tmp_path / "ks.sqlite") as kill_switch:
        notifier, sent = _notifier_with_fake_post(kill_switch=kill_switch)
        assert notifier.send("hello") is True
        assert sent == ["hello"]


def test_send_is_a_noop_when_system_is_killed(tmp_path: Path) -> None:
    # is_killed() honors the system-wide switch for any name, including
    # "alerting" -- killing the whole system mutes alerting too.
    with KillSwitch(db_path=tmp_path / "ks.sqlite") as kill_switch:
        kill_switch.set_killed("system", True, reason="test")

        notifier, sent = _notifier_with_fake_post(kill_switch=kill_switch)
        assert notifier.send("hello") is False
        assert sent == []


def test_notifier_without_a_kill_switch_is_never_muted() -> None:
    notifier, sent = _notifier_with_fake_post(kill_switch=None)
    assert notifier.send("hello") is True
    assert sent == ["hello"]


# ========================================================
# notifier_from_env
# ========================================================


def test_notifier_from_env_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notifier_from_env() is None


def test_notifier_from_env_returns_none_when_only_token_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notifier_from_env() is None


def test_notifier_from_env_builds_a_notifier_when_both_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token-123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-456")

    notifier = notifier_from_env()

    assert notifier is not None
    assert notifier.bot_token == "token-123"
    assert notifier.chat_id == "chat-456"
