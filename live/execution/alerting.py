from __future__ import annotations

import logging
import os

from live.execution.kill_switch import KillSwitch

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# A named switch like every other process this system tracks (see
# kill_switch.py's DEFAULT_SWITCHES) -- muting alerting is exactly as
# dashboard-killable as market_data or strategy_engine, no separate UI
# needed for it.
ALERTING_SWITCH = "alerting"


class TelegramNotifier:
    """
    Sends alert messages to a single Telegram chat via the Bot API --
    the same channel the user's other program already uses, chosen
    over Discord for that reason.

    `bot_token`/`chat_id` come from BotFather / the target chat and are
    never hardcoded here -- pass them in (see notifier_from_env() below
    for the env-var-based constructor this codebase actually uses).

    requests is imported lazily inside _post(), not at module level --
    same discipline as ibkr_client.py's lazy ib_async import: importing
    or unit-testing this module doesn't require requests to be
    installed (it happens to already be, as a yfinance dependency, but
    this module doesn't rely on that), and nothing here sends a real
    Telegram message just by being constructed.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        kill_switch: KillSwitch | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.kill_switch = kill_switch
        self.timeout_seconds = timeout_seconds

    def _is_muted(self) -> bool:
        return self.kill_switch is not None and self.kill_switch.is_killed(
            ALERTING_SWITCH
        )

    def _post(self, text: str) -> None:
        """The actual network call, isolated so tests monkeypatch this
        one method instead of touching the real Telegram API -- same
        pattern as IBKRClient.connect() being the one thing tests never
        call for real."""
        import requests

        response = requests.post(
            TELEGRAM_API_URL.format(token=self.bot_token),
            json={"chat_id": self.chat_id, "text": text},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def send(self, text: str) -> bool:
        """
        Send `text` to the configured chat. Returns False without
        sending (does not raise) if alerting is muted via the
        "alerting" kill switch -- a muted channel failing loudly would
        defeat the point of muting it. A real network/API failure DOES
        raise, though: a silently-broken notifier that swallows its own
        errors is worse than a script that stops and says so.
        """
        if self._is_muted():
            logger.info("Alerting muted (kill switch) -- not sending: %s", text)
            return False

        self._post(text)
        return True

    def notify_error(self, uid: str, error: str) -> bool:
        return self.send(f"\U0001f534 ERROR: {uid}\n{error}")

    def notify_kill_switch(self, name: str, killed: bool, reason: str = "") -> bool:
        state = "KILLED" if killed else "un-killed"
        text = f"⏸️ Kill switch {state}: {name}"
        if reason:
            text += f"\nReason: {reason}"
        return self.send(text)


def notifier_from_env(kill_switch: KillSwitch | None = None) -> TelegramNotifier | None:
    """
    Build a TelegramNotifier from TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID
    environment variables, or None if either is unset. Same "build
    against real config once it exists, degrade cleanly when it
    doesn't" shape as USE_REAL_IBKR in run_daily_check.py -- every
    caller that wants to alert gets this instead of constructing
    TelegramNotifier directly, so "no credentials configured" is
    handled in exactly one place.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return None

    return TelegramNotifier(bot_token=bot_token, chat_id=chat_id, kill_switch=kill_switch)
