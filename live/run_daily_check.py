"""
Daily scheduled signal check.

This is the script a cron job (or later, a systemd timer on the server)
invokes once a day, unattended -- nobody is watching it run. For each
configured UID it checks kill switches, runs one real day through
run_live_day() (see live/execution/live_ems.py), and logs the outcome
to the run log regardless of what happened. That last part is what
makes "did this actually run today" provable from the dashboard
afterward instead of ambiguous.

Reuses the same strategy classes and UID parsing runners/run_strats.py
uses for backtests -- a UID means the same thing here as it does there,
just live-wrapped via make_live_strategy() instead of run directly.

Only the script itself is new. The crontab entry / systemd timer that
calls it, and the DST-safe dual-registration pattern the reference
system uses for that, wait until the server exists -- nothing here
assumes a particular scheduling mechanism.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from live.execution.account_state import AccountStateProvider
from live.execution.alerting import TelegramNotifier, notifier_from_env
from live.execution.ibkr_client import IBKRClient
from live.execution.kill_switch import KillSwitch
from live.execution.live_ems import make_live_strategy, run_live_day
from live.execution.run_log import RunLog
from live.execution.signal_book import DEFAULT_SIGNAL_BOOK_PATH, SignalBook
from runners.run_strats import MOMENTUM_STRATEGIES, STRATEGY_CLASSES

# ============================================================
# Config
# ============================================================

UIDS: list[str] = [
    "momentum__u=nasdaq100__sig=price__lb=90__rb=monthly__n=10__alloc=score",
]

CAPITAL = 100_000.0
DB_PATH: str | None = None  # None -> market_data.duckdb default
TIMEFRAME = "1d"
UNIVERSE_ROOT = str(PROJECT_ROOT / "config" / "universes")

# signal_book / run_log / kill_switch all share this one SQLite file.
STATE_DB_PATH = DEFAULT_SIGNAL_BOOK_PATH

# Used only when USE_REAL_IBKR is False (the current default -- see
# below). None means every UID runs against flat CAPITAL with an empty
# simulated book, same as a backtest's day one -- enough to prove
# signal timing and correctness, not real position sizing.
ACCOUNT_STATE_PROVIDER: AccountStateProvider | None = None

# Flip to True once IB Gateway/TWS is actually running and reachable --
# see live/execution/ibkr_client.py's module docstring for what that
# requires (it's infrastructure, not code: the gateway process itself,
# 2FA, market data subscriptions). Until then this must stay False; no
# part of this codebase should ever attempt a real connection on its
# own. When True, IBKRClient replaces ACCOUNT_STATE_PROVIDER above for
# every UID in this run.
USE_REAL_IBKR = False
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497  # TWS paper port; IB Gateway paper is 4002 -- confirm against your setup
IBKR_CLIENT_ID = 1

LOG_PATH = PROJECT_ROOT / "live" / "logs" / "daily_check.log"


def configure_logging(log_path: Path = LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _resolve_strategy_class(uid: str) -> type:
    strategy_name = uid.split("__", 1)[0].strip().lower()
    if strategy_name not in STRATEGY_CLASSES:
        raise ValueError(f"Unknown strategy in UID: {strategy_name}")
    return STRATEGY_CLASSES[strategy_name]


def build_live_strategy(uid: str, signal_book: SignalBook) -> Any:
    strategy_class = _resolve_strategy_class(uid)
    strategy_name = uid.split("__", 1)[0].strip().lower()

    kwargs: dict[str, Any] = dict(
        uid=uid,
        capital=CAPITAL,
        db_path=DB_PATH,
        timeframe=TIMEFRAME,
        allow_fractional_shares=True,
    )
    if strategy_name in MOMENTUM_STRATEGIES:
        kwargs["universe_root"] = UNIVERSE_ROOT

    return make_live_strategy(strategy_class, signal_book=signal_book, **kwargs)


def check_one_uid(
    uid: str,
    *,
    as_of: pd.Timestamp,
    signal_book: SignalBook,
    run_log: RunLog,
    kill_switch: KillSwitch,
    account_state_provider: AccountStateProvider | None,
    notifier: TelegramNotifier | None = None,
) -> bool:
    """
    Run one UID's daily check. Returns True on success (including a
    clean no-op or a kill-switch skip), False if it errored -- never
    raises, since one UID's failure must not stop the others from being
    checked.
    """
    logger = logging.getLogger(__name__)

    # "strategy_engine" is the process-level switch; is_killed(uid) also
    # separately covers the system-wide switch (see kill_switch.py), so
    # together these honor the full per-UID / per-process / system
    # hierarchy without KillSwitch itself needing to know that
    # "strategy_engine" is special.
    if kill_switch.is_killed("strategy_engine") or kill_switch.is_killed(uid):
        logger.warning("SKIPPED (killed): %s", uid)
        run_log.record(uid, as_of=as_of, decision="KILLED")
        return True

    try:
        strategy = build_live_strategy(uid, signal_book)
    except Exception as exc:
        # Failure before run_live_day() is even reached (e.g. a
        # malformed/unknown UID) -- record it here, since run_live_day()
        # isn't the one that will catch it.
        logger.exception("FAILED to build strategy: %s", uid)
        run_log.record(uid, as_of=as_of, decision="ERROR", detail={"error": str(exc)})
        if notifier is not None:
            notifier.notify_error(uid, str(exc))
        return False

    try:
        run_live_day(
            strategy,
            as_of=as_of,
            account_state_provider=account_state_provider,
            run_log=run_log,
        )
    except Exception as exc:
        # run_live_day() already wrote its own ERROR entry to run_log
        # before re-raising -- nothing more to record here.
        logger.exception("FAILED: %s", uid)
        if notifier is not None:
            notifier.notify_error(uid, str(exc))
        return False

    logger.info("OK: %s", uid)
    return True


def main() -> int:
    configure_logging(LOG_PATH)
    logger = logging.getLogger(__name__)

    if not UIDS:
        logger.error("No UIDs configured -- nothing to check.")
        return 1

    as_of = pd.Timestamp.now().normalize()
    logger.info("=== Daily signal check starting, as_of=%s ===", as_of.date())

    ibkr_client: IBKRClient | None = None
    account_state_provider = ACCOUNT_STATE_PROVIDER

    if USE_REAL_IBKR:
        ibkr_client = IBKRClient(
            host=IBKR_HOST, port=IBKR_PORT, client_id=IBKR_CLIENT_ID, readonly=True
        )
        ibkr_client.ensure_connected()
        account_state_provider = ibkr_client
        logger.info("Connected to IBKR at %s:%d", IBKR_HOST, IBKR_PORT)
    elif account_state_provider is None:
        logger.warning(
            "No account_state_provider configured -- running with flat "
            "capital (%.2f), not real account state. Fine for proving "
            "signal timing, not for real position sizing.",
            CAPITAL,
        )

    try:
        with SignalBook(db_path=STATE_DB_PATH) as signal_book, RunLog(
            db_path=STATE_DB_PATH
        ) as run_log, KillSwitch(db_path=STATE_DB_PATH) as kill_switch:
            notifier = notifier_from_env(kill_switch)
            results = {
                uid: check_one_uid(
                    uid,
                    as_of=as_of,
                    signal_book=signal_book,
                    run_log=run_log,
                    kill_switch=kill_switch,
                    account_state_provider=account_state_provider,
                    notifier=notifier,
                )
                for uid in UIDS
            }
    finally:
        if ibkr_client is not None:
            ibkr_client.disconnect()

    failed = [uid for uid, ok in results.items() if not ok]
    logger.info(
        "=== Daily signal check complete: %d/%d OK ===",
        len(UIDS) - len(failed),
        len(UIDS),
    )

    if failed:
        logger.error("Failed UIDs: %s", failed)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
