"""
Daily scheduled market-data refresh.

Companion to live/run_daily_check.py, same shape: a script a cron job
(or later, a systemd timer on the server) invokes once a day,
unattended. Keeps duckdb/market_data.duckdb current by incrementally
re-downloading every symbol in the configured universes --
run_daily_check.py assumes this data is already fresh; nothing else in
this codebase enforces that.

Reuses the existing YahooDownloader/UniverseManager
(data/yahoo_downloader.py, data/universe.py) and the same universe list
data/download_market_data.py already defines for a full download --
this script differs only in always requesting incremental=True (only
what's new since the last stored bar per symbol, not a full
re-download) and in being wrapped with the same kill-switch/run-log/
exit-code discipline run_daily_check.py uses, so both scheduled jobs
are observable the same way on the dashboard.

Logs to the run log under the synthetic uid "market_data_refresh" --
not a strategy UID, but sharing the exact same RunLog/System Health
machinery on purpose rather than inventing a second observability
mechanism for a second kind of scheduled job. Checks the "market_data"
kill switch, which live/execution/kill_switch.py pre-seeded for
exactly this, unused until now.
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

from data.download_market_data import START_DATE, UNIVERSE_ROOT, UNIVERSES
from data.universe import UniverseManager
from data.yahoo_downloader import DEFAULT_DUCKDB_PATH, YahooDownloader
from live.execution.alerting import notifier_from_env
from live.execution.kill_switch import KillSwitch
from live.execution.run_log import RunLog
from live.execution.signal_book import DEFAULT_SIGNAL_BOOK_PATH

JOB_NAME = "market_data_refresh"

# signal_book / run_log / kill_switch all share this one SQLite file --
# same file run_daily_check.py uses, so both jobs show up together on
# the dashboard.
STATE_DB_PATH = DEFAULT_SIGNAL_BOOK_PATH

MARKET_DATA_DB_PATH = DEFAULT_DUCKDB_PATH
INTERVAL = "1d"
AUTO_ADJUST = True
PAUSE_SECONDS = 0.0

LOG_PATH = PROJECT_ROOT / "live" / "logs" / "data_refresh.log"


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


def load_universe_symbols() -> list[str]:
    """Every symbol across every configured universe, deduplicated --
    a pure local CSV read (config/universes/*.csv), no network call."""
    manager = UniverseManager(root=UNIVERSE_ROOT)
    symbols: list[str] = []
    for universe_name in UNIVERSES:
        symbols.extend(manager.load(universe_name))
    return list(dict.fromkeys(symbols))


def refresh_market_data(downloader: YahooDownloader, symbols: list[str]) -> dict[str, Any]:
    """The actual download call, factored out so it's fakeable in tests
    without hitting Yahoo Finance for real -- same discipline as
    ibkr_client.py's connect() being isolated behind a lazy import."""
    return downloader.download(
        symbols=symbols,
        start=START_DATE,
        end=None,
        interval=INTERVAL,
        auto_adjust=AUTO_ADJUST,
        incremental=True,
        pause_seconds=PAUSE_SECONDS,
    )


def _decision_for_summary(summary: dict[str, Any]) -> str:
    failed = summary["failed"]
    successful = summary["successful"]

    if failed and not successful:
        return "ERROR"
    if failed:
        return "PARTIAL_FAILURE"
    if summary["rows_processed"] > 0:
        return "REFRESHED"
    return "NO_NEW_DATA"


def main() -> int:
    configure_logging(LOG_PATH)
    logger = logging.getLogger(__name__)

    as_of = pd.Timestamp.now().normalize()
    logger.info("=== Market data refresh starting, as_of=%s ===", as_of.date())

    with KillSwitch(db_path=STATE_DB_PATH) as kill_switch, RunLog(
        db_path=STATE_DB_PATH
    ) as run_log:
        notifier = notifier_from_env(kill_switch)

        if kill_switch.is_killed("market_data"):
            logger.warning("SKIPPED (killed): %s", JOB_NAME)
            run_log.record(JOB_NAME, as_of=as_of, decision="KILLED")
            return 0

        try:
            symbols = load_universe_symbols()
            logger.info("Refreshing %d symbols across %s", len(symbols), UNIVERSES)

            with YahooDownloader(duckdb_path=MARKET_DATA_DB_PATH) as downloader:
                summary = refresh_market_data(downloader, symbols)
        except Exception as exc:
            logger.exception("FAILED: %s", JOB_NAME)
            run_log.record(
                JOB_NAME, as_of=as_of, decision="ERROR", detail={"error": str(exc)}
            )
            if notifier is not None:
                notifier.notify_error(JOB_NAME, str(exc))
            return 1

        decision = _decision_for_summary(summary)
        detail = {
            "symbols_requested": len(symbols),
            "successful": len(summary["successful"]),
            "empty": len(summary["empty"]),
            "failed": summary["failed"],
            "rows_processed": summary["rows_processed"],
        }
        run_log.record(JOB_NAME, as_of=as_of, decision=decision, detail=detail)

        if decision in ("ERROR", "PARTIAL_FAILURE") and notifier is not None:
            first_failure = summary["failed"][0] if summary["failed"] else "n/a"
            notifier.notify_error(
                JOB_NAME,
                f"{decision}: {len(summary['failed'])} of {len(symbols)} "
                f"symbol(s) failed. First failure: {first_failure}",
            )

        logger.info(
            "=== Market data refresh complete: decision=%s, %d rows, %d/%d symbols OK ===",
            decision,
            summary["rows_processed"],
            len(summary["successful"]),
            len(symbols),
        )

        return 1 if decision in ("ERROR", "PARTIAL_FAILURE") else 0


if __name__ == "__main__":
    sys.exit(main())
