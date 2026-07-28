from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from live.execution.signal_book import DEFAULT_SIGNAL_BOOK_PATH

SYSTEM_SWITCH = "system"

# Pre-seeded so they always show up in the dashboard, even before anyone
# has toggled them. Matches the granularity the reference system uses:
# a system-wide kill, a few named long-running processes, and (added
# freely, not pre-seeded) per-UID switches. "alerting" mutes
# TelegramNotifier (see live/execution/alerting.py) -- listed here like
# every other process so it gets the same dashboard kill/unkill button
# for free, no separate UI needed.
DEFAULT_SWITCHES = (
    SYSTEM_SWITCH,
    "market_data",
    "strategy_engine",
    "oms_trader",
    "alerting",
)


class KillSwitch:
    """
    Named on/off flags every long-running process is expected to check
    independently and stop/pause accordingly -- nothing in this class
    enforces that on its own, it's just the shared, dashboard-editable
    source of truth (same pattern as SignalBook/RunLog: a small SQLite
    table other processes poll).

    `is_killed(name)` also honors the reserved "system" switch: killing
    "system" kills everything, matching the reference system's "system
    stops everything" hierarchy (per-UID pauses just that UID's signals;
    a named process switch stops that process; "system" stops all of it).

    Shares its default SQLite file with SignalBook/RunLog.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(
            db_path if db_path is not None else DEFAULT_SIGNAL_BOOK_PATH
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False: Streamlit caches this connection
        # across script reruns via st.cache_resource, and a rerun (e.g.
        # after a button click) can execute on a different thread than
        # the one that created it. Actual thread-safety for writes still
        # comes from WAL mode + the `with self.connection:` transaction
        # blocks below, not from this flag -- this only disables
        # Python's same-thread check, not SQLite's own locking.
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()
        self._seed_defaults()

    def _create_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS kill_switch (
                name TEXT PRIMARY KEY,
                killed INTEGER NOT NULL DEFAULT 0,
                reason TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_by TEXT
            )
            """
        )
        self.connection.commit()

    def _seed_defaults(self) -> None:
        with self.connection:
            for name in DEFAULT_SWITCHES:
                self.connection.execute(
                    "INSERT OR IGNORE INTO kill_switch (name, killed) VALUES (?, 0)",
                    (name,),
                )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "KillSwitch":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ========================================================
    # Writing
    # ========================================================

    def set_killed(
        self,
        name: str,
        killed: bool,
        reason: str = "",
        updated_by: str = "dashboard",
    ) -> None:
        """Toggle a named switch on or off. Creates the row if it's new
        (e.g. a per-UID switch that's never been touched before)."""
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO kill_switch (name, killed, reason, updated_at, updated_by)
                VALUES (?, ?, ?, datetime('now'), ?)
                ON CONFLICT(name) DO UPDATE SET
                    killed = excluded.killed,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by
                """,
                (name, int(bool(killed)), reason, updated_by),
            )

    # ========================================================
    # Reading
    # ========================================================

    def is_killed(self, name: str) -> bool:
        """True if this specific switch is on, or the system-wide switch is."""
        if name != SYSTEM_SWITCH:
            row = self.connection.execute(
                "SELECT killed FROM kill_switch WHERE name = ?", (SYSTEM_SWITCH,)
            ).fetchone()
            if row is not None and bool(row[0]):
                return True

        row = self.connection.execute(
            "SELECT killed FROM kill_switch WHERE name = ?", (name,)
        ).fetchone()
        return row is not None and bool(row[0])

    def get_all(self) -> pd.DataFrame:
        """Every known switch (pre-seeded defaults plus any per-UID ones
        that have been set), alphabetically."""
        return pd.read_sql_query(
            "SELECT * FROM kill_switch ORDER BY name", self.connection
        )
