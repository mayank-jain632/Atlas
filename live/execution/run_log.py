from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from live.execution.signal_book import DEFAULT_SIGNAL_BOOK_PATH


class RunLog:
    """
    Append-only record of every scheduled signal check, regardless of
    outcome.

    The signal book only gets a new row when a rebalance actually
    fires -- on an ordinary non-rebalance day there'd otherwise be
    nothing recorded at all, which is ambiguous: "correctly decided not
    to rebalance today" and "the scheduled check never ran" look
    identical from the signal book alone. This is what proves the
    check actually happened, on time, against known data.

    Shares its default SQLite file with SignalBook (DEFAULT_SIGNAL_BOOK_PATH)
    but owns its own table -- two small, focused classes, one physical
    file, since there's no need to keep two separate database files for
    something this small.
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
        # block below, not from this flag -- this only disables
        # Python's same-thread check, not SQLite's own locking.
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT NOT NULL,
                ran_at TEXT NOT NULL,
                as_of TEXT NOT NULL,
                data_as_of TEXT,
                decision TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_log_uid_id ON run_log (uid, id)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ========================================================
    # Writing
    # ========================================================

    def record(
        self,
        uid: str,
        as_of: str | pd.Timestamp,
        decision: str,
        *,
        ran_at: str | pd.Timestamp | None = None,
        data_as_of: str | pd.Timestamp | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """
        Record one scheduled check, regardless of outcome.

        `ran_at` is real wall-clock time the check executed (defaults to
        now); `as_of` is the trading-calendar date being evaluated;
        `data_as_of` is the timestamp of the market data actually used,
        when known -- proof of freshness once a live data source is
        wired in, not just that a date was passed in.
        """
        ran_at_text = (
            pd.Timestamp(ran_at) if ran_at is not None else pd.Timestamp.now()
        ).isoformat()

        data_as_of_text = (
            pd.Timestamp(data_as_of).isoformat() if data_as_of is not None else None
        )

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO run_log
                    (uid, ran_at, as_of, data_as_of, decision, detail)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    uid,
                    ran_at_text,
                    pd.Timestamp(as_of).isoformat(),
                    data_as_of_text,
                    decision,
                    json.dumps(detail or {}, sort_keys=True),
                ),
            )

    # ========================================================
    # Reading
    # ========================================================

    def list_uids(self) -> list[str]:
        """Every UID that has ever had a check recorded, alphabetically."""
        rows = self.connection.execute(
            "SELECT DISTINCT uid FROM run_log ORDER BY uid"
        ).fetchall()
        return [row[0] for row in rows]

    def get_history(
        self,
        uid: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Every recorded check for `uid`, most recent first."""
        query = "SELECT * FROM run_log WHERE uid = ? ORDER BY id DESC"
        params: list[Any] = [uid]

        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))

        return pd.read_sql_query(query, self.connection, params=params)

    def get_all_history(
        self,
        limit: int | None = None,
        decision: str | None = None,
    ) -> pd.DataFrame:
        """
        Every recorded check across every UID, most recent first --
        the cross-cutting view an Alerts/error-log page needs, as
        opposed to get_history()'s single-UID view.
        """
        query = "SELECT * FROM run_log"
        params: list[Any] = []
        clauses: list[str] = []

        if decision is not None:
            clauses.append("decision = ?")
            params.append(decision)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY id DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))

        return pd.read_sql_query(query, self.connection, params=params)

    def get_latest_per_uid(self) -> pd.DataFrame:
        """
        One row per UID -- its most recently recorded check -- for a
        system-health-style overview across every UID at once.
        """
        query = """
            SELECT *
            FROM run_log
            WHERE id IN (
                SELECT MAX(id) FROM run_log GROUP BY uid
            )
            ORDER BY uid
        """
        return pd.read_sql_query(query, self.connection)

    def get_latest(self, uid: str) -> dict[str, Any] | None:
        """The most recent recorded check for `uid`, or None if there isn't one."""
        history = self.get_history(uid, limit=1)
        if history.empty:
            return None
        return history.iloc[0].to_dict()
