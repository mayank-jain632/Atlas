from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SIGNAL_BOOK_PATH = (
    Path(__file__).resolve().parent / "state" / "signal_book.sqlite"
)


class SignalBook:
    """
    Append-only store of target positions ("what should this UID hold")
    -- the handoff point between signal generation and order execution.

    A strategy computes target quantities the same way it does in a
    backtest; a live adapter writes those targets here instead of
    booking a simulated fill. A separate OMS process reads the latest
    target per symbol and reconciles actual broker positions toward it.
    Neither side calls the other directly, so either can restart
    independently.

    Every write is a new row, never an UPDATE, so the full history of
    target changes for a UID is always recoverable. "Latest row per
    (uid, symbol)" is the current target. A rebalance that drops a
    symbol must still write an explicit target_quantity=0 row for it --
    there is no implicit "absence means zero."
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
            CREATE TABLE IF NOT EXISTS signal_book (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT NOT NULL,
                symbol TEXT NOT NULL,
                target_quantity REAL NOT NULL,
                as_of TEXT NOT NULL,
                reason TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_signal_book_uid_symbol "
            "ON signal_book (uid, symbol, id)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SignalBook":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ========================================================
    # Writing
    # ========================================================

    def write_targets(
        self,
        uid: str,
        targets: dict[str, float],
        as_of: str | pd.Timestamp,
        reason: str,
        notes_by_symbol: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        Record one target-quantity row per symbol for this rebalance.

        `targets` must include every symbol this rebalance decided
        about, with an explicit 0.0 for any symbol being exited --
        callers should not omit a symbol just because its new target is
        zero (omitting it would just leave its previous target as the
        latest row).
        """
        if not targets:
            return

        notes_by_symbol = notes_by_symbol or {}
        as_of_text = pd.Timestamp(as_of).isoformat()

        rows = [
            (
                uid,
                symbol,
                float(quantity),
                as_of_text,
                reason,
                json.dumps(notes_by_symbol.get(symbol, {}), sort_keys=True),
            )
            for symbol, quantity in targets.items()
        ]

        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO signal_book
                    (uid, symbol, target_quantity, as_of, reason, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    # ========================================================
    # Reading
    # ========================================================

    def list_uids(self) -> list[str]:
        """Every UID that has ever had targets written, alphabetically."""
        rows = self.connection.execute(
            "SELECT DISTINCT uid FROM signal_book ORDER BY uid"
        ).fetchall()
        return [row[0] for row in rows]

    def get_latest_targets_detail(self, uid: str) -> pd.DataFrame:
        """
        The full latest row per symbol for `uid` -- same "latest row
        per (uid, symbol)" semantics as get_latest_targets(), but
        returns every column (as_of, reason, notes) instead of just the
        quantity, for callers that need what a strategy attached to
        each target (e.g. target_weight, score, rank -- see
        create_trade_notes() in strategies/momentum/base.py).
        """
        query = """
            SELECT *
            FROM signal_book
            WHERE id IN (
                SELECT MAX(id)
                FROM signal_book
                WHERE uid = ?
                GROUP BY symbol
            )
            ORDER BY symbol
        """
        return pd.read_sql_query(query, self.connection, params=[uid])

    def get_latest_targets(self, uid: str) -> dict[str, float]:
        """
        The most recently written target_quantity per symbol for `uid`.

        Symbols whose latest recorded target is 0.0 are still included
        -- callers that only care about open targets should filter
        those out themselves.
        """
        query = """
            SELECT symbol, target_quantity
            FROM signal_book
            WHERE id IN (
                SELECT MAX(id)
                FROM signal_book
                WHERE uid = ?
                GROUP BY symbol
            )
        """
        rows = self.connection.execute(query, [uid]).fetchall()
        return {symbol: float(quantity) for symbol, quantity in rows}

    def get_history(
        self,
        uid: str,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Full audit trail of targets written for a UID (optionally one symbol)."""
        query = "SELECT * FROM signal_book WHERE uid = ?"
        params: list[Any] = [uid]

        if symbol is not None:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY id"

        return pd.read_sql_query(query, self.connection, params=params)
