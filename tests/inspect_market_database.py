from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATABASE_PATH = (
    PROJECT_ROOT
    / "duckdb"
    / "market_data.duckdb"
)


def main() -> None:
    print()
    print("=" * 70)
    print("ATLAS MARKET DATABASE INSPECTION")
    print("=" * 70)
    print(f"Database: {DATABASE_PATH}")
    print(f"Exists:   {DATABASE_PATH.exists()}")

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:
        # ----------------------------------------------------
        # Tables
        # ----------------------------------------------------
        tables = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchdf()

        print()
        print("TABLES")
        print("-" * 70)
        print(tables.to_string(index=False))

        if "bars" not in tables["table_name"].tolist():
            raise RuntimeError(
                "The database does not contain a bars table."
            )

        # ----------------------------------------------------
        # Schema
        # ----------------------------------------------------
        schema = connection.execute(
            """
            DESCRIBE bars
            """
        ).fetchdf()

        print()
        print("BARS SCHEMA")
        print("-" * 70)
        print(schema.to_string(index=False))

        # ----------------------------------------------------
        # Total row count
        # ----------------------------------------------------
        total_rows = connection.execute(
            """
            SELECT COUNT(*)
            FROM bars
            """
        ).fetchone()[0]

        print()
        print("ROW COUNT")
        print("-" * 70)
        print(f"Total bars rows: {total_rows:,}")

        if total_rows == 0:
            print()
            print(
                "PROBLEM: The bars table exists but is empty."
            )
            return

        # ----------------------------------------------------
        # Raw sample
        # ----------------------------------------------------
        sample = connection.execute(
            """
            SELECT *
            FROM bars
            LIMIT 20
            """
        ).fetchdf()

        print()
        print("RAW SAMPLE")
        print("-" * 70)
        print(sample.to_string(index=False))

        # ----------------------------------------------------
        # Timeframes
        # ----------------------------------------------------
        timeframes = connection.execute(
            """
            SELECT
                timeframe,
                COUNT(*) AS rows,
                COUNT(DISTINCT symbol) AS symbols,
                MIN(timestamp) AS first_timestamp,
                MAX(timestamp) AS last_timestamp
            FROM bars
            GROUP BY timeframe
            ORDER BY rows DESC
            """
        ).fetchdf()

        print()
        print("TIMEFRAMES")
        print("-" * 70)
        print(timeframes.to_string(index=False))

        # ----------------------------------------------------
        # Symbols
        # ----------------------------------------------------
        symbols = connection.execute(
            """
            SELECT
                symbol,
                COUNT(*) AS rows,
                MIN(timestamp) AS first_timestamp,
                MAX(timestamp) AS last_timestamp
            FROM bars
            GROUP BY symbol
            ORDER BY symbol
            LIMIT 100
            """
        ).fetchdf()

        print()
        print("FIRST 100 DATABASE SYMBOLS")
        print("-" * 70)
        print(symbols.to_string(index=False))

        # ----------------------------------------------------
        # Specific known symbols
        # ----------------------------------------------------
        known = connection.execute(
            """
            SELECT
                symbol,
                timeframe,
                COUNT(*) AS rows,
                MIN(timestamp) AS first_timestamp,
                MAX(timestamp) AS last_timestamp
            FROM bars
            WHERE UPPER(TRIM(symbol)) IN (
                'AAPL',
                'MSFT',
                'NVDA',
                'SPY',
                'QQQ'
            )
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
            """
        ).fetchdf()

        print()
        print("KNOWN SYMBOL CHECK")
        print("-" * 70)

        if known.empty:
            print(
                "No exact AAPL/MSFT/NVDA/SPY/QQQ symbols found."
            )
        else:
            print(known.to_string(index=False))

        # ----------------------------------------------------
        # Search malformed symbols containing AAPL
        # ----------------------------------------------------
        malformed = connection.execute(
            """
            SELECT DISTINCT symbol
            FROM bars
            WHERE UPPER(CAST(symbol AS VARCHAR))
                  LIKE '%AAPL%'
            ORDER BY symbol
            LIMIT 50
            """
        ).fetchdf()

        print()
        print("SYMBOLS CONTAINING 'AAPL'")
        print("-" * 70)

        if malformed.empty:
            print("None found.")
        else:
            print(malformed.to_string(index=False))

        # ----------------------------------------------------
        # Null check
        # ----------------------------------------------------
        nulls = connection.execute(
            """
            SELECT
                SUM(CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END)
                    AS null_timestamp,
                SUM(CASE WHEN symbol IS NULL THEN 1 ELSE 0 END)
                    AS null_symbol,
                SUM(CASE WHEN timeframe IS NULL THEN 1 ELSE 0 END)
                    AS null_timeframe,
                SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END)
                    AS null_close
            FROM bars
            """
        ).fetchdf()

        print()
        print("NULL CHECK")
        print("-" * 70)
        print(nulls.to_string(index=False))

    finally:
        connection.close()


if __name__ == "__main__":
    main()