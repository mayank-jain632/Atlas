"""
Atlas -- Data page.

Coverage and sanity view across the local DuckDB files every backtest
and the live pipeline reads from: which files exist, what symbols/
timeframes/date ranges they hold, and a gap check per symbol. Read-only,
and deliberately opens/closes a fresh connection per query rather than
holding one open (same discipline the reference system's own dashboard
uses -- their documented fix for a historical dashboard-vs-cron lock
contention issue). These DuckDB files get written to by other processes
(grid sweeps, data downloads) that this page has no business blocking.
"""

from __future__ import annotations

from pathlib import Path
import sys

_DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

PROJECT_ROOT = _DASHBOARD_DIR.parent
DUCKDB_DIR = PROJECT_ROOT / "duckdb"

import duckdb
import pandas as pd
import streamlit as st

st.title("Data")
st.caption(f"Local DuckDB files under {DUCKDB_DIR}")

if st.sidebar.button("Refresh"):
    st.rerun()


def _discover_db_files() -> list[Path]:
    if not DUCKDB_DIR.exists():
        return []
    return sorted(DUCKDB_DIR.glob("*.duckdb"))


def _table_names(db_path: Path) -> list[str]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        con.close()


def _bars_overview(db_path: Path) -> pd.DataFrame | None:
    """One row per timeframe in this file's bars table, or None if
    there's no bars table at all."""
    if "bars" not in _table_names(db_path):
        return None

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            """
            SELECT
                timeframe,
                COUNT(DISTINCT symbol) AS symbols,
                COUNT(*) AS rows,
                MIN(timestamp) AS earliest,
                MAX(timestamp) AS latest
            FROM bars
            GROUP BY timeframe
            ORDER BY timeframe
            """
        ).fetchdf()
    finally:
        con.close()


db_files = _discover_db_files()

if not db_files:
    st.info(f"No .duckdb files found under {DUCKDB_DIR}.")
    st.stop()


# ============================================================
# Overview -- every DB file at a glance
# ============================================================

st.subheader("Overview")

now = pd.Timestamp.now()
overview_rows: list[dict] = []

for db_path in db_files:
    size_mb = db_path.stat().st_size / (1024 * 1024)
    overview = _bars_overview(db_path)

    if overview is None or overview.empty:
        overview_rows.append(
            {
                "file": db_path.name,
                "size_mb": round(size_mb, 1),
                "timeframe": "—",
                "symbols": 0,
                "rows": 0,
                "earliest": None,
                "latest": None,
                "days_since_latest": None,
            }
        )
        continue

    for _, row in overview.iterrows():
        latest = pd.Timestamp(row["latest"])
        overview_rows.append(
            {
                "file": db_path.name,
                "size_mb": round(size_mb, 1),
                "timeframe": row["timeframe"],
                "symbols": int(row["symbols"]),
                "rows": int(row["rows"]),
                "earliest": row["earliest"],
                "latest": row["latest"],
                "days_since_latest": round(
                    (now - latest).total_seconds() / 86400, 1
                ),
            }
        )

overview_df = pd.DataFrame(overview_rows)
st.dataframe(
    overview_df,
    width="stretch",
    hide_index=True,
    column_config={
        "size_mb": st.column_config.NumberColumn("size (MB)", format="%.1f"),
        "days_since_latest": st.column_config.NumberColumn(
            "days since latest bar", format="%.1f"
        ),
    },
)


# ============================================================
# Symbol-level coverage + gap check
# ============================================================

st.subheader("Symbol coverage & gap check")

file_names = [p.name for p in db_files]
selected_file = st.selectbox("Database", file_names)
selected_path = db_files[file_names.index(selected_file)]

if "bars" not in _table_names(selected_path):
    st.write("No `bars` table in this file.")
    st.stop()

connection = duckdb.connect(str(selected_path), read_only=True)
try:
    timeframes = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT timeframe FROM bars ORDER BY timeframe"
        ).fetchall()
    ]
    selected_timeframe = st.selectbox("Timeframe", timeframes)

    per_symbol = connection.execute(
        """
        SELECT
            symbol,
            COUNT(*) AS rows,
            MIN(timestamp) AS earliest,
            MAX(timestamp) AS latest
        FROM bars
        WHERE timeframe = ?
        GROUP BY symbol
        ORDER BY symbol
        """,
        [selected_timeframe],
    ).fetchdf()

    st.dataframe(per_symbol, width="stretch", hide_index=True)

    st.markdown(
        "**Gap check** — flags any gap between consecutive bars more "
        "than `gap multiple` × that symbol's own median gap. Catches "
        "missing data beyond the normal weekend/overnight spacing "
        "without needing a trading-calendar library."
    )

    symbols = per_symbol["symbol"].tolist()

    gap_col, multiple_col = st.columns([2, 1])
    gap_symbol = gap_col.selectbox("Symbol", symbols) if symbols else None
    # Default 4.5x: for daily data (median gap = 1 day), that's a 4.5-day
    # threshold -- above every single-holiday-adjacent-to-a-weekend gap
    # (routinely 4 days: Fri close -> Tue open) but still catches
    # genuinely unusual multi-day closures (verified against this
    # repo's own market_data.duckdb: a lower default buried real gaps
    # like the 2001-09-10 -> 2001-09-17 9/11 closure and the 2012-10-26
    # -> 2012-10-31 Hurricane Sandy closure under ~230 routine
    # single-holiday weekends).
    gap_multiple = multiple_col.slider("Gap multiple", 2.0, 10.0, 4.5)

    if gap_symbol:
        timestamps = connection.execute(
            """
            SELECT timestamp FROM bars
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp
            """,
            [gap_symbol, selected_timeframe],
        ).fetchdf()["timestamp"]

        if len(timestamps) < 3:
            st.write("Not enough bars to check for gaps.")
        else:
            diffs = timestamps.diff().dropna()
            median_gap = diffs.median()
            threshold = median_gap * gap_multiple
            flagged = diffs[diffs > threshold]

            if flagged.empty:
                st.success(
                    f"No gaps larger than {gap_multiple:g}x the median "
                    f"gap ({median_gap})."
                )
            else:
                gap_rows = [
                    {
                        "gap_start": timestamps.loc[idx - 1],
                        "gap_end": timestamps.loc[idx],
                        "gap_size": diffs.loc[idx],
                    }
                    for idx in flagged.index
                ]
                st.warning(
                    f"{len(flagged)} gap(s) larger than {gap_multiple:g}x "
                    f"the median gap ({median_gap}):"
                )
                st.dataframe(
                    pd.DataFrame(gap_rows), width="stretch", hide_index=True
                )
finally:
    connection.close()
