"""One-off materializer: resample futures_data.duckdb's 1m bars up to 1h,
storing the result in a dedicated duckdb + parquet store (not mixed into
futures_data.duckdb itself).

Source: duckdb/futures_data.duckdb        (bars where timeframe='1m')
Target: duckdb/futures_data_1h.duckdb      + futures_data_1h/parquet/1h/<SYMBOL>.parquet

Uses resample_ohlcv() (data/resample.py) -- the same resampling function
BaseFuturesStrategy uses for on-the-fly resampling -- so the materialized
1h bars are produced by the exact same logic, just computed once up front
instead of on every backtest run. Writes go through YahooDownloader's
existing save_parquet/upsert_duckdb code path, same as every other symbol
in this codebase.

This is a one-off run: re-run it manually whenever futures_data.duckdb's
1m data is refreshed with new bars. Other timeframes (2h, 3h, 6h, ...)
are left to on-the-fly resampling (source_timeframe param on
BaseFuturesStrategy) rather than also being materialized here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

ATLAS_ROOT = Path(__file__).resolve().parents[1]
if str(ATLAS_ROOT) not in sys.path:
    sys.path.insert(0, str(ATLAS_ROOT))

from data.resample import resample_ohlcv  # noqa: E402
from data.yahoo_downloader import BAR_COLUMNS, YahooDownloader  # noqa: E402

SOURCE_DB_PATH = ATLAS_ROOT / "duckdb" / "futures_data.duckdb"
SOURCE_TIMEFRAME = "1m"

TARGET_DB_PATH = ATLAS_ROOT / "duckdb" / "futures_data_1h.duckdb"
TARGET_PARQUET_ROOT = ATLAS_ROOT / "futures_data_1h" / "parquet"
TARGET_TIMEFRAME = "1h"

SYMBOLS = ["MES=F", "MNQ=F", "MGC=F", "MCL=F", "MNG=F", "MHG=F", "SIL=F"]


def load_source_bars(connection: duckdb.DuckDBPyConnection, symbol: str) -> pd.DataFrame:
    frame = connection.execute(
        """
        SELECT timestamp, open, high, low, close, volume
        FROM bars
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp
        """,
        [symbol, SOURCE_TIMEFRAME],
    ).fetchdf()
    return frame.set_index("timestamp")


def main() -> None:
    source_connection = duckdb.connect(str(SOURCE_DB_PATH), read_only=True)
    downloader = YahooDownloader(duckdb_path=TARGET_DB_PATH, parquet_root=TARGET_PARQUET_ROOT)

    summary: list[tuple[str, int, pd.Timestamp | None, pd.Timestamp | None]] = []

    try:
        for symbol in SYMBOLS:
            raw = load_source_bars(source_connection, symbol)

            if raw.empty:
                summary.append((symbol, 0, None, None))
                continue

            resampled = resample_ohlcv(
                raw,
                TARGET_TIMEFRAME,
                source_timeframe=SOURCE_TIMEFRAME,
            )

            out = resampled.reset_index()
            out["symbol"] = symbol
            out["timeframe"] = TARGET_TIMEFRAME
            out = out[BAR_COLUMNS]

            downloader.save_parquet(out, symbol=symbol, interval=TARGET_TIMEFRAME)
            rows = downloader.upsert_duckdb(out)

            summary.append((symbol, rows, out["timestamp"].min(), out["timestamp"].max()))
    finally:
        source_connection.close()
        downloader.close()

    print(f"{'symbol':<8} {'rows':>10}  window")
    for symbol, rows, start, end in summary:
        if rows:
            print(f"{symbol:<8} {rows:>10}  {start} -> {end}")
        else:
            print(f"{symbol:<8} {'0':>10}  (no source data)")


if __name__ == "__main__":
    main()
