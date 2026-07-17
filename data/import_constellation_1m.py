"""One-off importer: constellation's DataBento 1m cache -> atlas, split by asset class.

Source: /Users/mayankjain/Projects/constellation/data/historical/{futures_1m,etf_1m}/

Target: dedicated per-asset-class stores, NOT the existing market_data.duckdb (which
stays stocks-only / untouched by this script):
  - duckdb/futures_data.duckdb  + futures_data/parquet/1m/<SYMBOL>.parquet
  - duckdb/etf_data.duckdb      + etf_data/parquet/1m/<SYMBOL>.parquet

Each gets its own `bars` table, same shape as YahooDownloader._create_bars_table()
(timestamp, symbol, timeframe, open, high, low, close, volume) -- no `source` column
here since each file only ever holds one asset class from one source, so there's no
provenance ambiguity to resolve like there was in the original shared market_data.duckdb.

Symbol mapping:
  - Futures: constellation's DataBento continuous-contract symbols (e.g. MES1!) map
    to Yahoo-style micro-futures tickers (MES=F) -- same root, same contract size,
    different vendor's continuous-contract notation.
  - ETFs: identity mapping (SPY -> SPY, etc).

Uses YahooDownloader directly (same real save_parquet/upsert_duckdb code path as
every other symbol in this codebase), just pointed at the dedicated duckdb_path /
parquet_root for each asset class instead of the defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ATLAS_ROOT = Path(__file__).resolve().parents[1]
if str(ATLAS_ROOT) not in sys.path:
    sys.path.insert(0, str(ATLAS_ROOT))

from data.yahoo_downloader import BAR_COLUMNS, YahooDownloader  # noqa: E402

CONSTELLATION_ROOT = Path("/Users/mayankjain/Projects/constellation")
TIMEFRAME = "1m"

FUTURES_DUCKDB_PATH = ATLAS_ROOT / "duckdb" / "futures_data.duckdb"
FUTURES_PARQUET_ROOT = ATLAS_ROOT / "futures_data" / "parquet"

ETF_DUCKDB_PATH = ATLAS_ROOT / "duckdb" / "etf_data.duckdb"
ETF_PARQUET_ROOT = ATLAS_ROOT / "etf_data" / "parquet"

FUTURES_SYMBOL_MAP = {
    "MES1!": "MES=F",
    "MNQ1!": "MNQ=F",
    "MCL1!": "MCL=F",
    "MGC1!": "MGC=F",
    "MHG1!": "MHG=F",
    "MNG1!": "MNG=F",
    "SIL1!": "SIL=F",
}
ETF_SYMBOLS = ["SPY", "QQQ", "SMH", "GLD", "SLV"]


def load_futures(our_symbol: str) -> pd.DataFrame:
    root = CONSTELLATION_ROOT / "data" / "historical" / "futures_1m" / our_symbol
    frames = [pq.read_table(path).to_pandas() for path in sorted(root.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def load_etf(symbol: str) -> pd.DataFrame:
    root = CONSTELLATION_ROOT / "data" / "historical" / "etf_1m" / symbol
    frames = [pq.read_table(path).to_pandas() for path in sorted(root.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def normalize(raw: pd.DataFrame, atlas_symbol: str) -> pd.DataFrame:
    """Mirrors the tail end of YahooDownloader.normalize_yahoo_data, applied to
    already-clean constellation data instead of raw yfinance output."""

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    out["symbol"] = atlas_symbol
    out["timeframe"] = TIMEFRAME
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(raw[column], errors="coerce")

    out = out[BAR_COLUMNS]
    out = out.dropna(subset=["timestamp", "symbol", "close"])
    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp", "symbol", "timeframe"], keep="last")
    out = out.reset_index(drop=True)
    return out


def main() -> None:
    summary = []

    futures_downloader = YahooDownloader(duckdb_path=FUTURES_DUCKDB_PATH, parquet_root=FUTURES_PARQUET_ROOT)
    for our_symbol, atlas_symbol in FUTURES_SYMBOL_MAP.items():
        df = normalize(load_futures(our_symbol), atlas_symbol)
        futures_downloader.save_parquet(df, symbol=atlas_symbol, interval=TIMEFRAME)
        rows = futures_downloader.upsert_duckdb(df)
        summary.append((our_symbol, atlas_symbol, rows, df["timestamp"].min(), df["timestamp"].max()))
    futures_downloader.close()

    etf_downloader = YahooDownloader(duckdb_path=ETF_DUCKDB_PATH, parquet_root=ETF_PARQUET_ROOT)
    for symbol in ETF_SYMBOLS:
        df = normalize(load_etf(symbol), symbol)
        etf_downloader.save_parquet(df, symbol=symbol, interval=TIMEFRAME)
        rows = etf_downloader.upsert_duckdb(df)
        summary.append((symbol, symbol, rows, df["timestamp"].min(), df["timestamp"].max()))
    etf_downloader.close()

    print(f"{'source_symbol':<8} {'atlas_symbol':<8} {'rows':>10}  window")
    for our_symbol, atlas_symbol, rows, start, end in summary:
        print(f"{our_symbol:<8} {atlas_symbol:<8} {rows:>10}  {start} -> {end}")


if __name__ == "__main__":
    main()
