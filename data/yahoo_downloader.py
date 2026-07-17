from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
import yfinance as yf


# ============================================================
# Atlas paths
# ============================================================

ATLAS_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DUCKDB_PATH = (
    ATLAS_ROOT
    / "duckdb"
    / "market_data.duckdb"
)

DEFAULT_PARQUET_ROOT = (
    ATLAS_ROOT
    / "market_data"
    / "parquet"
)


BAR_COLUMNS = [
    "timestamp",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


class YahooDownloader:
    """
    Download Yahoo Finance OHLCV data and store it in:

        1. One Parquet file per symbol
        2. Atlas DuckDB bars table

    Expected Atlas table:

        bars(
            timestamp,
            symbol,
            timeframe,
            open,
            high,
            low,
            close,
            volume
        )
    """

    def __init__(
        self,
        duckdb_path: str | Path | None = None,
        parquet_root: str | Path | None = None,
    ) -> None:
        self.duckdb_path = Path(
            duckdb_path
            if duckdb_path is not None
            else DEFAULT_DUCKDB_PATH
        ).expanduser().resolve()

        self.parquet_root = Path(
            parquet_root
            if parquet_root is not None
            else DEFAULT_PARQUET_ROOT
        ).expanduser().resolve()

        self.duckdb_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.parquet_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = duckdb.connect(
            str(self.duckdb_path)
        )

        self._create_bars_table()

    # ========================================================
    # Setup
    # ========================================================

    def _create_bars_table(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS bars (
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,

                PRIMARY KEY (
                    timestamp,
                    symbol,
                    timeframe
                )
            )
            """
        )

    def close(self) -> None:
        if getattr(self, "connection", None) is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "YahooDownloader":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()

    # ========================================================
    # Symbol helpers
    # ========================================================

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        symbol = str(symbol).strip()

        if not symbol:
            raise ValueError("Symbol cannot be empty.")

        return symbol

    @staticmethod
    def _safe_filename(symbol: str) -> str:
        """
        Convert Yahoo symbols into safe filenames.

        Examples:
            BRK-B -> BRK-B.parquet
            ES=F  -> ES_F.parquet
            ^VIX  -> _VIX.parquet
        """
        return (
            symbol
            .replace("=", "_")
            .replace("^", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

    # ========================================================
    # Yahoo normalization
    # ========================================================

    @staticmethod
    def _flatten_yahoo_columns(
        dataframe: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Normalize Yahoo's normal and MultiIndex column formats.

        Possible Yahoo output:

            Open
            High
            Low
            Close
            Volume

        or:

            ('Open', 'AAPL')
            ('High', 'AAPL')
            ...
        """
        df = dataframe.copy()

        if isinstance(df.columns, pd.MultiIndex):
            flattened_columns = []

            for column in df.columns:
                parts = [
                    str(part).strip()
                    for part in column
                    if str(part).strip()
                ]

                lower_parts = [
                    part.lower()
                    for part in parts
                ]

                field = None

                for candidate in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj close",
                    "volume",
                ]:
                    if candidate in lower_parts:
                        field = candidate
                        break

                if field is None:
                    # Usually the first level is the price field.
                    field = lower_parts[0]

                flattened_columns.append(field)

            df.columns = flattened_columns

        else:
            df.columns = [
                str(column).strip().lower()
                for column in df.columns
            ]

        # Remove any duplicate columns that can appear after flattening.
        df = df.loc[
            :,
            ~pd.Index(df.columns).duplicated(
                keep="first"
            ),
        ]

        return df

    @classmethod
    def normalize_yahoo_data(
        cls,
        dataframe: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> pd.DataFrame:
        """
        Convert a yfinance DataFrame into the Atlas bars schema.
        """
        if dataframe is None or dataframe.empty:
            return pd.DataFrame(
                columns=BAR_COLUMNS
            )

        symbol = cls._clean_symbol(symbol)

        df = cls._flatten_yahoo_columns(
            dataframe=dataframe,
            symbol=symbol,
        )

        df = df.reset_index()

        df.columns = [
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            for column in df.columns
        ]

        # Yahoo uses Date for daily data and Datetime for intraday.
        if "timestamp" not in df.columns:
            if "date" in df.columns:
                df = df.rename(
                    columns={
                        "date": "timestamp",
                    }
                )

            elif "datetime" in df.columns:
                df = df.rename(
                    columns={
                        "datetime": "timestamp",
                    }
                )

        if "timestamp" not in df.columns:
            raise ValueError(
                f"Yahoo output for {symbol} has no "
                "Date, Datetime, or timestamp column."
            )

        # auto_adjust=True generally means Close is already adjusted.
        # If only adj_close is available, use it as close.
        if (
            "close" not in df.columns
            and "adj_close" in df.columns
        ):
            df["close"] = df["adj_close"]

        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            if column not in df.columns:
                df[column] = pd.NA

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        # Remove timezone because DuckDB stores naive TIMESTAMP.
        if hasattr(
            df["timestamp"].dt,
            "tz",
        ):
            try:
                df["timestamp"] = (
                    df["timestamp"]
                    .dt.tz_localize(None)
                )
            except TypeError:
                pass

        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df["symbol"] = symbol
        df["timeframe"] = str(interval)

        df = df[BAR_COLUMNS]

        df = df.dropna(
            subset=[
                "timestamp",
                "symbol",
                "close",
            ]
        )

        df = df.sort_values(
            "timestamp"
        )

        df = df.drop_duplicates(
            subset=[
                "timestamp",
                "symbol",
                "timeframe",
            ],
            keep="last",
        )

        df = df.reset_index(
            drop=True
        )

        return df

    # ========================================================
    # Yahoo download
    # ========================================================

    def download_symbol(
        self,
        symbol: str,
        start: str | None = "1990-01-01",
        end: str | None = None,
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        symbol = self._clean_symbol(symbol)

        kwargs = {
            "tickers": symbol,
            "interval": interval,
            "auto_adjust": auto_adjust,
            "progress": False,
            "threads": False,

            # Request a flat frame where supported.
            # The normalization function still handles MultiIndex
            # output for compatibility with other versions.
            "multi_level_index": False,
        }

        if start is not None:
            kwargs["start"] = start

        if end is not None:
            kwargs["end"] = end

        raw = yf.download(
            **kwargs
        )

        return self.normalize_yahoo_data(
            dataframe=raw,
            symbol=symbol,
            interval=interval,
        )

    # ========================================================
    # Parquet storage
    # ========================================================

    def parquet_path(
        self,
        symbol: str,
        interval: str,
    ) -> Path:
        directory = (
            self.parquet_root
            / interval
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            self._safe_filename(symbol)
            + ".parquet"
        )

        return directory / filename

    def save_parquet(
        self,
        dataframe: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> Path:
        if dataframe.empty:
            raise ValueError(
                f"Cannot save empty data for {symbol}."
            )

        path = self.parquet_path(
            symbol=symbol,
            interval=interval,
        )

        if path.exists():
            existing = pd.read_parquet(
                path
            )

            existing = self._normalize_existing_parquet(
                dataframe=existing,
                symbol=symbol,
                interval=interval,
            )

            combined = pd.concat(
                [
                    existing,
                    dataframe,
                ],
                ignore_index=True,
            )

        else:
            combined = dataframe.copy()

        combined = combined.sort_values(
            "timestamp"
        )

        combined = combined.drop_duplicates(
            subset=[
                "timestamp",
                "symbol",
                "timeframe",
            ],
            keep="last",
        )

        combined = combined.reset_index(
            drop=True
        )

        combined.to_parquet(
            path,
            index=False,
        )

        return path

    @staticmethod
    def _normalize_existing_parquet(
        dataframe: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> pd.DataFrame:
        """
        Normalize Parquet files produced by older Atlas downloaders.
        """
        df = dataframe.copy()

        df.columns = [
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            for column in df.columns
        ]

        if "timestamp" not in df.columns:
            if "date" in df.columns:
                df = df.rename(
                    columns={
                        "date": "timestamp",
                    }
                )
            elif "datetime" in df.columns:
                df = df.rename(
                    columns={
                        "datetime": "timestamp",
                    }
                )

        if "symbol" not in df.columns:
            df["symbol"] = symbol

        if "timeframe" not in df.columns:
            df["timeframe"] = interval

        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            if column not in df.columns:
                df[column] = pd.NA

        missing = set(BAR_COLUMNS) - set(
            df.columns
        )

        if missing:
            raise ValueError(
                "Existing Parquet file has unsupported "
                f"columns. Missing: {sorted(missing)}"
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df["symbol"] = (
            df["symbol"]
            .astype(str)
            .str.strip()
        )

        df["timeframe"] = str(interval)

        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        return (
            df[BAR_COLUMNS]
            .dropna(
                subset=[
                    "timestamp",
                    "symbol",
                    "close",
                ]
            )
        )

    # ========================================================
    # DuckDB storage
    # ========================================================

    def upsert_duckdb(
        self,
        dataframe: pd.DataFrame,
    ) -> int:
        """
        Upsert normalized bars into DuckDB.

        Existing matching rows are deleted first, then the new
        normalized rows are inserted.
        """
        if dataframe.empty:
            return 0

        frame = dataframe[
            BAR_COLUMNS
        ].copy()

        self.connection.register(
            "incoming_bars",
            frame,
        )

        try:
            self.connection.execute(
                """
                DELETE FROM bars
                USING incoming_bars
                WHERE bars.timestamp =
                      incoming_bars.timestamp
                  AND bars.symbol =
                      incoming_bars.symbol
                  AND bars.timeframe =
                      incoming_bars.timeframe
                """
            )

            self.connection.execute(
                """
                INSERT INTO bars (
                    timestamp,
                    symbol,
                    timeframe,
                    open,
                    high,
                    low,
                    close,
                    volume
                )
                SELECT
                    timestamp,
                    symbol,
                    timeframe,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM incoming_bars
                """
            )

        finally:
            self.connection.unregister(
                "incoming_bars"
            )

        return len(frame)

    # ========================================================
    # Incremental date calculation
    # ========================================================

    def latest_database_timestamp(
        self,
        symbol: str,
        interval: str,
    ) -> pd.Timestamp | None:
        row = self.connection.execute(
            """
            SELECT MAX(timestamp)
            FROM bars
            WHERE symbol = ?
              AND timeframe = ?
            """,
            [
                symbol,
                interval,
            ],
        ).fetchone()

        if (
            row is None
            or row[0] is None
        ):
            return None

        return pd.Timestamp(
            row[0]
        )

    def determine_start_date(
        self,
        symbol: str,
        interval: str,
        requested_start: str | None,
        incremental: bool,
    ) -> str | None:
        if not incremental:
            return requested_start

        latest = self.latest_database_timestamp(
            symbol=symbol,
            interval=interval,
        )

        if latest is None:
            return requested_start

        # Include a small overlap so the most recent bar can
        # be corrected if Yahoo revises it.
        overlap_start = latest - pd.Timedelta(
            days=7
        )

        if requested_start is None:
            return overlap_start.strftime(
                "%Y-%m-%d"
            )

        requested = pd.Timestamp(
            requested_start
        )

        return max(
            requested,
            overlap_start,
        ).strftime("%Y-%m-%d")

    # ========================================================
    # Main ingestion
    # ========================================================

    def download(
        self,
        symbols: Iterable[str],
        start: str | None = "1990-01-01",
        end: str | None = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        incremental: bool = False,
        pause_seconds: float = 0.0,
    ) -> dict:
        unique_symbols = list(
            dict.fromkeys(
                self._clean_symbol(symbol)
                for symbol in symbols
            )
        )

        if not unique_symbols:
            raise ValueError(
                "No symbols were supplied."
            )

        successful = []
        failed = []
        empty = []

        total_downloaded_rows = 0

        print()
        print("=" * 70)
        print("ATLAS YAHOO DATA DOWNLOAD")
        print("=" * 70)
        print(
            f"DuckDB:       {self.duckdb_path}"
        )
        print(
            f"Parquet root: {self.parquet_root}"
        )
        print(
            f"Symbols:      {len(unique_symbols)}"
        )
        print(
            f"Interval:     {interval}"
        )
        print(
            f"Start:        {start}"
        )
        print(
            f"End:          {end}"
        )
        print(
            f"Auto adjust:  {auto_adjust}"
        )
        print(
            f"Incremental:  {incremental}"
        )
        print("=" * 70)

        for index, symbol in enumerate(
            unique_symbols,
            start=1,
        ):
            try:
                symbol_start = self.determine_start_date(
                    symbol=symbol,
                    interval=interval,
                    requested_start=start,
                    incremental=incremental,
                )

                print(
                    f"[{index:>3}/{len(unique_symbols)}] "
                    f"{symbol:<12} start={symbol_start}"
                )

                frame = self.download_symbol(
                    symbol=symbol,
                    start=symbol_start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                )

                if frame.empty:
                    print("    No data returned.")
                    empty.append(symbol)
                    continue

                parquet_path = self.save_parquet(
                    dataframe=frame,
                    symbol=symbol,
                    interval=interval,
                )

                inserted_rows = self.upsert_duckdb(
                    dataframe=frame,
                )

                total_downloaded_rows += inserted_rows
                successful.append(symbol)

                print(
                    f"    {inserted_rows:,} rows -> "
                    f"{parquet_path.name}"
                )

            except Exception as error:
                failed.append(
                    {
                        "symbol": symbol,
                        "error": str(error),
                    }
                )

                print(
                    f"    ERROR: {error}"
                )

            if pause_seconds > 0:
                time.sleep(
                    pause_seconds
                )

        summary = self.database_summary(
            interval=interval
        )

        print()
        print("=" * 70)
        print("DOWNLOAD COMPLETE")
        print("=" * 70)
        print(
            f"Successful symbols: {len(successful)}"
        )
        print(
            f"Empty symbols:      {len(empty)}"
        )
        print(
            f"Failed symbols:     {len(failed)}"
        )
        print(
            f"Rows processed:     "
            f"{total_downloaded_rows:,}"
        )
        print()
        print(summary.to_string(index=False))

        if failed:
            print()
            print("FAILURES")
            print("-" * 70)

            for failure in failed:
                print(
                    f"{failure['symbol']}: "
                    f"{failure['error']}"
                )

        return {
            "successful": successful,
            "empty": empty,
            "failed": failed,
            "rows_processed": total_downloaded_rows,
            "database_summary": summary,
        }

    # ========================================================
    # Diagnostics
    # ========================================================

    def database_summary(
        self,
        interval: str | None = None,
    ) -> pd.DataFrame:
        if interval is None:
            return self.connection.execute(
                """
                SELECT
                    timeframe,
                    COUNT(*) AS rows,
                    COUNT(DISTINCT symbol)
                        AS symbols,
                    MIN(timestamp)
                        AS first_timestamp,
                    MAX(timestamp)
                        AS last_timestamp
                FROM bars
                GROUP BY timeframe
                ORDER BY timeframe
                """
            ).fetchdf()

        return self.connection.execute(
            """
            SELECT
                timeframe,
                COUNT(*) AS rows,
                COUNT(DISTINCT symbol)
                    AS symbols,
                MIN(timestamp)
                    AS first_timestamp,
                MAX(timestamp)
                    AS last_timestamp
            FROM bars
            WHERE timeframe = ?
            GROUP BY timeframe
            """,
            [interval],
        ).fetchdf()