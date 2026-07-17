from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import duckdb
import pandas as pd


ATLAS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUCKDB_PATH = ATLAS_ROOT / "duckdb" / "market_data.duckdb"
ALLOWED_FIELDS = {"open", "high", "low", "close", "volume"}


class DataInterface:
    """DuckDB-backed market-data interface used by all Atlas strategies."""

    def __init__(
        self,
        duckdb_path: str | Path | None = None,
        timeframe: str = "1d",
        read_only: bool = True,
    ) -> None:
        self.duckdb_path = Path(
            duckdb_path if duckdb_path is not None else DEFAULT_DUCKDB_PATH
        ).expanduser().resolve()
        self.timeframe = str(timeframe)
        self.current_timestamp: Optional[pd.Timestamp] = None

        if not self.duckdb_path.exists():
            raise FileNotFoundError(
                f"DuckDB database not found: {self.duckdb_path}"
            )

        self.connection = duckdb.connect(
            str(self.duckdb_path),
            read_only=read_only,
        )
        self._validate_bars_table()

    # ========================================================
    # Setup / validation
    # ========================================================

    def _validate_bars_table(self) -> None:
        table_names = {
            row[0]
            for row in self.connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            ).fetchall()
        }
        if "bars" not in table_names:
            raise RuntimeError(
                f"'bars' table not found in {self.duckdb_path}"
            )

        columns = {
            row[0]
            for row in self.connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'main'
                  AND table_name = 'bars'
                """
            ).fetchall()
        }
        required = {
            "timestamp",
            "symbol",
            "timeframe",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        missing = required - columns
        if missing:
            raise RuntimeError(
                "The bars table is missing required columns: "
                + ", ".join(sorted(missing))
            )

    def close_connection(self) -> None:
        if getattr(self, "connection", None) is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "DataInterface":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_connection()

    # ========================================================
    # Event timestamp
    # ========================================================

    def set_current_timestamp(
        self,
        timestamp: str | pd.Timestamp,
    ) -> None:
        self.current_timestamp = pd.Timestamp(timestamp)

    def clear_current_timestamp(self) -> None:
        self.current_timestamp = None

    def get_current_timestamp(self) -> pd.Timestamp:
        if self.current_timestamp is None:
            raise RuntimeError("Current timestamp has not been set.")
        return self.current_timestamp

    def _require_timestamp(self) -> pd.Timestamp:
        """Compatibility alias used by EMS."""
        return self.get_current_timestamp()

    # ========================================================
    # Internal helpers
    # ========================================================

    @staticmethod
    def _validate_field(field: str) -> str:
        field = str(field).lower()
        if field not in ALLOWED_FIELDS:
            raise ValueError(
                f"Unsupported field: {field}. "
                f"Expected one of {sorted(ALLOWED_FIELDS)}."
            )
        return field

    def _timestamp_filter(self) -> tuple[str, list]:
        if self.current_timestamp is None:
            return "", []
        return " AND timestamp <= ?", [self.current_timestamp]

    def _get_latest_field(self, symbol: str, field: str) -> float:
        field = self._validate_field(field)
        timestamp_sql, timestamp_params = self._timestamp_filter()
        query = f"""
            SELECT {field}
            FROM bars
            WHERE symbol = ?
              AND timeframe = ?
              {timestamp_sql}
            ORDER BY timestamp DESC
            LIMIT 1
        """
        row = self.connection.execute(
            query,
            [symbol, self.timeframe, *timestamp_params],
        ).fetchone()
        if row is None or row[0] is None:
            raise KeyError(
                f"No {field} data found for {symbol} "
                f"at timeframe {self.timeframe}"
            )
        return float(row[0])

    # ========================================================
    # Latest OHLCV access
    # ========================================================

    def get_price(self, symbol: str, field: str = "close") -> float:
        return self._get_latest_field(symbol, field)

    def get_open(self, symbol: str) -> float:
        return self.get_price(symbol, "open")

    def get_high(self, symbol: str) -> float:
        return self.get_price(symbol, "high")

    def get_low(self, symbol: str) -> float:
        return self.get_price(symbol, "low")

    def get_close(self, symbol: str) -> float:
        return self.get_price(symbol, "close")

    def get_volume(self, symbol: str) -> float:
        return self.get_price(symbol, "volume")

    def get_bar(self, symbol: str) -> dict:
        timestamp_sql, timestamp_params = self._timestamp_filter()
        query = f"""
            SELECT timestamp, symbol, timeframe,
                   open, high, low, close, volume
            FROM bars
            WHERE symbol = ?
              AND timeframe = ?
              {timestamp_sql}
            ORDER BY timestamp DESC
            LIMIT 1
        """
        row = self.connection.execute(
            query,
            [symbol, self.timeframe, *timestamp_params],
        ).fetchone()
        if row is None:
            raise KeyError(f"No bar found for {symbol}")
        return {
            "timestamp": pd.Timestamp(row[0]),
            "symbol": row[1],
            "timeframe": row[2],
            "open": float(row[3]),
            "high": float(row[4]),
            "low": float(row[5]),
            "close": float(row[6]),
            "volume": float(row[7]) if row[7] is not None else None,
        }

    # ========================================================
    # Historical access
    # ========================================================

    def history(
        self,
        symbol: str,
        field: str = "close",
        bars: int = 100,
    ) -> pd.Series:
        field = self._validate_field(field)
        if int(bars) <= 0:
            raise ValueError("bars must be positive.")

        timestamp_sql, timestamp_params = self._timestamp_filter()
        query = f"""
            SELECT timestamp, {field}
            FROM bars
            WHERE symbol = ?
              AND timeframe = ?
              {timestamp_sql}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = self.connection.execute(
            query,
            [symbol, self.timeframe, *timestamp_params, int(bars)],
        ).fetchdf()
        if df.empty:
            return pd.Series(dtype=float, name=symbol)

        df = df.sort_values("timestamp")
        return pd.Series(
            df[field].astype(float).to_numpy(),
            index=pd.to_datetime(df["timestamp"]),
            name=symbol,
        )

    def history_frame(
        self,
        symbols: Iterable[str],
        field: str = "close",
        bars: int = 100,
    ) -> pd.DataFrame:
        unique_symbols = list(dict.fromkeys(symbols))
        series_list = [
            series
            for symbol in unique_symbols
            if not (
                series := self.history(symbol, field=field, bars=bars)
            ).empty
        ]
        if not series_list:
            return pd.DataFrame()
        return pd.concat(series_list, axis=1).sort_index()

    def get_all_history(
        self,
        symbols: Iterable[str],
        field: str = "close",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        field = self._validate_field(field)
        unique_symbols = list(dict.fromkeys(symbols))
        if not unique_symbols:
            return pd.DataFrame()

        placeholders = ",".join(["?"] * len(unique_symbols))
        filters = [
            f"symbol IN ({placeholders})",
            "timeframe = ?",
        ]
        params: list = [*unique_symbols, self.timeframe]

        if start is not None:
            filters.append("timestamp >= ?")
            params.append(pd.Timestamp(start))
        if end is not None:
            filters.append("timestamp <= ?")
            params.append(pd.Timestamp(end))
        if self.current_timestamp is not None:
            filters.append("timestamp <= ?")
            params.append(self.current_timestamp)

        df = self.connection.execute(
            f"""
            SELECT timestamp, symbol, {field}
            FROM bars
            WHERE {' AND '.join(filters)}
            ORDER BY timestamp, symbol
            """,
            params,
        ).fetchdf()
        if df.empty:
            return pd.DataFrame()
        return (
            df.pivot(index="timestamp", columns="symbol", values=field)
            .sort_index()
        )

    # ========================================================
    # Metadata / availability
    # ========================================================

    def has_history(self, symbol: str, bars: int) -> bool:
        if int(bars) <= 0:
            raise ValueError("bars must be positive.")
        timestamp_sql, timestamp_params = self._timestamp_filter()
        count = self.connection.execute(
            f"""
            SELECT COUNT(*)
            FROM bars
            WHERE symbol = ?
              AND timeframe = ?
              {timestamp_sql}
            """,
            [symbol, self.timeframe, *timestamp_params],
        ).fetchone()[0]
        return int(count) >= int(bars)

    def symbols(self) -> list[str]:
        rows = self.connection.execute(
            """
            SELECT DISTINCT symbol
            FROM bars
            WHERE timeframe = ?
            ORDER BY symbol
            """,
            [self.timeframe],
        ).fetchall()
        return [row[0] for row in rows]

    def timestamps(
        self,
        symbols: Iterable[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DatetimeIndex:
        filters = ["timeframe = ?"]
        params: list = [self.timeframe]

        if symbols is not None:
            unique_symbols = list(dict.fromkeys(symbols))
            if not unique_symbols:
                return pd.DatetimeIndex([])
            placeholders = ",".join(["?"] * len(unique_symbols))
            filters.append(f"symbol IN ({placeholders})")
            params.extend(unique_symbols)

        if start is not None:
            filters.append("timestamp >= ?")
            params.append(pd.Timestamp(start))
        if end is not None:
            filters.append("timestamp <= ?")
            params.append(pd.Timestamp(end))

        df = self.connection.execute(
            f"""
            SELECT DISTINCT timestamp
            FROM bars
            WHERE {' AND '.join(filters)}
            ORDER BY timestamp
            """,
            params,
        ).fetchdf()
        if df.empty:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
