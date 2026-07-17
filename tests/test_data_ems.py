from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import duckdb
import pandas as pd
import pytest

from data.interface import DataInterface
from ems.ems import EMS, TRADEBOOK_COLUMNS


@pytest.fixture()
def market_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "market_data.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute(
        """
        CREATE TABLE bars (
            timestamp TIMESTAMP,
            symbol VARCHAR,
            timeframe VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
        """
    )

    rows = []
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = {
        "AAA": [10.0, 11.0, 12.0, 13.0],
        "BBB": [20.0, 20.0, 20.0, 20.0],
    }
    for symbol, closes in prices.items():
        for timestamp, close in zip(dates, closes):
            rows.append(
                (
                    timestamp,
                    symbol,
                    "1d",
                    close - 0.5,
                    close + 1.0,
                    close - 1.0,
                    close,
                    1000.0,
                )
            )

    connection.executemany(
        "INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.close()
    return db_path


def test_data_interface_latest_and_no_lookahead(market_db: Path) -> None:
    data = DataInterface(duckdb_path=market_db)

    assert data.get_close("AAA") == 13.0

    data.set_current_timestamp("2024-01-02")
    assert data.get_close("AAA") == 11.0
    assert data.get_open("AAA") == 10.5

    history = data.history("AAA", bars=10)
    assert history.tolist() == [10.0, 11.0]
    assert history.index.max() == pd.Timestamp("2024-01-02")

    frame = data.history_frame(["AAA", "BBB"], bars=2)
    assert list(frame.columns) == ["AAA", "BBB"]
    assert len(frame) == 2

    data.close_connection()


def test_timestamps_symbol_filter(market_db: Path) -> None:
    data = DataInterface(duckdb_path=market_db)
    dates = data.timestamps(
        symbols=["AAA"],
        start="2024-01-02",
        end="2024-01-03",
    )
    assert dates.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    data.close_connection()


def test_place_trade_and_tradebook(market_db: Path) -> None:
    ems = EMS(
        uid="test_uid",
        capital=1000.0,
        db_path=market_db,
    )
    ems.set_current_timestamp("2024-01-01")

    row = ems.place_trade(
        "AAA",
        10,
        reason="ENTRY",
        notes={"score": 1.25},
    )

    assert row is not None
    assert ems.get_position("AAA") == 10.0
    assert ems.cash == 900.0
    assert row["action"] == "BUY"
    assert json.loads(row["notes"]) == {"score": 1.25}

    tradebook = ems.get_tradebook()
    assert tradebook.columns.tolist() == TRADEBOOK_COLUMNS
    assert len(tradebook) == 1
    ems.close_connection()


def test_no_oversell_and_no_overspend(market_db: Path) -> None:
    ems = EMS(
        uid="test_uid",
        capital=100.0,
        db_path=market_db,
        allow_fractional_shares=False,
    )
    ems.set_current_timestamp("2024-01-01")

    ems.place_trade("AAA", 1000)
    assert ems.get_position("AAA") == 10.0
    assert ems.cash == 0.0

    ems.place_trade("AAA", -1000)
    assert ems.get_position("AAA") == 0.0
    assert ems.cash == 100.0
    ems.close_connection()


def test_rebalance_reduces_before_buying(market_db: Path) -> None:
    ems = EMS(
        uid="rebalance_uid",
        capital=1000.0,
        db_path=market_db,
        allow_fractional_shares=True,
    )
    ems.set_current_timestamp("2024-01-01")

    ems.rebalance_to_weights({"AAA": 0.8, "BBB": 0.2})
    assert pytest.approx(ems.portfolio_value(), rel=1e-12) == 1000.0

    ems.set_current_timestamp("2024-01-02")
    ems.rebalance_to_weights({"AAA": 0.2, "BBB": 0.8})

    tradebook = ems.get_tradebook()
    second_rebalance = tradebook[tradebook["timestamp"] == pd.Timestamp("2024-01-02")]
    assert not second_rebalance.empty
    assert second_rebalance.iloc[0]["action"] == "SELL"
    assert ems.cash >= -1e-9
    ems.close_connection()


def test_run_records_equity_and_resets(market_db: Path) -> None:
    class BuyOnceStrategy(EMS):
        strategy_name = "buy_once"

        def on_day_close(self) -> None:
            if self.get_position("AAA") == 0:
                self.target_percent("AAA", 1.0, reason="BUY_ONCE")

    strategy = BuyOnceStrategy(
        uid="buy_once_AAA",
        capital=1000.0,
        db_path=market_db,
    )

    result = strategy.run(
        symbols=["AAA"],
        start="2024-01-01",
        end="2024-01-04",
    )
    assert len(result["equity"]) == 4
    assert len(result["tradebook"]) == 1
    assert result["equity"].iloc[-1]["equity"] == pytest.approx(1300.0)

    second = strategy.run(
        symbols=["AAA"],
        start="2024-01-01",
        end="2024-01-02",
        reset=True,
    )
    assert len(second["equity"]) == 2
    assert len(second["tradebook"]) == 1
    strategy.close_connection()


def test_save_results(market_db: Path, tmp_path: Path) -> None:
    ems = EMS(
        uid="save_uid",
        capital=1000.0,
        db_path=market_db,
    )
    ems.set_current_timestamp("2024-01-01")
    ems.target_percent("AAA", 0.5)
    ems._record_equity()

    output_dir = tmp_path / "results"
    ems.save_results(output_dir)

    assert (output_dir / "tradebook.csv").exists()
    assert (output_dir / "equity.csv").exists()
    assert (output_dir / "positions.csv").exists()
    ems.close_connection()
