"""Golden-output regression test for the futures backtest engine.

Guards `BaseFuturesStrategy.run()` / `on_day_close()` against behavior
changes introduced by performance work (e.g. swapping per-bar DuckDB
queries for an in-memory history cache). The fixtures under
tests/fixtures/futures_regression/ were captured from the pre-optimization
implementation; this test reruns the identical UID/date range and asserts
the tradebook, equity, and signal outputs are unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.futures.factory import create_futures_strategy


DB_PATH = PROJECT_ROOT / "duckdb" / "futures_data_1h.duckdb"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "futures_regression"

UID = "stema__s=MES__st_period=10__st_mult=3__ema=50__atr=14__sl_atr=10"
START = "2021-01-01"
END = "2021-07-01"


def _run() -> dict[str, pd.DataFrame]:
    strategy = create_futures_strategy(
        uid=UID,
        capital=10000.0,
        db_path=DB_PATH,
        timeframe="1h",
        source_timeframe=None,
    )
    return strategy.run(
        symbols=strategy.required_symbols(),
        start=START,
        end=END,
    )


def _load_golden(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / f"{name}.csv")


def _round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    """Match the golden fixture's dtypes by going through the same CSV path."""
    from io import StringIO

    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer)


@pytest.fixture(scope="module")
def result() -> dict[str, pd.DataFrame]:
    if not DB_PATH.exists():
        pytest.skip(f"{DB_PATH} not available on this machine")
    return _run()


@pytest.mark.parametrize("name", ["tradebook", "equity", "signals"])
def test_futures_backtest_output_matches_golden_fixture(
    result: dict[str, pd.DataFrame],
    name: str,
) -> None:
    actual = _round_trip(result[name])
    expected = _load_golden(name)

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )
