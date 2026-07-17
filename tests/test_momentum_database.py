from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


# ============================================================
# Make Atlas imports work when running pytest directly
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from data.interface import DataInterface
from data.universe import UniverseManager

from strategies.momentum.signals import compute_signal
from strategies.momentum.ranking import rank_stocks
from strategies.momentum.allocation import allocate_weights


# ============================================================
# Test configuration
# ============================================================

UNIVERSE_NAME = "sp500"
UNIVERSE_ROOT = PROJECT_ROOT / "config" / "universes"

TIMEFRAME = "1d"

LOOKBACK = 90
TOP_N = 10
MINIMUM_SCORE = 0.0

# Use a recent date that should exist in your database.
# When None, the latest timestamp in DuckDB is used.
TEST_TIMESTAMP = None


@pytest.fixture(scope="module")
def data() -> DataInterface:
    interface = DataInterface(
        timeframe=TIMEFRAME,
        read_only=True,
    )

    yield interface

    interface.close_connection()


@pytest.fixture(scope="module")
def universe() -> list[str]:
    manager = UniverseManager(
        root=UNIVERSE_ROOT
    )

    symbols = manager.load(UNIVERSE_NAME)

    symbols = [
        str(symbol).strip()
        for symbol in symbols
        if str(symbol).strip()
    ]

    symbols = list(dict.fromkeys(symbols))

    assert symbols, (
        f"Universe '{UNIVERSE_NAME}' is empty. "
        f"Checked: {UNIVERSE_ROOT}"
    )

    return symbols


def get_test_timestamp(
    data: DataInterface,
    available_symbols: list[str],
) -> pd.Timestamp:
    dates = data.timestamps(
        symbols=available_symbols,
    )

    assert len(dates) > 0, (
        "No timestamps were returned from DuckDB for "
        "the configured universe."
    )

    if TEST_TIMESTAMP is None:
        return pd.Timestamp(dates[-1])

    requested = pd.Timestamp(TEST_TIMESTAMP)

    eligible = dates[dates <= requested]

    assert len(eligible) > 0, (
        f"No database timestamp exists on or before "
        f"{requested.date()}."
    )

    return pd.Timestamp(eligible[-1])


def test_universe_has_database_overlap(
    data: DataInterface,
    universe: list[str],
) -> None:
    database_symbols = set(data.symbols())
    configured_symbols = set(universe)

    available = sorted(
        configured_symbols & database_symbols
    )

    missing = sorted(
        configured_symbols - database_symbols
    )

    print()
    print("=" * 70)
    print("UNIVERSE / DATABASE OVERLAP")
    print("=" * 70)
    print(
        f"Configured universe: {len(configured_symbols)}"
    )
    print(
        f"Available in DuckDB: {len(available)}"
    )
    print(
        f"Missing from DuckDB: {len(missing)}"
    )

    if missing:
        print("Missing symbols:")
        print(missing)

    assert len(available) >= min(
        10,
        len(configured_symbols),
    ), (
        "Too few universe symbols were found in DuckDB."
    )

    overlap_ratio = (
        len(available) / len(configured_symbols)
    )

    assert overlap_ratio >= 0.80, (
        f"Only {overlap_ratio:.1%} of configured symbols "
        "exist in DuckDB."
    )


def test_database_has_meaningful_date_range(
    data: DataInterface,
    universe: list[str],
) -> None:
    database_symbols = set(data.symbols())

    available = [
        symbol
        for symbol in universe
        if symbol in database_symbols
    ]

    dates = data.timestamps(
        symbols=available,
    )

    assert len(dates) > LOOKBACK, (
        f"Only {len(dates)} dates available; "
        f"need more than {LOOKBACK}."
    )

    print()
    print("=" * 70)
    print("DATABASE DATE RANGE")
    print("=" * 70)
    print(f"First date: {dates.min()}")
    print(f"Last date:  {dates.max()}")
    print(f"Rows:       {len(dates)}")

    assert dates.is_monotonic_increasing
    assert dates.min() < dates.max()


def test_single_symbol_history_is_valid(
    data: DataInterface,
    universe: list[str],
) -> None:
    database_symbols = set(data.symbols())

    available = [
        symbol
        for symbol in universe
        if symbol in database_symbols
    ]

    assert available

    timestamp = get_test_timestamp(
        data=data,
        available_symbols=available,
    )

    data.set_current_timestamp(timestamp)

    symbol = available[0]

    history = data.history(
        symbol=symbol,
        field="close",
        bars=LOOKBACK + 1,
    )

    print()
    print("=" * 70)
    print("SINGLE SYMBOL HISTORY")
    print("=" * 70)
    print(f"Symbol:    {symbol}")
    print(f"Timestamp: {timestamp}")
    print(f"Rows:      {len(history)}")
    print(history.tail())

    assert not history.empty
    assert len(history) == LOOKBACK + 1
    assert history.index.max() <= timestamp
    assert history.index.is_monotonic_increasing
    assert history.notna().all()
    assert np.isfinite(history.to_numpy()).all()
    assert (history > 0).all()


def test_price_history_frame_is_meaningful(
    data: DataInterface,
    universe: list[str],
) -> None:
    database_symbols = set(data.symbols())

    available = [
        symbol
        for symbol in universe
        if symbol in database_symbols
    ]

    timestamp = get_test_timestamp(
        data=data,
        available_symbols=available,
    )

    data.set_current_timestamp(timestamp)

    prices = data.history_frame(
        symbols=available,
        field="close",
        bars=LOOKBACK + 1,
    )

    print()
    print("=" * 70)
    print("PRICE HISTORY FRAME")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Shape:     {prices.shape}")
    print(
        "Columns with latest values:",
        int(prices.iloc[-1].notna().sum()),
    )

    assert not prices.empty
    assert len(prices) >= LOOKBACK + 1
    assert prices.index.max() <= timestamp
    assert prices.index.is_monotonic_increasing

    valid_latest = prices.iloc[-1].dropna()

    assert len(valid_latest) >= TOP_N, (
        "The latest price row has fewer valid stocks "
        f"than TOP_N={TOP_N}."
    )

    assert np.isfinite(
        valid_latest.to_numpy(dtype=float)
    ).all()

    assert (
        valid_latest.to_numpy(dtype=float) > 0
    ).all()


def test_price_momentum_scores_are_valid(
    data: DataInterface,
    universe: list[str],
) -> None:
    database_symbols = set(data.symbols())

    available = [
        symbol
        for symbol in universe
        if symbol in database_symbols
    ]

    timestamp = get_test_timestamp(
        data=data,
        available_symbols=available,
    )

    data.set_current_timestamp(timestamp)

    prices = data.history_frame(
        symbols=available,
        field="close",
        bars=LOOKBACK + 1,
    )

    parameters = {
        "lookback": LOOKBACK,
        "rsi_window": 14,
        "rsi_threshold": 50.0,
        "ma_short_window": 40,
        "ma_long_window": 100,
    }

    scores = compute_signal(
        signal_name="price",
        prices=prices,
        parameters=parameters,
    )

    assert not scores.empty
    assert scores.shape == prices.shape

    latest = (
        scores.iloc[-1]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(ascending=False)
    )

    print()
    print("=" * 70)
    print("LATEST MOMENTUM SCORES")
    print("=" * 70)
    print(f"Valid scores: {len(latest)}")
    print(latest.head(20))

    assert len(latest) >= TOP_N, (
        f"Only {len(latest)} valid momentum scores."
    )

    assert np.isfinite(
        latest.to_numpy(dtype=float)
    ).all()

    # Momentum values should generally be plausible.
    # This deliberately allows large moves but catches obvious
    # corruption such as improperly scaled prices.
    assert (latest.abs() < 20.0).all(), (
        "Momentum scores contain implausibly large values."
    )


def test_ranking_and_allocation_are_valid(
    data: DataInterface,
    universe: list[str],
) -> None:
    database_symbols = set(data.symbols())

    available = [
        symbol
        for symbol in universe
        if symbol in database_symbols
    ]

    timestamp = get_test_timestamp(
        data=data,
        available_symbols=available,
    )

    data.set_current_timestamp(timestamp)

    prices = data.history_frame(
        symbols=available,
        field="close",
        bars=LOOKBACK + 1,
    )

    parameters = {
        "lookback": LOOKBACK,
        "rsi_window": 14,
        "rsi_threshold": 50.0,
        "ma_short_window": 40,
        "ma_long_window": 100,
    }

    scores = compute_signal(
        signal_name="price",
        prices=prices,
        parameters=parameters,
    )

    selected = rank_stocks(
        score_df=scores,
        top_n=TOP_N,
        minimum_score=MINIMUM_SCORE,
    )

    weights, selected_scores = allocate_weights(
        selected_stocks=selected,
        score_df=scores,
        allocator="score",
        minimum_score=MINIMUM_SCORE,
    )

    print()
    print("=" * 70)
    print("SELECTION AND ALLOCATION")
    print("=" * 70)
    print("Selected:")
    print(selected)
    print()
    print("Scores:")
    print(selected_scores)
    print()
    print("Weights:")
    print(weights)
    print("Weight sum:", sum(weights.values()))

    assert len(selected) > 0, (
        "Ranking returned no selected stocks."
    )

    assert len(selected) <= TOP_N

    assert weights, (
        "Allocation returned no weights."
    )

    assert set(weights).issubset(set(selected))

    assert all(
        weight > 0
        for weight in weights.values()
    )

    assert sum(weights.values()) == pytest.approx(
        1.0,
        abs=1e-9,
    )


def test_no_lookahead_in_momentum_history(
    data: DataInterface,
    universe: list[str],
) -> None:
    database_symbols = set(data.symbols())

    available = [
        symbol
        for symbol in universe
        if symbol in database_symbols
    ]

    dates = data.timestamps(
        symbols=available,
    )

    assert len(dates) > LOOKBACK + 20

    earlier_timestamp = pd.Timestamp(
        dates[-20]
    )

    later_timestamp = pd.Timestamp(
        dates[-1]
    )

    data.set_current_timestamp(
        earlier_timestamp
    )

    earlier_history = data.history(
        symbol=available[0],
        field="close",
        bars=LOOKBACK + 1,
    )

    data.set_current_timestamp(
        later_timestamp
    )

    later_history = data.history(
        symbol=available[0],
        field="close",
        bars=LOOKBACK + 1,
    )

    assert earlier_history.index.max() <= earlier_timestamp
    assert later_history.index.max() <= later_timestamp

    assert earlier_history.index.max() < later_history.index.max()