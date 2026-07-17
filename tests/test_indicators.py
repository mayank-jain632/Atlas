from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from indicators import (
    adx,
    aroon,
    atr,
    bollinger_bands,
    choppiness_index,
    donchian_channels,
    drawdown_from_rolling_high,
    dual_donchian_channels,
    ema,
    keltner_channels,
    macd,
    moving_average_crossover,
    parabolic_sar,
    roc,
    rolling_volatility,
    rsi,
    sma,
    stochastic_oscillator,
    supertrend,
    true_range,
    wma,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=400,
        freq="D",
    )

    trend = np.linspace(
        100.0,
        180.0,
        len(index),
    )

    cycle = (
        5.0
        * np.sin(
            np.arange(len(index))
            / 10.0
        )
    )

    close = trend + cycle

    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.5,
            "Low": close - 1.5,
            "Close": close,
            "Volume": np.full(
                len(index),
                1_000_000.0,
            ),
        },
        index=index,
    )


def test_moving_averages(
    sample_data: pd.DataFrame,
) -> None:
    close = sample_data["Close"]

    assert sma(close, 20).notna().sum() > 0
    assert ema(close, 20).notna().sum() > 0
    assert wma(close, 20).notna().sum() > 0

    crossover = moving_average_crossover(
        close,
        fast_period=20,
        slow_period=50,
    )

    assert {
        "fast_ma",
        "slow_ma",
        "spread",
        "direction",
    }.issubset(crossover.columns)


def test_true_range_and_atr(
    sample_data: pd.DataFrame,
) -> None:
    tr = true_range(sample_data)
    average_range = atr(
        sample_data,
        period=14,
    )

    assert len(tr) == len(sample_data)
    assert (tr.dropna() >= 0).all()
    assert (average_range.dropna() >= 0).all()


def test_rsi_bounds(
    sample_data: pd.DataFrame,
) -> None:
    values = rsi(
        sample_data["Close"],
        period=14,
    ).dropna()

    assert not values.empty
    assert values.between(
        0.0,
        100.0,
    ).all()


def test_macd(
    sample_data: pd.DataFrame,
) -> None:
    result = macd(
        sample_data["Close"]
    )

    assert {
        "macd",
        "signal",
        "histogram",
    }.issubset(result.columns)

    difference = (
        result["macd"]
        - result["signal"]
        - result["histogram"]
    ).dropna()

    assert np.allclose(
        difference,
        0.0,
    )


def test_adx(
    sample_data: pd.DataFrame,
) -> None:
    result = adx(
        sample_data,
        period=14,
    )

    assert {
        "adx",
        "positive_di",
        "negative_di",
        "dx",
    }.issubset(result.columns)

    assert (
        result["adx"].dropna()
        >= 0.0
    ).all()


def test_supertrend(
    sample_data: pd.DataFrame,
) -> None:
    result = supertrend(
        sample_data,
        period=10,
        multiplier=3.0,
    )

    assert {
        "supertrend",
        "direction",
        "upper_band",
        "lower_band",
        "atr",
    }.issubset(result.columns)

    valid_directions = set(
        result["direction"]
        .dropna()
        .unique()
    )

    assert valid_directions.issubset(
        {-1.0, 1.0}
    )


def test_donchian_channels(
    sample_data: pd.DataFrame,
) -> None:
    result = donchian_channels(
        sample_data,
        period=20,
        shift=1,
    )

    dual = dual_donchian_channels(
        sample_data,
        exit_period=50,
        entry_period=20,
    )

    assert {
        "upper",
        "lower",
        "middle",
    }.issubset(result.columns)

    assert {
        "exit_low",
        "entry_high",
    }.issubset(dual.columns)


def test_drawdown(
    sample_data: pd.DataFrame,
) -> None:
    result = drawdown_from_rolling_high(
        sample_data["Close"],
        period=100,
        min_periods=1,
    )

    assert (
        result["drawdown"]
        <= 1e-12
    ).all()


def test_volatility_channels(
    sample_data: pd.DataFrame,
) -> None:
    volatility = rolling_volatility(
        sample_data["Close"],
        period=20,
    )

    bollinger = bollinger_bands(
        sample_data["Close"],
        period=20,
    )

    keltner = keltner_channels(
        sample_data,
        ema_period=20,
        atr_period=10,
    )

    assert (
        volatility.dropna()
        >= 0.0
    ).all()

    assert (
        bollinger["upper"].dropna()
        >= bollinger["lower"].dropna()
    ).all()

    assert (
        keltner["upper"].dropna()
        >= keltner["lower"].dropna()
    ).all()


def test_additional_indicators(
    sample_data: pd.DataFrame,
) -> None:
    close = sample_data["Close"]

    assert roc(
        close,
        period=20,
    ).notna().sum() > 0

    stochastic = stochastic_oscillator(
        sample_data,
    )

    chop = choppiness_index(
        sample_data,
        period=14,
    )

    sar = parabolic_sar(
        sample_data,
    )

    aroon_result = aroon(
        sample_data,
        period=25,
    )

    assert {
        "percent_k",
        "percent_d",
    }.issubset(stochastic.columns)

    assert chop.notna().sum() > 0

    assert {
        "sar",
        "direction",
        "extreme_point",
        "acceleration_factor",
    }.issubset(sar.columns)

    assert {
        "aroon_up",
        "aroon_down",
        "oscillator",
    }.issubset(aroon_result.columns)


def test_lowercase_and_uppercase_columns(
    sample_data: pd.DataFrame,
) -> None:
    lowercase = sample_data.copy()

    lowercase.columns = [
        column.lower()
        for column in lowercase.columns
    ]

    upper_result = atr(
        sample_data,
        period=14,
    )

    lower_result = atr(
        lowercase,
        period=14,
    )

    pd.testing.assert_series_equal(
        upper_result,
        lower_result,
    )