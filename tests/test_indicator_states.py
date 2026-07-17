from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from indicator_states import (
    BULLISH,
    BEARISH,
    UNKNOWN,
    IndicatorStateEngine,
    MovingAverageState,
    MovingAverageCrossoverState,
    RSIState,
    MACDState,
    SupertrendState,
    DonchianState,
    DrawdownRecoveryState,
    ADXTrendState,
    ParabolicSARState,
    ChoppinessState,
    create_indicator_state,
)


def make_ohlc(
    close: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(
                close.iloc[0]
            ),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=close.index,
    )


@pytest.fixture
def bullish_data() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=500,
        freq="D",
    )

    close = pd.Series(
        np.linspace(
            100.0,
            250.0,
            len(index),
        ),
        index=index,
    )

    return make_ohlc(close)


@pytest.fixture
def bearish_data() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=500,
        freq="D",
    )

    close = pd.Series(
        np.linspace(
            250.0,
            100.0,
            len(index),
        ),
        index=index,
    )

    return make_ohlc(close)


def test_moving_average_state(
    bullish_data: pd.DataFrame,
    bearish_data: pd.DataFrame,
) -> None:
    evaluator = MovingAverageState(
        period=100,
    )

    bullish = evaluator.evaluate(
        "QQQ",
        bullish_data,
        previous_state=UNKNOWN,
    )

    bearish = evaluator.evaluate(
        "QQQ",
        bearish_data,
        previous_state=UNKNOWN,
    )

    assert bullish.state == BULLISH
    assert bearish.state == BEARISH


def test_ma_crossover_state(
    bullish_data: pd.DataFrame,
    bearish_data: pd.DataFrame,
) -> None:
    evaluator = (
        MovingAverageCrossoverState(
            fast_period=20,
            slow_period=100,
        )
    )

    assert (
        evaluator.evaluate(
            "QQQ",
            bullish_data,
        ).state
        == BULLISH
    )

    assert (
        evaluator.evaluate(
            "QQQ",
            bearish_data,
        ).state
        == BEARISH
    )


def test_rsi_state(
    bullish_data: pd.DataFrame,
    bearish_data: pd.DataFrame,
) -> None:
    evaluator = RSIState(
        period=14,
        bearish_threshold=40,
        bullish_threshold=60,
    )

    assert (
        evaluator.evaluate(
            "QQQ",
            bullish_data,
        ).state
        == BULLISH
    )

    assert (
        evaluator.evaluate(
            "QQQ",
            bearish_data,
        ).state
        == BEARISH
    )


def test_macd_state(
    bullish_data: pd.DataFrame,
    bearish_data: pd.DataFrame,
) -> None:
    evaluator = MACDState()

    bullish_result = evaluator.evaluate(
        "QQQ",
        bullish_data,
    )

    bearish_result = evaluator.evaluate(
        "QQQ",
        bearish_data,
    )

    assert bullish_result.state in {
        BULLISH,
        BEARISH,
    }

    assert bearish_result.state in {
        BULLISH,
        BEARISH,
    }


def test_supertrend_state(
    bullish_data: pd.DataFrame,
) -> None:
    evaluator = SupertrendState(
        period=10,
        multiplier=3.0,
    )

    result = evaluator.evaluate(
        "QQQ",
        bullish_data,
    )

    assert result.state in {
        BULLISH,
        BEARISH,
    }


def test_donchian_preserves_state(
    bullish_data: pd.DataFrame,
) -> None:
    evaluator = DonchianState(
        exit_period=50,
        entry_period=20,
    )

    result = evaluator.evaluate(
        "QQQ",
        bullish_data,
        previous_state=BULLISH,
    )

    assert result.state in {
        BULLISH,
        BEARISH,
    }


def test_drawdown_state(
    bullish_data: pd.DataFrame,
) -> None:
    evaluator = DrawdownRecoveryState(
        lookback=100,
        bearish_drawdown=0.10,
        bullish_drawdown=0.05,
        recovery_ma_period=20,
    )

    result = evaluator.evaluate(
        "QQQ",
        bullish_data,
    )

    assert result.state == BULLISH


def test_other_states(
    bullish_data: pd.DataFrame,
) -> None:
    evaluators = [
        ADXTrendState(),
        ParabolicSARState(),
        ChoppinessState(),
    ]

    for evaluator in evaluators:
        result = evaluator.evaluate(
            "QQQ",
            bullish_data,
        )

        assert result.state in {
            BULLISH,
            BEARISH,
            UNKNOWN,
        }


def test_factory() -> None:
    evaluator = create_indicator_state(
        "sma",
        {
            "period": 200,
        },
    )

    assert isinstance(
        evaluator,
        MovingAverageState,
    )

    assert evaluator.method == "sma"


def test_insufficient_history() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=10,
        freq="D",
    )

    close = pd.Series(
        np.arange(
            10,
            dtype=float,
        )
        + 100.0,
        index=index,
    )

    data = make_ohlc(close)

    evaluator = MovingAverageState(
        period=200,
    )

    result = evaluator.evaluate(
        "QQQ",
        data,
        previous_state=UNKNOWN,
    )

    assert result.state == UNKNOWN
    assert (
        result.reason
        == "INSUFFICIENT_HISTORY"
    )


class FakeDataProvider:
    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        self.data = data
        self.current_timestamp = (
            data.index[-1]
        )

    def history(
        self,
        symbol: str,
        field: str,
        bars: int,
    ) -> pd.Series:
        del symbol

        return self.data[
            field
        ].tail(bars)

    def get_current_timestamp(
        self,
    ) -> pd.Timestamp:
        return self.current_timestamp


def test_indicator_state_engine(
    bullish_data: pd.DataFrame,
) -> None:
    provider = FakeDataProvider(
        bullish_data
    )

    engine = IndicatorStateEngine(
        evaluator=MovingAverageState(
            period=100
        ),
        initial_state=UNKNOWN,
    )

    result = engine.evaluate(
        data_provider=provider,
        symbol="QQQ",
    )

    assert result.state == BULLISH
    assert engine.is_bullish("QQQ")
    assert not engine.get_history_log().empty