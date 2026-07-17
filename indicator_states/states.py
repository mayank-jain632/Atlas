from __future__ import annotations

import math

import pandas as pd

from indicators import (
    adx,
    choppiness_index,
    dual_donchian_channels,
    ema,
    macd,
    moving_average_crossover,
    parabolic_sar,
    rsi,
    sma,
    supertrend,
)

from indicators.price_action import (
    drawdown_from_rolling_high,
    recovery_moving_average,
)

from .base import (
    IndicatorState,
    normalize_ohlc,
    preserve_state_or_unknown,
)

from .result import (
    BULLISH,
    BEARISH,
    UNKNOWN,
    IndicatorStateResult,
)


def _latest_valid(
    series: pd.Series,
) -> float | None:
    valid = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
    )

    if valid.empty:
        return None

    value = float(valid.iloc[-1])

    if not math.isfinite(value):
        return None

    return value


# ============================================================
# PRICE VS MOVING AVERAGE
# ============================================================

class MovingAverageState(IndicatorState):
    """
    Bullish when price is above its moving average.
    Bearish when price is below its moving average.

    Optional buffers provide hysteresis:

        bullish -> bearish below MA * (1 - bearish_buffer)
        bearish -> bullish above MA * (1 + bullish_buffer)
    """

    name = "moving_average"

    def __init__(
        self,
        period: int = 200,
        method: str = "sma",
        bearish_buffer: float = 0.0,
        bullish_buffer: float = 0.0,
    ) -> None:
        self.period = int(period)
        self.method = str(method).lower()
        self.bearish_buffer = float(
            bearish_buffer
        )
        self.bullish_buffer = float(
            bullish_buffer
        )

        if self.period <= 0:
            raise ValueError(
                "period must be positive."
            )

        if self.method not in {
            "sma",
            "ema",
        }:
            raise ValueError(
                "method must be 'sma' or 'ema'."
            )

        if self.bearish_buffer < 0:
            raise ValueError(
                "bearish_buffer cannot be negative."
            )

        if self.bullish_buffer < 0:
            raise ValueError(
                "bullish_buffer cannot be negative."
            )

    def required_history(self) -> int:
        return self.period + 1

    def evaluate(
        self,
        symbol: str,
        data: pd.DataFrame,
        previous_state: str = UNKNOWN,
    ) -> IndicatorStateResult:
        previous_state = (
            self.validate_previous_state(
                previous_state
            )
        )

        frame = normalize_ohlc(data)

        if len(frame) < self.required_history():
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        close = frame["close"]

        average = (
            sma(close, self.period)
            if self.method == "sma"
            else ema(close, self.period)
        )

        latest_close = _latest_valid(close)
        latest_average = _latest_valid(
            average
        )

        if (
            latest_close is None
            or latest_average is None
        ):
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        bearish_level = (
            latest_average
            * (
                1.0
                - self.bearish_buffer
            )
        )

        bullish_level = (
            latest_average
            * (
                1.0
                + self.bullish_buffer
            )
        )

        if previous_state == BULLISH:
            if latest_close < bearish_level:
                state = BEARISH
                reason = "PRICE_BELOW_MA_BEARISH_LEVEL"
            else:
                state = BULLISH
                reason = "BULLISH_STATE_PRESERVED"

        elif previous_state == BEARISH:
            if latest_close > bullish_level:
                state = BULLISH
                reason = "PRICE_ABOVE_MA_BULLISH_LEVEL"
            else:
                state = BEARISH
                reason = "BEARISH_STATE_PRESERVED"

        else:
            if latest_close > bullish_level:
                state = BULLISH
                reason = "PRICE_ABOVE_MOVING_AVERAGE"

            elif latest_close < bearish_level:
                state = BEARISH
                reason = "PRICE_BELOW_MOVING_AVERAGE"

            else:
                state = UNKNOWN
                reason = "PRICE_INSIDE_MA_BUFFER"

        return self.build_result(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            reason=reason,
            values={
                "close": latest_close,
                "moving_average": latest_average,
                "bullish_level": bullish_level,
                "bearish_level": bearish_level,
                "period": self.period,
                "method": self.method,
            },
        )


# ============================================================
# MOVING AVERAGE CROSSOVER
# ============================================================

class MovingAverageCrossoverState(
    IndicatorState
):
    name = "moving_average_crossover"

    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        method: str = "sma",
    ) -> None:
        self.fast_period = int(
            fast_period
        )
        self.slow_period = int(
            slow_period
        )
        self.method = str(method).lower()

        if self.fast_period <= 0:
            raise ValueError(
                "fast_period must be positive."
            )

        if self.slow_period <= 0:
            raise ValueError(
                "slow_period must be positive."
            )

        if self.fast_period >= self.slow_period:
            raise ValueError(
                "fast_period must be smaller "
                "than slow_period."
            )

    def required_history(self) -> int:
        return self.slow_period + 1

    def evaluate(
        self,
        symbol: str,
        data: pd.DataFrame,
        previous_state: str = UNKNOWN,
    ) -> IndicatorStateResult:
        previous_state = (
            self.validate_previous_state(
                previous_state
            )
        )

        frame = normalize_ohlc(data)

        if len(frame) < self.required_history():
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        result = moving_average_crossover(
            frame["close"],
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            method=self.method,
        )

        fast = _latest_valid(
            result["fast_ma"]
        )

        slow = _latest_valid(
            result["slow_ma"]
        )

        direction = _latest_valid(
            result["direction"]
        )

        if direction is None:
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        state = (
            BULLISH
            if direction > 0
            else BEARISH
        )

        reason = (
            "FAST_MA_ABOVE_SLOW_MA"
            if state == BULLISH
            else "FAST_MA_BELOW_SLOW_MA"
        )

        return self.build_result(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            reason=reason,
            values={
                "fast_ma": fast,
                "slow_ma": slow,
                "direction": direction,
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "method": self.method,
            },
        )


# ============================================================
# RSI
# ============================================================

class RSIState(IndicatorState):
    """
    Hysteresis RSI state.

    RSI below bearish_threshold:
        BEARISH

    RSI above bullish_threshold:
        BULLISH

    Between thresholds:
        preserve the prior state
    """

    name = "rsi"

    def __init__(
        self,
        period: int = 14,
        bearish_threshold: float = 40.0,
        bullish_threshold: float = 50.0,
    ) -> None:
        self.period = int(period)
        self.bearish_threshold = float(
            bearish_threshold
        )
        self.bullish_threshold = float(
            bullish_threshold
        )

        if (
            self.bearish_threshold
            >= self.bullish_threshold
        ):
            raise ValueError(
                "bearish_threshold must be below "
                "bullish_threshold."
            )

    def required_history(self) -> int:
        return self.period + 2

    def evaluate(
        self,
        symbol: str,
        data: pd.DataFrame,
        previous_state: str = UNKNOWN,
    ) -> IndicatorStateResult:
        previous_state = (
            self.validate_previous_state(
                previous_state
            )
        )

        frame = normalize_ohlc(data)

        if len(frame) < self.required_history():
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        value = _latest_valid(
            rsi(
                frame["close"],
                period=self.period,
            )
        )

        if value is None:
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        if value < self.bearish_threshold:
            state = BEARISH
            reason = "RSI_BELOW_BEARISH_THRESHOLD"

        elif value > self.bullish_threshold:
            state = BULLISH
            reason = "RSI_ABOVE_BULLISH_THRESHOLD"

        else:
            state = preserve_state_or_unknown(
                previous_state
            )
            reason = "RSI_INSIDE_HYSTERESIS_RANGE"

        return self.build_result(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            reason=reason,
            values={
                "rsi": value,
                "period": self.period,
                "bearish_threshold": (
                    self.bearish_threshold
                ),
                "bullish_threshold": (
                    self.bullish_threshold
                ),
            },
        )


# ============================================================
# MACD
# ============================================================

class MACDState(IndicatorState):
    name = "macd"

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        self.fast_period = int(
            fast_period
        )
        self.slow_period = int(
            slow_period
        )
        self.signal_period = int(
            signal_period
        )

    def required_history(self) -> int:
        return (
            self.slow_period
            + self.signal_period
            + 2
        )

    def evaluate(
        self,
        symbol: str,
        data: pd.DataFrame,
        previous_state: str = UNKNOWN,
    ) -> IndicatorStateResult:
        previous_state = (
            self.validate_previous_state(
                previous_state
            )
        )

        frame = normalize_ohlc(data)

        if len(frame) < self.required_history():
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        result = macd(
            frame["close"],
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period,
        )

        macd_value = _latest_valid(
            result["macd"]
        )

        signal_value = _latest_valid(
            result["signal"]
        )

        histogram = _latest_valid(
            result["histogram"]
        )

        if (
            macd_value is None
            or signal_value is None
        ):
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        state = (
            BULLISH
            if macd_value > signal_value
            else BEARISH
        )

        reason = (
            "MACD_ABOVE_SIGNAL"
            if state == BULLISH
            else "MACD_BELOW_SIGNAL"
        )

        return self.build_result(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            reason=reason,
            values={
                "macd": macd_value,
                "signal": signal_value,
                "histogram": histogram,
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "signal_period": self.signal_period,
            },
        )


# ============================================================
# SUPERTREND
# ============================================================

class SupertrendState(IndicatorState):
    name = "supertrend"

    def __init__(
        self,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> None:
        self.period = int(period)
        self.multiplier = float(
            multiplier
        )

    def required_history(self) -> int:
        return self.period + 10

    def evaluate(
        self,
        symbol: str,
        data: pd.DataFrame,
        previous_state: str = UNKNOWN,
    ) -> IndicatorStateResult:
        previous_state = (
            self.validate_previous_state(
                previous_state
            )
        )

        frame = normalize_ohlc(data)

        if len(frame) < self.required_history():
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        result = supertrend(
            frame,
            period=self.period,
            multiplier=self.multiplier,
        )

        direction = _latest_valid(
            result["direction"]
        )

        line = _latest_valid(
            result["supertrend"]
        )

        latest_close = _latest_valid(
            frame["close"]
        )

        if direction is None:
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        state = (
            BULLISH
            if direction > 0
            else BEARISH
        )

        reason = (
            "SUPERTREND_BULLISH"
            if state == BULLISH
            else "SUPERTREND_BEARISH"
        )

        return self.build_result(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            reason=reason,
            values={
                "close": latest_close,
                "supertrend": line,
                "direction": direction,
                "period": self.period,
                "multiplier": self.multiplier,
            },
        )


# ============================================================
# DONCHIAN
# ============================================================

class DonchianState(IndicatorState):
    """
    Bullish after a close above the entry-period high.
    Bearish after a close below the exit-period low.
    Otherwise preserve the existing state.
    """

    name = "donchian"

    def __init__(
        self,
        exit_period: int = 50,
        entry_period: int = 20,
    ) -> None:
        self.exit_period = int(
            exit_period
        )
        self.entry_period = int(
            entry_period
        )

    def required_history(self) -> int:
        return max(
            self.exit_period,
            self.entry_period,
        ) + 2

    def evaluate(
        self,
        symbol: str,
        data: pd.DataFrame,
        previous_state: str = UNKNOWN,
    ) -> IndicatorStateResult:
        previous_state = (
            self.validate_previous_state(
                previous_state
            )
        )

        frame = normalize_ohlc(data)

        if len(frame) < self.required_history():
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        result = dual_donchian_channels(
            frame,
            exit_period=self.exit_period,
            entry_period=self.entry_period,
            shift=1,
        )

        latest_close = _latest_valid(
            frame["close"]
        )

        exit_low = _latest_valid(
            result["exit_low"]
        )

        entry_high = _latest_valid(
            result["entry_high"]
        )

        if (
            latest_close is None
            or exit_low is None
            or entry_high is None
        ):
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        if latest_close < exit_low:
            state = BEARISH
            reason = "CLOSE_BELOW_DONCHIAN_EXIT_LOW"

        elif latest_close > entry_high:
            state = BULLISH
            reason = "CLOSE_ABOVE_DONCHIAN_ENTRY_HIGH"

        else:
            state = preserve_state_or_unknown(
                previous_state
            )
            reason = "CLOSE_INSIDE_DONCHIAN_RANGE"

        return self.build_result(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            reason=reason,
            values={
                "close": latest_close,
                "exit_low": exit_low,
                "entry_high": entry_high,
                "exit_period": self.exit_period,
                "entry_period": self.entry_period,
            },
        )


# ============================================================
# DRAWDOWN RECOVERY
# ============================================================

class DrawdownRecoveryState(
    IndicatorState
):
    name = "drawdown_recovery"

    def __init__(
        self,
        lookback: int = 252,
        bearish_drawdown: float = 0.10,
        bullish_drawdown: float = 0.05,
        recovery_ma_period: int = 20,
    ) -> None:
        self.lookback = int(lookback)

        self.bearish_drawdown = abs(
            float(bearish_drawdown)
        )

        self.bullish_drawdown = abs(
            float(bullish_drawdown)
        )

        self.recovery_ma_period = int(
            recovery_ma_period
        )

        if (
            self.bullish_drawdown
            >= self.bearish_drawdown
        ):
            raise ValueError(
                "bullish_drawdown must be smaller "
                "than bearish_drawdown."
            )

    def required_history(self) -> int:
        return max(
            self.lookback,
            self.recovery_ma_period,
        ) + 1

    def evaluate(
        self,
        symbol: str,
        data: pd.DataFrame,
        previous_state: str = UNKNOWN,
    ) -> IndicatorStateResult:
        previous_state = (
            self.validate_previous_state(
                previous_state
            )
        )

        frame = normalize_ohlc(data)

        if len(frame) < self.required_history():
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        close = frame["close"]

        drawdown_data = (
            drawdown_from_rolling_high(
                close,
                period=self.lookback,
                min_periods=self.lookback,
            )
        )

        recovery_average = (
            recovery_moving_average(
                close,
                period=self.recovery_ma_period,
            )
        )

        latest_close = _latest_valid(close)

        latest_drawdown = _latest_valid(
            drawdown_data["drawdown"]
        )

        latest_recovery_average = (
            _latest_valid(
                recovery_average
            )
        )

        if (
            latest_close is None
            or latest_drawdown is None
            or latest_recovery_average is None
        ):
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        if (
            latest_drawdown
            <= -self.bearish_drawdown
        ):
            state = BEARISH
            reason = "DRAWDOWN_BELOW_BEARISH_LEVEL"

        elif (
            latest_drawdown
            >= -self.bullish_drawdown
            and latest_close
            > latest_recovery_average
        ):
            state = BULLISH
            reason = "DRAWDOWN_RECOVERY_CONFIRMED"

        else:
            state = preserve_state_or_unknown(
                previous_state
            )
            reason = "DRAWDOWN_RECOVERY_INCOMPLETE"

        return self.build_result(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            reason=reason,
            values={
                "close": latest_close,
                "drawdown": latest_drawdown,
                "recovery_ma": (
                    latest_recovery_average
                ),
                "lookback": self.lookback,
                "bearish_drawdown": (
                    self.bearish_drawdown
                ),
                "bullish_drawdown": (
                    self.bullish_drawdown
                ),
                "recovery_ma_period": (
                    self.recovery_ma_period
                ),
            },
        )


# ============================================================
# ADX TREND
# ============================================================

class ADXTrendState(IndicatorState):
    """
    A bullish state requires:
        +DI > -DI
        ADX > bullish_threshold

    A bearish state is entered when:
        -DI > +DI
        or ADX < bearish_threshold
    """

    name = "adx_trend"

    def __init__(
        self,
        period: int = 14,
        bearish_threshold: float = 15.0,
        bullish_threshold: float = 20.0,
    ) -> None:
        self.period = int(period)

        self.bearish_threshold = float(
            bearish_threshold
        )

        self.bullish_threshold = float(
            bullish_threshold
        )

        if (
            self.bearish_threshold
            >= self.bullish_threshold
        ):
            raise ValueError(
                "bearish_threshold must be below "
                "bullish_threshold."
            )

    def required_history(self) -> int:
        return self.period * 3

    def evaluate(
        self,
        symbol: str,
        data: pd.DataFrame,
        previous_state: str = UNKNOWN,
    ) -> IndicatorStateResult:
        previous_state = (
            self.validate_previous_state(
                previous_state
            )
        )

        frame = normalize_ohlc(data)

        if len(frame) < self.required_history():
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        result = adx(
            frame,
            period=self.period,
        )

        adx_value = _latest_valid(
            result["adx"]
        )

        positive_di = _latest_valid(
            result["positive_di"]
        )

        negative_di = _latest_valid(
            result["negative_di"]
        )

        if (
            adx_value is None
            or positive_di is None
            or negative_di is None
        ):
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        bullish_direction = (
            positive_di > negative_di
        )

        bearish_direction = (
            negative_di > positive_di
        )

        if (
            bullish_direction
            and adx_value
            > self.bullish_threshold
        ):
            state = BULLISH
            reason = "ADX_BULLISH_TREND"

        elif (
            bearish_direction
            or adx_value
            < self.bearish_threshold
        ):
            state = BEARISH
            reason = "ADX_BEARISH_OR_WEAK_TREND"

        else:
            state = preserve_state_or_unknown(
                previous_state
            )
            reason = "ADX_INSIDE_HYSTERESIS_RANGE"

        return self.build_result(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            reason=reason,
            values={
                "adx": adx_value,
                "positive_di": positive_di,
                "negative_di": negative_di,
                "period": self.period,
                "bearish_threshold": (
                    self.bearish_threshold
                ),
                "bullish_threshold": (
                    self.bullish_threshold
                ),
            },
        )


# ============================================================
# PARABOLIC SAR
# ============================================================

class ParabolicSARState(IndicatorState):
    name = "parabolic_sar"

    def __init__(
        self,
        step: float = 0.02,
        max_step: float = 0.20,
    ) -> None:
        self.step = float(step)
        self.max_step = float(max_step)

    def required_history(self) -> int:
        return 5

    def evaluate(
        self,
        symbol: str,
        data: pd.DataFrame,
        previous_state: str = UNKNOWN,
    ) -> IndicatorStateResult:
        previous_state = (
            self.validate_previous_state(
                previous_state
            )
        )

        frame = normalize_ohlc(data)

        if len(frame) < self.required_history():
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        result = parabolic_sar(
            frame,
            step=self.step,
            max_step=self.max_step,
        )

        direction = _latest_valid(
            result["direction"]
        )

        sar_value = _latest_valid(
            result["sar"]
        )

        latest_close = _latest_valid(
            frame["close"]
        )

        if direction is None:
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        state = (
            BULLISH
            if direction > 0
            else BEARISH
        )

        reason = (
            "PSAR_BULLISH"
            if state == BULLISH
            else "PSAR_BEARISH"
        )

        return self.build_result(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            reason=reason,
            values={
                "close": latest_close,
                "sar": sar_value,
                "direction": direction,
                "step": self.step,
                "max_step": self.max_step,
            },
        )


# ============================================================
# CHOPPINESS TREND FILTER
# ============================================================

class ChoppinessState(IndicatorState):
    """
    Choppiness is primarily a trend/range state rather than a
    directional signal.

    This implementation combines Choppiness Index with a
    moving-average direction:

        low choppiness + price above MA -> BULLISH
        high choppiness or price below MA -> BEARISH
    """

    name = "choppiness"

    def __init__(
        self,
        period: int = 14,
        moving_average_period: int = 50,
        bullish_threshold: float = 38.2,
        bearish_threshold: float = 61.8,
    ) -> None:
        self.period = int(period)

        self.moving_average_period = int(
            moving_average_period
        )

        self.bullish_threshold = float(
            bullish_threshold
        )

        self.bearish_threshold = float(
            bearish_threshold
        )

        if (
            self.bullish_threshold
            >= self.bearish_threshold
        ):
            raise ValueError(
                "bullish_threshold must be below "
                "bearish_threshold."
            )

    def required_history(self) -> int:
        return max(
            self.period,
            self.moving_average_period,
        ) + 2

    def evaluate(
        self,
        symbol: str,
        data: pd.DataFrame,
        previous_state: str = UNKNOWN,
    ) -> IndicatorStateResult:
        previous_state = (
            self.validate_previous_state(
                previous_state
            )
        )

        frame = normalize_ohlc(data)

        if len(frame) < self.required_history():
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        chop = _latest_valid(
            choppiness_index(
                frame,
                period=self.period,
            )
        )

        moving_average = _latest_valid(
            sma(
                frame["close"],
                period=self.moving_average_period,
            )
        )

        latest_close = _latest_valid(
            frame["close"]
        )

        if (
            chop is None
            or moving_average is None
            or latest_close is None
        ):
            return self.insufficient_history_result(
                symbol,
                previous_state,
                len(frame),
            )

        if (
            chop < self.bullish_threshold
            and latest_close > moving_average
        ):
            state = BULLISH
            reason = "LOW_CHOP_BULLISH_TREND"

        elif (
            chop > self.bearish_threshold
            or latest_close < moving_average
        ):
            state = BEARISH
            reason = "HIGH_CHOP_OR_PRICE_BELOW_MA"

        else:
            state = preserve_state_or_unknown(
                previous_state
            )
            reason = "CHOPPINESS_INSIDE_NEUTRAL_RANGE"

        return self.build_result(
            symbol=symbol,
            state=state,
            previous_state=previous_state,
            reason=reason,
            values={
                "close": latest_close,
                "moving_average": moving_average,
                "choppiness": chop,
                "period": self.period,
                "moving_average_period": (
                    self.moving_average_period
                ),
                "bullish_threshold": (
                    self.bullish_threshold
                ),
                "bearish_threshold": (
                    self.bearish_threshold
                ),
            },
        )