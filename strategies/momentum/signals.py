from __future__ import annotations

import numpy as np
import pandas as pd


def price_momentum(
    prices: pd.DataFrame,
    lookback: int = 90,
) -> pd.DataFrame:
    prices = prices.astype(float)

    return prices.pct_change(
        periods=lookback,
        fill_method=None,
    )


def calculate_rsi(
    prices: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    prices = prices.astype(float)

    delta = prices.diff()

    gain = (
        delta
        .clip(lower=0.0)
        .rolling(window=window, min_periods=window)
        .mean()
    )

    loss = (
        -delta
        .clip(upper=0.0)
        .rolling(window=window, min_periods=window)
        .mean()
    )

    rs = gain / loss.replace(0.0, np.nan)

    rsi = 100.0 - (100.0 / (1.0 + rs))

    # A sequence with gains and no losses has RSI 100.
    rsi = rsi.where(loss != 0.0, 100.0)

    return rsi


def rsi_momentum(
    prices: pd.DataFrame,
    window: int = 14,
    threshold: float = 50.0,
) -> pd.DataFrame:
    rsi = calculate_rsi(
        prices=prices,
        window=window,
    )

    return (rsi - float(threshold)) / float(threshold)


def moving_average_crossover(
    prices: pd.DataFrame,
    short_window: int = 40,
    long_window: int = 100,
) -> pd.DataFrame:
    prices = prices.astype(float)

    short_ma = prices.rolling(
        window=short_window,
        min_periods=short_window,
    ).mean()

    long_ma = prices.rolling(
        window=long_window,
        min_periods=long_window,
    ).mean()

    # Normalize the difference to make scores comparable
    # across stocks with different price levels.
    return (
        short_ma - long_ma
    ) / long_ma.replace(0.0, np.nan)


def volatility_adjusted_momentum(
    prices: pd.DataFrame,
    lookback: int = 90,
) -> pd.DataFrame:
    prices = prices.astype(float)

    momentum = prices.pct_change(
        periods=lookback,
        fill_method=None,
    )

    daily_returns = prices.pct_change(
        fill_method=None,
    )

    volatility = daily_returns.rolling(
        window=lookback,
        min_periods=lookback,
    ).std()

    return momentum / volatility.replace(0.0, np.nan)


def low_vol_signal(
    prices: pd.DataFrame,
    lookback: int = 90,
) -> pd.DataFrame:
    prices = prices.astype(float)

    daily_returns = prices.pct_change(
        fill_method=None,
    )

    volatility = daily_returns.rolling(
        window=lookback,
        min_periods=lookback,
    ).std()

    return 1.0 / volatility.replace(0.0, np.nan)


def trend_quality_signal(
    prices: pd.DataFrame,
    lookback: int = 90,
) -> pd.DataFrame:
    """
    Trend quality = log-price slope multiplied by R-squared.

    Higher values represent stronger and smoother positive trends.
    """
    prices = prices.astype(float)

    result = pd.DataFrame(
        index=prices.index,
        columns=prices.columns,
        dtype=float,
    )

    x = np.arange(lookback, dtype=float)

    for symbol in prices.columns:
        series = prices[symbol]
        values = np.full(len(series), np.nan, dtype=float)

        for index in range(lookback - 1, len(series)):
            window = series.iloc[
                index - lookback + 1:index + 1
            ]

            if window.isna().any() or (window <= 0).any():
                continue

            y = np.log(window.to_numpy(dtype=float))

            slope, intercept = np.polyfit(x, y, 1)
            fitted = slope * x + intercept

            residual_sum = np.sum((y - fitted) ** 2)
            total_sum = np.sum(
                (y - np.mean(y)) ** 2
            )

            r_squared = (
                1.0 - residual_sum / total_sum
                if total_sum > 0
                else 0.0
            )

            values[index] = (
                slope * max(r_squared, 0.0)
            )

        result[symbol] = values

    return result


def compute_signal(
    signal_name: str,
    prices: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """
    Common entry point for all momentum algorithms.
    """
    signal_name = signal_name.lower()

    if signal_name == "price":
        return price_momentum(
            prices,
            lookback=int(parameters["lookback"]),
        )

    if signal_name == "rsi":
        return rsi_momentum(
            prices,
            window=int(parameters["rsi_window"]),
            threshold=float(
                parameters["rsi_threshold"]
            ),
        )

    if signal_name == "ma_cross":
        return moving_average_crossover(
            prices,
            short_window=int(
                parameters["ma_short_window"]
            ),
            long_window=int(
                parameters["ma_long_window"]
            ),
        )

    if signal_name == "vol_adj":
        return volatility_adjusted_momentum(
            prices,
            lookback=int(parameters["lookback"]),
        )

    if signal_name == "low_vol":
        return low_vol_signal(
            prices,
            lookback=int(parameters["lookback"]),
        )

    if signal_name == "trend_quality":
        return trend_quality_signal(
            prices,
            lookback=int(parameters["lookback"]),
        )

    raise ValueError(
        f"Unsupported momentum signal: {signal_name}"
    )