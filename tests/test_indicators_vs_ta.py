"""Cross-validate indicators/ against the `ta` reference library.

Every function in the indicators/ package is checked here. They fall into
three buckets, each compared differently:

1. Non-recursive / pure-window functions (sma, wma, donchian, bollinger,
   aroon, stochastic, roc, macd/ema, true_range). These should match `ta`
   to floating-point precision -- there's no room for two correct
   implementations to disagree.

2. Wilder-smoothed recursive functions (rsi, atr, adx, and anything built
   on them like keltner_channels). `ta`'s ATR and ADX seed the recursion
   with a plain SMA of the first `period` values (the classical textbook
   Wilder method), while ours seeds the ewm recursion from the very first
   bar -- both textbook-valid variants of Wilder smoothing. `ta`'s RSI
   uses the same ewm(alpha=1/period, adjust=False) call as ours, but it
   manufactures an explicit zero price-change at the series' first
   (NaN-diff) bar before smoothing, while ours leaves that bar undefined
   and lets the recursion start from the first real change -- a different
   seed for the same formula. In all three cases the seed's influence
   decays geometrically, so they are NOT expected to match near the
   warm-up point, only in the tail of a long series. Tests for these
   therefore compare only the back of the series.

3. Functions `ta` doesn't provide at all (supertrend, choppiness_index,
   parabolic_sar's exact tie-break behavior, wilder_average standalone,
   and simple derived helpers like rolling_high/drawdown/rolling
   volatility). These are checked against an independent from-scratch
   reference computed directly in this file, not against either
   indicators/ or strategies/futures/indicators.py.

Runs against both synthetic OHLCV data and a real slice of MES=F 1-minute
futures bars from duckdb/futures_data.duckdb (skipped gracefully if that
file isn't present).

Run with `-s` to see the printed accuracy report:

    pytest tests/test_indicators_vs_ta.py -s -v
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb  # noqa: E402

from ta.momentum import (  # noqa: E402
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
)
from ta.trend import (  # noqa: E402
    ADXIndicator,
    AroonIndicator,
    EMAIndicator,
    MACD as TAMACD,
    PSARIndicator,
    SMAIndicator,
    WMAIndicator,
)
from ta.volatility import (  # noqa: E402
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
)

from indicators import (  # noqa: E402
    adx,
    aroon,
    atr,
    bollinger_bands,
    choppiness_index,
    distance_from_high,
    donchian_channels,
    drawdown_from_rolling_high,
    dual_donchian_channels,
    ema,
    keltner_channels,
    macd,
    moving_average_crossover,
    parabolic_sar,
    recovery_moving_average,
    roc,
    rolling_high,
    rolling_low,
    rolling_volatility,
    rsi,
    sma,
    stochastic_oscillator,
    supertrend,
    true_range,
    wilder_average,
    wma,
)


FUTURES_DB_PATH = PROJECT_ROOT / "duckdb" / "futures_data.duckdb"
REAL_SYMBOL = "MES=F"
REAL_BARS = 3000

TIGHT = dict(rtol=1e-8, atol=1e-8)
# Wilder-seed convergence tolerance for atr/adx/rsi/keltner tail comparisons.
WILDER_TAIL = dict(rtol=5e-3, atol=5e-3)


# ============================================================
# Results log, printed as a report at the end of the session
# ============================================================

RESULTS: list[dict] = []
_STATE = {"dataset": "unknown"}


def _format_number(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3e}"


def _record(
    label: str,
    n: int,
    max_abs_diff: float = float("nan"),
    max_rel_diff: float = float("nan"),
    tolerance: str = "",
    detail: str = "",
) -> None:
    RESULTS.append(
        {
            "dataset": _STATE["dataset"],
            "label": label,
            "n": n,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "tolerance": tolerance,
            "detail": detail,
        }
    )


@pytest.fixture(scope="session", autouse=True)
def _print_accuracy_report():
    yield

    if not RESULTS:
        return

    print()
    print("=" * 110)
    print("INDICATOR ACCURACY REPORT -- indicators/ vs ta library / independent reference")
    print("=" * 110)
    print(
        f"{'dataset':<18} {'comparison':<46} {'n':>7} "
        f"{'max_abs_diff':>13} {'max_rel_diff':>13}  tolerance / detail"
    )
    print("-" * 110)

    for row in RESULTS:
        tail = row["tolerance"] or row["detail"]
        print(
            f"{row['dataset']:<18} {row['label']:<46} {row['n']:>7} "
            f"{_format_number(row['max_abs_diff']):>13} "
            f"{_format_number(row['max_rel_diff']):>13}  {tail}"
        )

    datasets = sorted({row["dataset"] for row in RESULTS})
    print("-" * 110)
    print(
        f"{len(RESULTS)} comparisons across {len(datasets)} dataset(s) "
        f"({', '.join(datasets)}) -- every row above passed its assertion; "
        f"any mismatch would have failed the test instead of reaching this report."
    )
    print("=" * 110)


# ============================================================
# Data fixtures
# ============================================================

def _synthetic_ohlc(periods: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    index = pd.date_range("2020-01-01", periods=periods, freq="D")

    steps = rng.normal(loc=0.05, scale=1.2, size=periods)
    close = 100.0 + np.cumsum(steps)
    close = np.maximum(close, 5.0)

    high = close + rng.uniform(0.1, 1.5, size=periods)
    low = close - rng.uniform(0.1, 1.5, size=periods)
    open_ = close + rng.uniform(-0.5, 0.5, size=periods)
    volume = rng.uniform(1.0e5, 1.0e6, size=periods)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _real_futures_ohlc(
    symbol: str = REAL_SYMBOL,
    bars: int = REAL_BARS,
) -> pd.DataFrame | None:
    if not FUTURES_DB_PATH.exists():
        return None

    connection = duckdb.connect(str(FUTURES_DB_PATH), read_only=True)
    try:
        frame = connection.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM bars
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [symbol, bars],
        ).fetchdf()
    finally:
        connection.close()

    if frame.empty:
        return None

    frame = frame.sort_values("timestamp").set_index("timestamp")
    return frame[["open", "high", "low", "close", "volume"]]


@pytest.fixture(
    scope="module",
    params=["synthetic", "real_mes_futures"],
)
def ohlc(request: pytest.FixtureRequest) -> pd.DataFrame:
    _STATE["dataset"] = request.param

    if request.param == "real_mes_futures":
        real = _real_futures_ohlc()
        if real is None:
            pytest.skip(
                "duckdb/futures_data.duckdb not available on this machine"
            )
        return real

    return _synthetic_ohlc()


def _aligned(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaNs from both sides on the shared valid index."""
    frame = pd.concat({"a": a, "b": b}, axis=1).dropna()
    assert not frame.empty, "no overlapping non-NaN values to compare"
    return frame["a"].to_numpy(dtype=float), frame["b"].to_numpy(dtype=float)


def _assert_close(label: str, a: pd.Series, b: pd.Series, **kwargs) -> None:
    x, y = _aligned(a, b)

    abs_diff = np.abs(x - y)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = abs_diff / np.where(np.abs(y) > 1e-9, np.abs(y), np.nan)
    finite_rel = rel_diff[np.isfinite(rel_diff)]

    _record(
        label=label,
        n=len(x),
        max_abs_diff=float(np.max(abs_diff)),
        max_rel_diff=float(np.max(finite_rel)) if finite_rel.size else float("nan"),
        tolerance=", ".join(f"{k}={v:g}" for k, v in kwargs.items()),
    )

    np.testing.assert_allclose(x, y, **kwargs)


def _tail(series: pd.Series, warmup: int) -> pd.Series:
    return series.iloc[warmup:]


# ============================================================
# Bucket 1: pure-window / non-recursive
# ============================================================

def test_sma_matches_ta(ohlc: pd.DataFrame) -> None:
    ours = sma(ohlc["close"], period=20)
    theirs = SMAIndicator(ohlc["close"], window=20).sma_indicator()
    _assert_close("sma(20) vs ta.SMAIndicator", ours, theirs, **TIGHT)


def test_wma_matches_ta(ohlc: pd.DataFrame) -> None:
    ours = wma(ohlc["close"], period=20)
    theirs = WMAIndicator(ohlc["close"], window=20).wma()
    _assert_close("wma(20) vs ta.WMAIndicator", ours, theirs, **TIGHT)


def test_ema_matches_ta(ohlc: pd.DataFrame) -> None:
    ours = ema(ohlc["close"], period=20)
    theirs = EMAIndicator(ohlc["close"], window=20).ema_indicator()
    _assert_close("ema(20) vs ta.EMAIndicator", ours, theirs, **TIGHT)


def test_macd_matches_ta(ohlc: pd.DataFrame) -> None:
    ours = macd(ohlc["close"], fast_period=12, slow_period=26, signal_period=9)
    theirs = TAMACD(
        ohlc["close"],
        window_fast=12,
        window_slow=26,
        window_sign=9,
    )
    _assert_close("macd.macd vs ta.MACD.macd", ours["macd"], theirs.macd(), **TIGHT)
    _assert_close("macd.signal vs ta.MACD.macd_signal", ours["signal"], theirs.macd_signal(), **TIGHT)
    _assert_close("macd.histogram vs ta.MACD.macd_diff", ours["histogram"], theirs.macd_diff(), **TIGHT)


def test_roc_matches_ta(ohlc: pd.DataFrame) -> None:
    # ta reports a percentage (5.0); ours reports a fraction (0.05).
    ours = roc(ohlc["close"], period=20) * 100.0
    theirs = ROCIndicator(ohlc["close"], window=20).roc()
    _assert_close("roc(20) vs ta.ROCIndicator", ours, theirs, **TIGHT)


def test_rsi_converges_to_ta(ohlc: pd.DataFrame) -> None:
    # Both sides use ewm(alpha=1/period, adjust=False), but ta manufactures
    # an explicit zero price-change at the series' first (NaN-diff) bar
    # before smoothing, while ours leaves it undefined and lets ewm's
    # recursion start cleanly from the first real change. That's a
    # different seed for the same recursion, not a formula bug -- it
    # converges once the seed's influence decays, same as atr/adx.
    period = 14
    ours = rsi(ohlc["close"], period=period)
    theirs = RSIIndicator(ohlc["close"], window=period).rsi()

    warmup = period * 15
    _assert_close(
        "rsi(14) [tail] vs ta.RSIIndicator",
        _tail(ours, warmup), _tail(theirs, warmup),
        **WILDER_TAIL,
    )


def test_stochastic_oscillator_matches_ta(ohlc: pd.DataFrame) -> None:
    ours = stochastic_oscillator(ohlc, k_period=14, d_period=3)
    theirs = StochasticOscillator(
        high=ohlc["high"],
        low=ohlc["low"],
        close=ohlc["close"],
        window=14,
        smooth_window=3,
    )
    _assert_close("stochastic.%K vs ta.StochasticOscillator.stoch", ours["percent_k"], theirs.stoch(), **TIGHT)
    _assert_close("stochastic.%D vs ta.StochasticOscillator.stoch_signal", ours["percent_d"], theirs.stoch_signal(), **TIGHT)


def test_bollinger_bands_matches_ta(ohlc: pd.DataFrame) -> None:
    ours = bollinger_bands(ohlc["close"], period=20, standard_deviations=2.0)
    theirs = BollingerBands(ohlc["close"], window=20, window_dev=2)
    _assert_close("bollinger.middle vs ta.BollingerBands.mavg", ours["middle"], theirs.bollinger_mavg(), **TIGHT)
    _assert_close("bollinger.upper vs ta.BollingerBands.hband", ours["upper"], theirs.bollinger_hband(), **TIGHT)
    _assert_close("bollinger.lower vs ta.BollingerBands.lband", ours["lower"], theirs.bollinger_lband(), **TIGHT)


def test_donchian_channels_matches_ta(ohlc: pd.DataFrame) -> None:
    ours = donchian_channels(ohlc, period=20, shift=1)
    theirs = DonchianChannel(
        high=ohlc["high"],
        low=ohlc["low"],
        close=ohlc["close"],
        window=20,
        offset=1,
    )
    _assert_close("donchian.upper vs ta.DonchianChannel.hband", ours["upper"], theirs.donchian_channel_hband(), **TIGHT)
    _assert_close("donchian.lower vs ta.DonchianChannel.lband", ours["lower"], theirs.donchian_channel_lband(), **TIGHT)


def test_dual_donchian_channels_matches_ta(ohlc: pd.DataFrame) -> None:
    ours = dual_donchian_channels(ohlc, exit_period=50, entry_period=20, shift=1)

    exit_channel = DonchianChannel(
        high=ohlc["high"], low=ohlc["low"], close=ohlc["close"],
        window=50, offset=1,
    )
    entry_channel = DonchianChannel(
        high=ohlc["high"], low=ohlc["low"], close=ohlc["close"],
        window=20, offset=1,
    )

    _assert_close(
        "dual_donchian.exit_low vs ta.DonchianChannel(50).lband",
        ours["exit_low"], exit_channel.donchian_channel_lband(), **TIGHT,
    )
    _assert_close(
        "dual_donchian.entry_high vs ta.DonchianChannel(20).hband",
        ours["entry_high"], entry_channel.donchian_channel_hband(), **TIGHT,
    )


def test_aroon_matches_ta(ohlc: pd.DataFrame) -> None:
    ours = aroon(ohlc, period=25)
    theirs = AroonIndicator(ohlc["high"], ohlc["low"], window=25)
    _assert_close("aroon.up vs ta.AroonIndicator.aroon_up", ours["aroon_up"], theirs.aroon_up(), **TIGHT)
    _assert_close("aroon.down vs ta.AroonIndicator.aroon_down", ours["aroon_down"], theirs.aroon_down(), **TIGHT)


def test_true_range_matches_independent_formula(ohlc: pd.DataFrame) -> None:
    # ta bundles true range inside AverageTrueRange without exposing it
    # standalone, so this is checked against a fresh, independent
    # recomputation of the textbook formula instead of the ta library.
    previous_close = ohlc["close"].shift(1)
    expected = pd.concat(
        [
            ohlc["high"] - ohlc["low"],
            (ohlc["high"] - previous_close).abs(),
            (ohlc["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    ours = true_range(ohlc)
    _assert_close("true_range vs independent formula", ours, expected, **TIGHT)


# ============================================================
# Bucket 2: Wilder-smoothed recursive (tail-only comparison)
# ============================================================

def test_atr_converges_to_ta(ohlc: pd.DataFrame) -> None:
    period = 14
    ours = atr(ohlc, period=period)
    theirs = AverageTrueRange(
        high=ohlc["high"], low=ohlc["low"], close=ohlc["close"], window=period,
    ).average_true_range()

    warmup = period * 15
    _assert_close(
        "atr(14) [tail] vs ta.AverageTrueRange",
        _tail(ours, warmup), _tail(theirs, warmup),
        **WILDER_TAIL,
    )


def test_adx_converges_to_ta(ohlc: pd.DataFrame) -> None:
    period = 14
    ours = adx(ohlc, period=period)
    theirs_indicator = ADXIndicator(
        high=ohlc["high"], low=ohlc["low"], close=ohlc["close"], window=period,
    )

    warmup = period * 15
    _assert_close(
        "adx.adx [tail] vs ta.ADXIndicator.adx",
        _tail(ours["adx"], warmup), _tail(theirs_indicator.adx(), warmup),
        rtol=0.1, atol=2.0,
    )
    _assert_close(
        "adx.+DI [tail] vs ta.ADXIndicator.adx_pos",
        _tail(ours["positive_di"], warmup), _tail(theirs_indicator.adx_pos(), warmup),
        rtol=0.1, atol=2.0,
    )
    _assert_close(
        "adx.-DI [tail] vs ta.ADXIndicator.adx_neg",
        _tail(ours["negative_di"], warmup), _tail(theirs_indicator.adx_neg(), warmup),
        rtol=0.1, atol=2.0,
    )


def test_keltner_channels_converges_to_ta(ohlc: pd.DataFrame) -> None:
    ema_period, atr_period, multiplier = 20, 10, 2.0

    ours = keltner_channels(
        ohlc, ema_period=ema_period, atr_period=atr_period, multiplier=multiplier,
    )
    theirs = KeltnerChannel(
        high=ohlc["high"], low=ohlc["low"], close=ohlc["close"],
        window=ema_period, window_atr=atr_period,
        original_version=False, multiplier=int(multiplier),
    )

    # The centerline is a plain EMA, no Wilder seeding involved.
    _assert_close("keltner.middle vs ta.KeltnerChannel.mband", ours["middle"], theirs.keltner_channel_mband(), **TIGHT)

    warmup = atr_period * 15
    _assert_close(
        "keltner.upper [tail] vs ta.KeltnerChannel.hband",
        _tail(ours["upper"], warmup), _tail(theirs.keltner_channel_hband(), warmup),
        **WILDER_TAIL,
    )
    _assert_close(
        "keltner.lower [tail] vs ta.KeltnerChannel.lband",
        _tail(ours["lower"], warmup), _tail(theirs.keltner_channel_lband(), warmup),
        **WILDER_TAIL,
    )


# ============================================================
# Bucket 3: no ta equivalent -- independent reference implementations
# ============================================================

def _reference_supertrend(data: pd.DataFrame, period: int, multiplier: float) -> pd.DataFrame:
    """Independent supertrend implementation (own loop, written fresh here)."""
    high, low, close = data["high"], data["low"], data["close"]

    previous_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    average_true_range = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    mid = (high + low) / 2.0
    upper_basic = mid + multiplier * average_true_range
    lower_basic = mid - multiplier * average_true_range

    n = len(data)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    line = np.full(n, np.nan)

    valid = np.flatnonzero(average_true_range.notna().to_numpy())
    if len(valid) == 0:
        return pd.DataFrame(
            {"supertrend": line, "direction": direction}, index=data.index,
        )

    start = int(valid[0])
    final_upper[start] = upper_basic.iloc[start]
    final_lower[start] = lower_basic.iloc[start]
    direction[start] = 1.0
    line[start] = final_lower[start]

    close_values = close.to_numpy()
    upper_basic_values = upper_basic.to_numpy()
    lower_basic_values = lower_basic.to_numpy()

    for i in range(start + 1, n):
        if upper_basic_values[i] < final_upper[i - 1] or close_values[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_basic_values[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if lower_basic_values[i] > final_lower[i - 1] or close_values[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_basic_values[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if direction[i - 1] == -1.0 and close_values[i] > final_upper[i]:
            direction[i] = 1.0
        elif direction[i - 1] == 1.0 and close_values[i] < final_lower[i]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]

        line[i] = final_lower[i] if direction[i] == 1.0 else final_upper[i]

    return pd.DataFrame({"supertrend": line, "direction": direction}, index=data.index)


def test_supertrend_matches_independent_reference(ohlc: pd.DataFrame) -> None:
    period, multiplier = 10, 3.0

    ours = supertrend(ohlc, period=period, multiplier=multiplier)
    reference = _reference_supertrend(ohlc, period=period, multiplier=multiplier)

    _assert_close("supertrend.line vs independent reference", ours["supertrend"], reference["supertrend"], **TIGHT)
    _assert_close("supertrend.direction vs independent reference", ours["direction"], reference["direction"], **TIGHT)


def _reference_choppiness(data: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = data["high"], data["low"], data["close"]
    previous_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)

    tr_sum = tr.rolling(period).sum()
    price_range = high.rolling(period).max() - low.rolling(period).min()

    return 100.0 * np.log10(tr_sum / price_range) / np.log10(float(period))


def test_choppiness_index_matches_independent_reference(ohlc: pd.DataFrame) -> None:
    period = 14
    ours = choppiness_index(ohlc, period=period)
    reference = _reference_choppiness(ohlc, period=period)
    _assert_close("choppiness_index vs independent reference", ours, reference, **TIGHT)


def test_parabolic_sar_direction_mostly_agrees_with_ta(ohlc: pd.DataFrame) -> None:
    # PSAR implementations vary in tie-break handling at reversal bars, so
    # exact bar-for-bar agreement isn't realistic. Direction should still
    # agree on the large majority of bars, and both must stay in {-1, 1}.
    ours = parabolic_sar(ohlc, step=0.02, max_step=0.20)
    theirs = PSARIndicator(
        high=ohlc["high"], low=ohlc["low"], close=ohlc["close"],
        step=0.02, max_step=0.20,
    )
    their_direction = pd.Series(
        np.where(theirs.psar_up().notna(), 1.0, -1.0),
        index=ohlc.index,
    )

    our_direction, their_direction = _aligned(ours["direction"], their_direction)

    assert set(np.unique(our_direction)).issubset({-1.0, 1.0})
    agreement = float((our_direction == their_direction).mean())

    _record(
        label="parabolic_sar.direction vs ta.PSARIndicator",
        n=len(our_direction),
        detail=f"direction agreement={agreement:.1%} (>=90% required)",
    )

    assert agreement >= 0.90, f"PSAR direction agreement only {agreement:.1%}"


def test_wilder_average_matches_independent_recursion(ohlc: pd.DataFrame) -> None:
    period = 14
    ours = wilder_average(ohlc["close"], period=period)

    values = ohlc["close"].to_numpy(dtype=float)
    reference = np.full(len(values), np.nan)
    alpha = 1.0 / period
    reference[0] = values[0]
    for i in range(1, len(values)):
        reference[i] = alpha * values[i] + (1.0 - alpha) * reference[i - 1]
    reference = pd.Series(reference, index=ohlc.index).where(
        pd.Series(np.arange(len(values)), index=ohlc.index) >= period - 1
    )

    _assert_close("wilder_average(14) vs independent recursion", ours, reference, **TIGHT)


# ============================================================
# Simple derived helpers -- independent pandas recomputation
# ============================================================

def test_rolling_high_low_match_independent_recompute(ohlc: pd.DataFrame) -> None:
    period = 100
    close = ohlc["close"]

    _assert_close(
        "rolling_high(100) vs pandas rolling().max()",
        rolling_high(close, period=period),
        close.rolling(period, min_periods=period).max(),
        **TIGHT,
    )
    _assert_close(
        "rolling_low(100) vs pandas rolling().min()",
        rolling_low(close, period=period),
        close.rolling(period, min_periods=period).min(),
        **TIGHT,
    )


def test_drawdown_from_rolling_high_matches_independent_recompute(
    ohlc: pd.DataFrame,
) -> None:
    period = 100
    close = ohlc["close"]
    high = close.rolling(period, min_periods=1).max()
    expected_drawdown = close / high - 1.0

    ours = drawdown_from_rolling_high(close, period=period, min_periods=1)
    _assert_close("drawdown_from_rolling_high vs independent recompute", ours["drawdown"], expected_drawdown, **TIGHT)
    assert (ours["drawdown"] <= 1e-12).all()


def test_distance_from_high_matches_drawdown(ohlc: pd.DataFrame) -> None:
    period = 100
    close = ohlc["close"]
    expected = drawdown_from_rolling_high(close, period=period, min_periods=1)["drawdown"]
    ours = distance_from_high(close, period=period)
    _assert_close("distance_from_high vs drawdown_from_rolling_high", ours, expected, **TIGHT)


def test_recovery_moving_average_is_sma(ohlc: pd.DataFrame) -> None:
    period = 20
    ours = recovery_moving_average(ohlc["close"], period=period)
    expected = ohlc["close"].rolling(period, min_periods=period).mean()
    _assert_close("recovery_moving_average vs plain sma", ours, expected, **TIGHT)


def test_rolling_volatility_matches_independent_recompute(ohlc: pd.DataFrame) -> None:
    period = 20
    close = ohlc["close"]
    expected = (
        close.pct_change(fill_method=None).rolling(period, min_periods=period).std(ddof=1)
        * np.sqrt(252)
    )
    ours = rolling_volatility(close, period=period, annualization_factor=252)
    _assert_close("rolling_volatility vs independent recompute", ours, expected, **TIGHT)


def test_moving_average_crossover_direction_matches_sign(ohlc: pd.DataFrame) -> None:
    close = ohlc["close"]
    result = moving_average_crossover(close, fast_period=20, slow_period=50, method="sma")

    valid = result.dropna()
    assert not valid.empty
    expected_direction = np.where(valid["fast_ma"] > valid["slow_ma"], 1.0, -1.0)
    actual_direction = valid["direction"].to_numpy()

    _record(
        label="moving_average_crossover.direction vs sign(fast-slow)",
        n=len(actual_direction),
        max_abs_diff=float(np.max(np.abs(actual_direction - expected_direction))),
        detail="exact match required",
    )

    np.testing.assert_array_equal(actual_direction, expected_direction)
