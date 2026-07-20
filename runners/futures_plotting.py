from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# HELPERS
# ============================================================

def _prepare_frame(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Return a clean, time-sorted DataFrame.
    """

    if frame is None or frame.empty:
        return pd.DataFrame()

    result = frame.copy()

    if timestamp_column in result.columns:
        result[timestamp_column] = pd.to_datetime(
            result[timestamp_column],
            errors="coerce",
        )

        result = result.dropna(
            subset=[timestamp_column]
        )

        result = result.sort_values(
            timestamp_column
        )

        result = result.set_index(
            timestamp_column
        )

    else:
        result.index = pd.to_datetime(
            result.index,
            errors="coerce",
        )

        result = result[
            result.index.notna()
        ].sort_index()

    return result


def _trade_rows(
    tradebook: pd.DataFrame,
    action: str | list[str],
) -> pd.DataFrame:
    """
    Select tradebook rows for one or multiple actions.
    """

    if tradebook.empty:
        return tradebook.copy()

    if isinstance(action, str):
        actions = {action}
    else:
        actions = set(action)

    return tradebook[
        tradebook["action"].isin(actions)
    ].copy()


def _scatter_trades(
    *,
    axis: Any,
    tradebook: pd.DataFrame,
) -> None:
    """
    Add actual futures executions to a price chart.
    """

    if tradebook.empty:
        return

    # Long entry
    long_entries = _trade_rows(
        tradebook,
        "BUY",
    )

    if not long_entries.empty:
        axis.scatter(
            long_entries.index,
            long_entries["price"],
            marker="^",
            s=80,
            label="Long entry",
            zorder=5,
        )

    # Short entry
    short_entries = _trade_rows(
        tradebook,
        "SELL_SHORT",
    )

    if not short_entries.empty:
        axis.scatter(
            short_entries.index,
            short_entries["price"],
            marker="v",
            s=80,
            label="Short entry",
            zorder=5,
        )

    # Long exit
    long_exits = _trade_rows(
        tradebook,
        "SELL",
    )

    if not long_exits.empty:
        axis.scatter(
            long_exits.index,
            long_exits["price"],
            marker="x",
            s=70,
            label="Long exit",
            zorder=5,
        )

    # Short exit
    short_exits = _trade_rows(
        tradebook,
        "BUY_TO_COVER",
    )

    if not short_exits.empty:
        axis.scatter(
            short_exits.index,
            short_exits["price"],
            marker="P",
            s=70,
            label="Short exit",
            zorder=5,
        )

    # Highlight ATR stop executions separately.
    if "reason" in tradebook.columns:
        stops = tradebook[
            tradebook["reason"] == "ATR_STOP"
        ]

        if not stops.empty:
            axis.scatter(
                stops.index,
                stops["price"],
                marker="X",
                s=100,
                label="ATR stop",
                zorder=6,
            )


# ============================================================
# STEMA
# ============================================================

def plot_stema(
    *,
    result: dict[str, pd.DataFrame],
    uid: str,
    output_dir: str | Path,
    show: bool = False,
) -> Path:
    """
    Plot close, EMA, Supertrend, trade executions,
    position, equity and drawdown for STEMA.
    """

    signals = _prepare_frame(
        result["signals"]
    )

    tradebook = _prepare_frame(
        result["tradebook"]
    )

    equity = _prepare_frame(
        result["equity"]
    )

    if signals.empty:
        raise ValueError(
            "Cannot plot STEMA because signals are empty."
        )

    required_signal_columns = {
        "close",
        "ema",
        "supertrend",
    }

    missing = (
        required_signal_columns
        - set(signals.columns)
    )

    if missing:
        raise ValueError(
            "STEMA signal log is missing: "
            + ", ".join(sorted(missing))
        )

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(16, 13),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                4.0,
                1.0,
                1.5,
                1.2,
            ]
        },
    )

    price_axis = axes[0]
    position_axis = axes[1]
    equity_axis = axes[2]
    drawdown_axis = axes[3]

    # --------------------------------------------------------
    # Price and indicators
    # --------------------------------------------------------

    price_axis.plot(
        signals.index,
        signals["close"],
        label="Close",
        linewidth=1.2,
    )

    price_axis.plot(
        signals.index,
        signals["ema"],
        label="EMA",
        linewidth=1.1,
    )

    price_axis.plot(
        signals.index,
        signals["supertrend"],
        label="Supertrend",
        linewidth=1.0,
    )

    _scatter_trades(
        axis=price_axis,
        tradebook=tradebook,
    )

    price_axis.set_title(
        f"STEMA futures strategy\n{uid}"
    )

    price_axis.set_ylabel(
        "Futures price"
    )

    price_axis.grid(
        alpha=0.25
    )

    price_axis.legend(
        loc="best",
        ncol=3,
    )

    # --------------------------------------------------------
    # Position direction
    # --------------------------------------------------------

    if "actual_direction" in signals.columns:
        position_axis.step(
            signals.index,
            signals["actual_direction"],
            where="post",
            label="Position",
        )

    position_axis.axhline(
        0.0,
        linewidth=0.8,
    )

    position_axis.set_yticks(
        [-1, 0, 1]
    )

    position_axis.set_yticklabels(
        ["Short", "Flat", "Long"]
    )

    position_axis.set_ylabel(
        "Position"
    )

    position_axis.grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Equity
    # --------------------------------------------------------

    if not equity.empty:
        equity_axis.plot(
            equity.index,
            equity["equity"],
            label="Account equity",
            linewidth=1.2,
        )

        equity_axis.set_ylabel(
            "Equity"
        )

        equity_axis.grid(
            alpha=0.25
        )

        equity_axis.legend(
            loc="best"
        )

        running_peak = (
            equity["equity"].cummax()
        )

        drawdown = (
            equity["equity"]
            / running_peak
            - 1.0
        )

        drawdown_axis.fill_between(
            drawdown.index,
            drawdown,
            0.0,
            alpha=0.35,
        )

        drawdown_axis.plot(
            drawdown.index,
            drawdown,
            linewidth=0.9,
        )

        drawdown_axis.set_ylabel(
            "Drawdown"
        )

        drawdown_axis.set_xlabel(
            "Date"
        )

        drawdown_axis.grid(
            alpha=0.25
        )

    figure.tight_layout()

    file_path = (
        output_path
        / "stema_strategy_plot.png"
    )

    figure.savefig(
        file_path,
        dpi=160,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(
        figure
    )

    return file_path


# ============================================================
# PSAREMA
# ============================================================

def plot_psarema(
    *,
    result: dict[str, pd.DataFrame],
    uid: str,
    output_dir: str | Path,
    show: bool = False,
) -> Path:
    """
    Plot PSAR, EMA, trades, equity and drawdown.
    """

    signals = _prepare_frame(
        result["signals"]
    )

    tradebook = _prepare_frame(
        result["tradebook"]
    )

    equity = _prepare_frame(
        result["equity"]
    )

    if signals.empty:
        raise ValueError(
            "Cannot plot PSAREMA because signals are empty."
        )

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(16, 13),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                4.0,
                1.0,
                1.5,
                1.2,
            ]
        },
    )

    price_axis = axes[0]
    position_axis = axes[1]
    equity_axis = axes[2]
    drawdown_axis = axes[3]

    price_axis.plot(
        signals.index,
        signals["close"],
        label="Close",
        linewidth=1.2,
    )

    price_axis.plot(
        signals.index,
        signals["ema"],
        label="EMA",
        linewidth=1.1,
    )

    price_axis.scatter(
        signals.index,
        signals["psar"],
        s=7,
        label="PSAR",
    )

    _scatter_trades(
        axis=price_axis,
        tradebook=tradebook,
    )

    price_axis.set_title(
        f"PSAREMA futures strategy\n{uid}"
    )

    price_axis.set_ylabel(
        "Futures price"
    )

    price_axis.grid(
        alpha=0.25
    )

    price_axis.legend(
        loc="best",
        ncol=3,
    )

    position_axis.step(
        signals.index,
        signals["actual_direction"],
        where="post",
        label="Position",
    )

    position_axis.axhline(
        0.0,
        linewidth=0.8,
    )

    position_axis.set_yticks(
        [-1, 0, 1]
    )

    position_axis.set_yticklabels(
        ["Short", "Flat", "Long"]
    )

    position_axis.set_ylabel(
        "Position"
    )

    position_axis.grid(
        alpha=0.25
    )

    if not equity.empty:
        equity_axis.plot(
            equity.index,
            equity["equity"],
            label="Account equity",
        )

        equity_axis.set_ylabel(
            "Equity"
        )

        equity_axis.grid(
            alpha=0.25
        )

        equity_axis.legend(
            loc="best"
        )

        running_peak = (
            equity["equity"].cummax()
        )

        drawdown = (
            equity["equity"]
            / running_peak
            - 1.0
        )

        drawdown_axis.fill_between(
            drawdown.index,
            drawdown,
            0.0,
            alpha=0.35,
        )

        drawdown_axis.plot(
            drawdown.index,
            drawdown,
            linewidth=0.9,
        )

        drawdown_axis.set_ylabel(
            "Drawdown"
        )

        drawdown_axis.set_xlabel(
            "Date"
        )

        drawdown_axis.grid(
            alpha=0.25
        )

    figure.tight_layout()

    file_path = (
        output_path
        / "psarema_strategy_plot.png"
    )

    figure.savefig(
        file_path,
        dpi=160,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(
        figure
    )

    return file_path


# ============================================================
# DCEMACHOP
# ============================================================

def plot_dcemachop(
    *,
    result: dict[str, pd.DataFrame],
    uid: str,
    output_dir: str | Path,
    show: bool = False,
) -> Path:
    """
    Plot Donchian, EMA, ADX, DI, Choppiness,
    entries/exits and equity.
    """

    signals = _prepare_frame(
        result["signals"]
    )

    tradebook = _prepare_frame(
        result["tradebook"]
    )

    equity = _prepare_frame(
        result["equity"]
    )

    if signals.empty:
        raise ValueError(
            "Cannot plot DCEMACHOP because signals are empty."
        )

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(16, 16),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                4.0,
                1.4,
                1.4,
                1.0,
                1.6,
            ]
        },
    )

    price_axis = axes[0]
    adx_axis = axes[1]
    chop_axis = axes[2]
    position_axis = axes[3]
    equity_axis = axes[4]

    # Price and trend indicators
    price_axis.plot(
        signals.index,
        signals["close"],
        label="Close",
    )

    price_axis.plot(
        signals.index,
        signals["ema"],
        label="EMA",
    )

    price_axis.plot(
        signals.index,
        signals["donchian_upper"],
        label="Donchian upper",
        linewidth=0.9,
    )

    price_axis.plot(
        signals.index,
        signals["donchian_lower"],
        label="Donchian lower",
        linewidth=0.9,
    )

    _scatter_trades(
        axis=price_axis,
        tradebook=tradebook,
    )

    price_axis.set_title(
        f"DCEMACHOP futures strategy\n{uid}"
    )

    price_axis.set_ylabel(
        "Futures price"
    )

    price_axis.grid(
        alpha=0.25
    )

    price_axis.legend(
        loc="best",
        ncol=3,
    )

    # ADX and directional indicators
    adx_axis.plot(
        signals.index,
        signals["adx"],
        label="ADX",
    )

    adx_axis.plot(
        signals.index,
        signals["plus_di"],
        label="+DI",
    )

    adx_axis.plot(
        signals.index,
        signals["minus_di"],
        label="-DI",
    )

    adx_axis.grid(
        alpha=0.25
    )

    adx_axis.set_ylabel(
        "ADX / DI"
    )

    adx_axis.legend(
        loc="best",
        ncol=3,
    )

    # Choppiness
    chop_axis.plot(
        signals.index,
        signals["chop"],
        label="Choppiness",
    )

    chop_axis.grid(
        alpha=0.25
    )

    chop_axis.set_ylabel(
        "CHOP"
    )

    chop_axis.legend(
        loc="best"
    )

    # Position
    position_axis.step(
        signals.index,
        signals["actual_direction"],
        where="post",
    )

    position_axis.axhline(
        0.0,
        linewidth=0.8,
    )

    position_axis.set_yticks(
        [-1, 0, 1]
    )

    position_axis.set_yticklabels(
        ["Short", "Flat", "Long"]
    )

    position_axis.set_ylabel(
        "Position"
    )

    position_axis.grid(
        alpha=0.25
    )

    # Equity
    if not equity.empty:
        equity_axis.plot(
            equity.index,
            equity["equity"],
            label="Account equity",
        )

        equity_axis.set_ylabel(
            "Equity"
        )

        equity_axis.set_xlabel(
            "Date"
        )

        equity_axis.grid(
            alpha=0.25
        )

        equity_axis.legend(
            loc="best"
        )

    figure.tight_layout()

    file_path = (
        output_path
        / "dcemachop_strategy_plot.png"
    )

    figure.savefig(
        file_path,
        dpi=160,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(
        figure
    )

    return file_path


# ============================================================
# DISPATCHER
# ============================================================

def plot_futures_strategy(
    *,
    strategy_name: str,
    result: dict[str, pd.DataFrame],
    uid: str,
    output_dir: str | Path,
    show: bool = False,
) -> Path:
    """
    Dispatch to the appropriate futures strategy plot.
    """

    strategy_name = (
        strategy_name
        .strip()
        .lower()
    )

    plotters = {
        "stema": plot_stema,
        "psarema": plot_psarema,
        "dcemachop": plot_dcemachop,
    }

    if strategy_name not in plotters:
        raise ValueError(
            f"No futures plotter for "
            f"strategy '{strategy_name}'."
        )

    return plotters[strategy_name](
        result=result,
        uid=uid,
        output_dir=output_dir,
        show=show,
    )