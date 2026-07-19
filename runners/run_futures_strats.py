from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORTS
# ============================================================

from runners.reporting import save_and_print_results
from strategies.futures.factory import create_futures_strategy
from runners.futures_plotting import (
    plot_futures_strategy,
)

# ============================================================
# BACKTEST CONFIGURATION
# ============================================================

# Run one or multiple futures strategy UIDs.
UIDS = [
    # Supertrend flip + EMA confirmation
    "stema__s=MES__st_period=10__st_mult=3__ema=50__atr=14__sl_atr=10"

    # PSAR flip + EMA confirmation
    # "psarema__s=MNQ__psar_step=0.02__psar_max=0.20__ema=200__atr=14__sl_atr=2",

    # Donchian breakout + EMA + ADX + Choppiness
    # "dcemachop__s=MGC__dc=20__ema=200__adx_period=14__adx=25__chop_period=14__chop=35__atr=14__sl_atr=2",
]

CAPITAL = 10_000.0

# START_DATE = "2000-01-01"
START_DATE = None
END_DATE = None

TIMEFRAME = "1h"

DB_PATH = PROJECT_ROOT / "duckdb" / "futures_data_1h.duckdb"

OUTPUT_ROOT = Path(
    "results/futures"
)


# ============================================================
# CREATE STRATEGY
# ============================================================

def create_strategy(
    uid: str,
    capital: float,
):
    """
    Create one futures strategy from its UID.

    Supported strategy prefixes:
        stema
        psarema
        dcemachop
    """

    strategy_name = (
        uid.split("__", 1)[0]
        .strip()
        .lower()
    )

    strategy = create_futures_strategy(
        uid=uid,
        capital=capital,
        db_path=DB_PATH,
        timeframe=TIMEFRAME,
    )

    return strategy_name, strategy


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_result(
    result: dict[str, pd.DataFrame],
) -> None:
    """
    Validate the standard futures backtest outputs.
    """

    required_outputs = {
        "tradebook",
        "equity",
        "positions",
        "signals",
    }

    missing = (
        required_outputs
        - set(result)
    )

    if missing:
        raise ValueError(
            "Futures strategy result is missing: "
            + ", ".join(sorted(missing))
        )

    for name in required_outputs:
        if not isinstance(
            result[name],
            pd.DataFrame,
        ):
            raise TypeError(
                f"result['{name}'] must be "
                "a pandas DataFrame."
            )

    equity = result["equity"]

    if equity.empty:
        raise ValueError(
            "The futures backtest produced "
            "an empty equity curve."
        )

    required_equity_columns = {
        "timestamp",
        "equity",
        "daily_pnl",
        "cumulative_pnl",
    }

    missing_equity_columns = (
        required_equity_columns
        - set(equity.columns)
    )

    if missing_equity_columns:
        raise ValueError(
            "Equity output is missing: "
            + ", ".join(
                sorted(
                    missing_equity_columns
                )
            )
        )


# ============================================================
# PRINT FUTURES SUMMARY
# ============================================================

def print_futures_summary(
    *,
    uid: str,
    strategy: Any,
    result: dict[str, pd.DataFrame],
) -> None:
    """
    Print a small futures-specific validation summary.

    This is useful even when the general reporting module
    also prints portfolio metrics.
    """

    equity = result["equity"].copy()
    tradebook = result["tradebook"]
    signals = result["signals"]
    positions = result["positions"]

    initial_equity = float(
        equity["equity"].iloc[0]
    )

    final_equity = float(
        equity["equity"].iloc[-1]
    )

    net_pnl = (
        final_equity
        - initial_equity
    )

    total_return = (
        net_pnl / initial_equity
        if initial_equity != 0
        else float("nan")
    )

    peak = equity["equity"].cummax()

    drawdown = (
        equity["equity"] / peak
        - 1.0
    )

    max_drawdown = float(
        drawdown.min()
    )

    entry_actions = {
        "BUY",
        "SELL_SHORT",
    }

    exit_actions = {
        "SELL",
        "BUY_TO_COVER",
    }

    entries = (
        tradebook["action"].isin(
            entry_actions
        ).sum()
        if (
            not tradebook.empty
            and "action" in tradebook.columns
        )
        else 0
    )

    exits = (
        tradebook["action"].isin(
            exit_actions
        ).sum()
        if (
            not tradebook.empty
            and "action" in tradebook.columns
        )
        else 0
    )

    stop_exits = (
        (
            tradebook["reason"]
            == "ATR_STOP"
        ).sum()
        if (
            not tradebook.empty
            and "reason" in tradebook.columns
        )
        else 0
    )

    print("\n" + "-" * 76)
    print("FUTURES VALIDATION SUMMARY")
    print("-" * 76)

    print(
        f"UID:                 {uid}"
    )

    print(
        f"Contract:            "
        f"{strategy.symbol}"
    )

    print(
        f"Data symbol:         "
        f"{strategy.data_symbol}"
    )

    print(
        f"Multiplier:          "
        f"{strategy.instrument.multiplier:,.4f}"
    )

    print(
        f"Initial equity:      "
        f"${initial_equity:,.2f}"
    )

    print(
        f"Final equity:        "
        f"${final_equity:,.2f}"
    )

    print(
        f"Net P&L:             "
        f"${net_pnl:,.2f}"
    )

    print(
        f"Total return:        "
        f"{total_return:.2%}"
    )

    print(
        f"Maximum drawdown:    "
        f"{max_drawdown:.2%}"
    )

    print(
        f"Tradebook rows:      "
        f"{len(tradebook):,}"
    )

    print(
        f"Entries:             "
        f"{int(entries):,}"
    )

    print(
        f"Exits:               "
        f"{int(exits):,}"
    )

    print(
        f"ATR stop exits:      "
        f"{int(stop_exits):,}"
    )

    print(
        f"Signal rows:         "
        f"{len(signals):,}"
    )

    if not positions.empty:
        final_direction = int(
            positions[
                "direction"
            ].iloc[-1]
        )

        final_contracts = int(
            positions[
                "contracts"
            ].iloc[-1]
        )

        print(
            f"Final direction:     "
            f"{final_direction}"
        )

        print(
            f"Final contracts:     "
            f"{final_contracts}"
        )

    print("-" * 76)


# ============================================================
# RUN ONE UID
# ============================================================

def run_uid(
    *,
    uid: str,
    capital: float,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    """
    Run one futures UID and save its results.
    """

    strategy_name, strategy = (
        create_strategy(
            uid=uid,
            capital=capital,
        )
    )

    symbols = (
        strategy.required_symbols()
    )

    print("\n" + "=" * 76)
    print("ATLAS FUTURES BACKTEST")
    print("=" * 76)

    print(
        f"Strategy:       "
        f"{strategy_name}"
    )

    print(
        f"UID:            "
        f"{uid}"
    )

    print(
        f"Contract:       "
        f"{strategy.symbol}"
    )

    print(
        f"Data symbol:    "
        f"{strategy.data_symbol}"
    )

    print(
        f"Multiplier:     "
        f"{strategy.instrument.multiplier}"
    )

    print(
        f"Capital:        "
        f"${capital:,.2f}"
    )

    print(
        f"Start date:     "
        f"{start_date}"
    )

    print(
        f"End date:       "
        f"{end_date}"
    )

    print(
        f"Timeframe:      "
        f"{TIMEFRAME}"
    )

    print("=" * 76)

    result = strategy.run(
        symbols=symbols,
        start=start_date,
        end=end_date,
    )

    validate_result(
        result
    )

    output_dir = (
        OUTPUT_ROOT
        / strategy_name
        / uid
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The futures base class already has a dedicated
    # method for saving its raw outputs.
    strategy.save_results(
        output_dir
    )
    plot_path = plot_futures_strategy(
        strategy_name=strategy_name,
        result=result,
        uid=uid,
        output_dir=output_dir,
        show=False,
    )

    print(
        f"Strategy plot:  {plot_path}"
    )
    print_futures_summary(
        uid=uid,
        strategy=strategy,
        result=result,
    )

    # Reuse the common Atlas reporting system.
    #
    # This should work as long as reporting primarily uses
    # result["equity"] and its "equity" column.
    #
    # If the current reporting module assumes stock-specific
    # tradebook fields, the raw futures CSV files have already
    # been saved above and only this reporting call will fail.
    try:
        report = save_and_print_results(
            strategy=strategy,
            result=result,
            output_dir=output_dir,
            metadata=getattr(
                strategy,
                "parameters",
                None,
            ),
        )

    except Exception as exc:
        report = None

        print(
            "\nGeneral reporting failed, "
            "but the futures backtest and raw "
            "outputs were saved successfully."
        )

        print(
            f"Reporting error: {exc}"
        )

    return {
        "uid": uid,
        "strategy_name": strategy_name,
        "strategy": strategy,
        "symbols": symbols,
        "result": result,
        "report": report,
        "plot_path": plot_path,
        "output_dir": output_dir,
    }


# ============================================================
# RUN MULTIPLE UIDS
# ============================================================

def run_uids(
    uids: list[str],
) -> list[dict[str, Any]]:
    """
    Run multiple futures UIDs independently.

    Each UID receives the full CAPITAL amount because these are
    independent strategy backtests, not a combined portfolio.
    """

    outputs: list[
        dict[str, Any]
    ] = []

    failures: list[
        dict[str, str]
    ] = []

    for index, uid in enumerate(
        uids,
        start=1,
    ):
        print("\n" + "#" * 76)

        print(
            f"RUNNING FUTURES UID "
            f"{index} OF {len(uids)}"
        )

        print("#" * 76)

        try:
            output = run_uid(
                uid=uid,
                capital=CAPITAL,
                start_date=START_DATE,
                end_date=END_DATE,
            )

            outputs.append(
                output
            )

        except Exception as exc:
            failures.append(
                {
                    "uid": uid,
                    "error": str(exc),
                }
            )

            print("\nBACKTEST FAILED")

            print(
                f"UID:   {uid}"
            )

            print(
                f"Error: {exc}"
            )

    print("\n" + "=" * 76)
    print("ATLAS FUTURES MULTI-UID SUMMARY")
    print("=" * 76)

    print(
        f"Requested:  {len(uids)}"
    )

    print(
        f"Completed:  {len(outputs)}"
    )

    print(
        f"Failed:     {len(failures)}"
    )

    if failures:
        print("\nFAILURES")

        for failure in failures:
            print(
                f"- {failure['uid']}: "
                f"{failure['error']}"
            )

    print("=" * 76)

    return outputs


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not UIDS:
        raise ValueError(
            "UIDS cannot be empty."
        )

    run_uids(
        UIDS
    )


if __name__ == "__main__":
    main()