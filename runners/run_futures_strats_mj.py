from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.reporting import save_and_print_results
from strategies.futures.factory import create_futures_strategy

# ============================================================
# Configuration
# ============================================================

# Run one or many UIDs. See strategies/futures/README.md for the full UID
# grammar of each strategy.
UIDS = [
    "dcemachop__s=MES__dc=20__ema=200__adx_period=14__adx=25__chop_period=14__chop=35__atr=14__sl_atr=2",

    # Supertrend + EMA:
    # "stema__s=MES__st_period=10__st_mult=3__ema=200__atr=14__sl_atr=2",

    # Parabolic SAR + EMA:
    # "psarema__s=MNQ__psar_step=0.02__psar_max=0.20__ema=200__atr=14__sl_atr=2",

    # Donchian + EMA + ADX + Choppiness:
    # "dcemachop__s=MGC__dc=20__ema=200__adx_period=14__adx=25__chop_period=14__chop=35__atr=14__sl_atr=2",
]

CAPITAL = 10_000.0
START_DATE = None
END_DATE = None


DB_PATH = PROJECT_ROOT / "duckdb" / "futures_data_1h.duckdb"
# DB_PATH = PROJECT_ROOT / "duckdb" / "market_data.duckdb"

SOURCE_TIMEFRAME = "1h"
TIMEFRAME = "6h"

OUTPUT_ROOT = "results/futures"


def create_strategy(uid: str, capital: float):
    strategy_name = uid.split("__", 1)[0].strip().lower()

    strategy = create_futures_strategy(
        uid=uid,
        capital=capital,
        db_path=DB_PATH,
        timeframe=TIMEFRAME,
        source_timeframe=SOURCE_TIMEFRAME,
    )

    return strategy_name, strategy


def _periods_per_year(equity: pd.DataFrame | None) -> int:
    """Infer the annualization factor from the actual bar spacing in the
    equity curve instead of assuming daily bars -- these strategies can run
    on either 1d or 1m data depending on DB_PATH/TIMEFRAME above, and a
    fixed 252 would badly understate volatility/Sharpe on 1-minute bars."""
    if equity is None or equity.empty or len(equity) < 2:
        return 252

    timestamps = pd.to_datetime(equity["timestamp"])
    elapsed_years = (
        timestamps.iloc[-1] - timestamps.iloc[0]
    ).total_seconds() / (365.25 * 24 * 3600)

    if elapsed_years <= 0:
        return 252

    return max(1, round(len(equity) / elapsed_years))


def run_uid(
    uid: str,
    capital: float,
    start_date: str | None,
    end_date: str | None,
):
    strategy_name, strategy = create_strategy(uid=uid, capital=capital)
    symbols = strategy.required_symbols()

    print("\n" + "=" * 76)
    print("ATLAS FUTURES UID BACKTEST")
    print("=" * 76)
    print(f"Strategy:    {strategy_name}")
    print(f"UID:         {uid}")
    print(f"Data symbol: {strategy.data_symbol}")
    print(f"DB path:     {DB_PATH}")
    print(f"Timeframe:   {TIMEFRAME}")
    print(f"Capital:     ${capital:,.2f}")
    print(f"Start date:  {start_date}")
    print(f"End date:    {end_date}")
    print("=" * 76)

    print()

    result = strategy.run(
        symbols=symbols,
        start=start_date,
        end=end_date,
    )

    # Unique per (uid, timeframe, source_timeframe, db) -- the UID string
    # alone doesn't capture which db/timeframe combo produced it, so two
    # runs of the identical UID at different timeframes/sources would
    # otherwise silently overwrite each other's saved results.
    db_label = Path(DB_PATH).stem
    config_label = f"tf={TIMEFRAME}__src={SOURCE_TIMEFRAME}__db={db_label}"
    output_dir = Path(OUTPUT_ROOT) / strategy_name / uid / config_label

    report = save_and_print_results(
        strategy=strategy,
        result=result,
        output_dir=output_dir,
        metadata={
            **strategy.parameters,
            "run_timeframe": TIMEFRAME,
            "run_source_timeframe": SOURCE_TIMEFRAME,
            "run_db": db_label,
        },
        periods_per_year=_periods_per_year(result.get("equity")),
    )

    return {
        "uid": uid,
        "strategy_name": strategy_name,
        "strategy": strategy,
        "symbols": symbols,
        "result": result,
        "report": report,
        "output_dir": output_dir,
    }


def run_uids(uids: list[str]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, uid in enumerate(uids, start=1):
        print("\n" + "#" * 76)
        print(f"RUNNING UID {index} OF {len(uids)}")
        print("#" * 76)

        try:
            outputs.append(
                run_uid(
                    uid=uid,
                    capital=CAPITAL,
                    start_date=START_DATE,
                    end_date=END_DATE,
                )
            )
        except Exception as exc:
            failures.append({"uid": uid, "error": str(exc)})
            print("\nBACKTEST FAILED")
            print(f"UID:   {uid}")
            print(f"Error: {exc}")

    print("\n" + "=" * 76)
    print("ATLAS FUTURES MULTI-UID SUMMARY")
    print("=" * 76)
    print(f"Requested:  {len(uids)}")
    print(f"Completed:  {len(outputs)}")
    print(f"Failed:     {len(failures)}")

    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"- {failure['uid']}: {failure['error']}")

    print("=" * 76)
    return outputs


def main() -> None:
    if not UIDS:
        raise ValueError("UIDS cannot be empty.")

    run_uids(UIDS)


if __name__ == "__main__":
    main()
