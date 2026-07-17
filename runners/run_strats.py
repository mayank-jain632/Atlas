from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.reporting import save_and_print_results
from strategies.buy_hold import BuyHoldIndicatorStrategy, BuyHoldStrategy
from strategies.indicator import IndicatorBasketStrategy
from strategies.momentum import (
    MomentumDiversityStrategy,
    MomentumIndicatorStrategy,
    MomentumStrategy,
)

# Run one or many UIDs.
UIDS = [
    "buy_hold__s=QQQ__w=1",

    # Buy and hold:
    # "buy_hold__s=SPY__w=1",
    # "buy_hold__s=QQQ__w=0.75",

    # Buy and hold with indicator:
    # "buy_hold_indicator__s=QQQ__w=1__state=ma_crossover__fast=50__slow=200__method=sma",
    # "buy_hold_indicator__s=QQQ__w=1__state=supertrend__period=10__multiplier=3",
    # "buy_hold_indicator__s=QQQ__w=1__state=rsi__period=14__bearish=40__bullish=50",

    # Standard momentum:
    # "momentum__u=sp500__sig=price__lb=90__rb=monthly__n=10__alloc=score",
    # "momentum__u=nasdaq100__sig=vol_adj__lb=126__rb=monthly__n=10__alloc=equal",

    # Diversity momentum:
    # "momentum_diversity__u=sp500__sig=price__lb=90__rb=monthly__n=10__div=graph_cut__dlb=60__lam=0.25__alloc=score",

    # Indicator-controlled momentum:
    # "momentum_indicator__u=sp500__sig=price__lb=90__rb=monthly__n=10__alloc=score__filter=SPY__liquidate=true__reenter=true__state=ma_crossover__fast=50__slow=200__method=sma",

    # Indicator basket:
    # "indicator_basket__weights=SPY:0.25,QQQ:0.25,GLD:0.25,TLT:0.25__renorm=false__state=ma_crossover__fast=50__slow=200__method=sma",
    # "indicator_basket__weights=QQQ:0.40,SMH:0.30,GLD:0.20,TLT:0.10__renorm=true__state=supertrend__period=10__multiplier=3",
]

CAPITAL = 100_000.0
START_DATE = "2000-01-01"
END_DATE = None
TIMEFRAME = "1d"
ALLOW_FRACTIONAL_SHARES = True
DB_PATH = None
UNIVERSE_ROOT = "config/universes"
OUTPUT_ROOT = "results"

STRATEGY_CLASSES = {
    "buy_hold": BuyHoldStrategy,
    "buy_hold_indicator": BuyHoldIndicatorStrategy,
    "momentum": MomentumStrategy,
    "momentum_diversity": MomentumDiversityStrategy,
    "momentum_indicator": MomentumIndicatorStrategy,
    "indicator_basket": IndicatorBasketStrategy,
}

MOMENTUM_STRATEGIES = {
    "momentum",
    "momentum_diversity",
    "momentum_indicator",
}


def create_strategy(uid: str, capital: float):
    strategy_name = uid.split("__", 1)[0].strip().lower()

    if strategy_name not in STRATEGY_CLASSES:
        raise ValueError(f"Unknown strategy in UID: {strategy_name}")

    arguments: dict[str, Any] = {
        "uid": uid,
        "capital": capital,
        "db_path": DB_PATH,
        "timeframe": TIMEFRAME,
        "allow_fractional_shares": ALLOW_FRACTIONAL_SHARES,
    }

    # Infrastructure path, not a strategy parameter.
    if strategy_name in MOMENTUM_STRATEGIES:
        arguments["universe_root"] = UNIVERSE_ROOT

    strategy = STRATEGY_CLASSES[strategy_name](**arguments)
    return strategy_name, strategy


def run_uid(
    uid: str,
    capital: float,
    start_date: str | None,
    end_date: str | None,
):
    strategy_name, strategy = create_strategy(uid=uid, capital=capital)
    symbols = strategy.required_symbols()

    print("\n" + "=" * 76)
    print("ATLAS UID BACKTEST")
    print("=" * 76)
    print(f"Strategy:    {strategy_name}")
    print(f"UID:         {uid}")
    print(f"Capital:     ${capital:,.2f}")
    print(f"Symbols:     {len(symbols)}")
    print(f"Start date:  {start_date}")
    print(f"End date:    {end_date}")
    print("=" * 76)

    result = strategy.run(
        symbols=symbols,
        start=start_date,
        end=end_date,
    )

    output_dir = Path(OUTPUT_ROOT) / strategy_name / uid

    report = save_and_print_results(
        strategy=strategy,
        result=result,
        output_dir=output_dir,
        metadata=getattr(strategy, "parameters", None),
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
    print("ATLAS MULTI-UID SUMMARY")
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
