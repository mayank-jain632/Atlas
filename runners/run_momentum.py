from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.reporting import save_and_print_results
from strategies.momentum import (
    MomentumDiversityStrategy,
    MomentumIndicatorStrategy,
    MomentumStrategy,
)

# Select one UID.
UID = (
    "momentum"
    "__u=custom"
    "__sig=price"
    "__lb=90"
    "__rb=monthly"
    "__n=10"
    "__alloc=score"
)

# Standard momentum examples:
# UID = "momentum__u=sp500__sig=price__lb=60__rb=monthly__n=10__alloc=equal"
# UID = "momentum__u=nasdaq100__sig=price__lb=126__rb=monthly__n=10__alloc=score"
# UID = "momentum__u=dow30__sig=price__lb=252__rb=quarterly__n=5__alloc=equal"
# UID = "momentum__u=sp500__sig=rsi__lb=90__rsiw=14__rsit=50__rb=monthly__n=10__alloc=score"
# UID = "momentum__u=nasdaq100__sig=ma_cross__lb=90__short=40__long=100__rb=monthly__n=10__alloc=score"
# UID = "momentum__u=sp500__sig=vol_adj__lb=90__rb=monthly__n=10__alloc=score"
# UID = "momentum__u=sp500__sig=low_vol__lb=90__rb=monthly__n=10__alloc=equal"
# UID = "momentum__u=sp500__sig=trend_quality__lb=90__rb=monthly__n=10__alloc=score"

# Diversity examples:
# UID = "momentum_diversity__u=sp500__sig=price__lb=90__rb=monthly__n=10__div=graph_cut__dlb=60__lam=0.25__alloc=score"
# UID = "momentum_diversity__u=sp500__sig=price__lb=126__rb=monthly__n=10__div=facility_location__dlb=60__lam=0.25__alloc=equal"

# Indicator-controlled momentum examples:
#UID = "momentum_indicator__u=sp500__sig=price__lb=90__rb=monthly__n=10__alloc=score__filter=SPY__liquidate=true__reenter=true__initial=UNKNOWN__state=ma_crossover__fast=50__slow=200__method=sma"
UID = "momentum_indicator__u=custom__sig=price__lb=90__rb=monthly__n=10__alloc=score__filter=SPY__liquidate=true__reenter=true__initial=UNKNOWN__state=ma_crossover__fast=50__slow=200__method=sma"
#UID = "momentum_indicator__u=nasdaq100__sig=price__lb=90__rb=monthly__n=10__alloc=score__filter=QQQ__liquidate=true__reenter=true__state=supertrend__period=10__multiplier=3"

CAPITAL = 100_000.0
START_DATE = "1996-01-01"
END_DATE = None
TIMEFRAME = "1d"
ALLOW_FRACTIONAL_SHARES = True
DB_PATH = None
UNIVERSE_ROOT = "config/universes"
OUTPUT_ROOT = "results/momentum"

STRATEGY_CLASSES = {
    "momentum": MomentumStrategy,
    "momentum_diversity": MomentumDiversityStrategy,
    "momentum_indicator": MomentumIndicatorStrategy,
}


def create_strategy():
    strategy_name = UID.split("__", 1)[0].strip().lower()

    if strategy_name not in STRATEGY_CLASSES:
        raise ValueError(f"Unsupported momentum UID: {strategy_name}")

    strategy = STRATEGY_CLASSES[strategy_name](
        uid=UID,
        capital=CAPITAL,
        db_path=DB_PATH,
        timeframe=TIMEFRAME,
        allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
        universe_root=UNIVERSE_ROOT,
    )

    return strategy_name, strategy


def main() -> None:
    strategy_name, strategy = create_strategy()
    symbols = strategy.required_symbols()

    print("\n" + "=" * 76)
    print("ATLAS MOMENTUM BACKTEST")
    print("=" * 76)
    print(f"UID:            {UID}")
    print(f"Strategy:       {strategy_name}")
    print(f"Universe:       {strategy.universe_name}")
    print(f"Universe size:  {len(strategy.stock_universe)}")
    print(f"Signal:         {strategy.parameters['signal']}")

    if strategy_name == "momentum_indicator":
        print(f"Filter symbol:  {strategy.filter_symbol}")
        print(f"Filter state:   {strategy.parameters['state_type']}")

    print(f"Start date:     {START_DATE}")
    print(f"End date:       {END_DATE}")
    print("=" * 76)

    result = strategy.run(
        symbols=symbols,
        start=START_DATE,
        end=END_DATE,
    )

    output_dir = Path(OUTPUT_ROOT) / strategy_name / UID

    save_and_print_results(
        strategy=strategy,
        result=result,
        output_dir=output_dir,
        metadata=strategy.parameters,
    )


if __name__ == "__main__":
    main()
