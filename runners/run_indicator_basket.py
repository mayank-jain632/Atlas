from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.reporting import save_and_print_results
from strategies.indicator import IndicatorBasketStrategy

# Select one UID.
UID = "indicator_basket__weights=SPY:0.25,QQQ:0.25,GLD:0.25,SLV:0.1,SMH:0.15__renorm=false__state=ma_crossover__fast=50__slow=200__method=sma"

# UID = "indicator_basket__weights=SPY:0.25,QQQ:0.25,GLD:0.25,TLT:0.25__renorm=false__state=ma_crossover__fast=50__slow=200__method=sma"
# UID = "indicator_basket__weights=SPY:0.25,QQQ:0.25,GLD:0.25,TLT:0.25__renorm=true__state=ma_crossover__fast=50__slow=200__method=sma"
# UID = "indicator_basket__weights=QQQ:0.40,SMH:0.30,GLD:0.20,TLT:0.10__renorm=false__state=supertrend__period=10__multiplier=3"
# UID = "indicator_basket__weights=SPY:0.40,QQQ:0.40,GLD:0.20__renorm=false__state=rsi__period=14__bearish=40__bullish=50"
# UID = "indicator_basket__weights=SPY:0.25,QQQ:0.25,GLD:0.25,TLT:0.25__renorm=false__state=donchian__exit_period=50__entry_period=20"

CAPITAL = 100_000.0
START_DATE = "2000-01-01"
END_DATE = None
TIMEFRAME = "1d"
ALLOW_FRACTIONAL_SHARES = True
DB_PATH = None
OUTPUT_ROOT = "results/indicator_basket"


def main() -> None:
    strategy = IndicatorBasketStrategy(
        uid=UID,
        capital=CAPITAL,
        db_path=DB_PATH,
        timeframe=TIMEFRAME,
        allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
    )

    symbols = strategy.required_symbols()

    print("\n" + "=" * 76)
    print("ATLAS INDICATOR BASKET BACKTEST")
    print("=" * 76)
    print(f"UID:          {UID}")
    print(f"Basket:       {strategy.target_weights}")
    print(f"Indicator:    {strategy.parameters['state_type']}")
    print(f"Parameters:   {strategy.parameters['state_parameters']}")
    print(f"Renormalize:  {strategy.renormalize_bullish_weights}")
    print(f"Start date:   {START_DATE}")
    print(f"End date:     {END_DATE}")
    print("=" * 76)

    result = strategy.run(
        symbols=symbols,
        start=START_DATE,
        end=END_DATE,
    )

    output_dir = Path(OUTPUT_ROOT) / UID

    save_and_print_results(
        strategy=strategy,
        result=result,
        output_dir=output_dir,
        metadata=strategy.parameters,
    )


if __name__ == "__main__":
    main()
