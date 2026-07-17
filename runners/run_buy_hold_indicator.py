from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.reporting import save_and_print_results
from strategies.buy_hold import BuyHoldIndicatorStrategy

# Select one UID.
UID = (
    "buy_hold_indicator"
    "__s=QQQ"
    "__w=1"
    "__initial=UNKNOWN"
    "__state=ma_crossover"
    "__fast=50"
    "__slow=200"
    "__method=sma"
)

# UID = "buy_hold_indicator__s=QQQ__w=1__state=sma__period=200"
# UID = "buy_hold_indicator__s=QQQ__w=1__state=ema__period=200"
# UID = "buy_hold_indicator__s=GLD__w=1__state=ma_crossover__fast=20__slow=100__method=sma"
# UID = "buy_hold_indicator__s=QQQ__w=1__state=supertrend__period=10__multiplier=3"
# UID = "buy_hold_indicator__s=QQQ__w=1__state=rsi__period=14__bearish=40__bullish=50"
# UID = "buy_hold_indicator__s=GLD__w=1__state=macd__fast=12__slow=26__signal_period=9"
# UID = "buy_hold_indicator__s=QQQ__w=1__state=donchian__exit_period=50__entry_period=20"
# UID = "buy_hold_indicator__s=QQQ__w=1__state=adx__period=14__bearish=15__bullish=20"
# UID = "buy_hold_indicator__s=QQQ__w=1__state=psar__step=0.02__max_step=0.20"
# UID = "buy_hold_indicator__s=QQQ__w=1__state=drawdown_recovery__state_lookback=252__bearish_drawdown=0.10__bullish_drawdown=0.05__recovery_ma_period=20"
# UID = "buy_hold_indicator__s=QQQ__w=1__state=choppiness__period=14__ma_period=50__bullish=38.2__bearish=61.8"

CAPITAL = 100_000.0
START_DATE = "2000-01-01"
END_DATE = None
TIMEFRAME = "1d"
ALLOW_FRACTIONAL_SHARES = True
DB_PATH = None
OUTPUT_ROOT = "results/buy_hold_indicator"


def main() -> None:
    strategy = BuyHoldIndicatorStrategy(
        uid=UID,
        capital=CAPITAL,
        db_path=DB_PATH,
        timeframe=TIMEFRAME,
        allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
    )

    symbols = strategy.required_symbols()

    print("\n" + "=" * 76)
    print("ATLAS BUY-HOLD INDICATOR BACKTEST")
    print("=" * 76)
    print(f"UID:             {UID}")
    print(f"Symbol:          {strategy.symbol}")
    print(f"Target percent:  {strategy.target_allocation:.2%}")
    print(f"Indicator:       {strategy.parameters['state_type']}")
    print(f"Parameters:      {strategy.parameters['state_parameters']}")
    print(f"Start date:      {START_DATE}")
    print(f"End date:        {END_DATE}")
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
