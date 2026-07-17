from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.reporting import save_and_print_results
from strategies.buy_hold import BuyHoldStrategy

# Select one UID.
UID = "buy_hold__s=QQQ__w=1"
# UID = "buy_hold__s=SPY__w=1"
# UID = "buy_hold__s=QQQ__w=0.75"
# UID = "buy_hold__s=GLD__w=0.50"
# UID = "buy_hold__s=TLT__w=0.50"

CAPITAL = 100_000.0
START_DATE = "2000-01-01"
END_DATE = None
TIMEFRAME = "1d"
ALLOW_FRACTIONAL_SHARES = True
DB_PATH = None
OUTPUT_ROOT = "results/buy_hold"


def main() -> None:
    strategy = BuyHoldStrategy(
        uid=UID,
        capital=CAPITAL,
        db_path=DB_PATH,
        timeframe=TIMEFRAME,
        allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
    )

    symbols = strategy.required_symbols()

    print("\n" + "=" * 76)
    print("ATLAS BUY-AND-HOLD BACKTEST")
    print("=" * 76)
    print(f"UID:             {UID}")
    print(f"Symbol:          {strategy.symbol}")
    print(f"Target percent:  {strategy.target_allocation:.2%}")
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
