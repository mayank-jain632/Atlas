from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.run_strats import run_uid

UID = "buy_hold__s=QQQ__w=1"
# UID = "buy_hold_indicator__s=QQQ__w=1__state=ma_crossover__fast=50__slow=200__method=sma"
# UID = "momentum__u=sp500__sig=price__lb=90__rb=monthly__n=10__alloc=score"

CAPITAL = 100_000.0
START_DATE = "2000-01-01"
END_DATE = None


def main() -> None:
    run_uid(
        uid=UID,
        capital=CAPITAL,
        start_date=START_DATE,
        end_date=END_DATE,
    )


if __name__ == "__main__":
    main()
