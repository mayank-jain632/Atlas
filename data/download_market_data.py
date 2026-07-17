from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from data.universe import UniverseManager
from data.yahoo_downloader import YahooDownloader


# ============================================================
# Configuration
# ============================================================

UNIVERSE_ROOT = (
    PROJECT_ROOT
    / "config"
    / "universes"
)

UNIVERSES = [
    "sp500",
    "nasdaq100",
    "dow30",
    "etfs",
    "futures",
]

START_DATE = "1990-01-01"
END_DATE = None

INTERVAL = "1d"

AUTO_ADJUST = True

# False:
#     Download the entire requested period.
#
# True:
#     Start near the most recent DuckDB timestamp for each
#     symbol and update only recent data.
INCREMENTAL = False

PAUSE_SECONDS = 0.0


def main() -> None:
    manager = UniverseManager(
        root=UNIVERSE_ROOT
    )

    symbols = []

    for universe_name in UNIVERSES:
        universe_symbols = manager.load(
            universe_name
        )

        print(
            f"{universe_name}: "
            f"{len(universe_symbols)} symbols"
        )

        symbols.extend(
            universe_symbols
        )

    symbols = list(
        dict.fromkeys(symbols)
    )

    print()
    print(
        f"Total unique symbols: {len(symbols)}"
    )

    with YahooDownloader() as downloader:
        downloader.download(
            symbols=symbols,
            start=START_DATE,
            end=END_DATE,
            interval=INTERVAL,
            auto_adjust=AUTO_ADJUST,
            incremental=INCREMENTAL,
            pause_seconds=PAUSE_SECONDS,
        )


if __name__ == "__main__":
    main()