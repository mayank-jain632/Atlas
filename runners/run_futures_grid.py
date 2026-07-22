"""Parameter-grid sweep for futures strategies.

Reuses runners/run_futures_strats.py's execution engine (run_uid, the
module-level DB_PATH/TIMEFRAME/SOURCE_TIMEFRAME config it reads inside
create_strategy(), and its results/output-dir naming) instead of
duplicating any of that. This script's only job is to build a list of
UIDs and collect their results into one comparison table.

There's no build_uid() helper on the strategy modules (stema.py,
dcemachop.py, psarema.py) -- UID strings are built directly here with an
f-string against the grammar in strategies/futures/uid.py /
strategies/futures/README.md.

Per-combo plotting is disabled (plot_futures_strategy is monkeypatched
to a no-op): a sweep like this doesn't need one PNG per combo, and
rendering each one is a meaningful chunk of every run's wall time.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runners.run_futures_strats as base

base.plot_futures_strategy = lambda **_kwargs: None

# ============================================================
# Fixed across the whole sweep
# ============================================================

SYMBOL = "MES"
ATR_PERIOD = 14
STOP_ATR_MULTIPLE = 10  # Trailing ATR stop -- already validated, not swept.

CAPITAL = 10_000.0
START_DATE = None
END_DATE = None

TIMEFRAME = "4h"
SOURCE_TIMEFRAME = "1h"  # Only "1h" is materialized in this DB; 4h bars
                         # are resampled from it on the fly (see
                         # BaseFuturesStrategy's resampling note).
DB_PATH = PROJECT_ROOT / "duckdb" / "futures_data_1h.duckdb"

# ============================================================
# Indicator parameters being swept
# ============================================================

ST_PERIODS = [7, 10, 14]
ST_MULTS = [2, 3, 4]
EMA_PERIODS = [50, 100, 200]
# add a run grid for ATR from 2-10


def build_stema_uid(
    *,
    symbol: str,
    st_period: int,
    st_mult: float,
    ema_period: int,
    atr_period: int,
    stop_atr_multiple: float,
) -> str:
    return (
        f"stema__s={symbol}"
        f"__st_period={st_period}"
        f"__st_mult={st_mult}"
        f"__ema={ema_period}"
        f"__atr={atr_period}"
        f"__sl_atr={stop_atr_multiple}"
    )


RUNS: list[dict[str, Any]] = [
    {
        "uid": build_stema_uid(
            symbol=SYMBOL,
            st_period=st_period,
            st_mult=st_mult,
            ema_period=ema_period,
            atr_period=ATR_PERIOD,
            stop_atr_multiple=STOP_ATR_MULTIPLE,
        ),
        "capital": CAPITAL,
        "start_date": START_DATE,
        "end_date": END_DATE,
    }
    for st_period in ST_PERIODS
    for st_mult in ST_MULTS
    for ema_period in EMA_PERIODS
]


def run_grid(runs: list[dict[str, Any]]) -> pd.DataFrame:
    base.DB_PATH = DB_PATH
    base.TIMEFRAME = TIMEFRAME
    base.SOURCE_TIMEFRAME = SOURCE_TIMEFRAME

    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, run_config in enumerate(runs, start=1):
        print("\n" + "#" * 76)
        print(f"RUNNING {index} OF {len(runs)}")
        print("#" * 76)

        try:
            output = base.run_uid(
                uid=run_config["uid"],
                capital=run_config["capital"],
                start_date=run_config["start_date"],
                end_date=run_config["end_date"],
            )
            outputs.append(output)
        except Exception as exc:
            failures.append(
                {"uid": run_config["uid"], "error": str(exc)}
            )
            print("\nBACKTEST FAILED")
            print(f"UID:   {run_config['uid']}")
            print(f"Error: {exc}")

    print("\n" + "=" * 76)
    print("ATLAS FUTURES GRID SUMMARY")
    print("=" * 76)
    print(f"Requested:  {len(runs)}")
    print(f"Completed:  {len(outputs)}")
    print(f"Failed:     {len(failures)}")

    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"- {failure['uid']}: {failure['error']}")

    summary_rows = []
    for output in outputs:
        metrics = (output["report"] or {}).get("metrics", {})
        summary_rows.append(
            {
                "uid": output["uid"],
                "trades": len(output["result"]["tradebook"]),
                **metrics,
            }
        )

    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        display_columns = [
            c
            for c in (
                "uid",
                "trades",
                "cagr",
                "sharpe_ratio",
                "max_drawdown",
                "final_equity",
            )
            if c in summary.columns
        ]

        print()
        print("COMPARISON")
        print("-" * 76)
        print(
            summary[display_columns]
            .sort_values("sharpe_ratio", ascending=False)
            .to_string(index=False)
        )

        summary_path = Path(base.OUTPUT_ROOT) / "grid_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"\nFull comparison saved to: {summary_path}")

    print("=" * 76)
    return summary


def main() -> None:
    if not RUNS:
        raise ValueError("RUNS cannot be empty.")

    run_grid(RUNS)


if __name__ == "__main__":
    main()
