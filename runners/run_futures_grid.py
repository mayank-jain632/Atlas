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

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
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


# ============================================================
# Parallel execution
# ============================================================
#
# Each combo is an independent backtest, so the sweep is embarrassingly
# parallel. Processes (not threads) because the per-bar strategy/indicator
# work is CPU-bound Python -- threads would just serialize on the GIL.

PARALLEL = True

# Deliberately NOT (cpu_count - 1). Measured on this 10-core machine: running
# 9 of these backtests concurrently (there's real per-bar CPU work, not just
# I/O) roughly DOUBLES each one's wall time versus running alone (~150s vs
# ~77s), and sustaining that many concurrent processes for the length of a
# full 27-combo grid made the total slower than running sequentially. Running
# only 3 concurrently added just ~9% overhead per task. Staying well under
# the core count leaves headroom instead of saturating every core.
MAX_WORKERS = max(1, (os.cpu_count() or 2) // 2 - 1)


def _init_worker(
    db_path: Path,
    timeframe: str,
    source_timeframe: str,
) -> None:
    """
    Runs once per worker process at pool start-up.

    ProcessPoolExecutor's default ("spawn") start method gives each worker
    a fresh interpreter that re-imports this module from scratch, so it
    does NOT inherit the DB_PATH/TIMEFRAME/SOURCE_TIMEFRAME this file sets
    on `base` dynamically inside run_grid() in the parent process -- those
    assignments only ever ran in the parent. Re-apply them here instead.
    (base.plot_futures_strategy is a module-level assignment made at import
    time above, so it's already reapplied for free by the reimport -- this
    just makes that explicit rather than relying on it.)
    """

    base.DB_PATH = db_path
    base.TIMEFRAME = timeframe
    base.SOURCE_TIMEFRAME = source_timeframe
    base.plot_futures_strategy = lambda **_kwargs: None


def _run_one(run_config: dict[str, Any]) -> dict[str, Any]:
    """
    Run one UID and return a small summary row.

    Deliberately not the full tradebook/equity/signals DataFrames: those
    are already written to CSV by run_uid()/save_results(), and pickling
    them back across the process boundary for every combo would be pure
    overhead.
    """

    uid = run_config["uid"]

    try:
        output = base.run_uid(
            uid=uid,
            capital=run_config["capital"],
            start_date=run_config["start_date"],
            end_date=run_config["end_date"],
        )
    except Exception as exc:
        return {"uid": uid, "error": str(exc)}

    metrics = (output["report"] or {}).get("metrics", {})

    return {
        "uid": uid,
        "trades": len(output["result"]["tradebook"]),
        "error": None,
        **metrics,
    }


def run_grid(runs: list[dict[str, Any]]) -> pd.DataFrame:
    base.DB_PATH = DB_PATH
    base.TIMEFRAME = TIMEFRAME
    base.SOURCE_TIMEFRAME = SOURCE_TIMEFRAME

    print("\n" + "=" * 76)
    print("ATLAS FUTURES GRID")
    print("=" * 76)
    print(f"Runs:        {len(runs)}")
    print(
        f"Parallel:    {PARALLEL} (max_workers={MAX_WORKERS})"
        if PARALLEL
        else "Parallel:    False (sequential)"
    )
    print("=" * 76)

    rows: list[dict[str, Any]] = []

    if PARALLEL and len(runs) > 1:
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=_init_worker,
            initargs=(DB_PATH, TIMEFRAME, SOURCE_TIMEFRAME),
        ) as executor:
            futures = {
                executor.submit(_run_one, run_config): run_config["uid"]
                for run_config in runs
            }

            for completed, future in enumerate(
                as_completed(futures), start=1
            ):
                uid = futures[future]
                row = future.result()

                status = "FAILED" if row.get("error") else "done"
                print(f"[{completed}/{len(runs)}] {status:6s} {uid}")

                if row.get("error"):
                    print(f"    Error: {row['error']}")

                rows.append(row)
    else:
        for index, run_config in enumerate(runs, start=1):
            print(f"\n[{index}/{len(runs)}] running {run_config['uid']}")
            rows.append(_run_one(run_config))

    failures = [row for row in rows if row.get("error")]
    outputs = [row for row in rows if not row.get("error")]

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

    summary = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "error"}
            for row in outputs
        ]
    )

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
