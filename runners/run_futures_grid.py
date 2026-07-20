"""Parameter-grid sweep for futures strategies.

Reuses runners/run_futures_strats.py's execution engine (run_uid,
_periods_per_year, output-dir naming) instead of duplicating it -- this
script's only job is to build a list of run configs and collect their
results into one comparison table. Each strategy builds its own UIDs via
the build_uid() helper living alongside it (strategies/futures/dcemachop.py,
stema.py, psarema.py), so the parameter grammar for each strategy only
has one place it can drift out of sync with strategies/futures/uid.py.

See strategies/futures/README.md for the UID grammar and data sources.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runners.run_futures_strats_mj as base
from strategies.futures.dcemachop import build_uid as build_dcemachop_uid
from strategies.futures.stema import build_uid as build_stema_uid
from strategies.futures.psarema import build_uid as build_psarema_uid

# ============================================================
# Data sources (same three as run_futures_strats.py)
# ============================================================

DAILY_ROOT = dict(db_path=PROJECT_ROOT / "duckdb" / "market_data.duckdb", source_timeframe="1d")
MICRO_1M = dict(db_path=PROJECT_ROOT / "duckdb" / "futures_data.duckdb", source_timeframe="1m")
MICRO_1H = dict(db_path=PROJECT_ROOT / "duckdb" / "futures_data_1h.duckdb", source_timeframe="1h")

# ============================================================
# Grid to run
# ============================================================
# Each entry fully specifies one backtest. Build this list by hand or with
# a loop -- a loop only needs to vary axes that stay valid together (you
# can't resample "1d" bars up to "1h", so don't cross a data source with a
# timeframe it can't produce). Example ADX/Chop sweep at a fixed timeframe:
#
#   RUNS = [
#       {
#           "uid": build_dcemachop_uid(symbol="CL", adx_threshold=adx, chop_threshold=chop),
#           **DAILY_ROOT, "timeframe": "1d", "capital": 50_000.0,
#           "start_date": None, "end_date": None,
#       }
#       for adx in (20, 25, 30)
#       for chop in (30, 35, 40, 45)
#   ]

RUNS: list[dict[str, Any]] = [
    # Priority check: same dcemachop parameters, native daily bars on the
    # full-size root contract -- the direct apples-to-apples comparison
    # against a "documented on daily bars" reference.
    {
        "uid": build_dcemachop_uid(symbol="CL"),
        **DAILY_ROOT,
        "timeframe": "1d",
        "capital": 50_000.0,
        "start_date": None,
        "end_date": None,
    },
    # The same parameters as-is on 1h micro bars (what we've been running)
    # -- kept side by side so the comparison table shows the gap directly.
    {
        "uid": build_dcemachop_uid(symbol="MCL"),
        **MICRO_1H,
        "timeframe": "1h",
        "capital": 50_000.0,
        "start_date": "2025-01-01",
        "end_date": None,
    },
]


def run_grid(runs: list[dict[str, Any]]) -> pd.DataFrame:
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, run_config in enumerate(runs, start=1):
        print("\n" + "#" * 76)
        print(f"RUNNING {index} OF {len(runs)}")
        print("#" * 76)

        # run_futures_strats.run_uid()/create_strategy() read DB_PATH/
        # TIMEFRAME/SOURCE_TIMEFRAME as module-level config rather than
        # function parameters -- set them here per grid entry instead of
        # duplicating the run/report logic in this file.
        base.DB_PATH = run_config["db_path"]
        base.TIMEFRAME = run_config["timeframe"]
        base.SOURCE_TIMEFRAME = run_config["source_timeframe"]

        try:
            output = base.run_uid(
                uid=run_config["uid"],
                capital=run_config["capital"],
                start_date=run_config.get("start_date"),
                end_date=run_config.get("end_date"),
            )
            outputs.append({**output, **run_config})
        except Exception as exc:
            failures.append({"uid": run_config["uid"], "error": str(exc)})
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
        metrics = output["report"].get("metrics", {})
        summary_rows.append({
            "uid": output["uid"],
            "timeframe": output["timeframe"],
            "source_timeframe": output["source_timeframe"],
            "db": Path(output["db_path"]).stem,
            "trades": len(output["result"]["tradebook"]),
            **metrics,
        })

    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        display_columns = [
            c for c in
            ["uid", "timeframe", "db", "trades", "cagr", "sharpe_ratio", "max_drawdown", "final_equity"]
            if c in summary.columns
        ]
        print()
        print("COMPARISON")
        print("-" * 76)
        print(summary[display_columns].to_string(index=False))

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
