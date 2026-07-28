"""Parameter-grid sweep for momentum stock strategies.

Reuses runners/run_strats.py's execution engine (run_uid, and the
STRATEGY_CLASSES / DB_PATH / TIMEFRAME / UNIVERSE_ROOT / OUTPUT_ROOT
config baked into it) instead of duplicating any of that. This script's
only job is to build UID lists per strategy family and collect their
results into comparison tables.

Two families are swept:

1. momentum (standard): universe, signal, lookback, top_n,
   rebalance_period, and allocator all vary.
2. momentum_indicator: universe, signal, lookback, rebalance_period,
   top_n, allocator, and filter_symbol vary as the "base" momentum
   parameters, crossed with the state= indicator filter, which varies
   separately per indicator family since ma_crossover, macd, and ema
   have different parameter grammars (see strategies/momentum/uid.py and
   indicator_states/uid.py) -- three base-params x state-params sub-grids
   concatenated rather than one flat cartesian grid.

The full cartesian product across both families is thousands of UIDs
(720 momentum x 7,200 momentum_indicator at the sizes below) -- far too
slow to run end to end in one pass at momentum's unprofiled per-UID cost.
Instead of running everything, each family's full combination list is
built first (the "_ALL" lists below) and then randomly subsampled down
to MOMENTUM_SAMPLE_SIZE / INDICATOR_SAMPLE_SIZE unique UIDs -- sampling
without replacement from an already-duplicate-free list of combinations,
so "unique" falls out for free. Both samples are seeded
(MOMENTUM_SEED / INDICATOR_SEED) so a rerun reproduces the same draw;
change the seed to get a different one.

Sequential by design, deliberately not parallelized yet: momentum
backtests haven't been profiled the way futures backtests were, and
BaseMomentumStrategy.get_price_history() queries the database once per
universe symbol on every rebalance day -- for a 500-symbol universe like
sp500 that could be a meaningful per-UID cost we don't have numbers for.
Parallelizing without knowing the per-UID cost is exactly how the futures
grid's first parallel attempt (see runners/run_futures_grid.py's
MAX_WORKERS comment) ended up slower than sequential. Revisit once this
has been run once and we have real timing.   
"""

from __future__ import annotations

from pathlib import Path
import random
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runners.run_strats as base

# ============================================================
# Shared sweep config
# ============================================================

CAPITAL = 100_000.0
START_DATE = "2000-01-01"
END_DATE = None

# Overrides run_strats.py's own OUTPUT_ROOT ("results") for the duration
# of this grid -- same override pattern run_futures_grid.py uses for
# DB_PATH/TIMEFRAME, so run_strats.py itself doesn't need to change.
OUTPUT_ROOT = str(PROJECT_ROOT / "results" / "momentun_runs_random")

# How many unique UIDs to actually run per family, sampled from the full
# cartesian product below.
MOMENTUM_SAMPLE_SIZE = 60
INDICATOR_SAMPLE_SIZE = 90

MOMENTUM_SEED = 42
INDICATOR_SEED = 43


def _sample_unique(
    runs: list[dict[str, Any]],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    A reproducible random sample of `size` run configs.

    Every entry in `runs` already has a distinct UID (it comes from a
    cartesian product with no repeated parameter tuples), so a plain
    sample-without-replacement is already a unique-UID sample -- no
    extra dedup needed.
    """
    if size > len(runs):
        raise ValueError(
            f"Requested {size} samples but only {len(runs)} unique "
            "combinations exist."
        )

    return random.Random(seed).sample(runs, size)


# ============================================================
# Strategy 1: momentum -- universe / signal / lb / n / rb / alloc all vary
# ============================================================

MOMENTUM_UNIVERSE = ["nasdaq100", "custom", "sp500"]
MOMENTUM_SIGNAL = ["price", "vol_adj", "rsi", "ma_cross", "low_vol", "trend_quality"]

MOMENTUM_LOOKBACKS = [60, 90, 126, 180, 252]
MOMENTUM_TOP_NS = [5, 10, 15, 20]
MOMENTUM_REBALANCE_PERIODS = ["monthly", "quarterly"]
MOMENTUM_ALLOCATORS = ["score"]


def build_momentum_uid(
    *,
    universe: str,
    signal: str,
    lookback: int,
    rebalance_period: str,
    top_n: int,
    allocator: str,
) -> str:
    return (
        f"momentum__u={universe}"
        f"__sig={signal}"
        f"__lb={lookback}"
        f"__rb={rebalance_period}"
        f"__n={top_n}"
        f"__alloc={allocator}"
    )


_MOMENTUM_RUNS_ALL: list[dict[str, Any]] = [
    {
        "uid": build_momentum_uid(
            universe=universe,
            signal=signal,
            lookback=lookback,
            rebalance_period=rebalance_period,
            top_n=top_n,
            allocator=allocator,
        )
    }
    for universe in MOMENTUM_UNIVERSE
    for signal in MOMENTUM_SIGNAL
    for lookback in MOMENTUM_LOOKBACKS
    for top_n in MOMENTUM_TOP_NS
    for rebalance_period in MOMENTUM_REBALANCE_PERIODS
    for allocator in MOMENTUM_ALLOCATORS
]


# ============================================================
# Strategy 2: momentum_indicator -- base params vary, crossed with state=
# ============================================================

INDICATOR_UNIVERSE = ["nasdaq100", "custom", "sp500"]
INDICATOR_SIGNAL = ["price", "vol_adj", "rsi", "ma_cross", "low_vol", "trend_quality"]
INDICATOR_LOOKBACK = [60, 90, 126, 180, 252]
INDICATOR_REBALANCE_PERIOD = ["monthly", "quarterly"]
INDICATOR_TOP_N = [5, 10, 15, 20]
INDICATOR_ALLOCATOR = ["score"]
INDICATOR_FILTER_SYMBOL = ["QQQ"]
INDICATOR_LIQUIDATE = "true"
INDICATOR_REENTER = "true"

# ma_crossover: (fast, slow) pairs; method fixed at sma to match the
# reference UID (fast=50/slow=200/method=sma).
MA_CROSSOVER_PAIRS = [
    # (20, 100),
    # (20, 200),
    (50, 100),
    (50, 200),
]

# macd: (fast, slow, signal_period) triples -- classic 12/26/9, a faster
# variant, a slower variant, and a faster-signal variant.
MACD_TRIPLES = [
    (12, 26, 9),
    (8, 21, 5),
    (19, 39, 9),
    (12, 26, 5),
]

# ema: single-MA state -- indicator_states treats state=ema as
# MovingAverageState with method forced to "ema" (see
# indicator_states/uid.py), so only the period varies here.
EMA_PERIODS = [50, 100, 150, 200]


def _indicator_base_param_combos() -> list[dict[str, Any]]:
    """Every combination of the momentum-side parameters shared by all
    three indicator-state families."""
    return [
        {
            "universe": universe,
            "signal": signal,
            "lookback": lookback,
            "rebalance_period": rebalance_period,
            "top_n": top_n,
            "allocator": allocator,
            "filter_symbol": filter_symbol,
        }
        for universe in INDICATOR_UNIVERSE
        for signal in INDICATOR_SIGNAL
        for lookback in INDICATOR_LOOKBACK
        for rebalance_period in INDICATOR_REBALANCE_PERIOD
        for top_n in INDICATOR_TOP_N
        for allocator in INDICATOR_ALLOCATOR
        for filter_symbol in INDICATOR_FILTER_SYMBOL
    ]


def _indicator_base_tokens(base: dict[str, Any]) -> list[str]:
    return [
        f"u={base['universe']}",
        f"sig={base['signal']}",
        f"lb={base['lookback']}",
        f"rb={base['rebalance_period']}",
        f"n={base['top_n']}",
        f"alloc={base['allocator']}",
        f"filter={base['filter_symbol']}",
        f"liquidate={INDICATOR_LIQUIDATE}",
        f"reenter={INDICATOR_REENTER}",
    ]


def build_momentum_indicator_uid(
    state_tokens: list[str],
) -> str:
    parts = [
        "momentum_indicator",
        *state_tokens,
    ]
    return "__".join(parts)


def _ma_crossover_runs() -> list[dict[str, Any]]:
    return [
        {
            "uid": build_momentum_indicator_uid(
                _indicator_base_tokens(base)
                + [
                    "state=ma_crossover",
                    f"fast={fast}",
                    f"slow={slow}",
                    "method=sma",
                ]
            )
        }
        for base in _indicator_base_param_combos()
        for fast, slow in MA_CROSSOVER_PAIRS
    ]


def _macd_runs() -> list[dict[str, Any]]:
    return [
        {
            "uid": build_momentum_indicator_uid(
                _indicator_base_tokens(base)
                + [
                    "state=macd",
                    f"fast={fast}",
                    f"slow={slow}",
                    f"signal_period={signal_period}",
                ]
            )
        }
        for base in _indicator_base_param_combos()
        for fast, slow, signal_period in MACD_TRIPLES
    ]


def _ema_runs() -> list[dict[str, Any]]:
    return [
        {
            "uid": build_momentum_indicator_uid(
                _indicator_base_tokens(base)
                + [
                    "state=ema",
                    f"period={period}",
                ]
            )
        }
        for base in _indicator_base_param_combos()
        for period in EMA_PERIODS
    ]


_INDICATOR_RUNS_ALL: list[dict[str, Any]] = (
    _ma_crossover_runs() + _macd_runs() + _ema_runs()
)


# ============================================================
# Actual runs: a reproducible random sample of each full grid above
# ============================================================

MOMENTUM_RUNS: list[dict[str, Any]] = _sample_unique(
    _MOMENTUM_RUNS_ALL,
    MOMENTUM_SAMPLE_SIZE,
    MOMENTUM_SEED,
)

INDICATOR_RUNS: list[dict[str, Any]] = _sample_unique(
    _INDICATOR_RUNS_ALL,
    INDICATOR_SAMPLE_SIZE,
    INDICATOR_SEED,
)


# ============================================================
# Grid execution
# ============================================================

def run_grid(
    runs: list[dict[str, Any]],
    *,
    label: str,
    summary_filename: str,
) -> pd.DataFrame:
    """
    Run one family's UID list sequentially and collect a comparison table.
    """

    base.OUTPUT_ROOT = OUTPUT_ROOT

    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, run_config in enumerate(runs, start=1):
        print("\n" + "#" * 76)
        print(f"RUNNING {label} {index} OF {len(runs)}")
        print("#" * 76)

        try:
            output = base.run_uid(
                uid=run_config["uid"],
                capital=CAPITAL,
                start_date=START_DATE,
                end_date=END_DATE,
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
    print(f"ATLAS {label} GRID SUMMARY")
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

        summary_path = Path(base.OUTPUT_ROOT) / summary_filename
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"\nFull comparison saved to: {summary_path}")

    print("=" * 76)
    return summary


def main() -> None:
    if not MOMENTUM_RUNS and not INDICATOR_RUNS:
        raise ValueError("No runs configured.")

    if MOMENTUM_RUNS:
        run_grid(
            MOMENTUM_RUNS,
            label="MOMENTUM",
            summary_filename="momentum_grid_summary.csv",
        )

    if INDICATOR_RUNS:
        run_grid(
            INDICATOR_RUNS,
            label="MOMENTUM_INDICATOR",
            summary_filename="momentum_indicator_grid_summary.csv",
        )


if __name__ == "__main__":
    main()
