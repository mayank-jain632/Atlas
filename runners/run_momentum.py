from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from strategies.momentum import (
    MomentumDiversityStrategy,
    MomentumStrategy,
    build_uid,
)
from analytics.metrics import (
    calculate_metrics,
)

from analytics.tables import (
    save_performance_tables,
)

# ============================================================
# CONFIGURATION
# ============================================================

STRATEGY_TYPE = "momentum_diversity"
# Options:
#   "momentum"
#   "momentum_diversity"

CAPITAL = 100_000.0

START_DATE = "2000-01-01"
END_DATE = None

TIMEFRAME = "1d"

ALLOW_FRACTIONAL_SHARES = True

UNIVERSE_ROOT = "config/universes"

OUTPUT_ROOT = "results/momentum"

# ============================================================
# MOMENTUM PARAMETERS
# ============================================================

PARAMETERS = {
    "strategy_type": STRATEGY_TYPE,

    # Stock universe:
    # config/universes/sp500.csv
    # config/universes/nasdaq100.csv
    # config/universes/dow30.csv
    "universe": "custom",

    # Options:
    # price
    # rsi
    # ma_cross
    # vol_adj
    # low_vol
    # trend_quality
    "signal": "price",

    "lookback": 90,

    # Options:
    # half_monthly
    # monthly
    # two_monthly
    # quarterly
    "rebalance_period": "monthly",

    "top_n": 10,

    # Options:
    # score
    # equal
    "allocator": "score",

    # Stocks must have a score above this value.
    "minimum_score": 0.0,

    # Used only when signal="rsi"
    "rsi_window": 14,
    "rsi_threshold": 50.0,

    # Used only when signal="ma_cross"
    "ma_short_window": 40,
    "ma_long_window": 100,

    # Used only for momentum_diversity
    "diversity_method": "graph_cut",
    "diversity_lookback": 60,
    "diversity_lambda": 0.25,
}


# ============================================================
# CREATE STRATEGY
# ============================================================

def create_strategy():
    uid = build_uid(PARAMETERS)

    common_arguments = {
        "uid": uid,
        "capital": CAPITAL,
        "timeframe": TIMEFRAME,
        "allow_fractional_shares": (
            ALLOW_FRACTIONAL_SHARES
        ),
        "universe_root": UNIVERSE_ROOT,
    }

    if STRATEGY_TYPE == "momentum":
        strategy = MomentumStrategy(
            **common_arguments
        )

    elif STRATEGY_TYPE == "momentum_diversity":
        strategy = MomentumDiversityStrategy(
            **common_arguments
        )

    else:
        raise ValueError(
            f"Unsupported STRATEGY_TYPE: {STRATEGY_TYPE}"
        )

    return uid, strategy


# ============================================================
# RUN BACKTEST
# ============================================================

def main():
    uid, strategy = create_strategy()

    print()
    print("=" * 70)
    print("ATLAS MOMENTUM BACKTEST")
    print("=" * 70)
    print(f"UID:           {uid}")
    print(f"Strategy:      {STRATEGY_TYPE}")
    print(f"Universe:      {PARAMETERS['universe']}")
    print(
        f"Universe size: "
        f"{len(strategy.stock_universe)}"
    )
    print(f"Signal:        {PARAMETERS['signal']}")
    print(f"Start date:    {START_DATE}")
    print(f"End date:      {END_DATE}")
    print("=" * 70)

    result = strategy.run(
        symbols=strategy.stock_universe,
        start=START_DATE,
        end=END_DATE,
    )

    output_dir = (
        Path(OUTPUT_ROOT)
        / uid
    )

    strategy.save_results(
        output_dir=output_dir,
    )

    tradebook = result["tradebook"]
    equity = result["equity"]
    metrics = calculate_metrics(equity=equity, periods_per_year=252, risk_free_rate=0.0)
    print()
    print("=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print(f"Trades:        {len(tradebook)}")

    if metrics:
        print(
            f"Initial value:       "
            f"${metrics['initial_equity']:,.2f}"
        )
        print(
            f"Final value:         "
            f"${metrics['final_equity']:,.2f}"
        )
        print(
            f"Total return:        "
            f"{metrics['total_return']:.2%}"
        )
        print(
            f"CAGR:                "
            f"{metrics['cagr']:.2%}"
        )
        print(
            f"Annualized vol:      "
            f"{metrics['annualized_volatility']:.2%}"
        )
        print(
            f"Sharpe ratio:        "
            f"{metrics['sharpe_ratio']:.3f}"
        )
        print(
            f"Sortino ratio:       "
            f"{metrics['sortino_ratio']:.3f}"
        )
        print(
            f"Maximum drawdown:    "
            f"{metrics['max_drawdown']:.2%}"
        )
        print(
            f"Calmar ratio:        "
            f"{metrics['calmar_ratio']:.3f}"
        )
        print(
            f"Max DD duration:     "
            f"{metrics['max_drawdown_duration_days']:.0f} days"
        )
        print(
            f"Daily win rate:      "
            f"{metrics['win_rate']:.2%}"
        )
        print(
            f"Best day:            "
            f"{metrics['best_day']:.2%}"
        )
        print(
            f"Worst day:           "
            f"{metrics['worst_day']:.2%}"
        )
        print(
            f"Elapsed years:       "
            f"{metrics['elapsed_years']:.2f}"
        )
    else:
        print("No valid equity history was generated.")
    
    
    if not equity.empty:
        equity = equity.copy()

        equity["timestamp"] = pd.to_datetime(
            equity["timestamp"]
        )

        prepared_equity = (
            equity
            .copy()
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .set_index("timestamp")
        )
        initial_equity = float(prepared_equity.iloc[0]["equity"])
        final_equity = float(prepared_equity.iloc[-1]["equity"])

        total_return = (
            final_equity / initial_equity - 1.0
            if initial_equity > 0
            else 0.0
        )

        print(
            f"Initial value: ${initial_equity:,.2f}"
        )
        print(
            f"Final value:   ${final_equity:,.2f}"
        )
        print(
            f"Total return:  {total_return:.2%}"
        )

        tables = save_performance_tables(equity=equity, output_dir=output_dir,)

        yearly_results = tables["yearly"]
        monthly_results = tables["monthly"]
        monthly_returns = tables["monthly_returns"]
        print()
        print("=" * 70)
        print("YEARLY P&L")
        print("=" * 70)

        for row in yearly_results.itertuples(
            index=False
        ):
            print(
                f"{row.year}: "
                f"PnL ${row.yearly_pnl:>12,.2f} | "
                f"Return {row.yearly_return:>8.2%} | "
                f"Start ${row.start_equity:>12,.2f} | "
                f"End ${row.end_equity:>12,.2f}"
            )

    else:
        print("No equity history was generated.")

    print()
    print(f"Saved to:      {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()