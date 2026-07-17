from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from analytics.metrics import calculate_metrics
from analytics.tables import save_performance_tables


def save_and_print_results(
    *,
    strategy,
    result: dict[str, pd.DataFrame],
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """
    Save standard Atlas results and print common performance
    metrics and yearly performance.

    Expected result keys:
        tradebook
        equity
        positions
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    strategy.save_results(
        output_dir=output,
    )

    tradebook = result.get(
        "tradebook",
        pd.DataFrame(),
    )

    equity = result.get(
        "equity",
        pd.DataFrame(),
    )

    metrics = calculate_metrics(
        equity=equity,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )

    tables = save_performance_tables(
        equity=equity,
        output_dir=output,
    )

    yearly = tables["yearly"]
    monthly = tables["monthly"]
    monthly_returns = tables[
        "monthly_returns"
    ]

    metrics_row = {
        "uid": strategy.uid,
        "strategy": strategy.strategy_name,
        "trade_count": len(tradebook),
        **(metadata or {}),
        **metrics,
    }

    pd.DataFrame(
        [metrics_row]
    ).to_csv(
        output / "metrics.csv",
        index=False,
    )

    print()
    print("=" * 76)
    print("BACKTEST COMPLETE")
    print("=" * 76)
    print(f"UID:                 {strategy.uid}")
    print(f"Strategy:            {strategy.strategy_name}")
    print(f"Trades:              {len(tradebook)}")

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

    if not yearly.empty:
        print()
        print("=" * 76)
        print("YEARLY PERFORMANCE")
        print("=" * 76)

        for row in yearly.itertuples(
            index=False
        ):
            print(
                f"{row.year}: "
                f"PnL ${row.yearly_pnl:>12,.2f} | "
                f"Return {row.yearly_return:>8.2%} | "
                f"Start ${row.start_equity:>12,.2f} | "
                f"End ${row.end_equity:>12,.2f}"
            )

    print()
    print(f"Saved to:            {output}")
    print("=" * 76)

    return {
        "metrics": metrics,
        "yearly": yearly,
        "monthly": monthly,
        "monthly_returns": monthly_returns,
    }