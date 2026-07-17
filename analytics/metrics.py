# ============================================================
# PERFORMANCE METRICS
# ============================================================
import pandas as pd
import numpy as np

def calculate_metrics(
    equity: pd.DataFrame,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """
    Calculate common performance metrics from the equity curve.

    Expected columns:
        timestamp
        equity
    """
    if equity is None or equity.empty:
        return {}

    frame = equity.copy()

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"]
    )

    frame = (
        frame
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .set_index("timestamp")
    )

    equity_series = (
        pd.to_numeric(
            frame["equity"],
            errors="coerce",
        )
        .dropna()
    )

    if len(equity_series) < 2:
        return {}

    initial_equity = float(
        equity_series.iloc[0]
    )

    final_equity = float(
        equity_series.iloc[-1]
    )

    start_date = equity_series.index[0]
    end_date = equity_series.index[-1]

    elapsed_days = (
        end_date - start_date
    ).days

    elapsed_years = (
        elapsed_days / 365.25
        if elapsed_days > 0
        else 0.0
    )

    total_return = (
        final_equity / initial_equity - 1.0
        if initial_equity > 0
        else np.nan
    )

    cagr = (
        (final_equity / initial_equity)
        ** (1.0 / elapsed_years)
        - 1.0
        if (
            initial_equity > 0
            and final_equity > 0
            and elapsed_years > 0
        )
        else np.nan
    )

    daily_returns = (
        equity_series
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    annualized_volatility = (
        float(
            daily_returns.std(ddof=1)
            * np.sqrt(periods_per_year)
        )
        if len(daily_returns) > 1
        else np.nan
    )

    daily_risk_free_rate = (
        (1.0 + risk_free_rate)
        ** (1.0 / periods_per_year)
        - 1.0
    )

    excess_returns = (
        daily_returns
        - daily_risk_free_rate
    )

    sharpe_ratio = (
        float(
            excess_returns.mean()
            / excess_returns.std(ddof=1)
            * np.sqrt(periods_per_year)
        )
        if (
            len(excess_returns) > 1
            and excess_returns.std(ddof=1) > 0
        )
        else np.nan
    )

    downside_returns = excess_returns[
        excess_returns < 0
    ]

    downside_deviation = (
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        downside_returns
                    )
                )
            )
            * np.sqrt(periods_per_year)
        )
        if len(downside_returns) > 0
        else np.nan
    )

    annualized_excess_return = (
        float(
            excess_returns.mean()
            * periods_per_year
        )
        if len(excess_returns) > 0
        else np.nan
    )

    sortino_ratio = (
        annualized_excess_return
        / downside_deviation
        if (
            downside_deviation is not None
            and np.isfinite(downside_deviation)
            and downside_deviation > 0
        )
        else np.nan
    )

    running_peak = equity_series.cummax()

    drawdown = (
        equity_series / running_peak - 1.0
    )

    max_drawdown = float(
        drawdown.min()
    )

    calmar_ratio = (
        cagr / abs(max_drawdown)
        if (
            np.isfinite(cagr)
            and max_drawdown < 0
        )
        else np.nan
    )

    positive_days = int(
        (daily_returns > 0).sum()
    )

    negative_days = int(
        (daily_returns < 0).sum()
    )

    flat_days = int(
        (daily_returns == 0).sum()
    )

    win_rate = (
        positive_days
        / (
            positive_days
            + negative_days
        )
        if (
            positive_days
            + negative_days
        ) > 0
        else np.nan
    )

    best_day = (
        float(daily_returns.max())
        if not daily_returns.empty
        else np.nan
    )

    worst_day = (
        float(daily_returns.min())
        if not daily_returns.empty
        else np.nan
    )

    # Drawdown duration in calendar days.
    underwater = drawdown < 0

    max_drawdown_duration_days = 0
    current_start = None

    for timestamp, is_underwater in underwater.items():
        if is_underwater:
            if current_start is None:
                current_start = timestamp
        else:
            if current_start is not None:
                duration = (
                    timestamp - current_start
                ).days

                max_drawdown_duration_days = max(
                    max_drawdown_duration_days,
                    duration,
                )

                current_start = None

    # Handle an unrecovered drawdown at the end.
    if current_start is not None:
        duration = (
            underwater.index[-1]
            - current_start
        ).days

        max_drawdown_duration_days = max(
            max_drawdown_duration_days,
            duration,
        )

    return {
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": (
            annualized_volatility
        ),
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "max_drawdown_duration_days": float(
            max_drawdown_duration_days
        ),
        "win_rate": win_rate,
        "best_day": best_day,
        "worst_day": worst_day,
        "positive_days": float(positive_days),
        "negative_days": float(negative_days),
        "flat_days": float(flat_days),
        "elapsed_years": elapsed_years,
    }