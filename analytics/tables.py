from __future__ import annotations

from pathlib import Path

import pandas as pd


def prepare_equity_frame(
    equity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize an Atlas equity DataFrame.

    Expected columns:
        timestamp
        equity
    """
    if equity is None or equity.empty:
        return pd.DataFrame()

    frame = equity.copy()

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
    )

    frame["equity"] = pd.to_numeric(
        frame["equity"],
        errors="coerce",
    )

    frame = (
        frame
        .dropna(
            subset=[
                "timestamp",
                "equity",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .set_index("timestamp")
    )

    return frame


def yearly_performance_table(
    equity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create calendar-year equity, P&L, and return table.

    The prior calendar year's ending equity is used as the
    next year's starting equity.
    """
    frame = prepare_equity_frame(equity)

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "start_equity",
                "end_equity",
                "yearly_pnl",
                "yearly_return",
            ]
        )

    year_end_equity = (
        frame["equity"]
        .resample("YE")
        .last()
        .dropna()
    )

    result = pd.DataFrame(
        {
            "end_equity": year_end_equity,
        }
    )

    result["start_equity"] = (
        result["end_equity"].shift(1)
    )

    result.iloc[
        0,
        result.columns.get_loc(
            "start_equity"
        ),
    ] = float(frame["equity"].iloc[0])

    result["yearly_pnl"] = (
        result["end_equity"]
        - result["start_equity"]
    )

    result["yearly_return"] = (
        result["end_equity"]
        / result["start_equity"]
        - 1.0
    )

    result["year"] = result.index.year

    return (
        result[
            [
                "year",
                "start_equity",
                "end_equity",
                "yearly_pnl",
                "yearly_return",
            ]
        ]
        .reset_index(drop=True)
    )


def monthly_performance_table(
    equity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create calendar-month equity, P&L, and return table.
    """
    frame = prepare_equity_frame(equity)

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "month",
                "start_equity",
                "end_equity",
                "monthly_pnl",
                "monthly_return",
            ]
        )

    month_end_equity = (
        frame["equity"]
        .resample("ME")
        .last()
        .dropna()
    )

    result = pd.DataFrame(
        {
            "end_equity": month_end_equity,
        }
    )

    result["start_equity"] = (
        result["end_equity"].shift(1)
    )

    result.iloc[
        0,
        result.columns.get_loc(
            "start_equity"
        ),
    ] = float(frame["equity"].iloc[0])

    result["monthly_pnl"] = (
        result["end_equity"]
        - result["start_equity"]
    )

    result["monthly_return"] = (
        result["end_equity"]
        / result["start_equity"]
        - 1.0
    )

    result["year"] = result.index.year
    result["month"] = result.index.month

    return (
        result[
            [
                "year",
                "month",
                "start_equity",
                "end_equity",
                "monthly_pnl",
                "monthly_return",
            ]
        ]
        .reset_index(drop=True)
    )


def monthly_return_matrix(
    equity: pd.DataFrame,
    percent: bool = False,
) -> pd.DataFrame:
    """
    Create a year x month return matrix.

    Columns:
        Jan ... Dec

    When percent=True, returns are multiplied by 100.
    """
    monthly = monthly_performance_table(
        equity
    )

    if monthly.empty:
        return pd.DataFrame()

    matrix = monthly.pivot(
        index="year",
        columns="month",
        values="monthly_return",
    )

    month_names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    matrix = matrix.rename(
        columns=month_names
    )

    ordered_columns = [
        month_names[month]
        for month in range(1, 13)
        if month_names[month] in matrix.columns
    ]

    matrix = matrix.reindex(
        columns=ordered_columns
    )

    if percent:
        matrix = matrix * 100.0

    return matrix


def yearly_return_series(
    equity: pd.DataFrame,
    percent: bool = False,
) -> pd.Series:
    """
    Return one calendar-year return value per year.
    """
    yearly = yearly_performance_table(
        equity
    )

    if yearly.empty:
        return pd.Series(
            dtype=float,
            name="yearly_return",
        )

    result = yearly.set_index(
        "year"
    )["yearly_return"]

    if percent:
        result = result * 100.0

    return result


def save_performance_tables(
    equity: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """
    Build and save the standard Atlas performance tables.
    """
    output = Path(output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    yearly = yearly_performance_table(
        equity
    )

    monthly = monthly_performance_table(
        equity
    )

    monthly_matrix = monthly_return_matrix(
        equity
    )

    yearly.to_csv(
        output / "yearly_pnl.csv",
        index=False,
    )

    monthly.to_csv(
        output / "monthly_pnl.csv",
        index=False,
    )

    monthly_matrix.to_csv(
        output / "monthly_returns.csv"
    )

    return {
        "yearly": yearly,
        "monthly": monthly,
        "monthly_returns": monthly_matrix,
    }