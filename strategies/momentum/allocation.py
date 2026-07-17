from __future__ import annotations

import numpy as np
import pandas as pd


def score_weights(
    selected_stocks: list[str],
    score_df: pd.DataFrame,
    minimum_score: float = 0.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Allocate proportionally to valid positive scores.

    Returns:
        weights
        raw scores
    """
    if not selected_stocks:
        return {}, {}

    latest = (
        score_df[selected_stocks]
        .iloc[-1]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    latest = latest[
        latest > float(minimum_score)
    ]

    if latest.empty:
        return {}, {}

    total_score = float(latest.sum())

    if total_score <= 0:
        return {}, {}

    weights = latest / total_score

    return (
        weights.to_dict(),
        latest.to_dict(),
    )


def equal_weights(
    selected_stocks: list[str],
    score_df: pd.DataFrame,
    minimum_score: float = 0.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Allocate equally across stocks with valid scores above
    the minimum threshold.

    Returns:
        weights
        raw scores
    """
    if not selected_stocks:
        return {}, {}

    latest = (
        score_df[selected_stocks]
        .iloc[-1]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    latest = latest[
        latest > float(minimum_score)
    ]

    if latest.empty:
        return {}, {}

    weight = 1.0 / len(latest)

    weights = {
        symbol: weight
        for symbol in latest.index
    }

    return (
        weights,
        latest.to_dict(),
    )


def allocate_weights(
    selected_stocks: list[str],
    score_df: pd.DataFrame,
    allocator: str = "score",
    minimum_score: float = 0.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Common allocation entry point.

    Supported allocators:
        score
        equal
    """
    allocator = allocator.lower()

    if allocator == "score":
        return score_weights(
            selected_stocks=selected_stocks,
            score_df=score_df,
            minimum_score=minimum_score,
        )

    if allocator == "equal":
        return equal_weights(
            selected_stocks=selected_stocks,
            score_df=score_df,
            minimum_score=minimum_score,
        )

    raise ValueError(
        f"Unsupported momentum allocator: {allocator}"
    )