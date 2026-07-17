from __future__ import annotations

import numpy as np
import pandas as pd


def latest_valid_scores(
    score_df: pd.DataFrame,
    minimum_score: float = 0.0,
) -> pd.Series:
    if score_df is None or score_df.empty:
        return pd.Series(dtype=float)

    latest = (
        score_df
        .iloc[-1]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    latest = latest[
        latest > float(minimum_score)
    ]

    return latest.sort_values(ascending=False)


def rank_stocks(
    score_df: pd.DataFrame,
    top_n: int = 10,
    minimum_score: float = 0.0,
) -> list[str]:
    latest = latest_valid_scores(
        score_df=score_df,
        minimum_score=minimum_score,
    )

    if latest.empty:
        return []

    return latest.head(
        min(int(top_n), len(latest))
    ).index.tolist()


def build_similarity_matrix(
    prices: pd.DataFrame,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Similarity is recent daily-return correlation.

    Negative correlations are clipped to zero because they do
    not create a redundancy penalty in the diversity methods.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()

    recent = prices.tail(
        int(lookback) + 1
    ).astype(float)

    returns = recent.pct_change(
        fill_method=None,
    ).dropna(how="all")

    if returns.empty:
        return pd.DataFrame(
            index=prices.columns,
            columns=prices.columns,
            dtype=float,
        )

    correlation = returns.corr().fillna(0.0)
    similarity = correlation.clip(lower=0.0)

    for symbol in similarity.columns:
        similarity.loc[symbol, symbol] = 1.0

    return similarity


def graph_cut_selection(
    scores: pd.Series,
    similarity: pd.DataFrame,
    top_n: int,
    diversity_lambda: float,
) -> list[str]:
    """
    Greedy graph-cut-style selection.

    marginal gain =
        momentum score
        - lambda * similarity to selected assets
    """
    selected: list[str] = []
    candidates = list(scores.index)

    while len(selected) < min(int(top_n), len(candidates)):
        best_symbol = None
        best_gain = -np.inf

        for symbol in candidates:
            if symbol in selected:
                continue

            penalty = 0.0

            if selected and not similarity.empty:
                available = [
                    item
                    for item in selected
                    if (
                        item in similarity.columns
                        and symbol in similarity.index
                    )
                ]

                if available:
                    penalty = float(
                        similarity.loc[
                            symbol,
                            available,
                        ].sum()
                    )

            gain = (
                float(scores.loc[symbol])
                - float(diversity_lambda) * penalty
            )

            if gain > best_gain:
                best_gain = gain
                best_symbol = symbol

        if best_symbol is None:
            break

        selected.append(best_symbol)

    return selected


def facility_location_selection(
    scores: pd.Series,
    similarity: pd.DataFrame,
    top_n: int,
    diversity_lambda: float,
) -> list[str]:
    """
    Greedy facility-location selection.

    gain =
        momentum score
        + lambda * improvement in universe coverage
    """
    universe = list(scores.index)

    if similarity.empty:
        return universe[:min(int(top_n), len(universe))]

    selected: list[str] = []

    current_coverage = pd.Series(
        0.0,
        index=universe,
        dtype=float,
    )

    while len(selected) < min(int(top_n), len(universe)):
        best_symbol = None
        best_gain = -np.inf

        for symbol in universe:
            if symbol in selected:
                continue

            symbol_similarity = (
                similarity[symbol]
                .reindex(universe)
                .fillna(0.0)
            )

            new_coverage = pd.concat(
                [
                    current_coverage.rename("current"),
                    symbol_similarity.rename("candidate"),
                ],
                axis=1,
            ).max(axis=1)

            coverage_gain = float(
                (new_coverage - current_coverage).sum()
            )

            gain = (
                float(scores.loc[symbol])
                + float(diversity_lambda)
                * coverage_gain
            )

            if gain > best_gain:
                best_gain = gain
                best_symbol = symbol

        if best_symbol is None:
            break

        selected.append(best_symbol)

        selected_similarity = (
            similarity[best_symbol]
            .reindex(universe)
            .fillna(0.0)
        )

        current_coverage = pd.concat(
            [
                current_coverage.rename("current"),
                selected_similarity.rename("selected"),
            ],
            axis=1,
        ).max(axis=1)

    return selected


def diverse_rank_stocks(
    score_df: pd.DataFrame,
    price_history: pd.DataFrame,
    top_n: int = 10,
    diversity_method: str = "graph_cut",
    diversity_lookback: int = 60,
    diversity_lambda: float = 0.25,
    minimum_score: float = 0.0,
) -> list[str]:
    latest = latest_valid_scores(
        score_df=score_df,
        minimum_score=minimum_score,
    )

    if latest.empty:
        return []

    candidate_symbols = [
        symbol
        for symbol in latest.index
        if symbol in price_history.columns
    ]

    if not candidate_symbols:
        return []

    scores = latest.loc[candidate_symbols]

    price_subset = price_history[
        candidate_symbols
    ].copy()

    similarity = build_similarity_matrix(
        prices=price_subset,
        lookback=int(diversity_lookback),
    )

    method = diversity_method.lower()

    if method == "graph_cut":
        return graph_cut_selection(
            scores=scores,
            similarity=similarity,
            top_n=top_n,
            diversity_lambda=diversity_lambda,
        )

    if method == "facility_location":
        return facility_location_selection(
            scores=scores,
            similarity=similarity,
            top_n=top_n,
            diversity_lambda=diversity_lambda,
        )

    raise ValueError(
        f"Unsupported diversity method: {method}"
    )