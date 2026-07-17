from __future__ import annotations

import pandas as pd

from .base import BaseMomentumStrategy
from .ranking import diverse_rank_stocks


class MomentumDiversityStrategy(
    BaseMomentumStrategy
):
    """
    Momentum strategy with a diversity-aware stock selector.
    """

    strategy_name = "momentum_diversity"

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs
        )

        if (
            self.parameters["strategy_type"]
            != "momentum_diversity"
        ):
            raise ValueError(
                "MomentumDiversityStrategy requires "
                "a UID beginning with "
                "'momentum_diversity'."
            )

    def select_stocks(
        self,
        score_df: pd.DataFrame,
        price_history: pd.DataFrame,
    ) -> list[str]:
        return diverse_rank_stocks(
            score_df=score_df,
            price_history=price_history,
            top_n=int(
                self.parameters["top_n"]
            ),
            diversity_method=(
                self.parameters[
                    "diversity_method"
                ]
            ),
            diversity_lookback=int(
                self.parameters[
                    "diversity_lookback"
                ]
            ),
            diversity_lambda=float(
                self.parameters[
                    "diversity_lambda"
                ]
            ),
            minimum_score=float(
                self.parameters[
                    "minimum_score"
                ]
            ),
        )