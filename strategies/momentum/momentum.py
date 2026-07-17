from __future__ import annotations

import pandas as pd

from .base import BaseMomentumStrategy
from .ranking import rank_stocks


class MomentumStrategy(
    BaseMomentumStrategy
):
    """
    Standard momentum strategy.

    Stocks are ranked by their latest signal score and the top N
    valid stocks are selected.
    """

    strategy_name = "momentum"

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs
        )

        if (
            self.parameters["strategy_type"]
            != "momentum"
        ):
            raise ValueError(
                "MomentumStrategy requires a UID "
                "beginning with 'momentum'."
            )

    def select_stocks(
        self,
        score_df: pd.DataFrame,
        price_history: pd.DataFrame,
    ) -> list[str]:
        del price_history

        return rank_stocks(
            score_df=score_df,
            top_n=int(
                self.parameters["top_n"]
            ),
            minimum_score=float(
                self.parameters[
                    "minimum_score"
                ]
            ),
        )