from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from indicator_states import (
    BULLISH,
    IndicatorStateEngine,
)

from strategies.base import BaseStrategy

from .uid import (
    parse_indicator_basket_uid,
)

class IndicatorBasketStrategy(BaseStrategy):
    """
    Hold each basket symbol only while its own IndicatorState
    is bullish.

    UID example:

        indicator_basket
        __weights=SPY:0.25,QQQ:0.25,GLD:0.25,TLT:0.25
        __renorm=false
        __state=ma_crossover
        __fast=50
        __slow=200
        __method=sma

    Bearish allocations remain in cash unless renorm=true.
    """

    strategy_name = "indicator_basket"

    def __init__(
        self,
        uid: str,
        capital: float,
        db_path: str | Path | None = None,
        timeframe: str = "1d",
        allow_fractional_shares: bool = True,
    ) -> None:
        parsed = parse_indicator_basket_uid(
            uid
        )

        self.target_weights = dict(
            parsed["target_weights"]
        )

        self.renormalize_bullish_weights = bool(
            parsed["renormalize"]
        )

        self.parameters = {
            "target_weights": dict(
                self.target_weights
            ),
            "renormalize": (
                self.renormalize_bullish_weights
            ),
            "state_type": (
                parsed["state_type"]
            ),
            "state_parameters": (
                parsed["state_parameters"]
            ),
        }
        self.state_engine = (
            IndicatorStateEngine(
                evaluator=(
                    parsed[
                        "indicator_state"
                    ]
                ),
            )
        )

        self.state_log: list[
            dict[str, Any]
        ] = []
        super().__init__(
            uid=uid,
            capital=capital,
            db_path=db_path,
            timeframe=timeframe,
            allow_fractional_shares=(
                allow_fractional_shares
            ),
        )

    # ========================================================
    # Required data
    # ========================================================

    def required_symbols(
        self,
    ) -> list[str]:
        return list(
            self.target_weights
        )

    @property
    def symbols(
        self,
    ) -> list[str]:
        """
        Backward-compatible alias.
        """
        return self.required_symbols()

    # ========================================================
    # Reset
    # ========================================================

    def reset(self) -> None:
        super().reset()

        self.state_engine.reset()
        self.state_log = []

    # ========================================================
    # Daily event
    # ========================================================

    def on_day_close(self) -> None:
        results = (
            self.state_engine.evaluate_many(
                data_provider=self,
                symbols=self.required_symbols(),
            )
        )

        bullish_weights: dict[
            str,
            float,
        ] = {}

        notes_by_symbol: dict[
            str,
            dict[str, Any],
        ] = {}

        for symbol, result in results.items():
            self.state_log.append(
                {
                    "timestamp": (
                        self.get_current_timestamp()
                    ),
                    "indicator_type": (
                        self.parameters[
                            "state_type"
                        ]
                    ),
                    **result.to_dict(),
                }
            )

            if result.state == BULLISH:
                bullish_weights[symbol] = (
                    self.target_weights[
                        symbol
                    ]
                )

            notes_by_symbol[symbol] = {
                "base_target_weight": (
                    self.target_weights[
                        symbol
                    ]
                ),
                "indicator": (
                    self.parameters[
                        "state_type"
                    ]
                ),
                "indicator_parameters": (
                    self.parameters[
                        "state_parameters"
                    ]
                ),
                **result.to_dict(),
            }

        if (
            self.renormalize_bullish_weights
            and bullish_weights
        ):
            total = sum(
                bullish_weights.values()
            )

            bullish_weights = {
                symbol: weight / total
                for symbol, weight
                in bullish_weights.items()
            }

        self.rebalance_to_weights(
            weights=bullish_weights,
            reason=(
                "INDICATOR_BASKET_REBALANCE"
            ),
            notes_by_symbol=(
                notes_by_symbol
            ),
        )

    # ========================================================
    # Output
    # ========================================================

    def get_state_log(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self.state_log
        )

    def save_results(
        self,
        output_dir: str | Path,
    ) -> None:
        super().save_results(
            output_dir
        )

        output = Path(output_dir)

        self.get_state_log().to_csv(
            output
            / "indicator_state_log.csv",
            index=False,
        )