from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from indicator_states import (
    BEARISH,
    BULLISH,
    UNKNOWN,
    VALID_STATES,
    IndicatorStateEngine,
)

from indicator_states.uid import (
    create_state_from_parameters,
)

from .allocation import allocate_weights
from .base import BaseMomentumStrategy
from .ranking import rank_stocks
from .uid import (
    parse_bool,
    parse_uid_parts,
)


class MomentumIndicatorStrategy(
        BaseMomentumStrategy
):
    """
    Momentum strategy controlled by an indicator state on a
    market-filter symbol.

    Example UID:

        momentum_indicator
        __u=custom_momentum
        __sig=price
        __lb=90
        __rb=monthly
        __n=10
        __alloc=score
        __filter=SPY
        __liquidate=true
        __reenter=true
        __initial=UNKNOWN
        __state=ma_crossover
        __fast=50
        __slow=200
        __method=sma
    """

    strategy_name = "momentum_indicator"

    def __init__(
        self,
        uid: str,
        capital: float,
        db_path: str | Path | None = None,
        timeframe: str = "1d",
        allow_fractional_shares: bool = True,
        universe_root: str | Path = (
            "config/universes"
        ),
    ) -> None:
        strategy_type, raw_parameters = (
            parse_uid_parts(uid)
        )

        if strategy_type != "momentum_indicator":
            raise ValueError(
                "MomentumIndicatorStrategy requires "
                "a UID beginning with "
                "'momentum_indicator'."
            )

        if "filter" not in raw_parameters:
            raise ValueError(
                "Momentum-indicator UID must contain "
                "'filter'."
            )

        self.filter_symbol = (
            raw_parameters["filter"]
            .strip()
            .upper()
        )

        if not self.filter_symbol:
            raise ValueError(
                "UID parameter 'filter' cannot "
                "be empty."
            )

        self.liquidate_on_bearish = (
            parse_bool(
                raw_parameters.get(
                    "liquidate",
                    "true",
                ),
                "liquidate",
            )
        )

        self.rebalance_on_bullish_transition = (
            parse_bool(
                raw_parameters.get(
                    "reenter",
                    "true",
                ),
                "reenter",
            )
        )

        initial_state = (
            raw_parameters.get(
                "initial",
                UNKNOWN,
            )
            .strip()
            .upper()
        )

        if initial_state not in VALID_STATES:
            raise ValueError(
                "UID parameter 'initial' must be "
                f"one of {sorted(VALID_STATES)}."
            )

        (
            indicator_state,
            state_type,
            state_parameters,
        ) = create_state_from_parameters(
            raw_parameters
        )

        # IMPORTANT:
        # EMS.__init__ calls self.reset(), so attributes used by
        # MomentumIndicatorStrategy.reset() must exist before
        # calling super().__init__().
        self.state_engine = IndicatorStateEngine(
            evaluator=indicator_state,
            initial_state=initial_state,
        )

        self.market_state_log: list[
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
            universe_root=universe_root,
        )

        self.parameters.update(
            {
                "filter_symbol": (
                    self.filter_symbol
                ),
                "liquidate_on_bearish": (
                    self.liquidate_on_bearish
                ),
                "reenter": (
                    self.rebalance_on_bullish_transition
                ),
                "initial_state": initial_state,
                "state_type": state_type,
                "state_parameters": (
                    state_parameters
                ),
            }
        )

    def required_symbols(
        self,
    ) -> list[str]:
        return list(
            dict.fromkeys(
                list(self.stock_universe)
                + [self.filter_symbol]
            )
        )


    @property
    def symbols(
        self,
    ) -> list[str]:
        return self.required_symbols()


    def select_stocks(
        self,
        score_df: pd.DataFrame,
        price_history: pd.DataFrame,
    ) -> list[str]:
        """
        Use the standard momentum ranker after the market-state
        filter has permitted momentum trading.
        """
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

    def reset(self) -> None:
        super().reset()

        self.state_engine.reset()
        self.market_state_log = []

        # BaseMomentumStrategy keeps this value.
        self.last_rebalance_key = None

    def _calculate_momentum_weights(
        self,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        list[str],
    ]:
        """
        Calculate the current momentum portfolio without
        placing trades.
        """
        price_history = (
            self.get_price_history()
        )

        if price_history.empty:
            return {}, {}, []

        if (
            len(price_history)
            < self.required_history()
        ):
            return {}, {}, []

        score_df = self.compute_scores(
            price_history=price_history
        )

        if score_df.empty:
            return {}, {}, []

        selected = self.select_stocks(
            score_df=score_df,
            price_history=price_history,
        )

        weights, scores = allocate_weights(
            selected_stocks=selected,
            score_df=score_df,
            allocator=(
                self.parameters["allocator"]
            ),
            minimum_score=float(
                self.parameters[
                    "minimum_score"
                ]
            ),
        )

        return weights, scores, selected

    def _rebalance_momentum_portfolio(
        self,
        reason: str,
    ) -> bool:
        """
        Calculate and execute the current momentum portfolio.

        Returns True when a valid allocation was created.
        """
        weights, scores, selected = (
            self._calculate_momentum_weights()
        )

        if not weights:
            return False

        notes_by_symbol = (
            self.create_trade_notes(
                selected=selected,
                weights=weights,
                scores=scores,
            )
        )

        for symbol in notes_by_symbol:
            notes_by_symbol[symbol].update(
                {
                    "filter_symbol": (
                        self.filter_symbol
                    ),
                    "filter_state": BULLISH,
                    "filter_indicator": (
                        self.state_engine
                        .evaluator
                        .name
                    ),
                }
            )

        self.rebalance_to_weights(
            weights=weights,
            reason=reason,
            notes_by_symbol=(
                notes_by_symbol
            ),
        )

        self.rebalance_log.append(
            {
                "timestamp": (
                    self.get_current_timestamp()
                ),
                "uid": self.uid,
                "strategy": (
                    self.strategy_name
                ),
                "universe": (
                    self.universe_name
                ),
                "signal": (
                    self.parameters["signal"]
                ),
                "filter_symbol": (
                    self.filter_symbol
                ),
                "filter_state": BULLISH,
                "selected": list(weights),
                "weights": weights,
                "scores": scores,
            }
        )

        return True

    def _liquidate_for_bearish_state(
        self,
        state_result,
    ) -> None:
        """
        Close all momentum holdings because the market-state
        filter is bearish.
        """
        positions = list(
            self.get_positions()
        )

        for symbol in positions:
            notes = {
                "filter_symbol": (
                    self.filter_symbol
                ),
                "filter_indicator": (
                    self.state_engine
                    .evaluator
                    .name
                ),
                **state_result.to_dict(),
            }

            self.close_position(
                symbol=symbol,
                reason=(
                    "MARKET_INDICATOR_BEARISH"
                ),
                notes=notes,
            )

    def on_day_close(self) -> None:
        market_state = (
            self.state_engine.evaluate(
                data_provider=self,
                symbol=self.filter_symbol,
            )
        )

        self.market_state_log.append(
            {
                "timestamp": (
                    self.get_current_timestamp()
                ),
                **market_state.to_dict(),
            }
        )

        # ----------------------------------------------------
        # Bearish market filter
        # ----------------------------------------------------

        if market_state.state == BEARISH:
            if self.liquidate_on_bearish:
                self._liquidate_for_bearish_state(
                    market_state
                )

            # Momentum entries are blocked.
            return

        # ----------------------------------------------------
        # Unknown market filter
        # ----------------------------------------------------

        if market_state.state == UNKNOWN:
            # Do not enter new positions while the market
            # state cannot be determined.
            return

        # ----------------------------------------------------
        # Bullish market filter
        # ----------------------------------------------------

        rebalance_due = (
            self.should_rebalance()
        )

        bullish_transition = (
            market_state.changed
            and market_state.state
            == BULLISH
        )

        should_run_momentum = (
            rebalance_due
            or (
                bullish_transition
                and self.rebalance_on_bullish_transition
            )
        )

        if not should_run_momentum:
            return

        success = (
            self._rebalance_momentum_portfolio(
                reason=(
                    "MOMENTUM_INDICATOR_REBALANCE"
                )
            )
        )

        if success:
            # Mark the scheduled rebalance period as complete.
            #
            # If this was only a bullish transition in the same
            # month, this simply preserves the current key.
            key = self._rebalance_key(
                self.get_current_timestamp()
            )

            if key is not None:
                self.last_rebalance_key = key

    def get_market_state_log(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self.market_state_log
        )

    def save_results(
        self,
        output_dir: str | Path,
    ) -> None:
        super().save_results(
            output_dir
        )

        output = Path(output_dir)

        self.get_market_state_log().to_csv(
            output
            / "market_indicator_state_log.csv",
            index=False,
        )