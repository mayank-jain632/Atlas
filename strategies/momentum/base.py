from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data.universe import UniverseManager
from strategies.base import BaseStrategy

from .allocation import allocate_weights
from .signals import compute_signal
from .uid import parse_uid


class BaseMomentumStrategy(BaseStrategy):
    """
    Shared momentum strategy functionality.

    Subclasses implement select_stocks().
    """

    strategy_name = "momentum_base"

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
        # The UID is now the only source of momentum parameters.
        parameters = parse_uid(uid)

        kwargs = {
            "uid": uid,
            "capital": capital,
            "timeframe": timeframe,
            "allow_fractional_shares": (
                allow_fractional_shares
            ),
        }

        if db_path is not None:
            kwargs["db_path"] = db_path

        super().__init__(**kwargs)

        self.parameters = parameters

        self.universe_manager = (
            UniverseManager(
                root=universe_root
            )
        )

        self.universe_name = str(
            parameters["universe"]
        )

        self.stock_universe = list(
            dict.fromkeys(
                self.universe_manager.load(
                    self.universe_name
                )
            )
        )

        if not self.stock_universe:
            raise ValueError(
                f"Universe "
                f"'{self.universe_name}' is empty."
            )

        self.last_rebalance_key: str | None = (
            None
        )

        self.rebalance_log: list[
            dict[str, Any]
        ] = []

    # ========================================================
    # History requirements
    # ========================================================
    def required_symbols(
        self,
    ) -> list[str]:
        """
        Symbols needed by the generic strategy runner.
        """
        return list(
            self.stock_universe
        )


    @property
    def symbols(
        self,
    ) -> list[str]:
        """
        Backward-compatible alias for older runners.
        """
        return self.required_symbols()
        
    def required_history(self) -> int:
        signal = self.parameters["signal"]

        requirements = [
            int(self.parameters["lookback"]) + 1,
        ]

        if signal == "rsi":
            requirements.append(
                int(self.parameters["rsi_window"]) + 1
            )

        if signal == "ma_cross":
            requirements.append(
                int(
                    self.parameters["ma_long_window"]
                ) + 1
            )

        if (
            self.parameters["strategy_type"]
            == "momentum_diversity"
        ):
            requirements.append(
                int(
                    self.parameters[
                        "diversity_lookback"
                    ]
                ) + 1
            )

        return max(requirements)

    # ========================================================
    # Rebalancing schedule
    # ========================================================

    def _rebalance_key(
        self,
        timestamp: pd.Timestamp,
    ) -> str | None:
        period = self.parameters["rebalance_period"]

        year = timestamp.year
        month = timestamp.month
        day = timestamp.day

        if period == "monthly":
            return f"{year}-{month:02d}"

        if period == "two_monthly":
            if month not in {1, 3, 5, 7, 9, 11}:
                return None

            return f"{year}-{month:02d}"

        if period == "quarterly":
            if month not in {1, 4, 7, 10}:
                return None

            return f"{year}-Q{((month - 1) // 3) + 1}"

        if period == "half_monthly":
            half = 1 if day < 15 else 2
            return f"{year}-{month:02d}-H{half}"

        raise ValueError(
            "Unsupported rebalance period: "
            f"{period}"
        )

    def should_rebalance(self) -> bool:
        timestamp = self.get_current_timestamp()

        key = self._rebalance_key(timestamp)

        if key is None:
            return False

        if key == self.last_rebalance_key:
            return False

        self.last_rebalance_key = key
        return True

    # ========================================================
    # Signal and selection
    # ========================================================

    def get_price_history(self) -> pd.DataFrame:
        history = self.history_frame(
            symbols=self.stock_universe,
            field="close",
            bars=self.required_history(),
        )

        if history.empty:
            return history

        # Remove stocks that have no usable history.
        history = history.dropna(
            axis=1,
            how="all",
        )

        return history.sort_index()

    def compute_scores(
        self,
        price_history: pd.DataFrame,
    ) -> pd.DataFrame:
        return compute_signal(
            signal_name=self.parameters["signal"],
            prices=price_history,
            parameters=self.parameters,
        )

    def select_stocks(
        self,
        score_df: pd.DataFrame,
        price_history: pd.DataFrame,
    ) -> list[str]:
        raise NotImplementedError

    def create_trade_notes(
        self,
        selected: list[str],
        weights: dict[str, float],
        scores: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        ranked = sorted(
            selected,
            key=lambda symbol: float(
                scores.get(symbol, float("-inf"))
            ),
            reverse=True,
        )

        rank_by_symbol = {
            symbol: rank
            for rank, symbol in enumerate(
                ranked,
                start=1,
            )
        }

        notes: dict[str, dict[str, Any]] = {}

        for symbol in set(
            list(self.get_positions())
            + list(weights)
        ):
            notes[symbol] = {
                "universe": self.universe_name,
                "signal": self.parameters["signal"],
                "lookback": int(
                    self.parameters["lookback"]
                ),
                "rebalance_period": (
                    self.parameters["rebalance_period"]
                ),
                "allocator": (
                    self.parameters["allocator"]
                ),
                "score": (
                    float(scores[symbol])
                    if symbol in scores
                    else None
                ),
                "rank": rank_by_symbol.get(symbol),
                "target_weight": float(
                    weights.get(symbol, 0.0)
                ),
            }

            if (
                self.parameters["strategy_type"]
                == "momentum_diversity"
            ):
                notes[symbol].update(
                    {
                        "diversity_method": (
                            self.parameters[
                                "diversity_method"
                            ]
                        ),
                        "diversity_lookback": int(
                            self.parameters[
                                "diversity_lookback"
                            ]
                        ),
                        "diversity_lambda": float(
                            self.parameters[
                                "diversity_lambda"
                            ]
                        ),
                    }
                )

        return notes

    # ========================================================
    # Daily event
    # ========================================================

    def on_day_close(self) -> None:
        if not self.should_rebalance():
            return

        price_history = self.get_price_history()

        if price_history.empty:
            return

        if len(price_history) < self.required_history():
            return

        score_df = self.compute_scores(
            price_history=price_history
        )

        if score_df.empty:
            return

        selected = self.select_stocks(
            score_df=score_df,
            price_history=price_history,
        )

        weights, scores = allocate_weights(
            selected_stocks=selected,
            score_df=score_df,
            allocator=self.parameters["allocator"],
            minimum_score=float(
                self.parameters["minimum_score"]
            ),
        )

        notes_by_symbol = self.create_trade_notes(
            selected=selected,
            weights=weights,
            scores=scores,
        )

        self.rebalance_to_weights(
            weights=weights,
            reason="MOMENTUM_REBALANCE",
            notes_by_symbol=notes_by_symbol,
        )

        self.rebalance_log.append(
            {
                "timestamp": self.get_current_timestamp(),
                "uid": self.uid,
                "strategy": self.strategy_name,
                "universe": self.universe_name,
                "signal": self.parameters["signal"],
                "selected": list(weights),
                "weights": weights,
                "scores": scores,
            }
        )

    def get_rebalance_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.rebalance_log)

    def save_results(
        self,
        output_dir: str | Path,
    ) -> None:
        super().save_results(output_dir)

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        self.get_rebalance_log().to_csv(
            output / "rebalance_log.csv",
            index=False,
        )