import pandas as pd
from .base import BaseFuturesStrategy
from .indicators import donchian

class DonchianAsymmetricBreakoutStrategy(BaseFuturesStrategy):
    strategy_name = "dcbreakout"

    def strategy_required_history(self):
        return max(self.parameters["upper_period"], self.parameters["lower_period"]) + 2

    def generate_desired_position(self, data):
        dc_upper = donchian(data, self.parameters["upper_period"]).donchian_upper
        dc_lower = donchian(data, self.parameters["lower_period"]).donchian_lower

        c = float(data.close.iloc[-1])
        u = dc_upper.iloc[-1]
        l = dc_lower.iloc[-1]

        diag = {
            "donchian_upper": None if pd.isna(u) else float(u),
            "donchian_lower": None if pd.isna(l) else float(l),
        }

        if pd.isna(u) or pd.isna(l):
            return 0, diag

        # Long-only: enter on close above upper band, exit on close below lower band
        if c > u:
            return 1, diag
        if c < l and self.position_direction > 0:
            return 0, diag
        return int(self.position_direction), diag