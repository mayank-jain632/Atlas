import pandas as pd
from .base import BaseFuturesStrategy
from .indicators import bollinger, ema

class BollingerTrendStrategy(BaseFuturesStrategy):
    strategy_name = "bbtrend"

    def strategy_required_history(self):
        return max(self.parameters["bb_period"], self.parameters["trend_ema_period"], self.parameters["volume_ma_period"]) + 2

    def generate_desired_position(self, data):
        bb = bollinger(data, self.parameters["bb_period"], self.parameters["bb_std"])
        trend = ema(data.close, self.parameters["trend_ema_period"])
        vol_ma = data.volume.rolling(self.parameters["volume_ma_period"]).mean()

        c = float(data.close.iloc[-1])
        v = float(data.volume.iloc[-1])
        mid, upper = bb.bb_mid.iloc[-1], bb.bb_upper.iloc[-1]
        trend_val = trend.iloc[-1]
        vol_avg = vol_ma.iloc[-1]

        diag = {
            "bb_mid": None if pd.isna(mid) else float(mid),
            "bb_upper": None if pd.isna(upper) else float(upper),
            "trend_ema": None if pd.isna(trend_val) else float(trend_val),
            "volume_ma": None if pd.isna(vol_avg) else float(vol_avg),
        }

        required = [mid, upper, trend_val, vol_avg]
        if any(pd.isna(x) for x in required):
            return 0, diag

        # Exit: price crosses back down through the middle band
        if self.position_direction > 0 and c < mid:
            return 0, diag

        # Long entry: close above upper band, above trend EMA, volume above average
        entry = (c > upper) and (c > trend_val) and (v > vol_avg)
        if entry and self.position_direction == 0:
            return 1, diag

        return int(self.position_direction), diag