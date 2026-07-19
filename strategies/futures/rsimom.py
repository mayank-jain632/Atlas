import pandas as pd
from .base import BaseFuturesStrategy
from .indicators import rsi, ema

class RSIMomentumStrategy(BaseFuturesStrategy):
    strategy_name = "rsimom"

    def strategy_required_history(self):
        return self.parameters["trend_ema_period"] + self.parameters["rsi_period"] + 5

    def generate_desired_position(self, data):
        r = rsi(data.close, self.parameters["rsi_period"])
        r_ema = ema(r, self.parameters["rsi_ema_period"])
        trend_ema = ema(data.close, self.parameters["trend_ema_period"])
        exit_fast = ema(data.close, self.parameters["exit_fast_period"])
        exit_slow = ema(data.close, self.parameters["exit_slow_period"])

        c = float(data.close.iloc[-1])
        r_now, r_prev = r.iloc[-1], r.iloc[-2]
        re_now, re_prev = r_ema.iloc[-1], r_ema.iloc[-2]
        r_max_recent = r.iloc[-10:].max()  # for "climbed above 56" check
        trend = trend_ema.iloc[-1]
        ef, es = exit_fast.iloc[-1], exit_slow.iloc[-1]
        ef_prev, es_prev = exit_fast.iloc[-2], exit_slow.iloc[-2]

        diag = {
            "rsi": None if pd.isna(r_now) else float(r_now),
            "rsi_ema": None if pd.isna(re_now) else float(re_now),
            "trend_ema": None if pd.isna(trend) else float(trend),
            "exit_fast_ema": None if pd.isna(ef) else float(ef),
            "exit_slow_ema": None if pd.isna(es) else float(es),
        }

        required = [r_now, r_prev, re_now, re_prev, trend, ef, es, ef_prev, es_prev]
        if any(pd.isna(x) for x in required):
            return 0, diag

        volume_ok = float(data.volume.iloc[-1]) > 0
        price_above_trend = c > trend

        # Signal A: momentum cross - RSI crosses above its own 9-EMA, while that EMA is already above 50
        signal_a = (re_now > 50) and (r_prev <= re_prev) and (r_now > re_now)

        # Signal B: 50-level retest - RSI climbed above 56 recently, pulled back into 44-55 zone, RSI-EMA still above 50
        signal_b = (r_max_recent > 56) and (44 <= r_now <= 55) and (re_now > 50)

        entry = (signal_a or signal_b) and price_above_trend and volume_ok

        # Exit: fast EMA crosses below slow EMA
        exit_signal = (ef_prev >= es_prev) and (ef < es)

        if self.position_direction > 0 and exit_signal:
            return 0, diag
        if entry and self.position_direction == 0:
            return 1, diag
        return int(self.position_direction), diag