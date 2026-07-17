import pandas as pd
from .base import BaseFuturesStrategy
from indicators import ema,parabolic_sar
class PSAREMAStrategy(BaseFuturesStrategy):
    strategy_name="psarema"
    def strategy_required_history(self):return self.parameters["ema_period"]+5
    def generate_desired_position(self,data):
        p=parabolic_sar(data,step=self.parameters["psar_step"],max_step=self.parameters["psar_max"]);e=ema(data.close,self.parameters["ema_period"]);c=float(data.close.iloc[-1]);d=p.direction.iloc[-1];ev=e.iloc[-1];diag={"psar":None if pd.isna(p.sar.iloc[-1]) else float(p.sar.iloc[-1]),"psar_direction":None if pd.isna(d) else int(d),"ema":None if pd.isna(ev) else float(ev)}
        if pd.isna(d) or pd.isna(ev):return 0,diag
        return (1 if d>0 and c>ev else -1 if d<0 and c<ev else 0),diag
