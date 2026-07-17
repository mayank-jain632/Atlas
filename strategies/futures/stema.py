from pathlib import Path
import pandas as pd
from .base import BaseFuturesStrategy
from .indicators import ema,supertrend
class STEMAStrategy(BaseFuturesStrategy):
    strategy_name="stema"
    def strategy_required_history(self):return max(self.parameters["supertrend_period"]+2,self.parameters["ema_period"]+2)
    def generate_desired_position(self,data):
        st=supertrend(data,self.parameters["supertrend_period"],self.parameters["supertrend_multiplier"]);e=ema(data.close,self.parameters["ema_period"]);c=float(data.close.iloc[-1]);d=st.supertrend_direction.iloc[-1];ev=e.iloc[-1];diag={"supertrend":None if pd.isna(st.supertrend.iloc[-1]) else float(st.supertrend.iloc[-1]),"supertrend_direction":None if pd.isna(d) else int(d),"ema":None if pd.isna(ev) else float(ev)}
        if pd.isna(d) or pd.isna(ev):return 0,diag
        return (1 if d>0 and c>ev else -1 if d<0 and c<ev else 0),diag
