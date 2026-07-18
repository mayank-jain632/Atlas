import pandas as pd
from .base import BaseFuturesStrategy
from indicators import adx,choppiness_index,donchian_channels,ema

class DCEMACHOPStrategy(BaseFuturesStrategy):
    strategy_name="dcemachop"
    
    def strategy_required_history(self):
        return max(self.parameters["donchian_period"]+2,
                   self.parameters["ema_period"]+2,
                   2*self.parameters["adx_period"]+2,
                   self.parameters["chop_period"]+2)
    

    def generate_desired_position(self,data):
        dc=donchian_channels(data, period=self.parameters["donchian_period"], shift=1)
        a=adx(data, period=self.parameters["adx_period"])
        e=ema(data.close,self.parameters["ema_period"])
        ch=choppiness_index(data,period=self.parameters["chop_period"])
        c=float(data.close.iloc[-1])
        u=dc.upper.iloc[-1]
        l=dc.lower.iloc[-1]
        av=a.adx.iloc[-1]
        ev=e.iloc[-1]
        cv=ch.iloc[-1]
        diag={"donchian_upper":None if pd.isna(u) else float(u),"donchian_lower":None if pd.isna(l) else float(l),"adx":None if pd.isna(av) else float(av),"ema":None if pd.isna(ev) else float(ev),"chop":None if pd.isna(cv) else float(cv)}
        
        if any(pd.isna(x) for x in (u,l,av,ev,cv)):return 0,diag
        
        trend = av >= self.parameters["adx_threshold"]
        not_choppy = cv <= self.parameters["chop_threshold"]
        if c > u and c > ev and trend and not_choppy: return 1, diag
        if c<l and c<ev and trend and not_choppy:return -1,diag
        return int(self.position_direction),diag
