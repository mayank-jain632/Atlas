from __future__ import annotations
import json
from abc import ABC,abstractmethod
from pathlib import Path
from typing import Any,Iterable,Optional
import pandas as pd
from strategies.base import BaseStrategy
from .uid import parse_futures_uid
from .instruments import get_futures_instrument
from .indicators import atr,normalize_ohlc

class BaseFuturesStrategy(BaseStrategy,ABC):
    strategy_name="futures_base"
    def __init__(self,uid,capital,db_path=None,timeframe="1d",allow_fractional_shares=False):
        self.parameters=parse_futures_uid(uid);self.symbol=self.parameters["symbol"];self.instrument=get_futures_instrument(self.symbol);self.data_symbol=self.instrument.data_symbol;self.contracts=1
        super().__init__(uid=uid,capital=capital,db_path=db_path,timeframe=timeframe,allow_fractional_shares=False);self.reset_futures_state()
    def reset_futures_state(self):
        self.position_direction=0;self.entry_price=None;self.entry_timestamp=None;self.entry_atr=None;self.stop_price=None;self.previous_settlement=None;self.account_equity=float(self.initial_capital);self.cumulative_pnl=0.;self.futures_tradebook=[];self.futures_equity_history=[];self.signal_log=[]
    def required_symbols(self):return [self.data_symbol]
    @property
    def symbols(self):return self.required_symbols()
    @abstractmethod
    def strategy_required_history(self):...
    def required_history(self):return max(self.strategy_required_history(),int(self.parameters["atr_period"])+2)
    @abstractmethod
    def generate_desired_position(self,data):...
    def _history_ohlc(self,bars):return normalize_ohlc(pd.concat({f:self.history(self.data_symbol,f,bars) for f in ("open","high","low","close","volume")},axis=1))
    def portfolio_value(self):return float(self.account_equity)
    def _mtm(self,price):
        if self.previous_settlement is None:self.previous_settlement=price;return 0.
        pnl=self.position_direction*self.contracts*(price-self.previous_settlement)*self.instrument.multiplier;self.account_equity+=pnl;self.cumulative_pnl+=pnl;self.previous_settlement=price;return float(pnl)
    def _trade_pnl(self,price):return 0. if self.position_direction==0 or self.entry_price is None else self.position_direction*self.contracts*(price-self.entry_price)*self.instrument.multiplier
    def _log_trade(self,action,before,after,price,reason,realized=0.,notes=None):
        self.futures_tradebook.append({"timestamp":self.get_current_timestamp(),"uid":self.uid,"strategy":self.strategy_name,"contract_symbol":self.symbol,"data_symbol":self.data_symbol,"action":action,"direction_before":before,"direction_after":after,"contracts":self.contracts,"price":price,"multiplier":self.instrument.multiplier,"notional":abs(price*self.instrument.multiplier*self.contracts),"entry_price":self.entry_price,"stop_price":self.stop_price,"realized_trade_pnl":realized,"equity":self.account_equity,"reason":reason,"notes":json.dumps(notes or {},sort_keys=True)})
    def _close(self,price,reason,notes=None):
        if self.position_direction==0:return
        b=self.position_direction;rp=self._trade_pnl(price);self._log_trade("SELL" if b>0 else "BUY_TO_COVER",b,0,price,reason,rp,notes);self.position_direction=0;self.entry_price=self.entry_timestamp=self.entry_atr=self.stop_price=None
    def _open(self,direction,price,a,reason,notes=None):
        self.position_direction=direction;self.entry_price=price;self.entry_timestamp=self.get_current_timestamp();self.entry_atr=a;dist=self.parameters["stop_atr_multiple"]*a;self.stop_price=price-dist if direction>0 else price+dist;self._log_trade("BUY" if direction>0 else "SELL_SHORT",0,direction,price,reason,0.,notes)
    def _set_position(self,desired,price,a,notes):
        if desired==self.position_direction:return
        if self.position_direction!=0:self._close(price,"SIGNAL_EXIT" if desired==0 else "SIGNAL_REVERSAL_EXIT",notes)
        if desired!=0:self._open(desired,price,a,"SIGNAL_ENTRY" if self.position_direction==0 else "SIGNAL_REVERSAL_ENTRY",notes)
    def on_day_close(self):
        bars=self.required_history()
        if not self.has_history(self.data_symbol,bars):return
        data=self._history_ohlc(bars);cur=data.iloc[-1];close=float(cur.close)
        stop=None
        if self.position_direction>0 and self.stop_price is not None and cur.low<=self.stop_price:stop=self.stop_price
        if self.position_direction<0 and self.stop_price is not None and cur.high>=self.stop_price:stop=self.stop_price
        if stop is not None:
            dp=self._mtm(float(stop));self._close(float(stop),"ATR_STOP",{"daily_pnl":dp});self.previous_settlement=close;self._log_signal(0,{"stopped":True,"stop_fill":stop},close,dp);return
        dp=self._mtm(close);desired,diag=self.generate_desired_position(data);a=float(atr(data,self.parameters["atr_period"]).iloc[-1])
        if pd.notna(a) and a>0:self._set_position(int(desired),close,a,diag)
        self._log_signal(int(desired),diag,close,dp)
    def _log_signal(self,desired,diag,close,dp):self.signal_log.append({"timestamp":self.get_current_timestamp(),"uid":self.uid,"strategy":self.strategy_name,"contract_symbol":self.symbol,"data_symbol":self.data_symbol,"desired_direction":desired,"actual_direction":self.position_direction,"close":close,"daily_pnl":dp,"stop_price":self.stop_price,**diag})
    def _record(self):
        try:close=self.get_close(self.data_symbol)
        except KeyError:return
        notional=abs(self.position_direction)*self.contracts*close*self.instrument.multiplier;lev=notional/self.account_equity if self.account_equity>0 else float("inf")
        self.futures_equity_history.append({"timestamp":self.get_current_timestamp(),"uid":self.uid,"cash":self.account_equity,"market_value":0.,"equity":self.account_equity,"daily_pnl":self.signal_log[-1]["daily_pnl"] if self.signal_log and self.signal_log[-1]["timestamp"]==self.get_current_timestamp() else 0.,"cumulative_pnl":self.cumulative_pnl,"contract_symbol":self.symbol,"data_symbol":self.data_symbol,"position_direction":self.position_direction,"contracts":self.contracts,"close":close,"notional_exposure":notional,"effective_leverage":lev,"stop_price":self.stop_price})
    def run(self,symbols=None,start=None,end=None):
        req=list(symbols or self.required_symbols());self.reset_futures_state();dates=self.timestamps(symbols=req,start=start,end=end)
        for t in dates:self.set_current_timestamp(t);self.on_day_close();self._record()
        return {"tradebook":self.get_tradebook(),"equity":self.get_equity_history(),"positions":self.get_positions_frame(),"signals":self.get_signal_log()}
    def get_tradebook(self):return pd.DataFrame(self.futures_tradebook)
    def get_equity_history(self):return pd.DataFrame(self.futures_equity_history)
    def get_positions_frame(self):return pd.DataFrame([{"uid":self.uid,"contract_symbol":self.symbol,"data_symbol":self.data_symbol,"direction":self.position_direction,"contracts":self.contracts if self.position_direction else 0,"entry_price":self.entry_price,"stop_price":self.stop_price}])
    def get_signal_log(self):return pd.DataFrame(self.signal_log)
    def save_results(self,output_dir):
        o=Path(output_dir);o.mkdir(parents=True,exist_ok=True);self.get_tradebook().to_csv(o/"tradebook.csv",index=False);self.get_equity_history().to_csv(o/"equity.csv",index=False);self.get_positions_frame().to_csv(o/"positions.csv",index=False);self.get_signal_log().to_csv(o/"signal_log.csv",index=False)
