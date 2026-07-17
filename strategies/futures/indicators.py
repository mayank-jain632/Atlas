from __future__ import annotations
import numpy as np, pandas as pd

def normalize_ohlc(data):
    if data is None or data.empty:return pd.DataFrame(columns=["open","high","low","close","volume"])
    f=data.copy(); f.columns=[str(c).lower() for c in f.columns]
    miss={"high","low","close"}-set(f.columns)
    if miss: raise ValueError("Missing OHLC columns: "+", ".join(sorted(miss)))
    return f.sort_index()

def true_range(data):
    f=normalize_ohlc(data); pc=f.close.shift(1)
    return pd.concat([f.high-f.low,(f.high-pc).abs(),(f.low-pc).abs()],axis=1).max(axis=1)

def atr(data,period=14):return true_range(data).ewm(alpha=1/int(period),adjust=False,min_periods=int(period)).mean()
def ema(series,period):return series.ewm(span=int(period),adjust=False,min_periods=int(period)).mean()

def donchian(data,period=20):
    f=normalize_ohlc(data); r=f.copy(); r["donchian_upper"]=f.high.shift(1).rolling(period).max(); r["donchian_lower"]=f.low.shift(1).rolling(period).min(); return r

def adx(data,period=14):
    f=normalize_ohlc(data); up=f.high.diff(); down=-f.low.diff(); plus=pd.Series(np.where((up>down)&(up>0),up,0),index=f.index); minus=pd.Series(np.where((down>up)&(down>0),down,0),index=f.index); a=atr(f,period); pdi=100*plus.ewm(alpha=1/period,adjust=False,min_periods=period).mean()/a.replace(0,np.nan); mdi=100*minus.ewm(alpha=1/period,adjust=False,min_periods=period).mean()/a.replace(0,np.nan); dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan); r=f.copy(); r["plus_di"]=pdi;r["minus_di"]=mdi;r["adx"]=dx.ewm(alpha=1/period,adjust=False,min_periods=period).mean();return r

def choppiness(data,period=14):
    f=normalize_ohlc(data); return 100*np.log10(true_range(f).rolling(period).sum()/(f.high.rolling(period).max()-f.low.rolling(period).min()).replace(0,np.nan))/np.log10(period)

def supertrend(data,period=10,multiplier=3.0):
    f=normalize_ohlc(data); a=atr(f,period); hl2=(f.high+f.low)/2; bu=hl2+multiplier*a; bl=hl2-multiplier*a; fu=bu.copy();fl=bl.copy();d=pd.Series(np.nan,index=f.index);line=d.copy()
    for i in range(1,len(f)):
        if pd.isna(a.iloc[i]):continue
        fu.iloc[i]=bu.iloc[i] if bu.iloc[i]<fu.iloc[i-1] or f.close.iloc[i-1]>fu.iloc[i-1] else fu.iloc[i-1]
        fl.iloc[i]=bl.iloc[i] if bl.iloc[i]>fl.iloc[i-1] or f.close.iloc[i-1]<fl.iloc[i-1] else fl.iloc[i-1]
        prev=d.iloc[i-1] if pd.notna(d.iloc[i-1]) else (1 if f.close.iloc[i]>=fl.iloc[i] else -1)
        d.iloc[i]=(-1 if f.close.iloc[i]<fl.iloc[i] else 1) if prev>0 else (1 if f.close.iloc[i]>fu.iloc[i] else -1)
        line.iloc[i]=fl.iloc[i] if d.iloc[i]>0 else fu.iloc[i]
    r=f.copy();r["supertrend"]=line;r["supertrend_direction"]=d;return r

def psar(data,step=.02,maximum=.2):
    f=normalize_ohlc(data);r=f.copy();n=len(f);sar=np.full(n,np.nan);d=np.full(n,np.nan)
    if n<2:r["psar"]=sar;r["psar_direction"]=d;return r
    h=f.high.to_numpy();l=f.low.to_numpy();c=f.close.to_numpy();bull=c[1]>=c[0];af=step;ep=h[0] if bull else l[0];sar[0]=l[0] if bull else h[0];d[0]=1 if bull else -1
    for i in range(1,n):
        x=sar[i-1]+af*(ep-sar[i-1])
        if bull:
            x=min(x,l[i-1],l[i-2] if i>1 else l[i-1])
            if l[i]<x:bull=False;x=ep;ep=l[i];af=step
            elif h[i]>ep:ep=h[i];af=min(af+step,maximum)
        else:
            x=max(x,h[i-1],h[i-2] if i>1 else h[i-1])
            if h[i]>x:bull=True;x=ep;ep=h[i];af=step
            elif l[i]<ep:ep=l[i];af=min(af+step,maximum)
        sar[i]=x;d[i]=1 if bull else -1
    r["psar"]=sar;r["psar_direction"]=d;return r
