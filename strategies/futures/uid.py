from __future__ import annotations
from typing import Any
from .instruments import get_futures_instrument
SUPPORTED_FUTURES_STRATEGIES={"stema","psarema","dcemachop", "rsimom", "emamacd", "bbtrend", "dcbreakout"}

def parse_uid_parts(uid: str):
    parts=[p.strip() for p in uid.split("__") if p.strip()]
    if not parts: raise ValueError("UID must be non-empty")
    strategy=parts[0].lower()
    if strategy not in SUPPORTED_FUTURES_STRATEGIES: raise ValueError(f"Unsupported futures strategy: {strategy}")
    raw={}
    for token in parts[1:]:
        if "=" not in token: raise ValueError(f"Invalid UID token: {token}")
        k,v=token.split("=",1); k=k.strip().lower(); v=v.strip()
        if not k or not v or k in raw: raise ValueError(f"Invalid or duplicate UID token: {token}")
        raw[k]=v
    return strategy,raw

def _get(raw,*names):
    for n in names:
        if n in raw:return raw[n]
    raise ValueError(f"UID must contain one of: {', '.join(names)}")

def parse_futures_uid(uid: str)->dict[str,Any]:
    strategy,raw=parse_uid_parts(uid)
    symbol=_get(raw,"s","symbol").upper(); inst=get_futures_instrument(symbol)
    p={"strategy_type":strategy,"symbol":symbol,"data_symbol":inst.data_symbol,"root_symbol":inst.root_symbol,"multiplier":inst.multiplier,"atr_period":int(raw.get("atr",raw.get("atr_period","14"))),"stop_atr_multiple":float(raw.get("sl_atr",raw.get("sl","2")))}
    if p["atr_period"]<1 or p["stop_atr_multiple"]<=0: raise ValueError("ATR period and stop multiple must be positive")
    if strategy=="stema":
        p.update(supertrend_period=int(_get(raw,"st_period","stp")),supertrend_multiplier=float(_get(raw,"st_mult","stm")),ema_period=int(_get(raw,"ema","ema_period")))
    elif strategy=="psarema":
        p.update(psar_step=float(_get(raw,"psar_step","step")),psar_max=float(_get(raw,"psar_max","max")),ema_period=int(_get(raw,"ema","ema_period")))
    elif strategy=="dcemachop":
        p.update(donchian_period=int(_get(raw,"dc","donchian_period")),ema_period=int(_get(raw,"ema","ema_period")),adx_period=int(raw.get("adx_period","14")),adx_threshold=float(_get(raw,"adx","adx_threshold")),chop_period=int(raw.get("chop_period","14")),chop_threshold=float(_get(raw,"chop","chop_threshold")))
    elif strategy=="rsimom":
        p.update(rsi_period=int(raw.get("rsi_period","9")),rsi_ema_period=int(raw.get("rsi_ema","9")),trend_ema_period=int(raw.get("trend_ema","200")),exit_fast_period=int(raw.get("exit_fast","20")),exit_slow_period=int(raw.get("exit_slow","50")),target_timeframe=raw.get("tf","4h"))
    elif strategy=="emamacd":
        p.update(trend_ema_period=int(raw.get("trend_ema","200")),target_timeframe=raw.get("tf","4h"))
    elif strategy=="bbtrend":
        p.update(bb_period=int(raw.get("bb_period","20")),bb_std=float(raw.get("bb_std","1")),trend_ema_period=int(raw.get("trend_ema","200")),volume_ma_period=int(raw.get("vol_ma","20")),target_timeframe=raw.get("tf","8h"))
    elif strategy=="dcbreakout":
        p.update(upper_period=int(raw.get("upper","50")),lower_period=int(raw.get("lower","40")),target_timeframe=raw.get("tf","1w"))
    return p