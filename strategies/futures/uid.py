from __future__ import annotations
from typing import Any
from .instruments import get_futures_instrument
SUPPORTED_FUTURES_STRATEGIES={"stema","psarema","dcemachop"}

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
    else:
        p.update(donchian_period=int(_get(raw,"dc","donchian_period")),ema_period=int(_get(raw,"ema","ema_period")),adx_period=int(raw.get("adx_period","14")),adx_threshold=float(_get(raw,"adx","adx_threshold")),chop_period=int(raw.get("chop_period","14")),chop_threshold=float(_get(raw,"chop","chop_threshold")))
    return p
