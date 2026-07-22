from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class FuturesInstrument:
    symbol: str
    data_symbol: str
    root_symbol: str
    multiplier: float
    tick_size: float
    tick_value: float
    leverage_range: tuple[float, float]
    description: str

# leverage_range is informational only; actual margin changes over time.
#
# data_symbol notes:
#   - Root contracts (ES, NQ, YM, RTY, GC, SI, CL, NG, ZN, ZB) price off their
#     own daily continuous contract in market_data.duckdb (e.g. "ES=F").
#   - MES, MNQ, MGC, MCL, MNG, MHG, SIL have their own dedicated series in
#     duckdb/futures_data_1h.duckdb (e.g. "MES=F" -- the micro ticker
#     itself, not the root) and price off that directly.
#   - MYM and M2K have no dedicated micro data source yet, so they still
#     proxy off their root's daily series ("YM=F" / "RTY=F").
FUTURES_INSTRUMENTS = {
    "ES": FuturesInstrument("ES","ES=F","ES",50.0,0.25,12.50,(10.0,20.0),"E-mini S&P 500"),
    "MES": FuturesInstrument("MES","MES=F","ES",5.0,0.25,1.25,(10.0,20.0),"Micro E-mini S&P 500"),
    "NQ": FuturesInstrument("NQ","NQ=F","NQ",20.0,0.25,5.00,(10.0,18.0),"E-mini Nasdaq-100"),
    "MNQ": FuturesInstrument("MNQ","MNQ=F","NQ",2.0,0.25,0.50,(10.0,18.0),"Micro E-mini Nasdaq-100"),
    "YM": FuturesInstrument("YM","YM=F","YM",5.0,1.0,5.0,(10.0,20.0),"E-mini Dow"),
    "MYM": FuturesInstrument("MYM","YM=F","YM",0.5,1.0,0.5,(10.0,20.0),"Micro E-mini Dow"),
    "RTY": FuturesInstrument("RTY","RTY=F","RTY",50.0,0.10,5.0,(8.0,16.0),"E-mini Russell 2000"),
    "M2K": FuturesInstrument("M2K","RTY=F","RTY",5.0,0.10,0.5,(8.0,16.0),"Micro E-mini Russell 2000"),
    "GC": FuturesInstrument("GC","GC=F","GC",100.0,0.10,10.0,(7.0,12.0),"COMEX Gold"),
    "MGC": FuturesInstrument("MGC","MGC=F","GC",10.0,0.10,1.0,(7.0,12.0),"Micro Gold"),
    "SI": FuturesInstrument("SI","SI=F","SI",5000.0,0.005,25.0,(4.0,8.0),"COMEX Silver"),
    "SIL": FuturesInstrument("SIL","SIL=F","SI",1000.0,0.001,1.0,(4.0,8.0),"Micro Silver"),
    "CL": FuturesInstrument("CL","CL=F","CL",1000.0,0.01,10.0,(7.0,15.0),"WTI Crude Oil"),
    "MCL": FuturesInstrument("MCL","MCL=F","CL",100.0,0.01,1.0,(7.0,15.0),"Micro WTI Crude Oil"),
    "NG": FuturesInstrument("NG","NG=F","NG",10000.0,0.001,10.0,(5.0,12.0),"Henry Hub Natural Gas"),
    "MNG": FuturesInstrument("MNG","MNG=F","NG",1000.0,0.001,1.0,(5.0,12.0),"Micro Henry Hub Natural Gas"),
    "ZN": FuturesInstrument("ZN","ZN=F","ZN",1000.0,1/64,15.625,(20.0,40.0),"10-Year Treasury Note"),
    "ZB": FuturesInstrument("ZB","ZB=F","ZB",1000.0,1/32,31.25,(15.0,35.0),"Treasury Bond"),
    # NOTE: multiplier/tick figures below follow the standard-size/10 pattern
    # used by the other micro contracts here, but I could not fully verify
    # them against a current CME contract spec sheet -- double check before
    # trading MHG live.
    "MHG": FuturesInstrument("MHG","MHG=F","HG",2500.0,0.0005,1.25,(7.0,14.0),"Micro Copper"),
}

def normalize_futures_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()

def get_futures_instrument(symbol: str) -> FuturesInstrument:
    symbol = normalize_futures_symbol(symbol)
    try: return FUTURES_INSTRUMENTS[symbol]
    except KeyError as exc:
        raise ValueError(f"Unsupported futures contract '{symbol}'. Supported: {', '.join(sorted(FUTURES_INSTRUMENTS))}") from exc

def supported_futures_symbols() -> list[str]:
    return sorted(FUTURES_INSTRUMENTS)
