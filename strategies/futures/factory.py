from .stema import STEMAStrategy
from .psarema import PSAREMAStrategy
from .dcemachop import DCEMACHOPStrategy
FUTURES_STRATEGY_CLASSES={"stema":STEMAStrategy,"psarema":PSAREMAStrategy,"dcemachop":DCEMACHOPStrategy}
def create_futures_strategy(uid,capital,db_path=None,timeframe="1d",source_timeframe=None):
    name=uid.split("__",1)[0].strip().lower()
    if name not in FUTURES_STRATEGY_CLASSES:raise ValueError(f"Unknown futures strategy: {name}")
    return FUTURES_STRATEGY_CLASSES[name](uid=uid,capital=capital,db_path=db_path,timeframe=timeframe,source_timeframe=source_timeframe,allow_fractional_shares=False)
