from .stema import STEMAStrategy
from .psarema import PSAREMAStrategy
from .dcemachop import DCEMACHOPStrategy
from .rsimom import RSIMomentumStrategy
from .emamacd import EMAMACDStrategy
from .bbtrend import BollingerTrendStrategy
from .dcbreakout import DonchianAsymmetricBreakoutStrategy
FUTURES_STRATEGY_CLASSES={"stema":STEMAStrategy,"psarema":PSAREMAStrategy,"dcemachop":DCEMACHOPStrategy,"rsimom":RSIMomentumStrategy,"emamacd":EMAMACDStrategy,"bbtrend":BollingerTrendStrategy,"dcbreakout":DonchianAsymmetricBreakoutStrategy}
def create_futures_strategy(uid,capital,db_path=None,timeframe="1d"):
    name=uid.split("__",1)[0].strip().lower()
    if name not in FUTURES_STRATEGY_CLASSES:raise ValueError(f"Unknown futures strategy: {name}")
    return FUTURES_STRATEGY_CLASSES[name](uid=uid,capital=capital,db_path=db_path,timeframe=timeframe,allow_fractional_shares=False)