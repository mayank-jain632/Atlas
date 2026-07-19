# Atlas futures strategies

Single-symbol, one-contract futures framework.

Data symbols come from the existing futures universe: `ES=F`, `NQ=F`, `YM=F`, `RTY=F`, `GC=F`, `SI=F`, `CL=F`, `NG=F`, `ZN=F`, `ZB=F`. Tradable aliases such as MES, MNQ, MGC, MCL, MYM, M2K and SIL map to those roots with their own multipliers.

Example UIDs:

```python
UID = "stema__s=MES__st_period=10__st_mult=3__ema=200__atr=14__sl_atr=2"
UID = "psarema__s=MNQ__psar_step=0.02__psar_max=0.20__ema=200__atr=14__sl_atr=2"
UID = "dcemachop__s=MGC__dc=20__ema=200__adx_period=14__adx=25__chop_period=14__chop=35__atr=14__sl_atr=2"
```

The framework currently assumes one contract, close-to-close mark-to-market, close fills, a fixed ATR stop from entry ATR, and no commissions, slippage, roll costs or margin enforcement.
